#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V11.2: Safe Early-Target Inductive UDA Budget Sweep (HiCAT-lite, non-LLM)
------------------------------------------------------------------------
Why V11/V11.1 may look "worse with more target unlabeled":
  - Unsupervised domain alignment (CDAN/CORAL/MCC/prior/consistency) can cause NEGATIVE TRANSFER
    when conditional shift exists. As target coverage grows, the alignment signal strengthens and can
    move the decision boundary away from the good source-only solution.
  - Threshold/prior estimation from tiny unlabeled budgets is unstable.

This script implements a *conservative* inductive setting:
  - Target TEST is NEVER used in training, vocab, DA stats, pseudo selection, or threshold.
  - Budgets are NESTED per seed (10 ⊂ 20 ⊂ 50 ⊂ ...), so the curve is interpretable.
  - "Do-no-harm" training: checkpoint is selected by SOURCE-VAL by default (keeps source-only baseline).
  - FixMatch-style *masked* self-training on target (only high-confidence teacher predictions),
    optionally with prototype agreement filter.
  - Optional HiCAT components that use SOURCE ONLY:
      * source pseudo-domain clustering (TF-IDF+KMeans)
      * source pseudo-domain adversarial (GRL)
      * hierarchical prototype regularization (source only)
      * prototype-anchor class-conditional alignment (target KL to teacher)

Defaults are intentionally conservative to avoid the collapse you observed at large budgets.
If you want aggressive adaptation, increase unsup weights and lower confidence thresholds.

Run:
  python train_v11_2_safe_early_target.py
  python train_v11_2_safe_early_target.py --target_budgets "0,20,50,100,200,500,1000,2000,4000,-1"
  python train_v11_2_safe_early_target.py --seeds "42,43,44,45,46" --save_predictions

Notes:
  - Source label convention: 0=true, 1=false
  - COVID labels used only for evaluation: 1=true, 0=false
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise RuntimeError(
        "Failed to import PyTorch. Please install a working PyTorch build (CPU-only is fine).\n"
        f"Original error: {e}"
    )

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # type: ignore


# ----------------------------- Repro -----------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- Tokenization ----------------------------------


URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")

# Char vocab: ASCII 32..126 -> ids 1..95, pad=0, unk=96
CHAR_PAD = 0
CHAR_UNK = 96
CHAR_MIN = 32
CHAR_MAX = 126


def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    text = str(text).lower()
    text = URL_RE.sub(" URLTOKEN ", text)
    tokens: List[str] = []
    for tok in TOKEN_RE.findall(text):
        if tok == "URLTOKEN":
            tokens.append("<url>")
        elif tok.isdigit():
            tokens.append("<num>")
        else:
            tokens.append(tok)
    return tokens


def char_encode(text: str, max_len: int) -> List[int]:
    if text is None:
        text = ""
    s = str(text).lower()
    s = URL_RE.sub(" URLTOKEN ", s)
    ids: List[int] = []
    for ch in s[:max_len]:
        oc = ord(ch)
        if CHAR_MIN <= oc <= CHAR_MAX:
            ids.append(oc - CHAR_MIN + 1)
        else:
            ids.append(CHAR_UNK)
    if len(ids) == 0:
        ids = [CHAR_UNK]
    return ids


# ----------------------------- Vocabulary ------------------------------------


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in tokens]


def build_vocab(texts: Sequence[str], min_freq: int = 2, max_size: int = 50000) -> Vocab:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    specials = ["<pad>", "<unk>", "<url>", "<num>"]
    stoi: Dict[str, int] = {s: i for i, s in enumerate(specials)}
    itos: List[str] = list(specials)

    for tok, freq in counter.most_common():
        if tok in stoi:
            continue
        if freq < min_freq:
            break
        if len(itos) >= max_size:
            break
        stoi[tok] = len(itos)
        itos.append(tok)

    return Vocab(stoi=stoi, itos=itos, pad_id=stoi["<pad>"], unk_id=stoi["<unk>"])


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(seqs)
    lengths = [min(len(s), max_len) for s in seqs]
    L = max(1, min(max(lengths), max_len))

    input_ids = torch.full((batch_size, L), pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, L), dtype=torch.float32)

    for i, s in enumerate(seqs):
        s = list(s)[:L]
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1.0

    return input_ids, mask


# ----------------------------- Augmentations ---------------------------------


def random_token_dropout(ids: List[int], p: float, unk_id: int, rng: random.Random) -> List[int]:
    if p <= 0:
        return ids
    return [unk_id if rng.random() < p else tid for tid in ids]


def random_deletion(ids: List[int], p: float, rng: random.Random) -> List[int]:
    if p <= 0 or len(ids) <= 2:
        return ids
    kept = [tid for tid in ids if rng.random() > p]
    return kept if len(kept) >= 2 else ids


def random_swap(ids: List[int], n_swaps: int, rng: random.Random) -> List[int]:
    if n_swaps <= 0 or len(ids) <= 2:
        return ids
    out = ids[:]
    L = len(out)
    for _ in range(n_swaps):
        i, j = rng.randrange(L), rng.randrange(L)
        out[i], out[j] = out[j], out[i]
    return out


def weak_augment_word(ids: List[int], unk_id: int, rng: random.Random) -> List[int]:
    out = random_token_dropout(ids, p=0.05, unk_id=unk_id, rng=rng)
    out = random_swap(out, n_swaps=1, rng=rng)
    return out


def strong_augment_word(ids: List[int], unk_id: int, rng: random.Random) -> List[int]:
    out = random_deletion(ids, p=0.10, rng=rng)
    out = random_token_dropout(out, p=0.15, unk_id=unk_id, rng=rng)
    out = random_swap(out, n_swaps=3, rng=rng)
    return out


def weak_augment_char(ids: List[int], rng: random.Random) -> List[int]:
    out = [CHAR_UNK if rng.random() < 0.02 else x for x in ids]
    out = random_swap(out, n_swaps=1, rng=rng)
    return out


def strong_augment_char(ids: List[int], rng: random.Random) -> List[int]:
    out = random_deletion(ids, p=0.05, rng=rng)
    out = [CHAR_UNK if rng.random() < 0.08 else x for x in out]
    out = random_swap(out, n_swaps=3, rng=rng)
    return out


# ----------------------------- Source pseudo-domains (topic clustering) -------


def fit_source_pseudo_domains_tfidf_kmeans(
    texts: Sequence[str],
    n_clusters: int,
    seed: int,
    max_features: int = 60000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
):
    if n_clusters <= 1:
        return None, None, np.zeros(len(texts), dtype=np.int64)

    vec = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    X = vec.fit_transform(list(texts))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=2048,
        n_init=10,
    )
    dom = km.fit_predict(X).astype(np.int64)
    return vec, km, dom


def predict_source_pseudo_domains(vec: TfidfVectorizer, km: MiniBatchKMeans, texts: Sequence[str]) -> np.ndarray:
    X = vec.transform(list(texts))
    return km.predict(X).astype(np.int64)


# ----------------------------- Datasets --------------------------------------


class SourceLabeledDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Vocab,
        max_len: int,
        max_char_len: int,
        domains: Optional[Sequence[int]] = None,
    ):
        assert len(texts) == len(labels)
        if domains is None:
            domains = [0] * len(texts)
        assert len(domains) == len(texts)

        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len

        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        self.labels: List[int] = []
        self.domains: List[int] = []

        for t, y, d in zip(texts, labels, domains):
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])

            c = char_encode(t, max_len=max_char_len)
            self.char.append(c[:max_char_len])

            self.labels.append(int(y))
            self.domains.append(int(d))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.word[idx], self.char[idx], self.labels[idx], self.domains[idx]


def collate_source(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    word, char, y, d = zip(*batch)
    w_ids, w_mask = pad_batch(word, pad_id=pad_word, max_len=max_len)
    c_ids, c_mask = pad_batch(char, pad_id=pad_char, max_len=max_char_len)
    y_t = torch.tensor(y, dtype=torch.long)
    d_t = torch.tensor(d, dtype=torch.long)
    return w_ids, w_mask, c_ids, c_mask, y_t, d_t


class TargetUnlabeledDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int, max_char_len: int, seed: int):
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len
        self.rng = random.Random(seed)

        self.texts: List[str] = [str(t) for t in texts]
        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        for t in self.texts:
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])
            self.char.append(char_encode(t, max_len=max_char_len)[:max_char_len])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        w = self.word[idx]
        c = self.char[idx]
        w_w = weak_augment_word(w, unk_id=self.vocab.unk_id, rng=self.rng)
        w_s = strong_augment_word(w, unk_id=self.vocab.unk_id, rng=self.rng)
        c_w = weak_augment_char(c, rng=self.rng)
        c_s = strong_augment_char(c, rng=self.rng)
        return w_w, c_w, w_s, c_s, idx


def collate_target(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    w_w, c_w, w_s, c_s, idxs = zip(*batch)
    w_w_ids, w_w_mask = pad_batch(w_w, pad_id=pad_word, max_len=max_len)
    c_w_ids, c_w_mask = pad_batch(c_w, pad_id=pad_char, max_len=max_char_len)
    w_s_ids, w_s_mask = pad_batch(w_s, pad_id=pad_word, max_len=max_len)
    c_s_ids, c_s_mask = pad_batch(c_s, pad_id=pad_char, max_len=max_char_len)
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    return w_w_ids, w_w_mask, c_w_ids, c_w_mask, w_s_ids, w_s_mask, c_s_ids, c_s_mask, idxs_t


class InferenceDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int, max_char_len: int):
        self.texts = [str(t) for t in texts]
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len
        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        for t in self.texts:
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])
            self.char.append(char_encode(t, max_len=max_char_len)[:max_char_len])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.word[idx], self.char[idx], idx


def collate_infer(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    w, c, idxs = zip(*batch)
    w_ids, w_mask = pad_batch(w, pad_id=pad_word, max_len=max_len)
    c_ids, c_mask = pad_batch(c, pad_id=pad_char, max_len=max_char_len)
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    return w_ids, w_mask, c_ids, c_mask, idxs_t


# ----------------------------- Model -----------------------------------------


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grl(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradReverse.apply(x, lambda_)


class AttnPool(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.proj(self.dropout(h)).squeeze(-1)  # (B,L)
        scores = scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        return pooled


class HybridTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int,
        word_hidden: int,
        word_pad: int,
        char_vocab_size: int = 97,
        char_emb_dim: int = 32,
        char_channels: int = 96,
        char_kernels: Tuple[int, ...] = (3, 4, 5),
        char_pad: int = 0,
        char_proj_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=word_pad)
        self.word_gru = nn.GRU(word_emb_dim, word_hidden, batch_first=True, bidirectional=True)
        self.word_attn = AttnPool(word_hidden * 2, dropout=dropout)

        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=char_pad)
        self.char_convs = nn.ModuleList([nn.Conv1d(char_emb_dim, char_channels, kernel_size=k) for k in char_kernels])
        self.char_proj = nn.Linear(char_channels * len(char_kernels), char_proj_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_dim = word_hidden * 2 + char_proj_dim

    def forward(self, word_ids: torch.Tensor, word_mask: torch.Tensor, char_ids: torch.Tensor, char_mask: torch.Tensor) -> torch.Tensor:
        w = self.dropout(self.word_emb(word_ids))
        h, _ = self.word_gru(w)
        h = self.dropout(h)
        w_feat = self.word_attn(h, word_mask)

        c = self.dropout(self.char_emb(char_ids))  # (B,LC,Cemb)
        c = c.transpose(1, 2)  # (B,Cemb,LC)
        conv_outs: List[torch.Tensor] = []
        for conv in self.char_convs:
            x = torch.relu(conv(c))
            x = torch.max(x, dim=-1).values
            conv_outs.append(x)
        c_feat = torch.cat(conv_outs, dim=1)
        c_feat = self.dropout(c_feat)
        c_feat = torch.relu(self.char_proj(c_feat))

        feat = torch.cat([w_feat, c_feat], dim=1)
        feat = self.dropout(feat)
        return feat


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: HybridTextEncoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, w_ids: torch.Tensor, w_mask: torch.Tensor, c_ids: torch.Tensor, c_mask: torch.Tensor):
        feats = self.encoder(w_ids, w_mask, c_ids, c_mask)
        logits = self.classifier(feats)
        return feats, logits


class SrcDomainDiscriminator(nn.Module):
    def __init__(self, feat_dim: int, num_domains: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class StudentUDA(nn.Module):
    def __init__(self, backbone: EncoderClassifier, num_src_domains: int = 1, srcdom_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.src_dom_disc = SrcDomainDiscriminator(
            feat_dim=backbone.encoder.out_dim,
            num_domains=max(1, int(num_src_domains)),
            hidden=srcdom_hidden,
            dropout=dropout,
        )

    def forward(self, w_ids: torch.Tensor, w_mask: torch.Tensor, c_ids: torch.Tensor, c_mask: torch.Tensor):
        return self.backbone(w_ids, w_mask, c_ids, c_mask)


@torch.no_grad()
def ema_update(teacher: EncoderClassifier, student: EncoderClassifier, decay: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)


# ----------------------------- Loss utilities --------------------------------


def dann_grl_lambda(progress: float) -> float:
    p = float(np.clip(progress, 0.0, 1.0))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def entropy_torch(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)


def proto_pull_loss(feats: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    if feats.size(0) == 0:
        return torch.tensor(0.0, device=feats.device)
    z = F.normalize(feats, dim=1)
    p = prototypes[labels]
    sim = (z * p).sum(dim=1)
    return (1.0 - sim).mean()


def proto_kl_align_loss(feats: torch.Tensor, soft_targets: torch.Tensor, prototypes: torch.Tensor, temp: float = 0.1, eps: float = 1e-12) -> torch.Tensor:
    if feats.size(0) == 0:
        return torch.tensor(0.0, device=feats.device)
    z = F.normalize(feats, dim=1)
    logits = (z @ prototypes.t()) / max(1e-6, float(temp))
    p_proto = torch.softmax(logits, dim=1).clamp_min(eps)
    return F.kl_div(torch.log(p_proto), soft_targets.detach(), reduction="batchmean")


def hierarchical_proto_loss(feats: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    if feats.size(0) < 2:
        return torch.tensor(0.0, device=feats.device)
    z = F.normalize(feats, dim=1)
    loss = torch.tensor(0.0, device=feats.device)
    n_terms = 0
    for c in range(num_classes):
        m_c = (labels == c)
        if int(m_c.sum().item()) < 2:
            continue
        p_global = F.normalize(z[m_c].mean(dim=0), dim=0)
        for d in domains[m_c].unique():
            m_dc = m_c & (domains == d)
            if int(m_dc.sum().item()) < 1:
                continue
            p_dom = F.normalize(z[m_dc].mean(dim=0), dim=0)
            loss = loss + (1.0 - (p_dom * p_global).sum())
            n_terms += 1
    if n_terms == 0:
        return torch.tensor(0.0, device=feats.device)
    return loss / float(n_terms)


# ----------------------------- DA / Prior estimation (optional) ---------------


def distribution_align_np(probs: np.ndarray, p_ema: np.ndarray, pi_target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = probs / (p_ema[None, :] + eps) * pi_target[None, :]
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    return p.astype(np.float32)


def distribution_align_torch(probs: torch.Tensor, p_ema: torch.Tensor, pi_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = probs / (p_ema.unsqueeze(0) + eps) * pi_target.unsqueeze(0)
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    return p


def estimate_target_prior_em(p_source_post: np.ndarray, pi_source: np.ndarray, max_iter: int = 80, tol: float = 1e-6, eps: float = 1e-12) -> np.ndarray:
    pi_s = np.asarray(pi_source, dtype=np.float64)
    pi_s = pi_s / (pi_s.sum() + eps)
    q = pi_s.copy()
    for _ in range(max_iter):
        ratio = (q + eps) / (pi_s + eps)
        p_adj = p_source_post * ratio[None, :]
        p_adj = p_adj / (p_adj.sum(axis=1, keepdims=True) + eps)
        q_new = p_adj.mean(axis=0)
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q = q_new
    q = q / (q.sum() + eps)
    return q.astype(np.float32)


def clip_and_normalize_prior(pi: np.ndarray, min_prob: float) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float32).copy()
    pi = np.clip(pi, min_prob, 1.0 - min_prob)
    pi = pi / (pi.sum() + 1e-12)
    return pi


def clamp_prior_2class(pi_new: np.ndarray, pi_old: np.ndarray, max_delta: float, prior_min: float) -> np.ndarray:
    if max_delta <= 0:
        return clip_and_normalize_prior(pi_new, min_prob=prior_min)
    p0_old = float(pi_old[0])
    p0_new = float(pi_new[0])
    p0 = p0_old + float(np.clip(p0_new - p0_old, -max_delta, max_delta))
    p0 = float(np.clip(p0, prior_min, 1.0 - prior_min))
    return np.array([p0, 1.0 - p0], dtype=np.float32)


def entropy_norm_np(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    C = probs.shape[1]
    ent = -(probs * np.log(probs + eps)).sum(axis=1)
    return ent / math.log(C)


# ----------------------------- Prototypes ------------------------------------


@torch.no_grad()
def compute_source_prototypes(model: EncoderClassifier, loader: DataLoader, device: torch.device, num_classes: int = 2) -> torch.Tensor:
    model.eval()
    feat_dim = model.encoder.out_dim
    sums = torch.zeros((num_classes, feat_dim), device=device)
    counts = torch.zeros((num_classes,), device=device)
    for w_ids, w_mask, c_ids, c_mask, y, _d in loader:
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        y = y.to(device)
        feats, _ = model(w_ids, w_mask, c_ids, c_mask)
        for k in range(num_classes):
            m = (y == k)
            if bool(m.any()):
                sums[k] += feats[m].sum(dim=0)
                counts[k] += float(m.sum().item())
    protos = sums / counts.unsqueeze(1).clamp(min=1.0)
    protos = F.normalize(protos, dim=1)
    return protos


# ----------------------------- Calibration -----------------------------------


@torch.no_grad()
def collect_logits_labels(model: EncoderClassifier, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, y_list = [], []
    for w_ids, w_mask, c_ids, c_mask, y, _d in loader:
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        y = y.to(device)
        _f, logits = model(w_ids, w_mask, c_ids, c_mask)
        logits_list.append(logits.detach())
        y_list.append(y.detach())
    return torch.cat(logits_list, dim=0), torch.cat(y_list, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, device: torch.device, t_min: float, t_max: float) -> float:
    logits = logits.detach()
    labels = labels.detach()
    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_t).clamp(min=t_min, max=t_max)
        loss = ce(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_t).detach().cpu().item())
    return float(np.clip(T, t_min, t_max))


@torch.no_grad()
def eval_source_val_acc(model: EncoderClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for w_ids, w_mask, c_ids, c_mask, y, _d in loader:
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        y = y.to(device)
        _f, logits = model(w_ids, w_mask, c_ids, c_mask)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct / max(1, total))


# ----------------------------- CSV readers -----------------------------------


def _coerce_label_to_int01(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        s = s.replace({"true": "0", "false": "1", "real": "0", "fake": "1"})
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def read_pubhealth_csv(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    need_cols = {"claim", "main_text", "label"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")

    y_num = _coerce_label_to_int01(df["label"])
    valid = y_num.isin([0, 1])
    if not bool(valid.all()):
        invalid_vals = df.loc[~valid, "label"].value_counts(dropna=False).head(10)
        print(f"[Warn] PubHealth labels not in {{0,1}}. Will drop {int((~valid).sum())} rows. Examples:\n{invalid_vals}")

    df = df.loc[valid].copy()
    labels = y_num.loc[valid].astype(int).tolist()
    texts = (df["claim"].fillna("").astype(str) + " " + df["main_text"].fillna("").astype(str)).tolist()
    return texts, labels


def read_covid_csv_text_and_optional_binary_label(path: str) -> Tuple[List[str], Optional[List[int]], Optional[str]]:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "Text" not in df.columns:
        raise ValueError(f"{path} missing column 'Text'. Found: {list(df.columns)}")
    texts = df["Text"].fillna("").astype(str).tolist()

    col_lower_map = {str(c).strip().lower(): c for c in df.columns}
    candidates = ["binary label", "binary_label", "binarylabel", "label"]
    label_col = None
    for cand in candidates:
        if cand in col_lower_map:
            label_col = col_lower_map[cand]
            break
    if label_col is None:
        return texts, None, None

    s = df[label_col]
    if s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        ss = ss.replace({"true": "1", "false": "0", "real": "1", "fake": "0"})
        y_num = pd.to_numeric(ss, errors="coerce")
    else:
        y_num = pd.to_numeric(s, errors="coerce")

    if not bool(y_num.isin([0, 1]).all()):
        return texts, None, None

    return texts, y_num.astype(int).tolist(), str(label_col)


# ----------------------------- Inductive split (unique text) ------------------


def split_target_by_unique_text(texts: Sequence[str], labels: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    assert len(texts) == len(labels)
    df = pd.DataFrame({"Text": list(texts), "y": labels.astype(int)})

    grp = df.groupby("Text")["y"].agg(lambda s: int(round(s.mean()))).reset_index()
    y_u = grp["y"].values.astype(int)
    idx_u = np.arange(len(grp))

    rng = np.random.RandomState(seed)
    idx0 = idx_u[y_u == 0]
    idx1 = idx_u[y_u == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(round(len(idx0) * test_ratio))
    n1_test = int(round(len(idx1) * test_ratio))

    test_u = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_u = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(test_u)
    rng.shuffle(train_u)

    test_texts = set(grp.loc[test_u, "Text"].tolist())
    train_texts = set(grp.loc[train_u, "Text"].tolist())

    train_idx = df.index[df["Text"].isin(train_texts)].values
    test_idx = df.index[df["Text"].isin(test_texts)].values
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


# ----------------------------- Prediction / decisions -------------------------


@torch.no_grad()
def predict_probs(
    model: EncoderClassifier,
    texts: Sequence[str],
    vocab: Vocab,
    device: torch.device,
    max_len: int,
    max_char_len: int,
    batch_size: int,
    calib_T: float,
    teacher_temp: float,
) -> np.ndarray:
    ds = InferenceDataset(texts, vocab=vocab, max_len=max_len, max_char_len=max_char_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_infer, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=max_len, max_char_len=max_char_len),
        drop_last=False,
    )
    model.eval()
    probs = np.zeros((len(ds), 2), dtype=np.float32)
    T_total = max(1e-6, calib_T * teacher_temp)
    for w_ids, w_mask, c_ids, c_mask, idxs in loader:
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        _f, logits = model(w_ids, w_mask, c_ids, c_mask)
        p = torch.softmax(logits / T_total, dim=-1).cpu().numpy().astype(np.float32)
        probs[idxs.numpy()] = p
    return probs


def pred_argmax_covid(probs_source_space: np.ndarray) -> np.ndarray:
    pred_source = probs_source_space.argmax(axis=1).astype(int)  # 0=true,1=false
    return np.where(pred_source == 0, 1, 0).astype(int)


def compute_matchprior_thr_train(probs_train_source_space: np.ndarray, pi_target_source_space: np.ndarray, thr_smooth: float) -> float:
    score_true = probs_train_source_space[:, 0].astype(np.float32)
    pi_true = float(np.clip(float(pi_target_source_space[0]), 0.01, 0.99))
    q = float(np.quantile(score_true, 1.0 - pi_true))
    thr = thr_smooth * q + (1.0 - thr_smooth) * 0.5
    return float(np.clip(thr, 0.05, 0.95))


def pred_matchprior_covid(probs_source_space: np.ndarray, thr_true: float) -> np.ndarray:
    score_true = probs_source_space[:, 0].astype(np.float32)
    return (score_true >= thr_true).astype(int)  # covid label: 1=true,0=false


# ----------------------------- Training (single budget run) -------------------


def train_one(
    seed: int,
    vocab: Vocab,
    src_train_texts: Sequence[str],
    src_train_y: Sequence[int],
    src_val_texts: Sequence[str],
    src_val_y: Sequence[int],
    tgt_train_texts_unique_budget: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[EncoderClassifier, np.ndarray, np.ndarray, float]:
    """
    Train with a target unlabeled budget, strict inductive.
    Returns: teacher_model, pi_target(source space), p_ema, calib_T
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # --- source pseudo domains (SOURCE ONLY) ---
    if args.use_src_pseudo_domain and args.src_n_clusters > 1:
        cluster_seed = int(args.src_cluster_seed) if args.src_cluster_seed >= 0 else seed
        vec, km, src_train_dom = fit_source_pseudo_domains_tfidf_kmeans(
            src_train_texts,
            n_clusters=args.src_n_clusters,
            seed=cluster_seed,
            max_features=args.src_tfidf_max_features,
            min_df=args.src_tfidf_min_df,
        )
        src_val_dom = predict_source_pseudo_domains(vec, km, src_val_texts) if len(src_val_texts) > 0 else np.zeros(len(src_val_texts), dtype=np.int64)
        num_src_domains = int(args.src_n_clusters)
    else:
        src_train_dom = np.zeros(len(src_train_texts), dtype=np.int64)
        src_val_dom = np.zeros(len(src_val_texts), dtype=np.int64)
        num_src_domains = 1

    # loaders
    src_train_ds = SourceLabeledDataset(src_train_texts, src_train_y, vocab, args.max_len, args.max_char_len, domains=src_train_dom)
    src_train_loader = DataLoader(
        src_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_source, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
        drop_last=False,
    )
    src_val_ds = SourceLabeledDataset(src_val_texts, src_val_y, vocab, args.max_len, args.max_char_len, domains=src_val_dom)
    src_val_loader = DataLoader(
        src_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_source, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
        drop_last=False,
    )

    # model
    encoder = HybridTextEncoder(
        vocab_size=len(vocab.itos),
        word_emb_dim=args.emb_dim,
        word_hidden=args.hidden_size,
        word_pad=vocab.pad_id,
        char_vocab_size=97,
        char_emb_dim=args.char_emb_dim,
        char_channels=args.char_channels,
        char_kernels=tuple(args.char_kernels),
        char_pad=CHAR_PAD,
        char_proj_dim=args.char_proj_dim,
        dropout=args.dropout,
    )
    student_backbone = EncoderClassifier(encoder, num_classes=2)
    student = StudentUDA(student_backbone, num_src_domains=num_src_domains, srcdom_hidden=args.srcdom_hidden, dropout=args.dropout).to(device)

    teacher = copy.deepcopy(student_backbone).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # priors (source)
    y_np = np.asarray(src_train_y, dtype=np.int64)
    pi_source = np.array([np.mean(y_np == 0), np.mean(y_np == 1)], dtype=np.float32)
    pi_source = pi_source / (pi_source.sum() + 1e-12)

    # target prior init (source space)
    if args.pi_base == "source":
        pi_target = pi_source.copy()
    else:
        pi_target = np.array([0.5, 0.5], dtype=np.float32)
    pi_target = clip_and_normalize_prior(pi_target, min_prob=args.prior_min)

    p_ema = np.array([0.5, 0.5], dtype=np.float32)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------- pretrain (source only) ----------------
    best_state = None
    best_val = -1.0
    for ep in range(1, args.pretrain_epochs + 1):
        student.train()
        for step, (w_ids, w_mask, c_ids, c_mask, y, sd) in enumerate(src_train_loader):
            w_ids, w_mask = w_ids.to(device), w_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            y = y.to(device)
            sd = sd.to(device)

            feats, logits = student(w_ids, w_mask, c_ids, c_mask)
            loss = F.cross_entropy(logits, y)

            # source pseudo-domain adversarial (optional)
            if args.use_srcdom_adv and num_src_domains > 1 and args.lambda_srcdom_pre > 0:
                progress_pre = ((ep - 1) * len(src_train_loader) + step) / max(1, args.pretrain_epochs * len(src_train_loader))
                grl_l = dann_grl_lambda(progress_pre)
                dom_logits = student.src_dom_disc(grl(feats, grl_l))
                loss = loss + args.lambda_srcdom_pre * F.cross_entropy(dom_logits, sd)

            # hierarchical prototype regularization on source batch (optional)
            if args.use_hproto and num_src_domains > 1 and args.gamma_hproto_pre > 0:
                loss = loss + args.gamma_hproto_pre * hierarchical_proto_loss(feats, y, sd, num_classes=2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            ema_update(teacher, student.backbone, decay=args.ema_decay)

        val_acc = eval_source_val_acc(teacher, src_val_loader, device=device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(teacher.state_dict())

    if best_state is not None:
        teacher.load_state_dict(best_state)
        student.backbone.load_state_dict(best_state)

    # optional calibration on source-val
    calib_T = 1.0
    if args.use_calibration:
        logits_val, y_val = collect_logits_labels(teacher, src_val_loader, device=device)
        calib_T = fit_temperature(logits_val, y_val, device=device, t_min=args.temp_min, t_max=args.temp_max)

    # Save anchor teacher (for do-no-harm / distillation if enabled)
    anchor_teacher = copy.deepcopy(teacher).to(device)
    for p in anchor_teacher.parameters():
        p.requires_grad = False

    # ---------------- UDA (conservative FixMatch style) ----------------
    target_budget = len(tgt_train_texts_unique_budget)
    if target_budget < args.min_budget_for_uda or args.uda_epochs <= 0:
        # no UDA update; return source-only teacher
        return teacher, pi_target, p_ema, calib_T

    tgt_ds = TargetUnlabeledDataset(tgt_train_texts_unique_budget, vocab, args.max_len, args.max_char_len, seed=seed + 7)
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_target, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
        drop_last=False,
    )

    # Fix steps per epoch to SOURCE length to make budgets comparable
    steps_per_epoch = int(len(src_train_loader)) if args.steps_per_epoch <= 0 else int(args.steps_per_epoch)
    total_steps = args.uda_epochs * max(1, steps_per_epoch)
    global_step = 0

    # checkpoint selection: default by source val only
    best_ckpt = copy.deepcopy(teacher.state_dict())
    best_val_acc = eval_source_val_acc(teacher, src_val_loader, device=device)

    for uda_ep in range(1, args.uda_epochs + 1):
        # budget-aware scaling (sqrt) for stability
        if args.budget_aware:
            scale = min(1.0, math.sqrt(float(target_budget) / max(1.0, float(args.unsup_budget_ref))))
        else:
            scale = 1.0

        # (optional) update p_ema, pi_target from current teacher over ALL target-train budget
        if args.update_pi_target or args.use_da:
            # one pass over target to update p_ema / pi_target
            teacher.eval()
            probs_all = []
            with torch.no_grad():
                eval_loader = DataLoader(
                    tgt_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=partial(collate_target, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
                    drop_last=False,
                )
                T_total = max(1e-6, calib_T * args.teacher_temp)
                for w_w_ids, w_w_mask, c_w_ids, c_w_mask, _w_s_ids, _w_s_mask, _c_s_ids, _c_s_mask, _idxs in eval_loader:
                    w_w_ids, w_w_mask = w_w_ids.to(device), w_w_mask.to(device)
                    c_w_ids, c_w_mask = c_w_ids.to(device), c_w_mask.to(device)
                    _f, logits = teacher(w_w_ids, w_w_mask, c_w_ids, c_w_mask)
                    p = torch.softmax(logits / T_total, dim=-1).detach().cpu().numpy().astype(np.float32)
                    probs_all.append(p)
            probs_all = np.concatenate(probs_all, axis=0)
            p_mean = probs_all.mean(axis=0).astype(np.float32)
            p_mean = p_mean / (p_mean.sum() + 1e-12)
            p_ema = args.da_momentum * p_ema + (1.0 - args.da_momentum) * p_mean
            p_ema = p_ema / (p_ema.sum() + 1e-12)

            if args.update_pi_target:
                # reliable subset
                conf = probs_all.max(axis=1)
                ent = entropy_norm_np(probs_all)
                mask = (conf >= args.prior_conf) & (ent <= args.prior_ent)
                if int(mask.sum()) < int(min(args.prior_min_n, max(20, probs_all.shape[0]))):
                    thr = float(np.quantile(conf, 0.90))
                    mask = conf >= thr
                pi_hat = estimate_target_prior_em(probs_all[mask], pi_source, max_iter=args.prior_em_iter)
                pi_hat = clip_and_normalize_prior(pi_hat, min_prob=args.prior_min)
                pi_new = args.prior_momentum * pi_target + (1.0 - args.prior_momentum) * pi_hat
                pi_target = clamp_prior_2class(pi_new, pi_target, args.prior_max_delta, args.prior_min)

        # prototypes from current teacher (source only)
        prototypes = compute_source_prototypes(teacher, src_train_loader, device=device, num_classes=2) if (args.use_proto_align or args.use_proto_filter or args.gamma_protosrc > 0) else None

        # training epoch
        student.train()
        teacher.eval()
        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_loader)

        for _ in range(steps_per_epoch):
            progress = global_step / max(1, total_steps)
            grl_l = dann_grl_lambda(progress)

            try:
                sw_ids, sw_mask, sc_ids, sc_mask, sy, sd = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train_loader)
                sw_ids, sw_mask, sc_ids, sc_mask, sy, sd = next(src_iter)

            try:
                tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask, tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask, tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask, _ = next(tgt_iter)

            sw_ids, sw_mask = sw_ids.to(device), sw_mask.to(device)
            sc_ids, sc_mask = sc_ids.to(device), sc_mask.to(device)
            sy = sy.to(device)
            sd = sd.to(device)

            tw_w_ids, tw_w_mask = tw_w_ids.to(device), tw_w_mask.to(device)
            tc_w_ids, tc_w_mask = tc_w_ids.to(device), tc_w_mask.to(device)
            tw_s_ids, tw_s_mask = tw_s_ids.to(device), tw_s_mask.to(device)
            tc_s_ids, tc_s_mask = tc_s_ids.to(device), tc_s_mask.to(device)

            # -------- source supervised --------
            src_feats, src_logits = student(sw_ids, sw_mask, sc_ids, sc_mask)
            loss_sup = F.cross_entropy(src_logits, sy)

            # source pseudo-domain adversarial (optional, scaled)
            loss_srcdom = torch.tensor(0.0, device=device)
            if args.use_srcdom_adv and num_src_domains > 1 and args.lambda_srcdom > 0:
                dom_logits = student.src_dom_disc(grl(src_feats, grl_l))
                loss_srcdom = F.cross_entropy(dom_logits, sd)

            # hierarchical proto loss on source batch (optional, scaled)
            loss_hproto = torch.tensor(0.0, device=device)
            if args.use_hproto and num_src_domains > 1 and args.gamma_hproto > 0:
                loss_hproto = hierarchical_proto_loss(src_feats, sy, sd, num_classes=2)

            # source prototype pull (optional)
            loss_protosrc = torch.tensor(0.0, device=device)
            if prototypes is not None and args.gamma_protosrc > 0:
                loss_protosrc = proto_pull_loss(src_feats, sy, prototypes)

            # -------- target FixMatch-style masked self-training --------
            with torch.no_grad():
                _tf, t_logits = teacher(tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask)
                T_total = max(1e-6, calib_T * args.teacher_temp)
                t_probs = torch.softmax(t_logits / T_total, dim=-1)

                if args.use_da:
                    pi_t = torch.tensor(pi_target, dtype=torch.float32, device=device)
                    p_ema_t = torch.tensor(p_ema, dtype=torch.float32, device=device)
                    t_probs = distribution_align_torch(t_probs, p_ema_t, pi_t)

                conf, t_pred = t_probs.max(dim=1)
                ent = entropy_torch(t_probs) / math.log(2.0)  # normalized for 2-class

                mask = (conf >= args.fm_tau) & (ent <= args.fm_ent)

                # optional prototype agreement filter
                if args.use_proto_filter and prototypes is not None:
                    # compute proto pred/margin using teacher features (more stable)
                    t_feats, _ = teacher(tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask)
                    z = F.normalize(t_feats, dim=1)
                    sim = z @ prototypes.t()
                    proto_pred = sim.argmax(dim=1)
                    top1 = sim.max(dim=1).values
                    top2 = sim.min(dim=1).values
                    margin = top1 - top2
                    mask = mask & (proto_pred == t_pred) & (margin >= args.proto_margin)

            # student strong
            tgt_feats, tgt_logits_s = student(tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask)
            logp_s = torch.log_softmax(tgt_logits_s, dim=-1)

            # masked KL to teacher probs (FixMatch soft)
            loss_u = torch.tensor(0.0, device=device)
            if bool(mask.any()):
                # weight by confidence (optional)
                w = conf[mask].detach()
                kl = F.kl_div(logp_s[mask], t_probs[mask].detach(), reduction="none").sum(dim=1)
                loss_u = (kl * w).mean()

            # prototype target alignment (optional, masked)
            loss_prototgt = torch.tensor(0.0, device=device)
            if args.use_proto_align and prototypes is not None and bool(mask.any()):
                loss_prototgt = proto_kl_align_loss(tgt_feats[mask], t_probs[mask], prototypes, temp=args.proto_align_temp)

            # anchor distillation on source (optional) to avoid drift
            loss_anchor = torch.tensor(0.0, device=device)
            if args.use_anchor_distill and args.gamma_anchor > 0:
                with torch.no_grad():
                    _af, a_logits = anchor_teacher(sw_ids, sw_mask, sc_ids, sc_mask)
                    a_prob = torch.softmax(a_logits / max(1e-6, args.anchor_temp), dim=-1)
                s_logp = torch.log_softmax(src_logits / max(1e-6, args.anchor_temp), dim=-1)
                loss_anchor = F.kl_div(s_logp, a_prob, reduction="batchmean")

            # total loss
            loss = (
                loss_sup
                + scale * (args.lambda_srcdom * loss_srcdom + args.gamma_hproto * loss_hproto)
                + args.gamma_protosrc * loss_protosrc
                + scale * (args.lambda_u * loss_u + args.gamma_prototgt * loss_prototgt)
                + args.gamma_anchor * loss_anchor
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            ema_update(teacher, student.backbone, decay=args.ema_decay)
            global_step += 1

        # end epoch: checkpoint selection by source val (default)
        val_acc = eval_source_val_acc(teacher, src_val_loader, device=device)
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_ckpt = copy.deepcopy(teacher.state_dict())

        # optional early stop if source val drops too much (do-no-harm)
        if args.early_stop_drop > 0 and val_acc < (best_val_acc - args.early_stop_drop):
            break

    teacher.load_state_dict(best_ckpt)
    return teacher, pi_target, p_ema, calib_T


# ----------------------------- Helpers ----------------------------------------


def parse_int_list(s: str) -> List[int]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def parse_budget_list(s: str) -> List[int]:
    b = parse_int_list(s)
    if len(b) == 0:
        return [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, -1]
    return b


def mean_std(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ----------------------------- Main -------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent

    # Data paths
    parser.add_argument("--pubhealth_train", type=str, default=str(base_dir / "./pubhealth/pubhealth_train_clean.csv"))
    parser.add_argument("--pubhealth_val", type=str, default=str(base_dir / "./pubhealth/pubhealth_validation_clean.csv"))
    parser.add_argument("--covid_true", type=str, default=str(base_dir / "../covid/trueNews.csv"))
    parser.add_argument("--covid_fake", type=str, default=str(base_dir / "../covid/fakeNews.csv"))

    # Inductive split
    parser.add_argument("--target_test_ratio", type=float, default=0.2)
    parser.add_argument("--target_split_seed", type=int, default=42)

    # Budgets & seeds
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--target_budgets", type=str, default="0,10,20,50,100,200,500,1000,2000,4000,-1")
    parser.add_argument("--delta_to_full", type=float, default=0.01, help="reach full_mean - delta")

    # Output
    parser.add_argument("--out_dir", type=str, default=str(base_dir / "v11_2_runs"))
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--results_csv", type=str, default=str(base_dir / "v11_2_budget_results.csv"))
    parser.add_argument("--summary_csv", type=str, default=str(base_dir / "v11_2_budget_summary.csv"))

    # Vocab mode
    parser.add_argument("--vocab_mode", type=str, default="source_only", choices=["source_only", "source_plus_target"])
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)

    # Preprocess
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--max_char_len", type=int, default=512)

    # Model
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Char-CNN
    parser.add_argument("--char_emb_dim", type=int, default=32)
    parser.add_argument("--char_channels", type=int, default=96)
    parser.add_argument("--char_kernels", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--char_proj_dim", type=int, default=128)

    # Train
    parser.add_argument("--pretrain_epochs", type=int, default=6)
    parser.add_argument("--uda_epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="<=0: use len(source_loader). Fixing this improves budget comparability.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # FixMatch masked unsupervised
    parser.add_argument("--lambda_u", type=float, default=0.5, help="Weight for masked FixMatch KL on target.")
    parser.add_argument("--fm_tau", type=float, default=0.97, help="Teacher confidence threshold for target update.")
    parser.add_argument("--fm_ent", type=float, default=0.85, help="Teacher entropy threshold for target update (normalized).")

    # Prototype anchor (class-conditional)
    parser.add_argument("--use_proto_align", action="store_true", help="Enable prototype anchor alignment.")
    parser.add_argument("--gamma_protosrc", type=float, default=0.02)
    parser.add_argument("--gamma_prototgt", type=float, default=0.02)
    parser.add_argument("--proto_align_temp", type=float, default=0.1)

    # Prototype filter
    parser.add_argument("--use_proto_filter", action="store_true", help="Enable prototype agreement filter on target updates.")
    parser.add_argument("--proto_margin", type=float, default=0.05)

    # Source pseudo-domains + invariance
    parser.add_argument("--use_src_pseudo_domain", action="store_true", help="Enable SOURCE TF-IDF+KMeans pseudo-domains.")
    parser.add_argument("--src_n_clusters", type=int, default=12)
    parser.add_argument("--src_cluster_seed", type=int, default=123, help="Seed for source clustering. -1 ties to run seed.")
    parser.add_argument("--src_tfidf_max_features", type=int, default=60000)
    parser.add_argument("--src_tfidf_min_df", type=int, default=2)

    parser.add_argument("--use_srcdom_adv", action="store_true", help="Enable source pseudo-domain adversarial (GRL).")
    parser.add_argument("--srcdom_hidden", type=int, default=256)
    parser.add_argument("--lambda_srcdom_pre", type=float, default=0.05)
    parser.add_argument("--lambda_srcdom", type=float, default=0.05)

    parser.add_argument("--use_hproto", action="store_true", help="Enable hierarchical proto regularization on source.")
    parser.add_argument("--gamma_hproto_pre", type=float, default=0.02)
    parser.add_argument("--gamma_hproto", type=float, default=0.02)

    # DA / prior (optional)
    parser.add_argument("--use_da", action="store_true", help="Enable distribution alignment on teacher probs.")
    parser.add_argument("--da_momentum", type=float, default=0.99)
    parser.add_argument("--update_pi_target", action="store_true", help="Update target class prior pi_t via EM (unlabeled).")
    parser.add_argument("--prior_em_iter", type=int, default=80)
    parser.add_argument("--prior_warmup_epochs", type=int, default=2)  # not used directly but kept for compatibility
    parser.add_argument("--prior_conf", type=float, default=0.92)
    parser.add_argument("--prior_ent", type=float, default=0.75)
    parser.add_argument("--prior_min_n", type=int, default=1200)
    parser.add_argument("--prior_momentum", type=float, default=0.9)
    parser.add_argument("--prior_min", type=float, default=0.05)
    parser.add_argument("--prior_max_delta", type=float, default=0.06)
    parser.add_argument("--pi_base", type=str, default="uniform", choices=["uniform", "source"])

    # Temperatures / calibration
    parser.add_argument("--teacher_temp", type=float, default=1.2)
    parser.add_argument("--use_calibration", action="store_true", help="Calibrate teacher on source-val.")
    parser.add_argument("--temp_min", type=float, default=0.5)
    parser.add_argument("--temp_max", type=float, default=10.0)

    # Do-no-harm / stability
    parser.add_argument("--budget_aware", action="store_true", help="Scale target unsup losses by sqrt(budget/ref).")
    parser.add_argument("--unsup_budget_ref", type=int, default=500)
    parser.add_argument("--min_budget_for_uda", type=int, default=200, help="Below this, skip UDA updates (avoid harming baseline).")
    parser.add_argument("--early_stop_drop", type=float, default=0.02, help="Stop if source-val drops by this much from best.")

    # Anchor distillation (optional)
    parser.add_argument("--use_anchor_distill", action="store_true", help="Distill to source-only anchor teacher to reduce drift.")
    parser.add_argument("--gamma_anchor", type=float, default=0.02)
    parser.add_argument("--anchor_temp", type=float, default=1.0)

    # system
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    budgets = parse_budget_list(args.target_budgets)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[V11.2] device={device}")
    print(f"[V11.2] seeds={seeds}")
    print(f"[V11.2] budgets={budgets} (-1 means full)")
    print(f"[V11.2] split_seed={args.target_split_seed} test_ratio={args.target_test_ratio}")
    print(f"[V11.2] vocab_mode={args.vocab_mode} | budget_aware={args.budget_aware} | min_budget_for_uda={args.min_budget_for_uda}")

    # ---------- Load source ----------
    src_train_texts, src_train_y = read_pubhealth_csv(args.pubhealth_train)
    src_val_texts, src_val_y = read_pubhealth_csv(args.pubhealth_val)
    print(f"[Data] Source train={len(src_train_texts)} | val={len(src_val_texts)}")

    # ---------- Load target ----------
    true_texts, true_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_true)
    fake_texts, fake_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_fake)

    all_texts = true_texts + fake_texts
    y_file = np.array([1] * len(true_texts) + [0] * len(fake_texts), dtype=np.int64)

    if true_labels_opt is not None and fake_labels_opt is not None:
        y_col = np.array(true_labels_opt + fake_labels_opt, dtype=np.int64)
        agree = float((y_col == y_file).mean())
        agree_flip = float(((1 - y_col) == y_file).mean())
        if agree_flip > agree:
            y_col = 1 - y_col
        y = y_col
        print("[Data] Target label source: column (Binary Label / Label)")
    else:
        y = y_file
        print("[Data] Target label source: file membership")
    print(f"[Data] Target rows={len(all_texts)} | dist={pd.Series(y).value_counts().to_dict()}")

    # ---------- Inductive split by unique text ----------
    train_idx, test_idx = split_target_by_unique_text(all_texts, y, test_ratio=args.target_test_ratio, seed=args.target_split_seed)
    tgt_train_unique_full = list(dict.fromkeys([all_texts[i] for i in train_idx]))
    tgt_test_texts = [all_texts[i] for i in test_idx]
    y_test = y[test_idx]
    print(f"[Split] target-train unique={len(tgt_train_unique_full)} | target-test rows={len(test_idx)}")

    # ---------- Build base vocab (source-only) ----------
    vocab_source = build_vocab(src_train_texts + src_val_texts, min_freq=args.min_freq, max_size=args.vocab_size)

    # ---------- Sweep ----------
    records: List[Dict[str, object]] = []
    budget_to_seed_acc: Dict[int, List[float]] = {}

    full_budget = len(tgt_train_unique_full)

    for budget in budgets:
        budget_key = int(budget)
        budget_to_seed_acc[budget_key] = []

        for seed in seeds:
            set_seed(seed)
            rng = np.random.RandomState(seed)

            # nested sampling: one permutation per seed
            perm = rng.permutation(len(tgt_train_unique_full))
            if budget_key < 0:
                used = len(tgt_train_unique_full)
            else:
                used = int(min(budget_key, len(tgt_train_unique_full)))
            idx_use = perm[:used].tolist()
            tgt_budget_texts = [tgt_train_unique_full[i] for i in idx_use]

            # build vocab per budget if requested (still inductive; only uses target-train budget)
            if args.vocab_mode == "source_plus_target":
                vocab_run = build_vocab(src_train_texts + src_val_texts + tgt_budget_texts, min_freq=args.min_freq, max_size=args.vocab_size)
            else:
                vocab_run = vocab_source

            teacher, pi_target, p_ema, calib_T = train_one(
                seed=seed,
                vocab=vocab_run,
                src_train_texts=src_train_texts,
                src_train_y=src_train_y,
                src_val_texts=src_val_texts,
                src_val_y=src_val_y,
                tgt_train_texts_unique_budget=tgt_budget_texts,
                args=args,
            )

            # threshold from target-train budget (inductive)
            if len(tgt_budget_texts) > 0:
                probs_train = predict_probs(teacher, tgt_budget_texts, vocab_run, device, args.max_len, args.max_char_len, args.batch_size, calib_T, args.teacher_temp)
                probs_train_adj = distribution_align_np(probs_train, p_ema, pi_target) if args.use_da else probs_train

                if args.budget_aware:
                    thr_smooth = min(1.0, math.sqrt(float(len(tgt_budget_texts)) / max(1.0, float(args.unsup_budget_ref))))
                else:
                    thr_smooth = 1.0
                thr = compute_matchprior_thr_train(probs_train_adj, pi_target, thr_smooth=thr_smooth)
            else:
                thr = 0.5

            # predict on target-test
            probs_test = predict_probs(teacher, tgt_test_texts, vocab_run, device, args.max_len, args.max_char_len, args.batch_size, calib_T, args.teacher_temp)
            probs_test_adj = distribution_align_np(probs_test, p_ema, pi_target) if args.use_da else probs_test

            pred_a = pred_argmax_covid(probs_test_adj)
            pred_m = pred_matchprior_covid(probs_test_adj, thr_true=thr)

            acc_argmax = float(accuracy_score(y_test, pred_a))
            acc_match = float(accuracy_score(y_test, pred_m))
            budget_to_seed_acc[budget_key].append(acc_match)

            records.append(
                {
                    "budget_used": used,
                    "budget_raw": budget_key,
                    "seed": seed,
                    "train_unique_used": len(tgt_budget_texts),
                    "test_rows": int(len(test_idx)),
                    "acc_argmax": acc_argmax,
                    "acc_matchprior_inductive": acc_match,
                    "pi_target_true(source0)": float(pi_target[0]),
                    "pi_target_false(source1)": float(pi_target[1]),
                    "calib_T": float(calib_T),
                    "thr_true_from_train": float(thr),
                }
            )

            if args.save_predictions:
                out_csv = out_dir / f"v11_2_pred_budget{used}_seed{seed}.csv"
                pd.DataFrame(
                    {
                        "Text": tgt_test_texts,
                        "y_true": y_test,
                        "prob_true_source0": probs_test_adj[:, 0],
                        "prob_false_source1": probs_test_adj[:, 1],
                        "pred_argmax": pred_a,
                        "pred_matchprior": pred_m,
                        "thr": thr,
                    }
                ).to_csv(out_csv, index=False, encoding="utf-8-sig")

            print(f"[B={len(tgt_budget_texts):>5}] seed={seed} acc_match={acc_match:.4f} acc_argmax={acc_argmax:.4f} thr={thr:.3f} pi_t={pi_target}")

    # ---------- Save detailed ----------
    df_res = pd.DataFrame(records)
    df_res.to_csv(args.results_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved detailed results: {args.results_csv}")

    # ---------- Summary ----------
    summary_rows: List[Dict[str, object]] = []
    full_key = -1 if -1 in budgets else max([b for b in budgets if b >= 0] + [-1])
    full_accs = budget_to_seed_acc.get(full_key, [])
    full_mean, full_std = mean_std(full_accs)
    target_mean = full_mean - float(args.delta_to_full)

    for budget in budgets:
        key = int(budget)
        accs = budget_to_seed_acc.get(key, [])
        m, s = mean_std(accs)
        used = full_budget if key < 0 else int(key)
        summary_rows.append(
            {
                "budget_used": used,
                "budget_raw": key,
                "mean_acc_matchprior_inductive": m,
                "std_acc_matchprior_inductive": s,
                "full_mean": full_mean,
                "target_mean(full-delta)": target_mean,
            }
        )

    summary_sorted = sorted(summary_rows, key=lambda d: int(d["budget_used"]))
    best_budget_found = None
    for row in summary_sorted:
        if float(row["mean_acc_matchprior_inductive"]) >= target_mean:
            best_budget_found = int(row["budget_used"])
            break

    df_sum = pd.DataFrame(summary_sorted)
    df_sum.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved summary: {args.summary_csv}")

    print("\n========== V11.2 Safe Early-Target Summary ==========")
    print(f"Full budget (unique texts) = {full_budget}")
    print(f"Full mean±std (matchprior) = {full_mean:.4f} ± {full_std:.4f}")
    print(f"Target threshold (full - delta) = {target_mean:.4f}  (delta={args.delta_to_full})")
    if best_budget_found is None:
        print("[Result] No budget in your sweep reached (full_mean - delta). Try adding larger budgets or increase delta.")
    else:
        print(f"[Result] Minimum target-train unique Texts to reach current effect ≈ {best_budget_found}")


if __name__ == "__main__":
    main()
