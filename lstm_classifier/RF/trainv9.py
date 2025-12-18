#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DUPL / CAPA V9.1 (non-LLM UDA for Fake News Cross-Domain) - Robust Prior Edition
-------------------------------------------------------------------------------
Source domain (labeled): PubHealth (non-COVID medical fake news)
  - input text = claim + main_text
  - label: 0=true, 1=false

Target domain (unlabeled for training): COVID dataset (social/news)
  - input text = Text
  - labels (if exist) used ONLY for evaluation

V9.1 fixes the main failure mode observed in V9:
  - EM prior estimation can drift badly under domain shift (especially with over-sharpened temperature).
  - Wrong pi_target will corrupt distribution alignment and matchprior decision rule.

V9.1 upgrades:
  1) Robust prior estimation:
     - Warmup epochs (no prior update early)
     - Use only "reliable" target samples to estimate prior (conf + low entropy + weak/strong agree)
     - Minimum sample count fallback
     - Clamp per-epoch prior change (avoid runaway)
     - Optional mix with p_ema (stabilize)
  2) Safer default temperature bounds: temp_min=0.5 (avoid extreme sharpen)
  3) Keep all V9 components:
     Hybrid encoder (Word BiGRU+Attn + CharCNN),
     CDAN + entropy conditioning,
     CORAL,
     DA,
     pseudo labeling with agreement + prototype + clustering diversity,
     MCC,
     SupCon

Usage (defaults read from script directory):
  python train_v9_1.py --print_pseudo_stats

Recommended for paper (non-transductive eval):
  python train_v9_1.py --target_test_ratio 0.2 --target_split_seed 42 --print_pseudo_stats
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import PyTorch. Please install a working PyTorch build (CPU-only is fine).\n"
        f"Original error: {e}"
    )

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
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
    """
    Simple tokenizer (English-ish):
      - lowercase
      - replace URLs with <url>
      - numbers -> <num>
    """
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
    """
    ASCII char encoding with URL masking to stabilize.
    """
    if text is None:
        text = ""
    s = str(text).lower()
    s = URL_RE.sub(" URLTOKEN ", s)
    ids: List[int] = []
    for ch in s[:max_len]:
        oc = ord(ch)
        if CHAR_MIN <= oc <= CHAR_MAX:
            ids.append(oc - CHAR_MIN + 1)  # 1..95
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


def build_vocab(
    texts: Sequence[str],
    min_freq: int = 2,
    max_size: int = 50000,
) -> Vocab:
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


# ----------------------------- Datasets --------------------------------------


class SourceLabeledDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocab, max_len: int, max_char_len: int):
        assert len(texts) == len(labels)
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len

        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        self.labels: List[int] = []

        for t, y in zip(texts, labels):
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])

            c = char_encode(t, max_len=max_char_len)
            self.char.append(c[:max_char_len])

            self.labels.append(int(y))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.word[idx], self.char[idx], self.labels[idx]


def collate_source(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    word, char, y = zip(*batch)
    w_ids, w_mask = pad_batch(word, pad_id=pad_word, max_len=max_len)
    c_ids, c_mask = pad_batch(char, pad_id=pad_char, max_len=max_char_len)
    y_t = torch.tensor(y, dtype=torch.long)
    return w_ids, w_mask, c_ids, c_mask, y_t


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

            c = char_encode(t, max_len=max_char_len)
            self.char.append(c[:max_char_len])

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


class PseudoLabeledDataset(Dataset):
    def __init__(
        self,
        base_word: Sequence[Sequence[int]],
        base_char: Sequence[Sequence[int]],
        indices: Sequence[int],
        soft_labels: np.ndarray,
        weights: np.ndarray,
        vocab: Vocab,
        max_len: int,
        max_char_len: int,
        seed: int,
    ):
        self.base_word = base_word
        self.base_char = base_char
        self.indices = np.asarray(indices, dtype=np.int64)
        self.soft = np.asarray(soft_labels, dtype=np.float32)
        self.w = np.asarray(weights, dtype=np.float32)
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len
        self.rng = random.Random(seed)

        assert len(self.indices) == len(self.soft) == len(self.w)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        w = list(self.base_word[idx])[: self.max_len]
        c = list(self.base_char[idx])[: self.max_char_len]
        w_s = strong_augment_word(w, unk_id=self.vocab.unk_id, rng=self.rng)
        c_s = strong_augment_char(c, rng=self.rng)
        return w_s, c_s, self.soft[i], self.w[i]


def collate_pseudo(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    w, c, soft, ww = zip(*batch)
    w_ids, w_mask = pad_batch(w, pad_id=pad_word, max_len=max_len)
    c_ids, c_mask = pad_batch(c, pad_id=pad_char, max_len=max_char_len)
    soft_t = torch.tensor(np.stack(soft), dtype=torch.float32)
    w_t = torch.tensor(ww, dtype=torch.float32)
    return w_ids, w_mask, c_ids, c_mask, soft_t, w_t


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
        self.char_convs = nn.ModuleList(
            [nn.Conv1d(char_emb_dim, char_channels, kernel_size=k, bias=True) for k in char_kernels]
        )
        self.char_proj = nn.Linear(char_channels * len(char_kernels), char_proj_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_dim = word_hidden * 2 + char_proj_dim

    def forward(
        self,
        word_ids: torch.Tensor,
        word_mask: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Word branch
        w = self.dropout(self.word_emb(word_ids))  # (B,L,E)
        h, _ = self.word_gru(w)  # (B,L,2H)
        h = self.dropout(h)
        w_feat = self.word_attn(h, word_mask)  # (B,2H)

        # Char branch
        c = self.dropout(self.char_emb(char_ids))  # (B,LC,Cemb)
        c = c.transpose(1, 2)  # (B,Cemb,LC)
        conv_outs: List[torch.Tensor] = []
        for conv in self.char_convs:
            x = conv(c)  # (B,Ch,LC-k+1)
            x = torch.relu(x)
            x = torch.max(x, dim=-1).values  # (B,Ch)
            conv_outs.append(x)
        c_feat = torch.cat(conv_outs, dim=1)  # (B,Ch*#k)
        c_feat = self.dropout(c_feat)
        c_feat = torch.relu(self.char_proj(c_feat))  # (B,char_proj)

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


class CDANDomainDiscriminator(nn.Module):
    """
    CDAN-style domain discriminator on outer product of (probabilities, features).
    Input dim = feat_dim * num_classes.
    """
    def __init__(self, feat_dim: int, num_classes: int = 2, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        in_dim = feat_dim * num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cond_feat: torch.Tensor) -> torch.Tensor:
        return self.net(cond_feat).squeeze(-1)


class StudentUDA(nn.Module):
    def __init__(self, backbone: EncoderClassifier):
        super().__init__()
        self.backbone = backbone
        self.cdan_disc = CDANDomainDiscriminator(feat_dim=backbone.encoder.out_dim, num_classes=2)

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


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if feats.size(0) <= 2:
        return torch.tensor(0.0, device=feats.device)

    z = F.normalize(feats, dim=1)
    sim = (z @ z.t()) / max(1e-6, temperature)
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9

    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.t()).float()
    mask_pos = mask_pos - torch.eye(mask_pos.size(0), device=mask_pos.device)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1) + 1e-12
    num = (exp_sim * mask_pos).sum(dim=1)

    valid = (mask_pos.sum(dim=1) > 0).float()
    loss = -torch.log((num / denom).clamp_min(1e-12)) * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def prior_reg_loss(student_probs: torch.Tensor, pi_target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = student_probs.mean(dim=0).clamp_min(eps)
    m = m / m.sum()
    return F.kl_div(torch.log(m), pi_target, reduction="sum")


def mcc_loss(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = probs / (probs.sum(dim=0, keepdim=True) + eps)
    C = (p.t() @ probs) / float(probs.size(0))
    off = C - torch.diag(torch.diag(C))
    return off.sum()


def coral_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if src.size(0) < 2 or tgt.size(0) < 2:
        return torch.tensor(0.0, device=src.device)

    src_c = src - src.mean(dim=0, keepdim=True)
    tgt_c = tgt - tgt.mean(dim=0, keepdim=True)
    cov_src = (src_c.t() @ src_c) / (src.size(0) - 1)
    cov_tgt = (tgt_c.t() @ tgt_c) / (tgt.size(0) - 1)
    return ((cov_src - cov_tgt) ** 2).mean()


def entropy_torch(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)


# ----------------------------- DA / Prior estimation --------------------------


def distribution_align_np(probs: np.ndarray, p_ema: np.ndarray, pi_target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = probs / (p_ema[None, :] + eps) * pi_target[None, :]
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    return p.astype(np.float32)


def distribution_align_torch(probs: torch.Tensor, p_ema: torch.Tensor, pi_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = probs / (p_ema.unsqueeze(0) + eps) * pi_target.unsqueeze(0)
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    return p


def estimate_target_prior_em(
    p_source_post: np.ndarray,  # (N,C)
    pi_source: np.ndarray,      # (C,)
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Saerens et al. EM prior estimation under label shift:
      p_t(y|x) ∝ p_s(y|x) * (pi_t(y)/pi_s(y))
      pi_t = E_x[p_t(y|x)]
    """
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
    """
    Clamp the change of class-0 prior per epoch to avoid EM runaway.
    Works for 2-class case.
    """
    if max_delta <= 0:
        return clip_and_normalize_prior(pi_new, min_prob=prior_min)

    p0_old = float(pi_old[0])
    p0_new = float(pi_new[0])
    p0 = p0_old + float(np.clip(p0_new - p0_old, -max_delta, max_delta))
    p0 = float(np.clip(p0, prior_min, 1.0 - prior_min))
    return np.array([p0, 1.0 - p0], dtype=np.float32)


# ----------------------------- Prototypes ------------------------------------


@torch.no_grad()
def compute_source_prototypes(
    model: EncoderClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> torch.Tensor:
    model.eval()
    feat_dim = model.encoder.out_dim
    sums = torch.zeros((num_classes, feat_dim), device=device)
    counts = torch.zeros((num_classes,), device=device)

    for w_ids, w_mask, c_ids, c_mask, y in loader:
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


# ----------------------------- Temperature scaling ----------------------------


@torch.no_grad()
def collect_logits_labels(
    model: EncoderClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, y_list = [], []
    for w_ids, w_mask, c_ids, c_mask, y in loader:
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        y = y.to(device)
        _f, logits = model(w_ids, w_mask, c_ids, c_mask)
        logits_list.append(logits.detach())
        y_list.append(y.detach())
    return torch.cat(logits_list, dim=0), torch.cat(y_list, dim=0)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    t_min: float = 0.5,
    t_max: float = 10.0,
) -> float:
    """
    Fit temperature T>0 on validation set by minimizing CE(logits/T, labels).
    """
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


# ----------------------------- Pseudo labeling V9.1 ---------------------------


def entropy_norm_np(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    C = probs.shape[1]
    ent = -(probs * np.log(probs + eps)).sum(axis=1)
    return ent / math.log(C)


@torch.no_grad()
def build_pseudo_pool_v9_1(
    teacher: EncoderClassifier,
    target_loader: DataLoader,
    device: torch.device,
    calib_T: float,
    teacher_temp: float,
    p_ema: np.ndarray,
    pi_target: np.ndarray,
    pi_source: np.ndarray,
    prototypes: Optional[torch.Tensor],
    uda_epoch: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      sel_indices, sel_soft, sel_w, new_p_ema, new_pi_target
    """
    teacher.eval()

    feats_all: List[np.ndarray] = []
    pw_all: List[np.ndarray] = []
    ps_all: List[np.ndarray] = []
    idx_all: List[np.ndarray] = []

    for w_w_ids, w_w_mask, c_w_ids, c_w_mask, w_s_ids, w_s_mask, c_s_ids, c_s_mask, idxs in tqdm(
        target_loader, desc="Teacher forward (target weak/strong)", leave=False
    ):
        w_w_ids, w_w_mask = w_w_ids.to(device), w_w_mask.to(device)
        c_w_ids, c_w_mask = c_w_ids.to(device), c_w_mask.to(device)
        w_s_ids, w_s_mask = w_s_ids.to(device), w_s_mask.to(device)
        c_s_ids, c_s_mask = c_s_ids.to(device), c_s_mask.to(device)

        feats_w, logits_w = teacher(w_w_ids, w_w_mask, c_w_ids, c_w_mask)
        _feats_s, logits_s = teacher(w_s_ids, w_s_mask, c_s_ids, c_s_mask)

        T_total = max(1e-6, calib_T * teacher_temp)
        probs_w = torch.softmax(logits_w / T_total, dim=-1)
        probs_s = torch.softmax(logits_s / T_total, dim=-1)

        feats_all.append(feats_w.detach().cpu().numpy())
        pw_all.append(probs_w.detach().cpu().numpy())
        ps_all.append(probs_s.detach().cpu().numpy())
        idx_all.append(idxs.numpy())

    feats = np.concatenate(feats_all, axis=0)
    probs_w = np.concatenate(pw_all, axis=0)
    probs_s = np.concatenate(ps_all, axis=0)
    idxs = np.concatenate(idx_all, axis=0)

    # order by idx
    order = np.argsort(idxs)
    feats, probs_w, probs_s, idxs = feats[order], probs_w[order], probs_s[order], idxs[order]

    # ---- update p_ema first (mean prediction) ----
    p_mean = probs_w.mean(axis=0).astype(np.float32)
    p_mean = p_mean / (p_mean.sum() + 1e-12)
    new_p_ema = args.da_momentum * p_ema + (1.0 - args.da_momentum) * p_mean
    new_p_ema = new_p_ema / (new_p_ema.sum() + 1e-12)

    # ---- robust prior estimation (V9.1) ----
    if args.prior_estimation == "em":
        if uda_epoch <= args.prior_warmup_epochs:
            pi_hat = pi_target
        else:
            conf0 = probs_w.max(axis=1)
            ent0 = entropy_norm_np(probs_w)
            agree0 = (probs_w.argmax(axis=1) == probs_s.argmax(axis=1))

            mask = (conf0 >= args.prior_conf) & (ent0 <= args.prior_ent) & agree0

            # fallback: if too few reliable, take top-confidence slice
            if int(mask.sum()) < int(args.prior_min_n):
                thr = float(np.quantile(conf0, 0.90))
                mask = (conf0 >= thr)

            pi_hat = estimate_target_prior_em(probs_w[mask], pi_source, max_iter=args.prior_em_iter)

        pi_hat = clip_and_normalize_prior(pi_hat, min_prob=args.prior_min)

        # momentum update
        new_pi = args.prior_momentum * pi_target + (1.0 - args.prior_momentum) * pi_hat

        # clamp per-epoch change
        new_pi = clamp_prior_2class(new_pi, pi_target, args.prior_max_delta, args.prior_min)

        # optional mix with p_ema to stabilize
        if args.prior_mix_ema > 0:
            new_pi = (1.0 - args.prior_mix_ema) * new_pi + args.prior_mix_ema * new_p_ema
            new_pi = clip_and_normalize_prior(new_pi, min_prob=args.prior_min)

    elif args.prior_estimation == "source":
        new_pi = clip_and_normalize_prior(pi_source, min_prob=args.prior_min)
    else:
        new_pi = np.array([0.5, 0.5], dtype=np.float32)

    # ---- distribution alignment ----
    if args.use_da:
        probs_w_adj = distribution_align_np(probs_w, new_p_ema, new_pi)
        probs_s_adj = distribution_align_np(probs_s, new_p_ema, new_pi)
    else:
        probs_w_adj = probs_w.astype(np.float32)
        probs_s_adj = probs_s.astype(np.float32)

    conf = probs_w_adj.max(axis=1)
    ent = entropy_norm_np(probs_w_adj)
    pred_w = probs_w_adj.argmax(axis=1)
    pred_s = probs_s_adj.argmax(axis=1)

    # agreement: argmax agree + KL
    eps = 1e-12
    kl_ws = (probs_w_adj * (np.log(probs_w_adj + eps) - np.log(probs_s_adj + eps))).sum(axis=1)
    agree = (pred_w == pred_s)

    # prototype constraints
    proto_ok = np.ones_like(conf, dtype=bool)
    proto_margin = np.zeros_like(conf, dtype=np.float32)
    if args.use_proto and prototypes is not None:
        ft = torch.tensor(feats, dtype=torch.float32, device=device)
        z = F.normalize(ft, dim=1)
        sim = (z @ prototypes.t()).detach().cpu().numpy()  # (N,2)
        proto_pred = sim.argmax(axis=1)
        top1 = sim.max(axis=1)
        top2 = np.min(sim, axis=1)  # for 2 classes
        proto_margin = (top1 - top2).astype(np.float32)
        proto_ok = (proto_pred == pred_w)

    # schedule K
    if uda_epoch <= args.pl_warmup_epochs:
        k_eff = 0
    else:
        t = min(1.0, (uda_epoch - args.pl_warmup_epochs) / max(1.0, float(args.k_ramp_epochs)))
        k_eff = int(max(args.k_min_per_class, round(args.k_per_class * t)))

    if k_eff <= 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            new_p_ema,
            new_pi,
        )

    # clustering for diversity (feature-based)
    N = feats.shape[0]
    k = int(min(max(2, args.n_clusters), N))
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=2048, n_init=10)
    cluster_ids = kmeans.fit_predict(feats)

    selected: set[int] = set()

    for y in (0, 1):
        need = k_eff
        conf_thr = float(args.tau_conf)
        ent_thr = float(args.tau_ent)
        kl_thr = float(args.tau_kl)
        margin_thr = float(args.tau_proto_margin)

        for _step in range(int(args.relax_steps) + 1):
            mask = (pred_w == y) & (conf >= conf_thr) & (ent <= ent_thr)

            if args.use_agree:
                mask = mask & agree & (kl_ws <= kl_thr)
            if args.use_proto:
                mask = mask & proto_ok & (proto_margin >= margin_thr)

            cand = np.where(mask)[0]
            if cand.size > 0:
                score = conf[cand] * (1.0 - ent[cand])
                if args.use_agree:
                    score = score * np.clip(1.0 - (kl_ws[cand] / (kl_thr + 1e-6)), 0.0, 1.0)
                if args.use_proto:
                    score = score * np.clip(
                        (proto_margin[cand] - margin_thr) / max(1e-6, args.proto_margin_scale),
                        0.0, 1.0
                    )

                # per-cluster selection
                for c in range(k):
                    if need <= 0:
                        break
                    cc = cand[cluster_ids[cand] == c]
                    if cc.size == 0:
                        continue
                    sc = score[cluster_ids[cand] == c]
                    order_cc = cc[np.argsort(-sc)]
                    for ii in order_cc[: int(args.top_m_per_cluster)]:
                        if need <= 0:
                            break
                        ii = int(ii)
                        if ii in selected:
                            continue
                        selected.add(ii)
                        need -= 1

                # global fill
                if need > 0:
                    order_all = cand[np.argsort(-score)]
                    for ii in order_all:
                        if need <= 0:
                            break
                        ii = int(ii)
                        if ii in selected:
                            continue
                        selected.add(ii)
                        need -= 1

            if need <= 0:
                break

            # relax thresholds
            conf_thr = max(float(args.conf_floor), conf_thr - float(args.conf_step))
            ent_thr = min(float(args.ent_ceiling), ent_thr + float(args.ent_step))
            kl_thr = min(float(args.kl_ceiling), kl_thr + float(args.kl_step))
            margin_thr = max(0.0, margin_thr - float(args.margin_step))

    if len(selected) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            new_p_ema,
            new_pi,
        )

    sel_local = np.array(sorted(selected), dtype=np.int64)
    sel_soft = probs_w_adj[sel_local].astype(np.float32)

    # weights
    w = (conf[sel_local] * (1.0 - ent[sel_local])).astype(np.float32)
    if args.use_proto:
        w = w * np.clip(proto_margin[sel_local] / max(1e-6, args.proto_margin_scale), 0.0, 1.0).astype(np.float32)
    if args.use_agree:
        w = w * np.clip(1.0 - (kl_ws[sel_local] / (float(args.kl_ceiling) + 1e-6)), 0.0, 1.0).astype(np.float32)

    # normalize weights
    w = w - w.min()
    if w.max() > 1e-6:
        w = 0.2 + 0.8 * (w / (w.max() + 1e-6))
    else:
        w = np.ones_like(w, dtype=np.float32)

    # class-balance reweight
    hard = sel_soft.argmax(axis=1)
    counts = np.bincount(hard, minlength=2).astype(np.float32) + 1e-6
    inv = counts.sum() / counts
    inv = inv / inv.mean()
    w = (w * inv[hard]).astype(np.float32)

    sel_indices = idxs[sel_local].astype(np.int64)

    if args.print_pseudo_stats:
        dist = np.bincount(hard, minlength=2).tolist()
        print(
            f"[Pseudo][E{uda_epoch}] k_eff={k_eff} selected={len(sel_indices)} dist(source0/1)={dist} "
            f"pi_t=[{new_pi[0]:.3f},{new_pi[1]:.3f}] p_ema=[{new_p_ema[0]:.3f},{new_p_ema[1]:.3f}]"
        )

    return sel_indices, sel_soft, w, new_p_ema, new_pi


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

    # try common label columns
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


def stratified_split_indices(y: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0_test = int(round(len(idx0) * test_ratio))
    n1_test = int(round(len(idx1) * test_ratio))
    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


# ----------------------------- Training V9.1 ---------------------------------


def train_v9_1(
    src_train_texts: Sequence[str],
    src_train_y: Sequence[int],
    src_val_texts: Optional[Sequence[str]],
    src_val_y: Optional[Sequence[int]],
    tgt_texts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple["EncoderClassifier", Vocab, np.ndarray, np.ndarray, float, np.ndarray]:
    # vocab
    vocab = build_vocab(list(src_train_texts) + list(tgt_texts), min_freq=args.min_freq, max_size=args.vocab_size)

    # loaders
    src_train_ds = SourceLabeledDataset(src_train_texts, src_train_y, vocab, args.max_len, args.max_char_len)
    src_train_loader = DataLoader(
        src_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_source,
            pad_word=vocab.pad_id,
            pad_char=CHAR_PAD,
            max_len=args.max_len,
            max_char_len=args.max_char_len,
        ),
        drop_last=False,
    )

    src_val_loader = None
    if src_val_texts is not None and src_val_y is not None and len(src_val_texts) > 0:
        src_val_ds = SourceLabeledDataset(src_val_texts, src_val_y, vocab, args.max_len, args.max_char_len)
        src_val_loader = DataLoader(
            src_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=partial(
                collate_source,
                pad_word=vocab.pad_id,
                pad_char=CHAR_PAD,
                max_len=args.max_len,
                max_char_len=args.max_char_len,
            ),
            drop_last=False,
        )

    tgt_ds = TargetUnlabeledDataset(tgt_texts, vocab, args.max_len, args.max_char_len, seed=args.seed + 7)
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_target,
            pad_word=vocab.pad_id,
            pad_char=CHAR_PAD,
            max_len=args.max_len,
            max_char_len=args.max_char_len,
        ),
        drop_last=False,
    )
    tgt_eval_loader = DataLoader(
        tgt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_target,
            pad_word=vocab.pad_id,
            pad_char=CHAR_PAD,
            max_len=args.max_len,
            max_char_len=args.max_char_len,
        ),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

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
    student = StudentUDA(student_backbone).to(device)

    teacher = copy.deepcopy(student_backbone).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # source prior
    labels_np = np.asarray(src_train_y, dtype=np.int64)
    pi_source = np.array([np.mean(labels_np == 0), np.mean(labels_np == 1)], dtype=np.float32)
    pi_source = pi_source / (pi_source.sum() + 1e-12)

    # init target prior
    if args.prior_estimation == "source":
        pi_target = pi_source.copy()
    elif args.prior_estimation == "uniform":
        pi_target = np.array([0.5, 0.5], dtype=np.float32)
    else:
        pi_target = np.array([0.5, 0.5], dtype=np.float32)
    pi_target = clip_and_normalize_prior(pi_target, min_prob=args.prior_min)

    # DA p_ema
    p_ema = np.array([0.5, 0.5], dtype=np.float32)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------- Pretrain -------------
    best_val = -1.0
    best_state = None

    for ep in range(1, args.pretrain_epochs + 1):
        student.train()
        pbar = tqdm(src_train_loader, desc=f"Pretrain {ep}/{args.pretrain_epochs}", leave=True)
        for w_ids, w_mask, c_ids, c_mask, y in pbar:
            w_ids, w_mask = w_ids.to(device), w_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            y = y.to(device)

            feats, logits = student(w_ids, w_mask, c_ids, c_mask)
            loss_ce = F.cross_entropy(logits, y)

            loss = loss_ce
            loss_sc = torch.tensor(0.0, device=device)
            if args.use_supcon and args.lambda_supcon > 0:
                loss_sc = supcon_loss(feats, y, temperature=args.supcon_temp)
                loss = loss + args.lambda_supcon * loss_sc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            ema_update(teacher, student.backbone, decay=args.ema_decay)

            pbar.set_postfix(ce=float(loss_ce.detach().cpu()), supcon=float(loss_sc.detach().cpu()))

        if src_val_loader is not None:
            teacher.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for w_ids, w_mask, c_ids, c_mask, y in src_val_loader:
                    w_ids, w_mask = w_ids.to(device), w_mask.to(device)
                    c_ids, c_mask = c_ids.to(device), c_mask.to(device)
                    y = y.to(device)
                    _f, logits = teacher(w_ids, w_mask, c_ids, c_mask)
                    pred = logits.argmax(dim=-1)
                    correct += int((pred == y).sum().item())
                    total += int(y.numel())
            val_acc = float(correct / max(1, total))
            print(f"[Pretrain] val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                best_state = copy.deepcopy(teacher.state_dict())

    if best_state is not None:
        teacher.load_state_dict(best_state)
        student.backbone.load_state_dict(best_state)

    # ------------- Calibration -------------
    calib_T = 1.0
    if args.use_calibration and src_val_loader is not None:
        logits_val, y_val = collect_logits_labels(teacher, src_val_loader, device=device)
        calib_T = fit_temperature(
            logits_val, y_val, device=device, t_min=args.temp_min, t_max=args.temp_max
        )
        print(f"[Calib] Fitted temperature T={calib_T:.4f} on source val (bounds {args.temp_min}-{args.temp_max})")
    else:
        print("[Calib] Skipped temperature scaling (no val or disabled)")

    # ------------- UDA -------------
    steps_per_epoch = max(1, max(len(src_train_loader), len(tgt_loader)))
    total_steps = args.uda_epochs * steps_per_epoch
    global_step = 0

    for uda_ep in range(1, args.uda_epochs + 1):
        alpha_pl_eff = 0.0 if uda_ep <= args.pl_warmup_epochs else args.alpha_pl
        da_ramp = min(1.0, uda_ep / max(1.0, float(args.da_ramp_epochs)))
        lambda_da_eff = float(args.lambda_da) * da_ramp

        # prototypes
        prototypes = None
        if args.use_proto:
            prototypes = compute_source_prototypes(teacher, src_train_loader, device=device, num_classes=2)

        # pseudo pool + update priors
        sel_idx, sel_soft, sel_w, p_ema, pi_target = build_pseudo_pool_v9_1(
            teacher=teacher,
            target_loader=tgt_eval_loader,
            device=device,
            calib_T=calib_T,
            teacher_temp=args.teacher_temp,
            p_ema=p_ema,
            pi_target=pi_target,
            pi_source=pi_source,
            prototypes=prototypes,
            uda_epoch=uda_ep,
            args=args,
        )

        pi_target_t = torch.tensor(pi_target, dtype=torch.float32, device=device)
        p_ema_t = torch.tensor(p_ema, dtype=torch.float32, device=device)

        pseudo_loader = None
        if sel_idx.size > 0 and alpha_pl_eff > 0:
            pseudo_ds = PseudoLabeledDataset(
                base_word=tgt_ds.word,
                base_char=tgt_ds.char,
                indices=sel_idx,
                soft_labels=sel_soft,
                weights=sel_w,
                vocab=vocab,
                max_len=args.max_len,
                max_char_len=args.max_char_len,
                seed=args.seed + 100 + uda_ep,
            )
            pseudo_loader = DataLoader(
                pseudo_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=partial(
                    collate_pseudo,
                    pad_word=vocab.pad_id,
                    pad_char=CHAR_PAD,
                    max_len=args.max_len,
                    max_char_len=args.max_char_len,
                ),
                drop_last=False,
            )
        pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

        student.train()
        teacher.eval()

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_loader)

        pbar = tqdm(range(steps_per_epoch), desc=f"UDA {uda_ep}/{args.uda_epochs}", leave=True)
        for _ in pbar:
            try:
                sw_ids, sw_mask, sc_ids, sc_mask, sy = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train_loader)
                sw_ids, sw_mask, sc_ids, sc_mask, sy = next(src_iter)

            try:
                tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask, tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask, tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask, _ = next(tgt_iter)

            sw_ids, sw_mask = sw_ids.to(device), sw_mask.to(device)
            sc_ids, sc_mask = sc_ids.to(device), sc_mask.to(device)
            sy = sy.to(device)

            tw_w_ids, tw_w_mask = tw_w_ids.to(device), tw_w_mask.to(device)
            tc_w_ids, tc_w_mask = tc_w_ids.to(device), tc_w_mask.to(device)
            tw_s_ids, tw_s_mask = tw_s_ids.to(device), tw_s_mask.to(device)
            tc_s_ids, tc_s_mask = tc_s_ids.to(device), tc_s_mask.to(device)

            # supervised on source (+ SupCon optional)
            src_feats, src_logits = student(sw_ids, sw_mask, sc_ids, sc_mask)
            loss_sup = F.cross_entropy(src_logits, sy)
            loss_supcon = torch.tensor(0.0, device=device)
            if args.use_supcon and args.lambda_supcon_uda > 0:
                loss_supcon = supcon_loss(src_feats, sy, temperature=args.supcon_temp)
                loss_sup = loss_sup + args.lambda_supcon_uda * loss_supcon

            # target forward (student strong)
            tgt_feats, tgt_logits_s = student(tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask)
            tgt_probs_s = torch.softmax(tgt_logits_s, dim=-1)

            # teacher weak (calibrated + DA)
            with torch.no_grad():
                _tf, t_logits = teacher(tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask)
                T_total = max(1e-6, calib_T * args.teacher_temp)
                t_probs = torch.softmax(t_logits / T_total, dim=-1)
                if args.use_da:
                    t_probs = distribution_align_torch(t_probs, p_ema_t, pi_target_t)

            # consistency
            logp_s = torch.log_softmax(tgt_logits_s, dim=-1)
            kl_con = F.kl_div(logp_s, t_probs, reduction="none").sum(dim=1)
            ent_t = entropy_torch(t_probs) / math.log(2.0)
            w_con = (1.0 - ent_t).detach()
            loss_con = (kl_con * w_con).mean()

            # pseudo label loss
            loss_pl = torch.tensor(0.0, device=device)
            if pseudo_iter is not None and alpha_pl_eff > 0:
                try:
                    pw_ids, pw_mask, pc_ids, pc_mask, psoft, pwt = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)  # type: ignore
                    pw_ids, pw_mask, pc_ids, pc_mask, psoft, pwt = next(pseudo_iter)

                pw_ids, pw_mask = pw_ids.to(device), pw_mask.to(device)
                pc_ids, pc_mask = pc_ids.to(device), pc_mask.to(device)
                psoft, pwt = psoft.to(device), pwt.to(device)

                _pf, pl_logits = student(pw_ids, pw_mask, pc_ids, pc_mask)
                logp = torch.log_softmax(pl_logits, dim=-1)
                kl_pl = F.kl_div(logp, psoft, reduction="none").sum(dim=1)
                loss_pl = (kl_pl * pwt).mean()

            # prior regularization
            loss_prior = torch.tensor(0.0, device=device)
            if args.gamma_prior > 0:
                loss_prior = prior_reg_loss(tgt_probs_s, pi_target_t)

            # MCC
            loss_mcc = torch.tensor(0.0, device=device)
            if args.use_mcc and args.gamma_mcc > 0:
                loss_mcc = mcc_loss(tgt_probs_s)

            # CDAN domain alignment (conditional)
            progress = global_step / max(1, total_steps)
            grl_l = dann_grl_lambda(progress)

            src_probs = torch.softmax(src_logits, dim=-1).detach()
            tgt_probs = tgt_probs_s.detach()

            if args.entropy_condition:
                w_src = (1.0 + torch.exp(-entropy_torch(src_probs))).detach()
                w_tgt = (1.0 + torch.exp(-entropy_torch(tgt_probs))).detach()
            else:
                w_src = torch.ones(src_probs.size(0), device=device)
                w_tgt = torch.ones(tgt_probs.size(0), device=device)

            src_cond = torch.bmm(src_probs.unsqueeze(2), src_feats.unsqueeze(1)).view(src_feats.size(0), -1)
            tgt_cond = torch.bmm(tgt_probs.unsqueeze(2), tgt_feats.unsqueeze(1)).view(tgt_feats.size(0), -1)

            dom_src = student.cdan_disc(grl(src_cond, grl_l))
            dom_tgt = student.cdan_disc(grl(tgt_cond, grl_l))

            loss_da = 0.5 * (
                (F.binary_cross_entropy_with_logits(dom_src, torch.zeros_like(dom_src), reduction="none") * w_src).mean()
                + (F.binary_cross_entropy_with_logits(dom_tgt, torch.ones_like(dom_tgt), reduction="none") * w_tgt).mean()
            )

            # CORAL
            loss_coral = torch.tensor(0.0, device=device)
            if args.gamma_coral > 0:
                loss_coral = coral_loss(src_feats, tgt_feats)

            loss = (
                loss_sup
                + args.beta_con * loss_con
                + alpha_pl_eff * loss_pl
                + lambda_da_eff * loss_da
                + args.gamma_coral * loss_coral
                + args.gamma_prior * loss_prior
                + args.gamma_mcc * loss_mcc
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            ema_update(teacher, student.backbone, decay=args.ema_decay)
            global_step += 1

            pbar.set_postfix(
                sup=float(loss_sup.detach().cpu()),
                supcon=float(loss_supcon.detach().cpu()),
                con=float(loss_con.detach().cpu()),
                pl=float(loss_pl.detach().cpu()),
                da=float(loss_da.detach().cpu()),
                coral=float(loss_coral.detach().cpu()),
                prior=float(loss_prior.detach().cpu()),
                mcc=float(loss_mcc.detach().cpu()),
                alpha=float(alpha_pl_eff),
                pseudo=int(sel_idx.size),
                pi=f"[{pi_target[0]:.2f},{pi_target[1]:.2f}]",
            )

    return teacher, vocab, pi_target, p_ema, calib_T, pi_source


# ----------------------------- Inference / Evaluation -------------------------


def pred_argmax_covid(probs_source_space: np.ndarray) -> np.ndarray:
    pred_source = probs_source_space.argmax(axis=1).astype(int)  # 0=true,1=false
    return np.where(pred_source == 0, 1, 0).astype(int)


def pred_matchprior_covid(probs_source_space: np.ndarray, pi_target_source_space: np.ndarray) -> np.ndarray:
    score_true = probs_source_space[:, 0].astype(np.float32)
    pi_true = float(np.clip(float(pi_target_source_space[0]), 0.01, 0.99))
    thr = float(np.quantile(score_true, 1.0 - pi_true))
    return (score_true >= thr).astype(int)


# ----------------------------- Main ------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent

    # Data (defaults read from script directory)
    parser.add_argument("--pubhealth_train", type=str, default=str(base_dir / "./pubhealth/pubhealth_train_clean.csv"))
    parser.add_argument("--pubhealth_val", type=str, default=str(base_dir / "./pubhealth/pubhealth_validation_clean.csv"))
    parser.add_argument("--covid_true", type=str, default=str(base_dir / "../covid/trueNews.csv"))
    parser.add_argument("--covid_fake", type=str, default=str(base_dir / "../covid/fakeNews.csv"))
    parser.add_argument("--out_csv", type=str, default=str(base_dir / "covid_predictions_dupl_v9_1.csv"))

    # Optional: split target into unlabeled-train and labeled-test (recommended for paper)
    parser.add_argument("--target_test_ratio", type=float, default=0.0, help="e.g., 0.2 for 80/20 split")
    parser.add_argument("--target_split_seed", type=int, default=42)

    # Preprocess
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--max_char_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)

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
    parser.add_argument("--uda_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Loss weights
    parser.add_argument("--lambda_da", type=float, default=0.05)
    parser.add_argument("--da_ramp_epochs", type=int, default=5)
    parser.add_argument("--gamma_coral", type=float, default=0.01)
    parser.add_argument("--alpha_pl", type=float, default=1.0)
    parser.add_argument("--beta_con", type=float, default=1.0)
    parser.add_argument("--gamma_prior", type=float, default=0.2)
    parser.add_argument("--gamma_mcc", type=float, default=0.05)

    # SupCon
    parser.add_argument("--lambda_supcon", type=float, default=0.10)
    parser.add_argument("--lambda_supcon_uda", type=float, default=0.05)
    parser.add_argument("--supcon_temp", type=float, default=0.1)

    # Teacher temperature / calibration
    parser.add_argument("--teacher_temp", type=float, default=1.2)
    parser.add_argument("--no_calibration", action="store_true")
    # Safer default: temp_min=0.5 to avoid over-sharpening
    parser.add_argument("--temp_min", type=float, default=0.5)
    parser.add_argument("--temp_max", type=float, default=10.0)

    # DA / prior
    parser.add_argument("--no_da", action="store_true")
    parser.add_argument("--da_momentum", type=float, default=0.99)
    parser.add_argument("--prior_estimation", type=str, default="em", choices=["uniform", "source", "em"])
    parser.add_argument("--prior_em_iter", type=int, default=100)
    parser.add_argument("--prior_momentum", type=float, default=0.9)
    parser.add_argument("--prior_min", type=float, default=0.05)

    # Robust prior estimation (V9.1)
    parser.add_argument("--prior_warmup_epochs", type=int, default=2,
                        help="Do not update prior in early UDA epochs")
    parser.add_argument("--prior_conf", type=float, default=0.92,
                        help="Confidence threshold for samples used in prior estimation")
    parser.add_argument("--prior_ent", type=float, default=0.75,
                        help="Entropy threshold for samples used in prior estimation")
    parser.add_argument("--prior_min_n", type=int, default=1200,
                        help="Minimum sample count for prior estimation; otherwise fallback")
    parser.add_argument("--prior_max_delta", type=float, default=0.06,
                        help="Max change of prior (class0) per epoch to prevent runaway")
    parser.add_argument("--prior_mix_ema", type=float, default=0.50,
                        help="Mix prior with p_ema to stabilize (0 disables)")

    # Pseudo schedule
    parser.add_argument("--pl_warmup_epochs", type=int, default=2)
    parser.add_argument("--k_per_class", type=int, default=700)
    parser.add_argument("--k_min_per_class", type=int, default=150)
    parser.add_argument("--k_ramp_epochs", type=int, default=4)
    parser.add_argument("--n_clusters", type=int, default=60)
    parser.add_argument("--top_m_per_cluster", type=int, default=10)

    # Pseudo thresholds (base + relax)
    parser.add_argument("--tau_conf", type=float, default=0.95)
    parser.add_argument("--tau_ent", type=float, default=0.55)
    parser.add_argument("--tau_kl", type=float, default=0.60)
    parser.add_argument("--kl_ceiling", type=float, default=2.0)
    parser.add_argument("--kl_step", type=float, default=0.10)
    parser.add_argument("--tau_proto_margin", type=float, default=0.05)
    parser.add_argument("--proto_margin_scale", type=float, default=0.20)

    parser.add_argument("--relax_steps", type=int, default=8)
    parser.add_argument("--conf_floor", type=float, default=0.55)
    parser.add_argument("--conf_step", type=float, default=0.05)
    parser.add_argument("--ent_ceiling", type=float, default=0.95)
    parser.add_argument("--ent_step", type=float, default=0.05)
    parser.add_argument("--margin_step", type=float, default=0.01)

    # module switches
    parser.add_argument("--no_proto", action="store_true")
    parser.add_argument("--no_agree", action="store_true")
    parser.add_argument("--no_mcc", action="store_true")
    parser.add_argument("--no_supcon", action="store_true")
    parser.add_argument("--no_entropy_condition", action="store_true", help="Disable entropy conditioning for CDAN (default ON)")

    # system
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--print_pseudo_stats", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    args.use_da = not args.no_da
    args.use_calibration = not args.no_calibration
    args.use_proto = not args.no_proto
    args.use_agree = not args.no_agree
    args.use_mcc = not args.no_mcc
    args.use_supcon = not args.no_supcon
    args.entropy_condition = not args.no_entropy_condition  # default ON

    # read source
    src_train_texts, src_train_y = read_pubhealth_csv(args.pubhealth_train)
    src_val_texts, src_val_y = ([], [])
    if args.pubhealth_val and Path(args.pubhealth_val).exists():
        src_val_texts, src_val_y = read_pubhealth_csv(args.pubhealth_val)

    print(f"[Data] Source train size: {len(src_train_texts)}")
    print(f"[Data] Source train label dist: {pd.Series(src_train_y).value_counts().to_dict()}")
    if src_val_texts:
        print(f"[Data] Source val size: {len(src_val_texts)}")

    # read target
    true_texts, true_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_true)
    fake_texts, fake_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_fake)

    all_texts = true_texts + fake_texts
    y_file = np.array([1] * len(true_texts) + [0] * len(fake_texts), dtype=np.int64)

    eval_available = not args.skip_eval
    eval_y: Optional[np.ndarray]
    if true_labels_opt is not None and fake_labels_opt is not None:
        y_col = np.array(true_labels_opt + fake_labels_opt, dtype=np.int64)
        agree = float((y_col == y_file).mean())
        agree_flip = float(((1 - y_col) == y_file).mean())
        if agree_flip > agree:
            print(f"[Warn] COVID label column seems reversed vs file names. Flip it. (agree={agree:.3f}, flip={agree_flip:.3f})")
            y_col = 1 - y_col
        eval_y = y_col
        print("[Data] Evaluation labels source: label column (Binary Label / Label)")
    else:
        eval_y = y_file
        print("[Data] Evaluation labels source: file membership (trueNews=1, fakeNews=0)")

    print(f"[Data] Target size (rows): {len(all_texts)}")
    if eval_available and eval_y is not None:
        print(f"[Data] Target label dist (eval): {pd.Series(eval_y).value_counts().to_dict()}")

    # optional target split
    if args.target_test_ratio > 0 and eval_y is not None:
        train_idx, test_idx = stratified_split_indices(eval_y, test_ratio=args.target_test_ratio, seed=args.target_split_seed)
        tgt_train_texts = [all_texts[i] for i in train_idx]
        tgt_test_texts = [all_texts[i] for i in test_idx]
        y_test = eval_y[test_idx]
        print(f"[Split] Target train(unlabeled)={len(tgt_train_texts)} | test(labeled)={len(tgt_test_texts)}")
    else:
        tgt_train_texts = all_texts
        tgt_test_texts = all_texts
        y_test = eval_y

    # dedup target train texts
    tgt_train_unique = list(dict.fromkeys(tgt_train_texts))
    print(f"[Data] Target unique texts for training: {len(tgt_train_unique)}")

    print(
        f"[Config] DA={args.use_da}, prior={args.prior_estimation}, Calib={args.use_calibration}, "
        f"Proto={args.use_proto}, Agree={args.use_agree}, MCC={args.use_mcc}, SupCon={args.use_supcon}, "
        f"CDAN_entropy_cond={args.entropy_condition}, RobustPrior(warmup={args.prior_warmup_epochs},maxΔ={args.prior_max_delta})"
    )

    teacher, vocab, pi_target, p_ema_last, calib_T, pi_source = train_v9_1(
        src_train_texts=src_train_texts,
        src_train_y=src_train_y,
        src_val_texts=src_val_texts if src_val_texts else None,
        src_val_y=src_val_y if src_val_y else None,
        tgt_texts=tgt_train_unique,
        args=args,
    )

    # inference on test set
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    infer_ds = InferenceDataset(tgt_test_texts, vocab, args.max_len, args.max_char_len)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_infer,
            pad_word=vocab.pad_id,
            pad_char=CHAR_PAD,
            max_len=args.max_len,
            max_char_len=args.max_char_len,
        ),
        drop_last=False,
    )

    teacher.eval()
    probs = np.zeros((len(infer_ds), 2), dtype=np.float32)
    with torch.no_grad():
        for w_ids, w_mask, c_ids, c_mask, idxs in tqdm(infer_loader, desc="Inference", leave=True):
            w_ids, w_mask = w_ids.to(device), w_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            _f, logits = teacher(w_ids, w_mask, c_ids, c_mask)
            T_total = max(1e-6, calib_T * args.teacher_temp)
            p = torch.softmax(logits / T_total, dim=-1).cpu().numpy()
            probs[idxs.numpy()] = p

    probs_adj = distribution_align_np(probs, p_ema_last, pi_target) if args.use_da else probs
    pred_argmax = pred_argmax_covid(probs_adj)
    pred_match = pred_matchprior_covid(probs_adj, pi_target)

    out_df = pd.DataFrame(
        {
            "Text": tgt_test_texts,
            "prob_source_true(label0)": probs[:, 0],
            "prob_source_false(label1)": probs[:, 1],
            "prob_adj_true(label0)": probs_adj[:, 0],
            "prob_adj_false(label1)": probs_adj[:, 1],
            "pred_covid_argmax": pred_argmax,
            "pred_covid_matchprior": pred_match,
            "y_true_covid(1=true,0=false)": y_test if (eval_available and y_test is not None) else None,
        }
    )
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved predictions to: {args.out_csv}")
    print(f"[Info] pi_source={pi_source}, pi_target(final)={pi_target}, p_ema_last={p_ema_last}, calib_T={calib_T:.4f}")
    print(f"[Info] Pred dist argmax: {pd.Series(pred_argmax).value_counts().to_dict()}")
    print(f"[Info] Pred dist matchprior: {pd.Series(pred_match).value_counts().to_dict()}")

    if eval_available and y_test is not None:
        for name, pred in [("argmax", pred_argmax), ("matchprior", pred_match)]:
            acc = accuracy_score(y_test, pred)
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            print(f"\n[Eval:{name}] acc={acc:.6f}")
            print(cm)
            print(classification_report(y_test, pred, digits=4))


if __name__ == "__main__":
    main()
