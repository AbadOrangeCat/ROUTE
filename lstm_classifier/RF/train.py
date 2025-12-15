#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DUPL: Diversity & Uncertainty-aware Pseudo Labeling (non-LLM UDA baseline)

- Source domain: PubHealth (non-COVID medical fake news), labeled
  Uses only: claim + main_text + label
  label convention (source): 0=true, 1=false

- Target domain: COVID (trueNews.csv & fakeNews.csv), unlabeled for training
  Uses only: Text (train does NOT use any target label)
  evaluation label: if an existing binary label column is found (e.g., "Binary Label"),
                    use it; otherwise fall back to file membership:
                    trueNews=1 (true), fakeNews=0 (false)

Main ingredients:
  1) Supervised learning on source
  2) Domain-adversarial feature alignment (DANN, GRL + domain discriminator)
  3) Teacher-student (EMA) pseudo labeling with:
     - uncertainty filtering (entropy + confidence)
     - label-shift prior correction via EM
     - diversity selection via clustering coverage (top-m per cluster per class)
  4) Consistency regularization on all target samples (teacher weak vs student strong)

Output:
  - CSV with target texts + predicted probabilities and labels
  - Prints Accuracy/Confusion Matrix/Classification Report if evaluation labels are available
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

# ---- Torch import guard ------------------------------------------------------
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
    tqdm = lambda x, **kwargs: x


# ----------------------------- Utils -----------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Simple English tokenizer (non-LLM, no external deps)."""
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
    """Pad variable-length token id sequences."""
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


def weak_augment(ids: List[int], unk_id: int, rng: random.Random) -> List[int]:
    out = random_token_dropout(ids, p=0.05, unk_id=unk_id, rng=rng)
    out = random_swap(out, n_swaps=1, rng=rng)
    return out


def strong_augment(ids: List[int], unk_id: int, rng: random.Random) -> List[int]:
    out = random_deletion(ids, p=0.10, rng=rng)
    out = random_token_dropout(out, p=0.15, unk_id=unk_id, rng=rng)
    out = random_swap(out, n_swaps=3, rng=rng)
    return out


# ----------------------------- Datasets --------------------------------------


class SourceLabeledDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocab, max_len: int):
        assert len(texts) == len(labels)
        self.vocab = vocab
        self.max_len = max_len

        self.encoded: List[List[int]] = []
        self.labels: List[int] = []
        for t, y in zip(texts, labels):
            ids = vocab.encode(tokenize(t))
            if len(ids) == 0:
                ids = [vocab.unk_id]
            self.encoded.append(ids[:max_len])
            self.labels.append(int(y))

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        return self.encoded[idx], self.labels[idx]


def collate_source(batch, pad_id: int, max_len: int):
    seqs, labels = zip(*batch)
    input_ids, mask = pad_batch(seqs, pad_id=pad_id, max_len=max_len)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return input_ids, mask, labels_t


class TargetUnlabeledDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int, seed: int = 0):
        self.vocab = vocab
        self.max_len = max_len
        self.rng = random.Random(seed)

        self.texts: List[str] = [str(t) for t in texts]
        self.encoded: List[List[int]] = []
        for t in self.texts:
            ids = vocab.encode(tokenize(t))
            if len(ids) == 0:
                ids = [vocab.unk_id]
            self.encoded.append(ids[:max_len])

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        ids = self.encoded[idx]
        w = weak_augment(ids, unk_id=self.vocab.unk_id, rng=self.rng)
        s = strong_augment(ids, unk_id=self.vocab.unk_id, rng=self.rng)
        return w, s, idx


def collate_target(batch, pad_id: int, max_len: int):
    weak_seqs, strong_seqs, idxs = zip(*batch)
    weak_ids, weak_mask = pad_batch(weak_seqs, pad_id=pad_id, max_len=max_len)
    strong_ids, strong_mask = pad_batch(strong_seqs, pad_id=pad_id, max_len=max_len)
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    return weak_ids, weak_mask, strong_ids, strong_mask, idxs_t


class PseudoLabeledDataset(Dataset):
    def __init__(
        self,
        base_encoded: Sequence[Sequence[int]],
        indices: Sequence[int],
        soft_labels: np.ndarray,  # (N,C)
        weights: np.ndarray,      # (N,)
        vocab: Vocab,
        max_len: int,
        seed: int = 0,
    ):
        self.base = base_encoded
        self.indices = np.asarray(indices, dtype=np.int64)
        self.soft_labels = np.asarray(soft_labels, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.vocab = vocab
        self.max_len = max_len
        self.rng = random.Random(seed)

        assert len(self.indices) == len(self.soft_labels) == len(self.weights)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        ids = list(self.base[idx])[: self.max_len]
        ids_s = strong_augment(ids, unk_id=self.vocab.unk_id, rng=self.rng)
        y = self.soft_labels[i]
        w = self.weights[i]
        return ids_s, y, w


def collate_pseudo(batch, pad_id: int, max_len: int):
    seqs, soft_labels, weights = zip(*batch)
    input_ids, mask = pad_batch(seqs, pad_id=pad_id, max_len=max_len)
    y = torch.tensor(np.stack(soft_labels), dtype=torch.float32)  # (B,C)
    w = torch.tensor(weights, dtype=torch.float32)               # (B,)
    return input_ids, mask, y, w


class TargetInferenceDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int):
        self.texts: List[str] = [str(t) for t in texts]
        self.vocab = vocab
        self.max_len = max_len
        self.encoded: List[List[int]] = []
        for t in self.texts:
            ids = vocab.encode(tokenize(t))
            if len(ids) == 0:
                ids = [vocab.unk_id]
            self.encoded.append(ids[:max_len])

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        return self.encoded[idx], idx


def collate_infer(batch, pad_id: int, max_len: int):
    seqs, idxs = zip(*batch)
    input_ids, mask = pad_batch(seqs, pad_id=pad_id, max_len=max_len)
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    return input_ids, mask, idxs_t


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


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, pad_id: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = hidden_size * 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        out, _ = self.gru(x)  # (B,L,2H)

        mask = attention_mask.unsqueeze(-1)
        out = out * mask
        pooled = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return pooled


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: TextEncoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(input_ids, attention_mask)
        logits = self.classifier(feats)
        return feats, logits


class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


class StudentUDA(nn.Module):
    def __init__(self, backbone: EncoderClassifier):
        super().__init__()
        self.backbone = backbone
        self.domain_disc = DomainDiscriminator(backbone.encoder.out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(input_ids, attention_mask)


@torch.no_grad()
def ema_update(teacher: EncoderClassifier, student: EncoderClassifier, decay: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)


# ----------------------------- Pseudo label utilities ------------------------


def entropy_norm(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    C = probs.shape[1]
    ent = -(probs * np.log(probs + eps)).sum(axis=1)
    return ent / math.log(C)


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


def prior_correct(p: np.ndarray, pi_source: np.ndarray, pi_target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = (pi_target + eps) / (pi_source + eps)
    p_adj = p * w[None, :]
    p_adj = p_adj / (p_adj.sum(axis=1, keepdims=True) + eps)
    return p_adj.astype(np.float32)


def build_pseudo_label_pool(
    teacher: EncoderClassifier,
    target_loader: DataLoader,
    device: torch.device,
    pi_source: np.ndarray,
    n_clusters: int = 50,
    top_m: int = 20,
    tau_conf: float = 0.95,
    tau_ent: float = 0.35,
    min_pseudo: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      sel_indices: (M,)
      sel_soft_labels: (M,C)
      sel_weights: (M,)
      pi_target: (C,)
    """
    teacher.eval()
    feats_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    idx_all: List[np.ndarray] = []

    with torch.no_grad():
        for weak_ids, weak_mask, _strong_ids, _strong_mask, idxs in tqdm(
            target_loader, desc="Teacher forward (target)", leave=False
        ):
            weak_ids = weak_ids.to(device)
            weak_mask = weak_mask.to(device)
            feats, logits = teacher(weak_ids, weak_mask)
            probs = torch.softmax(logits, dim=-1)

            feats_all.append(feats.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            idx_all.append(idxs.numpy())

    feats = np.concatenate(feats_all, axis=0)
    probs = np.concatenate(probs_all, axis=0)
    idxs = np.concatenate(idx_all, axis=0)

    order = np.argsort(idxs)
    feats = feats[order]
    probs = probs[order]
    idxs = idxs[order]

    pi_t = estimate_target_prior_em(probs, pi_source)
    probs_adj = prior_correct(probs, pi_source=pi_source, pi_target=pi_t)

    conf = probs_adj.max(axis=1)
    ent = entropy_norm(probs_adj)
    pred = probs_adj.argmax(axis=1)

    N = feats.shape[0]
    k = int(min(max(2, n_clusters), N))
    top_m_eff = int(max(1, top_m))
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048, n_init=10)
    cluster_ids = kmeans.fit_predict(feats)

    keep = (conf >= tau_conf) & (ent <= tau_ent)
    selected_local: List[int] = []
    if keep.any():
        for c in range(k):
            for y in (0, 1):
                cand = np.where(keep & (cluster_ids == c) & (pred == y))[0]
                if cand.size == 0:
                    continue
                cand_sorted = cand[np.argsort(-conf[cand])]
                selected_local.extend(cand_sorted[:top_m_eff].tolist())

    selected_local = np.array(sorted(set(selected_local)), dtype=np.int64)

    if selected_local.size < min_pseudo:
        looser = (conf >= max(0.80, tau_conf - 0.10)) & (ent <= min(0.50, tau_ent + 0.10))
        cand = np.where(looser)[0]
        if cand.size > 0:
            cand_sorted = cand[np.argsort(-conf[cand])]
            need = min_pseudo - selected_local.size
            selected_local = np.array(
                sorted(set(selected_local.tolist() + cand_sorted[:need].tolist())), dtype=np.int64
            )

    if selected_local.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pi_t,
        )

    sel_soft = probs_adj[selected_local]
    sel_conf = conf[selected_local]
    sel_ent = ent[selected_local]

    w = np.clip((sel_conf - tau_conf) / max(1e-6, 1.0 - tau_conf), 0.0, 1.0) * (1.0 - sel_ent)
    w = w.astype(np.float32)

    sel_indices = idxs[selected_local].astype(np.int64)
    return sel_indices, sel_soft.astype(np.float32), w, pi_t


# ----------------------------- Training --------------------------------------


def dann_grl_lambda(progress: float) -> float:
    p = float(np.clip(progress, 0.0, 1.0))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def train(
    source_texts: Sequence[str],
    source_labels: Sequence[int],
    target_texts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[EncoderClassifier, Vocab, np.ndarray]:
    vocab = build_vocab(list(source_texts) + list(target_texts), min_freq=args.min_freq, max_size=args.vocab_size)

    src_ds = SourceLabeledDataset(source_texts, source_labels, vocab=vocab, max_len=args.max_len)
    tgt_ds = TargetUnlabeledDataset(target_texts, vocab=vocab, max_len=args.max_len, seed=args.seed + 7)

    src_loader = DataLoader(
        src_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_source, pad_id=vocab.pad_id, max_len=args.max_len),
        drop_last=False,
    )
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_target, pad_id=vocab.pad_id, max_len=args.max_len),
        drop_last=False,
    )
    tgt_eval_loader = DataLoader(
        tgt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_target, pad_id=vocab.pad_id, max_len=args.max_len),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    encoder = TextEncoder(
        vocab_size=len(vocab.itos),
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        pad_id=vocab.pad_id,
        dropout=args.dropout,
    )
    student_backbone = EncoderClassifier(encoder=encoder, num_classes=2)
    student = StudentUDA(backbone=student_backbone).to(device)

    teacher = copy.deepcopy(student_backbone).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    labels_np = np.asarray(source_labels, dtype=np.int64)
    pi_source = np.array([np.mean(labels_np == 0), np.mean(labels_np == 1)], dtype=np.float32)
    pi_source = pi_source / (pi_source.sum() + 1e-12)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    steps_per_epoch = max(1, max(len(src_loader), len(tgt_loader)))
    total_steps = args.epochs * steps_per_epoch

    last_pi_target = np.array([0.5, 0.5], dtype=np.float32)

    for epoch in range(1, args.epochs + 1):
        tau_conf = max(
            args.tau_conf_end,
            args.tau_conf_start - (epoch - 1) * (args.tau_conf_start - args.tau_conf_end) / max(1, args.epochs - 1),
        )

        sel_indices, sel_soft, sel_w, pi_t = build_pseudo_label_pool(
            teacher=teacher,
            target_loader=tgt_eval_loader,
            device=device,
            pi_source=pi_source,
            n_clusters=args.n_clusters,
            top_m=args.top_m_per_cluster,
            tau_conf=tau_conf,
            tau_ent=args.tau_ent,
            min_pseudo=args.min_pseudo,
        )
        last_pi_target = pi_t

        pseudo_loader: Optional[DataLoader]
        if sel_indices.size > 0:
            pseudo_ds = PseudoLabeledDataset(
                base_encoded=tgt_ds.encoded,
                indices=sel_indices,
                soft_labels=sel_soft,
                weights=sel_w,
                vocab=vocab,
                max_len=args.max_len,
                seed=args.seed + 17 + epoch,
            )
            pseudo_loader = DataLoader(
                pseudo_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=partial(collate_pseudo, pad_id=vocab.pad_id, max_len=args.max_len),
                drop_last=False,
            )
        else:
            pseudo_loader = None

        student.train()
        teacher.eval()

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for _ in pbar:
            try:
                src_ids, src_mask, src_y = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_ids, src_mask, src_y = next(src_iter)

            try:
                tgt_w_ids, tgt_w_mask, tgt_s_ids, tgt_s_mask, _tgt_idx = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_w_ids, tgt_w_mask, tgt_s_ids, tgt_s_mask, _tgt_idx = next(tgt_iter)

            src_ids = src_ids.to(device)
            src_mask = src_mask.to(device)
            src_y = src_y.to(device)

            tgt_w_ids = tgt_w_ids.to(device)
            tgt_w_mask = tgt_w_mask.to(device)
            tgt_s_ids = tgt_s_ids.to(device)
            tgt_s_mask = tgt_s_mask.to(device)

            src_feats, src_logits = student(src_ids, src_mask)
            loss_sup = F.cross_entropy(src_logits, src_y)

            tgt_feats, tgt_logits_s = student(tgt_s_ids, tgt_s_mask)
            with torch.no_grad():
                _tgt_feats_w, tgt_logits_w = teacher(tgt_w_ids, tgt_w_mask)
                tgt_probs_w = torch.softmax(tgt_logits_w, dim=-1)

            logp_s = torch.log_softmax(tgt_logits_s, dim=-1)
            kl = F.kl_div(logp_s, tgt_probs_w, reduction="none").sum(dim=1)
            ent_w = -(tgt_probs_w * torch.log(tgt_probs_w.clamp_min(1e-12))).sum(dim=1) / math.log(2.0)
            w_con = (1.0 - ent_w).detach()
            loss_con = (kl * w_con).mean()

            progress = global_step / max(1, total_steps)
            grl_l = dann_grl_lambda(progress)
            dom_src = student.domain_disc(grl(src_feats, grl_l))
            dom_tgt = student.domain_disc(grl(tgt_feats, grl_l))
            loss_da = 0.5 * (
                F.binary_cross_entropy_with_logits(dom_src, torch.zeros_like(dom_src))
                + F.binary_cross_entropy_with_logits(dom_tgt, torch.ones_like(dom_tgt))
            )

            loss_pl = torch.tensor(0.0, device=device)
            if pseudo_iter is not None:
                try:
                    pl_ids, pl_mask, pl_soft, pl_w = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)  # type: ignore[arg-type]
                    pl_ids, pl_mask, pl_soft, pl_w = next(pseudo_iter)

                pl_ids = pl_ids.to(device)
                pl_mask = pl_mask.to(device)
                pl_soft = pl_soft.to(device)
                pl_w = pl_w.to(device)

                _pl_feats, pl_logits = student(pl_ids, pl_mask)
                logp_pl = torch.log_softmax(pl_logits, dim=-1)
                kl_pl = F.kl_div(logp_pl, pl_soft, reduction="none").sum(dim=1)
                loss_pl = (kl_pl * pl_w).mean()

            loss = loss_sup + args.lambda_da * loss_da + args.alpha_pl * loss_pl + args.beta_con * loss_con

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            ema_update(teacher, student.backbone, decay=args.ema_decay)
            global_step += 1

            pbar.set_postfix(
                sup=float(loss_sup.detach().cpu()),
                da=float(loss_da.detach().cpu()),
                pl=float(loss_pl.detach().cpu()),
                con=float(loss_con.detach().cpu()),
                grl=float(grl_l),
                pseudo=int(sel_indices.size),
                pi_t="[{:.2f},{:.2f}]".format(float(pi_t[0]), float(pi_t[1])),
            )

    return teacher, vocab, last_pi_target


# ----------------------------- I/O -------------------------------------------


def _coerce_source_label_series_to_int01(label_series: pd.Series) -> pd.Series:
    """
    Source label convention: 0=true, 1=false.
    Accepts numeric or string labels like "true"/"false".
    Returns a numeric series (float) that may contain NaN for invalid.
    """
    if label_series.dtype == object:
        s = label_series.astype(str).str.strip().str.lower()
        s = s.replace({"true": "0", "false": "1", "real": "0", "fake": "1"})
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(label_series, errors="coerce")


def read_pubhealth_csv(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    need_cols = {"claim", "main_text", "label"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")

    y_num = _coerce_source_label_series_to_int01(df["label"])
    valid = y_num.isin([0, 1])

    if not bool(valid.all()):
        invalid_vals = df.loc[~valid, "label"].value_counts(dropna=False).head(10)
        print(f"[Warn] PubHealth labels not in {{0,1}}. Will drop {int((~valid).sum())} rows. Examples:\n{invalid_vals}")

    df = df.loc[valid].copy()
    y = y_num.loc[valid].astype(int).tolist()
    texts = (df["claim"].fillna("").astype(str) + " " + df["main_text"].fillna("").astype(str)).tolist()

    bad = [v for v in set(y) if v not in (0, 1)]
    if bad:
        raise ValueError(f"After cleaning, still found invalid source labels: {bad}")

    return texts, y


def read_covid_csv_text_and_optional_binary_label(path: str) -> Tuple[List[str], Optional[List[int]], Optional[str]]:
    """
    Returns:
      texts
      labels (0/1) if a suitable binary label column exists, else None
      which column used (or None)

    COVID label convention expected: 1=true, 0=false.
    """
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "Text" not in df.columns:
        raise ValueError(f"{path} missing column 'Text'. Found: {list(df.columns)}")

    texts = df["Text"].fillna("").astype(str).tolist()

    # Try to find a label column (case-insensitive)
    col_lower_map = {str(c).strip().lower(): c for c in df.columns}
    candidates = [
        "binary label",
        "binary_label",
        "binarylabel",
        "label",
    ]
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
        ss = ss.replace(
            {
                "true": "1",
                "false": "0",
                "real": "1",
                "fake": "0",
                "legit": "1",
                "legitimate": "1",
                "misleading": "0",
                "pants on fire": "0",
            }
        )
        y_num = pd.to_numeric(ss, errors="coerce")
    else:
        y_num = pd.to_numeric(s, errors="coerce")

    if not bool(y_num.isin([0, 1]).all()):
        return texts, None, None

    return texts, y_num.astype(int).tolist(), str(label_col)


# ----------------------------- Main ------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    base_dir = Path(__file__).resolve().parent  # script directory

    # Data defaults (can be overridden by CLI)
    parser.add_argument("--pubhealth_train", type=str, default=str(base_dir / "pubhealth_train_clean.csv"))
    parser.add_argument("--pubhealth_val", type=str, default=str(base_dir / "pubhealth_validation_clean.csv"))
    parser.add_argument("--covid_true", type=str, default=str(base_dir / "../covid/trueNews.csv"))
    parser.add_argument("--covid_fake", type=str, default=str(base_dir / "../covid/fakeNews.csv"))
    parser.add_argument("--out_csv", type=str, default=str(base_dir / "covid_predictions_dupl.csv"))

    # Preprocess
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)

    # Model
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Loss weights
    parser.add_argument("--lambda_da", type=float, default=1.0)
    parser.add_argument("--alpha_pl", type=float, default=1.0)
    parser.add_argument("--beta_con", type=float, default=1.0)

    # Pseudo labeling
    parser.add_argument("--tau_conf_start", type=float, default=0.95)
    parser.add_argument("--tau_conf_end", type=float, default=0.80)
    parser.add_argument("--tau_ent", type=float, default=0.35)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--top_m_per_cluster", type=int, default=20)
    parser.add_argument("--min_pseudo", type=int, default=500)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows safe
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # Evaluation switch
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation even if labels available")

    args = parser.parse_args()
    set_seed(args.seed)

    # required files
    must_exist = [args.pubhealth_train, args.covid_true, args.covid_fake]
    for fp in must_exist:
        if not Path(fp).exists():
            raise FileNotFoundError(
                f"File not found: {fp}\nbase_dir={base_dir}\n"
                f"Please fix path or override with CLI args."
            )

    if args.pubhealth_val and not Path(args.pubhealth_val).exists():
        print(f"[Warn] pubhealth_val not found: {args.pubhealth_val}, will skip.")
        args.pubhealth_val = None

    # source
    src_train_texts, src_train_y = read_pubhealth_csv(args.pubhealth_train)
    if args.pubhealth_val:
        src_val_texts, src_val_y = read_pubhealth_csv(args.pubhealth_val)
        source_texts = src_train_texts + src_val_texts
        source_y = src_train_y + src_val_y
    else:
        source_texts, source_y = src_train_texts, src_train_y

    print(f"[Data] Source size: {len(source_texts)}")
    print(f"[Data] Source label distribution: {pd.Series(source_y).value_counts().to_dict()}")

    # target: read texts (labels optional)
    true_texts, true_labels_opt, true_lab_col = read_covid_csv_text_and_optional_binary_label(args.covid_true)
    fake_texts, fake_labels_opt, fake_lab_col = read_covid_csv_text_and_optional_binary_label(args.covid_fake)

    eval_texts = true_texts + fake_texts

    # default eval labels by file membership
    y_file = np.array([1] * len(true_texts) + [0] * len(fake_texts), dtype=np.int64)

    eval_available = not args.skip_eval
    eval_y: Optional[np.ndarray] = None
    eval_src = "file membership (trueNews=1, fakeNews=0)"

    if true_labels_opt is not None and fake_labels_opt is not None:
        y_col = np.array(true_labels_opt + fake_labels_opt, dtype=np.int64)

        # Align to file membership if reversed
        agree = float((y_col == y_file).mean())
        agree_flip = float(((1 - y_col) == y_file).mean())
        if agree_flip > agree:
            print(
                f"[Warn] Detected target label column seems reversed vs file names "
                f"(agreement={agree:.3f}, flipped_agreement={agree_flip:.3f}). Will flip it."
            )
            y_col = 1 - y_col

        eval_y = y_col
        eval_src = f"label column ({true_lab_col or 'unknown'} / {fake_lab_col or 'unknown'})"
    else:
        eval_y = y_file

    print(f"[Data] Target size (rows): {len(eval_texts)}")
    if eval_available and eval_y is not None:
        print(f"[Data] Evaluation labels source: {eval_src}")
        print(f"[Data] Target label distribution (eval): {pd.Series(eval_y).value_counts().to_dict()}")

    # training target texts can be dedup for speed
    tgt_texts_train = list(dict.fromkeys(eval_texts))
    print(f"[Data] Target unique texts for training: {len(tgt_texts_train)}")

    teacher, vocab, pi_t = train(source_texts, source_y, tgt_texts_train, args)

    # inference on eval_texts
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    infer_ds = TargetInferenceDataset(eval_texts, vocab=vocab, max_len=args.max_len)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_infer, pad_id=vocab.pad_id, max_len=args.max_len),
        drop_last=False,
    )

    teacher.eval()
    probs_out = np.zeros((len(infer_ds), 2), dtype=np.float32)

    with torch.no_grad():
        for ids, mask, idxs in tqdm(infer_loader, desc="Inference", leave=True):
            ids = ids.to(device)
            mask = mask.to(device)
            _feats, logits = teacher(ids, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_out[idxs.numpy()] = probs

    # Source convention: 0=true, 1=false
    pred_source_label = probs_out.argmax(axis=1).astype(int)
    # COVID convention: 1=true, 0=false
    pred_covid_label = np.where(pred_source_label == 0, 1, 0).astype(int)

    # save output
    out_dict = {
        "Text": eval_texts,
        "prob_true": probs_out[:, 0],
        "prob_false": probs_out[:, 1],
        "pred_source_label(0=true,1=false)": pred_source_label,
        "pred_covid_label(1=true,0=false)": pred_covid_label,
    }
    if eval_available and eval_y is not None:
        out_dict["y_true_covid(1=true,0=false)"] = eval_y

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved predictions to: {args.out_csv}")
    print(f"[Info] Last estimated target prior pi_t (true/false): {pi_t}")

    # evaluation
    if eval_available and eval_y is not None:
        acc = accuracy_score(eval_y, pred_covid_label)
        cm = confusion_matrix(eval_y, pred_covid_label, labels=[0, 1])
        print(f"\n[Eval] COVID Accuracy (labels used ONLY for evaluation): {acc:.6f}")
        print("[Eval] Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
        print(cm)
        print("\n[Eval] Classification report:")
        print(classification_report(eval_y, pred_covid_label, digits=4))
    else:
        print("[Eval] Skipped (--skip_eval).")


if __name__ == "__main__":
    main()
