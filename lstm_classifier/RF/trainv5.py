#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DUPL V5 (Accuracy-oriented + anti-collapse):
- Non-LLM UDA baseline for PubHealth (source) -> COVID (target)
- Fixes target prediction bias via:
  (1) Distribution Alignment (DA) on teacher predictions (EMA of mean probs)
  (2) Prior Regularization on student target predictions
  (3) Inference: match-prior thresholding (unsupervised calibration)

Source labels (PubHealth): 0=true, 1=false
COVID eval labels: 1=true, 0=false
Target labels NEVER used in training; only for evaluation at the end.
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
except Exception as e:
    raise RuntimeError(
        "Failed to import PyTorch. Please install a working PyTorch build.\n"
        f"Original error: {e}"
    )

try:
    from tqdm import tqdm
except Exception:
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
        return [self.stoi.get(t, self.unk_id) for t in tokens]


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
    y = torch.tensor(np.stack(soft_labels), dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
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


# ----------------------------- Model (BiGRU + Attn) --------------------------


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
        attn = torch.softmax(scores, dim=-1)  # (B,L)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (B,D)
        return pooled


class TextEncoderAttn(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, pad_id: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.out_dim = hidden_size * 2
        self.attn = AttnPool(self.out_dim, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        h, _ = self.gru(x)  # (B,L,2H)
        h = self.dropout(h)
        pooled = self.attn(h, attention_mask)
        pooled = self.dropout(pooled)
        return pooled


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: TextEncoderAttn, num_classes: int = 2):
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


# ----------------------------- DA / Prior utils ------------------------------


def dann_grl_lambda(progress: float) -> float:
    p = float(np.clip(progress, 0.0, 1.0))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def entropy_norm(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    C = probs.shape[1]
    ent = -(probs * np.log(probs + eps)).sum(axis=1)
    return ent / math.log(C)


def distribution_align_np(probs: np.ndarray, p_ema: np.ndarray, pi_target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = probs / (p_ema[None, :] + eps) * pi_target[None, :]
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    return p.astype(np.float32)


def distribution_align_torch(probs: torch.Tensor, p_ema: torch.Tensor, pi_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = probs / (p_ema.unsqueeze(0) + eps) * pi_target.unsqueeze(0)
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    return p


def prior_reg_loss(student_probs: torch.Tensor, pi_target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL( mean(student_probs) || pi_target )
    """
    m = student_probs.mean(dim=0).clamp_min(eps)
    m = m / m.sum()
    return F.kl_div(torch.log(m), pi_target, reduction="sum")


# ----------------------------- Pseudo-label pool -----------------------------


@torch.no_grad()
def build_pseudo_pool(
    teacher: EncoderClassifier,
    target_loader: DataLoader,
    device: torch.device,
    p_ema: np.ndarray,
    pi_target: np.ndarray,
    uda_epoch: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      sel_indices, sel_soft, sel_w, new_p_ema
    """
    teacher.eval()
    feats_all, probs_all, idx_all = [], [], []

    for weak_ids, weak_mask, _strong_ids, _strong_mask, idxs in tqdm(target_loader, desc="Teacher forward (target)", leave=False):
        weak_ids = weak_ids.to(device)
        weak_mask = weak_mask.to(device)
        feats, logits = teacher(weak_ids, weak_mask)

        T = float(args.teacher_temp)
        probs = torch.softmax(logits / max(1e-6, T), dim=-1)

        feats_all.append(feats.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        idx_all.append(idxs.numpy())

    feats = np.concatenate(feats_all, axis=0)
    probs = np.concatenate(probs_all, axis=0)
    idxs = np.concatenate(idx_all, axis=0)

    order = np.argsort(idxs)
    feats, probs, idxs = feats[order], probs[order], idxs[order]

    # Update EMA of mean probs
    p_mean = probs.mean(axis=0).astype(np.float32)
    p_mean = p_mean / (p_mean.sum() + 1e-12)
    new_p_ema = args.da_momentum * p_ema + (1.0 - args.da_momentum) * p_mean
    new_p_ema = new_p_ema / (new_p_ema.sum() + 1e-12)

    # Apply distribution alignment for selection
    if args.use_da:
        probs_adj = distribution_align_np(probs, new_p_ema, pi_target)
    else:
        probs_adj = probs.astype(np.float32)

    conf = probs_adj.max(axis=1)
    ent = entropy_norm(probs_adj)
    pred = probs_adj.argmax(axis=1)
    score = conf * (1.0 - ent)

    # ramp k_per_class
    if uda_epoch <= args.pl_warmup_epochs:
        k_eff = 0
    else:
        t = min(1.0, (uda_epoch - args.pl_warmup_epochs) / max(1.0, float(args.k_ramp_epochs)))
        k_eff = int(max(args.k_min_per_class, round(args.k_per_class * t)))

    if k_eff <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), new_p_ema

    # clustering for diversity
    N = feats.shape[0]
    k = int(min(max(2, args.n_clusters), N))
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=2048, n_init=10)
    cluster_ids = kmeans.fit_predict(feats)

    selected: set[int] = set()

    def pick_for_class(y: int) -> List[int]:
        need = k_eff
        picked, picked_set = [], set()
        conf_thr, ent_thr = float(args.tau_conf), float(args.tau_ent)

        for _step in range(int(args.relax_steps) + 1):
            cand = np.where((pred == y) & (conf >= conf_thr) & (ent <= ent_thr))[0]
            if cand.size > 0:
                # per-cluster top-m
                for c in range(k):
                    if need <= 0:
                        break
                    cand_c = cand[cluster_ids[cand] == c]
                    if cand_c.size == 0:
                        continue
                    cand_sorted = cand_c[np.argsort(-score[cand_c])]
                    for ii in cand_sorted[: int(args.top_m_per_cluster)]:
                        if need <= 0:
                            break
                        ii = int(ii)
                        if ii in selected or ii in picked_set:
                            continue
                        picked.append(ii)
                        picked_set.add(ii)
                        need -= 1

                # fill remaining by global score
                if need > 0:
                    cand_sorted = cand[np.argsort(-score[cand])]
                    for ii in cand_sorted:
                        if need <= 0:
                            break
                        ii = int(ii)
                        if ii in selected or ii in picked_set:
                            continue
                        picked.append(ii)
                        picked_set.add(ii)
                        need -= 1

            if need <= 0:
                break

            # relax thresholds
            conf_thr = max(float(args.conf_floor), conf_thr - float(args.conf_step))
            ent_thr = min(float(args.ent_ceiling), ent_thr + float(args.ent_step))

        return picked

    for ii in pick_for_class(0):
        selected.add(ii)
    for ii in pick_for_class(1):
        selected.add(ii)

    if len(selected) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), new_p_ema

    sel_local = np.array(sorted(selected), dtype=np.int64)
    sel_soft = probs_adj[sel_local].astype(np.float32)

    # weights from score + class-balance reweight
    sel_score = score[sel_local].astype(np.float32)
    smin, smax = float(sel_score.min()), float(sel_score.max())
    if smax - smin < 1e-6:
        w = np.ones_like(sel_score, dtype=np.float32)
    else:
        w = (sel_score - smin) / (smax - smin + 1e-6)
        w = 0.2 + 0.8 * w

    hard = sel_soft.argmax(axis=1)
    counts = np.bincount(hard, minlength=2).astype(np.float32) + 1e-6
    inv = counts.sum() / counts
    inv = inv / inv.mean()
    w = w * inv[hard]
    w = w.astype(np.float32)

    sel_indices = idxs[sel_local].astype(np.int64)

    if args.print_pseudo_stats:
        dist = np.bincount(hard, minlength=2).tolist()
        print(
            f"[Pseudo][E{uda_epoch}] k_eff={k_eff} selected={len(sel_indices)} "
            f"dist(source 0/1 true/false)={dist} "
            f"p_ema=[{new_p_ema[0]:.2f},{new_p_ema[1]:.2f}] pi_tgt=[{pi_target[0]:.2f},{pi_target[1]:.2f}]"
        )

    return sel_indices, sel_soft, w, new_p_ema


# ----------------------------- Training --------------------------------------


@torch.no_grad()
def eval_source(model: EncoderClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        _f, logits = model(ids, mask)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct / max(1, total))


def alpha_ramp(epoch: int, warmup: int, ramp: int, target_alpha: float) -> float:
    if epoch <= warmup:
        return 0.0
    if ramp <= 0:
        return target_alpha
    t = min(1.0, (epoch - warmup) / float(ramp))
    return float(target_alpha * t)


def get_pi_target(mode: str, pi_source: np.ndarray) -> np.ndarray:
    mode = mode.lower().strip()
    if mode == "uniform":
        return np.array([0.5, 0.5], dtype=np.float32)
    if mode == "source":
        return np.asarray(pi_source, dtype=np.float32)
    # fallback
    return np.array([0.5, 0.5], dtype=np.float32)


def train_v5(
    src_train_texts: Sequence[str],
    src_train_y: Sequence[int],
    src_val_texts: Optional[Sequence[str]],
    src_val_y: Optional[Sequence[int]],
    tgt_texts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[EncoderClassifier, Vocab, np.ndarray, np.ndarray]:
    vocab = build_vocab(list(src_train_texts) + list(tgt_texts), min_freq=args.min_freq, max_size=args.vocab_size)

    src_train_ds = SourceLabeledDataset(src_train_texts, src_train_y, vocab=vocab, max_len=args.max_len)
    src_train_loader = DataLoader(
        src_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_source, pad_id=vocab.pad_id, max_len=args.max_len),
        drop_last=False,
    )

    src_val_loader = None
    if src_val_texts is not None and src_val_y is not None and len(src_val_texts) > 0:
        src_val_ds = SourceLabeledDataset(src_val_texts, src_val_y, vocab=vocab, max_len=args.max_len)
        src_val_loader = DataLoader(
            src_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=partial(collate_source, pad_id=vocab.pad_id, max_len=args.max_len),
            drop_last=False,
        )

    tgt_ds = TargetUnlabeledDataset(tgt_texts, vocab=vocab, max_len=args.max_len, seed=args.seed + 7)
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

    encoder = TextEncoderAttn(
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

    labels_np = np.asarray(src_train_y, dtype=np.int64)
    pi_source = np.array([np.mean(labels_np == 0), np.mean(labels_np == 1)], dtype=np.float32)
    pi_source = pi_source / (pi_source.sum() + 1e-12)

    pi_target = get_pi_target(args.target_prior_mode, pi_source)
    pi_target_t = torch.tensor(pi_target, dtype=torch.float32, device=device)

    # DA running mean init
    p_ema = np.array([0.5, 0.5], dtype=np.float32) if args.da_init_uniform else pi_source.copy()
    p_ema = p_ema / (p_ema.sum() + 1e-12)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------- A) Source pretrain ----------------
    best_val = -1.0
    best_state = None

    for ep in range(1, args.pretrain_epochs + 1):
        student.train()
        pbar = tqdm(src_train_loader, desc=f"Pretrain {ep}/{args.pretrain_epochs}", leave=True)
        for ids, mask, y in pbar:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            _f, logits = student(ids, mask)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            ema_update(teacher, student.backbone, decay=args.ema_decay)
            pbar.set_postfix(sup=float(loss.detach().cpu()))

        if src_val_loader is not None:
            val_acc = eval_source(teacher, src_val_loader, device)
            print(f"[Pretrain] val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                best_state = copy.deepcopy(teacher.state_dict())

    if best_state is not None:
        teacher.load_state_dict(best_state)
        student.backbone.load_state_dict(best_state)

    # ---------------- B) UDA stage ----------------
    steps_per_epoch = max(1, max(len(src_train_loader), len(tgt_loader)))
    total_steps = args.uda_epochs * steps_per_epoch
    global_step = 0

    for uda_ep in range(1, args.uda_epochs + 1):
        alpha_pl_eff = alpha_ramp(uda_ep, args.pl_warmup_epochs, args.pl_ramp_epochs, args.alpha_pl)

        da_ramp = min(1.0, uda_ep / max(1.0, float(args.da_ramp_epochs)))
        lambda_da_eff = float(args.lambda_da) * da_ramp

        # build pseudo pool once per epoch (and update p_ema)
        sel_indices, sel_soft, sel_w, p_ema = build_pseudo_pool(
            teacher=teacher,
            target_loader=tgt_eval_loader,
            device=device,
            p_ema=p_ema,
            pi_target=pi_target,
            uda_epoch=uda_ep,
            args=args,
        )

        pseudo_loader = None
        if sel_indices.size > 0:
            pseudo_ds = PseudoLabeledDataset(
                base_encoded=tgt_ds.encoded,
                indices=sel_indices,
                soft_labels=sel_soft,
                weights=sel_w,
                vocab=vocab,
                max_len=args.max_len,
                seed=args.seed + 100 + uda_ep,
            )
            pseudo_loader = DataLoader(
                pseudo_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=partial(collate_pseudo, pad_id=vocab.pad_id, max_len=args.max_len),
                drop_last=False,
            )
        pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

        # cache p_ema / pi_target on torch
        p_ema_t = torch.tensor(p_ema, dtype=torch.float32, device=device)

        student.train()
        teacher.eval()

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_loader)

        pbar = tqdm(range(steps_per_epoch), desc=f"UDA {uda_ep}/{args.uda_epochs}", leave=True)
        for _ in pbar:
            try:
                src_ids, src_mask, src_y = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train_loader)
                src_ids, src_mask, src_y = next(src_iter)

            try:
                tgt_w_ids, tgt_w_mask, tgt_s_ids, tgt_s_mask, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_w_ids, tgt_w_mask, tgt_s_ids, tgt_s_mask, _ = next(tgt_iter)

            src_ids, src_mask, src_y = src_ids.to(device), src_mask.to(device), src_y.to(device)
            tgt_w_ids, tgt_w_mask = tgt_w_ids.to(device), tgt_w_mask.to(device)
            tgt_s_ids, tgt_s_mask = tgt_s_ids.to(device), tgt_s_mask.to(device)

            # source supervised
            src_feats, src_logits = student(src_ids, src_mask)
            loss_sup = F.cross_entropy(src_logits, src_y)

            # target student strong
            tgt_feats, tgt_logits_s = student(tgt_s_ids, tgt_s_mask)
            tgt_probs_s = torch.softmax(tgt_logits_s, dim=-1)

            # teacher weak
            with torch.no_grad():
                _tf, tgt_logits_w = teacher(tgt_w_ids, tgt_w_mask)
                T = float(args.teacher_temp)
                tgt_probs_w = torch.softmax(tgt_logits_w / max(1e-6, T), dim=-1)

                if args.use_da:
                    tgt_probs_w = distribution_align_torch(tgt_probs_w, p_ema_t, pi_target_t)

            # consistency
            logp_s = torch.log_softmax(tgt_logits_s, dim=-1)
            kl = F.kl_div(logp_s, tgt_probs_w, reduction="none").sum(dim=1)

            ent_w = -(tgt_probs_w * torch.log(tgt_probs_w.clamp_min(1e-12))).sum(dim=1) / math.log(2.0)
            w_con = (1.0 - ent_w).detach()
            loss_con = (kl * w_con).mean()

            # domain adversarial
            progress = global_step / max(1, total_steps)
            grl_l = dann_grl_lambda(progress)

            dom_src = student.domain_disc(grl(src_feats, grl_l))
            dom_tgt = student.domain_disc(grl(tgt_feats, grl_l))

            # clamp logits to avoid rare huge BCE spikes
            dom_src = torch.clamp(dom_src, -10.0, 10.0)
            dom_tgt = torch.clamp(dom_tgt, -10.0, 10.0)

            loss_da = 0.5 * (
                F.binary_cross_entropy_with_logits(dom_src, torch.zeros_like(dom_src))
                + F.binary_cross_entropy_with_logits(dom_tgt, torch.ones_like(dom_tgt))
            )

            # pseudo-label loss
            loss_pl = torch.tensor(0.0, device=device)
            if pseudo_iter is not None and alpha_pl_eff > 0:
                try:
                    pl_ids, pl_mask, pl_soft, pl_w = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)  # type: ignore
                    pl_ids, pl_mask, pl_soft, pl_w = next(pseudo_iter)

                pl_ids, pl_mask = pl_ids.to(device), pl_mask.to(device)
                pl_soft, pl_w = pl_soft.to(device), pl_w.to(device)

                _pf, pl_logits = student(pl_ids, pl_mask)
                logp_pl = torch.log_softmax(pl_logits, dim=-1)
                kl_pl = F.kl_div(logp_pl, pl_soft, reduction="none").sum(dim=1)
                loss_pl = (kl_pl * pl_w).mean()

            # prior regularization (prevents collapse)
            loss_prior = prior_reg_loss(tgt_probs_s, pi_target_t) if args.gamma_prior > 0 else torch.tensor(0.0, device=device)

            loss = (
                loss_sup
                + lambda_da_eff * loss_da
                + args.beta_con * loss_con
                + alpha_pl_eff * loss_pl
                + args.gamma_prior * loss_prior
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            ema_update(teacher, student.backbone, decay=args.ema_decay)
            global_step += 1

            pbar.set_postfix(
                sup=float(loss_sup.detach().cpu()),
                da=float(loss_da.detach().cpu()),
                con=float(loss_con.detach().cpu()),
                pl=float(loss_pl.detach().cpu()),
                prior=float(loss_prior.detach().cpu()),
                alpha=float(alpha_pl_eff),
                lambda_da=float(lambda_da_eff),
                pseudo=int(sel_indices.size),
                p_ema=f"[{p_ema[0]:.2f},{p_ema[1]:.2f}]",
            )

    return teacher, vocab, pi_target, p_ema


# ----------------------------- I/O -------------------------------------------


def _coerce_source_label_series_to_int01(label_series: pd.Series) -> pd.Series:
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


# ----------------------------- Inference helper ------------------------------


def infer_predict_labels(
    probs_source: np.ndarray,  # (N,2) probs in source label space [true,false]
    infer_match_prior: bool,
    pi_target: np.ndarray,      # (2,) in source label space
) -> np.ndarray:
    """
    Convert to COVID label space (1=true, 0=false).
    If infer_match_prior:
      choose threshold so that predicted TRUE fraction matches pi_target[source_true]
    """
    score_true = probs_source[:, 0].astype(np.float32)  # source true prob -> covid true score
    if not infer_match_prior:
        pred_source = probs_source.argmax(axis=1).astype(int)        # 0=true,1=false
        pred_covid = np.where(pred_source == 0, 1, 0).astype(int)    # 1=true,0=false
        return pred_covid

    # desired fraction of covid_true is pi_target[source_true]
    pi_true = float(pi_target[0])
    pi_true = float(np.clip(pi_true, 0.05, 0.95))
    thr = float(np.quantile(score_true, 1.0 - pi_true))
    pred_covid = (score_true >= thr).astype(int)
    return pred_covid


# ----------------------------- Main ------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent

    parser.add_argument("--pubhealth_train", type=str, default=str(base_dir / "pubhealth_train_clean.csv"))
    parser.add_argument("--pubhealth_val", type=str, default=str(base_dir / "pubhealth_validation_clean.csv"))
    parser.add_argument("--covid_true", type=str, default=str(base_dir / "../covid/trueNews.csv"))
    parser.add_argument("--covid_fake", type=str, default=str(base_dir / "../covid/fakeNews.csv"))
    parser.add_argument("--out_csv", type=str, default=str(base_dir / "covid_predictions_dupl_v5.csv"))

    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)

    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--pretrain_epochs", type=int, default=4)
    parser.add_argument("--uda_epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--lambda_da", type=float, default=0.05)
    parser.add_argument("--da_ramp_epochs", type=int, default=5)

    parser.add_argument("--alpha_pl", type=float, default=1.0)
    parser.add_argument("--beta_con", type=float, default=1.0)
    parser.add_argument("--gamma_prior", type=float, default=0.2, help="Prior regularization weight (anti-collapse)")

    parser.add_argument("--pl_warmup_epochs", type=int, default=2)
    parser.add_argument("--pl_ramp_epochs", type=int, default=2)

    # pseudo selection
    parser.add_argument("--tau_conf", type=float, default=0.93)
    parser.add_argument("--tau_ent", type=float, default=0.60)
    parser.add_argument("--k_per_class", type=int, default=600)
    parser.add_argument("--k_min_per_class", type=int, default=150)
    parser.add_argument("--k_ramp_epochs", type=int, default=4)

    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--top_m_per_cluster", type=int, default=10)

    parser.add_argument("--relax_steps", type=int, default=8)
    parser.add_argument("--conf_floor", type=float, default=0.55)
    parser.add_argument("--conf_step", type=float, default=0.05)
    parser.add_argument("--ent_ceiling", type=float, default=0.95)
    parser.add_argument("--ent_step", type=float, default=0.05)

    # teacher temperature
    parser.add_argument("--teacher_temp", type=float, default=1.0)

    # distribution alignment
    parser.add_argument("--use_da", action="store_true", help="Enable distribution alignment (recommended)")
    parser.add_argument("--da_momentum", type=float, default=0.99)
    parser.add_argument("--da_init_uniform", action="store_true", help="Init DA EMA with uniform prior")

    # target prior choice for DA / prior reg / inference
    parser.add_argument("--target_prior_mode", type=str, default="uniform", choices=["uniform", "source"])

    # inference calibration
    parser.add_argument("--infer_match_prior", action="store_true", help="Match predicted positive rate to prior (recommended)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--print_pseudo_stats", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    # files check
    must_exist = [args.pubhealth_train, args.covid_true, args.covid_fake]
    for fp in must_exist:
        if not Path(fp).exists():
            raise FileNotFoundError(f"File not found: {fp}\nbase_dir={base_dir}")

    if args.pubhealth_val and not Path(args.pubhealth_val).exists():
        print(f"[Warn] pubhealth_val not found: {args.pubhealth_val}, will skip.")
        args.pubhealth_val = ""

    # read source
    src_train_texts, src_train_y = read_pubhealth_csv(args.pubhealth_train)
    src_val_texts, src_val_y = ([], [])
    if args.pubhealth_val:
        src_val_texts, src_val_y = read_pubhealth_csv(args.pubhealth_val)

    print(f"[Data] Source train size: {len(src_train_texts)}")
    print(f"[Data] Source train label dist: {pd.Series(src_train_y).value_counts().to_dict()}")
    if src_val_texts:
        print(f"[Data] Source val size: {len(src_val_texts)}")

    # read target
    true_texts, true_labels_opt, true_lab_col = read_covid_csv_text_and_optional_binary_label(args.covid_true)
    fake_texts, fake_labels_opt, fake_lab_col = read_covid_csv_text_and_optional_binary_label(args.covid_fake)

    eval_texts = true_texts + fake_texts
    y_file = np.array([1] * len(true_texts) + [0] * len(fake_texts), dtype=np.int64)

    eval_available = (not args.skip_eval)
    eval_y: Optional[np.ndarray] = None
    eval_src = "file membership (trueNews=1, fakeNews=0)"

    if true_labels_opt is not None and fake_labels_opt is not None:
        y_col = np.array(true_labels_opt + fake_labels_opt, dtype=np.int64)
        agree = float((y_col == y_file).mean())
        agree_flip = float(((1 - y_col) == y_file).mean())
        if agree_flip > agree:
            print(
                f"[Warn] Target label column seems reversed vs file names "
                f"(agreement={agree:.3f}, flipped_agreement={agree_flip:.3f}). Will flip."
            )
            y_col = 1 - y_col
        eval_y = y_col
        eval_src = f"label column ({true_lab_col or 'unknown'} / {fake_lab_col or 'unknown'})"
    else:
        eval_y = y_file

    print(f"[Data] Target size (rows): {len(eval_texts)}")
    if eval_available and eval_y is not None:
        print(f"[Data] Eval labels source: {eval_src}")
        print(f"[Data] Target label dist (eval): {pd.Series(eval_y).value_counts().to_dict()}")

    tgt_texts_train = list(dict.fromkeys(eval_texts))
    print(f"[Data] Target unique texts for training: {len(tgt_texts_train)}")

    teacher, vocab, pi_target, p_ema = train_v5(
        src_train_texts=src_train_texts,
        src_train_y=src_train_y,
        src_val_texts=src_val_texts if src_val_texts else None,
        src_val_y=src_val_y if src_val_y else None,
        tgt_texts=tgt_texts_train,
        args=args,
    )

    # inference
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

    pred_covid_label = infer_predict_labels(
        probs_source=probs_out,
        infer_match_prior=args.infer_match_prior,
        pi_target=pi_target,
    )

    out_dict = {
        "Text": eval_texts,
        "prob_source_true(label0)": probs_out[:, 0],
        "prob_source_false(label1)": probs_out[:, 1],
        "pred_covid_label(1=true,0=false)": pred_covid_label,
    }
    if eval_available and eval_y is not None:
        out_dict["y_true_covid(1=true,0=false)"] = eval_y

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved predictions to: {args.out_csv}")
    print(f"[Info] pi_target(source true/false)={pi_target}, p_ema_last={p_ema}")
    print(f"[Info] Predicted COVID label distribution (0/1): {pd.Series(pred_covid_label).value_counts().to_dict()}")

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
