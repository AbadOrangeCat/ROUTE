#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text-DuPL (DUPL/DuPL adapted to NLP) with DistilBERT backbone.

What you get:
- Dual students (two DistilBERT encoders, no parameter sharing)
- Cross pseudo-labeling: weak -> pseudo label, strong -> supervised
- Progressive confidence threshold (cosine descent)
- Adaptive noise filtering via 1D 2-component GMM (EM) on per-sample pseudo loss
- Consistency regularization on filtered / untrusted samples
- Representation discrepancy loss (stop-grad) to encourage diverse reps

Quick start:
  pip install -U torch transformers datasets tqdm
  python train_dupl_distilbert.py

Default runs a quick demo on AG News with a small subsample.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Schedules / math utils
# -------------------------

def cosine_descent(start: float, end: float, t: int, T: int) -> float:
    """Cosine descent schedule: start -> end as t goes 0..T."""
    if T <= 0:
        return end
    t = max(0, min(t, T))
    return start - 0.5 * (start - end) * (1.0 - math.cos(math.pi * t / T))


def _log_normal_1d(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Log pdf of univariate Gaussian N(mu, sigma^2) evaluated at x."""
    sigma = sigma.clamp(min=1e-3)
    return -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


@torch.no_grad()
def gmm_clean_mask_1d(
    losses: torch.Tensor,
    gamma: float = 0.5,
    eta: float = 0.2,
    em_iters: int = 10,
) -> torch.Tensor:
    """
    Fit 1D 2-Gaussian mixture by EM on losses.

    Args:
        losses: [N] per-sample loss values (detached)
        gamma: threshold on P(noise|loss) to treat as noisy
        eta: minimum separation between component means to enable filtering
        em_iters: EM iterations

    Returns:
        mask_clean: bool [N], True means "clean" (keep pseudo-label supervision)
    """
    x = losses.detach().float().flatten()
    N = x.numel()
    if N < 4:
        return torch.ones_like(x, dtype=torch.bool)

    # Init by median split
    median = x.median()
    hi = x > median
    if hi.all() or (~hi).all():
        return torch.ones_like(x, dtype=torch.bool)

    x_lo, x_hi = x[~hi], x[hi]
    mu1, mu2 = x_lo.mean(), x_hi.mean()
    sigma1 = x_lo.std(unbiased=False).clamp(min=1e-3)
    sigma2 = x_hi.std(unbiased=False).clamp(min=1e-3)
    pi1 = torch.tensor(float((~hi).float().mean()), device=x.device).clamp(1e-3, 1 - 1e-3)
    pi2 = 1.0 - pi1

    # EM
    for _ in range(em_iters):
        log_p1 = torch.log(pi1 + 1e-6) + _log_normal_1d(x, mu1, sigma1)
        log_p2 = torch.log(pi2 + 1e-6) + _log_normal_1d(x, mu2, sigma2)
        log_d = torch.logaddexp(log_p1, log_p2)

        r1 = torch.exp(log_p1 - log_d)  # responsibility for comp1
        r2 = 1.0 - r1

        pi1 = r1.mean().clamp(1e-3, 1 - 1e-3)
        pi2 = 1.0 - pi1

        mu1 = (r1 * x).sum() / (r1.sum() + 1e-6)
        mu2 = (r2 * x).sum() / (r2.sum() + 1e-6)

        var1 = (r1 * (x - mu1) ** 2).sum() / (r1.sum() + 1e-6)
        var2 = (r2 * (x - mu2) ** 2).sum() / (r2.sum() + 1e-6)
        sigma1 = torch.sqrt(var1 + 1e-6).clamp(min=1e-3)
        sigma2 = torch.sqrt(var2 + 1e-6).clamp(min=1e-3)

    # Decide noise component = larger mean loss
    if mu1 > mu2:
        noise_prob = r1
        mu_noise, mu_clean = mu1, mu2
    else:
        noise_prob = r2
        mu_noise, mu_clean = mu2, mu1

    # Not separable => don't filter
    if (mu_noise - mu_clean).item() <= eta:
        return torch.ones_like(x, dtype=torch.bool)

    mask_noise = noise_prob > gamma
    return ~mask_noise


def kl_div_per_sample(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(p || q) per sample. p,q are probabilities [B,C]."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def discrepancy_loss(f1: torch.Tensor, f2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Stop-grad discrepancy:
      D(a,b) = -log(1 - cos(a, stopgrad(b)))
      L = D(f1,f2) + D(f2,f1)
    """
    cos12 = F.cosine_similarity(f1, f2.detach(), dim=-1)
    cos21 = F.cosine_similarity(f2, f1.detach(), dim=-1)
    d12 = -torch.log((1.0 - cos12).clamp(min=eps))
    d21 = -torch.log((1.0 - cos21).clamp(min=eps))
    return d12.mean() + d21.mean()


def _filter_forward_inputs(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Only keep keys accepted by backbone.forward (prevents token_type_ids crash on DistilBERT).
    """
    sig = inspect.signature(model.forward)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in batch.items() if k in allowed}


def apply_token_masking(batch: Dict[str, torch.Tensor], tokenizer, p: float) -> Dict[str, torch.Tensor]:
    """
    Simple "strong/weak" augmentation for text:
    randomly replace non-special tokens with [MASK].
    """
    if p <= 0.0 or "input_ids" not in batch:
        return batch
    if tokenizer.mask_token_id is None:
        return batch

    input_ids = batch["input_ids"]
    attn = batch.get("attention_mask", torch.ones_like(input_ids))

    is_special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id):
        if sid is not None:
            is_special |= (input_ids == sid)

    maskable = attn.bool() & (~is_special)
    rnd = torch.rand_like(input_ids.float())
    to_mask = maskable & (rnd < p)

    out = dict(batch)
    out["input_ids"] = input_ids.clone()
    out["input_ids"][to_mask] = tokenizer.mask_token_id
    return out


# -------------------------
# Model: DistilBERT backbone + head
# -------------------------

class EncoderWithHead(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, **batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = _filter_forward_inputs(self.backbone, batch)
        out = self.backbone(**batch)
        # DistilBERT has no pooler_output; use CLS token.
        rep = out.last_hidden_state[:, 0]  # [B,H]
        logits = self.classifier(self.dropout(rep))
        return logits, rep


class DuPLTextModel(nn.Module):
    """
    Two independent students.
    """
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.s1 = EncoderWithHead(model_name, num_labels, dropout)
        self.s2 = EncoderWithHead(model_name, num_labels, dropout)


# -------------------------
# Config + one training step
# -------------------------

@dataclass
class DuPLConfig:
    model_name: str = "distilbert-base-uncased"
    dataset: str = "ag_news"               # e.g. "ag_news" or "glue"
    dataset_config: Optional[str] = None   # e.g. "sst2" when dataset="glue"

    text_col1: Optional[str] = None
    text_col2: Optional[str] = None
    label_col: Optional[str] = None

    max_length: int = 128
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    total_steps: int = 200
    eval_steps: int = 50
    seed: int = 42

    # Semi-supervised
    labeled_ratio: float = 0.1
    max_train_samples: int = 4000  # quick demo; set -1 to disable

    # Loss weights
    lambda_sup: float = 1.0
    lambda_u: float = 1.0
    lambda_dis: float = 0.1
    lambda_reg: float = 0.5

    # Progressive threshold
    tau_start: float = 0.95
    tau_end: float = 0.60

    # GMM
    gmm_gamma: float = 0.5
    gmm_eta: float = 0.2
    gmm_em_iters: int = 10

    # Weak/strong augmentation
    weak_mask_prob: float = 0.05
    strong_mask_prob: float = 0.30

    temperature: float = 1.0

    save_dir: str = "./dupl_distilbert_ckpt"


class DuPLStep:
    def __init__(self, cfg: DuPLConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def __call__(
        self,
        model: DuPLTextModel,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Dict[str, torch.Tensor],
        step: int,
        total_steps: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # labeled
        y = labeled_batch["labels"].to(device)
        x_l = {k: v.to(device) for k, v in labeled_batch.items() if k != "labels"}

        # unlabeled
        x_u = {k: v.to(device) for k, v in unlabeled_batch.items() if k != "labels"}

        # progressive threshold
        tau = cosine_descent(self.cfg.tau_start, self.cfg.tau_end, step, total_steps)

        # weak/strong views
        u_w = apply_token_masking(x_u, self.tokenizer, self.cfg.weak_mask_prob)
        u_s = apply_token_masking(x_u, self.tokenizer, self.cfg.strong_mask_prob)

        # forward labeled
        logits_l1, rep_l1 = model.s1(**x_l)
        logits_l2, rep_l2 = model.s2(**x_l)
        loss_sup = 0.5 * (F.cross_entropy(logits_l1, y) + F.cross_entropy(logits_l2, y))

        # forward unlabeled weak/strong
        logits_u1_w, rep_u1_w = model.s1(**u_w)
        logits_u2_w, rep_u2_w = model.s2(**u_w)
        logits_u1_s, _ = model.s1(**u_s)
        logits_u2_s, _ = model.s2(**u_s)

        # discrepancy: labeled + unlabeled(weak) reps
        loss_dis = 0.5 * (discrepancy_loss(rep_l1, rep_l2) + discrepancy_loss(rep_u1_w, rep_u2_w))

        # pseudo labels from weak (detach)
        with torch.no_grad():
            p1_w = F.softmax(logits_u1_w / self.cfg.temperature, dim=-1)
            p2_w = F.softmax(logits_u2_w / self.cfg.temperature, dim=-1)
            conf1, y1_hat = p1_w.max(dim=-1)
            conf2, y2_hat = p2_w.max(dim=-1)
            trust1 = conf1 >= tau
            trust2 = conf2 >= tau

        # cross supervision losses (per sample)
        loss_2_on_y1 = F.cross_entropy(logits_u2_s, y1_hat, reduction="none")
        loss_1_on_y2 = F.cross_entropy(logits_u1_s, y2_hat, reduction="none")

        pseudo_1to2 = torch.zeros_like(trust1)
        pseudo_2to1 = torch.zeros_like(trust2)

        # GMM filter on trusted subset
        if trust1.any():
            clean = gmm_clean_mask_1d(
                loss_2_on_y1[trust1].detach(),
                gamma=self.cfg.gmm_gamma,
                eta=self.cfg.gmm_eta,
                em_iters=self.cfg.gmm_em_iters,
            )
            pseudo_1to2[trust1] = clean

        if trust2.any():
            clean = gmm_clean_mask_1d(
                loss_1_on_y2[trust2].detach(),
                gamma=self.cfg.gmm_gamma,
                eta=self.cfg.gmm_eta,
                em_iters=self.cfg.gmm_em_iters,
            )
            pseudo_2to1[trust2] = clean

        # pseudo-label loss (only clean samples)
        loss_u_12 = loss_2_on_y1[pseudo_1to2].mean() if pseudo_1to2.any() else torch.tensor(0.0, device=device)
        loss_u_21 = loss_1_on_y2[pseudo_2to1].mean() if pseudo_2to1.any() else torch.tensor(0.0, device=device)
        loss_u = 0.5 * (loss_u_12 + loss_u_21)

        # consistency on filtered/untrusted samples
        with torch.no_grad():
            p1_w_det = F.softmax(logits_u1_w / self.cfg.temperature, dim=-1)
            p2_w_det = F.softmax(logits_u2_w / self.cfg.temperature, dim=-1)
        p1_s = F.softmax(logits_u1_s / self.cfg.temperature, dim=-1)
        p2_s = F.softmax(logits_u2_s / self.cfg.temperature, dim=-1)

        filt1 = ~pseudo_1to2
        filt2 = ~pseudo_2to1

        reg1 = kl_div_per_sample(p1_w_det, p1_s)
        reg2 = kl_div_per_sample(p2_w_det, p2_s)

        loss_reg1 = reg1[filt1].mean() if filt1.any() else torch.tensor(0.0, device=device)
        loss_reg2 = reg2[filt2].mean() if filt2.any() else torch.tensor(0.0, device=device)
        loss_reg = 0.5 * (loss_reg1 + loss_reg2)

        total = (
            self.cfg.lambda_sup * loss_sup
            + self.cfg.lambda_dis * loss_dis
            + self.cfg.lambda_u * loss_u
            + self.cfg.lambda_reg * loss_reg
        )

        logs = {
            "tau": float(tau),
            "loss_sup": float(loss_sup.detach().cpu()),
            "loss_dis": float(loss_dis.detach().cpu()),
            "loss_u": float(loss_u.detach().cpu()),
            "loss_reg": float(loss_reg.detach().cpu()),
            "pseudo_rate_1to2": float(pseudo_1to2.float().mean().detach().cpu()),
            "pseudo_rate_2to1": float(pseudo_2to1.float().mean().detach().cpu()),
        }
        return total, logs


# -------------------------
# Dataset utilities
# -------------------------

def infer_columns(ds: Dataset) -> Tuple[str, Optional[str], str]:
    """
    Infer (text1, text2, label) columns from common dataset schemas.
    """
    cols = set(ds.column_names)

    # label column
    if "label" in cols:
        label_col = "label"
    elif "labels" in cols:
        label_col = "labels"
    else:
        raise ValueError(f"Can't infer label column from columns: {ds.column_names}")

    # single text
    if "text" in cols:
        return "text", None, label_col
    if "sentence" in cols:
        return "sentence", None, label_col
    if "content" in cols:
        return "content", None, label_col

    # text pair
    if "sentence1" in cols and "sentence2" in cols:
        return "sentence1", "sentence2", label_col
    if "question" in cols and "sentence" in cols:
        return "question", "sentence", label_col

    raise ValueError(f"Can't infer text columns from columns: {ds.column_names}")


def tokenize_dataset(
    dsdict: DatasetDict,
    tokenizer,
    text1: str,
    text2: Optional[str],
    label_col: str,
    max_length: int,
) -> DatasetDict:
    def tok_fn(batch):
        if text2 is None:
            return tokenizer(batch[text1], truncation=True, max_length=max_length)
        return tokenizer(batch[text1], batch[text2], truncation=True, max_length=max_length)

    # Remove original text columns to save RAM; keep label col (and idx if present)
    remove_cols = []
    for _, ds in dsdict.items():
        for c in ds.column_names:
            if c not in {label_col, "idx"}:
                if c not in remove_cols:
                    remove_cols.append(c)

    return dsdict.map(tok_fn, batched=True, remove_columns=remove_cols)


def stratified_labeled_indices(
    ds: Dataset, label_col: str, labeled_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified labeled/unlabeled split by class.
    Ensures at least 1 labeled example per class.
    """
    labels = np.array(ds[label_col])
    unique = np.unique(labels)
    rng = np.random.default_rng(seed)

    labeled_idx = []
    for y in unique:
        idx = np.where(labels == y)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * labeled_ratio))
        labeled_idx.append(idx[:k])

    labeled_idx = np.concatenate(labeled_idx)
    labeled_idx = np.unique(labeled_idx)
    unlabeled_idx = np.setdiff1d(np.arange(len(ds)), labeled_idx)

    rng.shuffle(labeled_idx)
    rng.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


def infinite_loader(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model: DuPLTextModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct_s1 = 0
    correct_s2 = 0
    correct_ens = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        x = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        logits1, _ = model.s1(**x)
        logits2, _ = model.s2(**x)

        pred1 = logits1.argmax(dim=-1)
        pred2 = logits2.argmax(dim=-1)
        ens = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1)) / 2.0
        pred_ens = ens.argmax(dim=-1)

        total += labels.size(0)
        correct_s1 += (pred1 == labels).sum().item()
        correct_s2 += (pred2 == labels).sum().item()
        correct_ens += (pred_ens == labels).sum().item()

    return {
        "acc_s1": correct_s1 / max(1, total),
        "acc_s2": correct_s2 / max(1, total),
        "acc_ens": correct_ens / max(1, total),
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_any_dataset(cfg: DuPLConfig) -> DatasetDict:
    """
    Supports:
      - ag_news (default)
      - glue/sst2: --dataset glue --dataset_config sst2
      - any HF dataset where we can infer text/label columns
    """
    if cfg.dataset_config is None:
        dsdict = load_dataset(cfg.dataset)
    else:
        dsdict = load_dataset(cfg.dataset, cfg.dataset_config)

    # ensure validation exists
    if "validation" not in dsdict:
        split = dsdict["train"].train_test_split(test_size=0.1, seed=cfg.seed)
        dsdict = DatasetDict(train=split["train"], validation=split["test"], test=dsdict.get("test"))

    # remove empty test if None
    if dsdict.get("test") is None:
        dsdict.pop("test", None)

    return dsdict


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DuPLConfig.model_name)
    parser.add_argument("--dataset", type=str, default=DuPLConfig.dataset)
    parser.add_argument("--dataset_config", type=str, default=None)

    parser.add_argument("--text_col1", type=str, default=None)
    parser.add_argument("--text_col2", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=DuPLConfig.max_length)
    parser.add_argument("--batch_size", type=int, default=DuPLConfig.batch_size)
    parser.add_argument("--lr", type=float, default=DuPLConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=DuPLConfig.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=DuPLConfig.warmup_ratio)
    parser.add_argument("--total_steps", type=int, default=DuPLConfig.total_steps)
    parser.add_argument("--eval_steps", type=int, default=DuPLConfig.eval_steps)
    parser.add_argument("--seed", type=int, default=DuPLConfig.seed)

    parser.add_argument("--labeled_ratio", type=float, default=DuPLConfig.labeled_ratio)
    parser.add_argument("--max_train_samples", type=int, default=DuPLConfig.max_train_samples)

    parser.add_argument("--lambda_sup", type=float, default=DuPLConfig.lambda_sup)
    parser.add_argument("--lambda_u", type=float, default=DuPLConfig.lambda_u)
    parser.add_argument("--lambda_dis", type=float, default=DuPLConfig.lambda_dis)
    parser.add_argument("--lambda_reg", type=float, default=DuPLConfig.lambda_reg)

    parser.add_argument("--tau_start", type=float, default=DuPLConfig.tau_start)
    parser.add_argument("--tau_end", type=float, default=DuPLConfig.tau_end)

    parser.add_argument("--gmm_gamma", type=float, default=DuPLConfig.gmm_gamma)
    parser.add_argument("--gmm_eta", type=float, default=DuPLConfig.gmm_eta)
    parser.add_argument("--gmm_em_iters", type=int, default=DuPLConfig.gmm_em_iters)

    parser.add_argument("--weak_mask_prob", type=float, default=DuPLConfig.weak_mask_prob)
    parser.add_argument("--strong_mask_prob", type=float, default=DuPLConfig.strong_mask_prob)
    parser.add_argument("--temperature", type=float, default=DuPLConfig.temperature)

    parser.add_argument("--save_dir", type=str, default=DuPLConfig.save_dir)

    args = parser.parse_args()

    cfg = DuPLConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        text_col1=args.text_col1,
        text_col2=args.text_col2,
        label_col=args.label_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        labeled_ratio=args.labeled_ratio,
        max_train_samples=args.max_train_samples,
        lambda_sup=args.lambda_sup,
        lambda_u=args.lambda_u,
        lambda_dis=args.lambda_dis,
        lambda_reg=args.lambda_reg,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        gmm_gamma=args.gmm_gamma,
        gmm_eta=args.gmm_eta,
        gmm_em_iters=args.gmm_em_iters,
        weak_mask_prob=args.weak_mask_prob,
        strong_mask_prob=args.strong_mask_prob,
        temperature=args.temperature,
        save_dir=args.save_dir,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # 1) Load dataset
    dsdict = load_any_dataset(cfg)

    # 2) Optional subsample train for quick demo
    if cfg.max_train_samples and cfg.max_train_samples > 0 and len(dsdict["train"]) > cfg.max_train_samples:
        dsdict["train"] = dsdict["train"].shuffle(seed=cfg.seed).select(range(cfg.max_train_samples))
        print(f"[Info] train subsampled to {len(dsdict['train'])} examples")

    # 3) Infer columns
    text1, text2, label_col = infer_columns(dsdict["train"])
    if cfg.text_col1 is not None:
        text1 = cfg.text_col1
    if cfg.text_col2 is not None:
        text2 = cfg.text_col2
    if cfg.label_col is not None:
        label_col = cfg.label_col
    print(f"[Info] Using columns: text1={text1}, text2={text2}, label={label_col}")

    # 4) Determine num_labels
    num_labels = None
    feats = dsdict["train"].features
    if label_col in feats and getattr(feats[label_col], "num_classes", None) is not None:
        num_labels = feats[label_col].num_classes
    if num_labels is None:
        num_labels = int(max(dsdict["train"][label_col])) + 1
    print(f"[Info] num_labels = {num_labels}")

    # 5) Tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenized = tokenize_dataset(dsdict, tokenizer, text1, text2, label_col, cfg.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # 6) Labeled / unlabeled split
    labeled_idx, unlabeled_idx = stratified_labeled_indices(tokenized["train"], label_col, cfg.labeled_ratio, cfg.seed)
    labeled_ds = tokenized["train"].select(labeled_idx.tolist())
    unlabeled_ds = tokenized["train"].select(unlabeled_idx.tolist()).remove_columns([label_col])

    print(f"[Info] Labeled={len(labeled_ds)}  Unlabeled={len(unlabeled_ds)}  Val={len(tokenized['validation'])}")

    labeled_loader = DataLoader(labeled_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(tokenized["validation"], batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    # 7) Model
    model = DuPLTextModel(cfg.model_name, num_labels=num_labels, dropout=0.1).to(device)

    # 8) Optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    warmup_steps = int(cfg.total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, cfg.total_steps)

    step_fn = DuPLStep(cfg)
    l_iter = infinite_loader(labeled_loader)
    u_iter = infinite_loader(unlabeled_loader)

    ensure_dir(cfg.save_dir)
    best_acc = -1.0
    best_path = os.path.join(cfg.save_dir, "best.pt")

    # 9) Train
    pbar = tqdm(range(cfg.total_steps), desc="training", dynamic_ncols=True)
    for step in pbar:
        model.train()

        l_batch = next(l_iter)
        u_batch = next(u_iter)

        loss, logs = step_fn(model, l_batch, u_batch, step, cfg.total_steps, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 10 == 0:
            pbar.set_postfix(
                loss=float(loss.detach().cpu()),
                sup=logs["loss_sup"],
                u=logs["loss_u"],
                dis=logs["loss_dis"],
                reg=logs["loss_reg"],
                tau=logs["tau"],
                pr12=logs["pseudo_rate_1to2"],
            )

        if (step + 1) % cfg.eval_steps == 0 or (step + 1) == cfg.total_steps:
            metrics = evaluate(model, val_loader, device)
            print(f"\n[Eval @ step {step+1}] {metrics}")

            if metrics["acc_ens"] > best_acc:
                best_acc = metrics["acc_ens"]
                torch.save(
                    {
                        "cfg": cfg.__dict__,
                        "num_labels": num_labels,
                        "model_state": model.state_dict(),
                        "tokenizer_name": cfg.model_name,
                        "best_acc_ens": best_acc,
                        "step": step + 1,
                    },
                    best_path,
                )
                print(f"[Info] New best acc_ens={best_acc:.4f}, saved to {best_path}")

    print(f"\n[Done] Best ensemble accuracy on validation: {best_acc:.4f}")
    print(f"[Done] Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
