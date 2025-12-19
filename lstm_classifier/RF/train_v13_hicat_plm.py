#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V13: HiCAT-PLM (Prototype-Aware Self-Training + Optional DAPT) - Inductive Early-Target Budget Sweep
---------------------------------------------------------------------------------------------------

Why a new solution (vs your V11/V11.2/V12):
  - Your previous architectures (BiGRU/CharCNN + UDA) can easily saturate at ~0.65-0.70 on COVID
    because the representation capacity is limited and UDA often causes negative transfer.
  - This V13 switches to a PRETRAINED Transformer encoder (RoBERTa/DeBERTa/etc.), then uses:
      (1) Optional DAPT: continued masked-LM pretraining on (source + target-train-budget) text only
          -> improves *representation* on the emergent sub-event vocabulary/style.
      (2) Prototype-aware FixMatch (PA-FM): class-conditional alignment using SOURCE prototypes,
          with EMA teacher + confidence/entropy filtering -> reduces negative transfer.
      (3) Strictly inductive: target TEST is never used in training, DAPT, threshold, or any stat.

Data conventions:
  - Source PubHealth: label {0:true, 1:false}
  - Target COVID: label {1:true, 0:false} (from "Binary Label"/"Label" column or filename membership)
  - Internally, we train with label {0:true, 1:false} for BOTH domains.
    Therefore, COVID labels are converted as: y_internal = 1 - y_covid.

Outputs:
  - v13_budget_results.csv
  - v13_budget_summary.csv
  - (optional) predictions csv per budget x seed

Dependencies:
  pip install transformers>=4.40 tokenizers accelerate (optional) scikit-learn pandas tqdm

Run examples:
  # quick check (3 budgets, 1 seed)
  python train_v13_hicat_plm.py --seeds "42" --target_budgets "0,200,-1" --model_name distilroberta-base

  # full sweep
  python train_v13_hicat_plm.py --seeds "42,43,44,45,46" --target_budgets "0,10,20,50,100,200,500,1000,2000,4000,-1"

  # turn on DAPT (stronger but slower)
  python train_v13_hicat_plm.py --dapt_steps 800 --dapt_lr 5e-5

Notes:
  - For a fair "early-target" story, DAPT uses ONLY the target-train budget texts of that run.
  - If you only care about best performance, run budget=-1 with larger dapt_steps and uda_epochs.

Author: ChatGPT (for research prototype use)
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        get_linear_schedule_with_warmup,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'transformers'. Install with:\n"
        "  pip install -U transformers tokenizers\n\n"
        f"Original error: {e}"
    )


# ----------------------------- Repro -----------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- Text utils ------------------------------------


URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def basic_tokens(text: str) -> List[str]:
    if text is None:
        return []
    t = str(text)
    t = URL_RE.sub(" URLTOKEN ", t)
    return WORD_RE.findall(t)


def strong_augment_text(text: str, rng: random.Random) -> str:
    """
    A simple strong augmentation for FixMatch-like training:
      - token deletion
      - token swap
    Keeps it cheap and deterministic under rng.
    """
    toks = basic_tokens(text)
    if len(toks) <= 4:
        return str(text)

    # deletion
    p_del = 0.12
    kept = [tok for tok in toks if rng.random() > p_del]
    if len(kept) < 4:
        kept = toks[:]

    # swap
    n_swaps = 2
    out = kept[:]
    L = len(out)
    for _ in range(n_swaps):
        i, j = rng.randrange(L), rng.randrange(L)
        out[i], out[j] = out[j], out[i]

    return " ".join(out)


# ----------------------------- Style features --------------------------------


@dataclass
class StyleNorm:
    mean: np.ndarray  # (D,)
    std: np.ndarray   # (D,)


def extract_style_features(text: str) -> np.ndarray:
    """
    Lightweight domain-robust cues that often transfer across topics:
      length, punctuation, casing, urls, digits, etc.
    """
    if text is None:
        text = ""
    s = str(text)
    s_strip = s.strip()
    s_lower = s_strip.lower()

    # basic counts
    n_chars = max(1, len(s_strip))
    toks = basic_tokens(s_strip)
    n_toks = max(1, len(toks))
    n_words = sum(tok.isalpha() for tok in toks)
    n_upper = sum(1 for ch in s_strip if ch.isupper())
    n_alpha = sum(1 for ch in s_strip if ch.isalpha())
    n_digit = sum(1 for ch in s_strip if ch.isdigit())

    n_excl = s_strip.count("!")
    n_q = s_strip.count("?")
    n_quote = s_strip.count('"') + s_strip.count("'")
    n_comma = s_strip.count(",")
    n_dot = s_strip.count(".")
    n_colon = s_strip.count(":")
    n_semi = s_strip.count(";")

    has_url = 1.0 if bool(URL_RE.search(s_strip)) else 0.0
    n_url = len(URL_RE.findall(s_strip))

    # lexical diversity
    words = [w.lower() for w in toks if w.isalpha()]
    uniq = len(set(words)) if words else 0
    lex_div = uniq / max(1, len(words))

    # ratios
    r_upper = n_upper / max(1, n_alpha)
    r_digit = n_digit / n_chars
    r_punct = (n_excl + n_q + n_quote + n_comma + n_dot + n_colon + n_semi) / n_chars

    avg_tok_len = float(np.mean([len(t) for t in toks])) if toks else 0.0
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    # special markers
    n_numtok = sum(tok.isdigit() for tok in toks)
    n_urltok = sum(tok.upper() == "URLTOKEN" for tok in toks)
    n_caps_words = sum(1 for tok in toks if tok.isalpha() and tok.isupper() and len(tok) >= 3)

    feat = np.array(
        [
            math.log1p(n_chars),
            math.log1p(n_toks),
            math.log1p(n_words),
            avg_tok_len,
            avg_word_len,
            lex_div,
            r_upper,
            r_digit,
            r_punct,
            float(n_excl),
            float(n_q),
            float(n_quote),
            float(n_url),
            has_url,
            float(n_numtok),
            float(n_urltok),
            float(n_caps_words),
        ],
        dtype=np.float32,
    )
    return feat


def compute_style_norm(texts: Sequence[str]) -> StyleNorm:
    X = np.stack([extract_style_features(t) for t in texts], axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return StyleNorm(mean=mean.astype(np.float32), std=std.astype(np.float32))


def norm_style(feat: np.ndarray, sn: StyleNorm) -> np.ndarray:
    return ((feat - sn.mean) / sn.std).astype(np.float32)


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


def split_target_by_unique_text(
    texts: Sequence[str],
    labels: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict inductive split:
      - group by identical Text
      - assign whole groups to train/test to avoid leakage from duplicates
    """
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


# ----------------------------- Datasets --------------------------------------


class LabeledTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], style_norm: StyleNorm):
        assert len(texts) == len(labels)
        self.texts = [str(t) for t in texts]
        self.labels = [int(y) for y in labels]
        self.style = np.stack([norm_style(extract_style_features(t), style_norm) for t in self.texts], axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.style[idx], self.labels[idx]


class UnlabeledAugDataset(Dataset):
    def __init__(self, texts: Sequence[str], style_norm: StyleNorm, seed: int):
        self.texts = [str(t) for t in texts]
        self.style = np.stack([norm_style(extract_style_features(t), style_norm) for t in self.texts], axis=0)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        t = self.texts[idx]
        t_s = strong_augment_text(t, rng=self.rng)
        return t, t_s, self.style[idx], idx


class MLMDataset(Dataset):
    def __init__(self, tokenized_examples: List[Dict[str, torch.Tensor]]):
        self.examples = tokenized_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


# ----------------------------- Model -----------------------------------------


class PLMStyleClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        style_dim: int,
        style_hidden: int = 64,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, style_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size + style_hidden, num_classes)
        self.out_dim = hidden_size + style_hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, style_feats: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # for RoBERTa/DistilRoBERTa: use first token representation
        h = out.last_hidden_state[:, 0]
        h = self.dropout(h)
        s = self.style_proj(style_feats)
        feat = torch.cat([h, s], dim=1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return feat, logits


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(decay).add_(s_p.data, alpha=1.0 - decay)


# ----------------------------- Prototype utilities ---------------------------


@torch.no_grad()
def compute_prototypes(
    model: PLMStyleClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> torch.Tensor:
    model.eval()
    feat_dim = model.out_dim
    sums = torch.zeros((num_classes, feat_dim), device=device)
    counts = torch.zeros((num_classes,), device=device)

    for batch in loader:
        enc, style, y = batch
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        style = style.to(device)
        y = y.to(device)

        feat, _logits = model(input_ids, attn, style)
        feat = F.normalize(feat, dim=1)

        for k in range(num_classes):
            m = (y == k)
            if bool(m.any()):
                sums[k] += feat[m].sum(dim=0)
                counts[k] += float(m.sum().item())

    protos = sums / counts.unsqueeze(1).clamp(min=1.0)
    protos = F.normalize(protos, dim=1)
    return protos


def prototype_ce_loss(feats: torch.Tensor, pseudo_y: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z = F.normalize(feats, dim=1)
    sims = (z @ prototypes.t()) / max(1e-6, temperature)
    return F.cross_entropy(sims, pseudo_y)


# ----------------------------- Calibration (temperature scaling) --------------


@torch.no_grad()
def collect_logits_labels(model: PLMStyleClassifier, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, y_list = [], []
    for enc, style, y in loader:
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        style = style.to(device)
        y = y.to(device)
        _f, logits = model(input_ids, attn, style)
        logits_list.append(logits.detach())
        y_list.append(y.detach())
    return torch.cat(logits_list, dim=0), torch.cat(y_list, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, device: torch.device, t_min: float = 0.5, t_max: float = 10.0) -> float:
    logits = logits.detach()
    labels = labels.detach()

    log_t = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        T = torch.exp(log_t).clamp(min=t_min, max=t_max)
        loss = ce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_t).detach().cpu().item())
    return float(np.clip(T, t_min, t_max))


# ----------------------------- Prediction / matchprior ------------------------


@torch.no_grad()
def predict_probs(
    model: PLMStyleClassifier,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    model.eval()
    probs_all: List[np.ndarray] = []
    for enc, style, _y in loader:
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        style = style.to(device)
        _f, logits = model(input_ids, attn, style)
        p = torch.softmax(logits / max(1e-6, temperature), dim=-1)
        probs_all.append(p.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(probs_all, axis=0)


def compute_matchprior_threshold(
    probs_train: np.ndarray,
    pi_true: float = 0.5,
    smooth_w: float = 1.0,
) -> float:
    """
    score_true = P(class0=true)
    We want predicted true fraction approx pi_true on target-train.
    For stability with tiny budgets:
      thr = smooth_w * quantile + (1-smooth_w) * 0.5
    """
    score_true = probs_train[:, 0].astype(np.float32)
    pi_true = float(np.clip(pi_true, 0.01, 0.99))
    q = float(np.quantile(score_true, 1.0 - pi_true))
    thr = smooth_w * q + (1.0 - smooth_w) * 0.5
    return float(np.clip(thr, 0.01, 0.99))


def pred_internal_argmax(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1).astype(int)  # 0=true,1=false


def pred_internal_matchprior(probs: np.ndarray, thr_true: float) -> np.ndarray:
    score_true = probs[:, 0].astype(np.float32)
    return np.where(score_true >= thr_true, 0, 1).astype(int)


def internal_to_covid(y_internal: np.ndarray) -> np.ndarray:
    # internal: 0=true,1=false -> covid: 1=true,0=false
    return (1 - y_internal).astype(int)


# ----------------------------- Training: DAPT --------------------------------


def run_dapt_mlm(
    model_name: str,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: torch.device,
    max_len: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    mlm_prob: float,
    seed: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Continued pretraining (DAPT) with masked LM on unlabeled texts.
    Returns base-model state_dict to initialize classifier encoder, or None if steps<=0.
    """
    if steps <= 0:
        return None

    set_seed(seed)
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    mlm_model.train()

    # Tokenize once (cheap enough for your dataset sizes)
    tokenized: List[Dict[str, torch.Tensor]] = []
    for t in texts:
        enc = tokenizer(
            str(t),
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors="pt",
        )
        tokenized.append({k: v.squeeze(0) for k, v in enc.items()})

    ds = MLMDataset(tokenized)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collator, drop_last=True)

    opt = torch.optim.AdamW(mlm_model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = steps
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, int(0.06 * total_steps)), num_training_steps=total_steps)

    it = iter(loader)
    pbar = tqdm(range(total_steps), desc="DAPT-MLM", leave=False)
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}
        out = mlm_model(**batch)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        pbar.set_postfix(loss=float(loss.detach().cpu()))

    # export base model weights
    prefix = getattr(mlm_model, "base_model_prefix", None)
    if prefix is None or not hasattr(mlm_model, prefix):
        # fallback: return full state_dict
        state = mlm_model.state_dict()
    else:
        base = getattr(mlm_model, prefix)
        state = base.state_dict()

    del mlm_model
    torch.cuda.empty_cache()
    return state


# ----------------------------- Training: supervised + UDA --------------------


def collate_labeled(tokenizer: AutoTokenizer, max_len: int, batch):
    texts, style, y = zip(*batch)
    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    style_t = torch.tensor(np.stack(style), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return enc, style_t, y_t


def collate_unlabeled(tokenizer: AutoTokenizer, max_len: int, batch):
    tw, ts, style, idx = zip(*batch)
    enc_w = tokenizer(
        list(tw),
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    enc_s = tokenizer(
        list(ts),
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    style_t = torch.tensor(np.stack(style), dtype=torch.float32)
    idx_t = torch.tensor(idx, dtype=torch.long)
    return enc_w, enc_s, style_t, idx_t


@torch.no_grad()
def eval_acc(model: PLMStyleClassifier, loader: DataLoader, device: torch.device, temperature: float = 1.0) -> float:
    model.eval()
    correct, total = 0, 0
    for enc, style, y in loader:
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        style = style.to(device)
        y = y.to(device)
        _f, logits = model(input_ids, attn, style)
        pred = (logits / max(1e-6, temperature)).argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct / max(1, total))


def train_single_run(
    seed: int,
    model_name: str,
    src_train_texts: Sequence[str],
    src_train_y: Sequence[int],
    src_val_texts: Sequence[str],
    src_val_y: Sequence[int],
    tgt_train_texts_budget: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[PLMStyleClassifier, float, StyleNorm]:
    """
    Returns:
      teacher(best), temperature, style_norm
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # style normalization uses ONLY train-time visible texts (inductive safe)
    style_norm = compute_style_norm(list(src_train_texts) + list(src_val_texts) + list(tgt_train_texts_budget))

    # tokenizer (pretrained vocab; no leakage from target-test)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Optional DAPT (continued MLM on source + target-train budget)
    dapt_state = None
    if args.dapt_steps > 0 and len(tgt_train_texts_budget) > 0:
        unlabeled_for_dapt = list(src_train_texts) + list(src_val_texts) + list(tgt_train_texts_budget)
        dapt_state = run_dapt_mlm(
            model_name=model_name,
            tokenizer=tokenizer,
            texts=unlabeled_for_dapt,
            device=device,
            max_len=args.max_len,
            steps=args.dapt_steps,
            batch_size=args.dapt_batch_size,
            lr=args.dapt_lr,
            weight_decay=args.dapt_weight_decay,
            mlm_prob=args.dapt_mlm_prob,
            seed=seed + 17,
        )

    # Build encoder
    encoder = AutoModel.from_pretrained(model_name).to(device)
    if dapt_state is not None:
        # load only base weights
        missing, unexpected = encoder.load_state_dict(dapt_state, strict=False)
        if args.verbose:
            print(f"[DAPT] loaded into encoder. missing={len(missing)} unexpected={len(unexpected)}")

    # Hidden size
    config = getattr(encoder, "config", None)
    hidden = None
    if config is not None:
        hidden = getattr(config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(config, "dim", None)
    if hidden is None:
        raise RuntimeError("Cannot infer hidden size from encoder config.")

    model = PLMStyleClassifier(
        encoder=encoder,
        hidden_size=int(hidden),
        style_dim=int(style_norm.mean.shape[0]),
        style_hidden=args.style_hidden,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)

    teacher = copy.deepcopy(model).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # datasets / loaders
    ds_tr = LabeledTextDataset(src_train_texts, src_train_y, style_norm)
    ds_va = LabeledTextDataset(src_val_texts, src_val_y, style_norm)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_labeled(tokenizer, args.max_len, b),
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_labeled(tokenizer, args.max_len, b),
        drop_last=False,
    )

    # optimizer & scheduler (supervised)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.sup_epochs * max(1, len(dl_tr))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, int(0.06 * total_steps)), num_training_steps=total_steps)

    best_val = -1.0
    best_state = None

    # ---- supervised training ----
    for ep in range(1, args.sup_epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"[Seed {seed}] SUP {ep}/{args.sup_epochs}", leave=False)
        for enc, style, y in pbar:
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            style = style.to(device)
            y = y.to(device)

            _f, logits = model(input_ids, attn, style)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            ema_update(teacher, model, decay=args.ema_decay)

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        val_acc = eval_acc(teacher, dl_va, device=device, temperature=1.0)
        if args.verbose:
            print(f"[Seed {seed}][SUP] val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(teacher.state_dict())

    if best_state is not None:
        teacher.load_state_dict(best_state)
        model.load_state_dict(best_state)

    # ---- temperature scaling on source-val (optional but recommended) ----
    temperature = 1.0
    if args.use_calibration:
        logits_val, y_val = collect_logits_labels(teacher, dl_va, device=device)
        temperature = fit_temperature(logits_val, y_val, device=device, t_min=args.temp_min, t_max=args.temp_max)
        if args.verbose:
            print(f"[Seed {seed}] Temperature={temperature:.4f}")

    # ---- UDA (Prototype-aware FixMatch) ----
    budget = len(tgt_train_texts_budget)
    if budget < args.min_budget_for_uda or args.uda_epochs <= 0:
        return teacher, temperature, style_norm

    ds_tu = UnlabeledAugDataset(tgt_train_texts_budget, style_norm, seed=seed + 999)
    dl_tu = DataLoader(
        ds_tu,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_unlabeled(tokenizer, args.max_len, b),
        drop_last=False,
    )

    # UDA optimizer (new for student model; teacher is EMA)
    # (we continue training from supervised weights)
    opt_u = torch.optim.AdamW(model.parameters(), lr=args.uda_lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, max(len(dl_tr), len(dl_tu)))
    total_u_steps = args.uda_epochs * steps_per_epoch
    sched_u = get_linear_schedule_with_warmup(
        opt_u, num_warmup_steps=max(1, int(0.06 * total_u_steps)), num_training_steps=total_u_steps
    )

    # do-no-harm checkpointing:
    best_ckpt = copy.deepcopy(teacher.state_dict())
    best_ckpt_score = -1e9
    best_src_val = best_val

    src_iter = None
    tgt_iter = None

    for ep in range(1, args.uda_epochs + 1):
        # prototypes computed from SOURCE labeled (teacher)
        protos = compute_prototypes(teacher, dl_tr, device=device, num_classes=2)

        # budget-aware scaling
        scale = 1.0
        if args.budget_aware:
            scale = min(1.0, float(budget) / max(1.0, float(args.unsup_budget_ref)))

        # epoch-wise tau schedule (start high, then relax)
        if args.tau_schedule:
            tau_hi = float(args.tau_hi)
            tau_lo = float(args.tau_lo)
            prog = min(1.0, (ep - 1) / max(1.0, args.uda_epochs - 1))
            tau = tau_hi - (tau_hi - tau_lo) * prog
        else:
            tau = float(args.tau)

        ent_thr = float(args.ent_thr)

        lambda_u = float(args.lambda_u) * scale
        lambda_proto = float(args.lambda_proto) * scale

        model.train()
        teacher.eval()

        src_iter = iter(dl_tr)
        tgt_iter = iter(dl_tu)

        pbar = tqdm(range(steps_per_epoch), desc=f"[Seed {seed}] UDA {ep}/{args.uda_epochs}", leave=False)
        pseudo_kept = 0
        pseudo_total = 0

        for _ in pbar:
            try:
                enc_s, style_s, y_s = next(src_iter)
            except StopIteration:
                src_iter = iter(dl_tr)
                enc_s, style_s, y_s = next(src_iter)

            try:
                enc_w, enc_t, style_t, _idx = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(dl_tu)
                enc_w, enc_t, style_t, _idx = next(tgt_iter)

            # source batch
            input_ids_s = enc_s["input_ids"].to(device)
            attn_s = enc_s["attention_mask"].to(device)
            style_s = style_s.to(device)
            y_s = y_s.to(device)

            # target weak/strong
            input_ids_w = enc_w["input_ids"].to(device)
            attn_w = enc_w["attention_mask"].to(device)
            input_ids_t = enc_t["input_ids"].to(device)
            attn_t = enc_t["attention_mask"].to(device)
            style_t = style_t.to(device)

            # supervised loss on source
            feat_s, logit_s = model(input_ids_s, attn_s, style_s)
            loss_sup = F.cross_entropy(logit_s, y_s)

            # teacher pseudo-label on weak
            with torch.no_grad():
                _fw, logit_w = teacher(input_ids_w, attn_w, style_t)
                prob_w = torch.softmax(logit_w / max(1e-6, temperature), dim=-1)
                max_prob, pseudo_y = prob_w.max(dim=-1)
                ent = -(prob_w * torch.log(prob_w.clamp_min(1e-12))).sum(dim=-1) / math.log(2.0)
                mask = (max_prob >= tau) & (ent <= ent_thr)

            pseudo_total += int(mask.numel())
            pseudo_kept += int(mask.sum().item())

            # unsupervised loss on strong (masked)
            loss_u = torch.tensor(0.0, device=device)
            loss_p = torch.tensor(0.0, device=device)

            if bool(mask.any()):
                feat_t, logit_t = model(input_ids_t, attn_t, style_t)
                loss_u = F.cross_entropy(logit_t[mask], pseudo_y[mask])

                if lambda_proto > 0:
                    loss_p = prototype_ce_loss(feat_t[mask], pseudo_y[mask], protos, temperature=args.proto_temp)

            loss = loss_sup + lambda_u * loss_u + lambda_proto * loss_p

            opt_u.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_u.step()
            sched_u.step()

            ema_update(teacher, model, decay=args.ema_decay)

            keep_rate = pseudo_kept / max(1, pseudo_total)
            pbar.set_postfix(
                sup=float(loss_sup.detach().cpu()),
                u=float(loss_u.detach().cpu()),
                proto=float(loss_p.detach().cpu()),
                keep=f"{keep_rate:.2f}",
                tau=f"{tau:.2f}",
            )

        # ---- checkpoint selection (inductive) ----
        src_val_acc = eval_acc(teacher, dl_va, device=device, temperature=1.0)
        keep_rate = pseudo_kept / max(1, pseudo_total)
        # discourage collapse: keep_rate too small or too large is suspicious
        keep_score = float(1.0 - abs(keep_rate - args.keep_target))
        ckpt_score = src_val_acc + args.ckpt_w_keep * keep_score

        # do-no-harm constraint
        if src_val_acc >= (best_src_val - args.src_val_tol):
            if ckpt_score > best_ckpt_score:
                best_ckpt_score = ckpt_score
                best_ckpt = copy.deepcopy(teacher.state_dict())

        if src_val_acc > best_src_val:
            best_src_val = src_val_acc

        if args.verbose:
            print(f"[Seed {seed}][UDA ep{ep}] src_val={src_val_acc:.4f} keep={keep_rate:.3f} ckpt_score={ckpt_score:.4f}")

    teacher.load_state_dict(best_ckpt)
    return teacher, temperature, style_norm


# ----------------------------- Budget sweep ----------------------------------


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

    # Sweep
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--target_budgets", type=str, default="0,10,20,50,100,200,500,1000,2000,4000,-1")
    parser.add_argument("--delta_to_full", type=float, default=0.01)

    # Output
    parser.add_argument("--out_dir", type=str, default=str(base_dir / "v13_runs"))
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--results_csv", type=str, default=str(base_dir / "v13_budget_results.csv"))
    parser.add_argument("--summary_csv", type=str, default=str(base_dir / "v13_budget_summary.csv"))

    # Model
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--style_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Supervised train
    parser.add_argument("--sup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--uda_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Calibration
    parser.add_argument("--use_calibration", action="store_true")
    parser.add_argument("--temp_min", type=float, default=0.5)
    parser.add_argument("--temp_max", type=float, default=10.0)

    # UDA (PA-FixMatch)
    parser.add_argument("--uda_epochs", type=int, default=4)
    parser.add_argument("--min_budget_for_uda", type=int, default=200)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_proto", type=float, default=0.2)
    parser.add_argument("--proto_temp", type=float, default=0.1)

    # pseudo filters
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--ent_thr", type=float, default=0.85)
    parser.add_argument("--tau_schedule", action="store_true")
    parser.add_argument("--tau_hi", type=float, default=0.98)
    parser.add_argument("--tau_lo", type=float, default=0.93)

    # budget-aware scaling
    parser.add_argument("--budget_aware", action="store_true")
    parser.add_argument("--unsup_budget_ref", type=int, default=500)

    # inductive checkpointing
    parser.add_argument("--keep_target", type=float, default=0.35, help="expected pseudo keep-rate (0~1), used for ckpt heuristic")
    parser.add_argument("--ckpt_w_keep", type=float, default=0.05)
    parser.add_argument("--src_val_tol", type=float, default=0.03)

    # DAPT (MLM)
    parser.add_argument("--dapt_steps", type=int, default=0, help=">0 enables DAPT (continued MLM) per run/budget")
    parser.add_argument("--dapt_batch_size", type=int, default=16)
    parser.add_argument("--dapt_lr", type=float, default=5e-5)
    parser.add_argument("--dapt_weight_decay", type=float, default=0.01)
    parser.add_argument("--dapt_mlm_prob", type=float, default=0.15)

    # matchprior threshold
    parser.add_argument("--pi_true", type=float, default=0.5, help="assumed target true prior for matchprior; set 0.5 if unknown")
    parser.add_argument("--thr_budget_ref", type=int, default=500)

    # system
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[V13] device={device}")
    seeds = parse_int_list(args.seeds)
    budgets = parse_budget_list(args.target_budgets)
    print(f"[V13] seeds={seeds}")
    print(f"[V13] budgets={budgets} (-1 means full)")
    print(f"[V13] split_seed={args.target_split_seed} test_ratio={args.target_test_ratio}")
    print(f"[V13] model={args.model_name} | min_budget_for_uda={args.min_budget_for_uda} | dapt_steps={args.dapt_steps}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load source ----------
    src_train_texts, src_train_y = read_pubhealth_csv(args.pubhealth_train)
    src_val_texts, src_val_y = read_pubhealth_csv(args.pubhealth_val)
    print(f"[Data] Source train={len(src_train_texts)} | val={len(src_val_texts)}")

    # ---------- Load target ----------
    true_texts, true_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_true)
    fake_texts, fake_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_fake)
    all_texts = true_texts + fake_texts

    y_file = np.array([1] * len(true_texts) + [0] * len(fake_texts), dtype=np.int64)  # covid convention (1=true)

    if true_labels_opt is not None and fake_labels_opt is not None:
        y_col = np.array(true_labels_opt + fake_labels_opt, dtype=np.int64)
        # auto-fix if reversed
        agree = float((y_col == y_file).mean())
        agree_flip = float(((1 - y_col) == y_file).mean())
        if agree_flip > agree:
            y_col = 1 - y_col
        y_covid = y_col
        print("[Data] Target label source: column (Binary Label / Label)")
    else:
        y_covid = y_file
        print("[Data] Target label source: file membership")

    print(f"[Data] Target rows={len(all_texts)} | dist={pd.Series(y_covid).value_counts().to_dict()}")

    # convert to internal labels: 0=true,1=false
    y_internal = (1 - y_covid).astype(np.int64)

    # ---------- Inductive split ----------
    train_idx, test_idx = split_target_by_unique_text(all_texts, y_internal, test_ratio=args.target_test_ratio, seed=args.target_split_seed)
    tgt_train_unique_full = list(dict.fromkeys([all_texts[i] for i in train_idx]))
    tgt_test_texts = [all_texts[i] for i in test_idx]
    y_test_internal = y_internal[test_idx]
    print(f"[Split] target-train unique={len(tgt_train_unique_full)} | target-test rows={len(test_idx)}")

    # ---------- Sweep ----------
    records: List[Dict[str, object]] = []
    budget_to_seed_acc: Dict[int, List[float]] = {}

    for budget_raw in budgets:
        key = int(budget_raw)
        budget_to_seed_acc[key] = []

        for seed in seeds:
            set_seed(seed)
            rng = np.random.RandomState(seed)

            if key < 0:
                tgt_budget_texts = tgt_train_unique_full
            else:
                b = int(min(key, len(tgt_train_unique_full)))
                if b <= 0:
                    tgt_budget_texts = []
                else:
                    # nested subset (deterministic): use first b of a fixed permutation per seed
                    perm = rng.permutation(len(tgt_train_unique_full))
                    pick = perm[:b]
                    tgt_budget_texts = [tgt_train_unique_full[i] for i in pick.tolist()]

            teacher, temperature, style_norm = train_single_run(
                seed=seed,
                model_name=args.model_name,
                src_train_texts=src_train_texts,
                src_train_y=src_train_y,
                src_val_texts=src_val_texts,
                src_val_y=src_val_y,
                tgt_train_texts_budget=tgt_budget_texts,
                args=args,
            )

            # Build tokenizers/loader for predictions (inductive safe)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

            # target-train loader for threshold (only budget texts)
            if len(tgt_budget_texts) > 0:
                ds_thr = LabeledTextDataset(tgt_budget_texts, [0] * len(tgt_budget_texts), style_norm)  # dummy labels
                dl_thr = DataLoader(
                    ds_thr,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=lambda b: collate_labeled(tokenizer, args.max_len, b),
                    drop_last=False,
                )
                probs_thr = predict_probs(teacher, dl_thr, device=device, temperature=temperature)

                # smoothing for tiny budgets
                smooth_w = min(1.0, float(len(tgt_budget_texts)) / max(1.0, float(args.thr_budget_ref)))
                thr = compute_matchprior_threshold(probs_thr, pi_true=float(args.pi_true), smooth_w=smooth_w)
            else:
                thr = 0.5

            # target-test loader (with true labels)
            ds_test = LabeledTextDataset(tgt_test_texts, y_test_internal.tolist(), style_norm)
            dl_test = DataLoader(
                ds_test,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda b: collate_labeled(tokenizer, args.max_len, b),
                drop_last=False,
            )

            probs_test = predict_probs(teacher, dl_test, device=device, temperature=temperature)
            pred_a_internal = pred_internal_argmax(probs_test)
            pred_m_internal = pred_internal_matchprior(probs_test, thr_true=thr)

            acc_argmax = float(accuracy_score(y_test_internal, pred_a_internal))
            acc_match = float(accuracy_score(y_test_internal, pred_m_internal))

            f1_argmax = float(f1_score(y_test_internal, pred_a_internal, average="macro"))
            f1_match = float(f1_score(y_test_internal, pred_m_internal, average="macro"))

            budget_to_seed_acc[key].append(acc_match)

            # source-val acc for logging (needs loader)
            ds_va = LabeledTextDataset(src_val_texts, src_val_y, style_norm)
            dl_va = DataLoader(
                ds_va,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda b: collate_labeled(tokenizer, args.max_len, b),
                drop_last=False,
            )
            src_val_acc = eval_acc(teacher, dl_va, device=device, temperature=1.0)

            print(
                f"[B={len(tgt_budget_texts):>5}] seed={seed} "
                f"acc_match={acc_match:.4f} acc_argmax={acc_argmax:.4f} "
                f"f1_match={f1_match:.4f} thr={thr:.3f} T={temperature:.3f} src_val={src_val_acc:.4f}"
            )

            rec = {
                "budget_used": len(tgt_budget_texts),
                "budget_raw": key,
                "seed": seed,
                "train_unique_used": len(tgt_budget_texts),
                "test_rows": int(len(test_idx)),
                "acc_argmax": acc_argmax,
                "acc_matchprior_inductive": acc_match,
                "f1_argmax_macro": f1_argmax,
                "f1_matchprior_macro": f1_match,
                "thr_true_from_train": float(thr),
                "temperature": float(temperature),
                "src_val_acc": float(src_val_acc),
            }
            records.append(rec)

            if args.save_predictions:
                out_csv = out_dir / f"v13_pred_budget{key}_seed{seed}.csv"
                pd.DataFrame(
                    {
                        "Text": tgt_test_texts,
                        "y_true_internal(0=true,1=false)": y_test_internal,
                        "y_true_covid(1=true,0=false)": internal_to_covid(y_test_internal),
                        "prob_true_internal0": probs_test[:, 0],
                        "prob_false_internal1": probs_test[:, 1],
                        "pred_argmax_internal": pred_a_internal,
                        "pred_matchprior_internal": pred_m_internal,
                        "pred_argmax_covid": internal_to_covid(pred_a_internal),
                        "pred_matchprior_covid": internal_to_covid(pred_m_internal),
                        "thr": thr,
                    }
                ).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # save detailed
    df_res = pd.DataFrame(records)
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(args.results_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved detailed results: {args.results_csv}")

    # summary mean±std per budget
    summary_rows: List[Dict[str, object]] = []
    full_budget = len(tgt_train_unique_full)
    full_key = -1 if -1 in budgets else max([b for b in budgets if b >= 0] + [-1])
    full_accs = budget_to_seed_acc.get(full_key, [])
    full_mean, full_std = mean_std(full_accs)
    target_mean = full_mean - float(args.delta_to_full)

    # sort by used budget
    for budget_raw in budgets:
        key = int(budget_raw)
        accs = budget_to_seed_acc.get(key, [])
        m, s = mean_std(accs)
        used = full_budget if key < 0 else key
        summary_rows.append(
            {
                "budget_used": used,
                "budget_raw": key,
                "mean_acc_matchprior": m,
                "std_acc_matchprior": s,
                "full_mean": full_mean,
                "target_mean(full-delta)": target_mean,
            }
        )

    summary_sorted = sorted(summary_rows, key=lambda d: int(d["budget_used"]))
    best_budget_found = None
    for row in summary_sorted:
        if float(row["mean_acc_matchprior"]) >= target_mean:
            best_budget_found = int(row["budget_used"])
            break

    df_sum = pd.DataFrame(summary_sorted)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved summary: {args.summary_csv}")

    print("\n========== V13 Summary ==========")
    print(f"Full budget (unique texts) = {full_budget}")
    print(f"Full mean±std (matchprior) = {full_mean:.4f} ± {full_std:.4f}")
    print(f"Target threshold (full - delta) = {target_mean:.4f}  (delta={args.delta_to_full})")
    if best_budget_found is None:
        print("[Result] No budget in your sweep reached (full_mean - delta). Try larger budgets or increase delta.")
    else:
        print(f"[Result] Minimum target-train unique Texts to reach current effect ≈ {best_budget_found}")


if __name__ == "__main__":
    # Make tokenizers parallelism warning quiet (optional)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
