# -*- coding: utf-8 -*-
"""
PAS-UniDA++ (Practical & Stable UniDA for NLP) - Single file implementation

Pipeline:
1) Warm-up: DAPT (MLM domain-adaptive pretraining) on target train (unlabeled)
2) Source supervised fine-tuning with early stop on source validation
3) Temperature scaling calibration on source validation
4) Auto+Proxy-safe selection:
   - Energy score -> 1D 2-component GMM (custom EM) -> soft knownness q_i
   - Weighted label shift EM (Saerens/MLLS style) -> class prior weights alpha
   - Search tau to maximize proxy score (q-quality + coverage alignment + prototype agreement)
5) Iterative safe self-training with:
   - pseudo-labeling for known samples
   - uniform (max-entropy) regularization for unknown samples
   - outlier head trained with soft target (1-q_i)
   - proxy-based round selection & early stop

Paths are hardcoded to match user's file structure.
No terminal arguments required.
"""

import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)

# -----------------------------
# 0) Hardcoded file paths
# -----------------------------
SOURCE_TRAIN_PATH = "./sourcedata/source_train.csv"
SOURCE_TEST_PATH = "./sourcedata/source_test.csv"
SOURCE_VAL_PATH = "./sourcedata/source_validation.csv"

TARGET_TRAIN_PATH = "./targetdata/train.csv"
TARGET_TEST_PATH = "./targetdata/test.csv"
TARGET_VAL_PATH = "./targetdata/val.csv"

POLITICS_PATH = "./politics.csv"  # optional OE / disturbance data


# -----------------------------
# 1) Config (edit here if needed)
# -----------------------------
@dataclass
class Config:
    # Model
    model_name_override: Optional[str] = None   # e.g. "bert-base-uncased" / "bert-base-chinese"
    max_length: int = 256

    # Train
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DAPT (MLM)
    enable_dapt: bool = True
    dapt_epochs: int = 1
    dapt_batch_size: int = 16
    dapt_lr: float = 5e-5
    dapt_weight_decay: float = 0.01
    dapt_mlm_prob: float = 0.15
    dapt_max_steps: Optional[int] = None  # set to int to cap steps (faster)

    # Source supervised fine-tune
    src_epochs: int = 4
    src_batch_size: int = 16
    src_lr: float = 2e-5
    src_weight_decay: float = 0.01
    src_warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    early_stop_patience: int = 2

    # Temperature scaling
    enable_temp_scaling: bool = True
    temp_scaling_max_iter: int = 50

    # Auto+Proxy-safe
    tau_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.50, 0.95, 10), 2).tolist())
    min_coverage: float = 0.10
    max_coverage: float = 0.98
    min_selected: int = 64
    gmm_max_iter: int = 80
    gmm_tol: float = 1e-4
    gmm_min_var: float = 1e-4

    # Label shift
    enable_label_shift_auto: bool = True
    label_shift_max_iter: int = 50
    label_shift_tol: float = 1e-6
    alpha_clip: Tuple[float, float] = (0.1, 10.0)

    # Self-training
    selftrain_rounds: int = 3
    st_epochs_per_round: int = 1
    st_batch_size: int = 16
    st_lr: float = 2e-5
    st_weight_decay: float = 0.01
    st_warmup_ratio: float = 0.06
    st_lambda_pseudo: float = 1.0
    st_lambda_unk: float = 0.2
    st_lambda_outlier: float = 0.3

    # Pseudo label safety
    q_min: float = 0.50            # knownness threshold
    margin_min: float = 0.10       # p1 - p2
    pseudo_frac_schedule: Tuple[float, ...] = (0.20, 0.40, 0.60)  # round-wise keep top conf fraction

    # Optional politics OE
    use_politics_oe: bool = True
    politics_oe_weight: float = 0.5

    # Output
    output_dir: str = "./outputs_pas_unida_pp"
    save_best_only: bool = True


CFG = Config()

# Optional manual overrides for columns (if auto-detect fails)
TEXT_COL_OVERRIDE = None   # e.g. "text"
LABEL_COL_OVERRIDE = None  # e.g. "label"

# Optional model override (takes precedence)
MODEL_NAME_OVERRIDE = None  # e.g. "bert-base-uncased"


# -----------------------------
# 2) Utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # Try utf-8, fallback latin1
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def detect_text_and_label_columns(df: pd.DataFrame,
                                  text_override: Optional[str] = None,
                                  label_override: Optional[str] = None) -> Tuple[str, Optional[str]]:
    cols = list(df.columns)

    # Text column
    if text_override is not None:
        if text_override not in cols:
            raise ValueError(f"TEXT_COL_OVERRIDE={text_override} not in columns: {cols}")
        text_col = text_override
    else:
        # Prefer typical names
        for cand in ["text", "sentence", "content", "review", "comment", "abstract", "title", "body"]:
            if cand in cols and df[cand].dtype == object:
                text_col = cand
                break
        else:
            obj_cols = [c for c in cols if df[c].dtype == object]
            if not obj_cols:
                raise ValueError("No object/string column found to use as text.")
            # Choose the object col with largest average length
            lengths = {}
            for c in obj_cols:
                s = df[c].astype(str)
                lengths[c] = float(s.str.len().replace([np.inf, -np.inf], np.nan).fillna(0).mean())
            text_col = max(lengths, key=lengths.get)

    # Label column
    if label_override is not None:
        if label_override not in cols:
            raise ValueError(f"LABEL_COL_OVERRIDE={label_override} not in columns: {cols}")
        label_col = label_override
        return text_col, label_col

    # common label names
    for cand in ["label", "labels", "y", "target", "class", "category"]:
        if cand in cols and cand != text_col:
            return text_col, cand

    # heuristic: small-unique column that is not the text column
    best = None
    for c in cols:
        if c == text_col:
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 50:
            # prefer non-long-text columns
            score = nunique
            if best is None or score < best[0]:
                best = (score, c)

    label_col = best[1] if best is not None else None
    return text_col, label_col


def pick_model_name_by_language(sample_texts: List[str]) -> str:
    # crude detection: if contains CJK char -> use bert-base-chinese else bert-base-uncased
    cjk = 0
    total = 0
    for t in sample_texts[:50]:
        if not isinstance(t, str):
            continue
        total += 1
        for ch in t[:200]:
            if "\u4e00" <= ch <= "\u9fff":
                cjk += 1
                break
    if total > 0 and cjk / total >= 0.2:
        return "bert-base-chinese"
    return "bert-base-uncased"


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# -----------------------------
# 3) Datasets
# -----------------------------
class TokenizedTextDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TokenizedMLMDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}


class TokenizedSelfTrainDataset(Dataset):
    """
    A unified dataset for:
      - source labeled known samples
      - target pseudo-known samples
      - target unknown samples
      - optional politics unknown samples

    Each item contains:
      input_ids, attention_mask
      class_labels (long)          : valid if class_mask=1
      class_mask (float 0/1)
      class_weight (float)
      unk_mask (float 0/1)
      outlier_target (float 0/1)   : 0 known, 1 unknown
      outlier_weight (float)
    """
    def __init__(self,
                 encodings: Dict[str, List[List[int]]],
                 class_labels: List[int],
                 class_mask: List[float],
                 class_weight: List[float],
                 unk_mask: List[float],
                 outlier_target: List[float],
                 outlier_weight: List[float]):
        self.encodings = encodings
        self.class_labels = class_labels
        self.class_mask = class_mask
        self.class_weight = class_weight
        self.unk_mask = unk_mask
        self.outlier_target = outlier_target
        self.outlier_weight = outlier_weight

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["class_labels"] = torch.tensor(self.class_labels[idx], dtype=torch.long)
        item["class_mask"] = torch.tensor(self.class_mask[idx], dtype=torch.float32)
        item["class_weight"] = torch.tensor(self.class_weight[idx], dtype=torch.float32)
        item["unk_mask"] = torch.tensor(self.unk_mask[idx], dtype=torch.float32)
        item["outlier_target"] = torch.tensor(self.outlier_target[idx], dtype=torch.float32)
        item["outlier_weight"] = torch.tensor(self.outlier_weight[idx], dtype=torch.float32)
        return item


# -----------------------------
# 4) Model
# -----------------------------
class PASUniDAPlusPlusModel(nn.Module):
    def __init__(self, base_model_name_or_path: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name_or_path)
        hidden = getattr(self.encoder.config, "hidden_size", 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.outlier_head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]  # CLS

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        outlier_logit = self.outlier_head(pooled).squeeze(-1)
        return {"logits": logits, "outlier_logit": outlier_logit, "features": pooled}


# -----------------------------
# 5) Temperature scaling
# -----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def temperature(self) -> torch.Tensor:
        # positive temperature
        return torch.exp(self.raw).clamp(min=1e-6, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature()


@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(out["logits"].detach().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature_scaler(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, device: str = "cpu") -> TemperatureScaler:
    scaler = TemperatureScaler().to(device)
    logits = logits.to(device)
    labels = labels.to(device)

    nll_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS([scaler.raw], lr=0.5, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


# -----------------------------
# 6) Energy GMM (1D, 2 components) - custom EM (no sklearn)
# -----------------------------
def normal_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    coef = 1.0 / math.sqrt(2.0 * math.pi * var)
    return coef * np.exp(-(x - mean) ** 2 / (2.0 * var))


def fit_gmm_1d_two_comp(x: np.ndarray,
                        max_iter: int = 80,
                        tol: float = 1e-4,
                        min_var: float = 1e-4) -> Dict[str, Any]:
    """
    Fit a 2-component Gaussian mixture to 1D data x using EM.
    Returns means, vars, pi, responsibilities.
    """
    x = x.astype(np.float64)
    x = np.clip(x, np.percentile(x, 0.5), np.percentile(x, 99.5))  # robust trim
    n = x.shape[0]
    if n < 10:
        # Degenerate: return single comp
        m = float(np.mean(x))
        v = float(np.var(x) + min_var)
        resp = np.ones((n, 2), dtype=np.float64) * 0.5
        return {"means": [m, m + 1e-3], "vars": [v, v], "pi": 0.5, "resp": resp, "ll": -1e9}

    # init: use percentiles
    m1 = float(np.percentile(x, 25))
    m2 = float(np.percentile(x, 75))
    v1 = float(np.var(x) + min_var)
    v2 = float(np.var(x) + min_var)
    pi = 0.5

    prev_ll = None
    for _ in range(max_iter):
        # E-step
        p1 = pi * normal_pdf(x, m1, v1)
        p2 = (1.0 - pi) * normal_pdf(x, m2, v2)
        denom = p1 + p2 + 1e-12
        r1 = p1 / denom
        r2 = p2 / denom

        # M-step
        n1 = float(np.sum(r1) + 1e-12)
        n2 = float(np.sum(r2) + 1e-12)
        pi = n1 / (n1 + n2)

        m1 = float(np.sum(r1 * x) / n1)
        m2 = float(np.sum(r2 * x) / n2)

        v1 = float(np.sum(r1 * (x - m1) ** 2) / n1)
        v2 = float(np.sum(r2 * (x - m2) ** 2) / n2)
        v1 = max(v1, min_var)
        v2 = max(v2, min_var)

        # log-likelihood
        ll = float(np.sum(np.log(denom)))
        if prev_ll is not None and abs(ll - prev_ll) < tol * (abs(prev_ll) + 1.0):
            prev_ll = ll
            break
        prev_ll = ll

    resp = np.stack([r1, r2], axis=1)
    return {"means": [m1, m2], "vars": [v1, v2], "pi": pi, "resp": resp, "ll": prev_ll}


def energy_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # energy = -logsumexp(logits)
    return -torch.logsumexp(logits, dim=-1)


def gmm_known_probability_from_energy(energy: np.ndarray, gmm: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Determine known component as the one with LOWER mean energy (more confident).
    Return q_i = P(known | energy), and separation metric.
    """
    means = gmm["means"]
    vars_ = gmm["vars"]
    resp = gmm["resp"]

    # comp with lower mean energy is known
    known_comp = int(np.argmin(means))
    q = resp[:, known_comp].astype(np.float64)

    sep = abs(means[0] - means[1]) / math.sqrt(vars_[0] + vars_[1] + 1e-12)
    return q, float(sep)


# -----------------------------
# 7) Label shift EM (Saerens/MLLS style) with soft known weights q_i
# -----------------------------
def estimate_label_shift_em(p: np.ndarray,
                            q_known: np.ndarray,
                            source_prior: np.ndarray,
                            max_iter: int = 50,
                            tol: float = 1e-6) -> np.ndarray:
    """
    p: (N, K) predicted probabilities from source model (on target samples).
    q_known: (N,) soft weight indicating how likely sample is known (downweight unknown contamination).
    source_prior: (K,) P_s(y)
    Returns w_t: (K,) estimated target prior P_t(y) over known classes.
    """
    N, K = p.shape
    w = source_prior.copy().astype(np.float64)
    w = w / (w.sum() + 1e-12)

    q = q_known.astype(np.float64).clip(0.0, 1.0)
    q_sum = float(q.sum() + 1e-12)

    prev = None
    for _ in range(max_iter):
        # E-step: responsibilities r_ij ∝ w_j p_ij
        denom = (p * w.reshape(1, -1)).sum(axis=1, keepdims=True) + 1e-12
        r = (p * w.reshape(1, -1)) / denom  # (N,K)

        # M-step: w_j = sum_i q_i r_ij / sum_i q_i
        w_new = (q.reshape(-1, 1) * r).sum(axis=0) / q_sum
        w_new = np.clip(w_new, 1e-12, None)
        w_new = w_new / (w_new.sum() + 1e-12)

        diff = float(np.abs(w_new - w).sum())
        w = w_new
        if prev is not None and diff < tol:
            break
        prev = diff

    return w


def apply_label_shift(p: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    p_t(y|x) ∝ alpha_y * p_s(y|x)
    """
    p2 = p * alpha.reshape(1, -1)
    p2 = p2 / (p2.sum(axis=1, keepdims=True) + 1e-12)
    return p2


# -----------------------------
# 8) Prototypes (proxy-safe)
# -----------------------------
@torch.no_grad()
def compute_embeddings_and_logits(model: nn.Module,
                                 dataset: TokenizedTextDataset,
                                 batch_size: int,
                                 device: str) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    feats = []
    logits_list = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        feats.append(out["features"].detach().cpu().numpy())
        logits_list.append(out["logits"].detach().cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    return feats, logits


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt((x * x).sum(axis=axis, keepdims=True) + eps)
    return x / norm


def build_class_prototypes(source_feats: np.ndarray, source_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    prototypes: (K, D) normalized mean feature per class
    """
    D = source_feats.shape[1]
    protos = np.zeros((num_classes, D), dtype=np.float32)
    for c in range(num_classes):
        idx = np.where(source_labels == c)[0]
        if len(idx) == 0:
            continue
        v = source_feats[idx].mean(axis=0)
        protos[c] = v
    protos = l2_normalize(protos, axis=-1)
    return protos


def prototype_agreement_score(target_feats: np.ndarray, target_pred: np.ndarray, protos: np.ndarray) -> float:
    """
    Compute agreement between model predicted class and nearest prototype class.
    """
    if target_feats.shape[0] == 0:
        return 0.0
    tf = l2_normalize(target_feats, axis=-1)
    sims = tf @ protos.T  # (N,K)
    nn_cls = np.argmax(sims, axis=1)
    agree = (nn_cls == target_pred).mean()
    return float(agree)


# -----------------------------
# 9) Training loops
# -----------------------------
def train_dapt_mlm(model_name: str,
                   tokenizer: AutoTokenizer,
                   texts: List[str],
                   cfg: Config) -> str:
    """
    Train MLM on unlabeled target texts. Save base encoder to output_dir/dapt_encoder.
    Return saved path.
    """
    if not cfg.enable_dapt:
        return model_name

    print("\n[Phase 1] DAPT (MLM) warm-up on target train...")
    safe_mkdir(cfg.output_dir)
    dapt_dir = os.path.join(cfg.output_dir, "dapt_encoder")
    safe_mkdir(dapt_dir)

    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    mlm_model.to(cfg.device)

    # tokenize
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
        return_special_tokens_mask=True,
    )
    mlm_dataset = TokenizedMLMDataset(enc)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=cfg.dapt_mlm_prob)
    loader = DataLoader(mlm_dataset, batch_size=cfg.dapt_batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=cfg.dapt_lr, weight_decay=cfg.dapt_weight_decay)

    total_steps = len(loader) * cfg.dapt_epochs
    if cfg.dapt_max_steps is not None:
        total_steps = min(total_steps, cfg.dapt_max_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )

    mlm_model.train()
    step = 0
    for epoch in range(cfg.dapt_epochs):
        for batch in loader:
            if cfg.dapt_max_steps is not None and step >= cfg.dapt_max_steps:
                break

            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            out = mlm_model(**batch)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"  [DAPT] epoch={epoch+1}/{cfg.dapt_epochs} step={step}/{total_steps} loss={loss.item():.4f}")
            step += 1
        if cfg.dapt_max_steps is not None and step >= cfg.dapt_max_steps:
            break

    # Save only encoder/base model
    base = mlm_model.base_model
    base.save_pretrained(dapt_dir)
    tokenizer.save_pretrained(dapt_dir)
    print(f"  [DAPT] saved encoder to: {dapt_dir}")
    return dapt_dir


def train_source_supervised(model: PASUniDAPlusPlusModel,
                            train_ds: TokenizedTextDataset,
                            val_ds: TokenizedTextDataset,
                            cfg: Config) -> PASUniDAPlusPlusModel:
    """
    Supervised training on source labeled data, early stop on source validation accuracy.
    """
    print("\n[Phase 1b] Source supervised fine-tuning...")
    model.to(cfg.device)

    train_loader = DataLoader(train_ds, batch_size=cfg.src_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.src_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.src_lr, weight_decay=cfg.src_weight_decay)
    total_steps = len(train_loader) * cfg.src_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(cfg.src_warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    best_val = -1.0
    best_state = None
    patience = 0

    for epoch in range(cfg.src_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"]
            outlier_logit = out["outlier_logit"]

            # source classification loss
            ce = F.cross_entropy(logits, labels)

            # outlier head: source is known (0)
            outlier_target = torch.zeros_like(outlier_logit)
            bce = F.binary_cross_entropy_with_logits(outlier_logit, outlier_target)

            loss = ce + 0.05 * bce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        val_acc = evaluate_closed_set_accuracy(model, val_loader, cfg.device)
        print(f"  [SRC] epoch={epoch+1}/{cfg.src_epochs} train_loss={np.mean(losses):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val + 1e-5:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("  [SRC] early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_closed_set_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = out["logits"].argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(1, total)


# -----------------------------
# 10) Auto selection: mode + tau
# -----------------------------
def auto_select_mode_and_tau(model: PASUniDAPlusPlusModel,
                             temp_scaler: Optional[TemperatureScaler],
                             source_feats: np.ndarray,
                             source_labels: np.ndarray,
                             source_prior: np.ndarray,
                             target_ds: TokenizedTextDataset,
                             cfg: Config) -> Dict[str, Any]:
    """
    Auto select:
      mode in {none, em}
      tau in grid

    Proxy score uses:
      - avg q_known among selected samples (higher better)
      - coverage alignment to mean(q_known) (smaller mismatch better)
      - prototype agreement (higher better)
      - safety constraints (coverage bounds, min selected)

    Returns dict with best mode, tau, alpha, gmm params, q stats, sep.
    """
    print("\n[Phase 2] Auto + Proxy-safe selection (mode, tau)...")
    # prototypes
    K = int(source_prior.shape[0])
    protos = build_class_prototypes(source_feats, source_labels, K)

    # target embeddings & logits
    tgt_feats, tgt_logits = compute_embeddings_and_logits(model, target_ds, batch_size=cfg.src_batch_size, device=cfg.device)
    tgt_logits_t = torch.tensor(tgt_logits, dtype=torch.float32)

    if temp_scaler is not None:
        with torch.no_grad():
            tgt_logits_scaled = temp_scaler(tgt_logits_t).cpu().numpy()
    else:
        tgt_logits_scaled = tgt_logits

    p = softmax_np(tgt_logits_scaled, axis=-1)  # (N,K)
    pred_cls = np.argmax(p, axis=1)

    # energy + gmm -> q
    energy = energy_score_from_logits(torch.tensor(tgt_logits_scaled)).cpu().numpy()
    gmm = fit_gmm_1d_two_comp(energy, max_iter=cfg.gmm_max_iter, tol=cfg.gmm_tol, min_var=cfg.gmm_min_var)
    q_known, sep = gmm_known_probability_from_energy(energy, gmm)
    pi_known = float(np.mean(q_known))

    print(f"  [Auto] energy_gmm_sep={sep:.3f}  estimated_known_fraction(pi_known)={pi_known:.3f}")

    # candidates
    mode_candidates = ["none"]
    if cfg.enable_label_shift_auto:
        mode_candidates.append("em")

    best = {"proxy": -1e18}

    for mode in mode_candidates:
        if mode == "em":
            w_t = estimate_label_shift_em(p, q_known, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol)
            alpha = (w_t / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])
            p_adj = apply_label_shift(p, alpha)
        else:
            alpha = np.ones((K,), dtype=np.float64)
            p_adj = p

        conf = np.max(p_adj, axis=1)
        # margin
        top2 = np.partition(p_adj, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]  # top1 - top2 (since top2 returns sorted? not guaranteed)
        # ensure correct margin
        # safer margin compute:
        sorted_probs = np.sort(p_adj, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        pred_adj = np.argmax(p_adj, axis=1)

        for tau in cfg.tau_grid:
            selected = conf >= tau
            cov = float(np.mean(selected))
            sel_count = int(selected.sum())

            if cov < cfg.min_coverage or cov > cfg.max_coverage:
                continue
            if sel_count < cfg.min_selected:
                continue

            q_quality = float(np.mean(q_known[selected]))
            mismatch = abs(cov - pi_known)

            agree = prototype_agreement_score(tgt_feats[selected], pred_adj[selected], protos)

            # proxy score: quality + proto agreement - mismatch penalty
            proxy = q_quality + 0.30 * agree - 1.00 * mismatch

            # additional safety: if sep is low, penalize too aggressive tau (low tau)
            if sep < 0.8 and tau < 0.70:
                proxy -= 0.15

            if proxy > best["proxy"]:
                best = {
                    "proxy": proxy,
                    "mode": mode,
                    "tau": float(tau),
                    "alpha": alpha.astype(np.float64),
                    "w_t": None if mode == "none" else w_t.astype(np.float64),
                    "q_known": q_known.astype(np.float64),
                    "pi_known": pi_known,
                    "gmm": gmm,
                    "sep": sep,
                    "cov": cov,
                    "sel_count": sel_count,
                    "q_quality": q_quality,
                    "proto_agree": agree,
                }

    if best["proxy"] < -1e10:
        # fallback (very rare)
        best = {
            "proxy": -1e10,
            "mode": "none",
            "tau": 0.90,
            "alpha": np.ones((K,), dtype=np.float64),
            "w_t": None,
            "q_known": q_known.astype(np.float64),
            "pi_known": pi_known,
            "gmm": gmm,
            "sep": sep,
            "cov": float(np.mean(p.max(axis=1) >= 0.90)),
            "sel_count": int((p.max(axis=1) >= 0.90).sum()),
            "q_quality": float(np.mean(q_known[p.max(axis=1) >= 0.90])) if int((p.max(axis=1) >= 0.90).sum()) > 0 else 0.0,
            "proto_agree": 0.0,
        }

    print(
        f"  [Auto] best: mode={best['mode']} tau={best['tau']:.2f} "
        f"proxy={best['proxy']:.4f} cov={best['cov']:.3f} "
        f"q_quality={best['q_quality']:.3f} proto_agree={best['proto_agree']:.3f}"
    )
    return best


# -----------------------------
# 11) Build self-training dataset per round
# -----------------------------
def build_selftrain_dataset(tokenizer: AutoTokenizer,
                            cfg: Config,
                            source_texts: List[str],
                            source_labels: List[int],
                            target_texts: List[str],
                            p_adj: np.ndarray,
                            q_known: np.ndarray,
                            tau: float,
                            alpha: np.ndarray,
                            round_idx: int,
                            politics_texts: Optional[List[str]] = None) -> TokenizedSelfTrainDataset:
    """
    Create a combined dataset for one self-training round.
    - Source labeled samples: class loss + outlier known
    - Target pseudo-known: class loss weighted
    - Target unknown: uniform loss + outlier unknown (soft)
    - Optional politics: uniform loss + outlier unknown (hard)
    """
    K = p_adj.shape[1]
    conf = p_adj.max(axis=1)
    sorted_probs = np.sort(p_adj, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    pred = np.argmax(p_adj, axis=1)

    # safety masks
    known_candidate = (conf >= tau) & (q_known >= cfg.q_min) & (margin >= cfg.margin_min)
    idx_known = np.where(known_candidate)[0]
    if len(idx_known) > 0:
        # keep top fraction by confidence (curriculum)
        frac = cfg.pseudo_frac_schedule[min(round_idx, len(cfg.pseudo_frac_schedule) - 1)]
        k_keep = max(1, int(len(idx_known) * frac))
        order = np.argsort(-conf[idx_known])
        keep = idx_known[order[:k_keep]]
        pseudo_known_mask = np.zeros_like(known_candidate, dtype=bool)
        pseudo_known_mask[keep] = True
    else:
        pseudo_known_mask = np.zeros_like(known_candidate, dtype=bool)

    unknown_mask = ~pseudo_known_mask  # includes low-conf and low-q samples

    # Build combined lists
    texts_all: List[str] = []
    enc_source = tokenizer(source_texts, truncation=True, padding="max_length", max_length=cfg.max_length)
    enc_target = tokenizer(target_texts, truncation=True, padding="max_length", max_length=cfg.max_length)

    # source block
    texts_all.extend(source_texts)
    src_n = len(source_texts)

    # target block
    texts_all.extend(target_texts)
    tgt_n = len(target_texts)

    # optional politics block
    pol_n = 0
    enc_pol = None
    if politics_texts is not None and len(politics_texts) > 0:
        enc_pol = tokenizer(politics_texts, truncation=True, padding="max_length", max_length=cfg.max_length)
        texts_all.extend(politics_texts)
        pol_n = len(politics_texts)

    # Merge encodings (since we used fixed max_length)
    def merge_enc(enc1, enc2, enc3=None):
        out = {}
        for k in enc1.keys():
            out[k] = enc1[k] + enc2[k] + (enc3[k] if enc3 is not None else [])
        return out

    merged_enc = merge_enc(enc_source, enc_target, enc_pol)

    # Build supervision arrays aligned with merged_enc order:
    # [0..src_n-1] source
    # [src_n..src_n+tgt_n-1] target
    # [src_n+tgt_n..] politics

    class_labels: List[int] = []
    class_mask: List[float] = []
    class_weight: List[float] = []
    unk_mask: List[float] = []
    outlier_target: List[float] = []
    outlier_weight: List[float] = []

    # Source: always known + true label
    for i in range(src_n):
        y = int(source_labels[i])
        class_labels.append(y)
        class_mask.append(1.0)
        class_weight.append(1.0)
        unk_mask.append(0.0)
        outlier_target.append(0.0)
        outlier_weight.append(1.0)

    # Target: pseudo-known vs unknown
    for i in range(tgt_n):
        if pseudo_known_mask[i]:
            y = int(pred[i])
            # weight uses q * conf * alpha_y (label shift)
            w = float(q_known[i] * conf[i] * float(alpha[y]))
            class_labels.append(y)
            class_mask.append(1.0)
            class_weight.append(max(1e-6, w))
            unk_mask.append(0.0)
            outlier_target.append(0.0)
            outlier_weight.append(1.0)
        else:
            # unknown: no class supervision, but unk regularization + outlier=1
            class_labels.append(0)     # dummy
            class_mask.append(0.0)
            class_weight.append(0.0)
            unk_mask.append(1.0)
            outlier_target.append(1.0)
            # use soft weight based on (1-q)
            outlier_weight.append(float(max(0.1, 1.0 - q_known[i])))

    # Politics: treat as unknown (outlier exposure), optional
    for _ in range(pol_n):
        class_labels.append(0)
        class_mask.append(0.0)
        class_weight.append(0.0)
        unk_mask.append(1.0)
        outlier_target.append(1.0)
        outlier_weight.append(float(cfg.politics_oe_weight))

    ds = TokenizedSelfTrainDataset(
        encodings=merged_enc,
        class_labels=class_labels,
        class_mask=class_mask,
        class_weight=class_weight,
        unk_mask=unk_mask,
        outlier_target=outlier_target,
        outlier_weight=outlier_weight,
    )

    info = {
        "src_n": src_n,
        "tgt_n": tgt_n,
        "pol_n": pol_n,
        "pseudo_known_n": int(pseudo_known_mask.sum()),
        "unknown_n": int(unknown_mask.sum()),
    }
    return ds, info


def train_one_selftrain_round(model: PASUniDAPlusPlusModel,
                              train_ds: TokenizedSelfTrainDataset,
                              cfg: Config,
                              round_idx: int) -> PASUniDAPlusPlusModel:
    """
    Train model for one round on combined self-training dataset.
    """
    print(f"\n[Phase 3] Self-training round {round_idx+1}/{cfg.selftrain_rounds}...")
    model.to(cfg.device)
    model.train()

    loader = DataLoader(train_ds, batch_size=cfg.st_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.st_lr, weight_decay=cfg.st_weight_decay)
    total_steps = len(loader) * cfg.st_epochs_per_round
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(cfg.st_warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    for epoch in range(cfg.st_epochs_per_round):
        losses = []
        for batch in loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)

            class_labels = batch["class_labels"].to(cfg.device)
            class_mask = batch["class_mask"].to(cfg.device)
            class_weight = batch["class_weight"].to(cfg.device)

            unk_mask = batch["unk_mask"].to(cfg.device)
            outlier_target = batch["outlier_target"].to(cfg.device)
            outlier_weight = batch["outlier_weight"].to(cfg.device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"]
            outlier_logit = out["outlier_logit"]

            # classification loss for (source + pseudo-known)
            ce_per = F.cross_entropy(logits, class_labels, reduction="none")
            ce = (ce_per * class_mask * class_weight).sum() / (class_mask * class_weight).sum().clamp(min=1.0)

            # unknown uniform loss (maximize entropy): cross-entropy to uniform
            logp = F.log_softmax(logits, dim=-1)
            uniform_ce = (-logp.mean(dim=-1))  # (B,)
            unk = (uniform_ce * unk_mask * outlier_weight).sum() / (unk_mask * outlier_weight).sum().clamp(min=1.0)

            # outlier head loss
            bce_per = F.binary_cross_entropy_with_logits(outlier_logit, outlier_target, reduction="none")
            bce = (bce_per * outlier_weight).mean()

            loss = cfg.st_lambda_pseudo * ce + cfg.st_lambda_unk * unk + cfg.st_lambda_outlier * bce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"  [ST] epoch={epoch+1}/{cfg.st_epochs_per_round} loss={np.mean(losses):.4f}")

    return model


# -----------------------------
# 12) Unlabeled proxy score for choosing best round
# -----------------------------
def proxy_score_on_target(model: PASUniDAPlusPlusModel,
                          temp_scaler: Optional[TemperatureScaler],
                          source_feats: np.ndarray,
                          source_labels: np.ndarray,
                          source_prior: np.ndarray,
                          target_ds: TokenizedTextDataset,
                          mode: str,
                          tau: float,
                          cfg: Config) -> Dict[str, float]:
    """
    Compute proxy score on unlabeled target val to select best round.
    """
    K = int(source_prior.shape[0])
    protos = build_class_prototypes(source_feats, source_labels, K)

    tgt_feats, tgt_logits = compute_embeddings_and_logits(model, target_ds, batch_size=cfg.src_batch_size, device=cfg.device)
    tgt_logits_t = torch.tensor(tgt_logits, dtype=torch.float32)

    if temp_scaler is not None:
        with torch.no_grad():
            tgt_logits_scaled = temp_scaler(tgt_logits_t).cpu().numpy()
    else:
        tgt_logits_scaled = tgt_logits

    p = softmax_np(tgt_logits_scaled, axis=-1)
    energy = energy_score_from_logits(torch.tensor(tgt_logits_scaled)).cpu().numpy()
    gmm = fit_gmm_1d_two_comp(energy, max_iter=cfg.gmm_max_iter, tol=cfg.gmm_tol, min_var=cfg.gmm_min_var)
    q_known, sep = gmm_known_probability_from_energy(energy, gmm)
    pi_known = float(np.mean(q_known))

    if mode == "em":
        w_t = estimate_label_shift_em(p, q_known, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol)
        alpha = (w_t / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])
        p_adj = apply_label_shift(p, alpha)
    else:
        p_adj = p

    conf = p_adj.max(axis=1)
    selected = conf >= tau
    cov = float(np.mean(selected))
    sel_count = int(selected.sum())
    if sel_count > 0:
        q_quality = float(np.mean(q_known[selected]))
        pred = np.argmax(p_adj, axis=1)
        agree = prototype_agreement_score(tgt_feats[selected], pred[selected], protos)
    else:
        q_quality = 0.0
        agree = 0.0

    mismatch = abs(cov - pi_known)
    proxy = q_quality + 0.30 * agree - 1.00 * mismatch
    if sep < 0.8 and tau < 0.70:
        proxy -= 0.15

    return {
        "proxy": float(proxy),
        "cov": float(cov),
        "sel_count": float(sel_count),
        "q_quality": float(q_quality),
        "proto_agree": float(agree),
        "sep": float(sep),
        "pi_known": float(pi_known),
    }


# -----------------------------
# 13) Evaluation (if labels exist)
# -----------------------------
def compute_basic_metrics(y_true: List[int], y_pred: List[int], num_classes: int, unknown_id: int) -> Dict[str, float]:
    """
    Macro-F1 over known+unknown, plus known-only accuracy.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # known mask
    known_mask = y_true != unknown_id
    known_acc = float((y_pred[known_mask] == y_true[known_mask]).mean()) if known_mask.any() else 0.0

    # macro f1 over all (including unknown)
    C = num_classes + 1  # include unknown
    f1s = []
    for c in range(C):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    return {"known_acc": known_acc, "macro_f1_including_unknown": macro_f1}


@torch.no_grad()
def predict_with_rejection(model: PASUniDAPlusPlusModel,
                           temp_scaler: Optional[TemperatureScaler],
                           dataset: TokenizedTextDataset,
                           alpha: np.ndarray,
                           tau: float,
                           device: str) -> Tuple[List[int], np.ndarray]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    K = int(alpha.shape[0])
    preds = []
    maxps = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]

        if temp_scaler is not None:
            logits = temp_scaler(logits)

        p = F.softmax(logits, dim=-1).detach().cpu().numpy()  # (B,K)
        p_adj = apply_label_shift(p, alpha)
        conf = p_adj.max(axis=1)
        cls = p_adj.argmax(axis=1)

        # unknown id = K
        pred = []
        for i in range(len(cls)):
            if conf[i] < tau:
                pred.append(K)
            else:
                pred.append(int(cls[i]))
        preds.extend(pred)
        maxps.extend(conf.tolist())
    return preds, np.array(maxps, dtype=np.float64)


# -----------------------------
# 14) Main
# -----------------------------
def main():
    seed_everything(CFG.seed)
    safe_mkdir(CFG.output_dir)

    # Load CSVs
    src_train_df = read_csv_safely(SOURCE_TRAIN_PATH)
    src_val_df = read_csv_safely(SOURCE_VAL_PATH)
    src_test_df = read_csv_safely(SOURCE_TEST_PATH)

    tgt_train_df = read_csv_safely(TARGET_TRAIN_PATH)
    tgt_val_df = read_csv_safely(TARGET_VAL_PATH)
    tgt_test_df = read_csv_safely(TARGET_TEST_PATH)

    pol_df = None
    if CFG.use_politics_oe and os.path.exists(POLITICS_PATH):
        pol_df = read_csv_safely(POLITICS_PATH)

    # Detect columns (source is labeled)
    src_text_col, src_label_col = detect_text_and_label_columns(
        src_train_df, text_override=TEXT_COL_OVERRIDE, label_override=LABEL_COL_OVERRIDE
    )
    print(f"[Data] source text_col='{src_text_col}', label_col='{src_label_col}'")

    # Target text col (unlabeled)
    tgt_text_col, tgt_label_col_guess = detect_text_and_label_columns(
        tgt_train_df, text_override=TEXT_COL_OVERRIDE, label_override=None
    )
    print(f"[Data] target text_col='{tgt_text_col}' (label col guessed: {tgt_label_col_guess})")

    pol_text_col = None
    if pol_df is not None:
        pol_text_col, _ = detect_text_and_label_columns(pol_df, text_override=TEXT_COL_OVERRIDE, label_override=None)
        print(f"[Data] politics text_col='{pol_text_col}'")

    # Extract texts & labels
    def clean_text_list(series: pd.Series) -> List[str]:
        s = series.fillna("").astype(str).tolist()
        return [x if isinstance(x, str) else str(x) for x in s]

    src_train_texts = clean_text_list(src_train_df[src_text_col])
    src_val_texts = clean_text_list(src_val_df[src_text_col])
    src_test_texts = clean_text_list(src_test_df[src_text_col])

    if src_label_col is None:
        raise ValueError("Source label column not found. Please set LABEL_COL_OVERRIDE.")
    src_train_labels_raw = src_train_df[src_label_col].tolist()
    src_val_labels_raw = src_val_df[src_label_col].tolist()
    src_test_labels_raw = src_test_df[src_label_col].tolist()

    # Encode labels based on source
    unique_labels = sorted(list(pd.Series(src_train_labels_raw).dropna().unique()))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    K = len(unique_labels)
    print(f"[Data] num_known_classes(K)={K}  labels={unique_labels}")

    def map_labels(raw_list: List[Any]) -> List[int]:
        out = []
        for r in raw_list:
            if r in label2id:
                out.append(label2id[r])
            else:
                # try cast
                try:
                    rr = int(r)
                    # if source label values are ints but stored differently
                    if rr in label2id:
                        out.append(label2id[rr])
                    else:
                        # unknown -> -1 (shouldn't happen for source)
                        out.append(-1)
                except Exception:
                    out.append(-1)
        return out

    src_train_labels = map_labels(src_train_labels_raw)
    src_val_labels = map_labels(src_val_labels_raw)
    src_test_labels = map_labels(src_test_labels_raw)

    # Source prior
    src_counts = np.bincount(np.array(src_train_labels, dtype=int), minlength=K).astype(np.float64)
    source_prior = src_counts / (src_counts.sum() + 1e-12)

    # Determine model name
    model_name = MODEL_NAME_OVERRIDE or CFG.model_name_override
    if model_name is None:
        model_name = pick_model_name_by_language(src_train_texts[:50] + tgt_train_texts[:50])
    print(f"[Model] base model = {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Phase 1: DAPT
    dapt_path = train_dapt_mlm(model_name=model_name, tokenizer=tokenizer, texts=tgt_train_texts, cfg=CFG)

    # Build tokenized datasets (fixed max_length for simplicity)
    src_train_enc = tokenizer(src_train_texts, truncation=True, padding="max_length", max_length=CFG.max_length)
    src_val_enc = tokenizer(src_val_texts, truncation=True, padding="max_length", max_length=CFG.max_length)
    src_test_enc = tokenizer(src_test_texts, truncation=True, padding="max_length", max_length=CFG.max_length)

    tgt_train_enc = tokenizer(clean_text_list(tgt_train_df[tgt_text_col]), truncation=True, padding="max_length", max_length=CFG.max_length)
    tgt_val_enc = tokenizer(clean_text_list(tgt_val_df[tgt_text_col]), truncation=True, padding="max_length", max_length=CFG.max_length)
    tgt_test_enc = tokenizer(clean_text_list(tgt_test_df[tgt_text_col]), truncation=True, padding="max_length", max_length=CFG.max_length)

    src_train_ds = TokenizedTextDataset(src_train_enc, src_train_labels)
    src_val_ds = TokenizedTextDataset(src_val_enc, src_val_labels)
    src_test_ds = TokenizedTextDataset(src_test_enc, src_test_labels)

    tgt_train_ds = TokenizedTextDataset(tgt_train_enc, labels=None)
    tgt_val_ds = TokenizedTextDataset(tgt_val_enc, labels=None)
    tgt_test_ds = TokenizedTextDataset(tgt_test_enc, labels=None)

    # Model init from DAPT encoder (or base model)
    model = PASUniDAPlusPlusModel(base_model_name_or_path=dapt_path, num_classes=K, dropout=0.1)

    # Phase 1b: source supervised
    model = train_source_supervised(model, src_train_ds, src_val_ds, CFG)

    # Evaluate on source test (closed-set)
    src_test_loader = DataLoader(src_test_ds, batch_size=CFG.src_batch_size, shuffle=False)
    src_test_acc = evaluate_closed_set_accuracy(model, src_test_loader, CFG.device)
    print(f"\n[Check] Source test closed-set acc = {src_test_acc:.4f}")

    # Temperature scaling (optional)
    temp_scaler = None
    if CFG.enable_temp_scaling:
        print("\n[Calib] Temperature scaling on source validation...")
        src_val_loader = DataLoader(src_val_ds, batch_size=CFG.src_batch_size, shuffle=False)
        logits_val, labels_val = collect_logits_labels(model, src_val_loader, CFG.device)
        temp_scaler = fit_temperature_scaler(logits_val, labels_val, max_iter=CFG.temp_scaling_max_iter, device=CFG.device)
        print(f"  [Calib] learned temperature = {temp_scaler.temperature().item():.4f}")

    # Precompute source feats/prototypes base
    src_feats, _ = compute_embeddings_and_logits(model, src_train_ds, batch_size=CFG.src_batch_size, device=CFG.device)
    src_labels_np = np.array(src_train_labels, dtype=int)

    # Phase 2: Auto select mode & tau using target train
    auto = auto_select_mode_and_tau(
        model=model,
        temp_scaler=temp_scaler,
        source_feats=src_feats,
        source_labels=src_labels_np,
        source_prior=source_prior,
        target_ds=tgt_train_ds,
        cfg=CFG,
    )
    mode = auto["mode"]
    tau = auto["tau"]

    # Self-training rounds: keep best by proxy on target val (unlabeled)
    best_round_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_proxy = -1e18
    best_info = None

    # Prepare politics texts if needed
    politics_texts = None
    if CFG.use_politics_oe and pol_df is not None:
        politics_texts = clean_text_list(pol_df[pol_text_col])

    # We'll re-estimate label shift alpha each round (if mode == em)
    alpha = auto["alpha"]

    for r in range(CFG.selftrain_rounds):
        # compute p_adj & q_known on target train with current model
        tgt_feats_r, tgt_logits_r = compute_embeddings_and_logits(model, tgt_train_ds, batch_size=CFG.src_batch_size, device=CFG.device)
        tgt_logits_t = torch.tensor(tgt_logits_r, dtype=torch.float32)

        if temp_scaler is not None:
            with torch.no_grad():
                tgt_logits_scaled = temp_scaler(tgt_logits_t).cpu().numpy()
        else:
            tgt_logits_scaled = tgt_logits_r

        p = softmax_np(tgt_logits_scaled, axis=-1)
        energy = energy_score_from_logits(torch.tensor(tgt_logits_scaled)).cpu().numpy()
        gmm = fit_gmm_1d_two_comp(energy, max_iter=CFG.gmm_max_iter, tol=CFG.gmm_tol, min_var=CFG.gmm_min_var)
        q_known, sep = gmm_known_probability_from_energy(energy, gmm)

        if mode == "em":
            w_t = estimate_label_shift_em(p, q_known, source_prior, max_iter=CFG.label_shift_max_iter, tol=CFG.label_shift_tol)
            alpha = (w_t / (source_prior + 1e-12)).clip(CFG.alpha_clip[0], CFG.alpha_clip[1])
            p_adj = apply_label_shift(p, alpha)
        else:
            alpha = np.ones((K,), dtype=np.float64)
            p_adj = p

        # build self-train dataset
        st_ds, st_info = build_selftrain_dataset(
            tokenizer=tokenizer,
            cfg=CFG,
            source_texts=src_train_texts,
            source_labels=src_train_labels,
            target_texts=clean_text_list(tgt_train_df[tgt_text_col]),
            p_adj=p_adj,
            q_known=q_known,
            tau=tau,
            alpha=alpha,
            round_idx=r,
            politics_texts=politics_texts if politics_texts is not None else None,
        )
        print(f"  [ST data] {st_info}")

        # train one round
        model = train_one_selftrain_round(model, st_ds, CFG, round_idx=r)

        # proxy on target val
        proxy = proxy_score_on_target(
            model=model,
            temp_scaler=temp_scaler,
            source_feats=src_feats,
            source_labels=src_labels_np,
            source_prior=source_prior,
            target_ds=tgt_val_ds,
            mode=mode,
            tau=tau,
            cfg=CFG,
        )
        print(f"  [Proxy@val] {proxy}")

        if proxy["proxy"] > best_proxy:
            best_proxy = proxy["proxy"]
            best_round_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_info = {"round": r, "proxy": proxy}

    # load best
    model.load_state_dict(best_round_state)
    print(f"\n[Select] best_round={best_info['round'] if best_info else 'N/A'} best_proxy={best_proxy:.4f}")

    # Save final model
    safe_mkdir(CFG.output_dir)
    final_dir = os.path.join(CFG.output_dir, "final_model")
    safe_mkdir(final_dir)
    torch.save(model.state_dict(), os.path.join(final_dir, "model_state.pt"))
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"mode": mode, "tau": tau, "alpha": alpha.tolist(), "labels": unique_labels, "id2label": id2label, "best_proxy": best_proxy, "best_info": best_info},
            f, ensure_ascii=False, indent=2
        )
    print(f"[Save] saved final model to: {final_dir}")

    # -------------------------
    # Evaluate on target test IF labels exist
    # -------------------------
    # Try to detect target test label column
    _, tgt_test_label_col = detect_text_and_label_columns(tgt_test_df, text_override=tgt_text_col, label_override=None)
    if tgt_test_label_col is not None and tgt_test_label_col in tgt_test_df.columns:
        # Build y_true with mapping: if label not in source -> unknown_id
        raw = tgt_test_df[tgt_test_label_col].tolist()
        unknown_id = K

        # Build set of known raw labels from source
        known_raw_set = set(unique_labels)

        y_true = []
        for rlab in raw:
            if rlab in label2id:
                y_true.append(label2id[rlab])
            else:
                # try cast to int if source labels are ints
                mapped = None
                try:
                    rr = int(rlab)
                    if rr in label2id:
                        mapped = label2id[rr]
                except Exception:
                    mapped = None
                if mapped is None:
                    y_true.append(unknown_id)
                else:
                    y_true.append(mapped)

        # Predict with rejection
        preds, maxps = predict_with_rejection(
            model=model,
            temp_scaler=temp_scaler,
            dataset=tgt_test_ds,
            alpha=alpha if mode == "em" else np.ones((K,), dtype=np.float64),
            tau=tau,
            device=CFG.device,
        )

        metrics = compute_basic_metrics(y_true, preds, num_classes=K, unknown_id=unknown_id)
        print("\n[Eval] Target test (if labels exist):")
        print(f"  label_col='{tgt_test_label_col}'  mode={mode}  tau={tau:.2f}")
        print(f"  known_acc={metrics['known_acc']:.4f}  macro_f1(incl_unknown)={metrics['macro_f1_including_unknown']:.4f}")
    else:
        print("\n[Eval] Target test: label column not found (or not provided). Skipping supervised evaluation.")
        print("      (This is OK for pure unlabeled target protocol.)")


if __name__ == "__main__":
    main()
