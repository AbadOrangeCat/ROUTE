
# -*- coding: utf-8 -*-
"""
PAS-UniDA++ v2 (Fixed & Safer) + Strong Baseline Comparisons

This script:
- Reads data directly from the provided file structure (no terminal args):
  ./sourcedata/source_train.csv
  ./sourcedata/source_validation.csv
  ./sourcedata/source_test.csv
  ./targetdata/train.csv
  ./targetdata/val.csv
  ./targetdata/test.csv
  ./politics.csv (optional)

- Runs multiple methods and prints a comparison table:
  1) SourceOnly
  2) DAPT+Source
  3) DAPT+Source + LabelShift-EM (inference correction)
  4) DAPT+EntropyMin (InfoMax)
  5) DAPT+SelfTraining (closed-set)
  6) DANN (Domain Adversarial)
  7) PAS-UniDA++ (ours, improved: 3-bucket target handling + decoupled tau)

Important design fixes vs previous version:
- 3-bucket target partition: pseudo-known / pseudo-unknown (strong evidence) / ignore (no unknown loss)
- Decouple tau_pseudo (for pseudo-label selection) and tau_reject (for inference rejection)
- Optional OE (politics) used primarily to train outlier head, not to force target to unknown
- Proxy score penalizes collapse (too low coverage), preventing "unknown collapse"

Notes:
- This is an UNSUPERVISED target adaptation pipeline: it never uses target labels for training/selection.
- Target labels (if present) are used ONLY for final evaluation/comparison.

Author: ChatGPT (GPT-5.2 Pro)
"""

import os
import math
import json
import random
import hashlib
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)

# ============================================================
# 0) Paths (script-dir first, then CWD)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(rel_path: str) -> str:
    p1 = os.path.abspath(os.path.join(BASE_DIR, rel_path))
    if os.path.exists(p1):
        return p1
    return os.path.abspath(rel_path)


SOURCE_TRAIN_PATH = resolve_path("sourcedata/source_train.csv")
SOURCE_VAL_PATH = resolve_path("sourcedata/source_validation.csv")
SOURCE_TEST_PATH = resolve_path("sourcedata/source_test.csv")

TARGET_TRAIN_PATH = resolve_path("targetdata/train.csv")
TARGET_VAL_PATH = resolve_path("targetdata/val.csv")
TARGET_TEST_PATH = resolve_path("targetdata/test.csv")

POLITICS_PATH = resolve_path("politics.csv")


# ============================================================
# 1) Config
# ============================================================
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # Repro
    seed: int = 42
    device: str = pick_device()

    # Text/label columns override (optional)
    text_col_override: Optional[str] = None
    label_col_override: Optional[str] = None

    # Model
    model_name_override: Optional[str] = None
    max_length: int = 256

    # Output
    output_dir: str = os.path.join(BASE_DIR, "outputs_pas_unida_pp_multiseed")

    # Common training
    grad_clip: float = 1.0

    # DAPT (MLM)
    enable_dapt: bool = True
    dapt_epochs: int = 1
    dapt_batch_size: int = 16
    dapt_lr: float = 5e-5
    dapt_weight_decay: float = 0.01
    dapt_mlm_prob: float = 0.15
    dapt_max_steps: Optional[int] = None  # cap steps for speed

    # Supervised source fine-tune
    src_epochs: int = 4
    src_batch_size: int = 16
    src_lr: float = 2e-5
    src_weight_decay: float = 0.01
    src_warmup_ratio: float = 0.06
    early_stop_patience: int = 2

    # OE (politics) used to train outlier head
    use_politics_oe: bool = True
    politics_max_samples: int = 5000
    lambda_out_src: float = 0.05
    lambda_oe_out: float = 0.30
    lambda_oe_unif: float = 0.05

    # Temperature scaling
    enable_temp_scaling: bool = True
    temp_scaling_max_iter: int = 50

    # Auto selection grid
    tau_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.50, 0.95, 10), 2).tolist())
    gmm_max_iter: int = 80
    gmm_tol: float = 1e-4
    gmm_min_var: float = 1e-4

    # Label shift EM
    enable_label_shift_auto: bool = True
    label_shift_max_iter: int = 50
    label_shift_tol: float = 1e-6
    alpha_clip: Tuple[float, float] = (0.1, 10.0)

    # PAS-UniDA++ self-training
    ours_rounds: int = 3
    ours_epochs_per_round: int = 1
    ours_batch_size: int = 16
    ours_lr: float = 2e-5
    ours_weight_decay: float = 0.01
    ours_warmup_ratio: float = 0.06

    # 3-bucket selection
    q_min_known: float = 0.55
    margin_min_known: float = 0.08
    q_max_unknown: float = 0.20
    conf_max_unknown: float = 0.60
    outlier_min_unknown: float = 0.80  # from OE head if available

    # pseudo-label curriculum
    pseudo_frac_open: Tuple[float, ...] = (0.20, 0.40, 0.60)
    pseudo_frac_closed: Tuple[float, ...] = (0.40, 0.70, 1.00)

    # losses
    lambda_pseudo_ce: float = 1.0
    lambda_unknown_unif: float = 0.05
    lambda_outlier_bce: float = 0.20
    lambda_entropy_ignore: float = 0.05  # only in closed-set mode

    # Baseline: EntropyMin / InfoMax
    ent_epochs: int = 2
    ent_batch_size: int = 16
    ent_lr: float = 2e-5
    ent_weight_decay: float = 0.01
    ent_warmup_ratio: float = 0.06
    lambda_ent: float = 0.10
    lambda_div: float = 0.10

    # Baseline: Self-training closed-set
    st_rounds: int = 3
    st_epochs_per_round: int = 1
    st_batch_size: int = 16
    st_lr: float = 2e-5
    st_weight_decay: float = 0.01
    st_warmup_ratio: float = 0.06
    st_tau: float = 0.90

    # Baseline: DANN
    dann_epochs: int = 2
    dann_batch_size: int = 16
    dann_lr: float = 2e-5
    dann_weight_decay: float = 0.01
    dann_warmup_ratio: float = 0.06
    lambda_domain: float = 0.20
    grl_lambda: float = 1.0


CFG = Config()

# Choose which methods to run
METHODS_TO_RUN = [
    "source_only",
    "dapt_source",
    "labelshift_em",
    "entropy_min",
    "selftrain_closed",
    "dann",
    "pas_unida_pp",
]


# Multi-seed evaluation
SEEDS_TO_RUN = [42, 43, 44, 45, 46]
# Optional manual overrides (you can edit)
MODEL_NAME_OVERRIDE = None  # e.g. "bert-base-uncased"
TEXT_COL_OVERRIDE = None    # e.g. "text"
LABEL_COL_OVERRIDE = None   # e.g. "label"


# ============================================================
# 2) Utils
# ============================================================
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
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def detect_text_and_label_columns(df: pd.DataFrame,
                                  text_override: Optional[str] = None,
                                  label_override: Optional[str] = None) -> Tuple[str, Optional[str]]:
    cols = list(df.columns)

    # text col
    if text_override is not None:
        if text_override not in cols:
            raise ValueError(f"text_override={text_override} not in columns: {cols}")
        text_col = text_override
    else:
        for cand in ["text", "sentence", "content", "review", "comment", "abstract", "title", "body"]:
            if cand in cols and df[cand].dtype == object:
                text_col = cand
                break
        else:
            obj_cols = [c for c in cols if df[c].dtype == object]
            if not obj_cols:
                raise ValueError("No object/string column found to use as text.")
            lengths = {}
            for c in obj_cols:
                s = df[c].astype(str)
                lengths[c] = float(s.str.len().replace([np.inf, -np.inf], np.nan).fillna(0).mean())
            text_col = max(lengths, key=lengths.get)

    # label col
    if label_override is not None:
        if label_override not in cols:
            raise ValueError(f"label_override={label_override} not in columns: {cols}")
        return text_col, label_override

    for cand in ["label", "labels", "y", "target", "class", "category"]:
        if cand in cols and cand != text_col:
            return text_col, cand

    best = None
    for c in cols:
        if c == text_col:
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 50:
            score = nunique
            if best is None or score < best[0]:
                best = (score, c)
    label_col = best[1] if best is not None else None
    return text_col, label_col


def ensure_text_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    tc, _ = detect_text_and_label_columns(df, text_override=None, label_override=None)
    return tc


def clean_text_list(series: pd.Series) -> List[str]:
    s = series.fillna("").astype(str).tolist()
    return [x if isinstance(x, str) else str(x) for x in s]


def pick_model_name_by_language(sample_texts: List[str]) -> str:
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


def load_tokenizer_robust(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"[Warn] fast tokenizer failed ({e}). Fallback use_fast=False.")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-12)


# ============================================================
# 3) Metrics
# ============================================================
def confusion_matrix_int(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]) -> float:
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else 0.0


def balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    # assumes labels in {0,1}
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    return float(0.5 * (tpr + tnr))


# ============================================================
# 4) Datasets
# ============================================================
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
    Fields:
      - class supervision: class_labels, class_mask, class_weight
      - unknown uniform loss: unk_mask, unk_weight
      - outlier head supervision: outlier_target (float), outlier_weight
      - entropy minimization on ignore: ent_mask, ent_weight
    """
    def __init__(self,
                 encodings: Dict[str, List[List[int]]],
                 class_labels: List[int],
                 class_mask: List[float],
                 class_weight: List[float],
                 unk_mask: List[float],
                 unk_weight: List[float],
                 outlier_target: List[float],
                 outlier_weight: List[float],
                 ent_mask: List[float],
                 ent_weight: List[float]):
        self.encodings = encodings
        self.class_labels = class_labels
        self.class_mask = class_mask
        self.class_weight = class_weight
        self.unk_mask = unk_mask
        self.unk_weight = unk_weight
        self.outlier_target = outlier_target
        self.outlier_weight = outlier_weight
        self.ent_mask = ent_mask
        self.ent_weight = ent_weight

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["class_labels"] = torch.tensor(self.class_labels[idx], dtype=torch.long)
        item["class_mask"] = torch.tensor(self.class_mask[idx], dtype=torch.float32)
        item["class_weight"] = torch.tensor(self.class_weight[idx], dtype=torch.float32)
        item["unk_mask"] = torch.tensor(self.unk_mask[idx], dtype=torch.float32)
        item["unk_weight"] = torch.tensor(self.unk_weight[idx], dtype=torch.float32)
        item["outlier_target"] = torch.tensor(self.outlier_target[idx], dtype=torch.float32)
        item["outlier_weight"] = torch.tensor(self.outlier_weight[idx], dtype=torch.float32)
        item["ent_mask"] = torch.tensor(self.ent_mask[idx], dtype=torch.float32)
        item["ent_weight"] = torch.tensor(self.ent_weight[idx], dtype=torch.float32)
        return item


# ============================================================
# 5) Models (multi-head + optional domain head)
# ============================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class MultiHeadNLPModel(nn.Module):
    """
    Encoder + classifier + outlier head.
    Optional: domain head for DANN (with GRL).
    """
    def __init__(self, base_model_name_or_path: str, num_classes: int, use_domain: bool = False, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name_or_path)
        hidden = getattr(self.encoder.config, "hidden_size", 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.outlier_head = nn.Linear(hidden, 1)

        self.use_domain = use_domain
        if use_domain:
            self.domain_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Always use CLS token to avoid pooler instability
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return cls

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        feat = self.encode(input_ids, attention_mask)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        outlier_logit = self.outlier_head(feat).squeeze(-1)

        domain_logit = None
        if self.use_domain:
            rev = grad_reverse(feat, grl_lambda)
            domain_logit = self.domain_head(rev).squeeze(-1)

        return {
            "features": feat,
            "logits": logits,
            "outlier_logit": outlier_logit,
            "domain_logit": domain_logit,
        }


# ============================================================
# 6) Calibration: Temperature scaling
# ============================================================
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.raw).clamp(min=1e-6, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature()


@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
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

    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([scaler.raw], lr=0.5, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return scaler


# ============================================================
# 7) DAPT (MLM)
# ============================================================
def train_dapt_mlm(model_name: str, tokenizer: AutoTokenizer, texts: List[str], cfg: Config) -> str:
    if not cfg.enable_dapt:
        return model_name

    print("\n[Phase 1] DAPT (MLM) warm-up on target train...")
    safe_mkdir(cfg.output_dir)
    dapt_dir = os.path.join(cfg.output_dir, "dapt_encoder")
    safe_mkdir(dapt_dir)
    # Reuse cached DAPT encoder if it already exists
    if os.path.exists(os.path.join(dapt_dir, "config.json")):
        print(f"  [DAPT] found existing encoder at: {dapt_dir} (reuse)")
        return dapt_dir


    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(cfg.device)

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

    base = mlm_model.base_model
    base.save_pretrained(dapt_dir)
    tokenizer.save_pretrained(dapt_dir)
    print(f"  [DAPT] saved encoder to: {dapt_dir}")
    return dapt_dir


# ============================================================
# 8) Forward helpers
# ============================================================
@torch.no_grad()
def forward_logits_outlier(model: MultiHeadNLPModel,
                           dataset: TokenizedTextDataset,
                           batch_size: int,
                           device: str) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    logits_list, outlier_list = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_list.append(out["logits"].detach().cpu().numpy())
        outlier_list.append(out["outlier_logit"].detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(outlier_list, axis=0)


@torch.no_grad()
def forward_features(model: MultiHeadNLPModel,
                     dataset: TokenizedTextDataset,
                     batch_size: int,
                     device: str) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    feats = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        feats.append(out["features"].detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt((x * x).sum(axis=axis, keepdims=True) + eps)
    return x / norm


def build_class_prototypes(source_feats: np.ndarray, source_labels: np.ndarray, num_classes: int) -> np.ndarray:
    D = source_feats.shape[1]
    protos = np.zeros((num_classes, D), dtype=np.float32)
    for c in range(num_classes):
        idx = np.where(source_labels == c)[0]
        if len(idx) == 0:
            continue
        protos[c] = source_feats[idx].mean(axis=0)
    return l2_normalize(protos, axis=-1)


def prototype_agreement(target_feats: np.ndarray, pred_cls: np.ndarray, protos: np.ndarray) -> float:
    if target_feats.shape[0] == 0:
        return 0.0
    tf = l2_normalize(target_feats, axis=-1)
    sims = tf @ protos.T
    nn_cls = np.argmax(sims, axis=1)
    return float((nn_cls == pred_cls).mean())


# ============================================================
# 9) Energy GMM (1D, 2 components) - custom EM
# ============================================================
def energy_score_from_logits_np(logits: np.ndarray) -> np.ndarray:
    # energy = -logsumexp(logits)
    x = logits.astype(np.float64)
    m = np.max(x, axis=1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True) + 1e-12)
    return (-lse.squeeze(1)).astype(np.float64)


def normal_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    coef = 1.0 / math.sqrt(2.0 * math.pi * var)
    return coef * np.exp(-(x - mean) ** 2 / (2.0 * var))


def fit_gmm_1d_two_comp(x: np.ndarray, max_iter: int = 80, tol: float = 1e-4, min_var: float = 1e-4) -> Dict[str, Any]:
    x = x.astype(np.float64)
    x = np.clip(x, np.percentile(x, 0.5), np.percentile(x, 99.5))
    n = x.shape[0]
    if n < 10:
        m = float(np.mean(x))
        v = float(np.var(x) + min_var)
        resp = np.ones((n, 2), dtype=np.float64) * 0.5
        return {"means": [m, m + 1e-3], "vars": [v, v], "pi": 0.5, "resp": resp, "ll": -1e9}

    m1, m2 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
    v1 = float(np.var(x) + min_var)
    v2 = float(np.var(x) + min_var)
    pi = 0.5
    prev_ll = None

    for _ in range(max_iter):
        p1 = pi * normal_pdf(x, m1, v1)
        p2 = (1.0 - pi) * normal_pdf(x, m2, v2)
        denom = p1 + p2 + 1e-12
        r1 = p1 / denom
        r2 = p2 / denom

        n1 = float(np.sum(r1) + 1e-12)
        n2 = float(np.sum(r2) + 1e-12)
        pi = n1 / (n1 + n2)

        m1 = float(np.sum(r1 * x) / n1)
        m2 = float(np.sum(r2 * x) / n2)

        v1 = float(np.sum(r1 * (x - m1) ** 2) / n1)
        v2 = float(np.sum(r2 * (x - m2) ** 2) / n2)
        v1, v2 = max(v1, min_var), max(v2, min_var)

        ll = float(np.sum(np.log(denom)))
        if prev_ll is not None and abs(ll - prev_ll) < tol * (abs(prev_ll) + 1.0):
            prev_ll = ll
            break
        prev_ll = ll

    resp = np.stack([r1, r2], axis=1)
    return {"means": [m1, m2], "vars": [v1, v2], "pi": pi, "resp": resp, "ll": prev_ll}


def gmm_known_probability(energy: np.ndarray, gmm: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    means, vars_, resp = gmm["means"], gmm["vars"], gmm["resp"]
    known_comp = int(np.argmin(means))  # lower mean energy => known
    q = resp[:, known_comp].astype(np.float64)
    sep = abs(means[0] - means[1]) / math.sqrt(vars_[0] + vars_[1] + 1e-12)
    return q, float(sep)


# ============================================================
# 10) Label shift EM (Saerens/MLLS-style) with soft known weights
# ============================================================
def estimate_label_shift_em(p: np.ndarray,
                            q_known: np.ndarray,
                            source_prior: np.ndarray,
                            max_iter: int = 50,
                            tol: float = 1e-6) -> np.ndarray:
    N, K = p.shape
    w = source_prior.copy().astype(np.float64)
    w = w / (w.sum() + 1e-12)

    q = q_known.astype(np.float64).clip(0.0, 1.0)
    q_sum = float(q.sum() + 1e-12)

    prev = None
    for _ in range(max_iter):
        denom = (p * w.reshape(1, -1)).sum(axis=1, keepdims=True) + 1e-12
        r = (p * w.reshape(1, -1)) / denom
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
    p2 = p * alpha.reshape(1, -1)
    return p2 / (p2.sum(axis=1, keepdims=True) + 1e-12)


# ============================================================
# 11) Training: supervised source (+ optional OE)
# ============================================================
@torch.no_grad()
def evaluate_source_acc(model: MultiHeadNLPModel, dataset: TokenizedTextDataset, batch_size: int, device: str) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = out["logits"].argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(1, total)


def train_supervised_source(model: MultiHeadNLPModel,
                            src_train: TokenizedTextDataset,
                            src_val: TokenizedTextDataset,
                            cfg: Config,
                            politics_texts: Optional[List[str]],
                            tokenizer: AutoTokenizer) -> MultiHeadNLPModel:
    """
    Train on source labels. If politics_texts provided, do OE to train outlier head:
      - source outlier target = 0
      - politics outlier target = 1
      - optional uniform loss on politics classifier logits
    """
    print("\n[Train] Supervised source fine-tuning" + (" + OE(politics)" if politics_texts else "") + "...")
    model.to(cfg.device)

    src_loader = DataLoader(src_train, batch_size=cfg.src_batch_size, shuffle=True)

    pol_loader = None
    if politics_texts is not None and len(politics_texts) > 0:
        enc_pol = tokenizer(politics_texts, truncation=True, padding="max_length", max_length=cfg.max_length)
        pol_ds = TokenizedTextDataset(enc_pol, labels=None)
        pol_loader = DataLoader(pol_ds, batch_size=cfg.src_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.src_lr, weight_decay=cfg.src_weight_decay)
    total_steps = len(src_loader) * cfg.src_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(cfg.src_warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    best_val = -1.0
    best_state = None
    patience = 0

    pol_iter = iter(pol_loader) if pol_loader is not None else None

    for epoch in range(cfg.src_epochs):
        model.train()
        losses = []

        for src_batch in src_loader:
            input_ids_s = src_batch["input_ids"].to(cfg.device)
            attn_s = src_batch["attention_mask"].to(cfg.device)
            y_s = src_batch["labels"].to(cfg.device)

            out_s = model(input_ids=input_ids_s, attention_mask=attn_s)
            logits_s = out_s["logits"]
            outlier_s = out_s["outlier_logit"]

            ce = F.cross_entropy(logits_s, y_s)
            bce_src = F.binary_cross_entropy_with_logits(outlier_s, torch.zeros_like(outlier_s))

            loss = ce + cfg.lambda_out_src * bce_src

            # OE: politics as unknown for outlier head + uniform classifier
            if pol_loader is not None:
                try:
                    pol_batch = next(pol_iter)
                except StopIteration:
                    pol_iter = iter(pol_loader)
                    pol_batch = next(pol_iter)

                input_ids_p = pol_batch["input_ids"].to(cfg.device)
                attn_p = pol_batch["attention_mask"].to(cfg.device)

                out_p = model(input_ids=input_ids_p, attention_mask=attn_p)
                logits_p = out_p["logits"]
                outlier_p = out_p["outlier_logit"]

                bce_pol = F.binary_cross_entropy_with_logits(outlier_p, torch.ones_like(outlier_p))
                loss = loss + cfg.lambda_oe_out * bce_pol

                if cfg.lambda_oe_unif > 0:
                    logp = F.log_softmax(logits_p, dim=-1)
                    unif = (-logp.mean(dim=-1)).mean()
                    loss = loss + cfg.lambda_oe_unif * unif

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        val_acc = evaluate_source_acc(model, src_val, batch_size=cfg.src_batch_size, device=cfg.device)
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


# ============================================================
# 12) Baselines
# ============================================================
def train_entropy_minimization(model: MultiHeadNLPModel,
                               src_train: TokenizedTextDataset,
                               tgt_train: TokenizedTextDataset,
                               cfg: Config) -> MultiHeadNLPModel:
    """
    InfoMax baseline:
      L = CE(source) + lambda_ent * H(p_t) - lambda_div * H(mean(p_t))
    """
    print("\n[Baseline] Entropy Minimization / InfoMax adaptation...")
    model.to(cfg.device)
    src_loader = DataLoader(src_train, batch_size=cfg.ent_batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_train, batch_size=cfg.ent_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ent_lr, weight_decay=cfg.ent_weight_decay)
    total_steps = max(len(src_loader), len(tgt_loader)) * cfg.ent_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(cfg.ent_warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    tgt_iter = iter(tgt_loader)
    for epoch in range(cfg.ent_epochs):
        model.train()
        losses = []
        for src_batch in src_loader:
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            # source
            input_ids_s = src_batch["input_ids"].to(cfg.device)
            attn_s = src_batch["attention_mask"].to(cfg.device)
            y_s = src_batch["labels"].to(cfg.device)

            out_s = model(input_ids=input_ids_s, attention_mask=attn_s)
            logits_s = out_s["logits"]
            ce = F.cross_entropy(logits_s, y_s)

            # target unlabeled
            input_ids_t = tgt_batch["input_ids"].to(cfg.device)
            attn_t = tgt_batch["attention_mask"].to(cfg.device)

            out_t = model(input_ids=input_ids_t, attention_mask=attn_t)
            logits_t = out_t["logits"]
            p_t = F.softmax(logits_t, dim=-1)

            ent = (- (p_t * (p_t.clamp(min=1e-12)).log()).sum(dim=-1)).mean()  # H(p)
            mean_p = p_t.mean(dim=0)
            div = - (mean_p * (mean_p.clamp(min=1e-12)).log()).sum()  # H(mean_p)

            loss = ce + cfg.lambda_ent * ent - cfg.lambda_div * div

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"  [InfoMax] epoch={epoch+1}/{cfg.ent_epochs} loss={np.mean(losses):.4f}")

    return model


def train_selftraining_closed(model: MultiHeadNLPModel,
                              src_train: TokenizedTextDataset,
                              tgt_train_texts: List[str],
                              tokenizer: AutoTokenizer,
                              cfg: Config,
                              tau: float) -> MultiHeadNLPModel:
    """
    Closed-set self-training baseline:
      - pseudo-label target samples with maxprob >= tau
      - train on source + pseudo-labeled target
    """
    print("\n[Baseline] Closed-set Self-Training...")
    model.to(cfg.device)

    # Because we need correct mapping to texts, we do pseudo-label in one shot with indices.
    enc_t = tokenizer(tgt_train_texts, truncation=True, padding="max_length", max_length=cfg.max_length)
    ds_t = TokenizedTextDataset(enc_t, labels=None)
    loader_t = DataLoader(ds_t, batch_size=cfg.st_batch_size, shuffle=False)

    # source loader for training
    src_loader = DataLoader(src_train, batch_size=cfg.st_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.st_lr, weight_decay=cfg.st_weight_decay)

    for r in range(cfg.st_rounds):
        # generate pseudo labels
        model.eval()
        confs = []
        preds = []
        for batch in loader_t:
            input_ids = batch["input_ids"].to(cfg.device)
            attn = batch["attention_mask"].to(cfg.device)
            out = model(input_ids=input_ids, attention_mask=attn)
            p = F.softmax(out["logits"], dim=-1)
            conf, pred = p.max(dim=-1)
            confs.append(conf.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
        confs = np.concatenate(confs, axis=0)
        preds = np.concatenate(preds, axis=0)

        keep_idx = np.where(confs >= tau)[0]
        if len(keep_idx) == 0:
            print(f"  [ST] round={r+1}/{cfg.st_rounds} keep=0 -> stop")
            break

        # build pseudo dataset
        pseudo_texts = [tgt_train_texts[i] for i in keep_idx.tolist()]
        pseudo_labels = preds[keep_idx].astype(int).tolist()
        enc_p = tokenizer(pseudo_texts, truncation=True, padding="max_length", max_length=cfg.max_length)
        pseudo_ds = TokenizedTextDataset(enc_p, pseudo_labels)

        # combined loader: concatenate by cycling (simple)
        pseudo_loader = DataLoader(pseudo_ds, batch_size=cfg.st_batch_size, shuffle=True)
        pseudo_iter = iter(pseudo_loader)

        total_steps = max(len(src_loader), len(pseudo_loader)) * cfg.st_epochs_per_round
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(cfg.st_warmup_ratio * total_steps)),
            num_training_steps=total_steps,
        )

        model.train()
        losses = []
        for epoch in range(cfg.st_epochs_per_round):
            for src_batch in src_loader:
                try:
                    p_batch = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    p_batch = next(pseudo_iter)

                # source
                input_ids_s = src_batch["input_ids"].to(cfg.device)
                attn_s = src_batch["attention_mask"].to(cfg.device)
                y_s = src_batch["labels"].to(cfg.device)

                out_s = model(input_ids=input_ids_s, attention_mask=attn_s)
                loss_s = F.cross_entropy(out_s["logits"], y_s)

                # pseudo target
                input_ids_p = p_batch["input_ids"].to(cfg.device)
                attn_p = p_batch["attention_mask"].to(cfg.device)
                y_p = p_batch["labels"].to(cfg.device)

                out_p = model(input_ids=input_ids_p, attention_mask=attn_p)
                loss_p = F.cross_entropy(out_p["logits"], y_p)

                loss = 0.5 * loss_s + 0.5 * loss_p

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

        print(f"  [ST] round={r+1}/{cfg.st_rounds} pseudo_keep={len(keep_idx)} loss={np.mean(losses):.4f}")

    return model


def train_dann(model: MultiHeadNLPModel,
               src_train: TokenizedTextDataset,
               tgt_train: TokenizedTextDataset,
               cfg: Config) -> MultiHeadNLPModel:
    """
    DANN baseline (closed-set):
      L = CE(source labels) + lambda_domain * BCE(domain(source)=0, domain(target)=1)
    """
    print("\n[Baseline] DANN (Domain Adversarial) adaptation...")
    model.to(cfg.device)
    if not model.use_domain:
        raise ValueError("DANN model must be initialized with use_domain=True")

    src_loader = DataLoader(src_train, batch_size=cfg.dann_batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_train, batch_size=cfg.dann_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dann_lr, weight_decay=cfg.dann_weight_decay)

    total_steps = max(len(src_loader), len(tgt_loader)) * cfg.dann_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(cfg.dann_warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    tgt_iter = iter(tgt_loader)
    for epoch in range(cfg.dann_epochs):
        model.train()
        losses = []
        for src_batch in src_loader:
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            # source forward
            input_ids_s = src_batch["input_ids"].to(cfg.device)
            attn_s = src_batch["attention_mask"].to(cfg.device)
            y_s = src_batch["labels"].to(cfg.device)

            out_s = model(input_ids=input_ids_s, attention_mask=attn_s, grl_lambda=cfg.grl_lambda)
            logits_s = out_s["logits"]
            dom_s = out_s["domain_logit"]

            # target forward
            input_ids_t = tgt_batch["input_ids"].to(cfg.device)
            attn_t = tgt_batch["attention_mask"].to(cfg.device)
            out_t = model(input_ids=input_ids_t, attention_mask=attn_t, grl_lambda=cfg.grl_lambda)
            dom_t = out_t["domain_logit"]

            # losses
            ce = F.cross_entropy(logits_s, y_s)

            dom_loss_s = F.binary_cross_entropy_with_logits(dom_s, torch.zeros_like(dom_s))
            dom_loss_t = F.binary_cross_entropy_with_logits(dom_t, torch.ones_like(dom_t))
            dom_loss = 0.5 * (dom_loss_s + dom_loss_t)

            loss = ce + cfg.lambda_domain * dom_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"  [DANN] epoch={epoch+1}/{cfg.dann_epochs} loss={np.mean(losses):.4f}")

    return model


# ============================================================
# 13) PAS-UniDA++ v2 (ours)
# ============================================================
def compute_q_known(model: MultiHeadNLPModel,
                    temp_scaler: Optional[TemperatureScaler],
                    dataset: TokenizedTextDataset,
                    cfg: Config,
                    use_oe: bool) -> Dict[str, Any]:
    """
    q_known is a combination of:
      - q_gmm from energy (unsupervised)
      - q_oe from outlier head (if OE used)
    """
    logits, outlier_logit = forward_logits_outlier(model, dataset, batch_size=cfg.src_batch_size, device=cfg.device)
    if temp_scaler is not None:
        with torch.no_grad():
            lt = torch.tensor(logits, dtype=torch.float32).to(cfg.device)
            logits_scaled = temp_scaler(lt).detach().cpu().numpy()
    else:
        logits_scaled = logits

    p = softmax_np(logits_scaled, axis=-1)
    energy = energy_score_from_logits_np(logits_scaled)
    gmm = fit_gmm_1d_two_comp(energy, max_iter=cfg.gmm_max_iter, tol=cfg.gmm_tol, min_var=cfg.gmm_min_var)
    q_gmm, sep = gmm_known_probability(energy, gmm)

    # outlier head -> prob_unknown = sigmoid(outlier_logit)
    prob_unk_oe = 1.0 / (1.0 + np.exp(-outlier_logit.astype(np.float64)))
    q_oe = 1.0 - prob_unk_oe  # prob_known

    if use_oe:
        w_oe = 0.7
    else:
        w_oe = 0.2

    q = (w_oe * q_oe + (1.0 - w_oe) * q_gmm).clip(0.0, 1.0)
    pi_known = float(np.mean(q))
    mean_unk_oe = float(np.mean(prob_unk_oe))

    return {
        "p": p,
        "logits_scaled": logits_scaled,
        "q": q,
        "q_gmm": q_gmm,
        "q_oe": q_oe,
        "prob_unk_oe": prob_unk_oe,
        "pi_known": pi_known,
        "mean_unk_oe": mean_unk_oe,
        "sep": float(sep),
    }



def decide_open_set_flag(stats: Dict[str, Any], use_oe: bool) -> bool:
    """
    Conservative open-set decision.

    Key point:
    - If we did NOT train outlier head with OE (e.g., politics), its probability is not reliable.
      In that case, we default to closed-set to avoid "unknown collapse".

    When OE is available, we require multiple signals:
    - outlier head says many unknown-like samples
    - energy is somewhat separable
    - not almost-all-known
    """
    if not use_oe:
        return False

    mean_unk_oe = float(stats["mean_unk_oe"])
    sep = float(stats["sep"])
    pi_known = float(stats["pi_known"])

    # Strong evidence: many unknown-like samples + separable energy + not almost-all-known
    if mean_unk_oe > 0.25 and sep > 1.0 and pi_known < 0.92:
        return True
    return False
def auto_select_tau_and_mode(model: MultiHeadNLPModel,
                             temp_scaler: Optional[TemperatureScaler],
                             source_feats: np.ndarray,
                             source_labels: np.ndarray,
                             source_prior: np.ndarray,
                             target_ds: TokenizedTextDataset,
                             cfg: Config,
                             use_oe: bool) -> Dict[str, Any]:
    """
    Select:
      - tau_pseudo (for pseudo-label selection)
      - mode in {none, em}
      - open_set_flag
      - tau_reject (for inference), only enabled when open_set_flag=True
    """
    print("\n[Phase 2] Auto selection (tau_pseudo, mode, open_set_flag)...")

    K = int(source_prior.shape[0])
    protos = build_class_prototypes(source_feats, source_labels, K)
    tgt_feats = forward_features(model, target_ds, batch_size=cfg.src_batch_size, device=cfg.device)

    stats = compute_q_known(model, temp_scaler, target_ds, cfg, use_oe=use_oe)
    p = stats["p"]
    q = stats["q"]
    pi_known = stats["pi_known"]
    sep = stats["sep"]
    prob_unk_oe = stats["prob_unk_oe"]

    open_set_flag = decide_open_set_flag(stats, use_oe=use_oe)
    print(f"  [Auto] sep={sep:.3f} pi_known={pi_known:.3f} mean_unk_oe={stats['mean_unk_oe']:.3f} -> open_set={open_set_flag}")

    # Candidate modes
    modes = ["none"]
    if cfg.enable_label_shift_auto:
        modes.append("em")

    best = {"proxy": -1e18}

    # In closed-set, we want pseudo coverage not too low, to avoid "tiny pseudo set".
    if open_set_flag:
        min_cov = 0.10
        max_cov = 0.95
        cov_floor = 0.10
    else:
        min_cov = 0.30
        max_cov = 0.98
        cov_floor = 0.60

    for mode in modes:
        if mode == "em":
            w_t = estimate_label_shift_em(p, q, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol)
            alpha = (w_t / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])
            p_adj = apply_label_shift(p, alpha)
        else:
            alpha = np.ones((K,), dtype=np.float64)
            p_adj = p

        conf = np.max(p_adj, axis=1)
        pred = np.argmax(p_adj, axis=1)

        for tau in cfg.tau_grid:
            selected = conf >= tau
            cov = float(np.mean(selected))
            sel_count = int(selected.sum())

            if cov < min_cov or cov > max_cov:
                continue
            if sel_count < max(cfg.src_batch_size * 4, 64):
                continue

            q_quality = float(np.mean(q[selected]))
            agree = prototype_agreement(tgt_feats[selected], pred[selected], protos)
            mismatch = abs(cov - pi_known)

            # penalty for collapse (too low coverage)
            low_cov_pen = 0.0
            if cov < cov_floor:
                low_cov_pen = (cov_floor - cov) ** 2 * 2.0

            # penalty if OE says unknown but selected set includes many "unknown-like"
            unk_in_selected = float(np.mean(prob_unk_oe[selected]))  # higher means more unknown-ish
            unk_pen = 0.0
            if open_set_flag and unk_in_selected > 0.35:
                unk_pen = (unk_in_selected - 0.35) * 0.5

            proxy = q_quality + 0.30 * agree - 1.00 * mismatch - low_cov_pen - unk_pen

            if proxy > best["proxy"]:
                best = {
                    "proxy": float(proxy),
                    "mode": mode,
                    "tau_pseudo": float(tau),
                    "alpha": alpha.astype(np.float64),
                    "pi_known": float(pi_known),
                    "sep": float(sep),
                    "open_set": bool(open_set_flag),
                    "cov_sel": float(cov),
                    "q_quality": float(q_quality),
                    "proto_agree": float(agree),
                    "mismatch": float(mismatch),
                }

    if best["proxy"] < -1e10:
        # fallback
        best = {
            "proxy": -1e10,
            "mode": "none",
            "tau_pseudo": 0.90,
            "alpha": np.ones((K,), dtype=np.float64),
            "pi_known": float(pi_known),
            "sep": float(sep),
            "open_set": bool(open_set_flag),
            "cov_sel": 0.0,
            "q_quality": 0.0,
            "proto_agree": 0.0,
            "mismatch": 0.0,
        }

    # Decide tau_reject:
    # - open_set: use same tau (conservative)
    # - closed_set: disable rejection at inference
    if best["open_set"]:
        tau_reject = best["tau_pseudo"]
        reject_enabled = True
    else:
        tau_reject = 0.0
        reject_enabled = False

    best["tau_reject"] = float(tau_reject)
    best["reject_enabled"] = bool(reject_enabled)

    print(
        f"  [Auto] best: mode={best['mode']} tau_pseudo={best['tau_pseudo']:.2f} "
        f"reject={best['reject_enabled']} tau_reject={best['tau_reject']:.2f} "
        f"proxy={best['proxy']:.4f} cov_sel={best['cov_sel']:.3f}"
    )
    return best


def build_selftrain_dataset_ours(tokenizer: AutoTokenizer,
                                 cfg: Config,
                                 source_texts: List[str],
                                 source_labels: List[int],
                                 target_texts: List[str],
                                 p_adj: np.ndarray,
                                 q: np.ndarray,
                                 q_oe: np.ndarray,
                                 prob_unk_oe: np.ndarray,
                                 alpha: np.ndarray,
                                 tau_pseudo: float,
                                 open_set_flag: bool,
                                 round_idx: int,
                                 politics_texts: Optional[List[str]]) -> Tuple[TokenizedSelfTrainDataset, Dict[str, int]]:
    """
    3-bucket target:
      - pseudo-known: used for CE
      - pseudo-unknown: used for uniform loss + outlier BCE=1
      - ignore: no CE, no uniform; optional entropy-min (closed-set); outlier BCE soft label

    politics:
      - treated as strong unknown for outlier + uniform (small weight)
    """
    K = p_adj.shape[1]
    conf = p_adj.max(axis=1)
    sorted_probs = np.sort(p_adj, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    pred = np.argmax(p_adj, axis=1)

    # pseudo-known candidates
    known_cand = (conf >= tau_pseudo) & (q >= cfg.q_min_known) & (margin >= cfg.margin_min_known)
    idx_known = np.where(known_cand)[0]

    # curriculum fraction
    if open_set_flag:
        frac_list = cfg.pseudo_frac_open
    else:
        frac_list = cfg.pseudo_frac_closed
    frac = frac_list[min(round_idx, len(frac_list) - 1)]

    pseudo_known_mask = np.zeros_like(known_cand, dtype=bool)
    if len(idx_known) > 0:
        k_keep = max(1, int(len(idx_known) * frac))
        order = np.argsort(-conf[idx_known])
        keep = idx_known[order[:k_keep]]
        pseudo_known_mask[keep] = True

    # pseudo-unknown: strong evidence only
    unknown_cand = (q <= cfg.q_max_unknown) & (conf <= cfg.conf_max_unknown)
    # if OE signal available, require outlier prob high
    unknown_cand = unknown_cand & (prob_unk_oe >= cfg.outlier_min_unknown)
    pseudo_unknown_mask = unknown_cand & (~pseudo_known_mask)

    # ignore: the rest of target
    ignore_mask = (~pseudo_known_mask) & (~pseudo_unknown_mask)

    # Build encodings
    src_n = len(source_texts)
    tgt_n = len(target_texts)

    enc_src = tokenizer(source_texts, truncation=True, padding="max_length", max_length=cfg.max_length)
    enc_tgt = tokenizer(target_texts, truncation=True, padding="max_length", max_length=cfg.max_length)

    pol_n = 0
    enc_pol = None
    if politics_texts is not None and len(politics_texts) > 0:
        pol_n = len(politics_texts)
        enc_pol = tokenizer(politics_texts, truncation=True, padding="max_length", max_length=cfg.max_length)

    def merge_enc(enc1, enc2, enc3=None):
        out = {}
        for k in enc1.keys():
            out[k] = enc1[k] + enc2[k] + (enc3[k] if enc3 is not None else [])
        return out

    enc = merge_enc(enc_src, enc_tgt, enc_pol)

    # Build supervision arrays
    class_labels: List[int] = []
    class_mask: List[float] = []
    class_weight: List[float] = []

    unk_mask: List[float] = []
    unk_weight: List[float] = []

    outlier_target: List[float] = []
    outlier_weight: List[float] = []

    ent_mask: List[float] = []
    ent_weight: List[float] = []

    # Source: known
    for i in range(src_n):
        y = int(source_labels[i])
        class_labels.append(y)
        class_mask.append(1.0)
        class_weight.append(1.0)

        unk_mask.append(0.0)
        unk_weight.append(0.0)

        outlier_target.append(0.0)
        outlier_weight.append(1.0)

        ent_mask.append(0.0)
        ent_weight.append(0.0)

    # Target
    for i in range(tgt_n):
        if pseudo_known_mask[i]:
            y = int(pred[i])
            w = float(max(1e-6, q[i] * conf[i] * float(alpha[y])))
            class_labels.append(y)
            class_mask.append(1.0)
            class_weight.append(w)

            unk_mask.append(0.0)
            unk_weight.append(0.0)

            outlier_target.append(0.0)
            outlier_weight.append(1.0)

            ent_mask.append(0.0)
            ent_weight.append(0.0)
        elif pseudo_unknown_mask[i]:
            # strong unknown
            class_labels.append(0)
            class_mask.append(0.0)
            class_weight.append(0.0)

            unk_mask.append(1.0)
            unk_weight.append(1.0)

            outlier_target.append(1.0)
            outlier_weight.append(1.0)

            ent_mask.append(0.0)
            ent_weight.append(0.0)
        else:
            # ignore
            class_labels.append(0)
            class_mask.append(0.0)
            class_weight.append(0.0)

            unk_mask.append(0.0)
            unk_weight.append(0.0)

            # soft outlier target to keep head calibrated, but low weight
            soft_unk = float(1.0 - q_oe[i])
            outlier_target.append(soft_unk)
            outlier_weight.append(0.2)

            # entropy min only in closed-set mode
            if not open_set_flag:
                ent_mask.append(1.0)
                # weight higher when q is high (likely known) but still uncertain
                ent_weight.append(float(max(0.1, q[i])))
            else:
                ent_mask.append(0.0)
                ent_weight.append(0.0)

    # Politics as unknown (OE) - small uniform/outlier
    for _ in range(pol_n):
        class_labels.append(0)
        class_mask.append(0.0)
        class_weight.append(0.0)

        unk_mask.append(1.0)
        unk_weight.append(0.5)

        outlier_target.append(1.0)
        outlier_weight.append(0.5)

        ent_mask.append(0.0)
        ent_weight.append(0.0)

    ds = TokenizedSelfTrainDataset(
        encodings=enc,
        class_labels=class_labels,
        class_mask=class_mask,
        class_weight=class_weight,
        unk_mask=unk_mask,
        unk_weight=unk_weight,
        outlier_target=outlier_target,
        outlier_weight=outlier_weight,
        ent_mask=ent_mask,
        ent_weight=ent_weight,
    )

    info = {
        "src_n": src_n,
        "tgt_n": tgt_n,
        "pol_n": pol_n,
        "pseudo_known_n": int(pseudo_known_mask.sum()),
        "pseudo_unknown_n": int(pseudo_unknown_mask.sum()),
        "ignore_n": int(ignore_mask.sum()),
    }
    return ds, info


def train_one_round_ours(model: MultiHeadNLPModel,
                         train_ds: TokenizedSelfTrainDataset,
                         cfg: Config,
                         open_set_flag: bool) -> MultiHeadNLPModel:
    model.to(cfg.device)
    model.train()

    loader = DataLoader(train_ds, batch_size=cfg.ours_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ours_lr, weight_decay=cfg.ours_weight_decay)

    total_steps = len(loader) * cfg.ours_epochs_per_round
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(cfg.ours_warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    for epoch in range(cfg.ours_epochs_per_round):
        losses = []
        for batch in loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attn = batch["attention_mask"].to(cfg.device)

            class_labels = batch["class_labels"].to(cfg.device)
            class_mask = batch["class_mask"].to(cfg.device)
            class_weight = batch["class_weight"].to(cfg.device)

            unk_mask = batch["unk_mask"].to(cfg.device)
            unk_weight = batch["unk_weight"].to(cfg.device)

            outlier_target = batch["outlier_target"].to(cfg.device)
            outlier_weight = batch["outlier_weight"].to(cfg.device)

            ent_mask = batch["ent_mask"].to(cfg.device)
            ent_weight = batch["ent_weight"].to(cfg.device)

            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out["logits"]
            outlier_logit = out["outlier_logit"]

            # CE for labeled (source + pseudo-known)
            ce_per = F.cross_entropy(logits, class_labels, reduction="none")
            ce_denom = (class_mask * class_weight).sum().clamp(min=1.0)
            ce = (ce_per * class_mask * class_weight).sum() / ce_denom

            # Unknown uniform loss for strong unknowns
            logp = F.log_softmax(logits, dim=-1)
            unif_per = (-logp.mean(dim=-1))
            unif_denom = (unk_mask * unk_weight).sum().clamp(min=1.0)
            unif = (unif_per * unk_mask * unk_weight).sum() / unif_denom

            # Outlier head loss (soft targets allowed)
            bce_per = F.binary_cross_entropy_with_logits(outlier_logit, outlier_target, reduction="none")
            bce = (bce_per * outlier_weight).mean()

            # Entropy minimization on ignore (closed-set only)
            ent_loss = torch.tensor(0.0, device=cfg.device)
            if not open_set_flag and cfg.lambda_entropy_ignore > 0:
                p = F.softmax(logits, dim=-1)
                ent_per = - (p * (p.clamp(min=1e-12)).log()).sum(dim=-1)
                ent_denom = (ent_mask * ent_weight).sum().clamp(min=1.0)
                ent_loss = (ent_per * ent_mask * ent_weight).sum() / ent_denom

            loss = (
                cfg.lambda_pseudo_ce * ce
                + cfg.lambda_unknown_unif * unif
                + cfg.lambda_outlier_bce * bce
                + (cfg.lambda_entropy_ignore * ent_loss if not open_set_flag else 0.0)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"  [OURS] epoch={epoch+1}/{cfg.ours_epochs_per_round} loss={np.mean(losses):.4f}")

    return model


def proxy_score_ours(model: MultiHeadNLPModel,
                     temp_scaler: Optional[TemperatureScaler],
                     source_feats: np.ndarray,
                     source_labels: np.ndarray,
                     source_prior: np.ndarray,
                     target_val_ds: TokenizedTextDataset,
                     cfg: Config,
                     use_oe: bool,
                     mode: str,
                     tau_pseudo: float,
                     open_set_flag: bool) -> Dict[str, float]:
    """
    Proxy score for selecting best round.
    Key idea: avoid rewarding collapse.
    """
    K = int(source_prior.shape[0])
    protos = build_class_prototypes(source_feats, source_labels, K)

    feats = forward_features(model, target_val_ds, batch_size=cfg.src_batch_size, device=cfg.device)
    stats = compute_q_known(model, temp_scaler, target_val_ds, cfg, use_oe=use_oe)
    p = stats["p"]
    q = stats["q"]
    pi_known = stats["pi_known"]
    sep = stats["sep"]

    if mode == "em":
        w_t = estimate_label_shift_em(p, q, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol)
        alpha = (w_t / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])
        p_adj = apply_label_shift(p, alpha)
    else:
        p_adj = p

    conf = p_adj.max(axis=1)
    pred = np.argmax(p_adj, axis=1)

    # define "coverage" differently:
    # - open-set: cov = conf>=tau_pseudo (known candidate)
    # - closed-set: cov = conf>=0.5 (should be high)
    if open_set_flag:
        sel = conf >= tau_pseudo
    else:
        sel = conf >= 0.5

    cov = float(np.mean(sel))
    q_quality = float(np.mean(q[sel])) if sel.any() else 0.0
    agree = prototype_agreement(feats[sel], pred[sel], protos) if sel.any() else 0.0

    # entropy stats (lower is better)
    ent = -np.sum(p_adj * np.log(p_adj + 1e-12), axis=1)
    mean_ent = float(np.mean(ent))

    # penalties
    if open_set_flag:
        cov_floor = 0.10
    else:
        cov_floor = 0.75
    low_cov_pen = 0.0
    if cov < cov_floor:
        low_cov_pen = (cov_floor - cov) ** 2 * 2.0

    mismatch = abs(cov - (pi_known if open_set_flag else 1.0))

    proxy = q_quality + 0.30 * agree - 0.30 * mean_ent - 0.50 * mismatch - low_cov_pen
    return {
        "proxy": float(proxy),
        "cov": float(cov),
        "q_quality": float(q_quality),
        "proto_agree": float(agree),
        "mean_ent": float(mean_ent),
        "sep": float(sep),
        "pi_known": float(pi_known),
    }


# ============================================================
# 14) Evaluation (target test)
# ============================================================
@torch.no_grad()
def predict_labels(model: MultiHeadNLPModel,
                   dataset: TokenizedTextDataset,
                   device: str,
                   temp_scaler: Optional[TemperatureScaler],
                   alpha: Optional[np.ndarray],
                   reject_enabled: bool,
                   tau_reject: float) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()

    all_pred = []
    all_conf = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out["logits"]
        if temp_scaler is not None:
            logits = temp_scaler(logits)

        p = F.softmax(logits, dim=-1).detach().cpu().numpy()

        if alpha is not None:
            p = apply_label_shift(p, alpha)

        conf = p.max(axis=1)
        pred = p.argmax(axis=1).astype(int)

        if reject_enabled:
            unknown_id = p.shape[1]
            pred = np.where(conf < tau_reject, unknown_id, pred)

        all_pred.append(pred)
        all_conf.append(conf)

    return np.concatenate(all_pred, axis=0), np.concatenate(all_conf, axis=0)


def evaluate_on_target(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       num_known: int) -> Dict[str, float]:
    unknown_id = num_known
    unk_rate = float(np.mean(y_pred == unknown_id))

    known_mask = (y_true >= 0) & (y_true < num_known)
    # known-only accuracy (unknown predictions are counted as wrong)
    acc = float(np.mean(y_pred[known_mask] == y_true[known_mask])) if known_mask.any() else 0.0

    # macro-f1 on known classes only
    f1 = macro_f1(y_true[known_mask], y_pred[known_mask], classes=list(range(num_known))) if known_mask.any() else 0.0

    bal = 0.0
    if num_known == 2 and known_mask.any():
        bal = balanced_accuracy_binary(y_true[known_mask], y_pred[known_mask], pos_label=1)

    return {"acc": acc, "macro_f1": f1, "bal_acc": float(bal), "pred_unknown_rate": unk_rate}


# ============================================================
# 15) Main runner for methods
# ============================================================

# ============================================================
# 13b) Extra helpers for PAS-UniDA++ (v3): OE calibration + InfoMax proxy
# ============================================================
@torch.no_grad()
def infomax_proxy_on_dataset(model: MultiHeadNLPModel,
                             dataset: TokenizedTextDataset,
                             cfg: Config,
                             temp_scaler: Optional[TemperatureScaler] = None) -> Dict[str, float]:
    """
    Unlabeled proxy (higher is better):
        score = - H(p(x)) + H( mean_x p(x) )
    This is the standard InfoMax objective without needing labels.

    Returns:
      proxy, mean_ent, div_ent, collapse_pen, min_class_prob, max_class_prob
    """
    logits_np, _ = forward_logits_outlier(model, dataset, batch_size=cfg.src_batch_size, device=cfg.device)
    if temp_scaler is not None:
        lt = torch.tensor(logits_np, dtype=torch.float32, device=cfg.device)
        logits_np = temp_scaler(lt).detach().cpu().numpy()

    p = softmax_np(logits_np, axis=-1)
    ent = -np.sum(p * np.log(p + 1e-12), axis=1)
    mean_ent = float(np.mean(ent))

    mean_p = p.mean(axis=0)
    div_ent = float(-np.sum(mean_p * np.log(mean_p + 1e-12)))

    # mild collapse penalty (do NOT over-penalize in case of real label shift)
    min_prob = float(np.min(mean_p))
    max_prob = float(np.max(mean_p))
    collapse_pen = 0.0
    if min_prob < 0.01:
        collapse_pen = (0.01 - min_prob) * 2.0

    proxy = float((-mean_ent + div_ent) - collapse_pen)
    return {
        "proxy": proxy,
        "mean_ent": float(mean_ent),
        "div_ent": float(div_ent),
        "collapse_pen": float(collapse_pen),
        "min_class_prob": float(min_prob),
        "max_class_prob": float(max_prob),
    }


def calibrate_outlier_head_frozen(model: MultiHeadNLPModel,
                                  tokenizer: AutoTokenizer,
                                  src_texts: List[str],
                                  politics_texts: List[str],
                                  cfg: Config) -> MultiHeadNLPModel:
    """
    Train ONLY the outlier head with frozen encoder+classifier:
      - source texts as "known" (target=0)
      - politics texts as "unknown" (target=1)

    Purpose:
      - make OE signal usable for open-set detection,
      - avoid negative transfer to the classification boundary in closed-set.
    """
    if politics_texts is None or len(politics_texts) == 0:
        return model

    print("\n[OE-Calib] Calibrating outlier head (encoder & classifier frozen)...")
    model.to(cfg.device)
    model.train()

    # save requires_grad and freeze
    saved_flags = {}
    for name, p in model.named_parameters():
        saved_flags[name] = p.requires_grad

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False
    for p in model.outlier_head.parameters():
        p.requires_grad = True
    if model.use_domain:
        for p in model.domain_head.parameters():
            p.requires_grad = False

    # build a balanced small calibration set
    n_each = min(len(src_texts), len(politics_texts), 2000)
    if n_each <= 0:
        return model
    src_sub = src_texts[:n_each]
    pol_sub = politics_texts[:n_each]

    texts = src_sub + pol_sub
    y = np.concatenate([np.zeros((n_each,), dtype=np.float32), np.ones((n_each,), dtype=np.float32)], axis=0)

    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=cfg.max_length)
    # reuse TokenizedTextDataset but treat `labels` as outlier targets (0/1)
    ds = TokenizedTextDataset(enc, labels=y.astype(int).tolist())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    opt = torch.optim.AdamW(model.outlier_head.parameters(), lr=5e-4, weight_decay=0.0)

    # a few steps are enough
    steps = 0
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(cfg.device)
        attn = batch["attention_mask"].to(cfg.device)

        yb = batch["labels"].to(cfg.device).float()

        out = model(input_ids=input_ids, attention_mask=attn)
        outlier_logit = out["outlier_logit"]
        loss = F.binary_cross_entropy_with_logits(outlier_logit, yb)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.outlier_head.parameters(), cfg.grad_clip)
        opt.step()

        losses.append(float(loss.item()))
        steps += 1
        if steps >= 200:  # hard cap for speed
            break

    print(f"  [OE-Calib] steps={steps} loss={float(np.mean(losses)):.4f}")

    # restore requires_grad
    for name, p in model.named_parameters():
        p.requires_grad = saved_flags.get(name, True)

    return model
def run_method(method: str,
               base_model_name: str,
               dapt_path: Optional[str],
               tokenizer: AutoTokenizer,
               data: Dict[str, Any],
               cfg: Config) -> Dict[str, Any]:
    """
    Returns:
      dict with method name, metrics, and key settings.
    """
    K = data["K"]
    source_prior = data["source_prior"]
    src_train_ds = data["src_train_ds"]
    src_val_ds = data["src_val_ds"]
    src_test_ds = data["src_test_ds"]
    tgt_train_ds = data["tgt_train_ds"]
    tgt_val_ds = data["tgt_val_ds"]
    tgt_test_ds = data["tgt_test_ds"]
    tgt_train_texts = data["tgt_train_texts"]
    politics_texts = data.get("politics_texts")
    y_tgt_test = data["y_tgt_test"]  # mapped target labels (unknown->K)

    # default inference correction
    alpha = None
    tau_reject = 0.0
    reject_enabled = False
    temp_scaler = None

    # choose encoder init
    encoder_path = base_model_name
    if method in ["dapt_source", "labelshift_em", "entropy_min", "selftrain_closed", "pas_unida_pp", "dann"] and dapt_path is not None:
        encoder_path = dapt_path

    # build model
    if method == "dann":
        model = MultiHeadNLPModel(encoder_path, num_classes=K, use_domain=True, dropout=0.1)
    else:
        model = MultiHeadNLPModel(encoder_path, num_classes=K, use_domain=False, dropout=0.1)

    # ---- Train per method ----
    if method == "source_only":
        model = train_supervised_source(model, src_train_ds, src_val_ds, cfg, politics_texts=None, tokenizer=tokenizer)

    elif method == "dapt_source":
        model = train_supervised_source(model, src_train_ds, src_val_ds, cfg, politics_texts=None, tokenizer=tokenizer)

    elif method == "labelshift_em":
        model = train_supervised_source(model, src_train_ds, src_val_ds, cfg, politics_texts=None, tokenizer=tokenizer)

        # temp scaling for stable EM
        if cfg.enable_temp_scaling:
            src_val_loader = DataLoader(src_val_ds, batch_size=cfg.src_batch_size, shuffle=False)
            logits_val, labels_val = collect_logits_labels(model, src_val_loader, cfg.device)
            fit_dev = "cpu" if cfg.device == "mps" else cfg.device
            temp_scaler = fit_temperature_scaler(logits_val, labels_val, max_iter=cfg.temp_scaling_max_iter, device=fit_dev).to(cfg.device)

        # estimate alpha on target train (assume all known; q=1)
        logits_t, _ = forward_logits_outlier(model, tgt_train_ds, batch_size=cfg.src_batch_size, device=cfg.device)
        if temp_scaler is not None:
            lt = torch.tensor(logits_t, dtype=torch.float32).to(cfg.device)
            logits_scaled = temp_scaler(lt).detach().cpu().numpy()
        else:
            logits_scaled = logits_t
        p = softmax_np(logits_scaled, axis=-1)
        q_all = np.ones((p.shape[0],), dtype=np.float64)
        w_t = estimate_label_shift_em(p, q_all, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol)
        alpha = (w_t / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])

    elif method == "entropy_min":
        model = train_supervised_source(model, src_train_ds, src_val_ds, cfg, politics_texts=None, tokenizer=tokenizer)
        model = train_entropy_minimization(model, src_train_ds, tgt_train_ds, cfg)

    elif method == "selftrain_closed":
        model = train_supervised_source(model, src_train_ds, src_val_ds, cfg, politics_texts=None, tokenizer=tokenizer)
        model = train_selftraining_closed(model, src_train_ds, tgt_train_texts, tokenizer, cfg, tau=cfg.st_tau)

    elif method == "dann":
        # DANN usually trained with source+target from the start
        # For a fairer comparison, we warm-start with source supervised for 1 epoch (optional)
        warm_cfg = Config(**{**cfg.__dict__})
        warm_cfg.src_epochs = 1
        model = train_supervised_source(model, src_train_ds, src_val_ds, warm_cfg, politics_texts=None, tokenizer=tokenizer)
        model = train_dann(model, src_train_ds, tgt_train_ds, cfg)

    elif method == "pas_unida_pp":
        # PAS-UniDA++ v3 (AutoMix):
        #   - Always start from a clean source-supervised model (NO politics mixed into CE training).
        #   - Calibrate outlier head with frozen encoder (optional, only if politics.csv exists).
        #   - Auto-diagnose open-set vs closed-set.
        #   - Closed-set: automatically choose among {none, InfoMax, InfoMax+ST} by unlabeled proxy on target val.
        #   - Open-set: run 3-bucket PAS pipeline (tau + safe self-training).

        use_oe = bool(politics_texts is not None and len(politics_texts) > 0 and cfg.use_politics_oe)

        # 1) Source supervised fine-tuning (avoid negative transfer from OE in closed-set)
        model = train_supervised_source(model, src_train_ds, src_val_ds, cfg,
                                        politics_texts=None,
                                        tokenizer=tokenizer)

        # 1b) Outlier head calibration (frozen encoder) if OE data exists
        if use_oe:
            model = calibrate_outlier_head_frozen(
                model=model,
                tokenizer=tokenizer,
                src_texts=data["src_train_texts"],
                politics_texts=politics_texts,
                cfg=cfg,
            )

        # 1c) Temperature scaling for stable q/proxy
        if cfg.enable_temp_scaling:
            src_val_loader = DataLoader(src_val_ds, batch_size=cfg.src_batch_size, shuffle=False)
            logits_val, labels_val = collect_logits_labels(model, src_val_loader, cfg.device)
            fit_dev = "cpu" if cfg.device == "mps" else cfg.device
            temp_scaler = fit_temperature_scaler(logits_val, labels_val, max_iter=cfg.temp_scaling_max_iter, device=fit_dev).to(cfg.device)

        # 2) Prepare prototypes and run auto selection (decide open_set, tau, mode, alpha)
        src_feats = forward_features(model, src_train_ds, batch_size=cfg.src_batch_size, device=cfg.device)
        src_labels_np = np.array(data["src_train_labels"], dtype=int)

        auto = auto_select_tau_and_mode(
            model=model,
            temp_scaler=temp_scaler,
            source_feats=src_feats,
            source_labels=src_labels_np,
            source_prior=source_prior,
            target_ds=tgt_train_ds,
            cfg=cfg,
            use_oe=use_oe,
        )
        mode = auto["mode"]
        tau_pseudo = auto["tau_pseudo"]
        tau_reject = auto["tau_reject"]
        reject_enabled = auto["reject_enabled"]
        open_set_flag = auto["open_set"]
        alpha = auto["alpha"]

        # 3) Closed-set AutoMix path: prefer strong closed-set baselines (InfoMax) and avoid negative transfer
        if not open_set_flag:
            print("\n[Phase 3] Closed-set detected -> AutoMix selection among {none, InfoMax, InfoMax+ST}...")

            base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            candidates: List[Tuple[str, Dict[str, torch.Tensor], Dict[str, float]]] = []

            # Candidate A: none (source-only after DAPT if enabled)
            proxy_none = infomax_proxy_on_dataset(model, tgt_val_ds, cfg, temp_scaler=temp_scaler)
            candidates.append(("none", base_state, proxy_none))

            # Candidate B: InfoMax
            model.load_state_dict(base_state)
            model = train_entropy_minimization(model, src_train_ds, tgt_train_ds, cfg)
            proxy_im = infomax_proxy_on_dataset(model, tgt_val_ds, cfg, temp_scaler=temp_scaler)
            state_im = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            candidates.append(("infomax", state_im, proxy_im))

            # Candidate C: InfoMax + (small) self-training (often harmless when InfoMax is already strong)
            model.load_state_dict(state_im)
            post_cfg = Config(**{**cfg.__dict__})
            post_cfg.st_rounds = min(2, cfg.st_rounds)  # keep it short & safe
            model = train_selftraining_closed(model, src_train_ds, tgt_train_texts, tokenizer, post_cfg, tau=post_cfg.st_tau)
            proxy_imst = infomax_proxy_on_dataset(model, tgt_val_ds, cfg, temp_scaler=temp_scaler)
            state_imst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            candidates.append(("infomax+st", state_imst, proxy_imst))

            # Choose by unlabeled proxy (higher better)
            best_name, best_state, best_p = None, None, -1e18
            print("  [AutoMix] candidate proxies (target val, unlabeled):")
            for name, st, px in candidates:
                print(f"    - {name:10s} proxy={px['proxy']:.4f}  mean_ent={px['mean_ent']:.4f}  div_ent={px['div_ent']:.4f}  "
                      f"min_p={px['min_class_prob']:.3f} max_p={px['max_class_prob']:.3f}")
                if px["proxy"] > best_p:
                    best_p = px["proxy"]
                    best_name = name
                    best_state = st

            assert best_state is not None
            model.load_state_dict(best_state)
            print(f"  [AutoMix] selected={best_name} best_proxy={best_p:.4f}")

            # Closed-set: disable rejection at inference
            reject_enabled = False
            tau_reject = 0.0

            # Closed-set: do NOT apply label-shift unless explicitly selected (often harmful under non-label-shift)
            if mode == "em":
                # re-estimate alpha with the selected model
                stats_final = compute_q_known(model, temp_scaler, tgt_train_ds, cfg, use_oe=False)
                p_final = stats_final["p"]
                q_final = np.ones((p_final.shape[0],), dtype=np.float64)
                w_t_final = estimate_label_shift_em(
                    p_final, q_final, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol
                )
                alpha = (w_t_final / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])
            else:
                alpha = np.ones((K,), dtype=np.float64)

        else:
            # 4) Open-set path: run PAS 3-bucket safe self-training
            print("\n[Phase 3] Open-set detected -> running PAS 3-bucket self-training...")

            # iterative self-training (select best by proxy on target val)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_proxy = -1e18
            best_round = -1

            for r in range(cfg.ours_rounds):
                stats = compute_q_known(model, temp_scaler, tgt_train_ds, cfg, use_oe=use_oe)
                p = stats["p"]
                q = stats["q"]
                q_oe = stats["q_oe"]
                prob_unk_oe = stats["prob_unk_oe"]

                if mode == "em":
                    w_t = estimate_label_shift_em(p, q, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol)
                    alpha = (w_t / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])
                    p_adj = apply_label_shift(p, alpha)
                else:
                    alpha = np.ones((K,), dtype=np.float64)
                    p_adj = p

                st_ds, st_info = build_selftrain_dataset_ours(
                    tokenizer=tokenizer,
                    cfg=cfg,
                    source_texts=data["src_train_texts"],
                    source_labels=data["src_train_labels"],
                    target_texts=tgt_train_texts,
                    p_adj=p_adj,
                    q=q,
                    q_oe=q_oe,
                    prob_unk_oe=prob_unk_oe,
                    alpha=alpha,
                    tau_pseudo=tau_pseudo,
                    open_set_flag=open_set_flag,
                    round_idx=r,
                    politics_texts=politics_texts if use_oe else None,
                )
                print(f"  [OURS data] round={r+1}/{cfg.ours_rounds} {st_info}")

                model = train_one_round_ours(model, st_ds, cfg, open_set_flag=open_set_flag)

                proxy = proxy_score_ours(
                    model=model,
                    temp_scaler=temp_scaler,
                    source_feats=src_feats,
                    source_labels=src_labels_np,
                    source_prior=source_prior,
                    target_val_ds=tgt_val_ds,
                    cfg=cfg,
                    use_oe=use_oe,
                    mode=mode,
                    tau_pseudo=tau_pseudo,
                    open_set_flag=open_set_flag,
                )
                print(f"  [OURS proxy@val] {proxy}")

                if proxy["proxy"] > best_proxy:
                    best_proxy = proxy["proxy"]
                    best_round = r
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            print(f"  [OURS] selected round={best_round} best_proxy={best_proxy:.4f}")

            if mode == "em":
                stats_final = compute_q_known(model, temp_scaler, tgt_train_ds, cfg, use_oe=use_oe)
                p_final = stats_final["p"]
                q_final = stats_final["q"]
                w_t_final = estimate_label_shift_em(
                    p_final, q_final, source_prior, max_iter=cfg.label_shift_max_iter, tol=cfg.label_shift_tol
                )
                alpha = (w_t_final / (source_prior + 1e-12)).clip(cfg.alpha_clip[0], cfg.alpha_clip[1])
            else:
                alpha = np.ones((K,), dtype=np.float64)

        # Save final model (ours) for later reuse
        save_dir = os.path.join(cfg.output_dir, "final_pas_unida_pp")
        safe_mkdir(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))
        try:
            model.encoder.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        except Exception:
            pass
        meta = {
            "method": "pas_unida_pp",
            "mode": mode,
            "tau_pseudo": float(tau_pseudo),
            "reject_enabled": bool(reject_enabled),
            "tau_reject": float(tau_reject),
            "open_set": bool(open_set_flag),
        }
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    if temp_scaler is None and cfg.enable_temp_scaling and method in ["source_only", "dapt_source", "entropy_min", "selftrain_closed", "dann"]:
        src_val_loader = DataLoader(src_val_ds, batch_size=cfg.src_batch_size, shuffle=False)
        logits_val, labels_val = collect_logits_labels(model, src_val_loader, cfg.device)
        fit_dev = "cpu" if cfg.device == "mps" else cfg.device
        temp_scaler = fit_temperature_scaler(logits_val, labels_val, max_iter=cfg.temp_scaling_max_iter, device=fit_dev).to(cfg.device)

    y_pred, conf = predict_labels(
        model=model,
        dataset=tgt_test_ds,
        device=cfg.device,
        temp_scaler=temp_scaler,
        alpha=alpha,
        reject_enabled=reject_enabled,
        tau_reject=tau_reject,
    )
    metrics = evaluate_on_target(y_true=y_tgt_test, y_pred=y_pred, num_known=K)

    # also report source test acc for reference
    src_test_acc = evaluate_source_acc(model, src_test_ds, batch_size=cfg.src_batch_size, device=cfg.device)

    result = {
        "method": method,
        "src_test_acc": float(src_test_acc),
        "tgt_test_acc": float(metrics["acc"]),
        "tgt_test_macro_f1": float(metrics["macro_f1"]),
        "tgt_test_bal_acc": float(metrics["bal_acc"]),
        "tgt_pred_unknown_rate": float(metrics["pred_unknown_rate"]),
        "reject_enabled": bool(reject_enabled),
        "tau_reject": float(tau_reject),
    }
    return result



# ============================================================
# 16) Data leakage / duplicate checks
# ============================================================
def _normalize_text_for_hash(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _hash_text(s: str) -> str:
    norm = _normalize_text_for_hash(s)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _hash_texts(texts: List[str]) -> Tuple[List[str], Dict[str, str]]:
    hs: List[str] = []
    ex: Dict[str, str] = {}
    for t in texts:
        h = _hash_text(t)
        hs.append(h)
        if h not in ex:
            ex[h] = t
    return hs, ex


def _dup_stats(name: str, hashes: List[str]) -> Dict[str, Any]:
    n = len(hashes)
    u = len(set(hashes))
    dup = n - u
    return {
        "split": name,
        "n": int(n),
        "unique": int(u),
        "duplicate": int(dup),
        "dup_ratio": float(dup / max(1, n)),
    }


def _overlap_stats(name_a: str, hashes_a: List[str], ex_a: Dict[str, str],
                   name_b: str, hashes_b: List[str], ex_b: Dict[str, str],
                   max_show: int = 3) -> Dict[str, Any]:
    set_a = set(hashes_a)
    set_b = set(hashes_b)
    inter = list(set_a.intersection(set_b))
    inter.sort()
    count = len(inter)
    ratio_a = count / max(1, len(set_a))
    ratio_b = count / max(1, len(set_b))

    examples = []
    for h in inter[:max_show]:
        ta = ex_a.get(h, "")
        tb = ex_b.get(h, "")
        # show a short snippet (prefer a)
        t = ta if ta else tb
        t = _normalize_text_for_hash(t)
        examples.append(t[:160])

    return {
        "pair": f"{name_a}__INTERSECT__{name_b}",
        "overlap_unique": int(count),
        "ratio_in_a_unique": float(ratio_a),
        "ratio_in_b_unique": float(ratio_b),
        "examples_norm_head": examples,
    }


def run_data_leakage_check(src_train_texts: List[str],
                           src_val_texts: List[str],
                           src_test_texts: List[str],
                           tgt_train_texts: List[str],
                           tgt_val_texts: List[str],
                           tgt_test_texts: List[str],
                           out_dir: str) -> Dict[str, Any]:
    """
    Exact-duplicate and cross-split overlap check (after light normalization).
    This is a cheap but very effective leakage/duplication sanity check.
    """
    print("\n" + "=" * 80)
    print("[Check] Data leakage / duplicate sample sanity check (exact match after normalization)")
    safe_mkdir(out_dir)

    splits = {
        "source_train": src_train_texts,
        "source_val": src_val_texts,
        "source_test": src_test_texts,
        "target_train": tgt_train_texts,
        "target_val": tgt_val_texts,
        "target_test": tgt_test_texts,
    }

    hashes: Dict[str, List[str]] = {}
    examples: Dict[str, Dict[str, str]] = {}
    dup_stats = []
    for name, texts in splits.items():
        hs, ex = _hash_texts(texts)
        hashes[name] = hs
        examples[name] = ex
        st = _dup_stats(name, hs)
        dup_stats.append(st)
        print(f"  - {name:12s}: n={st['n']:5d} unique={st['unique']:5d} dup={st['duplicate']:5d} dup_ratio={st['dup_ratio']:.4f}")

    pairs = [
        ("source_train", "source_test"),
        ("source_train", "source_val"),
        ("source_val", "source_test"),
        ("target_train", "target_test"),
        ("target_val", "target_test"),
        ("target_train", "target_val"),
        ("source_train", "target_test"),
        ("source_train", "target_train"),
    ]

    overlaps = []
    print("\n  [Overlap] unique-text overlap between splits:")
    for a, b in pairs:
        ov = _overlap_stats(a, hashes[a], examples[a], b, hashes[b], examples[b], max_show=3)
        overlaps.append(ov)
        print(
            f"  - {a:12s} ∩ {b:12s}: overlap_unique={ov['overlap_unique']:4d} "
            f"ratio_in_{a}={ov['ratio_in_a_unique']:.4f} ratio_in_{b}={ov['ratio_in_b_unique']:.4f}"
        )
        if ov["overlap_unique"] > 0 and ov["examples_norm_head"]:
            for i, ex in enumerate(ov["examples_norm_head"], 1):
                print(f"      example#{i}: {ex}")

    report = {
        "dup_stats": dup_stats,
        "overlaps": overlaps,
        "note": "Exact match after normalization (lower+strip+collapse spaces). For near-duplicate, consider simhash/embedding."
    }

    out_path = os.path.join(out_dir, "data_leakage_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  [Save] data leakage report saved to: {out_path}")
    return report


# ============================================================
# 17) Multi-seed runner and aggregation
# ============================================================
def _cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_seed(seed: int,
                 cfg_template: Config,
                 model_name: str,
                 tokenizer: AutoTokenizer,
                 data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run all METHODS_TO_RUN for one random seed, return list of result dicts.
    """
    # per-seed output directory
    root_dir = cfg_template.output_dir
    seed_dir = os.path.join(root_dir, f"seed_{seed}")
    cfg = replace(cfg_template, seed=seed, output_dir=seed_dir)
    safe_mkdir(cfg.output_dir)

    print("\n" + "=" * 80)
    print(f"[Seed] Running seed={seed}  output_dir={cfg.output_dir}")
    seed_everything(seed)

    # DAPT once for this seed (shared by methods)
    dapt_path = None
    if cfg.enable_dapt and any(m in METHODS_TO_RUN for m in ["dapt_source", "labelshift_em", "entropy_min", "selftrain_closed", "pas_unida_pp", "dann"]):
        dapt_path = train_dapt_mlm(model_name, tokenizer, data["tgt_train_texts"], cfg)

    results = []
    for m in METHODS_TO_RUN:
        print("\n" + "-" * 80)
        print(f"[Seed {seed}] Method: {m}")
        seed_everything(seed)  # reset seed per method for fair compare inside this seed
        res = run_method(m, model_name, dapt_path, tokenizer, data, cfg)
        res["seed"] = int(seed)
        results.append(res)
        print(f"[Seed {seed}] Done {m} -> tgt_acc={res['tgt_test_acc']:.4f} tgt_f1={res['tgt_test_macro_f1']:.4f}")
        _cleanup_cuda()

    # save per-seed results
    results_sorted = sorted(results, key=lambda x: x["tgt_test_acc"], reverse=True)
    out_path = os.path.join(cfg.output_dir, "comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    print(f"\n[Seed {seed}] Saved seed results to: {out_path}")
    return results_sorted


def aggregate_multi_seed(all_seed_results: Dict[int, List[Dict[str, Any]]],
                         methods: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      - summary_rows: list of per-method mean/std rows
      - win_info: ours vs entropy_min win-rate info
    """
    metrics = ["tgt_test_acc", "tgt_test_macro_f1", "tgt_test_bal_acc", "src_test_acc"]
    per_method: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in metrics} for m in methods}

    # fill
    for seed, rows in all_seed_results.items():
        row_map = {r["method"]: r for r in rows}
        for m in methods:
            if m not in row_map:
                continue
            for k in metrics:
                per_method[m][k].append(float(row_map[m][k]))

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        if len(vals) == 0:
            return 0.0, 0.0
        arr = np.array(vals, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
        return mean, std

    summary = []
    for m in methods:
        row = {"method": m, "n_seeds": int(len(per_method[m]["tgt_test_acc"]))}
        for k in metrics:
            mu, sd = mean_std(per_method[m][k])
            row[f"{k}_mean"] = mu
            row[f"{k}_std"] = sd
        summary.append(row)

    summary_sorted = sorted(summary, key=lambda x: x["tgt_test_acc_mean"], reverse=True)

    # win-rate: pas_unida_pp vs entropy_min (by tgt_test_acc)
    wins = ties = losses = 0
    deltas = []
    for seed, rows in all_seed_results.items():
        row_map = {r["method"]: r for r in rows}
        if "pas_unida_pp" not in row_map or "entropy_min" not in row_map:
            continue
        ours = float(row_map["pas_unida_pp"]["tgt_test_acc"])
        base = float(row_map["entropy_min"]["tgt_test_acc"])
        deltas.append(ours - base)
        if ours > base + 1e-12:
            wins += 1
        elif abs(ours - base) <= 1e-12:
            ties += 1
        else:
            losses += 1

    win_info = {
        "compare": "pas_unida_pp_vs_entropy_min",
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "total": int(wins + ties + losses),
        "win_rate": float(wins / max(1, wins + ties + losses)),
        "avg_delta_acc": float(np.mean(deltas)) if deltas else 0.0,
        "std_delta_acc": float(np.std(np.array(deltas), ddof=1)) if len(deltas) >= 2 else 0.0,
        "deltas_acc": [float(x) for x in deltas],
    }
    return summary_sorted, win_info


# ============================================================
# 18) Main (multi-seed)
# ============================================================
def main():
    safe_mkdir(CFG.output_dir)

    # Load dataframes once
    src_train_df = read_csv_safely(SOURCE_TRAIN_PATH)
    src_val_df = read_csv_safely(SOURCE_VAL_PATH)
    src_test_df = read_csv_safely(SOURCE_TEST_PATH)

    tgt_train_df = read_csv_safely(TARGET_TRAIN_PATH)
    tgt_val_df = read_csv_safely(TARGET_VAL_PATH)
    tgt_test_df = read_csv_safely(TARGET_TEST_PATH)

    pol_df = None
    if CFG.use_politics_oe and os.path.exists(POLITICS_PATH):
        try:
            pol_df = read_csv_safely(POLITICS_PATH)
        except Exception:
            pol_df = None

    # Detect columns
    text_override = TEXT_COL_OVERRIDE or CFG.text_col_override
    label_override = LABEL_COL_OVERRIDE or CFG.label_col_override

    src_text_col, src_label_col = detect_text_and_label_columns(src_train_df, text_override=text_override, label_override=label_override)
    print(f"[Data] source text_col='{src_text_col}', label_col='{src_label_col}'")

    tgt_text_col, tgt_label_guess = detect_text_and_label_columns(tgt_train_df, text_override=text_override, label_override=None)
    print(f"[Data] target text_col='{tgt_text_col}' (label col guessed: {tgt_label_guess})")

    # ensure split columns
    src_text_col_val = ensure_text_col(src_val_df, src_text_col)
    src_text_col_test = ensure_text_col(src_test_df, src_text_col)
    tgt_text_col_val = ensure_text_col(tgt_val_df, tgt_text_col)
    tgt_text_col_test = ensure_text_col(tgt_test_df, tgt_text_col)

    if src_label_col is None:
        raise ValueError("Source label column not found. Please set LABEL_COL_OVERRIDE.")

    # Extract texts
    src_train_texts = clean_text_list(src_train_df[src_text_col])
    src_val_texts = clean_text_list(src_val_df[src_text_col_val])
    src_test_texts = clean_text_list(src_test_df[src_text_col_test])

    tgt_train_texts = clean_text_list(tgt_train_df[tgt_text_col])
    tgt_val_texts = clean_text_list(tgt_val_df[tgt_text_col_val])
    tgt_test_texts = clean_text_list(tgt_test_df[tgt_text_col_test])

    # Extract labels (source)
    src_train_labels_raw = src_train_df[src_label_col].tolist()
    src_val_labels_raw = src_val_df[src_label_col].tolist()
    src_test_labels_raw = src_test_df[src_label_col].tolist()

    unique_labels = sorted(list(pd.Series(src_train_labels_raw).dropna().unique()))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    K = len(unique_labels)
    print(f"[Data] num_known_classes(K)={K} labels={unique_labels}")

    def map_label_one(x: Any) -> int:
        if x in label2id:
            return int(label2id[x])
        try:
            xi = int(x)
            if xi in label2id:
                return int(label2id[xi])
        except Exception:
            pass
        xs = str(x)
        for k_raw, idx in label2id.items():
            if str(k_raw) == xs:
                return int(idx)
        return -1

    def map_labels(lst: List[Any]) -> List[int]:
        return [map_label_one(x) for x in lst]

    src_train_labels = map_labels(src_train_labels_raw)
    src_val_labels = map_labels(src_val_labels_raw)
    src_test_labels = map_labels(src_test_labels_raw)

    # source prior
    src_counts = np.bincount(np.array(src_train_labels, dtype=int), minlength=K).astype(np.float64)
    source_prior = src_counts / (src_counts.sum() + 1e-12)

    # Politics texts
    politics_texts = None
    if pol_df is not None:
        pol_text_col, _ = detect_text_and_label_columns(pol_df, text_override=text_override, label_override=None)
        politics_texts = clean_text_list(pol_df[pol_text_col])
        if len(politics_texts) > CFG.politics_max_samples:
            random.shuffle(politics_texts)
            politics_texts = politics_texts[:CFG.politics_max_samples]
        print(f"[Data] politics loaded: n={len(politics_texts)}")
    else:
        print("[Data] politics not found or disabled.")

    # Data leakage / duplication check (once)
    run_data_leakage_check(
        src_train_texts=src_train_texts,
        src_val_texts=src_val_texts,
        src_test_texts=src_test_texts,
        tgt_train_texts=tgt_train_texts,
        tgt_val_texts=tgt_val_texts,
        tgt_test_texts=tgt_test_texts,
        out_dir=CFG.output_dir,
    )

    # Decide model name
    model_name = MODEL_NAME_OVERRIDE or CFG.model_name_override
    if model_name is None:
        model_name = pick_model_name_by_language(src_train_texts[:50] + tgt_train_texts[:50])
    print(f"[Model] base model = {model_name}")
    print(f"[Sys] device = {CFG.device}")

    # Tokenizer once
    tokenizer = load_tokenizer_robust(model_name)

    # Tokenize datasets once
    src_train_enc = tokenizer(src_train_texts, truncation=True, padding="max_length", max_length=CFG.max_length)
    src_val_enc = tokenizer(src_val_texts, truncation=True, padding="max_length", max_length=CFG.max_length)
    src_test_enc = tokenizer(src_test_texts, truncation=True, padding="max_length", max_length=CFG.max_length)

    tgt_train_enc = tokenizer(tgt_train_texts, truncation=True, padding="max_length", max_length=CFG.max_length)
    tgt_val_enc = tokenizer(tgt_val_texts, truncation=True, padding="max_length", max_length=CFG.max_length)
    tgt_test_enc = tokenizer(tgt_test_texts, truncation=True, padding="max_length", max_length=CFG.max_length)

    src_train_ds = TokenizedTextDataset(src_train_enc, src_train_labels)
    src_val_ds = TokenizedTextDataset(src_val_enc, src_val_labels)
    src_test_ds = TokenizedTextDataset(src_test_enc, src_test_labels)

    tgt_train_ds = TokenizedTextDataset(tgt_train_enc, labels=None)
    tgt_val_ds = TokenizedTextDataset(tgt_val_enc, labels=None)
    tgt_test_ds = TokenizedTextDataset(tgt_test_enc, labels=None)

    # Map target test labels for evaluation
    _, tgt_test_label_col = detect_text_and_label_columns(tgt_test_df, text_override=tgt_text_col_test, label_override=None)
    if tgt_test_label_col is not None and tgt_test_label_col in tgt_test_df.columns:
        tgt_test_labels_raw = tgt_test_df[tgt_test_label_col].tolist()
        y_tgt_test = np.array([map_label_one(x) if map_label_one(x) >= 0 else K for x in tgt_test_labels_raw], dtype=int)
        print(f"[Eval] target test label_col='{tgt_test_label_col}' detected. (Unknown mapped to id={K})")
    else:
        y_tgt_test = np.full((len(tgt_test_texts),), -1, dtype=int)
        print("[Eval] target test label column not found -> evaluation will be limited.")

    data = {
        "K": K,
        "source_prior": source_prior,
        "src_train_ds": src_train_ds,
        "src_val_ds": src_val_ds,
        "src_test_ds": src_test_ds,
        "tgt_train_ds": tgt_train_ds,
        "tgt_val_ds": tgt_val_ds,
        "tgt_test_ds": tgt_test_ds,
        "tgt_train_texts": tgt_train_texts,
        "src_train_texts": src_train_texts,
        "src_train_labels": src_train_labels,
        "politics_texts": politics_texts,
        "y_tgt_test": y_tgt_test,
    }

    # Run multi-seed
    all_seed_results: Dict[int, List[Dict[str, Any]]] = {}
    for seed in SEEDS_TO_RUN:
        res_seed = run_one_seed(seed, CFG, model_name, tokenizer, data)
        all_seed_results[int(seed)] = res_seed

    # Aggregate
    summary_rows, win_info = aggregate_multi_seed(all_seed_results, METHODS_TO_RUN)

    print("\n" + "=" * 80)
    print("[Multi-Seed Summary] mean ± std over seeds")
    header = [
        "method", "n",
        "tgt_acc(mean±std)", "tgt_f1(mean±std)", "tgt_bal_acc(mean±std)",
        "src_acc(mean±std)"
    ]
    print("\t".join(header))

    for r in summary_rows:
        print(
            f"{r['method']}\t{r['n_seeds']}\t"
            f"{r['tgt_test_acc_mean']:.4f}±{r['tgt_test_acc_std']:.4f}\t"
            f"{r['tgt_test_macro_f1_mean']:.4f}±{r['tgt_test_macro_f1_std']:.4f}\t"
            f"{r['tgt_test_bal_acc_mean']:.4f}±{r['tgt_test_bal_acc_std']:.4f}\t"
            f"{r['src_test_acc_mean']:.4f}±{r['src_test_acc_std']:.4f}"
        )

    print("\n" + "-" * 80)
    print("[Win-Rate] PAS-UniDA++ vs entropy_min (by target test accuracy)")
    print(
        f"  wins/ties/losses = {win_info['wins']}/{win_info['ties']}/{win_info['losses']} "
        f"(total={win_info['total']}, win_rate={win_info['win_rate']:.3f})"
    )
    print(f"  avg_delta_acc = {win_info['avg_delta_acc']:.6f} ± {win_info['std_delta_acc']:.6f}")

    # Save multi-seed outputs
    out_all = os.path.join(CFG.output_dir, "all_seed_results.json")
    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, ensure_ascii=False, indent=2)
    out_sum = os.path.join(CFG.output_dir, "multi_seed_summary.json")
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "win_info": win_info}, f, ensure_ascii=False, indent=2)

    print(f"\n[Save] all-seed results: {out_all}")
    print(f"[Save] multi-seed summary: {out_sum}")


if __name__ == "__main__":
    main()
