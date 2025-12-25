#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-ready experiment runner:
Binary domain adaptation under compound shift (conditional shift + label shift) with open-set noise
and decision-threshold drift.

This script is designed to generate ALL experiments typically needed for a publishable paper:
  - Clean split check (no overlap between target train/val/test) is assumed (you already did it).
  - Baselines + ablations:
      * Source-only (report @0.5 and @tau*)
      * Naive self-training (unbalanced confidence-based)
      * Top-k balanced pseudo-labeling (student)
      * Top-k balanced pseudo-labeling (EMA teacher)
      * Full method: EMA teacher + consistency + top-k balanced
      * BBSE-prior baseline (uses pi_bbse and quantile threshold) to show fragility under conditional shift
      * (Open-set suite only) with/without OOD filtering for pseudo-labeling
  - Controlled suites:
      * base (clean target, alpha=0, natural prior)
      * open_set (mix unknown politics.csv with target unlabeled at alpha in grid)
      * label_shift (controlled target prior pi in grid; labels used ONLY to construct the split, NOT for training)
      * combined (optional): label shift + open-set noise together

Protocol:
  - Training uses: source labeled + target unlabeled (which may be a mixture of ID and unknown).
  - Model/threshold selection uses: target validation labels (Protocol A).
    This is realistic for deployment and produces the best accuracy, but MUST be stated as using a small labeled
    calibration/dev set in the paper.

Directory layout (place this script in your project folder, e.g. Third/):
  Third/
    paper_uda_experiments.py   (this file)
    sourcedata/
      source_train.csv
      source_validation.csv
      source_test.csv
    targetdata_clean/          (recommended; if missing, falls back to targetdata/)
      train.csv
      val.csv
      test.csv
    politics.csv               (unknown/OOD pool; same schema as above, at least contains 'text')
    runs/
      paper_experiments/       (auto-created outputs)

Run:
  - Click Run in your IDE, or:
      python paper_uda_experiments.py

Outputs:
  runs/paper_experiments/
    per_run/ ... (json logs for each run)
    summary_<suite>.csv        (mean/std aggregated across seeds)
    all_results.csv            (raw per-run results)
"""

import os
import re
import csv
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# User-adjustable switches (paper experiments)
# ============================================================

MODEL_NAME = "bert-base-uncased"

# To reduce runtime, you can shorten these.
SEEDS: List[int] = [13, 17, 19, 23, 29]  # 5 seeds for paper-style reporting
# SEEDS = [13, 17, 19]  # faster

RUN_SUITES = {
    "base": True,
    "open_set": True,
    "label_shift": True,
    "combined": True,   # set True if you want pi x alpha combined grid
}

# Open-set noise levels (alpha) for mixing politics.csv into target unlabeled.
OPENSET_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3]

# Controlled label-shift target priors (pi = P(y=1)).
LABELSHIFT_PI_GRID = [0.2, 0.5, 0.8]

# Combined grid (kept small by default)
COMBINED_ALPHA_GRID = [0.2]
COMBINED_PI_GRID = [0.2, 0.5, 0.8]

# Unknown mixing into evaluation splits (val/test) for open-set suite.
# If you want pure ID evaluation, set to 0.0. If you want open-set metrics, set >0.
OPENSET_EVAL_ALPHA = 0.2

# OOD filter during pseudo-label selection (only meaningful when alpha>0)
# For experiments we evaluate both on/off in open_set suite.
OOD_FILTER_QUANTILE = True
OOD_FILTER_ALPHA_GUESS = None  # None => use current train alpha; else set float like 0.2


# ============================================================
# Training hyperparameters (tuned for Mac MPS stability)
# ============================================================

MAX_LEN = 384
TRAIN_BATCH = 8
PRED_BATCH = 32
GRAD_ACCUM = 2  # effective batch ~ 16
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EPOCHS_SOURCE = 3

ADAPT_ROUNDS = 4
ADAPT_EPOCHS_PER_ROUND = 1

# Pseudo-label schedule (fraction of target unlabeled selected per round)
PSEUDO_FRAC_SCHEDULE = [0.20, 0.25, 0.30, 0.35]

# Unsupervised consistency strength schedule (ramps up)
LAMBDA_U_SCHEDULE = [0.25, 0.50, 0.75, 1.00]

# Teacher EMA decay
EMA_DECAY = 0.999

# Consistency: use only teacher-confident unlabeled examples
CONSISTENCY_CONF_THRESH = 0.55

# Temperature scaling
USE_TEMPERATURE_SCALING = True
RECALIBRATE_EACH_ROUND = True

# For Protocol A: tune tau on target val to maximize Accuracy (ID-only).
TUNE_THRESHOLD_ON_TARGET_VAL = True
SELECT_BEST_BY_TARGET_VAL_ACC = True

# Use fixed prior pi_used = pi_source for stability under strong conditional shift
FIX_PRIOR_TO_SOURCE = True

# If you enable BBSE baseline, it will use BBSE pi and quantile threshold; can be unstable under conditional shift.
ENABLE_BBSE_BASELINE = True


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def safe_json_dump(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_torch_load(path: str, map_location: torch.device) -> Dict[str, torch.Tensor]:
    """Try weights_only=True if available, fallback otherwise."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Robust CSV loaders
# ============================================================

def _open_text(path: str, encoding: str = "utf-8"):
    return open(path, "r", encoding=encoding, errors="replace", newline="")

def load_csv_text_label_robust(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
    sep: str = ",",
    encoding: str = "utf-8",
    allowed_labels: Tuple[int, ...] = (0, 1, -1),
    drop_bad: bool = True,
) -> pd.DataFrame:
    rows: List[Tuple[str, int]] = []
    allowed_set = set(allowed_labels)

    with _open_text(path, encoding=encoding) as f:
        reader = csv.reader(f, delimiter=sep)
        header = next(reader, None)
        if header is None:
            return pd.DataFrame({text_col: [], label_col: []})

        try:
            text_idx = header.index(text_col)
        except ValueError:
            text_idx = 0
        try:
            label_idx = header.index(label_col)
        except ValueError:
            label_idx = 1

        for row in reader:
            if len(row) <= max(text_idx, label_idx):
                if drop_bad:
                    continue
                txt = row[text_idx] if len(row) > text_idx else ""
                rows.append((txt, 0))
                continue

            txt = row[text_idx]
            lab = row[label_idx].strip()

            if re.fullmatch(r"-?\d+(?:\.0+)?", lab or ""):
                lab_int = int(float(lab))
            else:
                if drop_bad:
                    continue
                lab_int = 0

            if lab_int not in allowed_set:
                if drop_bad:
                    continue
                lab_int = 0

            rows.append((txt, lab_int))

    return pd.DataFrame(rows, columns=[text_col, label_col])

def load_csv_text_only_robust(
    path: str,
    text_col: str = "text",
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    rows: List[str] = []
    with _open_text(path, encoding=encoding) as f:
        reader = csv.reader(f, delimiter=sep)
        header = next(reader, None)
        if header is None:
            return pd.DataFrame({text_col: []})
        try:
            text_idx = header.index(text_col)
        except ValueError:
            text_idx = 0
        for row in reader:
            if len(row) <= text_idx:
                continue
            rows.append(row[text_idx])
    return pd.DataFrame({text_col: rows})

def load_table_auto(path: str, text_col: str = "text", label_col: Optional[str] = "label") -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        if label_col is None:
            return load_csv_text_only_robust(path, text_col=text_col, sep=",")
        return load_csv_text_label_robust(path, text_col=text_col, label_col=label_col, sep=",")
    if ext == ".tsv":
        if label_col is None:
            return load_csv_text_only_robust(path, text_col=text_col, sep="\t")
        return load_csv_text_label_robust(path, text_col=text_col, label_col=label_col, sep="\t")
    if ext == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported extension for {path}")

    if text_col not in df.columns:
        raise ValueError(f"Missing {text_col} in {path}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Missing {label_col} in {path}")
    return df


def norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# Dataset
# ============================================================

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, weights: Optional[List[float]] = None):
        self.texts = texts
        self.labels = labels
        self.weights = weights
        if labels is not None and len(labels) != len(texts):
            raise ValueError("labels length mismatch")
        if weights is not None and len(weights) != len(texts):
            raise ValueError("weights length mismatch")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {"text": self.texts[idx]}
        if self.labels is not None:
            d["labels"] = int(self.labels[idx])
        if self.weights is not None:
            d["weights"] = float(self.weights[idx])
        return d


def make_collate_fn(tokenizer, max_len: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        if "labels" in batch[0]:
            enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        if "weights" in batch[0]:
            enc["weights"] = torch.tensor([b.get("weights", 1.0) for b in batch], dtype=torch.float)
        else:
            enc["weights"] = torch.ones(len(batch), dtype=torch.float)
        return enc
    return collate


def make_loader(tokenizer, texts: List[str], labels: Optional[List[int]], weights: Optional[List[float]],
                batch_size: int, max_len: int, shuffle: bool, num_workers: int = 0) -> torch.utils.data.DataLoader:
    ds = TextDataset(texts, labels=labels, weights=weights)
    collate_fn = make_collate_fn(tokenizer, max_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


# ============================================================
# Temperature scaling
# ============================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.clamp(self.temperature, min=1e-3)
        return logits / T

    @torch.no_grad()
    def get_T(self) -> float:
        return float(torch.clamp(self.temperature, min=1e-3).item())

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        """LBFGS fit; fallback to Adam if LBFGS fails on some backends."""
        device = logits.device
        self.to(device)
        nll = nn.CrossEntropyLoss()

        try:
            optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)

            def closure():
                optimizer.zero_grad()
                loss = nll(self.forward(logits), labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            return self.get_T()
        except Exception:
            # Fallback: simple Adam steps
            opt = torch.optim.Adam([self.temperature], lr=0.05)
            for _ in range(max_iter * 5):
                opt.zero_grad()
                loss = nll(self.forward(logits), labels)
                loss.backward()
                opt.step()
            return self.get_T()


# ============================================================
# Metrics
# ============================================================

def softmax_p1(logits: torch.Tensor) -> np.ndarray:
    logits = logits.detach()
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs[:, 1]

def score_maxsoft(p1: np.ndarray) -> np.ndarray:
    return np.maximum(p1, 1.0 - p1)

def eval_id_metrics(y: np.ndarray, p1: np.ndarray, tau: float) -> Dict[str, float]:
    mask = np.isin(y, [0, 1])
    yk = y[mask]
    pk = p1[mask]
    pred = (pk >= tau).astype(int)
    return {
        "id_n": int(len(yk)),
        "id_acc": float(accuracy_score(yk, pred)) if len(yk) else float("nan"),
        "id_bal_acc": float(balanced_accuracy_score(yk, pred)) if len(yk) else float("nan"),
        "id_macro_f1": float(f1_score(yk, pred, average="macro")) if len(yk) else float("nan"),
    }

def best_tau_for_accuracy(y: np.ndarray, p1: np.ndarray, grid_size: int = 400) -> Tuple[float, float]:
    """Pick tau that maximizes ID accuracy on y in {0,1}."""
    mask = np.isin(y, [0, 1])
    yk = y[mask]
    pk = p1[mask]
    if len(yk) == 0:
        return 0.5, float("nan")
    # Use a mixture of quantiles for stable search
    qs = np.linspace(0.01, 0.99, grid_size)
    taus = np.quantile(pk, qs)
    best_tau, best_acc = 0.5, -1.0
    for tau in taus:
        pred = (pk >= tau).astype(int)
        acc = accuracy_score(yk, pred)
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)
    return best_tau, float(best_acc)

def ood_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """
    y: -1 for unknown, 0/1 for known.
    score: higher => more ID-like (e.g., max-softmax).
    Returns AUROC(known vs unknown) and FPR@95TPR.
    """
    known = (y != -1).astype(int)  # known=1, unknown=0
    if len(np.unique(known)) < 2:
        return {"ood_auroc": float("nan"), "ood_fpr95": float("nan")}
    try:
        auroc = roc_auc_score(known, score)
    except Exception:
        auroc = float("nan")

    # FPR@95TPR: threshold on score to get TPR>=0.95 on known, then report FPR on unknown
    # Compute thresholds from known scores.
    s_known = score[known == 1]
    s_unk = score[known == 0]
    if len(s_known) == 0 or len(s_unk) == 0:
        return {"ood_auroc": float(auroc), "ood_fpr95": float("nan")}
    thr = np.quantile(s_known, 0.05)  # keep top 95% as known
    tpr = float((s_known >= thr).mean())
    fpr = float((s_unk >= thr).mean())
    # tpr should be ~0.95 by construction (approx).
    return {"ood_auroc": float(auroc), "ood_fpr95": float(fpr), "ood_thr95": float(thr), "ood_tpr": float(tpr)}

def confusion(y: np.ndarray, p1: np.ndarray, tau: float) -> Dict[str, int]:
    mask = np.isin(y, [0, 1])
    yk = y[mask]
    pk = p1[mask]
    pred = (pk >= tau).astype(int)
    cm = confusion_matrix(yk, pred, labels=[0,1])
    # cm rows true, cols pred
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


# ============================================================
# Prediction helpers
# ============================================================

@torch.inference_mode()
def predict_logits(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    outs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0)

@torch.inference_mode()
def predict_p1(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device,
               scaler: Optional[TemperatureScaler] = None) -> np.ndarray:
    logits = predict_logits(model, loader, device)
    if scaler is not None:
        scaler.eval()
        with torch.no_grad():
            logits = scaler(logits.to(device)).detach().cpu()
    return softmax_p1(logits)


# ============================================================
# Training (supervised + adaptation)
# ============================================================

@dataclass
class TrainState:
    best_ckpt: str
    best_metric: float
    best_epoch: int

def train_supervised(
    model: nn.Module,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    train_weights: Optional[List[float]],
    val_texts: List[str],
    val_labels: List[int],
    device: torch.device,
    out_dir: str,
    stage_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
) -> TrainState:
    ensure_dir(out_dir)
    tr_loader = make_loader(tokenizer, train_texts, train_labels, train_weights, batch_size, MAX_LEN, shuffle=True)
    va_loader = make_loader(tokenizer, val_texts, val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, epochs * math.ceil(len(tr_loader) / max(1, grad_accum)))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / max(1, warmup_steps), 1.0)
        if step < warmup_steps else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )

    ce = nn.CrossEntropyLoss(reduction="none")
    best_f1 = -1.0
    best_path = os.path.join(out_dir, f"{stage_name}_best.pt")
    best_epoch = 0

    global_step = 0
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(tr_loader, desc=f"Train {stage_name} e{ep}/{epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss_vec = ce(logits, labels)
            loss = (loss_vec * weights).mean()
            loss = loss / max(1, grad_accum)
            loss.backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        # val
        model.eval()
        val_p1 = predict_p1(model, va_loader, device, scaler=None)
        met = eval_id_metrics(np.array(val_labels, dtype=int), val_p1, tau=0.5)
        f1 = met["id_macro_f1"]
        if f1 > best_f1:
            best_f1 = float(f1)
            best_epoch = ep
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(safe_torch_load(best_path, map_location=device))
    return TrainState(best_ckpt=best_path, best_metric=float(best_f1), best_epoch=int(best_epoch))


def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=(1.0 - decay))


def kl_divergence_with_logits(p_teacher: torch.Tensor, logits_student: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """KL( teacher || student ) with temperature."""
    p_t = torch.softmax(p_teacher / temp, dim=-1)
    log_p_s = torch.log_softmax(logits_student / temp, dim=-1)
    # KL = sum p_t * (log p_t - log p_s). log p_t is constant for grads, so cross-entropy is enough.
    ce = -(p_t * log_p_s).sum(dim=-1)  # per-sample cross-entropy
    return ce


# ============================================================
# Pseudo-label selection + OOD filtering
# ============================================================

def topk_balanced_indices(p1: np.ndarray, k_pos: int, k_neg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices and labels (0/1) using top-k (pos) and bottom-k (neg) by p1."""
    n = len(p1)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(p1)  # ascending
    neg_idx = order[:max(0, k_neg)]
    pos_idx = order[::-1][:max(0, k_pos)]
    idx = np.concatenate([pos_idx, neg_idx], axis=0).astype(int)
    y = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(neg_idx), dtype=int)], axis=0)
    return idx, y

def confidence_weights(p1: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weight by margin to the opposite side (higher = more confident)."""
    p = np.clip(p1, 1e-6, 1-1e-6)
    # if y==1 weight ~ p ; if y==0 weight ~ (1-p)
    w = np.where(y == 1, p, 1.0 - p)
    # normalize to [0,1]
    w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    return w.astype(np.float32)

def ood_filter_mask_by_quantile(score: np.ndarray, alpha_guess: float) -> np.ndarray:
    """Treat lowest alpha_guess quantile as OOD, keep the rest as ID."""
    alpha_guess = float(np.clip(alpha_guess, 0.0, 0.9))
    if len(score) == 0:
        return np.array([], dtype=bool)
    thr = float(np.quantile(score, alpha_guess))
    return score >= thr

def build_pseudo_pool(
    p1: np.ndarray,
    pi_used: float,
    pseudo_frac: float,
    seed: int,
    id_mask: Optional[np.ndarray] = None,
    strategy: str = "topk_balanced",
) -> Tuple[Dict[int, Tuple[int, float]], Dict[str, Any]]:
    """
    Returns pseudo_pool: idx -> (label, weight)
    """
    rng = np.random.RandomState(seed)
    n = len(p1)
    if id_mask is None:
        id_mask = np.ones(n, dtype=bool)
    id_idx = np.where(id_mask)[0]
    p_id = p1[id_mask]
    n_id = len(p_id)

    total = int(round(pseudo_frac * n))
    total = min(total, n_id)
    pi_used = float(np.clip(pi_used, 1e-6, 1-1e-6))
    k_pos = int(round(total * pi_used))
    k_neg = total - k_pos

    if strategy == "topk_balanced":
        sel_local, sel_y = topk_balanced_indices(p_id, k_pos=k_pos, k_neg=k_neg)
    elif strategy == "naive_conf":
        # pick most confident points by score (max-softmax) without balancing
        score = score_maxsoft(p_id)
        order = np.argsort(-score)
        sel_local = order[:total]
        sel_y = (p_id[sel_local] >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown pseudo strategy: {strategy}")

    sel_global = id_idx[sel_local]
    w = confidence_weights(p1[sel_global], sel_y)
    # shuffle to avoid any bias
    perm = rng.permutation(len(sel_global))
    sel_global = sel_global[perm]
    sel_y = sel_y[perm]
    w = w[perm]

    pool = {int(i): (int(y), float(wi)) for i, y, wi in zip(sel_global, sel_y, w)}
    info = {
        "strategy": strategy,
        "n_total": int(n),
        "n_id": int(n_id),
        "pseudo_frac": float(pseudo_frac),
        "total_budget": int(total),
        "k_pos": int(k_pos),
        "k_neg": int(k_neg),
        "selected": int(len(pool)),
        "selected_pos": int(sum(1 for _, (yy, _) in pool.items() if yy == 1)),
        "selected_neg": int(sum(1 for _, (yy, _) in pool.items() if yy == 0)),
    }
    return pool, info


# ============================================================
# Data construction for suites
# ============================================================

@dataclass
class DataBundle:
    # Source (always labeled)
    src_train_texts: List[str]
    src_train_labels: List[int]
    src_val_texts: List[str]
    src_val_labels: List[int]
    src_test_texts: List[str]
    src_test_labels: List[int]
    # Target splits (ID only, labeled for val/test; train may also have labels but treated as unlabeled)
    tgt_train_df: pd.DataFrame   # may contain labels
    tgt_train_has_labels: bool
    tgt_val_df: pd.DataFrame
    tgt_test_df: pd.DataFrame
    # Unknown pool (politics) (labels ignored)
    unk_df: Optional[pd.DataFrame]


def resample_to_prior(df: pd.DataFrame, pi: float, n: int, seed: int, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df0 = df[df[label_col].astype(int) == 0]
    df1 = df[df[label_col].astype(int) == 1]
    n1 = int(round(pi * n))
    n0 = n - n1
    if len(df0) == 0 or len(df1) == 0:
        # fallback: just sample with replacement from whole df
        return df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    s0 = df0.sample(n=n0, replace=True, random_state=seed)
    s1 = df1.sample(n=n1, replace=True, random_state=seed + 1)
    out = pd.concat([s0, s1], axis=0).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return out

def mix_unknown(df_id: pd.DataFrame, df_unk: pd.DataFrame, alpha: float, seed: int,
                text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Create a mixed dataset of the same size as df_id, replacing alpha fraction with unknown.
    Unknown labels are set to -1 (if label_col exists).
    """
    rng = np.random.RandomState(seed)
    n = len(df_id)
    k_unk = int(round(alpha * n))
    k_id = n - k_unk
    df_id_s = df_id.sample(n=k_id, replace=False if k_id <= n else True, random_state=seed).copy()
    df_unk_s = df_unk.sample(n=k_unk, replace=True, random_state=seed + 7).copy()
    if label_col in df_unk_s.columns:
        df_unk_s[label_col] = -1
    else:
        df_unk_s[label_col] = -1
    out = pd.concat([df_id_s, df_unk_s], axis=0).sample(frac=1.0, random_state=seed + 9).reset_index(drop=True)
    return out

def df_to_texts_labels(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> Tuple[List[str], List[int]]:
    texts = df[text_col].map(norm_text).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


# ============================================================
# Method configs (baselines + full)
# ============================================================

@dataclass
class MethodConfig:
    name: str
    use_teacher: bool
    use_consistency: bool
    pseudo_strategy: str  # "topk_balanced" or "naive_conf"
    use_ood_filter: bool
    use_temp_scaling: bool
    fixed_prior: bool
    use_bbse_pi: bool  # only for bbse baseline


def get_methods_for_suite(suite: str) -> List[MethodConfig]:
    methods: List[MethodConfig] = [
        MethodConfig(
            name="source_only",
            use_teacher=False,
            use_consistency=False,
            pseudo_strategy="topk_balanced",  # unused
            use_ood_filter=False,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=True,
            use_bbse_pi=False,
        ),
        MethodConfig(
            name="naive_self_train",
            use_teacher=False,
            use_consistency=False,
            pseudo_strategy="naive_conf",
            use_ood_filter=False,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=True,
            use_bbse_pi=False,
        ),
        MethodConfig(
            name="topk_student",
            use_teacher=False,
            use_consistency=False,
            pseudo_strategy="topk_balanced",
            use_ood_filter=False,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=True,
            use_bbse_pi=False,
        ),
        MethodConfig(
            name="topk_teacher",
            use_teacher=True,
            use_consistency=False,
            pseudo_strategy="topk_balanced",
            use_ood_filter=False,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=True,
            use_bbse_pi=False,
        ),
        MethodConfig(
            name="full",
            use_teacher=True,
            use_consistency=True,
            pseudo_strategy="topk_balanced",
            use_ood_filter=False,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=True,
            use_bbse_pi=False,
        ),
    ]

    if ENABLE_BBSE_BASELINE:
        methods.append(MethodConfig(
            name="bbse_prior",
            use_teacher=True,
            use_consistency=False,
            pseudo_strategy="topk_balanced",
            use_ood_filter=False,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=False,
            use_bbse_pi=True,
        ))

    if suite in ("open_set", "combined"):
        # Add explicit OOD-filter variants for open-set robustness
        methods.append(MethodConfig(
            name="full+oodfilter",
            use_teacher=True,
            use_consistency=True,
            pseudo_strategy="topk_balanced",
            use_ood_filter=True,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=True,
            use_bbse_pi=False,
        ))
        methods.append(MethodConfig(
            name="topk_teacher+oodfilter",
            use_teacher=True,
            use_consistency=False,
            pseudo_strategy="topk_balanced",
            use_ood_filter=True,
            use_temp_scaling=USE_TEMPERATURE_SCALING,
            fixed_prior=True,
            use_bbse_pi=False,
        ))
    return methods


# ============================================================
# BBSE (binary) for logging/baseline
# ============================================================

def estimate_pi_bbse_binary(y_true_src: np.ndarray, y_pred_src: np.ndarray, y_pred_tgt: np.ndarray, eps: float = 1e-6) -> float:
    cm = confusion_matrix(y_true_src, y_pred_src, labels=[0, 1]).astype(np.float64)
    true_counts = cm.sum(axis=1)
    C = np.zeros((2, 2), dtype=np.float64)
    for true_y in [0, 1]:
        denom = max(true_counts[true_y], eps)
        C[0, true_y] = cm[true_y, 0] / denom
        C[1, true_y] = cm[true_y, 1] / denom
    q1 = float((y_pred_tgt == 1).mean()) if len(y_pred_tgt) else 0.5
    q = np.array([1.0 - q1, q1], dtype=np.float64)
    pi = np.linalg.pinv(C) @ q
    pi1 = float(pi[1])
    return float(np.clip(pi1, eps, 1.0 - eps))


# ============================================================
# Core run: one seed, one condition, one method
# ============================================================

def run_one(
    bundle: DataBundle,
    method: MethodConfig,
    suite_name: str,
    condition: Dict[str, Any],
    seed: int,
    out_root: str,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device()

    run_id = f"{suite_name}__{method.name}__seed{seed}__{condition.get('tag','')}".strip("_")
    run_dir = os.path.join(out_root, "per_run", run_id)
    ensure_dir(run_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # ------------------------------------------------------------
    # Prepare target splits according to condition
    # ------------------------------------------------------------
    tgt_train_df = bundle.tgt_train_df.copy()
    tgt_val_df = bundle.tgt_val_df.copy()
    tgt_test_df = bundle.tgt_test_df.copy()
    unk_df = bundle.unk_df

    # Controlled label shift
    pi_target = condition.get("pi_target", None)
    if pi_target is not None:
        # resample each split to keep sizes but change class ratio
        tgt_train_df = resample_to_prior(tgt_train_df, pi=float(pi_target), n=len(tgt_train_df), seed=seed + 101)
        tgt_val_df = resample_to_prior(tgt_val_df, pi=float(pi_target), n=len(tgt_val_df), seed=seed + 202)
        tgt_test_df = resample_to_prior(tgt_test_df, pi=float(pi_target), n=len(tgt_test_df), seed=seed + 303)

    # Open-set mix into target unlabeled train and into eval splits if requested
    alpha_train = float(condition.get("alpha_train", 0.0))
    alpha_eval = float(condition.get("alpha_eval", 0.0))
    if (alpha_train > 0.0 or alpha_eval > 0.0) and unk_df is None:
        raise RuntimeError("Open-set suite requested but politics.csv unknown pool not found.")

    if alpha_train > 0.0:
        tgt_train_mix = mix_unknown(tgt_train_df, unk_df, alpha=alpha_train, seed=seed + 11)
    else:
        tgt_train_mix = tgt_train_df.copy()

    if alpha_eval > 0.0:
        tgt_val_mix = mix_unknown(tgt_val_df, unk_df, alpha=alpha_eval, seed=seed + 22)
        tgt_test_mix = mix_unknown(tgt_test_df, unk_df, alpha=alpha_eval, seed=seed + 33)
    else:
        tgt_val_mix = tgt_val_df.copy()
        tgt_test_mix = tgt_test_df.copy()

    # Convert to lists
    src_train_texts, src_train_labels = bundle.src_train_texts, bundle.src_train_labels
    src_val_texts, src_val_labels = bundle.src_val_texts, bundle.src_val_labels
    src_test_texts, src_test_labels = bundle.src_test_texts, bundle.src_test_labels

    tgt_unl_texts = tgt_train_mix["text"].map(norm_text).tolist()  # unlabeled by design
    tgt_val_texts, tgt_val_labels = df_to_texts_labels(tgt_val_mix)
    tgt_test_texts, tgt_test_labels = df_to_texts_labels(tgt_test_mix)

    # ------------------------------------------------------------
    # Model init
    # ------------------------------------------------------------
    student = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    teacher = None
    if method.use_teacher:
        teacher = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        teacher.load_state_dict(student.state_dict())

    # ------------------------------------------------------------
    # Stage0 source supervised (with caching per seed)
    # ------------------------------------------------------------
    cache_dir = os.path.join(out_root, "cache_source", f"seed{seed}")
    ensure_dir(cache_dir)
    source_ckpt_path = os.path.join(cache_dir, "source_best.pt")
    if os.path.exists(source_ckpt_path):
        student.load_state_dict(safe_torch_load(source_ckpt_path, map_location=device))
    else:
        st = train_supervised(
            model=student,
            tokenizer=tokenizer,
            train_texts=src_train_texts,
            train_labels=src_train_labels,
            train_weights=[1.0] * len(src_train_texts),
            val_texts=src_val_texts,
            val_labels=src_val_labels,
            device=device,
            out_dir=cache_dir,
            stage_name="source",
            epochs=EPOCHS_SOURCE,
            lr=LR,
            batch_size=TRAIN_BATCH,
            grad_accum=GRAD_ACCUM,
        )
        # copy to stable name
        torch.save(student.state_dict(), source_ckpt_path)

    # init teacher after stage0
    if teacher is not None:
        teacher.load_state_dict(student.state_dict())

    # ------------------------------------------------------------
    # Temperature scaling on source val (optional)
    # ------------------------------------------------------------
    scaler = None
    T_src = None
    if method.use_temp_scaling:
        scaler = TemperatureScaler().to(device)
        src_val_loader_pred = make_loader(tokenizer, src_val_texts, src_val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)
        with torch.no_grad():
            logits = predict_logits(student.to(device), src_val_loader_pred, device).to(device)
            labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
            T_src = scaler.fit(logits, labels_t, max_iter=50)

    # ------------------------------------------------------------
    # Eval before adapt
    # ------------------------------------------------------------
    def eval_all(tag: str, tau: float, use_scaler: bool = True) -> Dict[str, Any]:
        s = scaler if (use_scaler and scaler is not None) else None
        src_test_loader = make_loader(tokenizer, src_test_texts, src_test_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)
        tgt_val_loader = make_loader(tokenizer, tgt_val_texts, tgt_val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)
        tgt_test_loader = make_loader(tokenizer, tgt_test_texts, tgt_test_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)

        p1_src = predict_p1(student, src_test_loader, device, scaler=s)
        p1_val = predict_p1(student, tgt_val_loader, device, scaler=s)
        p1_test = predict_p1(student, tgt_test_loader, device, scaler=s)

        y_src = np.array(src_test_labels, dtype=int)
        y_val = np.array(tgt_val_labels, dtype=int)
        y_test = np.array(tgt_test_labels, dtype=int)

        out = {"tag": tag, "tau": float(tau)}
        out.update({f"src_{k}": v for k, v in eval_id_metrics(y_src, p1_src, tau=tau).items()})
        out.update({f"tgt_val_{k}": v for k, v in eval_id_metrics(y_val, p1_val, tau=tau).items()})
        out.update({f"tgt_test_{k}": v for k, v in eval_id_metrics(y_test, p1_test, tau=tau).items()})
        out["cm_test"] = confusion(y_test, p1_test, tau=tau)

        # open-set metrics if unknown in val/test
        if np.any(y_val == -1) or np.any(y_test == -1):
            out["ood_val"] = ood_metrics(y_val, score_maxsoft(p1_val))
            out["ood_test"] = ood_metrics(y_test, score_maxsoft(p1_test))
        return out

    before = eval_all("before_adapt", tau=0.5)

    # tau* tuning on target val (ID-only)
    tau_star = 0.5
    val_acc_star = None
    if TUNE_THRESHOLD_ON_TARGET_VAL:
        tgt_val_loader = make_loader(tokenizer, tgt_val_texts, tgt_val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)
        p1_val = predict_p1(student, tgt_val_loader, device, scaler=scaler)
        tau_star, val_acc_star = best_tau_for_accuracy(np.array(tgt_val_labels, dtype=int), p1_val)
    before_star = eval_all("before_adapt_tau*", tau=tau_star)

    # ------------------------------------------------------------
    # If source_only, stop here
    # ------------------------------------------------------------
    if method.name == "source_only":
        result = {
            "suite": suite_name,
            "condition": condition,
            "seed": seed,
            "method": method.name,
            "selected_round": 0,
            "tau_star": float(tau_star),
            "T_src": float(T_src) if T_src is not None else None,
            "metrics_before": before,
            "metrics_before_tau": before_star,
            # For CSV flattening we always expose both the default-threshold (0.5)
            # metrics and the tuned-threshold (tau*) metrics.
            "metrics_final": before_star,     # tuned threshold metrics (tau*)
            "metrics_final_05": before,       # fixed threshold metrics (0.5)
        }
        safe_json_dump(result, os.path.join(run_dir, "result.json"))
        return flatten_result_for_csv(result)

    # ------------------------------------------------------------
    # Prepare BBSE confusion on source-val for baseline
    # ------------------------------------------------------------
    src_val_loader = make_loader(tokenizer, src_val_texts, src_val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)
    p1_src_val = predict_p1(student, src_val_loader, device, scaler=scaler)
    y_true_src_val = np.array(src_val_labels, dtype=int)
    y_pred_src_val = (p1_src_val >= 0.5).astype(int)

    # ------------------------------------------------------------
    # Adaptation rounds
    # ------------------------------------------------------------
    pseudo_pool: Dict[int, Tuple[int, float]] = {}
    best_sel = {"round": 0, "val_acc": float(val_acc_star) if val_acc_star is not None else -1.0,
                "tau_star": float(tau_star), "T_src": float(T_src) if T_src is not None else None,
                "ckpt": None}

    per_round_logs: List[Dict[str, Any]] = []

    # Unlabeled loader for consistency (iterated each step)
    unl_loader = make_loader(tokenizer, tgt_unl_texts, labels=None, weights=None, batch_size=TRAIN_BATCH,
                             max_len=MAX_LEN, shuffle=True)

    for r in range(1, ADAPT_ROUNDS + 1):
        pseudo_frac = PSEUDO_FRAC_SCHEDULE[min(r-1, len(PSEUDO_FRAC_SCHEDULE)-1)]
        lam_u = LAMBDA_U_SCHEDULE[min(r-1, len(LAMBDA_U_SCHEDULE)-1)]

        # 1) Predict on all target unlabeled for pseudo selection
        unl_pred_loader = make_loader(tokenizer, tgt_unl_texts, labels=None, weights=None,
                                      batch_size=PRED_BATCH, max_len=MAX_LEN, shuffle=False)
        if method.use_teacher and teacher is not None:
            p1_unl = predict_p1(teacher.to(device), unl_pred_loader, device, scaler=scaler)
        else:
            p1_unl = predict_p1(student.to(device), unl_pred_loader, device, scaler=scaler)

        diag = {
            "round": r,
            "tgt_pred_pos@0.5": float((p1_unl >= 0.5).mean()),
            "p1_min": float(np.min(p1_unl)) if len(p1_unl) else None,
            "p1_med": float(np.median(p1_unl)) if len(p1_unl) else None,
            "p1_max": float(np.max(p1_unl)) if len(p1_unl) else None,
            "p1_mean": float(np.mean(p1_unl)) if len(p1_unl) else None,
        }

        # 2) Prior choice
        pi_source = float(np.mean(bundle.src_train_labels))
        pi_used = pi_source
        pi_bbse = None
        if method.use_bbse_pi:
            y_pred_tgt = (p1_unl >= 0.5).astype(int)
            pi_bbse = estimate_pi_bbse_binary(y_true_src_val, y_pred_src_val, y_pred_tgt)
            pi_used = float(pi_bbse)
        else:
            # log only
            y_pred_tgt = (p1_unl >= 0.5).astype(int)
            pi_bbse = estimate_pi_bbse_binary(y_true_src_val, y_pred_src_val, y_pred_tgt)
            if not FIX_PRIOR_TO_SOURCE and not method.fixed_prior:
                pi_used = float(pi_bbse)
            else:
                pi_used = pi_source

        diag["pi_bbse"] = float(pi_bbse) if pi_bbse is not None else None
        diag["pi_used"] = float(pi_used)

        # 3) OOD filter for pseudo selection (optional)
        id_mask = None
        if method.use_ood_filter:
            score = score_maxsoft(p1_unl)
            alpha_guess = alpha_train if OOD_FILTER_ALPHA_GUESS is None else float(OOD_FILTER_ALPHA_GUESS)
            id_mask = ood_filter_mask_by_quantile(score, alpha_guess=alpha_guess)
            diag["ood_keep_frac"] = float(id_mask.mean())
        else:
            id_mask = np.ones(len(p1_unl), dtype=bool)

        # 4) Build/accumulate pseudo pool
        new_pool, pseudo_info = build_pseudo_pool(
            p1=p1_unl,
            pi_used=pi_used,
            pseudo_frac=pseudo_frac,
            seed=seed + 1000 * r,
            id_mask=id_mask,
            strategy=method.pseudo_strategy if method.pseudo_strategy else "topk_balanced",
        )
        # accumulate: keep higher weight if conflict
        for idx, (yy, ww) in new_pool.items():
            if idx in pseudo_pool:
                old_y, old_w = pseudo_pool[idx]
                if ww > old_w:
                    pseudo_pool[idx] = (yy, ww)
            else:
                pseudo_pool[idx] = (yy, ww)

        pseudo_stats = {
            "round": r,
            "pseudo_frac": pseudo_frac,
            "lam_u": lam_u,
            "pool_size": int(len(pseudo_pool)),
            "pool_pos": int(sum(1 for _, (yy, _) in pseudo_pool.items() if yy == 1)),
            "pool_neg": int(sum(1 for _, (yy, _) in pseudo_pool.items() if yy == 0)),
            **pseudo_info,
        }

        # 5) Build training data: source + pseudo
        pseudo_idx = list(pseudo_pool.keys())
        pseudo_texts = [tgt_unl_texts[i] for i in pseudo_idx]
        pseudo_labels = [pseudo_pool[i][0] for i in pseudo_idx]
        pseudo_weights = [pseudo_pool[i][1] for i in pseudo_idx]

        train_texts = src_train_texts + pseudo_texts
        train_labels = src_train_labels + pseudo_labels
        train_weights = [1.0] * len(src_train_texts) + pseudo_weights

        tr_loader = make_loader(tokenizer, train_texts, train_labels, train_weights, TRAIN_BATCH, MAX_LEN, shuffle=True)
        va_loader = make_loader(tokenizer, src_val_texts, src_val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)

        # 6) Adapt training (with optional consistency)
        student.to(device)
        if teacher is not None:
            teacher.to(device)
            teacher.eval()

        optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = max(1, ADAPT_EPOCHS_PER_ROUND * math.ceil(len(tr_loader) / max(1, GRAD_ACCUM)))
        warmup_steps = int(WARMUP_RATIO * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / max(1, warmup_steps), 1.0)
            if step < warmup_steps else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
        )
        ce = nn.CrossEntropyLoss(reduction="none")

        best_f1 = -1.0
        best_path = os.path.join(run_dir, f"adapt_round{r}_best.pt")

        unl_iter = iter(unl_loader)
        for ep in range(1, ADAPT_EPOCHS_PER_ROUND + 1):
            student.train()
            optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(tqdm(tr_loader, desc=f"Train adapt_round{r} e{ep}/{ADAPT_EPOCHS_PER_ROUND}")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                weights = batch["weights"].to(device)

                logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
                loss_sup = (ce(logits, labels) * weights).mean()

                loss = loss_sup

                # Consistency on a batch of unlabeled target texts
                if method.use_consistency and teacher is not None and lam_u > 0:
                    try:
                        u_batch = next(unl_iter)
                    except StopIteration:
                        unl_iter = iter(unl_loader)
                        u_batch = next(unl_iter)

                    u_input_ids = u_batch["input_ids"].to(device)
                    u_attention = u_batch["attention_mask"].to(device)

                    with torch.no_grad():
                        t_logits = teacher(input_ids=u_input_ids, attention_mask=u_attention).logits
                        t_probs = torch.softmax(t_logits, dim=-1)
                        conf = torch.max(t_probs, dim=-1).values
                        mask = (conf >= CONSISTENCY_CONF_THRESH).float()

                    s_logits = student(input_ids=u_input_ids, attention_mask=u_attention).logits
                    kl = kl_divergence_with_logits(t_logits.detach(), s_logits, temp=1.0)  # per-sample
                    if mask.sum() > 0:
                        loss_u = (kl * mask).sum() / (mask.sum() + 1e-9)
                        loss = loss + lam_u * loss_u

                loss = loss / max(1, GRAD_ACCUM)
                loss.backward()

                if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(tr_loader):
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Update EMA teacher after each optimizer step
                    if teacher is not None and method.use_teacher:
                        update_ema(teacher, student, decay=EMA_DECAY)

            # validate on source-val macro-F1@0.5 (model selection during training)
            val_p1 = predict_p1(student, va_loader, device, scaler=None)
            met = eval_id_metrics(np.array(src_val_labels, dtype=int), val_p1, tau=0.5)
            f1 = met["id_macro_f1"]
            if f1 > best_f1:
                best_f1 = float(f1)
                torch.save(student.state_dict(), best_path)

        # Load best round checkpoint
        student.load_state_dict(safe_torch_load(best_path, map_location=device))

        # Recalibrate temperature
        if method.use_temp_scaling and RECALIBRATE_EACH_ROUND:
            scaler = TemperatureScaler().to(device)
            with torch.no_grad():
                logits = predict_logits(student, src_val_loader, device).to(device)
                labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
                T_src = scaler.fit(logits, labels_t, max_iter=50)

        # Evaluate on target val (ID only) and tune tau*
        tau_star_r = 0.5
        val_acc_r = float("nan")
        if TUNE_THRESHOLD_ON_TARGET_VAL:
            tgt_val_loader = make_loader(tokenizer, tgt_val_texts, tgt_val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)
            p1_val = predict_p1(student, tgt_val_loader, device, scaler=scaler)
            tau_star_r, val_acc_r = best_tau_for_accuracy(np.array(tgt_val_labels, dtype=int), p1_val)

        # Save selection based on target val accuracy
        if SELECT_BEST_BY_TARGET_VAL_ACC and (val_acc_r is not None) and (val_acc_r > best_sel["val_acc"]):
            best_sel = {
                "round": r,
                "val_acc": float(val_acc_r),
                "tau_star": float(tau_star_r),
                "T_src": float(T_src) if T_src is not None else None,
                "ckpt": best_path,
            }

        # Record round log
        round_eval_05 = eval_all(f"round{r}@0.5", tau=0.5)
        round_eval_star = eval_all(f"round{r}@tau*", tau=tau_star_r)

        per_round_logs.append({
            "diag": diag,
            "pseudo": pseudo_stats,
            "best_sourceval_f1": float(best_f1),
            "T_src": float(T_src) if T_src is not None else None,
            "tau_star": float(tau_star_r),
            "val_acc_star": float(val_acc_r),
            "eval_0.5": round_eval_05,
            "eval_tau": round_eval_star,
        })

    # ------------------------------------------------------------
    # Final: load best checkpoint by target-val accuracy
    # ------------------------------------------------------------
    if best_sel["ckpt"] is not None:
        student.load_state_dict(safe_torch_load(best_sel["ckpt"], map_location=device))
        # restore best temperature
        if method.use_temp_scaling and best_sel["T_src"] is not None:
            scaler = TemperatureScaler().to(device)
            with torch.no_grad():
                scaler.temperature.fill_(float(best_sel["T_src"]))
    # recompute tau* on val for final (stability)
    if TUNE_THRESHOLD_ON_TARGET_VAL:
        tgt_val_loader = make_loader(tokenizer, tgt_val_texts, tgt_val_labels, None, PRED_BATCH, MAX_LEN, shuffle=False)
        p1_val = predict_p1(student, tgt_val_loader, device, scaler=scaler)
        tau_star, val_acc_star = best_tau_for_accuracy(np.array(tgt_val_labels, dtype=int), p1_val)
    final_05 = eval_all("final@0.5", tau=0.5)
    final_star = eval_all("final@tau*", tau=tau_star)

    result = {
        "suite": suite_name,
        "condition": condition,
        "seed": seed,
        "method": method.name,
        "selected_round": int(best_sel["round"]),
        "tau_star": float(tau_star),
        "T_src": float(best_sel["T_src"]) if best_sel["T_src"] is not None else float(T_src) if T_src is not None else None,
        "metrics_before": before,
        "metrics_before_tau": before_star,
        "metrics_final": final_star,
        "metrics_final_05": final_05,
        "per_round": per_round_logs,
    }
    safe_json_dump(result, os.path.join(run_dir, "result.json"))
    return flatten_result_for_csv(result)


def flatten_result_for_csv(res: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested result json into one row for CSV summaries."""
    out: Dict[str, Any] = {}
    out["suite"] = res["suite"]
    out["seed"] = res["seed"]
    out["method"] = res["method"]
    cond = res.get("condition", {})
    for k, v in cond.items():
        out[f"cond_{k}"] = v
    out["selected_round"] = res.get("selected_round", None)
    out["tau_star"] = res.get("tau_star", None)
    out["T_src"] = res.get("T_src", None)

    # final metrics (ID)
    # Some early-exit baselines may not populate all keys; be defensive.
    m = res.get("metrics_final", {})
    out["tgt_test_acc_tau"] = m.get("tgt_test_id_acc", None)
    out["tgt_test_bal_acc_tau"] = m.get("tgt_test_id_bal_acc", None)
    out["tgt_test_macro_f1_tau"] = m.get("tgt_test_id_macro_f1", None)
    out["tgt_val_acc_tau"] = m.get("tgt_val_id_acc", None)
    out["tgt_val_macro_f1_tau"] = m.get("tgt_val_id_macro_f1", None)
    # 0.5 metrics
    m05 = res.get("metrics_final_05", res.get("metrics_before", {}))
    out["tgt_test_acc_05"] = m05.get("tgt_test_id_acc", None)
    out["tgt_test_macro_f1_05"] = m05.get("tgt_test_id_macro_f1", None)

    # OOD metrics if present
    if "ood_test" in m:
        out["ood_test_auroc"] = m["ood_test"].get("ood_auroc", None)
        out["ood_test_fpr95"] = m["ood_test"].get("ood_fpr95", None)
    else:
        out["ood_test_auroc"] = None
        out["ood_test_fpr95"] = None

    # confusion at tau*
    cm = m.get("cm_test", {})
    out["cm_TN_tau"] = cm.get("TN", None)
    out["cm_FP_tau"] = cm.get("FP", None)
    out["cm_FN_tau"] = cm.get("FN", None)
    out["cm_TP_tau"] = cm.get("TP", None)
    return out


# ============================================================
# Aggregation
# ============================================================

def aggregate_results(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        for mc in metric_cols:
            vals = g[mc].astype(float)
            row[f"{mc}_mean"] = float(vals.mean())
            row[f"{mc}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{mc}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# ============================================================
# Main experiment runner
# ============================================================

def find_politics_path(base_dir: str) -> Optional[str]:
    cand = [
        os.path.join(base_dir, "politics.csv"),
        os.path.join(base_dir, "targetdata_clean", "politics.csv"),
        os.path.join(base_dir, "targetdata", "politics.csv"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None


def load_bundle(base_dir: str) -> DataBundle:
    # Source
    source_dir = os.path.join(base_dir, "sourcedata")
    src_train = load_table_auto(os.path.join(source_dir, "source_train.csv"))
    src_val = load_table_auto(os.path.join(source_dir, "source_validation.csv"))
    src_test = load_table_auto(os.path.join(source_dir, "source_test.csv"))

    # Target (prefer clean)
    target_dir = os.path.join(base_dir, "targetdata_clean")
    if not os.path.exists(target_dir):
        target_dir = os.path.join(base_dir, "targetdata")
    # Target train may or may not contain labels. We try labeled first (needed for label-shift suite),
    # then fallback to text-only.
    tgt_train_has_labels = True
    try:
        tgt_train = load_table_auto(os.path.join(target_dir, "train.csv"), label_col="label")
    except Exception:
        tgt_train_has_labels = False
        tgt_train = load_table_auto(os.path.join(target_dir, "train.csv"), label_col=None)
        tgt_train["label"] = 0  # placeholder
    tgt_val = load_table_auto(os.path.join(target_dir, "val.csv"), label_col="label")
    tgt_test = load_table_auto(os.path.join(target_dir, "test.csv"), label_col="label")

    # Normalize text + drop empty
    for _df in [src_train, src_val, src_test, tgt_train, tgt_val, tgt_test]:
        _df["text"] = _df["text"].map(norm_text)
        _df.dropna(subset=["text"], inplace=True)
        _df = _df[_df["text"].astype(str).str.len() > 0]


    # Unknown pool
    pol_path = find_politics_path(base_dir)
    unk_df = None
    if pol_path is not None:
        # If label exists, load it, otherwise text-only and add label=-1
        try:
            unk_df = load_table_auto(pol_path, label_col="label")
        except Exception:
            unk_df = load_table_auto(pol_path, label_col=None)
            unk_df["label"] = -1
        if "text" not in unk_df.columns:
            raise ValueError("politics.csv must contain a 'text' column.")
        unk_df["text"] = unk_df["text"].map(norm_text)
        if "label" not in unk_df.columns:
            unk_df["label"] = -1
        # Force unknown label
        unk_df["label"] = -1

    bundle = DataBundle(
        src_train_texts=src_train["text"].tolist(),
        src_train_labels=src_train["label"].astype(int).tolist(),
        src_val_texts=src_val["text"].tolist(),
        src_val_labels=src_val["label"].astype(int).tolist(),
        src_test_texts=src_test["text"].tolist(),
        src_test_labels=src_test["label"].astype(int).tolist(),
        tgt_train_df=tgt_train,
        tgt_train_has_labels=bool(tgt_train_has_labels),
        tgt_val_df=tgt_val,
        tgt_test_df=tgt_test,
        unk_df=unk_df,
    )
    return bundle


def run_suite(bundle: DataBundle, suite: str, out_root: str) -> pd.DataFrame:
    methods = get_methods_for_suite(suite)
    rows: List[Dict[str, Any]] = []

    # Build condition grid
    conditions: List[Dict[str, Any]] = []
    if suite == "base":
        conditions = [{"tag": "base", "alpha_train": 0.0, "alpha_eval": 0.0, "pi_target": None}]
    elif suite == "open_set":
        for a in OPENSET_ALPHA_GRID:
            conditions.append({"tag": f"alpha{a:.2f}", "alpha_train": float(a), "alpha_eval": float(OPENSET_EVAL_ALPHA), "pi_target": None})
    elif suite == "label_shift":
        for pi in LABELSHIFT_PI_GRID:
            conditions.append({"tag": f"pi{pi:.2f}", "alpha_train": 0.0, "alpha_eval": 0.0, "pi_target": float(pi)})
    elif suite == "combined":
        for a in COMBINED_ALPHA_GRID:
            for pi in COMBINED_PI_GRID:
                conditions.append({"tag": f"alpha{a:.2f}_pi{pi:.2f}", "alpha_train": float(a), "alpha_eval": float(OPENSET_EVAL_ALPHA), "pi_target": float(pi)})
    else:
        raise ValueError(f"Unknown suite: {suite}")

    for cond in conditions:
        for seed in SEEDS:
            for method in methods:
                print("\n" + "=" * 80)
                print(f"[RUN] suite={suite} method={method.name} seed={seed} condition={cond}")
                print("=" * 80)
                row = run_one(bundle, method, suite, cond, seed, out_root)
                rows.append(row)
                # Save incremental raw results
                df_tmp = pd.DataFrame(rows)
                df_tmp.to_csv(os.path.join(out_root, "all_results_partial.csv"), index=False, encoding="utf-8")

    df = pd.DataFrame(rows)
    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base_dir, "runs", "paper_experiments")
    ensure_dir(out_root)

    device = get_device()
    print(f"[Device] {device}")
    bundle = load_bundle(base_dir)
    print("[Data] Loaded.")
    print(f"[Source] train={len(bundle.src_train_texts)} val={len(bundle.src_val_texts)} test={len(bundle.src_test_texts)}")
    print(f"[Target] train={len(bundle.tgt_train_df)} val={len(bundle.tgt_val_df)} test={len(bundle.tgt_test_df)}")
    if bundle.unk_df is not None:
        print(f"[Unknown] politics={len(bundle.unk_df)}")
    else:
        print("[Unknown] politics.csv not found -> open-set suite will fail if enabled.")

    all_rows = []

    for suite, enabled in RUN_SUITES.items():
        if not enabled:
            continue
        if suite in ("open_set", "combined") and bundle.unk_df is None:
            print(f"[Skip] suite={suite} because politics.csv not found.")
            continue

        if suite in ("label_shift", "combined") and (not bundle.tgt_train_has_labels):
            print(f"[Skip] suite={suite} because target train.csv has no labels (needed only to CONSTRUCT controlled priors).")
            continue

        df = run_suite(bundle, suite, out_root)
        df.to_csv(os.path.join(out_root, f"all_results_{suite}.csv"), index=False, encoding="utf-8")
        all_rows.append(df)

        # Aggregate
        group_cols = ["suite", "method"]
        # add condition tags in group if present
        if "cond_tag" in df.columns:
            group_cols = ["suite", "cond_tag", "method"]

        metrics = [
            "tgt_test_acc_tau",
            "tgt_test_macro_f1_tau",
            "tgt_test_acc_05",
            "ood_test_auroc",
            "ood_test_fpr95",
        ]
        # keep only those existing
        metrics = [m for m in metrics if m in df.columns]

        summary = aggregate_results(df, group_cols=group_cols, metric_cols=metrics)
        summary_path = os.path.join(out_root, f"summary_{suite}.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8")
        print(f"[Saved] {summary_path}")

    if all_rows:
        df_all = pd.concat(all_rows, axis=0).reset_index(drop=True)
        df_all.to_csv(os.path.join(out_root, "all_results.csv"), index=False, encoding="utf-8")
        print(f"[Saved] {os.path.join(out_root, 'all_results.csv')}")


if __name__ == "__main__":
    main()
