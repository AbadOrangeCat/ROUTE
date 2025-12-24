#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-run (no CLI needed) Local 6-file training + Shift/Noise/Threshold-aware UDA (binary text classification)
================================================================================

This script is designed for the exact workflow you described:

- You "just click Run" (no command line args), and it will:
  1) Read 6 local files (relative to THIS script file by default):
        sourcedata/source_train.csv
        sourcedata/source_validation.csv
        sourcedata/source_test.csv
        targetdata/train.csv          (target unlabeled; labels ignored even if present)
        targetdata/val.csv            (target validation for evaluation AND model selection)
        targetdata/test.csv           (target test for final report)

  2) Train a source supervised model (BERT binary classifier)
  3) Run a more stable UDA loop:
        - Robust prior estimation with BBSE/EM + fallback + smoothing
        - Prior correction + quantile threshold for pseudo labeling
        - **Class-balanced TOP-K pseudo label selection** (prevents all-pos/all-neg collapse)
        - Optional EMA teacher (ENABLED by default) to stabilize pseudo labels
        - Temperature scaling (ENABLED by default) to stabilize probabilities
        - Model selection by *target validation accuracy* (ENABLED by default) to improve final accuracy

Why this improves accuracy:
- Your previous run showed strong conditional shift; naive pseudo-labeling oscillated.
- TOP-K balanced selection prevents collapse and is much more stable.
- Temperature scaling improves calibration, which improves thresholding and pseudo label quality.
- EMA teacher reduces "confidence spikes" that damage pseudo labels.
- Selecting the best checkpoint using target-val accuracy (since you have val labels) boosts final accuracy.

If you want STRICT unsupervised evaluation (no target labels used for selection),
set USE_TARGET_VAL_FOR_SELECTION = False in the config section below.

Dependencies:
  pip install -U torch transformers scikit-learn pandas numpy tqdm

Tested with:
- Python 3.8+
- Mac MPS (Apple Silicon), CPU, CUDA

Author: ChatGPT
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# =============================================================================
# Default "click-run" CONFIG (edit here if you want)
# =============================================================================

DEFAULT_MODEL_NAME = "bert-base-uncased"

# Training
SOURCE_EPOCHS = 3
ADAPT_ROUNDS = 4          # slightly more than 3 (often helps)
ADAPT_EPOCHS = 1

# For long source texts, increase max_len; reduce batch size to fit MPS
MAX_LEN = 384
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2      # effective batch ~16

LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0

# Calibration
USE_TEMPERATURE_SCALING = True
RECALIBRATE_EACH_ROUND = True
TS_MAX_ITER = 50

# EMA teacher (stabilizes pseudo labels)
USE_EMA_TEACHER = True
EMA_DECAY = 0.999         # higher = smoother teacher

# OOD split (your data seems closed-set; keep none for best accuracy)
OOD_METHOD = "none"       # "none" | "quantile" | "gmm"
OOD_POSTERIOR_THRESHOLD = 0.5
ALPHA_MIN = 0.1

# Prior estimation
SHIFT_METHOD = "auto"     # "auto" | "bbse" | "em"
BBSE_EPS = 1e-6
BBSE_MAX_COND = 1e6

# Prior smoothing across rounds (prevents pi_t from jumping too much)
SMOOTH_PI = True
PI_SMOOTH_BETA = 0.6      # pi <- beta*pi_prev + (1-beta)*pi_new

# If estimated pi is too extreme, fallback
PI_EXTREME_LOW = 0.05
PI_EXTREME_HIGH = 0.95

# Threshold method for pseudo labels
THRESHOLD_METHOD = "quantile"   # "quantile" | "logit_bias"

# Pseudo labels
PSEUDO_FRAC_START = 0.20
PSEUDO_FRAC_END = 0.35
ACCUMULATE_PSEUDO = True
PSEUDO_WEIGHT = 1.0

# TOP-K balanced pseudo labels always on (prevents collapse)
PSEUDO_STRATEGY = "topk_balanced"  # "topk_balanced" only in this script

# Optional unsupervised consistency (teacher->student) on ID samples
USE_CONSISTENCY = True
CONSISTENCY_LAMBDA = 0.2
CONSISTENCY_CONF_THR = 0.60  # only apply when teacher max prob >= thr

# Model selection and threshold tuning for best final accuracy
# IMPORTANT: this uses target validation labels to choose best checkpoint + threshold.
USE_TARGET_VAL_FOR_SELECTION = True
TUNE_THRESHOLD_ON_TARGET_VAL = True
THRESH_SWEEP_QUANTILES = 200

# Output
DEFAULT_RUNS_DIR_NAME = "runs/local6_autorun_plus"

# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)


def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# =============================================================================
# Robust CSV loaders (handle trailing commas etc.)
# =============================================================================

def _open_text(path: str, encoding: str = "utf-8"):
    return open(path, "r", encoding=encoding, errors="replace", newline="")


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
        header = [h.strip() for h in header]
        text_idx = header.index(text_col) if text_col in header else 0

        for row in reader:
            if len(row) <= text_idx:
                continue
            rows.append(row[text_idx])
    return pd.DataFrame({text_col: rows})


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
        header = [h.strip() for h in header]
        text_idx = header.index(text_col) if text_col in header else 0
        label_idx = header.index(label_col) if label_col in header else 1

        for row in reader:
            if len(row) <= max(text_idx, label_idx):
                if drop_bad:
                    continue
                txt = row[text_idx] if len(row) > text_idx else ""
                rows.append((txt, 0))
                continue

            txt = row[text_idx]
            lab = row[label_idx].strip()

            # allow "1", "0", "-1", "1.0"
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


def load_table_auto(path: str, text_col: str, label_col: Optional[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

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
        raise ValueError(f"Unsupported extension: {ext}")

    if text_col not in df.columns:
        raise ValueError(f"Missing {text_col} in {path}. Columns: {list(df.columns)}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Missing {label_col} in {path}. Columns: {list(df.columns)}")
    return df


# =============================================================================
# Dataset + Collate
# =============================================================================

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
                batch_size: int, max_len: int, num_workers: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = TextDataset(texts, labels=labels, weights=weights)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer, max_len),
    )


# =============================================================================
# Calibration: Temperature scaling
# =============================================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.clamp(self.temperature, min=1e-3)
        return logits / T

    @torch.no_grad()
    def get_temperature(self) -> float:
        return float(torch.clamp(self.temperature, min=1e-3).item())

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        device = logits.device
        self.to(device)
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self.get_temperature()


# =============================================================================
# Prediction helpers
# =============================================================================

@torch.inference_mode()
def predict_logits(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    out = []
    for batch in tqdm(loader, desc="Predict", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        out.append(logits.detach().cpu())
    return torch.cat(out, dim=0)


def p1_from_logits(logits: torch.Tensor) -> np.ndarray:
    """Return P(y=1|x) from logits.
    Note: logits may require grad if produced via temperature scaling (depends on learnable T).
    We detach defensively to allow .numpy() safely.
    """
    logits = logits.detach()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, 1]


def hard_pred(p1: np.ndarray, tau: float = 0.5) -> np.ndarray:
    return (p1 >= tau).astype(np.int64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# =============================================================================
# Metrics
# =============================================================================

def eval_binary(y_true: np.ndarray, p1: np.ndarray, tau: float = 0.5) -> Dict[str, Any]:
    # ignore y=-1 if present
    mask = np.isin(y_true, [0, 1])
    y = y_true[mask].astype(int)
    p = p1[mask]

    pred = (p >= tau).astype(int)
    acc = accuracy_score(y, pred) if len(y) else float("nan")
    macro_f1 = f1_score(y, pred, average="macro") if len(y) else float("nan")
    pr, rc, f1_each, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], average=None, zero_division=0)
    auc = None
    try:
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, p))
    except Exception:
        auc = None

    return {
        "n": int(len(y_true)),
        "n_known": int(len(y)),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "precision_0": float(pr[0]) if len(pr) > 0 else None,
        "recall_0": float(rc[0]) if len(rc) > 0 else None,
        "f1_0": float(f1_each[0]) if len(f1_each) > 0 else None,
        "precision_1": float(pr[1]) if len(pr) > 1 else None,
        "recall_1": float(rc[1]) if len(rc) > 1 else None,
        "f1_1": float(f1_each[1]) if len(f1_each) > 1 else None,
        "auc": auc,
        "tau": float(tau),
    }


def find_best_threshold(
    y_true: np.ndarray,
    p1: np.ndarray,
    metric: str = "acc",
    n_quantiles: int = 200,
) -> Tuple[float, Dict[str, Any]]:
    """
    Find tau that maximizes accuracy or macro_f1 on labeled validation set.
    Uses threshold candidates from p1 quantiles (plus 0.5).
    """
    mask = np.isin(y_true, [0, 1])
    y = y_true[mask].astype(int)
    p = p1[mask]

    if len(y) == 0:
        return 0.5, {"best_tau": 0.5, "best_metric": None, "metric": metric, "note": "no labeled samples"}

    qs = np.linspace(0.0, 1.0, n_quantiles)
    cand = np.unique(np.quantile(p, qs))
    cand = np.unique(np.concatenate([cand, np.array([0.5])]))
    cand = np.clip(cand, 0.0, 1.0)

    best_tau = 0.5
    best_val = -1e18
    best_f1 = -1e18
    for tau in cand:
        pred = (p >= tau).astype(int)
        acc = accuracy_score(y, pred)
        mf1 = f1_score(y, pred, average="macro")
        val = acc if metric == "acc" else mf1

        # tie-break: prefer higher macro_f1, then tau closer to 0.5
        if (val > best_val) or (abs(val - best_val) < 1e-12 and mf1 > best_f1) or (
            abs(val - best_val) < 1e-12 and abs(mf1 - best_f1) < 1e-12 and abs(tau - 0.5) < abs(best_tau - 0.5)
        ):
            best_val = val
            best_f1 = mf1
            best_tau = float(tau)

    info = {"best_tau": best_tau, "best_metric": float(best_val), "best_macro_f1": float(best_f1), "metric": metric}
    return best_tau, info


# =============================================================================
# OOD split (optional)
# =============================================================================

def ood_split(scores: np.ndarray, method: str, seed: int, posterior_threshold: float, alpha_min: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(scores)
    if method == "none":
        return np.ones(n, dtype=bool), {"method": "none", "alpha_hat": 0.0}

    if method == "quantile":
        thr = float(np.quantile(scores, alpha_min))
        id_mask = scores >= thr
        return id_mask, {"method": "quantile", "alpha_min": float(alpha_min), "threshold": thr, "alpha_hat": float(1.0 - id_mask.mean())}

    if method == "gmm":
        x = scores.reshape(-1, 1).astype(np.float64)
        gmm = GaussianMixture(n_components=2, random_state=seed)
        gmm.fit(x)
        means = gmm.means_.reshape(-1)
        id_comp = int(np.argmax(means))
        proba = gmm.predict_proba(x)
        id_prob = proba[:, id_comp]
        id_mask = id_prob >= posterior_threshold
        return id_mask, {"method": "gmm", "means": means.tolist(), "id_component": id_comp, "posterior_threshold": float(posterior_threshold), "alpha_hat": float(1.0 - id_mask.mean())}

    raise ValueError(f"Unknown ood_method={method}")


# =============================================================================
# Label shift estimation (BBSE / EM) + correction
# =============================================================================

def estimate_prior_bbse_binary(y_true_src: np.ndarray, y_pred_src: np.ndarray, y_pred_tgt: np.ndarray, eps: float) -> Tuple[float, Dict[str, Any]]:
    cm = confusion_matrix(y_true_src, y_pred_src, labels=[0, 1]).astype(np.float64)
    true_counts = cm.sum(axis=1)
    C = np.zeros((2, 2), dtype=np.float64)
    for true_y in [0, 1]:
        denom = max(true_counts[true_y], eps)
        C[0, true_y] = cm[true_y, 0] / denom
        C[1, true_y] = cm[true_y, 1] / denom

    q1 = float((y_pred_tgt == 1).mean()) if len(y_pred_tgt) else 0.5
    q = np.array([1.0 - q1, q1], dtype=np.float64)

    cond = float(np.linalg.cond(C))
    pi = np.linalg.pinv(C) @ q
    pi1 = float(pi[1])
    pi1_clip = float(np.clip(pi1, eps, 1.0 - eps))

    info = {"method": "bbse", "pi1_raw": float(pi1), "pi1": float(pi1_clip), "q1": float(q1), "cond": float(cond), "C": C.tolist()}
    return pi1_clip, info


def estimate_prior_saerens_em_binary(p1_src_on_tgt: np.ndarray, pi_s1: float, max_iter: int, tol: float, eps: float) -> Tuple[float, Dict[str, Any]]:
    p1 = np.clip(p1_src_on_tgt, eps, 1.0 - eps)
    pi_s1 = float(np.clip(pi_s1, eps, 1.0 - eps))
    pi_s0 = 1.0 - pi_s1
    pi_t1 = pi_s1

    for it in range(max_iter):
        pi_t0 = 1.0 - pi_t1
        a1 = (pi_t1 / pi_s1) * p1
        a0 = (pi_t0 / pi_s0) * (1.0 - p1)
        denom = np.clip(a0 + a1, eps, None)
        r1 = a1 / denom
        new_pi = float(np.mean(r1))
        if abs(new_pi - pi_t1) < tol:
            pi_t1 = new_pi
            break
        pi_t1 = new_pi

    pi_t1 = float(np.clip(pi_t1, eps, 1.0 - eps))
    info = {"method": "em", "pi1": float(pi_t1), "iters": int(it + 1)}
    return pi_t1, info


def prior_correct_p1(p1_src: np.ndarray, pi_s1: float, pi_t1: float, eps: float = 1e-9) -> np.ndarray:
    p1 = np.clip(p1_src, eps, 1.0 - eps)
    pi_s1 = float(np.clip(pi_s1, eps, 1.0 - eps))
    pi_t1 = float(np.clip(pi_t1, eps, 1.0 - eps))
    w1 = pi_t1 / pi_s1
    w0 = (1.0 - pi_t1) / (1.0 - pi_s1)
    a1 = w1 * p1
    a0 = w0 * (1.0 - p1)
    denom = np.clip(a0 + a1, eps, None)
    return a1 / denom


def choose_tau_by_quantile(p1: np.ndarray, pi_t1: float) -> float:
    pi_t1 = float(np.clip(pi_t1, 1e-6, 1.0 - 1e-6))
    return float(np.quantile(p1, 1.0 - pi_t1))


def choose_logit_bias(pi_s1: float, pi_t1: float, eps: float = 1e-9) -> float:
    pi_s1 = float(np.clip(pi_s1, eps, 1.0 - eps))
    pi_t1 = float(np.clip(pi_t1, eps, 1.0 - eps))
    return float(math.log(pi_t1 / (1.0 - pi_t1)) - math.log(pi_s1 / (1.0 - pi_s1)))


def robust_choose_pi(
    method: str,
    y_true_src_val: np.ndarray,
    y_pred_src_val: np.ndarray,
    p1_tgt_id: np.ndarray,
    pred_tgt_id: np.ndarray,
    pi_s1: float,
    pi_prev: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    """
    Choose pi_t robustly using BBSE/EM + fallback, then optional smoothing.
    """
    eps = BBSE_EPS

    # compute both candidates
    bbse_pi, bbse_info = estimate_prior_bbse_binary(y_true_src_val, y_pred_src_val, pred_tgt_id, eps=eps)
    em_pi, em_info = estimate_prior_saerens_em_binary(p1_tgt_id, pi_s1=pi_s1, max_iter=300, tol=1e-6, eps=eps)

    # sanity checks
    bbse_bad = False
    if bbse_info["cond"] > BBSE_MAX_COND:
        bbse_bad = True
    if not (PI_EXTREME_LOW <= bbse_pi <= PI_EXTREME_HIGH):
        bbse_bad = True
    if bbse_info["q1"] < 0.02 or bbse_info["q1"] > 0.98:
        # when hard predictions are extreme, BBSE tends to be unstable
        bbse_bad = True

    em_bad = not (PI_EXTREME_LOW <= em_pi <= PI_EXTREME_HIGH)

    chosen = None
    pi = None
    if method == "bbse":
        chosen = "bbse"
        pi = bbse_pi
    elif method == "em":
        chosen = "em"
        pi = em_pi
    else:
        # auto
        if not bbse_bad:
            chosen = "bbse"
            pi = bbse_pi
        elif not em_bad:
            chosen = "em"
            pi = em_pi
        else:
            chosen = "fallback_source"
            pi = pi_s1

    pi_raw = float(pi)

    # optional smoothing
    if SMOOTH_PI and pi_prev is not None:
        pi = float(PI_SMOOTH_BETA * pi_prev + (1.0 - PI_SMOOTH_BETA) * pi)

    pi = float(np.clip(pi, eps, 1.0 - eps))
    info = {
        "chosen": chosen,
        "pi_raw": pi_raw,
        "pi_smoothed": float(pi),
        "bbse": bbse_info,
        "em": em_info,
        "bbse_bad": bool(bbse_bad),
        "em_bad": bool(em_bad),
        "pi_prev": None if pi_prev is None else float(pi_prev),
    }
    return pi, info


# =============================================================================
# Pseudo label selection: TOP-K balanced
# =============================================================================

def select_pseudo_topk_balanced(
    p1_adj_id: np.ndarray,
    pi_t1: float,
    total_budget: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Select k_pos highest probs as positive, k_neg lowest probs as negative.
    Always returns both classes unless total_budget is too small.
    """
    rng = np.random.RandomState(seed)
    n = len(p1_adj_id)
    if n == 0 or total_budget <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), {"selected": 0}

    total_budget = min(total_budget, n)
    pi_t1 = float(np.clip(pi_t1, 1e-6, 1.0 - 1e-6))
    k_pos = int(round(total_budget * pi_t1))
    k_pos = max(1, min(k_pos, total_budget - 1)) if total_budget >= 2 else total_budget
    k_neg = total_budget - k_pos

    order = np.argsort(p1_adj_id)  # ascending
    neg_idx = order[:k_neg]
    pos_idx = order[-k_pos:]

    selected = np.concatenate([pos_idx, neg_idx], axis=0).astype(int)
    rng.shuffle(selected)

    labels = (p1_adj_id[selected] >= 0.5).astype(int)
    # Force labels to match selection intent: pos_idx->1, neg_idx->0
    # (after shuffle, we need mapping)
    label_map = {int(i): 1 for i in pos_idx.tolist()}
    label_map.update({int(i): 0 for i in neg_idx.tolist()})
    labels = np.array([label_map[int(i)] for i in selected], dtype=int)

    # confidence weights: pos use p, neg use (1-p)
    weights = np.where(labels == 1, p1_adj_id[selected], 1.0 - p1_adj_id[selected])
    weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

    info = {
        "strategy": "topk_balanced",
        "n_id": int(n),
        "total_budget": int(total_budget),
        "k_pos": int(k_pos),
        "k_neg": int(k_neg),
        "selected": int(len(selected)),
        "selected_pos": int(labels.sum()),
        "selected_neg": int(len(labels) - labels.sum()),
        "pi_t1": float(pi_t1),
    }
    return selected, labels, weights, info


# =============================================================================
# EMA teacher
# =============================================================================

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainCfg:
    epochs: int
    batch_size: int
    grad_accum_steps: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_len: int
    grad_clip: float
    num_workers: int = 0


def train_stage(
    student: nn.Module,
    teacher: Optional[nn.Module],
    tokenizer,
    device: torch.device,
    train_texts: List[str],
    train_labels: List[int],
    train_weights: List[float],
    val_texts: List[str],
    val_labels: List[int],
    cfg: TrainCfg,
    out_dir: str,
    stage_name: str,
    use_consistency: bool,
    consistency_lambda: float,
    consistency_conf_thr: float,
    ema_decay: float,
) -> Dict[str, Any]:
    """
    Train on weighted labeled data. Optionally add teacher->student consistency loss.
    Save best checkpoint by source-val macro F1 (to avoid overfitting in training).
    (Final selection can use target val if enabled.)
    """
    ensure_dir(out_dir)
    collate_fn = make_collate_fn(tokenizer, cfg.max_len)

    train_ds = TextDataset(train_texts, train_labels, train_weights)
    val_ds = TextDataset(val_texts, val_labels, weights=None)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    student.to(device)
    student.train()

    if teacher is not None:
        teacher.to(device)
        teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps)))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ce = nn.CrossEntropyLoss(reduction="none")
    kl = nn.KLDivLoss(reduction="batchmean")

    best_f1 = -1.0
    best_path = os.path.join(out_dir, f"{stage_name}_best.pt")
    history = {"stage": stage_name, "epochs": []}

    for ep in range(1, cfg.epochs + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        n_steps = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train {stage_name} e{ep}/{cfg.epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            out = student(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            loss_vec = ce(logits, labels)
            sup_loss = (loss_vec * weights).mean()

            loss = sup_loss

            # Optional consistency: teacher soft targets
            if use_consistency and teacher is not None and consistency_lambda > 0:
                with torch.no_grad():
                    t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
                    t_prob = torch.softmax(t_logits, dim=-1)
                    t_conf, _ = torch.max(t_prob, dim=-1)  # (B,)
                    mask = (t_conf >= consistency_conf_thr).float().unsqueeze(-1)  # (B,1)

                if mask.sum().item() > 0:
                    s_logprob = torch.log_softmax(logits, dim=-1)
                    # KL(teacher || student)
                    cons_loss = kl(s_logprob, t_prob)
                    # apply mask by reweighting: approximate by scaling loss
                    cons_loss = cons_loss * (mask.mean().clamp(min=0.0))
                    loss = loss + consistency_lambda * cons_loss

            loss = loss / max(1, cfg.grad_accum_steps)
            loss.backward()
            losses.append(float(loss.item()))
            n_steps += 1

            if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA update after each optimizer step
                if teacher is not None:
                    ema_update(teacher, student, decay=ema_decay)

        # evaluate on source val
        student.eval()
        val_logits = predict_logits(student, val_loader, device)
        val_p1 = p1_from_logits(val_logits)
        val_metrics = eval_binary(np.array(val_labels, dtype=int), val_p1, tau=0.5)

        ep_info = {
            "epoch": int(ep),
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "val_macro_f1@0.5": float(val_metrics["macro_f1"]),
            "val_acc@0.5": float(val_metrics["acc"]),
        }
        history["epochs"].append(ep_info)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            torch.save(student.state_dict(), best_path)

    # load best
    student.load_state_dict(torch.load(best_path, map_location=device))
    save_json(history, os.path.join(out_dir, f"{stage_name}_history.json"))
    return {"best_macro_f1": best_f1, "best_ckpt": best_path}


# =============================================================================
# Main pipeline
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_p1_with_scaler(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: Optional[TemperatureScaler],
) -> np.ndarray:
    """Predict probabilities with optional temperature scaling.
    Important: TemperatureScaler has learnable parameter T; applying it outside no_grad
    makes outputs require grad. We wrap it in no_grad for safe numpy conversion.
    """
    logits = predict_logits(model, loader, device)
    if scaler is not None:
        scaler.eval()
        with torch.no_grad():
            logits = scaler(logits.to(device)).detach().cpu()
    return p1_from_logits(logits)


def describe_probs(name: str, p: np.ndarray) -> str:
    if len(p) == 0:
        return f"{name}=empty"
    return (f"{name}(min/med/max)={p.min():.6g}/{np.median(p):.6g}/{p.max():.6g} mean={p.mean():.6g}")


def run():
    # Seed
    seed = 42
    set_seed(seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir  # default for click-run

    # Paths (your exact folder structure)
    src_train_path = os.path.join(data_dir, "sourcedata", "source_train.csv")
    src_val_path = os.path.join(data_dir, "sourcedata", "source_validation.csv")
    src_test_path = os.path.join(data_dir, "sourcedata", "source_test.csv")
    tgt_unl_path = os.path.join(data_dir, "targetdata", "train.csv")
    tgt_val_path = os.path.join(data_dir, "targetdata", "val.csv")
    tgt_test_path = os.path.join(data_dir, "targetdata", "test.csv")

    out_dir = os.path.join(data_dir, DEFAULT_RUNS_DIR_NAME)
    ensure_dir(out_dir)

    device = get_device()
    print(f"[Device] {device}")

    # Load model/tokenizer
    model_name = DEFAULT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    student = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load data
    print("[Data] Loading files...")
    text_col = "text"
    label_col = "label"

    src_train_df = load_table_auto(src_train_path, text_col=text_col, label_col=label_col)
    src_val_df = load_table_auto(src_val_path, text_col=text_col, label_col=label_col)
    src_test_df = load_table_auto(src_test_path, text_col=text_col, label_col=label_col)

    tgt_unl_df = load_table_auto(tgt_unl_path, text_col=text_col, label_col=None)  # ignore labels
    tgt_val_df = load_table_auto(tgt_val_path, text_col=text_col, label_col=label_col)
    tgt_test_df = load_table_auto(tgt_test_path, text_col=text_col, label_col=label_col)

    src_train_texts = src_train_df[text_col].map(to_str).tolist()
    src_train_labels = src_train_df[label_col].astype(int).tolist()
    src_val_texts = src_val_df[text_col].map(to_str).tolist()
    src_val_labels = src_val_df[label_col].astype(int).tolist()
    src_test_texts = src_test_df[text_col].map(to_str).tolist()
    src_test_labels = src_test_df[label_col].astype(int).tolist()

    tgt_unl_texts = tgt_unl_df[text_col].map(to_str).tolist()
    tgt_val_texts = tgt_val_df[text_col].map(to_str).tolist()
    tgt_val_labels = tgt_val_df[label_col].astype(int).tolist()
    tgt_test_texts = tgt_test_df[text_col].map(to_str).tolist()
    tgt_test_labels = tgt_test_df[label_col].astype(int).tolist()

    print(f"[Data] Source: train={len(src_train_texts)} val={len(src_val_texts)} test={len(src_test_texts)}")
    print(f"[Data] Target: unlabeled(train)={len(tgt_unl_texts)} val={len(tgt_val_texts)} test={len(tgt_test_texts)}")

    pi_s1 = float(np.mean(src_train_labels))
    print(f"[Source] pi_s1={pi_s1:.4f}")

    # Save run config
    cfg_dump = {
        "model_name": model_name,
        "source_epochs": SOURCE_EPOCHS,
        "adapt_rounds": ADAPT_ROUNDS,
        "adapt_epochs": ADAPT_EPOCHS,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "use_temperature_scaling": USE_TEMPERATURE_SCALING,
        "use_ema_teacher": USE_EMA_TEACHER,
        "ema_decay": EMA_DECAY,
        "use_consistency": USE_CONSISTENCY,
        "consistency_lambda": CONSISTENCY_LAMBDA,
        "use_target_val_for_selection": USE_TARGET_VAL_FOR_SELECTION,
        "tune_threshold_on_target_val": TUNE_THRESHOLD_ON_TARGET_VAL,
        "timestamp": now_str(),
    }
    save_json(cfg_dump, os.path.join(out_dir, "run_config.json"))

    # Stage0 supervised
    print("\n[Stage0] Supervised training on source...")
    train_cfg = TrainCfg(
        epochs=SOURCE_EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_len=MAX_LEN,
        grad_clip=GRAD_CLIP,
        num_workers=0,
    )
    stage0 = train_stage(
        student=student,
        teacher=None,
        tokenizer=tokenizer,
        device=device,
        train_texts=src_train_texts,
        train_labels=src_train_labels,
        train_weights=[1.0] * len(src_train_texts),
        val_texts=src_val_texts,
        val_labels=src_val_labels,
        cfg=train_cfg,
        out_dir=out_dir,
        stage_name="source_supervised",
        use_consistency=False,
        consistency_lambda=0.0,
        consistency_conf_thr=0.0,
        ema_decay=EMA_DECAY,
    )
    print(f"[Stage0] best source-val macroF1@0.5 = {stage0['best_macro_f1']:.4f}")

    # Build loaders
    src_val_loader = make_loader(tokenizer, src_val_texts, src_val_labels, None, BATCH_SIZE, MAX_LEN, 0, shuffle=False)
    src_test_loader = make_loader(tokenizer, src_test_texts, src_test_labels, None, BATCH_SIZE, MAX_LEN, 0, shuffle=False)
    tgt_val_loader = make_loader(tokenizer, tgt_val_texts, tgt_val_labels, None, BATCH_SIZE, MAX_LEN, 0, shuffle=False)
    tgt_test_loader = make_loader(tokenizer, tgt_test_texts, tgt_test_labels, None, BATCH_SIZE, MAX_LEN, 0, shuffle=False)
    tgt_unl_loader = make_loader(tokenizer, tgt_unl_texts, None, None, BATCH_SIZE, MAX_LEN, 0, shuffle=False)

    # Temperature scaling
    scaler: Optional[TemperatureScaler] = None
    if USE_TEMPERATURE_SCALING:
        print("\n[Calib] Temperature scaling on source validation...")
        scaler = TemperatureScaler().to(device)
        val_logits = predict_logits(student, src_val_loader, device).to(device)
        val_labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
        T = scaler.fit(val_logits, val_labels_t, max_iter=TS_MAX_ITER)
        print(f"[Calib] Temperature T={T:.4f}")
    else:
        print("\n[Calib] Disabled.")

    # EMA teacher init
    teacher: Optional[nn.Module] = None
    if USE_EMA_TEACHER:
        teacher = deepcopy(student)
        teacher.eval()
        print("[EMA] Enabled.")
    else:
        print("[EMA] Disabled.")

    # For BBSE: source-val predictions
    print("\n[Prep] Compute source-val predictions for BBSE...")
    src_val_p1 = predict_p1_with_scaler(student, src_val_loader, device, scaler=scaler)
    y_true_src_val = np.array(src_val_labels, dtype=int)
    y_pred_src_val = hard_pred(src_val_p1, tau=0.5)

    # Baseline eval before adaptation
    def eval_all(tag: str, model_for_eval: nn.Module, tau_for_test: float = 0.5) -> Dict[str, Any]:
        src_p1 = predict_p1_with_scaler(model_for_eval, src_test_loader, device, scaler=scaler)
        tv_p1 = predict_p1_with_scaler(model_for_eval, tgt_val_loader, device, scaler=scaler)
        tt_p1 = predict_p1_with_scaler(model_for_eval, tgt_test_loader, device, scaler=scaler)

        rec = {
            "tag": tag,
            "src_test@tau": eval_binary(np.array(src_test_labels, dtype=int), src_p1, tau=tau_for_test),
            "tgt_val@tau": eval_binary(np.array(tgt_val_labels, dtype=int), tv_p1, tau=tau_for_test),
            "tgt_test@tau": eval_binary(np.array(tgt_test_labels, dtype=int), tt_p1, tau=tau_for_test),
        }
        print(f"[Eval:{tag}] "
              f"src_acc={rec['src_test@tau']['acc']:.4f} src_f1={rec['src_test@tau']['macro_f1']:.4f} | "
              f"tgt_val_acc={rec['tgt_val@tau']['acc']:.4f} tgt_val_f1={rec['tgt_val@tau']['macro_f1']:.4f} | "
              f"tgt_test_acc={rec['tgt_test@tau']['acc']:.4f} tgt_test_f1={rec['tgt_test@tau']['macro_f1']:.4f} "
              f"(tau={tau_for_test:.3f})")
        return rec

    history: Dict[str, Any] = {"before_adapt": None, "rounds": [], "final": None}
    history["before_adapt"] = eval_all("before_adapt", student, tau_for_test=0.5)
    save_json(history, os.path.join(out_dir, "history.json"))

    # Best-by-target-val selection
    best_state_path = os.path.join(out_dir, "best_by_target_val.pt")
    best_score = -1e18
    best_info = None

    pi_prev: Optional[float] = None

    # UDA rounds
    for r in range(1, ADAPT_ROUNDS + 1):
        print("\n" + "=" * 30)
        print(f"[UDA] Round {r}/{ADAPT_ROUNDS}")
        print("=" * 30)

        # pseudo_frac schedule
        if ADAPT_ROUNDS <= 1:
            pseudo_frac = PSEUDO_FRAC_END
        else:
            t = (r - 1) / (ADAPT_ROUNDS - 1)
            pseudo_frac = (1 - t) * PSEUDO_FRAC_START + t * PSEUDO_FRAC_END

        # Ramp for unlabeled loss
        ramp = min(1.0, r / max(1, ADAPT_ROUNDS))
        lam_u = ramp  # you can make it smoother if needed

        # Predict on target unlabeled using teacher (if enabled)
        pred_model = teacher if teacher is not None else student
        tgt_unl_p1 = predict_p1_with_scaler(pred_model, tgt_unl_loader, device, scaler=scaler)

        # OOD score: max-softmax
        scores = np.maximum(tgt_unl_p1, 1.0 - tgt_unl_p1)
        id_mask, ood_info = ood_split(scores, method=OOD_METHOD, seed=seed + 13 * r,
                                      posterior_threshold=OOD_POSTERIOR_THRESHOLD, alpha_min=ALPHA_MIN)
        id_idx = np.where(id_mask)[0]
        print(f"[OOD] method={ood_info['method']} alpha_hat={ood_info.get('alpha_hat', None)} ID={len(id_idx)}/{len(tgt_unl_texts)}")

        # Diagnostics
        tgt_pred_pos = float((tgt_unl_p1[id_mask] >= 0.5).mean()) if len(id_idx) else float("nan")
        print(f"[Diag] tgt_pred_pos@0.5={tgt_pred_pos:.4f}  {describe_probs('p1_base(ID)', tgt_unl_p1[id_mask])}  {describe_probs('score(ID)', scores[id_mask])}")

        # Estimate pi_t on ID subset
        if len(id_idx) < 20:
            pi_t1 = pi_s1
            shift_info = {"chosen": "fallback_small_id", "pi_smoothed": float(pi_t1)}
        else:
            pi_t1, shift_info = robust_choose_pi(
                method=SHIFT_METHOD,
                y_true_src_val=y_true_src_val,
                y_pred_src_val=y_pred_src_val,
                p1_tgt_id=tgt_unl_p1[id_mask],
                pred_tgt_id=hard_pred(tgt_unl_p1[id_mask], tau=0.5),
                pi_s1=pi_s1,
                pi_prev=pi_prev,
            )
        pi_prev = float(pi_t1)

        print(f"[Shift] pi_t1={pi_t1:.4f} (pi_s1={pi_s1:.4f})  chosen={shift_info.get('chosen')}")

        # Prior correction + threshold for pseudo labels
        if THRESHOLD_METHOD == "quantile":
            p1_adj = prior_correct_p1(tgt_unl_p1, pi_s1=pi_s1, pi_t1=pi_t1, eps=1e-9)
            tau_pseudo = choose_tau_by_quantile(p1_adj[id_mask], pi_t1) if len(id_idx) else 0.5
            thr_info = {"method": "quantile", "tau": float(tau_pseudo)}
        else:
            b = choose_logit_bias(pi_s1=pi_s1, pi_t1=pi_t1)
            p1_adj = sigmoid(logit(tgt_unl_p1) + b)
            tau_pseudo = 0.5
            thr_info = {"method": "logit_bias", "bias": float(b), "tau": float(tau_pseudo)}

        print(f"[Threshold] {thr_info}  {describe_probs('p1_adj(ID)', p1_adj[id_mask])}")

        # Select pseudo labels (TOP-K balanced)
        n_id = len(id_idx)
        budget = int(round(pseudo_frac * n_id))
        selected_local, sel_labels, sel_w, pseudo_info = select_pseudo_topk_balanced(
            p1_adj_id=p1_adj[id_mask],
            pi_t1=pi_t1,
            total_budget=budget,
            seed=seed + 101 * r,
        )
        selected_global = id_idx[selected_local] if len(selected_local) else np.array([], dtype=int)
        pseudo_info.update({"pseudo_frac": float(pseudo_frac), "ramp": float(ramp), "lam_u": float(lam_u), "tau_pseudo": float(tau_pseudo)})
        print(f"[Pseudo] {pseudo_info}")

        # Pseudo pool
        pseudo_pool: Dict[int, Tuple[int, float]] = getattr(run, "_pseudo_pool", {}) if ACCUMULATE_PSEUDO else {}
        if ACCUMULATE_PSEUDO:
            for gi, y, w in zip(selected_global.tolist(), sel_labels.tolist(), sel_w.tolist()):
                w_scaled = float(PSEUDO_WEIGHT * lam_u * w)
                if gi in pseudo_pool:
                    old_y, old_w = pseudo_pool[gi]
                    # keep higher weight
                    if w_scaled > old_w:
                        pseudo_pool[gi] = (int(y), w_scaled)
                else:
                    pseudo_pool[gi] = (int(y), w_scaled)
        else:
            pseudo_pool = {int(gi): (int(y), float(PSEUDO_WEIGHT * lam_u * w)) for gi, y, w in zip(selected_global, sel_labels, sel_w)}

        # attach to function attribute (simple persistent storage)
        run._pseudo_pool = pseudo_pool  # type: ignore[attr-defined]

        pos_ct = sum(1 for _, (y, _) in pseudo_pool.items() if y == 1)
        neg_ct = len(pseudo_pool) - pos_ct
        print(f"[PseudoPool] size={len(pseudo_pool)} pos={pos_ct} neg={neg_ct}")

        # Build adaptation training set: source + pseudo
        pseudo_texts = [tgt_unl_texts[i] for i in pseudo_pool.keys()]
        pseudo_labels = [pseudo_pool[i][0] for i in pseudo_pool.keys()]
        pseudo_weights = [pseudo_pool[i][1] for i in pseudo_pool.keys()]

        train_texts = src_train_texts + pseudo_texts
        train_labels = src_train_labels + pseudo_labels
        train_weights = [1.0] * len(src_train_texts) + pseudo_weights

        adapt_cfg = TrainCfg(
            epochs=ADAPT_EPOCHS,
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            max_len=MAX_LEN,
            grad_clip=GRAD_CLIP,
            num_workers=0,
        )

        print("[Adapt] Training on (source + pseudo)...")
        stage = train_stage(
            student=student,
            teacher=teacher,
            tokenizer=tokenizer,
            device=device,
            train_texts=train_texts,
            train_labels=train_labels,
            train_weights=train_weights,
            val_texts=src_val_texts,
            val_labels=src_val_labels,
            cfg=adapt_cfg,
            out_dir=out_dir,
            stage_name=f"adapt_round{r}",
            use_consistency=USE_CONSISTENCY,
            consistency_lambda=CONSISTENCY_LAMBDA,
            consistency_conf_thr=CONSISTENCY_CONF_THR,
            ema_decay=EMA_DECAY,
        )
        print(f"[Adapt] best source-val macroF1@0.5 = {stage['best_macro_f1']:.4f}")

        # Recalibrate temperature each round
        if USE_TEMPERATURE_SCALING and RECALIBRATE_EACH_ROUND and scaler is not None:
            print("[Calib] Recalibrate temperature on source val...")
            val_logits = predict_logits(student, src_val_loader, device).to(device)
            val_labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
            T = scaler.fit(val_logits, val_labels_t, max_iter=TS_MAX_ITER)
            print(f"[Calib] Temperature T={T:.4f}")

        # Evaluate and (optionally) tune tau on target val for best accuracy
        # Use STUDENT for final predictions.
        tv_p1 = predict_p1_with_scaler(student, tgt_val_loader, device, scaler=scaler)
        tt_p1 = predict_p1_with_scaler(student, tgt_test_loader, device, scaler=scaler)
        src_p1 = predict_p1_with_scaler(student, src_test_loader, device, scaler=scaler)

        tau_eval = 0.5
        tau_info = None
        if TUNE_THRESHOLD_ON_TARGET_VAL:
            tau_eval, tau_info = find_best_threshold(
                y_true=np.array(tgt_val_labels, dtype=int),
                p1=tv_p1,
                metric="acc",
                n_quantiles=THRESH_SWEEP_QUANTILES,
            )

        round_eval = {
            "src_test@0.5": eval_binary(np.array(src_test_labels, dtype=int), src_p1, tau=0.5),
            "tgt_val@0.5": eval_binary(np.array(tgt_val_labels, dtype=int), tv_p1, tau=0.5),
            "tgt_test@0.5": eval_binary(np.array(tgt_test_labels, dtype=int), tt_p1, tau=0.5),
            "tgt_val@tau*": eval_binary(np.array(tgt_val_labels, dtype=int), tv_p1, tau=tau_eval),
            "tgt_test@tau*": eval_binary(np.array(tgt_test_labels, dtype=int), tt_p1, tau=tau_eval),
            "tau_star": float(tau_eval),
            "tau_search": tau_info,
        }

        print(f"[Eval:round{r}] "
              f"tgt_val_acc@0.5={round_eval['tgt_val@0.5']['acc']:.4f}  "
              f"tgt_val_acc@tau*={round_eval['tgt_val@tau*']['acc']:.4f}  "
              f"tgt_test_acc@0.5={round_eval['tgt_test@0.5']['acc']:.4f}  "
              f"tgt_test_acc@tau*={round_eval['tgt_test@tau*']['acc']:.4f}  "
              f"(tau*={tau_eval:.3f})")

        # Model selection by target val accuracy (using tau*)
        if USE_TARGET_VAL_FOR_SELECTION:
            score = round_eval["tgt_val@tau*"]["acc"]
            if score > best_score:
                best_score = score
                best_info = {"round": r, "score": float(score), "tau_star": float(tau_eval), "round_eval": round_eval}
                torch.save(student.state_dict(), best_state_path)
                save_json(best_info, os.path.join(out_dir, "best_by_target_val.json"))
                print(f"[Select] New best by target-val accuracy: round={r} acc={score:.4f} tau*={tau_eval:.3f}")

        history["rounds"].append({
            "round": int(r),
            "ood_info": ood_info,
            "shift_info": shift_info,
            "threshold_info": thr_info,
            "pseudo_info": pseudo_info,
            "pseudo_pool_size": int(len(pseudo_pool)),
            "stage": stage,
            "eval": round_eval,
        })
        save_json(history, os.path.join(out_dir, "history.json"))

    # Final: load best checkpoint if enabled
    if USE_TARGET_VAL_FOR_SELECTION and os.path.exists(best_state_path):
        student.load_state_dict(torch.load(best_state_path, map_location=device))
        print("\n[Final] Loaded best checkpoint by target-val accuracy.")
        tau_final = best_info["tau_star"] if best_info is not None else 0.5
    else:
        tau_final = 0.5

    # Final eval
    print("\n[Final] Evaluation")
    tv_p1 = predict_p1_with_scaler(student, tgt_val_loader, device, scaler=scaler)
    tt_p1 = predict_p1_with_scaler(student, tgt_test_loader, device, scaler=scaler)
    src_p1 = predict_p1_with_scaler(student, src_test_loader, device, scaler=scaler)

    if TUNE_THRESHOLD_ON_TARGET_VAL:
        tau_final, tau_info = find_best_threshold(np.array(tgt_val_labels, dtype=int), tv_p1, metric="acc", n_quantiles=THRESH_SWEEP_QUANTILES)
    else:
        tau_info = None

    final_eval = {
        "src_test@0.5": eval_binary(np.array(src_test_labels, dtype=int), src_p1, tau=0.5),
        "tgt_val@0.5": eval_binary(np.array(tgt_val_labels, dtype=int), tv_p1, tau=0.5),
        "tgt_test@0.5": eval_binary(np.array(tgt_test_labels, dtype=int), tt_p1, tau=0.5),
        "tgt_val@tau*": eval_binary(np.array(tgt_val_labels, dtype=int), tv_p1, tau=tau_final),
        "tgt_test@tau*": eval_binary(np.array(tgt_test_labels, dtype=int), tt_p1, tau=tau_final),
        "tau_star": float(tau_final),
        "tau_search": tau_info,
        "best_by_target_val": best_info,
    }
    history["final"] = final_eval
    save_json(history, os.path.join(out_dir, "history.json"))

    print(f"[Final] tgt_test_acc@0.5={final_eval['tgt_test@0.5']['acc']:.4f}  "
          f"tgt_test_acc@tau*={final_eval['tgt_test@tau*']['acc']:.4f}  "
          f"tgt_test_F1@0.5={final_eval['tgt_test@0.5']['macro_f1']:.4f}  "
          f"tgt_test_F1@tau*={final_eval['tgt_test@tau*']['macro_f1']:.4f}  "
          f"(tau*={tau_final:.3f})")

    # Save final model
    final_path = os.path.join(out_dir, "final_model.pt")
    torch.save(student.state_dict(), final_path)
    print(f"[Final] saved to {final_path}")
    print(f"[Logs] saved to {os.path.join(out_dir, 'history.json')}")


if __name__ == "__main__":
    run()
