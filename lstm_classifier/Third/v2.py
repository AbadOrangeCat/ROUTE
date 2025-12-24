#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local 6-file training + Shift/Noise/Threshold-aware UDA (binary text classification)

Reads these 6 files by default (under --data_dir):
  sourcedata/source_train.csv
  sourcedata/source_validation.csv
  sourcedata/source_test.csv
  targetdata/train.csv   (target unlabeled; labels ignored even if present)
  targetdata/val.csv     (target validation for evaluation only)
  targetdata/test.csv    (target test for evaluation only)

Key robustness upgrades vs the previous version:
1) Label-shift estimation:
   - Better "auto" logic: BBSE is marked invalid if it is extreme (near 0/1),
     or if target predicted-positive rate at 0.5 is extreme (near 0/1),
     or if the confusion matrix condition number is too large.
   - Computes Saerens EM in parallel; auto prefers EM when BBSE looks invalid.
   - Optional prior fallback (default: source prior) when both estimates are extreme.

2) Pseudo-label selection:
   - New strategy: --pseudo_strategy hybrid (default).
     It progressively relaxes min_margin; if still empty, falls back to rank-based selection
     (top-k positives + bottom-k negatives).
   - Optional minimum positive/negative pseudo counts to avoid degenerate "all-one-class" pseudo sets.

3) Diagnostics:
   - Prints target score statistics each round (p1 min/median/max, pred_pos@0.5).

4) Training:
   - Uses torch.optim.AdamW (transformers.AdamW is deprecated).

Dependencies:
  pip install -U torch transformers scikit-learn pandas numpy tqdm

Example:
  python shift_noise_threshold_uda_local6_robust.py --data_dir . --model_name bert-base-uncased --ood_method none --use_temperature_scaling --eval_each_round
"""

import argparse
import csv
import json
import math
import os
import random
import re
import inspect
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


# -----------------------------
# Utils
# -----------------------------
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
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def safe_torch_load_state_dict(path: str, device: torch.device) -> Dict[str, Any]:
    """
    Loads state dict more safely when torch.load supports weights_only (PyTorch newer versions).
    Falls back gracefully on older versions.
    """
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        pass
    return torch.load(path, map_location=device)


def stats_str(x: np.ndarray, name: str) -> str:
    if x.size == 0:
        return f"{name}=[empty]"
    return (f"{name}(min/med/max)={float(np.min(x)):.4g}/{float(np.median(x)):.4g}/{float(np.max(x)):.4g} "
            f"mean={float(np.mean(x)):.4g}")


# -----------------------------
# Robust CSV loaders
# -----------------------------
def _open_text(path: str, encoding: str = "utf-8"):
    return open(path, "r", encoding=encoding, errors="replace", newline="")


def load_csv_text_only_robust(
    path: str,
    text_col: str = "text",
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Robustly read a CSV/TSV and keep only one column: text_col.
    Works even if the CSV has many trailing empty columns.
    """
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


def load_csv_text_label_robust(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
    sep: str = ",",
    encoding: str = "utf-8",
    allowed_labels: Tuple[int, ...] = (0, 1, -1),
    drop_bad: bool = True,
) -> pd.DataFrame:
    """
    Robustly read a CSV/TSV and keep only (text,label).

    - Finds the indices of text_col/label_col from header (fallback to 0/1).
    - Converts label to int, supports "0","1","-1" or float-ish strings "1.0".
    - If parsing fails and drop_bad=True, skips that row.
    """
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

            # Accept ints or float-ish strings like "1.0"
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


def load_table_auto(
    path: str,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> pd.DataFrame:
    """
    Auto load:
      - csv -> robust csv loader
      - tsv -> robust tsv loader
      - json/jsonl -> pandas
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

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
        raise ValueError(f"Unsupported extension: {ext}. Use csv/tsv/jsonl/json")

    if text_col not in df.columns:
        raise ValueError(f"Missing {text_col} in {path}. Columns: {list(df.columns)}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Missing {label_col} in {path}. Columns: {list(df.columns)}")
    return df


# -----------------------------
# Dataset + Collate
# -----------------------------
class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
    ):
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


# -----------------------------
# Temperature Scaling
# -----------------------------
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


# -----------------------------
# Prediction helpers
# -----------------------------
@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    all_logits = []
    for batch in tqdm(dataloader, desc="Predict", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.detach().cpu())
    return torch.cat(all_logits, dim=0)


def p1_from_logits(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, 1]


def hard_pred_from_p1(p1: np.ndarray, tau: float = 0.5) -> np.ndarray:
    return (p1 >= tau).astype(np.int64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# -----------------------------
# Train / Eval
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_len: int = 256
    grad_clip: float = 1.0
    num_workers: int = 0


@torch.inference_mode()
def eval_binary(
    y_true: np.ndarray,
    p1: np.ndarray,
    tau: float = 0.5,
) -> Dict[str, Any]:
    mask_known = np.isin(y_true, [0, 1])
    y_true_k = y_true[mask_known]
    p1_k = p1[mask_known]
    pred = (p1_k >= tau).astype(int)

    macro_f1 = f1_score(y_true_k, pred, average="macro") if len(y_true_k) else float("nan")
    acc = accuracy_score(y_true_k, pred) if len(y_true_k) else float("nan")
    pr, rc, f1_each, _ = precision_recall_fscore_support(
        y_true_k, pred, labels=[0, 1], average=None, zero_division=0
    )
    auc = None
    try:
        if len(np.unique(y_true_k)) == 2:
            auc = float(roc_auc_score(y_true_k, p1_k))
    except Exception:
        auc = None

    return {
        "n": int(len(y_true)),
        "n_known": int(len(y_true_k)),
        "macro_f1": float(macro_f1),
        "acc": float(acc),
        "precision_0": float(pr[0]) if len(pr) > 0 else None,
        "recall_0": float(rc[0]) if len(rc) > 0 else None,
        "f1_0": float(f1_each[0]) if len(f1_each) > 0 else None,
        "precision_1": float(pr[1]) if len(pr) > 1 else None,
        "recall_1": float(rc[1]) if len(rc) > 1 else None,
        "f1_1": float(f1_each[1]) if len(f1_each) > 1 else None,
        "auc": auc,
    }


def train_one_stage(
    model: nn.Module,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    train_weights: Optional[List[float]],
    val_texts: List[str],
    val_labels: List[int],
    cfg: TrainConfig,
    device: torch.device,
    output_dir: str,
    stage_name: str,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    collate_fn = make_collate_fn(tokenizer, cfg.max_len)

    train_ds = TextDataset(train_texts, train_labels, train_weights)
    val_ds = TextDataset(val_texts, val_labels, weights=None)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn
    )

    model.to(device)

    # Use PyTorch AdamW (transformers.AdamW is deprecated)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * len(train_loader))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ce = nn.CrossEntropyLoss(reduction="none")
    best_f1 = -1.0
    best_path = os.path.join(output_dir, f"{stage_name}_best.pt")

    history = {"stage": stage_name, "epochs": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Train {stage_name} e{epoch}/{cfg.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss_vec = ce(logits, labels)
            loss = (loss_vec * weights).mean()
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        # validation
        model.eval()
        val_logits = predict_logits(model, val_loader, device)
        val_p1 = p1_from_logits(val_logits)
        val_metrics = eval_binary(np.array(val_labels, dtype=int), val_p1, tau=0.5)

        ep_info = {
            "epoch": int(epoch),
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "val_macro_f1@0.5": val_metrics["macro_f1"],
            "val_acc@0.5": val_metrics["acc"],
        }
        history["epochs"].append(ep_info)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(safe_torch_load_state_dict(best_path, device))
    save_json(history, os.path.join(output_dir, f"{stage_name}_train_history.json"))

    return {
        "best_macro_f1": float(best_f1),
        "best_ckpt": best_path,
    }


# -----------------------------
# OOD split
# -----------------------------
def ood_split(
    scores: np.ndarray,
    method: str,
    seed: int,
    posterior_threshold: float,
    alpha_min: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    scores higher => more ID-like
    """
    n = len(scores)
    if method == "none":
        id_mask = np.ones(n, dtype=bool)
        return id_mask, {"method": "none", "alpha_hat": 0.0}

    if method == "quantile":
        if not (0.0 <= alpha_min < 1.0):
            raise ValueError("alpha_min must be in [0,1).")
        thr = float(np.quantile(scores, alpha_min))
        id_mask = scores >= thr
        return id_mask, {
            "method": "quantile",
            "alpha_min": float(alpha_min),
            "threshold": thr,
            "alpha_hat": float(1.0 - id_mask.mean()),
        }

    if method == "gmm":
        x = scores.reshape(-1, 1).astype(np.float64)
        gmm = GaussianMixture(n_components=2, random_state=seed)
        gmm.fit(x)
        means = gmm.means_.reshape(-1)
        id_comp = int(np.argmax(means))
        proba = gmm.predict_proba(x)
        id_prob = proba[:, id_comp]
        id_mask = id_prob >= posterior_threshold
        return id_mask, {
            "method": "gmm",
            "means": means.tolist(),
            "id_component": id_comp,
            "posterior_threshold": float(posterior_threshold),
            "alpha_hat": float(1.0 - id_mask.mean()),
        }

    raise ValueError(f"Unknown ood_method: {method}")


# -----------------------------
# Label shift estimation (BBSE / EM)
# -----------------------------
def estimate_prior_bbse_binary(
    y_true_src: np.ndarray,
    y_pred_src: np.ndarray,
    y_pred_tgt: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, Any]]:
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
    info = {
        "method": "bbse",
        "C": C.tolist(),
        "confusion_counts": cm.tolist(),
        "q1_predpos@0.5": float(q1),
        "pi_raw": pi.tolist(),
        "pi1_clipped": pi1_clip,
        "cond_number": cond,
    }
    return pi1_clip, info


def estimate_prior_saerens_em_binary(
    p1_src_on_tgt: np.ndarray,
    pi_s1: float,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, Any]]:
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
    return pi_t1, {"method": "saerens_em", "pi_t1": pi_t1, "iters": int(it + 1)}


def prior_correct_p1(
    p1_src: np.ndarray,
    pi_s1: float,
    pi_t1: float,
    eps: float = 1e-9,
) -> np.ndarray:
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


# -----------------------------
# Pseudo-label selection (robust)
# -----------------------------
def _compute_budgets(
    total_budget: int,
    pi_t1: float,
    min_pos: int,
    min_neg: int,
) -> Tuple[int, int]:
    """
    Returns (n_pos, n_neg) for pseudo labels with optional minimums.
    """
    if total_budget <= 0:
        return 0, 0
    pi_t1 = float(np.clip(pi_t1, 1e-6, 1.0 - 1e-6))
    n_pos = int(round(total_budget * pi_t1))
    n_neg = total_budget - n_pos

    # Apply minimums (while keeping sum <= total_budget)
    min_pos = max(0, int(min_pos))
    min_neg = max(0, int(min_neg))

    if min_pos + min_neg > total_budget:
        # scale down proportionally
        # keep at least 1 for each if possible
        if total_budget >= 2:
            min_pos = min(min_pos, total_budget - 1)
            min_neg = min(min_neg, total_budget - min_pos)
        else:
            min_pos, min_neg = total_budget, 0

    n_pos = max(n_pos, min_pos)
    n_neg = max(n_neg, min_neg)

    # Fix overflow
    if n_pos + n_neg > total_budget:
        overflow = n_pos + n_neg - total_budget
        # reduce the larger side
        if n_pos >= n_neg and n_pos > min_pos:
            n_pos = max(min_pos, n_pos - overflow)
        else:
            n_neg = max(min_neg, n_neg - overflow)

    # Final adjust if still overflow
    while n_pos + n_neg > total_budget:
        if n_pos > min_pos:
            n_pos -= 1
        elif n_neg > min_neg:
            n_neg -= 1
        else:
            break

    return int(n_pos), int(n_neg)


def select_pseudo_labels_rank(
    p1_adj_id: np.ndarray,
    tau: float,
    pi_t1: float,
    total_budget: int,
    seed: int,
    min_pos: int = 0,
    min_neg: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Rank-based selection (always works):
      - pick top-n_pos p1 as positive
      - pick bottom-n_neg p1 as negative
    """
    rng = np.random.RandomState(seed)
    n = len(p1_adj_id)
    if n == 0 or total_budget <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), {"selected": 0, "strategy": "rank"}

    total_budget = min(total_budget, n)
    n_pos, n_neg = _compute_budgets(total_budget, pi_t1, min_pos=min_pos, min_neg=min_neg)

    # If budgets exceed n (can happen after min constraints), shrink
    if n_pos + n_neg > total_budget:
        n_pos = min(n_pos, total_budget)
        n_neg = total_budget - n_pos

    order = np.argsort(p1_adj_id)  # ascending
    neg_idx = order[:n_neg]
    pos_idx = order[-n_pos:] if n_pos > 0 else np.array([], dtype=int)

    selected = np.concatenate([pos_idx, neg_idx], axis=0).astype(int)
    rng.shuffle(selected)

    labels = (p1_adj_id[selected] >= tau).astype(int)
    # For rank selection, force labels by bucket membership (safer):
    forced_labels = np.zeros_like(labels)
    # Mark those that are in pos_idx as 1
    pos_set = set(pos_idx.tolist())
    for i, idx in enumerate(selected.tolist()):
        forced_labels[i] = 1 if idx in pos_set else 0

    # weights based on distance to tau (bigger distance => higher confidence)
    margin = np.abs(p1_adj_id[selected] - tau)
    denom = max(tau, 1.0 - tau, 1e-6)
    weights = np.clip(margin / denom, 0.0, 1.0).astype(np.float32)

    info = {
        "strategy": "rank",
        "n_id": int(n),
        "total_budget": int(total_budget),
        "selected": int(len(selected)),
        "selected_pos": int(forced_labels.sum()),
        "selected_neg": int(len(forced_labels) - forced_labels.sum()),
    }
    return selected, forced_labels.astype(int), weights, info


def select_pseudo_labels_margin(
    p1_adj_id: np.ndarray,
    tau: float,
    pi_t1: float,
    total_budget: int,
    min_margin: float,
    seed: int,
    min_pos: int = 0,
    min_neg: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    n = len(p1_adj_id)
    if n == 0 or total_budget <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), {"selected": 0, "strategy": "margin"}

    total_budget = min(total_budget, n)
    margin = np.abs(p1_adj_id - tau)
    cand_mask = margin >= float(min_margin)
    cand_idx = np.where(cand_mask)[0]

    if cand_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), {
            "strategy": "margin",
            "n_id": int(n),
            "min_margin": float(min_margin),
            "cand_total": 0,
            "selected": 0,
        }

    n_pos, n_neg = _compute_budgets(total_budget, pi_t1, min_pos=min_pos, min_neg=min_neg)
    pseudo = (p1_adj_id >= tau).astype(np.int64)

    pos_cands = cand_idx[pseudo[cand_idx] == 1]
    neg_cands = cand_idx[pseudo[cand_idx] == 0]

    # Sort by margin descending
    pos_sorted = pos_cands[np.argsort(-margin[pos_cands])] if len(pos_cands) else np.array([], dtype=int)
    neg_sorted = neg_cands[np.argsort(-margin[neg_cands])] if len(neg_cands) else np.array([], dtype=int)

    pos_take = pos_sorted[:n_pos]
    neg_take = neg_sorted[:n_neg]
    selected = np.concatenate([pos_take, neg_take], axis=0)

    # Fill remaining from highest margin among remaining candidates
    if len(selected) < total_budget:
        remaining_budget = total_budget - len(selected)
        remaining = np.setdiff1d(cand_idx, selected, assume_unique=False)
        remaining_sorted = remaining[np.argsort(-margin[remaining])]
        fill = remaining_sorted[:remaining_budget]
        selected = np.concatenate([selected, fill], axis=0)

    selected = selected.astype(int)
    rng.shuffle(selected)

    # labels are model's pseudo labels (>=tau)
    labels = pseudo[selected].astype(int)
    denom = max(tau, 1.0 - tau, 1e-6)
    weights = np.clip(margin[selected] / denom, 0.0, 1.0).astype(np.float32)

    info = {
        "strategy": "margin",
        "n_id": int(n),
        "min_margin": float(min_margin),
        "cand_total": int(len(cand_idx)),
        "total_budget": int(total_budget),
        "selected": int(len(selected)),
        "selected_pos": int(labels.sum()),
        "selected_neg": int(len(labels) - labels.sum()),
        "cand_pos": int(len(pos_cands)),
        "cand_neg": int(len(neg_cands)),
        "budget_pos": int(n_pos),
        "budget_neg": int(n_neg),
    }
    return selected, labels, weights, info


def select_pseudo_labels_hybrid(
    p1_adj_id: np.ndarray,
    tau: float,
    pi_t1: float,
    pseudo_frac: float,
    min_margin: float,
    max_total: Optional[int],
    seed: int,
    min_pos: int,
    min_neg: int,
    margin_decay: float = 0.5,
    margin_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n = len(p1_adj_id)
    total_budget = int(round(pseudo_frac * n))
    if max_total is not None:
        total_budget = min(total_budget, int(max_total))
    total_budget = max(total_budget, 0)

    # Try margin selection with progressively relaxed margins
    cur = float(min_margin)
    attempts = 0
    while cur >= margin_floor:
        sel, lab, w, info = select_pseudo_labels_margin(
            p1_adj_id=p1_adj_id,
            tau=tau,
            pi_t1=pi_t1,
            total_budget=total_budget,
            min_margin=cur,
            seed=seed + attempts * 17,
            min_pos=min_pos,
            min_neg=min_neg,
        )
        info["attempt"] = attempts
        if info.get("selected", 0) > 0:
            info["hybrid_used"] = "margin"
            return sel, lab, w, info
        if cur == 0.0:
            break
        cur = max(margin_floor, cur * margin_decay)
        attempts += 1

    # Fallback to rank selection
    sel, lab, w, info2 = select_pseudo_labels_rank(
        p1_adj_id=p1_adj_id,
        tau=tau,
        pi_t1=pi_t1,
        total_budget=total_budget,
        seed=seed + 999,
        min_pos=min_pos,
        min_neg=min_neg,
    )
    info2["hybrid_used"] = "rank"
    info2["min_margin_start"] = float(min_margin)
    return sel, lab, w, info2


# -----------------------------
# Main UDA
# -----------------------------
def make_loader_texts_labels(
    tokenizer,
    texts: List[str],
    labels: Optional[List[int]],
    batch_size: int,
    max_len: int,
    num_workers: int,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    ds = TextDataset(texts, labels=labels, weights=None)
    collate_fn = make_collate_fn(tokenizer, max_len)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn
    )


def predict_p1(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: Optional[TemperatureScaler] = None,
) -> np.ndarray:
    logits = predict_logits(model, loader, device)
    if scaler is not None:
        logits = scaler(logits.to(device)).cpu()
    return p1_from_logits(logits)


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = get_device()
    print(f"[Device] {device}")

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # ---- Load 6 files ----
    print("[Data] Loading files...")
    src_train_df = load_table_auto(args.source_train, text_col=args.text_col, label_col=args.label_col)
    src_val_df = load_table_auto(args.source_val, text_col=args.text_col, label_col=args.label_col)
    src_test_df = load_table_auto(args.source_test, text_col=args.text_col, label_col=args.label_col)

    # target unlabeled: ignore labels by design, even if CSV contains label column
    tgt_unl_df = load_table_auto(args.target_unlabeled, text_col=args.text_col, label_col=None)
    tgt_val_df = load_table_auto(args.target_val, text_col=args.text_col, label_col=args.label_col)
    tgt_test_df = load_table_auto(args.target_test, text_col=args.text_col, label_col=args.label_col)

    # Convert to python lists
    src_train_texts = src_train_df[args.text_col].map(to_str).tolist()
    src_train_labels = src_train_df[args.label_col].astype(int).tolist()

    src_val_texts = src_val_df[args.text_col].map(to_str).tolist()
    src_val_labels = src_val_df[args.label_col].astype(int).tolist()

    src_test_texts = src_test_df[args.text_col].map(to_str).tolist()
    src_test_labels = src_test_df[args.label_col].astype(int).tolist()

    tgt_unl_texts = tgt_unl_df[args.text_col].map(to_str).tolist()

    tgt_val_texts = tgt_val_df[args.text_col].map(to_str).tolist()
    tgt_val_labels = tgt_val_df[args.label_col].astype(int).tolist()

    tgt_test_texts = tgt_test_df[args.text_col].map(to_str).tolist()
    tgt_test_labels = tgt_test_df[args.label_col].astype(int).tolist()

    print(f"[Data] Source: train={len(src_train_texts)} val={len(src_val_texts)} test={len(src_test_texts)}")
    print(f"[Data] Target: unlabeled(train)={len(tgt_unl_texts)} val={len(tgt_val_texts)} test={len(tgt_test_texts)}")

    if len(src_train_texts) == 0:
        raise ValueError("Source train set is empty. Check your CSV parsing and file paths.")

    pi_s1 = float(np.mean(src_train_labels))
    print(f"[Source] pi_s1={pi_s1:.4f}")

    # Save config
    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    # ---- Stage 0: source supervised ----
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_len=args.max_len,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
    )
    print("\n[Stage0] Supervised training on source...")
    stage0 = train_one_stage(
        model=model,
        tokenizer=tokenizer,
        train_texts=src_train_texts,
        train_labels=src_train_labels,
        train_weights=[1.0] * len(src_train_texts),
        val_texts=src_val_texts,
        val_labels=src_val_labels,
        cfg=train_cfg,
        device=device,
        output_dir=args.output_dir,
        stage_name="source_supervised",
    )
    print(f"[Stage0] best source-val macroF1@0.5 = {stage0['best_macro_f1']:.4f}")

    # Build loaders
    src_val_loader = make_loader_texts_labels(tokenizer, src_val_texts, src_val_labels, args.batch_size, args.max_len, args.num_workers, shuffle=False)
    src_test_loader = make_loader_texts_labels(tokenizer, src_test_texts, src_test_labels, args.batch_size, args.max_len, args.num_workers, shuffle=False)
    tgt_val_loader = make_loader_texts_labels(tokenizer, tgt_val_texts, tgt_val_labels, args.batch_size, args.max_len, args.num_workers, shuffle=False)
    tgt_test_loader = make_loader_texts_labels(tokenizer, tgt_test_texts, tgt_test_labels, args.batch_size, args.max_len, args.num_workers, shuffle=False)

    # ---- Temperature scaling on source val ----
    scaler: Optional[TemperatureScaler] = None
    if args.use_temperature_scaling:
        print("\n[Calib] Temperature scaling on source validation...")
        scaler = TemperatureScaler().to(device)
        val_logits = predict_logits(model, src_val_loader, device).to(device)
        val_labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
        T = scaler.fit(val_logits, val_labels_t, max_iter=args.ts_max_iter)
        print(f"[Calib] Temperature T={T:.4f}")
    else:
        print("\n[Calib] Disabled (use_temperature_scaling=False).")

    # ---- Source-val predictions for BBSE ----
    print("\n[Prep] Compute source-val predictions for BBSE...")
    src_val_p1_for_bbse = predict_p1(model, src_val_loader, device, scaler=scaler)
    y_true_src_val = np.array(src_val_labels, dtype=int)
    y_pred_src_val = hard_pred_from_p1(src_val_p1_for_bbse, tau=0.5)

    # ---- Target unlabeled loader ----
    tgt_unl_loader = make_loader_texts_labels(tokenizer, tgt_unl_texts, labels=None, batch_size=args.batch_size, max_len=args.max_len, num_workers=args.num_workers, shuffle=False)

    pseudo_pool: Dict[int, Tuple[int, float]] = {}
    history: Dict[str, Any] = {"rounds": []}

    # Optional baseline eval
    def do_eval(tag: str, pi_t1: Optional[float] = None, tau: Optional[float] = None, p1_adj_method: str = "quantile") -> Dict[str, Any]:
        out: Dict[str, Any] = {"tag": tag}

        src_test_p1 = predict_p1(model, src_test_loader, device, scaler=scaler)
        tgt_val_p1 = predict_p1(model, tgt_val_loader, device, scaler=scaler)
        tgt_test_p1 = predict_p1(model, tgt_test_loader, device, scaler=scaler)

        out["source_test@0.5"] = eval_binary(np.array(src_test_labels, dtype=int), src_test_p1, tau=0.5)
        out["target_val@0.5"] = eval_binary(np.array(tgt_val_labels, dtype=int), tgt_val_p1, tau=0.5)
        out["target_test@0.5"] = eval_binary(np.array(tgt_test_labels, dtype=int), tgt_test_p1, tau=0.5)

        if pi_t1 is not None and tau is not None:
            if p1_adj_method == "quantile":
                tgt_val_p1_adj = prior_correct_p1(tgt_val_p1, pi_s1=pi_s1, pi_t1=pi_t1)
                tgt_test_p1_adj = prior_correct_p1(tgt_test_p1, pi_s1=pi_s1, pi_t1=pi_t1)
            elif p1_adj_method == "logit_bias":
                b = choose_logit_bias(pi_s1=pi_s1, pi_t1=pi_t1)
                tgt_val_p1_adj = sigmoid(logit(tgt_val_p1) + b)
                tgt_test_p1_adj = sigmoid(logit(tgt_test_p1) + b)
            else:
                tgt_val_p1_adj = tgt_val_p1
                tgt_test_p1_adj = tgt_test_p1

            out["target_val@tau(adapt)"] = eval_binary(np.array(tgt_val_labels, dtype=int), tgt_val_p1_adj, tau=tau)
            out["target_test@tau(adapt)"] = eval_binary(np.array(tgt_test_labels, dtype=int), tgt_test_p1_adj, tau=tau)

        print(f"[Eval:{tag}] "
              f"src_testF1@0.5={out['source_test@0.5']['macro_f1']:.4f} | "
              f"tgt_valF1@0.5={out['target_val@0.5']['macro_f1']:.4f} | "
              f"tgt_testF1@0.5={out['target_test@0.5']['macro_f1']:.4f}")
        if pi_t1 is not None and tau is not None:
            print(f"[Eval:{tag}] "
                  f"tgt_valF1@tau={out['target_val@tau(adapt)']['macro_f1']:.4f} | "
                  f"tgt_testF1@tau={out['target_test@tau(adapt)']['macro_f1']:.4f}")
        return out

    if args.eval_each_round:
        history["before_adapt"] = do_eval("before_adapt")
        save_json(history, os.path.join(args.output_dir, "history.json"))

    # ---- UDA rounds ----
    last_pi_t1: float = pi_s1
    last_tau: float = 0.5
    last_p1_adj_method: str = "quantile"

    for r in range(1, args.adapt_rounds + 1):
        print("\n" + "=" * 30)
        print(f"[UDA] Round {r}/{args.adapt_rounds}")
        print("=" * 30)

        tgt_unl_p1 = predict_p1(model, tgt_unl_loader, device, scaler=scaler)
        pred_base = hard_pred_from_p1(tgt_unl_p1, tau=0.5)

        # OOD score: max-softmax
        scores = np.maximum(tgt_unl_p1, 1.0 - tgt_unl_p1)
        id_mask, ood_info = ood_split(
            scores=scores,
            method=args.ood_method,
            seed=args.seed + 13 * r,
            posterior_threshold=args.ood_posterior_threshold,
            alpha_min=args.alpha_min,
        )
        id_idx = np.where(id_mask)[0]
        print(f"[OOD] method={ood_info['method']} alpha_hat={ood_info.get('alpha_hat', None)} ID={len(id_idx)}/{len(tgt_unl_texts)}")

        # Diagnostics on target
        q1 = float(np.mean(pred_base[id_mask])) if id_idx.size else float("nan")
        print(f"[Diag] tgt_pred_pos@0.5={q1:.4f}  {stats_str(tgt_unl_p1[id_mask], 'p1_base(ID)')}  {stats_str(scores[id_mask], 'score(ID)')}")

        # ---- Estimate pi_t1 on ID subset ----
        pi_t1: float = pi_s1
        shift_info: Dict[str, Any] = {}

        if id_idx.size < 10:
            pi_t1 = pi_s1
            shift_info = {"method": "fallback_small_id", "pi_t1": float(pi_t1)}
        else:
            # Compute both BBSE and EM (for robust auto)
            pi_bbse, bbse_info = estimate_prior_bbse_binary(
                y_true_src=y_true_src_val,
                y_pred_src=y_pred_src_val,
                y_pred_tgt=pred_base[id_mask],
                eps=args.bbse_eps,
            )
            pi_em, em_info = estimate_prior_saerens_em_binary(
                p1_src_on_tgt=tgt_unl_p1[id_mask],
                pi_s1=pi_s1,
                max_iter=args.em_max_iter,
                tol=args.em_tol,
                eps=args.bbse_eps,
            )

            # Validity checks
            bbse_invalid = False
            if bbse_info.get("cond_number", 0.0) > args.bbse_max_cond:
                bbse_invalid = True
            if not np.isfinite(pi_bbse):
                bbse_invalid = True
            if pi_bbse < args.pi_extreme_low or pi_bbse > args.pi_extreme_high:
                bbse_invalid = True
            if np.isfinite(q1) and (q1 < args.q_extreme_low or q1 > args.q_extreme_high):
                # If the base classifier predicts almost all-one-class at 0.5,
                # BBSE based on hard preds becomes very brittle.
                bbse_invalid = True

            em_invalid = False
            if not np.isfinite(pi_em):
                em_invalid = True
            if pi_em < args.pi_extreme_low or pi_em > args.pi_extreme_high:
                em_invalid = True

            if args.shift_method == "bbse":
                pi_t1 = pi_bbse
                shift_info = {"chosen": "bbse", "bbse": bbse_info, "em": em_info, "bbse_invalid": bbse_invalid}
            elif args.shift_method == "em":
                pi_t1 = pi_em
                shift_info = {"chosen": "em", "bbse": bbse_info, "em": em_info, "em_invalid": em_invalid}
            else:
                # auto
                if not bbse_invalid:
                    pi_t1 = pi_bbse
                    shift_info = {"chosen": "bbse", "bbse": bbse_info, "em": em_info, "bbse_invalid": bbse_invalid}
                else:
                    pi_t1 = pi_em
                    shift_info = {"chosen": "em", "bbse": bbse_info, "em": em_info, "bbse_invalid": bbse_invalid, "em_invalid": em_invalid}

            # Optional prior fallback if still extreme or non-finite
            if args.prior_fallback != "none":
                if (not np.isfinite(pi_t1)) or (pi_t1 < args.pi_extreme_low) or (pi_t1 > args.pi_extreme_high):
                    if args.prior_fallback == "source":
                        pi_t1 = float(pi_s1)
                        shift_info["fallback_applied"] = "source"
                    elif args.prior_fallback == "uniform":
                        pi_t1 = 0.5
                        shift_info["fallback_applied"] = "uniform"
                    else:
                        shift_info["fallback_applied"] = "none"

        print(f"[Shift] pi_t1={pi_t1:.4f} (pi_s1={pi_s1:.4f})  chosen={shift_info.get('chosen', shift_info.get('method','?'))}")

        # ---- Prior correction / bias ----
        if args.threshold_method == "quantile":
            p1_adj = prior_correct_p1(tgt_unl_p1, pi_s1=pi_s1, pi_t1=pi_t1)
            tau = choose_tau_by_quantile(p1_adj[id_mask], pi_t1) if id_idx.size else 0.5
            p1_adj_method = "quantile"
            thr_info = {"method": "quantile", "tau": float(tau)}
        else:
            b = choose_logit_bias(pi_s1=pi_s1, pi_t1=pi_t1)
            p1_adj = sigmoid(logit(tgt_unl_p1) + b)
            tau = 0.5
            p1_adj_method = "logit_bias"
            thr_info = {"method": "logit_bias", "bias": float(b), "tau": float(tau)}

        # Guard tau
        if not np.isfinite(tau):
            tau = 0.5
            thr_info["tau_fallback"] = 0.5
        tau = float(np.clip(tau, args.tau_clip_low, args.tau_clip_high))
        thr_info["tau"] = float(tau)

        print(f"[Threshold] {thr_info}  {stats_str(p1_adj[id_mask], 'p1_adj(ID)')}")

        # ---- Pseudo-label selection on ID subset ----
        p1_adj_id = p1_adj[id_mask]
        if args.pseudo_strategy == "rank":
            n = len(p1_adj_id)
            total_budget = int(round(args.pseudo_frac * n))
            if args.pseudo_max_total and args.pseudo_max_total > 0:
                total_budget = min(total_budget, args.pseudo_max_total)
            selected_local, sel_labels, sel_w, pseudo_info = select_pseudo_labels_rank(
                p1_adj_id=p1_adj_id,
                tau=tau,
                pi_t1=pi_t1,
                total_budget=total_budget,
                seed=args.seed + 101 * r,
                min_pos=args.pseudo_min_pos,
                min_neg=args.pseudo_min_neg,
            )
        elif args.pseudo_strategy == "margin":
            n = len(p1_adj_id)
            total_budget = int(round(args.pseudo_frac * n))
            if args.pseudo_max_total and args.pseudo_max_total > 0:
                total_budget = min(total_budget, args.pseudo_max_total)
            selected_local, sel_labels, sel_w, pseudo_info = select_pseudo_labels_margin(
                p1_adj_id=p1_adj_id,
                tau=tau,
                pi_t1=pi_t1,
                total_budget=total_budget,
                min_margin=args.min_margin,
                seed=args.seed + 101 * r,
                min_pos=args.pseudo_min_pos,
                min_neg=args.pseudo_min_neg,
            )
        else:
            selected_local, sel_labels, sel_w, pseudo_info = select_pseudo_labels_hybrid(
                p1_adj_id=p1_adj_id,
                tau=tau,
                pi_t1=pi_t1,
                pseudo_frac=args.pseudo_frac,
                min_margin=args.min_margin,
                max_total=args.pseudo_max_total if args.pseudo_max_total > 0 else None,
                seed=args.seed + 101 * r,
                min_pos=args.pseudo_min_pos,
                min_neg=args.pseudo_min_neg,
                margin_decay=args.margin_decay,
                margin_floor=args.margin_floor,
            )

        selected_global = id_idx[selected_local] if len(selected_local) else np.array([], dtype=int)
        print(f"[Pseudo] {pseudo_info}")

        # ---- Update pseudo pool ----
        if args.accumulate_pseudo:
            for gi, y, w in zip(selected_global.tolist(), sel_labels.tolist(), sel_w.tolist()):
                w_scaled = float(args.pseudo_weight * w)
                if gi in pseudo_pool:
                    old_y, old_w = pseudo_pool[gi]
                    if w_scaled > old_w:
                        pseudo_pool[gi] = (int(y), w_scaled)
                else:
                    pseudo_pool[gi] = (int(y), w_scaled)
        else:
            pseudo_pool = {int(gi): (int(y), float(args.pseudo_weight * w)) for gi, y, w in zip(selected_global, sel_labels, sel_w)}

        print(f"[PseudoPool] size={len(pseudo_pool)}")

        # ---- Adaptation training set: source + pseudo ----
        pseudo_texts = [tgt_unl_texts[i] for i in pseudo_pool.keys()]
        pseudo_labels = [pseudo_pool[i][0] for i in pseudo_pool.keys()]
        pseudo_weights = [pseudo_pool[i][1] for i in pseudo_pool.keys()]

        train_texts = src_train_texts + pseudo_texts
        train_labels = src_train_labels + pseudo_labels
        train_weights = [1.0] * len(src_train_texts) + pseudo_weights

        adapt_cfg = TrainConfig(
            epochs=args.adapt_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_len=args.max_len,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
        )
        print("[Adapt] Training on (source + pseudo)...")
        stage_adapt = train_one_stage(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            train_labels=train_labels,
            train_weights=train_weights,
            val_texts=src_val_texts,
            val_labels=src_val_labels,
            cfg=adapt_cfg,
            device=device,
            output_dir=args.output_dir,
            stage_name=f"adapt_round{r}",
        )
        print(f"[Adapt] best source-val macroF1@0.5 = {stage_adapt['best_macro_f1']:.4f}")

        # Optional recalibration each round
        if args.use_temperature_scaling and args.recalibrate_each_round and scaler is not None:
            print("[Calib] Recalibrate temperature on source val...")
            val_logits = predict_logits(model, src_val_loader, device).to(device)
            val_labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
            T = scaler.fit(val_logits, val_labels_t, max_iter=args.ts_max_iter)
            print(f"[Calib] Temperature T={T:.4f}")

        last_pi_t1, last_tau, last_p1_adj_method = float(pi_t1), float(tau), str(p1_adj_method)

        round_record = {
            "round": int(r),
            "ood_info": ood_info,
            "shift_info": shift_info,
            "threshold_info": thr_info,
            "pseudo_info": pseudo_info,
            "pseudo_pool_size": int(len(pseudo_pool)),
            "stage_adapt": stage_adapt,
        }

        if args.eval_each_round:
            round_record["eval"] = do_eval(f"round{r}", pi_t1=last_pi_t1, tau=last_tau, p1_adj_method=last_p1_adj_method)

        history["rounds"].append(round_record)
        save_json(history, os.path.join(args.output_dir, "history.json"))

    # Final eval
    print("\n[Final] Evaluation after adaptation")
    if args.eval_each_round:
        history["final"] = do_eval("final", pi_t1=last_pi_t1, tau=last_tau, p1_adj_method=last_p1_adj_method)
    else:
        history["final"] = do_eval("final")
    save_json(history, os.path.join(args.output_dir, "history.json"))

    # Save final model
    ckpt = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[Final] saved to {ckpt}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default=".", help="Directory containing sourcedata/ and targetdata/ subfolders.")
    p.add_argument("--source_train", type=str, default=None)
    p.add_argument("--source_val", type=str, default=None)
    p.add_argument("--source_test", type=str, default=None)

    p.add_argument("--target_unlabeled", type=str, default=None, help="Target unlabeled file (default: targetdata/train.csv). Labels will be ignored.")
    p.add_argument("--target_val", type=str, default=None)
    p.add_argument("--target_test", type=str, default=None)

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")

    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output_dir", type=str, default="runs/local6")
    p.add_argument("--seed", type=int, default=42)

    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--adapt_rounds", type=int, default=3)
    p.add_argument("--adapt_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)

    # calibration
    p.add_argument("--use_temperature_scaling", action="store_true")
    p.add_argument("--recalibrate_each_round", action="store_true")
    p.add_argument("--ts_max_iter", type=int, default=50)

    # OOD split
    p.add_argument("--ood_method", type=str, default="none", choices=["none", "quantile", "gmm"])
    p.add_argument("--ood_posterior_threshold", type=float, default=0.5)
    p.add_argument("--alpha_min", type=float, default=0.1)

    # shift
    p.add_argument("--shift_method", type=str, default="auto", choices=["auto", "bbse", "em"])
    p.add_argument("--bbse_eps", type=float, default=1e-6)
    p.add_argument("--bbse_max_cond", type=float, default=1e6)
    p.add_argument("--em_max_iter", type=int, default=200)
    p.add_argument("--em_tol", type=float, default=1e-6)

    # New: "extreme" thresholds to mark estimates invalid in auto mode
    p.add_argument("--pi_extreme_low", type=float, default=0.05, help="pi_t1 below this is treated as extreme in auto mode.")
    p.add_argument("--pi_extreme_high", type=float, default=0.95, help="pi_t1 above this is treated as extreme in auto mode.")
    p.add_argument("--q_extreme_low", type=float, default=0.02, help="target pred_pos@0.5 below this makes BBSE brittle.")
    p.add_argument("--q_extreme_high", type=float, default=0.98, help="target pred_pos@0.5 above this makes BBSE brittle.")
    p.add_argument("--prior_fallback", type=str, default="source", choices=["none", "source", "uniform"],
                   help="If both BBSE/EM estimates are extreme, fallback prior to 'source' or 'uniform'.")

    # threshold
    p.add_argument("--threshold_method", type=str, default="quantile", choices=["quantile", "logit_bias"])
    p.add_argument("--tau_clip_low", type=float, default=1e-4)
    p.add_argument("--tau_clip_high", type=float, default=1.0 - 1e-4)

    # pseudo labels
    p.add_argument("--pseudo_frac", type=float, default=0.25)
    p.add_argument("--min_margin", type=float, default=0.10)
    p.add_argument("--pseudo_weight", type=float, default=1.0)
    p.add_argument("--pseudo_max_total", type=int, default=0, help="0 means no limit")
    p.add_argument("--accumulate_pseudo", action="store_true")

    # New: pseudo selection robustness
    p.add_argument("--pseudo_strategy", type=str, default="hybrid", choices=["hybrid", "margin", "rank"])
    p.add_argument("--pseudo_min_pos", type=int, default=50, help="Minimum positive pseudo labels per round (when budget allows).")
    p.add_argument("--pseudo_min_neg", type=int, default=50, help="Minimum negative pseudo labels per round (when budget allows).")
    p.add_argument("--margin_decay", type=float, default=0.5, help="In hybrid strategy, shrink min_margin by this factor when empty.")
    p.add_argument("--margin_floor", type=float, default=0.0, help="Lowest margin to try in hybrid before falling back to rank.")

    # eval
    p.add_argument("--eval_each_round", action="store_true")

    args = p.parse_args()

    # Fill local 6-file defaults (under sourcedata/ and targetdata/)
    if args.source_train is None:
        args.source_train = os.path.join(args.data_dir, "sourcedata", "source_train.csv")
    if args.source_val is None:
        args.source_val = os.path.join(args.data_dir, "sourcedata", "source_validation.csv")
    if args.source_test is None:
        args.source_test = os.path.join(args.data_dir, "sourcedata", "source_test.csv")

    if args.target_unlabeled is None:
        args.target_unlabeled = os.path.join(args.data_dir, "targetdata", "train.csv")
    if args.target_val is None:
        args.target_val = os.path.join(args.data_dir, "targetdata", "val.csv")
    if args.target_test is None:
        args.target_test = os.path.join(args.data_dir, "targetdata", "test.csv")

    # Existence checks
    for k in ["source_train", "source_val", "source_test", "target_unlabeled", "target_val", "target_test"]:
        path = getattr(args, k)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for --{k}: {path}")

    # Normalization: pseudo_max_total=0 means None
    if args.pseudo_max_total == 0:
        args.pseudo_max_total = 0

    return args


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
