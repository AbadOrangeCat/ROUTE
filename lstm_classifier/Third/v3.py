#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local 6-file training + Improved Shift/Noise/Threshold-aware UDA (binary text classification)

This version is designed to improve stability and target performance on real cross-domain binary text tasks.

Key upgrades vs previous versions:
1) More stable pseudo-labeling:
   - Default pseudo strategy is class-balanced TOP-K selection (both classes guaranteed).
   - Confidence weights are computed in *logit-margin* space (relative to decision threshold tau).
   - Optional pseudo ramp-up and pseudo pool accumulation.

2) More robust label-shift estimation:
   - Supports BBSE and Saerens EM; AUTO chooses the safer one.
   - Optional smoothing over rounds + fallback to source prior when estimates become extreme.

3) Optional EMA teacher:
   - Use an exponential moving-average teacher model to generate pseudo-labels (less oscillation).

4) Better diagnostics & evaluation:
   - Prints target predicted positive rate at 0.5, probability stats, and pseudo-label class counts.
   - If --eval_each_round is set, prints both F1@0.5 and F1@tau(adapt) on target val/test.

Default local file layout under --data_dir:
  sourcedata/source_train.csv
  sourcedata/source_validation.csv
  sourcedata/source_test.csv
  targetdata/train.csv      (target unlabeled; labels ignored)
  targetdata/val.csv        (target validation for evaluation only)
  targetdata/test.csv       (target test for evaluation only)

Robust CSV parsing:
  - Handles "trailing commas" and messy CSVs by using csv.reader and keeping only (text,label).

Recommended run (Mac MPS):
  python shift_noise_threshold_uda_local6_plus.py \
    --data_dir /path/to/Third \
    --model_name bert-base-uncased \
    --ood_method none \
    --use_ema_teacher \
    --use_temperature_scaling \
    --recalibrate_each_round \
    --shift_method auto \
    --pseudo_strategy topk_balanced \
    --eval_each_round

Dependencies:
  pip install -U torch transformers scikit-learn pandas numpy tqdm
"""

import argparse
import csv
import json
import math
import os
import random
import re
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


def safe_torch_load_state_dict(path: str, map_location: torch.device) -> Dict[str, torch.Tensor]:
    """
    torch.load is changing defaults around weights_only. This helper keeps compatibility.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def make_loader(
    tokenizer,
    texts: List[str],
    labels: Optional[List[int]],
    weights: Optional[List[float]],
    batch_size: int,
    max_len: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    ds = TextDataset(texts, labels=labels, weights=weights)
    collate_fn = make_collate_fn(tokenizer, max_len)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn
    )


# -----------------------------
# Temperature Scaling (fit on CPU for MPS stability)
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

    def fit_cpu(self, logits_cpu: torch.Tensor, labels_cpu: torch.Tensor, max_iter: int = 50) -> float:
        """
        Fit T on CPU tensors (recommended when using MPS).
        """
        self.to(torch.device("cpu"))
        logits_cpu = logits_cpu.to("cpu")
        labels_cpu = labels_cpu.to("cpu")
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits_cpu), labels_cpu)
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


def apply_temperature(logits: torch.Tensor, temperature: Optional[float]) -> torch.Tensor:
    if temperature is None:
        return logits
    T = max(float(temperature), 1e-3)
    return logits / T


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


def describe_array(x: np.ndarray) -> str:
    if len(x) == 0:
        return "empty"
    return f"{np.min(x):.6g}/{np.median(x):.6g}/{np.max(x):.6g} mean={np.mean(x):.6g}"


# -----------------------------
# Eval
# -----------------------------
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


# -----------------------------
# Training
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
    eval_every_epoch: bool = True


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
    lr_override: Optional[float] = None,
) -> Dict[str, Any]:
    ensure_dir(output_dir)

    train_loader = make_loader(
        tokenizer, train_texts, train_labels, train_weights,
        batch_size=cfg.batch_size, max_len=cfg.max_len, num_workers=cfg.num_workers, shuffle=True
    )
    val_loader = make_loader(
        tokenizer, val_texts, val_labels, weights=None,
        batch_size=cfg.batch_size, max_len=cfg.max_len, num_workers=cfg.num_workers, shuffle=False
    )

    model.to(device)
    model.train()

    lr = float(lr_override) if lr_override is not None else float(cfg.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    total_steps = max(1, cfg.epochs * len(train_loader))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ce = nn.CrossEntropyLoss(reduction="none")

    best_f1 = -1.0
    best_path = os.path.join(output_dir, f"{stage_name}_best.pt")
    history = {"stage": stage_name, "epochs": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: List[float] = []

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

        # Validation each epoch
        model.eval()
        val_logits = predict_logits(model, val_loader, device)
        val_p1 = p1_from_logits(val_logits)
        val_metrics = eval_binary(np.array(val_labels, dtype=int), val_p1, tau=0.5)

        ep_info = {
            "epoch": int(epoch),
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "val_macro_f1@0.5": float(val_metrics["macro_f1"]),
            "val_acc@0.5": float(val_metrics["acc"]),
        }
        history["epochs"].append(ep_info)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(safe_torch_load_state_dict(best_path, map_location=device))
    save_json(history, os.path.join(output_dir, f"{stage_name}_train_history.json"))

    return {"best_macro_f1": float(best_f1), "best_ckpt": best_path}


# -----------------------------
# EMA Teacher
# -----------------------------
@torch.no_grad()
def update_ema(ema_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), student_model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))


def make_ema_copy(model: nn.Module) -> nn.Module:
    import copy
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema


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
# Label shift estimation (BBSE / EM) + safe selection
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
        "q1": float(q1),
        "cond_number": float(cond),
        "pi1_raw": float(pi1),
        "pi1_clipped": float(pi1_clip),
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
    return pi_t1, {"method": "saerens_em", "pi_t1": float(pi_t1), "iters": int(it + 1)}


def pick_prior_auto(
    pi_s1: float,
    bbse: Tuple[float, Dict[str, Any]],
    em: Tuple[float, Dict[str, Any]],
    cond_max: float,
    pi_clip_low: float,
    pi_clip_high: float,
    q_extreme_low: float,
    q_extreme_high: float,
    prior_fallback: str,
    prev_pi: Optional[float],
    smooth_beta: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Decide pi_t1 by reliability rules + optional smoothing.
    """
    pi_bbse, info_b = bbse
    pi_em, info_e = em

    # reliability checks
    bbse_ok = True
    if float(info_b.get("cond_number", 0.0)) > cond_max:
        bbse_ok = False
    if not (pi_clip_low <= pi_bbse <= pi_clip_high):
        bbse_ok = False
    q1 = float(info_b.get("q1", 0.5))
    if q1 <= q_extreme_low or q1 >= q_extreme_high:
        # hard predictions are saturated -> BBSE unreliable
        bbse_ok = False

    em_ok = True
    if not (pi_clip_low <= pi_em <= pi_clip_high):
        em_ok = False

    chosen = None
    if bbse_ok:
        chosen = ("bbse", pi_bbse, info_b)
    elif em_ok:
        chosen = ("em", pi_em, info_e)
    else:
        # fallback
        if prior_fallback == "source":
            chosen = ("fallback_source", float(pi_s1), {"reason": "both_extreme"})
        elif prior_fallback == "uniform":
            chosen = ("fallback_uniform", 0.5, {"reason": "both_extreme"})
        else:
            # none: use EM even if extreme
            chosen = ("em_extreme", float(pi_em), info_e)

    method, pi = chosen[0], float(chosen[1])
    meta = dict(chosen[2])

    # smoothing over rounds
    if prev_pi is not None and 0.0 < smooth_beta < 1.0:
        pi_smooth = (1.0 - smooth_beta) * float(prev_pi) + smooth_beta * pi
        meta["pi_before_smooth"] = pi
        meta["pi_after_smooth"] = pi_smooth
        pi = pi_smooth

    pi = float(np.clip(pi, 1e-6, 1.0 - 1e-6))
    meta["chosen"] = method
    meta["pi_t1"] = pi
    meta["pi_s1"] = float(pi_s1)
    meta["bbse_ok"] = bbse_ok
    meta["em_ok"] = em_ok
    return pi, meta


# -----------------------------
# Prior correction + threshold adaptation
# -----------------------------
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


def choose_tau_by_quantile(p1_adj: np.ndarray, pi_t1: float) -> float:
    pi_t1 = float(np.clip(pi_t1, 1e-6, 1.0 - 1e-6))
    return float(np.quantile(p1_adj, 1.0 - pi_t1))


def choose_logit_bias(pi_s1: float, pi_t1: float, eps: float = 1e-9) -> float:
    pi_s1 = float(np.clip(pi_s1, eps, 1.0 - eps))
    pi_t1 = float(np.clip(pi_t1, eps, 1.0 - eps))
    return float(math.log(pi_t1 / (1.0 - pi_t1)) - math.log(pi_s1 / (1.0 - pi_s1)))


# -----------------------------
# Improved pseudo-label selection
# -----------------------------
def pseudo_weights_from_logit_margin(p1: np.ndarray, tau: float) -> np.ndarray:
    """
    Weight in [0,1): w = 1 - exp(-|logit(p) - logit(tau)|)
    """
    m = np.abs(logit(p1) - logit(np.full_like(p1, tau)))
    w = 1.0 - np.exp(-m)
    return np.clip(w, 0.0, 1.0)


def select_pseudo_topk_balanced(
    p1_adj_id: np.ndarray,
    pi_t1: float,
    tau: float,
    total_budget: int,
    min_pos: int,
    min_neg: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Always selects both classes by construction:
      - pick top-k_pos as positives
      - pick bottom-k_neg (from remaining) as negatives
    """
    rng = np.random.RandomState(seed)
    n = len(p1_adj_id)
    if n == 0 or total_budget <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), {"selected": 0}

    pi_t1 = float(np.clip(pi_t1, 1e-6, 1.0 - 1e-6))
    k_pos = int(round(total_budget * pi_t1))
    k_neg = total_budget - k_pos

    # enforce minimum per class if possible
    k_pos = max(k_pos, min_pos)
    k_neg = max(k_neg, min_neg)
    if k_pos + k_neg > n:
        # shrink proportionally but keep at least 1
        scale = n / max(k_pos + k_neg, 1)
        k_pos = max(1, int(math.floor(k_pos * scale)))
        k_neg = max(1, n - k_pos)

    # sort
    desc = np.argsort(-p1_adj_id)  # high -> pos
    pos_idx = desc[:k_pos]

    mask = np.ones(n, dtype=bool)
    mask[pos_idx] = False
    remaining = np.where(mask)[0]
    asc_rem = remaining[np.argsort(p1_adj_id[remaining])]  # low -> neg
    neg_idx = asc_rem[:k_neg]

    selected = np.concatenate([pos_idx, neg_idx], axis=0).astype(int)
    rng.shuffle(selected)

    labels = np.zeros(len(selected), dtype=int)
    # pos indices are those in pos_idx
    pos_set = set(pos_idx.tolist())
    for i, idx in enumerate(selected.tolist()):
        labels[i] = 1 if idx in pos_set else 0

    weights = pseudo_weights_from_logit_margin(p1_adj_id[selected], tau=tau)

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
        "tau": float(tau),
    }
    return selected, labels, weights, info


# -----------------------------
# Main pipeline
# -----------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_p1(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    temperature: Optional[float],
) -> np.ndarray:
    logits = predict_logits(model, loader, device)
    logits = apply_temperature(logits, temperature)
    return p1_from_logits(logits)


def do_eval_pack(
    model: nn.Module,
    device: torch.device,
    temperature: Optional[float],
    tokenizer,
    args,
    src_test: Tuple[List[str], List[int]],
    tgt_val: Tuple[List[str], List[int]],
    tgt_test: Tuple[List[str], List[int]],
    tau_adapt: Optional[float],
    pi_t1: Optional[float],
    threshold_method: str,
) -> Dict[str, Any]:
    src_test_texts, src_test_labels = src_test
    tgt_val_texts, tgt_val_labels = tgt_val
    tgt_test_texts, tgt_test_labels = tgt_test

    src_test_loader = make_loader(tokenizer, src_test_texts, src_test_labels, None, args.batch_size, args.max_len, args.num_workers, shuffle=False)
    tgt_val_loader = make_loader(tokenizer, tgt_val_texts, tgt_val_labels, None, args.batch_size, args.max_len, args.num_workers, shuffle=False)
    tgt_test_loader = make_loader(tokenizer, tgt_test_texts, tgt_test_labels, None, args.batch_size, args.max_len, args.num_workers, shuffle=False)

    src_p1 = predict_p1(model, src_test_loader, device, temperature)
    val_p1 = predict_p1(model, tgt_val_loader, device, temperature)
    test_p1 = predict_p1(model, tgt_test_loader, device, temperature)

    out = {
        "source_test@0.5": eval_binary(np.array(src_test_labels, dtype=int), src_p1, tau=0.5),
        "target_val@0.5": eval_binary(np.array(tgt_val_labels, dtype=int), val_p1, tau=0.5),
        "target_test@0.5": eval_binary(np.array(tgt_test_labels, dtype=int), test_p1, tau=0.5),
    }

    if tau_adapt is not None and pi_t1 is not None:
        if threshold_method == "quantile":
            val_adj = prior_correct_p1(val_p1, pi_s1=float(np.mean(src_test_labels)), pi_t1=pi_t1)
            test_adj = prior_correct_p1(test_p1, pi_s1=float(np.mean(src_test_labels)), pi_t1=pi_t1)
        else:
            b = choose_logit_bias(pi_s1=float(np.mean(src_test_labels)), pi_t1=pi_t1)
            val_adj = sigmoid(logit(val_p1) + b)
            test_adj = sigmoid(logit(test_p1) + b)

        out["target_val@tau"] = eval_binary(np.array(tgt_val_labels, dtype=int), val_adj, tau=tau_adapt)
        out["target_test@tau"] = eval_binary(np.array(tgt_test_labels, dtype=int), test_adj, tau=tau_adapt)

    return out


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()
    print(f"[Device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # ---- Load 6 files ----
    print("[Data] Loading files...")
    src_train_df = load_table_auto(args.source_train, text_col=args.text_col, label_col=args.label_col)
    src_val_df = load_table_auto(args.source_val, text_col=args.text_col, label_col=args.label_col)
    src_test_df = load_table_auto(args.source_test, text_col=args.text_col, label_col=args.label_col)

    tgt_unl_df = load_table_auto(args.target_unlabeled, text_col=args.text_col, label_col=None)  # ignore labels
    tgt_val_df = load_table_auto(args.target_val, text_col=args.text_col, label_col=args.label_col)
    tgt_test_df = load_table_auto(args.target_test, text_col=args.text_col, label_col=args.label_col)

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

    pi_s1 = float(np.mean(src_train_labels)) if len(src_train_labels) else 0.5
    print(f"[Source] pi_s1={pi_s1:.4f}")

    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    # ---- Stage 0: supervised on source ----
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
        lr_override=None,
    )
    print(f"[Stage0] best source-val macroF1@0.5 = {stage0['best_macro_f1']:.4f}")

    # ---- Temperature scaling on source val (optional) ----
    temperature: Optional[float] = None
    if args.use_temperature_scaling:
        print("\n[Calib] Temperature scaling on source validation...")
        src_val_loader = make_loader(tokenizer, src_val_texts, src_val_labels, None, args.batch_size, args.max_len, args.num_workers, shuffle=False)
        val_logits = predict_logits(model, src_val_loader, device).cpu()
        val_labels_t = torch.tensor(src_val_labels, dtype=torch.long).cpu()
        scaler = TemperatureScaler()
        temperature = scaler.fit_cpu(val_logits, val_labels_t, max_iter=args.ts_max_iter)
        print(f"[Calib] Temperature T={temperature:.4f}")
    else:
        print("\n[Calib] Disabled (use_temperature_scaling=False).")

    # ---- Prepare BBSE reference on source val ----
    print("\n[Prep] Compute source-val predictions for BBSE...")
    src_val_loader = make_loader(tokenizer, src_val_texts, src_val_labels, None, args.batch_size, args.max_len, args.num_workers, shuffle=False)
    src_val_logits = apply_temperature(predict_logits(model, src_val_loader, device), temperature)
    src_val_p1 = p1_from_logits(src_val_logits)
    y_true_src_val = np.array(src_val_labels, dtype=int)
    y_pred_src_val = hard_pred_from_p1(src_val_p1, tau=0.5)

    # ---- Target unlabeled loader ----
    tgt_unl_loader = make_loader(tokenizer, tgt_unl_texts, labels=None, weights=None, batch_size=args.batch_size, max_len=args.max_len, num_workers=args.num_workers, shuffle=False)

    # ---- EMA teacher ----
    ema_model: Optional[nn.Module] = None
    if args.use_ema_teacher:
        ema_model = make_ema_copy(model).to(device)
        print(f"[EMA] Enabled. decay={args.ema_decay}")
    else:
        print("[EMA] Disabled.")

    # pseudo pool: global idx -> (label, weight)
    pseudo_pool: Dict[int, Tuple[int, float]] = {}

    history: Dict[str, Any] = {"rounds": []}

    # Baseline eval
    if args.eval_each_round:
        base_eval = do_eval_pack(
            model=model,
            device=device,
            temperature=temperature,
            tokenizer=tokenizer,
            args=args,
            src_test=(src_test_texts, src_test_labels),
            tgt_val=(tgt_val_texts, tgt_val_labels),
            tgt_test=(tgt_test_texts, tgt_test_labels),
            tau_adapt=None,
            pi_t1=None,
            threshold_method=args.threshold_method,
        )
        print(f"[Eval:before] src_testF1@0.5={base_eval['source_test@0.5']['macro_f1']:.4f} | "
              f"tgt_valF1@0.5={base_eval['target_val@0.5']['macro_f1']:.4f} | "
              f"tgt_testF1@0.5={base_eval['target_test@0.5']['macro_f1']:.4f}")
        history["before"] = base_eval
        save_json(history, os.path.join(args.output_dir, "history.json"))

    # Keep best checkpoint by target_val@tau (if eval_each_round)
    best_global = {"round": 0, "metric": -1.0, "ckpt": None}

    prev_pi: Optional[float] = None

    for r in range(1, args.adapt_rounds + 1):
        print("\n" + "=" * 30)
        print(f"[UDA] Round {r}/{args.adapt_rounds}")
        print("=" * 30)

        # Choose predictor model for pseudo-labels
        pred_model = ema_model if ema_model is not None else model

        tgt_unl_p1 = predict_p1(pred_model, tgt_unl_loader, device, temperature)
        pred_pos_rate_05 = float((tgt_unl_p1 >= 0.5).mean()) if len(tgt_unl_p1) else 0.0
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
        print(f"[Diag] tgt_pred_pos@0.5={pred_pos_rate_05:.4f}  p1_base(ID)(min/med/max)={describe_array(tgt_unl_p1[id_mask])}  score(ID)(min/med/max)={describe_array(scores[id_mask])}")

        # Estimate priors on ID subset
        if len(id_idx) < 10:
            pi_t1 = float(pi_s1)
            shift_info = {"chosen": "fallback_small_id", "pi_t1": pi_t1}
        else:
            bbse = estimate_prior_bbse_binary(y_true_src_val, y_pred_src_val, pred_base[id_mask], eps=args.bbse_eps)
            em = estimate_prior_saerens_em_binary(tgt_unl_p1[id_mask], pi_s1=pi_s1, max_iter=args.em_max_iter, tol=args.em_tol, eps=args.bbse_eps)

            if args.shift_method == "bbse":
                pi_t1, shift_info = bbse[0], bbse[1]
                shift_info["chosen"] = "bbse_forced"
            elif args.shift_method == "em":
                pi_t1, shift_info = em[0], em[1]
                shift_info["chosen"] = "em_forced"
            else:
                pi_t1, shift_info = pick_prior_auto(
                    pi_s1=pi_s1,
                    bbse=bbse,
                    em=em,
                    cond_max=args.bbse_max_cond,
                    pi_clip_low=args.pi_clip_low,
                    pi_clip_high=args.pi_clip_high,
                    q_extreme_low=args.q_extreme_low,
                    q_extreme_high=args.q_extreme_high,
                    prior_fallback=args.prior_fallback,
                    prev_pi=prev_pi,
                    smooth_beta=args.pi_smooth_beta,
                )

        prev_pi = float(pi_t1)
        print(f"[Shift] pi_t1={pi_t1:.4f} (pi_s1={pi_s1:.4f})  chosen={shift_info.get('chosen', None)}")

        # Prior correction / threshold
        if args.threshold_method == "quantile":
            p1_adj = prior_correct_p1(tgt_unl_p1, pi_s1=pi_s1, pi_t1=pi_t1)
            tau = choose_tau_by_quantile(p1_adj[id_mask], pi_t1) if len(id_idx) else 0.5
            thr_info = {"method": "quantile", "tau": float(tau)}
        else:
            b = choose_logit_bias(pi_s1=pi_s1, pi_t1=pi_t1)
            p1_adj = sigmoid(logit(tgt_unl_p1) + b)
            tau = 0.5
            thr_info = {"method": "logit_bias", "bias": float(b), "tau": float(tau)}

        print(f"[Threshold] {thr_info}  p1_adj(ID)(min/med/max)={describe_array(p1_adj[id_mask])}")

        # Pseudo-label selection on ID subset (TOP-K balanced by default)
        n_id = len(id_idx)
        total_budget = int(round(args.pseudo_frac * n_id))
        total_budget = min(total_budget, args.pseudo_max_total) if args.pseudo_max_total > 0 else total_budget
        total_budget = max(0, total_budget)

        p1_adj_id = p1_adj[id_mask]

        if args.pseudo_strategy == "topk_balanced":
            selected_local, sel_labels, sel_w, pseudo_info = select_pseudo_topk_balanced(
                p1_adj_id=p1_adj_id,
                pi_t1=pi_t1,
                tau=tau,
                total_budget=total_budget,
                min_pos=args.pseudo_min_pos,
                min_neg=args.pseudo_min_neg,
                seed=args.seed + 101 * r,
            )
        else:
            raise ValueError("Only pseudo_strategy=topk_balanced is supported in this 'plus' version (for stability).")

        selected_global = id_idx[selected_local] if len(selected_local) else np.array([], dtype=int)
        # pseudo ramp-up factor
        ramp = min(1.0, float(r) / max(1.0, float(args.pseudo_rampup_rounds)))
        lam_u = float(args.pseudo_weight) * ramp

        print(f"[Pseudo] {pseudo_info}  ramp={ramp:.3f} lam_u={lam_u:.3f}")

        # Update pseudo pool (accumulate by default recommended)
        if args.accumulate_pseudo:
            for gi, y, w in zip(selected_global.tolist(), sel_labels.tolist(), sel_w.tolist()):
                w_scaled = float(lam_u * w)
                if gi in pseudo_pool:
                    old_y, old_w = pseudo_pool[gi]
                    if w_scaled > old_w:
                        pseudo_pool[gi] = (int(y), w_scaled)
                else:
                    pseudo_pool[gi] = (int(y), w_scaled)
        else:
            pseudo_pool = {int(gi): (int(y), float(lam_u * w)) for gi, y, w in zip(selected_global, sel_labels, sel_w)}

        # Simple pool stats
        pool_pos = sum(v[0] for v in pseudo_pool.values())
        pool_neg = len(pseudo_pool) - pool_pos
        print(f"[PseudoPool] size={len(pseudo_pool)} pos={pool_pos} neg={pool_neg}")

        # Build adaptation training set: source + pseudo
        pseudo_texts = [tgt_unl_texts[i] for i in pseudo_pool.keys()]
        pseudo_labels = [pseudo_pool[i][0] for i in pseudo_pool.keys()]
        pseudo_weights = [pseudo_pool[i][1] for i in pseudo_pool.keys()]

        train_texts = src_train_texts + pseudo_texts
        train_labels = src_train_labels + pseudo_labels
        train_weights = [1.0] * len(src_train_texts) + pseudo_weights

        # Adapt LR (often smaller than source fine-tune)
        adapt_lr = args.adapt_lr if args.adapt_lr is not None else args.lr
        adapt_cfg = TrainConfig(
            epochs=args.adapt_epochs,
            batch_size=args.batch_size,
            lr=args.lr,  # overridden
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
            lr_override=adapt_lr,
        )
        print(f"[Adapt] best source-val macroF1@0.5 = {stage_adapt['best_macro_f1']:.4f}")

        # Update EMA after adaptation stage (cheap: one-shot copy) OR step-wise EMA (optional)
        if ema_model is not None:
            # One-shot EMA update using current student parameters (works reasonably well round-wise).
            update_ema(ema_model, model, decay=args.ema_decay)

        # Optional recalibration each round
        if args.use_temperature_scaling and args.recalibrate_each_round:
            print("[Calib] Recalibrate temperature on source val...")
            val_logits = predict_logits(model, src_val_loader, device).cpu()
            val_labels_t = torch.tensor(src_val_labels, dtype=torch.long).cpu()
            scaler = TemperatureScaler()
            temperature = scaler.fit_cpu(val_logits, val_labels_t, max_iter=args.ts_max_iter)
            print(f"[Calib] Temperature T={temperature:.4f}")

        # Evaluation each round
        eval_rec = None
        if args.eval_each_round:
            eval_rec = do_eval_pack(
                model=model,
                device=device,
                temperature=temperature,
                tokenizer=tokenizer,
                args=args,
                src_test=(src_test_texts, src_test_labels),
                tgt_val=(tgt_val_texts, tgt_val_labels),
                tgt_test=(tgt_test_texts, tgt_test_labels),
                tau_adapt=tau,
                pi_t1=pi_t1,
                threshold_method=args.threshold_method,
            )
            print(f"[Eval:round{r}] src_testF1@0.5={eval_rec['source_test@0.5']['macro_f1']:.4f} | "
                  f"tgt_valF1@0.5={eval_rec['target_val@0.5']['macro_f1']:.4f} | "
                  f"tgt_testF1@0.5={eval_rec['target_test@0.5']['macro_f1']:.4f}")
            if "target_val@tau" in eval_rec:
                print(f"[Eval:round{r}] tgt_valF1@tau={eval_rec['target_val@tau']['macro_f1']:.4f} | "
                      f"tgt_testF1@tau={eval_rec['target_test@tau']['macro_f1']:.4f}")

            # Track best by target_val@tau (if available), else by target_val@0.5
            metric = None
            if "target_val@tau" in eval_rec:
                metric = float(eval_rec["target_val@tau"]["macro_f1"])
            else:
                metric = float(eval_rec["target_val@0.5"]["macro_f1"])
            if metric > best_global["metric"]:
                best_global = {"round": r, "metric": metric, "ckpt": stage_adapt["best_ckpt"]}

        round_record = {
            "round": int(r),
            "ood_info": ood_info,
            "shift_info": shift_info,
            "threshold_info": thr_info,
            "pseudo_info": pseudo_info,
            "pseudo_pool_size": int(len(pseudo_pool)),
            "pseudo_pool_pos": int(pool_pos),
            "pseudo_pool_neg": int(pool_neg),
            "stage_adapt": stage_adapt,
            "temperature": float(temperature) if temperature is not None else None,
        }
        if eval_rec is not None:
            round_record["eval"] = eval_rec

        history["rounds"].append(round_record)
        save_json(history, os.path.join(args.output_dir, "history.json"))

    # Restore best checkpoint (if any)
    if best_global["ckpt"] is not None:
        print(f"\n[SelectBest] Loading best ckpt from round{best_global['round']} metric={best_global['metric']:.4f}")
        model.load_state_dict(safe_torch_load_state_dict(best_global["ckpt"], map_location=device))

    # Final eval
    print("\n[Final] Evaluation after adaptation")
    final_eval = do_eval_pack(
        model=model,
        device=device,
        temperature=temperature,
        tokenizer=tokenizer,
        args=args,
        src_test=(src_test_texts, src_test_labels),
        tgt_val=(tgt_val_texts, tgt_val_labels),
        tgt_test=(tgt_test_texts, tgt_test_labels),
        tau_adapt=None,
        pi_t1=None,
        threshold_method=args.threshold_method,
    )
    print(f"[Eval:final] src_testF1@0.5={final_eval['source_test@0.5']['macro_f1']:.4f} | "
          f"tgt_valF1@0.5={final_eval['target_val@0.5']['macro_f1']:.4f} | "
          f"tgt_testF1@0.5={final_eval['target_test@0.5']['macro_f1']:.4f}")

    # Save final model
    ckpt = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[Final] saved to {ckpt}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default=".", help="Directory containing sourcedata/ and targetdata/.")

    # If not provided, filled from data_dir
    p.add_argument("--source_train", type=str, default=None)
    p.add_argument("--source_val", type=str, default=None)
    p.add_argument("--source_test", type=str, default=None)

    p.add_argument("--target_unlabeled", type=str, default=None, help="Target unlabeled file (default: targetdata/train.csv). labels ignored.")
    p.add_argument("--target_val", type=str, default=None)
    p.add_argument("--target_test", type=str, default=None)

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")

    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output_dir", type=str, default="runs/local6_plus")
    p.add_argument("--seed", type=int, default=42)

    # Train
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--adapt_rounds", type=int, default=3)
    p.add_argument("--adapt_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--adapt_lr", type=float, default=1e-5, help="Often smaller than lr for adaptation stage.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)

    # Calibration
    p.add_argument("--use_temperature_scaling", action="store_true")
    p.add_argument("--recalibrate_each_round", action="store_true")
    p.add_argument("--ts_max_iter", type=int, default=50)

    # EMA teacher
    p.add_argument("--use_ema_teacher", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)

    # OOD split
    p.add_argument("--ood_method", type=str, default="none", choices=["none", "quantile", "gmm"])
    p.add_argument("--ood_posterior_threshold", type=float, default=0.5)
    p.add_argument("--alpha_min", type=float, default=0.1)

    # Shift
    p.add_argument("--shift_method", type=str, default="auto", choices=["auto", "bbse", "em"])
    p.add_argument("--bbse_eps", type=float, default=1e-6)
    p.add_argument("--bbse_max_cond", type=float, default=1e6)
    p.add_argument("--em_max_iter", type=int, default=200)
    p.add_argument("--em_tol", type=float, default=1e-6)

    # Prior auto selection params
    p.add_argument("--pi_clip_low", type=float, default=0.05)
    p.add_argument("--pi_clip_high", type=float, default=0.95)
    p.add_argument("--q_extreme_low", type=float, default=0.02)
    p.add_argument("--q_extreme_high", type=float, default=0.98)
    p.add_argument("--prior_fallback", type=str, default="source", choices=["source", "uniform", "none"])
    p.add_argument("--pi_smooth_beta", type=float, default=0.3, help="Smooth pi_t over rounds: pi <- (1-beta)*prev + beta*new")

    # Threshold
    p.add_argument("--threshold_method", type=str, default="quantile", choices=["quantile", "logit_bias"])

    # Pseudo labels
    p.add_argument("--pseudo_strategy", type=str, default="topk_balanced", choices=["topk_balanced"])
    p.add_argument("--pseudo_frac", type=float, default=0.25)
    p.add_argument("--pseudo_weight", type=float, default=1.0)
    p.add_argument("--pseudo_rampup_rounds", type=int, default=2)
    p.add_argument("--pseudo_max_total", type=int, default=0, help="0 means no limit")
    p.add_argument("--pseudo_min_pos", type=int, default=200)
    p.add_argument("--pseudo_min_neg", type=int, default=200)
    p.add_argument("--accumulate_pseudo", action="store_true", help="Recommended for stability")

    # Eval
    p.add_argument("--eval_each_round", action="store_true")

    args = p.parse_args()

    def _p(rel: str) -> str:
        return os.path.join(args.data_dir, rel)

    if args.source_train is None:
        args.source_train = _p("sourcedata/source_train.csv")
    if args.source_val is None:
        args.source_val = _p("sourcedata/source_validation.csv")
    if args.source_test is None:
        args.source_test = _p("sourcedata/source_test.csv")

    if args.target_unlabeled is None:
        args.target_unlabeled = _p("targetdata/train.csv")
    if args.target_val is None:
        args.target_val = _p("targetdata/val.csv")
    if args.target_test is None:
        args.target_test = _p("targetdata/test.csv")

    # Existence checks
    for k in ["source_train", "source_val", "source_test", "target_unlabeled", "target_val", "target_test"]:
        path = getattr(args, k)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for --{k}: {path}")

    return args


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
