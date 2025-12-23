#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shift + Noise + Threshold-aware Unsupervised Domain Adaptation (UDA) for Binary Text Classification.

Implements a realistic "hard" setting:
- label shift (target class prior changes)
- conditional shift (handled via self-training on target ID subset)
- open-set noise (target includes out-of-distribution / unknown texts)
- decision threshold drift (unsupervised threshold adaptation)

Pipeline:
1) Train binary classifier on labeled source data.
2) Calibrate on source val via temperature scaling.
3) On target unlabeled: split ID vs OOD using GMM on max-softmax score (or quantile).
4) Estimate target prior pi_t with BBSE (fallback Saerens EM).
5) Prior-correct probabilities + choose threshold tau by matching predicted positive rate to pi_t.
6) Class-balanced pseudo-label self-training on target ID subset for several rounds.

Author: (You)
License: MIT-like (feel free to modify)
"""

import argparse
import json
import math
import os
import random
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
    AdamW,
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_table(path: str) -> pd.DataFrame:
    """Load CSV/TSV/JSON/JSONL into DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".tsv"]:
        return pd.read_csv(path, sep="\t")
    if ext in [".jsonl"]:
        return pd.read_json(path, lines=True)
    if ext in [".json"]:
        return pd.read_json(path)
    raise ValueError(f"Unsupported file extension: {ext}. Use csv/tsv/jsonl/json")


def validate_columns(df: pd.DataFrame, text_col: str, label_col: Optional[str] = None) -> None:
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Available: {list(df.columns)}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'. Available: {list(df.columns)}")


def to_list_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


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
            raise ValueError("labels length mismatch with texts")
        if weights is not None and len(weights) != len(texts):
            raise ValueError("weights length mismatch with texts")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        if self.weights is not None:
            item["weights"] = float(self.weights[idx])
        return item


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
# Temperature Scaling Calibration
# -----------------------------
class TemperatureScaler(nn.Module):
    """
    Temperature scaling for classification logits.
    Optimizes scalar T>0 on a labeled validation set to minimize NLL.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Avoid negative or extremely small temperature
        T = torch.clamp(self.temperature, min=1e-3)
        return logits / T

    @torch.no_grad()
    def get_temperature(self) -> float:
        return float(torch.clamp(self.temperature, min=1e-3).item())

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        """
        logits: (N, C)
        labels: (N,)
        """
        device = logits.device
        self.to(device)

        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)

        def _eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
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
    for batch in tqdm(dataloader, desc="Predict logits", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.detach().cpu()
        all_logits.append(logits)
    return torch.cat(all_logits, dim=0)


def softmax_probs_from_logits(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def binary_p1_from_logits(logits: torch.Tensor) -> np.ndarray:
    probs = softmax_probs_from_logits(logits)
    return probs[:, 1]


def hard_pred_from_logits(logits: torch.Tensor, tau: float = 0.5) -> np.ndarray:
    p1 = binary_p1_from_logits(logits)
    return (p1 >= tau).astype(np.int64)


# -----------------------------
# Training / Evaluation
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
    """
    Train on (texts, labels, weights). Evaluate on val. Save best model by macro-F1.
    """
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
    model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss_vec = ce(logits, labels)
            loss = (loss_vec * weights).mean()
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, tau=0.5)
        epoch_info = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "val_macro_f1@0.5": val_metrics["macro_f1"],
            "val_acc@0.5": val_metrics["acc"],
        }
        history["epochs"].append(epoch_info)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)

    # Load best
    model.load_state_dict(torch.load(best_path, map_location=device))
    save_json(history, os.path.join(output_dir, f"{stage_name}_train_history.json"))

    return {
        "best_macro_f1": best_f1,
        "best_ckpt": best_path,
        "history_path": os.path.join(output_dir, f"{stage_name}_train_history.json"),
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tau: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    y_p1 = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.detach().cpu()
        p1 = binary_p1_from_logits(logits)
        pred = (p1 >= tau).astype(np.int64)

        y_true.append(labels)
        y_pred.append(pred)
        y_p1.append(p1)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_p1 = np.concatenate(y_p1, axis=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    auc = None
    try:
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_p1))
    except Exception:
        auc = None

    return {
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
# Open-set (ID/OOD) split
# -----------------------------
def ood_split_by_score_gmm(
    scores: np.ndarray,
    seed: int = 42,
    posterior_threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    scores: higher => more ID-like (use max-softmax probability)
    Returns:
        id_mask: bool array
        info: dict (gmm means, estimated alpha, etc.)
    """
    x = scores.reshape(-1, 1).astype(np.float64)
    gmm = GaussianMixture(n_components=2, random_state=seed)
    gmm.fit(x)
    means = gmm.means_.reshape(-1)
    id_comp = int(np.argmax(means))
    proba = gmm.predict_proba(x)  # (N,2)
    id_prob = proba[:, id_comp]
    id_mask = id_prob >= posterior_threshold

    # estimate alpha = fraction predicted OOD
    alpha_hat = float(1.0 - id_mask.mean())

    info = {
        "method": "gmm",
        "means": means.tolist(),
        "id_component": id_comp,
        "posterior_threshold": posterior_threshold,
        "alpha_hat": alpha_hat,
        "score_min": float(scores.min()) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
        "score_mean": float(scores.mean()) if len(scores) else None,
    }
    return id_mask, info


def ood_split_by_quantile(
    scores: np.ndarray,
    alpha_min: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simple unsupervised split: assume at least alpha_min are OOD, keep top (1-alpha_min) as ID.
    """
    if not (0.0 <= alpha_min < 1.0):
        raise ValueError("alpha_min must be in [0,1).")
    thr = np.quantile(scores, alpha_min)  # bottom alpha_min are OOD
    id_mask = scores >= thr
    alpha_hat = float(1.0 - id_mask.mean())
    info = {
        "method": "quantile",
        "alpha_min": float(alpha_min),
        "threshold": float(thr),
        "alpha_hat": alpha_hat,
    }
    return id_mask, info


# -----------------------------
# Label shift estimation
# -----------------------------
def estimate_prior_bbse_binary(
    y_true_src: np.ndarray,
    y_pred_src: np.ndarray,
    y_pred_tgt: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, Any]]:
    """
    BBSE for binary:
    q = P_t(h(x)=i) = sum_j P_s(h(x)=i|y=j) * pi_t(j)
    => q = C * pi_t ; solve for pi_t

    Returns pi_t1 (positive prior) and diagnostics.
    """
    # confusion matrix counts: rows = true y, cols = pred y
    cm = confusion_matrix(y_true_src, y_pred_src, labels=[0, 1]).astype(np.float64)
    # cm[true, pred]
    # Build C_{pred, true} = P(pred|true)
    # Column-normalize by true class count
    true_counts = cm.sum(axis=1)  # per true class
    C = np.zeros((2, 2), dtype=np.float64)
    for true_y in [0, 1]:
        denom = max(true_counts[true_y], eps)
        # pred=0
        C[0, true_y] = cm[true_y, 0] / denom
        # pred=1
        C[1, true_y] = cm[true_y, 1] / denom

    # q = P_t(pred=i)
    q1 = float((y_pred_tgt == 1).mean()) if len(y_pred_tgt) else 0.5
    q0 = 1.0 - q1
    q = np.array([q0, q1], dtype=np.float64)

    # Solve pi = pinv(C) q
    cond = float(np.linalg.cond(C))
    pi = np.linalg.pinv(C) @ q
    pi0, pi1 = float(pi[0]), float(pi[1])

    # Clip to probability simplex
    pi1_clipped = float(np.clip(pi1, eps, 1.0 - eps))
    pi0_clipped = 1.0 - pi1_clipped

    info = {
        "method": "bbse",
        "C": C.tolist(),
        "confusion_counts": cm.tolist(),
        "q": q.tolist(),
        "pi_raw": [pi0, pi1],
        "pi_clipped": [pi0_clipped, pi1_clipped],
        "cond_number": cond,
    }
    return pi1_clipped, info


def estimate_prior_saerens_em_binary(
    p1_src_model_on_tgt: np.ndarray,
    pi_s1: float,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, Any]]:
    """
    Saerens et al. EM for prior shift:
    Given base posteriors p_s(y|x) and source priors pi_s(y), estimate target priors pi_t(y)
    and corrected posteriors.

    Here binary-only; returns pi_t1.
    """
    p1 = np.clip(p1_src_model_on_tgt, eps, 1.0 - eps)
    pi_s1 = float(np.clip(pi_s1, eps, 1.0 - eps))
    pi_s0 = 1.0 - pi_s1

    # Initialize with source prior
    pi_t1 = pi_s1

    for it in range(max_iter):
        pi_t0 = 1.0 - pi_t1

        # E-step: corrected posterior proportional to (pi_t/pi_s) * p_s(y|x)
        # For y=1
        a1 = (pi_t1 / pi_s1) * p1
        # For y=0
        a0 = (pi_t0 / pi_s0) * (1.0 - p1)
        denom = a0 + a1
        denom = np.clip(denom, eps, None)
        r1 = a1 / denom  # corrected posterior for class 1

        # M-step: update priors
        new_pi_t1 = float(np.mean(r1))
        if abs(new_pi_t1 - pi_t1) < tol:
            pi_t1 = new_pi_t1
            break
        pi_t1 = new_pi_t1

    pi_t1 = float(np.clip(pi_t1, eps, 1.0 - eps))
    info = {
        "method": "saerens_em",
        "pi_t1": pi_t1,
        "pi_s1": float(pi_s1),
        "iters": it + 1,
        "tol": float(tol),
    }
    return pi_t1, info


# -----------------------------
# Prior correction + threshold adaptation
# -----------------------------
def prior_correct_p1(
    p1_src: np.ndarray,
    pi_s1: float,
    pi_t1: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Given base posterior p_s(y=1|x), apply prior correction under (approx) label shift:
    p_t(y=1|x) ∝ (pi_t1/pi_s1) * p_s(1|x)
    p_t(y=0|x) ∝ ((1-pi_t1)/(1-pi_s1)) * (1-p_s(1|x))
    """
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
    """
    Choose tau so that predicted positives proportion matches pi_t1:
    tau = quantile_{1-pi_t1}(p1)
    """
    pi_t1 = float(np.clip(pi_t1, 1e-6, 1.0 - 1e-6))
    tau = float(np.quantile(p1, 1.0 - pi_t1))
    # Keep in [0,1]
    tau = float(np.clip(tau, 0.0, 1.0))
    return tau


def choose_logit_bias(pi_s1: float, pi_t1: float, eps: float = 1e-9) -> float:
    """
    Bias b = logit(pi_t) - logit(pi_s)
    If base classifier uses threshold 0.5 on corrected logit, this b accounts for prior shift.
    """
    pi_s1 = float(np.clip(pi_s1, eps, 1.0 - eps))
    pi_t1 = float(np.clip(pi_t1, eps, 1.0 - eps))
    b = math.log(pi_t1 / (1.0 - pi_t1)) - math.log(pi_s1 / (1.0 - pi_s1))
    return float(b)


# -----------------------------
# Pseudo-label selection (class-balanced, margin-based)
# -----------------------------
def select_pseudo_labels_class_balanced(
    texts: List[str],
    p1_corr: np.ndarray,
    tau: float,
    pi_t1: float,
    pseudo_frac: float = 0.2,
    min_margin: float = 0.1,
    max_total: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[float], Dict[str, Any]]:
    """
    Select pseudo labels from target ID subset with:
    - pseudo label = 1 if p1 >= tau else 0
    - confidence = |p1 - tau|
    - filter by min_margin
    - class-balanced sample sizes roughly matching pi_t1

    Returns:
      indices (into texts list), pseudo_labels, weights, info
    """
    rng = np.random.RandomState(seed)

    n = len(texts)
    if n == 0:
        return [], [], [], {"selected": 0}

    p1 = p1_corr.astype(np.float64)
    margin = np.abs(p1 - tau)

    # Candidate mask
    cand_mask = margin >= min_margin
    cand_idx = np.where(cand_mask)[0]

    # Determine how many to pick
    total_budget = int(round(pseudo_frac * n))
    if max_total is not None:
        total_budget = min(total_budget, int(max_total))
    total_budget = max(total_budget, 0)

    # Class budgets
    pi_t1 = float(np.clip(pi_t1, 1e-6, 1.0 - 1e-6))
    n_pos = int(round(total_budget * pi_t1))
    n_neg = total_budget - n_pos

    # Split candidates by pseudo label
    pseudo = (p1 >= tau).astype(np.int64)
    pos_cands = cand_idx[pseudo[cand_idx] == 1]
    neg_cands = cand_idx[pseudo[cand_idx] == 0]

    # Sort by margin descending
    pos_sorted = pos_cands[np.argsort(-margin[pos_cands])] if len(pos_cands) else np.array([], dtype=int)
    neg_sorted = neg_cands[np.argsort(-margin[neg_cands])] if len(neg_cands) else np.array([], dtype=int)

    # If not enough, fill from the other side or lower-margin pool
    pos_take = pos_sorted[:n_pos]
    neg_take = neg_sorted[:n_neg]

    selected_idx = np.concatenate([pos_take, neg_take], axis=0)
    if len(selected_idx) < total_budget:
        # Fill remaining from best remaining margins regardless of class
        remaining_budget = total_budget - len(selected_idx)
        remaining = np.setdiff1d(cand_idx, selected_idx, assume_unique=False)
        remaining_sorted = remaining[np.argsort(-margin[remaining])]
        fill = remaining_sorted[:remaining_budget]
        selected_idx = np.concatenate([selected_idx, fill], axis=0)

    # Shuffle selected indices for training
    selected_idx = np.array(selected_idx, dtype=int)
    rng.shuffle(selected_idx)

    # Build outputs
    sel_labels = pseudo[selected_idx].tolist()
    # Weight: normalize margin to [0,1] roughly
    denom = max(tau, 1.0 - tau, 1e-6)
    sel_weights = np.clip(margin[selected_idx] / denom, 0.0, 1.0).tolist()

    info = {
        "n_total": n,
        "pseudo_frac": float(pseudo_frac),
        "min_margin": float(min_margin),
        "total_budget": int(total_budget),
        "selected": int(len(selected_idx)),
        "selected_pos": int(sum(sel_labels)),
        "selected_neg": int(len(sel_labels) - sum(sel_labels)),
        "cand_total": int(len(cand_idx)),
        "cand_pos": int(len(pos_cands)),
        "cand_neg": int(len(neg_cands)),
    }
    return selected_idx.tolist(), sel_labels, sel_weights, info


# -----------------------------
# Synthetic benchmark builder (optional)
# -----------------------------
def build_target_stream(
    target_labeled_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    pos_ratio: float,
    ood_ratio: float,
    n_total: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a mixed target stream:
      - (1-ood_ratio) from target_labeled_df with label ratio pos_ratio
      - ood_ratio from ood_df with label = -1

    Returns a DataFrame with columns [text, label] (label includes -1 for OOD).
    """
    rng = np.random.RandomState(seed)
    pos_ratio = float(np.clip(pos_ratio, 1e-6, 1.0 - 1e-6))
    ood_ratio = float(np.clip(ood_ratio, 0.0, 1.0))

    n_ood = int(round(n_total * ood_ratio))
    n_known = n_total - n_ood

    # Sample known with desired pos ratio
    df_pos = target_labeled_df[target_labeled_df[label_col] == 1]
    df_neg = target_labeled_df[target_labeled_df[label_col] == 0]

    n_pos = int(round(n_known * pos_ratio))
    n_neg = n_known - n_pos

    def _sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
        if k <= 0:
            return df.head(0)
        replace = len(df) < k
        return df.sample(n=k, replace=replace, random_state=int(rng.randint(0, 2**31 - 1)))

    known = pd.concat([_sample(df_pos, n_pos), _sample(df_neg, n_neg)], axis=0).sample(
        frac=1.0, random_state=int(rng.randint(0, 2**31 - 1))
    )

    # Sample ood
    ood_sample = _sample(ood_df, n_ood).copy()
    ood_sample[label_col] = -1

    mixed = pd.concat([known, ood_sample], axis=0).sample(frac=1.0, random_state=int(rng.randint(0, 2**31 - 1)))
    mixed[text_col] = mixed[text_col].map(to_list_str)
    return mixed[[text_col, label_col]].reset_index(drop=True)


# -----------------------------
# Main UDA Loop
# -----------------------------
def run_uda(
    args: argparse.Namespace,
    model: nn.Module,
    tokenizer,
    src_train_df: pd.DataFrame,
    src_val_df: pd.DataFrame,
    tgt_unl_df: pd.DataFrame,
    tgt_test_df: Optional[pd.DataFrame],
    device: torch.device,
) -> None:
    text_col = args.text_col
    label_col = args.label_col

    out_dir = args.output_dir
    ensure_dir(out_dir)

    # Prepare source data
    src_train_texts = src_train_df[text_col].map(to_list_str).tolist()
    src_train_labels = src_train_df[label_col].astype(int).tolist()
    src_val_texts = src_val_df[text_col].map(to_list_str).tolist()
    src_val_labels = src_val_df[label_col].astype(int).tolist()

    pi_s1 = float(np.mean(src_train_labels))
    print(f"[Source] pi_s1 (positive prior) = {pi_s1:.4f}")

    # Step 0: Supervised training on source
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
        output_dir=out_dir,
        stage_name="source_supervised",
    )
    print(f"[Stage0] Best source-val macro-F1@0.5 = {stage0['best_macro_f1']:.4f}")

    # Prepare loaders for calibration and later use
    collate_fn = make_collate_fn(tokenizer, args.max_len)
    src_val_loader = torch.utils.data.DataLoader(
        TextDataset(src_val_texts, src_val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Step 0.5: Temperature scaling on source val
    temperature = None
    scaler = TemperatureScaler().to(device)
    if args.use_temperature_scaling:
        val_logits = predict_logits(model, src_val_loader, device).to(device)
        val_labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
        temperature = scaler.fit(val_logits, val_labels_t)
        print(f"[Calib] Temperature = {temperature:.4f}")
    else:
        print("[Calib] Temperature scaling disabled.")

    # Pre-load target unlabeled texts
    tgt_texts_all = tgt_unl_df[text_col].map(to_list_str).tolist()
    tgt_ds_all = TextDataset(tgt_texts_all, labels=None, weights=None)
    tgt_loader_all = torch.utils.data.DataLoader(
        tgt_ds_all,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # For BBSE we need source val hard predictions
    with torch.inference_mode():
        src_val_logits_cpu = predict_logits(model, src_val_loader, device).cpu()
        if args.use_temperature_scaling:
            src_val_logits_cpu = (scaler(src_val_logits_cpu.to(device)).cpu())
        y_pred_src_val = hard_pred_from_logits(src_val_logits_cpu, tau=0.5).astype(np.int64)
        y_true_src_val = np.array(src_val_labels, dtype=np.int64)

    # Pool for pseudo-labels (accumulated)
    pseudo_pool: Dict[int, Tuple[int, float]] = {}  # idx -> (label, weight)

    uda_history = {"rounds": []}

    for r in range(1, args.adapt_rounds + 1):
        print(f"\n==============================")
        print(f"[UDA] Round {r}/{args.adapt_rounds}")
        print(f"==============================")

        # Predict on target (base probabilities)
        tgt_logits = predict_logits(model, tgt_loader_all, device)
        if args.use_temperature_scaling:
            tgt_logits = scaler(tgt_logits.to(device)).cpu()

        p1_base = binary_p1_from_logits(tgt_logits)  # base probs
        pred_base = (p1_base >= 0.5).astype(np.int64)

        # Step 1: Open-set split using max-softmax score
        # For binary, max-softmax = max(p1, 1-p1)
        scores = np.maximum(p1_base, 1.0 - p1_base)

        if args.ood_method == "gmm":
            id_mask, ood_info = ood_split_by_score_gmm(
                scores, seed=args.seed + r, posterior_threshold=args.ood_posterior_threshold
            )
        else:
            id_mask, ood_info = ood_split_by_quantile(scores, alpha_min=args.alpha_min)

        id_indices = np.where(id_mask)[0].tolist()
        ood_indices = np.where(~id_mask)[0].tolist()
        print(f"[OOD] method={ood_info['method']} alpha_hat={ood_info['alpha_hat']:.4f} "
              f"ID={len(id_indices)} OOD={len(ood_indices)}")

        # Step 2: Estimate target prior pi_t1 on ID subset
        if len(id_indices) < 10:
            print("[Shift] Too few ID samples after OOD filtering; fallback pi_t1 = pi_s1.")
            pi_t1 = pi_s1
            shift_info = {"method": "fallback_small_id", "pi_t1": pi_t1}
        else:
            y_pred_tgt_id = pred_base[id_mask]
            pi_t1_bbse, bbse_info = estimate_prior_bbse_binary(
                y_true_src=y_true_src_val,
                y_pred_src=y_pred_src_val,
                y_pred_tgt=y_pred_tgt_id,
                eps=args.bbse_eps,
            )

            # Heuristic validity checks
            invalid = False
            if not (0.0 < pi_t1_bbse < 1.0):
                invalid = True
            if bbse_info["cond_number"] > args.bbse_max_cond:
                invalid = True

            if args.shift_method == "bbse" and not invalid:
                pi_t1 = pi_t1_bbse
                shift_info = bbse_info
            else:
                # fallback to Saerens EM on ID subset
                p1_id = p1_base[id_mask]
                pi_t1_em, em_info = estimate_prior_saerens_em_binary(
                    p1_src_model_on_tgt=p1_id,
                    pi_s1=pi_s1,
                    max_iter=args.em_max_iter,
                    tol=args.em_tol,
                    eps=args.bbse_eps,
                )
                # Choose better option (if BBSE seems ok, prefer BBSE; else EM)
                if not invalid and args.shift_method in ["auto", "bbse"]:
                    pi_t1 = pi_t1_bbse
                    shift_info = {"auto_chosen": "bbse", **bbse_info}
                else:
                    pi_t1 = pi_t1_em
                    shift_info = {"auto_chosen": "em", **em_info}

        print(f"[Shift] estimated pi_t1 = {pi_t1:.4f} (source pi_s1={pi_s1:.4f})")

        # Step 3: Prior correction
        p1_corr = prior_correct_p1(p1_base, pi_s1=pi_s1, pi_t1=pi_t1, eps=1e-9)

        # Step 4: Threshold adaptation
        if args.threshold_method == "quantile":
            tau = choose_tau_by_quantile(p1_corr[id_mask], pi_t1) if len(id_indices) else 0.5
            thr_info = {"method": "quantile", "tau": float(tau)}
        else:
            b = choose_logit_bias(pi_s1=pi_s1, pi_t1=pi_t1)
            # Convert bias to equivalent tau in probability space for reporting (optional)
            # Here we keep tau=0.5 and apply bias at logit time is more exact if you keep logits.
            tau = 0.5
            thr_info = {"method": "logit_bias", "bias": float(b), "tau": float(tau)}

        print(f"[Threshold] method={thr_info['method']} tau={tau:.4f}")

        # Step 5: Select pseudo labels on target ID subset
        # We select within the ID subset only
        texts_id = [tgt_texts_all[i] for i in id_indices]
        p1_id_corr = p1_corr[id_mask]

        selected_local_idx, pseudo_labels, pseudo_weights, pseudo_info = select_pseudo_labels_class_balanced(
            texts=texts_id,
            p1_corr=p1_id_corr,
            tau=tau,
            pi_t1=pi_t1,
            pseudo_frac=args.pseudo_frac,
            min_margin=args.min_margin,
            max_total=args.pseudo_max_total if args.pseudo_max_total > 0 else None,
            seed=args.seed + 17 * r,
        )
        # Map local indices back to global indices
        selected_global_idx = [id_indices[i] for i in selected_local_idx]

        print(f"[Pseudo] selected={pseudo_info['selected']} pos={pseudo_info['selected_pos']} "
              f"neg={pseudo_info['selected_neg']} cand_total={pseudo_info['cand_total']}")

        # Update pseudo pool
        if args.accumulate_pseudo:
            for gi, y, w in zip(selected_global_idx, pseudo_labels, pseudo_weights):
                # Keep the higher confidence if duplicate
                if gi in pseudo_pool:
                    old_y, old_w = pseudo_pool[gi]
                    if w > old_w:
                        pseudo_pool[gi] = (y, w)
                else:
                    pseudo_pool[gi] = (y, w)
        else:
            pseudo_pool = {gi: (y, w) for gi, y, w in zip(selected_global_idx, pseudo_labels, pseudo_weights)}

        print(f"[PseudoPool] size={len(pseudo_pool)}")

        # Step 6: Adaptation training on (source + pseudo)
        pseudo_texts = [tgt_texts_all[i] for i in pseudo_pool.keys()]
        pseudo_labels_pool = [pseudo_pool[i][0] for i in pseudo_pool.keys()]
        pseudo_weights_pool = [args.pseudo_weight * pseudo_pool[i][1] for i in pseudo_pool.keys()]

        # Combine datasets
        train_texts = src_train_texts + pseudo_texts
        train_labels = src_train_labels + pseudo_labels_pool
        train_weights = [1.0] * len(src_train_texts) + pseudo_weights_pool

        # Train a short stage
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
            output_dir=out_dir,
            stage_name=f"adapt_round{r}",
        )
        print(f"[Adapt] round{r} best source-val macro-F1@0.5 = {stage_adapt['best_macro_f1']:.4f}")

        # Optional: recalibrate temperature each round (recommended)
        if args.use_temperature_scaling and args.recalibrate_each_round:
            val_logits = predict_logits(model, src_val_loader, device).to(device)
            val_labels_t = torch.tensor(src_val_labels, dtype=torch.long, device=device)
            temperature = scaler.fit(val_logits, val_labels_t)
            print(f"[Calib] (recalibrated) Temperature = {temperature:.4f}")

        # Record round info
        round_info = {
            "round": r,
            "ood_info": ood_info,
            "shift_info": shift_info,
            "threshold_info": thr_info,
            "pseudo_info": pseudo_info,
            "pseudo_pool_size": int(len(pseudo_pool)),
            "stage_adapt": stage_adapt,
        }
        uda_history["rounds"].append(round_info)
        save_json(uda_history, os.path.join(out_dir, "uda_history.json"))

    # -----------------------------
    # Final evaluation (optional)
    # -----------------------------
    print("\n==============================")
    print("[Final] Evaluation")
    print("==============================")

    # Save final model
    final_ckpt = os.path.join(out_dir, "final_model.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"[Final] Saved final model to: {final_ckpt}")

    # Evaluate on source val with 0.5
    src_val_metrics = evaluate_model(model, src_val_loader, device, tau=0.5)
    print(f"[Source-Val] macroF1@0.5={src_val_metrics['macro_f1']:.4f} acc@0.5={src_val_metrics['acc']:.4f}")

    if tgt_test_df is not None and args.eval_target and args.label_col in tgt_test_df.columns:
        validate_columns(tgt_test_df, text_col, label_col)
        tgt_test_texts = tgt_test_df[text_col].map(to_list_str).tolist()
        tgt_test_labels = tgt_test_df[label_col].astype(int).tolist()

        # Split known vs OOD for evaluation if label=-1 is used
        known_mask = np.array([y in [0, 1] for y in tgt_test_labels], dtype=bool)
        ood_mask = np.array([y == -1 for y in tgt_test_labels], dtype=bool)

        # Build dataloader for all test
        tgt_test_ds = TextDataset(tgt_test_texts, labels=None, weights=None)
        tgt_test_loader = torch.utils.data.DataLoader(
            tgt_test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        # Predict probs on target test
        test_logits = predict_logits(model, tgt_test_loader, device)
        if args.use_temperature_scaling:
            test_logits = scaler(test_logits.to(device)).cpu()

        p1_test = binary_p1_from_logits(test_logits)
        # Use max-softmax score as ID score for reference
        score_test = np.maximum(p1_test, 1.0 - p1_test)

        # Evaluate classification on known subset using tau=0.5 and also show AUROC
        if known_mask.sum() > 0:
            y_true_known = np.array(tgt_test_labels, dtype=int)[known_mask]
            y_pred_05 = (p1_test[known_mask] >= 0.5).astype(int)
            macro_f1_05 = f1_score(y_true_known, y_pred_05, average="macro")
            acc_05 = accuracy_score(y_true_known, y_pred_05)
            auc = None
            try:
                if len(np.unique(y_true_known)) == 2:
                    auc = float(roc_auc_score(y_true_known, p1_test[known_mask]))
            except Exception:
                auc = None
            print(f"[Target-Test Known] macroF1@0.5={macro_f1_05:.4f} acc@0.5={acc_05:.4f} auc={auc}")

        # If there are OOD labels, report a simple OOD detection baseline using score threshold via quantile
        if ood_mask.sum() > 0 and known_mask.sum() > 0:
            # naive: threshold at quantile alpha_min on test scores
            # (in practice use unlabeled target to fit GMM; this is just a quick diagnostic)
            thr = np.quantile(score_test, args.alpha_min)
            pred_ood = score_test < thr
            y_true_ood = ood_mask.astype(int)  # 1 means OOD
            y_pred_ood = pred_ood.astype(int)

            f1_ood = f1_score(y_true_ood, y_pred_ood, average="binary", zero_division=0)
            acc_ood = accuracy_score(y_true_ood, y_pred_ood)
            print(f"[Target-Test OOD] f1={f1_ood:.4f} acc={acc_ood:.4f} (baseline)")

    print("[Done]")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--source_train", type=str, default=None)
    p.add_argument("--source_val", type=str, default=None)
    p.add_argument("--target_unlabeled", type=str, default=None)
    p.add_argument("--target_test", type=str, default=None)

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")

    # Model
    p.add_argument("--model_name", type=str, default="bert-base-chinese")

    # Output
    p.add_argument("--output_dir", type=str, default="runs/exp")
    p.add_argument("--seed", type=int, default=42)

    # Training hyperparams
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

    # Calibration
    p.add_argument("--use_temperature_scaling", action="store_true")
    p.add_argument("--recalibrate_each_round", action="store_true")

    # OOD split
    p.add_argument("--ood_method", type=str, default="gmm", choices=["gmm", "quantile"])
    p.add_argument("--ood_posterior_threshold", type=float, default=0.5)
    p.add_argument("--alpha_min", type=float, default=0.1)  # for quantile OOD

    # Label shift
    p.add_argument("--shift_method", type=str, default="auto", choices=["auto", "bbse", "em"])
    p.add_argument("--bbse_eps", type=float, default=1e-6)
    p.add_argument("--bbse_max_cond", type=float, default=1e6)
    p.add_argument("--em_max_iter", type=int, default=200)
    p.add_argument("--em_tol", type=float, default=1e-6)

    # Threshold
    p.add_argument("--threshold_method", type=str, default="quantile", choices=["quantile", "logit_bias"])

    # Pseudo label
    p.add_argument("--pseudo_frac", type=float, default=0.25)
    p.add_argument("--min_margin", type=float, default=0.10)
    p.add_argument("--pseudo_weight", type=float, default=1.0)
    p.add_argument("--pseudo_max_total", type=int, default=0, help="0 means no limit")
    p.add_argument("--accumulate_pseudo", action="store_true")

    # Eval
    p.add_argument("--eval_target", action="store_true")

    # Synthetic benchmark builder
    p.add_argument("--build_synthetic_target", action="store_true")
    p.add_argument("--target_labeled", type=str, default=None, help="Labeled target (for building synthetic stream)")
    p.add_argument("--ood_data", type=str, default=None, help="OOD texts dataset (for building synthetic stream)")
    p.add_argument("--out_target_stream", type=str, default="data/target_stream.csv")
    p.add_argument("--target_pos_ratio", type=float, default=0.3)
    p.add_argument("--target_ood_ratio", type=float, default=0.2)
    p.add_argument("--target_n_total", type=int, default=5000)

    args = p.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # If building synthetic benchmark
    if args.build_synthetic_target:
        if args.target_labeled is None or args.ood_data is None:
            raise ValueError("--build_synthetic_target requires --target_labeled and --ood_data")

        df_t = load_table(args.target_labeled)
        df_o = load_table(args.ood_data)
        validate_columns(df_t, args.text_col, args.label_col)
        validate_columns(df_o, args.text_col, None)

        # Ensure labels are ints 0/1 in target_labeled
        df_t = df_t.copy()
        df_t[args.label_col] = df_t[args.label_col].astype(int)

        df_o = df_o.copy()
        # For OOD data, if label col exists ignore; we will set -1
        stream = build_target_stream(
            target_labeled_df=df_t,
            ood_df=df_o,
            text_col=args.text_col,
            label_col=args.label_col,
            pos_ratio=args.target_pos_ratio,
            ood_ratio=args.target_ood_ratio,
            n_total=args.target_n_total,
            seed=args.seed,
        )
        out_path = args.out_target_stream
        ensure_dir(os.path.dirname(out_path) or ".")
        stream.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[Synthetic] Saved target stream to: {out_path}")
        print(stream.head(3))
        return

    # Normal UDA run
    if args.source_train is None or args.source_val is None or args.target_unlabeled is None:
        raise ValueError("Need --source_train --source_val --target_unlabeled")

    src_train_df = load_table(args.source_train)
    src_val_df = load_table(args.source_val)
    tgt_unl_df = load_table(args.target_unlabeled)

    validate_columns(src_train_df, args.text_col, args.label_col)
    validate_columns(src_val_df, args.text_col, args.label_col)
    validate_columns(tgt_unl_df, args.text_col, None)

    # Enforce binary labels in source
    src_train_df = src_train_df.copy()
    src_val_df = src_val_df.copy()
    src_train_df[args.label_col] = src_train_df[args.label_col].astype(int)
    src_val_df[args.label_col] = src_val_df[args.label_col].astype(int)

    # Optional target test for evaluation
    tgt_test_df = None
    if args.target_test is not None and os.path.exists(args.target_test):
        tgt_test_df = load_table(args.target_test)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Save config
    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    run_uda(
        args=args,
        model=model,
        tokenizer=tokenizer,
        src_train_df=src_train_df,
        src_val_df=src_val_df,
        tgt_unl_df=tgt_unl_df,
        tgt_test_df=tgt_test_df,
        device=device,
    )


if __name__ == "__main__":
    main()
