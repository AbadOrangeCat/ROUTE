
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTO-RUN (no CLI args) - Accuracy-optimized Domain Adaptation for your 6 local files.

Goal (Mode A): maximize *target test accuracy* using target validation labels for:
  - selecting the best round checkpoint
  - tuning a decision threshold tau* (threshold drift fix)

Key upgrades vs your v2 log:
  1) Fix "best checkpoint should restore its temperature" (we store & restore T).
  2) Freeze target prior pi_t to source prior by default (prevents BBSE-driven drift hurting accuracy).
  3) Use EMA teacher *for pseudo-label generation* (more stable than student).
  4) Add Mean-Teacher style consistency loss on *all* target unlabeled data (improves score ordering, thus accuracy).
  5) Keep TOP-K class-balanced pseudo labels (prevents all-positive/all-negative collapse).

Directory layout (same as your runs):
  Third/
    this_script.py
    sourcedata/
      source_train.csv
      source_validation.csv
      source_test.csv
    targetdata/
      train.csv
      val.csv
      test.csv
    runs/
      ...

Dependencies:
  pip install -U torch transformers scikit-learn pandas numpy tqdm

Notes:
- We intentionally "use target val labels" for threshold tuning & model selection (Mode A).
- We do NOT train on target val labels (only use them for selection/tuning).
"""

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


# =========================
# AUTO CONFIG (edit here)
# =========================
SEED = 42

MODEL_NAME = "bert-base-uncased"

# data paths relative to this script
TEXT_COL = "text"
LABEL_COL = "label"
SOURCE_DIR = "sourcedata"
TARGET_DIR = "targetdata"

# training hyperparams (MPS friendly)
MAX_LEN = 384
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2  # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0

EPOCHS_SOURCE = 3
ADAPT_ROUNDS = 4
EPOCHS_ADAPT_PER_ROUND = 1

# pseudo-label schedule (per round)
PSEUDO_FRAC_SCHEDULE = [0.20, 0.25, 0.30, 0.35]  # length should be >= ADAPT_ROUNDS
# ramp schedule for unlabeled-related loss weights (per round)
RAMP_SCHEDULE = [0.25, 0.50, 0.75, 1.00]

# mode A knobs (keep these True)
USE_TARGET_VAL_FOR_SELECTION = True
TUNE_THRESHOLD_ON_TARGET_VAL = True

# prior mode:
# - "fixed_source": use pi_t = pi_s always (recommended for your dataset to maximize accuracy)
# - "bbse_log_only": compute BBSE for logging but still use pi_s for training
PRIOR_MODE = "fixed_source"

# EMA teacher
USE_EMA_TEACHER = True
EMA_DECAY = 0.999

# Consistency loss on all target unlabeled data (Mean Teacher style)
USE_CONSISTENCY = True
CONSISTENCY_CONF_THRESH = 0.55  # only apply consistency on confident teacher predictions
CONSISTENCY_WEIGHT_MAX = 1.0    # multiplied by ramp schedule

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "local6_autorun_acc_v3")


# =========================
# Utils
# =========================
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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_torch_load(path: str, device: torch.device):
    # PyTorch versions differ on weights_only; handle both.
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=device)


# =========================
# Robust CSV loaders
# =========================
def _open_text(path: str, encoding: str = "utf-8"):
    return open(path, "r", encoding=encoding, errors="replace", newline="")


def load_csv_text_only_robust(path: str, text_col: str = "text", sep: str = ",") -> pd.DataFrame:
    rows: List[str] = []
    with _open_text(path) as f:
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
    allowed_labels: Tuple[int, ...] = (0, 1, -1),
) -> pd.DataFrame:
    rows: List[Tuple[str, int]] = []
    allowed = set(allowed_labels)
    with _open_text(path) as f:
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
                continue
            txt = row[text_idx]
            lab = row[label_idx].strip()
            if not re.fullmatch(r"-?\d+(?:\.0+)?", lab or ""):
                continue
            lab_int = int(float(lab))
            if lab_int not in allowed:
                continue
            rows.append((txt, lab_int))
    return pd.DataFrame(rows, columns=[text_col, label_col])


def load_table_auto(path: str, text_col: str, label_col: Optional[str]) -> pd.DataFrame:
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
        raise ValueError(f"Missing {text_col} in {path}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Missing {label_col} in {path}")
    return df


# =========================
# Dataset / Collate
# =========================
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


def make_loader(tokenizer, texts: List[str], labels: Optional[List[int]], weights: Optional[List[float]],
                shuffle: bool) -> torch.utils.data.DataLoader:
    ds = TextDataset(texts=texts, labels=labels, weights=weights)
    collate_fn = make_collate_fn(tokenizer, MAX_LEN)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)


# =========================
# Model helpers
# =========================
@torch.inference_mode()
def predict_logits(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    outs = []
    for batch in tqdm(loader, desc="Predict", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0)


def p1_from_logits(logits: torch.Tensor) -> np.ndarray:
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


# =========================
# Temperature Scaling
# =========================
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
        self.to(logits.device)
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self.get_T()


@torch.inference_mode()
def predict_p1(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device,
               scaler_T: Optional[float] = None) -> np.ndarray:
    logits = predict_logits(model, loader, device)
    if scaler_T is not None:
        logits = (logits / float(max(scaler_T, 1e-6))).detach()
    return p1_from_logits(logits)


# =========================
# EMA Teacher
# =========================
def clone_model(model: nn.Module) -> nn.Module:
    # create a new instance with same weights
    new_m = AutoModelForSequenceClassification.from_config(model.config)
    new_m.load_state_dict(model.state_dict())
    return new_m


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=(1.0 - decay))


# =========================
# Metrics / Threshold tuning (Accuracy-first)
# =========================
def eval_acc_f1(y_true: np.ndarray, p1: np.ndarray, tau: float) -> Dict[str, float]:
    pred = (p1 >= tau).astype(int)
    acc = float(accuracy_score(y_true, pred))
    f1 = float(f1_score(y_true, pred, average="macro"))
    return {"acc": acc, "f1": f1}


def tune_tau_for_accuracy(y_true: np.ndarray, p1: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Find tau maximizing accuracy on val.
    Tie-breaks:
      1) higher macro-F1
      2) tau closer to 0.5
    """
    y_true = y_true.astype(int)
    p1 = p1.astype(np.float64)

    # candidates: unique scores + sentinel
    uniq = np.unique(p1)
    # Add edge candidates
    cand = np.unique(np.concatenate([uniq, np.array([0.0, 1.0])]))
    best_tau = 0.5
    best_acc = -1.0
    best_f1 = -1.0

    for tau in cand:
        m = eval_acc_f1(y_true, p1, float(tau))
        acc = m["acc"]
        f1 = m["f1"]
        if acc > best_acc + 1e-12:
            best_acc, best_f1, best_tau = acc, f1, float(tau)
        elif abs(acc - best_acc) <= 1e-12:
            if f1 > best_f1 + 1e-12:
                best_acc, best_f1, best_tau = acc, f1, float(tau)
            elif abs(f1 - best_f1) <= 1e-12:
                # prefer tau nearer 0.5 for stability
                if abs(float(tau) - 0.5) < abs(best_tau - 0.5):
                    best_tau = float(tau)

    return best_tau, {"acc": best_acc, "f1": best_f1}


def print_confusion(y_true: np.ndarray, p1: np.ndarray, tau: float, name: str) -> None:
    pred = (p1 >= tau).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"[CM:{name}] tau={tau:.3f}  TN={tn} FP={fp} FN={fn} TP={tp}")


# =========================
# Shift estimate (BBSE) - LOG ONLY
# =========================
def estimate_pi_bbse_log_only(y_true_src: np.ndarray, y_pred_src: np.ndarray, y_pred_tgt: np.ndarray, eps: float = 1e-6) -> float:
    cm = confusion_matrix(y_true_src, y_pred_src, labels=[0, 1]).astype(np.float64)
    true_counts = cm.sum(axis=1)
    C = np.zeros((2, 2), dtype=np.float64)  # P(pred | true)
    for true_y in [0, 1]:
        denom = max(true_counts[true_y], eps)
        C[0, true_y] = cm[true_y, 0] / denom
        C[1, true_y] = cm[true_y, 1] / denom
    q1 = float((y_pred_tgt == 1).mean()) if len(y_pred_tgt) else 0.5
    q = np.array([1.0 - q1, q1], dtype=np.float64)
    pi = np.linalg.pinv(C) @ q
    return float(np.clip(pi[1], eps, 1.0 - eps))


# =========================
# Pseudo labels: TOP-K balanced
# =========================
def select_topk_balanced(p1_scores: np.ndarray, total_budget: int, pi: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select k_pos highest p1 as positive, k_neg lowest p1 as negative.
    Returns:
      idx_selected (int indices)
      y_pseudo (0/1)
      w_pseudo (0..1 confidence weight)
    """
    rng = np.random.RandomState(seed)
    n = len(p1_scores)
    total_budget = int(min(max(total_budget, 0), n))
    pi = float(np.clip(pi, 1e-6, 1.0 - 1e-6))
    k_pos = int(round(total_budget * pi))
    k_neg = total_budget - k_pos

    # rank
    order = np.argsort(p1_scores)  # ascending
    neg_idx = order[:k_neg]
    pos_idx = order[-k_pos:] if k_pos > 0 else np.array([], dtype=int)

    idx = np.concatenate([neg_idx, pos_idx], axis=0).astype(int)
    rng.shuffle(idx)

    y = (p1_scores[idx] >= 0.5).astype(int)
    # We want labels exactly: selected neg -> 0, selected pos -> 1
    # Build explicit:
    y = np.zeros(len(idx), dtype=int)
    # Mark pos positions (those indices originally from pos_idx)
    pos_set = set(pos_idx.tolist())
    for i, gi in enumerate(idx.tolist()):
        if gi in pos_set:
            y[i] = 1

    # confidence weight: distance from 0.5
    w = np.clip(np.abs(p1_scores[idx] - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    return idx, y, w


# =========================
# Training loops
# =========================
@dataclass
class TrainState:
    best_path: str
    best_metric: float
    history: List[Dict[str, Any]]


def train_supervised_source(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    src_train_texts: List[str], src_train_labels: List[int],
    src_val_texts: List[str], src_val_labels: List[int],
) -> TrainState:
    ensure_dir(OUTPUT_DIR)
    train_loader = make_loader(tokenizer, src_train_texts, src_train_labels, weights=[1.0]*len(src_train_texts), shuffle=True)
    val_loader = make_loader(tokenizer, src_val_texts, src_val_labels, weights=None, shuffle=False)

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, EPOCHS_SOURCE * math.ceil(len(train_loader) / 1))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    ce = nn.CrossEntropyLoss(reduction="none")

    best_f1 = -1.0
    best_path = os.path.join(OUTPUT_DIR, "source_supervised_best.pt")
    hist: List[Dict[str, Any]] = []

    for epoch in range(1, EPOCHS_SOURCE + 1):
        model.train()
        losses = []
        optim.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train source_supervised e{epoch}/{EPOCHS_SOURCE}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss_vec = ce(logits, labels)
            loss = (loss_vec * weights).mean()
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if GRAD_CLIP and GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()
                sched.step()
                optim.zero_grad()

            losses.append(float(loss.item() * GRAD_ACCUM_STEPS))

        # validate (macro-F1@0.5)
        val_p1 = predict_p1(model, val_loader, device, scaler_T=None)
        yv = np.array(src_val_labels, dtype=int)
        m = eval_acc_f1(yv, val_p1, tau=0.5)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc@0.5": m["acc"], "val_f1@0.5": m["f1"]})

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(safe_torch_load(best_path, device))
    save_json(hist, os.path.join(OUTPUT_DIR, "source_train_history.json"))
    return TrainState(best_path=best_path, best_metric=best_f1, history=hist)


def train_adapt_one_round(
    student: nn.Module,
    teacher: Optional[nn.Module],
    tokenizer,
    device: torch.device,
    # labeled training set: source + pseudo
    labeled_texts: List[str], labeled_labels: List[int], labeled_weights: List[float],
    # unlabeled target set for consistency
    tgt_unl_texts: List[str],
    round_idx: int,
    lambda_u: float,
    conf_thresh: float,
    scaler_T_for_pseudo: Optional[float],
) -> str:
    """
    Train one round. Returns path to best checkpoint by source-val macroF1@0.5 (sanity).
    """
    train_loader = make_loader(tokenizer, labeled_texts, labeled_labels, labeled_weights, shuffle=True)
    unl_loader = make_loader(tokenizer, tgt_unl_texts, labels=None, weights=None, shuffle=True)

    # We use source validation for sanity best checkpoint inside round
    # (selection for final is by target val accuracy later).
    # We'll load its texts/labels from closure variables in run().
    # We will pass them via globals set in run() to avoid huge signature.
    global _SRC_VAL_LOADER, _SRC_VAL_LABELS

    optim = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, EPOCHS_ADAPT_PER_ROUND * len(train_loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    ce = nn.CrossEntropyLoss(reduction="none")
    kl = nn.KLDivLoss(reduction="none")

    best_f1 = -1.0
    best_path = os.path.join(OUTPUT_DIR, f"adapt_round{round_idx}_best.pt")

    unl_iter = iter(unl_loader)

    for epoch in range(1, EPOCHS_ADAPT_PER_ROUND + 1):
        student.train()
        if teacher is not None:
            teacher.eval()

        optim.zero_grad()
        for step, batch_l in enumerate(tqdm(train_loader, desc=f"Train adapt_round{round_idx} e{epoch}/{EPOCHS_ADAPT_PER_ROUND}")):
            # labeled batch
            input_ids = batch_l["input_ids"].to(device)
            attention_mask = batch_l["attention_mask"].to(device)
            labels = batch_l["labels"].to(device)
            weights = batch_l["weights"].to(device)

            logits_l = student(input_ids=input_ids, attention_mask=attention_mask).logits
            loss_l = (ce(logits_l, labels) * weights).mean()

            loss = loss_l

            # unlabeled consistency batch
            if USE_CONSISTENCY and teacher is not None:
                try:
                    batch_u = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(unl_loader)
                    batch_u = next(unl_iter)

                ui = batch_u["input_ids"].to(device)
                um = batch_u["attention_mask"].to(device)

                # student probs
                logits_u_s = student(input_ids=ui, attention_mask=um).logits
                logp_s = torch.log_softmax(logits_u_s, dim=-1)

                with torch.no_grad():
                    logits_u_t = teacher(input_ids=ui, attention_mask=um).logits
                    # optional temperature for pseudo/teacher (monotonic; kept for stability)
                    if scaler_T_for_pseudo is not None:
                        logits_u_t = logits_u_t / float(max(scaler_T_for_pseudo, 1e-6))
                    p_t = torch.softmax(logits_u_t, dim=-1)  # (B,2)

                    conf = torch.max(p_t, dim=-1).values  # (B,)
                    mask = (conf >= conf_thresh).float()
                    # weight by confidence above threshold
                    w_u = torch.clamp((conf - conf_thresh) / max(1e-6, (1.0 - conf_thresh)), 0.0, 1.0) * mask

                # KL(p_t || p_s): sum_c p_t * (log p_t - log p_s)
                # kl_div in torch expects input log-probs, target probs
                # We'll compute per-sample:
                loss_u_full = kl(logp_s, p_t).sum(dim=-1)  # (B,)
                # weighted mean
                if w_u.sum().item() > 0:
                    loss_u = (loss_u_full * w_u).sum() / (w_u.sum() + 1e-6)
                else:
                    loss_u = torch.zeros([], device=device)

                loss = loss + (lambda_u * CONSISTENCY_WEIGHT_MAX) * loss_u

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if GRAD_CLIP and GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optim.step()
                sched.step()
                optim.zero_grad()

                # EMA update after optimizer step
                if teacher is not None:
                    ema_update(teacher, student, EMA_DECAY)

        # sanity validate on source val F1@0.5
        student.eval()
        val_p1 = predict_p1(student, _SRC_VAL_LOADER, device, scaler_T=None)
        m = eval_acc_f1(np.array(_SRC_VAL_LABELS, dtype=int), val_p1, tau=0.5)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(student.state_dict(), best_path)

    student.load_state_dict(safe_torch_load(best_path, device))
    # keep teacher in sync with student
    if teacher is not None:
        teacher.load_state_dict(student.state_dict())
    return best_path


# =========================
# Main run
# =========================
def run():
    set_seed(SEED)
    device = get_device()
    print(f"[Device] {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_train_path = os.path.join(base_dir, SOURCE_DIR, "source_train.csv")
    src_val_path = os.path.join(base_dir, SOURCE_DIR, "source_validation.csv")
    src_test_path = os.path.join(base_dir, SOURCE_DIR, "source_test.csv")

    tgt_unl_path = os.path.join(base_dir, TARGET_DIR, "train.csv")
    tgt_val_path = os.path.join(base_dir, TARGET_DIR, "val.csv")
    tgt_test_path = os.path.join(base_dir, TARGET_DIR, "test.csv")

    for p in [src_train_path, src_val_path, src_test_path, tgt_unl_path, tgt_val_path, tgt_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    print("[Data] Loading files...")
    src_train_df = load_table_auto(src_train_path, TEXT_COL, LABEL_COL)
    src_val_df = load_table_auto(src_val_path, TEXT_COL, LABEL_COL)
    src_test_df = load_table_auto(src_test_path, TEXT_COL, LABEL_COL)

    tgt_unl_df = load_table_auto(tgt_unl_path, TEXT_COL, None)  # ignore label
    tgt_val_df = load_table_auto(tgt_val_path, TEXT_COL, LABEL_COL)
    tgt_test_df = load_table_auto(tgt_test_path, TEXT_COL, LABEL_COL)

    src_train_texts = src_train_df[TEXT_COL].map(to_str).tolist()
    src_train_labels = src_train_df[LABEL_COL].astype(int).tolist()
    src_val_texts = src_val_df[TEXT_COL].map(to_str).tolist()
    src_val_labels = src_val_df[LABEL_COL].astype(int).tolist()
    src_test_texts = src_test_df[TEXT_COL].map(to_str).tolist()
    src_test_labels = src_test_df[LABEL_COL].astype(int).tolist()

    tgt_unl_texts = tgt_unl_df[TEXT_COL].map(to_str).tolist()
    tgt_val_texts = tgt_val_df[TEXT_COL].map(to_str).tolist()
    tgt_val_labels = tgt_val_df[LABEL_COL].astype(int).tolist()
    tgt_test_texts = tgt_test_df[TEXT_COL].map(to_str).tolist()
    tgt_test_labels = tgt_test_df[LABEL_COL].astype(int).tolist()

    print(f"[Data] Source: train={len(src_train_texts)} val={len(src_val_texts)} test={len(src_test_texts)}")
    print(f"[Data] Target: unlabeled(train)={len(tgt_unl_texts)} val={len(tgt_val_texts)} test={len(tgt_test_texts)}")

    pi_s1 = float(np.mean(src_train_labels))
    print(f"[Source] pi_s1={pi_s1:.4f}")

    ensure_dir(OUTPUT_DIR)
    save_json({
        "SEED": SEED, "MODEL_NAME": MODEL_NAME,
        "MAX_LEN": MAX_LEN, "BATCH_SIZE": BATCH_SIZE, "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY, "EPOCHS_SOURCE": EPOCHS_SOURCE,
        "ADAPT_ROUNDS": ADAPT_ROUNDS, "EPOCHS_ADAPT_PER_ROUND": EPOCHS_ADAPT_PER_ROUND,
        "PSEUDO_FRAC_SCHEDULE": PSEUDO_FRAC_SCHEDULE, "RAMP_SCHEDULE": RAMP_SCHEDULE,
        "PRIOR_MODE": PRIOR_MODE, "USE_EMA_TEACHER": USE_EMA_TEACHER, "EMA_DECAY": EMA_DECAY,
        "USE_CONSISTENCY": USE_CONSISTENCY, "CONSISTENCY_CONF_THRESH": CONSISTENCY_CONF_THRESH,
    }, os.path.join(OUTPUT_DIR, "config.json"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    student = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Stage0: supervised on source
    print("\n[Stage0] Supervised training on source...")
    state0 = train_supervised_source(student, tokenizer, device, src_train_texts, src_train_labels, src_val_texts, src_val_labels)
    print(f"[Stage0] best source-val macroF1@0.5 = {state0.best_metric:.4f}")

    # Load best source checkpoint
    student.load_state_dict(safe_torch_load(state0.best_path, device))
    student.to(device)

    # Prepare loaders for eval/sanity
    src_val_loader = make_loader(tokenizer, src_val_texts, src_val_labels, weights=None, shuffle=False)
    src_test_loader = make_loader(tokenizer, src_test_texts, src_test_labels, weights=None, shuffle=False)
    tgt_val_loader = make_loader(tokenizer, tgt_val_texts, tgt_val_labels, weights=None, shuffle=False)
    tgt_test_loader = make_loader(tokenizer, tgt_test_texts, tgt_test_labels, weights=None, shuffle=False)
    tgt_unl_loader_eval = make_loader(tokenizer, tgt_unl_texts, labels=None, weights=None, shuffle=False)

    # expose to train_adapt_one_round via globals
    global _SRC_VAL_LOADER, _SRC_VAL_LABELS
    _SRC_VAL_LOADER, _SRC_VAL_LABELS = src_val_loader, src_val_labels

    # Calibrate temperature on source val (for logging & pseudo/teacher confidence)
    print("\n[Calib] Temperature scaling on source validation...")
    scaler_src = TemperatureScaler().to(device)
    src_val_logits = predict_logits(student, src_val_loader, device).to(device)
    T_src = scaler_src.fit(src_val_logits, torch.tensor(src_val_labels, dtype=torch.long, device=device), max_iter=50)
    print(f"[Calib] Source Temperature T_src={T_src:.4f}")

    # EMA teacher
    teacher: Optional[nn.Module] = None
    if USE_EMA_TEACHER:
        teacher = clone_model(student).to(device)
        teacher.eval()
        print("[EMA] Enabled.")
    else:
        print("[EMA] Disabled.")

    # For BBSE logging: need source-val hard preds
    print("\n[Prep] Compute source-val predictions for BBSE...")
    src_val_p1 = predict_p1(student, src_val_loader, device, scaler_T=T_src)
    y_true_src_val = np.array(src_val_labels, dtype=int)
    y_pred_src_val = hard_pred(src_val_p1, tau=0.5)

    # baseline evaluation (tau=0.5)
    def eval_pack(tag: str, model: nn.Module, T_eval: Optional[float], tau_eval: float) -> Dict[str, Any]:
        src_p1 = predict_p1(model, src_test_loader, device, scaler_T=T_eval)
        val_p1 = predict_p1(model, tgt_val_loader, device, scaler_T=T_eval)
        test_p1 = predict_p1(model, tgt_test_loader, device, scaler_T=T_eval)
        ms = eval_acc_f1(np.array(src_test_labels, dtype=int), src_p1, tau=0.5)
        mv = eval_acc_f1(np.array(tgt_val_labels, dtype=int), val_p1, tau=tau_eval)
        mt = eval_acc_f1(np.array(tgt_test_labels, dtype=int), test_p1, tau=tau_eval)
        print(f"[Eval:{tag}] src_acc={ms['acc']:.4f} src_f1={ms['f1']:.4f} | "
              f"tgt_val_acc={mv['acc']:.4f} tgt_val_f1={mv['f1']:.4f} | "
              f"tgt_test_acc={mt['acc']:.4f} tgt_test_f1={mt['f1']:.4f} (tau={tau_eval:.3f})")
        return {"src": ms, "tgt_val": mv, "tgt_test": mt, "tau": float(tau_eval), "T": None if T_eval is None else float(T_eval)}

    before = eval_pack("before_adapt", student, T_eval=T_src, tau_eval=0.5)

    # selection bookkeeping
    best = {
        "round": 0,
        "val_acc": before["tgt_val"]["acc"],
        "tau_star": 0.5,
        "T_src": float(T_src),
        "state_path": state0.best_path,
    }
    history: Dict[str, Any] = {"before": before, "rounds": []}

    # pseudo pool accumulates indices -> (label, weight)
    pseudo_pool: Dict[int, Tuple[int, float]] = {}

    # =========================
    # UDA rounds
    # =========================
    for r in range(1, ADAPT_ROUNDS + 1):
        print("\n" + "=" * 30)
        print(f"[UDA] Round {r}/{ADAPT_ROUNDS}")
        print("=" * 30)

        # schedules
        pseudo_frac = PSEUDO_FRAC_SCHEDULE[min(r - 1, len(PSEUDO_FRAC_SCHEDULE) - 1)]
        ramp = RAMP_SCHEDULE[min(r - 1, len(RAMP_SCHEDULE) - 1)]
        lambda_u = ramp  # for consistency
        # pi_t
        pi_t = pi_s1  # fixed_source
        # (Optional) BBSE log on target unlabeled (using teacher/student at tau=0.5)
        # use teacher for prediction if available
        pred_model = teacher if teacher is not None else student
        p1_unl = predict_p1(pred_model, tgt_unl_loader_eval, device, scaler_T=T_src)
        pred_unl_05 = hard_pred(p1_unl, tau=0.5)
        pi_bbse = estimate_pi_bbse_log_only(y_true_src_val, y_pred_src_val, pred_unl_05, eps=1e-6)
        tgt_pred_pos = float(pred_unl_05.mean())
        print(f"[Diag] tgt_pred_pos@0.5={tgt_pred_pos:.4f}  p1(min/med/max)={p1_unl.min():.6g}/{np.median(p1_unl):.6g}/{p1_unl.max():.6g} mean={p1_unl.mean():.6g}")
        print(f"[Shift] (log) pi_bbse={pi_bbse:.4f}  pi_used={pi_t:.4f}  mode={PRIOR_MODE}")

        # pseudo selection on ALL target unlabeled by teacher (topk balanced)
        n_id = len(tgt_unl_texts)
        total_budget = int(round(pseudo_frac * n_id))
        idx_sel, y_pseudo, w_pseudo = select_topk_balanced(p1_unl, total_budget=total_budget, pi=pi_t, seed=SEED + 17 * r)

        # update pool (accumulate)
        for gi, yy, ww in zip(idx_sel.tolist(), y_pseudo.tolist(), w_pseudo.tolist()):
            ww_scaled = float(ww) * float(ramp)  # ramp weight up
            if gi in pseudo_pool:
                old_y, old_w = pseudo_pool[gi]
                # keep higher weight
                if ww_scaled > old_w:
                    pseudo_pool[gi] = (int(yy), ww_scaled)
            else:
                pseudo_pool[gi] = (int(yy), ww_scaled)

        pos_cnt = sum(1 for _, (yy, _) in pseudo_pool.items() if yy == 1)
        neg_cnt = len(pseudo_pool) - pos_cnt
        print(f"[Pseudo] round={r} pseudo_frac={pseudo_frac:.2f} ramp={ramp:.2f} selected={len(idx_sel)} "
              f"pool={len(pseudo_pool)} pos={pos_cnt} neg={neg_cnt} lam_u={lambda_u:.2f}")

        # build labeled training set: source + pseudo pool
        pseudo_texts = [tgt_unl_texts[i] for i in pseudo_pool.keys()]
        pseudo_labels = [pseudo_pool[i][0] for i in pseudo_pool.keys()]
        pseudo_weights = [pseudo_pool[i][1] for i in pseudo_pool.keys()]

        labeled_texts = src_train_texts + pseudo_texts
        labeled_labels = src_train_labels + pseudo_labels
        labeled_weights = [1.0] * len(src_train_texts) + pseudo_weights

        # train one round with optional consistency
        best_path = train_adapt_one_round(
            student=student,
            teacher=teacher,
            tokenizer=tokenizer,
            device=device,
            labeled_texts=labeled_texts,
            labeled_labels=labeled_labels,
            labeled_weights=labeled_weights,
            tgt_unl_texts=tgt_unl_texts,
            round_idx=r,
            lambda_u=lambda_u if USE_CONSISTENCY else 0.0,
            conf_thresh=CONSISTENCY_CONF_THRESH,
            scaler_T_for_pseudo=T_src,
        )
        print(f"[Adapt] round{r} best checkpoint: {best_path}")

        # re-fit source temperature after training (optional but helps stable pseudo confidence)
        print("[Calib] Recalibrate T_src on source val...")
        src_val_logits = predict_logits(student, src_val_loader, device).to(device)
        T_src = scaler_src.fit(src_val_logits, torch.tensor(src_val_labels, dtype=torch.long, device=device), max_iter=50)
        print(f"[Calib] Source Temperature T_src={T_src:.4f}")

        # Evaluate on target val/test with tau*
        # Use the same T_src for scoring (monotonic -> tau* tuning handles it).
        tgt_val_p1 = predict_p1(student, tgt_val_loader, device, scaler_T=T_src)
        tgt_test_p1 = predict_p1(student, tgt_test_loader, device, scaler_T=T_src)
        yv = np.array(tgt_val_labels, dtype=int)
        yt = np.array(tgt_test_labels, dtype=int)

        if TUNE_THRESHOLD_ON_TARGET_VAL:
            tau_star, bestm = tune_tau_for_accuracy(yv, tgt_val_p1)
        else:
            tau_star, bestm = 0.5, eval_acc_f1(yv, tgt_val_p1, 0.5)

        mv05 = eval_acc_f1(yv, tgt_val_p1, 0.5)
        mt05 = eval_acc_f1(yt, tgt_test_p1, 0.5)
        mvt = eval_acc_f1(yv, tgt_val_p1, tau_star)
        mtt = eval_acc_f1(yt, tgt_test_p1, tau_star)

        print(f"[Eval:round{r}] tgt_val_acc@0.5={mv05['acc']:.4f}  tgt_val_acc@tau*={mvt['acc']:.4f}  "
              f"tgt_test_acc@0.5={mt05['acc']:.4f}  tgt_test_acc@tau*={mtt['acc']:.4f}  (tau*={tau_star:.3f})")

        round_rec = {
            "round": r,
            "T_src": float(T_src),
            "pi_bbse_log": float(pi_bbse),
            "tgt_pred_pos@0.5": float(tgt_pred_pos),
            "pseudo_pool_size": int(len(pseudo_pool)),
            "pseudo_pool_pos": int(pos_cnt),
            "pseudo_pool_neg": int(neg_cnt),
            "val_acc@0.5": float(mv05["acc"]),
            "val_acc@tau*": float(mvt["acc"]),
            "val_f1@tau*": float(mvt["f1"]),
            "test_acc@0.5": float(mt05["acc"]),
            "test_acc@tau*": float(mtt["acc"]),
            "tau_star": float(tau_star),
            "best_path": best_path,
        }
        history["rounds"].append(round_rec)
        save_json(history, os.path.join(OUTPUT_DIR, "history.json"))

        # select best by target-val accuracy (Mode A)
        if USE_TARGET_VAL_FOR_SELECTION and (mvt["acc"] > best["val_acc"] + 1e-12):
            best = {
                "round": r,
                "val_acc": float(mvt["acc"]),
                "tau_star": float(tau_star),
                "T_src": float(T_src),
                "state_path": best_path,
            }
            print(f"[Select] New best: round={r} val_acc={best['val_acc']:.4f} tau*={best['tau_star']:.3f} T_src={best['T_src']:.4f}")

    # =========================
    # Final: load best checkpoint + restore its T + re-tune tau*
    # =========================
    print("\n[Final] Loading best checkpoint by target-val accuracy...")
    student.load_state_dict(safe_torch_load(best["state_path"], device))
    student.to(device)

    # restore best T_src
    T_final = float(best["T_src"])
    print(f"[Final] best_round={best['round']}  best_val_acc={best['val_acc']:.4f}  restored_T_src={T_final:.4f}")

    # final tuning tau* on target val (with restored T)
    tgt_val_p1 = predict_p1(student, tgt_val_loader, device, scaler_T=T_final)
    tgt_test_p1 = predict_p1(student, tgt_test_loader, device, scaler_T=T_final)
    yv = np.array(tgt_val_labels, dtype=int)
    yt = np.array(tgt_test_labels, dtype=int)

    if TUNE_THRESHOLD_ON_TARGET_VAL:
        tau_star, bestm = tune_tau_for_accuracy(yv, tgt_val_p1)
    else:
        tau_star = 0.5

    m05 = eval_acc_f1(yt, tgt_test_p1, 0.5)
    mt = eval_acc_f1(yt, tgt_test_p1, tau_star)

    print(f"[Final] tgt_test_acc@0.5={m05['acc']:.4f}  tgt_test_acc@tau*={mt['acc']:.4f}  "
          f"tgt_test_F1@0.5={m05['f1']:.4f}  tgt_test_F1@tau*={mt['f1']:.4f} (tau*={tau_star:.3f})")

    print_confusion(yt, tgt_test_p1, 0.5, name="test@0.5")
    print_confusion(yt, tgt_test_p1, tau_star, name="test@tau*")

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(student.state_dict(), final_path)
    print(f"[Final] saved to {final_path}")
    print(f"[Logs] saved to {os.path.join(OUTPUT_DIR, 'history.json')}")


# Globals used by training function
_SRC_VAL_LOADER = None
_SRC_VAL_LABELS = None


if __name__ == "__main__":
    run()
