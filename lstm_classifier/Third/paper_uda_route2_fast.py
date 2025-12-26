#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route-2 (Method paper) + Fast runner:
Binary UDA under compound shift with open-set noise and decision-threshold drift.

Key idea (ours_route2):
  - Dual-head model: (1) y-head for binary classification, (2) ood-head for knownness (known vs unknown).
  - OOD-aware pseudo-labeling: select pseudo labels ONLY from high-knownness region, to avoid unknown pollution.
  - Unknown mining: pick low-knownness samples from target unlabeled + sample from politics.csv as unknown anchors.
  - Unknown entropy regularization: discourage over-confident binary predictions on unknown samples.
  - Optional teacher EMA + consistency regularization.

Protocols:
  - Protocol A (calibrated): use small labeled target val for selecting tau and best round (strong results fast).
  - Protocol B (fully unlabeled tau): estimate target prior (shrink-BBSE on knownness-filtered target) and set tau by quantile.

Outputs:
  runs/route2_fast/
    per_run/<run_id>/result.json
    all_results_partial.csv
    all_results_<suite>.csv
    summary_<suite>.csv
"""

import os
import re
import csv
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
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

from transformers import AutoTokenizer, AutoModel


# ============================================================
# Default config (FAST by default)
# ============================================================

DEFAULT_SEEDS = [13]

# Fast model (good enough to see trend quickly)
FAST_MODEL_NAME = "distilbert-base-uncased"
FULL_MODEL_NAME = "bert-base-uncased"


# ============================================================
# Utils
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

def norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


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
    raise ValueError(f"Unsupported extension for {path}")


# ============================================================
# Metrics
# ============================================================

def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

def eval_id_metrics(y: np.ndarray, p1: np.ndarray, tau: float) -> Dict[str, float]:
    mask = np.isin(y, [0, 1])
    yk = y[mask]
    pk = p1[mask]
    if len(yk) == 0:
        return {"id_n": 0, "id_acc": float("nan"), "id_bal_acc": float("nan"), "id_macro_f1": float("nan")}
    pred = (pk >= tau).astype(int)
    return {
        "id_n": int(len(yk)),
        "id_acc": float(accuracy_score(yk, pred)),
        "id_bal_acc": float(balanced_accuracy_score(yk, pred)),
        "id_macro_f1": float(f1_score(yk, pred, average="macro")),
    }

def confusion_id(y: np.ndarray, p1: np.ndarray, tau: float) -> Dict[str, int]:
    mask = np.isin(y, [0, 1])
    yk = y[mask]
    pk = p1[mask]
    if len(yk) == 0:
        return {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    pred = (pk >= tau).astype(int)
    cm = confusion_matrix(yk, pred, labels=[0,1])
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

def score_maxsoft(p1: np.ndarray) -> np.ndarray:
    return np.maximum(p1, 1.0 - p1)

def ood_metrics(y: np.ndarray, score_known: np.ndarray) -> Dict[str, float]:
    """
    y: -1 for unknown, 0/1 for known.
    score_known: higher => more known-like.
    """
    known = (y != -1).astype(int)  # known=1, unknown=0
    if len(np.unique(known)) < 2:
        return {"ood_auroc": float("nan"), "ood_fpr95": float("nan")}
    try:
        auroc = float(roc_auc_score(known, score_known))
    except Exception:
        auroc = float("nan")

    s_known = score_known[known == 1]
    s_unk = score_known[known == 0]
    if len(s_known) == 0 or len(s_unk) == 0:
        return {"ood_auroc": auroc, "ood_fpr95": float("nan")}

    thr = float(np.quantile(s_known, 0.05))  # keep top 95% known
    fpr = float((s_unk >= thr).mean())
    tpr = float((s_known >= thr).mean())
    return {"ood_auroc": auroc, "ood_fpr95": fpr, "ood_thr95": thr, "ood_tpr": tpr}

def best_tau_for_macro_f1(y: np.ndarray, p1: np.ndarray, grid: int = 400) -> Tuple[float, float]:
    mask = np.isin(y, [0,1])
    yk = y[mask]
    pk = p1[mask]
    if len(yk) == 0:
        return 0.5, float("nan")
    qs = np.linspace(0.01, 0.99, grid)
    taus = np.quantile(pk, qs)
    best_tau, best_f1 = 0.5, -1.0
    for tau in taus:
        pred = (pk >= tau).astype(int)
        f1 = f1_score(yk, pred, average="macro")
        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau = float(tau)
    return best_tau, float(best_f1)

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
# Data construction
# ============================================================

@dataclass
class DataBundle:
    src_train_texts: List[str]
    src_train_labels: List[int]
    src_val_texts: List[str]
    src_val_labels: List[int]
    src_test_texts: List[str]
    src_test_labels: List[int]
    tgt_train_df: pd.DataFrame
    tgt_train_has_labels: bool
    tgt_val_df: pd.DataFrame
    tgt_test_df: pd.DataFrame
    unk_df: Optional[pd.DataFrame]

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
    source_dir = os.path.join(base_dir, "sourcedata")
    src_train = load_table_auto(os.path.join(source_dir, "source_train.csv"))
    src_val   = load_table_auto(os.path.join(source_dir, "source_validation.csv"))
    src_test  = load_table_auto(os.path.join(source_dir, "source_test.csv"))

    target_dir = os.path.join(base_dir, "targetdata_clean")
    if not os.path.exists(target_dir):
        target_dir = os.path.join(base_dir, "targetdata")

    tgt_train_has_labels = True
    try:
        tgt_train = load_table_auto(os.path.join(target_dir, "train.csv"), label_col="label")
    except Exception:
        tgt_train_has_labels = False
        tgt_train = load_table_auto(os.path.join(target_dir, "train.csv"), label_col=None)
        tgt_train["label"] = 0  # placeholder

    tgt_val  = load_table_auto(os.path.join(target_dir, "val.csv"),  label_col="label")
    tgt_test = load_table_auto(os.path.join(target_dir, "test.csv"), label_col="label")

    for df in [src_train, src_val, src_test, tgt_train, tgt_val, tgt_test]:
        df["text"] = df["text"].map(norm_text)
        df.dropna(subset=["text"], inplace=True)
        df = df[df["text"].astype(str).str.len() > 0]

    pol_path = find_politics_path(base_dir)
    unk_df = None
    if pol_path is not None:
        try:
            unk_df = load_table_auto(pol_path, label_col="label")
        except Exception:
            unk_df = load_table_auto(pol_path, label_col=None)
            unk_df["label"] = -1
        unk_df["text"] = unk_df["text"].map(norm_text)
        unk_df["label"] = -1

    return DataBundle(
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

def resample_to_prior(df: pd.DataFrame, pi: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df0 = df[df["label"].astype(int) == 0]
    df1 = df[df["label"].astype(int) == 1]
    n1 = int(round(pi * n))
    n0 = n - n1
    if len(df0) == 0 or len(df1) == 0:
        return df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    s0 = df0.sample(n=n0, replace=True, random_state=seed)
    s1 = df1.sample(n=n1, replace=True, random_state=seed + 1)
    out = pd.concat([s0, s1], axis=0).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return out

def mix_unknown(df_id: pd.DataFrame, df_unk: pd.DataFrame, alpha: float, seed: int) -> pd.DataFrame:
    n = len(df_id)
    k_unk = int(round(alpha * n))
    k_id = n - k_unk
    df_id_s = df_id.sample(n=k_id, replace=False if k_id <= n else True, random_state=seed).copy()
    df_unk_s = df_unk.sample(n=k_unk, replace=True, random_state=seed + 7).copy()
    df_unk_s["label"] = -1
    out = pd.concat([df_id_s, df_unk_s], axis=0).sample(frac=1.0, random_state=seed + 9).reset_index(drop=True)
    return out


# ============================================================
# Dataset / Loader (supports y-label, ood-label, weights)
# ============================================================

IGNORE_Y = -100

class TextMultiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        y_labels: Optional[List[int]],
        y_weights: Optional[List[float]],
        ood_labels: Optional[List[int]],
        unk_flags: Optional[List[int]],
    ):
        self.texts = texts
        self.y_labels = y_labels
        self.y_weights = y_weights
        self.ood_labels = ood_labels
        self.unk_flags = unk_flags

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = {"text": self.texts[idx]}
        if self.y_labels is not None:
            d["y_labels"] = int(self.y_labels[idx])
        if self.y_weights is not None:
            d["y_weights"] = float(self.y_weights[idx])
        if self.ood_labels is not None:
            d["ood_labels"] = int(self.ood_labels[idx])
        if self.unk_flags is not None:
            d["unk_flags"] = int(self.unk_flags[idx])
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
        # defaults
        y = [b.get("y_labels", IGNORE_Y) for b in batch]
        w = [b.get("y_weights", 1.0) for b in batch]
        o = [b.get("ood_labels", 1) for b in batch]  # default known
        u = [b.get("unk_flags", 0) for b in batch]

        enc["y_labels"] = torch.tensor(y, dtype=torch.long)
        enc["y_weights"] = torch.tensor(w, dtype=torch.float)
        enc["ood_labels"] = torch.tensor(o, dtype=torch.float)  # BCE target
        enc["unk_flags"] = torch.tensor(u, dtype=torch.float)
        return enc
    return collate

def make_loader(tokenizer, texts, y_labels, y_weights, ood_labels, unk_flags, batch_size, max_len, shuffle):
    ds = TextMultiDataset(texts, y_labels, y_weights, ood_labels, unk_flags)
    collate_fn = make_collate_fn(tokenizer, max_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)


# ============================================================
# Dual-head model
# ============================================================

class DualHeadTransformer(nn.Module):
    def __init__(self, base_name: str, hidden_dropout: float = 0.1):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_name)
        hidden_size = self.base.config.hidden_size
        self.drop = nn.Dropout(hidden_dropout)
        self.y_head = nn.Linear(hidden_size, 2)
        self.ood_head = nn.Linear(hidden_size, 1)

    def pooled(self, outputs) -> torch.Tensor:
        # BERT has pooler_output, DistilBERT doesn't -> use CLS token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        h = self.drop(self.pooled(out))
        logits_y = self.y_head(h)           # [B, 2]
        logits_ood = self.ood_head(h).squeeze(-1)  # [B]
        return logits_y, logits_ood


@torch.inference_mode()
def predict_scores(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y = []
    all_ood = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits_y, logits_ood = model(input_ids=input_ids, attention_mask=attn)
        all_y.append(logits_y.detach().cpu().numpy())
        all_ood.append(logits_ood.detach().cpu().numpy())
    ly = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,2), dtype=np.float32)
    lo = np.concatenate(all_ood, axis=0) if all_ood else np.zeros((0,), dtype=np.float32)
    return ly, lo

def logits_to_p1_and_known(logits_y: np.ndarray, logits_ood: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = softmax_np(logits_y)
    p1 = p[:, 1]
    maxsoft = np.maximum(p1, 1.0 - p1)
    knownness = 1.0 / (1.0 + np.exp(-logits_ood))  # sigmoid
    return p1, maxsoft, knownness


# ============================================================
# Pseudo-labeling helpers
# ============================================================

def topk_balanced_indices(p1: np.ndarray, k_pos: int, k_neg: int) -> Tuple[np.ndarray, np.ndarray]:
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
    p = np.clip(p1, 1e-6, 1 - 1e-6)
    w = np.where(y == 1, p, 1.0 - p)
    w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    return w.astype(np.float32)

def build_pseudo_pool(
    p1: np.ndarray,
    pi_used: float,
    frac: float,
    seed: int,
    id_mask: Optional[np.ndarray],
    strategy: str,
) -> Dict[int, Tuple[int, float]]:
    rng = np.random.RandomState(seed)
    n = len(p1)
    if id_mask is None:
        id_mask = np.ones(n, dtype=bool)
    id_idx = np.where(id_mask)[0]
    p_id = p1[id_mask]
    n_id = len(p_id)

    total = int(round(frac * n))
    total = min(total, n_id)
    pi_used = float(np.clip(pi_used, 1e-6, 1-1e-6))
    k_pos = int(round(total * pi_used))
    k_neg = total - k_pos

    if strategy == "topk_balanced":
        sel_local, sel_y = topk_balanced_indices(p_id, k_pos=k_pos, k_neg=k_neg)
    elif strategy == "naive_conf":
        score = score_maxsoft(p_id)
        order = np.argsort(-score)
        sel_local = order[:total]
        sel_y = (p_id[sel_local] >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    sel_global = id_idx[sel_local]
    w = confidence_weights(p1[sel_global], sel_y)

    perm = rng.permutation(len(sel_global))
    sel_global = sel_global[perm]
    sel_y = sel_y[perm]
    w = w[perm]

    return {int(i): (int(y), float(wi)) for i, y, wi in zip(sel_global, sel_y, w)}


# ============================================================
# Training
# ============================================================

def make_optimizer(model: nn.Module, lr: float, wd: float, train_ood_head: bool) -> torch.optim.Optimizer:
    # Exclude ood head if not used -> fairer baseline + slightly faster
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (not train_ood_head) and n.startswith("ood_head"):
            continue
        params.append(p)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=(1.0 - decay))

def kl_div_teacher_student(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    p_t = torch.softmax(teacher_logits, dim=-1)
    logp_s = torch.log_softmax(student_logits, dim=-1)
    return -(p_t * logp_s).sum(dim=-1)  # per-sample CE(teacher, student)

def train_one_stage(
    student: DualHeadTransformer,
    teacher: Optional[DualHeadTransformer],
    tokenizer,
    device: torch.device,
    train_loader,
    unl_loader,
    *,
    epochs: int,
    lr: float,
    wd: float,
    grad_accum: int,
    use_teacher: bool,
    use_consistency: bool,
    use_ood_head: bool,
    lambda_ood: float,
    lambda_unk: float,
    lambda_u: float,
    known_conf_thresh: float,
    y_conf_thresh: float,
    ema_decay: float,
) -> None:
    student.to(device)
    if teacher is not None:
        teacher.to(device)
        teacher.eval()

    opt = make_optimizer(student, lr=lr, wd=wd, train_ood_head=use_ood_head)
    ce = nn.CrossEntropyLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")

    unl_iter = iter(unl_loader)

    student.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train stage e{ep}/{epochs}")):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_labels = batch["y_labels"].to(device)       # [B]
            y_weights = batch["y_weights"].to(device)     # [B]
            ood_labels = batch["ood_labels"].to(device)   # [B] float
            unk_flags = batch["unk_flags"].to(device)     # [B] float

            logits_y, logits_ood = student(input_ids=input_ids, attention_mask=attn)

            # y-loss only for labeled samples
            labeled_mask = (y_labels != IGNORE_Y).float()
            if labeled_mask.sum() > 0:
                loss_y = ce(logits_y, y_labels)
                loss_y = (loss_y * y_weights * labeled_mask).sum() / (labeled_mask.sum() + 1e-9)
            else:
                loss_y = torch.zeros([], device=device)

            # ood-loss (known=1, unknown=0)
            if use_ood_head:
                loss_ood = bce(logits_ood, ood_labels)
                loss_ood = loss_ood.mean()
            else:
                loss_ood = torch.zeros([], device=device)

            # unknown entropy regularization (discourage overconfidence on unknown)
            if use_ood_head and lambda_unk > 0 and unk_flags.sum() > 0:
                p = torch.softmax(logits_y, dim=-1)
                ent_pen = (p * torch.log(p + 1e-9)).sum(dim=-1)  # minimize -> increase entropy
                loss_unk = (ent_pen * unk_flags).sum() / (unk_flags.sum() + 1e-9)
            else:
                loss_unk = torch.zeros([], device=device)

            loss = loss_y + lambda_ood * loss_ood + lambda_unk * loss_unk

            # consistency on unlabeled target
            if use_consistency and use_teacher and (teacher is not None) and lambda_u > 0:
                try:
                    u_batch = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(unl_loader)
                    u_batch = next(unl_iter)

                u_ids = u_batch["input_ids"].to(device)
                u_attn = u_batch["attention_mask"].to(device)

                with torch.no_grad():
                    t_y, t_ood = teacher(input_ids=u_ids, attention_mask=u_attn)
                    t_p = torch.softmax(t_y, dim=-1)
                    t_conf = torch.max(t_p, dim=-1).values
                    t_known = torch.sigmoid(t_ood)

                    mask = ((t_conf >= y_conf_thresh) & (t_known >= known_conf_thresh)).float()

                s_y, s_ood = student(input_ids=u_ids, attention_mask=u_attn)

                if mask.sum() > 0:
                    kl = kl_div_teacher_student(t_y.detach(), s_y)  # [B]
                    loss_u_y = (kl * mask).sum() / (mask.sum() + 1e-9)
                    loss = loss + lambda_u * loss_u_y

                    if use_ood_head:
                        # match knownness logits by MSE (simple)
                        loss_u_ood = ((torch.sigmoid(s_ood) - torch.sigmoid(t_ood.detach())) ** 2)
                        loss_u_ood = (loss_u_ood * mask).sum() / (mask.sum() + 1e-9)
                        loss = loss + 0.5 * lambda_u * loss_u_ood

            loss = loss / max(1, grad_accum)
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if use_teacher and teacher is not None:
                    update_ema(teacher, student, decay=ema_decay)


# ============================================================
# Methods & Suites
# ============================================================

@dataclass
class MethodConfig:
    name: str
    pseudo_strategy: str        # "naive_conf" / "topk_balanced" / "none"
    use_teacher: bool
    use_consistency: bool
    use_ood_head: bool
    use_unknown_mining: bool
    use_bbse_pi: bool

def get_methods(route2: bool = True) -> List[MethodConfig]:
    ms = [
        MethodConfig("source_only",        "none",         False, False, False, False, False),
        MethodConfig("naive_self_train",   "naive_conf",   False, False, False, False, False),
        MethodConfig("topk_student",       "topk_balanced",False, False, False, False, False),
        MethodConfig("topk_teacher",       "topk_balanced",True,  False, False, False, False),
        MethodConfig("full",               "topk_balanced",True,  True,  False, False, False),
        MethodConfig("bbse_prior",         "topk_balanced",True,  False, False, False, True),
    ]
    if route2:
        ms += [
            MethodConfig("ours_route2",          "topk_balanced", True, True,  True,  True,  False),
            MethodConfig("ours_route2_no_ood",   "topk_balanced", True, True,  False, False, False),
        ]
    return ms


# ============================================================
# Runner
# ============================================================

@dataclass
class RunConfig:
    fast: bool
    mode: str              # "quick" or "paper"
    protocol: str          # "A" or "B" (A uses target val labels for selection/tau)
    out_dir: str
    model_name: str

    max_len: int
    train_batch: int
    pred_batch: int
    grad_accum: int
    lr: float
    wd: float

    epochs_source: int
    adapt_rounds: int
    adapt_epochs: int

    # schedules
    pseudo_fracs: List[float]
    lambda_u: List[float]

    # ood-aware knobs (ours_route2)
    pseudo_known_thresh: float
    pseudo_y_conf_thresh: float
    unk_mine_frac: float
    unk_anchor_max: int
    lambda_ood: float
    lambda_unk: float
    consistency_known_thresh: float
    consistency_y_conf_thresh: float
    ema_decay: float

    # unsupervised threshold (Protocol B)
    bbse_shrink: float

def make_run_config(args) -> RunConfig:
    fast = bool(args.fast)
    model_name = FAST_MODEL_NAME if fast else FULL_MODEL_NAME

    if args.mode == "paper":
        # fuller grids -> you can enlarge later
        pseudo_fracs = [0.20, 0.25, 0.30, 0.35]
        lambda_u = [0.25, 0.50, 0.75, 1.00]
        adapt_rounds = 4
    else:
        # quick
        pseudo_fracs = [0.25, 0.35]
        lambda_u = [0.50, 1.00]
        adapt_rounds = 2

    # speed knobs
    max_len = 256 if fast else 384
    train_batch = 16 if fast else 8
    pred_batch = 64 if fast else 32
    grad_accum = 1 if fast else 2
    epochs_source = 1 if fast else 3
    adapt_epochs = 1

    return RunConfig(
        fast=fast,
        mode=args.mode,
        protocol=args.protocol,
        out_dir=args.out_dir,
        model_name=model_name,
        max_len=max_len,
        train_batch=train_batch,
        pred_batch=pred_batch,
        grad_accum=grad_accum,
        lr=2e-5,
        wd=0.01,
        epochs_source=epochs_source,
        adapt_rounds=adapt_rounds,
        adapt_epochs=adapt_epochs,
        pseudo_fracs=pseudo_fracs,
        lambda_u=lambda_u,
        pseudo_known_thresh=0.70,
        pseudo_y_conf_thresh=0.70,
        unk_mine_frac=0.10,
        unk_anchor_max=2000,
        lambda_ood=0.50,
        lambda_unk=0.10,
        consistency_known_thresh=0.70,
        consistency_y_conf_thresh=0.55,
        ema_decay=0.999,
        bbse_shrink=0.50,
    )

def build_conditions(mode: str, suite: str) -> List[Dict[str, Any]]:
    # alpha_eval is where you evaluate open-set; keep 0.2 as your earlier setting.
    if mode == "paper":
        openset_alpha_grid = [0.0, 0.1, 0.2, 0.3]
        label_pi_grid = [0.2, 0.5, 0.8]
        combined_alpha_grid = [0.2]
        combined_pi_grid = [0.2, 0.5, 0.8]
    else:
        openset_alpha_grid = [0.0, 0.2]
        label_pi_grid = [0.2, 0.8]
        combined_alpha_grid = [0.2]
        combined_pi_grid = [0.2, 0.5]

    if suite == "base":
        return [{"tag": "base", "alpha_train": 0.0, "alpha_eval": 0.0, "pi_target": None}]
    if suite == "open_set":
        return [{"tag": f"alpha{a:.2f}", "alpha_train": float(a), "alpha_eval": 0.2, "pi_target": None} for a in openset_alpha_grid]
    if suite == "label_shift":
        return [{"tag": f"pi{pi:.2f}", "alpha_train": 0.0, "alpha_eval": 0.0, "pi_target": float(pi)} for pi in label_pi_grid]
    if suite == "combined":
        out = []
        for a in combined_alpha_grid:
            for pi in combined_pi_grid:
                out.append({"tag": f"alpha{a:.2f}_pi{pi:.2f}", "alpha_train": float(a), "alpha_eval": 0.2, "pi_target": float(pi)})
        return out
    raise ValueError(f"Unknown suite: {suite}")

def evaluate(
    model: DualHeadTransformer,
    tokenizer,
    device: torch.device,
    texts: List[str],
    labels: List[int],
    *,
    max_len: int,
    pred_batch: int,
    tau: float,
    ood_score_mode: str,
) -> Dict[str, Any]:
    loader = make_loader(tokenizer, texts, None, None, None, None, batch_size=pred_batch, max_len=max_len, shuffle=False)
    ly, lo = predict_scores(model, loader, device)
    p1, maxsoft, knownness = logits_to_p1_and_known(ly, lo)

    y = np.array(labels, dtype=int)

    # ID metrics
    mid = eval_id_metrics(y, p1, tau=tau)
    cm = confusion_id(y, p1, tau=tau)

    # OOD score
    if ood_score_mode == "knownness":
        score = knownness
    else:
        score = maxsoft

    out = {
        "tau": float(tau),
        "id": mid,
        "cm": cm,
    }
    if np.any(y == -1):
        out["ood"] = ood_metrics(y, score_known=score)
    return out

def flatten_row(res: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    out["suite"] = res["suite"]
    out["seed"] = res["seed"]
    out["method"] = res["method"]
    cond = res["condition"]
    out["cond_tag"] = cond.get("tag")
    out["cond_alpha_train"] = cond.get("alpha_train")
    out["cond_alpha_eval"] = cond.get("alpha_eval")
    out["cond_pi_target"] = cond.get("pi_target")

    out["selected_round"] = res.get("selected_round", 0)
    out["tau_star"] = res.get("tau_star", 0.5)
    out["select_metric_name"] = res.get("select_metric_name")
    out["select_metric_value"] = res.get("select_metric_value")

    # tau metrics
    m = res["final_tau"]
    out["tgt_test_acc_tau"] = m["tgt_test"]["id"]["id_acc"]
    out["tgt_test_bal_acc_tau"] = m["tgt_test"]["id"]["id_bal_acc"]
    out["tgt_test_macro_f1_tau"] = m["tgt_test"]["id"]["id_macro_f1"]
    out["tgt_val_acc_tau"] = m["tgt_val"]["id"]["id_acc"]
    out["tgt_val_macro_f1_tau"] = m["tgt_val"]["id"]["id_macro_f1"]
    out["cm_TN_tau"] = m["tgt_test"]["cm"]["TN"]
    out["cm_FP_tau"] = m["tgt_test"]["cm"]["FP"]
    out["cm_FN_tau"] = m["tgt_test"]["cm"]["FN"]
    out["cm_TP_tau"] = m["tgt_test"]["cm"]["TP"]

    # fixed 0.5 metrics
    m05 = res["final_05"]
    out["tgt_test_acc_05"] = m05["tgt_test"]["id"]["id_acc"]
    out["tgt_test_macro_f1_05"] = m05["tgt_test"]["id"]["id_macro_f1"]

    # OOD metrics if any
    if "ood" in m["tgt_test"]:
        out["ood_test_auroc"] = m["tgt_test"]["ood"]["ood_auroc"]
        out["ood_test_fpr95"] = m["tgt_test"]["ood"]["ood_fpr95"]
    else:
        out["ood_test_auroc"] = None
        out["ood_test_fpr95"] = None

    return out

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

def run_one(
    bundle: DataBundle,
    rcfg: RunConfig,
    suite: str,
    cond: Dict[str, Any],
    method: MethodConfig,
    seed: int,
    out_root: str,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device()

    # prepare condition-specific target splits
    tgt_train_df = bundle.tgt_train_df.copy()
    tgt_val_df = bundle.tgt_val_df.copy()
    tgt_test_df = bundle.tgt_test_df.copy()

    if cond.get("pi_target") is not None:
        pi = float(cond["pi_target"])
        tgt_train_df = resample_to_prior(tgt_train_df, pi=pi, n=len(tgt_train_df), seed=seed + 101)
        tgt_val_df   = resample_to_prior(tgt_val_df,   pi=pi, n=len(tgt_val_df),   seed=seed + 202)
        tgt_test_df  = resample_to_prior(tgt_test_df,  pi=pi, n=len(tgt_test_df),  seed=seed + 303)

    alpha_train = float(cond.get("alpha_train", 0.0))
    alpha_eval = float(cond.get("alpha_eval", 0.0))

    if (alpha_train > 0 or alpha_eval > 0) and bundle.unk_df is None:
        raise RuntimeError("Open-set requested but politics.csv not found.")

    if alpha_train > 0:
        tgt_train_mix = mix_unknown(tgt_train_df, bundle.unk_df, alpha=alpha_train, seed=seed + 11)
    else:
        tgt_train_mix = tgt_train_df.copy()

    if alpha_eval > 0:
        tgt_val_mix = mix_unknown(tgt_val_df, bundle.unk_df, alpha=alpha_eval, seed=seed + 22)
        tgt_test_mix = mix_unknown(tgt_test_df, bundle.unk_df, alpha=alpha_eval, seed=seed + 33)
    else:
        tgt_val_mix = tgt_val_df.copy()
        tgt_test_mix = tgt_test_df.copy()

    # text lists
    src_train_texts = [norm_text(x) for x in bundle.src_train_texts]
    src_train_labels = list(map(int, bundle.src_train_labels))
    src_val_texts = [norm_text(x) for x in bundle.src_val_texts]
    src_val_labels = list(map(int, bundle.src_val_labels))
    src_test_texts = [norm_text(x) for x in bundle.src_test_texts]
    src_test_labels = list(map(int, bundle.src_test_labels))

    tgt_unl_texts = tgt_train_mix["text"].map(norm_text).tolist()
    tgt_val_texts = tgt_val_mix["text"].map(norm_text).tolist()
    tgt_val_labels = tgt_val_mix["label"].astype(int).tolist()
    tgt_test_texts = tgt_test_mix["text"].map(norm_text).tolist()
    tgt_test_labels = tgt_test_mix["label"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(rcfg.model_name, use_fast=True)

    # caching for source stage (per seed)
    cache_dir = os.path.join(out_root, "cache_source", f"seed{seed}", rcfg.model_name.replace("/", "_"))
    ensure_dir(cache_dir)
    ckpt_path = os.path.join(cache_dir, "source_best.pt")

    student = DualHeadTransformer(rcfg.model_name)
    teacher = None
    if method.use_teacher:
        teacher = DualHeadTransformer(rcfg.model_name)
        teacher.load_state_dict(student.state_dict())

    # stage0 source training (once per seed)
    if os.path.exists(ckpt_path):
        student.load_state_dict(safe_torch_load(ckpt_path, map_location=device))
        if teacher is not None:
            teacher.load_state_dict(student.state_dict())
    else:
        # build source + (optional) unknown batches for ood head pretrain
        # source labeled: known=1, unk_flag=0
        src_y = src_train_labels
        src_w = [1.0] * len(src_train_texts)
        src_ood = [1] * len(src_train_texts)
        src_unk = [0] * len(src_train_texts)

        # optional unknown anchors from politics
        if bundle.unk_df is not None:
            unk_texts = bundle.unk_df["text"].map(norm_text).tolist()
            # sample a limited amount for speed
            k = min(rcfg.unk_anchor_max, len(unk_texts))
            rng = np.random.RandomState(seed + 999)
            sel = rng.choice(len(unk_texts), size=k, replace=False if k <= len(unk_texts) else True)
            unk_s = [unk_texts[i] for i in sel]
        else:
            unk_s = []

        # unknown anchors: y ignored, known=0, unk_flag=1
        unk_y = [IGNORE_Y] * len(unk_s)
        unk_w = [1.0] * len(unk_s)
        unk_ood = [0] * len(unk_s)
        unk_flag = [1] * len(unk_s)

        train_texts = src_train_texts + unk_s
        train_y = src_y + unk_y
        train_w = src_w + unk_w
        train_ood = src_ood + unk_ood
        train_unk = src_unk + unk_flag

        tr_loader = make_loader(tokenizer, train_texts, train_y, train_w, train_ood, train_unk,
                                batch_size=rcfg.train_batch, max_len=rcfg.max_len, shuffle=True)

        # unlabeled loader for consistency (use target unlabeled for stage0 too, mild regularization)
        unl_loader = make_loader(tokenizer, tgt_unl_texts, None, None, None, None,
                                 batch_size=rcfg.train_batch, max_len=rcfg.max_len, shuffle=True)

        # stage0: train y-head strongly, ood-head only if method uses it later OR to enable ours.
        # For baseline fairness, we still can pretrain ood-head; evaluation of baselines uses maxsoft anyway.
        train_one_stage(
            student=student,
            teacher=teacher,
            tokenizer=tokenizer,
            device=device,
            train_loader=tr_loader,
            unl_loader=unl_loader,
            epochs=rcfg.epochs_source,
            lr=rcfg.lr,
            wd=rcfg.wd,
            grad_accum=rcfg.grad_accum,
            use_teacher=bool(method.use_teacher),
            use_consistency=False,
            use_ood_head=True,          # always pretrain ood head with anchors if available
            lambda_ood=0.5,
            lambda_unk=0.1,
            lambda_u=0.0,
            known_conf_thresh=0.7,
            y_conf_thresh=0.6,
            ema_decay=rcfg.ema_decay,
        )

        torch.save(student.state_dict(), ckpt_path)
        if teacher is not None:
            teacher.load_state_dict(student.state_dict())

    # run id
    run_id = f"{suite}__{method.name}__seed{seed}__{cond.get('tag','')}"
    run_dir = os.path.join(out_root, "per_run", run_id)
    ensure_dir(run_dir)

    # evaluation helpers
    def eval_pack(m: DualHeadTransformer, tau: float, ood_mode: str) -> Dict[str, Any]:
        return {
            "tgt_val": evaluate(m, tokenizer, device, tgt_val_texts, tgt_val_labels,
                                max_len=rcfg.max_len, pred_batch=rcfg.pred_batch, tau=tau, ood_score_mode=ood_mode),
            "tgt_test": evaluate(m, tokenizer, device, tgt_test_texts, tgt_test_labels,
                                 max_len=rcfg.max_len, pred_batch=rcfg.pred_batch, tau=tau, ood_score_mode=ood_mode),
        }

    # choose OOD score mode
    # baselines -> maxsoft, ours_route2 -> knownness
    ood_mode = "knownness" if method.use_ood_head else "maxsoft"

    # tau selection (Protocol A uses target val labels; Protocol B uses unlabeled prior matching)
    def pick_tau_protocolA(m: DualHeadTransformer) -> Tuple[float, float]:
        loader = make_loader(tokenizer, tgt_val_texts, None, None, None, None,
                             batch_size=rcfg.pred_batch, max_len=rcfg.max_len, shuffle=False)
        ly, lo = predict_scores(m, loader, device)
        p1, _, _ = logits_to_p1_and_known(ly, lo)
        tau, f1 = best_tau_for_macro_f1(np.array(tgt_val_labels, dtype=int), p1)
        return tau, f1

    def pick_tau_protocolB(m: DualHeadTransformer) -> Tuple[float, float, float]:
        # Use target unlabeled predictions filtered by knownness (if available) to estimate pi, then tau by quantile.
        loader = make_loader(tokenizer, tgt_unl_texts, None, None, None, None,
                             batch_size=rcfg.pred_batch, max_len=rcfg.max_len, shuffle=False)
        ly, lo = predict_scores(m, loader, device)
        p1, _, known = logits_to_p1_and_known(ly, lo)

        # knownness filter if we have ood head; else no filter
        if method.use_ood_head:
            mask = known >= 0.5
        else:
            mask = np.ones_like(p1, dtype=bool)

        p1k = p1[mask]
        if len(p1k) < 50:
            # fallback
            return 0.5, float(np.mean(bundle.src_train_labels)), float("nan")

        # BBSE pi (shrink)
        # Build source-val confusion
        src_val_loader = make_loader(tokenizer, src_val_texts, None, None, None, None,
                                     batch_size=rcfg.pred_batch, max_len=rcfg.max_len, shuffle=False)
        ly_s, lo_s = predict_scores(m, src_val_loader, device)
        p1_s, _, _ = logits_to_p1_and_known(ly_s, lo_s)
        y_true_src = np.array(src_val_labels, dtype=int)
        y_pred_src = (p1_s >= 0.5).astype(int)
        y_pred_tgt = (p1k >= 0.5).astype(int)

        pi_bbse = estimate_pi_bbse_binary(y_true_src, y_pred_src, y_pred_tgt)
        pi_src = float(np.mean(bundle.src_train_labels))
        pi_hat = (1.0 - rcfg.bbse_shrink) * pi_bbse + rcfg.bbse_shrink * pi_src
        pi_hat = float(np.clip(pi_hat, 1e-3, 1.0 - 1e-3))

        tau = float(np.quantile(p1k, 1.0 - pi_hat))
        return tau, pi_hat, pi_bbse

    # quick eval before adapt
    tauA, selA = pick_tau_protocolA(student) if rcfg.protocol == "A" else (0.5, float("nan"))
    if rcfg.protocol == "B":
        tauB, pi_hat, pi_bbse = pick_tau_protocolB(student)
        tau_init = tauB
        select_metric_name = "unsup_proxy"  # no labels used
        select_metric_value = float("nan")
    else:
        tau_init = tauA
        select_metric_name = "macro_f1"
        select_metric_value = float(selA)

    before_05 = eval_pack(student, tau=0.5, ood_mode=ood_mode)
    before_tau = eval_pack(student, tau=tau_init, ood_mode=ood_mode)

    # source_only ends here
    if method.name == "source_only":
        res = {
            "suite": suite,
            "seed": seed,
            "method": method.name,
            "condition": cond,
            "selected_round": 0,
            "tau_star": float(tau_init),
            "select_metric_name": select_metric_name,
            "select_metric_value": float(select_metric_value) if select_metric_value is not None else None,
            "before_05": before_05,
            "before_tau": before_tau,
            "final_05": before_05,
            "final_tau": before_tau,
        }
        safe_json_dump(res, os.path.join(run_dir, "result.json"))
        return res

    # ============================================================
    # Adaptation rounds
    # ============================================================

    # pseudo pool
    pseudo_pool: Dict[int, Tuple[int, float]] = {}

    # best selection
    best = {
        "round": 0,
        "tau": float(tau_init),
        "metric": float(select_metric_value) if select_metric_value is not None else -1e9,
        "ckpt": None,
    }

    unl_loader = make_loader(tokenizer, tgt_unl_texts, None, None, None, None,
                             batch_size=rcfg.train_batch, max_len=rcfg.max_len, shuffle=True)

    # prepare src-val for BBSE
    src_val_loader_pred = make_loader(tokenizer, src_val_texts, None, None, None, None,
                                      batch_size=rcfg.pred_batch, max_len=rcfg.max_len, shuffle=False)
    ly_sv, lo_sv = predict_scores(student, src_val_loader_pred, device)
    p1_sv, _, _ = logits_to_p1_and_known(ly_sv, lo_sv)
    y_true_src = np.array(src_val_labels, dtype=int)
    y_pred_src = (p1_sv >= 0.5).astype(int)
    pi_src = float(np.mean(bundle.src_train_labels))

    # optional teacher init
    if teacher is not None:
        teacher.load_state_dict(student.state_dict())

    for r in range(1, rcfg.adapt_rounds + 1):
        frac = rcfg.pseudo_fracs[min(r-1, len(rcfg.pseudo_fracs)-1)]
        lam_u = rcfg.lambda_u[min(r-1, len(rcfg.lambda_u)-1)]

        # predict on all target unlabeled
        unl_pred_loader = make_loader(tokenizer, tgt_unl_texts, None, None, None, None,
                                      batch_size=rcfg.pred_batch, max_len=rcfg.max_len, shuffle=False)
        pred_model = teacher if (method.use_teacher and teacher is not None) else student
        ly_u, lo_u = predict_scores(pred_model, unl_pred_loader, device)
        p1_u, maxsoft_u, known_u = logits_to_p1_and_known(ly_u, lo_u)

        # build id_mask for pseudo labels
        if method.use_ood_head:
            id_mask = (known_u >= rcfg.pseudo_known_thresh) & (maxsoft_u >= rcfg.pseudo_y_conf_thresh)
        else:
            # baseline has no knownness head -> no ood filtering
            id_mask = (maxsoft_u >= rcfg.pseudo_y_conf_thresh)

        # choose pi_used
        if method.use_bbse_pi:
            y_pred_tgt = (p1_u[id_mask] >= 0.5).astype(int) if id_mask.sum() > 0 else (p1_u >= 0.5).astype(int)
            pi_bbse = estimate_pi_bbse_binary(y_true_src, y_pred_src, y_pred_tgt)
            pi_used = float(np.clip(pi_bbse, 1e-3, 1-1e-3))
        else:
            pi_used = pi_src

        # build new pseudo pool
        if method.pseudo_strategy != "none":
            new_pool = build_pseudo_pool(
                p1=p1_u,
                pi_used=pi_used,
                frac=frac,
                seed=seed + 1000 * r,
                id_mask=id_mask,
                strategy=method.pseudo_strategy,
            )
            for idx, (yy, ww) in new_pool.items():
                if idx in pseudo_pool:
                    if ww > pseudo_pool[idx][1]:
                        pseudo_pool[idx] = (yy, ww)
                else:
                    pseudo_pool[idx] = (yy, ww)

        # unknown mining (ours_route2 only)
        unk_texts_round: List[str] = []
        if method.use_unknown_mining and method.use_ood_head:
            k_mine = int(round(rcfg.unk_mine_frac * len(tgt_unl_texts)))
            k_mine = max(0, min(k_mine, len(tgt_unl_texts)))
            if k_mine > 0:
                order = np.argsort(known_u)  # low knownness first
                mined_idx = order[:k_mine].tolist()
                unk_texts_round.extend([tgt_unl_texts[i] for i in mined_idx])

            # add politics anchors
            if bundle.unk_df is not None:
                unk_pool = bundle.unk_df["text"].map(norm_text).tolist()
                k = min(rcfg.unk_anchor_max, len(unk_pool))
                rng = np.random.RandomState(seed + 777 + r)
                sel = rng.choice(len(unk_pool), size=k, replace=False if k <= len(unk_pool) else True)
                unk_texts_round.extend([unk_pool[i] for i in sel])

        # build training set = source + pseudo (+ unknown if enabled)
        pseudo_idx = list(pseudo_pool.keys())
        pseudo_texts = [tgt_unl_texts[i] for i in pseudo_idx]
        pseudo_labels = [pseudo_pool[i][0] for i in pseudo_idx]
        pseudo_weights = [pseudo_pool[i][1] for i in pseudo_idx]

        # source: y labeled, known=1
        tr_texts = src_train_texts + pseudo_texts
        tr_y = src_train_labels + pseudo_labels
        tr_w = [1.0] * len(src_train_texts) + pseudo_weights
        tr_ood = [1] * (len(src_train_texts) + len(pseudo_texts))
        tr_unk_flag = [0] * (len(src_train_texts) + len(pseudo_texts))

        # unknown: y ignored, known=0, unk_flag=1
        if method.use_ood_head and method.use_unknown_mining and len(unk_texts_round) > 0:
            tr_texts += unk_texts_round
            tr_y += [IGNORE_Y] * len(unk_texts_round)
            tr_w += [1.0] * len(unk_texts_round)
            tr_ood += [0] * len(unk_texts_round)
            tr_unk_flag += [1] * len(unk_texts_round)

        train_loader = make_loader(tokenizer, tr_texts, tr_y, tr_w, tr_ood, tr_unk_flag,
                                   batch_size=rcfg.train_batch, max_len=rcfg.max_len, shuffle=True)

        # train one round
        train_one_stage(
            student=student,
            teacher=teacher,
            tokenizer=tokenizer,
            device=device,
            train_loader=train_loader,
            unl_loader=unl_loader,
            epochs=rcfg.adapt_epochs,
            lr=rcfg.lr,
            wd=rcfg.wd,
            grad_accum=rcfg.grad_accum,
            use_teacher=method.use_teacher,
            use_consistency=method.use_consistency,
            use_ood_head=method.use_ood_head,
            lambda_ood=rcfg.lambda_ood if method.use_ood_head else 0.0,
            lambda_unk=rcfg.lambda_unk if (method.use_ood_head and method.use_unknown_mining) else 0.0,
            lambda_u=lam_u if method.use_consistency else 0.0,
            known_conf_thresh=rcfg.consistency_known_thresh,
            y_conf_thresh=rcfg.consistency_y_conf_thresh,
            ema_decay=rcfg.ema_decay,
        )

        # select tau + compute selection metric
        if rcfg.protocol == "A":
            tau_r, f1_r = pick_tau_protocolA(student)
            sel_metric = f1_r
            sel_name = "macro_f1"
        else:
            tau_r, pi_hat, pi_bbse = pick_tau_protocolB(student)
            # selection without labels is hard; keep placeholder
            sel_metric = -1e9
            sel_name = "unsup_proxy"

        # update best checkpoint (Protocol A only, to get strong results quickly)
        if rcfg.protocol == "A" and sel_metric > best["metric"]:
            best = {"round": r, "tau": float(tau_r), "metric": float(sel_metric), "ckpt": os.path.join(run_dir, f"best_round{r}.pt")}
            torch.save(student.state_dict(), best["ckpt"])

    # load best if exists
    if best["ckpt"] is not None and os.path.exists(best["ckpt"]):
        student.load_state_dict(safe_torch_load(best["ckpt"], map_location=device))

    # final tau
    if rcfg.protocol == "A":
        tau_star, f1_star = pick_tau_protocolA(student)
        select_metric_name = "macro_f1"
        select_metric_value = float(f1_star)
    else:
        tau_star, pi_hat, pi_bbse = pick_tau_protocolB(student)
        select_metric_name = "unsup_proxy"
        select_metric_value = None

    final_05 = eval_pack(student, tau=0.5, ood_mode=ood_mode)
    final_tau = eval_pack(student, tau=tau_star, ood_mode=ood_mode)

    res = {
        "suite": suite,
        "seed": seed,
        "method": method.name,
        "condition": cond,
        "selected_round": int(best["round"]),
        "tau_star": float(tau_star),
        "select_metric_name": select_metric_name,
        "select_metric_value": select_metric_value,
        "before_05": before_05,
        "before_tau": before_tau,
        "final_05": final_05,
        "final_tau": final_tau,
    }
    safe_json_dump(res, os.path.join(run_dir, "result.json"))
    return res


def run_suite(bundle: DataBundle, rcfg: RunConfig, suite: str, seeds: List[int], out_root: str) -> pd.DataFrame:
    methods = get_methods(route2=True)
    conditions = build_conditions(rcfg.mode, suite)

    rows = []
    for cond in conditions:
        for seed in seeds:
            for method in methods:
                print("\n" + "=" * 80)
                print(f"[RUN] suite={suite} method={method.name} seed={seed} cond={cond}")
                print("=" * 80)
                res = run_one(bundle, rcfg, suite, cond, method, seed, out_root)
                row = flatten_row(res)
                rows.append(row)
                pd.DataFrame(rows).to_csv(os.path.join(out_root, "all_results_partial.csv"), index=False, encoding="utf-8")

    df = pd.DataFrame(rows)
    return df


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", type=int, default=1, help="1=fast(DistilBERT), 0=full(BERT)")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "paper"])
    parser.add_argument("--protocol", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--out_dir", type=str, default="runs/route2_fast")
    parser.add_argument("--seeds", type=str, default="13")
    parser.add_argument("--suites", type=str, default="base,open_set,combined,label_shift")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    rcfg = make_run_config(args)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base_dir, rcfg.out_dir)
    ensure_dir(out_root)

    device = get_device()
    print(f"[Device] {device}")
    print(f"[Config] model={rcfg.model_name} fast={rcfg.fast} mode={rcfg.mode} protocol={rcfg.protocol}")
    print(f"[Seeds] {seeds}")
    print(f"[Suites] {suites}")

    bundle = load_bundle(base_dir)
    print("[Data] Loaded.")
    print(f"[Source] train={len(bundle.src_train_texts)} val={len(bundle.src_val_texts)} test={len(bundle.src_test_texts)}")
    print(f"[Target] train={len(bundle.tgt_train_df)} val={len(bundle.tgt_val_df)} test={len(bundle.tgt_test_df)}")
    if bundle.unk_df is not None:
        print(f"[Unknown] politics={len(bundle.unk_df)}")
    else:
        print("[Unknown] politics.csv not found.")

    all_dfs = []
    for suite in suites:
        if suite in ("open_set", "combined") and bundle.unk_df is None:
            print(f"[Skip] suite={suite} (politics.csv missing)")
            continue
        if suite in ("label_shift", "combined") and (not bundle.tgt_train_has_labels):
            print(f"[Skip] suite={suite} (target train has no labels -> cannot construct controlled priors)")
            continue

        df = run_suite(bundle, rcfg, suite, seeds, out_root)
        df.to_csv(os.path.join(out_root, f"all_results_{suite}.csv"), index=False, encoding="utf-8")
        all_dfs.append(df)

        group_cols = ["suite", "cond_tag", "method"]
        metrics = [
            "tgt_test_macro_f1_tau",
            "tgt_test_macro_f1_05",
            "ood_test_auroc",
            "ood_test_fpr95",
        ]
        metrics = [m for m in metrics if m in df.columns]
        summary = aggregate_results(df, group_cols, metrics)
        summary.to_csv(os.path.join(out_root, f"summary_{suite}.csv"), index=False, encoding="utf-8")
        print(f"[Saved] summary_{suite}.csv")

    if all_dfs:
        df_all = pd.concat(all_dfs, axis=0).reset_index(drop=True)
        df_all.to_csv(os.path.join(out_root, "all_results.csv"), index=False, encoding="utf-8")
        print(f"[Saved] all_results.csv")


if __name__ == "__main__":
    main()
