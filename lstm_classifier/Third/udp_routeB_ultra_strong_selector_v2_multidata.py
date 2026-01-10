#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route B++ (Ultra): Candidate Adaptation + Robust Unsupervised Selector (+ optional Ensemble)
==========================================================================================

This script is a **drop-in upgrade** of the previous Route-B prototype:

- Keeps the same "read from local files" style used in:
  `udp_multiseed_pas_unida_pp_multiseed_protoalign_best_fixed10.py`
- No CLI arguments are required. All hyperparameters are inside the Config dataclass.
- Runs multi-seed experiments and saves:
  - per-seed JSON results
  - all-seed results + summary JSON
  - per-seed candidate logs (metrics + selection decisions)

Why this version is stronger / more stable
------------------------------------------
Your previous Route-B was already a good idea: **generate candidates** and use an **unsupervised selector**
to avoid hand-tuning. But the logs show a recurring failure mode:

- Self-training variants can alternate between "excellent" and "bad" depending on seed.
- The selector sometimes chooses the wrong candidate (large regressions on a few seeds),
  which pulls down the multi-seed mean.

This upgrade targets that problem directly:

1) Stronger candidates
   - Adds a **Mean-Teacher FixMatch**-style text UDA candidate (EMA teacher + confidence threshold
     + strong token-masking augmentation). This is typically more robust than plain self-training
     because pseudo-labels come from a stabilized teacher.

2) More informative unsupervised selection
   The selector no longer relies on a single proxy. We compute multiple signals:

   - Reverse validation accuracy (RVA): pseudo-label target -> train a linear probe -> test on source-val
   - InfoMax proxy on target-val (mutual information proxy): H(mean p) - mean H(p)
   - Prototype agreement: candidate prediction vs. source-prototype nearest-centroid labels on target-val
   - Target cluster separation: inter-class centroid distance / intra-class spread on target-val embeddings
   - Prediction collapse checks: minimum class fraction + KL to source class prior
   - Dropout consistency: KL divergence between two stochastic forward passes

3) Conservative decision rule (to avoid "selector accidents")
   - Use `selftrain_closed` as an **anchor**.
   - Switch away from anchor only if the best candidate is *clearly* better in rank-aggregated score
     AND does not hurt source-val beyond a small tolerance.
   This preserves the good cases while preventing the catastrophic ones.

4) Optional ensemble
   - If you enable `ensemble_k>=2`, the script will also report an **unsupervised weighted ensemble**
     of the top-k safe candidates. This often improves mean accuracy and reduces variance.

Important: this is still "no target labels in selection"
--------------------------------------------------------
Target labels (val/test) are used **only for reporting** when present. The selector uses only:
- source labeled validation (safety)
- target unlabeled texts (proxy)

"""

from __future__ import annotations

import os
import re
import json
import math
import random
import sys
import traceback
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)

# =============================================================================
# Logging helpers
# =============================================================================
class FileTee:
    """
    Redirect stdout/stderr to both terminal and a log file.
    Usage:
        tee = FileTee(log_path)
        ...
        tee.close()
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file_obj = open(log_path, "a", encoding="utf-8")

        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr

        sys.stdout = StreamTee(self.orig_stdout, self.file_obj)
        sys.stderr = StreamTee(self.orig_stderr, self.file_obj)

    def close(self) -> None:
        # restore streams first
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        try:
            self.file_obj.flush()
        finally:
            self.file_obj.close()

class StreamTee:
    """Duplicate writes to both a stream (terminal) and a log file."""
    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file_obj = file_obj

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file_obj.write(data)
        self.file_obj.flush()

    def flush(self):
        self.stream.flush()
        self.file_obj.flush()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_json_dump(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic can slow down; keep fast by default.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Text normalization + leakage cleaning
# =============================================================================

_ws_re = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _ws_re.sub(" ", s)
    return s


def guess_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    cols = [c.strip() for c in df.columns.tolist()]
    lower = {c.lower(): c for c in cols}

    text_candidates = ["text", "sentence", "content", "tweet", "review", "comment", "body"]
    text_col = None
    for k in text_candidates:
        if k in lower:
            text_col = lower[k]
            break
    if text_col is None:
        for c in cols:
            if df[c].dtype == object:
                text_col = c
                break
    if text_col is None:
        text_col = cols[0]

    label_candidates = ["label", "labels", "y", "target", "class"]
    label_col = None
    for k in label_candidates:
        if k in lower:
            label_col = lower[k]
            break
    return text_col, label_col


def leakage_report(
    src_splits: Dict[str, List[str]],
    tgt_splits: Dict[str, List[str]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"source": {}, "target": {}, "overlap": {}}

    def stats(texts: List[str]) -> Dict[str, Any]:
        norm = [normalize_text(x) for x in texts]
        uniq = len(set(norm))
        dup = len(norm) - uniq
        return {"n": len(norm), "unique": uniq, "dup": dup, "dup_ratio": (dup / max(1, len(norm)))}

    for k, v in src_splits.items():
        report["source"][k] = stats(v)
    for k, v in tgt_splits.items():
        report["target"][k] = stats(v)

    def overlap(a: List[str], b: List[str]) -> Dict[str, Any]:
        A = set(normalize_text(x) for x in a)
        B = set(normalize_text(x) for x in b)
        inter = A & B
        return {"overlap_unique": len(inter), "ratio_in_A": len(inter) / max(1, len(A)), "ratio_in_B": len(inter) / max(1, len(B))}

    # within source
    sk = list(src_splits.keys())
    for i in range(len(sk)):
        for j in range(i + 1, len(sk)):
            report["overlap"][f"source_{sk[i]}__{sk[j]}"] = overlap(src_splits[sk[i]], src_splits[sk[j]])
    # within target
    tk = list(tgt_splits.keys())
    for i in range(len(tk)):
        for j in range(i + 1, len(tk)):
            report["overlap"][f"target_{tk[i]}__{tk[j]}"] = overlap(tgt_splits[tk[i]], tgt_splits[tk[j]])

    return report


def remove_overlaps_keep_test(
    train_texts: List[str], train_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    test_texts: List[str], test_labels: List[int],
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int], Dict[str, int]]:
    """
    Remove exact normalized overlaps:
      - remove train items that appear in val/test
      - remove val items that appear in test
    Keep test unchanged.
    """
    norm_train = [normalize_text(x) for x in train_texts]
    norm_val = [normalize_text(x) for x in val_texts]
    norm_test = [normalize_text(x) for x in test_texts]

    set_test = set(norm_test)
    set_val = set(norm_val)

    keep_train = [i for i, t in enumerate(norm_train) if (t not in set_val and t not in set_test)]
    removed_train = len(train_texts) - len(keep_train)
    train_texts2 = [train_texts[i] for i in keep_train]
    train_labels2 = [train_labels[i] for i in keep_train]

    keep_val = [i for i, t in enumerate(norm_val) if (t not in set_test)]
    removed_val = len(val_texts) - len(keep_val)
    val_texts2 = [val_texts[i] for i in keep_val]
    val_labels2 = [val_labels[i] for i in keep_val]

    info = {"removed_train": removed_train, "removed_val": removed_val}
    return train_texts2, train_labels2, val_texts2, val_labels2, test_texts, test_labels, info


# =============================================================================
# Metrics
# =============================================================================

def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean([int(t == p) for t, p in zip(y_true, y_pred)]))


def _confusion_counts(y_true: List[int], y_pred: List[int], K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.zeros(K, dtype=np.int64)
    fp = np.zeros(K, dtype=np.int64)
    fn = np.zeros(K, dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t < 0 or t >= K:
            if 0 <= p < K:
                fp[p] += 1
            continue
        if p == t:
            tp[t] += 1
        else:
            fn[t] += 1
            if 0 <= p < K:
                fp[p] += 1
    return tp, fp, fn


def macro_f1_balanced_acc(y_true: List[int], y_pred: List[int], K: int) -> Tuple[float, float]:
    tp, fp, fn = _confusion_counts(y_true, y_pred, K)
    f1s = []
    recalls = []
    for c in range(K):
        prec = tp[c] / max(1, tp[c] + fp[c])
        rec = tp[c] / max(1, tp[c] + fn[c])
        f1 = 0.0 if (prec + rec == 0) else (2 * prec * rec / (prec + rec))
        f1s.append(f1)
        recalls.append(rec)
    return float(np.mean(f1s)), float(np.mean(recalls))


# =============================================================================
# Datasets / Collators
# =============================================================================

class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


def make_collate_fn(tokenizer, max_len: int, with_labels: bool):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        if with_labels:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc
    return collate


class MLMDataset(Dataset):
    """
    Pre-tokenized dataset for MLM DAPT.
    Returns dict of 1D tensors (variable length), DataCollatorForLanguageModeling will pad+mask.
    """
    def __init__(self, texts: List[str], tokenizer, max_len: int):
        self.enc = tokenizer(texts, padding=False, truncation=True, max_length=max_len)

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.enc.items()}
        return item


# =============================================================================
# Model utilities
# =============================================================================

def get_base_encoder(model: nn.Module) -> nn.Module:
    if hasattr(model, "bert"):
        return model.bert
    if hasattr(model, "roberta"):
        return model.roberta
    if hasattr(model, "deberta"):
        return model.deberta
    if hasattr(model, "electra"):
        return model.electra
    if hasattr(model, "base_model"):
        return model.base_model
    raise AttributeError("Cannot find base encoder module (bert/roberta/deberta/electra/base_model).")


def load_dapt_encoder_into_classifier(model: nn.Module, dapt_encoder_dir: str) -> None:
    enc_model = AutoModel.from_pretrained(dapt_encoder_dir)
    base = get_base_encoder(model)
    missing, unexpected = base.load_state_dict(enc_model.state_dict(), strict=False)
    if missing or unexpected:
        print(f"[Model] DAPT load: missing={len(missing)} unexpected={len(unexpected)}")
    del enc_model


def build_classifier(model_name: str, num_labels: int, device: torch.device, dapt_encoder_dir: Optional[str] = None) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if dapt_encoder_dir is not None and os.path.isdir(dapt_encoder_dir):
        load_dapt_encoder_into_classifier(model, dapt_encoder_dir)
        print(f"[Model] loaded DAPT encoder weights from: {dapt_encoder_dir}")
    model.to(device)
    return model


# =============================================================================
# Train / Eval
# =============================================================================

@torch.no_grad()
def predict_proba(model: nn.Module, tokenizer, texts: List[str], cfg, device: torch.device) -> np.ndarray:
    model.eval()
    ds = TextLabelDataset(texts, labels=None)
    loader = DataLoader(
        ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(tokenizer, cfg.max_len, with_labels=False),
    )
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs, return_dict=True)
        probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
    if not all_probs:
        return np.zeros((0, cfg.num_labels), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def predict_label(model: nn.Module, tokenizer, texts: List[str], cfg, device: torch.device) -> List[int]:
    probs = predict_proba(model, tokenizer, texts, cfg, device)
    if probs.shape[0] == 0:
        return []
    return probs.argmax(axis=1).astype(int).tolist()


@torch.no_grad()
def eval_split(model: nn.Module, tokenizer, texts: List[str], labels: List[int], cfg, device: torch.device, K: int) -> Dict[str, Any]:
    """Evaluate on a labeled split.

    Returns:
      - acc: overall accuracy (includes unknown if present in labels and enabled via rejection)
      - f1, bal_acc: macro-F1 / balanced accuracy on *known* classes only (0..K-1)
      - acc_known: accuracy on known subset only
      - unknown_rate / pred_unknown_rate (if unknown exists)
      - per_class: precision/recall/f1/support for each class (known + optional unknown)
      - confusion: confusion matrix (known + optional unknown) as nested lists
    """

    n = len(texts)
    if n == 0:
        return {"acc": float("nan"), "f1": float("nan"), "bal_acc": float("nan")}

    # Unknown class is represented as id == K (consistent with load_data()).
    unknown_id = K if any((y == K) for y in labels) else None

    if cfg.enable_unknown_reject and unknown_id is not None:
        proba = predict_proba(model, tokenizer, texts, cfg, device)
        pmax = proba.max(axis=1)
        argm = proba.argmax(axis=1)

        y_pred: List[int] = []
        for i in range(n):
            if float(pmax[i]) < float(cfg.reject_tau):
                y_pred.append(int(unknown_id))
            else:
                y_pred.append(int(argm[i]))
    else:
        y_pred = predict_label(model, tokenizer, texts, cfg, device)

    acc = accuracy(labels, y_pred)

    # Known-only metrics (paper-friendly when unknown exists)
    f1, bal = macro_f1_balanced_acc(labels, y_pred, K=K)

    # Known-only accuracy
    known_idx = [i for i, y in enumerate(labels) if 0 <= int(y) < K]
    if known_idx:
        acc_known = accuracy([labels[i] for i in known_idx], [y_pred[i] for i in known_idx])
    else:
        acc_known = float("nan")

    # Unknown rates (if present)
    if unknown_id is not None:
        unknown_rate = float(sum(1 for y in labels if int(y) == int(unknown_id))) / float(n)
        pred_unknown_rate = float(sum(1 for y in y_pred if int(y) == int(unknown_id))) / float(n)
    else:
        unknown_rate = 0.0
        pred_unknown_rate = 0.0

    # Per-class metrics (known + optional unknown)
    per_class: Dict[str, Dict[str, float]] = {}
    class_ids: List[int] = list(range(K))
    if unknown_id is not None:
        class_ids.append(int(unknown_id))

    def safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    # Confusion matrix (only for the classes we track)
    id_to_row = {cid: i for i, cid in enumerate(class_ids)}
    C = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for t, p in zip(labels, y_pred):
        t = int(t)
        p = int(p)
        if (t in id_to_row) and (p in id_to_row):
            C[id_to_row[t], id_to_row[p]] += 1

    for cid in class_ids:
        tp = int(sum(1 for t, p in zip(labels, y_pred) if int(t) == cid and int(p) == cid))
        fp = int(sum(1 for t, p in zip(labels, y_pred) if int(t) != cid and int(p) == cid))
        fn = int(sum(1 for t, p in zip(labels, y_pred) if int(t) == cid and int(p) != cid))
        support = int(sum(1 for t in labels if int(t) == cid))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1_c = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

        per_class[str(cid)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_c),
            "support": float(support),
        }

    out: Dict[str, Any] = {
        "acc": float(acc),
        "f1": float(f1),
        "bal_acc": float(bal),
        "acc_known": float(acc_known),
        "unknown_rate": float(unknown_rate),
        "pred_unknown_rate": float(pred_unknown_rate),
        "per_class": per_class if cfg.report_per_class else None,
        "confusion": C.tolist() if cfg.report_per_class else None,
    }
    return out

def train_supervised_source(
    model: nn.Module,
    tokenizer,
    train_texts: List[str], train_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    cfg,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Supervised fine-tuning on labeled source.
    """
    ds_tr = TextLabelDataset(train_texts, train_labels)
    ds_va = TextLabelDataset(val_texts, val_labels)

    loader_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(tokenizer, cfg.max_len, with_labels=True),
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(tokenizer, cfg.max_len, with_labels=True),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    best_state = None
    bad = 0
    history = {"val_acc": [], "train_loss": []}

    for epoch in range(cfg.src_epochs):
        model.train()
        losses = []
        for batch in loader_tr:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids", "labels")}
            out = model(**inputs, return_dict=True)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        tr_loss = float(np.mean(losses)) if losses else 0.0

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader_va:
                inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
                labels = batch["labels"].to(device)
                out = model(**inputs, return_dict=True)
                pred = out.logits.argmax(dim=-1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
        val_acc = correct / max(1, total)
        history["val_acc"].append(float(val_acc))
        history["train_loss"].append(float(tr_loss))
        print(f"  [SRC] epoch={epoch+1}/{cfg.src_epochs} train_loss={tr_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val + 1e-6:
            best_val = float(val_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                print("  [SRC] early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    log = {"best_val_acc": float(best_val), "history": history}
    return model, log


# =============================================================================
# DAPT (MLM)
# =============================================================================

def run_dapt_mlm(
    tgt_train_texts: List[str],
    tokenizer,
    cfg,
    device: torch.device,
    save_dir: str,
) -> str:
    """
    MLM warm-up on target-train unlabeled texts.
    Saves encoder to: save_dir/dapt_encoder
    """
    enc_dir = os.path.join(save_dir, "dapt_encoder")
    if os.path.isdir(enc_dir) and os.path.isfile(os.path.join(enc_dir, "config.json")):
        print(f"  [DAPT] found existing encoder at: {enc_dir} (reuse)")
        return enc_dir

    ensure_dir(enc_dir)
    print("[Phase 1] DAPT (MLM) warm-up on target train...")

    mlm_model = AutoModelForMaskedLM.from_pretrained(cfg.model_name).to(device)
    mlm_model.train()

    ds = MLMDataset(tgt_train_texts, tokenizer, cfg.max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mlm_prob)

    loader = DataLoader(
        ds,
        batch_size=cfg.dapt_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )

    opt = torch.optim.AdamW(mlm_model.parameters(), lr=cfg.dapt_lr, weight_decay=cfg.weight_decay)

    step = 0
    for epoch in range(cfg.dapt_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = mlm_model(**batch, return_dict=True)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), cfg.grad_clip)
            opt.step()

            if (step % cfg.dapt_log_every) == 0:
                print(f"  [DAPT] epoch={epoch+1}/{cfg.dapt_epochs} step={step} loss={float(loss.detach().cpu().item()):.4f}")
            step += 1

    # Save encoder part only
    if hasattr(mlm_model, "bert"):
        mlm_model.bert.save_pretrained(enc_dir)
    else:
        base = getattr(mlm_model, "base_model", None)
        if base is None:
            raise RuntimeError("Cannot locate encoder to save in MLM model.")
        base.save_pretrained(enc_dir)
    tokenizer.save_pretrained(enc_dir)
    print(f"  [DAPT] saved encoder to: {enc_dir}")
    return enc_dir


# =============================================================================
# Adaptation methods (candidates)
# =============================================================================

def _train_one_epoch_on_mixed_hard(
    model: nn.Module,
    tokenizer,
    src_texts: List[str], src_labels: List[int],
    tgt_texts: List[str], tgt_pseudo: List[int], tgt_weights: Optional[np.ndarray],
    cfg,
    device: torch.device,
) -> float:
    """
    Train 1 epoch on mixture of (src labeled) + (tgt pseudo labeled) with hard labels.
    Carries sample weights inside the batch.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_adapt, weight_decay=cfg.weight_decay)

    mix_texts = src_texts + tgt_texts
    mix_labels = src_labels + tgt_pseudo

    if tgt_weights is None:
        mix_w = np.ones(len(mix_texts), dtype=np.float32)
    else:
        mix_w = np.concatenate([np.ones(len(src_texts), dtype=np.float32), tgt_weights.astype(np.float32)], axis=0)

    class _MixDataset(Dataset):
        def __init__(self, texts, labels, weights):
            self.texts = texts
            self.labels = labels
            self.weights = weights

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {"text": self.texts[idx], "label": int(self.labels[idx]), "weight": float(self.weights[idx])}

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_len,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc["weights"] = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
        return enc

    ds = _MixDataset(mix_texts, mix_labels, mix_w)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=_collate)

    losses = []
    for batch in loader:
        weights = batch["weights"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        labels = batch["labels"].to(device)

        out = model(**inputs, return_dict=True)
        ce = F.cross_entropy(out.logits, labels, reduction="none")
        loss = (ce * weights).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def adapt_selftrain_closed(
    model: nn.Module,
    tokenizer,
    src_train_texts: List[str], src_train_labels: List[int],
    src_val_texts: List[str], src_val_labels: List[int],
    tgt_train_texts: List[str],
    cfg,
    device: torch.device,
    tau_schedule: List[float],
    balanced: bool = False,
    per_class_cap: Optional[int] = None,
    use_distribution_alignment: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Closed-set self-training on unlabeled target-train.

    Improvements over the "plain" variant:
    - optional distribution alignment using source label prior (reduces class-collapse risk)
    - confidence-based weighting mapped to [0.5, 1.0]
    - optional class-balanced selection / cap

    Still uses HARD pseudo labels (keeps runtime small).
    """
    K = cfg.num_labels
    log: Dict[str, Any] = {"rounds": [], "balanced": balanced, "use_DA": use_distribution_alignment}

    # source prior
    src_counts = np.bincount(np.array(src_train_labels, dtype=np.int64), minlength=K).astype(np.float64)
    p_src = src_counts / max(1.0, float(src_counts.sum()))
    p_src = np.clip(p_src, 1e-6, 1.0)
    p_src = p_src / p_src.sum()

    for r, tau in enumerate(tau_schedule):
        model.eval()
        probs = predict_proba(model, tokenizer, tgt_train_texts, cfg, device)  # [N, K]
        if probs.shape[0] == 0:
            log["rounds"].append({"round": r+1, "tau": tau, "kept": 0})
            continue

        p = probs.astype(np.float64)
        p = np.clip(p, 1e-12, 1.0)

        if use_distribution_alignment:
            p_mean = p.mean(axis=0)
            p_mean = np.clip(p_mean, 1e-6, 1.0)
            p_adj = p * (p_src / p_mean)[None, :]
            p_adj = p_adj / np.clip(p_adj.sum(axis=1, keepdims=True), 1e-12, 1e12)
            p = p_adj

        conf = p.max(axis=1)
        pred = p.argmax(axis=1).astype(int)

        keep_idx = np.where(conf >= float(tau))[0].tolist()

        if balanced:
            idx_by_c = {c: [] for c in range(K)}
            for i in keep_idx:
                idx_by_c[int(pred[i])].append(i)
            new_keep = []
            for c in range(K):
                ci = idx_by_c[c]
                ci.sort(key=lambda i: float(conf[i]), reverse=True)
                if per_class_cap is not None:
                    ci = ci[:int(per_class_cap)]
                new_keep.extend(ci)
            keep_idx = sorted(new_keep)

        kept = len(keep_idx)
        if kept == 0:
            log["rounds"].append({"round": r+1, "tau": float(tau), "kept": 0})
            continue

        tgt_texts_sel = [tgt_train_texts[i] for i in keep_idx]
        tgt_pseudo = pred[keep_idx].tolist()

        w = conf[keep_idx].astype(np.float32)
        # normalize -> [0.5, 1.0]
        w = 0.5 + 0.5 * (w - w.min()) / max(1e-6, float(w.max() - w.min()))

        loss = _train_one_epoch_on_mixed_hard(
            model, tokenizer,
            src_train_texts, src_train_labels,
            tgt_texts_sel, tgt_pseudo, w,
            cfg, device,
        )

        src_val = eval_split(model, tokenizer, src_val_texts, src_val_labels, cfg, device, K=K)["acc"]
        log["rounds"].append({"round": r+1, "tau": float(tau), "kept": int(kept), "loss": float(loss), "src_val_acc": float(src_val)})
        print(f"  [ST] round={r+1}/{len(tau_schedule)} tau={tau:.2f} kept={kept} loss={loss:.4f} src_val_acc={src_val:.4f}")

    return model, log


def _mask_tokens_strong_aug(batch: Dict[str, torch.Tensor], tokenizer, p: float) -> Dict[str, torch.Tensor]:
    """
    Strong augmentation for FixMatch-style training: random token masking (classification-time).
    """
    input_ids = batch["input_ids"].clone()
    attn = batch["attention_mask"]
    tok_type = batch.get("token_type_ids", None)

    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
    special = set(tokenizer.all_special_ids)

    # sample mask positions
    B, L = input_ids.size()
    rand = torch.rand((B, L), device=input_ids.device)
    can_mask = attn.bool()
    # do not mask special tokens
    for sid in special:
        can_mask = can_mask & (input_ids != sid)

    mask_pos = (rand < p) & can_mask
    input_ids[mask_pos] = int(mask_id)

    out = {"input_ids": input_ids, "attention_mask": attn}
    if tok_type is not None:
        out["token_type_ids"] = tok_type
    return out


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(decay).add_(s_p.data, alpha=(1.0 - decay))


def adapt_fixmatch_mean_teacher(
    student: nn.Module,
    tokenizer,
    src_train_texts: List[str], src_train_labels: List[int],
    src_val_texts: List[str], src_val_labels: List[int],
    tgt_train_texts: List[str],
    cfg,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Mean-Teacher FixMatch-style UDA for text:
      - Student optimized on source supervised CE + target pseudo-label CE
      - Teacher is EMA of student, used to generate pseudo labels on weak view (original text)
      - Strong view: random token masking in input_ids

    This candidate tends to be more stable than plain self-training.
    """
    K = cfg.num_labels
    log: Dict[str, Any] = {"epochs": [], "name": "fixmatch_mt"}

    # teacher init
    teacher = copy.deepcopy(student)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # source prior for distribution alignment
    src_counts = np.bincount(np.array(src_train_labels, dtype=np.int64), minlength=K).astype(np.float64)
    p_src = src_counts / max(1.0, float(src_counts.sum()))
    p_src = np.clip(p_src, 1e-6, 1.0)
    p_src = p_src / p_src.sum()
    p_src_t = torch.tensor(p_src, dtype=torch.float32, device=device)

    # moving average of target predicted distribution (teacher)
    p_model = torch.ones((K,), dtype=torch.float32, device=device) / float(K)

    ds_s = TextLabelDataset(src_train_texts, src_train_labels)
    ds_t = TextLabelDataset(tgt_train_texts, labels=None)

    loader_s = DataLoader(
        ds_s,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(tokenizer, cfg.max_len, with_labels=True),
        drop_last=True,
    )
    loader_t = DataLoader(
        ds_t,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(tokenizer, cfg.max_len, with_labels=False),
        drop_last=True,
    )

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr_adapt, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, max(len(loader_s), len(loader_t)))
    global_step = 0

    best_src_val = -1.0
    best_state = None

    for ep in range(cfg.fm_epochs):
        student.train()
        it_s = iter(loader_s)
        it_t = iter(loader_t)

        sup_losses = []
        unsup_losses = []
        keep_rates = []

        for st in range(steps_per_epoch):
            try:
                b_s = next(it_s)
            except StopIteration:
                it_s = iter(loader_s)
                b_s = next(it_s)

            try:
                b_t = next(it_t)
            except StopIteration:
                it_t = iter(loader_t)
                b_t = next(it_t)

            # source supervised
            xs = {k: v.to(device) for k, v in b_s.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            ys = b_s["labels"].to(device)

            out_s = student(**xs, labels=ys, return_dict=True)
            loss_sup = out_s.loss

            # target pseudo (teacher on weak)
            xt_weak = {k: v.to(device) for k, v in b_t.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            with torch.no_grad():
                teacher.eval()
                out_t = teacher(**xt_weak, return_dict=True)
                p = torch.softmax(out_t.logits, dim=-1)  # [B,K]

                # update moving average p_model
                batch_mean = p.mean(dim=0)
                p_model = cfg.da_momentum * p_model + (1.0 - cfg.da_momentum) * batch_mean

                # distribution alignment
                if cfg.fm_use_da:
                    p_adj = p * (p_src_t / p_model.clamp_min(1e-6)).unsqueeze(0)
                    p_adj = p_adj / p_adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    p = p_adj

                conf, y_hat = torch.max(p, dim=-1)  # [B]
                mask = conf.ge(cfg.fm_tau)

            # strong augmentation
            xt_strong = _mask_tokens_strong_aug(xt_weak, tokenizer, p=cfg.fm_strong_mask_prob)
            out_u = student(**xt_strong, return_dict=True)
            ce_u = F.cross_entropy(out_u.logits, y_hat, reduction="none")  # [B]

            if mask.any():
                # confidence weight (optional)
                w = conf.detach()
                if cfg.fm_conf_weight:
                    w = (w - w.min()) / (w.max() - w.min() + 1e-6)
                    w = 0.5 + 0.5 * w
                else:
                    w = torch.ones_like(w)
                loss_unsup = (ce_u * w * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
                keep_rate = float(mask.float().mean().item())
            else:
                loss_unsup = torch.tensor(0.0, device=device)
                keep_rate = 0.0

            # ramp-up unsup weight
            if cfg.fm_warmup_steps > 0:
                ramp = min(1.0, global_step / float(cfg.fm_warmup_steps))
            else:
                ramp = 1.0
            lambda_u = cfg.fm_lambda_u_max * ramp

            loss = loss_sup + lambda_u * loss_unsup

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            opt.step()

            # EMA update teacher
            with torch.no_grad():
                _ema_update(teacher, student, decay=cfg.fm_ema_decay)

            sup_losses.append(float(loss_sup.detach().cpu().item()))
            unsup_losses.append(float(loss_unsup.detach().cpu().item()))
            keep_rates.append(float(keep_rate))

            if (global_step % cfg.fm_log_every) == 0:
                print(f"  [FixMatch] ep={ep+1}/{cfg.fm_epochs} step={global_step} "
                      f"sup={sup_losses[-1]:.4f} unsup={unsup_losses[-1]:.4f} keep={keep_rate:.3f} lambda_u={lambda_u:.3f}")
            global_step += 1

        # epoch end: evaluate on source val to keep it safe
        src_val_acc = eval_split(student, tokenizer, src_val_texts, src_val_labels, cfg, device, K=K)["acc"]
        log["epochs"].append({
            "epoch": ep+1,
            "sup_loss": float(np.mean(sup_losses)) if sup_losses else 0.0,
            "unsup_loss": float(np.mean(unsup_losses)) if unsup_losses else 0.0,
            "keep_rate": float(np.mean(keep_rates)) if keep_rates else 0.0,
            "src_val_acc": float(src_val_acc),
        })
        print(f"  [FixMatch] epoch_end src_val_acc={src_val_acc:.4f}")

        if src_val_acc > best_src_val + 1e-6:
            best_src_val = float(src_val_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        # hard safety stop (avoid destructive adaptation)
        if (cfg.fm_hard_stop_drop > 0) and (best_src_val >= 0) and (src_val_acc < best_src_val - cfg.fm_hard_stop_drop):
            print(f"  [FixMatch] hard-stop: src_val dropped by >{cfg.fm_hard_stop_drop:.3f}")
            break

    if best_state is not None:
        student.load_state_dict(best_state, strict=True)

    # Return teacher-smoothed model for evaluation (often slightly better)
    # Here we keep the student weights (already best-by-src-val), but you can switch to teacher if desired.
    del teacher
    torch.cuda.empty_cache()
    return student, log


# =============================================================================
# Selector metrics
# =============================================================================

@torch.no_grad()
def compute_infomax_proxy(model: nn.Module, tokenizer, texts: List[str], cfg, device: torch.device) -> Dict[str, float]:
    """
    Mutual information proxy:
      proxy = H(mean p(y|x)) - mean H(p(y|x))
    """
    probs = predict_proba(model, tokenizer, texts, cfg, device)
    if probs.shape[0] == 0:
        return {"proxy": 0.0, "mean_ent": 0.0, "div_ent": 0.0, "pmax_mean": 0.0, "pmax_std": 0.0}

    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    ent_each = -(p * np.log(p)).sum(axis=1)
    mean_ent = float(ent_each.mean())

    p_mean = p.mean(axis=0)
    div_ent = float(-(p_mean * np.log(p_mean)).sum())

    proxy = float(div_ent - mean_ent)
    pmax = probs.max(axis=1)
    return {
        "proxy": proxy,
        "mean_ent": mean_ent,
        "div_ent": div_ent,
        "pmax_mean": float(pmax.mean()),
        "pmax_std": float(pmax.std()),
    }


@torch.no_grad()
def encode_cls_embeddings(model: nn.Module, tokenizer, texts: List[str], cfg, device: torch.device, max_samples: Optional[int] = None) -> np.ndarray:
    """
    Extract L2-normalized CLS embeddings.
    max_samples: optionally subsample for speed (deterministic: first N).
    """
    if max_samples is not None and len(texts) > max_samples:
        texts = texts[:max_samples]

    model.eval()
    ds = TextLabelDataset(texts, labels=None)
    loader = DataLoader(
        ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(tokenizer, cfg.max_len, with_labels=False),
    )
    base = get_base_encoder(model)
    embs = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = base(**inputs, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, dim=-1)
        embs.append(cls.detach().cpu().numpy().astype(np.float32))
    if not embs:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(embs, axis=0)


def reverse_validation_accuracy(
    model: nn.Module,
    tokenizer,
    tgt_train_texts: List[str],
    src_val_texts: List[str],
    src_val_labels: List[int],
    cfg,
    device: torch.device,
) -> Dict[str, Any]:
    """
    RVA:
      - pseudo-label target-train with candidate
      - train a linear classifier on frozen target embeddings (pseudo labels)
      - evaluate this classifier on source-val
    """
    probs = predict_proba(model, tokenizer, tgt_train_texts, cfg, device)
    if probs.shape[0] == 0:
        return {"rva": 0.0, "n": 0, "margin_mean": 0.0}

    y_hat = probs.argmax(axis=1).astype(np.int64)
    conf = probs.max(axis=1).astype(np.float32)

    if probs.shape[1] >= 2:
        sp = np.sort(probs, axis=1)
        margin = (sp[:, -1] - sp[:, -2]).astype(np.float32)
    else:
        margin = conf
    margin_mean = float(margin.mean())

    X_tgt = encode_cls_embeddings(model, tokenizer, tgt_train_texts, cfg, device, max_samples=cfg.rva_max_tgt_samples)
    X_src = encode_cls_embeddings(model, tokenizer, src_val_texts, cfg, device, max_samples=None)

    if X_tgt.shape[0] == 0 or X_src.shape[0] == 0:
        return {"rva": 0.0, "n": int(X_tgt.shape[0]), "margin_mean": margin_mean}

    H = X_tgt.shape[1]
    K = int(cfg.num_labels)
    lin = nn.Linear(H, K)
    lin.train()
    opt = torch.optim.AdamW(lin.parameters(), lr=cfg.reverse_lr, weight_decay=cfg.reverse_weight_decay)

    X_t = torch.tensor(X_tgt, dtype=torch.float32)
    y_t = torch.tensor(y_hat[:X_tgt.shape[0]], dtype=torch.long)
    w_t = torch.tensor(conf[:X_tgt.shape[0]], dtype=torch.float32)

    w_t = (w_t - w_t.min()) / (w_t.max() - w_t.min() + 1e-6)
    w_t = 0.5 + 0.5 * w_t

    bs = cfg.reverse_batch_size
    idx = np.arange(X_t.shape[0])

    for ep in range(cfg.reverse_epochs):
        np.random.shuffle(idx)
        for st in range(0, len(idx), bs):
            j = idx[st:st+bs]
            xb = X_t[j]
            yb = y_t[j]
            wb = w_t[j]
            logits = lin(xb)
            ce = F.cross_entropy(logits, yb, reduction="none")
            loss = (ce * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    lin.eval()
    with torch.no_grad():
        Xs = torch.tensor(X_src, dtype=torch.float32)
        logits = lin(Xs)
        pred = logits.argmax(dim=-1).cpu().numpy().astype(int).tolist()
    rva = accuracy(src_val_labels, pred)

    return {"rva": float(rva), "n": int(X_tgt.shape[0]), "margin_mean": margin_mean}


def compute_class_balance_metrics(
    probs: np.ndarray,
    p_src: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Given predicted probs on target (N,K) compute:
      - predicted label distribution q
      - min fraction
      - KL(q || p_src)
    """
    if probs.shape[0] == 0:
        return {"min_frac": 0.0, "kl_to_src": 1e9}

    y = probs.argmax(axis=1)
    K = probs.shape[1]
    counts = np.bincount(y.astype(np.int64), minlength=K).astype(np.float64)
    q = counts / max(1.0, float(counts.sum()))
    min_frac = float(q.min())
    q_ = np.clip(q, eps, 1.0)
    p_ = np.clip(p_src, eps, 1.0)
    kl = float(np.sum(q_ * np.log(q_ / p_)))
    return {"min_frac": min_frac, "kl_to_src": kl}


def compute_cluster_separation(embs: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple 2-class/ K-class separation proxy:
      sep = mean inter-centroid distance / (mean intra-class spread + eps)
    """
    if embs.shape[0] == 0:
        return 0.0
    K = int(np.max(y_pred)) + 1 if y_pred.size > 0 else 0
    if K <= 1:
        return 0.0

    centroids = []
    intra = []
    for c in range(K):
        idx = np.where(y_pred == c)[0]
        if idx.size < 2:
            continue
        Xc = embs[idx]
        mu = Xc.mean(axis=0, keepdims=True)
        centroids.append(mu)
        intra.append(float(np.mean(np.linalg.norm(Xc - mu, axis=1))))
    if len(centroids) < 2:
        return 0.0
    centroids = np.concatenate(centroids, axis=0)
    # average pairwise centroid distance
    dists = []
    for i in range(centroids.shape[0]):
        for j in range(i+1, centroids.shape[0]):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    inter = float(np.mean(dists)) if dists else 0.0
    intra_m = float(np.mean(intra)) if intra else 1e9
    return float(inter / (intra_m + 1e-6))


def compute_prototype_agreement(
    model: nn.Module,
    tokenizer,
    src_train_texts: List[str], src_train_labels: List[int],
    tgt_val_texts: List[str],
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    """
    Build class prototypes from (a subset of) source-train embeddings, then label target-val by nearest prototype.
    Return agreement between candidate predictions and prototype labels on "confident" prototype assignments.
    """
    K = cfg.num_labels
    # subsample source per class for speed
    per_c = int(cfg.proto_per_class)
    src_idx = []
    for c in range(K):
        ids = [i for i, y in enumerate(src_train_labels) if y == c]
        if not ids:
            continue
        src_idx.extend(ids[:per_c] if len(ids) > per_c else ids)

    src_texts_sub = [src_train_texts[i] for i in src_idx]
    src_labels_sub = [src_train_labels[i] for i in src_idx]

    Xs = encode_cls_embeddings(model, tokenizer, src_texts_sub, cfg, device, max_samples=None)
    if Xs.shape[0] == 0:
        return {"proto_agree": 0.0, "proto_cov": 0.0}

    protos = []
    for c in range(K):
        idx = np.where(np.array(src_labels_sub) == c)[0]
        if idx.size == 0:
            protos.append(np.zeros((Xs.shape[1],), dtype=np.float32))
        else:
            mu = Xs[idx].mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-6)
            protos.append(mu.astype(np.float32))
    P = np.stack(protos, axis=0)  # [K,H]

    Xt = encode_cls_embeddings(model, tokenizer, tgt_val_texts, cfg, device, max_samples=cfg.proto_max_tgt_val_samples)
    if Xt.shape[0] == 0:
        return {"proto_agree": 0.0, "proto_cov": 0.0}

    sims = Xt @ P.T  # cosine since normalized
    top = sims.max(axis=1)
    y_proto = sims.argmax(axis=1)

    # prototype margin
    if K >= 2:
        sp = np.sort(sims, axis=1)
        margin = sp[:, -1] - sp[:, -2]
    else:
        margin = top

    keep = margin >= float(cfg.proto_margin)
    cov = float(np.mean(keep.astype(np.float32)))

    if keep.sum() == 0:
        return {"proto_agree": 0.0, "proto_cov": cov}

    # candidate predictions
    probs = predict_proba(model, tokenizer, tgt_val_texts[:Xt.shape[0]], cfg, device)
    y_pred = probs.argmax(axis=1)
    agree = float(np.mean((y_pred[keep] == y_proto[keep]).astype(np.float32)))
    return {"proto_agree": agree, "proto_cov": cov}


@torch.no_grad()
def dropout_consistency_kl(model: nn.Module, tokenizer, texts: List[str], cfg, device: torch.device) -> float:
    """
    KL(p1 || p2) averaged over target-val between two stochastic dropout passes.
    Lower is usually better (more stable).
    """
    if len(texts) == 0:
        return 0.0

    def _predict_with_dropout() -> np.ndarray:
        model.train()  # enable dropout
        ds = TextLabelDataset(texts, labels=None)
        loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers,
                            collate_fn=make_collate_fn(tokenizer, cfg.max_len, with_labels=False))
        all_probs = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            out = model(**inputs, return_dict=True)
            p = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
            all_probs.append(p)
        if not all_probs:
            return np.zeros((0, cfg.num_labels), dtype=np.float32)
        return np.concatenate(all_probs, axis=0)

    p1 = _predict_with_dropout()
    p2 = _predict_with_dropout()
    model.eval()

    if p1.shape[0] == 0:
        return 0.0
    p1 = np.clip(p1.astype(np.float64), 1e-12, 1.0)
    p2 = np.clip(p2.astype(np.float64), 1e-12, 1.0)
    kl = np.sum(p1 * (np.log(p1) - np.log(p2)), axis=1)
    return float(np.mean(kl))


# =============================================================================
# Config + Data
# =============================================================================

@dataclass
class Config:
    # Model
    model_name: str = "bert-base-uncased"
    max_len: int = 128
    num_labels: int = 2  # overwritten after reading source labels

    # IO
    output_root: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_routeB_ultra"))
    log_name: str = "run_routeB_ultra.log"

    # Data paths (same style as fixed10)
    source_train_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "sourcedata", "source_train.csv"))
    source_val_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "sourcedata", "source_validation.csv"))
    source_test_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "sourcedata", "source_test.csv"))

    target_train_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "targetdata", "train.csv"))
    target_val_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "targetdata", "val.csv"))
    target_test_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "targetdata", "test.csv"))

    # Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])

    # Hardware
    num_workers: int = 2
    use_cuda_if_available: bool = True

    # Supervised source training
    batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    lr_adapt: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    src_epochs: int = 4
    early_stop_patience: int = 1

    # DAPT
    dapt_epochs: int = 1
    dapt_batch_size: int = 16
    dapt_lr: float = 5e-5
    mlm_prob: float = 0.15
    dapt_log_every: int = 100

    # Self-training candidates
    st_tau_base: float = 0.90
    st_tau_strict: float = 0.95
    st_rounds_base: int = 2
    st_rounds_sched: int = 3
    st_tau_schedule: List[float] = field(default_factory=lambda: [0.97, 0.95, 0.93])
    st_balanced: bool = True
    st_per_class_cap: Optional[int] = None
    st_use_da: bool = True

    # FixMatch Mean-Teacher candidate
    fm_epochs: int = 1
    fm_tau: float = 0.95
    fm_lambda_u_max: float = 1.0
    fm_warmup_steps: int = 200
    fm_ema_decay: float = 0.999
    fm_strong_mask_prob: float = 0.15
    fm_log_every: int = 100
    fm_use_da: bool = True
    da_momentum: float = 0.999
    fm_conf_weight: bool = True
    fm_hard_stop_drop: float = 0.08  # stop if src_val drops too much (relative to best seen during FM)

    # Selector safety constraints
    min_src_val_acc: float = 0.65
    hard_max_src_drop: float = 0.10
    src_val_tolerance_vs_anchor: float = 0.02

    min_balance_frac: float = 0.10  # collapse check
    max_balance_kl: float = 0.80

    proto_per_class: int = 256
    proto_margin: float = 0.10
    proto_max_tgt_val_samples: int = 753
    # -------------------------
    # Multi-dataset suite
    # -------------------------
    run_main_csv_task: bool = True
    run_acl_tasks: bool = True

    # ACL Amazon Reviews (processed_acl/<domain>/*.review)
    # Each domain folder should contain:
    #   - positive.review
    #   - negatiev.review (note: some releases misspell "negative" as "negatiev")
    #   - unlabeled.review
    acl_root_rel: str = "./processed_acl"
    acl_domains: List[str] = field(default_factory=lambda: ["books", "dvd", "electronics", "kitchen"])
    acl_mode: str = "one_source"  # "all_pairs" | "one_source" | "single"
    acl_source_domain: str = "books"
    acl_target_domains: List[str] = field(default_factory=lambda: ["dvd", "electronics", "kitchen"])
    acl_single_pair: Tuple[str, str] = ("books", "dvd")

    # Deterministic splits for ACL labeled source (train/val/test) and target unlabeled (train/val)
    acl_split_seed: int = 1337
    acl_source_split_fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    acl_tgt_val_size: int = 2000
    acl_max_unlabeled_train: Optional[int] = None

    # -------------------------
    # Reporting / analysis
    # -------------------------
    report_per_class: bool = True
    enable_unknown_reject: bool = False
    reject_tau: float = 0.5
    analyze_selector_correlation: bool = True
    min_proto_cov: float = 0.20
    min_proto_agree: float = 0.55

    # Reverse validation
    reverse_epochs: int = 15
    reverse_batch_size: int = 256
    reverse_lr: float = 1e-2
    reverse_weight_decay: float = 1e-4
    rva_max_tgt_samples: int = 5000

    # Rank aggregation selection
    rank_margin: float = 0.35  # must beat anchor by this margin to switch

    # Strong selector (Proto-Entropy Gate)
    use_strong_selector: bool = True
    fixmatch_proto_gate: float = 0.995  # if FixMatch prototype-agreement drops below this, it likely adapted usefully
    ent_target: float = 0.25            # desired predictive entropy region for stable self-training selection (binary task)
    st_score_w_ent: float = 1.0         # weight for |mean_ent - ent_target|
    st_score_w_dropout_kl: float = 0.5  # penalty weight for dropout instability
    st_score_w_proxy: float = 0.10      # reward weight for mutual-information proxy
    st_score_w_src_drop: float = 1.0    # penalty weight for source validation drop

    # Ensemble
    ensemble_k: int = 2  # 1 => disable, 2 or 3 often helps
    ensemble_temp: float = 1.0

# =============================================================================
# Task specs (multi-dataset evaluation)
# =============================================================================

@dataclass
class TaskSpec:
    name: str
    kind: str  # "csv" | "acl"
    src_domain: Optional[str] = None
    tgt_domain: Optional[str] = None


def build_tasks(cfg: Config) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []

    if cfg.run_main_csv_task:
        tasks.append(TaskSpec(name="csv_default", kind="csv"))

    if cfg.run_acl_tasks:
        domains = list(cfg.acl_domains)

        if cfg.acl_mode == "all_pairs":
            for s in domains:
                for t in domains:
                    if s == t:
                        continue
                    tasks.append(TaskSpec(name=f"acl_{s}_to_{t}", kind="acl", src_domain=s, tgt_domain=t))

        elif cfg.acl_mode == "one_source":
            s = cfg.acl_source_domain
            tgts = list(cfg.acl_target_domains) if cfg.acl_target_domains else [d for d in domains if d != s]
            for t in tgts:
                if t == s:
                    continue
                tasks.append(TaskSpec(name=f"acl_{s}_to_{t}", kind="acl", src_domain=s, tgt_domain=t))

        elif cfg.acl_mode == "single":
            s, t = cfg.acl_single_pair
            if s == t:
                raise ValueError("acl_single_pair must be (src!=tgt)")
            tasks.append(TaskSpec(name=f"acl_{s}_to_{t}", kind="acl", src_domain=s, tgt_domain=t))

        else:
            raise ValueError(f"Unknown acl_mode: {cfg.acl_mode}")

    return tasks

@dataclass
class DataBundle:
    src_train_texts: List[str]
    src_train_labels: List[int]
    src_val_texts: List[str]
    src_val_labels: List[int]
    src_test_texts: List[str]
    src_test_labels: List[int]

    tgt_train_texts: List[str]
    tgt_val_texts: List[str]
    tgt_test_texts: List[str]
    tgt_val_labels: Optional[List[int]] = None
    tgt_test_labels: Optional[List[int]] = None

    label2id: Dict[Any, int] = field(default_factory=dict)


def load_csv_split(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def map_source_labels(raw_labels: List[Any]) -> Tuple[List[int], Dict[Any, int]]:
    uniq = []
    for y in raw_labels:
        if y not in uniq:
            uniq.append(y)
    uniq_sorted = sorted(uniq)
    label2id = {y: i for i, y in enumerate(uniq_sorted)}
    mapped = [int(label2id[y]) for y in raw_labels]
    return mapped, label2id


def map_target_labels(raw_labels: List[Any], label2id: Dict[Any, int], unknown_id: int) -> List[int]:
    return [int(label2id.get(y, unknown_id)) for y in raw_labels]


def load_data(cfg: Config) -> Tuple[DataBundle, Dict[str, Any]]:
    src_tr = load_csv_split(cfg.source_train_path)
    src_va = load_csv_split(cfg.source_val_path)
    src_te = load_csv_split(cfg.source_test_path)

    tgt_tr = load_csv_split(cfg.target_train_path)
    tgt_va = load_csv_split(cfg.target_val_path)
    tgt_te = load_csv_split(cfg.target_test_path)

    src_text_col, src_label_col = guess_text_and_label_columns(src_tr)
    if src_label_col is None:
        raise ValueError("Cannot detect source label column.")
    print(f"[Data] source text_col='{src_text_col}', label_col='{src_label_col}'")

    tgt_text_col, tgt_label_col = guess_text_and_label_columns(tgt_tr)
    if tgt_label_col is None or tgt_label_col not in tgt_te.columns:
        _, tgt_label_col2 = guess_text_and_label_columns(tgt_te)
        tgt_label_col = tgt_label_col2
    print(f"[Data] target text_col='{tgt_text_col}' (label col guessed: {tgt_label_col})")

    # source
    src_train_texts = src_tr[src_text_col].astype(str).tolist()
    src_val_texts = src_va[src_text_col].astype(str).tolist()
    src_test_texts = src_te[src_text_col].astype(str).tolist()

    src_train_labels_raw = src_tr[src_label_col].tolist()
    src_val_labels_raw = src_va[src_label_col].tolist()
    src_test_labels_raw = src_te[src_label_col].tolist()

    src_train_labels, label2id = map_source_labels(src_train_labels_raw)
    src_val_labels = [label2id[y] for y in src_val_labels_raw]
    src_test_labels = [label2id[y] for y in src_test_labels_raw]

    # target
    tgt_train_texts = tgt_tr[tgt_text_col].astype(str).tolist()
    tgt_val_texts = tgt_va[tgt_text_col].astype(str).tolist()
    tgt_test_texts = tgt_te[tgt_text_col].astype(str).tolist()

    tgt_val_labels = None
    tgt_test_labels = None
    unknown_id = len(label2id)

    if tgt_label_col is not None and tgt_label_col in tgt_va.columns:
        tgt_val_labels = map_target_labels(tgt_va[tgt_label_col].tolist(), label2id, unknown_id=unknown_id)
    if tgt_label_col is not None and tgt_label_col in tgt_te.columns:
        tgt_test_labels = map_target_labels(tgt_te[tgt_label_col].tolist(), label2id, unknown_id=unknown_id)
        print("[Eval] target test label_col detected. (Unknown mapped to id=%d)" % unknown_id)

    report_raw = leakage_report(
        src_splits={"train": src_train_texts, "val": src_val_texts, "test": src_test_texts},
        tgt_splits={"train": tgt_train_texts, "val": tgt_val_texts, "test": tgt_test_texts},
    )

    # clean overlaps within each domain
    src_train_texts2, src_train_labels2, src_val_texts2, src_val_labels2, src_test_texts2, src_test_labels2, info_src = remove_overlaps_keep_test(
        src_train_texts, src_train_labels,
        src_val_texts, src_val_labels,
        src_test_texts, src_test_labels,
    )
    tgt_train_texts2, _, tgt_val_texts2, _, tgt_test_texts2, _, info_tgt = remove_overlaps_keep_test(
        tgt_train_texts, [0]*len(tgt_train_texts),
        tgt_val_texts, [0]*len(tgt_val_texts),
        tgt_test_texts, [0]*len(tgt_test_texts),
    )

    if tgt_val_labels is not None:
        set_test = set(normalize_text(x) for x in tgt_test_texts)
        keep_val = [i for i, t in enumerate([normalize_text(x) for x in tgt_val_texts]) if t not in set_test]
        tgt_val_labels2 = [tgt_val_labels[i] for i in keep_val]
    else:
        tgt_val_labels2 = None

    tgt_test_labels2 = tgt_test_labels if tgt_test_labels is not None else None

    print("\n" + "="*80)
    print("[Clean] Removing exact overlaps to prevent adaptation/test contamination:")
    print(f"  - removed_source_train: {info_src['removed_train']}")
    print(f"  - removed_source_val:   {info_src['removed_val']}")
    print(f"  - removed_target_train: {info_tgt['removed_train']}")
    print(f"  - removed_target_val:   {info_tgt['removed_val']}")
    print("  - note: Removed overlaps train↔(val/test) and val↔test by normalized exact match; kept test unchanged.")

    report_clean = leakage_report(
        src_splits={"train": src_train_texts2, "val": src_val_texts2, "test": src_test_texts2},
        tgt_splits={"train": tgt_train_texts2, "val": tgt_val_texts2, "test": tgt_test_texts2},
    )

    def _print_stats(prefix: str, rep: Dict[str, Any]):
        print("\n" + "="*80)
        print(prefix)
        for dom in ("source", "target"):
            for split in ("train", "val", "test"):
                st = rep[dom][split]
                print(f"  - {dom}_{split:5s}: n={st['n']:5d} unique={st['unique']:5d} dup={st['dup']:4d} dup_ratio={st['dup_ratio']:.4f}")

    _print_stats("[Check] Data leakage / duplicate sample sanity check (exact match after normalization)", report_raw)
    _print_stats("[Check] Data leakage / duplicate sample sanity check (after cleaning)", report_clean)

    data = DataBundle(
        src_train_texts=src_train_texts2, src_train_labels=src_train_labels2,
        src_val_texts=src_val_texts2, src_val_labels=src_val_labels2,
        src_test_texts=src_test_texts2, src_test_labels=src_test_labels2,
        tgt_train_texts=tgt_train_texts2,
        tgt_val_texts=tgt_val_texts2,
        tgt_test_texts=tgt_test_texts2,
        tgt_val_labels=tgt_val_labels2,
        tgt_test_labels=tgt_test_labels2,
        label2id=label2id,
    )

    meta = {
        "src_text_col": src_text_col,
        "src_label_col": src_label_col,
        "tgt_text_col": tgt_text_col,
        "tgt_label_col": tgt_label_col,
        "label2id": label2id,
        "unknown_id": unknown_id,
        "report_raw": report_raw,
        "report_clean": report_clean,
    }
    return data, meta


def _find_first_existing(dir_path: str, candidates: List[str]) -> str:
    for fname in candidates:
        p = os.path.join(dir_path, fname)
        if os.path.isfile(p):
            return p
    tried = ", ".join(candidates)
    raise FileNotFoundError(f"No file found in {dir_path}. Tried: {tried}")


def _read_review_file(path: str) -> List[str]:
    """Read a .review file (one sample per line). Be robust to simple label prefixes."""
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Common variants:
            #   1\ttext
            #   0\ttext
            #   __label__1\ttext
            #   text
            if "\t" in line:
                parts = line.split("\t")
                head = parts[0].strip()
                if head.isdigit() or head.startswith("__label__"):
                    line = "\t".join(parts[1:]).strip()

            if line:
                texts.append(line)
    return texts


def _stratified_split(
    texts: List[str],
    labels: List[int],
    fracs: Tuple[float, float, float],
    seed: int,
) -> Tuple[
    List[str], List[int],
    List[str], List[int],
    List[str], List[int],
]:
    """Stratified split into train/val/test with deterministic seed."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    a, b, c = fracs
    if abs((a + b + c) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1. Got: {fracs}")

    rng = random.Random(seed)

    idx_by: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        idx_by.setdefault(int(y), []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for y, idxs in idx_by.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * a)
        n_val = int(n * b)
        # remainder goes to test
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    def pick(idxs: List[int]) -> Tuple[List[str], List[int]]:
        return [texts[i] for i in idxs], [labels[i] for i in idxs]

    tr_t, tr_y = pick(train_idx)
    va_t, va_y = pick(val_idx)
    te_t, te_y = pick(test_idx)
    return tr_t, tr_y, va_t, va_y, te_t, te_y


def load_acl_task(cfg: Config, base_dir: str, src_domain: str, tgt_domain: str) -> Tuple[DataBundle, Dict[str, Any]]:
    """Build a sentiment DA task on ACL Amazon Reviews.

    Folder layout:
      processed_acl/<domain>/{positive.review, negatiev.review, unlabeled.review}

    Protocol used here (paper-friendly):
      - Source: labeled pos/neg split into train/val/test (stratified, deterministic).
      - Target: unlabeled.review split into train/val (deterministic), with leakage removal vs target test.
      - Target test: labeled pos/neg (all used as test).

    Notes:
      - We never use target labels for training or selection.
      - Exact-text leakage between target unlabeled and target labeled test is removed.
    """

    root = os.path.join(base_dir, cfg.acl_root_rel)
    src_dir = os.path.join(root, src_domain)
    tgt_dir = os.path.join(root, tgt_domain)

    neg_candidates = [
        "negative.review",
        "negatiev.review",  # misspelled in some releases
        "neg.review",
        "negative.txt",
        "neg.txt",
    ]
    pos_candidates = [
        "positive.review",
        "pos.review",
        "positive.txt",
        "pos.txt",
    ]
    unl_candidates = [
        "unlabeled.review",
        "unlabelled.review",
        "unlabeled.txt",
        "unlabelled.txt",
    ]

    src_neg_path = _find_first_existing(src_dir, neg_candidates)
    src_pos_path = _find_first_existing(src_dir, pos_candidates)

    tgt_neg_path = _find_first_existing(tgt_dir, neg_candidates)
    tgt_pos_path = _find_first_existing(tgt_dir, pos_candidates)
    tgt_unl_path = _find_first_existing(tgt_dir, unl_candidates)

    src_neg = _read_review_file(src_neg_path)
    src_pos = _read_review_file(src_pos_path)

    tgt_neg = _read_review_file(tgt_neg_path)
    tgt_pos = _read_review_file(tgt_pos_path)
    tgt_unl = _read_review_file(tgt_unl_path)

    # Label mapping for sentiment
    label2id = {"negative": 0, "positive": 1}

    # Build labeled source pool
    src_texts = src_neg + src_pos
    src_labels = [label2id["negative"]] * len(src_neg) + [label2id["positive"]] * len(src_pos)

    # Deterministic stratified split
    src_train_texts, src_train_labels, src_val_texts, src_val_labels, src_test_texts, src_test_labels = _stratified_split(
        src_texts, src_labels, cfg.acl_source_split_fracs, seed=cfg.acl_split_seed
    )

    # Target unlabeled split
    rng = random.Random(cfg.acl_split_seed)
    rng.shuffle(tgt_unl)
    tgt_val_texts = tgt_unl[: cfg.acl_tgt_val_size]
    tgt_train_texts = tgt_unl[cfg.acl_tgt_val_size :]

    if cfg.acl_max_unlabeled_train is not None:
        tgt_train_texts = tgt_train_texts[: cfg.acl_max_unlabeled_train]

    # Target labeled test
    tgt_test_texts = tgt_neg + tgt_pos
    tgt_test_labels = [label2id["negative"]] * len(tgt_neg) + [label2id["positive"]] * len(tgt_pos)

    # Leakage control (exact normalized overlaps)
    src_train_texts, src_train_labels, src_val_texts, src_val_labels, src_test_texts, src_test_labels, src_info = remove_overlaps_keep_test(
        src_train_texts, src_train_labels, src_val_texts, src_val_labels, src_test_texts, src_test_labels
    )

    dummy_train_labels = [0] * len(tgt_train_texts)
    dummy_val_labels = [0] * len(tgt_val_texts)

    tgt_train_texts, dummy_train_labels, tgt_val_texts, dummy_val_labels, tgt_test_texts, tgt_test_labels, tgt_info = remove_overlaps_keep_test(
        tgt_train_texts, dummy_train_labels, tgt_val_texts, dummy_val_labels, tgt_test_texts, tgt_test_labels
    )

    data = DataBundle(
        src_train_texts=src_train_texts,
        src_train_labels=src_train_labels,
        src_val_texts=src_val_texts,
        src_val_labels=src_val_labels,
        src_test_texts=src_test_texts,
        src_test_labels=src_test_labels,
        tgt_train_texts=tgt_train_texts,
        tgt_val_texts=tgt_val_texts,
        tgt_val_labels=None,
        tgt_test_texts=tgt_test_texts,
        tgt_test_labels=tgt_test_labels,
    )

    meta: Dict[str, Any] = {
        "label2id": label2id,
        "id2label": {v: k for k, v in label2id.items()},
        "src_domain": src_domain,
        "tgt_domain": tgt_domain,
        "paths": {
            "src_pos": src_pos_path,
            "src_neg": src_neg_path,
            "tgt_pos": tgt_pos_path,
            "tgt_neg": tgt_neg_path,
            "tgt_unl": tgt_unl_path,
        },
        "leakage_removed": {"source": src_info, "target": tgt_info},
    }

    print(f"\n[ACL Task] {src_domain} -> {tgt_domain}")
    print(f"  Source: train/val/test = {len(src_train_texts)}/{len(src_val_texts)}/{len(src_test_texts)}")
    print(f"  Target: train/val/test = {len(tgt_train_texts)}/{len(tgt_val_texts)}/{len(tgt_test_texts)}")
    leakage_report(data)

    return data, meta



# =============================================================================
# Candidate evaluation + selection
# =============================================================================

def _rank(values: Dict[str, float], higher_is_better: bool) -> Dict[str, float]:
    items = list(values.items())
    items.sort(key=lambda x: x[1], reverse=higher_is_better)
    ranks = {}
    # dense ranking (1,2,3...), ties get same rank
    rank = 1
    prev = None
    for k, v in items:
        if prev is None:
            ranks[k] = float(rank)
            prev = v
            continue
        if abs(v - prev) > 1e-12:
            rank += 1
            prev = v
        ranks[k] = float(rank)
    return ranks


def select_candidate_ultra(
    cand_metrics: Dict[str, Dict[str, Any]],
    cfg: Config,
    anchor_name: str = "st_hard",
) -> Dict[str, Any]:
    """Route-B++ candidate selector.

    This selector is designed to be **stable** and to avoid the two common failure modes
    we observed in long runs:

    1) **"Over-confident wrong" self-training**: extremely low predictive entropy can look
       great for many unsupervised scores (high MI proxy, large cluster separation), while
       target accuracy actually drops.

    2) **"FixMatch did nothing"**: when FixMatch stays almost perfectly aligned with the
       source prototypes (prototype agreement ~ 1.0), it is often just reproducing the
       source decision boundary and is not the best choice.

    Strategy (Proto-Entropy Gate):
      - If FixMatch is safe AND its prototype-agreement drops below a high threshold
        (cfg.fixmatch_proto_gate), we treat it as "effective adaptation" and pick it.
      - Otherwise, choose the safest self-training variant whose mean predictive entropy
        is closest to cfg.ent_target, penalizing dropout instability and source-val drop.
      - If nothing is safe, fall back to the best-ranked candidate.

    Note: We still compute rank aggregation (avg_rank) for logging/ensembling.
    """

    # ---------------------------
    # 1) Safety filtering
    # ---------------------------
    safe: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}

    for name, m in cand_metrics.items():
        ok = True
        why = []

        src_val = float(m.get("src_val_acc", 0.0))
        src_drop = float(m.get("src_drop", 0.0))
        min_frac = float(m.get("min_frac", 1.0))
        kl_to_src = float(m.get("kl_to_src", 0.0))
        proto_agree = float(m.get("proto_agree", 0.0))
        proto_cov = float(m.get("proto_cov", 0.0))

        if src_val < cfg.min_src_val_acc:
            ok = False
            why.append(f"src_val<{cfg.min_src_val_acc:.3f}")
        if src_drop > cfg.hard_max_src_drop:
            ok = False
            why.append(f"src_drop>{cfg.hard_max_src_drop:.3f}")
        if min_frac < cfg.min_balance_frac:
            ok = False
            why.append(f"min_frac<{cfg.min_balance_frac:.3f}")
        if kl_to_src > cfg.max_balance_kl:
            ok = False
            why.append(f"kl_to_src>{cfg.max_balance_kl:.3f}")
        if proto_cov < cfg.min_proto_cov:
            ok = False
            why.append(f"proto_cov<{cfg.min_proto_cov:.3f}")
        if proto_agree < cfg.min_proto_agree:
            ok = False
            why.append(f"proto_agree<{cfg.min_proto_agree:.3f}")

        safe[name] = ok
        reasons[name] = ";".join(why) if why else "ok"

    # ---------------------------
    # 2) Rank aggregation (for logging/optional ensemble)
    # ---------------------------
    names = list(cand_metrics.keys())
    rank_keys_hi = ["proxy", "rva", "cluster_sep", "proto_cov", "src_val_acc"]
    rank_keys_lo = ["src_drop", "dropout_kl", "kl_to_src"]

    ranks: Dict[str, List[int]] = {n: [] for n in names}

    def dense_rank(vals: Dict[str, float], reverse: bool) -> Dict[str, int]:
        items = sorted(vals.items(), key=lambda x: x[1], reverse=reverse)
        out: Dict[str, int] = {}
        r = 1
        prev_v = None
        for i, (k, v) in enumerate(items):
            if prev_v is not None and v != prev_v:
                r = i + 1
            out[k] = r
            prev_v = v
        return out

    for k in rank_keys_hi:
        vals = {n: float(cand_metrics[n].get(k, -1e9)) for n in names}
        rk = dense_rank(vals, reverse=True)
        for n in names:
            ranks[n].append(rk[n])

    for k in rank_keys_lo:
        vals = {n: float(cand_metrics[n].get(k, 1e9)) for n in names}
        rk = dense_rank(vals, reverse=False)
        for n in names:
            ranks[n].append(rk[n])

    avg_rank = {n: float(sum(ranks[n])) / float(len(ranks[n]) or 1) for n in names}
    ranked = sorted(names, key=lambda n: avg_rank.get(n, 1e9))

    safe_names = [n for n in ranked if safe.get(n, False)]
    if not safe_names:
        # nothing passed safety → choose best rank anyway
        chosen = ranked[0] if ranked else "none"
        return {
            "chosen": chosen,
            "decision": "no_safe_fallback_best_rank",
            "ranked": ranked,
            "avg_rank": avg_rank,
            "safe": safe,
            "reasons": reasons,
            "best_name": chosen,
            "anchor": anchor_name if anchor_name in cand_metrics else None,
            "strong": {"enabled": bool(cfg.use_strong_selector)},
        }

    best_name = safe_names[0]

    # ---------------------------
    # 3) Strong selector (Proto-Entropy Gate)
    # ---------------------------
    chosen = None
    decision = ""
    strong_info: Dict[str, Any] = {"enabled": bool(cfg.use_strong_selector)}

    if cfg.use_strong_selector:
        # (A) FixMatch gate: select FixMatch only if it *actually* moved (proto_agree drops)
        fx = "fixmatch_mt"
        if fx in safe_names:
            fx_pa = float(cand_metrics[fx].get("proto_agree", 1.0))
            strong_info["fixmatch_proto_agree"] = fx_pa
            strong_info["fixmatch_proto_gate"] = float(cfg.fixmatch_proto_gate)
            if fx_pa < float(cfg.fixmatch_proto_gate):
                chosen = fx
                decision = "gate_fixmatch_proto"

        # (B) Otherwise, among self-training methods choose the most "well-calibrated" one
        if chosen is None:
            st_candidates = [n for n in safe_names if n.startswith("st_")]
            st_scores: Dict[str, float] = {}
            for n in st_candidates:
                m = cand_metrics[n]
                ent = float(m.get("mean_ent", 0.5))
                score = (
                    -float(cfg.st_score_w_ent) * abs(ent - float(cfg.ent_target))
                    -float(cfg.st_score_w_dropout_kl) * float(m.get("dropout_kl", 0.0))
                    +float(cfg.st_score_w_proxy) * float(m.get("proxy", 0.0))
                    -float(cfg.st_score_w_src_drop) * float(m.get("src_drop", 0.0))
                )
                st_scores[n] = float(score)

            strong_info["st_scores"] = st_scores

            if st_scores:
                chosen = max(st_scores, key=st_scores.get)
                decision = "gate_selftrain_entropy"

    # ---------------------------
    # 4) Conservative fallback (anchor policy)
    # ---------------------------
    if chosen is None:
        anchor = anchor_name if anchor_name in cand_metrics else None
        if anchor is None or anchor not in safe_names:
            chosen = best_name
            decision = "anchor_unsafe_choose_best_safe"
        else:
            # keep anchor unless a safe model beats it by rank margin AND doesn't hurt src_val too much
            anchor_rank = avg_rank.get(anchor, 1e9)
            best_rank = avg_rank.get(best_name, 1e9)
            rank_improve = anchor_rank - best_rank
            src_ok = float(cand_metrics[best_name].get("src_val_acc", 0.0)) >= float(
                cand_metrics[anchor].get("src_val_acc", 0.0)
            ) - float(cfg.src_val_tolerance_vs_anchor)

            if best_name != anchor and rank_improve >= float(cfg.rank_margin) and src_ok:
                chosen = best_name
                decision = "switch_from_anchor"
            else:
                chosen = anchor
                decision = "keep_anchor"

    # Final sanity: if chosen not safe, drop to best safe
    if not safe.get(chosen, False):
        chosen = best_name
        decision = decision + "|fallback_best_safe"

    return {
        "chosen": chosen,
        "decision": decision,
        "ranked": ranked,
        "avg_rank": avg_rank,
        "safe": safe,
        "reasons": reasons,
        "best_name": best_name,
        "anchor": anchor_name if anchor_name in cand_metrics else None,
        "strong": strong_info,
    }
def ensemble_predict_proba(
    model_states: Dict[str, Dict[str, torch.Tensor]],
    weights: Dict[str, float],
    model_name: str,
    dapt_encoder_dir: str,
    tokenizer,
    texts: List[str],
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    """
    Weighted probability ensemble across multiple candidates.
    Loads candidates sequentially (GPU memory friendly).
    """
    names = [n for n, w in weights.items() if w > 0]
    if not names:
        return np.zeros((len(texts), cfg.num_labels), dtype=np.float32)

    probs_sum = None
    total_w = 0.0
    for n in names:
        w = float(weights[n])
        m = build_classifier(model_name, cfg.num_labels, device, dapt_encoder_dir=dapt_encoder_dir)
        m.load_state_dict(model_states[n], strict=True)
        p = predict_proba(m, tokenizer, texts, cfg, device)
        if probs_sum is None:
            probs_sum = w * p
        else:
            probs_sum += w * p
        total_w += w
        del m
        torch.cuda.empty_cache()

    if probs_sum is None:
        return np.zeros((len(texts), cfg.num_labels), dtype=np.float32)
    probs_sum = probs_sum / max(1e-12, total_w)
    return probs_sum.astype(np.float32)


# =============================================================================
# Seed runner
# =============================================================================

def run_seed(seed: int, cfg: Config, data: DataBundle, seed_dir: str, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(seed)
    ensure_dir(seed_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    K = cfg.num_labels

    # -------------------------------------------------------------------------
    # Baseline: source_only (no DAPT)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print(f"[Seed {seed}] Baseline: source_only (no DAPT)")
    m_src = build_classifier(cfg.model_name, K, device, dapt_encoder_dir=None)
    m_src, _ = train_supervised_source(
        m_src, tokenizer,
        data.src_train_texts, data.src_train_labels,
        data.src_val_texts, data.src_val_labels,
        cfg, device,
    )
    src_test = eval_split(m_src, tokenizer, data.src_test_texts, data.src_test_labels, cfg, device, K=K)
    res_source_only = {
        "method": "source_only",
        "seed": seed,
        "src_test_acc": src_test["acc"],
        "src_acc_known": src_test.get("acc_known"),
        "src_unknown_rate": src_test.get("unknown_rate"),
        "src_pred_unknown_rate": src_test.get("pred_unknown_rate"),
        "src_per_class": src_test.get("per_class"),
        "src_confusion": src_test.get("confusion"),
    }
    if data.tgt_test_labels is not None:
        tgt_test = eval_split(m_src, tokenizer, data.tgt_test_texts, data.tgt_test_labels, cfg, device, K=K)
        res_source_only.update({"tgt_acc": tgt_test["acc"], "tgt_f1": tgt_test["f1"], "tgt_bal_acc": tgt_test["bal_acc"], "tgt_acc_known": tgt_test.get("acc_known"), "tgt_unknown_rate": tgt_test.get("unknown_rate"), "tgt_pred_unknown_rate": tgt_test.get("pred_unknown_rate"), "tgt_per_class": tgt_test.get("per_class"), "tgt_confusion": tgt_test.get("confusion")})
    del m_src
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # DAPT encoder
    # -------------------------------------------------------------------------
    dapt_encoder_dir = run_dapt_mlm(data.tgt_train_texts, tokenizer, cfg, device, save_dir=seed_dir)

    # -------------------------------------------------------------------------
    # Baseline: dapt_source (supervised on source)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print(f"[Seed {seed}] Baseline: dapt_source")
    m_base = build_classifier(cfg.model_name, K, device, dapt_encoder_dir=dapt_encoder_dir)
    m_base, _ = train_supervised_source(
        m_base, tokenizer,
        data.src_train_texts, data.src_train_labels,
        data.src_val_texts, data.src_val_labels,
        cfg, device,
    )
    base_src_val_rep = eval_split(m_base, tokenizer, data.src_val_texts, data.src_val_labels, cfg, device, K=K)
    base_src_val = base_src_val_rep["acc"]
    base_src_test_rep = eval_split(m_base, tokenizer, data.src_test_texts, data.src_test_labels, cfg, device, K=K)
    base_src_test = base_src_test_rep["acc"]

    res_dapt_source = {
        "method": "dapt_source",
        "seed": seed,
        "src_test_acc": base_src_test,
        "src_acc_known": base_src_test_rep.get("acc_known"),
        "src_unknown_rate": base_src_test_rep.get("unknown_rate"),
        "src_pred_unknown_rate": base_src_test_rep.get("pred_unknown_rate"),
        "src_per_class": base_src_test_rep.get("per_class"),
        "src_confusion": base_src_test_rep.get("confusion"),
    }
    if data.tgt_test_labels is not None:
        tt = eval_split(m_base, tokenizer, data.tgt_test_texts, data.tgt_test_labels, cfg, device, K=K)
        res_dapt_source.update({"tgt_acc": tt["acc"], "tgt_f1": tt["f1"], "tgt_bal_acc": tt["bal_acc"], "tgt_acc_known": tt.get("acc_known"), "tgt_unknown_rate": tt.get("unknown_rate"), "tgt_pred_unknown_rate": tt.get("pred_unknown_rate"), "tgt_per_class": tt.get("per_class"), "tgt_confusion": tt.get("confusion")})

    base_state_cpu = {k: v.detach().cpu().clone() for k, v in m_base.state_dict().items()}

    # source prior for balance metrics
    src_counts = np.bincount(np.array(data.src_train_labels, dtype=np.int64), minlength=K).astype(np.float64)
    p_src = src_counts / max(1.0, float(src_counts.sum()))
    p_src = np.clip(p_src, 1e-6, 1.0)
    p_src = p_src / p_src.sum()

    # -------------------------------------------------------------------------
    # Candidate pool
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print(f"[Seed {seed}] Route-B++: build candidates + robust unsupervised selection")

    candidates = [
        {"name": "none", "kind": "none"},
        {"name": "st_hard", "kind": "selftrain", "tau_schedule": [cfg.st_tau_base] * cfg.st_rounds_base, "balanced": False},
        {"name": "st_balanced", "kind": "selftrain", "tau_schedule": [cfg.st_tau_base] * cfg.st_rounds_base, "balanced": True, "per_class_cap": cfg.st_per_class_cap},
        {"name": "st_sched", "kind": "selftrain", "tau_schedule": cfg.st_tau_schedule[:cfg.st_rounds_sched], "balanced": False},
        {"name": "fixmatch_mt", "kind": "fixmatch"},
    ]
    anchor = "st_hard"

    cand_logs: List[Dict[str, Any]] = []
    cand_metrics: Dict[str, Dict[str, Any]] = {}
    cand_states: Dict[str, Dict[str, torch.Tensor]] = {}
    method_results: List[Dict[str, Any]] = []

    for cand in candidates:
        name = cand["name"]
        kind = cand["kind"]
        print("\n" + "-"*60)
        print(f"  [Candidate] {name} ({kind})")

        m = build_classifier(cfg.model_name, K, device, dapt_encoder_dir=dapt_encoder_dir)
        m.load_state_dict(base_state_cpu, strict=True)

        adapt_log: Dict[str, Any] = {"name": name, "kind": kind}

        try:
            if kind == "none":
                pass
            elif kind == "selftrain":
                tau_schedule = cand.get("tau_schedule", [cfg.st_tau_base] * cfg.st_rounds_base)
                balanced = bool(cand.get("balanced", False))
                per_class_cap = cand.get("per_class_cap", None)
                m, st_log = adapt_selftrain_closed(
                    m, tokenizer,
                    data.src_train_texts, data.src_train_labels,
                    data.src_val_texts, data.src_val_labels,
                    data.tgt_train_texts,
                    cfg, device,
                    tau_schedule=tau_schedule,
                    balanced=balanced,
                    per_class_cap=per_class_cap,
                    use_distribution_alignment=cfg.st_use_da,
                )
                adapt_log["selftrain"] = st_log
            elif kind == "fixmatch":
                m, fm_log = adapt_fixmatch_mean_teacher(
                    m, tokenizer,
                    data.src_train_texts, data.src_train_labels,
                    data.src_val_texts, data.src_val_labels,
                    data.tgt_train_texts,
                    cfg, device,
                )
                adapt_log["fixmatch"] = fm_log
            else:
                raise ValueError(f"Unknown candidate kind: {kind}")

            # evaluate src_val for safety + report src_test
            src_val_acc = eval_split(m, tokenizer, data.src_val_texts, data.src_val_labels, cfg, device, K=K)["acc"]
            src_test_rep = eval_split(m, tokenizer, data.src_test_texts, data.src_test_labels, cfg, device, K=K)
            src_test_acc = src_test_rep["acc"]
            src_drop = max(0.0, float(base_src_val) - float(src_val_acc))

            # target probs on val (unlabeled)
            tgt_val_probs = predict_proba(m, tokenizer, data.tgt_val_texts, cfg, device)
            balm = compute_class_balance_metrics(tgt_val_probs, p_src=p_src)

            # proxies
            proxy = compute_infomax_proxy(m, tokenizer, data.tgt_val_texts, cfg, device)
            rva = reverse_validation_accuracy(m, tokenizer, data.tgt_train_texts, data.src_val_texts, data.src_val_labels, cfg, device)
            proto = compute_prototype_agreement(m, tokenizer, data.src_train_texts, data.src_train_labels, data.tgt_val_texts, cfg, device)
            # cluster sep on target val embeddings
            Xt = encode_cls_embeddings(m, tokenizer, data.tgt_val_texts, cfg, device, max_samples=cfg.proto_max_tgt_val_samples)
            ypv = tgt_val_probs.argmax(axis=1) if tgt_val_probs.shape[0] > 0 else np.zeros((Xt.shape[0],), dtype=np.int64)
            cluster_sep = compute_cluster_separation(Xt, ypv)
            dkl = dropout_consistency_kl(m, tokenizer, data.tgt_val_texts, cfg, device)

            metrics = {
                "name": name,
                "kind": kind,
                "src_val_acc": float(src_val_acc),
                "src_test_acc": float(src_test_acc),
                "src_drop": float(src_drop),
                **proxy,
                **rva,
                **proto,
                "cluster_sep": float(cluster_sep),
                "dropout_kl": float(dkl),
                **balm,
                "base_src_val_acc": float(base_src_val),
            }
            cand_metrics[name] = metrics
            cand_states[name] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

            # per-candidate reporting on target test (if labels exist)
            res = {"method": name, "seed": seed, "src_test_acc": float(src_test_acc)}
            if data.tgt_test_labels is not None:
                tt = eval_split(m, tokenizer, data.tgt_test_texts, data.tgt_test_labels, cfg, device, K=K)
                res.update({"tgt_acc": tt["acc"], "tgt_f1": tt["f1"], "tgt_bal_acc": tt["bal_acc"], "tgt_acc_known": tt.get("acc_known"), "tgt_unknown_rate": tt.get("unknown_rate"), "tgt_pred_unknown_rate": tt.get("pred_unknown_rate"), "tgt_per_class": tt.get("per_class"), "tgt_confusion": tt.get("confusion")})
            method_results.append(res)

            cand_logs.append({"metrics": metrics, "adapt_log": adapt_log})

        except Exception as e:
            print(f"  [Candidate] {name} failed: {e}")
            traceback.print_exc()
            cand_logs.append({"name": name, "failed": True, "error": str(e), "traceback": traceback.format_exc()})
        finally:
            del m
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------
    sel = select_candidate_ultra(cand_metrics, anchor_name=anchor, cfg=cfg)
    chosen = sel["chosen"]
    print(f"[Route-B++] Selection: chosen={chosen} decision={sel.get('decision')}")

    # evaluate selected
    m_sel = build_classifier(cfg.model_name, K, device, dapt_encoder_dir=dapt_encoder_dir)
    m_sel.load_state_dict(cand_states.get(chosen, base_state_cpu), strict=True)
    sel_src_test_rep = eval_split(m_sel, tokenizer, data.src_test_texts, data.src_test_labels, cfg, device, K=K)
    sel_src_test = sel_src_test_rep["acc"]

    res_select = {"method": "routeB_ultra", "seed": seed, "chosen": chosen, "src_test_acc": float(sel_src_test), "selector": sel}
    if data.tgt_test_labels is not None:
        tt = eval_split(m_sel, tokenizer, data.tgt_test_texts, data.tgt_test_labels, cfg, device, K=K)
        res_select.update({"tgt_acc": tt["acc"], "tgt_f1": tt["f1"], "tgt_bal_acc": tt["bal_acc"], "tgt_acc_known": tt.get("acc_known"), "tgt_unknown_rate": tt.get("unknown_rate"), "tgt_pred_unknown_rate": tt.get("pred_unknown_rate"), "tgt_per_class": tt.get("per_class"), "tgt_confusion": tt.get("confusion")})
    del m_sel
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Optional ensemble of top-k safe candidates (unsupervised weights)
    # -------------------------------------------------------------------------
    res_ens = None
    ens_info = {}
    if cfg.ensemble_k >= 2 and data.tgt_test_labels is not None:
        safe_flags = sel.get("safe", {})
        avg_rank = sel.get("avg_rank", {})
        safe_names = [n for n, ok in safe_flags.items() if ok and n in avg_rank]

        if safe_names:
            safe_names.sort(key=lambda n: avg_rank[n])
            top = safe_names[: cfg.ensemble_k]

            # soft weights from avg_rank
            temp = max(1e-6, float(cfg.ensemble_temp))
            ws = {n: math.exp(-avg_rank[n] / temp) for n in top}
            z = sum(ws.values())
            ws = {n: (w / z) for n, w in ws.items()}

            ens_probs = ensemble_predict_proba(
                cand_states,
                ws,
                cfg.model_name,
                dapt_encoder_dir=dapt_encoder_dir,
                tokenizer=tokenizer,
                texts=data.tgt_test_texts,
                cfg=cfg,
                device=device,
            )
            y_pred = ens_probs.argmax(axis=1).astype(int).tolist()
            y_true = data.tgt_test_labels
            ens_acc = accuracy(y_true, y_pred)
            ens_f1, ens_bal = macro_f1_balanced_acc(y_true, y_pred, K=K)

            ens_info = {"top": top, "weights": ws, "avg_rank": {n: avg_rank[n] for n in top}}
            res_ens = {"method": "routeB_ensemble", "seed": seed, "src_test_acc": float("nan"), "ensemble": ens_info,
                       "tgt_acc": float(ens_acc), "tgt_f1": float(ens_f1), "tgt_bal_acc": float(ens_bal)}

    # Save candidate logs
    safe_json_dump(cand_logs, os.path.join(seed_dir, "routeB_ultra_candidate_logs.json"))
    safe_json_dump(cand_metrics, os.path.join(seed_dir, "routeB_ultra_candidate_metrics.json"))
    safe_json_dump(sel, os.path.join(seed_dir, "routeB_ultra_selection.json"))

    # -------------------------------------------------------------------------
    # Pack results
    # -------------------------------------------------------------------------
    results = [res_source_only, res_dapt_source]
    # add per-candidate method results (optional, but useful for debugging)
    results.extend(method_results)
    # add selected and ensemble
    results.append(res_select)
    if res_ens is not None:
        results.append(res_ens)

    seed_meta = {
        "seed": seed,
        "base_src_val_acc": float(base_src_val),
        "chosen": chosen,
        "selection": sel,
        "ensemble": ens_info,
    }
    return results, seed_meta


def summarize(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    methods: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        methods.setdefault(r["method"], []).append(r)

    summary: Dict[str, Any] = {"methods": {}}
    for m, rs in methods.items():
        tgt_acc = [x.get("tgt_acc") for x in rs if "tgt_acc" in x and x.get("tgt_acc") is not None and not (isinstance(x.get("tgt_acc"), float) and math.isnan(x.get("tgt_acc")))]
        tgt_f1 = [x.get("tgt_f1") for x in rs if "tgt_f1" in x and x.get("tgt_f1") is not None and not (isinstance(x.get("tgt_f1"), float) and math.isnan(x.get("tgt_f1")))]
        tgt_bal = [x.get("tgt_bal_acc") for x in rs if "tgt_bal_acc" in x and x.get("tgt_bal_acc") is not None and not (isinstance(x.get("tgt_bal_acc"), float) and math.isnan(x.get("tgt_bal_acc")))]
        src_acc = [x.get("src_test_acc") for x in rs if "src_test_acc" in x and x.get("src_test_acc") is not None and not (isinstance(x.get("src_test_acc"), float) and math.isnan(x.get("src_test_acc")))]

        def mean_std(vals: List[float]) -> Tuple[float, float]:
            if len(vals) == 0:
                return (float("nan"), float("nan"))
            v = np.array(vals, dtype=np.float64)
            return float(v.mean()), float(v.std(ddof=0))

        out = {"n": len(rs)}
        out["tgt_acc_mean"], out["tgt_acc_std"] = mean_std([float(v) for v in tgt_acc])
        out["tgt_f1_mean"], out["tgt_f1_std"] = mean_std([float(v) for v in tgt_f1])
        out["tgt_bal_acc_mean"], out["tgt_bal_acc_std"] = mean_std([float(v) for v in tgt_bal])
        out["src_acc_mean"], out["src_acc_std"] = mean_std([float(v) for v in src_acc])
        summary["methods"][m] = out
    return summary


# =============================================================================
# Main
# =============================================================================

def _mean_std_ignore_nan(values: List[Any]) -> Tuple[float, float]:
    vals: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        vals.append(fv)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def summarize_all_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize multi-seed results per method, including per-class F1 when available."""
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        by_method.setdefault(str(r.get("method", "unknown")), []).append(r)

    summary: Dict[str, Any] = {}
    for method, rows in sorted(by_method.items(), key=lambda kv: kv[0]):
        src_mean, src_std = _mean_std_ignore_nan([x.get("src_test_acc") for x in rows])
        tgt_mean, tgt_std = _mean_std_ignore_nan([x.get("tgt_acc") for x in rows])
        f1_mean, f1_std = _mean_std_ignore_nan([x.get("tgt_f1") for x in rows])
        bal_mean, bal_std = _mean_std_ignore_nan([x.get("tgt_bal_acc") for x in rows])

        unk_mean, unk_std = _mean_std_ignore_nan([x.get("tgt_unknown_rate") for x in rows])
        p_unk_mean, p_unk_std = _mean_std_ignore_nan([x.get("tgt_pred_unknown_rate") for x in rows])

        method_sum: Dict[str, Any] = {
            "n": int(len(rows)),
            "src_test_acc_mean": src_mean,
            "src_test_acc_std": src_std,
            "tgt_acc_mean": tgt_mean,
            "tgt_acc_std": tgt_std,
            "tgt_f1_mean": f1_mean,
            "tgt_f1_std": f1_std,
            "tgt_bal_acc_mean": bal_mean,
            "tgt_bal_acc_std": bal_std,
            "tgt_unknown_rate_mean": unk_mean,
            "tgt_unknown_rate_std": unk_std,
            "tgt_pred_unknown_rate_mean": p_unk_mean,
            "tgt_pred_unknown_rate_std": p_unk_std,
        }

        # Per-class F1 (target) aggregation
        per_class_f1: Dict[str, List[float]] = {}
        for r in rows:
            pc = r.get("tgt_per_class")
            if not isinstance(pc, dict):
                continue
            for cid, md in pc.items():
                if not isinstance(md, dict):
                    continue
                f1 = md.get("f1")
                if f1 is None:
                    continue
                try:
                    per_class_f1.setdefault(str(cid), []).append(float(f1))
                except Exception:
                    pass

        if per_class_f1:
            method_sum["tgt_per_class_f1"] = {
                cid: {"f1_mean": _mean_std_ignore_nan(vs)[0], "f1_std": _mean_std_ignore_nan(vs)[1], "n": len(vs)}
                for cid, vs in sorted(per_class_f1.items(), key=lambda kv: int(kv[0]))
            }

        summary[method] = method_sum

    return summary


def compute_selector_metric_correlation(
    all_results: List[Dict[str, Any]],
    seed_metas: List[Dict[str, Any]],
    target_key: str = "tgt_acc",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation between selector metrics (unsupervised) and true target performance.

    We use:
      - selector metrics from seed_meta["candidates"][i]["metrics"]
      - target performance from all_results (same seed + method name)

    Output:
      - rows_df: per-(seed,candidate) table with both selector metrics and tgt_acc
      - corr_df: metric-level Pearson/Spearman correlations vs tgt_acc
    """

    perf: Dict[Tuple[int, str], float] = {}
    for r in all_results:
        seed = r.get("seed")
        method = r.get("method")
        if seed is None or method is None:
            continue
        v = r.get(target_key)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        perf[(int(seed), str(method))] = fv

    rows: List[Dict[str, Any]] = []
    for sm in seed_metas:
        seed = sm.get("seed")
        if seed is None:
            continue
        seed = int(seed)

        cands = sm.get("candidates", [])
        if not isinstance(cands, list):
            continue

        for c in cands:
            if not isinstance(c, dict):
                continue
            met = c.get("metrics", {})
            if not isinstance(met, dict):
                continue
            name = met.get("name")
            if name is None:
                continue

            key = (seed, str(name))
            if key not in perf:
                continue

            row: Dict[str, Any] = {
                "seed": seed,
                "candidate": str(name),
                "kind": str(met.get("kind", "")),
                target_key: float(perf[key]),
            }

            # Keep only scalar numeric metrics
            for k, v in met.items():
                if k in ["name", "kind"]:
                    continue
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    fv = float(v)
                    if not math.isnan(fv):
                        row[str(k)] = fv

            rows.append(row)

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        return rows_df, pd.DataFrame([])

    # Correlations per metric
    exclude = {"seed", "candidate", "kind", target_key}
    metric_cols = [c for c in rows_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(rows_df[c])]

    corr_rows: List[Dict[str, Any]] = []
    for m in metric_cols:
        sub = rows_df[[m, target_key]].dropna()
        if len(sub) < 3:
            continue
        pearson = float(sub[m].corr(sub[target_key], method="pearson"))
        spearman = float(sub[m].corr(sub[target_key], method="spearman"))
        corr_rows.append({"metric": m, "pearson": pearson, "spearman": spearman, "n": int(len(sub))})

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["abs_pearson"] = corr_df["pearson"].abs()
        corr_df = corr_df.sort_values("abs_pearson", ascending=False).drop(columns=["abs_pearson"])

    return rows_df, corr_df


def run_task(task: TaskSpec, cfg: Config, base_dir: str, device: torch.device) -> Dict[str, Any]:
    """Run one task (one dataset/domain pair) across cfg.seeds."""
    task_dir = ensure_dir(os.path.join(cfg.output_root, task.name))

    # Load data
    if task.kind == "csv":
        data, meta = load_data(cfg)
    elif task.kind == "acl":
        if not task.src_domain or not task.tgt_domain:
            raise ValueError("ACL task requires src_domain and tgt_domain")
        data, meta = load_acl_task(cfg, base_dir, task.src_domain, task.tgt_domain)
    else:
        raise ValueError(f"Unknown task kind: {task.kind}")

    # Update number of known labels for this task
    cfg.num_labels = int(len(meta.get("label2id", {})))

    # Save task meta for reproducibility
    with open(os.path.join(task_dir, "task_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    all_results: List[Dict[str, Any]] = []
    seed_metas: List[Dict[str, Any]] = []

    for seed in cfg.seeds:
        seed_dir = ensure_dir(os.path.join(task_dir, f"seed_{seed}"))
        print(f"\n[Task={task.name}] Seed={seed}  --> {seed_dir}")

        results, seed_meta = run_seed(seed, cfg, data, seed_dir, device)
        all_results.extend(results)
        seed_metas.append(seed_meta)

    # Persist raw results
    with open(os.path.join(task_dir, "all_seed_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(task_dir, "seed_metas.json"), "w", encoding="utf-8") as f:
        json.dump(seed_metas, f, indent=2, ensure_ascii=False)

    # Summaries
    summary = summarize_all_results(all_results)
    with open(os.path.join(task_dir, "multi_seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Selector correlation (only meaningful if target labels exist for evaluation)
    if cfg.analyze_selector_correlation and (data.tgt_test_labels is not None):
        rows_df, corr_df = compute_selector_metric_correlation(all_results, seed_metas, target_key="tgt_acc")
        rows_path = os.path.join(task_dir, "selector_metric_rows.csv")
        corr_path = os.path.join(task_dir, "selector_metric_correlation.csv")
        rows_df.to_csv(rows_path, index=False)
        corr_df.to_csv(corr_path, index=False)

        with open(os.path.join(task_dir, "selector_metric_correlation.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"rows": rows_df.to_dict(orient="records"), "correlation": corr_df.to_dict(orient="records")},
                f,
                indent=2,
                ensure_ascii=False,
            )

        if not corr_df.empty:
            print("\n[Selector correlation] Top metrics (by |Pearson|):")
            print(corr_df.head(10).to_string(index=False))

    # Free memory between tasks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    cfg = Config()

    # Project root (this script's directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Output root (one folder per task)
    cfg.output_root = os.path.join(base_dir, "outputs_routeB_ultra_multidatav2")
    ensure_dir(cfg.output_root)

    # Log (global)
    log_file = os.path.join(cfg.output_root, cfg.log_name)
    tee = FileTee(log_file)


    try:
        print("=" * 80)
        print("[RouteB-Ultra] Multi-dataset evaluation entrypoint")
        print(f"Output root: {cfg.output_root}")
        print(f"Log file: {log_file}")
        print("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] {device}")

        tasks = build_tasks(cfg)
        if not tasks:
            raise RuntimeError("No tasks to run. Check cfg.run_main_csv_task / cfg.run_acl_tasks.")

        print("\n[Tasks]")
        for t in tasks:
            if t.kind == "acl":
                print(f"  - {t.name}  (ACL: {t.src_domain} -> {t.tgt_domain})")
            else:
                print(f"  - {t.name}  (CSV)")

        all_task_summaries: Dict[str, Any] = {}

        for t in tasks:
            print("\n" + "#" * 80)
            if t.kind == "acl":
                print(f"[Run Task] {t.name}  (ACL: {t.src_domain} -> {t.tgt_domain})")
            else:
                print(f"[Run Task] {t.name}  (CSV)")
            print("#" * 80)

            task_summary = run_task(t, cfg, base_dir, device)
            all_task_summaries[t.name] = task_summary

        with open(os.path.join(cfg.output_root, "all_tasks_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_task_summaries, f, indent=2, ensure_ascii=False)

        print("\n[Done] All tasks finished.")
        print(f"See: {os.path.join(cfg.output_root, 'all_tasks_summary.json')}")

    finally:
        tee.close()


if __name__ == "__main__":
    main()
