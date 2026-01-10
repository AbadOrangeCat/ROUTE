#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAS-UniDA (Local-Run Version)
- 直接在脚本里配置本地文件路径
- 不使用控制台参数（无 argparse）
- 直接运行即可

支持的数据文件：
- CSV / TSV / JSONL / JSON（若装了 pandas）
- TXT（每行一个样本；source 带标签时支持：label\\ttext 或 text\\tlabel）

你只需要修改下面的 [USER CONFIG]。
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from transformers import (
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        get_linear_schedule_with_warmup,
    )
except Exception as e:
    raise RuntimeError(
        "This script requires 'transformers'. Install with: pip install transformers"
    ) from e


# ==========================================================
# ======================= USER CONFIG =======================
# ==========================================================
# 1) 把下面路径改成你本机的（Windows 用 r"..." 原始字符串）
DATA_CFG = {
    # 源域：训练集（必须）
    "SOURCE_TRAIN_PATH": r"./sourcedata/source_train.csv",

    # 源域：验证集（可选；为 None 则从 train 自动切 10%）
    "SOURCE_VAL_PATH": r"./sourcedata/source_validation.csv",

    # 目标域：无标签数据（必须）
    "TARGET_UNLABELED_PATH": r"./targetdata/train.csv",

    # 目标域：测试集（可选；有标签则可计算 open-set 指标）
    "TARGET_TEST_PATH": r"./targetdata/test.csv",

    # 数据列名（CSV/TSV/JSONL/JSON 时使用）
    "TEXT_COL": "text",
    "LABEL_COL": "label",

    # 模型
    "MODEL_NAME": "bert-base-english",  # 中文任务推荐
    # "MODEL_NAME": "distilbert-base-uncased",

    # 输出目录
    "OUTPUT_DIR": r"E:\TransferLearning\lstm_classifier\runs\pas_unida_local",

    # 如果你用的是 .txt：
    # - source/train 每行格式：label \t text 或 text \t label
    # - target/unlabeled 每行格式：text
}

# 2) 训练/方法配置（可以先不动，跑通后再调）
RUN_CFG = {
    "seed": 13,
    "device": "auto",      # "auto" / "cuda" / "cpu"
    "num_workers": 0,

    "max_len": 256,
    "bs_train": 16,
    "bs_pred": 64,
    "accum_steps": 1,
    "fp16": True,

    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "grad_clip": 1.0,

    # Phase 1：源域 supervised
    "epochs_src": 3,

    # Phase 1b：DAPT（可选；默认关，开了更稳但更耗时）
    "do_dapt": False,
    "dapt_steps": 200,
    "dapt_lr": 5e-5,
    "dapt_mlm_prob": 0.15,

    # Auto: tau/correction 搜索空间
    "auto_tau_grid": ("0.5", "0.7", "0.9", "adaptive", "none"),
    "auto_correct_grid": ("none", "mlls", "bbse"),
    "auto_target_subsample": 4000,
    "auto_train_steps": 200,
    "auto_lr": 1e-2,
    "auto_min_cov": 0.10,
    "auto_max_cov": 1.00,

    # Phase 3：自训练
    "rounds": 3,
    "epochs_st": 1,
    "lambda_pseudo": 1.0,
    "lambda_unk": 0.2,
    "pseudo_frac": (0.15, 0.30, 0.50),

    # tau 细化：每类动态阈值（更稳）
    "per_class_tau": True,
    "tau_beta0": 1.2,
    "tau_beta_eta": 0.5,
    "tau_shift_lambda": 0.5,

    # 置信度定义
    "score_type": "entropy",  # "entropy" 更稳 / "maxprob" 更直觉

    # KNN 安全阀（建议开）
    "use_knn": True,
    "knn_k": 10,
    "knn_agree": 0.8,
    "knn_min_sim": 0.25,
    "knn_bank_size": 8000,
}
# ==========================================================
# =================== END USER CONFIG =======================
# ==========================================================


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


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def log_softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    z = logits - m
    logsum = np.log(np.exp(z).sum(axis=axis, keepdims=True) + 1e-12)
    return z - logsum


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax_np(logits, axis=axis))


def entropy_from_probs_np(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def confidence_from_probs_np(p: np.ndarray, score_type: str) -> np.ndarray:
    if p.size == 0:
        return np.zeros((p.shape[0],), dtype=np.float32)
    if score_type == "maxprob":
        return np.max(p, axis=1)
    if score_type == "entropy":
        K = p.shape[1]
        ent = entropy_from_probs_np(p, axis=1)
        return 1.0 - ent / math.log(K + 1e-12)
    raise ValueError(f"Unknown score_type={score_type}. Use 'maxprob' or 'entropy'.")


def macro_f1_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp == 0 and (fp == 0 or fn == 0):
            f1 = 0.0
        else:
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)) if len(f1s) else 0.0


def accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    recalls = []
    for c in labels:
        mask = (y_true == c)
        if mask.sum() == 0:
            continue
        recalls.append(np.mean(y_pred[mask] == c))
    return float(np.mean(recalls)) if len(recalls) else 0.0


def roc_auc_binary_np(y_true01: np.ndarray, score: np.ndarray) -> float:
    y_true01 = y_true01.astype(np.int64)
    score = score.astype(np.float64)
    pos = score[y_true01 == 1]
    neg = score[y_true01 == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg], axis=0)
    ranks = all_scores.argsort().argsort().astype(np.float64) + 1.0
    ranks_pos = ranks[: len(pos)]
    u = ranks_pos.sum() - len(pos) * (len(pos) + 1) / 2.0
    auc = u / (len(pos) * len(neg) + 1e-12)
    return float(auc)


def fpr_at_tpr_np(y_true01: np.ndarray, score: np.ndarray, tpr: float = 0.95) -> float:
    y_true01 = y_true01.astype(np.int64)
    score = score.astype(np.float64)
    order = np.argsort(-score)
    y = y_true01[order]
    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        return float("nan")
    tp = 0
    fp = 0
    best_fpr = 1.0
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        cur_tpr = tp / P
        cur_fpr = fp / N
        if cur_tpr >= tpr:
            best_fpr = cur_fpr
            break
    return float(best_fpr)


# -----------------------------
# IO: reading datasets
# -----------------------------

UNKNOWN_STRINGS = {"unknown", "unk", "ood", "open", "other", "others", "unseen", "none", "null"}


def _read_table_with_pandas(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if ext in {".csv"}:
        return pd.read_csv(path)
    if ext in {".jsonl"}:
        return pd.read_json(path, lines=True)
    if ext in {".json"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported file extension for pandas: {ext}")


def read_text_label(
    path: str,
    text_col: str,
    label_col: str,
    has_label: bool,
) -> Tuple[List[str], Optional[List[Any]]]:
    ext = os.path.splitext(path)[1].lower()

    # TXT：每行一个样本
    if ext == ".txt":
        texts: List[str] = []
        labels: List[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n").strip("\r").strip()
                if not line:
                    continue
                if not has_label:
                    texts.append(line)
                else:
                    # 支持：label\ttext 或 text\tlabel
                    if "\t" not in line:
                        raise ValueError(
                            f"{path}: .txt with labels expects tab-separated 'label\\ttext' or 'text\\tlabel'. "
                            f"Got line without tab: {line[:80]}"
                        )
                    a, b = line.split("\t", 1)
                    a_s = a.strip()
                    b_s = b.strip()

                    def _is_int(s: str) -> bool:
                        try:
                            int(s)
                            return True
                        except Exception:
                            return False

                    if _is_int(a_s) and not _is_int(b_s):
                        labels.append(a_s)
                        texts.append(b_s)
                    elif _is_int(b_s) and not _is_int(a_s):
                        labels.append(b_s)
                        texts.append(a_s)
                    else:
                        # 实在分不清，就默认第一列是 label
                        labels.append(a_s)
                        texts.append(b_s)

        return texts, (labels if has_label else None)

    # 表格类：CSV/TSV/JSONL/JSON
    if pd is None:
        import csv
        if ext not in {".csv", ".tsv"}:
            raise RuntimeError(
                f"pandas not installed, only .csv/.tsv supported. "
                f"Install pandas or convert your file. path={path}"
            )
        sep = "\t" if ext == ".tsv" else ","
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for r in reader:
                rows.append(r)

        if len(rows) == 0:
            return [], ([] if has_label else None)

        if text_col not in rows[0]:
            raise KeyError(f"{path}: missing text_col='{text_col}' in header {list(rows[0].keys())}")

        texts = [str(r[text_col]) for r in rows]
        labels = [r[label_col] for r in rows] if has_label else None
        return texts, labels

    else:
        df = _read_table_with_pandas(path)
        if text_col not in df.columns:
            raise KeyError(f"{path}: missing text_col='{text_col}' in columns={list(df.columns)}")
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].tolist() if has_label else None
        return texts, labels


def build_label_mapping(source_labels: Sequence[Any]) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    uniq = []
    seen = set()
    for y in source_labels:
        if y not in seen:
            seen.add(y)
            uniq.append(y)
    try:
        uniq_sorted = sorted(uniq, key=lambda x: int(x))
        uniq = uniq_sorted
    except Exception:
        pass
    to_id = {y: i for i, y in enumerate(uniq)}
    to_label = {i: y for y, i in to_id.items()}
    return to_id, to_label


def normalize_label(y: Any) -> Any:
    if y is None:
        return None
    if isinstance(y, float) and np.isnan(y):
        return None
    if isinstance(y, str):
        s = y.strip().lower()
        if s in UNKNOWN_STRINGS:
            return "unknown"
        try:
            return int(s)
        except Exception:
            return y
    return y


def map_labels(
    labels: Sequence[Any],
    to_id: Dict[Any, int],
    unknown_id: int = -1,
) -> np.ndarray:
    out = []
    for y in labels:
        y2 = normalize_label(y)
        if y2 is None:
            out.append(unknown_id)
        elif isinstance(y2, str) and y2.strip().lower() in UNKNOWN_STRINGS:
            out.append(unknown_id)
        elif y2 in to_id:
            out.append(int(to_id[y2]))
        else:
            out.append(unknown_id)
    return np.asarray(out, dtype=np.int64)


# -----------------------------
# Datasets & Collate
# -----------------------------

class RawTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
        is_unknown: Optional[Sequence[bool]] = None,
        domain: str = "src",
    ) -> None:
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.weights = None if weights is None else list(weights)
        self.is_unknown = None if is_unknown is None else list(is_unknown)
        self.domain = domain

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": self.texts[idx], "domain": self.domain}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        else:
            item["label"] = -1
        item["weight"] = float(self.weights[idx]) if self.weights is not None else 1.0
        if self.is_unknown is not None:
            item["is_unknown"] = bool(self.is_unknown[idx])
        else:
            item["is_unknown"] = (item["label"] < 0)
        return item


def make_collate_fn(tokenizer, max_len: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        weights = torch.tensor([b.get("weight", 1.0) for b in batch], dtype=torch.float)
        is_unknown = torch.tensor([bool(b.get("is_unknown", False)) for b in batch], dtype=torch.bool)
        domains = [b.get("domain", "src") for b in batch]
        return {
            "enc": enc,
            "labels": labels,
            "weights": weights,
            "is_unknown": is_unknown,
            "domains": domains,
            "texts": texts,
        }
    return collate


# -----------------------------
# Model helpers
# -----------------------------

def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def maybe_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    from contextlib import nullcontext
    return nullcontext()


def apply_logit_bias_torch(logits: torch.Tensor, log_ratio: Optional[np.ndarray]) -> torch.Tensor:
    if log_ratio is None:
        return logits
    bias = torch.tensor(log_ratio, dtype=logits.dtype, device=logits.device).view(1, -1)
    return logits + bias


def probs_and_confidence(
    logits: torch.Tensor,
    score_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    if score_type == "maxprob":
        conf = torch.max(probs, dim=-1).values
    elif score_type == "entropy":
        K = probs.shape[-1]
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        conf = 1.0 - ent / math.log(K + 1e-12)
    else:
        raise ValueError(f"Unknown score_type={score_type}. Use 'maxprob' or 'entropy'.")
    return probs, pred, conf


@torch.no_grad()
def extract_embeddings_and_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    feats, logits_all, labels_all = [], [], []
    has_labels = False
    for batch in loader:
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        labels = batch["labels"].numpy()
        if np.any(labels >= 0):
            has_labels = True
        with maybe_autocast(device, fp16):
            out = model(**enc, output_hidden_states=True, return_dict=True)
            logits = out.logits
            hs = out.hidden_states[-1]
            feat = hs[:, 0]
        feats.append(to_numpy(feat))
        logits_all.append(to_numpy(logits))
        labels_all.append(labels)
    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    logits_all = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 1), dtype=np.float32)
    labels_all = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0,), dtype=np.int64)
    return feats, logits_all, (labels_all if has_labels else None)


# -----------------------------
# Label shift estimation
# -----------------------------

def estimate_priors_saerens_em(
    probs_t: np.ndarray,
    pi_s: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    K = probs_t.shape[1]
    pi_s = np.asarray(pi_s, dtype=np.float64)
    pi_s = np.clip(pi_s, 1e-12, 1.0)
    pi_s = pi_s / pi_s.sum()
    pi_t = pi_s.copy()
    for _ in range(max_iter):
        r = pi_t / pi_s
        q = probs_t * r.reshape(1, K)
        q = q / (q.sum(axis=1, keepdims=True) + 1e-12)
        pi_new = q.mean(axis=0)
        pi_new = np.clip(pi_new, 1e-12, 1.0)
        pi_new = pi_new / pi_new.sum()
        if np.max(np.abs(pi_new - pi_t)) < tol:
            pi_t = pi_new
            break
        pi_t = pi_new
    return pi_t


def confusion_matrix_pred_given_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    K: int,
) -> np.ndarray:
    C = np.zeros((K, K), dtype=np.float64)
    for j in range(K):
        mask = (y_true == j)
        denom = mask.sum()
        if denom == 0:
            C[:, j] = 0.0
            C[j, j] = 1.0
        else:
            for i in range(K):
                C[i, j] = np.mean(y_pred[mask] == i)
    return C


def estimate_priors_bbse(
    y_pred_t: np.ndarray,
    C_pred_given_true: np.ndarray,
    reg: float = 1e-3,
) -> np.ndarray:
    K = C_pred_given_true.shape[0]
    mu = np.zeros((K,), dtype=np.float64)
    for i in range(K):
        mu[i] = np.mean(y_pred_t == i) if len(y_pred_t) else 1.0 / K
    C = C_pred_given_true.copy()
    C = C + reg * np.eye(K)
    try:
        pi_hat = np.linalg.solve(C, mu)
    except np.linalg.LinAlgError:
        pi_hat, *_ = np.linalg.lstsq(C, mu, rcond=None)
    pi_hat = np.clip(pi_hat, 1e-12, 1.0)
    pi_hat = pi_hat / pi_hat.sum()
    return pi_hat


# -----------------------------
# Adaptive tau via 1D 2-Gaussian GMM
# -----------------------------

def gmm2_em_1d_threshold(
    scores: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[float, float]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if len(s) < 50:
        tau = float(np.median(s)) if len(s) else 0.5
        return float(np.clip(tau, 0.05, 0.95)), 0.0

    p25, p75 = np.percentile(s, [25, 75])
    mu1, mu2 = float(p25), float(p75)
    var1 = var2 = float(np.var(s) + 1e-4)
    w1 = 0.5
    w2 = 0.5

    def norm_pdf(x, mu, var):
        return np.exp(-0.5 * (x - mu) ** 2 / (var + 1e-12)) / math.sqrt(2 * math.pi * (var + 1e-12))

    ll_old = -1e18
    for _ in range(max_iter):
        p1 = w1 * norm_pdf(s, mu1, var1)
        p2 = w2 * norm_pdf(s, mu2, var2)
        denom = p1 + p2 + 1e-12
        r1 = p1 / denom
        r2 = p2 / denom

        w1_new = float(np.mean(r1))
        w2_new = 1.0 - w1_new
        mu1_new = float(np.sum(r1 * s) / (np.sum(r1) + 1e-12))
        mu2_new = float(np.sum(r2 * s) / (np.sum(r2) + 1e-12))
        var1_new = float(np.sum(r1 * (s - mu1_new) ** 2) / (np.sum(r1) + 1e-12) + 1e-6)
        var2_new = float(np.sum(r2 * (s - mu2_new) ** 2) / (np.sum(r2) + 1e-12) + 1e-6)

        ll = float(np.mean(np.log(denom)))
        if abs(ll - ll_old) < tol:
            w1, w2, mu1, mu2, var1, var2 = w1_new, w2_new, mu1_new, mu2_new, var1_new, var2_new
            break
        ll_old = ll
        w1, w2, mu1, mu2, var1, var2 = w1_new, w2_new, mu1_new, mu2_new, var1_new, var2_new

    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        var1, var2 = var2, var1
        w1, w2 = w2, w1

    a = 1.0 / (2 * var1) - 1.0 / (2 * var2)
    b = mu2 / var2 - mu1 / var1
    c = (mu1 ** 2) / (2 * var1) - (mu2 ** 2) / (2 * var2) + math.log((w2 * math.sqrt(var1)) / (w1 * math.sqrt(var2) + 1e-12) + 1e-12)

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            tau = (mu1 + mu2) / 2.0
        else:
            tau = -c / (b + 1e-12)
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            tau = (mu1 + mu2) / 2.0
        else:
            r1 = (-b - math.sqrt(disc)) / (2 * a)
            r2 = (-b + math.sqrt(disc)) / (2 * a)
            candidates = [r for r in [r1, r2] if (mu1 <= r <= mu2)]
            if len(candidates) > 0:
                tau = float(candidates[0])
            else:
                mid = (mu1 + mu2) / 2.0
                tau = float(r1 if abs(r1 - mid) < abs(r2 - mid) else r2)

    sep = abs(mu2 - mu1) / math.sqrt(var1 + var2 + 1e-12)
    tau = float(np.clip(tau, 0.05, 0.95))
    return tau, float(sep)


# -----------------------------
# KNN pseudo-label checking
# -----------------------------

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def knn_agree_mask(
    tgt_feats: np.ndarray,
    tgt_pred: np.ndarray,
    bank_feats: np.ndarray,
    bank_labels: np.ndarray,
    k: int = 10,
    agree_ratio: float = 0.8,
    min_sim: float = 0.25,
) -> np.ndarray:
    if len(tgt_feats) == 0 or len(bank_feats) == 0:
        return np.ones((len(tgt_feats),), dtype=bool)
    k = int(max(1, min(k, len(bank_feats))))
    bank_feats_n = l2_normalize(bank_feats)
    tgt_feats_n = l2_normalize(tgt_feats)

    mask = np.zeros((len(tgt_feats),), dtype=bool)
    block = 2048
    for i0 in range(0, len(tgt_feats), block):
        f = tgt_feats_n[i0 : i0 + block]
        sims = f @ bank_feats_n.T
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        neigh_labels = bank_labels[idx]
        neigh_sims = np.take_along_axis(sims, idx, axis=1)
        for j in range(len(f)):
            labels = neigh_labels[j]
            sims_j = neigh_sims[j]
            uniq, cnt = np.unique(labels, return_counts=True)
            maj_i = int(uniq[np.argmax(cnt)])
            maj_ratio = float(np.max(cnt) / k)
            avg_sim = float(np.mean(sims_j))
            pred = int(tgt_pred[i0 + j])
            mask[i0 + j] = (maj_i == pred) and (maj_ratio >= agree_ratio) and (avg_sim >= min_sim)
    return mask


# -----------------------------
# Training loops
# -----------------------------

@dataclass
class TrainCfg:
    model_name: str = "bert-base-chinese"
    max_len: int = 256

    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    fp16: bool = True

    bs_train: int = 16
    bs_pred: int = 64
    accum_steps: int = 1

    epochs_src: int = 3

    do_dapt: bool = False
    dapt_steps: int = 200
    dapt_lr: float = 5e-5
    dapt_mlm_prob: float = 0.15

    auto_tau_grid: Tuple[str, ...] = ("0.5", "0.7", "0.9", "adaptive", "none")
    auto_correct_grid: Tuple[str, ...] = ("none", "mlls", "bbse")
    auto_target_subsample: int = 4000
    auto_train_steps: int = 200
    auto_lr: float = 1e-2
    auto_min_cov: float = 0.10
    auto_max_cov: float = 1.00

    rounds: int = 3
    epochs_st: int = 1
    lambda_pseudo: float = 1.0
    lambda_unk: float = 0.2
    pseudo_frac: Union[float, Sequence[float]] = (0.15, 0.30, 0.50)

    per_class_tau: bool = True
    tau_beta0: float = 1.2
    tau_beta_eta: float = 0.5
    tau_shift_lambda: float = 0.5

    score_type: str = "entropy"  # "maxprob" or "entropy"

    use_knn: bool = True
    knn_k: int = 10
    knn_agree: float = 0.8
    knn_min_sim: float = 0.25
    knn_bank_size: int = 8000

    seed: int = 13
    device: str = "auto"
    num_workers: int = 0

    output_dir: str = "runs/pas_unida_local"


def parse_float_list(x: Union[float, Sequence[float]]) -> List[float]:
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight"]
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
        params.append({"params": [p], "weight_decay": wd})
    return torch.optim.AdamW(params, lr=lr)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    lambda_unk: float,
    fp16: bool,
    grad_clip: float,
    accum_steps: int,
) -> Dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    total = 0
    loss_sum = 0.0
    loss_l_sum = 0.0
    loss_u_sum = 0.0

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        is_unknown = batch["is_unknown"].to(device)

        with maybe_autocast(device, fp16):
            out = model(**enc, return_dict=True)
            logits = out.logits

            mask_l = (labels >= 0) & (~is_unknown)
            mask_u = is_unknown | (labels < 0)

            loss = torch.tensor(0.0, device=device)
            loss_l = torch.tensor(0.0, device=device)
            loss_u = torch.tensor(0.0, device=device)

            if mask_l.any():
                ce = F.cross_entropy(logits[mask_l], labels[mask_l], reduction="none")
                loss_l = (ce * weights[mask_l]).mean()
                loss = loss + loss_l

            if lambda_unk > 0 and mask_u.any():
                p = F.softmax(logits[mask_u], dim=-1)
                # uniform CE: - mean_c log p_c
                loss_u = -torch.log(p + 1e-12).mean(dim=1).mean()
                loss = loss + lambda_unk * loss_u

            loss = loss / float(accum_steps)

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bsz = labels.shape[0]
        total += bsz
        loss_sum += float(loss.detach().cpu()) * bsz * accum_steps
        loss_l_sum += float(loss_l.detach().cpu()) * bsz
        loss_u_sum += float(loss_u.detach().cpu()) * bsz

    return {
        "loss": loss_sum / max(1, total),
        "loss_labeled": loss_l_sum / max(1, total),
        "loss_unknown": loss_u_sum / max(1, total),
    }


@torch.no_grad()
def evaluate_known_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    log_ratio: Optional[np.ndarray],
) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        labels = batch["labels"].numpy()
        mask = labels >= 0
        if not np.any(mask):
            continue
        with maybe_autocast(device, fp16):
            out = model(**enc, return_dict=True)
            logits = apply_logit_bias_torch(out.logits, log_ratio)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).cpu().numpy()
        y_true.append(labels[mask])
        y_pred.append(pred[mask])
    if not y_true:
        return {"acc": float("nan"), "macro_f1": float("nan"), "bacc": float("nan")}
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labs = sorted(list(set(y_true.tolist())))
    return {
        "acc": accuracy_np(y_true, y_pred),
        "macro_f1": macro_f1_np(y_true, y_pred, labs),
        "bacc": balanced_accuracy_np(y_true, y_pred, labs),
    }


# -----------------------------
# Per-class tau schedule
# -----------------------------

def compute_tau_per_class(
    base_tau: float,
    round_idx: int,
    num_rounds: int,
    pred: np.ndarray,
    conf: np.ndarray,
    pi_s: np.ndarray,
    pi_t: Optional[np.ndarray],
    beta0: float,
    eta: float,
    shift_lambda: float,
    clip: Tuple[float, float] = (0.05, 0.95),
) -> np.ndarray:
    K = len(pi_s)
    base_tau = float(base_tau)

    if num_rounds <= 1:
        beta_global = 1.0
    else:
        frac = round_idx / float(num_rounds - 1)
        beta_global = beta0 - (beta0 - 1.0) * min(max(frac, 0.0), 1.0)

    conf_c = np.zeros((K,), dtype=np.float64)
    conf_global = float(np.mean(conf)) if len(conf) else 0.5
    for c in range(K):
        mask = (pred == c)
        conf_c[c] = float(np.mean(conf[mask])) if mask.sum() > 0 else conf_global

    beta_c = beta_global * (conf_global / (conf_c + 1e-12)) ** eta
    beta_c = np.clip(beta_c, 0.8, 1.6)

    if pi_t is None:
        gamma = np.ones((K,), dtype=np.float64)
    else:
        ratio = (pi_t + 1e-12) / (pi_s + 1e-12)
        gamma = ratio ** shift_lambda
        gamma = np.clip(gamma, 0.7, 1.3)

    tau_c = base_tau * beta_c * gamma
    tau_c = np.clip(tau_c, clip[0], clip[1])
    return tau_c.astype(np.float64)


# -----------------------------
# Pseudo-labeling
# -----------------------------

@torch.no_grad()
def pseudo_label_target(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    base_tau: Optional[float],
    score_type: str,
    log_ratio: Optional[np.ndarray],
    per_class_tau: bool,
    tau_c: Optional[np.ndarray],
    pseudo_frac: Optional[float],
    use_knn: bool,
    bank_feats: Optional[np.ndarray],
    bank_labels: Optional[np.ndarray],
    knn_k: int,
    knn_agree: float,
    knn_min_sim: float,
) -> Dict[str, Any]:
    model.eval()

    if base_tau is None:
        use_knn = False
        pseudo_frac = None
        per_class_tau = False
        tau_c = None

    all_pred, all_conf, all_probs, all_feats = [], [], [], []
    texts: List[str] = []

    for batch in loader:
        texts.extend(batch["texts"])
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        with maybe_autocast(device, fp16):
            out = model(**enc, output_hidden_states=True, return_dict=True)
            logits = apply_logit_bias_torch(out.logits, log_ratio)
            hs = out.hidden_states[-1][:, 0]
            probs, pred, conf = probs_and_confidence(logits, score_type)

        all_pred.append(to_numpy(pred))
        all_conf.append(to_numpy(conf))
        all_probs.append(to_numpy(probs))
        all_feats.append(to_numpy(hs))

    pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0,), dtype=np.int64)
    conf = np.concatenate(all_conf, axis=0) if all_conf else np.zeros((0,), dtype=np.float32)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 1), dtype=np.float32)
    feats = np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 1), dtype=np.float32)

    N = len(pred)
    if base_tau is None:
        known_by_tau = np.ones((N,), dtype=bool)
        unk_by_tau = np.zeros((N,), dtype=bool)
    else:
        if per_class_tau and (tau_c is not None):
            thr = tau_c[pred]
        else:
            thr = float(base_tau)
        known_by_tau = conf >= thr
        unk_by_tau = ~known_by_tau

    if use_knn and bank_feats is not None and bank_labels is not None and N > 0:
        agree = knn_agree_mask(
            tgt_feats=feats,
            tgt_pred=pred,
            bank_feats=bank_feats,
            bank_labels=bank_labels,
            k=knn_k,
            agree_ratio=knn_agree,
            min_sim=knn_min_sim,
        )
        known_by_tau = known_by_tau & agree
        unk_by_tau = unk_by_tau | (~agree)

    known_mask = known_by_tau.copy()
    ignore_mask = np.zeros((N,), dtype=bool)

    if pseudo_frac is not None and base_tau is not None:
        frac = float(pseudo_frac)
        idx = np.where(known_mask)[0]
        if len(idx) > 0:
            if frac <= 0:
                keep = np.zeros((N,), dtype=bool)
            elif frac < 1.0:
                kkeep = int(max(1, round(frac * len(idx))))
                order = idx[np.argsort(-conf[idx])]
                keep_idx = order[:kkeep]
                keep = np.zeros((N,), dtype=bool)
                keep[keep_idx] = True
            else:
                keep = known_mask.copy()
            ignore_mask = known_mask & (~keep)
            known_mask = keep

    pseudo_y = np.full((N,), -1, dtype=np.int64)
    pseudo_y[known_mask] = pred[known_mask]

    return {
        "texts": texts,
        "pred": pred,
        "conf": conf,
        "probs": probs,
        "feats": feats,
        "pseudo_y": pseudo_y,
        "known_mask": known_mask,
        "unk_mask": unk_by_tau,
        "ignore_mask": ignore_mask,
        "known_rate": float(np.mean(known_mask)) if N else 0.0,
        "unk_rate": float(np.mean(unk_by_tau)) if N else 0.0,
    }


# -----------------------------
# Proxy-safe Auto Selection
# -----------------------------

@torch.no_grad()
def compute_source_priors(y_src: np.ndarray, K: int) -> np.ndarray:
    pi = np.zeros((K,), dtype=np.float64)
    for c in range(K):
        pi[c] = np.mean(y_src == c) if len(y_src) else 1.0 / K
    pi = np.clip(pi, 1e-12, 1.0)
    pi = pi / pi.sum()
    return pi


def train_reverse_head(
    feats_t: np.ndarray,
    y_t: np.ndarray,
    feat_dim: int,
    num_classes: int,
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_seed(seed)
    head = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    x = torch.tensor(feats_t, dtype=torch.float32, device=device)
    y = torch.tensor(y_t, dtype=torch.long, device=device)

    bs = min(256, len(x)) if len(x) else 1
    for _ in range(int(steps)):
        if len(x) == 0:
            break
        idx = torch.randint(0, len(x), (bs,), device=device)
        xb = x[idx]
        yb = y[idx]
        logits = head(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


@torch.no_grad()
def proxy_score_reverse_validation(
    src_val_feats: np.ndarray,
    src_val_y: np.ndarray,
    rev_head: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    rev_head.eval()
    x = torch.tensor(src_val_feats, dtype=torch.float32, device=device)
    logits = rev_head(x)
    pred = torch.argmax(logits, dim=-1).cpu().numpy()
    unk_id = logits.shape[-1] - 1
    unk_rate = float(np.mean(pred == unk_id)) if len(pred) else 0.0
    acc = float(np.mean(pred == src_val_y)) if len(pred) else 0.0
    return acc, unk_rate


def auto_select_config(
    model: nn.Module,
    cfg: TrainCfg,
    K: int,
    pi_s: np.ndarray,
    src_val_loader: DataLoader,
    tgt_loader_for_auto: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    src_val_feats, src_val_logits, src_val_y = extract_embeddings_and_logits(
        model, src_val_loader, device=device, fp16=cfg.fp16
    )
    assert src_val_y is not None, "Source val must have labels for proxy scoring."

    tgt_feats, tgt_logits, _ = extract_embeddings_and_logits(
        model, tgt_loader_for_auto, device=device, fp16=cfg.fp16
    )
    tgt_probs = softmax_np(tgt_logits, axis=-1)
    tgt_conf = confidence_from_probs_np(tgt_probs, cfg.score_type)

    src_val_probs = softmax_np(src_val_logits, axis=-1)
    src_val_pred = np.argmax(src_val_probs, axis=-1)
    C = confusion_matrix_pred_given_true(src_val_y, src_val_pred, K=K)

    best = {
        "proxy_acc": -1.0,
        "proxy_unk_rate": 1.0,
        "tau": None,
        "tau_mode": "none",
        "correct": "none",
        "pi_t": None,
        "log_ratio": None,
        "notes": "",
    }

    for tau_s in cfg.auto_tau_grid:
        if tau_s.lower() == "none":
            tau_base = None
            sep = 0.0
        elif tau_s.lower() == "adaptive":
            tau_base, sep = gmm2_em_1d_threshold(scores=tgt_conf)
        else:
            tau_base = float(tau_s)
            sep = 0.0

        for corr in cfg.auto_correct_grid:
            corr = corr.lower()
            if corr == "none":
                pi_t = None
                log_ratio = None
            else:
                if tau_base is None:
                    known_mask = np.ones((len(tgt_probs),), dtype=bool)
                else:
                    known_mask = tgt_conf >= float(tau_base)
                probs_known = tgt_probs[known_mask]
                if len(probs_known) < 20:
                    pi_t = None
                    log_ratio = None
                else:
                    if corr == "mlls":
                        pi_t = estimate_priors_saerens_em(probs_known, pi_s=pi_s)
                    elif corr == "bbse":
                        y_pred_known = np.argmax(probs_known, axis=-1)
                        pi_t = estimate_priors_bbse(y_pred_known, C_pred_given_true=C)
                    else:
                        raise ValueError(f"Unknown correction mode {corr}")
                    log_ratio = np.log((pi_t + 1e-12) / (pi_s + 1e-12))

            unk_id = K
            logits_adj = tgt_logits if log_ratio is None else (tgt_logits + log_ratio.reshape(1, -1))
            probs_adj = softmax_np(logits_adj, axis=-1)
            pred = np.argmax(probs_adj, axis=-1)
            conf = confidence_from_probs_np(probs_adj, cfg.score_type)

            if tau_base is None:
                known = np.ones((len(pred),), dtype=bool)
            else:
                known = conf >= float(tau_base)

            cov = float(np.mean(known)) if len(known) else 0.0
            if cov < cfg.auto_min_cov or cov > cfg.auto_max_cov:
                continue

            y_t = np.where(known, pred, unk_id).astype(np.int64)
            rev_head = train_reverse_head(
                feats_t=tgt_feats,
                y_t=y_t,
                feat_dim=tgt_feats.shape[1],
                num_classes=K + 1,
                steps=cfg.auto_train_steps,
                lr=cfg.auto_lr,
                seed=cfg.seed + 17,
                device=device,
            )
            acc, unk_rate = proxy_score_reverse_validation(
                src_val_feats=src_val_feats,
                src_val_y=src_val_y,
                rev_head=rev_head,
                device=device,
            )

            better = False
            if acc > best["proxy_acc"] + 1e-6:
                better = True
            elif abs(acc - best["proxy_acc"]) <= 1e-6 and unk_rate < best["proxy_unk_rate"] - 1e-6:
                better = True

            if better:
                best.update(
                    {
                        "proxy_acc": float(acc),
                        "proxy_unk_rate": float(unk_rate),
                        "tau": (None if tau_base is None else float(tau_base)),
                        "tau_mode": tau_s,
                        "correct": corr,
                        "pi_t": (None if pi_t is None else pi_t.tolist()),
                        "log_ratio": (None if log_ratio is None else log_ratio.tolist()),
                        "notes": f"sep={sep:.3f} cov={cov:.3f}",
                    }
                )
    return best


# -----------------------------
# DAPT (optional)
# -----------------------------

def maybe_run_dapt(
    cfg: TrainCfg,
    tokenizer,
    target_texts: Sequence[str],
    device: torch.device,
    out_dir: str,
) -> Optional[str]:
    if not cfg.do_dapt:
        return None

    print(f"[DAPT] Starting MLM on target unlabeled, steps={cfg.dapt_steps} ...")
    dapt_dir = os.path.join(out_dir, "dapt_mlm")
    ensure_dir(dapt_dir)

    mlm_model = AutoModelForMaskedLM.from_pretrained(cfg.model_name)
    mlm_model.to(device)
    mlm_model.train()

    class MLMDataset(Dataset):
        def __init__(self, texts: Sequence[str], tokenizer, max_len: int):
            self.texts = list(texts)
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    ds = MLMDataset(target_texts, tokenizer, cfg.max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=cfg.dapt_mlm_prob)
    loader = DataLoader(ds, batch_size=cfg.bs_train, shuffle=True, num_workers=cfg.num_workers, collate_fn=collator)

    opt = build_optimizer(mlm_model, lr=cfg.dapt_lr, weight_decay=cfg.weight_decay)
    total_steps = int(cfg.dapt_steps)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.fp16 and device.type == "cuda"))

    step = 0
    opt.zero_grad(set_to_none=True)
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with maybe_autocast(device, cfg.fp16):
                out = mlm_model(**batch, return_dict=True)
                loss = out.loss / float(cfg.accum_steps)
            scaler.scale(loss).backward()
            if (step + 1) % cfg.accum_steps == 0:
                scaler.unscale_(opt)
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if step % 50 == 0:
                print(f"[DAPT] step {step}/{total_steps} loss={float(loss.detach().cpu())*cfg.accum_steps:.4f}")
            step += 1

    mlm_model.save_pretrained(dapt_dir)
    tokenizer.save_pretrained(dapt_dir)
    print(f"[DAPT] Saved MLM checkpoint to: {dapt_dir}")
    return dapt_dir


def build_knn_bank(
    model: nn.Module,
    tokenizer,
    cfg: TrainCfg,
    src_texts: Sequence[str],
    src_labels: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(src_texts)
    if n == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if cfg.knn_bank_size and n > cfg.knn_bank_size:
        idx = np.random.RandomState(cfg.seed).choice(n, size=cfg.knn_bank_size, replace=False)
        texts = [src_texts[i] for i in idx]
        labels = src_labels[idx]
    else:
        texts = list(src_texts)
        labels = src_labels

    ds = RawTextDataset(texts, labels=labels.tolist(), domain="src_bank")
    loader = DataLoader(
        ds,
        batch_size=cfg.bs_pred,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(tokenizer, cfg.max_len),
    )
    feats, _, _ = extract_embeddings_and_logits(model, loader, device=device, fp16=cfg.fp16)
    return feats.astype(np.float32), labels.astype(np.int64)


# -----------------------------
# Main pipeline
# -----------------------------

def run_pas_unida(
    cfg: TrainCfg,
    src_train_texts: List[str],
    src_train_y: np.ndarray,
    src_val_texts: List[str],
    src_val_y: np.ndarray,
    tgt_unl_texts: List[str],
    tgt_test_texts: Optional[List[str]] = None,
    tgt_test_y: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    ensure_dir(cfg.output_dir)
    run_dir = os.path.join(cfg.output_dir, f"run_{now_ts()}")
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, ensure_ascii=False, indent=2)

    print(f"[Device] {device}")
    print(f"[Data] src_train={len(src_train_texts)} src_val={len(src_val_texts)} tgt_unl={len(tgt_unl_texts)}")

    K = int(np.max(src_train_y) + 1)
    pi_s = compute_source_priors(src_train_y, K=K)
    print(f"[Source priors] {pi_s}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    dapt_ckpt = maybe_run_dapt(cfg, tokenizer, tgt_unl_texts, device=device, out_dir=run_dir)
    init_ckpt = dapt_ckpt if dapt_ckpt is not None else cfg.model_name

    model = AutoModelForSequenceClassification.from_pretrained(init_ckpt, num_labels=K)
    model.to(device)

    collate = make_collate_fn(tokenizer, cfg.max_len)

    src_train_loader = DataLoader(
        RawTextDataset(src_train_texts, labels=src_train_y.tolist(), domain="src"),
        batch_size=cfg.bs_train, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate
    )
    src_val_loader = DataLoader(
        RawTextDataset(src_val_texts, labels=src_val_y.tolist(), domain="src_val"),
        batch_size=cfg.bs_pred, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate
    )
    tgt_unl_loader_pred = DataLoader(
        RawTextDataset(tgt_unl_texts, labels=None, domain="tgt_unl"),
        batch_size=cfg.bs_pred, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate
    )

    # Phase 1: source training
    print("[Phase 1] Source supervised fine-tuning...")
    opt = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_src * math.ceil(len(src_train_loader) / max(1, cfg.accum_steps))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val = -1.0
    best_path = os.path.join(run_dir, "best_src")
    for ep in range(cfg.epochs_src):
        tr = train_one_epoch(
            model, src_train_loader, opt, sched, device,
            lambda_unk=0.0, fp16=cfg.fp16, grad_clip=cfg.grad_clip, accum_steps=cfg.accum_steps
        )
        ev = evaluate_known_only(model, src_val_loader, device=device, fp16=cfg.fp16, log_ratio=None)
        print(f"[Src] epoch {ep+1}/{cfg.epochs_src} train={tr} val={ev}")
        if ev["macro_f1"] > best_val:
            best_val = ev["macro_f1"]
            ensure_dir(best_path)
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    model = AutoModelForSequenceClassification.from_pretrained(best_path, num_labels=K)
    model.to(device)

    bank_feats = bank_labels = None
    if cfg.use_knn:
        print("[KNN] Building source memory bank...")
        bank_feats, bank_labels = build_knn_bank(model, tokenizer, cfg, src_train_texts, src_train_y, device=device)
        print(f"[KNN] bank size={len(bank_labels)} feat_dim={bank_feats.shape[1]}")

    # Phase 2: auto selection
    print("[Phase 2] Auto selection via proxy-safe reverse validation...")
    if len(tgt_unl_texts) > cfg.auto_target_subsample:
        rng = np.random.RandomState(cfg.seed + 123)
        idx = rng.choice(len(tgt_unl_texts), size=cfg.auto_target_subsample, replace=False)
        tgt_auto_texts = [tgt_unl_texts[i] for i in idx]
    else:
        tgt_auto_texts = tgt_unl_texts

    tgt_auto_loader = DataLoader(
        RawTextDataset(tgt_auto_texts, labels=None, domain="tgt_auto"),
        batch_size=cfg.bs_pred, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate
    )

    best_cfg = auto_select_config(
        model=model,
        cfg=cfg,
        K=K,
        pi_s=pi_s,
        src_val_loader=src_val_loader,
        tgt_loader_for_auto=tgt_auto_loader,
        device=device,
    )
    print(f"[Auto] selected: {best_cfg}")
    with open(os.path.join(run_dir, "auto_selected.json"), "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)

    base_tau = best_cfg["tau"]
    corr_mode = best_cfg["correct"]
    log_ratio = None if best_cfg["log_ratio"] is None else np.asarray(best_cfg["log_ratio"], dtype=np.float64)
    pi_t = None if best_cfg["pi_t"] is None else np.asarray(best_cfg["pi_t"], dtype=np.float64)

    # Phase 3: iterative self-training
    print("[Phase 3] Iterative safe self-training...")
    pseudo_frac_list = parse_float_list(cfg.pseudo_frac)
    if len(pseudo_frac_list) < cfg.rounds:
        pseudo_frac_list = pseudo_frac_list + [pseudo_frac_list[-1]] * (cfg.rounds - len(pseudo_frac_list))

    history = []
    for r in range(cfg.rounds):
        # re-estimate pi_t if enabled
        if corr_mode != "none":
            _, logits_t, _ = extract_embeddings_and_logits(model, tgt_unl_loader_pred, device=device, fp16=cfg.fp16)
            probs_t = softmax_np(logits_t, axis=-1)
            conf_t = confidence_from_probs_np(probs_t, cfg.score_type)

            known_mask = np.ones((len(probs_t),), dtype=bool) if base_tau is None else (conf_t >= float(base_tau))
            probs_known = probs_t[known_mask]
            if len(probs_known) >= 20:
                if corr_mode == "mlls":
                    pi_t = estimate_priors_saerens_em(probs_known, pi_s=pi_s)
                elif corr_mode == "bbse":
                    _, src_val_logits2, src_val_y2 = extract_embeddings_and_logits(model, src_val_loader, device=device, fp16=cfg.fp16)
                    src_val_probs2 = softmax_np(src_val_logits2, axis=-1)
                    src_val_pred2 = np.argmax(src_val_probs2, axis=-1)
                    C2 = confusion_matrix_pred_given_true(src_val_y2, src_val_pred2, K=K)
                    y_pred_known = np.argmax(probs_known, axis=-1)
                    pi_t = estimate_priors_bbse(y_pred_known, C_pred_given_true=C2)
                log_ratio = np.log((pi_t + 1e-12) / (pi_s + 1e-12))

        tau_c = None
        if (base_tau is not None) and cfg.per_class_tau:
            tmp = pseudo_label_target(
                model=model,
                loader=tgt_unl_loader_pred,
                device=device,
                fp16=cfg.fp16,
                base_tau=base_tau,
                score_type=cfg.score_type,
                log_ratio=log_ratio,
                per_class_tau=False,
                tau_c=None,
                pseudo_frac=None,
                use_knn=False,
                bank_feats=None,
                bank_labels=None,
                knn_k=cfg.knn_k,
                knn_agree=cfg.knn_agree,
                knn_min_sim=cfg.knn_min_sim,
            )
            tau_c = compute_tau_per_class(
                base_tau=float(base_tau),
                round_idx=r,
                num_rounds=cfg.rounds,
                pred=tmp["pred"],
                conf=tmp["conf"],
                pi_s=pi_s,
                pi_t=pi_t,
                beta0=cfg.tau_beta0,
                eta=cfg.tau_beta_eta,
                shift_lambda=cfg.tau_shift_lambda,
            )

        pseudo = pseudo_label_target(
            model=model,
            loader=tgt_unl_loader_pred,
            device=device,
            fp16=cfg.fp16,
            base_tau=base_tau,
            score_type=cfg.score_type,
            log_ratio=log_ratio,
            per_class_tau=cfg.per_class_tau,
            tau_c=tau_c,
            pseudo_frac=(pseudo_frac_list[r] if base_tau is not None else None),
            use_knn=cfg.use_knn,
            bank_feats=bank_feats,
            bank_labels=bank_labels,
            knn_k=cfg.knn_k,
            knn_agree=cfg.knn_agree,
            knn_min_sim=cfg.knn_min_sim,
        )

        known_idx = np.where(pseudo["known_mask"])[0]
        unk_idx = np.where(pseudo["unk_mask"])[0]
        print(
            f"[Round {r+1}/{cfg.rounds}] known_rate={pseudo['known_rate']:.3f} "
            f"unk_rate={pseudo['unk_rate']:.3f} keep={len(known_idx)} unk={len(unk_idx)}"
        )

        alpha = np.ones((K,), dtype=np.float64)
        if pi_t is not None:
            alpha = (pi_t + 1e-12) / (pi_s + 1e-12)
            alpha = alpha / np.mean(alpha)

        pseudo_texts = [pseudo["texts"][i] for i in known_idx]
        pseudo_y = pseudo["pseudo_y"][known_idx]
        pseudo_w = [float(cfg.lambda_pseudo) * float(alpha[y]) for y in pseudo_y]
        unk_texts = [pseudo["texts"][i] for i in unk_idx]

        mix_texts = list(src_train_texts) + pseudo_texts + unk_texts
        mix_labels = list(src_train_y.tolist()) + list(pseudo_y.tolist()) + ([-1] * len(unk_texts))
        mix_weights = ([1.0] * len(src_train_texts)) + pseudo_w + ([1.0] * len(unk_texts))
        mix_isunk = ([False] * len(src_train_texts)) + ([False] * len(pseudo_texts)) + ([True] * len(unk_texts))

        mixed = RawTextDataset(mix_texts, labels=mix_labels, weights=mix_weights, is_unknown=mix_isunk, domain="mix")
        st_loader = DataLoader(
            mixed, batch_size=cfg.bs_train, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate
        )

        opt_st = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps_st = cfg.epochs_st * math.ceil(len(st_loader) / max(1, cfg.accum_steps))
        warmup_steps_st = int(cfg.warmup_ratio * total_steps_st)
        sched_st = get_linear_schedule_with_warmup(opt_st, num_warmup_steps=warmup_steps_st, num_training_steps=total_steps_st)

        for ep in range(cfg.epochs_st):
            tr = train_one_epoch(
                model, st_loader, opt_st, sched_st, device,
                lambda_unk=(cfg.lambda_unk if base_tau is not None else 0.0),
                fp16=cfg.fp16, grad_clip=cfg.grad_clip, accum_steps=cfg.accum_steps
            )
            ev = evaluate_known_only(model, src_val_loader, device=device, fp16=cfg.fp16, log_ratio=None)
            print(f"[ST] round {r+1} epoch {ep+1}/{cfg.epochs_st} train={tr} src_val={ev}")
            history.append({"round": r + 1, "epoch": ep + 1, "train": tr, "src_val": ev})

    final_dir = os.path.join(run_dir, "final_model")
    ensure_dir(final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    src_final = evaluate_known_only(model, src_val_loader, device=device, fp16=cfg.fp16, log_ratio=None)
    print(f"[Final] source val: {src_final}")

    results: Dict[str, Any] = {
        "run_dir": run_dir,
        "auto_selected": best_cfg,
        "source_val_final": src_final,
        "history": history,
    }

    if tgt_test_texts is not None and tgt_test_y is not None:
        tgt_test_loader = DataLoader(
            RawTextDataset(tgt_test_texts, labels=tgt_test_y.tolist(), domain="tgt_test"),
            batch_size=cfg.bs_pred, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate
        )
        _, logits_tt, y_tt = extract_embeddings_and_logits(model, tgt_test_loader, device=device, fp16=cfg.fp16)
        logits_tt_adj = logits_tt if log_ratio is None else (logits_tt + log_ratio.reshape(1, -1))
        probs_tt = softmax_np(logits_tt_adj, axis=-1)
        pred_tt = np.argmax(probs_tt, axis=-1)
        conf_tt = confidence_from_probs_np(probs_tt, cfg.score_type)

        if base_tau is None:
            pred_open = pred_tt
        else:
            is_known = conf_tt >= float(base_tau)
            pred_open = np.where(is_known, pred_tt, -1).astype(np.int64)

        y_true = y_tt
        known_mask = y_true >= 0
        unk_mask = y_true < 0

        out: Dict[str, Any] = {}
        if known_mask.sum() > 0:
            out["known_acc"] = accuracy_np(y_true[known_mask], pred_open[known_mask])
            out["known_macro_f1"] = macro_f1_np(y_true[known_mask], pred_open[known_mask], labels=list(range(K)))
            out["known_bacc"] = balanced_accuracy_np(y_true[known_mask], pred_open[known_mask], labels=list(range(K)))

        if unk_mask.sum() > 0 and base_tau is not None:
            ood_score = 1.0 - conf_tt
            y_ood = (y_true < 0).astype(np.int64)
            out["ood_auc"] = roc_auc_binary_np(y_ood, ood_score)
            out["ood_fpr95"] = fpr_at_tpr_np(y_ood, ood_score, tpr=0.95)
            out["unk_recall"] = float(np.mean(pred_open[unk_mask] < 0))
            out["known_recall"] = float(np.mean(pred_open[known_mask] >= 0)) if known_mask.sum() > 0 else float("nan")

        results["target_test"] = out
        print(f"[Final] target test: {out}")

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved everything to: {run_dir}")
    return results


# -----------------------------
# Local entry (NO console args)
# -----------------------------

def _check_paths(cfg: Dict[str, Any]) -> None:
    must = ["SOURCE_TRAIN_PATH", "TARGET_UNLABELED_PATH"]
    for k in must:
        p = cfg.get(k, None)
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"[Path Error] {k} not found: {p}")

    opt = ["SOURCE_VAL_PATH", "TARGET_TEST_PATH"]
    for k in opt:
        p = cfg.get(k, None)
        if p and (not os.path.exists(p)):
            raise FileNotFoundError(f"[Path Error] {k} set but not found: {p}")


def main_local_run() -> None:
    _check_paths(DATA_CFG)

    # build TrainCfg from RUN_CFG + DATA_CFG
    cfg = TrainCfg(
        model_name=DATA_CFG["MODEL_NAME"],
        output_dir=DATA_CFG["OUTPUT_DIR"],
        max_len=int(RUN_CFG["max_len"]),
        bs_train=int(RUN_CFG["bs_train"]),
        bs_pred=int(RUN_CFG["bs_pred"]),
        accum_steps=int(RUN_CFG["accum_steps"]),
        fp16=bool(RUN_CFG["fp16"]),
        lr=float(RUN_CFG["lr"]),
        weight_decay=float(RUN_CFG["weight_decay"]),
        warmup_ratio=float(RUN_CFG["warmup_ratio"]),
        grad_clip=float(RUN_CFG["grad_clip"]),
        epochs_src=int(RUN_CFG["epochs_src"]),
        do_dapt=bool(RUN_CFG["do_dapt"]),
        dapt_steps=int(RUN_CFG["dapt_steps"]),
        dapt_lr=float(RUN_CFG["dapt_lr"]),
        dapt_mlm_prob=float(RUN_CFG["dapt_mlm_prob"]),
        auto_tau_grid=tuple(RUN_CFG["auto_tau_grid"]),
        auto_correct_grid=tuple(RUN_CFG["auto_correct_grid"]),
        auto_target_subsample=int(RUN_CFG["auto_target_subsample"]),
        auto_train_steps=int(RUN_CFG["auto_train_steps"]),
        auto_lr=float(RUN_CFG["auto_lr"]),
        auto_min_cov=float(RUN_CFG["auto_min_cov"]),
        auto_max_cov=float(RUN_CFG["auto_max_cov"]),
        rounds=int(RUN_CFG["rounds"]),
        epochs_st=int(RUN_CFG["epochs_st"]),
        lambda_pseudo=float(RUN_CFG["lambda_pseudo"]),
        lambda_unk=float(RUN_CFG["lambda_unk"]),
        pseudo_frac=RUN_CFG["pseudo_frac"],
        per_class_tau=bool(RUN_CFG["per_class_tau"]),
        tau_beta0=float(RUN_CFG["tau_beta0"]),
        tau_beta_eta=float(RUN_CFG["tau_beta_eta"]),
        tau_shift_lambda=float(RUN_CFG["tau_shift_lambda"]),
        score_type=str(RUN_CFG["score_type"]),
        use_knn=bool(RUN_CFG["use_knn"]),
        knn_k=int(RUN_CFG["knn_k"]),
        knn_agree=float(RUN_CFG["knn_agree"]),
        knn_min_sim=float(RUN_CFG["knn_min_sim"]),
        knn_bank_size=int(RUN_CFG["knn_bank_size"]),
        seed=int(RUN_CFG["seed"]),
        device=str(RUN_CFG["device"]),
        num_workers=int(RUN_CFG["num_workers"]),
    )

    text_col = DATA_CFG["TEXT_COL"]
    label_col = DATA_CFG["LABEL_COL"]

    # read source train/val
    src_train_texts, src_train_labels_raw = read_text_label(
        DATA_CFG["SOURCE_TRAIN_PATH"], text_col, label_col, has_label=True
    )

    src_val_path = DATA_CFG.get("SOURCE_VAL_PATH", None)
    if src_val_path:
        src_val_texts, src_val_labels_raw = read_text_label(
            src_val_path, text_col, label_col, has_label=True
        )
    else:
        # split 10% from train
        rng = np.random.RandomState(cfg.seed)
        idx = np.arange(len(src_train_texts))
        rng.shuffle(idx)
        n_val = max(1, int(0.1 * len(idx)))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        src_val_texts = [src_train_texts[i] for i in val_idx]
        src_val_labels_raw = [src_train_labels_raw[i] for i in val_idx]
        src_train_texts = [src_train_texts[i] for i in tr_idx]
        src_train_labels_raw = [src_train_labels_raw[i] for i in tr_idx]

    # build label mapping from source train
    to_id, _ = build_label_mapping([normalize_label(y) for y in src_train_labels_raw])
    src_train_y = map_labels(src_train_labels_raw, to_id, unknown_id=-1)
    src_val_y = map_labels(src_val_labels_raw, to_id, unknown_id=-1)

    if np.any(src_train_y < 0) or np.any(src_val_y < 0):
        raise ValueError(
            "Source labels contain unknown/unmapped labels. "
            "请检查 source 的 label 是否都是同一套已知类（不能含 unknown/ood）。"
        )

    # read target unlabeled
    tgt_unl_texts, _ = read_text_label(
        DATA_CFG["TARGET_UNLABELED_PATH"], text_col, label_col, has_label=False
    )

    # read target test (optional)
    tgt_test_texts = None
    tgt_test_y = None
    tgt_test_path = DATA_CFG.get("TARGET_TEST_PATH", None)
    if tgt_test_path:
        tgt_test_texts, tgt_test_labels_raw = read_text_label(
            tgt_test_path, text_col, label_col, has_label=True
        )
        tgt_test_y = map_labels(tgt_test_labels_raw, to_id, unknown_id=-1)

    # run
    run_pas_unida(
        cfg=cfg,
        src_train_texts=src_train_texts,
        src_train_y=src_train_y,
        src_val_texts=src_val_texts,
        src_val_y=src_val_y,
        tgt_unl_texts=tgt_unl_texts,
        tgt_test_texts=tgt_test_texts,
        tgt_test_y=tgt_test_y,
    )


if __name__ == "__main__":
    main_local_run()
