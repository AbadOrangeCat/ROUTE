#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Open-Set + Label-Shift Test-Time Adaptation (runnable demo)

What this script does (end-to-end):
1) Train a source model on 2 known classes.
2) (Optional) Outlier Exposure (OE): use "unknown" samples to train an OOD head + enforce high entropy on unknown.
3) Calibrate:
   - Temperature scaling for the known-class softmax head
   - Classification threshold tau (on source val, macro-F1)
   - OOD threshold (on source val, fixed known false rejection rate)
4) Target-time adaptation on an unlabeled stream (contains label shift + open-set unknowns):
   - OOD filtering via OOD head
   - Label-shift estimation via Saerens EM on filtered samples
   - Mean-teacher self-training + consistency on confident known samples
   - Unknown regularization (push OOD head to unknown + maximize entropy of known-class head)

This is a "directly runnable" reference implementation. Default uses CIFAR-10 and treats:
known classes = {airplane(0), automobile(1)}, unknown classes = {2..9}.
If you don't have internet access (can't download CIFAR-10), run with --dataset synthetic.

Dependencies:
  pip install torch torchvision scikit-learn tqdm

Run examples:
  # Quick offline sanity check
  python robust_oslsda.py --dataset synthetic --use_oe

  # CIFAR-10 demo (downloads automatically)
  python robust_oslsda.py --dataset cifar10 --use_oe --epochs_src 20 --epochs_tta 5 --alpha_ood 0.2 --pi_target 0.2

Notes:
- This code is designed to be easy to read/modify for your own datasets.
- If you want to plug in your own data, see the "Custom dataset hook" section near the bottom.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from torchvision import datasets, transforms
except Exception:  # pragma: no cover
    datasets = None
    transforms = None


# -----------------------------
# Repro / device
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (slower but repeatable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# -----------------------------
# Math helpers
# -----------------------------
def softmax_T(logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.softmax(logits / T, dim=-1)


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0)
    return -(p * p.log()).sum(dim=-1)


# -----------------------------
# Metrics
# -----------------------------
def metrics_binary(y_true: np.ndarray, y_prob_pos: np.ndarray, tau: float) -> Dict[str, float]:
    y_pred = (y_prob_pos >= tau).astype(np.int64)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def metrics_ood(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    """
    id_scores / ood_scores: higher => more ID (known)
    AUROC uses ID as positive class (1), OOD as negative class (0).
    FPR95: choose a threshold so that TPR(ID) ~= 95%, then compute FPR on OOD
          (fraction of OOD mistakenly accepted as ID).
    """
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auroc = float("nan")

    # TPR(ID) = 0.95  ==> threshold = 5th percentile of ID scores
    thr = float(np.quantile(id_scores, 0.05))
    fpr95 = float((ood_scores >= thr).mean())
    return {"auroc": auroc, "fpr95": fpr95}


def metrics_stream_3c(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int = 2) -> Dict[str, float]:
    # Macro-F1 over {0,1,unknown}
    macro_f1_3c = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, unknown_label]))
    unknown_f1 = float(
        f1_score((y_true == unknown_label).astype(int), (y_pred == unknown_label).astype(int))
    )
    coverage = float((y_pred != unknown_label).mean())
    return {"macro_f1_3c": macro_f1_3c, "unknown_f1": unknown_f1, "coverage": coverage}


# -----------------------------
# Synthetic dataset (offline runnable)
# -----------------------------
class SyntheticOpenSet(Dataset):
    """
    2 known classes: y in {0,1}
    unknown: y=2
    shift=True -> rotate + noise (domain shift)
    """

    def __init__(
        self,
        n: int,
        pi_pos: float,
        alpha_ood: float,
        split: str,
        shift: bool,
        seed: int,
    ):
        super().__init__()
        rng = np.random.default_rng(seed + (0 if split == "train" else 12345))
        n_ood = int(round(n * alpha_ood))
        n_id = n - n_ood
        n_pos = int(round(n_id * pi_pos))
        n_neg = n_id - n_pos

        x_neg = rng.normal(loc=[-2.0, 0.0], scale=[1.0, 1.0], size=(n_neg, 2))
        x_pos = rng.normal(loc=[+2.0, 0.0], scale=[1.0, 1.0], size=(n_pos, 2))
        x_ood = rng.normal(loc=[0.0, +5.0], scale=[2.0, 2.0], size=(n_ood, 2))

        x = np.concatenate([x_neg, x_pos, x_ood], axis=0).astype(np.float32)
        y = np.concatenate(
            [np.zeros(n_neg, np.int64), np.ones(n_pos, np.int64), np.full(n_ood, 2, np.int64)]
        )

        idx = rng.permutation(len(y))
        x = x[idx]
        y = y[idx]

        if shift:
            theta = math.radians(30.0)
            R = np.array(
                [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32
            )
            x = (x @ R.T) + rng.normal(scale=0.3, size=x.shape).astype(np.float32)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx].item())


class SyntheticTwoView(Dataset):
    """(weak, strong, y) for synthetic, where strong adds small noise."""

    def __init__(self, base: Dataset, noise_std: float = 0.05):
        self.base = base
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        xw = x
        xs = x + self.noise_std * torch.randn_like(x)
        return xw, xs, y


# -----------------------------
# CIFAR-10 binary open-set dataset
# -----------------------------
class CIFAR10BinaryOpenSet(Dataset):
    """
    Wrap CIFAR-10 and expose:
      - known classes (two classes mapped to {0,1})
      - unknown classes (mapped to label 2)
    """

    def __init__(
        self,
        root: str,
        train: bool,
        transform,
        download: bool = True,
        known_classes: Tuple[int, int] = (0, 1),
        unknown_classes: Tuple[int, ...] = tuple(range(2, 10)),
    ):
        if datasets is None:
            raise RuntimeError("torchvision is required for cifar10 mode. Please install torchvision.")
        self.base = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        self.known_classes = known_classes
        self.unknown_classes = unknown_classes
        self.targets = np.array(self.base.targets, dtype=np.int64)
        self.known_map = {known_classes[0]: 0, known_classes[1]: 1}
        self.indices_known = np.where(np.isin(self.targets, np.array(known_classes)))[0].tolist()
        self.indices_unknown = np.where(np.isin(self.targets, np.array(unknown_classes)))[0].tolist()

    def subset_known(self) -> Dataset:
        return _CIFARSubset(self.base, self.indices_known, label_map=self.known_map, unknown_label=None)

    def subset_unknown(self) -> Dataset:
        return _CIFARSubset(self.base, self.indices_unknown, label_map=None, unknown_label=2)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if int(y) in self.known_map:
            y = self.known_map[int(y)]
        else:
            y = 2
        return x, int(y)


class _CIFARSubset(Dataset):
    def __init__(self, base, indices: List[int], label_map: Optional[Dict[int, int]], unknown_label: Optional[int]):
        self.base = base
        self.indices = indices
        self.label_map = label_map
        self.unknown_label = unknown_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y = self.base[self.indices[i]]
        y_int = int(y)
        if self.label_map is not None:
            y_int = self.label_map[y_int]
        elif self.unknown_label is not None:
            y_int = self.unknown_label
        return x, int(y_int)


class TwoViewWrapper(Dataset):
    """Wrap a dataset that returns (PIL_image, y) and produce (weak_tensor, strong_tensor, y)."""

    def __init__(self, base: Dataset, weak_t, strong_t):
        self.base = base
        self.weak_t = weak_t
        self.strong_t = strong_t

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return self.weak_t(x), self.strong_t(x), y


# -----------------------------
# Model
# -----------------------------
class MLP2D(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 64, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    """Simple CNN for CIFAR10 32x32."""

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return F.relu(x, inplace=True)


class RobustBinaryModel(nn.Module):
    """
    Feature extractor + 2-class head + OOD head.
    OOD head outputs logit for "unknown" (sigmoid -> p_ood).
    """

    def __init__(self, backbone: nn.Module, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.clf = nn.Linear(feat_dim, 2)
        self.ood = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.clf(feat)
        ood_logit = self.ood(feat).squeeze(-1)
        return logits, ood_logit, feat


# -----------------------------
# Calibration
# -----------------------------
@dataclass
class Calibration:
    temperature: float
    tau: float
    ood_threshold: float
    pi_source: np.ndarray  # shape (2,)


def fit_temperature_adam(model: RobustBinaryModel, loader: DataLoader, device: torch.device) -> float:
    """
    Fit a single temperature T on source val to minimize NLL (classification head only).
    Uses Adam (fast and stable).
    """
    model.eval()
    logits_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _, _ = model(x)
            logits_list.append(logits.detach().cpu())
            y_list.append(y.detach().cpu())
    logits = torch.cat(logits_list, dim=0)
    y = torch.cat(y_list, dim=0)

    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        T = logT.exp().clamp(min=0.05, max=10.0)
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        opt.step()
    return float(logT.detach().exp().clamp(min=0.05, max=10.0).item())


def select_tau_val(model: RobustBinaryModel, loader: DataLoader, device: torch.device, T: float) -> float:
    """
    Choose tau maximizing macro-F1 on source val (known classes only).
    """
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _, _ = model(x)
            p_pos = softmax_T(logits, T)[:, 1].cpu().numpy()
            probs.append(p_pos)
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)

    best_tau = 0.5
    best_f1 = -1.0
    for tau in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (y_prob >= tau).astype(int), average="macro")
        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau = float(tau)
    return best_tau


def select_ood_threshold_from_known(
    model: RobustBinaryModel,
    loader_known: DataLoader,
    device: torch.device,
    known_false_reject: float = 0.05,
) -> float:
    """
    Set threshold on p_ood such that P(p_ood < thr | known) ~= 1 - known_false_reject.
    i.e., accept at least 1-known_false_reject of known samples as known.
    """
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader_known:
            x = x.to(device)
            _, ood_logit, _ = model(x)
            p_ood = torch.sigmoid(ood_logit).cpu().numpy()
            scores.append(p_ood)
    p_ood_known = np.concatenate(scores, axis=0)
    thr = float(np.quantile(p_ood_known, 1.0 - known_false_reject))
    return thr


# -----------------------------
# Label shift: Saerens EM (binary)
# -----------------------------
def saerens_em_binary(p_yx: np.ndarray, pi_source: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """
    Estimate target prior pi_t under label shift using Saerens EM.
    p_yx: (N,2) posteriors predicted by source-trained classifier.
    """
    pi_s = pi_source.astype(np.float64)
    pi_t = pi_s.copy()
    eps = 1e-9
    for _ in range(max_iter):
        w = pi_t / (pi_s + eps)
        q = p_yx * w[None, :]
        q = q / (q.sum(axis=1, keepdims=True) + eps)
        pi_new = q.mean(axis=0)
        if float(np.max(np.abs(pi_new - pi_t))) < tol:
            pi_t = pi_new
            break
        pi_t = pi_new
    pi_t = pi_t / (pi_t.sum() + eps)
    return pi_t.astype(np.float32)


def adjust_posteriors(p_yx: np.ndarray, pi_source: np.ndarray, pi_target: np.ndarray) -> np.ndarray:
    eps = 1e-9
    w = pi_target / (pi_source + eps)
    q = p_yx * w[None, :]
    q = q / (q.sum(axis=1, keepdims=True) + eps)
    return q


@torch.no_grad()
def estimate_pi_target(
    model: RobustBinaryModel,
    loader_stream: DataLoader,
    device: torch.device,
    T: float,
    ood_thr: float,
    pi_source: np.ndarray,
    max_batches: int = 50,
) -> np.ndarray:
    """
    Estimate pi_target from an unlabeled stream:
    - Use OOD head to filter likely known samples (p_ood < ood_thr)
    - Run EM on their predicted posteriors
    """
    model.eval()
    probs: List[np.ndarray] = []
    for i, batch in enumerate(loader_stream):
        if i >= max_batches:
            break
        if len(batch) == 2:
            x, _ = batch
        else:
            x, _, _ = batch  # weak view
        x = x.to(device)
        logits, ood_logit, _ = model(x)
        p_ood = torch.sigmoid(ood_logit)
        mask = (p_ood < ood_thr)
        if mask.any():
            p = softmax_T(logits[mask], T).cpu().numpy()
            probs.append(p)
    if not probs:
        return pi_source.copy()
    p_yx = np.concatenate(probs, axis=0)
    return saerens_em_binary(p_yx, pi_source)


# -----------------------------
# Training: source + optional OE
# -----------------------------
def train_source(
    model: RobustBinaryModel,
    loader_known: DataLoader,
    loader_oe: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_ood: float,
    lambda_oe_entropy: float,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    oe_iter = iter(loader_oe) if loader_oe is not None else None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        iterator = loader_known
        if tqdm is not None:
            iterator = tqdm(loader_known, desc=f"[src] epoch {ep}/{epochs}", leave=False)

        for xb, yb in iterator:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, ood_logit, _ = model(xb)
            loss_cls = F.cross_entropy(logits, yb)
            # known => ood label 0
            loss_ood_known = F.binary_cross_entropy_with_logits(ood_logit, torch.zeros_like(ood_logit))
            loss = loss_cls + lambda_ood * loss_ood_known

            # OE batch: unknown => ood label 1, plus maximize entropy for clf head
            if oe_iter is not None:
                try:
                    x_oe, _ = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(loader_oe)
                    x_oe, _ = next(oe_iter)
                x_oe = x_oe.to(device)

                logits_oe, ood_logit_oe, _ = model(x_oe)
                loss_ood_oe = F.binary_cross_entropy_with_logits(ood_logit_oe, torch.ones_like(ood_logit_oe))

                p_oe = F.softmax(logits_oe, dim=-1)
                # maximize entropy => minimize negative entropy
                loss_ent = -entropy(p_oe).mean()

                loss = loss + lambda_ood * loss_ood_oe + lambda_oe_entropy * loss_ent

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)

        print(f"[src] epoch {ep}/{epochs} loss={total_loss/max(total_n,1):.4f}")


# -----------------------------
# Adaptation (ours): OOD-filtered mean-teacher self-training + EM label shift
# -----------------------------
@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, ema: float) -> None:
    """EMA update for both parameters and buffers."""
    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        if t_buf.dtype.is_floating_point:
            t_buf.data.mul_(ema).add_(s_buf.data * (1.0 - ema))
        else:
            t_buf.data.copy_(s_buf.data)

    for t_par, s_par in zip(teacher.parameters(), student.parameters()):
        t_par.data.mul_(ema).add_(s_par.data * (1.0 - ema))


def adapt_ours(
    student: RobustBinaryModel,
    stream_loader_train: DataLoader,
    device: torch.device,
    calib: Calibration,
    epochs: int = 5,
    lr: float = 1e-4,
    ema: float = 0.99,
    pseudo_th: float = 0.9,
    lambda_cons: float = 1.0,
    lambda_unknown_entropy: float = 0.01,
    lambda_ood_pos: float = 0.5,
    lambda_ood_known: float = 0.5,
    unk_th: float = 0.5,
    anchor_ood_th: float = 0.2,
    known_conf_th: float = 0.7,
    update_pi_every: int = 100,
) -> Tuple[RobustBinaryModel, np.ndarray]:
    """Open-set + label-shift TTA (robustified).

    Compared to the naive version, this adds two practical stabilizers that prevent the
    common *collapse-to-unknown* failure mode:

    1) **Two-threshold unknown selection**
       - We still *predict* unknown using the calibrated threshold `calib.ood_threshold`.
       - But we only *train* unknown losses on *very confident* unknown samples:
         `p_ood >= max(calib.ood_threshold, unk_th)`.

    2) **Known anchor for the OOD head**
       - During TTA we continue to push `p_ood -> 0` on confident, likely-known samples.
       - This keeps the OOD head calibrated and prevents its bias drifting upward.

    Notes:
      - Uses BCEWithLogitsLoss for numerical stability.
      - The calibrated prediction threshold is *not* changed here.

    Returns:
      (adapted_model, estimated_pi_target)
    """

    student = student.to(device)
    student.train()

    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=0.0)
    bce_logits = torch.nn.BCEWithLogitsLoss()

    # Start from source prior, then refresh with EM on the target stream.
    pi_t = calib.pi_source.copy()

    # thresholds used *for training selection*
    # - `calib.ood_threshold` is used for final prediction/evaluation
    thr_unk = max(float(calib.ood_threshold), float(unk_th))
    thr_anchor = min(thr_unk, float(anchor_ood_th))

    step = 0
    for ep in range(1, epochs + 1):
        for x_w, x_s, _ in tqdm(stream_loader_train, desc=f"[ours] epoch {ep}/{epochs}"):
            x_w = x_w.to(device)
            x_s = x_s.to(device)

            # (Re-)estimate label-shift prior periodically (step==0 also triggers this).
            if (step % update_pi_every) == 0:
                pi_t = estimate_pi_target(
                    student,
                    stream_loader_train,
                    device,
                    ood_thr=float(calib.ood_threshold),
                    pi_source=calib.pi_source,
                    T=calib.temperature,
                    iters=25,
                )

            with torch.no_grad():
                logits_t, ood_logit_t, _ = teacher(x_w)
                p_known_t = softmax_T(logits_t, calib.temperature)  # (B,2)
                p_ood_t = torch.sigmoid(ood_logit_t).view(-1)        # (B,)

                conf_t, _ = p_known_t.max(dim=1)

                # Label-shift corrected pseudo label (argmax of p(y|x)/pi_t).
                pi_t_torch = torch.from_numpy(pi_t).to(device).clamp_min(1e-6)
                q = p_known_t / pi_t_torch
                q = q / q.sum(dim=1, keepdim=True)
                pseudo_y = torch.argmax(q, dim=1)

                # Masks
                mask_pl = (conf_t >= pseudo_th) & (p_ood_t <= thr_anchor)
                mask_known_ood = (conf_t >= known_conf_th) & (p_ood_t <= thr_anchor)
                mask_unk = p_ood_t >= thr_unk

            logits_s, ood_logit_s, _ = student(x_s)
            p_known_s = softmax_T(logits_s, calib.temperature)

            # (1) Pseudo-label supervised loss on confident likely-known samples.
            if mask_pl.any():
                loss_pl = F.cross_entropy(logits_s[mask_pl], pseudo_y[mask_pl])
                loss_cons = torch.mean((p_known_s[mask_pl] - p_known_t[mask_pl]).pow(2))
            else:
                loss_pl = torch.tensor(0.0, device=device)
                loss_cons = torch.tensor(0.0, device=device)

            # (2) Unknown entropy maximization on *very confident unknown* samples.
            if mask_unk.any():
                p_unk = torch.softmax(logits_s[mask_unk], dim=1)
                loss_unk_ent = -(entropy(p_unk).mean())
            else:
                loss_unk_ent = torch.tensor(0.0, device=device)

            # (3) OOD head: positive BCE on confident unknown, and negative BCE on confident known.
            if mask_unk.any():
                loss_ood_pos = bce_logits(ood_logit_s[mask_unk].view(-1), torch.ones(int(mask_unk.sum()), device=device))
            else:
                loss_ood_pos = torch.tensor(0.0, device=device)

            if mask_known_ood.any():
                loss_ood_known = bce_logits(ood_logit_s[mask_known_ood].view(-1), torch.zeros(int(mask_known_ood.sum()), device=device))
            else:
                loss_ood_known = torch.tensor(0.0, device=device)

            loss = (
                loss_pl
                + lambda_cons * loss_cons
                + lambda_unknown_entropy * loss_unk_ent
                + lambda_ood_pos * loss_ood_pos
                + lambda_ood_known * loss_ood_known
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            ema_update(teacher, student, ema=ema)
            step += 1

        print(f"[ours] epoch {ep}/{epochs} pi_t={pi_t} (thr_pred={calib.ood_threshold:.4f}, thr_anchor={thr_anchor:.3f}, thr_unk={thr_unk:.3f})")

    return student.eval(), pi_t
# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def predict_known_posprob(model: RobustBinaryModel, loader: DataLoader, device: torch.device, T: float) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits, _, _ = model(x)
        p_pos = softmax_T(logits, T)[:, 1].cpu().numpy()
        ys.append(y.numpy())
        probs.append(p_pos)
    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def ood_scores_knownprob(model: RobustBinaryModel, loader_id: DataLoader, loader_ood: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    id_scores: List[np.ndarray] = []
    ood_scores: List[np.ndarray] = []
    for x, _ in loader_id:
        x = x.to(device)
        _, ood_logit, _ = model(x)
        p_known = (1.0 - torch.sigmoid(ood_logit)).cpu().numpy()
        id_scores.append(p_known)
    for x, _ in loader_ood:
        x = x.to(device)
        _, ood_logit, _ = model(x)
        p_known = (1.0 - torch.sigmoid(ood_logit)).cpu().numpy()
        ood_scores.append(p_known)
    return np.concatenate(id_scores), np.concatenate(ood_scores)


@torch.no_grad()
def predict_stream_3c(
    model: RobustBinaryModel,
    loader_stream_eval: DataLoader,
    device: torch.device,
    T: float,
    tau: float,
    ood_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    yt: List[np.ndarray] = []
    yp: List[np.ndarray] = []
    for batch in loader_stream_eval:
        if len(batch) == 2:
            x, y = batch
        else:
            x, _, y = batch  # weak view
        x = x.to(device)
        logits, ood_logit, _ = model(x)
        p_ood = torch.sigmoid(ood_logit)
        p_pos = softmax_T(logits, T)[:, 1]
        pred = (p_pos >= tau).long()
        pred[p_ood >= ood_thr] = 2
        yt.append(y.numpy())
        yp.append(pred.cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)


# -----------------------------
# Data builders
# -----------------------------
def build_synthetic(args) -> Dict[str, DataLoader]:
    src_train = SyntheticOpenSet(
        n=args.n_src_train, pi_pos=0.5, alpha_ood=0.0, split="train", shift=False, seed=args.seed
    )
    src_val = SyntheticOpenSet(
        n=args.n_src_val, pi_pos=0.5, alpha_ood=0.0, split="val", shift=False, seed=args.seed
    )

    # Outlier exposure: all unknown
    oe = SyntheticOpenSet(n=args.n_oe, pi_pos=0.5, alpha_ood=1.0, split="train", shift=False, seed=args.seed)

    tgt_stream = SyntheticOpenSet(
        n=args.n_tgt_stream, pi_pos=args.pi_target, alpha_ood=args.alpha_ood, split="test", shift=True, seed=args.seed
    )
    tgt_known = SyntheticOpenSet(
        n=args.n_tgt_test, pi_pos=args.pi_target, alpha_ood=0.0, split="test", shift=True, seed=args.seed
    )
    tgt_ood = SyntheticOpenSet(
        n=args.n_tgt_test, pi_pos=args.pi_target, alpha_ood=1.0, split="test", shift=True, seed=args.seed
    )

    # Filter known/unknown for source loaders
    src_train_known = [(x, y) for x, y in src_train if y != 2]
    src_val_known = [(x, y) for x, y in src_val if y != 2]
    oe_unknown = [(x, y) for x, y in oe if y == 2]

    loaders = {
        "src_train": DataLoader(src_train_known, batch_size=args.batch_size, shuffle=True),
        "src_val": DataLoader(src_val_known, batch_size=args.batch_size, shuffle=False),
        "oe": DataLoader(oe_unknown, batch_size=args.batch_size, shuffle=True),
        "tgt_stream_train": DataLoader(SyntheticTwoView(tgt_stream), batch_size=args.batch_size, shuffle=True),
        "tgt_stream_eval": DataLoader(SyntheticTwoView(tgt_stream), batch_size=args.batch_size, shuffle=False),
        "tgt_known": DataLoader([(x, y) for x, y in tgt_known if y != 2], batch_size=args.batch_size, shuffle=False),
        "tgt_id_for_ood": DataLoader([(x, y) for x, y in tgt_stream if y != 2], batch_size=args.batch_size, shuffle=False),
        "tgt_ood": DataLoader([(x, y) for x, y in tgt_ood if y == 2], batch_size=args.batch_size, shuffle=False),
    }
    return loaders


def build_cifar10(args) -> Dict[str, DataLoader]:
    if transforms is None:
        raise RuntimeError("torchvision is required for cifar10 mode. Please install torchvision.")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Source transforms
    src_train_t = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    src_eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Target transforms
    tgt_weak_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tgt_strong_t = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Separate train/eval views for source to avoid random aug on val
    ds_src_train = CIFAR10BinaryOpenSet(args.data_dir, train=True, transform=src_train_t, download=True)
    ds_src_val = CIFAR10BinaryOpenSet(args.data_dir, train=True, transform=src_eval_t, download=True)

    src_known_train = ds_src_train.subset_known()
    src_known_val = ds_src_val.subset_known()

    # Split indices on the known subset (same order across both datasets)
    n_known = len(src_known_train)
    n_val = min(args.n_src_val, max(1, n_known // 5))
    indices = list(range(n_known))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    src_train = Subset(src_known_train, train_idx)
    src_val = Subset(src_known_val, val_idx)

    # OE unknown from source train
    oe_ds = ds_src_train.subset_unknown()

    # Target pools from CIFAR10 test with transform=None (PIL) so we can apply two-view transforms
    ds_tgt_raw = CIFAR10BinaryOpenSet(args.data_dir, train=False, transform=None, download=True)
    id_pool = ds_tgt_raw.subset_known()
    ood_pool = ds_tgt_raw.subset_unknown()

    # Decide how many ID / OOD for stream
    n_stream = int(args.n_tgt_stream)
    n_ood = int(round(n_stream * args.alpha_ood))
    n_id = n_stream - n_ood
    n_pos = int(round(n_id * args.pi_target))
    n_neg = n_id - n_pos

    # Get labels for id_pool without loading images repeatedly:
    # id_pool is a subset; we can infer labels from ds_tgt_raw.base.targets using indices_known.
    # But _CIFARSubset hides indices; so simplest is to iterate labels only (still small: ~2000).
    id_labels = np.empty(len(id_pool), dtype=np.int64)
    for i in range(len(id_pool)):
        _, y = id_pool[i]
        id_labels[i] = int(y)

    idx_neg = np.where(id_labels == 0)[0]
    idx_pos = np.where(id_labels == 1)[0]

    rng = np.random.default_rng(args.seed)
    sel_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False).tolist()
    sel_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False).tolist()
    sel_ood = rng.choice(np.arange(len(ood_pool)), size=min(n_ood, len(ood_pool)), replace=False).tolist()

    stream_id = Subset(id_pool, sel_neg + sel_pos)
    stream_ood = Subset(ood_pool, sel_ood)

    stream_twoview_train = torch.utils.data.ConcatDataset(
        [
            TwoViewWrapper(stream_id, tgt_weak_t, tgt_strong_t),
            TwoViewWrapper(stream_ood, tgt_weak_t, tgt_strong_t),
        ]
    )
    stream_twoview_eval = torch.utils.data.ConcatDataset(
        [
            TwoViewWrapper(stream_id, tgt_weak_t, tgt_strong_t),
            TwoViewWrapper(stream_ood, tgt_weak_t, tgt_strong_t),
        ]
    )

    loaders = {
        "src_train": DataLoader(src_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "src_val": DataLoader(src_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "oe": DataLoader(oe_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "tgt_stream_train": DataLoader(stream_twoview_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "tgt_stream_eval": DataLoader(stream_twoview_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "tgt_known": DataLoader(
            CIFAR10BinaryOpenSet(args.data_dir, train=False, transform=tgt_weak_t, download=True).subset_known(),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "tgt_id_for_ood": DataLoader(
            CIFAR10BinaryOpenSet(args.data_dir, train=False, transform=tgt_weak_t, download=True).subset_known(),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "tgt_ood": DataLoader(
            CIFAR10BinaryOpenSet(args.data_dir, train=False, transform=tgt_weak_t, download=True).subset_unknown(),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }
    return loaders


# -----------------------------
# Main
# -----------------------------
def run(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    if args.dataset == "synthetic":
        loaders = build_synthetic(args)
        backbone = MLP2D(in_dim=2, hidden=64, feat_dim=args.feat_dim)
    elif args.dataset == "cifar10":
        loaders = build_cifar10(args)
        backbone = SmallCNN(feat_dim=args.feat_dim)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model = RobustBinaryModel(backbone=backbone, feat_dim=args.feat_dim)

    # Source prior pi_s
    ys_all: List[np.ndarray] = []
    for _, y in loaders["src_train"]:
        ys_all.append(y.numpy())
    ys_all_np = np.concatenate(ys_all)
    pi_source = np.array([(ys_all_np == 0).mean(), (ys_all_np == 1).mean()], dtype=np.float32)

    # 1) Train source
    train_source(
        model=model,
        loader_known=loaders["src_train"],
        loader_oe=loaders["oe"] if args.use_oe else None,
        device=device,
        epochs=args.epochs_src,
        lr=args.lr_src,
        weight_decay=args.weight_decay,
        lambda_ood=args.lambda_ood_src,
        lambda_oe_entropy=args.lambda_oe_entropy,
    )

    # 2) Calibrate
    T = fit_temperature_adam(model, loaders["src_val"], device)
    tau = select_tau_val(model, loaders["src_val"], device, T)
    ood_thr = select_ood_threshold_from_known(model, loaders["src_val"], device, known_false_reject=args.known_false_reject)

    calib = Calibration(temperature=T, tau=tau, ood_threshold=ood_thr, pi_source=pi_source)
    print(f"[calib] T={T:.3f} tau={tau:.3f} ood_thr={ood_thr:.3f} pi_source={pi_source}")

    # 3) Baseline: source-only
    y_true, y_prob = predict_known_posprob(model, loaders["tgt_known"], device, T)
    m_src = metrics_binary(y_true, y_prob, tau)
    id_scores, ood_scores = ood_scores_knownprob(model, loaders["tgt_id_for_ood"], loaders["tgt_ood"], device)
    m_ood = metrics_ood(id_scores, ood_scores)
    ys_stream, yp_stream = predict_stream_3c(model, loaders["tgt_stream_eval"], device, T, tau, ood_thr)
    m_stream = metrics_stream_3c(ys_stream, yp_stream)

    print("[source_only] tgt_known:", m_src)
    print("[source_only] OOD:", m_ood)
    print("[source_only] stream:", m_stream)

    # 4) EM label shift correction (no parameter update)
    pi_t_em = estimate_pi_target(model, loaders["tgt_stream_eval"], device, T, ood_thr, pi_source, max_batches=100)
    print(f"[em_labelshift] estimated pi_t={pi_t_em}")

    # Evaluate adjusted posteriors on target known test
    model.eval()
    ys: List[np.ndarray] = []
    probs_adj: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loaders["tgt_known"]:
            x = x.to(device)
            logits, _, _ = model(x)
            p = softmax_T(logits, T).cpu().numpy()
            p_adj = adjust_posteriors(p, pi_source, pi_t_em)
            ys.append(y.numpy())
            probs_adj.append(p_adj[:, 1])
    y_true = np.concatenate(ys)
    y_prob_adj = np.concatenate(probs_adj)
    m_em = metrics_binary(y_true, y_prob_adj, tau)
    print("[em_labelshift] tgt_known:", m_em)

    # 5) Ours: robust adaptation
    adapted, pi_t = adapt_ours(
        student=model,
        stream_loader_train=loaders["tgt_stream_train"],
        device=device,
        calib=calib,
        epochs=args.epochs_tta,
        lr=args.lr_tta,
        ema=args.ema,
        pseudo_th=args.pseudo_th,
        lambda_cons=args.lambda_consistency,
        lambda_unknown_entropy=args.lambda_unknown_entropy,
        lambda_ood_pos=args.lambda_ood_tta,
        lambda_ood_known=args.lambda_ood_known,
        unk_th=args.unk_th,
        anchor_ood_th=args.anchor_ood_th,
        known_conf_th=args.known_conf_th,
        update_pi_every=args.update_pi_every,
    )

    y_true, y_prob = predict_known_posprob(adapted, loaders["tgt_known"], device, T)
    m_ours = metrics_binary(y_true, y_prob, tau)
    id_scores, ood_scores = ood_scores_knownprob(adapted, loaders["tgt_id_for_ood"], loaders["tgt_ood"], device)
    m_ood2 = metrics_ood(id_scores, ood_scores)
    ys_stream, yp_stream = predict_stream_3c(adapted, loaders["tgt_stream_eval"], device, T, tau, ood_thr)
    m_stream2 = metrics_stream_3c(ys_stream, yp_stream)

    print(f"[ours] estimated pi_t={pi_t}")
    print("[ours] tgt_known:", m_ours)
    print("[ours] OOD:", m_ood2)
    print("[ours] stream:", m_stream2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "synthetic"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    # Scenario knobs
    p.add_argument("--pi_target", type=float, default=0.2, help="Target P(y=1 | known)")
    p.add_argument("--alpha_ood", type=float, default=0.2, help="Fraction of unknown/OOD in target stream")

    # Sizes
    p.add_argument("--n_src_train", type=int, default=4000)
    p.add_argument("--n_src_val", type=int, default=1000)
    p.add_argument("--n_oe", type=int, default=2000)
    p.add_argument("--n_tgt_stream", type=int, default=4000)
    p.add_argument("--n_tgt_test", type=int, default=2000)

    # Model
    p.add_argument("--feat_dim", type=int, default=128)

    # Source training
    p.add_argument("--epochs_src", type=int, default=20)
    p.add_argument("--lr_src", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--use_oe", action="store_true")
    p.add_argument("--lambda_ood_src", type=float, default=1.0)
    p.add_argument("--lambda_oe_entropy", type=float, default=0.1)

    # Calibration
    p.add_argument("--known_false_reject", type=float, default=0.05, help="Known rejection rate to set OOD threshold")

    # Target adaptation
    p.add_argument("--epochs_tta", type=int, default=5)
    p.add_argument("--lr_tta", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--pseudo_th", type=float, default=0.9)
    p.add_argument("--lambda_consistency", type=float, default=1.0)
    p.add_argument("--lambda_unknown_entropy", type=float, default=0.1)
    p.add_argument("--lambda_ood_tta", type=float, default=0.5)
    # ---- Stabilizers for open-set TTA (prevents collapse-to-unknown) ----
    p.add_argument("--unk_th", type=float, default=0.5,
                   help="Only treat samples as *unknown* for adaptation if p_ood >= max(ood_thr, unk_th).")
    p.add_argument("--anchor_ood_th", type=float, default=0.2,
                   help="Max p_ood to consider a sample as a likely-known anchor for the OOD head.")
    p.add_argument("--known_conf_th", type=float, default=0.7,
                   help="Min class confidence (max softmax prob) to use a sample as a known anchor for the OOD head.")
    p.add_argument("--lambda_ood_known", type=float, default=0.5,
                   help="Weight for the known-anchor BCE loss (push p_ood->0 on likely-known samples during TTA).")
    p.add_argument("--update_pi_every", type=int, default=100)

    p.add_argument("--batch_size", type=int, default=128)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
