#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V13: Budget Supervised Sweep (PLM) with V11-style file reading
--------------------------------------------------------------
- Read PubHealth train/val from default relative paths (like V11).
- Read COVID true/fake from default relative paths (like V11).
- Inductive split by UNIQUE text (no target-test leakage).
- Train source once, then for each budget B:
    * sample B labeled target-train UNIQUE texts (stratified),
    * finetune from the same source checkpoint,
    * evaluate on fixed target-test.
- Only use argmax accuracy/F1 (no thr/matchprior).

Run:
  python train_v13_budget_sup_v11io.py
  python train_v13_budget_sup_v11io.py --model_name roberta-base --tapt_steps 1000
"""

from __future__ import annotations

import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    set_seed,
)

# ---------------------------- Repro ------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ---------------------------- CSV readers (copied from V11) ------------------

def _coerce_label_to_int01(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        s = s.replace({"true": "0", "false": "1", "real": "0", "fake": "1"})
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")

def read_pubhealth_csv(path: str) -> Tuple[List[str], List[int]]:
    """
    PubHealth: claim + main_text, label {0:true, 1:false} in source files (as in your V11).
    We convert it into "target/covid semantics": {1:true, 0:false} by flipping: y = 1 - y_source.
    """
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    need_cols = {"claim", "main_text", "label"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")

    y_num = _coerce_label_to_int01(df["label"])
    valid = y_num.isin([0, 1])
    if not bool(valid.all()):
        invalid_vals = df.loc[~valid, "label"].value_counts(dropna=False).head(10)
        print(f"[Warn] PubHealth labels not in {{0,1}}. Will drop {int((~valid).sum())} rows. Examples:\n{invalid_vals}")

    df = df.loc[valid].copy()
    y_source = y_num.loc[valid].astype(int).values  # 0 true, 1 false (source semantics)
    y = (1 - y_source).astype(int)                  # 1 true, 0 false (target semantics)
    texts = (df["claim"].fillna("").astype(str) + " " + df["main_text"].fillna("").astype(str)).tolist()
    return texts, y.tolist()

def read_covid_csv_text_and_optional_binary_label(path: str) -> Tuple[List[str], Optional[List[int]], Optional[str]]:
    """
    COVID csv:
      - must contain 'Text'
      - may contain 'Binary Label'/'Label' (0/1)
    returns texts, optional labels, and label col name.
    """
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "Text" not in df.columns:
        raise ValueError(f"{path} missing column 'Text'. Found: {list(df.columns)}")
    texts = df["Text"].fillna("").astype(str).tolist()

    col_lower_map = {str(c).strip().lower(): c for c in df.columns}
    candidates = ["binary label", "binary_label", "binarylabel", "label"]
    label_col = None
    for cand in candidates:
        if cand in col_lower_map:
            label_col = col_lower_map[cand]
            break
    if label_col is None:
        return texts, None, None

    s = df[label_col]
    if s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        ss = ss.replace({"true": "1", "false": "0", "real": "1", "fake": "0"})
        y_num = pd.to_numeric(ss, errors="coerce")
    else:
        y_num = pd.to_numeric(s, errors="coerce")

    if not bool(y_num.isin([0, 1]).all()):
        return texts, None, None

    return texts, y_num.astype(int).tolist(), str(label_col)

# ---------------------------- Inductive split (unique text) ------------------

def split_target_by_unique_text(
    texts: List[str],
    labels: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Split by unique text to prevent leakage.
    Returns train_idx, test_idx (row indices in original list), and mapping text->label (unique aggregated).
    """
    assert len(texts) == len(labels)
    df = pd.DataFrame({"Text": list(texts), "y": labels.astype(int)})

    # aggregate label per unique text (in case duplicates)
    grp = df.groupby("Text")["y"].agg(lambda s: int(round(s.mean()))).reset_index()
    text_to_label = dict(zip(grp["Text"].tolist(), grp["y"].astype(int).tolist()))

    y_u = grp["y"].values.astype(int)
    idx_u = np.arange(len(grp))

    rng = np.random.RandomState(seed)
    idx0 = idx_u[y_u == 0]
    idx1 = idx_u[y_u == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(round(len(idx0) * test_ratio))
    n1_test = int(round(len(idx1) * test_ratio))

    test_u = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_u = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(test_u)
    rng.shuffle(train_u)

    test_texts = set(grp.loc[test_u, "Text"].tolist())
    train_texts = set(grp.loc[train_u, "Text"].tolist())

    train_idx = df.index[df["Text"].isin(train_texts)].values
    test_idx = df.index[df["Text"].isin(test_texts)].values

    return train_idx.astype(np.int64), test_idx.astype(np.int64), text_to_label

# ---------------------------- Dataset ----------------------------------------

class TextClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.labels = labels
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item

class TextOnlyDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i: int):
        return {k: torch.tensor(v[i]) for k, v in self.enc.items()}

# ---------------------------- R-Drop Trainer ---------------------------------

class RDropTrainer(Trainer):
    def __init__(self, *args, rdrop_alpha=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = float(rdrop_alpha)

    # ✅ 关键：加上 **kwargs（或 num_items_in_batch=None）来兼容新 Trainer
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # 不把 labels 传进 model，避免模型内部重复计算 loss
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        out1 = model(**model_inputs)
        out2 = model(**model_inputs)

        logits1 = out1.logits if hasattr(out1, "logits") else out1[0]
        logits2 = out2.logits if hasattr(out2, "logits") else out2[0]

        ce1 = F.cross_entropy(logits1, labels)
        ce2 = F.cross_entropy(logits2, labels)

        logp1 = F.log_softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        kl12 = F.kl_div(logp1, p2, reduction="batchmean")

        logp2 = F.log_softmax(logits2, dim=-1)
        p1 = F.softmax(logits1, dim=-1)
        kl21 = F.kl_div(logp2, p1, reduction="batchmean")

        loss = 0.5 * (ce1 + ce2) + 0.5 * self.rdrop_alpha * (kl12 + kl21)

        return (loss, out1) if return_outputs else loss

# ---------------------------- Metrics / sampling -----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
    }

def stratified_sample(texts: List[str], labels: List[int], n: int, seed: int) -> Tuple[List[str], List[int]]:
    if n <= 0:
        return [], []
    n = min(n, len(texts))
    rng = np.random.RandomState(seed)
    y = np.asarray(labels, dtype=np.int64)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0 = n // 2
    n1 = n - n0
    take0 = min(n0, len(idx0))
    take1 = min(n1, len(idx1))

    if take0 + take1 < n:
        rest = n - (take0 + take1)
        if (len(idx0) - take0) > (len(idx1) - take1):
            take0 = min(len(idx0), take0 + rest)
        else:
            take1 = min(len(idx1), take1 + rest)

    sel = np.concatenate([idx0[:take0], idx1[:take1]])
    rng.shuffle(sel)

    return [texts[i] for i in sel.tolist()], [int(labels[i]) for i in sel.tolist()]

def train_val_split(texts: List[str], labels: List[int], val_ratio: float, seed: int) -> Tuple[List[str], List[int], List[str], List[int]]:
    if len(texts) < 8:
        return texts, labels, [], []
    try:
        tr_t, va_t, tr_y, va_y = train_test_split(
            texts, labels, test_size=val_ratio, random_state=seed, stratify=labels
        )
        return list(tr_t), list(tr_y), list(va_t), list(va_y)
    except Exception:
        return texts, labels, [], []

def hyperparams_by_budget(B: int) -> Dict[str, Any]:
    if B <= 20:
        return dict(lr=5e-5, epochs=30, max_len=256, rdrop_alpha=8.0)
    if B <= 50:
        return dict(lr=2e-5, epochs=25, max_len=256, rdrop_alpha=7.0)
    if B <= 200:
        return dict(lr=2e-5, epochs=20, max_len=256, rdrop_alpha=6.0)
    if B <= 1000:
        return dict(lr=2e-5, epochs=12, max_len=256, rdrop_alpha=5.0)
    return dict(lr=2e-5, epochs=8, max_len=256, rdrop_alpha=4.0)

def _get_base(model):
    for attr in ["roberta", "distilbert", "bert", "deberta", "deberta_v2", "xlm_roberta"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "base_model"):
        return model.base_model
    return None

def apply_freeze_policy(model, budget: int):
    base = _get_base(model)
    if base is None:
        return

    if budget <= 20:
        for p in base.parameters():
            p.requires_grad = False
        return

    if budget <= 50:
        if hasattr(base, "embeddings"):
            for p in base.embeddings.parameters():
                p.requires_grad = False

        layers = None
        if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            layers = base.encoder.layer
        elif hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
            layers = base.transformer.layer

        if layers is not None:
            for i in range(min(3, len(layers))):
                for p in layers[i].parameters():
                    p.requires_grad = False

# ---------------------------- TAPT (optional) --------------------------------

def run_tapt_if_needed(
    base_model: str,
    target_texts: List[str],
    out_dir: Path,
    seed: int,
    steps: int,
    max_len: int,
    batch_size: int,
    lr: float,
) -> str:
    if steps <= 0:
        return base_model

    tapt_dir = out_dir / f"tapt_steps{steps}_seed{seed}"
    tapt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{now_str()}] [TAPT] start: steps={steps} -> {tapt_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    mlm_model = AutoModelForMaskedLM.from_pretrained(base_model)

    ds = TextOnlyDataset(target_texts, tokenizer, max_len=max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    tr_args = TrainingArguments(
        output_dir=str(tapt_dir),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        max_steps=steps,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=max(200, steps // 2),
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=mlm_model, args=tr_args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(str(tapt_dir))
    tokenizer.save_pretrained(str(tapt_dir))
    print(f"[{now_str()}] [TAPT] done: {tapt_dir}")
    return str(tapt_dir)

# ---------------------------- Train / Eval -----------------------------------

def train_source_once(
    model_init: str,
    src_train_texts: List[str],
    src_train_y: List[int],
    src_val_texts: List[str],
    src_val_y: List[int],
    out_dir: Path,
    seed: int,
    max_len: int,
    batch_size: int,
) -> str:
    ckpt_dir = out_dir / f"source_ckpt_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_init, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_init, num_labels=2)

    train_ds = TextClsDataset(src_train_texts, src_train_y, tokenizer, max_len=max_len)
    val_ds = TextClsDataset(src_val_texts, src_val_y, tokenizer, max_len=max_len)

    tr_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(64, batch_size * 4),
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print(f"[{now_str()}] [Source] train start -> {ckpt_dir}")
    trainer.train()
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"[{now_str()}] [Source] done -> {ckpt_dir}")
    return str(ckpt_dir)

def eval_model_on_target(
    model_dir: str,
    test_texts: List[str],
    test_y: List[int],
    batch_size: int,
    max_len: int,
    seed: int,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

    test_ds = TextClsDataset(test_texts, test_y, tokenizer, max_len=max_len)

    ev_args = TrainingArguments(
        output_dir=str(Path(model_dir) / "_eval_tmp"),
        per_device_eval_batch_size=max(64, batch_size * 4),
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(model=model, args=ev_args, eval_dataset=test_ds, compute_metrics=compute_metrics)
    metrics = trainer.evaluate()
    return {
        "acc": float(metrics.get("eval_accuracy", float("nan"))),
        "f1": float(metrics.get("eval_f1", float("nan"))),
    }

def finetune_on_budget(
    source_ckpt: str,
    budget_raw: int,
    train_texts: List[str],
    train_y: List[int],
    val_texts: List[str],
    val_y: List[int],
    test_texts: List[str],
    test_y: List[int],
    out_dir: Path,
    seed: int,
    hp: Dict[str, Any],
    batch_size: int,
) -> Dict[str, Any]:
    bdir = out_dir / f"budget_{budget_raw}_seed{seed}"
    bdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source_ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(source_ckpt, num_labels=2)

    apply_freeze_policy(model, budget=len(train_texts))

    max_len = int(hp["max_len"])
    train_ds = TextClsDataset(train_texts, train_y, tokenizer, max_len=max_len)
    test_ds = TextClsDataset(test_texts, test_y, tokenizer, max_len=max_len)

    use_val = len(val_texts) > 0
    val_ds = TextClsDataset(val_texts, val_y, tokenizer, max_len=max_len) if use_val else None

    per_train_bs = min(batch_size, max(4, len(train_texts)))
    per_eval_bs = max(64, per_train_bs * 4)

    tr_args = TrainingArguments(
        output_dir=str(bdir),
        per_device_train_batch_size=per_train_bs,
        per_device_eval_batch_size=per_eval_bs,
        learning_rate=float(hp["lr"]),
        num_train_epochs=float(hp["epochs"]),
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        evaluation_strategy="epoch" if use_val else "no",
        save_strategy="epoch" if use_val else "no",
        load_best_model_at_end=True if use_val else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = RDropTrainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics if use_val else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if use_val else None,
        rdrop_alpha=float(hp["rdrop_alpha"]),
    )

    print(f"[{now_str()}] [B={budget_raw:>5}] finetune start | hp={hp}")
    trainer.train()

    eval_trainer = Trainer(
        model=trainer.model,
        args=TrainingArguments(
            output_dir=str(bdir / "_eval_tmp"),
            per_device_eval_batch_size=per_eval_bs,
            fp16=torch.cuda.is_available(),
            seed=seed,
            report_to="none",
            dataloader_num_workers=0,
        ),
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    metrics = eval_trainer.evaluate()
    acc = float(metrics.get("eval_accuracy", float("nan")))
    f1 = float(metrics.get("eval_f1", float("nan")))

    trainer.save_model(str(bdir))
    tokenizer.save_pretrained(str(bdir))

    print(f"[{now_str()}] [B={budget_raw:>5}] done | acc={acc:.4f} f1={f1:.4f} saved={bdir}")
    return {"acc": acc, "f1": f1, "model_dir": str(bdir)}

# ---------------------------- Main -------------------------------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    # V11-style defaults
    parser.add_argument("--pubhealth_train", type=str, default=str(base_dir / "./pubhealth/pubhealth_train_clean.csv"))
    parser.add_argument("--pubhealth_val", type=str, default=str(base_dir / "./pubhealth/pubhealth_validation_clean.csv"))
    parser.add_argument("--covid_true", type=str, default=str(base_dir / "../covid/trueNews.csv"))
    parser.add_argument("--covid_fake", type=str, default=str(base_dir / "../covid/fakeNews.csv"))

    parser.add_argument("--target_test_ratio", type=float, default=0.2)
    parser.add_argument("--target_split_seed", type=int, default=42)

    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--budgets", type=str, default="0,10,20,50,100,200,500,1000,2000,4000,-1")

    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--tapt_steps", type=int, default=0)
    parser.add_argument("--tapt_lr", type=float, default=5e-5)
    parser.add_argument("--tapt_batch_size", type=int, default=16)
    parser.add_argument("--tapt_max_len", type=int, default=256)

    parser.add_argument("--out_dir", type=str, default=str(base_dir / "v13_sup_runs"))
    parser.add_argument("--results_csv", type=str, default=str(base_dir / "v13_budget_results.csv"))

    parser.add_argument("--cpu", action="store_true")

    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    budgets = parse_int_list(args.budgets)

    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps and not args.cpu else "cpu")
    print("mps_available=", use_mps, "device=", device)

    print(f"[V13] device={device}")
    print(f"[V13] seeds={seeds}")
    print(f"[V13] budgets={budgets} (-1 means full)")
    print(f"[V13] split_seed={args.target_split_seed} test_ratio={args.target_test_ratio}")
    print(f"[V13] model={args.model_name} | TAPT steps={args.tapt_steps}")

    # ---------- Load source ----------
    src_train_texts, src_train_y = read_pubhealth_csv(args.pubhealth_train)
    src_val_texts, src_val_y = read_pubhealth_csv(args.pubhealth_val)
    print(f"[Data] Source train={len(src_train_texts)} | val={len(src_val_texts)}")

    # ---------- Load target ----------
    true_texts, true_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_true)
    fake_texts, fake_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(args.covid_fake)

    all_texts = true_texts + fake_texts
    y_file = np.array([1] * len(true_texts) + [0] * len(fake_texts), dtype=np.int64)

    if true_labels_opt is not None and fake_labels_opt is not None:
        y_col = np.array(true_labels_opt + fake_labels_opt, dtype=np.int64)
        agree = float((y_col == y_file).mean())
        agree_flip = float(((1 - y_col) == y_file).mean())
        if agree_flip > agree:
            y_col = 1 - y_col
        y = y_col
        print("[Data] Target label source: column (Binary Label / Label)")
    else:
        y = y_file
        print("[Data] Target label source: file membership (trueNews=1, fakeNews=0)")

    print(f"[Data] Target rows={len(all_texts)} | dist={pd.Series(y).value_counts().to_dict()}")

    # ---------- Inductive split by unique text ----------
    train_idx, test_idx, text_to_label = split_target_by_unique_text(
        all_texts, y, test_ratio=args.target_test_ratio, seed=args.target_split_seed
    )

    tgt_train_unique_full = list(dict.fromkeys([all_texts[i] for i in train_idx]))
    tgt_train_unique_labels = [int(text_to_label[t]) for t in tgt_train_unique_full]

    tgt_test_texts = [all_texts[i] for i in test_idx.tolist()]
    y_test = y[test_idx].astype(int).tolist()

    print(f"[Split] target-train unique={len(tgt_train_unique_full)} | target-test rows={len(test_idx)}")

    records: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_everything(seed)

        # TAPT on target-train (unlabeled but inductive-safe)
        model_init = run_tapt_if_needed(
            base_model=args.model_name,
            target_texts=tgt_train_unique_full,
            out_dir=out_dir,
            seed=seed,
            steps=int(args.tapt_steps),
            max_len=int(args.tapt_max_len),
            batch_size=int(args.tapt_batch_size),
            lr=float(args.tapt_lr),
        )

        # Train source once
        src_ckpt = train_source_once(
            model_init=model_init,
            src_train_texts=src_train_texts,
            src_train_y=src_train_y,
            src_val_texts=src_val_texts,
            src_val_y=src_val_y,
            out_dir=out_dir,
            seed=seed,
            max_len=int(args.max_len),
            batch_size=int(args.batch_size),
        )

        # Baseline
        base_metrics = eval_model_on_target(
            model_dir=src_ckpt,
            test_texts=tgt_test_texts,
            test_y=y_test,
            batch_size=int(args.batch_size),
            max_len=int(args.max_len),
            seed=seed,
        )
        b0_acc, b0_f1 = base_metrics["acc"], base_metrics["f1"]
        print(f"[B={0:>5}] seed={seed} acc_argmax={b0_acc:.4f} f1={b0_f1:.4f} (source-only)")

        records.append({
            "seed": seed,
            "budget_raw": 0,
            "budget_used": 0,
            "acc": b0_acc,
            "f1": b0_f1,
            "delta_acc_vs_b0": 0.0,
            "model_dir": src_ckpt,
        })

        for B in budgets:
            if B == 0:
                continue

            if B == -1:
                b_texts = tgt_train_unique_full
                b_labels = tgt_train_unique_labels
                budget_used = len(b_texts)
            else:
                b_texts, b_labels = stratified_sample(
                    tgt_train_unique_full, tgt_train_unique_labels, n=B, seed=seed
                )
                budget_used = len(b_texts)

            tr_texts, tr_y, va_texts, va_y = train_val_split(
                b_texts, b_labels, val_ratio=0.2, seed=seed
            )

            hp = hyperparams_by_budget(budget_used)
            res = finetune_on_budget(
                source_ckpt=src_ckpt,
                budget_raw=int(B),
                train_texts=tr_texts,
                train_y=tr_y,
                val_texts=va_texts,
                val_y=va_y,
                test_texts=tgt_test_texts,
                test_y=y_test,
                out_dir=out_dir,
                seed=seed,
                hp=hp,
                batch_size=int(args.batch_size),
            )

            delta = float(res["acc"] - b0_acc)
            print(f"[B={B:>5}] seed={seed} acc_argmax={res['acc']:.4f} f1={res['f1']:.4f} delta_vs_B0={delta:+.4f}")

            records.append({
                "seed": seed,
                "budget_raw": int(B),
                "budget_used": int(budget_used),
                "acc": float(res["acc"]),
                "f1": float(res["f1"]),
                "delta_acc_vs_b0": float(delta),
                "model_dir": str(res["model_dir"]),
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv(args.results_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved results -> {args.results_csv}")

if __name__ == "__main__":
    main()
