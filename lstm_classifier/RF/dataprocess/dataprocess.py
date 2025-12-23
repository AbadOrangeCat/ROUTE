# v11_dataio.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd


# ---------------------------- CSV readers (copied from V11/V13) ------------------

def _coerce_label_to_int01(series: pd.Series) -> pd.Series:
    """
    PubHealth label coercion:
      - object labels allowed: true/false/real/fake
      - PubHealth source semantics in your code: {0:true, 1:false}
    """
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        s = s.replace({"true": "0", "false": "1", "real": "0", "fake": "1"})
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def read_pubhealth_csv(path: str) -> Tuple[List[str], List[int]]:
    """
    PubHealth:
      - columns: claim, main_text, label
      - label in source files: {0:true, 1:false} (as in your V11/V13)
    We convert to "target/covid semantics": {1:true, 0:false} by flipping: y = 1 - y_source.
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
        print(f"[Warn] PubHealth labels not in {{0,1}}. Drop {int((~valid).sum())} rows. Examples:\n{invalid_vals}")

    df = df.loc[valid].copy()
    y_source = y_num.loc[valid].astype(int).values  # 0 true, 1 false (source semantics)
    y = (1 - y_source).astype(int)                  # 1 true, 0 false (target semantics)

    texts = (df["claim"].fillna("").astype(str) + " " + df["main_text"].fillna("").astype(str)).tolist()
    return texts, y.tolist()


def read_covid_csv_text_and_optional_binary_label(path: str) -> Tuple[List[str], Optional[List[int]], Optional[str]]:
    """
    COVID csv:
      - must contain 'Text'
      - may contain label col among: 'Binary Label'/'Label' variants
    returns texts, optional labels, and label col name.

    If label col exists but cannot be fully parsed to {0,1}, return (texts, None, None)
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
        # COVID/target semantics in your code: {1:true/real, 0:false/fake}
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
    Split by UNIQUE text to prevent leakage.
    Returns:
      - train_idx: row indices in original list for train split
      - test_idx: row indices in original list for test split
      - text_to_label: dict mapping UNIQUE text -> aggregated label

    Aggregation: label per unique text = round(mean(labels_of_duplicates))
    """
    assert len(texts) == len(labels)
    df = pd.DataFrame({"Text": list(texts), "y": labels.astype(int)})

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


# ---------------------------- Wrapper: one-call prepare ----------------------

def prepare_pubhealth_covid_v11io(
    pubhealth_train: str,
    pubhealth_val: str,
    covid_true: str,
    covid_fake: str,
    target_test_ratio: float = 0.2,
    target_split_seed: int = 42,
    out_text_col: str = "text",
    out_label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Prepare datasets with EXACT V13/V11 IO + inductive unique-text split.

    Returns:
      src_train_df: columns [text,label]  (label: 1=true,0=fake)
      src_val_df:   columns [text,label]
      tgt_unl_df:   columns [text]        (target-train UNIQUE only; inductive-safe)
      tgt_test_df:  columns [text,label]  (row-level test; may contain duplicates)
      meta: dict (stats + mapping source)
    """
    # ---------- Load source ----------
    src_train_texts, src_train_y = read_pubhealth_csv(pubhealth_train)
    src_val_texts, src_val_y = read_pubhealth_csv(pubhealth_val)

    # ---------- Load target ----------
    true_texts, true_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(covid_true)
    fake_texts, fake_labels_opt, _ = read_covid_csv_text_and_optional_binary_label(covid_fake)

    all_texts = true_texts + fake_texts
    y_file = np.array([1] * len(true_texts) + [0] * len(fake_texts), dtype=np.int64)

    # Prefer label column if BOTH files have label column and are parseable
    if true_labels_opt is not None and fake_labels_opt is not None:
        y_col = np.array(true_labels_opt + fake_labels_opt, dtype=np.int64)
        agree = float((y_col == y_file).mean())
        agree_flip = float(((1 - y_col) == y_file).mean())
        if agree_flip > agree:
            y_col = 1 - y_col
        y = y_col
        y_source = "column"
    else:
        y = y_file
        y_source = "file_membership"

    # ---------- Inductive split by unique text ----------
    train_idx, test_idx, text_to_label = split_target_by_unique_text(
        all_texts, y, test_ratio=target_test_ratio, seed=target_split_seed
    )

    # target-train UNIQUE texts only (inductive safe)
    tgt_train_unique_full = list(dict.fromkeys([all_texts[i] for i in train_idx]))
    tgt_train_unique_labels = [int(text_to_label[t]) for t in tgt_train_unique_full]

    # target-test rows (keep row-level, like your V13)
    tgt_test_texts = [all_texts[i] for i in test_idx.tolist()]
    tgt_test_y = y[test_idx].astype(int).tolist()

    src_train_df = pd.DataFrame({out_text_col: src_train_texts, out_label_col: src_train_y})
    src_val_df = pd.DataFrame({out_text_col: src_val_texts, out_label_col: src_val_y})

    tgt_unl_df = pd.DataFrame({out_text_col: tgt_train_unique_full})
    tgt_test_df = pd.DataFrame({out_text_col: tgt_test_texts, out_label_col: tgt_test_y})

    meta = {
        "source_train_n": len(src_train_df),
        "source_val_n": len(src_val_df),
        "target_rows_n": int(len(all_texts)),
        "target_unique_train_n": int(len(tgt_unl_df)),
        "target_test_rows_n": int(len(tgt_test_df)),
        "target_label_source": y_source,
        "target_label_dist_all_rows": pd.Series(y).value_counts().to_dict(),
        "target_label_dist_test_rows": pd.Series(tgt_test_y).value_counts().to_dict(),
        "target_label_dist_unique_train": pd.Series(tgt_train_unique_labels).value_counts().to_dict(),
    }
    return src_train_df, src_val_df, tgt_unl_df, tgt_test_df, meta


def default_paths_like_v11(script_file: str) -> Dict[str, str]:
    """
    Helper to reproduce V11-style relative defaults:
      pubhealth in ./pubhealth/
      covid in ../covid/
    """
    base_dir = Path(script_file).resolve().parent
    return {
        "pubhealth_train": str(base_dir / "./pubhealth/pubhealth_train_clean.csv"),
        "pubhealth_val": str(base_dir / "./pubhealth/pubhealth_validation_clean.csv"),
        "covid_true": str(base_dir / "../covid/trueNews.csv"),
        "covid_fake": str(base_dir / "../covid/fakeNews.csv"),
    }
