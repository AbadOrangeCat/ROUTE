#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V12 (HiCAT-CL): Contrastive Pretraining + Source Robust Training + Safe FixMatch (Inductive)
===========================================================================================

This script is a fresh solution (can replace V10/V11) designed for your setting:

  Source domain (labeled): PubHealth (non-COVID medical fake news)
    - text input: claim + main_text
    - label: 0=true, 1=false

  Target domain (unlabeled for training): COVID
    - text input: Text
    - labels (if present) are ONLY for evaluation.

Goal:
  - Use ONLY labeled source data + unlabeled target-train (budgeted) to improve target-test performance.
  - Strictly inductive: target TEST texts are never used in vocab, pretraining, adaptation, thresholds, etc.
  - Provide a "bigger jump" than pure threshold tricks by adding:
      (A) Unsupervised contrastive pretraining (SimCLR-style) on unlabeled mixture (source + target-train)
      (B) Topic-environment robust supervised training on source (GroupDRO over pseudo environments)
      (C) Safe FixMatch (masked pseudo-labeling) on target-train with EMA teacher

Why this can help compared to your V11:
  - Your previous UDA mainly pushes alignment/consistency directly; if conditional shift exists it can hurt.
  - Contrastive pretraining on target-train helps the encoder learn target vocabulary/style WITHOUT using labels.
  - GroupDRO reduces reliance on topic-specific shortcuts in the parent category (medical).
  - FixMatch is "selective": only high-confidence target samples influence training.

Run examples:
  python train_v12_hicat_cl.py
  python train_v12_hicat_cl.py --seeds 42,43,44,45,46 --target_budgets 0,50,200,500,1000,-1
  python train_v12_hicat_cl.py --budget_aware --use_calibration --vocab_mode source_plus_target

Outputs:
  - v12_budget_results.csv
  - v12_budget_summary.csv
  - optional per-run predictions CSV

Notes:
  - No transformers are used (works in minimal environments).
  - If you want faster runs: reduce --cl_epochs, --pretrain_epochs, --uda_epochs.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise RuntimeError(
        "Failed to import PyTorch. Please install a working PyTorch build (CPU-only is fine).\n"
        f"Original error: {e}"
    )

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # type: ignore


# ----------------------------- Repro -----------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- Tokenization ----------------------------------


URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")

# Char vocab: ASCII 32..126 -> ids 1..95, pad=0, unk=96
CHAR_PAD = 0
CHAR_UNK = 96
CHAR_MIN = 32
CHAR_MAX = 126


def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    text = str(text).lower()
    text = URL_RE.sub(" URLTOKEN ", text)
    tokens: List[str] = []
    for tok in TOKEN_RE.findall(text):
        if tok == "URLTOKEN":
            tokens.append("<url>")
        elif tok.isdigit():
            tokens.append("<num>")
        else:
            tokens.append(tok)
    return tokens


def char_encode(text: str, max_len: int) -> List[int]:
    if text is None:
        text = ""
    s = str(text).lower()
    s = URL_RE.sub(" URLTOKEN ", s)
    ids: List[int] = []
    for ch in s[:max_len]:
        oc = ord(ch)
        if CHAR_MIN <= oc <= CHAR_MAX:
            ids.append(oc - CHAR_MIN + 1)
        else:
            ids.append(CHAR_UNK)
    if len(ids) == 0:
        ids = [CHAR_UNK]
    return ids


def compute_style_feats(text: str) -> np.ndarray:
    """
    Domain-robust stylistic features (topic-agnostic).
    Keep it small and stable; do NOT use external NLP libs.
    """
    if text is None:
        text = ""
    s = str(text)
    s_low = s.lower()
    n_char = len(s)
    toks = tokenize(s)
    n_tok = len(toks)

    n_url = int(bool(URL_RE.search(s)))
    n_excl = s.count("!")
    n_quest = s.count("?")
    n_quote = s.count('"') + s.count("'")
    n_digit = sum(ch.isdigit() for ch in s)
    n_upper = sum(ch.isupper() for ch in s)
    n_punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)

    # ratios (avoid div0)
    denom_char = max(1, n_char)
    denom_tok = max(1, n_tok)

    feats = np.array(
        [
            float(n_char),
            float(n_tok),
            float(n_url),
            float(n_excl),
            float(n_quest),
            float(n_quote),
            float(n_digit) / denom_char,
            float(n_upper) / denom_char,
            float(n_punct) / denom_char,
            float(sum(t == "<num>" for t in toks)) / denom_tok,
            float(sum(t == "<url>" for t in toks)) / denom_tok,
        ],
        dtype=np.float32,
    )
    # log-scale for length-like features
    feats[0] = math.log1p(feats[0])
    feats[1] = math.log1p(feats[1])
    feats[3] = math.log1p(feats[3])
    feats[4] = math.log1p(feats[4])
    feats[5] = math.log1p(feats[5])
    return feats


# ----------------------------- Vocabulary ------------------------------------


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in tokens]


def build_vocab(texts: Sequence[str], min_freq: int = 2, max_size: int = 50000) -> Vocab:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    specials = ["<pad>", "<unk>", "<url>", "<num>"]
    stoi: Dict[str, int] = {s: i for i, s in enumerate(specials)}
    itos: List[str] = list(specials)

    for tok, freq in counter.most_common():
        if tok in stoi:
            continue
        if freq < min_freq:
            break
        if len(itos) >= max_size:
            break
        stoi[tok] = len(itos)
        itos.append(tok)

    return Vocab(stoi=stoi, itos=itos, pad_id=stoi["<pad>"], unk_id=stoi["<unk>"])


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(seqs)
    lengths = [min(len(s), max_len) for s in seqs]
    L = max(1, min(max(lengths), max_len))

    input_ids = torch.full((batch_size, L), pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, L), dtype=torch.float32)

    for i, s in enumerate(seqs):
        s = list(s)[:L]
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1.0

    return input_ids, mask


# ----------------------------- Augmentations ---------------------------------


def random_token_dropout(ids: List[int], p: float, unk_id: int, rng: random.Random) -> List[int]:
    if p <= 0:
        return ids
    return [unk_id if rng.random() < p else tid for tid in ids]


def random_deletion(ids: List[int], p: float, rng: random.Random) -> List[int]:
    if p <= 0 or len(ids) <= 2:
        return ids
    kept = [tid for tid in ids if rng.random() > p]
    return kept if len(kept) >= 2 else ids


def random_swap(ids: List[int], n_swaps: int, rng: random.Random) -> List[int]:
    if n_swaps <= 0 or len(ids) <= 2:
        return ids
    out = ids[:]
    L = len(out)
    for _ in range(n_swaps):
        i, j = rng.randrange(L), rng.randrange(L)
        out[i], out[j] = out[j], out[i]
    return out


def weak_augment_word(ids: List[int], unk_id: int, rng: random.Random) -> List[int]:
    out = random_token_dropout(ids, p=0.05, unk_id=unk_id, rng=rng)
    out = random_swap(out, n_swaps=1, rng=rng)
    return out


def strong_augment_word(ids: List[int], unk_id: int, rng: random.Random) -> List[int]:
    out = random_deletion(ids, p=0.10, rng=rng)
    out = random_token_dropout(out, p=0.15, unk_id=unk_id, rng=rng)
    out = random_swap(out, n_swaps=3, rng=rng)
    return out


def weak_augment_char(ids: List[int], rng: random.Random) -> List[int]:
    out = [CHAR_UNK if rng.random() < 0.02 else x for x in ids]
    out = random_swap(out, n_swaps=1, rng=rng)
    return out


def strong_augment_char(ids: List[int], rng: random.Random) -> List[int]:
    out = random_deletion(ids, p=0.05, rng=rng)
    out = [CHAR_UNK if rng.random() < 0.08 else x for x in out]
    out = random_swap(out, n_swaps=3, rng=rng)
    return out


# ----------------------------- CSV readers -----------------------------------


def _coerce_label_to_int01(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        s = s.replace({"true": "0", "false": "1", "real": "0", "fake": "1"})
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def read_pubhealth_csv(path: str) -> Tuple[List[str], List[int]]:
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
    labels = y_num.loc[valid].astype(int).tolist()
    texts = (df["claim"].fillna("").astype(str) + " " + df["main_text"].fillna("").astype(str)).tolist()
    return texts, labels


def read_covid_csv_text_and_optional_binary_label(path: str) -> Tuple[List[str], Optional[List[int]], Optional[str]]:
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


# ----------------------------- Inductive split (unique text) ------------------


def split_target_by_unique_text(
    texts: Sequence[str],
    labels: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split target into train/test indices WITHOUT leakage through duplicated texts:
      - group by exact Text string
      - stratify on majority label per unique text
      - assign whole groups to train/test
    """
    assert len(texts) == len(labels)
    df = pd.DataFrame({"Text": list(texts), "y": labels.astype(int)})
    grp = df.groupby("Text")["y"].agg(lambda s: int(round(s.mean()))).reset_index()
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
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


# ----------------------------- Datasets --------------------------------------


class UnlabeledContrastiveDataset(Dataset):
    """
    Unlabeled dataset for contrastive pretraining: returns (weak_view, strong_view) of the SAME text.
    """
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int, max_char_len: int, seed: int):
        self.texts = [str(t) for t in texts]
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len
        self.rng = random.Random(seed)

        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        self.style: List[np.ndarray] = []
        for t in self.texts:
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])
            self.char.append(char_encode(t, max_len=max_char_len)[:max_char_len])
            self.style.append(compute_style_feats(t))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        w = self.word[idx]
        c = self.char[idx]
        st = self.style[idx]
        w1 = weak_augment_word(w, unk_id=self.vocab.unk_id, rng=self.rng)
        w2 = strong_augment_word(w, unk_id=self.vocab.unk_id, rng=self.rng)
        c1 = weak_augment_char(c, rng=self.rng)
        c2 = strong_augment_char(c, rng=self.rng)
        return w1, c1, w2, c2, st


def collate_contrastive(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    w1, c1, w2, c2, st = zip(*batch)
    w1_ids, w1_mask = pad_batch(w1, pad_id=pad_word, max_len=max_len)
    c1_ids, c1_mask = pad_batch(c1, pad_id=pad_char, max_len=max_char_len)
    w2_ids, w2_mask = pad_batch(w2, pad_id=pad_word, max_len=max_len)
    c2_ids, c2_mask = pad_batch(c2, pad_id=pad_char, max_len=max_char_len)
    st_t = torch.tensor(np.stack(st), dtype=torch.float32)
    return w1_ids, w1_mask, c1_ids, c1_mask, w2_ids, w2_mask, c2_ids, c2_mask, st_t


class SourceLabeledEnvDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        env_ids: Sequence[int],
        vocab: Vocab,
        max_len: int,
        max_char_len: int,
    ):
        assert len(texts) == len(labels) == len(env_ids)
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len

        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        self.style: List[np.ndarray] = []
        self.labels: List[int] = []
        self.env: List[int] = []

        for t, y, e in zip(texts, labels, env_ids):
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])
            self.char.append(char_encode(t, max_len=max_char_len)[:max_char_len])
            self.style.append(compute_style_feats(t))
            self.labels.append(int(y))
            self.env.append(int(e))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.word[idx], self.char[idx], self.style[idx], self.labels[idx], self.env[idx]


def collate_source_env(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    w, c, st, y, e = zip(*batch)
    w_ids, w_mask = pad_batch(w, pad_id=pad_word, max_len=max_len)
    c_ids, c_mask = pad_batch(c, pad_id=pad_char, max_len=max_char_len)
    st_t = torch.tensor(np.stack(st), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    e_t = torch.tensor(e, dtype=torch.long)
    return w_ids, w_mask, c_ids, c_mask, st_t, y_t, e_t


class TargetUnlabeledFixMatchDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int, max_char_len: int, seed: int):
        self.texts = [str(t) for t in texts]
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len
        self.rng = random.Random(seed)

        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        self.style: List[np.ndarray] = []
        for t in self.texts:
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])
            self.char.append(char_encode(t, max_len=max_char_len)[:max_char_len])
            self.style.append(compute_style_feats(t))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        w = self.word[idx]
        c = self.char[idx]
        st = self.style[idx]
        w_w = weak_augment_word(w, unk_id=self.vocab.unk_id, rng=self.rng)
        w_s = strong_augment_word(w, unk_id=self.vocab.unk_id, rng=self.rng)
        c_w = weak_augment_char(c, rng=self.rng)
        c_s = strong_augment_char(c, rng=self.rng)
        return w_w, c_w, w_s, c_s, st, idx


def collate_target_fixmatch(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    w_w, c_w, w_s, c_s, st, idxs = zip(*batch)
    w_w_ids, w_w_mask = pad_batch(w_w, pad_id=pad_word, max_len=max_len)
    c_w_ids, c_w_mask = pad_batch(c_w, pad_id=pad_char, max_len=max_char_len)
    w_s_ids, w_s_mask = pad_batch(w_s, pad_id=pad_word, max_len=max_len)
    c_s_ids, c_s_mask = pad_batch(c_s, pad_id=pad_char, max_len=max_char_len)
    st_t = torch.tensor(np.stack(st), dtype=torch.float32)
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    return w_w_ids, w_w_mask, c_w_ids, c_w_mask, w_s_ids, w_s_mask, c_s_ids, c_s_mask, st_t, idxs_t


class InferenceDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int, max_char_len: int):
        self.texts = [str(t) for t in texts]
        self.vocab = vocab
        self.max_len = max_len
        self.max_char_len = max_char_len

        self.word: List[List[int]] = []
        self.char: List[List[int]] = []
        self.style: List[np.ndarray] = []
        for t in self.texts:
            w = vocab.encode(tokenize(t))
            if len(w) == 0:
                w = [vocab.unk_id]
            self.word.append(w[:max_len])
            self.char.append(char_encode(t, max_len=max_char_len)[:max_char_len])
            self.style.append(compute_style_feats(t))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.word[idx], self.char[idx], self.style[idx], idx


def collate_infer(batch, pad_word: int, pad_char: int, max_len: int, max_char_len: int):
    w, c, st, idxs = zip(*batch)
    w_ids, w_mask = pad_batch(w, pad_id=pad_word, max_len=max_len)
    c_ids, c_mask = pad_batch(c, pad_id=pad_char, max_len=max_char_len)
    st_t = torch.tensor(np.stack(st), dtype=torch.float32)
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    return w_ids, w_mask, c_ids, c_mask, st_t, idxs_t


# ----------------------------- Model -----------------------------------------


class AttnPool(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.proj(self.dropout(h)).squeeze(-1)  # (B,L)
        scores = scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        return pooled


class HybridTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int,
        word_hidden: int,
        word_pad: int,
        char_vocab_size: int = 97,
        char_emb_dim: int = 32,
        char_channels: int = 96,
        char_kernels: Tuple[int, ...] = (3, 4, 5),
        char_pad: int = 0,
        char_proj_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=word_pad)
        self.word_gru = nn.GRU(word_emb_dim, word_hidden, batch_first=True, bidirectional=True)
        self.word_attn = AttnPool(word_hidden * 2, dropout=dropout)

        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=char_pad)
        self.char_convs = nn.ModuleList(
            [nn.Conv1d(char_emb_dim, char_channels, kernel_size=k, bias=True) for k in char_kernels]
        )
        self.char_proj = nn.Linear(char_channels * len(char_kernels), char_proj_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_dim = word_hidden * 2 + char_proj_dim

    def forward(
        self,
        word_ids: torch.Tensor,
        word_mask: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> torch.Tensor:
        w = self.dropout(self.word_emb(word_ids))  # (B,L,E)
        h, _ = self.word_gru(w)  # (B,L,2H)
        h = self.dropout(h)
        w_feat = self.word_attn(h, word_mask)  # (B,2H)

        c = self.dropout(self.char_emb(char_ids))  # (B,LC,Cemb)
        c = c.transpose(1, 2)  # (B,Cemb,LC)
        conv_outs: List[torch.Tensor] = []
        for conv in self.char_convs:
            x = torch.relu(conv(c))
            x = torch.max(x, dim=-1).values
            conv_outs.append(x)
        c_feat = torch.cat(conv_outs, dim=1)
        c_feat = self.dropout(c_feat)
        c_feat = torch.relu(self.char_proj(c_feat))

        feat = torch.cat([w_feat, c_feat], dim=1)
        feat = self.dropout(feat)
        return feat


class StyleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: HybridTextEncoder, style_dim: int, style_proj: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.style_mlp = StyleMLP(style_dim, style_proj, dropout=dropout)
        self.layer_norm = nn.LayerNorm(encoder.out_dim + style_proj)
        self.classifier = nn.Linear(encoder.out_dim + style_proj, num_classes)

    def forward(self, w_ids: torch.Tensor, w_mask: torch.Tensor, c_ids: torch.Tensor, c_mask: torch.Tensor, style: torch.Tensor):
        feats = self.encoder(w_ids, w_mask, c_ids, c_mask)
        sproj = self.style_mlp(style)
        z = torch.cat([feats, sproj], dim=1)
        z = self.layer_norm(z)
        logits = self.classifier(z)
        return z, logits


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)


# ----------------------------- Losses ----------------------------------------


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    SimCLR-style contrastive loss for paired batches.
    z1, z2: (B,D)
    """
    if z1.size(0) <= 1:
        return torch.tensor(0.0, device=z1.device)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = (z1 @ z2.t()) / max(1e-6, temperature)  # (B,B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def entropy_torch(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)


# ----------------------------- Temperature scaling ----------------------------


@torch.no_grad()
def collect_logits_labels(model: EncoderClassifier, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, y_list = [], []
    for w_ids, w_mask, c_ids, c_mask, st, y, _e in loader:
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        st = st.to(device)
        y = y.to(device)
        _f, logits = model(w_ids, w_mask, c_ids, c_mask, st)
        logits_list.append(logits.detach())
        y_list.append(y.detach())
    return torch.cat(logits_list, dim=0), torch.cat(y_list, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, device: torch.device, t_min: float, t_max: float) -> float:
    logits = logits.detach()
    labels = labels.detach()

    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_t).clamp(min=t_min, max=t_max)
        loss = ce(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_t).detach().cpu().item())
    return float(np.clip(T, t_min, t_max))


# ----------------------------- Environment building --------------------------


def build_pseudo_envs(
    src_texts: Sequence[str],
    n_envs: int,
    seed: int,
    max_features: int = 30000,
) -> Tuple[np.ndarray, TfidfVectorizer, MiniBatchKMeans]:
    """
    Cluster source texts into pseudo environments by TF-IDF topics.
    Returns env_ids for src_texts and the fitted vectorizer/kmeans.
    """
    n_envs = int(max(2, n_envs))
    vec = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        token_pattern=r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+",
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(list(src_texts))
    km = MiniBatchKMeans(n_clusters=n_envs, random_state=seed, batch_size=2048, n_init=10)
    env = km.fit_predict(X).astype(np.int64)
    return env, vec, km


def predict_envs(texts: Sequence[str], vec: TfidfVectorizer, km: MiniBatchKMeans) -> np.ndarray:
    X = vec.transform(list(texts))
    return km.predict(X).astype(np.int64)


# ----------------------------- Training steps --------------------------------


def eval_acc(model: EncoderClassifier, loader: DataLoader, device: torch.device, calib_T: float = 1.0) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for w_ids, w_mask, c_ids, c_mask, st, y, _e in loader:
            w_ids, w_mask = w_ids.to(device), w_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            st = st.to(device)
            y = y.to(device)
            _f, logits = model(w_ids, w_mask, c_ids, c_mask, st)
            logits = logits / max(1e-6, calib_T)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return float(correct / max(1, total))


def contrastive_pretrain(
    encoder: HybridTextEncoder,
    style_dim: int,
    style_proj: int,
    unlabeled_texts: Sequence[str],
    vocab: Vocab,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> HybridTextEncoder:
    """
    Unsupervised contrastive pretraining on unlabeled_texts.
    We train a temporary (encoder + style_mlp + proj_head) and return the encoder weights.
    """
    if len(unlabeled_texts) == 0 or args.cl_epochs <= 0:
        return encoder

    ds = UnlabeledContrastiveDataset(unlabeled_texts, vocab=vocab, max_len=args.max_len, max_char_len=args.max_char_len, seed=seed + 999)
    loader = DataLoader(
        ds,
        batch_size=args.cl_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_contrastive, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
        drop_last=True,
    )

    # Build a small wrapper model
    class CLModel(nn.Module):
        def __init__(self, enc: HybridTextEncoder):
            super().__init__()
            self.enc = enc
            self.style = StyleMLP(style_dim, style_proj, dropout=args.dropout)
            self.proj = ContrastiveHead(enc.out_dim + style_proj, proj_dim=args.cl_proj_dim)

        def forward(self, w_ids, w_mask, c_ids, c_mask, st):
            z = self.enc(w_ids, w_mask, c_ids, c_mask)
            sp = self.style(st)
            h = torch.cat([z, sp], dim=1)
            p = self.proj(h)
            return p

    model = CLModel(encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)

    model.train()
    for ep in range(1, args.cl_epochs + 1):
        pbar = tqdm(loader, desc=f"[CL] epoch {ep}/{args.cl_epochs}", leave=False)
        for w1_ids, w1_mask, c1_ids, c1_mask, w2_ids, w2_mask, c2_ids, c2_mask, st in pbar:
            w1_ids, w1_mask = w1_ids.to(device), w1_mask.to(device)
            c1_ids, c1_mask = c1_ids.to(device), c1_mask.to(device)
            w2_ids, w2_mask = w2_ids.to(device), w2_mask.to(device)
            c2_ids, c2_mask = c2_ids.to(device), c2_mask.to(device)
            st = st.to(device)

            z1 = model(w1_ids, w1_mask, c1_ids, c1_mask, st)
            z2 = model(w2_ids, w2_mask, c2_ids, c2_mask, st)
            loss = nt_xent_loss(z1, z2, temperature=args.cl_temp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

    return encoder  # encoder updated in-place


def train_source_groupdro(
    model: EncoderClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_envs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[EncoderClassifier, float]:
    """
    Supervised training on source with optional GroupDRO.
    Returns best model (by val acc) and best val acc.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # GroupDRO weights
    q = torch.ones(n_envs, device=device) / float(n_envs)

    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0

    for ep in range(1, args.pretrain_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[SRC] epoch {ep}/{args.pretrain_epochs}", leave=False)

        for w_ids, w_mask, c_ids, c_mask, st, y, env in pbar:
            w_ids, w_mask = w_ids.to(device), w_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            st = st.to(device)
            y = y.to(device)
            env = env.to(device)

            _f, logits = model(w_ids, w_mask, c_ids, c_mask, st)
            per_ex_loss = F.cross_entropy(logits, y, reduction="none")

            if args.use_groupdro:
                # compute env-wise losses in this batch
                env_losses = torch.zeros(n_envs, device=device)
                env_present = torch.zeros(n_envs, device=device)
                for e in range(n_envs):
                    m = (env == e)
                    if bool(m.any()):
                        env_losses[e] = per_ex_loss[m].mean()
                        env_present[e] = 1.0

                # update q only on present envs
                if bool(env_present.any()):
                    # exponential update
                    q = q * torch.exp(args.groupdro_eta * env_losses.detach())
                    q = q / q.sum().clamp_min(1e-12)

                    # weighted objective over present envs
                    weights = q * env_present
                    denom = weights.sum().clamp_min(1e-12)
                    loss = (weights * env_losses).sum() / denom
                else:
                    loss = per_ex_loss.mean()
            else:
                loss = per_ex_loss.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        val_acc = eval_acc(model, val_loader, device=device, calib_T=1.0)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_acc


def safe_fixmatch_uda(
    student: EncoderClassifier,
    teacher: EncoderClassifier,
    src_loader: DataLoader,
    src_val_loader: DataLoader,
    tgt_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    calib_T: float,
) -> Tuple[EncoderClassifier, EncoderClassifier]:
    """
    Safe FixMatch adaptation on target unlabeled, anchored by source supervised loss.
    Checkpoint selection uses ONLY source-val accuracy (do-no-harm).
    """
    opt = torch.optim.AdamW(student.parameters(), lr=args.uda_lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, max(len(src_loader), len(tgt_loader)))
    global_step = 0

    best_state = copy.deepcopy(teacher.state_dict())
    best_val = eval_acc(teacher, src_val_loader, device=device, calib_T=calib_T)

    for ep in range(1, args.uda_epochs + 1):
        student.train()
        teacher.eval()

        # budget-aware ramp for lambda_u
        if args.budget_aware:
            # lambda_u ramps with epochs too
            ramp = min(1.0, ep / max(1.0, float(args.uda_ramp_epochs)))
            lambda_u = args.lambda_u * ramp
        else:
            lambda_u = args.lambda_u

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        pbar = tqdm(range(steps_per_epoch), desc=f"[UDA] epoch {ep}/{args.uda_epochs}", leave=False)
        for _ in pbar:
            try:
                sw_ids, sw_mask, sc_ids, sc_mask, sst, sy, _se = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                sw_ids, sw_mask, sc_ids, sc_mask, sst, sy, _se = next(src_iter)

            try:
                tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask, tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask, tst, _idx = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask, tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask, tst, _idx = next(tgt_iter)

            sw_ids, sw_mask = sw_ids.to(device), sw_mask.to(device)
            sc_ids, sc_mask = sc_ids.to(device), sc_mask.to(device)
            sst = sst.to(device)
            sy = sy.to(device)

            tw_w_ids, tw_w_mask = tw_w_ids.to(device), tw_w_mask.to(device)
            tc_w_ids, tc_w_mask = tc_w_ids.to(device), tc_w_mask.to(device)
            tw_s_ids, tw_s_mask = tw_s_ids.to(device), tw_s_mask.to(device)
            tc_s_ids, tc_s_mask = tc_s_ids.to(device), tc_s_mask.to(device)
            tst = tst.to(device)

            # source supervised
            _sf, slogits = student(sw_ids, sw_mask, sc_ids, sc_mask, sst)
            loss_sup = F.cross_entropy(slogits, sy)

            # target pseudo-labels from teacher (weak)
            with torch.no_grad():
                _tf, tlogits_w = teacher(tw_w_ids, tw_w_mask, tc_w_ids, tc_w_mask, tst)
                probs = torch.softmax(tlogits_w / max(1e-6, calib_T), dim=-1)
                conf, pseudo = probs.max(dim=1)
                ent = entropy_torch(probs) / math.log(2.0)
                mask = (conf >= args.fm_tau) & (ent <= args.fm_ent)

            # student on strong
            _uf, ulogits_s = student(tw_s_ids, tw_s_mask, tc_s_ids, tc_s_mask, tst)
            loss_u_all = F.cross_entropy(ulogits_s, pseudo, reduction="none")
            if bool(mask.any()):
                loss_u = loss_u_all[mask].mean()
            else:
                loss_u = torch.tensor(0.0, device=device)

            loss = loss_sup + lambda_u * loss_u

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            opt.step()

            ema_update(teacher, student, decay=args.ema_decay)

            global_step += 1
            pbar.set_postfix(ls=float(loss_sup.detach().cpu()), lu=float(loss_u.detach().cpu()), m=int(mask.sum().item()))

        # checkpoint by source-val only
        val_acc = eval_acc(teacher, src_val_loader, device=device, calib_T=calib_T)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(teacher.state_dict())

    teacher.load_state_dict(best_state)
    # also sync student for clean inference if needed
    student.load_state_dict(best_state)
    return student, teacher


# ----------------------------- Prediction / Threshold -------------------------


@torch.no_grad()
def predict_probs(
    model: EncoderClassifier,
    texts: Sequence[str],
    vocab: Vocab,
    device: torch.device,
    max_len: int,
    max_char_len: int,
    batch_size: int,
    calib_T: float,
) -> np.ndarray:
    ds = InferenceDataset(texts, vocab=vocab, max_len=max_len, max_char_len=max_char_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_infer, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=max_len, max_char_len=max_char_len),
        drop_last=False,
    )
    model.eval()
    probs = np.zeros((len(ds), 2), dtype=np.float32)
    for w_ids, w_mask, c_ids, c_mask, st, idxs in loader:
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        st = st.to(device)
        _f, logits = model(w_ids, w_mask, c_ids, c_mask, st)
        p = torch.softmax(logits / max(1e-6, calib_T), dim=-1).cpu().numpy().astype(np.float32)
        probs[idxs.numpy()] = p
    return probs


def pred_argmax_covid(probs_source_space: np.ndarray) -> np.ndarray:
    pred_source = probs_source_space.argmax(axis=1).astype(int)  # 0=true,1=false
    return np.where(pred_source == 0, 1, 0).astype(int)


def compute_thr_median_smooth(score_true: np.ndarray, budget: int, thr_budget_ref: int) -> float:
    """
    We set pi_true=0.5 (unknown), so matchprior -> median threshold.
    Smooth with 0.5 when budget is tiny.
    """
    if score_true.size == 0:
        return 0.5
    q = float(np.quantile(score_true, 0.5))
    w = min(1.0, float(budget) / max(1.0, float(thr_budget_ref)))
    thr = w * q + (1.0 - w) * 0.5
    return float(np.clip(thr, 0.05, 0.95))


def pred_matchprior_covid(probs_source_space: np.ndarray, thr_true: float) -> np.ndarray:
    score_true = probs_source_space[:, 0].astype(np.float32)
    return (score_true >= thr_true).astype(int)  # covid label: 1=true,0=false


# ----------------------------- Helpers ----------------------------------------


def parse_int_list(s: str) -> List[int]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def parse_budget_list(s: str) -> List[int]:
    b = parse_int_list(s)
    if len(b) == 0:
        return [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, -1]
    return b


def mean_std(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ----------------------------- Main -------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    base_dir = Path(__file__).resolve().parent

    # Data paths
    parser.add_argument("--pubhealth_train", type=str, default=str(base_dir / "./pubhealth/pubhealth_train_clean.csv"))
    parser.add_argument("--pubhealth_val", type=str, default=str(base_dir / "./pubhealth/pubhealth_validation_clean.csv"))
    parser.add_argument("--covid_true", type=str, default=str(base_dir / "../covid/trueNews.csv"))
    parser.add_argument("--covid_fake", type=str, default=str(base_dir / "../covid/fakeNews.csv"))

    # Inductive split
    parser.add_argument("--target_test_ratio", type=float, default=0.2)
    parser.add_argument("--target_split_seed", type=int, default=42)

    # Budgets & seeds
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--target_budgets", type=str, default="0,10,20,50,100,200,500,1000,2000,4000,-1")

    # Output
    parser.add_argument("--out_dir", type=str, default=str(base_dir / "v12_runs"))
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--results_csv", type=str, default=str(base_dir / "v12_budget_results.csv"))
    parser.add_argument("--summary_csv", type=str, default=str(base_dir / "v12_budget_summary.csv"))

    # Vocab usage (inductive safe)
    parser.add_argument("--vocab_mode", type=str, default="source_only", choices=["source_only", "source_plus_target"])
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)

    # Preprocess
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--max_char_len", type=int, default=512)

    # Model
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Char-CNN
    parser.add_argument("--char_emb_dim", type=int, default=32)
    parser.add_argument("--char_channels", type=int, default=96)
    parser.add_argument("--char_kernels", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--char_proj_dim", type=int, default=128)

    # Style
    parser.add_argument("--style_proj_dim", type=int, default=64)

    # Stage A: contrastive pretrain
    parser.add_argument("--cl_epochs", type=int, default=3)
    parser.add_argument("--cl_batch_size", type=int, default=64)
    parser.add_argument("--cl_lr", type=float, default=3e-4)
    parser.add_argument("--cl_temp", type=float, default=0.1)
    parser.add_argument("--cl_proj_dim", type=int, default=256)

    # Stage B: supervised on source
    parser.add_argument("--pretrain_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Pseudo environments + GroupDRO
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--use_groupdro", action="store_true")
    parser.add_argument("--groupdro_eta", type=float, default=0.05)

    # Temperature calibration
    parser.add_argument("--use_calibration", action="store_true")
    parser.add_argument("--temp_min", type=float, default=0.5)
    parser.add_argument("--temp_max", type=float, default=10.0)

    # Stage C: Safe FixMatch UDA
    parser.add_argument("--min_budget_for_uda", type=int, default=200)
    parser.add_argument("--uda_epochs", type=int, default=6)
    parser.add_argument("--uda_lr", type=float, default=2e-4)
    parser.add_argument("--uda_ramp_epochs", type=int, default=3)
    parser.add_argument("--lambda_u", type=float, default=0.7)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--fm_tau", type=float, default=0.95)
    parser.add_argument("--fm_ent", type=float, default=0.85)

    # Threshold smoothing
    parser.add_argument("--thr_budget_ref", type=int, default=500)

    # Budget-aware (mostly affects UDA ramping and threshold smoothing; can add more later)
    parser.add_argument("--budget_aware", action="store_true")

    # System
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    budgets = parse_budget_list(args.target_budgets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[V12] device={device}")
    print(f"[V12] seeds={seeds}")
    print(f"[V12] budgets={budgets} (-1 means full)")
    print(f"[V12] split_seed={args.target_split_seed} test_ratio={args.target_test_ratio}")
    print(f"[V12] vocab_mode={args.vocab_mode} | use_groupdro={args.use_groupdro} | min_budget_for_uda={args.min_budget_for_uda}")

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
        print("[Data] Target label source: file membership")

    print(f"[Data] Target rows={len(all_texts)} | dist={pd.Series(y).value_counts().to_dict()}")

    # ---------- Inductive split by unique text ----------
    train_idx, test_idx = split_target_by_unique_text(all_texts, y, test_ratio=args.target_test_ratio, seed=args.target_split_seed)
    tgt_train_unique_full = list(dict.fromkeys([all_texts[i] for i in train_idx]))
    tgt_test_texts = [all_texts[i] for i in test_idx]
    y_test = y[test_idx]
    print(f"[Split] target-train unique={len(tgt_train_unique_full)} | target-test rows={len(test_idx)}")

    # We use nested budgets for cleaner curves:
    # per seed, we create one permutation of tgt_train_unique_full; budget b uses first b.
    records: List[Dict[str, object]] = []
    budget_to_seed_acc: Dict[int, List[float]] = {int(b): [] for b in budgets}

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(tgt_train_unique_full))
        tgt_train_perm = [tgt_train_unique_full[i] for i in perm.tolist()]

        for budget_raw in budgets:
            budget_key = int(budget_raw)
            if budget_key < 0:
                tgt_budget_texts = tgt_train_perm
            else:
                b = int(min(budget_key, len(tgt_train_perm)))
                tgt_budget_texts = tgt_train_perm[:b]

            # ---------- build vocab (inductive safe) ----------
            if args.vocab_mode == "source_only":
                vocab_texts = src_train_texts + src_val_texts
            else:
                vocab_texts = src_train_texts + src_val_texts + tgt_budget_texts

            vocab = build_vocab(vocab_texts, min_freq=args.min_freq, max_size=args.vocab_size)

            # ---------- pseudo environments on SOURCE ----------
            env_train, vec, km = build_pseudo_envs(src_train_texts, n_envs=args.n_envs, seed=seed)
            env_val = predict_envs(src_val_texts, vec=vec, km=km)

            # ---------- loaders (source) ----------
            src_train_ds = SourceLabeledEnvDataset(src_train_texts, src_train_y, env_train, vocab, args.max_len, args.max_char_len)
            src_val_ds = SourceLabeledEnvDataset(src_val_texts, src_val_y, env_val, vocab, args.max_len, args.max_char_len)

            src_train_loader = DataLoader(
                src_train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=partial(collate_source_env, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
                drop_last=False,
            )
            src_val_loader = DataLoader(
                src_val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=partial(collate_source_env, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
                drop_last=False,
            )

            # ---------- model init ----------
            style_dim = int(compute_style_feats("x").shape[0])
            encoder = HybridTextEncoder(
                vocab_size=len(vocab.itos),
                word_emb_dim=args.emb_dim,
                word_hidden=args.hidden_size,
                word_pad=vocab.pad_id,
                char_vocab_size=97,
                char_emb_dim=args.char_emb_dim,
                char_channels=args.char_channels,
                char_kernels=tuple(args.char_kernels),
                char_pad=CHAR_PAD,
                char_proj_dim=args.char_proj_dim,
                dropout=args.dropout,
            )

            # ---------- Stage A: contrastive pretraining on unlabeled ----------
            # IMPORTANT: B=0 => only source unlabeled; B>0 => source unlabeled + target budget.
            unlabeled = list(src_train_texts) + list(src_val_texts) + list(tgt_budget_texts)
            if budget_key == 0:
                unlabeled = list(src_train_texts) + list(src_val_texts)

            encoder = contrastive_pretrain(
                encoder=encoder,
                style_dim=style_dim,
                style_proj=args.style_proj_dim,
                unlabeled_texts=unlabeled,
                vocab=vocab,
                args=args,
                device=device,
                seed=seed,
            )

            model = EncoderClassifier(encoder, style_dim=style_dim, style_proj=args.style_proj_dim, num_classes=2, dropout=args.dropout).to(device)

            # ---------- Stage B: supervised robust on source ----------
            model, best_src_val = train_source_groupdro(
                model=model,
                train_loader=src_train_loader,
                val_loader=src_val_loader,
                n_envs=args.n_envs,
                args=args,
                device=device,
            )

            # ---------- Optional calibration on source-val ----------
            calib_T = 1.0
            if args.use_calibration:
                logits_val, y_val = collect_logits_labels(model, src_val_loader, device=device)
                calib_T = fit_temperature(logits_val, y_val, device=device, t_min=args.temp_min, t_max=args.temp_max)

            # ---------- Stage C: Safe FixMatch UDA (only if enough target budget) ----------
            teacher = copy.deepcopy(model).to(device)
            for p in teacher.parameters():
                p.requires_grad = False

            if len(tgt_budget_texts) >= int(args.min_budget_for_uda) and args.uda_epochs > 0:
                tgt_ds = TargetUnlabeledFixMatchDataset(
                    tgt_budget_texts,
                    vocab=vocab,
                    max_len=args.max_len,
                    max_char_len=args.max_char_len,
                    seed=seed + 7,
                )
                tgt_loader = DataLoader(
                    tgt_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=partial(collate_target_fixmatch, pad_word=vocab.pad_id, pad_char=CHAR_PAD, max_len=args.max_len, max_char_len=args.max_char_len),
                    drop_last=True,
                )

                student = copy.deepcopy(model).to(device)
                student, teacher = safe_fixmatch_uda(
                    student=student,
                    teacher=teacher,
                    src_loader=src_train_loader,
                    src_val_loader=src_val_loader,
                    tgt_loader=tgt_loader,
                    args=args,
                    device=device,
                    calib_T=calib_T,
                )
            # else: keep teacher = source-trained model

            # ---------- Threshold from target-train budget (inductive) ----------
            if len(tgt_budget_texts) > 0:
                probs_train = predict_probs(teacher, tgt_budget_texts, vocab, device, args.max_len, args.max_char_len, args.batch_size, calib_T=calib_T)
                thr = compute_thr_median_smooth(probs_train[:, 0], budget=len(tgt_budget_texts), thr_budget_ref=args.thr_budget_ref)
            else:
                thr = 0.5

            # ---------- Evaluate on target-test ----------
            probs_test = predict_probs(teacher, tgt_test_texts, vocab, device, args.max_len, args.max_char_len, args.batch_size, calib_T=calib_T)
            pred_a = pred_argmax_covid(probs_test)
            pred_m = pred_matchprior_covid(probs_test, thr_true=thr)

            acc_argmax = float(accuracy_score(y_test, pred_a))
            acc_match = float(accuracy_score(y_test, pred_m))

            budget_used = len(tgt_budget_texts) if budget_key >= 0 else len(tgt_train_unique_full)
            budget_to_seed_acc[budget_key].append(acc_match)

            records.append(
                {
                    "budget_used": int(budget_used),
                    "budget_raw": int(budget_key),
                    "seed": int(seed),
                    "train_unique_used": int(len(tgt_budget_texts)),
                    "test_rows": int(len(test_idx)),
                    "acc_argmax": acc_argmax,
                    "acc_matchprior_inductive": acc_match,
                    "calib_T": float(calib_T),
                    "thr_true_from_train": float(thr),
                    "best_src_val_acc": float(best_src_val),
                }
            )

            if args.save_predictions:
                out_csv = out_dir / f"v12_pred_budget{budget_key}_seed{seed}.csv"
                pd.DataFrame(
                    {
                        "Text": tgt_test_texts,
                        "y_true": y_test,
                        "prob_true_source0": probs_test[:, 0],
                        "prob_false_source1": probs_test[:, 1],
                        "pred_argmax": pred_a,
                        "pred_matchprior": pred_m,
                        "thr": thr,
                    }
                ).to_csv(out_csv, index=False, encoding="utf-8-sig")

            print(f"[B={budget_used:>5}] seed={seed} acc_match={acc_match:.4f} acc_argmax={acc_argmax:.4f} thr={thr:.3f} src_val={best_src_val:.4f}")

    # ---------- Save results ----------
    df_res = pd.DataFrame(records)
    df_res.to_csv(args.results_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved detailed results: {args.results_csv}")

    # ---------- Summary ----------
    summary_rows: List[Dict[str, object]] = []
    for budget_raw in budgets:
        key = int(budget_raw)
        accs = budget_to_seed_acc[key]
        m, s = mean_std(accs)
        used = len(tgt_train_unique_full) if key < 0 else int(key)
        summary_rows.append(
            {
                "budget_used": int(used),
                "budget_raw": int(key),
                "mean_acc_matchprior_inductive": float(m),
                "std_acc_matchprior_inductive": float(s),
            }
        )

    df_sum = pd.DataFrame(sorted(summary_rows, key=lambda d: int(d["budget_used"])))
    df_sum.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved summary: {args.summary_csv}")

    print("\n========== V12 Summary ==========")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
