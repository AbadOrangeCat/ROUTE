import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "targetdata")
OUT_DIR = os.path.join(BASE_DIR, "targetdata_clean")
os.makedirs(OUT_DIR, exist_ok=True)

TEXT_COL = "text"
LABEL_COL = "label"  # val/test会保留；train无标签也无所谓

def norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)  # 合并多空格
    return s  # 建议先不要lower()，避免过度合并；如你想更严格可加 .lower()

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column {TEXT_COL} in {path}, columns={list(df.columns)}")
    df[TEXT_COL] = df[TEXT_COL].map(norm_text)
    df = df[df[TEXT_COL].astype(str).str.len() > 0].copy()
    return df

train_path = os.path.join(SRC_DIR, "train.csv")
val_path   = os.path.join(SRC_DIR, "val.csv")
test_path  = os.path.join(SRC_DIR, "test.csv")

train = load_csv(train_path)
val   = load_csv(val_path)
test  = load_csv(test_path)

# 去掉各自内部的重复（可选但推荐）
train = train.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
val   = val.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
test  = test.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

val_set  = set(val[TEXT_COL].tolist())
test_set = set(test[TEXT_COL].tolist())
valtest_set = val_set | test_set

# 关键：从 train 删除出现在 val/test 的文本
train_clean = train[~train[TEXT_COL].isin(valtest_set)].copy()

# 从 val 删除出现在 test 的文本
val_clean = val[~val[TEXT_COL].isin(test_set)].copy()

test_clean = test.copy()

print("=== BEFORE ===")
print("train:", len(train), "val:", len(val), "test:", len(test))

print("\n=== AFTER PURGE ===")
print("train_clean:", len(train_clean), "val_clean:", len(val_clean), "test_clean:", len(test_clean))

print("\nRemoved from train:", len(train) - len(train_clean))
print("Removed from val  :", len(val) - len(val_clean))

# 保存
train_clean.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8")
val_clean.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8")
test_clean.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8")

print("\nSaved to:", OUT_DIR)
