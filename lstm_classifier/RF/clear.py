import csv
from pathlib import Path

import pandas as pd


def read_csv_safely(path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                engine="python",
                quoting=csv.QUOTE_MINIMAL,
                on_bad_lines="warn",
            )
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path}: {last_err}")


def clean_by_date_and_label(df: pd.DataFrame, cutoff: str = "2019-11-01") -> pd.DataFrame:
    if "date_published" not in df.columns:
        raise KeyError("Column 'date_published' not found in the CSV.")

    # ---- date filter ----
    s = df["date_published"].astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA})
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    cutoff_ts = pd.Timestamp(cutoff)
    keep_mask = parsed.notna() & (parsed <= cutoff_ts)
    df = df.loc[keep_mask].copy()

    # ---- drop label==3 rows ----
    if "label" not in df.columns:
        raise KeyError("Column 'label' not found in the CSV.")
    label_num = pd.to_numeric(df["label"], errors="coerce")  # handles '3' as string
    df = df.loc[(label_num.isna()) | (label_num != 3)].copy()

    return df


def main():
    files = [
        "pubhealth_train.csv",
        "pubhealth_test.csv",
        "pubhealth_validation.csv",
    ]
    cutoff = "2019-11-01"

    for f in files:
        in_path = Path(f)
        df = read_csv_safely(str(in_path))

        before = len(df)
        df_clean = clean_by_date_and_label(df, cutoff=cutoff)
        after = len(df_clean)

        out_path = in_path.with_name(in_path.stem + "_clean.csv")
        df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[{in_path.name}] kept {after}/{before} rows, removed {before - after}. -> {out_path.name}")


if __name__ == "__main__":
    main()
