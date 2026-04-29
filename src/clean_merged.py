from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / "processed_data" / "merged"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "cleaned_merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOAT_COLS = ["Open", "High", "Low", "Close", "Return_1D", "MA_5", "MA_21", "MA_126", "MA_252", "Volatility_21", "sentiment_score"]
INT_COLS = ["Volume"]
TEXT_COLS = ["title", "description", "content", "author", "source_name"]
DATE_COL = "date"
DATETIME_COL = "published_at"

# regex for NewsAPI truncation artifact at end of content
TRUNCATION_RE = re.compile(r"[\u2026\.]{1,3}\s*\[.*?chars\]\s*$", flags=re.DOTALL)


def strip_invisible_unicode(text: str) -> str:
    # remove zero-width/control chars but keep normal whitespace
    return "".join(
        ch for ch in text
        if not (unicodedata.category(ch) in ("Cf", "Cc") and ch not in ("\n", "\t", "\r", " "))
    )


def clean_content(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    # returns cleaned content and a boolean flag for truncated rows
    truncated = series.astype(str).str.contains(r"\[.*?chars\]", regex=True, na=False)
    cleaned = series.astype(str).apply(lambda x: TRUNCATION_RE.sub("", x).rstrip())
    return cleaned, truncated


def clean_dataframe(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, dict]:
    report: dict[str, object] = {"file": filename, "original_rows": len(df)}

    # drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = before - len(df)

    # drop columns that are duplicates of others
    redundant = []
    if "publishedAt" in df.columns and "published_at" in df.columns:
        df = df.drop(columns=["publishedAt"])
        redundant.append("publishedAt")
    if "Date" in df.columns and DATE_COL in df.columns:
        df = df.drop(columns=["Date"])
        redundant.append("Date")
    report["redundant_cols_dropped"] = redundant

    # normalize datetime and date columns to consistent formats
    if DATETIME_COL in df.columns:
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")

    # strip whitespace from all string columns
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # remove invisible unicode chars from text columns
    invisible_fixed: dict[str, int] = {}
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        before_vals = df.loc[mask, col].copy()
        df.loc[mask, col] = df.loc[mask, col].apply(strip_invisible_unicode)
        changed = (df.loc[mask, col] != before_vals).sum()
        if changed:
            invisible_fixed[col] = int(changed)
    report["invisible_unicode_fixed"] = invisible_fixed

    # clean truncated content and flag affected rows
    if "content" in df.columns:
        cleaned_content, truncated_flag = clean_content(df["content"])
        df["content"] = cleaned_content
        df["content_truncated"] = truncated_flag
        report["content_truncated_rows"] = int(truncated_flag.sum())
    else:
        report["content_truncated_rows"] = 0

    # fill missing author and description
    fill_counts: dict[str, int] = {}
    if "author" in df.columns:
        n = df["author"].isna().sum()
        df["author"] = df["author"].fillna("Unknown")
        if n:
            fill_counts["author"] = int(n)
    if "description" in df.columns:
        n = df["description"].isna().sum()
        df["description"] = df["description"].fillna("")
        if n:
            fill_counts["description"] = int(n)
    report["missing_filled"] = fill_counts

    # enforce numeric dtypes for stock and sentiment columns
    dtype_errors: dict[str, str] = {}
    for col in FLOAT_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            except Exception as exc:
                dtype_errors[col] = str(exc)
    for col in INT_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            except Exception as exc:
                dtype_errors[col] = str(exc)
    if dtype_errors:
        report["dtype_errors"] = dtype_errors

    # normalize sentiment_label to lowercase and flag invalid values
    valid_labels = {"positive", "neutral", "negative"}
    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].str.lower().str.strip()
        invalid_mask = ~df["sentiment_label"].isin(valid_labels) & df["sentiment_label"].notna()
        report["invalid_sentiment_labels"] = int(invalid_mask.sum())

    # drop rows missing essential columns
    essential = [c for c in [DATE_COL, "ticker", "Close", "sentiment_score"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=essential)
    report["rows_dropped_missing_essential"] = before - len(df)

    # reset index and reissue article_id
    df = df.reset_index(drop=True)
    if "article_id" in df.columns:
        df["article_id"] = df.index

    report["final_rows"] = len(df)
    return df, report


PREFERRED_ORDER = [
    "article_id", "ticker",
    "published_at", "date",
    "source_name", "author", "title", "description", "content", "content_truncated", "url",
    "sentiment_score", "sentiment_label",
    "Open", "High", "Low", "Close", "Volume",
    "Return_1D", "MA_5", "MA_21", "MA_126", "MA_252", "Volatility_21",
]


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in PREFERRED_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def main() -> None:
    csv_files = sorted(MERGED_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {MERGED_DIR}")

    print(f"Found {len(csv_files)} file(s) in {MERGED_DIR}\n")

    for csv_path in csv_files:
        print(f"--- {csv_path.name}")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            print(f"ERROR reading file: {exc}\n")
            continue

        df_clean, report = clean_dataframe(df, csv_path.name)
        df_clean = reorder_columns(df_clean)

        out_name = csv_path.stem + "_clean.csv"
        out_path = OUTPUT_DIR / out_name
        df_clean.to_csv(out_path, index=False)

        print(f"rows: {report['original_rows']} -> {report['final_rows']}")
        print(f"dupes dropped: {report['duplicate_rows_dropped']}")
        print(f"cols dropped: {report['redundant_cols_dropped']}")
        print(f"content truncated: {report['content_truncated_rows']}")
        print(f"missing filled: {report['missing_filled']}")
        print(f"unicode fixed: {report['invisible_unicode_fixed']}")
        if report.get("invalid_sentiment_labels"):
            print(f"invalid sentiment labels: {report['invalid_sentiment_labels']}")
        if "dtype_errors" in report:
            print(f"dtype errors: {report['dtype_errors']}")
        print(f"saved to: {out_path}\n")

    print(f"Done. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()