from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTIMENT_DIR = PROJECT_ROOT / "processed_data" / "sentiment"
STOCK_DIR = PROJECT_ROOT / "processed_data" / "stock_data"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_stock_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Date" not in df.columns:
        raise ValueError(f"Stock dataset missing 'Date' column. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date"])

    return df


def normalize_sentiment_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    elif "publishedAt" in df.columns:
        df["date"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True).dt.date
    elif "published_at" in df.columns:
        df["date"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.date
    else:
        raise ValueError(
            f"Sentiment dataset missing 'date', 'publishedAt', or 'published_at'. "
            f"Columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["date"])
    return df


def join_one_ticker(ticker: str) -> pd.DataFrame:
    sentiment_path = SENTIMENT_DIR / f"{ticker}_news_sentiment_articles.csv"
    stock_path = STOCK_DIR / f"{ticker}_stock_data.csv"

    if not sentiment_path.exists():
        raise FileNotFoundError(f"Missing sentiment file: {sentiment_path}")
    if not stock_path.exists():
        raise FileNotFoundError(f"Missing stock file: {stock_path}")

    sentiment_df = pd.read_csv(sentiment_path)
    stock_df = pd.read_csv(stock_path)

    sentiment_df = normalize_sentiment_dates(sentiment_df)
    stock_df = normalize_stock_dates(stock_df)

    if "ticker" not in sentiment_df.columns:
        sentiment_df["ticker"] = ticker
    else:
        sentiment_df["ticker"] = sentiment_df["ticker"].astype(str).str.strip()

    stock_df["ticker"] = ticker

    merged_df = pd.merge(
        sentiment_df,
        stock_df,
        on=["ticker", "date"],
        how="inner",
        suffixes=("_news", "_stock")
    )

    return merged_df


def main() -> None:
    sentiment_files = sorted(SENTIMENT_DIR.glob("*_news_sentiment_articles.csv"))

    tickers = [
        path.name.replace("_news_sentiment_articles.csv", "")
        for path in sentiment_files
        if not path.name.startswith("all_companies")
    ]

    if not tickers:
        raise FileNotFoundError(
            f"No per-company sentiment article files found in {SENTIMENT_DIR}"
        )

    all_merged = []

    for ticker in tickers:
        try:
            print(f"Joining {ticker}...")
            merged_df = join_one_ticker(ticker)

            if merged_df.empty:
                print(f"No matching dates found for {ticker}.")
                continue

            merged_df.to_csv(
                OUTPUT_DIR / f"{ticker}_merged_articles_stock.csv",
                index=False
            )

            all_merged.append(merged_df)
            print(f"Saved merged dataset for {ticker} ({len(merged_df)} rows).")

        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    if not all_merged:
        raise ValueError("No datasets were successfully joined.")

    combined_merged = pd.concat(all_merged, ignore_index=True)
    combined_merged.to_csv(
        OUTPUT_DIR / "all_companies_merged_articles_stock.csv",
        index=False
    )

    print("Dataset joining complete.")
    print(f"Saved merged files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()