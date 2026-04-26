from pathlib import Path
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "processed_data" / "news"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "sentiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def get_text_column(df: pd.DataFrame) -> str:
    candidates = ["content", "description", "title", "clean_text"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No usable text column found. Expected one of: {candidates}. "
        f"Columns present: {list(df.columns)}"
    )


def score_sentiment(text: str) -> float:
    if pd.isna(text) or not str(text).strip():
        return 0.0
    return analyzer.polarity_scores(str(text))["compound"]


def label_sentiment(score: float) -> str:
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"


def normalize_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    datetime_candidates = ["publishedAt", "published_at", "date", "datetime"]
    found = None

    for col in datetime_candidates:
        if col in df.columns:
            found = col
            break

    if found is None:
        raise ValueError(
            f"No datetime column found. Expected one of: {datetime_candidates}"
        )

    df = df.copy()
    df["publishedAt"] = pd.to_datetime(df[found], errors="coerce", utc=True)
    df = df.dropna(subset=["publishedAt"])
    df["date"] = df["publishedAt"].dt.date
    return df


def add_ticker_if_missing(df: pd.DataFrame, fallback_ticker: str) -> pd.DataFrame:
    df = df.copy()
    if "ticker" not in df.columns:
        df["ticker"] = fallback_ticker
    return df


def process_news_file(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    ticker = csv_path.name.replace("_news_clean.csv", "")
    df = add_ticker_if_missing(df, ticker)
    df = normalize_datetime_column(df)

    text_col = get_text_column(df)

    df["sentiment_score"] = df[text_col].apply(score_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    article_output = df.copy()

    daily_output = (
        df.groupby(["ticker", "date"], as_index=False)
        .agg(
            sentiment_score_mean=("sentiment_score", "mean"),
            sentiment_score_median=("sentiment_score", "median"),
            sentiment_score_std=("sentiment_score", "std"),
            article_count=("sentiment_score", "count"),
            positive_count=("sentiment_label", lambda s: (s == "positive").sum()),
            neutral_count=("sentiment_label", lambda s: (s == "neutral").sum()),
            negative_count=("sentiment_label", lambda s: (s == "negative").sum()),
        )
        .sort_values(["ticker", "date"])
    )

    daily_output["sentiment_score_std"] = daily_output["sentiment_score_std"].fillna(0)

    return article_output, daily_output


def main() -> None:
    csv_files = sorted(
        [
            path for path in INPUT_DIR.glob("*_news_clean.csv")
            if not path.name.startswith("all_companies")
        ]
    )

    if not csv_files:
        raise FileNotFoundError(f"No cleaned per-company news CSV files found in {INPUT_DIR}")

    all_articles = []
    all_daily = []

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        article_df, daily_df = process_news_file(csv_file)

        ticker = article_df["ticker"].iloc[0]

        article_df.to_csv(
            OUTPUT_DIR / f"{ticker}_news_sentiment_articles.csv",
            index=False
        )
        daily_df.to_csv(
            OUTPUT_DIR / f"{ticker}_news_sentiment_daily.csv",
            index=False
        )

        all_articles.append(article_df)
        all_daily.append(daily_df)

    combined_articles = pd.concat(all_articles, ignore_index=True)
    combined_daily = pd.concat(all_daily, ignore_index=True)

    combined_articles.to_csv(
        OUTPUT_DIR / "all_companies_news_sentiment_articles.csv",
        index=False
    )
    combined_daily.to_csv(
        OUTPUT_DIR / "all_companies_news_sentiment_daily.csv",
        index=False
    )

    print("Sentiment analysis complete.")
    print(f"Saved article-level and daily sentiment files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()