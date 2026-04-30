from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Define constants for file paths and target column
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "processed_data" / "cleaned_merged" / "all_companies_merged_articles_stock_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
TARGET_COL = "Return_1D"

def load_data(input_path=INPUT_PATH) -> pd.DataFrame:
    # Load the dataset
    df = pd.read_csv(input_path)

    # Convert 'date' column to datetime and drop rows with missing values in critical columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=["ticker", "date", TARGET_COL, "sentiment_score"])

    return df

def build_daily_regression_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create binary columns for sentiment labels
    df["is_positive"] = (df["sentiment_label"] == "positive").astype(int)
    df["is_neutral"] = (df["sentiment_label"] == "neutral").astype(int)
    df["is_negative"] = (df["sentiment_label"] == "negative").astype(int)

    # Aggregate the data to daily level for each ticker
    daily_df = (
        df.groupby(["ticker", "date"], as_index=False)
        .agg(
            # Stock target and features
            Return_1D=("Return_1D", "first"),
            Open=("Open", "first"),
            High=("High", "first"),
            Low=("Low", "first"),
            Close=("Close", "first"),
            Volume=("Volume", "first"),
            MA_5=("MA_5", "first"),
            MA_21=("MA_21", "first"),
            MA_126=("MA_126", "first"),
            MA_252=("MA_252", "first"),
            Volatility_21=("Volatility_21", "first"),

            # Sentiment features
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_median=("sentiment_score", "median"),
            sentiment_std=("sentiment_score", "std"),
            sentiment_min=("sentiment_score", "min"),
            sentiment_max=("sentiment_score", "max"),

            # News volume features
            article_count=("article_id", "count"),
            unique_sources=("source_name", "nunique"),
            truncated_articles=("content_truncated", "sum"),

            # Sentiment label counts
            positive_count=("is_positive", "sum"),
            neutral_count=("is_neutral", "sum"),
            negative_count=("is_negative", "sum"),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # Handle cases where there are no articles for a given day (e.g., fill NaN values for sentiment features)
    daily_df["sentiment_std"] = daily_df["sentiment_std"].fillna(0)

    daily_df["positive_ratio"] = daily_df["positive_count"] / daily_df["article_count"]
    daily_df["neutral_ratio"] = daily_df["neutral_count"] / daily_df["article_count"]
    daily_df["negative_ratio"] = daily_df["negative_count"] / daily_df["article_count"]

    daily_df["truncated_ratio"] = (
        daily_df["truncated_articles"] / daily_df["article_count"]
    )
    
    return daily_df

def get_feature_columns() -> tuple[list[str], list[str]]:
    
    # Define feature columns
    numeric_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_5",
        "MA_21",
        "MA_126",
        "MA_252",
        "Volatility_21",
        "sentiment_mean",
        "sentiment_median",
        "sentiment_std",
        "sentiment_min",
        "sentiment_max",
        "article_count",
        "unique_sources",
        "truncated_articles",
        "positive_count",
        "neutral_count",
        "negative_count",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "truncated_ratio",
    ]

    categorical_features = ["ticker"]

    return numeric_features, categorical_features

def prepare_features(daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        "ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_5",
        "MA_21",
        "MA_126",
        "MA_252",
        "Volatility_21",
        "sentiment_mean",
        "sentiment_median",
        "sentiment_std",
        "sentiment_min",
        "sentiment_max",
        "article_count",
        "unique_sources",
        "positive_count",
        "neutral_count",
        "negative_count",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
    ]

    X = daily_df[feature_cols].copy()
    y = daily_df["Return_1D"]

    # Convert ticker into numeric dummy columns
    X = pd.get_dummies(X, columns=["ticker"], drop_first=True)

    # Fill missing numeric values
    X = X.fillna(X.median(numeric_only=True))

    return X, y

def train_models(daily_df: pd.DataFrame) -> pd.DataFrame:
    X, y = prepare_features(daily_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            random_state=42
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })

    return pd.DataFrame(results).sort_values("rmse")

def main():
    df = load_data()
    daily_df = build_daily_regression_dataset(df)

    results = train_models(daily_df)

    print(results)

    results.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    daily_df.to_csv(OUTPUT_DIR / "regression_daily_dataset.csv", index=False)

if __name__ == "__main__":
    main()