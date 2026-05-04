# DS3010 Final Project: Predicting Stock Movement with News Sentiment

This project investigates whether financial news sentiment and stock-market indicators can help predict short-term stock movement for major S&P 500 companies. The pipeline combines NewsAPI article data, Yahoo Finance stock data, VADER sentiment scores, and machine learning models for both regression and classification tasks.

## Project Goal

The main goal is to evaluate whether news sentiment features improve prediction of daily stock returns and daily stock direction.

The project includes two modeling tasks:

1. **Regression:** predict the numeric one-day return (`Return_1D`)
2. **Classification:** predict whether the stock moves up or down/flat on a given day

## Data Sources

The project uses two primary data sources:

- **NewsAPI**: company-related news articles
- **Yahoo Finance / yfinance**: historical stock price and volume data

The analysis focuses on major companies including:

- AAPL
- AMZN
- AVGO
- BRK-B
- GOOG
- GOOGL
- META
- MSFT
- NVDA
- TSLA


## Methodology

1. **Collect Data**  
   News articles were gathered from NewsAPI, and historical stock data was collected from Yahoo Finance using `yfinance`.

2. **Score Sentiment**  
   Each news article was scored with VADER sentiment analysis and labeled as positive, neutral, or negative.

3. **Merge Datasets**  
   Daily sentiment features were aggregated by ticker and date, then merged with stock price and technical indicator data.

4. **Build Features**  
   The final dataset includes sentiment statistics, article counts, stock prices, moving averages, volatility, and ticker information.

5. **Train Models**  
   Regression models were used to predict daily stock returns, while classification models were used to predict whether a stock moved up or down/flat.

6. **Evaluate Results**  
   Regression models were evaluated with MAE, RMSE, and R². Classification models were evaluated with accuracy, precision, recall, and F1-score.
  
## How to Run

1. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Add your NewsAPI key**

Create a `.env` file:

```text
NEWS_API_KEY=your_api_key_here
```

4. **Run the main scripts**

```bash
python src/sentiment_model.py
python src/join_datasets.py
python src/clean_merged.py
python src/regression.py
```

5. **Optional: generate visuals**

```bash
python src/regvisualizations.py
python src/classvisuals.py
```

Classification models can also be run from:

```text
notebooks/classification_models_notebook.ipynb
```