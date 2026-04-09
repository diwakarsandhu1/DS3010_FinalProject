# DS3010 Final Project

## Project Structure

- `raw_data/`
  - Stores original, unmodified data pulled from sources.
  - Examples: raw NewsAPI article dumps, raw Yahoo Finance stock data CSVs.

- `processed_data/`
  - Stores cleaned, transformed, or merged datasets ready for analysis/modeling.
  - Examples: sentiment-scored articles, merged news + stock datasets.

- `notebooks/`
  - Used for exploration, testing ideas, and step-by-step analysis.
  - Examples:
    - `data_collection.ipynb`
    - `sentiment_analysis.ipynb`
    - `stock_trend_exploration.ipynb`

- `src/`
  - Stores reusable Python scripts and project logic.
  - Examples:
    - `data_collection.py` for pulling data from APIs
    - `preprocessing.py` for cleaning text/data
    - `sentiment_model.py` for scoring article sentiment
    - `train_model.py` for predictive modeling

- `models/`
  - Stores saved trained models and related files.
  - Examples: `.pkl` model files, vectorizers, scalers.

- `figures/`
  - Stores charts, plots, and visualizations used in reports or presentations.

- `outputs/`
  - Stores final generated results from scripts or notebooks.
  - Examples: prediction CSVs, evaluation summaries, exported tables.

- `main.py`
  - Entry point for running the project pipeline or main analysis steps.