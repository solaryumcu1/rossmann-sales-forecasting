# Rossmann Store Sales Forecasting (Kaggle)

Time-series retail sales forecasting for 1,115 stores using feature engineering (lags/rolling stats) and an XGBoost baseline pipeline.

**Competition:** Rossmann Store Sales (Kaggle)  
Kaggle dataset page: [https://www.kaggle.com/c/rossmann-store-sales](https://www.kaggle.com/competitions/rossmann-store-sales/data)

---

## Project Goal

Rossmann store managers forecast daily sales up to 6 weeks ahead. Sales are affected by promotions, competition, school/state holidays, seasonality, and store locality.  
This repository builds a forecasting pipeline and a baseline model, focusing on correct time-based validation and leakage-safe feature engineering.

---

## Data

The dataset is publicly available on Kaggle (see link above).  
Typical input files:

- `train.csv` (daily sales by store/date)
- `store.csv` (store metadata)

> Note: Large datasets are **not** included in this repository. Please download from Kaggle.

---

## Repository Contents

### Scripts
1) **`rossmann_data_preprocessing.py`**
   - Loads and merges data sources (`train.csv` + `store.csv`)
   - Cleans data, handles missing values
   - Creates core calendar features (date parts, etc.)
   - Outputs a processed dataset used by later steps

2) **`rossmann_step4_lags.py`**
   - Creates **lag features** and **rolling statistics** (e.g., lag_7, rolling means)
   - Adds leakage-safe time-series features per store
   - Outputs a feature-enriched dataset ready for training

3) **`rossmann_step5_model_baseline.py`**
   - Trains a baseline forecasting model (XGBoost)
   - Uses time-based validation (no shuffling)
   - Reports performance metrics and feature importance

---

## Feature Engineering Highlights

Typical feature groups used in this project:

- **Calendar features:** day-of-week, week-of-year, month, day-of-year
- **Lag features:** previous sales values (e.g., 7-day lag)
- **Rolling statistics:** rolling mean / std over 7/14/28 days
- **Promo interactions:** promo-aware lags/rollings (optional)
- **Store profile features:** store type, assortment, competition/promo2 fields

All time-series features are created in a way that avoids using future information.

---

## Evaluation Metric (Kaggle)

Kaggle evaluates submissions with **RMSPE** (Root Mean Squared Percentage Error):

\[
RMSPE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}
\]

> In practice, rows with `Sales = 0` must be handled carefully to avoid division issues.

---

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

2)Preprocess + merge
python rossmann_data_preprocessing.py

3) Generate lag/rolling features
python rossmann_step4_lags.py

4) Train baseline model
python rossmann_step5_model_baseline.py

Notes

This repo is structured as a step-by-step pipeline for clarity and reproducibility.

If you change feature windows (e.g., 7/14/28), re-run step4 before training.

You can extend the baseline with hyperparameter tuning, better validation splits, and additional feature sets.

Author

Mustafa Efe Kılıç
Industrial Engineering | Forecasting | Machine Learning

pandas
numpy
scikit-learn
matplotlib
xgboost
