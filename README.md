# ⚙️ Reusable Regression Pipeline (RRP)

> **A dataset-agnostic, config-driven supervised learning pipeline** — point it at any CSV, select your target variable, and get a fully evaluated linear regression model with cross-validated metrics, ranked feature coefficients, and a serialized pipeline ready for live inference.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Math-NumPy-013243?style=flat-square)](https://numpy.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

---

## What is this?

RRP is not a trained model — it is a **reusable workflow** that can be aimed at any structured dataset without writing a single line of code. Drop in a CSV, select what you want to predict, and the pipeline handles everything else: cleaning, scaling, training, evaluation, and serialization.

```
Select Mode:
  [1] Train a new Regression Model on a CSV dataset
  [2] Load an existing saved pipeline (.pkl) for Live Predictions

Enter choice: 1

[?] Enter the path to your CSV file: housing_data.csv
[?] Enter the name or number of the column you want to predict: price

═══════════════════════════════════════════════════════════════════════════
 REUSABLE REGRESSION PIPELINE (RRP) — PERFORMANCE DASHBOARD
═══════════════════════════════════════════════════════════════════════════
  R-squared (R²) Score  : 0.9991  ← held-out test set
  CV R² Mean            : 0.9992 (+/- 0.0002)  ← 5-fold average
  Mean Absolute Error   : 2352.56
  Root Mean Sq. Error   : 2819.56

  FEATURE IMPACT RANKING
  [1]   size_sqft        104462.92    ████████████████████ (+)
  [2]   distance_km      -16426.36    ███                  (-)
  [3]   bedrooms          14496.98    ██                   (+)
  [4]   age_years         -6680.54    █                    (-)
═══════════════════════════════════════════════════════════════════════════
```

## Features

- **Automatic CSV cleaning** — detects and drops non-numeric columns, removes ID columns, patches missing values with column means
- **Interactive target selection** — select your prediction target by name or number at runtime
- **Config-driven workflow** — all split ratios, random seeds, CV folds, and scaling toggles live in `pipeline_settings.json`
- **5-fold cross-validation** — produces a stable R² estimate alongside the single held-out test score
- **Feature impact ranking** — ranked coefficient table with a visual bar graph showing relative feature power
- **Pipeline serialization** — exports a trained `.pkl` file containing the full scaler + model pipeline
- **Live inference mode** — load any saved pipeline and predict from a new CSV or manually entered values

---

## How It Works

The pipeline runs four sequential phases:

### Phase 1 — Data Ingestion & Cleaning
- Loads CSV with Latin-1 encoding fallback for special characters
- Filters columns to only those that are at least 50% numeric (configurable)
- Detects and drops integer ID columns using a two-signal heuristic — name pattern + primary key check
- Fills remaining missing values with column means

### Phase 2 — Interactive Target Selection
- Lists all cleaned numeric columns with index numbers
- Accepts target selection by column name or number
- Automatically assigns all remaining columns as input features (X)

### Phase 3 — Training & Cross-Validation
- Splits data into configurable train/test ratio (default 80/20)
- Builds a `sklearn.Pipeline` with optional `StandardScaler` and `LinearRegression`
- Fits on training data, then runs k-fold cross-validation on the full dataset for a stable performance estimate

### Phase 4 — Analytics Dashboard
- Evaluates on held-out test set (R², MAE, RMSE)
- Displays CV scores alongside single-split score for comparison
- Ranks features by absolute coefficient magnitude with a visual bar graph
- Optionally serializes the full pipeline to disk with `joblib`

---

## Project Structure

```
reusable-regression-pipeline/
├── reusable_regression_pipeline.py   # Full pipeline — ingestion, training, evaluation, inference
├── pipeline_settings.json            # All configurable parameters
└── requirements.txt
```

The pipeline is deliberately config-driven — `reusable_regression_pipeline.py` contains only logic, never hardcoded operational values.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| ML pipeline & model | `scikit-learn` |
| Data ingestion & cleaning | `Pandas` |
| Numerical computation | `NumPy` |
| Pipeline serialization | `joblib` |
| Configuration loading | `json` (stdlib) |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python reusable_regression_pipeline.py
```

### 3. Follow the prompts

```
Select Mode:
  [1] Train a new Regression Model on a CSV dataset
  [2] Load an existing saved pipeline (.pkl) for Live Predictions
```

**Training mode** — provide a CSV path, select your target column, review the dashboard, optionally save the pipeline.

**Inference mode** — provide a saved `.pkl` path, then either supply a new CSV or enter feature values manually for live predictions.

---

## Configuration

### `pipeline_settings.json`

| Parameter | What it controls |
|-----------|-----------------|
| `test_split_ratio` | Fraction of data held out for final evaluation (default: 0.2) |
| `random_state` | Seed for reproducible train/test splits |
| `numeric_threshold` | Minimum ratio of numeric values for a column to be retained (default: 0.50) |
| `cross_validation_folds` | Number of CV folds for stable performance estimation (default: 5) |
| `scaling_enabled` | Toggle StandardScaler normalization before training (default: true) |

All parameters have safe fallback defaults — the pipeline runs correctly even if the config file is missing.

---

## Understanding the Output

**R² (held-out)** — proportion of variance in the target explained by the model on the unseen test set. 1.0 is perfect, 0.0 means the model is no better than predicting the mean.

**CV R² Mean ± std** — average R² across k independent train/test splits. When this is close to the held-out R², your model is stable. When they diverge significantly, your single split got lucky or unlucky.

**Feature coefficients** — scaled weights showing how much each feature moves the prediction per unit change. Ranked by absolute magnitude so the most influential features appear first. Positive means increasing the feature increases the prediction; negative means the reverse.

---

## Design Decisions

**Why config-driven?**
Hardcoded split ratios and thresholds make the pipeline fragile and non-reproducible. Externalizing them into `pipeline_settings.json` means any experiment can be precisely reproduced or adjusted without touching the code.

**Why cross-validation alongside a single split?**
A single train/test split R² is sensitive to which rows randomly landed in the test set. Cross-validation averages performance across k independent splits, producing a stable estimate with a measurable variance. Both scores together tell you more than either alone.

**Why sklearn Pipeline instead of manual scaling?**
A `sklearn.Pipeline` keeps the scaler and model as one serializable unit. This guarantees that new data passed to the saved `.pkl` during inference is always scaled with the same parameters fitted on training data — a common source of silent bugs when scaling is done manually.

---

## Roadmap

- [ ] Classification mode — logistic regression, decision tree, with `pipeline_settings.json` mode toggle
- [ ] Feature selection — configurable column include/exclude list
- [ ] Polynomial and Ridge regression recipe support
- [ ] Correlation matrix report in Phase 1
- [ ] HTML report export alongside terminal dashboard

---

## Limitations

- Currently supports only linear regression — non-linear relationships will produce poor R² scores
- Mean imputation for missing values may introduce bias on datasets with significant missingness
- Unusual ID formats may not be caught
- Feature coefficients reflect scaled weights and are not directly interpretable in original units

---

*Built as a dataset-agnostic supervised learning pipeline to explore reusable ML workflow architecture — part of a broader portfolio of config-driven data science tools.*
