# ⚙️ Supervised Learning Pipelines (SLP)

> **A dataset-agnostic, config-driven supervised learning suite** — point either pipeline at any CSV, select your target, and get a fully evaluated, serializable model with cross-validated metrics and live inference support. No dataset-specific code. No hardcoded values.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Math-NumPy-013243?style=flat-square)](https://numpy.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

---

## What is this?

SLP is a two-pipeline supervised learning suite sharing a common data core. Both pipelines follow identical four-phase workflows — ingestion, target selection, training, and reporting — with task-specific logic for regression and classification kept cleanly separated.

```
supervised-learning-pipelines/
├── core/
│   └── data_cleaner.py              # Shared data engine — both pipelines import from here
├── reusable_regression_pipeline.py  # RRP — continuous target prediction
├── common_classification_pipeline.py # CCP — discrete class prediction
├── pipeline_settings.json           # Single config file controls both pipelines
└── requirements.txt
```

---

## The Two Pipelines

### 📈 RRP — Reusable Regression Pipeline
*Point it at any CSV. Select a continuous target. Get a fully evaluated linear regression model.*

```
python reusable_regression_pipeline.py

Select Mode:
  [1] Train a new Regression Model on a CSV dataset
  [2] Load an existing saved pipeline (.pkl) for Live Predictions
```

**Output:**
- R² score (held-out test set + 5-fold CV mean)
- MAE and RMSE
- Feature impact ranking with visual coefficient bar graph
- Serialized `.pkl` pipeline for live inference

---

### 🎯 CCP — Common Classification Pipeline
*Point it at any CSV. Select a discrete target. Get a fully evaluated logistic classifier — binary or multi-class.*

```
python common_classification_pipeline.py

Select Executive Workflow Mode:
  [1] Train a new Logistic Classification Model
  [2] Spin up Live Predict Engine from a saved (.pkl) asset
```

**Output:**
- Accuracy, Precision, Recall, F1-Score
- Stratified 5-fold CV accuracy mean
- Feature log-odds influence ranking with visual bar graph
- Serialized `.pkl` pipeline + companion label encoder for original class name decoding

---

## Shared Data Core

Both pipelines import from `core/data_cleaner.py` — a universal data engine that handles:

- **Numeric filtering** — retains only columns that are at least 50% numeric (configurable)
- **ID column detection** — drops sequential integer primary keys using a name token + integer wholeness heuristic, avoiding false drops on legitimate features
- **Missing value imputation** — mean for numeric columns, mode for categorical columns
- **Categorical target encoding** — auto-detects text class labels and encodes them to integers via `LabelEncoder`, with the encoder saved alongside the pipeline for inverse decoding during inference
- **One-hot encoding** — automatically converts categorical feature columns to binary integer columns before training

---

## How It Works

Both pipelines run four sequential phases:

**Phase 1 — Universal Data Ingestion**
Loads any CSV, filters columns, detects and drops ID keys, imputes missing values, and prints a full parsing summary dashboard.

**Phase 2 — Interactive Target Selection**
Lists all cleaned columns with index numbers. Accepts target by name or number. Auto-encodes text targets for CCP. Assigns all remaining columns as features.

**Phase 3 — Training & Cross-Validation**

| | RRP | CCP |
|--|-----|-----|
| Model | `LinearRegression` | `LogisticRegression` |
| CV Strategy | `KFold` (shuffled) | `StratifiedKFold` (class-balanced) |
| Primary Metric | R² | Accuracy |

**Phase 4 — Analytics Dashboard**
Prints a full performance report with metrics, CV scores, and a ranked feature impact table with a visual bar graph.

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run either pipeline

```bash
python reusable_regression_pipeline.py
python common_classification_pipeline.py
```

### 3. Live inference from a saved pipeline

Both pipelines offer a predict mode at launch — load any saved `.pkl` and either:
- Batch predict from a new CSV file
- Manually enter feature values for single-row inference

For CCP with text class labels, the pipeline auto-saves a companion `ccp_encoder_{target}.pkl` — keep both files together for original label decoding during inference.

---

## Configuration

A single `pipeline_settings.json` controls both pipelines:

| Parameter | What it controls |
|-----------|-----------------|
| `test_split_ratio` | Fraction of data held out for final evaluation (default: 0.2) |
| `random_state` | Seed for reproducible splits |
| `numeric_threshold` | Minimum numeric ratio for a column to be retained (default: 0.50) |
| `cross_validation_folds` | Number of CV folds (default: 5) |
| `scaling_enabled` | Toggle `StandardScaler` normalization before training (default: true) |

All parameters have safe fallback defaults — both pipelines run correctly even without the config file.

---

## Roadmap

- [ ] Ridge and Polynomial regression modes via `pipeline_settings.json` recipe key
- [ ] Decision tree and random forest classification support
- [ ] Correlation matrix report in Phase 1
- [ ] HTML report export alongside terminal dashboard
- [ ] Unified SLP runner — single entry point that routes to RRP or CCP based on config

---

## Limitations

- RRP supports only linear regression — non-linear relationships will produce poor R² scores
- CCP supports only logistic regression — complex decision boundaries may need tree-based models
- Mean/mode imputation may introduce bias on datasets with significant missingness
- One-hot encoding expands feature space significantly on high-cardinality categorical columns

---

*Part of a broader portfolio of config-driven data science tools — built to explore reusable supervised learning workflow architecture.*
