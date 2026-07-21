# ⚙️ Supervised Learning Pipelines (SLP)

> A reusable machine learning pipeline for regression and classification using configurable preprocessing and evaluation workflows.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Numerical-NumPy-013243?style=flat-square)](https://numpy.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Overview

SLP provides reusable pipelines for supervised learning tasks by combining configurable preprocessing, model training, evaluation, and model serialization. The project includes separate workflows for regression and classification while sharing a common data preparation module.

The project was built to explore how reusable machine learning workflows can be structured independently of any specific dataset.

---

## Features

- Regression and classification pipelines
- Shared data preprocessing module
- Automatic handling of missing values and categorical features
- Cross-validation and evaluation metrics
- Model serialization for later inference

---

## Pipeline

```text
CSV Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Feature & Target Selection
     │
     ▼
Model Training
     │
     ▼
Evaluation
     │
     ▼
Saved Pipeline
```

---

## Tech Stack

- Python
- scikit-learn
- pandas
- NumPy
- Joblib
- JSON Configuration

---

## Project Structure

```text
supervised_learning_pipelines/
├── core/
│   └── data_cleaner.py
├── reusable_regression_pipeline.py
├── common_classification_pipeline.py
├── pipeline_settings.json
└── requirements.txt
```

---

## Design

SLP separates data preprocessing from model-specific workflows, allowing regression and classification pipelines to reuse the same preparation logic. Configuration files are used to control preprocessing and evaluation settings without modifying the implementation.

---

## Limitations

- Supports linear regression and logistic regression only.
- Performance depends on dataset quality and feature engineering.
- Mean and mode imputation may not be suitable for every dataset.

---

## Future Improvements

- Additional regression and classification algorithms
- Feature selection utilities
- Automated experiment reports
- Unified command-line interface
- Hyperparameter tuning support

---

*Built as an exploration of reusable supervised learning workflows.*
