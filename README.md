# Brain Cancer Outcome Classifier

[![CI](https://github.com/Ajafarnezhad/brain-cancer-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajafarnezhad/brain-cancer-classifier/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

A production-style, end-to-end machine learning pipeline that predicts brain
tumor patient survival outcomes from clinical and molecular-pathology
markers (IDH mutation status, MGMT methylation, WHO diagnostic
classification, and related biomarkers), built with scikit-learn.

## Why this project

Small clinical cohorts are a common, high-stakes setting for applied ML: the
dataset is tiny (~60 patients), heterogeneous (mixed categorical/numeric
biomarkers, missing values), and every modeling choice — how you impute,
encode, select features, and validate — directly affects whether reported
performance is trustworthy. This repository is built around getting those
choices right and making them inspectable, rather than optimizing a single
leaderboard number.

## Highlights

- **One inference-safe pipeline.** Imputation, categorical encoding, scaling,
  feature selection, and the classifier all live inside a single
  `sklearn.pipeline.Pipeline`. The exact transformations fit during training
  are what run at prediction time — there is no separate, silently
  inconsistent preprocessing step at inference.
- **Honest validation.** Stratified train/test split plus stratified
  cross-validation computed only on the training fold, so the reported
  cross-validation score is never contaminated by the held-out test set.
- **Hyperparameter search.** `GridSearchCV` tunes feature-selection width and
  Gradient Boosting depth/learning-rate/estimator-count jointly with the
  preprocessing pipeline.
- **Interpretability.** SHAP feature importance, ROC/AUC, and a confusion
  matrix are generated automatically after training.
- **Exploratory analysis.** Correlation heatmap, outcome distribution, and a
  UMAP projection of the numeric biomarkers.
- **Typed, tested, documented package.** A proper `src/` layout, a
  `pytest` suite covering data cleaning, the pipeline, and the CLI end to
  end, GitHub Actions CI across three Python versions, and a
  [data card](docs/DATA_CARD.md) describing the dataset and its caveats.
- **A real CLI**, installable as `brain-cancer-classifier`, with `train`,
  `predict`, and `explore` subcommands.

## Installation

```bash
git clone https://github.com/Ajafarnezhad/brain-cancer-classifier.git
cd brain-cancer-classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.9+.

## Usage

### Train

```bash
brain-cancer-classifier train \
  --data data/brain_cancer_clinical_data.csv \
  --model-path models/brain_cancer_pipeline.joblib \
  --metrics-path results/metrics.json \
  --plots-dir results/plots
```

This fits the pipeline with grid-searched hyperparameters, evaluates it on a
held-out stratified test split, writes evaluation plots (confusion matrix,
ROC curve, SHAP importance) to `results/plots/`, and saves the fitted
pipeline and a `metrics.json` summary.

### Predict

```bash
brain-cancer-classifier predict \
  --input path/to/new_patients.csv \
  --model-path models/brain_cancer_pipeline.joblib \
  --output results/predictions.csv
```

`new_patients.csv` must contain the same feature columns as the training
data (see the [data card](docs/DATA_CARD.md) for the schema); extra columns
such as a patient ID are preserved but ignored by the model. Output adds
`predicted_event_death` and `predicted_probability` columns.

### Explore

```bash
brain-cancer-classifier explore --data data/brain_cancer_clinical_data.csv --plots-dir results/plots
```

Run `brain-cancer-classifier --help` or `brain-cancer-classifier <command> --help` for the full option list.

## Project layout

```
brain-cancer-classifier/
├── src/brain_cancer_classifier/
│   ├── config.py           # schema, paths, hyperparameter grid
│   ├── data_loading.py     # CSV loading + missing-value cleanup
│   ├── pipeline.py         # ColumnTransformer + model pipeline
│   ├── train.py            # search, fit, evaluate, persist
│   ├── evaluate.py         # metrics + confusion matrix / ROC / SHAP plots
│   ├── visualize.py        # EDA plots (heatmap, distribution, UMAP)
│   ├── predict.py          # inference on new data
│   └── cli.py              # train / predict / explore CLI
├── tests/                  # pytest suite (data, pipeline, CLI)
├── data/                   # dataset + data card
├── docs/DATA_CARD.md       # dataset schema, provenance, ethics notes
├── .github/workflows/ci.yml
├── pyproject.toml
└── requirements.txt
```

## Dataset

See [`docs/DATA_CARD.md`](docs/DATA_CARD.md) for the full schema, column
dictionary, and an important note on this being a real (not synthetic)
clinical dataset from a single small cohort — read it before drawing
conclusions from model performance or reusing the data elsewhere.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest --cov=brain_cancer_classifier --cov-report=term-missing
```

CI runs the same lint + test suite on every push and pull request across
Python 3.10–3.12.

## Disclaimer

This project is for research and educational purposes only. It is **not** a
validated clinical decision-support tool and must not be used to inform
diagnosis or treatment.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Amirhossein Jafarnezhad** — [github.com/Ajafarnezhad](https://github.com/Ajafarnezhad)
