"""Static configuration: schema, default paths, and reproducibility settings."""

from __future__ import annotations

from dataclasses import dataclass

RANDOM_STATE = 42

TARGET_COLUMN = "Event_death"

ID_COLUMNS = ["Patient"]

# Categorical / ordinal clinical and molecular markers (encoded as small integer codes
# in the source data, e.g. mutation status or pathology grade).
CATEGORICAL_COLUMNS = [
    "gender",
    "TMZ",
    "Radiology",
    "DxWHO2007",
    "IDH1_molecular",
    "H3.3K27M",
    "H3.3G34R",
    "BRAFV600E",
    "IntegratedDxStep1",
    "EGFR",
    "IDH1_2",
    "IDH1_tarkibi",
    "IDH2",
    "H3F3A",
    "MGMT",
    "V600E",
    "EGFR_A",
    "@1p19q",
    "PTEN",
    "Integrateddxstep2",
    "CD44",
    "MGMT_new",
]

# Continuous / count clinical measurements.
NUMERICAL_COLUMNS = [
    "Age",
    "timefordeath",
    "time_Recurrence",
    "PCV_new",
    "Event_reccurrence",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS


@dataclass(frozen=True)
class ProjectPaths:
    """Central place for default filesystem locations used by the CLI."""

    data_path: str = "data/brain_cancer_clinical_data.csv"
    model_dir: str = "models"
    model_path: str = "models/brain_cancer_pipeline.joblib"
    metrics_path: str = "results/metrics.json"
    predictions_path: str = "results/predictions.csv"
    plots_dir: str = "results/plots"


DEFAULT_PATHS = ProjectPaths()

#: Hyperparameter grid searched during training. Kept intentionally small so the
#: default `train` command finishes in well under a minute on a laptop CPU while
#: still exercising the most impactful Gradient Boosting knobs.
PARAM_GRID: dict = {
    "feature_selection__k": [5, 10, "all"],
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [2, 3, 4],
    "classifier__learning_rate": [0.05, 0.1],
}

__all__ = [
    "CATEGORICAL_COLUMNS",
    "DEFAULT_PATHS",
    "FEATURE_COLUMNS",
    "ID_COLUMNS",
    "NUMERICAL_COLUMNS",
    "PARAM_GRID",
    "RANDOM_STATE",
    "TARGET_COLUMN",
    "ProjectPaths",
]
