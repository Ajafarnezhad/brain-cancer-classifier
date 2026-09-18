"""Model pipeline construction.

Everything the model needs at inference time -- imputation, encoding, scaling,
feature selection and the classifier itself -- lives inside a single
``sklearn.pipeline.Pipeline``. This is the key correctness fix over a naive
implementation that fits imputers/encoders separately at train time and then
*re-fits fresh ones on whatever data is passed to predict()*: that pattern
silently invalidates predictions because the encoding learned in training is
never reused. Persisting one fitted pipeline object guarantees training and
inference always apply identical transformations.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    """Build the column-wise imputation/encoding/scaling transformer."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numerical", numerical_pipeline, NUMERICAL_COLUMNS),
        ]
    )


def build_pipeline(k_best: int | str = 10) -> Pipeline:
    """Build the full preprocessing + feature-selection + classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("feature_selection", SelectKBest(score_func=f_classif, k=k_best)),
            (
                "classifier",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )


__all__ = ["build_pipeline", "build_preprocessor"]
