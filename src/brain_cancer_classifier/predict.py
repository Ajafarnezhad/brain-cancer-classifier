"""Inference on new, unlabeled clinical data using a persisted pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from .config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    """Raised when attempting to load a pipeline that has not been trained yet."""


def load_pipeline(model_path: str | Path) -> Pipeline:
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(
            f"No trained model found at '{model_path}'. Run `train` first."
        )
    return load(model_path)


def predict(model: Pipeline, input_df: pd.DataFrame) -> pd.DataFrame:
    """Predict survival outcome for new patients.

    Parameters
    ----------
    model:
        A fitted pipeline as produced by :func:`brain_cancer_classifier.train.train`.
    input_df:
        A dataframe containing at least all columns in
        :data:`brain_cancer_classifier.config.FEATURE_COLUMNS`. Extra columns
        (e.g. an id column) are preserved but ignored by the model.

    Returns
    -------
    A copy of ``input_df`` with two extra columns: ``predicted_event_death``
    (0 = survived, 1 = deceased) and ``predicted_probability`` (model
    confidence that the patient is deceased).
    """
    missing = set(FEATURE_COLUMNS) - set(input_df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    X = input_df[FEATURE_COLUMNS]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = input_df.copy()
    result["predicted_event_death"] = predictions.astype(int)
    result["predicted_probability"] = np.round(probabilities, 4)
    logger.info("Generated %d predictions.", len(result))
    return result


__all__ = ["ModelNotFoundError", "load_pipeline", "predict"]
