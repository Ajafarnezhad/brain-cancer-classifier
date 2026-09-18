"""Dataset loading and cleaning utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, ID_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the input dataset does not match the expected schema."""


def _blank_to_na(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize whitespace-only cells (a real artifact of this dataset) to NaN.

    The source CSV encodes a handful of missing clinical values as a single
    space character rather than an empty cell, which silently defeats
    pandas'/scikit-learn's default missing-value detection. All of these
    columns are numeric category codes, so after stripping blanks we coerce
    to numeric float (using ``np.nan``, not ``pd.NA``) -- scikit-learn's
    imputers require ``np.nan`` for missing-value comparisons to work.
    """
    df = df.copy()
    for col in columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"": None, "nan": None})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the raw clinical dataset and apply minimal, schema-preserving cleanup.

    Parameters
    ----------
    data_path:
        Path to a CSV file matching the project's clinical data schema.

    Returns
    -------
    A cleaned dataframe containing the id column(s), all feature columns and
    the target column, with numeric columns coerced to numeric dtype and
    blank-string missing values normalized to NaN.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise DatasetError(f"Dataset file '{data_path}' was not found.")

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset '%s' with shape %s", data_path, df.shape)

    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise DatasetError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns], errors="ignore")
    df = _blank_to_na(df, CATEGORICAL_COLUMNS)
    df = _blank_to_na(df, NUMERICAL_COLUMNS)

    for col in NUMERICAL_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_missing_target = df[TARGET_COLUMN].isna().sum()
    if n_missing_target:
        logger.warning("Dropping %d rows with missing target label.", n_missing_target)
        df = df.dropna(subset=[TARGET_COLUMN])

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into the feature matrix ``X`` and target ``y``."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


__all__ = ["DatasetError", "load_dataset", "split_features_target"]
