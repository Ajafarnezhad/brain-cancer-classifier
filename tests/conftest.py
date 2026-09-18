from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain_cancer_classifier.config import CATEGORICAL_COLUMNS, ID_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    """A small, fully synthetic dataset matching the project's schema.

    Deliberately includes a blank-string missing value to exercise the
    project's cleanup logic without touching the real clinical dataset.
    """
    rng = np.random.default_rng(0)
    n = 40

    data = {col: rng.integers(0, n) for col in ID_COLUMNS}
    data = {ID_COLUMNS[0]: np.arange(n)}
    for col in CATEGORICAL_COLUMNS:
        data[col] = rng.integers(1, 4, size=n).astype(object)
    for col in NUMERICAL_COLUMNS:
        data[col] = rng.integers(1, 90, size=n).astype(float)
    data[TARGET_COLUMN] = rng.integers(0, 2, size=n)

    df = pd.DataFrame(data)
    df.loc[0, CATEGORICAL_COLUMNS[0]] = " "  # simulate the real dataset's blank-cell artifact
    return df


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_dataframe: pd.DataFrame) -> Path:
    path = tmp_path / "synthetic.csv"
    synthetic_dataframe.to_csv(path, index=False)
    return path
