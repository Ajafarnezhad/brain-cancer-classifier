from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from brain_cancer_classifier.config import FEATURE_COLUMNS, TARGET_COLUMN
from brain_cancer_classifier.data_loading import DatasetError, load_dataset, split_features_target


def test_load_dataset_cleans_blank_values(synthetic_csv: Path) -> None:
    df = load_dataset(synthetic_csv)
    assert "Patient" not in df.columns
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns)
    # the injected blank-string cell must not survive as a literal " "
    assert not (df[FEATURE_COLUMNS] == " ").any().any()


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_load_dataset_missing_columns(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"Age": [1, 2]}).to_csv(bad_path, index=False)
    with pytest.raises(DatasetError):
        load_dataset(bad_path)


def test_split_features_target(synthetic_csv: Path) -> None:
    df = load_dataset(synthetic_csv)
    X, y = split_features_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y)
