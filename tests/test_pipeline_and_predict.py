from __future__ import annotations

from pathlib import Path

from joblib import dump, load

from brain_cancer_classifier.data_loading import load_dataset, split_features_target
from brain_cancer_classifier.pipeline import build_pipeline
from brain_cancer_classifier.predict import predict


def test_pipeline_fits_and_predicts(synthetic_csv: Path) -> None:
    df = load_dataset(synthetic_csv)
    X, y = split_features_target(df)

    pipeline = build_pipeline(k_best=5)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert proba.shape == (len(X), 2)


def test_predict_roundtrips_through_disk(tmp_path: Path, synthetic_csv: Path) -> None:
    df = load_dataset(synthetic_csv)
    X, y = split_features_target(df)

    pipeline = build_pipeline(k_best=5)
    pipeline.fit(X, y)

    model_path = tmp_path / "model.joblib"
    dump(pipeline, model_path)
    reloaded = load(model_path)

    input_df = X.copy()
    input_df["Patient"] = range(len(input_df))  # extra column must be tolerated

    result = predict(reloaded, input_df)
    assert "predicted_event_death" in result.columns
    assert "predicted_probability" in result.columns
    assert set(result["predicted_event_death"].unique()).issubset({0, 1})
    assert result["predicted_probability"].between(0, 1).all()


def test_predict_rejects_missing_columns(synthetic_csv: Path) -> None:
    df = load_dataset(synthetic_csv)
    X, y = split_features_target(df)

    pipeline = build_pipeline(k_best=5)
    pipeline.fit(X, y)

    incomplete = X.drop(columns=[X.columns[0]])
    try:
        predict(pipeline, incomplete)
        assert False, "expected ValueError for missing columns"
    except ValueError:
        pass
