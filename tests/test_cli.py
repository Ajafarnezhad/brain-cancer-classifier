from __future__ import annotations

from pathlib import Path

from brain_cancer_classifier.cli import main


def test_train_predict_explore_end_to_end(tmp_path: Path, synthetic_csv: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    plots_dir = tmp_path / "plots"

    exit_code = main(
        [
            "train",
            "--data",
            str(synthetic_csv),
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
            "--plots-dir",
            str(plots_dir),
            "--cv-folds",
            "3",
            "--no-shap",
        ]
    )
    assert exit_code == 0
    assert model_path.exists()
    assert metrics_path.exists()

    predictions_path = tmp_path / "predictions.csv"
    exit_code = main(
        [
            "predict",
            "--input",
            str(synthetic_csv),
            "--model-path",
            str(model_path),
            "--output",
            str(predictions_path),
        ]
    )
    assert exit_code == 0
    assert predictions_path.exists()

    explore_plots_dir = tmp_path / "explore_plots"
    exit_code = main(["explore", "--data", str(synthetic_csv), "--plots-dir", str(explore_plots_dir)])
    assert exit_code == 0


def test_predict_without_trained_model_fails_gracefully(tmp_path: Path, synthetic_csv: Path) -> None:
    exit_code = main(
        [
            "predict",
            "--input",
            str(synthetic_csv),
            "--model-path",
            str(tmp_path / "nonexistent.joblib"),
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    assert exit_code == 1
