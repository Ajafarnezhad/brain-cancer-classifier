"""Command-line interface.

Examples
--------
Train the model::

    brain-cancer-classifier train --data data/brain_cancer_clinical_data.csv

Run inference on new data::

    brain-cancer-classifier predict --input new_patients.csv --output results/predictions.csv

Generate exploratory-data-analysis plots::

    brain-cancer-classifier explore --data data/brain_cancer_clinical_data.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import DEFAULT_PATHS, RANDOM_STATE
from .data_loading import DatasetError, load_dataset
from .predict import ModelNotFoundError, load_pipeline, predict
from .train import train
from .visualize import run_exploration

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-cancer-classifier",
        description="Predict brain cancer patient survival outcomes from clinical and molecular markers.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the model.")
    train_parser.add_argument("--data", default=DEFAULT_PATHS.data_path, help="Path to the training CSV.")
    train_parser.add_argument("--model-path", default=DEFAULT_PATHS.model_path, help="Where to save the fitted pipeline.")
    train_parser.add_argument("--metrics-path", default=DEFAULT_PATHS.metrics_path, help="Where to save evaluation metrics (JSON).")
    train_parser.add_argument("--plots-dir", default=DEFAULT_PATHS.plots_dir, help="Directory for evaluation plots.")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test set fraction.")
    train_parser.add_argument("--cv-folds", type=int, default=5, help="Number of stratified cross-validation folds.")
    train_parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed for reproducibility.")
    train_parser.add_argument("--no-shap", action="store_true", help="Skip the (slower) SHAP importance plot.")

    predict_parser = subparsers.add_parser("predict", help="Run inference on new patient data.")
    predict_parser.add_argument("--input", required=True, help="Path to a CSV of new, unlabeled patient records.")
    predict_parser.add_argument("--model-path", default=DEFAULT_PATHS.model_path, help="Path to a trained pipeline.")
    predict_parser.add_argument("--output", default=DEFAULT_PATHS.predictions_path, help="Where to save predictions (CSV).")

    explore_parser = subparsers.add_parser("explore", help="Generate exploratory-data-analysis plots.")
    explore_parser.add_argument("--data", default=DEFAULT_PATHS.data_path, help="Path to the dataset CSV.")
    explore_parser.add_argument("--plots-dir", default=DEFAULT_PATHS.plots_dir, help="Directory for EDA plots.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        if args.command == "train":
            result = train(
                data_path=args.data,
                model_path=args.model_path,
                metrics_path=args.metrics_path,
                plots_dir=args.plots_dir,
                test_size=args.test_size,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                with_shap=not args.no_shap,
            )
            print(f"Best hyperparameters: {result.best_params}")
            print(f"Train CV F1: {result.cv_f1_mean:.3f} ± {result.cv_f1_std:.3f}")
            print(f"Test metrics: {result.test_metrics}")

        elif args.command == "predict":
            model = load_pipeline(args.model_path)
            input_df = pd.read_csv(args.input)
            predictions = predict(model, input_df)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)
            print(f"Wrote {len(predictions)} predictions to {output_path}")

        elif args.command == "explore":
            df = load_dataset(args.data)
            run_exploration(df, Path(args.plots_dir))
            print(f"Wrote exploratory plots to {args.plots_dir}")

    except (DatasetError, ModelNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
