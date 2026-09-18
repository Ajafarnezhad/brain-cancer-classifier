"""Training orchestration: search, fit, evaluate, and persist the model."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from joblib import dump
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import PARAM_GRID, RANDOM_STATE
from .data_loading import load_dataset, split_features_target
from .evaluate import evaluate_model
from .pipeline import build_pipeline

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    best_params: dict
    cv_f1_mean: float
    cv_f1_std: float
    test_metrics: dict
    n_train: int
    n_test: int

    def to_dict(self) -> dict:
        return asdict(self)


def train(
    data_path: str,
    model_path: str,
    metrics_path: str,
    plots_dir: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = RANDOM_STATE,
    with_shap: bool = True,
) -> TrainingResult:
    """Train, tune, evaluate, and persist the full inference pipeline.

    The returned model is a single fitted ``sklearn.pipeline.Pipeline`` that
    embeds preprocessing, feature selection, and the classifier, so it can be
    reloaded and applied to new raw data without any separate fitting step.
    """
    df = load_dataset(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model: Pipeline = grid_search.best_estimator_
    logger.info("Best hyperparameters: %s", grid_search.best_params_)

    # Cross-validated F1 on the training split only (never touches the held-out
    # test set), giving an honest estimate of generalization for the chosen
    # hyperparameters before final evaluation.
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="f1")
    logger.info("Train-set CV F1: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    plots_dir_path = Path(plots_dir)
    test_metrics = evaluate_model(best_model, X_test, y_test, plots_dir_path, with_shap=with_shap)

    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, model_path_obj)
    logger.info("Saved trained pipeline to %s", model_path_obj)

    result = TrainingResult(
        best_params=grid_search.best_params_,
        cv_f1_mean=float(cv_scores.mean()),
        cv_f1_std=float(cv_scores.std()),
        test_metrics=test_metrics,
        n_train=len(X_train),
        n_test=len(X_test),
    )

    metrics_path_obj = Path(metrics_path)
    metrics_path_obj.parent.mkdir(parents=True, exist_ok=True)
    metrics_path_obj.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Saved metrics to %s", metrics_path_obj)

    return result


__all__ = ["TrainingResult", "train"]
