"""Model evaluation: metrics and diagnostic plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI / servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Compute the standard binary-classification metric bundle."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Survived", "Deceased"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)
    return path


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "roc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve to %s", path)
    return path


def plot_shap_importance(model: Pipeline, X_test: pd.DataFrame, output_dir: Path) -> Path | None:
    """Best-effort SHAP feature-importance plot; skipped gracefully if shap is absent."""
    try:
        import shap
    except ImportError:
        logger.warning("`shap` is not installed; skipping SHAP importance plot.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shap_feature_importance.png"

    preprocessor = model.named_steps["preprocessor"]
    selector = model.named_steps["feature_selection"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    selected_mask = selector.get_support()
    selected_names = feature_names[selected_mask]

    X_transformed = selector.transform(preprocessor.transform(X_test))
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=selected_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP feature importance plot to %s", path)
    return path


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
    with_shap: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation suite and return a metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info(
        "Test metrics -> accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )

    plot_confusion_matrix(y_test, y_pred, plots_dir)
    plot_roc_curve(y_test, y_proba, plots_dir)
    if with_shap:
        plot_shap_importance(model, X_test, plots_dir)

    return metrics


__all__ = [
    "compute_metrics",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_shap_importance",
]
