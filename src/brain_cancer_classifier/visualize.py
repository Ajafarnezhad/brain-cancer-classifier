"""Exploratory data analysis plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import NUMERICAL_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "correlation_heatmap.png"
    numeric_df = df[NUMERICAL_COLUMNS + [TARGET_COLUMN]].apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (Numerical Features)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", path)
    return path


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "event_death_distribution.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET_COLUMN, data=df, ax=ax)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(["Survived", "Deceased"])
    ax.set_title("Outcome Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved outcome distribution plot to %s", path)
    return path


def plot_umap_projection(df: pd.DataFrame, output_dir: Path, random_state: int = 42) -> Path | None:
    """Best-effort 2D UMAP projection; skipped gracefully if umap-learn is absent."""
    try:
        import umap
    except ImportError:
        logger.warning("`umap-learn` is not installed; skipping UMAP projection plot.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "umap_projection.png"

    numeric_df = df[NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(
        df[NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce").median()
    )
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(numeric_df)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=df[TARGET_COLUMN].map({0: "Survived", 1: "Deceased"}),
        palette="deep",
        ax=ax,
    )
    ax.set_title("UMAP Projection of Numerical Features")
    ax.legend(title="Outcome")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved UMAP projection to %s", path)
    return path


def run_exploration(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate the full exploratory-data-analysis plot suite."""
    plot_correlation_heatmap(df, output_dir)
    plot_target_distribution(df, output_dir)
    plot_umap_projection(df, output_dir)


__all__ = [
    "plot_correlation_heatmap",
    "plot_target_distribution",
    "plot_umap_projection",
    "run_exploration",
]
