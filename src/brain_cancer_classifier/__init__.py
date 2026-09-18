"""Brain Cancer Outcome Classifier.

An end-to-end, production-style machine learning pipeline for predicting
brain cancer patient survival outcomes from clinical and molecular markers.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brain-cancer-classifier")
except PackageNotFoundError:  # pragma: no cover - local/dev checkout
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
