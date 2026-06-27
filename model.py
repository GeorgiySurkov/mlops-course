"""Pure ML logic for the sentiment classifier.

No ClearML here, so it can be unit-tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")  # headless, so it works on the agent and in CI
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402


@dataclass
class HParams:
    max_features: int = 10000
    ngram_min: int = 1
    ngram_max: int = 1
    C: float = 1.0
    max_iter: int = 300
    solver: str = "liblinear"


def build_pipeline(hp: HParams) -> Pipeline:
    """One TF-IDF + LogReg pipeline, so the model is a single pickle that sklearn serving can load directly."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=hp.max_features,
                    ngram_range=(hp.ngram_min, hp.ngram_max),
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(C=hp.C, max_iter=hp.max_iter, solver=hp.solver),
            ),
        ]
    )


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def confusion_figure(y_true, y_pred, class_names: Sequence[str], title: str) -> "plt.Figure":
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=list(class_names),
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig
