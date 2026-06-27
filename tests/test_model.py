"""Offline tests for the pure ML logic. No ClearML or server needed."""
from __future__ import annotations

import pathlib
import sys

import matplotlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model import HParams, build_pipeline, confusion_figure, evaluate  # noqa: E402

TRAIN_TEXTS = [
    "fantastic movie loved every minute",
    "brilliant acting and a wonderful story",
    "great film highly recommend it",
    "an excellent and delightful experience",
    "terrible boring waste of time",
    "awful acting and a dull plot",
    "horrible movie i hated it",
    "bad film do not watch",
]
TRAIN_LABELS = [1, 1, 1, 1, 0, 0, 0, 0]
TEST_TEXTS = ["a wonderful delightful film", "boring and awful waste"]
TEST_LABELS = [1, 0]


def test_pipeline_trains_and_predicts():
    hp = HParams(max_features=50, ngram_min=1, ngram_max=2, C=1.0, max_iter=200)
    pipe = build_pipeline(hp)
    pipe.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = pipe.predict(TEST_TEXTS)
    assert list(preds) == TEST_LABELS


def test_evaluate_returns_metrics():
    metrics = evaluate(TEST_LABELS, TEST_LABELS)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert set(metrics) == {"accuracy", "f1"}


def test_confusion_figure_is_a_figure():
    fig = confusion_figure(TEST_LABELS, TEST_LABELS, ["negative", "positive"], "t")
    assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == "__main__":
    test_pipeline_trains_and_predicts()
    test_evaluate_returns_metrics()
    test_confusion_figure_is_a_figure()
    print("all offline model tests passed")
