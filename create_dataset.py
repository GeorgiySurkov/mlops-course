"""Stage 1: register the prepared CSVs as a versioned ClearML Dataset.

train.py fetches it by project and name, so no dataset_id is hardcoded.
"""
from __future__ import annotations

import pathlib
import sys

from clearml import Dataset

from config import cfg


def main() -> None:
    data_dir = pathlib.Path("data")
    train_csv, test_csv = data_dir / "train.csv", data_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        sys.exit("data/train.csv or data/test.csv missing — run `python prepare_data.py` first")

    dataset = Dataset.create(
        dataset_name=cfg.clearml.dataset_name,
        dataset_project=cfg.clearml.dataset_project,
        dataset_tags=["imdb", "sentiment"],
        description="Balanced IMDB sentiment subset (text,label) for the MLOps course.",
    )
    dataset.add_files(path=str(train_csv))
    dataset.add_files(path=str(test_csv))
    dataset.upload(show_progress=True)
    dataset.finalize()

    print(f"created ClearML Dataset id={dataset.id} version={dataset.version}")
    print(f"  project='{cfg.clearml.dataset_project}' name='{cfg.clearml.dataset_name}'")


if __name__ == "__main__":
    main()
