"""Stage 1 helper: build a small, balanced IMDB sentiment subset as CSV.

Kept small and balanced so it trains on CPU.
"""
from __future__ import annotations

import pathlib

import pandas as pd
from datasets import load_dataset

from config import cfg


def _balanced(split, per_class: int, seed: int) -> pd.DataFrame:
    df = split.to_pandas()[["text", "label"]]
    parts = [
        df[df["label"] == label].sample(n=per_class, random_state=seed)
        for label in sorted(df["label"].unique())
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)

    ds = load_dataset(cfg.data.hf_dataset)
    train = _balanced(ds["train"], cfg.data.train_per_class, cfg.data.random_seed)
    test = _balanced(ds["test"], cfg.data.test_per_class, cfg.data.random_seed)

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    print(f"wrote {len(train)} train / {len(test)} test rows to {data_dir}/")


if __name__ == "__main__":
    main()
