from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import pandas as pd

from src.data.build_dataset import build_training_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training and fixture features.")
    parser.add_argument("--cutoff-date", required=True, help="Prediction cutoff date in YYYY-MM-DD format.")
    args = parser.parse_args()
    cutoff_date = pd.Timestamp(args.cutoff_date)
    build_training_features(cutoff_date)


if __name__ == "__main__":
    main()
