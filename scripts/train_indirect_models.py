from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from src.models.train_indirect_models import train_indirect_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe indirect team models.")
    parser.add_argument("--cutoff-date", required=True)
    args = parser.parse_args()
    train_indirect_models(args.cutoff_date)


if __name__ == "__main__":
    main()
