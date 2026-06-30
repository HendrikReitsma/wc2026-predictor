from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from src.data.fetch_data import fetch_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or load historical football results.")
    parser.add_argument("--dataset-slug", default=None, help="Optional Kaggle dataset slug.")
    args = parser.parse_args()
    fetch_data(dataset_slug=args.dataset_slug)


if __name__ == "__main__":
    main()
