from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.validate_data import validate_results_dataframe
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import MANUAL_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)


@dataclass(frozen=True)
class FetchMetadata:
    source: str
    fetched_at: str
    rows: int
    file_name: str
    dataset_slug: str | None = None


def _write_metadata(metadata_path: Path, metadata: FetchMetadata) -> None:
    metadata_path.write_text(json.dumps(metadata.__dict__, indent=2), encoding="utf-8")


def _copy_local_results(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists():
        LOGGER.info("Raw results already exist at %s; not overwriting.", destination_path)
        return
    shutil.copy2(source_path, destination_path)


def _download_results_from_kaggle(destination_dir: Path, dataset_slug: str) -> Path:
    try:
        import kaggle  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Kaggle download is unavailable because the kaggle package is not installed. "
            "Install it or place results.csv in data/raw/ manually."
        ) from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_slug, path=str(destination_dir), unzip=True, quiet=False)

    for candidate in destination_dir.glob("**/results.csv"):
        return candidate
    raise FileNotFoundError(
        f"Kaggle download completed but results.csv was not found under {destination_dir}."
    )


def fetch_results_csv(dataset_slug: str | None = None) -> Path:
    ensure_project_dirs()
    raw_results_path = RAW_DATA_DIR / "results.csv"

    if raw_results_path.exists():
        LOGGER.info("Using existing raw results at %s", raw_results_path)
        return raw_results_path

    local_manual_path = MANUAL_DATA_DIR / "results.csv"
    if local_manual_path.exists():
        LOGGER.info("Copying manual results file from %s", local_manual_path)
        _copy_local_results(local_manual_path, raw_results_path)
        return raw_results_path

    resolved_slug = (
        dataset_slug
        or os.environ.get("KAGGLE_DATASET_SLUG", "").strip()
        or str(config_value("data", "kaggle_dataset_slug", default="")).strip()
    )
    if resolved_slug:
        LOGGER.info("Attempting Kaggle download for dataset %s", resolved_slug)
        downloaded_path = _download_results_from_kaggle(RAW_DATA_DIR, resolved_slug)
        if downloaded_path != raw_results_path:
            shutil.copy2(downloaded_path, raw_results_path)
        return raw_results_path

    raise FileNotFoundError(
        "Historical results.csv is missing. Download the Kaggle dataset 'International football results from 1872 to 2026' "
            "manually and place results.csv in data/raw/, or configure a Kaggle dataset slug and credentials."
    )


def fetch_data(dataset_slug: str | None = None) -> Path:
    existed_before_fetch = (RAW_DATA_DIR / "results.csv").exists()
    manual_source_exists = (MANUAL_DATA_DIR / "results.csv").exists()
    results_path = fetch_results_csv(dataset_slug=dataset_slug)
    frame = pd.read_csv(results_path)
    validation = validate_results_dataframe(frame)
    if not validation.is_valid:
        LOGGER.warning("Validation issues detected in raw results: %s", validation.issues)

    metadata = FetchMetadata(
        source="cached_raw" if existed_before_fetch else "manual" if manual_source_exists else "kaggle",
        fetched_at=datetime.now(timezone.utc).isoformat(),
        rows=int(len(frame)),
        file_name=results_path.name,
        dataset_slug=dataset_slug,
    )
    _write_metadata(RAW_DATA_DIR / "results_metadata.json", metadata)
    return results_path


if __name__ == "__main__":
    fetch_data()
