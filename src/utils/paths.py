from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANUAL_DATA_DIR = DATA_DIR / "manual"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
INTERNAL_DOCS_DIR = DOCS_DIR / "internal"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_project_dirs() -> None:
    for directory in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MANUAL_DATA_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        DOCS_DIR,
        INTERNAL_DOCS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def resolve_manual_path(*parts: str) -> Path:
    return MANUAL_DATA_DIR.joinpath(*parts)


def resolve_processed_path(*parts: str) -> Path:
    return PROCESSED_DATA_DIR.joinpath(*parts)


def resolve_predictions_path(*parts: str) -> Path:
    return PREDICTIONS_DIR.joinpath(*parts)


def resolve_models_path(*parts: str) -> Path:
    return MODELS_DIR.joinpath(*parts)
