from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


RESULTS_REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}

FIXTURE_REQUIRED_COLUMNS = {
    "match_id",
    "stage",
    "group",
    "match_date",
    "venue",
    "city",
    "country",
    "home_team",
    "away_team",
    "neutral",
    "bracket_slot_home",
    "bracket_slot_away",
}

DEFAULT_TEAM_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "us": "United States",
    "ir iran": "Iran",
    "islamic republic of iran": "Iran",
    "republic of korea": "South Korea",
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "pr korea": "North Korea",
    "cote d ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "turkiye": "Turkey",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "russia": "Russia",
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern ireland": "Northern Ireland",
    "republic of ireland": "Ireland",
}


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    issues: list[str]


def canonicalize_team_name(team_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(team_name))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", normalized).strip().lower()
    return normalized


def load_team_mappings(mapping_path: Path | None = None) -> dict[str, str]:
    mappings = dict(DEFAULT_TEAM_ALIASES)
    if mapping_path is None or not mapping_path.exists():
        return mappings

    frame = pd.read_csv(mapping_path)
    columns = {column.lower(): column for column in frame.columns}
    source_column = columns.get("source_name") or columns.get("alias") or columns.get("source")
    target_column = columns.get("target_name") or columns.get("canonical_name") or columns.get("target")
    if source_column is None or target_column is None:
        raise ValueError(
            f"Team mapping file {mapping_path} must contain source_name/target_name columns or equivalents."
        )

    for _, row in frame.iterrows():
        source_name = canonicalize_team_name(row[source_column])
        target_name = str(row[target_column]).strip()
        if source_name and target_name:
            mappings[source_name] = target_name
    return mappings


def normalize_team_name(team_name: str, mappings: dict[str, str]) -> str:
    canonical = canonicalize_team_name(team_name)
    return mappings.get(canonical, str(team_name).strip())


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", ""}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def ensure_required_columns(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing_columns = set(required_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {sorted(missing_columns)}")


def validate_results_dataframe(
    results_frame: pd.DataFrame,
    prediction_cutoff_date: datetime | pd.Timestamp | None = None,
) -> ValidationResult:
    issues: list[str] = []
    ensure_required_columns(results_frame, RESULTS_REQUIRED_COLUMNS, "Historical results")

    parsed_dates = pd.to_datetime(results_frame["date"], errors="coerce")
    if parsed_dates.isna().any():
        issues.append("Some historical match dates could not be parsed.")

    numeric_home = pd.to_numeric(results_frame["home_score"], errors="coerce")
    numeric_away = pd.to_numeric(results_frame["away_score"], errors="coerce")
    if numeric_home.isna().any() or numeric_away.isna().any():
        issues.append("Some historical match scores are missing or non-numeric.")
    elif (numeric_home < 0).any() or (numeric_away < 0).any():
        issues.append("Historical match scores cannot be negative.")

    missing_teams = results_frame[["home_team", "away_team"]].isna().any().any()
    empty_teams = results_frame[["home_team", "away_team"]].astype(str).apply(lambda column: column.str.strip().eq("")).any().any()
    if missing_teams or empty_teams:
        issues.append("Historical results contain missing team names.")

    duplicate_mask = results_frame.duplicated(
        subset=["date", "home_team", "away_team", "home_score", "away_score", "tournament"],
        keep=False,
    )
    if duplicate_mask.any():
        issues.append(f"{int(duplicate_mask.sum())} potential duplicate historical matches found.")

    if prediction_cutoff_date is not None and not parsed_dates.isna().all():
        cutoff = pd.Timestamp(prediction_cutoff_date)
        if (parsed_dates > cutoff).any():
            issues.append("Historical training data includes matches after the prediction cutoff date.")

    return ValidationResult(is_valid=not issues, issues=issues)


def validate_fixtures_dataframe(
    fixtures_frame: pd.DataFrame,
    team_mappings: dict[str, str],
    known_teams: set[str] | None = None,
) -> ValidationResult:
    issues: list[str] = []
    ensure_required_columns(fixtures_frame, FIXTURE_REQUIRED_COLUMNS, "World Cup 2026 fixtures")

    parsed_dates = pd.to_datetime(fixtures_frame["match_date"], errors="coerce")
    if parsed_dates.isna().any():
        issues.append("Some fixture dates could not be parsed.")
    if fixtures_frame.empty:
        issues.append("The World Cup 2026 fixture file contains no fixtures.")
    if fixtures_frame["match_id"].duplicated().any():
        issues.append("Fixture match_id values must be unique.")

    group_mask = fixtures_frame["stage"].astype(str).str.contains("group", case=False, na=False)
    for column, slot_column in [("home_team", "bracket_slot_home"), ("away_team", "bracket_slot_away")]:
        missing_team = fixtures_frame[column].isna() | fixtures_frame[column].astype(str).str.strip().eq("")
        missing_slot = fixtures_frame[slot_column].isna() | fixtures_frame[slot_column].astype(str).str.strip().eq("")
        if (missing_team & (group_mask | missing_slot)).any():
            issues.append(f"Fixtures contain unresolved values in {column}.")

    try:
        fixtures_frame["neutral"].map(parse_bool)
    except ValueError as exc:
        issues.append(str(exc))

    group_fixtures = fixtures_frame.loc[group_mask]
    if not group_fixtures.empty:
        group_counts = group_fixtures.groupby("group").size()
        if len(group_counts) != 12 or not group_counts.eq(6).all():
            issues.append("Group-stage fixtures must contain exactly 12 groups with 6 matches each.")

    fixed_home = fixtures_frame.loc[
        ~fixtures_frame["home_team"].isna() & fixtures_frame["home_team"].astype(str).str.strip().ne(""),
        "home_team",
    ]
    fixed_away = fixtures_frame.loc[
        ~fixtures_frame["away_team"].isna() & fixtures_frame["away_team"].astype(str).str.strip().ne(""),
        "away_team",
    ]
    normalized_home = fixed_home.astype(str).map(lambda value: normalize_team_name(value, team_mappings))
    normalized_away = fixed_away.astype(str).map(lambda value: normalize_team_name(value, team_mappings))

    if known_teams is not None:
        missing_teams = sorted(set(normalized_home).union(set(normalized_away)) - set(known_teams))
        if missing_teams:
            issues.append(
                "Fixtures contain teams that are not known from the historical dataset or mapping file: "
                + ", ".join(missing_teams[:20])
            )

    return ValidationResult(is_valid=not issues, issues=issues)


def prepare_team_columns(frame: pd.DataFrame, team_mappings: dict[str, str]) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["home_team"] = cleaned["home_team"].astype(str).map(lambda value: normalize_team_name(value, team_mappings))
    cleaned["away_team"] = cleaned["away_team"].astype(str).map(lambda value: normalize_team_name(value, team_mappings))
    if "neutral" in cleaned.columns:
        cleaned["neutral"] = cleaned["neutral"].map(parse_bool)
    return cleaned
