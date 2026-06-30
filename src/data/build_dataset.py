from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.fetch_data import fetch_data
from src.data.validate_data import (
    load_team_mappings,
    normalize_team_name,
    parse_bool,
    prepare_team_columns,
    validate_fixtures_dataframe,
    validate_results_dataframe,
)
from src.features.feature_engineering import FeatureState, build_feature_rows
from src.utils.logging import setup_logging
from src.utils.paths import MANUAL_DATA_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)
MANUAL_WORLD_CUP_2026_GROUP_RESULTS = MANUAL_DATA_DIR / "worldcup_2026_group_results.csv"


def _load_manual_worldcup_group_results(cutoff_date: datetime | pd.Timestamp | None) -> pd.DataFrame:
    if not MANUAL_WORLD_CUP_2026_GROUP_RESULTS.exists():
        return pd.DataFrame()
    frame = pd.read_csv(MANUAL_WORLD_CUP_2026_GROUP_RESULTS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if cutoff_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(cutoff_date)].copy()
    if frame.empty:
        return frame
    return frame[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ]
    ].copy()


def load_results_frame(cutoff_date: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    results_path = fetch_data()
    results_frame = pd.read_csv(results_path)
    results_frame["date"] = pd.to_datetime(results_frame["date"], errors="coerce")
    if cutoff_date is not None:
        results_frame = results_frame[results_frame["date"] <= pd.Timestamp(cutoff_date)].copy()
    results_frame = results_frame.dropna(subset=["home_score", "away_score"]).copy()
    manual_group_results = _load_manual_worldcup_group_results(cutoff_date)
    if not manual_group_results.empty:
        results_frame = pd.concat([results_frame, manual_group_results], ignore_index=True, sort=False)
    duplicate_columns = ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    duplicate_count = int(results_frame.duplicated(subset=duplicate_columns, keep="first").sum())
    if duplicate_count:
        LOGGER.warning("Dropping %s duplicate historical match rows.", duplicate_count)
        results_frame = results_frame.drop_duplicates(subset=duplicate_columns, keep="first")
    validation = validate_results_dataframe(results_frame)
    if not validation.is_valid:
        raise ValueError("Historical results validation failed: " + "; ".join(validation.issues))
    return results_frame


def load_fixtures_frame(team_mappings: dict[str, str], known_teams: set[str] | None = None) -> pd.DataFrame:
    fixtures_path = MANUAL_DATA_DIR / "worldcup_2026_fixtures.csv"
    if not fixtures_path.exists():
        raise FileNotFoundError(
            f"Missing required World Cup 2026 fixture file: {fixtures_path}. "
            "Create it with the required columns before running predictions."
        )
    fixtures_frame = pd.read_csv(fixtures_path)
    validation = validate_fixtures_dataframe(fixtures_frame, team_mappings, known_teams=known_teams)
    if not validation.is_valid:
        raise ValueError("Fixture validation failed: " + "; ".join(validation.issues))
    fixtures_frame = fixtures_frame.copy()
    for column in ["home_team", "away_team"]:
        fixtures_frame[column] = fixtures_frame[column].fillna("").astype(str).map(
            lambda value: normalize_team_name(value, team_mappings)
        )
    fixtures_frame["neutral"] = fixtures_frame["neutral"].map(parse_bool)
    return fixtures_frame


def build_training_features(cutoff_date: datetime | pd.Timestamp) -> tuple[pd.DataFrame, FeatureState]:
    ensure_project_dirs()
    results_frame = load_results_frame(cutoff_date=cutoff_date)
    results_frame = prepare_team_columns(results_frame, load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv"))
    results_frame["date"] = pd.to_datetime(results_frame["date"])
    cutoff_validation = validate_results_dataframe(results_frame, cutoff_date)
    if not cutoff_validation.is_valid:
        raise ValueError("Cutoff-filtered training data validation failed: " + "; ".join(cutoff_validation.issues))
    if results_frame.empty:
        raise ValueError(f"No historical results exist on or before cutoff date {pd.Timestamp(cutoff_date).date()}.")

    feature_frame, feature_state = build_feature_rows(results_frame, FeatureState(), update_state=True)
    feature_frame = feature_frame.dropna(subset=["result"]).reset_index(drop=True)

    processed_path = PROCESSED_DATA_DIR / "match_features.csv"
    feature_frame.to_csv(processed_path, index=False)

    state_path = PROCESSED_DATA_DIR / "feature_state.json"
    state_path.write_text(json.dumps(feature_state.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Saved training features to %s", processed_path)
    LOGGER.info("Saved feature state to %s", state_path)
    return feature_frame, feature_state


def build_fixture_features(cutoff_date: datetime | pd.Timestamp, feature_state: FeatureState) -> pd.DataFrame:
    team_mappings = load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv")
    known_teams = set(feature_state.team_history).union(feature_state.elo_state.ratings)
    fixtures_frame = load_fixtures_frame(team_mappings, known_teams=known_teams)
    fixtures_frame = fixtures_frame.copy()
    fixtures_frame["match_date"] = pd.to_datetime(fixtures_frame["match_date"])
    fixtures_frame["home_team"] = fixtures_frame["home_team"].fillna("").astype(str).map(lambda value: normalize_team_name(value, team_mappings))
    fixtures_frame["away_team"] = fixtures_frame["away_team"].fillna("").astype(str).map(lambda value: normalize_team_name(value, team_mappings))
    fixtures_frame["date"] = fixtures_frame["match_date"]
    fixtures_frame["tournament"] = "World Cup"
    fixed_team_mask = fixtures_frame["home_team"].str.strip().ne("") & fixtures_frame["away_team"].str.strip().ne("")
    fixtures_frame = fixtures_frame.loc[fixed_team_mask].reset_index(drop=True)
    if fixtures_frame.empty:
        raise ValueError("The fixture file contains no matches with both teams resolved.")

    future_features, _ = build_feature_rows(fixtures_frame, feature_state, update_state=False)
    future_features["match_id"] = fixtures_frame["match_id"].values
    future_features["stage"] = fixtures_frame["stage"].values
    future_features["group"] = fixtures_frame["group"].values
    future_features["match_date"] = fixtures_frame["match_date"].values
    future_features["venue"] = fixtures_frame["venue"].values
    future_features["city"] = fixtures_frame["city"].values
    future_features["country"] = fixtures_frame["country"].values
    future_features["bracket_slot_home"] = fixtures_frame["bracket_slot_home"].values
    future_features["bracket_slot_away"] = fixtures_frame["bracket_slot_away"].values
    future_features["neutral"] = fixtures_frame["neutral"].astype(int).values

    processed_path = PROCESSED_DATA_DIR / "fixture_features.csv"
    future_features.to_csv(processed_path, index=False)
    LOGGER.info("Saved fixture features to %s", processed_path)
    return future_features


def build_datasets(cutoff_date: datetime | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, FeatureState]:
    training_features, feature_state = build_training_features(cutoff_date)
    fixture_features = build_fixture_features(cutoff_date, feature_state)
    return training_features, fixture_features, feature_state
