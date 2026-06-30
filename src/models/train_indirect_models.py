from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.build_dataset import load_fixtures_frame
from src.data.validate_data import load_team_mappings
from src.features.dynamic_rating import build_rating_features
from src.features.team_strength_features import build_team_strength_snapshots, latest_team_snapshots
from src.features.tournament_readiness_features import (
    build_historical_readiness_dataset,
    readiness_snapshot_for_teams,
)
from src.models.indirect_adjustments import build_team_adjustments, variant_by_name
from src.models.train_team_strength_trend_model import train_team_strength_trend_model
from src.models.train_tournament_readiness_model import train_tournament_readiness_model
from src.utils.paths import MANUAL_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR


def _fixture_teams_and_start() -> tuple[set[str], pd.Timestamp, set[str]]:
    fixtures = load_fixtures_frame(load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv"))
    group = fixtures[fixtures["stage"].astype(str).str.contains("group", case=False, na=False)].copy()
    teams = set(group["home_team"]).union(group["away_team"])
    start = pd.to_datetime(group["match_date"]).min()
    hosts = {team for team in teams if team in set(group["country"].dropna().astype(str))}
    return teams, start, hosts


def train_indirect_models(cutoff_date: str | pd.Timestamp) -> dict[str, object]:
    selection = json.loads((MODELS_DIR / "model_selection.json").read_text(encoding="utf-8"))
    strategy = selection.get("training_strategy_recommendation", {})
    indirect_selection = json.loads((MODELS_DIR / "indirect_model_selection.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv", parse_dates=["date"])
    frame, _ = build_rating_features(
        frame,
        str(strategy.get("rating_model", "standard_elo")),
        float(strategy.get("elo_k_scale", 1.0)),
    )
    minimum_year = int(strategy.get("minimum_year", selection.get("minimum_training_year", 1990)))
    snapshots = build_team_strength_snapshots(frame)
    eligible = snapshots[snapshots["date"].dt.year.ge(minimum_year)].copy()
    readiness_dataset = build_historical_readiness_dataset(frame, snapshots)
    readiness_dataset = readiness_dataset[readiness_dataset["tournament_year"].ge(minimum_year)].copy()
    snapshots.to_csv(PROCESSED_DATA_DIR / "team_strength_snapshots.csv", index=False)
    readiness_dataset.to_csv(PROCESSED_DATA_DIR / "tournament_readiness_dataset.csv", index=False)
    trend_bundle = train_team_strength_trend_model(eligible, cutoff_date, persist=True)
    readiness_bundle = train_tournament_readiness_model(readiness_dataset, cutoff_date, persist=True)
    teams, tournament_start, hosts = _fixture_teams_and_start()
    latest = latest_team_snapshots(eligible, teams, pd.Timestamp(cutoff_date) + pd.Timedelta(days=1))
    readiness_rows = readiness_snapshot_for_teams(eligible, teams, tournament_start, 2026, hosts)

    selected_variant = variant_by_name(str(indirect_selection["selected_variant"]))
    comparison_variant = variant_by_name(str(indirect_selection["comparison_variant"]))
    _, _, selected_adjustments = build_team_adjustments(
        teams, latest, trend_bundle, readiness_rows, readiness_bundle, selected_variant
    )
    trend_scores, readiness_scores, comparison_adjustments = build_team_adjustments(
        teams, latest, trend_bundle, readiness_rows, readiness_bundle, comparison_variant
    )
    trend_scores = trend_scores.merge(
        comparison_adjustments[["team", "trend_adjustment"]], on="team", how="left", suffixes=("", "_comparison")
    )
    if "trend_adjustment_comparison" in trend_scores:
        trend_scores["trend_adjustment"] = trend_scores["trend_adjustment_comparison"]
    readiness_scores = readiness_scores.merge(
        comparison_adjustments[["team", "readiness_adjustment"]],
        on="team",
        how="left",
        suffixes=("", "_comparison"),
    )
    if "readiness_adjustment_comparison" in readiness_scores:
        readiness_scores["readiness_adjustment"] = readiness_scores["readiness_adjustment_comparison"]
    trend_columns = [
        "team", "base_elo", "trend_score", "expected_future_elo_delta",
        "expected_future_performance_above_expectation", "trend_adjustment", "trend_data_quality_flag",
    ]
    readiness_columns = [
        "team", "base_elo", "tournament_readiness_score", "expected_group_points_adjustment",
        "expected_goal_difference_adjustment", "overperformance_probability",
        "underperformance_probability", "readiness_adjustment", "readiness_data_quality_flag",
    ]
    adjustment_columns = [
        "team", "base_elo", "trend_adjustment", "readiness_adjustment", "total_indirect_adjustment",
        "adjusted_team_strength", "adjustment_cap_applied", "data_quality_flag",
    ]
    trend_scores.reindex(columns=trend_columns).to_csv(
        PREDICTIONS_DIR / "team_strength_trend_scores_2026.csv", index=False
    )
    readiness_scores.reindex(columns=readiness_columns).to_csv(
        PREDICTIONS_DIR / "tournament_readiness_scores_2026.csv", index=False
    )
    selected_adjustments.reindex(columns=adjustment_columns).to_csv(
        PREDICTIONS_DIR / "team_strength_adjustments_2026.csv", index=False
    )
    payload = {
        "selected_variant": selected_variant.name,
        "comparison_variant": comparison_variant.name,
        "selected_adjustments": selected_adjustments.set_index("team")[
            ["trend_adjustment", "readiness_adjustment", "total_indirect_adjustment"]
        ].to_dict(orient="index"),
        "comparison_adjustments": comparison_adjustments.set_index("team")[
            ["trend_adjustment", "readiness_adjustment", "total_indirect_adjustment"]
        ].to_dict(orient="index"),
    }
    (MODELS_DIR / "indirect_adjustments.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
