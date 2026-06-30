from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_classification_metrics, compute_scoreline_metrics
from src.evaluation.training_strategy import StrategyConfig, _fit_pipeline, _predict_pipeline
from src.features.dynamic_rating import build_rating_features
from src.features.target_engineering import add_match_targets, elo_baseline_expectations
from src.features.team_strength_features import build_team_strength_snapshots, latest_team_snapshots
from src.features.tournament_readiness_features import (
    build_historical_readiness_dataset,
    readiness_snapshot_for_teams,
)
from src.models.indirect_adjustments import (
    apply_adjustments_to_matches,
    build_team_adjustments,
    configured_variants,
)
from src.models.train_goal_model import GOAL_FEATURE_GROUPS
from src.models.train_team_strength_trend_model import train_team_strength_trend_model
from src.models.train_tournament_readiness_model import train_tournament_readiness_model
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)
WORLD_CUP_YEARS = [2010, 2014, 2018, 2022]


def _load_selected_frame() -> tuple[pd.DataFrame, dict[str, Any], list[str], StrategyConfig]:
    frame = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv", parse_dates=["date"])
    selection = json.loads((MODELS_DIR / "model_selection.json").read_text(encoding="utf-8"))
    strategy = selection.get("training_strategy_recommendation", {})
    rating_model = str(strategy.get("rating_model", "standard_elo"))
    k_scale = float(strategy.get("elo_k_scale", 1.0))
    frame, _ = build_rating_features(frame, rating_model, k_scale)
    frame = add_match_targets(frame)
    feature_group = str(selection.get("goal_feature_group", "attack_defence_poisson"))
    config = StrategyConfig(
        minimum_year=int(strategy.get("minimum_year", selection.get("minimum_training_year", 1990))),
        half_life_years=float(strategy.get("training_half_life_years", 1e9)),
        importance_profile=str(strategy.get("importance_profile", "aggressive")),
        goal_cap=int(strategy.get("goal_cap", selection.get("goal_cap", 8))),
        rating_model=rating_model,
        elo_k_scale=k_scale,
    )
    return frame, selection, GOAL_FEATURE_GROUPS[feature_group], config


def _metrics(evaluation: pd.DataFrame, probabilities: np.ndarray, home: np.ndarray, away: np.ndarray) -> dict[str, float]:
    classification = compute_classification_metrics(
        evaluation["result"].to_numpy(dtype=int), probabilities, np.argmax(probabilities, axis=1)
    )
    score = compute_scoreline_metrics(evaluation, home, away, max_goals=10)
    return {
        **classification,
        "home_goals_mae": score["home_goal_mae"],
        "away_goals_mae": score["away_goal_mae"],
        "goal_difference_mae": score["goal_difference_mae"],
        "scoreline_top_1_accuracy": score["top_1_scoreline_accuracy"],
        "scoreline_top_5_hit_rate": score["top_5_scoreline_hit_rate"],
    }


def _historical_host_teams(evaluation: pd.DataFrame) -> set[str]:
    hosts: set[str] = set()
    for side in ["home", "away"]:
        flag = f"playing_in_home_country_{side}"
        if flag in evaluation:
            hosts.update(evaluation.loc[evaluation[flag].eq(1), f"{side}_team"].astype(str))
    return hosts


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["model_variant", "trend_weight", "readiness_weight"], as_index=False)
        .agg(
            world_cups_covered=("worldcup_year", "nunique"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier_score=("brier_score", "mean"),
            avg_calibration_error=("calibration_error", "mean"),
            avg_scoreline_top_5_hit_rate=("scoreline_top_5_hit_rate", "mean"),
            stability_score=("log_loss", "std"),
        )
        .sort_values(["avg_log_loss", "avg_brier_score"])
    )


def _select_variant(summary: pd.DataFrame) -> tuple[str, str, str]:
    baseline = summary[summary["model_variant"].eq("baseline")].iloc[0]
    challengers = summary[
        ~summary["model_variant"].isin(["baseline", "indirect_only"])
        & summary["world_cups_covered"].eq(len(WORLD_CUP_YEARS))
    ].sort_values(["avg_log_loss", "avg_brier_score", "stability_score"])
    best = challengers.iloc[0]
    minimum_gain = float(config_value("indirect_models", "minimum_log_loss_improvement", default=0.0005))
    proven = (
        float(best["avg_log_loss"]) <= float(baseline["avg_log_loss"]) - minimum_gain
        and float(best["avg_brier_score"]) < float(baseline["avg_brier_score"])
        and float(best["stability_score"]) <= float(baseline["stability_score"]) * 1.20
    )
    selected = str(best["model_variant"]) if proven else "baseline"
    reason = (
        f"{best['model_variant']} cleared the configured log-loss gain, Brier, and stability checks."
        if proven
        else f"Best indirect challenger {best['model_variant']} did not clear all configured proof checks; baseline retained."
    )
    return selected, str(best["model_variant"]), reason


def run_indirect_model_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    frame, selection, features, config = _load_selected_frame()
    snapshots = build_team_strength_snapshots(frame)
    readiness_dataset = build_historical_readiness_dataset(frame, snapshots)
    snapshots.to_csv(PROCESSED_DATA_DIR / "team_strength_snapshots.csv", index=False)
    readiness_dataset.to_csv(PROCESSED_DATA_DIR / "tournament_readiness_dataset.csv", index=False)
    rows: list[dict[str, Any]] = []
    for year in WORLD_CUP_YEARS:
        evaluation = frame[
            frame["date"].dt.year.eq(year)
            & frame["tournament"].astype(str).str.strip().eq("FIFA World Cup")
        ].copy()
        start = evaluation["date"].min()
        teams = set(evaluation["home_team"]).union(evaluation["away_team"])
        training = frame[
            (frame["date"] < start) & frame["date"].dt.year.ge(config.minimum_year)
        ].copy()
        baseline_model = _fit_pipeline(training, config, features)
        eligible_snapshots = snapshots[snapshots["date"].dt.year.ge(config.minimum_year)].copy()
        trend_bundle = train_team_strength_trend_model(eligible_snapshots, start, persist=False)
        historical_readiness = readiness_dataset[
            readiness_dataset["tournament_year"].ge(config.minimum_year)
        ].copy()
        readiness_bundle = train_tournament_readiness_model(historical_readiness, start, persist=False)
        latest = latest_team_snapshots(eligible_snapshots, teams, start)
        readiness_rows = readiness_snapshot_for_teams(
            eligible_snapshots, teams, start, year, _historical_host_teams(evaluation)
        )
        for variant in configured_variants():
            _, _, adjustments = build_team_adjustments(
                teams, latest, trend_bundle, readiness_rows, readiness_bundle, variant
            )
            adjusted = apply_adjustments_to_matches(evaluation, adjustments)
            if variant.indirect_only:
                expectation = elo_baseline_expectations(adjusted)
                probabilities = expectation[["elo_p_home_win", "elo_p_draw", "elo_p_away_win"]].to_numpy()
                home = expectation["elo_expected_goals_home"].to_numpy()
                away = expectation["elo_expected_goals_away"].to_numpy()
            else:
                prediction = _predict_pipeline(baseline_model, adjusted)
                probabilities = prediction["margin_probabilities"]
                home = prediction["home"]
                away = prediction["away"]
            metrics = _metrics(evaluation, probabilities, home, away)
            rows.append(
                {
                    "model_variant": variant.name,
                    "worldcup_year": year,
                    "trend_target": "future_elo_delta_next_5_matches",
                    "readiness_target": "tournament_performance_above_elo_expectation",
                    "trend_weight": variant.trend_weight,
                    "readiness_weight": variant.readiness_weight,
                    "max_total_strength_adjustment": variant.max_total_adjustment,
                    **metrics,
                    "group_points_mae": np.nan,
                    "group_qualification_accuracy": np.nan,
                    "notes": "Historical source lacks reliable group-stage labels; group metrics unavailable.",
                }
            )
        LOGGER.info("Completed indirect-model backtest for World Cup %s.", year)
    results = pd.DataFrame(rows)
    summary = _summarize(results)
    selected, comparison, reason = _select_variant(summary)
    selection_payload = {
        "selected_variant": selected,
        "comparison_variant": comparison,
        "selection_reason": reason,
        "trend_target": "future_elo_delta_next_5_matches",
        "readiness_target": "tournament_performance_above_elo_expectation",
        "minimum_log_loss_improvement": float(
            config_value("indirect_models", "minimum_log_loss_improvement", default=0.0005)
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (MODELS_DIR / "indirect_model_selection.json").write_text(
        json.dumps(selection_payload, indent=2), encoding="utf-8"
    )
    results.to_csv(PREDICTIONS_DIR / "indirect_model_backtest_results.csv", index=False)
    summary.to_csv(PREDICTIONS_DIR / "indirect_model_backtest_summary.csv", index=False)
    return results, summary
