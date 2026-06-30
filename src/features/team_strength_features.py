from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.target_engineering import add_match_targets


TREND_FEATURE_COLUMNS = [
    "base_elo",
    "elo_change_last_3_matches",
    "elo_change_last_5_matches",
    "elo_change_last_10_matches",
    "elo_change_last_6_months",
    "elo_change_last_12_months",
    "recent_points_per_match_5",
    "recent_points_per_match_10",
    "recent_points_above_expectation_5",
    "recent_points_above_expectation_10",
    "recent_goal_difference_5",
    "recent_goal_difference_10",
    "recent_goal_difference_above_expectation_5",
    "recent_goal_difference_above_expectation_10",
    "recent_goals_for_5",
    "recent_goals_against_5",
    "recent_opponent_elo_5",
    "recent_opponent_elo_10",
    "days_since_last_match",
    "matches_last_12_months",
]


def _side_rows(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    other = "away" if side == "home" else "home"
    home_side = side == "home"
    actual_points = np.select(
        [
            frame["home_score"].gt(frame["away_score"]) if home_side else frame["away_score"].gt(frame["home_score"]),
            frame["home_score"].eq(frame["away_score"]),
        ],
        [3.0, 1.0],
        default=0.0,
    )
    expected_points = (
        3.0 * frame["elo_p_home_win"] + frame["elo_p_draw"]
        if home_side
        else 3.0 * frame["elo_p_away_win"] + frame["elo_p_draw"]
    )
    goal_difference = frame["goal_difference"] if home_side else -frame["goal_difference"]
    expected_goal_difference = (
        frame["elo_expected_goal_difference"] if home_side else -frame["elo_expected_goal_difference"]
    )
    return pd.DataFrame(
        {
            "match_index": frame.index,
            "date": frame["date"],
            "team": frame[f"{side}_team"],
            "opponent": frame[f"{other}_team"],
            "tournament": frame["tournament"],
            "is_friendly": frame["is_friendly"],
            "is_world_cup": frame["is_world_cup"],
            "is_continental_competition": frame["is_continental_competition"],
            "base_elo": frame[f"{side}_elo_pre"],
            "opponent_elo": frame[f"{other}_elo_pre"],
            "actual_points": actual_points,
            "points_above_expectation": actual_points - expected_points,
            "goal_difference": goal_difference,
            "goal_difference_above_expectation": goal_difference - expected_goal_difference,
            "goals_for": frame[f"{side}_score"],
            "goals_against": frame[f"{other}_score"],
            "days_since_last_match": frame[f"days_since_last_match_{side}"],
            "playing_in_home_country": frame.get(
                f"playing_in_home_country_{side}", pd.Series(0, index=frame.index)
            ),
        }
    )


def _lookback_change(dates: np.ndarray, values: np.ndarray, months: int) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    timestamps = pd.DatetimeIndex(dates)
    for index, date in enumerate(timestamps):
        target = date - pd.DateOffset(months=months)
        prior_index = int(timestamps.searchsorted(target, side="right") - 1)
        output[index] = float(values[index] - values[prior_index]) if prior_index >= 0 else 0.0
    return output


def _matches_last_year(dates: np.ndarray) -> np.ndarray:
    timestamps = pd.DatetimeIndex(dates)
    return np.asarray(
        [
            index - int(timestamps.searchsorted(date - pd.DateOffset(years=1), side="left"))
            for index, date in enumerate(timestamps)
        ],
        dtype=float,
    )


def _future_sum(values: pd.Series, window: int) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    output = np.full(len(raw), np.nan, dtype=float)
    for index in range(len(raw) - window):
        output[index] = float(np.sum(raw[index + 1 : index + window + 1]))
    return output


def build_team_strength_snapshots(match_features: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe team-before-match rows and isolated future labels."""
    targeted = add_match_targets(match_features).copy()
    targeted["date"] = pd.to_datetime(targeted["date"])
    snapshots = pd.concat([_side_rows(targeted, "home"), _side_rows(targeted, "away")], ignore_index=True)
    snapshots = snapshots.sort_values(["team", "date", "match_index"], kind="stable").reset_index(drop=True)
    output: list[pd.DataFrame] = []
    for _, group in snapshots.groupby("team", sort=False):
        group = group.copy().reset_index(drop=True)
        for window in [3, 5, 10]:
            group[f"elo_change_last_{window}_matches"] = (
                group["base_elo"] - group["base_elo"].shift(window)
            ).fillna(0.0)
        group["elo_change_last_6_months"] = _lookback_change(
            group["date"].to_numpy(), group["base_elo"].to_numpy(dtype=float), 6
        )
        group["elo_change_last_12_months"] = _lookback_change(
            group["date"].to_numpy(), group["base_elo"].to_numpy(dtype=float), 12
        )
        group["matches_last_12_months"] = _matches_last_year(group["date"].to_numpy())
        rolling_sources = {
            "recent_points_per_match": "actual_points",
            "recent_points_above_expectation": "points_above_expectation",
            "recent_goal_difference": "goal_difference",
            "recent_goal_difference_above_expectation": "goal_difference_above_expectation",
            "recent_goals_for": "goals_for",
            "recent_goals_against": "goals_against",
            "recent_opponent_elo": "opponent_elo",
        }
        for output_name, source in rolling_sources.items():
            for window in [5, 10]:
                group[f"{output_name}_{window}"] = (
                    group[source].shift(1).rolling(window, min_periods=1).mean().fillna(0.0)
                )
        for window in [5, 10]:
            group[f"future_elo_delta_next_{window}_matches"] = group["base_elo"].shift(-window) - group["base_elo"]
            group[f"future_target_end_date_next_{window}"] = group["date"].shift(-window)
        group["future_points_above_expectation_next_5_matches"] = _future_sum(
            group["points_above_expectation"], 5
        )
        group["future_goal_difference_above_expectation_next_5_matches"] = _future_sum(
            group["goal_difference_above_expectation"], 5
        )
        group["prior_matches"] = np.arange(len(group), dtype=int)
        output.append(group)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def latest_team_snapshots(
    snapshots: pd.DataFrame,
    teams: list[str] | set[str],
    cutoff_date: str | pd.Timestamp,
) -> pd.DataFrame:
    eligible = snapshots[
        snapshots["team"].isin(set(teams)) & (pd.to_datetime(snapshots["date"]) < pd.Timestamp(cutoff_date))
    ].copy()
    if eligible.empty:
        return eligible
    return eligible.sort_values(["team", "date"]).groupby("team", as_index=False).tail(1).reset_index(drop=True)
