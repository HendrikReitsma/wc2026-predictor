from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import expected_goals_from_elo
from src.utils.config import config_value


MARGIN_CLASSES = [
    "away_win_3_plus",
    "away_win_2",
    "away_win_1",
    "draw",
    "home_win_1",
    "home_win_2",
    "home_win_3_plus",
]
MARGIN_CLASS_TO_INDEX = {label: index for index, label in enumerate(MARGIN_CLASSES)}
MARGIN_CLASS_CENTERS = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


def margin_class_from_goal_difference(goal_difference: int | float) -> str:
    difference = int(goal_difference)
    if difference <= -3:
        return "away_win_3_plus"
    if difference == -2:
        return "away_win_2"
    if difference == -1:
        return "away_win_1"
    if difference == 0:
        return "draw"
    if difference == 1:
        return "home_win_1"
    if difference == 2:
        return "home_win_2"
    return "home_win_3_plus"


def elo_baseline_expectations(frame: pd.DataFrame) -> pd.DataFrame:
    neutral = pd.to_numeric(frame["neutral"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    home_advantage = float(config_value("elo", "home_advantage", default=65.0)) * (1.0 - neutral)
    home_elo = pd.to_numeric(frame["home_elo_pre"], errors="coerce").fillna(1500.0).to_numpy(dtype=float)
    away_elo = pd.to_numeric(frame["away_elo_pre"], errors="coerce").fillna(1500.0).to_numpy(dtype=float)
    adjusted_home = home_elo + home_advantage
    decisive_home = 1.0 / (1.0 + np.power(10.0, -(adjusted_home - away_elo) / 400.0))
    draw = np.clip(0.28 * np.exp(-np.abs(adjusted_home - away_elo) / 600.0), 0.12, 0.30)
    p_home = (1.0 - draw) * decisive_home
    p_away = (1.0 - draw) * (1.0 - decisive_home)
    home_xg = np.asarray([expected_goals_from_elo(home, away) for home, away in zip(adjusted_home, away_elo)])
    away_xg = np.asarray([expected_goals_from_elo(away, home) for home, away in zip(adjusted_home, away_elo)])
    return pd.DataFrame(
        {
            "elo_p_home_win": p_home,
            "elo_p_draw": draw,
            "elo_p_away_win": p_away,
            "elo_expected_points": 3.0 * p_home + draw,
            "elo_expected_goals_home": home_xg,
            "elo_expected_goals_away": away_xg,
            "elo_expected_goal_difference": home_xg - away_xg,
        },
        index=frame.index,
    )


def add_match_targets(
    frame: pd.DataFrame,
    goal_cap: int | None = None,
    goal_difference_cap: int | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    resolved_goal_cap = int(goal_cap if goal_cap is not None else config_value("target_experiments", "goal_cap", default=6))
    resolved_difference_cap = int(
        goal_difference_cap
        if goal_difference_cap is not None
        else config_value("target_experiments", "goal_difference_cap", default=4)
    )
    home = pd.to_numeric(output["home_score"], errors="coerce")
    away = pd.to_numeric(output["away_score"], errors="coerce")
    output["home_goals"] = home
    output["away_goals"] = away
    output["capped_home_goals"] = home.clip(lower=0, upper=resolved_goal_cap)
    output["capped_away_goals"] = away.clip(lower=0, upper=resolved_goal_cap)
    output["goal_difference"] = home - away
    output["capped_goal_difference"] = output["goal_difference"].clip(
        lower=-resolved_difference_cap, upper=resolved_difference_cap
    )
    output["home_win"] = (home > away).astype(int)
    output["draw"] = (home == away).astype(int)
    output["away_win"] = (home < away).astype(int)
    output["outcome_target"] = np.select(
        [output["home_win"].eq(1), output["draw"].eq(1)],
        ["home_win", "draw"],
        default="away_win",
    )
    output["margin_class"] = output["goal_difference"].map(margin_class_from_goal_difference)
    output["margin_class_index"] = output["margin_class"].map(MARGIN_CLASS_TO_INDEX).astype(int)
    output["actual_points"] = np.select(
        [output["home_win"].eq(1), output["draw"].eq(1)],
        [3.0, 1.0],
        default=0.0,
    )
    expectations = elo_baseline_expectations(output)
    for column in expectations:
        output[column] = expectations[column]
    output["goal_diff_residual"] = output["goal_difference"] - output["elo_expected_goal_difference"]
    output["points_residual"] = output["actual_points"] - output["elo_expected_points"]
    return output


def build_future_team_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Build isolated research labels. These rows must never be joined as same-match features."""
    targeted = add_match_targets(frame).copy()
    home = pd.DataFrame(
        {
            "match_index": targeted.index,
            "date": targeted["date"],
            "team": targeted["home_team"],
            "goal_diff_above_expectation": targeted["goal_diff_residual"],
            "points_above_expectation": targeted["points_residual"],
            "elo_pre": targeted["home_elo_pre"],
        }
    )
    away = pd.DataFrame(
        {
            "match_index": targeted.index,
            "date": targeted["date"],
            "team": targeted["away_team"],
            "goal_diff_above_expectation": -targeted["goal_difference"] + targeted["elo_expected_goal_difference"],
            "points_above_expectation": (
                np.select(
                    [targeted["away_win"].eq(1), targeted["draw"].eq(1)],
                    [3.0, 1.0],
                    default=0.0,
                )
                - (3.0 * targeted["elo_p_away_win"] + targeted["elo_p_draw"])
            ),
            "elo_pre": targeted["away_elo_pre"],
        }
    )
    team_rows = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "match_index"])
    output_groups: list[pd.DataFrame] = []
    for _, group in team_rows.groupby("team", sort=False):
        group = group.copy().reset_index(drop=True)
        for window in [5, 10]:
            group[f"future_elo_delta_next_{window}"] = group["elo_pre"].shift(-window) - group["elo_pre"]
        group["future_goal_diff_above_expectation_next_5"] = [
            group["goal_diff_above_expectation"].iloc[index + 1 : index + 6].sum()
            if index + 5 < len(group)
            else np.nan
            for index in range(len(group))
        ]
        group["future_points_above_expectation_next_5"] = [
            group["points_above_expectation"].iloc[index + 1 : index + 6].sum()
            if index + 5 < len(group)
            else np.nan
            for index in range(len(group))
        ]
        output_groups.append(group)
    return pd.concat(output_groups, ignore_index=True) if output_groups else pd.DataFrame()
