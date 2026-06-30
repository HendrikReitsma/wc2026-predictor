from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.target_engineering import (
    MARGIN_CLASSES,
    add_match_targets,
    build_future_team_targets,
    margin_class_from_goal_difference,
)


def _matches() -> pd.DataFrame:
    rows = []
    for index, difference in enumerate([-5, -2, -1, 0, 1, 2, 5, 1, -1, 0, 2, -2]):
        home_score = max(difference, 0) + 1
        away_score = max(-difference, 0) + 1
        rows.append(
            {
                "date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=index * 30),
                "home_team": "A" if index % 2 == 0 else "B",
                "away_team": "B" if index % 2 == 0 else "A",
                "home_score": home_score,
                "away_score": away_score,
                "home_elo_pre": 1520 + index,
                "away_elo_pre": 1480 - index,
                "neutral": 1,
            }
        )
    return pd.DataFrame(rows)


def test_match_targets_have_requested_caps_and_residuals() -> None:
    targets = add_match_targets(_matches(), goal_cap=6, goal_difference_cap=4)
    assert targets["capped_home_goals"].max() <= 6
    assert targets["capped_away_goals"].max() <= 6
    assert targets["capped_goal_difference"].between(-4, 4).all()
    assert set(targets["margin_class"]).issubset(MARGIN_CLASSES)
    assert np.allclose(
        targets["goal_diff_residual"],
        targets["goal_difference"] - targets["elo_expected_goal_difference"],
    )
    assert np.allclose(targets[["elo_p_home_win", "elo_p_draw", "elo_p_away_win"]].sum(axis=1), 1.0)


def test_margin_class_boundaries() -> None:
    assert [margin_class_from_goal_difference(value) for value in [-4, -2, -1, 0, 1, 2, 4]] == MARGIN_CLASSES


def test_future_targets_are_separate_and_use_only_later_matches() -> None:
    future = build_future_team_targets(_matches())
    first_a = future[future["team"] == "A"].sort_values("date").iloc[0]
    assert pd.notna(first_a["future_points_above_expectation_next_5"])
    assert "future_points_above_expectation_next_5" not in add_match_targets(_matches()).columns
