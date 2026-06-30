from __future__ import annotations

import pandas as pd

from src.features.elo import EloRatingState
from src.features.elo import classify_tournament
from src.features.feature_engineering import build_feature_rows
from src.features.dynamic_rating import build_rating_features


def test_features_are_pre_match_and_same_day_safe() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "country": "A",
                "neutral": "False",
            },
            {
                "date": "2024-01-01",
                "home_team": "C",
                "away_team": "A",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "country": "C",
                "neutral": "False",
            },
            {
                "date": "2024-02-01",
                "home_team": "A",
                "away_team": "C",
                "home_score": 2,
                "away_score": 1,
                "tournament": "World Cup qualification",
                "country": "A",
                "neutral": False,
            },
        ]
    )

    features, state = build_feature_rows(matches)

    assert features.loc[0, "home_elo_pre"] == 1500.0
    assert features.loc[1, "away_elo_pre"] == 1500.0
    assert features.loc[0, "recent_form_points_home_5"] == 0.0
    assert features.loc[1, "recent_form_points_away_5"] == 0.0
    assert features.loc[2, "recent_form_points_home_5"] == 4.0
    assert features.loc[2, "opponent_adjusted_form_home_5"] != 0.0
    assert features.loc[2, "points_above_expectation_home_5"] != 0.0
    assert features.loc[2, "recent_elo_change_home_5"] != 0.0
    assert features.loc[2, "rolling_goals_for_home_5"] > 0.0
    assert features.loc[2, "home_attack_rating_pre"] != 0.0
    assert all(len(snapshot["ratings"]) <= 2 for snapshot in state.elo_state.snapshots)


def test_elo_before_date_is_strictly_before_match_date() -> None:
    state = EloRatingState()
    state.update_match("A", "B", 2, 0, "Friendly", pd.Timestamp("2024-01-01"), neutral=True)

    assert state.get_team_elo_before_date("A", pd.Timestamp("2024-01-01")) == 1500.0
    assert state.get_team_elo_before_date("A", pd.Timestamp("2024-01-02")) > 1500.0
    assert state.snapshots[0]["ratings"].keys() == {"A", "B"}


def test_world_cup_qualifier_is_not_world_cup_finals() -> None:
    flags = classify_tournament("FIFA World Cup qualification")
    assert flags["is_world_cup"] == 0
    assert flags["match_type"] == "qualifier"


def test_dynamic_rating_is_chronological_and_same_day_safe() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 4,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": True,
            },
            {
                "date": "2020-01-01",
                "home_team": "C",
                "away_team": "A",
                "home_score": 0,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": True,
            },
            {
                "date": "2028-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "neutral": True,
            },
        ]
    )
    standard, _ = build_rating_features(matches, "standard_elo", 1.0)
    smoothed, state = build_rating_features(matches, "smoothed_dynamic", 1.0)

    assert standard.loc[0, "home_elo_pre"] == standard.loc[1, "away_elo_pre"] == 1500.0
    assert smoothed.loc[0, "home_elo_pre"] == smoothed.loc[1, "away_elo_pre"] == 1500.0
    assert abs(smoothed.loc[2, "home_elo_pre"] - 1500.0) < abs(standard.loc[2, "home_elo_pre"] - 1500.0)
    assert state.model_name == "smoothed_dynamic"
