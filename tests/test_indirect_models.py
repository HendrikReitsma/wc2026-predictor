from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import build_feature_rows
from src.features.team_strength_features import build_team_strength_snapshots
from src.models.indirect_adjustments import IndirectVariant, build_team_adjustments


def _features() -> pd.DataFrame:
    rows = []
    teams = ["A", "B", "C", "D"]
    for index in range(28):
        rows.append(
            {
                "date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=index * 40),
                "home_team": teams[index % 4],
                "away_team": teams[(index + 1) % 4],
                "home_score": index % 3,
                "away_score": (index + 1) % 2,
                "tournament": "World Cup qualification",
                "country": teams[index % 4],
                "neutral": False,
            }
        )
    return build_feature_rows(pd.DataFrame(rows))[0]


def test_future_trend_targets_end_after_snapshot() -> None:
    snapshots = build_team_strength_snapshots(_features())
    completed = snapshots.dropna(subset=["future_target_end_date_next_5"])

    assert (pd.to_datetime(completed["future_target_end_date_next_5"]) > completed["date"]).all()


def test_indirect_adjustment_cap_is_enforced() -> None:
    class Trend:
        def predict(self, rows):
            return pd.DataFrame(
                {
                    "team": rows["team"],
                    "date": rows["date"],
                    "base_elo": rows["base_elo"],
                    "prior_matches": rows["prior_matches"],
                    "expected_future_elo_delta": 200.0,
                    "expected_future_performance_above_expectation": 0.0,
                    "trend_score": 4.0,
                    "trend_model_confidence": 1.0,
                    "trend_data_quality_flag": "ok",
                }
            )

    class Readiness:
        def predict(self, rows):
            return pd.DataFrame(
                {
                    "team": rows["team"],
                    "base_elo": rows["base_elo"],
                    "tournament_year": 2026,
                    "tournament_readiness_score": 2.0,
                    "expected_group_points_adjustment": 0.0,
                    "expected_goal_difference_adjustment": 0.0,
                    "overperformance_probability": 0.8,
                    "underperformance_probability": 0.2,
                    "readiness_data_quality_flag": "ok",
                }
            )

    latest = pd.DataFrame({"team": ["A"], "date": [pd.Timestamp("2026-01-01")], "base_elo": [1600.0], "prior_matches": [20]})
    readiness = pd.DataFrame({"team": ["A"], "base_elo": [1600.0], "tournament_year": [2026]})
    _, _, adjustments = build_team_adjustments(
        {"A"}, latest, Trend(), readiness, Readiness(), IndirectVariant("large", 1.0, 1.0, 75.0)
    )

    assert np.isclose(adjustments.loc[0, "total_indirect_adjustment"], 75.0)
    assert bool(adjustments.loc[0, "adjustment_cap_applied"])
