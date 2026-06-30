from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.weighting import combined_sample_weights, match_importance_weights, time_decay_weights


def test_time_decay_half_life() -> None:
    dates = pd.Series([pd.Timestamp("2026-01-01"), pd.Timestamp("2018-01-01")])
    weights = time_decay_weights(dates, pd.Timestamp("2026-01-01"), half_life_years=8)
    assert np.isclose(weights[0], 1.0)
    assert np.isclose(weights[1], 0.5, atol=0.002)


def test_world_cup_weight_exceeds_friendly_weight() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "tournament": ["Friendly", "FIFA World Cup"],
            "is_friendly": [1, 0],
            "is_world_cup": [0, 1],
            "is_continental_competition": [0, 0],
        }
    )
    importance = match_importance_weights(frame)
    combined = combined_sample_weights(frame, "2020-01-01", half_life_years=8)
    assert importance[1] > importance[0]
    assert combined[1] > combined[0]


def test_configured_importance_profiles_and_none() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "tournament": ["Friendly", "FIFA World Cup qualification", "FIFA World Cup"],
            "stage": ["", "", "Final"],
            "is_friendly": [1, 0, 0],
            "is_world_cup": [0, 0, 1],
            "is_continental_competition": [0, 0, 0],
        }
    )
    aggressive = match_importance_weights(frame, "aggressive")
    none = combined_sample_weights(
        frame,
        "2024-01-01",
        half_life_years=1e9,
        use_match_importance=True,
        importance_profile="none",
    )

    assert aggressive[0] < aggressive[1] < aggressive[2]
    assert np.allclose(none, np.ones(3))
