from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from update_results import _extract_knockout_result  # noqa: E402


def _fixture(match_id: int = 74) -> pd.Series:
    return pd.Series(
        {
            "match_id": match_id,
            "match_date": "2026-06-29T16:30:00",
            "city": "Boston",
            "country": "United States",
            "neutral": True,
            "stage": "Round of 32",
        }
    )


def test_extract_knockout_result_handles_reversed_source_order() -> None:
    result = _extract_knockout_result(
        "Sunday, June 28\nCanada 1, South Africa 0",
        "https://example.test",
        _fixture(73),
        "South Africa",
        "Canada",
        {},
    )

    assert result["home_score"] == 0
    assert result["away_score"] == 1
    assert result["winner"] == "Canada"
    assert result["decided_by"] == "score"


def test_extract_knockout_result_handles_penalties() -> None:
    result = _extract_knockout_result(
        "Monday, June 29\nParaguay 1, Germany 1 (Paraguay wins 4-3 on penalties)",
        "https://example.test",
        _fixture(74),
        "Germany",
        "Paraguay",
        {},
    )

    assert result["home_score"] == 1
    assert result["away_score"] == 1
    assert result["winner"] == "Paraguay"
    assert result["decided_by"] == "penalties"
    assert result["penalty_home_score"] == 3
    assert result["penalty_away_score"] == 4
