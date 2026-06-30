from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.features.elo import classify_tournament, expected_score, goal_diff_multiplier
from src.utils.config import config_value


@dataclass
class DynamicRatingState:
    model_name: str = "standard_elo"
    k_scale: float = 1.0
    ratings: dict[str, float] = field(default_factory=dict)
    uncertainties: dict[str, float] = field(default_factory=dict)
    last_dates: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "k_scale": self.k_scale,
            "ratings": self.ratings,
            "uncertainties": self.uncertainties,
            "last_dates": self.last_dates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicRatingState":
        return cls(
            model_name=str(data.get("model_name", "standard_elo")),
            k_scale=float(data.get("k_scale", 1.0)),
            ratings={str(team): float(value) for team, value in data.get("ratings", {}).items()},
            uncertainties={str(team): float(value) for team, value in data.get("uncertainties", {}).items()},
            last_dates={str(team): str(value) for team, value in data.get("last_dates", {}).items()},
        )


def _base_k(tournament: str) -> float:
    flags = classify_tournament(tournament)
    if flags["is_world_cup"]:
        return float(config_value("elo", "world_cup_k", default=40.0))
    if flags["is_continental_competition"]:
        return float(config_value("elo", "continental_k", default=30.0))
    if flags["match_type"] == "qualifier":
        return float(config_value("elo", "qualifier_k", default=25.0))
    if flags["is_friendly"]:
        return float(config_value("elo", "friendly_k", default=20.0))
    return float(config_value("elo", "default_k", default=25.0))


def _rating_before(state: DynamicRatingState, team: str, match_date: pd.Timestamp) -> tuple[float, float]:
    base = float(config_value("elo", "base_rating", default=1500.0))
    rating = float(state.ratings.get(team, base))
    uncertainty = float(state.uncertainties.get(team, 1.0))
    if state.model_name != "smoothed_dynamic" or team not in state.last_dates:
        return rating, uncertainty
    inactive_years = max((match_date - pd.Timestamp(state.last_dates[team])).days / 365.25, 0.0)
    shrinkage = float(np.exp(-inactive_years / 8.0))
    rating = base + (rating - base) * shrinkage
    uncertainty = min(1.6, uncertainty + 0.12 * np.sqrt(inactive_years))
    return rating, uncertainty


def build_rating_features(
    frame: pd.DataFrame,
    model_name: str = "standard_elo",
    k_scale: float = 1.0,
) -> tuple[pd.DataFrame, DynamicRatingState]:
    if model_name not in {"standard_elo", "smoothed_dynamic"}:
        raise ValueError(f"Unsupported rating model {model_name!r}.")
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values("date", kind="stable")
    state = DynamicRatingState(model_name=model_name, k_scale=float(k_scale))
    rows: list[dict[str, float]] = []
    home_advantage = float(config_value("elo", "home_advantage", default=65.0))
    for match_date, same_day in working.groupby("date", sort=True):
        pending: list[dict[str, Any]] = []
        for index, match in same_day.iterrows():
            home_team = str(match["home_team"])
            away_team = str(match["away_team"])
            home_rating, home_uncertainty = _rating_before(state, home_team, pd.Timestamp(match_date))
            away_rating, away_uncertainty = _rating_before(state, away_team, pd.Timestamp(match_date))
            rows.append(
                {
                    "_index": index,
                    "home_elo_pre": home_rating,
                    "away_elo_pre": away_rating,
                    "elo_diff": home_rating - away_rating,
                }
            )
            if pd.notna(match.get("home_score")) and pd.notna(match.get("away_score")):
                expected_home = expected_score(
                    home_rating,
                    away_rating,
                    neutral=bool(match.get("neutral", True)),
                    home_advantage=home_advantage,
                )
                actual_home = 1.0 if match["home_score"] > match["away_score"] else 0.5 if match["home_score"] == match["away_score"] else 0.0
                multiplier = goal_diff_multiplier(abs(int(match["home_score"]) - int(match["away_score"])))
                base_k = _base_k(str(match.get("tournament", ""))) * float(k_scale) * multiplier
                home_k = base_k * (home_uncertainty if model_name == "smoothed_dynamic" else 1.0)
                away_k = base_k * (away_uncertainty if model_name == "smoothed_dynamic" else 1.0)
                pending.append(
                    {
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_rating": home_rating,
                        "away_rating": away_rating,
                        "home_delta": home_k * (actual_home - expected_home),
                        "away_delta": away_k * ((1.0 - actual_home) - (1.0 - expected_home)),
                        "home_uncertainty": home_uncertainty,
                        "away_uncertainty": away_uncertainty,
                    }
                )
        deltas: dict[str, float] = defaultdict(float)
        pre_ratings: dict[str, float] = {}
        new_uncertainties: dict[str, list[float]] = defaultdict(list)
        for update in pending:
            for side in ["home", "away"]:
                team = update[f"{side}_team"]
                pre_ratings[team] = update[f"{side}_rating"]
                deltas[team] += update[f"{side}_delta"]
                new_uncertainties[team].append(update[f"{side}_uncertainty"])
        for team, delta in deltas.items():
            state.ratings[team] = float(pre_ratings[team] + delta)
            if model_name == "smoothed_dynamic":
                state.uncertainties[team] = max(0.45, float(np.mean(new_uncertainties[team])) * 0.90)
            else:
                state.uncertainties[team] = 1.0
            state.last_dates[team] = pd.Timestamp(match_date).isoformat()
    rating_frame = pd.DataFrame(rows).set_index("_index").reindex(working.index)
    output = working.copy()
    for column in ["home_elo_pre", "away_elo_pre", "elo_diff"]:
        output[column] = rating_frame[column]
    return output.sort_index(), state


def apply_rating_state_to_match(
    features: pd.DataFrame,
    state: DynamicRatingState,
    home_team: str,
    away_team: str,
    match_date: str | pd.Timestamp,
) -> pd.DataFrame:
    output = features.copy()
    home, _ = _rating_before(state, home_team, pd.Timestamp(match_date))
    away, _ = _rating_before(state, away_team, pd.Timestamp(match_date))
    output.loc[:, "home_elo_pre"] = home
    output.loc[:, "away_elo_pre"] = away
    output.loc[:, "elo_diff"] = home - away
    return output
