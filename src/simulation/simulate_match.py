from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from src.models.predict_match import predict_match
from src.models.scoreline import scoreline_matrix
from src.utils.config import config_value


@dataclass
class SimulatedMatchResult:
    home_goals: int
    away_goals: int
    went_to_penalties: bool = False
    penalty_winner: str | None = None


@lru_cache(maxsize=65536)
def _match_distribution(
    home_team: str,
    away_team: str,
    match_date: str,
    neutral: bool,
    venue_country: str | None,
    tournament: str,
) -> tuple[dict[str, Any], np.ndarray, tuple[int, int]]:
    prediction = predict_match(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        neutral=neutral,
        venue_country=venue_country,
        tournament=tournament,
    )
    matrix_values = prediction.get("_scoreline_matrix_values")
    if matrix_values is None:
        matrix_values = scoreline_matrix(
            prediction["expected_goals_home"],
            prediction["expected_goals_away"],
            max_goals=int(config_value("modeling", "max_goals", default=10)),
            dixon_coles_rho=float(prediction.get("dixon_coles_rho", 0.0)),
        ).to_numpy()
    probabilities = np.asarray(matrix_values, dtype=float).ravel()
    probabilities.setflags(write=False)
    return prediction, probabilities, np.asarray(matrix_values).shape


def simulate_match(
    home_team: str,
    away_team: str,
    match_date,
    neutral: bool = True,
    venue_country: str | None = None,
    tournament: str = "World Cup",
    knockout: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    generator = rng or np.random.default_rng()
    prediction, probabilities, matrix_shape = _match_distribution(
        home_team,
        away_team,
        str(match_date),
        bool(neutral),
        venue_country,
        tournament,
    )
    sampled_index = int(generator.choice(probabilities.size, p=probabilities))
    home_goals, away_goals = np.unravel_index(sampled_index, matrix_shape)
    home_goals = int(home_goals)
    away_goals = int(away_goals)

    if knockout and home_goals == away_goals:
        extra_time_scale = float(config_value("simulation", "extra_time_goal_scale", default=0.35))
        extra_home = max(0, int(generator.poisson(prediction["expected_goals_home"] * extra_time_scale)))
        extra_away = max(0, int(generator.poisson(prediction["expected_goals_away"] * extra_time_scale)))
        home_goals += extra_home
        away_goals += extra_away
        if home_goals == away_goals:
            home_penalty_edge = prediction["p_home_win"] / max(prediction["p_home_win"] + prediction["p_away_win"], 1e-9)
            home_penalty_edge = float(np.clip(home_penalty_edge, 0.4, 0.6))
            home_wins = bool(generator.random() < home_penalty_edge)
            return {
                **prediction,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "went_to_penalties": True,
                "penalty_winner": home_team if home_wins else away_team,
            }

    return {
        **prediction,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "went_to_penalties": False,
        "penalty_winner": None,
    }
