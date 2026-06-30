from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.validate_data import parse_bool


TOURNAMENT_K_FACTORS = {
    "friendly": 20.0,
    "qualifier": 25.0,
    "continental": 30.0,
    "world cup": 40.0,
}


def classify_tournament(tournament: str) -> dict[str, float | int]:
    label = str(tournament).strip().lower()
    is_qualifier = int(any(keyword in label for keyword in ["qualifier", "qualification", "cup qualification", "wcq"]))
    is_world_cup = int("world cup" in label and not is_qualifier)
    is_friendly = int("friendly" in label)
    is_minor_competitive = int("nations league" in label)
    is_continental = int(
        any(keyword in label for keyword in ["copa america", "euro", "africa cup", "asian cup", "gold cup"])
        and not is_qualifier
    )
    match_type = (
        "world_cup"
        if is_world_cup
        else "continental"
        if is_continental
        else "qualifier"
        if is_qualifier
        else "minor_competitive"
        if is_minor_competitive
        else "friendly"
        if is_friendly
        else "default"
    )
    return {
        "tournament_importance": float(
            TOURNAMENT_K_FACTORS["world cup"]
            if is_world_cup
            else TOURNAMENT_K_FACTORS["continental"]
            if is_continental
            else TOURNAMENT_K_FACTORS["qualifier"]
            if is_qualifier or is_minor_competitive
            else TOURNAMENT_K_FACTORS["friendly"]
            if is_friendly
            else TOURNAMENT_K_FACTORS["qualifier"]
        ),
        "is_friendly": is_friendly,
        "is_world_cup": is_world_cup,
        "is_continental_competition": is_continental,
        "is_minor_competitive": is_minor_competitive,
        "match_type": match_type,
    }


def get_match_k_factor(tournament: str, base_k: float) -> float:
    classification = classify_tournament(tournament)
    importance = classification["tournament_importance"]
    return float(importance if importance else base_k)


def goal_diff_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.25
    if goal_diff == 3:
        return 1.5
    return min(2.0, 1.75 + 0.05 * (goal_diff - 3))


def expected_score(team_rating: float, opponent_rating: float, neutral: bool, home_advantage: float) -> float:
    adjusted_home_rating = team_rating + (0.0 if neutral else home_advantage)
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - adjusted_home_rating) / 400.0))


@dataclass
class EloRatingState:
    base_rating: float = 1500.0
    home_advantage: float = 65.0
    friendly_k: float = 20.0
    qualifier_k: float = 25.0
    continental_k: float = 30.0
    world_cup_k: float = 40.0
    default_k: float = 25.0
    ratings: dict[str, float] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def get_rating(self, team: str) -> float:
        return float(self.ratings.get(team, self.base_rating))

    def get_team_elo_before_date(self, team: str, date: datetime | pd.Timestamp) -> float:
        target_timestamp = pd.Timestamp(date)
        latest_rating = self.base_rating
        latest_date: pd.Timestamp | None = None
        for snapshot in self.snapshots:
            snapshot_date = pd.Timestamp(snapshot["date"])
            snapshot_ratings = snapshot.get("ratings", {})
            if (
                team in snapshot_ratings
                and snapshot_date < target_timestamp
                and (latest_date is None or snapshot_date >= latest_date)
            ):
                latest_date = snapshot_date
                latest_rating = float(snapshot_ratings[team])
        return latest_rating

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        match_date: datetime | pd.Timestamp,
        neutral: bool,
    ) -> dict[str, float]:
        home_pre = self.get_rating(home_team)
        away_pre = self.get_rating(away_team)
        home_expected = expected_score(home_pre, away_pre, neutral=neutral, home_advantage=self.home_advantage)
        away_expected = 1.0 - home_expected

        actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        actual_away = 1.0 - actual_home if home_score != away_score else 0.5

        classification = classify_tournament(tournament)
        if classification["is_world_cup"]:
            k_factor = self.world_cup_k
        elif classification["is_continental_competition"]:
            k_factor = self.continental_k
        elif "qualifier" in str(tournament).lower() or "qualification" in str(tournament).lower():
            k_factor = self.qualifier_k
        elif classification["is_friendly"]:
            k_factor = self.friendly_k
        else:
            k_factor = self.default_k
        goal_multiplier = goal_diff_multiplier(abs(int(home_score) - int(away_score)))
        effective_k = k_factor * goal_multiplier

        home_new = home_pre + effective_k * (actual_home - home_expected)
        away_new = away_pre + effective_k * (actual_away - away_expected)

        self.ratings[home_team] = float(home_new)
        self.ratings[away_team] = float(away_new)

        snapshot = {
            "date": pd.Timestamp(match_date).isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "ratings": {home_team: float(home_new), away_team: float(away_new)},
        }
        self.snapshots.append(snapshot)
        return {
            "home_elo_pre": float(home_pre),
            "away_elo_pre": float(away_pre),
            "home_elo_post": float(home_new),
            "away_elo_post": float(away_new),
            "home_expected": float(home_expected),
            "away_expected": float(away_expected),
            "k_factor": float(effective_k),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_rating": self.base_rating,
            "home_advantage": self.home_advantage,
            "friendly_k": self.friendly_k,
            "qualifier_k": self.qualifier_k,
            "continental_k": self.continental_k,
            "world_cup_k": self.world_cup_k,
            "default_k": self.default_k,
            "ratings": self.ratings,
            "snapshots": self.snapshots,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EloRatingState":
        state = cls(
            base_rating=float(data.get("base_rating", 1500.0)),
            home_advantage=float(data.get("home_advantage", 65.0)),
            friendly_k=float(data.get("friendly_k", 20.0)),
            qualifier_k=float(data.get("qualifier_k", 25.0)),
            continental_k=float(data.get("continental_k", 30.0)),
            world_cup_k=float(data.get("world_cup_k", 40.0)),
            default_k=float(data.get("default_k", 25.0)),
        )
        state.ratings = {str(key): float(value) for key, value in data.get("ratings", {}).items()}
        state.snapshots = list(data.get("snapshots", []))
        return state


def build_elo_history(results_frame: pd.DataFrame, state: EloRatingState | None = None) -> pd.DataFrame:
    working_state = state or EloRatingState()
    frame = results_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    feature_rows: list[dict[str, float | str | int | pd.Timestamp]] = []
    for _, row in frame.iterrows():
        match_state = working_state.update_match(
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            home_score=int(row["home_score"]),
            away_score=int(row["away_score"]),
            tournament=str(row["tournament"]),
            match_date=row["date"],
            neutral=parse_bool(row["neutral"]),
        )
        feature_rows.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_elo_pre": match_state["home_elo_pre"],
                "away_elo_pre": match_state["away_elo_pre"],
                "home_elo_post": match_state["home_elo_post"],
                "away_elo_post": match_state["away_elo_post"],
            }
        )
    return pd.DataFrame(feature_rows)
