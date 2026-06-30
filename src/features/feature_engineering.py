from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.validate_data import parse_bool
from src.features.elo import EloRatingState, classify_tournament


@dataclass
class TeamMatchHistory:
    recent_results: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_goal_diff: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_elo_performance: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_points_above_expectation: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_goal_diff_above_expectation: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_elo_change: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_adjusted_goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_adjusted_goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    last_match_date: pd.Timestamp | None = None

    def update(
        self,
        match_date: pd.Timestamp,
        goals_for: int,
        goals_against: int,
        expected_result: float,
        expected_goals_for: float,
        expected_goals_against: float,
        elo_change: float,
    ) -> None:
        points = 3.0 if goals_for > goals_against else 1.0 if goals_for == goals_against else 0.0
        actual_result = 1.0 if goals_for > goals_against else 0.5 if goals_for == goals_against else 0.0
        self.recent_results.append(points)
        self.recent_goal_diff.append(float(goals_for - goals_against))
        self.recent_goals_for.append(float(goals_for))
        self.recent_goals_against.append(float(goals_against))
        self.recent_elo_performance.append(float(actual_result - expected_result))
        self.recent_points_above_expectation.append(float(points - 3.0 * expected_result))
        self.recent_goal_diff_above_expectation.append(
            float((goals_for - goals_against) - (expected_goals_for - expected_goals_against))
        )
        self.recent_elo_change.append(float(elo_change))
        self.recent_adjusted_goals_for.append(float(goals_for - expected_goals_for))
        self.recent_adjusted_goals_against.append(float(goals_against - expected_goals_against))
        self.last_match_date = match_date


@dataclass
class FeatureState:
    elo_state: EloRatingState = field(default_factory=EloRatingState)
    team_history: dict[str, TeamMatchHistory] = field(default_factory=dict)
    attack_ratings: dict[str, float] = field(default_factory=dict)
    defence_ratings: dict[str, float] = field(default_factory=dict)

    def history_for_team(self, team: str) -> TeamMatchHistory:
        if team not in self.team_history:
            self.team_history[team] = TeamMatchHistory()
        return self.team_history[team]

    def to_dict(self) -> dict[str, Any]:
        return {
            "elo_state": self.elo_state.to_dict(),
            "team_history": {
                team: {
                    "recent_results": list(history.recent_results),
                    "recent_goal_diff": list(history.recent_goal_diff),
                    "recent_goals_for": list(history.recent_goals_for),
                    "recent_goals_against": list(history.recent_goals_against),
                    "recent_elo_performance": list(history.recent_elo_performance),
                    "recent_points_above_expectation": list(history.recent_points_above_expectation),
                    "recent_goal_diff_above_expectation": list(history.recent_goal_diff_above_expectation),
                    "recent_elo_change": list(history.recent_elo_change),
                    "recent_adjusted_goals_for": list(history.recent_adjusted_goals_for),
                    "recent_adjusted_goals_against": list(history.recent_adjusted_goals_against),
                    "last_match_date": history.last_match_date.isoformat() if history.last_match_date is not None else None,
                }
                for team, history in self.team_history.items()
            },
            "attack_ratings": self.attack_ratings,
            "defence_ratings": self.defence_ratings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureState":
        elo_state = EloRatingState.from_dict(data.get("elo_state", {}))
        state = cls(elo_state=elo_state)
        for team, history_data in data.get("team_history", {}).items():
            history = TeamMatchHistory()
            history.recent_results.extend(history_data.get("recent_results", []))
            history.recent_goal_diff.extend(history_data.get("recent_goal_diff", []))
            history.recent_goals_for.extend(history_data.get("recent_goals_for", []))
            history.recent_goals_against.extend(history_data.get("recent_goals_against", []))
            history.recent_elo_performance.extend(history_data.get("recent_elo_performance", []))
            history.recent_points_above_expectation.extend(history_data.get("recent_points_above_expectation", []))
            history.recent_goal_diff_above_expectation.extend(history_data.get("recent_goal_diff_above_expectation", []))
            history.recent_elo_change.extend(history_data.get("recent_elo_change", []))
            history.recent_adjusted_goals_for.extend(history_data.get("recent_adjusted_goals_for", []))
            history.recent_adjusted_goals_against.extend(history_data.get("recent_adjusted_goals_against", []))
            last_match_date = history_data.get("last_match_date")
            history.last_match_date = pd.Timestamp(last_match_date) if last_match_date else None
            state.team_history[team] = history
        state.attack_ratings = {str(team): float(value) for team, value in data.get("attack_ratings", {}).items()}
        state.defence_ratings = {str(team): float(value) for team, value in data.get("defence_ratings", {}).items()}
        return state


def recent_form_points(history: TeamMatchHistory, window: int) -> float:
    return float(sum(list(history.recent_results)[-window:]))


def recent_goal_diff(history: TeamMatchHistory, window: int) -> float:
    return float(sum(list(history.recent_goal_diff)[-window:]))


def average_goals(values: deque[float], window: int) -> float:
    subset = list(values)[-window:]
    if not subset:
        return 0.0
    return float(np.mean(subset))


def recent_average(values: deque[float], window: int) -> float:
    subset = list(values)[-window:]
    return float(np.mean(subset)) if subset else 0.0


def expected_goals_from_elo(team_elo: float, opponent_elo: float) -> float:
    return float(np.clip(1.35 * 10.0 ** ((team_elo - opponent_elo) / 800.0), 0.25, 4.5))


def days_since_last_match(history: TeamMatchHistory, match_date: pd.Timestamp) -> float:
    if history.last_match_date is None:
        return -1.0
    return float((pd.Timestamp(match_date) - history.last_match_date).days)


def _base_feature_row(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    tournament: str,
    neutral: bool,
    venue_country: str | None,
    stage: str | None,
    feature_state: FeatureState,
) -> dict[str, Any]:
    home_history = feature_state.history_for_team(home_team)
    away_history = feature_state.history_for_team(away_team)
    tournament_flags = classify_tournament(tournament)

    home_elo = feature_state.elo_state.get_rating(home_team)
    away_elo = feature_state.elo_state.get_rating(away_team)
    venue_label = str(venue_country).strip().lower() if pd.notna(venue_country) else ""
    playing_in_home_country_home = int(bool(venue_label) and venue_label == home_team.strip().lower())
    playing_in_home_country_away = int(bool(venue_label) and venue_label == away_team.strip().lower())
    host_country_flag = int(playing_in_home_country_home or playing_in_home_country_away)
    stage_label = str(stage or "").strip().lower()
    knockout_flag = int(any(label in stage_label for label in ["round", "quarter", "semi", "final", "knockout"]))
    group_stage_flag = int("group" in stage_label)

    return {
        "date": pd.Timestamp(match_date),
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "match_type": tournament_flags["match_type"],
        "neutral": int(bool(neutral)),
        "is_friendly": tournament_flags["is_friendly"],
        "is_world_cup": tournament_flags["is_world_cup"],
        "is_continental_competition": tournament_flags["is_continental_competition"],
        "home_advantage_flag": int(not neutral),
        "host_country_flag": host_country_flag,
        "playing_in_home_country_home": playing_in_home_country_home,
        "playing_in_home_country_away": playing_in_home_country_away,
        "group_stage_flag": group_stage_flag,
        "knockout_flag": knockout_flag,
        "home_elo_pre": float(home_elo),
        "away_elo_pre": float(away_elo),
        "elo_diff": float(home_elo - away_elo),
        "recent_form_points_home_5": recent_form_points(home_history, 5),
        "recent_form_points_away_5": recent_form_points(away_history, 5),
        "recent_form_points_home_10": recent_form_points(home_history, 10),
        "recent_form_points_away_10": recent_form_points(away_history, 10),
        "recent_goal_diff_home_5": recent_goal_diff(home_history, 5),
        "recent_goal_diff_away_5": recent_goal_diff(away_history, 5),
        "recent_goal_diff_home_10": recent_goal_diff(home_history, 10),
        "recent_goal_diff_away_10": recent_goal_diff(away_history, 10),
        "rolling_goals_for_home_5": average_goals(home_history.recent_goals_for, 5),
        "rolling_goals_against_home_5": average_goals(home_history.recent_goals_against, 5),
        "rolling_goals_for_away_5": average_goals(away_history.recent_goals_for, 5),
        "rolling_goals_against_away_5": average_goals(away_history.recent_goals_against, 5),
        "rolling_goals_for_home_10": average_goals(home_history.recent_goals_for, 10),
        "rolling_goals_against_home_10": average_goals(home_history.recent_goals_against, 10),
        "rolling_goals_for_away_10": average_goals(away_history.recent_goals_for, 10),
        "rolling_goals_against_away_10": average_goals(away_history.recent_goals_against, 10),
        "goals_for_avg_home_10": average_goals(home_history.recent_goals_for, 10),
        "goals_against_avg_home_10": average_goals(home_history.recent_goals_against, 10),
        "goals_for_avg_away_10": average_goals(away_history.recent_goals_for, 10),
        "goals_against_avg_away_10": average_goals(away_history.recent_goals_against, 10),
        "days_since_last_match_home": days_since_last_match(home_history, pd.Timestamp(match_date)),
        "days_since_last_match_away": days_since_last_match(away_history, pd.Timestamp(match_date)),
        "rest_days_diff": days_since_last_match(home_history, pd.Timestamp(match_date))
        - days_since_last_match(away_history, pd.Timestamp(match_date)),
        "tournament_importance": float(tournament_flags["tournament_importance"]),
        "opponent_adjusted_form_home_5": recent_average(home_history.recent_elo_performance, 5),
        "opponent_adjusted_form_away_5": recent_average(away_history.recent_elo_performance, 5),
        "opponent_adjusted_form_home_10": recent_average(home_history.recent_elo_performance, 10),
        "opponent_adjusted_form_away_10": recent_average(away_history.recent_elo_performance, 10),
        "points_above_expectation_home_5": recent_average(home_history.recent_points_above_expectation, 5),
        "points_above_expectation_away_5": recent_average(away_history.recent_points_above_expectation, 5),
        "points_above_expectation_home_10": recent_average(home_history.recent_points_above_expectation, 10),
        "points_above_expectation_away_10": recent_average(away_history.recent_points_above_expectation, 10),
        "goal_diff_above_expectation_home_5": recent_average(home_history.recent_goal_diff_above_expectation, 5),
        "goal_diff_above_expectation_away_5": recent_average(away_history.recent_goal_diff_above_expectation, 5),
        "goal_diff_above_expectation_home_10": recent_average(home_history.recent_goal_diff_above_expectation, 10),
        "goal_diff_above_expectation_away_10": recent_average(away_history.recent_goal_diff_above_expectation, 10),
        "recent_elo_change_home_5": float(sum(list(home_history.recent_elo_change)[-5:])),
        "recent_elo_change_away_5": float(sum(list(away_history.recent_elo_change)[-5:])),
        "recent_elo_change_home_10": float(sum(list(home_history.recent_elo_change)[-10:])),
        "recent_elo_change_away_10": float(sum(list(away_history.recent_elo_change)[-10:])),
        "opponent_adjusted_goals_for_home_10": recent_average(home_history.recent_adjusted_goals_for, 10),
        "opponent_adjusted_goals_against_home_10": recent_average(home_history.recent_adjusted_goals_against, 10),
        "opponent_adjusted_goals_for_away_10": recent_average(away_history.recent_adjusted_goals_for, 10),
        "opponent_adjusted_goals_against_away_10": recent_average(away_history.recent_adjusted_goals_against, 10),
        "home_attack_rating_pre": float(feature_state.attack_ratings.get(home_team, 0.0)),
        "home_defence_rating_pre": float(feature_state.defence_ratings.get(home_team, 0.0)),
        "away_attack_rating_pre": float(feature_state.attack_ratings.get(away_team, 0.0)),
        "away_defence_rating_pre": float(feature_state.defence_ratings.get(away_team, 0.0)),
    }


def build_feature_rows(
    matches_frame: pd.DataFrame,
    feature_state: FeatureState | None = None,
    update_state: bool = True,
) -> tuple[pd.DataFrame, FeatureState]:
    state = feature_state or FeatureState()
    frame = matches_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "match_date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["match_date"].fillna(frame["date"]))
    frame = frame.sort_values("date").reset_index(drop=True)

    feature_rows: list[dict[str, Any]] = []
    for _, same_day_matches in frame.groupby("date", sort=True):
        pending_updates: list[tuple[str, str, pd.Timestamp, str, bool, int, int, float, float]] = []
        for _, row in same_day_matches.iterrows():
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            match_date = pd.Timestamp(row["date"])
            tournament = str(row.get("tournament", ""))
            neutral = parse_bool(row.get("neutral", False))
            venue_country = row.get("country")
            feature_row = _base_feature_row(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                tournament=tournament,
                neutral=neutral,
                venue_country=venue_country,
                stage=str(row.get("stage", "")),
                feature_state=state,
            )
            if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
                home_score = int(row["home_score"])
                away_score = int(row["away_score"])
                feature_row.update(
                    {
                        "home_score": home_score,
                        "away_score": away_score,
                        "goal_diff": home_score - away_score,
                        "result": 0 if home_score > away_score else 1 if home_score == away_score else 2,
                    }
                )
                pending_updates.append(
                    (
                        home_team,
                        away_team,
                        match_date,
                        tournament,
                        neutral,
                        home_score,
                        away_score,
                        float(feature_row["home_elo_pre"]),
                        float(feature_row["away_elo_pre"]),
                    )
                )
            feature_rows.append(feature_row)

        if update_state:
            for home_team, away_team, match_date, tournament, neutral, home_score, away_score, home_elo, away_elo in pending_updates:
                elo_update = state.elo_state.update_match(
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    tournament=tournament,
                    match_date=match_date,
                    neutral=neutral,
                )
                expected_home_goals = expected_goals_from_elo(home_elo, away_elo)
                expected_away_goals = expected_goals_from_elo(away_elo, home_elo)
                home_attack_residual = home_score - expected_home_goals
                away_attack_residual = away_score - expected_away_goals
                learning_rate = 0.12
                state.attack_ratings[home_team] = (
                    (1.0 - learning_rate) * state.attack_ratings.get(home_team, 0.0)
                    + learning_rate * home_attack_residual
                )
                state.attack_ratings[away_team] = (
                    (1.0 - learning_rate) * state.attack_ratings.get(away_team, 0.0)
                    + learning_rate * away_attack_residual
                )
                state.defence_ratings[home_team] = (
                    (1.0 - learning_rate) * state.defence_ratings.get(home_team, 0.0)
                    + learning_rate * (away_score - expected_away_goals)
                )
                state.defence_ratings[away_team] = (
                    (1.0 - learning_rate) * state.defence_ratings.get(away_team, 0.0)
                    + learning_rate * (home_score - expected_home_goals)
                )
                state.history_for_team(home_team).update(
                    match_date,
                    home_score,
                    away_score,
                    elo_update["home_expected"],
                    expected_home_goals,
                    expected_away_goals,
                    elo_update["home_elo_post"] - elo_update["home_elo_pre"],
                )
                state.history_for_team(away_team).update(
                    match_date,
                    away_score,
                    home_score,
                    elo_update["away_expected"],
                    expected_away_goals,
                    expected_home_goals,
                    elo_update["away_elo_post"] - elo_update["away_elo_pre"],
                )

    return pd.DataFrame(feature_rows), state


def build_single_match_features(
    home_team: str,
    away_team: str,
    match_date: datetime | pd.Timestamp,
    neutral: bool,
    venue_country: str | None,
    tournament: str,
    feature_state: FeatureState,
    stage: str | None = None,
) -> pd.DataFrame:
    row = _base_feature_row(
        home_team=home_team,
        away_team=away_team,
        match_date=pd.Timestamp(match_date),
        tournament=tournament,
        neutral=neutral,
        venue_country=venue_country,
        stage=stage,
        feature_state=feature_state,
    )
    return pd.DataFrame([row])
