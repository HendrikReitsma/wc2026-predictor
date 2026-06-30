from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.simulate_match import simulate_match


@dataclass
class GroupTableEntry:
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0
    random_tiebreaker: float = 0.0


def _build_group_table(teams: list[str], rng: np.random.Generator) -> dict[str, GroupTableEntry]:
    return {team: GroupTableEntry(team=team, random_tiebreaker=float(rng.random())) for team in teams}


def _update_table(table: dict[str, GroupTableEntry], home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
    home_entry = table[home_team]
    away_entry = table[away_team]
    home_entry.goals_for += home_goals
    home_entry.goals_against += away_goals
    away_entry.goals_for += away_goals
    away_entry.goals_against += home_goals
    home_entry.goal_difference = home_entry.goals_for - home_entry.goals_against
    away_entry.goal_difference = away_entry.goals_for - away_entry.goals_against
    if home_goals > away_goals:
        home_entry.points += 3
    elif home_goals < away_goals:
        away_entry.points += 3
    else:
        home_entry.points += 1
        away_entry.points += 1


def rank_group_table(table: dict[str, GroupTableEntry]) -> list[GroupTableEntry]:
    return sorted(
        table.values(),
        key=lambda entry: (entry.points, entry.goal_difference, entry.goals_for, entry.random_tiebreaker),
        reverse=True,
    )


def simulate_single_group(
    group_label: str,
    fixtures: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[list[GroupTableEntry], dict[str, GroupTableEntry]]:
    teams = sorted(set(fixtures["home_team"]).union(set(fixtures["away_team"])))
    if len(teams) != 4:
        raise ValueError(f"Group {group_label} must contain exactly 4 teams; found {len(teams)}.")
    if len(fixtures) != 6:
        raise ValueError(f"Group {group_label} must contain exactly 6 fixtures; found {len(fixtures)}.")
    table = _build_group_table(teams, rng)
    for _, fixture in fixtures.sort_values("match_date").iterrows():
        result = simulate_match(
            home_team=str(fixture["home_team"]),
            away_team=str(fixture["away_team"]),
            match_date=fixture["match_date"],
            neutral=bool(fixture["neutral"]),
            venue_country=str(fixture.get("country")) if pd.notna(fixture.get("country")) else None,
            tournament="World Cup",
            knockout=False,
            rng=rng,
        )
        _update_table(table, str(fixture["home_team"]), str(fixture["away_team"]), int(result["home_goals"]), int(result["away_goals"]))
    return rank_group_table(table), table
