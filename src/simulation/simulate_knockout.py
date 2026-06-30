from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.simulate_match import simulate_match


def _resolve_slot(slot: str, slot_mapping: dict[str, str]) -> str:
    normalized = slot.strip()
    compact_upper = normalized.upper().replace(" ", "")
    candidates = [
        normalized,
        normalized.upper(),
        normalized.replace(" ", ""),
        compact_upper.replace("WINNER", "W").replace(":", ""),
    ]
    if len(compact_upper) >= 2 and compact_upper[0].isdigit() and compact_upper[1:].isalpha():
        candidates.append(f"{compact_upper[1:]}{compact_upper[0]}")
    for candidate in candidates:
        if candidate in slot_mapping:
            return slot_mapping[candidate]
    raise KeyError(
        f"Bracket slot '{slot}' could not be resolved. Provide a complete bracket mapping in data/manual/worldcup_2026_fixtures.csv."
    )


def simulate_knockout_round(
    fixtures: pd.DataFrame,
    slot_mapping: dict[str, str],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    round_teams: set[str] = set()
    for _, fixture in fixtures.sort_values("match_date").iterrows():
        home_team = _resolve_slot(str(fixture["bracket_slot_home"]), slot_mapping)
        away_team = _resolve_slot(str(fixture["bracket_slot_away"]), slot_mapping)
        if home_team == away_team or home_team in round_teams or away_team in round_teams:
            raise ValueError(
                f"Invalid bracket mapping in stage {fixture['stage']}: teams cannot appear twice in one round."
            )
        round_teams.update([home_team, away_team])
        result = simulate_match(
            home_team=home_team,
            away_team=away_team,
            match_date=fixture["match_date"],
            neutral=bool(fixture["neutral"]),
            venue_country=str(fixture.get("country")) if pd.notna(fixture.get("country")) else None,
            tournament="World Cup",
            knockout=True,
            rng=rng,
        )
        winner = result["penalty_winner"] if result["went_to_penalties"] else home_team if result["home_goals"] > result["away_goals"] else away_team
        results.append(
            {
                "match_id": str(fixture["match_id"]),
                "stage": str(fixture["stage"]),
                "winner": winner,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": int(result["home_goals"]),
                "away_goals": int(result["away_goals"]),
            }
        )
        match_id = str(fixture["match_id"]).strip()
        for output_slot in [match_id, f"W{match_id}", f"WINNER{match_id}", f"WINNER:{match_id}"]:
            slot_mapping[output_slot] = winner
    return results
