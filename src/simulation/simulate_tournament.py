from __future__ import annotations

import json
from collections import defaultdict
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.data.build_dataset import load_fixtures_frame
from src.data.validate_data import load_team_mappings
from src.simulation.simulate_group_stage import simulate_single_group
from src.simulation.simulate_knockout import simulate_knockout_round
from src.utils.logging import setup_logging
from src.utils.paths import MANUAL_DATA_DIR, PREDICTIONS_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)


def _build_group_fixture_map(fixtures: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for group_label, group_frame in fixtures.groupby(fixtures["group"].fillna("")):
        grouped[str(group_label)] = group_frame.copy()
    return grouped


def _best_third_rank(third_place_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        third_place_rows,
        key=lambda row: (row["points"], row["goal_difference"], row["goals_for"], row["random_tiebreaker"]),
        reverse=True,
    )


def _load_third_place_allocations() -> dict[str, dict[str, str]]:
    path = MANUAL_DATA_DIR / "worldcup_2026_third_place_allocations.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FIFA third-place allocation table at {path}. "
            "Run scripts/populate_worldcup_2026_fixtures.py."
        )
    frame = pd.read_csv(path, dtype=str)
    winner_slots = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]
    return {
        str(row["qualifying_groups"]): {winner_slot: str(row[winner_slot]) for winner_slot in winner_slots}
        for _, row in frame.iterrows()
    }


def _stage_key(stage_label: str) -> str:
    normalized = str(stage_label).lower().replace(" ", "_")
    if "round_of_32" in normalized or "r32" in normalized:
        return "round_of_32"
    if "round_of_16" in normalized or "r16" in normalized:
        return "round_of_16"
    if "quarter" in normalized or normalized == "qf":
        return "quarter_final"
    if "semi" in normalized or normalized == "sf":
        return "semi_final"
    if "final" in normalized:
        return "final"
    return normalized


def _load_fixtures() -> pd.DataFrame:
    mappings = load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv")
    fixtures = load_fixtures_frame(mappings)
    fixtures["match_date"] = pd.to_datetime(fixtures["match_date"])
    return fixtures


def _validate_tournament_structure(group_map: dict[str, pd.DataFrame], knockout_fixtures: pd.DataFrame) -> None:
    if len(group_map) != 12:
        raise ValueError(f"World Cup 2026 simulation requires exactly 12 groups; found {len(group_map)}.")
    all_group_teams: set[str] = set()
    for group_label, fixtures in group_map.items():
        teams = set(fixtures["home_team"]).union(fixtures["away_team"])
        if len(teams) != 4 or len(fixtures) != 6:
            raise ValueError(
                f"Group {group_label} must contain exactly 4 teams and 6 fixtures; "
                f"found {len(teams)} teams and {len(fixtures)} fixtures."
            )
        all_group_teams.update(teams)
    if len(all_group_teams) != 48:
        raise ValueError(f"World Cup 2026 group fixtures must contain 48 unique teams; found {len(all_group_teams)}.")

    stage_counts = knockout_fixtures["stage"].astype(str).map(_stage_key).value_counts().to_dict()
    required_counts = {
        "round_of_32": 16,
        "round_of_16": 8,
        "quarter_final": 4,
        "semi_final": 2,
        "final": 1,
    }
    incorrect = {
        stage: (stage_counts.get(stage, 0), required)
        for stage, required in required_counts.items()
        if stage_counts.get(stage, 0) != required
    }
    if incorrect:
        details = ", ".join(f"{stage}: found {found}, expected {expected}" for stage, (found, expected) in incorrect.items())
        raise ValueError(f"Incomplete World Cup 2026 knockout bracket. {details}")


def simulate_tournament(n_simulations: int = 10000, seed: int = 42) -> pd.DataFrame:
    ensure_project_dirs()
    fixtures = _load_fixtures()
    group_fixtures = fixtures[fixtures["stage"].astype(str).str.contains("group", case=False, na=False)].copy()
    knockout_fixtures = fixtures[~fixtures.index.isin(group_fixtures.index)].copy()
    group_map = _build_group_fixture_map(group_fixtures)
    third_place_allocations = _load_third_place_allocations()

    team_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    advancement_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    team_groups = {
        team: str(group_label)
        for group_label, group_frame in group_map.items()
        for team in set(group_frame["home_team"]).union(group_frame["away_team"])
    }
    group_labels = sorted(group_map.keys())
    _validate_tournament_structure(group_map, knockout_fixtures)
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    started_at = perf_counter()
    progress_interval = max(1, n_simulations // 20)
    LOGGER.info("Starting %s tournament simulations with seed %s.", n_simulations, seed)
    for simulation_index in range(n_simulations):
        simulation_rng = np.random.default_rng(seed + simulation_index)
        qualifiers: dict[str, str] = {}
        group_positions: list[dict[str, Any]] = []
        qualified_teams: set[str] = set()

        for group_label, group_frame in group_map.items():
            ranked, table = simulate_single_group(group_label, group_frame, simulation_rng)
            for position, entry in enumerate(ranked, start=1):
                advancement_counts[entry.team][f"group_{position}"] += 1
                team_stats[entry.team]["expected_points_group"] += entry.points
                team_stats[entry.team]["expected_goal_difference_group"] += entry.goal_difference
                team_stats[entry.team]["expected_goals_for_group"] += entry.goals_for
                team_stats[entry.team]["expected_goals_against_group"] += entry.goals_against
            for position, entry in enumerate(ranked[:2], start=1):
                qualifiers[f"{group_label}{position}"] = entry.team
                qualifiers[f"{position}{group_label}"] = entry.team
                qualified_teams.add(entry.team)
            third_entry = ranked[2]
            group_positions.append(
                {
                    "team": third_entry.team,
                    "group": group_label,
                    "points": third_entry.points,
                    "goal_difference": third_entry.goal_difference,
                    "goals_for": third_entry.goals_for,
                    "random_tiebreaker": float(simulation_rng.random()),
                }
            )

        best_third_teams = _best_third_rank(group_positions)[:8]
        qualifying_groups = "".join(sorted(str(entry["group"]) for entry in best_third_teams))
        if qualifying_groups not in third_place_allocations:
            raise KeyError(f"No FIFA third-place allocation found for qualifying groups {qualifying_groups}.")
        third_teams_by_group = {f"3{entry['group']}": entry["team"] for entry in best_third_teams}
        for entry in best_third_teams:
            qualified_teams.add(entry["team"])
        for winner_slot, third_slot in third_place_allocations[qualifying_groups].items():
            qualifiers[f"THIRD_FOR_{winner_slot}"] = third_teams_by_group[third_slot]

        for team in qualified_teams:
            advancement_counts[team]["reached_r32"] += 1

        if knockout_fixtures.empty:
            continue

        slot_mapping = dict(qualifiers)
        knockout_groups = knockout_fixtures.groupby(knockout_fixtures["stage"].astype(str).map(_stage_key))
        for stage_label in ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]:
            if stage_label not in knockout_groups.groups:
                continue
            stage_fixtures = knockout_groups.get_group(stage_label)
            stage_results = simulate_knockout_round(stage_fixtures, slot_mapping, simulation_rng)
            next_reach_key = {
                "round_of_32": "reached_r16",
                "round_of_16": "reached_qf",
                "quarter_final": "reached_sf",
                "semi_final": "reached_final",
                "final": "champion",
            }[stage_label]
            for match_result in stage_results:
                advancement_counts[match_result["winner"]][next_reach_key] += 1

        completed = simulation_index + 1
        if completed % progress_interval == 0 or completed == n_simulations:
            elapsed = perf_counter() - started_at
            LOGGER.info(
                "Simulation progress: %s/%s (%.0f%%), elapsed %.1fs, %.1f tournaments/s.",
                completed,
                n_simulations,
                completed * 100.0 / n_simulations,
                elapsed,
                completed / max(elapsed, 1e-9),
            )

    summary_rows: list[dict[str, Any]] = []
    for team, counts in advancement_counts.items():
        summary_rows.append(
            {
                "team": team,
                "group": team_groups.get(team, ""),
                "p_group_1st": counts.get("group_1", 0.0) / n_simulations,
                "p_group_2nd": counts.get("group_2", 0.0) / n_simulations,
                "p_group_3rd": counts.get("group_3", 0.0) / n_simulations,
                "p_group_4th": counts.get("group_4", 0.0) / n_simulations,
                "p_reach_r32": float(counts.get("reached_r32", 0.0) / n_simulations),
                "p_reach_r16": float(counts.get("reached_r16", 0.0) / n_simulations),
                "p_reach_qf": float(counts.get("reached_qf", 0.0) / n_simulations),
                "p_reach_sf": float(counts.get("reached_sf", 0.0) / n_simulations),
                "p_reach_final": float(counts.get("reached_final", 0.0) / n_simulations),
                "p_champion": float(counts.get("champion", 0.0) / n_simulations),
                "expected_points_group": float(team_stats[team].get("expected_points_group", 0.0) / n_simulations),
                "expected_goal_difference_group": float(team_stats[team].get("expected_goal_difference_group", 0.0) / n_simulations),
                "expected_goals_for_group": float(team_stats[team].get("expected_goals_for_group", 0.0) / n_simulations),
                "expected_goals_against_group": float(team_stats[team].get("expected_goals_against_group", 0.0) / n_simulations),
                "expected_group_points": float(team_stats[team].get("expected_points_group", 0.0) / n_simulations),
                "expected_group_goal_difference": float(team_stats[team].get("expected_goal_difference_group", 0.0) / n_simulations),
                "expected_group_goals_for": float(team_stats[team].get("expected_goals_for_group", 0.0) / n_simulations),
                "expected_group_goals_against": float(team_stats[team].get("expected_goals_against_group", 0.0) / n_simulations),
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values("p_champion", ascending=False)
    summary_path = PREDICTIONS_DIR / "tournament_simulation_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    LOGGER.info("Saved tournament simulation summary to %s", summary_path)
    return summary_frame
