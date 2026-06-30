from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.models.predict_match import predict_match
from src.simulation.simulate_group_stage import GroupTableEntry, rank_group_table
from src.simulation.simulate_knockout import _resolve_slot, simulate_knockout_round
from src.simulation.simulate_tournament import (
    _load_fixtures,
    _load_third_place_allocations,
    _stage_key,
    _validate_tournament_structure,
)
from src.utils.logging import setup_logging
from src.utils.paths import MANUAL_DATA_DIR, PREDICTIONS_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)
GROUP_RESULTS_PATH = MANUAL_DATA_DIR / "worldcup_2026_group_results.csv"


def _update_actual_table(
    table: dict[str, GroupTableEntry],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
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


def load_actual_group_results() -> pd.DataFrame:
    if not GROUP_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing actual group-stage results at {GROUP_RESULTS_PATH}. "
            "Run scripts/fetch_worldcup_2026_group_results.py first."
        )
    results = pd.read_csv(GROUP_RESULTS_PATH)
    if len(results) != 72 or results["match_id"].nunique() != 72:
        raise ValueError("Actual group-stage results must contain exactly 72 unique match_id values.")
    missing_scores = results[["home_score", "away_score"]].isna().any(axis=1)
    if missing_scores.any():
        raise ValueError("Actual group-stage results contain missing scores.")
    return results


def build_actual_group_tables(
    group_results: pd.DataFrame | None = None,
) -> tuple[dict[str, list[GroupTableEntry]], pd.DataFrame, pd.DataFrame]:
    results = (group_results.copy() if group_results is not None else load_actual_group_results()).sort_values(
        ["group", "date", "match_id"]
    )
    ranked_by_group: dict[str, list[GroupTableEntry]] = {}
    rows: list[dict[str, Any]] = []
    third_rows: list[dict[str, Any]] = []
    for group_label, group_frame in results.groupby("group", sort=True):
        teams = sorted(set(group_frame["home_team"]).union(group_frame["away_team"]))
        if len(teams) != 4 or len(group_frame) != 6:
            raise ValueError(f"Group {group_label} must have four teams and six results.")
        table = {team: GroupTableEntry(team=team, random_tiebreaker=0.0) for team in teams}
        for _, result in group_frame.iterrows():
            _update_actual_table(
                table,
                str(result["home_team"]),
                str(result["away_team"]),
                int(result["home_score"]),
                int(result["away_score"]),
            )
        ranked = rank_group_table(table)
        ranked_by_group[str(group_label)] = ranked
        for position, entry in enumerate(ranked, start=1):
            rows.append(
                {
                    "group": str(group_label),
                    "position": position,
                    "team": entry.team,
                    "points": entry.points,
                    "goals_for": entry.goals_for,
                    "goals_against": entry.goals_against,
                    "goal_difference": entry.goal_difference,
                }
            )
        third = ranked[2]
        third_rows.append(
            {
                "team": third.team,
                "group": str(group_label),
                "points": third.points,
                "goal_difference": third.goal_difference,
                "goals_for": third.goals_for,
                "random_tiebreaker": 0.0,
            }
        )
    standings = pd.DataFrame(rows)
    third_rank = pd.DataFrame(
        sorted(
            third_rows,
            key=lambda row: (row["points"], row["goal_difference"], row["goals_for"], row["team"]),
            reverse=True,
        )
    ).reset_index(drop=True)
    third_rank["third_place_rank"] = third_rank.index + 1
    third_rank["qualified"] = third_rank["third_place_rank"].le(8)
    return ranked_by_group, standings, third_rank


def build_actual_slot_mapping() -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    ranked_by_group, standings, third_rank = build_actual_group_tables()
    allocations = _load_third_place_allocations()
    slot_mapping: dict[str, str] = {}
    for group_label, ranked in ranked_by_group.items():
        for position, entry in enumerate(ranked[:2], start=1):
            slot_mapping[f"{group_label}{position}"] = entry.team
            slot_mapping[f"{position}{group_label}"] = entry.team
    best_thirds = third_rank[third_rank["qualified"]].copy()
    qualifying_groups = "".join(sorted(best_thirds["group"].astype(str)))
    if qualifying_groups not in allocations:
        raise KeyError(f"No third-place allocation for qualifying groups {qualifying_groups}.")
    third_teams_by_group = {f"3{row.group}": row.team for row in best_thirds.itertuples(index=False)}
    for winner_slot, third_slot in allocations[qualifying_groups].items():
        slot_mapping[f"THIRD_FOR_{winner_slot}"] = third_teams_by_group[third_slot]
    standings["qualified"] = standings["position"].le(2) | standings["team"].isin(best_thirds["team"])
    third_rank["qualifying_groups"] = qualifying_groups
    return slot_mapping, standings, third_rank


def build_round_of_32_predictions(slot_mapping: dict[str, str]) -> pd.DataFrame:
    fixtures = _load_fixtures()
    round_fixtures = fixtures[fixtures["stage"].astype(str).map(_stage_key).eq("round_of_32")].copy()
    rows: list[dict[str, Any]] = []
    for _, fixture in round_fixtures.sort_values("match_date").iterrows():
        home_team = _resolve_slot(str(fixture["bracket_slot_home"]), slot_mapping)
        away_team = _resolve_slot(str(fixture["bracket_slot_away"]), slot_mapping)
        prediction = predict_match(
            home_team=home_team,
            away_team=away_team,
            match_date=fixture["match_date"],
            neutral=bool(fixture["neutral"]),
            venue_country=str(fixture.get("country")) if pd.notna(fixture.get("country")) else None,
            tournament="World Cup",
            stage=str(fixture["stage"]),
        )
        rows.append(
            {
                "match_id": fixture["match_id"],
                "stage": fixture["stage"],
                "match_date": fixture["match_date"],
                "home_team": home_team,
                "away_team": away_team,
                "expected_goals_home": prediction["expected_goals_home"],
                "expected_goals_away": prediction["expected_goals_away"],
                "p_home_win_90": prediction["p_home_win"],
                "p_draw_90": prediction["p_draw"],
                "p_away_win_90": prediction["p_away_win"],
                "most_likely_score": prediction["most_likely_score"],
                "most_likely_score_probability": prediction["most_likely_score_probability"],
                "top_5_scorelines": prediction["top_5_scorelines"],
                "p_over_2_5_goals": prediction["p_over_2_5_goals"],
            }
        )
    return pd.DataFrame(rows)


def simulate_remaining_tournament(n_simulations: int = 10000, seed: int = 42) -> pd.DataFrame:
    ensure_project_dirs()
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")
    fixtures = _load_fixtures()
    group_fixtures = fixtures[fixtures["stage"].astype(str).str.contains("group", case=False, na=False)].copy()
    knockout_fixtures = fixtures[~fixtures.index.isin(group_fixtures.index)].copy()
    _validate_tournament_structure(
        {str(group): frame for group, frame in group_fixtures.groupby("group")},
        knockout_fixtures,
    )
    slot_mapping, standings, third_rank = build_actual_slot_mapping()
    standings.to_csv(PREDICTIONS_DIR / "actual_group_standings_2026.csv", index=False)
    third_rank.to_csv(PREDICTIONS_DIR / "actual_third_place_ranking_2026.csv", index=False)

    r32_predictions = build_round_of_32_predictions(slot_mapping)
    advancement_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_teams = set(standings["team"])
    qualified_teams = set(standings.loc[standings["qualified"], "team"])

    knockout_groups = knockout_fixtures.groupby(knockout_fixtures["stage"].astype(str).map(_stage_key))
    started_at = perf_counter()
    progress_interval = max(1, n_simulations // 20)
    LOGGER.info("Starting %s remaining-tournament simulations with seed %s.", n_simulations, seed)
    for simulation_index in range(n_simulations):
        simulation_rng = np.random.default_rng(seed + simulation_index)
        simulated_slots = dict(slot_mapping)
        for stage_label in ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]:
            if stage_label not in knockout_groups.groups:
                continue
            stage_results = simulate_knockout_round(
                knockout_groups.get_group(stage_label),
                simulated_slots,
                simulation_rng,
            )
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
                "Remaining simulation progress: %s/%s (%.0f%%), elapsed %.1fs, %.1f tournaments/s.",
                completed,
                n_simulations,
                completed * 100.0 / n_simulations,
                elapsed,
                completed / max(elapsed, 1e-9),
            )

    group_lookup = standings.set_index("team").to_dict("index")
    rows: list[dict[str, Any]] = []
    for team in sorted(all_teams):
        group_info = group_lookup[team]
        counts = advancement_counts[team]
        rows.append(
            {
                "team": team,
                "group": group_info["group"],
                "actual_group_position": int(group_info["position"]),
                "actual_group_points": int(group_info["points"]),
                "actual_group_goal_difference": int(group_info["goal_difference"]),
                "p_reach_r32": 1.0 if team in qualified_teams else 0.0,
                "p_reach_r16": float(counts.get("reached_r16", 0.0) / n_simulations),
                "p_reach_qf": float(counts.get("reached_qf", 0.0) / n_simulations),
                "p_reach_sf": float(counts.get("reached_sf", 0.0) / n_simulations),
                "p_reach_final": float(counts.get("reached_final", 0.0) / n_simulations),
                "p_champion": float(counts.get("champion", 0.0) / n_simulations),
            }
        )
    summary = pd.DataFrame(rows).sort_values("p_champion", ascending=False)
    summary.to_csv(PREDICTIONS_DIR / "remaining_tournament_simulation_summary.csv", index=False)
    r32_predictions = r32_predictions.merge(
        summary[["team", "p_reach_r16"]].rename(
            columns={"team": "home_team", "p_reach_r16": "p_home_advance"}
        ),
        on="home_team",
        how="left",
    ).merge(
        summary[["team", "p_reach_r16"]].rename(
            columns={"team": "away_team", "p_reach_r16": "p_away_advance"}
        ),
        on="away_team",
        how="left",
    )
    r32_predictions.to_csv(PREDICTIONS_DIR / "knockout_match_predictions_2026.csv", index=False)
    return summary
