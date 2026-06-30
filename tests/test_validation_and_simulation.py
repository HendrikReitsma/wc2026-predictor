from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate_data import parse_bool, validate_fixtures_dataframe
from src.simulation.simulate_group_stage import GroupTableEntry, rank_group_table
from src.simulation.simulate_knockout import _resolve_slot
from src.simulation.simulate_remaining_tournament import build_actual_group_tables, load_actual_group_results
from src.simulation.simulate_tournament import _validate_tournament_structure
from src.data.populate_2026_fixtures import build_third_place_allocations
from src.utils.paths import MANUAL_DATA_DIR


def test_parse_bool_handles_csv_values() -> None:
    assert parse_bool("False") is False
    assert parse_bool("TRUE") is True
    assert parse_bool(0) is False
    assert parse_bool(1) is True


def test_empty_fixture_file_is_rejected() -> None:
    columns = [
        "match_id",
        "stage",
        "group",
        "match_date",
        "venue",
        "city",
        "country",
        "home_team",
        "away_team",
        "neutral",
        "bracket_slot_home",
        "bracket_slot_away",
    ]
    result = validate_fixtures_dataframe(pd.DataFrame(columns=columns), {})
    assert not result.is_valid
    assert any("no fixtures" in issue for issue in result.issues)


def test_group_ranking_and_bracket_aliases() -> None:
    table = {
        "A": GroupTableEntry("A", points=6, goals_for=4, goals_against=1, goal_difference=3, random_tiebreaker=0.1),
        "B": GroupTableEntry("B", points=4, goals_for=3, goals_against=2, goal_difference=1, random_tiebreaker=0.2),
        "C": GroupTableEntry("C", points=4, goals_for=2, goals_against=1, goal_difference=1, random_tiebreaker=0.3),
        "D": GroupTableEntry("D", points=0, goals_for=0, goals_against=5, goal_difference=-5, random_tiebreaker=0.4),
    }
    ranked = rank_group_table(table)
    slots = {"A1": "A", "W73": "B"}

    assert [entry.team for entry in ranked] == ["A", "B", "C", "D"]
    assert _resolve_slot("1a", slots) == "A"
    assert _resolve_slot("Winner 73", slots) == "B"


def test_incomplete_tournament_structure_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "exactly 12 groups"):
        _validate_tournament_structure({}, pd.DataFrame(columns=["stage"]))


def test_third_place_allocation_source_has_all_combinations() -> None:
    import json

    source = json.loads((MANUAL_DATA_DIR / "third_place_table_wikipedia.json").read_text(encoding="utf-8"))
    allocations = build_third_place_allocations(source)
    assert len(allocations) == 495
    assert allocations["qualifying_groups"].nunique() == 495


def test_worldcup_2026_group_results_align_with_fixtures() -> None:
    results = load_actual_group_results()
    fixtures = pd.read_csv(MANUAL_DATA_DIR / "worldcup_2026_fixtures.csv")
    group_fixtures = fixtures[fixtures["stage"].str.contains("group", case=False, na=False)]
    merged = results.merge(
        group_fixtures[["match_id", "home_team", "away_team", "group"]],
        on="match_id",
        suffixes=("_result", "_fixture"),
        how="outer",
        indicator=True,
    )
    assert len(results) == 72
    assert results["match_id"].nunique() == 72
    assert not results[["home_score", "away_score"]].isna().any().any()
    assert merged["_merge"].eq("both").all()
    assert (merged["home_team_result"] == merged["home_team_fixture"]).all()
    assert (merged["away_team_result"] == merged["away_team_fixture"]).all()
    assert (merged["group_result"] == merged["group_fixture"]).all()

    germany_curacao = results[(results["home_team"] == "Germany") & (results["away_team"] == "Curaçao")].iloc[0]
    assert int(germany_curacao["home_score"]) == 7
    assert int(germany_curacao["away_score"]) == 1


def test_worldcup_2026_actual_group_e_standings() -> None:
    _, standings, third_rank = build_actual_group_tables()
    group_e = standings[standings["group"].eq("E")].sort_values("position")
    assert group_e["team"].tolist() == ["Germany", "Ivory Coast", "Ecuador", "Curaçao"]
    assert group_e["points"].tolist() == [6, 6, 4, 1]
    assert group_e["goal_difference"].tolist() == [6, 2, 0, -8]
    assert set(third_rank[third_rank["qualified"]]["group"]) == {"B", "D", "E", "F", "I", "J", "K", "L"}
