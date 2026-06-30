from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import setup_logging
from src.utils.paths import MANUAL_DATA_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)

TEAM_NAME_OVERRIDES = {
    "Democratic Republic of the Congo": "DR Congo",
}

STAGE_NAMES = {
    "group": "Group Stage",
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarter-final",
    "sf": "Semi-final",
    "third": "Third Place",
    "final": "Final",
}


def _canonical_team(name: str) -> str:
    return TEAM_NAME_OVERRIDES.get(name, name)


def _slot_from_label(label: str, opponent_label: str = "") -> str:
    winner_group = re.fullmatch(r"Winner Group ([A-L])", label)
    if winner_group:
        return f"1{winner_group.group(1)}"
    runner_up = re.fullmatch(r"Runner-up Group ([A-L])", label)
    if runner_up:
        return f"2{runner_up.group(1)}"
    winner_match = re.fullmatch(r"Winner Match (\d+)", label)
    if winner_match:
        return f"W{winner_match.group(1)}"
    loser_match = re.fullmatch(r"Loser Match (\d+)", label)
    if loser_match:
        return f"L{loser_match.group(1)}"
    if label.startswith("3rd Group "):
        winner = re.fullmatch(r"Winner Group ([A-L])", opponent_label)
        if not winner:
            raise ValueError(f"Could not identify group winner paired with conditional slot {label!r}.")
        return f"THIRD_FOR_1{winner.group(1)}"
    raise ValueError(f"Unsupported bracket label: {label!r}")


def build_fixture_frame(
    matches: list[dict[str, Any]],
    teams: list[dict[str, Any]],
    stadiums: list[dict[str, Any]],
) -> pd.DataFrame:
    team_names = {str(team["id"]): _canonical_team(str(team["name_en"])) for team in teams}
    stadium_map = {str(stadium["id"]): stadium for stadium in stadiums}
    rows: list[dict[str, Any]] = []

    for match in matches:
        match_type = str(match["type"]).lower()
        stadium = stadium_map[str(match["stadium_id"])]
        is_group = match_type == "group"
        home_team = team_names.get(str(match["home_team_id"]), "") if is_group else ""
        away_team = team_names.get(str(match["away_team_id"]), "") if is_group else ""
        country = str(stadium["country_en"])
        neutral = country not in {home_team, away_team}

        home_label = str(match.get("home_team_label", ""))
        away_label = str(match.get("away_team_label", ""))
        rows.append(
            {
                "match_id": str(match["id"]),
                "stage": STAGE_NAMES[match_type],
                "group": str(match["group"]) if is_group else "",
                "match_date": pd.to_datetime(match["local_date"], format="%m/%d/%Y %H:%M").isoformat(),
                "venue": str(stadium["name_en"]),
                "city": str(stadium["city_en"]),
                "country": country,
                "home_team": home_team,
                "away_team": away_team,
                "neutral": neutral,
                "bracket_slot_home": "" if is_group else _slot_from_label(home_label, away_label),
                "bracket_slot_away": "" if is_group else _slot_from_label(away_label, home_label),
            }
        )

    return pd.DataFrame(rows).sort_values("match_id", key=lambda series: series.astype(int)).reset_index(drop=True)


def build_third_place_allocations(wikipedia_parse_json: dict[str, Any]) -> pd.DataFrame:
    html = wikipedia_parse_json["parse"]["text"]["*"]
    source = pd.read_html(StringIO(html))[0]
    group_columns = list(source.columns[1:13])
    opponent_columns = ["1A vs", "1B vs", "1D vs", "1E vs", "1G vs", "1I vs", "1K vs", "1L vs"]
    rows: list[dict[str, str]] = []

    for _, row in source.iterrows():
        qualifying_groups = "".join(sorted(str(row[column]) for column in group_columns if pd.notna(row[column])))
        output = {"qualifying_groups": qualifying_groups}
        output.update({column.replace(" vs", ""): str(row[column]) for column in opponent_columns})
        rows.append(output)

    result = pd.DataFrame(rows)
    if len(result) != 495 or result["qualifying_groups"].nunique() != 495:
        raise ValueError("Expected exactly 495 unique third-place allocation combinations.")
    return result


def populate_worldcup_2026_fixtures(
    matches_path: Path | None = None,
    teams_path: Path | None = None,
    stadiums_path: Path | None = None,
    third_place_source_path: Path | None = None,
) -> tuple[Path, Path]:
    ensure_project_dirs()
    matches_path = matches_path or MANUAL_DATA_DIR / "source_football.matches.json"
    teams_path = teams_path or MANUAL_DATA_DIR / "source_football.teams.json"
    stadiums_path = stadiums_path or MANUAL_DATA_DIR / "source_football.stadiums.json"
    third_place_source_path = third_place_source_path or MANUAL_DATA_DIR / "third_place_table_wikipedia.json"

    for path in [matches_path, teams_path, stadiums_path, third_place_source_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing fixture source file: {path}")

    matches = json.loads(matches_path.read_text(encoding="utf-8"))
    teams = json.loads(teams_path.read_text(encoding="utf-8"))
    stadiums = json.loads(stadiums_path.read_text(encoding="utf-8"))
    third_place_source = json.loads(third_place_source_path.read_text(encoding="utf-8"))

    fixtures = build_fixture_frame(matches, teams, stadiums)
    if len(fixtures) != 104:
        raise ValueError(f"Expected 104 World Cup fixtures, found {len(fixtures)}.")
    fixtures_path = MANUAL_DATA_DIR / "worldcup_2026_fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)

    allocations = build_third_place_allocations(third_place_source)
    allocations_path = MANUAL_DATA_DIR / "worldcup_2026_third_place_allocations.csv"
    allocations.to_csv(allocations_path, index=False)

    LOGGER.info("Saved %s fixtures to %s", len(fixtures), fixtures_path)
    LOGGER.info("Saved %s third-place allocations to %s", len(allocations), allocations_path)
    return fixtures_path, allocations_path

