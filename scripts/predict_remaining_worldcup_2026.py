from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd

from src.simulation.simulate_remaining_tournament import simulate_remaining_tournament
from src.utils.config import config_value
from src.utils.paths import MANUAL_DATA_DIR, PREDICTIONS_DIR, REPORTS_DIR


SOURCE_RESULTS_URL = "https://www.sbnation.com/soccer/1117513/world-cup-schedule-2026-how-to-watch-every-match-scores-and-more"
SOURCE_STANDINGS_URL = "https://www.foxsports.com/soccer/fifa-world-cup/standings"
SOURCE_FIFA_GERMANY_URL = "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/teams/germany/fixtures"


def _markdown_table(frame: pd.DataFrame, precision: int = 3) -> str:
    selected = frame.copy()
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(selected.columns)) + " |"
    rows = []
    for row in selected.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.{precision}f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in row
            )
            + " |"
        )
    return "\n".join([header, separator, *rows])


def _audit_group_results() -> dict[str, object]:
    results = pd.read_csv(MANUAL_DATA_DIR / "worldcup_2026_group_results.csv")
    fixtures = pd.read_csv(MANUAL_DATA_DIR / "worldcup_2026_fixtures.csv")
    group_fixtures = fixtures[fixtures["stage"].astype(str).str.contains("group", case=False, na=False)]
    merged = results.merge(
        group_fixtures[["match_id", "home_team", "away_team", "group", "city", "country", "neutral"]],
        on="match_id",
        suffixes=("_result", "_fixture"),
        how="outer",
        indicator=True,
    )
    field_mismatches = {}
    for column in ["home_team", "away_team", "group", "city", "country", "neutral"]:
        left = merged[f"{column}_result"].astype(str)
        right = merged[f"{column}_fixture"].astype(str)
        field_mismatches[column] = int((left != right).sum())
    germany = results[results["home_team"].eq("Germany") & results["away_team"].eq("Curaçao")].iloc[0]
    audit = {
        "result_rows": int(len(results)),
        "unique_result_match_ids": int(results["match_id"].nunique()),
        "fixture_group_rows": int(len(group_fixtures)),
        "merge_left_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "merge_right_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "field_mismatches": field_mismatches,
        "missing_score_rows": int(results[["home_score", "away_score"]].isna().any(axis=1).sum()),
        "germany_curacao": {
            "match_id": int(germany["match_id"]),
            "date": str(germany["date"]),
            "score": f"Germany {int(germany['home_score'])}-{int(germany['away_score'])} Curaçao",
        },
        "source_results_url": SOURCE_RESULTS_URL,
        "source_standings_url": SOURCE_STANDINGS_URL,
        "source_fifa_germany_url": SOURCE_FIFA_GERMANY_URL,
    }
    (PREDICTIONS_DIR / "actual_group_results_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    return audit


def create_remaining_report(cutoff_date: str, n_simulations: int) -> None:
    audit = _audit_group_results()
    standings = pd.read_csv(PREDICTIONS_DIR / "actual_group_standings_2026.csv")
    third_rank = pd.read_csv(PREDICTIONS_DIR / "actual_third_place_ranking_2026.csv")
    r32 = pd.read_csv(PREDICTIONS_DIR / "knockout_match_predictions_2026.csv")
    summary = pd.read_csv(PREDICTIONS_DIR / "remaining_tournament_simulation_summary.csv")
    group_e = standings[standings["group"].eq("E")][
        ["position", "team", "points", "goals_for", "goals_against", "goal_difference", "qualified"]
    ]
    lines = [
        "# World Cup 2026 Remaining Tournament Prediction",
        "",
        f"- Cutoff after group stage: **{cutoff_date}**.",
        f"- Simulations: **{n_simulations:,}**.",
        "- Scope: actual group-stage results are locked in; no knockout results are included.",
        f"- Group-score source: {SOURCE_RESULTS_URL}",
        f"- Standings cross-check source: {SOURCE_STANDINGS_URL}",
        f"- Germany fixture cross-check source: {SOURCE_FIFA_GERMANY_URL}",
        "",
        "## Results Audit",
        "",
        f"- Group result rows: **{audit['result_rows']}**.",
        f"- Unique result match IDs: **{audit['unique_result_match_ids']}**.",
        f"- Fixture group rows: **{audit['fixture_group_rows']}**.",
        f"- Missing score rows: **{audit['missing_score_rows']}**.",
        f"- Fixture merge misses: left-only **{audit['merge_left_only_rows']}**, right-only **{audit['merge_right_only_rows']}**.",
        f"- Fixture field mismatches: **{audit['field_mismatches']}**.",
        f"- Example check: **{audit['germany_curacao']['score']}**, match {audit['germany_curacao']['match_id']} on {audit['germany_curacao']['date']}.",
        "",
        "## Group E Check",
        "",
        _markdown_table(group_e),
        "",
        "## Qualified Third-Place Teams",
        "",
        _markdown_table(
            third_rank[third_rank["qualified"]][
                ["third_place_rank", "team", "group", "points", "goal_difference", "goals_for"]
            ]
        ),
        "",
        "## Round Of 32 Match Predictions",
        "",
        _markdown_table(
            r32[
                [
                    "match_id",
                    "home_team",
                    "away_team",
                    "expected_goals_home",
                    "expected_goals_away",
                    "p_home_win_90",
                    "p_draw_90",
                    "p_away_win_90",
                    "p_home_advance",
                    "p_away_advance",
                    "most_likely_score",
                ]
            ],
        ),
        "",
        "## Top Champion Probabilities",
        "",
        _markdown_table(
            summary.nlargest(15, "p_champion")[
                ["team", "p_reach_r16", "p_reach_qf", "p_reach_sf", "p_reach_final", "p_champion"]
            ],
        ),
    ]
    (REPORTS_DIR / "worldcup_2026_remaining_prediction_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the remaining World Cup 2026 tournament after group stage.")
    parser.add_argument("--cutoff-date", required=True, help="Cutoff date after completed group stage.")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10000,
        help="Monte Carlo simulation count.",
    )
    args = parser.parse_args()
    max_simulations = int(config_value("simulation", "max_n_simulations", default=100000))
    if args.n_simulations > max_simulations:
        raise ValueError(f"n-simulations cannot exceed configured maximum of {max_simulations}.")
    simulate_remaining_tournament(n_simulations=args.n_simulations, seed=42)
    create_remaining_report(args.cutoff_date, args.n_simulations)


if __name__ == "__main__":
    main()
