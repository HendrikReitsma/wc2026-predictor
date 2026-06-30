from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import re
from datetime import datetime, timezone
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_group_stage_predictions import create_report
from fetch_worldcup_2026_group_results import (
    DEFAULT_SOURCE_URL as DEFAULT_GROUP_SOURCE_URL,
    OUTPUT_COLUMNS,
    _fetch_source_text,
    _score_pattern,
    _team_aliases,
    build_group_results,
)
from src.data.validate_data import load_team_mappings, normalize_team_name
from src.models.predict_match import predict_match
from src.simulation.simulate_knockout import _resolve_slot
from src.simulation.simulate_remaining_tournament import build_actual_slot_mapping
from src.simulation.simulate_tournament import _stage_key
from src.utils.paths import MANUAL_DATA_DIR, PREDICTIONS_DIR, PROJECT_ROOT, REPORTS_DIR


README_METRICS_START = "<!-- wc2026-metrics:start -->"
README_METRICS_END = "<!-- wc2026-metrics:end -->"

DEFAULT_KNOCKOUT_SOURCE_URLS = [
    "https://www.sbnation.com/soccer/1120771/world-cup-schedule-scores-round-32",
    "https://www.sbnation.com/soccer/1120850/world-cup-2026-round-of-16",
]

GROUP_RESULTS_PATH = MANUAL_DATA_DIR / "worldcup_2026_group_results.csv"
KNOCKOUT_RESULTS_PATH = MANUAL_DATA_DIR / "worldcup_2026_knockout_results.csv"
KNOCKOUT_PREDICTIONS_PATH = PREDICTIONS_DIR / "knockout_match_predictions_2026.csv"
KNOCKOUT_EVALUATION_PATH = PREDICTIONS_DIR / "knockout_prediction_evaluation.csv"
KNOCKOUT_METRICS_PATH = PREDICTIONS_DIR / "knockout_prediction_evaluation_metrics.json"
KNOCKOUT_REPORT_PATH = REPORTS_DIR / "worldcup_2026_knockout_model_performance.md"

KNOCKOUT_RESULT_COLUMNS = [
    *OUTPUT_COLUMNS,
    "stage",
    "winner",
    "loser",
    "decided_by",
    "penalty_home_score",
    "penalty_away_score",
]

KNOCKOUT_PREDICTION_COLUMNS = [
    "match_id",
    "stage",
    "match_date",
    "home_team",
    "away_team",
    "expected_goals_home",
    "expected_goals_away",
    "p_home_win_90",
    "p_draw_90",
    "p_away_win_90",
    "most_likely_score",
    "most_likely_score_probability",
    "top_5_scorelines",
    "p_over_2_5_goals",
    "p_home_advance",
    "p_away_advance",
]


def _validate_results_frame(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError("Results file is missing columns: " + ", ".join(missing_columns))
    if frame.empty:
        return
    if frame["match_id"].duplicated().any():
        duplicates = frame.loc[frame["match_id"].duplicated(), "match_id"].tolist()
        raise ValueError(f"Duplicate match_id values found after merge: {duplicates}")
    if frame[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Results contain missing home_score or away_score values.")
    if (frame[["home_score", "away_score"]] < 0).any().any():
        raise ValueError("Results contain negative score values.")


def merge_results(new_results: pd.DataFrame, output_path: Path, columns: list[str]) -> pd.DataFrame:
    new_results = new_results.reindex(columns=columns).copy()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_results], ignore_index=True, sort=False)
    else:
        combined = new_results

    combined = combined.reindex(columns=columns)
    combined = combined.drop_duplicates(subset=["match_id"], keep="last")
    combined = combined.sort_values("match_id").reset_index(drop=True)
    _validate_results_frame(combined, columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def _markdown_table(frame: pd.DataFrame, precision: int = 3) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.{precision}f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in row
            )
            + " |"
        )
    return "\n".join([header, separator, *rows])


def _actual_outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def _actual_outcome_label(outcome: int) -> str:
    return ["home", "draw", "away"][outcome]


def _ranked_probability_scores(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    encoded = np.eye(probabilities.shape[1])[np.asarray(y_true, dtype=int)]
    cumulative_error = np.cumsum(probabilities, axis=1)[:, :-1] - np.cumsum(encoded, axis=1)[:, :-1]
    return np.sum(cumulative_error**2, axis=1) / (probabilities.shape[1] - 1)


def _top_scores(raw: str) -> list[str]:
    try:
        return [str(entry["score"]) for entry in json.loads(raw)]
    except Exception:
        return []


def _find_score_line(
    text: str,
    home_team: str,
    away_team: str,
    mappings: dict[str, str],
) -> tuple[str, int, int]:
    for line in text.splitlines():
        for home_alias in _team_aliases(home_team, mappings):
            for away_alias in _team_aliases(away_team, mappings):
                match = _score_pattern(home_alias, away_alias).search(line)
                if match:
                    return line, int(match.group(1)), int(match.group(2))
                reverse_match = _score_pattern(away_alias, home_alias).search(line)
                if reverse_match:
                    return line, int(reverse_match.group(2)), int(reverse_match.group(1))
    raise ValueError(f"Could not find score for {home_team} vs {away_team}.")


def _penalty_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\s+wins\s+(\d+)[-–](\d+)\s+on\s+penalt", re.IGNORECASE)


def _extract_penalty_decision(
    line: str,
    home_team: str,
    away_team: str,
    mappings: dict[str, str],
) -> tuple[str, str, int | None, int | None]:
    for team, opponent, winner_is_home in [(home_team, away_team, True), (away_team, home_team, False)]:
        for alias in _team_aliases(team, mappings):
            match = _penalty_pattern(alias).search(line)
            if not match:
                continue
            winner_penalties = int(match.group(1))
            loser_penalties = int(match.group(2))
            if winner_is_home:
                return team, opponent, winner_penalties, loser_penalties
            return team, opponent, loser_penalties, winner_penalties
    raise ValueError(f"Could not find penalty winner in line: {line}")


def _extract_knockout_result(
    text: str,
    source_url: str,
    fixture: pd.Series,
    home_team: str,
    away_team: str,
    mappings: dict[str, str],
) -> dict[str, object]:
    line, home_score, away_score = _find_score_line(text, home_team, away_team, mappings)
    penalty_home_score: int | None = None
    penalty_away_score: int | None = None
    if home_score > away_score:
        winner = home_team
        loser = away_team
        decided_by = "score"
    elif home_score < away_score:
        winner = away_team
        loser = home_team
        decided_by = "score"
    else:
        winner, loser, penalty_home_score, penalty_away_score = _extract_penalty_decision(
            line, home_team, away_team, mappings
        )
        decided_by = "penalties"

    return {
        "match_id": int(fixture["match_id"]),
        "date": pd.Timestamp(fixture["match_date"]).date().isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": int(home_score),
        "away_score": int(away_score),
        "tournament": "FIFA World Cup",
        "city": fixture["city"],
        "country": fixture["country"],
        "neutral": bool(fixture["neutral"]),
        "group": "",
        "source_name": "SB Nation",
        "source_url": source_url,
        "stage": fixture["stage"],
        "winner": winner,
        "loser": loser,
        "decided_by": decided_by,
        "penalty_home_score": penalty_home_score,
        "penalty_away_score": penalty_away_score,
    }


def _home_advance_probability(prediction: dict[str, Any]) -> float:
    home_win = float(prediction["p_home_win"])
    draw = float(prediction["p_draw"])
    away_win = float(prediction["p_away_win"])
    penalty_edge = home_win / max(home_win + away_win, 1e-9)
    penalty_edge = float(np.clip(penalty_edge, 0.4, 0.6))
    return float(np.clip(home_win + draw * penalty_edge, 0.0, 1.0))


def _predict_knockout_fixture(fixture: pd.Series, home_team: str, away_team: str) -> dict[str, object]:
    prediction = predict_match(
        home_team=home_team,
        away_team=away_team,
        match_date=pd.Timestamp(fixture["match_date"]),
        neutral=bool(fixture["neutral"]),
        venue_country=str(fixture.get("country")) if pd.notna(fixture.get("country")) else None,
        tournament="World Cup",
        stage=str(fixture["stage"]),
    )
    p_home_advance = _home_advance_probability(prediction)
    return {
        "match_id": int(fixture["match_id"]),
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
        "p_home_advance": p_home_advance,
        "p_away_advance": 1.0 - p_home_advance,
    }


def _existing_predictions_by_match_id() -> dict[int, dict[str, object]]:
    if not KNOCKOUT_PREDICTIONS_PATH.exists():
        return {}
    frame = pd.read_csv(KNOCKOUT_PREDICTIONS_PATH)
    return {
        int(row["match_id"]): row.reindex(KNOCKOUT_PREDICTION_COLUMNS).to_dict()
        for _, row in frame.iterrows()
        if pd.notna(row.get("match_id"))
    }


def _existing_knockout_results_by_match_id() -> dict[int, dict[str, object]]:
    if not KNOCKOUT_RESULTS_PATH.exists():
        return {}
    frame = pd.read_csv(KNOCKOUT_RESULTS_PATH)
    return {
        int(row["match_id"]): row.reindex(KNOCKOUT_RESULT_COLUMNS).to_dict()
        for _, row in frame.iterrows()
        if pd.notna(row.get("match_id"))
    }


def build_knockout_updates(
    source_urls: list[str],
    skip_fetch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures = pd.read_csv(MANUAL_DATA_DIR / "worldcup_2026_fixtures.csv")
    knockout_fixtures = fixtures[~fixtures["stage"].astype(str).str.contains("group", case=False, na=False)].copy()
    knockout_fixtures["match_date"] = pd.to_datetime(knockout_fixtures["match_date"])
    knockout_fixtures = knockout_fixtures.sort_values(["match_date", "match_id"])

    mappings = load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv")
    source_texts = [] if skip_fetch else [(url, _fetch_source_text(url)) for url in source_urls]
    existing_predictions = _existing_predictions_by_match_id()
    existing_results = _existing_knockout_results_by_match_id()
    slot_mapping, _, _ = build_actual_slot_mapping()
    prediction_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_failure: Exception | None = None

    for _, fixture in knockout_fixtures.iterrows():
        try:
            home_team = _resolve_slot(str(fixture["bracket_slot_home"]), slot_mapping)
            away_team = _resolve_slot(str(fixture["bracket_slot_away"]), slot_mapping)
        except KeyError:
            continue

        match_id = int(fixture["match_id"])
        prediction_row = existing_predictions.get(match_id)
        if prediction_row is not None and (
            str(prediction_row.get("home_team", "")) != home_team
            or str(prediction_row.get("away_team", "")) != away_team
        ):
            prediction_row = None
        if prediction_row is None:
            try:
                prediction_row = _predict_knockout_fixture(fixture, home_team, away_team)
            except Exception as exc:
                prediction_failure = exc
        if prediction_row is not None:
            prediction_rows.append(dict(prediction_row))

        result_row = existing_results.get(match_id)
        if result_row is None:
            for source_url, text in source_texts:
                try:
                    result_row = _extract_knockout_result(
                        text,
                        source_url,
                        fixture,
                        normalize_team_name(home_team, mappings),
                        normalize_team_name(away_team, mappings),
                        mappings,
                    )
                    break
                except ValueError:
                    continue
        if result_row is None:
            continue

        result_rows.append(dict(result_row))
        winner = str(result_row["winner"])
        loser = str(result_row["loser"])
        slot_mapping[str(match_id)] = winner
        slot_mapping[f"W{match_id}"] = winner
        slot_mapping[f"WINNER{match_id}"] = winner
        slot_mapping[f"L{match_id}"] = loser
        slot_mapping[f"LOSER{match_id}"] = loser

    if prediction_failure and not prediction_rows:
        print(f"Knockout predictions skipped because models/features are missing: {prediction_failure}")

    predictions = pd.DataFrame(prediction_rows, columns=KNOCKOUT_PREDICTION_COLUMNS)
    results = pd.DataFrame(result_rows, columns=KNOCKOUT_RESULT_COLUMNS)
    return predictions, results


def save_knockout_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    existing = pd.read_csv(KNOCKOUT_PREDICTIONS_PATH) if KNOCKOUT_PREDICTIONS_PATH.exists() else pd.DataFrame()
    combined = pd.concat([existing, predictions], ignore_index=True, sort=False)
    combined = combined.reindex(columns=KNOCKOUT_PREDICTION_COLUMNS)
    combined = combined.drop_duplicates(subset=["match_id"], keep="last")
    combined = combined.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    combined.to_csv(KNOCKOUT_PREDICTIONS_PATH, index=False)
    return combined


def create_knockout_report() -> dict[str, Any]:
    if not KNOCKOUT_RESULTS_PATH.exists() or not KNOCKOUT_PREDICTIONS_PATH.exists():
        metrics = {"matches": 0}
        KNOCKOUT_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        KNOCKOUT_REPORT_PATH.write_text(
            "# Knockout Model Performance\n\nNo completed knockout matches with predictions are available yet.\n",
            encoding="utf-8",
        )
        return metrics

    results = pd.read_csv(KNOCKOUT_RESULTS_PATH)
    predictions = pd.read_csv(KNOCKOUT_PREDICTIONS_PATH)
    if results.empty or predictions.empty:
        metrics = {"matches": 0}
        KNOCKOUT_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        KNOCKOUT_REPORT_PATH.write_text(
            "# Knockout Model Performance\n\nNo completed knockout matches with predictions are available yet.\n",
            encoding="utf-8",
        )
        return metrics

    merged = predictions.merge(
        results,
        on="match_id",
        how="inner",
        suffixes=("_pred", "_actual"),
        validate="one_to_one",
    )
    if merged.empty:
        metrics = {"matches": 0}
        KNOCKOUT_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        KNOCKOUT_REPORT_PATH.write_text(
            "# Knockout Model Performance\n\nNo completed knockout matches matched the current prediction file.\n",
            encoding="utf-8",
        )
        return metrics

    probabilities = merged[["p_home_win_90", "p_draw_90", "p_away_win_90"]].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    actual_outcomes = np.asarray(
        [_actual_outcome(int(home), int(away)) for home, away in zip(merged["home_score"], merged["away_score"])],
        dtype=int,
    )
    predicted_outcomes = np.argmax(probabilities, axis=1)
    actual_probabilities = probabilities[np.arange(len(merged)), actual_outcomes]
    encoded = np.eye(3)[actual_outcomes]
    merged["actual_score"] = merged["home_score"].astype(int).astype(str) + "-" + merged["away_score"].astype(int).astype(str)
    merged["actual_outcome"] = [_actual_outcome_label(outcome) for outcome in actual_outcomes]
    merged["predicted_outcome"] = [_actual_outcome_label(outcome) for outcome in predicted_outcomes]
    merged["actual_outcome_probability"] = actual_probabilities
    merged["outcome_correct"] = predicted_outcomes == actual_outcomes
    merged["probability_rps"] = _ranked_probability_scores(actual_outcomes, probabilities)
    merged["actual_score_top1"] = merged["most_likely_score"].eq(merged["actual_score"])
    merged["actual_score_top5"] = [
        actual_score in _top_scores(raw)
        for actual_score, raw in zip(merged["actual_score"], merged["top_5_scorelines"])
    ]
    merged["predicted_advancer"] = np.where(
        merged["p_home_advance"].astype(float) >= merged["p_away_advance"].astype(float),
        merged["home_team_pred"],
        merged["away_team_pred"],
    )
    merged["advance_correct"] = merged["predicted_advancer"].eq(merged["winner"])
    merged["actual_advance_probability"] = np.where(
        merged["winner"].eq(merged["home_team_pred"]),
        merged["p_home_advance"].astype(float),
        merged["p_away_advance"].astype(float),
    )
    merged["total_goals"] = merged["home_score"].astype(int) + merged["away_score"].astype(int)
    merged["expected_total_goals"] = merged["expected_goals_home"] + merged["expected_goals_away"]
    merged["home_goal_error"] = (merged["home_score"] - merged["expected_goals_home"]).abs()
    merged["away_goal_error"] = (merged["away_score"] - merged["expected_goals_away"]).abs()
    merged.to_csv(KNOCKOUT_EVALUATION_PATH, index=False)

    metrics = {
        "matches": int(len(merged)),
        "outcome_accuracy": float(merged["outcome_correct"].mean()),
        "outcome_correct_count": int(merged["outcome_correct"].sum()),
        "outcome_log_loss": float(-np.mean(np.log(actual_probabilities))),
        "outcome_brier_score": float(np.mean(np.sum((encoded - probabilities) ** 2, axis=1))),
        "probability_rps": float(merged["probability_rps"].mean()),
        "average_actual_outcome_probability": float(np.mean(actual_probabilities)),
        "advance_accuracy": float(merged["advance_correct"].mean()),
        "advance_correct_count": int(merged["advance_correct"].sum()),
        "average_actual_advance_probability": float(merged["actual_advance_probability"].mean()),
        "top1_score_accuracy": float(merged["actual_score_top1"].mean()),
        "top1_score_hit_count": int(merged["actual_score_top1"].sum()),
        "top5_score_hit_rate": float(merged["actual_score_top5"].mean()),
        "top5_score_hit_count": int(merged["actual_score_top5"].sum()),
        "expected_total_goals": float(merged["expected_total_goals"].sum()),
        "actual_total_goals": int(merged["total_goals"].sum()),
        "penalty_decisions": int(merged["decided_by"].eq("penalties").sum()),
    }
    KNOCKOUT_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    display = merged[
        [
            "match_id",
            "stage_pred",
            "home_team_pred",
            "away_team_pred",
            "actual_score",
            "winner",
            "decided_by",
            "predicted_advancer",
            "advance_correct",
            "p_home_advance",
            "p_away_advance",
            "actual_outcome",
            "predicted_outcome",
            "probability_rps",
        ]
    ].rename(
        columns={
            "stage_pred": "stage",
            "home_team_pred": "home_team",
            "away_team_pred": "away_team",
        }
    )
    lines = [
        "# Knockout Model Performance",
        "",
        f"- Evaluated knockout matches: **{metrics['matches']}**.",
        f"- Advance accuracy: **{metrics['advance_accuracy']:.1%}** "
        f"({metrics['advance_correct_count']} / {metrics['matches']}).",
        f"- Match-score outcome accuracy: **{metrics['outcome_accuracy']:.1%}** "
        f"({metrics['outcome_correct_count']} / {metrics['matches']}).",
        f"- Outcome RPS: **{metrics['probability_rps']:.3f}**.",
        f"- Outcome log loss: **{metrics['outcome_log_loss']:.3f}**.",
        f"- Exact-score hit rate: **{metrics['top1_score_accuracy']:.1%}** "
        f"({metrics['top1_score_hit_count']} / {metrics['matches']}).",
        f"- Top-5 scoreline hit rate: **{metrics['top5_score_hit_rate']:.1%}** "
        f"({metrics['top5_score_hit_count']} / {metrics['matches']}).",
        f"- Total goals: expected **{metrics['expected_total_goals']:.1f}**, actual **{metrics['actual_total_goals']}**.",
        f"- Penalty decisions: **{metrics['penalty_decisions']}**.",
        "",
        "## Match Details",
        "",
        _markdown_table(display),
        "",
        "Note: knockout score outcomes use the published score before penalties. Advancement accuracy uses the published advancing team.",
    ]
    KNOCKOUT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return metrics


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _combined_metric(
    group_metrics: dict[str, object],
    knockout_metrics: dict[str, object] | None,
    key: str,
) -> float:
    group_matches = int(group_metrics["matches"])
    knockout_matches = int((knockout_metrics or {}).get("matches", 0))
    total_matches = group_matches + knockout_matches
    if total_matches == 0:
        return 0.0
    group_total = float(group_metrics[key]) * group_matches
    knockout_total = (
        float(knockout_metrics[key]) * knockout_matches
        if knockout_metrics and knockout_matches > 0 and key in knockout_metrics
        else 0.0
    )
    return (group_total + knockout_total) / total_matches


def render_readme_metrics(
    group_metrics: dict[str, object],
    knockout_metrics: dict[str, object] | None = None,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    group_matches = int(group_metrics["matches"])
    knockout_matches = int((knockout_metrics or {}).get("matches", 0))
    matches = group_matches + knockout_matches
    outcome_correct = int(group_metrics["outcome_correct_count"]) + int(
        (knockout_metrics or {}).get("outcome_correct_count", 0)
    )
    top1_hits = int(group_metrics["top1_score_hit_count"]) + int(
        (knockout_metrics or {}).get("top1_score_hit_count", 0)
    )
    top5_hits = int(group_metrics["top5_score_hit_count"]) + int(
        (knockout_metrics or {}).get("top5_score_hit_count", 0)
    )
    expected_total_goals = float(group_metrics["expected_total_goals"]) + float(
        (knockout_metrics or {}).get("expected_total_goals", 0.0)
    )
    actual_total_goals = int(group_metrics["actual_total_goals"]) + int(
        (knockout_metrics or {}).get("actual_total_goals", 0)
    )
    rounded_matches = group_matches
    rows = [
        f"_Last updated by `python scripts/update_results.py` at {generated_at}._",
        "",
        "| Metric | Current value |",
        "| --- | ---: |",
        f"| Matches evaluated | {matches} |",
        f"| Outcome accuracy | {_format_percent(outcome_correct / max(matches, 1))} |",
        f"| Correct outcomes / total matches | {outcome_correct} / {matches} |",
        f"| Log loss | {_combined_metric(group_metrics, knockout_metrics, 'outcome_log_loss'):.3f} |",
        f"| Brier score | {_combined_metric(group_metrics, knockout_metrics, 'outcome_brier_score'):.3f} |",
        f"| Ranked Probability Score | {_combined_metric(group_metrics, knockout_metrics, 'probability_rps'):.3f} |",
        (
            "| Avg probability on actual result | "
            f"{_format_percent(_combined_metric(group_metrics, knockout_metrics, 'average_actual_outcome_probability'))} |"
        ),
        (
            "| Exact score hit rate | "
            f"{_format_percent(top1_hits / max(matches, 1))} "
            f"({top1_hits} / {matches}) |"
        ),
        (
            "| Top-5 scoreline hit rate | "
            f"{_format_percent(top5_hits / max(matches, 1))} "
            f"({top5_hits} / {matches}) |"
        ),
        (
            "| Total goals expected vs actual | "
            f"{expected_total_goals:.1f} vs {actual_total_goals} |"
        ),
        (
            "| Rounded-xG outcome accuracy | "
            f"{_format_percent(float(group_metrics['rounded_outcome_accuracy']))} "
            f"({int(group_metrics['rounded_outcome_correct_count'])} / {rounded_matches} group matches) |"
        ),
    ]
    if knockout_metrics and int(knockout_metrics.get("matches", 0)) > 0:
        rows.extend(
            [
                (
                    "| Knockout advance accuracy | "
                    f"{_format_percent(float(knockout_metrics['advance_accuracy']))} "
                    f"({int(knockout_metrics['advance_correct_count'])} / {knockout_matches}) |"
                ),
            ]
        )
    return "\n".join(rows)


def update_readme_metrics(
    group_metrics: dict[str, object],
    knockout_metrics: dict[str, object] | None = None,
    readme_path: Path = PROJECT_ROOT / "README.md",
) -> None:
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")
    readme = readme_path.read_text(encoding="utf-8")
    if README_METRICS_START not in readme or README_METRICS_END not in readme:
        raise ValueError(
            f"README must contain {README_METRICS_START!r} and {README_METRICS_END!r} markers."
        )
    before, rest = readme.split(README_METRICS_START, 1)
    _, after = rest.split(README_METRICS_END, 1)
    replacement = (
        f"{README_METRICS_START}\n"
        f"{render_readme_metrics(group_metrics, knockout_metrics)}\n"
        f"{README_METRICS_END}"
    )
    readme_path.write_text(before + replacement + after, encoding="utf-8")


def run_update(
    group_source_url: str,
    knockout_source_urls: list[str],
    skip_fetch: bool,
    update_readme: bool,
) -> dict[str, dict[str, object]]:
    if skip_fetch:
        if not GROUP_RESULTS_PATH.exists():
            raise FileNotFoundError(f"Cannot skip fetch because {GROUP_RESULTS_PATH} does not exist.")
        print(f"Using existing group results at {GROUP_RESULTS_PATH}.")
    else:
        print(f"Fetching group-stage results from {group_source_url}")
        fetched_group = build_group_results(group_source_url)
        merged_group = merge_results(fetched_group, GROUP_RESULTS_PATH, OUTPUT_COLUMNS)
        print(f"Saved {len(merged_group)} unique group-stage results to {GROUP_RESULTS_PATH}.")

    knockout_predictions, knockout_results = build_knockout_updates(
        knockout_source_urls,
        skip_fetch=skip_fetch,
    )
    saved_predictions = save_knockout_predictions(knockout_predictions)
    if not knockout_results.empty:
        saved_results = merge_results(knockout_results, KNOCKOUT_RESULTS_PATH, KNOCKOUT_RESULT_COLUMNS)
        print(f"Saved {len(saved_results)} unique knockout results to {KNOCKOUT_RESULTS_PATH}.")
    elif KNOCKOUT_RESULTS_PATH.exists():
        print(f"No new knockout results found; using existing results at {KNOCKOUT_RESULTS_PATH}.")
    else:
        print("No completed knockout results found from the configured sources.")
    if not saved_predictions.empty:
        print(f"Saved {len(saved_predictions)} resolved knockout prediction rows to {KNOCKOUT_PREDICTIONS_PATH}.")

    group_metrics = create_report()
    knockout_metrics = create_knockout_report()
    if update_readme:
        update_readme_metrics(group_metrics, knockout_metrics)
        print("Updated README metrics block.")

    print(
        "Group evaluation complete: "
        f"{int(group_metrics['outcome_correct_count'])}/{int(group_metrics['matches'])} correct, "
        f"RPS {float(group_metrics['probability_rps']):.3f}, "
        f"log loss {float(group_metrics['outcome_log_loss']):.3f}."
    )
    if int(knockout_metrics.get("matches", 0)) > 0:
        print(
            "Knockout evaluation complete: "
            f"{int(knockout_metrics['advance_correct_count'])}/{int(knockout_metrics['matches'])} advance picks correct, "
            f"RPS {float(knockout_metrics['probability_rps']):.3f}."
        )
    return {"group": group_metrics, "knockout": knockout_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch latest World Cup 2026 results, re-evaluate predictions, and update README metrics."
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_GROUP_SOURCE_URL,
        help="Source page containing completed group-stage scores.",
    )
    parser.add_argument(
        "--knockout-source-url",
        action="append",
        default=None,
        help="Additional source page containing completed knockout scores. Can be repeated.",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Use existing local result CSVs.")
    parser.add_argument("--no-readme", action="store_true", help="Do not update README.md metrics.")
    args = parser.parse_args()
    knockout_source_urls = args.knockout_source_url or DEFAULT_KNOCKOUT_SOURCE_URLS
    try:
        run_update(
            group_source_url=args.source_url,
            knockout_source_urls=knockout_source_urls,
            skip_fetch=args.skip_fetch,
            update_readme=not args.no_readme,
        )
    except Exception as exc:
        raise SystemExit(f"Update failed: {exc}") from exc


if __name__ == "__main__":
    main()
