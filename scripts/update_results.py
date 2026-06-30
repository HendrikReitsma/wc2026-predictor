from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluate_group_stage_predictions import create_report
from fetch_worldcup_2026_group_results import DEFAULT_SOURCE_URL, OUTPUT_COLUMNS, build_group_results
from src.utils.paths import MANUAL_DATA_DIR, PROJECT_ROOT


README_METRICS_START = "<!-- wc2026-metrics:start -->"
README_METRICS_END = "<!-- wc2026-metrics:end -->"


def _validate_results_frame(frame: pd.DataFrame) -> None:
    missing_columns = sorted(set(OUTPUT_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError("Results file is missing columns: " + ", ".join(missing_columns))
    if frame["match_id"].duplicated().any():
        duplicates = frame.loc[frame["match_id"].duplicated(), "match_id"].tolist()
        raise ValueError(f"Duplicate match_id values found after merge: {duplicates}")
    if frame[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Results contain missing home_score or away_score values.")
    if (frame[["home_score", "away_score"]] < 0).any().any():
        raise ValueError("Results contain negative score values.")


def merge_group_results(new_results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Merge newly fetched group results with existing rows without duplicating matches."""
    new_results = new_results.reindex(columns=OUTPUT_COLUMNS).copy()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_results], ignore_index=True, sort=False)
    else:
        combined = new_results

    combined = combined.reindex(columns=OUTPUT_COLUMNS)
    combined = combined.drop_duplicates(subset=["match_id"], keep="last")
    combined = combined.sort_values("match_id").reset_index(drop=True)
    _validate_results_frame(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def render_readme_metrics(metrics: dict[str, object]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    matches = int(metrics["matches"])
    rows = [
        f"_Last updated by `python scripts/update_results.py` at {generated_at}._",
        "",
        "| Metric | Current value |",
        "| --- | ---: |",
        f"| Outcome accuracy | {_format_percent(float(metrics['outcome_accuracy']))} |",
        f"| Correct outcomes / total matches | {int(metrics['outcome_correct_count'])} / {matches} |",
        f"| Log loss | {float(metrics['outcome_log_loss']):.3f} |",
        f"| Brier score | {float(metrics['outcome_brier_score']):.3f} |",
        f"| Ranked Probability Score | {float(metrics['probability_rps']):.3f} |",
        f"| Avg probability on actual result | {_format_percent(float(metrics['average_actual_outcome_probability']))} |",
        (
            "| Exact score hit rate | "
            f"{_format_percent(float(metrics['top1_score_accuracy']))} "
            f"({int(metrics['top1_score_hit_count'])} / {matches}) |"
        ),
        (
            "| Top-5 scoreline hit rate | "
            f"{_format_percent(float(metrics['top5_score_hit_rate']))} "
            f"({int(metrics['top5_score_hit_count'])} / {matches}) |"
        ),
        (
            "| Total goals expected vs actual | "
            f"{float(metrics['expected_total_goals']):.1f} vs {int(metrics['actual_total_goals'])} |"
        ),
        (
            "| Rounded-xG outcome accuracy | "
            f"{_format_percent(float(metrics['rounded_outcome_accuracy']))} "
            f"({int(metrics['rounded_outcome_correct_count'])} / {matches}) |"
        ),
    ]
    return "\n".join(rows)


def update_readme_metrics(metrics: dict[str, object], readme_path: Path = PROJECT_ROOT / "README.md") -> None:
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
        f"{render_readme_metrics(metrics)}\n"
        f"{README_METRICS_END}"
    )
    readme_path.write_text(before + replacement + after, encoding="utf-8")


def run_update(source_url: str, skip_fetch: bool, update_readme: bool) -> dict[str, object]:
    output_path = MANUAL_DATA_DIR / "worldcup_2026_group_results.csv"
    if skip_fetch:
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot skip fetch because {output_path} does not exist.")
        print(f"Using existing group results at {output_path}.")
    else:
        print(f"Fetching group-stage results from {source_url}")
        fetched = build_group_results(source_url)
        merged = merge_group_results(fetched, output_path)
        print(f"Saved {len(merged)} unique group-stage results to {output_path}.")

    metrics = create_report()
    if update_readme:
        update_readme_metrics(metrics)
        print("Updated README metrics block.")
    print(
        "Evaluation complete: "
        f"{int(metrics['outcome_correct_count'])}/{int(metrics['matches'])} correct, "
        f"RPS {float(metrics['probability_rps']):.3f}, "
        f"log loss {float(metrics['outcome_log_loss']):.3f}."
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch latest World Cup 2026 group results, re-evaluate predictions, and update README metrics."
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="Source page containing completed scores.")
    parser.add_argument("--skip-fetch", action="store_true", help="Use the existing local results CSV.")
    parser.add_argument("--no-readme", action="store_true", help="Do not update README.md metrics.")
    args = parser.parse_args()
    try:
        run_update(args.source_url, skip_fetch=args.skip_fetch, update_readme=not args.no_readme)
    except Exception as exc:
        raise SystemExit(f"Update failed: {exc}") from exc


if __name__ == "__main__":
    main()
