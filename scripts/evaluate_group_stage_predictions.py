from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from math import log
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import MANUAL_DATA_DIR, PREDICTIONS_DIR, REPORTS_DIR


def _markdown_table(frame: pd.DataFrame, precision: int = 3) -> str:
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


def _round_goals(values: pd.Series) -> pd.Series:
    return np.floor(values.astype(float) + 0.5).astype(int)


def _top_scores(raw: str) -> list[str]:
    try:
        return [str(entry["score"]) for entry in json.loads(raw)]
    except Exception:
        return []


def _evaluate_rows() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_DIR / "match_predictions_2026.csv")
    actual = pd.read_csv(MANUAL_DATA_DIR / "worldcup_2026_group_results.csv")
    group_predictions = predictions[predictions["stage"].astype(str).str.contains("group", case=False, na=False)].copy()
    merged = group_predictions.merge(
        actual[
            [
                "match_id",
                "home_score",
                "away_score",
                "source_name",
                "source_url",
            ]
        ],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 72:
        raise ValueError(f"Expected 72 evaluated group matches; found {len(merged)}.")
    probabilities = merged[["p_home_win", "p_draw", "p_away_win"]].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    actual_outcomes = np.asarray(
        [_actual_outcome(int(home), int(away)) for home, away in zip(merged["home_score"], merged["away_score"])],
        dtype=int,
    )
    predicted_outcomes = np.argmax(probabilities, axis=1)
    actual_probabilities = probabilities[np.arange(len(merged)), actual_outcomes]
    encoded = np.eye(3)[actual_outcomes]
    probability_rps = _ranked_probability_scores(actual_outcomes, probabilities)
    merged["actual_score"] = merged["home_score"].astype(int).astype(str) + "-" + merged["away_score"].astype(int).astype(str)
    merged["actual_outcome"] = [_actual_outcome_label(outcome) for outcome in actual_outcomes]
    merged["predicted_outcome"] = [_actual_outcome_label(outcome) for outcome in predicted_outcomes]
    merged["predicted_outcome_probability"] = probabilities.max(axis=1)
    merged["actual_outcome_probability"] = actual_probabilities
    merged["probability_rps"] = probability_rps
    merged["outcome_correct"] = predicted_outcomes == actual_outcomes
    merged["rounded_pred_home_goals"] = _round_goals(merged["expected_goals_home"])
    merged["rounded_pred_away_goals"] = _round_goals(merged["expected_goals_away"])
    merged["rounded_pred_score"] = (
        merged["rounded_pred_home_goals"].astype(str) + "-" + merged["rounded_pred_away_goals"].astype(str)
    )
    rounded_outcomes = np.asarray(
        [
            _actual_outcome(int(home), int(away))
            for home, away in zip(merged["rounded_pred_home_goals"], merged["rounded_pred_away_goals"])
        ],
        dtype=int,
    )
    merged["rounded_pred_outcome"] = [_actual_outcome_label(outcome) for outcome in rounded_outcomes]
    merged["rounded_outcome_correct"] = rounded_outcomes == actual_outcomes
    merged["rounded_score_exact"] = merged["rounded_pred_score"].eq(merged["actual_score"])
    rounded_probabilities = np.eye(3)[rounded_outcomes]
    merged["rounded_rps"] = _ranked_probability_scores(actual_outcomes, rounded_probabilities)
    merged["actual_score_top1"] = merged["most_likely_score"].eq(merged["actual_score"])
    merged["actual_score_top5"] = [
        actual_score in _top_scores(raw)
        for actual_score, raw in zip(merged["actual_score"], merged["top_5_scorelines"])
    ]
    merged["home_goal_error"] = (merged["home_score"] - merged["expected_goals_home"]).abs()
    merged["away_goal_error"] = (merged["away_score"] - merged["expected_goals_away"]).abs()
    merged["total_goals"] = merged["home_score"] + merged["away_score"]
    merged["expected_total_goals"] = merged["expected_goals_home"] + merged["expected_goals_away"]
    merged["total_goal_error"] = (merged["total_goals"] - merged["expected_total_goals"]).abs()
    merged["goal_difference"] = merged["home_score"] - merged["away_score"]
    merged["expected_goal_difference"] = merged["expected_goals_home"] - merged["expected_goals_away"]
    merged["goal_difference_error"] = (merged["goal_difference"] - merged["expected_goal_difference"]).abs()
    merged["actual_over_2_5"] = merged["total_goals"].ge(3)
    merged["over_2_5_brier"] = (merged["p_over_2_5_goals"] - merged["actual_over_2_5"].astype(float)) ** 2
    merged.attrs["log_loss"] = float(-np.mean(np.log(actual_probabilities)))
    merged.attrs["brier_score"] = float(np.mean(np.sum((encoded - probabilities) ** 2, axis=1)))
    merged.attrs["accuracy"] = float(np.mean(merged["outcome_correct"]))
    merged.attrs["average_actual_probability"] = float(np.mean(actual_probabilities))
    return merged


def _confidence_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"]
    working = evaluation.copy()
    working["confidence_bin"] = pd.cut(
        working["predicted_outcome_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return (
        working.groupby("confidence_bin", observed=False)
        .agg(
            matches=("match_id", "size"),
            mean_confidence=("predicted_outcome_probability", "mean"),
            accuracy=("outcome_correct", "mean"),
        )
        .reset_index()
        .dropna(subset=["mean_confidence"])
    )


def _group_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    return (
        evaluation.groupby("group", as_index=False)
        .agg(
            matches=("match_id", "size"),
            outcome_accuracy=("outcome_correct", "mean"),
            top5_score_rate=("actual_score_top5", "mean"),
            mean_actual_prob=("actual_outcome_probability", "mean"),
            expected_goals=("expected_total_goals", "sum"),
            actual_goals=("total_goals", "sum"),
        )
        .sort_values("group")
    )


def _rounded_outcome_mix(evaluation: pd.DataFrame) -> pd.DataFrame:
    actual_counts = (
        evaluation["actual_outcome"]
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="actual_count")
    )
    rounded_counts = (
        evaluation["rounded_pred_outcome"]
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="rounded_pred_count")
    )
    probability_counts = (
        evaluation["predicted_outcome"]
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="probability_argmax_count")
    )
    output = (
        actual_counts.merge(rounded_counts, on="outcome", how="outer")
        .merge(probability_counts, on="outcome", how="outer")
        .fillna(0)
        .sort_values("outcome")
    )
    for column in ["actual_count", "rounded_pred_count", "probability_argmax_count"]:
        output[column] = output[column].astype(int)
    return output


def _rounded_group_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    return (
        evaluation.groupby("group", as_index=False)
        .agg(
            matches=("match_id", "size"),
            rounded_outcome_accuracy=("rounded_outcome_correct", "mean"),
            rounded_exact_score_rate=("rounded_score_exact", "mean"),
            mean_home_goal_error=("home_goal_error", "mean"),
            mean_away_goal_error=("away_goal_error", "mean"),
            total_goal_error=("total_goal_error", "mean"),
        )
        .sort_values("group")
    )


def create_report() -> dict[str, Any]:
    evaluation = _evaluate_rows()
    evaluation.to_csv(PREDICTIONS_DIR / "group_stage_prediction_evaluation.csv", index=False)

    exact_hits = evaluation[evaluation["actual_score_top1"]][
        ["match_id", "home_team", "away_team", "actual_score", "most_likely_score"]
    ]
    biggest_hits = evaluation.nlargest(8, "actual_outcome_probability")[
        [
            "match_id",
            "group",
            "home_team",
            "away_team",
            "actual_score",
            "actual_outcome",
            "actual_outcome_probability",
            "predicted_outcome",
        ]
    ]
    biggest_misses = evaluation.nsmallest(8, "actual_outcome_probability")[
        [
            "match_id",
            "group",
            "home_team",
            "away_team",
            "actual_score",
            "actual_outcome",
            "actual_outcome_probability",
            "predicted_outcome",
        ]
    ]
    rounded_exact_hits = evaluation[evaluation["rounded_score_exact"]][
        ["match_id", "home_team", "away_team", "actual_score", "rounded_pred_score"]
    ]
    rounded_outcome_misses = evaluation[~evaluation["rounded_outcome_correct"]][
        [
            "match_id",
            "group",
            "home_team",
            "away_team",
            "expected_goals_home",
            "expected_goals_away",
            "rounded_pred_score",
            "actual_score",
            "rounded_pred_outcome",
            "actual_outcome",
        ]
    ].sort_values(["group", "match_id"])
    metrics = {
        "matches": int(len(evaluation)),
        "prediction_cutoff_date": str(evaluation["prediction_cutoff_date"].iloc[0]),
        "outcome_log_loss": float(evaluation.attrs["log_loss"]),
        "outcome_brier_score": float(evaluation.attrs["brier_score"]),
        "outcome_accuracy": float(evaluation.attrs["accuracy"]),
        "probability_rps": float(evaluation["probability_rps"].mean()),
        "average_actual_outcome_probability": float(evaluation.attrs["average_actual_probability"]),
        "top1_score_accuracy": float(evaluation["actual_score_top1"].mean()),
        "top5_score_hit_rate": float(evaluation["actual_score_top5"].mean()),
        "home_goal_mae": float(evaluation["home_goal_error"].mean()),
        "away_goal_mae": float(evaluation["away_goal_error"].mean()),
        "mean_goal_mae": float(pd.concat([evaluation["home_goal_error"], evaluation["away_goal_error"]]).mean()),
        "total_goals_mae": float(evaluation["total_goal_error"].mean()),
        "goal_difference_mae": float(evaluation["goal_difference_error"].mean()),
        "expected_total_goals": float(evaluation["expected_total_goals"].sum()),
        "actual_total_goals": int(evaluation["total_goals"].sum()),
        "over_2_5_brier": float(evaluation["over_2_5_brier"].mean()),
        "over_2_5_accuracy_at_0_5": float(
            (evaluation["p_over_2_5_goals"].ge(0.5) == evaluation["actual_over_2_5"]).mean()
        ),
        "outcome_correct_count": int(evaluation["outcome_correct"].sum()),
        "top1_score_hit_count": int(evaluation["actual_score_top1"].sum()),
        "top5_score_hit_count": int(evaluation["actual_score_top5"].sum()),
        "rounded_outcome_accuracy": float(evaluation["rounded_outcome_correct"].mean()),
        "rounded_rps": float(evaluation["rounded_rps"].mean()),
        "rounded_outcome_correct_count": int(evaluation["rounded_outcome_correct"].sum()),
        "rounded_exact_score_accuracy": float(evaluation["rounded_score_exact"].mean()),
        "rounded_exact_score_hit_count": int(evaluation["rounded_score_exact"].sum()),
        "rounded_predicted_draws": int(evaluation["rounded_pred_outcome"].eq("draw").sum()),
        "actual_draws": int(evaluation["actual_outcome"].eq("draw").sum()),
        "rounded_home_goal_exact_rate": float((evaluation["rounded_pred_home_goals"] == evaluation["home_score"]).mean()),
        "rounded_away_goal_exact_rate": float((evaluation["rounded_pred_away_goals"] == evaluation["away_score"]).mean()),
    }
    outcome_counts = _rounded_outcome_mix(evaluation)
    (PREDICTIONS_DIR / "group_stage_prediction_evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Group Stage Model Performance",
        "",
        f"- Evaluated matches: **{metrics['matches']}**.",
        f"- Prediction cutoff used: **{metrics['prediction_cutoff_date']}**.",
        "- Actual result source: `data/manual/worldcup_2026_group_results.csv`.",
        "",
        "## Headline Metrics",
        "",
        f"- Outcome accuracy: **{metrics['outcome_accuracy']:.1%}**.",
        f"  - Correct outcome picks: **{metrics['outcome_correct_count']} / {metrics['matches']}**.",
        f"- Outcome log loss: **{metrics['outcome_log_loss']:.3f}**.",
        f"- Outcome Brier score: **{metrics['outcome_brier_score']:.3f}**.",
        f"- Outcome RPS from model probabilities: **{metrics['probability_rps']:.3f}**.",
        f"- Average probability assigned to the actual outcome: **{metrics['average_actual_outcome_probability']:.1%}**.",
        f"- Exact-score top-1 hit rate: **{metrics['top1_score_accuracy']:.1%}** "
        f"({metrics['top1_score_hit_count']} / {metrics['matches']}).",
        f"- Exact-score top-5 hit rate: **{metrics['top5_score_hit_rate']:.1%}** "
        f"({metrics['top5_score_hit_count']} / {metrics['matches']}).",
        f"- Mean per-team goal MAE: **{metrics['mean_goal_mae']:.3f}**.",
        f"- Goal-difference MAE: **{metrics['goal_difference_mae']:.3f}**.",
        f"- Total goals: expected **{metrics['expected_total_goals']:.1f}**, actual **{metrics['actual_total_goals']}**.",
        f"- Over 2.5 goals Brier score: **{metrics['over_2_5_brier']:.3f}**; 0.5-threshold accuracy **{metrics['over_2_5_accuracy_at_0_5']:.1%}**.",
        "",
        "## Rounded Expected-Goals Score Evaluation",
        "",
        "This section evaluates the prediction as if the entered score was made by rounding expected home and away goals with spreadsheet-style rounding: `floor(x + 0.5)`. For example, `1.41-0.89` becomes `1-1`, and `1.94-0.45` becomes `2-0`.",
        "",
        f"- Rounded-score outcome accuracy: **{metrics['rounded_outcome_accuracy']:.1%}** "
        f"({metrics['rounded_outcome_correct_count']} / {metrics['matches']}).",
        f"- Rounded deterministic RPS: **{metrics['rounded_rps']:.3f}**.",
        f"- Rounded exact-score accuracy: **{metrics['rounded_exact_score_accuracy']:.1%}** "
        f"({metrics['rounded_exact_score_hit_count']} / {metrics['matches']}).",
        f"- Rounded predicted draws: **{metrics['rounded_predicted_draws']}**; actual draws: **{metrics['actual_draws']}**.",
        f"- Rounded home-goal exact rate: **{metrics['rounded_home_goal_exact_rate']:.1%}**.",
        f"- Rounded away-goal exact rate: **{metrics['rounded_away_goal_exact_rate']:.1%}**.",
        "",
        "## Outcome Mix",
        "",
        _markdown_table(outcome_counts),
        "",
        "## Calibration By Favorite Confidence",
        "",
        _markdown_table(_confidence_table(evaluation)),
        "",
        "## Performance By Group",
        "",
        _markdown_table(_group_table(evaluation)),
        "",
        "## Rounded Expected-Goals Performance By Group",
        "",
        _markdown_table(_rounded_group_table(evaluation)),
        "",
        "## Exact Score Hits",
        "",
        _markdown_table(exact_hits) if not exact_hits.empty else "No exact top-1 score hits.",
        "",
        "## Highest Probability Correct Outcomes",
        "",
        _markdown_table(biggest_hits),
        "",
        "## Lowest Probability Actual Outcomes",
        "",
        _markdown_table(biggest_misses),
        "",
        "## Rounded Exact Score Hits",
        "",
        _markdown_table(rounded_exact_hits) if not rounded_exact_hits.empty else "No rounded exact-score hits.",
        "",
        "## Rounded Outcome Misses",
        "",
        _markdown_table(rounded_outcome_misses),
        "",
        "## Short Read",
        "",
        "For this group stage, the rounded expected-goals score was slightly better for outcome picking than simply taking the highest-probability W/D/L class: 47 correct outcomes versus 46. The reason is intuitive for a score-prediction pool: rounding created 16 predicted draws, while the probability argmax predicted no draws at all. That helped in a group stage with 20 actual draws.",
        "",
        "The tradeoff is exact scores. The model's modal score hit 9 exact results, while rounded expected goals hit 6. The expected-goals numbers were informative for broad outcomes and aggregate scoring, but turning them into one rounded score still loses a lot of distributional information.",
    ]
    (REPORTS_DIR / "worldcup_2026_group_stage_model_performance.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    metrics = create_report()
    print(
        "Saved group-stage evaluation. "
        f"Accuracy {metrics['outcome_accuracy']:.1%}, "
        f"log loss {metrics['outcome_log_loss']:.3f}, "
        f"top-5 score hit rate {metrics['top5_score_hit_rate']:.1%}."
    )


if __name__ == "__main__":
    main()
