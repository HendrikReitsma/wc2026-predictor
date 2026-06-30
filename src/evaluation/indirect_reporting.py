from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils.paths import MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def create_indirect_model_report() -> None:
    required = [
        PREDICTIONS_DIR / "indirect_model_backtest_results.csv",
        PREDICTIONS_DIR / "indirect_model_backtest_summary.csv",
        PREDICTIONS_DIR / "team_strength_trend_scores_2026.csv",
        PREDICTIONS_DIR / "tournament_readiness_scores_2026.csv",
        PREDICTIONS_DIR / "team_strength_adjustments_2026.csv",
        PREDICTIONS_DIR / "team_only_vs_indirect_match_comparison.csv",
        MODELS_DIR / "indirect_model_selection.json",
    ]
    if any(not path.exists() for path in required):
        return
    results = pd.read_csv(required[0])
    summary = pd.read_csv(required[1])
    trend = pd.read_csv(required[2])
    readiness = pd.read_csv(required[3])
    adjustments = pd.read_csv(required[4])
    matches = pd.read_csv(required[5])
    selection = json.loads(required[6].read_text(encoding="utf-8"))
    snapshots = pd.read_csv(PROCESSED_DATA_DIR / "team_strength_snapshots.csv")
    readiness_dataset = pd.read_csv(PROCESSED_DATA_DIR / "tournament_readiness_dataset.csv")
    matches["largest_probability_change"] = matches[
        ["delta_p_home_win", "delta_p_away_win"]
    ].abs().max(axis=1)
    match_predictions_path = PREDICTIONS_DIR / "match_predictions_2026.csv"
    group_impact = pd.DataFrame()
    if match_predictions_path.exists():
        match_predictions = pd.read_csv(match_predictions_path)
        if "group" in match_predictions:
            group_impact = matches.merge(match_predictions[["match_id", "group"]], on="match_id", how="left")
            group_impact = (
                group_impact.groupby("group", as_index=False)
                .agg(
                    average_probability_change=("largest_probability_change", "mean"),
                    maximum_probability_change=("largest_probability_change", "max"),
                )
                .sort_values("average_probability_change", ascending=False)
            )
    baseline = summary[summary["model_variant"].eq("baseline")].iloc[0]
    comparison = summary[summary["model_variant"].eq(selection["comparison_variant"])].iloc[0]
    champion_comparison_path = PREDICTIONS_DIR / "tournament_simulation_summary_indirect_comparison.csv"
    champion_lines = [
        "The final model retained the baseline. A separate indirect-challenger tournament simulation was not available."
    ]
    if champion_comparison_path.exists() and (PREDICTIONS_DIR / "tournament_simulation_summary.csv").exists():
        base_champions = pd.read_csv(PREDICTIONS_DIR / "tournament_simulation_summary.csv")
        indirect_champions = pd.read_csv(champion_comparison_path)
        champion = base_champions[["team", "p_champion"]].merge(
            indirect_champions[["team", "p_champion"]], on="team", suffixes=("_baseline", "_indirect")
        )
        champion["delta_p_champion"] = champion["p_champion_indirect"] - champion["p_champion_baseline"]
        champion_lines = [
            _table(
                champion.reindex(champion["delta_p_champion"].abs().sort_values(ascending=False).index).head(15),
                ["team", "p_champion_baseline", "p_champion_indirect", "delta_p_champion"],
            )
        ]
    lines = [
        "# Indirect Model Report",
        "",
        "## A. Executive Summary",
        "",
        f"- Selected final variant: **{selection['selected_variant']}**.",
        f"- Best conventional indirect challenger: **{selection['comparison_variant']}**.",
        f"- Decision: {selection['selection_reason']}",
        f"- Baseline mean log loss: **{baseline['avg_log_loss']:.4f}**; challenger: **{comparison['avg_log_loss']:.4f}**.",
        f"- Baseline Brier score: **{baseline['avg_brier_score']:.4f}**; challenger: **{comparison['avg_brier_score']:.4f}**.",
        "- Indirect continuous targets use the selected match pipeline's existing probability calibration; no separate indirect probability calibrator was fitted.",
        "- The indirect-only sanity check had lower log loss, but worse Brier score, calibration, and top-5 scoreline coverage; it was not eligible to replace the coherent match-and-score pipeline.",
        "- No betting odds or player-level data are used.",
        "",
        "## B. Team Strength Trend Model",
        "",
        f"The trend dataset contains **{len(snapshots):,}** team-before-match snapshots. Features use only matches before each snapshot. The primary target is next-five-match Elo change; its target completion date must be before a training cutoff.",
        "",
        "### Strongest Positive 2026 Trend Scores",
        "",
        _table(trend.nlargest(10, "expected_future_elo_delta"), ["team", "base_elo", "expected_future_elo_delta", "expected_future_performance_above_expectation", "trend_adjustment"]),
        "",
        "### Strongest Negative 2026 Trend Scores",
        "",
        _table(trend.nsmallest(10, "expected_future_elo_delta"), ["team", "base_elo", "expected_future_elo_delta", "expected_future_performance_above_expectation", "trend_adjustment"]),
        "",
        "The smallest trend correction was the best conventional indirect challenger, but it slightly worsened both log loss and Brier score, so it was not selected.",
        "",
        "## C. Tournament Readiness Model",
        "",
        f"The readiness dataset contains **{len(readiness_dataset):,}** team-before-World-Cup rows. The target is tournament goal-difference performance above pre-match Elo expectation. Features stop before tournament start; targets complete at tournament end.",
        "",
        "### Strongest Positive 2026 Readiness Scores",
        "",
        _table(readiness.nlargest(10, "tournament_readiness_score"), ["team", "base_elo", "tournament_readiness_score", "expected_group_points_adjustment", "overperformance_probability"]),
        "",
        "### Strongest Negative 2026 Readiness Scores",
        "",
        _table(readiness.nsmallest(10, "tournament_readiness_score"), ["team", "base_elo", "tournament_readiness_score", "expected_group_points_adjustment", "overperformance_probability"]),
        "",
        "Readiness corrections did not improve the primary outcome metrics and generally reduced scoreline quality.",
        "The required `expected_group_points_adjustment` output is a tournament-points-residual proxy, not a directly trained group-stage target, because reliable historical group-stage labels are unavailable.",
        "",
        "## D. Baseline Versus Indirect Comparison",
        "",
        _table(summary, ["model_variant", "trend_weight", "readiness_weight", "avg_log_loss", "avg_brier_score", "avg_calibration_error", "avg_scoreline_top_5_hit_rate", "stability_score"]),
        "",
        "### World Cup By World Cup",
        "",
        _table(results, ["model_variant", "worldcup_year", "log_loss", "brier_score", "calibration_error", "scoreline_top_5_hit_rate"]),
        "",
        "Historical source data does not provide reliable group-stage labels, so group-points MAE and group-qualification accuracy are intentionally left unavailable.",
        "These indirect experiments use exact `FIFA World Cup` rows and train through each tournament start. Their absolute scores are therefore not directly comparable with earlier reports that use older frozen training cutoffs.",
        "",
        "## E. 2026 Prediction Impact",
        "",
        "Because the baseline was selected, final production adjustments are zero. The tables below show the best indirect challenger for comparison.",
        "",
        "### Most Changed Matches",
        "",
        _table(matches.nlargest(15, "largest_probability_change"), ["home_team", "away_team", "baseline_p_home_win", "indirect_p_home_win", "baseline_p_draw", "indirect_p_draw", "baseline_p_away_win", "indirect_p_away_win", "largest_probability_change"]),
        "",
        "### Most Affected Groups",
        "",
        _table(group_impact, ["group", "average_probability_change", "maximum_probability_change"]) if not group_impact.empty else "Group impact unavailable.",
        "",
        "### Champion Probabilities Before Versus Challenger",
        "",
        *champion_lines,
        "",
        "The rejected-challenger tournament comparison used 10,000 simulations while the selected baseline used 100,000. Small champion-probability deltas therefore include Monte Carlo sampling noise and should not be interpreted as precise effects.",
        "",
        "## F. Limitations",
        "",
        "- Tournament readiness has a small effective sample: one row per team per historical World Cup.",
        "- Qualification difficulty and confederation strength are imperfectly controlled.",
        "- Sparse intercontinental matches can bias internal Elo comparisons.",
        "- Historical World Cup stage labels are incomplete, preventing honest group-stage target evaluation.",
        "- The indirect models may overfit historical tournament patterns.",
        "- World Cup 2026 group and bracket assumptions remain uncertain.",
        "- No player data or betting data are used.",
        "",
        "These are probabilistic comparisons, not certainties.",
    ]
    (REPORTS_DIR / "indirect_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
