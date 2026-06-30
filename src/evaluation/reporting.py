from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_dataset import load_fixtures_frame
from src.data.validate_data import load_team_mappings
from src.models.train_goal_model import (
    GOAL_ATTACK_DEFENCE_FEATURES,
    GOAL_BASIC_FEATURES,
)
from src.models.train_outcome_model import OPPONENT_ADJUSTED_FEATURES, RAW_FORM_FEATURES
from src.utils.config import config_value
from src.utils.paths import MANUAL_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.evaluation.indirect_reporting import create_indirect_model_report


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, precision: int = 3) -> str:
    selected = frame[columns].copy() if columns else frame.copy()
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(selected.columns)) + " |"
    rows = [
        "| "
        + " | ".join(
            f"{value:.{precision}f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        )
        + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _load_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def audit_worldcup_2026_fixtures(known_teams: set[str]) -> dict[str, object]:
    fixtures = load_fixtures_frame(
        load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv"),
        known_teams=known_teams,
    )
    group = fixtures[fixtures["stage"].str.contains("group", case=False, na=False)]
    knockout = fixtures[~fixtures.index.isin(group.index)]
    host_matches = group[
        group.apply(lambda row: str(row["country"]) in {str(row["home_team"]), str(row["away_team"])}, axis=1)
    ]
    suspicious_host_neutral = host_matches[host_matches["neutral"].astype(bool)]
    audit = {
        "fixture_rows": int(len(fixtures)),
        "group_matches": int(len(group)),
        "knockout_matches": int(len(knockout)),
        "recognized_group_teams": int(len(set(group["home_team"]).union(group["away_team"]))),
        "unresolved_group_teams": int(
            group[["home_team", "away_team"]].fillna("").apply(lambda column: column.str.strip().eq("")).sum().sum()
        ),
        "unresolved_knockout_slots": int(
            knockout[["bracket_slot_home", "bracket_slot_away"]]
            .fillna("")
            .apply(lambda column: column.str.strip().eq(""))
            .sum()
            .sum()
        ),
        "unparseable_dates": int(pd.to_datetime(fixtures["match_date"], errors="coerce").isna().sum()),
        "suspicious_host_neutral_flags": int(len(suspicious_host_neutral)),
    }
    (REPORTS_DIR / "fixture_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def create_prediction_summary(cutoff_date: str, n_simulations: int | None = None) -> None:
    simulation = pd.read_csv(PREDICTIONS_DIR / "tournament_simulation_summary.csv")
    matches = pd.read_csv(PREDICTIONS_DIR / "match_predictions_2026.csv")
    selection = json.loads((MODELS_DIR / "model_selection.json").read_text(encoding="utf-8"))
    target_selection = selection.get("target_experiment_recommendation", {})
    strategy = selection.get("training_strategy_recommendation", {})
    indirect_path = MODELS_DIR / "indirect_model_selection.json"
    indirect_selection = json.loads(indirect_path.read_text(encoding="utf-8")) if indirect_path.exists() else {}
    selected_prediction_model = target_selection.get("recommended_model_name", selection["model_name"])
    feature_state = json.loads((PREDICTIONS_DIR.parent / "processed" / "feature_state.json").read_text(encoding="utf-8"))
    known_teams = set(feature_state["elo_state"]["ratings"])
    fixture_audit = audit_worldcup_2026_fixtures(known_teams)

    top_champions = simulation.nlargest(20, "p_champion")[["team", "p_champion"]]
    group_winners = simulation.nlargest(10, "p_group_1st")[["team", "p_group_1st"]]
    finalists = simulation.nlargest(10, "p_reach_final")[["team", "p_reach_final"]]
    simulation["champion_uncertainty"] = simulation["p_champion"] * (1.0 - simulation["p_champion"])
    uncertainty = simulation.nlargest(10, "champion_uncertainty")[["team", "p_champion", "p_reach_final"]]
    matches["outcome_spread"] = (
        matches[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
        - matches[["p_home_win", "p_draw", "p_away_win"]].min(axis=1)
    )
    even_matches = matches.nsmallest(10, "outcome_spread")[
        [
            "home_team",
            "away_team",
            "expected_goals_home",
            "expected_goals_away",
            "p_home_win",
            "p_draw",
            "p_away_win",
        ]
    ]
    matches["away_edge"] = matches["p_away_win"] - matches["p_home_win"]
    upsets = matches.nlargest(10, "away_edge")[
        [
            "home_team",
            "away_team",
            "expected_goals_home",
            "expected_goals_away",
            "p_home_win",
            "p_draw",
            "p_away_win",
        ]
    ]

    lines = [
        "# World Cup 2026 Prediction Summary",
        "",
        f"- Training cutoff: **{cutoff_date}**",
        f"- Selected model: **{selected_prediction_model}**",
        f"- Selected model feature set: **{selection.get('selected_model_feature_set', selection['feature_group'])}**",
        f"- Best classifier challenger feature group: **{selection['feature_group']}**",
        f"- Selected weighting: **{selection['weighting_scheme']}**",
        f"- Production time decay: **{selection.get('production_time_decay_half_life_years') or 'disabled'}**",
        f"- Rating update: **{strategy.get('rating_model', 'standard_elo')}**, K scale **{strategy.get('elo_k_scale', 1.0)}**",
        f"- Draw correction / calibration: **{strategy.get('draw_correction', 'none')} / {strategy.get('calibration_method', 'uncalibrated')}**",
        f"- Indirect correction: **{indirect_selection.get('selected_variant', 'not evaluated')}**; best challenger **{indirect_selection.get('comparison_variant', 'not evaluated')}**",
        "- Data sources: martj42 international results; FIFA 2026 official schedule and regulations.",
        "",
        "## Top 20 Champion Probabilities",
        "",
        _markdown_table(top_champions),
        "",
        "## Top 10 Group Winners",
        "",
        _markdown_table(group_winners),
        "",
        "## Top 10 Most Likely Finalists",
        "",
        _markdown_table(finalists),
        "",
        "## Biggest Uncertainty Teams",
        "",
        _markdown_table(uncertainty),
        "",
        "## Most Evenly Matched Group Games",
        "",
        _markdown_table(even_matches),
        "",
        "## Most Likely Away-Side Upsets",
        "",
        _markdown_table(upsets),
        "",
        "## Fixture Audit",
        "",
        *[f"- {key}: {value}" for key, value in fixture_audit.items()],
        "",
        "## Known Limitations",
        "",
        "- Historical results do not include reliable stage labels for all competitions.",
        "- No player availability, injuries, betting odds, or squad-strength inputs are used.",
        "- Monte Carlo probabilities have sampling error and are not claims of certainty.",
    ]
    (REPORTS_DIR / "worldcup_2026_prediction_summary.md").write_text("\n".join(lines), encoding="utf-8")
    create_csv_summaries(simulation, matches)
    create_prediction_report(cutoff_date, n_simulations, simulation, matches, selection, fixture_audit)
    create_model_card(cutoff_date, selection)
    create_figures(simulation, matches)
    create_indirect_model_report()


def create_csv_summaries(simulation: pd.DataFrame, matches: pd.DataFrame) -> None:
    top_champions = simulation.sort_values("p_champion", ascending=False)[
        ["team", "group", "p_champion", "p_reach_final", "p_reach_sf", "p_reach_qf"]
    ]
    top_champions.to_csv(PREDICTIONS_DIR / "top_champion_probabilities.csv", index=False)
    simulation.sort_values(["group", "p_group_1st"], ascending=[True, False]).to_csv(
        PREDICTIONS_DIR / "group_prediction_summary.csv", index=False
    )
    matches[
        [
            "match_id",
            "group",
            "home_team",
            "away_team",
            "expected_goals_home",
            "expected_goals_away",
            "most_likely_score",
            "most_likely_score_probability",
            "top_5_scorelines",
        ]
    ].to_csv(PREDICTIONS_DIR / "most_likely_scores.csv", index=False)
    scored = matches.copy()
    scored["outcome_spread"] = scored[["p_home_win", "p_draw", "p_away_win"]].max(axis=1) - scored[
        ["p_home_win", "p_draw", "p_away_win"]
    ].min(axis=1)
    scored["away_edge"] = scored["p_away_win"] - scored["p_home_win"]
    scored["expected_total_goals"] = scored["expected_goals_home"] + scored["expected_goals_away"]
    summary_columns = [
        "match_id",
        "group",
        "home_team",
        "away_team",
        "expected_goals_home",
        "expected_goals_away",
        "most_likely_score",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "expected_total_goals",
    ]
    scored.nsmallest(20, "outcome_spread")[summary_columns].to_csv(
        PREDICTIONS_DIR / "most_even_matches.csv", index=False
    )
    scored.nlargest(20, "away_edge")[summary_columns].to_csv(
        PREDICTIONS_DIR / "most_likely_upsets.csv", index=False
    )
    scored.nlargest(20, "expected_total_goals")[summary_columns].to_csv(
        PREDICTIONS_DIR / "highest_scoring_matches.csv", index=False
    )
    scored.nsmallest(20, "expected_total_goals")[summary_columns].to_csv(
        PREDICTIONS_DIR / "lowest_scoring_matches.csv", index=False
    )


def create_prediction_report(
    cutoff_date: str,
    n_simulations: int | None,
    simulation: pd.DataFrame,
    matches: pd.DataFrame,
    selection: dict[str, object],
    fixture_audit: dict[str, object],
) -> None:
    features = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv")
    target_selection = selection.get("target_experiment_recommendation", {})
    strategy = selection.get("training_strategy_recommendation", {})
    indirect_path = MODELS_DIR / "indirect_model_selection.json"
    indirect_selection = json.loads(indirect_path.read_text(encoding="utf-8")) if indirect_path.exists() else {}
    selected_prediction_model = target_selection.get("recommended_model_name", selection["model_name"])
    target_summary = _load_optional_csv(PREDICTIONS_DIR / "target_experiment_summary.csv")
    features["date"] = pd.to_datetime(features["date"])
    minimum_year = int(selection.get("minimum_training_year", config_value("modeling", "minimum_training_year", default=1990)))
    training = features[(features["date"].dt.year >= minimum_year) & (features["date"] <= pd.Timestamp(cutoff_date))]
    model_comparison = _load_optional_csv(PREDICTIONS_DIR / "backtest_model_comparison.csv")
    scoreline_comparison = _load_optional_csv(PREDICTIONS_DIR / "backtest_scoreline_comparison.csv")
    ablations = _load_optional_csv(PREDICTIONS_DIR / "feature_ablation_results.csv")
    training_strategies = _load_optional_csv(PREDICTIONS_DIR / "training_strategy_comparison.csv")
    calibration_comparison = _load_optional_csv(PREDICTIONS_DIR / "calibration_comparison.csv")
    rating_comparison = _load_optional_csv(PREDICTIONS_DIR / "rating_model_comparison.csv")
    goal_summary = (
        scoreline_comparison.groupby("model_name", as_index=False)[
            ["log_loss", "brier_score", "top_5_scoreline_hit_rate", "mean_goal_mae", "scoreline_log_loss"]
        ]
        .mean()
        .sort_values("log_loss")
        if not scoreline_comparison.empty
        else pd.DataFrame()
    )
    scored = matches.copy()
    scored["expected_total_goals"] = scored["expected_goals_home"] + scored["expected_goals_away"]
    scored["outcome_spread"] = scored[["p_home_win", "p_draw", "p_away_win"]].max(axis=1) - scored[
        ["p_home_win", "p_draw", "p_away_win"]
    ].min(axis=1)
    scored["favorite_probability"] = scored[["p_home_win", "p_away_win"]].max(axis=1)
    scored["away_edge"] = scored["p_away_win"] - scored["p_home_win"]
    match_columns = [
        "group", "home_team", "away_team", "expected_goals_home", "expected_goals_away", "most_likely_score",
        "p_home_win", "p_draw", "p_away_win", "expected_total_goals",
    ]
    group_uncertainty = (
        simulation.groupby("group")
        .apply(lambda group: pd.Series({"group_winner_entropy": float(-(group["p_group_1st"].clip(lower=1e-12) * np.log(group["p_group_1st"].clip(lower=1e-12))).sum())}), include_groups=False)
        .reset_index()
        .sort_values("group_winner_entropy", ascending=False)
    )
    production_rows = _load_optional_csv(PREDICTIONS_DIR / "backtest_results.csv")
    production_rows = production_rows[production_rows["model_name"] == selection.get("production_backtest_model_name", "")]
    production_score_note = (
        f"Selected goal-model top-5 exact-score hit rate: {goal_summary.iloc[0]['top_5_scoreline_hit_rate']:.1%}; "
        f"mean goal MAE: {goal_summary.iloc[0]['mean_goal_mae']:.3f}."
        if not goal_summary.empty
        else "Scoreline comparison was unavailable."
    )
    final_feature_columns = json.loads((MODELS_DIR / "goal_model_metadata.json").read_text(encoding="utf-8")).get("feature_columns", [])
    helped_components = (
        ", ".join(ablations.loc[ablations["helped"].astype(bool), "component"].astype(str))
        if not ablations.empty
        else "unavailable"
    )
    hurt_components = (
        ", ".join(ablations.loc[~ablations["helped"].astype(bool), "component"].astype(str))
        if not ablations.empty
        else "unavailable"
    )
    feature_groups = {
        "Team strength": [feature for feature in final_feature_columns if "elo" in feature],
        "Recent form": [feature for feature in final_feature_columns if feature in RAW_FORM_FEATURES or "above_expectation" in feature],
        "Attack/defence": [feature for feature in final_feature_columns if feature in GOAL_ATTACK_DEFENCE_FEATURES and feature not in GOAL_BASIC_FEATURES],
        "Match context": [feature for feature in final_feature_columns if feature in {"neutral", "is_friendly", "is_world_cup", "is_continental_competition", "home_advantage_flag", "host_country_flag"}],
        "Tournament importance": [feature for feature in final_feature_columns if feature == "tournament_importance"],
        "Rest/travel/venue": [feature for feature in final_feature_columns if "rest" in feature or "country" in feature],
    }
    feature_lines: list[str] = []
    for group, group_features in feature_groups.items():
        feature_lines.extend([f"### {group}", "", ", ".join(f"`{feature}`" for feature in group_features) or "No final features in this group.", ""])
    lines = [
        "# World Cup 2026 Prediction Report",
        "",
        "## A. Executive Summary",
        "",
        f"- Selected model: **{selected_prediction_model}**, using **{selection['goal_model_type']} / {selection['goal_feature_group']}** goals as its score-distribution base.",
        f"- Selected training strategy: **{strategy.get('minimum_year', minimum_year)} onward, {'no time decay' if strategy.get('time_decay_half_life_years') is None else str(strategy.get('time_decay_half_life_years')) + '-year half-life'}, {strategy.get('importance_profile', selection.get('weighting_scheme', 'configured'))} importance weights, {strategy.get('rating_model', 'standard_elo')} K scale {strategy.get('elo_k_scale', 1.0)}**.",
        f"- Indirect model decision: **{indirect_selection.get('selected_variant', 'not evaluated')}**; {indirect_selection.get('selection_reason', 'no indirect backtest available')}",
        f"- Training cutoff: **{cutoff_date}**; historical matches used: **{len(training):,}**.",
        f"- Tournament simulations: **{n_simulations or config_value('simulation', 'default_n_simulations', default=10000)}**, random seed 42.",
        "- Data sources: martj42 international results plus the official FIFA 2026 schedule/regulations.",
        f"- Biggest caveat: target-selection gains are small across only four frozen World Cups. {production_score_note}",
        "",
        "### Top 10 Champion Probabilities",
        "",
        _markdown_table(simulation.nlargest(10, "p_champion"), ["team", "p_champion", "p_reach_final"]),
        "",
        "## B. Model Description",
        "",
        "The selected system starts with two separately fitted Poisson goal regressors using pre-match Elo, rolling goals, attack/defence state, and match context. A seven-class margin classifier then estimates probabilities from away-win-by-3+ through home-win-by-3+. Its expected margin adjusts the Poisson score shape, and the score matrix is reweighted so its win/draw/loss totals match the margin probabilities.",
        "",
        f"Elo starts teams at 1500, adds a 65-point non-neutral home advantage, updates only after each match, and uses larger K-factors for more important matches plus a capped goal-difference multiplier. The selected Elo K-factor scale is **{strategy.get('elo_k_scale', 1.0)}**. The score matrix covers 0-0 through 10-10 and expected goals are capped at 5.5.",
        "",
        f"Dixon-Coles low-score adjustment rho is **{selection.get('dixon_coles_rho', 0.0)}**. The final W/D/L probabilities use **{strategy.get('draw_correction', 'no')}** draw correction with alpha **{strategy.get('draw_alpha', 1.0):.4f}** and **{strategy.get('calibration_method', 'uncalibrated')}** probability calibration.",
        "",
        "Group matches are sampled from the same score matrix used for reporting. Knockout draws receive an extra-time Poisson period; remaining ties use a strength-weighted penalty probability capped between 40% and 60%.",
        "",
        "## C. Final Feature List",
        "",
        *feature_lines,
        f"Ablation components that helped their stated comparison: {helped_components}.",
        f"Components that were neutral or hurt: {hurt_components}.",
        "A feature is retained only when it belongs to the independently selected goal-model feature set or a separately evaluated challenger.",
        "",
        "Home-confederation and reliable historical group/knockout-stage features are not used because the source data does not provide them consistently.",
        "",
        "## D. Training Setup",
        "",
        f"- Training range: {training['date'].min().date()} through {training['date'].max().date()}.",
        f"- Minimum year: {minimum_year}.",
        f"- Time decay: {selection.get('production_time_decay_half_life_years') or 'disabled'}.",
        f"- Match weighting: {selection.get('weighting_scheme')}; configurable values are in `config/config.yaml`.",
        f"- Dynamic rating: {strategy.get('rating_model', 'standard_elo')}, K scale {strategy.get('elo_k_scale', 1.0)}.",
        f"- Draw correction: {strategy.get('draw_correction', 'none')}, alpha {strategy.get('draw_alpha', 1.0):.4f}.",
        "- Validation: frozen World Cups with train cutoffs 2006, 2010, 2014, and 2018.",
        f"- Calibration: {selection.get('calibration_method_selected', 'uncalibrated')}.",
        f"- Goal cap for training targets: {selection.get('goal_cap', config_value('modeling', 'goal_cap', default=8))}.",
        "",
        "## E. Backtest Performance",
        "",
        _markdown_table(model_comparison) if not model_comparison.empty else "Model comparison unavailable.",
        "",
        "### Alternative Target Comparison",
        "",
        _markdown_table(target_summary.head(12)) if not target_summary.empty else "Target experiment comparison unavailable.",
        "",
        "### Training Strategy Search",
        "",
        _markdown_table(
            training_strategies.nsmallest(12, "avg_log_loss"),
            [
                "search_stage", "minimum_year", "half_life_years", "importance_profile",
                "goal_cap", "rating_model", "elo_k_scale", "avg_log_loss", "avg_brier_score",
                "avg_calibration_error", "avg_scoreline_top_5_hit_rate", "stability_score",
            ],
        ) if not training_strategies.empty else "Training strategy comparison unavailable.",
        "",
        "### Calibration And Draw Correction",
        "",
        _markdown_table(
            calibration_comparison.nsmallest(12, "avg_log_loss"),
            [
                "probability_source", "draw_correction", "calibration_method", "avg_draw_alpha",
                "avg_log_loss", "avg_brier_score", "avg_calibration_error",
                "avg_similar_strength_log_loss", "avg_similar_strength_draw_brier",
            ],
        ) if not calibration_comparison.empty else "Calibration comparison unavailable.",
        "",
        "### Rating Model Comparison",
        "",
        _markdown_table(
            rating_comparison,
            [
                "rating_model", "elo_k_scale", "avg_log_loss", "avg_brier_score",
                "avg_calibration_error", "avg_scoreline_top_5_hit_rate", "stability_score",
            ],
        ) if not rating_comparison.empty else "Rating model comparison unavailable.",
        "",
        "Outcome-only models do not emit coherent exact-score distributions; their scoreline hit-rate fields are marked unavailable rather than borrowed from another model.",
        "",
        "### Production Configuration By World Cup",
        "",
        _markdown_table(production_rows, ["tournament_year", "training_cutoff_year", "log_loss", "brier_score", "accuracy", "ranked_probability_score"]) if not production_rows.empty else "Production rows unavailable.",
        "",
        "### Scoreline Model Comparison",
        "",
        _markdown_table(goal_summary.head(10)) if not goal_summary.empty else "Scoreline comparison unavailable.",
        "",
        "## F. Feature Ablation Summary",
        "",
        _markdown_table(ablations) if not ablations.empty else "Feature ablation unavailable.",
        "",
        "Positive log-loss improvement helped. Negative values hurt. Noisy or unavailable context features are not presented as proven improvements.",
        "",
        "## G. Match Prediction Summary",
        "",
        "### All Group-Stage Matches",
        "",
        _markdown_table(scored.sort_values(["group", "match_date"]), match_columns),
        "",
        "### Most One-Sided Matches",
        "",
        _markdown_table(scored.nlargest(10, "favorite_probability"), match_columns),
        "",
        "### Most Even Matches",
        "",
        _markdown_table(scored.nsmallest(10, "outcome_spread"), match_columns),
        "",
        "### Most Likely Away-Side Upsets",
        "",
        _markdown_table(scored.nlargest(10, "away_edge"), match_columns),
        "",
        "### Highest Draw Probability",
        "",
        _markdown_table(scored.nlargest(10, "p_draw"), match_columns),
        "",
        "### Highest Expected Goals",
        "",
        _markdown_table(scored.nlargest(10, "expected_total_goals"), match_columns),
        "",
        "### Lowest Expected Goals",
        "",
        _markdown_table(scored.nsmallest(10, "expected_total_goals"), match_columns),
        "",
        "## H. Tournament Simulation Summary",
        "",
        "### Top 20 Champion Probabilities",
        "",
        _markdown_table(simulation.nlargest(20, "p_champion"), ["team", "group", "p_champion", "p_reach_final"]),
        "",
        "### Top 20 Final Probabilities",
        "",
        _markdown_table(simulation.nlargest(20, "p_reach_final"), ["team", "group", "p_reach_final", "p_champion"]),
        "",
        "### Most Likely Group Winners",
        "",
        _markdown_table(simulation.nlargest(12, "p_group_1st"), ["team", "group", "p_group_1st"]),
        "",
        "### Most Uncertain Groups",
        "",
        _markdown_table(group_uncertainty),
        "",
        "Overperformance/disappointment relative to external seeding and most-common final matchups are not reported: no reliable seed/ranking input or final-matchup counter is currently modeled.",
        "",
        "## I. Limitations",
        "",
        "- Football outcomes and penalties are intrinsically noisy.",
        "- No squads, player availability, injuries, betting odds, or tactical inputs are used.",
        "- Final squads and pre-tournament form may materially change team strength.",
        "- Historical international data has uneven opponent quality and incomplete stage labels.",
        "- Penalties use a deliberately simple strength-weighted approximation.",
        "- Confederation effects and travel burden are not modeled reliably.",
        "- The fixture bracket is assumed to match the populated official schedule.",
        "",
        "## J. Updating Predictions Later",
        "",
        "1. Add new completed matches to `data/raw/results.csv`, then rerun the pipeline with a later cutoff.",
        "2. Squad or injury information requires a new validated input and backtested feature before use.",
        "3. After group-stage results, update the fixture/simulation state; live tournament updating is not yet implemented.",
        "4. Re-run `scripts/train_models.py` and `scripts/predict_worldcup_2026.py` with the desired cutoff and a simulation count no higher than the configured maximum.",
        "",
        "These are probabilistic estimates, not certainties.",
    ]
    (REPORTS_DIR / "worldcup_2026_prediction_report.md").write_text("\n".join(lines), encoding="utf-8")


def create_model_card(cutoff_date: str, selection: dict[str, object]) -> None:
    goal_metadata = json.loads((MODELS_DIR / "goal_model_metadata.json").read_text(encoding="utf-8"))
    outcome_metadata = json.loads((MODELS_DIR / "outcome_model_metadata.json").read_text(encoding="utf-8"))
    target_selection = selection.get("target_experiment_recommendation", {})
    strategy = selection.get("training_strategy_recommendation", {})
    indirect_path = MODELS_DIR / "indirect_model_selection.json"
    indirect_selection = json.loads(indirect_path.read_text(encoding="utf-8")) if indirect_path.exists() else {}
    selected_prediction_model = target_selection.get("recommended_model_name", selection["model_name"])
    lines = [
        "# Model Card",
        "",
        "## Identity",
        "",
        f"- Model/version: `wc2026-predictor 0.1.0`, selected target/model `{selected_prediction_model}`.",
        f"- Goal model: `{selection['goal_model_type']}` with `{selection['goal_feature_group']}`.",
        f"- Training strategy: `{strategy.get('minimum_year', selection['minimum_training_year'])}` onward, `{strategy.get('importance_profile', selection.get('weighting_scheme', 'configured'))}` weights, `{strategy.get('rating_model', 'standard_elo')}` K scale `{strategy.get('elo_k_scale', 1.0)}`.",
        f"- Probability adjustment: draw correction `{strategy.get('draw_correction', 'none')}`, calibration `{strategy.get('calibration_method', 'uncalibrated')}`.",
        f"- Indirect correction: `{indirect_selection.get('selected_variant', 'not evaluated')}`; comparison challenger `{indirect_selection.get('comparison_variant', 'not evaluated')}`.",
        f"- Prediction cutoff: `{cutoff_date}`.",
        "",
        "## Targets",
        "",
        "- Home and away goals, capped only as training targets.",
        "- Derived home-win, draw, away-win, scoreline, and tournament advancement probabilities.",
        "- Seven-class match-margin probabilities used as the selected outcome correction target.",
        "",
        "## Training And Excluded Data",
        "",
        f"- Historical results from {selection['minimum_training_year']} through {cutoff_date}.",
        "- Excludes unplayed rows, rows after the cutoff, and clear duplicate records.",
        "- Does not use squads, injuries, betting odds, post-match rankings, or future tournament results.",
        "",
        "## Feature Groups",
        "",
        f"- Goal features: {', '.join(f'`{column}`' for column in goal_metadata['feature_columns'])}.",
        f"- Challenger outcome features: {', '.join(f'`{column}`' for column in outcome_metadata['feature_columns'])}.",
        "",
        "## Leakage Controls",
        "",
        "- Features are captured before state updates; same-day matches share only prior-day state.",
        "- Elo is pre-match Elo. Rolling form excludes the current match.",
        "- Frozen World Cup backtests train only through 2006, 2010, 2014, and 2018.",
        "- Future-form research labels are kept separate and never joined into same-match features.",
        "- Calibration uses a chronological tail of the available training period.",
        "",
        "## Validation Design",
        "",
        "- Primary: average outcome log loss across World Cups 2010, 2014, 2018, and 2022.",
        "- Secondary: Brier score, ranked probability score, goal MAE, exact-score log loss, top-1/top-5 score hits, stability, then accuracy.",
        "",
        "## Known Failure Modes",
        "",
        "- Sudden squad/injury/tactical changes are invisible.",
        "- Sparse teams and cross-confederation comparisons are uncertain.",
        "- Scoreline tails and penalties are simplified.",
        "- Small tournament test sets make model ranking noisy.",
        "",
        "## Intended Use",
        "",
        "- Scenario analysis, probabilistic tournament forecasting, and model comparison.",
        "",
        "## Not Intended Use",
        "",
        "- Claims of certainty, gambling guarantees, player-level decisions, or live in-match forecasting.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python scripts/fetch_data.py",
        f"python scripts/build_features.py --cutoff-date {cutoff_date}",
        f"python scripts/train_models.py --cutoff-date {cutoff_date}",
        "python scripts/run_backtest.py",
        f"python scripts/predict_worldcup_2026.py --cutoff-date {cutoff_date} --n-simulations 10000",
        "python -m pytest -q",
        "```",
    ]
    (REPORTS_DIR / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def create_figures(simulation: pd.DataFrame, matches: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def save_bar(frame: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        plt.barh(frame[x][::-1], frame[y][::-1])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=140)
        plt.close()

    save_bar(simulation.nlargest(20, "p_champion"), "team", "p_champion", "World Cup 2026 Champion Probabilities", "champion_probabilities.png")
    model_comparison = _load_optional_csv(PREDICTIONS_DIR / "backtest_model_comparison.csv")
    if not model_comparison.empty:
        save_bar(model_comparison.nsmallest(12, "average_log_loss"), "model_name", "average_log_loss", "Frozen World Cup Mean Log Loss", "model_comparison_log_loss.png")
    save_bar(simulation.nlargest(20, "p_reach_r32"), "team", "p_reach_r32", "Round-of-32 Qualification Probabilities", "group_qualification_probabilities.png")
    plt.figure(figsize=(8, 5))
    plt.hist(matches["expected_goals_home"] + matches["expected_goals_away"], bins=12)
    plt.title("Expected Total Goals: Group Matches")
    plt.tight_layout()
    plt.savefig(figures_dir / "expected_goals_distribution.png", dpi=140)
    plt.close()
    calibration = _load_optional_csv(PROCESSED_DATA_DIR / "calibration_tables.csv")
    if not calibration.empty:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.scatter(calibration["predicted_rate"], calibration["observed_rate"], alpha=0.35)
        plt.xlabel("Predicted home-win probability")
        plt.ylabel("Observed home-win rate")
        plt.title("Calibration Across Frozen Backtests")
        plt.tight_layout()
        plt.savefig(figures_dir / "calibration_plot.png", dpi=140)
        plt.close()
