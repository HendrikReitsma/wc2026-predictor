from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.advanced_backtest import (
    run_calibration_comparison,
    run_goal_model_comparison,
    run_minimum_year_comparison,
)
from src.evaluation.metrics import compute_classification_metrics, reliability_table
from src.models.scoreline import outcome_probabilities, scoreline_matrix
from src.models.train_goal_model import GOAL_FEATURE_GROUPS, train_goal_model
from src.models.train_outcome_model import (
    CORE_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    RAW_FORM_FEATURES,
    _fit_baseline_model,
    elo_outcome_probabilities,
    train_outcome_model,
)
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import INTERNAL_DOCS_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)
WORLD_CUP_YEARS = [2010, 2014, 2018, 2022]
WORLD_CUP_TRAINING_CUTOFFS = {2010: 2006, 2014: 2010, 2018: 2014, 2022: 2018}


def _load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "match_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed features not found at {path}. Run scripts/build_features.py first.")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    minimum_year = int(config_value("modeling", "minimum_training_year", default=1990))
    return frame[frame["date"].dt.year >= minimum_year].copy()


def _world_cup_frame(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    return frame.loc[
        (frame["date"].dt.year == year) & frame["is_world_cup"].fillna(0).astype(bool)
    ].copy()


def _poisson_outcome_probabilities(
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
    max_goals: int = 10,
    dixon_coles_rho: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [
            outcome_probabilities(
                scoreline_matrix(home, away, max_goals=max_goals, dixon_coles_rho=dixon_coles_rho)
            )
            for home, away in zip(home_lambdas, away_lambdas)
        ]
    )


def _metrics_row(
    model_name: str,
    year: int,
    half_life: float,
    evaluation_frame: pd.DataFrame,
    probabilities: np.ndarray,
    notes: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "tournament_year": year,
        "half_life_years": half_life,
        "training_cutoff_year": WORLD_CUP_TRAINING_CUTOFFS[year],
        **compute_classification_metrics(
            evaluation_frame["result"].to_numpy(),
            probabilities,
            np.argmax(probabilities, axis=1),
        ),
        "notes": notes,
    }


def _train_frame_for_world_cup(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    return frame[frame["date"].dt.year <= WORLD_CUP_TRAINING_CUTOFFS[year]].copy()


def _calibrated_poisson_probabilities(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    half_life: float,
    use_match_importance: bool | None = None,
    goal_feature_columns: list[str] | None = None,
    goal_model_type: str = "poisson",
    dixon_coles_rho: float = 0.0,
    minimum_year: int | None = None,
) -> np.ndarray:
    calibration_cutoff = training_frame["date"].max() - pd.DateOffset(years=2)
    base_frame = training_frame[training_frame["date"] < calibration_cutoff].copy()
    calibration_frame = training_frame[training_frame["date"] >= calibration_cutoff].copy()
    goal_bundle = train_goal_model(
        base_frame,
        base_frame["date"].max(),
        persist=False,
        half_life_years=half_life,
        use_match_importance=use_match_importance,
        feature_columns=goal_feature_columns,
        model_type=goal_model_type,
        minimum_year=minimum_year,
    )
    goal_columns = goal_bundle.feature_columns
    calibration_home = np.clip(goal_bundle.home_model.predict(calibration_frame[goal_columns]), 0.05, 5.5)
    calibration_away = np.clip(goal_bundle.away_model.predict(calibration_frame[goal_columns]), 0.05, 5.5)
    raw_calibration = _poisson_outcome_probabilities(calibration_home, calibration_away, dixon_coles_rho=dixon_coles_rho)
    calibrator = LogisticRegression(max_iter=2000, random_state=int(config_value("project", "random_seed", default=42)))
    calibrator.fit(np.log(np.clip(raw_calibration, 1e-8, 1.0)), calibration_frame["result"])
    evaluation_home = np.clip(goal_bundle.home_model.predict(evaluation_frame[goal_columns]), 0.05, 5.5)
    evaluation_away = np.clip(goal_bundle.away_model.predict(evaluation_frame[goal_columns]), 0.05, 5.5)
    raw_evaluation = _poisson_outcome_probabilities(evaluation_home, evaluation_away, dixon_coles_rho=dixon_coles_rho)
    return calibrator.predict_proba(np.log(np.clip(raw_evaluation, 1e-8, 1.0)))


def _backtest_candidates(
    frame: pd.DataFrame,
    half_lives: list[float],
    feature_columns: list[str],
    goal_feature_columns: list[str],
    goal_model_type: str,
    dixon_coles_rho: float,
    minimum_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    poisson_weight = float(config_value("modeling", "outcome_weight_poisson", default=0.5))
    ml_weight = float(config_value("modeling", "outcome_weight_ml", default=0.5))

    for half_life in half_lives:
        for year in WORLD_CUP_YEARS:
            training_frame = _train_frame_for_world_cup(frame, year)
            evaluation_frame = _world_cup_frame(frame, year)
            if training_frame.empty or evaluation_frame.empty:
                continue

            outcome = train_outcome_model(
                training_frame,
                pd.Timestamp(f"{year}-01-01"),
                persist=False,
                half_life_years=half_life,
                feature_columns=feature_columns,
            )
            goals = train_goal_model(
                training_frame,
                pd.Timestamp(f"{year}-01-01"),
                persist=False,
                half_life_years=half_life,
                feature_columns=goal_feature_columns,
                model_type=goal_model_type,
                minimum_year=minimum_year,
            )
            probabilities = {
                "elo_baseline": elo_outcome_probabilities(evaluation_frame),
                "ml_challenger": outcome.challenger_model.predict_proba(evaluation_frame[feature_columns]),
                "ml_calibrated": outcome.calibrated_model.predict_proba(evaluation_frame[feature_columns]),
            }
            form_columns = CORE_FEATURES + RAW_FORM_FEATURES
            form_unweighted = _fit_baseline_model(training_frame, form_columns, 1e9, False)
            form_weighted = _fit_baseline_model(training_frame, form_columns, half_life, True)
            probabilities["elo_recent_form"] = form_unweighted.predict_proba(evaluation_frame[form_columns])
            probabilities["elo_recent_form_weighted"] = form_weighted.predict_proba(evaluation_frame[form_columns])
            home_lambda = np.clip(goals.home_model.predict(evaluation_frame[goals.feature_columns]), 0.05, 5.5)
            away_lambda = np.clip(goals.away_model.predict(evaluation_frame[goals.feature_columns]), 0.05, 5.5)
            probabilities["poisson_goal"] = _poisson_outcome_probabilities(
                home_lambda, away_lambda, dixon_coles_rho=dixon_coles_rho
            )
            probabilities["poisson_calibrated"] = _calibrated_poisson_probabilities(
                training_frame,
                evaluation_frame,
                half_life,
                goal_feature_columns=goal_feature_columns,
                goal_model_type=goal_model_type,
                dixon_coles_rho=dixon_coles_rho,
                minimum_year=minimum_year,
            )
            probabilities["ensemble"] = (
                poisson_weight * probabilities["poisson_goal"] + ml_weight * probabilities["ml_calibrated"]
            )

            for model_name, model_probabilities in probabilities.items():
                rows.append(
                    _metrics_row(
                        model_name,
                        year,
                        half_life,
                        evaluation_frame,
                        model_probabilities,
                        "Strictly trained on matches before the tested World Cup year.",
                    )
                )
                for calibration_row in reliability_table(
                    evaluation_frame["result"].to_numpy(), model_probabilities, n_bins=5
                ):
                    calibration_rows.append(
                        {
                            "model_name": model_name,
                            "tournament_year": year,
                            "half_life_years": half_life,
                            **calibration_row,
                        }
                    )
            LOGGER.info("Completed candidate backtest for %s with half-life %s.", year, half_life)
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows)


def _run_feature_ablations(frame: pd.DataFrame, half_life: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        for year in WORLD_CUP_YEARS:
            training_frame = _train_frame_for_world_cup(frame, year)
            evaluation_frame = _world_cup_frame(frame, year)
            model = _fit_baseline_model(training_frame, columns, half_life, True)
            probabilities = model.predict_proba(evaluation_frame[columns])
            rows.append(_metrics_row(group_name, year, half_life, evaluation_frame, probabilities, "Feature ablation."))
    ablations = pd.DataFrame(rows)
    means = ablations.groupby("model_name", as_index=False)[["log_loss", "brier_score", "accuracy"]].mean()
    baseline_log_loss = float(means.loc[means["model_name"] == "core", "log_loss"].iloc[0])
    means["log_loss_improvement_vs_core"] = baseline_log_loss - means["log_loss"]
    return means.sort_values("log_loss")


def _run_weighting_ablations(
    frame: pd.DataFrame,
    feature_columns: list[str],
    half_life: float,
    selected_model_name: str,
    goal_feature_columns: list[str],
    goal_model_type: str,
    dixon_coles_rho: float,
    minimum_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configurations = {
        "unweighted": (1e9, False),
        "time_decay_only": (half_life, False),
        "importance_only": (1e9, True),
        "time_decay_and_importance": (half_life, True),
    }
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for label, (configured_half_life, use_importance) in configurations.items():
        for year in WORLD_CUP_YEARS:
            training_frame = _train_frame_for_world_cup(frame, year)
            evaluation_frame = _world_cup_frame(frame, year)
            outcome = train_outcome_model(
                training_frame,
                pd.Timestamp(f"{year}-01-01"),
                persist=False,
                half_life_years=configured_half_life,
                use_match_importance=use_importance,
                feature_columns=feature_columns,
            )
            goals = train_goal_model(
                training_frame,
                pd.Timestamp(f"{year}-01-01"),
                persist=False,
                half_life_years=configured_half_life,
                use_match_importance=use_importance,
                feature_columns=goal_feature_columns,
                model_type=goal_model_type,
                minimum_year=minimum_year,
            )
            ml_probabilities = outcome.calibrated_model.predict_proba(evaluation_frame[feature_columns])
            home_lambda = np.clip(goals.home_model.predict(evaluation_frame[goals.feature_columns]), 0.05, 5.5)
            away_lambda = np.clip(goals.away_model.predict(evaluation_frame[goals.feature_columns]), 0.05, 5.5)
            poisson_probabilities = _poisson_outcome_probabilities(
                home_lambda, away_lambda, dixon_coles_rho=dixon_coles_rho
            )
            ensemble_probabilities = (
                float(config_value("modeling", "outcome_weight_poisson", default=0.5)) * poisson_probabilities
                + float(config_value("modeling", "outcome_weight_ml", default=0.5)) * ml_probabilities
            )
            probabilities_by_model = {
                "poisson_goal": poisson_probabilities,
                "poisson_calibrated": _calibrated_poisson_probabilities(
                    training_frame,
                    evaluation_frame,
                    configured_half_life,
                    use_match_importance=use_importance,
                    goal_feature_columns=goal_feature_columns,
                    goal_model_type=goal_model_type,
                    dixon_coles_rho=dixon_coles_rho,
                    minimum_year=minimum_year,
                ),
                "ml_challenger": outcome.challenger_model.predict_proba(evaluation_frame[feature_columns]),
                "ml_calibrated": ml_probabilities,
                "ensemble": ensemble_probabilities,
            }
            if selected_model_name not in probabilities_by_model:
                raise ValueError(f"Weighting ablation is not defined for selected model {selected_model_name!r}.")
            rows.append(
                _metrics_row(
                    label,
                    year,
                    configured_half_life,
                    evaluation_frame,
                    probabilities_by_model[selected_model_name],
                    f"{selected_model_name} weighting ablation.",
                )
            )
            for calibration_row in reliability_table(
                evaluation_frame["result"].to_numpy(),
                probabilities_by_model[selected_model_name],
                n_bins=5,
            ):
                calibration_rows.append(
                    {
                        "model_name": label,
                        "model_family": selected_model_name,
                        "tournament_year": year,
                        "half_life_years": configured_half_life,
                        **calibration_row,
                    }
                )
    details = pd.DataFrame(rows)
    means = details.groupby("model_name", as_index=False)[["log_loss", "brier_score", "accuracy"]].mean()
    unweighted_log_loss = float(means.loc[means["model_name"] == "unweighted", "log_loss"].iloc[0])
    means["log_loss_improvement_vs_unweighted"] = unweighted_log_loss - means["log_loss"]
    return means.sort_values("log_loss"), details, pd.DataFrame(calibration_rows)


def _build_ablation_summary(
    feature_ablations: pd.DataFrame,
    weighting_ablations: pd.DataFrame,
    model_summary: pd.DataFrame,
    selected_half_life: float,
) -> pd.DataFrame:
    feature_metrics = feature_ablations.set_index("model_name")
    weighting_metrics = weighting_ablations.set_index("model_name")
    model_metrics = model_summary[
        model_summary["half_life_years"] == selected_half_life
    ].set_index("model_name")
    rows: list[dict[str, Any]] = []

    def add_comparison(
        component: str,
        candidate: str,
        reference: str,
        metrics: pd.DataFrame,
    ) -> None:
        candidate_row = metrics.loc[candidate]
        reference_row = metrics.loc[reference]
        log_loss_improvement = float(reference_row["log_loss"] - candidate_row["log_loss"])
        brier_improvement = float(reference_row["brier_score"] - candidate_row["brier_score"])
        rows.append(
            {
                "component": component,
                "candidate": candidate,
                "reference": reference,
                "candidate_log_loss": float(candidate_row["log_loss"]),
                "reference_log_loss": float(reference_row["log_loss"]),
                "log_loss_improvement": log_loss_improvement,
                "brier_improvement": brier_improvement,
                "helped": bool(log_loss_improvement > 0),
            }
        )

    add_comparison("Elo features", "core", "core_without_elo", feature_metrics)
    add_comparison("recent form", "core_raw_form", "core", feature_metrics)
    add_comparison("opponent-adjusted form", "core_opponent_adjusted", "core", feature_metrics)
    add_comparison("attack/defence features", "all_features", "all_without_attack_defence", feature_metrics)
    add_comparison("rest days", "core", "core_without_rest", feature_metrics)
    add_comparison("venue/host features", "core", "core_without_venue_host", feature_metrics)
    add_comparison("match importance weighting", "importance_only", "unweighted", weighting_metrics)
    add_comparison("time decay weighting", "time_decay_only", "unweighted", weighting_metrics)
    add_comparison("combined weighting", "time_decay_and_importance", "unweighted", weighting_metrics)
    add_comparison("calibration", "poisson_calibrated", "poisson_goal", model_metrics)
    add_comparison("Poisson model", "poisson_goal", "elo_baseline", model_metrics)
    add_comparison("ML model", "ml_challenger", "elo_baseline", model_metrics)
    better_component = min(
        ["poisson_goal", "ml_calibrated"],
        key=lambda model_name: float(model_metrics.loc[model_name, "log_loss"]),
    )
    add_comparison("ensemble", "ensemble", better_component, model_metrics)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(f"{row[column]:.5f}" if isinstance(row[column], float) else str(row[column]) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    ]
    return "\n".join([header, separator, *body])


def _write_report(
    backtest_frame: pd.DataFrame,
    half_life_summary: pd.DataFrame,
    ablation_frame: pd.DataFrame,
    weighting_ablation_frame: pd.DataFrame,
    requested_ablation_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    goal_model_summary: pd.DataFrame,
    minimum_year_summary: pd.DataFrame,
    selection: dict[str, Any],
) -> None:
    selected_calibration = calibration_frame[
        (calibration_frame["model_name"] == selection.get("calibration_model_name", selection["model_name"]))
        & (
            calibration_frame["half_life_years"]
            == selection.get("calibration_half_life_years", selection["half_life_years"])
        )
    ]
    calibration_summary = (
        selected_calibration.groupby(["lower", "upper"], as_index=False)[["count", "observed_rate", "predicted_rate"]]
        .mean()
        .sort_values("lower")
    )
    calibration_method_summary = (
        calibration_comparison.groupby("calibration_method", as_index=False)[
            ["log_loss", "brier_score", "accuracy", "ranked_probability_score"]
        ]
        .mean()
        .sort_values("log_loss")
    )
    lines = [
        "# Evaluation Summary",
        "",
        "## Selected Configuration",
        "",
        f"- Selected model family: **{selection['model_name']}**.",
        f"- Best tested decayed half-life: **{selection['half_life_years']} years**.",
        f"- Production time-decay half-life: **{selection['production_time_decay_half_life_years'] or 'disabled'}**.",
        f"- Selected model feature set: **{selection['selected_model_feature_set']}**.",
        f"- Selected goal model: **{selection['goal_model_type']} / {selection['goal_feature_group']}**.",
        f"- Dixon-Coles low-score rho: **{selection['dixon_coles_rho']}**.",
        f"- Selected minimum training year: **{selection['minimum_training_year']}**.",
        f"- Selected calibration method: **{selection['calibration_method_selected']}**.",
        f"- Best classifier challenger feature group: **{selection['feature_group']}**.",
        f"- Selected weighting scheme: **{selection['weighting_scheme']}**.",
        "- Selection criterion: lowest mean log loss across the 2010, 2014, 2018, and 2022 World Cups.",
        "- Accuracy is secondary.",
        "",
        "## Model And Half-Life Comparison",
        "",
        _markdown_table(half_life_summary, ["model_name", "half_life_years", "log_loss", "brier_score", "accuracy"]),
        "",
        "## Goal And Scoreline Model Comparison",
        "",
        _markdown_table(
            goal_model_summary,
            ["model_name", "log_loss", "brier_score", "mean_goal_mae", "scoreline_log_loss", "top_1_scoreline_accuracy", "top_5_scoreline_hit_rate"],
        ),
        "",
        "## Minimum Training Year Comparison",
        "",
        _markdown_table(
            minimum_year_summary,
            ["minimum_training_year", "world_cups_covered", "log_loss", "brier_score", "mean_goal_mae", "top_5_scoreline_hit_rate"],
        ),
        "",
        "The 2010 minimum-year candidate cannot cover the 2010 backtest because its frozen training cutoff is 2006; selection favors complete four-tournament coverage.",
        "",
        "## Calibration Method Comparison",
        "",
        _markdown_table(
            calibration_method_summary,
            ["calibration_method", "log_loss", "brier_score", "accuracy", "ranked_probability_score"],
        ),
        "",
        "## Requested Ablation Summary",
        "",
        _markdown_table(
            requested_ablation_frame,
            ["component", "candidate", "reference", "candidate_log_loss", "reference_log_loss", "log_loss_improvement", "brier_improvement", "helped"],
        ),
        "",
        "Positive improvement means the candidate helped against the named reference on frozen World Cup backtests.",
        "",
        "## Detailed Feature Ablation",
        "",
        _markdown_table(
            ablation_frame,
            ["model_name", "log_loss", "brier_score", "accuracy", "log_loss_improvement_vs_core"],
        ),
        "",
        "Positive `log_loss_improvement_vs_core` means the feature group helped relative to Elo/context/rest alone.",
        "",
        "## Weighting Ablation",
        "",
        _markdown_table(
            weighting_ablation_frame,
            ["model_name", "log_loss", "brier_score", "accuracy", "log_loss_improvement_vs_unweighted"],
        ),
        "",
        "Positive `log_loss_improvement_vs_unweighted` means the weighting choice helped.",
        "",
        "## Selected-Model Calibration Table",
        "",
        _markdown_table(calibration_summary, ["lower", "upper", "count", "observed_rate", "predicted_rate"]),
        "",
        "## World Cup Backtests",
        "",
        _markdown_table(
            backtest_frame,
            ["model_name", "tournament_year", "training_cutoff_year", "half_life_years", "log_loss", "brier_score", "accuracy", "average_probability_actual_outcome"],
        ),
        "",
        "## Leakage And Data-Quality Checks",
        "",
        "- Every feature is computed before the match and same-day matches update state only after all same-day features are captured.",
        "- All validation and calibration splits are chronological; there are no random train/test splits.",
        "- Frozen World Cup training cutoffs are 2006, 2010, 2014, and 2018 for the 2010, 2014, 2018, and 2022 tests.",
        "- Upsets and surprising results are retained.",
        "- Only clear duplicate/unscored rows are excluded; goal-model targets are capped rather than matches deleted.",
        "- Match importance affects both Elo updates and model sample weights.",
        "- The World Cup qualification classifier was corrected so qualifiers are not labeled as World Cup finals.",
        "- Historical results do not contain reliable tournament-stage labels; group/knockout-specific weights fall back to tournament-level weights for those rows.",
        "- Detailed feature-group ablations use the classifier challenger; the selected Poisson model is assessed as a complete family with its fixed goal-feature set.",
        "- No FIFA ranking, squad, player availability, injury, or betting-odds inputs are used.",
    ]
    INTERNAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (INTERNAL_DOCS_DIR / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")


def backtest_world_cups() -> pd.DataFrame:
    ensure_project_dirs()
    frame = _load_features()
    goal_backtests, goal_summary = run_goal_model_comparison(frame, WORLD_CUP_YEARS, WORLD_CUP_TRAINING_CUTOFFS)
    goal_selection = goal_summary.iloc[0]
    goal_feature_group = str(goal_selection["goal_feature_group"])
    goal_feature_columns = GOAL_FEATURE_GROUPS[goal_feature_group]
    goal_model_type = str(goal_selection["goal_model_type"])
    dixon_coles_rho = float(goal_selection["dixon_coles_rho"])
    minimum_year_backtests, minimum_year_summary = run_minimum_year_comparison(
        frame,
        WORLD_CUP_YEARS,
        WORLD_CUP_TRAINING_CUTOFFS,
        goal_model_type,
        goal_feature_columns,
        dixon_coles_rho,
    )
    complete_coverage = int(minimum_year_summary["world_cups_covered"].max())
    minimum_year = int(
        minimum_year_summary.loc[
            minimum_year_summary["world_cups_covered"] == complete_coverage
        ].sort_values("log_loss").iloc[0]["minimum_training_year"]
    )
    half_lives = [float(value) for value in config_value("modeling", "time_decay_candidates", default=[4, 8, 12])]
    initial_backtests, _ = _backtest_candidates(
        frame,
        half_lives,
        FEATURE_COLUMNS,
        goal_feature_columns,
        goal_model_type,
        dixon_coles_rho,
        minimum_year,
    )
    initial_summary = (
        initial_backtests.groupby(["model_name", "half_life_years"], as_index=False)[["log_loss", "brier_score", "accuracy"]]
        .mean()
        .sort_values("log_loss")
    )
    initial_half_life = float(initial_summary.iloc[0]["half_life_years"])
    ablations = _run_feature_ablations(frame, initial_half_life)
    best_feature_group = str(ablations.iloc[0]["model_name"])
    selected_columns = FEATURE_GROUPS[best_feature_group]

    backtests, calibration = _backtest_candidates(
        frame,
        half_lives,
        selected_columns,
        goal_feature_columns,
        goal_model_type,
        dixon_coles_rho,
        minimum_year,
    )
    summary = (
        backtests.groupby(["model_name", "half_life_years"], as_index=False)[["log_loss", "brier_score", "accuracy"]]
        .mean()
        .sort_values("log_loss")
    )
    selection_row = summary.iloc[0]
    selection = {
        "model_name": str(selection_row["model_name"]),
        "half_life_years": float(selection_row["half_life_years"]),
        "feature_group": best_feature_group,
        "feature_columns": selected_columns,
        "selected_model_feature_set": (
            goal_feature_group
            if str(selection_row["model_name"]).startswith("poisson")
            else best_feature_group
        ),
        "goal_model_type": goal_model_type,
        "goal_feature_group": goal_feature_group,
        "dixon_coles_rho": dixon_coles_rho,
        "minimum_training_year": minimum_year,
        "selection_metric": "mean_world_cup_log_loss",
        "mean_log_loss": float(selection_row["log_loss"]),
    }
    weighting_ablations, weighting_backtests, weighting_calibration = _run_weighting_ablations(
        frame,
        selected_columns,
        selection["half_life_years"],
        selection["model_name"],
        goal_feature_columns,
        goal_model_type,
        dixon_coles_rho,
        minimum_year,
    )
    weighting_choice = str(weighting_ablations.iloc[0]["model_name"])
    weighting_config = {
        "unweighted": (1e9, False),
        "time_decay_only": (selection["half_life_years"], False),
        "importance_only": (1e9, True),
        "time_decay_and_importance": (selection["half_life_years"], True),
    }[weighting_choice]
    selection["weighting_scheme"] = weighting_choice
    selection["training_half_life_years"] = float(weighting_config[0])
    selection["use_match_importance_weights"] = bool(weighting_config[1])
    selection["production_time_decay_half_life_years"] = (
        None if weighting_config[0] >= 1e8 else float(weighting_config[0])
    )
    selected_weighting_row = weighting_ablations.loc[
        weighting_ablations["model_name"] == weighting_choice
    ].iloc[0]
    selection["mean_log_loss"] = float(selected_weighting_row["log_loss"])
    selection["mean_brier_score"] = float(selected_weighting_row["brier_score"])
    selection["production_backtest_model_name"] = f"{selection['model_name']}_production"
    production_backtests = weighting_backtests.loc[
        weighting_backtests["model_name"] == weighting_choice
    ].copy()
    production_backtests["model_name"] = selection["production_backtest_model_name"]
    production_calibration = weighting_calibration.loc[
        weighting_calibration["model_name"] == weighting_choice
    ].copy()
    production_calibration["model_name"] = selection["production_backtest_model_name"]
    selection["calibration_model_name"] = selection["production_backtest_model_name"]
    selection["calibration_half_life_years"] = float(weighting_config[0])
    calibration_comparison = run_calibration_comparison(
        frame,
        WORLD_CUP_YEARS,
        WORLD_CUP_TRAINING_CUTOFFS,
        goal_model_type,
        goal_feature_columns,
        dixon_coles_rho,
        minimum_year,
    )
    calibration_method_summary = (
        calibration_comparison.groupby("calibration_method", as_index=False)[["log_loss", "brier_score"]]
        .mean()
        .sort_values("log_loss")
    )
    selection["calibration_method_selected"] = str(calibration_method_summary.iloc[0]["calibration_method"])
    backtests = pd.concat([backtests, production_backtests], ignore_index=True)
    calibration = pd.concat([calibration, production_calibration], ignore_index=True)
    requested_ablations = _build_ablation_summary(
        ablations,
        weighting_ablations,
        summary,
        selection["half_life_years"],
    )

    backtests.to_csv(PROCESSED_DATA_DIR / "backtest_results.csv", index=False)
    backtests.to_csv(PREDICTIONS_DIR / "backtest_results.csv", index=False)
    model_best_rows = summary.sort_values("log_loss").groupby("model_name", as_index=False).first()
    model_comparison_rows: list[dict[str, Any]] = []
    for _, model_row in model_best_rows.iterrows():
        model_name = str(model_row["model_name"])
        half_life = float(model_row["half_life_years"])
        details = backtests[
            (backtests["model_name"] == model_name) & (backtests["half_life_years"] == half_life)
        ]
        output_row: dict[str, Any] = {
            "model_name": model_name,
            "half_life_years": half_life,
            "average_log_loss": float(model_row["log_loss"]),
            "average_brier_score": float(model_row["brier_score"]),
            "average_accuracy": float(model_row["accuracy"]),
            "average_scoreline_top_5_hit_rate": (
                float(goal_selection["top_5_scoreline_hit_rate"])
                if model_name in {"poisson_goal", "poisson_calibrated", "ensemble"}
                else np.nan
            ),
            "notes": (
                f"Uses {goal_feature_group} score distribution."
                if model_name in {"poisson_goal", "poisson_calibrated", "ensemble"}
                else "Outcome-only model; no coherent exact-score distribution."
            ),
        }
        for year in WORLD_CUP_YEARS:
            year_rows = details[details["tournament_year"] == year]
            output_row[f"world_cup_{year}_log_loss"] = (
                float(year_rows.iloc[0]["log_loss"]) if not year_rows.empty else np.nan
            )
        model_comparison_rows.append(output_row)
    pd.DataFrame(model_comparison_rows).sort_values("average_log_loss").to_csv(
        PREDICTIONS_DIR / "backtest_model_comparison.csv", index=False
    )
    goal_backtests.to_csv(PREDICTIONS_DIR / "backtest_scoreline_comparison.csv", index=False)
    calibration_comparison.to_csv(PREDICTIONS_DIR / "calibration_results.csv", index=False)
    minimum_year_backtests.to_csv(PROCESSED_DATA_DIR / "minimum_year_backtests.csv", index=False)
    minimum_year_summary.to_csv(PROCESSED_DATA_DIR / "minimum_year_comparison.csv", index=False)
    goal_summary.to_csv(PROCESSED_DATA_DIR / "goal_model_comparison.csv", index=False)
    summary.to_csv(PROCESSED_DATA_DIR / "model_comparison.csv", index=False)
    calibration.to_csv(PROCESSED_DATA_DIR / "calibration_tables.csv", index=False)
    ablations.to_csv(PROCESSED_DATA_DIR / "feature_ablation_results.csv", index=False)
    requested_ablations.to_csv(PROCESSED_DATA_DIR / "ablation_results.csv", index=False)
    requested_ablations.to_csv(PREDICTIONS_DIR / "ablation_results.csv", index=False)
    requested_ablations.to_csv(PREDICTIONS_DIR / "feature_ablation_results.csv", index=False)
    weighting_ablations.to_csv(PROCESSED_DATA_DIR / "weighting_ablation_results.csv", index=False)
    weighting_backtests.to_csv(PROCESSED_DATA_DIR / "weighting_ablation_backtests.csv", index=False)
    (MODELS_DIR / "model_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    _write_report(
        backtests,
        summary,
        ablations,
        weighting_ablations,
        requested_ablations,
        calibration,
        calibration_comparison,
        goal_summary,
        minimum_year_summary,
        selection,
    )
    LOGGER.info("Selected %s with a %s-year half-life.", selection["model_name"], selection["half_life_years"])
    return backtests
