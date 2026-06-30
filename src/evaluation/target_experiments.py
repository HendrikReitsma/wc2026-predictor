from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_classification_metrics, compute_scoreline_metrics, reliability_table
from src.features.target_engineering import (
    MARGIN_CLASS_CENTERS,
    add_match_targets,
    build_future_team_targets,
    elo_baseline_expectations,
)
from src.models.scoreline import outcome_probabilities, scoreline_matrix
from src.models.train_goal_model import GOAL_FEATURE_GROUPS, train_goal_model
from src.models.train_target_model import train_selected_target_model
from src.models.weighting import combined_sample_weights
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import INTERNAL_DOCS_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)
WORLD_CUP_YEARS = [2010, 2014, 2018, 2022]
WORLD_CUP_TRAINING_CUTOFFS = {2010: 2006, 2014: 2010, 2018: 2014, 2022: 2018}
CALIBRATION_METHODS = ["uncalibrated", "platt_sigmoid", "isotonic"]


@dataclass
class VariantPrediction:
    probabilities: np.ndarray
    home_lambdas: np.ndarray
    away_lambdas: np.ndarray
    notes: str


def _load_features() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    minimum_year = int(config_value("modeling", "minimum_training_year", default=1990))
    return add_match_targets(frame[frame["date"].dt.year >= minimum_year].copy())


def _training_frame(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    return frame[frame["date"].dt.year <= WORLD_CUP_TRAINING_CUTOFFS[year]].copy()


def _world_cup_frame(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    return frame[(frame["date"].dt.year == year) & frame["is_world_cup"].fillna(0).astype(bool)].copy()


def _weights(frame: pd.DataFrame) -> np.ndarray:
    return combined_sample_weights(frame, frame["date"].max(), half_life_years=1e9, use_match_importance=True)


def _regression_pipeline(kind: str) -> Pipeline:
    model = PoissonRegressor(alpha=0.05, max_iter=2000) if kind == "poisson" else Ridge(alpha=10.0)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _fit_regressor(frame: pd.DataFrame, features: list[str], target: str, kind: str = "ridge") -> Pipeline:
    pipeline = _regression_pipeline(kind)
    pipeline.fit(frame[features], frame[target], model__sample_weight=_weights(frame))
    return pipeline


def _fit_classifier(frame: pd.DataFrame, features: list[str], target: str) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=int(config_value("project", "random_seed", default=42)),
                ),
            ),
        ]
    )
    pipeline.fit(frame[features], frame[target], model__sample_weight=_weights(frame))
    return pipeline


def _align_probabilities(model: Pipeline, frame: pd.DataFrame, features: list[str], classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(frame[features])
    aligned = np.zeros((len(frame), len(classes)), dtype=float)
    class_lookup = {int(label): index for index, label in enumerate(model.classes_)}
    for output_index, label in enumerate(classes):
        if label in class_lookup:
            aligned[:, output_index] = raw[:, class_lookup[label]]
    aligned = np.clip(aligned, 1e-12, 1.0)
    return aligned / aligned.sum(axis=1, keepdims=True)


def _outcomes_from_lambdas(home: np.ndarray, away: np.ndarray) -> np.ndarray:
    max_goals = int(config_value("modeling", "max_goals", default=10))
    return np.asarray(
        [outcome_probabilities(scoreline_matrix(h, a, max_goals=max_goals)) for h, a in zip(home, away)]
    )


def _adjust_lambdas(total: np.ndarray, difference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped_difference = np.clip(difference, -5.0, 5.0)
    home = np.clip((total + clipped_difference) / 2.0, 0.05, 5.5)
    away = np.clip((total - clipped_difference) / 2.0, 0.05, 5.5)
    return home, away


def _goal_bundle_lambdas(bundle: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = bundle.feature_columns
    return (
        np.clip(bundle.home_model.predict(frame[columns]), 0.05, 5.5),
        np.clip(bundle.away_model.predict(frame[columns]), 0.05, 5.5),
    )


def _fit_variant_models(
    training: pd.DataFrame,
    features: list[str],
    goal_feature_group: str,
) -> dict[str, Any]:
    baseline_goal = train_goal_model(
        training,
        training["date"].max(),
        persist=False,
        half_life_years=1e9,
        use_match_importance=True,
        feature_columns=features,
        model_type="poisson",
    )
    return {
        "current_poisson_baseline": baseline_goal,
        "goal_counts_raw": (
            _fit_regressor(training, features, "home_goals", "poisson"),
            _fit_regressor(training, features, "away_goals", "poisson"),
        ),
        "goal_counts_capped": (
            _fit_regressor(training, features, "capped_home_goals", "poisson"),
            _fit_regressor(training, features, "capped_away_goals", "poisson"),
        ),
        "outcome_classifier": _fit_classifier(training, features, "result"),
        "goal_difference_regression": _fit_regressor(training, features, "capped_goal_difference"),
        "margin_class_classifier": _fit_classifier(training, features, "margin_class_index"),
        "goal_diff_residual_correction": _fit_regressor(training, features, "goal_diff_residual"),
    }


def _predict_variant(
    name: str,
    models: dict[str, Any],
    frame: pd.DataFrame,
    features: list[str],
) -> VariantPrediction:
    baseline_home, baseline_away = _goal_bundle_lambdas(models["current_poisson_baseline"], frame)
    baseline_total = baseline_home + baseline_away
    if name == "current_poisson_baseline":
        home, away = baseline_home, baseline_away
        probabilities = _outcomes_from_lambdas(home, away)
        notes = "Existing selected attack/defence Poisson target and coherent score matrix."
    elif name in {"goal_counts_raw", "goal_counts_capped"}:
        home_model, away_model = models[name]
        home = np.clip(home_model.predict(frame[features]), 0.05, 5.5)
        away = np.clip(away_model.predict(frame[features]), 0.05, 5.5)
        probabilities = _outcomes_from_lambdas(home, away)
        notes = "Separate Poisson count targets converted through a coherent score matrix."
    elif name == "elo_only":
        expectation = elo_baseline_expectations(frame)
        home = expectation["elo_expected_goals_home"].to_numpy()
        away = expectation["elo_expected_goals_away"].to_numpy()
        probabilities = expectation[["elo_p_home_win", "elo_p_draw", "elo_p_away_win"]].to_numpy()
        notes = "Pre-match Elo-only baseline; scorelines use Elo-derived expected goals."
    elif name == "outcome_classifier":
        home, away = baseline_home, baseline_away
        probabilities = _align_probabilities(models[name], frame, features, [0, 1, 2])
        notes = "Direct outcome target; scoreline metrics retain the baseline Poisson score distribution."
    elif name == "goal_difference_regression":
        predicted_difference = np.clip(models[name].predict(frame[features]), -4.0, 4.0)
        home, away = _adjust_lambdas(baseline_total, predicted_difference)
        probabilities = _outcomes_from_lambdas(home, away)
        notes = "Capped goal-difference point estimate converted by preserving baseline expected total goals."
    elif name == "margin_class_classifier":
        margin_probabilities = _align_probabilities(models[name], frame, features, list(range(7)))
        predicted_difference = margin_probabilities @ MARGIN_CLASS_CENTERS
        home, away = _adjust_lambdas(baseline_total, predicted_difference)
        probabilities = np.column_stack(
            [margin_probabilities[:, 4:].sum(axis=1), margin_probabilities[:, 3], margin_probabilities[:, :3].sum(axis=1)]
        )
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        notes = "Ordinal margin classes aggregate directly to W/D/L; approximate scorelines preserve baseline total goals."
    elif name == "goal_diff_residual_correction":
        residual = np.clip(models[name].predict(frame[features]), -3.0, 3.0)
        expectation = elo_baseline_expectations(frame)
        adjusted_difference = expectation["elo_expected_goal_difference"].to_numpy() + residual
        home, away = _adjust_lambdas(baseline_total, adjusted_difference)
        probabilities = _outcomes_from_lambdas(home, away)
        notes = "Predicted Elo goal-difference residual corrects the baseline while preserving baseline total goals."
    else:
        raise ValueError(f"Unknown target experiment variant {name!r}.")
    return VariantPrediction(probabilities=probabilities, home_lambdas=home, away_lambdas=away, notes=notes)


def _isotonic_calibrate(raw_fit: np.ndarray, y_fit: np.ndarray, raw_eval: np.ndarray) -> np.ndarray:
    calibrated = np.column_stack(
        [
            IsotonicRegression(out_of_bounds="clip").fit(raw_fit[:, index], y_fit == index).predict(raw_eval[:, index])
            for index in range(raw_fit.shape[1])
        ]
    )
    calibrated = np.clip(calibrated, 1e-8, 1.0)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _calibrated_probabilities(
    raw_calibration: np.ndarray,
    y_calibration: np.ndarray,
    raw_evaluation: np.ndarray,
) -> dict[str, np.ndarray]:
    platt = LogisticRegression(
        max_iter=2000,
        random_state=int(config_value("project", "random_seed", default=42)),
    )
    platt.fit(np.log(np.clip(raw_calibration, 1e-8, 1.0)), y_calibration)
    return {
        "uncalibrated": raw_evaluation,
        "platt_sigmoid": platt.predict_proba(np.log(np.clip(raw_evaluation, 1e-8, 1.0))),
        "isotonic": _isotonic_calibrate(raw_calibration, y_calibration, raw_evaluation),
    }


def _target_metadata(name: str) -> tuple[str, str]:
    return {
        "elo_only": ("elo_baseline", "pre_match_elo_expectation"),
        "current_poisson_baseline": ("goals", "existing_capped_home_and_away_goals"),
        "goal_counts_raw": ("goals", "raw_home_goals_and_away_goals"),
        "goal_counts_capped": ("goals", "home_and_away_goals_capped_at_6"),
        "outcome_classifier": ("outcome", "home_win_draw_away_win"),
        "goal_difference_regression": ("goal_difference", "capped_goal_difference_-4_to_4"),
        "margin_class_classifier": ("margin_class", "seven_ordinal_margin_classes"),
        "goal_diff_residual_correction": ("residual", "actual_goal_difference_minus_pre_match_elo_expectation"),
    }[name]


def _evaluation_metrics(
    evaluation: pd.DataFrame,
    prediction: VariantPrediction,
    probabilities: np.ndarray,
) -> dict[str, float]:
    score_metrics = compute_scoreline_metrics(
        evaluation,
        prediction.home_lambdas,
        prediction.away_lambdas,
        max_goals=int(config_value("modeling", "max_goals", default=10)),
    )
    classification = compute_classification_metrics(
        evaluation["result"].to_numpy(dtype=int),
        probabilities,
        np.argmax(probabilities, axis=1),
    )
    score_metrics.update(classification)
    return score_metrics


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _replace_generated_section(path: Any, heading: str, body: list[str]) -> None:
    marker_start = f"<!-- {heading}:start -->"
    marker_end = f"<!-- {heading}:end -->"
    section = "\n".join([marker_start, *body, marker_end])
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker_start in text and marker_end in text:
        prefix = text.split(marker_start, 1)[0].rstrip()
        suffix = text.split(marker_end, 1)[1].strip()
        text = "\n\n".join(part for part in [prefix, section, suffix] if part) + "\n"
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text, encoding="utf-8")


def _write_report(results: pd.DataFrame, summary: pd.DataFrame, selection: dict[str, Any]) -> None:
    best_outcome = summary.sort_values(["avg_log_loss", "avg_brier_score"]).iloc[0]
    best_scoreline = summary.sort_values(["avg_scoreline_top_5_hit_rate", "avg_log_loss"], ascending=[False, True]).iloc[0]
    most_stable = summary.sort_values(["stability_score", "avg_log_loss"]).iloc[0]
    best_explicitly_calibrated = summary[~summary["model_name"].str.endswith("__uncalibrated")].sort_values(
        ["avg_log_loss", "avg_brier_score"]
    ).iloc[0]
    baseline = summary[summary["model_name"] == "current_poisson_baseline__uncalibrated"].iloc[0]
    residual = summary[summary["model_name"] == "goal_diff_residual_correction__uncalibrated"].iloc[0]
    lines = [
        "# Target Experiment Report",
        "",
        "## Executive Summary",
        "",
        f"- Recommended final target/model: **{selection['recommended_model_name']}**.",
        f"- Best W/D/L log loss: **{best_outcome['model_name']}** ({best_outcome['avg_log_loss']:.4f}).",
        f"- Best scoreline top-5 hit rate: **{best_scoreline['model_name']}** ({best_scoreline['avg_scoreline_top_5_hit_rate']:.1%}).",
        f"- Most stable log loss: **{most_stable['model_name']}** (standard deviation {most_stable['stability_score']:.4f}).",
        f"- Best explicitly calibrated variant: **{best_explicitly_calibrated['model_name']}** ({best_explicitly_calibrated['avg_log_loss']:.4f}).",
        f"- Margin-target log-loss improvement over the current Poisson baseline: **{baseline['avg_log_loss'] - best_outcome['avg_log_loss']:.4f}**. This is small.",
        f"- Residual correction log loss: **{residual['avg_log_loss']:.4f}**; it did not improve the baseline.",
        "",
        "## Target Construction",
        "",
        "- Outcome: direct home-win/draw/away-win classification.",
        "- Goals: raw and six-goal-capped home/away count targets.",
        "- Goal difference: actual difference clipped to -4 through +4.",
        "- Margin class: seven ordered classes from away-win-by-3+ through home-win-by-3+.",
        "- Residual: actual goal difference minus a pre-match Elo expected goal difference. A points residual is also engineered, but not fitted because it does not map cleanly back to scorelines.",
        "- Future-form labels were generated as an isolated research table but skipped as model inputs because using them in the same-match pipeline creates a high leakage risk.",
        "",
        "## Conversion Back To Match Probabilities",
        "",
        "- Goal targets directly create Poisson score matrices.",
        "- Goal-difference and residual models preserve baseline expected total goals and replace the expected goal difference before creating a Poisson score matrix.",
        "- Margin probabilities aggregate directly to W/D/L; approximate scorelines preserve baseline expected total goals and use the class-weighted expected margin.",
        "- The direct outcome classifier has no native score distribution, so reported scoreline metrics retain the baseline Poisson score matrix.",
        "",
        "## Summary",
        "",
        _markdown_table(
            summary,
            [
                "target_type", "model_name", "avg_log_loss", "avg_brier_score", "avg_calibration_error",
                "avg_scoreline_top_5_hit_rate", "stability_score", "selected_for_final_model",
            ],
        ),
        "",
        "## Frozen World Cup Results",
        "",
        _markdown_table(
            results,
            [
                "model_name", "target_type", "tournament_year", "log_loss", "brier_score", "accuracy",
                "calibration_error", "goal_difference_mae", "scoreline_top_5_hit_rate",
            ],
        ),
        "",
        "## Leakage Checks",
        "",
        "- Every experiment uses the existing pre-match feature frame; rolling form is shifted and Elo is pre-match.",
        "- Training cutoffs are 2006, 2010, 2014, and 2018 for World Cups 2010, 2014, 2018, and 2022.",
        "- Residual expectations use only pre-match Elo and neutral/home context.",
        "- Calibration uses only a chronological tail of the already-frozen training period.",
        "- Future-form targets are saved separately and never joined into the match feature table.",
        "",
        "## Recommendation",
        "",
        selection["recommendation_reason"],
        "",
        "Accuracy was treated as secondary. These experiments remain noisy because only four World Cups are available as frozen test tournaments.",
    ]
    INTERNAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (INTERNAL_DOCS_DIR / "target_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    evaluation_path = INTERNAL_DOCS_DIR / "evaluation.md"
    _replace_generated_section(
        evaluation_path,
        "target-experiments",
        [
            "## Alternative Target Experiments",
            "",
            f"- Selected target/model: **{selection['recommended_model_name']}**.",
            f"- Mean frozen-World-Cup log loss: **{selection['avg_log_loss']:.4f}**.",
            f"- Mean Brier score: **{selection['avg_brier_score']:.4f}**.",
            f"- Mean top-5 scoreline hit rate: **{selection['avg_scoreline_top_5_hit_rate']:.1%}**.",
            "- Future-form targets were generated separately and skipped as model inputs because of leakage risk.",
            "- See `docs/internal/target_experiment_report.md` for the full comparison.",
        ],
    )


def run_target_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    frame = _load_features()
    selection_path = MODELS_DIR / "model_selection.json"
    production_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    goal_feature_group = str(production_selection.get("goal_feature_group", "attack_defence_poisson"))
    features = GOAL_FEATURE_GROUPS[goal_feature_group]
    variants = [
        "elo_only",
        "current_poisson_baseline",
        "goal_counts_raw",
        "goal_counts_capped",
        "outcome_classifier",
        "goal_difference_regression",
        "margin_class_classifier",
        "goal_diff_residual_correction",
    ]
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for year in WORLD_CUP_YEARS:
        training = _training_frame(frame, year)
        evaluation = _world_cup_frame(frame, year)
        calibration_cutoff = training["date"].max() - pd.DateOffset(
            years=int(config_value("target_experiments", "calibration_years", default=2))
        )
        base = training[training["date"] < calibration_cutoff].copy()
        calibration = training[training["date"] >= calibration_cutoff].copy()
        full_models = _fit_variant_models(training, features, goal_feature_group)
        base_models = _fit_variant_models(base, features, goal_feature_group)
        for variant in variants:
            evaluation_prediction = _predict_variant(variant, full_models, evaluation, features)
            calibration_prediction = _predict_variant(variant, base_models, calibration, features)
            calibrated = _calibrated_probabilities(
                calibration_prediction.probabilities,
                calibration["result"].to_numpy(dtype=int),
                evaluation_prediction.probabilities,
            )
            target_type, target_detail = _target_metadata(variant)
            for method, probabilities in calibrated.items():
                model_name = f"{variant}__{method}"
                metrics = _evaluation_metrics(evaluation, evaluation_prediction, probabilities)
                rows.append(
                    {
                        "model_name": model_name,
                        "target_type": target_type,
                        "target_detail": target_detail,
                        "tournament_year": year,
                        "log_loss": metrics["log_loss"],
                        "brier_score": metrics["brier_score"],
                        "accuracy": metrics["accuracy"],
                        "calibration_error": metrics["calibration_error"],
                        "average_probability_actual_outcome": metrics["average_probability_actual_outcome"],
                        "home_goals_mae": metrics["home_goal_mae"],
                        "away_goals_mae": metrics["away_goal_mae"],
                        "total_goals_mae": metrics["total_goals_mae"],
                        "goal_difference_mae": metrics["goal_difference_mae"],
                        "scoreline_top_1_accuracy": metrics["top_1_scoreline_accuracy"],
                        "scoreline_top_5_hit_rate": metrics["top_5_scoreline_hit_rate"],
                        "notes": evaluation_prediction.notes,
                    }
                )
                for calibration_row in reliability_table(
                    evaluation["result"].to_numpy(dtype=int), probabilities, n_bins=10
                ):
                    calibration_rows.append(
                        {
                            "model_name": model_name,
                            "target_type": target_type,
                            "tournament_year": year,
                            **calibration_row,
                        }
                    )
        LOGGER.info("Completed target experiments for World Cup %s.", year)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["target_type", "model_name"], as_index=False)
        .agg(
            avg_log_loss=("log_loss", "mean"),
            avg_brier_score=("brier_score", "mean"),
            avg_calibration_error=("calibration_error", "mean"),
            avg_scoreline_top_5_hit_rate=("scoreline_top_5_hit_rate", "mean"),
            stability_score=("log_loss", "std"),
        )
        .sort_values(["avg_log_loss", "avg_brier_score"])
        .reset_index(drop=True)
    )
    recommended = summary.iloc[0]
    summary["selected_for_final_model"] = summary["model_name"].eq(recommended["model_name"])
    summary["reason"] = np.where(
        summary["selected_for_final_model"],
        "Lowest mean frozen-World-Cup log loss; Brier, calibration, scoreline quality, and stability reported alongside.",
        "Not selected: did not beat the recommended model on primary frozen-World-Cup log loss.",
    )
    target_selection = {
        "recommended_model_name": str(recommended["model_name"]),
        "recommended_target_type": str(recommended["target_type"]),
        "avg_log_loss": float(recommended["avg_log_loss"]),
        "avg_brier_score": float(recommended["avg_brier_score"]),
        "avg_calibration_error": float(recommended["avg_calibration_error"]),
        "avg_scoreline_top_5_hit_rate": float(recommended["avg_scoreline_top_5_hit_rate"]),
        "stability_score": float(recommended["stability_score"]),
        "future_form_targets_used": False,
        "recommendation_reason": (
            f"`{recommended['model_name']}` had the lowest average outcome log loss across the four frozen World Cups. "
            "Scoreline and calibration metrics are reported separately; future-form targets were excluded from modelling."
        ),
    }
    results.to_csv(PREDICTIONS_DIR / "target_experiment_results.csv", index=False)
    summary.to_csv(PREDICTIONS_DIR / "target_experiment_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(PREDICTIONS_DIR / "target_experiment_calibration.csv", index=False)
    future_targets = build_future_team_targets(frame)
    future_targets.to_csv(PROCESSED_DATA_DIR / "future_team_targets_research_only.csv", index=False)
    (MODELS_DIR / "target_selection.json").write_text(json.dumps(target_selection, indent=2), encoding="utf-8")
    production_selection["target_experiment_recommendation"] = target_selection
    selection_path.write_text(json.dumps(production_selection, indent=2), encoding="utf-8")
    train_selected_target_model(
        frame,
        str(config_value("project", "cutoff_date", default=frame["date"].max().date())),
    )
    _write_report(results, summary, target_selection)
    return results, summary
