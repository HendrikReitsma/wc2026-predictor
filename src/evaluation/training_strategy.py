from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_classification_metrics, compute_scoreline_metrics
from src.features.dynamic_rating import DynamicRatingState, build_rating_features
from src.features.target_engineering import MARGIN_CLASS_CENTERS, add_match_targets
from src.models.scoreline import outcome_probabilities, scoreline_matrix
from src.models.train_goal_model import GOAL_FEATURE_GROUPS
from src.models.weighting import combined_sample_weights
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import INTERNAL_DOCS_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)
WORLD_CUP_YEARS = [2010, 2014, 2018, 2022]
TRAINING_CUTOFFS = {2010: 2006, 2014: 2010, 2018: 2014, 2022: 2018}


@dataclass(frozen=True)
class StrategyConfig:
    minimum_year: int = 1990
    half_life_years: float = 1e9
    importance_profile: str = "aggressive"
    goal_cap: int = 8
    rating_model: str = "standard_elo"
    elo_k_scale: float = 1.0
    draw_correction: str = "none"
    calibration_method: str = "uncalibrated"


@dataclass
class FittedPipeline:
    home_goal_model: Pipeline
    away_goal_model: Pipeline
    margin_model: Pipeline
    features: list[str]


def _load_features() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    return add_match_targets(frame)


def _half_life(value: Any) -> float:
    return 1e9 if str(value).lower() == "none" else float(value)


def _weights(frame: pd.DataFrame, config: StrategyConfig) -> np.ndarray:
    use_importance = config.importance_profile != "none"
    return combined_sample_weights(
        frame,
        frame["date"].max(),
        half_life_years=config.half_life_years,
        use_match_importance=use_importance,
        importance_profile=config.importance_profile,
    )


def _goal_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.05, max_iter=2000)),
        ]
    )


def _margin_pipeline() -> Pipeline:
    return Pipeline(
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


def _fit_pipeline(training: pd.DataFrame, config: StrategyConfig, features: list[str]) -> FittedPipeline:
    working = training.copy()
    working["home_goal_target"] = working["home_score"].clip(lower=0, upper=config.goal_cap)
    working["away_goal_target"] = working["away_score"].clip(lower=0, upper=config.goal_cap)
    weights = _weights(working, config)
    home = _goal_pipeline()
    away = _goal_pipeline()
    margin = _margin_pipeline()
    home.fit(working[features], working["home_goal_target"], model__sample_weight=weights)
    away.fit(working[features], working["away_goal_target"], model__sample_weight=weights)
    margin.fit(working[features], working["margin_class_index"], model__sample_weight=weights)
    return FittedPipeline(home, away, margin, features)


def _aligned_margin_probabilities(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[features])
    aligned = np.zeros((len(frame), 7), dtype=float)
    for raw_index, label in enumerate(model.classes_):
        aligned[:, int(label)] = raw[:, raw_index]
    aligned = np.clip(aligned, 1e-12, 1.0)
    return aligned / aligned.sum(axis=1, keepdims=True)


def _predict_pipeline(model: FittedPipeline, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    home_base = np.clip(model.home_goal_model.predict(frame[model.features]), 0.05, 5.5)
    away_base = np.clip(model.away_goal_model.predict(frame[model.features]), 0.05, 5.5)
    margin = _aligned_margin_probabilities(model.margin_model, frame, model.features)
    difference = margin @ MARGIN_CLASS_CENTERS
    total = home_base + away_base
    home = np.clip((total + difference) / 2.0, 0.05, 5.5)
    away = np.clip((total - difference) / 2.0, 0.05, 5.5)
    margin_outcome = np.column_stack([margin[:, 4:].sum(axis=1), margin[:, 3], margin[:, :3].sum(axis=1)])
    poisson_outcome = np.asarray(
        [outcome_probabilities(scoreline_matrix(h, a, max_goals=10)) for h, a in zip(home_base, away_base)]
    )
    return {
        "home": home,
        "away": away,
        "margin_probabilities": margin_outcome / margin_outcome.sum(axis=1, keepdims=True),
        "poisson_probabilities": poisson_outcome,
    }


def _metrics(evaluation: pd.DataFrame, prediction: dict[str, np.ndarray], probabilities: np.ndarray) -> dict[str, float]:
    score = compute_scoreline_metrics(evaluation, prediction["home"], prediction["away"], max_goals=10)
    score.update(
        compute_classification_metrics(
            evaluation["result"].to_numpy(dtype=int),
            probabilities,
            np.argmax(probabilities, axis=1),
        )
    )
    return score


def _prepare_rating_frames(frame: pd.DataFrame, candidates: list[tuple[str, float]]) -> dict[tuple[str, float], tuple[pd.DataFrame, DynamicRatingState]]:
    frames: dict[tuple[str, float], tuple[pd.DataFrame, DynamicRatingState]] = {}
    for model_name, k_scale in candidates:
        frames[(model_name, k_scale)] = build_rating_features(frame, model_name=model_name, k_scale=k_scale)
    return frames


def _evaluate_config(
    frame: pd.DataFrame,
    config: StrategyConfig,
    search_stage: str,
    features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in WORLD_CUP_YEARS:
        training = frame[
            (frame["date"].dt.year >= config.minimum_year)
            & (frame["date"].dt.year <= TRAINING_CUTOFFS[year])
        ].copy()
        evaluation = frame[(frame["date"].dt.year == year) & frame["is_world_cup"].fillna(0).astype(bool)].copy()
        if len(training) < 100 or evaluation.empty:
            continue
        model = _fit_pipeline(training, config, features)
        prediction = _predict_pipeline(model, evaluation)
        metrics = _metrics(evaluation, prediction, prediction["margin_probabilities"])
        rows.append(
            {
                "search_stage": search_stage,
                **config.__dict__,
                "tournament_year": year,
                "training_cutoff_year": TRAINING_CUTOFFS[year],
                **metrics,
            }
        )
    return rows


def _summarize(details: pd.DataFrame) -> pd.DataFrame:
    config_columns = [
        "search_stage", "minimum_year", "half_life_years", "importance_profile", "goal_cap",
        "rating_model", "elo_k_scale", "draw_correction", "calibration_method",
    ]
    summary = (
        details.groupby(config_columns, as_index=False)
        .agg(
            world_cups_covered=("tournament_year", "nunique"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier_score=("brier_score", "mean"),
            avg_calibration_error=("calibration_error", "mean"),
            avg_scoreline_top_5_hit_rate=("top_5_scoreline_hit_rate", "mean"),
            stability_score=("log_loss", "std"),
        )
        .sort_values(["world_cups_covered", "avg_log_loss"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def _best_complete(summary: pd.DataFrame) -> pd.Series:
    complete = summary[summary["world_cups_covered"] == len(WORLD_CUP_YEARS)]
    if complete.empty:
        raise ValueError("No strategy configuration covered all four frozen World Cups.")
    return complete.sort_values(
        [
            "avg_log_loss",
            "avg_brier_score",
            "avg_calibration_error",
            "avg_scoreline_top_5_hit_rate",
            "stability_score",
        ],
        ascending=[True, True, True, False, True],
    ).iloc[0]


def _draw_adjust(probabilities: np.ndarray, elo_diff: np.ndarray, method: str, alpha: float) -> np.ndarray:
    if method == "none":
        return probabilities
    scale = float(config_value("training_strategy", "draw_similarity_scale", default=180))
    multiplier = np.full(len(probabilities), alpha)
    if method == "similar_strength":
        multiplier = 1.0 + (alpha - 1.0) * np.exp(-np.abs(elo_diff) / scale)
    adjusted = probabilities.copy()
    adjusted[:, 1] *= multiplier
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _fit_draw_alpha(probabilities: np.ndarray, y_true: np.ndarray, elo_diff: np.ndarray, method: str) -> float:
    if method == "none":
        return 1.0
    candidates = np.linspace(0.65, 1.45, 33)
    return float(
        min(
            candidates,
            key=lambda alpha: log_loss(
                y_true,
                _draw_adjust(probabilities, elo_diff, method, float(alpha)),
                labels=[0, 1, 2],
            ),
        )
    )


def _calibrate(
    fit_probabilities: np.ndarray,
    fit_y: np.ndarray,
    evaluation_probabilities: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "uncalibrated":
        return evaluation_probabilities
    if method == "sigmoid":
        calibrator = LogisticRegression(
            max_iter=2000,
            random_state=int(config_value("project", "random_seed", default=42)),
        )
        calibrator.fit(np.log(np.clip(fit_probabilities, 1e-8, 1.0)), fit_y)
        return calibrator.predict_proba(np.log(np.clip(evaluation_probabilities, 1e-8, 1.0)))
    if method == "isotonic":
        calibrated = np.column_stack(
            [
                IsotonicRegression(out_of_bounds="clip")
                .fit(fit_probabilities[:, index], fit_y == index)
                .predict(evaluation_probabilities[:, index])
                for index in range(3)
            ]
        )
        calibrated = np.clip(calibrated, 1e-8, 1.0)
        return calibrated / calibrated.sum(axis=1, keepdims=True)
    raise ValueError(f"Unsupported calibration method {method!r}.")


def _calibration_comparison(
    frame: pd.DataFrame,
    config: StrategyConfig,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    draw_methods = list(config_value("training_strategy", "draw_correction_methods", default=["none", "global", "similar_strength"]))
    calibration_methods = list(config_value("training_strategy", "calibration_methods", default=["uncalibrated", "sigmoid", "isotonic"]))
    calibration_years = int(config_value("training_strategy", "calibration_years", default=2))
    similar_threshold = float(config_value("training_strategy", "similar_strength_elo_threshold", default=100))
    for year in WORLD_CUP_YEARS:
        frozen = frame[
            (frame["date"].dt.year >= config.minimum_year)
            & (frame["date"].dt.year <= TRAINING_CUTOFFS[year])
        ].copy()
        split = frozen["date"].max() - pd.DateOffset(years=calibration_years)
        training = frozen[frozen["date"] < split].copy()
        calibration = frozen[frozen["date"] >= split].copy()
        evaluation = frame[(frame["date"].dt.year == year) & frame["is_world_cup"].fillna(0).astype(bool)].copy()
        model = _fit_pipeline(training, config, features)
        calibration_prediction = _predict_pipeline(model, calibration)
        evaluation_prediction = _predict_pipeline(model, evaluation)
        for source in ["margin_probabilities", "poisson_probabilities"]:
            raw_fit = calibration_prediction[source]
            raw_evaluation = evaluation_prediction[source]
            for draw_method in draw_methods:
                alpha = _fit_draw_alpha(
                    raw_fit,
                    calibration["result"].to_numpy(dtype=int),
                    calibration["elo_diff"].to_numpy(dtype=float),
                    draw_method,
                )
                draw_fit = _draw_adjust(raw_fit, calibration["elo_diff"].to_numpy(dtype=float), draw_method, alpha)
                draw_evaluation = _draw_adjust(
                    raw_evaluation,
                    evaluation["elo_diff"].to_numpy(dtype=float),
                    draw_method,
                    alpha,
                )
                for method in calibration_methods:
                    probabilities = _calibrate(
                        draw_fit,
                        calibration["result"].to_numpy(dtype=int),
                        draw_evaluation,
                        method,
                    )
                    metrics = _metrics(evaluation, evaluation_prediction, probabilities)
                    similar = np.abs(evaluation["elo_diff"].to_numpy(dtype=float)) <= similar_threshold
                    if similar.any():
                        similar_y = evaluation.loc[similar, "result"].to_numpy(dtype=int)
                        similar_prob = probabilities[similar]
                        similar_log_loss = float(log_loss(similar_y, similar_prob, labels=[0, 1, 2]))
                        similar_draw_brier = float(np.mean(((similar_y == 1).astype(float) - similar_prob[:, 1]) ** 2))
                    else:
                        similar_log_loss = np.nan
                        similar_draw_brier = np.nan
                    rows.append(
                        {
                            "probability_source": source.replace("_probabilities", ""),
                            "draw_correction": draw_method,
                            "draw_alpha": alpha,
                            "calibration_method": method,
                            "tournament_year": year,
                            "similar_strength_matches": int(similar.sum()),
                            "similar_strength_log_loss": similar_log_loss,
                            "similar_strength_draw_brier": similar_draw_brier,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def _calibration_summary(details: pd.DataFrame) -> pd.DataFrame:
    return (
        details.groupby(["probability_source", "draw_correction", "calibration_method"], as_index=False)
        .agg(
            avg_draw_alpha=("draw_alpha", "mean"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier_score=("brier_score", "mean"),
            avg_calibration_error=("calibration_error", "mean"),
            avg_scoreline_top_5_hit_rate=("top_5_scoreline_hit_rate", "mean"),
            stability_score=("log_loss", "std"),
            avg_similar_strength_log_loss=("similar_strength_log_loss", "mean"),
            avg_similar_strength_draw_brier=("similar_strength_draw_brier", "mean"),
        )
        .sort_values(["probability_source", "avg_log_loss"])
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write_report(
    strategy_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    rating_summary: pd.DataFrame,
    selection: dict[str, Any],
) -> None:
    window = strategy_summary[strategy_summary["search_stage"] == "window_decay_importance"]
    best_window = window.sort_values(["world_cups_covered", "avg_log_loss"], ascending=[False, True]).iloc[0]
    no_weight = window[window["importance_profile"] == "none"].sort_values(["world_cups_covered", "avg_log_loss"], ascending=[False, True]).iloc[0]
    calibrated = calibration_summary[calibration_summary["probability_source"] == "margin"].sort_values("avg_log_loss")
    raw_poisson = calibration_summary[
        (calibration_summary["probability_source"] == "poisson")
        & (calibration_summary["draw_correction"] == "none")
        & (calibration_summary["calibration_method"] == "uncalibrated")
    ].iloc[0]
    best_poisson_draw = calibration_summary[calibration_summary["probability_source"] == "poisson"].sort_values("avg_log_loss").iloc[0]
    standard_rating = rating_summary[
        (rating_summary["rating_model"] == "standard_elo") & np.isclose(rating_summary["elo_k_scale"], 1.0)
    ].iloc[0]
    best_dynamic = rating_summary[rating_summary["rating_model"] == "smoothed_dynamic"].sort_values("avg_log_loss").iloc[0]
    lines = [
        "# Training Strategy Report",
        "",
        "## Recommended Configuration",
        "",
        f"- Minimum training year: **{selection['minimum_year']}**.",
        f"- Time-decay half-life: **{selection['time_decay_half_life_years'] or 'none'}**.",
        f"- Match-importance profile: **{selection['importance_profile']}**.",
        f"- Goal cap: **{selection['goal_cap']}**.",
        f"- Rating model: **{selection['rating_model']}**, Elo K scale **{selection['elo_k_scale']}**.",
        f"- Draw correction: **{selection['draw_correction']}**.",
        f"- Calibration: **{selection['calibration_method']}**.",
        f"- Base-pipeline frozen-World-Cup mean log loss: **{selection['base_pipeline_avg_log_loss']:.4f}**.",
        f"- Draw/calibration comparison mean log loss: **{selection['draw_calibration_stage_avg_log_loss']:.4f}**.",
        "",
        "The search was staged rather than a huge unconstrained Cartesian sweep: window/decay/importance first, then goal cap, then rating update, then draw correction and calibration. This reduces overfitting risk with only four World Cups. The draw/calibration stage reserves a chronological two-year calibration tail, so its absolute score is not directly comparable with the base-pipeline score fitted on the full frozen training window.",
        "",
        "## Findings",
        "",
        f"- Best training window: **{int(best_window['minimum_year'])} onward**.",
        f"- Best time decay: **{'none' if best_window['half_life_years'] >= 1e8 else str(best_window['half_life_years']) + ' years'}**.",
        f"- Match weighting {'helped' if best_window['avg_log_loss'] < no_weight['avg_log_loss'] else 'did not help'}: best weighted log loss {best_window['avg_log_loss']:.4f}, best unweighted {no_weight['avg_log_loss']:.4f}.",
        f"- Best margin-pipeline calibration/draw choice: **{calibrated.iloc[0]['draw_correction']} + {calibrated.iloc[0]['calibration_method']}**.",
        f"- Raw Poisson log loss: **{raw_poisson['avg_log_loss']:.4f}**; best Poisson draw/calibration variant: **{best_poisson_draw['draw_correction']} + {best_poisson_draw['calibration_method']}** at **{best_poisson_draw['avg_log_loss']:.4f}**.",
        f"- Best smoothed dynamic rating log loss: **{best_dynamic['avg_log_loss']:.4f}** versus standard Elo scale 1.0 at **{standard_rating['avg_log_loss']:.4f}**.",
        "",
        "## Training Strategy Comparison",
        "",
        _markdown_table(
            strategy_summary,
            [
                "search_stage", "minimum_year", "half_life_years", "importance_profile", "goal_cap",
                "rating_model", "elo_k_scale", "world_cups_covered", "avg_log_loss", "avg_brier_score",
                "avg_calibration_error", "avg_scoreline_top_5_hit_rate", "stability_score", "selected",
            ],
        ),
        "",
        "## Calibration And Draw Correction",
        "",
        _markdown_table(
            calibration_summary,
            [
                "probability_source", "draw_correction", "calibration_method", "avg_draw_alpha",
                "avg_log_loss", "avg_brier_score", "avg_calibration_error", "avg_similar_strength_log_loss",
                "avg_similar_strength_draw_brier", "stability_score",
            ],
        ),
        "",
        "## Rating Model Comparison",
        "",
        _markdown_table(
            rating_summary,
            [
                "rating_model", "elo_k_scale", "avg_log_loss", "avg_brier_score",
                "avg_calibration_error", "avg_scoreline_top_5_hit_rate", "stability_score", "beats_standard_elo",
            ],
        ),
        "",
        "## Limitations",
        "",
        "- No new feature groups were added; rating alternatives replace the existing Elo columns only.",
        "- Four World Cups are a small model-selection sample, so tiny differences should not be overinterpreted.",
        "- The smoothed dynamic rating is Glicko-style, not a complete implementation of official Glicko-2.",
        "- Calibration and draw correction use chronological calibration tails only.",
    ]
    INTERNAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (INTERNAL_DOCS_DIR / "training_strategy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training_strategy_search() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    frame = _load_features()
    selection_path = MODELS_DIR / "model_selection.json"
    model_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    feature_group = str(model_selection.get("goal_feature_group", "attack_defence_poisson"))
    features = GOAL_FEATURE_GROUPS[feature_group]
    minimum_years = [int(value) for value in config_value("training_strategy", "minimum_year_candidates", default=[1990, 1998, 2002, 2010])]
    half_lives = [_half_life(value) for value in config_value("training_strategy", "time_decay_candidates", default=["none", 4, 8, 12])]
    importance_profiles = list(config_value("training_strategy", "importance_profiles", default=["none", "balanced", "aggressive"]))
    goal_caps = [int(value) for value in config_value("training_strategy", "goal_cap_candidates", default=[6, 8, 10])]
    k_scales = [float(value) for value in config_value("training_strategy", "elo_k_scale_candidates", default=[0.8, 1.0, 1.2])]
    rating_models = list(config_value("training_strategy", "rating_models", default=["standard_elo", "smoothed_dynamic"]))

    rating_candidates = sorted(set([("standard_elo", 1.0), *product(rating_models, k_scales)]))
    rating_frames = _prepare_rating_frames(frame, rating_candidates)
    standard_frame = rating_frames[("standard_elo", 1.0)][0]

    detail_rows: list[dict[str, Any]] = []
    for minimum_year, half_life, profile in product(minimum_years, half_lives, importance_profiles):
        config = StrategyConfig(minimum_year=minimum_year, half_life_years=half_life, importance_profile=profile)
        detail_rows.extend(_evaluate_config(standard_frame, config, "window_decay_importance", features))
    stage_summary = _summarize(pd.DataFrame(detail_rows))
    best = _best_complete(stage_summary[stage_summary["search_stage"] == "window_decay_importance"])
    base_config = StrategyConfig(
        minimum_year=int(best["minimum_year"]),
        half_life_years=float(best["half_life_years"]),
        importance_profile=str(best["importance_profile"]),
    )
    LOGGER.info("Completed window/decay/importance search; current best %s.", base_config)

    for goal_cap in goal_caps:
        config = StrategyConfig(**{**base_config.__dict__, "goal_cap": goal_cap})
        detail_rows.extend(_evaluate_config(standard_frame, config, "goal_cap", features))
    stage_summary = _summarize(pd.DataFrame(detail_rows))
    best_cap = _best_complete(stage_summary[stage_summary["search_stage"] == "goal_cap"])
    cap_config = StrategyConfig(**{**base_config.__dict__, "goal_cap": int(best_cap["goal_cap"])})

    rating_detail_rows: list[dict[str, Any]] = []
    for rating_model, k_scale in rating_candidates:
        config = StrategyConfig(**{**cap_config.__dict__, "rating_model": rating_model, "elo_k_scale": k_scale})
        rows = _evaluate_config(rating_frames[(rating_model, k_scale)][0], config, "rating_model", features)
        detail_rows.extend(rows)
        rating_detail_rows.extend(rows)
    stage_summary = _summarize(pd.DataFrame(detail_rows))
    best_rating = _best_complete(stage_summary[stage_summary["search_stage"] == "rating_model"])
    rating_config = StrategyConfig(
        **{
            **cap_config.__dict__,
            "rating_model": str(best_rating["rating_model"]),
            "elo_k_scale": float(best_rating["elo_k_scale"]),
        }
    )
    selected_frame, selected_rating_state = rating_frames[(rating_config.rating_model, rating_config.elo_k_scale)]
    LOGGER.info("Completed goal-cap and rating search; current best %s.", rating_config)

    calibration_details = _calibration_comparison(selected_frame, rating_config, features)
    calibration_summary = _calibration_summary(calibration_details)
    margin_calibration = calibration_summary[calibration_summary["probability_source"] == "margin"].sort_values(
        ["avg_log_loss", "avg_brier_score", "stability_score"]
    )
    best_calibration = margin_calibration.iloc[0]
    final_config = StrategyConfig(
        **{
            **rating_config.__dict__,
            "draw_correction": str(best_calibration["draw_correction"]),
            "calibration_method": str(best_calibration["calibration_method"]),
        }
    )

    strategy_summary = _summarize(pd.DataFrame(detail_rows))
    strategy_summary["selected"] = (
        strategy_summary["minimum_year"].eq(final_config.minimum_year)
        & strategy_summary["half_life_years"].eq(final_config.half_life_years)
        & strategy_summary["importance_profile"].eq(final_config.importance_profile)
        & strategy_summary["goal_cap"].eq(final_config.goal_cap)
        & strategy_summary["rating_model"].eq(final_config.rating_model)
        & strategy_summary["elo_k_scale"].eq(final_config.elo_k_scale)
    )
    rating_summary = _summarize(pd.DataFrame(rating_detail_rows)).drop(columns=["search_stage", "draw_correction", "calibration_method"])
    standard_row = rating_summary[
        (rating_summary["rating_model"] == "standard_elo") & np.isclose(rating_summary["elo_k_scale"], 1.0)
    ].iloc[0]
    rating_summary["beats_standard_elo"] = rating_summary["avg_log_loss"] < float(standard_row["avg_log_loss"])

    selection = {
        "minimum_year": final_config.minimum_year,
        "time_decay_half_life_years": None if final_config.half_life_years >= 1e8 else final_config.half_life_years,
        "training_half_life_years": final_config.half_life_years,
        "importance_profile": final_config.importance_profile,
        "use_match_importance_weights": final_config.importance_profile != "none",
        "goal_cap": final_config.goal_cap,
        "rating_model": final_config.rating_model,
        "elo_k_scale": final_config.elo_k_scale,
        "draw_correction": final_config.draw_correction,
        "draw_alpha": float(best_calibration["avg_draw_alpha"]),
        "calibration_method": final_config.calibration_method,
        "base_pipeline_avg_log_loss": float(best_rating["avg_log_loss"]),
        "base_pipeline_avg_brier_score": float(best_rating["avg_brier_score"]),
        "base_pipeline_stability_score": float(best_rating["stability_score"]),
        "draw_calibration_stage_avg_log_loss": float(best_calibration["avg_log_loss"]),
        "draw_calibration_stage_avg_brier_score": float(best_calibration["avg_brier_score"]),
        "avg_log_loss": float(best_calibration["avg_log_loss"]),
        "avg_brier_score": float(best_calibration["avg_brier_score"]),
        "avg_calibration_error": float(best_calibration["avg_calibration_error"]),
        "avg_scoreline_top_5_hit_rate": float(best_calibration["avg_scoreline_top_5_hit_rate"]),
        "stability_score": float(best_calibration["stability_score"]),
        "feature_group_unchanged": feature_group,
        "selection_method": "staged rolling World Cup pipeline search",
    }
    model_selection["training_strategy_recommendation"] = selection
    model_selection["minimum_training_year"] = final_config.minimum_year
    model_selection["training_half_life_years"] = final_config.half_life_years
    model_selection["production_time_decay_half_life_years"] = selection["time_decay_half_life_years"]
    model_selection["use_match_importance_weights"] = selection["use_match_importance_weights"]
    model_selection["weighting_scheme"] = final_config.importance_profile
    model_selection["goal_cap"] = final_config.goal_cap
    model_selection["rating_model"] = final_config.rating_model
    model_selection["elo_k_scale"] = final_config.elo_k_scale
    model_selection["draw_correction"] = final_config.draw_correction
    model_selection["draw_alpha"] = selection["draw_alpha"]
    model_selection["calibration_method_selected"] = final_config.calibration_method
    selection_path.write_text(json.dumps(model_selection, indent=2), encoding="utf-8")
    (MODELS_DIR / "rating_model_state.json").write_text(json.dumps(selected_rating_state.to_dict(), indent=2), encoding="utf-8")

    strategy_summary.to_csv(PREDICTIONS_DIR / "training_strategy_comparison.csv", index=False)
    calibration_summary.to_csv(PREDICTIONS_DIR / "calibration_comparison.csv", index=False)
    rating_summary.to_csv(PREDICTIONS_DIR / "rating_model_comparison.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(PROCESSED_DATA_DIR / "training_strategy_backtests.csv", index=False)
    calibration_details.to_csv(PROCESSED_DATA_DIR / "training_strategy_calibration_backtests.csv", index=False)
    _write_report(strategy_summary, calibration_summary, rating_summary, selection)
    return strategy_summary, calibration_summary, rating_summary
