from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureState, build_single_match_features
from src.features.dynamic_rating import DynamicRatingState, apply_rating_state_to_match
from src.features.target_engineering import MARGIN_CLASS_CENTERS
from src.models.scoreline import (
    expected_goals_from_matrix,
    outcome_probabilities,
    reweight_scoreline_outcomes,
    scoreline_matrix,
    scoreline_probabilities,
    top_scorelines,
)
from src.models.train_goal_model import GOAL_FEATURE_COLUMNS
from src.models.train_outcome_model import FEATURE_COLUMNS, elo_outcome_probabilities
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)


@dataclass
class PredictionContext:
    feature_state: FeatureState
    outcome_model: Any | None
    goal_home_model: Any | None
    goal_away_model: Any | None
    outcome_weight_poisson: float
    outcome_weight_ml: float
    max_goals: int
    training_cutoff: str
    model_version: str
    selected_model_type: str
    outcome_feature_columns: tuple[str, ...]
    goal_feature_columns: tuple[str, ...]
    feature_set: str
    dixon_coles_rho: float
    target_model: Any | None
    target_model_name: str
    target_feature_columns: tuple[str, ...]
    calibration_method: str
    draw_correction: str
    draw_alpha: float
    probability_calibrator: Any | None
    rating_state: DynamicRatingState | None
    selected_indirect_variant: str
    comparison_indirect_variant: str
    selected_indirect_adjustments: dict[str, dict[str, float]]
    comparison_indirect_adjustments: dict[str, dict[str, float]]


def _load_feature_state() -> FeatureState:
    state_path = PROCESSED_DATA_DIR / "feature_state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing feature state at {state_path}. Build features before predicting."
        )
    return FeatureState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))


def _load_model(path: Path) -> Any | None:
    if not path.exists():
        return None
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_prediction_context(models_dir: Path | None = None) -> PredictionContext:
    models_dir = models_dir or MODELS_DIR
    ensure_project_dirs()
    feature_state = _load_feature_state()
    selection_path = models_dir / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    configured_type = str(config_value("modeling", "model_type", default="best_backtest"))
    selected_model_type = str(selection.get("model_name", "ensemble")) if configured_type == "best_backtest" else configured_type
    outcome_file = {
        "ml_challenger": "outcome_model_challenger.joblib",
        "ml_calibrated": "outcome_model_calibrated.joblib",
        "multinomial_logit": "outcome_model_baseline.joblib",
        "ensemble": "outcome_model_calibrated.joblib",
    }.get(selected_model_type, "outcome_model_best.joblib")
    outcome_model = _load_model(models_dir / outcome_file)
    goal_home_model = _load_model(models_dir / "goal_model_home.joblib")
    goal_away_model = _load_model(models_dir / "goal_model_away.joblib")
    metadata_path = models_dir / "outcome_model_metadata.json"
    if metadata_path.exists():
        outcome_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training_cutoff = str(outcome_metadata.get("training_cutoff", ""))
        outcome_feature_columns = tuple(outcome_metadata.get("feature_columns", FEATURE_COLUMNS))
    else:
        training_cutoff = ""
        outcome_feature_columns = tuple(FEATURE_COLUMNS)
    goal_meta_path = models_dir / "goal_model_metadata.json"
    if goal_meta_path.exists():
        goal_metadata = json.loads(goal_meta_path.read_text(encoding="utf-8"))
        training_cutoff = training_cutoff or str(goal_metadata.get("training_cutoff", ""))
        goal_feature_columns = tuple(goal_metadata.get("feature_columns", GOAL_FEATURE_COLUMNS))
    else:
        goal_feature_columns = tuple(GOAL_FEATURE_COLUMNS)
    target_meta_path = models_dir / "target_model_metadata.json"
    target_metadata = json.loads(target_meta_path.read_text(encoding="utf-8")) if target_meta_path.exists() else {}
    target_model_name = str(target_metadata.get("model_name", "current_poisson_baseline__uncalibrated"))
    target_model = _load_model(models_dir / "target_model.joblib") if target_metadata.get("artifact") else None
    probability_calibrator = (
        _load_model(models_dir / str(target_metadata["calibrator_artifact"]))
        if target_metadata.get("calibrator_artifact")
        else None
    )
    rating_state_path = models_dir / "rating_model_state.json"
    rating_state = (
        DynamicRatingState.from_dict(json.loads(rating_state_path.read_text(encoding="utf-8")))
        if rating_state_path.exists()
        else None
    )
    indirect_path = models_dir / "indirect_adjustments.json"
    indirect = json.loads(indirect_path.read_text(encoding="utf-8")) if indirect_path.exists() else {}

    return PredictionContext(
        feature_state=feature_state,
        outcome_model=outcome_model,
        goal_home_model=goal_home_model,
        goal_away_model=goal_away_model,
        outcome_weight_poisson=float(config_value("modeling", "outcome_weight_poisson", default=0.7)),
        outcome_weight_ml=float(config_value("modeling", "outcome_weight_ml", default=0.3)),
        max_goals=int(config_value("modeling", "max_goals", default=10)),
        training_cutoff=training_cutoff,
        model_version="0.1.0",
        selected_model_type=selected_model_type,
        outcome_feature_columns=outcome_feature_columns,
        goal_feature_columns=goal_feature_columns,
        feature_set=str(selection.get("selected_model_feature_set", "unknown")),
        dixon_coles_rho=float(selection.get("dixon_coles_rho", 0.0)),
        target_model=target_model,
        target_model_name=target_model_name,
        target_feature_columns=tuple(target_metadata.get("feature_columns", goal_feature_columns)),
        calibration_method=str(target_metadata.get("calibration_method", "uncalibrated")),
        draw_correction=str(target_metadata.get("draw_correction", "none")),
        draw_alpha=float(target_metadata.get("draw_alpha", 1.0)),
        probability_calibrator=probability_calibrator,
        rating_state=rating_state,
        selected_indirect_variant=str(indirect.get("selected_variant", "baseline")),
        comparison_indirect_variant=str(indirect.get("comparison_variant", "baseline")),
        selected_indirect_adjustments=indirect.get("selected_adjustments", {}),
        comparison_indirect_adjustments=indirect.get("comparison_adjustments", {}),
    )


def _blend_probabilities(poisson_probs: np.ndarray, ml_probs: np.ndarray | None, context: PredictionContext) -> np.ndarray:
    if ml_probs is None:
        return poisson_probs
    blended = context.outcome_weight_poisson * poisson_probs + context.outcome_weight_ml * ml_probs
    total = blended.sum()
    if total <= 0:
        return poisson_probs
    return blended / total


def _aligned_margin_probabilities(model: Any, features: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    raw = model.predict_proba(features[list(columns)])[0]
    aligned = np.zeros(7, dtype=float)
    for index, label in enumerate(model.classes_):
        aligned[int(label)] = raw[index]
    aligned = np.clip(aligned, 1e-12, 1.0)
    return aligned / aligned.sum()


def _adjust_draw_probability(probabilities: np.ndarray, elo_diff: float, method: str, alpha: float) -> np.ndarray:
    if method == "none":
        return probabilities
    multiplier = alpha
    if method == "similar_strength":
        scale = float(config_value("training_strategy", "draw_similarity_scale", default=180))
        multiplier = 1.0 + (alpha - 1.0) * np.exp(-abs(elo_diff) / scale)
    adjusted = probabilities.copy()
    adjusted[1] *= multiplier
    return adjusted / adjusted.sum()


def _calibrate_target_probabilities(probabilities: np.ndarray, calibrator: Any | None) -> np.ndarray:
    if calibrator is None:
        return probabilities
    method = str(calibrator["method"])
    model = calibrator["model"]
    if method == "sigmoid":
        return model.predict_proba(np.log(np.clip(probabilities, 1e-8, 1.0)).reshape(1, -1))[0]
    calibrated = np.asarray(
        [model[index].predict([probabilities[index]])[0] for index in range(3)],
        dtype=float,
    )
    calibrated = np.clip(calibrated, 1e-8, 1.0)
    return calibrated / calibrated.sum()


def _apply_indirect_features(
    features: pd.DataFrame,
    home_team: str,
    away_team: str,
    adjustments: dict[str, dict[str, float]],
) -> pd.DataFrame:
    output = features.copy()
    home = float(adjustments.get(home_team, {}).get("total_indirect_adjustment", 0.0))
    away = float(adjustments.get(away_team, {}).get("total_indirect_adjustment", 0.0))
    output.loc[:, "home_elo_pre"] = output["home_elo_pre"].astype(float) + home
    output.loc[:, "away_elo_pre"] = output["away_elo_pre"].astype(float) + away
    output.loc[:, "elo_diff"] = output["home_elo_pre"] - output["away_elo_pre"]
    return output


def _predict_features(features: pd.DataFrame, context: PredictionContext) -> dict[str, Any]:
    if context.goal_home_model is None or context.goal_away_model is None:
        raise FileNotFoundError("Goal models are missing. Run scripts/train_models.py first.")
    expected_goals_home = float(
        np.clip(context.goal_home_model.predict(features[list(context.goal_feature_columns)])[0], 0.05, 5.5)
    )
    expected_goals_away = float(
        np.clip(context.goal_away_model.predict(features[list(context.goal_feature_columns)])[0], 0.05, 5.5)
    )
    matrix = scoreline_matrix(
        expected_goals_home,
        expected_goals_away,
        max_goals=context.max_goals,
        dixon_coles_rho=context.dixon_coles_rho,
    )
    poisson_probs = outcome_probabilities(matrix)
    if context.target_model_name == "margin_class_classifier__uncalibrated":
        if context.target_model is None:
            raise FileNotFoundError("Selected margin target model is missing. Run scripts/train_models.py.")
        margin_probabilities = _aligned_margin_probabilities(
            context.target_model, features, context.target_feature_columns
        )
        expected_difference = float(margin_probabilities @ MARGIN_CLASS_CENTERS)
        expected_total = expected_goals_home + expected_goals_away
        adjusted_home = float(np.clip((expected_total + expected_difference) / 2.0, 0.05, 5.5))
        adjusted_away = float(np.clip((expected_total - expected_difference) / 2.0, 0.05, 5.5))
        matrix = scoreline_matrix(
            adjusted_home,
            adjusted_away,
            max_goals=context.max_goals,
            dixon_coles_rho=context.dixon_coles_rho,
        )
        target_outcome_probabilities = np.asarray(
            [margin_probabilities[4:].sum(), margin_probabilities[3], margin_probabilities[:3].sum()]
        )
        target_outcome_probabilities = _adjust_draw_probability(
            target_outcome_probabilities,
            float(features.iloc[0]["elo_diff"]),
            context.draw_correction,
            context.draw_alpha,
        )
        target_outcome_probabilities = _calibrate_target_probabilities(
            target_outcome_probabilities,
            context.probability_calibrator,
        )
        matrix = reweight_scoreline_outcomes(matrix, target_outcome_probabilities)
        expected_goals_home, expected_goals_away = expected_goals_from_matrix(matrix)
        poisson_probs = outcome_probabilities(matrix)

    ml_probs = None
    requires_ml = context.selected_model_type in {
        "ml_challenger",
        "ml_calibrated",
        "multinomial_logit",
        "ensemble",
    }
    if requires_ml:
        if context.outcome_model is None:
            raise FileNotFoundError(
                f"Outcome model is required for selected model type {context.selected_model_type!r}."
            )
        ml_probs = context.outcome_model.predict_proba(features[list(context.outcome_feature_columns)])[0]
    elo_probs = elo_outcome_probabilities(features)[0]
    if context.selected_model_type == "elo_baseline":
        probabilities = elo_probs
    elif context.selected_model_type == "poisson_goal":
        probabilities = poisson_probs
    elif context.selected_model_type in {"ml_challenger", "ml_calibrated", "multinomial_logit"}:
        probabilities = ml_probs
    elif context.selected_model_type == "ensemble":
        probabilities = _blend_probabilities(poisson_probs, ml_probs, context)
    else:
        raise ValueError(f"Unsupported selected model type {context.selected_model_type!r}.")
    most_likely_index = np.unravel_index(np.argmax(matrix.values), matrix.shape)
    return {
        "expected_goals_home": expected_goals_home,
        "expected_goals_away": expected_goals_away,
        "p_home_win": float(probabilities[0]),
        "p_draw": float(probabilities[1]),
        "p_away_win": float(probabilities[2]),
        "most_likely_score": f"{most_likely_index[0]}-{most_likely_index[1]}",
        "most_likely_score_probability": float(matrix.iloc[most_likely_index]),
        "top_5_scorelines": json.dumps(top_scorelines(matrix)),
        **scoreline_probabilities(matrix),
        "_scoreline_matrix_values": matrix.to_numpy(),
    }


@lru_cache(maxsize=8192)
def predict_match(
    home_team: str,
    away_team: str,
    match_date: str | pd.Timestamp,
    neutral: bool = True,
    venue_country: str | None = None,
    tournament: str = "World Cup",
    stage: str | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    context = load_prediction_context(models_dir=models_dir)
    features = build_single_match_features(
        home_team=home_team,
        away_team=away_team,
        match_date=pd.Timestamp(match_date),
        neutral=neutral,
        venue_country=venue_country,
        tournament=tournament,
        feature_state=context.feature_state,
        stage=stage,
    )
    if context.rating_state is not None:
        features = apply_rating_state_to_match(
            features,
            context.rating_state,
            home_team,
            away_team,
            match_date,
        )
    baseline = _predict_features(features, context)
    indirect_features = _apply_indirect_features(
        features, home_team, away_team, context.comparison_indirect_adjustments
    )
    indirect = _predict_features(indirect_features, context)
    selected_features = _apply_indirect_features(
        features, home_team, away_team, context.selected_indirect_adjustments
    )
    selected = baseline if context.selected_indirect_variant == "baseline" else _predict_features(selected_features, context)
    selected_home = context.selected_indirect_adjustments.get(home_team, {})
    selected_away = context.selected_indirect_adjustments.get(away_team, {})

    return {
        "home_team": home_team,
        "away_team": away_team,
        **selected,
        "baseline_expected_goals_home": baseline["expected_goals_home"],
        "baseline_expected_goals_away": baseline["expected_goals_away"],
        "baseline_p_home_win": baseline["p_home_win"],
        "baseline_p_draw": baseline["p_draw"],
        "baseline_p_away_win": baseline["p_away_win"],
        "baseline_most_likely_score": baseline["most_likely_score"],
        "indirect_expected_goals_home": indirect["expected_goals_home"],
        "indirect_expected_goals_away": indirect["expected_goals_away"],
        "indirect_p_home_win": indirect["p_home_win"],
        "indirect_p_draw": indirect["p_draw"],
        "indirect_p_away_win": indirect["p_away_win"],
        "indirect_most_likely_score": indirect["most_likely_score"],
        "indirect_adjustment_used": context.selected_indirect_variant != "baseline",
        "indirect_comparison_variant": context.comparison_indirect_variant,
        "trend_adjustment_home": float(selected_home.get("trend_adjustment", 0.0)),
        "trend_adjustment_away": float(selected_away.get("trend_adjustment", 0.0)),
        "readiness_adjustment_home": float(selected_home.get("readiness_adjustment", 0.0)),
        "readiness_adjustment_away": float(selected_away.get("readiness_adjustment", 0.0)),
        "model_used": context.target_model_name,
        "feature_set": f"{context.feature_set}+{context.target_model_name.split('__', 1)[0]}",
        "calibration_method": context.calibration_method,
        "draw_correction": context.draw_correction,
        "dixon_coles_rho": context.dixon_coles_rho,
        "model_version": f"{context.model_version}:{context.target_model_name}",
        "prediction_cutoff_date": context.training_cutoff,
    }
