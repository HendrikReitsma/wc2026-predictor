from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.target_engineering import add_match_targets
from src.models.train_goal_model import GOAL_FEATURE_GROUPS
from src.models.weighting import combined_sample_weights
from src.utils.config import config_value
from src.utils.paths import MODELS_DIR, ensure_project_dirs


def _fit_margin_classifier(
    frame: pd.DataFrame,
    feature_columns: list[str],
    half_life_years: float,
    use_match_importance: bool,
    importance_profile: str,
) -> Pipeline:
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
    weights = combined_sample_weights(
        frame,
        frame["date"].max(),
        half_life_years=half_life_years,
        use_match_importance=use_match_importance,
        importance_profile=importance_profile,
    )
    pipeline.fit(frame[feature_columns], frame["margin_class_index"], model__sample_weight=weights)
    return pipeline


def _margin_outcome_probabilities(model: Pipeline, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[columns])
    aligned = np.zeros((len(frame), 7), dtype=float)
    for index, label in enumerate(model.classes_):
        aligned[:, int(label)] = raw[:, index]
    output = np.column_stack([aligned[:, 4:].sum(axis=1), aligned[:, 3], aligned[:, :3].sum(axis=1)])
    return output / output.sum(axis=1, keepdims=True)


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


def train_selected_target_model(
    features_frame: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_project_dirs()
    output_dir = models_dir or MODELS_DIR
    selection_path = output_dir / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    target_selection = selection.get("target_experiment_recommendation", {})
    strategy = selection.get("training_strategy_recommendation", {})
    model_name = str(target_selection.get("recommended_model_name", "current_poisson_baseline__uncalibrated"))
    goal_feature_group = str(selection.get("goal_feature_group", "attack_defence_poisson"))
    feature_columns = GOAL_FEATURE_GROUPS[goal_feature_group]
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "training_cutoff": pd.Timestamp(cutoff_date).isoformat(),
        "feature_columns": feature_columns,
        "calibration_method": model_name.rsplit("__", 1)[-1],
        "target_type": str(target_selection.get("recommended_target_type", "goals")),
    }
    artifact_path = output_dir / "target_model.joblib"
    calibrator_path = output_dir / "target_probability_calibrator.joblib"
    if model_name == "margin_class_classifier__uncalibrated":
        frame = add_match_targets(features_frame).dropna(subset=["margin_class_index"]).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        minimum_year = int(selection.get("minimum_training_year", config_value("modeling", "minimum_training_year", default=1990)))
        frame = frame[(frame["date"].dt.year >= minimum_year) & (frame["date"] <= pd.Timestamp(cutoff_date))]
        half_life = float(strategy.get("training_half_life_years", selection.get("training_half_life_years", 1e9)))
        importance_profile = str(strategy.get("importance_profile", selection.get("weighting_scheme", "aggressive")))
        use_importance = bool(strategy.get("use_match_importance_weights", selection.get("use_match_importance_weights", True)))
        model = _fit_margin_classifier(frame, feature_columns, half_life, use_importance, importance_profile)
        joblib.dump(model, artifact_path)
        metadata["artifact"] = artifact_path.name
        metadata["classes"] = [int(value) for value in model.classes_]
        metadata["draw_correction"] = str(strategy.get("draw_correction", "none"))
        metadata["draw_alpha"] = float(strategy.get("draw_alpha", 1.0))
        metadata["calibration_method"] = str(strategy.get("calibration_method", "uncalibrated"))
        calibration_method = metadata["calibration_method"]
        if calibration_method != "uncalibrated":
            split = frame["date"].max() - pd.DateOffset(
                years=int(config_value("training_strategy", "calibration_years", default=2))
            )
            base = frame[frame["date"] < split].copy()
            calibration = frame[frame["date"] >= split].copy()
            base_model = _fit_margin_classifier(base, feature_columns, half_life, use_importance, importance_profile)
            raw = _margin_outcome_probabilities(base_model, calibration, feature_columns)
            raw = _draw_adjust(
                raw,
                calibration["elo_diff"].to_numpy(dtype=float),
                metadata["draw_correction"],
                metadata["draw_alpha"],
            )
            if calibration_method == "sigmoid":
                calibrator: Any = LogisticRegression(
                    max_iter=2000,
                    random_state=int(config_value("project", "random_seed", default=42)),
                )
                calibrator.fit(np.log(np.clip(raw, 1e-8, 1.0)), calibration["result"])
            elif calibration_method == "isotonic":
                calibrator = [
                    IsotonicRegression(out_of_bounds="clip").fit(raw[:, index], calibration["result"].eq(index))
                    for index in range(3)
                ]
            else:
                raise ValueError(f"Unsupported selected calibration method {calibration_method!r}.")
            joblib.dump({"method": calibration_method, "model": calibrator}, calibrator_path)
            metadata["calibrator_artifact"] = calibrator_path.name
        else:
            if calibrator_path.exists():
                calibrator_path.unlink()
            metadata["calibrator_artifact"] = None
    else:
        if artifact_path.exists():
            artifact_path.unlink()
        if calibrator_path.exists():
            calibrator_path.unlink()
        metadata["artifact"] = None
        metadata["notes"] = "The selected target uses the existing production model and needs no extra artifact."
    (output_dir / "target_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
