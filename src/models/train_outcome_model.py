from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import multiclass_brier_score, reliability_table
from src.models.weighting import combined_sample_weights
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import MODELS_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)
OUTCOME_LABELS = [0, 1, 2]

CORE_FEATURES = [
    "elo_diff",
    "neutral",
    "is_friendly",
    "is_world_cup",
    "is_continental_competition",
    "home_advantage_flag",
    "host_country_flag",
    "tournament_importance",
    "days_since_last_match_home",
    "days_since_last_match_away",
    "rest_days_diff",
]
ELO_FEATURES = ["elo_diff"]
REST_FEATURES = [
    "days_since_last_match_home",
    "days_since_last_match_away",
    "rest_days_diff",
]
VENUE_HOST_FEATURES = [
    "neutral",
    "home_advantage_flag",
    "host_country_flag",
]
RAW_FORM_FEATURES = [
    "recent_form_points_home_5",
    "recent_form_points_away_5",
    "recent_form_points_home_10",
    "recent_form_points_away_10",
    "recent_goal_diff_home_5",
    "recent_goal_diff_away_5",
    "recent_goal_diff_home_10",
    "recent_goal_diff_away_10",
]
OPPONENT_ADJUSTED_FEATURES = [
    "opponent_adjusted_form_home_5",
    "opponent_adjusted_form_away_5",
    "opponent_adjusted_form_home_10",
    "opponent_adjusted_form_away_10",
    "opponent_adjusted_goals_for_home_10",
    "opponent_adjusted_goals_against_home_10",
    "opponent_adjusted_goals_for_away_10",
    "opponent_adjusted_goals_against_away_10",
    "points_above_expectation_home_5",
    "points_above_expectation_away_5",
    "points_above_expectation_home_10",
    "points_above_expectation_away_10",
    "goal_diff_above_expectation_home_5",
    "goal_diff_above_expectation_away_5",
    "goal_diff_above_expectation_home_10",
    "goal_diff_above_expectation_away_10",
    "recent_elo_change_home_5",
    "recent_elo_change_away_5",
    "recent_elo_change_home_10",
    "recent_elo_change_away_10",
]
ATTACK_DEFENCE_FEATURES = [
    "rolling_goals_for_home_5",
    "rolling_goals_against_home_5",
    "rolling_goals_for_away_5",
    "rolling_goals_against_away_5",
    "rolling_goals_for_home_10",
    "rolling_goals_against_home_10",
    "rolling_goals_for_away_10",
    "rolling_goals_against_away_10",
    "goals_for_avg_home_10",
    "goals_against_avg_home_10",
    "goals_for_avg_away_10",
    "goals_against_avg_away_10",
    "home_attack_rating_pre",
    "home_defence_rating_pre",
    "away_attack_rating_pre",
    "away_defence_rating_pre",
]
FEATURE_COLUMNS = CORE_FEATURES + RAW_FORM_FEATURES + OPPONENT_ADJUSTED_FEATURES + ATTACK_DEFENCE_FEATURES
FEATURE_GROUPS = {
    "core_without_elo": [column for column in CORE_FEATURES if column not in ELO_FEATURES],
    "core_without_rest": [column for column in CORE_FEATURES if column not in REST_FEATURES],
    "core_without_venue_host": [column for column in CORE_FEATURES if column not in VENUE_HOST_FEATURES],
    "core": CORE_FEATURES,
    "core_raw_form": CORE_FEATURES + RAW_FORM_FEATURES,
    "core_opponent_adjusted": CORE_FEATURES + OPPONENT_ADJUSTED_FEATURES,
    "all_without_attack_defence": CORE_FEATURES + RAW_FORM_FEATURES + OPPONENT_ADJUSTED_FEATURES,
    "all_features": FEATURE_COLUMNS,
}


@dataclass
class CalibratedProbabilityModel:
    base_model: Any
    calibrator: LogisticRegression
    feature_columns: list[str]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self.base_model.predict_proba(frame[self.feature_columns]), 1e-8, 1.0)
        return self.calibrator.predict_proba(np.log(raw))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(frame), axis=1)


def _prepare_training_frame(features_frame: pd.DataFrame, minimum_year: int | None = None) -> pd.DataFrame:
    frame = features_frame.copy().dropna(subset=["result"])
    frame["date"] = pd.to_datetime(frame["date"])
    resolved_minimum = int(
        minimum_year if minimum_year is not None else config_value("modeling", "minimum_training_year", default=1990)
    )
    return frame[frame["date"].dt.year >= resolved_minimum].sort_values("date").reset_index(drop=True)


def _train_validation_split(frame: pd.DataFrame, validation_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = frame["date"].max() - pd.DateOffset(years=validation_years)
    train_frame = frame[frame["date"] < cutoff].copy()
    validation_frame = frame[frame["date"] >= cutoff].copy()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Insufficient chronological history for the requested validation period.")
    return train_frame, validation_frame


def _sample_weights(
    frame: pd.DataFrame,
    reference_date: pd.Timestamp,
    half_life_years: float | None,
    use_match_importance: bool | None,
) -> np.ndarray:
    return combined_sample_weights(frame, reference_date, half_life_years, use_match_importance)


def _fit_baseline_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    half_life_years: float | None = None,
    use_match_importance: bool | None = None,
) -> Pipeline:
    numeric_features = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_columns,
            )
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=2000, random_state=int(config_value("project", "random_seed", default=42)))
    pipeline = Pipeline([("preprocessor", numeric_features), ("classifier", model)])
    weights = _sample_weights(train_frame, train_frame["date"].max(), half_life_years, use_match_importance)
    pipeline.fit(train_frame[feature_columns], train_frame["result"], classifier__sample_weight=weights)
    return pipeline


def _fit_challenger_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    half_life_years: float | None = None,
    use_match_importance: bool | None = None,
) -> Pipeline:
    classifier: Any = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        l2_regularization=0.1,
        random_state=int(config_value("project", "random_seed", default=42)),
    )
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])
    weights = _sample_weights(train_frame, train_frame["date"].max(), half_life_years, use_match_importance)
    pipeline.fit(train_frame[feature_columns], train_frame["result"], classifier__sample_weight=weights)
    return pipeline


def _fit_calibrator(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    half_life_years: float | None,
    use_match_importance: bool | None,
) -> LogisticRegression:
    split_date = train_frame["date"].max() - pd.DateOffset(years=2)
    base_frame = train_frame[train_frame["date"] < split_date]
    calibration_frame = train_frame[train_frame["date"] >= split_date]
    if base_frame.empty or calibration_frame.empty:
        raise ValueError("Insufficient chronological data for probability calibration.")
    calibration_base = _fit_challenger_model(base_frame, feature_columns, half_life_years, use_match_importance)
    raw = np.clip(calibration_base.predict_proba(calibration_frame[feature_columns]), 1e-8, 1.0)
    calibrator = LogisticRegression(max_iter=2000, random_state=int(config_value("project", "random_seed", default=42)))
    weights = _sample_weights(
        calibration_frame,
        calibration_frame["date"].max(),
        half_life_years,
        use_match_importance,
    )
    calibrator.fit(np.log(raw), calibration_frame["result"], sample_weight=weights)
    return calibrator


def _evaluate_probabilities(probabilities: np.ndarray, validation_frame: pd.DataFrame, model_name: str) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    return {
        "model_name": model_name,
        "log_loss": float(log_loss(validation_frame["result"], probabilities, labels=OUTCOME_LABELS)),
        "brier_score": float(multiclass_brier_score(validation_frame["result"], probabilities)),
        "accuracy": float(accuracy_score(validation_frame["result"], predictions)),
        "home_win_reliability": reliability_table(validation_frame["result"].to_numpy(), probabilities),
    }


def _evaluate_model(model: Any, validation_frame: pd.DataFrame, model_name: str, feature_columns: list[str]) -> dict[str, Any]:
    return _evaluate_probabilities(model.predict_proba(validation_frame[feature_columns]), validation_frame, model_name)


def elo_outcome_probabilities(frame: pd.DataFrame, home_advantage: float = 65.0) -> np.ndarray:
    adjusted_diff = frame["elo_diff"].to_numpy(dtype=float) + home_advantage * (1.0 - frame["neutral"].to_numpy(dtype=float))
    decisive_home = 1.0 / (1.0 + np.power(10.0, -adjusted_diff / 400.0))
    draw = np.clip(0.28 * np.exp(-np.abs(adjusted_diff) / 600.0), 0.12, 0.30)
    return np.column_stack([(1.0 - draw) * decisive_home, draw, (1.0 - draw) * (1.0 - decisive_home)])


@dataclass
class OutcomeModelBundle:
    baseline_model: Pipeline
    challenger_model: Pipeline
    calibrated_model: CalibratedProbabilityModel
    baseline_metrics: dict[str, Any]
    challenger_metrics: dict[str, Any]
    calibrated_metrics: dict[str, Any]
    elo_metrics: dict[str, Any]
    best_model_name: str
    training_cutoff: str
    half_life_years: float
    feature_columns: list[str]

    def save(self, models_dir: Path | None = None) -> None:
        models_dir = models_dir or MODELS_DIR
        ensure_project_dirs()
        joblib.dump(self.baseline_model, models_dir / "outcome_model_baseline.joblib")
        joblib.dump(self.challenger_model, models_dir / "outcome_model_challenger.joblib")
        joblib.dump(self.calibrated_model, models_dir / "outcome_model_calibrated.joblib")
        candidates = {
            "baseline": self.baseline_model,
            "challenger": self.challenger_model,
            "calibrated": self.calibrated_model,
        }
        joblib.dump(candidates[self.best_model_name], models_dir / "outcome_model_best.joblib")
        metadata = {
            "best_model_name": self.best_model_name,
            "training_cutoff": self.training_cutoff,
            "half_life_years": self.half_life_years,
            "baseline_metrics": self.baseline_metrics,
            "challenger_metrics": self.challenger_metrics,
            "calibrated_metrics": self.calibrated_metrics,
            "elo_metrics": self.elo_metrics,
            "feature_columns": self.feature_columns,
        }
        (models_dir / "outcome_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_outcome_model(
    features_frame: pd.DataFrame,
    cutoff_date: datetime | pd.Timestamp,
    persist: bool = True,
    half_life_years: float | None = None,
    use_match_importance: bool | None = None,
    feature_columns: list[str] | None = None,
    minimum_year: int | None = None,
) -> OutcomeModelBundle:
    ensure_project_dirs()
    columns = feature_columns or FEATURE_COLUMNS
    half_life = float(
        half_life_years
        if half_life_years is not None
        else config_value("modeling", "time_decay_half_life_years", default=8)
    )
    frame = _prepare_training_frame(features_frame, minimum_year)
    train_frame, validation_frame = _train_validation_split(
        frame, int(config_value("modeling", "validation_years", default=4))
    )

    baseline_validation_model = _fit_baseline_model(train_frame, columns, half_life, use_match_importance)
    challenger_validation_model = _fit_challenger_model(train_frame, columns, half_life, use_match_importance)
    calibrator = _fit_calibrator(train_frame, columns, half_life, use_match_importance)
    calibrated_validation_model = CalibratedProbabilityModel(challenger_validation_model, calibrator, columns)

    baseline_metrics = _evaluate_model(baseline_validation_model, validation_frame, "multinomial_logit", columns)
    challenger_metrics = _evaluate_model(challenger_validation_model, validation_frame, "ml_challenger", columns)
    calibrated_metrics = _evaluate_model(calibrated_validation_model, validation_frame, "ml_calibrated", columns)
    elo_metrics = _evaluate_probabilities(elo_outcome_probabilities(validation_frame), validation_frame, "elo_baseline")

    baseline_model = _fit_baseline_model(frame, columns, half_life, use_match_importance)
    challenger_model = _fit_challenger_model(frame, columns, half_life, use_match_importance)
    calibrated_model = CalibratedProbabilityModel(challenger_model, calibrator, columns)
    metrics_by_name = {
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "calibrated": calibrated_metrics,
    }
    best_model_name = min(metrics_by_name, key=lambda name: metrics_by_name[name]["log_loss"])
    bundle = OutcomeModelBundle(
        baseline_model=baseline_model,
        challenger_model=challenger_model,
        calibrated_model=calibrated_model,
        baseline_metrics=baseline_metrics,
        challenger_metrics=challenger_metrics,
        calibrated_metrics=calibrated_metrics,
        elo_metrics=elo_metrics,
        best_model_name=best_model_name,
        training_cutoff=pd.Timestamp(cutoff_date).isoformat(),
        half_life_years=half_life,
        feature_columns=columns,
    )
    if persist:
        bundle.save()
    LOGGER.info(
        "Outcome metrics, half-life %s years: baseline %.4f, challenger %.4f, calibrated %.4f, Elo %.4f.",
        half_life,
        baseline_metrics["log_loss"],
        challenger_metrics["log_loss"],
        calibrated_metrics["log_loss"],
        elo_metrics["log_loss"],
    )
    return bundle
