from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.weighting import combined_sample_weights
from src.utils.config import config_value
from src.utils.logging import setup_logging
from src.utils.paths import MODELS_DIR, ensure_project_dirs


LOGGER = setup_logging(__name__)

GOAL_BASIC_FEATURES = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff",
    "neutral",
    "is_friendly",
    "is_world_cup",
    "is_continental_competition",
    "home_advantage_flag",
    "host_country_flag",
    "tournament_importance",
]
GOAL_ATTACK_DEFENCE_FEATURES = GOAL_BASIC_FEATURES + [
    "rolling_goals_for_home_5",
    "rolling_goals_against_home_5",
    "rolling_goals_for_away_5",
    "rolling_goals_against_away_5",
    "rolling_goals_for_home_10",
    "rolling_goals_against_home_10",
    "rolling_goals_for_away_10",
    "rolling_goals_against_away_10",
    "home_attack_rating_pre",
    "home_defence_rating_pre",
    "away_attack_rating_pre",
    "away_defence_rating_pre",
]
GOAL_FEATURE_COLUMNS = GOAL_ATTACK_DEFENCE_FEATURES + [
    "recent_goal_diff_home_5",
    "recent_goal_diff_away_5",
    "recent_goal_diff_home_10",
    "recent_goal_diff_away_10",
    "goals_for_avg_home_10",
    "goals_against_avg_home_10",
    "goals_for_avg_away_10",
    "goals_against_avg_away_10",
    "opponent_adjusted_goals_for_home_10",
    "opponent_adjusted_goals_against_home_10",
    "opponent_adjusted_goals_for_away_10",
    "opponent_adjusted_goals_against_away_10",
    "goal_diff_above_expectation_home_5",
    "goal_diff_above_expectation_away_5",
    "goal_diff_above_expectation_home_10",
    "goal_diff_above_expectation_away_10",
]
GOAL_FEATURE_GROUPS = {
    "basic_poisson": GOAL_BASIC_FEATURES,
    "attack_defence_poisson": GOAL_ATTACK_DEFENCE_FEATURES,
    "full_poisson": GOAL_FEATURE_COLUMNS,
}


def _build_goal_model(
    train_frame: pd.DataFrame,
    target_column: str,
    half_life_years: float | None,
    use_match_importance: bool | None,
    feature_columns: list[str],
    model_type: str,
    importance_profile: str | None = None,
) -> Pipeline:
    if model_type == "poisson":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.05, max_iter=2000)),
        ]
    elif model_type == "gradient_boosting":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=250,
                    l2_regularization=0.2,
                    random_state=int(config_value("project", "random_seed", default=42)),
                ),
            ),
        ]
    else:
        raise ValueError(f"Unsupported goal model type {model_type!r}.")
    pipeline = Pipeline(steps)
    weights = combined_sample_weights(
        train_frame,
        train_frame["date"].max(),
        half_life_years,
        use_match_importance,
        importance_profile=importance_profile,
    )
    pipeline.fit(train_frame[feature_columns], train_frame[target_column], model__sample_weight=weights)
    return pipeline


def _evaluate_goal_model(
    model: Pipeline,
    validation_frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> dict[str, Any]:
    predictions = np.clip(model.predict(validation_frame[feature_columns]), 0.05, 6.0)
    return {
        "poisson_deviance": float(mean_poisson_deviance(validation_frame[target_column], predictions)),
        "mean_expected_goals": float(np.mean(predictions)),
    }


@dataclass
class GoalModelBundle:
    home_model: Pipeline
    away_model: Pipeline
    home_metrics: dict[str, Any]
    away_metrics: dict[str, Any]
    training_cutoff: str
    goal_cap: int
    feature_columns: list[str]
    model_type: str

    def save(self, models_dir: Path | None = None) -> None:
        models_dir = models_dir or MODELS_DIR
        ensure_project_dirs()
        joblib.dump(self.home_model, models_dir / "goal_model_home.joblib")
        joblib.dump(self.away_model, models_dir / "goal_model_away.joblib")
        metadata = {
            "training_cutoff": self.training_cutoff,
            "goal_cap": self.goal_cap,
            "home_metrics": self.home_metrics,
            "away_metrics": self.away_metrics,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
        }
        (models_dir / "goal_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_goal_model(
    features_frame: pd.DataFrame,
    cutoff_date: datetime | pd.Timestamp,
    persist: bool = True,
    half_life_years: float | None = None,
    use_match_importance: bool | None = None,
    minimum_year: int | None = None,
    feature_columns: list[str] | None = None,
    model_type: str = "poisson",
    goal_cap: int | None = None,
    importance_profile: str | None = None,
) -> GoalModelBundle:
    ensure_project_dirs()
    resolved_goal_cap = int(goal_cap if goal_cap is not None else config_value("modeling", "goal_cap", default=8))
    resolved_minimum = int(
        minimum_year if minimum_year is not None else config_value("modeling", "minimum_training_year", default=1990)
    )
    frame = features_frame.copy().dropna(subset=["home_score", "away_score"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"].dt.year >= resolved_minimum].sort_values("date").reset_index(drop=True)
    # Keep upsets and unusual matches; cap only target severity for robust Poisson fitting.
    frame["home_score_model"] = frame["home_score"].clip(upper=resolved_goal_cap)
    frame["away_score_model"] = frame["away_score"].clip(upper=resolved_goal_cap)
    columns = feature_columns or GOAL_FEATURE_COLUMNS

    validation_cutoff = frame["date"].max() - pd.DateOffset(
        years=int(config_value("modeling", "validation_years", default=4))
    )
    train_frame = frame[frame["date"] < validation_cutoff].copy()
    validation_frame = frame[frame["date"] >= validation_cutoff].copy()
    if train_frame.empty or validation_frame.empty:
        split_index = max(1, int(len(frame) * 0.8))
        train_frame = frame.iloc[:split_index].copy()
        validation_frame = frame.iloc[split_index:].copy()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Insufficient chronological history for goal-model validation.")

    home_validation_model = _build_goal_model(
        train_frame, "home_score_model", half_life_years, use_match_importance, columns, model_type, importance_profile
    )
    away_validation_model = _build_goal_model(
        train_frame, "away_score_model", half_life_years, use_match_importance, columns, model_type, importance_profile
    )
    home_metrics = _evaluate_goal_model(home_validation_model, validation_frame, "home_score_model", columns)
    away_metrics = _evaluate_goal_model(away_validation_model, validation_frame, "away_score_model", columns)

    home_model = _build_goal_model(
        frame, "home_score_model", half_life_years, use_match_importance, columns, model_type, importance_profile
    )
    away_model = _build_goal_model(
        frame, "away_score_model", half_life_years, use_match_importance, columns, model_type, importance_profile
    )
    bundle = GoalModelBundle(
        home_model=home_model,
        away_model=away_model,
        home_metrics=home_metrics,
        away_metrics=away_metrics,
        training_cutoff=pd.Timestamp(cutoff_date).isoformat(),
        goal_cap=resolved_goal_cap,
        feature_columns=columns,
        model_type=model_type,
    )
    if persist:
        bundle.save()
    LOGGER.info("Goal metrics: home %s, away %s", home_metrics, away_metrics)
    return bundle
