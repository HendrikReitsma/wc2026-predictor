from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.team_strength_features import TREND_FEATURE_COLUMNS
from src.utils.paths import MODELS_DIR


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=25.0)),
        ]
    )


@dataclass
class TeamStrengthTrendBundle:
    elo_delta_model: Pipeline
    performance_model: Pipeline
    residual_std: float
    training_rows: int

    def save(self, path: Path | None = None) -> None:
        joblib.dump(self, path or MODELS_DIR / "team_strength_trend_model.joblib")

    def predict(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        output = snapshots[["team", "date", "base_elo", "prior_matches"]].copy()
        output["expected_future_elo_delta"] = np.clip(
            self.elo_delta_model.predict(snapshots[TREND_FEATURE_COLUMNS]), -100.0, 100.0
        )
        output["expected_future_performance_above_expectation"] = np.clip(
            self.performance_model.predict(snapshots[TREND_FEATURE_COLUMNS]), -5.0, 5.0
        )
        output["trend_score"] = (
            output["expected_future_elo_delta"] / 50.0
            + output["expected_future_performance_above_expectation"] / 3.0
        )
        output["trend_model_confidence"] = np.clip(output["prior_matches"] / 20.0, 0.0, 1.0)
        output["trend_data_quality_flag"] = np.where(
            output["prior_matches"].lt(10), "limited_history", "ok"
        )
        return output


def train_team_strength_trend_model(
    snapshots: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    persist: bool = True,
    models_dir: Path | None = None,
) -> TeamStrengthTrendBundle:
    target = "future_elo_delta_next_5_matches"
    performance_target = "future_points_above_expectation_next_5_matches"
    training = snapshots[
        snapshots[target].notna()
        & snapshots[performance_target].notna()
        & (pd.to_datetime(snapshots["future_target_end_date_next_5"]) < pd.Timestamp(cutoff_date))
        & snapshots["prior_matches"].ge(5)
    ].copy()
    if len(training) < 100:
        raise ValueError("Insufficient completed future windows for the team-strength trend model.")
    elo_model = _pipeline()
    performance_model = _pipeline()
    elo_model.fit(training[TREND_FEATURE_COLUMNS], training[target])
    performance_model.fit(training[TREND_FEATURE_COLUMNS], training[performance_target])
    residual = training[target].to_numpy(dtype=float) - elo_model.predict(training[TREND_FEATURE_COLUMNS])
    bundle = TeamStrengthTrendBundle(
        elo_delta_model=elo_model,
        performance_model=performance_model,
        residual_std=float(np.std(residual)),
        training_rows=int(len(training)),
    )
    if persist:
        bundle.save((models_dir or MODELS_DIR) / "team_strength_trend_model.joblib")
    return bundle
