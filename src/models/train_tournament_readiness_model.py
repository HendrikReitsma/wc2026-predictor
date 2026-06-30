from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.tournament_readiness_features import READINESS_FEATURE_COLUMNS
from src.utils.paths import MODELS_DIR


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=15.0)),
        ]
    )


@dataclass
class TournamentReadinessBundle:
    performance_model: Pipeline
    points_model: Pipeline
    residual_std: float
    training_rows: int

    def save(self, path: Path | None = None) -> None:
        joblib.dump(self, path or MODELS_DIR / "tournament_readiness_model.joblib")

    def predict(self, readiness_rows: pd.DataFrame) -> pd.DataFrame:
        output = readiness_rows[["team", "base_elo", "tournament_year"]].copy()
        performance = np.clip(
            self.performance_model.predict(readiness_rows[READINESS_FEATURE_COLUMNS]), -2.5, 2.5
        )
        points = np.clip(self.points_model.predict(readiness_rows[READINESS_FEATURE_COLUMNS]), -8.0, 8.0)
        z = performance / max(self.residual_std, 1e-6)
        over = np.asarray([0.5 * (1.0 + erf(value / sqrt(2.0))) for value in z])
        output["tournament_readiness_score"] = performance
        output["expected_group_points_adjustment"] = points
        output["expected_goal_difference_adjustment"] = performance * 3.0
        output["overperformance_probability"] = over
        output["underperformance_probability"] = 1.0 - over
        output["readiness_data_quality_flag"] = np.where(
            readiness_rows["qualification_matches"].fillna(0).lt(4), "limited_qualification_history", "ok"
        )
        return output


def train_tournament_readiness_model(
    readiness_dataset: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    persist: bool = True,
    models_dir: Path | None = None,
) -> TournamentReadinessBundle:
    training = readiness_dataset[
        pd.to_datetime(readiness_dataset["readiness_target_end_date"]) < pd.Timestamp(cutoff_date)
    ].copy()
    if len(training) < 50:
        raise ValueError("Insufficient completed historical tournament rows for readiness modelling.")
    performance_model = _pipeline()
    points_model = _pipeline()
    performance_model.fit(training[READINESS_FEATURE_COLUMNS], training["tournament_performance_above_elo_expectation"])
    points_model.fit(training[READINESS_FEATURE_COLUMNS], training["tournament_points_above_expectation"])
    residual = (
        training["tournament_performance_above_elo_expectation"].to_numpy(dtype=float)
        - performance_model.predict(training[READINESS_FEATURE_COLUMNS])
    )
    bundle = TournamentReadinessBundle(
        performance_model=performance_model,
        points_model=points_model,
        residual_std=float(np.std(residual)),
        training_rows=int(len(training)),
    )
    if persist:
        bundle.save((models_dir or MODELS_DIR) / "tournament_readiness_model.joblib")
    return bundle
