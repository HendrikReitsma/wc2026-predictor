from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import compute_classification_metrics, compute_scoreline_metrics
from src.models.scoreline import outcome_probabilities, scoreline_matrix
from src.models.train_goal_model import GOAL_FEATURE_GROUPS, train_goal_model
from src.utils.config import config_value


def _world_cup_frame(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    return frame.loc[
        (frame["date"].dt.year == year) & frame["is_world_cup"].fillna(0).astype(bool)
    ].copy()


def _training_frame(frame: pd.DataFrame, cutoff_year: int, minimum_year: int) -> pd.DataFrame:
    return frame.loc[
        (frame["date"].dt.year <= cutoff_year) & (frame["date"].dt.year >= minimum_year)
    ].copy()


def _predict_goal_bundle(bundle, evaluation_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = bundle.feature_columns
    home = np.clip(bundle.home_model.predict(evaluation_frame[columns]), 0.05, 5.5)
    away = np.clip(bundle.away_model.predict(evaluation_frame[columns]), 0.05, 5.5)
    return home, away


def _outcomes_from_lambdas(
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
    rho: float,
    max_goals: int,
) -> np.ndarray:
    return np.asarray(
        [
            outcome_probabilities(scoreline_matrix(home, away, max_goals=max_goals, dixon_coles_rho=rho))
            for home, away in zip(home_lambdas, away_lambdas)
        ]
    )


def run_goal_model_comparison(
    frame: pd.DataFrame,
    world_cup_years: list[int],
    training_cutoffs: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants: list[tuple[str, str, list[str], float]] = [
        ("basic_poisson", "poisson", GOAL_FEATURE_GROUPS["basic_poisson"], 0.0),
        ("attack_defence_poisson", "poisson", GOAL_FEATURE_GROUPS["attack_defence_poisson"], 0.0),
        ("full_poisson", "poisson", GOAL_FEATURE_GROUPS["full_poisson"], 0.0),
        ("ml_goal_full", "gradient_boosting", GOAL_FEATURE_GROUPS["full_poisson"], 0.0),
    ]
    for rho in config_value("modeling", "dixon_coles_rho_candidates", default=[0.0, -0.05, -0.1]):
        if float(rho) != 0.0:
            variants.append((f"full_poisson_dixon_coles_{float(rho):.2f}", "poisson", GOAL_FEATURE_GROUPS["full_poisson"], float(rho)))

    rows: list[dict[str, Any]] = []
    minimum_year = int(config_value("modeling", "minimum_training_year", default=1990))
    max_goals = int(config_value("modeling", "max_goals", default=10))
    for year in world_cup_years:
        evaluation = _world_cup_frame(frame, year)
        training = _training_frame(frame, training_cutoffs[year], minimum_year)
        trained: dict[tuple[str, tuple[str, ...]], Any] = {}
        for variant_name, model_type, columns, rho in variants:
            key = (model_type, tuple(columns))
            if key not in trained:
                trained[key] = train_goal_model(
                    training,
                    pd.Timestamp(f"{year}-01-01"),
                    persist=False,
                    half_life_years=1e9,
                    use_match_importance=True,
                    minimum_year=minimum_year,
                    feature_columns=columns,
                    model_type=model_type,
                )
            home, away = _predict_goal_bundle(trained[key], evaluation)
            rows.append(
                {
                    "model_name": variant_name,
                    "goal_model_type": model_type,
                    "goal_feature_group": next(
                        name for name, group_columns in GOAL_FEATURE_GROUPS.items() if group_columns == columns
                    ),
                    "dixon_coles_rho": rho,
                    "tournament_year": year,
                    "training_cutoff_year": training_cutoffs[year],
                    **compute_scoreline_metrics(evaluation, home, away, rho, max_goals),
                }
            )
    details = pd.DataFrame(rows)
    metric_columns = [
        "log_loss",
        "brier_score",
        "accuracy",
        "ranked_probability_score",
        "home_goal_mae",
        "away_goal_mae",
        "mean_goal_mae",
        "scoreline_log_loss",
        "top_1_scoreline_accuracy",
        "top_5_scoreline_hit_rate",
    ]
    summary = (
        details.groupby(["model_name", "goal_model_type", "goal_feature_group", "dixon_coles_rho"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["log_loss", "scoreline_log_loss"])
    )
    return details, summary


def run_minimum_year_comparison(
    frame: pd.DataFrame,
    world_cup_years: list[int],
    training_cutoffs: dict[int, int],
    model_type: str,
    feature_columns: list[str],
    rho: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    max_goals = int(config_value("modeling", "max_goals", default=10))
    candidates = [
        int(value)
        for value in config_value("modeling", "minimum_training_year_candidates", default=[1990, 1998, 2002, 2010])
    ]
    for minimum_year in candidates:
        for year in world_cup_years:
            training = _training_frame(frame, training_cutoffs[year], minimum_year)
            evaluation = _world_cup_frame(frame, year)
            if len(training) < 100:
                continue
            bundle = train_goal_model(
                training,
                pd.Timestamp(f"{year}-01-01"),
                persist=False,
                half_life_years=1e9,
                use_match_importance=True,
                minimum_year=minimum_year,
                feature_columns=feature_columns,
                model_type=model_type,
            )
            home, away = _predict_goal_bundle(bundle, evaluation)
            rows.append(
                {
                    "minimum_training_year": minimum_year,
                    "tournament_year": year,
                    "training_cutoff_year": training_cutoffs[year],
                    **compute_scoreline_metrics(evaluation, home, away, rho, max_goals),
                }
            )
    details = pd.DataFrame(rows)
    metrics = ["log_loss", "brier_score", "mean_goal_mae", "scoreline_log_loss", "top_5_scoreline_hit_rate"]
    summary = details.groupby("minimum_training_year", as_index=False)[metrics].mean()
    coverage = details.groupby("minimum_training_year").size().rename("world_cups_covered").reset_index()
    return details, summary.merge(coverage, on="minimum_training_year").sort_values(["world_cups_covered", "log_loss"], ascending=[False, True])


def _isotonic_calibrate(raw_fit: np.ndarray, y_fit: np.ndarray, raw_eval: np.ndarray) -> np.ndarray:
    calibrated = np.column_stack(
        [
            IsotonicRegression(out_of_bounds="clip").fit(raw_fit[:, index], y_fit == index).predict(raw_eval[:, index])
            for index in range(raw_fit.shape[1])
        ]
    )
    calibrated = np.clip(calibrated, 1e-8, 1.0)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def run_calibration_comparison(
    frame: pd.DataFrame,
    world_cup_years: list[int],
    training_cutoffs: dict[int, int],
    model_type: str,
    feature_columns: list[str],
    rho: float,
    minimum_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_goals = int(config_value("modeling", "max_goals", default=10))
    for year in world_cup_years:
        training = _training_frame(frame, training_cutoffs[year], minimum_year)
        evaluation = _world_cup_frame(frame, year)
        calibration_cutoff = training["date"].max() - pd.DateOffset(years=2)
        base = training[training["date"] < calibration_cutoff].copy()
        calibration = training[training["date"] >= calibration_cutoff].copy()
        bundle = train_goal_model(
            base,
            base["date"].max(),
            persist=False,
            half_life_years=1e9,
            use_match_importance=True,
            minimum_year=minimum_year,
            feature_columns=feature_columns,
            model_type=model_type,
        )
        calibration_home, calibration_away = _predict_goal_bundle(bundle, calibration)
        evaluation_home, evaluation_away = _predict_goal_bundle(bundle, evaluation)
        raw_calibration = _outcomes_from_lambdas(calibration_home, calibration_away, rho, max_goals)
        raw_evaluation = _outcomes_from_lambdas(evaluation_home, evaluation_away, rho, max_goals)
        platt = LogisticRegression(max_iter=2000, random_state=int(config_value("project", "random_seed", default=42)))
        platt.fit(np.log(np.clip(raw_calibration, 1e-8, 1.0)), calibration["result"])
        probabilities = {
            "uncalibrated": raw_evaluation,
            "platt_sigmoid": platt.predict_proba(np.log(np.clip(raw_evaluation, 1e-8, 1.0))),
            "isotonic": _isotonic_calibrate(raw_calibration, calibration["result"].to_numpy(dtype=int), raw_evaluation),
        }
        for method, predicted in probabilities.items():
            rows.append(
                {
                    "calibration_method": method,
                    "tournament_year": year,
                    "training_cutoff_year": training_cutoffs[year],
                    **compute_classification_metrics(
                        evaluation["result"].to_numpy(dtype=int),
                        predicted,
                        np.argmax(predicted, axis=1),
                    ),
                }
            )
    return pd.DataFrame(rows)
