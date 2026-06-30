from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import build_feature_rows
from src.evaluation.metrics import compute_scoreline_metrics
from src.models.predict_match import scoreline_matrix
from src.models.scoreline import outcome_probabilities, reweight_scoreline_outcomes, scoreline_probabilities, top_scorelines
from src.models.train_goal_model import GOAL_FEATURE_COLUMNS, train_goal_model
from src.models.train_outcome_model import FEATURE_COLUMNS, train_outcome_model


def _synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    teams = [f"Team {index}" for index in range(8)]
    rows = []
    for index in range(120):
        home_team = teams[index % len(teams)]
        away_team = teams[(index * 3 + 1) % len(teams)]
        if away_team == home_team:
            away_team = teams[(index + 2) % len(teams)]
        home_score = int(rng.poisson(1.5 + (index % 3) * 0.15))
        away_score = int(rng.poisson(1.2 + ((index + 1) % 3) * 0.1))
        rows.append(
            {
                "date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=index * 45),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": ["Friendly", "World Cup qualification", "World Cup"][index % 3],
                "country": home_team,
                "neutral": index % 4 == 0,
            }
        )
    features, _ = build_feature_rows(pd.DataFrame(rows))
    return features


def test_models_train_and_emit_probabilities() -> None:
    features = _synthetic_features()
    cutoff = features["date"].max()
    outcome = train_outcome_model(features, cutoff, persist=False)
    goals = train_goal_model(features, cutoff, persist=False)

    probabilities = outcome.baseline_model.predict_proba(features.iloc[-5:][FEATURE_COLUMNS])
    expected_home = goals.home_model.predict(features.iloc[-5:][GOAL_FEATURE_COLUMNS])

    assert probabilities.shape == (5, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(expected_home > 0)
    assert outcome.baseline_metrics["log_loss"] > 0


def test_scoreline_matrix_is_normalized() -> None:
    matrix = scoreline_matrix(1.6, 1.1, max_goals=8)
    assert matrix.shape == (9, 9)
    assert np.isclose(matrix.to_numpy().sum(), 1.0)
    derived = scoreline_probabilities(matrix)
    assert np.isclose(derived["p_over_2_5_goals"] + derived["p_under_2_5_goals"], 1.0)
    assert len(top_scorelines(matrix)) == 5


def test_scoreline_outcome_reweighting_matches_target_probabilities() -> None:
    matrix = scoreline_matrix(1.8, 1.1, max_goals=8)
    target = np.array([0.62, 0.23, 0.15])
    adjusted = reweight_scoreline_outcomes(matrix, target)
    assert np.allclose(outcome_probabilities(adjusted), target)
    assert np.isclose(adjusted.to_numpy().sum(), 1.0)


def test_scoreline_metrics_include_goal_and_hit_quality() -> None:
    frame = pd.DataFrame({"home_score": [1, 2], "away_score": [0, 1], "result": [0, 0]})
    metrics = compute_scoreline_metrics(frame, np.array([1.2, 1.8]), np.array([0.7, 0.9]))
    assert metrics["home_goal_mae"] >= 0
    assert 0 <= metrics["top_1_scoreline_accuracy"] <= 1
    assert 0 <= metrics["top_5_scoreline_hit_rate"] <= 1


def test_model_feature_contract_accepts_single_match_features() -> None:
    features = _synthetic_features()
    cutoff = features["date"].max()
    outcome = train_outcome_model(features, cutoff, persist=False)
    goals = train_goal_model(features, cutoff, persist=False)
    row = features.iloc[[-1]]

    assert goals.home_model.predict(row[GOAL_FEATURE_COLUMNS]).shape == (1,)
    assert outcome.challenger_model.predict_proba(row[FEATURE_COLUMNS]).shape == (1, 3)
