from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss

from src.models.scoreline import outcome_probabilities, scoreline_matrix, top_scorelines


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    classes = y_prob.shape[1]
    encoded = np.eye(classes)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((encoded - y_prob) ** 2, axis=1)))


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual_probability = y_prob[np.arange(len(y_prob)), np.asarray(y_true, dtype=int)]
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1])))),
        "brier_score": float(multiclass_brier_score(y_true, y_prob)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "average_probability_actual_outcome": float(np.mean(actual_probability)),
        "ranked_probability_score": ranked_probability_score(y_true, y_prob),
        "calibration_error": multiclass_calibration_error(y_true, y_prob),
    }


def multiclass_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    actual = np.eye(y_prob.shape[1])[np.asarray(y_true, dtype=int)].ravel()
    predicted = np.asarray(y_prob, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (predicted >= lower) & (predicted < upper if upper < 1.0 else predicted <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(actual[mask].mean()) - float(predicted[mask].mean()))
    return float(error)


def ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    classes = y_prob.shape[1]
    encoded = np.eye(classes)[np.asarray(y_true, dtype=int)]
    cumulative_error = np.cumsum(y_prob, axis=1)[:, :-1] - np.cumsum(encoded, axis=1)[:, :-1]
    return float(np.mean(np.sum(cumulative_error**2, axis=1) / (classes - 1)))


def compute_scoreline_metrics(
    evaluation_frame,
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
    dixon_coles_rho: float = 0.0,
    max_goals: int = 10,
) -> dict[str, float]:
    home_actual = evaluation_frame["home_score"].to_numpy(dtype=int)
    away_actual = evaluation_frame["away_score"].to_numpy(dtype=int)
    scoreline_losses: list[float] = []
    top_1_hits: list[float] = []
    top_5_hits: list[float] = []
    outcome_rows: list[np.ndarray] = []
    for actual_home, actual_away, home_lambda, away_lambda in zip(
        home_actual,
        away_actual,
        home_lambdas,
        away_lambdas,
    ):
        matrix = scoreline_matrix(home_lambda, away_lambda, max_goals=max_goals, dixon_coles_rho=dixon_coles_rho)
        clipped_home = min(int(actual_home), max_goals)
        clipped_away = min(int(actual_away), max_goals)
        actual_probability = float(matrix.iloc[clipped_home, clipped_away])
        scoreline_losses.append(-float(np.log(max(actual_probability, 1e-12))))
        predicted_scores = [entry["score"] for entry in top_scorelines(matrix, count=5)]
        actual_score = f"{actual_home}-{actual_away}"
        top_1_hits.append(float(actual_score == predicted_scores[0]))
        top_5_hits.append(float(actual_score in predicted_scores))
        outcome_rows.append(outcome_probabilities(matrix))
    outcome_prob = np.asarray(outcome_rows)
    return {
        "home_goal_mae": float(np.mean(np.abs(home_actual - home_lambdas))),
        "away_goal_mae": float(np.mean(np.abs(away_actual - away_lambdas))),
        "total_goals_mae": float(np.mean(np.abs((home_actual + away_actual) - (home_lambdas + away_lambdas)))),
        "goal_difference_mae": float(np.mean(np.abs((home_actual - away_actual) - (home_lambdas - away_lambdas)))),
        "mean_goal_mae": float(
            np.mean(np.concatenate([np.abs(home_actual - home_lambdas), np.abs(away_actual - away_lambdas)]))
        ),
        "scoreline_log_loss": float(np.mean(scoreline_losses)),
        "top_1_scoreline_accuracy": float(np.mean(top_1_hits)),
        "top_5_scoreline_hit_rate": float(np.mean(top_5_hits)),
        **compute_classification_metrics(
            evaluation_frame["result"].to_numpy(dtype=int),
            outcome_prob,
            np.argmax(outcome_prob, axis=1),
        ),
    }


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    positive_prob = y_prob[:, 0]
    table: list[dict[str, float]] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (positive_prob >= lower) & (positive_prob < upper)
        if not mask.any():
            continue
        table.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": float(mask.sum()),
                "observed_rate": float(np.mean(np.asarray(y_true)[mask] == 0)),
                "predicted_rate": float(np.mean(positive_prob[mask])),
            }
        )
    return table
