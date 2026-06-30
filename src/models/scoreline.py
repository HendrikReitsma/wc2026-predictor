from __future__ import annotations

from math import exp, factorial

import numpy as np
import pandas as pd


def _poisson_pmf(rate: float, goals: int) -> float:
    return float(exp(-rate) * rate**goals / factorial(goals))


def scoreline_matrix(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 10,
    dixon_coles_rho: float = 0.0,
) -> pd.DataFrame:
    """Return a normalized score matrix, with optional Dixon-Coles low-score adjustment."""
    home_lambda = float(np.clip(home_lambda, 0.05, 5.5))
    away_lambda = float(np.clip(away_lambda, 0.05, 5.5))
    matrix = np.outer(
        [_poisson_pmf(home_lambda, goals) for goals in range(max_goals + 1)],
        [_poisson_pmf(away_lambda, goals) for goals in range(max_goals + 1)],
    )
    rho = float(np.clip(dixon_coles_rho, -0.25, 0.25))
    if rho:
        matrix[0, 0] *= max(0.0, 1.0 - home_lambda * away_lambda * rho)
        matrix[0, 1] *= max(0.0, 1.0 + home_lambda * rho)
        matrix[1, 0] *= max(0.0, 1.0 + away_lambda * rho)
        matrix[1, 1] *= max(0.0, 1.0 - rho)
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("Scoreline probability matrix has no probability mass.")
    return pd.DataFrame(matrix / total, index=range(max_goals + 1), columns=range(max_goals + 1))


def outcome_probabilities(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy()
    probabilities = np.array(
        [np.tril(values, -1).sum(), np.trace(values), np.triu(values, 1).sum()],
        dtype=float,
    )
    return probabilities / probabilities.sum()


def reweight_scoreline_outcomes(matrix: pd.DataFrame, target_probabilities: np.ndarray) -> pd.DataFrame:
    """Preserve within-outcome scoreline shape while matching target W/D/L probabilities."""
    values = matrix.to_numpy(dtype=float).copy()
    masks = [np.tril(np.ones_like(values, dtype=bool), -1), np.eye(len(values), dtype=bool), np.triu(np.ones_like(values, dtype=bool), 1)]
    targets = np.asarray(target_probabilities, dtype=float)
    targets = np.clip(targets, 1e-12, 1.0)
    targets = targets / targets.sum()
    for mask, target in zip(masks, targets):
        current = float(values[mask].sum())
        if current > 0:
            values[mask] *= target / current
    values /= values.sum()
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def expected_goals_from_matrix(matrix: pd.DataFrame) -> tuple[float, float]:
    values = matrix.to_numpy(dtype=float)
    return (
        float((values * matrix.index.to_numpy()[:, None]).sum()),
        float((values * matrix.columns.to_numpy()[None, :]).sum()),
    )


def scoreline_probabilities(matrix: pd.DataFrame) -> dict[str, float]:
    values = matrix.to_numpy()
    total_goals = np.add.outer(matrix.index.to_numpy(), matrix.columns.to_numpy())
    return {
        "p_over_2_5_goals": float(values[total_goals >= 3].sum()),
        "p_under_2_5_goals": float(values[total_goals <= 2].sum()),
        "p_both_teams_score": float(values[1:, 1:].sum()),
        "p_clean_sheet_home": float(values[:, 0].sum()),
        "p_clean_sheet_away": float(values[0, :].sum()),
    }


def top_scorelines(matrix: pd.DataFrame, count: int = 5) -> list[dict[str, float | str]]:
    scores = matrix.stack().sort_values(ascending=False).head(count)
    return [
        {"score": f"{int(home)}-{int(away)}", "probability": float(probability)}
        for (home, away), probability in scores.items()
    ]
