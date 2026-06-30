from __future__ import annotations

import pandas as pd

from src.evaluation.training_strategy import _best_complete


def test_strategy_tie_break_prefers_better_scoreline_coverage() -> None:
    summary = pd.DataFrame(
        [
            {
                "world_cups_covered": 4,
                "goal_cap": 6,
                "avg_log_loss": 0.96,
                "avg_brier_score": 0.56,
                "avg_calibration_error": 0.06,
                "avg_scoreline_top_5_hit_rate": 0.54,
                "stability_score": 0.08,
            },
            {
                "world_cups_covered": 4,
                "goal_cap": 8,
                "avg_log_loss": 0.96,
                "avg_brier_score": 0.56,
                "avg_calibration_error": 0.06,
                "avg_scoreline_top_5_hit_rate": 0.56,
                "stability_score": 0.08,
            },
        ]
    )

    assert int(_best_complete(summary)["goal_cap"]) == 8
