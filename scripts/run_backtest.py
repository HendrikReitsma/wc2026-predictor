from __future__ import annotations

import _bootstrap  # noqa: F401

from src.evaluation.backtest import backtest_world_cups
from src.evaluation.target_experiments import run_target_experiments
from src.evaluation.training_strategy import run_training_strategy_search
from src.evaluation.indirect_model_backtest import run_indirect_model_backtest


def main() -> None:
    backtest_world_cups()
    run_target_experiments()
    run_training_strategy_search()
    run_indirect_model_backtest()


if __name__ == "__main__":
    main()
