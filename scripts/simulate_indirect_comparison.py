from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json

from src.models.predict_match import load_prediction_context, predict_match
from src.simulation.simulate_match import _match_distribution
from src.simulation.simulate_tournament import simulate_tournament
from src.utils.paths import MODELS_DIR, PREDICTIONS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the best rejected indirect challenger.")
    parser.add_argument("--n-simulations", type=int, default=10000)
    args = parser.parse_args()
    path = MODELS_DIR / "indirect_adjustments.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    comparison = dict(original)
    comparison["selected_variant"] = original["comparison_variant"]
    comparison["selected_adjustments"] = original["comparison_adjustments"]
    try:
        path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        load_prediction_context.cache_clear()
        predict_match.cache_clear()
        _match_distribution.cache_clear()
        summary = simulate_tournament(n_simulations=args.n_simulations, seed=4242)
        summary.to_csv(PREDICTIONS_DIR / "tournament_simulation_summary_indirect_comparison.csv", index=False)
    finally:
        path.write_text(json.dumps(original, indent=2), encoding="utf-8")
        load_prediction_context.cache_clear()
        predict_match.cache_clear()
        _match_distribution.cache_clear()


if __name__ == "__main__":
    main()
