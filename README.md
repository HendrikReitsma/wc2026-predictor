# World Cup 2026 Predictor

Transparent FIFA World Cup 2026 forecasting with Elo ratings, Poisson expected goals, Monte Carlo simulation, and live evaluation as results come in.

The group-stage forecast was frozen before the tournament. Completed matches are scored against the original probabilities; results are not retro-fitted.

## Current Record

<!-- wc2026-metrics:start -->
_Last updated by `python scripts/update_results.py` at 2026-06-30 18:35 UTC._

| Metric | Current value |
| --- | ---: |
| Matches evaluated | 76 |
| Outcome accuracy | 63.2% |
| Correct outcomes / total | 48 / 76 |
| Ranked Probability Score | 0.153 |
| Log loss | 0.885 |
| Brier score | 0.521 |
| Avg probability on actual result | 48.8% |
| Exact score hit rate | 15.8% |
| Top-5 scoreline hit rate | 47.4% |
| Total goals expected vs actual | 212.5 vs 221 |
<!-- wc2026-metrics:end -->

Current public performance report: [reports/current_performance.md](reports/current_performance.md)

## Why Trust This?

- Predictions use chronological pre-match features only.
- No betting odds, player availability, or post-match ranking information is used.
- Results are fetched and merged into fixed CSVs, then evaluated against saved prediction files.
- The main refresh command is one script: `python scripts/update_results.py`.

## Honest Findings

Simple rating/goal models were hard to beat. Extra ML, calibration, ensembles, rest-day features, and heavier recent-form machinery mostly added noise. The deployed model is:

```text
pre-match Elo + attack/defence Poisson + margin-class adjustment
```

The Poisson layer is kept because it gives one coherent distribution for expected goals, scorelines, totals, clean sheets, and tournament simulation. Rounded expected goals worked reasonably well for Scorito-style score picks, but the full probability distribution is better for evaluation.

More detail: [reports/methodology.md](reports/methodology.md) and [reports/benchmark_comparison.md](reports/benchmark_comparison.md)

## Quickstart

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Reproduce the current forecast and evaluation:

```bash
python scripts/fetch_data.py
python scripts/train_models.py --cutoff-date 2026-06-10
python scripts/predict_worldcup_2026.py --cutoff-date 2026-06-10 --n-simulations 10000
python scripts/update_results.py
python -m pytest -q
```

## Update After Matches

After new World Cup results are available:

```bash
python scripts/update_results.py
```

The command fetches available results, avoids duplicate match rows, re-runs evaluation, updates reports, and rewrites the metrics table in this README.

If the result source is unavailable but local result CSVs are current:

```bash
python scripts/update_results.py --skip-fetch
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `config/` | Main configuration. |
| `data/manual/` | Small checked-in inputs: fixtures, mappings, completed World Cup results. |
| `scripts/` | Runnable workflow entry points. |
| `src/` | Feature, model, simulation, and evaluation logic. |
| `reports/` | Public reports and referenced methodology figures. |
| `docs/internal/` | Generated deep-dive reports, audits, and historical comparisons. |
| `tests/` | Regression and parser tests. |

Generated raw data, processed features, predictions, and model artifacts are ignored by Git where practical.

## Documentation

- [Current performance](reports/current_performance.md)
- [Methodology](reports/methodology.md)
- [Model card](reports/model_card.md)
- [Public GitHub comparison](reports/benchmark_comparison.md)

## License

MIT
