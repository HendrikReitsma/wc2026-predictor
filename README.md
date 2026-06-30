# World Cup 2026 Predictor

Reproducible pre-match forecasting for the FIFA World Cup 2026.

The project combines chronological team ratings, rolling team form, Poisson expected-goals models, a margin/outcome adjustment, and Monte Carlo tournament simulation. It is built for transparent experimentation, not betting guarantees.

## Current Performance

These metrics evaluate the frozen pre-tournament group-stage predictions against completed World Cup 2026 group matches.

<!-- wc2026-metrics:start -->
_Last updated by `python scripts/update_results.py` at 2026-06-30 17:14 UTC._

| Metric | Current value |
| --- | ---: |
| Outcome accuracy | 63.9% |
| Correct outcomes / total matches | 46 / 72 |
| Log loss | 0.878 |
| Brier score | 0.516 |
| Ranked Probability Score | 0.153 |
| Avg probability on actual result | 49.4% |
| Exact score hit rate | 12.5% (9 / 72) |
| Top-5 scoreline hit rate | 44.4% (32 / 72) |
| Total goals expected vs actual | 202.3 vs 213 |
| Rounded-xG outcome accuracy | 65.3% (47 / 72) |
<!-- wc2026-metrics:end -->

Full evaluation report: [reports/worldcup_2026_group_stage_model_performance.md](reports/worldcup_2026_group_stage_model_performance.md)

## What This Does

- Builds leakage-safe pre-match features from international results.
- Trains home/away goal models and outcome models.
- Predicts match expected goals, W/D/L probabilities, scorelines, totals, and clean sheets.
- Simulates the World Cup tournament with 10,000 Monte Carlo runs by default.
- Evaluates completed group-stage predictions and updates this README.

## Install

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

For a non-editable minimal install:

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e .[data]  # Kaggle download support
pip install -e .[viz]   # report figures
pip install -e .[dev]   # pytest
```

## Update After Matches

After new group-stage results are available, run:

```bash
python scripts/update_results.py
```

That command:

- fetches the published World Cup 2026 group-stage scores,
- merges them into `data/manual/worldcup_2026_group_results.csv` without duplicating matches,
- re-runs the group-stage evaluation,
- rewrites `reports/worldcup_2026_group_stage_model_performance.md`,
- updates the metrics table in this README.

If the source page is unavailable but the local results CSV is already current:

```bash
python scripts/update_results.py --skip-fetch
```

## Reproduce The Current Forecast

Run commands from the repository root.

```bash
python scripts/fetch_data.py
python scripts/train_models.py --cutoff-date 2026-06-10
python scripts/predict_worldcup_2026.py --cutoff-date 2026-06-10 --n-simulations 10000
python scripts/update_results.py
python -m pytest -q
```

After the group stage, the remaining fixed knockout bracket can be simulated with actual group results locked in:

```bash
python scripts/train_models.py --cutoff-date 2026-06-27
python scripts/predict_remaining_worldcup_2026.py --cutoff-date 2026-06-27 --n-simulations 10000
```

## Important Files And Folders

| Path | Purpose |
| --- | --- |
| `config/config.yaml` | Main configuration and default simulation settings. |
| `data/manual/` | Small checked-in manual inputs: fixtures, mappings, group results, allocation table sources. |
| `data/raw/` | Raw downloaded historical results. Ignored because it is reproducible/large. |
| `data/processed/` | Generated feature tables and intermediate outputs. Ignored. |
| `data/predictions/` | Generated predictions and metric files. Mostly ignored except the small current metrics JSON. |
| `models/` | Generated model artifacts. Ignored. |
| `scripts/` | Runnable workflow entry points. |
| `src/` | Model, feature, simulation, and evaluation logic. |
| `reports/` | Public-facing reports and figures. |
| `archive/` | Old exploratory work kept for reference. |

## Main Metrics

- **Outcome accuracy**: share of matches where the highest-probability W/D/L outcome was correct.
- **Log loss**: rewards probability assigned to the actual result; lower is better and confident wrong calls are punished strongly.
- **Brier score**: squared error over the three outcome probabilities; lower is better.
- **RPS**: Ranked Probability Score for ordered outcomes home/draw/away; lower is better.
- **Average probability on actual result**: mean probability the model assigned to what actually happened.
- **Exact score hit rate**: share of matches where the single most likely score was exactly right.
- **Top-5 scoreline hit rate**: share where the actual score was among the model's five most likely scorelines.
- **Rounded-xG outcome accuracy**: evaluates a Scorito-style entered score created by rounding expected home and away goals.

## Project Notes

- The main forecast snapshot uses a cutoff of `2026-06-10`, before the World Cup group stage.
- The model does not use betting odds, player availability, or post-match ranking information.
- All rolling features and ratings are computed chronologically before each match is used to update state.
- Extreme scores are retained as observations; only goal-model training targets are capped for robustness.
- A 72-match group stage is still a small sample, so close metric differences should not be overinterpreted.

## TODOs

- Automate knockout-result ingestion and evaluation. The current `update_results.py` workflow is focused on the completed group-stage forecast evaluation.
- Add a small public sample dataset if you want the repo to run fully without downloading `data/raw/results.csv`.

## More Detail

- Method paper: [reports/methodology_research_paper.md](reports/methodology_research_paper.md)
- Model card: [reports/model_card.md](reports/model_card.md)
- Remaining tournament report: [reports/worldcup_2026_remaining_prediction_report.md](reports/worldcup_2026_remaining_prediction_report.md)
- Public GitHub comparison: [reports/github_method_comparison.md](reports/github_method_comparison.md)

