# Current Performance

This is the compact public performance report. Detailed generated audits stay in `docs/internal/` so the public `reports/` folder remains readable.

## Total Metrics

_Last updated by `python scripts/update_results.py` at 2026-06-30 19:07 UTC._

| Metric | Current value |
| --- | ---: |
| Matches evaluated | 77 |
| Outcome accuracy | 63.6% |
| Correct outcomes / total | 49 / 77 |
| Ranked Probability Score | 0.154 |
| Log loss | 0.884 |
| Brier score | 0.520 |
| Avg probability on actual result | 48.7% |
| Exact score hit rate | 15.6% |
| Top-5 scoreline hit rate | 48.1% |
| Total goals expected vs actual | 215.4 vs 224 |

## Scope

- Total evaluated matches: **77**.
- Group-stage matches evaluated: **72**.
- Knockout matches evaluated: **5**.
- Group-stage detail: [`docs/internal/worldcup_2026_group_stage_model_performance.md`](../docs/internal/worldcup_2026_group_stage_model_performance.md).
- Knockout detail: [`docs/internal/worldcup_2026_knockout_model_performance.md`](../docs/internal/worldcup_2026_knockout_model_performance.md).
- Remaining-tournament forecast: [`docs/internal/worldcup_2026_remaining_prediction_report.md`](../docs/internal/worldcup_2026_remaining_prediction_report.md).
