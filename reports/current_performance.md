# Current Performance

This is the compact public performance report. Detailed generated audits stay in `docs/internal/` so the public `reports/` folder remains readable.

## Total Metrics

_Last updated by `python scripts/update_results.py` at 2026-06-30 18:45 UTC._

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

## Scope

- Total evaluated matches: **76**.
- Group-stage matches evaluated: **72**.
- Knockout matches evaluated: **4**.
- Group-stage detail: [`docs/internal/worldcup_2026_group_stage_model_performance.md`](../docs/internal/worldcup_2026_group_stage_model_performance.md).
- Knockout detail: [`docs/internal/worldcup_2026_knockout_model_performance.md`](../docs/internal/worldcup_2026_knockout_model_performance.md).
- Remaining-tournament forecast: [`docs/internal/worldcup_2026_remaining_prediction_report.md`](../docs/internal/worldcup_2026_remaining_prediction_report.md).
