# Current Performance

This is the compact public performance report. Detailed generated audits stay in `docs/internal/` so the public `reports/` folder remains readable.

## Total Metrics

_Last updated by `python scripts/update_results.py` at 2026-07-02 10:20 UTC._

| Metric | Current value |
| --- | ---: |
| Matches evaluated | 82 |
| Outcome accuracy | 65.9% |
| Correct outcomes / total | 54 / 82 |
| Ranked Probability Score | 0.150 |
| Log loss | 0.859 |
| Brier score | 0.502 |
| Avg probability on actual result | 49.7% |
| Exact score hit rate | 15.9% |
| Top-5 scoreline hit rate | 50.0% |
| Total goals expected vs actual | 229.9 vs 239 |

## Scope

- Total evaluated matches: **82**.
- Group-stage matches evaluated: **72**.
- Knockout matches evaluated: **10**.
- Group-stage detail: [`docs/internal/worldcup_2026_group_stage_model_performance.md`](../docs/internal/worldcup_2026_group_stage_model_performance.md).
- Knockout detail: [`docs/internal/worldcup_2026_knockout_model_performance.md`](../docs/internal/worldcup_2026_knockout_model_performance.md).
- Remaining-tournament forecast: [`docs/internal/worldcup_2026_remaining_prediction_report.md`](../docs/internal/worldcup_2026_remaining_prediction_report.md).
