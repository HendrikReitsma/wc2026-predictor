# Model Card

## Identity

- Model/version: `wc2026-predictor 0.1.0`, selected target/model `margin_class_classifier__uncalibrated`.
- Goal model: `poisson` with `attack_defence_poisson`.
- Training strategy: `1990` onward, `aggressive` weights, `standard_elo` K scale `1.2`.
- Probability adjustment: draw correction `similar_strength`, calibration `uncalibrated`.
- Indirect correction: `baseline`; comparison challenger `trend_small`.
- Prediction cutoff: `2026-06-10`.

## Targets

- Home and away goals, capped only as training targets.
- Derived home-win, draw, away-win, scoreline, and tournament advancement probabilities.
- Seven-class match-margin probabilities used as the selected outcome correction target.

## Training And Excluded Data

- Historical results from 1990 through 2026-06-10.
- Excludes unplayed rows, rows after the cutoff, and clear duplicate records.
- Does not use squads, injuries, betting odds, post-match rankings, or future tournament results.

## Feature Groups

- Goal features: `home_elo_pre`, `away_elo_pre`, `elo_diff`, `neutral`, `is_friendly`, `is_world_cup`, `is_continental_competition`, `home_advantage_flag`, `host_country_flag`, `tournament_importance`, `rolling_goals_for_home_5`, `rolling_goals_against_home_5`, `rolling_goals_for_away_5`, `rolling_goals_against_away_5`, `rolling_goals_for_home_10`, `rolling_goals_against_home_10`, `rolling_goals_for_away_10`, `rolling_goals_against_away_10`, `home_attack_rating_pre`, `home_defence_rating_pre`, `away_attack_rating_pre`, `away_defence_rating_pre`.
- Challenger outcome features: `elo_diff`, `neutral`, `is_friendly`, `is_world_cup`, `is_continental_competition`, `home_advantage_flag`, `host_country_flag`, `tournament_importance`, `days_since_last_match_home`, `days_since_last_match_away`, `rest_days_diff`, `recent_form_points_home_5`, `recent_form_points_away_5`, `recent_form_points_home_10`, `recent_form_points_away_10`, `recent_goal_diff_home_5`, `recent_goal_diff_away_5`, `recent_goal_diff_home_10`, `recent_goal_diff_away_10`.

## Leakage Controls

- Features are captured before state updates; same-day matches share only prior-day state.
- Elo is pre-match Elo. Rolling form excludes the current match.
- Frozen World Cup backtests train only through 2006, 2010, 2014, and 2018.
- Future-form research labels are kept separate and never joined into same-match features.
- Calibration uses a chronological tail of the available training period.

## Validation Design

- Primary: average outcome log loss across World Cups 2010, 2014, 2018, and 2022.
- Secondary: Brier score, ranked probability score, goal MAE, exact-score log loss, top-1/top-5 score hits, stability, then accuracy.

## Known Failure Modes

- Sudden squad/injury/tactical changes are invisible.
- Sparse teams and cross-confederation comparisons are uncertain.
- Scoreline tails and penalties are simplified.
- Small tournament test sets make model ranking noisy.

## Intended Use

- Scenario analysis, probabilistic tournament forecasting, and model comparison.

## Not Intended Use

- Claims of certainty, gambling guarantees, player-level decisions, or live in-match forecasting.

## Reproducibility

```bash
python scripts/fetch_data.py
python scripts/build_features.py --cutoff-date 2026-06-10
python scripts/train_models.py --cutoff-date 2026-06-10
python scripts/run_backtest.py
python scripts/predict_worldcup_2026.py --cutoff-date 2026-06-10 --n-simulations 10000
python -m pytest -q
```