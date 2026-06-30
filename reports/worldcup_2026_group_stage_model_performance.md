# Group Stage Model Performance

- Evaluated matches: **72**.
- Prediction cutoff used: **2026-06-10T00:00:00**.
- Actual result source: `data/manual/worldcup_2026_group_results.csv`.

## Headline Metrics

- Outcome accuracy: **63.9%**.
  - Correct outcome picks: **46 / 72**.
- Outcome log loss: **0.878**.
- Outcome Brier score: **0.516**.
- Outcome RPS from model probabilities: **0.153**.
- Average probability assigned to the actual outcome: **49.4%**.
- Exact-score top-1 hit rate: **12.5%** (9 / 72).
- Exact-score top-5 hit rate: **44.4%** (32 / 72).
- Mean per-team goal MAE: **0.958**.
- Goal-difference MAE: **1.296**.
- Total goals: expected **202.3**, actual **213**.
- Over 2.5 goals Brier score: **0.250**; 0.5-threshold accuracy **59.7%**.

## Rounded Expected-Goals Score Evaluation

This section evaluates the prediction as if the entered score was made by rounding expected home and away goals with spreadsheet-style rounding: `floor(x + 0.5)`. For example, `1.41-0.89` becomes `1-1`, and `1.94-0.45` becomes `2-0`.

- Rounded-score outcome accuracy: **65.3%** (47 / 72).
- Rounded deterministic RPS: **0.194**.
- Rounded exact-score accuracy: **8.3%** (6 / 72).
- Rounded predicted draws: **16**; actual draws: **20**.
- Rounded home-goal exact rate: **29.2%**.
- Rounded away-goal exact rate: **34.7%**.

## Outcome Mix

| outcome | actual_count | rounded_pred_count | probability_argmax_count |
| --- | --- | --- | --- |
| away | 18 | 21 | 27 |
| draw | 20 | 16 | 0 |
| home | 34 | 35 | 45 |

## Calibration By Favorite Confidence

| confidence_bin | matches | mean_confidence | accuracy |
| --- | --- | --- | --- |
| 0.0-0.4 | 4 | 0.365 | 0.250 |
| 0.4-0.5 | 15 | 0.453 | 0.467 |
| 0.5-0.6 | 12 | 0.567 | 0.667 |
| 0.6-0.7 | 15 | 0.650 | 0.800 |
| 0.7-0.8 | 16 | 0.744 | 0.750 |
| 0.8-1.0 | 10 | 0.878 | 0.600 |

## Performance By Group

| group | matches | outcome_accuracy | top5_score_rate | mean_actual_prob | expected_goals | actual_goals |
| --- | --- | --- | --- | --- | --- | --- |
| A | 6 | 0.667 | 0.667 | 0.477 | 16.072 | 12 |
| B | 6 | 0.667 | 0.333 | 0.507 | 17.854 | 22 |
| C | 6 | 0.833 | 0.833 | 0.618 | 17.474 | 16 |
| D | 6 | 0.500 | 0.500 | 0.412 | 15.841 | 15 |
| E | 6 | 0.667 | 0.500 | 0.480 | 18.286 | 17 |
| F | 6 | 0.667 | 0.167 | 0.512 | 17.141 | 26 |
| G | 6 | 0.333 | 0.333 | 0.369 | 16.051 | 16 |
| H | 6 | 0.333 | 0.333 | 0.358 | 17.657 | 11 |
| I | 6 | 1.000 | 0.167 | 0.610 | 16.153 | 27 |
| J | 6 | 0.833 | 0.500 | 0.615 | 16.472 | 22 |
| K | 6 | 0.500 | 0.333 | 0.475 | 15.995 | 16 |
| L | 6 | 0.667 | 0.667 | 0.494 | 17.332 | 13 |

## Rounded Expected-Goals Performance By Group

| group | matches | rounded_outcome_accuracy | rounded_exact_score_rate | mean_home_goal_error | mean_away_goal_error | total_goal_error |
| --- | --- | --- | --- | --- | --- | --- |
| A | 6 | 0.333 | 0.000 | 0.653 | 0.788 | 1.005 |
| B | 6 | 0.667 | 0.167 | 1.319 | 0.542 | 1.441 |
| C | 6 | 1.000 | 0.167 | 0.848 | 0.740 | 1.260 |
| D | 6 | 0.333 | 0.000 | 1.265 | 0.862 | 1.591 |
| E | 6 | 0.500 | 0.167 | 1.466 | 0.514 | 1.872 |
| F | 6 | 0.833 | 0.000 | 1.519 | 0.705 | 1.832 |
| G | 6 | 0.500 | 0.000 | 0.572 | 1.278 | 1.832 |
| H | 6 | 0.500 | 0.000 | 1.115 | 0.951 | 1.693 |
| I | 6 | 1.000 | 0.000 | 1.165 | 1.115 | 1.808 |
| J | 6 | 1.000 | 0.333 | 0.761 | 0.597 | 1.079 |
| K | 6 | 0.667 | 0.000 | 1.476 | 0.721 | 1.747 |
| L | 6 | 0.500 | 0.167 | 1.225 | 0.806 | 1.903 |

## Exact Score Hits

| match_id | home_team | away_team | actual_score | most_likely_score |
| --- | --- | --- | --- | --- |
| 1 | Mexico | South Africa | 2-0 | 2-0 |
| 5 | Haiti | Scotland | 0-1 | 0-1 |
| 7 | Brazil | Morocco | 1-1 | 1-1 |
| 25 | Mexico | South Korea | 1-0 | 1-0 |
| 30 | Scotland | Morocco | 0-1 | 0-1 |
| 46 | Panama | Croatia | 0-1 | 0-1 |
| 55 | Curaçao | Ivory Coast | 0-2 | 0-2 |
| 66 | Uruguay | Spain | 0-1 | 0-1 |
| 67 | Panama | England | 0-2 | 0-2 |

## Highest Probability Correct Outcomes

| match_id | group | home_team | away_team | actual_score | actual_outcome | actual_outcome_probability | predicted_outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | E | Germany | Curaçao | 7-1 | home | 0.913 | home |
| 27 | B | Canada | Qatar | 6-0 | home | 0.889 | home |
| 39 | H | Spain | Saudi Arabia | 4-0 | home | 0.881 | home |
| 1 | A | Mexico | South Africa | 2-0 | home | 0.857 | home |
| 29 | C | Brazil | Haiti | 3-0 | home | 0.857 | home |
| 70 | J | Jordan | Argentina | 1-3 | away | 0.847 | away |
| 68 | L | Croatia | Ghana | 2-1 | home | 0.796 | home |
| 41 | I | France | Iraq | 3-0 | home | 0.785 | home |

## Lowest Probability Actual Outcomes

| match_id | group | home_team | away_team | actual_score | actual_outcome | actual_outcome_probability | predicted_outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | H | Spain | Cape Verde | 0-0 | draw | 0.064 | home |
| 34 | E | Ecuador | Curaçao | 0-0 | draw | 0.071 | home |
| 8 | B | Qatar | Switzerland | 1-1 | draw | 0.097 | away |
| 48 | L | England | Ghana | 0-0 | draw | 0.097 | home |
| 40 | H | Uruguay | Cape Verde | 2-2 | draw | 0.153 | home |
| 3 | B | Canada | Bosnia and Herzegovina | 1-1 | draw | 0.159 | home |
| 59 | F | Japan | Sweden | 1-1 | draw | 0.165 | home |
| 21 | K | Portugal | DR Congo | 1-1 | draw | 0.168 | home |

## Rounded Exact Score Hits

| match_id | home_team | away_team | actual_score | rounded_pred_score |
| --- | --- | --- | --- | --- |
| 7 | Brazil | Morocco | 1-1 | 1-1 |
| 33 | Germany | Ivory Coast | 2-1 | 2-1 |
| 44 | Jordan | Algeria | 1-2 | 1-2 |
| 54 | Switzerland | Canada | 2-1 | 2-1 |
| 68 | Croatia | Ghana | 2-1 | 2-1 |
| 70 | Jordan | Argentina | 1-3 | 1-3 |

## Rounded Outcome Misses

| match_id | group | home_team | away_team | expected_goals_home | expected_goals_away | rounded_pred_score | actual_score | rounded_pred_outcome | actual_outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | A | South Korea | Czech Republic | 1.489 | 1.072 | 1-1 | 2-1 | draw | home |
| 28 | A | Czech Republic | South Africa | 1.721 | 0.950 | 2-1 | 1-1 | home | draw |
| 51 | A | South Africa | South Korea | 0.823 | 1.762 | 1-2 | 1-0 | away | home |
| 52 | A | Czech Republic | Mexico | 1.009 | 1.453 | 1-1 | 0-3 | draw | away |
| 3 | B | Canada | Bosnia and Herzegovina | 2.133 | 0.576 | 2-1 | 1-1 | home | draw |
| 8 | B | Qatar | Switzerland | 0.726 | 2.816 | 1-3 | 1-1 | away | draw |
| 4 | D | United States | Paraguay | 1.379 | 1.187 | 1-1 | 4-1 | draw | home |
| 6 | D | Australia | Turkey | 1.127 | 1.431 | 1-1 | 2-0 | draw | home |
| 31 | D | United States | Australia | 1.451 | 1.145 | 1-1 | 2-0 | draw | home |
| 32 | D | Turkey | Paraguay | 1.465 | 1.154 | 1-1 | 0-1 | draw | away |
| 9 | E | Ivory Coast | Ecuador | 0.807 | 1.604 | 1-2 | 1-0 | away | home |
| 34 | E | Ecuador | Curaçao | 3.053 | 0.641 | 3-1 | 0-0 | home | draw |
| 56 | E | Ecuador | Germany | 1.170 | 1.129 | 1-1 | 2-1 | draw | home |
| 59 | F | Japan | Sweden | 2.276 | 0.790 | 2-1 | 1-1 | home | draw |
| 13 | G | Iran | New Zealand | 1.944 | 0.823 | 2-1 | 2-2 | home | draw |
| 15 | G | Belgium | Egypt | 1.688 | 0.946 | 2-1 | 1-1 | home | draw |
| 37 | G | Belgium | Iran | 1.517 | 1.089 | 2-1 | 0-0 | home | draw |
| 14 | H | Spain | Cape Verde | 3.194 | 0.659 | 3-1 | 0-0 | home | draw |
| 16 | H | Saudi Arabia | Uruguay | 0.680 | 1.897 | 1-2 | 1-1 | away | draw |
| 40 | H | Uruguay | Cape Verde | 2.191 | 0.600 | 2-1 | 2-2 | home | draw |
| 21 | K | Portugal | DR Congo | 2.142 | 0.674 | 2-1 | 1-1 | home | draw |
| 72 | K | DR Congo | Uzbekistan | 1.031 | 1.256 | 1-1 | 3-1 | draw | home |
| 22 | L | England | Croatia | 1.486 | 0.970 | 1-1 | 4-2 | draw | home |
| 24 | L | Ghana | Panama | 0.955 | 1.756 | 1-2 | 1-0 | away | home |
| 48 | L | England | Ghana | 2.727 | 0.603 | 3-1 | 0-0 | home | draw |

## Short Read

For this group stage, the rounded expected-goals score was slightly better for outcome picking than simply taking the highest-probability W/D/L class: 47 correct outcomes versus 46. The reason is intuitive for a score-prediction pool: rounding created 16 predicted draws, while the probability argmax predicted no draws at all. That helped in a group stage with 20 actual draws.

The tradeoff is exact scores. The model's modal score hit 9 exact results, while rounded expected goals hit 6. The expected-goals numbers were informative for broad outcomes and aggregate scoring, but turning them into one rounded score still loses a lot of distributional information.