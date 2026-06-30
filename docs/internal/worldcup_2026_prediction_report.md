# World Cup 2026 Prediction Report

## A. Executive Summary

- Selected model: **margin_class_classifier__uncalibrated**, using **poisson / attack_defence_poisson** goals as its score-distribution base.
- Selected training strategy: **1990 onward, no time decay, aggressive importance weights, standard_elo K scale 1.2**.
- Indirect model decision: **baseline**; Best indirect challenger trend_small did not clear all configured proof checks; baseline retained.
- Training cutoff: **2026-06-10**; historical matches used: **32,281**.
- Tournament simulations: **10000**, random seed 42.
- Data sources: martj42 international results plus the official FIFA 2026 schedule/regulations.
- Biggest caveat: target-selection gains are small across only four frozen World Cups. Selected goal-model top-5 exact-score hit rate: 54.0%; mean goal MAE: 0.894.

### Top 10 Champion Probabilities

| team | p_champion | p_reach_final |
| --- | --- | --- |
| Argentina | 0.208 | 0.308 |
| Spain | 0.197 | 0.308 |
| France | 0.093 | 0.173 |
| England | 0.061 | 0.122 |
| Brazil | 0.057 | 0.115 |
| Colombia | 0.054 | 0.113 |
| Portugal | 0.050 | 0.105 |
| Ecuador | 0.036 | 0.083 |
| Japan | 0.031 | 0.070 |
| Germany | 0.031 | 0.072 |

## B. Model Description

The selected system starts with two separately fitted Poisson goal regressors using pre-match Elo, rolling goals, attack/defence state, and match context. A seven-class margin classifier then estimates probabilities from away-win-by-3+ through home-win-by-3+. Its expected margin adjusts the Poisson score shape, and the score matrix is reweighted so its win/draw/loss totals match the margin probabilities.

Elo starts teams at 1500, adds a 65-point non-neutral home advantage, updates only after each match, and uses larger K-factors for more important matches plus a capped goal-difference multiplier. The selected Elo K-factor scale is **1.2**. The score matrix covers 0-0 through 10-10 and expected goals are capped at 5.5.

Dixon-Coles low-score adjustment rho is **0.0**. The final W/D/L probabilities use **similar_strength** draw correction with alpha **1.0188** and **uncalibrated** probability calibration.

Group matches are sampled from the same score matrix used for reporting. Knockout draws receive an extra-time Poisson period; remaining ties use a strength-weighted penalty probability capped between 40% and 60%.

## C. Final Feature List

### Team strength

`home_elo_pre`, `away_elo_pre`, `elo_diff`

### Recent form

No final features in this group.

### Attack/defence

`rolling_goals_for_home_5`, `rolling_goals_against_home_5`, `rolling_goals_for_away_5`, `rolling_goals_against_away_5`, `rolling_goals_for_home_10`, `rolling_goals_against_home_10`, `rolling_goals_for_away_10`, `rolling_goals_against_away_10`, `home_attack_rating_pre`, `home_defence_rating_pre`, `away_attack_rating_pre`, `away_defence_rating_pre`

### Match context

`neutral`, `is_friendly`, `is_world_cup`, `is_continental_competition`, `home_advantage_flag`, `host_country_flag`

### Tournament importance

`tournament_importance`

### Rest/travel/venue

`host_country_flag`

Ablation components that helped their stated comparison: Elo features, recent form, attack/defence features, match importance weighting, Poisson model.
Components that were neutral or hurt: opponent-adjusted form, rest days, venue/host features, time decay weighting, combined weighting, calibration, ML model, ensemble.
A feature is retained only when it belongs to the independently selected goal-model feature set or a separately evaluated challenger.

Home-confederation and reliable historical group/knockout-stage features are not used because the source data does not provide them consistently.

## D. Training Setup

- Training range: 1990-01-12 through 2026-06-09.
- Minimum year: 1990.
- Time decay: disabled.
- Match weighting: aggressive; configurable values are in `config/config.yaml`.
- Dynamic rating: standard_elo, K scale 1.2.
- Draw correction: similar_strength, alpha 1.0188.
- Validation: frozen World Cups with train cutoffs 2006, 2010, 2014, and 2018.
- Calibration: uncalibrated.
- Goal cap for training targets: 8.

## E. Backtest Performance

| model_name | half_life_years | average_log_loss | average_brier_score | average_accuracy | average_scoreline_top_5_hit_rate | notes | world_cup_2010_log_loss | world_cup_2014_log_loss | world_cup_2018_log_loss | world_cup_2022_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| poisson_goal | 12.000 | 0.965 | 0.568 | 0.584 | 0.540 | Uses attack_defence_poisson score distribution. | 0.921 | 0.920 | 0.943 | 1.075 |
| poisson_calibrated | 12.000 | 0.968 | 0.570 | 0.579 | 0.540 | Uses attack_defence_poisson score distribution. | 0.918 | 0.922 | 0.959 | 1.072 |
| elo_recent_form_weighted | 12.000 | 0.969 | 0.570 | 0.561 | nan | Outcome-only model; no coherent exact-score distribution. | 0.945 | 0.919 | 0.947 | 1.063 |
| elo_recent_form | 12.000 | 0.969 | 0.571 | 0.561 | nan | Outcome-only model; no coherent exact-score distribution. | 0.945 | 0.919 | 0.951 | 1.059 |
| elo_baseline | 12.000 | 0.969 | 0.572 | 0.576 | nan | Outcome-only model; no coherent exact-score distribution. | 0.942 | 0.938 | 0.964 | 1.031 |
| ensemble | 8.000 | 0.971 | 0.571 | 0.553 | 0.540 | Uses attack_defence_poisson score distribution. | 0.929 | 0.937 | 0.942 | 1.077 |
| ml_challenger | 8.000 | 0.981 | 0.577 | 0.550 | nan | Outcome-only model; no coherent exact-score distribution. | 0.941 | 0.970 | 0.949 | 1.063 |
| ml_calibrated | 8.000 | 0.989 | 0.579 | 0.556 | nan | Outcome-only model; no coherent exact-score distribution. | 0.945 | 0.962 | 0.948 | 1.103 |

### Alternative Target Comparison

| target_type | model_name | avg_log_loss | avg_brier_score | avg_calibration_error | avg_scoreline_top_5_hit_rate | stability_score | selected_for_final_model | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| margin_class | margin_class_classifier__uncalibrated | 0.962 | 0.565 | 0.059 | 0.555 | 0.076 | True | Lowest mean frozen-World-Cup log loss; Brier, calibration, scoreline quality, and stability reported alongside. |
| outcome | outcome_classifier__uncalibrated | 0.962 | 0.564 | 0.060 | 0.540 | 0.087 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goal_difference | goal_difference_regression__uncalibrated | 0.962 | 0.567 | 0.068 | 0.543 | 0.076 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| outcome | outcome_classifier__platt_sigmoid | 0.962 | 0.567 | 0.068 | 0.540 | 0.068 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goals | goal_counts_raw__uncalibrated | 0.962 | 0.566 | 0.060 | 0.537 | 0.077 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goals | current_poisson_baseline__uncalibrated | 0.963 | 0.567 | 0.058 | 0.540 | 0.075 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| margin_class | margin_class_classifier__platt_sigmoid | 0.963 | 0.568 | 0.065 | 0.555 | 0.066 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goals | goal_counts_capped__uncalibrated | 0.964 | 0.568 | 0.062 | 0.544 | 0.073 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goal_difference | goal_difference_regression__platt_sigmoid | 0.966 | 0.570 | 0.079 | 0.543 | 0.074 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goals | goal_counts_raw__platt_sigmoid | 0.966 | 0.569 | 0.081 | 0.537 | 0.071 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goals | current_poisson_baseline__platt_sigmoid | 0.967 | 0.570 | 0.073 | 0.540 | 0.072 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |
| goals | goal_counts_capped__platt_sigmoid | 0.967 | 0.570 | 0.067 | 0.544 | 0.071 | False | Not selected: did not beat the recommended model on primary frozen-World-Cup log loss. |

### Training Strategy Search

| search_stage | minimum_year | half_life_years | importance_profile | goal_cap | rating_model | elo_k_scale | avg_log_loss | avg_brier_score | avg_calibration_error | avg_scoreline_top_5_hit_rate | stability_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rating_model | 1990 | 1000000000.000 | aggressive | 8 | standard_elo | 1.200 | 0.961 | 0.565 | 0.060 | 0.540 | 0.081 |
| goal_cap | 1990 | 1000000000.000 | aggressive | 6 | standard_elo | 1.000 | 0.962 | 0.565 | 0.060 | 0.543 | 0.076 |
| goal_cap | 1990 | 1000000000.000 | aggressive | 8 | standard_elo | 1.000 | 0.962 | 0.565 | 0.060 | 0.555 | 0.076 |
| goal_cap | 1990 | 1000000000.000 | aggressive | 10 | standard_elo | 1.000 | 0.962 | 0.565 | 0.060 | 0.555 | 0.076 |
| rating_model | 1990 | 1000000000.000 | aggressive | 8 | standard_elo | 1.000 | 0.962 | 0.565 | 0.060 | 0.555 | 0.076 |
| window_decay_importance | 1990 | 1000000000.000 | aggressive | 8 | standard_elo | 1.000 | 0.962 | 0.565 | 0.060 | 0.555 | 0.076 |
| window_decay_importance | 1990 | 1000000000.000 | balanced | 8 | standard_elo | 1.000 | 0.962 | 0.566 | 0.065 | 0.559 | 0.075 |
| window_decay_importance | 1990 | 1000000000.000 | none | 8 | standard_elo | 1.000 | 0.962 | 0.566 | 0.063 | 0.552 | 0.073 |
| window_decay_importance | 1990 | 12.000 | aggressive | 8 | standard_elo | 1.000 | 0.962 | 0.566 | 0.063 | 0.548 | 0.077 |
| window_decay_importance | 1990 | 12.000 | balanced | 8 | standard_elo | 1.000 | 0.962 | 0.566 | 0.065 | 0.559 | 0.075 |
| window_decay_importance | 1990 | 8.000 | aggressive | 8 | standard_elo | 1.000 | 0.963 | 0.566 | 0.069 | 0.552 | 0.077 |
| window_decay_importance | 1990 | 8.000 | balanced | 8 | standard_elo | 1.000 | 0.963 | 0.567 | 0.068 | 0.552 | 0.076 |

### Calibration And Draw Correction

| probability_source | draw_correction | calibration_method | avg_draw_alpha | avg_log_loss | avg_brier_score | avg_calibration_error | avg_similar_strength_log_loss | avg_similar_strength_draw_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| margin | similar_strength | uncalibrated | 1.019 | 0.961 | 0.565 | 0.065 | 1.035 | 0.182 |
| poisson | none | uncalibrated | 1.000 | 0.962 | 0.566 | 0.058 | 1.036 | 0.184 |
| margin | similar_strength | sigmoid | 1.019 | 0.962 | 0.567 | 0.065 | 1.041 | 0.184 |
| margin | global | uncalibrated | 1.012 | 0.962 | 0.565 | 0.067 | 1.036 | 0.183 |
| margin | none | uncalibrated | 1.000 | 0.963 | 0.566 | 0.067 | 1.039 | 0.184 |
| margin | none | sigmoid | 1.000 | 0.963 | 0.567 | 0.061 | 1.041 | 0.184 |
| margin | global | sigmoid | 1.012 | 0.963 | 0.567 | 0.061 | 1.041 | 0.184 |
| poisson | similar_strength | uncalibrated | 1.237 | 0.966 | 0.568 | 0.075 | 1.038 | 0.185 |
| poisson | none | sigmoid | 1.000 | 0.966 | 0.569 | 0.078 | 1.041 | 0.185 |
| poisson | global | sigmoid | 1.137 | 0.966 | 0.569 | 0.078 | 1.041 | 0.185 |
| poisson | similar_strength | sigmoid | 1.237 | 0.967 | 0.570 | 0.077 | 1.044 | 0.186 |
| poisson | global | uncalibrated | 1.137 | 0.967 | 0.569 | 0.075 | 1.038 | 0.185 |

### Rating Model Comparison

| rating_model | elo_k_scale | avg_log_loss | avg_brier_score | avg_calibration_error | avg_scoreline_top_5_hit_rate | stability_score |
| --- | --- | --- | --- | --- | --- | --- |
| standard_elo | 1.200 | 0.961 | 0.565 | 0.060 | 0.540 | 0.081 |
| standard_elo | 1.000 | 0.962 | 0.565 | 0.060 | 0.555 | 0.076 |
| standard_elo | 0.800 | 0.963 | 0.567 | 0.063 | 0.543 | 0.069 |
| smoothed_dynamic | 1.200 | 0.980 | 0.582 | 0.068 | 0.552 | 0.058 |
| smoothed_dynamic | 1.000 | 0.983 | 0.583 | 0.064 | 0.544 | 0.057 |
| smoothed_dynamic | 0.800 | 0.986 | 0.585 | 0.070 | 0.539 | 0.055 |

Outcome-only models do not emit coherent exact-score distributions; their scoreline hit-rate fields are marked unavailable rather than borrowed from another model.

### Production Configuration By World Cup

| tournament_year | training_cutoff_year | log_loss | brier_score | accuracy | ranked_probability_score |
| --- | --- | --- | --- | --- | --- |
| 2010 | 2006 | 0.917 | 0.537 | 0.600 | 0.185 |
| 2014 | 2010 | 0.917 | 0.542 | 0.641 | 0.188 |
| 2018 | 2014 | 0.944 | 0.559 | 0.562 | 0.198 |
| 2022 | 2018 | 1.074 | 0.629 | 0.500 | 0.227 |

### Scoreline Model Comparison

| model_name | log_loss | brier_score | top_5_scoreline_hit_rate | mean_goal_mae | scoreline_log_loss |
| --- | --- | --- | --- | --- | --- |
| attack_defence_poisson | 0.963 | 0.567 | 0.540 | 0.894 | 2.892 |
| full_poisson | 0.964 | 0.567 | 0.545 | 0.897 | 2.900 |
| full_poisson_dixon_coles_-0.05 | 0.967 | 0.568 | 0.541 | 0.897 | 2.903 |
| full_poisson_dixon_coles_-0.10 | 0.970 | 0.570 | 0.533 | 0.897 | 2.908 |
| basic_poisson | 0.972 | 0.573 | 0.560 | 0.903 | 2.908 |
| ml_goal_full | 0.976 | 0.576 | 0.507 | 0.905 | 2.902 |

## F. Feature Ablation Summary

| component | candidate | reference | candidate_log_loss | reference_log_loss | log_loss_improvement | brier_improvement | helped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Elo features | core | core_without_elo | 0.974 | 1.078 | 0.105 | 0.082 | True |
| recent form | core_raw_form | core | 0.969 | 0.974 | 0.005 | 0.002 | True |
| opponent-adjusted form | core_opponent_adjusted | core | 0.976 | 0.974 | -0.002 | -0.001 | False |
| attack/defence features | all_features | all_without_attack_defence | 0.976 | 0.977 | 0.002 | 0.001 | True |
| rest days | core | core_without_rest | 0.974 | 0.973 | -0.000 | -0.000 | False |
| venue/host features | core | core_without_venue_host | 0.974 | 0.971 | -0.003 | -0.002 | False |
| match importance weighting | importance_only | unweighted | 0.963 | 0.965 | 0.001 | 0.001 | True |
| time decay weighting | time_decay_only | unweighted | 0.966 | 0.965 | -0.001 | -0.001 | False |
| combined weighting | time_decay_and_importance | unweighted | 0.965 | 0.965 | -0.000 | 0.000 | False |
| calibration | poisson_calibrated | poisson_goal | 0.968 | 0.965 | -0.003 | -0.002 | False |
| Poisson model | poisson_goal | elo_baseline | 0.965 | 0.969 | 0.004 | 0.004 | True |
| ML model | ml_challenger | elo_baseline | 0.985 | 0.969 | -0.017 | -0.007 | False |
| ensemble | ensemble | poisson_goal | 0.972 | 0.965 | -0.008 | -0.004 | False |

Positive log-loss improvement helped. Negative values hurt. Noisy or unavailable context features are not presented as proven improvements.

## G. Match Prediction Summary

### All Group-Stage Matches

| group | home_team | away_team | expected_goals_home | expected_goals_away | most_likely_score | p_home_win | p_draw | p_away_win | expected_total_goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Mexico | South Africa | 2.658 | 0.576 | 2-0 | 0.857 | 0.105 | 0.038 | 3.234 |
| A | South Korea | Czech Republic | 1.489 | 1.072 | 1-0 | 0.477 | 0.252 | 0.271 | 2.560 |
| A | Czech Republic | South Africa | 1.721 | 0.950 | 1-0 | 0.570 | 0.220 | 0.210 | 2.671 |
| A | Mexico | South Korea | 1.840 | 0.720 | 1-0 | 0.653 | 0.214 | 0.133 | 2.560 |
| A | South Africa | South Korea | 0.823 | 1.762 | 0-1 | 0.173 | 0.216 | 0.611 | 2.585 |
| A | Czech Republic | Mexico | 1.009 | 1.453 | 0-1 | 0.275 | 0.242 | 0.483 | 2.462 |
| B | Canada | Bosnia and Herzegovina | 2.133 | 0.576 | 2-0 | 0.764 | 0.159 | 0.077 | 2.709 |
| B | Qatar | Switzerland | 0.726 | 2.816 | 0-2 | 0.055 | 0.097 | 0.848 | 3.542 |
| B | Switzerland | Bosnia and Herzegovina | 2.249 | 0.762 | 2-0 | 0.736 | 0.160 | 0.104 | 3.011 |
| B | Canada | Qatar | 2.844 | 0.560 | 2-0 | 0.889 | 0.084 | 0.027 | 3.404 |
| B | Bosnia and Herzegovina | Qatar | 1.775 | 0.918 | 1-0 | 0.593 | 0.209 | 0.198 | 2.693 |
| B | Switzerland | Canada | 1.625 | 0.870 | 1-0 | 0.566 | 0.234 | 0.200 | 2.494 |
| C | Brazil | Morocco | 1.375 | 1.076 | 1-1 | 0.439 | 0.272 | 0.289 | 2.451 |
| C | Haiti | Scotland | 1.078 | 1.776 | 0-1 | 0.237 | 0.215 | 0.548 | 2.853 |
| C | Scotland | Morocco | 0.897 | 1.719 | 0-1 | 0.189 | 0.236 | 0.575 | 2.616 |
| C | Brazil | Haiti | 2.846 | 0.708 | 2-0 | 0.857 | 0.098 | 0.044 | 3.554 |
| C | Scotland | Brazil | 0.874 | 2.167 | 0-2 | 0.132 | 0.178 | 0.690 | 3.041 |
| C | Morocco | Haiti | 2.288 | 0.670 | 2-0 | 0.766 | 0.157 | 0.077 | 2.958 |
| D | United States | Paraguay | 1.379 | 1.187 | 1-0 | 0.435 | 0.228 | 0.337 | 2.565 |
| D | Australia | Turkey | 1.127 | 1.431 | 1-1 | 0.300 | 0.255 | 0.445 | 2.558 |
| D | United States | Australia | 1.451 | 1.145 | 1-0 | 0.463 | 0.225 | 0.313 | 2.595 |
| D | Turkey | Paraguay | 1.465 | 1.154 | 1-1 | 0.456 | 0.241 | 0.303 | 2.618 |
| D | Paraguay | Australia | 1.291 | 1.131 | 1-1 | 0.409 | 0.262 | 0.329 | 2.422 |
| D | Turkey | United States | 2.207 | 0.875 | 2-0 | 0.707 | 0.156 | 0.137 | 3.082 |
| E | Germany | Curaçao | 3.299 | 0.761 | 3-0 | 0.913 | 0.062 | 0.025 | 4.061 |
| E | Ivory Coast | Ecuador | 0.807 | 1.604 | 0-1 | 0.185 | 0.244 | 0.571 | 2.412 |
| E | Germany | Ivory Coast | 1.758 | 0.908 | 1-0 | 0.588 | 0.225 | 0.187 | 2.666 |
| E | Ecuador | Curaçao | 3.053 | 0.641 | 2-0 | 0.903 | 0.071 | 0.025 | 3.694 |
| E | Curaçao | Ivory Coast | 0.775 | 2.380 | 0-2 | 0.100 | 0.142 | 0.758 | 3.155 |
| E | Ecuador | Germany | 1.170 | 1.129 | 1-1 | 0.363 | 0.297 | 0.340 | 2.299 |
| F | Netherlands | Japan | 1.180 | 1.204 | 1-1 | 0.352 | 0.285 | 0.363 | 2.384 |
| F | Sweden | Tunisia | 1.605 | 1.240 | 1-0 | 0.478 | 0.211 | 0.311 | 2.845 |
| F | Netherlands | Sweden | 2.334 | 0.835 | 2-0 | 0.733 | 0.159 | 0.109 | 3.169 |
| F | Tunisia | Japan | 0.727 | 2.049 | 0-1 | 0.114 | 0.186 | 0.701 | 2.777 |
| F | Japan | Sweden | 2.276 | 0.790 | 2-0 | 0.732 | 0.165 | 0.103 | 3.066 |
| F | Tunisia | Netherlands | 0.769 | 2.132 | 0-2 | 0.117 | 0.174 | 0.709 | 2.901 |
| G | Belgium | Egypt | 1.688 | 0.946 | 1-0 | 0.559 | 0.235 | 0.206 | 2.633 |
| G | Iran | New Zealand | 1.944 | 0.823 | 1-0 | 0.657 | 0.192 | 0.152 | 2.767 |
| G | Belgium | Iran | 1.517 | 1.089 | 1-1 | 0.474 | 0.260 | 0.266 | 2.606 |
| G | New Zealand | Egypt | 0.995 | 1.557 | 0-1 | 0.253 | 0.230 | 0.517 | 2.552 |
| G | Egypt | Iran | 1.038 | 1.243 | 1-1 | 0.305 | 0.287 | 0.407 | 2.281 |
| G | New Zealand | Belgium | 0.871 | 2.340 | 0-2 | 0.114 | 0.162 | 0.724 | 3.211 |
| H | Spain | Cape Verde | 3.194 | 0.659 | 2-0 | 0.918 | 0.064 | 0.018 | 3.853 |
| H | Saudi Arabia | Uruguay | 0.680 | 1.897 | 0-1 | 0.125 | 0.189 | 0.687 | 2.577 |
| H | Spain | Saudi Arabia | 2.852 | 0.607 | 2-0 | 0.881 | 0.089 | 0.030 | 3.459 |
| H | Uruguay | Cape Verde | 2.191 | 0.600 | 2-0 | 0.768 | 0.153 | 0.079 | 2.791 |
| H | Uruguay | Spain | 0.746 | 1.752 | 0-1 | 0.145 | 0.230 | 0.626 | 2.497 |
| H | Cape Verde | Saudi Arabia | 1.090 | 1.389 | 0-1 | 0.312 | 0.234 | 0.453 | 2.479 |
| I | France | Senegal | 1.826 | 0.806 | 1-0 | 0.630 | 0.213 | 0.157 | 2.632 |
| I | Iraq | Norway | 0.834 | 1.939 | 0-1 | 0.158 | 0.190 | 0.652 | 2.773 |
| I | France | Iraq | 2.280 | 0.596 | 2-0 | 0.785 | 0.148 | 0.067 | 2.876 |
| I | Norway | Senegal | 1.523 | 1.180 | 1-1 | 0.460 | 0.242 | 0.298 | 2.703 |
| I | Senegal | Iraq | 1.556 | 0.819 | 1-0 | 0.559 | 0.243 | 0.198 | 2.375 |
| I | Norway | France | 0.992 | 1.802 | 0-1 | 0.200 | 0.226 | 0.574 | 2.794 |
| J | Argentina | Algeria | 1.788 | 0.644 | 1-0 | 0.663 | 0.223 | 0.114 | 2.432 |
| J | Austria | Jordan | 1.988 | 0.953 | 1-0 | 0.634 | 0.189 | 0.177 | 2.941 |
| J | Argentina | Austria | 1.807 | 0.665 | 1-0 | 0.660 | 0.223 | 0.117 | 2.472 |
| J | Jordan | Algeria | 0.936 | 1.852 | 0-1 | 0.191 | 0.206 | 0.603 | 2.788 |
| J | Algeria | Austria | 1.221 | 1.191 | 1-1 | 0.367 | 0.283 | 0.351 | 2.412 |
| J | Jordan | Argentina | 0.694 | 2.732 | 0-2 | 0.039 | 0.114 | 0.847 | 3.426 |
| K | Portugal | DR Congo | 2.142 | 0.674 | 2-0 | 0.740 | 0.168 | 0.093 | 2.816 |
| K | Uzbekistan | Colombia | 0.755 | 1.978 | 0-1 | 0.128 | 0.192 | 0.680 | 2.733 |
| K | Portugal | Uzbekistan | 1.967 | 0.768 | 1-0 | 0.676 | 0.192 | 0.132 | 2.736 |
| K | Colombia | DR Congo | 2.182 | 0.670 | 2-0 | 0.749 | 0.163 | 0.089 | 2.852 |
| K | Colombia | Portugal | 1.286 | 1.285 | 1-1 | 0.367 | 0.267 | 0.366 | 2.571 |
| K | DR Congo | Uzbekistan | 1.031 | 1.256 | 0-1 | 0.309 | 0.269 | 0.421 | 2.288 |
| L | England | Croatia | 1.486 | 0.970 | 1-0 | 0.499 | 0.255 | 0.246 | 2.456 |
| L | Ghana | Panama | 0.955 | 1.756 | 0-1 | 0.215 | 0.203 | 0.581 | 2.711 |
| L | England | Ghana | 2.727 | 0.603 | 2-0 | 0.864 | 0.097 | 0.039 | 3.330 |
| L | Panama | Croatia | 0.918 | 1.925 | 0-1 | 0.181 | 0.190 | 0.629 | 2.842 |
| L | Panama | England | 0.739 | 2.183 | 0-2 | 0.103 | 0.168 | 0.729 | 2.923 |
| L | Croatia | Ghana | 2.409 | 0.660 | 2-0 | 0.796 | 0.128 | 0.076 | 3.069 |

### Most One-Sided Matches

| group | home_team | away_team | expected_goals_home | expected_goals_away | most_likely_score | p_home_win | p_draw | p_away_win | expected_total_goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | Spain | Cape Verde | 3.194 | 0.659 | 2-0 | 0.918 | 0.064 | 0.018 | 3.853 |
| E | Germany | Curaçao | 3.299 | 0.761 | 3-0 | 0.913 | 0.062 | 0.025 | 4.061 |
| E | Ecuador | Curaçao | 3.053 | 0.641 | 2-0 | 0.903 | 0.071 | 0.025 | 3.694 |
| B | Canada | Qatar | 2.844 | 0.560 | 2-0 | 0.889 | 0.084 | 0.027 | 3.404 |
| H | Spain | Saudi Arabia | 2.852 | 0.607 | 2-0 | 0.881 | 0.089 | 0.030 | 3.459 |
| L | England | Ghana | 2.727 | 0.603 | 2-0 | 0.864 | 0.097 | 0.039 | 3.330 |
| A | Mexico | South Africa | 2.658 | 0.576 | 2-0 | 0.857 | 0.105 | 0.038 | 3.234 |
| C | Brazil | Haiti | 2.846 | 0.708 | 2-0 | 0.857 | 0.098 | 0.044 | 3.554 |
| B | Qatar | Switzerland | 0.726 | 2.816 | 0-2 | 0.055 | 0.097 | 0.848 | 3.542 |
| J | Jordan | Argentina | 0.694 | 2.732 | 0-2 | 0.039 | 0.114 | 0.847 | 3.426 |

### Most Even Matches

| group | home_team | away_team | expected_goals_home | expected_goals_away | most_likely_score | p_home_win | p_draw | p_away_win | expected_total_goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | Ecuador | Germany | 1.170 | 1.129 | 1-1 | 0.363 | 0.297 | 0.340 | 2.299 |
| F | Netherlands | Japan | 1.180 | 1.204 | 1-1 | 0.352 | 0.285 | 0.363 | 2.384 |
| J | Algeria | Austria | 1.221 | 1.191 | 1-1 | 0.367 | 0.283 | 0.351 | 2.412 |
| K | Colombia | Portugal | 1.286 | 1.285 | 1-1 | 0.367 | 0.267 | 0.366 | 2.571 |
| G | Egypt | Iran | 1.038 | 1.243 | 1-1 | 0.305 | 0.287 | 0.407 | 2.281 |
| D | Paraguay | Australia | 1.291 | 1.131 | 1-1 | 0.409 | 0.262 | 0.329 | 2.422 |
| K | DR Congo | Uzbekistan | 1.031 | 1.256 | 0-1 | 0.309 | 0.269 | 0.421 | 2.288 |
| C | Brazil | Morocco | 1.375 | 1.076 | 1-1 | 0.439 | 0.272 | 0.289 | 2.451 |
| D | Australia | Turkey | 1.127 | 1.431 | 1-1 | 0.300 | 0.255 | 0.445 | 2.558 |
| D | United States | Paraguay | 1.379 | 1.187 | 1-0 | 0.435 | 0.228 | 0.337 | 2.565 |

### Most Likely Away-Side Upsets

| group | home_team | away_team | expected_goals_home | expected_goals_away | most_likely_score | p_home_win | p_draw | p_away_win | expected_total_goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J | Jordan | Argentina | 0.694 | 2.732 | 0-2 | 0.039 | 0.114 | 0.847 | 3.426 |
| B | Qatar | Switzerland | 0.726 | 2.816 | 0-2 | 0.055 | 0.097 | 0.848 | 3.542 |
| E | Curaçao | Ivory Coast | 0.775 | 2.380 | 0-2 | 0.100 | 0.142 | 0.758 | 3.155 |
| L | Panama | England | 0.739 | 2.183 | 0-2 | 0.103 | 0.168 | 0.729 | 2.923 |
| G | New Zealand | Belgium | 0.871 | 2.340 | 0-2 | 0.114 | 0.162 | 0.724 | 3.211 |
| F | Tunisia | Netherlands | 0.769 | 2.132 | 0-2 | 0.117 | 0.174 | 0.709 | 2.901 |
| F | Tunisia | Japan | 0.727 | 2.049 | 0-1 | 0.114 | 0.186 | 0.701 | 2.777 |
| H | Saudi Arabia | Uruguay | 0.680 | 1.897 | 0-1 | 0.125 | 0.189 | 0.687 | 2.577 |
| C | Scotland | Brazil | 0.874 | 2.167 | 0-2 | 0.132 | 0.178 | 0.690 | 3.041 |
| K | Uzbekistan | Colombia | 0.755 | 1.978 | 0-1 | 0.128 | 0.192 | 0.680 | 2.733 |

### Highest Draw Probability

| group | home_team | away_team | expected_goals_home | expected_goals_away | most_likely_score | p_home_win | p_draw | p_away_win | expected_total_goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | Ecuador | Germany | 1.170 | 1.129 | 1-1 | 0.363 | 0.297 | 0.340 | 2.299 |
| G | Egypt | Iran | 1.038 | 1.243 | 1-1 | 0.305 | 0.287 | 0.407 | 2.281 |
| F | Netherlands | Japan | 1.180 | 1.204 | 1-1 | 0.352 | 0.285 | 0.363 | 2.384 |
| J | Algeria | Austria | 1.221 | 1.191 | 1-1 | 0.367 | 0.283 | 0.351 | 2.412 |
| C | Brazil | Morocco | 1.375 | 1.076 | 1-1 | 0.439 | 0.272 | 0.289 | 2.451 |
| K | DR Congo | Uzbekistan | 1.031 | 1.256 | 0-1 | 0.309 | 0.269 | 0.421 | 2.288 |
| K | Colombia | Portugal | 1.286 | 1.285 | 1-1 | 0.367 | 0.267 | 0.366 | 2.571 |
| D | Paraguay | Australia | 1.291 | 1.131 | 1-1 | 0.409 | 0.262 | 0.329 | 2.422 |
| G | Belgium | Iran | 1.517 | 1.089 | 1-1 | 0.474 | 0.260 | 0.266 | 2.606 |
| D | Australia | Turkey | 1.127 | 1.431 | 1-1 | 0.300 | 0.255 | 0.445 | 2.558 |

### Highest Expected Goals

| group | home_team | away_team | expected_goals_home | expected_goals_away | most_likely_score | p_home_win | p_draw | p_away_win | expected_total_goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | Germany | Curaçao | 3.299 | 0.761 | 3-0 | 0.913 | 0.062 | 0.025 | 4.061 |
| H | Spain | Cape Verde | 3.194 | 0.659 | 2-0 | 0.918 | 0.064 | 0.018 | 3.853 |
| E | Ecuador | Curaçao | 3.053 | 0.641 | 2-0 | 0.903 | 0.071 | 0.025 | 3.694 |
| C | Brazil | Haiti | 2.846 | 0.708 | 2-0 | 0.857 | 0.098 | 0.044 | 3.554 |
| B | Qatar | Switzerland | 0.726 | 2.816 | 0-2 | 0.055 | 0.097 | 0.848 | 3.542 |
| H | Spain | Saudi Arabia | 2.852 | 0.607 | 2-0 | 0.881 | 0.089 | 0.030 | 3.459 |
| J | Jordan | Argentina | 0.694 | 2.732 | 0-2 | 0.039 | 0.114 | 0.847 | 3.426 |
| B | Canada | Qatar | 2.844 | 0.560 | 2-0 | 0.889 | 0.084 | 0.027 | 3.404 |
| L | England | Ghana | 2.727 | 0.603 | 2-0 | 0.864 | 0.097 | 0.039 | 3.330 |
| A | Mexico | South Africa | 2.658 | 0.576 | 2-0 | 0.857 | 0.105 | 0.038 | 3.234 |

### Lowest Expected Goals

| group | home_team | away_team | expected_goals_home | expected_goals_away | most_likely_score | p_home_win | p_draw | p_away_win | expected_total_goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G | Egypt | Iran | 1.038 | 1.243 | 1-1 | 0.305 | 0.287 | 0.407 | 2.281 |
| K | DR Congo | Uzbekistan | 1.031 | 1.256 | 0-1 | 0.309 | 0.269 | 0.421 | 2.288 |
| E | Ecuador | Germany | 1.170 | 1.129 | 1-1 | 0.363 | 0.297 | 0.340 | 2.299 |
| I | Senegal | Iraq | 1.556 | 0.819 | 1-0 | 0.559 | 0.243 | 0.198 | 2.375 |
| F | Netherlands | Japan | 1.180 | 1.204 | 1-1 | 0.352 | 0.285 | 0.363 | 2.384 |
| E | Ivory Coast | Ecuador | 0.807 | 1.604 | 0-1 | 0.185 | 0.244 | 0.571 | 2.412 |
| J | Algeria | Austria | 1.221 | 1.191 | 1-1 | 0.367 | 0.283 | 0.351 | 2.412 |
| D | Paraguay | Australia | 1.291 | 1.131 | 1-1 | 0.409 | 0.262 | 0.329 | 2.422 |
| J | Argentina | Algeria | 1.788 | 0.644 | 1-0 | 0.663 | 0.223 | 0.114 | 2.432 |
| C | Brazil | Morocco | 1.375 | 1.076 | 1-1 | 0.439 | 0.272 | 0.289 | 2.451 |

## H. Tournament Simulation Summary

### Top 20 Champion Probabilities

| team | group | p_champion | p_reach_final |
| --- | --- | --- | --- |
| Argentina | J | 0.208 | 0.308 |
| Spain | H | 0.197 | 0.308 |
| France | I | 0.093 | 0.173 |
| England | L | 0.061 | 0.122 |
| Brazil | C | 0.057 | 0.115 |
| Colombia | K | 0.054 | 0.113 |
| Portugal | K | 0.050 | 0.105 |
| Ecuador | E | 0.036 | 0.083 |
| Japan | F | 0.031 | 0.070 |
| Germany | E | 0.031 | 0.072 |
| Netherlands | F | 0.026 | 0.062 |
| Mexico | A | 0.024 | 0.061 |
| Belgium | G | 0.020 | 0.055 |
| Morocco | C | 0.020 | 0.052 |
| Turkey | D | 0.014 | 0.041 |
| Switzerland | B | 0.012 | 0.034 |
| Uruguay | H | 0.012 | 0.036 |
| Croatia | L | 0.011 | 0.031 |
| Norway | I | 0.009 | 0.028 |
| Canada | B | 0.006 | 0.020 |

### Top 20 Final Probabilities

| team | group | p_reach_final | p_champion |
| --- | --- | --- | --- |
| Argentina | J | 0.308 | 0.208 |
| Spain | H | 0.308 | 0.197 |
| France | I | 0.173 | 0.093 |
| England | L | 0.122 | 0.061 |
| Brazil | C | 0.115 | 0.057 |
| Colombia | K | 0.113 | 0.054 |
| Portugal | K | 0.105 | 0.050 |
| Ecuador | E | 0.083 | 0.036 |
| Germany | E | 0.072 | 0.031 |
| Japan | F | 0.070 | 0.031 |
| Netherlands | F | 0.062 | 0.026 |
| Mexico | A | 0.061 | 0.024 |
| Belgium | G | 0.055 | 0.020 |
| Morocco | C | 0.052 | 0.020 |
| Turkey | D | 0.041 | 0.014 |
| Uruguay | H | 0.036 | 0.012 |
| Switzerland | B | 0.034 | 0.012 |
| Croatia | L | 0.031 | 0.011 |
| Norway | I | 0.028 | 0.009 |
| Canada | B | 0.020 | 0.006 |

### Most Likely Group Winners

| team | group | p_group_1st |
| --- | --- | --- |
| Spain | H | 0.795 |
| Argentina | J | 0.739 |
| Mexico | A | 0.638 |
| France | I | 0.632 |
| Switzerland | B | 0.619 |
| England | L | 0.618 |
| Brazil | C | 0.573 |
| Belgium | G | 0.524 |
| Turkey | D | 0.469 |
| Japan | F | 0.463 |
| Colombia | K | 0.461 |
| Portugal | K | 0.460 |

### Most Uncertain Groups

| group | group_winner_entropy |
| --- | --- |
| D | 1.272 |
| G | 1.124 |
| E | 0.981 |
| A | 0.971 |
| I | 0.971 |
| F | 0.969 |
| K | 0.965 |
| C | 0.932 |
| L | 0.869 |
| J | 0.808 |
| B | 0.801 |
| H | 0.581 |

Overperformance/disappointment relative to external seeding and most-common final matchups are not reported: no reliable seed/ranking input or final-matchup counter is currently modeled.

## I. Limitations

- Football outcomes and penalties are intrinsically noisy.
- No squads, player availability, injuries, betting odds, or tactical inputs are used.
- Final squads and pre-tournament form may materially change team strength.
- Historical international data has uneven opponent quality and incomplete stage labels.
- Penalties use a deliberately simple strength-weighted approximation.
- Confederation effects and travel burden are not modeled reliably.
- The fixture bracket is assumed to match the populated official schedule.

## J. Updating Predictions Later

1. Add new completed matches to `data/raw/results.csv`, then rerun the pipeline with a later cutoff.
2. Squad or injury information requires a new validated input and backtested feature before use.
3. After group-stage results, update the fixture/simulation state; live tournament updating is not yet implemented.
4. Re-run `scripts/train_models.py` and `scripts/predict_worldcup_2026.py` with the desired cutoff and a simulation count no higher than the configured maximum.

These are probabilistic estimates, not certainties.