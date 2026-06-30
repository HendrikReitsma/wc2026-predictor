# Evaluation Summary

## Selected Configuration

- Selected model family: **poisson_goal**.
- Best tested decayed half-life: **12.0 years**.
- Production time-decay half-life: **disabled**.
- Selected model feature set: **attack_defence_poisson**.
- Selected goal model: **poisson / attack_defence_poisson**.
- Dixon-Coles low-score rho: **0.0**.
- Selected minimum training year: **1990**.
- Selected calibration method: **uncalibrated**.
- Best classifier challenger feature group: **core_raw_form**.
- Selected weighting scheme: **importance_only**.
- Selection criterion: lowest mean log loss across the 2010, 2014, 2018, and 2022 World Cups.
- Accuracy is secondary.

## Model And Half-Life Comparison

| model_name | half_life_years | log_loss | brier_score | accuracy |
| --- | --- | --- | --- | --- |
| poisson_goal | 12.00000 | 0.96457 | 0.56795 | 0.58359 |
| poisson_goal | 8.00000 | 0.96539 | 0.56851 | 0.57578 |
| poisson_calibrated | 12.00000 | 0.96764 | 0.57038 | 0.57911 |
| poisson_goal | 4.00000 | 0.96783 | 0.57015 | 0.57969 |
| poisson_calibrated | 8.00000 | 0.96839 | 0.57090 | 0.58302 |
| elo_recent_form_weighted | 12.00000 | 0.96851 | 0.57002 | 0.56073 |
| elo_recent_form | 12.00000 | 0.96859 | 0.57075 | 0.56073 |
| elo_recent_form | 4.00000 | 0.96859 | 0.57075 | 0.56073 |
| elo_recent_form | 8.00000 | 0.96859 | 0.57075 | 0.56073 |
| elo_baseline | 12.00000 | 0.96864 | 0.57200 | 0.57578 |
| elo_baseline | 4.00000 | 0.96864 | 0.57200 | 0.57578 |
| elo_baseline | 8.00000 | 0.96864 | 0.57200 | 0.57578 |
| elo_recent_form_weighted | 8.00000 | 0.96874 | 0.57015 | 0.56464 |
| elo_recent_form_weighted | 4.00000 | 0.96963 | 0.57037 | 0.56911 |
| poisson_calibrated | 4.00000 | 0.97080 | 0.57261 | 0.58245 |
| ensemble | 8.00000 | 0.97144 | 0.57114 | 0.55349 |
| ensemble | 12.00000 | 0.97242 | 0.57146 | 0.54958 |
| ensemble | 4.00000 | 0.97354 | 0.57311 | 0.54292 |
| ml_challenger | 8.00000 | 0.98087 | 0.57713 | 0.54958 |
| ml_challenger | 4.00000 | 0.98463 | 0.58178 | 0.55349 |
| ml_challenger | 12.00000 | 0.98534 | 0.57907 | 0.54625 |
| ml_calibrated | 8.00000 | 0.98921 | 0.57894 | 0.55625 |
| ml_calibrated | 4.00000 | 0.99119 | 0.58244 | 0.55234 |
| ml_calibrated | 12.00000 | 0.99136 | 0.57980 | 0.56016 |

## Goal And Scoreline Model Comparison

| model_name | log_loss | brier_score | mean_goal_mae | scoreline_log_loss | top_1_scoreline_accuracy | top_5_scoreline_hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| attack_defence_poisson | 0.96312 | 0.56688 | 0.89378 | 2.89198 | 0.13984 | 0.54005 |
| full_poisson | 0.96426 | 0.56723 | 0.89663 | 2.90013 | 0.14318 | 0.54453 |
| full_poisson_dixon_coles_-0.05 | 0.96657 | 0.56846 | 0.89663 | 2.90323 | 0.12927 | 0.54063 |
| full_poisson_dixon_coles_-0.10 | 0.96958 | 0.57009 | 0.89663 | 2.90788 | 0.11594 | 0.53281 |
| basic_poisson | 0.97205 | 0.57259 | 0.90345 | 2.90826 | 0.14214 | 0.56016 |
| ml_goal_full | 0.97629 | 0.57579 | 0.90526 | 2.90231 | 0.13260 | 0.50661 |

## Minimum Training Year Comparison

| minimum_training_year | world_cups_covered | log_loss | brier_score | mean_goal_mae | top_5_scoreline_hit_rate |
| --- | --- | --- | --- | --- | --- |
| 1990.00000 | 4.00000 | 0.96312 | 0.56688 | 0.89378 | 0.54005 |
| 1998.00000 | 4.00000 | 0.96502 | 0.56837 | 0.89353 | 0.54120 |
| 2002.00000 | 4.00000 | 0.96782 | 0.57048 | 0.89215 | 0.53339 |
| 2010.00000 | 3.00000 | 0.98804 | 0.58385 | 0.90684 | 0.51042 |

The 2010 minimum-year candidate cannot cover the 2010 backtest because its frozen training cutoff is 2006; selection favors complete four-tournament coverage.

## Calibration Method Comparison

| calibration_method | log_loss | brier_score | accuracy | ranked_probability_score |
| --- | --- | --- | --- | --- |
| uncalibrated | 0.96264 | 0.56662 | 0.58693 | 0.19964 |
| platt_sigmoid | 0.96636 | 0.56956 | 0.57188 | 0.20001 |
| isotonic | 1.02742 | 0.57051 | 0.58245 | 0.20082 |

## Requested Ablation Summary

| component | candidate | reference | candidate_log_loss | reference_log_loss | log_loss_improvement | brier_improvement | helped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Elo features | core | core_without_elo | 0.97365 | 1.07817 | 0.10452 | 0.08247 | True |
| recent form | core_raw_form | core | 0.96851 | 0.97365 | 0.00514 | 0.00242 | True |
| opponent-adjusted form | core_opponent_adjusted | core | 0.97553 | 0.97365 | -0.00188 | -0.00129 | False |
| attack/defence features | all_features | all_without_attack_defence | 0.97557 | 0.97726 | 0.00169 | 0.00096 | True |
| rest days | core | core_without_rest | 0.97365 | 0.97350 | -0.00015 | -0.00010 | False |
| venue/host features | core | core_without_venue_host | 0.97365 | 0.97096 | -0.00270 | -0.00157 | False |
| match importance weighting | importance_only | unweighted | 0.96312 | 0.96450 | 0.00139 | 0.00137 | True |
| time decay weighting | time_decay_only | unweighted | 0.96596 | 0.96450 | -0.00145 | -0.00102 | False |
| combined weighting | time_decay_and_importance | unweighted | 0.96457 | 0.96450 | -0.00007 | 0.00030 | False |
| calibration | poisson_calibrated | poisson_goal | 0.96764 | 0.96457 | -0.00307 | -0.00243 | False |
| Poisson model | poisson_goal | elo_baseline | 0.96457 | 0.96864 | 0.00407 | 0.00405 | True |
| ML model | ml_challenger | elo_baseline | 0.98534 | 0.96864 | -0.01670 | -0.00707 | False |
| ensemble | ensemble | poisson_goal | 0.97242 | 0.96457 | -0.00785 | -0.00351 | False |

Positive improvement means the candidate helped against the named reference on frozen World Cup backtests.

## Detailed Feature Ablation

| model_name | log_loss | brier_score | accuracy | log_loss_improvement_vs_core |
| --- | --- | --- | --- | --- |
| core_raw_form | 0.96851 | 0.57002 | 0.56073 | 0.00514 |
| core_without_venue_host | 0.97096 | 0.57087 | 0.56911 | 0.00270 |
| core_without_rest | 0.97350 | 0.57234 | 0.57245 | 0.00015 |
| core | 0.97365 | 0.57244 | 0.57188 | 0.00000 |
| core_opponent_adjusted | 0.97553 | 0.57373 | 0.56521 | -0.00188 |
| all_features | 0.97557 | 0.57324 | 0.55682 | -0.00192 |
| all_without_attack_defence | 0.97726 | 0.57420 | 0.57245 | -0.00361 |
| core_without_elo | 1.07817 | 0.65492 | 0.42031 | -0.10452 |

Positive `log_loss_improvement_vs_core` means the feature group helped relative to Elo/context/rest alone.

## Weighting Ablation

| model_name | log_loss | brier_score | accuracy | log_loss_improvement_vs_unweighted |
| --- | --- | --- | --- | --- |
| importance_only | 0.96312 | 0.56688 | 0.57578 | 0.00139 |
| unweighted | 0.96450 | 0.56825 | 0.57635 | 0.00000 |
| time_decay_and_importance | 0.96457 | 0.56795 | 0.58359 | -0.00007 |
| time_decay_only | 0.96596 | 0.56928 | 0.57635 | -0.00145 |

Positive `log_loss_improvement_vs_unweighted` means the weighting choice helped.

## Selected-Model Calibration Table

| lower | upper | count | observed_rate | predicted_rate |
| --- | --- | --- | --- | --- |
| 0.00000 | 0.20000 | 8.25000 | 0.15758 | 0.14558 |
| 0.20000 | 0.40000 | 21.75000 | 0.21942 | 0.31095 |
| 0.40000 | 0.60000 | 24.25000 | 0.55139 | 0.49994 |
| 0.60000 | 0.80000 | 11.00000 | 0.67708 | 0.68973 |
| 0.80000 | 1.00000 | 2.00000 | 0.88889 | 0.84131 |

## World Cup Backtests

| model_name | tournament_year | training_cutoff_year | half_life_years | log_loss | brier_score | accuracy | average_probability_actual_outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| elo_baseline | 2010 | 2006 | 4.00000 | 0.94215 | 0.55394 | 0.60000 | 0.42501 |
| ml_challenger | 2010 | 2006 | 4.00000 | 0.95479 | 0.56370 | 0.57333 | 0.44146 |
| ml_calibrated | 2010 | 2006 | 4.00000 | 0.94821 | 0.55928 | 0.60000 | 0.41626 |
| elo_recent_form | 2010 | 2006 | 4.00000 | 0.94469 | 0.55479 | 0.58667 | 0.43911 |
| elo_recent_form_weighted | 2010 | 2006 | 4.00000 | 0.94610 | 0.55365 | 0.57333 | 0.44677 |
| poisson_goal | 2010 | 2006 | 4.00000 | 0.92741 | 0.54410 | 0.60000 | 0.44191 |
| poisson_calibrated | 2010 | 2006 | 4.00000 | 0.91905 | 0.53916 | 0.62667 | 0.44070 |
| ensemble | 2010 | 2006 | 4.00000 | 0.93287 | 0.54825 | 0.54667 | 0.42908 |
| elo_baseline | 2014 | 2010 | 4.00000 | 0.93812 | 0.55481 | 0.59375 | 0.42118 |
| ml_challenger | 2014 | 2010 | 4.00000 | 0.98403 | 0.58046 | 0.57812 | 0.42497 |
| ml_calibrated | 2014 | 2010 | 4.00000 | 0.97838 | 0.57956 | 0.56250 | 0.43257 |
| elo_recent_form | 2014 | 2010 | 4.00000 | 0.91948 | 0.54671 | 0.60938 | 0.44154 |
| elo_recent_form_weighted | 2014 | 2010 | 4.00000 | 0.92245 | 0.54841 | 0.59375 | 0.44523 |
| poisson_goal | 2014 | 2010 | 4.00000 | 0.92494 | 0.54710 | 0.64062 | 0.43258 |
| poisson_calibrated | 2014 | 2010 | 4.00000 | 0.92679 | 0.54960 | 0.65625 | 0.43359 |
| ensemble | 2014 | 2010 | 4.00000 | 0.94506 | 0.55980 | 0.57812 | 0.43258 |
| elo_baseline | 2018 | 2014 | 4.00000 | 0.96351 | 0.57151 | 0.54688 | 0.41425 |
| ml_challenger | 2018 | 2014 | 4.00000 | 0.94217 | 0.56092 | 0.57812 | 0.44586 |
| ml_calibrated | 2018 | 2014 | 4.00000 | 0.94773 | 0.56250 | 0.56250 | 0.43455 |
| elo_recent_form | 2018 | 2014 | 4.00000 | 0.95087 | 0.56581 | 0.54688 | 0.43364 |
| elo_recent_form_weighted | 2018 | 2014 | 4.00000 | 0.94716 | 0.56366 | 0.54688 | 0.44128 |
| poisson_goal | 2018 | 2014 | 4.00000 | 0.94205 | 0.55769 | 0.56250 | 0.42580 |
| poisson_calibrated | 2018 | 2014 | 4.00000 | 0.96418 | 0.57093 | 0.54688 | 0.42212 |
| ensemble | 2018 | 2014 | 4.00000 | 0.94109 | 0.55774 | 0.56250 | 0.43018 |
| elo_baseline | 2022 | 2018 | 4.00000 | 1.03077 | 0.60773 | 0.56250 | 0.40091 |
| ml_challenger | 2022 | 2018 | 4.00000 | 1.05753 | 0.62205 | 0.48438 | 0.41971 |
| ml_calibrated | 2022 | 2018 | 4.00000 | 1.09045 | 0.62841 | 0.48438 | 0.41255 |
| elo_recent_form | 2022 | 2018 | 4.00000 | 1.05931 | 0.61571 | 0.50000 | 0.41833 |
| elo_recent_form_weighted | 2022 | 2018 | 4.00000 | 1.06282 | 0.61574 | 0.56250 | 0.42367 |
| poisson_goal | 2022 | 2018 | 4.00000 | 1.07693 | 0.63170 | 0.51562 | 0.38948 |
| poisson_calibrated | 2022 | 2018 | 4.00000 | 1.07317 | 0.63075 | 0.50000 | 0.38491 |
| ensemble | 2022 | 2018 | 4.00000 | 1.07513 | 0.62663 | 0.48438 | 0.40102 |
| elo_baseline | 2010 | 2006 | 8.00000 | 0.94215 | 0.55394 | 0.60000 | 0.42501 |
| ml_challenger | 2010 | 2006 | 8.00000 | 0.94082 | 0.55179 | 0.57333 | 0.44544 |
| ml_calibrated | 2010 | 2006 | 8.00000 | 0.94465 | 0.55639 | 0.60000 | 0.41779 |
| elo_recent_form | 2010 | 2006 | 8.00000 | 0.94469 | 0.55479 | 0.58667 | 0.43911 |
| elo_recent_form_weighted | 2010 | 2006 | 8.00000 | 0.94525 | 0.55327 | 0.58667 | 0.44513 |
| poisson_goal | 2010 | 2006 | 8.00000 | 0.92230 | 0.54060 | 0.60000 | 0.44360 |
| poisson_calibrated | 2010 | 2006 | 8.00000 | 0.91844 | 0.53855 | 0.61333 | 0.44168 |
| ensemble | 2010 | 2006 | 8.00000 | 0.92943 | 0.54563 | 0.57333 | 0.43070 |
| elo_baseline | 2014 | 2010 | 8.00000 | 0.93812 | 0.55481 | 0.59375 | 0.42118 |
| ml_challenger | 2014 | 2010 | 8.00000 | 0.97022 | 0.57235 | 0.57812 | 0.42830 |
| ml_calibrated | 2014 | 2010 | 8.00000 | 0.96161 | 0.57020 | 0.57812 | 0.43315 |
| elo_recent_form | 2014 | 2010 | 8.00000 | 0.91948 | 0.54671 | 0.60938 | 0.44154 |
| elo_recent_form_weighted | 2014 | 2010 | 8.00000 | 0.92020 | 0.54744 | 0.59375 | 0.44525 |
| poisson_goal | 2014 | 2010 | 8.00000 | 0.92138 | 0.54472 | 0.64062 | 0.43506 |
| poisson_calibrated | 2014 | 2010 | 8.00000 | 0.92279 | 0.54712 | 0.67188 | 0.43324 |
| ensemble | 2014 | 2010 | 8.00000 | 0.93672 | 0.55495 | 0.57812 | 0.43411 |
| elo_baseline | 2018 | 2014 | 8.00000 | 0.96351 | 0.57151 | 0.54688 | 0.41425 |
| ml_challenger | 2018 | 2014 | 8.00000 | 0.94940 | 0.56205 | 0.54688 | 0.44163 |
| ml_calibrated | 2018 | 2014 | 8.00000 | 0.94767 | 0.56119 | 0.54688 | 0.43339 |
| elo_recent_form | 2018 | 2014 | 8.00000 | 0.95087 | 0.56581 | 0.54688 | 0.43364 |
| elo_recent_form_weighted | 2018 | 2014 | 8.00000 | 0.94664 | 0.56365 | 0.54688 | 0.44029 |
| poisson_goal | 2018 | 2014 | 8.00000 | 0.94286 | 0.55833 | 0.56250 | 0.42667 |
| poisson_calibrated | 2018 | 2014 | 8.00000 | 0.96015 | 0.56781 | 0.54688 | 0.42266 |
| ensemble | 2018 | 2014 | 8.00000 | 0.94240 | 0.55783 | 0.56250 | 0.43003 |
| elo_baseline | 2022 | 2018 | 8.00000 | 1.03077 | 0.60773 | 0.56250 | 0.40091 |
| ml_challenger | 2022 | 2018 | 8.00000 | 1.06305 | 0.62234 | 0.50000 | 0.42030 |
| ml_calibrated | 2022 | 2018 | 8.00000 | 1.10291 | 0.62796 | 0.50000 | 0.41371 |
| elo_recent_form | 2022 | 2018 | 8.00000 | 1.05931 | 0.61571 | 0.50000 | 0.41833 |
| elo_recent_form_weighted | 2022 | 2018 | 8.00000 | 1.06288 | 0.61625 | 0.53125 | 0.42311 |
| poisson_goal | 2022 | 2018 | 8.00000 | 1.07500 | 0.63039 | 0.50000 | 0.39037 |
| poisson_calibrated | 2022 | 2018 | 8.00000 | 1.07216 | 0.63015 | 0.50000 | 0.38474 |
| ensemble | 2022 | 2018 | 8.00000 | 1.07720 | 0.62614 | 0.50000 | 0.40204 |
| elo_baseline | 2010 | 2006 | 12.00000 | 0.94215 | 0.55394 | 0.60000 | 0.42501 |
| ml_challenger | 2010 | 2006 | 12.00000 | 0.94938 | 0.55777 | 0.56000 | 0.44141 |
| ml_calibrated | 2010 | 2006 | 12.00000 | 0.94928 | 0.55913 | 0.60000 | 0.41693 |
| elo_recent_form | 2010 | 2006 | 12.00000 | 0.94469 | 0.55479 | 0.58667 | 0.43911 |
| elo_recent_form_weighted | 2010 | 2006 | 12.00000 | 0.94479 | 0.55300 | 0.58667 | 0.44474 |
| poisson_goal | 2010 | 2006 | 12.00000 | 0.92056 | 0.53943 | 0.60000 | 0.44420 |
| poisson_calibrated | 2010 | 2006 | 12.00000 | 0.91819 | 0.53834 | 0.61333 | 0.44200 |
| ensemble | 2010 | 2006 | 12.00000 | 0.93119 | 0.54643 | 0.57333 | 0.43057 |
| elo_baseline | 2014 | 2010 | 12.00000 | 0.93812 | 0.55481 | 0.59375 | 0.42118 |
| ml_challenger | 2014 | 2010 | 12.00000 | 0.97423 | 0.57247 | 0.57812 | 0.42782 |
| ml_calibrated | 2014 | 2010 | 12.00000 | 0.96403 | 0.56975 | 0.57812 | 0.43354 |
| elo_recent_form | 2014 | 2010 | 12.00000 | 0.91948 | 0.54671 | 0.60938 | 0.44154 |
| elo_recent_form_weighted | 2014 | 2010 | 12.00000 | 0.91938 | 0.54711 | 0.60938 | 0.44516 |
| poisson_goal | 2014 | 2010 | 12.00000 | 0.92000 | 0.54383 | 0.65625 | 0.43596 |
| poisson_calibrated | 2014 | 2010 | 12.00000 | 0.92158 | 0.54639 | 0.65625 | 0.43308 |
| ensemble | 2014 | 2010 | 12.00000 | 0.93643 | 0.55395 | 0.57812 | 0.43475 |
| elo_baseline | 2018 | 2014 | 12.00000 | 0.96351 | 0.57151 | 0.54688 | 0.41425 |
| ml_challenger | 2018 | 2014 | 12.00000 | 0.95411 | 0.56637 | 0.54688 | 0.43548 |
| ml_calibrated | 2018 | 2014 | 12.00000 | 0.95198 | 0.56494 | 0.54688 | 0.42628 |
| elo_recent_form | 2018 | 2014 | 12.00000 | 0.95087 | 0.56581 | 0.54688 | 0.43364 |
| elo_recent_form_weighted | 2018 | 2014 | 12.00000 | 0.94673 | 0.56378 | 0.54688 | 0.43956 |
| poisson_goal | 2018 | 2014 | 12.00000 | 0.94317 | 0.55857 | 0.57812 | 0.42706 |
| poisson_calibrated | 2018 | 2014 | 12.00000 | 0.95883 | 0.56679 | 0.54688 | 0.42279 |
| ensemble | 2018 | 2014 | 12.00000 | 0.94513 | 0.56011 | 0.54688 | 0.42667 |
| elo_baseline | 2022 | 2018 | 12.00000 | 1.03077 | 0.60773 | 0.56250 | 0.40091 |
| ml_challenger | 2022 | 2018 | 12.00000 | 1.06364 | 0.61966 | 0.50000 | 0.41996 |
| ml_calibrated | 2022 | 2018 | 12.00000 | 1.10016 | 0.62538 | 0.51562 | 0.41262 |
| elo_recent_form | 2022 | 2018 | 12.00000 | 1.05931 | 0.61571 | 0.50000 | 0.41833 |
| elo_recent_form_weighted | 2022 | 2018 | 12.00000 | 1.06313 | 0.61620 | 0.50000 | 0.42268 |
| poisson_goal | 2022 | 2018 | 12.00000 | 1.07457 | 0.62997 | 0.50000 | 0.39081 |
| poisson_calibrated | 2022 | 2018 | 12.00000 | 1.07196 | 0.62999 | 0.50000 | 0.38463 |
| ensemble | 2022 | 2018 | 12.00000 | 1.07694 | 0.62534 | 0.50000 | 0.40171 |
| poisson_goal_production | 2010 | 2006 | 1000000000.00000 | 0.91715 | 0.53720 | 0.60000 | 0.44542 |
| poisson_goal_production | 2014 | 2010 | 1000000000.00000 | 0.91727 | 0.54214 | 0.64062 | 0.43765 |
| poisson_goal_production | 2018 | 2014 | 1000000000.00000 | 0.94370 | 0.55893 | 0.56250 | 0.42791 |
| poisson_goal_production | 2022 | 2018 | 1000000000.00000 | 1.07435 | 0.62924 | 0.50000 | 0.39177 |

## Leakage And Data-Quality Checks

- Every feature is computed before the match and same-day matches update state only after all same-day features are captured.
- All validation and calibration splits are chronological; there are no random train/test splits.
- Frozen World Cup training cutoffs are 2006, 2010, 2014, and 2018 for the 2010, 2014, 2018, and 2022 tests.
- Upsets and surprising results are retained.
- Only clear duplicate/unscored rows are excluded; goal-model targets are capped rather than matches deleted.
- Match importance affects both Elo updates and model sample weights.
- The World Cup qualification classifier was corrected so qualifiers are not labeled as World Cup finals.
- Historical results do not contain reliable tournament-stage labels; group/knockout-specific weights fall back to tournament-level weights for those rows.
- Detailed feature-group ablations use the classifier challenger; the selected Poisson model is assessed as a complete family with its fixed goal-feature set.
- No FIFA ranking, squad, player availability, injury, or betting-odds inputs are used.

<!-- target-experiments:start -->
## Alternative Target Experiments

- Selected target/model: **margin_class_classifier__uncalibrated**.
- Mean frozen-World-Cup log loss: **0.9617**.
- Mean Brier score: **0.5654**.
- Mean top-5 scoreline hit rate: **55.5%**.
- Future-form targets were generated separately and skipped as model inputs because of leakage risk.
- See `reports/target_experiment_report.md` for the full comparison.
<!-- target-experiments:end -->
