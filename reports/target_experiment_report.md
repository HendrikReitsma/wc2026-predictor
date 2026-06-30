# Target Experiment Report

## Executive Summary

- Recommended final target/model: **margin_class_classifier__uncalibrated**.
- Best W/D/L log loss: **margin_class_classifier__uncalibrated** (0.9617).
- Best scoreline top-5 hit rate: **margin_class_classifier__uncalibrated** (55.5%).
- Most stable log loss: **elo_only__uncalibrated** (standard deviation 0.0429).
- Best explicitly calibrated variant: **outcome_classifier__platt_sigmoid** (0.9623).
- Margin-target log-loss improvement over the current Poisson baseline: **0.0015**. This is small.
- Residual correction log loss: **0.9788**; it did not improve the baseline.

## Target Construction

- Outcome: direct home-win/draw/away-win classification.
- Goals: raw and six-goal-capped home/away count targets.
- Goal difference: actual difference clipped to -4 through +4.
- Margin class: seven ordered classes from away-win-by-3+ through home-win-by-3+.
- Residual: actual goal difference minus a pre-match Elo expected goal difference. A points residual is also engineered, but not fitted because it does not map cleanly back to scorelines.
- Future-form labels were generated as an isolated research table but skipped as model inputs because using them in the same-match pipeline creates a high leakage risk.

## Conversion Back To Match Probabilities

- Goal targets directly create Poisson score matrices.
- Goal-difference and residual models preserve baseline expected total goals and replace the expected goal difference before creating a Poisson score matrix.
- Margin probabilities aggregate directly to W/D/L; approximate scorelines preserve baseline expected total goals and use the class-weighted expected margin.
- The direct outcome classifier has no native score distribution, so reported scoreline metrics retain the baseline Poisson score matrix.

## Summary

| target_type | model_name | avg_log_loss | avg_brier_score | avg_calibration_error | avg_scoreline_top_5_hit_rate | stability_score | selected_for_final_model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| margin_class | margin_class_classifier__uncalibrated | 0.9617 | 0.5654 | 0.0587 | 0.5551 | 0.0761 | True |
| outcome | outcome_classifier__uncalibrated | 0.9621 | 0.5642 | 0.0596 | 0.5401 | 0.0869 | False |
| goal_difference | goal_difference_regression__uncalibrated | 0.9623 | 0.5673 | 0.0682 | 0.5434 | 0.0762 | False |
| outcome | outcome_classifier__platt_sigmoid | 0.9623 | 0.5671 | 0.0682 | 0.5401 | 0.0682 | False |
| goals | goal_counts_raw__uncalibrated | 0.9623 | 0.5662 | 0.0601 | 0.5367 | 0.0765 | False |
| goals | current_poisson_baseline__uncalibrated | 0.9631 | 0.5669 | 0.0581 | 0.5401 | 0.0752 | False |
| margin_class | margin_class_classifier__platt_sigmoid | 0.9632 | 0.5675 | 0.0650 | 0.5551 | 0.0658 | False |
| goals | goal_counts_capped__uncalibrated | 0.9641 | 0.5678 | 0.0620 | 0.5440 | 0.0726 | False |
| goal_difference | goal_difference_regression__platt_sigmoid | 0.9657 | 0.5696 | 0.0791 | 0.5434 | 0.0744 | False |
| goals | goal_counts_raw__platt_sigmoid | 0.9662 | 0.5695 | 0.0809 | 0.5367 | 0.0712 | False |
| goals | current_poisson_baseline__platt_sigmoid | 0.9666 | 0.5696 | 0.0732 | 0.5401 | 0.0717 | False |
| goals | goal_counts_capped__platt_sigmoid | 0.9671 | 0.5698 | 0.0667 | 0.5440 | 0.0713 | False |
| goals | goal_counts_raw__isotonic | 0.9673 | 0.5700 | 0.0718 | 0.5367 | 0.0794 | False |
| margin_class | margin_class_classifier__isotonic | 0.9675 | 0.5709 | 0.0617 | 0.5551 | 0.0777 | False |
| elo_baseline | elo_only__uncalibrated | 0.9686 | 0.5720 | 0.0632 | 0.5167 | 0.0429 | False |
| goal_difference | goal_difference_regression__isotonic | 0.9700 | 0.5726 | 0.0617 | 0.5434 | 0.0767 | False |
| residual | goal_diff_residual_correction__isotonic | 0.9762 | 0.5766 | 0.0632 | 0.5496 | 0.0798 | False |
| elo_baseline | elo_only__platt_sigmoid | 0.9776 | 0.5769 | 0.0673 | 0.5167 | 0.0447 | False |
| elo_baseline | elo_only__isotonic | 0.9776 | 0.5766 | 0.0780 | 0.5167 | 0.0569 | False |
| residual | goal_diff_residual_correction__uncalibrated | 0.9788 | 0.5712 | 0.0619 | 0.5496 | 0.1168 | False |
| residual | goal_diff_residual_correction__platt_sigmoid | 0.9796 | 0.5775 | 0.0794 | 0.5496 | 0.0840 | False |
| goals | current_poisson_baseline__isotonic | 1.0279 | 0.5709 | 0.0612 | 0.5401 | 0.2017 | False |
| outcome | outcome_classifier__isotonic | 1.0306 | 0.5725 | 0.0641 | 0.5401 | 0.2008 | False |
| goals | goal_counts_capped__isotonic | 1.0307 | 0.5729 | 0.0634 | 0.5440 | 0.1995 | False |

## Frozen World Cup Results

| model_name | target_type | tournament_year | log_loss | brier_score | accuracy | calibration_error | goal_difference_mae | scoreline_top_5_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elo_only__uncalibrated | elo_baseline | 2010 | 0.9422 | 0.5539 | 0.6000 | 0.0604 | 1.3139 | 0.5200 |
| elo_only__platt_sigmoid | elo_baseline | 2010 | 0.9451 | 0.5555 | 0.6133 | 0.0527 | 1.3139 | 0.5200 |
| elo_only__isotonic | elo_baseline | 2010 | 0.9448 | 0.5542 | 0.6133 | 0.0601 | 1.3139 | 0.5200 |
| current_poisson_baseline__uncalibrated | goals | 2010 | 0.9171 | 0.5372 | 0.6000 | 0.0482 | 1.1572 | 0.6133 |
| current_poisson_baseline__platt_sigmoid | goals | 2010 | 0.9220 | 0.5404 | 0.6000 | 0.0793 | 1.1572 | 0.6133 |
| current_poisson_baseline__isotonic | goals | 2010 | 0.9208 | 0.5423 | 0.6133 | 0.0657 | 1.1572 | 0.6133 |
| goal_counts_raw__uncalibrated | goals | 2010 | 0.9180 | 0.5373 | 0.6000 | 0.0611 | 1.1638 | 0.6000 |
| goal_counts_raw__platt_sigmoid | goals | 2010 | 0.9234 | 0.5411 | 0.5867 | 0.0744 | 1.1638 | 0.6000 |
| goal_counts_raw__isotonic | goals | 2010 | 0.9254 | 0.5437 | 0.5867 | 0.0668 | 1.1638 | 0.6000 |
| goal_counts_capped__uncalibrated | goals | 2010 | 0.9183 | 0.5381 | 0.6000 | 0.0547 | 1.1531 | 0.6133 |
| goal_counts_capped__platt_sigmoid | goals | 2010 | 0.9217 | 0.5399 | 0.6000 | 0.0780 | 1.1531 | 0.6133 |
| goal_counts_capped__isotonic | goals | 2010 | 0.9187 | 0.5434 | 0.6267 | 0.0624 | 1.1531 | 0.6133 |
| outcome_classifier__uncalibrated | outcome | 2010 | 0.9105 | 0.5323 | 0.6000 | 0.0566 | 1.1572 | 0.6133 |
| outcome_classifier__platt_sigmoid | outcome | 2010 | 0.9154 | 0.5357 | 0.6133 | 0.0725 | 1.1572 | 0.6133 |
| outcome_classifier__isotonic | outcome | 2010 | 0.9335 | 0.5467 | 0.6000 | 0.0705 | 1.1572 | 0.6133 |
| goal_difference_regression__uncalibrated | goal_difference | 2010 | 0.9184 | 0.5387 | 0.6133 | 0.0655 | 1.1469 | 0.6267 |
| goal_difference_regression__platt_sigmoid | goal_difference | 2010 | 0.9228 | 0.5410 | 0.6133 | 0.0768 | 1.1469 | 0.6267 |
| goal_difference_regression__isotonic | goal_difference | 2010 | 0.9251 | 0.5463 | 0.6000 | 0.0658 | 1.1469 | 0.6267 |
| margin_class_classifier__uncalibrated | margin_class | 2010 | 0.9141 | 0.5340 | 0.5867 | 0.0619 | 1.1314 | 0.6267 |
| margin_class_classifier__platt_sigmoid | margin_class | 2010 | 0.9156 | 0.5354 | 0.6133 | 0.0763 | 1.1314 | 0.6267 |
| margin_class_classifier__isotonic | margin_class | 2010 | 0.9192 | 0.5381 | 0.6000 | 0.0651 | 1.1314 | 0.6267 |
| goal_diff_residual_correction__uncalibrated | residual | 2010 | 0.9267 | 0.5405 | 0.6000 | 0.0370 | 1.2242 | 0.5733 |
| goal_diff_residual_correction__platt_sigmoid | residual | 2010 | 0.9400 | 0.5522 | 0.6000 | 0.0758 | 1.2242 | 0.5733 |
| goal_diff_residual_correction__isotonic | residual | 2010 | 0.9408 | 0.5525 | 0.6000 | 0.0536 | 1.2242 | 0.5733 |
| elo_only__uncalibrated | elo_baseline | 2014 | 0.9381 | 0.5548 | 0.5938 | 0.0700 | 1.3510 | 0.5156 |
| elo_only__platt_sigmoid | elo_baseline | 2014 | 0.9456 | 0.5615 | 0.5938 | 0.1005 | 1.3510 | 0.5156 |
| elo_only__isotonic | elo_baseline | 2014 | 0.9362 | 0.5578 | 0.6094 | 0.1166 | 1.3510 | 0.5156 |
| current_poisson_baseline__uncalibrated | goals | 2014 | 0.9173 | 0.5421 | 0.6406 | 0.0752 | 1.2057 | 0.5469 |
| current_poisson_baseline__platt_sigmoid | goals | 2014 | 0.9205 | 0.5459 | 0.6406 | 0.0795 | 1.2057 | 0.5469 |
| current_poisson_baseline__isotonic | goals | 2014 | 0.9082 | 0.5396 | 0.6250 | 0.0808 | 1.2057 | 0.5469 |
| goal_counts_raw__uncalibrated | goals | 2014 | 0.9141 | 0.5400 | 0.6562 | 0.0762 | 1.2043 | 0.5625 |
| goal_counts_raw__platt_sigmoid | goals | 2014 | 0.9197 | 0.5454 | 0.6406 | 0.0896 | 1.2043 | 0.5625 |
| goal_counts_raw__isotonic | goals | 2014 | 0.9113 | 0.5407 | 0.6406 | 0.0858 | 1.2043 | 0.5625 |
| goal_counts_capped__uncalibrated | goals | 2014 | 0.9205 | 0.5443 | 0.6406 | 0.0782 | 1.2131 | 0.5469 |
| goal_counts_capped__platt_sigmoid | goals | 2014 | 0.9221 | 0.5470 | 0.6406 | 0.0667 | 1.2131 | 0.5469 |
| goal_counts_capped__isotonic | goals | 2014 | 0.9190 | 0.5451 | 0.6094 | 0.0891 | 1.2131 | 0.5469 |
| outcome_classifier__uncalibrated | outcome | 2014 | 0.9058 | 0.5362 | 0.6406 | 0.0521 | 1.2057 | 0.5469 |
| outcome_classifier__platt_sigmoid | outcome | 2014 | 0.9248 | 0.5485 | 0.6406 | 0.0672 | 1.2057 | 0.5469 |
| outcome_classifier__isotonic | outcome | 2014 | 0.9084 | 0.5400 | 0.6406 | 0.0631 | 1.2057 | 0.5469 |
| goal_difference_regression__uncalibrated | goal_difference | 2014 | 0.9147 | 0.5397 | 0.6406 | 0.0846 | 1.1940 | 0.5469 |
| goal_difference_regression__platt_sigmoid | goal_difference | 2014 | 0.9185 | 0.5432 | 0.6562 | 0.0839 | 1.1940 | 0.5469 |
| goal_difference_regression__isotonic | goal_difference | 2014 | 0.9216 | 0.5466 | 0.6250 | 0.0815 | 1.1940 | 0.5469 |
| margin_class_classifier__uncalibrated | margin_class | 2014 | 0.9134 | 0.5410 | 0.6406 | 0.0615 | 1.2189 | 0.5312 |
| margin_class_classifier__platt_sigmoid | margin_class | 2014 | 0.9281 | 0.5509 | 0.6406 | 0.0687 | 1.2189 | 0.5312 |
| margin_class_classifier__isotonic | margin_class | 2014 | 0.9184 | 0.5467 | 0.6562 | 0.0738 | 1.2189 | 0.5312 |
| goal_diff_residual_correction__uncalibrated | residual | 2014 | 0.9028 | 0.5337 | 0.6250 | 0.0796 | 1.2268 | 0.5625 |
| goal_diff_residual_correction__platt_sigmoid | residual | 2014 | 0.9211 | 0.5457 | 0.6562 | 0.0961 | 1.2268 | 0.5625 |
| goal_diff_residual_correction__isotonic | residual | 2014 | 0.9209 | 0.5443 | 0.6562 | 0.1105 | 1.2268 | 0.5625 |
| elo_only__uncalibrated | elo_baseline | 2018 | 0.9635 | 0.5715 | 0.5469 | 0.0492 | 1.2918 | 0.5781 |
| elo_only__platt_sigmoid | elo_baseline | 2018 | 0.9793 | 0.5820 | 0.5469 | 0.0509 | 1.2918 | 0.5781 |
| elo_only__isotonic | elo_baseline | 2018 | 0.9692 | 0.5757 | 0.5469 | 0.0525 | 1.2918 | 0.5781 |
| current_poisson_baseline__uncalibrated | goals | 2018 | 0.9437 | 0.5589 | 0.5625 | 0.0509 | 1.1389 | 0.5781 |
| current_poisson_baseline__platt_sigmoid | goals | 2018 | 0.9520 | 0.5618 | 0.5625 | 0.0725 | 1.1389 | 0.5781 |
| current_poisson_baseline__isotonic | goals | 2018 | 0.9536 | 0.5630 | 0.5625 | 0.0461 | 1.1389 | 0.5781 |
| goal_counts_raw__uncalibrated | goals | 2018 | 0.9416 | 0.5577 | 0.5625 | 0.0491 | 1.1429 | 0.5625 |
| goal_counts_raw__platt_sigmoid | goals | 2018 | 0.9508 | 0.5611 | 0.5625 | 0.0993 | 1.1429 | 0.5625 |
| goal_counts_raw__isotonic | goals | 2018 | 0.9484 | 0.5609 | 0.5625 | 0.0632 | 1.1429 | 0.5625 |
| goal_counts_capped__uncalibrated | goals | 2018 | 0.9463 | 0.5606 | 0.5469 | 0.0653 | 1.1411 | 0.5781 |
| goal_counts_capped__platt_sigmoid | goals | 2018 | 0.9528 | 0.5624 | 0.5469 | 0.0651 | 1.1411 | 0.5781 |
| goal_counts_capped__isotonic | goals | 2018 | 0.9563 | 0.5666 | 0.5625 | 0.0430 | 1.1411 | 0.5781 |
| outcome_classifier__uncalibrated | outcome | 2018 | 0.9418 | 0.5577 | 0.5469 | 0.0568 | 1.1389 | 0.5781 |
| outcome_classifier__platt_sigmoid | outcome | 2018 | 0.9463 | 0.5581 | 0.5469 | 0.0727 | 1.1389 | 0.5781 |
| outcome_classifier__isotonic | outcome | 2018 | 0.9499 | 0.5629 | 0.5469 | 0.0443 | 1.1389 | 0.5781 |
| goal_difference_regression__uncalibrated | goal_difference | 2018 | 0.9407 | 0.5566 | 0.5625 | 0.0667 | 1.1324 | 0.5781 |
| goal_difference_regression__platt_sigmoid | goal_difference | 2018 | 0.9457 | 0.5587 | 0.5469 | 0.0902 | 1.1324 | 0.5781 |
| goal_difference_regression__isotonic | goal_difference | 2018 | 0.9498 | 0.5626 | 0.5469 | 0.0276 | 1.1324 | 0.5781 |
| margin_class_classifier__uncalibrated | margin_class | 2018 | 0.9455 | 0.5604 | 0.5469 | 0.0552 | 1.1431 | 0.5938 |
| margin_class_classifier__platt_sigmoid | margin_class | 2018 | 0.9497 | 0.5598 | 0.5469 | 0.0523 | 1.1431 | 0.5938 |
| margin_class_classifier__isotonic | margin_class | 2018 | 0.9506 | 0.5612 | 0.5625 | 0.0380 | 1.1431 | 0.5938 |
| goal_diff_residual_correction__uncalibrated | residual | 2018 | 0.9329 | 0.5525 | 0.5781 | 0.0527 | 1.1707 | 0.6094 |
| goal_diff_residual_correction__platt_sigmoid | residual | 2018 | 0.9531 | 0.5637 | 0.5469 | 0.1055 | 1.1707 | 0.6094 |
| goal_diff_residual_correction__isotonic | residual | 2018 | 0.9486 | 0.5616 | 0.5781 | 0.0511 | 1.1707 | 0.6094 |
| elo_only__uncalibrated | elo_baseline | 2022 | 1.0308 | 0.6077 | 0.5625 | 0.0732 | 1.4151 | 0.4531 |
| elo_only__platt_sigmoid | elo_baseline | 2022 | 1.0402 | 0.6084 | 0.5000 | 0.0653 | 1.4151 | 0.4531 |
| elo_only__isotonic | elo_baseline | 2022 | 1.0603 | 0.6187 | 0.4688 | 0.0826 | 1.4151 | 0.4531 |
| current_poisson_baseline__uncalibrated | goals | 2022 | 1.0744 | 0.6292 | 0.5000 | 0.0582 | 1.3930 | 0.4219 |
| current_poisson_baseline__platt_sigmoid | goals | 2022 | 1.0719 | 0.6303 | 0.5000 | 0.0617 | 1.3930 | 0.4219 |
| current_poisson_baseline__isotonic | goals | 2022 | 1.3291 | 0.6385 | 0.5156 | 0.0522 | 1.3930 | 0.4219 |
| goal_counts_raw__uncalibrated | goals | 2022 | 1.0757 | 0.6298 | 0.5000 | 0.0540 | 1.3958 | 0.4219 |
| goal_counts_raw__platt_sigmoid | goals | 2022 | 1.0710 | 0.6303 | 0.5000 | 0.0605 | 1.3958 | 0.4219 |
| goal_counts_raw__isotonic | goals | 2022 | 1.0841 | 0.6347 | 0.5000 | 0.0714 | 1.3958 | 0.4219 |
| goal_counts_capped__uncalibrated | goals | 2022 | 1.0714 | 0.6281 | 0.5000 | 0.0498 | 1.3888 | 0.4375 |
| goal_counts_capped__platt_sigmoid | goals | 2022 | 1.0717 | 0.6298 | 0.5000 | 0.0569 | 1.3888 | 0.4375 |
| goal_counts_capped__isotonic | goals | 2022 | 1.3288 | 0.6364 | 0.5156 | 0.0593 | 1.3888 | 0.4375 |
| outcome_classifier__uncalibrated | outcome | 2022 | 1.0902 | 0.6305 | 0.5156 | 0.0730 | 1.3930 | 0.4219 |
| outcome_classifier__platt_sigmoid | outcome | 2022 | 1.0628 | 0.6260 | 0.5156 | 0.0603 | 1.3930 | 0.4219 |
| outcome_classifier__isotonic | outcome | 2022 | 1.3308 | 0.6401 | 0.5156 | 0.0785 | 1.3930 | 0.4219 |
| goal_difference_regression__uncalibrated | goal_difference | 2022 | 1.0754 | 0.6340 | 0.5000 | 0.0561 | 1.4008 | 0.4219 |
| goal_difference_regression__platt_sigmoid | goal_difference | 2022 | 1.0758 | 0.6355 | 0.5000 | 0.0653 | 1.4008 | 0.4219 |
| goal_difference_regression__isotonic | goal_difference | 2022 | 1.0836 | 0.6348 | 0.5000 | 0.0720 | 1.4008 | 0.4219 |
| margin_class_classifier__uncalibrated | margin_class | 2022 | 1.0736 | 0.6262 | 0.5156 | 0.0563 | 1.3633 | 0.4688 |
| margin_class_classifier__platt_sigmoid | margin_class | 2022 | 1.0596 | 0.6240 | 0.5000 | 0.0627 | 1.3633 | 0.4688 |
| margin_class_classifier__isotonic | margin_class | 2022 | 1.0820 | 0.6377 | 0.5312 | 0.0698 | 1.3633 | 0.4688 |
| goal_diff_residual_correction__uncalibrated | residual | 2022 | 1.1529 | 0.6582 | 0.4688 | 0.0785 | 1.4637 | 0.4531 |
| goal_diff_residual_correction__platt_sigmoid | residual | 2022 | 1.1041 | 0.6485 | 0.4531 | 0.0401 | 1.4637 | 0.4531 |
| goal_diff_residual_correction__isotonic | residual | 2022 | 1.0947 | 0.6481 | 0.4531 | 0.0377 | 1.4637 | 0.4531 |

## Leakage Checks

- Every experiment uses the existing pre-match feature frame; rolling form is shifted and Elo is pre-match.
- Training cutoffs are 2006, 2010, 2014, and 2018 for World Cups 2010, 2014, 2018, and 2022.
- Residual expectations use only pre-match Elo and neutral/home context.
- Calibration uses only a chronological tail of the already-frozen training period.
- Future-form targets are saved separately and never joined into the match feature table.

## Recommendation

`margin_class_classifier__uncalibrated` had the lowest average outcome log loss across the four frozen World Cups. Scoreline and calibration metrics are reported separately; future-form targets were excluded from modelling.

Accuracy was treated as secondary. These experiments remain noisy because only four World Cups are available as frozen test tournaments.
