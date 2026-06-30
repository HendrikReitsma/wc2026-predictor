# Training Strategy Report

## Recommended Configuration

- Minimum training year: **1990**.
- Time-decay half-life: **none**.
- Match-importance profile: **aggressive**.
- Goal cap: **8**.
- Rating model: **standard_elo**, Elo K scale **1.2**.
- Draw correction: **similar_strength**.
- Calibration: **uncalibrated**.
- Base-pipeline frozen-World-Cup mean log loss: **0.9613**.
- Draw/calibration comparison mean log loss: **0.9614**.

The search was staged rather than a huge unconstrained Cartesian sweep: window/decay/importance first, then goal cap, then rating update, then draw correction and calibration. This reduces overfitting risk with only four World Cups. The draw/calibration stage reserves a chronological two-year calibration tail, so its absolute score is not directly comparable with the base-pipeline score fitted on the full frozen training window.

## Findings

- Best training window: **1990 onward**.
- Best time decay: **none**.
- Match weighting helped: best weighted log loss 0.9617, best unweighted 0.9623.
- Best margin-pipeline calibration/draw choice: **similar_strength + uncalibrated**.
- Raw Poisson log loss: **0.9621**; best Poisson draw/calibration variant: **none + uncalibrated** at **0.9621**.
- Best smoothed dynamic rating log loss: **0.9804** versus standard Elo scale 1.0 at **0.9617**.

## Training Strategy Comparison

| search_stage | minimum_year | half_life_years | importance_profile | goal_cap | rating_model | elo_k_scale | world_cups_covered | avg_log_loss | avg_brier_score | avg_calibration_error | avg_scoreline_top_5_hit_rate | stability_score | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rating_model | 1990 | 1000000000.0000 | aggressive | 8 | standard_elo | 1.2000 | 4 | 0.9613 | 0.5650 | 0.0604 | 0.5401 | 0.0807 | True |
| goal_cap | 1990 | 1000000000.0000 | aggressive | 6 | standard_elo | 1.0000 | 4 | 0.9617 | 0.5654 | 0.0604 | 0.5434 | 0.0761 | False |
| goal_cap | 1990 | 1000000000.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9617 | 0.5654 | 0.0604 | 0.5551 | 0.0761 | False |
| goal_cap | 1990 | 1000000000.0000 | aggressive | 10 | standard_elo | 1.0000 | 4 | 0.9617 | 0.5654 | 0.0604 | 0.5551 | 0.0761 | False |
| rating_model | 1990 | 1000000000.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9617 | 0.5654 | 0.0604 | 0.5551 | 0.0761 | False |
| window_decay_importance | 1990 | 1000000000.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9617 | 0.5654 | 0.0604 | 0.5551 | 0.0761 | False |
| window_decay_importance | 1990 | 1000000000.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9617 | 0.5657 | 0.0647 | 0.5590 | 0.0747 | False |
| window_decay_importance | 1990 | 1000000000.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9623 | 0.5664 | 0.0634 | 0.5518 | 0.0730 | False |
| window_decay_importance | 1990 | 12.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9623 | 0.5660 | 0.0630 | 0.5479 | 0.0768 | False |
| window_decay_importance | 1990 | 12.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9625 | 0.5664 | 0.0651 | 0.5590 | 0.0754 | False |
| window_decay_importance | 1990 | 8.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9628 | 0.5664 | 0.0691 | 0.5518 | 0.0771 | False |
| window_decay_importance | 1990 | 8.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9629 | 0.5667 | 0.0676 | 0.5518 | 0.0758 | False |
| window_decay_importance | 1998 | 1000000000.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9629 | 0.5663 | 0.0601 | 0.5440 | 0.0762 | False |
| window_decay_importance | 1990 | 12.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9631 | 0.5670 | 0.0679 | 0.5557 | 0.0738 | False |
| window_decay_importance | 1998 | 1000000000.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9632 | 0.5668 | 0.0652 | 0.5440 | 0.0746 | False |
| window_decay_importance | 1998 | 12.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9632 | 0.5667 | 0.0722 | 0.5479 | 0.0765 | False |
| rating_model | 1990 | 1000000000.0000 | aggressive | 8 | standard_elo | 0.8000 | 4 | 0.9633 | 0.5669 | 0.0630 | 0.5434 | 0.0685 | False |
| window_decay_importance | 1998 | 12.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9633 | 0.5671 | 0.0694 | 0.5518 | 0.0750 | False |
| window_decay_importance | 1998 | 8.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9635 | 0.5670 | 0.0751 | 0.5518 | 0.0766 | False |
| window_decay_importance | 1990 | 8.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9635 | 0.5674 | 0.0702 | 0.5557 | 0.0740 | False |
| window_decay_importance | 1998 | 8.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9637 | 0.5674 | 0.0697 | 0.5484 | 0.0751 | False |
| window_decay_importance | 1998 | 1000000000.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9637 | 0.5675 | 0.0700 | 0.5596 | 0.0725 | False |
| window_decay_importance | 1998 | 12.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9641 | 0.5679 | 0.0714 | 0.5557 | 0.0731 | False |
| window_decay_importance | 1998 | 8.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9645 | 0.5682 | 0.0673 | 0.5484 | 0.0734 | False |
| window_decay_importance | 1990 | 4.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9645 | 0.5677 | 0.0714 | 0.5479 | 0.0778 | False |
| window_decay_importance | 1990 | 4.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9646 | 0.5680 | 0.0775 | 0.5484 | 0.0764 | False |
| window_decay_importance | 1998 | 4.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9649 | 0.5681 | 0.0823 | 0.5518 | 0.0775 | False |
| window_decay_importance | 1998 | 4.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9649 | 0.5684 | 0.0782 | 0.5484 | 0.0760 | False |
| window_decay_importance | 1990 | 4.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9653 | 0.5687 | 0.0702 | 0.5484 | 0.0746 | False |
| window_decay_importance | 1998 | 4.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9659 | 0.5693 | 0.0756 | 0.5445 | 0.0741 | False |
| window_decay_importance | 2002 | 1000000000.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9661 | 0.5692 | 0.0672 | 0.5490 | 0.0711 | False |
| window_decay_importance | 2002 | 1000000000.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9662 | 0.5691 | 0.0685 | 0.5445 | 0.0720 | False |
| window_decay_importance | 2002 | 12.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9663 | 0.5694 | 0.0678 | 0.5484 | 0.0722 | False |
| window_decay_importance | 2002 | 12.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9664 | 0.5693 | 0.0715 | 0.5484 | 0.0732 | False |
| window_decay_importance | 2002 | 8.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9665 | 0.5696 | 0.0675 | 0.5445 | 0.0727 | False |
| window_decay_importance | 2002 | 8.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9665 | 0.5695 | 0.0724 | 0.5484 | 0.0739 | False |
| window_decay_importance | 2002 | 1000000000.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9668 | 0.5697 | 0.0697 | 0.5490 | 0.0697 | False |
| window_decay_importance | 2002 | 4.0000 | aggressive | 8 | standard_elo | 1.0000 | 4 | 0.9670 | 0.5700 | 0.0842 | 0.5518 | 0.0756 | False |
| window_decay_importance | 2002 | 4.0000 | balanced | 8 | standard_elo | 1.0000 | 4 | 0.9670 | 0.5701 | 0.0782 | 0.5479 | 0.0742 | False |
| window_decay_importance | 2002 | 12.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9672 | 0.5701 | 0.0730 | 0.5563 | 0.0708 | False |
| window_decay_importance | 2002 | 8.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9674 | 0.5702 | 0.0704 | 0.5484 | 0.0713 | False |
| window_decay_importance | 2002 | 4.0000 | none | 8 | standard_elo | 1.0000 | 4 | 0.9680 | 0.5708 | 0.0695 | 0.5484 | 0.0724 | False |
| rating_model | 1990 | 1000000000.0000 | aggressive | 8 | smoothed_dynamic | 1.2000 | 4 | 0.9804 | 0.5818 | 0.0676 | 0.5518 | 0.0584 | False |
| rating_model | 1990 | 1000000000.0000 | aggressive | 8 | smoothed_dynamic | 1.0000 | 4 | 0.9826 | 0.5832 | 0.0639 | 0.5440 | 0.0567 | False |
| rating_model | 1990 | 1000000000.0000 | aggressive | 8 | smoothed_dynamic | 0.8000 | 4 | 0.9856 | 0.5852 | 0.0699 | 0.5395 | 0.0547 | False |
| window_decay_importance | 2010 | 1000000000.0000 | none | 8 | standard_elo | 1.0000 | 3 | 0.9787 | 0.5764 | 0.0816 | 0.5156 | 0.0833 | False |
| window_decay_importance | 2010 | 12.0000 | none | 8 | standard_elo | 1.0000 | 3 | 0.9794 | 0.5769 | 0.0832 | 0.5156 | 0.0836 | False |
| window_decay_importance | 2010 | 8.0000 | none | 8 | standard_elo | 1.0000 | 3 | 0.9797 | 0.5771 | 0.0797 | 0.5208 | 0.0836 | False |
| window_decay_importance | 2010 | 4.0000 | none | 8 | standard_elo | 1.0000 | 3 | 0.9808 | 0.5779 | 0.0754 | 0.5208 | 0.0843 | False |
| window_decay_importance | 2010 | 1000000000.0000 | balanced | 8 | standard_elo | 1.0000 | 3 | 0.9831 | 0.5797 | 0.0700 | 0.5260 | 0.0812 | False |
| window_decay_importance | 2010 | 12.0000 | balanced | 8 | standard_elo | 1.0000 | 3 | 0.9836 | 0.5801 | 0.0696 | 0.5208 | 0.0813 | False |
| window_decay_importance | 2010 | 8.0000 | balanced | 8 | standard_elo | 1.0000 | 3 | 0.9839 | 0.5803 | 0.0699 | 0.5208 | 0.0814 | False |
| window_decay_importance | 2010 | 4.0000 | balanced | 8 | standard_elo | 1.0000 | 3 | 0.9849 | 0.5811 | 0.0735 | 0.5312 | 0.0821 | False |
| window_decay_importance | 2010 | 1000000000.0000 | aggressive | 8 | standard_elo | 1.0000 | 3 | 0.9878 | 0.5829 | 0.0758 | 0.5052 | 0.0793 | False |
| window_decay_importance | 2010 | 12.0000 | aggressive | 8 | standard_elo | 1.0000 | 3 | 0.9882 | 0.5832 | 0.0773 | 0.5104 | 0.0794 | False |
| window_decay_importance | 2010 | 8.0000 | aggressive | 8 | standard_elo | 1.0000 | 3 | 0.9886 | 0.5835 | 0.0786 | 0.5156 | 0.0796 | False |
| window_decay_importance | 2010 | 4.0000 | aggressive | 8 | standard_elo | 1.0000 | 3 | 0.9895 | 0.5842 | 0.0795 | 0.5208 | 0.0800 | False |

## Calibration And Draw Correction

| probability_source | draw_correction | calibration_method | avg_draw_alpha | avg_log_loss | avg_brier_score | avg_calibration_error | avg_similar_strength_log_loss | avg_similar_strength_draw_brier | stability_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| margin | similar_strength | uncalibrated | 1.0188 | 0.9614 | 0.5649 | 0.0646 | 1.0349 | 0.1825 | 0.0789 |
| margin | similar_strength | sigmoid | 1.0188 | 0.9623 | 0.5669 | 0.0646 | 1.0406 | 0.1840 | 0.0687 |
| margin | global | uncalibrated | 1.0125 | 0.9623 | 0.5651 | 0.0671 | 1.0360 | 0.1829 | 0.0802 |
| margin | none | uncalibrated | 1.0000 | 0.9626 | 0.5659 | 0.0665 | 1.0388 | 0.1840 | 0.0796 |
| margin | none | sigmoid | 1.0000 | 0.9629 | 0.5672 | 0.0609 | 1.0408 | 0.1840 | 0.0693 |
| margin | global | sigmoid | 1.0125 | 0.9630 | 0.5672 | 0.0609 | 1.0409 | 0.1841 | 0.0693 |
| margin | similar_strength | isotonic | 1.0188 | 1.0270 | 0.5705 | 0.0595 | 1.0493 | 0.1832 | 0.2041 |
| margin | none | isotonic | 1.0000 | 1.0271 | 0.5709 | 0.0558 | 1.0497 | 0.1826 | 0.2040 |
| margin | global | isotonic | 1.0125 | 1.0277 | 0.5717 | 0.0503 | 1.0521 | 0.1827 | 0.2044 |
| poisson | none | uncalibrated | 1.0000 | 0.9621 | 0.5661 | 0.0582 | 1.0357 | 0.1841 | 0.0802 |
| poisson | similar_strength | uncalibrated | 1.2375 | 0.9657 | 0.5683 | 0.0752 | 1.0384 | 0.1851 | 0.0784 |
| poisson | none | sigmoid | 1.0000 | 0.9657 | 0.5690 | 0.0781 | 1.0406 | 0.1855 | 0.0765 |
| poisson | global | sigmoid | 1.1375 | 0.9658 | 0.5690 | 0.0782 | 1.0405 | 0.1855 | 0.0766 |
| poisson | similar_strength | sigmoid | 1.2375 | 0.9667 | 0.5698 | 0.0773 | 1.0441 | 0.1860 | 0.0759 |
| poisson | global | uncalibrated | 1.1375 | 0.9668 | 0.5689 | 0.0753 | 1.0383 | 0.1851 | 0.0808 |
| poisson | similar_strength | isotonic | 1.2375 | 1.0263 | 0.5694 | 0.0689 | 1.0400 | 0.1836 | 0.2082 |
| poisson | global | isotonic | 1.1375 | 1.0278 | 0.5705 | 0.0668 | 1.0436 | 0.1848 | 0.2106 |
| poisson | none | isotonic | 1.0000 | 1.0285 | 0.5707 | 0.0651 | 1.0456 | 0.1849 | 0.2113 |

## Rating Model Comparison

| rating_model | elo_k_scale | avg_log_loss | avg_brier_score | avg_calibration_error | avg_scoreline_top_5_hit_rate | stability_score | beats_standard_elo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard_elo | 1.2000 | 0.9613 | 0.5650 | 0.0604 | 0.5401 | 0.0807 | True |
| standard_elo | 1.0000 | 0.9617 | 0.5654 | 0.0604 | 0.5551 | 0.0761 | False |
| standard_elo | 0.8000 | 0.9633 | 0.5669 | 0.0630 | 0.5434 | 0.0685 | False |
| smoothed_dynamic | 1.2000 | 0.9804 | 0.5818 | 0.0676 | 0.5518 | 0.0584 | False |
| smoothed_dynamic | 1.0000 | 0.9826 | 0.5832 | 0.0639 | 0.5440 | 0.0567 | False |
| smoothed_dynamic | 0.8000 | 0.9856 | 0.5852 | 0.0699 | 0.5395 | 0.0547 | False |

## Limitations

- No new feature groups were added; rating alternatives replace the existing Elo columns only.
- Four World Cups are a small model-selection sample, so tiny differences should not be overinterpreted.
- The smoothed dynamic rating is Glicko-style, not a complete implementation of official Glicko-2.
- Calibration and draw correction use chronological calibration tails only.
