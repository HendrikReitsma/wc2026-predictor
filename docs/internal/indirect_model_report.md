# Indirect Model Report

## A. Executive Summary

- Selected final variant: **baseline**.
- Best conventional indirect challenger: **trend_small**.
- Decision: Best indirect challenger trend_small did not clear all configured proof checks; baseline retained.
- Baseline mean log loss: **0.9757**; challenger: **0.9757**.
- Baseline Brier score: **0.5745**; challenger: **0.5745**.
- Indirect continuous targets use the selected match pipeline's existing probability calibration; no separate indirect probability calibrator was fitted.
- The indirect-only sanity check had lower log loss, but worse Brier score, calibration, and top-5 scoreline coverage; it was not eligible to replace the coherent match-and-score pipeline.
- No betting odds or player-level data are used.

## B. Team Strength Trend Model

The trend dataset contains **98,798** team-before-match snapshots. Features use only matches before each snapshot. The primary target is next-five-match Elo change; its target completion date must be before a training cutoff.

### Strongest Positive 2026 Trend Scores

| team | base_elo | expected_future_elo_delta | expected_future_performance_above_expectation | trend_adjustment |
| --- | --- | --- | --- | --- |
| Bosnia and Herzegovina | 1635.0415 | 7.4476 | 0.3979 | 1.8619 |
| Haiti | 1671.6267 | 5.8133 | 0.2614 | 1.4533 |
| New Zealand | 1684.1593 | 5.8117 | 0.4467 | 1.4529 |
| Egypt | 1780.0067 | 4.4535 | 0.1586 | 1.1134 |
| Ghana | 1606.4393 | 4.3689 | 0.2120 | 1.0922 |
| United States | 1825.5070 | 4.1180 | 0.4314 | 1.0295 |
| Brazil | 2022.6959 | 3.8643 | 0.5188 | 0.9661 |
| Paraguay | 1848.1977 | 3.4239 | 0.2737 | 0.8560 |
| Japan | 1956.5233 | 3.0909 | 0.3914 | 0.7727 |
| Norway | 1896.2748 | 2.7454 | 0.3787 | 0.6864 |

### Strongest Negative 2026 Trend Scores

| team | base_elo | expected_future_elo_delta | expected_future_performance_above_expectation | trend_adjustment |
| --- | --- | --- | --- | --- |
| Senegal | 1873.6559 | -5.6254 | -0.2071 | -1.4064 |
| Panama | 1791.0773 | -5.0708 | -0.2183 | -1.2677 |
| Croatia | 1929.5397 | -4.8176 | -0.0381 | -1.2044 |
| Argentina | 2125.2607 | -4.5799 | 0.0603 | -1.1450 |
| France | 2063.1363 | -4.2352 | 0.0232 | -1.0588 |
| Morocco | 1950.0218 | -4.0579 | -0.1088 | -1.0145 |
| Czech Republic | 1771.2706 | -2.8155 | -0.0931 | -0.7039 |
| England | 2007.2943 | -2.0705 | 0.0642 | -0.5176 |
| DR Congo | 1717.8651 | -1.5847 | -0.1872 | -0.3962 |
| Jordan | 1734.8846 | -1.5489 | -0.0865 | -0.3872 |

The smallest trend correction was the best conventional indirect challenger, but it slightly worsened both log loss and Brier score, so it was not selected.

## C. Tournament Readiness Model

The readiness dataset contains **272** team-before-World-Cup rows. The target is tournament goal-difference performance above pre-match Elo expectation. Features stop before tournament start; targets complete at tournament end.

### Strongest Positive 2026 Readiness Scores

| team | base_elo | tournament_readiness_score | expected_group_points_adjustment | overperformance_probability |
| --- | --- | --- | --- | --- |
| Bosnia and Herzegovina | 1635.0415 | 1.3553 | 0.3686 | 0.9589 |
| Ghana | 1606.4393 | 1.0757 | -0.3788 | 0.9161 |
| Qatar | 1565.5994 | 0.8396 | -1.2611 | 0.8591 |
| Curaçao | 1556.0716 | 0.8327 | -1.4373 | 0.8571 |
| Cape Verde | 1665.6257 | 0.7976 | -0.4847 | 0.8467 |
| Haiti | 1671.6267 | 0.7781 | -0.1522 | 0.8408 |
| Sweden | 1767.0834 | 0.6704 | -0.2305 | 0.8050 |
| New Zealand | 1684.1593 | 0.6685 | -0.9200 | 0.8043 |
| South Africa | 1634.6428 | 0.6301 | -1.0468 | 0.7904 |
| Czech Republic | 1771.2706 | 0.4712 | -0.1605 | 0.7271 |

### Strongest Negative 2026 Readiness Scores

| team | base_elo | tournament_readiness_score | expected_group_points_adjustment | overperformance_probability |
| --- | --- | --- | --- | --- |
| France | 2063.1363 | -0.9432 | 0.1590 | 0.1133 |
| Spain | 2128.2789 | -0.9301 | 0.3782 | 0.1165 |
| Morocco | 1950.0218 | -0.7362 | -0.7097 | 0.1726 |
| Argentina | 2125.2607 | -0.6944 | 1.1578 | 0.1867 |
| Mexico | 1915.7246 | -0.6904 | 0.3736 | 0.1880 |
| Portugal | 2002.7842 | -0.6014 | 0.1994 | 0.2204 |
| England | 2007.2943 | -0.5876 | 0.4781 | 0.2256 |
| Canada | 1860.9571 | -0.5774 | 0.1670 | 0.2296 |
| Senegal | 1873.6559 | -0.5082 | -0.9776 | 0.2574 |
| Croatia | 1929.5397 | -0.4434 | -0.0867 | 0.2849 |

Readiness corrections did not improve the primary outcome metrics and generally reduced scoreline quality.
The required `expected_group_points_adjustment` output is a tournament-points-residual proxy, not a directly trained group-stage target, because reliable historical group-stage labels are unavailable.

## D. Baseline Versus Indirect Comparison

| model_variant | trend_weight | readiness_weight | avg_log_loss | avg_brier_score | avg_calibration_error | avg_scoreline_top_5_hit_rate | stability_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| indirect_only | 0.5000 | 0.5000 | 0.9742 | 0.5756 | 0.0657 | 0.5156 | 0.0476 |
| baseline | 0.0000 | 0.0000 | 0.9757 | 0.5745 | 0.0552 | 0.5430 | 0.0734 |
| trend_small | 0.2500 | 0.0000 | 0.9757 | 0.5745 | 0.0564 | 0.5430 | 0.0738 |
| trend_medium | 0.5000 | 0.0000 | 0.9758 | 0.5746 | 0.0581 | 0.5430 | 0.0743 |
| readiness_small | 0.0000 | 0.2500 | 0.9758 | 0.5749 | 0.0607 | 0.5352 | 0.0732 |
| both_small | 0.2500 | 0.2500 | 0.9759 | 0.5749 | 0.0591 | 0.5352 | 0.0737 |
| trend_large | 1.0000 | 0.0000 | 0.9759 | 0.5746 | 0.0628 | 0.5352 | 0.0752 |
| readiness_medium | 0.0000 | 0.5000 | 0.9763 | 0.5755 | 0.0659 | 0.5430 | 0.0730 |
| both_medium | 0.5000 | 0.5000 | 0.9765 | 0.5757 | 0.0655 | 0.5469 | 0.0739 |
| readiness_large | 0.0000 | 1.0000 | 0.9782 | 0.5774 | 0.0554 | 0.5469 | 0.0725 |
| both_large | 1.0000 | 1.0000 | 0.9792 | 0.5780 | 0.0606 | 0.5430 | 0.0741 |

### World Cup By World Cup

| model_variant | worldcup_year | log_loss | brier_score | calibration_error | scoreline_top_5_hit_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | 2010 | 0.9738 | 0.5756 | 0.0441 | 0.6094 |
| trend_small | 2010 | 0.9739 | 0.5757 | 0.0439 | 0.6094 |
| trend_medium | 2010 | 0.9740 | 0.5758 | 0.0438 | 0.5938 |
| trend_large | 2010 | 0.9743 | 0.5761 | 0.0434 | 0.5938 |
| readiness_small | 2010 | 0.9738 | 0.5761 | 0.0501 | 0.5938 |
| readiness_medium | 2010 | 0.9744 | 0.5770 | 0.0524 | 0.6094 |
| readiness_large | 2010 | 0.9775 | 0.5800 | 0.0324 | 0.5938 |
| both_small | 2010 | 0.9740 | 0.5762 | 0.0501 | 0.5938 |
| both_medium | 2010 | 0.9749 | 0.5774 | 0.0533 | 0.6094 |
| both_large | 2010 | 0.9791 | 0.5814 | 0.0327 | 0.5938 |
| indirect_only | 2010 | 0.9676 | 0.5720 | 0.0345 | 0.4688 |
| baseline | 2014 | 0.9042 | 0.5335 | 0.0658 | 0.5469 |
| trend_small | 2014 | 0.9036 | 0.5331 | 0.0658 | 0.5469 |
| trend_medium | 2014 | 0.9031 | 0.5327 | 0.0596 | 0.5469 |
| trend_large | 2014 | 0.9020 | 0.5320 | 0.0638 | 0.5469 |
| readiness_small | 2014 | 0.9044 | 0.5336 | 0.0672 | 0.5469 |
| readiness_medium | 2014 | 0.9050 | 0.5339 | 0.0729 | 0.5469 |
| readiness_large | 2014 | 0.9070 | 0.5352 | 0.0579 | 0.5781 |
| both_small | 2014 | 0.9039 | 0.5333 | 0.0672 | 0.5469 |
| both_medium | 2014 | 0.9040 | 0.5333 | 0.0582 | 0.5625 |
| both_large | 2014 | 0.9055 | 0.5341 | 0.0598 | 0.5781 |
| indirect_only | 2014 | 0.9264 | 0.5458 | 0.0809 | 0.5312 |
| baseline | 2018 | 0.9478 | 0.5618 | 0.0575 | 0.5781 |
| trend_small | 2018 | 0.9478 | 0.5618 | 0.0575 | 0.5781 |
| trend_medium | 2018 | 0.9479 | 0.5618 | 0.0701 | 0.5781 |
| trend_large | 2018 | 0.9481 | 0.5619 | 0.0717 | 0.5781 |
| readiness_small | 2018 | 0.9481 | 0.5619 | 0.0603 | 0.5781 |
| readiness_medium | 2018 | 0.9486 | 0.5621 | 0.0695 | 0.5781 |
| readiness_large | 2018 | 0.9506 | 0.5631 | 0.0763 | 0.5781 |
| both_small | 2018 | 0.9482 | 0.5619 | 0.0604 | 0.5781 |
| both_medium | 2018 | 0.9489 | 0.5622 | 0.0696 | 0.5781 |
| both_large | 2018 | 0.9514 | 0.5635 | 0.0690 | 0.5781 |
| indirect_only | 2018 | 0.9627 | 0.5705 | 0.0580 | 0.6094 |
| baseline | 2022 | 1.0770 | 0.6272 | 0.0535 | 0.4375 |
| trend_small | 2022 | 1.0776 | 0.6275 | 0.0585 | 0.4375 |
| trend_medium | 2022 | 1.0782 | 0.6279 | 0.0591 | 0.4531 |
| trend_large | 2022 | 1.0794 | 0.6286 | 0.0723 | 0.4219 |
| readiness_small | 2022 | 1.0770 | 0.6281 | 0.0654 | 0.4219 |
| readiness_medium | 2022 | 1.0771 | 0.6290 | 0.0688 | 0.4375 |
| readiness_large | 2022 | 1.0779 | 0.6313 | 0.0551 | 0.4375 |
| both_small | 2022 | 1.0775 | 0.6284 | 0.0587 | 0.4219 |
| both_medium | 2022 | 1.0783 | 0.6298 | 0.0808 | 0.4375 |
| both_large | 2022 | 1.0806 | 0.6329 | 0.0810 | 0.4219 |
| indirect_only | 2022 | 1.0399 | 0.6140 | 0.0895 | 0.4531 |

Historical source data does not provide reliable group-stage labels, so group-points MAE and group-qualification accuracy are intentionally left unavailable.
These indirect experiments use exact `FIFA World Cup` rows and train through each tournament start. Their absolute scores are therefore not directly comparable with earlier reports that use older frozen training cutoffs.

## E. 2026 Prediction Impact

Because the baseline was selected, final production adjustments are zero. The tables below show the best indirect challenger for comparison.

### Most Changed Matches

| home_team | away_team | baseline_p_home_win | indirect_p_home_win | baseline_p_draw | indirect_p_draw | baseline_p_away_win | indirect_p_away_win | largest_probability_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ghana | Panama | 0.2151 | 0.2179 | 0.2034 | 0.2043 | 0.5815 | 0.5778 | 0.0037 |
| Norway | Senegal | 0.4600 | 0.4633 | 0.2416 | 0.2411 | 0.2985 | 0.2956 | 0.0033 |
| Bosnia and Herzegovina | Qatar | 0.5929 | 0.5960 | 0.2093 | 0.2085 | 0.1977 | 0.1955 | 0.0031 |
| Brazil | Morocco | 0.4393 | 0.4421 | 0.2717 | 0.2714 | 0.2890 | 0.2865 | 0.0029 |
| Morocco | Haiti | 0.7663 | 0.7635 | 0.1571 | 0.1586 | 0.0766 | 0.0779 | 0.0028 |
| Norway | France | 0.1999 | 0.2017 | 0.2263 | 0.2271 | 0.5738 | 0.5712 | 0.0026 |
| Croatia | Ghana | 0.7960 | 0.7935 | 0.1285 | 0.1296 | 0.0755 | 0.0768 | 0.0025 |
| Turkey | United States | 0.7074 | 0.7054 | 0.1561 | 0.1567 | 0.1366 | 0.1378 | 0.0019 |
| Scotland | Morocco | 0.1891 | 0.1904 | 0.2363 | 0.2369 | 0.5746 | 0.5727 | 0.0019 |
| Turkey | Paraguay | 0.4558 | 0.4539 | 0.2415 | 0.2417 | 0.3027 | 0.3044 | 0.0018 |
| Haiti | Scotland | 0.2370 | 0.2383 | 0.2149 | 0.2154 | 0.5481 | 0.5463 | 0.0018 |
| Canada | Bosnia and Herzegovina | 0.7638 | 0.7621 | 0.1595 | 0.1604 | 0.0767 | 0.0775 | 0.0017 |
| Switzerland | Bosnia and Herzegovina | 0.7360 | 0.7343 | 0.1605 | 0.1613 | 0.1035 | 0.1045 | 0.0017 |
| United States | Australia | 0.4625 | 0.4642 | 0.2245 | 0.2244 | 0.3130 | 0.3114 | 0.0017 |
| Iran | New Zealand | 0.6566 | 0.6550 | 0.1919 | 0.1925 | 0.1515 | 0.1525 | 0.0016 |

### Most Affected Groups

| group | average_probability_change | maximum_probability_change |
| --- | --- | --- |
| C | 0.0018 | 0.0029 |
| I | 0.0017 | 0.0033 |
| L | 0.0016 | 0.0037 |
| B | 0.0013 | 0.0031 |
| D | 0.0012 | 0.0019 |
| G | 0.0009 | 0.0016 |
| A | 0.0007 | 0.0014 |
| H | 0.0006 | 0.0011 |
| J | 0.0005 | 0.0011 |
| K | 0.0005 | 0.0008 |
| F | 0.0004 | 0.0008 |
| E | 0.0004 | 0.0008 |

### Champion Probabilities Before Versus Challenger

| team | p_champion_baseline | p_champion_indirect | delta_p_champion |
| --- | --- | --- | --- |
| Colombia | 0.0537 | 0.0573 | 0.0036 |
| Japan | 0.0315 | 0.0281 | -0.0034 |
| Argentina | 0.2078 | 0.2109 | 0.0031 |
| Germany | 0.0308 | 0.0286 | -0.0022 |
| England | 0.0608 | 0.0630 | 0.0022 |
| Mexico | 0.0236 | 0.0215 | -0.0021 |
| Morocco | 0.0204 | 0.0183 | -0.0021 |
| Portugal | 0.0497 | 0.0477 | -0.0020 |
| France | 0.0934 | 0.0915 | -0.0019 |
| Croatia | 0.0106 | 0.0091 | -0.0015 |
| Norway | 0.0091 | 0.0104 | 0.0013 |
| Brazil | 0.0572 | 0.0583 | 0.0011 |
| Belgium | 0.0204 | 0.0214 | 0.0009 |
| Paraguay | 0.0045 | 0.0036 | -0.0009 |
| Ecuador | 0.0363 | 0.0371 | 0.0008 |

The rejected-challenger tournament comparison used 10,000 simulations while the selected baseline used 100,000. Small champion-probability deltas therefore include Monte Carlo sampling noise and should not be interpreted as precise effects.

## F. Limitations

- Tournament readiness has a small effective sample: one row per team per historical World Cup.
- Qualification difficulty and confederation strength are imperfectly controlled.
- Sparse intercontinental matches can bias internal Elo comparisons.
- Historical World Cup stage labels are incomplete, preventing honest group-stage target evaluation.
- The indirect models may overfit historical tournament patterns.
- World Cup 2026 group and bracket assumptions remain uncertain.
- No player data or betting data are used.

These are probabilistic comparisons, not certainties.
