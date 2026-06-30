# World Cup 2026 Prediction Summary

- Training cutoff: **2026-06-10**
- Selected model: **margin_class_classifier__uncalibrated**
- Selected model feature set: **attack_defence_poisson**
- Best classifier challenger feature group: **core_raw_form**
- Selected weighting: **aggressive**
- Production time decay: **disabled**
- Rating update: **standard_elo**, K scale **1.2**
- Draw correction / calibration: **similar_strength / uncalibrated**
- Indirect correction: **baseline**; best challenger **trend_small**
- Data sources: martj42 international results; FIFA 2026 official schedule and regulations.

## Top 20 Champion Probabilities

| team | p_champion |
| --- | --- |
| Argentina | 0.208 |
| Spain | 0.197 |
| France | 0.093 |
| England | 0.061 |
| Brazil | 0.057 |
| Colombia | 0.054 |
| Portugal | 0.050 |
| Ecuador | 0.036 |
| Japan | 0.031 |
| Germany | 0.031 |
| Netherlands | 0.026 |
| Mexico | 0.024 |
| Belgium | 0.020 |
| Morocco | 0.020 |
| Turkey | 0.014 |
| Switzerland | 0.012 |
| Uruguay | 0.012 |
| Croatia | 0.011 |
| Norway | 0.009 |
| Canada | 0.006 |

## Top 10 Group Winners

| team | p_group_1st |
| --- | --- |
| Spain | 0.795 |
| Argentina | 0.739 |
| Mexico | 0.638 |
| France | 0.632 |
| Switzerland | 0.619 |
| England | 0.618 |
| Brazil | 0.573 |
| Belgium | 0.524 |
| Turkey | 0.469 |
| Japan | 0.463 |

## Top 10 Most Likely Finalists

| team | p_reach_final |
| --- | --- |
| Argentina | 0.308 |
| Spain | 0.308 |
| France | 0.173 |
| England | 0.122 |
| Brazil | 0.115 |
| Colombia | 0.113 |
| Portugal | 0.105 |
| Ecuador | 0.083 |
| Germany | 0.072 |
| Japan | 0.070 |

## Biggest Uncertainty Teams

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

## Most Evenly Matched Group Games

| home_team | away_team | expected_goals_home | expected_goals_away | p_home_win | p_draw | p_away_win |
| --- | --- | --- | --- | --- | --- | --- |
| Ecuador | Germany | 1.170 | 1.129 | 0.363 | 0.297 | 0.340 |
| Netherlands | Japan | 1.180 | 1.204 | 0.352 | 0.285 | 0.363 |
| Algeria | Austria | 1.221 | 1.191 | 0.367 | 0.283 | 0.351 |
| Colombia | Portugal | 1.286 | 1.285 | 0.367 | 0.267 | 0.366 |
| Egypt | Iran | 1.038 | 1.243 | 0.305 | 0.287 | 0.407 |
| Paraguay | Australia | 1.291 | 1.131 | 0.409 | 0.262 | 0.329 |
| DR Congo | Uzbekistan | 1.031 | 1.256 | 0.309 | 0.269 | 0.421 |
| Brazil | Morocco | 1.375 | 1.076 | 0.439 | 0.272 | 0.289 |
| Australia | Turkey | 1.127 | 1.431 | 0.300 | 0.255 | 0.445 |
| United States | Paraguay | 1.379 | 1.187 | 0.435 | 0.228 | 0.337 |

## Most Likely Away-Side Upsets

| home_team | away_team | expected_goals_home | expected_goals_away | p_home_win | p_draw | p_away_win |
| --- | --- | --- | --- | --- | --- | --- |
| Jordan | Argentina | 0.694 | 2.732 | 0.039 | 0.114 | 0.847 |
| Qatar | Switzerland | 0.726 | 2.816 | 0.055 | 0.097 | 0.848 |
| Curaçao | Ivory Coast | 0.775 | 2.380 | 0.100 | 0.142 | 0.758 |
| Panama | England | 0.739 | 2.183 | 0.103 | 0.168 | 0.729 |
| New Zealand | Belgium | 0.871 | 2.340 | 0.114 | 0.162 | 0.724 |
| Tunisia | Netherlands | 0.769 | 2.132 | 0.117 | 0.174 | 0.709 |
| Tunisia | Japan | 0.727 | 2.049 | 0.114 | 0.186 | 0.701 |
| Saudi Arabia | Uruguay | 0.680 | 1.897 | 0.125 | 0.189 | 0.687 |
| Scotland | Brazil | 0.874 | 2.167 | 0.132 | 0.178 | 0.690 |
| Uzbekistan | Colombia | 0.755 | 1.978 | 0.128 | 0.192 | 0.680 |

## Fixture Audit

- fixture_rows: 104
- group_matches: 72
- knockout_matches: 32
- recognized_group_teams: 48
- unresolved_group_teams: 0
- unresolved_knockout_slots: 0
- unparseable_dates: 0
- suspicious_host_neutral_flags: 0

## Known Limitations

- Historical results do not include reliable stage labels for all competitions.
- No player availability, injuries, betting odds, or squad-strength inputs are used.
- Monte Carlo probabilities have sampling error and are not claims of certainty.