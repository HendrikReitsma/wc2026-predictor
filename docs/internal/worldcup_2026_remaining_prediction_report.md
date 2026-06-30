# World Cup 2026 Remaining Tournament Prediction

- Cutoff after group stage: **2026-06-27**.
- Simulations: **10,000**.
- Scope: actual group-stage results are locked in; no knockout results are included.
- Group-score source: https://www.sbnation.com/soccer/1117513/world-cup-schedule-2026-how-to-watch-every-match-scores-and-more
- Standings cross-check source: https://www.foxsports.com/soccer/fifa-world-cup/standings
- Germany fixture cross-check source: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/teams/germany/fixtures

## Results Audit

- Group result rows: **72**.
- Unique result match IDs: **72**.
- Fixture group rows: **72**.
- Missing score rows: **0**.
- Fixture merge misses: left-only **0**, right-only **0**.
- Fixture field mismatches: **{'home_team': 0, 'away_team': 0, 'group': 0, 'city': 0, 'country': 0, 'neutral': 0}**.
- Example check: **Germany 7-1 Curaçao**, match 10 on 2026-06-14.

## Group E Check

| position | team | points | goals_for | goals_against | goal_difference | qualified |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Germany | 6 | 10 | 4 | 6 | True |
| 2 | Ivory Coast | 6 | 4 | 2 | 2 | True |
| 3 | Ecuador | 4 | 2 | 2 | 0 | True |
| 4 | Curaçao | 1 | 1 | 9 | -8 | False |

## Qualified Third-Place Teams

| third_place_rank | team | group | points | goal_difference | goals_for |
| --- | --- | --- | --- | --- | --- |
| 1 | DR Congo | K | 4 | 1 | 4 |
| 2 | Sweden | F | 4 | 0 | 7 |
| 3 | Ghana | L | 4 | 0 | 2 |
| 4 | Ecuador | E | 4 | 0 | 2 |
| 5 | Bosnia and Herzegovina | B | 4 | -1 | 5 |
| 6 | Algeria | J | 4 | -2 | 5 |
| 7 | Paraguay | D | 4 | -2 | 2 |
| 8 | Senegal | I | 3 | 2 | 8 |

## Round Of 32 Match Predictions

| match_id | home_team | away_team | expected_goals_home | expected_goals_away | p_home_win_90 | p_draw_90 | p_away_win_90 | p_home_advance | p_away_advance | most_likely_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 73 | South Africa | Canada | 0.947 | 1.541 | 0.233 | 0.253 | 0.515 | 0.324 | 0.676 | 0-1 |
| 76 | Brazil | Japan | 1.468 | 1.053 | 0.467 | 0.270 | 0.263 | 0.635 | 0.365 | 1-1 |
| 74 | Germany | Paraguay | 1.611 | 1.055 | 0.509 | 0.246 | 0.245 | 0.656 | 0.344 | 1-1 |
| 75 | Netherlands | Morocco | 1.316 | 1.225 | 0.383 | 0.280 | 0.337 | 0.532 | 0.468 | 1-1 |
| 78 | Ivory Coast | Norway | 1.277 | 1.553 | 0.322 | 0.233 | 0.445 | 0.428 | 0.572 | 1-1 |
| 77 | France | Sweden | 2.912 | 0.788 | 0.846 | 0.102 | 0.052 | 0.924 | 0.075 | 2-0 |
| 79 | Mexico | Ecuador | 1.230 | 0.942 | 0.434 | 0.278 | 0.287 | 0.595 | 0.405 | 1-0 |
| 80 | England | DR Congo | 1.882 | 0.661 | 0.679 | 0.212 | 0.109 | 0.830 | 0.171 | 1-0 |
| 82 | Belgium | Senegal | 1.785 | 1.116 | 0.538 | 0.228 | 0.233 | 0.675 | 0.325 | 1-1 |
| 81 | United States | Bosnia and Herzegovina | 2.337 | 0.841 | 0.741 | 0.130 | 0.130 | 0.837 | 0.164 | 2-0 |
| 84 | Spain | Austria | 2.041 | 0.727 | 0.693 | 0.200 | 0.106 | 0.836 | 0.164 | 1-0 |
| 83 | Portugal | Croatia | 1.641 | 1.019 | 0.520 | 0.251 | 0.229 | 0.682 | 0.318 | 1-1 |
| 85 | Switzerland | Algeria | 1.636 | 1.171 | 0.491 | 0.234 | 0.275 | 0.630 | 0.369 | 1-1 |
| 88 | Australia | Egypt | 1.294 | 1.046 | 0.426 | 0.275 | 0.298 | 0.583 | 0.417 | 1-1 |
| 86 | Argentina | Cape Verde | 2.713 | 0.557 | 0.870 | 0.103 | 0.027 | 0.951 | 0.049 | 2-0 |
| 87 | Colombia | Ghana | 2.434 | 0.601 | 0.811 | 0.136 | 0.053 | 0.910 | 0.090 | 2-0 |

## Top Champion Probabilities

| team | p_reach_r16 | p_reach_qf | p_reach_sf | p_reach_final | p_champion |
| --- | --- | --- | --- | --- | --- |
| Argentina | 0.951 | 0.817 | 0.600 | 0.429 | 0.283 |
| Spain | 0.836 | 0.583 | 0.479 | 0.307 | 0.173 |
| France | 0.924 | 0.699 | 0.507 | 0.302 | 0.163 |
| Colombia | 0.910 | 0.620 | 0.252 | 0.134 | 0.060 |
| Brazil | 0.635 | 0.474 | 0.275 | 0.120 | 0.059 |
| England | 0.830 | 0.471 | 0.264 | 0.113 | 0.054 |
| Portugal | 0.682 | 0.279 | 0.200 | 0.094 | 0.043 |
| Netherlands | 0.532 | 0.407 | 0.165 | 0.073 | 0.029 |
| Mexico | 0.595 | 0.316 | 0.157 | 0.061 | 0.023 |
| Morocco | 0.468 | 0.351 | 0.141 | 0.061 | 0.022 |
| Belgium | 0.675 | 0.449 | 0.149 | 0.055 | 0.019 |
| Japan | 0.365 | 0.241 | 0.116 | 0.042 | 0.016 |
| Germany | 0.656 | 0.205 | 0.105 | 0.042 | 0.013 |
| Ecuador | 0.405 | 0.180 | 0.079 | 0.026 | 0.009 |
| Switzerland | 0.630 | 0.248 | 0.064 | 0.024 | 0.006 |