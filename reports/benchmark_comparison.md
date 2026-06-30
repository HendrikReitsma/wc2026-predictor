# Comparison With Public GitHub World Cup Predictors

Snapshot date: 2026-06-29

This note compares the local `wc2026-predictor` group-stage forecasts with two public GitHub projects:

- `hjjbh1314/worldcup-predictor`: https://github.com/hjjbh1314/worldcup-predictor
- `Hicruben/world-cup-2026-prediction-model` / Cup26: https://github.com/Hicruben/world-cup-2026-prediction-model

## Sources Used

Local sources:

- `data/predictions/group_stage_prediction_evaluation_metrics.json`
- `data/predictions/group_stage_prediction_evaluation.csv`
- `reports/model_card.md`
- `docs/internal/evaluation.md`

External sources:

- `hjjbh1314` raw pre-tournament predictions: https://raw.githubusercontent.com/hjjbh1314/worldcup-predictor/main/docs/pretournament.json
- `hjjbh1314` raw scoreboard: https://raw.githubusercontent.com/hjjbh1314/worldcup-predictor/main/docs/scoreboard.json
- `Hicruben` README and track record: https://raw.githubusercontent.com/Hicruben/world-cup-2026-prediction-model/main/README.md

## Headline Group-Stage Results

| Model / method | Sample scored | Outcome accuracy | RPS | Brier | Log loss | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Local probabilistic model | 72 group matches | 46/72 = 63.9% | 0.153 | 0.516 | 0.878 | Uses full home/draw/away probabilities. |
| Local rounded xG score | 72 group matches | 47/72 = 65.3% | 0.194 | n/a | n/a | Deterministic Scorito-style score from rounded expected goals. |
| `hjjbh1314/worldcup-predictor` | 72 group matches | 45/72 = 62.5% | 0.157 | 0.522 | 0.874 | From published scoreboard JSON, regenerated 2026-06-29. |
| Hicruben / Cup26 | 72 group matches for accuracy | 47/72 = 65.3% | n/a group-only | n/a | n/a | README publishes 49/74 correct and RPS 0.147 including 2 knockout matches. |

The local probabilistic model and `hjjbh1314` are the cleanest metric comparison because both publish all group-stage probability metrics over the same 72 matches. The local model is slightly better on accuracy, RPS, and Brier; `hjjbh1314` is slightly better on log loss. These differences are small enough that I would treat them as practically tied.

Hicruben / Cup26 has the best published RPS number, 0.147, but that number is over 74 finished matches, not group-stage-only. Its group-stage accuracy can be counted from the README track record as 47/72, matching the local rounded-score method and one match above the local probabilistic argmax method.

## Method Comparison

| Area | Local model | `hjjbh1314/worldcup-predictor` | Hicruben / Cup26 |
| --- | --- | --- | --- |
| Core strength signal | Pre-match Elo plus rolling form and attack/defence ratings | Elo-first model, with confederation adjustment and simple probability head | Elo ratings |
| Goals / scorelines | Separate Poisson goal models, then score matrix adjusted by a seven-class margin classifier | Dixon-Coles scoreline/xG used for score display, not preferred for W/D/L | Dixon-Coles bivariate Poisson |
| W/D/L probabilities | Hybrid: Poisson score matrix plus margin-class correction | Multinomial/logistic-style Elo probability model; README says gradient boosting added little | Derived from Elo-to-expected-goals and Dixon-Coles |
| Model complexity | Highest of the three | Lowest/most transparent | Middle: simple rating model plus richer score distribution |
| Calibration posture | Selected uncalibrated because historical World Cup validation preferred it | Published notes emphasize calibration/confederation tuning | README reports reliability/ECE work, with frozen pre-match calls |
| Tournament simulation | Monte Carlo, user-requested 10,000 simulations for the later-round run | Daily regenerated predictions and tournament pages | 50,000 Monte Carlo trials |
| Odds/player data | Not used | Not central in published method | README says no bookmaker odds and no ML black box |

The useful lesson from the two GitHub repos is that simple rating-based models are still very hard to beat. `hjjbh1314` explicitly reports that Elo dominates its feature importance and that gradient boosting adds almost nothing. That supports keeping the local Elo and Poisson baselines visible, even though the local hybrid model currently edges the Elo-style public benchmark on RPS and Brier.

The Hicruben method is closer to a classic rating-to-goal model: Elo creates expected goals, Dixon-Coles reshapes low-score probabilities, and Monte Carlo handles the tournament. The local method is more feature-rich and more diagnostic, but that extra machinery only matters if it keeps winning on frozen evaluations. For this group stage, the gain is modest rather than decisive.

## Example Matches

### Germany vs Curacao

Actual result: Germany 7-1 Curacao.

| Model | Home/draw/away view | Expected goals / score | Evaluation |
| --- | --- | --- | --- |
| Local model | Germany 91.3%, draw 6.2%, Curacao 2.5% | xG 3.30-0.76, rounded score 3-1 | Correct outcome, RPS 0.004 |
| `hjjbh1314` | Germany 84.9%, draw 11.6%, Curacao 3.5% | xG 2.88-0.66, displayed score 2-0 | Correct outcome, RPS 0.012 |
| Hicruben / Cup26 | Germany 80% in track record | Score distribution not exposed in README row | Correct outcome |

All three models got the direction right. The actual 7-1 was far above the central score estimate, but that is not automatically a model failure: expected goals describe the mean of a distribution, and the most likely/rounded score is usually much lower than the high-scoring tail.

### Spain vs Cape Verde

Actual result: Spain 0-0 Cape Verde.

| Model | Favorite probability | Central score view | Evaluation |
| --- | ---: | --- | --- |
| Local model | Spain 91.8% | Strong Spain win expectation | Wrong outcome; high RPS penalty |
| `hjjbh1314` | Spain 88.7% | xG 2.15-0.50, displayed score 2-0 | Wrong outcome |
| Hicruben / Cup26 | Spain 78% | README row only | Wrong outcome |

This match is the clearest shared failure mode: heavy favorites drawing 0-0. The local model was the most confident of the three, so it was punished more heavily by RPS/log loss. That suggests the most promising improvement is not "make favorites stronger"; it is better uncertainty and draw calibration for mismatches.

## Interpretation

The local probabilistic method held up well. Against `hjjbh1314`, it scored:

- Accuracy: 63.9% vs 62.5%
- RPS: 0.153 vs 0.157
- Brier: 0.516 vs 0.522
- Log loss: 0.878 vs 0.874

That is a near tie with a slight local edge on RPS/Brier and a slight `hjjbh1314` edge on log loss.

The local rounded xG method had the best local outcome accuracy, 47/72, but its deterministic RPS was worse at 0.194. This is expected. A rounded score can be useful for a Scorito entry because the contest requires a single scoreline, but as a probability forecast it is overconfident: it assigns all probability to one outcome.

Cup26/Hicruben is the strongest public comparison on headline RPS, but the published 0.147 is not group-stage-only. Its group-stage accuracy was 47/72, equal to the local rounded-score method.

## Recommended Improvements

1. Keep the local hybrid model, but benchmark it continuously against a very simple Elo/Dixon-Coles baseline.
2. Add a favorite-confidence shrinkage or calibration experiment focused on mismatches, especially the 0-0 and 1-1 draw tail.
3. Report both probabilistic and Scorito-style metrics. RPS/log loss answer "were the probabilities good?" while rounded-score accuracy answers "was the entered score useful?"
4. Treat the public-repo differences as small-sample evidence, not proof. A 72-match World Cup group stage is informative, but one or two surprising draws can move the leaderboard.
