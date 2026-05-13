# TPC Round 6 QQQ Alpha Search

Elapsed minutes: 163.6

## Guardrails
- Structural candidates run through the shared TPC core and replay engine.
- Score-model candidates use `alpha7`, a seven-component setup score.
- Objective has exactly seven components.
- Main gate requires QQQ excellent-trade count to rise in both train and OOS.
- OOS is selection OOS; small QQQ OOS samples require fresh validation.

## Baseline
Train QQQ: trades 20, excellent 14, excellent rate 70.0%.
OOS QQQ: trades 2, excellent 0, excellent rate 0.0%.
Whole book: train net +96.90%, OOS net +3.58%.

## Train+OOS Leaders
| Candidate | Gate | Train QQQ Ex | OOS QQQ Ex | Train QQQ Trades | OOS QQQ Trades | Train Net | OOS Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alpha7_balanced_pair_ma100_030_no_di__t1r090_partial70_stop035 | True | 20 | 1 | 27 | 5 | +116.05% | +2.44% |
| alpha7_supply_pair_ma100_030_no_di__t1r090_partial70_stop035 | True | 20 | 1 | 27 | 5 | +116.05% | +2.44% |
| alpha7_balanced_pair_ma100_030_no_di__profit_floor_light | True | 20 | 1 | 27 | 5 | +117.19% | +2.04% |
| alpha7_supply_pair_ma100_030_no_di__profit_floor_light | True | 20 | 1 | 27 | 5 | +117.19% | +2.04% |
| alpha7_strict_pair_ma100_030_no_di__profit_floor_light | True | 20 | 1 | 27 | 5 | +107.34% | +2.04% |
| alpha7_balanced_pair_shorts_score12__fib_a38_72 | True | 20 | 1 | 34 | 4 | +107.95% | +4.50% |
| alpha7_supply_pair_shorts_score12__fib_a38_72 | True | 20 | 1 | 34 | 4 | +107.95% | +4.50% |
| alpha7_balanced_pair_shorts_score12__t1r110_partial65_stop040 | False | 22 | 1 | 43 | 4 | +65.69% | +6.38% |
| alpha7_strict_pair_shorts_score12__t1r110_partial65_stop040 | False | 22 | 1 | 43 | 4 | +61.28% | +6.38% |
| alpha7_supply_pair_shorts_score12__t1r110_partial65_stop040 | False | 22 | 1 | 43 | 4 | +65.69% | +6.38% |
| alpha7_strict_pair_ma100_030_no_di__t1r090_partial70_stop035 | True | 20 | 1 | 27 | 5 | +102.57% | +2.44% |
| alpha7_balanced_pair_shorts_score12__profit_floor_runner | False | 23 | 1 | 44 | 4 | +85.83% | +4.87% |
| alpha7_supply_pair_shorts_score12__profit_floor_runner | False | 23 | 1 | 44 | 4 | +85.83% | +4.87% |
| alpha7_balanced_pair_shorts_score12__t1r090_partial70_stop035 | False | 24 | 1 | 45 | 4 | +69.11% | +5.28% |
| alpha7_strict_pair_shorts_score12__t1r090_partial70_stop035 | False | 24 | 1 | 45 | 4 | +60.69% | +5.28% |
| alpha7_supply_pair_shorts_score12__t1r090_partial70_stop035 | False | 24 | 1 | 45 | 4 | +69.11% | +5.28% |
| alpha7_strict_pair_shorts_score12__profit_floor_runner | False | 23 | 1 | 44 | 4 | +73.67% | +4.87% |
| alpha7_balanced_pair_ma100_030_no_di__t1r100_partial65_stop040 | True | 19 | 1 | 26 | 5 | +120.32% | +2.53% |

## Gate Counts
- Passed QQQ alpha gate: 12
- Real-alpha evidence before all preservation gates: 12

## Recommendation
Selected candidate: `alpha7_balanced_pair_ma100_030_no_di__t1r090_partial70_stop035`.
Train QQQ: trades 27, excellent 20, excellent rate 74.1%.
OOS QQQ: trades 5, excellent 1, excellent rate 20.0%.
Whole book: train net +116.05%, OOS net +2.44%.
