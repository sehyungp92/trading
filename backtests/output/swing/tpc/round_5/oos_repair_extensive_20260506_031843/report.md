# TPC Round 5 OOS Repair Report

Elapsed minutes: 16.5

## Baseline
Train: net +131.82%, trades 214, avgR +1.423, $PF 1.52.
OOS: net -24.97%, trades 28, avgR -0.549, win rate 17.9%, $PF 0.30, DD 28.35%.

## OOS Weakness
- OOS loss is broad across the long book, not a single-tail-event problem.
- QQQ OOS sample has zero winning trades; GLD contributes more trades and more dollars lost.
- Shorts are comparatively resilient; blanket long removal is diagnostic, not a deployable repair because it crushes in-sample supply.
- The highest-value repair family should filter weak 4h trend quality before entry, then check whether exits/sizing merely mitigate dollars.

## Prior Ablation/Perturbation OOS Leaders
| Candidate | OOS Net | Delta | Trades | AvgR | Win |
| --- | ---: | ---: | ---: | ---: | ---: |
| probe_all_longs_off | +3.35% | +28.32% | 9 | +0.234 | 44.4% |
| remove::all.max_position_notional_pct | -1.03% | +23.94% | 28 | -0.559 | 17.9% |
| di_min_adx_18 | -3.88% | +21.10% | 14 | -0.027 | 35.7% |
| di_min_adx_15 | -5.75% | +19.22% | 15 | -0.095 | 33.3% |
| di_min_adx_20 | -5.87% | +19.10% | 11 | -0.205 | 27.3% |
| di_alignment | -6.45% | +18.53% | 16 | -0.160 | 31.2% |
| di_ma50_slope_0.02 | -6.45% | +18.53% | 16 | -0.160 | 31.2% |
| di_ma50_slope_0.04 | -6.45% | +18.53% | 16 | -0.160 | 31.2% |

## New Targeted OOS Leaders
| Candidate | OOS Net | Delta | Trades | AvgR | Win |
| --- | ---: | ---: | ---: | ---: | ---: |
| trend_ma100_0.03_di_adx12 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |
| trend_ma100_0.03_di_adx15 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |
| trend_ma100_0.03_di_adx18 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |
| trend_ma100_0.04_di_adx12 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |
| trend_ma100_0.04_di_adx15 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |
| trend_ma100_0.04_di_adx18 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |
| trend_ma100_0.05_di_adx12 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |
| trend_ma100_0.05_di_adx15 | -0.44% | +24.54% | 12 | +0.143 | 41.7% |

## Train+OOS Validation Leaders
| Candidate | Gate | Train Net | Train Trades | OOS Net | OOS Trades | OOS AvgR | OOS DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| probe_all_longs_off | False | +8.50% | 78 | +3.35% | 9 | +0.234 | 7.01% |
| trend_ma100_0.03_di_adx12 | True | +78.79% | 127 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.04_di_adx12 | True | +78.79% | 127 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.05_di_adx12 | True | +78.79% | 127 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.06_di_adx12 | True | +78.79% | 127 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.03_di_adx15 | True | +80.37% | 108 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.04_di_adx15 | True | +80.37% | 108 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.05_di_adx15 | True | +80.37% | 108 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.06_di_adx15 | True | +80.37% | 108 | -0.44% | 12 | +0.143 | 7.01% |
| trend_ma100_0.03_di_adx18 | False | +47.60% | 79 | -0.44% | 12 | +0.143 | 7.01% |

## Recommended Repair
Selected candidate: `trend_ma100_0.03_di_adx12`.
Train: net +78.79%, trades 127, avgR +2.235.
OOS: net -0.44%, trades 12, avgR +0.143, win 41.7%.
This is a selection-OOS repair, so it should be promoted only after a fresh holdout or paper-trade validation.
