# TPC Round 5 OOS Repair Report

Elapsed minutes: 21.0

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
| combo_ma100_006_di_no11_valuehits2 | +7.91% | +32.89% | 8 | +0.648 | 62.5% |
| combo_ma100_006_di_adx15_no11_valuehits2 | +7.91% | +32.89% | 8 | +0.648 | 62.5% |
| combo_ma100_004_di_adx15_no11_valuehits2 | +7.91% | +32.89% | 8 | +0.648 | 62.5% |
| combo_ma100_006_di_no11_gld_no8 | +6.90% | +31.87% | 8 | +0.638 | 62.5% |
| combo_ma100_006_di_adx15_no11_gld_no8 | +6.90% | +31.87% | 8 | +0.638 | 62.5% |
| combo_ma100_004_di_adx15_no11_gld_no8 | +6.90% | +31.87% | 8 | +0.638 | 62.5% |
| combo_ma100_006_di_no11_t1stop04 | +3.72% | +28.70% | 10 | +0.392 | 50.0% |
| combo_ma100_006_di_adx15_no11_t1stop04 | +3.72% | +28.70% | 10 | +0.392 | 50.0% |

## Train+OOS Validation Leaders
| Candidate | Gate | Train Net | Train Trades | OOS Net | OOS Trades | OOS AvgR | OOS DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combo_ma100_006_di_no11 | True | +96.90% | 126 | +3.58% | 10 | +0.383 | 4.45% |
| combo_ma100_006_di_no11_t1stop04 | True | +91.53% | 128 | +3.72% | 10 | +0.392 | 4.45% |
| combo_ma100_006_di_no11_addon_off | True | +92.94% | 126 | +3.58% | 10 | +0.383 | 4.45% |
| combo_ma100_006_di_no11_gld_no8 | False | +101.48% | 86 | +6.90% | 8 | +0.638 | 4.45% |
| combo_ma100_006_di_no11_risk020 | True | +86.33% | 126 | +3.02% | 10 | +0.384 | 3.87% |
| combo_ma100_006_di_adx15_no11 | True | +81.62% | 103 | +3.58% | 10 | +0.383 | 4.45% |
| combo_ma100_004_di_adx15_no11 | True | +81.62% | 103 | +3.58% | 10 | +0.383 | 4.45% |
| combo_ma100_006_di_adx15_no11_gld_no8 | False | +93.32% | 72 | +6.90% | 8 | +0.638 | 4.45% |
| combo_ma100_004_di_adx15_no11_gld_no8 | False | +93.32% | 72 | +6.90% | 8 | +0.638 | 4.45% |
| combo_ma100_006_di_adx15_no11_t1stop04 | True | +76.87% | 105 | +3.72% | 10 | +0.392 | 4.45% |

## Recommended Repair
Selected candidate: `combo_ma100_006_di_no11`.
Train: net +96.90%, trades 126, avgR +2.353.
OOS: net +3.58%, trades 10, avgR +0.383, win 50.0%.
This is a selection-OOS repair, so it should be promoted only after a fresh holdout or paper-trade validation.
