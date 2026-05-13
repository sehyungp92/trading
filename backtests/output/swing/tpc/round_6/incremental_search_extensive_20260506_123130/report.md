# TPC Round 6 Incremental Mutation Search

Elapsed minutes: 80.1

## Baseline
Train: net +96.90%, trades 126, avgR +2.353, $PF 1.77, DD 13.75%.
OOS: net +3.58%, trades 10, avgR +0.383, win 50.0%, $PF 1.36, DD 4.45%.

## OOS-Only Leaders
| Candidate | Stage | OOS Net | Delta | Trades | AvgR | $PF | DD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combo_gld_no8_session_qqq_afternoon_only | anchor_x_session | +12.50% | +8.92% | 6 | +1.192 | 9.28 | 2.05% |
| combo_gld_no8_signal_all_confirm_max3 | anchor_x_signal | +12.17% | +8.59% | 6 | +1.312 | 9.06 | 2.05% |
| combo_value_hits2_session_qqq_afternoon_only | anchor_x_session | +10.65% | +7.08% | 7 | +0.887 | 4.82 | 3.46% |
| combo_gld_value_hits2_session_qqq_afternoon_only | anchor_x_session | +10.65% | +7.08% | 7 | +0.887 | 4.82 | 3.46% |
| combo_gld_no8_signal_qqq_value_hits2 | anchor_x_signal | +9.70% | +6.12% | 7 | +0.874 | 3.40 | 4.45% |
| combo_gld_no8_session_avoid_1030_1200 | anchor_x_session | +9.61% | +6.04% | 7 | +0.877 | 3.26 | 4.33% |
| combo_gld_no8_session_avoid_1045_1145 | anchor_x_session | +9.61% | +6.04% | 7 | +0.877 | 3.26 | 4.33% |
| combo_value_hits2_exit_all_t1_partial_65 | anchor_x_exit | +9.34% | +5.76% | 8 | +0.745 | 2.76 | 4.07% |
| combo_value_hits2_exit_gld_t1_partial_65 | anchor_x_exit | +9.34% | +5.76% | 8 | +0.745 | 2.76 | 4.07% |
| combo_gld_no8_signal_all_value_hits2 | anchor_x_signal | +9.33% | +5.75% | 7 | +0.891 | 3.30 | 4.45% |
| combo_value_hits2_session_gld_regular_hours | anchor_x_session | +9.33% | +5.75% | 7 | +0.891 | 3.30 | 4.45% |
| combo_value_hits2_signal_gld_room35 | anchor_x_signal | +9.33% | +5.75% | 7 | +0.891 | 3.30 | 4.45% |

## Train+OOS Validation Leaders
| Candidate | Balanced | Both Ret | Freq | Train Net | Train Trades | OOS Net | OOS Trades | OOS AvgR | OOS $PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combo_gld_no8_session_avoid_1030_1200 | False | True | False | +111.26% | 79 | +9.61% | 7 | +0.877 | 3.26 |
| combo_gld_no8_session_avoid_1045_1145 | False | True | False | +100.81% | 79 | +9.61% | 7 | +0.877 | 3.26 |
| combo_gld_no8_session_qqq_afternoon_only | False | True | False | +97.28% | 75 | +12.50% | 6 | +1.192 | 9.28 |
| session_qqq_afternoon_only | True | False | True | +92.81% | 115 | +9.00% | 8 | +0.735 | 2.88 |
| combo_qqq_typeb_supply_session_qqq_afternoon_only | True | False | True | +92.81% | 115 | +9.00% | 8 | +0.735 | 2.88 |
| combo_gld_no8_signal_qqq_value_hits2 | False | False | False | +91.46% | 85 | +9.70% | 7 | +0.874 | 3.40 |
| combo_source17_session_qqq_afternoon_only | True | False | True | +90.56% | 115 | +9.00% | 8 | +0.735 | 2.88 |
| combo_gld_no8_signal_all_confirm_max3 | False | False | False | +73.36% | 66 | +12.17% | 6 | +1.312 | 9.06 |
| combo_t1stop040_session_qqq_afternoon_only | True | False | True | +87.35% | 117 | +9.15% | 8 | +0.746 | 2.91 |
| combo_t1stop035_session_qqq_afternoon_only | False | False | True | +86.98% | 117 | +9.08% | 8 | +0.741 | 2.89 |
| combo_addon_off_session_qqq_afternoon_only | True | False | True | +88.24% | 115 | +9.00% | 8 | +0.735 | 2.88 |
| combo_value_hits2_session_gld_regular_hours | False | False | False | +76.05% | 73 | +9.33% | 7 | +0.891 | 3.30 |
| combo_gld_no8_signal_all_value_hits2 | False | False | False | +69.79% | 74 | +9.33% | 7 | +0.891 | 3.30 |
| combo_t1stop040_signal_all_confirm_max3 | False | False | False | +59.84% | 107 | +9.15% | 8 | +0.746 | 2.91 |
| combo_t1stop035_signal_all_confirm_max3 | False | False | False | +59.42% | 107 | +9.08% | 8 | +0.741 | 2.89 |

## Gate Counts
- Balanced OOS uplift with material in-sample preservation: 8
- Improves both train and OOS headline return: 14
- Improves OOS while preserving frequency and at least 85% train net: 17

## Recommendation
Selected candidate: `session_qqq_afternoon_only`.
Train: net +92.81%, trades 115, $PF 1.84, DD 13.28%.
OOS: net +9.00%, trades 8, avgR +0.735, $PF 2.88, DD 3.46%.
Treat this as selection-OOS evidence. A fresh holdout or forward paper window is still required before promotion.
