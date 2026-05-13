# TPC Round 8 OOS Candidate Check

Elapsed minutes: 16.7
Split: train <= `2025-11-01`, OOS after split with warmup at the cutoff.

## Baseline
Train: net +133.01%, avgR +0.681, trades 127, trades/month 2.30, DD 13.68%, PF$ 2.08.
OOS: net +11.00%, avgR +0.661, trades 10, trades/month 1.69, DD 3.46%, PF$ 3.28, low-MFE 20.0%.

## Top OOS Net
| Candidate | Family | Train Net | OOS Net | dOOS | OOS Trades | OOS AvgR | OOS DD | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| type_a_value_hits2 | discrimination | +121.79% | +12.82% | +1.82% | 9 | +0.777 | 3.46% | False |
| top3_pb30_qqq_no_mfe | combo | +160.59% | +11.32% | +0.32% | 10 | +0.679 | 3.46% | True |
| top4_pb30_qqq_no_mfe_ext | combo | +160.59% | +11.32% | +0.32% | 10 | +0.679 | 3.46% | True |
| pb30_5_18_plus_no_mfe | combo | +149.89% | +11.32% | +0.32% | 10 | +0.679 | 3.46% | True |
| qqq_t1_100_plus_no_mfe | combo | +147.13% | +11.32% | +0.32% | 10 | +0.679 | 3.46% | True |
| disable_mfe_giveback | management | +137.30% | +11.32% | +0.32% | 10 | +0.679 | 3.46% | True |
| top3_pb30_qqq_trail6 | combo | +161.80% | +11.00% | +0.00% | 10 | +0.661 | 3.46% | True |
| pb30_5_18_plus_qqq_t1_100 | combo | +155.88% | +11.00% | +0.00% | 10 | +0.661 | 3.46% | True |
| pb30_5_18_plus_trail6 | combo | +150.60% | +11.00% | +0.00% | 10 | +0.661 | 3.46% | True |
| qqq_t1_100_plus_trail6 | combo | +148.20% | +11.00% | +0.00% | 10 | +0.661 | 3.46% | True |
| pb30_duration_5_18 | signal | +145.38% | +11.00% | +0.00% | 10 | +0.661 | 3.46% | True |
| qqq_t1_100 | management | +142.67% | +11.00% | +0.00% | 10 | +0.661 | 3.46% | True |

## Top Blended Objective
| Candidate | Family | Objective | Train Net | OOS Net | OOS Trades | OOS AvgR | OOS DD | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top3_pb30_qqq_trail6 | combo | 0.864 | +161.80% | +11.00% | 10 | +0.661 | 3.46% | True |
| top3_pb30_qqq_no_mfe | combo | 0.861 | +160.59% | +11.32% | 10 | +0.679 | 3.46% | True |
| top4_pb30_qqq_no_mfe_ext | combo | 0.861 | +160.59% | +11.32% | 10 | +0.679 | 3.46% | True |
| pb30_5_18_plus_qqq_t1_100 | combo | 0.856 | +155.88% | +11.00% | 10 | +0.661 | 3.46% | True |
| pb30_5_18_plus_trail6 | combo | 0.852 | +150.60% | +11.00% | 10 | +0.661 | 3.46% | True |
| pb30_5_18_plus_no_mfe | combo | 0.850 | +149.89% | +11.32% | 10 | +0.679 | 3.46% | True |
| qqq_t1_100_plus_trail6 | combo | 0.848 | +148.20% | +11.00% | 10 | +0.661 | 3.46% | True |
| qqq_t1_100_plus_no_mfe | combo | 0.845 | +147.13% | +11.32% | 10 | +0.679 | 3.46% | True |
| pb30_duration_5_18 | signal | 0.844 | +145.38% | +11.00% | 10 | +0.661 | 3.46% | True |
| disable_structure_trail | risk_check | 0.844 | +208.86% | +9.40% | 10 | +0.571 | 3.46% | False |
| qqq_t1_100 | management | 0.840 | +142.67% | +11.00% | 10 | +0.661 | 3.46% | True |
| trail_after_t1_6 | management | 0.837 | +137.90% | +11.00% | 10 | +0.661 | 3.46% | True |

## Robust-Gate Passes
| Candidate | Train Net | OOS Net | OOS Trades | OOS AvgR | OOS DD | Thesis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| top3_pb30_qqq_no_mfe | +160.59% | +11.32% | 10 | +0.679 | 3.46% | Best train stack: PB30 5-18, QQQ T1 1.0, no MFE giveback. |
| top4_pb30_qqq_no_mfe_ext | +160.59% | +11.32% | 10 | +0.679 | 3.46% | Best train stack plus QQQ extension control. |
| pb30_5_18_plus_no_mfe | +149.89% | +11.32% | 10 | +0.679 | 3.46% | PB30 duration plus no MFE giveback. |
| qqq_t1_100_plus_no_mfe | +147.13% | +11.32% | 10 | +0.679 | 3.46% | QQQ target plus no MFE giveback. |
| disable_mfe_giveback | +137.30% | +11.32% | 10 | +0.679 | 3.46% | Remove MFE giveback, which improved train without DD increase. |
| top3_pb30_qqq_trail6 | +161.80% | +11.00% | 10 | +0.661 | 3.46% | Strong train stack using tighter trail instead of no MFE giveback. |
| pb30_5_18_plus_qqq_t1_100 | +155.88% | +11.00% | 10 | +0.661 | 3.46% | Top two clean in-sample leads. |
| pb30_5_18_plus_trail6 | +150.60% | +11.00% | 10 | +0.661 | 3.46% | PB30 duration plus tighter post-T1 trail. |
| qqq_t1_100_plus_trail6 | +148.20% | +11.00% | 10 | +0.661 | 3.46% | QQQ target plus tighter post-T1 trail. |
| pb30_duration_5_18 | +145.38% | +11.00% | 10 | +0.661 | 3.46% | Best clean signal-duration lead. |
| qqq_t1_100 | +142.67% | +11.00% | 10 | +0.661 | 3.46% | Best clean QQQ management lead. |
| trail_after_t1_6 | +137.90% | +11.00% | 10 | +0.661 | 3.46% | Slightly tighter post-T1 structure/VWAP trail. |
| pb30_value_touch_limit | +135.22% | +11.00% | 10 | +0.661 | 3.46% | PB30 EMA20 value-touch limit route. |
| max_extension_qqq_175 | +135.51% | +11.00% | 10 | +0.661 | 3.46% | Small clean QQQ extension filter. |

## Recommendation
Selected: `top3_pb30_qqq_no_mfe`.
Train: net +160.59%, avgR +0.733, trades 126, DD 13.85%.
OOS: net +11.32%, avgR +0.679, trades 10, DD 3.46%.
OOS is now selection data for these candidates, so the selected config still needs a fresh validation slice or paper-trade pass before promotion.
