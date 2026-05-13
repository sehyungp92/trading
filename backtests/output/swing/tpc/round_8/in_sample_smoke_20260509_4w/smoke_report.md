# TPC Round 8 In-Sample Smoke Tests

Output: `C:\Users\sehyu\Documents\Other\Projects\trading\backtests\output\swing\tpc\round_8\in_sample_smoke_20260509_4w`
Train-only window end: `2025-11-01`
Baseline: net +133.01%, avgR +0.681, totalR +86.47, trades 127, trades/month 2.30, PF$ 2.08, DD 13.68%, low-MFE losses 33.1%, MFE capture 33.8%.
Variants tested: 127. Smoke-promotable guardrail passes: 52.

## Best By Category
| Category | Variant | Score | dScore | dNet | dAvgR | Trades | DD | Low-MFE | Reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ablation | disable_mfe_giveback | 66.45 | +1.26 | +4.29% | +0.017 | 127 | 13.68% | 33.1% | - |
| combo | strict_discrimination_keep_frequency | 65.19 | +0.00 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | - |
| discrimination | type_a_value_hits2 | 65.93 | +0.74 | -11.22% | +0.078 | 97 | 10.81% | 33.0% | - |
| entry | entry_limit_012 | 65.19 | +0.00 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | - |
| followup | top3_pb30_qqq_no_mfe | 70.50 | +5.31 | +27.58% | +0.052 | 126 | 13.85% | 32.5% | - |
| management | qqq_t1_100 | 66.38 | +1.18 | +9.66% | +0.017 | 126 | 13.68% | 33.3% | - |
| signal | pb30_duration_5_18 | 68.58 | +3.39 | +12.37% | +0.017 | 127 | 13.85% | 32.3% | - |

## Top Round-8 Score
| Variant | Category | Score | dScore | dNet | dAvgR | Trades | DD | Low-MFE | Reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| top3_pb30_qqq_no_mfe | followup | 70.50 | +5.31 | +27.58% | +0.052 | 126 | 13.85% | 32.5% | - |
| top4_pb30_qqq_no_mfe_ext | followup | 70.50 | +5.31 | +27.58% | +0.052 | 126 | 13.85% | 32.5% | - |
| pb30_5_18_plus_no_mfe_giveback | followup | 69.88 | +4.68 | +16.88% | +0.034 | 127 | 13.85% | 32.3% | - |
| top3_pb30_qqq_trail6 | followup | 69.87 | +4.68 | +28.79% | +0.044 | 127 | 13.85% | 33.1% | - |
| pb30_5_18_plus_qqq_t1_100 | followup | 69.48 | +4.29 | +22.87% | +0.035 | 126 | 13.85% | 32.5% | - |
| pb30_5_18_plus_trail6 | followup | 69.19 | +4.00 | +17.59% | +0.025 | 128 | 13.85% | 32.8% | - |
| pb30_5_18_plus_qqq_ext175 | followup | 68.58 | +3.39 | +12.37% | +0.017 | 127 | 13.85% | 32.3% | - |
| pb30_value_touch_limit_plus_duration | followup | 68.58 | +3.39 | +12.37% | +0.017 | 127 | 13.85% | 32.3% | - |
| pb30_duration_5_18 | signal | 68.58 | +3.39 | +12.37% | +0.017 | 127 | 13.85% | 32.3% | - |
| qqq_t1_100_plus_no_mfe_giveback | followup | 67.64 | +2.45 | +14.12% | +0.034 | 126 | 13.68% | 33.3% | - |
| qqq_t1_100_plus_trail6 | followup | 67.17 | +1.97 | +15.19% | +0.026 | 127 | 13.63% | 33.9% | - |
| disable_mfe_giveback | ablation | 66.45 | +1.26 | +4.29% | +0.017 | 127 | 13.68% | 33.1% | - |

## Top Smoke Objective Delta
| Variant | Category | dObj | dNet | dAvgR | Trades | DD | Low-MFE | Reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_trail_qqq_t1_100 | followup | +105.75 | +86.05% | +1.438 | 107 | 23.56% | 29.9% | max_dd_pct |
| no_trail_mfe175 | followup | +98.13 | +77.92% | +1.390 | 109 | 22.68% | 29.4% | max_dd_pct |
| disable_structure_trail | ablation | +96.45 | +75.85% | +1.408 | 108 | 22.68% | 29.6% | max_dd_pct |
| no_trail_t1_stop050 | followup | +86.63 | +66.76% | +1.374 | 109 | 22.61% | 30.3% | max_dd_pct |
| no_trail_profit_floor | followup | +80.31 | +61.91% | +1.234 | 113 | 21.61% | 30.1% | max_dd_pct |
| top3_pb30_qqq_trail6 | followup | +29.65 | +28.79% | +0.044 | 127 | 13.85% | 33.1% | - |
| top3_pb30_qqq_no_mfe | followup | +28.38 | +27.58% | +0.052 | 126 | 13.85% | 32.5% | - |
| top4_pb30_qqq_no_mfe_ext | followup | +28.38 | +27.58% | +0.052 | 126 | 13.85% | 32.5% | - |
| pb30_5_18_plus_qqq_t1_100 | followup | +23.23 | +22.87% | +0.035 | 126 | 13.85% | 32.5% | - |
| pb30_5_18_plus_trail6 | followup | +18.23 | +17.59% | +0.025 | 128 | 13.85% | 32.8% | - |
| pb30_5_18_plus_no_mfe_giveback | followup | +17.49 | +16.88% | +0.034 | 127 | 13.85% | 32.3% | - |
| qqq_t1_100_plus_trail6 | followup | +15.92 | +15.19% | +0.026 | 127 | 13.63% | 33.9% | - |

## Largest Net Return Deltas
| Variant | Category | dNet | Net | AvgR | Trades | DD | Low-MFE | Reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_trail_qqq_t1_100 | followup | +86.05% | +219.06% | +2.119 | 107 | 23.56% | 29.9% | max_dd_pct |
| no_trail_mfe175 | followup | +77.92% | +210.93% | +2.071 | 109 | 22.68% | 29.4% | max_dd_pct |
| disable_structure_trail | ablation | +75.85% | +208.86% | +2.089 | 108 | 22.68% | 29.6% | max_dd_pct |
| no_trail_t1_stop050 | followup | +66.76% | +199.77% | +2.055 | 109 | 22.61% | 30.3% | max_dd_pct |
| no_trail_profit_floor | followup | +61.91% | +194.92% | +1.915 | 113 | 21.61% | 30.1% | max_dd_pct |
| top3_pb30_qqq_trail6 | followup | +28.79% | +161.80% | +0.725 | 127 | 13.85% | 33.1% | - |
| top3_pb30_qqq_no_mfe | followup | +27.58% | +160.59% | +0.733 | 126 | 13.85% | 32.5% | - |
| top4_pb30_qqq_no_mfe_ext | followup | +27.58% | +160.59% | +0.733 | 126 | 13.85% | 32.5% | - |
| pb30_5_18_plus_qqq_t1_100 | followup | +22.87% | +155.88% | +0.716 | 126 | 13.85% | 32.5% | - |
| pb30_5_18_plus_trail6 | followup | +17.59% | +150.60% | +0.706 | 128 | 13.85% | 32.8% | - |

## Smoke-Promotable Guardrail Passes
| Variant | Category | Score | dNet | dAvgR | Trades | DD | Low-MFE | Thesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| top3_pb30_qqq_no_mfe | followup | 70.50 | +27.58% | +0.052 | 126 | 13.85% | 32.5% | Stack the three cleanest first-pass leads. |
| top4_pb30_qqq_no_mfe_ext | followup | 70.50 | +27.58% | +0.052 | 126 | 13.85% | 32.5% | Stack clean leads while adding QQQ extension control. |
| pb30_5_18_plus_no_mfe_giveback | followup | 69.88 | +16.88% | +0.034 | 127 | 13.85% | 32.3% | Combine PB30 duration with removal of MFE giveback. |
| top3_pb30_qqq_trail6 | followup | 69.87 | +28.79% | +0.044 | 127 | 13.85% | 33.1% | Stack PB30 duration, QQQ target, and tighter trail. |
| pb30_5_18_plus_qqq_t1_100 | followup | 69.48 | +22.87% | +0.035 | 126 | 13.85% | 32.5% | Combine the best signal-duration and QQQ target leads. |
| pb30_5_18_plus_trail6 | followup | 69.19 | +17.59% | +0.025 | 128 | 13.85% | 32.8% | Combine PB30 duration with slightly tighter post-T1 trail. |
| pb30_5_18_plus_qqq_ext175 | followup | 68.58 | +12.37% | +0.017 | 127 | 13.85% | 32.3% | Combine PB30 duration with QQQ extension rejection. |
| pb30_value_touch_limit_plus_duration | followup | 68.58 | +12.37% | +0.017 | 127 | 13.85% | 32.3% | Check if value-touch limit adds anything once PB30 duration is improved. |
| pb30_duration_5_18 | signal | 68.58 | +12.37% | +0.017 | 127 | 13.85% | 32.3% | Shift 30m pullback duration shorter. |
| qqq_t1_100_plus_no_mfe_giveback | followup | 67.64 | +14.12% | +0.034 | 126 | 13.68% | 33.3% | Combine QQQ target lead with no MFE giveback. |
| qqq_t1_100_plus_trail6 | followup | 67.17 | +15.19% | +0.026 | 127 | 13.63% | 33.9% | Combine QQQ target lead with tighter post-T1 trail. |
| disable_mfe_giveback | ablation | 66.45 | +4.29% | +0.017 | 127 | 13.68% | 33.1% | Measure exit-giveback contribution. |
| qqq_t1_100 | management | 66.38 | +9.66% | +0.017 | 126 | 13.68% | 33.3% | Delay QQQ T1 modestly. |
| trail_after_t1_6 | management | 65.86 | +4.89% | +0.008 | 128 | 13.63% | 33.6% | Slightly tighten structure/VWAP trail. |
| t2_300 | management | 65.76 | +3.36% | +0.004 | 127 | 13.78% | 33.1% | Stress train-only larger target. |
| max_extension_qqq_175 | discrimination | 65.72 | +2.50% | +0.007 | 125 | 13.68% | 32.8% | Reject extended QQQ entries. |
| qqq_partial_060 | management | 65.59 | +1.97% | +0.003 | 127 | 13.77% | 33.1% | Let QQQ runners keep more size. |
| t1_stop_025 | management | 65.59 | +1.40% | +0.006 | 127 | 13.68% | 33.1% | Reduce post-T1 stop lock. |
| t2_250 | management | 65.31 | +1.53% | +0.001 | 127 | 13.78% | 33.1% | Extend T2 if MFE supports it. |
| t2_175 | management | 65.31 | +0.54% | +0.000 | 127 | 13.68% | 33.1% | Take T2 earlier to increase conversion. |
| t2_225 | management | 65.24 | +0.63% | +0.002 | 127 | 13.78% | 33.1% | Slightly extend T2. |
| mfe_giveback_after_t1_only | management | 65.20 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Avoid premature giveback flattening before T1. |
| disable_type_c_reentry | ablation | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Check whether real re-entry supply is carrying robust expectancy. |
| disable_qqq_type_b | ablation | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Test QQQ shallow Type-B contribution. |
| disable_stall_exit | ablation | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Check whether stall exits improve realised R. |
| disable_time_stop | ablation | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Check whether the low-MFE time stop is protective or premature. |
| strict_discrimination_keep_frequency | combo | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Target GLD/type-C false positives while preserving QQQ supply. |
| score_gld_plus1 | discrimination | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Target GLD false positives. |
| score_gld_plus2 | discrimination | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Stress stricter GLD setup scoring. |
| score_qqq_plus1 | discrimination | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Check whether QQQ supply can be cleaned without starving it. |
| all_confirm2 | discrimination | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Require two confirmations globally. |
| ma100_slope_all_008 | discrimination | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Demand stronger 4h slow MA slope. |
| ma50_slope_all_004 | discrimination | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Add 4h fast MA slope filter. |
| entry_limit_012 | entry | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Relax stop-limit slippage cap. |
| entry_limit_020 | entry | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Stress wider stop-limit cap. |
| max_hold_48 | management | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Give slow starters more time. |
| stall_exit_64 | management | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Let stalled winners breathe longer. |
| pb30_confirm2 | signal | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Demand more 15m proof on 30m pullbacks. |
| pb30_ema20_touch_context | signal | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Qualify 30m pullbacks by EMA20 value touch. |
| pb30_value_touch_market | signal | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Add a 30m EMA20 value-touch entry route. |
| type_c_aplus_only | signal | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Check if Type-C needs higher regime quality. |
| second_entry_score14 | signal | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Broaden second-entry supply modestly. |
| second_entry_score16 | signal | 65.19 | +0.00% | +0.000 | 127 | 13.68% | 33.1% | Tighten second-entry source quality. |
| mfe_giveback_175_04_05 | management | 65.03 | +0.28% | -0.007 | 128 | 13.68% | 32.8% | Tighten giveback trigger. |
| qqq_t1_080 | management | 64.94 | -0.65% | -0.001 | 127 | 13.68% | 33.1% | Improve QQQ first-target conversion. |
| disable_pb30_additive | ablation | 64.79 | -0.29% | +0.001 | 123 | 13.68% | 32.5% | Measure whether the round-8 30m additive lane is genuinely additive. |
| pb30_ema20_reclaim_context | signal | 64.79 | -0.29% | +0.001 | 123 | 13.68% | 32.5% | Require 30m EMA20 reclaim geometry. |
| qqq_partial_080 | management | 64.79 | -1.99% | -0.003 | 127 | 13.59% | 33.1% | Realise more QQQ size at T1. |
| addon_cautious | management | 64.66 | +2.66% | -0.018 | 127 | 13.81% | 33.1% | Test small adds only after proven continuation. |
| addon_score15 | management | 64.55 | +3.53% | -0.023 | 127 | 13.86% | 33.1% | Broaden add-ons but keep confirmation holds. |
| daily_room_all_30 | discrimination | 64.50 | -0.23% | +0.021 | 124 | 13.68% | 33.1% | Stress daily-room filter globally. |
| gld_t1_120 | management | 64.45 | -0.16% | -0.006 | 127 | 13.68% | 33.1% | Delay GLD T1 modestly. |

## Guardrails
- This report is in-sample only by request; it is useful for smoke-screening mechanisms, not promotion.
- The smoke-promotable flag requires no hard reject, preserved trade count, preserved alpha, controlled drawdown, low-MFE loss, and concentration.
- Variants that improve net only by increasing concentration, increasing drawdown, or starving frequency should be treated as research leads at most.
