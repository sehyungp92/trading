# Round 3 Signal Selection Diagnostics

Split time UTC: 2025-06-10T15:25:00+00:00
Score components: 7
Max workers: 2

## Accepted Phases

| Phase | Accepted | Candidate | Score | Validation Score |
|---:|---|---|---:|---:|
| 1 | False |  | 0.9956 | 0.8876 |
| 2 | False |  | 0.9956 | 0.8876 |
| 3 | True | capacity_10_00_contracts_40_positions_8_risk_2_0 | 1.1256 | 0.9175 |
| 4 | False |  | 1.1256 | 0.9175 |

## Candidate Frontier

| Phase | Candidate | Robust | Reason | Score | Val Score | Net Profit | Trades | Block Rate | WR | PF | Max DD | Val Net |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | filter_nq_regime_wide_ib | False | signal_filter_sample_below_threshold | 0.9991 | 0.8812 | $208,143 | 1092 | 13.3% | 66.5% | 3.77 | 6.63% | $75,073 |
| 1 | filter_vdubus_close | False | signal_filter_sample_below_threshold | 0.9899 | 0.8829 | $193,633 | 1112 | 11.7% | 66.6% | 3.54 | 7.07% | $75,704 |
| 1 | filter_nqdtc_score_2_5 | False | signal_filter_sample_below_threshold | 0.9883 | 0.8861 | $198,220 | 1092 | 13.3% | 67.4% | 3.80 | 7.07% | $76,225 |
| 2 | vdubus_second_slot | False | validation_score_not_improved | 1.0323 | 0.8876 | $200,777 | 1218 | 3.3% | 65.2% | 3.50 | 7.07% | $75,040 |
| 2 | selective_extra_slots | False | validation_score_not_improved | 1.0323 | 0.8876 | $200,777 | 1218 | 3.3% | 65.2% | 3.50 | 7.07% | $75,040 |
| 2 | nq_regime_third_slot | False | validation_score_not_improved | 0.9956 | 0.8876 | $196,778 | 1119 | 11.1% | 66.7% | 3.58 | 7.07% | $75,040 |
| 3 | capacity_10_00_contracts_40_positions_8_risk_2_0 | True |  | 1.1256 | 0.9175 | $297,275 | 1219 | 3.2% | 65.1% | 3.54 | 7.07% | $99,915 |
| 3 | capacity_10_00_contracts_40_positions_8_risk_1_75 | True |  | 1.1090 | 0.9173 | $270,814 | 1219 | 3.2% | 65.0% | 3.72 | 7.07% | $94,576 |
| 3 | capacity_9_00_contracts_34_positions_7_risk_1_75 | True |  | 1.0878 | 0.9136 | $251,292 | 1219 | 3.2% | 65.0% | 3.55 | 7.07% | $91,157 |
| 3 | capacity_8_25_contracts_30_positions_7_risk_1_5 | True |  | 1.0633 | 0.9018 | $226,522 | 1227 | 2.5% | 65.0% | 3.62 | 7.07% | $81,357 |
| 3 | capacity_7_25_contracts_26_risk_1_5 | True |  | 1.0462 | 0.8926 | $210,813 | 1227 | 2.5% | 65.0% | 3.47 | 7.07% | $77,260 |
| 3 | capacity_6_75_contracts_24_risk_1_5 | False | validation_score_not_improved | 1.0363 | 0.8876 | $201,712 | 1227 | 2.5% | 65.1% | 3.39 | 7.07% | $75,040 |
| 3 | capacity_6_25_contracts_22_risk_1_5 | False | validation_score_not_improved | 1.0220 | 0.8822 | $191,143 | 1227 | 2.5% | 65.1% | 3.30 | 7.07% | $72,482 |
| 4 | filter_vdubus_close_after_capacity | False | signal_filter_sample_below_threshold | 1.1194 | 0.9150 | $292,685 | 1200 | 4.7% | 65.4% | 3.57 | 7.07% | $101,784 |
| 4 | filter_nqdtc_score_2_5_after_capacity | False | signal_filter_sample_below_threshold | 1.1181 | 0.9149 | $303,023 | 1196 | 5.0% | 65.8% | 3.77 | 7.07% | $101,706 |
| 4 | filter_nq_regime_wide_ib_after_capacity | False | signal_filter_sample_below_threshold | 1.1161 | 0.9098 | $306,391 | 1190 | 5.5% | 65.0% | 3.69 | 6.55% | $99,530 |
| 4 | filter_nq_regime_wide_and_nqdtc_low_score | False | signal_filter_sample_below_threshold | 1.1086 | 0.9034 | $312,166 | 1167 | 7.3% | 65.7% | 3.94 | 6.51% | $101,312 |

## Signal Filter Sample Checks

| Phase | Candidate | Rule | Full N | Train N | Val N | Train Avg R | Val Avg R | Raw Val PnL | Gate Note |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | filter_nq_regime_wide_ib | nq_regime_wide_ib | 31 | 17 | 14 | 1.11 | -0.00 | $158 | sample_below_threshold |
| 1 | filter_vdubus_close | vdubus_close_window | 19 | 7 | 12 | 1.57 | -0.17 | $-2,965 | sample_below_threshold |
| 1 | filter_nqdtc_score_2_5 | nqdtc_score_2_5 | 29 | 17 | 12 | 0.08 | -0.32 | $-3,980 | sample_below_threshold |
| 4 | filter_vdubus_close_after_capacity | vdubus_close_window | 19 | 7 | 12 | 1.57 | -0.17 | $-2,965 | sample_below_threshold |
| 4 | filter_nqdtc_score_2_5_after_capacity | nqdtc_score_2_5 | 29 | 17 | 12 | 0.08 | -0.32 | $-3,980 | sample_below_threshold |
| 4 | filter_nq_regime_wide_ib_after_capacity | nq_regime_wide_ib | 31 | 17 | 14 | 1.11 | -0.00 | $158 | sample_below_threshold |
| 4 | filter_nq_regime_wide_and_nqdtc_low_score | nq_regime_wide_ib | 31 | 17 | 14 | 1.11 | -0.00 | $158 | sample_below_threshold |
| 4 | filter_nq_regime_wide_and_nqdtc_low_score | nqdtc_score_2_5 | 29 | 17 | 12 | 0.08 | -0.32 | $-3,980 | sample_below_threshold |

## Interpretation

- Round 3 only used entry-time metadata for signal filters.
- Small or train/validation-unstable filter buckets were treated as overfit risks.
- The accepted improvement came from letting robust blocked opportunity through shared live-style caps, not from a brittle low-sample exclusion rule.
