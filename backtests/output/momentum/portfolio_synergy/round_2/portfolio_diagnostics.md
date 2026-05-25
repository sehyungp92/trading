# Momentum Family Portfolio Diagnostics

## Executive Read

Final local-best tested portfolio fired 1295 candidates, accepted 1238, and blocked 57 (4.4% block rate).
Net profit was $301,556.84, return 603.1%, PF 3.40, win rate 63.5%, bar-close MTM max DD 7.55%, and 42.63 trades/month.
Key ratios: Sharpe 5.98, Sortino 10.69, Calmar 16.39.

Portfolio max DD is reported on a bar-close mark-to-market basis, matching the individual momentum strategy diagnostics. The prior daily realized-only DD for this same run was 1.88%.

This is a local optimum for the tested seven-component portfolio score, not proof of a global optimum.

## Portfolio Risk Basis

| Basis | Max DD | Final Equity | Net Return | Calmar | Points | Source |
|---|---:|---:|---:|---:|---:|---|
| Bar-close MTM | 7.55% | $351,557 | 603.1% | 16.39 | 170268 | backtests\momentum\data\raw\NQ_5m.parquet |
| Daily realized legacy | 1.88% | $352,294 | 604.6% | 66.11 | 885 | closed-trade daily curve |

## Scenario Comparison

| Scenario | Trades | Blocked | Block Rate | Net Profit | Trades/Mo | Win Rate | PF | MTM Max DD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_live_rules | 1238 | 57 | 4.4% | $301,557 | 42.63 | 63.5% | 3.40 | 7.55% | 5.98 | 10.69 | 16.39 |
| same_allocations_relaxed_shared_caps | 1282 | 13 | 1.0% | $391,188 | 44.14 | 63.7% | 3.91 | 7.80% | 5.91 | 13.10 | 18.69 |
| live_rules_risk_1_5x | 1237 | 58 | 4.5% | $314,815 | 42.59 | 63.5% | 3.41 | 9.95% | 5.60 | 10.12 | 12.79 |
| live_rules_risk_2_0x | 1237 | 58 | 4.5% | $317,280 | 42.59 | 63.5% | 3.42 | 11.69% | 5.47 | 10.04 | 10.94 |

## Fired, Accepted, Blocked By Strategy

| Strategy | Fired | Accepted | Blocked | Accept Rate | Accepted WR | Blocked Raw WR | Adjusted PnL | Blocked Raw PnL | Avg Accepted R | Avg Blocked R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ_REGIME | 800 | 775 | 25 | 96.9% | 71.9% | 72.0% | $219,959 | $4,502 | 0.82 | 0.56 |
| VdubusNQ_v4 | 212 | 207 | 5 | 97.6% | 45.4% | 60.0% | $36,677 | $1,970 | 0.40 | 0.83 |
| NQDTC_v2.1 | 158 | 152 | 6 | 96.2% | 55.3% | 50.0% | $32,932 | $2,861 | 0.51 | 0.86 |
| DownturnDominator_v1 | 125 | 104 | 21 | 83.2% | 49.0% | 81.0% | $11,989 | $8,556 | 0.38 | 0.85 |

## Block Reasons

| Reason | Count | Raw PnL Of Blocked | Raw WR | Avg Blocked R | Avg Open Positions | Main Strategies |
|---|---:|---:|---:|---:|---:|---|
| portfolio_daily_stop | 30 | $9,675 | 76.7% | 0.84 | 0.00 | NQ_REGIME:23, VdubusNQ_v4:3, NQDTC_v2.1:2, DownturnDominator_v1:2 |
| dynamic_capacity_floor | 13 | $7,177 | 84.6% | 0.81 | 0.15 | DownturnDominator_v1:13 |
| strategy_daily_stop | 10 | $-1,759 | 50.0% | 0.21 | 0.10 | DownturnDominator_v1:6, NQDTC_v2.1:4 |
| portfolio_weekly_stop | 4 | $2,797 | 50.0% | 0.86 | 0.00 | VdubusNQ_v4:2, NQ_REGIME:2 |

## Candidate Size Pressure

| Reason | Avg Current Heat R | Avg Base Risk R | Avg Current MNQ-eq | Avg Base MNQ-eq | Single Order > Heat Cap | Single Order > Contract Cap |
|---|---:|---:|---:|---:|---:|---:|
| portfolio_daily_stop | 0.00 | 0.00 | 0.0 | 0.0 | 0.0% | 0.0% |
| dynamic_capacity_floor | 2.57 | 0.00 | 8.2 | 0.0 | 0.0% | 0.0% |
| strategy_daily_stop | 0.00 | 0.00 | 0.0 | 0.0 | 0.0% | 0.0% |
| portfolio_weekly_stop | 0.00 | 0.00 | 0.0 | 0.0 | 0.0% | 0.0% |

## Signal Crowding

- Candidates with another family signal within 15m: 16.5%
- Candidates with another family signal within 60m: 50.6%
- Blocked candidates with an accepted position already open: 5.3%
- Average accepted open positions at blocked entry time: 0.05

Top within-15m strategy pairs:
- NQ_REGIME / NQ_REGIME: 86
- NQ_REGIME / VdubusNQ_v4: 15
- NQDTC_v2.1 / VdubusNQ_v4: 4
- DownturnDominator_v1 / NQ_REGIME: 4
- NQDTC_v2.1 / NQ_REGIME: 3
- DownturnDominator_v1 / DownturnDominator_v1: 2
- DownturnDominator_v1 / NQDTC_v2.1: 1

## Individual Strategy Reference

| Strategy | Individual Trades | Individual Return | PF | Max DD | Trades/Mo | High-value diagnostic note |
|---|---:|---:|---:|---:|---:|---|
| NQDTC_v2.1 | 161 | 352.8% | 1.87 | 17.0% |  |  |
| VdubusNQ_v4 | 212 | 1373.6% | 2.65 | 17.5% | 6.26 | Fixed-qty headline return is 1373.6% with 212 trades, PF=2.65, fixed-qty DD=17.5%; deployable comparison should use 87.8 total R and 2.59 R/month. Normalized simple return would be 22.0% at 0.25% risk/R, 43.9% at 0.50% risk/R, and 87.8% at 1.00% risk/R over the sample. The fixed-qty run implies about $1,564 per R, which is why raw return is not comparable to dynamically sized strategies. |
| DownturnDominator_v1 | 127 | 145.6% | 3.14 | 7.3% |  | Correction PnL is 134.0% with coverage 57.1%. Bear capture ratio is 11.1%. |
| NQ_REGIME | 681 |  | 7.46 |  | 25.59 |  |

## Tested Frontier

| Phase | Candidate | Score | Net Profit | Trades/Mo | Trades | PF | MTM Max DD | Block Rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | capacity_10_00_contracts_40_positions_8_risk_2_0 | 1.1256 | $297,275 | 41.97 | 1219 | 3.54 | 7.55% | 3.2% |
| 4 | filter_vdubus_close_after_capacity | 1.1194 | $292,685 | 41.32 | 1200 | 3.57 | 7.60% | 4.7% |
| 4 | filter_nqdtc_score_2_5_after_capacity | 1.1181 | $303,023 | 41.18 | 1196 | 3.77 | 7.50% | 5.0% |
| 4 | filter_nq_regime_wide_ib_after_capacity | 1.1161 | $306,391 | 40.97 | 1190 | 3.69 | 7.63% | 5.5% |
| 3 | capacity_10_00_contracts_40_positions_8_risk_1_75 | 1.1090 | $270,814 | 41.97 | 1219 | 3.72 | 7.52% | 3.2% |
| 4 | filter_nq_regime_wide_and_nqdtc_low_score | 1.1086 | $312,166 | 40.18 | 1167 | 3.94 | 7.59% | 7.3% |
| 3 | capacity_9_00_contracts_34_positions_7_risk_1_75 | 1.0878 | $251,292 | 41.97 | 1219 | 3.55 | 7.55% | 3.2% |
| 3 | capacity_8_25_contracts_30_positions_7_risk_1_5 | 1.0633 | $226,522 | 42.25 | 1227 | 3.62 | 7.82% | 2.5% |
| 3 | capacity_7_25_contracts_26_risk_1_5 | 1.0462 | $210,813 | 42.25 | 1227 | 3.47 | 7.77% | 2.5% |
| 3 | capacity_6_75_contracts_24_risk_1_5 | 1.0363 | $201,712 | 42.25 | 1227 | 3.39 | 7.56% | 2.5% |

## Implementation Safeguards

| Safeguard | Status |
|---|---|
| Replay contract | completed_source_trade_replay_live_portfolio_rules.v1 |
| Evidence scope | portfolio_sizing_evidence_not_full_source_execution_simulation |
| Live portfolio rules | yes |
| Shared capital ledger | yes |
| Source artifact hashes recorded | yes |
| Source artifacts fingerprint | 544f777ad2c435c9597006568740bf1580beed558fe76464069dc7c70d1b1f4f |
| Headline risk basis | bar_close_mark_to_market |
| Decision stream status | not_provided_completed_trade_replay |
| Full source execution simulation | no |

The portfolio result is official for shared-capital sizing/routing evidence. It does not replace source-strategy live/backtest parity tests for fills, order paths, or intrabar execution.

## Interpretation

- The lower portfolio profit is not mainly because the individual strategies lost their edge. The relaxed shared-cap scenario demonstrates much more gross opportunity, but it requires position stacking that the live engine should not allow.
- The current local optimum is mainly a capital/risk-budget and simultaneous-signal problem: high-value signals cluster, then the live heat, directional, contract, and per-strategy concurrency rules decide which one gets the slot.
- Optimized live rules captured 77.1% of relaxed-cap net profit and 96.6% of relaxed-cap trades.
- The most blocked strategy was NQ_REGIME.
- Frequency clears the 24 trades/month target; the remaining improvement problem is alpha per accepted slot and reducing avoidable max-concurrent blocks.
