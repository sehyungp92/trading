# Momentum Family Portfolio Diagnostics

## Executive Read

Final local-best tested portfolio fired 1259 candidates, accepted 441, and blocked 818 (65.0% block rate).
Net profit was $91,402.42, return 182.8%, PF 2.67, win rate 61.5%, bar-close MTM max DD 9.43%, and 15.50 trades/month.
Key ratios: Sharpe 3.10, Sortino 4.06, Calmar 5.84.

Portfolio max DD is reported on a bar-close mark-to-market basis, matching the individual momentum strategy diagnostics. The prior daily realized-only DD for this same run was 4.09%.

This is a local optimum for the tested seven-component portfolio score, not proof of a global optimum.

## Portfolio Risk Basis

| Basis | Max DD | Final Equity | Net Return | Calmar | Points | Source |
|---|---:|---:|---:|---:|---:|---|
| Bar-close MTM | 9.43% | $141,402 | 182.8% | 5.84 | 166260 | backtests\momentum\data\raw\NQ_5m.parquet |
| Daily realized legacy | 4.09% | $142,161 | 184.3% | 13.55 | 867 | closed-trade daily curve |

## Scenario Comparison

| Scenario | Trades | Blocked | Block Rate | Net Profit | Trades/Mo | Win Rate | PF | MTM Max DD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_live_rules | 441 | 818 | 65.0% | $91,402 | 15.50 | 61.5% | 2.67 | 9.43% | 3.10 | 4.06 | 5.84 |
| same_allocations_relaxed_shared_caps | 1259 | 0 | 0.0% | $6,925,875 | 43.35 | 65.4% | 3.76 | 11.90% | 6.69 | 15.94 | 56.21 |
| live_rules_risk_1_5x | 351 | 908 | 72.1% | $114,571 | 12.09 | 59.3% | 2.39 | 13.77% | 2.49 | 3.46 | 4.62 |
| live_rules_risk_2_0x | 216 | 1043 | 82.8% | $100,573 | 7.64 | 61.6% | 2.89 | 16.66% | 2.39 | 2.43 | 3.58 |

## Fired, Accepted, Blocked By Strategy

| Strategy | Fired | Accepted | Blocked | Accept Rate | Accepted WR | Blocked Raw WR | Adjusted PnL | Blocked Raw PnL | Avg Accepted R | Avg Blocked R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ_REGIME | 800 | 183 | 617 | 22.9% | 68.9% | 72.8% | $21,844 | $173,495 | 0.29 | 0.97 |
| VdubusNQ_v4 | 210 | 98 | 112 | 46.7% | 54.1% | 49.1% | $36,751 | $44,368 | 0.63 | 0.26 |
| NQDTC_v2.1 | 124 | 56 | 68 | 45.2% | 60.7% | 57.4% | $12,118 | $29,527 | 0.56 | 0.54 |
| DownturnDominator_v1 | 125 | 104 | 21 | 83.2% | 55.8% | 47.6% | $20,690 | $4,678 | 0.43 | 0.57 |

## Block Reasons

| Reason | Count | Raw PnL Of Blocked | Raw WR | Avg Blocked R | Avg Open Positions | Main Strategies |
|---|---:|---:|---:|---:|---:|---|
| directional_cap | 376 | $120,503 | 67.8% | 0.74 | 0.12 | NQ_REGIME:284, NQDTC_v2.1:55, VdubusNQ_v4:37 |
| family_contract_cap | 286 | $87,263 | 71.7% | 1.10 | 0.09 | NQ_REGIME:272, NQDTC_v2.1:8, DownturnDominator_v1:6 |
| heat_cap | 140 | $41,175 | 60.0% | 0.50 | 0.21 | VdubusNQ_v4:75, NQ_REGIME:55, NQDTC_v2.1:5, DownturnDominator_v1:5 |
| strategy_daily_stop | 10 | $1,052 | 50.0% | 0.43 | 0.00 | DownturnDominator_v1:10 |
| portfolio_daily_stop | 6 | $2,074 | 66.7% | 0.80 | 0.00 | NQ_REGIME:6 |

## Candidate Size Pressure

| Reason | Avg Current Heat R | Avg Base Risk R | Avg Current MNQ-eq | Avg Base MNQ-eq | Single Order > Heat Cap | Single Order > Contract Cap |
|---|---:|---:|---:|---:|---:|---:|
| directional_cap | 2.95 | 2.89 | 11.2 | 35.3 | 0.0% | 0.0% |
| family_contract_cap | 1.20 | 2.41 | 4.6 | 41.4 | 0.0% | 55.9% |
| heat_cap | 3.39 | 3.53 | 12.5 | 22.7 | 0.0% | 0.0% |
| strategy_daily_stop | 0.00 | 0.00 | 0.0 | 0.0 | 0.0% | 0.0% |
| portfolio_daily_stop | 0.00 | 0.00 | 0.0 | 0.0 | 0.0% | 0.0% |

## Signal Crowding

- Candidates with another family signal within 15m: 17.7%
- Candidates with another family signal within 60m: 50.1%
- Blocked candidates with an accepted position already open: 11.7%
- Average accepted open positions at blocked entry time: 0.12

Top within-15m strategy pairs:
- NQ_REGIME / NQ_REGIME: 86
- NQ_REGIME / VdubusNQ_v4: 15
- NQDTC_v2.1 / NQ_REGIME: 7
- DownturnDominator_v1 / NQ_REGIME: 4
- DownturnDominator_v1 / NQDTC_v2.1: 2
- NQDTC_v2.1 / VdubusNQ_v4: 2
- DownturnDominator_v1 / DownturnDominator_v1: 2

## Individual Strategy Reference

| Strategy | Individual Trades | Individual Return | PF | Max DD | Trades/Mo | High-value diagnostic note |
|---|---:|---:|---:|---:|---:|---|
| NQDTC_v2.1 | 123 | 372.2% | 2.10 | 16.3% |  |  |
| VdubusNQ_v4 | 198 | 1166.0% | 2.78 | 18.1% | 6.05 | Net return 1166.0% with 198 trades, PF=2.78, DD=18.1%. R/month throughput is 2.66 (0.440 avgR x 6.0 trades/month). |
| DownturnDominator_v1 | 127 | 145.6% | 3.14 | 7.3% |  | Correction PnL is 134.0% with coverage 57.1%. Bear capture ratio is 11.1%. |
| NQ_REGIME | 681 |  | 7.46 |  | 25.59 |  |

## Tested Frontier

| Phase | Candidate | Score | Net Profit | Trades/Mo | Trades | PF | MTM Max DD | Block Rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | weekly_9_0 | 0.6377 | $91,402 | 15.50 | 441 | 2.67 | 9.43% | 65.0% |
| 5 | frequency_frontier | 0.6377 | $91,402 | 15.50 | 441 | 2.67 | 9.43% | 65.0% |
| 3 | direction_filter_off | 0.6375 | $91,388 | 15.43 | 439 | 2.67 | 9.43% | 65.1% |
| 4 | dd_tiers_looser | 0.6375 | $91,388 | 15.43 | 439 | 2.67 | 9.43% | 65.1% |
| 4 | dd_tiers_tighter | 0.6375 | $91,388 | 15.43 | 439 | 2.67 | 9.43% | 65.1% |
| 4 | weekly_6_0 | 0.6372 | $91,404 | 15.36 | 437 | 2.67 | 9.43% | 65.3% |
| 3 | oppose_quarter | 0.6363 | $90,315 | 15.46 | 440 | 2.66 | 9.43% | 65.1% |
| 4 | portfolio_daily_3_25 | 0.6362 | $91,016 | 15.50 | 441 | 2.65 | 9.43% | 65.0% |
| 3 | oppose_block | 0.6344 | $89,346 | 15.50 | 441 | 2.64 | 9.43% | 65.0% |
| 3 | agree_150 | 0.6329 | $89,452 | 15.43 | 439 | 2.63 | 9.43% | 65.1% |

## Interpretation

- The lower portfolio profit is not mainly because the individual strategies lost their edge. The relaxed shared-cap scenario demonstrates much more gross opportunity, but it requires position stacking that the live engine should not allow.
- The current local optimum is mainly a capital/risk-budget and simultaneous-signal problem: high-value signals cluster, then the live heat, directional, contract, and per-strategy concurrency rules decide which one gets the slot.
- Optimized live rules captured 1.3% of relaxed-cap net profit and 35.0% of relaxed-cap trades.
- The most blocked strategy was NQ_REGIME.
- Frequency remains below target; pushing it materially higher needs either better signal staggering/ranking or a deliberate increase in allowed shared heat, not independent-account recombination.
