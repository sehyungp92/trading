# Momentum Family Portfolio Diagnostics

## Executive Read

Final local-best tested portfolio fired 1242 candidates, accepted 414, and blocked 828 (66.7% block rate).
Net profit was $87,295.53, return 174.6%, PF 2.57, win rate 59.7%, bar-close MTM max DD 10.32%, and 14.55 trades/month.
Key ratios: Sharpe 2.94, Sortino 3.47, Calmar 5.15.

Portfolio max DD is reported on a bar-close mark-to-market basis, matching the individual momentum strategy diagnostics. The prior daily realized-only DD for this same run was 3.89%.

This is a local optimum for the tested seven-component portfolio score, not proof of a global optimum.

## Portfolio Risk Basis

| Basis | Max DD | Final Equity | Net Return | Calmar | Points | Source |
|---|---:|---:|---:|---:|---:|---|
| Bar-close MTM | 10.32% | $137,296 | 174.6% | 5.15 | 166260 | backtests\momentum\data\raw\NQ_5m.parquet |
| Daily realized legacy | 3.89% | $137,991 | 176.0% | 13.74 | 867 | closed-trade daily curve |

## Scenario Comparison

| Scenario | Trades | Blocked | Block Rate | Net Profit | Trades/Mo | Win Rate | PF | MTM Max DD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_live_rules | 414 | 828 | 66.7% | $87,296 | 14.55 | 59.7% | 2.57 | 10.32% | 2.94 | 3.47 | 5.15 |
| same_allocations_relaxed_shared_caps | 1242 | 0 | 0.0% | $6,171,526 | 42.76 | 64.5% | 3.36 | 12.03% | 6.43 | 13.25 | 52.66 |
| live_rules_risk_1_5x | 347 | 895 | 72.1% | $100,434 | 12.20 | 58.8% | 2.24 | 14.97% | 2.48 | 2.94 | 3.95 |
| live_rules_risk_2_0x | 251 | 991 | 79.8% | $83,767 | 8.82 | 61.0% | 2.35 | 9.84% | 2.31 | 1.90 | 5.23 |

## Fired, Accepted, Blocked By Strategy

| Strategy | Fired | Accepted | Blocked | Accept Rate | Accepted WR | Blocked Raw WR | Adjusted PnL | Blocked Raw PnL | Avg Accepted R | Avg Blocked R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ_REGIME | 800 | 171 | 629 | 21.4% | 69.0% | 72.7% | $22,033 | $173,505 | 0.31 | 0.95 |
| VdubusNQ_v4 | 220 | 101 | 119 | 45.9% | 46.5% | 45.4% | $35,085 | $41,073 | 0.63 | 0.24 |
| NQDTC_v2.1 | 97 | 38 | 59 | 39.2% | 63.2% | 55.9% | $9,394 | $13,662 | 0.57 | 0.33 |
| DownturnDominator_v1 | 125 | 104 | 21 | 83.2% | 55.8% | 47.6% | $20,782 | $4,678 | 0.43 | 0.57 |

## Block Reasons

| Reason | Count | Raw PnL Of Blocked | Raw WR | Avg Blocked R | Avg Open Positions | Main Strategies |
|---|---:|---:|---:|---:|---:|---|
| directional_cap | 441 | $115,048 | 66.0% | 0.65 | 0.10 | NQ_REGIME:310, VdubusNQ_v4:90, NQDTC_v2.1:41 |
| family_contract_cap | 289 | $88,594 | 71.3% | 1.10 | 0.09 | NQ_REGIME:276, NQDTC_v2.1:7, DownturnDominator_v1:6 |
| heat_cap | 59 | $15,781 | 54.2% | 0.48 | 0.64 | VdubusNQ_v4:25, NQ_REGIME:22, NQDTC_v2.1:7, DownturnDominator_v1:5 |
| portfolio_daily_stop | 28 | $6,125 | 67.9% | 0.64 | 0.00 | NQ_REGIME:21, NQDTC_v2.1:4, VdubusNQ_v4:3 |
| strategy_daily_stop | 11 | $7,370 | 54.5% | 0.68 | 0.00 | DownturnDominator_v1:10, VdubusNQ_v4:1 |

## Candidate Size Pressure

| Reason | Avg Current Heat R | Avg Base Risk R | Avg Current MNQ-eq | Avg Base MNQ-eq | Single Order > Heat Cap | Single Order > Contract Cap |
|---|---:|---:|---:|---:|---:|---:|
| directional_cap | 2.72 | 2.86 | 10.4 | 31.5 | 0.0% | 0.0% |
| family_contract_cap | 1.17 | 2.38 | 4.5 | 41.3 | 0.0% | 53.6% |
| heat_cap | 3.98 | 3.31 | 12.2 | 19.5 | 0.0% | 0.0% |
| portfolio_daily_stop | 0.00 | 0.00 | 0.0 | 0.0 | 0.0% | 0.0% |
| strategy_daily_stop | 0.00 | 0.00 | 0.0 | 0.0 | 0.0% | 0.0% |

## Signal Crowding

- Candidates with another family signal within 15m: 17.7%
- Candidates with another family signal within 60m: 51.4%
- Blocked candidates with an accepted position already open: 12.4%
- Average accepted open positions at blocked entry time: 0.13

Top within-15m strategy pairs:
- NQ_REGIME / NQ_REGIME: 86
- NQ_REGIME / VdubusNQ_v4: 15
- NQDTC_v2.1 / NQ_REGIME: 5
- NQDTC_v2.1 / VdubusNQ_v4: 5
- DownturnDominator_v1 / NQ_REGIME: 4
- DownturnDominator_v1 / DownturnDominator_v1: 2
- DownturnDominator_v1 / NQDTC_v2.1: 1

## Individual Strategy Reference

| Strategy | Individual Trades | Individual Return | PF | Max DD | Trades/Mo | High-value diagnostic note |
|---|---:|---:|---:|---:|---:|---|
| DownturnDominator_v1 | 127 | 145.6% | 3.14 | 7.3% |  | Correction PnL is 134.0% with coverage 57.1%. Bear capture ratio is 11.1%. |
| NQ_REGIME | 681 |  | 7.46 |  | 25.59 |  |

## Tested Frontier

| Phase | Candidate | Score | Net Profit | Trades/Mo | Trades | PF | MTM Max DD | Block Rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | portfolio_daily_2_25 | 0.5931 | $87,296 | 14.55 | 414 | 2.57 | 10.32% | 66.7% |
| 5 | frequency_frontier | 0.5931 | $87,296 | 14.55 | 414 | 2.57 | 10.32% | 66.7% |
| 2 | heat_6_25_contracts_22 | 0.5914 | $87,114 | 14.80 | 421 | 2.49 | 10.32% | 66.1% |
| 3 | oppose_block | 0.5914 | $87,114 | 14.80 | 421 | 2.49 | 10.32% | 66.1% |
| 3 | oppose_quarter | 0.5914 | $87,114 | 14.80 | 421 | 2.49 | 10.32% | 66.1% |
| 3 | direction_filter_off | 0.5914 | $87,114 | 14.80 | 421 | 2.49 | 10.32% | 66.1% |
| 4 | dd_tiers_tighter | 0.5914 | $87,114 | 14.80 | 421 | 2.49 | 10.32% | 66.1% |
| 4 | dd_tiers_looser | 0.5914 | $87,114 | 14.80 | 421 | 2.49 | 10.32% | 66.1% |
| 3 | agree_150 | 0.5912 | $87,059 | 14.80 | 421 | 2.49 | 10.32% | 66.1% |
| 4 | weekly_9_0 | 0.5902 | $86,712 | 14.87 | 423 | 2.48 | 10.32% | 65.9% |

## Implementation Safeguards

| Safeguard | Status |
|---|---|
| Replay contract | completed_source_trade_replay_live_portfolio_rules.v1 |
| Evidence scope | portfolio_sizing_evidence_not_full_source_execution_simulation |
| Live portfolio rules | yes |
| Shared capital ledger | yes |
| Source artifact hashes recorded | yes |
| Source artifacts fingerprint | a6aa738ca18df8d1b5ffce8c5526e84c970e8474fbf0f44ec1f27b80f7ce63fd |
| Headline risk basis | bar_close_mark_to_market |
| Decision stream status | not_provided_completed_trade_replay |
| Full source execution simulation | no |

The portfolio result is official for shared-capital sizing/routing evidence. It does not replace source-strategy live/backtest parity tests for fills, order paths, or intrabar execution.

## Interpretation

- The lower portfolio profit is not mainly because the individual strategies lost their edge. The relaxed shared-cap scenario demonstrates much more gross opportunity, but it requires position stacking that the live engine should not allow.
- The current local optimum is mainly a capital/risk-budget and simultaneous-signal problem: high-value signals cluster, then the live heat, directional, contract, and per-strategy concurrency rules decide which one gets the slot.
- Optimized live rules captured 1.4% of relaxed-cap net profit and 33.3% of relaxed-cap trades.
- The most blocked strategy was NQ_REGIME.
- Frequency remains below target; pushing it materially higher needs either better signal staggering/ranking or a deliberate increase in allowed shared heat, not independent-account recombination.
