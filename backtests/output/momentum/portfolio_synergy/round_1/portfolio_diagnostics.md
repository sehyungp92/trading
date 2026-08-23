# Momentum Family Portfolio Diagnostics

## Executive Read

Final local-best tested portfolio fired 1242 candidates, accepted 414, and blocked 828 (66.7% block rate).
Net profit was $87,295.53, return 174.6%, PF 2.57, win rate 59.7%, bar-close MTM max DD 10.32%, and 14.55 trades/month.
Key ratios: Sharpe 2.94, Sortino 3.47, Calmar 5.15.

Portfolio max DD is reported on a bar-close mark-to-market basis, matching the individual momentum strategy diagnostics. The prior daily realized-only DD for this same run was 3.89%.

This is a local optimum for the tested seven-component portfolio score, not proof of a global optimum.

## Explicit Synergy Verdict

**Is portfolio synergy maximized? No. Live-rule synergy is not maximized: the rejected trades contain more realized R than the losses avoided, and the relaxed shared-cap counterfactual has better return efficiency.**

| Question | Evidence | Answer |
|---|---|---|
| Are worse trades preferentially blocked? | Accepted-minus-blocked average R -0.347R; net block value -655.37R | No |
| Are winning-trade blocks minimized? | 69.08% of all eventual winners blocked; 554 blocked winners | No |
| Do live rules preserve opportunity? | 33.3% of relaxed trades and 1.4% of relaxed net profit captured | No |
| Do live rules improve max DD? | Optimized minus relaxed MTM DD -1.71% | Yes |
| Is the return/DD trade-off superior? | Calmar delta -47.51; PF delta -0.79 | No |

The relaxed scenario is an opportunity upper bound, not a deployable recommendation: it removes shared heat, contract, concurrency and stop constraints. Its purpose is to measure how much potentially valuable alpha the live rules discard.

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

## Live-Rule Synergy Counterfactual

| Measure | Optimized live rules | Relaxed shared caps | Optimized delta/capture |
|---|---:|---:|---:|
| Net profit | $87,296 | $6,171,526 | 1.4% captured |
| Accepted trades | 414 | 1242 | 33.3% captured |
| MTM max DD | 10.32% | 12.03% | -1.71% |
| Calmar | 5.15 | 52.66 | -47.51 |
| Profit factor | 2.57 | 3.36 | -0.79 |
| Immutable score | 0.5931 | 1.0756 | -0.4825 |

## Overall Blocker Discrimination

| Population | Count | Win rate | Total R | Avg R | Median R | P10 | P25 | P75 | P90 | Raw PnL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accepted | 414 | 59.9% | 184.10R | 0.445R | 0.181R | -1.008 | -0.404 | 0.890 | 1.916 | $121,638 |
| Blocked | 828 | 66.9% | 655.37R | 0.792R | 0.312R | -1.004 | -0.241 | 1.312 | 2.922 | $232,918 |

| Blocker-quality measure | Value | Interpretation |
|---|---:|---|
| Positive-trade block rate | 69.08% | Share of all eventual winners rejected |
| Non-positive-trade block rate | 62.27% | Share of all eventual non-winners rejected |
| Blocker precision | 33.09% | Share of blocks that were non-positive |
| Forgone winning R | 832.52R | Ex-post opportunity cost |
| Avoided losing R | 177.14R | Ex-post protection |
| Net block value | -655.37R | Positive means losses avoided exceeded winners forgone |
| Block efficiency | 17.54% | Avoided-loss share of gross blocked absolute R |
| Accepted-minus-blocked average R | -0.347R | Positive means accepted outcomes were stronger |
| Accepted win-rate uplift | -4.67% | Accepted WR minus fired-candidate WR |
| Net block value, source raw dollars | $-232,918 | Diagnostic only; source quantities are not shared-ledger sizing |

## Fired, Accepted, Blocked By Strategy

| Strategy | Fired | Accepted | Blocked | Accept Rate | Accepted WR | Blocked WR | Adjusted PnL | Blocked Raw PnL | Avg Accepted R | Avg Blocked R | Good-trade block rate | Bad-trade block rate | R discrimination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ_REGIME | 800 | 171 | 629 | 21.4% | 69.0% | 72.7% | $22,033 | $173,505 | 0.31 | 0.95 | 79.48% | 76.44% | -0.634R |
| VdubusNQ_v4 | 220 | 101 | 119 | 45.9% | 46.5% | 45.4% | $35,085 | $41,073 | 0.63 | 0.24 | 52.94% | 55.08% | +0.393R |
| NQDTC_v2.1 | 97 | 38 | 59 | 39.2% | 63.2% | 55.9% | $9,394 | $13,662 | 0.57 | 0.33 | 57.89% | 65.00% | +0.244R |
| DownturnDominator_v1 | 125 | 104 | 21 | 83.2% | 55.8% | 47.6% | $20,782 | $4,678 | 0.43 | 0.57 | 14.71% | 19.30% | -0.140R |

## Block Reasons

| Reason | Count | Winners | Non-winners | Raw PnL | Raw WR | Total R | Avg R | Net block value R | Avg open positions | Main strategies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| directional_cap | 441 | 291 | 150 | $115,048 | 66.0% | 284.71R | 0.65 | -284.71R | 0.10 | NQ_REGIME:310, VdubusNQ_v4:90, NQDTC_v2.1:41 |
| family_contract_cap | 289 | 206 | 83 | $88,594 | 71.3% | 317.25R | 1.10 | -317.25R | 0.09 | NQ_REGIME:276, NQDTC_v2.1:7, DownturnDominator_v1:6 |
| heat_cap | 59 | 32 | 27 | $15,781 | 54.2% | 28.10R | 0.48 | -28.10R | 0.64 | VdubusNQ_v4:25, NQ_REGIME:22, NQDTC_v2.1:7, DownturnDominator_v1:5 |
| portfolio_daily_stop | 28 | 19 | 9 | $6,125 | 67.9% | 17.81R | 0.64 | -17.81R | 0.00 | NQ_REGIME:21, NQDTC_v2.1:4, VdubusNQ_v4:3 |
| strategy_daily_stop | 11 | 6 | 5 | $7,370 | 54.5% | 7.51R | 0.68 | -7.51R | 0.00 | DownturnDominator_v1:10, VdubusNQ_v4:1 |

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

## Drawdown Path and Sleeve Attribution

| Scenario | Max DD | Peak | Trough | Recovery | Peak equity | Trough equity | Duration hours | Peak-to-trough contribution by strategy |
|---|---:|---|---|---|---:|---:|---:|---|
| optimized_live_rules | 10.32% | 2023-12-13T20:00:00+00:00 | 2023-12-20T19:40:00+00:00 | 2024-01-09T18:00:00+00:00 | $58,768 | $52,704 | 167.7 | NQ_REGIME:$-5,396, VdubusNQ_v4:$-668, NQDTC_v2.1:$+0, DownturnDominator_v1:$+0 |
| same_allocations_relaxed_shared_caps | 12.03% | 2024-11-12T15:25:00+00:00 | 2024-11-18T18:40:00+00:00 | 2024-11-19T19:45:00+00:00 | $414,713 | $364,818 | 147.2 | NQ_REGIME:$-45,871, VdubusNQ_v4:$-4,024, NQDTC_v2.1:$+0, DownturnDominator_v1:$+0 |
| live_rules_risk_1_5x | 14.97% | 2023-12-13T20:00:00+00:00 | 2023-12-20T19:40:00+00:00 | 2024-01-05T06:35:00+00:00 | $63,920 | $54,353 | 167.7 | NQ_REGIME:$-8,511, VdubusNQ_v4:$-1,056, NQDTC_v2.1:$+0, DownturnDominator_v1:$+0 |
| live_rules_risk_2_0x | 9.84% | 2024-03-07T19:00:00+00:00 | 2024-03-22T18:15:00+00:00 | 2024-05-06T19:50:00+00:00 | $69,849 | $62,973 | 359.2 | NQ_REGIME:$-4,345, VdubusNQ_v4:$-1,925, NQDTC_v2.1:$-606, DownturnDominator_v1:$+0 |

Negative sleeve contribution identifies MTM PnL lost from the portfolio drawdown peak to trough. It is path attribution, not proof that the sleeve should be removed.

## Monthly Opportunity and Blocking Path

| Month | Accepted | Blocked | Adjusted PnL | Accepted R | Blocked R | Blocked source raw PnL |
|---|---:|---:|---:|---:|---:|---:|
| 2023-11 | 1 | 0 | $-127 | -0.52R | 0.00R | $+0 |
| 2023-12 | 10 | 2 | $+3,149 | 11.23R | 3.53R | $+1,025 |
| 2024-01 | 16 | 3 | $+6,860 | 19.93R | 25.29R | $+4,155 |
| 2024-02 | 13 | 4 | $+2,843 | 7.75R | 21.21R | $+1,900 |
| 2024-03 | 32 | 11 | $+2,722 | 7.80R | 5.85R | $+1,499 |
| 2024-04 | 24 | 9 | $+2,490 | 5.92R | 8.23R | $+2,718 |
| 2024-05 | 27 | 22 | $+5,121 | 13.31R | 14.52R | $+3,592 |
| 2024-06 | 32 | 26 | $+9,773 | 20.83R | 15.70R | $+3,948 |
| 2024-07 | 18 | 16 | $+1,139 | 2.38R | 30.69R | $+10,228 |
| 2024-08 | 32 | 26 | $+4,896 | 9.34R | 58.07R | $+19,156 |
| 2024-09 | 24 | 16 | $+3,575 | 8.36R | 17.50R | $+6,116 |
| 2024-10 | 22 | 33 | $+6,758 | 12.74R | 29.48R | $+11,292 |
| 2024-11 | 21 | 28 | $+1,591 | 1.73R | 22.86R | $+4,875 |
| 2024-12 | 22 | 20 | $+12,936 | 17.99R | 23.14R | $+8,168 |
| 2025-01 | 14 | 19 | $+640 | 1.10R | 16.96R | $+10,809 |
| 2025-02 | 15 | 26 | $+1,128 | 1.33R | 18.00R | $+5,323 |
| 2025-03 | 13 | 27 | $+1,423 | 3.02R | 39.56R | $+20,262 |
| 2025-04 | 23 | 27 | $+1,632 | 3.06R | 12.63R | $+4,836 |
| 2025-05 | 12 | 38 | $-331 | 0.14R | 32.76R | $+11,484 |
| 2025-06 | 0 | 49 | $+0 | 0.00R | 23.61R | $+5,864 |
| 2025-07 | 0 | 58 | $+0 | 0.00R | 34.33R | $+17,706 |
| 2025-08 | 4 | 42 | $+4,980 | 8.95R | 11.30R | $+6,225 |
| 2025-09 | 0 | 49 | $+0 | 0.00R | 25.42R | $+9,774 |
| 2025-10 | 2 | 45 | $+1,729 | 3.28R | 16.27R | $+11,720 |
| 2025-11 | 0 | 30 | $+0 | 0.00R | 27.26R | $+15,230 |
| 2025-12 | 1 | 40 | $-590 | -1.01R | 17.72R | $+2,663 |
| 2026-01 | 3 | 39 | $+611 | 1.62R | 26.21R | $+15,570 |
| 2026-02 | 16 | 33 | $+6,492 | 12.25R | 19.47R | $+4,547 |
| 2026-03 | 12 | 43 | $+4,975 | 9.72R | 29.99R | $-1,091 |
| 2026-04 | 5 | 46 | $+881 | 1.86R | 27.91R | $+13,343 |
| 2026-05 | 0 | 1 | $+0 | 0.00R | -0.08R | $-19 |

## Individual Strategy Reference

| Strategy | Source round | Resolution | Individual Trades | Individual Return | PF | Max DD | Trades/Mo | High-value diagnostic note |
|---|---:|---|---:|---:|---:|---:|---:|---|
| NQDTC_v2.1 | 2 | configured_fallback | 161 | 352.8% | 1.87 | 17.0% |  |  |
| VdubusNQ_v4 | 1 | manifest_direct | 220 | 1217.5% | 2.18 | 16.7% | 6.49 |  |
| DownturnDominator_v1 | 4 | archived_sha256_match | 127 | 145.6% | 3.14 | 7.3% |  |  |
| NQ_REGIME | 6 | manifest_direct | 681 |  | 7.46 |  | 25.59 |  |

## Immutable Score, Scaling and Known Gap

Selection rule: weighted seven-component score; hard reject for non-positive PnL, MTM drawdown above 20%, PF below 1.35, or inactive sleeve.

**Important:** The immutable score penalizes aggregate block rate but has no direct component for blocker outcome discrimination or forgone winning R.

| Component | Weight | Target/scaling anchor | Optimized component | Weighted contribution | Relaxed component |
|---|---:|---:|---:|---:|---:|
| expected_return | 0.24 | $220,000 | 0.3968 | 0.0952 | 1.3000 |
| trade_frequency | 0.18 | 40.00 | 0.3638 | 0.0655 | 1.0691 |
| drawdown_control | 0.18 | 18.0% | 0.8068 | 0.1452 | 0.6641 |
| profit_quality | 0.13 | 2.80 | 0.9193 | 0.1195 | 1.2000 |
| risk_efficiency | 0.12 | 8.00 | 0.6437 | 0.0772 | 1.2500 |
| strategy_balance | 0.10 | 80.00 | 0.8163 | 0.0816 | 1.0000 |
| live_rule_health | 0.05 | 15.0% | 0.1750 | 0.0087 | 0.9125 |

Optimized aggregate score: 0.5931; relaxed shared-cap score: 1.0756; final validation score: .

Because blocker discrimination is absent from the score, the comprehensive verdict uses direct accepted-versus-blocked R diagnostics in addition to the immutable selection score.

## Phase Progression

| Phase | Candidates | Robust passes | Rejected | Accepted mutation | Accepted candidate | Current score | Best tested candidate | Best score |
|---:|---:|---:|---:|:---:|---|---:|---|---:|
| 1 | 5 | 0 | 0 | yes | vdubus_65bp | 0.5410 | vdubus_65bp | 0.5410 |
| 2 | 8 | 0 | 0 | yes | heat_6_25_contracts_22 | 0.5914 | heat_6_25_contracts_22 | 0.5914 |
| 3 | 4 | 0 | 0 | no |  | 0.5914 | oppose_block | 0.5914 |
| 4 | 6 | 0 | 0 | yes | portfolio_daily_2_25 | 0.5931 | portfolio_daily_2_25 | 0.5931 |
| 5 | 5 | 0 | 0 | no |  | 0.5931 | frequency_frontier | 0.5931 |

## Full Tested Frontier

| Phase | Candidate | Selected | Score | Validation score | Robust pass | Net Profit | Trades/Mo | Trades | PF | WR | MTM DD | Calmar | Block Rate | Validation PnL | Validation PF | Validation DD | Warnings/reason |
|---:|---|:---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | portfolio_daily_2_25 | yes | 0.5931 |  | no | $87,296 | 14.55 | 414 | 2.57 | 59.7% | 10.32% | 5.15 | 66.7% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 5 | frequency_frontier |  | 0.5931 |  | no | $87,296 | 14.55 | 414 | 2.57 | 59.7% | 10.32% | 5.15 | 66.7% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | heat_6_25_contracts_22 | yes | 0.5914 |  | no | $87,114 | 14.80 | 421 | 2.49 | 59.4% | 10.32% | 5.14 | 66.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 3 | oppose_block |  | 0.5914 |  | no | $87,114 | 14.80 | 421 | 2.49 | 59.4% | 10.32% | 5.14 | 66.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 3 | oppose_quarter |  | 0.5914 |  | no | $87,114 | 14.80 | 421 | 2.49 | 59.4% | 10.32% | 5.14 | 66.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 3 | direction_filter_off |  | 0.5914 |  | no | $87,114 | 14.80 | 421 | 2.49 | 59.4% | 10.32% | 5.14 | 66.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 4 | dd_tiers_tighter |  | 0.5914 |  | no | $87,114 | 14.80 | 421 | 2.49 | 59.4% | 10.32% | 5.14 | 66.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 4 | dd_tiers_looser |  | 0.5914 |  | no | $87,114 | 14.80 | 421 | 2.49 | 59.4% | 10.32% | 5.14 | 66.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 3 | agree_150 |  | 0.5912 |  | no | $87,059 | 14.80 | 421 | 2.49 | 59.4% | 10.32% | 5.14 | 66.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 4 | weekly_9_0 |  | 0.5902 |  | no | $86,712 | 14.87 | 423 | 2.48 | 59.1% | 10.32% | 5.12 | 65.9% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 4 | portfolio_daily_3_25 |  | 0.5896 |  | no | $86,587 | 14.87 | 423 | 2.47 | 59.3% | 10.32% | 5.12 | 65.9% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | capacity_combo |  | 0.5882 |  | no | $84,261 | 14.16 | 403 | 2.57 | 59.1% | 10.32% | 5.01 | 67.6% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | heat_5_75_contracts_20 |  | 0.5853 |  | no | $84,568 | 14.38 | 409 | 2.51 | 58.9% | 10.32% | 5.02 | 67.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 4 | weekly_6_0 |  | 0.5850 |  | no | $84,300 | 14.66 | 417 | 2.48 | 59.5% | 10.32% | 5.01 | 66.4% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | contracts_18 |  | 0.5705 |  | no | $77,721 | 13.22 | 374 | 2.62 | 59.1% | 10.32% | 4.74 | 69.9% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 5 | guarded_capacity_plus |  | 0.5621 |  | no | $79,754 | 14.76 | 420 | 2.36 | 59.3% | 11.40% | 4.35 | 66.2% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 5 | balanced_aggressive_combo |  | 0.5471 |  | no | $80,266 | 13.36 | 380 | 2.51 | 58.9% | 11.40% | 4.37 | 69.4% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | dir_caps_asym_4_75_5_25 |  | 0.5432 |  | no | $67,773 | 11.31 | 320 | 2.65 | 59.4% | 10.35% | 4.23 | 74.2% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 5 | dd_guarded_frequency_combo |  | 0.5426 |  | no | $76,786 | 14.59 | 415 | 2.33 | 58.6% | 11.40% | 4.22 | 66.6% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 1 | vdubus_65bp | yes | 0.5410 |  | no | $68,341 | 12.06 | 343 | 2.53 | 58.3% | 10.35% | 4.23 | 72.4% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | max_positions_6 |  | 0.5410 |  | no | $68,341 | 12.06 | 343 | 2.53 | 58.3% | 10.35% | 4.23 | 72.4% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | priority_headroom_050 |  | 0.5410 |  | no | $68,341 | 12.06 | 343 | 2.53 | 58.3% | 10.35% | 4.23 | 72.4% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 1 | downturn_50bp |  | 0.5386 |  | no | $62,472 | 12.13 | 345 | 2.53 | 60.3% | 10.11% | 4.03 | 72.2% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 1 | nqdtc_55bp |  | 0.5378 |  | no | $63,595 | 12.55 | 357 | 2.49 | 58.3% | 10.11% | 4.09 | 71.3% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 2 | heat_5_25 |  | 0.5365 |  | no | $67,033 | 12.16 | 346 | 2.46 | 58.1% | 10.35% | 4.17 | 72.1% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 5 | alpha_frequency_combo |  | 0.5240 |  | no | $70,276 | 12.20 | 345 | 2.51 | 59.4% | 11.40% | 3.96 | 72.2% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 1 | balanced_plus_10pct |  | 0.4963 |  | no | $59,662 | 10.82 | 306 | 2.49 | 59.2% | 11.43% | 3.46 | 75.4% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |
| 1 | nq_regime_70bp |  | 0.4903 |  | no | $53,215 | 10.09 | 287 | 2.57 | 59.2% | 11.20% | 3.19 | 76.9% |  |  |  | frequency_below_18_trades_per_month, block_rate_above_target, net_profit_below_target, profit_factor_below_target, strategy_balance_below_target |

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

## Synergy Decision Criteria

| Criterion | Status |
|---|:---:|
| block_net_value_r_nonnegative | FAIL |
| accepted_average_r_exceeds_blocked | FAIL |
| nonpositive_block_rate_exceeds_positive_block_rate | FAIL |
| optimized_drawdown_no_worse_than_relaxed | PASS |
| optimized_calmar_at_least_relaxed | FAIL |
| optimized_profit_factor_at_least_relaxed | FAIL |
| optimized_score_at_least_relaxed | FAIL |
| trade_capture_at_least_90pct | FAIL |
| block_rate_within_15pct_target | FAIL |
| frequency_at_least_40_per_month | FAIL |
| all_four_strategies_active | PASS |

## Interpretation

- The lower portfolio profit is not mainly because the individual strategies lost their edge. The relaxed shared-cap scenario demonstrates much more gross opportunity, but it requires position stacking that the live engine should not allow.
- The current local optimum is mainly a capital/risk-budget and simultaneous-signal problem: high-value signals cluster, then the live heat, directional, contract, and per-strategy concurrency rules decide which one gets the slot.
- Optimized live rules captured 1.4% of relaxed-cap net profit and 33.3% of relaxed-cap trades.
- The most blocked strategy was NQ_REGIME.
- Frequency remains below target; pushing it materially higher needs either better signal staggering/ranking or a deliberate increase in allowed shared heat, not independent-account recombination.

## Final Answer to the Portfolio-Synergy Objective

1. **Opportunity capture:** live rules retain 33.3% of relaxed trades and 1.4% of relaxed net profit. The latter is a diagnostic upper-bound comparison because relaxed compounding violates deployable shared caps.
2. **Blocking worse trades:** fails. Net block value is -655.37R and accepted-minus-blocked average R is -0.347R.
3. **Minimizing good-trade blocks:** 554 winners were blocked, representing 69.08% of all eventual winners. Blocker precision is 33.09%.
4. **Max-drawdown trade-off:** live rules change MTM DD by -1.71%, but Calmar changes by -47.51; the return/DD trade-off does not pass.
5. **Maximized synergy:** not established. Verdict: `severe_destructive_rule_interference_not_synergistic`.

The correct use of this result is as shared-capital sizing/routing evidence. A subsequent optimization should add a frozen blocker-discrimination term to the score, rank simultaneous candidates using causal expected value, and validate that the admitted set has higher R than the rejected set without weakening live heat and contract safeguards.
