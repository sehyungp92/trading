# TPC Hidden Alpha Variant Tests

Elapsed minutes: 8.1

## Full Replay OOS

| Option | Variant | Trades | Net | AvgR | PF | QQQ AvgR | GLD AvgR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | current_round7 | 10 | +11.00% | +0.661 | 3.28 | +0.168 | +0.716 |
| option1 | pb30_same_params | 0 | +0.00% | +0.000 | 0.00 | +0.000 | +0.000 |
| option1 | pb30_scaled_duration | 0 | +0.00% | +0.000 | 0.00 | +0.000 | +0.000 |
| option1 | pb30_quality | 0 | +0.00% | +0.000 | 0.00 | +0.000 | +0.000 |
| option1 | pb30_loose_confirm | 0 | +0.00% | +0.000 | 0.00 | +0.000 | +0.000 |
| option2 | daily_sma50 | 2 | +4.86% | +1.499 | 4858.23 | +0.000 | +1.499 |
| option2 | daily_sma20 | 0 | +0.00% | +0.000 | 0.00 | +0.000 | +0.000 |
| option2 | daily_sma20_50 | 0 | +0.00% | +0.000 | 0.00 | +0.000 | +0.000 |
| option2 | daily_ret20_sma20 | 0 | +0.00% | +0.000 | 0.00 | +0.000 | +0.000 |

## Event Study OOS

| Option | Variant | Signals | AvgR | PF | Hit1 | MFE | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | current_confirm_market | 34 | -0.235 | 0.62 | 38.2% | 0.84 | 61.8% |
| option1 | pb30_confirm_quality | 26 | +0.154 | 1.36 | 57.7% | 1.29 | 42.3% |
| option1 | pb30_confirm_same | 51 | +0.059 | 1.12 | 52.9% | 1.26 | 47.1% |
| option1 | pb30_confirm_scaled | 17 | +0.059 | 1.12 | 52.9% | 1.07 | 47.1% |
| option2 | daily_sma20_confirm | 6 | +0.000 | 1.00 | 50.0% | 0.81 | 50.0% |
| option2 | daily_sma50_confirm | 5 | -0.200 | 0.67 | 40.0% | 1.40 | 60.0% |
| option2 | daily_sma20_50_confirm | 1 | -1.000 | 0.00 | 0.0% | 0.09 | 100.0% |
| option2 | daily_ret20_sma20_confirm | 1 | -1.000 | 0.00 | 0.0% | 0.09 | 100.0% |
| option3 | touch_ema20_limit4 | 4 | +0.500 | 3.00 | 75.0% | 1.25 | 25.0% |
| option3 | touch_ema20_market | 7 | +0.429 | 2.50 | 71.4% | 1.21 | 28.6% |
| option3 | touch_vwap_limit4 | 9 | +0.111 | 1.25 | 55.6% | 1.48 | 44.4% |
| option3 | touch_ema20_limit4_daily | 0 | +0.000 | 0.00 | 0.0% | 0.00 | 0.0% |
| option3 | touch_vwap_market | 12 | -0.333 | 0.50 | 33.3% | 0.84 | 66.7% |
| option4 | confirm_ema20_retest4_daily | 0 | +0.000 | 0.00 | 0.0% | 0.00 | 0.0% |
| option4 | confirm_midpoint_retest4 | 20 | -0.100 | 0.82 | 45.0% | 1.25 | 55.0% |
| option4 | confirm_vwap_retest4 | 9 | -0.111 | 0.80 | 44.4% | 1.70 | 55.6% |
| option4 | confirm_vwap_retest8 | 10 | -0.200 | 0.67 | 40.0% | 1.92 | 60.0% |
| option4 | confirm_ema20_retest4 | 2 | -1.000 | 0.00 | 0.0% | 0.79 | 100.0% |

## Interpretation Guardrails
- Full replay uses the round-7 TPC implementation and includes available QQQ->NQ / GLD->GC context indicators.
- Event-study R is a conservative 1R/stop probe, not the production T1/T2/runner exit model.
- Holdout rows are the post-2025-11-01 sample and remain small; strong-looking low-N rows are research leads, not promotion evidence.
