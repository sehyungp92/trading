# ALCB Round 4 final optimization

The March-May 2026 interval is consumed development data. Selection is anchored to IS, early/late IS, the historical drawdown stress interval, paired resampling, and cost sensitivity.

- Candidates screened on IS: 42
- Selected candidate: `daily__cap125_sel30_flow8__stop2p35`
- Promotion eligible under the predeclared final guardrails: True

## Balanced versus selected

| Metric | Balanced IS | Selected IS | Balanced OOS | Selected OOS |
|---|---:|---:|---:|---:|
| Expected total R | 265.23 | 273.69 | 50.52 | 54.79 |
| Net profit | 31031.81 | 34199.94 | 2611.15 | 2623.51 |
| Trades/month | 55.86 | 54.91 | 55.29 | 53.77 |
| Profit factor | 1.89 | 1.99 | 3.35 | 3.48 |
| Max drawdown | 11.94% | 9.44% | 1.97% | 2.10% |

## Robustness

- Early/late IS R retention: 1.122 / 0.957.
- Stress R delta: +2.67R.
- Stress DD ratio: 0.807.
- Paired day-block bootstrap probability of positive R uplift: 67.1%.
- Paired day-block bootstrap probability of positive dollar uplift: 94.7%.

## Cost sensitivity

| Candidate | Window | Slip bps | Commission/share | R | Net | PF | DD |
|---|---|---:|---:|---:|---:|---:|---:|
| control__balanced | is | 7.5 | 0.0075 | 138.68 | 18299.71 | 1.551 | 14.06% |
| control__balanced | oos | 7.5 | 0.0075 | 45.51 | 2275.76 | 2.874 | 2.26% |
| control__balanced | is | 10.0 | 0.0100 | 32.95 | 8552.17 | 1.291 | 15.70% |
| control__balanced | oos | 10.0 | 0.0100 | 37.41 | 1882.69 | 2.401 | 2.55% |
| control__balanced | is | 15.0 | 0.0100 | -131.18 | -2503.58 | 0.889 | 33.21% |
| control__balanced | oos | 15.0 | 0.0100 | 22.47 | 1014.42 | 1.603 | 3.85% |
| daily__cap125_sel30_flow8__stop2p35 | is | 7.5 | 0.0075 | 164.71 | 20318.93 | 1.651 | 11.42% |
| daily__cap125_sel30_flow8__stop2p35 | oos | 7.5 | 0.0075 | 47.00 | 2275.08 | 2.999 | 2.36% |
| daily__cap125_sel30_flow8__stop2p35 | is | 10.0 | 0.0100 | 74.71 | 10607.97 | 1.380 | 12.41% |
| daily__cap125_sel30_flow8__stop2p35 | oos | 10.0 | 0.0100 | 38.78 | 1897.90 | 2.494 | 2.60% |
| daily__cap125_sel30_flow8__stop2p35 | is | 15.0 | 0.0100 | -88.52 | -1289.38 | 0.941 | 26.64% |
| daily__cap125_sel30_flow8__stop2p35 | oos | 15.0 | 0.0100 | 23.89 | 1069.48 | 1.675 | 3.39% |

The result is saved as a Round-4 candidate only after artifact promotion. The unavailable authoritative frozen direct-RTH bundle remains a provenance limitation.
