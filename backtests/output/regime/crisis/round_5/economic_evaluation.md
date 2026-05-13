# Crisis Action-Layer Economic Optimization

Thresholds are frozen at the latest optimized crisis detector config.
This run scores portfolio economics: Calmar, Sortino, max drawdown, crisis drawdown, CAGR, recovery, and rebound/turnover drag.

## Optimized Policy

```json
null
```

Base score: 0.702973
Optimized score: 0.702973
Accepted: none

## Standard Scenarios

| Scenario | Score | Rejected | Action Days | Avg Exposure | Regime Calmar Delta | Regime CAGR Delta | Regime MaxDD Delta |
|---|---:|---|---:|---:|---:|---:|---:|
| A_no_crisis | 0.525000 | False | 0.0% | 1.000 | +0.000 | +0.00% | +0.00% |
| B_r5_threshold_only | 0.702973 | False | 6.8% | 0.976 | +0.072 | +0.60% | -1.98% |
| C_r5_family_proxy | 0.719423 | False | 6.8% | 0.972 | +0.081 | +0.67% | -2.22% |
| D_advisory_light | 0.000000 | True | 47.0% | 0.955 | +0.069 | +0.34% | -2.38% |
| E_shock_grind_light | 0.717289 | False | 11.6% | 0.973 | +0.088 | +0.68% | -2.45% |

## Greedy Rounds

| Round | Best | Score | Delta | Kept | Rejected |
|---:|---|---:|---:|---|---:|
