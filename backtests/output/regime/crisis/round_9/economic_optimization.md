# Crisis Action-Layer Economic Optimization

Thresholds are frozen at the latest optimized crisis detector config.
This run scores portfolio economics: Calmar, Sortino, max drawdown, crisis drawdown, CAGR, recovery, and rebound/turnover drag.

## Optimized Policy

```json
{
  "name": "current_live",
  "warning_mult": 0.65,
  "crisis_mult": 0.3,
  "advisory_mult": 1.0,
  "shock_mult": 0.75,
  "grind_mult": 0.9,
  "credit_impulse_mult": 0.75
}
```

Base score: 0.872575
Optimized score: 0.872575
Accepted: none

## Standard Scenarios

| Scenario | Score | Rejected | Action Days | Avg Exposure | Regime Calmar Delta | Regime CAGR Delta | Regime MaxDD Delta |
|---|---:|---|---:|---:|---:|---:|---:|
| A_no_crisis | 0.525000 | False | 0.0% | 1.000 | +0.000 | +0.00% | +0.00% |
| B_threshold_only | 0.753051 | False | 7.2% | 0.964 | +0.116 | +0.54% | -3.69% |
| C_current_live | 0.872575 | False | 13.0% | 0.952 | +0.262 | +1.76% | -5.87% |
| D_advisory_light | 0.858760 | False | 36.8% | 0.940 | +0.238 | +1.37% | -5.86% |
| E_legacy_r5_threshold | 0.699393 | False | 7.2% | 0.974 | +0.084 | +0.40% | -2.83% |

## Greedy Rounds

| Round | Best | Score | Delta | Kept | Rejected |
|---:|---|---:|---:|---|---:|
| 1 | grind_0.95 | 0.869973 | -0.002601 | False | 0 |
