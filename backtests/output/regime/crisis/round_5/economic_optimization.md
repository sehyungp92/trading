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
  "shock_mult": 0.8,
  "grind_mult": 0.9
}
```

Base score: 0.786294
Optimized score: 0.786294
Accepted: none

## Standard Scenarios

| Scenario | Score | Rejected | Action Days | Avg Exposure | Regime Calmar Delta | Regime CAGR Delta | Regime MaxDD Delta |
|---|---:|---|---:|---:|---:|---:|---:|
| A_no_crisis | 0.525000 | False | 0.0% | 1.000 | +0.000 | +0.00% | +0.00% |
| B_threshold_only | 0.757356 | False | 6.8% | 0.966 | +0.097 | +0.83% | -2.49% |
| C_current_live | 0.786294 | False | 11.6% | 0.961 | +0.132 | +0.98% | -3.45% |
| D_advisory_light | 0.000000 | True | 47.0% | 0.943 | +0.112 | +0.65% | -3.38% |
| E_legacy_r5_threshold | 0.702973 | False | 6.8% | 0.976 | +0.072 | +0.60% | -1.98% |

## Greedy Rounds

| Round | Best | Score | Delta | Kept | Rejected |
|---:|---|---:|---:|---|---:|
| 1 | shock_0.90 | 0.784380 | -0.001913 | False | 4 |
