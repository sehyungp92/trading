# Crisis Action-Layer Economic Optimization

Thresholds are frozen at the latest optimized crisis detector config.
This run scores portfolio economics: Calmar, Sortino, max drawdown, crisis drawdown, CAGR, recovery, and rebound/turnover drag.

## Optimized Policy

```json
{
  "name": "shock_0.75",
  "warning_mult": 0.65,
  "crisis_mult": 0.3,
  "advisory_mult": 1.0,
  "shock_mult": 0.75,
  "grind_mult": 0.9,
  "credit_impulse_mult": 0.75
}
```

Base score: 0.871815
Optimized score: 0.878598
Accepted: shock_0.75

## Standard Scenarios

| Scenario | Score | Rejected | Action Days | Avg Exposure | Regime Calmar Delta | Regime CAGR Delta | Regime MaxDD Delta |
|---|---:|---|---:|---:|---:|---:|---:|
| A_no_crisis | 0.525000 | False | 0.0% | 1.000 | +0.000 | +0.00% | +0.00% |
| B_threshold_only | 0.754578 | False | 6.9% | 0.964 | +0.115 | +0.61% | -3.54% |
| C_current_live | 0.871815 | False | 13.7% | 0.950 | +0.246 | +1.77% | -5.48% |
| D_advisory_light | 0.852625 | False | 36.8% | 0.939 | +0.222 | +1.36% | -5.46% |
| E_legacy_r5_threshold | 0.701553 | False | 6.9% | 0.974 | +0.084 | +0.45% | -2.72% |

## Greedy Rounds

| Round | Best | Score | Delta | Kept | Rejected |
|---:|---|---:|---:|---|---:|
| 1 | shock_0.75 | 0.878598 | +0.006783 | True | 0 |
| 2 | grind_0.95 | 0.876054 | -0.002544 | False | 0 |
