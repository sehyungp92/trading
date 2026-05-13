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

Base score: 0.810796
Optimized score: 0.810796
Accepted: none

## Standard Scenarios

| Scenario | Score | Rejected | Action Days | Avg Exposure | Regime Calmar Delta | Regime CAGR Delta | Regime MaxDD Delta |
|---|---:|---|---:|---:|---:|---:|---:|
| A_no_crisis | 0.525000 | False | 0.0% | 1.000 | +0.000 | +0.00% | +0.00% |
| B_threshold_only | 0.737113 | False | 6.9% | 0.964 | +0.093 | +0.52% | -2.91% |
| C_current_live | 0.810796 | False | 12.5% | 0.957 | +0.174 | +0.93% | -4.83% |
| D_advisory_light | 0.000000 | True | 47.4% | 0.939 | +0.152 | +0.58% | -4.76% |
| E_legacy_r5_threshold | 0.690150 | False | 6.9% | 0.974 | +0.069 | +0.39% | -2.27% |

## Greedy Rounds

| Round | Best | Score | Delta | Kept | Rejected |
|---:|---|---:|---:|---|---:|
| 1 | grind_0.95 | 0.801142 | -0.009654 | False | 4 |
