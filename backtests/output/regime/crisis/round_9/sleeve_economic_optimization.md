# Crisis Sleeve-Aware Economic Optimization

Thresholds are frozen at the latest optimized crisis detector config.
This replay separates equity-beta longs, GLD, and short-beta exposure. QQQ is proxied by SPY when QQQ returns are unavailable in the raw regime data.

## Optimized Policy

```json
{
  "name": "equity_current_short_preserved",
  "equity_warning_mult": 0.65,
  "equity_crisis_mult": 0.3,
  "gld_warning_mult": 0.65,
  "gld_crisis_mult": 0.3,
  "short_warning_mult": 1.0,
  "short_crisis_mult": 1.0
}
```

Base score: 0.936576
Optimized score: 0.950234
Accepted: equity_current_short_preserved

## Candidate Ranking

| Scenario | Score | Rejected | Action Days | Avg Exposure | Risk-On Calmar Delta | Risk-On CAGR Delta | Risk-On MaxDD Delta |
|---|---:|---|---:|---:|---:|---:|---:|
| equity_current_short_preserved | 0.950234 | False | 13.0% | 0.955 | +0.764 | +4.82% | -22.59% |
| conservative_asym | 0.948007 | False | 13.0% | 0.959 | +0.740 | +4.80% | -22.29% |
| strong_equity_keep_def | 0.946728 | False | 13.0% | 0.960 | +0.861 | +5.25% | -23.35% |
| balanced_asym | 0.946392 | False | 13.0% | 0.961 | +0.753 | +4.86% | -22.41% |
| preserve_defensive | 0.944623 | False | 13.0% | 0.963 | +0.754 | +4.93% | -22.35% |
| preserve_gld_short | 0.941453 | False | 13.0% | 0.965 | +0.749 | +4.95% | -22.26% |
| current_symmetric | 0.936576 | False | 13.0% | 0.952 | +0.647 | +4.36% | -21.39% |
| equity_current_gld_preserved | 0.926531 | False | 13.0% | 0.961 | +0.582 | +4.49% | -20.08% |

## Current vs Optimized

| Portfolio Proxy | Current Score | Optimized Score | Current Calmar Delta | Optimized Calmar Delta | Current CAGR Delta | Optimized CAGR Delta | Current MaxDD Delta | Optimized MaxDD Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| equity_only | 0.963463 | 0.963463 | +0.688 | +0.688 | +7.92% | +7.92% | -36.03% | -36.03% |
| overlay_qqq_gld_60_40 | 0.970106 | 0.970106 | +0.609 | +0.609 | +4.87% | +4.87% | -21.57% | -21.57% |
| overlay_qqq_gld_50_50 | 0.962574 | 0.962574 | +0.573 | +0.573 | +4.00% | +4.00% | -19.96% | -19.96% |
| risk_on_with_hedges | 0.967291 | 0.968002 | +0.647 | +0.764 | +4.36% | +4.82% | -21.39% | -22.59% |
| crisis_stack | 0.933667 | 0.958742 | +0.485 | +0.671 | +2.32% | +3.19% | -11.48% | -12.76% |
| long_short_macro | 0.877391 | 0.949573 | +0.330 | +0.564 | +1.90% | +3.15% | -7.04% | -8.61% |
| defensive_mix | 0.643324 | 0.693341 | +0.049 | +0.089 | +0.53% | +1.36% | -2.57% | -2.78% |
