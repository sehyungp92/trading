# Granular Accepted-Key Holdout Ablation

Accepted keys tested: 4; subsets tested: 15.
Holdout-negative subsets: 0.
Holdout-changing subsets: 0.

Baseline holdout: trades 10, net +11.00%, avgR +0.661, dollar PF 3.278, QQQ excellent 0.

## Subset Deltas
- QQQ.asset_context_min_score: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +10.32%, trades +2, avgR +0.004, PF +0.056, QQQ excellent +2.
- GLD.min_ma50_slope_atr_4h: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +1.40%, trades -6, avgR +0.044, PF +0.054, QQQ excellent +0.
- GLD.min_ma100_slope_atr_4h: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0.
- GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0.
- QQQ.asset_context_min_score + GLD.min_ma50_slope_atr_4h: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +11.79%, trades -4, avgR +0.048, PF +0.111, QQQ excellent +2.
- QQQ.asset_context_min_score + GLD.min_ma100_slope_atr_4h: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +10.32%, trades +2, avgR +0.004, PF +0.056, QQQ excellent +2.
- QQQ.asset_context_min_score + GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +10.32%, trades +2, avgR +0.004, PF +0.056, QQQ excellent +2.
- GLD.min_ma50_slope_atr_4h + GLD.min_ma100_slope_atr_4h: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +1.40%, trades -6, avgR +0.044, PF +0.054, QQQ excellent +0.
- GLD.min_ma50_slope_atr_4h + GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +1.40%, trades -6, avgR +0.044, PF +0.054, QQQ excellent +0.
- GLD.min_ma100_slope_atr_4h + GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0.
- QQQ.asset_context_min_score + GLD.min_ma50_slope_atr_4h + GLD.min_ma100_slope_atr_4h: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +11.79%, trades -4, avgR +0.048, PF +0.111, QQQ excellent +2.
- QQQ.asset_context_min_score + GLD.min_ma50_slope_atr_4h + GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +11.79%, trades -4, avgR +0.048, PF +0.111, QQQ excellent +2.
- QQQ.asset_context_min_score + GLD.min_ma100_slope_atr_4h + GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +10.32%, trades +2, avgR +0.004, PF +0.056, QQQ excellent +2.
- GLD.min_ma50_slope_atr_4h + GLD.min_ma100_slope_atr_4h + GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +1.40%, trades -6, avgR +0.044, PF +0.054, QQQ excellent +0.
- QQQ.asset_context_min_score + GLD.min_ma50_slope_atr_4h + GLD.min_ma100_slope_atr_4h + GLD.require_di_alignment: holdout net +0.00%, trades +0, avgR +0.000, PF +0.000, QQQ excellent +0; train net +11.79%, trades -4, avgR +0.048, PF +0.111, QQQ excellent +2.

## Decision
No accepted key or accepted-key combination reduced holdout return, holdout expectancy, holdout dollar PF, holdout trade count, or holdout QQQ excellent-trade count versus corrected round 6. No removal is warranted on holdout-impact grounds.
The GLD MA100 and DI overrides are redundant with existing all-symbol settings in the corrected baseline; GLD MA50 is the only GLD-specific incremental filter in this set.
