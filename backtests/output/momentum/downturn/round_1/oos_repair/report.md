# Downturn Round 1 OOS Repair

Disposition: **SHADOW_RESEARCH_ONLY**. The specified OOS interval is no longer untouched OOS because this round explicitly examined and optimized against the 2026-03-21 through 2026-05-01 interval. A new future holdout is required before promotion.

| Configuration | IS trades | IS return | IS PF | IS DD | Validation trades | Validation return | Validation PF | Validation WR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 107 | 102.02% | 2.94 | 3.60% | 7 | 12.83% | 13.93 | 85.71% |
| Shadow recommendation | 127 | 107.08% | 2.76 | 4.81% | 9 | 20.45% | 11.85 | 77.78% |

## Root cause

The reported severe OOS loss used the wrong interval (2026-05-02 onward). On the specified 2026-03-21 through 2026-05-01 OOS, the frozen baseline is profitable with six winners and one loser.

The specified baseline OOS contains seven trades across three active days: six winners and one -$99.24 loser. There is no catastrophic-loss cluster to repair in this interval; its weakness is low sample size and frequency.

Selected candidate: `targeted:adx21+chandelier_32+adx_range_12`

Unique configurations compared: 525

Strict repair-gate passes: 169

IS frequency: 4.02 -> 4.77 trades/month.

OOS frequency: 5.07 -> 6.52 trades/month.

## Stability and execution

- Local surface: 60/90 strict passes; all 90 points had positive OOS PnL.
- Selected OOS bootstrap probability of positive net PnL: 99.4%.
- Selected OOS bootstrap 95% PnL interval: $332 to $4,265.
- One-bar entry latency remains the main fragility; see `summary.json` for the matched candidate/baseline stress table.

| Stress | Candidate IS | Baseline IS | Candidate OOS | Baseline OOS |
|---|---:|---:|---:|---:|
| commission_2x | 105.50% | 100.70% | 20.34% | 12.75% |
| slippage_3ticks | 101.46% | 97.97% | 20.13% | 12.54% |
| entry_latency_1bar | 66.94% | 59.75% | 8.86% | 13.74% |
| combined_execution | 61.46% | 55.25% | 8.71% | 13.28% |

## Mutation delta

- `param_overrides.adx_range_threshold`: `18` -> `12`
- `param_overrides.adx_trending_threshold`: `20` -> `21.0`
- `param_overrides.chandelier_lookback`: `22` -> `32`

## Ablation findings

- Exact-default redundancies: `flags.chandelier_trailing=True`, `flags.progressive_sma=True`, `flags.vol_percentile_gate=0`, and `param_overrides.ema_fast_period=20` do not add behavior over defaults.
- Sample-path no-ops: `base_risk_pct` is floor-dominated at one contract; `divergence_mag_threshold` is dormant because reversal contributes zero trades; `regime_mult_counter` is dormant while counter entries are blocked.
- These functional no-ops should be documented or pruned only with care: they can become active if contract sizing, reversal, or counter-regime policies change later.
- The selected ADX/chandelier interaction is supported by the local surface, not by one isolated parameter point.

## Limitations

- OOS has only nine selected-candidate trades. The bootstrap PnL interval is positive, but the net-R interval still crosses zero; uncertainty remains material.
- Because OOS was examined and used for selection, it is now validation. The recommendation must remain shadow-only until fresh future data accrues.
- Entry latency materially reduces both configurations and should be monitored in paper/shadow execution.

## Interpretation

See `baseline_attribution.json` for trade-level loss concentration, `historical_ablation.json` for every cumulative/atomic historical test, `perturbation.json` for all numeric neighbourhoods, `targeted.json` for the additional robustness mechanisms, and `verification.json` for the final local surface and execution stresses.
