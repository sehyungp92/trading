# ALCB Round 2 OOS robustness audit

## Executive finding

The repaired-cache replay does **not** reproduce aggregate OOS underperformance. This is a diagnostic-only result because the repository has no accepted frozen direct-RTH bundle.

| Window | Trades | Win rate | Avg R | Total R | PF | Net PnL | Trades/month | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS 2024-03-25..2026-03-01 | 1473 | 57.0% | +0.120 | +177.49 | 1.67 | $17,370.08 | 63.7 | 3.9% |
| OOS 2026-03-02..2026-05-01 | 94 | 63.8% | +0.403 | +37.89 | 3.14 | $1,947.17 | 47.7 | 1.5% |

## Edge-case loss concentration

OOS gross loss is spread across 34 losses. The worst 1/3/5 trades account for 9.7% / 23.0% / 32.6% of gross loss.

Holding-period attribution is much more informative than a tail event: the 33 trades held 0-24 bars contribute -13.58R, while the 61 trades held 25+ bars contribute +51.47R. The weakness is repeated early trade failure, not an unbounded-loss outlier.

The short holdout is also temporally concentrated (2026-03: 16 trades, +3.01R; 2026-04: 73 trades, +36.89R; 2026-05: 5 trades, -2.01R). Almost all aggregate OOS profit comes from April, so the high OOS win rate is not evidence of stable month-to-month performance.

| Worst baseline OOS trade | Entry type | Exit | Hold bars | R | Net PnL |
|---|---|---|---:|---:|---:|
| AMZN 2026-05-01 | OR_BREAKOUT | FLOW_REVERSAL | 10 | -0.74 | $-87.99 |
| MPWR 2026-05-01 | OR_BREAKOUT | FLOW_REVERSAL | 8 | -0.77 | $-74.66 |
| GOOG 2026-04-29 | COMBINED_BREAKOUT | FAILURE_STOP | 12 | -0.59 | $-46.90 |
| NFLX 2026-03-05 | PDH_BREAKOUT | CLOSE_STOP | 4 | -1.06 | $-46.33 |
| CAT 2026-04-21 | COMBINED_BREAKOUT | FLOW_REVERSAL | 8 | -0.80 | $-41.04 |

For the recommended patch, 0-24-bar trades contribute -17.70R across 41 trades and 25+-bar trades contribute +57.33R across 67 trades. This is an opportunity/selection uplift; it does not eliminate the structurally negative short-hold cohort.

## Mutation lineage warning

Only 8 of 50 top-level literal mutation removals change the effective runtime configuration. Most accepted parameter values were later baked into `StrategySettings`, so a naive delete-key ablation falsely reports no effect. This audit therefore uses explicit historical/neutral controls and separately removes every nested sizing-map member. Behavioral coverage is 50/50 cumulative keys.

## Ablation conclusions

The core exit architecture is indispensable: adaptive trailing, its fast-runner activation, and partial takes all suffer large cross-window damage when ablated. Restoring the complete Round 1 delta also fails, so a blanket Round 2 rollback is not supported. Removing only the late adaptive-tightening phase cuts IS to about 66R. The failure stop adds about 7R in-sample but is not the source of an OOS tail.

| Candidate | OOS delta R | OOS delta trades/mo | IS delta R | IS PF | IS trades/mo | IS DD | Guardrail |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ablate__use_adaptive_trail | -1.49 | -6.59 | -279.46 | 0.80 | 59.1 | 36.8% | fail |
| ablate__adaptive_trail_start_bars | -1.49 | -6.59 | -279.46 | 0.80 | 59.1 | 36.8% | fail |
| ablate__adaptive_trail_tighten_bars | -8.54 | -1.52 | -207.47 | 0.94 | 63.0 | 18.6% | fail |
| ablate__fr_trailing_activate_r | -24.36 | +13.19 | -206.32 | 0.97 | 71.1 | 14.9% | fail |
| ablate__use_partial_takes | -2.72 | +0.00 | -4.54 | 1.64 | 63.9 | 3.8% | fail |
| restore__round1_exact_delta | -4.47 | -0.51 | -100.87 | 1.30 | 63.2 | 4.7% | fail |
| ablate__combined_avwap_cap_pct | +0.49 | -3.04 | -0.79 | 1.72 | 56.4 | 3.7% | fail |
| ablate__use_combined_quality_gate | +0.96 | +6.09 | +2.36 | 1.65 | 68.8 | 3.1% | pass |
| ablate__failure_stop_bars | -0.23 | -0.51 | +5.36 | 1.66 | 63.1 | 3.9% | pass |
| ablate__flow_reversal_min_hold_bars | -5.29 | +1.01 | +10.08 | 1.71 | 65.1 | 3.5% | pass |
| ablate__carry_min_cpr | +0.00 | +0.00 | +0.00 | 1.67 | 63.7 | 3.9% | pass |
| ablate__carry_min_r | +0.00 | +0.00 | -2.26 | 1.65 | 63.6 | 3.9% | pass |
| ablate__fr_cpr_threshold | +0.06 | +0.00 | -1.68 | 1.65 | 63.7 | 3.9% | pass |
| ablate__use_mfe_conviction_exit | +0.00 | +0.00 | +2.93 | 1.67 | 63.6 | 3.9% | pass |
| ablate__combined_breakout_score_min | +0.00 | +0.00 | +4.92 | 1.67 | 64.1 | 3.8% | pass |

Several accepted micro-mutations are low-value rather than catastrophic. The carry CPR threshold is an exact no-op, the carry-R floor is nearly flat, and the flow-reversal CPR gate, MFE-conviction exit, combined-breakout score floor, and individual score/detail map entries move full-history results only a few R. They are simplification candidates, but removing them does not explain or fix the claimed OOS gap.

## Perturbation stability and rejected OOS fits

The strongest raw OOS winners mostly relax entry filters, but the aggressive versions fail the historical risk guardrail. Removing the quality gate, removing the OR-width floor, loosening the combined AVWAP cap, and RVOL 1.8 are holdout-favored fits. Historical RVOL 1.5 raises IS return/frequency to 202R/58.8 trades per month but misses the PF retention floor at 1.62, making it an aggressive near-miss rather than a robust recommendation. Nearby milder values are stable: RVOL 1.9, OR width 0.10%, opening range 8-9 bars, and late-trail distance 0.08R all pass.

| Candidate | OOS delta R | OOS delta trades/mo | IS delta R | IS PF | IS trades/mo | IS DD | Guardrail |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ablate__rvol_threshold | -2.35 | -5.07 | -22.36 | 1.61 | 58.3 | 3.4% | fail |
| ablate__use_or_width_min | +0.00 | +0.00 | +0.00 | 1.67 | 63.7 | 3.9% | pass |

## Targeted post-diagnostic phase

The targeted phase deliberately avoids symbol, sector, or date exclusions. Its robust frontier is driven by RVOL 1.9 plus modest sizing/timing refinements; permissive combined-breakout recipes remain OOS-only fits.

| Candidate | OOS delta R | OOS delta trades/mo | IS delta R | IS PF | IS trades/mo | IS DD | Guardrail |
|---|---:|---:|---:|---:|---:|---:|:---:|

## Top screened configurations

| Candidate | Stage | OOS utility | OOS uplift | IS checked | IS guardrail | OOS total R | OOS trades/mo | OOS PF |
|---|---|---:|:---:|:---:|:---:|---:|---:|---:|
| target__or5_rvol150 | targeted | +0.164 | yes | yes | no | +45.81 | 55.3 | 3.11 |
| target__or5_base_risk0065 | targeted | +0.113 | no | yes | no | +44.86 | 60.4 | 2.76 |
| target__or5_entry1200 | targeted | +0.092 | yes | yes | no | +41.23 | 54.8 | 2.95 |
| target__or5_score67_sizing | targeted | +0.083 | no | yes | no | +40.61 | 60.9 | 2.80 |
| target__or5_daily300_score67 | targeted | +0.083 | no | yes | no | +40.61 | 60.9 | 2.80 |
| combo__perturb__failure_stop_to_r__m0p1__plus__perturb__entry_window_end__1330 | combination | +0.072 | yes | yes | yes | +41.09 | 55.3 | 3.32 |
| perturb__opening_range_bars__5 | perturbation | +0.067 | no | yes | no | +40.52 | 59.9 | 2.62 |
| target__or5_daily_stop300 | targeted | +0.067 | no | yes | no | +40.52 | 59.9 | 2.62 |
| target__or5_daily_stop250 | targeted | +0.067 | no | yes | no | +40.52 | 59.9 | 2.62 |
| target__or5_orb_cap125 | targeted | +0.067 | no | yes | no | +40.52 | 59.9 | 2.62 |
| perturb__pdh_avwap_cap_pct__0p006 | perturbation | +0.064 | no | yes | no | +40.54 | 41.6 | 3.54 |
| perturb__adaptive_trail_start_bars__30 | perturbation | +0.059 | no | yes | yes | +40.60 | 45.7 | 3.41 |
| combo__perturb__entry_window_end__1330__plus__perturb__mfe_conviction_floor_r__0p0 | combination | +0.057 | yes | yes | yes | +40.45 | 55.8 | 3.15 |
| combo__perturb__failure_stop_to_r__m0p1__plus__perturb__rvol_threshold__1p1 | combination | +0.055 | yes | yes | yes | +40.43 | 56.8 | 3.04 |
| ablate__pdh_avwap_cap_pct | ablation | +0.054 | no | yes | no | +39.54 | 39.1 | 3.63 |

## Exploratory recommendation

`perturb__rvol_threshold__1p1` is the highest balanced-score exploratory configuration among candidates that strictly raise OOS total R, net PnL, and frequency and pass the predefined IS degradation limits. It is **not approved for promotion** because the targeted search inspected this OOS sample and the authoritative bundle is absent.

Effective delta: `param_overrides.rvol_threshold` = `1.1`.

| Configuration | Window | Trades | Win rate | Avg R | Total R | PF | Net PnL | Trades/month | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recommended | IS | 1904 | 57.1% | +0.128 | +243.77 | 1.73 | $26,103.18 | 82.3 | 4.4% |
| recommended | OOS | 108 | 61.1% | +0.367 | +39.64 | 2.94 | $2,100.27 | 54.8 | 1.4% |

## Statistical interpretation

- Atomic ablations answer mutation dependence; perturbations test local stability; targeted and pairwise candidates are exploratory searches.
- The OOS sample is only about two months. Reusing it for targeted design consumes the lockbox and creates selection bias.
- Symbol/sector/day exclusions are intentionally absent: they are the easiest route to small-sample overfit.
- Before any production change, rebuild the missing frozen direct-RTH bundle and rerun the recommended configuration on a fresh later lockbox.

## Artifacts

See `literal_removal_audit.json`, `lineage_coverage.json`, `baseline_diagnostics.json`, `oos_screen.json`, `is_validation.json`, `combination_oos.json`, `combination_is.json`, `all_results.csv`, `robust_eligible.json`, `recommended_config.json`, and `recommended_oos_diagnostics.json` in this directory.
