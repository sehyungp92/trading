# ALCB Round 2 OOS robustness audit

## Executive finding

The repaired-cache replay does **not** reproduce aggregate OOS underperformance. This is a diagnostic-only result because the repository has no accepted frozen direct-RTH bundle.

| Window | Trades | Win rate | Avg R | Total R | PF | Net PnL | Trades/month | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS 2024-03-25..2026-03-01 | 1050 | 57.5% | +0.175 | +183.76 | 1.76 | $19,640.70 | 45.4 | 12.3% |
| OOS 2026-03-02..2026-05-01 | 89 | 65.2% | +0.343 | +30.50 | 2.37 | $1,651.85 | 45.1 | 2.3% |

## Edge-case loss concentration

OOS gross loss is spread across 31 losses. The worst 1/3/5 trades account for 8.0% / 20.5% / 31.0% of gross loss.

Holding-period attribution is much more informative than a tail event: the 30 trades held 0-24 bars contribute -23.72R, while the 59 trades held 25+ bars contribute +54.22R. The weakness is repeated early trade failure, not an unbounded-loss outlier.

The short holdout is also temporally concentrated (2026-03: 20 trades, -1.45R; 2026-04: 63 trades, +32.60R; 2026-05: 6 trades, -0.65R). Almost all aggregate OOS profit comes from April, so the high OOS win rate is not evidence of stable month-to-month performance.

| Worst baseline OOS trade | Entry type | Exit | Hold bars | R | Net PnL |
|---|---|---|---:|---:|---:|
| TER 2026-05-01 | OR_BREAKOUT | CLOSE_STOP | 6 | -1.02 | $-96.36 |
| LRCX 2026-05-01 | OR_BREAKOUT | CLOSE_STOP | 11 | -1.04 | $-81.62 |
| AMAT 2026-04-21 | OR_BREAKOUT | CLOSE_STOP | 11 | -0.77 | $-68.58 |
| MSFT 2026-04-20 | OR_BREAKOUT | CLOSE_STOP | 14 | -1.06 | $-63.92 |
| ADI 2026-04-24 | OR_BREAKOUT | CLOSE_STOP | 12 | -1.08 | $-62.79 |

For the recommended patch, 0-24-bar trades contribute -24.61R across 31 trades and 25+-bar trades contribute +60.60R across 63 trades. This is an opportunity/selection uplift; it does not eliminate the structurally negative short-hold cohort.

## Mutation lineage warning

Only 5 of 35 top-level literal mutation removals change the effective runtime configuration. Most accepted parameter values were later baked into `StrategySettings`, so a naive delete-key ablation falsely reports no effect. This audit therefore uses explicit historical/neutral controls and separately removes every nested sizing-map member. Behavioral coverage is 35/35 cumulative keys.

## Ablation conclusions

The core exit architecture is indispensable: adaptive trailing, its fast-runner activation, and partial takes all suffer large cross-window damage when ablated. Restoring the complete Round 1 delta also fails, so a blanket Round 2 rollback is not supported. Removing only the late adaptive-tightening phase cuts IS to about 66R. The failure stop adds about 7R in-sample but is not the source of an OOS tail.

| Candidate | OOS delta R | OOS delta trades/mo | IS delta R | IS PF | IS trades/mo | IS DD | Guardrail |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ablate__use_adaptive_trail | -8.53 | -1.01 | -215.01 | 0.98 | 43.8 | 22.0% | fail |
| ablate__adaptive_trail_start_bars | -8.53 | -1.01 | -215.01 | 0.98 | 43.8 | 22.0% | fail |
| ablate__adaptive_trail_tighten_bars | -11.36 | +0.00 | -117.67 | 1.29 | 44.7 | 16.1% | fail |
| ablate__fr_trailing_activate_r | -18.48 | +9.64 | -193.87 | 1.03 | 48.7 | 26.8% | fail |
| ablate__use_partial_takes | -3.75 | +2.03 | -38.27 | 1.63 | 46.0 | 12.7% | fail |
| restore__round1_exact_delta | -3.41 | -0.51 | -36.24 | 1.60 | 45.6 | 13.7% | fail |
| ablate__combined_avwap_cap_pct | +3.51 | +9.64 | -41.30 | 1.49 | 55.9 | 16.3% | fail |
| ablate__use_combined_quality_gate | +6.16 | +12.18 | -31.22 | 1.45 | 59.8 | 16.6% | fail |
| ablate__failure_stop_bars | +0.13 | -0.51 | -7.24 | 1.77 | 45.1 | 12.4% | fail |
| ablate__flow_reversal_min_hold_bars | -2.82 | +5.58 | -12.70 | 1.66 | 47.0 | 10.0% | fail |
| ablate__carry_min_cpr | +0.00 | +0.00 | +0.00 | 1.76 | 45.4 | 12.3% | pass |
| ablate__carry_min_r | -0.08 | +0.00 | -0.07 | 1.76 | 45.4 | 12.3% | pass |
| ablate__fr_cpr_threshold | +0.00 | +0.00 | +4.12 | 1.77 | 45.3 | 12.3% | pass |
| ablate__use_mfe_conviction_exit | +0.00 | +0.00 | +4.00 | 1.77 | 45.4 | 12.3% | pass |
| ablate__entry_detail_size__or_breakout_5_not_bar_vol_surge | +0.00 | +0.00 | -1.35 | 1.76 | 45.4 | 12.3% | pass |
| ablate__combined_breakout_score_min | +0.00 | +0.00 | +3.50 | 1.77 | 45.4 | 12.3% | pass |

Several accepted micro-mutations are low-value rather than catastrophic. The carry CPR threshold is an exact no-op, the carry-R floor is nearly flat, and the flow-reversal CPR gate, MFE-conviction exit, combined-breakout score floor, and individual score/detail map entries move full-history results only a few R. They are simplification candidates, but removing them does not explain or fix the claimed OOS gap.

## Perturbation stability and rejected OOS fits

The strongest raw OOS winners mostly relax entry filters, but the aggressive versions fail the historical risk guardrail. Removing the quality gate, removing the OR-width floor, loosening the combined AVWAP cap, and RVOL 1.8 are holdout-favored fits. Historical RVOL 1.5 raises IS return/frequency to 202R/58.8 trades per month but misses the PF retention floor at 1.62, making it an aggressive near-miss rather than a robust recommendation. Nearby milder values are stable: RVOL 1.9, OR width 0.10%, opening range 8-9 bars, and late-trail distance 0.08R all pass.

| Candidate | OOS delta R | OOS delta trades/mo | IS delta R | IS PF | IS trades/mo | IS DD | Guardrail |
|---|---:|---:|---:|---:|---:|---:|:---:|
| perturb__adaptive_trail_late_distance_r__0p08 | +1.54 | +0.00 | +17.89 | 1.83 | 45.6 | 11.5% | pass |
| perturb__opening_range_bars__8 | +2.66 | +1.01 | +10.76 | 1.81 | 47.4 | 11.1% | pass |
| perturb__opening_range_bars__9 | +3.25 | +1.01 | +10.37 | 1.83 | 48.1 | 10.9% | pass |
| perturb__or_width_min_pct__0p001 | +3.43 | +1.52 | -2.45 | 1.73 | 49.5 | 13.0% | pass |
| perturb__rvol_threshold__1p9 | +5.50 | +2.54 | +7.84 | 1.76 | 48.3 | 13.0% | pass |
| perturb__rvol_threshold__1p8 | +13.29 | +6.59 | +28.08 | 1.73 | 50.6 | 14.5% | fail |
| ablate__rvol_threshold | +14.54 | +9.64 | +18.41 | 1.62 | 58.8 | 13.4% | fail |
| ablate__use_or_width_min | +13.70 | +6.59 | +38.37 | 1.63 | 67.6 | 16.6% | fail |

## Targeted post-diagnostic phase

The targeted phase deliberately avoids symbol, sector, or date exclusions. Its robust frontier is driven by RVOL 1.9 plus modest sizing/timing refinements; permissive combined-breakout recipes remain OOS-only fits.

| Candidate | OOS delta R | OOS delta trades/mo | IS delta R | IS PF | IS trades/mo | IS DD | Guardrail |
|---|---:|---:|---:|---:|---:|---:|:---:|
| target__rvol190_pdh090 | +5.50 | +2.54 | +11.42 | 1.74 | 47.9 | 13.3% | pass |
| target__rvol190_score_gradient | +5.13 | +2.03 | +9.26 | 1.78 | 47.6 | 13.4% | pass |
| target__entry1300_rvol19 | +4.51 | +3.55 | +11.91 | 1.76 | 50.6 | 12.5% | pass |
| combo__target__rvol190_pdh090__plus__target__entry1300_rvol19 | +4.29 | +4.06 | +15.49 | 1.77 | 50.5 | 12.7% | pass |
| target__rvol190_failstop8 | +5.87 | +4.06 | +4.49 | 1.71 | 48.3 | 12.0% | pass |
| target__combined_rvol22_cap005_score5 | +7.64 | +7.61 | -13.99 | 1.59 | 51.3 | 14.4% | fail |

## Top screened configurations

| Candidate | Stage | OOS utility | OOS uplift | IS checked | IS guardrail | OOS total R | OOS trades/mo | OOS PF |
|---|---|---:|:---:|:---:|:---:|---:|---:|---:|
| ablate__use_combined_quality_gate | ablation | +0.347 | yes | yes | no | +36.66 | 57.3 | 3.42 |
| ablate__rvol_threshold | ablation | +0.336 | yes | yes | no | +45.04 | 54.8 | 3.01 |
| ablate__or_width_min_pct | ablation | +0.323 | yes | yes | no | +44.19 | 51.7 | 2.86 |
| ablate__use_or_width_min | ablation | +0.323 | yes | yes | no | +44.19 | 51.7 | 2.86 |
| perturb__rvol_threshold__1p8 | perturbation | +0.308 | yes | yes | no | +43.79 | 51.7 | 3.02 |
| perturb__combined_avwap_cap_pct__0p005 | perturbation | +0.219 | yes | yes | no | +38.34 | 50.7 | 2.96 |
| ablate__combined_avwap_cap_pct | ablation | +0.209 | yes | yes | no | +34.01 | 54.8 | 2.95 |
| target__combined_rvol22_cap005_score5 | targeted | +0.187 | yes | yes | no | +38.14 | 52.8 | 2.80 |
| combo__target__rvol190_pdh090__plus__target__rvol190_failstop8 | combination | +0.170 | yes | yes | yes | +36.37 | 49.2 | 2.63 |
| combo__target__rvol190_pdh090__plus__target__rvol190_failstop12 | combination | +0.164 | yes | yes | no | +36.79 | 47.7 | 2.67 |
| combo__target__rvol190_pdh090__plus__target__rvol190_score_gradient | combination | +0.159 | yes | yes | no | +35.99 | 47.7 | 2.62 |
| target__rvol190_pdh090 | targeted | +0.156 | yes | yes | yes | +35.99 | 47.7 | 2.66 |
| combo__target__rvol190_pdh090__plus__perturb__rvol_threshold__1p9 | combination | +0.156 | yes | yes | yes | +35.99 | 47.7 | 2.66 |
| target__rvol190_failstop8 | targeted | +0.139 | yes | yes | yes | +36.37 | 49.2 | 2.49 |
| combo__target__rvol190_failstop8__plus__perturb__rvol_threshold__1p9 | combination | +0.139 | yes | yes | yes | +36.37 | 49.2 | 2.49 |

## Exploratory recommendation

`target__rvol190_pdh090` is the highest balanced-score exploratory configuration among candidates that strictly raise OOS total R, net PnL, and frequency and pass the predefined IS degradation limits. It is **not approved for promotion** because the targeted search inspected this OOS sample and the authoritative bundle is absent.

Effective delta: `param_overrides.rvol_threshold` = `1.9`, `param_overrides.pdh_size_mult` = `0.9`.

| Configuration | Window | Trades | Win rate | Avg R | Total R | PF | Net PnL | Trades/month | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recommended | IS | 1107 | 56.6% | +0.176 | +195.18 | 1.74 | $20,323.30 | 47.9 | 13.3% |
| recommended | OOS | 94 | 66.0% | +0.383 | +35.99 | 2.66 | $2,099.29 | 47.7 | 2.0% |

## Statistical interpretation

- Atomic ablations answer mutation dependence; perturbations test local stability; targeted and pairwise candidates are exploratory searches.
- The OOS sample is only about two months. Reusing it for targeted design consumes the lockbox and creates selection bias.
- Symbol/sector/day exclusions are intentionally absent: they are the easiest route to small-sample overfit.
- Before any production change, rebuild the missing frozen direct-RTH bundle and rerun the recommended configuration on a fresh later lockbox.

## Artifacts

See `literal_removal_audit.json`, `lineage_coverage.json`, `baseline_diagnostics.json`, `oos_screen.json`, `is_validation.json`, `combination_oos.json`, `combination_is.json`, `all_results.csv`, `robust_eligible.json`, `recommended_config.json`, and `recommended_oos_diagnostics.json` in this directory.
