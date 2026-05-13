# IARIC OOS Ablation and Perturbation Diagnostic

- Generated: 2026-05-04T16:44:58.272335+00:00
- OOS window: 2026-03-21 to 2026-05-01
- IS window: 2024-01-01 to 2026-03-20
- Candidates evaluated on OOS: 117
- Candidates evaluated on IS: 117

## Current OOS Failure Shape

Current champion: 12 OOS trades, 0.449 PF, -0.144 avgR, -1.725 netR.

Largest OOS losses:
- 2026-04-22 MCK -1.022R, STOP_HIT, sector=Healthcare, score=79.05, rank_pct=100.0, gap=-0.176%, sma_dist=-7.652%
- 2026-04-28 HSIC -0.723R, STOP_HIT, sector=Healthcare, score=75.04, rank_pct=100.0, gap=0.466%, sma_dist=0.488%
- 2026-04-20 NFLX -0.716R, STOP_HIT, sector=Communication Services, score=82.05, rank_pct=50.0, gap=-0.195%, sma_dist=5.825%
- 2026-04-21 NFLX -0.415R, GAP_STOP, sector=Communication Services, score=72.05, rank_pct=100.0, gap=-0.939%, sma_dist=2.816%
- 2026-05-01 MSFT -0.161R, EOD_FLATTEN, sector=Technology, score=86.03, rank_pct=50.0, gap=1.231%, sma_dist=3.029%

Bad buckets:
- sector=Healthcare: n=2, netR=-1.745, avgR=-0.873
- sector=Communication Services: n=3, netR=-0.948, avgR=-0.316
- sector=Technology: n=7, netR=+1.087, avgR=+0.155
- exit_reason=STOP_HIT: n=5, netR=-1.475, avgR=-0.295
- exit_reason=GAP_STOP: n=2, netR=-0.507, avgR=-0.254
- exit_reason=EOD_FLATTEN: n=2, netR=-0.010, avgR=-0.005
- regime=A: n=12, netR=-1.606, avgR=-0.134

## Best IS-Checked Candidates

| Candidate | Group | Eligible | OOS trades | OOS netR | OOS avgR | IS trades | IS netR | IS avgR | Patch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| target_floor70_gap25_slots6 | targeted_combo | True | 19 | 1.322 | 0.070 | 1121 | 104.400 | 0.093 | `{"param_overrides.pb_v2_gap_max_pct": 2.5, "param_overrides.pb_v2_open_scored_max_slots": 6, "param_overrides.pb_v2_signal_floor": 70.0}` |
| sweep_signal_floor_70p0 | single_perturbation | True | 19 | 1.322 | 0.070 | 1103 | 106.608 | 0.097 | `{"param_overrides.pb_v2_signal_floor": 70.0}` |
| target_floor70_slots6 | targeted_combo | True | 19 | 1.322 | 0.070 | 1103 | 106.608 | 0.097 | `{"param_overrides.pb_v2_open_scored_max_slots": 6, "param_overrides.pb_v2_signal_floor": 70.0}` |
| target_sector3_floor70_slots6 | targeted_combo | True | 19 | 1.322 | 0.070 | 1056 | 103.357 | 0.098 | `{"max_per_sector": 3, "param_overrides.pb_v2_open_scored_max_slots": 6, "param_overrides.pb_v2_signal_floor": 70.0}` |
| target_floor70_rank75 | targeted_combo | True | 18 | 1.198 | 0.067 | 1041 | 99.205 | 0.095 | `{"param_overrides.pb_entry_rank_pct_max": 75.0, "param_overrides.pb_v2_signal_floor": 70.0}` |
| target_floor70_carry50_mfe20 | targeted_combo | True | 19 | 0.827 | 0.044 | 1105 | 105.156 | 0.095 | `{"param_overrides.pb_open_scored_carry_close_pct_min": 0.5, "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.2, "param_overrides.pb_v2_signal_floor": 70.0}` |
| sweep_partial_trigger_0p1 | single_perturbation | True | 12 | 0.666 | 0.055 | 961 | 167.885 | 0.175 | `{"param_overrides.pb_v2_partial_profit_trigger_r": 0.1}` |
| sweep_gap_min_neg5p0 | single_perturbation | True | 12 | -1.702 | -0.142 | 940 | 99.244 | 0.106 | `{"param_overrides.pb_v2_gap_min_pct": -5.0}` |
| disable_delayed_confirm | single_perturbation | True | 12 | -1.702 | -0.142 | 929 | 99.000 | 0.107 | `{"param_overrides.pb_delayed_confirm_enabled": false}` |
| sweep_signal_floor_66p0 | single_perturbation | False | 34 | 3.032 | 0.089 | 1459 | 123.879 | 0.085 | `{"param_overrides.pb_v2_signal_floor": 66.0}` |
| sweep_signal_floor_68p0 | single_perturbation | False | 25 | 2.613 | 0.105 | 1291 | 109.427 | 0.085 | `{"param_overrides.pb_v2_signal_floor": 68.0}` |
| target_floor68_rank75_slots6 | targeted_combo | False | 23 | 2.023 | 0.088 | 1220 | 101.672 | 0.083 | `{"param_overrides.pb_entry_rank_pct_max": 75.0, "param_overrides.pb_v2_open_scored_max_slots": 6, "param_overrides.pb_v2_signal_floor": 68.0}` |

## Interpretation

At least one diagnostic candidate improved OOS netR and held OOS frequency without breaching the 90% IS PF/netR and 85% IS trade-retention checks.
Best eligible candidate: `target_floor70_gap25_slots6` with OOS 19 trades / 1.322 netR and IS 104.400 netR.
The current OOS loss is not a single catastrophic print; it is a small-sample cluster of normal-sized mean-reversion failures concentrated in late April, especially Healthcare and Communication Services.
