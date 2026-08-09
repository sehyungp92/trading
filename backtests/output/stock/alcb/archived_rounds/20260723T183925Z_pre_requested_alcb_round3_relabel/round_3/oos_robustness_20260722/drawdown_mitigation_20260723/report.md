# ALCB Round 2 drawdown mitigation and alpha recovery

Diagnostic-only repaired-cache research. OOS has been consumed and is not a fresh lockbox.

- Atomic candidates: 118
- Orthogonal follow-up combinations: 29
- Baseline: RVOL 1.70 / OR 9 / late trail distance 0.04.

## Leading aggregate candidates

| Candidate | Category | IS R | IS freq | IS PF | IS DD | OOS R | OOS freq | OOS PF | OOS DD | Guardrails |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| reclaim__or_avwap__structure_stop__buffer_0p0025 | entry_expansion | 286.77 | 69.57 | 1.66 | 12.57% | 55.83 | 58.85 | 3.05 | 1.71% | fail |
| combo__entry_expansion__capacity | orthogonal_combination | 279.73 | 62.09 | 1.84 | 10.57% | 50.47 | 59.35 | 2.94 | 2.82% | pass |
| combo_balanced__selection30__flow8 | balanced_geometry_combination | 255.66 | 57.33 | 1.87 | 11.37% | 56.29 | 56.82 | 3.54 | 1.89% | pass |
| reclaim__or_avwap__structure_stop | entry_expansion | 279.36 | 69.48 | 1.63 | 13.45% | 54.30 | 59.86 | 3.09 | 1.64% | fail |
| combo_balanced__geom125__selection30 | balanced_geometry_combination | 254.84 | 56.03 | 1.89 | 11.38% | 56.82 | 54.79 | 3.59 | 1.98% | pass |
| combo__signal_geometry__selection_expansion | orthogonal_combination | 253.74 | 55.43 | 1.89 | 10.98% | 57.04 | 53.77 | 3.70 | 1.98% | pass |
| combo_balanced__geom125__selection30__mfeoff | balanced_geometry_combination | 254.67 | 56.03 | 1.89 | 11.38% | 56.82 | 54.79 | 3.59 | 1.98% | pass |
| reclaim__or_avwap__structure_stop__minrisk_0p75 | entry_expansion | 295.76 | 69.65 | 1.69 | 13.95% | 50.87 | 59.35 | 3.15 | 1.54% | fail |
| combo__capacity__signal_quality_sizing | orthogonal_combination | 271.29 | 58.63 | 1.88 | 11.44% | 51.58 | 56.82 | 3.30 | 2.21% | pass |
| reclaim__or_avwap__structure_stop__minrisk_0p9 | entry_expansion | 275.83 | 69.65 | 1.69 | 12.66% | 48.77 | 59.86 | 3.38 | 1.44% | fail |
| combo__signal_geometry__signal_quality_sizing__selection_expansion | orthogonal_combination | 253.08 | 55.90 | 1.89 | 10.43% | 57.10 | 55.80 | 3.66 | 1.87% | pass |
| combo_balanced__geom115__selection30__flow8__mfeoff | balanced_geometry_combination | 250.36 | 55.69 | 1.88 | 10.42% | 55.59 | 54.28 | 3.60 | 1.89% | pass |
| combo_balanced__geom115__selection30__flow8 | balanced_geometry_combination | 249.32 | 55.73 | 1.88 | 10.42% | 55.59 | 54.28 | 3.60 | 1.89% | pass |
| capacity__leverage_3p0 | capacity | 269.19 | 58.58 | 1.87 | 12.33% | 50.23 | 55.80 | 3.12 | 2.52% | pass |
| combo_balanced__geom125__quality070__selection30 | balanced_geometry_combination | 253.47 | 56.12 | 1.90 | 10.68% | 55.75 | 56.82 | 3.63 | 1.87% | pass |
| combo__selection_expansion__signal_quality_sizing | orthogonal_combination | 263.18 | 57.55 | 1.88 | 11.24% | 54.97 | 56.82 | 3.48 | 1.87% | pass |
| combo_balanced__geom125__selection30__flow8__mfeoff | balanced_geometry_combination | 251.26 | 56.25 | 1.88 | 10.77% | 55.23 | 54.28 | 3.52 | 1.89% | pass |
| combo_balanced__geom125__selection30__flow8 | balanced_geometry_combination | 250.31 | 56.29 | 1.88 | 10.77% | 55.23 | 54.28 | 3.52 | 1.89% | pass |
| exit__flow_hold_8 | exit_logic | 256.94 | 56.08 | 1.89 | 11.37% | 52.29 | 56.31 | 3.50 | 1.89% | pass |
| reclaim__or_avwap__cpr_0p65 | entry_expansion | 280.31 | 68.31 | 1.74 | 13.02% | 46.83 | 60.37 | 3.31 | 1.47% | fail |
| selection__long_count_30 | selection_expansion | 253.56 | 57.42 | 1.87 | 11.97% | 53.42 | 54.79 | 3.37 | 1.98% | pass |
| reclaim__or_avwap__default | entry_expansion | 266.81 | 69.22 | 1.74 | 12.63% | 46.83 | 60.37 | 3.29 | 1.47% | fail |
| reclaim__or_avwap__cpr_0p55 | entry_expansion | 266.81 | 69.22 | 1.74 | 12.63% | 46.83 | 60.37 | 3.29 | 1.47% | fail |
| reclaim__or_avwap__cpr_0p6 | entry_expansion | 266.81 | 69.22 | 1.74 | 12.63% | 46.83 | 60.37 | 3.29 | 1.47% | fail |
| geometry__entry_range_cap_1p15 | signal_geometry | 242.16 | 54.48 | 1.90 | 11.01% | 55.16 | 53.77 | 3.67 | 1.97% | pass |
| geometry__entry_range_cap_1p25 | signal_geometry | 258.12 | 54.87 | 1.91 | 11.45% | 53.73 | 53.77 | 3.55 | 1.97% | pass |
| combo__signal_geometry__score_discrimination | orthogonal_combination | 243.68 | 54.74 | 1.91 | 10.84% | 54.50 | 53.27 | 3.68 | 1.97% | pass |
| combo_balanced__geom125__adx15 | balanced_geometry_combination | 254.63 | 55.12 | 1.92 | 11.26% | 53.07 | 53.27 | 3.61 | 1.97% | pass |
| combo_balanced__geom125__flow8 | balanced_geometry_combination | 256.16 | 55.04 | 1.90 | 10.89% | 53.59 | 54.79 | 3.47 | 1.89% | pass |
| combo__signal_geometry__quality_risk_restore | orthogonal_combination | 246.24 | 54.39 | 1.92 | 10.67% | 54.21 | 54.28 | 3.66 | 1.87% | pass |

## Segment and drawdown-stress validation

| Candidate | Early R ratio | Late R ratio | Stress R delta | Stress DD ratio | OOS R | OOS freq |
|---|---:|---:|---:|---:|---:|---:|
| combo__entry_expansion__capacity | 1.194 | 0.995 | +3.78 | 0.870 | 50.47 | 59.35 |
| reclaim__or_pdh_avwap__default | 1.112 | 0.961 | +4.88 | 0.814 | 45.71 | 54.28 |
| combo_balanced__geom115__selection30__flow8__mfeoff | 0.947 | 0.945 | +1.77 | 0.875 | 55.59 | 54.28 |
| combo_balanced__geom115__selection30__flow8 | 0.947 | 0.934 | +1.77 | 0.875 | 55.59 | 54.28 |
| combo__signal_geometry__selection_expansion | 0.938 | 0.950 | +1.22 | 0.921 | 57.04 | 53.77 |
| combo__capacity__signal_quality_sizing | 0.888 | 1.029 | +1.06 | 0.907 | 51.58 | 56.82 |
| combo__signal_geometry__signal_quality_sizing__selection_expansion | 0.947 | 0.923 | +1.12 | 0.812 | 57.10 | 55.80 |
| capacity__leverage_3p0 | 0.901 | 1.024 | +1.06 | 1.012 | 50.23 | 55.80 |
| quality__size_floor_0p7 | 1.057 | 0.991 | +1.06 | 0.880 | 51.98 | 56.31 |
| combo__selection_expansion__signal_quality_sizing | 0.904 | 0.988 | +1.06 | 0.880 | 54.97 | 56.82 |
| geometry__entry_range_cap_1p15 | 0.922 | 0.925 | +1.22 | 0.921 | 55.16 | 53.77 |
| combo_balanced__selection30__flow8 | 0.862 | 0.984 | +0.55 | 0.948 | 56.29 | 56.82 |
| quality__hard_min_55p0 | 0.824 | 0.937 | +2.85 | 0.880 | 48.56 | 52.76 |
| reclaim__or_pdh__default | 1.230 | 0.954 | +1.38 | 0.935 | 44.28 | 55.29 |
| combo__signal_geometry__score_discrimination | 0.867 | 0.939 | +1.22 | 0.908 | 54.50 | 53.27 |
| quality__size_floor_0p7__risk_0p00723 | 0.917 | 0.999 | +1.06 | 0.908 | 50.56 | 55.29 |
| exit__flow_hold_8 | 0.923 | 0.959 | +0.55 | 0.948 | 52.29 | 56.31 |
| quality__size_floor_0p7__risk_0p00737 | 0.874 | 0.948 | +1.06 | 0.935 | 50.56 | 55.29 |
| combo_balanced__geom125__selection30__flow8 | 0.954 | 0.939 | +0.05 | 0.913 | 55.23 | 54.28 |
| combo_balanced__geom125__selection30__flow8__mfeoff | 0.954 | 0.936 | +0.05 | 0.913 | 55.23 | 54.28 |
| combo_balanced__geom125__selection30 | 0.946 | 0.961 | -0.50 | 0.956 | 56.82 | 54.79 |
| combo_balanced__geom125__selection30__mfeoff | 0.946 | 0.965 | -0.50 | 0.956 | 56.82 | 54.79 |
| exit__mfe_conviction_off | 1.000 | 0.997 | +0.00 | 1.000 | 50.52 | 55.29 |
| control__balanced_rvol170_or9_trail004 | 1.000 | 1.000 | +0.00 | 1.000 | 50.52 | 55.29 |
| capacity__max_positions_7 | 0.993 | 0.984 | +0.00 | 1.000 | 50.22 | 55.80 |
| combo_balanced__geom125__flow8 | 0.975 | 0.903 | +0.05 | 0.913 | 53.59 | 54.79 |
| risk__thursday_mult_0p75 | 0.785 | 1.014 | +1.06 | 0.921 | 52.90 | 55.29 |
| selection__long_count_30 | 0.843 | 0.987 | +0.00 | 1.000 | 53.42 | 54.79 |
| risk__industrials_mult_025 | 1.000 | 1.002 | +0.00 | 1.000 | 49.68 | 54.79 |
| combo_balanced__geom125__adx15 | 0.962 | 0.983 | -0.50 | 0.945 | 53.07 | 53.27 |

Automated scores are triage aids, not promotion decisions. Mechanism coherence, local smoothness, segment stability, and complexity are reviewed separately.

No production configuration was modified.
