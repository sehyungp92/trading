# ALCB Round 2 targeted follow-up

This is a diagnostic-only, consumed-OOS study. No configuration is promotion-authorized.

Tested 56 evidence-driven candidates on aggregate IS/OOS and 12 finalists across three IS folds.

## Leading fold-validated candidates

| Candidate | Final score | OOS R | OOS TPM | OOS PF | IS R | IS TPM | IS PF | Positive IS folds | Worst fold utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r110__entry1330_late_score5_failure_m010 | +0.2454 | +43.47 | 67.0 | 2.91 | +259.18 | 94.5 | 1.75 | 3/3 | +0.2309 |
| rvol_fine__1p0 | +0.2313 | +38.84 | 60.9 | 3.01 | +271.36 | 88.8 | 1.72 | 3/3 | +0.2753 |
| control__rvol110_entry1330 | +0.2213 | +41.26 | 69.5 | 2.40 | +264.33 | 95.9 | 1.71 | 3/3 | +0.2123 |
| r110__trail_start32 | +0.2170 | +41.79 | 53.3 | 3.01 | +265.84 | 80.4 | 1.79 | 3/3 | +0.2132 |
| r110__entry1330_late_rvol_add0p1 | +0.1989 | +41.06 | 57.3 | 3.14 | +243.76 | 82.3 | 1.76 | 3/3 | +0.1786 |
| r110__entry1330_late_rvol_add0p05 | +0.1910 | +39.70 | 60.9 | 2.89 | +251.21 | 88.1 | 1.72 | 3/3 | +0.1945 |
| r110__trail30_failure_m010 | +0.1884 | +42.11 | 54.8 | 3.12 | +247.96 | 81.7 | 1.78 | 3/3 | +0.1366 |
| r110__failure_bars6_to_m010 | +0.1689 | +42.68 | 59.4 | 3.33 | +231.34 | 86.3 | 1.75 | 3/3 | +0.0335 |
| control__rvol110 | +0.1529 | +39.64 | 54.8 | 2.94 | +243.77 | 82.3 | 1.73 | 3/3 | +0.1582 |
| control__rvol110_failure_m010 | +0.1468 | +40.43 | 56.8 | 3.04 | +233.99 | 83.4 | 1.75 | 3/3 | +0.0730 |
| entry1330__failure_m010_combined_quality_off | +0.1078 | +41.54 | 62.9 | 3.15 | +196.09 | 79.9 | 1.64 | 2/3 | -0.0131 |
| control__entry1330_failure_m010 | +0.0754 | +41.09 | 55.3 | 3.32 | +196.33 | 74.2 | 1.67 | 3/3 | +0.0439 |

## Interpretation rule

Prefer the Pareto set and fold stability over the scalar rank. A high OOS score is not sufficient when it comes from lower expectancy, PF collapse, or one IS subperiod.

See `aggregate_results.json`, `pareto_return_frequency.json`, `pareto_quality.json`, `fold_results.json`, `finalists.json`, and `finalist_oos_diagnostics.json`.
