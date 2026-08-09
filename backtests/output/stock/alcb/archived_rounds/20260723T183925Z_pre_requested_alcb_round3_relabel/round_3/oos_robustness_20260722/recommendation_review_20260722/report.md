# ALCB Round 2 recommendation review

## Decision

The original recommendation is dominated by at least one tested candidate on IS/OOS total R, net PnL, and trade frequency under the predefined risk guardrails.
This remains diagnostic-only: the repaired legacy OOS window has been reused and no frozen direct-RTH bundle exists.

Segment-qualified selection: `boundary__rvol1p65__or9__trail0p04`.

## Candidates that dominate the original recommendation

| Candidate | IS R | IS trades/mo | IS PF | IS DD | OOS R | OOS trades/mo | OOS PF | OOS DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| boundary__rvol1p65__or9__trail0p04 | 253.73 | 57.9 | 1.88 | 12.6% | 51.98 | 54.3 | 3.39 | 1.9% |
| boundary__rvol1p75__or9__trail0p04 | 266.55 | 54.7 | 1.92 | 12.2% | 51.00 | 53.3 | 3.38 | 2.0% |
| boundary__rvol1p7__or9__trail0p04 | 265.23 | 55.9 | 1.89 | 11.9% | 50.52 | 55.3 | 3.35 | 2.0% |
| boundary__rvol1p65__or10__trail0p04 | 235.87 | 58.9 | 1.89 | 12.5% | 51.88 | 54.3 | 3.38 | 1.9% |
| boundary__rvol1p75__or10__trail0p04 | 245.85 | 55.7 | 1.94 | 11.9% | 50.90 | 53.3 | 3.38 | 2.0% |
| boundary__rvol1p7__or10__trail0p04 | 239.57 | 57.2 | 1.90 | 11.7% | 50.42 | 55.3 | 3.35 | 2.0% |
| boundary__rvol1p8__or9__trail0p04 | 236.55 | 53.2 | 1.90 | 12.3% | 52.23 | 52.3 | 3.34 | 2.0% |
| boundary__rvol1p8__or10__trail0p04 | 222.82 | 54.1 | 1.93 | 12.1% | 51.89 | 51.7 | 3.34 | 2.0% |
| boundary__rvol1p65__or9__trail0p06 | 242.26 | 57.6 | 1.85 | 13.0% | 49.54 | 55.3 | 3.28 | 1.9% |
| boundary__rvol1p7__or10__trail0p06 | 243.48 | 56.8 | 1.87 | 12.0% | 49.20 | 54.8 | 3.25 | 2.0% |
| stability__rvol1p75__or9__trail0p06 | 238.43 | 54.6 | 1.89 | 12.5% | 49.83 | 52.8 | 3.28 | 2.0% |
| boundary__rvol1p7__or9__trail0p06 | 237.96 | 55.9 | 1.86 | 12.3% | 49.31 | 54.8 | 3.25 | 2.0% |
| stability__rvol1p75__or10__trail0p06 | 229.50 | 55.6 | 1.91 | 12.3% | 49.72 | 52.8 | 3.28 | 2.0% |
| boundary__rvol175__or9__trail06__flowhold8 | 228.92 | 54.8 | 1.87 | 11.9% | 50.58 | 52.8 | 3.37 | 1.9% |
| boundary__rvol175__or10__trail06__flowhold8 | 221.14 | 55.6 | 1.89 | 11.7% | 50.68 | 53.3 | 3.37 | 1.9% |

## Top robust candidates by equal-window, complexity-adjusted utility

| Candidate | Score | IS R | IS trades/mo | IS PF | IS DD | OOS R | OOS trades/mo | OOS PF | OOS DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boundary__rvol1p65__or9__trail0p04 | +0.384 | 253.73 | 57.9 | 1.88 | 12.6% | 51.98 | 54.3 | 3.39 | 1.9% |
| boundary__rvol1p75__or9__trail0p04 | +0.383 | 266.55 | 54.7 | 1.92 | 12.2% | 51.00 | 53.3 | 3.38 | 2.0% |
| boundary__rvol1p7__or9__trail0p04 | +0.380 | 265.23 | 55.9 | 1.89 | 11.9% | 50.52 | 55.3 | 3.35 | 2.0% |
| boundary__rvol1p65__or10__trail0p04 | +0.369 | 235.87 | 58.9 | 1.89 | 12.5% | 51.88 | 54.3 | 3.38 | 1.9% |
| boundary__rvol1p75__or10__trail0p04 | +0.367 | 245.85 | 55.7 | 1.94 | 11.9% | 50.90 | 53.3 | 3.38 | 2.0% |
| boundary__rvol1p7__or10__trail0p04 | +0.359 | 239.57 | 57.2 | 1.90 | 11.7% | 50.42 | 55.3 | 3.35 | 2.0% |
| boundary__rvol1p8__or9__trail0p04 | +0.348 | 236.55 | 53.2 | 1.90 | 12.3% | 52.23 | 52.3 | 3.34 | 2.0% |
| boundary__rvol1p8__or10__trail0p04 | +0.338 | 222.82 | 54.1 | 1.93 | 12.1% | 51.89 | 51.7 | 3.34 | 2.0% |
| boundary__rvol1p65__or9__trail0p06 | +0.334 | 242.26 | 57.6 | 1.85 | 13.0% | 49.54 | 55.3 | 3.28 | 1.9% |
| boundary__rvol1p7__or10__trail0p06 | +0.332 | 243.48 | 56.8 | 1.87 | 12.0% | 49.20 | 54.8 | 3.25 | 2.0% |
| stability__rvol1p75__or9__trail0p06 | +0.324 | 238.43 | 54.6 | 1.89 | 12.5% | 49.83 | 52.8 | 3.28 | 2.0% |
| boundary__rvol1p7__or9__trail0p06 | +0.324 | 237.96 | 55.9 | 1.86 | 12.3% | 49.31 | 54.8 | 3.25 | 2.0% |
| stability__rvol1p75__or10__trail0p06 | +0.321 | 229.50 | 55.6 | 1.91 | 12.3% | 49.72 | 52.8 | 3.28 | 2.0% |
| boundary__rvol175__or9__trail06__flowhold8 | +0.319 | 228.92 | 54.8 | 1.87 | 11.9% | 50.58 | 52.8 | 3.37 | 1.9% |
| boundary__rvol175__or10__trail06__flowhold8 | +0.317 | 221.14 | 55.6 | 1.89 | 11.7% | 50.68 | 53.3 | 3.37 | 1.9% |
| boundary__rvol1p65__or9__trail0p08 | +0.316 | 242.54 | 57.9 | 1.82 | 13.4% | 49.43 | 54.3 | 3.23 | 2.0% |
| boundary__rvol1p65__or10__trail0p06 | +0.313 | 218.70 | 58.9 | 1.86 | 12.8% | 49.43 | 55.3 | 3.28 | 1.9% |
| stability__rvol1p8__or10__trail0p06 | +0.311 | 222.49 | 54.2 | 1.89 | 12.4% | 50.68 | 52.3 | 3.28 | 2.0% |
| stability__rvol1p8__or9__trail0p06 | +0.310 | 226.84 | 53.4 | 1.87 | 12.6% | 50.79 | 52.3 | 3.28 | 2.0% |
| boundary__rvol1p65__or10__trail0p08 | +0.291 | 213.20 | 58.8 | 1.82 | 13.1% | 49.54 | 54.8 | 3.24 | 2.0% |

The four-objective Pareto frontier contains 10 candidate(s); see `pareto_frontier.json`.

## Early/late IS stability

| Finalist | Both pass | Early delta R | Early delta trades/mo | Late delta R | Late delta trades/mo |
|---|:---:|---:|---:|---:|---:|
| control__recommended_rvol190_pdh090 | yes | +1.84 | +2.68 | +13.16 | +3.05 |
| boundary__rvol1p65__or9__trail0p04 | yes | +19.89 | +11.79 | +47.95 | +13.47 |
| boundary__rvol1p75__or9__trail0p04 | yes | +27.60 | +8.61 | +49.11 | +10.77 |
| boundary__rvol1p7__or9__trail0p04 | yes | +31.69 | +9.53 | +54.28 | +11.22 |
| boundary__rvol1p65__or10__trail0p04 | yes | +11.73 | +12.79 | +44.34 | +14.37 |
| boundary__rvol1p75__or10__trail0p04 | yes | +22.96 | +9.37 | +52.63 | +12.21 |
| boundary__rvol1p7__or10__trail0p04 | yes | +14.83 | +10.87 | +48.21 | +12.48 |
| boundary__rvol1p8__or9__trail0p04 | yes | +16.17 | +7.27 | +43.84 | +9.52 |
| boundary__rvol1p8__or10__trail0p04 | yes | +6.84 | +7.61 | +42.51 | +9.79 |
| boundary__rvol1p65__or9__trail0p06 | yes | +19.34 | +11.96 | +38.92 | +13.38 |
| boundary__rvol1p7__or10__trail0p06 | yes | +21.87 | +10.20 | +38.47 | +12.48 |
| stability__rvol1p75__or9__trail0p06 | yes | +12.30 | +8.45 | +46.84 | +10.06 |
| boundary__rvol1p7__or9__trail0p06 | yes | +17.05 | +9.78 | +47.62 | +11.58 |
| stability__rvol1p75__or10__trail0p06 | yes | +8.70 | +9.11 | +42.82 | +11.04 |
| boundary__rvol175__or9__trail06__flowhold8 | yes | +9.24 | +8.70 | +46.76 | +11.13 |
| boundary__rvol175__or10__trail06__flowhold8 | yes | +7.20 | +9.11 | +39.07 | +12.03 |
| boundary__rvol1p65__or9__trail0p08 | yes | +20.60 | +12.04 | +39.17 | +12.84 |
| boundary__rvol1p65__or10__trail0p06 | yes | +4.52 | +12.71 | +35.50 | +13.47 |
| stability__rvol1p8__or10__trail0p06 | yes | +7.82 | +7.94 | +42.29 | +9.70 |
| stability__rvol1p8__or9__trail0p06 | yes | +10.19 | +7.44 | +34.22 | +9.16 |
| boundary__rvol1p65__or10__trail0p08 | yes | -0.18 | +12.88 | +32.37 | +14.01 |
| stability__rvol1p75__or9__trail0p08 | yes | +4.90 | +8.53 | +43.38 | +10.68 |
| boundary__rvol1p7__or9__trail0p08 | yes | +13.38 | +9.78 | +38.73 | +11.76 |
| boundary__rvol1p7__or10__trail0p08 | yes | +11.10 | +10.62 | +34.64 | +12.66 |
| stability__rvol1p75__or10__trail0p08 | yes | +1.25 | +9.45 | +31.70 | +11.04 |
| rvol180__pdh090__or9__trail08 | yes | +8.37 | +7.19 | +30.59 | +8.44 |
| stability__rvol180__or9__trail08__flowhold8 | yes | +5.77 | +7.27 | +28.11 | +9.70 |
| stability__rvol1p8__or10__trail0p08 | yes | -0.37 | +7.78 | +33.37 | +9.97 |
| stability__rvol180__or9__trail08__pdh0p85 | yes | +14.68 | +7.02 | +34.18 | +8.89 |
| rvol180__or9__trail08 | yes | +5.54 | +7.11 | +37.81 | +8.71 |
| stability__rvol180__or9__trail08__pdh0p8 | yes | +6.59 | +7.11 | +36.09 | +8.71 |
| stability__rvol180__or9__trail08__entry130000 | yes | +19.00 | +10.37 | +30.22 | +12.48 |
| stability__rvol1p75__or8__trail0p06 | yes | +21.64 | +8.36 | +45.64 | +9.43 |
| stability__rvol180__or9__trail08__pdh0p95 | yes | +4.88 | +7.36 | +32.50 | +8.62 |
| stability__rvol180__or9__trail08__entry124500 | yes | +8.06 | +8.53 | +30.93 | +10.33 |
| stability__rvol1p8__or8__trail0p06 | yes | +13.87 | +7.19 | +29.04 | +8.08 |
| stability__rvol1p85__or9__trail0p06 | yes | +14.27 | +6.36 | +32.68 | +7.63 |
| stability__rvol1p85__or10__trail0p06 | yes | +10.47 | +6.94 | +33.39 | +8.17 |
| stability__rvol1p75__or9__trail0p1 | yes | +9.13 | +8.28 | +34.24 | +10.50 |
| stability__rvol1p75__or10__trail0p1 | yes | +3.73 | +9.37 | +32.58 | +11.94 |
| stability__rvol1p8__or10__trail0p1 | yes | +2.94 | +8.53 | +26.30 | +10.42 |
| stability__rvol1p75__or8__trail0p08 | yes | +15.70 | +8.03 | +32.96 | +9.52 |
| stability__rvol1p8__or9__trail0p1 | yes | +1.50 | +7.19 | +28.37 | +9.07 |
| stability__rvol1p8__or8__trail0p08 | yes | +14.16 | +6.94 | +20.08 | +8.08 |
| stability__rvol1p85__or9__trail0p08 | yes | +7.48 | +5.94 | +24.42 | +7.72 |
| stability__rvol1p85__or10__trail0p08 | yes | -0.43 | +6.86 | +25.22 | +8.80 |
| stability__rvol1p85__or8__trail0p06 | yes | +21.88 | +6.27 | +24.06 | +6.20 |
| stability__rvol1p85__or9__trail0p1 | yes | +13.84 | +6.02 | +16.73 | +7.72 |
| rec__trail_distance_0p06 | yes | +12.66 | +3.09 | +29.04 | +2.87 |
| stability__rvol1p75__or8__trail0p1 | yes | +8.13 | +8.70 | +25.57 | +9.16 |
| stability__rvol1p85__or10__trail0p1 | yes | +0.75 | +7.02 | +25.89 | +8.26 |
| stability__rvol1p85__or8__trail0p08 | yes | +27.04 | +5.69 | +12.20 | +6.64 |
| stability__rvol1p8__or8__trail0p1 | yes | +8.37 | +7.02 | +21.90 | +7.90 |
| rec__trail08__entry1300 | yes | +14.70 | +5.10 | +21.02 | +6.11 |
| rec__trail_distance_0p08 | yes | +8.86 | +2.84 | +20.15 | +3.50 |
| rec__trail_distance_0p1 | yes | +7.63 | +2.93 | +20.02 | +2.96 |
| rec__entry_124500 | yes | +1.26 | +3.51 | +4.07 | +4.13 |
| rec__flow_hold8__trail08 | yes | +14.48 | +2.76 | +14.53 | +3.14 |
| rec__or9__trail08__entry1300 | yes | +11.04 | +7.69 | +41.56 | +9.88 |
| score__rvol190__or9__trail08 | yes | +10.12 | +4.77 | +22.43 | +5.75 |
| stability__rvol1p85__or8__trail0p1 | yes | +13.41 | +5.94 | +5.57 | +6.55 |
| risk__or9__trail08__entry1300 | yes | +10.60 | +4.93 | +38.60 | +7.27 |

## Interpretation

- A strict dominator is stronger evidence than winning one arbitrary composite score.
- Smooth local RVOL/PDH neighborhoods are preferred to isolated optima.
- Segment failure vetoes a candidate even when aggregate IS and OOS pass.
- No result is eligible for production promotion until a fresh authoritative lockbox is available.
