# ALCB Round 2 final recommendation memo

## Bottom line

The recommendation from the first robustness pass (`rvol_threshold=1.90`,
`pdh_size_mult=0.90`) is not the best tested candidate. It is strictly
dominated by a large, smooth family of alternatives after aggregate IS/OOS
guardrails and early/late IS validation. The improvement is not explained by
one catastrophic loss or one isolated parameter value.

The generated equal-window utility score selects:

- `rvol_threshold=1.65`
- `opening_range_bars=9`
- `adaptive_trail_late_distance_r=0.04`
- no PDH sizing override

However, its score is only 0.3840 versus 0.3800 for RVOL 1.70, and the score
uses an OOS window that has now been examined repeatedly. The more defensible
candidate to carry into a genuinely fresh confirmatory lockbox is therefore:

- `rvol_threshold=1.70`
- `opening_range_bars=9`
- `adaptive_trail_late_distance_r=0.04`
- no PDH sizing override

This is the **balanced research candidate**, not an authorization to promote
the configuration to production.

## Candidate comparison

| Candidate | Role | IS R | IS trades/mo | IS PF | IS net | IS DD | OOS R | OOS trades/mo | OOS PF | OOS net | OOS DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current Round 2 | control | 183.76 | 45.40 | 1.764 | $19,641 | 12.28% | 30.50 | 45.15 | 2.372 | $1,652 | 2.31% |
| Prior recommendation: RVOL 1.90 + PDH 0.90 | displaced recommendation | 195.18 | 47.86 | 1.742 | $20,323 | 13.27% | 35.99 | 47.69 | 2.662 | $2,099 | 2.04% |
| RVOL 1.65 / OR 9 / trail 0.04 | automated score winner | 253.73 | 57.94 | 1.883 | $31,864 | 12.63% | 51.98 | 54.28 | 3.386 | $2,636 | 1.92% |
| RVOL 1.70 / OR 9 / trail 0.04 | balanced research candidate | 265.23 | 55.86 | 1.891 | $31,032 | 11.94% | 50.52 | 55.29 | 3.347 | $2,611 | 1.97% |
| RVOL 1.75 / OR 9 / trail 0.04 | aggregate R/PF alternative | 266.55 | 54.69 | 1.924 | $30,965 | 12.17% | 51.00 | 53.27 | 3.380 | $2,587 | 1.97% |
| RVOL 1.65 / OR 10 / trail 0.04 | frequency/net frontier | 235.87 | 58.89 | 1.889 | $32,419 | 12.47% | 51.88 | 54.28 | 3.385 | $2,636 | 1.92% |

Relative to the prior recommendation, the balanced RVOL 1.70 candidate raises
IS R by 35.9%, IS frequency by 16.7%, and IS net by 52.7%, while lowering IS
drawdown by 1.33 percentage points. In the reused OOS window it raises R by
40.4%, frequency by 16.0%, and net by 24.4%, while lowering drawdown by 0.07
percentage points.

RVOL 1.70 is preferred over the automated RVOL 1.65 winner because it produces
more R in both IS halves, a lower full-IS drawdown, higher OOS frequency, and
nearly identical OOS economics. Its early/late IS R values are 66.14/198.63,
versus 54.34/192.31 at RVOL 1.65. RVOL 1.65 remains a valid higher-activity,
higher-OOS-score alternative rather than a uniquely best point.

## What caused the discrepancy

The repaired legacy cache does not reproduce aggregate OOS underperformance
for the current Round-2 control: the window has 89 trades, 65.17% wins,
30.50R, PF 2.37, and 2.31% drawdown. It does reveal strong temporal
concentration: March was -1.45R, April +32.60R, and 1 May -0.65R. Therefore the
main weakness is regime/month concentration and selection fragility, not a
uniform collapse in edge.

OOS losses are not dominated by a few catastrophes. There are 31 losing
trades; the worst one, three, and five account for about 8.0%, 20.5%, and
31.0% of gross loss. The repeated weakness is instead the short-hold cohort:
0-24 bar holds contributed -23.72R, while 25+ bar holds contributed +54.22R.

The prior recommendation was too narrow and gave too much status to a small
PDH sizing effect. PDH 0.80-0.95 forms a reasonably smooth secondary sizing
surface, but it is not required for the main uplift. The dominant mechanism is:

1. Lower RVOL admits more opportunities and raises frequency.
2. OR 9-10 and a tighter late trail recover quality and contain drawdown.
3. The 0.04-0.08 trail neighborhood is broadly positive, with 0.04 strongest
   in the tested range.

Direct failure-stop mutations generally damaged return or quality. Extending
entry to 13:00 raised frequency but reduced OOS R/PF. Flow-hold 8 is a useful
conservative control, improving OOS and drawdown modestly, but it gives up
full-IS R. These should not be added to the balanced candidate without fresh
evidence.

## Breadth of the review

- The first audit covered all 35/35 cumulative accepted-mutation lineages, not
  only the most recent round.
- The recommendation review evaluated 102 unique aggregate candidates.
- Sixty-three names, including controls, received early/late IS validation;
  every strict dominator was admitted rather than pruned by a shortlist.
- The final response surface is smooth across RVOL 1.65-1.80, OR 9-10, and
  trail distances 0.04-0.08. This is materially stronger evidence than an
  isolated optimum.

## Experiments that should still be run

Do not choose another mutation by reusing the March-May 2026 OOS window. It is
now a development set and further point optimization would increase selection
bias. The next phase should be preregistered and evaluated on authoritative,
frozen direct-RTH data:

1. Carry the four frontier settings in the table into an untouched lockbox
   after 1 May 2026, with RVOL 1.70 / OR 9 / trail 0.04 named in advance as the
   balanced primary candidate.
2. Run a trail confirmation grid of 0.03/0.04/0.05 without using the consumed
   OOS window for selection. The current 0.04 optimum is at the tested trail
   boundary even though it is stable across RVOL and OR values.
3. Run rolling and anchored walk-forward folds, leave-one-month-out, and
   leave-one-symbol-out tests. Report the distribution of fold uplift, not
   only pooled totals.
4. Perturb fills, slippage, spread, and entry/exit timing; bootstrap trades and
   months; verify that the advantage survives realistic cost and ordering
   uncertainty.
5. Recheck the short-hold loss cohort by signal type and symbol on fresh data.
   Only consider flow-hold 8 if its benefit repeats without lowering full-fold
   return materially.

No live strategy configuration was modified by this analysis.
