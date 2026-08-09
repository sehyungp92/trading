# ALCB Round 2 drawdown diagnosis and final candidate review

## Status and scope

This is diagnostic-only research on the repaired legacy cache. The OOS window
2026-03-02 through 2026-05-01 has already been consumed repeatedly, and an
authoritative frozen direct-RTH data bundle is unavailable. No production or
live strategy configuration was modified.

All 118 atomic candidates and 29 combinations inherited the full accepted
Round-2 optimized configuration. The experiments therefore test cumulative
mutations from all prior rounds, not only the latest changes. Forty-one
finalists were selected for early IS, late IS, and July-September 2024 stress
validation.

This phase also sits on top of the preceding lineage audit, which covered all
35/35 cumulative accepted-mutation keys using explicit historical/neutral
controls where deleting a configuration literal would otherwise be a runtime
no-op. The results below therefore combine exhaustive accepted-lineage
ablation with the new drawdown-specific perturbation and mutation sweep.

Baseline for this phase:

- RVOL threshold 1.70
- opening range 9 bars
- adaptive late-trail distance 0.04R
- every other accepted setting from `optimized_config.json`

## Conclusion

The 11.94% IS maximum drawdown is not caused by a handful of catastrophic
losses. It is a systematic, regime-clustered false-breakout problem amplified
by correlated exposure and quality-insensitive sizing.

The best balanced drawdown-correcting research candidate is:

`combo_balanced__geom125__selection30__flow8`

Additional mutations over the balanced baseline:

```json
{
  "ablation.use_orb_entry_range_gate": true,
  "param_overrides.orb_entry_range_cap_r": 1.25,
  "param_overrides.selection_long_count": 30,
  "param_overrides.flow_reversal_min_hold_bars": 8
}
```

This is not a strict dominator. It is the best compromise when the objective
requires a material IS drawdown reduction while protecting IS dollar return
and frequency and significantly improving OOS performance.

## OOS discrepancy and data limitation

The available repaired-cache replay does not reproduce a uniform aggregate
OOS collapse. The balanced baseline records 50.52R, PF 3.35, 55.29
trades/month, and 1.97% DD in the two-month OOS window. In the earlier current
Round-2 control replay, March was -1.45R, April +32.60R, and 1 May -0.65R:
almost all aggregate profit came from one month. The meaningful discrepancy is
therefore temporal/regime concentration and a very short holdout, not a
demonstrated permanent loss of edge.

If the user's observed underperformance came from the unavailable frozen
direct-RTH bundle, it cannot be exactly reproduced from this repository. That
is why the findings are diagnostic and every proposed candidate remains
unpromoted.

## Why IS drawdown is high

### Not a tail-loss problem

- 575 losing trades generated $34,823 of gross losses.
- Worst 1 / 3 / 5 / 10 / 20 losses account for only
  1.0% / 2.6% / 4.1% / 7.7% / 13.2% of gross losses.
- Median loss is -1.04R; the minimum is -1.50R.
- Removing or capping a few edge cases cannot solve the drawdown.

### The main drawdown is a prolonged failure cluster

- Equity peak: 2024-07-18.
- Trough: 2024-09-30.
- Recovery: 2024-12-03.
- 74 calendar days to trough; 138 days to recovery.
- 96 trades closed during the descent; 59 were losses.
- Descent result: -17.46R and -$1,497.

The period is dominated by ordinary close-stop exits:

- 95 of 96 trades exited via `CLOSE_STOP`.
- OR breakouts contributed -7.92R.
- PDH breakouts contributed -6.29R.
- Combined breakouts contributed -3.25R.

This is a sequence of normal failed signals, not a fat-tail execution accident.

### The systematic failure signature

- Score 6 trades lost -17.14R in the descent; score 7 lost -4.12R.
  Higher score increased sizing during a bad regime, but score 6/7 trades are
  profitable over the complete IS period. Blocking them is an overfit response.
- RVOL 2.00-2.49 lost -14.20R and RVOL 1.80-1.99 lost -5.36R in the
  descent. Those cohorts are not uniformly negative outside the episode, so a
  narrow RVOL dead-band is not robust.
- Regime A contributed -15.74R versus -1.72R in regime B.
- September 2024 contributed -11.95R.
- Technology and Consumer Discretionary were the largest dollar contributors,
  but both sectors are strongly positive over full IS. Sector or symbol
  blacklists would be sample-specific.

### False breakouts fail to establish MFE

Across full IS:

- 272 trades with MFE below 0.10R lost -201.67R and about $15,251.
- 129 trades with MFE 0.10-0.19R lost -70.39R.
- 208 trades with MFE 0.20-0.39R lost -51.20R.
- 320 trades with MAE at or above 1.0R lost about -353.6R in total.

The main weakness is entry acceptance/geometry: too many breakouts never
establish favorable excursion. It is not principally a winner-exit problem.

## Signal extraction and alpha left on the table

### Existing filters mostly reject negative or neutral setups

The simplified rejected-signal shadow found:

- AVWAP, RVOL-max, conditional, OR-width, combined-quality, and long-only
  rejections were approximately neutral or negative.
- Buying-power rejections showed +39.2R in the simplified shadow.
- Sector-limit rejections showed +8.1R.

Portfolio replay overturned the optimistic capacity inference:

- Sector limit 4 improved IS but degraded OOS R, PF, and net.
- Extra position slots were neutral to negative.
- Leverage 2.5-3.0 raised IS dollars but materially worsened OOS PF and
  drawdown.

The shadow simulator does not model correlated exposure, capital sequencing, or
changed exits. Capacity relaxation is not robust recovered alpha.

### Scanner expansion is useful only with a discriminator

Increasing `selection_long_count` from 20 to 30 by itself is recent-regime
biased:

- Aggregate/OOS improve modestly.
- Early-IS R retention is only 84.3%.

Pairing the wider scanner with an entry-range geometry gate rescues early
stability. This is the useful signal-extraction result: broaden the opportunity
set, then discriminate with causal intraday geometry.

### Quality score is useful for sizing, not hard rejection

ORB quality is directionally monotonic at the top end, but low-score groups are
still positive in aggregate.

- Hard floors improve PF/DD but delete too much R and frequency.
- A continuous size floor of 0.70 raises normalized IS/OOS R and frequency and
  reduces drawdown.
- It lowers IS dollar PnL by 6.5%, so it is a risk-adjusted alternative rather
  than the primary return candidate.

## Entry, management, and exit findings

### Entry confirmation is too expensive

One- to three-bar delayed confirmation, price-progress checks, MFE checks,
RVOL persistence, AVWAP holds, and breakout holds were tested separately.

- Delay-only confirmation destroys IS alpha.
- The least harmful adverse-excursion cap improves PF/DD but loses too much OOS
  R.
- Modest size restoration after confirmation does not repair it.

The strategy's alpha depends on prompt next-bar execution. Delayed confirmation
is not the fix.

### Early tightening and quick exits close eventual winners

- Maturation stops at 2/4/6/8 bars materially reduce R and PF.
- More lenient -0.25R/-0.40R maturation stops remain below the return guardrail.
- Quick exits reduce DD but cost 10% or more IS R/net in their least harmful
  form.

Weak early MFE is diagnostic, but using it as a blunt close/tighten rule removes
too many eventual winners.

### Reclaim/retest entries contain alpha but are unstable

Reclaim entries can raise IS R/frequency substantially. The best OOS reclaim
structure-stop variants also raise OOS net/R. However:

- IS PF falls about 11-12%.
- IS drawdown rises or becomes path-sensitive.
- Combining reclaim with capacity can look excellent in IS and stress while
  degrading OOS PF/DD.

Reclaim is a promising future family, not part of the final candidate.

### Flow reversal timing has a small robust effect

- Disabling flow reversal is neutral OOS and slightly worse IS.
- Delaying it to 16/24 bars is neutral or worse.
- Reducing the grace period from 12 to 8 bars improves OOS and reduces stress
  drawdown while both IS halves retain at least 90% of baseline R.

The 8-bar value is locally supported by 6/8/10-bar perturbations, although the
effect is smaller than the geometry/selection changes.

### MFE-conviction exit is low-value, not the loss driver

- Disabling it alone adds about 1.17R IS and changes nothing OOS/stress.
- On the leading finalists, disabling it has identical OOS/DD and mixed,
  economically trivial IS changes.

It is weakly evidenced complexity, but removing it does not solve
underperformance. Keep the enabled version for the research candidate; revisit
simplification only on fresh data.

### Retracement and later adaptive trails are harmful

Retracement trails at 10/15/20 bars materially damage IS R/PF and can increase
drawdown. Moving adaptive tightening later hurts OOS. The accepted 0.04R late
trail should remain.

## Candidate comparison

| Candidate | IS R | IS net | IS freq/mo | IS PF | IS DD | OOS R | OOS net | OOS freq/mo | OOS PF | OOS DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 265.23 | $31,032 | 55.86 | 1.891 | 11.94% | 50.52 | $2,611 | 55.29 | 3.347 | 1.97% |
| **Primary: geom 1.25 + selection 30 + flow 8** | 250.31 | $31,481 | 56.29 | 1.881 | 10.77% | 55.23 | $2,662 | 54.28 | 3.518 | 1.89% |
| Return-forward: geom 1.25 + selection 30 | 254.84 | $31,714 | 56.03 | 1.891 | 11.38% | 56.82 | $2,687 | 54.79 | 3.594 | 1.98% |
| Stress-forward: geom 1.15 + selection 30 + flow 8 | 249.32 | $30,671 | 55.73 | 1.879 | 10.42% | 55.59 | $2,683 | 54.28 | 3.601 | 1.89% |

Primary candidate deltas:

- IS: R -5.6%, net +1.4%, frequency +0.8%, PF -0.5%, DD -9.8%.
- OOS: R +9.3%, net +2.0%, frequency -1.8%, PF +5.1%, DD -4.1%.
- Early IS R retention: 95.4%.
- Late IS R retention: 93.9%.
- Stress R: effectively unchanged (+0.05R).
- Stress DD: -8.7%.

The primary candidate reduces the monetary/equity impact of the systematic
cluster rather than pretending the cluster can be eliminated with a narrow
historical exclusion.

## Is the primary really the best candidate?

There is no strict dominator across normalized R, dollar PnL, frequency, PF,
and drawdown. The primary is selected using the user's joint objective, not the
automated validation score alone.

Two candidates can look superior under narrower objectives:

- `geometry 1.15 + quality sizing + selection 30` produces 253.08 IS R,
  57.10 OOS R, 10.43% IS DD, and 1.87% OOS DD. But quality resizing lowers IS
  net PnL to $28,817 (-7.1%) and OOS net to $2,529 (-3.1%), despite higher
  normalized R. It is a useful lower-capital-at-risk alternative, not the best
  expected-dollar-return candidate.
- `reclaim + leverage 3.0` produces $38,963 IS net and 62.09 trades/month.
  Its OOS PF falls to 2.94, OOS DD rises to 2.82% (+42.8%), and OOS R is
  essentially flat. Its apparent IS/capacity advantage does not transfer.

The return-forward 1.25R geometry candidate has the highest protected IS
dollar PnL of the stable geometry finalists and the 1.15R candidates have the
strongest stress reduction. Adding flow-hold 8 to the 1.25R candidate is the
best middle ground:

- it is the only listed finalist that improves IS net and frequency together
  while reducing IS DD by nearly 10%;
- it improves every reported OOS quality/economic measure except frequency,
  whose decline is only 1.8%;
- both IS halves retain more than 93% of baseline R;
- its stress improvement is achieved without the 7% dollar-PnL haircut of the
  quality-sizing candidate.

Thus "best" means the most defensible balanced research candidate. If the
mandate were exclusively maximum OOS R, choose the 1.15R geometry/selection
variant; if it were exclusively drawdown minimization, choose the quality-sized
or 1.15R-flow variant. Neither better satisfies the stated joint objective.

## Accepted-mutation overfit assessment

The cumulative lineage audit does not support a wholesale rollback:

- Adaptive trailing, fast-runner activation, partial takes, and the complete
  exit architecture suffer large cross-window damage when removed.
- Restoring the complete Round-1 delta also fails.
- The combined-quality and OR-width gates can look attractive when relaxed on
  OOS, but their aggressive removals breach historical PF/DD guardrails.
- Failure-stop removal is approximately OOS-neutral and costs IS return.

Several accepted micro-mutations are low-value complexity:

- `carry_min_cpr` is an exact no-op in the audited replay.
- `carry_min_r` is nearly flat.
- the flow-reversal CPR gate, combined-breakout score floor, and individual
  score/detail size-map members move full-history results only a few R;
- the MFE-conviction exit is also economically negligible in this phase.

These settings are candidates for code/config simplification, but they are not
the cause of the high drawdown or the OOS discrepancy. Removing MFE conviction
from each leading finalist leaves OOS and drawdown unchanged and produces
mixed, sub-1.1R IS effects. Consequently no low-value accepted mutation should
be credited with the performance uplift, and no removal is bundled into the
primary candidate merely to make it more different from the baseline.

## Why the apparent aggregate winner is rejected

`selection30 + flow8` looks stronger in aggregate and OOS:

- IS net/frequency/DD improve.
- OOS R/frequency/net/PF/DD all improve.

But early-IS R retention is only 86.2%, versus 98.4% in late IS. The improvement
is concentrated in the recent regime and is exactly the kind of low-dimensional
overfit that should not be accepted.

## Daily-stop parity

The backtest engine does not currently enforce the live `daily_stop_r` or
portfolio daily-stop settings.

An approximate realized-R lockout using original fills/exits found:

- 2.35R threshold: 40 trades skipped, +7.41R / +$916 versus baseline, and
  approximate DD 11.63% versus 11.94%.
- 2.00R threshold: DD falls to 10.86%, but R/net deteriorate.
- 3.50R threshold is nearly inert.

This indicates a real parity gap but not the primary cause of the 74-day
drawdown. A causal engine-level daily-stop replay is still required before any
promotion; the approximation cannot model freed capital, simultaneous
positions, or changed subsequent fills.

## Final recommendation

Use `combo_balanced__geom125__selection30__flow8` as the new research candidate,
not as a production promotion.

Retain the return-forward and stress-forward variants as a three-candidate
pre-registered comparison for the next fresh lockbox:

1. Primary balanced drawdown candidate: 1.25R range cap + selection 30 + flow 8.
2. Return-forward: 1.25R range cap + selection 30.
3. Stress-forward: 1.15R range cap + selection 30 + flow 8.

Do not add symbol/sector blacklists, a narrow RVOL dead-band, delayed
confirmation, quick exits, maturity stops, retracement trails, reclaim/capacity
combinations, or higher leverage.

## Required work before promotion

1. Rebuild an authoritative frozen direct-RTH dataset and reserve a truly fresh
   lockbox.
2. Implement a causal backtest parity flag for the live daily-stop logic and
   rerun the three pre-registered candidates.
3. Run bootstrap/day-block and symbol/sector-cluster resampling on the three
   finalists.
4. Confirm transaction-cost and slippage sensitivity around the 1.15-1.25R
   geometry neighborhood.
5. Promote only if the primary candidate retains its split stability and the
   geometry response remains smooth on fresh data.

## Pre-registered targeted experiments still worth running

These can plausibly change the ranking, but should not be selected using the
already-consumed March-May 2026 OOS window:

1. **Exact portfolio daily-stop replay:** test 2.35R and 3.0R causally, allowing
   capital and subsequent fills to change. The original-fill approximation is
   promising enough to justify implementation.
2. **Continuous entry-geometry sizing:** taper size smoothly for entries
   between roughly 1.0R and 1.35R from the reference range, compared with the
   pre-registered 1.15R and 1.25R hard caps. This tests whether the hard
   boundary is unnecessarily leaving alpha on the table.
3. **Causal failure-density throttle:** reduce new risk only after a rolling,
   prior-trade cluster of failed breakouts or elevated portfolio correlation.
   It must use information available at entry and be evaluated without
   September-2024/date labels. This directly targets the actual systematic
   drawdown mechanism.
4. **Quality-score reconstruction:** `bar_vol_surge` is nearly universal and
   `adx_trending` is weakly discriminatory. Re-estimate a continuous score
   using entry geometry, volume acceleration, AVWAP slope/location, and
   breakout close quality. Keep execution prompt; delayed confirmation has
   already failed.
5. **Unlevered reclaim isolation:** retest only the causal reclaim/structure
   entry on fresh folds, without simultaneous leverage or capacity expansion.
   Reclaim contains incremental signal alpha, but the current interaction is
   too path-dependent to accept.

Further symbol, sector, calendar-date, RVOL dead-band, or single-OOS-month
optimization should not be run. Those searches target historical labels rather
than a transferable failure mechanism.
