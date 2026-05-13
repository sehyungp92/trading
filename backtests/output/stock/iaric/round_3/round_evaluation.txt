# IARIC OOS Discrepancy Goal Evaluation

Generated from the focused 98-name run in `backtests/output/iaric_oos_ablation_focused_mw2`.

## Verdict

The goal was substantially carried out, but the result should be treated as a strong diagnostic challenger selection rather than a final production promotion.

The two required phases were both completed:

- Phase 1 evaluated accepted prefixes, accepted rollbacks/removals, single perturbations, and targeted combinations on the focused universe.
- Phase 2 used the OOS weakness shape to test additional targeted combinations, especially the interaction between lower entry score floor and earlier partial-profit harvesting.

The most important finding is that the discrepancy was not fixed by removing a single bad accepted mutation. Accepted rollbacks and prefixes mostly remained OOS-negative. The uplift came from changing the calibration of two live levers that interact:

- `param_overrides.pb_v2_signal_floor`
- `param_overrides.pb_v2_partial_profit_trigger_r`

## Coverage Check

Phase 1 evaluated 117 candidates:

- 84 single perturbations
- 12 accepted rollbacks/removals
- 11 targeted combinations
- 8 accepted prefixes
- 2 baselines

Current focused champion:

- OOS: 12 trades, 50.0% win rate, 0.449 PF, -1.725 netR, -0.144 avgR
- IS: 955 trades, 71.6% win rate, 1.706 PF, +102.933 netR, +0.108 avgR

Best accepted rollback/prefix did not solve the problem. The best rollback was still OOS-negative:

- `rollback_signal_floor_75`: 7 OOS trades, -1.531 netR, -0.219 avgR

That argues against the issue being one isolated accepted mutation that should simply be removed.

## Weakness Shape

The OOS loss shape was not a few catastrophic edge cases. It was a small sample of normal stop/gap failures:

- 12 total OOS trades
- 5 `STOP_HIT` trades, -1.475 netR
- 2 `GAP_STOP` trades, -0.507 netR
- Healthcare: 2 trades, -1.745 netR
- Communication Services: 3 trades, -0.948 netR
- Technology: 7 trades, +1.087 netR

The funnel also showed low effective frequency:

- `triggered`: 191
- `daily_signal_floor_reject`: 179
- `entered`: 12
- `partial`: 2

This supported testing lower signal floors and earlier partial-profit capture rather than only sector quarantines or tail-loss filters.

## Phase 1 Result

The first phase found that lowering the signal floor from 72 to 70 was the best simple frequency uplift:

- `sweep_signal_floor_70p0`
- Patch: `{"param_overrides.pb_v2_signal_floor": 70.0}`
- OOS: 19 trades, +1.322 netR, +0.070 avgR, 1.422 PF
- IS: 1103 trades, +106.608 netR, +0.097 avgR, 1.600 PF

It also found that earlier partial profit alone was the best simple expectancy uplift:

- `sweep_partial_trigger_0p1`
- Patch: `{"param_overrides.pb_v2_partial_profit_trigger_r": 0.1}`
- OOS: 12 trades, +0.666 netR, +0.055 avgR, 1.342 PF
- IS: 961 trades, +167.885 netR, +0.175 avgR, 2.635 PF

The first-phase generated best eligible config was therefore directionally right, but incomplete: it did not combine the two most explanatory levers.

## Phase 2 Result

Phase 2 tested the missing interaction. The combination was strongly positive.

Conservative challenger:

- `floor70_partial01`
- Patch: `{"param_overrides.pb_v2_signal_floor": 70.0, "param_overrides.pb_v2_partial_profit_trigger_r": 0.1}`
- OOS: 19 trades, 78.9% win rate, 2.727 PF, +3.361 netR, +0.177 avgR
- IS: 1114 trades, 80.0% win rate, 2.348 PF, +177.308 netR, +0.159 avgR

Max-return/frequency challenger among tested candidates:

- `floor66_partial01`
- Patch: `{"param_overrides.pb_v2_signal_floor": 66.0, "param_overrides.pb_v2_partial_profit_trigger_r": 0.1}`
- OOS: 34 trades, 76.5% win rate, 2.583 PF, +5.876 netR, +0.173 avgR
- IS: 1470 trades, 79.7% win rate, 2.290 PF, +225.013 netR, +0.153 avgR

Best OOS avgR among the tested second-stage candidates:

- `floor68_partial01`
- Patch: `{"param_overrides.pb_v2_signal_floor": 68.0, "param_overrides.pb_v2_partial_profit_trigger_r": 0.1}`
- OOS: 25 trades, 84.0% win rate, 3.764 PF, +5.379 netR, +0.215 avgR
- IS: 1303 trades, 79.3% win rate, 2.246 PF, +196.956 netR, +0.151 avgR

## Was The Goal Optimally Carried Out?

Within the explored search space, yes for diagnosis and challenger discovery:

- The missing-data warning was removed by focusing the replay on the same 98-name intraday universe.
- Accepted mutation removal was directly tested and did not explain the discrepancy.
- The OOS failure mode was examined before targeted changes were selected.
- The targeted second phase found candidates that improved both IS and OOS, including return, win rate, profit factor, and frequency.

Not fully final in a statistical sense:

- The final challenger selection used the same short OOS window that motivated the search.
- The OOS window contains only 12 baseline trades, so parameter rankings are sensitive to a few added or removed trades.
- The first-phase eligibility rules would have discarded `signal_floor` 66/68 as standalone changes, even though they became strong when combined with `partial_profit_trigger_r=0.1`. That means the search process should explicitly model interactions, not only single mutations plus a small hand-built combo set.
- The persisted `best_eligible_challenger_config.json` reflects the phase-1 best eligible candidate, not the stronger phase-2 candidates.

## Recommendation

Treat these as three distinct candidates:

- Conservative practical challenger: `floor70_partial01`
- Balanced higher-return challenger: `floor68_partial01`
- Max-return/frequency challenger: `floor66_partial01`

Before production promotion, run a third validation step:

- pre-register those three candidates plus current;
- compare them across rolling pseudo-OOS windows inside the IS history;
- stress by sector, exit reason, month, and trade-count bootstrap;
- only then promote the simplest candidate that remains on the Pareto frontier.

