# Family causal replay R1B NQDTC checkpoint

Status: the accepted R0/R1A checkpoint is frozen, and only the bounded NQDTC R1B subincrement is complete for review on 2026-07-16. This does not claim full R1B, Momentum family parity, optimizer cutover, Vdub, Downturn, R2, broker calibration, Stock, Swing, or operational lifecycle expansion.

## Scope and selective salvage

The implementation inspected dangling G4M-v2 commit `b163011` and selectively retained only the strategy-owned NQDTC entry-decision extraction idea. No G4M generic runtime, market-state package, identity framework, evidence infrastructure, recovery machinery, or branch was merged or cherry-picked.

The real live and source-backtest wrappers now call `strategies/momentum/nqdtc/core/entry_decision.py` for entry A, B, C, fallback selection, sizing, stops, order type, and time-in-force. Wrapper policy fields preserve the existing live/backtest differences explicitly. The old live placement helpers and source-backtest A/B/C/fallback decision blocks were removed; replay does not contain a third copy of those decisions.

NQDTC core state now owns proposal, pending authorization, acknowledgement, fill, and protective-stop feedback. The live wrapper submits the proposal through the existing OMS, then returns its approval, resized quantity, denial reason, acknowledgement, and fill to the same core. The replay wrapper calls the same entry and lifecycle functions. The existing OMS state machine, fill processor, fake IBKR path, in-memory repositories, coordinator overrides, and family coordinator remain authoritative.

The existing live portfolio checker now builds an explicit directional-risk snapshot and delegates its unchanged cap calculation to `evaluate_directional_cap`. The bounded replay builds the same snapshot synchronously in memory. No replacement portfolio engine or top-level runtime package was added.

## Migration boundary and bounded oracle

Compatibility extraction completed before causal authorization was enabled. The unchanged NQDTC unit baseline passed 20 tests before extraction and the same 20 tests passed after both wrappers adopted the shared entry core. Only then was pending authorization and feedback added; the expanded core suite now passes 23 tests, including exact compatibility-versus-causal proposals and approved degenerate authorization.

Compatibility and causal `on_bar` calls produce the exact same proposal and decision event. Their first internal state divergence is the causal pending-authorization reservation; with full approval, identical quantity, and immediate full fill, their submitted product is exact. The approved bounded fixture also produces exact normalized live/replay orders, terminal fills, ledger, and final state. The two feedback-dependent scenarios have the expected first observable divergence:

- disabled NQDTC first diverges at `ENTRY_DENIED`; neither NQDTC proposal is submitted or filled, NQDTC remains flat, and the later NQ_REGIME child is approved and filled;
- directional contention first diverges at NQ_REGIME authorization after NQDTC's two working entries reserve $500 of long risk; NQ_REGIME is denied by the shared 1.2R/$600 cap and remains flat.

The approved scenario fills both children. NQDTC fill feedback opens real NQDTC core position state and produces the protective stop before final comparison. The shared raw timeline is ordered by timestamp and configured priority, with NQDTC priority 0 before NQ_REGIME priority 1 at the common timestamp. Its input is three raw NQDTC OHLCV bars plus the existing raw NQ_REGIME bar, seeded state, materialized family configuration, and the existing scripted fake-broker events. It consumes no completed trade, known PnL, or fixed state-dependent intent.

Live and replay receive the same fixture object. The existing runtime source fingerprint covers normalized market input, materialized configuration and priority, initial state, execution/cost inputs, and broker script; both producers report the same fingerprint in every scenario. No fixture JSON, verifier script, archived trace, or evidence tree was added.

## Product and test growth

Relative to the frozen R1A production baseline, eight production files changed: 1,178 inserted lines and 962 deleted lines, net +216. One strategy-owned module was added, no production file was deleted or moved, and no generic package was added. The gross wrapper deletion is 850 lines across the live and source-backtest NQDTC engines. The shared module is used by live, source backtest, and the bounded causal replay.

Production changes are limited to:

- the NQDTC shared entry and lifecycle core, serializers, and state;
- the existing live and source-backtest NQDTC wrappers;
- the existing portfolio-rule module's provider-neutral directional-cap calculation.

Existing tests and parity helpers were extended in place: NQDTC core tests, the Layer-2 live-shadow test, source-input normalization, live/replay Layer-2 drivers, replay runner/candidate plumbing, live state compaction, and the existing parity harness. No new test framework or standalone fixture file was created.

## Performance

Measurements ran on Windows with Python 3.12.6, an Intel i7-9750H (6 cores/12 logical processors), and 16,498,004 KiB visible physical memory.

The measured candidate loop is the synchronous, in-memory NQDTC + NQ_REGIME decision/authorization/lifecycle path only. The fixture is materialized once; every timed candidate receives new strategy, portfolio, order, and replay state. The loop performs no database or JSON writes, logging, sleeps, process startup, live coordinator startup, or external I/O.

Imports took 5.684375 seconds and one-time fixture preparation took 0.006263 seconds. After 25 warm-ups, five 500-candidate runs took 11.704926, 9.527342, 11.347295, 9.468565, and 8.016196 seconds. Median processing was 0.019054685 seconds/candidate, 52.48 candidates/second, or 787.21 deterministic domain transitions/second using the path's 15 decision, portfolio, authorization, acknowledgement, and fill transitions per candidate.

Prepared RSS was 70.852 MiB. Peak and final RSS were 74.539 MiB, a 3.688 MiB hot-loop increase. A 28-candidate bounded cohort therefore projects to about 0.53 seconds of transition time and remains structurally below the frozen 120-second/288-MiB R2 budget. This is not a full-window or optimizer-cutover measurement.

## Verification

| Command | Result |
|---|---|
| `python -m pytest tests/unit/test_nqdtc_core.py -q` before extraction | 20 passed in 5.81s |
| `python -m pytest tests/unit/test_nqdtc_core.py -q` after compatibility extraction, before causal correction | 20 passed |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py::test_momentum_r1b_nqdtc_shared_raw_timeline -q -x` | 3 passed in 31.28s |
| `python -m pytest tests/unit/test_nqdtc_core.py tests/unit/test_nq_regime_core.py tests/unit/test_nq_regime_live_engine.py tests/unit/test_momentum_family_portfolio_replay.py tests/unit/test_momentum_portfolio_synergy_live_parity.py tests/unit/test_in_memory_repo_family_scoped_methods.py tests/integration/parity/test_live_shadow_layer2.py tests/integration/parity/test_live_shadow_families.py -q` | 86 passed, 1 skipped in 76.38s |
| `python -m pytest tests/unit/test_nqdtc_core.py tests/integration/parity/test_live_shadow_layer2.py::test_momentum_r1b_nqdtc_shared_raw_timeline -q` after final shared-timeline cleanup and degenerate-boundary test | 26 passed in 15.06s |
| `python -m compileall -q strategies/momentum/nqdtc/core strategies/momentum/nqdtc/engine.py backtests/momentum/engine/nqdtc_engine.py tests/integration/parity` | passed |
| `python -m ruff check strategies/momentum/nqdtc/core/entry_decision.py strategies/momentum/nqdtc/core/logic.py strategies/momentum/nqdtc/core/serializers.py strategies/momentum/nqdtc/core/state.py libs/oms/risk/portfolio_rules.py tests/integration/parity/source_inputs.py tests/integration/parity/replay_layer2.py tests/integration/parity/replay_runners.py tests/integration/parity/test_live_shadow_layer2.py tests/unit/test_nqdtc_core.py` | passed |
| `git diff --check -- backtests/momentum/engine/nqdtc_engine.py libs/oms/risk/portfolio_rules.py strategies/momentum/nqdtc tests/integration/parity tests/unit/test_nqdtc_core.py` | passed; line-ending conversion warnings only |

The one skip is the existing optional PostgreSQL repository comparison because `PARITY_POSTGRES_DSN` was not set; the in-memory repository tests ran.

## Remaining R1B work

Vdub, Downturn, the frozen overlapping historical slice, remaining family feedback scenarios, and full simultaneous Momentum completion remain unstarted in this increment. Resize, partial-fill effects on a later child, cancellation/rejection effects on later eligibility, realized-loss/cooldown propagation, and full materialized-configuration replay remain for their reviewed increments. R2 optimizer integration, official cutover, worker/cache determinism, broker calibration, and production-confidence claims remain explicitly out of scope.
