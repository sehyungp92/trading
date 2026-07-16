# Family causal replay R1B Downturn entry checkpoint

Status: the accepted Vdub R1B checkpoint is frozen in commit `c202a2e` (`feat(momentum): checkpoint Vdub causal replay`). That commit was created from the following literal 14-file allowlist; its cached diff contained 14 files, no blocked path, and no unexpected path. Unrelated raw data, manifests, diagnostics, generated outputs, and state files were excluded and remain unstaged.

```text
backtests/momentum/engine/vdubus_engine.py
strategies/momentum/vdub/core/__init__.py
strategies/momentum/vdub/core/entry_decision.py
strategies/momentum/vdub/core/logic.py
strategies/momentum/vdub/core/state.py
strategies/momentum/vdub/engine.py
tests/integration/parity/harness.py
tests/integration/parity/live_layer2.py
tests/integration/parity/replay_layer2.py
tests/integration/parity/replay_runners.py
tests/integration/parity/source_inputs.py
tests/integration/parity/test_live_shadow_layer2.py
tests/unit/test_remaining_strategy_core_contracts.py
family-causal-replay-r1b-vdub-checkpoint.md
```

Only the bounded Downturn entry/proposal subincrement is complete for review on 2026-07-16. The Downturn changes described here are deliberately unstaged and uncommitted. This does not claim the remaining feedback matrix, the historical slice, full Momentum family parity, R2, optimizer cutover, broker calibration, or production-confidence parity.

## Compatibility characterization and extraction

Before editing, the existing Downturn core suite passed 7 tests and the existing live order-correlation contract passed its Downturn case. The live wrapper selected reversal, Fade, then momentum. The source wrapper selected breakdown, reversal, then Fade/momentum and retained its source-only progressive-SMA, chop, volatility-percentile, and conviction policies. Both wrappers already shared the strategy's signal primitives, regime sizing, stop computation, and TP schedule but duplicated the pre-submit proposal assembly.

The minimum common gate, Fade/momentum selection, signal classification, sizing, stop, order-type, TTL, and immutable proposal now live in `strategies/momentum/downturn/core/entry_decision.py`. The real live engine and the existing source backtest both call that module. Exact compatibility products are locked for the two materialized wrapper policies before causal authorization is considered.

The wrapper differences remain explicit inputs:

- live keeps the established 2-trigger-tick, 4-limit-tick, 72-bar TTL, no fixed contract cap, `MAX_LEVERAGE_MULT`, and no source non-correction penalty policy;
- source keeps configurable trigger and limit offsets, TTL, maximum contracts, maximum notional leverage, and the source non-correction penalty;
- both preserve the established Downturn `0.50` MNQ proposal tick/execution assumption even though the shared parity instrument registry uses the broker-facing `0.25` tick elsewhere;
- source retains breakdown/reversal precedence and its source-owned extra gates; live retains its existing reversal-first policy.

For the compatibility fixture, the shared function exactly returns live `(entry=19996.0, stop=20010.0, limit=19994.0, qty=26, TTL=72)` and source `(entry=19995.5, stop=20010.0, limit=19993.0, qty=10, TTL=24)` products.

The intentional first causal divergence is authorization after `ENTRY_PROPOSED`. A proposal is pending and is not registered as working risk until the existing OMS/family portfolio path accepts it. Disabled-strategy denial produces `ENTRY_DENIED`, clears pending state, and submits nothing. Approval returns the approved quantity to the same core; the established order acknowledgement remains the authority that registers a working entry. Fill, partial-fill, cancel, reject, replace, and restart expansion were not added in this subincrement.

The existing portfolio rules, OMS state machine, fill processor, fake IBKR adapter, in-memory repository, serializer, parity normalizers, coordinator overrides, and Momentum family coordinator remain authoritative. No generic package, replacement runtime, fixture framework, database path, worker, or evidence infrastructure was added.

## Bounded four-child oracle

The existing shared raw-event timeline now orders NQDTC at priority 0, Vdub at priority 1, Downturn at priority 2, and NQ_REGIME at priority 3. Downturn consumes its raw 5-minute and 15-minute OHLCV plus materialized decision state. No completed trade, known PnL, cached intent, or fixed state-dependent order is an input.

The runtime source fingerprint covers raw input, materialized family configuration, initial state, ordering, execution/cost inputs, and broker script. Each live/replay pair matches, and the three scenarios are distinct:

- approved: `a4db9dd42a318b6b4561190de73873da0b1a807980133dc32b49038dc8897c1b`;
- disabled denial: `ee4aa447cbd4b89a28b743f199e2680ec4196f0e00a428ea7152ddeef668a326`;
- contention: `86eee08432e32581a9282f4262b9b825a44a4a4d7fc71d9c6aa33446fa757568`.

The approved scenario authorizes all four children and leaves the bounded Downturn entry working. The disabled scenario denies Downturn through the materialized portfolio rule, returns denial to real core state, and emits no Downturn order. In the Downturn-driven contention scenario, Downturn first reserves `$1,134.00` of short working risk. The later raw NQ_REGIME short proposal adds `$122.50`; `$1,256.50` exceeds the configured 2.5R/`$1,250.00` short cap, so NQ_REGIME is denied. Without Downturn, the `$122.50` NQ_REGIME proposal fits. Live and replay normalized orders, terminal events, ledger, and state snapshots are exact in all three scenarios.

## Product and test growth

Relative to frozen commit `c202a2e`, six production files changed: 824 inserted lines and 342 deleted lines, net +482. One strategy-owned module was added; no production file was deleted or moved. The live and source wrappers contain all 342 gross production deletions and are net -66 lines together; those deletions remove their duplicated common gate, Fade/momentum selection, sizing, stop, order-type, TTL, and proposal construction blocks.

Eight existing test/parity files changed: 675 inserted lines and 39 deleted lines, net +636. No fixture JSON, new fixture framework, verifier, or archived evidence output was added. The existing Downturn core, serializer, OMS/order-correlation, completed-bar, family portfolio, and Layer-2 parity surfaces were extended in place.

## Performance

The measured candidate path is synchronous virtual time and in memory. The approved fixture, source fingerprint, and immutable lineage are prepared once. Lifecycle instrumentation callbacks are disabled only for the timed candidate loop; the authoritative parity tests below run the unchanged OMS instrumentation path. The timed loop performs no per-candidate JSON or database writes, logging, sleeps, subprocesses, live coordinator startup, or external I/O.

Cold imports took 5.571822 seconds and one-time fixture/identity preparation took 0.020500 seconds. After 10 warm-ups, five 50-candidate four-child runs took 4.846285, 4.653043, 4.262187, 4.144233, and 3.876330 seconds. Median processing was 0.085243730 seconds/candidate, or 11.73 candidates/second. Counting the same explicit core/portfolio transition boundary used by the preceding checkpoint gives 269.81 transitions/second; the measurement is also 46.92 raw events/second and 58.66 portfolio authorizations/second.

Prepared RSS was 70.523 MiB. Peak RSS was 78.332 MiB, a 7.809 MiB hot-loop increase. A same-method Vdub three-child control on the same host measured 11.22 candidates/second and 70.805 to 77.625 MiB RSS, so the added Downturn child caused no measured throughput regression in this run. A 28-candidate cohort projects to about 2.39 seconds of transition time. This is a bounded structural measurement, not the deferred historical-slice or optimizer-cutover benchmark.

The isolated Downturn decision/proposal/authorization path measured 1,382.89 candidates/second and 11,063.15 explicit transitions/second after 250 warm-ups. Its prepared RSS was 40.172 MiB and peak RSS was 40.562 MiB, a 0.391 MiB increase.

## Verification

| Command | Result |
|---|---|
| `python -m pytest tests/unit/test_downturn_core.py -q` before extraction | 7 passed in 2.80s |
| `python -m pytest tests/unit/test_real_order_correlation_contract.py -q -k downturn` before extraction | 1 passed, 8 deselected in 4.63s |
| `python -m pytest tests/unit/test_downturn_core.py tests/unit/test_real_order_correlation_contract.py tests/unit/test_live_completed_bar_wiring.py -q -k downturn` | 11 passed, 12 deselected in 13.45s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py -q -k momentum_r1b_downturn` | 4 passed, 13 deselected in 39.05s |
| `python -m pytest tests/unit/test_nqdtc_core.py tests/unit/test_nq_regime_core.py tests/unit/test_nq_regime_live_engine.py tests/unit/test_remaining_strategy_core_contracts.py tests/unit/test_remaining_strategy_core_snapshots.py tests/unit/test_downturn_core.py tests/unit/test_downturn_instrumentation.py tests/unit/test_real_order_correlation_contract.py tests/unit/test_live_completed_bar_wiring.py tests/unit/test_momentum_family_portfolio_replay.py tests/unit/test_momentum_portfolio_synergy_live_parity.py tests/unit/test_in_memory_repo_family_scoped_methods.py tests/integration/parity/test_live_shadow_layer2.py tests/integration/parity/test_live_shadow_families.py -q` | 145 passed, 1 skipped in 129.05s |
| `python -m compileall -q strategies/momentum/downturn backtests/momentum/engine/downturn_engine.py tests/integration/parity` | passed |
| `python -m ruff check strategies/momentum/downturn/core/entry_decision.py strategies/momentum/downturn/core/logic.py strategies/momentum/downturn/core/serializers.py strategies/momentum/downturn/core/state.py tests/unit/test_downturn_core.py` | passed |
| `git diff --check --` with the 14-file Downturn implementation/test allowlist | passed; line-ending conversion warnings only |

The one skip is the existing optional PostgreSQL repository comparison because `PARITY_POSTGRES_DSN` was not set; the in-memory repository tests ran.

## Deferred boundary

The remaining Downturn feedback matrix, partial-fill effects on later children, cancel/reject risk release, replacement and restart behavior, frozen historical slice, and full simultaneous Momentum completion remain for later reviewed R1B work. R2, optimizer integration/cutover, cache or worker changes, broker calibration, and production-confidence claims remain explicitly out of scope.
