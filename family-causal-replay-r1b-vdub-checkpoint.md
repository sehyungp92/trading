# Family causal replay R1B Vdub checkpoint

Status: the accepted R0/R1A and NQDTC checkpoints are frozen. Commit `2dbad41` durably contains the controlling plan, R1A/NQDTC production code, tests, and both preceding checkpoint reports. The commit was built from an explicit allowlist; unrelated raw data, manifests, diagnostics, and state files were excluded and remain unstaged.

Only the bounded Vdub R1B subincrement is complete for review on 2026-07-16. This does not claim Downturn, the full historical slice, full Momentum family parity, R2, optimizer cutover, broker calibration, Stock, Swing, or production-confidence parity.

## Shared decision and migration boundary

The existing Vdub compatibility baseline was captured before edits: 20 tests passed. The minimum common Type A/B/C selection, position/opposite-direction and pyramid gates, entry/stop construction, sizing, viability, heat checks, and immutable order proposal were then extracted to `strategies/momentum/vdub/core/entry_decision.py`. Both the real live engine and the existing source backtest call this strategy-owned module. The same 20-test compatibility baseline passed unchanged after extraction and before causal authorization was enabled.

The Vdub core now owns pending proposal, portfolio authorization feedback, approved quantity, working-entry registration, entry fill, and protective-stop proposal state. The live wrapper obtains the proposal from the core, submits it through the existing OMS and family portfolio checker, and returns approval or denial to the same core. The source backtest executes the same proposal transition before its existing SimBroker submission. The bounded replay calls the same signal, proposal, gate, authorization, and fill functions. No live decision block was copied into replay.

The intentional first causal divergence is portfolio authorization. Before feedback, causal state contains a pending proposal and no working entry. A denial produces `ENTRY_DENIED`, removes the pending proposal, submits nothing, and reserves no flip or add-on state. An approval commits only the approved quantity to working state. Under the approved, immediate-full-fill degenerate case, live and replay normalized orders, fills, portfolio state, family state, and strategy state are exact.

The existing OMS state machine, family portfolio rules, fill processor, fake IBKR adapter, in-memory repository, snapshot serializer, parity normalizers, coordinator overrides, and Momentum family coordinator remain authoritative. Protective stops use the existing OMS stop lifecycle. No generic runtime, top-level package, fixture framework, database path, process worker, or evidence infrastructure was added.

## Bounded three-child oracle

The existing in-memory NQDTC fixture now materializes one Vdub raw event on the same timestamp/priority timeline:

- NQDTC priority 0 consumes its three raw 5-minute OHLCV bars and produces two proposals;
- Vdub priority 1 consumes four raw 15-minute OHLCV bars, two raw hourly bars, materialized completed-bar indicator arrays, regime state, and MNQ point value;
- NQ_REGIME priority 2 consumes the existing raw 5-minute bar.

The runtime source fingerprint covers the raw inputs, materialized family configuration and priorities, initial state, execution/cost inputs, and broker script. Approved, denied, and contention fixtures have distinct fingerprints, and each live/replay pair reports the same fingerprint.

The scenarios prove:

- approved: NQDTC, Vdub, and NQ_REGIME are authorized and filled; Vdub fill feedback creates real Vdub position state and its protective stop;
- denied: the materialized disabled-strategy rule denies Vdub, Vdub remains flat with no working entry, and the later NQ_REGIME child is still approved;
- Vdub-driven contention: NQDTC reserves $500 of long working risk, Vdub is then approved for $448, and the later $280 NQ_REGIME proposal is denied because $1,228 would exceed the materialized 2.0R/$1,000 directional cap. Without Vdub's working risk, the NQDTC plus NQ_REGIME total is $780 and fits, so the interaction is specifically Vdub-driven.

No completed trade, known PnL, cached intent, or fixed state-dependent order is an input.

## Product and test growth

Relative to frozen commit `2dbad41`, six production files changed: 867 inserted lines and 300 deleted lines, net +567. One strategy-owned module was added; no production file was deleted or moved. The live and source Vdub wrappers contain 288 gross deleted lines, including their duplicated signal-selection and pre-submit construction blocks. No generic package was added.

Existing tests and parity helpers were extended in place. The core contract test is parameterized for approved and denied authorization. The existing Layer-2 fixture builder is parameterized for approved, denied, and contention scenarios; no fixture JSON or dedicated verifier was added.

## Performance

The measured candidate path is synchronous virtual time and in memory. The approved fixture and sorted raw timeline are prepared once. Each candidate receives new Vdub, NQDTC, NQ_REGIME, family-portfolio, action-timeline, and fill state. The timed loop performs no database or JSON writes, logging, sleeps, process startup, live coordinator startup, or external I/O.

Imports took 4.325897 seconds and one-time fixture preparation took 0.003817 seconds. After 25 warm-ups, five 300-candidate transition runs took 5.151655, 8.381678, 6.675995, 5.340276, and 4.297190 seconds. Median processing was 0.017800921 seconds/candidate, or 56.18 candidates/second. Counting the 18 explicit core and portfolio transitions in the bounded path gives 1,011.18 transitions/second; the same measurement is 168.53 raw events/second and 224.71 portfolio authorizations/second.

Prepared RSS was 73.840 MiB. Peak and final RSS were 74.781 MiB, a 0.941 MiB hot-loop increase. A 28-candidate cohort therefore projects to about 0.50 seconds of transition time. This is a bounded structural measurement, not the deferred historical slice or optimizer-cutover benchmark.

## Verification

| Command | Result |
|---|---|
| `python -m pytest tests/unit/test_remaining_strategy_core_contracts.py tests/unit/test_remaining_strategy_core_snapshots.py tests/unit/test_real_order_correlation_contract.py::test_vdub_runtime_submit_and_fallback_carry_source_signal_and_bar_context tests/unit/test_live_completed_bar_wiring.py::test_vdub_helper_fetches_completed_only_bars -q` before extraction | 20 passed in 13.20s |
| Same command after live and source compatibility extraction, before causal correction | 20 passed in 18.62s |
| Same core/snapshot/correlation/completed-bar command after causal correction | 22 passed in 7.77s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py -k "momentum_r1b_vdub_shared_raw_timeline" -q` | 3 passed in 23.43s |
| `python -m pytest tests/unit/test_remaining_strategy_core_contracts.py tests/unit/test_remaining_strategy_core_snapshots.py tests/integration/parity/test_live_shadow_layer2.py -k "vdub or momentum_r1b_nqdtc" -q` after explicit replay point-value materialization | 17 passed in 44.46s |
| `python -m pytest tests/unit/test_nqdtc_core.py tests/unit/test_nq_regime_core.py tests/unit/test_nq_regime_live_engine.py tests/unit/test_remaining_strategy_core_contracts.py tests/unit/test_remaining_strategy_core_snapshots.py tests/unit/test_real_order_correlation_contract.py tests/unit/test_live_completed_bar_wiring.py tests/unit/test_momentum_family_portfolio_replay.py tests/unit/test_momentum_portfolio_synergy_live_parity.py tests/unit/test_in_memory_repo_family_scoped_methods.py tests/integration/parity/test_live_shadow_layer2.py tests/integration/parity/test_live_shadow_families.py -q` | 125 passed, 1 skipped in 99.31s |
| `python -m compileall -q strategies/momentum/vdub backtests/momentum/engine/vdubus_engine.py tests/integration/parity` | passed |
| `python -m ruff check` on the new strategy core and changed core/parity tests | passed |
| `git diff --check -- backtests/momentum/engine/vdubus_engine.py strategies/momentum/vdub tests/integration/parity tests/unit/test_remaining_strategy_core_contracts.py` | passed; line-ending conversion warnings only |

The one skip is the existing optional PostgreSQL repository comparison because `PARITY_POSTGRES_DSN` was not set; the in-memory repository tests ran.

## Remaining R1B work

Downturn, the frozen overlapping historical slice, remaining family feedback scenarios, and full simultaneous Momentum completion remain unstarted. Resize and partial-fill effects on later children, cancellation/rejection release of shared risk, realized-loss/cooldown propagation, and the full materialized historical replay remain for later reviewed R1B increments. R2, optimizer integration/cutover, cache and worker determinism, broker calibration, and production-confidence claims remain explicitly out of scope.
