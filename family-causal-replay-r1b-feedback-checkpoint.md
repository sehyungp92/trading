# Family causal replay R1B feedback-closure checkpoint

Status: the accepted bounded Downturn entry checkpoint is frozen in commit `48bfbcf` (`feat(momentum): checkpoint Downturn entry replay`). That commit was created from the following literal 15-file allowlist; the cached diff contained all 15 paths, no unexpected path, and no missing path. Unrelated raw data, manifests, diagnostics, generated outputs, and state files were excluded and remain unstaged.

```text
backtests/momentum/engine/downturn_engine.py
strategies/momentum/downturn/core/entry_decision.py
strategies/momentum/downturn/core/logic.py
strategies/momentum/downturn/core/serializers.py
strategies/momentum/downturn/core/state.py
strategies/momentum/downturn/engine.py
tests/integration/parity/harness.py
tests/integration/parity/live_layer2.py
tests/integration/parity/replay_layer2.py
tests/integration/parity/replay_runners.py
tests/integration/parity/source_inputs.py
tests/integration/parity/test_live_shadow_layer2.py
tests/unit/test_downturn_core.py
tests/unit/test_real_order_correlation_contract.py
family-causal-replay-r1b-downturn-entry-checkpoint.md
```

Only the bounded R1B four-child family-feedback closure is complete for review on 2026-07-16. The changes described below are deliberately unstaged and uncommitted. This does not claim a historical slice, full Momentum completion beyond the bounded fixture, R2, optimizer integration or cutover, restart/reconnect coverage, broker calibration, or production-confidence parity.

## Closure architecture

The existing four-child fixture builder now supports later Downturn raw events and broker events phased after a named raw event. It remains the same fixture framework. The raw prefix is still NQDTC, Vdub, Downturn, and NQ_REGIME; only a later Downturn event is permitted by this bounded replay runner.

The live product continues through the real Momentum family coordinator, strategy-owned cores, family portfolio checks, per-strategy OMS services, fake IBKR adapters, fill processor, shared in-memory repository, engine feedback, fill ledger, and normalized state builders. The source replay continues through the same strategy-owned cores and production portfolio-rule functions, then sends every generated action and broker event through the existing multi-strategy OMS, fill processor, ledger, and in-memory repository sink. No completed trade, fixed intent, known PnL, or mutable candidate state is an input.

Two deterministic defects in existing OMS wiring were exposed by the new lifecycle oracle and corrected in the existing factory. Ack, reject, status, and fill-fallback callbacks now use the injected event clock, and the single- and multi-OMS risk trade date uses the same clock. Without those corrections, otherwise identical cancel/reject products received different wall timestamps and a historical fill was reset against the host's current date. This is the only production-code change. Replay control actions also promote core client IDs to repository OMS IDs before cancel or replace, matching the live wrapper's existing promotion behavior.

No strategy decision logic, generic runtime, fixture package, top-level module, persistence implementation, cache infrastructure, or evidence verifier was added.

## Bounded scenarios and first divergences

All six feedback fixtures materialize raw market input, configuration, initial state, ordering policy, cost/execution inputs, and broker script in the runtime source identity. Every live/replay pair has identical fixture identity and exact normalized order, terminal-event, fill-ledger, and state products.

| Scenario | Expected first divergence from the no-feedback path | Proved result | Fixture identity |
|---|---|---|---|
| cancel release | Downturn's working entry becomes `CANCELLED` immediately after `downturn_1`, before NQ_REGIME consumes its raw event | The `$1,134.00` Downturn short reservation is released; the later `$122.50` NQ_REGIME short proposal fits below the `$1,250.00` cap and is approved | `718264f1f90c9d4b47d5b620300cbb0632d60392845ab2ba32c00face8203fba` |
| reject release | Downturn's working entry becomes `REJECTED` immediately after `downturn_1`, before NQ_REGIME | The same reservation release and later-child approval occur through the reject callback | `169d4a7340176e06e2aa499da6eb38be4332ccc9dc78377c7ec9377a0bd01236` |
| NQ partial exit | The first NQ target fill follows its entry fill | Quantity falls from 5 to 3, open risk from `$280.00` to `$168.00`, realized PnL becomes `+$56.00`, and the real core requests the protective stop replacement to breakeven | `19793b891c0abf17aeaf0f9c0610529845ee322886ec6fe438632faa8278b462` |
| realized-loss resize | The NQ stop fill follows its entry fill | The position realizes `-$6,000.00`, equity becomes `$94,000.00`, the 6% drawdown selects the 0.50 tier, and the otherwise identical later Downturn quantity is resized from 42 to 21 | `9510b745670b5fee57f51005f81f8ca826ca585b6047d992cfac38944ad3027b` |
| Downturn cooldown block | Downturn entry and protective-exit fills reset entry age before the later raw event | At one elapsed 5-minute bar the real core remains flat, submits no second entry, and reports `bars_since_last_entry=1` | `a61ad0f9f8cc73ef7dfb23745e8337e96600a1122ce1e81d1d382c71ef9945d7` |
| Downturn re-entry | The same fill-driven reset is followed by 24 elapsed bars | At the configured cooldown boundary the later raw signal is admitted and a second Downturn entry is submitted | `837affd7e0909ae57c541ee33a38912f762e7d1b61d1556ef60d21c7ce65da2d` |

The cancel and reject cases are one parameterized lifecycle test. The cooldown block and re-entry cases are a second parameterized lifecycle test. The existing authoritative OMS callback, fill, strategy-core, order-correlation, and parity suites were extended or reused; no separate lifecycle fixture framework was created.

## Product growth

Relative to frozen commit `48bfbcf`, one production file changed: `libs/oms/services/factory.py` has 34 inserted and 22 deleted lines, net +12. No production file was added, deleted, or moved, and no strategy decision logic was duplicated.

Eleven existing test/parity files changed: 908 inserted and 104 deleted lines, net +804. Most growth is the six materialized scenarios and assertions in the existing four-child fixture builder. This checkpoint report is the only new file. Wrapper deletions are zero because this closure connects existing feedback paths rather than extracting another strategy decision surface.

## Performance

The measured fixture is `realized_loss_resize`, the closure's five-raw-event, six-authorization, five-broker-feedback path. Cold imports took 3.361127 seconds and one-time fixture, fingerprint, lineage, and temporary-context preparation took 0.035622 seconds. The fingerprint and immutable lineage were prepared once. Lifecycle instrumentation writers were disabled only in the timed loop, and one temporary context and event loop were reused. The timed candidate path performed no per-candidate JSON, database writes, logging, subprocess startup, sleeps, live coordinator startup, or external I/O.

After 10 warm-ups, five 50-candidate batches took 4.528099, 2.789471, 3.909573, 4.386117, and 5.645417 seconds. Median transition processing was 0.087722346 seconds/candidate, or 11.40 candidates/second. Defining the reported observable causal boundary as five raw events plus six portfolio authorizations plus five broker feedback applications gives 182.39 transitions/second. The same run processed 57.00 raw events/second, 68.40 authorizations/second, and 57.00 broker feedback events/second. Generated strategy actions and internal OMS state-machine steps are executed but are not inflated into that boundary count.

Prepared RSS was 75.277 MiB. Sampled peak RSS was 79.637 MiB, a 4.359 MiB hot-loop increase. A 28-candidate bounded cohort projects to approximately 2.46 seconds of transition processing. The accepted four-child entry-only checkpoint measured 11.73 candidates/second on the same host and method family; the feedback closure measured 11.40 candidates/second while processing the additional raw event and lifecycle feedback. This remains suitable for the bounded optimization hot path, but it is not the deferred historical-slice or optimizer-cutover benchmark.

## Verification

| Command | Result |
|---|---|
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py -q -k momentum_r1b_downturn` before closure edits | 4 passed, 13 deselected in 28.04s |
| `python -m pytest tests/unit/test_downturn_core.py tests/unit/test_nq_regime_core.py -q -k "downturn or partial or final_stop_fill or cancelled_entry"` before closure edits | 12 passed, 21 deselected in 5.76s |
| `python -m pytest tests/unit/test_audit_fixes.py tests/unit/test_oms_atomic_persistence.py -q -k "cancel_status or partial_fills or submit_failure_persists_rejection"` before closure edits | 3 passed, 108 deselected in 3.78s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py -q -k "releases_working_risk" -vv` | 2 passed, 22 deselected in 26.00s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py::test_momentum_r1b_nq_partial_exit_updates_family_exposure -q -vv` | 1 passed in 15.87s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py::test_momentum_r1b_realized_loss_resizes_later_downturn_quantity -q` | 1 passed in 7.62s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py::test_momentum_r1b_downturn_fill_drives_cooldown_and_reentry -q -vv` | 2 passed in 14.17s |
| `python -m pytest tests/unit/test_oms_atomic_persistence.py -q -k "callbacks_serialize_status_ack_then_fill"` | 2 passed, 50 deselected in 1.77s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py -q -k "momentum_r1b"` | 18 passed, 6 deselected in 68.66s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py -q` | 24 passed in 91.01s |
| `python -m pytest tests/unit/test_downturn_core.py tests/unit/test_nq_regime_core.py tests/unit/test_real_order_correlation_contract.py -q` | 42 passed in 20.51s |
| `python -m pytest tests/unit/test_audit_fixes.py tests/unit/test_oms_atomic_persistence.py -q -k "cancel_status or partial_fills or submit_failure_persists_rejection or callbacks_serialize_status_ack_then_fill"` | 5 passed, 106 deselected in 2.10s |
| Broader relevant 14-file Momentum core, lifecycle, portfolio, and parity command | 152 passed, 1 skipped in 139.44s |
| `python -m compileall -q libs/oms/services/factory.py tests/integration/parity tests/unit/test_oms_atomic_persistence.py` | passed |
| `python -m ruff check --ignore E731,F841,F821` on the 12 changed code/test files | passed; the three ignored rule classes are existing diagnostics outside this subincrement's lines |
| `git diff --check --` on the feedback allowlist | passed; line-ending conversion warnings only |

The one broader-suite skip is the existing optional PostgreSQL repository comparison because `PARITY_POSTGRES_DSN` was not set; the in-memory repository paths ran.

The exact broader-suite command was:

```powershell
python -m pytest tests/unit/test_nqdtc_core.py tests/unit/test_nq_regime_core.py tests/unit/test_nq_regime_live_engine.py tests/unit/test_remaining_strategy_core_contracts.py tests/unit/test_remaining_strategy_core_snapshots.py tests/unit/test_downturn_core.py tests/unit/test_downturn_instrumentation.py tests/unit/test_real_order_correlation_contract.py tests/unit/test_live_completed_bar_wiring.py tests/unit/test_momentum_family_portfolio_replay.py tests/unit/test_momentum_portfolio_synergy_live_parity.py tests/unit/test_in_memory_repo_family_scoped_methods.py tests/integration/parity/test_live_shadow_layer2.py tests/integration/parity/test_live_shadow_families.py -q
```

It produced `152 passed, 1 skipped in 139.44s`.

## Review boundary

Stop here for checkpoint review. The historical slice, any additional Momentum expansion, R2, optimizer integration or cutover, restart/reconnect coverage, generic runtime work, cache or worker changes, and broker calibration remain explicitly deferred.
