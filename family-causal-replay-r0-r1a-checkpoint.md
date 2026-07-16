# Family causal replay R0/R1A checkpoint

Status: accepted and frozen on 2026-07-16 as the R0 and NQ_REGIME R1A baseline. This checkpoint does not authorize or claim R1B, Momentum family parity, optimizer cutover, broker calibration, Stock, Swing, or operational lifecycle expansion.

## R0 disposition

No G1-G4M branch was merged wholesale and no historical evidence tree was copied.

| Disposition | G1-G4M salvage decision |
|---|---|
| Keep | The behavior-focused ordering and identity requirements, the pure-portfolio-rule extraction idea, and useful compatibility/first-divergence tests. They are implemented through the existing runtime fingerprint, parity normalizers, `PortfolioRuleChecker`, NQ_REGIME core, and normal tests. |
| Simplify | G2's portfolio reducer is only the provider-free static prefix in `libs/oms/risk/portfolio_rules.py`. G1/G3 ordering remains the order already expressed by the one-child fixture, core authorization transition, OMS, broker script, and SimBroker. No canonical event framework is introduced. |
| Defer | State-dependent family caps, sibling contention, working-order reservations, partial-fill effects on later family decisions, generic simultaneous ordering, recovery/restart/replace expansion, remaining Momentum children, generic repositories, and optimizer integration. These are R1B or later. |
| Drop | The proposed `libs/family_causal_replay` package, a replay-only/third causal strategy implementation, generic runtime and port scaffolds, compatibility archives, evidence JSON collections, and dedicated gate verifier scripts. |

Exact reused seams are `CompletedBarPolicy`; NQ_REGIME `on_bar`, `on_order_update`, and `on_fill`; `CoordinatorRuntimeOverrides`; `MomentumFamilyCoordinator`; `PortfolioRuleChecker`; the OMS intent/state/fill path; `SimBroker`; fake IBKR; parity runners/normalizers; and in-memory repositories. NQ_REGIME remains the closest shared-core child; current code provided no contrary evidence.

The frozen future R1B historical slice is 2026-03-20 14:00-19:30 UTC. Retained historical outputs show activity from Downturn, NQDTC, Vdub, and NQ_REGIME in that interval, including overlapping exposure. The eventual causal run must consume raw events, not those completed outputs. The current raw selection source is `data/raw/NQ_1m_bid_ask.parquet`, 313,834 bytes, SHA-256 `5C85B52A2159E2E2BD516916184C0438506BA6B0B53EE6463124486D9F57C3CC`. No R1B run has begun.

The recorded Momentum operational budget is the retained 28-candidate, one-worker cohort: 120-second target, 180-second hard ceiling, and 288 MiB peak resident memory per worker. This is the R2 cutover budget and remains subject to checkpoint review; R1A only establishes that the transition path is structurally suitable.

## R1A implementation and migration boundary

The existing portfolio checker now delegates its provider-free strategy multiplier, drawdown, regime multiplier, and directional regime multiplier calculation to one synchronous pure function. All async providers and state-dependent rules remain in the existing checker. R1A rejects configurations containing those deferred rules instead of silently approximating them.

The NQ_REGIME core now separates proposal from authorization. Compatibility mode is unchanged. In causal mode, `on_bar` emits the same proposal and decision event but does not mark it working; `on_authorization` commits the OMS-approved quantity or removes a denied proposal. Both the live engine and causal replay wrapper use this core transition. The live shell still delegates authorization and routing to the existing OMS; replay uses the shared pure portfolio decision and existing deterministic fill path.

The production NQ backtest has an opt-in causal authorization setting and uses the existing SimBroker. Its default remains legacy-compatible, so this increment does not cut over the optimizer. The ordinary parity replay also remains in compatibility mode; only the explicit R1A wrapper enables the correction. No completed trade, known PnL, or fixed state-dependent intent is an R1A input.

Compatibility is exact for the pre-correction products. The tested first causal divergence is the working-order commit: after `ENTRY_REQUESTED`, legacy state has `working_entry_order_id` and `order_to_role`, while causal state retains the identical proposal/action/event but waits for authorization. The approved, immediate-full-fill degenerate fixture is exact across legacy replay and the causal live wrapper. The denial fixture first diverges at portfolio authorization and produces `ENTRY_DENIED` with no submission, fill, or position.

No production files were added, deleted, or moved. Six existing production files changed by 384 insertions and 37 deletions. Nine existing test/parity files changed by 571 insertions and 13 deletions. Duplicated static portfolio calculation and integer quantity adjustment were removed from the live checker; no strategy decision block was copied into replay.

## Bounded oracle and performance

The bounded oracle extends `nq_regime_entry_fill.json` in memory with one NQ_REGIME child and static-only portfolio rules. The source is a raw 5-minute market bar plus seeded strategy/repository state and a broker script. The approved full-fill and disabled-strategy denial variants use the real Momentum coordinator/live NQ engine and the causal replay wrapper. Seed, normalized market data, materialized configuration, initial state, ordering, execution/cost profile, and broker-script identity are hashed together. A portfolio-rule mutation changes that identity. Orders, terminal events, ledger, strategy state, portfolio state, and family state match after existing narrow normalization.

Measurements ran on Windows, Python 3.12.6, an Intel i7-9750H (6 cores/12 logical processors), and 16,893,956,096 bytes of physical RAM.

The production causal NQ replay loaded and normalized 59,668 five-minute bars once, then prepared a 420-bar 2026-03-16 through 2026-03-20 window. Imports took 6.386 seconds and data preparation took 9.476 seconds. Three in-memory SimBroker transition runs took 0.530, 0.499, and 0.583 seconds (793.0, 841.6, and 720.8 bars/second); each produced the same seven trades. Peak process RSS, including pandas and the full prepared source, was 207,593,472 bytes (198.0 MiB), below the 288 MiB worker ceiling.

A narrower isolated-candidate measurement, including state hydration plus raw-bar decision, authorization, and immediate fill feedback, processed 500 candidates in 6.166, 5.756, and 6.709 seconds: 74.5-86.9 candidates/second and 223.6-260.6 state transitions/second. Its one-time imports took 1.452 seconds, fixture preparation took 0.011 seconds, peak process RSS was 44,838,912 bytes, and a separate 50-candidate allocation trace peaked at 257,733 bytes. Neither timed loop performed JSON serialization, database access, logging, sleeps, process startup, live coordinator startup, or external I/O.

## Verification

| Command | Result |
|---|---|
| `python -m pytest tests/unit/test_nq_regime_core.py tests/unit/test_nq_regime_live_engine.py tests/unit/test_momentum_portfolio_synergy_live_parity.py tests/unit/test_nq_regime_plugin.py tests/unit/test_stock_portfolio_rules.py tests/unit/test_swing_portfolio_rules.py -q` | 106 passed in 8.67s |
| `python -m pytest tests/unit/test_oms_atomic_persistence.py::test_intent_handler_denial_uses_atomic_helper tests/unit/test_oms_atomic_persistence.py::test_intent_handler_approval_uses_atomic_helper tests/unit/test_oms_atomic_persistence.py::test_intent_handler_account_gate_denial_does_not_persist_approval tests/unit/test_oms_atomic_persistence.py::test_fill_processor_ignores_duplicate_fill_after_race tests/unit/test_oms_atomic_persistence.py::test_fill_processor_serializes_distinct_partial_fills_per_order tests/unit/test_oms_atomic_persistence.py::test_single_oms_callbacks_serialize_status_ack_then_fill tests/integration/parity/test_oms_restart_parity.py -q` | 12 passed in 3.73s |
| `python -m pytest tests/integration/parity/test_live_shadow_layer2.py tests/integration/parity/test_live_shadow_families.py -q` | 9 passed in 35.17s |
| `python -m pytest tests/integration/parity/test_live_shadow_negative.py -k "not idle_child and not configured_idle" -q` | 22 passed in 69.79s |
| `python -m pytest tests/integration/parity/test_live_shadow_negative.py -k "idle_child or configured_idle" -q` | 17 passed in 60.50s |
| `python -m pytest tests/integration/parity/test_parity_acceptance_gate.py -q` | 1 passed, 1 skipped in 0.21s; the skip is the existing broker-calibration gate |
| `python -m pytest tests/integration/parity/test_coordinator_runtime_overrides.py tests/integration/parity/test_fake_ibkr_callback_smoke.py -q` | 6 passed in 14.16s |

The broad checkpoint initially exposed extraction drift in the no-drawdown case: a reduced first drawdown tier was incorrectly applied at exactly initial equity, changing a Swing quantity from 5 to 4. The pure helper was corrected to preserve the existing full-size behavior, a regression test was added, and the affected Swing test plus the split parity suite passed. The monolithic parity command exceeded the local 120-second command ceiling, so the same modules were run in the successful split commands above.

## Remaining R1B work

R1B must add NQDTC, Vdub, and Downturn through strategy-owned shared cores; introduce one declared simultaneous timeline and shared capital/exposure/working-order state; extend the pure portfolio boundary to state-dependent authorization without copying the checker; and prove contention, resize, working-order, partial-fill, realized-loss, and cooldown effects only where they alter later family behavior. It must then run the frozen overlapping slice with matching identities. Optimizer integration, full cohort performance, cache/worker determinism, cutover, Stock, Swing, broker calibration, and operational lifecycle expansion remain outside R1B and outside this checkpoint.
