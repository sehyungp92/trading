# Family Causal Replay and Thin-Backtest Implementation Plan

**Status:** active scope-reduced plan; Momentum R1 complete and R2A fast-path feasibility in progress
**Date:** 2026-07-17
**Revision:** surgical clarification of R2 feasibility sequencing, identity versus semantic digests, and safe per-candidate derived state views
**Supersedes:** the 2026-07-15 G0-G8 platform-first plan and its 64-item completion checklist
**Applies to:** Momentum, Stock, and Swing family replay, phased auto-optimization, portfolio authorization, deterministic execution, and live/replay parity
**Governing intent:** `docs/strategy-implementation-lessons.md` and `docs/live_backtest_parity_plan.md`, with broker calibration treated as a separate production-confidence track

## 1. Decision

The implementation will close the most consequential gap between bounded fixture parity and simultaneous historical family parity without rebuilding the live runtime or introducing a new parallel trading platform.

The reusable authority is the deterministic trading domain already used, or intended to be used, by live:

```text
market/session event
    -> strategy decision core
    -> family portfolio authorization
    -> order lifecycle
    -> deterministic or live execution adapter
    -> order/fill feedback
    -> strategy and shared portfolio state
```

The two shells remain deliberately different:

- Live normalizes `ib_async` callbacks, calls the shared domain logic, submits through the live OMS, and persists operational state.
- Replay supplies historical events in virtual time, calls the same domain logic, executes through an in-memory deterministic model, and collects metrics.

The optimizer must not connect to IBKR, start the live coordinator stack, sleep, write databases, emit verbose instrumentation, or construct a fake live service for every candidate. It must rerun every causally relevant state transition as fast as the CPU permits.

An `await` is not itself the performance problem; network I/O, timers, queues, persistence, subprocesses, and repeated runtime startup are. The shared decision boundary should nevertheless remain synchronous and deterministic, with async confined to the live shell.

Success is earned independently per family. Momentum may cut over when Momentum passes; it does not wait for Stock, Swing, broker calibration, or a repository-wide abstraction programme.

From this point forward, admit a change only when it closes a measured causal/parity gap, removes a measured per-candidate hot-path cost while preserving exact semantic products, or supplies evidence required by the current family's next gate. Speculative generalization, pre-building for later families, unrelated cleanup, and replacing an adaptable existing seam remain out of scope.

### 1.1 Current implementation status

- Momentum R0 and R1 are complete and frozen at commit `6428d859c0d20d7c15831c51b78beaf5eb2aade0`.
- The simultaneous historical oracle passes P00, P01, P10, and P11 with all four real children, shared causal state, and exact normalized live/replay products.
- R2A-1 prepared stateless features and monotonic cursors are frozen at commit `bf719233316281d1adfc876a55cf1cb425aeee4e`; its exact semantic products, 90.380-second warm median, and 242.78 MiB peak RSS improved the frozen R1 baseline but did not meet the 6.43-second hard-feasibility threshold.
- R2A-2 mutation-versioned per-candidate repository risk views reached its review boundary with exact scan-versus-view and historical products. The targeted repository risk stack fell from approximately 42.6 seconds cumulative to 4.24 seconds, while the 110.87-second warm median was host-inconclusive and does not establish total-wall improvement or R2A feasibility.
- R2A-3 is frozen at commit `c723ca20852e9a3a36729b1835223d4bd4874c4a`; exactness and memory passed, with a 56.863-second warm aggregate median. The 6.43-second gate failed by 8.84×, so R2B remains unauthorized; the next boundary is a bounded aggregate-only feasibility decision.

## 2. The two questions and the intended assurance

### 2.1 Bounded live/replay oracle

> Given this exact seeded state, market input, strategy fixture, and broker event script, do the selected live and replay paths emit equivalent normalized orders, events, ledger rows, and final state?

This remains the high-detail oracle for representative interactions. It uses the real live wrappers with fake broker infrastructure only in bounded tests.

### 2.2 Historical simultaneous-family replay

> If all family strategies trade simultaneously over a historical period using a materialized candidate configuration, will live and replay generate the same causal sequence under shared portfolio state and execution?

The optimizer answers this by driving all strategies from one ordered historical timeline and rerunning their state machines under shared capital, positions, exposure, working orders, realized PnL, and execution feedback.

### 2.3 What can and cannot be claimed

| Claim | Required evidence | Status attainable |
|---|---|---|
| Shared-domain parity | Live and replay call the same deterministic decision and lifecycle functions | Strong and mechanically testable |
| Historical family causality | Every candidate reruns simultaneous strategies with shared state and authorization-before-fill | Strong and mechanically testable |
| Execution-model parity | Deterministic fill/cancel rules are tested and calibrated against selected broker evidence | Incrementally improvable |
| Exact future outcome parity | Replay predicts every live latency, queue position, race, and fill | Impossible and never claimed |

The optimizer cutover requires the first two claims and an honest baseline execution model. Broker-backed calibration improves the third claim but is not a prerequisite for removing completed-trade replay.

## 3. Scope discipline

### 3.1 Required now

- raw market/session events drive all strategies in a family;
- one immutable materialized candidate configuration is used for the run;
- strategy decisions occur before family authorization;
- authorization occurs before submission and fill;
- denials, resizes, acknowledgements, fills, partial fills, cancels, rejects, working orders, realized PnL, and cooldown effects feed back before later decisions;
- all strategies share portfolio capital, exposure, positions, and working-order reservations;
- replay time and same-time ordering are deterministic;
- the hot path is in memory and performs no external I/O;
- bars, calendars, immutable metadata, and configuration-independent features may be cached;
- completed trades, known PnL, and state-dependent fixed intents may not be official candidate inputs;
- selected live-wrapper and replay fixtures compare normalized semantic outputs;
- each family meets an explicit optimizer runtime and memory budget before cutover.

### 3.2 Bounded oracle or finalist scope

The following do not run for every candidate:

- stressed partial-fill and delayed-ack profiles;
- fill/cancel races and duplicate callbacks;
- replacement, reconnect, hydration, and restart fixtures;
- verbose full traces and first-divergence reports;
- real live coordinator startup with fake broker infrastructure;
- database, repository, and instrumentation equivalence checks;
- broker-backed calibration.

These remain valuable tests. They are invoked when the affected shared reducer changes, in the bounded parity suite, or for finalists and promotion. Existing transport-neutral OMS/order/fill tests remain authoritative for lifecycle semantics. Add a family-level form only when that event can alter later strategy or shared portfolio behavior, and prefer one parameterized fixture over separate scenario modules.

### 3.3 Explicitly deferred unless evidence makes them necessary

- a repository-wide universal event platform;
- migrating unrelated live operational orchestration into replay;
- replacing working, transport-neutral OMS components merely to fit a new abstraction;
- materializing every static constant when a stable defaults/code fingerprint is sufficient;
- a generic execution profile for every broker race before a family baseline works;
- process pools, shared memory, or feature stores before profiling demonstrates a need;
- repository copies of large results already preserved by Git or the existing output store;
- per-gate verifier programmes, owner-approval schemas, and repeated audit receipts;
- removing every legacy comparison path before the replacement for that family passes.

Deferred work must not be silently reintroduced as a prerequisite for a family cutover.

## 4. Reuse-first rule

The default action is to call, extract, or adapt an existing component. Creating a replacement requires evidence that the existing component cannot express the required deterministic transition.

| Existing asset | Planned use |
|---|---|
| Strategy `core` modules and `on_bar`, `on_order_update`, `on_fill` surfaces | Remain the decision and lifecycle authority; expose only missing pre-decision seams |
| `strategies/contracts.py` runtime overrides | Inject offline execution, OMS, clock, and repository ports where already supported |
| `libs/oms/risk/portfolio_rules.py` | Supply one shared pure authorization/sizing decision; live builds its snapshot asynchronously and replay builds it in memory |
| `libs/oms/engine/state_machine.py` | Remain the order-status authority |
| `libs/oms/engine/fill_processor.py` and existing ledger/repository behavior | Reuse or minimally extract deterministic arithmetic and idempotency; do not build a second OMS |
| Existing family coordinators | Supply live-wrapper oracle behavior and decision priority |
| Existing `SimBroker` and strategy-specific bar-fill rules | Supply the baseline deterministic execution model where semantically adequate |
| `tests/integration/parity/live_runners.py`, `replay_runners.py`, and fake IBKR infrastructure | Extend for selected simultaneous-family oracle fixtures |
| Existing in-memory repositories and sinks | Use in replay instead of adding parallel repository frameworks |
| Existing optimization data loaders and feature calculations | Retain outside the causal state loop and cache only when configuration-independent |

### 4.1 Existing G1-G4M experimental work

The previous branches and artifacts are implementation inputs, not a package to merge wholesale and not prerequisites that must all be completed.

For each component, use this disposition:

| Disposition | Rule |
|---|---|
| Keep | It is used by a real live wrapper and the real replay path in the same family, or directly replaces duplicated deterministic logic |
| Simplify | It provides needed ordering, configuration identity, or trace comparison but exposes more schema or indirection than the vertical slice requires |
| Defer | It implements recovery, stressed execution, generalized repositories, or audit governance not needed by the fast baseline |
| Drop | It is scaffold-only, duplicates an existing authority, or is used only to satisfy an obsolete gate |

Likely salvage candidates include the extracted pure portfolio rules, deterministic ordering primitives, shared bar-execution rules, useful NQDTC decision extraction, and behavior-focused tests. Generic runtime, recovery, repository, compatibility, and evidence machinery must justify itself against a real vertical consumer.

Git history already preserves the previous plan and evidence. Do not copy them into another archive.

### 4.2 Verified starting points

Use the repository's mixed current state rather than treating every family as equally incomplete:

| Surface | Verified starting point | Minimal action |
|---|---|---|
| Offline live seam | Real coordinators already accept injected OMS builders, repositories, and fake execution | Reuse it unchanged as the bounded live oracle |
| Portfolio and OMS | Portfolio rules, broker-neutral order transitions, fill processing, and in-memory repositories already exist | Extract only deterministic calculation blocked by async providers or persistence; do not replace the components |
| Momentum | Family evaluation consumes completed source outcomes; NQ_REGIME is closest to shared-core replay | Use NQ_REGIME for R1A, then expose only missing child decision seams |
| Stock | Family candidates reuse completed ALCB/IARIC bundles | Rerun both child cores under shared family state |
| Swing | ATRSS authorizes candidates before submission; Helix authorizes after position creation; TPC is readmitted as already filled | Preserve ATRSS and correct only the Helix and TPC causal boundaries |

## 5. Minimum runtime shape

### 5.1 Family replay driver

The implementation may use controlled mutable state for speed. It does not need an immutable universal reducer API if the existing core owns deterministic mutation correctly.

The minimum behavior is equivalent to:

```python
for event in ordered_historical_timeline:
    for strategy in strategies_due_for(event):
        actions = strategy.on_event(event, strategy_state, config)

        for action in actions:
            if action.opens_or_increases_exposure:
                decision = portfolio.authorize(action, shared_state, config)
                strategy.on_authorization(decision)
                if decision.denied:
                    continue
                action = action.with_authorized_quantity(decision.quantity)

            order = oms.submit(action)
            for broker_event in execution_model.accept(order, event):
                oms.apply(broker_event)
                portfolio.apply(broker_event)
                strategy.apply(broker_event)
```

Generated broker events are inserted into the same deterministic queue. They are applied before any causally later market or strategy event.

### 5.2 Minimum common data

Use existing domain types where possible. At boundaries, the replay needs only enough stable information to order, route, compare, and deduplicate:

- virtual timestamp;
- same-time phase and monotonic sequence;
- family, strategy, and instrument identity;
- semantic event or action kind;
- order/correlation identity where applicable;
- behavior-relevant payload;
- optional causation identity for diagnostic traces.

Do not migrate every internal object to a canonical envelope. Normalize at the parity boundary unless the shared runtime genuinely needs the field.

### 5.3 Same-time ordering

Declare one small policy and test it:

1. session/calendar changes;
2. completed market data;
3. strategy decisions in declared family priority;
4. portfolio authorization feedback;
5. submissions and acknowledgements;
6. fills, cancels, rejects, and replacements in scripted sequence;
7. position, PnL, cooldown, and lifecycle feedback;
8. observations and checkpoints.

Live fixtures preserve the scripted broker arrival sequence. Replay must not depend on dictionary order, wall clock, random UUIDs, or asyncio task scheduling.

### 5.4 Materialized configuration

A candidate run must snapshot:

- candidate strategy parameters;
- family allocation, sizing, and risk parameters;
- strategy priority and enablement;
- execution and cost profile identities;
- relevant runtime defaults;
- data, feature-policy, and code identities.

Both live oracle and replay construct strategy cores through the same configuration factory. Static defaults need not all become new dataclass fields immediately: if a value is not candidate-controlled and remains static, its source/defaults fingerprint may identify it. Any setting that differs between live and replay or varies by candidate must be injected explicitly.

## 6. Execution modes

### 6.1 Causal fast mode - every candidate

- synchronous virtual-time loop;
- preloaded arrays and features;
- in-memory strategy, portfolio, order, position, and ledger state;
- baseline deterministic bar execution;
- primitive decisions, orders, fills, trades, portfolio observations, aggregate metrics, and a rolling semantic digest without retaining a verbose object trace;
- no JSON serialization, database access, logging, process startup, sleeps, or full traces inside the hot loop.

P00 run identity and the semantic-product digest are separate products. Run identity covers data, configuration, initial state, ordering, execution, cost, feature policy, and behavior-relevant code. The semantic-product digest covers normalized causal products in order. An implementation change may therefore change P00 as intended while still proving exact semantic-product equivalence through a diagnostic comparison.

### 6.2 Diagnostic replay - failures and sampled candidates

Uses exactly the same state transitions as fast mode but retains normalized decisions, orders, events, ledger rows, and state digests. A candidate may be rerun in this mode to locate the first divergence.

### 6.3 Live-wrapper oracle - bounded tests

Runs selected deterministic fixtures through the actual family coordinator/live adapters with fake broker infrastructure, then runs the same fixture through replay. It compares semantic outputs after narrow, documented normalization of volatile transport fields.

### 6.4 Stressed replay - finalists

Partial-fill, delayed-ack, cancellation, rejection, and illiquidity profiles run where the optimization approval policy requires them. Exploratory candidates use the baseline profile unless a family specifically depends on stressed execution for correct signal state.

### 6.5 Broker calibration - separate promotion track

Recent broker evidence may tune or validate the deterministic execution model. Failure or absence withholds the broker-calibrated production-confidence label; it does not force phased optimization through `ib_async` or invalidate shared-domain causal replay.

## 7. Non-negotiable causal invariants

1. No official candidate consumes completed child trades, known child PnL, or a fixed state-dependent intent stream.
2. No position exists before an authorized order receives a fill.
3. A denial produces no submission, fill, or transient position.
4. A resize changes submitted, filled, protected, and exited quantity.
5. Working orders reserve the same risk capacity used by later sibling decisions.
6. Partial fills update exposure before later family decisions.
7. Realized PnL and cooldown changes affect later decisions.
8. Duplicate broker events have one economic effect.
9. Every strategy in the family observes one ordered timeline and shared portfolio state.
10. Candidate state, identifiers, and caches do not leak between runs.
11. Fast and diagnostic modes produce identical metrics, final state, and semantic-product digests under the same P00 identity.
12. Performance work may remove I/O and repeated computation, never causal transitions.
13. A parity comparison is invalid unless the seed, normalized market input, materialized configuration, initial state, ordering policy, execution/cost profile, and broker event script identities match.

### 7.1 Migration drift policy

Each migrated decision or lifecycle seam has two checkpoints:

1. **Compatibility extraction:** under the old inputs and execution assumptions, the available pre- and post-extraction normalized decisions, orders, fills, trades, accounting, and final digest remain exact. Products the legacy path never emitted are not invented for compatibility evidence. Any difference in comparable products is extraction drift and must be fixed.
2. **Causal correction:** completed-trade replay, post-fill authorization, or missing feedback is then removed as an explicitly named behavior change. Equality with the legacy result is not required, but the first divergence must be expected and tested.

Use existing baselines and compact trace digests for the compatibility checkpoint; do not create another archived result tree. Legacy and causal paths must still agree in degenerate fixtures where every entry is approved, fills are immediate and complete, there is no shared-state contention, timing and costs match, and no later decision depends on altered state. Purpose-built denial, resize, partial-fill, working-order, cooldown, or shared-capital fixtures must diverge first at the expected causal event.

## 8. Caching and performance contract

### 8.1 Safe caches

- normalized bars and session calendars;
- corporate actions and immutable instrument metadata;
- configuration-independent indicators and feature tensors;
- deterministic execution lookup tables;
- parsed materialized base configuration;
- source/data/feature fingerprints.

### 8.2 Conditional caches

Candidate-dependent warm-up may be cached only when its key includes every relevant candidate and execution dependency and cold-versus-warm tests are exact.

Pre-authorization opportunities may be cached only if a proof and test show that later opportunity generation cannot depend on admission, fill, position, working orders, cooldown, realized PnL, capital, or exposure. The current families should be assumed not to meet that condition.

A derived risk or order index inside one candidate's in-memory state is allowed when every economic mutation invalidates or updates it and scan-versus-index tests are exact. It must not be shared between candidates or become a second repository or OMS authority.

### 8.3 Prohibited official caches

- completed trades;
- known entry/exit pairs or PnL;
- fixed intent streams derived from a counterfactual state history;
- mutable strategy or family state shared between candidates;
- unkeyed configuration-dependent warm-up.

### 8.4 Performance gate

Before a family optimizer cuts over:

- record one representative frozen window, candidate cohort, hardware description, and worker count;
- record the maximum acceptable phase wall-clock and peak-memory budget in one compact summary;
- measure causal fast mode cold and warm at least three times;
- report startup/data preparation separately from candidate transition time;
- verify candidate results across supported worker counts;
- profile before adding new cache or parallel infrastructure.

Derive a single-candidate feasibility threshold from the frozen cohort budget after shared preparation. If one prepared candidate exceeds that threshold, profile and optimize the replay-only path before optimizer integration, cohort execution, or process parallelism. For Momentum, the 28-candidate budget implies 4.29 seconds per candidate for the target and 6.43 seconds per candidate for the hard ceiling.

The semantically narrower completed-trade evaluator is a historical reference, not a like-for-like performance requirement. Approval is based on the operational phase budget and preserved optimization throughput, not on pretending the two evaluators perform the same work.

## 9. Delivery sequence

### Phase R0 - Reset scope and salvage

**Objective:** stop horizontal platform growth and select only components needed for a real vertical path.

**Work:**

- freeze new work on the obsolete G0-G8 checklist;
- create one concise keep/simplify/defer/drop table for existing G1-G4M changes;
- select Momentum as the first vertical family because implementation is already in progress;
- select a small historical slice containing overlapping child activity;
- record the optimizer runtime and memory budget;
- retain existing goldens and results in place without copying them.

**Gate R0:**

- no scaffold-only component is required by policy;
- the vertical path names the exact existing strategy, portfolio, OMS, execution, and parity seams it will call;
- baseline tests for touched components pass;
- the performance budget and historical slice are explicit.

### Phase R1 - Momentum vertical causal replay

**Objective:** make one real family causal end to end before expanding generic infrastructure.

#### R1A - One-child proof

Start with NQ_REGIME, the verified Momentum child closest to shared-core operation, unless R0 finds contrary current-code evidence. Drive raw market events through its real decision core, shared portfolio evaluator, existing order/fill transitions, and deterministic execution rules. Connect the corresponding live wrapper to the same decision functions.

This checkpoint must demonstrate:

- raw events rather than prepared trades or fixed intents;
- exact compatibility behavior before causal feedback is changed;
- authorization before submission;
- denial and fill feedback reaching the real strategy state;
- a bounded live-wrapper/replay trace match;
- measured transition throughput.

No generic abstraction may expand at this checkpoint unless the vertical path uses it.

#### R1B - Simultaneous Momentum family

- add the remaining Momentum children to one timeline and declared priority;
- for NQDTC, Vdub, and Downturn, move only the missing signal, eligibility, sizing, order-selection, management, and cooldown logic required for shared causal feedback;
- move logic into a shared strategy-owned core; do not copy it into a replay adapter;
- complete the compatibility-extraction checkpoint for each moved seam before enabling its causal correction;
- delete or reduce the duplicated wrapper/backtest decision block once both wrappers call the shared function;
- use one shared family portfolio, working-order, capital, position, PnL, and cooldown state;
- replay the latest materialized Momentum configuration;
- add contention, denial, resize, working-order, partial-fill, and realized-loss fixtures;
- compare one simultaneous historical slice through live-wrapper oracle and replay.

**Gate R1:**

- all enabled Momentum children are driven from raw historical inputs;
- the official causal evaluator accepts no completed child trades or fixed intents;
- live and replay call the same deterministic decision and lifecycle functions for the migrated path;
- the causal invariants pass;
- compatibility extraction is exact, and the defined degenerate legacy-versus-causal fixtures match after allowed normalization;
- causal feedback fixtures first diverge at the intended denial, resize, fill, working-order, cooldown, or shared-capital event;
- every intentional result change has one first-divergence explanation;
- no third Momentum strategy implementation remains.

### Phase R2 - Momentum fast-path feasibility, optimizer integration, and cutover

**Objective:** prove the causal path is practical in phased auto-optimization before making it an official candidate evaluator.

#### R2A - Replay-only fast-path feasibility

- profile one already-prepared replay candidate without the live-wrapper oracle, repeated determinism runs, or optimizer startup;
- precompute or incrementally maintain only dependency-keyed stateless completed-bar features, session classifications, timeframe availability, and futures-calendar lookups;
- replace growing-history reconstruction with prepared arrays and monotonic cursors;
- maintain exact per-candidate derived risk views inside the existing in-memory repository or reducer state instead of repeatedly scanning unchanged state;
- make aggregate mode retain only primitive metrics, final state, and a versioned rolling semantic-product digest;
- disable callback-side file instrumentation through an explicit offline policy while preserving lifecycle events and economic state;
- implement those output-path changes at the existing historical runner, normalizer, and OMS callback-injection seams; do not introduce a general event/digest platform, alternate OMS, or second candidate evaluator;
- re-profile after each bounded change and retain diagnostic equality with the frozen R1 products;
- do not add workers or integrate the optimizer while one prepared candidate exceeds 6.43 seconds or peak RSS lacks credible headroom below 288 MiB.

If these corrections do not reach feasibility, stop at a new profile checkpoint and locate the remaining leaf-level bottleneck before considering a synchronous orchestration extraction. This is a review gate, not authorization to reconstruct the runtime: any proposal must identify the reused existing authorities, a bounded allowlist and growth budget, and why adapting the current seams cannot meet the gate. Any approved extraction must still call the same strategy decision functions, pure portfolio evaluator, OMS state/fill reducers, and deterministic execution policy; it may not become a second trading engine.

**Gate R2A:**

- repeated fast runs and diagnostic replay are exact in metrics, final state, and semantic-product digest;
- a diagnostic migration run shows no order, event, ledger, or state drift from accepted R1 semantics;
- every cache or prepared product is dependency-keyed and contains no mutable strategy, family, order, position, PnL, cooldown, decision, intent, or fill state;
- one prepared candidate is at or below 6.43 seconds, with 4.29 seconds retained as the target, and peak RSS has credible headroom below 288 MiB;
- no per-candidate external I/O, live coordinator startup, verbose trace retention, or process startup occurs.

#### R2B - Optimizer integration and cohort proof

- integrate the proven R2A evaluator directly into the existing Momentum candidate-evaluation surface;
- preload data and configuration-independent features once per window or worker;
- create fresh, isolated strategy, family, OMS, position, and ledger state for each candidate;
- retain aggregate metrics and semantic-product digests by default, rerunning only failed or sampled candidates diagnostically;
- prove A-B-A candidate isolation, cold-versus-warm equality, and supported worker-count determinism;
- run the representative 28-candidate cohort and meet the 120-second target, 180-second hard ceiling, and 288 MiB peak-RSS ceiling;
- add bounded process-level parallelism only if profiling demonstrates benefit within the memory budget.

**Gate R2B:**

- cold/warm, A-B-A, and worker-count results and rankings are deterministic;
- the operational cohort time and memory budgets pass;
- official candidate evaluation contains no I/O, verbose instrumentation, live-wrapper oracle, or completed-trade input.

#### R2C - Shadow, result migration, and cutover

- replay the latest materialized configuration and classify its first difference from the scoped legacy result;
- shadow the latest configuration and a small candidate cohort without mixing legacy and causal scores in one ranking;
- rerun only affected Momentum optimization phases and regenerate only required winning evidence;
- make causal replay the only official Momentum selection path after the evidence is accepted;
- retain completed-trade replay only as a test/reference path or remove it.

**Gate R2 - Momentum cutover:**

- R2A and R2B pass;
- the latest configuration delta is classified;
- required summaries identify configuration, data, code, feature, and execution profile;
- official Momentum selection cannot call completed-trade replay;
- the old evaluator remains test/reference-only or is removed.

Momentum becomes independently complete at R2.

### Phase R3 - Remaining family verticals

Repeat R1 and R2 separately for Stock and Swing. Choose the next family based on the smallest verified missing shared-decision surface, not on a predetermined platform order.

#### Stock-specific minimum

- drive ALCB and IARIC together from raw data;
- remove complete trade bundles from official selection;
- feed admission, fill quantity, positions, PnL, capital, and cooldown back to both strategies;
- preserve only safe data/features across candidates.

#### Swing-specific minimum

- retain ATRSS candidate-before-submit behavior;
- split Helix candidate generation from position creation;
- remove TPC `entry_already_filled` behavior from official causal entry;
- ensure all three children use submitted/fill quantities and one shared family state.

**Gate R3 per family:**

The R1 causal and R2 performance/cutover criteria pass for that family. One family failure does not revoke another family's completed cutover.

### Phase R4 - Consolidate only proven common seams

**Objective:** reduce duplication after multiple working verticals, not before.

**Work:**

- identify code now used by at least two families;
- consolidate only repeated ordering, configuration, observation, or execution behavior;
- run each completed family's causal and performance gates after consolidation;
- remove temporary compatibility adapters and obsolete scaffold;
- keep operational restart/reconnect/calibration work in its existing parity or production track.

**Gate R4:**

- consolidation reduces or holds product-code size;
- no family acquires a new independent decision or OMS implementation;
- performance remains within each family budget;
- repository dependency direction remains live/replay shells toward shared domain logic.

## 10. Minimal parity matrix

| ID | Scenario | Compared products | Required tier |
|---|---|---|---|
| P00 | Fixture identity | Same fixture materializes identical seed/config/data/state/ordering/execution/script identities; a behavior-relevant mutation changes the applicable identity | Fast path and oracle precondition |
| P01 | Same config and input repeated | Final digest and metrics exact | Fast path |
| P02 | Same-time sibling decisions | Authorization order and reasons exact | Fast path |
| P03 | Portfolio denial | No submit/fill/position; later state exact | Fast path and oracle |
| P04 | Portfolio resize | Order, fill, stop, exit quantities exact | Fast path and oracle |
| P05 | Working order consumes capacity | Later sibling decision exact | Fast path |
| P06 | Partial fill changes exposure | Subsequent decision and ledger exact | Baseline if relevant; otherwise oracle |
| P07 | Reject or cancel | Working state and later eligibility exact | Fast path and oracle |
| P08 | Duplicate callback | One economic effect | Oracle/shared-reducer test |
| P09 | Realized loss or cooldown | Later decision and size exact | Fast path |
| P10 | All family children on historical slice | After P00 passes, orders, events, ledger, and final state exact under the same script | Oracle |
| P11 | Aggregate versus diagnostic mode | Metrics and final digest exact | Fast path |
| P12 | Candidate A, B, then A | Both A results exact | Fast path |
| P13 | Cold versus warm cache | Semantic result exact | Fast path |
| P14 | One versus supported workers | Results and ranking exact | Optimizer gate |
| P15 | Restart/reconnect/replace | State and ledger per existing policy | Oracle when affected |
| P16 | Legacy versus causal discriminator | Degenerate fixture exact; feedback-dependent fixture first diverges at the intended causal event | Fast path |

Normalization may remove only transport volatility such as broker-assigned identifiers or wall-clock receipt fields. Quantity, price, side, order type, decision reason, event order, ledger economics, and final domain state may not be ignored.

## 11. Result migration

For each family:

1. Run the latest materialized configuration on causal replay.
2. Locate the first difference from the scoped legacy result.
3. Classify it as extraction drift, intentional causal correction, execution-model difference, or data/config mismatch.
4. Fix extraction drift and unexplained differences.
5. Approve intentional causal corrections through tests, not through relabeling old evidence.
6. Rerun only optimization phases affected by the changed causal semantics.
7. Persist the selected configuration and concise winning evidence.

Existing result artifacts remain valid for their original portfolio-only or mixed evidence scope. Git history and existing output locations are sufficient preservation; do not duplicate full result trees into `docs/audit`.

## 12. Repository-growth controls

1. No new production module may be scaffold-only at the end of a phase.
2. A new generic abstraction needs two real consumers, except for the first minimal vertical implementation.
3. Strategy extraction must move authority; it may not leave live, source backtest, and causal copies.
4. Existing OMS, execution, repository, and parity components are adapted before replacements are considered.
5. Gate checks use normal unit/integration tests. Add a dedicated verifier script only when it will remain a reusable CI or release tool.
6. Commit at most one concise evidence summary per family cutover; keep raw benchmark and trace output outside version control unless it is small and indispensable.
7. Full traces are generated on demand, not stored for every candidate.
8. Temporary migration adapters are deleted after the family cutover.
9. Every phase reports product/test/evidence diff statistics and identifies copied or deleted decision logic.
10. A phase is not approved if it leaves a third trading engine or grows a generic framework without advancing a real family path.

## 13. Rollout and rollback

- Cut over one family at a time behind an explicit evaluator selection.
- Do not mix legacy portfolio-only scores and causal scores in one ranking.
- Shadow the latest configuration and a small candidate cohort before official cutover.
- After a family passes its causal and performance gates, make causal replay its official optimizer path.
- If a defect appears, roll back that family to its last approved evaluator while retaining the failing causal trace.
- Once corrected causal results have been approved and re-optimization completed, rollback targets the last approved causal version, not completed-trade replay.

## 14. Finite checklist

### Scope and reuse

- [x] 1. Record the G1-G4M keep/simplify/defer/drop disposition.
- [x] 2. Confirm the verified family starting-point map and name the reused strategy, portfolio, OMS, execution, repository, and parity seams for the first vertical.
- [x] 3. Freeze one representative slice and operational performance budget.

### Momentum vertical and cutover

- [x] 4. Complete the one-child raw-event vertical proof with exact compatibility extraction before causal correction.
- [x] 5. Connect its live and replay wrappers to the same decision functions.
- [x] 6. Drive all enabled Momentum children on one ordered timeline.
- [x] 7. Remove completed trades and fixed intents from official Momentum causal input.
- [x] 8. Prove degenerate equality plus denial, resize, working-order, fill, PnL, and cooldown feedback at the expected first divergence.
- [x] 9. Pass the selected simultaneous Momentum oracle slice with matching fixture identities.
- [ ] 10. Meet the single-candidate fast-path feasibility gate, then integrate causal Momentum candidate evaluation.
- [ ] 11. Meet Momentum runtime, memory, determinism, and cache-isolation budgets.
- [ ] 12. Replay the latest config, classify differences, and cut over Momentum.

### Remaining families

- [ ] 13. Complete and cut over Stock using the same per-family criteria.
- [ ] 14. Complete and cut over Swing using the same per-family criteria.

### Consolidation and evidence

- [ ] 15. Consolidate only abstractions proven common by real verticals.
- [ ] 16. Remove temporary scaffold and duplicate decision implementations.
- [ ] 17. Retain compact winning evidence and preserve old results in their original scope.
- [ ] 18. Keep broker calibration and operational lifecycle expansion in the separate promotion track.

## 15. Definition of done

A family is complete when:

- its official optimizer drives every enabled family strategy from raw historical events;
- every candidate follows the complete causal sequence under shared portfolio and execution state;
- live and replay call the same deterministic strategy, portfolio, order, fill, and lifecycle authorities for the migrated path;
- the backtest contains data driving, deterministic execution, in-memory state, analytics, and diagnostics, but no independent trading decisions;
- behavior-preserving extraction is exact and every causal correction has an expected first-divergence test;
- selected bounded live-wrapper fixtures and a simultaneous historical slice use matching fixture identities and agree after narrow normalization;
- the causal fast path meets the family's operational optimization budget;
- the latest configuration has been replayed, differences classified, and affected optimization results regenerated;
- no completed-trade, prefilled-entry, or third-engine path can produce an official selection.

The programme is complete when Momentum, Stock, and Swing independently meet that definition and any shared consolidation preserves their parity and performance.

Broker calibration may still qualify execution-model confidence. The remaining unavoidable gap is then real broker and market uncertainty, not completed-trade replay, post-fill authorization, duplicated strategy decisions, or a slow reproduction of the live asynchronous shell.
