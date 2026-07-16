from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from typing import Any

from tests.integration.parity.family_decisions import FAMILY_DECISION_STATUSES
from tests.integration.parity.family_surface_names import coordinator_class_name
from tests.integration.parity.family_state import build_family_state
from tests.integration.parity.live_shadow_contract import ParityTrace
from tests.integration.parity.normalizers import (
    normalize_oms_events,
    normalize_order_intents,
    normalize_state_snapshot,
    normalize_trade_ledger,
)
from tests.integration.parity.replay_candidates import ReplayDecisionTimeline
from tests.integration.parity.replay_family_surfaces import (
    family_surface_adapter_name as _family_surface_adapter_name,
    run_family_portfolio_surface as _run_family_portfolio_surface,
)
from tests.integration.parity.replay_family_timeline import (
    apply_replay_oms_outcomes_to_strategy_state as _apply_replay_oms_outcomes_to_strategy_state,
    assert_family_surface_matches_sink as _assert_family_surface_matches_sink,
    authoritative_family_timeline as _authoritative_family_timeline,
)
from tests.integration.parity.replay_idle import (
    idle_replay_strategy_state as _idle_replay_strategy_state,
    replay_idle_market_children as _replay_idle_market_children,
    run_idle_market_core,
)
from tests.integration.parity.replay_layer2 import (
    replay_iaric as _replay_iaric,
    replay_nq_regime as _replay_nq_regime,
    replay_nqdtc_r1b as _replay_nqdtc_r1b,
    replay_tpc as _replay_tpc,
)
from tests.integration.parity.replay_oms import run_replay_oms_sink
from tests.integration.parity.runtime_source import runtime_source_fingerprint
from tests.integration.parity.source_inputs import (
    family_resolver,
    instrument_ticks,
    momentum_r1b_raw_timeline,
    point_value,
    strategy_ids,
)


_FAMILY_DECISION_STATUSES = FAMILY_DECISION_STATUSES
_ReplayDecisionTimeline = ReplayDecisionTimeline
_run_idle_market_core = run_idle_market_core


class _MomentumR1BPortfolio:
    """Small in-memory snapshot builder around the production pure rules."""

    def __init__(self, fixture: Mapping[str, Any]) -> None:
        from tests.integration.parity.portfolio_rules import (
            portfolio_rules_config_from_fixture,
        )

        self.fixture = fixture
        self.config = portfolio_rules_config_from_fixture(fixture)
        if self.config is None:
            raise AssertionError("R1B fixture must materialize portfolio rules")
        self.current_directional_risk_dollars = {"LONG": 0.0, "SHORT": 0.0}
        self.sequence = 0

    def __call__(self, strategy_id: str, action: Any, *, timestamp: Any) -> dict[str, Any]:
        from libs.oms.risk.portfolio_rules import (
            DirectionalRiskSnapshot,
            adjusted_entry_quantity,
            evaluate_directional_cap,
            evaluate_static_portfolio_entry,
        )

        self.sequence += 1
        direction = "LONG" if action.side == "BUY" else "SHORT"
        equity = float((self.fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
        static = evaluate_static_portfolio_entry(
            self.config,
            strategy_id=strategy_id,
            direction=direction,
            current_equity=equity,
        )
        approved_qty = (
            adjusted_entry_quantity(action.qty, static.size_multiplier)
            if static.approved
            else 0
        )
        risk_context = action.risk_context or {}
        planned_entry = float(
            risk_context.get("planned_entry_price")
            or action.price
            or action.limit_price
            or action.stop_price
            or 0.0
        )
        stop_for_risk = float(
            risk_context.get("stop_for_risk")
            or (action.metadata or {}).get("stop_for_risk")
            or planned_entry
        )
        risk_per_contract = (
            abs(planned_entry - stop_for_risk)
            * point_value(self.fixture, action.symbol)
        )
        risk_dollars = risk_per_contract * approved_qty
        ref_urd = float(self.config.reference_unit_risk_dollars or 0.0)
        risk_R = risk_dollars / ref_urd if ref_urd > 0 else 0.0
        denial_reason = static.denial_reason or ""
        if static.approved:
            denial_reason = evaluate_directional_cap(
                self.config,
                DirectionalRiskSnapshot(
                    current_risk_dollars=self.current_directional_risk_dollars[direction],
                    reference_unit_risk_dollars=ref_urd,
                ),
                strategy_id=strategy_id,
                direction=direction,
                new_risk_R=risk_R,
                new_risk_dollars=risk_dollars,
            ) or ""
        approved = static.approved and not denial_reason and approved_qty > 0
        if approved:
            self.current_directional_risk_dollars[direction] += risk_dollars
        else:
            approved_qty = 0
        return {
            "approved": approved,
            "approved_qty": approved_qty,
            "denial_reason": denial_reason,
            "portfolio_decision_ref": f"momentum-r1b-{self.sequence}",
            "family_surface": "momentum_r1b_nqdtc_shared_portfolio",
            "timestamp": timestamp,
            "risk_dollars": risk_dollars if approved else 0.0,
        }


def run_layer2_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    return _run_coro_blocking(_run_replay_trace(fixture))


def run_layer3_family_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    return _run_coro_blocking(_run_family_replay_trace(fixture))


def run_nq_regime_r1a_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    """Run the one-child causal family path without a completed-trade surface."""

    return _run_coro_blocking(_run_nq_regime_r1a_replay_trace(fixture))


def run_momentum_r1b_nqdtc_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    """Run the bounded NQDTC + NQ_REGIME causal subincrement."""

    return _run_coro_blocking(_run_momentum_r1b_nqdtc_replay_trace(fixture))


async def _run_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    ticks = instrument_ticks(fixture)
    family_for_strategy = family_resolver(fixture)
    source_hash = _replay_source_fingerprint(fixture)
    out = ReplayDecisionTimeline(fixture)

    if _surface_enabled(fixture, "TPC"):
        _replay_tpc(fixture, out)
    if _surface_enabled(fixture, "NQ_REGIME"):
        _replay_nq_regime(fixture, out)
    if _surface_enabled(fixture, "IARIC_v1"):
        _replay_iaric(fixture, out)

    out.apply_broker_script()
    sink = await run_replay_oms_sink(
        fixture,
        out.timeline,
        strategy_state=out.strategy_state,
        family_mode=False,
    )

    state = sink.state
    _apply_replay_oms_outcomes_to_strategy_state(state)
    state["family_state"] = build_family_state(
        fixture,
        coordinator_class=_coordinator_class(fixture),
        orders=sink.orders,
        positions=sink.positions,
        strategy_state=sink.state.get("strategy_state", {}),
        strategy_risk=sink.state.get("strategy_risk", {}),
        portfolio_risk=sink.state.get("portfolio_risk", []),
        portfolio_rules=sink.state.get("portfolio_rules", []),
        blocked_reasons=sink.state.get("blocked_reasons", {}),
        surface_adapter=_family_surface_adapter_name(fixture),
    )

    return ParityTrace(
        producer="backtest_replay",
        source_fingerprint=source_hash,
        order_intents=normalize_order_intents(
            sink.submitted_orders,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        terminal_events=normalize_oms_events(
            sink.events,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        trade_ledger=normalize_trade_ledger(
            sink.trade_ledger,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        state_snapshot=normalize_state_snapshot(state),
    )


async def _run_nq_regime_r1a_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    ticks = instrument_ticks(fixture)
    family_for_strategy = family_resolver(fixture)
    source_hash = _replay_source_fingerprint(fixture)
    out = ReplayDecisionTimeline(fixture)

    _replay_nq_regime(fixture, out, causal_authorization=True)
    out.apply_broker_script()
    sink = await run_replay_oms_sink(
        fixture,
        out.timeline,
        strategy_state=out.strategy_state,
        family_mode=True,
    )

    state = sink.state
    _apply_replay_oms_outcomes_to_strategy_state(state)
    state["family_state"] = build_family_state(
        fixture,
        coordinator_class=_coordinator_class(fixture),
        orders=sink.orders,
        positions=sink.positions,
        strategy_state=sink.state.get("strategy_state", {}),
        strategy_risk=sink.state.get("strategy_risk", {}),
        portfolio_risk=sink.state.get("portfolio_risk", []),
        portfolio_rules=sink.state.get("portfolio_rules", []),
        blocked_reasons=sink.state.get("blocked_reasons", {}),
        surface_adapter=_family_surface_adapter_name(fixture),
    )

    return ParityTrace(
        producer="nq_regime_causal_replay",
        source_fingerprint=source_hash,
        order_intents=normalize_order_intents(
            sink.submitted_orders,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        terminal_events=normalize_oms_events(
            sink.events,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        trade_ledger=normalize_trade_ledger(
            sink.trade_ledger,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        state_snapshot=normalize_state_snapshot(state),
    )


async def _run_momentum_r1b_nqdtc_replay_trace(
    fixture: Mapping[str, Any],
) -> ParityTrace:
    ticks = instrument_ticks(fixture)
    family_for_strategy = family_resolver(fixture)
    source_hash = _replay_source_fingerprint(fixture)
    out = ReplayDecisionTimeline(fixture)
    portfolio = _MomentumR1BPortfolio(fixture)

    raw_timeline = momentum_r1b_raw_timeline(fixture)
    grouped = [
        (strategy_id, list(events))
        for strategy_id, events in groupby(
            raw_timeline,
            key=lambda event: str(event["strategy_id"]),
        )
    ]
    grouped_ids = [strategy_id for strategy_id, _events in grouped]
    if grouped_ids != ["NQDTC_v2.1", "NQ_REGIME"]:
        raise AssertionError(
            "R1B bounded timeline must contain one contiguous NQDTC group "
            "before one NQ_REGIME group"
        )
    for strategy_id, events in grouped:
        if strategy_id == "NQDTC_v2.1":
            if len(events) != 1:
                raise AssertionError("R1B bounded timeline expects one NQDTC raw event")
            _replay_nqdtc_r1b(fixture, out, portfolio_authorizer=portfolio)
        else:
            _replay_nq_regime(
                fixture,
                out,
                causal_authorization=True,
                portfolio_authorizer=portfolio,
                market_rows=[event["payload"] for event in events],
            )
    out.apply_broker_script()
    sink = await run_replay_oms_sink(
        fixture,
        out.timeline,
        strategy_state=out.strategy_state,
        family_mode=True,
    )

    state = sink.state
    _apply_replay_oms_outcomes_to_strategy_state(state)
    state["family_state"] = build_family_state(
        fixture,
        coordinator_class=_coordinator_class(fixture),
        orders=sink.orders,
        positions=sink.positions,
        strategy_state=sink.state.get("strategy_state", {}),
        strategy_risk=sink.state.get("strategy_risk", {}),
        portfolio_risk=sink.state.get("portfolio_risk", []),
        portfolio_rules=sink.state.get("portfolio_rules", []),
        blocked_reasons=sink.state.get("blocked_reasons", {}),
        surface_adapter=_family_surface_adapter_name(fixture),
    )
    return ParityTrace(
        producer="momentum_r1b_nqdtc_causal_replay",
        source_fingerprint=source_hash,
        order_intents=normalize_order_intents(
            sink.submitted_orders,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        terminal_events=normalize_oms_events(
            sink.events,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        trade_ledger=normalize_trade_ledger(
            sink.trade_ledger,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        state_snapshot=normalize_state_snapshot(state),
    )


async def _run_family_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    ticks = instrument_ticks(fixture)
    family_for_strategy = family_resolver(fixture)
    source_hash = _replay_source_fingerprint(fixture)
    out = ReplayDecisionTimeline(fixture)

    if _surface_enabled(fixture, "TPC"):
        _replay_tpc(fixture, out)
    if _surface_enabled(fixture, "NQ_REGIME"):
        _replay_nq_regime(fixture, out)
    if _surface_enabled(fixture, "IARIC_v1"):
        _replay_iaric(fixture, out)
    _replay_idle_market_children(fixture, out)

    family_surface = _run_family_portfolio_surface(fixture, out)
    if family_surface.get("overlay"):
        out.strategy_state["OVERLAY"] = dict(family_surface["overlay"])
    for strategy_id in strategy_ids(fixture):
        out.strategy_state.setdefault(strategy_id, _idle_replay_strategy_state(fixture, strategy_id))
    timeline = _authoritative_family_timeline(fixture, out, family_surface)
    sink = await run_replay_oms_sink(
        fixture,
        timeline,
        strategy_state=out.strategy_state,
        family_mode=True,
    )
    _assert_family_surface_matches_sink(family_surface, sink.orders)
    state = sink.state
    _apply_replay_oms_outcomes_to_strategy_state(state)
    state["family_state"] = build_family_state(
        fixture,
        coordinator_class=_coordinator_class(fixture),
        orders=sink.orders,
        positions=sink.positions,
        strategy_state=sink.state.get("strategy_state", {}),
        strategy_risk=sink.state.get("strategy_risk", {}),
        portfolio_risk=sink.state.get("portfolio_risk", []),
        portfolio_rules=sink.state.get("portfolio_rules", []),
        overlay_state=family_surface.get("overlay", {}),
        surface_adapter=family_surface.get("adapter", ""),
        blocked_counts=family_surface.get("blocked_counts", {}),
        blocked_reasons=sink.state.get("blocked_reasons", {}),
        accepted_quantities=family_surface.get("accepted_quantities", {}),
    )

    return ParityTrace(
        producer="family_backtest_replay",
        source_fingerprint=source_hash,
        order_intents=normalize_order_intents(
            sink.submitted_orders,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        terminal_events=normalize_oms_events(
            sink.events,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        trade_ledger=normalize_trade_ledger(
            sink.trade_ledger,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        state_snapshot=normalize_state_snapshot(state),
    )


def _surface_enabled(fixture: Mapping[str, Any], strategy_id: str) -> bool:
    surface = str(fixture.get("surface", "")).upper()
    if surface == strategy_id.upper() or (surface == "IARIC" and strategy_id == "IARIC_v1"):
        return True
    return strategy_id in set(strategy_ids(fixture))


def _replay_source_fingerprint(fixture: Mapping[str, Any]) -> str:
    return runtime_source_fingerprint(fixture)


def _coordinator_class(fixture: Mapping[str, Any]) -> str:
    return coordinator_class_name(fixture)


def _run_blocking(fn):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()


def _run_coro_blocking(coro):
    return _run_blocking(lambda: asyncio.run(coro))
