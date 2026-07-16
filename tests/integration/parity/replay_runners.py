from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
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
    replay_downturn_r1b as _replay_downturn_r1b,
    replay_nq_regime as _replay_nq_regime,
    replay_nqdtc_r1b as _replay_nqdtc_r1b,
    replay_tpc as _replay_tpc,
    replay_vdub_r1b as _replay_vdub_r1b,
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
        self.current_equity = float(
            (fixture.get("account_state", {}) or {}).get("equity", 100_000.0)
        )
        self._reservations: dict[str, dict[str, Any]] = {}
        self._positions: dict[tuple[str, str], dict[str, Any]] = {}
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
        static = evaluate_static_portfolio_entry(
            self.config,
            strategy_id=strategy_id,
            direction=direction,
            current_equity=self.current_equity,
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
        explicit_risk_dollars = risk_context.get("risk_dollars")
        if explicit_risk_dollars not in (None, "") and action.qty > 0:
            risk_dollars = (
                float(explicit_risk_dollars) * approved_qty / action.qty
            )
        else:
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
            self._reservations[str(action.client_order_id)] = {
                "strategy_id": strategy_id,
                "symbol": str(action.symbol),
                "direction": direction,
                "risk_dollars": risk_dollars,
                "qty": approved_qty,
                "entry_price": planned_entry,
                "point_value": point_value(self.fixture, action.symbol),
            }
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

    def apply_broker_event(
        self,
        out: ReplayDecisionTimeline,
        event: Mapping[str, Any],
    ) -> None:
        """Apply one real order lifecycle result to the next decision snapshot."""

        order = out._match_order(event.get("order_match", {}))
        if order is None:
            return
        event_type = str(event.get("event", "fill")).lower()
        role = str(order.get("role", "")).upper()
        client_order_id = str(order.get("client_tag", ""))
        if role == "ENTRY":
            reservation = self._reservations.get(client_order_id)
            if reservation is None:
                return
            terminal_status = str(event.get("status", "")).lower()
            if event_type == "reject" or (
                event_type == "status"
                and terminal_status
                in {"cancelled", "canceled", "rejected", "expired", "inactive"}
            ):
                self._release_reservation(client_order_id)
                return
            if event_type == "fill":
                self._positions[
                    (str(order.get("strategy_id", "")), str(order.get("symbol", "")))
                ] = {
                    **reservation,
                    "reservation_id": client_order_id,
                    "qty": int(float(event.get("qty", order.get("qty", 0))) or 0),
                    "entry_price": float(
                        event.get(
                            "price",
                            order.get("limit_price") or order.get("stop_price") or 0.0,
                        )
                    ),
                }
            return

        if event_type != "fill":
            return
        position_key = (
            str(order.get("strategy_id", "")),
            str(order.get("symbol", "")),
        )
        position = self._positions.get(position_key)
        if position is None:
            return
        exit_qty = min(
            int(float(event.get("qty", order.get("qty", 0))) or 0),
            int(position.get("qty", 0)),
        )
        if exit_qty <= 0:
            return
        reservation_id = str(position["reservation_id"])
        reservation = self._reservations.get(reservation_id)
        if reservation is not None:
            original_qty = max(int(reservation.get("qty", 0)), 1)
            released = min(
                float(reservation.get("risk_dollars", 0.0)),
                float(reservation.get("risk_dollars", 0.0)) * exit_qty / original_qty,
            )
            direction = str(reservation["direction"])
            self.current_directional_risk_dollars[direction] = max(
                0.0,
                self.current_directional_risk_dollars[direction] - released,
            )
            reservation["risk_dollars"] = max(
                0.0,
                float(reservation["risk_dollars"]) - released,
            )
            reservation["qty"] = max(0, int(reservation["qty"]) - exit_qty)
            if reservation["qty"] <= 0:
                self._reservations.pop(reservation_id, None)

        exit_price = float(event.get("price", 0.0))
        entry_price = float(position.get("entry_price", 0.0))
        point_value_value = float(position.get("point_value", 0.0))
        direction_mult = 1.0 if position.get("direction") == "LONG" else -1.0
        pnl = (
            (exit_price - entry_price)
            * direction_mult
            * exit_qty
            * point_value_value
            - float(event.get("commission", 0.0) or 0.0)
        )
        self.current_equity += pnl
        position["qty"] = max(0, int(position["qty"]) - exit_qty)
        if position["qty"] <= 0:
            self._positions.pop(position_key, None)

    def _release_reservation(self, client_order_id: str) -> None:
        reservation = self._reservations.pop(client_order_id, None)
        if reservation is None:
            return
        direction = str(reservation["direction"])
        self.current_directional_risk_dollars[direction] = max(
            0.0,
            self.current_directional_risk_dollars[direction]
            - float(reservation.get("risk_dollars", 0.0)),
        )


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


def run_momentum_r1b_vdub_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    """Run the bounded NQDTC + Vdub + NQ_REGIME causal subincrement."""

    return _run_coro_blocking(_run_momentum_r1b_nqdtc_replay_trace(fixture))


def run_momentum_r1b_downturn_replay_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    """Run the bounded four-child Downturn entry/proposal subincrement."""

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
    expected_ids = ["NQDTC_v2.1"]
    if "VdubusNQ_v4" in strategy_ids(fixture):
        expected_ids.append("VdubusNQ_v4")
    if "DownturnDominator_v1" in strategy_ids(fixture):
        expected_ids.append("DownturnDominator_v1")
    expected_ids.append("NQ_REGIME")
    initial_ids = [str(event["strategy_id"]) for event in raw_timeline[: len(expected_ids)]]
    if initial_ids != expected_ids:
        raise AssertionError(
            f"R1B bounded timeline prefix must match {expected_ids}, got {initial_ids}"
        )
    if any(
        str(event["strategy_id"]) != "DownturnDominator_v1"
        for event in raw_timeline[len(expected_ids) :]
    ):
        raise AssertionError("R1B feedback closure only permits later Downturn raw events")

    downturn_state = None
    for event in raw_timeline:
        strategy_id = str(event["strategy_id"])
        timeline_id = str(event.get("timeline_id", ""))
        if strategy_id == "NQDTC_v2.1":
            _replay_nqdtc_r1b(fixture, out, portfolio_authorizer=portfolio)
        elif strategy_id == "VdubusNQ_v4":
            _replay_vdub_r1b(fixture, out, portfolio_authorizer=portfolio)
        elif strategy_id == "DownturnDominator_v1":
            downturn_state = _replay_downturn_r1b(
                fixture,
                out,
                portfolio_authorizer=portfolio,
                market_input=event["payload"],
                core_state=downturn_state,
                timeline_id=timeline_id,
            )
        else:
            _replay_nq_regime(
                fixture,
                out,
                causal_authorization=True,
                portfolio_authorizer=portfolio,
                market_rows=[event["payload"]],
                timeline_id=timeline_id,
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
        producer=(
            "momentum_r1b_downturn_causal_replay"
            if "DownturnDominator_v1" in strategy_ids(fixture)
            else "momentum_r1b_vdub_causal_replay"
            if "VdubusNQ_v4" in strategy_ids(fixture)
            else "momentum_r1b_nqdtc_causal_replay"
        ),
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
