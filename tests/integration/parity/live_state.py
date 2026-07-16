from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.integration.parity.family_state import build_family_state
from tests.integration.parity.live_family import family_surface_adapter_name
from tests.integration.parity.live_layer2 import compact_engine_state, compact_overlay_state
from tests.integration.parity.live_oms import (
    blocked_reasons_from_repo_events,
    plain_dataclass as _plain_dataclass,
    portfolio_rules_state,
)
from tests.integration.parity.portfolio_rules import portfolio_rules_config_from_fixture
from tests.integration.parity.source_inputs import strategy_ids


async def _state_from_repos(
    repos: list[Any],
    oms_services: list[Any],
    fixture: Mapping[str, Any],
    engines: Mapping[str, Any],
    coordinator: Any,
) -> dict[str, Any]:
    orders = []
    positions = []
    for repo in repos:
        for order in repo._orders.values():
            orders.append(
                {
                    "oms_order_id": order.oms_order_id,
                    "strategy_id": order.strategy_id,
                    "symbol": order.instrument.symbol if order.instrument else "",
                    "side": order.side.value,
                    "qty": order.qty,
                    "order_type": order.order_type.value,
                    "limit_price": order.limit_price,
                    "stop_price": order.stop_price,
                    "role": order.role.value,
                    "status": order.status.value,
                    "filled_qty": order.filled_qty,
                    "remaining_qty": order.remaining_qty,
                    "avg_fill_price": order.avg_fill_price,
                    "client_tag": order.client_order_id,
                    "reject_reason": order.reject_reason,
                }
            )
        for pos in repo._positions.values():
            positions.append(
                {
                    "strategy_id": pos.strategy_id,
                    "symbol": pos.instrument_symbol,
                    "net_qty": pos.net_qty,
                    "avg_price": pos.avg_price,
                    "realized_pnl": pos.realized_pnl,
                    "open_risk_dollars": pos.open_risk_dollars,
                    "open_risk_R": pos.open_risk_R,
                }
            )
    strategy_risk = {}
    portfolio_risk = []
    configured_strategy_ids = strategy_ids(fixture)
    coordinator_strategy_ids = list(getattr(coordinator, "_strategy_ids", []) or [])
    coordinator_services = list(getattr(coordinator, "_oms_services", []) or [])
    owner_services = (
        dict(zip(coordinator_strategy_ids, coordinator_services, strict=True))
        if len(coordinator_strategy_ids) == len(coordinator_services)
        else {}
    )
    for service in oms_services:
        owned_ids = [
            sid for sid, owner in owner_services.items() if owner is service
        ] or configured_strategy_ids
        for sid in owned_ids:
            get_strategy_risk = getattr(service, "get_strategy_risk", None)
            if get_strategy_risk is not None:
                await get_strategy_risk(sid)
        get_portfolio_risk = getattr(service, "get_portfolio_risk", None)
        if get_portfolio_risk is not None:
            await get_portfolio_risk()
        for sid, state in getattr(service, "_strategy_risk_states", {}).items():
            if owner_services and owner_services.get(sid) is not service:
                continue
            strategy_risk[sid] = _plain_dataclass(state)
        prs = getattr(service, "_portfolio_risk_state", None)
        if prs is not None:
            plain = _plain_dataclass(prs)
            if plain not in portfolio_risk:
                portfolio_risk.append(plain)
    if owner_services and portfolio_risk:
        rules = portfolio_rules_config_from_fixture(fixture)
        reference_urd = float(
            getattr(rules, "reference_unit_risk_dollars", 0.0) or 0.0
        )
        if reference_urd > 0:
            owner_portfolio = [
                getattr(service, "_portfolio_risk_state", None)
                for service in owner_services.values()
            ]
            owner_portfolio = [state for state in owner_portfolio if state is not None]
            realized_by_strategy = {}
            for sid, state in strategy_risk.items():
                owner_portfolio_state = getattr(
                    owner_services.get(sid),
                    "_portfolio_risk_state",
                    None,
                )
                owner_pnl = dict(
                    getattr(owner_portfolio_state, "strategy_daily_pnl", {}) or {}
                )
                realized_pnl = float(state.get("daily_realized_pnl", 0.0) or 0.0)
                if sid in owner_pnl or realized_pnl != 0.0:
                    realized_by_strategy[sid] = realized_pnl
            open_risk_dollars = sum(
                float(pos.open_risk_dollars or 0.0)
                for repo in repos[:1]
                for pos in repo._positions.values()
                if pos.net_qty != 0
            )
            pending_risk_R = await repos[0].get_pending_entry_risk_R_for_strategies(
                configured_strategy_ids,
                reference_urd,
            )
            daily_pnl = sum(realized_by_strategy.values())
            weekly_pnl = sum(
                float(state.weekly_realized_pnl or 0.0)
                for state in owner_portfolio
            )
            first = owner_portfolio[0]
            portfolio_risk = [
                {
                    "daily_realized_R": daily_pnl / reference_urd,
                    "daily_realized_pnl": daily_pnl,
                    "halt_reason": next(
                        (str(state.halt_reason) for state in owner_portfolio if state.halt_reason),
                        "",
                    ),
                    "halted": any(bool(state.halted) for state in owner_portfolio),
                    "open_risk_R": open_risk_dollars / reference_urd,
                    "open_risk_dollars": open_risk_dollars,
                    "pending_entry_risk_R": pending_risk_R,
                    "strategy_daily_pnl": realized_by_strategy,
                    "trade_date": first.trade_date,
                    "week_start_date": first.week_start_date,
                    "weekly_realized_R": weekly_pnl / reference_urd,
                    "weekly_realized_pnl": weekly_pnl,
                }
            ]
    strategy_state = {
        strategy_id: compact_engine_state(engine, strategy_id)
        for strategy_id, engine in sorted(engines.items())
        if strategy_id in set(strategy_ids(fixture))
    }
    overlay_state = compact_overlay_state(engines.get("OVERLAY"))
    coordinator_class = type(coordinator).__name__ if coordinator is not None else ""
    blocked_reasons = blocked_reasons_from_repo_events(repos, orders)
    portfolio_rules = []
    for rule_state in portfolio_rules_state(oms_services):
        if rule_state not in portfolio_rules:
            portfolio_rules.append(rule_state)
    return {
        "orders": orders,
        "positions": positions,
        "strategy_risk": strategy_risk,
        "portfolio_risk": portfolio_risk,
        "portfolio_rules": portfolio_rules,
        "blocked_reasons": blocked_reasons,
        "strategy_state": strategy_state,
        "family_state": build_family_state(
            fixture,
            coordinator_class=coordinator_class,
            orders=orders,
            positions=positions,
            strategy_risk=strategy_risk,
            portfolio_risk=portfolio_risk,
            portfolio_rules=portfolio_rules,
            strategy_state=strategy_state,
            overlay_state=overlay_state,
            surface_adapter=family_surface_adapter_name(fixture),
            blocked_reasons=blocked_reasons,
        ),
    }


state_from_repos = _state_from_repos
