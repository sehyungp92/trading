from __future__ import annotations

from libs.oms.models.order import EntryPolicy, OMSOrder, OrderRole, OrderSide, OrderType, RiskContext

from .actions import (
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitExit,
    SubmitMarketExit,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)


def neutral_action_to_oms_order(
    action: (
        SubmitEntry
        | SubmitExit
        | SubmitAddOnEntry
        | SubmitProtectiveStop
        | ReplaceProtectiveStop
        | SubmitProfitTarget
        | SubmitPartialExit
        | SubmitMarketExit
    ),
    *,
    strategy_id: str,
    instrument,
    account_id: str = "",
) -> OMSOrder:
    order_type = _order_type_for(
        "STOP" if isinstance(action, (SubmitProtectiveStop, ReplaceProtectiveStop)) else getattr(action, "order_type", "MARKET")
    )
    role = _role_for(action)
    risk_context = _risk_context_for(action)
    entry_policy = _entry_policy_for(action)

    client_order_id = getattr(action, "client_order_id", "")
    return OMSOrder(
        client_order_id=client_order_id,
        strategy_id=strategy_id,
        account_id=account_id,
        instrument=instrument,
        side=OrderSide.BUY if action.side == "BUY" else OrderSide.SELL,
        qty=action.qty,
        order_type=order_type,
        limit_price=getattr(action, "limit_price", None) or getattr(action, "price", None),
        stop_price=getattr(action, "stop_price", None),
        tif=getattr(action, "tif", "DAY"),
        role=role,
        entry_policy=entry_policy,
        risk_context=risk_context,
        oca_group=getattr(action, "oca_group", ""),
    )


def _role_for(action) -> OrderRole:
    if isinstance(action, (SubmitEntry, SubmitAddOnEntry)):
        return OrderRole.ENTRY
    if isinstance(action, (SubmitProtectiveStop, ReplaceProtectiveStop)):
        return OrderRole.STOP
    if isinstance(action, SubmitProfitTarget):
        return OrderRole.TP
    return OrderRole.EXIT


def _entry_policy_for(action) -> EntryPolicy | None:
    if not isinstance(action, (SubmitEntry, SubmitAddOnEntry)):
        return None
    ttl_bars = action.metadata.get("ttl_bars")
    ttl_seconds = action.metadata.get("ttl_seconds")
    if ttl_bars is None and ttl_seconds is None:
        return None
    return EntryPolicy(
        ttl_bars=int(ttl_bars) if ttl_bars is not None else None,
        ttl_seconds=int(ttl_seconds) if ttl_seconds is not None else None,
    )


def _risk_context_for(action) -> RiskContext | None:
    if not isinstance(action, (SubmitEntry, SubmitAddOnEntry)):
        return None
    payload = dict(getattr(action, "risk_context", {}) or {})
    stop_for_risk = payload.get("stop_for_risk")
    planned_entry_price = payload.get("planned_entry_price")
    if stop_for_risk is None or planned_entry_price is None:
        return None
    return RiskContext(
        stop_for_risk=float(stop_for_risk),
        planned_entry_price=float(planned_entry_price),
        risk_budget_tag=str(payload.get("risk_budget_tag", "")),
        risk_dollars=float(payload.get("risk_dollars", 0.0) or 0.0),
        portfolio_size_mult=float(payload.get("portfolio_size_mult", 1.0) or 1.0),
    )


def _order_type_for(order_type: str) -> OrderType:
    mapping = {
        "LIMIT": OrderType.LIMIT,
        "MARKET": OrderType.MARKET,
        "STOP": OrderType.STOP,
        "STOP_LIMIT": OrderType.STOP_LIMIT,
    }
    return mapping[order_type]
