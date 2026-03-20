"""OMS order construction helpers."""

from __future__ import annotations

import uuid

from shared.oms.models.instrument import Instrument
from shared.oms.models.instrument_registry import InstrumentRegistry
from shared.oms.models.order import EntryPolicy, OMSOrder, OrderRole, OrderSide, OrderType, RiskContext

from .config import STRATEGY_ID, StrategySettings
from .models import CachedSymbol, SymbolContext


def build_stock_instrument(cached: CachedSymbol) -> Instrument:
    instrument = Instrument(
        symbol=cached.symbol,
        root=cached.symbol,
        venue=cached.exchange,
        primary_exchange=cached.primary_exchange,
        sec_type="STK",
        tick_size=cached.tick_size,
        tick_value=cached.tick_size,
        multiplier=1.0,
        point_value=cached.point_value,
        currency=cached.currency,
    )
    InstrumentRegistry.register(instrument)
    return instrument


def build_entry_order(
    ctx: SymbolContext,
    account_id: str,
    settings: StrategySettings,
) -> OMSOrder:
    instrument = build_stock_instrument(ctx.cached)
    return OMSOrder(
        client_order_id=f"{ctx.symbol}-entry-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=OrderSide.BUY,
        qty=ctx.qty,
        order_type=OrderType.STOP_LIMIT,
        limit_price=ctx.planned_limit,
        stop_price=ctx.planned_entry,
        role=OrderRole.ENTRY,
        entry_policy=EntryPolicy(ttl_seconds=settings.entry_ttl_s, max_reprices=settings.max_rearms),
        risk_context=RiskContext(
            stop_for_risk=ctx.final_stop or 0.0,
            planned_entry_price=ctx.planned_entry or 0.0,
            risk_budget_tag="ORB",
        ),
    )


def build_stop_order(ctx: SymbolContext, account_id: str, qty: int, stop_price: float) -> OMSOrder:
    instrument = build_stock_instrument(ctx.cached)
    return OMSOrder(
        client_order_id=f"{ctx.symbol}-stop-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=OrderSide.SELL,
        qty=qty,
        order_type=OrderType.STOP,
        stop_price=stop_price,
        role=OrderRole.STOP,
    )


def build_market_exit(ctx: SymbolContext, account_id: str, qty: int, role: OrderRole = OrderRole.EXIT) -> OMSOrder:
    instrument = build_stock_instrument(ctx.cached)
    return OMSOrder(
        client_order_id=f"{ctx.symbol}-exit-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=OrderSide.SELL,
        qty=qty,
        order_type=OrderType.MARKET,
        role=role,
    )
