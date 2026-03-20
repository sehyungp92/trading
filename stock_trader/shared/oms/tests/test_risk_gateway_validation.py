import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.oms.config.risk_config import RiskConfig, StrategyRiskConfig
from shared.oms.models.instrument import Instrument
from shared.oms.models.order import OMSOrder, OrderRole, OrderType, RiskContext
from shared.oms.models.risk_state import PortfolioRiskState, StrategyRiskState
from shared.oms.risk.calendar import EventCalendar
from shared.oms.risk.gateway import RiskGateway


TRADE_DAY = date(2026, 3, 10)


def _make_gateway(*, unit_risk_dollars: float = 100.0) -> RiskGateway:
    async def _get_strategy_risk(strategy_id: str) -> StrategyRiskState:
        return StrategyRiskState(strategy_id=strategy_id, trade_date=TRADE_DAY)

    async def _get_portfolio_risk() -> PortfolioRiskState:
        return PortfolioRiskState(trade_date=TRADE_DAY)

    config = RiskConfig(
        strategy_configs={
            "TestStrategy": StrategyRiskConfig(
                strategy_id="TestStrategy",
                unit_risk_dollars=unit_risk_dollars,
            )
        }
    )
    return RiskGateway(
        config=config,
        calendar=EventCalendar(),
        get_strategy_risk=_get_strategy_risk,
        get_portfolio_risk=_get_portfolio_risk,
    )


def _make_order(*, qty: int = 2, point_value: float = 2.0,
                planned_entry: float = 19010.0, stop_for_risk: float = 19000.0) -> OMSOrder:
    instrument = Instrument(
        symbol="MNQ",
        root="MNQ",
        venue="CME",
        tick_size=0.25,
        tick_value=0.5,
        multiplier=point_value,
    )
    return OMSOrder(
        strategy_id="TestStrategy",
        instrument=instrument,
        qty=qty,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        risk_context=RiskContext(
            stop_for_risk=stop_for_risk,
            planned_entry_price=planned_entry,
        ),
    )


def test_check_entry_accepts_valid_order_and_sets_risk_dollars():
    gateway = _make_gateway()
    order = _make_order()

    denial = asyncio.run(gateway.check_entry(order))

    assert denial is None
    assert order.risk_context.risk_dollars == 40.0


def test_check_entry_rejects_non_positive_qty():
    gateway = _make_gateway()
    order = _make_order(qty=0)

    denial = asyncio.run(gateway.check_entry(order))

    assert denial == "ENTRY qty must be positive: 0"


def test_check_entry_rejects_non_finite_risk_prices():
    gateway = _make_gateway()
    order = _make_order(planned_entry=float("nan"))

    denial = asyncio.run(gateway.check_entry(order))

    assert denial == "ENTRY risk prices must be finite"


def test_check_entry_rejects_zero_risk_distance():
    gateway = _make_gateway()
    order = _make_order(planned_entry=19000.0, stop_for_risk=19000.0)

    denial = asyncio.run(gateway.check_entry(order))

    assert denial == "ENTRY risk distance must be positive: entry=19000.0 stop=19000.0"


def test_check_entry_rejects_invalid_point_value():
    gateway = _make_gateway()
    order = _make_order(point_value=0.0)

    denial = asyncio.run(gateway.check_entry(order))

    assert denial == "Instrument point_value must be positive: 0.0"


def test_check_entry_rejects_non_positive_unit_risk_config():
    gateway = _make_gateway(unit_risk_dollars=0.0)
    order = _make_order()

    denial = asyncio.run(gateway.check_entry(order))

    assert denial == "Strategy unit_risk_dollars must be positive: 0.0"
