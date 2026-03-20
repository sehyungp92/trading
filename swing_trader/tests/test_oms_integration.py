"""Comprehensive OMS integration tests.

Tests how the shared OMS files (state machine, risk calculator, risk gateway,
strategy coordinator, event bus, tick rules) interact with all 3 strategies
(ATRSS, AKC_HELIX, SWING_BREAKOUT_V3) and the infra layer.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Source imports
# ---------------------------------------------------------------------------
from shared.oms.engine.state_machine import (
    TRANSITIONS,
    is_done,
    is_terminal,
    transition,
)
from shared.oms.risk.calculator import RiskCalculator
from shared.oms.risk.gateway import RiskGateway
from shared.oms.risk.calendar import EventCalendar, EventWindow
from shared.oms.coordination.coordinator import StrategyCoordinator
from shared.oms.events.bus import EventBus
from shared.oms.models.order import (
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskContext,
    TERMINAL_STATUSES,
)
from shared.oms.models.instrument import Instrument
from shared.oms.models.risk_state import PortfolioRiskState, StrategyRiskState
from shared.oms.models.events import OMSEvent, OMSEventType
from shared.oms.config.risk_config import RiskConfig, StrategyRiskConfig
from shared.ibkr_core.risk_support.tick_rules import (
    round_to_tick,
    round_qty,
    validate_price,
)

# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------
ATRSS = "ATRSS"
HELIX = "AKC_HELIX"
BREAKOUT = "SWING_BREAKOUT_V3"

TODAY = date.today()

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _make_instrument(
    symbol: str = "QQQ",
    tick_size: float = 0.01,
    multiplier: float = 1.0,
) -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=tick_size,
        tick_value=tick_size,
        multiplier=multiplier,
    )


def _make_order(
    strategy_id: str = ATRSS,
    symbol: str = "QQQ",
    side: OrderSide = OrderSide.BUY,
    qty: int = 10,
    order_type: OrderType = OrderType.LIMIT,
    role: OrderRole = OrderRole.ENTRY,
    planned_entry: float = 450.0,
    stop_for_risk: float = 445.0,
    status: OrderStatus = OrderStatus.CREATED,
    multiplier: float = 1.0,
) -> OMSOrder:
    inst = _make_instrument(symbol=symbol, multiplier=multiplier)
    return OMSOrder(
        strategy_id=strategy_id,
        instrument=inst,
        side=side,
        qty=qty,
        order_type=order_type,
        role=role,
        risk_context=RiskContext(
            stop_for_risk=stop_for_risk,
            planned_entry_price=planned_entry,
        ),
        status=status,
    )


def _make_strategy_config(
    strategy_id: str,
    priority: int,
    unit_risk_dollars: float = 350.0,
    daily_stop_R: float = 2.0,
    max_working_orders: int = 4,
    max_heat_R: float = 0.0,
    allowed_order_types: dict | None = None,
) -> StrategyRiskConfig:
    return StrategyRiskConfig(
        strategy_id=strategy_id,
        priority=priority,
        unit_risk_dollars=unit_risk_dollars,
        daily_stop_R=daily_stop_R,
        max_working_orders=max_working_orders,
        max_heat_R=max_heat_R,
        allowed_order_types=allowed_order_types or {},
    )


def _default_risk_config() -> RiskConfig:
    """Baseline risk config with all 3 strategies registered."""
    return RiskConfig(
        heat_cap_R=1.25,
        portfolio_daily_stop_R=3.0,
        global_standdown=False,
        strategy_configs={
            ATRSS: _make_strategy_config(ATRSS, priority=0),
            HELIX: _make_strategy_config(HELIX, priority=1),
            BREAKOUT: _make_strategy_config(BREAKOUT, priority=2),
        },
    )


def _clean_strategy_risk(strategy_id: str = ATRSS) -> StrategyRiskState:
    return StrategyRiskState(strategy_id=strategy_id, trade_date=TODAY)


def _clean_portfolio_risk() -> PortfolioRiskState:
    return PortfolioRiskState(trade_date=TODAY)


# ============================================================================
# 1. State Machine Tests
# ============================================================================
class TestStateMachine:
    """Tests for shared.oms.engine.state_machine."""

    # -- Valid transitions ------------------------------------------------

    def test_created_to_risk_approved(self):
        order = _make_order()
        assert order.status == OrderStatus.CREATED
        assert transition(order, OrderStatus.RISK_APPROVED) is True
        assert order.status == OrderStatus.RISK_APPROVED

    def test_created_to_rejected(self):
        order = _make_order()
        assert transition(order, OrderStatus.REJECTED) is True
        assert order.status == OrderStatus.REJECTED

    def test_risk_approved_to_routed(self):
        order = _make_order(status=OrderStatus.RISK_APPROVED)
        assert transition(order, OrderStatus.ROUTED) is True
        assert order.status == OrderStatus.ROUTED

    def test_risk_approved_to_rejected(self):
        order = _make_order(status=OrderStatus.RISK_APPROVED)
        assert transition(order, OrderStatus.REJECTED) is True

    def test_routed_to_acked(self):
        order = _make_order(status=OrderStatus.ROUTED)
        assert transition(order, OrderStatus.ACKED) is True
        assert order.status == OrderStatus.ACKED

    def test_routed_to_rejected(self):
        order = _make_order(status=OrderStatus.ROUTED)
        assert transition(order, OrderStatus.REJECTED) is True

    def test_routed_to_cancelled_race(self):
        """H3: Broker can cancel before ACK (race condition)."""
        order = _make_order(status=OrderStatus.ROUTED)
        assert transition(order, OrderStatus.CANCELLED) is True
        assert order.status == OrderStatus.CANCELLED

    def test_acked_to_working(self):
        order = _make_order(status=OrderStatus.ACKED)
        assert transition(order, OrderStatus.WORKING) is True

    def test_acked_to_filled(self):
        order = _make_order(status=OrderStatus.ACKED)
        assert transition(order, OrderStatus.FILLED) is True

    def test_acked_to_cancelled(self):
        order = _make_order(status=OrderStatus.ACKED)
        assert transition(order, OrderStatus.CANCELLED) is True

    def test_acked_to_rejected(self):
        order = _make_order(status=OrderStatus.ACKED)
        assert transition(order, OrderStatus.REJECTED) is True

    def test_working_to_partially_filled(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.PARTIALLY_FILLED) is True

    def test_working_to_filled(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.FILLED) is True

    def test_working_to_cancel_requested(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.CANCEL_REQUESTED) is True

    def test_working_to_replace_requested(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.REPLACE_REQUESTED) is True

    def test_working_to_cancelled(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.CANCELLED) is True

    def test_working_to_expired(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.EXPIRED) is True

    def test_working_to_rejected(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.REJECTED) is True

    def test_partially_filled_to_partially_filled(self):
        order = _make_order(status=OrderStatus.PARTIALLY_FILLED)
        assert transition(order, OrderStatus.PARTIALLY_FILLED) is True

    def test_partially_filled_to_filled(self):
        order = _make_order(status=OrderStatus.PARTIALLY_FILLED)
        assert transition(order, OrderStatus.FILLED) is True

    def test_partially_filled_to_cancel_requested(self):
        order = _make_order(status=OrderStatus.PARTIALLY_FILLED)
        assert transition(order, OrderStatus.CANCEL_REQUESTED) is True

    def test_partially_filled_to_cancelled(self):
        order = _make_order(status=OrderStatus.PARTIALLY_FILLED)
        assert transition(order, OrderStatus.CANCELLED) is True

    def test_cancel_requested_to_cancelled(self):
        order = _make_order(status=OrderStatus.CANCEL_REQUESTED)
        assert transition(order, OrderStatus.CANCELLED) is True

    def test_cancel_requested_to_filled(self):
        """Fill can arrive after cancel request."""
        order = _make_order(status=OrderStatus.CANCEL_REQUESTED)
        assert transition(order, OrderStatus.FILLED) is True

    def test_cancel_requested_to_partially_filled(self):
        order = _make_order(status=OrderStatus.CANCEL_REQUESTED)
        assert transition(order, OrderStatus.PARTIALLY_FILLED) is True

    def test_replace_requested_to_replaced(self):
        order = _make_order(status=OrderStatus.REPLACE_REQUESTED)
        assert transition(order, OrderStatus.REPLACED) is True

    def test_replace_requested_to_cancelled(self):
        order = _make_order(status=OrderStatus.REPLACE_REQUESTED)
        assert transition(order, OrderStatus.CANCELLED) is True

    def test_replace_requested_to_filled(self):
        order = _make_order(status=OrderStatus.REPLACE_REQUESTED)
        assert transition(order, OrderStatus.FILLED) is True

    def test_replace_requested_to_rejected(self):
        order = _make_order(status=OrderStatus.REPLACE_REQUESTED)
        assert transition(order, OrderStatus.REJECTED) is True

    def test_terminal_to_done(self):
        """All terminal statuses can transition to DONE."""
        for terminal in (
            OrderStatus.CANCELLED,
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ):
            order = _make_order(status=terminal)
            assert transition(order, OrderStatus.DONE) is True
            assert order.status == OrderStatus.DONE

    def test_replaced_to_done(self):
        order = _make_order(status=OrderStatus.REPLACED)
        assert transition(order, OrderStatus.DONE) is True

    # -- Invalid transitions ----------------------------------------------

    def test_created_cannot_jump_to_working(self):
        order = _make_order()
        assert transition(order, OrderStatus.WORKING) is False
        assert order.status == OrderStatus.CREATED  # unchanged

    def test_created_cannot_jump_to_filled(self):
        order = _make_order()
        assert transition(order, OrderStatus.FILLED) is False

    def test_working_cannot_go_back_to_created(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.CREATED) is False

    def test_filled_cannot_go_to_working(self):
        order = _make_order(status=OrderStatus.FILLED)
        assert transition(order, OrderStatus.WORKING) is False

    def test_done_is_absorbing(self):
        """DONE has no outgoing transitions."""
        order = _make_order(status=OrderStatus.DONE)
        for target in OrderStatus:
            if target != OrderStatus.DONE:
                assert transition(order, target) is False

    def test_cancelled_cannot_go_to_filled(self):
        order = _make_order(status=OrderStatus.CANCELLED)
        assert transition(order, OrderStatus.FILLED) is False

    def test_rejected_cannot_go_to_working(self):
        order = _make_order(status=OrderStatus.REJECTED)
        assert transition(order, OrderStatus.WORKING) is False

    # -- Terminal / done detection ----------------------------------------

    def test_is_terminal_for_cancelled(self):
        order = _make_order(status=OrderStatus.CANCELLED)
        assert is_terminal(order) is True

    def test_is_terminal_for_filled(self):
        order = _make_order(status=OrderStatus.FILLED)
        assert is_terminal(order) is True

    def test_is_terminal_for_rejected(self):
        order = _make_order(status=OrderStatus.REJECTED)
        assert is_terminal(order) is True

    def test_is_terminal_for_expired(self):
        order = _make_order(status=OrderStatus.EXPIRED)
        assert is_terminal(order) is True

    def test_is_terminal_false_for_working(self):
        order = _make_order(status=OrderStatus.WORKING)
        assert is_terminal(order) is False

    def test_is_terminal_false_for_done(self):
        order = _make_order(status=OrderStatus.DONE)
        assert is_terminal(order) is False

    def test_is_done_true(self):
        order = _make_order(status=OrderStatus.DONE)
        assert is_done(order) is True

    def test_is_done_false_for_filled(self):
        order = _make_order(status=OrderStatus.FILLED)
        assert is_done(order) is False

    # -- Full lifecycle paths ---------------------------------------------

    def test_happy_path_lifecycle(self):
        """CREATED -> RISK_APPROVED -> ROUTED -> ACKED -> WORKING -> FILLED -> DONE"""
        order = _make_order()
        steps = [
            OrderStatus.RISK_APPROVED,
            OrderStatus.ROUTED,
            OrderStatus.ACKED,
            OrderStatus.WORKING,
            OrderStatus.FILLED,
            OrderStatus.DONE,
        ]
        for step in steps:
            assert transition(order, step) is True, f"Failed at {step}"
        assert is_done(order) is True

    def test_cancel_flow(self):
        """CREATED -> ... -> WORKING -> CANCEL_REQUESTED -> CANCELLED -> DONE"""
        order = _make_order()
        for step in (
            OrderStatus.RISK_APPROVED,
            OrderStatus.ROUTED,
            OrderStatus.ACKED,
            OrderStatus.WORKING,
            OrderStatus.CANCEL_REQUESTED,
            OrderStatus.CANCELLED,
            OrderStatus.DONE,
        ):
            assert transition(order, step) is True
        assert is_done(order)

    def test_replace_flow(self):
        """CREATED -> ... -> WORKING -> REPLACE_REQUESTED -> REPLACED -> DONE"""
        order = _make_order()
        for step in (
            OrderStatus.RISK_APPROVED,
            OrderStatus.ROUTED,
            OrderStatus.ACKED,
            OrderStatus.WORKING,
            OrderStatus.REPLACE_REQUESTED,
            OrderStatus.REPLACED,
            OrderStatus.DONE,
        ):
            assert transition(order, step) is True
        assert is_done(order)

    def test_partial_fill_then_complete(self):
        """WORKING -> PARTIALLY_FILLED -> PARTIALLY_FILLED -> FILLED -> DONE"""
        order = _make_order(status=OrderStatus.WORKING)
        assert transition(order, OrderStatus.PARTIALLY_FILLED) is True
        assert transition(order, OrderStatus.PARTIALLY_FILLED) is True
        assert transition(order, OrderStatus.FILLED) is True
        assert transition(order, OrderStatus.DONE) is True

    def test_cancel_requested_but_filled(self):
        """Cancel race: CANCEL_REQUESTED -> FILLED -> DONE"""
        order = _make_order(status=OrderStatus.CANCEL_REQUESTED)
        assert transition(order, OrderStatus.FILLED) is True
        assert transition(order, OrderStatus.DONE) is True

    def test_transitions_dict_covers_all_non_done_statuses(self):
        """Every non-DONE status should have an entry in TRANSITIONS."""
        for status in OrderStatus:
            if status != OrderStatus.DONE:
                assert status in TRANSITIONS, f"{status} missing from TRANSITIONS"


# ============================================================================
# 2. Risk Calculator Tests
# ============================================================================
class TestRiskCalculator:
    """Tests for shared.oms.risk.calculator.RiskCalculator."""

    def test_compute_order_risk_dollars_basic(self):
        # |450 - 445| * 1.0 * 10 = 50.0
        result = RiskCalculator.compute_order_risk_dollars(450.0, 445.0, 10, 1.0)
        assert result == pytest.approx(50.0)

    def test_compute_order_risk_dollars_futures(self):
        # MNQ: |20000 - 19950| * 2.0 * 3 = 300.0
        result = RiskCalculator.compute_order_risk_dollars(20000.0, 19950.0, 3, 2.0)
        assert result == pytest.approx(300.0)

    def test_compute_order_risk_dollars_short(self):
        # Short: |440 - 445| * 1.0 * 5 = 25.0 (abs handles direction)
        result = RiskCalculator.compute_order_risk_dollars(440.0, 445.0, 5, 1.0)
        assert result == pytest.approx(25.0)

    def test_compute_order_risk_dollars_zero_qty(self):
        result = RiskCalculator.compute_order_risk_dollars(450.0, 445.0, 0, 1.0)
        assert result == pytest.approx(0.0)

    def test_compute_order_risk_dollars_high_multiplier(self):
        # ES mini: |5000 - 4990| * 50.0 * 2 = 1000.0
        result = RiskCalculator.compute_order_risk_dollars(5000.0, 4990.0, 2, 50.0)
        assert result == pytest.approx(1000.0)

    def test_compute_risk_R_normal(self):
        # 350 / 350 = 1.0R
        result = RiskCalculator.compute_risk_R(350.0, 350.0)
        assert result == pytest.approx(1.0)

    def test_compute_risk_R_half(self):
        result = RiskCalculator.compute_risk_R(175.0, 350.0)
        assert result == pytest.approx(0.5)

    def test_compute_risk_R_zero_unit_risk_returns_inf(self):
        result = RiskCalculator.compute_risk_R(100.0, 0.0)
        assert result == float("inf")

    def test_compute_risk_R_negative_unit_risk_returns_inf(self):
        result = RiskCalculator.compute_risk_R(100.0, -10.0)
        assert result == float("inf")

    def test_compute_unit_risk_dollars_default(self):
        # 100_000 * 0.0035 * 1.0 = 350.0
        result = RiskCalculator.compute_unit_risk_dollars(100_000.0, 0.0035)
        assert result == pytest.approx(350.0)

    def test_compute_unit_risk_dollars_with_vol_factor(self):
        # 100_000 * 0.0035 * 0.8 = 280.0
        result = RiskCalculator.compute_unit_risk_dollars(100_000.0, 0.0035, 0.8)
        assert result == pytest.approx(280.0)

    def test_compute_unit_risk_dollars_zero_nav(self):
        result = RiskCalculator.compute_unit_risk_dollars(0.0, 0.0035)
        assert result == pytest.approx(0.0)


# ============================================================================
# 3. Risk Gateway Tests (async)
# ============================================================================
class TestRiskGateway:
    """Tests for shared.oms.risk.gateway.RiskGateway."""

    @pytest.mark.asyncio
    async def test_exit_order_always_passes(self):
        """Exit / stop / TP orders bypass all risk checks."""
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        for role in (OrderRole.EXIT, OrderRole.STOP, OrderRole.TP):
            order = _make_order(role=role)
            reason = await gw.check_entry(order)
            assert reason is None, f"Exit role {role} should pass but got: {reason}"

    @pytest.mark.asyncio
    async def test_global_standdown_blocks_entry(self):
        cfg = _default_risk_config()
        cfg.global_standdown = True
        gw = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        order = _make_order(strategy_id=ATRSS)
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "stand-down" in reason.lower() or "standdown" in reason.lower()

    @pytest.mark.asyncio
    async def test_event_blackout_blocks_entry(self):
        now = datetime.now(timezone.utc)
        window = EventWindow(
            name="CPI",
            start_utc=now - timedelta(minutes=5),
            end_utc=now + timedelta(minutes=30),
        )
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(windows=[window]),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        order = _make_order(strategy_id=HELIX)
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "blackout" in reason.lower()

    @pytest.mark.asyncio
    async def test_strategy_daily_halt_blocks_entry(self):
        halted_risk = _clean_strategy_risk(ATRSS)
        halted_risk.halted = True
        halted_risk.halt_reason = "Daily stop hit"
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=halted_risk),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        order = _make_order(strategy_id=ATRSS)
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "halted" in reason.lower() or "halt" in reason.lower()

    @pytest.mark.asyncio
    async def test_strategy_daily_stop_R_blocks_entry(self):
        """Strategy that has lost >= daily_stop_R should be blocked."""
        strat_risk = _clean_strategy_risk(BREAKOUT)
        strat_risk.daily_realized_R = -2.5  # past the -2.0 default threshold
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=strat_risk),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        order = _make_order(strategy_id=BREAKOUT)
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "daily stop" in reason.lower()

    @pytest.mark.asyncio
    async def test_portfolio_daily_halt_blocks_entry(self):
        port_risk = _clean_portfolio_risk()
        port_risk.halted = True
        port_risk.halt_reason = "Portfolio daily stop hit"
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
        )
        order = _make_order(strategy_id=ATRSS)
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "portfolio" in reason.lower()

    @pytest.mark.asyncio
    async def test_portfolio_daily_stop_R_blocks_entry(self):
        port_risk = _clean_portfolio_risk()
        port_risk.daily_realized_R = -3.5  # past -3.0 default
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
        )
        order = _make_order(strategy_id=HELIX)
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "portfolio daily stop" in reason.lower()

    @pytest.mark.asyncio
    async def test_max_working_orders_blocks_entry(self):
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
            get_working_order_count=AsyncMock(return_value=4),  # at max
        )
        order = _make_order(strategy_id=ATRSS)
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "working orders" in reason.lower()

    @pytest.mark.asyncio
    async def test_max_working_orders_allows_below_limit(self):
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
            get_working_order_count=AsyncMock(return_value=2),
        )
        # Small risk to stay within heat cap: planned_entry=450, stop=449.5 => 0.5*1*10=5 dollars
        order = _make_order(
            strategy_id=ATRSS, planned_entry=450.0, stop_for_risk=449.5, qty=1,
        )
        reason = await gw.check_entry(order)
        assert reason is None

    @pytest.mark.asyncio
    async def test_heat_cap_blocks_entry(self):
        """Order risk pushes total above heat_cap_R -> denied."""
        port_risk = _clean_portfolio_risk()
        port_risk.open_risk_R = 1.0
        port_risk.pending_entry_risk_R = 0.0

        gw = RiskGateway(
            config=_default_risk_config(),  # heat_cap_R=1.25
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )
        # big order: |450 - 445| * 1 * 10 = 50$; 50/350 = 0.143R -> 1.0+0.143 = 1.143 < 1.25 ok
        # so make it bigger: |450 - 440| * 1 * 10 = 100$; 100/350 = 0.286R -> 1.0+0.286 = 1.286 > 1.25
        order = _make_order(
            strategy_id=ATRSS, planned_entry=450.0, stop_for_risk=440.0, qty=10,
        )
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "heat cap" in reason.lower()

    @pytest.mark.asyncio
    async def test_per_strategy_heat_ceiling(self):
        """Per-strategy max_heat_R blocks an entry even if portfolio heat is fine."""
        cfg = _default_risk_config()
        cfg.strategy_configs[HELIX].max_heat_R = 0.5

        strat_risk = _clean_strategy_risk(HELIX)
        strat_risk.open_risk_R = 0.4  # already 0.4R; new order adds ~0.143R -> 0.54 > 0.5

        gw = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=strat_risk),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
            get_working_order_count=AsyncMock(return_value=0),
        )
        order = _make_order(
            strategy_id=HELIX, planned_entry=450.0, stop_for_risk=445.0, qty=10,
        )
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "heat ceiling" in reason.lower()

    @pytest.mark.asyncio
    async def test_priority_reservation_blocks_low_priority(self):
        """When heat is tight, low-priority strategies are denied to reserve for higher."""
        port_risk = _clean_portfolio_risk()
        # Remaining = 1.25 - 1.1 = 0.15R; new order ~0.143R; 2*0.143 = 0.286 > 0.15
        port_risk.open_risk_R = 1.1
        port_risk.pending_entry_risk_R = 0.0

        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk(BREAKOUT)),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )
        # BREAKOUT is priority=2, and ATRSS (priority=0) exists -> reservation kicks in
        order = _make_order(
            strategy_id=BREAKOUT, planned_entry=450.0, stop_for_risk=445.0, qty=10,
        )
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "reserved" in reason.lower()

    @pytest.mark.asyncio
    async def test_priority_reservation_allows_top_priority(self):
        """ATRSS (priority=0) has no higher-priority strategy -> reservation skipped."""
        port_risk = _clean_portfolio_risk()
        port_risk.open_risk_R = 1.1
        port_risk.pending_entry_risk_R = 0.0

        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk(ATRSS)),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )
        order = _make_order(
            strategy_id=ATRSS, planned_entry=450.0, stop_for_risk=445.0, qty=10,
        )
        reason = await gw.check_entry(order)
        # ATRSS is highest priority -> should not be blocked by reservation
        assert reason is None

    @pytest.mark.asyncio
    async def test_order_type_not_allowed(self):
        cfg = _default_risk_config()
        cfg.strategy_configs[ATRSS].allowed_order_types = {
            "ENTRY": ["STOP_LIMIT"],  # only STOP_LIMIT allowed for entries
        }

        gw = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
            get_working_order_count=AsyncMock(return_value=0),
        )
        order = _make_order(
            strategy_id=ATRSS,
            order_type=OrderType.MARKET,  # MARKET not in allowed
            planned_entry=450.0,
            stop_for_risk=449.5,
            qty=1,
        )
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "order type" in reason.lower()

    @pytest.mark.asyncio
    async def test_approved_entry_all_checks_pass(self):
        """Clean state: entry should be approved with no denial."""
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
            get_working_order_count=AsyncMock(return_value=0),
        )
        order = _make_order(
            strategy_id=ATRSS, planned_entry=450.0, stop_for_risk=449.5, qty=1,
        )
        reason = await gw.check_entry(order)
        assert reason is None

    @pytest.mark.asyncio
    async def test_entry_missing_risk_context_denied(self):
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        order = _make_order(strategy_id=ATRSS)
        order.risk_context = None
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "risk_context" in reason.lower()

    @pytest.mark.asyncio
    async def test_unknown_strategy_denied(self):
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        order = _make_order(strategy_id="UNKNOWN_STRAT")
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "no risk config" in reason.lower()


# ============================================================================
# 4. Strategy Coordinator Tests
# ============================================================================
class TestStrategyCoordinator:
    """Tests for shared.oms.coordination.coordinator.StrategyCoordinator."""

    def test_rule1_atrss_entry_emits_tighten_to_helix(self):
        """Rule 1: ATRSS entry fill on symbol X -> TIGHTEN_STOP_BE to Helix on X."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        # Helix must have an open position on QQQ for rule to fire
        coord.on_position_update(HELIX, "QQQ", qty=5, direction="LONG", entry_price=448.0)

        # Subscribe to Helix events BEFORE the fill
        q = bus.subscribe(HELIX)

        # ATRSS entry fill on QQQ
        coord.on_fill(ATRSS, "QQQ", side="BUY", role="ENTRY", price=450.0)

        assert not q.empty()
        event: OMSEvent = q.get_nowait()
        assert event.event_type == OMSEventType.COORDINATION
        assert event.strategy_id == HELIX
        assert event.payload["coordination_type"] == "TIGHTEN_STOP_BE"
        assert event.payload["symbol"] == "QQQ"

    def test_rule1_only_fires_when_helix_has_position(self):
        """Rule 1 should NOT fire if Helix has no open position on that symbol."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)
        q = bus.subscribe(HELIX)

        # No Helix position registered
        coord.on_fill(ATRSS, "QQQ", side="BUY", role="ENTRY", price=450.0)

        assert q.empty(), "No coordination event should fire without Helix position"

    def test_rule1_does_not_fire_for_non_atrss(self):
        """Rule 1 only applies to ATRSS entry fills."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)
        coord.on_position_update(HELIX, "QQQ", qty=5, direction="LONG")
        q = bus.subscribe(HELIX)

        coord.on_fill(BREAKOUT, "QQQ", side="BUY", role="ENTRY", price=450.0)
        assert q.empty()

    def test_rule1_does_not_fire_for_exit_fill(self):
        """Rule 1 only applies to ENTRY fills, not EXIT."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)
        coord.on_position_update(HELIX, "QQQ", qty=5, direction="LONG")
        q = bus.subscribe(HELIX)

        coord.on_fill(ATRSS, "QQQ", side="SELL", role="EXIT", price=455.0)
        assert q.empty()

    def test_rule1_different_symbol_no_fire(self):
        """Helix has position on SPY, ATRSS entry on QQQ -> no coordination."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)
        coord.on_position_update(HELIX, "SPY", qty=5, direction="LONG")
        q = bus.subscribe(HELIX)

        coord.on_fill(ATRSS, "QQQ", side="BUY", role="ENTRY", price=450.0)
        assert q.empty()

    def test_rule2_has_atrss_position_true(self):
        """Rule 2: has_atrss_position returns True when ATRSS has active position."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        coord.on_position_update(ATRSS, "QQQ", qty=10, direction="LONG", entry_price=450.0)

        assert coord.has_atrss_position("QQQ") is True
        assert coord.has_atrss_position("QQQ", direction="LONG") is True

    def test_rule2_direction_mismatch(self):
        """Rule 2: Direction must match for size boost."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        coord.on_position_update(ATRSS, "QQQ", qty=10, direction="LONG")

        assert coord.has_atrss_position("QQQ", direction="SHORT") is False

    def test_rule2_no_atrss_position(self):
        bus = EventBus()
        coord = StrategyCoordinator(bus)
        assert coord.has_atrss_position("QQQ") is False

    def test_rule2_atrss_position_removed_on_qty_zero(self):
        """When ATRSS exits (qty=0), position is removed."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        coord.on_position_update(ATRSS, "QQQ", qty=10, direction="LONG")
        assert coord.has_atrss_position("QQQ") is True

        coord.on_position_update(ATRSS, "QQQ", qty=0, direction="LONG")
        assert coord.has_atrss_position("QQQ") is False

    def test_on_position_update_tracks_multiple_strategies(self):
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        coord.on_position_update(ATRSS, "QQQ", qty=10, direction="LONG")
        coord.on_position_update(HELIX, "QQQ", qty=5, direction="LONG")
        coord.on_position_update(BREAKOUT, "SPY", qty=20, direction="SHORT")

        assert coord.get_position(ATRSS, "QQQ") is not None
        assert coord.get_position(HELIX, "QQQ") is not None
        assert coord.get_position(BREAKOUT, "SPY") is not None
        assert coord.get_position(BREAKOUT, "QQQ") is None

    def test_on_position_update_remove(self):
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        coord.on_position_update(HELIX, "QQQ", qty=5, direction="LONG")
        assert coord.get_position(HELIX, "QQQ") is not None

        coord.on_position_update(HELIX, "QQQ", qty=0, direction="LONG")
        assert coord.get_position(HELIX, "QQQ") is None

    def test_get_all_positions(self):
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        coord.on_position_update(ATRSS, "QQQ", qty=10, direction="LONG")
        coord.on_position_update(HELIX, "SPY", qty=5, direction="SHORT")

        positions = coord.get_all_positions()
        assert len(positions) == 2
        assert (ATRSS, "QQQ") in positions
        assert (HELIX, "SPY") in positions


# ============================================================================
# 5. Event Bus Tests (async)
# ============================================================================
class TestEventBus:
    """Tests for shared.oms.events.bus.EventBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self):
        bus = EventBus()
        q = bus.subscribe(ATRSS)

        order = _make_order(strategy_id=ATRSS, status=OrderStatus.CREATED)
        bus.emit_order_event(order)

        assert not q.empty()
        event = q.get_nowait()
        assert event.event_type == OMSEventType.ORDER_CREATED
        assert event.strategy_id == ATRSS

    @pytest.mark.asyncio
    async def test_subscribe_all_gets_all_events(self):
        bus = EventBus()
        global_q = bus.subscribe_all()

        order_atrss = _make_order(strategy_id=ATRSS, status=OrderStatus.CREATED)
        order_helix = _make_order(strategy_id=HELIX, status=OrderStatus.FILLED)

        bus.emit_order_event(order_atrss)
        bus.emit_order_event(order_helix)

        events = []
        while not global_q.empty():
            events.append(global_q.get_nowait())

        assert len(events) == 2
        strategy_ids = {e.strategy_id for e in events}
        assert ATRSS in strategy_ids
        assert HELIX in strategy_ids

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        q = bus.subscribe(ATRSS)
        bus.unsubscribe(ATRSS, q)

        order = _make_order(strategy_id=ATRSS, status=OrderStatus.CREATED)
        bus.emit_order_event(order)

        assert q.empty(), "After unsubscribe, queue should not receive events"

    @pytest.mark.asyncio
    async def test_emit_order_event_maps_status_to_event_type(self):
        """Verify status -> OMSEventType mapping for key statuses."""
        bus = EventBus()
        q = bus.subscribe(ATRSS)

        expected_mappings = {
            OrderStatus.CREATED: OMSEventType.ORDER_CREATED,
            OrderStatus.RISK_APPROVED: OMSEventType.ORDER_RISK_APPROVED,
            OrderStatus.ROUTED: OMSEventType.ORDER_ROUTED,
            OrderStatus.ACKED: OMSEventType.ORDER_ACKED,
            OrderStatus.WORKING: OMSEventType.ORDER_WORKING,
            OrderStatus.PARTIALLY_FILLED: OMSEventType.ORDER_PARTIALLY_FILLED,
            OrderStatus.FILLED: OMSEventType.ORDER_FILLED,
            OrderStatus.CANCELLED: OMSEventType.ORDER_CANCELLED,
            OrderStatus.REJECTED: OMSEventType.ORDER_REJECTED,
            OrderStatus.EXPIRED: OMSEventType.ORDER_EXPIRED,
        }

        for status, event_type in expected_mappings.items():
            order = _make_order(strategy_id=ATRSS, status=status)
            bus.emit_order_event(order)
            event = q.get_nowait()
            assert event.event_type == event_type, (
                f"Status {status} should map to {event_type}"
            )

    @pytest.mark.asyncio
    async def test_emit_order_event_no_event_for_intermediate_states(self):
        """CANCEL_REQUESTED, REPLACE_REQUESTED, DONE produce no event."""
        bus = EventBus()
        q = bus.subscribe(ATRSS)

        for status in (
            OrderStatus.CANCEL_REQUESTED,
            OrderStatus.REPLACE_REQUESTED,
            OrderStatus.REPLACED,
            OrderStatus.DONE,
        ):
            order = _make_order(strategy_id=ATRSS, status=status)
            bus.emit_order_event(order)

        assert q.empty(), "Intermediate statuses should not produce events"

    @pytest.mark.asyncio
    async def test_emit_fill_event(self):
        bus = EventBus()
        q = bus.subscribe(HELIX)

        fill_data = {"price": 451.0, "qty": 5, "commission": 1.0}
        bus.emit_fill_event(HELIX, "order-123", fill_data)

        event = q.get_nowait()
        assert event.event_type == OMSEventType.FILL
        assert event.strategy_id == HELIX
        assert event.oms_order_id == "order-123"
        assert event.payload == fill_data

    @pytest.mark.asyncio
    async def test_emit_risk_denial(self):
        bus = EventBus()
        q = bus.subscribe(BREAKOUT)

        bus.emit_risk_denial(BREAKOUT, "order-456", "Heat cap breach")

        event = q.get_nowait()
        assert event.event_type == OMSEventType.RISK_DENIAL
        assert event.payload["reason"] == "Heat cap breach"

    @pytest.mark.asyncio
    async def test_emit_risk_halt(self):
        bus = EventBus()
        q = bus.subscribe(ATRSS)

        bus.emit_risk_halt(ATRSS, "Daily stop -2R hit")

        event = q.get_nowait()
        assert event.event_type == OMSEventType.RISK_HALT
        assert event.payload["reason"] == "Daily stop -2R hit"

    @pytest.mark.asyncio
    async def test_emit_coordination_event(self):
        bus = EventBus()
        q = bus.subscribe(HELIX)

        bus.emit_coordination_event(
            target_strategy=HELIX,
            event_type="TIGHTEN_STOP_BE",
            symbol="QQQ",
        )

        event = q.get_nowait()
        assert event.event_type == OMSEventType.COORDINATION
        assert event.strategy_id == HELIX
        assert event.payload["coordination_type"] == "TIGHTEN_STOP_BE"
        assert event.payload["symbol"] == "QQQ"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_strategy(self):
        bus = EventBus()
        q1 = bus.subscribe(ATRSS)
        q2 = bus.subscribe(ATRSS)

        order = _make_order(strategy_id=ATRSS, status=OrderStatus.CREATED)
        bus.emit_order_event(order)

        assert not q1.empty()
        assert not q2.empty()

    @pytest.mark.asyncio
    async def test_subscriber_isolation(self):
        """Events for ATRSS should not arrive at HELIX subscriber."""
        bus = EventBus()
        q_atrss = bus.subscribe(ATRSS)
        q_helix = bus.subscribe(HELIX)

        order = _make_order(strategy_id=ATRSS, status=OrderStatus.CREATED)
        bus.emit_order_event(order)

        assert not q_atrss.empty()
        assert q_helix.empty()


# ============================================================================
# 6. Tick Rules Tests
# ============================================================================
class TestTickRules:
    """Tests for shared.ibkr_core.risk_support.tick_rules."""

    # -- round_to_tick ----------------------------------------------------

    def test_round_to_tick_nearest(self):
        # 450.123 -> nearest 0.01 -> 450.12
        result = round_to_tick(450.123, 0.01, "nearest")
        assert result == pytest.approx(450.12)

    def test_round_to_tick_nearest_half_up(self):
        result = round_to_tick(450.125, 0.01, "nearest")
        assert result == pytest.approx(450.12) or result == pytest.approx(450.13)

    def test_round_to_tick_up(self):
        result = round_to_tick(450.121, 0.01, "up")
        assert result == pytest.approx(450.13)

    def test_round_to_tick_down(self):
        result = round_to_tick(450.129, 0.01, "down")
        assert result == pytest.approx(450.12)

    def test_round_to_tick_already_on_tick(self):
        result = round_to_tick(450.12, 0.01, "nearest")
        assert result == pytest.approx(450.12)

    def test_round_to_tick_futures_quarter(self):
        """MNQ tick size = 0.25"""
        result = round_to_tick(20000.30, 0.25, "nearest")
        assert result == pytest.approx(20000.25)

    def test_round_to_tick_futures_up(self):
        result = round_to_tick(20000.01, 0.25, "up")
        assert result == pytest.approx(20000.25)

    def test_round_to_tick_futures_down(self):
        result = round_to_tick(20000.49, 0.25, "down")
        assert result == pytest.approx(20000.25)

    # -- round_qty --------------------------------------------------------

    def test_round_qty_floors(self):
        assert round_qty(3.7) == 3

    def test_round_qty_enforces_minimum(self):
        assert round_qty(0.3) == 1

    def test_round_qty_zero_becomes_min(self):
        assert round_qty(0.0) == 1

    def test_round_qty_exact(self):
        assert round_qty(5.0) == 5

    def test_round_qty_custom_min(self):
        assert round_qty(0.5, min_qty=2.0) == 2

    # -- validate_price ---------------------------------------------------

    def test_validate_price_valid(self):
        assert validate_price(450.12, 0.01) is True

    def test_validate_price_invalid(self):
        assert validate_price(450.123, 0.01) is False

    def test_validate_price_on_zero(self):
        assert validate_price(0.0, 0.01) is True

    def test_validate_price_futures_valid(self):
        assert validate_price(20000.25, 0.25) is True

    def test_validate_price_futures_invalid(self):
        assert validate_price(20000.10, 0.25) is False


# ============================================================================
# 7. Cross-Strategy Integration Tests
# ============================================================================
class TestCrossStrategyIntegration:
    """End-to-end tests combining multiple OMS components across strategies."""

    @pytest.mark.asyncio
    async def test_atrss_entry_fill_triggers_helix_stop_tighten_via_bus(self):
        """Full flow: ATRSS entry fill -> coordinator -> bus -> Helix receives event."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        # Helix has an open long on QQQ
        coord.on_position_update(HELIX, "QQQ", qty=5, direction="LONG", entry_price=448.0)

        # Subscribe Helix to its event queue
        helix_q = bus.subscribe(HELIX)

        # Simulate ATRSS entry fill on QQQ
        coord.on_fill(ATRSS, "QQQ", side="BUY", role="ENTRY", price=450.0)

        # Verify Helix received the coordination event
        assert not helix_q.empty()
        event = helix_q.get_nowait()
        assert event.event_type == OMSEventType.COORDINATION
        assert event.payload["coordination_type"] == "TIGHTEN_STOP_BE"
        assert event.payload["symbol"] == "QQQ"

    @pytest.mark.asyncio
    async def test_heat_cap_prevents_entry_with_multiple_strategy_positions(self):
        """When ATRSS + Helix already consume risk, Breakout entry is denied."""
        port_risk = _clean_portfolio_risk()
        port_risk.open_risk_R = 1.0  # ATRSS and Helix positions
        port_risk.pending_entry_risk_R = 0.0

        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=_clean_strategy_risk(BREAKOUT),
            ),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )
        # Breakout wants to enter: risk = |450-440| * 1 * 10 = 100$; 100/350=0.286R
        # total = 1.0 + 0.286 = 1.286 > 1.25
        order = _make_order(
            strategy_id=BREAKOUT,
            planned_entry=450.0,
            stop_for_risk=440.0,
            qty=10,
        )
        reason = await gw.check_entry(order)
        assert reason is not None
        assert "heat cap" in reason.lower()

    @pytest.mark.asyncio
    async def test_priority_reservation_blocks_breakout_but_not_atrss(self):
        """Priority reservation reserves heat for higher-priority strategies."""
        port_risk = _clean_portfolio_risk()
        port_risk.open_risk_R = 1.1
        port_risk.pending_entry_risk_R = 0.0

        cfg = _default_risk_config()

        # Breakout (priority=2) should be blocked
        gw_breakout = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=_clean_strategy_risk(BREAKOUT),
            ),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )
        order_bo = _make_order(
            strategy_id=BREAKOUT, planned_entry=450.0, stop_for_risk=445.0, qty=10,
        )
        reason_bo = await gw_breakout.check_entry(order_bo)
        assert reason_bo is not None
        assert "reserved" in reason_bo.lower()

        # ATRSS (priority=0) should NOT be blocked
        gw_atrss = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=_clean_strategy_risk(ATRSS),
            ),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )
        order_at = _make_order(
            strategy_id=ATRSS, planned_entry=450.0, stop_for_risk=445.0, qty=10,
        )
        reason_at = await gw_atrss.check_entry(order_at)
        assert reason_at is None

    @pytest.mark.asyncio
    async def test_state_machine_plus_risk_gateway_flow(self):
        """Create order -> risk check -> approve -> transition through lifecycle."""
        gw = RiskGateway(
            config=_default_risk_config(),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
            get_working_order_count=AsyncMock(return_value=0),
        )
        bus = EventBus()
        q = bus.subscribe(ATRSS)

        # Create order
        order = _make_order(
            strategy_id=ATRSS, planned_entry=450.0, stop_for_risk=449.5, qty=1,
        )
        assert order.status == OrderStatus.CREATED
        bus.emit_order_event(order)

        # Risk check
        reason = await gw.check_entry(order)
        assert reason is None  # approved

        # Transition: CREATED -> RISK_APPROVED
        assert transition(order, OrderStatus.RISK_APPROVED) is True
        bus.emit_order_event(order)

        # Transition: RISK_APPROVED -> ROUTED
        assert transition(order, OrderStatus.ROUTED) is True
        bus.emit_order_event(order)

        # Transition: ROUTED -> ACKED
        assert transition(order, OrderStatus.ACKED) is True
        bus.emit_order_event(order)

        # Transition: ACKED -> WORKING
        assert transition(order, OrderStatus.WORKING) is True
        bus.emit_order_event(order)

        # Transition: WORKING -> FILLED
        assert transition(order, OrderStatus.FILLED) is True
        assert is_terminal(order) is True
        bus.emit_order_event(order)

        # Transition: FILLED -> DONE
        assert transition(order, OrderStatus.DONE) is True
        assert is_done(order) is True

        # Verify all expected events were emitted
        events = []
        while not q.empty():
            events.append(q.get_nowait())

        event_types = [e.event_type for e in events]
        assert OMSEventType.ORDER_CREATED in event_types
        assert OMSEventType.ORDER_RISK_APPROVED in event_types
        assert OMSEventType.ORDER_ROUTED in event_types
        assert OMSEventType.ORDER_ACKED in event_types
        assert OMSEventType.ORDER_WORKING in event_types
        assert OMSEventType.ORDER_FILLED in event_types

    @pytest.mark.asyncio
    async def test_denied_entry_emits_risk_denial_event(self):
        """When risk gateway denies entry, bus emits RISK_DENIAL to strategy."""
        cfg = _default_risk_config()
        cfg.global_standdown = True

        gw = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
        )
        bus = EventBus()
        q = bus.subscribe(ATRSS)

        order = _make_order(strategy_id=ATRSS)
        reason = await gw.check_entry(order)
        assert reason is not None

        # Emit denial event (as the OMS engine would)
        bus.emit_risk_denial(ATRSS, order.oms_order_id, reason)

        event = q.get_nowait()
        assert event.event_type == OMSEventType.RISK_DENIAL
        assert "stand" in event.payload["reason"].lower()

    @pytest.mark.asyncio
    async def test_full_three_strategy_coordination_scenario(self):
        """Scenario: All 3 strategies have positions; ATRSS entry triggers Helix tighten."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        # All 3 strategies have positions
        coord.on_position_update(ATRSS, "SPY", qty=10, direction="LONG", entry_price=500.0)
        coord.on_position_update(HELIX, "QQQ", qty=5, direction="LONG", entry_price=448.0)
        coord.on_position_update(BREAKOUT, "IWM", qty=20, direction="SHORT", entry_price=210.0)

        helix_q = bus.subscribe(HELIX)
        atrss_q = bus.subscribe(ATRSS)
        breakout_q = bus.subscribe(BREAKOUT)

        # ATRSS enters QQQ (same symbol as Helix position)
        coord.on_fill(ATRSS, "QQQ", side="BUY", role="ENTRY", price=450.0)

        # Helix should get TIGHTEN_STOP_BE
        assert not helix_q.empty()
        event = helix_q.get_nowait()
        assert event.payload["coordination_type"] == "TIGHTEN_STOP_BE"
        assert event.payload["symbol"] == "QQQ"

        # ATRSS and Breakout should NOT receive any coordination events
        assert atrss_q.empty()
        assert breakout_q.empty()

        # Verify Helix can query ATRSS position for size boost
        coord.on_position_update(ATRSS, "QQQ", qty=10, direction="LONG", entry_price=450.0)
        assert coord.has_atrss_position("QQQ", direction="LONG") is True

    @pytest.mark.asyncio
    async def test_risk_calculator_feeds_into_gateway_decision(self):
        """RiskCalculator computes unit_risk_dollars that feeds strategy config for gateway."""
        nav = 100_000.0
        unit_risk_pct = 0.0035
        vol_factor = 1.0

        unit_risk = RiskCalculator.compute_unit_risk_dollars(nav, unit_risk_pct, vol_factor)
        assert unit_risk == pytest.approx(350.0)

        cfg = _default_risk_config()
        cfg.strategy_configs[ATRSS].unit_risk_dollars = unit_risk

        # Compute order risk: |450 - 445| * 1.0 * 10 = 50$
        order_risk = RiskCalculator.compute_order_risk_dollars(450.0, 445.0, 10, 1.0)
        assert order_risk == pytest.approx(50.0)

        risk_R = RiskCalculator.compute_risk_R(order_risk, unit_risk)
        assert risk_R == pytest.approx(50.0 / 350.0)

        # Now test gateway with that config
        gw = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=_clean_strategy_risk()),
            get_portfolio_risk=AsyncMock(return_value=_clean_portfolio_risk()),
            get_working_order_count=AsyncMock(return_value=0),
        )
        order = _make_order(
            strategy_id=ATRSS, planned_entry=450.0, stop_for_risk=445.0, qty=10,
        )
        reason = await gw.check_entry(order)
        assert reason is None  # 0.143R is well within 1.25R cap

    @pytest.mark.asyncio
    async def test_tick_rules_validate_order_price_before_submission(self):
        """Tick rules validate price, then order proceeds through state machine."""
        inst = _make_instrument("QQQ", tick_size=0.01)

        # Valid limit price
        limit_price = 450.12
        assert validate_price(limit_price, inst.tick_size) is True

        # Round a non-conforming stop
        raw_stop = 445.123
        rounded_stop = round_to_tick(raw_stop, inst.tick_size, "down")
        assert validate_price(rounded_stop, inst.tick_size) is True
        assert rounded_stop == pytest.approx(445.12)

        # Create order with validated prices and walk through lifecycle
        order = OMSOrder(
            strategy_id=ATRSS,
            instrument=inst,
            side=OrderSide.BUY,
            qty=round_qty(10.7),  # -> 10
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
            limit_price=limit_price,
            risk_context=RiskContext(
                stop_for_risk=rounded_stop,
                planned_entry_price=limit_price,
            ),
        )
        assert order.qty == 10
        assert transition(order, OrderStatus.RISK_APPROVED) is True
        assert transition(order, OrderStatus.ROUTED) is True

    def test_instrument_point_value_defaults_to_multiplier(self):
        """Instrument.__post_init__ sets point_value = multiplier when 0."""
        inst = Instrument(
            symbol="MNQ", root="MNQ", venue="CME",
            tick_size=0.25, tick_value=0.50, multiplier=2.0,
        )
        assert inst.point_value == 2.0

    def test_instrument_explicit_point_value(self):
        inst = Instrument(
            symbol="MNQ", root="MNQ", venue="CME",
            tick_size=0.25, tick_value=0.50, multiplier=2.0, point_value=5.0,
        )
        assert inst.point_value == 5.0

    @pytest.mark.asyncio
    async def test_event_bus_global_subscriber_sees_all_strategy_events(self):
        """Global subscriber receives events from all 3 strategies."""
        bus = EventBus()
        global_q = bus.subscribe_all()

        for sid in (ATRSS, HELIX, BREAKOUT):
            order = _make_order(strategy_id=sid, status=OrderStatus.CREATED)
            bus.emit_order_event(order)

        events = []
        while not global_q.empty():
            events.append(global_q.get_nowait())

        assert len(events) == 3
        seen_strategies = {e.strategy_id for e in events}
        assert seen_strategies == {ATRSS, HELIX, BREAKOUT}

    @pytest.mark.asyncio
    async def test_helix_size_boost_with_coordinator_and_calculator(self):
        """Helix queries coordinator for ATRSS position, applies 1.25x boost to qty."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        # ATRSS has a long on QQQ
        coord.on_position_update(ATRSS, "QQQ", qty=10, direction="LONG", entry_price=450.0)

        # Helix wants to enter QQQ LONG
        base_qty = 4
        if coord.has_atrss_position("QQQ", direction="LONG"):
            boosted_qty = round_qty(base_qty * 1.25)  # 5.0 -> 5
        else:
            boosted_qty = base_qty

        assert boosted_qty == 5

        # Confirm risk calc with boosted qty
        risk = RiskCalculator.compute_order_risk_dollars(450.0, 445.0, boosted_qty, 1.0)
        assert risk == pytest.approx(25.0)

    @pytest.mark.asyncio
    async def test_concurrent_entries_from_all_strategies(self):
        """Simulate all 3 strategies submitting entries concurrently; risk gateway
        processes them in priority order (ATRSS first, then Helix, then Breakout)."""
        cfg = _default_risk_config()
        port_risk = _clean_portfolio_risk()
        port_risk.open_risk_R = 0.0

        strat_risks = {
            ATRSS: _clean_strategy_risk(ATRSS),
            HELIX: _clean_strategy_risk(HELIX),
            BREAKOUT: _clean_strategy_risk(BREAKOUT),
        }

        async def get_strat_risk(sid):
            return strat_risks[sid]

        gw = RiskGateway(
            config=cfg,
            calendar=EventCalendar(),
            get_strategy_risk=get_strat_risk,
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )

        # Small entries that fit within heat cap
        orders = [
            _make_order(
                strategy_id=sid, planned_entry=450.0, stop_for_risk=449.5, qty=1,
            )
            for sid in (ATRSS, HELIX, BREAKOUT)
        ]

        results = await asyncio.gather(*[gw.check_entry(o) for o in orders])

        # All should be approved (small risk, ample heat room)
        for i, reason in enumerate(results):
            assert reason is None, (
                f"{orders[i].strategy_id} should be approved but got: {reason}"
            )

    def test_oms_order_default_uuid_generated(self):
        """Each OMSOrder gets a unique UUID by default."""
        o1 = _make_order()
        o2 = _make_order()
        assert o1.oms_order_id != o2.oms_order_id
        assert len(o1.oms_order_id) > 0

    def test_terminal_statuses_set_contents(self):
        """Verify TERMINAL_STATUSES matches expected set."""
        assert TERMINAL_STATUSES == {
            OrderStatus.CANCELLED,
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    def test_risk_state_defaults(self):
        """StrategyRiskState and PortfolioRiskState have sane defaults."""
        srs = StrategyRiskState(strategy_id=ATRSS, trade_date=TODAY)
        assert srs.daily_realized_pnl == 0.0
        assert srs.daily_realized_R == 0.0
        assert srs.open_risk_dollars == 0.0
        assert srs.halted is False

        prs = PortfolioRiskState(trade_date=TODAY)
        assert prs.pending_entry_risk_R == 0.0
        assert prs.halted is False

    def test_event_calendar_not_blocked_when_empty(self):
        cal = EventCalendar()
        assert cal.is_blocked(datetime.now(timezone.utc)) is False

    def test_event_calendar_blocked_during_window(self):
        now = datetime.now(timezone.utc)
        window = EventWindow(
            name="FOMC",
            start_utc=now - timedelta(minutes=10),
            end_utc=now + timedelta(minutes=30),
        )
        cal = EventCalendar(windows=[window])
        assert cal.is_blocked(now) is True

    def test_event_calendar_not_blocked_outside_window(self):
        now = datetime.now(timezone.utc)
        window = EventWindow(
            name="NFP",
            start_utc=now + timedelta(hours=2),
            end_utc=now + timedelta(hours=3),
        )
        cal = EventCalendar(windows=[window])
        assert cal.is_blocked(now) is False


# ---------------------------------------------------------------------------
# 13. Coordinator Action Logging
# ---------------------------------------------------------------------------
class TestCoordinatorActionLogging:
    """Tests for the StrategyCoordinator action logging callback."""

    def test_rule1_calls_action_logger(self):
        """on_fill emitting Rule 1 should call the registered action logger."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        logged_calls = []
        coord.set_action_logger(lambda **kwargs: logged_calls.append(kwargs))

        # Set up a Helix position so Rule 1 triggers
        coord.on_position_update("AKC_HELIX", "QQQ", 10, "LONG", entry_price=480.0)

        # ATRSS entry fill on same symbol
        coord.on_fill("ATRSS", "QQQ", "BUY", "ENTRY", price=485.0)

        assert len(logged_calls) == 1
        call = logged_calls[0]
        assert call["action"] == "tighten_stop_be"
        assert call["trigger_strategy"] == "ATRSS"
        assert call["target_strategy"] == "AKC_HELIX"
        assert call["symbol"] == "QQQ"
        assert call["rule"] == "rule_1"
        assert call["outcome"] == "emitted"
        assert call["details"]["fill_price"] == 485.0

    def test_log_action_no_logger_no_crash(self):
        """log_action with no registered logger should be a no-op."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)
        # No set_action_logger called — should not raise
        coord.log_action(
            action="test", trigger_strategy="A", target_strategy="B",
            symbol="QQQ", rule="test", outcome="applied",
        )

    def test_log_action_callback_exception_swallowed(self):
        """If the callback raises, log_action should not propagate."""
        bus = EventBus()
        coord = StrategyCoordinator(bus)

        def bad_logger(**kwargs):
            raise RuntimeError("boom")

        coord.set_action_logger(bad_logger)
        # Should not raise
        coord.log_action(
            action="test", trigger_strategy="A", target_strategy="B",
            symbol="QQQ", rule="test", outcome="applied",
        )
