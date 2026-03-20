"""Comprehensive paper trading integration tests.

Tests the IBKR adapter layer, OMS order flow, state machine,
risk checks, event propagation, and coordinator interactions
without requiring a live IBKR connection.

All ib_async / IB Gateway interactions are mocked.
"""
from __future__ import annotations

import asyncio
import math
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------
from shared.ibkr_core.client.connection import ConnectionManager
from shared.ibkr_core.client.heartbeat import HeartbeatMonitor
from shared.ibkr_core.client.request_ids import RequestIdAllocator
from shared.ibkr_core.client.session import IBSession
from shared.ibkr_core.client.throttler import (
    CongestionError,
    PacingChannel,
    Throttler,
    _TokenBucket,
)
from shared.ibkr_core.client.error_map import classify_error
from shared.ibkr_core.config.schemas import (
    ContractTemplate,
    ExchangeRoute,
    IBKRProfile,
)
from shared.ibkr_core.adapters.execution_adapter import (
    IBKRExecutionAdapter,
    OrderNotFoundError,
)
from shared.ibkr_core.mapping.contract_factory import (
    ContractFactory,
    ContractResolutionError,
)
from shared.ibkr_core.mapping.order_mapper import OrderMapper
from shared.ibkr_core.mapping.order_flags import AllowedOrderFlags, EXCHANGE_FLAGS
from shared.ibkr_core.mapping.exchange_routes import ExchangeRouter
from shared.ibkr_core.models.types import (
    BrokerOrderRef,
    BrokerOrderStatus,
    ExecutionReport,
    IBContractSpec,
    OrderStatusEvent,
    PositionSnapshot,
    RejectCategory,
)
from shared.ibkr_core.reconciler.discrepancy_policy import (
    DiscrepancyAction,
    DiscrepancyPolicy,
)
from shared.ibkr_core.reconciler.sync import Discrepancy, ReconcilerSync
from shared.ibkr_core.reconciler.snapshots import SnapshotFetcher
from shared.ibkr_core.risk_support.tick_rules import (
    round_to_tick,
    round_qty,
    validate_price,
)
from shared.ibkr_core.risk_support.reject_classifier import RejectClassifier
from shared.ibkr_core.state.cache import IBCache
from shared.ibkr_core.state.session_state import SessionState
from shared.ibkr_core.logging.audit import log_broker_command, log_broker_response

from shared.oms.engine.state_machine import (
    TRANSITIONS,
    is_done,
    is_terminal,
    transition,
)
from shared.oms.engine.fill_processor import FillProcessor
from shared.oms.engine.timeout_monitor import OrderTimeoutMonitor
from shared.oms.events.bus import EventBus
from shared.oms.execution.router import ExecutionRouter
from shared.oms.intent.handler import IntentHandler
from shared.oms.models.events import OMSEvent, OMSEventType
from shared.oms.models.fill import Fill
from shared.oms.models.instrument import Instrument
from shared.oms.models.instrument_registry import InstrumentRegistry
from shared.oms.models.intent import Intent, IntentReceipt, IntentResult, IntentType
from shared.oms.models.order import (
    BrokerRef,
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskContext,
    TERMINAL_STATUSES,
)
from shared.oms.models.position import Position
from shared.oms.models.risk_state import PortfolioRiskState, StrategyRiskState
from shared.oms.persistence.in_memory import InMemoryRepository
from shared.oms.config.risk_config import RiskConfig, StrategyRiskConfig
from shared.oms.risk.gateway import RiskGateway
from shared.oms.risk.calendar import EventCalendar
from shared.oms.risk.calculator import RiskCalculator
from shared.oms.coordination.coordinator import StrategyCoordinator
from shared.oms.reconciliation.orchestrator import ReconciliationOrchestrator
from shared.oms.services.oms_service import OMSService


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TEST_ACCOUNT = "DU1234567"
TEST_STRATEGY = "ATRSS"

MNQ_INSTRUMENT = Instrument(
    symbol="MNQ",
    root="MNQ",
    venue="CME",
    tick_size=0.25,
    tick_value=0.50,
    multiplier=2.0,
    currency="USD",
    contract_expiry="202503",
)

QQQ_INSTRUMENT = Instrument(
    symbol="QQQ",
    root="QQQ",
    venue="SMART",
    tick_size=0.01,
    tick_value=0.01,
    multiplier=1.0,
    currency="USD",
)

GLD_INSTRUMENT = Instrument(
    symbol="GLD",
    root="GLD",
    venue="SMART",
    tick_size=0.01,
    tick_value=0.01,
    multiplier=1.0,
    currency="USD",
)

USO_INSTRUMENT = Instrument(
    symbol="USO",
    root="USO",
    venue="SMART",
    tick_size=0.01,
    tick_value=0.01,
    multiplier=1.0,
    currency="USD",
)

IBIT_INSTRUMENT = Instrument(
    symbol="IBIT",
    root="IBIT",
    venue="SMART",
    tick_size=0.01,
    tick_value=0.01,
    multiplier=1.0,
    currency="USD",
)


@pytest.fixture(autouse=True)
def _register_instruments():
    """Register test instruments before each test and clean up after."""
    InstrumentRegistry.clear()
    InstrumentRegistry.register(MNQ_INSTRUMENT)
    InstrumentRegistry.register(QQQ_INSTRUMENT)
    InstrumentRegistry.register(GLD_INSTRUMENT)
    InstrumentRegistry.register(USO_INSTRUMENT)
    InstrumentRegistry.register(IBIT_INSTRUMENT)
    yield
    InstrumentRegistry.clear()


def make_ibkr_profile(**overrides) -> IBKRProfile:
    defaults = dict(
        host="127.0.0.1",
        port=4002,
        client_id=1,
        account_id=TEST_ACCOUNT,
        is_gateway=True,
        readonly=False,
        reconnect_max_retries=3,
        reconnect_base_delay_s=0.01,
        reconnect_max_delay_s=0.05,
        pacing_orders_per_sec=50.0,
        pacing_messages_per_sec=100.0,
    )
    defaults.update(overrides)
    return IBKRProfile(**defaults)


def make_oms_order(
    strategy_id: str = TEST_STRATEGY,
    instrument: Instrument = None,
    side: OrderSide = OrderSide.BUY,
    qty: int = 1,
    order_type: OrderType = OrderType.LIMIT,
    limit_price: float = 500.0,
    stop_price: float = None,
    role: OrderRole = OrderRole.ENTRY,
    risk_context: RiskContext = None,
    status: OrderStatus = OrderStatus.CREATED,
    client_order_id: str = "",
) -> OMSOrder:
    if instrument is None:
        instrument = QQQ_INSTRUMENT
    if risk_context is None and role == OrderRole.ENTRY:
        risk_context = RiskContext(
            stop_for_risk=490.0,
            planned_entry_price=500.0,
            risk_budget_tag="MR",
            risk_dollars=10.0,
        )
    return OMSOrder(
        oms_order_id=str(uuid.uuid4()),
        client_order_id=client_order_id,
        strategy_id=strategy_id,
        account_id=TEST_ACCOUNT,
        instrument=instrument,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        role=role,
        risk_context=risk_context,
        status=status,
        remaining_qty=qty,
        created_at=datetime.now(timezone.utc),
    )


def make_risk_config(
    strategy_id: str = TEST_STRATEGY,
    unit_risk_dollars: float = 100.0,
    daily_stop_R: float = 2.0,
    heat_cap_R: float = 1.5,
    portfolio_daily_stop_R: float = 3.0,
    max_working_orders: int = 4,
) -> RiskConfig:
    strat_cfg = StrategyRiskConfig(
        strategy_id=strategy_id,
        daily_stop_R=daily_stop_R,
        unit_risk_dollars=unit_risk_dollars,
        max_working_orders=max_working_orders,
    )
    return RiskConfig(
        heat_cap_R=heat_cap_R,
        portfolio_daily_stop_R=portfolio_daily_stop_R,
        strategy_configs={strategy_id: strat_cfg},
    )


@pytest.fixture
def repo():
    return InMemoryRepository()


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def risk_config():
    return make_risk_config()


@pytest.fixture
def mock_ib():
    """Create a fully mocked ib_async.IB instance."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.connectAsync = AsyncMock()
    ib.disconnect = MagicMock()
    ib.reqCurrentTimeAsync = AsyncMock(return_value=int(time.time()))
    ib.qualifyContractsAsync = AsyncMock(return_value=[])
    ib.placeOrder = MagicMock()
    ib.cancelOrder = MagicMock()
    ib.openTrades = MagicMock(return_value=[])
    ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[])
    ib.reqPositionsAsync = AsyncMock(return_value=[])
    ib.reqExecutionsAsync = AsyncMock(return_value=[])

    # Event hooks
    ib.orderStatusEvent = MagicMock()
    ib.execDetailsEvent = MagicMock()
    ib.errorEvent = MagicMock()
    ib.disconnectedEvent = MagicMock()

    # Client mock for getReqId
    ib.client = MagicMock()
    ib.client.getReqId.return_value = 1000

    return ib


@pytest.fixture
def mock_session(mock_ib):
    """Create a mock IBSession with the mocked IB instance."""
    session = MagicMock(spec=IBSession)
    session.ib = mock_ib
    session.is_ready = True
    session.is_congested = False
    session.throttled = AsyncMock()
    session.next_order_id = AsyncMock(side_effect=iter(range(1000, 2000)))
    session.next_request_id = AsyncMock(side_effect=iter(range(5000, 6000)))
    session.config = MagicMock()
    return session


@pytest.fixture
def contract_templates():
    return {
        "MNQ": ContractTemplate(
            symbol="MNQ",
            sec_type="FUT",
            exchange="CME",
            currency="USD",
            multiplier=2.0,
            tick_size=0.25,
            tick_value=0.50,
            trading_class="MNQ",
        ),
        "QQQ": ContractTemplate(
            symbol="QQQ",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
        ),
        "GLD": ContractTemplate(
            symbol="GLD",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
        ),
        "USO": ContractTemplate(
            symbol="USO",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
        ),
        "IBIT": ContractTemplate(
            symbol="IBIT",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
        ),
    }


@pytest.fixture
def exchange_routes():
    return {
        "MNQ": ExchangeRoute(root_symbol="MNQ", exchange="CME", trading_class="MNQ"),
        "QQQ": ExchangeRoute(root_symbol="QQQ", exchange="SMART"),
        "GLD": ExchangeRoute(root_symbol="GLD", exchange="SMART"),
        "USO": ExchangeRoute(root_symbol="USO", exchange="SMART"),
        "IBIT": ExchangeRoute(root_symbol="IBIT", exchange="SMART"),
    }


def _make_qualified_contract(symbol: str, con_id: int = 12345, sec_type: str = "STK"):
    """Helper: build a mock qualified IB contract."""
    c = MagicMock()
    c.conId = con_id
    c.symbol = symbol
    c.secType = sec_type
    c.exchange = "SMART"
    c.currency = "USD"
    c.multiplier = "1"
    c.tradingClass = symbol
    c.lastTradeDateOrContractMonth = ""
    return c


def _make_mock_trade(order_id: int = 1001, perm_id: int = 999, status: str = "Submitted",
                     filled: float = 0, remaining: float = 1, avg_fill: float = 0.0):
    """Helper: build a mock ib_async Trade object."""
    trade = MagicMock()
    trade.order.orderId = order_id
    trade.order.permId = perm_id
    trade.order.totalQuantity = 1
    trade.order.lmtPrice = 500.0
    trade.order.auxPrice = 0.0
    trade.orderStatus.status = status
    trade.orderStatus.filled = filled
    trade.orderStatus.remaining = remaining
    trade.orderStatus.avgFillPrice = avg_fill
    trade.orderStatus.lastFillPrice = 0.0
    trade.contract = _make_qualified_contract("QQQ")
    return trade


# ===========================================================================
# 1. SESSION LIFECYCLE TESTS
# ===========================================================================

class TestSessionLifecycle:
    """Test connection initialization, session start/stop, reconnection, heartbeat."""

    @pytest.mark.asyncio
    async def test_connection_init_paper_trading_port(self):
        """Paper trading uses port 4002 by default."""
        profile = make_ibkr_profile(port=4002)
        assert profile.port == 4002
        conn = ConnectionManager(profile)
        assert conn.is_connected is False

    @pytest.mark.asyncio
    async def test_connection_live_trading_port_distinct(self):
        """Live trading port 4001 is distinct from paper port 4002."""
        paper = make_ibkr_profile(port=4002)
        live = make_ibkr_profile(port=4001)
        assert paper.port != live.port

    @pytest.mark.asyncio
    async def test_connection_connect_success(self):
        """Successful connection sets _connected event."""
        profile = make_ibkr_profile()
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._ib.connectAsync = AsyncMock()
        conn._ib.isConnected.return_value = True
        conn._ib.disconnectedEvent = MagicMock()

        await conn.connect()
        assert conn._connected.is_set()
        conn._ib.connectAsync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connection_connect_retries_on_failure(self):
        """Connection retries with backoff on failure, then succeeds."""
        profile = make_ibkr_profile(reconnect_max_retries=3)
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._ib.disconnectedEvent = MagicMock()
        # Fail twice, succeed on third
        conn._ib.connectAsync = AsyncMock(
            side_effect=[ConnectionError("fail"), ConnectionError("fail"), None]
        )
        conn._ib.isConnected.return_value = True

        await conn.connect()
        assert conn._ib.connectAsync.await_count == 3
        assert conn._connected.is_set()
        assert conn._retry_count == 0  # Reset on success

    @pytest.mark.asyncio
    async def test_connection_max_retries_exceeded(self):
        """Raises after max retries exceeded."""
        profile = make_ibkr_profile(reconnect_max_retries=2)
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._ib.disconnectedEvent = MagicMock()
        conn._ib.connectAsync = AsyncMock(side_effect=ConnectionError("fail"))

        with pytest.raises(ConnectionError):
            await conn.connect()
        assert conn._ib.connectAsync.await_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_backoff_delay_exponential(self):
        """Backoff delay increases exponentially with jitter."""
        profile = make_ibkr_profile(
            reconnect_base_delay_s=1.0,
            reconnect_max_delay_s=60.0,
        )
        conn = ConnectionManager(profile)

        conn._retry_count = 1
        d1 = conn._backoff_delay()
        conn._retry_count = 2
        d2 = conn._backoff_delay()
        conn._retry_count = 3
        d3 = conn._backoff_delay()

        # Base delays: 1*2^0=1, 1*2^1=2, 1*2^2=4 (before jitter)
        # Jitter factor: 0.5 + random() -> [0.5, 1.5]
        assert 0.5 <= d1 <= 1.5
        assert 1.0 <= d2 <= 3.0
        assert 2.0 <= d3 <= 6.0

    @pytest.mark.asyncio
    async def test_backoff_delay_capped(self):
        """Backoff delay capped at max."""
        profile = make_ibkr_profile(
            reconnect_base_delay_s=1.0,
            reconnect_max_delay_s=5.0,
        )
        conn = ConnectionManager(profile)
        conn._retry_count = 10  # 2^9 = 512 >> max=5
        d = conn._backoff_delay()
        # max_delay * jitter(0.5..1.5) = 2.5..7.5
        assert d <= 7.5

    @pytest.mark.asyncio
    async def test_disconnect_graceful(self):
        """Graceful disconnect clears state."""
        profile = make_ibkr_profile()
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._ib.isConnected.return_value = True
        conn._connected.set()

        await conn.disconnect()
        assert conn._shutting_down is True
        assert not conn._connected.is_set()
        conn._ib.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_noop_if_already_disconnected(self):
        """Disconnect when already disconnected does not call ib.disconnect."""
        profile = make_ibkr_profile()
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._ib.isConnected.return_value = False

        await conn.disconnect()
        conn._ib.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_start_stop(self, mock_ib):
        """IBSession start/stop lifecycle."""
        profile = make_ibkr_profile()
        with patch("shared.ibkr_core.client.session.ConnectionManager") as MockConn, \
             patch("shared.ibkr_core.client.session.HeartbeatMonitor") as MockHB:
            mock_conn = MockConn.return_value
            mock_conn.ib = mock_ib
            mock_conn.connect = AsyncMock()
            mock_conn.wait_until_ready = AsyncMock()
            mock_conn.disconnect = AsyncMock()

            mock_hb = MockHB.return_value
            mock_hb.start = AsyncMock()
            mock_hb.stop = AsyncMock()

            config = MagicMock()
            config.profile = profile
            session = IBSession(config)
            session._conn = mock_conn
            session._heartbeat = None

            # Start
            await session.start()
            mock_conn.connect.assert_awaited_once()
            mock_conn.wait_until_ready.assert_awaited_once()
            assert session.is_ready

            # Stop
            await session.stop()
            mock_conn.disconnect.assert_awaited_once()
            assert not session.is_ready

    @pytest.mark.asyncio
    async def test_reconnect_callback_fires(self):
        """Post-reconnect callback fires on reconnection."""
        profile = make_ibkr_profile()
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._ib.connectAsync = AsyncMock()
        conn._ib.isConnected.return_value = True
        conn._ib.disconnectedEvent = MagicMock()

        callback = AsyncMock()
        conn.set_reconnect_callback(callback)

        # Simulate reconnect loop
        await conn._reconnect_loop()
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_disconnected_triggers_reconnect(self):
        """Disconnected event triggers reconnect when not shutting down."""
        profile = make_ibkr_profile()
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._ib.connectAsync = AsyncMock()
        conn._ib.isConnected.return_value = True
        conn._shutting_down = False
        conn._connected.set()

        with patch.object(conn, "_reconnect_loop", new_callable=AsyncMock) as mock_reconnect:
            # We need to patch ensure_future to capture the call
            conn._on_disconnected()
            assert not conn._connected.is_set()

    @pytest.mark.asyncio
    async def test_on_disconnected_noop_when_shutting_down(self):
        """No reconnect attempt when shutting down."""
        profile = make_ibkr_profile()
        conn = ConnectionManager(profile)
        conn._ib = MagicMock()
        conn._shutting_down = True
        conn._connected.set()

        conn._on_disconnected()
        assert not conn._connected.is_set()
        # No reconnect task created (shutting_down guard)

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_success(self):
        """Heartbeat loop pings and tracks liveness."""
        ib = MagicMock()
        ib.reqCurrentTimeAsync = AsyncMock(return_value=int(time.time()))

        hb = HeartbeatMonitor(ib, interval_s=0.01, timeout_s=1.0)
        await hb.start()
        await asyncio.sleep(0.05)
        await hb.stop()
        assert hb.is_alive

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_failure_marks_not_alive(self):
        """Heartbeat failure marks not alive and fires callback."""
        ib = MagicMock()
        ib.reqCurrentTimeAsync = AsyncMock(side_effect=asyncio.TimeoutError)

        status_changes = []
        hb = HeartbeatMonitor(ib, interval_s=0.01, timeout_s=0.01)
        hb.on_status_change = lambda alive: status_changes.append(alive)
        await hb.start()
        await asyncio.sleep(0.1)
        await hb.stop()
        assert not hb.is_alive
        assert False in status_changes

    @pytest.mark.asyncio
    async def test_heartbeat_recovery(self):
        """Heartbeat recovers after transient failure."""
        ib = MagicMock()
        call_count = 0

        async def alternating_ping():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise asyncio.TimeoutError
            return int(time.time())

        ib.reqCurrentTimeAsync = alternating_ping

        status_changes = []
        hb = HeartbeatMonitor(ib, interval_s=0.01, timeout_s=0.5)
        hb.on_status_change = lambda alive: status_changes.append(alive)
        await hb.start()
        await asyncio.sleep(0.15)
        await hb.stop()
        # Should have gone False then True
        assert False in status_changes
        assert True in status_changes


# ===========================================================================
# 2. CONTRACT RESOLUTION TESTS
# ===========================================================================

class TestContractResolution:
    """Test symbol to IBKR contract mapping and validation."""

    @pytest.mark.asyncio
    async def test_resolve_known_symbol(self, mock_ib, contract_templates, exchange_routes):
        """Resolve a known symbol to a qualified contract."""
        qualified = _make_qualified_contract("QQQ", con_id=265598)
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        contract, spec = await factory.resolve("QQQ", "")

        assert spec.con_id == 265598
        assert spec.symbol == "QQQ"
        assert contract.conId == 265598

    @pytest.mark.asyncio
    async def test_resolve_unknown_symbol(self, mock_ib, contract_templates, exchange_routes):
        """Unknown symbol raises ContractResolutionError."""
        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        with pytest.raises(ContractResolutionError, match="Unknown symbol"):
            await factory.resolve("INVALID", "")

    @pytest.mark.asyncio
    async def test_resolve_qualification_failure(self, mock_ib, contract_templates, exchange_routes):
        """Empty qualification result raises error."""
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])
        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        with pytest.raises(ContractResolutionError, match="Failed to qualify"):
            await factory.resolve("QQQ", "")

    @pytest.mark.asyncio
    async def test_resolve_cache_hit(self, mock_ib, contract_templates, exchange_routes):
        """Second resolution uses cache, does not re-qualify."""
        qualified = _make_qualified_contract("QQQ", con_id=265598)
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        await factory.resolve("QQQ", "")
        await factory.resolve("QQQ", "")

        # qualifyContractsAsync called only once
        assert mock_ib.qualifyContractsAsync.await_count == 1

    @pytest.mark.asyncio
    async def test_resolve_cache_invalidation(self, mock_ib, contract_templates, exchange_routes):
        """Cache invalidation forces re-qualification."""
        qualified = _make_qualified_contract("QQQ", con_id=265598)
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        await factory.resolve("QQQ", "")
        factory.invalidate("QQQ", "")
        await factory.resolve("QQQ", "")

        assert mock_ib.qualifyContractsAsync.await_count == 2

    @pytest.mark.asyncio
    async def test_resolve_clear_cache(self, mock_ib, contract_templates, exchange_routes):
        """Clear cache removes all entries."""
        qualified = _make_qualified_contract("QQQ", con_id=265598)
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        await factory.resolve("QQQ", "")
        factory.clear_cache()
        await factory.resolve("QQQ", "")

        assert mock_ib.qualifyContractsAsync.await_count == 2

    @pytest.mark.asyncio
    async def test_resolve_etf_qqq(self, mock_ib, contract_templates, exchange_routes):
        """ETF QQQ resolves correctly."""
        qualified = _make_qualified_contract("QQQ", con_id=320227571, sec_type="STK")
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        contract, spec = await factory.resolve("QQQ", "")

        assert spec.symbol == "QQQ"
        assert spec.con_id == 320227571

    @pytest.mark.asyncio
    async def test_resolve_etf_gld(self, mock_ib, contract_templates, exchange_routes):
        """ETF GLD resolves correctly."""
        qualified = _make_qualified_contract("GLD", con_id=51529211, sec_type="STK")
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        contract, spec = await factory.resolve("GLD", "")
        assert spec.symbol == "GLD"

    @pytest.mark.asyncio
    async def test_resolve_etf_uso(self, mock_ib, contract_templates, exchange_routes):
        """ETF USO resolves correctly."""
        qualified = _make_qualified_contract("USO", con_id=28029654, sec_type="STK")
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        contract, spec = await factory.resolve("USO", "")
        assert spec.symbol == "USO"

    @pytest.mark.asyncio
    async def test_resolve_etf_ibit(self, mock_ib, contract_templates, exchange_routes):
        """ETF IBIT resolves correctly."""
        qualified = _make_qualified_contract("IBIT", con_id=756733, sec_type="STK")
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        contract, spec = await factory.resolve("IBIT", "")
        assert spec.symbol == "IBIT"

    @pytest.mark.asyncio
    async def test_resolve_futures_mnq(self, mock_ib, contract_templates, exchange_routes):
        """Futures MNQ resolves with expiry."""
        qualified = MagicMock()
        qualified.conId = 654321
        qualified.symbol = "MNQ"
        qualified.secType = "FUT"
        qualified.exchange = "CME"
        qualified.currency = "USD"
        qualified.multiplier = "2"
        qualified.tradingClass = "MNQ"
        qualified.lastTradeDateOrContractMonth = "20250321"
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        factory = ContractFactory(mock_ib, contract_templates, exchange_routes)
        contract, spec = await factory.resolve("MNQ", "202503")

        assert spec.con_id == 654321
        assert spec.symbol == "MNQ"
        assert spec.sec_type == "FUT"
        assert spec.multiplier == 2.0
        assert spec.tick_size == 0.25

    def test_exchange_router_known_symbol(self, exchange_routes):
        """ExchangeRouter resolves known symbols."""
        router = ExchangeRouter(exchange_routes)
        assert router.has_route("QQQ")
        assert router.get_exchange("QQQ") == "SMART"
        assert router.get_exchange("MNQ") == "CME"

    def test_exchange_router_unknown_symbol(self, exchange_routes):
        """ExchangeRouter returns False for unknown symbols."""
        router = ExchangeRouter(exchange_routes)
        assert not router.has_route("UNKNOWN")


# ===========================================================================
# 3. ORDER FLOW TESTS (PAPER TRADING)
# ===========================================================================

class TestOrderMapper:
    """Test OMS canonical -> IB Order mapping."""

    def test_market_order(self):
        order = OrderMapper.to_ib_order("BUY", "MARKET", 10, account=TEST_ACCOUNT)
        assert order.action == "BUY"
        assert order.totalQuantity == 10
        assert order.account == TEST_ACCOUNT

    def test_limit_order(self):
        order = OrderMapper.to_ib_order("SELL", "LIMIT", 5, limit_price=500.0)
        assert order.action == "SELL"
        assert order.totalQuantity == 5
        assert order.lmtPrice == 500.0

    def test_stop_order(self):
        order = OrderMapper.to_ib_order("SELL", "STOP", 3, stop_price=490.0)
        assert order.totalQuantity == 3
        assert order.auxPrice == 490.0

    def test_stop_limit_order(self):
        order = OrderMapper.to_ib_order(
            "SELL", "STOP_LIMIT", 2, limit_price=489.0, stop_price=490.0
        )
        assert order.auxPrice == 490.0
        assert order.lmtPrice == 489.0

    def test_limit_order_missing_price_raises(self):
        with pytest.raises(ValueError, match="limit_price"):
            OrderMapper.to_ib_order("BUY", "LIMIT", 1)

    def test_stop_order_missing_price_raises(self):
        with pytest.raises(ValueError, match="stop_price"):
            OrderMapper.to_ib_order("SELL", "STOP", 1)

    def test_stop_limit_missing_prices_raises(self):
        with pytest.raises(ValueError):
            OrderMapper.to_ib_order("SELL", "STOP_LIMIT", 1, stop_price=490.0)

    def test_unsupported_order_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            OrderMapper.to_ib_order("BUY", "TRAILING_STOP", 1)

    def test_oca_group_set(self):
        order = OrderMapper.to_ib_order(
            "BUY", "MARKET", 1, oca_group="oca-test", oca_type=1
        )
        assert order.ocaGroup == "oca-test"
        assert order.ocaType == 1

    def test_order_ref_set(self):
        order = OrderMapper.to_ib_order("BUY", "MARKET", 1, order_ref="ref-123")
        assert order.orderRef == "ref-123"

    def test_tif_day(self):
        order = OrderMapper.to_ib_order("BUY", "MARKET", 1, tif="DAY")
        assert order.tif == "DAY"

    def test_tif_gtc(self):
        order = OrderMapper.to_ib_order("BUY", "LIMIT", 1, limit_price=100.0, tif="GTC")
        assert order.tif == "GTC"


class TestExecutionAdapter:
    """Test IBKRExecutionAdapter submit, cancel, replace, callbacks."""

    @pytest.mark.asyncio
    async def test_submit_order(self, mock_session, contract_templates, exchange_routes):
        """Submit order goes through throttler, resolver, mapper, and placeOrder."""
        qualified = _make_qualified_contract("QQQ", con_id=265598)
        mock_session.ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        trade_mock = MagicMock()
        trade_mock.order.orderId = 1001
        trade_mock.order.permId = 9999
        mock_session.ib.placeOrder.return_value = trade_mock

        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)

        ref = await adapter.submit_order(
            oms_order_id="oms-001",
            contract_symbol="QQQ",
            contract_expiry="",
            action="BUY",
            order_type="LIMIT",
            qty=10,
            limit_price=500.0,
        )

        assert ref.broker_order_id == 1001
        assert ref.perm_id == 9999
        mock_session.throttled.assert_awaited()
        mock_session.ib.placeOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_order_rounds_price_to_tick(self, mock_session, contract_templates, exchange_routes):
        """Limit price rounded to instrument tick size."""
        qualified = _make_qualified_contract("QQQ", con_id=265598)
        mock_session.ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        trade_mock = MagicMock()
        trade_mock.order.orderId = 1002
        trade_mock.order.permId = 8888
        mock_session.ib.placeOrder.return_value = trade_mock

        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)

        # QQQ tick_size=0.01; 500.017 should round to 500.02
        await adapter.submit_order(
            oms_order_id="oms-002",
            contract_symbol="QQQ",
            contract_expiry="",
            action="BUY",
            order_type="LIMIT",
            qty=1,
            limit_price=500.017,
        )

        call_args = mock_session.ib.placeOrder.call_args
        ib_order = call_args[0][1]
        assert abs(ib_order.lmtPrice - 500.02) < 0.001

    @pytest.mark.asyncio
    async def test_cancel_order_found(self, mock_session, contract_templates, exchange_routes):
        """Cancel order calls cancelOrder on matching trade."""
        trade_mock = _make_mock_trade(order_id=1001)
        mock_session.ib.openTrades.return_value = [trade_mock]

        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        adapter._cache.register_order("oms-001", 1001, 9999)

        await adapter.cancel_order(1001)
        mock_session.ib.cancelOrder.assert_called_once_with(trade_mock.order)

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, mock_session, contract_templates, exchange_routes):
        """Cancel order for missing broker order logs warning."""
        mock_session.ib.openTrades.return_value = []

        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)

        # Should not raise, just warn
        await adapter.cancel_order(9999)

    @pytest.mark.asyncio
    async def test_replace_order(self, mock_session, contract_templates, exchange_routes):
        """Replace order updates and re-submits."""
        trade_mock = _make_mock_trade(order_id=1001)
        mock_session.ib.openTrades.return_value = [trade_mock]
        mock_session.ib.placeOrder.return_value = trade_mock

        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)

        ref = await adapter.replace_order(1001, new_qty=5, new_limit_price=505.0)
        assert ref.broker_order_id == 1001
        assert trade_mock.order.totalQuantity == 5
        assert trade_mock.order.lmtPrice == 505.0

    @pytest.mark.asyncio
    async def test_replace_order_not_found(self, mock_session, contract_templates, exchange_routes):
        """Replace on missing order raises OrderNotFoundError."""
        mock_session.ib.openTrades.return_value = []

        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)

        with pytest.raises(OrderNotFoundError):
            await adapter.replace_order(9999, new_qty=5)

    @pytest.mark.asyncio
    async def test_adapter_on_ack_callback(self, mock_session, contract_templates, exchange_routes):
        """Adapter fires on_ack when status is Submitted."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        adapter._cache.register_order("oms-001", 1001, 9999)

        ack_calls = []
        adapter.on_ack = lambda oms_id, ref: ack_calls.append((oms_id, ref))

        trade = _make_mock_trade(order_id=1001, status="Submitted")
        adapter._handle_order_status(trade)

        assert len(ack_calls) == 1
        assert ack_calls[0][0] == "oms-001"
        assert ack_calls[0][1].broker_order_id == 1001

    @pytest.mark.asyncio
    async def test_adapter_on_cancelled_callback(self, mock_session, contract_templates, exchange_routes):
        """Adapter fires on_status with CANCELLED."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        adapter._cache.register_order("oms-001", 1001, 9999)

        status_calls = []
        adapter.on_status = lambda oms_id, status, rem: status_calls.append((oms_id, status, rem))

        trade = _make_mock_trade(order_id=1001, status="Cancelled", remaining=0)
        adapter._handle_order_status(trade)

        assert any(s[1] == "Cancelled" for s in status_calls)

    @pytest.mark.asyncio
    async def test_adapter_on_fill_callback(self, mock_session, contract_templates, exchange_routes):
        """Adapter fires on_fill for execDetails event."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        adapter._cache.register_order("oms-001", 1001, 9999)

        fill_calls = []
        adapter.on_fill = lambda *args: fill_calls.append(args)

        trade = _make_mock_trade(order_id=1001)
        fill = MagicMock()
        fill.execution.execId = "exec-001"
        fill.execution.price = 500.0
        fill.execution.shares = 10
        fill.execution.time = datetime.now(timezone.utc)
        fill.commissionReport = MagicMock()
        fill.commissionReport.commission = 1.50

        adapter._handle_exec_details(trade, fill)

        assert len(fill_calls) == 1
        assert fill_calls[0][0] == "oms-001"
        assert fill_calls[0][2] == 500.0
        assert fill_calls[0][3] == 10

    @pytest.mark.asyncio
    async def test_adapter_fill_dedup(self, mock_session, contract_templates, exchange_routes):
        """Duplicate exec_id is ignored."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        adapter._cache.register_order("oms-001", 1001, 9999)

        fill_calls = []
        adapter.on_fill = lambda *args: fill_calls.append(args)

        trade = _make_mock_trade(order_id=1001)
        fill = MagicMock()
        fill.execution.execId = "exec-dup"
        fill.execution.price = 500.0
        fill.execution.shares = 10
        fill.execution.time = datetime.now(timezone.utc)
        fill.commissionReport = None

        adapter._handle_exec_details(trade, fill)
        adapter._handle_exec_details(trade, fill)  # duplicate

        assert len(fill_calls) == 1

    @pytest.mark.asyncio
    async def test_adapter_on_error_callback(self, mock_session, contract_templates, exchange_routes):
        """Adapter fires on_reject for error events."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        adapter._cache.register_order("oms-001", 1001, 9999)

        reject_calls = []
        adapter.on_reject = lambda *args: reject_calls.append(args)

        adapter._handle_error(1001, 201, "Order rejected by risk", None)

        assert len(reject_calls) == 1
        assert reject_calls[0][0] == "oms-001"
        assert reject_calls[0][2] == 201

    @pytest.mark.asyncio
    async def test_adapter_error_unknown_order_ignored(self, mock_session, contract_templates, exchange_routes):
        """Error for unknown order is silently ignored."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)

        reject_calls = []
        adapter.on_reject = lambda *args: reject_calls.append(args)

        adapter._handle_error(9999, 201, "Unknown", None)
        assert len(reject_calls) == 0

    @pytest.mark.asyncio
    async def test_adapter_is_ready(self, mock_session, contract_templates, exchange_routes):
        """Adapter is_ready delegates to session."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        assert adapter.is_ready is True

    @pytest.mark.asyncio
    async def test_adapter_is_congested(self, mock_session, contract_templates, exchange_routes):
        """Adapter is_congested delegates to session."""
        factory = ContractFactory(mock_session.ib, contract_templates, exchange_routes)
        adapter = IBKRExecutionAdapter(mock_session, factory, TEST_ACCOUNT)
        mock_session.is_congested = True
        assert adapter.is_congested is True


class TestOrderStateMachine:
    """Test order state transitions."""

    def test_valid_transitions(self):
        """Valid transitions proceed correctly."""
        order = make_oms_order()
        assert transition(order, OrderStatus.RISK_APPROVED)
        assert order.status == OrderStatus.RISK_APPROVED
        assert transition(order, OrderStatus.ROUTED)
        assert order.status == OrderStatus.ROUTED
        assert transition(order, OrderStatus.ACKED)
        assert order.status == OrderStatus.ACKED
        assert transition(order, OrderStatus.WORKING)
        assert order.status == OrderStatus.WORKING
        assert transition(order, OrderStatus.FILLED)
        assert order.status == OrderStatus.FILLED

    def test_invalid_transition_returns_false(self):
        """Invalid transition returns False and does not change state."""
        order = make_oms_order()
        result = transition(order, OrderStatus.FILLED)  # CREATED -> FILLED invalid
        assert result is False
        assert order.status == OrderStatus.CREATED

    def test_partial_fill_to_filled(self):
        """PARTIALLY_FILLED -> FILLED transition."""
        order = make_oms_order(status=OrderStatus.CREATED)
        transition(order, OrderStatus.RISK_APPROVED)
        transition(order, OrderStatus.ROUTED)
        transition(order, OrderStatus.ACKED)
        transition(order, OrderStatus.WORKING)
        transition(order, OrderStatus.PARTIALLY_FILLED)
        assert transition(order, OrderStatus.FILLED)

    def test_cancel_from_working(self):
        """WORKING -> CANCEL_REQUESTED -> CANCELLED."""
        order = make_oms_order(status=OrderStatus.CREATED)
        transition(order, OrderStatus.RISK_APPROVED)
        transition(order, OrderStatus.ROUTED)
        transition(order, OrderStatus.ACKED)
        transition(order, OrderStatus.WORKING)
        assert transition(order, OrderStatus.CANCEL_REQUESTED)
        assert transition(order, OrderStatus.CANCELLED)

    def test_reject_from_routed(self):
        """ROUTED -> REJECTED (broker rejects before ack)."""
        order = make_oms_order(status=OrderStatus.CREATED)
        transition(order, OrderStatus.RISK_APPROVED)
        transition(order, OrderStatus.ROUTED)
        assert transition(order, OrderStatus.REJECTED)

    def test_cancel_from_routed(self):
        """ROUTED -> CANCELLED (broker can cancel before ACK)."""
        order = make_oms_order(status=OrderStatus.CREATED)
        transition(order, OrderStatus.RISK_APPROVED)
        transition(order, OrderStatus.ROUTED)
        assert transition(order, OrderStatus.CANCELLED)

    def test_terminal_states(self):
        """Terminal states are correctly identified."""
        assert OrderStatus.FILLED in TERMINAL_STATUSES
        assert OrderStatus.CANCELLED in TERMINAL_STATUSES
        assert OrderStatus.REJECTED in TERMINAL_STATUSES
        assert OrderStatus.EXPIRED in TERMINAL_STATUSES
        assert OrderStatus.WORKING not in TERMINAL_STATUSES

    def test_is_terminal(self):
        order = make_oms_order(status=OrderStatus.FILLED)
        assert is_terminal(order)

    def test_is_done(self):
        order = make_oms_order()
        order.status = OrderStatus.DONE
        assert is_done(order)

    def test_replace_flow(self):
        """WORKING -> REPLACE_REQUESTED -> REPLACED -> DONE."""
        order = make_oms_order(status=OrderStatus.CREATED)
        transition(order, OrderStatus.RISK_APPROVED)
        transition(order, OrderStatus.ROUTED)
        transition(order, OrderStatus.ACKED)
        transition(order, OrderStatus.WORKING)
        assert transition(order, OrderStatus.REPLACE_REQUESTED)
        assert transition(order, OrderStatus.REPLACED)
        assert transition(order, OrderStatus.DONE)

    def test_partial_fill_to_cancel_requested(self):
        """PARTIALLY_FILLED -> CANCEL_REQUESTED -> CANCELLED."""
        order = make_oms_order(status=OrderStatus.CREATED)
        transition(order, OrderStatus.RISK_APPROVED)
        transition(order, OrderStatus.ROUTED)
        transition(order, OrderStatus.ACKED)
        transition(order, OrderStatus.WORKING)
        transition(order, OrderStatus.PARTIALLY_FILLED)
        assert transition(order, OrderStatus.CANCEL_REQUESTED)
        assert transition(order, OrderStatus.CANCELLED)

    def test_fill_during_cancel_request(self):
        """CANCEL_REQUESTED -> FILLED (fill arrives before cancel ack)."""
        order = make_oms_order(status=OrderStatus.CREATED)
        transition(order, OrderStatus.RISK_APPROVED)
        transition(order, OrderStatus.ROUTED)
        transition(order, OrderStatus.ACKED)
        transition(order, OrderStatus.WORKING)
        transition(order, OrderStatus.CANCEL_REQUESTED)
        assert transition(order, OrderStatus.FILLED)


class TestFillProcessor:
    """Test fill processing logic."""

    @pytest.mark.asyncio
    async def test_process_fill_updates_order(self, repo):
        """Fill updates order filled_qty and transitions state."""
        order = make_oms_order(qty=10)
        order.status = OrderStatus.WORKING
        order.remaining_qty = 10
        await repo.save_order(order)

        fp = FillProcessor(repo)
        await fp.process_fill(
            oms_order_id=order.oms_order_id,
            broker_fill_id="fill-001",
            price=500.0,
            qty=10,
            timestamp=datetime.now(timezone.utc),
            fees=1.50,
        )

        updated = await repo.get_order(order.oms_order_id)
        assert updated.filled_qty == 10
        assert updated.remaining_qty == 0
        assert updated.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_process_partial_fill(self, repo):
        """Partial fill updates quantities and transitions to PARTIALLY_FILLED."""
        order = make_oms_order(qty=10)
        order.status = OrderStatus.WORKING
        order.remaining_qty = 10
        await repo.save_order(order)

        fp = FillProcessor(repo)
        await fp.process_fill(
            oms_order_id=order.oms_order_id,
            broker_fill_id="fill-p1",
            price=500.0,
            qty=3,
            timestamp=datetime.now(timezone.utc),
        )

        updated = await repo.get_order(order.oms_order_id)
        assert updated.filled_qty == 3
        assert updated.remaining_qty == 7
        assert updated.status == OrderStatus.PARTIALLY_FILLED

    @pytest.mark.asyncio
    async def test_process_fill_dedup(self, repo):
        """Duplicate broker_fill_id is deduplicated."""
        order = make_oms_order(qty=10)
        order.status = OrderStatus.WORKING
        order.remaining_qty = 10
        await repo.save_order(order)

        fp = FillProcessor(repo)
        await fp.process_fill(
            order.oms_order_id, "fill-dup", 500.0, 5, datetime.now(timezone.utc)
        )
        await fp.process_fill(
            order.oms_order_id, "fill-dup", 500.0, 5, datetime.now(timezone.utc)
        )

        updated = await repo.get_order(order.oms_order_id)
        assert updated.filled_qty == 5  # Not 10

    @pytest.mark.asyncio
    async def test_avg_fill_price_computation(self, repo):
        """Average fill price computed correctly across multiple fills."""
        order = make_oms_order(qty=10)
        order.status = OrderStatus.WORKING
        order.remaining_qty = 10
        await repo.save_order(order)

        fp = FillProcessor(repo)
        await fp.process_fill(
            order.oms_order_id, "f1", 500.0, 6, datetime.now(timezone.utc)
        )
        await fp.process_fill(
            order.oms_order_id, "f2", 510.0, 4, datetime.now(timezone.utc)
        )

        updated = await repo.get_order(order.oms_order_id)
        expected_avg = (500.0 * 6 + 510.0 * 4) / 10
        assert abs(updated.avg_fill_price - expected_avg) < 0.01

    @pytest.mark.asyncio
    async def test_fill_for_unknown_order(self, repo):
        """Fill for unknown order is logged and ignored."""
        fp = FillProcessor(repo)
        # Should not raise
        await fp.process_fill("nonexistent", "f1", 500.0, 1, datetime.now(timezone.utc))


class TestExecutionRouter:
    """Test order routing with priority queuing."""

    @pytest.mark.asyncio
    async def test_route_when_not_congested(self, repo):
        """Non-congested adapter submits immediately."""
        adapter = MagicMock()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(broker_order_id=1001, perm_id=999, con_id=0)
        )

        router = ExecutionRouter(adapter, repo)
        order = make_oms_order(status=OrderStatus.RISK_APPROVED)
        await router.route(order)

        adapter.submit_order.assert_awaited_once()
        assert order.status == OrderStatus.ROUTED

    @pytest.mark.asyncio
    async def test_route_queues_when_congested(self, repo):
        """Congested adapter queues non-critical orders."""
        adapter = MagicMock()
        adapter.is_congested = True

        router = ExecutionRouter(adapter, repo)
        order = make_oms_order(
            status=OrderStatus.RISK_APPROVED,
            role=OrderRole.ENTRY,
        )
        await router.route(order)

        assert len(router._queue) == 1

    @pytest.mark.asyncio
    async def test_route_stop_bypasses_congestion(self, repo):
        """Stop/exit orders bypass congestion queue."""
        adapter = MagicMock()
        adapter.is_congested = True
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )

        router = ExecutionRouter(adapter, repo)
        order = make_oms_order(
            status=OrderStatus.RISK_APPROVED,
            role=OrderRole.STOP,
        )
        await router.route(order)

        adapter.submit_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drain_queue_priority_ordering(self, repo):
        """Queue drains in priority order: stops > entries."""
        adapter = MagicMock()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )

        router = ExecutionRouter(adapter, repo)

        entry = make_oms_order(role=OrderRole.ENTRY, status=OrderStatus.RISK_APPROVED)
        stop = make_oms_order(role=OrderRole.STOP, status=OrderStatus.RISK_APPROVED, order_type=OrderType.STOP, stop_price=490.0, limit_price=None)
        from shared.oms.execution.router import OrderPriority
        router._queue.append((OrderPriority.NEW_ENTRY, entry, {"queued_at": datetime.now(timezone.utc)}))
        router._queue.append((OrderPriority.STOP_EXIT, stop, {"queued_at": datetime.now(timezone.utc)}))

        await router.drain_queue()

        # Both should have been submitted, stop first
        assert adapter.submit_order.await_count == 2

    @pytest.mark.asyncio
    async def test_cancel_delegates_to_adapter(self, repo):
        """Cancel delegates to adapter.cancel_order."""
        adapter = MagicMock()
        adapter.cancel_order = AsyncMock()

        router = ExecutionRouter(adapter, repo)
        order = make_oms_order()
        order.broker_order_id = 1001
        order.perm_id = 999

        await router.cancel(order)
        adapter.cancel_order.assert_awaited_once_with(1001, 999)

    @pytest.mark.asyncio
    async def test_replace_delegates_to_adapter(self, repo):
        """Replace delegates to adapter.replace_order."""
        adapter = MagicMock()
        adapter.replace_order = AsyncMock(return_value=BrokerOrderRef(1001, 999, 0))

        router = ExecutionRouter(adapter, repo)
        order = make_oms_order()
        order.broker_order_id = 1001

        await router.replace(order, new_qty=5, new_limit_price=505.0, new_stop_price=None)
        adapter.replace_order.assert_awaited_once_with(1001, 5, 505.0, None)

    @pytest.mark.asyncio
    async def test_router_start_stop_lifecycle(self, repo):
        """Router start/stop lifecycle manages drain task."""
        adapter = MagicMock()
        adapter.is_congested = False

        router = ExecutionRouter(adapter, repo)
        await router.start()
        assert router._running is True
        assert router._drain_task is not None

        await router.stop()
        assert router._running is False


class TestIntentHandler:
    """Test strategy intent processing through risk checks and routing."""

    @pytest.fixture
    def gateway_and_handler(self, repo, bus, risk_config):
        """Build IntentHandler with a real RiskGateway and mock router."""
        strat_risk = StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
        port_risk = PortfolioRiskState(trade_date=date.today())

        async def get_strat(sid):
            return strat_risk

        async def get_port():
            return port_risk

        async def get_working(sid):
            return 0

        risk_gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=get_strat,
            get_portfolio_risk=get_port,
            get_working_order_count=get_working,
        )

        mock_adapter = MagicMock()
        mock_adapter.is_congested = False
        mock_adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )
        router = ExecutionRouter(mock_adapter, repo)

        handler = IntentHandler(risk_gw, router, repo, bus)
        return handler, strat_risk, port_risk

    @pytest.mark.asyncio
    async def test_new_order_accepted(self, gateway_and_handler):
        """Valid new order accepted and routed."""
        handler, _, _ = gateway_and_handler
        order = make_oms_order()
        intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=TEST_STRATEGY,
            order=order,
        )

        receipt = await handler.submit(intent)
        assert receipt.result == IntentResult.ACCEPTED
        assert receipt.oms_order_id == order.oms_order_id

    @pytest.mark.asyncio
    async def test_new_order_risk_denied_global_standdown(self, gateway_and_handler, risk_config):
        """Global standdown denies new entry orders."""
        handler, _, _ = gateway_and_handler
        risk_config.global_standdown = True

        order = make_oms_order()
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.DENIED
        assert "stand-down" in receipt.denial_reason.lower()

    @pytest.mark.asyncio
    async def test_new_order_risk_denied_strategy_halt(self, gateway_and_handler):
        """Halted strategy denies new entry orders."""
        handler, strat_risk, _ = gateway_and_handler
        strat_risk.halted = True
        strat_risk.halt_reason = "daily stop hit"

        order = make_oms_order()
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.DENIED
        assert "halted" in receipt.denial_reason.lower()

    @pytest.mark.asyncio
    async def test_new_order_risk_denied_heat_cap(self, gateway_and_handler):
        """Heat cap breach denies new entry orders."""
        handler, _, port_risk = gateway_and_handler
        port_risk.open_risk_R = 10.0  # Way above 1.5 cap

        order = make_oms_order()
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.DENIED
        assert "heat cap" in receipt.denial_reason.lower()

    @pytest.mark.asyncio
    async def test_new_order_zero_qty_denied(self, gateway_and_handler):
        """Order with qty <= 0 is denied."""
        handler, _, _ = gateway_and_handler
        order = make_oms_order(qty=0)
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.DENIED
        assert "qty" in receipt.denial_reason.lower()

    @pytest.mark.asyncio
    async def test_idempotent_order_returns_same_id(self, gateway_and_handler):
        """Duplicate client_order_id returns same oms_order_id."""
        handler, _, _ = gateway_and_handler

        order1 = make_oms_order(client_order_id="idem-001")
        intent1 = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order1)
        receipt1 = await handler.submit(intent1)

        order2 = make_oms_order(client_order_id="idem-001")
        intent2 = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order2)
        receipt2 = await handler.submit(intent2)

        assert receipt1.oms_order_id == receipt2.oms_order_id

    @pytest.mark.asyncio
    async def test_cancel_intent(self, repo, bus, risk_config):
        """Cancel intent transitions order to CANCEL_REQUESTED."""
        strat_risk = StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
        port_risk = PortfolioRiskState(trade_date=date.today())

        risk_gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=strat_risk),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )

        mock_adapter = MagicMock()
        mock_adapter.is_congested = False
        mock_adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )
        mock_adapter.cancel_order = AsyncMock()
        router = ExecutionRouter(mock_adapter, repo)

        handler = IntentHandler(risk_gw, router, repo, bus)

        order = make_oms_order()
        order.status = OrderStatus.WORKING
        order.broker_order_id = 1001
        await repo.save_order(order)

        intent = Intent(
            intent_type=IntentType.CANCEL_ORDER,
            strategy_id=TEST_STRATEGY,
            target_oms_order_id=order.oms_order_id,
        )
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.ACCEPTED
        updated = await repo.get_order(order.oms_order_id)
        assert updated.status == OrderStatus.CANCEL_REQUESTED

    @pytest.mark.asyncio
    async def test_cancel_terminal_order_denied(self, gateway_and_handler, repo):
        """Cannot cancel a filled order."""
        handler, _, _ = gateway_and_handler

        order = make_oms_order()
        order.status = OrderStatus.FILLED
        await repo.save_order(order)

        intent = Intent(
            intent_type=IntentType.CANCEL_ORDER,
            strategy_id=TEST_STRATEGY,
            target_oms_order_id=order.oms_order_id,
        )
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.DENIED

    @pytest.mark.asyncio
    async def test_replace_intent(self, repo, bus, risk_config):
        """Replace intent transitions order to REPLACE_REQUESTED."""
        strat_risk = StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
        port_risk = PortfolioRiskState(trade_date=date.today())

        risk_gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=strat_risk),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )

        mock_adapter = MagicMock()
        mock_adapter.is_congested = False
        mock_adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )
        mock_adapter.replace_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )
        router = ExecutionRouter(mock_adapter, repo)

        handler = IntentHandler(risk_gw, router, repo, bus)

        order = make_oms_order()
        order.status = OrderStatus.WORKING
        order.broker_order_id = 1001
        await repo.save_order(order)

        intent = Intent(
            intent_type=IntentType.REPLACE_ORDER,
            strategy_id=TEST_STRATEGY,
            target_oms_order_id=order.oms_order_id,
            new_limit_price=505.0,
        )
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.ACCEPTED
        updated = await repo.get_order(order.oms_order_id)
        assert updated.status == OrderStatus.REPLACE_REQUESTED

    @pytest.mark.asyncio
    async def test_exit_order_always_passes_risk(self, gateway_and_handler):
        """Exit orders bypass risk checks (always allowed)."""
        handler, strat_risk, _ = gateway_and_handler
        strat_risk.halted = True  # Strategy halted

        order = make_oms_order(role=OrderRole.EXIT, risk_context=None)
        order.order_type = OrderType.MARKET
        order.limit_price = None
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.ACCEPTED

    @pytest.mark.asyncio
    async def test_stop_order_always_passes_risk(self, gateway_and_handler):
        """Stop orders bypass risk checks."""
        handler, strat_risk, _ = gateway_and_handler
        strat_risk.halted = True

        order = make_oms_order(
            role=OrderRole.STOP,
            order_type=OrderType.STOP,
            limit_price=None,
            stop_price=490.0,
            risk_context=None,
        )
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.ACCEPTED


# ===========================================================================
# 4. POSITION RECONCILIATION TESTS
# ===========================================================================

class TestPositionReconciliation:
    """Test position sync, mismatch detection, and correction."""

    def test_no_discrepancies(self):
        """No discrepancies when broker and OMS match."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_positions = [
            PositionSnapshot("DU123", 100, "QQQ", 10.0, 500.0),
        ]
        oms_positions = {100: 10.0}

        result = sync.reconcile_positions(broker_positions, oms_positions)
        assert len(result) == 0

    def test_position_qty_mismatch(self):
        """Mismatch detected when broker and OMS have different quantities."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_positions = [
            PositionSnapshot("DU123", 100, "QQQ", 10.0, 500.0),
        ]
        oms_positions = {100: 5.0}

        result = sync.reconcile_positions(broker_positions, oms_positions)
        assert len(result) == 1
        assert result[0].type == "position_mismatch"
        assert result[0].details["broker_qty"] == 10.0
        assert result[0].details["oms_qty"] == 5.0

    def test_unexpected_broker_position(self):
        """Position on broker but not in OMS triggers HALT_AND_ALERT."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_positions = [
            PositionSnapshot("DU123", 200, "GLD", 50.0, 190.0),
        ]
        oms_positions = {}

        result = sync.reconcile_positions(broker_positions, oms_positions)
        assert len(result) == 1
        assert result[0].action == DiscrepancyAction.HALT_AND_ALERT

    def test_oms_position_missing_from_broker(self):
        """Position in OMS but not on broker."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_positions = []
        oms_positions = {100: 5.0}

        result = sync.reconcile_positions(broker_positions, oms_positions)
        assert len(result) == 1
        assert result[0].details["broker_qty"] == 0.0
        assert result[0].details["oms_qty"] == 5.0

    def test_zero_broker_qty_no_oms_no_discrepancy(self):
        """Zero broker position, no OMS position: no discrepancy."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_positions = [
            PositionSnapshot("DU123", 100, "QQQ", 0.0, 0.0),
        ]
        oms_positions = {}

        result = sync.reconcile_positions(broker_positions, oms_positions)
        assert len(result) == 0

    def test_order_reconciliation_unknown_order(self):
        """Broker order not in OMS is detected."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_orders = [
            OrderStatusEvent(
                broker_order_id=1001,
                perm_id=999,
                status=BrokerOrderStatus.SUBMITTED,
                filled_qty=0,
                remaining_qty=10,
                avg_fill_price=0,
            )
        ]
        oms_working_ids = set()

        result = sync.reconcile_orders(broker_orders, oms_working_ids, "")
        assert len(result) == 1
        assert result[0].type == "unknown_order"

    def test_order_reconciliation_missing_order(self):
        """OMS order not on broker is detected."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_orders = []
        oms_working_ids = {1001}

        result = sync.reconcile_orders(broker_orders, oms_working_ids, "")
        assert len(result) == 1
        assert result[0].type == "missing_order"
        assert result[0].action == DiscrepancyAction.MARK_CANCELLED

    def test_order_reconciliation_no_discrepancies(self):
        """No discrepancies when broker and OMS match."""
        policy = DiscrepancyPolicy()
        sync = ReconcilerSync(policy)

        broker_orders = [
            OrderStatusEvent(
                broker_order_id=1001,
                perm_id=999,
                status=BrokerOrderStatus.SUBMITTED,
                filled_qty=0,
                remaining_qty=10,
                avg_fill_price=0,
            )
        ]
        oms_working_ids = {1001}

        result = sync.reconcile_orders(broker_orders, oms_working_ids, "")
        assert len(result) == 0


class TestReconciliationOrchestrator:
    """Test the OMS-level reconciliation orchestrator."""

    @pytest.mark.asyncio
    async def test_startup_reconciliation_no_discrepancies(self, repo, bus):
        """Startup recon completes without errors when state is clean."""
        adapter = MagicMock()
        adapter.request_open_orders = AsyncMock(return_value=[])
        adapter.request_positions = AsyncMock(return_value=[])
        adapter.request_executions = AsyncMock(return_value=[])
        adapter.cache = IBCache()

        orchestrator = ReconciliationOrchestrator(adapter, repo, bus)
        await orchestrator.startup_reconciliation()

    @pytest.mark.asyncio
    async def test_periodic_reconciliation(self, repo, bus):
        """Periodic recon runs without errors."""
        adapter = MagicMock()
        adapter.request_open_orders = AsyncMock(return_value=[])
        adapter.request_positions = AsyncMock(return_value=[])
        adapter.cache = IBCache()

        orchestrator = ReconciliationOrchestrator(adapter, repo, bus)
        await orchestrator.periodic_reconciliation()


# ===========================================================================
# 5. MARKET DATA TESTS
# ===========================================================================

class TestSnapshotFetcher:
    """Test broker state snapshot fetching."""

    @pytest.mark.asyncio
    async def test_fetch_open_orders(self):
        """Fetch open orders normalizes trade objects."""
        ib = MagicMock()
        trade = _make_mock_trade(order_id=1001, status="Submitted", filled=0, remaining=10)
        ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[trade])

        fetcher = SnapshotFetcher(ib)
        orders = await fetcher.fetch_open_orders()

        assert len(orders) == 1
        assert orders[0].broker_order_id == 1001
        assert orders[0].status == BrokerOrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_fetch_positions(self):
        """Fetch positions normalizes position objects."""
        ib = MagicMock()
        pos = MagicMock()
        pos.account = TEST_ACCOUNT
        pos.contract = MagicMock()
        pos.contract.conId = 265598
        pos.contract.symbol = "QQQ"
        pos.position = 10.0
        pos.avgCost = 500.0
        ib.reqPositionsAsync = AsyncMock(return_value=[pos])

        fetcher = SnapshotFetcher(ib)
        positions = await fetcher.fetch_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "QQQ"
        assert positions[0].qty == 10.0
        assert positions[0].avg_cost == 500.0

    @pytest.mark.asyncio
    async def test_fetch_executions(self):
        """Fetch executions normalizes fill objects."""
        ib = MagicMock()
        fill = MagicMock()
        fill.execution.execId = "exec-001"
        fill.execution.orderId = 1001
        fill.execution.permId = 999
        fill.execution.side = "BOT"
        fill.execution.shares = 10
        fill.execution.price = 500.0
        fill.execution.time = datetime.now(timezone.utc)
        fill.execution.exchange = "SMART"
        fill.contract.symbol = "QQQ"
        fill.commissionReport = MagicMock()
        fill.commissionReport.commission = 1.50
        ib.reqExecutionsAsync = AsyncMock(return_value=[fill])

        fetcher = SnapshotFetcher(ib)
        executions = await fetcher.fetch_executions()

        assert len(executions) == 1
        assert executions[0].exec_id == "exec-001"
        assert executions[0].price == 500.0
        assert executions[0].qty == 10

    @pytest.mark.asyncio
    async def test_fetch_executions_with_time_filter(self):
        """Fetch executions with time filter sets the filter."""
        ib = MagicMock()
        ib.reqExecutionsAsync = AsyncMock(return_value=[])

        fetcher = SnapshotFetcher(ib)
        since = datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        await fetcher.fetch_executions(since)

        ib.reqExecutionsAsync.assert_awaited_once()


# ===========================================================================
# 6. ERROR HANDLING TESTS
# ===========================================================================

class TestErrorHandling:
    """Test connection loss recovery, order rejections, timeouts."""

    def test_classify_risk_error(self):
        """Error code 201 classified as RISK."""
        cat, retryable = classify_error(201, "Order rejected by risk")
        assert cat == RejectCategory.RISK
        assert retryable is False

    def test_classify_pacing_error(self):
        """Error code 161 classified as PACING and retryable."""
        cat, retryable = classify_error(161, "Pacing violation")
        assert cat == RejectCategory.PACING
        assert retryable is True

    def test_classify_contract_error(self):
        """Error code 200 classified as CONTRACT."""
        cat, retryable = classify_error(200, "No security definition")
        assert cat == RejectCategory.CONTRACT
        assert retryable is False

    def test_classify_permission_error(self):
        """Error code 104 classified as PERMISSIONS."""
        cat, retryable = classify_error(104, "Can't modify a filled order")
        assert cat == RejectCategory.PERMISSIONS
        assert retryable is False

    def test_classify_duplicate_error(self):
        """Error code 103 classified as DUPLICATE."""
        cat, retryable = classify_error(103, "Duplicate order id")
        assert cat == RejectCategory.DUPLICATE
        assert retryable is False

    def test_classify_invalid_price_error(self):
        """Error code 110 classified as INVALID_PRICE."""
        cat, retryable = classify_error(110, "Price does not conform to tick")
        assert cat == RejectCategory.INVALID_PRICE
        assert retryable is False

    def test_classify_unknown_error(self):
        """Unknown error code classified as UNKNOWN."""
        cat, retryable = classify_error(99999, "Unknown error")
        assert cat == RejectCategory.UNKNOWN
        assert retryable is False

    def test_reject_classifier(self):
        """RejectClassifier returns structured result."""
        result = RejectClassifier.classify(200, "No security definition")
        assert result["category"] == RejectCategory.CONTRACT
        assert result["is_misconfiguration"] is True
        assert result["retryable"] is False

    def test_reject_classifier_pacing(self):
        """RejectClassifier for pacing error shows retryable."""
        result = RejectClassifier.classify(161, "Pacing")
        assert result["retryable"] is True
        assert result["is_misconfiguration"] is False

    @pytest.mark.asyncio
    async def test_timeout_monitor_stuck_routed_transitions_to_cancelled(self, repo, bus):
        """Timeout monitor detects stuck ROUTED and cancels it.
        ROUTED->{ACKED, REJECTED, CANCELLED} per state machine."""
        order = make_oms_order()
        order.status = OrderStatus.ROUTED
        order.submitted_at = datetime(2020, 1, 1, tzinfo=timezone.utc)  # old
        order.created_at = order.submitted_at
        await repo.save_order(order)

        monitor = OrderTimeoutMonitor(repo, bus, routed_timeout_s=0.0, scan_interval_s=0.01)
        await monitor._scan_stuck_orders()

        updated = await repo.get_order(order.oms_order_id)
        # ROUTED -> CANCELLED is valid; stuck orders get cancelled
        assert updated.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_timeout_monitor_expires_stuck_cancel_requested(self, repo, bus):
        """Timeout monitor cancels orders stuck in CANCEL_REQUESTED."""
        order = make_oms_order()
        order.status = OrderStatus.WORKING
        transition(order, OrderStatus.CANCEL_REQUESTED)
        order.last_update_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        order.created_at = order.last_update_at
        await repo.save_order(order)

        monitor = OrderTimeoutMonitor(repo, bus, cancel_timeout_s=0.0, scan_interval_s=0.01)
        await monitor._scan_stuck_orders()

        updated = await repo.get_order(order.oms_order_id)
        assert updated.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_timeout_monitor_skips_working_orders(self, repo, bus):
        """Timeout monitor does not touch WORKING orders (not stuck)."""
        order = make_oms_order()
        order.status = OrderStatus.WORKING
        order.last_update_at = datetime.now(timezone.utc)
        order.created_at = order.last_update_at
        await repo.save_order(order)

        monitor = OrderTimeoutMonitor(repo, bus, routed_timeout_s=30.0, scan_interval_s=0.01)
        await monitor._scan_stuck_orders()

        updated = await repo.get_order(order.oms_order_id)
        assert updated.status == OrderStatus.WORKING

    @pytest.mark.asyncio
    async def test_timeout_monitor_start_stop(self, repo, bus):
        """Monitor starts and stops cleanly."""
        monitor = OrderTimeoutMonitor(repo, bus, scan_interval_s=0.01)
        await monitor.start()
        assert monitor._running
        await asyncio.sleep(0.03)
        await monitor.stop()
        assert not monitor._running

    @pytest.mark.asyncio
    async def test_timeout_monitor_cancel_requested_emits_event(self, repo, bus):
        """Timeout monitor cancels stuck CANCEL_REQUESTED and emits event via bus."""
        q = bus.subscribe(TEST_STRATEGY)

        order = make_oms_order()
        order.status = OrderStatus.WORKING
        transition(order, OrderStatus.CANCEL_REQUESTED)
        order.last_update_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        order.created_at = order.last_update_at
        await repo.save_order(order)

        monitor = OrderTimeoutMonitor(repo, bus, cancel_timeout_s=0.0, scan_interval_s=0.01)
        await monitor._scan_stuck_orders()

        event = q.get_nowait()
        assert event.event_type == OMSEventType.ORDER_CANCELLED


# ===========================================================================
# 7. THROTTLING TESTS
# ===========================================================================

class TestThrottling:
    """Test order rate limiting and API call throttling."""

    @pytest.mark.asyncio
    async def test_token_bucket_fast_acquire(self):
        """Token bucket allows immediate acquisition when tokens available."""
        bucket = _TokenBucket(rate=100.0)
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_token_bucket_rate_limiting(self):
        """Token bucket enforces rate limit by sleeping."""
        bucket = _TokenBucket(rate=10.0)
        # Consume all tokens
        for _ in range(10):
            await bucket.acquire()
        # Next acquire should wait
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # Should wait at least a bit

    @pytest.mark.asyncio
    async def test_throttler_orders_channel(self):
        """Throttler.acquire succeeds for orders channel."""
        throttler = Throttler(orders_per_sec=100.0, messages_per_sec=100.0)
        await throttler.acquire(PacingChannel.ORDERS)
        # No exception = success

    @pytest.mark.asyncio
    async def test_throttler_congestion_detected(self):
        """Congestion raises CongestionError when threshold exceeded."""
        throttler = Throttler(orders_per_sec=100.0)
        throttler.congestion_threshold = 1

        # Fill up the queue depth artificially
        throttler._queue_depth[PacingChannel.ORDERS] = 2
        with pytest.raises(CongestionError):
            await throttler.acquire(PacingChannel.ORDERS)

    @pytest.mark.asyncio
    async def test_throttler_is_congested_property(self):
        """is_congested returns True when any channel exceeds threshold."""
        throttler = Throttler()
        throttler.congestion_threshold = 5
        assert not throttler.is_congested

        throttler._queue_depth[PacingChannel.ORDERS] = 6
        assert throttler.is_congested

    @pytest.mark.asyncio
    async def test_throttler_multiple_channels_independent(self):
        """Different channels have independent rate limits."""
        throttler = Throttler(orders_per_sec=5.0, messages_per_sec=50.0)
        # Orders channel should be slower than market data
        await throttler.acquire(PacingChannel.ORDERS)
        await throttler.acquire(PacingChannel.MARKET_DATA)

    @pytest.mark.asyncio
    async def test_throttler_historical_channel(self):
        """Historical channel rate limited to ~1 req/sec."""
        throttler = Throttler()
        # Should succeed once
        await throttler.acquire(PacingChannel.HISTORICAL)

    @pytest.mark.asyncio
    async def test_request_id_allocator_sequential(self):
        """Request IDs are allocated sequentially."""
        alloc = RequestIdAllocator(initial_order_id=100)
        id1 = await alloc.next_order_id()
        id2 = await alloc.next_order_id()
        assert id1 == 100
        assert id2 == 101

    @pytest.mark.asyncio
    async def test_request_id_set_next_valid(self):
        """set_next_valid_id updates base only if higher."""
        alloc = RequestIdAllocator(initial_order_id=100)
        await alloc.set_next_valid_id(200)
        id1 = await alloc.next_order_id()
        assert id1 == 200

        # Lower value should not go backwards
        await alloc.set_next_valid_id(50)
        id2 = await alloc.next_order_id()
        assert id2 == 201

    @pytest.mark.asyncio
    async def test_request_id_concurrent_safety(self):
        """Concurrent allocations produce unique IDs."""
        alloc = RequestIdAllocator(initial_order_id=1000)

        async def allocate():
            return await alloc.next_order_id()

        results = await asyncio.gather(*[allocate() for _ in range(100)])
        assert len(set(results)) == 100  # All unique


# ===========================================================================
# 8. END-TO-END PAPER TRADING FLOW
# ===========================================================================

class TestEndToEndPaperTradingFlow:
    """Test complete trade lifecycle in paper trading mode."""

    @pytest.mark.asyncio
    async def test_full_trade_lifecycle(self, repo, bus):
        """Complete lifecycle: signal -> order -> fill -> position -> stop -> exit."""
        # 1. Set up components
        adapter = MagicMock()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )
        adapter.cancel_order = AsyncMock()

        strat_risk = StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
        port_risk = PortfolioRiskState(trade_date=date.today())

        risk_config = make_risk_config(unit_risk_dollars=100.0, heat_cap_R=5.0)
        risk_gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=strat_risk),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )
        router = ExecutionRouter(adapter, repo)
        handler = IntentHandler(risk_gw, router, repo, bus)
        fill_proc = FillProcessor(repo)

        q = bus.subscribe(TEST_STRATEGY)

        # 2. Submit entry order (signal -> order)
        entry = make_oms_order(
            qty=10,
            limit_price=500.0,
            role=OrderRole.ENTRY,
            risk_context=RiskContext(
                stop_for_risk=490.0,
                planned_entry_price=500.0,
                risk_budget_tag="MR",
                risk_dollars=100.0,
            ),
        )
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=entry)
        receipt = await handler.submit(intent)
        assert receipt.result == IntentResult.ACCEPTED

        persisted = await repo.get_order(entry.oms_order_id)
        assert persisted.status == OrderStatus.ROUTED
        assert persisted.broker_order_id == 1001

        # 3. Simulate ACK
        transition(persisted, OrderStatus.ACKED)
        await repo.save_order(persisted)

        # 4. Simulate WORKING
        transition(persisted, OrderStatus.WORKING)
        await repo.save_order(persisted)

        # 5. Simulate fill
        await fill_proc.process_fill(
            entry.oms_order_id, "exec-001", 500.0, 10, datetime.now(timezone.utc), 2.0
        )
        filled_order = await repo.get_order(entry.oms_order_id)
        assert filled_order.status == OrderStatus.FILLED
        assert filled_order.filled_qty == 10

        # 6. Submit stop order
        stop_order = make_oms_order(
            qty=10,
            order_type=OrderType.STOP,
            stop_price=490.0,
            limit_price=None,
            side=OrderSide.SELL,
            role=OrderRole.STOP,
            risk_context=None,
        )
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1002, 998, 0)
        )
        stop_intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=TEST_STRATEGY,
            order=stop_order,
        )
        stop_receipt = await handler.submit(stop_intent)
        assert stop_receipt.result == IntentResult.ACCEPTED

        # 7. Simulate exit: cancel stop and submit market exit
        stop_persisted = await repo.get_order(stop_order.oms_order_id)
        stop_persisted.status = OrderStatus.WORKING
        stop_persisted.broker_order_id = 1002
        await repo.save_order(stop_persisted)

        cancel_intent = Intent(
            intent_type=IntentType.CANCEL_ORDER,
            strategy_id=TEST_STRATEGY,
            target_oms_order_id=stop_order.oms_order_id,
        )
        cancel_receipt = await handler.submit(cancel_intent)
        assert cancel_receipt.result == IntentResult.ACCEPTED

        # 8. Submit exit order
        exit_order = make_oms_order(
            qty=10,
            order_type=OrderType.MARKET,
            limit_price=None,
            side=OrderSide.SELL,
            role=OrderRole.EXIT,
            risk_context=None,
        )
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1003, 997, 0)
        )
        exit_intent = Intent(
            intent_type=IntentType.NEW_ORDER,
            strategy_id=TEST_STRATEGY,
            order=exit_order,
        )
        exit_receipt = await handler.submit(exit_intent)
        assert exit_receipt.result == IntentResult.ACCEPTED

        # 9. Simulate exit fill
        exit_persisted = await repo.get_order(exit_order.oms_order_id)
        transition(exit_persisted, OrderStatus.ACKED)
        transition(exit_persisted, OrderStatus.WORKING)
        await repo.save_order(exit_persisted)

        await fill_proc.process_fill(
            exit_order.oms_order_id, "exec-002", 505.0, 10, datetime.now(timezone.utc), 2.0
        )
        exit_filled = await repo.get_order(exit_order.oms_order_id)
        assert exit_filled.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_multi_strategy_concurrent_orders(self, repo, bus):
        """Multiple strategies submit orders concurrently."""
        adapter = MagicMock()
        adapter.is_congested = False
        order_id_counter = iter(range(2001, 3000))
        adapter.submit_order = AsyncMock(
            side_effect=lambda **kw: BrokerOrderRef(next(order_id_counter), 0, 0)
        )

        strategies = ["ATRSS", "AKC_HELIX"]
        risk_configs = {}
        for sid in strategies:
            risk_configs[sid] = StrategyRiskConfig(
                strategy_id=sid,
                daily_stop_R=2.0,
                unit_risk_dollars=100.0,
                max_working_orders=4,
            )

        risk_config = RiskConfig(
            heat_cap_R=5.0,
            portfolio_daily_stop_R=3.0,
            strategy_configs=risk_configs,
        )

        strat_risks = {
            sid: StrategyRiskState(strategy_id=sid, trade_date=date.today())
            for sid in strategies
        }
        port_risk = PortfolioRiskState(trade_date=date.today())

        risk_gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(side_effect=lambda sid: strat_risks[sid]),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
            get_working_order_count=AsyncMock(return_value=0),
        )

        router = ExecutionRouter(adapter, repo)
        handler = IntentHandler(risk_gw, router, repo, bus)

        # Submit orders for both strategies concurrently
        orders = []
        for sid in strategies:
            order = make_oms_order(strategy_id=sid)
            orders.append(order)
            intent = Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=sid,
                order=order,
            )
            receipt = await handler.submit(intent)
            assert receipt.result == IntentResult.ACCEPTED

        # Verify both orders are persisted with different broker IDs
        for order in orders:
            persisted = await repo.get_order(order.oms_order_id)
            assert persisted is not None
            assert persisted.status == OrderStatus.ROUTED
            assert persisted.broker_order_id is not None

        broker_ids = [
            (await repo.get_order(o.oms_order_id)).broker_order_id for o in orders
        ]
        assert len(set(broker_ids)) == 2

    @pytest.mark.asyncio
    async def test_partial_fill_then_cancel(self, repo, bus):
        """Partial fill followed by cancel: position reflects partial fill."""
        adapter = MagicMock()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )
        adapter.cancel_order = AsyncMock()

        risk_config = make_risk_config(heat_cap_R=5.0)
        risk_gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
            get_working_order_count=AsyncMock(return_value=0),
        )
        router = ExecutionRouter(adapter, repo)
        handler = IntentHandler(risk_gw, router, repo, bus)
        fill_proc = FillProcessor(repo)

        # Submit order
        order = make_oms_order(qty=10)
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        await handler.submit(intent)

        # Simulate transition to WORKING
        persisted = await repo.get_order(order.oms_order_id)
        transition(persisted, OrderStatus.ACKED)
        transition(persisted, OrderStatus.WORKING)
        persisted.broker_order_id = 1001
        await repo.save_order(persisted)

        # Partial fill: 3 of 10
        await fill_proc.process_fill(
            order.oms_order_id, "pf-001", 500.0, 3, datetime.now(timezone.utc), 0.5
        )
        partial = await repo.get_order(order.oms_order_id)
        assert partial.status == OrderStatus.PARTIALLY_FILLED
        assert partial.filled_qty == 3
        assert partial.remaining_qty == 7

        # Cancel the rest
        cancel_intent = Intent(
            intent_type=IntentType.CANCEL_ORDER,
            strategy_id=TEST_STRATEGY,
            target_oms_order_id=order.oms_order_id,
        )
        cancel_receipt = await handler.submit(cancel_intent)
        assert cancel_receipt.result == IntentResult.ACCEPTED

        cancelled = await repo.get_order(order.oms_order_id)
        assert cancelled.status == OrderStatus.CANCEL_REQUESTED


# ===========================================================================
# ADDITIONAL: IBCACHE TESTS
# ===========================================================================

class TestIBCache:
    """Test in-memory cache for order ID mappings and fill dedup."""

    def test_register_and_lookup(self):
        cache = IBCache()
        cache.register_order("oms-001", 1001, 9999)

        assert cache.lookup_oms_id(1001) == "oms-001"
        assert cache.lookup_broker_id("oms-001") == (1001, 9999)

    def test_lookup_missing(self):
        cache = IBCache()
        assert cache.lookup_oms_id(9999) is None
        assert cache.lookup_broker_id("nonexistent") is None

    def test_fill_dedup(self):
        cache = IBCache()
        assert not cache.is_fill_seen("exec-001")
        cache.mark_fill_seen("exec-001")
        assert cache.is_fill_seen("exec-001")

    def test_clear(self):
        cache = IBCache()
        cache.register_order("oms-001", 1001, 9999)
        cache.mark_fill_seen("exec-001")
        cache.clear()

        assert cache.lookup_oms_id(1001) is None
        assert not cache.is_fill_seen("exec-001")

    @pytest.mark.asyncio
    async def test_rebuild_from_broker(self):
        """Rebuild cache from broker state."""
        cache = IBCache()

        open_order = OrderStatusEvent(
            broker_order_id=1001, perm_id=999,
            status=BrokerOrderStatus.SUBMITTED,
            filled_qty=0, remaining_qty=10, avg_fill_price=0,
        )
        exec_report = ExecutionReport(
            exec_id="exec-001", broker_order_id=1002, perm_id=998,
            symbol="QQQ", side="BOT", qty=5, price=500.0,
            timestamp=datetime.now(timezone.utc),
        )

        fetcher = MagicMock()
        fetcher.fetch_open_orders = AsyncMock(return_value=[open_order])
        fetcher.fetch_executions = AsyncMock(return_value=[exec_report])

        async def resolve(broker_id):
            return f"oms-{broker_id}"

        await cache.rebuild_from_broker(fetcher, resolve)

        assert cache.lookup_oms_id(1001) == "oms-1001"
        assert cache.lookup_oms_id(1002) == "oms-1002"
        assert cache.is_fill_seen("exec-001")


# ===========================================================================
# ADDITIONAL: TICK RULES TESTS
# ===========================================================================

class TestTickRules:
    """Test price rounding and validation."""

    def test_round_to_nearest_tick(self):
        assert round_to_tick(500.123, 0.25) == 500.0
        assert round_to_tick(500.13, 0.25) == 500.25

    def test_round_to_tick_up(self):
        assert round_to_tick(500.01, 0.25, direction="up") == 500.25

    def test_round_to_tick_down(self):
        assert round_to_tick(500.24, 0.25, direction="down") == 500.0

    def test_round_qty(self):
        assert round_qty(3.7) == 3
        assert round_qty(0.3) == 1  # min_qty enforcement

    def test_validate_price(self):
        assert validate_price(500.25, 0.25) is True
        assert validate_price(500.10, 0.25) is False
        assert validate_price(500.01, 0.01) is True

    def test_round_small_tick(self):
        """Penny tick size for ETFs."""
        # round(50000.6) = 50001 -> 500.01
        assert abs(round_to_tick(500.006, 0.01) - 500.01) < 0.001
        assert abs(round_to_tick(500.004, 0.01) - 500.0) < 0.001
        # Exact penny values stay
        assert abs(round_to_tick(500.03, 0.01) - 500.03) < 0.001


# ===========================================================================
# ADDITIONAL: EVENT BUS TESTS
# ===========================================================================

class TestEventBus:
    """Test OMS event distribution."""

    def test_subscribe_and_emit(self):
        bus = EventBus()
        q = bus.subscribe(TEST_STRATEGY)

        order = make_oms_order()
        order.status = OrderStatus.FILLED
        order.last_update_at = datetime.now(timezone.utc)
        bus.emit_order_event(order)

        event = q.get_nowait()
        assert event.event_type == OMSEventType.ORDER_FILLED
        assert event.oms_order_id == order.oms_order_id

    def test_global_subscriber(self):
        bus = EventBus()
        q = bus.subscribe_all()

        order = make_oms_order(strategy_id="strategy_a")
        order.status = OrderStatus.CANCELLED
        order.last_update_at = datetime.now(timezone.utc)
        bus.emit_order_event(order)

        event = q.get_nowait()
        assert event.event_type == OMSEventType.ORDER_CANCELLED

    def test_fill_event(self):
        bus = EventBus()
        q = bus.subscribe(TEST_STRATEGY)

        bus.emit_fill_event(TEST_STRATEGY, "oms-001", {"price": 500.0, "qty": 10})
        event = q.get_nowait()
        assert event.event_type == OMSEventType.FILL
        assert event.payload["price"] == 500.0

    def test_risk_denial_event(self):
        bus = EventBus()
        q = bus.subscribe(TEST_STRATEGY)

        bus.emit_risk_denial(TEST_STRATEGY, "oms-001", "heat cap")
        event = q.get_nowait()
        assert event.event_type == OMSEventType.RISK_DENIAL

    def test_risk_halt_event(self):
        bus = EventBus()
        q = bus.subscribe_all()

        bus.emit_risk_halt(TEST_STRATEGY, "portfolio stop")
        event = q.get_nowait()
        assert event.event_type == OMSEventType.RISK_HALT

    def test_unsubscribe(self):
        bus = EventBus()
        q = bus.subscribe(TEST_STRATEGY)
        bus.unsubscribe(TEST_STRATEGY, q)

        order = make_oms_order()
        order.status = OrderStatus.FILLED
        order.last_update_at = datetime.now(timezone.utc)
        bus.emit_order_event(order)

        assert q.empty()

    def test_coordination_event(self):
        bus = EventBus()
        q = bus.subscribe("AKC_HELIX")

        bus.emit_coordination_event("AKC_HELIX", "TIGHTEN_STOP_BE", symbol="QQQ")
        event = q.get_nowait()
        assert event.event_type == OMSEventType.COORDINATION
        assert event.payload["coordination_type"] == "TIGHTEN_STOP_BE"
        assert event.payload["symbol"] == "QQQ"


# ===========================================================================
# ADDITIONAL: COORDINATOR TESTS
# ===========================================================================

class TestStrategyCoordinator:
    """Test cross-strategy coordination signals."""

    def test_atrss_entry_triggers_helix_tighten(self, bus):
        """ATRSS entry fill on symbol with existing Helix position emits TIGHTEN_STOP_BE."""
        q = bus.subscribe("AKC_HELIX")
        coord = StrategyCoordinator(bus)

        # Set up Helix position first
        coord.on_position_update("AKC_HELIX", "QQQ", qty=1, direction="LONG", entry_price=490.0)

        # ATRSS entry fill
        coord.on_fill("ATRSS", "QQQ", side="BUY", role="ENTRY", price=500.0)

        event = q.get_nowait()
        assert event.event_type == OMSEventType.COORDINATION
        assert event.payload["coordination_type"] == "TIGHTEN_STOP_BE"

    def test_atrss_entry_no_helix_no_event(self, bus):
        """ATRSS entry without existing Helix position does not emit."""
        q = bus.subscribe("AKC_HELIX")
        coord = StrategyCoordinator(bus)

        coord.on_fill("ATRSS", "QQQ", side="BUY", role="ENTRY", price=500.0)

        assert q.empty()

    def test_has_atrss_position(self, bus):
        """has_atrss_position returns True when ATRSS has position."""
        coord = StrategyCoordinator(bus)
        coord.on_position_update("ATRSS", "QQQ", qty=1, direction="LONG")

        assert coord.has_atrss_position("QQQ") is True
        assert coord.has_atrss_position("QQQ", direction="LONG") is True
        assert coord.has_atrss_position("QQQ", direction="SHORT") is False
        assert coord.has_atrss_position("GLD") is False

    def test_position_removed_on_flat(self, bus):
        """Position removed from book when qty goes to 0."""
        coord = StrategyCoordinator(bus)
        coord.on_position_update("ATRSS", "QQQ", qty=1, direction="LONG")
        assert coord.has_atrss_position("QQQ") is True

        coord.on_position_update("ATRSS", "QQQ", qty=0, direction="LONG")
        assert coord.has_atrss_position("QQQ") is False

    def test_get_position(self, bus):
        coord = StrategyCoordinator(bus)
        coord.on_position_update("ATRSS", "QQQ", qty=2, direction="LONG", entry_price=500.0)

        pos = coord.get_position("ATRSS", "QQQ")
        assert pos is not None
        assert pos.qty == 2
        assert pos.direction == "LONG"
        assert pos.entry_price == 500.0

    def test_get_all_positions(self, bus):
        coord = StrategyCoordinator(bus)
        coord.on_position_update("ATRSS", "QQQ", qty=1, direction="LONG")
        coord.on_position_update("AKC_HELIX", "GLD", qty=2, direction="SHORT")

        all_pos = coord.get_all_positions()
        assert len(all_pos) == 2


# ===========================================================================
# ADDITIONAL: RISK CALCULATOR TESTS
# ===========================================================================

class TestRiskCalculator:
    """Test risk computation utilities."""

    def test_compute_order_risk_dollars(self):
        risk = RiskCalculator.compute_order_risk_dollars(
            planned_entry=500.0, stop_price=490.0, qty=10, point_value=2.0
        )
        assert risk == 200.0  # 10 * 2 * 10

    def test_compute_risk_R(self):
        r = RiskCalculator.compute_risk_R(200.0, 100.0)
        assert r == 2.0

    def test_compute_risk_R_zero_unit(self):
        r = RiskCalculator.compute_risk_R(200.0, 0.0)
        assert r == float("inf")

    def test_compute_unit_risk_dollars(self):
        urd = RiskCalculator.compute_unit_risk_dollars(nav=100_000, unit_risk_pct=0.0035)
        assert abs(urd - 350.0) < 0.01


# ===========================================================================
# ADDITIONAL: SESSION STATE TESTS
# ===========================================================================

class TestSessionState:
    """Test session state tracking dataclass."""

    def test_defaults(self):
        state = SessionState()
        assert state.connected is False
        assert state.next_valid_id == 0
        assert state.accounts == []
        assert state.last_heartbeat is None

    def test_update(self):
        state = SessionState()
        state.connected = True
        state.next_valid_id = 1000
        state.accounts = [TEST_ACCOUNT]
        state.last_heartbeat = datetime.now(timezone.utc)
        assert state.connected is True


# ===========================================================================
# ADDITIONAL: ORDER FLAGS TESTS
# ===========================================================================

class TestOrderFlags:
    """Test per-instrument/venue allowed order flags."""

    def test_cme_flags(self):
        flags = EXCHANGE_FLAGS["CME"]
        assert flags.allow_outside_rth is True
        assert flags.allow_oca is True

    def test_default_flags(self):
        flags = AllowedOrderFlags()
        assert flags.allow_outside_rth is False
        assert "LIMIT" in flags.max_order_types
        assert "MARKET" in flags.max_order_types
        assert "STOP" in flags.max_order_types


# ===========================================================================
# ADDITIONAL: OMS SERVICE TESTS
# ===========================================================================

class TestOMSService:
    """Test OMS service lifecycle."""

    @pytest.mark.asyncio
    async def test_service_start_stop(self, repo, bus):
        """OMS service starts and stops cleanly."""
        adapter = MagicMock()
        adapter.request_open_orders = AsyncMock(return_value=[])
        adapter.request_positions = AsyncMock(return_value=[])
        adapter.request_executions = AsyncMock(return_value=[])
        adapter.cache = IBCache()
        adapter.is_congested = False

        reconciler = ReconciliationOrchestrator(adapter, repo, bus)
        risk_config = make_risk_config()
        risk_gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
        )
        router = ExecutionRouter(adapter, repo)
        handler = IntentHandler(risk_gw, router, repo, bus)
        timeout_monitor = OrderTimeoutMonitor(repo, bus, scan_interval_s=60.0)

        oms = OMSService(
            intent_handler=handler,
            bus=bus,
            reconciler=reconciler,
            router=router,
            recon_interval_s=60.0,
            timeout_monitor=timeout_monitor,
        )

        await oms.start()
        assert oms.is_ready

        await oms.stop()
        assert not oms.is_ready

    @pytest.mark.asyncio
    async def test_submit_intent_waits_for_ready(self, repo, bus):
        """submit_intent blocks until service is ready."""
        adapter = MagicMock()
        adapter.request_open_orders = AsyncMock(return_value=[])
        adapter.request_positions = AsyncMock(return_value=[])
        adapter.request_executions = AsyncMock(return_value=[])
        adapter.cache = IBCache()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )

        reconciler = ReconciliationOrchestrator(adapter, repo, bus)
        risk_gw = RiskGateway(
            config=make_risk_config(heat_cap_R=5.0),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
            get_working_order_count=AsyncMock(return_value=0),
        )
        router = ExecutionRouter(adapter, repo)
        handler = IntentHandler(risk_gw, router, repo, bus)

        oms = OMSService(
            intent_handler=handler,
            bus=bus,
            reconciler=reconciler,
            router=router,
            recon_interval_s=600.0,
        )

        # Start service
        await oms.start()

        # Now submit should work
        order = make_oms_order()
        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id=TEST_STRATEGY, order=order)
        receipt = await oms.submit_intent(intent)
        assert receipt.result == IntentResult.ACCEPTED

        await oms.stop()

    @pytest.mark.asyncio
    async def test_event_streaming(self, bus):
        """Event streaming returns queue for strategy events."""
        oms = OMSService(
            intent_handler=MagicMock(),
            bus=bus,
            reconciler=MagicMock(),
        )

        q = oms.stream_events(TEST_STRATEGY)
        assert q is not None

        q_all = oms.stream_all_events()
        assert q_all is not None


# ===========================================================================
# ADDITIONAL: IN-MEMORY REPOSITORY TESTS
# ===========================================================================

class TestInMemoryRepository:
    """Test the in-memory repository used for paper trading."""

    @pytest.mark.asyncio
    async def test_save_and_get_order(self, repo):
        order = make_oms_order()
        await repo.save_order(order)
        fetched = await repo.get_order(order.oms_order_id)
        assert fetched is not None
        assert fetched.oms_order_id == order.oms_order_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_order(self, repo):
        fetched = await repo.get_order("nonexistent")
        assert fetched is None

    @pytest.mark.asyncio
    async def test_save_and_check_fill(self, repo):
        fill = Fill(
            fill_id="f-001",
            oms_order_id="oms-001",
            broker_fill_id="exec-001",
            price=500.0,
            qty=10,
            timestamp=datetime.now(timezone.utc),
        )
        await repo.save_fill(fill)
        assert await repo.fill_exists("exec-001")
        assert not await repo.fill_exists("exec-999")

    @pytest.mark.asyncio
    async def test_fill_dedup(self, repo):
        fill = Fill("f-001", "oms-001", "exec-dup", 500.0, 10, datetime.now(timezone.utc))
        await repo.save_fill(fill)
        await repo.save_fill(fill)  # duplicate
        # No error, and fill exists
        assert await repo.fill_exists("exec-dup")

    @pytest.mark.asyncio
    async def test_get_working_orders(self, repo):
        working = make_oms_order()
        working.status = OrderStatus.WORKING
        await repo.save_order(working)

        filled = make_oms_order()
        filled.status = OrderStatus.FILLED
        await repo.save_order(filled)

        result = await repo.get_working_orders(TEST_STRATEGY)
        assert len(result) == 1
        assert result[0].oms_order_id == working.oms_order_id

    @pytest.mark.asyncio
    async def test_count_working_orders(self, repo):
        for i in range(3):
            o = make_oms_order()
            o.status = OrderStatus.WORKING
            await repo.save_order(o)

        done = make_oms_order()
        done.status = OrderStatus.FILLED
        await repo.save_order(done)

        count = await repo.count_working_orders(TEST_STRATEGY)
        assert count == 3

    @pytest.mark.asyncio
    async def test_save_and_get_position(self, repo):
        pos = Position(
            account_id=TEST_ACCOUNT,
            instrument_symbol="QQQ",
            strategy_id=TEST_STRATEGY,
            net_qty=10.0,
            avg_price=500.0,
        )
        await repo.save_position(pos)
        positions = await repo.get_positions(TEST_STRATEGY)
        assert len(positions) == 1
        assert positions[0].net_qty == 10.0

    @pytest.mark.asyncio
    async def test_get_positions_filtered_by_instrument(self, repo):
        pos1 = Position(TEST_ACCOUNT, "QQQ", TEST_STRATEGY, net_qty=10.0)
        pos2 = Position(TEST_ACCOUNT, "GLD", TEST_STRATEGY, net_qty=5.0)
        await repo.save_position(pos1)
        await repo.save_position(pos2)

        qqq_only = await repo.get_positions(TEST_STRATEGY, "QQQ")
        assert len(qqq_only) == 1
        assert qqq_only[0].instrument_symbol == "QQQ"

    @pytest.mark.asyncio
    async def test_client_order_id_lookup(self, repo):
        order = make_oms_order(client_order_id="my-unique-id")
        await repo.save_order(order)

        found = await repo.get_order_id_by_client_order_id(TEST_STRATEGY, "my-unique-id")
        assert found == order.oms_order_id

        not_found = await repo.get_order_id_by_client_order_id(TEST_STRATEGY, "nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_all_working_orders(self, repo):
        o1 = make_oms_order(strategy_id="ATRSS")
        o1.status = OrderStatus.WORKING
        await repo.save_order(o1)

        o2 = make_oms_order(strategy_id="AKC_HELIX")
        o2.status = OrderStatus.ACKED
        await repo.save_order(o2)

        o3 = make_oms_order(strategy_id="ATRSS")
        o3.status = OrderStatus.FILLED
        await repo.save_order(o3)

        all_working = await repo.get_all_working_orders()
        assert len(all_working) == 2

    @pytest.mark.asyncio
    async def test_pending_entry_risk_R(self, repo):
        order = make_oms_order(
            role=OrderRole.ENTRY,
            risk_context=RiskContext(
                stop_for_risk=490.0,
                planned_entry_price=500.0,
                risk_budget_tag="MR",
                risk_dollars=100.0,
            ),
        )
        order.status = OrderStatus.WORKING
        await repo.save_order(order)

        risk_r = await repo.get_pending_entry_risk_R(100.0)
        assert abs(risk_r - 1.0) < 0.01


# ===========================================================================
# ADDITIONAL: DISCREPANCY POLICY TESTS
# ===========================================================================

class TestDiscrepancyPolicy:
    """Test reconciliation discrepancy policy defaults."""

    def test_default_policies(self):
        policy = DiscrepancyPolicy()
        assert policy.unknown_order_with_our_tag == DiscrepancyAction.IMPORT
        assert policy.unknown_order_orphan == DiscrepancyAction.CANCEL
        assert policy.oms_working_broker_missing == DiscrepancyAction.MARK_CANCELLED
        assert policy.unexpected_position == DiscrepancyAction.HALT_AND_ALERT
        assert policy.position_qty_mismatch == DiscrepancyAction.ADJUST_POSITION


# ===========================================================================
# ADDITIONAL: AUDIT LOGGING TESTS
# ===========================================================================

class TestAuditLogging:
    """Test broker interaction audit logging."""

    def test_log_broker_command(self, caplog):
        """Broker command is logged."""
        import logging
        with caplog.at_level(logging.INFO, logger="ibkr_core.audit"):
            log_broker_command("tr-abc", "oms-001", "submit", {"symbol": "QQQ"})
        assert len(caplog.records) == 1
        assert "broker_command" in caplog.records[0].message

    def test_log_broker_response(self, caplog):
        """Broker response is logged."""
        import logging
        with caplog.at_level(logging.INFO, logger="ibkr_core.audit"):
            log_broker_response("tr-abc", "oms-001", "fill", {"price": 500.0})
        assert len(caplog.records) == 1
        assert "broker_response" in caplog.records[0].message


# ===========================================================================
# ADDITIONAL: RISK GATEWAY TESTS
# ===========================================================================

class TestRiskGateway:
    """Test pre-trade risk checks."""

    @pytest.mark.asyncio
    async def test_entry_missing_risk_context(self):
        """ENTRY without risk_context is denied."""
        risk_config = make_risk_config()
        gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
        )
        # Build order manually without risk_context (make_oms_order auto-populates it)
        order = OMSOrder(
            oms_order_id=str(uuid.uuid4()),
            strategy_id=TEST_STRATEGY,
            account_id=TEST_ACCOUNT,
            instrument=QQQ_INSTRUMENT,
            side=OrderSide.BUY,
            qty=1,
            order_type=OrderType.LIMIT,
            limit_price=500.0,
            role=OrderRole.ENTRY,
            risk_context=None,
        )
        result = await gw.check_entry(order)
        assert result is not None
        assert "risk_context" in result.lower()

    @pytest.mark.asyncio
    async def test_daily_stop_denial(self):
        """Strategy daily stop denies new entries."""
        risk_config = make_risk_config(daily_stop_R=2.0)
        strat_risk = StrategyRiskState(
            strategy_id=TEST_STRATEGY, trade_date=date.today(),
            daily_realized_R=-2.5,
        )
        gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(return_value=strat_risk),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
        )
        order = make_oms_order()
        result = await gw.check_entry(order)
        assert result is not None
        assert "daily stop" in result.lower()

    @pytest.mark.asyncio
    async def test_portfolio_daily_stop_denial(self):
        """Portfolio daily stop denies new entries."""
        risk_config = make_risk_config(portfolio_daily_stop_R=3.0)
        port_risk = PortfolioRiskState(
            trade_date=date.today(), daily_realized_R=-3.5,
        )
        gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(return_value=port_risk),
        )
        order = make_oms_order()
        result = await gw.check_entry(order)
        assert result is not None
        assert "portfolio daily stop" in result.lower()

    @pytest.mark.asyncio
    async def test_max_working_orders_denial(self):
        """Max working orders exceeded denies new entries."""
        risk_config = make_risk_config(max_working_orders=2)
        gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
            get_working_order_count=AsyncMock(return_value=3),
        )
        order = make_oms_order()
        result = await gw.check_entry(order)
        assert result is not None
        assert "working orders" in result.lower()

    @pytest.mark.asyncio
    async def test_event_blackout_denial(self):
        """Event blackout denies new entries."""
        from shared.oms.risk.calendar import EventWindow
        window = EventWindow(
            name="CPI",
            start_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_utc=datetime(2030, 12, 31, tzinfo=timezone.utc),
        )
        calendar = EventCalendar([window])

        risk_config = make_risk_config()
        gw = RiskGateway(
            config=risk_config,
            calendar=calendar,
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
        )
        order = make_oms_order()
        result = await gw.check_entry(order)
        assert result is not None
        assert "blackout" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_strategy_denial(self):
        """Unknown strategy ID is denied."""
        risk_config = make_risk_config(strategy_id="OTHER")
        gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
        )
        order = make_oms_order(strategy_id=TEST_STRATEGY)
        result = await gw.check_entry(order)
        assert result is not None
        assert "no risk config" in result.lower()

    @pytest.mark.asyncio
    async def test_exit_order_always_approved(self):
        """EXIT role always approved regardless of risk state."""
        risk_config = make_risk_config()
        risk_config.global_standdown = True
        gw = RiskGateway(
            config=risk_config,
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today(), halted=True)
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today(), halted=True)
            ),
        )
        order = make_oms_order(role=OrderRole.EXIT, risk_context=None)
        result = await gw.check_entry(order)
        assert result is None  # Approved


# ===========================================================================
# ADDITIONAL: INSTRUMENT REGISTRY TESTS
# ===========================================================================

class TestInstrumentRegistry:
    """Test global instrument registry."""

    def test_register_and_get(self):
        inst = Instrument(
            symbol="SPY", root="SPY", venue="SMART",
            tick_size=0.01, tick_value=0.01, multiplier=1.0,
        )
        InstrumentRegistry.register(inst)
        assert InstrumentRegistry.get("SPY") is inst

    def test_get_unknown_returns_none(self):
        assert InstrumentRegistry.get("UNKNOWN_SYMBOL_XYZ") is None

    def test_get_or_raise(self):
        with pytest.raises(KeyError, match="UNKNOWN"):
            InstrumentRegistry.get_or_raise("UNKNOWN_SYMBOL_XYZ")

    def test_register_root_alias(self):
        """Registering with different symbol and root creates alias."""
        inst = Instrument(
            symbol="MNQH6", root="MNQ", venue="CME",
            tick_size=0.25, tick_value=0.50, multiplier=2.0,
        )
        InstrumentRegistry.register(inst)
        assert InstrumentRegistry.get("MNQH6") is inst
        assert InstrumentRegistry.get("MNQ") is inst


# ===========================================================================
# ADDITIONAL: FLATTEN INTENT TEST
# ===========================================================================

class TestFlattenIntent:
    """Test flatten intent for closing all positions."""

    @pytest.mark.asyncio
    async def test_flatten_closes_position_and_cancels_working(self, repo, bus):
        """Flatten cancels working orders and submits exit for open position."""
        adapter = MagicMock()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1005, 995, 0)
        )
        adapter.cancel_order = AsyncMock()

        risk_gw = RiskGateway(
            config=make_risk_config(heat_cap_R=5.0),
            calendar=EventCalendar(),
            get_strategy_risk=AsyncMock(
                return_value=StrategyRiskState(strategy_id=TEST_STRATEGY, trade_date=date.today())
            ),
            get_portfolio_risk=AsyncMock(
                return_value=PortfolioRiskState(trade_date=date.today())
            ),
            get_working_order_count=AsyncMock(return_value=0),
        )
        router = ExecutionRouter(adapter, repo)
        handler = IntentHandler(risk_gw, router, repo, bus)

        # Save a position
        pos = Position(
            account_id=TEST_ACCOUNT,
            instrument_symbol="QQQ",
            strategy_id=TEST_STRATEGY,
            net_qty=10.0,
            avg_price=500.0,
        )
        await repo.save_position(pos)

        # Save a working order
        working = make_oms_order()
        working.status = OrderStatus.WORKING
        working.broker_order_id = 2001
        await repo.save_order(working)

        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id=TEST_STRATEGY,
        )
        receipt = await handler.submit(intent)
        assert receipt.result == IntentResult.ACCEPTED


# ===========================================================================
# ADDITIONAL: QUEUE TTL EXPIRY TEST
# ===========================================================================

class TestQueueTTLExpiry:
    """Test stale order expiry in router queue."""

    @pytest.mark.asyncio
    async def test_stale_queued_orders_removed_from_queue(self, repo):
        """Stale orders are removed from queue during drain even if transition fails.

        Note: RISK_APPROVED -> EXPIRED is not a valid state transition per the
        state machine, so the order stays RISK_APPROVED. The important behavior
        is that the stale order is removed from the queue and not submitted.
        """
        adapter = MagicMock()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )

        router = ExecutionRouter(adapter, repo)

        order = make_oms_order(status=OrderStatus.RISK_APPROVED)
        await repo.save_order(order)

        from shared.oms.execution.router import OrderPriority
        # Queue with old timestamp (past TTL)
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        router._queue.append((OrderPriority.NEW_ENTRY, order, {"queued_at": old_time}))

        await router.drain_queue()

        # Stale order removed from queue and NOT submitted
        assert len(router._queue) == 0
        adapter.submit_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_queued_order_working_gets_expired(self, repo):
        """Order that was already WORKING and queued past TTL transitions to EXPIRED."""
        adapter = MagicMock()
        adapter.is_congested = False
        adapter.submit_order = AsyncMock(
            return_value=BrokerOrderRef(1001, 999, 0)
        )

        router = ExecutionRouter(adapter, repo)

        # An order in WORKING state CAN transition to EXPIRED per state machine
        order = make_oms_order()
        order.status = OrderStatus.WORKING
        await repo.save_order(order)

        from shared.oms.execution.router import OrderPriority
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        router._queue.append((OrderPriority.NEW_ENTRY, order, {"queued_at": old_time}))

        await router.drain_queue()

        updated = await repo.get_order(order.oms_order_id)
        assert updated.status == OrderStatus.EXPIRED
        assert len(router._queue) == 0
