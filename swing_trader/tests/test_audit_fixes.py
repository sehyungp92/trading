"""Tests for audit-driven OMS/infrastructure bug fixes.

Covers:
- Fix 1: Timeout monitor ROUTED→CANCELLED (not EXPIRED)
- Fix 2: DB schema column alignment (reject_reason, retry_count)
- Fix 3: ContractFactory sec_type dispatch (STK vs FUT)
- Fix 4: FLATTEN path ordering and return values
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.oms.engine.state_machine import transition
from shared.oms.engine.timeout_monitor import OrderTimeoutMonitor
from shared.oms.events.bus import EventBus
from shared.oms.execution.router import ExecutionRouter
from shared.oms.intent.handler import IntentHandler
from shared.oms.models.instrument import Instrument
from shared.oms.models.instrument_registry import InstrumentRegistry
from shared.oms.models.intent import Intent, IntentResult, IntentType
from shared.oms.models.order import (
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderStatus,
    OrderType,
)
from shared.oms.models.position import Position
from shared.oms.persistence.in_memory import InMemoryRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instrument(symbol: str = "QQQ") -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        currency="USD",
    )


def _make_adapter() -> SimpleNamespace:
    """Minimal mock adapter for ExecutionRouter."""
    adapter = SimpleNamespace(
        is_congested=False,
        submit_order=AsyncMock(
            return_value=SimpleNamespace(broker_order_id=12345, perm_id=99)
        ),
        cancel_order=AsyncMock(),
        replace_order=AsyncMock(),
    )
    return adapter


# ---------------------------------------------------------------------------
# Fix 1: Timeout monitor — ROUTED orders get CANCELLED (not EXPIRED)
# ---------------------------------------------------------------------------

class TestFix1TimeoutMonitor:
    """Stuck ROUTED orders must transition to CANCELLED, not EXPIRED."""

    @pytest.mark.asyncio
    async def test_stuck_routed_order_transitions_to_cancelled(self):
        repo = InMemoryRepository()
        bus = EventBus()
        monitor = OrderTimeoutMonitor(repo, bus, routed_timeout_s=1.0)

        order = OMSOrder(
            strategy_id="test",
            instrument=_make_instrument(),
            side=OrderSide.BUY,
            qty=5,
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
            status=OrderStatus.ROUTED,
            submitted_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            created_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        await repo.save_order(order)

        await monitor._scan_stuck_orders()

        saved = await repo.get_order(order.oms_order_id)
        assert saved.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_stuck_routed_emits_timeout_cancelled_event(self):
        repo = InMemoryRepository()
        bus = EventBus()
        monitor = OrderTimeoutMonitor(repo, bus, routed_timeout_s=1.0)

        order = OMSOrder(
            strategy_id="test",
            instrument=_make_instrument(),
            side=OrderSide.BUY,
            qty=5,
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
            status=OrderStatus.ROUTED,
            submitted_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            created_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        await repo.save_order(order)
        await monitor._scan_stuck_orders()

        # Check event was saved with correct type
        assert len(repo._events) == 1
        assert repo._events[0]["event_type"] == "TIMEOUT_CANCELLED"
        assert repo._events[0]["payload"]["reason"] == "routed_timeout"

    @pytest.mark.asyncio
    async def test_fresh_routed_order_not_cancelled(self):
        repo = InMemoryRepository()
        bus = EventBus()
        monitor = OrderTimeoutMonitor(repo, bus, routed_timeout_s=30.0)

        order = OMSOrder(
            strategy_id="test",
            instrument=_make_instrument(),
            side=OrderSide.BUY,
            qty=5,
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
            status=OrderStatus.ROUTED,
            submitted_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        await repo.save_order(order)
        await monitor._scan_stuck_orders()

        saved = await repo.get_order(order.oms_order_id)
        assert saved.status == OrderStatus.ROUTED


# ---------------------------------------------------------------------------
# Fix 2: DB schema — DDL must have reject_reason and retry_count
# ---------------------------------------------------------------------------

class TestFix2DDLSchema:
    """DDL must match OMSOrder model columns."""

    def test_ddl_has_reject_reason_not_rejection_reason(self):
        from shared.oms.persistence.postgres import DDL
        assert "reject_reason" in DDL
        # Column definition must use reject_reason, not rejection_reason
        # (migration block references rejection_reason for RENAME — that's fine)
        create_block = DDL.split("CREATE TABLE IF NOT EXISTS orders")[1].split(");")[0]
        assert "reject_reason" in create_block
        assert "rejection_reason" not in create_block

    def test_ddl_has_retry_count(self):
        from shared.oms.persistence.postgres import DDL
        assert "retry_count" in DDL

    def test_ddl_migration_renames_rejection_reason(self):
        from shared.oms.persistence.postgres import DDL
        assert "RENAME COLUMN rejection_reason TO reject_reason" in DDL

    def test_ddl_migration_adds_retry_count(self):
        from shared.oms.persistence.postgres import DDL
        assert "ADD COLUMN retry_count" in DDL


# ---------------------------------------------------------------------------
# Fix 3: ContractFactory — sec_type dispatch (STK vs FUT)
# ---------------------------------------------------------------------------

class TestFix3ContractFactory:
    """ContractFactory must branch on sec_type: STK→Stock(), FUT→Future()."""

    @pytest.mark.asyncio
    async def test_stk_template_creates_stock_contract(self):
        from shared.ibkr_core.config.schemas import ContractTemplate
        from shared.ibkr_core.mapping.contract_factory import ContractFactory

        tmpl = ContractTemplate(
            symbol="QQQ",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
            primary_exchange="NASDAQ",
        )

        mock_ib = AsyncMock()
        qualified_contract = MagicMock()
        qualified_contract.conId = 320227571
        qualified_contract.symbol = "QQQ"
        qualified_contract.secType = "STK"
        qualified_contract.exchange = "SMART"
        qualified_contract.currency = "USD"
        qualified_contract.multiplier = ""
        qualified_contract.tradingClass = "QQQ"
        qualified_contract.lastTradeDateOrContractMonth = ""
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified_contract])

        factory = ContractFactory(mock_ib, {"QQQ": tmpl}, {})
        contract, spec = await factory.resolve("QQQ")

        # Verify Stock was passed to qualifyContractsAsync
        call_args = mock_ib.qualifyContractsAsync.call_args
        passed_contract = call_args[0][0]
        from ib_async import Stock
        assert isinstance(passed_contract, Stock)
        assert passed_contract.primaryExchange == "NASDAQ"
        assert spec.sec_type == "STK"

    @pytest.mark.asyncio
    async def test_fut_template_creates_future_contract(self):
        from shared.ibkr_core.config.schemas import ContractTemplate
        from shared.ibkr_core.mapping.contract_factory import ContractFactory

        tmpl = ContractTemplate(
            symbol="MNQ",
            sec_type="FUT",
            exchange="CME",
            currency="USD",
            multiplier=2.0,
            tick_size=0.25,
            tick_value=0.50,
            trading_class="MNQ",
        )

        mock_ib = AsyncMock()
        qualified_contract = MagicMock()
        qualified_contract.conId = 553444806
        qualified_contract.symbol = "MNQ"
        qualified_contract.secType = "FUT"
        qualified_contract.exchange = "CME"
        qualified_contract.currency = "USD"
        qualified_contract.multiplier = "2"
        qualified_contract.tradingClass = "MNQ"
        qualified_contract.lastTradeDateOrContractMonth = "20260620"
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified_contract])

        factory = ContractFactory(mock_ib, {"MNQ": tmpl}, {})
        contract, spec = await factory.resolve("MNQ", "20260620")

        call_args = mock_ib.qualifyContractsAsync.call_args
        passed_contract = call_args[0][0]
        from ib_async import Future
        assert isinstance(passed_contract, Future)
        assert spec.sec_type == "FUT"

    @pytest.mark.asyncio
    async def test_stk_resolve_ignores_expiry(self):
        from shared.ibkr_core.config.schemas import ContractTemplate
        from shared.ibkr_core.mapping.contract_factory import ContractFactory

        tmpl = ContractTemplate(
            symbol="QQQ",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
        )

        mock_ib = AsyncMock()
        qualified_contract = MagicMock()
        qualified_contract.conId = 320227571
        qualified_contract.symbol = "QQQ"
        qualified_contract.secType = "STK"
        qualified_contract.exchange = "SMART"
        qualified_contract.currency = "USD"
        qualified_contract.multiplier = ""
        qualified_contract.tradingClass = "QQQ"
        qualified_contract.lastTradeDateOrContractMonth = ""
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified_contract])

        factory = ContractFactory(mock_ib, {"QQQ": tmpl}, {})

        # First call with no expiry
        await factory.resolve("QQQ")
        assert mock_ib.qualifyContractsAsync.call_count == 1

        # Second call with an expiry — should use cache (STK ignores expiry)
        await factory.resolve("QQQ", "20260620")
        assert mock_ib.qualifyContractsAsync.call_count == 1  # Still cached

    @pytest.mark.asyncio
    async def test_fut_without_expiry_raises(self):
        """FUT symbols require expiry — omitting it must raise."""
        from shared.ibkr_core.config.schemas import ContractTemplate
        from shared.ibkr_core.mapping.contract_factory import (
            ContractFactory,
            ContractResolutionError,
        )

        tmpl = ContractTemplate(
            symbol="MNQ",
            sec_type="FUT",
            exchange="CME",
            currency="USD",
            multiplier=2.0,
            tick_size=0.25,
            tick_value=0.50,
        )

        mock_ib = AsyncMock()
        factory = ContractFactory(mock_ib, {"MNQ": tmpl}, {})

        with pytest.raises(ContractResolutionError, match="Expiry required"):
            await factory.resolve("MNQ")  # no expiry for FUT

    @pytest.mark.asyncio
    async def test_expiry_optional_for_stk(self):
        """resolve() can be called without expiry for STK symbols."""
        from shared.ibkr_core.config.schemas import ContractTemplate
        from shared.ibkr_core.mapping.contract_factory import ContractFactory

        tmpl = ContractTemplate(
            symbol="GLD",
            sec_type="STK",
            exchange="SMART",
            currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
        )

        mock_ib = AsyncMock()
        qualified_contract = MagicMock()
        qualified_contract.conId = 12345
        qualified_contract.symbol = "GLD"
        qualified_contract.secType = "STK"
        qualified_contract.exchange = "SMART"
        qualified_contract.currency = "USD"
        qualified_contract.multiplier = ""
        qualified_contract.tradingClass = "GLD"
        qualified_contract.lastTradeDateOrContractMonth = ""
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified_contract])

        factory = ContractFactory(mock_ib, {"GLD": tmpl}, {})
        contract, spec = await factory.resolve("GLD")  # no expiry arg
        assert spec.symbol == "GLD"


# ---------------------------------------------------------------------------
# Fix 4: FLATTEN path — router returns order, handler ordering
# ---------------------------------------------------------------------------

class TestFix4aRouterFlatten:
    """Router.flatten() must return OMSOrder with remaining_qty and created_at."""

    @pytest.mark.asyncio
    async def test_flatten_returns_order(self):
        repo = InMemoryRepository()
        adapter = _make_adapter()
        router = ExecutionRouter(adapter, repo)

        inst = _make_instrument("QQQ")
        InstrumentRegistry.register(inst)

        position = Position(
            account_id="U123",
            instrument_symbol="QQQ",
            strategy_id="test",
            net_qty=10,
        )

        order = await router.flatten(position)
        assert isinstance(order, OMSOrder)
        assert order.oms_order_id is not None

    @pytest.mark.asyncio
    async def test_flatten_order_has_remaining_qty(self):
        repo = InMemoryRepository()
        adapter = _make_adapter()
        router = ExecutionRouter(adapter, repo)

        inst = _make_instrument("QQQ")
        InstrumentRegistry.register(inst)

        position = Position(
            account_id="U123",
            instrument_symbol="QQQ",
            strategy_id="test",
            net_qty=10,
        )

        order = await router.flatten(position)
        assert order.remaining_qty == 10
        assert order.qty == 10

    @pytest.mark.asyncio
    async def test_flatten_order_has_created_at(self):
        repo = InMemoryRepository()
        adapter = _make_adapter()
        router = ExecutionRouter(adapter, repo)

        inst = _make_instrument("QQQ")
        InstrumentRegistry.register(inst)

        position = Position(
            account_id="U123",
            instrument_symbol="QQQ",
            strategy_id="test",
            net_qty=-5,
        )

        before = datetime.now(timezone.utc)
        order = await router.flatten(position)
        after = datetime.now(timezone.utc)

        assert order.created_at is not None
        assert before <= order.created_at <= after
        assert order.side == OrderSide.BUY  # short position → BUY to flatten


class TestFix4bHandlerFlatten:
    """Handler._handle_flatten() must query working orders BEFORE submitting
    flatten exits, so flatten exits are not self-cancelled."""

    @pytest.mark.asyncio
    async def test_flatten_does_not_cancel_its_own_exits(self):
        repo = InMemoryRepository()
        adapter = _make_adapter()
        bus = EventBus()
        router = ExecutionRouter(adapter, repo, bus)
        risk = AsyncMock()
        handler = IntentHandler(risk, router, repo, bus)

        inst = _make_instrument("QQQ")
        InstrumentRegistry.register(inst)

        # Pre-existing working entry order
        entry_order = OMSOrder(
            strategy_id="test",
            instrument=inst,
            side=OrderSide.BUY,
            qty=10,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
            role=OrderRole.ENTRY,
            status=OrderStatus.WORKING,
            broker_order_id=111,
            created_at=datetime.now(timezone.utc),
        )
        await repo.save_order(entry_order)

        # Position to flatten
        position = Position(
            account_id="default",
            instrument_symbol="QQQ",
            strategy_id="test",
            net_qty=10,
        )
        await repo.save_position(position)

        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id="test",
            instrument_symbol="QQQ",
        )
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.ACCEPTED
        # Receipt should have the flatten exit order ID
        assert receipt.oms_order_id is not None

        # Entry order should be cancel-requested
        saved_entry = await repo.get_order(entry_order.oms_order_id)
        assert saved_entry.status == OrderStatus.CANCEL_REQUESTED

        # Flatten exit order should NOT be cancelled
        flatten_order = await repo.get_order(receipt.oms_order_id)
        assert flatten_order is not None
        assert flatten_order.status != OrderStatus.CANCEL_REQUESTED
        assert flatten_order.role == OrderRole.EXIT

    @pytest.mark.asyncio
    async def test_flatten_receipt_contains_exit_order_id(self):
        repo = InMemoryRepository()
        adapter = _make_adapter()
        bus = EventBus()
        router = ExecutionRouter(adapter, repo, bus)
        risk = AsyncMock()
        handler = IntentHandler(risk, router, repo, bus)

        inst = _make_instrument("QQQ")
        InstrumentRegistry.register(inst)

        position = Position(
            account_id="default",
            instrument_symbol="QQQ",
            strategy_id="test",
            net_qty=5,
        )
        await repo.save_position(position)

        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id="test",
            instrument_symbol="QQQ",
        )
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.ACCEPTED
        assert receipt.oms_order_id is not None

        # Verify the order exists and is an EXIT
        order = await repo.get_order(receipt.oms_order_id)
        assert order.role == OrderRole.EXIT
        assert order.order_type == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_flatten_cancels_only_preexisting_working_orders(self):
        repo = InMemoryRepository()
        adapter = _make_adapter()
        bus = EventBus()
        router = ExecutionRouter(adapter, repo, bus)
        risk = AsyncMock()
        handler = IntentHandler(risk, router, repo, bus)

        inst = _make_instrument("QQQ")
        InstrumentRegistry.register(inst)

        # Two pre-existing working orders
        for i in range(2):
            order = OMSOrder(
                strategy_id="test",
                instrument=inst,
                side=OrderSide.BUY,
                qty=5,
                order_type=OrderType.LIMIT,
                limit_price=100.0 + i,
                role=OrderRole.ENTRY,
                status=OrderStatus.WORKING,
                broker_order_id=200 + i,
                created_at=datetime.now(timezone.utc),
            )
            await repo.save_order(order)

        position = Position(
            account_id="default",
            instrument_symbol="QQQ",
            strategy_id="test",
            net_qty=10,
        )
        await repo.save_position(position)

        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id="test",
            instrument_symbol="QQQ",
        )
        receipt = await handler.submit(intent)

        # Both pre-existing orders cancelled
        assert adapter.cancel_order.call_count == 2

        # Flatten exit was NOT cancelled
        flatten_order = await repo.get_order(receipt.oms_order_id)
        assert flatten_order.status == OrderStatus.ROUTED

    @pytest.mark.asyncio
    async def test_flatten_handles_routed_order_via_state_machine(self):
        """ROUTED orders can't go to CANCEL_REQUESTED — flatten must use
        transition() and fall back to CANCELLED directly."""
        repo = InMemoryRepository()
        adapter = _make_adapter()
        bus = EventBus()
        router = ExecutionRouter(adapter, repo, bus)
        risk = AsyncMock()
        handler = IntentHandler(risk, router, repo, bus)

        inst = _make_instrument("QQQ")
        InstrumentRegistry.register(inst)

        # Pre-existing ROUTED order (just submitted, no ACK yet)
        routed_order = OMSOrder(
            strategy_id="test",
            instrument=inst,
            side=OrderSide.BUY,
            qty=5,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
            role=OrderRole.ENTRY,
            status=OrderStatus.ROUTED,
            created_at=datetime.now(timezone.utc),
        )
        await repo.save_order(routed_order)

        position = Position(
            account_id="default",
            instrument_symbol="QQQ",
            strategy_id="test",
            net_qty=5,
        )
        await repo.save_position(position)

        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id="test",
            instrument_symbol="QQQ",
        )
        receipt = await handler.submit(intent)
        assert receipt.result == IntentResult.ACCEPTED

        # ROUTED order should be CANCELLED (not invalid CANCEL_REQUESTED)
        saved = await repo.get_order(routed_order.oms_order_id)
        assert saved.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_flatten_no_positions_returns_accepted_no_order_id(self):
        repo = InMemoryRepository()
        adapter = _make_adapter()
        bus = EventBus()
        router = ExecutionRouter(adapter, repo, bus)
        risk = AsyncMock()
        handler = IntentHandler(risk, router, repo, bus)

        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id="test",
            instrument_symbol="QQQ",
        )
        receipt = await handler.submit(intent)
        assert receipt.result == IntentResult.ACCEPTED
        assert receipt.oms_order_id is None
