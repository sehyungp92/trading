"""Execution router for order dispatch."""
import asyncio
import logging
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional, TYPE_CHECKING

from ..models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType
from ..models.position import Position
from ..engine.state_machine import transition

if TYPE_CHECKING:
    from ..persistence.repository import OMSRepository

logger = logging.getLogger(__name__)

DRAIN_INTERVAL_SEC = 1.0


class OrderPriority(IntEnum):
    STOP_EXIT = 0  # Highest
    CANCEL = 1
    REPLACE = 2
    NEW_ENTRY = 3  # Lowest


class ExecutionRouter:
    """Routes RISK_APPROVED orders to the broker adapter with priority queuing.
    Priority: stops/exits > cancels > replaces > new entries.
    """

    def __init__(self, adapter, repo: "OMSRepository"):
        self._adapter = adapter  # IBKRExecutionAdapter
        self._repo = repo
        self._queue: list[tuple[OrderPriority, OMSOrder, dict]] = []
        self._drain_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background queue drain loop."""
        if self._running:
            return
        self._running = True
        self._drain_task = asyncio.create_task(self._drain_loop())
        logger.info("ExecutionRouter drain loop started")

    async def stop(self) -> None:
        """Stop the background queue drain loop."""
        self._running = False
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        logger.info("ExecutionRouter drain loop stopped")

    async def _drain_loop(self) -> None:
        """Background loop that drains queue when adapter is not congested."""
        while self._running:
            try:
                if self._queue and not self._adapter.is_congested:
                    await self.drain_queue()
                await asyncio.sleep(DRAIN_INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in drain loop: {e}")

    async def route(self, order: OMSOrder) -> None:
        """Route a risk-approved order to the adapter."""
        priority = self._get_priority(order)

        if self._adapter.is_congested:
            if priority > OrderPriority.CANCEL:
                logger.warning(f"Adapter congested; queueing {order.oms_order_id}")
                self._queue.append((priority, order, {}))
                return

        await self._submit_to_adapter(order)

    async def cancel(self, order: OMSOrder) -> None:
        if order.broker_order_id is None:
            return
        await self._adapter.cancel_order(order.broker_order_id, order.perm_id or 0)

    async def replace(
        self,
        order: OMSOrder,
        new_qty: Optional[int],
        new_limit: Optional[float],
        new_stop: Optional[float],
    ) -> None:
        if order.broker_order_id is None:
            return
        await self._adapter.replace_order(
            order.broker_order_id, new_qty, new_limit, new_stop
        )

    async def flatten(self, position: Position) -> None:
        """Submit a market order to flatten a position."""
        from ..models.instrument_registry import InstrumentRegistry
        side = OrderSide.SELL if position.net_qty > 0 else OrderSide.BUY
        qty = abs(int(position.net_qty))
        instrument = InstrumentRegistry.get(position.instrument_symbol)
        order = OMSOrder(
            strategy_id=position.strategy_id,
            account_id=position.account_id,
            instrument=instrument,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            role=OrderRole.EXIT,
            status=OrderStatus.RISK_APPROVED,
        )
        await self._submit_to_adapter(order)

    async def _submit_to_adapter(self, order: OMSOrder) -> None:
        if not transition(order, OrderStatus.ROUTED):
            logger.warning(
                "Cannot route order %s: invalid transition from %s — aborting submission",
                order.oms_order_id, order.status.value,
            )
            return
        order.submitted_at = datetime.now(timezone.utc)
        await self._repo.save_order(order)

        if order.instrument is None:
            raise ValueError(f"Order {order.oms_order_id} missing instrument")

        ref = await self._adapter.submit_order(
            oms_order_id=order.oms_order_id,
            instrument=order.instrument,
            action=order.side.value,
            order_type=order.order_type.value,
            qty=order.qty,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            tif=order.tif,
            oca_group=order.oca_group,
            oca_type=order.oca_type,
            client_order_id=order.client_order_id or None,
        )

        order.broker_order_id = ref.broker_order_id
        order.perm_id = ref.perm_id
        await self._repo.save_order(order)

    @staticmethod
    def _get_priority(order: OMSOrder) -> OrderPriority:
        if order.role in {OrderRole.STOP, OrderRole.EXIT}:
            return OrderPriority.STOP_EXIT
        return OrderPriority.NEW_ENTRY

    async def drain_queue(self) -> None:
        """Process queued orders by priority when adapter is no longer congested."""
        if self._adapter.is_congested:
            return
        self._queue.sort(key=lambda x: x[0])
        while self._queue and not self._adapter.is_congested:
            _, order, _ = self._queue.pop(0)
            await self._submit_to_adapter(order)
