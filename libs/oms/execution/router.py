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
    from ..events.bus import EventBus
    from ..persistence.repository import OMSRepository

logger = logging.getLogger(__name__)

DRAIN_INTERVAL_SEC = 1.0
QUEUE_TTL_SECONDS = 300  # H2: queued orders expire after 5 minutes


class OrderPriority(IntEnum):
    STOP_EXIT = 0  # Highest
    CANCEL = 1
    REPLACE = 2
    NEW_ENTRY = 3  # Lowest


class ExecutionRouter:
    """Routes RISK_APPROVED orders to the broker adapter with priority queuing.
    Priority: stops/exits > cancels > replaces > new entries.
    """

    def __init__(self, adapter, repo: "OMSRepository", bus: "EventBus | None" = None):
        self._adapter = adapter  # IBKRExecutionAdapter
        self._repo = repo
        self._bus = bus
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
                if self._queue:
                    await self._expire_stale_queued_orders()
                    if not self._adapter.is_congested:
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
                # H2: Record queued_at timestamp for TTL expiry
                self._queue.append((priority, order, {"queued_at": datetime.now(timezone.utc)}))
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
        new_limit_price: Optional[float],
        new_stop_price: Optional[float],
    ) -> None:
        if order.broker_order_id is None:
            return
        await self._adapter.replace_order(
            order.broker_order_id, new_qty, new_limit_price, new_stop_price
        )

    async def flatten(self, position: Position) -> OMSOrder:
        """Submit a market order to flatten a position. Returns the created order."""
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
            remaining_qty=qty,
            created_at=datetime.now(timezone.utc),
        )
        await self._submit_to_adapter(order)
        return order

    async def _submit_to_adapter(self, order: OMSOrder) -> None:
        if not transition(order, OrderStatus.ROUTED):
            logger.warning(
                "Cannot route order %s — transition from %s to ROUTED invalid, skipping submission",
                order.oms_order_id, order.status,
            )
            return
        order.submitted_at = datetime.now(timezone.utc)
        await self._repo.save_order(order)

        # Get contract expiry from instrument
        contract_expiry = order.instrument.contract_expiry if order.instrument else ""

        try:
            ref = await self._adapter.submit_order(
                oms_order_id=order.oms_order_id,
                contract_symbol=order.instrument.root if order.instrument else "",
                contract_expiry=contract_expiry,
                action=order.side.value,
                order_type=order.order_type.value,
                qty=order.qty,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                tif=order.tif,
                oca_group=order.oca_group,
                oca_type=order.oca_type,
                client_order_id=order.client_order_id or None,
                instrument=order.instrument,
            )
        except Exception as exc:
            logger.exception(
                "Broker submission failed for order %s — rolling back to REJECTED",
                order.oms_order_id,
            )
            message = str(exc) or exc.__class__.__name__
            transition(order, OrderStatus.REJECTED)
            order.reject_reason = message
            order.last_update_at = datetime.now(timezone.utc)
            await self._repo.save_order_and_event(
                order,
                "BROKER_SUBMIT_FAILED",
                {
                    "error_type": exc.__class__.__name__,
                    "error": message,
                    "instrument_backed": order.instrument is not None,
                },
            )
            if self._bus is not None:
                self._bus.emit_order_event(order)
            return

        order.broker_order_id = ref.broker_order_id
        order.perm_id = ref.perm_id
        await self._repo.save_order(order)

    @staticmethod
    def _get_priority(order: OMSOrder) -> OrderPriority:
        if order.role in {OrderRole.STOP, OrderRole.EXIT}:
            return OrderPriority.STOP_EXIT
        return OrderPriority.NEW_ENTRY

    async def drain_queue(self) -> None:
        """Process queued orders by priority when adapter is no longer congested.

        H2: Skip and expire orders older than QUEUE_TTL_SECONDS.
        """
        if self._adapter.is_congested:
            return
        await self._expire_stale_queued_orders()
        self._queue.sort(key=lambda x: x[0])
        while self._queue and not self._adapter.is_congested:
            _, order, _ = self._queue.pop(0)
            await self._submit_to_adapter(order)

    async def _expire_stale_queued_orders(self) -> None:
        """Expire stale queued orders even if adapter congestion persists."""
        now = datetime.now(timezone.utc)
        fresh: list[tuple[OrderPriority, OMSOrder, dict]] = []

        for priority, order, meta in self._queue:
            queued_at = meta.get("queued_at")
            if queued_at and (now - queued_at).total_seconds() > QUEUE_TTL_SECONDS:
                logger.warning(
                    f"Expiring stale queued order {order.oms_order_id} "
                    f"(queued {(now - queued_at).total_seconds():.0f}s ago)"
                )
                if transition(order, OrderStatus.EXPIRED):
                    order.last_update_at = now
                    await self._repo.save_order_and_event(
                        order,
                        "QUEUE_EXPIRED",
                        {"queued_seconds": (now - queued_at).total_seconds()},
                    )
                    if self._bus is not None:
                        self._bus.emit_order_event(order)
                else:
                    fresh.append((priority, order, meta))
            else:
                fresh.append((priority, order, meta))

        self._queue = fresh
