"""Order timeout monitor for stuck transient states (C4 fix).

Scans for orders stuck in ROUTED or CANCEL_REQUESTED beyond configurable
thresholds and transitions them to terminal states.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..models.order import OrderStatus
from .state_machine import transition

if TYPE_CHECKING:
    from ..persistence.repository import OMSRepository
    from ..events.bus import EventBus

logger = logging.getLogger(__name__)

DEFAULT_ROUTED_TIMEOUT_S = 30.0
DEFAULT_CANCEL_REQUESTED_TIMEOUT_S = 15.0
DEFAULT_SCAN_INTERVAL_S = 5.0


class OrderTimeoutMonitor:
    """Background task that detects and resolves stuck orders.

    Orders can get stuck in transient states (ROUTED, CANCEL_REQUESTED)
    if the broker never sends an ACK or cancel confirmation (e.g., network
    drop, gateway restart). This monitor detects such orders and transitions
    them to terminal states.
    """

    def __init__(
        self,
        repo: "OMSRepository",
        bus: "EventBus",
        routed_timeout_s: float = DEFAULT_ROUTED_TIMEOUT_S,
        cancel_timeout_s: float = DEFAULT_CANCEL_REQUESTED_TIMEOUT_S,
        scan_interval_s: float = DEFAULT_SCAN_INTERVAL_S,
    ):
        self._repo = repo
        self._bus = bus
        self._routed_timeout_s = routed_timeout_s
        self._cancel_timeout_s = cancel_timeout_s
        self._scan_interval_s = scan_interval_s
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"OrderTimeoutMonitor started: routed={self._routed_timeout_s}s, "
            f"cancel={self._cancel_timeout_s}s, scan={self._scan_interval_s}s"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        backoff = self._scan_interval_s
        while self._running:
            try:
                await self._scan_stuck_orders()
                backoff = self._scan_interval_s  # reset on success
                await asyncio.sleep(self._scan_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Timeout monitor error (retry in %.0fs): %s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _scan_stuck_orders(self) -> None:
        now = datetime.now(timezone.utc)
        working_orders = await self._repo.get_all_working_orders()

        for order in working_orders:
            if order.status == OrderStatus.ROUTED:
                ref_time = order.submitted_at or order.created_at
                if ref_time and (now - ref_time).total_seconds() > self._routed_timeout_s:
                    logger.warning(
                        f"Order {order.oms_order_id} stuck in ROUTED for "
                        f">{self._routed_timeout_s}s — cancelling"
                    )
                    if transition(order, OrderStatus.CANCELLED):
                        order.last_update_at = now
                        await self._repo.save_order(order)
                        await self._repo.save_event(
                            order.oms_order_id, "TIMEOUT_CANCELLED",
                            {"reason": "routed_timeout", "timeout_s": self._routed_timeout_s},
                        )
                        self._bus.emit_order_event(order)

            elif order.status == OrderStatus.CANCEL_REQUESTED:
                ref_time = order.last_update_at or order.created_at
                if ref_time and (now - ref_time).total_seconds() > self._cancel_timeout_s:
                    logger.warning(
                        f"Order {order.oms_order_id} stuck in CANCEL_REQUESTED for "
                        f">{self._cancel_timeout_s}s — marking cancelled"
                    )
                    if transition(order, OrderStatus.CANCELLED):
                        order.last_update_at = now
                        await self._repo.save_order(order)
                        await self._repo.save_event(
                            order.oms_order_id, "TIMEOUT_CANCELLED",
                            {"reason": "cancel_timeout", "timeout_s": self._cancel_timeout_s},
                        )
                        self._bus.emit_order_event(order)
