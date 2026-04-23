"""Fill processing logic."""
import asyncio
import copy
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from ..models.fill import Fill
from ..models.order import OMSOrder, OrderStatus
from .state_machine import transition

if TYPE_CHECKING:
    from ..persistence.repository import OMSRepository

logger = logging.getLogger(__name__)


class FillProcessor:
    """Processes fills from broker adapter. Updates orders, positions, risk."""

    def __init__(self, repo: "OMSRepository"):
        self._repo = repo
        self._fill_locks: dict[str, asyncio.Lock] = {}

    async def process_fill(
        self,
        oms_order_id: str,
        broker_fill_id: str,
        price: float,
        qty: float,
        timestamp: datetime,
        fees: float = 0.0,
    ) -> None:
        lock = self._fill_locks.setdefault(oms_order_id, asyncio.Lock())
        async with lock:
            await self._process_fill_locked(
                oms_order_id=oms_order_id,
                broker_fill_id=broker_fill_id,
                price=price,
                qty=qty,
                timestamp=timestamp,
                fees=fees,
            )

    async def _process_fill_locked(
        self,
        *,
        oms_order_id: str,
        broker_fill_id: str,
        price: float,
        qty: float,
        timestamp: datetime,
        fees: float = 0.0,
    ) -> None:
        # Deduplicate
        if await self._repo.fill_exists(broker_fill_id):
            logger.info(f"Duplicate fill ignored: {broker_fill_id}")
            return

        order = await self._repo.get_order(oms_order_id)
        if not order:
            logger.error(f"Fill for unknown order: {oms_order_id}")
            return

        # Create and persist fill
        fill = Fill(
            fill_id=f"f-{broker_fill_id}",
            oms_order_id=oms_order_id,
            broker_fill_id=broker_fill_id,
            price=price,
            qty=qty,
            timestamp=timestamp,
            fees=fees,
        )

        updated_order = copy.deepcopy(order)

        # Update order quantities
        old_filled = updated_order.filled_qty
        updated_order.filled_qty += qty
        updated_order.remaining_qty = max(0, updated_order.qty - updated_order.filled_qty)
        updated_order.avg_fill_price = self._compute_avg(old_filled, updated_order.avg_fill_price, price, qty)

        # Transition state
        if updated_order.remaining_qty <= 0:
            transition(updated_order, OrderStatus.FILLED)
        else:
            transition(updated_order, OrderStatus.PARTIALLY_FILLED)

        updated_order.last_update_at = timestamp
        inserted = await self._repo.save_order_fill_and_event(
            updated_order,
            fill,
            "FILL",
            {
                "broker_fill_id": broker_fill_id,
                "price": price,
                "qty": qty,
                "fees": fees,
            },
        )
        if not inserted:
            logger.info(f"Duplicate fill ignored after race: {broker_fill_id}")

    @staticmethod
    def _compute_avg(
        old_filled: float, old_avg: float, new_price: float, new_qty: float
    ) -> float:
        total_filled = old_filled + new_qty
        if total_filled <= 0:
            return new_price
        prev_total = old_avg * old_filled
        return (prev_total + new_price * new_qty) / total_filled
