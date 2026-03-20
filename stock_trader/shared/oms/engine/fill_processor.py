"""Fill processing logic."""
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

    async def process_fill(
        self,
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
        await self._repo.save_fill(fill)

        # Update order quantities
        old_filled = order.filled_qty
        order.filled_qty += qty
        order.remaining_qty = max(0, order.qty - order.filled_qty)
        order.avg_fill_price = self._compute_avg(old_filled, order.avg_fill_price, price, qty)

        # Transition state
        if order.remaining_qty <= 0:
            target_status = OrderStatus.FILLED
        else:
            target_status = OrderStatus.PARTIALLY_FILLED
        if not transition(order, target_status):
            logger.warning(
                "Preserving existing status for fill on %s after invalid transition to %s",
                oms_order_id,
                target_status.value,
            )

        order.last_update_at = timestamp
        await self._repo.save_order(order)
        await self._repo.save_event(
            oms_order_id,
            "FILL",
            {
                "broker_fill_id": broker_fill_id,
                "price": price,
                "qty": qty,
                "fees": fees,
            },
        )

    @staticmethod
    def _compute_avg(
        old_filled: float, old_avg: float, new_price: float, new_qty: float
    ) -> float:
        total_filled = old_filled + new_qty
        if total_filled <= 0:
            return new_price
        prev_total = old_avg * old_filled
        return (prev_total + new_price * new_qty) / total_filled
