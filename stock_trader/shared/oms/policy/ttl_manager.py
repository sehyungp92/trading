"""TTL management for working orders."""
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..models.order import OMSOrder, OrderStatus
from ..engine.state_machine import transition

if TYPE_CHECKING:
    from ..persistence.repository import OMSRepository

logger = logging.getLogger(__name__)


class TTLManager:
    """Evaluates TTL expiration for working orders.
    Called on bar close (bar-based TTL) or periodically (time-based TTL).
    """

    def __init__(self, repo: "OMSRepository"):
        self._repo = repo

    async def check_bar_ttl(self, order: OMSOrder, bars_elapsed: int) -> bool:
        """Returns True if order should be expired."""
        if not order.entry_policy or order.entry_policy.ttl_bars is None:
            return False
        # H8 fix: include PARTIALLY_FILLED orders in TTL checks
        if order.status not in {OrderStatus.WORKING, OrderStatus.ACKED, OrderStatus.PARTIALLY_FILLED}:
            return False
        return bars_elapsed >= order.entry_policy.ttl_bars

    async def check_time_ttl(self, order: OMSOrder, now: datetime) -> bool:
        """Returns True if order should be expired."""
        if not order.entry_policy or order.entry_policy.ttl_seconds is None:
            return False
        # H8 fix: include PARTIALLY_FILLED orders in TTL checks
        if order.status not in {OrderStatus.WORKING, OrderStatus.ACKED, OrderStatus.PARTIALLY_FILLED}:
            return False
        if order.submitted_at is None:
            return False
        elapsed = (now - order.submitted_at).total_seconds()
        return elapsed >= order.entry_policy.ttl_seconds

    async def expire_order(self, order: OMSOrder, reason: str = "TTL") -> None:
        """Mark order as EXPIRED and persist."""
        if not transition(order, OrderStatus.EXPIRED):
            logger.warning(
                "Cannot expire order %s: invalid transition from %s",
                order.oms_order_id, order.status.value,
            )
            return
        order.last_update_at = datetime.now(timezone.utc)
        await self._repo.save_order(order)
        await self._repo.save_event(order.oms_order_id, "EXPIRED", {"reason": reason})
