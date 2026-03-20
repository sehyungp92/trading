"""Event bus for OMS event distribution."""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from ..models.events import OMSEvent, OMSEventType
from ..models.order import OMSOrder, OrderStatus

logger = logging.getLogger(__name__)

_STATUS_TO_EVENT: dict[OrderStatus, OMSEventType] = {
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


class EventBus:
    """Simple async event bus. Strategies subscribe by strategy_id."""

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._global_subscribers: list[asyncio.Queue] = []

    def subscribe(self, strategy_id: str) -> asyncio.Queue:
        """Returns an asyncio.Queue that receives OMSEvent objects for this strategy."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[strategy_id].append(q)
        return q

    def subscribe_all(self) -> asyncio.Queue:
        """Subscribe to all events (for dashboard/logging)."""
        q: asyncio.Queue = asyncio.Queue()
        self._global_subscribers.append(q)
        return q

    def unsubscribe(self, strategy_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscription."""
        if strategy_id in self._subscribers:
            try:
                self._subscribers[strategy_id].remove(queue)
            except ValueError:
                pass

    def unsubscribe_all(self, queue: asyncio.Queue) -> None:
        """Remove a global subscription."""
        try:
            self._global_subscribers.remove(queue)
        except ValueError:
            pass

    def emit_order_event(self, order: OMSOrder) -> None:
        event_type = _STATUS_TO_EVENT.get(order.status)
        if not event_type:
            return  # No event for intermediate states (CANCEL_REQUESTED, etc.)

        payload = {
            "symbol": order.instrument.symbol if order.instrument else "",
            "side": order.side.value if order.side else "",
            "order_type": order.order_type.value if order.order_type else "",
            "role": order.role.value if order.role else "",
            "qty": order.qty,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "rejection_reason": getattr(order, "rejection_reason", ""),
        }
        event = OMSEvent(
            event_type=event_type,
            timestamp=order.last_update_at or order.created_at or datetime.now(timezone.utc),
            strategy_id=order.strategy_id,
            oms_order_id=order.oms_order_id,
            payload=payload,
        )
        self._dispatch(event)

    def emit_fill_event(
        self, strategy_id: str, oms_order_id: str, fill_data: dict
    ) -> None:
        event = OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id=strategy_id,
            oms_order_id=oms_order_id,
            payload=fill_data,
        )
        self._dispatch(event)

    def emit_risk_denial(
        self, strategy_id: str, oms_order_id: str, reason: str,
        extra_payload: Optional[dict] = None,
    ) -> None:
        payload = {"reason": reason}
        if extra_payload:
            payload.update(extra_payload)
        event = OMSEvent(
            event_type=OMSEventType.RISK_DENIAL,
            timestamp=datetime.now(timezone.utc),
            strategy_id=strategy_id,
            oms_order_id=oms_order_id,
            payload=payload,
        )
        self._dispatch(event)

    def emit_risk_halt(self, strategy_id: str, reason: str) -> None:
        event = OMSEvent(
            event_type=OMSEventType.RISK_HALT,
            timestamp=datetime.now(timezone.utc),
            strategy_id=strategy_id,
            payload={"reason": reason},
        )
        self._dispatch(event)

    def _dispatch(self, event: OMSEvent) -> None:
        for q in self._subscribers.get(event.strategy_id, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Event queue full for strategy {event.strategy_id}")
        for q in self._global_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Global event queue full")
