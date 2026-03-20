"""In-memory repository for development and testing."""
import logging
from datetime import datetime, timezone
from typing import Optional

from ..models.fill import Fill
from ..models.instrument import Instrument
from ..models.instrument_registry import InstrumentRegistry
from ..models.order import (
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskContext,
)
from ..models.position import Position

logger = logging.getLogger(__name__)


class InMemoryRepository:
    """In-memory implementation of OMSRepository for development.
    Same interface as the asyncpg-backed repository.
    """

    def __init__(self):
        self._orders: dict[str, OMSOrder] = {}
        self._events: list[dict] = []
        self._fills: dict[str, Fill] = {}
        self._positions: dict[tuple[str, str, str], Position] = {}  # (strategy, account, symbol)

    async def save_order(self, order: OMSOrder) -> None:
        """Upsert current order state."""
        self._orders[order.oms_order_id] = order

    async def save_event(self, oms_order_id: str, event_type: str, payload: dict) -> None:
        """Append to order events."""
        self._events.append({
            "oms_order_id": oms_order_id,
            "event_type": event_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc),
        })

    async def save_fill(self, fill: Fill) -> None:
        if fill.broker_fill_id not in self._fills:
            self._fills[fill.broker_fill_id] = fill

    async def fill_exists(self, broker_fill_id: str) -> bool:
        return broker_fill_id in self._fills

    async def get_order(self, oms_order_id: str) -> Optional[OMSOrder]:
        """Load order by ID. Returns None if not found."""
        return self._orders.get(oms_order_id)

    async def get_order_id_by_client_order_id(
        self, strategy_id: str, client_order_id: str
    ) -> Optional[str]:
        """Look up oms_order_id by client_order_id for idempotency."""
        for order in self._orders.values():
            if order.strategy_id == strategy_id and order.client_order_id == client_order_id:
                return order.oms_order_id
        return None

    async def get_order_id_by_broker_order_id(
        self, broker_order_id: int
    ) -> Optional[str]:
        """Resolve an OMS order ID from a broker order ID."""
        for order in self._orders.values():
            if order.broker_order_id == broker_order_id:
                return order.oms_order_id
        return None

    async def get_pending_entry_risk_R(self, unit_risk_dollars: float) -> float:
        """Sum risk_R of working ENTRY orders."""
        working_statuses = {
            OrderStatus.RISK_APPROVED,
            OrderStatus.ROUTED,
            OrderStatus.ACKED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
        }
        total_risk = 0.0
        for order in self._orders.values():
            if order.role == OrderRole.ENTRY and order.status in working_statuses:
                if order.risk_context:
                    risk = order.risk_context.risk_dollars or 0.0
                    # Scale by remaining qty for partially filled orders
                    if order.status == OrderStatus.PARTIALLY_FILLED:
                        qty = order.qty or 1
                        remaining = order.remaining_qty or 0
                        risk = risk * (remaining / qty) if qty > 0 else 0.0
                    total_risk += risk
        return total_risk / unit_risk_dollars if unit_risk_dollars > 0 else 0.0

    async def get_working_orders(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[OMSOrder]:
        """Get all non-terminal orders for a strategy."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.DONE,
        }
        result = []
        for order in self._orders.values():
            if order.strategy_id == strategy_id and order.status not in terminal:
                if instrument_symbol is None or (
                    order.instrument and order.instrument.symbol == instrument_symbol
                ):
                    result.append(order)
        return result

    async def count_working_orders(self, strategy_id: str) -> int:
        """Count non-terminal orders for a strategy."""
        orders = await self.get_working_orders(strategy_id)
        return len(orders)

    async def get_positions(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[Position]:
        result = []
        for key, pos in self._positions.items():
            strat, _, symbol = key
            if strat == strategy_id:
                if instrument_symbol is None or symbol == instrument_symbol:
                    result.append(pos)
        return result

    async def save_position(self, position: Position) -> None:
        """Upsert position."""
        key = (position.strategy_id, position.account_id, position.instrument_symbol)
        self._positions[key] = position

    async def get_all_working_orders(self) -> list[OMSOrder]:
        """Get all non-terminal orders across all strategies."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.DONE,
        }
        return [o for o in self._orders.values() if o.status not in terminal]

    async def get_all_positions(self) -> list[Position]:
        """Get all positions across all strategies."""
        return list(self._positions.values())
