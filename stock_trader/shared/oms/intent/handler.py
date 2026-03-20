"""Intent handler for processing strategy requests."""
import uuid
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..models.intent import Intent, IntentType, IntentReceipt, IntentResult
from ..models.order import OMSOrder, OrderStatus

if TYPE_CHECKING:
    from ..risk.gateway import RiskGateway
    from ..execution.router import ExecutionRouter
    from ..persistence.repository import OMSRepository
    from ..events.bus import EventBus

logger = logging.getLogger(__name__)


class IntentHandler:
    """Processes strategy intents. Validates, risk-checks, routes."""

    def __init__(
        self,
        risk: "RiskGateway",
        router: "ExecutionRouter",
        repo: "OMSRepository",
        bus: "EventBus",
    ):
        self._risk = risk
        self._router = router
        self._repo = repo
        self._bus = bus
        self._idempotency: dict[str, str] = {}  # client_order_id -> oms_order_id

    async def submit(self, intent: Intent) -> IntentReceipt:
        intent_id = str(uuid.uuid4())

        if intent.intent_type == IntentType.NEW_ORDER:
            return await self._handle_new_order(intent, intent_id)
        elif intent.intent_type == IntentType.CANCEL_ORDER:
            return await self._handle_cancel(intent, intent_id)
        elif intent.intent_type == IntentType.REPLACE_ORDER:
            return await self._handle_replace(intent, intent_id)
        elif intent.intent_type == IntentType.FLATTEN:
            return await self._handle_flatten(intent, intent_id)
        else:
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Unknown intent type"
            )

    async def _handle_new_order(
        self, intent: Intent, intent_id: str
    ) -> IntentReceipt:
        order = intent.order
        if not order:
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="No order in intent"
            )

        # Idempotency check: cache first, then DB
        if order.client_order_id:
            existing_id = self._idempotency.get(order.client_order_id)
            if not existing_id:
                existing_id = await self._repo.get_order_id_by_client_order_id(
                    order.strategy_id, order.client_order_id
                )
                if existing_id:
                    self._idempotency[order.client_order_id] = existing_id
            if existing_id:
                return IntentReceipt(
                    IntentResult.ACCEPTED, intent_id, oms_order_id=existing_id
                )

        # Set timestamps
        order.created_at = datetime.now(timezone.utc)
        order.remaining_qty = order.qty

        # Risk check
        denial = await self._risk.check_entry(order)
        if denial:
            await self._repo.save_event(
                order.oms_order_id, "RISK_DENIED", {"reason": denial}
            )
            self._bus.emit_risk_denial(
                order.strategy_id, order.oms_order_id, denial,
                extra_payload={
                    "symbol": order.instrument.symbol if order.instrument else "",
                    "side": order.side.value,
                    "strategy_id": order.strategy_id,
                },
            )
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason=denial
            )

        # Approve and persist
        order.status = OrderStatus.RISK_APPROVED
        await self._repo.save_order(order)
        await self._repo.save_event(order.oms_order_id, "RISK_APPROVED", {})

        # Register idempotency
        if order.client_order_id:
            self._idempotency[order.client_order_id] = order.oms_order_id

        # Route to execution
        await self._router.route(order)

        self._bus.emit_order_event(order)
        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id, oms_order_id=order.oms_order_id
        )

    async def _handle_cancel(self, intent: Intent, intent_id: str) -> IntentReceipt:
        """Cancel a working order."""
        order = await self._repo.get_order(intent.target_oms_order_id)
        if not order:
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Order not found"
            )
        if order.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.DONE,
        }:
            return IntentReceipt(
                IntentResult.DENIED,
                intent_id,
                denial_reason=f"Order in terminal state: {order.status}",
            )

        order.status = OrderStatus.CANCEL_REQUESTED
        await self._repo.save_order(order)
        await self._router.cancel(order)
        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id, oms_order_id=order.oms_order_id
        )

    async def _handle_replace(self, intent: Intent, intent_id: str) -> IntentReceipt:
        """Replace (modify) a working order."""
        order = await self._repo.get_order(intent.target_oms_order_id)
        if not order:
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Order not found"
            )

        order.status = OrderStatus.REPLACE_REQUESTED
        await self._repo.save_order(order)
        await self._router.replace(
            order, intent.new_qty, intent.new_limit_price, intent.new_stop_price
        )
        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id, oms_order_id=order.oms_order_id
        )

    async def _handle_flatten(self, intent: Intent, intent_id: str) -> IntentReceipt:
        """Flatten positions for a strategy, optionally filtered by instrument."""
        positions = await self._repo.get_positions(
            intent.strategy_id, intent.instrument_symbol
        )
        for pos in positions:
            if pos.net_qty != 0:
                await self._router.flatten(pos)
        # Also cancel all working entry orders for this strategy
        working = await self._repo.get_working_orders(
            intent.strategy_id, intent.instrument_symbol
        )
        for order in working:
            order.status = OrderStatus.CANCEL_REQUESTED
            await self._repo.save_order(order)
            await self._router.cancel(order)
        return IntentReceipt(IntentResult.ACCEPTED, intent_id)
