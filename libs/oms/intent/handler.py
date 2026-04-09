"""Intent handler for processing strategy requests."""
import asyncio
import uuid
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..models.intent import Intent, IntentType, IntentReceipt, IntentResult
from ..models.order import OMSOrder, OrderRole, OrderStatus
from ..engine.state_machine import transition

if TYPE_CHECKING:
    from ..risk.gateway import RiskGateway
    from ..execution.router import ExecutionRouter
    from ..persistence.repository import OMSRepository
    from ..events.bus import EventBus

logger = logging.getLogger(__name__)

_MAX_IDEMP_CACHE = 5000
_IDEMP_PRUNE_BATCH = 1000


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
        self._idempotency: OrderedDict[str, str] = OrderedDict()  # client_order_id -> oms_order_id
        # C1: per-client_order_id locks to prevent duplicate orders across concurrent tasks
        self._idemp_locks: dict[str, asyncio.Lock] = {}
        # 1C: serialize entry risk-check → persist to prevent concurrent race in shared OMS
        self._entry_lock = asyncio.Lock()

    def _prune_idemp_cache(self) -> None:
        """Evict oldest entries when cache exceeds max size.

        DB fallback at get_order_id_by_client_order_id handles cache misses.
        """
        if len(self._idempotency) <= _MAX_IDEMP_CACHE:
            return
        for _ in range(_IDEMP_PRUNE_BATCH):
            if not self._idempotency:
                break
            key, _ = self._idempotency.popitem(last=False)
            self._idemp_locks.pop(key, None)

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

        # M1: Validate qty > 0
        if order.qty <= 0:
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Order qty must be > 0"
            )

        # M2: For EXIT orders, validate qty doesn't exceed open position
        if order.role in (OrderRole.EXIT, OrderRole.STOP):
            positions = await self._repo.get_positions(
                order.strategy_id,
                order.instrument.symbol if order.instrument else None,
            )
            open_qty = sum(abs(p.net_qty) for p in positions)
            if open_qty > 0 and order.qty > open_qty:
                return IntentReceipt(
                    IntentResult.DENIED,
                    intent_id,
                    denial_reason=f"Exit qty {order.qty} exceeds open position {open_qty}",
                )

        # C1: Idempotency check under per-key lock to prevent race between
        # cache lookup and DB fallback across concurrent async tasks.
        if order.client_order_id:
            if order.client_order_id not in self._idemp_locks:
                self._idemp_locks[order.client_order_id] = asyncio.Lock()
            async with self._idemp_locks[order.client_order_id]:
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
                # Register idempotency early (inside lock) to block concurrent duplicates
                self._idempotency[order.client_order_id] = order.oms_order_id
                self._prune_idemp_cache()

        # Set timestamps
        order.created_at = datetime.now(timezone.utc)
        order.remaining_qty = order.qty

        # 1C: Serialize ENTRY risk-check → persist to prevent concurrent entries
        # from both passing heat cap before either persists (swing shared OMS).
        # Exits/stops skip the lock since RiskGateway auto-approves non-ENTRY orders.
        use_entry_lock = order.role == OrderRole.ENTRY

        async def _risk_check_and_route():
            denial = await self._risk.check_entry(order)
            if denial:
                # Roll back idempotency registration on denial so order can be retried
                if order.client_order_id:
                    self._idempotency.pop(order.client_order_id, None)
                    self._idemp_locks.pop(order.client_order_id, None)
                await self._repo.save_event(
                    order.oms_order_id, "RISK_DENIED", {"reason": denial}
                )
                self._bus.emit_risk_denial(order.strategy_id, order.oms_order_id, denial)
                return IntentReceipt(
                    IntentResult.DENIED, intent_id, denial_reason=denial
                )

            # Apply portfolio size multiplier to order qty (C2 fix: was write-only)
            if order.risk_context and order.risk_context.portfolio_size_mult != 1.0:
                mult = order.risk_context.portfolio_size_mult
                original_qty = order.qty
                order.qty = max(1, int(order.qty * mult))
                order.remaining_qty = order.qty
                if original_qty > 0:
                    order.risk_context.risk_dollars *= order.qty / original_qty
                logger.info(
                    "Portfolio size mult %.2fx: qty %d -> %d for %s",
                    mult, original_qty, order.qty, order.strategy_id,
                )

            # Approve and persist
            order.status = OrderStatus.RISK_APPROVED
            await self._repo.save_order(order)
            await self._repo.save_event(order.oms_order_id, "RISK_APPROVED", {})

            # Route to execution
            await self._router.route(order)
            return None  # success

        if use_entry_lock:
            async with self._entry_lock:
                receipt = await _risk_check_and_route()
        else:
            receipt = await _risk_check_and_route()

        if receipt is not None:
            return receipt

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
        # 1. Snapshot working orders BEFORE creating flatten exits
        working = await self._repo.get_working_orders(
            intent.strategy_id, intent.instrument_symbol
        )

        # 2. Submit flatten exits
        flatten_order_ids: list[str] = []
        positions = await self._repo.get_positions(
            intent.strategy_id, intent.instrument_symbol
        )
        for pos in positions:
            if pos.net_qty != 0:
                order = await self._router.flatten(pos)
                if order is not None:
                    flatten_order_ids.append(order.oms_order_id)

        # 3. Cancel pre-existing working orders only (not the new flatten exits)
        for order in working:
            if transition(order, OrderStatus.CANCEL_REQUESTED):
                await self._repo.save_order(order)
                await self._router.cancel(order)
            elif transition(order, OrderStatus.CANCELLED):
                # ROUTED/ACKED go directly to CANCELLED
                order.last_update_at = datetime.now(timezone.utc)
                await self._repo.save_order(order)
                await self._router.cancel(order)

        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id,
            oms_order_id=flatten_order_ids[0] if flatten_order_ids else None,
        )
