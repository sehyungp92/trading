"""Reconciliation orchestrator."""
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from libs.broker_ibkr.models.types import BrokerOrderRef, BrokerOrderStatus, OrderStatusEvent
from libs.broker_ibkr.reconciler.sync import ReconcilerSync, Discrepancy
from libs.broker_ibkr.reconciler.discrepancy_policy import DiscrepancyAction, DiscrepancyPolicy

if TYPE_CHECKING:
    from ..engine.fill_processor import FillProcessor
    from ..persistence.repository import OMSRepository
    from ..events.bus import EventBus

logger = logging.getLogger(__name__)


class ReconciliationOrchestrator:
    """Startup + periodic reconciliation.
    Calls broker_ibkr reconciler and applies results to OMS state.
    """

    def __init__(
        self,
        adapter,
        repo: "OMSRepository",
        bus: "EventBus",
        halt_trading: Optional[Callable[[str], Awaitable[None]]] = None,
        fill_processor: Optional["FillProcessor"] = None,
        offline_fill_importer: Optional[Callable[[str, object], Awaitable[bool]]] = None,
    ):
        self._adapter = adapter  # IBKRExecutionAdapter
        self._repo = repo
        self._bus = bus
        self._halt_trading = halt_trading
        # OMS-3: production passes offline_fill_importer so startup broker
        # executions run through the same side-effect path as live fills.
        self._offline_fill_importer = offline_fill_importer
        self._fill_processor = fill_processor
        self._policy = DiscrepancyPolicy()
        self._reconciler = ReconcilerSync(self._policy)

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    async def _working_order_context(self) -> tuple[list, set[int], dict[str, str]]:
        """Return working OMS orders, broker IDs, and exact repair refs."""
        oms_working_orders = await self._repo.get_all_working_orders()
        oms_working_broker_ids: set[int] = set()
        known_order_refs: dict[str, str] = {}
        duplicate_refs: set[str] = set()

        for order in oms_working_orders:
            broker_order_id = self._int_or_none(order.broker_order_id)
            if broker_order_id is not None:
                oms_working_broker_ids.add(broker_order_id)
            for ref in (order.client_order_id, order.oms_order_id):
                ref = (ref or "").strip()
                if not ref:
                    continue
                existing = known_order_refs.get(ref)
                if existing is not None and existing != order.oms_order_id:
                    duplicate_refs.add(ref)
                    known_order_refs.pop(ref, None)
                    continue
                if ref not in duplicate_refs:
                    known_order_refs[ref] = order.oms_order_id

        return oms_working_orders, oms_working_broker_ids, known_order_refs

    async def _halt_for_discrepancy(self, d: Discrepancy, reason: str | None = None) -> None:
        reason = reason or f"Reconciliation: {d.type} - {d.details}"
        logger.error("CRITICAL DISCREPANCY - halting: %s %s", d.type, d.details)
        if self._halt_trading is not None:
            await self._halt_trading(reason)
        self._bus.emit_risk_halt("", reason)

    @staticmethod
    def _target_status_from_broker_event(order_event: OrderStatusEvent):
        from ..models.order import OrderStatus

        if order_event.filled_qty > 0 and order_event.remaining_qty <= 0:
            return OrderStatus.FILLED
        if order_event.filled_qty > 0:
            return OrderStatus.PARTIALLY_FILLED
        if order_event.status == BrokerOrderStatus.PENDING_SUBMIT:
            return OrderStatus.ROUTED
        if order_event.status == BrokerOrderStatus.PRE_SUBMITTED:
            return OrderStatus.ACKED
        if order_event.status == BrokerOrderStatus.SUBMITTED:
            return OrderStatus.WORKING
        if order_event.status == BrokerOrderStatus.PENDING_CANCEL:
            return OrderStatus.CANCEL_REQUESTED
        if order_event.status == BrokerOrderStatus.CANCELLED:
            return OrderStatus.CANCELLED
        if order_event.status == BrokerOrderStatus.FILLED:
            return OrderStatus.FILLED
        if order_event.status == BrokerOrderStatus.INACTIVE:
            return OrderStatus.REJECTED
        return None

    @staticmethod
    def _advance_order_status(order, target_status) -> None:
        from ..engine.state_machine import TRANSITIONS, transition
        from ..models.order import OrderStatus

        if target_status is None or order.status == target_status:
            return
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            return

        paths = {
            OrderStatus.ROUTED: (OrderStatus.ROUTED,),
            OrderStatus.ACKED: (OrderStatus.ROUTED, OrderStatus.ACKED),
            OrderStatus.WORKING: (OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.WORKING),
            OrderStatus.PARTIALLY_FILLED: (
                OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED,
            ),
            OrderStatus.FILLED: (
                OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
            ),
            OrderStatus.CANCEL_REQUESTED: (OrderStatus.CANCEL_REQUESTED,),
            OrderStatus.CANCELLED: (OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.CANCELLED),
            OrderStatus.REJECTED: (OrderStatus.REJECTED,),
        }
        for candidate in paths.get(target_status, (target_status,)):
            if order.status == candidate:
                continue
            if candidate in TRANSITIONS.get(order.status, set()):
                transition(order, candidate)
            if order.status == target_status:
                return

    async def _repair_order_mapping(self, d: Discrepancy) -> None:
        """Repair a broker order that exactly matches a live OMS ref."""
        order_event = d.details.get("order")
        oms_order_id = d.details.get("oms_order_id")
        if order_event is None or not oms_order_id:
            await self._halt_for_discrepancy(d, f"Reconciliation repair missing details: {d.details}")
            return

        order = await self._repo.get_order(oms_order_id)
        if order is None:
            await self._halt_for_discrepancy(
                d,
                f"Reconciliation repair ref resolved to missing OMS order: {oms_order_id}",
            )
            return

        broker_order_id = self._int_or_none(order_event.broker_order_id)
        if broker_order_id is None:
            await self._halt_for_discrepancy(
                d,
                f"Reconciliation repair has invalid broker order id: {order_event.broker_order_id}",
            )
            return
        perm_id = self._int_or_none(order_event.perm_id) or 0

        self._adapter.cache.register_order(
            order.oms_order_id,
            broker_order_id,
            perm_id,
        )
        if order_event.status in {BrokerOrderStatus.PRE_SUBMITTED, BrokerOrderStatus.SUBMITTED}:
            self._adapter.cache.mark_acked(order.oms_order_id)

        order.broker_order_id = broker_order_id
        order.perm_id = perm_id
        order.broker_order_ref = BrokerOrderRef(
            broker_order_id=broker_order_id,
            perm_id=perm_id,
            con_id=0,
        )
        order.filled_qty = max(float(order.filled_qty or 0.0), float(order_event.filled_qty or 0.0))
        order.remaining_qty = float(order_event.remaining_qty or 0.0)
        if order_event.avg_fill_price:
            order.avg_fill_price = float(order_event.avg_fill_price)
        order.last_update_at = datetime.now(timezone.utc)
        self._advance_order_status(order, self._target_status_from_broker_event(order_event))

        await self._repo.save_order(order)
        self._bus.emit_order_event(order)
        logger.info(
            "Repaired OMS/broker mapping: oms_order_id=%s broker_order_id=%s order_ref=%s",
            order.oms_order_id,
            order_event.broker_order_id,
            d.details.get("order_ref", ""),
        )

    async def _build_oms_position_map(self) -> dict[int, float]:
        """Build con_id -> net_qty map from OMS positions, aggregating across strategies."""
        oms_positions = await self._repo.get_all_positions()
        oms_position_map: dict[int, float] = defaultdict(float)
        broker_contracts = self._adapter.cache.contracts
        for pos in oms_positions:
            for con_id, spec in broker_contracts.items():
                if spec.symbol == pos.instrument_symbol:
                    oms_position_map[con_id] += pos.net_qty
                    break
        return oms_position_map

    async def startup_reconciliation(self) -> None:
        """MANDATORY: run before accepting any intents.
        1. Load OMS state from DB
        2. Pull broker snapshots
        3. Reconcile
        4. Apply discrepancy actions
        """
        logger.info("Starting reconciliation...")

        # OMS-3: import broker executions that are missing locally. Production
        # uses the live fill callback pipeline; FillProcessor is a compatibility
        # fallback for older tests/callers.
        async def _fill_importer(oms_order_id: str, exec_report) -> bool:
            if self._offline_fill_importer is not None:
                return await self._offline_fill_importer(oms_order_id, exec_report)
            if self._fill_processor is None:
                return False
            ts = getattr(exec_report, "fill_time", None) or datetime.now(timezone.utc)
            commission = getattr(exec_report, "commission", 0.0) or 0.0
            try:
                return await self._fill_processor.process_fill(
                    oms_order_id=oms_order_id,
                    broker_fill_id=exec_report.exec_id,
                    price=float(exec_report.price),
                    qty=float(exec_report.qty),
                    timestamp=ts,
                    fees=float(commission),
                )
            except Exception as e:
                logger.exception(
                    "Offline fill import failed for exec_id=%s oms_order_id=%s: %s",
                    exec_report.exec_id, oms_order_id, e,
                )
                raise

        # Rebuild broker->OMS order mappings first so fills/cancels remain routable
        # after a process restart.
        try:
            await self._adapter.rebuild_cache(
                self._repo.get_order_id_by_broker_order_id,
                fill_exists_check=self._repo.fill_exists,
                fill_importer=_fill_importer,
            )
        except Exception as e:
            logger.warning("Cache rebuild before reconciliation failed: %s", e)

        # Fetch broker state
        broker_orders = await self._adapter.request_open_orders()
        broker_positions = await self._adapter.request_positions()
        broker_executions = await self._adapter.request_executions()

        logger.info(
            f"Broker state: {len(broker_orders)} orders, "
            f"{len(broker_positions)} positions, "
            f"{len(broker_executions)} executions"
        )

        # C3 fix: Compare against OMS DB state
        _oms_working_orders, oms_working_broker_ids, known_order_refs = (
            await self._working_order_context()
        )

        # Reconcile orders
        order_discrepancies = self._reconciler.reconcile_orders(
            broker_orders,
            oms_working_broker_ids,
            known_order_refs=known_order_refs,
        )

        # Gather OMS position quantities by con_id
        oms_position_map = await self._build_oms_position_map()

        # Reconcile positions
        position_discrepancies = self._reconciler.reconcile_positions(
            broker_positions, oms_position_map
        )

        all_discrepancies = order_discrepancies + position_discrepancies

        if all_discrepancies:
            logger.warning(f"Reconciliation found {len(all_discrepancies)} discrepancies")
            await self._apply_discrepancies(all_discrepancies)
        else:
            logger.info("Reconciliation complete: no discrepancies found")

    async def _apply_discrepancies(self, discrepancies: list[Discrepancy]) -> None:
        """Apply policy-driven actions for each discrepancy."""
        for d in discrepancies:
            logger.warning(f"Discrepancy: type={d.type}, action={d.action.value}, details={d.details}")

            if d.action == DiscrepancyAction.HALT_AND_ALERT:
                await self._halt_for_discrepancy(d)
                continue
            if d.action == DiscrepancyAction.REPAIR_MAPPING:
                await self._repair_order_mapping(d)
                continue
            if d.action == DiscrepancyAction.IMPORT:
                await self._halt_for_discrepancy(
                    d,
                    f"Reconciliation unsafe unknown tagged broker order: {d.details}",
                )
                continue
            if d.action == DiscrepancyAction.ADJUST_POSITION:
                await self._halt_for_discrepancy(
                    d,
                    "Reconciliation position mismatch: "
                    f"broker_qty={d.details.get('broker_qty')}, "
                    f"oms_qty={d.details.get('oms_qty')} for "
                    f"{d.details.get('symbol', d.details.get('con_id'))}",
                )
                continue

            if d.action == DiscrepancyAction.MARK_CANCELLED:
                # OMS thinks order is working but broker doesn't have it
                oms_broker_id = d.details.get("oms_broker_order_id")
                if oms_broker_id is not None:
                    oms_id = self._adapter.cache.lookup_oms_id(oms_broker_id)
                    if not oms_id:
                        oms_id = await self._repo.get_order_id_by_broker_order_id(oms_broker_id)
                    if oms_id:
                        order = await self._repo.get_order(oms_id)
                        if order:
                            from ..models.order import OrderStatus
                            from ..engine.state_machine import transition
                            if transition(order, OrderStatus.CANCELLED):
                                await self._repo.save_order(order)
                                self._bus.emit_order_event(order)
                                logger.info(f"Marked missing order as cancelled: {oms_id}")

            elif d.action == DiscrepancyAction.CANCEL:
                # Unknown orphan order at broker — cancel it
                order_event = d.details.get("order")
                if order_event:
                    try:
                        await self._adapter.cancel_order(
                            order_event.broker_order_id, order_event.perm_id
                        )
                        logger.info(f"Cancelled orphan broker order: {order_event.broker_order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel orphan order: {e}")

    async def periodic_reconciliation(self) -> None:
        """Run every 60-180 seconds. Verifies open orders and positions."""
        broker_orders = await self._adapter.request_open_orders()
        broker_positions = await self._adapter.request_positions()

        # Lightweight comparison
        _oms_working_orders, oms_working_broker_ids, known_order_refs = (
            await self._working_order_context()
        )

        order_discrepancies = self._reconciler.reconcile_orders(
            broker_orders,
            oms_working_broker_ids,
            known_order_refs=known_order_refs,
        )

        oms_position_map = await self._build_oms_position_map()
        position_discrepancies = self._reconciler.reconcile_positions(
            broker_positions, oms_position_map
        )

        all_discrepancies = order_discrepancies + position_discrepancies

        if all_discrepancies:
            logger.warning(f"Periodic recon: {len(all_discrepancies)} discrepancies ({len(order_discrepancies)} order, {len(position_discrepancies)} position)")
            await self._apply_discrepancies(all_discrepancies)
        else:
            logger.debug(
                f"Periodic recon: {len(broker_orders)} orders, "
                f"{len(broker_positions)} positions — OK"
            )

    async def on_reconnect_reconciliation(self) -> None:
        """Immediate recon after reconnection."""
        logger.info("Running post-reconnect reconciliation")
        await self.startup_reconciliation()
