"""Reconciliation orchestrator."""
import logging
from typing import TYPE_CHECKING

from ...ibkr_core.reconciler.sync import ReconcilerSync, Discrepancy
from ...ibkr_core.reconciler.discrepancy_policy import DiscrepancyAction, DiscrepancyPolicy

if TYPE_CHECKING:
    from ..persistence.repository import OMSRepository
    from ..events.bus import EventBus

logger = logging.getLogger(__name__)


class ReconciliationOrchestrator:
    """Startup + periodic reconciliation.
    Calls ibkr_core reconciler and applies results to OMS state.
    """

    def __init__(self, adapter, repo: "OMSRepository", bus: "EventBus"):
        self._adapter = adapter  # IBKRExecutionAdapter
        self._repo = repo
        self._bus = bus
        self._policy = DiscrepancyPolicy()
        self._reconciler = ReconcilerSync(self._policy)

    async def startup_reconciliation(self) -> None:
        """MANDATORY: run before accepting any intents.
        1. Load OMS state from DB
        2. Pull broker snapshots
        3. Reconcile
        4. Apply discrepancy actions
        """
        logger.info("Starting reconciliation...")

        # Rebuild broker->OMS order mappings first so fills/cancels remain routable
        # after a process restart.
        try:
            await self._adapter.rebuild_cache(self._repo.get_order_id_by_broker_order_id)
        except Exception as e:
            logger.warning("Cache rebuild before reconciliation failed: %s", e)

        # Fetch broker state
        broker_orders = await self._adapter.request_open_orders()
        broker_positions = await self._adapter.request_positions()
        logger.info(
            f"Broker state: {len(broker_orders)} orders, "
            f"{len(broker_positions)} positions"
        )

        # C3 fix: Compare against OMS DB state
        # Gather OMS working order broker IDs
        oms_working_orders = await self._repo.get_all_working_orders()
        oms_working_broker_ids = set()
        for order in oms_working_orders:
            if order.broker_order_id is not None:
                oms_working_broker_ids.add(order.broker_order_id)

        # Reconcile orders
        order_discrepancies = self._reconciler.reconcile_orders(
            broker_orders, oms_working_broker_ids, our_client_id_pattern=""
        )

        # Gather OMS position quantities by con_id
        oms_positions = await self._repo.get_all_positions()
        oms_position_map: dict[int, float] = {}
        for pos in oms_positions:
            # Map instrument symbol to con_id via adapter cache
            broker_info = self._adapter.cache.contracts
            for con_id, spec in broker_info.items():
                if spec.symbol == pos.instrument_symbol:
                    oms_position_map[con_id] = oms_position_map.get(con_id, 0.0) + pos.net_qty
                    break

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
                logger.error(
                    f"CRITICAL DISCREPANCY — halting: {d.type} {d.details}"
                )
                # Emit risk halt event for all affected strategies
                self._bus.emit_risk_halt("", f"Reconciliation: {d.type} — {d.details}")

            elif d.action == DiscrepancyAction.MARK_CANCELLED:
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

            elif d.action == DiscrepancyAction.IMPORT:
                # Unknown order with our tag — log for manual review
                logger.warning(f"Unknown order with our tag — needs manual import: {d.details}")

            elif d.action == DiscrepancyAction.ADJUST_POSITION:
                # Position qty mismatch — log for manual review
                logger.warning(
                    f"Position mismatch: broker_qty={d.details.get('broker_qty')}, "
                    f"oms_qty={d.details.get('oms_qty')} for {d.details.get('symbol', d.details.get('con_id'))}"
                )

    async def periodic_reconciliation(self) -> None:
        """Run every 60-180 seconds. Verifies open orders and positions."""
        broker_orders = await self._adapter.request_open_orders()
        broker_positions = await self._adapter.request_positions()

        # Lightweight comparison
        oms_working_orders = await self._repo.get_all_working_orders()
        oms_working_broker_ids = set()
        for order in oms_working_orders:
            if order.broker_order_id is not None:
                oms_working_broker_ids.add(order.broker_order_id)

        order_discrepancies = self._reconciler.reconcile_orders(
            broker_orders, oms_working_broker_ids, our_client_id_pattern=""
        )

        if order_discrepancies:
            logger.warning(f"Periodic recon: {len(order_discrepancies)} order discrepancies")
            await self._apply_discrepancies(order_discrepancies)
        else:
            logger.debug(
                f"Periodic recon: {len(broker_orders)} orders, "
                f"{len(broker_positions)} positions — OK"
            )

    async def on_reconnect_reconciliation(self) -> None:
        """Immediate recon after reconnection."""
        logger.info("Running post-reconnect reconciliation")
        await self.startup_reconciliation()
