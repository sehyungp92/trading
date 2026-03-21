"""Reconciliation sync logic."""
import logging
from dataclasses import dataclass
from ..models.types import OrderStatusEvent, PositionSnapshot
from .discrepancy_policy import DiscrepancyAction, DiscrepancyPolicy

logger = logging.getLogger(__name__)


@dataclass
class Discrepancy:
    type: str  # "unknown_order", "missing_order", "position_mismatch", etc.
    action: DiscrepancyAction
    details: dict


class ReconcilerSync:
    """Compares broker snapshots against OMS expectations.

    Returns list of discrepancies with policy-driven actions.
    """

    def __init__(self, policy: DiscrepancyPolicy):
        self._policy = policy

    def reconcile_orders(
        self,
        broker_orders: list[OrderStatusEvent],
        oms_working_ids: set[int],
        our_client_id_pattern: str,
    ) -> list[Discrepancy]:
        """Compare broker open orders vs OMS working orders."""
        discrepancies = []
        broker_ids = {o.broker_order_id for o in broker_orders}

        # Orders on broker but not in OMS
        for bo in broker_orders:
            if bo.broker_order_id not in oms_working_ids:
                action = self._policy.unknown_order_with_our_tag
                discrepancies.append(
                    Discrepancy("unknown_order", action, {"order": bo})
                )

        # Orders in OMS but not on broker
        for oms_id in oms_working_ids:
            if oms_id not in broker_ids:
                discrepancies.append(
                    Discrepancy(
                        "missing_order",
                        self._policy.oms_working_broker_missing,
                        {"oms_broker_order_id": oms_id},
                    )
                )

        return discrepancies

    def reconcile_positions(
        self,
        broker_positions: list[PositionSnapshot],
        oms_positions: dict[int, float],  # con_id -> expected qty
    ) -> list[Discrepancy]:
        """Compare broker positions vs OMS positions."""
        discrepancies = []
        seen_con_ids = set()

        for bp in broker_positions:
            seen_con_ids.add(bp.con_id)
            expected = oms_positions.get(bp.con_id, 0.0)
            if bp.qty != expected:
                if bp.con_id not in oms_positions and bp.qty != 0:
                    action = self._policy.unexpected_position
                else:
                    action = self._policy.position_qty_mismatch
                discrepancies.append(
                    Discrepancy(
                        "position_mismatch",
                        action,
                        {
                            "con_id": bp.con_id,
                            "symbol": bp.symbol,
                            "broker_qty": bp.qty,
                            "oms_qty": expected,
                        },
                    )
                )

        # OMS positions not on broker
        for con_id, expected_qty in oms_positions.items():
            if con_id not in seen_con_ids and expected_qty != 0:
                discrepancies.append(
                    Discrepancy(
                        "position_mismatch",
                        self._policy.position_qty_mismatch,
                        {
                            "con_id": con_id,
                            "broker_qty": 0.0,
                            "oms_qty": expected_qty,
                        },
                    )
                )

        return discrepancies
