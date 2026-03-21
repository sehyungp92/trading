"""Reconciliation discrepancy policy configuration."""
from dataclasses import dataclass
from enum import Enum


class DiscrepancyAction(Enum):
    IMPORT = "import"
    CANCEL = "cancel"
    MARK_CANCELLED = "mark_cancelled"
    ADJUST_POSITION = "adjust_position"
    HALT_AND_ALERT = "halt_and_alert"


@dataclass
class DiscrepancyPolicy:
    """Config-driven policy for reconciliation mismatches."""

    unknown_order_with_our_tag: DiscrepancyAction = DiscrepancyAction.IMPORT
    unknown_order_orphan: DiscrepancyAction = DiscrepancyAction.CANCEL
    oms_working_broker_missing: DiscrepancyAction = DiscrepancyAction.MARK_CANCELLED
    unexpected_position: DiscrepancyAction = DiscrepancyAction.HALT_AND_ALERT
    position_qty_mismatch: DiscrepancyAction = DiscrepancyAction.ADJUST_POSITION
