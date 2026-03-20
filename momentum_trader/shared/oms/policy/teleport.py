"""Teleport detection for entry orders."""
from ..models.order import OMSOrder


class TeleportChecker:
    """Checks if price has 'teleported' past the entry without fill.
    If so, the order should be cancelled (no chasing).
    """

    @staticmethod
    def should_cancel(order: OMSOrder, current_price: float) -> bool:
        if not order.entry_policy or order.entry_policy.teleport_ticks is None:
            return False
        if not order.instrument:
            return False

        tick_size = order.instrument.tick_size
        teleport_dist = order.entry_policy.teleport_ticks * tick_size

        # Use limit_price (expected fill) or fall back to planned_entry_price
        ref_price = order.limit_price
        if ref_price is None and order.risk_context:
            ref_price = order.risk_context.planned_entry_price
        if ref_price is None:
            return False

        if order.side.value == "BUY":
            # For long entry: if price > entry_limit + teleport, cancel
            return current_price > ref_price + teleport_dist
        else:
            # For short entry: if price < entry_limit - teleport, cancel
            return current_price < ref_price - teleport_dist
