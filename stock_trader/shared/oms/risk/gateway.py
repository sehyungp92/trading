"""Risk gateway for pre-trade checks."""
import math
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional

from ..models.order import OMSOrder, OrderRole
from ..models.risk_state import StrategyRiskState, PortfolioRiskState
from ..config.risk_config import RiskConfig
from .calendar import EventCalendar
from .portfolio_rules import PortfolioRuleChecker

logger = logging.getLogger(__name__)


class RiskGateway:
    """Pre-trade risk checks. Only gates ENTRY orders; exits always allowed."""

    def __init__(
        self,
        config: RiskConfig,
        calendar: EventCalendar,
        get_strategy_risk: Callable[[str], Awaitable[StrategyRiskState]],
        get_portfolio_risk: Callable[[], Awaitable[PortfolioRiskState]],
        get_working_order_count: Callable[[str], Awaitable[int]] = None,
        portfolio_checker: Optional[PortfolioRuleChecker] = None,
    ):
        self._config = config
        self._calendar = calendar
        self._get_strat_risk = get_strategy_risk
        self._get_port_risk = get_portfolio_risk
        self._get_working_count = get_working_order_count
        self._portfolio_checker = portfolio_checker

    @staticmethod
    def _validate_entry_order(order: OMSOrder) -> Optional[str]:
        """Reject malformed entries before they can distort live risk state."""
        if order.qty <= 0:
            return f"ENTRY qty must be positive: {order.qty}"

        instrument = order.instrument
        if not instrument:
            return "Order missing instrument"
        if not math.isfinite(instrument.point_value) or instrument.point_value <= 0:
            return f"Instrument point_value must be positive: {instrument.point_value}"

        risk_ctx = order.risk_context
        if risk_ctx is None:
            return "ENTRY order missing risk_context"

        planned_entry = risk_ctx.planned_entry_price
        stop_for_risk = risk_ctx.stop_for_risk
        if not math.isfinite(planned_entry) or not math.isfinite(stop_for_risk):
            return "ENTRY risk prices must be finite"
        if planned_entry <= 0 or stop_for_risk <= 0:
            return (
                "ENTRY risk prices must be positive: "
                f"entry={planned_entry} stop={stop_for_risk}"
            )

        risk_per_contract = abs(planned_entry - stop_for_risk) * instrument.point_value
        if not math.isfinite(risk_per_contract) or risk_per_contract <= 0:
            return (
                "ENTRY risk distance must be positive: "
                f"entry={planned_entry} stop={stop_for_risk}"
            )

        return None

    async def check_entry(self, order: OMSOrder) -> Optional[str]:
        """Returns denial reason string, or None if approved."""
        # Exits/stops always allowed
        if order.role != OrderRole.ENTRY:
            return None

        # Must have risk context for entries
        if not order.risk_context:
            return "ENTRY order missing risk_context"

        strat_cfg = self._config.strategy_configs.get(order.strategy_id)
        if not strat_cfg:
            return f"No risk config for strategy {order.strategy_id}"

        validation_error = self._validate_entry_order(order)
        if validation_error:
            return validation_error
        if not math.isfinite(strat_cfg.unit_risk_dollars) or strat_cfg.unit_risk_dollars <= 0:
            return (
                "Strategy unit_risk_dollars must be positive: "
                f"{strat_cfg.unit_risk_dollars}"
            )

        now_utc = datetime.now(timezone.utc)

        # 1. Global stand-down
        if self._config.global_standdown:
            return "Global stand-down active"

        # 2. Event blackout
        if self._calendar.is_blocked(now_utc):
            return "Event blackout active"

        # 3. Market session block (strategy-specific)
        session_block = strat_cfg.check_session_block(now_utc)
        if session_block:
            return session_block

        # 4. Strategy daily halt
        strat_risk = await self._get_strat_risk(order.strategy_id)
        if strat_risk.halted:
            return f"Strategy halted: {strat_risk.halt_reason}"
        if strat_risk.daily_realized_R <= -strat_cfg.daily_stop_R:
            return (
                f"Strategy daily stop: realized {strat_risk.daily_realized_R:.2f}R "
                f"<= -{strat_cfg.daily_stop_R}R"
            )

        # 5. Portfolio daily halt
        port_risk = await self._get_port_risk()
        if port_risk.halted:
            return f"Portfolio halted: {port_risk.halt_reason}"
        if port_risk.daily_realized_R <= -self._config.portfolio_daily_stop_R:
            return f"Portfolio daily stop: {port_risk.daily_realized_R:.2f}R"

        # 5a. Portfolio weekly halt (consolidated cross-strategy P&L)
        if (self._config.portfolio_weekly_stop_R > 0
                and port_risk.weekly_realized_R <= -self._config.portfolio_weekly_stop_R):
            return (
                f"Portfolio weekly stop: {port_risk.weekly_realized_R:.2f}R "
                f"<= -{self._config.portfolio_weekly_stop_R}R"
            )

        # 5.5 Max working orders
        if self._get_working_count and strat_cfg.max_working_orders > 0:
            working = await self._get_working_count(order.strategy_id)
            if working >= strat_cfg.max_working_orders:
                return f"Max working orders ({strat_cfg.max_working_orders}) reached: {working} active"

        # 6. Heat cap check
        instrument = order.instrument
        risk_ctx = order.risk_context
        risk_per_contract = (
            abs(risk_ctx.planned_entry_price - risk_ctx.stop_for_risk)
            * instrument.point_value
        )
        new_risk_dollars = order.qty * risk_per_contract
        new_risk_R = (
            new_risk_dollars / strat_cfg.unit_risk_dollars
            if strat_cfg.unit_risk_dollars > 0
            else float("inf")
        )

        total_risk_R = port_risk.open_risk_R + port_risk.pending_entry_risk_R + new_risk_R
        if total_risk_R > self._config.heat_cap_R:
            return (
                f"Heat cap breach: open {port_risk.open_risk_R:.2f}R + "
                f"pending {port_risk.pending_entry_risk_R:.2f}R + "
                f"new {new_risk_R:.2f}R > cap {self._config.heat_cap_R}R"
            )

        # 7. Order type allowed
        if not strat_cfg.is_order_type_allowed(order.role, order.order_type):
            return f"Order type {order.order_type} not allowed for role {order.role}"

        # 8. Cross-strategy portfolio rules
        if self._portfolio_checker:
            direction = "LONG" if order.side.value == "BUY" else "SHORT"
            port_result = await self._portfolio_checker.check_entry(
                strategy_id=order.strategy_id,
                direction=direction,
                new_risk_R=new_risk_R,
            )
            if not port_result.approved:
                return f"Portfolio rule: {port_result.denial_reason}"
            # Apply size multiplier to risk context for downstream sizing
            if port_result.size_multiplier != 1.0:
                risk_ctx.portfolio_size_mult = port_result.size_multiplier
                logger.info(
                    "Portfolio size multiplier %.2fx applied to %s %s",
                    port_result.size_multiplier, order.strategy_id, direction,
                )

        # Store computed risk for pending-entry tracking
        risk_ctx.risk_dollars = new_risk_dollars

        return None  # Approved
