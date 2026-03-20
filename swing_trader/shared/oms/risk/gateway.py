"""Risk gateway for pre-trade checks."""
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional, TYPE_CHECKING

from ..models.order import OMSOrder, OrderRole
from ..models.risk_state import StrategyRiskState, PortfolioRiskState
from ..config.risk_config import RiskConfig
from .calendar import EventCalendar

if TYPE_CHECKING:
    from shared.market_calendar import MarketCalendar

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
        market_calendar: Optional["MarketCalendar"] = None,
    ):
        self._config = config
        self._calendar = calendar
        self._get_strat_risk = get_strategy_risk
        self._get_port_risk = get_portfolio_risk
        self._get_working_count = get_working_order_count
        self._market_cal = market_calendar

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

        now_utc = datetime.now(timezone.utc)

        # 1. Global stand-down
        if self._config.global_standdown:
            return "Global stand-down active"

        # 2. Event blackout
        if self._calendar.is_blocked(now_utc):
            return "Event blackout active"

        # 2b. Market holiday / half-day block
        if self._market_cal:
            from shared.market_calendar import AssetClass
            instrument = order.instrument
            asset_class = (
                AssetClass.CME_FUTURES
                if instrument and instrument.venue in ("CME", "COMEX", "NYMEX")
                else AssetClass.EQUITY
            )
            holiday_block = self._market_cal.is_entry_blocked(now_utc, asset_class)
            if holiday_block:
                return holiday_block

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

        # 5.5 Max working orders
        if self._get_working_count and strat_cfg.max_working_orders > 0:
            working = await self._get_working_count(order.strategy_id)
            if working >= strat_cfg.max_working_orders:
                return f"Max working orders ({strat_cfg.max_working_orders}) reached: {working} active"

        # 6. Heat cap check
        instrument = order.instrument
        if not instrument:
            return "Order missing instrument"

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

        # 6b. Per-strategy heat ceiling: prevent one strategy from monopolising
        # the shared pool. Soft cap — portfolio heat cap is still the hard limit.
        if strat_cfg.max_heat_R > 0:
            strat_heat_R = strat_risk.open_risk_R + new_risk_R
            if strat_heat_R > strat_cfg.max_heat_R:
                return (
                    f"Strategy heat ceiling: {order.strategy_id} open "
                    f"{strat_risk.open_risk_R:.2f}R + new {new_risk_R:.2f}R "
                    f"> cap {strat_cfg.max_heat_R:.2f}R"
                )

        # Priority-aware heat reservation: when remaining heat is tight,
        # reserve capacity for higher-priority strategies that are IDLE
        # (no open exposure). Strategies already deployed don't need reservation.
        remaining_R = self._config.heat_cap_R - (port_risk.open_risk_R + port_risk.pending_entry_risk_R)
        if remaining_R < 2 * new_risk_R:
            for other_cfg in self._config.strategy_configs.values():
                if other_cfg.priority < strat_cfg.priority:
                    other_risk = await self._get_strat_risk(other_cfg.strategy_id)
                    if other_risk.open_risk_R == 0:
                        return (
                            f"Heat cap reserved: {remaining_R:.2f}R remaining, "
                            f"priority strategy {other_cfg.strategy_id} may need it"
                        )

        # 7. Order type allowed
        if not strat_cfg.is_order_type_allowed(order.role, order.order_type):
            return f"Order type {order.order_type} not allowed for role {order.role}"

        # Store computed risk for pending-entry tracking
        risk_ctx.risk_dollars = new_risk_dollars

        return None  # Approved
