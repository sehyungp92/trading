"""Risk configuration models."""
from dataclasses import dataclass, field
from typing import Optional
from ..models.order import OrderRole, OrderType


@dataclass
class StrategyRiskConfig:
    strategy_id: str
    daily_stop_R: float = 2.0
    unit_risk_pct: float = 0.0035  # e.g. 0.35%
    vol_factor: float = 1.0
    unit_risk_dollars: float = 0.0  # computed at runtime from NAV
    max_working_orders: int = 4
    priority: int = 99  # lower = higher priority (0=ATRSS, 1=Helix, 2=Breakout)
    max_heat_R: float = 0.0  # per-strategy heat ceiling (0 = no limit, uses portfolio cap)
    allowed_order_types: dict[str, list[str]] = field(default_factory=dict)
    # e.g. {"ENTRY": ["STOP_LIMIT"], "EXIT": ["MARKET", "STOP"]}
    session_block_rules: dict = field(default_factory=dict)
    # e.g. {"no_entry_first_minutes": 10, "no_entry_last_minutes": 10}

    def is_order_type_allowed(self, role: OrderRole, order_type: OrderType) -> bool:
        allowed = self.allowed_order_types.get(role.value, [])
        return order_type.value in allowed if allowed else True

    def check_session_block(self, now_utc) -> Optional[str]:
        """Override in subclass or implement via session_block_rules."""
        return None  # Strategy-layer responsibility in practice


@dataclass
class RiskConfig:
    heat_cap_R: float = 1.25
    portfolio_daily_stop_R: float = 3.0
    global_standdown: bool = False
    strategy_configs: dict[str, StrategyRiskConfig] = field(default_factory=dict)
