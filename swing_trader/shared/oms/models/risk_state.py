"""Risk state models."""
from dataclasses import dataclass
from datetime import date


@dataclass
class StrategyRiskState:
    strategy_id: str
    trade_date: date
    daily_realized_pnl: float = 0.0
    daily_realized_R: float = 0.0
    open_risk_dollars: float = 0.0
    open_risk_R: float = 0.0
    halted: bool = False
    halt_reason: str = ""


@dataclass
class PortfolioRiskState:
    trade_date: date
    daily_realized_pnl: float = 0.0
    daily_realized_R: float = 0.0
    open_risk_dollars: float = 0.0
    open_risk_R: float = 0.0
    pending_entry_risk_R: float = 0.0  # Risk from working ENTRY orders not yet filled
    halted: bool = False
    halt_reason: str = ""
