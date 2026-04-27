from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.swing.breakout.models import (
    CircuitBreakerState,
    Direction,
    EntryType,
    HourlyState,
    PositionState,
    RollingHistories,
    SetupInstance,
    SymbolCampaign,
)


@dataclass(slots=True)
class BreakoutCoreState:
    campaigns: dict[str, SymbolCampaign] = field(default_factory=dict)
    histories: dict[str, RollingHistories] = field(default_factory=dict)
    hourly_states: dict[str, HourlyState] = field(default_factory=dict)
    positions: dict[str, PositionState] = field(default_factory=dict)
    active_setups: dict[str, SetupInstance] = field(default_factory=dict)
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)
    correlation_map: dict[tuple[str, str], float] = field(default_factory=dict)
    order_to_setup: dict[str, str] = field(default_factory=dict)
    order_kind: dict[str, str] = field(default_factory=dict)
    order_requested_qty: dict[str, int] = field(default_factory=dict)
    oca_counter: int = 0
    risk_halted: bool = False
    risk_halt_reason: str = ""
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class BreakoutBarInput:
    symbol: str = ""
    timeframe: str = ""
    bar_ts: datetime | None = None
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BreakoutEntryRequest:
    client_order_id: str
    setup: SetupInstance
    order_type: Literal["MARKET", "LIMIT"] = "LIMIT"
    tif: str = "GTC"


@dataclass(slots=True)
class BreakoutStopUpdateRequest:
    setup_id: str
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class BreakoutPartialExitRequest:
    client_order_id: str
    setup_id: str
    symbol: str
    qty: int
    reason: str
    order_type: Literal["MARKET"] = "MARKET"
    tif: str = "GTC"


@dataclass(slots=True)
class BreakoutFlattenRequest:
    setup_id: str
    symbol: str
    reason: str


@dataclass(slots=True)
class BreakoutOrderUpdate:
    oms_order_id: str
    status: str = ""
    symbol: str = ""
    timestamp: datetime | None = None
    order_role: Literal["entry", "add", "partial", "stop", "flatten", "unknown"] = "unknown"
    timeframe: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BreakoutFill:
    oms_order_id: str
    fill_price: float = 0.0
    fill_qty: int = 0
    symbol: str = ""
    fill_time: datetime | None = None
    commission: float = 0.0
    order_role: Literal["entry", "add", "partial", "stop", "flatten", "unknown"] = "unknown"
    exit_type: str = ""
    timeframe: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)
