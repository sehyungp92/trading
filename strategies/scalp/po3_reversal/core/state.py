from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.scalp.po3_reversal.config import Po3Phase, SetupTier, TradeDirection
from strategies.scalp.po3_reversal.fvg import FairValueGap
from strategies.scalp.po3_reversal.models import Po3Context, Po3Position, Po3Setup, SmtDivergence, SweepEvent


@dataclass(slots=True)
class Po3ReversalCoreState:
    phase: Po3Phase = Po3Phase.IDLE
    context: Po3Context = field(default_factory=Po3Context)
    active_setup: Po3Setup | None = None
    position: Po3Position | None = None
    order_to_setup: dict[str, str] = field(default_factory=dict)
    order_kind: dict[str, str] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    trades_today: int = 0
    full_losses_today: int = 0
    last_signal_score: float = 0.0
    last_tier: str = ""
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class Po3BarInput:
    symbol: str
    bar_ts: datetime
    bar_ohlcv: tuple[float, float, float, float, float]
    context: Po3Context
    sweep: SweepEvent | None = None
    smt: SmtDivergence | None = None
    ifvg: FairValueGap | None = None
    signal_score: float = 0.0
    signal_threshold: float = 0.0
    tier: SetupTier = SetupTier.NONE
    direction: TradeDirection = TradeDirection.FLAT
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    qty: int = 0
    rr: float = 0.0
    depth_available: bool = False
    depth_confirmed: bool | None = None
    risk_approved: bool = True
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Po3EntryRequest:
    client_order_id: str
    setup: Po3Setup
    order_type: Literal["STOP", "MARKET", "LIMIT"] = "STOP"
    tif: str = "DAY"


@dataclass(slots=True)
class Po3StopUpdateRequest:
    setup_id: str
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class Po3FlattenRequest:
    setup_id: str
    symbol: str
    reason: str


@dataclass(slots=True)
class Po3OrderUpdate:
    oms_order_id: str
    status: str = ""
    symbol: str = ""
    timestamp: datetime | None = None
    order_role: Literal["entry", "stop", "target", "flatten", "unknown"] = "unknown"
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Po3Fill:
    oms_order_id: str
    fill_price: float = 0.0
    fill_qty: int = 0
    symbol: str = ""
    fill_time: datetime | None = None
    commission: float = 0.0
    order_role: Literal["entry", "stop", "target", "flatten", "unknown"] = "unknown"
    exit_type: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)

