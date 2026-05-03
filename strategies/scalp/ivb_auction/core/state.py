from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp._shared.session import ScalpSessionBlock
from strategies.scalp.ivb_auction.config import EntryTrigger, IvbAuctionPhase, IvbModule, TradeDirection
from strategies.scalp.ivb_auction.models import AbsorptionEvent, FootprintBarData, IvbSetup, ScalpPosition


@dataclass(slots=True)
class IvbAuctionCoreState:
    phase: IvbAuctionPhase = IvbAuctionPhase.S0_PRE_OPEN
    ivb_levels: IVBLevels | None = None
    break_direction: TradeDirection = TradeDirection.FLAT
    break_price: float = 0.0
    break_bar_idx: int = -1
    positions: dict[str, ScalpPosition] = field(default_factory=dict)
    active_setups: dict[str, IvbSetup] = field(default_factory=dict)
    order_to_setup: dict[str, str] = field(default_factory=dict)
    order_kind: dict[str, str] = field(default_factory=dict)
    cumulative_delta: float = 0.0
    absorption_events: list[AbsorptionEvent] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_losses: int = 0
    daily_risk_used: float = 0.0
    last_signal_score: float = 0.0
    last_signal_module: str = ""
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None
    bar_index: int = 0


@dataclass(slots=True)
class IvbBarInput:
    symbol: str
    bar_ts: datetime
    bar_ohlcv: tuple[float, float, float, float, float]
    session_block: ScalpSessionBlock
    ivb_levels: IVBLevels | None = None
    breakout_direction: TradeDirection = TradeDirection.FLAT
    breakout_accepted: bool = False
    module: IvbModule | None = None
    trigger: EntryTrigger | None = None
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    qty: int = 0
    rr_to_tp1: float = 0.0
    signal_score: float = 0.0
    size_multiplier: float = 0.0
    footprint_state: FootprintBarData | None = None
    depth_available: bool = False
    depth_confirmed: bool | None = None
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IvbEntryRequest:
    client_order_id: str
    setup: IvbSetup
    order_type: Literal["MARKET", "LIMIT", "STOP"] = "LIMIT"
    tif: str = "DAY"


@dataclass(slots=True)
class IvbStopUpdateRequest:
    setup_id: str
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class IvbPartialExitRequest:
    client_order_id: str
    setup_id: str
    symbol: str
    qty: int
    reason: str
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None
    tif: str = "DAY"


@dataclass(slots=True)
class IvbFlattenRequest:
    setup_id: str
    symbol: str
    reason: str


@dataclass(slots=True)
class IvbFill:
    oms_order_id: str
    fill_price: float = 0.0
    fill_qty: int = 0
    symbol: str = ""
    fill_time: datetime | None = None
    commission: float = 0.0
    order_role: Literal["entry", "stop", "tp1", "tp2", "partial", "flatten", "unknown"] = "unknown"
    exit_type: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IvbOrderUpdate:
    oms_order_id: str
    status: str = ""
    symbol: str = ""
    timestamp: datetime | None = None
    order_role: Literal["entry", "stop", "tp1", "tp2", "partial", "flatten", "unknown"] = "unknown"
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)
