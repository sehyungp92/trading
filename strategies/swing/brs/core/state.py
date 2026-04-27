from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.swing.brs.models import (
    BDArmState,
    BRSRegime,
    BiasState,
    DailyContext,
    EntrySignal,
    EntryType,
    LHArmState,
    S2ArmState,
    S3ArmState,
)
from strategies.swing.brs.positions import BRSPositionState, PendingOrder


@dataclass(slots=True)
class BRSCoreState:
    daily_ctx: dict[str, DailyContext] = field(default_factory=dict)
    bias: dict[str, BiasState] = field(default_factory=dict)
    prev_regime_on: dict[str, bool] = field(default_factory=dict)
    prev_regime: dict[str, BRSRegime] = field(default_factory=dict)
    position: dict[str, BRSPositionState | None] = field(default_factory=dict)
    cooldown_until: dict[str, datetime] = field(default_factory=dict)
    lh_arm: dict[str, LHArmState] = field(default_factory=dict)
    bd_arm: dict[str, BDArmState] = field(default_factory=dict)
    s2_arm: dict[str, S2ArmState] = field(default_factory=dict)
    s3_arm: dict[str, S3ArmState] = field(default_factory=dict)
    swing_highs: dict[str, list[float]] = field(default_factory=dict)
    short_bias_no_trade: dict[str, int] = field(default_factory=dict)
    hourly_bar_count: dict[str, int] = field(default_factory=dict)
    last_daily_close: dict[str, float] = field(default_factory=dict)
    pending_orders: dict[str, PendingOrder] = field(default_factory=dict)
    filled_order_ids: set[str] = field(default_factory=set)
    closing: dict[str, str] = field(default_factory=dict)
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class BRSEntryRequest:
    client_order_id: str
    symbol: str
    signal: EntrySignal
    qty: int
    pos_id: str
    tif: str = "IOC"
    order_type: Literal["LIMIT"] = "LIMIT"


@dataclass(slots=True)
class BRSAddOnRequest:
    client_order_id: str
    symbol: str
    signal: EntrySignal
    qty: int
    pos_id: str
    tif: str = "IOC"
    order_type: Literal["LIMIT"] = "LIMIT"


@dataclass(slots=True)
class BRSStopUpdateRequest:
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class BRSScaleOutRequest:
    client_order_id: str
    symbol: str
    qty: int
    limit_price: float
    pos_id: str
    tif: str = "IOC"
    order_type: Literal["LIMIT"] = "LIMIT"


@dataclass(slots=True)
class BRSFlattenRequest:
    symbol: str
    reason: str


@dataclass(slots=True)
class BRSEntryFillContext:
    cooldown_until: datetime | None = None
    reset_arm_state: EntryType | None = None
    reset_short_bias: bool = True
    tranche_b_qty: int = 0


@dataclass(slots=True)
class BRSOrderUpdate:
    oms_order_id: str
    status: str
    timestamp: datetime | None = None
    order_role: Literal["entry", "pyramid", "scale_out", "stop", "flatten", "unknown"] = "unknown"
    symbol: str = ""
    accepted_order: PendingOrder | None = None


@dataclass(slots=True)
class BRSFill:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    fill_time: datetime | None = None
    commission: float = 0.0
    symbol: str = ""
    exit_type: str | None = None
    entry_context: BRSEntryFillContext | None = None
