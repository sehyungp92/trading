from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import EntryTrigger, IvbModule, TradeDirection


@dataclass(frozen=True, slots=True)
class ScalpTick:
    ts: datetime
    price: float
    size: float
    bid: float | None = None
    ask: float | None = None
    side: int = 0


@dataclass(frozen=True, slots=True)
class AbsorptionEvent:
    ts: datetime
    direction: TradeDirection
    price: float
    aggression_volume: float
    price_progress: float
    score: float


@dataclass(frozen=True, slots=True)
class FootprintBarData:
    start_ts: datetime
    end_ts: datetime
    ask_volume: float
    bid_volume: float
    delta: float
    cumulative_delta: float
    aggression_ratio: float
    imbalance_ratio: float
    absorption_score: float
    large_trade_count: int = 0


@dataclass(slots=True)
class IvbSetup:
    setup_id: str
    symbol: str
    module: IvbModule
    direction: TradeDirection
    trigger: EntryTrigger
    signal_time: datetime
    score: float
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    qty: int
    size_multiplier: float
    rr_to_tp1: float
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_order_id: str = ""
    stop_order_id: str = ""
    tp1_order_id: str = ""
    tp2_order_id: str = ""
    qty_open: int = 0
    avg_entry: float = 0.0
    state: str = "NEW"


@dataclass(slots=True)
class ScalpPosition:
    setup_id: str
    symbol: str
    module: IvbModule
    direction: TradeDirection
    qty: int
    avg_entry: float
    current_stop: float
    tp1_price: float
    tp2_price: float
    opened_at: datetime
    initial_risk_points: float
    tp1_filled: bool = False


@dataclass(slots=True)
class TradeRecord:
    symbol: str
    side: str
    qty: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    pnl_dollars: float
    r_multiple: float
    exit_reason: str
    setup_id: str = ""
    module: str = ""
    trigger: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0

