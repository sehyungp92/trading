from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from strategies.scalp._shared.levels import DealingRange

from .config import EntryType, SetupTier, TradeDirection


@dataclass(frozen=True, slots=True)
class PriceBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body_percent(self) -> float:
        rng = max(self.high - self.low, 0.0)
        return abs(self.close - self.open) / rng if rng > 0 else 0.0


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    symbol: str
    side: str
    price: float
    touches: int
    first_index: int
    last_index: int
    source: str = "fractal"


@dataclass(frozen=True, slots=True)
class SweepEvent:
    swept: bool
    direction: TradeDirection = TradeDirection.FLAT
    pool: LiquidityPool | None = None
    sweep_price: float = 0.0
    distance_ticks: float = 0.0


@dataclass(frozen=True, slots=True)
class SmtDivergence:
    present: bool
    direction: TradeDirection = TradeDirection.FLAT
    strength: float = 0.0
    nq_extreme: float = 0.0
    es_extreme: float = 0.0
    lag_bars: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Po3Context:
    daily_bias: TradeDirection = TradeDirection.FLAT
    h4_bias: TradeDirection = TradeDirection.FLAT
    h1_bias: TradeDirection = TradeDirection.FLAT
    dealing_range: DealingRange | None = None
    h1_target: float = 0.0


@dataclass(slots=True)
class Po3Setup:
    setup_id: str
    symbol: str
    direction: TradeDirection
    tier: SetupTier
    entry_type: EntryType
    signal_time: datetime
    score: float
    entry_price: float
    stop_price: float
    target_price: float
    qty: int
    rr: float
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_order_id: str = ""
    stop_order_id: str = ""
    target_order_id: str = ""
    qty_open: int = 0
    avg_entry: float = 0.0
    state: str = "NEW"


@dataclass(slots=True)
class Po3Position:
    setup_id: str
    symbol: str
    direction: TradeDirection
    qty: int
    avg_entry: float
    stop_price: float
    target_price: float
    opened_at: datetime
    initial_risk_points: float
    break_even_set: bool = False


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
    tier: str = ""
    entry_type: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0

