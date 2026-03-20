"""Typed models for the live U.S. ORB strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import ET


class State(str, Enum):
    PRECHECK = "PRECHECK"
    CANDIDATE = "CANDIDATE"
    ARMED = "ARMED"
    WAIT_BREAK = "WAIT_BREAK"
    WAIT_ACCEPTANCE = "WAIT_ACCEPTANCE"
    READY = "READY"
    ORDER_SENT = "ORDER_SENT"
    IN_POSITION = "IN_POSITION"
    COOLDOWN = "COOLDOWN"
    DONE = "DONE"


class DangerState(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"
    BLOCKED = "BLOCKED"


@dataclass(slots=True, frozen=True)
class DangerSnapshot:
    state: DangerState = DangerState.SAFE
    score: int = 0
    reasons: tuple[str, ...] = ()
    cooldown_seconds: int = 0


@dataclass(slots=True)
class CachedSymbol:
    symbol: str
    exchange: str
    primary_exchange: str
    currency: str
    tick_size: float
    point_value: float
    adv20: float
    prior_close: float
    sma20: float
    sma60: float
    sma20_slope: float
    atr1m14: float
    opening_value15_baseline_0935_0950: float
    minute_volume_baseline_0935_1115: dict[str, float]
    sector: str = ""
    float_shares: float | None = None
    float_bucket: str = ""
    catalyst_tag: str = ""
    luld_tier: str = "tier_1"
    tech_tag: bool = False
    secondary_universe: bool = False

    @property
    def trend_ok(self) -> bool:
        return (
            self.prior_close > self.sma20
            and self.sma20 >= self.sma60
            and self.sma20_slope >= 0
        )

    @property
    def baseline_value15(self) -> float:
        return self.opening_value15_baseline_0935_0950

    @property
    def minute_volume_baseline(self) -> dict[str, float]:
        return self.minute_volume_baseline_0935_1115


@dataclass(slots=True)
class MinuteBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_value: float

    @property
    def minute_key(self) -> str:
        return self.ts.astimezone(ET).strftime("%H:%M")


@dataclass(slots=True)
class QuoteSnapshot:
    ts: datetime
    bid: float
    ask: float
    last: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    cumulative_volume: float = 0.0
    cumulative_value: float = 0.0
    vwap: float | None = None
    is_halted: bool = False
    past_limit: bool = False
    bid_past_low: bool = False
    ask_past_high: bool = False
    spread_pct: float = 0.0
    quote_gap_pct: float = 0.0
    midpoint_velocity: float = 0.0
    quote_expansion_streak: int = 0
    resumed_from_halt: bool = False


@dataclass(slots=True)
class AcceptanceState:
    break_time: datetime | None = None
    deadline: datetime | None = None
    pulled_back: bool = False
    held_support: bool = False
    reclaimed: bool = False
    retest_low: float | None = None

    def reset(self) -> None:
        self.break_time = None
        self.deadline = None
        self.pulled_back = False
        self.held_support = False
        self.reclaimed = False
        self.retest_low = None


@dataclass(slots=True)
class PendingOrderState:
    oms_order_id: str
    submitted_at: datetime
    role: str
    requested_qty: int
    limit_price: float | None = None
    stop_price: float | None = None
    cancel_requested: bool = False


@dataclass(slots=True)
class PositionState:
    entry_price: float
    qty_entry: int
    qty_open: int
    final_stop: float
    current_stop: float
    entry_time: datetime
    initial_risk_per_share: float
    max_favorable_price: float
    max_adverse_price: float
    partial_taken: bool = False
    stop_order_id: str = ""
    trade_id: str = ""
    realized_pnl_usd: float = 0.0
    entry_commission: float = 0.0

    @property
    def total_initial_risk_usd(self) -> float:
        return self.initial_risk_per_share * self.qty_entry


@dataclass(slots=True)
class SymbolContext:
    cached: CachedSymbol
    state: State = State.PRECHECK
    bars: deque[MinuteBar] = field(default_factory=lambda: deque(maxlen=240))
    quote: QuoteSnapshot | None = None

    or_high: float | None = None
    or_low: float | None = None
    or_mid: float | None = None
    or_pct: float | None = None
    value15: float = 0.0
    surge: float = 0.0

    gap_pct: float = 0.0
    spread_pct: float = 0.0
    vwap: float | None = None
    rvol_1m: float = 0.0
    imbalance_90s: float = 0.0
    relative_strength_5m: float = 0.0
    last5m_value: float = 0.0

    pre_score: float = 0.0
    quality_score: float = 0.0
    size_penalty: float = 1.0

    planned_entry: float | None = None
    planned_limit: float | None = None
    final_stop: float | None = None
    qty: int = 0

    acceptance: AcceptanceState = field(default_factory=AcceptanceState)
    entry_order: PendingOrderState | None = None
    exit_order: PendingOrderState | None = None
    position: PositionState | None = None
    vdm: DangerSnapshot = field(default_factory=DangerSnapshot)

    pending_hard_exit: bool = False
    rearms_used: int = 0
    cooldown_until: datetime | None = None
    block_reason: str = ""
    done_reason: str = ""
    was_halted: bool = False
    past_limit_events: deque[datetime] = field(default_factory=lambda: deque(maxlen=32))
    ask_past_high_events: deque[datetime] = field(default_factory=lambda: deque(maxlen=32))
    bid_past_low_events: deque[datetime] = field(default_factory=lambda: deque(maxlen=32))
    halt_events: deque[datetime] = field(default_factory=lambda: deque(maxlen=32))
    resume_events: deque[datetime] = field(default_factory=lambda: deque(maxlen=32))

    @property
    def symbol(self) -> str:
        return self.cached.symbol

    @property
    def sector(self) -> str:
        return self.cached.sector

    @property
    def last_price(self) -> float | None:
        if self.quote and self.quote.last > 0:
            return self.quote.last
        if self.bars:
            return self.bars[-1].close
        return None

    @property
    def spread(self) -> float:
        if self.quote and self.quote.ask > 0 and self.quote.bid > 0:
            return max(0.0, self.quote.ask - self.quote.bid)
        return 0.0


@dataclass(slots=True)
class RegimeSnapshot:
    regime_ok: bool = False
    risk_off: bool = False
    chop: bool = False
    leader_count: int = 0
    leader_breadth_ok: bool = False
    flow_regime: str = "mixed"
    spy_from_open: float = 0.0
    qqq_from_open: float = 0.0
    iwm_from_open: float = 0.0
    breadth_or_tick_negative_persistent: bool = False
    weak_directional_progress: bool = False
    repeated_vwap_crossings: bool = False
    poor_leader_follow_through: bool = False


@dataclass(slots=True)
class PortfolioSnapshot:
    nav: float
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    halt_new_entries: bool = False
    flatten_all: bool = False
    sectors_in_use: set[str] = field(default_factory=set)
    open_positions: int = 0

    @property
    def total_pnl_usd(self) -> float:
        return self.realized_pnl_usd + self.unrealized_pnl_usd

    @property
    def total_pnl_pct(self) -> float:
        if self.nav <= 0:
            return 0.0
        return self.total_pnl_usd / self.nav
