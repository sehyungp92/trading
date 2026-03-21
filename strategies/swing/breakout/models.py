"""Multi-Asset Swing Breakout v3.3-ETF — enums and dataclasses."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum, IntEnum
from typing import Optional, Dict, List, Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CampaignState(str, Enum):
    """Campaign state machine states (spec §4)."""
    INACTIVE = "INACTIVE"
    COMPRESSION = "COMPRESSION"
    BREAKOUT = "BREAKOUT"
    POSITION_OPEN = "POSITION_OPEN"
    CONTINUATION = "CONTINUATION"
    DIRTY = "DIRTY"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class EntryType(str, Enum):
    A_AVWAP_RETEST = "A"
    B_SWEEP_RECLAIM = "B"
    C_STANDARD = "C_standard"
    C_CONTINUATION = "C_continuation"
    ADD = "ADD"
    BREAKOUT_DAY = "BREAKOUT_DAY"


class SetupState(str, Enum):
    NEW = "NEW"
    ARMED = "ARMED"
    TRIGGERED = "TRIGGERED"
    FILLED = "FILLED"
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


class ExitTier(str, Enum):
    ALIGNED = "ALIGNED"
    NEUTRAL = "NEUTRAL"
    CAUTION = "CAUTION"


class Regime4H(str, Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    RANGE_CHOP = "RANGE_CHOP"


class TradeRegime(str, Enum):
    """Trade regime for exit tier determination."""
    ALIGNED = "ALIGNED"
    NEUTRAL = "NEUTRAL"
    CAUTION = "CAUTION"


class ChopMode(str, Enum):
    """Graduated chop response mode (spec §7.2)."""
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    HALT = "HALT"


# ---------------------------------------------------------------------------
# Pending Entry (spec §19)
# ---------------------------------------------------------------------------

@dataclass
class PendingEntry:
    """Pending entry snapshot for re-check mechanism."""
    created_ts: datetime
    direction: Direction
    entry_type_requested: EntryType
    snapshot_quality: Dict[str, Any] = field(default_factory=dict)
    snapshot_risk: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    rth_hours_elapsed: int = 0


# ---------------------------------------------------------------------------
# Symbol Campaign (spec §4)
# ---------------------------------------------------------------------------

@dataclass
class SymbolCampaign:
    """Per-symbol campaign state machine."""
    # State
    state: CampaignState = CampaignState.INACTIVE
    campaign_id: int = 0
    box_version: int = 0
    
    # Frozen box (compression detection)
    box_high: float = 0.0
    box_low: float = 0.0
    box_mid: float = 0.0
    box_height: float = 0.0
    anchor_time: Optional[datetime] = None  # next RTH open of activation day
    box_age_days: int = 0
    L: int = 12  # adaptive box length
    
    # Breakout state
    breakout_direction: Optional[Direction] = None
    breakout_date: Optional[date] = None
    bars_since_breakout: int = 0
    expiry_bars: int = 0
    hard_expiry_bars: int = 0
    expiry_mult: float = 1.0
    continuation: bool = False
    inside_close_count: int = 0  # consecutive closes back inside box
    chop_mode: str = "NORMAL"  # cached for hourly engine

    # DIRTY state
    dirty_start_date: Optional[date] = None
    dirty_high: Optional[float] = None
    dirty_low: Optional[float] = None
    L_at_dirty: Optional[int] = None
    
    # Counters
    reentry_count: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: {"LONG": {}, "SHORT": {}}
    )
    add_count: int = 0
    campaign_risk_used: float = 0.0  # dollars
    
    # Pending entry
    pending: Optional[PendingEntry] = None

    # Re-entry tracking (spec §21)
    last_exit_date: Optional[date] = None
    last_exit_direction: Optional[Direction] = None
    last_exit_realized_r: float = 0.0
    
    # Daily context (cached for hourly engine)
    daily_context: Optional[Dict[str, Any]] = None
    
    # Hysteresis for L selection
    L_bucket_current: Optional[str] = None       # committed bucket
    L_bucket_bars_held: int = 0
    L_bucket_candidate: Optional[str] = None      # candidate being evaluated
    L_bucket_candidate_bars: int = 0              # consecutive bars in candidate


# ---------------------------------------------------------------------------
# Daily Context (cached from daily close routine)
# ---------------------------------------------------------------------------

@dataclass
class DailyContext:
    """Cached daily indicators for hourly decisions."""
    # Displacement
    disp: float = 0.0
    disp_th: float = 0.0
    disp_mult: float = 1.0
    displacement_pass: bool = False
    
    # Volume
    rvol_d: float = 1.0
    vol_score: int = 0
    
    # Squeeze
    sq_good: bool = False
    sq_loose: bool = False
    squeeze_metric: float = 1.0
    
    # ATR
    atr_expanding: bool = False
    atr14_d: float = 0.0
    atr50_d: float = 0.0
    risk_regime: float = 1.0
    
    # Regime
    regime_4h: Regime4H = Regime4H.RANGE_CHOP
    hard_blocked_direction: Optional[Direction] = None
    
    # Breakout
    direction: Optional[Direction] = None
    structural_breakout: bool = False
    breakout_rejected: bool = False
    
    # Quality
    quality_mult: float = 1.0
    score_total: int = 0
    
    # Timestamps
    bar_date: Optional[date] = None
    computed_ts: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Hourly State
# ---------------------------------------------------------------------------

@dataclass
class HourlyState:
    """Per-symbol hourly bar state."""
    atr14_h: float = 0.0
    avwap_h: float = 0.0
    wvwap: float = 0.0
    ema20_h: float = 0.0
    rvol_h: float = 1.0
    close: float = 0.0
    bar_time: Optional[datetime] = None
    
    # Rolling data for indicators
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Setup Instance
# ---------------------------------------------------------------------------

@dataclass
class SetupInstance:
    """Central tracking object for a trade setup lifecycle."""
    # Identity
    setup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    direction: Direction = Direction.FLAT
    entry_type: EntryType = EntryType.A_AVWAP_RETEST
    state: SetupState = SetupState.NEW
    created_ts: Optional[datetime] = None
    armed_ts: Optional[datetime] = None
    
    # Campaign reference
    campaign_id: int = 0
    box_version: int = 0
    
    # Entry levels
    entry_price: float = 0.0
    stop0: float = 0.0
    buffer: float = 0.0
    
    # Risk
    final_risk_dollars: float = 0.0
    quality_mult: float = 1.0
    expiry_mult: float = 1.0
    shares_planned: int = 0
    
    # OCA
    oca_group: str = ""
    primary_order_id: str = ""
    stop_order_id: str = ""
    tp1_order_id: str = ""
    tp2_order_id: str = ""
    
    # TTL
    expiry_ts: Optional[datetime] = None
    triggered_ts: Optional[datetime] = None
    
    # Fill
    fill_price: float = 0.0
    fill_qty: int = 0
    fill_ts: Optional[datetime] = None
    avg_entry: float = 0.0
    r_price: float = 0.0  # dollar risk per share for R calc
    
    # Position management
    qty_open: int = 0
    current_stop: float = 0.0
    trail_active: bool = False
    trail_mult: float = 4.5
    
    # Partial fills
    tp1_done: bool = False
    tp2_done: bool = False
    runner_active: bool = False
    
    # Adds
    is_add: bool = False
    add_count: int = 0
    
    # Exit tier
    exit_tier: ExitTier = ExitTier.NEUTRAL
    
    # Regime at entry
    regime_at_entry: Optional[str] = None
    
    # Stale tracking
    bars_held: int = 0
    days_held: int = 0
    
    # R state
    realized_pnl: float = 0.0
    r_state: float = 0.0
    
    # Excursion tracking
    mfe_price: float = 0.0
    mae_price: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0

    # Recording
    trade_id: str = ""
    
    # Gap handling
    gap_stop_event: bool = False
    gap_stop_fill_price: Optional[float] = None

    # Filter decision telemetry (populated at entry time)
    filter_decisions: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Position State
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Live position tracking."""
    symbol: str = ""
    direction: Direction = Direction.FLAT
    qty: int = 0
    avg_cost: float = 0.0
    current_stop: float = 0.0
    unrealized_pnl: float = 0.0
    r_state: float = 0.0
    
    # Campaign reference
    campaign_id: int = 0
    box_version: int = 0
    
    # Exit tracking
    tp1_done: bool = False
    tp2_done: bool = False
    runner_active: bool = False
    trail_active: bool = False
    trail_mult: float = 4.5
    
    # Add tracking
    add_count: int = 0
    total_risk_dollars: float = 0.0

    # Regime at entry (for correlation multiplier alignment check)
    regime_at_entry: Optional[str] = None


# ---------------------------------------------------------------------------
# Circuit Breaker State
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreakerState:
    """Portfolio-level circuit breakers."""
    weekly_realized_r: float = 0.0
    weekly_start_date: Optional[date] = None
    monthly_realized_r: float = 0.0
    monthly_start_date: Optional[date] = None
    throttled_until: Optional[datetime] = None
    halted: bool = False


# ---------------------------------------------------------------------------
# Rolling Histories
# ---------------------------------------------------------------------------

@dataclass
class RollingHistories:
    """Per-symbol rolling histories for past-only quantiles."""
    squeeze_hist: List[float] = field(default_factory=list)
    disp_hist: List[float] = field(default_factory=list)
    atr_hist: List[float] = field(default_factory=list)
    
    # Slot volume medians for RVOL_H
    slot_medians: Dict[str, Dict[int, float]] = field(
        default_factory=lambda: {}  # {(dow, hour_slot): median_volume}
    )
    
    max_size: int = 1000
    
    def add_squeeze(self, value: float) -> None:
        self.squeeze_hist.append(value)
        if len(self.squeeze_hist) > self.max_size:
            self.squeeze_hist = self.squeeze_hist[-self.max_size:]
    
    def add_disp(self, value: float) -> None:
        self.disp_hist.append(value)
        if len(self.disp_hist) > self.max_size:
            self.disp_hist = self.disp_hist[-self.max_size:]
    
    def add_atr(self, value: float) -> None:
        self.atr_hist.append(value)
        if len(self.atr_hist) > self.max_size:
            self.atr_hist = self.atr_hist[-self.max_size:]


# ---------------------------------------------------------------------------
# Trade Record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Completed trade record for telemetry."""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    direction: Direction = Direction.FLAT
    entry_type: EntryType = EntryType.A_AVWAP_RETEST
    exit_tier: ExitTier = ExitTier.NEUTRAL
    
    # Campaign
    campaign_id: int = 0
    box_version: int = 0
    
    # Timing
    entry_ts: Optional[datetime] = None
    exit_ts: Optional[datetime] = None
    bars_held: int = 0
    days_held: int = 0
    
    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    
    # Size
    shares: int = 0
    
    # PnL
    realized_pnl: float = 0.0
    r_multiple: float = 0.0
    
    # Context at entry
    disp_at_entry: float = 0.0
    disp_th_at_entry: float = 0.0
    rvol_d_at_entry: float = 0.0
    rvol_h_at_entry: float = 0.0
    regime_at_entry: str = ""
    sq_good_at_entry: bool = False
    quality_mult_at_entry: float = 1.0
    
    # Adds
    add_count: int = 0
    add_contribution_r: float = 0.0
    
    # Gap events
    gap_stop_event: bool = False
    
    # Friction
    friction_dollars: float = 0.0
    friction_bps: float = 0.0
