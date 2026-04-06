"""BRS data models — enums, dataclasses, and type aliases."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BRSRegime(Enum):
    """5-state regime classification (spec §2.3)."""
    BEAR_STRONG = "BEAR_STRONG"
    BEAR_TREND = "BEAR_TREND"
    BEAR_FORMING = "BEAR_FORMING"
    BULL_TREND = "BULL_TREND"
    RANGE_CHOP = "RANGE_CHOP"


class EntryType(Enum):
    """Entry signal types (spec §6)."""
    S1_PULLBACK = "PULLBACK"        # Pullback to EMA in bear trend
    S2_BREAKDOWN = "BREAKDOWN"      # Breakdown-pullback after campaign break
    S3_IMPULSE = "IMPULSE"          # Impulse continuation in BEAR_STRONG
    L1_GLD_LONG = "GLD_LONG"        # GLD defensive long
    LH_REJECTION = "LH_REJECTION"       # Lower-High Rejection
    BD_CONTINUATION = "BD_CONTINUATION"  # Breakdown Continuation


class Direction(Enum):
    """Trade direction."""
    LONG = 1
    SHORT = -1
    FLAT = 0

    def __int__(self) -> int:
        return self.value


class ExitReason(Enum):
    """Exit trigger categories."""
    STOP = "STOP"
    CATASTROPHIC = "CATASTROPHIC"
    BREAK_EVEN = "BREAK_EVEN"
    CHANDELIER = "CHANDELIER"
    STALE_EARLY = "STALE_EARLY"
    STALE = "STALE"
    TIME_DECAY = "TIME_DECAY"
    REGIME_TIGHTEN = "REGIME_TIGHTEN"


class Regime4H(Enum):
    """4-hour timeframe regime (spec §2.4)."""
    BEAR = "BEAR_4H"
    BULL = "BULL_4H"
    CHOP = "CHOP_4H"


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BiasState:
    """Tracks confirmed bias with hold counts (spec §2.6)."""
    raw_direction: Direction = Direction.FLAT
    confirmed_direction: Direction = Direction.FLAT
    hold_count: int = 0
    bear_score: float = 0.0
    adx: float = 0.0
    di_diff: float = 0.0
    ema_sep_pct: float = 0.0
    crash_override: bool = False  # bypasses regime_on on extreme daily drops
    peak_drop_override: bool = False  # forced SHORT from peak drawdown trigger
    cum_return_override: bool = False  # confirmed SHORT via cumulative return (Path G)
    prev_short_hold: int = 0    # hold_count before FLAT interruption (R8 churn bridge)
    flat_streak: int = 0         # consecutive FLAT bars since last SHORT (R8 churn bridge)


@dataclass
class VolState:
    """Volatility engine state (spec §3)."""
    vol_factor: float = 1.0
    vol_pct: float = 50.0
    risk_regime: float = 1.0
    base_risk_adj: float = 1.0
    extreme_vol: bool = False


@dataclass
class DailyContext:
    """Computed daily-bar context carried into hourly loop."""
    regime: BRSRegime = BRSRegime.RANGE_CHOP
    regime_on: bool = False
    regime_4h: Regime4H = Regime4H.CHOP
    bias: BiasState = field(default_factory=BiasState)
    vol: VolState = field(default_factory=VolState)
    bear_conviction: float = 0.0
    # Indicator values
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_fast_slope: float = 0.0
    ema_slow_slope: float = 0.0
    ema_sep_pct: float = 0.0
    atr14_d: float = 0.0
    atr50_d: float = 0.0
    atr_ratio: float = 1.0  # ATR14/ATR50 — vol expansion indicator
    close: float = 0.0
    persistence_active: bool = False  # R8: SHORT-bias drought exceeds threshold


@dataclass
class HourlyContext:
    """Computed hourly-bar context for signal evaluation."""
    close: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    prior_high: float = 0.0
    prior_low: float = 0.0
    ema_pull: float = 0.0
    ema_mom: float = 0.0
    atr14_h: float = 0.0
    avwap_h: float = 0.0
    volume: float = 0.0
    volume_sma20: float = 0.0
    prior_close_3: float = 0.0  # close 3 bars ago (for V-reversal filter)


@dataclass
class EntrySignal:
    """A generated entry signal ready for submission."""
    entry_type: EntryType
    direction: Direction
    signal_price: float
    signal_high: float
    signal_low: float
    stop_price: float
    risk_per_unit: float
    bear_conviction: float = 0.0
    quality_score: float = 0.0
    regime_at_entry: BRSRegime = BRSRegime.RANGE_CHOP
    vol_factor: float = 1.0


@dataclass
class S2ArmState:
    """Tracks S2 breakdown-pullback arming (spec §8.1)."""
    armed: bool = False
    armed_until_bar: int = 0
    box_high: float = 0.0
    box_low: float = 0.0
    avwap_anchor: float = 0.0


@dataclass
class S3ArmState:
    """Tracks S3 impulse continuation arming (spec §9.1)."""
    armed: bool = False
    armed_until_bar: int = 0


@dataclass
class LHArmState:
    """Tracks Lower-High Rejection arming."""
    armed: bool = False
    swing_high_price: float = 0.0
    prior_swing_high_price: float = 0.0
    armed_bar: int = 0
    armed_until_bar: int = 0


@dataclass
class BDArmState:
    """Tracks Breakdown Continuation arming."""
    armed: bool = False
    breakdown_bar_high: float = 0.0
    armed_bar: int = 0
    armed_until_bar: int = 0
