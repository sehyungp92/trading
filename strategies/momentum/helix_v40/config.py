"""AKC-Helix NQ TrendWrap v4.0 Apex Trail — configuration, enums, and dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, auto
from typing import Optional

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry

# ── Identity ────────────────────────────────────────────────────────
STRATEGY_ID = "AKC_Helix_v40"
DEFAULT_SYMBOL = os.environ.get("TRADING_SYMBOL", "NQ")

# ── Contract ────────────────────────────────────────────────────────
NQ_SPECS = {
    "NQ":  {"tick": 0.25, "tick_value": 5.00, "point_value": 20.00},
    "MNQ": {"tick": 0.25, "tick_value": 0.50, "point_value":  2.00},
}
_ACTIVE_SPEC = NQ_SPECS[DEFAULT_SYMBOL]
NQ_TICK = _ACTIVE_SPEC["tick"]
NQ_TICK_VALUE = _ACTIVE_SPEC["tick_value"]
NQ_POINT_VALUE = _ACTIVE_SPEC["point_value"]

# ── Position limits ────────────────────────────────────────────────
MAX_CONCURRENT_POSITIONS = 3
DUPLICATE_OVERRIDE_MIN_R = 0.25

# ── Risk ────────────────────────────────────────────────────────────
BASE_RISK_PCT = 0.0035
DAILY_STOP_R = 2.0                  # v7: aligned with backtest (was 1.5 in live)
HEAT_CAP_R = 3.00                   # single tier (was 3.50/4.00/2.00)
HEAT_CAP_DIR_R = 2.50               # single tier (was 3.00)
PORTFOLIO_DAILY_STOP_R = 2.25
EXTREME_VOL_RISK_MULT = 0.80
STRONG_TREND_BONUS = 1.10

# ── Volatility ──────────────────────────────────────────────────────
VOL_PERCENTILE_WINDOW = 60
VOL_FACTOR_MIN = 0.40
VOL_FACTOR_MAX = 1.50
EXTREME_VOL_THRESHOLD = 95
LOW_VOL_THRESHOLD = 20
CHOP_ZONE_VOL_LOW = 20
CHOP_ZONE_VOL_HIGH = 50
HIGH_VOL_M_THRESHOLD = 97

# ── MACD ────────────────────────────────────────────────────────────
MACD_FAST = 8
MACD_SLOW = 21
MACD_SIGNAL = 5

# ── EMAs ────────────────────────────────────────────────────────────
EMA_FAST_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14

# ── Pivots ──────────────────────────────────────────────────────────
PIVOT_BARS = 5

# ── Trend ───────────────────────────────────────────────────────────
STRONG_TREND_THRESHOLD = 0.80

# ── Extension ───────────────────────────────────────────────────────
EXTENSION_ATR_MULT = 2.0

# ── Catastrophic stop ───────────────────────────────────────────────
CATASTROPHIC_STOP_R: float = -1.5

# ── Stop ────────────────────────────────────────────────────────────
MIN_STOP_NORMAL = 5.0
MIN_STOP_ATR_FRAC = 0.35
MIN_STOP_EXTREME = 7.5
MIN_STOP_ATR_FRAC_EXTREME = 0.50
SPIKE_FILTER_ATR_MULT = 4.0

# ── Class M ─────────────────────────────────────────────────────────
CLASS_M_STOP_ATR_MULT = 0.50
CLASS_M_PB_MIN = 0.2
CLASS_M_PB_MAX = 1.6
CLASS_M_MOMENTUM_LOOKBACK = 3
ENTRY_BUFFER_ATR_FRAC = 0.05
ENTRY_BUFFER_MIN_PTS = 0.25

# ── Class F (Fast Re-entry) ────────────────────────────────────────
CLASS_F_WINDOW_BARS = 8              # bars after trailing exit to detect re-entry
CLASS_F_MIN_PB_ATR = 0.25           # min pullback depth in ATR1H
CLASS_F_MAX_PB_ATR = 1.0            # max pullback depth in ATR1H
CLASS_F_STOP_ATR_MULT = 0.40        # tighter stop for re-entries
CLASS_F_SIZE_MULT = 0.50            # half-size
CLASS_F_MIN_ALIGNMENT_SCORE = 0
CLASS_F_MIN_CONTEXT_SCORE = -99
CLASS_F_REQUIRE_STRONG_TREND = False
CLASS_F_MIN_VOL_PCT = 0
CLASS_F_BLOCK_ETH_QUALITY_AM = False
CLASS_F_BLOCK_RTH_PRIME1 = False

# ── Class T (4H Trend Continuation) ──────────────────────────────
CLASS_T_TREND_STRENGTH_MIN = 0.60    # daily EMA separation / ATR
CLASS_T_HIST_LOOKBACK = 1            # 4H MACD histogram expansion lookback
CLASS_T_MACD_1H_LOOKBACK = 2         # 1H MACD line rising lookback
CLASS_T_PIVOT_RECENCY_H = 6          # max hours since most recent 1H pivot high
CLASS_T_COMPRESSION_MAX = 1.2        # atr_rolling(6) / atr_rolling(20) cap
CLASS_T_STOP_ATR_MULT = 0.40         # tighter stop than M's 0.50
CLASS_T_MAX_STOP_DIST_ATR = 2.0      # reject if stop > 2 ATR away
CLASS_T_SIZE_MULT = 0.50             # half-size until validated
CLASS_T_MIN_BARS_SINCE_M = 4         # bars since last M placement same direction
CLASS_T_SECONDARY_ENABLED = False
CLASS_T_SECONDARY_ALLOW_RTH_PRIME1 = False
CLASS_T_SECONDARY_ALLOW_RTH_PRIME2 = False
CLASS_T_SECONDARY_ALLOW_ETH_QUALITY_PM = False
CLASS_T_SECONDARY_TREND_STRENGTH_MIN = 0.35
CLASS_T_SECONDARY_PIVOT_RECENCY_H = 18
CLASS_T_SECONDARY_COMPRESSION_MAX = 1.75
CLASS_T_SECONDARY_MIN_ALIGNMENT_SCORE = 1
CLASS_T_SECONDARY_MIN_CONTEXT_SCORE = -99
CLASS_T_SECONDARY_MIN_HIST_RATIO = 0.70
CLASS_T_SECONDARY_MAX_BREAKOUT_GAP_ATR = 0.65
CLASS_T_SECONDARY_STOP_ATR_MULT = 0.35
CLASS_T_SECONDARY_MAX_STOP_DIST_ATR = 2.40
CLASS_T_SHORT_ENABLED = False
CLASS_T_SHORT_ALLOW_ETH_QUALITY_PM = False
CLASS_T_SHORT_ALLOW_RTH_PRIME2 = False
CLASS_T_SHORT_TREND_STRENGTH_MIN = 0.25
CLASS_T_SHORT_PIVOT_RECENCY_H = 12
CLASS_T_SHORT_COMPRESSION_MAX = 1.60
CLASS_T_SHORT_MIN_ALIGNMENT_SCORE = 1
CLASS_T_SHORT_MIN_CONTEXT_SCORE = -99
CLASS_T_SHORT_MIN_HIST_RATIO = 0.80
CLASS_T_SHORT_MAX_BREAKOUT_GAP_ATR = 0.65
CLASS_T_SHORT_STOP_ATR_MULT = 0.35
CLASS_T_SHORT_MAX_STOP_DIST_ATR = 2.20

# ── SetupSizeMult ───────────────────────────────────────────────────
SIZE_MULT_M = {2: 0.65, 1: 0.65, 0: 0.65}  # flattened
SIZE_MULT_M_STRONG = 1.10
SIZE_MULT_M_CHOP = 0.40             # chop zone sizing penalty (was hard block, now sizing)
REENTRY_SIZE_PENALTY = 0.85
CHOP_LONG_QUALITY_GATE = False
CHOP_LONG_MIN_SCORE = 2
CHOP_LONG_REQUIRE_STRONG_TREND = True
NOON_QUALITY_GATE = False
NOON_QUALITY_HOUR_ET = 12
NOON_QUALITY_MIN_SCORE = 2
NOON_QUALITY_REQUIRE_STRONG_TREND = True
HIGH_VOL_CONTINUATION_REOPEN_ENABLED = False
HIGH_VOL_CONTINUATION_MAX_VOL_PCT = 95
HIGH_VOL_CONTINUATION_MIN_ALIGNMENT_SCORE = 2
HIGH_VOL_CONTINUATION_REQUIRE_STRONG_TREND = True
SHORT_SCORE0_REOPEN_ENABLED = False
SHORT_SCORE0_ALLOW_RTH_DEAD = False
SHORT_SCORE0_ALLOW_RTH_PRIME2 = False
SHORT_SCORE0_ALLOW_ETH_QUALITY_PM = False
SHORT_SCORE0_MIN_VOL_PCT = 50
SHORT_SCORE0_MAX_VOL_PCT = 80
SHORT_MIN_SCORE = 3

# ── TTL (seconds) ───────────────────────────────────────────────────
TTL_1H_S = 12 * 3600
TTL_CATCHUP_S = 5 * 60

# ── Execution ───────────────────────────────────────────────────────
CATCHUP_OVERSHOOT_ATR_FRAC = 2.00
CATCHUP_OFFSET_PTS = 0.50
FOLLOWTHROUGH_CATCHUP_ENABLED = True
FOLLOWTHROUGH_ALLOW_CLASS_M = True
FOLLOWTHROUGH_ALLOW_CLASS_T = True
FOLLOWTHROUGH_ALLOW_RTH_PRIME1 = True
FOLLOWTHROUGH_ALLOW_RTH_PRIME2 = False
FOLLOWTHROUGH_ALLOW_ETH_QUALITY_PM = True
FOLLOWTHROUGH_MIN_ALIGNMENT_SCORE = 1
FOLLOWTHROUGH_REQUIRE_STRONG_TREND = False
FOLLOWTHROUGH_MAX_GAP_ATR = 0.25
FOLLOWTHROUGH_MAX_BARS_SINCE_ARM = 2
FOLLOWTHROUGH_OFFSET_PTS = 0.25
FOLLOWTHROUGH_TTL_S = 60 * 60

# Structural entry expansion: second-chance rearms
SECOND_CHANCE_REARM_ENABLED = False
SECOND_CHANCE_ALLOW_CLASS_M = True
SECOND_CHANCE_ALLOW_CLASS_T = False
SECOND_CHANCE_ALLOW_ETH_QUALITY_AM = False
SECOND_CHANCE_ALLOW_RTH_PRIME1 = False
SECOND_CHANCE_ALLOW_RTH_PRIME2 = False
SECOND_CHANCE_ALLOW_ETH_QUALITY_PM = False
SECOND_CHANCE_MIN_ALIGNMENT_SCORE = 1
SECOND_CHANCE_REQUIRE_STRONG_TREND = False
SECOND_CHANCE_MAX_BARS_SINCE_CANCEL = 10
SECOND_CHANCE_MAX_BREAKOUT_GAP_ATR = 0.55
SECOND_CHANCE_STOP_ATR_MULT = 0.35
SECOND_CHANCE_MAX_STOP_DIST_ATR = 2.40
SECOND_CHANCE_PULLBACK_EMA_TOL_ATR = 0.35
SECOND_CHANCE_REQUIRE_FRESH_PIVOT = True
SECOND_CHANCE_USE_BASE_TRIGGER = False

# Structural entry expansion: dedicated high-vol continuation / retest
HIGH_VOL_RETEST_ENABLED = False
HIGH_VOL_RETEST_ALLOW_RTH_PRIME1 = False
HIGH_VOL_RETEST_ALLOW_RTH_PRIME2 = False
HIGH_VOL_RETEST_ALLOW_ETH_QUALITY_PM = False
HIGH_VOL_RETEST_MIN_VOL_PCT = 94
HIGH_VOL_RETEST_MAX_VOL_PCT = 101
HIGH_VOL_RETEST_MIN_ALIGNMENT_SCORE = 2
HIGH_VOL_RETEST_REQUIRE_STRONG_TREND = True
HIGH_VOL_RETEST_MIN_TREND_STRENGTH = 0.60
HIGH_VOL_RETEST_PIVOT_RECENCY_H = 10
HIGH_VOL_RETEST_MAX_PULLBACK_ATR = 1.25
HIGH_VOL_RETEST_MAX_BREAKOUT_GAP_ATR = 0.50
HIGH_VOL_RETEST_STOP_ATR_MULT = 0.35
HIGH_VOL_RETEST_MAX_STOP_DIST_ATR = 2.20
HIGH_VOL_RETEST_H4_HIST_RATIO_MIN = 0.70
HIGH_VOL_RETEST_EMA_HOLD_TOL_ATR = 0.35

# Structural entry expansion: selective high-vol Class M continuation
HIGH_VOL_M_CONT_ENABLED = False
HIGH_VOL_M_CONT_ALLOW_LONG = True
HIGH_VOL_M_CONT_ALLOW_SHORT = False
HIGH_VOL_M_CONT_ALLOW_RTH_PRIME1 = False
HIGH_VOL_M_CONT_ALLOW_RTH_PRIME2 = False
HIGH_VOL_M_CONT_ALLOW_RTH_DEAD = False
HIGH_VOL_M_CONT_ALLOW_ETH_QUALITY_PM = False
HIGH_VOL_M_CONT_MIN_VOL_PCT = 93
HIGH_VOL_M_CONT_MAX_VOL_PCT = 96
HIGH_VOL_M_CONT_MIN_ALIGNMENT_SCORE = 1
HIGH_VOL_M_CONT_MIN_CONTEXT_SCORE_LONG = -99
HIGH_VOL_M_CONT_MIN_CONTEXT_SCORE_SHORT = -99
HIGH_VOL_M_CONT_REQUIRE_STRONG_TREND = False
HIGH_VOL_M_CONT_MIN_TREND_STRENGTH = 0.35
HIGH_VOL_M_CONT_MAX_PULLBACK_ATR = 1.20
HIGH_VOL_M_CONT_MAX_BREAKOUT_GAP_ATR = 0.55
HIGH_VOL_M_CONT_STOP_ATR_MULT = 0.35
HIGH_VOL_M_CONT_MAX_STOP_DIST_ATR = 2.30
HIGH_VOL_M_CONT_EMA_HOLD_TOL_ATR = 0.35

# Structural entry expansion: normal-vol Class M short precision
NORM_VOL_M_SHORT_ENABLED = False
NORM_VOL_M_SHORT_ALLOW_RTH_DEAD = False
NORM_VOL_M_SHORT_ALLOW_RTH_PRIME2 = False
NORM_VOL_M_SHORT_ALLOW_ETH_QUALITY_PM = False
NORM_VOL_M_SHORT_MIN_VOL_PCT = 50
NORM_VOL_M_SHORT_MAX_VOL_PCT = 80
NORM_VOL_M_SHORT_MIN_ALIGNMENT_SCORE = 0
NORM_VOL_M_SHORT_REQUIRE_STRONG_TREND = False
NORM_VOL_M_SHORT_MIN_TREND_STRENGTH = 0.0
NORM_VOL_M_SHORT_MIN_CONTEXT_SCORE = 2
NORM_VOL_M_SHORT_MAX_BREAKOUT_GAP_ATR = 0.55
NORM_VOL_M_SHORT_STOP_ATR_MULT = 0.35
NORM_VOL_M_SHORT_MAX_STOP_DIST_ATR = 2.30

# Structural entry expansion: stronger reclaim-based Class F
CLASS_F_RECLAIM_ENABLED = False
CLASS_F_RECLAIM_ALLOW_RTH_PRIME1 = False
CLASS_F_RECLAIM_ALLOW_RTH_PRIME2 = False
CLASS_F_RECLAIM_ALLOW_ETH_QUALITY_PM = False
CLASS_F_RECLAIM_MIN_ALIGNMENT_SCORE = 1
CLASS_F_RECLAIM_MIN_CONTEXT_SCORE = -99
CLASS_F_RECLAIM_REQUIRE_STRONG_TREND = True
CLASS_F_RECLAIM_MIN_VOL_PCT = 20
CLASS_F_RECLAIM_MAX_EXIT_BARS = 8
CLASS_F_RECLAIM_MAX_PULLBACK_ATR = 1.25
CLASS_F_RECLAIM_EMA_HOLD_TOL_ATR = 0.25
CLASS_F_RECLAIM_STOP_ATR_MULT = 0.35
CLASS_F_RECLAIM_MAX_STOP_DIST_ATR = 2.10

# Structural entry expansion: trend retest continuation.
TREND_RETEST_ENABLED = False
TREND_RETEST_ALLOW_LONG = True
TREND_RETEST_ALLOW_SHORT = False
TREND_RETEST_ALLOW_RTH_PRIME1 = False
TREND_RETEST_ALLOW_RTH_PRIME2 = False
TREND_RETEST_ALLOW_RTH_DEAD = False
TREND_RETEST_ALLOW_ETH_QUALITY_PM = False
TREND_RETEST_MIN_VOL_PCT = 0
TREND_RETEST_MAX_VOL_PCT = 95
TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG = 1
TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT = 1
TREND_RETEST_MIN_CONTEXT_SCORE_LONG = 2
TREND_RETEST_MIN_CONTEXT_SCORE_SHORT = 2
TREND_RETEST_REQUIRE_STRONG_TREND = False
TREND_RETEST_MIN_TREND_STRENGTH = 0.35
TREND_RETEST_LOOKBACK_BARS = 4
TREND_RETEST_MIN_BARS_BETWEEN = 3
TREND_RETEST_PULLBACK_EMA_TOL_ATR = 0.30
TREND_RETEST_MAX_PULLBACK_ATR = 1.20
TREND_RETEST_MAX_BREAKOUT_GAP_ATR = 0.45
TREND_RETEST_STOP_ATR_MULT = 0.35
TREND_RETEST_MAX_STOP_DIST_ATR = 2.20

# Structural entry expansion: short failed-reclaim continuation.
FAILED_RECLAIM_SHORT_ENABLED = True
FAILED_RECLAIM_SHORT_ALLOW_RTH_DEAD = False
FAILED_RECLAIM_SHORT_ALLOW_RTH_PRIME2 = True
FAILED_RECLAIM_SHORT_ALLOW_ETH_QUALITY_PM = False
FAILED_RECLAIM_SHORT_MIN_VOL_PCT = 35
FAILED_RECLAIM_SHORT_MAX_VOL_PCT = 95
FAILED_RECLAIM_SHORT_MIN_ALIGNMENT_SCORE = 0
FAILED_RECLAIM_SHORT_MIN_CONTEXT_SCORE = 2
FAILED_RECLAIM_SHORT_REQUIRE_STRONG_TREND = False
FAILED_RECLAIM_SHORT_MIN_TREND_STRENGTH = 0.0
FAILED_RECLAIM_SHORT_LOOKBACK_BARS = 5
FAILED_RECLAIM_SHORT_MIN_BARS_BETWEEN = 3
FAILED_RECLAIM_SHORT_EMA_TOL_ATR = 0.35
FAILED_RECLAIM_SHORT_MAX_BREAKOUT_GAP_ATR = 0.55
FAILED_RECLAIM_SHORT_STOP_ATR_MULT = 0.35
FAILED_RECLAIM_SHORT_MAX_STOP_DIST_ATR = 2.30

# Structural entry expansion: completed 5m continuation retest.
MICRO_TREND_RETEST_ENABLED = True
MICRO_TREND_RETEST_ALLOW_LONG = True
MICRO_TREND_RETEST_ALLOW_SHORT = True
MICRO_TREND_RETEST_ALLOW_RTH_PRIME1 = True
MICRO_TREND_RETEST_ALLOW_RTH_PRIME2 = False
MICRO_TREND_RETEST_ALLOW_RTH_DEAD = True
MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_AM = False
MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_PM = True
MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME1 = True
MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME2 = False
MICRO_TREND_RETEST_ALLOW_LONG_RTH_DEAD = True
MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_AM = False
MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_PM = False
MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME1 = True
MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME2 = False
MICRO_TREND_RETEST_ALLOW_SHORT_RTH_DEAD = False
MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_AM = False
MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_PM = True
MICRO_TREND_RETEST_MIN_VOL_PCT = 20
MICRO_TREND_RETEST_MAX_VOL_PCT = 80
MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG = 1
MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT = 1
MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_LONG = 2
MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT = 2
MICRO_TREND_RETEST_REQUIRE_STRONG_TREND = False
MICRO_TREND_RETEST_MIN_TREND_STRENGTH = 0.1
MICRO_TREND_RETEST_LOOKBACK_BARS = 4
MICRO_TREND_RETEST_MIN_BARS_BETWEEN = 3
MICRO_TREND_RETEST_PULLBACK_EMA_TOL_ATR = 0.6
MICRO_TREND_RETEST_MAX_PULLBACK_ATR = 3.0
MICRO_TREND_RETEST_MAX_BREAKOUT_GAP_ATR = 1.1
MICRO_TREND_RETEST_STOP_M5_ATR_MULT = 1.40
MICRO_TREND_RETEST_STOP_H1_ATR_FRAC = 0.35
MICRO_TREND_RETEST_MAX_STOP_DIST_ATR = 1.85

# Structural entry expansion: completed 15m continuation retest.
M15_TREND_RETEST_ENABLED = True
M15_TREND_RETEST_ALLOW_LONG = True
M15_TREND_RETEST_ALLOW_SHORT = False
M15_TREND_RETEST_ALLOW_RTH_PRIME1 = True
M15_TREND_RETEST_ALLOW_RTH_PRIME2 = False
M15_TREND_RETEST_ALLOW_RTH_DEAD = True
M15_TREND_RETEST_ALLOW_ETH_QUALITY_AM = False
M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM = False
M15_TREND_RETEST_MIN_VOL_PCT = 0
M15_TREND_RETEST_MAX_VOL_PCT = 95
M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG = 1
M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT = 1
M15_TREND_RETEST_MIN_CONTEXT_SCORE_LONG = 2
M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT = 2
M15_TREND_RETEST_REQUIRE_STRONG_TREND = False
M15_TREND_RETEST_MIN_TREND_STRENGTH = 0.15
M15_TREND_RETEST_LOOKBACK_BARS = 5
M15_TREND_RETEST_MIN_BARS_BETWEEN = 4
M15_TREND_RETEST_PULLBACK_EMA_TOL_ATR = 0.55
M15_TREND_RETEST_MAX_PULLBACK_ATR = 2.6
M15_TREND_RETEST_MAX_BREAKOUT_GAP_ATR = 0.9
M15_TREND_RETEST_STOP_M15_ATR_MULT = 1.10
M15_TREND_RETEST_STOP_H1_ATR_FRAC = 0.30
M15_TREND_RETEST_MAX_STOP_DIST_ATR = 1.75

# Diagnostic guard: when enabled, structural signal variants are logged but not traded.
STRUCTURAL_VARIANT_SHADOW_ONLY = False

# ── Slippage / teleport ────────────────────────────────────────────
TELEPORT_RTH_TICKS = 4
TELEPORT_RTH_ATR_FRAC = 0.08
TELEPORT_ETH_TICKS = 6
TELEPORT_ETH_ATR_FRAC = 0.12
CATASTROPHIC_FILL_MULT = 3.0

# ── Position management — Partials (uniform) ───────────────────────
PARTIAL1_R = 1.0                     # take 30% at +1.0R (all scores)
PARTIAL1_FRAC = 0.30
PARTIAL2_R = 1.25
PARTIAL2_FRAC = 0.30
POST_PARTIAL_LOCK_R = 0.0            # 0 disables post-partial profit lock

# ── MFE ratchet floor ──────────────────────────────────────────────
MFE_RATCHET_ACTIVATION_R = 1.0
MFE_RATCHET_FLOOR_PCT = 0.55

# ── Trailing (single-phase) ────────────────────────────────────────
TRAIL_ACTIVATION_R = 0.55
TRAIL_LOOKBACK_1H = 30
TRAIL_MULT_MAX = 3.0
TRAIL_MULT_FLOOR = 1.75
TRAIL_R_DIVISOR = 6.0

# ── Early adverse ──────────────────────────────────────────────────
EARLY_ADVERSE_BARS = 5
EARLY_ADVERSE_R = -0.7

# ── Momentum stall ─────────────────────────────────────────────────
MOMENTUM_STALL_BAR = 8
MOMENTUM_STALL_MIN_R = 0.30
MOMENTUM_STALL_MIN_MFE = 0.80

# ── Time-decay stop tightening (single BE force) ───────────────────
TIME_DECAY_BARS = 14
TIME_DECAY_PROGRESS_R = 0.30

# ── Stale exits ────────────────────────────────────────────────────
EARLY_STALE_BARS = 12
EARLY_STALE_MAX_BARS = 24
STALE_M_BARS = 20                   # reverted to v3.1
STALE_R_THRESHOLD = 0.3

# ── Re-entry ───────────────────────────────────────────────────────
REENTRY_LOOKBACK_1H = 20

# ── Spread gates (points) ──────────────────────────────────────────
MAX_SPREAD_RTH = 1.25
MAX_SPREAD_ETH_QUALITY = 2.00
MAX_SPREAD_ETH_OVERNIGHT = 2.50
SPREAD_RECHECK_BARS = 2

# ── Roll ────────────────────────────────────────────────────────────
ROLL_VOLUME_RATIO = 2.0
ROLL_DAYS_BEFORE_EXPIRY = 5
ROLL_BLACKOUT_MINUTES = 30

# ── Sunday gap ──────────────────────────────────────────────────────
SUNDAY_GAP_ATR_FRAC = 0.30

# ── ETH_EUROPE regime expansion ────────────────────────────────────
ETH_EUROPE_REGIME_MIN_TREND = 0.80   # strongest trends only
ETH_EUROPE_REGIME_MAX_VOL = 50       # calm markets
ETH_EUROPE_REGIME_SIZE_MULT = 0.40   # reduced from existing 0.50

# ── RTH_DEAD enhanced sizing ──────────────────────────────────────
RTH_DEAD_ENHANCED_SIZE = 0.85        # up from 0.70 when conditions met
RTH_DEAD_ENHANCED_MIN_TREND = 0.50   # moderate trend strength required
RTH_DEAD_ENHANCED_MAX_VOL = 70       # below high vol

# ── Sizing modifiers ───────────────────────────────────────────────
VOL_50_80_SIZING_MULT: float = 0.4

# ── Day-of-week ────────────────────────────────────────────────────
DOW_BLOCKED = {0, 2}                # Monday=0, Wednesday=2
DOW_SIZE_MULT = {}                   # v7: removed Thursday halving — optimizer validated without it

# ── Hour-of-day sizing ─────────────────────────────────────────────
HOUR_SIZE_MULT = {13: 1.10, 17: 1.10, 12: 0.70}

# ── Drawdown throttle (earlier, more aggressive) ───────────────────
DRAWDOWN_THROTTLE_ENABLED = False  # v7: disabled — optimizer found throttle hurts net returns
DRAWDOWN_THROTTLE_LEVELS = [
    (0.05, 0.75),    # at 5% DD: 75% sizing (was 8%/75%)
    (0.08, 0.50),    # at 8% DD: 50% sizing (was 12%/50%)
    (0.11, 0.0),     # at 11% DD: stop trading (was 15%/0%)
]


# ── Enums ───────────────────────────────────────────────────────────

class TF(Enum):
    M5 = "5m"
    M15 = "15m"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"


class SetupClass(Enum):
    M = "M"
    F = "F"
    T = "T"


class SetupState(Enum):
    DETECTED = auto()
    ARMED = auto()
    PENDING = auto()
    FILLED = auto()
    MANAGED = auto()
    EXITED = auto()
    CANCELED = auto()
    EXPIRED = auto()


class SessionBlock(Enum):
    RTH_PRIME1 = "RTH_PRIME1"
    RTH_DEAD = "RTH_DEAD"
    RTH_PRIME2 = "RTH_PRIME2"
    ETH_QUALITY_AM = "ETH_QUALITY_AM"
    ETH_QUALITY_PM = "ETH_QUALITY_PM"
    DAILY_HALT = "DAILY_HALT"
    REOPEN_DEAD = "REOPEN_DEAD"
    ETH_OVERNIGHT = "ETH_OVERNIGHT"
    ETH_EUROPE = "ETH_EUROPE"


# ── Session schedule (ET) — RTH_DEAD narrowed to 11:30-12:30 ──────
SESSION_SCHEDULE: list[tuple[time, time, SessionBlock]] = [
    (time(3, 0), time(8, 0), SessionBlock.ETH_EUROPE),
    (time(8, 0), time(9, 35), SessionBlock.ETH_QUALITY_AM),
    (time(9, 35), time(11, 30), SessionBlock.RTH_PRIME1),
    (time(11, 30), time(12, 30), SessionBlock.RTH_DEAD),       # narrowed from 13:30
    (time(12, 30), time(15, 45), SessionBlock.RTH_PRIME2),     # expanded from 13:30
    (time(15, 45), time(17, 0), SessionBlock.ETH_QUALITY_PM),
    (time(17, 0), time(18, 0), SessionBlock.DAILY_HALT),
    (time(18, 0), time(20, 0), SessionBlock.REOPEN_DEAD),
    (time(20, 0), time(23, 59, 59), SessionBlock.ETH_OVERNIGHT),
    (time(0, 0), time(3, 0), SessionBlock.ETH_OVERNIGHT),
]

SESSION_SIZE_MULT = {
    SessionBlock.RTH_PRIME1: 1.00,
    SessionBlock.RTH_PRIME2: 1.00,
    SessionBlock.RTH_DEAD: 0.70,
    SessionBlock.ETH_QUALITY_AM: 0.50,
    SessionBlock.ETH_QUALITY_PM: 0.70,
    SessionBlock.ETH_EUROPE: 0.50,          # selective entry only, half-size
    SessionBlock.ETH_OVERNIGHT: 0.50,
}

MAX_SPREAD_BY_SESSION = {
    SessionBlock.RTH_PRIME1: MAX_SPREAD_RTH,
    SessionBlock.RTH_PRIME2: MAX_SPREAD_RTH,
    SessionBlock.RTH_DEAD: MAX_SPREAD_RTH,
    SessionBlock.ETH_QUALITY_AM: MAX_SPREAD_ETH_QUALITY,
    SessionBlock.ETH_QUALITY_PM: MAX_SPREAD_ETH_QUALITY,
    SessionBlock.ETH_EUROPE: MAX_SPREAD_ETH_QUALITY,
    SessionBlock.ETH_OVERNIGHT: MAX_SPREAD_ETH_OVERNIGHT,
}

# ── News windows (event_type -> (pre_minutes, post_minutes)) ──────
NEWS_WINDOWS = {
    "CPI": (60, 30),
    "NFP": (60, 30),
    "FOMC": (60, 60),
    "FED_SPEECH": (30, 30),
    "GDP": (30, 15),
    "PCE": (30, 15),
    "ISM": (15, 10),
}


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class Pivot:
    ts: datetime
    kind: str               # "PH" or "PL"
    price: float
    macd: float
    atr: float


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Setup:
    setup_id: str
    cls: SetupClass
    direction: int          # +1 long, -1 short
    tf_origin: TF
    detected_ts: datetime
    state: SetupState = SetupState.DETECTED

    # Structure anchors
    P1: Optional[Pivot] = None
    P2: Optional[Pivot] = None
    breakout_pivot: Optional[Pivot] = None

    entry_stop: float = 0.0
    stop0: float = 0.0
    buffer: float = 0.0

    alignment_score: int = 0
    strong_trend: bool = False
    is_extended: bool = False
    is_reentry: bool = False
    signal_variant: str = ""
    parent_setup_id: str = ""
    diagnostic_context: dict[str, object] = field(default_factory=dict)

    # Class F: reference to the trailing exit that triggered re-entry
    parent_exit_bar: int = 0

    placed_ts: Optional[datetime] = None
    ttl_expiry_ts: Optional[datetime] = None

    # OMS order IDs
    oca_group: str = ""
    entry_oms_id: Optional[str] = None
    catchup_oms_id: Optional[str] = None
    followthrough_used: bool = False

    # Sizing (computed at arm time)
    contracts: int = 0
    unit1_risk_usd: float = 0.0
    setup_size_mult: float = 0.0
    session_size_mult: float = 0.0
    armed_risk_r: float = 0.0

    def ttl_seconds(self) -> int:
        return TTL_1H_S


@dataclass
class PositionState:
    pos_id: str
    direction: int
    avg_entry: float
    contracts: int
    unit1_risk_usd: float
    origin_class: SetupClass
    origin_setup_id: str
    entry_ts: datetime

    stop_price: float
    trailing_active: bool = False
    teleport_penalty: bool = False

    # Milestones
    did_1r: bool = False
    partial_done: bool = False
    partial2_done: bool = False
    entry_contracts: int = 0

    # Stop order tracking
    stop_oms_id: Optional[str] = None

    # Stale tracking
    bars_held_1h: int = 0

    # Trailing state
    trail_mult: float = 0.0
    alignment_score_at_entry: int = 0
    current_alignment_score: int = 0
    peak_mfe_r: float = 0.0
    peak_mae_r: float = 0.0  # worst adverse excursion in R-multiples (always >= 0)
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float('inf')

    # Risk tracking
    current_risk_r: float = 0.0

    # R realized from partials
    realized_partial_usd: float = 0.0

    # Exit order tracking
    exit_oms_ids: list[str] = field(default_factory=list)
    pending_exit_estimates: dict[str, float] = field(default_factory=dict)

    # Session transition tracking (#17)
    session_transitions: list = field(default_factory=list)
    entry_session: str = ""
    _last_session: str = ""


# ── Instrument builder (OMS) ──────────────────────────────────────

def build_instruments() -> list[Instrument]:
    inst = Instrument(
        symbol=DEFAULT_SYMBOL, root=DEFAULT_SYMBOL, venue="CME",
        tick_size=NQ_TICK, tick_value=NQ_TICK_VALUE,
        multiplier=NQ_POINT_VALUE, currency="USD",
        point_value=NQ_POINT_VALUE, contract_expiry="",
    )
    InstrumentRegistry.register(inst)
    return [inst]
