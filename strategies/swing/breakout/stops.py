"""Multi-Asset Swing Breakout v3.3-ETF — stop calculation and lifecycle.

Mid/edge selection, add stops, BE, trailing, gap handling.
"""
from __future__ import annotations

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from .config import (
    ADD_STOP_ATR_MULT,
    BE_BUFFER_ATR_MULT,
    EMA_4H_FLOOR_ATR_MULT,
    STALE_EXIT_DAYS_MAX,
    STALE_EXIT_DAYS_MIN,
    STALE_R_THRESH,
    STALE_TIGHTEN_MULT,
    STALE_WARN_DAYS,
    TRAIL_4H_ATR_MULT,
    SymbolConfig,
)
from .models import Direction


# ---------------------------------------------------------------------------
# Initial stop selection (spec §16.1)
# ---------------------------------------------------------------------------

def compute_initial_stop(
    direction: Direction,
    entry_type: str,
    box_high: float,
    box_low: float,
    box_mid: float,
    atr14_d: float,
    atr_stop_mult: float,
    sq_good: bool,
    tick_size: float = 0.01,
) -> float:
    """Compute initial stop based on entry type and squeeze quality.

    A1 fix: Always anchor from box_mid to reduce R-unit size.
    Previously only used box_mid when sq_good=True; now box_mid is default.
    Entry B uses box edge for wider protection on sweep entries.
    """
    buffer = atr_stop_mult * atr14_d

    # Entry B keeps edge stop (sweep entries need wider protection)
    if entry_type == "B":
        if direction == Direction.LONG:
            raw = box_low - buffer
        else:
            raw = box_high + buffer
    else:
        # A1: All other entries use box_mid anchor
        if direction == Direction.LONG:
            raw = box_mid - buffer
        else:
            raw = box_mid + buffer

    rounding = "down" if direction == Direction.LONG else "up"
    return round_to_tick(raw, tick_size, rounding)


# ---------------------------------------------------------------------------
# Add stop (spec §13.4)
# ---------------------------------------------------------------------------

def compute_add_stop(
    direction: Direction,
    pullback_low: float,
    pullback_high: float,
    ref: float,
    atr14_d: float,
    atr_stop_mult: float,
    tick_size: float = 0.01,
) -> float:
    """Add stop: min(pullback_low, ref) - 0.5*ATR14_D*atr_mult (long); mirror short."""
    buffer = ADD_STOP_ATR_MULT * atr14_d * atr_stop_mult

    if direction == Direction.LONG:
        anchor = min(pullback_low, ref)
        raw = anchor - buffer
        return round_to_tick(raw, tick_size, "down")
    else:
        anchor = max(pullback_high, ref)
        raw = anchor + buffer
        return round_to_tick(raw, tick_size, "up")


# ---------------------------------------------------------------------------
# BE stop after TP1 (spec §20.2)
# ---------------------------------------------------------------------------

def compute_be_stop(
    direction: Direction,
    avg_entry: float,
    atr14_d: float,
    tick_size: float = 0.01,
) -> float:
    """BE + small buffer (0.1*ATR14_D) after TP1."""
    buffer = BE_BUFFER_ATR_MULT * atr14_d
    if direction == Direction.LONG:
        return round_to_tick(avg_entry + buffer, tick_size, "up")
    else:
        return round_to_tick(avg_entry - buffer, tick_size, "down")


# ---------------------------------------------------------------------------
# Runner trailing stop (spec §20.3)
# ---------------------------------------------------------------------------

def compute_trailing_stop(
    direction: Direction,
    highs_4h: list[float],
    lows_4h: list[float],
    atr14_4h: float,
    trail_mult: float,
    ema50_4h: float,
    current_stop: float,
    tick_size: float = 0.01,
) -> float:
    """4H ATR trailing stop with ratchet-only behavior.

    Trail distance = trail_mult * ATR14_4H from highest/lowest.
    Optional EMA50_4H floor.
    """
    if direction == Direction.LONG:
        if len(highs_4h) == 0:
            return current_stop
        hh = max(highs_4h[-20:]) if len(highs_4h) >= 20 else max(highs_4h)
        raw = hh - trail_mult * atr14_4h
        # EMA50_4H floor
        ema_floor = ema50_4h - EMA_4H_FLOOR_ATR_MULT * atr14_4h
        raw = max(raw, ema_floor)
        candidate = round_to_tick(raw, tick_size, "down")
        # Ratchet only — never lower the stop
        return max(candidate, current_stop)
    else:
        if len(lows_4h) == 0:
            return current_stop
        ll = min(lows_4h[-20:]) if len(lows_4h) >= 20 else min(lows_4h)
        raw = ll + trail_mult * atr14_4h
        ema_ceil = ema50_4h + EMA_4H_FLOOR_ATR_MULT * atr14_4h
        raw = min(raw, ema_ceil)
        candidate = round_to_tick(raw, tick_size, "up")
        # Ratchet only — never raise the stop (for shorts, stop goes down)
        return min(candidate, current_stop)


# ---------------------------------------------------------------------------
# Trail multiplier tightening (spec §20.3)
# ---------------------------------------------------------------------------

def compute_trail_mult(
    r_state: float,
    r_proxy: float,
    continuation: bool,
) -> float:
    """Trail distance tightens after MM or high R.

    Base: TRAIL_4H_ATR_MULT * 5.0 ATR14_4H (was 9.0), tightens as R grows.
    """
    base = TRAIL_4H_ATR_MULT * 5.0  # 2.0 with 0.40 mult (was 3.6)
    # Tighten after measured move (earlier threshold)
    if r_proxy >= 1.0 or continuation:  # was 2.0
        base *= 0.70
    if r_state >= 2.0:                  # was 4.0
        base *= 0.60
    return max(1.0, base)               # was 1.5


# ---------------------------------------------------------------------------
# Gap-through-stop handling (spec §16.2)
# ---------------------------------------------------------------------------

def handle_gap_through_stop(
    direction: Direction,
    stop_price: float,
    session_open: float,
) -> tuple[bool, float]:
    """Check if session open gaps through stop.

    Returns (gap_stop_triggered, fill_price).
    Fill at open price (adverse) if triggered.
    """
    if direction == Direction.LONG:
        if session_open < stop_price:
            return True, session_open
    else:
        if session_open > stop_price:
            return True, session_open
    return False, 0.0


# ---------------------------------------------------------------------------
# Stale exit + tighten warning (spec §20.4)
# ---------------------------------------------------------------------------

def check_stale_exit(
    days_held: int,
    r_state: float,
) -> tuple[bool, bool, bool]:
    """Check stale conditions.

    Returns (should_warn, should_tighten, should_exit).
    Lowered threshold: only stale-exit trades below STALE_R_THRESH (0.0).
    Gives more time (STALE_EXIT_DAYS_MIN = 10) for trades to develop.
    Hard cap at STALE_EXIT_DAYS_MAX to prevent indefinite holds.
    """
    should_warn = days_held >= STALE_WARN_DAYS and r_state < STALE_R_THRESH
    should_tighten = should_warn
    # Normal stale exit: below threshold after min days
    should_exit = days_held >= STALE_EXIT_DAYS_MIN and r_state < STALE_R_THRESH
    # Hard cap: exit after max days regardless (prevents infinite holds near zero)
    if days_held >= STALE_EXIT_DAYS_MAX and r_state < 0.10:
        should_exit = True
    return should_warn, should_tighten, should_exit


def apply_stale_tighten(trail_mult: float) -> float:
    """Tighten trail multiplier by 0.8× for stale positions."""
    return trail_mult * STALE_TIGHTEN_MULT


# ---------------------------------------------------------------------------
# TP levels (spec §20.2)
# ---------------------------------------------------------------------------

def compute_tp_levels(
    direction: Direction,
    entry_price: float,
    risk_per_share: float,
    tp1_r: float,
    tp2_r: float,
    tick_size: float = 0.01,
) -> tuple[float, float]:
    """Compute TP1 and TP2 prices."""
    if direction == Direction.LONG:
        tp1 = entry_price + tp1_r * risk_per_share
        tp2 = entry_price + tp2_r * risk_per_share
    else:
        tp1 = entry_price - tp1_r * risk_per_share
        tp2 = entry_price - tp2_r * risk_per_share
    return (
        round_to_tick(tp1, tick_size, "up" if direction == Direction.LONG else "down"),
        round_to_tick(tp2, tick_size, "up" if direction == Direction.LONG else "down"),
    )
