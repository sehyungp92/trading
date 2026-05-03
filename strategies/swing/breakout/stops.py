"""Multi-asset breakout stop and exit helpers."""
from __future__ import annotations

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from .config import (
    ADD_STOP_ATR_MULT,
    BE_BUFFER_ATR_MULT,
    CONTINUATION_BRANCH_MANAGEMENT_ENABLE,
    CONTINUATION_BRANCH_PRE_RUNNER_LOCK_FRAC,
    CONTINUATION_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R,
    CONTINUATION_BRANCH_TP1_PARTIAL_FRAC,
    CONTINUATION_BRANCH_TP1_R_ALIGNED,
    CONTINUATION_BRANCH_TP1_R_CAUTION,
    CONTINUATION_BRANCH_TP1_R_NEUTRAL,
    CONTINUATION_BRANCH_TRAIL_4H_ATR_MULT,
    CONTINUATION_BRANCH_TRAIL_MULT_BASE_FACTOR,
    EMA_4H_FLOOR_ATR_MULT,
    FAST_BRANCH_MANAGEMENT_ENABLE,
    FAST_BRANCH_PRE_RUNNER_LOCK_FRAC,
    FAST_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R,
    FAST_BRANCH_TP1_R_ALIGNED,
    FAST_BRANCH_TP1_R_CAUTION,
    FAST_BRANCH_TP1_R_NEUTRAL,
    FAST_BRANCH_TRAIL_4H_ATR_MULT,
    FAST_BRANCH_TRAIL_MULT_BASE_FACTOR,
    MOMENTUM_BRANCH_MANAGEMENT_ENABLE,
    MOMENTUM_BRANCH_PRE_RUNNER_LOCK_FRAC,
    MOMENTUM_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R,
    MOMENTUM_BRANCH_TP1_PARTIAL_FRAC,
    MOMENTUM_BRANCH_TP1_R_ALIGNED,
    MOMENTUM_BRANCH_TP1_R_CAUTION,
    MOMENTUM_BRANCH_TP1_R_NEUTRAL,
    MOMENTUM_BRANCH_TRAIL_4H_ATR_MULT,
    MOMENTUM_BRANCH_TRAIL_MULT_BASE_FACTOR,
    PRE_RUNNER_LOCK_FRAC,
    PRE_RUNNER_LOCK_THRESHOLD_R,
    STALE_EXIT_DAYS_MAX,
    STALE_EXIT_DAYS_MIN,
    STALE_R_THRESH,
    STALE_TIGHTEN_MULT,
    STALE_WARN_DAYS,
    TP1_R_ALIGNED,
    TP1_R_CAUTION,
    TP1_R_NEUTRAL,
    TP2_R_ALIGNED,
    TP2_R_CAUTION,
    TP2_R_NEUTRAL,
    TRAIL_4H_ATR_MULT,
    TRAIL_MULT_BASE_FACTOR,
)
from .models import Direction, TradeRegime


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
    """Compute the initial stop from the box anchor."""
    del sq_good
    buffer = atr_stop_mult * atr14_d

    if entry_type == "B":
        raw = box_low - buffer if direction == Direction.LONG else box_high + buffer
    elif _uses_early_branch_stop(entry_type):
        raw = box_high - buffer if direction == Direction.LONG else box_low + buffer
    else:
        raw = box_mid - buffer if direction == Direction.LONG else box_mid + buffer

    rounding = "down" if direction == Direction.LONG else "up"
    return round_to_tick(raw, tick_size, rounding)


def _uses_early_branch_stop(entry_type: str) -> bool:
    return entry_type in {
        "A",
        "A_strong",
        "A_strong_stop",
        "C_early_standard",
        "C_continuation",
        "C_fresh_market",
        "C_fresh_stop",
        "C_momentum_market",
        "C_momentum_stop",
    }


def compute_add_stop(
    direction: Direction,
    pullback_low: float,
    pullback_high: float,
    ref: float,
    atr14_d: float,
    atr_stop_mult: float,
    tick_size: float = 0.01,
) -> float:
    """Compute the add-on stop from the pullback anchor."""
    buffer = ADD_STOP_ATR_MULT * atr14_d * atr_stop_mult
    if direction == Direction.LONG:
        raw = min(pullback_low, ref) - buffer
        return round_to_tick(raw, tick_size, "down")
    raw = max(pullback_high, ref) + buffer
    return round_to_tick(raw, tick_size, "up")


def compute_be_stop(
    direction: Direction,
    avg_entry: float,
    atr14_d: float,
    tick_size: float = 0.01,
) -> float:
    """Compute the break-even stop with a small ATR buffer."""
    buffer = BE_BUFFER_ATR_MULT * atr14_d
    if direction == Direction.LONG:
        return round_to_tick(avg_entry + buffer, tick_size, "up")
    return round_to_tick(avg_entry - buffer, tick_size, "down")


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
    """Compute the ratcheting 4H trailing stop."""
    if direction == Direction.LONG:
        if len(highs_4h) == 0:
            return current_stop
        hh = max(highs_4h[-20:]) if len(highs_4h) >= 20 else max(highs_4h)
        raw = hh - trail_mult * atr14_4h
        raw = max(raw, ema50_4h - EMA_4H_FLOOR_ATR_MULT * atr14_4h)
        candidate = round_to_tick(raw, tick_size, "down")
        return max(candidate, current_stop)

    if len(lows_4h) == 0:
        return current_stop
    ll = min(lows_4h[-20:]) if len(lows_4h) >= 20 else min(lows_4h)
    raw = ll + trail_mult * atr14_4h
    raw = min(raw, ema50_4h + EMA_4H_FLOOR_ATR_MULT * atr14_4h)
    candidate = round_to_tick(raw, tick_size, "up")
    return min(candidate, current_stop)


def is_fast_branch_entry(entry_type: str) -> bool:
    return entry_type in {
        "A",
        "A_strong",
        "A_strong_stop",
        "C_early_standard",
        "C_fresh_market",
        "C_fresh_stop",
        "C_momentum_market",
        "C_momentum_stop",
    }


def is_momentum_branch_entry(entry_type: str) -> bool:
    return entry_type in {"C_momentum_market", "C_momentum_stop"}


def is_continuation_branch_entry(entry_type: str) -> bool:
    return entry_type == "C_continuation"


def compute_trail_mult(
    r_state: float,
    r_proxy: float,
    continuation: bool,
    entry_type: str = "",
) -> float:
    """Compute the trailing distance multiplier."""
    base_mult = TRAIL_4H_ATR_MULT
    base_factor = TRAIL_MULT_BASE_FACTOR
    if CONTINUATION_BRANCH_MANAGEMENT_ENABLE and is_continuation_branch_entry(entry_type):
        base_mult = CONTINUATION_BRANCH_TRAIL_4H_ATR_MULT
        base_factor = CONTINUATION_BRANCH_TRAIL_MULT_BASE_FACTOR
    elif MOMENTUM_BRANCH_MANAGEMENT_ENABLE and is_momentum_branch_entry(entry_type):
        base_mult = MOMENTUM_BRANCH_TRAIL_4H_ATR_MULT
        base_factor = MOMENTUM_BRANCH_TRAIL_MULT_BASE_FACTOR
    elif FAST_BRANCH_MANAGEMENT_ENABLE and is_fast_branch_entry(entry_type):
        base_mult = FAST_BRANCH_TRAIL_4H_ATR_MULT
        base_factor = FAST_BRANCH_TRAIL_MULT_BASE_FACTOR
    base = base_mult * base_factor
    if r_proxy >= 1.0 or continuation:
        base *= 0.70
    if r_state >= 2.0:
        base *= 0.60
    return max(1.0, base)


def handle_gap_through_stop(
    direction: Direction,
    stop_price: float,
    session_open: float,
) -> tuple[bool, float]:
    """Return whether the session open gapped through the stop."""
    if direction == Direction.LONG and session_open < stop_price:
        return True, session_open
    if direction == Direction.SHORT and session_open > stop_price:
        return True, session_open
    return False, 0.0


def check_stale_exit(days_held: int, r_state: float) -> tuple[bool, bool, bool]:
    """Return stale warning, tighten, and exit decisions."""
    should_warn = days_held >= STALE_WARN_DAYS and r_state < STALE_R_THRESH
    should_tighten = should_warn
    should_exit = days_held >= STALE_EXIT_DAYS_MIN and r_state < STALE_R_THRESH
    if days_held >= STALE_EXIT_DAYS_MAX and r_state < 0.10:
        should_exit = True
    return should_warn, should_tighten, should_exit


def apply_stale_tighten(trail_mult: float) -> float:
    return trail_mult * STALE_TIGHTEN_MULT


def get_tp_r_multiples(
    entry_type: str,
    trade_regime: TradeRegime,
    tp_scale: float = 1.0,
) -> tuple[float, float]:
    if CONTINUATION_BRANCH_MANAGEMENT_ENABLE and is_continuation_branch_entry(entry_type):
        if trade_regime == TradeRegime.ALIGNED:
            return CONTINUATION_BRANCH_TP1_R_ALIGNED * tp_scale, TP2_R_ALIGNED * tp_scale
        if trade_regime == TradeRegime.CAUTION:
            return CONTINUATION_BRANCH_TP1_R_CAUTION * tp_scale, TP2_R_CAUTION * tp_scale
        return CONTINUATION_BRANCH_TP1_R_NEUTRAL * tp_scale, TP2_R_NEUTRAL * tp_scale

    if MOMENTUM_BRANCH_MANAGEMENT_ENABLE and is_momentum_branch_entry(entry_type):
        if trade_regime == TradeRegime.ALIGNED:
            return MOMENTUM_BRANCH_TP1_R_ALIGNED * tp_scale, TP2_R_ALIGNED * tp_scale
        if trade_regime == TradeRegime.CAUTION:
            return MOMENTUM_BRANCH_TP1_R_CAUTION * tp_scale, TP2_R_CAUTION * tp_scale
        return MOMENTUM_BRANCH_TP1_R_NEUTRAL * tp_scale, TP2_R_NEUTRAL * tp_scale

    if FAST_BRANCH_MANAGEMENT_ENABLE and is_fast_branch_entry(entry_type):
        if trade_regime == TradeRegime.ALIGNED:
            return FAST_BRANCH_TP1_R_ALIGNED * tp_scale, TP2_R_ALIGNED * tp_scale
        if trade_regime == TradeRegime.CAUTION:
            return FAST_BRANCH_TP1_R_CAUTION * tp_scale, TP2_R_CAUTION * tp_scale
        return FAST_BRANCH_TP1_R_NEUTRAL * tp_scale, TP2_R_NEUTRAL * tp_scale

    if trade_regime == TradeRegime.ALIGNED:
        return TP1_R_ALIGNED * tp_scale, TP2_R_ALIGNED * tp_scale
    if trade_regime == TradeRegime.CAUTION:
        return TP1_R_CAUTION * tp_scale, TP2_R_CAUTION * tp_scale
    return TP1_R_NEUTRAL * tp_scale, TP2_R_NEUTRAL * tp_scale


def get_pre_runner_lock_params(entry_type: str) -> tuple[float, float]:
    if CONTINUATION_BRANCH_MANAGEMENT_ENABLE and is_continuation_branch_entry(entry_type):
        return CONTINUATION_BRANCH_PRE_RUNNER_LOCK_FRAC, CONTINUATION_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R
    if MOMENTUM_BRANCH_MANAGEMENT_ENABLE and is_momentum_branch_entry(entry_type):
        return MOMENTUM_BRANCH_PRE_RUNNER_LOCK_FRAC, MOMENTUM_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R
    if FAST_BRANCH_MANAGEMENT_ENABLE and is_fast_branch_entry(entry_type):
        return FAST_BRANCH_PRE_RUNNER_LOCK_FRAC, FAST_BRANCH_PRE_RUNNER_LOCK_THRESHOLD_R
    return PRE_RUNNER_LOCK_FRAC, PRE_RUNNER_LOCK_THRESHOLD_R


def get_tp1_partial_frac(entry_type: str, default_frac: float) -> float:
    if CONTINUATION_BRANCH_MANAGEMENT_ENABLE and is_continuation_branch_entry(entry_type):
        return CONTINUATION_BRANCH_TP1_PARTIAL_FRAC
    if MOMENTUM_BRANCH_MANAGEMENT_ENABLE and is_momentum_branch_entry(entry_type):
        return MOMENTUM_BRANCH_TP1_PARTIAL_FRAC
    return default_frac


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
