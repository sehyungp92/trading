"""BRS 4-state regime classifier with ADX hysteresis + bias confirmation.

Spec 2.2-2.6: Asymmetric speed -- bear regimes confirm faster than bull.
"""
from __future__ import annotations

from .models import (
    BRSRegime,
    BiasState,
    Direction,
)


# ---------------------------------------------------------------------------
# Regime ON/OFF with ADX hysteresis (spec 2.2)
# ---------------------------------------------------------------------------

def compute_regime_on(
    adx: float,
    prev_regime_on: bool,
    adx_on: int,
    adx_off: int,
) -> bool:
    """ADX hysteresis: turns ON at adx_on, stays ON until adx < adx_off."""
    if prev_regime_on:
        return adx >= adx_off
    return adx >= adx_on


# ---------------------------------------------------------------------------
# 4-state classification (spec 2.3)
# ---------------------------------------------------------------------------

def classify_regime(
    adx: float,
    plus_di: float,
    minus_di: float,
    close: float,
    ema_fast: float,
    ema_slow: float,
    regime_on: bool,
    adx_strong: int = 28,
    bear_min_conditions: int = 4,
) -> BRSRegime:
    """Classify into BEAR_STRONG, BEAR_TREND, BEAR_FORMING, BULL_TREND, or RANGE_CHOP."""
    if not regime_on:
        return BRSRegime.RANGE_CHOP

    bear_conds = sum([
        ema_fast < ema_slow,
        close < ema_fast,
        minus_di > plus_di,
    ])

    if adx >= adx_strong and ema_fast < ema_slow and minus_di > plus_di:
        return BRSRegime.BEAR_STRONG

    if bear_conds == 3:
        return BRSRegime.BEAR_TREND

    if bear_min_conditions < 4 and bear_conds >= bear_min_conditions:
        return BRSRegime.BEAR_FORMING

    if ema_fast > ema_slow and close > ema_fast and plus_di > minus_di:
        return BRSRegime.BULL_TREND

    return BRSRegime.RANGE_CHOP


# ---------------------------------------------------------------------------
# Raw bias (spec 2.6)
# ---------------------------------------------------------------------------

def compute_raw_bias(
    close: float,
    ema_fast: float,
    ema_slow: float,
) -> Direction:
    """Compute raw (unconfirmed) directional bias."""
    if close < ema_fast and ema_fast < ema_slow:
        return Direction.SHORT
    elif close > ema_fast and ema_fast > ema_slow:
        return Direction.LONG
    return Direction.FLAT


# ---------------------------------------------------------------------------
# Bias confirmation (spec 2.6) -- asymmetric speed
# ---------------------------------------------------------------------------

def update_bias(
    prev_bias: BiasState,
    regime: BRSRegime,
    regime_on: bool,
    raw_direction: Direction,
    bear_score: float,
    adx: float,
    di_diff: float,
    ema_sep_pct: float,
    daily_return: float = 0.0,
    atr_ratio: float = 1.0,
    fast_crash_enabled: bool = False,
    fast_crash_return_thresh: float = -0.02,
    fast_crash_atr_ratio: float = 2.0,
    crash_override_enabled: bool = False,
    crash_override_return: float = -0.03,
    crash_override_atr_ratio: float = 1.8,
    return_only_enabled: bool = False,
    return_only_thresh: float = -0.015,
    close: float = 0.0,
    ema_fast: float = 0.0,
    bias_4h_accel_enabled: bool = False,
    bias_4h_accel_reduction: int = 1,
    regime_4h_bear: bool = False,
    cum_return_enabled: bool = False,
    cum_return: float = 0.0,
    cum_return_thresh: float = -0.03,
    churn_bridge_bars: int = 0,
) -> BiasState:
    """Update bias state with hold counting and fast-confirm paths."""
    new_bias = BiasState(
        raw_direction=raw_direction,
        bear_score=bear_score,
        adx=adx,
        di_diff=di_diff,
        ema_sep_pct=ema_sep_pct,
    )

    # Update hold count (with R8 churn bridge support)
    if raw_direction == prev_bias.raw_direction and raw_direction != Direction.FLAT:
        new_bias.hold_count = prev_bias.hold_count + 1
        new_bias.flat_streak = 0
        if raw_direction == Direction.SHORT:
            new_bias.prev_short_hold = new_bias.hold_count
    elif raw_direction != Direction.FLAT:
        if (raw_direction == Direction.SHORT
                and churn_bridge_bars > 0
                and prev_bias.flat_streak > 0
                and prev_bias.flat_streak <= churn_bridge_bars
                and prev_bias.prev_short_hold >= 2):
            new_bias.hold_count = prev_bias.prev_short_hold + 1
            new_bias.prev_short_hold = new_bias.hold_count
        else:
            new_bias.hold_count = 1
        new_bias.flat_streak = 0
    else:
        new_bias.flat_streak = prev_bias.flat_streak + 1
        new_bias.prev_short_hold = prev_bias.prev_short_hold
        new_bias.hold_count = 0
        new_bias.confirmed_direction = Direction.FLAT
        return new_bias

    # Path E: Crash override
    if (crash_override_enabled
            and raw_direction == Direction.SHORT
            and new_bias.hold_count >= 1
            and daily_return < crash_override_return
            and atr_ratio > crash_override_atr_ratio):
        new_bias.confirmed_direction = Direction.SHORT
        new_bias.crash_override = True
        return new_bias

    # Path F: Return-only bias confirmation
    if (return_only_enabled
            and raw_direction == Direction.SHORT
            and new_bias.hold_count >= 1
            and daily_return < return_only_thresh
            and close < ema_fast
            and ema_fast > 0):
        new_bias.confirmed_direction = Direction.SHORT
        return new_bias

    # Path G: Cumulative return
    if (cum_return_enabled
            and raw_direction == Direction.SHORT
            and new_bias.hold_count >= 1
            and cum_return <= cum_return_thresh
            and close < ema_fast
            and ema_fast > 0):
        new_bias.confirmed_direction = Direction.SHORT
        new_bias.cum_return_override = True
        return new_bias

    if prev_bias.crash_override and raw_direction == Direction.SHORT:
        new_bias.crash_override = True

    # Normal confirmation paths (regime_on required)
    if raw_direction == Direction.SHORT and regime_on:
        hold_req = 2
        if bias_4h_accel_enabled and regime_4h_bear:
            hold_req = max(1, hold_req - bias_4h_accel_reduction)
        if new_bias.hold_count >= hold_req:
            new_bias.confirmed_direction = Direction.SHORT
        elif new_bias.hold_count >= 1 and bear_score >= 50 and adx >= 20:
            new_bias.confirmed_direction = Direction.SHORT
        elif new_bias.hold_count >= 1 and di_diff >= 8 and ema_sep_pct >= 0.15 and adx >= 18:
            new_bias.confirmed_direction = Direction.SHORT
        elif (fast_crash_enabled and new_bias.hold_count >= 1
              and daily_return < fast_crash_return_thresh
              and atr_ratio > fast_crash_atr_ratio):
            new_bias.confirmed_direction = Direction.SHORT
        else:
            new_bias.confirmed_direction = (
                prev_bias.confirmed_direction
                if prev_bias.confirmed_direction == Direction.SHORT
                else Direction.FLAT
            )

    elif raw_direction == Direction.LONG and regime_on:
        if new_bias.hold_count >= 3:
            new_bias.confirmed_direction = Direction.LONG
        else:
            new_bias.confirmed_direction = (
                prev_bias.confirmed_direction
                if prev_bias.confirmed_direction == Direction.LONG
                else Direction.FLAT
            )
    else:
        if new_bias.crash_override and raw_direction == Direction.SHORT:
            new_bias.confirmed_direction = Direction.SHORT
        else:
            new_bias.confirmed_direction = Direction.FLAT

    return new_bias
