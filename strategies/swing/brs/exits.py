"""BRS exit logic -- stop placement, trailing, catastrophic cap, stale/time exits.

Spec 12: Structure-aware + ATR hybrid stops, asymmetric trailing,
catastrophic cap, break-even, profit floor, stale, time decay.
"""
from __future__ import annotations

from .models import (
    BRSRegime,
    Direction,
    ExitReason,
)
from .config import BRSSymbolConfig


# ---------------------------------------------------------------------------
# Initial stop placement (spec 12.1)
# ---------------------------------------------------------------------------

def compute_initial_stop(
    direction: Direction,
    entry_price: float,
    signal_high: float,
    signal_low: float,
    atr14_d: float,
    atr14_h: float,
    sym_cfg: BRSSymbolConfig,
) -> float:
    """Compute initial stop: structure distance + ATR buffer, with ATR floor."""
    if direction == Direction.SHORT:
        struct_dist = abs(signal_high + 0.01 - entry_price)
    else:
        struct_dist = abs(entry_price - (signal_low - 0.01))

    stop_distance = struct_dist + sym_cfg.stop_buffer_atr * atr14_h
    stop_distance = max(stop_distance, sym_cfg.stop_floor_atr * atr14_h)

    if direction == Direction.SHORT:
        return entry_price + stop_distance
    else:
        return entry_price - stop_distance


# ---------------------------------------------------------------------------
# Exit checking -- returns a broker-routed exit decision or None
# ---------------------------------------------------------------------------

def check_exits(
    risk_per_unit: float,
    bars_held: int,
    cur_r: float,
    be_triggered: bool,
    regime: BRSRegime,
    catastrophic_cap_r: float = 2.0,
    stale_bars: int = 50,
    stale_early_bars: int = 35,
    time_decay_hours: int = 360,
    is_long: bool = False,
    min_hold_bars: int = 0,
) -> ExitReason | None:
    """Check discretionary exit conditions in priority order.

    Protective stop fills are handled separately by the broker simulation.
    This helper only answers whether the strategy wants to route a market
    exit on the next bar.
    """
    # (1) Catastrophic cap
    if risk_per_unit > 0 and cur_r < -catastrophic_cap_r:
        return ExitReason.CATASTROPHIC

    # (1b) Stop immunity during initial hold period
    if min_hold_bars > 0 and bars_held < min_hold_bars:
        return None

    # (2) Early stale
    if bars_held >= stale_early_bars and not be_triggered and cur_r < 0:
        return ExitReason.STALE_EARLY

    # (3) Standard stale
    stale_threshold = stale_bars
    if is_long:
        stale_threshold = min(stale_bars, 30)
    if bars_held >= stale_threshold and cur_r < 0.5:
        return ExitReason.STALE

    # (4) Time decay
    max_hours = time_decay_hours
    if not is_long and regime in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND):
        max_hours = 480
    if bars_held >= max_hours and cur_r < 1.0:
        return ExitReason.TIME_DECAY

    return None


# ---------------------------------------------------------------------------
# Trailing stop update (spec 12.3-12.6)
# ---------------------------------------------------------------------------

def check_scale_out(cur_r: float, target_r: float, tranche_b_open: bool) -> bool:
    """Check if tranche B should be exited at target."""
    return tranche_b_open and cur_r >= target_r


def update_trailing_stop(
    direction: Direction,
    current_stop: float,
    entry_price: float,
    risk_per_unit: float,
    mfe_r: float,
    atr14_d: float,
    atr14_h: float,
    be_triggered: bool,
    regime: BRSRegime,
    prev_regime: BRSRegime,
    sym_cfg: BRSSymbolConfig,
    be_trigger_r: float = 0.75,
    trail_trigger_r: float = 1.25,
    profit_floor_scale: float = 1.0,
    hourly_highs: list[float] | None = None,
    hourly_lows: list[float] | None = None,
    chand_bonus: float = 0.0,
    trail_regime_scaling: bool = False,
) -> tuple[float, bool]:
    """Update trailing stop. Returns (new_stop, be_triggered)."""
    new_stop = current_stop
    new_be = be_triggered

    # (a) Break-even trigger
    if not be_triggered and mfe_r >= be_trigger_r:
        if direction == Direction.SHORT:
            be_stop = entry_price - 0.1 * atr14_d
            if be_stop < new_stop:
                new_stop = be_stop
                new_be = True
        else:
            be_stop = entry_price + 0.1 * atr14_d
            if be_stop > new_stop:
                new_stop = be_stop
                new_be = True

    # (b) Chandelier trail
    if new_be and mfe_r >= trail_trigger_r:
        chand_mult = sym_cfg.chand_mult + chand_bonus

        if prev_regime in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND, BRSRegime.BEAR_FORMING):
            if regime == BRSRegime.RANGE_CHOP:
                chand_mult = max(2.0, chand_mult - 0.25)
            elif regime == BRSRegime.BULL_TREND and direction == Direction.SHORT:
                chand_mult = max(2.0, chand_mult - 0.5)

        if trail_regime_scaling:
            if regime == BRSRegime.BEAR_STRONG and direction == Direction.SHORT:
                chand_mult *= 1.15
            elif regime == BRSRegime.RANGE_CHOP:
                chand_mult = max(2.0, chand_mult * 0.80)
            elif regime == BRSRegime.BULL_TREND and direction == Direction.SHORT:
                chand_mult = max(2.0, chand_mult * 0.65)

        if direction == Direction.SHORT:
            if hourly_lows and len(hourly_lows) >= 25:
                lowest_low = min(hourly_lows[-25:])
                chand_stop = lowest_low + chand_mult * atr14_h
                if chand_stop < new_stop:
                    new_stop = chand_stop
        else:
            tight_mult = max(2.0, chand_mult - 0.5)
            if hourly_highs and len(hourly_highs) >= 15:
                highest_high = max(hourly_highs[-15:])
                chand_stop = highest_high - tight_mult * atr14_h
                if chand_stop > new_stop:
                    new_stop = chand_stop

    # (c) Profit floor
    if risk_per_unit > 0:
        floors = [
            (1.0, 0.25),
            (1.5, 0.50),
            (2.0, 1.00),
            (3.0, 1.50),
            (4.0, 2.50),
        ]
        for mfe_threshold, lock_r in floors:
            lock_r = lock_r * profit_floor_scale
            if mfe_r >= mfe_threshold:
                if direction == Direction.SHORT:
                    floor_stop = entry_price - lock_r * risk_per_unit
                    if floor_stop < new_stop:
                        new_stop = floor_stop
                else:
                    floor_stop = entry_price + lock_r * risk_per_unit
                    if floor_stop > new_stop:
                        new_stop = floor_stop

    return new_stop, new_be
