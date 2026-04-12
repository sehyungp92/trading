"""BRS exit logic — stop placement, trailing, catastrophic cap, stale/time exits.

Spec §12: Structure-aware + ATR hybrid stops, asymmetric trailing,
catastrophic cap, break-even, profit floor, stale, time decay.
"""
from __future__ import annotations

from backtests.swing.engine.brs_models import (
    BRSRegime,
    Direction,
    ExitReason,
)
from backtests.swing.config_brs import BRSSymbolConfig


# ---------------------------------------------------------------------------
# Initial stop placement (spec §12.1)
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
    """Compute initial stop: structure distance + ATR buffer, with ATR floor.

    Short stop: entry + max(struct_dist + buffer, floor)
    Long stop:  entry - max(struct_dist + buffer, floor)
    """
    if direction == Direction.SHORT:
        struct_dist = abs(signal_high + 0.01 - entry_price)
    else:
        struct_dist = abs(entry_price - (signal_low - 0.01))

    # Structure + ATR buffer (prevents noise whipsaws)
    stop_distance = struct_dist + sym_cfg.stop_buffer_atr * atr14_h
    # Floor: at least stop_floor_atr * ATR_h
    stop_distance = max(stop_distance, sym_cfg.stop_floor_atr * atr14_h)

    if direction == Direction.SHORT:
        return entry_price + stop_distance
    else:
        return entry_price - stop_distance


# ---------------------------------------------------------------------------
# Exit checking — returns (exit_reason, exit_price) or None
# ---------------------------------------------------------------------------

def check_exits(
    direction: Direction,
    entry_price: float,
    current_stop: float,
    risk_per_unit: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    atr14_h: float,
    atr14_d: float,
    bars_held: int,
    mfe_r: float,
    cur_r: float,
    be_triggered: bool,
    regime: BRSRegime,
    sym_cfg: BRSSymbolConfig,
    catastrophic_cap_r: float = 2.0,
    be_trigger_r: float = 0.75,
    trail_trigger_r: float = 1.25,
    stale_bars: int = 50,
    stale_early_bars: int = 35,
    time_decay_hours: int = 360,
    is_long: bool = False,
    hourly_highs: list[float] | None = None,
    hourly_lows: list[float] | None = None,
    min_hold_bars: int = 0,
) -> tuple[ExitReason, float] | None:
    """Check all exit conditions in priority order.

    Returns (exit_reason, exit_price) if an exit triggers, else None.
    """
    # (1) Catastrophic cap — highest priority (spec §12.2)
    if risk_per_unit > 0 and cur_r < -catastrophic_cap_r:
        return (ExitReason.CATASTROPHIC, bar_close)

    # (1b) Stop immunity during initial hold period (Change #4)
    # Only catastrophic can exit during min_hold window
    if min_hold_bars > 0 and bars_held < min_hold_bars:
        return None

    # (2) Stop hit
    if direction == Direction.SHORT:
        if bar_high >= current_stop:
            fill = min(current_stop, bar_high)  # assume fill at stop
            return (ExitReason.STOP, fill)
    else:
        if bar_low <= current_stop:
            fill = max(current_stop, bar_low)
            return (ExitReason.STOP, fill)

    # (3) Early stale (spec §15.3) — losers stuck
    if bars_held >= stale_early_bars and not be_triggered and cur_r < 0:
        return (ExitReason.STALE_EARLY, bar_close)

    # (4) Standard stale (spec §15.3)
    stale_threshold = stale_bars
    if is_long:
        stale_threshold = min(stale_bars, 30)  # tighter for longs
    if bars_held >= stale_threshold and cur_r < 0.5:
        return (ExitReason.STALE, bar_close)

    # (5) Time decay (spec §15.4)
    max_hours = time_decay_hours
    if not is_long and regime in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND):
        max_hours = 480  # extended for shorts in BEAR_STRONG/BEAR_TREND
    if bars_held >= max_hours and cur_r < 1.0:
        return (ExitReason.TIME_DECAY, bar_close)

    return None


# ---------------------------------------------------------------------------
# Trailing stop update (spec §12.3-12.6)
# ---------------------------------------------------------------------------

def check_scale_out(cur_r: float, target_r: float, tranche_b_open: bool) -> bool:
    """Check if tranche B should be exited at target (Change #6)."""
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
    """Update trailing stop. Returns (new_stop, be_triggered).

    Progression: initial → break-even → chandelier trail → profit floor.
    Ratchets only in favorable direction.
    """
    new_stop = current_stop
    new_be = be_triggered

    # (a) Break-even trigger at be_trigger_r (spec §12.3)
    if not be_triggered and mfe_r >= be_trigger_r:
        if direction == Direction.SHORT:
            be_stop = entry_price - 0.1 * atr14_d  # small cushion below entry for short
            if be_stop < new_stop:
                new_stop = be_stop
                new_be = True
        else:
            be_stop = entry_price + 0.1 * atr14_d
            if be_stop > new_stop:
                new_stop = be_stop
                new_be = True

    # (b) Chandelier trail at trail_trigger_r (spec §12.4)
    if new_be and mfe_r >= trail_trigger_r:
        chand_mult = sym_cfg.chand_mult + chand_bonus

        # Regime tightening (spec §12.6) — transition-based
        if prev_regime in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND, BRSRegime.BEAR_FORMING):
            if regime == BRSRegime.RANGE_CHOP:
                chand_mult = max(2.0, chand_mult - 0.25)
            elif regime == BRSRegime.BULL_TREND and direction == Direction.SHORT:
                chand_mult = max(2.0, chand_mult - 0.5)

        # Regime-adaptive continuous scaling (Change #8)
        if trail_regime_scaling:
            if regime == BRSRegime.BEAR_STRONG and direction == Direction.SHORT:
                chand_mult *= 1.15  # wider, let winners run
            elif regime == BRSRegime.RANGE_CHOP:
                chand_mult = max(2.0, chand_mult * 0.80)  # tighter, trend uncertain
            elif regime == BRSRegime.BULL_TREND and direction == Direction.SHORT:
                chand_mult = max(2.0, chand_mult * 0.65)  # very tight, trend against

        if direction == Direction.SHORT:
            # Loose trail for shorts — let winners run (spec §12.4)
            if hourly_lows and len(hourly_lows) >= 25:
                lowest_low = min(hourly_lows[-25:])
                chand_stop = lowest_low + chand_mult * atr14_h
                if chand_stop < new_stop:
                    new_stop = chand_stop
        else:
            # Tight trail for longs (spec §12.4)
            tight_mult = max(2.0, chand_mult - 0.5)
            if hourly_highs and len(hourly_highs) >= 15:
                highest_high = max(hourly_highs[-15:])
                chand_stop = highest_high - tight_mult * atr14_h
                if chand_stop > new_stop:
                    new_stop = chand_stop

    # (c) Profit floor (spec §12.5) — scaled by profit_floor_scale
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
