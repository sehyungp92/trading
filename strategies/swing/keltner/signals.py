"""Keltner Momentum Breakout — pure signal detection.

Four entry modes:
  breakout  — price breaks above/below Keltner band
  pullback  — price crosses back through Keltner midline
  momentum  — RSI crosses threshold with trend confirmation
  dual      — breakout OR pullback (maximum frequency)

Three exit modes:
  trail_only — no signal exit, rely on trailing stop (max hold period)
  reversal   — exit when opposite entry conditions fire
  midline    — exit on Keltner midline cross (quickest)

All functions are stateless.
"""
from __future__ import annotations

from .config import SymbolConfig
from .models import DailyState, Direction


def entry_signal(state: DailyState, cfg: SymbolConfig) -> Direction:
    """Detect entry based on configured entry mode."""
    # Volume filter
    if cfg.volume_filter and state.volume_sma > 0 and state.volume < state.volume_sma:
        return Direction.FLAT

    mode = cfg.entry_mode
    if mode == "breakout":
        return _breakout_entry(state, cfg)
    elif mode == "pullback":
        return _pullback_entry(state, cfg)
    elif mode == "momentum":
        return _momentum_entry(state, cfg)
    elif mode == "dual":
        return _dual_entry(state, cfg)
    return Direction.FLAT


def _breakout_entry(state: DailyState, cfg: SymbolConfig) -> Direction:
    """Long: close > upper band, RSI bullish, ROC positive.
    Short: close < lower band, RSI bearish, ROC negative.
    """
    if (
        state.close > state.kelt_upper
        and state.rsi > cfg.rsi_entry_long
        and state.roc > 0
    ):
        return Direction.LONG

    if (
        cfg.shorts_enabled
        and state.close < state.kelt_lower
        and state.rsi < cfg.rsi_entry_short
        and state.roc < 0
    ):
        return Direction.SHORT

    return Direction.FLAT


def _pullback_entry(state: DailyState, cfg: SymbolConfig) -> Direction:
    """Long: price crosses above Keltner midline (pullback recovery).
    Short: price crosses below Keltner midline.
    """
    if (
        state.close >= state.kelt_middle
        and state.close_prev < state.kelt_middle_prev
        and state.rsi > cfg.rsi_entry_long
        and state.roc > 0
    ):
        return Direction.LONG

    if (
        cfg.shorts_enabled
        and state.close <= state.kelt_middle
        and state.close_prev > state.kelt_middle_prev
        and state.rsi < cfg.rsi_entry_short
        and state.roc < 0
    ):
        return Direction.SHORT

    return Direction.FLAT


def _momentum_entry(state: DailyState, cfg: SymbolConfig) -> Direction:
    """Long: RSI crosses above threshold + price above midline + ROC positive.
    Short: RSI crosses below threshold + price below midline + ROC negative.
    """
    if (
        state.rsi > cfg.rsi_entry_long
        and state.rsi_prev <= cfg.rsi_entry_long
        and state.close > state.kelt_middle
        and state.roc > 0
    ):
        return Direction.LONG

    if (
        cfg.shorts_enabled
        and state.rsi < cfg.rsi_entry_short
        and state.rsi_prev >= cfg.rsi_entry_short
        and state.close < state.kelt_middle
        and state.roc < 0
    ):
        return Direction.SHORT

    return Direction.FLAT


def _dual_entry(state: DailyState, cfg: SymbolConfig) -> Direction:
    """Breakout OR pullback — maximum trading frequency."""
    result = _breakout_entry(state, cfg)
    if result != Direction.FLAT:
        return result
    return _pullback_entry(state, cfg)


def should_exit_signal(
    state: DailyState,
    position_direction: Direction,
    cfg: SymbolConfig,
) -> tuple[bool, str]:
    """Check exit based on configured exit mode."""
    mode = cfg.exit_mode

    if mode == "trail_only":
        return False, ""

    if mode == "reversal":
        return _reversal_exit(state, position_direction, cfg)

    # Default: midline exit
    if position_direction == Direction.LONG:
        if state.close < state.kelt_middle and state.close_prev >= state.kelt_middle_prev:
            return True, "KELT_MID_EXIT"
    elif position_direction == Direction.SHORT:
        if state.close > state.kelt_middle and state.close_prev <= state.kelt_middle_prev:
            return True, "KELT_MID_EXIT"

    return False, ""


def _reversal_exit(
    state: DailyState,
    position_direction: Direction,
    cfg: SymbolConfig,
) -> tuple[bool, str]:
    """Exit only when opposite entry conditions are fully met."""
    if position_direction == Direction.LONG:
        if _is_bearish(state, cfg):
            return True, "REVERSAL_EXIT"
    elif position_direction == Direction.SHORT:
        if _is_bullish(state, cfg):
            return True, "REVERSAL_EXIT"
    return False, ""


def _is_bullish(state: DailyState, cfg: SymbolConfig) -> bool:
    """Are bullish conditions present? (for reversal exit from short)."""
    return (
        state.close > state.kelt_upper
        and state.rsi > cfg.rsi_entry_long
        and state.roc > 0
    )


def _is_bearish(state: DailyState, cfg: SymbolConfig) -> bool:
    """Are bearish conditions present? (for reversal exit from long)."""
    return (
        state.close < state.kelt_lower
        and state.rsi < cfg.rsi_entry_short
        and state.roc < 0
    )
