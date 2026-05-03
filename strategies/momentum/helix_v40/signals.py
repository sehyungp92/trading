"""Setup detection for Helix shared signal logic."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .config import (
    TF,
    SessionBlock,
    Setup,
    SetupClass,
    Pivot,
    ENTRY_BUFFER_ATR_FRAC,
    ENTRY_BUFFER_MIN_PTS,
    CLASS_M_STOP_ATR_MULT,
    CLASS_M_PB_MIN,
    CLASS_M_PB_MAX,
    CLASS_M_MOMENTUM_LOOKBACK,
    CLASS_F_WINDOW_BARS,
    CLASS_F_MIN_PB_ATR,
    CLASS_F_MAX_PB_ATR,
    CLASS_F_STOP_ATR_MULT,
    CLASS_F_MIN_ALIGNMENT_SCORE,
    CLASS_F_MIN_CONTEXT_SCORE,
    CLASS_F_REQUIRE_STRONG_TREND,
    CLASS_F_RECLAIM_ENABLED,
    CLASS_F_RECLAIM_ALLOW_RTH_PRIME1,
    CLASS_F_RECLAIM_ALLOW_RTH_PRIME2,
    CLASS_F_RECLAIM_ALLOW_ETH_QUALITY_PM,
    CLASS_F_RECLAIM_MIN_ALIGNMENT_SCORE,
    CLASS_F_RECLAIM_MIN_CONTEXT_SCORE,
    CLASS_F_RECLAIM_REQUIRE_STRONG_TREND,
    CLASS_F_RECLAIM_MIN_VOL_PCT,
    CLASS_F_RECLAIM_MAX_EXIT_BARS,
    CLASS_F_RECLAIM_MAX_PULLBACK_ATR,
    CLASS_F_RECLAIM_EMA_HOLD_TOL_ATR,
    CLASS_F_RECLAIM_STOP_ATR_MULT,
    CLASS_F_RECLAIM_MAX_STOP_DIST_ATR,
    TREND_RETEST_ENABLED,
    TREND_RETEST_ALLOW_LONG,
    TREND_RETEST_ALLOW_SHORT,
    TREND_RETEST_ALLOW_RTH_PRIME1,
    TREND_RETEST_ALLOW_RTH_PRIME2,
    TREND_RETEST_ALLOW_RTH_DEAD,
    TREND_RETEST_ALLOW_ETH_QUALITY_PM,
    TREND_RETEST_MIN_VOL_PCT,
    TREND_RETEST_MAX_VOL_PCT,
    TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG,
    TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT,
    TREND_RETEST_MIN_CONTEXT_SCORE_LONG,
    TREND_RETEST_MIN_CONTEXT_SCORE_SHORT,
    TREND_RETEST_REQUIRE_STRONG_TREND,
    TREND_RETEST_MIN_TREND_STRENGTH,
    TREND_RETEST_LOOKBACK_BARS,
    TREND_RETEST_MIN_BARS_BETWEEN,
    TREND_RETEST_PULLBACK_EMA_TOL_ATR,
    TREND_RETEST_MAX_PULLBACK_ATR,
    TREND_RETEST_MAX_BREAKOUT_GAP_ATR,
    TREND_RETEST_STOP_ATR_MULT,
    TREND_RETEST_MAX_STOP_DIST_ATR,
    FAILED_RECLAIM_SHORT_ENABLED,
    FAILED_RECLAIM_SHORT_ALLOW_RTH_DEAD,
    FAILED_RECLAIM_SHORT_ALLOW_RTH_PRIME2,
    FAILED_RECLAIM_SHORT_ALLOW_ETH_QUALITY_PM,
    FAILED_RECLAIM_SHORT_MIN_VOL_PCT,
    FAILED_RECLAIM_SHORT_MAX_VOL_PCT,
    FAILED_RECLAIM_SHORT_MIN_ALIGNMENT_SCORE,
    FAILED_RECLAIM_SHORT_MIN_CONTEXT_SCORE,
    FAILED_RECLAIM_SHORT_REQUIRE_STRONG_TREND,
    FAILED_RECLAIM_SHORT_MIN_TREND_STRENGTH,
    FAILED_RECLAIM_SHORT_LOOKBACK_BARS,
    FAILED_RECLAIM_SHORT_MIN_BARS_BETWEEN,
    FAILED_RECLAIM_SHORT_EMA_TOL_ATR,
    FAILED_RECLAIM_SHORT_MAX_BREAKOUT_GAP_ATR,
    FAILED_RECLAIM_SHORT_STOP_ATR_MULT,
    FAILED_RECLAIM_SHORT_MAX_STOP_DIST_ATR,
    MICRO_TREND_RETEST_ENABLED,
    MICRO_TREND_RETEST_ALLOW_LONG,
    MICRO_TREND_RETEST_ALLOW_SHORT,
    MICRO_TREND_RETEST_ALLOW_RTH_PRIME1,
    MICRO_TREND_RETEST_ALLOW_RTH_PRIME2,
    MICRO_TREND_RETEST_ALLOW_RTH_DEAD,
    MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_AM,
    MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_PM,
    MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME1,
    MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME2,
    MICRO_TREND_RETEST_ALLOW_LONG_RTH_DEAD,
    MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_AM,
    MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_PM,
    MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME1,
    MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME2,
    MICRO_TREND_RETEST_ALLOW_SHORT_RTH_DEAD,
    MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_AM,
    MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_PM,
    MICRO_TREND_RETEST_MIN_VOL_PCT,
    MICRO_TREND_RETEST_MAX_VOL_PCT,
    MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG,
    MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT,
    MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_LONG,
    MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT,
    MICRO_TREND_RETEST_REQUIRE_STRONG_TREND,
    MICRO_TREND_RETEST_MIN_TREND_STRENGTH,
    MICRO_TREND_RETEST_LOOKBACK_BARS,
    MICRO_TREND_RETEST_MIN_BARS_BETWEEN,
    MICRO_TREND_RETEST_PULLBACK_EMA_TOL_ATR,
    MICRO_TREND_RETEST_MAX_PULLBACK_ATR,
    MICRO_TREND_RETEST_MAX_BREAKOUT_GAP_ATR,
    MICRO_TREND_RETEST_STOP_M5_ATR_MULT,
    MICRO_TREND_RETEST_STOP_H1_ATR_FRAC,
    MICRO_TREND_RETEST_MAX_STOP_DIST_ATR,
    M15_TREND_RETEST_ENABLED,
    M15_TREND_RETEST_ALLOW_LONG,
    M15_TREND_RETEST_ALLOW_SHORT,
    M15_TREND_RETEST_ALLOW_RTH_PRIME1,
    M15_TREND_RETEST_ALLOW_RTH_PRIME2,
    M15_TREND_RETEST_ALLOW_RTH_DEAD,
    M15_TREND_RETEST_ALLOW_ETH_QUALITY_AM,
    M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM,
    M15_TREND_RETEST_MIN_VOL_PCT,
    M15_TREND_RETEST_MAX_VOL_PCT,
    M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG,
    M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT,
    M15_TREND_RETEST_MIN_CONTEXT_SCORE_LONG,
    M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT,
    M15_TREND_RETEST_REQUIRE_STRONG_TREND,
    M15_TREND_RETEST_MIN_TREND_STRENGTH,
    M15_TREND_RETEST_LOOKBACK_BARS,
    M15_TREND_RETEST_MIN_BARS_BETWEEN,
    M15_TREND_RETEST_PULLBACK_EMA_TOL_ATR,
    M15_TREND_RETEST_MAX_PULLBACK_ATR,
    M15_TREND_RETEST_MAX_BREAKOUT_GAP_ATR,
    M15_TREND_RETEST_STOP_M15_ATR_MULT,
    M15_TREND_RETEST_STOP_H1_ATR_FRAC,
    M15_TREND_RETEST_MAX_STOP_DIST_ATR,
    CLASS_T_TREND_STRENGTH_MIN,
    CLASS_T_HIST_LOOKBACK,
    CLASS_T_MACD_1H_LOOKBACK,
    CLASS_T_PIVOT_RECENCY_H,
    CLASS_T_COMPRESSION_MAX,
    CLASS_T_STOP_ATR_MULT,
    CLASS_T_MAX_STOP_DIST_ATR,
    CLASS_T_SECONDARY_ENABLED,
    CLASS_T_SECONDARY_ALLOW_RTH_PRIME1,
    CLASS_T_SECONDARY_ALLOW_RTH_PRIME2,
    CLASS_T_SECONDARY_ALLOW_ETH_QUALITY_PM,
    CLASS_T_SECONDARY_TREND_STRENGTH_MIN,
    CLASS_T_SECONDARY_PIVOT_RECENCY_H,
    CLASS_T_SECONDARY_COMPRESSION_MAX,
    CLASS_T_SECONDARY_MIN_ALIGNMENT_SCORE,
    CLASS_T_SECONDARY_MIN_CONTEXT_SCORE,
    CLASS_T_SECONDARY_MIN_HIST_RATIO,
    CLASS_T_SECONDARY_MAX_BREAKOUT_GAP_ATR,
    CLASS_T_SECONDARY_STOP_ATR_MULT,
    CLASS_T_SECONDARY_MAX_STOP_DIST_ATR,
    CLASS_T_SHORT_ENABLED,
    CLASS_T_SHORT_ALLOW_ETH_QUALITY_PM,
    CLASS_T_SHORT_ALLOW_RTH_PRIME2,
    CLASS_T_SHORT_TREND_STRENGTH_MIN,
    CLASS_T_SHORT_PIVOT_RECENCY_H,
    CLASS_T_SHORT_COMPRESSION_MAX,
    CLASS_T_SHORT_MIN_ALIGNMENT_SCORE,
    CLASS_T_SHORT_MIN_CONTEXT_SCORE,
    CLASS_T_SHORT_MIN_HIST_RATIO,
    CLASS_T_SHORT_MAX_BREAKOUT_GAP_ATR,
    CLASS_T_SHORT_STOP_ATR_MULT,
    CLASS_T_SHORT_MAX_STOP_DIST_ATR,
    FOLLOWTHROUGH_CATCHUP_ENABLED,
    FOLLOWTHROUGH_ALLOW_CLASS_M,
    FOLLOWTHROUGH_ALLOW_CLASS_T,
    FOLLOWTHROUGH_ALLOW_RTH_PRIME1,
    FOLLOWTHROUGH_ALLOW_RTH_PRIME2,
    FOLLOWTHROUGH_ALLOW_ETH_QUALITY_PM,
    FOLLOWTHROUGH_MIN_ALIGNMENT_SCORE,
    FOLLOWTHROUGH_REQUIRE_STRONG_TREND,
    FOLLOWTHROUGH_MAX_GAP_ATR,
    FOLLOWTHROUGH_MAX_BARS_SINCE_ARM,
    SECOND_CHANCE_REARM_ENABLED,
    SECOND_CHANCE_ALLOW_CLASS_M,
    SECOND_CHANCE_ALLOW_CLASS_T,
    SECOND_CHANCE_ALLOW_ETH_QUALITY_AM,
    SECOND_CHANCE_ALLOW_RTH_PRIME1,
    SECOND_CHANCE_ALLOW_RTH_PRIME2,
    SECOND_CHANCE_ALLOW_ETH_QUALITY_PM,
    SECOND_CHANCE_MIN_ALIGNMENT_SCORE,
    SECOND_CHANCE_REQUIRE_STRONG_TREND,
    SECOND_CHANCE_MAX_BARS_SINCE_CANCEL,
    SECOND_CHANCE_MAX_BREAKOUT_GAP_ATR,
    SECOND_CHANCE_STOP_ATR_MULT,
    SECOND_CHANCE_MAX_STOP_DIST_ATR,
    SECOND_CHANCE_PULLBACK_EMA_TOL_ATR,
    SECOND_CHANCE_REQUIRE_FRESH_PIVOT,
    SECOND_CHANCE_USE_BASE_TRIGGER,
    HIGH_VOL_RETEST_ENABLED,
    HIGH_VOL_RETEST_ALLOW_RTH_PRIME1,
    HIGH_VOL_RETEST_ALLOW_RTH_PRIME2,
    HIGH_VOL_RETEST_ALLOW_ETH_QUALITY_PM,
    HIGH_VOL_RETEST_MIN_VOL_PCT,
    HIGH_VOL_RETEST_MAX_VOL_PCT,
    HIGH_VOL_RETEST_MIN_ALIGNMENT_SCORE,
    HIGH_VOL_RETEST_REQUIRE_STRONG_TREND,
    HIGH_VOL_RETEST_MIN_TREND_STRENGTH,
    HIGH_VOL_RETEST_PIVOT_RECENCY_H,
    HIGH_VOL_RETEST_MAX_PULLBACK_ATR,
    HIGH_VOL_RETEST_MAX_BREAKOUT_GAP_ATR,
    HIGH_VOL_RETEST_STOP_ATR_MULT,
    HIGH_VOL_RETEST_MAX_STOP_DIST_ATR,
    HIGH_VOL_RETEST_H4_HIST_RATIO_MIN,
    HIGH_VOL_RETEST_EMA_HOLD_TOL_ATR,
    HIGH_VOL_M_CONT_ENABLED,
    HIGH_VOL_M_CONT_ALLOW_LONG,
    HIGH_VOL_M_CONT_ALLOW_SHORT,
    HIGH_VOL_M_CONT_ALLOW_RTH_PRIME1,
    HIGH_VOL_M_CONT_ALLOW_RTH_PRIME2,
    HIGH_VOL_M_CONT_ALLOW_RTH_DEAD,
    HIGH_VOL_M_CONT_ALLOW_ETH_QUALITY_PM,
    HIGH_VOL_M_CONT_MIN_VOL_PCT,
    HIGH_VOL_M_CONT_MAX_VOL_PCT,
    HIGH_VOL_M_CONT_MIN_ALIGNMENT_SCORE,
    HIGH_VOL_M_CONT_MIN_CONTEXT_SCORE_LONG,
    HIGH_VOL_M_CONT_MIN_CONTEXT_SCORE_SHORT,
    HIGH_VOL_M_CONT_REQUIRE_STRONG_TREND,
    HIGH_VOL_M_CONT_MIN_TREND_STRENGTH,
    HIGH_VOL_M_CONT_MAX_PULLBACK_ATR,
    HIGH_VOL_M_CONT_MAX_BREAKOUT_GAP_ATR,
    HIGH_VOL_M_CONT_STOP_ATR_MULT,
    HIGH_VOL_M_CONT_MAX_STOP_DIST_ATR,
    HIGH_VOL_M_CONT_EMA_HOLD_TOL_ATR,
    NORM_VOL_M_SHORT_ENABLED,
    NORM_VOL_M_SHORT_ALLOW_RTH_DEAD,
    NORM_VOL_M_SHORT_ALLOW_RTH_PRIME2,
    NORM_VOL_M_SHORT_ALLOW_ETH_QUALITY_PM,
    NORM_VOL_M_SHORT_MIN_VOL_PCT,
    NORM_VOL_M_SHORT_MAX_VOL_PCT,
    NORM_VOL_M_SHORT_MIN_ALIGNMENT_SCORE,
    NORM_VOL_M_SHORT_REQUIRE_STRONG_TREND,
    NORM_VOL_M_SHORT_MIN_TREND_STRENGTH,
    NORM_VOL_M_SHORT_MIN_CONTEXT_SCORE,
    NORM_VOL_M_SHORT_MAX_BREAKOUT_GAP_ATR,
    NORM_VOL_M_SHORT_STOP_ATR_MULT,
    NORM_VOL_M_SHORT_MAX_STOP_DIST_ATR,
    STRONG_TREND_THRESHOLD,
    EXTENSION_ATR_MULT,
    REENTRY_LOOKBACK_1H,
)
from .context import context_quality_score_for_features, meets_context_quality
from .indicators import BarSeries
from .pivots import PivotDetector
from .session import get_session_block


def _buffer(atr_val: float) -> float:
    return max(ENTRY_BUFFER_MIN_PTS, ENTRY_BUFFER_ATR_FRAC * atr_val)


def alignment_score(direction: int, daily: BarSeries, h4: BarSeries) -> int:
    score = 0
    if direction == 1:
        if daily.ema_fast() > daily.ema_slow() and daily.last_close > daily.ema_fast():
            score += 1
        if h4.ema_fast() > h4.ema_slow() and h4.last_close > h4.ema_fast():
            score += 1
    else:
        if daily.ema_fast() < daily.ema_slow() and daily.last_close < daily.ema_fast():
            score += 1
        if h4.ema_fast() < h4.ema_slow() and h4.last_close < h4.ema_fast():
            score += 1
    return score


def trend_strength(daily: BarSeries) -> float:
    atr_d = daily.current_atr()
    if atr_d <= 0:
        return 0.0
    return abs(daily.ema_fast() - daily.ema_slow()) / atr_d


def strong_trend_flag(score: int, daily: BarSeries) -> bool:
    return trend_strength(daily) > STRONG_TREND_THRESHOLD and score == 2


def extension_blocked(direction: int, daily: BarSeries) -> bool:
    atr_d = daily.current_atr()
    if direction == 1:
        return daily.last_close > daily.ema_fast() + EXTENSION_ATR_MULT * atr_d
    return daily.last_close < daily.ema_fast() - EXTENSION_ATR_MULT * atr_d


def _safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-9:
        return 0.0
    return num / den


def _hist_ratio(series: BarSeries) -> float:
    prev = series.macd_hist_n_ago(1)
    return _safe_div(series.macd_hist_now(), prev)


def _price_ema_state(series: BarSeries) -> str:
    close = series.last_close
    fast = series.ema_fast()
    slow = series.ema_slow()
    fast_side = "above_fast" if close >= fast else "below_fast"
    slow_side = "above_slow" if close >= slow else "below_slow"
    stack = "fast_gt_slow" if fast >= slow else "fast_lt_slow"
    return f"{fast_side}_{slow_side}_{stack}"


def _signed_vs_ema_atr(direction: int, close: float, ema: float, atr: float) -> float:
    return _safe_div(direction * (close - ema), atr)


def _setup_diagnostics(
    *,
    direction: int,
    entry: float,
    current_price: float,
    h1: BarSeries,
    h4: BarSeries,
    daily: BarSeries,
    context_score: int,
    pullback_depth: float = 0.0,
    pullback_atr: float = 0.0,
    ltf: BarSeries | None = None,
    ltf_price: float | None = None,
    ltf_pullback_depth: float = 0.0,
    ltf_atr: float = 0.0,
) -> dict[str, object]:
    atr1 = h1.current_atr()
    gap = direction * (entry - current_price)
    diag: dict[str, object] = {
        "entry_gap_points": gap,
        "entry_gap_atr": _safe_div(gap, atr1),
        "pullback_depth_atr": _safe_div(pullback_depth, pullback_atr or atr1),
        "pullback_depth_h1_atr": _safe_div(pullback_depth, atr1),
        "h4_hist_ratio": _hist_ratio(h4),
        "price_ema_state": _price_ema_state(h1),
        "price_vs_ema_fast_atr": _signed_vs_ema_atr(direction, h1.last_close, h1.ema_fast(), atr1),
        "price_vs_ema_slow_atr": _signed_vs_ema_atr(direction, h1.last_close, h1.ema_slow(), atr1),
        "context_quality_score": context_score,
        "trend_strength_daily": trend_strength(daily),
    }
    if ltf is not None:
        price = ltf.last_close if ltf_price is None else ltf_price
        atr_ltf = ltf_atr or ltf.current_atr()
        diag.update({
            "entry_gap_ltf_atr": _safe_div(gap, atr_ltf),
            "pullback_depth_m5_atr": _safe_div(ltf_pullback_depth or pullback_depth, atr_ltf),
            "ltf_price_ema_state": _price_ema_state(ltf),
            "ltf_price_vs_ema_fast_atr": _signed_vs_ema_atr(direction, price, ltf.ema_fast(), atr_ltf),
            "ltf_price_vs_ema_slow_atr": _signed_vs_ema_atr(direction, price, ltf.ema_slow(), atr_ltf),
        })
    return diag


def _class_t_secondary_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and CLASS_T_SECONDARY_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and CLASS_T_SECONDARY_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and CLASS_T_SECONDARY_ALLOW_ETH_QUALITY_PM)
    )


def _class_t_short_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.ETH_QUALITY_PM and CLASS_T_SHORT_ALLOW_ETH_QUALITY_PM)
        or (block == SessionBlock.RTH_PRIME2 and CLASS_T_SHORT_ALLOW_RTH_PRIME2)
    )


def _followthrough_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and FOLLOWTHROUGH_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and FOLLOWTHROUGH_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and FOLLOWTHROUGH_ALLOW_ETH_QUALITY_PM)
    )


def _second_chance_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.ETH_QUALITY_AM and SECOND_CHANCE_ALLOW_ETH_QUALITY_AM)
        or (block == SessionBlock.RTH_PRIME1 and SECOND_CHANCE_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and SECOND_CHANCE_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and SECOND_CHANCE_ALLOW_ETH_QUALITY_PM)
    )


def _high_vol_retest_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and HIGH_VOL_RETEST_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and HIGH_VOL_RETEST_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and HIGH_VOL_RETEST_ALLOW_ETH_QUALITY_PM)
    )


def _high_vol_m_cont_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and HIGH_VOL_M_CONT_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and HIGH_VOL_M_CONT_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.RTH_DEAD and HIGH_VOL_M_CONT_ALLOW_RTH_DEAD)
        or (block == SessionBlock.ETH_QUALITY_PM and HIGH_VOL_M_CONT_ALLOW_ETH_QUALITY_PM)
    )


def _norm_vol_m_short_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_DEAD and NORM_VOL_M_SHORT_ALLOW_RTH_DEAD)
        or (block == SessionBlock.RTH_PRIME2 and NORM_VOL_M_SHORT_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and NORM_VOL_M_SHORT_ALLOW_ETH_QUALITY_PM)
    )


def _class_f_reclaim_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and CLASS_F_RECLAIM_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and CLASS_F_RECLAIM_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and CLASS_F_RECLAIM_ALLOW_ETH_QUALITY_PM)
    )


def _trend_retest_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and TREND_RETEST_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and TREND_RETEST_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.RTH_DEAD and TREND_RETEST_ALLOW_RTH_DEAD)
        or (block == SessionBlock.ETH_QUALITY_PM and TREND_RETEST_ALLOW_ETH_QUALITY_PM)
    )


def _failed_reclaim_short_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_DEAD and FAILED_RECLAIM_SHORT_ALLOW_RTH_DEAD)
        or (block == SessionBlock.RTH_PRIME2 and FAILED_RECLAIM_SHORT_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and FAILED_RECLAIM_SHORT_ALLOW_ETH_QUALITY_PM)
    )


def _fallback_flag(value: object, default: bool) -> bool:
    return default if value is None else bool(value)


def _micro_trend_retest_session_allowed(block: SessionBlock, direction: int) -> bool:
    shared = {
        SessionBlock.RTH_PRIME1: MICRO_TREND_RETEST_ALLOW_RTH_PRIME1,
        SessionBlock.RTH_PRIME2: MICRO_TREND_RETEST_ALLOW_RTH_PRIME2,
        SessionBlock.RTH_DEAD: MICRO_TREND_RETEST_ALLOW_RTH_DEAD,
        SessionBlock.ETH_QUALITY_AM: MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_AM,
        SessionBlock.ETH_QUALITY_PM: MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_PM,
    }
    long_specific = {
        SessionBlock.RTH_PRIME1: MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME1,
        SessionBlock.RTH_PRIME2: MICRO_TREND_RETEST_ALLOW_LONG_RTH_PRIME2,
        SessionBlock.RTH_DEAD: MICRO_TREND_RETEST_ALLOW_LONG_RTH_DEAD,
        SessionBlock.ETH_QUALITY_AM: MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_AM,
        SessionBlock.ETH_QUALITY_PM: MICRO_TREND_RETEST_ALLOW_LONG_ETH_QUALITY_PM,
    }
    short_specific = {
        SessionBlock.RTH_PRIME1: MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME1,
        SessionBlock.RTH_PRIME2: MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME2,
        SessionBlock.RTH_DEAD: MICRO_TREND_RETEST_ALLOW_SHORT_RTH_DEAD,
        SessionBlock.ETH_QUALITY_AM: MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_AM,
        SessionBlock.ETH_QUALITY_PM: MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_PM,
    }
    default = bool(shared.get(block, False))
    specific = long_specific.get(block) if direction == 1 else short_specific.get(block)
    return _fallback_flag(specific, default)


def _m15_trend_retest_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and M15_TREND_RETEST_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and M15_TREND_RETEST_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.RTH_DEAD and M15_TREND_RETEST_ALLOW_RTH_DEAD)
        or (block == SessionBlock.ETH_QUALITY_AM and M15_TREND_RETEST_ALLOW_ETH_QUALITY_AM)
        or (block == SessionBlock.ETH_QUALITY_PM and M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM)
    )


def _second_chance_cancel_reason_safe(reason: str) -> bool:
    reason = (reason or "").upper()
    if not reason:
        return False
    safe_prefixes = (
        "TTL_EXPIRED",
        "EOB_BACKSTOP",
        "EXPIRED",
        "CANCELLED",
    )
    unsafe_prefixes = (
        "PRE_HALT",
        "STRUCTURE_INVALID",
        "OMS_CANCELLED",
    )
    return reason.startswith(safe_prefixes) and not reason.startswith(unsafe_prefixes)


def _directional_trend_ok(direction: int, h1: BarSeries) -> bool:
    if direction == 1:
        return h1.ema_fast() > h1.ema_slow() and h1.macd_line_now() > 0 and h1.last_close > h1.ema_fast()
    return h1.ema_fast() < h1.ema_slow() and h1.macd_line_now() < 0 and h1.last_close < h1.ema_fast()


def followthrough_catchup_allowed(
    setup: Setup,
    h1: BarSeries,
    h4: BarSeries,
    daily: BarSeries,
    now_et: datetime,
) -> bool:
    if not FOLLOWTHROUGH_CATCHUP_ENABLED or setup.followthrough_used:
        return False
    if setup.cls == SetupClass.M:
        if not FOLLOWTHROUGH_ALLOW_CLASS_M:
            return False
    elif setup.cls == SetupClass.T:
        if not FOLLOWTHROUGH_ALLOW_CLASS_T:
            return False
    else:
        return False

    if setup.placed_ts is not None:
        age_h = (now_et - setup.placed_ts).total_seconds() / 3600.0
        if age_h > FOLLOWTHROUGH_MAX_BARS_SINCE_ARM:
            return False

    if not _followthrough_session_allowed(get_session_block(now_et)):
        return False

    atr1 = h1.current_atr()
    if atr1 <= 0:
        return False

    score = alignment_score(setup.direction, daily, h4)
    if score < FOLLOWTHROUGH_MIN_ALIGNMENT_SCORE:
        return False
    if FOLLOWTHROUGH_REQUIRE_STRONG_TREND and not strong_trend_flag(score, daily):
        return False

    if setup.direction == 1:
        gap = setup.entry_stop - h1.last_close
        trend_ok = h1.ema_fast() > h1.ema_slow() and h1.macd_line_now() > 0
    else:
        gap = h1.last_close - setup.entry_stop
        trend_ok = h1.ema_fast() < h1.ema_slow() and h1.macd_line_now() < 0
    if not trend_ok:
        return False
    if gap <= 0 or gap > FOLLOWTHROUGH_MAX_GAP_ATR * atr1:
        return False
    return True


@dataclass
class _TrailingExitRecord:
    direction: int
    bars_since: int = 0
    exit_price: float = 0.0
    exit_ts: Optional[datetime] = None


@dataclass
class _UnfilledSetupRecord:
    setup: Setup
    cancel_ts: datetime
    cancel_reason: str = ""
    bars_since: int = 0
    rearmed: bool = False


class SignalEngine:
    """Detects setups from pivot and indicator state."""

    def __init__(self):
        self._last_exit_dir: dict[int, int] = {}
        self._reentry_used: dict[int, bool] = {}
        self._trend_strength_3d: Optional[float] = None
        self._trailing_exits: list[_TrailingExitRecord] = []
        self._recent_unfilled: list[_UnfilledSetupRecord] = []
        self._variant_cooldowns: dict[tuple[str, int], int] = {}
        self._micro_variant_cooldowns: dict[tuple[str, int], int] = {}
        self._m15_variant_cooldowns: dict[tuple[str, int], int] = {}

    def update_trend_strength_3d(self, val: Optional[float]) -> None:
        self._trend_strength_3d = val

    def record_exit(
        self,
        direction: int,
        exit_reason: str = "",
        bars_held: int = 0,
        exit_price: float = 0.0,
        exit_ts: Optional[datetime] = None,
    ) -> None:
        del bars_held
        self._last_exit_dir[direction] = 0
        if exit_reason == "TRAILING_STOP":
            self._trailing_exits.append(
                _TrailingExitRecord(
                    direction=direction,
                    bars_since=0,
                    exit_price=exit_price,
                    exit_ts=exit_ts,
                )
            )

    def record_unfilled_setup(
        self,
        setup: Setup,
        cancel_ts: datetime,
        cancel_reason: str = "",
    ) -> None:
        if setup.cls not in {SetupClass.M, SetupClass.T}:
            return
        if setup.signal_variant:
            return
        if any(record.setup.setup_id == setup.setup_id for record in self._recent_unfilled):
            return
        self._recent_unfilled.append(
            _UnfilledSetupRecord(
                setup=setup,
                cancel_ts=cancel_ts,
                cancel_reason=cancel_reason,
            )
        )

    def tick_bars(self) -> None:
        for direction in list(self._last_exit_dir):
            self._last_exit_dir[direction] += 1
            if self._last_exit_dir[direction] > REENTRY_LOOKBACK_1H:
                del self._last_exit_dir[direction]
        max_trailing_age = max(CLASS_F_WINDOW_BARS, CLASS_F_RECLAIM_MAX_EXIT_BARS) + 1
        for record in list(self._trailing_exits):
            record.bars_since += 1
            if record.bars_since > max_trailing_age:
                self._trailing_exits.remove(record)
        max_unfilled_age = SECOND_CHANCE_MAX_BARS_SINCE_CANCEL + 2
        for record in list(self._recent_unfilled):
            record.bars_since += 1
            if record.bars_since > max_unfilled_age:
                self._recent_unfilled.remove(record)
        max_variant_age = max(TREND_RETEST_MIN_BARS_BETWEEN, FAILED_RECLAIM_SHORT_MIN_BARS_BETWEEN) + 1
        for key in list(self._variant_cooldowns):
            self._variant_cooldowns[key] += 1
            if self._variant_cooldowns[key] > max_variant_age:
                del self._variant_cooldowns[key]

    def tick_micro_bars(self) -> None:
        max_micro_age = MICRO_TREND_RETEST_MIN_BARS_BETWEEN + 1
        for key in list(self._micro_variant_cooldowns):
            self._micro_variant_cooldowns[key] += 1
            if self._micro_variant_cooldowns[key] > max_micro_age:
                del self._micro_variant_cooldowns[key]

    def tick_m15_bars(self) -> None:
        max_m15_age = M15_TREND_RETEST_MIN_BARS_BETWEEN + 1
        for key in list(self._m15_variant_cooldowns):
            self._m15_variant_cooldowns[key] += 1
            if self._m15_variant_cooldowns[key] > max_m15_age:
                del self._m15_variant_cooldowns[key]

    def _variant_on_cooldown(self, variant: str, direction: int, min_bars: int) -> bool:
        age = self._variant_cooldowns.get((variant, direction))
        return age is not None and age < min_bars

    def _mark_variant_fired(self, variant: str, direction: int) -> None:
        self._variant_cooldowns[(variant, direction)] = 0

    def _micro_variant_on_cooldown(self, variant: str, direction: int, min_bars: int) -> bool:
        age = self._micro_variant_cooldowns.get((variant, direction))
        return age is not None and age < min_bars

    def _mark_micro_variant_fired(self, variant: str, direction: int) -> None:
        self._micro_variant_cooldowns[(variant, direction)] = 0

    def _m15_variant_on_cooldown(self, variant: str, direction: int, min_bars: int) -> bool:
        age = self._m15_variant_cooldowns.get((variant, direction))
        return age is not None and age < min_bars

    def _mark_m15_variant_fired(self, variant: str, direction: int) -> None:
        self._m15_variant_cooldowns[(variant, direction)] = 0

    def detect_class_M(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
    ) -> list[Setup]:
        setups: list[Setup] = []
        for direction in (1, -1):
            if direction == 1:
                p1, p2 = pivots_1h.last_two_lows()
                if not (p1 and p2 and p2.price > p1.price):
                    continue
            else:
                p1, p2 = pivots_1h.last_two_highs()
                if not (p1 and p2 and p2.price < p1.price):
                    continue

            score = alignment_score(direction, daily, h4)
            if extension_blocked(direction, daily):
                continue

            atr1 = h1.current_atr()
            if atr1 <= 0:
                continue

            if direction == 1:
                recent_high = pivots_1h.most_recent_pivot_high()
                if not recent_high:
                    continue
                pb_depth = recent_high.price - p2.price
            else:
                recent_low = pivots_1h.most_recent_pivot_low()
                if not recent_low:
                    continue
                pb_depth = p2.price - recent_low.price

            if not (CLASS_M_PB_MIN * atr1 <= pb_depth <= CLASS_M_PB_MAX * atr1):
                continue

            macd_now = h1.macd_line_now()
            macd_3ago = h1.macd_line_n_ago(CLASS_M_MOMENTUM_LOOKBACK)
            if direction == 1:
                if not (macd_now > macd_3ago or macd_now > p2.macd):
                    continue
                breakout_pivot = pivots_1h.pivot_high_between(p1.ts, p2.ts)
                if not breakout_pivot:
                    continue
                buf = _buffer(atr1)
                entry = breakout_pivot.price + buf
                stop0 = p2.price - CLASS_M_STOP_ATR_MULT * atr1
            else:
                if not (macd_now < macd_3ago or macd_now < p2.macd):
                    continue
                breakout_pivot = pivots_1h.pivot_low_between(p1.ts, p2.ts)
                if not breakout_pivot:
                    continue
                buf = _buffer(atr1)
                entry = breakout_pivot.price - buf
                stop0 = p2.price + CLASS_M_STOP_ATR_MULT * atr1

            setup = Setup(
                setup_id=str(uuid.uuid4())[:8],
                cls=SetupClass.M,
                direction=direction,
                tf_origin=TF.H1,
                detected_ts=now_et,
                P1=p1,
                P2=p2,
                breakout_pivot=breakout_pivot,
                entry_stop=entry,
                stop0=stop0,
                buffer=buf,
                alignment_score=score,
                strong_trend=strong_trend_flag(score, daily),
                is_extended=False,
            )

            if (
                direction in self._last_exit_dir
                and self._last_exit_dir[direction] <= REENTRY_LOOKBACK_1H
                and score >= 1
                and not self._reentry_used.get(direction, False)
            ):
                setup.is_reentry = True
                self._reentry_used[direction] = True

            setups.append(setup)
        return setups

    def detect_second_chance_setups(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
    ) -> list[Setup]:
        setups: list[Setup] = []
        if not SECOND_CHANCE_REARM_ENABLED:
            return setups
        if not _second_chance_session_allowed(get_session_block(now_et)):
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        for record in self._recent_unfilled:
            if record.rearmed or record.bars_since > SECOND_CHANCE_MAX_BARS_SINCE_CANCEL:
                continue
            if not _second_chance_cancel_reason_safe(record.cancel_reason):
                continue
            base = record.setup
            if base.cls == SetupClass.M and not SECOND_CHANCE_ALLOW_CLASS_M:
                continue
            if base.cls == SetupClass.T and not SECOND_CHANCE_ALLOW_CLASS_T:
                continue
            if base.cls not in {SetupClass.M, SetupClass.T}:
                continue

            direction = base.direction
            score = alignment_score(direction, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            if score < SECOND_CHANCE_MIN_ALIGNMENT_SCORE:
                continue
            if SECOND_CHANCE_REQUIRE_STRONG_TREND and not strong_trend:
                continue
            if not _directional_trend_ok(direction, h1):
                continue

            if SECOND_CHANCE_USE_BASE_TRIGGER:
                breakout_pivot = base.breakout_pivot
                if breakout_pivot is None:
                    continue
                buf = base.buffer or _buffer(atr1)
                entry = base.entry_stop

                if direction == 1:
                    recent_support = pivots_1h.most_recent_pivot_low()
                    if base.P2 is None:
                        continue
                    if recent_support and recent_support.ts > base.detected_ts and recent_support.price <= base.P2.price:
                        continue
                    support_price = base.P2.price
                    if recent_support and recent_support.ts > base.detected_ts and recent_support.price > support_price:
                        support_price = recent_support.price
                    if support_price < h1.ema_slow() - SECOND_CHANCE_PULLBACK_EMA_TOL_ATR * atr1:
                        continue
                    breakout_gap = entry - h1.last_close
                    if breakout_gap < 0 or breakout_gap > SECOND_CHANCE_MAX_BREAKOUT_GAP_ATR * atr1:
                        continue
                    stop0 = support_price - SECOND_CHANCE_STOP_ATR_MULT * atr1
                    stop_dist = entry - stop0
                else:
                    recent_support = pivots_1h.most_recent_pivot_high()
                    if base.P2 is None:
                        continue
                    if recent_support and recent_support.ts > base.detected_ts and recent_support.price >= base.P2.price:
                        continue
                    support_price = base.P2.price
                    if recent_support and recent_support.ts > base.detected_ts and recent_support.price < support_price:
                        support_price = recent_support.price
                    if support_price > h1.ema_slow() + SECOND_CHANCE_PULLBACK_EMA_TOL_ATR * atr1:
                        continue
                    breakout_gap = h1.last_close - entry
                    if breakout_gap < 0 or breakout_gap > SECOND_CHANCE_MAX_BREAKOUT_GAP_ATR * atr1:
                        continue
                    stop0 = support_price + SECOND_CHANCE_STOP_ATR_MULT * atr1
                    stop_dist = stop0 - entry
                if stop_dist <= 0 or stop_dist > SECOND_CHANCE_MAX_STOP_DIST_ATR * atr1:
                    continue
            elif direction == 1:
                breakout_pivot = pivots_1h.most_recent_pivot_high()
                support_pivot = pivots_1h.most_recent_pivot_low()
                if not breakout_pivot or not support_pivot:
                    continue
                anchor_ts = record.cancel_ts if SECOND_CHANCE_REQUIRE_FRESH_PIVOT else base.detected_ts
                if breakout_pivot.ts <= anchor_ts and support_pivot.ts <= anchor_ts:
                    continue
                if support_pivot.price < h1.ema_slow() - SECOND_CHANCE_PULLBACK_EMA_TOL_ATR * atr1:
                    continue
                breakout_gap = breakout_pivot.price - h1.last_close
                if breakout_gap < 0 or breakout_gap > SECOND_CHANCE_MAX_BREAKOUT_GAP_ATR * atr1:
                    continue
                buf = _buffer(atr1)
                entry = breakout_pivot.price + buf
                stop0 = support_pivot.price - SECOND_CHANCE_STOP_ATR_MULT * atr1
                stop_dist = entry - stop0
                if stop_dist <= 0 or stop_dist > SECOND_CHANCE_MAX_STOP_DIST_ATR * atr1:
                    continue
            else:
                breakout_pivot = pivots_1h.most_recent_pivot_low()
                support_pivot = pivots_1h.most_recent_pivot_high()
                if not breakout_pivot or not support_pivot:
                    continue
                anchor_ts = record.cancel_ts if SECOND_CHANCE_REQUIRE_FRESH_PIVOT else base.detected_ts
                if breakout_pivot.ts <= anchor_ts and support_pivot.ts <= anchor_ts:
                    continue
                if support_pivot.price > h1.ema_slow() + SECOND_CHANCE_PULLBACK_EMA_TOL_ATR * atr1:
                    continue
                breakout_gap = h1.last_close - breakout_pivot.price
                if breakout_gap < 0 or breakout_gap > SECOND_CHANCE_MAX_BREAKOUT_GAP_ATR * atr1:
                    continue
                buf = _buffer(atr1)
                entry = breakout_pivot.price - buf
                stop0 = support_pivot.price + SECOND_CHANCE_STOP_ATR_MULT * atr1
                stop_dist = stop0 - entry
                if stop_dist <= 0 or stop_dist > SECOND_CHANCE_MAX_STOP_DIST_ATR * atr1:
                    continue

            record.rearmed = True
            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=base.cls,
                    direction=direction,
                    tf_origin=base.tf_origin,
                    detected_ts=now_et,
                    P1=None,
                    P2=None,
                    breakout_pivot=breakout_pivot,
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    is_reentry=True,
                    signal_variant="second_chance",
                    parent_setup_id=base.setup_id,
                )
            )
        return setups

    def detect_class_F(
        self,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        pivots_1h: PivotDetector | None = None,
        vol_pct: float | None = None,
        include_base: bool = True,
    ) -> list[Setup]:
        setups = self._detect_class_F_base(h1, h4, daily, now_et, vol_pct) if include_base else []
        if CLASS_F_RECLAIM_ENABLED:
            reclaim_setups = self._detect_class_F_reclaim(
                h1,
                h4,
                daily,
                now_et,
                pivots_1h,
                vol_pct,
                blocked_directions={setup.direction for setup in setups},
            )
            setups.extend(reclaim_setups)
        return setups

    def _detect_class_F_base(
        self,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float | None = None,
    ) -> list[Setup]:
        setups: list[Setup] = []
        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        for record in self._trailing_exits:
            direction = record.direction
            last_close = h1.last_close
            macd_now = h1.macd_line_now()
            if direction == 1 and macd_now <= 0:
                continue
            if direction == -1 and macd_now >= 0:
                continue

            lookback = min(record.bars_since + 1, len(h1.bars))
            if lookback < 2:
                continue

            if direction == 1:
                ref_extreme = h1.highest_high(lookback)
                pullback = ref_extreme - last_close
            else:
                ref_extreme = h1.lowest_low(lookback)
                pullback = last_close - ref_extreme
            if not (CLASS_F_MIN_PB_ATR * atr1 <= pullback <= CLASS_F_MAX_PB_ATR * atr1):
                continue

            score = alignment_score(direction, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            if score < CLASS_F_MIN_ALIGNMENT_SCORE:
                continue
            block = get_session_block(now_et)
            context_score = context_quality_score_for_features(
                SetupClass.F,
                direction,
                block,
                vol_pct,
                strong_trend,
            )
            if not meets_context_quality(CLASS_F_MIN_CONTEXT_SCORE, context_score):
                continue
            if CLASS_F_REQUIRE_STRONG_TREND and not strong_trend:
                continue

            buf = _buffer(atr1)
            if direction == 1:
                entry = last_close + buf
                stop0 = last_close - CLASS_F_STOP_ATR_MULT * atr1
            else:
                entry = last_close - buf
                stop0 = last_close + CLASS_F_STOP_ATR_MULT * atr1

            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=SetupClass.F,
                    direction=direction,
                    tf_origin=TF.H1,
                    detected_ts=now_et,
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    parent_exit_bar=record.bars_since,
                )
            )
        return setups

    def _detect_class_F_reclaim(
        self,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        pivots_1h: PivotDetector | None,
        vol_pct: float | None,
        blocked_directions: set[int] | None = None,
    ) -> list[Setup]:
        setups: list[Setup] = []
        if pivots_1h is None:
            return setups
        if not _class_f_reclaim_session_allowed(get_session_block(now_et)):
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        for record in self._trailing_exits:
            if record.bars_since > CLASS_F_RECLAIM_MAX_EXIT_BARS:
                continue
            if blocked_directions and record.direction in blocked_directions:
                continue
            if vol_pct is not None and vol_pct < CLASS_F_RECLAIM_MIN_VOL_PCT:
                continue

            direction = record.direction
            score = alignment_score(direction, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            if score < CLASS_F_RECLAIM_MIN_ALIGNMENT_SCORE:
                continue
            block = get_session_block(now_et)
            context_score = context_quality_score_for_features(
                SetupClass.F,
                direction,
                block,
                vol_pct,
                strong_trend,
            )
            if not meets_context_quality(CLASS_F_RECLAIM_MIN_CONTEXT_SCORE, context_score):
                continue
            if CLASS_F_RECLAIM_REQUIRE_STRONG_TREND and not strong_trend:
                continue
            if not _directional_trend_ok(direction, h1):
                continue

            lookback = min(max(record.bars_since + 2, 2), len(h1.bars))
            if lookback < 2:
                continue
            buf = _buffer(atr1)

            if direction == 1:
                recent_low = pivots_1h.most_recent_pivot_low()
                if not recent_low:
                    continue
                if record.exit_ts is not None and recent_low.ts <= record.exit_ts:
                    continue
                if recent_low.price < h1.ema_slow() - CLASS_F_RECLAIM_EMA_HOLD_TOL_ATR * atr1:
                    continue
                ref_extreme = h1.highest_high(lookback)
                pullback = ref_extreme - recent_low.price
                if not (CLASS_F_MIN_PB_ATR * atr1 <= pullback <= CLASS_F_RECLAIM_MAX_PULLBACK_ATR * atr1):
                    continue
                breakout_ref = max(h1.highest_high(min(lookback, 3)), h1.last_close)
                entry = breakout_ref + buf
                stop0 = recent_low.price - CLASS_F_RECLAIM_STOP_ATR_MULT * atr1
                stop_dist = entry - stop0
                if stop_dist <= 0 or stop_dist > CLASS_F_RECLAIM_MAX_STOP_DIST_ATR * atr1:
                    continue
            else:
                recent_high = pivots_1h.most_recent_pivot_high()
                if not recent_high:
                    continue
                if record.exit_ts is not None and recent_high.ts <= record.exit_ts:
                    continue
                if recent_high.price > h1.ema_slow() + CLASS_F_RECLAIM_EMA_HOLD_TOL_ATR * atr1:
                    continue
                ref_extreme = h1.lowest_low(lookback)
                pullback = recent_high.price - ref_extreme
                if not (CLASS_F_MIN_PB_ATR * atr1 <= pullback <= CLASS_F_RECLAIM_MAX_PULLBACK_ATR * atr1):
                    continue
                breakout_ref = min(h1.lowest_low(min(lookback, 3)), h1.last_close)
                entry = breakout_ref - buf
                stop0 = recent_high.price + CLASS_F_RECLAIM_STOP_ATR_MULT * atr1
                stop_dist = stop0 - entry
                if stop_dist <= 0 or stop_dist > CLASS_F_RECLAIM_MAX_STOP_DIST_ATR * atr1:
                    continue

            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=SetupClass.F,
                    direction=direction,
                    tf_origin=TF.H1,
                    detected_ts=now_et,
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    parent_exit_bar=record.bars_since,
                    signal_variant="f_reclaim",
                )
            )
        return setups

    def detect_high_vol_retest_setups(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float,
    ) -> list[Setup]:
        setups: list[Setup] = []
        if not HIGH_VOL_RETEST_ENABLED:
            return setups
        if not _high_vol_retest_session_allowed(get_session_block(now_et)):
            return setups
        if vol_pct < HIGH_VOL_RETEST_MIN_VOL_PCT or vol_pct > HIGH_VOL_RETEST_MAX_VOL_PCT:
            return setups
        if not (daily.ema_fast() > daily.ema_slow() and h4.ema_fast() > h4.ema_slow()):
            return setups
        if not _directional_trend_ok(1, h1):
            return setups
        if trend_strength(daily) < HIGH_VOL_RETEST_MIN_TREND_STRENGTH:
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        h4_hist_now = h4.macd_hist_now()
        h4_hist_prev = h4.macd_hist_n_ago(CLASS_T_HIST_LOOKBACK)
        if h4_hist_now <= 0:
            return setups
        if h4_hist_prev > 0 and h4_hist_now < HIGH_VOL_RETEST_H4_HIST_RATIO_MIN * h4_hist_prev:
            return setups

        recent_ph = pivots_1h.most_recent_pivot_high()
        recent_pl = pivots_1h.most_recent_pivot_low()
        if not recent_ph or not recent_pl:
            return setups
        if recent_pl.ts <= recent_ph.ts:
            return setups
        hours_since_pullback = (now_et - recent_pl.ts).total_seconds() / 3600.0
        if hours_since_pullback > HIGH_VOL_RETEST_PIVOT_RECENCY_H:
            return setups

        pullback = recent_ph.price - recent_pl.price
        if pullback <= 0 or pullback > HIGH_VOL_RETEST_MAX_PULLBACK_ATR * atr1:
            return setups
        if recent_pl.price < h1.ema_fast() - HIGH_VOL_RETEST_EMA_HOLD_TOL_ATR * atr1:
            return setups

        score = alignment_score(1, daily, h4)
        strong_trend = strong_trend_flag(score, daily)
        if score < HIGH_VOL_RETEST_MIN_ALIGNMENT_SCORE:
            return setups
        if HIGH_VOL_RETEST_REQUIRE_STRONG_TREND and not strong_trend:
            return setups

        breakout_gap = recent_ph.price - h1.last_close
        if breakout_gap < 0 or breakout_gap > HIGH_VOL_RETEST_MAX_BREAKOUT_GAP_ATR * atr1:
            return setups

        buf = _buffer(atr1)
        entry = recent_ph.price + buf
        stop0 = recent_pl.price - HIGH_VOL_RETEST_STOP_ATR_MULT * atr1
        stop_dist = entry - stop0
        if stop_dist <= 0 or stop_dist > HIGH_VOL_RETEST_MAX_STOP_DIST_ATR * atr1:
            return setups

        setups.append(
            Setup(
                setup_id=str(uuid.uuid4())[:8],
                cls=SetupClass.T,
                direction=1,
                tf_origin=TF.H4,
                detected_ts=now_et,
                P1=None,
                P2=None,
                breakout_pivot=recent_ph,
                entry_stop=entry,
                stop0=stop0,
                buffer=buf,
                alignment_score=score,
                strong_trend=strong_trend,
                is_extended=False,
                signal_variant="high_vol_retest",
            )
        )
        return setups

    def detect_high_vol_m_continuation_setups(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float,
        base_setups: list[Setup] | None = None,
    ) -> list[Setup]:
        setups: list[Setup] = []
        if not HIGH_VOL_M_CONT_ENABLED:
            return setups
        block = get_session_block(now_et)
        if not _high_vol_m_cont_session_allowed(block):
            return setups
        if vol_pct < HIGH_VOL_M_CONT_MIN_VOL_PCT or vol_pct > HIGH_VOL_M_CONT_MAX_VOL_PCT:
            return setups
        if trend_strength(daily) < HIGH_VOL_M_CONT_MIN_TREND_STRENGTH:
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        source_setups = [
            setup for setup in (base_setups or [])
            if setup.cls == SetupClass.M and not setup.signal_variant
        ]

        for base in source_setups:
            direction = base.direction
            if direction == 1 and not HIGH_VOL_M_CONT_ALLOW_LONG:
                continue
            if direction == -1 and not HIGH_VOL_M_CONT_ALLOW_SHORT:
                continue
            if base.P1 is None or base.P2 is None or base.breakout_pivot is None:
                continue

            score = alignment_score(direction, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            if score < HIGH_VOL_M_CONT_MIN_ALIGNMENT_SCORE:
                continue
            min_context_score = (
                HIGH_VOL_M_CONT_MIN_CONTEXT_SCORE_LONG
                if direction == 1
                else HIGH_VOL_M_CONT_MIN_CONTEXT_SCORE_SHORT
            )
            context_score = context_quality_score_for_features(
                SetupClass.M,
                direction,
                block,
                vol_pct,
                strong_trend,
            )
            if not meets_context_quality(min_context_score, context_score):
                continue
            if HIGH_VOL_M_CONT_REQUIRE_STRONG_TREND and not strong_trend:
                continue

            buf = _buffer(atr1)
            if direction == 1:
                entry = base.breakout_pivot.price + buf
                stop0 = base.P2.price - HIGH_VOL_M_CONT_STOP_ATR_MULT * atr1
                stop_dist = entry - stop0
            else:
                entry = base.breakout_pivot.price - buf
                stop0 = base.P2.price + HIGH_VOL_M_CONT_STOP_ATR_MULT * atr1
                stop_dist = stop0 - entry

            if stop_dist <= 0 or stop_dist > HIGH_VOL_M_CONT_MAX_STOP_DIST_ATR * atr1:
                continue

            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=SetupClass.M,
                    direction=direction,
                    tf_origin=TF.H1,
                    detected_ts=now_et,
                    P1=base.P1,
                    P2=base.P2,
                    breakout_pivot=base.breakout_pivot,
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    signal_variant="high_vol_m_cont",
                )
            )
        return setups

    def detect_normal_vol_m_short_setups(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float,
        base_setups: list[Setup] | None = None,
    ) -> list[Setup]:
        del pivots_1h
        setups: list[Setup] = []
        if not NORM_VOL_M_SHORT_ENABLED:
            return setups
        block = get_session_block(now_et)
        if not _norm_vol_m_short_session_allowed(block):
            return setups
        if vol_pct < NORM_VOL_M_SHORT_MIN_VOL_PCT or vol_pct > NORM_VOL_M_SHORT_MAX_VOL_PCT:
            return setups
        if trend_strength(daily) < NORM_VOL_M_SHORT_MIN_TREND_STRENGTH:
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups

        source_setups = [
            setup for setup in (base_setups or [])
            if setup.cls == SetupClass.M and setup.direction == -1 and not setup.signal_variant
        ]

        for base in source_setups:
            if base.P1 is None or base.P2 is None or base.breakout_pivot is None:
                continue

            score = alignment_score(-1, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            if score < NORM_VOL_M_SHORT_MIN_ALIGNMENT_SCORE:
                continue
            context_score = context_quality_score_for_features(
                SetupClass.M,
                -1,
                block,
                vol_pct,
                strong_trend,
            )
            if not meets_context_quality(NORM_VOL_M_SHORT_MIN_CONTEXT_SCORE, context_score):
                continue
            if NORM_VOL_M_SHORT_REQUIRE_STRONG_TREND and not strong_trend:
                continue

            breakout_gap = h1.last_close - base.breakout_pivot.price
            if breakout_gap < 0 or breakout_gap > NORM_VOL_M_SHORT_MAX_BREAKOUT_GAP_ATR * atr1:
                continue

            buf = _buffer(atr1)
            entry = base.breakout_pivot.price - buf
            stop0 = base.P2.price + NORM_VOL_M_SHORT_STOP_ATR_MULT * atr1
            stop_dist = stop0 - entry
            if stop_dist <= 0 or stop_dist > NORM_VOL_M_SHORT_MAX_STOP_DIST_ATR * atr1:
                continue

            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=SetupClass.M,
                    direction=-1,
                    tf_origin=TF.H1,
                    detected_ts=now_et,
                    P1=base.P1,
                    P2=base.P2,
                    breakout_pivot=base.breakout_pivot,
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    signal_variant="normal_vol_m_short",
                )
            )
        return setups

    def detect_trend_retest_setups(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float,
    ) -> list[Setup]:
        del pivots_1h
        setups: list[Setup] = []
        variant = "trend_retest"
        if not TREND_RETEST_ENABLED:
            return setups
        block = get_session_block(now_et)
        if not _trend_retest_session_allowed(block):
            return setups
        if vol_pct < TREND_RETEST_MIN_VOL_PCT or vol_pct > TREND_RETEST_MAX_VOL_PCT:
            return setups
        if trend_strength(daily) < TREND_RETEST_MIN_TREND_STRENGTH:
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups
        lookback = max(2, int(TREND_RETEST_LOOKBACK_BARS))
        bars = list(h1.bars)
        if len(bars) < lookback + 1:
            return setups
        current = bars[-1]
        prior = bars[-lookback - 1:-1]
        recent = prior + [current]
        buf = _buffer(atr1)

        for direction in (1, -1):
            if direction == 1 and not TREND_RETEST_ALLOW_LONG:
                continue
            if direction == -1 and not TREND_RETEST_ALLOW_SHORT:
                continue
            if self._variant_on_cooldown(variant, direction, TREND_RETEST_MIN_BARS_BETWEEN):
                continue
            if extension_blocked(direction, daily):
                continue

            score = alignment_score(direction, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            min_score = (
                TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG
                if direction == 1
                else TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT
            )
            if score < min_score:
                continue
            if TREND_RETEST_REQUIRE_STRONG_TREND and not strong_trend:
                continue
            min_context = (
                TREND_RETEST_MIN_CONTEXT_SCORE_LONG
                if direction == 1
                else TREND_RETEST_MIN_CONTEXT_SCORE_SHORT
            )
            context_score = context_quality_score_for_features(
                SetupClass.T,
                direction,
                block,
                vol_pct,
                strong_trend,
            )
            if not meets_context_quality(min_context, context_score):
                continue

            if direction == 1:
                if not (h1.ema_fast() > h1.ema_slow() and h1.macd_line_now() > 0):
                    continue
                pullback_low = min(bar.low for bar in recent)
                trigger = max(bar.high for bar in prior)
                if pullback_low > h1.ema_fast() + TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr1:
                    continue
                if pullback_low < h1.ema_slow() - TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr1:
                    continue
                if current.close <= h1.ema_fast() or current.close < current.open:
                    continue
                pullback = trigger - pullback_low
                if pullback <= 0 or pullback > TREND_RETEST_MAX_PULLBACK_ATR * atr1:
                    continue
                breakout_gap = trigger - current.close
                if abs(breakout_gap) > TREND_RETEST_MAX_BREAKOUT_GAP_ATR * atr1:
                    continue
                entry = trigger + buf
                stop0 = pullback_low - TREND_RETEST_STOP_ATR_MULT * atr1
                stop_dist = entry - stop0
                kind = "PH"
                anchor_kind = "PL"
                anchor_price = pullback_low
            else:
                if not (h1.ema_fast() < h1.ema_slow() and h1.macd_line_now() < 0):
                    continue
                pullback_high = max(bar.high for bar in recent)
                trigger = min(bar.low for bar in prior)
                if pullback_high < h1.ema_fast() - TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr1:
                    continue
                if pullback_high > h1.ema_slow() + TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr1:
                    continue
                if current.close >= h1.ema_fast() or current.close > current.open:
                    continue
                pullback = pullback_high - trigger
                if pullback <= 0 or pullback > TREND_RETEST_MAX_PULLBACK_ATR * atr1:
                    continue
                breakout_gap = current.close - trigger
                if abs(breakout_gap) > TREND_RETEST_MAX_BREAKOUT_GAP_ATR * atr1:
                    continue
                entry = trigger - buf
                stop0 = pullback_high + TREND_RETEST_STOP_ATR_MULT * atr1
                stop_dist = stop0 - entry
                kind = "PL"
                anchor_kind = "PH"
                anchor_price = pullback_high

            if stop_dist <= 0 or stop_dist > TREND_RETEST_MAX_STOP_DIST_ATR * atr1:
                continue

            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=SetupClass.T,
                    direction=direction,
                    tf_origin=TF.H1,
                    detected_ts=now_et,
                    P2=Pivot(
                        ts=current.ts,
                        kind=anchor_kind,
                        price=anchor_price,
                        macd=h1.macd_line_now(),
                        atr=atr1,
                    ),
                    breakout_pivot=Pivot(
                        ts=current.ts,
                        kind=kind,
                        price=trigger,
                        macd=h1.macd_line_now(),
                        atr=atr1,
                    ),
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    signal_variant=variant,
                    diagnostic_context=_setup_diagnostics(
                        direction=direction,
                        entry=entry,
                        current_price=current.close,
                        h1=h1,
                        h4=h4,
                        daily=daily,
                        context_score=context_score,
                        pullback_depth=pullback,
                        pullback_atr=atr1,
                    ),
                )
            )
            self._mark_variant_fired(variant, direction)
        return setups

    def detect_failed_reclaim_short_setups(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float,
    ) -> list[Setup]:
        del pivots_1h
        setups: list[Setup] = []
        variant = "failed_reclaim_short"
        if not FAILED_RECLAIM_SHORT_ENABLED:
            return setups
        if self._variant_on_cooldown(variant, -1, FAILED_RECLAIM_SHORT_MIN_BARS_BETWEEN):
            return setups
        block = get_session_block(now_et)
        if not _failed_reclaim_short_session_allowed(block):
            return setups
        if vol_pct < FAILED_RECLAIM_SHORT_MIN_VOL_PCT or vol_pct > FAILED_RECLAIM_SHORT_MAX_VOL_PCT:
            return setups
        if trend_strength(daily) < FAILED_RECLAIM_SHORT_MIN_TREND_STRENGTH:
            return setups
        if extension_blocked(-1, daily):
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups
        lookback = max(2, int(FAILED_RECLAIM_SHORT_LOOKBACK_BARS))
        bars = list(h1.bars)
        if len(bars) < lookback + 1:
            return setups
        current = bars[-1]
        prior = bars[-lookback - 1:-1]
        recent = prior + [current]

        score = alignment_score(-1, daily, h4)
        strong_trend = strong_trend_flag(score, daily)
        if score < FAILED_RECLAIM_SHORT_MIN_ALIGNMENT_SCORE:
            return setups
        if FAILED_RECLAIM_SHORT_REQUIRE_STRONG_TREND and not strong_trend:
            return setups
        context_score = context_quality_score_for_features(
            SetupClass.T,
            -1,
            block,
            vol_pct,
            strong_trend,
        )
        if not meets_context_quality(FAILED_RECLAIM_SHORT_MIN_CONTEXT_SCORE, context_score):
            return setups

        if not (h1.ema_fast() < h1.ema_slow() or h1.macd_line_now() < h1.macd_line_n_ago(1)):
            return setups
        if current.close >= h1.ema_fast() or current.close > current.open:
            return setups

        reclaim_high = max(bar.high for bar in recent)
        trigger = min(bar.low for bar in prior)
        if reclaim_high < h1.ema_fast() - FAILED_RECLAIM_SHORT_EMA_TOL_ATR * atr1:
            return setups
        if score < 1 and reclaim_high > h1.ema_slow() + FAILED_RECLAIM_SHORT_EMA_TOL_ATR * atr1:
            return setups

        breakout_gap = current.close - trigger
        if abs(breakout_gap) > FAILED_RECLAIM_SHORT_MAX_BREAKOUT_GAP_ATR * atr1:
            return setups

        buf = _buffer(atr1)
        entry = trigger - buf
        stop0 = reclaim_high + FAILED_RECLAIM_SHORT_STOP_ATR_MULT * atr1
        stop_dist = stop0 - entry
        if stop_dist <= 0 or stop_dist > FAILED_RECLAIM_SHORT_MAX_STOP_DIST_ATR * atr1:
            return setups
        pullback = reclaim_high - trigger

        setups.append(
            Setup(
                setup_id=str(uuid.uuid4())[:8],
                cls=SetupClass.T,
                direction=-1,
                tf_origin=TF.H1,
                detected_ts=now_et,
                P2=Pivot(
                    ts=current.ts,
                    kind="PH",
                    price=reclaim_high,
                    macd=h1.macd_line_now(),
                    atr=atr1,
                ),
                breakout_pivot=Pivot(
                    ts=current.ts,
                    kind="PL",
                    price=trigger,
                    macd=h1.macd_line_now(),
                    atr=atr1,
                ),
                entry_stop=entry,
                stop0=stop0,
                buffer=buf,
                alignment_score=score,
                strong_trend=strong_trend,
                is_extended=False,
                signal_variant=variant,
                diagnostic_context=_setup_diagnostics(
                    direction=-1,
                    entry=entry,
                    current_price=current.close,
                    h1=h1,
                    h4=h4,
                    daily=daily,
                    context_score=context_score,
                    pullback_depth=pullback,
                    pullback_atr=atr1,
                ),
            )
        )
        self._mark_variant_fired(variant, -1)
        return setups

    def detect_micro_trend_retest_setups(
        self,
        m5: BarSeries,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float,
    ) -> list[Setup]:
        setups: list[Setup] = []
        variant = "micro_trend_retest"
        if not MICRO_TREND_RETEST_ENABLED:
            return setups
        block = get_session_block(now_et)
        if vol_pct < MICRO_TREND_RETEST_MIN_VOL_PCT or vol_pct > MICRO_TREND_RETEST_MAX_VOL_PCT:
            return setups
        if trend_strength(daily) < MICRO_TREND_RETEST_MIN_TREND_STRENGTH:
            return setups

        atr1 = h1.current_atr()
        atr5 = m5.current_atr()
        if atr1 <= 0 or atr5 <= 0:
            return setups
        lookback = max(3, int(MICRO_TREND_RETEST_LOOKBACK_BARS))
        bars = list(m5.bars)
        if len(bars) < lookback + 1:
            return setups
        current = bars[-1]
        prior = bars[-lookback - 1:-1]
        recent = prior + [current]
        buf = _buffer(atr5)

        for direction in (1, -1):
            if direction == 1 and not MICRO_TREND_RETEST_ALLOW_LONG:
                continue
            if direction == -1 and not MICRO_TREND_RETEST_ALLOW_SHORT:
                continue
            if not _micro_trend_retest_session_allowed(block, direction):
                continue
            if self._micro_variant_on_cooldown(
                    variant, direction, MICRO_TREND_RETEST_MIN_BARS_BETWEEN):
                continue
            if extension_blocked(direction, daily):
                continue

            score = alignment_score(direction, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            min_score = (
                MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG
                if direction == 1
                else MICRO_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT
            )
            if score < min_score:
                continue
            if MICRO_TREND_RETEST_REQUIRE_STRONG_TREND and not strong_trend:
                continue
            min_context = (
                MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_LONG
                if direction == 1
                else MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT
            )
            context_score = context_quality_score_for_features(
                SetupClass.T,
                direction,
                block,
                vol_pct,
                strong_trend,
            )
            if not meets_context_quality(min_context, context_score):
                continue

            if direction == 1:
                htf_ok = (
                    h1.ema_fast() > h1.ema_slow()
                    and h1.last_close > h1.ema_fast()
                    and (h1.macd_line_now() > 0 or h1.macd_line_now() > h1.macd_line_n_ago(1))
                )
                ltf_ok = (
                    m5.ema_fast() > m5.ema_slow()
                    and current.close >= m5.ema_fast()
                    and current.close >= current.open
                )
                if not (htf_ok and ltf_ok):
                    continue
                pullback_low = min(bar.low for bar in recent)
                trigger = max(bar.high for bar in prior)
                if pullback_low > m5.ema_fast() + MICRO_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr5:
                    continue
                if pullback_low < m5.ema_slow() - MICRO_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr5:
                    continue
                pullback = trigger - pullback_low
                if pullback <= 0 or pullback > MICRO_TREND_RETEST_MAX_PULLBACK_ATR * atr5:
                    continue
                breakout_gap = trigger - current.close
                if abs(breakout_gap) > MICRO_TREND_RETEST_MAX_BREAKOUT_GAP_ATR * atr5:
                    continue
                entry = trigger + buf
                required_stop = max(
                    MICRO_TREND_RETEST_STOP_M5_ATR_MULT * atr5,
                    MICRO_TREND_RETEST_STOP_H1_ATR_FRAC * atr1,
                )
                raw_stop = pullback_low - MICRO_TREND_RETEST_STOP_M5_ATR_MULT * atr5
                stop0 = min(raw_stop, entry - required_stop)
                stop_dist = entry - stop0
                kind = "PH"
                anchor_kind = "PL"
                anchor_price = pullback_low
            else:
                htf_ok = (
                    h1.ema_fast() < h1.ema_slow()
                    and h1.last_close < h1.ema_fast()
                    and (h1.macd_line_now() < 0 or h1.macd_line_now() < h1.macd_line_n_ago(1))
                )
                ltf_ok = (
                    m5.ema_fast() < m5.ema_slow()
                    and current.close <= m5.ema_fast()
                    and current.close <= current.open
                )
                if not (htf_ok and ltf_ok):
                    continue
                pullback_high = max(bar.high for bar in recent)
                trigger = min(bar.low for bar in prior)
                if pullback_high < m5.ema_fast() - MICRO_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr5:
                    continue
                if pullback_high > m5.ema_slow() + MICRO_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr5:
                    continue
                pullback = pullback_high - trigger
                if pullback <= 0 or pullback > MICRO_TREND_RETEST_MAX_PULLBACK_ATR * atr5:
                    continue
                breakout_gap = current.close - trigger
                if abs(breakout_gap) > MICRO_TREND_RETEST_MAX_BREAKOUT_GAP_ATR * atr5:
                    continue
                entry = trigger - buf
                required_stop = max(
                    MICRO_TREND_RETEST_STOP_M5_ATR_MULT * atr5,
                    MICRO_TREND_RETEST_STOP_H1_ATR_FRAC * atr1,
                )
                raw_stop = pullback_high + MICRO_TREND_RETEST_STOP_M5_ATR_MULT * atr5
                stop0 = max(raw_stop, entry + required_stop)
                stop_dist = stop0 - entry
                kind = "PL"
                anchor_kind = "PH"
                anchor_price = pullback_high

            if stop_dist <= 0 or stop_dist > MICRO_TREND_RETEST_MAX_STOP_DIST_ATR * atr1:
                continue

            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=SetupClass.T,
                    direction=direction,
                    tf_origin=TF.M5,
                    detected_ts=now_et,
                    P2=Pivot(
                        ts=current.ts,
                        kind=anchor_kind,
                        price=anchor_price,
                        macd=m5.macd_line_now(),
                        atr=atr5,
                    ),
                    breakout_pivot=Pivot(
                        ts=current.ts,
                        kind=kind,
                        price=trigger,
                        macd=m5.macd_line_now(),
                        atr=atr5,
                    ),
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    signal_variant=variant,
                    diagnostic_context=_setup_diagnostics(
                        direction=direction,
                        entry=entry,
                        current_price=current.close,
                        h1=h1,
                        h4=h4,
                        daily=daily,
                        context_score=context_score,
                        pullback_depth=pullback,
                        pullback_atr=atr5,
                        ltf=m5,
                        ltf_price=current.close,
                        ltf_pullback_depth=pullback,
                        ltf_atr=atr5,
                    ),
                )
            )
            self._mark_micro_variant_fired(variant, direction)
        return setups

    def detect_m15_trend_retest_setups(
        self,
        m15: BarSeries,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float,
    ) -> list[Setup]:
        setups: list[Setup] = []
        variant = "m15_trend_retest"
        if not M15_TREND_RETEST_ENABLED:
            return setups
        block = get_session_block(now_et)
        if not _m15_trend_retest_session_allowed(block):
            return setups
        if vol_pct < M15_TREND_RETEST_MIN_VOL_PCT or vol_pct > M15_TREND_RETEST_MAX_VOL_PCT:
            return setups
        if trend_strength(daily) < M15_TREND_RETEST_MIN_TREND_STRENGTH:
            return setups

        atr1 = h1.current_atr()
        atr15 = m15.current_atr()
        if atr1 <= 0 or atr15 <= 0:
            return setups
        lookback = max(3, int(M15_TREND_RETEST_LOOKBACK_BARS))
        bars = list(m15.bars)
        if len(bars) < lookback + 1:
            return setups
        current = bars[-1]
        prior = bars[-lookback - 1:-1]
        recent = prior + [current]
        buf = _buffer(atr15)

        for direction in (1, -1):
            if direction == 1 and not M15_TREND_RETEST_ALLOW_LONG:
                continue
            if direction == -1 and not M15_TREND_RETEST_ALLOW_SHORT:
                continue
            if self._m15_variant_on_cooldown(
                    variant, direction, M15_TREND_RETEST_MIN_BARS_BETWEEN):
                continue
            if extension_blocked(direction, daily):
                continue

            score = alignment_score(direction, daily, h4)
            strong_trend = strong_trend_flag(score, daily)
            min_score = (
                M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_LONG
                if direction == 1
                else M15_TREND_RETEST_MIN_ALIGNMENT_SCORE_SHORT
            )
            if score < min_score:
                continue
            if M15_TREND_RETEST_REQUIRE_STRONG_TREND and not strong_trend:
                continue
            min_context = (
                M15_TREND_RETEST_MIN_CONTEXT_SCORE_LONG
                if direction == 1
                else M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT
            )
            context_score = context_quality_score_for_features(
                SetupClass.T,
                direction,
                block,
                vol_pct,
                strong_trend,
            )
            if not meets_context_quality(min_context, context_score):
                continue

            if direction == 1:
                htf_ok = (
                    h1.ema_fast() > h1.ema_slow()
                    and h1.last_close > h1.ema_fast()
                    and (
                        h4.macd_hist_now() > 0
                        or h4.macd_hist_now() >= h4.macd_hist_n_ago(1)
                    )
                )
                ltf_ok = (
                    m15.ema_fast() > m15.ema_slow()
                    and current.close >= m15.ema_fast()
                    and current.close >= current.open
                )
                if not (htf_ok and ltf_ok):
                    continue
                pullback_low = min(bar.low for bar in recent)
                trigger = max(bar.high for bar in prior)
                if pullback_low > m15.ema_fast() + M15_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr15:
                    continue
                if pullback_low < m15.ema_slow() - M15_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr15:
                    continue
                pullback = trigger - pullback_low
                if pullback <= 0 or pullback > M15_TREND_RETEST_MAX_PULLBACK_ATR * atr15:
                    continue
                breakout_gap = trigger - current.close
                if abs(breakout_gap) > M15_TREND_RETEST_MAX_BREAKOUT_GAP_ATR * atr15:
                    continue
                entry = trigger + buf
                required_stop = max(
                    M15_TREND_RETEST_STOP_M15_ATR_MULT * atr15,
                    M15_TREND_RETEST_STOP_H1_ATR_FRAC * atr1,
                )
                raw_stop = pullback_low - M15_TREND_RETEST_STOP_M15_ATR_MULT * atr15
                stop0 = min(raw_stop, entry - required_stop)
                stop_dist = entry - stop0
                kind = "PH"
                anchor_kind = "PL"
                anchor_price = pullback_low
            else:
                htf_ok = (
                    h1.ema_fast() < h1.ema_slow()
                    and h1.last_close < h1.ema_fast()
                    and (
                        h4.macd_hist_now() < 0
                        or h4.macd_hist_now() <= h4.macd_hist_n_ago(1)
                    )
                )
                ltf_ok = (
                    m15.ema_fast() < m15.ema_slow()
                    and current.close <= m15.ema_fast()
                    and current.close <= current.open
                )
                if not (htf_ok and ltf_ok):
                    continue
                pullback_high = max(bar.high for bar in recent)
                trigger = min(bar.low for bar in prior)
                if pullback_high < m15.ema_fast() - M15_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr15:
                    continue
                if pullback_high > m15.ema_slow() + M15_TREND_RETEST_PULLBACK_EMA_TOL_ATR * atr15:
                    continue
                pullback = pullback_high - trigger
                if pullback <= 0 or pullback > M15_TREND_RETEST_MAX_PULLBACK_ATR * atr15:
                    continue
                breakout_gap = current.close - trigger
                if abs(breakout_gap) > M15_TREND_RETEST_MAX_BREAKOUT_GAP_ATR * atr15:
                    continue
                entry = trigger - buf
                required_stop = max(
                    M15_TREND_RETEST_STOP_M15_ATR_MULT * atr15,
                    M15_TREND_RETEST_STOP_H1_ATR_FRAC * atr1,
                )
                raw_stop = pullback_high + M15_TREND_RETEST_STOP_M15_ATR_MULT * atr15
                stop0 = max(raw_stop, entry + required_stop)
                stop_dist = stop0 - entry
                kind = "PL"
                anchor_kind = "PH"
                anchor_price = pullback_high

            if stop_dist <= 0 or stop_dist > M15_TREND_RETEST_MAX_STOP_DIST_ATR * atr1:
                continue

            setups.append(
                Setup(
                    setup_id=str(uuid.uuid4())[:8],
                    cls=SetupClass.T,
                    direction=direction,
                    tf_origin=TF.M15,
                    detected_ts=now_et,
                    P2=Pivot(
                        ts=current.ts,
                        kind=anchor_kind,
                        price=anchor_price,
                        macd=m15.macd_line_now(),
                        atr=atr15,
                    ),
                    breakout_pivot=Pivot(
                        ts=current.ts,
                        kind=kind,
                        price=trigger,
                        macd=m15.macd_line_now(),
                        atr=atr15,
                    ),
                    entry_stop=entry,
                    stop0=stop0,
                    buffer=buf,
                    alignment_score=score,
                    strong_trend=strong_trend,
                    is_extended=False,
                    signal_variant=variant,
                    diagnostic_context=_setup_diagnostics(
                        direction=direction,
                        entry=entry,
                        current_price=current.close,
                        h1=h1,
                        h4=h4,
                        daily=daily,
                        context_score=context_score,
                        pullback_depth=pullback,
                        pullback_atr=atr15,
                        ltf=m15,
                        ltf_price=current.close,
                        ltf_pullback_depth=pullback,
                        ltf_atr=atr15,
                    ),
                )
            )
            self._mark_m15_variant_fired(variant, direction)
        return setups

    def detect_class_T(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float | None = None,
    ) -> list[Setup]:
        setups: list[Setup] = []
        primary_long = self._detect_class_T_primary_long(pivots_1h, h1, h4, daily, now_et)
        setups.extend(primary_long)
        if not primary_long:
            setups.extend(self._detect_class_T_secondary_long(pivots_1h, h1, h4, daily, now_et, vol_pct))
        setups.extend(self._detect_class_T_selective_short(pivots_1h, h1, h4, daily, now_et, vol_pct))
        return setups

    def _detect_class_T_primary_long(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
    ) -> list[Setup]:
        setups: list[Setup] = []
        direction = 1
        if not (daily.ema_fast() > daily.ema_slow()):
            return setups
        if not (h4.ema_fast() > h4.ema_slow()):
            return setups

        h4_hist_now = h4.macd_hist_now()
        h4_hist_prev = h4.macd_hist_n_ago(CLASS_T_HIST_LOOKBACK)
        if not (h4_hist_now > h4_hist_prev and h4_hist_now > 0):
            return setups
        if trend_strength(daily) <= CLASS_T_TREND_STRENGTH_MIN:
            return setups
        if not (h1.ema_fast() > h1.ema_slow()):
            return setups

        macd_1h_now = h1.macd_line_now()
        macd_1h_prev = h1.macd_line_n_ago(CLASS_T_MACD_1H_LOOKBACK)
        if not (macd_1h_now > 0 and macd_1h_now > macd_1h_prev):
            return setups

        recent_ph = pivots_1h.most_recent_pivot_high()
        recent_pl = pivots_1h.most_recent_pivot_low()
        if not recent_ph or not recent_pl:
            return setups
        hours_since = (now_et - recent_ph.ts).total_seconds() / 3600
        if hours_since > CLASS_T_PIVOT_RECENCY_H:
            return setups

        atr_short = h1.atr_rolling(6)
        atr_long = h1.atr_rolling(20)
        if atr_long <= 0:
            return setups
        if atr_short / atr_long >= CLASS_T_COMPRESSION_MAX:
            return setups
        if extension_blocked(direction, daily):
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups
        buf = _buffer(atr1)
        entry = recent_ph.price + buf
        stop0 = recent_pl.price - CLASS_T_STOP_ATR_MULT * atr1
        stop_dist = entry - stop0
        if stop_dist <= 0 or stop_dist > CLASS_T_MAX_STOP_DIST_ATR * atr1:
            return setups

        score = alignment_score(direction, daily, h4)
        setups.append(self._build_t_setup(now_et, direction, recent_ph, entry, stop0, buf, score, daily))
        return setups

    def _detect_class_T_secondary_long(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float | None = None,
    ) -> list[Setup]:
        setups: list[Setup] = []
        direction = 1
        block = get_session_block(now_et)
        if not CLASS_T_SECONDARY_ENABLED:
            return setups
        if not _class_t_secondary_session_allowed(block):
            return setups
        if not (daily.ema_fast() > daily.ema_slow()):
            return setups
        if not (h4.ema_fast() > h4.ema_slow()):
            return setups

        h4_hist_now = h4.macd_hist_now()
        h4_hist_prev = h4.macd_hist_n_ago(CLASS_T_HIST_LOOKBACK)
        if h4_hist_now <= 0:
            return setups
        if h4_hist_prev > 0 and h4_hist_now < CLASS_T_SECONDARY_MIN_HIST_RATIO * h4_hist_prev:
            return setups
        if trend_strength(daily) <= CLASS_T_SECONDARY_TREND_STRENGTH_MIN:
            return setups
        if not (h1.ema_fast() > h1.ema_slow()):
            return setups

        macd_1h_now = h1.macd_line_now()
        macd_1h_prev = h1.macd_line_n_ago(CLASS_T_MACD_1H_LOOKBACK)
        if not (macd_1h_now > 0 and macd_1h_now >= macd_1h_prev):
            return setups

        recent_ph = pivots_1h.most_recent_pivot_high()
        recent_pl = pivots_1h.most_recent_pivot_low()
        if not recent_ph or not recent_pl:
            return setups
        hours_since = (now_et - recent_ph.ts).total_seconds() / 3600
        if hours_since > CLASS_T_SECONDARY_PIVOT_RECENCY_H:
            return setups

        atr_short = h1.atr_rolling(6)
        atr_long = h1.atr_rolling(20)
        if atr_long <= 0:
            return setups
        if atr_short / atr_long >= CLASS_T_SECONDARY_COMPRESSION_MAX:
            return setups
        if extension_blocked(direction, daily):
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups
        if h1.last_close <= h1.ema_fast():
            return setups
        if recent_pl.price < h1.ema_slow() - 0.25 * atr1:
            return setups

        breakout_gap = recent_ph.price - h1.last_close
        if breakout_gap < 0 or breakout_gap > CLASS_T_SECONDARY_MAX_BREAKOUT_GAP_ATR * atr1:
            return setups

        buf = _buffer(atr1)
        entry = recent_ph.price + buf
        stop0 = recent_pl.price - CLASS_T_SECONDARY_STOP_ATR_MULT * atr1
        stop_dist = entry - stop0
        if stop_dist <= 0 or stop_dist > CLASS_T_SECONDARY_MAX_STOP_DIST_ATR * atr1:
            return setups

        score = alignment_score(direction, daily, h4)
        if score < CLASS_T_SECONDARY_MIN_ALIGNMENT_SCORE:
            return setups
        context_score = context_quality_score_for_features(
            SetupClass.T,
            direction,
            block,
            vol_pct,
            strong_trend_flag(score, daily),
        )
        if not meets_context_quality(CLASS_T_SECONDARY_MIN_CONTEXT_SCORE, context_score):
            return setups

        setups.append(self._build_t_setup(now_et, direction, recent_ph, entry, stop0, buf, score, daily))
        return setups

    def _detect_class_T_selective_short(
        self,
        pivots_1h: PivotDetector,
        h1: BarSeries,
        h4: BarSeries,
        daily: BarSeries,
        now_et: datetime,
        vol_pct: float | None = None,
    ) -> list[Setup]:
        setups: list[Setup] = []
        direction = -1
        block = get_session_block(now_et)
        if not CLASS_T_SHORT_ENABLED:
            return setups
        if not _class_t_short_session_allowed(block):
            return setups
        if not (h4.ema_fast() < h4.ema_slow()):
            return setups
        if not (h1.ema_fast() < h1.ema_slow()):
            return setups

        score = alignment_score(direction, daily, h4)
        if score < CLASS_T_SHORT_MIN_ALIGNMENT_SCORE:
            return setups
        context_score = context_quality_score_for_features(
            SetupClass.T,
            direction,
            block,
            vol_pct,
            strong_trend_flag(score, daily),
        )
        if not meets_context_quality(CLASS_T_SHORT_MIN_CONTEXT_SCORE, context_score):
            return setups
        if score == 1 and trend_strength(daily) < CLASS_T_SHORT_TREND_STRENGTH_MIN:
            return setups
        if score < 2 and daily.last_close > daily.ema_fast():
            return setups

        h4_hist_now = h4.macd_hist_now()
        h4_hist_prev = h4.macd_hist_n_ago(CLASS_T_HIST_LOOKBACK)
        if h4_hist_now >= 0:
            return setups
        if h4_hist_prev < 0 and abs(h4_hist_now) < CLASS_T_SHORT_MIN_HIST_RATIO * abs(h4_hist_prev):
            return setups

        macd_1h_now = h1.macd_line_now()
        macd_1h_prev = h1.macd_line_n_ago(CLASS_T_MACD_1H_LOOKBACK)
        if not (macd_1h_now < 0 and macd_1h_now <= macd_1h_prev):
            return setups

        recent_pl = pivots_1h.most_recent_pivot_low()
        recent_ph = pivots_1h.most_recent_pivot_high()
        if not recent_pl or not recent_ph:
            return setups
        hours_since = (now_et - recent_pl.ts).total_seconds() / 3600
        if hours_since > CLASS_T_SHORT_PIVOT_RECENCY_H:
            return setups

        atr_short = h1.atr_rolling(6)
        atr_long = h1.atr_rolling(20)
        if atr_long <= 0:
            return setups
        if atr_short / atr_long >= CLASS_T_SHORT_COMPRESSION_MAX:
            return setups
        if extension_blocked(direction, daily):
            return setups

        atr1 = h1.current_atr()
        if atr1 <= 0:
            return setups
        breakout_gap = h1.last_close - recent_pl.price
        if breakout_gap < 0 or breakout_gap > CLASS_T_SHORT_MAX_BREAKOUT_GAP_ATR * atr1:
            return setups

        buf = _buffer(atr1)
        entry = recent_pl.price - buf
        stop0 = recent_ph.price + CLASS_T_SHORT_STOP_ATR_MULT * atr1
        stop_dist = stop0 - entry
        if stop_dist <= 0 or stop_dist > CLASS_T_SHORT_MAX_STOP_DIST_ATR * atr1:
            return setups

        setups.append(self._build_t_setup(now_et, direction, recent_pl, entry, stop0, buf, score, daily))
        return setups

    def _build_t_setup(
        self,
        now_et: datetime,
        direction: int,
        breakout_pivot,
        entry: float,
        stop0: float,
        buf: float,
        score: int,
        daily: BarSeries,
    ) -> Setup:
        return Setup(
            setup_id=str(uuid.uuid4())[:8],
            cls=SetupClass.T,
            direction=direction,
            tf_origin=TF.H4,
            detected_ts=now_et,
            P1=None,
            P2=None,
            breakout_pivot=breakout_pivot,
            entry_stop=entry,
            stop0=stop0,
            buffer=buf,
            alignment_score=score,
            strong_trend=strong_trend_flag(score, daily),
            is_extended=False,
        )
