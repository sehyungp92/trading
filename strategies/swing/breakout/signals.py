"""Multi-Asset Swing Breakout v3.3-ETF — signal detection (stateless pure functions).

Compression, breakout qualification, entry signals A/B/C, adds, continuation, DIRTY.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np

from .config import (
    ADD_CLV_STRONG_Q,
    ATR_RATIO_HIGH,
    ATR_RATIO_LOW,
    CHOP_ATR_PCTL_HIGH,
    CHOP_CROSS_LOOKBACK,
    CHOP_CROSS_THRESHOLD,
    CONTAINMENT_MIN,
    CONT_BARS_MIN,
    DIRTY_BOX_SHIFT_ATR_MULT,
    DIRTY_DURATION_L_FRAC,
    DIRTY_OPPOSITE_MULT,
    DISP_STRONG_MULT,
    ENTRYB_DISPMULT_OVERRIDE,
    ENTRYB_RVOLH_MIN,
    ENTRY_A_ACTIVE_BLOCKS_C,
    ENTRY_A_STRONG_CLV_Q,
    ENTRY_A_STRONG_CONFIRM_PRIOR_STRUCTURE,
    ENTRY_A_STRONG_ENABLE,
    ENTRY_A_STRONG_LIMIT_OFFSET_ATR_H,
    ENTRY_A_STRONG_MIN_QUALITY,
    ENTRY_A_STRONG_MIN_SCORE,
    ENTRY_A_STRONG_REQUIRE_ALIGNED,
    ENTRY_A_STRONG_STOP_BUFFER_ATR_H,
    ENTRY_A_STRONG_USE_STOP_LIMIT,
    ENTRY_B_RESUME_CLV_Q,
    ENTRY_B_RESUME_ENABLE,
    ENTRY_B_RESUME_LIMIT_OFFSET_ATR_H,
    ENTRY_B_RESUME_MAX_BREAKOUT_BARS,
    ENTRY_B_RESUME_MIN_QUALITY,
    ENTRY_B_RESUME_MIN_SCORE,
    ENTRY_B_RESUME_REQUIRE_ALIGNED,
    ENTRY_B_RESUME_STOP_BUFFER_ATR_H,
    ENTRY_B_RESUME_SWEEP_ATR_D,
    ENTRY_B_RESUME_USE_STOP_LIMIT,
    ENTRY_C_EARLY_ALLOW_NEUTRAL,
    ENTRY_C_EARLY_CLV_Q,
    ENTRY_C_EARLY_ENABLE,
    ENTRY_C_EARLY_MAX_BREAKOUT_BARS,
    ENTRY_C_EARLY_MAX_DISP_H,
    ENTRY_C_EARLY_MIN_QUALITY,
    ENTRY_C_EARLY_MIN_RVOL_H,
    ENTRY_C_EARLY_MIN_SCORE,
    ENTRY_C_EARLY_REQUIRE_ALIGNED,
    ENTRY_C_FAST_MARKET,
    ENTRY_C_FAST_MIN_QUALITY,
    ENTRY_C_FAST_MIN_SCORE,
    ENTRY_C_FAST_REQUIRE_ALIGNED,
    ENTRY_C_CONTINUATION_ALLOW_NEUTRAL,
    ENTRY_C_CONTINUATION_CLV_Q,
    ENTRY_C_CONTINUATION_ENABLE,
    ENTRY_C_CONTINUATION_HOLD_BARS,
    ENTRY_C_CONTINUATION_MAX_BREAKOUT_BARS,
    ENTRY_C_CONTINUATION_MAX_DISP_H,
    ENTRY_C_CONTINUATION_MIN_QUALITY,
    ENTRY_C_CONTINUATION_MIN_RVOL_H,
    ENTRY_C_CONTINUATION_MIN_SCORE,
    ENTRY_C_CONTINUATION_PAUSE_ATR_H,
    ENTRY_C_CONTINUATION_REQUIRE_ALIGNED,
    ENTRY_C_STANDARD_ALLOW_CONTINUATION,
    ENTRY_C_STANDARD_MAX_BREAKOUT_BARS,
    ENTRY_C_STANDARD_MAX_DISP_H,
    ENTRY_C_FRESH_ALLOW_COUNTERTREND,
    ENTRY_C_FRESH_CLV_Q,
    ENTRY_C_FRESH_ENABLE,
    ENTRY_C_FRESH_LIMIT_OFFSET_ATR_H,
    ENTRY_C_FRESH_MAX_BREAKOUT_BARS,
    ENTRY_C_FRESH_MAX_DISP_H,
    ENTRY_C_FRESH_MIN_QUALITY,
    ENTRY_C_FRESH_MIN_SCORE,
    ENTRY_C_FRESH_REQUIRE_ALIGNED,
    ENTRY_C_FRESH_STOP_ALLOW_COUNTERTREND,
    ENTRY_C_FRESH_STOP_CLV_Q,
    ENTRY_C_FRESH_STOP_ENABLE,
    ENTRY_C_FRESH_STOP_MAX_BREAKOUT_BARS,
    ENTRY_C_FRESH_STOP_MAX_DISP_H,
    ENTRY_C_FRESH_STOP_MIN_QUALITY,
    ENTRY_C_FRESH_STOP_MIN_RVOL_H,
    ENTRY_C_FRESH_STOP_MIN_SCORE,
    ENTRY_C_FRESH_STOP_REQUIRE_ALIGNED,
    ENTRY_C_FRESH_STOP_TOUCH_TOL_ATR_H,
    ENTRY_C_FRESH_STOP_BUFFER_ATR_H,
    ENTRY_C_FRESH_TOUCH_TOL_ATR_H,
    ENTRY_C_FRESH_USE_STOP_LIMIT,
    ENTRY_C_HOLD_BARS,
    ENTRY_C_MOMENTUM_CLV_Q,
    ENTRY_C_MOMENTUM_ENABLE,
    ENTRY_C_MOMENTUM_MAX_DISP_H,
    ENTRY_C_MOMENTUM_MIN_QUALITY,
    ENTRY_C_MOMENTUM_MIN_SCORE,
    ENTRY_C_MOMENTUM_REQUIRE_ALIGNED,
    ENTRY_C_MOMENTUM_USE_STOP_LIMIT,
    ENTRY_OUTSIDE_WINDOW_CARRY_ENABLE,
    ENTRY_OUTSIDE_WINDOW_CARRY_A_OR_FRESH_ONLY,
    ENTRY_OUTSIDE_WINDOW_CARRY_FRESH_ONLY,
    ENTRY_OUTSIDE_WINDOW_CARRY_MIN_QUALITY,
    ENTRY_OUTSIDE_WINDOW_CARRY_MIN_SCORE,
    ENTRY_OUTSIDE_WINDOW_CARRY_REQUIRE_ALIGNED,
    EXPIRY_BARS_MAX,
    EXPIRY_BARS_MIN,
    HARD_EXPIRY_BARS_ADD,
    HYSTERESIS_BARS,
    L_HIGH,
    L_LOW,
    L_MID,
    LOOKBACK_SQ,
    MM_BOX_HEIGHT_MULT,
    Q_DISP,
    Q_DISP_ATR_EXPAND_ADJ,
    DISP_LOOKBACK_MAX,
    R_PROXY_CONT_THRESHOLD,
    R_PROXY_TIME_THRESHOLD,
    RECLAIM_BUFFER_ATR_D_MULT,
    RECLAIM_BUFFER_ATR_H_MULT,
    SQUEEZE_CEIL,
    SQUEEZE_CEIL_ADAPTIVE,
    SQUEEZE_CEIL_FALLBACK,
    SQUEEZE_CEIL_PCTL,
    SWEEP_DEPTH_ATR_D_MULT,
    ENTRY_A_TOUCH_TOL_ATR_H,
    ENTRYB_REQUIRE_ALIGNED,
    ENTRYB_NEUTRAL_QUALITY_MIN,
    SymbolConfig,
)
from .indicators import (
    containment_ratio,
    highest,
    lowest,
    past_only_quantile,
    volume_score_component_daily,
)
from .models import (
    CampaignState,
    ChopMode,
    DailyContext,
    Direction,
    EntryType,
    Regime4H,
    SymbolCampaign,
    TradeRegime,
)


# ---------------------------------------------------------------------------
# Adaptive L with hysteresis (spec §5.1)
# ---------------------------------------------------------------------------

def _atr_ratio_bucket(atr_ratio: float) -> str:
    if atr_ratio < ATR_RATIO_LOW:
        return "low"
    if atr_ratio > ATR_RATIO_HIGH:
        return "high"
    return "mid"


def _bucket_to_L(bucket: str) -> int:
    return {"low": L_LOW, "mid": L_MID, "high": L_HIGH}[bucket]


def choose_L_with_hysteresis(
    atr_ratio: float,
    campaign: SymbolCampaign,
) -> int:
    """Adaptive box length with hysteresis — switch only if bucket holds >= 3 consecutive bars."""
    new_bucket = _atr_ratio_bucket(atr_ratio)

    # First call: initialize committed bucket immediately
    if campaign.L_bucket_current is None:
        campaign.L_bucket_current = new_bucket
        campaign.L_bucket_bars_held = 1
        campaign.L_bucket_candidate = None
        campaign.L_bucket_candidate_bars = 0
        return _bucket_to_L(new_bucket)

    # Already in committed bucket — stay there
    if new_bucket == campaign.L_bucket_current:
        campaign.L_bucket_bars_held += 1
        # Reset any pending candidate
        campaign.L_bucket_candidate = None
        campaign.L_bucket_candidate_bars = 0
        return _bucket_to_L(campaign.L_bucket_current)

    # Different bucket — track as candidate
    if new_bucket == campaign.L_bucket_candidate:
        # Same candidate as last bar: increment consecutive count
        campaign.L_bucket_candidate_bars += 1
    else:
        # New candidate: reset consecutive counter
        campaign.L_bucket_candidate = new_bucket
        campaign.L_bucket_candidate_bars = 1

    # Candidate reached hysteresis threshold — commit the switch
    if campaign.L_bucket_candidate_bars >= HYSTERESIS_BARS:
        campaign.L_bucket_current = campaign.L_bucket_candidate
        campaign.L_bucket_bars_held = campaign.L_bucket_candidate_bars
        campaign.L_bucket_candidate = None
        campaign.L_bucket_candidate_bars = 0
        return _bucket_to_L(campaign.L_bucket_current)

    # Candidate hasn't proven itself — keep current
    return campaign.L


# ---------------------------------------------------------------------------
# Compression detection (spec §5.2–5.3)
# ---------------------------------------------------------------------------

def detect_compression(
    containment: float,
    squeeze_metric: float,
    squeeze_hist: list[float] | None = None,
) -> bool:
    """Box activation: containment >= 0.80 AND squeeze_metric <= ceil.

    When SQUEEZE_CEIL_ADAPTIVE is True and sufficient history exists,
    the ceiling is derived from the rolling squeeze distribution rather
    than the static SQUEEZE_CEIL constant.
    """
    if containment < CONTAINMENT_MIN:
        return False
    if SQUEEZE_CEIL_ADAPTIVE and squeeze_hist and len(squeeze_hist) >= 20:
        ceil = past_only_quantile(squeeze_hist, SQUEEZE_CEIL_PCTL, LOOKBACK_SQ)
        ceil = min(ceil, SQUEEZE_CEIL_FALLBACK)
    else:
        ceil = SQUEEZE_CEIL
    return squeeze_metric <= ceil


# ---------------------------------------------------------------------------
# Squeeze tier (spec §5.4)
# ---------------------------------------------------------------------------

def classify_squeeze_tier(
    squeeze_metric: float,
    squeeze_hist: list[float],
) -> tuple[bool, bool]:
    """Returns (sq_good, sq_loose) based on past-only quantiles."""
    if len(squeeze_hist) < 10:
        return False, False
    sq_th_good = past_only_quantile(squeeze_hist, 0.30, LOOKBACK_SQ)
    sq_th_loose = past_only_quantile(squeeze_hist, 0.65, LOOKBACK_SQ)
    return squeeze_metric <= sq_th_good, squeeze_metric >= sq_th_loose


# ---------------------------------------------------------------------------
# Structural breakout (spec §8.1)
# ---------------------------------------------------------------------------

def check_structural_breakout(
    close_d: float,
    box_high: float,
    box_low: float,
) -> Optional[Direction]:
    """Returns breakout direction or None."""
    if close_d > box_high:
        return Direction.LONG
    if close_d < box_low:
        return Direction.SHORT
    return None


# ---------------------------------------------------------------------------
# Displacement pass (spec §8.2)
# ---------------------------------------------------------------------------

def check_displacement(
    close_d: float,
    avwap_d: float,
    atr14_d: float,
    disp_hist: list[float],
    atr_expanding: bool,
) -> tuple[bool, float, float]:
    """Returns (pass, disp, disp_th)."""
    if atr14_d <= 0:
        return False, 0.0, 0.0
    disp = abs(close_d - avwap_d) / atr14_d
    q_eff = Q_DISP - Q_DISP_ATR_EXPAND_ADJ if atr_expanding else Q_DISP
    lookback = min(len(disp_hist), DISP_LOOKBACK_MAX)
    disp_th = past_only_quantile(disp_hist, q_eff, lookback) if disp_hist else 0.0
    return disp >= disp_th, disp, disp_th


# ---------------------------------------------------------------------------
# Breakout quality reject (spec §8.3)
# ---------------------------------------------------------------------------

def check_breakout_quality_reject(
    high_d: float,
    low_d: float,
    open_d: float,
    close_d: float,
    atr14_d: float,
    direction: Optional[Direction] = None,
) -> bool:
    """Reject if anomalous bar: wide range + weak body or big adverse wick."""
    bar_range = high_d - low_d
    if atr14_d <= 0 or bar_range <= 0:
        return False
    if bar_range <= 2.0 * atr14_d:
        return False

    body = abs(close_d - open_d)
    body_ratio = body / bar_range

    # Adverse wick based on breakout direction (not candle color)
    if direction == Direction.LONG:
        adverse_wick = high_d - close_d  # rejection from above
    elif direction == Direction.SHORT:
        adverse_wick = close_d - low_d  # bounce from below
    elif close_d >= open_d:
        adverse_wick = high_d - close_d
    else:
        adverse_wick = close_d - low_d
    adverse_wick_ratio = adverse_wick / bar_range

    return body_ratio < 0.25 or adverse_wick_ratio > 0.55


# ---------------------------------------------------------------------------
# Evidence score (spec §9)
# ---------------------------------------------------------------------------

def compute_evidence_score(
    rvol_d: float,
    disp: float,
    disp_th: float,
    displacement_pass: bool,
    sq_good: bool,
    sq_loose: bool,
    regime_4h: Regime4H,
    direction: Direction,
    atr_expanding: bool,
    closes_d: np.ndarray,
    box_high: float,
    box_low: float,
) -> tuple[int, int]:
    """Compute evidence score and volume score.

    Returns (total_score, vol_score).
    """
    vol_score = volume_score_component_daily(rvol_d)
    # Low-volume day exception (spec §9.1)
    if displacement_pass and disp >= DISP_STRONG_MULT * disp_th and rvol_d < 0.8:
        vol_score = max(vol_score, -1)

    score = vol_score

    # Squeeze component (penalty-only: no bonus for sq_good)
    if sq_loose:
        score -= 1

    # Regime component (penalty-only: no bonus for aligned)
    opposes = (
        (direction == Direction.LONG and regime_4h == Regime4H.BEAR_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BULL_TREND)
    )
    if opposes:
        score -= 1

    # Two consecutive closes outside box
    if len(closes_d) >= 2:
        if direction == Direction.LONG:
            if closes_d[-1] > box_high and closes_d[-2] > box_high:
                score += 1
        elif direction == Direction.SHORT:
            if closes_d[-1] < box_low and closes_d[-2] < box_low:
                score += 1

    # ATR expanding (optional)
    if atr_expanding:
        score += 1

    return score, vol_score


def compute_evidence_score_detailed(
    rvol_d: float,
    disp: float,
    disp_th: float,
    displacement_pass: bool,
    sq_good: bool,
    sq_loose: bool,
    regime_4h: Regime4H,
    direction: Direction,
    atr_expanding: bool,
    closes_d: np.ndarray,
    box_high: float,
    box_low: float,
) -> dict:
    """Compute evidence score with individual component breakdown.

    Returns dict with keys: total, vol, squeeze, regime, consec, atr.
    """
    vol_score = volume_score_component_daily(rvol_d)
    if displacement_pass and disp >= DISP_STRONG_MULT * disp_th and rvol_d < 0.8:
        vol_score = max(vol_score, -1)

    squeeze_score = 0
    if sq_loose:
        squeeze_score = -1

    regime_score = 0
    opposes = (
        (direction == Direction.LONG and regime_4h == Regime4H.BEAR_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BULL_TREND)
    )
    if opposes:
        regime_score = -1

    consec_score = 0
    if len(closes_d) >= 2:
        if direction == Direction.LONG:
            if closes_d[-1] > box_high and closes_d[-2] > box_high:
                consec_score = 1
        elif direction == Direction.SHORT:
            if closes_d[-1] < box_low and closes_d[-2] < box_low:
                consec_score = 1

    atr_score = 1 if atr_expanding else 0

    total = vol_score + squeeze_score + regime_score + consec_score + atr_score

    return {
        "total": total,
        "vol": vol_score,
        "squeeze": squeeze_score,
        "regime": regime_score,
        "consec": consec_score,
        "atr": atr_score,
    }


# ---------------------------------------------------------------------------
# Expiry (spec §11.2)
# ---------------------------------------------------------------------------

def compute_expiry_bars(atr14_d: float, atr_hist: list[float]) -> tuple[int, int]:
    """Compute expiry_bars and hard_expiry_bars from ATR percentile (60-bar lookback)."""
    if not atr_hist:
        return 5, 10
    # Spec §11.2: ATR_pctl_60 uses a 60-bar lookback window
    window = atr_hist[-60:] if len(atr_hist) >= 60 else atr_hist
    arr = np.array(window)
    pctl = float(np.sum(arr <= atr14_d) / len(arr) * 100.0) if len(arr) > 0 else 50.0
    raw = round(5.0 * (pctl / 50.0))
    expiry = max(EXPIRY_BARS_MIN, min(EXPIRY_BARS_MAX, raw))
    return expiry, expiry + HARD_EXPIRY_BARS_ADD


def compute_expiry_mult(bars_since_breakout: int, expiry_bars: int) -> float:
    """Expiry multiplier with step decay after expiry_bars."""
    if bars_since_breakout <= expiry_bars:
        return 1.0
    steps = bars_since_breakout - expiry_bars
    mult = 1.0 - 0.12 * steps
    return max(0.30, mult)


# ---------------------------------------------------------------------------
# Continuation mode (spec §11.3)
# ---------------------------------------------------------------------------

def check_continuation_mode(
    close_d: float,
    box_high: float,
    box_low: float,
    box_height: float,
    atr14_d: float,
    direction: Direction,
    bars_since_breakout: int,
    regime_4h: Regime4H,
    disp_mult: float,
) -> bool:
    """Check if continuation mode should be enabled."""
    # Measured Move
    if direction == Direction.LONG:
        mm = box_high + MM_BOX_HEIGHT_MULT * box_height
        r_proxy = (close_d - box_high) / atr14_d if atr14_d > 0 else 0.0
    else:
        mm = box_low - MM_BOX_HEIGHT_MULT * box_height
        r_proxy = (box_low - close_d) / atr14_d if atr14_d > 0 else 0.0

    if direction == Direction.LONG and close_d >= mm:
        return True
    if direction == Direction.SHORT and close_d <= mm:
        return True
    if r_proxy >= R_PROXY_CONT_THRESHOLD:
        return True

    # Time + partial conditions
    aligned = (
        (direction == Direction.LONG and regime_4h == Regime4H.BULL_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BEAR_TREND)
    )
    if (r_proxy >= R_PROXY_TIME_THRESHOLD
            and bars_since_breakout >= CONT_BARS_MIN
            and (aligned or disp_mult >= 0.85)):
        return True

    return False


# ---------------------------------------------------------------------------
# DIRTY handling (spec §10)
# ---------------------------------------------------------------------------

def check_dirty_trigger(
    close_d: float,
    box_high: float,
    box_low: float,
    breakout_direction: Direction,
    days_since_breakout: int,
    m_break: int,
) -> bool:
    """Check if breakout failure triggers DIRTY state."""
    if days_since_breakout < m_break:
        return False
    if breakout_direction == Direction.LONG:
        return close_d <= box_high  # closed back inside
    if breakout_direction == Direction.SHORT:
        return close_d >= box_low
    return False


def check_dirty_reset(
    new_high: float,
    new_low: float,
    dirty_high: float,
    dirty_low: float,
    atr14_d: float,
    squeeze_metric: float,
    squeeze_hist: list[float],
    dirty_duration: int,
    L_used: int,
) -> tuple[bool, bool]:
    """Check DIRTY reset conditions.

    Returns (should_reset, box_shifted).
    """
    sq_good, _ = classify_squeeze_tier(squeeze_metric, squeeze_hist)
    if not sq_good:
        return False, False

    high_shift = abs(new_high - dirty_high) >= DIRTY_BOX_SHIFT_ATR_MULT * atr14_d
    low_shift = abs(new_low - dirty_low) >= DIRTY_BOX_SHIFT_ATR_MULT * atr14_d
    box_shifted = high_shift and low_shift
    duration_ok = dirty_duration >= DIRTY_DURATION_L_FRAC * L_used

    if box_shifted or duration_ok:
        return True, box_shifted
    return False, False


def dirty_opposite_allowed(
    disp: float,
    disp_th: float,
) -> bool:
    """Opposite direction allowed while DIRTY if displacement is strong."""
    return disp >= DIRTY_OPPOSITE_MULT * disp_th


# ---------------------------------------------------------------------------
# Entry A: AVWAP_H retest + reclaim (spec §12.3)
# ---------------------------------------------------------------------------

def entry_a_signal(
    direction: Direction,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    prev_hourly_low: float = 0.0,
    prev_hourly_high: float = 0.0,
    prev2_hourly_low: float = 0.0,
    prev2_hourly_high: float = 0.0,
) -> bool:
    """Entry A: touch AVWAP_H and reclaim with buffer.

    3-bar pattern — any of current, prev, or prev2 bar's extreme touched
    AVWAP_H, AND current bar's close reclaims above it.
    """
    reclaim_buffer = max(RECLAIM_BUFFER_ATR_H_MULT * atr14_h,
                         RECLAIM_BUFFER_ATR_D_MULT * atr14_d)
    touch_tolerance = ENTRY_A_TOUCH_TOL_ATR_H * atr14_h

    if direction == Direction.LONG:
        touched_now = hourly_low <= avwap_h + touch_tolerance
        touched_prev = prev_hourly_low > 0 and prev_hourly_low <= avwap_h + touch_tolerance
        touched_prev2 = prev2_hourly_low > 0 and prev2_hourly_low <= avwap_h + touch_tolerance
        reclaimed = hourly_close > avwap_h + reclaim_buffer
        return (touched_now or touched_prev or touched_prev2) and reclaimed
    else:
        touched_now = hourly_high >= avwap_h - touch_tolerance
        touched_prev = prev_hourly_high > 0 and prev_hourly_high >= avwap_h - touch_tolerance
        touched_prev2 = prev2_hourly_high > 0 and prev2_hourly_high >= avwap_h - touch_tolerance
        reclaimed = hourly_close < avwap_h - reclaim_buffer
        return (touched_now or touched_prev or touched_prev2) and reclaimed


# ---------------------------------------------------------------------------
# Entry B: sweep + reclaim (spec §12.4)
# ---------------------------------------------------------------------------

def entry_a_reclaim_strong_signal(
    direction: Direction,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
    prev_hourly_low: float = 0.0,
    prev_hourly_high: float = 0.0,
    prev2_hourly_low: float = 0.0,
    prev2_hourly_high: float = 0.0,
) -> bool:
    """High-conviction reclaim branch for stronger AVWAP retests."""
    if not ENTRY_A_STRONG_ENABLE:
        return False
    if continuation:
        return False
    if ENTRY_A_STRONG_REQUIRE_ALIGNED and not is_regime_aligned(direction, regime_4h):
        return False
    if score_total < ENTRY_A_STRONG_MIN_SCORE:
        return False
    if quality_mult < ENTRY_A_STRONG_MIN_QUALITY:
        return False
    if not entry_a_signal(
        direction,
        hourly_close,
        hourly_low,
        hourly_high,
        avwap_h,
        atr14_h,
        atr14_d,
        prev_hourly_low,
        prev_hourly_high,
        prev2_hourly_low,
        prev2_hourly_high,
    ):
        return False
    if not strong_close_location(
        direction,
        hourly_high,
        hourly_low,
        hourly_close,
        q=ENTRY_A_STRONG_CLV_Q,
    ):
        return False
    if ENTRY_A_STRONG_CONFIRM_PRIOR_STRUCTURE:
        if direction == Direction.LONG and prev_hourly_high > 0 and hourly_close <= prev_hourly_high:
            return False
        if direction == Direction.SHORT and prev_hourly_low > 0 and hourly_close >= prev_hourly_low:
            return False
    return True


def entry_b_resume_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
) -> bool:
    """Failed-probe resume entry for fresh breakout pullbacks."""
    if not ENTRY_B_RESUME_ENABLE:
        return False
    if continuation:
        return False
    if ENTRY_B_RESUME_REQUIRE_ALIGNED and not is_regime_aligned(direction, regime_4h):
        return False
    if score_total < ENTRY_B_RESUME_MIN_SCORE:
        return False
    if quality_mult < ENTRY_B_RESUME_MIN_QUALITY:
        return False
    if campaign.bars_since_breakout > ENTRY_B_RESUME_MAX_BREAKOUT_BARS:
        return False
    if len(hourly_closes) < 2 or len(hourly_highs) < 2 or len(hourly_lows) < 2:
        return False

    sweep_depth = ENTRY_B_RESUME_SWEEP_ATR_D * atr14_d if atr14_d > 0 else 0.0
    prev_high = float(hourly_highs[-2])
    prev_low = float(hourly_lows[-2])

    if direction == Direction.LONG:
        swept = hourly_low <= avwap_h - sweep_depth or prev_low <= avwap_h - sweep_depth
        resumed = hourly_close > max(avwap_h, prev_high)
    else:
        swept = hourly_high >= avwap_h + sweep_depth or prev_high >= avwap_h + sweep_depth
        resumed = hourly_close < min(avwap_h, prev_low)

    if not swept or not resumed:
        return False

    return strong_close_location(
        direction,
        hourly_high,
        hourly_low,
        hourly_close,
        q=ENTRY_B_RESUME_CLV_Q,
    )


def entry_b_signal(
    direction: Direction,
    hourly_low: float,
    hourly_high: float,
    hourly_close: float,
    avwap_h: float,
    atr14_d: float,
    prev_hourly_low: float = 0.0,
    prev_hourly_high: float = 0.0,
) -> bool:
    """Entry B: sweep past AVWAP then reclaim (1- or 2-bar pattern)."""
    sweep_depth = SWEEP_DEPTH_ATR_D_MULT * atr14_d

    if direction == Direction.LONG:
        swept_now = hourly_low < avwap_h - sweep_depth
        swept_prev = prev_hourly_low > 0 and prev_hourly_low < avwap_h - sweep_depth
        reclaimed = hourly_close > avwap_h
        return (swept_now or swept_prev) and reclaimed
    else:
        swept_now = hourly_high > avwap_h + sweep_depth
        swept_prev = prev_hourly_high > 0 and prev_hourly_high > avwap_h + sweep_depth
        reclaimed = hourly_close < avwap_h
        return (swept_now or swept_prev) and reclaimed


def entry_b_permitted(
    rvol_h: float,
    disp_mult: float,
    quality_mult: float,
    regime_4h: Regime4H,
    direction: Direction,
    campaign: SymbolCampaign,
) -> bool:
    """Check Entry B permission gates (spec §12.4).

    A4 fix: Removed quality_mult >= 0.55 hard gate. In CAUTION regime,
    regime_mult=0.40 makes quality_mult < 0.55 mathematically impossible,
    so the gate was self-defeating. Quality now affects sizing via risk
    allocation, not Entry B eligibility.
    """
    aligned = (
        (direction == Direction.LONG and regime_4h == Regime4H.BULL_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BEAR_TREND)
    )
    if not aligned:
        if ENTRYB_REQUIRE_ALIGNED:
            return False
        # Cat 2: allow NEUTRAL regime with sufficient quality (baseline: blocked)
        neutral = regime_4h == Regime4H.RANGE_CHOP
        if not (neutral and quality_mult >= ENTRYB_NEUTRAL_QUALITY_MIN):
            return False
    # A4: quality_mult >= 0.55 gate removed — quality affects sizing, not eligibility
    if campaign.continuation:
        return False
    # Spec §12.4: Entry B blocked only for same-direction while DIRTY
    if campaign.state == CampaignState.DIRTY and direction == campaign.breakout_direction:
        return False

    # RVOL_H gating
    if rvol_h >= ENTRYB_RVOLH_MIN:
        return True
    return disp_mult >= ENTRYB_DISPMULT_OVERRIDE


# ---------------------------------------------------------------------------
# Entry C standard: 2-hour hold above/below AVWAP_H (spec §12.5)
# ---------------------------------------------------------------------------

def entry_c_standard_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_closes: list[float],
    avwap_h: float,
    atr_expanding: bool = True,
    rvol_h: float = 1.0,
    atr14_h: float = 0.0,
    atr14_d: float = 0.0,
    ema20_h: float = 0.0,
    quality_mult: float = 0.0,
    regime_4h: Regime4H | None = None,
    score_total: int = 0,
    continuation: bool = False,
) -> bool:
    """Entry C hold logic with optional high-conviction fast mode.

    Default behavior is the legacy 2-bar hold. When mutated to 1-bar hold,
    require stronger score/quality/alignment so frequency expands through
    higher-conviction setups instead of a blanket relaxation.
    """
    hold_bars = max(int(ENTRY_C_HOLD_BARS), 1)
    if len(hourly_closes) < hold_bars:
        return False
    if not ENTRY_C_STANDARD_ALLOW_CONTINUATION and continuation:
        return False
    if campaign.bars_since_breakout > ENTRY_C_STANDARD_MAX_BREAKOUT_BARS:
        return False

    last_close = hourly_closes[-1]

    # Extension guard: reject if price too far from EMA20.
    if ema20_h > 0 and atr14_h > 0:
        extension = abs(last_close - ema20_h)
        if extension > 2.5 * atr14_h:
            return False
    if atr14_d > 0:
        disp_from_avwap = abs(last_close - avwap_h) / atr14_d
        if disp_from_avwap > ENTRY_C_STANDARD_MAX_DISP_H:
            return False

    if hold_bars == 1:
        if ENTRY_C_FAST_REQUIRE_ALIGNED and regime_4h is not None and not is_regime_aligned(direction, regime_4h):
            return False
        if score_total < ENTRY_C_FAST_MIN_SCORE:
            return False
        if quality_mult < ENTRY_C_FAST_MIN_QUALITY:
            return False
        return last_close > avwap_h if direction == Direction.LONG else last_close < avwap_h

    recent_closes = hourly_closes[-hold_bars:]
    if direction == Direction.LONG:
        return all(close > avwap_h for close in recent_closes)
    return all(close < avwap_h for close in recent_closes)


# ---------------------------------------------------------------------------
# Entry C continuation (spec §12.6)
# ---------------------------------------------------------------------------

def entry_c_continuation_signal(
    direction: Direction,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
) -> bool:
    """Entry C continuation: hold + pause constraint.

    Requires 2-bar hold AND max range of last 2 bars <= 0.40*ATR14_H.
    """
    if len(hourly_closes) < 2 or len(hourly_highs) < 2:
        return False

    # Hold check
    if direction == Direction.LONG:
        if not (hourly_closes[-1] > avwap_h and hourly_closes[-2] > avwap_h):
            return False
    else:
        if not (hourly_closes[-1] < avwap_h and hourly_closes[-2] < avwap_h):
            return False

    # Pause: max range of last 2 bars <= 0.40 * ATR14_H
    range_1 = hourly_highs[-1] - hourly_lows[-1]
    range_2 = hourly_highs[-2] - hourly_lows[-2]
    max_range = max(range_1, range_2)
    return max_range <= 0.40 * atr14_h


# ---------------------------------------------------------------------------
# Entry C fresh / early standard: reclaim and early hold extraction
# ---------------------------------------------------------------------------

def _entry_disp_from_avwap(hourly_close: float, avwap_h: float, atr14_d: float) -> float:
    if atr14_d <= 0:
        return 0.0
    return abs(hourly_close - avwap_h) / atr14_d


def _regime_allows_branch(
    direction: Direction,
    regime_4h: Regime4H,
    *,
    require_aligned: bool,
    allow_countertrend: bool,
    allow_neutral: bool = True,
) -> tuple[bool, str]:
    if require_aligned:
        return is_regime_aligned(direction, regime_4h), "regime_misaligned"
    if regime_4h == Regime4H.RANGE_CHOP:
        return allow_neutral, "neutral_blocked"
    if is_regime_aligned(direction, regime_4h):
        return True, "ok"
    if allow_countertrend:
        return True, "ok"
    return False, "countertrend"


def _assess_c_continuation_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    rvol_h: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
) -> tuple[bool, str]:
    if not ENTRY_C_CONTINUATION_ENABLE:
        return False, "disabled"
    if not campaign.continuation:
        return False, "not_continuation"

    allowed, reason = _regime_allows_branch(
        direction,
        regime_4h,
        require_aligned=ENTRY_C_CONTINUATION_REQUIRE_ALIGNED,
        allow_countertrend=False,
        allow_neutral=ENTRY_C_CONTINUATION_ALLOW_NEUTRAL,
    )
    if not allowed:
        return False, reason
    if score_total < ENTRY_C_CONTINUATION_MIN_SCORE:
        return False, "score"
    if quality_mult < ENTRY_C_CONTINUATION_MIN_QUALITY:
        return False, "quality"
    if rvol_h < ENTRY_C_CONTINUATION_MIN_RVOL_H:
        return False, "rvol_h"
    if campaign.bars_since_breakout > ENTRY_C_CONTINUATION_MAX_BREAKOUT_BARS:
        return False, "bars_since_breakout"

    hold_bars = max(1, int(ENTRY_C_CONTINUATION_HOLD_BARS))
    if len(hourly_closes) < hold_bars or len(hourly_highs) < hold_bars or len(hourly_lows) < hold_bars:
        return False, "history"

    recent_closes = [float(v) for v in hourly_closes[-hold_bars:]]
    if direction == Direction.LONG:
        if any(close <= avwap_h for close in recent_closes):
            return False, "hold_fail"
    else:
        if any(close >= avwap_h for close in recent_closes):
            return False, "hold_fail"

    if atr14_h > 0:
        max_range = max(
            float(high) - float(low)
            for high, low in zip(hourly_highs[-hold_bars:], hourly_lows[-hold_bars:])
        )
        if max_range > ENTRY_C_CONTINUATION_PAUSE_ATR_H * atr14_h:
            return False, "pause"

    if _entry_disp_from_avwap(hourly_close, avwap_h, atr14_d) > ENTRY_C_CONTINUATION_MAX_DISP_H:
        return False, "displacement"

    if ENTRY_C_CONTINUATION_CLV_Q > 0 and not strong_close_location(
        direction,
        hourly_high,
        hourly_low,
        hourly_close,
        q=ENTRY_C_CONTINUATION_CLV_Q,
    ):
        return False, "strong_close_fail"

    return True, "ok"


def _assess_c_fresh_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
    *,
    enabled: bool,
    min_score: int,
    min_quality: float,
    require_aligned: bool,
    allow_countertrend: bool,
    clv_q: float,
    max_disp_h: float,
    touch_tol_atr_h: float,
    max_breakout_bars: int,
    min_rvol_h: float = 0.0,
    rvol_h: float = 0.0,
) -> tuple[bool, str]:
    if not enabled:
        return False, "disabled"
    if continuation:
        return False, "continuation"
    allowed, reason = _regime_allows_branch(
        direction,
        regime_4h,
        require_aligned=require_aligned,
        allow_countertrend=allow_countertrend,
    )
    if not allowed:
        return False, reason
    if score_total < min_score:
        return False, "score"
    if quality_mult < min_quality:
        return False, "quality"
    if min_rvol_h > 0 and rvol_h < min_rvol_h:
        return False, "rvol_h"
    if campaign.bars_since_breakout > max_breakout_bars:
        return False, "bars_since_breakout"
    if len(hourly_closes) < 2 or len(hourly_highs) < 2 or len(hourly_lows) < 2:
        return False, "history"

    prev_close = float(hourly_closes[-2])
    prev_high = float(hourly_highs[-2])
    prev_low = float(hourly_lows[-2])
    touch_tol = touch_tol_atr_h * atr14_h if atr14_h > 0 else 0.0

    if direction == Direction.LONG:
        confirmed = hourly_close > avwap_h
        fresh_reclaim = (
            prev_close <= avwap_h
            or hourly_low <= avwap_h + touch_tol
            or prev_low <= avwap_h + touch_tol
        )
    else:
        confirmed = hourly_close < avwap_h
        fresh_reclaim = (
            prev_close >= avwap_h
            or hourly_high >= avwap_h - touch_tol
            or prev_high >= avwap_h - touch_tol
        )

    fresh_context = fresh_reclaim or campaign.bars_since_breakout <= 1
    if not confirmed or not fresh_context:
        return False, "reclaim_fail"

    if _entry_disp_from_avwap(hourly_close, avwap_h, atr14_d) > max_disp_h:
        return False, "displacement"

    if not strong_close_location(direction, hourly_high, hourly_low, hourly_close, q=clv_q):
        return False, "strong_close_fail"
    return True, "ok"


def entry_c_fresh_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
) -> bool:
    """Broader market-entry fresh branch."""
    ok, _reason = _assess_c_fresh_signal(
        direction,
        campaign,
        hourly_close,
        hourly_low,
        hourly_high,
        hourly_closes,
        hourly_highs,
        hourly_lows,
        avwap_h,
        atr14_h,
        atr14_d,
        score_total,
        quality_mult,
        regime_4h,
        continuation,
        enabled=ENTRY_C_FRESH_ENABLE,
        min_score=ENTRY_C_FRESH_MIN_SCORE,
        min_quality=ENTRY_C_FRESH_MIN_QUALITY,
        require_aligned=ENTRY_C_FRESH_REQUIRE_ALIGNED,
        allow_countertrend=ENTRY_C_FRESH_ALLOW_COUNTERTREND,
        clv_q=ENTRY_C_FRESH_CLV_Q,
        max_disp_h=ENTRY_C_FRESH_MAX_DISP_H,
        touch_tol_atr_h=ENTRY_C_FRESH_TOUCH_TOL_ATR_H,
        max_breakout_bars=ENTRY_C_FRESH_MAX_BREAKOUT_BARS,
    )
    return ok


def entry_c_fresh_stop_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
    rvol_h: float,
) -> bool:
    """Top-decile confirmation fresh branch that keeps stop-limit confirmation separate."""
    ok, _reason = _assess_c_fresh_signal(
        direction,
        campaign,
        hourly_close,
        hourly_low,
        hourly_high,
        hourly_closes,
        hourly_highs,
        hourly_lows,
        avwap_h,
        atr14_h,
        atr14_d,
        score_total,
        quality_mult,
        regime_4h,
        continuation,
        enabled=ENTRY_C_FRESH_STOP_ENABLE,
        min_score=ENTRY_C_FRESH_STOP_MIN_SCORE,
        min_quality=ENTRY_C_FRESH_STOP_MIN_QUALITY,
        require_aligned=ENTRY_C_FRESH_STOP_REQUIRE_ALIGNED,
        allow_countertrend=ENTRY_C_FRESH_STOP_ALLOW_COUNTERTREND,
        clv_q=ENTRY_C_FRESH_STOP_CLV_Q,
        max_disp_h=ENTRY_C_FRESH_STOP_MAX_DISP_H,
        touch_tol_atr_h=ENTRY_C_FRESH_STOP_TOUCH_TOL_ATR_H,
        max_breakout_bars=ENTRY_C_FRESH_STOP_MAX_BREAKOUT_BARS,
        min_rvol_h=ENTRY_C_FRESH_STOP_MIN_RVOL_H,
        rvol_h=rvol_h,
    )
    return ok


def entry_c_early_standard_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    rvol_h: float,
    atr_expanding: bool,
    ema20_h: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
) -> bool:
    """Extract the earlier, lower-displacement subset hiding inside legacy C_standard."""
    ok, _reason = _assess_c_early_standard_signal(
        direction,
        campaign,
        hourly_close,
        hourly_low,
        hourly_high,
        hourly_closes,
        avwap_h,
        atr14_h,
        atr14_d,
        rvol_h,
        atr_expanding,
        ema20_h,
        score_total,
        quality_mult,
        regime_4h,
        continuation,
    )
    return ok


def _assess_c_early_standard_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    rvol_h: float,
    atr_expanding: bool,
    ema20_h: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
) -> tuple[bool, str]:
    """Return early-standard decision plus first rejection reason for attribution."""
    if not ENTRY_C_EARLY_ENABLE:
        return False, "disabled"
    if continuation:
        return False, "continuation"
    allowed, _reason = _regime_allows_branch(
        direction,
        regime_4h,
        require_aligned=ENTRY_C_EARLY_REQUIRE_ALIGNED,
        allow_countertrend=False,
        allow_neutral=ENTRY_C_EARLY_ALLOW_NEUTRAL,
    )
    if not allowed:
        return False, "countertrend"
    if score_total < ENTRY_C_EARLY_MIN_SCORE:
        return False, "score"
    if quality_mult < ENTRY_C_EARLY_MIN_QUALITY:
        return False, "quality"
    if rvol_h < ENTRY_C_EARLY_MIN_RVOL_H:
        return False, "rvol_h"
    if campaign.bars_since_breakout > ENTRY_C_EARLY_MAX_BREAKOUT_BARS:
        return False, "bars_since_breakout"
    if not entry_c_standard_signal(
        direction,
        campaign,
        hourly_closes,
        avwap_h,
        atr_expanding=atr_expanding,
        rvol_h=rvol_h,
        atr14_h=atr14_h,
        atr14_d=atr14_d,
        ema20_h=ema20_h,
        quality_mult=quality_mult,
        regime_4h=regime_4h,
        score_total=score_total,
        continuation=False,
    ):
        return False, "hold_fail"
    if _entry_disp_from_avwap(hourly_close, avwap_h, atr14_d) > ENTRY_C_EARLY_MAX_DISP_H:
        return False, "displacement"
    if not strong_close_location(
        direction,
        hourly_high,
        hourly_low,
        hourly_close,
        q=ENTRY_C_EARLY_CLV_Q,
    ):
        return False, "strong_close_fail"
    return True, "ok"


# ---------------------------------------------------------------------------
# Entry selection (spec §12, A→B→C priority)
# ---------------------------------------------------------------------------

def _assess_c_momentum_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_high: float,
    hourly_low: float,
    hourly_closes: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    rvol_h: float,
    atr_expanding: bool,
    ema20_h: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
) -> tuple[bool, str]:
    if not ENTRY_C_MOMENTUM_ENABLE:
        return False, "disabled"
    if continuation:
        return False, "continuation"
    if ENTRY_C_MOMENTUM_REQUIRE_ALIGNED and not is_regime_aligned(direction, regime_4h):
        return False, "regime_misaligned"
    if score_total < ENTRY_C_MOMENTUM_MIN_SCORE:
        return False, "score"
    if quality_mult < ENTRY_C_MOMENTUM_MIN_QUALITY:
        return False, "quality"
    if not entry_c_standard_signal(
        direction,
        campaign,
        hourly_closes,
        avwap_h,
        atr_expanding=atr_expanding,
        rvol_h=rvol_h,
        atr14_h=atr14_h,
        atr14_d=atr14_d,
        ema20_h=ema20_h,
        quality_mult=quality_mult,
        regime_4h=regime_4h,
        score_total=score_total,
        continuation=continuation,
    ):
        return False, "standard_hold"
    if atr14_d > 0:
        disp_from_avwap = abs(hourly_close - avwap_h) / atr14_d
        if disp_from_avwap > ENTRY_C_MOMENTUM_MAX_DISP_H:
            return False, "displacement"

    if not strong_close_location(
        direction,
        hourly_high,
        hourly_low,
        hourly_close,
        q=ENTRY_C_MOMENTUM_CLV_Q,
    ):
        return False, "strong_close_fail"
    return True, "ok"


def entry_c_momentum_signal(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_high: float,
    hourly_low: float,
    hourly_closes: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    rvol_h: float,
    atr_expanding: bool,
    ema20_h: float,
    score_total: int,
    quality_mult: float,
    regime_4h: Regime4H,
    continuation: bool,
) -> bool:
    """Fast breakout branch for high-conviction C-hold signals."""
    ok, _reason = _assess_c_momentum_signal(
        direction,
        campaign,
        hourly_close,
        hourly_high,
        hourly_low,
        hourly_closes,
        avwap_h,
        atr14_h,
        atr14_d,
        rvol_h,
        atr_expanding,
        ema20_h,
        score_total,
        quality_mult,
        regime_4h,
        continuation,
    )
    return ok


def entry_outside_window_carry_allowed(
    entry_type: EntryType,
    direction: Direction,
    regime_4h: Regime4H,
    score_total: int,
    quality_mult: float,
    continuation: bool,
) -> bool:
    """Allow after-hours carry only for high-conviction non-continuation entries."""
    if not ENTRY_OUTSIDE_WINDOW_CARRY_ENABLE:
        return False
    if continuation:
        return False
    if ENTRY_OUTSIDE_WINDOW_CARRY_REQUIRE_ALIGNED and not is_regime_aligned(direction, regime_4h):
        return False
    if score_total < ENTRY_OUTSIDE_WINDOW_CARRY_MIN_SCORE:
        return False
    if quality_mult < ENTRY_OUTSIDE_WINDOW_CARRY_MIN_QUALITY:
        return False
    if ENTRY_OUTSIDE_WINDOW_CARRY_FRESH_ONLY:
        return entry_type in (
            EntryType.C_FRESH_MARKET,
            EntryType.C_FRESH_STOP,
        )
    if ENTRY_OUTSIDE_WINDOW_CARRY_A_OR_FRESH_ONLY:
        return entry_type in (
            EntryType.A_AVWAP_RETEST,
            EntryType.A_RECLAIM_STRONG,
            EntryType.A_RECLAIM_STRONG_STOP,
            EntryType.C_EARLY_STANDARD,
            EntryType.C_FRESH_MARKET,
            EntryType.C_FRESH_STOP,
        )
    return entry_type in (
        EntryType.A_AVWAP_RETEST,
        EntryType.A_RECLAIM_STRONG,
        EntryType.A_RECLAIM_STRONG_STOP,
        EntryType.B_RESUME_MARKET,
        EntryType.B_RESUME_STOP,
        EntryType.C_EARLY_STANDARD,
        EntryType.C_STANDARD,
        EntryType.C_FRESH_MARKET,
        EntryType.C_FRESH_STOP,
        EntryType.C_MOMENTUM_MARKET,
        EntryType.C_MOMENTUM_STOP,
    )


def select_entry_type(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    rvol_h: float,
    disp_mult: float,
    quality_mult: float,
    regime_4h: Regime4H,
    score_total: int = 0,
    entry_a_active: bool = False,
    atr_expanding: bool = True,
    ema20_h: float = 0.0,
    selection_trace: dict | None = None,
) -> Optional[EntryType]:
    """Select entry type: A -> B -> fresh/early C branches -> legacy C_standard."""
    if selection_trace is not None:
        selection_trace.clear()
        selection_trace["bars_since_breakout"] = campaign.bars_since_breakout
        selection_trace["continuation"] = campaign.continuation
        selection_trace["entry_disp_from_avwap"] = _entry_disp_from_avwap(hourly_close, avwap_h, atr14_d)
    """Select entry type: A → B → C_standard (or C_continuation if in continuation)."""
    # Continuation: only C_continuation allowed (require CONTINUATION state, not just flag)
    if campaign.continuation and campaign.state == CampaignState.CONTINUATION:
        if entry_c_continuation_signal(
            direction, hourly_closes, hourly_highs, hourly_lows, avwap_h, atr14_h
        ):
            # Quality gate: require regime alignment + minimum quality
            if not is_regime_aligned(direction, regime_4h):
                return None
            if quality_mult < 0.70:
                return None
            return EntryType.C_CONTINUATION
        return None

    # Entry B resume — fresh failed probe back through AVWAP.
    if entry_b_resume_signal(
        direction,
        campaign,
        hourly_close,
        hourly_low,
        hourly_high,
        hourly_closes,
        hourly_highs,
        hourly_lows,
        avwap_h,
        atr14_h,
        atr14_d,
        score_total,
        quality_mult,
        regime_4h,
        campaign.continuation,
    ):
        if ENTRY_B_RESUME_USE_STOP_LIMIT:
            return EntryType.B_RESUME_STOP
        return EntryType.B_RESUME_MARKET

    # Entry A — 3-bar lookback for AVWAP touch
    prev_low = float(hourly_lows[-2]) if len(hourly_lows) >= 2 else 0.0
    prev_high = float(hourly_highs[-2]) if len(hourly_highs) >= 2 else 0.0
    prev2_low = float(hourly_lows[-3]) if len(hourly_lows) >= 3 else 0.0
    prev2_high = float(hourly_highs[-3]) if len(hourly_highs) >= 3 else 0.0
    if entry_a_reclaim_strong_signal(
        direction,
        hourly_close,
        hourly_low,
        hourly_high,
        avwap_h,
        atr14_h,
        atr14_d,
        score_total,
        quality_mult,
        regime_4h,
        campaign.continuation,
        prev_low,
        prev_high,
        prev2_low,
        prev2_high,
    ):
        if ENTRY_A_STRONG_USE_STOP_LIMIT:
            return EntryType.A_RECLAIM_STRONG_STOP
        return EntryType.A_RECLAIM_STRONG

    if entry_a_signal(direction, hourly_close, hourly_low, hourly_high,
                      avwap_h, atr14_h, atr14_d, prev_low, prev_high,
                      prev2_low, prev2_high):
        return EntryType.A_AVWAP_RETEST

    # Entry B (if permitted)
    if entry_b_signal(direction, hourly_low, hourly_high, hourly_close,
                      avwap_h, atr14_d, prev_low, prev_high):
        if entry_b_permitted(rvol_h, disp_mult, quality_mult,
                             regime_4h, direction, campaign):
            return EntryType.B_SWEEP_RECLAIM

    # Entry C standard — only if Entry A is not outstanding (spec §12.5)
    if (not entry_a_active or not ENTRY_A_ACTIVE_BLOCKS_C) and entry_c_fresh_signal(
        direction,
        campaign,
        hourly_close,
        hourly_low,
        hourly_high,
        hourly_closes,
        hourly_highs,
        hourly_lows,
        avwap_h,
        atr14_h,
        atr14_d,
        score_total,
        quality_mult,
        regime_4h,
        campaign.continuation,
    ):
        if ENTRY_C_FRESH_USE_STOP_LIMIT:
            return EntryType.C_FRESH_STOP
        return EntryType.C_FRESH_MARKET

    if (not entry_a_active or not ENTRY_A_ACTIVE_BLOCKS_C) and entry_c_momentum_signal(
        direction,
        campaign,
        hourly_close,
        hourly_high,
        hourly_low,
        hourly_closes,
        avwap_h,
        atr14_h,
        atr14_d,
        rvol_h,
        atr_expanding,
        ema20_h,
        score_total,
        quality_mult,
        regime_4h,
        campaign.continuation,
    ):
        if ENTRY_C_MOMENTUM_USE_STOP_LIMIT:
            return EntryType.C_MOMENTUM_STOP
        return EntryType.C_MOMENTUM_MARKET

    if (not entry_a_active or not ENTRY_A_ACTIVE_BLOCKS_C) and entry_c_standard_signal(
        direction, campaign, hourly_closes, avwap_h,
        atr_expanding=atr_expanding, rvol_h=rvol_h, atr14_h=atr14_h,
        atr14_d=atr14_d,
        ema20_h=ema20_h, quality_mult=quality_mult, regime_4h=regime_4h,
        score_total=score_total,
        continuation=campaign.continuation,
    ):
        return EntryType.C_STANDARD

    return None


# ---------------------------------------------------------------------------
# Add trigger (spec §13.2)
# ---------------------------------------------------------------------------

def check_add_trigger(
    direction: Direction,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    ref: float,
    prev_close: float,
    rvol_h: float,
    high_vol_regime: bool,
    reclaim_buffer: float,
) -> bool:
    """Check if pullback add trigger fires.

    Touch ref + resume (first close back in direction).
    """
    if direction == Direction.LONG:
        touched = hourly_low <= ref
        if high_vol_regime:
            resumed = hourly_close > ref + reclaim_buffer
        else:
            resumed = hourly_close > ref and prev_close <= ref
        return touched and resumed
    else:
        touched = hourly_high >= ref
        if high_vol_regime:
            resumed = hourly_close < ref - reclaim_buffer
        else:
            resumed = hourly_close < ref and prev_close >= ref
        return touched and resumed


def strong_close_location(
    direction: Direction,
    hourly_high: float,
    hourly_low: float,
    hourly_close: float,
    q: float = ADD_CLV_STRONG_Q,
) -> bool:
    """CLV acceptance for low-volume adds.

    Long: close in top q% of bar range. Short: bottom q%.
    """
    bar_range = hourly_high - hourly_low
    if bar_range <= 0:
        return False
    if direction == Direction.LONG:
        return (hourly_close - hourly_low) / bar_range >= (1.0 - q)
    else:
        return (hourly_high - hourly_close) / bar_range >= (1.0 - q)


# ---------------------------------------------------------------------------
# Regime alignment helpers
# ---------------------------------------------------------------------------

def is_regime_aligned(direction: Direction, regime_4h: Regime4H) -> bool:
    """Check if direction aligns with 4H regime."""
    return (
        (direction == Direction.LONG and regime_4h == Regime4H.BULL_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BEAR_TREND)
    )


def determine_trade_regime(
    direction: Direction,
    regime_4h: Regime4H,
) -> TradeRegime:
    """Map direction + 4H regime to trade regime for exit tier."""
    if is_regime_aligned(direction, regime_4h):
        return TradeRegime.ALIGNED
    opposes = (
        (direction == Direction.LONG and regime_4h == Regime4H.BEAR_TREND)
        or (direction == Direction.SHORT and regime_4h == Regime4H.BULL_TREND)
    )
    if opposes:
        return TradeRegime.CAUTION
    return TradeRegime.NEUTRAL


# ---------------------------------------------------------------------------
# Chop mode (spec §7.2)
# ---------------------------------------------------------------------------

def compute_chop_score(
    atr14_d: float,
    atr_hist: list[float],
    closes: np.ndarray,
    ema_ref: np.ndarray,
) -> int:
    """Daily chop score: ATR percentile + EMA cross count."""
    score = 0
    # ATR percentile in 60-bar window
    window = atr_hist[-60:] if len(atr_hist) >= 60 else atr_hist
    if window:
        pctl = sum(1 for x in window if x <= atr14_d) / len(window)
        if pctl > CHOP_ATR_PCTL_HIGH:
            score += 1
    # EMA cross count in last N bars
    n = min(CHOP_CROSS_LOOKBACK, len(closes) - 1, len(ema_ref) - 1)
    if n > 0:
        crosses = 0
        for i in range(-n, -1):
            if (closes[i] > ema_ref[i]) != (closes[i + 1] > ema_ref[i + 1]):
                crosses += 1
        if crosses >= CHOP_CROSS_THRESHOLD:
            score += 1
    return score


def classify_chop_mode(chop_score: int) -> ChopMode:
    """Classify chop mode from score: 0=NORMAL, 1=DEGRADED, 2+=HALT."""
    if chop_score >= 2:
        return ChopMode.HALT
    if chop_score >= 1:
        return ChopMode.DEGRADED
    return ChopMode.NORMAL


def select_entry_type(
    direction: Direction,
    campaign: SymbolCampaign,
    hourly_close: float,
    hourly_low: float,
    hourly_high: float,
    hourly_closes: list[float],
    hourly_highs: list[float],
    hourly_lows: list[float],
    avwap_h: float,
    atr14_h: float,
    atr14_d: float,
    rvol_h: float,
    disp_mult: float,
    quality_mult: float,
    regime_4h: Regime4H,
    score_total: int = 0,
    entry_a_active: bool = False,
    atr_expanding: bool = True,
    ema20_h: float = 0.0,
    selection_trace: dict | None = None,
) -> Optional[EntryType]:
    """Select entry type with branch attribution for fresh/early rerouting."""
    if selection_trace is not None:
        selection_trace.clear()
        selection_trace["bars_since_breakout"] = campaign.bars_since_breakout
        selection_trace["continuation"] = campaign.continuation
        selection_trace["entry_disp_from_avwap"] = _entry_disp_from_avwap(hourly_close, avwap_h, atr14_d)

    if entry_b_resume_signal(
        direction,
        campaign,
        hourly_close,
        hourly_low,
        hourly_high,
        hourly_closes,
        hourly_highs,
        hourly_lows,
        avwap_h,
        atr14_h,
        atr14_d,
        score_total,
        quality_mult,
        regime_4h,
        campaign.continuation,
    ):
        return EntryType.B_RESUME_STOP if ENTRY_B_RESUME_USE_STOP_LIMIT else EntryType.B_RESUME_MARKET

    prev_low = float(hourly_lows[-2]) if len(hourly_lows) >= 2 else 0.0
    prev_high = float(hourly_highs[-2]) if len(hourly_highs) >= 2 else 0.0
    prev2_low = float(hourly_lows[-3]) if len(hourly_lows) >= 3 else 0.0
    prev2_high = float(hourly_highs[-3]) if len(hourly_highs) >= 3 else 0.0
    if entry_a_reclaim_strong_signal(
        direction,
        hourly_close,
        hourly_low,
        hourly_high,
        avwap_h,
        atr14_h,
        atr14_d,
        score_total,
        quality_mult,
        regime_4h,
        campaign.continuation,
        prev_low,
        prev_high,
        prev2_low,
        prev2_high,
    ):
        return EntryType.A_RECLAIM_STRONG_STOP if ENTRY_A_STRONG_USE_STOP_LIMIT else EntryType.A_RECLAIM_STRONG

    if entry_a_signal(
        direction,
        hourly_close,
        hourly_low,
        hourly_high,
        avwap_h,
        atr14_h,
        atr14_d,
        prev_low,
        prev_high,
        prev2_low,
        prev2_high,
    ):
        return EntryType.A_AVWAP_RETEST

    if entry_b_signal(direction, hourly_low, hourly_high, hourly_close, avwap_h, atr14_d, prev_low, prev_high):
        if entry_b_permitted(rvol_h, disp_mult, quality_mult, regime_4h, direction, campaign):
            return EntryType.B_SWEEP_RECLAIM

    if not entry_a_active or not ENTRY_A_ACTIVE_BLOCKS_C:
        if campaign.continuation:
            continuation_ok, continuation_reason = _assess_c_continuation_signal(
                direction,
                campaign,
                hourly_close,
                hourly_low,
                hourly_high,
                hourly_closes,
                hourly_highs,
                hourly_lows,
                avwap_h,
                atr14_h,
                atr14_d,
                rvol_h,
                score_total,
                quality_mult,
                regime_4h,
            )
            if continuation_ok:
                if selection_trace is not None:
                    selection_trace["selected_entry_type"] = EntryType.C_CONTINUATION.value
                return EntryType.C_CONTINUATION
            if selection_trace is not None:
                selection_trace["continuation_reject_reason"] = continuation_reason
            if not ENTRY_C_STANDARD_ALLOW_CONTINUATION:
                if selection_trace is not None:
                    selection_trace.setdefault("selected_entry_type", "")
                return None

        fresh_stop_ok, fresh_stop_reason = _assess_c_fresh_signal(
            direction,
            campaign,
            hourly_close,
            hourly_low,
            hourly_high,
            hourly_closes,
            hourly_highs,
            hourly_lows,
            avwap_h,
            atr14_h,
            atr14_d,
            score_total,
            quality_mult,
            regime_4h,
            campaign.continuation,
            enabled=ENTRY_C_FRESH_STOP_ENABLE,
            min_score=ENTRY_C_FRESH_STOP_MIN_SCORE,
            min_quality=ENTRY_C_FRESH_STOP_MIN_QUALITY,
            require_aligned=ENTRY_C_FRESH_STOP_REQUIRE_ALIGNED,
            allow_countertrend=ENTRY_C_FRESH_STOP_ALLOW_COUNTERTREND,
            clv_q=ENTRY_C_FRESH_STOP_CLV_Q,
            max_disp_h=ENTRY_C_FRESH_STOP_MAX_DISP_H,
            touch_tol_atr_h=ENTRY_C_FRESH_STOP_TOUCH_TOL_ATR_H,
            max_breakout_bars=ENTRY_C_FRESH_STOP_MAX_BREAKOUT_BARS,
            min_rvol_h=ENTRY_C_FRESH_STOP_MIN_RVOL_H,
            rvol_h=rvol_h,
        )
        if fresh_stop_ok:
            if selection_trace is not None:
                selection_trace["selected_entry_type"] = EntryType.C_FRESH_STOP.value
            return EntryType.C_FRESH_STOP
        if selection_trace is not None:
            selection_trace["fresh_stop_reject_reason"] = fresh_stop_reason

        fresh_market_ok, fresh_market_reason = _assess_c_fresh_signal(
            direction,
            campaign,
            hourly_close,
            hourly_low,
            hourly_high,
            hourly_closes,
            hourly_highs,
            hourly_lows,
            avwap_h,
            atr14_h,
            atr14_d,
            score_total,
            quality_mult,
            regime_4h,
            campaign.continuation,
            enabled=ENTRY_C_FRESH_ENABLE,
            min_score=ENTRY_C_FRESH_MIN_SCORE,
            min_quality=ENTRY_C_FRESH_MIN_QUALITY,
            require_aligned=ENTRY_C_FRESH_REQUIRE_ALIGNED,
            allow_countertrend=ENTRY_C_FRESH_ALLOW_COUNTERTREND,
            clv_q=ENTRY_C_FRESH_CLV_Q,
            max_disp_h=ENTRY_C_FRESH_MAX_DISP_H,
            touch_tol_atr_h=ENTRY_C_FRESH_TOUCH_TOL_ATR_H,
            max_breakout_bars=ENTRY_C_FRESH_MAX_BREAKOUT_BARS,
        )
        if fresh_market_ok:
            if selection_trace is not None:
                selection_trace["selected_entry_type"] = EntryType.C_FRESH_MARKET.value
            return EntryType.C_FRESH_MARKET
        if selection_trace is not None:
            selection_trace["fresh_market_reject_reason"] = fresh_market_reason

        early_standard_ok, early_standard_reason = _assess_c_early_standard_signal(
            direction,
            campaign,
            hourly_close,
            hourly_low,
            hourly_high,
            hourly_closes,
            avwap_h,
            atr14_h,
            atr14_d,
            rvol_h,
            atr_expanding,
            ema20_h,
            score_total,
            quality_mult,
            regime_4h,
            campaign.continuation,
        )
        if early_standard_ok:
            if selection_trace is not None:
                selection_trace["selected_entry_type"] = EntryType.C_EARLY_STANDARD.value
            return EntryType.C_EARLY_STANDARD
        if selection_trace is not None:
            selection_trace["early_standard_reject_reason"] = early_standard_reason

        momentum_ok, momentum_reason = _assess_c_momentum_signal(
            direction,
            campaign,
            hourly_close,
            hourly_high,
            hourly_low,
            hourly_closes,
            avwap_h,
            atr14_h,
            atr14_d,
            rvol_h,
            atr_expanding,
            ema20_h,
            score_total,
            quality_mult,
            regime_4h,
            campaign.continuation,
        )
        if momentum_ok:
            selected = EntryType.C_MOMENTUM_STOP if ENTRY_C_MOMENTUM_USE_STOP_LIMIT else EntryType.C_MOMENTUM_MARKET
            if selection_trace is not None:
                selection_trace["selected_entry_type"] = selected.value
            return selected
        if selection_trace is not None:
            selection_trace["momentum_reject_reason"] = momentum_reason

        if entry_c_standard_signal(
            direction,
            campaign,
            hourly_closes,
            avwap_h,
            atr_expanding=atr_expanding,
            rvol_h=rvol_h,
            atr14_h=atr14_h,
            atr14_d=atr14_d,
            ema20_h=ema20_h,
            quality_mult=quality_mult,
            regime_4h=regime_4h,
            score_total=score_total,
            continuation=campaign.continuation,
        ):
            if selection_trace is not None:
                selection_trace["selected_entry_type"] = EntryType.C_STANDARD.value
            return EntryType.C_STANDARD

    if selection_trace is not None:
        selection_trace.setdefault("selected_entry_type", "")
    return None
