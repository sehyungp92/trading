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
    hourly_closes: list[float],
    avwap_h: float,
    atr_expanding: bool = True,
    rvol_h: float = 1.0,
    atr14_h: float = 0.0,
    ema20_h: float = 0.0,
) -> bool:
    """Entry C: 2 consecutive hourly closes on the correct side of AVWAP_H.

    Extension guard: reject if price > 2.5 ATR_H from EMA20 (overextended).
    """
    if len(hourly_closes) < 2:
        return False

    # Extension guard: reject if price too far from EMA20
    if ema20_h > 0 and atr14_h > 0:
        extension = abs(hourly_closes[-1] - ema20_h)
        if extension > 2.5 * atr14_h:
            return False

    if direction == Direction.LONG:
        return hourly_closes[-1] > avwap_h and hourly_closes[-2] > avwap_h
    else:
        return hourly_closes[-1] < avwap_h and hourly_closes[-2] < avwap_h


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
# Entry selection (spec §12, A→B→C priority)
# ---------------------------------------------------------------------------

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
    entry_a_active: bool = False,
    atr_expanding: bool = True,
    ema20_h: float = 0.0,
) -> Optional[EntryType]:
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

    # Entry A — 3-bar lookback for AVWAP touch
    prev_low = float(hourly_lows[-2]) if len(hourly_lows) >= 2 else 0.0
    prev_high = float(hourly_highs[-2]) if len(hourly_highs) >= 2 else 0.0
    prev2_low = float(hourly_lows[-3]) if len(hourly_lows) >= 3 else 0.0
    prev2_high = float(hourly_highs[-3]) if len(hourly_highs) >= 3 else 0.0
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
    if not entry_a_active and entry_c_standard_signal(
        direction, hourly_closes, avwap_h,
        atr_expanding=atr_expanding, rvol_h=rvol_h, atr14_h=atr14_h,
        ema20_h=ema20_h,
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
