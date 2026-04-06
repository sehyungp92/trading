"""BRS entry signal generators -- S1, S2, S3, L1, LH Rejection, BD Continuation.

Each check_*() returns EntrySignal | None. All are pure functions
operating on pre-computed indicator state.
"""
from __future__ import annotations

import numpy as np

from .models import (
    BDArmState,
    BRSRegime,
    DailyContext,
    Direction,
    EntrySignal,
    EntryType,
    HourlyContext,
    LHArmState,
    Regime4H,
    S2ArmState,
    S3ArmState,
)
from .config import BRSConfig, BRSSymbolConfig


# ---------------------------------------------------------------------------
# S1 -- Pullback to EMA in bear trend (spec 7)
# ---------------------------------------------------------------------------

def check_s1(
    d: DailyContext,
    h: HourlyContext,
    sym_cfg: BRSSymbolConfig,
    cfg: BRSConfig,
) -> EntrySignal | None:
    """Check S1 pullback-to-EMA entry in bear trend."""
    if cfg.disable_s1:
        return None

    if d.bias.confirmed_direction != Direction.SHORT:
        return None

    _crash = cfg.crash_override_enabled and d.bias.crash_override
    _forming = (cfg.forming_entry_enabled
                and d.regime == BRSRegime.BEAR_FORMING
                and d.atr_ratio > cfg.forming_vol_gate)

    if not d.regime_on and not _crash:
        return None
    if d.regime not in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND):
        if not (_crash or _forming):
            return None
    if d.regime_4h == Regime4H.BULL and not _crash:
        return None

    ema_pull = h.ema_pull
    if ema_pull <= 0:
        return None

    if h.high < ema_pull:
        return None
    if h.close >= ema_pull:
        return None
    if h.close >= h.ema_mom:
        return None
    if h.prior_low > 0 and h.close >= h.prior_low:
        return None

    struct_dist = abs(h.high + 0.01 - h.close)
    stop_distance = struct_dist + sym_cfg.stop_buffer_atr * h.atr14_h
    stop_distance = max(stop_distance, sym_cfg.stop_floor_atr * h.atr14_h)
    stop_price = h.close + stop_distance
    risk_per_unit = stop_distance

    if risk_per_unit <= 0:
        return None

    return EntrySignal(
        entry_type=EntryType.S1_PULLBACK,
        direction=Direction.SHORT,
        signal_price=h.close,
        signal_high=h.high,
        signal_low=h.low,
        stop_price=stop_price,
        risk_per_unit=risk_per_unit,
        bear_conviction=d.bear_conviction,
        quality_score=d.vol.vol_factor,
        regime_at_entry=d.regime,
        vol_factor=d.vol.vol_factor,
    )


# ---------------------------------------------------------------------------
# S2 -- Breakdown-pullback after campaign break (spec 8)
# ---------------------------------------------------------------------------

def check_s2(
    d: DailyContext,
    h: HourlyContext,
    sym_cfg: BRSSymbolConfig,
    cfg: BRSConfig,
    s2_arm: S2ArmState,
) -> EntrySignal | None:
    """Check S2 breakdown-pullback entry while armed."""
    if cfg.disable_s2:
        return None
    if not s2_arm.armed:
        return None

    if d.bias.confirmed_direction != Direction.SHORT:
        return None
    if d.regime not in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND):
        return None

    pull_ref = h.avwap_h if h.avwap_h > 0 else h.ema_pull
    if pull_ref <= 0:
        return None

    reclaim_buffer = max(0.12 * h.atr14_h, 0.03 * d.atr14_d)

    if h.high < pull_ref:
        return None
    if h.close >= pull_ref - reclaim_buffer:
        return None
    if h.prior_low > 0 and h.close >= h.prior_low:
        return None

    atr_stop_mult = 1.0 if sym_cfg.daily_mult <= 2.3 else 1.3
    stop_price = s2_arm.box_high + atr_stop_mult * d.atr14_d
    stop_distance = abs(stop_price - h.close)

    if stop_distance > 1.6 * d.atr14_d:
        return None

    risk_per_unit = stop_distance
    if risk_per_unit <= 0:
        return None

    return EntrySignal(
        entry_type=EntryType.S2_BREAKDOWN,
        direction=Direction.SHORT,
        signal_price=h.close,
        signal_high=h.high,
        signal_low=h.low,
        stop_price=stop_price,
        risk_per_unit=risk_per_unit,
        bear_conviction=d.bear_conviction,
        quality_score=d.vol.vol_factor,
        regime_at_entry=d.regime,
        vol_factor=d.vol.vol_factor,
    )


# ---------------------------------------------------------------------------
# S3 -- Impulse continuation in BEAR_STRONG (spec 9)
# ---------------------------------------------------------------------------

def check_s3(
    d: DailyContext,
    h: HourlyContext,
    sym_cfg: BRSSymbolConfig,
    cfg: BRSConfig,
    s3_arm: S3ArmState,
) -> EntrySignal | None:
    """Check S3 impulse continuation entry while armed."""
    if cfg.disable_s3:
        return None
    if not s3_arm.armed:
        return None

    if d.regime not in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_FORMING):
        return None
    if d.bias.confirmed_direction != Direction.SHORT:
        return None

    ema_mom = h.ema_mom
    if ema_mom <= 0:
        return None

    if h.high < ema_mom:
        return None
    if h.close >= ema_mom:
        return None
    if h.prior_low > 0 and h.close >= h.prior_low:
        return None

    struct_dist = abs(h.high + 0.01 - h.close)
    stop_distance = struct_dist + sym_cfg.stop_buffer_atr * h.atr14_h
    stop_distance = max(stop_distance, sym_cfg.stop_floor_atr * h.atr14_h)
    stop_price = h.close + stop_distance
    risk_per_unit = stop_distance

    if risk_per_unit <= 0:
        return None

    return EntrySignal(
        entry_type=EntryType.S3_IMPULSE,
        direction=Direction.SHORT,
        signal_price=h.close,
        signal_high=h.high,
        signal_low=h.low,
        stop_price=stop_price,
        risk_per_unit=risk_per_unit,
        bear_conviction=d.bear_conviction,
        quality_score=d.vol.vol_factor,
        regime_at_entry=d.regime,
        vol_factor=d.vol.vol_factor,
    )


# ---------------------------------------------------------------------------
# LH -- Lower-High Rejection
# ---------------------------------------------------------------------------

def check_lh_rejection(
    d: DailyContext,
    h: HourlyContext,
    sym_cfg: BRSSymbolConfig,
    cfg: BRSConfig,
    lh_arm: LHArmState,
) -> EntrySignal | None:
    """Check LH Rejection entry -- armed when lower-high confirmed on hourly."""
    if cfg.disable_lh:
        return None

    _persistence = d.persistence_active and d.bias.confirmed_direction == Direction.SHORT
    if not lh_arm.armed and not _persistence:
        return None

    _crash = cfg.crash_override_enabled and d.bias.crash_override
    _forming = (cfg.forming_entry_enabled
                and d.regime == BRSRegime.BEAR_FORMING
                and d.atr_ratio > cfg.forming_vol_gate)
    _chop_override = (cfg.chop_short_entry_enabled
                      and d.regime == BRSRegime.RANGE_CHOP
                      and d.bias.confirmed_direction == Direction.SHORT
                      and d.bias.peak_drop_override)

    if d.regime == BRSRegime.BEAR_STRONG:
        if d.bias.confirmed_direction != Direction.SHORT:
            return None
    elif d.regime == BRSRegime.BEAR_TREND:
        if d.bias.confirmed_direction != Direction.SHORT:
            return None
        if h.volume_sma20 > 0 and h.volume < cfg.bt_volume_mult * h.volume_sma20:
            return None
    elif _chop_override or _persistence:
        pass
    elif _crash or _forming:
        if d.bias.confirmed_direction != Direction.SHORT:
            return None
    else:
        return None

    if not d.regime_on and not _crash and not _chop_override:
        return None
    if d.regime_4h == Regime4H.BULL and not _crash and not _chop_override:
        return None

    ema_pull = h.ema_pull
    if ema_pull <= 0:
        return None
    if h.high < ema_pull:
        return None
    if h.close >= ema_pull:
        return None
    if h.close >= h.ema_mom:
        return None
    if h.prior_low > 0 and h.close >= h.prior_low:
        return None

    if h.prior_close_3 > 0 and h.atr14_h > 0:
        rally_speed = (h.close - h.prior_close_3) / h.atr14_h
        if rally_speed > cfg.v_reversal_max_rally:
            return None

    swing_stop_dist = abs(lh_arm.swing_high_price + 0.01 - h.close)
    bar_stop_dist = abs(h.high + 0.01 - h.close)

    if h.atr14_h > 0 and swing_stop_dist <= cfg.lh_max_stop_atr * h.atr14_h:
        struct_dist = swing_stop_dist
    else:
        struct_dist = bar_stop_dist

    stop_distance = struct_dist + sym_cfg.stop_buffer_atr * h.atr14_h
    stop_distance = max(stop_distance, sym_cfg.stop_floor_atr * h.atr14_h)
    stop_price = h.close + stop_distance
    risk_per_unit = stop_distance

    if risk_per_unit <= 0:
        return None

    quality = d.vol.vol_factor
    if d.regime == BRSRegime.BEAR_TREND:
        quality *= 0.70
    if _chop_override:
        quality *= cfg.chop_quality_mult
    if _persistence and not lh_arm.armed:
        quality *= cfg.persist_quality_mult_lh

    return EntrySignal(
        entry_type=EntryType.LH_REJECTION,
        direction=Direction.SHORT,
        signal_price=h.close,
        signal_high=h.high,
        signal_low=h.low,
        stop_price=stop_price,
        risk_per_unit=risk_per_unit,
        bear_conviction=d.bear_conviction,
        quality_score=quality,
        regime_at_entry=d.regime,
        vol_factor=d.vol.vol_factor,
    )


# ---------------------------------------------------------------------------
# BD -- Breakdown Continuation
# ---------------------------------------------------------------------------

def check_bd_continuation(
    d: DailyContext,
    h: HourlyContext,
    sym_cfg: BRSSymbolConfig,
    cfg: BRSConfig,
    bd_arm: BDArmState,
) -> EntrySignal | None:
    """Check BD Continuation entry -- armed after volume-confirmed range break."""
    if cfg.disable_bd:
        return None
    if not bd_arm.armed:
        return None

    _crash = cfg.crash_override_enabled and d.bias.crash_override
    _chop_override = (cfg.chop_short_entry_enabled
                      and d.regime == BRSRegime.RANGE_CHOP
                      and d.bias.confirmed_direction == Direction.SHORT
                      and d.bias.peak_drop_override)
    _persistence = d.persistence_active and d.bias.confirmed_direction == Direction.SHORT

    allowed = set()
    if cfg.bd_allow_bear_strong:
        allowed.add(BRSRegime.BEAR_STRONG)
    if cfg.bd_allow_bear_trend:
        allowed.add(BRSRegime.BEAR_TREND)
    if d.regime not in allowed and not _crash and not _chop_override and not _persistence:
        return None
    if not d.regime_on and not _crash and not _chop_override:
        return None

    ema_mom = h.ema_mom
    if ema_mom <= 0:
        return None

    if h.high < ema_mom:
        return None
    if h.close >= ema_mom:
        return None
    if h.prior_low > 0 and h.close >= h.prior_low:
        return None

    struct_dist = abs(bd_arm.breakdown_bar_high + 0.01 - h.close)
    stop_distance = struct_dist + sym_cfg.stop_buffer_atr * h.atr14_h
    stop_distance = max(stop_distance, sym_cfg.stop_floor_atr * h.atr14_h)

    if h.atr14_h > 0 and stop_distance > cfg.bd_max_stop_atr * h.atr14_h:
        return None

    stop_price = h.close + stop_distance
    risk_per_unit = stop_distance

    if risk_per_unit <= 0:
        return None

    quality = d.vol.vol_factor
    if _chop_override:
        quality *= cfg.chop_quality_mult
    if _persistence:
        quality *= cfg.persist_quality_mult_bd

    return EntrySignal(
        entry_type=EntryType.BD_CONTINUATION,
        direction=Direction.SHORT,
        signal_price=h.close,
        signal_high=h.high,
        signal_low=h.low,
        stop_price=stop_price,
        risk_per_unit=risk_per_unit,
        bear_conviction=d.bear_conviction,
        quality_score=quality,
        regime_at_entry=d.regime,
        vol_factor=d.vol.vol_factor,
    )


# ---------------------------------------------------------------------------
# BD arming check
# ---------------------------------------------------------------------------

def check_bd_arm(
    hourly_low: float,
    hourly_high: float,
    hourly_close: float,
    hourly_volume: float,
    donchian_low_val: float,
    volume_sma20: float,
    regime: BRSRegime,
    cfg: BRSConfig,
    bar_idx: int,
    chop_arm_allowed: bool = False,
    persistence_override: bool = False,
) -> BDArmState | None:
    """Check if an hourly bar qualifies for BD arming."""
    allowed = set()
    if cfg.bd_allow_bear_strong:
        allowed.add(BRSRegime.BEAR_STRONG)
    if cfg.bd_allow_bear_trend:
        allowed.add(BRSRegime.BEAR_TREND)
    if regime not in allowed and not chop_arm_allowed and not persistence_override:
        return None

    if hourly_low >= donchian_low_val:
        return None

    if not persistence_override:
        if volume_sma20 <= 0 or hourly_volume < cfg.bd_volume_mult * volume_sma20:
            return None

    bar_range = hourly_high - hourly_low
    if bar_range <= 0:
        return None
    close_position = (hourly_close - hourly_low) / bar_range
    if close_position > cfg.bd_close_quality:
        return None

    return BDArmState(
        armed=True,
        breakdown_bar_high=hourly_high,
        armed_bar=bar_idx,
        armed_until_bar=bar_idx + cfg.bd_arm_bars,
    )


# ---------------------------------------------------------------------------
# L1 -- GLD defensive long (spec 10)
# ---------------------------------------------------------------------------

def check_l1(
    d: DailyContext,
    h: HourlyContext,
    sym_cfg: BRSSymbolConfig,
    cfg: BRSConfig,
    symbol: str,
    ema_fast_slope: float,
    ema_slow_slope: float,
) -> EntrySignal | None:
    """Check L1 GLD defensive long entry."""
    if cfg.disable_l1:
        return None

    if not sym_cfg.allow_long:
        return None

    if cfg.min_quality_score > 0 and d.vol.vol_factor < cfg.min_quality_score:
        return None

    if d.bias.confirmed_direction != Direction.LONG:
        return None
    if not d.regime_on:
        return None

    if not cfg.disable_long_safety:
        if symbol == "GLD":
            if ema_fast_slope < -0.02 * d.atr14_d:
                return None
        else:
            if ema_fast_slope < 0:
                return None
        if ema_slow_slope < -0.05 * d.atr14_d:
            return None

    if d.regime_4h == Regime4H.BEAR:
        return None

    ema_pull = h.ema_pull
    if ema_pull <= 0:
        return None

    if h.low > ema_pull:
        return None
    if h.close <= ema_pull:
        return None
    if h.close <= h.ema_mom:
        return None
    if h.prior_high > 0 and h.close <= h.prior_high:
        return None

    struct_dist = abs(h.close - (h.low - 0.01))
    stop_distance = struct_dist + sym_cfg.stop_buffer_atr * h.atr14_h
    stop_distance = max(stop_distance, sym_cfg.stop_floor_atr * h.atr14_h)
    stop_price = h.close - stop_distance
    risk_per_unit = stop_distance

    if risk_per_unit <= 0:
        return None

    quality = d.vol.vol_factor
    if d.regime in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_TREND, BRSRegime.BEAR_FORMING):
        quality *= 0.70

    return EntrySignal(
        entry_type=EntryType.L1_GLD_LONG,
        direction=Direction.LONG,
        signal_price=h.close,
        signal_high=h.high,
        signal_low=h.low,
        stop_price=stop_price,
        risk_per_unit=risk_per_unit,
        bear_conviction=d.bear_conviction,
        quality_score=quality,
        regime_at_entry=d.regime,
        vol_factor=d.vol.vol_factor,
    )


# ---------------------------------------------------------------------------
# S2 arming check (spec 5)
# ---------------------------------------------------------------------------

def check_s2_arm(
    daily_closes: np.ndarray,
    daily_lows: np.ndarray,
    daily_highs: np.ndarray,
    daily_volumes: np.ndarray,
    atr14_d: float,
    atr50_d: float,
    cfg: BRSConfig,
    current_bar_idx: int,
) -> S2ArmState | None:
    """Check if a campaign box breakdown qualifies for S2 arming."""
    from .indicators import (
        compute_avwap,
        compute_box_length,
        containment_ratio,
    )

    box_len = compute_box_length(atr14_d, atr50_d)
    if current_bar_idx < box_len:
        return None

    window = slice(current_bar_idx - box_len, current_bar_idx)
    box_high = float(np.max(daily_highs[window]))
    box_low = float(np.min(daily_lows[window]))
    box_height = box_high - box_low

    if box_height <= 0:
        return None

    cr = containment_ratio(daily_closes, box_low, box_high, box_len)
    if cr < cfg.s2_box_containment:
        return None

    squeeze = box_height / atr14_d if atr14_d > 0 else 999
    if squeeze > 1.20:
        return None

    close_today = float(daily_closes[current_bar_idx])
    if close_today >= box_low:
        return None

    avwap = compute_avwap(daily_closes[:current_bar_idx + 1], daily_volumes[:current_bar_idx + 1], current_bar_idx - box_len)
    disp = abs(close_today - avwap) / atr14_d if atr14_d > 0 else 0
    if disp < cfg.s2_disp_quantile:
        return None

    bar_range = float(daily_highs[current_bar_idx] - daily_lows[current_bar_idx])
    body = abs(float(daily_closes[current_bar_idx]) - float(daily_closes[current_bar_idx - 1])) if current_bar_idx > 0 else bar_range
    if bar_range > 2.5 * atr14_d and bar_range > 0 and body / bar_range < 0.20:
        return None

    return S2ArmState(
        armed=True,
        armed_until_bar=current_bar_idx + 16,
        box_high=box_high,
        box_low=box_low,
        avwap_anchor=avwap,
    )


# ---------------------------------------------------------------------------
# S3 arming check (spec 9.1)
# ---------------------------------------------------------------------------

def check_s3_arm(
    regime: BRSRegime,
    confirmed_bear: bool,
    hourly_low: float,
    hourly_close: float,
    donchian_low_val: float,
    ema_mom: float,
    current_bar_idx: int,
) -> S3ArmState | None:
    """Check if a Donchian breakout qualifies for S3 arming."""
    if regime not in (BRSRegime.BEAR_STRONG, BRSRegime.BEAR_FORMING):
        return None
    if not confirmed_bear:
        return None

    if hourly_low >= donchian_low_val:
        return None
    if hourly_close >= ema_mom:
        return None

    return S3ArmState(
        armed=True,
        armed_until_bar=current_bar_idx + 12,
    )
