from __future__ import annotations

from dataclasses import dataclass, field

from strategies.scalp._shared.levels import IVBLevels

from .config import (
    ADAPTIVE_MAX_IVB_MULT,
    ADAPTIVE_MIN_IVB_MULT,
    DAILY_HARD_LOSS_PCT,
    MIN_BUFFER_PTS,
    MIN_BUFFER_RANGE_FRACTION,
    MIN_HOLD_SECONDS,
    MAX_IVB_RANGE_POINTS,
    MAX_LOSSES_PER_DAY,
    MAX_TRADES_PER_DAY,
    MIN_IVB_RANGE_POINTS,
)


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    details: dict[str, float | str | bool] = field(default_factory=dict)


def ivb_range_gate(ivb: IVBLevels, *, rolling_20day_median: float | None = None) -> GateResult:
    min_range = MIN_IVB_RANGE_POINTS
    max_range = MAX_IVB_RANGE_POINTS
    if rolling_20day_median and rolling_20day_median > 0:
        min_range = max(min_range, ADAPTIVE_MIN_IVB_MULT * rolling_20day_median)
        max_range = min(max_range, ADAPTIVE_MAX_IVB_MULT * rolling_20day_median)
    if ivb.range_pts < min_range:
        return GateResult(False, ("IVB_TOO_NARROW",), {"range": ivb.range_pts, "min": min_range})
    if ivb.range_pts > max_range:
        return GateResult(False, ("IVB_TOO_WIDE",), {"range": ivb.range_pts, "max": max_range})
    return GateResult(True, details={"range": ivb.range_pts})


def risk_gate(
    *,
    daily_pnl: float,
    equity: float,
    trades_today: int,
    losses_today: int,
) -> GateResult:
    reasons: list[str] = []
    if equity > 0 and daily_pnl <= -equity * DAILY_HARD_LOSS_PCT:
        reasons.append("DAILY_HARD_LOSS")
    if trades_today >= MAX_TRADES_PER_DAY:
        reasons.append("MAX_TRADES_PER_DAY")
    if losses_today >= MAX_LOSSES_PER_DAY:
        reasons.append("MAX_LOSSES_PER_DAY")
    return GateResult(not reasons, tuple(reasons), {"daily_pnl": daily_pnl, "trades_today": trades_today})


def breakout_acceptance(
    *,
    direction: int,
    close: float,
    high: float,
    low: float,
    ivb: IVBLevels,
    held_seconds: float,
    breakout_volume: float,
    rolling_volume_median: float,
    delta_60s: float | None,
    rolling_delta_median: float | None,
) -> tuple[bool, dict[str, bool]]:
    buffer = max(MIN_BUFFER_PTS, MIN_BUFFER_RANGE_FRACTION * ivb.range_pts)
    if direction > 0:
        conditions = {
            "close_beyond": close > ivb.high,
            "buffer_beyond": high >= ivb.high + buffer,
            "held": held_seconds >= MIN_HOLD_SECONDS,
            "volume": breakout_volume > rolling_volume_median if rolling_volume_median > 0 else True,
            "delta": delta_60s is not None and (rolling_delta_median is None or delta_60s > rolling_delta_median),
        }
    else:
        conditions = {
            "close_beyond": close < ivb.low,
            "buffer_beyond": low <= ivb.low - buffer,
            "held": held_seconds >= MIN_HOLD_SECONDS,
            "volume": breakout_volume > rolling_volume_median if rolling_volume_median > 0 else True,
            "delta": delta_60s is not None and (rolling_delta_median is None or delta_60s < rolling_delta_median),
        }
    return sum(conditions.values()) >= 3, conditions
