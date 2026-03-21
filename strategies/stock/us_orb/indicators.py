"""Pure indicator and exit helpers."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from statistics import fmean

from .config import ET
from .config import StrategySettings
from .models import DangerState, MinuteBar, PositionState


def minute_key(ts: datetime) -> str:
    return ts.astimezone(ET).strftime("%H:%M")


def compute_opening_range(bars: list[MinuteBar]) -> tuple[float, float, float, float, float]:
    if not bars:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    or_high = max(bar.high for bar in bars)
    or_low = min(bar.low for bar in bars)
    or_mid = (or_high + or_low) / 2.0 if (or_high + or_low) else 0.0
    or_pct = ((or_high - or_low) / or_mid) if or_mid else 0.0
    value15 = sum(bar.dollar_value for bar in bars)
    return or_high, or_low, or_mid, or_pct, value15


def compute_rvol(curr_1m_volume: float, baseline_volume: float) -> float:
    if baseline_volume <= 0:
        return 0.0
    return curr_1m_volume / baseline_volume


def last_n_value(bars: deque[MinuteBar], n: int) -> float:
    return sum(bar.dollar_value for bar in list(bars)[-n:])


def five_min_return(bars: deque[MinuteBar]) -> float:
    sample = list(bars)[-5:]
    if len(sample) < 5 or sample[0].open <= 0:
        return 0.0
    return (sample[-1].close - sample[0].open) / sample[0].open


def rolling_reference_price(bars: deque[MinuteBar]) -> float | None:
    sample = list(bars)[-5:]
    if not sample:
        return None
    return fmean(bar.close for bar in sample)


def entry_buffer(price: float, spread: float, atr1m: float) -> float:
    return max(
        0.03,
        0.05 if price < 10 else 0.0,
        1.5 * spread,
        0.10 * atr1m,
    )


def support_level(or_high: float, vwap: float | None) -> float:
    vwap_ref = vwap if vwap is not None else or_high
    return min(vwap_ref, or_high) * 0.9985


def structure_stop(or_high: float, vwap: float | None, retest_low: float | None, buffer_value: float) -> float:
    if retest_low is not None:
        return retest_low - buffer_value
    vwap_ref = vwap if vwap is not None else or_high
    return min(or_high, vwap_ref) - buffer_value


def atr_stop(entry_price: float, atr1m: float) -> float:
    return entry_price - (1.1 * atr1m)


def adaptive_trail_stop(
    position: PositionState,
    minutes_held: int,
    flow_regime: str,
    imbalance_90s: float,
    spread_pct: float,
    vdm_state: DangerState,
    settings: StrategySettings,
) -> float:
    gain = max(0.0, position.max_favorable_price - position.entry_price)
    if minutes_held <= 15:
        retracement = 0.45
    else:
        retracement = 0.45 + min(0.30, (minutes_held - 15) * 0.02)

    if flow_regime == "outflow":
        retracement = max(retracement, 0.70)
    if imbalance_90s < 0:
        retracement = max(retracement, 0.72)
    if spread_pct > settings.spread_trail_floor_pct:
        retracement = max(retracement, 0.75)
    if vdm_state == DangerState.CAUTION:
        retracement = max(retracement, settings.caution_trail_floor)

    return max(position.final_stop, position.entry_price + gain * retracement)
