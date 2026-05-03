from __future__ import annotations

from dataclasses import dataclass

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp._shared.nq_contract import round_to_tick

from .config import TP1_QUANTILE, TP2_QUANTILE, TICK_SIZE, TradeDirection


@dataclass(frozen=True, slots=True)
class TargetEstimate:
    tp1: float
    tp2: float
    tp1_distance: float
    tp2_distance: float
    model: str = "static_quantile_fallback"


def fallback_targets(
    *,
    entry_price: float,
    direction: TradeDirection,
    ivb: IVBLevels,
    atr_1m: float = 0.0,
    tp1_quantile: float = TP1_QUANTILE,
    tp2_quantile: float = TP2_QUANTILE,
    tick_size: float = TICK_SIZE,
) -> TargetEstimate:
    base = max(20.0, 0.65 * ivb.range_pts, 0.40 * atr_1m)
    tp1_distance = base * (tp1_quantile / TP1_QUANTILE if TP1_QUANTILE else 1.0)
    tp2_distance = max(tp1_distance, base * 1.45 * (tp2_quantile / TP2_QUANTILE if TP2_QUANTILE else 1.0))
    if direction is TradeDirection.LONG:
        tp1 = round_to_tick(entry_price + tp1_distance, tick_size, "up")
        tp2 = round_to_tick(entry_price + tp2_distance, tick_size, "up")
    else:
        tp1 = round_to_tick(entry_price - tp1_distance, tick_size, "down")
        tp2 = round_to_tick(entry_price - tp2_distance, tick_size, "down")
    return TargetEstimate(tp1=tp1, tp2=tp2, tp1_distance=tp1_distance, tp2_distance=tp2_distance)


def reclaim_targets(
    *,
    entry_price: float,
    direction: TradeDirection,
    ivb: IVBLevels,
    tick_size: float = TICK_SIZE,
) -> TargetEstimate:
    poc = ivb.poc or ivb.mid
    vah = ivb.vah or ivb.high
    val = ivb.val or ivb.low
    if direction is TradeDirection.SHORT:
        tp1 = poc if entry_price > poc else val
        tp2 = ivb.low
        tp1 = round_to_tick(min(entry_price - tick_size, tp1), tick_size, "down")
        tp2 = round_to_tick(min(tp1 - tick_size, tp2), tick_size, "down")
        return TargetEstimate(
            tp1=tp1,
            tp2=tp2,
            tp1_distance=max(0.0, entry_price - tp1),
            tp2_distance=max(0.0, entry_price - tp2),
            model="auction_reclaim_profile",
        )
    if direction is TradeDirection.LONG:
        tp1 = poc if entry_price < poc else vah
        tp2 = ivb.high
        tp1 = round_to_tick(max(entry_price + tick_size, tp1), tick_size, "up")
        tp2 = round_to_tick(max(tp1 + tick_size, tp2), tick_size, "up")
        return TargetEstimate(
            tp1=tp1,
            tp2=tp2,
            tp1_distance=max(0.0, tp1 - entry_price),
            tp2_distance=max(0.0, tp2 - entry_price),
            model="auction_reclaim_profile",
        )
    return TargetEstimate(tp1=entry_price, tp2=entry_price, tp1_distance=0.0, tp2_distance=0.0, model="auction_reclaim_profile")
