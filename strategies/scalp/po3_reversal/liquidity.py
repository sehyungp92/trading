from __future__ import annotations

from collections.abc import Sequence

from .config import TradeDirection
from .models import LiquidityPool, PriceBar, SweepEvent


def detect_liquidity_pools(
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    symbol: str = "NQ",
    min_touches: int = 2,
    tolerance_ticks: int = 2,
    tick_size: float = 0.25,
) -> list[LiquidityPool]:
    tolerance = tolerance_ticks * tick_size
    pools: list[LiquidityPool] = []
    pools.extend(_cluster_levels(highs, "buy_side", symbol, min_touches, tolerance))
    pools.extend(_cluster_levels(lows, "sell_side", symbol, min_touches, tolerance))
    return sorted(pools, key=lambda pool: (pool.side, pool.price))


def detect_sweep(
    price: float,
    pools: Sequence[LiquidityPool],
    *,
    min_ticks: int,
    tick_size: float = 0.25,
    side: str | None = None,
) -> SweepEvent:
    threshold = min_ticks * tick_size
    candidates = [pool for pool in pools if side is None or pool.side == side]
    for pool in candidates:
        if pool.side == "buy_side" and price >= pool.price + threshold:
            return SweepEvent(
                swept=True,
                direction=TradeDirection.SHORT,
                pool=pool,
                sweep_price=float(price),
                distance_ticks=(float(price) - pool.price) / tick_size,
            )
        if pool.side == "sell_side" and price <= pool.price - threshold:
            return SweepEvent(
                swept=True,
                direction=TradeDirection.LONG,
                pool=pool,
                sweep_price=float(price),
                distance_ticks=(pool.price - float(price)) / tick_size,
            )
    return SweepEvent(swept=False)


def classify_po3_phase(
    bars: Sequence[PriceBar],
    dealing_range,
) -> str:
    if len(bars) < 3 or dealing_range is None:
        return "unknown"
    sample = bars[-min(len(bars), 80):]
    high = max(bar.high for bar in sample)
    low = min(bar.low for bar in sample)
    rng = high - low
    last = sample[-1]
    if rng <= 0:
        return "accumulation"
    if high > dealing_range.high or low < dealing_range.low:
        return "manipulation"
    if last.close > high - 0.25 * rng or last.close < low + 0.25 * rng:
        return "distribution"
    return "accumulation"


def _cluster_levels(
    values: Sequence[float],
    side: str,
    symbol: str,
    min_touches: int,
    tolerance: float,
) -> list[LiquidityPool]:
    clusters: list[dict] = []
    for idx, raw in enumerate(values):
        price = float(raw)
        match = None
        for cluster in clusters:
            if abs(price - cluster["price"]) <= tolerance:
                match = cluster
                break
        if match is None:
            clusters.append({"price": price, "touches": 1, "first": idx, "last": idx})
        else:
            touches = match["touches"] + 1
            match["price"] = ((match["price"] * match["touches"]) + price) / touches
            match["touches"] = touches
            match["last"] = idx
    return [
        LiquidityPool(
            symbol=symbol,
            side=side,
            price=cluster["price"],
            touches=cluster["touches"],
            first_index=cluster["first"],
            last_index=cluster["last"],
        )
        for cluster in clusters
        if cluster["touches"] >= min_touches
    ]

