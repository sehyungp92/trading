from __future__ import annotations

from collections.abc import Sequence

from strategies.scalp._shared.levels import DealingRange, latest_confirmed_dealing_range

from .config import DAILY_LOOKBACK, SetupTier, TradeDirection
from .models import Po3Context, PriceBar


def compute_daily_bias(daily_bars: Sequence[PriceBar], lookback: int = 20) -> TradeDirection:
    if len(daily_bars) < 3:
        return TradeDirection.FLAT
    sample = list(daily_bars[-lookback:])
    high = max(bar.high for bar in sample)
    low = min(bar.low for bar in sample)
    eq = (high + low) / 2.0
    last = sample[-1]
    swept_low = last.low <= min(bar.low for bar in sample[:-1])
    swept_high = last.high >= max(bar.high for bar in sample[:-1])
    if last.close > eq or (last.close < eq and swept_low):
        return TradeDirection.LONG
    if last.close < eq or (last.close > eq and swept_high):
        return TradeDirection.SHORT
    return TradeDirection.FLAT


def compute_h4_bias(
    h4_bars: Sequence[PriceBar],
    daily_bias: TradeDirection,
    *,
    pivot_n: int = 3,
) -> tuple[TradeDirection, DealingRange | None]:
    if len(h4_bars) < pivot_n * 2 + 3:
        return TradeDirection.FLAT, None
    as_of_index = len(h4_bars) - 1
    dealing_range = latest_confirmed_dealing_range(
        [bar.high for bar in h4_bars],
        [bar.low for bar in h4_bars],
        left_n=pivot_n,
        right_n=pivot_n,
        as_of_index=as_of_index,
    )
    if dealing_range is None:
        sample = h4_bars[-20:]
        dealing_range = DealingRange.from_bounds(max(bar.high for bar in sample), min(bar.low for bar in sample))
    last = h4_bars[-1]
    if last.close < dealing_range.eq and daily_bias != TradeDirection.SHORT:
        return TradeDirection.LONG, dealing_range
    if last.close > dealing_range.eq and daily_bias != TradeDirection.LONG:
        return TradeDirection.SHORT, dealing_range
    if daily_bias in {TradeDirection.LONG, TradeDirection.SHORT}:
        return daily_bias, dealing_range
    return TradeDirection.FLAT, dealing_range


def compute_h1_execution_context(
    h1_bars: Sequence[PriceBar],
    direction: TradeDirection,
) -> tuple[TradeDirection, float]:
    if len(h1_bars) < 3:
        return TradeDirection.FLAT, 0.0
    sample = h1_bars[-12:]
    high = max(bar.high for bar in sample)
    low = min(bar.low for bar in sample)
    eq = (high + low) / 2.0
    last = sample[-1]
    if direction == TradeDirection.LONG:
        return (TradeDirection.LONG if last.close <= eq else TradeDirection.FLAT), max(eq, high)
    if direction == TradeDirection.SHORT:
        return (TradeDirection.SHORT if last.close >= eq else TradeDirection.FLAT), min(eq, low)
    return TradeDirection.FLAT, 0.0


def determine_trade_tier(
    daily_bias: TradeDirection,
    h4_bias: TradeDirection,
    confirmations: dict[str, bool],
) -> SetupTier:
    if h4_bias is TradeDirection.FLAT:
        return SetupTier.NONE
    score = sum(1 for value in confirmations.values() if value)
    if daily_bias == h4_bias and score >= 5:
        return SetupTier.A_PLUS
    if daily_bias == h4_bias and score >= 4:
        return SetupTier.A
    if daily_bias is TradeDirection.FLAT and confirmations.get("prime_window") and score >= 5:
        return SetupTier.B
    return SetupTier.NONE


def build_context(
    daily_bars: Sequence[PriceBar],
    h4_bars: Sequence[PriceBar],
    h1_bars: Sequence[PriceBar],
) -> Po3Context:
    daily_bias = compute_daily_bias(daily_bars, lookback=DAILY_LOOKBACK)
    h4_bias, dealing_range = compute_h4_bias(h4_bars, daily_bias)
    h1_bias, target = compute_h1_execution_context(h1_bars, h4_bias)
    return Po3Context(
        daily_bias=daily_bias,
        h4_bias=h4_bias,
        h1_bias=h1_bias,
        dealing_range=dealing_range,
        h1_target=target,
    )
