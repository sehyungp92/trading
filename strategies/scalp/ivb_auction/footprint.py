from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import FootprintBarData, ScalpTick


@dataclass(slots=True)
class _WindowAccum:
    start_ts: datetime
    end_ts: datetime
    ask_volume: float = 0.0
    bid_volume: float = 0.0
    high: float = 0.0
    low: float = 0.0
    large_trade_count: int = 0


class FootprintBuilder:
    def __init__(self, *, bar_seconds: int = 30, tick_size: float = 0.25) -> None:
        self.bar_seconds = int(bar_seconds)
        self.tick_size = float(tick_size)
        self.cumulative_delta = 0.0
        self._current: _WindowAccum | None = None
        self._prev_trade_price: float | None = None
        self._sizes: list[float] = []

    def on_tick(self, tick: ScalpTick) -> FootprintBarData | None:
        completed = self._roll_if_needed(tick.ts)
        if self._current is None:
            self._current = self._new_window(tick.ts)
        side = tick.side or self._classify_tick(tick)
        if side > 0:
            self._current.ask_volume += tick.size
        elif side < 0:
            self._current.bid_volume += tick.size
        self._current.high = tick.price if self._current.high == 0 else max(self._current.high, tick.price)
        self._current.low = tick.price if self._current.low == 0 else min(self._current.low, tick.price)
        self._sizes.append(tick.size)
        if len(self._sizes) >= 10:
            threshold = sorted(self._sizes)[int(0.9 * (len(self._sizes) - 1))]
            if tick.size >= threshold:
                self._current.large_trade_count += 1
        self._prev_trade_price = tick.price
        return completed

    def flush(self) -> FootprintBarData | None:
        if self._current is None:
            return None
        current = self._current
        self._current = None
        return self._to_bar(current)

    def _roll_if_needed(self, ts: datetime) -> FootprintBarData | None:
        if self._current is None:
            return None
        if ts < self._current.end_ts:
            return None
        completed = self._to_bar(self._current)
        while ts >= self._current.end_ts:
            self._current = self._new_window(self._current.end_ts)
        return completed

    def _classify_tick(self, tick: ScalpTick) -> int:
        if tick.ask is not None and tick.price >= tick.ask:
            return 1
        if tick.bid is not None and tick.price <= tick.bid:
            return -1
        if self._prev_trade_price is None:
            return 0
        if tick.price > self._prev_trade_price:
            return 1
        if tick.price < self._prev_trade_price:
            return -1
        return 0

    def _new_window(self, ts: datetime) -> _WindowAccum:
        epoch = int(ts.timestamp())
        start_epoch = epoch - (epoch % self.bar_seconds)
        start = datetime.fromtimestamp(start_epoch, tz=ts.tzinfo)
        return _WindowAccum(start_ts=start, end_ts=start + timedelta(seconds=self.bar_seconds))

    def _to_bar(self, window: _WindowAccum) -> FootprintBarData:
        ask = window.ask_volume
        bid = window.bid_volume
        delta = ask - bid
        self.cumulative_delta += delta
        smaller = max(1.0, min(ask, bid))
        larger = max(ask, bid)
        price_progress = max(self.tick_size, window.high - window.low)
        absorption_score = larger / price_progress if price_progress > 0 else 0.0
        return FootprintBarData(
            start_ts=window.start_ts,
            end_ts=window.end_ts,
            ask_volume=ask,
            bid_volume=bid,
            delta=delta,
            cumulative_delta=self.cumulative_delta,
            aggression_ratio=larger / smaller,
            imbalance_ratio=ask / max(1.0, bid),
            absorption_score=absorption_score,
            large_trade_count=window.large_trade_count,
        )

