"""Token-bucket rate limiter for IB pacing."""
import asyncio
import time
from enum import Enum


class PacingChannel(Enum):
    ORDERS = "orders"
    MARKET_DATA = "market_data"
    HISTORICAL = "historical"
    GENERAL = "general"


class CongestionError(Exception):
    """Raised when queue exceeds congestion threshold."""

    pass


class Throttler:
    """Token-bucket rate limiter per channel."""

    def __init__(
        self, orders_per_sec: float = 5.0, messages_per_sec: float = 50.0
    ):
        self._buckets: dict[PacingChannel, _TokenBucket] = {
            PacingChannel.ORDERS: _TokenBucket(orders_per_sec),
            PacingChannel.MARKET_DATA: _TokenBucket(messages_per_sec),
            PacingChannel.HISTORICAL: _TokenBucket(1.0),  # IB: ~1 hist req/sec
            PacingChannel.GENERAL: _TokenBucket(messages_per_sec),
        }
        self._queue_depth: dict[PacingChannel, int] = {c: 0 for c in PacingChannel}
        self.congestion_threshold = 20

    async def acquire(self, channel: PacingChannel) -> None:
        """Wait until a token is available. Raises CongestionError if congested."""
        self._queue_depth[channel] += 1
        try:
            if self._queue_depth[channel] > self.congestion_threshold:
                raise CongestionError(f"Channel {channel.value} congested")
            await self._buckets[channel].acquire()
        finally:
            self._queue_depth[channel] -= 1

    @property
    def is_congested(self) -> bool:
        return any(d > self.congestion_threshold for d in self._queue_depth.values())


class _TokenBucket:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        # M6 fix: compute wait time under lock, then sleep outside lock
        # to avoid blocking all other coroutines during the sleep
        wait = 0.0
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                self.tokens = 0.0
            else:
                self.tokens -= 1.0
        if wait > 0:
            await asyncio.sleep(wait)
