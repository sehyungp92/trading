"""IBSession - single orchestration entry point for IB interactions."""
import asyncio
import logging
from ib_async import IB
from ..config.loader import IBKRConfig
from .connection import ConnectionManager
from .request_ids import RequestIdAllocator
from .throttler import Throttler, PacingChannel
from .heartbeat import HeartbeatMonitor

logger = logging.getLogger(__name__)


class IBSession:
    """Single entry point for all IB interactions.

    Must be started via `await session.start()` and
    gated via `await session.wait_ready()`.
    """

    def __init__(self, config: IBKRConfig):
        self._config = config
        self._conn = ConnectionManager(config.profile)
        self._ids = RequestIdAllocator()
        self._throttler = Throttler(
            config.profile.pacing_orders_per_sec,
            config.profile.pacing_messages_per_sec,
        )
        self._heartbeat: HeartbeatMonitor | None = None
        self._ready = asyncio.Event()

    @property
    def ib(self) -> IB:
        return self._conn.ib

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def config(self) -> IBKRConfig:
        return self._config

    async def start(self) -> None:
        """Connect, receive nextValidId, start heartbeat, mark ready."""
        await self._conn.connect()
        await self._conn.wait_until_ready()
        # Request live (type 1) market data — without this, paper accounts
        # may default to delayed/frozen and reject streaming subscriptions.
        self._conn.ib.reqMarketDataType(1)
        # Seed order ID from IB
        self._ids.set_next_valid_id(self._conn.ib.client.getReqId())
        self._heartbeat = HeartbeatMonitor(self._conn.ib)
        await self._heartbeat.start()
        self._ready.set()
        logger.info("IBSession ready (marketDataType=1)")

    async def stop(self) -> None:
        if self._heartbeat:
            await self._heartbeat.stop()
        await self._conn.disconnect()
        self._ready.clear()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    def next_order_id(self) -> int:
        return self._ids.next_order_id()

    def next_request_id(self) -> int:
        return self._ids.next_request_id()

    async def throttled(self, channel: PacingChannel) -> None:
        """Await pacing token before sending."""
        await self._throttler.acquire(channel)

    def set_reconnect_callback(self, callback) -> None:
        """Register a callback invoked after successful IB reconnection."""
        self._conn.set_reconnect_callback(callback)

    @property
    def is_congested(self) -> bool:
        return self._throttler.is_congested
