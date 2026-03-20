"""IBSession - single orchestration entry point for IB interactions."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from ib_async import IB
from ..config.loader import IBKRConfig
from .connection import ConnectionManager
from .farm_monitor import FarmMonitor
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
        self._farm_monitor: FarmMonitor | None = None
        self._farm_recovery_callbacks: list[Callable[[str], None]] = []
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
        # Request live (streaming) market data — must be set before any reqMktData calls
        self._conn.ib.reqMarketDataType(1)
        # Seed order ID from IB
        await self._ids.set_next_valid_id(self._conn.ib.client.getReqId())
        self._heartbeat = HeartbeatMonitor(self._conn.ib)
        await self._heartbeat.start()
        self._farm_monitor = FarmMonitor(self._conn.ib)
        self._farm_monitor.on_farm_recovered = self._dispatch_farm_recovery
        self._farm_monitor.start()
        self._ready.set()
        logger.info("IBSession ready")

    async def stop(self) -> None:
        if self._farm_monitor:
            self._farm_monitor.stop()
        if self._heartbeat:
            await self._heartbeat.stop()
        await self._conn.disconnect()
        self._ready.clear()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    async def next_order_id(self) -> int:
        return await self._ids.next_order_id()

    async def next_request_id(self) -> int:
        return await self._ids.next_request_id()

    async def throttled(self, channel: PacingChannel) -> None:
        """Await pacing token before sending."""
        await self._throttler.acquire(channel)

    def register_farm_recovery_callback(self, cb: Callable[[str], None]) -> None:
        """Register a callback to be invoked when a data farm recovers."""
        self._farm_recovery_callbacks.append(cb)

    def _dispatch_farm_recovery(self, farm_name: str) -> None:
        for cb in self._farm_recovery_callbacks:
            try:
                cb(farm_name)
            except Exception:
                logger.exception("Farm recovery callback failed")

    def set_reconnect_callback(self, callback: Callable) -> None:
        """Register a callback to invoke after successful reconnection."""
        self._conn.set_reconnect_callback(callback)

    @property
    def is_congested(self) -> bool:
        return self._throttler.is_congested
