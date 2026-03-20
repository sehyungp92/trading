"""IB Gateway/TWS connection lifecycle management."""
import asyncio
import logging
import random
from typing import Callable, Optional
from ib_async import IB
from ..config.schemas import IBKRProfile

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages IB Gateway/TWS connection with exponential backoff reconnect."""

    def __init__(self, profile: IBKRProfile):
        self._profile = profile
        self._ib = IB()
        self._connected = asyncio.Event()
        self._shutting_down = False
        self._retry_count = 0
        # H5 fix: callback for post-reconnect reconciliation
        self._on_reconnect_callback: Optional[Callable] = None

    @property
    def ib(self) -> IB:
        return self._ib

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    async def connect(self) -> None:
        """Connect with exponential backoff. Sets _connected event when ready."""
        self._ib.disconnectedEvent += self._on_disconnected
        await self._attempt_connect()

    async def _attempt_connect(self) -> None:
        long_backoff_s = 300.0  # 5 minutes between retries after max attempts
        while not self._shutting_down:
            try:
                await self._ib.connectAsync(
                    host=self._profile.host,
                    port=self._profile.port,
                    clientId=self._profile.client_id,
                    readonly=self._profile.readonly,
                )
                self._retry_count = 0
                self._connected.set()
                logger.info(
                    f"Connected to IB Gateway at {self._profile.host}:{self._profile.port}"
                )
                return
            except Exception as e:
                self._retry_count += 1
                if self._retry_count > self._profile.reconnect_max_retries:
                    logger.error(
                        f"Max reconnect attempts reached: {e}. "
                        f"Entering long-backoff mode (every {long_backoff_s:.0f}s)"
                    )
                    delay = long_backoff_s
                else:
                    delay = self._backoff_delay()
                logger.warning(f"Connection failed: {e}. Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)

    def _backoff_delay(self) -> float:
        delay = self._profile.reconnect_base_delay_s * (2 ** (self._retry_count - 1))
        delay = min(delay, self._profile.reconnect_max_delay_s)
        # M5 fix: add random jitter to prevent thundering herd
        delay = delay * (0.5 + random.random())
        return delay

    def set_reconnect_callback(self, callback: Callable) -> None:
        """H5 fix: Register a callback to invoke after successful reconnection."""
        self._on_reconnect_callback = callback

    async def disconnect(self) -> None:
        """Graceful disconnect."""
        self._shutting_down = True
        if self._ib.isConnected():
            self._ib.disconnect()
        self._connected.clear()

    async def wait_until_ready(self) -> None:
        """Block until connected."""
        await self._connected.wait()

    def _on_disconnected(self) -> None:
        """Callback: IB fires this on socket drop."""
        self._connected.clear()
        if not self._shutting_down:
            logger.warning("Disconnected from IB Gateway, initiating reconnect")
            # M7 fix: wrap ensure_future with proper exception handling
            task = asyncio.ensure_future(self._reconnect_loop())
            task.add_done_callback(self._reconnect_done_callback)

    @staticmethod
    def _reconnect_done_callback(task: asyncio.Task) -> None:
        """M7 fix: Handle unhandled exceptions from reconnect loop."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Reconnect loop terminated unexpectedly: %s. "
                "Broker connection may be unavailable until restart.",
                exc,
                exc_info=exc,
            )

    async def _reconnect_loop(self) -> None:
        """Reconnect with exponential backoff."""
        await self._attempt_connect()
        # Re-assert live market data type after reconnect
        if self._connected.is_set():
            self._ib.reqMarketDataType(1)
        # H5 fix: trigger reconciliation after successful reconnection
        if self._on_reconnect_callback and self._connected.is_set():
            try:
                result = self._on_reconnect_callback()
                if asyncio.iscoroutine(result):
                    await result
                logger.info("Post-reconnect callback completed")
            except Exception as e:
                logger.error(f"Post-reconnect callback failed: {e}")
