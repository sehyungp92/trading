"""Connection liveness monitoring."""
import asyncio
import time
import logging
from typing import Callable
from ib_async import IB

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """Monitors IB connection liveness via lightweight requests."""

    def __init__(self, ib: IB, interval_s: float = 10.0, timeout_s: float = 5.0):
        self._ib = ib
        self._interval = interval_s
        self._timeout = timeout_s
        self._last_heartbeat: float = 0.0
        self._alive = True
        self._task: asyncio.Task | None = None
        self.on_status_change: Callable[[bool], None] = lambda alive: None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                t = await asyncio.wait_for(
                    self._ping(),
                    timeout=self._timeout,
                )
                self._last_heartbeat = time.monotonic()
                if not self._alive:
                    self._alive = True
                    self.on_status_change(True)
            except (asyncio.TimeoutError, Exception) as e:
                if self._alive:
                    logger.warning(f"Heartbeat failed: {e}")
                    self._alive = False
                    self.on_status_change(False)

    async def _ping(self) -> int:
        """Use async request for actual network round-trip."""
        return await self._ib.reqCurrentTimeAsync()

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def last_heartbeat(self) -> float:
        return self._last_heartbeat
