"""Unified multi-group IBKR session management."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

from ib_async import IB

from libs.config.models import ConnectionGroupConfig

from .connection import ConnectionManager
from .farm_monitor import FarmMonitor
from .heartbeat import HeartbeatMonitor
from .request_ids import RequestIdAllocator
from .throttler import GlobalThrottler, PacingChannel

logger = logging.getLogger(__name__)


@dataclass
class ConnectionGroup:
    """One IBKR client connection for a group of strategies."""

    group_id: str
    config: ConnectionGroupConfig
    conn: ConnectionManager
    ids: RequestIdAllocator = field(default_factory=RequestIdAllocator)
    heartbeat: HeartbeatMonitor | None = None
    farm_monitor: FarmMonitor | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    farm_recovery_callbacks: list[Callable[[str], None]] = field(default_factory=list)


class UnifiedIBSession:
    """Owns the runtime's set of IBKR connection groups."""

    def __init__(
        self,
        connection_groups: dict[str, ConnectionGroupConfig],
        strategy_group_map: dict[str, str],
    ):
        self._groups = {
            group_id: ConnectionGroup(
                group_id=group_id,
                config=group_config,
                conn=ConnectionManager(group_config),
            )
            for group_id, group_config in connection_groups.items()
        }
        self._strategy_group_map = dict(strategy_group_map)
        first_group = next(iter(connection_groups.values()), None)
        global_rate = first_group.pacing_messages_per_sec if first_group else 45.0
        order_rate = first_group.pacing_orders_per_sec if first_group else 5.0
        self._throttler = GlobalThrottler(
            global_msg_per_sec=global_rate,
            orders_per_sec=order_rate,
        )

    @property
    def groups(self) -> dict[str, ConnectionGroup]:
        return self._groups

    @property
    def is_congested(self) -> bool:
        return self._throttler.is_congested

    async def start(self) -> None:
        """Connect all groups with staggered startup. Cleans up on partial failure."""
        try:
            for group in self._groups.values():
                logger.info(
                    "Connecting group %s to %s:%s client_id=%s",
                    group.group_id,
                    group.config.host,
                    group.config.port,
                    group.config.client_id,
                )
                await group.conn.connect()
                await group.conn.wait_until_ready()
                group.conn.ib.reqMarketDataType(1)
                await group.ids.set_next_valid_id(group.conn.ib.client.getReqId())
                group.heartbeat = HeartbeatMonitor(group.conn.ib)
                await group.heartbeat.start()
                group.farm_monitor = FarmMonitor(group.conn.ib)
                group.farm_monitor.on_farm_recovered = lambda farm, gid=group.group_id: (
                    self._dispatch_farm_recovery(gid, farm)
                )
                group.farm_monitor.start()
                group.ready.set()
                await asyncio.sleep(1.0)
        except Exception:
            logger.error("Partial startup failure — cleaning up already-started groups")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Disconnect all groups in reverse order. Continues on per-group errors."""
        first_error: Exception | None = None
        for group in reversed(list(self._groups.values())):
            try:
                if group.farm_monitor:
                    group.farm_monitor.stop()
                if group.heartbeat:
                    await group.heartbeat.stop()
                await group.conn.disconnect()
                group.ready.clear()
            except Exception as exc:
                logger.error("Error stopping group %s: %s", group.group_id, exc, exc_info=True)
                group.ready.clear()
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _resolve_group(self, strategy_id: str) -> ConnectionGroup:
        """Map strategy_id → ConnectionGroup, raising ValueError on unknown IDs."""
        try:
            group_id = self._strategy_group_map[strategy_id]
        except KeyError:
            raise ValueError(
                f"Unknown strategy_id {strategy_id!r}. "
                f"Registered: {sorted(self._strategy_group_map)}"
            ) from None
        return self._groups[group_id]

    def _get_group(self, group_id: str) -> ConnectionGroup:
        """Resolve group_id → ConnectionGroup, raising ValueError on unknown IDs."""
        try:
            return self._groups[group_id]
        except KeyError:
            raise ValueError(
                f"Unknown group_id {group_id!r}. "
                f"Available: {sorted(self._groups)}"
            ) from None

    async def wait_ready(self, group_id: str | None = None) -> None:
        if group_id is None:
            for group in self._groups.values():
                await group.ready.wait()
            return
        await self._get_group(group_id).ready.wait()

    def get_ib(self, strategy_id: str) -> IB:
        return self._resolve_group(strategy_id).conn.ib

    def strategy_group(self, strategy_id: str) -> str:
        return self._resolve_group(strategy_id).group_id

    async def next_order_id(self, strategy_id: str) -> int:
        return await self._resolve_group(strategy_id).ids.next_order_id()

    async def next_request_id(self, strategy_id: str) -> int:
        return await self._resolve_group(strategy_id).ids.next_request_id()

    async def throttled(self, channel: PacingChannel) -> None:
        """Acquire a pacing token. Raises CongestionError if queue is overloaded."""
        await self._throttler.acquire(channel)

    def register_farm_recovery_callback(self, group_id: str, cb: Callable[[str], None]) -> None:
        self._get_group(group_id).farm_recovery_callbacks.append(cb)

    def set_reconnect_callback(self, group_id: str, callback: Callable) -> None:
        self._get_group(group_id).conn.set_reconnect_callback(callback)

    def _dispatch_farm_recovery(self, group_id: str, farm_name: str) -> None:
        for callback in self._groups[group_id].farm_recovery_callbacks:
            try:
                callback(farm_name)
            except Exception:
                logger.exception("Farm recovery callback failed for %s", group_id)

