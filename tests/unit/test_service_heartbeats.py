from __future__ import annotations

import asyncio

import pytest

from libs.services.heartbeat import emit_family_heartbeats


class _FakeHeartbeat:
    def __init__(self) -> None:
        self.strategy_calls: list[str] = []
        self.adapter_calls: list[tuple[str, bool]] = []
        self.slow_strategy_ids: set[str] = set()
        self.slow_adapter = False

    async def strategy_heartbeat(self, **payload) -> None:
        sid = payload["strategy_id"]
        self.strategy_calls.append(sid)
        if sid in self.slow_strategy_ids:
            await asyncio.sleep(0.05)

    async def adapter_heartbeat(self, adapter_id: str, connected: bool, broker: str = "IBKR") -> None:
        self.adapter_calls.append((adapter_id, connected))
        if self.slow_adapter:
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_family_heartbeats_do_not_block_on_slow_strategy() -> None:
    heartbeat = _FakeHeartbeat()
    heartbeat.slow_strategy_ids.add("slow")

    await emit_family_heartbeats(
        heartbeat,
        "family",
        [{"strategy_id": "slow"}, {"strategy_id": "fast"}],
        adapter_connected=True,
        timeout_s=0.01,
    )

    assert set(heartbeat.strategy_calls) == {"slow", "fast"}
    assert heartbeat.adapter_calls == [("family", True)]


@pytest.mark.asyncio
async def test_family_heartbeats_adapter_timeout_is_isolated() -> None:
    heartbeat = _FakeHeartbeat()
    heartbeat.slow_adapter = True

    await emit_family_heartbeats(
        heartbeat,
        "family",
        [{"strategy_id": "fast"}],
        adapter_connected=False,
        timeout_s=0.01,
    )

    assert heartbeat.strategy_calls == ["fast"]
    assert heartbeat.adapter_calls == [("family", False)]
