"""Unit tests for libs.broker_ibkr.heartbeat — HeartbeatMonitor."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_monitor(interval: float = 0.05, timeout: float = 0.05):
    """Create a HeartbeatMonitor with mocked IB and short timings."""
    with patch("libs.broker_ibkr.heartbeat.IB"):
        from libs.broker_ibkr.heartbeat import HeartbeatMonitor

    mock_ib = MagicMock()
    monitor = HeartbeatMonitor(mock_ib, interval_s=interval, timeout_s=timeout)
    return monitor


# ---------------------------------------------------------------------------
# Status transitions: alive -> dead -> alive
# ---------------------------------------------------------------------------
class TestStatusTransitions:
    """Verify the monitor fires on_status_change at the right moments."""

    @pytest.mark.asyncio
    async def test_alive_to_dead_to_alive(self) -> None:
        """Ping fails (dead), then succeeds (alive) — callback should fire twice."""
        monitor = _make_monitor()

        statuses: list[bool] = []
        monitor.on_status_change = lambda alive: statuses.append(alive)

        call_count = 0

        async def flaky_ping() -> int:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two pings timeout
                await asyncio.sleep(10)
            return 1  # subsequent pings succeed

        monitor._ping = flaky_ping  # type: ignore[assignment]

        await monitor.start()
        # Give enough time for ~5 loop iterations (interval=0.05s)
        await asyncio.sleep(0.6)
        await monitor.stop()

        # Should have gone dead then alive: [False, True]
        assert len(statuses) >= 2, f"Expected at least 2 transitions, got {statuses}"
        assert statuses[0] is False, f"First transition should be dead, got {statuses}"
        assert statuses[1] is True, f"Second transition should be alive, got {statuses}"

    @pytest.mark.asyncio
    async def test_stays_alive_on_successful_pings(self) -> None:
        """If all pings succeed, on_status_change should never fire (stays alive)."""
        monitor = _make_monitor()

        statuses: list[bool] = []
        monitor.on_status_change = lambda alive: statuses.append(alive)

        # _ping always succeeds immediately
        monitor._ping = AsyncMock(return_value=1)  # type: ignore[assignment]

        await monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()

        # Monitor starts alive, never transitions
        assert statuses == [], f"Expected no transitions, got {statuses}"

    @pytest.mark.asyncio
    async def test_stays_dead_on_repeated_failures(self) -> None:
        """Multiple consecutive failures should only fire dead once."""
        monitor = _make_monitor()

        statuses: list[bool] = []
        monitor.on_status_change = lambda alive: statuses.append(alive)

        async def always_timeout() -> int:
            await asyncio.sleep(10)
            return 1

        monitor._ping = always_timeout  # type: ignore[assignment]

        await monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()

        # Should only report dead once
        assert statuses == [False], f"Expected [False], got {statuses}"


# ---------------------------------------------------------------------------
# Async callback is properly awaited
# ---------------------------------------------------------------------------
class TestAsyncCallback:
    """Verify that async on_status_change callbacks get awaited."""

    @pytest.mark.asyncio
    async def test_async_callback_is_awaited(self) -> None:
        """An async on_status_change must be awaited, not just called."""
        monitor = _make_monitor()

        callback_called = asyncio.Event()

        async def async_callback(alive: bool) -> None:
            callback_called.set()

        monitor.on_status_change = async_callback  # type: ignore[assignment]

        # Make ping timeout so _invoke_status_change fires
        async def timeout_ping() -> int:
            await asyncio.sleep(10)
            return 1

        monitor._ping = timeout_ping  # type: ignore[assignment]

        await monitor.start()
        # Wait for the callback to be triggered
        try:
            await asyncio.wait_for(callback_called.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Async callback was never awaited within 1 second")
        finally:
            await monitor.stop()

        assert callback_called.is_set()

    @pytest.mark.asyncio
    async def test_sync_callback_still_works(self) -> None:
        """A regular sync callback should also work without errors."""
        monitor = _make_monitor()

        called_with: list[bool] = []

        def sync_callback(alive: bool) -> None:
            called_with.append(alive)

        monitor.on_status_change = sync_callback

        async def timeout_ping() -> int:
            await asyncio.sleep(10)
            return 1

        monitor._ping = timeout_ping  # type: ignore[assignment]

        await monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()

        assert False in called_with, f"Expected False in {called_with}"
