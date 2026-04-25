from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from apps.watchdog.checks import check_heartbeats


def _row(strategy_id: str, age: int, status: str = "OK") -> dict:
    return {
        "strategy_id": strategy_id,
        "heartbeat_age_sec": age,
        "health_status": status,
    }


@pytest.mark.asyncio
async def test_single_stale_heartbeat_keeps_individual_alert() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [_row("s1", 181, "STALE"), _row("s2", 5)]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"family"},
        {"s1": "family", "s2": "family"},
    )

    problems = [r for r in results if r.is_problem]
    assert [r.key for r in problems] == ["heartbeat:s1"]


@pytest.mark.asyncio
async def test_three_stale_heartbeats_are_aggregated() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [
        _row("s1", 181, "STALE"),
        _row("s2", 220, "STALE"),
        _row("s3", 260, "STALE"),
    ]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"family"},
        {"s1": "family", "s2": "family", "s3": "family"},
    )

    problems = [r for r in results if r.is_problem]
    assert [r.key for r in problems] == ["heartbeat:systemic"]
    assert "3 strategies stale" in problems[0].detail
    assert "s1, s2, s3" in problems[0].detail


@pytest.mark.asyncio
async def test_inactive_family_heartbeat_is_skipped() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [_row("s1", 300, "STALE")]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"other"},
        {"s1": "family"},
    )

    assert not [r for r in results if r.is_problem]


@pytest.mark.asyncio
async def test_unknown_strategy_heartbeat_is_skipped() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [_row("US_ORB_v1", 300, "STALE")]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"stock"},
        {"IARIC_v1": "stock", "ALCB_v1": "stock"},
    )

    assert not [r for r in results if r.is_problem]
