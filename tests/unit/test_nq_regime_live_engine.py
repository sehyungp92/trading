from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from strategies.momentum.nq_regime.engine import NQRegimeEngine


class _IB:
    @staticmethod
    def isConnected() -> bool:
        return True

    async def qualifyContractsAsync(self, contract):
        return [contract]


class _Session:
    def __init__(self) -> None:
        self.ib = _IB()
        self.requests: list[str] = []

    async def req_historical_data(self, contract, **kwargs):
        bar_size = kwargs["barSizeSetting"]
        self.requests.append(bar_size)
        if bar_size == "1 day":
            return [
                {
                    "timestamp": datetime(2026, 5, 6, tzinfo=timezone.utc),
                    "open": 100,
                    "high": 110,
                    "low": 95,
                    "close": 105,
                },
                {
                    "timestamp": datetime(2026, 5, 7, tzinfo=timezone.utc),
                    "open": 105,
                    "high": 120,
                    "low": 101,
                    "close": 118,
                },
            ]
        return [
            {
                "timestamp": datetime(2026, 5, 8, 14, 35, tzinfo=timezone.utc),
                "open": 118,
                "high": 121,
                "low": 117,
                "close": 120,
            }
        ]


@pytest.mark.asyncio
async def test_nq_regime_scheduled_bar_uses_cached_daily_levels():
    session = _Session()
    engine = NQRegimeEngine(ib_session=session)
    engine._get_analysis_contract = lambda: object()
    engine.on_bar = AsyncMock()

    bar_ts = datetime(2026, 5, 8, 14, 35, tzinfo=timezone.utc)
    await engine._refresh_daily_context(force=True, for_ts=bar_ts)
    await engine._fetch_and_emit_bar(request_kind="test")

    engine.on_bar.assert_awaited_once()
    levels = engine.on_bar.await_args.kwargs["daily_context"]
    assert levels.pdh == 120
    assert levels.pdl == 101
    assert levels.pdm == 110.5
    assert levels.weekly_high == 120
    assert levels.weekly_low == 95
    assert session.requests == ["1 day", "5 mins"]


@pytest.mark.asyncio
async def test_nq_regime_scheduled_bar_refreshes_daily_levels_on_session_change():
    session = _Session()
    engine = NQRegimeEngine(ib_session=session)
    engine._get_analysis_contract = lambda: object()
    engine.on_bar = AsyncMock()

    await engine._fetch_and_emit_bar(request_kind="test")

    engine.on_bar.assert_awaited_once()
    assert engine.on_bar.await_args.kwargs["daily_context"] is not None
    assert session.requests == ["5 mins", "1 day"]


@pytest.mark.asyncio
async def test_nq_regime_on_bar_persists_after_successful_mutation(monkeypatch, tmp_path):
    import strategies.momentum.nq_regime.engine as engine_mod
    from strategies.momentum.nq_regime.core.levels import KeyLevels

    engine = NQRegimeEngine(state_dir=tmp_path)
    persisted = {"count": 0}
    engine._persist_state = lambda: persisted.__setitem__("count", persisted["count"] + 1)
    monkeypatch.setattr(
        engine_mod.CompletedBarPolicy,
        "build_event",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        engine_mod,
        "core_on_bar",
        lambda state, event, scheduled_news, settings: (state, [], []),
    )

    await engine.on_bar(
        {
            "timestamp": datetime(2026, 5, 8, 14, 35, tzinfo=timezone.utc),
            "open": 118,
            "high": 121,
            "low": 117,
            "close": 120,
        },
        daily_context=KeyLevels(pdh=120, pdl=101, pdm=110.5, weekly_high=120, weekly_low=95),
    )

    assert persisted["count"] == 1


def test_nq_regime_state_persist_uses_atomic_replace(tmp_path):
    engine = NQRegimeEngine(state_dir=tmp_path)

    engine._persist_state()

    snap_path = tmp_path / "NQ_REGIME.json"
    assert snap_path.exists()
    assert not (tmp_path / "NQ_REGIME.json.tmp").exists()
