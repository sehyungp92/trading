from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from strategies.momentum.downturn.engine import DownturnEngine
from strategies.momentum.helix_v40.config import TF as HELIX4_TF
from strategies.momentum.helix_v40.engine import Helix4Engine
from strategies.momentum.nqdtc.engine import NQDTCEngine
from strategies.momentum.vdub.engine import VdubNQv4Engine
from strategies.swing.brs.engine import BRSLiveEngine
from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.atrss.engine import ATRSSEngine
from strategies.swing.breakout.engine import BreakoutEngine

UTC = timezone.utc


def _bars() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            date=datetime(2024, 1, 5, 14, 0, tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        ),
        SimpleNamespace(
            date=datetime(2024, 1, 5, 15, 0, tzinfo=UTC),
            open=100.5,
            high=101.5,
            low=100.0,
            close=101.0,
            volume=12.0,
        ),
    ]


class _FakeIBClient:
    def isConnected(self) -> bool:
        return True

    async def qualifyContractsAsync(self, contract: object) -> list[object]:
        return [contract]


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.ib = _FakeIBClient()
        self.is_connected = True

    async def req_historical_data(self, contract: object, **kwargs):
        self.calls.append(kwargs)
        return _bars()


class _FakeEventHook:
    def __init__(self) -> None:
        self.callbacks = []

    def __iadd__(self, callback):
        self.callbacks.append(callback)
        return self


class _FakeBars(list):
    def __init__(self, bars) -> None:
        super().__init__(bars)
        self.updateEvent = _FakeEventHook()


class _FakeStreamingSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self._contract_factory = SimpleNamespace(resolve=self._resolve)
        self.ib = SimpleNamespace(
            cancelHistoricalData=lambda _bars: None,
            reqMktData=lambda *_args, **_kwargs: SimpleNamespace(updateEvent=_FakeEventHook()),
        )

    async def _resolve(self, symbol: str, *args, **kwargs):
        return SimpleNamespace(symbol=symbol), object()

    async def req_historical_data(self, contract: object, **kwargs):
        self.calls.append(kwargs)
        return _FakeBars(_bars())


@pytest.mark.asyncio
async def test_nqdtc_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = NQDTCEngine.__new__(NQDTCEngine)
    engine._ib = session

    await engine._req_completed_bars(object(), "2 D", "5 mins", request_kind="recurring")

    assert session.calls[-1]["completed_only"] is True


@pytest.mark.asyncio
async def test_downturn_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = DownturnEngine.__new__(DownturnEngine)
    engine._ib = session
    engine._symbol = "MNQ"
    engine._get_contract = lambda: object()

    await engine._fetch_bars()

    assert session.calls
    assert all(call["completed_only"] is True for call in session.calls)


@pytest.mark.asyncio
async def test_vdub_helper_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = VdubNQv4Engine.__new__(VdubNQv4Engine)
    engine._ib = session

    await engine._req_bars(object(), "30 D", "15 mins", request_kind="recurring")

    assert session.calls[-1]["completed_only"] is True


@pytest.mark.asyncio
async def test_atrss_daily_and_hourly_fetches_request_completed_only_bars() -> None:
    session = _FakeSession()
    engine = ATRSSEngine.__new__(ATRSSEngine)
    engine._ib = session
    engine._get_contract = lambda sym: object()

    await engine._fetch_daily_bars("SPY", None)
    await engine._fetch_hourly_bars("SPY", None)

    assert [call["barSizeSetting"] for call in session.calls] == ["1 day", "1 hour"]
    assert all(call["completed_only"] is True for call in session.calls)


@pytest.mark.asyncio
async def test_akc_helix_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = HelixEngine.__new__(HelixEngine)
    engine._ib = session
    engine._get_contract = lambda sym: object()

    await engine._fetch_bars("ES", None, "1 hour", "30 D")

    assert session.calls[-1]["completed_only"] is True


@pytest.mark.asyncio
async def test_breakout_helper_fetches_completed_only_bars() -> None:
    session = _FakeSession()
    engine = BreakoutEngine.__new__(BreakoutEngine)
    engine._ib = session

    await engine._req_completed_bars(
        object(),
        duration="2 Y",
        bar_size="1 hour",
        use_rth=True,
        request_kind="startup",
    )

    assert session.calls[-1]["completed_only"] is True


@pytest.mark.asyncio
async def test_breakout_hourly_fetch_routes_through_completed_bar_helper() -> None:
    session = _FakeSession()
    engine = BreakoutEngine.__new__(BreakoutEngine)
    engine._ib = session
    engine.contracts = {"SPY": object()}

    await engine._fetch_hourly_bars("SPY")

    assert session.calls[-1]["completed_only"] is True
    assert session.calls[-1]["barSizeSetting"] == "1 hour"


@pytest.mark.asyncio
async def test_brs_safe_path_keeps_streaming_subscriptions_and_drops_developing_backfill_bar() -> None:
    session = _FakeStreamingSession()
    engine = BRSLiveEngine(
        ib_session=session,
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: asyncio.Queue()),
        instruments=[SimpleNamespace(symbol="QQQ"), SimpleNamespace(symbol="GLD")],
    )

    async def _wait_forever(_queue) -> None:
        await asyncio.sleep(3600)

    engine._process_events = _wait_forever  # type: ignore[method-assign]
    await engine.start()
    await engine.stop()

    assert session.calls
    assert all(call["keepUpToDate"] is True for call in session.calls)
    assert engine._bar_processed_count[("QQQ", "D1")] == len(_bars()) - 1
    assert engine._bar_processed_count[("GLD", "H1")] == len(_bars()) - 1


@pytest.mark.asyncio
async def test_helix_v40_safe_path_uses_streaming_updates_and_completed_bar_gate() -> None:
    session = _FakeStreamingSession()
    engine = Helix4Engine(
        ib_session=session,
        oms_service=object(),
        instruments=[SimpleNamespace(symbol="MNQ", point_value=2.0)],
        instrumentation=None,
    )
    engine._contract = object()
    engine._running = True

    await engine._subscribe_and_backfill()

    created: list[object] = []
    original_create_task = asyncio.create_task

    def _capture(task):
        created.append(task)
        return original_create_task(task)

    try:
        asyncio.create_task = _capture  # type: ignore[assignment]
        engine._on_bar_update(_FakeBars(_bars()), False, HELIX4_TF.H1)
    finally:
        asyncio.create_task = original_create_task  # type: ignore[assignment]

    assert all(call["keepUpToDate"] is True for call in session.calls)
    assert created == []
