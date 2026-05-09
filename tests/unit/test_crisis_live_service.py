from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from regime.crisis import config as C
from regime.crisis.service import CrisisService, _next_crisis_compute_after_close


ET = ZoneInfo("America/New_York")


def test_next_crisis_compute_runs_after_completed_daily_bars() -> None:
    before_cutoff = datetime(2026, 5, 4, 16, 59, tzinfo=ET)
    after_cutoff = datetime(2026, 5, 4, 17, 10, tzinfo=ET)
    friday_after_cutoff = datetime(2026, 5, 8, 17, 10, tzinfo=ET)

    assert _next_crisis_compute_after_close(before_cutoff).isoformat() == (
        "2026-05-04T17:05:00-04:00"
    )
    assert _next_crisis_compute_after_close(after_cutoff).isoformat() == (
        "2026-05-05T17:05:00-04:00"
    )
    assert _next_crisis_compute_after_close(friday_after_cutoff).isoformat() == (
        "2026-05-11T17:05:00-04:00"
    )


@pytest.mark.asyncio
async def test_crisis_fetches_completed_daily_spy_tlt_bars(tmp_path) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def req_historical_data(self, contract: object, **kwargs):
            self.calls.append(kwargs)
            start = datetime(2026, 4, 1, tzinfo=timezone.utc)
            return [
                SimpleNamespace(
                    date=start + timedelta(days=i),
                    close=100.0 + i,
                )
                for i in range(30)
            ]

    session = FakeSession()
    service = CrisisService(
        session,
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 5, 6, 21, 10, tzinfo=timezone.utc),
    )
    service._contracts = {"SPY": object(), "TLT": object()}

    returns = await service._fetch_etf_returns()

    assert not returns.empty
    assert {call["barSizeSetting"] for call in session.calls} == {"1 day"}
    assert all(call["useRTH"] is True for call in session.calls)
    assert all(call["completed_only"] is True for call in session.calls)


@pytest.mark.asyncio
async def test_crisis_compute_publishes_fresh_context_to_listeners(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-01-21", periods=100, freq="D")
    market_df = pd.DataFrame(
        {
            "VIX": [20.0] * 100,
            "SPREAD": [2.0] * 100,
            "SLOPE_10Y2Y": [1.0] * 100,
        },
        index=dates,
    )
    strat_ret_df = pd.DataFrame(
        {
            "SPY": [0.0] * 100,
            "TLT": [0.0] * 100,
        },
        index=dates,
    )

    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 30, 21, 10, tzinfo=timezone.utc),
    )

    async def fake_fetch_data():
        return market_df, strat_ret_df, None, "unit=fresh"

    monkeypatch.setattr(service, "_fetch_data", fake_fetch_data)
    delivered = []
    service.add_listener(lambda ctx: delivered.append(ctx))

    ctx = await service.compute_now()

    assert ctx.alert_level == C.ALERT_NORMAL
    assert ctx.data_as_of == "2026-04-30"
    assert ctx.data_status == "unit=fresh;backlog=ok"
    assert delivered == [ctx]


@pytest.mark.asyncio
async def test_crisis_compute_uses_latest_completed_etf_date_not_fred_only_date(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-01-21", periods=100, freq="D")
    market_df = pd.DataFrame(
        {
            "VIX": [20.0] * 100,
            "SPREAD": [2.0] * 100,
            "SLOPE_10Y2Y": [1.0] * 100,
        },
        index=dates,
    )
    strat_ret_df = pd.DataFrame(
        {
            "SPY": [0.0] * 97 + [-0.01, -0.01, -0.01],
            "TLT": [0.0] * 100,
        },
        index=dates,
    )
    market_df.to_parquet(tmp_path / "market_df.parquet")
    strat_ret_df.to_parquet(tmp_path / "strat_ret_df.parquet")

    fred_only_next_day = pd.DataFrame(
        {
            "VIX": [21.0],
            "SPREAD": [2.1],
            "SLOPE_10Y2Y": [1.0],
        },
        index=[dates[-1] + pd.Timedelta(days=1)],
    )
    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 30, 21, 10, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(service, "_fetch_fred", lambda: fred_only_next_day)

    async def no_fresh_returns():
        return pd.DataFrame()

    monkeypatch.setattr(service, "_fetch_etf_returns", no_fresh_returns)

    ctx = await service.compute_now(notify=False)

    assert ctx.data_as_of == dates[-1].date().isoformat()
    assert ctx.spy_3d_return == pytest.approx((0.99 ** 3) - 1.0)


@pytest.mark.asyncio
async def test_crisis_compute_keeps_prior_context_when_spy_tlt_backlog_is_incomplete(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-04-01", periods=10, freq="D")
    market_df = pd.DataFrame(
        {
            "VIX": [20.0] * 10,
            "SPREAD": [2.0] * 10,
            "SLOPE_10Y2Y": [1.0] * 10,
        },
        index=dates,
    )
    strat_ret_df = pd.DataFrame(
        {
            "SPY": [0.0] * 10,
            "TLT": [0.0] * 10,
        },
        index=dates,
    )
    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 10, 21, 10, tzinfo=timezone.utc),
    )

    async def fake_fetch_data():
        return market_df, strat_ret_df, None, "unit=short"

    monkeypatch.setattr(service, "_fetch_data", fake_fetch_data)

    ctx = await service.compute_now(notify=False)

    assert ctx.alert_level == C.ALERT_NORMAL
    assert ctx.computed_at == ""


@pytest.mark.asyncio
async def test_crisis_startup_fails_loudly_when_live_backlog_is_incomplete(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-04-01", periods=10, freq="D")
    market_df = pd.DataFrame(
        {
            "VIX": [20.0] * 10,
            "SPREAD": [2.0] * 10,
            "SLOPE_10Y2Y": [1.0] * 10,
        },
        index=dates,
    )
    strat_ret_df = pd.DataFrame(
        {
            "SPY": [0.0] * 10,
            "TLT": [0.0] * 10,
        },
        index=dates,
    )
    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=True,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 10, 21, 10, tzinfo=timezone.utc),
    )

    async def fake_fetch_data():
        return market_df, strat_ret_df, None, "unit=short"

    async def noop_qualify():
        return None

    monkeypatch.setattr(service, "_fetch_data", fake_fetch_data)
    monkeypatch.setattr(service, "_qualify_contracts", noop_qualify)

    with pytest.raises(RuntimeError, match="insufficient live backlog"):
        await service.start()
