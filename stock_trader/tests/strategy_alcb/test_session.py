"""Session/bootstrap tests for the ALCB runtime."""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy_alcb.config import ET, StrategySettings
from strategy_alcb.engine import ALCBEngine
from strategy_alcb.models import CandidateArtifact, RegimeSnapshot
import strategy_alcb.main as alcb_main
import strategy_alcb.session as alcb_session


class _FakeIBSession:
    def __init__(self, config):
        self.config = config
        self.ib = object()

    async def start(self):
        return None

    async def wait_ready(self):
        return None

    async def stop(self):
        return None


class _FakeOMS:
    def __init__(self):
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


class _FakeInstrumentationManager:
    def __init__(self, oms, strategy_id, strategy_type):
        self.oms = oms
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type

    async def stop(self):
        return None


class _FakeInstrumentationKit:
    def __init__(self, instrumentation, strategy_type):
        self.instrumentation = instrumentation
        self.strategy_type = strategy_type


class _FakeSummaryItem:
    def __init__(self, tag: str, currency: str, value: str) -> None:
        self.tag = tag
        self.currency = currency
        self.value = value


class _FetchNavSession:
    def __init__(self, accounts=None, summary=None, summary_error: Exception | None = None):
        self.ib = SimpleNamespace(
            managedAccounts=lambda: list(accounts or []),
            accountSummaryAsync=self._account_summary,
        )
        self._summary = list(summary or [])
        self._summary_error = summary_error

    async def _account_summary(self, _account: str):
        if self._summary_error is not None:
            raise self._summary_error
        return list(self._summary)


def _artifact() -> CandidateArtifact:
    return CandidateArtifact(
        trade_date=datetime(2026, 3, 12, tzinfo=ET).date(),
        generated_at=datetime(2026, 3, 12, 8, 0, tzinfo=ET).astimezone(timezone.utc),
        regime=RegimeSnapshot(0.8, "A", 1.0, True, True, True, True, "BULL"),
        items=[],
        tradable=[],
        overflow=[],
    )


def _install_runtime_patches(monkeypatch, captured: dict, *, nav: float):
    monkeypatch.setattr(
        alcb_session,
        "IBKRConfig",
        lambda config_dir: SimpleNamespace(profile=SimpleNamespace(account_id="DU1"), contracts={}, routes={}),
    )
    monkeypatch.setattr(alcb_session, "IBSession", _FakeIBSession)
    monkeypatch.setattr(alcb_session, "ContractFactory", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(alcb_session, "IBKRExecutionAdapter", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        alcb_session,
        "bootstrap_database",
        lambda: asyncio.sleep(0, result=SimpleNamespace(pool=None, trade_recorder=None, has_db=False)),
    )
    monkeypatch.setattr(alcb_session, "fetch_nav", lambda session, default_nav=100_000.0: asyncio.sleep(0, result=nav))
    monkeypatch.setattr(alcb_session, "InstrumentationManager", _FakeInstrumentationManager)
    monkeypatch.setattr(alcb_session, "InstrumentationKit", _FakeInstrumentationKit)

    async def _build_oms_service(**kwargs):
        captured.update(kwargs)
        return _FakeOMS()

    monkeypatch.setattr(alcb_session, "build_oms_service", _build_oms_service)


def test_bootstrap_runtime_uses_allocated_nav_in_combined_mode(monkeypatch):
    captured: dict = {}
    _install_runtime_patches(monkeypatch, captured, nav=120_000.0)
    monkeypatch.setenv("STOCK_TRADER_DEPLOY_MODE", "both")
    monkeypatch.delenv("STOCK_TRADER_CAPITAL_ALLOCATION_IARIC_PCT", raising=False)
    monkeypatch.delenv("STOCK_TRADER_CAPITAL_ALLOCATION_US_ORB_PCT", raising=False)
    monkeypatch.delenv("STOCK_TRADER_CAPITAL_ALLOCATION_ALCB_PCT", raising=False)

    services = asyncio.run(alcb_session.bootstrap_runtime("config", settings=StrategySettings()))

    assert services.raw_nav == 120_000.0
    assert services.allocated_nav == pytest.approx(40_000.0)
    assert services.capital_fraction == pytest.approx(1.0 / 3.0)
    assert services.nav == pytest.approx(40_000.0)
    assert captured["unit_risk_dollars"] == pytest.approx(200.0)
    assert captured["get_current_equity"]() == pytest.approx(40_000.0)

    engine = ALCBEngine(
        oms_service=services.oms,
        artifact=_artifact(),
        account_id=services.account_id,
        nav=services.allocated_nav,
        settings=StrategySettings(),
    )
    assert engine._portfolio.account_equity == pytest.approx(40_000.0)

    asyncio.run(alcb_session.shutdown_runtime(services))


def test_bootstrap_runtime_uses_full_nav_in_solo_mode(monkeypatch):
    captured: dict = {}
    _install_runtime_patches(monkeypatch, captured, nav=100_000.0)
    monkeypatch.setenv("STOCK_TRADER_DEPLOY_MODE", "alcb")

    services = asyncio.run(alcb_session.bootstrap_runtime("config", settings=StrategySettings()))

    assert services.raw_nav == 100_000.0
    assert services.allocated_nav == 100_000.0
    assert services.capital_fraction == 1.0
    assert captured["unit_risk_dollars"] == pytest.approx(500.0)
    assert captured["get_current_equity"]() == pytest.approx(100_000.0)

    asyncio.run(alcb_session.shutdown_runtime(services))


def test_fetch_nav_falls_back_in_dev(monkeypatch):
    monkeypatch.setenv("ALGO_TRADER_ENV", "dev")
    session = _FetchNavSession(accounts=[])

    nav = asyncio.run(alcb_session.fetch_nav(session, default_nav=12_345.0))

    assert nav == 12_345.0


def test_fetch_nav_reads_net_liquidation(monkeypatch):
    monkeypatch.setenv("ALGO_TRADER_ENV", "paper")
    session = _FetchNavSession(
        accounts=["DU1"],
        summary=[_FakeSummaryItem("NetLiquidation", "USD", "150000.25")],
    )

    nav = asyncio.run(alcb_session.fetch_nav(session))

    assert nav == 150000.25


def test_run_intraday_session_starts_engine_before_startup_reconciliation(monkeypatch):
    calls: list[str] = []
    reconnect_callbacks: list = []

    class _FakeEngine:
        _snapshot = object()

        def __init__(self, **kwargs):
            self._running = False

        @classmethod
        def try_load_state(cls, *args, **kwargs):
            return cls._snapshot

        def hydrate_state(self, snapshot):
            calls.append("hydrate")
            assert snapshot is self._snapshot

        async def start(self):
            self._running = True
            calls.append("start")

        async def stop(self):
            calls.append("stop")

        async def _reconcile_after_reconnect(self):
            calls.append(f"reconcile:{self._running}")

        async def advance(self, now):
            return None

        def subscription_instruments(self):
            return []

        def polling_instruments(self):
            return []

        def open_order_count(self):
            return 0

        def get_position_snapshot(self):
            return []

        def on_quote(self, *_args, **_kwargs):
            return None

        def on_bar(self, *_args, **_kwargs):
            return None

    class _FakeMarketData:
        def __init__(self, **kwargs):
            calls.append("market_init")

        async def start(self):
            calls.append("market_start")

        async def stop(self):
            calls.append("market_stop")

        async def ensure_hot_symbols(self, instruments):
            calls.append("ensure_hot_symbols")

        async def poll_due_bars(self, instruments, now=None):
            calls.append("poll_due_bars")

        def invalidate_subscriptions(self):
            calls.append("invalidate_subscriptions")

    class _FakeSession:
        def __init__(self):
            self.ib = object()

        def set_reconnect_callback(self, callback):
            reconnect_callbacks.append(callback)

    services = SimpleNamespace(
        oms=SimpleNamespace(),
        account_id="DU1",
        allocated_nav=100_000.0,
        trade_recorder=None,
        instrumentation_kit=None,
        session=_FakeSession(),
        contract_factory=SimpleNamespace(),
        instrumentation=None,
        bootstrap=SimpleNamespace(has_db=False, pg_store=None),
    )

    async def _bootstrap_runtime(*args, **kwargs):
        return services

    async def _shutdown_runtime(_services):
        calls.append("shutdown")

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = datetime(2026, 3, 12, 16, 2, tzinfo=ET)
            if tz is None:
                return current.replace(tzinfo=None)
            return current.astimezone(tz)

    monkeypatch.setattr(alcb_main, "bootstrap_runtime", _bootstrap_runtime)
    monkeypatch.setattr(alcb_main, "shutdown_runtime", _shutdown_runtime)
    monkeypatch.setattr(alcb_main, "IBMarketDataSource", _FakeMarketData)
    monkeypatch.setattr(alcb_main, "ALCBEngine", _FakeEngine)
    monkeypatch.setattr(alcb_main, "datetime", _FixedDateTime)

    asyncio.run(alcb_main.run_intraday_session(trading_date=_artifact().trade_date, artifact=_artifact(), settings=StrategySettings()))

    assert calls.index("start") < calls.index("reconcile:True")
