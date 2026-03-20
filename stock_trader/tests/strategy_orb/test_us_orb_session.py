"""Session/bootstrap tests for the U.S. ORB runtime."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy_orb.config import StrategySettings
from strategy_orb.engine import USORBEngine
import strategy_orb.session as us_orb_session


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


def _install_runtime_patches(monkeypatch, captured: dict, *, nav: float):
    monkeypatch.setattr(
        us_orb_session,
        "IBKRConfig",
        lambda config_dir: SimpleNamespace(profile=SimpleNamespace(account_id="DU1"), contracts={}, routes={}),
    )
    monkeypatch.setattr(us_orb_session, "IBSession", _FakeIBSession)
    monkeypatch.setattr(
        us_orb_session,
        "ContractFactory",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        us_orb_session,
        "IBKRExecutionAdapter",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        us_orb_session,
        "bootstrap_database",
        lambda: asyncio.sleep(0, result=SimpleNamespace(pool=None, trade_recorder=None, has_db=False)),
    )
    monkeypatch.setattr(us_orb_session, "fetch_nav", lambda session, default_nav=100_000.0: asyncio.sleep(0, result=nav))
    monkeypatch.setattr(us_orb_session, "InstrumentationManager", _FakeInstrumentationManager)
    monkeypatch.setattr(us_orb_session, "InstrumentationKit", _FakeInstrumentationKit)

    async def _build_oms_service(**kwargs):
        captured.update(kwargs)
        return _FakeOMS()

    monkeypatch.setattr(us_orb_session, "build_oms_service", _build_oms_service)


def test_bootstrap_runtime_uses_allocated_nav_in_combined_mode(monkeypatch):
    captured: dict = {}
    _install_runtime_patches(monkeypatch, captured, nav=100_000.0)
    monkeypatch.setenv("STOCK_TRADER_DEPLOY_MODE", "both")
    monkeypatch.setenv("STOCK_TRADER_CAPITAL_ALLOCATION_IARIC_PCT", "25")
    monkeypatch.setenv("STOCK_TRADER_CAPITAL_ALLOCATION_US_ORB_PCT", "50")
    monkeypatch.setenv("STOCK_TRADER_CAPITAL_ALLOCATION_ALCB_PCT", "25")

    services = asyncio.run(us_orb_session.bootstrap_runtime("config", settings=StrategySettings()))

    assert services.raw_nav == 100_000.0
    assert services.allocated_nav == 50_000.0
    assert services.capital_fraction == 0.5
    assert services.nav == 50_000.0
    assert captured["unit_risk_dollars"] == 175.0
    assert captured["get_current_equity"]() == 50_000.0

    engine = USORBEngine(
        oms_service=services.oms,
        cache={},
        account_id=services.account_id,
        nav=services.allocated_nav,
        settings=StrategySettings(),
    )
    assert engine._portfolio.nav == 50_000.0

    asyncio.run(us_orb_session.shutdown_runtime(services))


def test_bootstrap_runtime_uses_full_nav_in_solo_mode(monkeypatch):
    captured: dict = {}
    _install_runtime_patches(monkeypatch, captured, nav=100_000.0)
    monkeypatch.setenv("STOCK_TRADER_DEPLOY_MODE", "us_orb")

    services = asyncio.run(us_orb_session.bootstrap_runtime("config", settings=StrategySettings()))

    assert services.raw_nav == 100_000.0
    assert services.allocated_nav == 100_000.0
    assert services.capital_fraction == 1.0
    assert captured["unit_risk_dollars"] == 350.0
    assert captured["get_current_equity"]() == 100_000.0

    asyncio.run(us_orb_session.shutdown_runtime(services))


def test_fetch_nav_falls_back_in_dev(monkeypatch):
    monkeypatch.setenv("ALGO_TRADER_ENV", "dev")
    session = _FetchNavSession(accounts=[])

    nav = asyncio.run(us_orb_session.fetch_nav(session, default_nav=12_345.0))

    assert nav == 12_345.0


def test_fetch_nav_raises_in_paper_without_account_data(monkeypatch):
    monkeypatch.setenv("ALGO_TRADER_ENV", "paper")
    session = _FetchNavSession(accounts=[])

    with pytest.raises(RuntimeError, match="Refusing to size positions off a guessed NAV"):
        asyncio.run(us_orb_session.fetch_nav(session))


def test_fetch_nav_reads_net_liquidation(monkeypatch):
    monkeypatch.setenv("ALGO_TRADER_ENV", "paper")
    session = _FetchNavSession(
        accounts=["DU1"],
        summary=[_FakeSummaryItem("NetLiquidation", "USD", "150000.25")],
    )

    nav = asyncio.run(us_orb_session.fetch_nav(session))

    assert nav == 150000.25
