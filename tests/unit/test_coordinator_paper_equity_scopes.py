from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class _FakeOMS:
    _portfolio_checker = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _FakeEngine:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def subscription_instruments(self):
        return []

    def polling_instruments(self):
        return []

    async def on_quote(self, *args, **kwargs) -> None:
        pass

    async def on_bar(self, *args, **kwargs) -> None:
        pass


class _FakeInstrumentation:
    sidecar = object()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _FakeMarketData:
    def __init__(self, **kwargs) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def ensure_hot_symbols(self, symbols) -> None:
        pass

    async def poll_due_bars(self, instruments, now=None) -> None:
        pass


class _IB:
    async def qualifyContractsAsync(self, contract):
        return [contract]


class _Session:
    def __init__(self) -> None:
        self.ib = _IB()

    async def req_historical_data(self, *args, **kwargs):
        return [SimpleNamespace(close=21000.0)]

    def add_reconnect_callback(self, callback) -> None:
        pass


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        registry=SimpleNamespace(enabled_strategies=lambda live=False: []),
        session=_Session(),
        db_pool=object(),
        account_gate=None,
        portfolio=SimpleNamespace(
            capital=SimpleNamespace(paper_initial_equity=100_000.0)
        ),
        trade_recorder=None,
    )


@pytest.mark.asyncio
async def test_momentum_uses_per_strategy_paper_equity_scopes(monkeypatch):
    from strategies.momentum.coordinator import MomentumFamilyCoordinator

    scopes: list[tuple[str, float]] = []
    navs = {"M1": 40_000.0, "M2": 60_000.0}

    class FakePaperEquityManager:
        def __init__(self, pool, account_scope: str, initial_equity: float) -> None:
            scopes.append((account_scope, initial_equity))
            self.scope = account_scope

        async def load(self) -> float:
            return navs[self.scope]

    build_oms = AsyncMock(return_value=_FakeOMS())
    monkeypatch.setattr("strategies.momentum.coordinator.get_environment", lambda: "paper")
    monkeypatch.setattr(
        "libs.broker_ibkr.config.loader.IBKRConfig",
        lambda config_dir: SimpleNamespace(
            profile=SimpleNamespace(account_id="DU123"),
            contracts={},
            routes={},
        ),
    )
    monkeypatch.setattr(
        "libs.broker_ibkr.mapping.contract_factory.ContractFactory",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "libs.broker_ibkr.adapters.execution_adapter.IBKRExecutionAdapter",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr("libs.persistence.paper_equity.PaperEquityManager", FakePaperEquityManager)
    monkeypatch.setattr(
        "libs.config.capital_bootstrap.bootstrap_capital",
        lambda equity, config_dir: {
            "M1": SimpleNamespace(allocated_nav=55_000.0, capital_pct=55.0),
            "M2": SimpleNamespace(allocated_nav=45_000.0, capital_pct=45.0),
        },
    )
    monkeypatch.setattr("libs.oms.services.factory.build_oms_service", build_oms)
    monkeypatch.setattr(
        "strategies.momentum.instrumentation.src.bootstrap.InstrumentationManager",
        lambda *args, **kwargs: _FakeInstrumentation(),
    )

    coordinator = MomentumFamilyCoordinator(_ctx())
    monkeypatch.setattr(
        coordinator,
        "_build_strategy_descriptors",
        lambda: [
            {
                "strategy_id": "M1",
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "heat_cap_R": 1.0,
                "portfolio_daily_stop_R": 3.0,
                "engine_cls": _FakeEngine,
                "build_instruments": lambda: {},
                "instr_type": "m1",
            },
            {
                "strategy_id": "M2",
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "heat_cap_R": 1.0,
                "portfolio_daily_stop_R": 3.0,
                "engine_cls": _FakeEngine,
                "build_instruments": lambda: {},
                "instr_type": "m2",
            },
        ],
    )

    await coordinator.start()
    await coordinator.stop()

    assert scopes == [("M1", 55_000.0), ("M2", 45_000.0)]
    kwargs_by_sid = {
        call.kwargs["strategy_id"]: call.kwargs for call in build_oms.await_args_list
    }
    assert kwargs_by_sid["M1"]["paper_equity_scope"] == "M1"
    assert kwargs_by_sid["M1"]["paper_initial_equity"] == 55_000.0
    assert kwargs_by_sid["M2"]["paper_equity_scope"] == "M2"
    assert kwargs_by_sid["M2"]["paper_initial_equity"] == 45_000.0


@pytest.mark.asyncio
async def test_stock_uses_per_strategy_paper_equity_scopes(monkeypatch):
    from strategies.stock.coordinator import StockFamilyCoordinator

    scopes: list[tuple[str, float]] = []
    navs = {"IARIC_v1": 40_000.0, "ALCB_v1": 60_000.0}

    class FakePaperEquityManager:
        def __init__(self, pool, account_scope: str, initial_equity: float) -> None:
            scopes.append((account_scope, initial_equity))
            self.scope = account_scope

        async def load(self) -> float:
            return navs[self.scope]

    build_oms = AsyncMock(return_value=_FakeOMS())
    monkeypatch.setattr("strategies.stock.coordinator.get_environment", lambda: "paper")
    monkeypatch.setattr(
        "strategies.stock.coordinator.validate_stock_readiness",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr("libs.persistence.paper_equity.PaperEquityManager", FakePaperEquityManager)
    monkeypatch.setattr(
        "libs.config.capital_bootstrap.bootstrap_capital",
        lambda equity, config_dir: {
            "IARIC_v1": SimpleNamespace(allocated_nav=50_000.0, capital_pct=50.0),
            "ALCB_v1": SimpleNamespace(allocated_nav=50_000.0, capital_pct=50.0),
        },
    )
    monkeypatch.setattr("libs.oms.services.factory.build_oms_service", build_oms)
    monkeypatch.setattr(
        "strategies.stock.instrumentation.src.bootstrap.InstrumentationManager",
        lambda *args, **kwargs: _FakeInstrumentation(),
    )

    coordinator = StockFamilyCoordinator(_ctx())
    coordinator._contract_factory = object()
    monkeypatch.setattr(coordinator, "_enabled_stock_strategy_ids", lambda: ("IARIC_v1", "ALCB_v1"))
    monkeypatch.setattr(
        coordinator,
        "_build_strategy_descriptors",
        lambda artifacts, ids: [
            {
                "strategy_id": "IARIC_v1",
                "data_key": "artifact",
                "data_value": object(),
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "heat_cap_R": 1.0,
                "portfolio_daily_stop_R": 3.0,
                "adapter": lambda session: object(),
                "account_id": "DU123",
                "settings": object(),
                "trade_recorder": object(),
                "diagnostics_factory": lambda: object(),
                "engine_cls": lambda: _FakeEngine,
                "market_data_cls": lambda: _FakeMarketData,
                "instr_type": "iaric",
            },
            {
                "strategy_id": "ALCB_v1",
                "data_key": "artifact",
                "data_value": object(),
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "heat_cap_R": 1.0,
                "portfolio_daily_stop_R": 3.0,
                "adapter": lambda session: object(),
                "account_id": "DU123",
                "settings": object(),
                "trade_recorder": object(),
                "diagnostics_factory": lambda: object(),
                "engine_cls": lambda: _FakeEngine,
                "market_data_cls": lambda: _FakeMarketData,
                "instr_type": "alcb",
            },
        ],
    )

    await coordinator.start()
    await coordinator.stop()

    assert scopes == [("IARIC_v1", 50_000.0), ("ALCB_v1", 50_000.0)]
    kwargs_by_sid = {
        call.kwargs["strategy_id"]: call.kwargs for call in build_oms.await_args_list
    }
    assert kwargs_by_sid["IARIC_v1"]["paper_equity_scope"] == "IARIC_v1"
    assert kwargs_by_sid["IARIC_v1"]["paper_initial_equity"] == 50_000.0
    assert kwargs_by_sid["ALCB_v1"]["paper_equity_scope"] == "ALCB_v1"
    assert kwargs_by_sid["ALCB_v1"]["paper_initial_equity"] == 50_000.0
