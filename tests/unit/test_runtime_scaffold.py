import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.runtime.runtime import RuntimeShell
from libs.config.loader import load_strategy_registry
from libs.config.registry import build_registry_artifact


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_strategy_registry_loads_expected_inventory() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    assert len(registry.connection_groups) == 1
    assert len(registry.strategies) == 10
    assert len(registry.enabled_strategies()) == 10
    assert registry.strategies["ATRSS"].connection_group == "default"
    assert registry.strategies["AKC_Helix_v40"].connection_group == "default"
    assert "US_ORB_v1" not in registry.strategies
    assert "S5_PB" not in registry.strategies
    assert "S5_DUAL" not in registry.strategies


def test_registry_artifact_contains_all_strategies() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    artifact = build_registry_artifact(registry)

    assert len(artifact["strategies"]) == 10
    assert {item["strategy_id"] for item in artifact["strategies"]} >= {
        "ATRSS",
        "IARIC_v1",
        "NQDTC_v2.1",
    }
    by_id = {item["strategy_id"]: item for item in artifact["strategies"]}
    assert "US_ORB_v1" not in by_id
    assert {item["strategy_id"] for item in artifact["strategies"]}.isdisjoint({"S5_PB", "S5_DUAL"})


def test_checked_in_registry_artifact_matches_config() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    artifact_path = CONFIG_DIR.parent / "data" / "strategy-registry.json"

    checked_in_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert checked_in_artifact == build_registry_artifact(registry)


def test_runtime_preflight_flags_stock_readiness_failures_on_scaffold_config(monkeypatch) -> None:
    monkeypatch.delenv("IB_ACCOUNT_ID", raising=False)
    shell = RuntimeShell(CONFIG_DIR)

    checks = shell.run_preflight()

    assert checks
    by_name = {check.name: check for check in checks}
    assert "stock-account-config:default" in by_name
    assert not by_name["stock-account-config:default"].ok


@pytest.mark.asyncio
async def test_runtime_run_filters_paper_mode_only_in_live(monkeypatch) -> None:
    class DummyRegistry:
        def __init__(self) -> None:
            self.connection_groups = {}
            self.calls: list[bool] = []

        def enabled_strategies(self, *, live: bool = False):
            self.calls.append(live)
            return []

    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr(shell, "_run_async_preflight", AsyncMock(return_value=[]))

    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "paper")
    await shell.run(once=True, connect_ib=False)
    assert shell.registry.calls == [False]

    shell.registry.calls.clear()
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "live")
    await shell.run(once=True, connect_ib=False)
    assert shell.registry.calls == [True]


@pytest.mark.asyncio
async def test_runtime_run_allows_mixed_family_startup_with_stock_readiness_warnings(monkeypatch) -> None:
    class DummyRegistry:
        connection_groups = {}

        @staticmethod
        def enabled_strategies(*, live: bool = False):
            return [
                SimpleNamespace(family="swing"),
                SimpleNamespace(family="stock"),
            ]

    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr(
        shell,
        "_run_async_preflight",
        AsyncMock(return_value=[
            SimpleNamespace(
                name="stock-account-config:default",
                ok=False,
                detail="account_id is unresolved placeholder ${IB_ACCOUNT_ID}",
            ),
            SimpleNamespace(
                name="stock-artifact-readiness:IARIC_v1",
                ok=False,
                detail="watchlist unavailable",
            ),
        ]),
    )

    await shell.run(once=True, connect_ib=False)


@pytest.mark.asyncio
async def test_runtime_run_still_hard_fails_stock_only_startup_when_readiness_missing(monkeypatch) -> None:
    class DummyRegistry:
        connection_groups = {}

        @staticmethod
        def enabled_strategies(*, live: bool = False):
            return [SimpleNamespace(family="stock")]

    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr(
        shell,
        "_run_async_preflight",
        AsyncMock(return_value=[
            SimpleNamespace(
                name="stock-account-config:default",
                ok=False,
                detail="account_id is unresolved placeholder ${IB_ACCOUNT_ID}",
            ),
        ]),
    )

    with pytest.raises(RuntimeError, match="Preflight failed: 1 critical check"):
        await shell.run(once=True, connect_ib=False)
