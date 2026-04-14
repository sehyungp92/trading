from pathlib import Path

import pytest

from apps.runtime.runtime import RuntimeShell
from libs.config.loader import load_strategy_registry
from libs.config.registry import build_registry_artifact


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_strategy_registry_loads_expected_inventory() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    assert len(registry.connection_groups) == 1
    assert len(registry.strategies) == 13
    assert registry.strategies["ATRSS"].connection_group == "default"
    assert registry.strategies["AKC_Helix_v40"].connection_group == "default"


def test_registry_artifact_contains_all_strategies() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    artifact = build_registry_artifact(registry)

    assert len(artifact["strategies"]) == 13
    assert {item["strategy_id"] for item in artifact["strategies"]} >= {
        "ATRSS",
        "IARIC_v1",
        "NQDTC_v2.1",
    }


def test_runtime_preflight_passes_on_scaffold_config() -> None:
    shell = RuntimeShell(CONFIG_DIR)

    checks = shell.run_preflight()

    assert checks
    assert all(check.ok for check in checks), checks


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

    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "paper")
    await shell.run(once=True, connect_ib=False)
    assert shell.registry.calls == [False]

    shell.registry.calls.clear()
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "live")
    await shell.run(once=True, connect_ib=False)
    assert shell.registry.calls == [True]
