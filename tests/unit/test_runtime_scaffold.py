from pathlib import Path

from apps.runtime.runtime import RuntimeShell
from libs.config.loader import load_strategy_registry
from libs.config.registry import build_registry_artifact


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_strategy_registry_loads_expected_inventory() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    assert len(registry.connection_groups) == 3
    assert len(registry.strategies) == 11
    assert registry.strategies["ATRSS"].connection_group == "swing"
    assert registry.strategies["AKC_Helix_v40"].connection_group == "momentum"


def test_registry_artifact_contains_all_strategies() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    artifact = build_registry_artifact(registry)

    assert len(artifact["strategies"]) == 11
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

