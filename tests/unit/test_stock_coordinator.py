from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from libs.config.loader import load_strategy_registry
from strategies.stock.readiness import StockReadinessFailure
from strategies.stock.coordinator import StockFamilyCoordinator


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_stock_coordinator_skips_legacy_orb_descriptor_from_active_rule_inputs(monkeypatch) -> None:
    monkeypatch.setattr("strategies.stock.coordinator.get_environment", lambda: "paper")

    registry = load_strategy_registry(CONFIG_DIR)
    coordinator = StockFamilyCoordinator(
        SimpleNamespace(registry=registry, session=SimpleNamespace(ib=None))
    )

    active_ids = coordinator._enabled_stock_strategy_ids()
    assert active_ids == ("IARIC_v1", "ALCB_v1")

    descriptors = coordinator._build_strategy_descriptors({}, ("IARIC_v1", "US_ORB_v1", "ALCB_v1"))
    assert [descriptor["strategy_id"] for descriptor in descriptors] == ["IARIC_v1", "ALCB_v1"]

    rule_inputs = coordinator._portfolio_rule_inputs(
        tuple(descriptor["strategy_id"] for descriptor in descriptors)
    )
    assert rule_inputs["family_strategy_ids"] == ("IARIC_v1", "ALCB_v1")
    assert rule_inputs["symbol_collision_pairs"] == ()
    assert rule_inputs["strategy_priorities"] == (("IARIC_v1", 0), ("ALCB_v1", 1))


@pytest.mark.asyncio
async def test_stock_coordinator_start_fails_fast_on_unresolved_account_config(monkeypatch) -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    coordinator = StockFamilyCoordinator(SimpleNamespace(registry=registry))
    build_oms_service = AsyncMock()

    monkeypatch.setattr(coordinator, "_enabled_stock_strategy_ids", lambda: ("IARIC_v1",))
    monkeypatch.setattr(
        "strategies.stock.coordinator.validate_stock_readiness",
        lambda *args, **kwargs: (
            {},
            [
                StockReadinessFailure(
                    category="account-config",
                    identifier="default",
                    detail="account_id is unresolved placeholder ${IB_ACCOUNT_ID}",
                )
            ],
        ),
    )
    monkeypatch.setattr("libs.oms.services.factory.build_oms_service", build_oms_service)

    with pytest.raises(RuntimeError, match="stock-account-config:default"):
        await coordinator.start()

    build_oms_service.assert_not_awaited()
    assert coordinator._heartbeat_task is None
    assert coordinator._market_data_task is None


@pytest.mark.asyncio
async def test_stock_coordinator_start_fails_fast_on_missing_artifact(monkeypatch) -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    coordinator = StockFamilyCoordinator(SimpleNamespace(registry=registry))
    build_oms_service = AsyncMock()

    monkeypatch.setattr(coordinator, "_enabled_stock_strategy_ids", lambda: ("ALCB_v1",))
    monkeypatch.setattr(
        "strategies.stock.coordinator.validate_stock_readiness",
        lambda *args, **kwargs: (
            {},
            [
                StockReadinessFailure(
                    category="artifact-readiness",
                    identifier="ALCB_v1",
                    detail="candidate unavailable for 2026-04-24: missing file",
                )
            ],
        ),
    )
    monkeypatch.setattr("libs.oms.services.factory.build_oms_service", build_oms_service)

    with pytest.raises(RuntimeError, match="stock-artifact-readiness:ALCB_v1"):
        await coordinator.start()

    build_oms_service.assert_not_awaited()
    assert coordinator._heartbeat_task is None
    assert coordinator._market_data_task is None
