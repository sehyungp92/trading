from __future__ import annotations

from types import SimpleNamespace

import pytest

from strategies.momentum.helix_v40.plugin import HelixV40Plugin
from strategies.momentum.vdub.plugin import VdubNQv4Plugin
from strategies.swing.akc_helix.plugin import AKCHelixPlugin
from strategies.swing.atrss.plugin import ATRSSPlugin
from strategies.swing.breakout.plugin import BreakoutPlugin
from strategies.swing.brs.plugin import BRSPlugin


@pytest.mark.parametrize(
    ("plugin_cls", "strategy_id"),
    [
        (ATRSSPlugin, "ATRSS"),
        (BreakoutPlugin, "SWING_BREAKOUT_V3"),
        (AKCHelixPlugin, "AKC_HELIX"),
        (BRSPlugin, "BRS_R9"),
        (VdubNQv4Plugin, "VdubusNQ_v4"),
        (HelixV40Plugin, "AKC_Helix_v40"),
    ],
)
def test_plugin_health_status_delegates_full_engine_payload(plugin_cls, strategy_id) -> None:
    plugin = plugin_cls.__new__(plugin_cls)
    plugin._engine = SimpleNamespace(
        health_status=lambda: {
            "strategy_id": strategy_id,
            "running": True,
            "last_decision_code": "ENTRY_SUBMITTED",
        }
    )

    assert plugin.health_status() == {
        "strategy_id": strategy_id,
        "running": True,
        "last_decision_code": "ENTRY_SUBMITTED",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin_cls",
    [
        ATRSSPlugin,
        BreakoutPlugin,
        AKCHelixPlugin,
        BRSPlugin,
        VdubNQv4Plugin,
        HelixV40Plugin,
    ],
)
async def test_plugin_snapshot_and_hydrate_delegate_when_engine_supports_contract(plugin_cls) -> None:
    hydrated: list[dict[str, object]] = []

    class _FakeEngine:
        async def hydrate(self, snapshot: dict[str, object]) -> None:
            hydrated.append(snapshot)

        def snapshot_state(self) -> dict[str, object]:
            return {"restored": True}

    plugin = plugin_cls.__new__(plugin_cls)
    plugin._engine = _FakeEngine()

    await plugin.hydrate({"sequence": 1})

    assert hydrated == [{"sequence": 1}]
    assert plugin.snapshot_state() == {"restored": True}


@pytest.mark.asyncio
async def test_plugin_hydrate_falls_back_to_sync_hydrate_state() -> None:
    hydrated: list[dict[str, object]] = []

    class _FakeEngine:
        def hydrate_state(self, snapshot: dict[str, object]) -> None:
            hydrated.append(snapshot)

        def snapshot_state(self) -> dict[str, object]:
            return {"restored": "sync"}

    plugin = ATRSSPlugin.__new__(ATRSSPlugin)
    plugin._engine = _FakeEngine()

    await plugin.hydrate({"sequence": 2})

    assert hydrated == [{"sequence": 2}]
    assert plugin.snapshot_state() == {"restored": "sync"}
