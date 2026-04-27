from __future__ import annotations

from typing import Any, Mapping

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import BreakoutCoreState


def snapshot_state(state: BreakoutCoreState) -> dict[str, Any]:
    return snapshot_dataclass(state)


def restore_state(snapshot: Mapping[str, Any]) -> BreakoutCoreState:
    return restore_dataclass(BreakoutCoreState, snapshot)
