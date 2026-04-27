from __future__ import annotations

from typing import Any, Mapping

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import HelixV40CoreState


def snapshot_state(state: HelixV40CoreState) -> dict[str, Any]:
    return snapshot_dataclass(state)


def restore_state(snapshot: Mapping[str, Any]) -> HelixV40CoreState:
    return restore_dataclass(HelixV40CoreState, snapshot)
