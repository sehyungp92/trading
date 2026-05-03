from __future__ import annotations

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import Po3ReversalCoreState


def snapshot_state(state: Po3ReversalCoreState) -> dict:
    return snapshot_dataclass(state)


def restore_state(payload: dict) -> Po3ReversalCoreState:
    return restore_dataclass(Po3ReversalCoreState, payload)

