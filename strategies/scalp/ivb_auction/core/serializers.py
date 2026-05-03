from __future__ import annotations

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .state import IvbAuctionCoreState


def snapshot_state(state: IvbAuctionCoreState) -> dict:
    return snapshot_dataclass(state)


def restore_state(payload: dict) -> IvbAuctionCoreState:
    return restore_dataclass(IvbAuctionCoreState, payload)

