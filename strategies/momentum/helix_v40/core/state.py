from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.momentum.helix_v40.config import PositionState, Setup


@dataclass(slots=True)
class HelixV40CoreState:
    positions: list[PositionState] = field(default_factory=list)
    pending_setups: list[Setup] = field(default_factory=list)
    open_risk_r: float = 0.0
    pending_risk_r: float = 0.0
    dir_risk_r: dict[int, float] = field(default_factory=dict)
    vol_pct: float = 0.0
    ts_history: list[float] = field(default_factory=list)
    placed_signatures: set[tuple[Any, ...]] = field(default_factory=set)
    sig_expiry: dict[tuple[Any, ...], datetime] = field(default_factory=dict)
    last_m_bar: dict[int, int] = field(default_factory=dict)
    spread_recheck: list[tuple[Setup, int]] = field(default_factory=list)
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class HelixV40EntryArmed:
    """Built by engine when a setup is armed and submitted to OMS."""
    setup: Setup
    contracts: int
    risk_r: float
    signature: tuple[Any, ...]
    sig_expiry_ts: datetime
    bar_idx_1h: int = 0


@dataclass(slots=True)
class HelixV40ExpireSignatures:
    """Request to expire stale dedup signatures."""
    now: datetime


@dataclass(slots=True)
class HelixV40EntryFillContext:
    """Attached to a Fill when the fill matches a pending entry setup."""
    setup: Setup
    is_teleport: bool = False
    is_catastrophic: bool = False


@dataclass(slots=True)
class HelixV40StopUpdateRequest:
    pos_id: str
    stop_price: float
    reason: str
    symbol: str = ""
    qty: int = 0


@dataclass(slots=True)
class HelixV40FlattenRequest:
    pos_id: str
    reason: str
    symbol: str = ""


@dataclass(slots=True)
class HelixV40OrderUpdate:
    oms_order_id: str
    status: str
    timestamp: datetime | None = None
    order_role: Literal["entry", "catchup", "stop", "exit", "unknown"] = "unknown"
    pos_id: str = ""
    reason: str = ""


@dataclass(slots=True)
class HelixV40Fill:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    fill_time: datetime | None = None
    point_value: float = 2.0
    commission: float = 0.0
    is_teleport: bool = False
    is_catastrophic: bool = False
    entry_context: HelixV40EntryFillContext | None = None
