from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class DecisionEvent:
    code: str
    ts: datetime
    symbol: str
    timeframe: str
    details: dict[str, Any] = field(default_factory=dict)
