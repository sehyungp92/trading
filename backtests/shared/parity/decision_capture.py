from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

from strategies.core.events import DecisionEvent


def normalize_decision_event(event: DecisionEvent) -> dict[str, Any]:
    return {
        "code": event.code,
        "ts": event.ts.isoformat(),
        "symbol": event.symbol,
        "timeframe": event.timeframe,
        "details": _normalize_value(event.details),
    }


def normalize_decision_stream(events: Iterable[DecisionEvent]) -> list[dict[str, Any]]:
    return [normalize_decision_event(event) for event in events]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _normalize_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value
