"""Minimal structured diagnostics for the ORB runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class JsonlDiagnostics:
    root: Path
    enabled: bool = True
    _cache: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def emit(self, stream: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._cache.setdefault(stream, self.root / f"{stream}.jsonl")
        record = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def log_state(self, symbol: str, state: str, reason: str = "") -> None:
        self.emit("state", {"symbol": symbol, "state": state, "reason": reason})

    def log_candidate(self, symbol: str, score: float, surge: float, rvol: float) -> None:
        self.emit(
            "candidates",
            {"symbol": symbol, "score": round(score, 2), "surge": round(surge, 3), "rvol": round(rvol, 3)},
        )

    def log_regime(self, snapshot: dict[str, Any]) -> None:
        self.emit("regime", snapshot)

    def log_vdm(self, symbol: str, state: str, score: int, reasons: list[str] | tuple[str, ...]) -> None:
        self.emit("vdm", {"symbol": symbol, "state": state, "score": score, "reasons": list(reasons)})

    def log_missed_fill(self, symbol: str, payload: dict[str, Any]) -> None:
        self.emit("missed_fills", {"symbol": symbol, **payload})
