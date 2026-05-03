from __future__ import annotations

from typing import Any

from .config import ANALYSIS_SYMBOL, TRADE_SYMBOL
from .core.serializers import restore_state, snapshot_state
from .core.state import Po3ReversalCoreState


class Po3ReversalEngine:
    strategy_id = "SCALP_PO3_REVERSAL"

    def __init__(
        self,
        *,
        ib_session: Any = None,
        oms_service: Any = None,
        instruments: dict[str, Any] | None = None,
        trade_recorder: Any = None,
        equity: float = 100_000.0,
        instrumentation: Any = None,
        **_: Any,
    ) -> None:
        self.ib_session = ib_session
        self.oms_service = oms_service
        self.instruments = dict(instruments or {})
        self.trade_recorder = trade_recorder
        self.equity = float(equity)
        self.instrumentation = instrumentation
        self.analysis_symbol = str(_.get("analysis_symbol", ANALYSIS_SYMBOL)).upper()
        self.trade_symbol = str(_.get("trade_symbol", TRADE_SYMBOL)).upper()
        self.core_state = Po3ReversalCoreState()
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.core_state = restore_state(snapshot)

    def snapshot_state(self) -> dict[str, Any]:
        return snapshot_state(self.core_state)

    def health_status(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "running": self._running,
            "analysis_symbol": self.analysis_symbol,
            "trade_symbol": self.trade_symbol,
            "phase": self.core_state.phase.value,
            "last_decision_code": self.core_state.last_decision_code,
            "last_decision_details": self.core_state.last_decision_details,
            "last_seen_bar_ts": self.core_state.last_bar_ts,
        }
