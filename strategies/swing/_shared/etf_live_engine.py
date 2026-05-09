"""Thin live-engine adapter for shared-core 15m ETF swing strategies."""
from __future__ import annotations

import logging
from typing import Any

from strategies.swing._shared.etf_core import ETFBarInput, ETFFill, ETFOrderUpdate

logger = logging.getLogger(__name__)


class ETFCoreLiveEngine:
    """Runtime shell that keeps live state on the same core path as backtests.

    The production data/OMS integrations can call ``process_bar_input``,
    ``process_fill`` and ``process_order_update`` after constructing the same
    completed-bar payloads used by the replay engine.
    """

    def __init__(
        self,
        *,
        strategy_id: str,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        config: dict[str, Any],
        core_logic: Any,
        serializers: Any,
        trade_recorder: Any | None = None,
        equity: float = 100_000.0,
        market_calendar: Any | None = None,
        kit: Any | None = None,
        equity_offset: float = 0.0,
        equity_alloc_pct: float = 1.0,
        coordinator: Any | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = instruments
        self._config = config
        self._core_logic = core_logic
        self._serializers = serializers
        self._recorder = trade_recorder
        self._equity = equity
        self._equity_offset = equity_offset
        self._equity_alloc_pct = equity_alloc_pct
        self._market_calendar = market_calendar
        self._kit = kit
        self._coordinator = coordinator
        self._running = False
        self._state = serializers.restore_state(None)
        self._last_actions: list[Any] = []
        self._last_events: list[Any] = []
        self._bars_processed = 0

    async def start(self) -> None:
        self._running = True
        logger.info("%s ETF core live engine started", self.strategy_id)

    async def stop(self) -> None:
        self._running = False
        logger.info("%s ETF core live engine stopped", self.strategy_id)

    def health_status(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "running": self._running,
            "last_decision_code": self._state.last_decision_code,
            "last_decision_details": dict(self._state.last_decision_details),
            "last_bar_ts": self._state.last_bar_ts.isoformat() if self._state.last_bar_ts else None,
            "bars_processed": self._bars_processed,
        }

    def liveness_payload(self) -> dict[str, Any]:
        return {
            "bars_processed": self._bars_processed,
            "open_positions": sorted(self._state.positions),
            "pending_orders": sorted(self._state.pending_orders),
        }

    def snapshot_state(self) -> dict[str, Any]:
        return self._serializers.snapshot_state(self._state)

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        self._state = self._serializers.restore_state(snapshot)

    def process_bar_input(self, bar_input: ETFBarInput) -> tuple[list[Any], list[Any]]:
        cfg = self._config.get(bar_input.symbol)
        if cfg is None:
            return [], []
        if not bar_input.equity:
            bar_input.equity = max(0.0, (self._equity + self._equity_offset) * self._equity_alloc_pct)
        self._state, actions, events = self._core_logic.on_bar(self._state, bar_input, cfg)
        self._last_actions = list(actions)
        self._last_events = list(events)
        self._bars_processed += 1
        return actions, events

    def process_fill(self, fill: ETFFill) -> tuple[list[Any], list[Any]]:
        self._state, actions, events = self._core_logic.on_fill(self._state, fill)
        self._last_actions = list(actions)
        self._last_events = list(events)
        return actions, events

    def process_order_update(self, update: ETFOrderUpdate) -> tuple[list[Any], list[Any]]:
        self._state, actions, events = self._core_logic.on_order_update(self._state, update)
        self._last_actions = list(actions)
        self._last_events = list(events)
        return actions, events
