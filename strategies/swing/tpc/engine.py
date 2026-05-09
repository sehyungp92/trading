"""Live adapter for TPC."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from strategies.core.actions import ReplaceProtectiveStop
from strategies.swing._shared.etf_core import ETFBarInput, ETFFill, ETFOrderUpdate
from strategies.swing._shared.etf_live_engine import ETFCoreLiveEngine
from strategies.swing.tpc import STRATEGY_ID
from strategies.swing.tpc import instrumentation_adapter as _adapter
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.core import logic, serializers


class TPCEngine(ETFCoreLiveEngine):
    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        config: dict[str, TPCSymbolConfig],
        trade_recorder: Any | None = None,
        equity: float = 100_000.0,
        market_calendar: Any | None = None,
        kit: Any | None = None,
        equity_offset: float = 0.0,
        equity_alloc_pct: float = 1.0,
        coordinator: Any | None = None,
    ) -> None:
        super().__init__(
            strategy_id=STRATEGY_ID,
            ib_session=ib_session,
            oms_service=oms_service,
            instruments=instruments,
            config=config,
            core_logic=logic,
            serializers=serializers,
            trade_recorder=trade_recorder,
            equity=equity,
            market_calendar=market_calendar,
            kit=kit,
            equity_offset=equity_offset,
            equity_alloc_pct=equity_alloc_pct,
            coordinator=coordinator,
        )
        self._setup_cache: dict[str, Any] = {}
        self._position_stop_history: dict[str, float] = {}

    def process_bar_input(self, bar_input: ETFBarInput) -> tuple[list[Any], list[Any]]:
        cfg = self._config.get(bar_input.symbol)
        pre_position = deepcopy(self._state.positions.get(bar_input.symbol))
        actions, events = super().process_bar_input(bar_input)
        if self._kit is None or cfg is None:
            return actions, events

        decision = "NO_SIGNAL"
        emitted_setup = None
        rejections: list[dict[str, Any]] = []
        for event in events:
            code = event.code
            if code == "ENTRY_REQUESTED":
                setup_id = event.details.get("setup_id")
                setup = self._state.setups.get(setup_id) if setup_id else None
                if setup is not None:
                    self._setup_cache[setup.setup_id] = setup
                    emitted_setup = setup
                decision = "ENTRY_REQUESTED"
            elif code == "SETUP_REJECTED":
                rejection = dict(event.details)
                rejections.append(rejection)
                bar_ts = bar_input.bar_15m.timestamp if bar_input.bar_15m else None
                _adapter.route_missed(self._kit, rejection, cfg, bar_ts=bar_ts)
            elif code in {"MANAGING_POSITION", "ENTRY_PENDING", "NO_SIGNAL"}:
                decision = code

        post_position = self._state.positions.get(bar_input.symbol)
        position = post_position or pre_position
        for action in actions:
            if isinstance(action, ReplaceProtectiveStop):
                setup_id = self._lookup_setup_id_for_symbol(action.symbol, position)
                if not setup_id:
                    continue
                if setup_id not in self._position_stop_history and pre_position is not None:
                    self._position_stop_history[setup_id] = float(pre_position.current_stop)
                old_stop = self._position_stop_history.get(setup_id)
                new_stop = float(action.stop_price)
                if old_stop is not None and abs(old_stop - new_stop) > 1e-9:
                    _adapter.route_stop_adjustment(
                        self._kit,
                        setup_id=setup_id,
                        symbol=action.symbol,
                        old_stop=float(old_stop),
                        new_stop=new_stop,
                        action_reason=str(getattr(action, "reason", "") or ""),
                        position=position,
                    )
                self._position_stop_history[setup_id] = new_stop

        if post_position is not None:
            self._position_stop_history.setdefault(post_position.setup_id, float(post_position.current_stop))

        _adapter.route_filter_decisions(
            self._kit, bar_input, cfg, rejections=rejections, entry_setup=emitted_setup,
        )
        _adapter.route_indicator_snapshot(self._kit, bar_input, self._state, cfg, decision, emitted_setup)
        return actions, events

    def process_fill(self, fill: ETFFill) -> tuple[list[Any], list[Any]]:
        role = (fill.order_role or "").lower()
        pre_position = None
        if role != "entry":
            pre_position = deepcopy(self._state.positions.get(fill.symbol)) if fill.symbol in self._state.positions else None
        actions, events = super().process_fill(fill)
        if self._kit is None:
            return actions, events
        cfg = self._config.get(fill.symbol)

        for event in events:
            if event.code == "ENTRY_FILLED":
                setup_id = event.details.get("setup_id")
                setup = self._setup_cache.pop(setup_id, None) if setup_id else None
                if setup is None:
                    continue
                if cfg is None:
                    continue
                _adapter.route_entry(self._kit, setup, fill, cfg, self._state)
                position = self._state.positions.get(fill.symbol)
                if position is not None:
                    self._position_stop_history[setup.setup_id] = float(position.current_stop)
            elif event.code in {"EXIT_FILLED", "STOP_FILLED"}:
                if pre_position is None:
                    continue
                _adapter.route_exit(
                    self._kit,
                    pre_position=pre_position,
                    fill=fill,
                    event_code=event.code,
                    event_reason=event.details.get("reason"),
                )
                self._position_stop_history.pop(pre_position.setup_id, None)
            # PARTIAL_EXIT_FILLED: cache update only — no log_exit. Pre/post state delta
            # is captured by subsequent stop-adjustment routing on the partial_resize.
        return actions, events

    def process_order_update(self, update: ETFOrderUpdate) -> tuple[list[Any], list[Any]]:
        actions, events = super().process_order_update(update)
        if self._kit is None:
            return actions, events
        for event in events:
            if event.code in {"ORDER_TERMINAL", "ADDON_ORDER_TERMINAL"}:
                _adapter.route_order_event(self._kit, update, event)
        return actions, events

    def _lookup_setup_id_for_symbol(self, symbol: str, position: Any | None) -> str:
        if position is not None and getattr(position, "setup_id", ""):
            return position.setup_id
        # fallback: scan state.positions in case override doesn't have it
        for sym, pos in self._state.positions.items():
            if sym == symbol:
                return pos.setup_id
        return ""
