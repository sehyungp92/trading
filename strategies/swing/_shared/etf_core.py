"""Reusable shared-core contracts for 15m ETF swing strategies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.swing._shared.models import Direction


@dataclass(frozen=True, slots=True)
class BarData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True, slots=True)
class BarWindow:
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    times: tuple[datetime, ...]

    def __len__(self) -> int:
        return len(self.closes)

    @property
    def last(self) -> BarData | None:
        if len(self.closes) == 0:
            return None
        return BarData(
            timestamp=self.times[-1],
            open=float(self.opens[-1]),
            high=float(self.highs[-1]),
            low=float(self.lows[-1]),
            close=float(self.closes[-1]),
            volume=float(self.volumes[-1]) if len(self.volumes) else 0.0,
        )


@dataclass(slots=True)
class SetupSnapshot:
    setup_id: str
    strategy_id: str
    symbol: str
    direction: Direction
    grade: str
    setup_type: str
    entry_model: str
    state: str
    created_ts: datetime
    entry_price: float
    stop_price: float
    qty: int
    score: float
    risk_pct: float
    t1_r: float
    t1_partial_pct: float
    t2_r: float
    t2_partial_pct: float
    entry_order_type: str = "MARKET"
    entry_limit_price: float = 0.0
    entry_stop_price: float = 0.0
    entry_ttl_hours: float = 0.0
    target_price: float = 0.0
    t2_price: float = 0.0
    max_hold_bars_15m: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)


@dataclass(slots=True)
class ETFPosition:
    setup_id: str
    symbol: str
    direction: Direction
    qty_open: int
    qty_initial: int
    entry_price: float
    current_stop: float
    initial_stop: float
    entry_ts: datetime
    risk_per_share: float
    setup_type: str
    grade: str
    entry_model: str
    score: float
    stop_order_id: str = ""
    t1_done: bool = False
    t2_done: bool = False
    runner_active: bool = False
    mfe_price: float = 0.0
    mae_price: float = 0.0
    bars_held_15m: int = 0
    pending_exit_roles: set[str] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mfe_r(self) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        if self.direction == Direction.LONG:
            return (self.mfe_price - self.entry_price) / self.risk_per_share
        return (self.entry_price - self.mfe_price) / self.risk_per_share

    @property
    def mae_r(self) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        if self.direction == Direction.LONG:
            return (self.entry_price - self.mae_price) / self.risk_per_share
        return (self.mae_price - self.entry_price) / self.risk_per_share


@dataclass(slots=True)
class ETFCoreState:
    setups: dict[str, SetupSnapshot] = field(default_factory=dict)
    positions: dict[str, ETFPosition] = field(default_factory=dict)
    pending_orders: dict[str, SetupSnapshot] = field(default_factory=dict)
    daily_loss_r: float = 0.0
    weekly_loss_r: float = 0.0
    failed_entries: dict[str, int] = field(default_factory=dict)
    last_bar_ts: datetime | None = None
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ETFBarInput:
    symbol: str = ""
    bar_15m: BarData | None = None
    bars_15m: BarWindow | None = None
    bars_30m: BarWindow | None = None
    bars_1h: BarWindow | None = None
    bars_4h: BarWindow | None = None
    bars_daily: BarWindow | None = None
    indicators: dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    timestamp: datetime | None = None
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ETFFill:
    oms_order_id: str
    fill_price: float = 0.0
    fill_qty: int = 0
    symbol: str = ""
    fill_time: datetime | None = None
    commission: float = 0.0
    order_role: str = "entry"
    exit_type: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ETFOrderUpdate:
    oms_order_id: str
    status: str = ""
    symbol: str = ""
    timestamp: datetime | None = None
    order_role: str = "unknown"
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


EvaluateSetup = Callable[[ETFCoreState, ETFBarInput, Any], SetupSnapshot | None]
ManagePosition = Callable[[ETFCoreState, ETFBarInput, Any, ETFPosition], list[Any]]


def _shallow_copy_state(state: ETFCoreState) -> ETFCoreState:
    """Construct a new state with shared dict references (cheap) and fresh scalar fields."""
    base_kwargs = {
        "setups": state.setups,
        "positions": state.positions,
        "pending_orders": state.pending_orders,
        "daily_loss_r": state.daily_loss_r,
        "weekly_loss_r": state.weekly_loss_r,
        "failed_entries": state.failed_entries,
        "last_bar_ts": state.last_bar_ts,
        "last_decision_code": state.last_decision_code,
        "last_decision_details": {},
    }
    if type(state) is ETFCoreState:
        return ETFCoreState(**base_kwargs)
    for item in fields(state):
        if item.name not in base_kwargs:
            base_kwargs[item.name] = getattr(state, item.name)
    return state.__class__(**base_kwargs)


_EVAL_SETUP_ACCEPTS_COLLECTOR: dict[int, bool] = {}


def _evaluate_setup_accepts_collector(fn: EvaluateSetup) -> bool:
    fn_id = id(fn)
    cached = _EVAL_SETUP_ACCEPTS_COLLECTOR.get(fn_id)
    if cached is not None:
        return cached
    import inspect

    try:
        params = inspect.signature(fn).parameters
        accepts = "rejection_collector" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (ValueError, TypeError):
        accepts = False
    _EVAL_SETUP_ACCEPTS_COLLECTOR[fn_id] = accepts
    return accepts


def on_bar_common(
    state: ETFCoreState,
    bar_input: ETFBarInput | None,
    cfg: Any,
    *,
    strategy_id: str,
    evaluate_setup: EvaluateSetup,
    manage_position: ManagePosition | None = None,
) -> tuple[ETFCoreState, list[Any], list[DecisionEvent]]:
    next_state = _shallow_copy_state(state)
    actions: list[Any] = []
    events: list[DecisionEvent] = []
    if bar_input is None:
        return next_state, actions, events

    ts = bar_input.timestamp or (bar_input.bar_15m.timestamp if bar_input.bar_15m else None) or datetime.now(timezone.utc)
    next_state.last_bar_ts = ts

    if bar_input.decision_code and bar_input.bar_15m is None:
        events.append(_event(strategy_id, bar_input.decision_code, ts, bar_input.symbol, bar_input.decision_details))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    symbol = bar_input.symbol
    bar = bar_input.bar_15m
    if bar is None:
        return next_state, actions, events

    position = state.positions.get(symbol)
    if position is not None:
        # Deep-copy only the specific position being mutated
        next_state.positions = dict(state.positions)
        next_state.positions[symbol] = deepcopy(position)
        position = next_state.positions[symbol]
        _mark_position(position, bar)
        if manage_position is not None:
            actions.extend(manage_position(next_state, bar_input, cfg, position))
        if not actions:
            events.append(_event(strategy_id, "MANAGING_POSITION", ts, symbol, {"qty": position.qty_open}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if any(setup.symbol == symbol for setup in next_state.pending_orders.values()):
        events.append(_event(strategy_id, "ENTRY_PENDING", ts, symbol, {}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    rejection_collector: list[dict[str, Any]] = []
    if _evaluate_setup_accepts_collector(evaluate_setup):
        setup = evaluate_setup(next_state, bar_input, cfg, rejection_collector=rejection_collector)
    else:
        setup = evaluate_setup(next_state, bar_input, cfg)
    for rejection in rejection_collector:
        events.append(_event(strategy_id, "SETUP_REJECTED", ts, symbol, dict(rejection)))
    if setup is None or setup.qty <= 0:
        events.append(_event(strategy_id, "NO_SIGNAL", ts, symbol, {}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    # Entry path mutates setups and pending_orders — create new dicts
    next_state.setups = dict(state.setups)
    next_state.pending_orders = dict(state.pending_orders)
    client_order_id = setup.setup_id
    next_state.setups[setup.setup_id] = setup
    next_state.pending_orders[client_order_id] = setup
    actions.append(
        SubmitEntry(
            client_order_id=client_order_id,
            symbol=symbol,
            side="BUY" if setup.direction == Direction.LONG else "SELL",
            qty=setup.qty,
            order_type=setup.entry_order_type,  # type: ignore[arg-type]
            tif="DAY",
            price=setup.entry_price,
            limit_price=setup.entry_limit_price or None,
            stop_price=setup.entry_stop_price or None,
            risk_context={
                "stop_for_risk": setup.stop_price,
                "planned_entry_price": setup.entry_price,
                "risk_pct": setup.risk_pct,
            },
            metadata={
                "setup_id": setup.setup_id,
                "setup_type": setup.setup_type,
                "entry_model": setup.entry_model,
                "grade": setup.grade,
                "score": setup.score,
                "ttl_hours": setup.entry_ttl_hours,
            },
        )
    )
    events.append(
        _event(
            strategy_id,
            "ENTRY_REQUESTED",
            ts,
            symbol,
            {
                "setup_id": setup.setup_id,
                "direction": int(setup.direction),
                "qty": setup.qty,
                "entry_price": setup.entry_price,
                "stop_price": setup.stop_price,
                "score": setup.score,
                "setup_type": setup.setup_type,
                "entry_model": setup.entry_model,
            },
        )
    )
    _update_last_decision(next_state, events)
    return next_state, actions, events


def default_manage_position(
    state: ETFCoreState,
    bar_input: ETFBarInput,
    cfg: Any,
    position: ETFPosition,
) -> list[Any]:
    bar = bar_input.bar_15m
    if bar is None:
        return []
    actions: list[Any] = []
    direction = position.direction
    risk = max(position.risk_per_share, 1e-9)
    close = bar.close
    current_r = _current_r(position, close)
    t1_r = float(position.meta.get("t1_r", getattr(cfg, "t1_r", 1.5)))
    t2_r = float(position.meta.get("t2_r", getattr(cfg, "t2_r", 2.5)))
    t1_hit = close >= position.entry_price + t1_r * risk if direction == Direction.LONG else close <= position.entry_price - t1_r * risk
    t2_hit = close >= position.entry_price + t2_r * risk if direction == Direction.LONG else close <= position.entry_price - t2_r * risk

    if not position.t1_done and t1_hit and "t1" not in position.pending_exit_roles:
        qty = _target_qty(
            position.qty_open,
            float(position.meta.get("t1_partial_pct", getattr(cfg, "t1_partial_pct", 0.4))),
            bool(position.meta.get("exit_all_at_t1", False)),
        )
        if qty > 0:
            position.pending_exit_roles.add("t1")
            actions.append(
                SubmitPartialExit(
                    client_order_id=f"{position.setup_id}-t1",
                    symbol=position.symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=qty,
                    order_type="MARKET",
                    metadata={"reason": "T1", "setup_id": position.setup_id},
                )
            )
            if position.stop_order_id:
                be = position.entry_price
                if (direction == Direction.LONG and be > position.current_stop) or (
                    direction == Direction.SHORT and be < position.current_stop
                ):
                    position.current_stop = be
                    actions.append(
                        ReplaceProtectiveStop(
                            symbol=position.symbol,
                            target_order_id=position.stop_order_id,
                            side="SELL" if direction == Direction.LONG else "BUY",
                            stop_price=be,
                            qty=position.qty_open,
                            reason="t1_breakeven",
                        )
                    )

    if position.t1_done and not position.t2_done and t2_hit and "t2" not in position.pending_exit_roles:
        qty = _partial_qty(position.qty_open, float(position.meta.get("t2_partial_pct", getattr(cfg, "t2_partial_pct", 0.3))))
        if qty > 0:
            position.pending_exit_roles.add("t2")
            actions.append(
                SubmitPartialExit(
                    client_order_id=f"{position.setup_id}-t2",
                    symbol=position.symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=qty,
                    order_type="MARKET",
                    metadata={"reason": "T2", "setup_id": position.setup_id},
                )
            )

    floor_trigger = float(position.meta.get("mfe_floor_trigger_r", 0.0) or 0.0)
    floor_lock = float(position.meta.get("mfe_floor_lock_r", 0.0) or 0.0)
    if floor_trigger > 0 and position.mfe_r >= floor_trigger and position.stop_order_id:
        floor_stop = (
            position.entry_price + floor_lock * risk
            if direction == Direction.LONG
            else position.entry_price - floor_lock * risk
        )
        improves_stop = (
            direction == Direction.LONG and floor_stop > position.current_stop
        ) or (
            direction == Direction.SHORT and floor_stop < position.current_stop
        )
        if improves_stop:
            position.current_stop = floor_stop
            actions.append(
                ReplaceProtectiveStop(
                    symbol=position.symbol,
                    target_order_id=position.stop_order_id,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    stop_price=floor_stop,
                    qty=position.qty_open,
                    reason="mfe_floor",
                )
            )

    _apply_structure_trail(actions, position, bar_input, cfg)

    giveback_trigger = float(position.meta.get("mfe_giveback_trigger_r", 0.0) or 0.0)
    if giveback_trigger > 0 and position.mfe_r >= giveback_trigger and not _has_flatten(actions):
        after_t1_only = bool(position.meta.get("mfe_giveback_after_t1_only", getattr(cfg, "mfe_giveback_after_t1_only", False)))
        if after_t1_only and not position.t1_done:
            giveback_trigger = 0.0
    if giveback_trigger > 0 and position.mfe_r >= giveback_trigger and not _has_flatten(actions):
        retain_frac = min(max(float(position.meta.get("mfe_giveback_retain_frac", 0.50) or 0.0), 0.0), 1.0)
        lock_r = float(position.meta.get("mfe_giveback_lock_r", 0.0) or 0.0)
        giveback_floor_r = max(lock_r, position.mfe_r * retain_frac)
        if current_r <= giveback_floor_r:
            actions.append(
                FlattenPosition(
                    symbol=position.symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=position.qty_open,
                    reason="MFE_GIVEBACK",
                    metadata={
                        "setup_id": position.setup_id,
                        "mfe_r": position.mfe_r,
                        "current_r": current_r,
                        "floor_r": giveback_floor_r,
                    },
                )
            )

    _maybe_submit_continuation_addon(actions, position, bar_input, cfg, current_r)

    max_hold = int(position.meta.get("max_hold_bars_15m", 0) or 0)
    min_mfe = float(position.meta.get("time_stop_min_mfe_r", 0.0) or 0.0)
    if max_hold > 0 and position.bars_held_15m >= max_hold and not position.t1_done and position.mfe_r < min_mfe:
        actions.append(
            FlattenPosition(
                symbol=position.symbol,
                side="SELL" if direction == Direction.LONG else "BUY",
                qty=position.qty_open,
                reason="TIME_STOP",
                metadata={"setup_id": position.setup_id},
            )
        )
    early_failure_bars = int(position.meta.get("early_failure_exit_bars_15m", getattr(cfg, "early_failure_exit_bars_15m", 0)) or 0)
    if early_failure_bars > 0 and position.bars_held_15m >= early_failure_bars and not position.t1_done and not _has_flatten(actions):
        max_mfe = float(position.meta.get("early_failure_max_mfe_r", getattr(cfg, "early_failure_max_mfe_r", 0.35)) or 0.0)
        max_current = float(
            position.meta.get("early_failure_max_current_r", getattr(cfg, "early_failure_max_current_r", -0.25)) or 0.0
        )
        if position.mfe_r <= max_mfe and current_r <= max_current:
            actions.append(
                FlattenPosition(
                    symbol=position.symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=position.qty_open,
                    reason="EARLY_FAILURE",
                    metadata={
                        "setup_id": position.setup_id,
                        "mfe_r": position.mfe_r,
                        "current_r": current_r,
                    },
                )
            )
    if bool(position.meta.get("exit_on_vwap_reloss", False)) and not _has_flatten(actions):
        after_t1_only = bool(position.meta.get("vwap_reloss_after_t1_only", True))
        min_reloss_mfe = float(position.meta.get("vwap_reloss_min_mfe_r", 0.75) or 0.0)
        reloss_grace = int(position.meta.get("vwap_reloss_grace_bars", 4) or 0)
        target_price = float(position.meta.get("target_price", 0.0) or 0.0)
        reloss_armed = position.mfe_r >= min_reloss_mfe and position.bars_held_15m >= reloss_grace
        if target_price > 0 and reloss_armed and (position.t1_done or not after_t1_only):
            relost = (
                (direction == Direction.LONG and close < target_price)
                or (direction == Direction.SHORT and close > target_price)
            )
            if relost:
                actions.append(
                    FlattenPosition(
                        symbol=position.symbol,
                        side="SELL" if direction == Direction.LONG else "BUY",
                        qty=position.qty_open,
                        reason="VWAP_RELOSS",
                        metadata={"setup_id": position.setup_id},
                    )
                )
    runner_max_hold = int(position.meta.get("runner_max_hold_bars_15m", 0) or 0)
    if bool(position.meta.get("runner_requires_vwap_acceptance", False)) and position.t1_done and not _has_flatten(actions):
        grace = int(position.meta.get("runner_acceptance_grace_bars", 2) or 0)
        t1_bar = int(position.meta.get("t1_done_bars_15m", position.bars_held_15m) or 0)
        bars_since_t1 = max(0, position.bars_held_15m - t1_bar)
        target_price = float(position.meta.get("target_price", 0.0) or 0.0)
        atr15 = float(bar_input.indicators.get("atr_15m", 0.0) or 0.0)
        close_offset = max(float(position.meta.get("runner_acceptance_close_atr15", 0.0) or 0.0) * atr15, 0.0)
        accepted = (
            (direction == Direction.LONG and close >= target_price + close_offset)
            or (direction == Direction.SHORT and close <= target_price - close_offset)
        )
        if target_price > 0 and bars_since_t1 >= grace and not accepted:
            actions.append(
                FlattenPosition(
                    symbol=position.symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=position.qty_open,
                    reason="VWAP_ACCEPTANCE_FAIL",
                    metadata={"setup_id": position.setup_id},
                )
            )
    if runner_max_hold > 0 and position.t1_done and position.bars_held_15m >= runner_max_hold and not _has_flatten(actions):
        actions.append(
            FlattenPosition(
                symbol=position.symbol,
                side="SELL" if direction == Direction.LONG else "BUY",
                qty=position.qty_open,
                reason="RUNNER_TIME_STOP",
                metadata={"setup_id": position.setup_id},
            )
        )
    return actions


def _current_r(position: ETFPosition, close: float) -> float:
    risk = max(position.risk_per_share, 1e-9)
    if position.direction == Direction.LONG:
        return (close - position.entry_price) / risk
    return (position.entry_price - close) / risk


def _apply_structure_trail(
    actions: list[Any],
    position: ETFPosition,
    bar_input: ETFBarInput,
    cfg: Any,
) -> None:
    if _has_flatten(actions) or not position.stop_order_id:
        return
    if position.t2_done:
        lookback = int(position.meta.get("structure_trail_after_t2_1h_bars", getattr(cfg, "structure_trail_after_t2_1h_bars", 0)) or 0)
        window = bar_input.bars_1h
    elif position.t1_done:
        lookback = int(position.meta.get("structure_trail_after_t1_30m_bars", getattr(cfg, "structure_trail_after_t1_30m_bars", 0)) or 0)
        window = bar_input.bars_30m
    else:
        lookback = 0
        window = None
    if window is None or lookback < 2 or len(window) < 2:
        return
    lookback = min(lookback, len(window))
    if position.direction == Direction.LONG:
        stop = float(np.nanmin(window.lows[-lookback:]))
    else:
        stop = float(np.nanmax(window.highs[-lookback:]))
    if not np.isfinite(stop):
        return
    if bool(position.meta.get("structure_trail_use_vwap_after_t1", getattr(cfg, "structure_trail_use_vwap_after_t1", False))):
        vwap = float(bar_input.indicators.get("vwap_30m", np.nan))
        if np.isfinite(vwap):
            stop = max(stop, vwap) if position.direction == Direction.LONG else min(stop, vwap)
    lock_r = float(position.meta.get("structure_trail_min_lock_r", getattr(cfg, "structure_trail_min_lock_r", 0.0)) or 0.0)
    if position.t1_done and lock_r > 0:
        risk = max(position.risk_per_share, 1e-9)
        lock_stop = position.entry_price + lock_r * risk if position.direction == Direction.LONG else position.entry_price - lock_r * risk
        stop = max(stop, lock_stop) if position.direction == Direction.LONG else min(stop, lock_stop)
    _raise_stop_to_price(actions, position, stop, reason="structure_trail")


def _maybe_submit_continuation_addon(
    actions: list[Any],
    position: ETFPosition,
    bar_input: ETFBarInput,
    cfg: Any,
    current_r: float,
) -> None:
    if _has_flatten(actions) or not bool(position.meta.get("continuation_addon_enabled", getattr(cfg, "continuation_addon_enabled", False))):
        return
    if position.meta.get("addon_done") or position.meta.get("addon_pending") or position.qty_open <= 0:
        return
    if "t2" in position.pending_exit_roles:
        return
    requires_t1 = bool(position.meta.get("continuation_addon_requires_t1", getattr(cfg, "continuation_addon_requires_t1", True)))
    if requires_t1 and not position.t1_done:
        return
    trigger_r = float(position.meta.get("continuation_addon_trigger_r", getattr(cfg, "continuation_addon_trigger_r", 0.0)) or 0.0)
    if trigger_r <= 0 or current_r < trigger_r or position.mfe_r < trigger_r:
        return
    min_score = float(position.meta.get("continuation_addon_min_score", getattr(cfg, "continuation_addon_min_score", 0.0)) or 0.0)
    if position.score < min_score:
        return
    if not _continuation_addon_confirmation_holds(position, bar_input, cfg):
        return
    qty = _continuation_addon_qty(position, bar_input, cfg)
    if qty <= 0:
        return
    bar = bar_input.bar_15m
    if bar is None:
        return
    position.meta["addon_pending"] = True
    actions.append(
        SubmitAddOnEntry(
            client_order_id=f"{position.setup_id}-addon-{position.bars_held_15m}",
            symbol=position.symbol,
            side="BUY" if position.direction == Direction.LONG else "SELL",
            qty=qty,
            order_type="MARKET",
            price=bar.close,
            risk_context={
                "stop_for_risk": position.current_stop,
                "current_r": current_r,
                "mfe_r": position.mfe_r,
            },
            metadata={
                "setup_id": position.setup_id,
                "reason": "ACCEPTED_CONTINUATION_ADDON",
                "score": position.score,
                "current_r": current_r,
                "mfe_r": position.mfe_r,
            },
        )
    )


def _continuation_addon_confirmation_holds(position: ETFPosition, bar_input: ETFBarInput, cfg: Any) -> bool:
    bar = bar_input.bar_15m
    if bar is None:
        return False
    direction = position.direction
    if bool(
        position.meta.get(
            "continuation_addon_require_vwap_acceptance",
            getattr(cfg, "continuation_addon_require_vwap_acceptance", True),
        )
    ):
        target_price = float(position.meta.get("target_price", 0.0) or 0.0)
        atr15 = float(bar_input.indicators.get("atr_15m", 0.0) or 0.0)
        offset = max(
            float(
                position.meta.get(
                    "continuation_addon_acceptance_atr15",
                    getattr(cfg, "continuation_addon_acceptance_atr15", 0.03),
                )
                or 0.0
            )
            * atr15,
            0.0,
        )
        if target_price <= 0:
            return False
        if direction == Direction.LONG and bar.close < target_price + offset:
            return False
        if direction == Direction.SHORT and bar.close > target_price - offset:
            return False
    if bool(position.meta.get("continuation_addon_require_ema20_15m", getattr(cfg, "continuation_addon_require_ema20_15m", True))):
        ema20 = float(bar_input.indicators.get("ema20_15m", np.nan))
        if not np.isfinite(ema20):
            return False
        if direction == Direction.LONG and bar.close <= ema20:
            return False
        if direction == Direction.SHORT and bar.close >= ema20:
            return False
    return True


def _continuation_addon_qty(position: ETFPosition, bar_input: ETFBarInput, cfg: Any) -> int:
    bar = bar_input.bar_15m
    if bar is None or bar.close <= 0:
        return 0
    size_mult = max(
        float(position.meta.get("continuation_addon_size_mult", getattr(cfg, "continuation_addon_size_mult", 0.0)) or 0.0),
        0.0,
    )
    if size_mult <= 0:
        return 0
    qty = max(1, int(round(position.qty_initial * size_mult)))
    cap_pct = float(
        position.meta.get("continuation_addon_max_notional_pct", getattr(cfg, "continuation_addon_max_notional_pct", 0.0)) or 0.0
    )
    if cap_pct <= 0:
        cap_pct = float(getattr(cfg, "max_position_notional_pct", 0.0) or 0.0)
    if cap_pct > 0 and bar_input.equity > 0:
        notional_room = max(0.0, bar_input.equity * cap_pct - position.qty_open * bar.close)
        qty = min(qty, int(notional_room // bar.close))
    risk_cap_pct = float(
        position.meta.get(
            "continuation_addon_max_total_risk_pct",
            getattr(cfg, "continuation_addon_max_total_risk_pct", 0.0),
        )
        or 0.0
    )
    if risk_cap_pct > 0 and bar_input.equity > 0:
        risk_per_share = max(abs(bar.close - position.current_stop), 1e-9)
        current_risk = max(0.0, position.qty_open * risk_per_share)
        risk_room = max(0.0, bar_input.equity * risk_cap_pct - current_risk)
        qty = min(qty, int(risk_room // risk_per_share))
    return max(qty, 0)


def on_fill_common(
    state: ETFCoreState,
    fill: ETFFill,
    *,
    strategy_id: str,
) -> tuple[ETFCoreState, list[Any], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[Any] = []
    events: list[DecisionEvent] = []
    ts = fill.fill_time or datetime.now(timezone.utc)
    role = fill.order_role.lower()
    setup = next_state.pending_orders.pop(fill.oms_order_id, None)
    symbol = fill.symbol or (setup.symbol if setup else "")

    if setup is not None or role == "entry":
        if setup is None:
            events.append(_event(strategy_id, fill.decision_code or "UNMATCHED_ENTRY_FILL", ts, symbol, fill.decision_details))
            _update_last_decision(next_state, events, preserve_last_bar_ts=True)
            return next_state, actions, events
        qty = int(fill.fill_qty or setup.qty)
        price = float(fill.fill_price or setup.entry_price)
        stop_order_id = f"{setup.symbol}-stop-{setup.setup_id}"
        pos = ETFPosition(
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            direction=setup.direction,
            qty_open=qty,
            qty_initial=qty,
            entry_price=price,
            current_stop=setup.stop_price,
            initial_stop=setup.stop_price,
            entry_ts=ts,
            risk_per_share=max(abs(price - setup.stop_price), 1e-9),
            setup_type=setup.setup_type,
            grade=setup.grade,
            entry_model=setup.entry_model,
            score=setup.score,
            stop_order_id=stop_order_id,
            mfe_price=price,
            mae_price=price,
            meta={
                **setup.meta,
                "t1_r": setup.t1_r,
                "t1_partial_pct": setup.t1_partial_pct,
                "t2_r": setup.t2_r,
                "t2_partial_pct": setup.t2_partial_pct,
                "target_price": setup.target_price,
                "t2_price": setup.t2_price,
                "max_hold_bars_15m": setup.max_hold_bars_15m,
            },
        )
        next_state.positions[setup.symbol] = pos
        actions.append(
            SubmitProtectiveStop(
                client_order_id=stop_order_id,
                symbol=setup.symbol,
                side="SELL" if setup.direction == Direction.LONG else "BUY",
                qty=qty,
                stop_price=setup.stop_price,
                metadata={"setup_id": setup.setup_id},
            )
        )
        if setup.target_price > 0:
            target_qty = _target_qty(qty, setup.t1_partial_pct, bool(setup.meta.get("exit_all_at_t1", False)))
            target_is_favorable = (
                (setup.direction == Direction.LONG and setup.target_price > price)
                or (setup.direction == Direction.SHORT and setup.target_price < price)
            )
            if target_qty > 0 and target_is_favorable:
                pos.pending_exit_roles.add("t1")
                actions.append(
                    SubmitProfitTarget(
                        client_order_id=f"{setup.setup_id}-t1",
                        symbol=setup.symbol,
                        side="SELL" if setup.direction == Direction.LONG else "BUY",
                        qty=target_qty,
                        limit_price=setup.target_price,
                        metadata={"reason": "T1", "setup_id": setup.setup_id},
                    )
                )
        events.append(
            _event(strategy_id, "ENTRY_FILLED", ts, setup.symbol, {"setup_id": setup.setup_id, "qty": qty, "price": price})
        )
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    position = next_state.positions.get(symbol)
    if position is None:
        if fill.decision_code:
            events.append(_event(strategy_id, fill.decision_code, ts, symbol, fill.decision_details))
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    if role in {"add_on_entry", "addon", "addon_entry"}:
        qty = int(fill.fill_qty or 0)
        if qty <= 0:
            events.append(_event(strategy_id, "ADDON_FILL_IGNORED", ts, symbol, {"reason": "invalid_qty"}))
            _update_last_decision(next_state, events, preserve_last_bar_ts=True)
            return next_state, actions, events
        position.qty_open += qty
        position.qty_initial += qty
        position.meta["addon_done"] = True
        position.meta["addon_pending"] = False
        if position.stop_order_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=symbol,
                    target_order_id=position.stop_order_id,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    stop_price=position.current_stop,
                    qty=position.qty_open,
                    reason="addon_resize",
                )
            )
        events.append(
            _event(
                strategy_id,
                "ADDON_FILLED",
                ts,
                symbol,
                {"qty": qty, "price": fill.fill_price, "setup_id": position.setup_id},
            )
        )
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    qty = min(int(fill.fill_qty or position.qty_open), position.qty_open)
    if role in {"partial", "partial_exit", "profit_target", "target", "t1"}:
        position.qty_open -= qty
        reason = fill.exit_type or fill.decision_details.get("reason", "")
        if reason.upper() == "T1":
            position.t1_done = True
            position.meta["t1_done_bars_15m"] = position.bars_held_15m
            position.pending_exit_roles.discard("t1")
            if bool(position.meta.get("promote_stop_on_t1", False)):
                stop_offset_r = float(position.meta.get("t1_stop_offset_r", 0.0) or 0.0)
                be_stop = (
                    position.entry_price + stop_offset_r * position.risk_per_share
                    if position.direction == Direction.LONG
                    else position.entry_price - stop_offset_r * position.risk_per_share
                )
                improves_stop = (
                    position.direction == Direction.LONG and be_stop > position.current_stop
                ) or (
                    position.direction == Direction.SHORT and be_stop < position.current_stop
                )
                if improves_stop:
                    position.current_stop = be_stop
        if reason.upper() == "T2":
            position.t2_done = True
            position.runner_active = position.qty_open > 0
            position.pending_exit_roles.discard("t2")
        if position.qty_open <= 0:
            next_state.positions.pop(symbol, None)
            code = "EXIT_FILLED"
        else:
            code = "PARTIAL_EXIT_FILLED"
            if position.stop_order_id:
                actions.append(
                    ReplaceProtectiveStop(
                        symbol=symbol,
                        target_order_id=position.stop_order_id,
                        side="SELL" if position.direction == Direction.LONG else "BUY",
                        stop_price=position.current_stop,
                        qty=position.qty_open,
                        reason="partial_resize",
                    )
                )
        events.append(_event(strategy_id, code, ts, symbol, {"qty": qty, "price": fill.fill_price, "reason": reason}))
    else:
        next_state.positions.pop(symbol, None)
        code = "STOP_FILLED" if role == "stop" or (fill.exit_type or "").upper() == "STOP" else "EXIT_FILLED"
        events.append(_event(strategy_id, code, ts, symbol, {"qty": qty, "price": fill.fill_price, "reason": fill.exit_type or role}))

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, actions, events


def on_order_update_common(
    state: ETFCoreState,
    update: ETFOrderUpdate,
    *,
    strategy_id: str,
) -> tuple[ETFCoreState, list[Any], list[DecisionEvent]]:
    next_state = deepcopy(state)
    events: list[DecisionEvent] = []
    status = update.status.lower()
    ts = update.timestamp or datetime.now(timezone.utc)
    if status in {"cancelled", "expired", "rejected", "order_cancelled", "order_expired", "order_rejected"}:
        role = str(update.order_role or "").lower()
        if role in {"add_on_entry", "addon", "addon_entry"} and update.symbol in next_state.positions:
            next_state.positions[update.symbol].meta["addon_pending"] = False
            events.append(
                _event(
                    strategy_id,
                    "ADDON_ORDER_TERMINAL",
                    ts,
                    update.symbol,
                    {"status": status, "setup_id": next_state.positions[update.symbol].setup_id},
                )
            )
            _update_last_decision(next_state, events, preserve_last_bar_ts=True)
            return next_state, [], events
        setup = next_state.pending_orders.pop(update.oms_order_id, None)
        if setup is not None:
            events.append(_event(strategy_id, "ORDER_TERMINAL", ts, setup.symbol, {"status": status, "setup_id": setup.setup_id}))
    elif update.decision_code:
        events.append(_event(strategy_id, update.decision_code, ts, update.symbol, update.decision_details))
    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, [], events


def _mark_position(position: ETFPosition, bar: BarData) -> None:
    position.bars_held_15m += 1
    if position.direction == Direction.LONG:
        position.mfe_price = max(position.mfe_price or position.entry_price, bar.high)
        position.mae_price = min(position.mae_price or position.entry_price, bar.low)
    else:
        position.mfe_price = min(position.mfe_price or position.entry_price, bar.low)
        position.mae_price = max(position.mae_price or position.entry_price, bar.high)


def _partial_qty(qty_open: int, frac: float) -> int:
    if qty_open <= 1:
        return qty_open
    qty = max(1, int(round(qty_open * frac)))
    return min(qty, qty_open - 1)


def _target_qty(qty_open: int, frac: float, exit_all: bool = False) -> int:
    if exit_all:
        return max(0, qty_open)
    return _partial_qty(qty_open, frac)


def _has_flatten(actions: list[Any]) -> bool:
    return any(isinstance(action, FlattenPosition) for action in actions)


def _raise_stop_to_price(actions: list[Any], position: ETFPosition, stop: float, *, reason: str) -> None:
    if not np.isfinite(stop) or not position.stop_order_id:
        return
    improves_stop = (
        position.direction == Direction.LONG and stop > position.current_stop
    ) or (
        position.direction == Direction.SHORT and stop < position.current_stop
    )
    if not improves_stop:
        return
    position.current_stop = float(stop)
    actions.append(
        ReplaceProtectiveStop(
            symbol=position.symbol,
            target_order_id=position.stop_order_id,
            side="SELL" if position.direction == Direction.LONG else "BUY",
            stop_price=float(stop),
            qty=position.qty_open,
            reason=reason,
        )
    )


def _event(strategy_id: str, code: str, ts: datetime, symbol: str, details: dict[str, Any]) -> DecisionEvent:
    return DecisionEvent(
        code=code,
        ts=ts,
        symbol=symbol,
        timeframe="15m",
        details=dict(details),
        strategy_id=strategy_id,
    )


def _update_last_decision(
    state: ETFCoreState,
    events: list[DecisionEvent],
    *,
    preserve_last_bar_ts: bool = False,
) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    if latest.ts is not None and not preserve_last_bar_ts:
        state.last_bar_ts = latest.ts
