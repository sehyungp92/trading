"""Pure Downturn entry signal, gate, sizing, stop, and proposal decisions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from strategies.core.actions import SubmitEntry
from strategies.momentum.downturn.models import (
    CompositeRegime,
    EngineTag,
    FadeSignal,
    FadeState,
    VolState,
)
from strategies.momentum.downturn.regime import regime_sizing_mult
from strategies.momentum.downturn.signals import (
    compute_entry_subtype_stop,
    detect_fade_short,
    detect_momentum_impulse,
)
from strategies.momentum.downturn.stops import compute_tiered_tp_schedule


@dataclass(frozen=True, slots=True)
class DownturnEntryGateInput:
    circuit_breaker_enabled: bool
    circuit_breaker_tripped: bool
    shock_block_enabled: bool
    vol_state: VolState
    entry_windows_enabled: bool
    session_allowed: bool
    dead_zones_enabled: bool
    minute_et: int
    directional_entry_caps: bool
    daily_trades: int
    max_daily_entries: int
    news_blackout_enabled: bool
    friction_gate_enabled: bool
    atr_daily: float
    atr_daily_percentile: float
    friction_min_percentile: float
    vol_percentile_gate: float
    bear_conviction: float
    regime_confidence_gate: float
    block_counter_regime: bool
    composite_regime: CompositeRegime
    allow_reversal_in_correction: bool
    in_correction: bool


@dataclass(frozen=True, slots=True)
class DownturnSignalSelection:
    engine_tag: EngineTag
    signal_class: str
    signal: Any


@dataclass(frozen=True, slots=True)
class DownturnProposalPolicy:
    trigger_low_buffer_ticks: float = 2.0
    entry_limit_offset_ticks: float = 4.0
    entry_ttl_bars: int = 72
    max_contracts: int = 0
    max_notional_leverage: float = 20.0
    non_correction_penalty_enabled: bool = False


@dataclass(frozen=True, slots=True)
class DownturnEntryProposal:
    client_order_id: str
    symbol: str
    engine_tag: EngineTag
    signal_class: str
    qty: int
    entry_price: float
    stop0: float
    order_type: str
    price: float | None
    limit_price: float | None
    stop_price: float
    submitted_bar_idx: int
    ttl_bars: int
    composite_regime: CompositeRegime
    vol_state: VolState
    in_correction: bool
    predator: bool
    tp_schedule: tuple[tuple[float, float], ...]
    signal_strength: float
    risk_dollars: float
    signal_id: str
    bar_id: str
    exchange_timestamp: datetime
    tif: str = "DAY"
    side: str = "SELL"


def evaluate_entry_gate(gate: DownturnEntryGateInput) -> str | None:
    """Evaluate the common entry gate prefix from explicit wrapper policy."""

    if gate.circuit_breaker_enabled and gate.circuit_breaker_tripped:
        return "circuit_breaker"
    if gate.shock_block_enabled and gate.vol_state == VolState.SHOCK:
        return "vol_shock"
    if gate.dead_zones_enabled and (
        565 <= gate.minute_et < 575 or 950 <= gate.minute_et < 960
    ):
        return "dead_zone"
    if gate.entry_windows_enabled and not gate.session_allowed:
        return "session_window"
    if gate.directional_entry_caps and gate.daily_trades >= gate.max_daily_entries:
        return "daily_cap"
    if gate.news_blackout_enabled and (
        570 <= gate.minute_et < 575 or 955 <= gate.minute_et < 960
    ):
        return "news_blackout"
    if (
        gate.friction_gate_enabled
        and gate.atr_daily > 0
        and gate.atr_daily_percentile < gate.friction_min_percentile
    ):
        return "friction_gate"
    if (
        gate.vol_percentile_gate > 0
        and gate.atr_daily_percentile * 100.0 < gate.vol_percentile_gate
    ):
        return "vol_percentile_gate"
    if (
        gate.regime_confidence_gate > 0
        and gate.bear_conviction < gate.regime_confidence_gate
    ):
        return "regime_confidence_gate"
    if (
        gate.block_counter_regime
        and gate.composite_regime == CompositeRegime.COUNTER
        and not (gate.allow_reversal_in_correction and gate.in_correction)
    ):
        return "counter_regime"
    return None


def select_fade_signal(
    *,
    fade_state: FadeState,
    close_15m: float,
    high_15m_recent: np.ndarray,
    closes_15m: Sequence[float],
    effective_regime: CompositeRegime,
    mom_slope_ok: bool,
    extension_short: bool,
    atr_15m: float,
    session_type: str,
    ema_fast_15m: float,
    bars_since_last_entry: int,
    flags: Any,
    param_overrides: dict[str, float],
    evaluate_fade: bool,
    evaluate_momentum: bool,
    momentum_reference_close: float | None = None,
) -> DownturnSignalSelection | None:
    """Select the shared Fade/momentum signal without wrapper side effects."""

    if evaluate_fade:
        signal = detect_fade_short(
            fade_state,
            close_15m,
            high_15m_recent,
            effective_regime,
            mom_slope_ok,
            extension_short,
            atr_15m,
            session_type,
            flags,
            param_overrides,
        )
        if signal is not None:
            return DownturnSignalSelection(
                engine_tag=EngineTag.FADE,
                signal_class="vwap_rejection",
                signal=signal,
            )

    cooldown = int(param_overrides.get("momentum_cooldown_bars", 36))
    if (
        evaluate_momentum
        and bars_since_last_entry >= cooldown
        and (momentum_reference_close is not None or len(closes_15m) > 5)
    ):
        close_5ago = (
            float(momentum_reference_close)
            if momentum_reference_close is not None
            else float(closes_15m[-6])
        )
        roc_5bar = (
            (close_15m - close_5ago) / close_5ago if close_5ago > 0 else 0.0
        )
        if detect_momentum_impulse(
            close_15m,
            ema_fast_15m,
            roc_5bar,
            effective_regime,
            param_overrides,
        ):
            return DownturnSignalSelection(
                engine_tag=EngineTag.FADE,
                signal_class="momentum_impulse",
                signal=FadeSignal(
                    vwap_used=fade_state.vwap_used,
                    rejection_close=close_15m,
                    class_mult=0.70,
                    predator_present=False,
                ),
            )
    return None


def selection_from_signal(
    signal: Any,
    *,
    momentum_impulse: bool = False,
) -> DownturnSignalSelection:
    """Classify a reversal/breakdown signal selected by wrapper-owned policy."""

    name = type(signal).__name__
    if name == "BreakdownSignal":
        return DownturnSignalSelection(EngineTag.BREAKDOWN, "box_breakdown", signal)
    if name == "ReversalSignal":
        return DownturnSignalSelection(EngineTag.REVERSAL, "classic_divergence", signal)
    return DownturnSignalSelection(
        EngineTag.FADE,
        "momentum_impulse" if momentum_impulse else "vwap_rejection",
        signal,
    )


def build_entry_proposal(
    *,
    selection: DownturnSignalSelection,
    client_order_id: str,
    symbol: str,
    bar_idx: int,
    bar_ts: datetime,
    close: float,
    atr_1h: float,
    atr_30m: float,
    equity: float,
    notional_equity: float,
    tick_size: float,
    point_value: float,
    composite_regime: CompositeRegime,
    vol_state: VolState,
    vol_factor: float,
    strong_bear: bool,
    in_correction: bool,
    flags: Any,
    param_overrides: dict[str, float],
    policy: DownturnProposalPolicy,
    signal_id: str = "",
    bar_id: str = "",
) -> tuple[DownturnEntryProposal | None, str]:
    """Build one immutable proposal while preserving wrapper policy inputs."""

    if close <= 0 or tick_size <= 0 or point_value <= 0:
        return None, "invalid_market_or_instrument"
    tag = selection.engine_tag
    atr = atr_30m if tag == EngineTag.BREAKDOWN else atr_1h
    trigger_ticks = max(0.0, float(policy.trigger_low_buffer_ticks))
    low_recent = close - trigger_ticks * tick_size
    entry_price, stop0, entry_type = compute_entry_subtype_stop(
        tag,
        selection.signal,
        close,
        atr,
        low_recent,
        tick_size,
        param_overrides,
    )
    if entry_price <= 0 or stop0 <= 0:
        return None, "invalid_entry_or_stop"
    risk_per_contract = abs(stop0 - entry_price) * point_value
    if risk_per_contract <= 0:
        return None, "zero_risk"

    base_risk = float(param_overrides.get("base_risk_pct", 0.01))
    risk_budget = (
        equity
        * base_risk
        * regime_sizing_mult(composite_regime, param_overrides)
        * (vol_factor if flags.use_volatility_states else 1.0)
        * (1.25 if strong_bear and flags.use_strong_bear_bonus else 1.0)
    )
    if in_correction and flags.correction_sizing_bonus:
        risk_budget *= float(param_overrides.get("correction_sizing_mult", 1.30))
    if (
        not in_correction
        and policy.non_correction_penalty_enabled
        and flags.non_correction_penalty
    ):
        risk_budget *= float(param_overrides.get("non_correction_sizing_mult", 0.60))

    qty = max(1, int(risk_budget / risk_per_contract))
    if policy.max_contracts > 0:
        qty = min(qty, int(policy.max_contracts))
    if policy.max_notional_leverage > 0:
        notional_per = entry_price * point_value
        if notional_per > 0:
            max_qty = max(
                1,
                int(notional_equity * policy.max_notional_leverage / notional_per),
            )
            qty = min(qty, max_qty)

    order_type = "STOP" if entry_type == "stop_market" else "STOP_LIMIT"
    limit_price = None
    price = None
    if order_type == "STOP_LIMIT":
        price = entry_price
        limit_price = entry_price - max(
            0.0,
            float(policy.entry_limit_offset_ticks),
        ) * tick_size
    resolved_signal_id = signal_id or (
        f"{symbol}:{tag.value}:{selection.signal_class}:{bar_ts.isoformat()}"
    )
    resolved_bar_id = bar_id or f"{symbol}:5m:{bar_ts.isoformat()}"
    risk_dollars = risk_per_contract * qty
    return DownturnEntryProposal(
        client_order_id=client_order_id,
        symbol=symbol,
        engine_tag=tag,
        signal_class=selection.signal_class,
        qty=qty,
        entry_price=entry_price,
        stop0=stop0,
        order_type=order_type,
        price=price,
        limit_price=limit_price,
        stop_price=entry_price,
        submitted_bar_idx=bar_idx,
        ttl_bars=max(1, int(policy.entry_ttl_bars)),
        composite_regime=composite_regime,
        vol_state=vol_state,
        in_correction=in_correction,
        predator=bool(getattr(selection.signal, "predator_present", False)),
        tp_schedule=tuple(compute_tiered_tp_schedule(tag, composite_regime, param_overrides)),
        signal_strength=float(getattr(selection.signal, "class_mult", 0.5)),
        risk_dollars=risk_dollars,
        signal_id=resolved_signal_id,
        bar_id=resolved_bar_id,
        exchange_timestamp=bar_ts,
    ), ""


def with_quantity(proposal: DownturnEntryProposal, qty: int) -> DownturnEntryProposal:
    qty = max(0, int(qty))
    per_contract = proposal.risk_dollars / proposal.qty if proposal.qty > 0 else 0.0
    return replace(proposal, qty=qty, risk_dollars=per_contract * qty)


def entry_action(proposal: DownturnEntryProposal) -> SubmitEntry:
    return SubmitEntry(
        client_order_id=proposal.client_order_id,
        symbol=proposal.symbol,
        side=proposal.side,
        qty=proposal.qty,
        order_type=proposal.order_type,
        tif=proposal.tif,
        price=proposal.price,
        limit_price=proposal.limit_price,
        stop_price=proposal.stop_price,
        role="entry",
        risk_context={
            "stop_for_risk": proposal.stop0,
            "planned_entry_price": proposal.entry_price,
            "risk_dollars": proposal.risk_dollars,
            "signal_id": proposal.signal_id,
            "bar_id": proposal.bar_id,
            "exchange_timestamp": proposal.exchange_timestamp,
        },
        metadata={
            "engine_tag": proposal.engine_tag.value,
            "signal_class": proposal.signal_class,
            "role": "entry",
            "ttl_bars": proposal.ttl_bars,
        },
    )


def entry_request(proposal: DownturnEntryProposal):
    """Adapt an immutable proposal to the established lifecycle request."""

    from .state import DownturnEntryRequest

    return DownturnEntryRequest(
        client_order_id=proposal.client_order_id,
        symbol=proposal.symbol,
        engine_tag=proposal.engine_tag,
        signal_class=proposal.signal_class,
        qty=proposal.qty,
        entry_price=proposal.entry_price,
        stop0=proposal.stop0,
        tif=proposal.tif,
        order_type=proposal.order_type,
        side=proposal.side,
        price=proposal.price,
        limit_price=proposal.limit_price,
        stop_price=proposal.stop_price,
        submitted_bar_idx=proposal.submitted_bar_idx,
        ttl_bars=proposal.ttl_bars,
        composite_regime=proposal.composite_regime,
        vol_state=proposal.vol_state,
        in_correction=proposal.in_correction,
        predator=proposal.predator,
        tp_schedule=list(proposal.tp_schedule),
        signal_strength=proposal.signal_strength,
    )
