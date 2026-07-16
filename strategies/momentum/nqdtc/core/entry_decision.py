"""Pure NQDTC entry qualification, sizing, and order selection.

This is the strategy-owned portion selectively retained from the historical
G4M-v2 NQDTC extraction.  Runtime orchestration, timeline infrastructure, and
configuration materialization intentionally remain outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick
from strategies.momentum.nqdtc import config as C
from strategies.momentum.nqdtc import indicators as ind
from strategies.momentum.nqdtc import signals, sizing, stops
from strategies.momentum.nqdtc.models import (
    CompositeRegime,
    Direction,
    EntrySubtype,
    RegimeState,
    Session,
    SessionEngineState,
    WorkingOrder,
)


@dataclass(frozen=True, slots=True)
class NQDTCEntryDecisionSnapshot:
    """All mutable and raw-market inputs needed for one entry decision.

    Fields describing wrapper policy preserve the existing live/backtest
    behavior during extraction.  They are explicit inputs, not alternate
    decision implementations.
    """

    now: datetime
    symbol: str
    equity: float
    engine: SessionEngineState
    regime: RegimeState
    bars_5m: Mapping[str, np.ndarray]
    bars_15m: Mapping[str, np.ndarray]
    bars_daily: Mapping[str, np.ndarray]
    working_orders: tuple[WorkingOrder, ...] = ()
    position_open: bool = False
    cooldown_bars: int = 0
    last_fill_time: datetime | None = None
    a_fallback_eligible: bool = False
    news_blackout: bool = False
    risk_halted: bool = False
    drawdown_size_multiplier: float = 1.0
    fixed_quantity: int | None = None
    entry_oca_group: str = ""
    a_oca_group: str = ""
    entry_a_retest: bool = True
    entry_a_latch: bool = True
    entry_b_sweep: bool = True
    entry_c_standard: bool = True
    entry_c_continuation: bool = False
    continuation_mode: bool = True
    friction_gate: bool = True
    tp1_viability_gate: bool = True
    block_04_et: bool = True
    block_05_et: bool = True
    block_06_et: bool = True
    block_09_et: bool = True
    block_12_et: bool = True
    block_thursday: bool = False
    recompute_composite: bool = True
    daily_gate_requires_history: bool = False
    es_opposing: bool = False
    apply_continuation_size_multiplier: bool = True
    apply_eth_short_size_multiplier: bool = False
    c_stop_reference: Literal["entry_price", "bar_close"] = "entry_price"
    c_ttl_bars: int = 6
    fallback_order_type: Literal["MARKET", "LIMIT"] = "LIMIT"
    fallback_tif: str = "IOC"


@dataclass(frozen=True, slots=True)
class NQDTCEntryPlan:
    subtype: EntrySubtype
    direction: Direction
    order_type: Literal["MARKET", "LIMIT", "STOP_LIMIT"]
    qty: int
    stop_for_risk: float
    price: float | None = None
    stop_price: float | None = None
    tif: str = "DAY"
    oca_group: str = ""
    is_limit: bool = False
    quality_mult: float = 1.0
    disp_norm: float = 0.0
    ttl_bars: int = 6


@dataclass(frozen=True, slots=True)
class NQDTCEntryDecision:
    plans: tuple[NQDTCEntryPlan, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    composite_regime: CompositeRegime = CompositeRegime.NEUTRAL
    consume_fallback: bool = False


def evaluate_entry_decision(snapshot: NQDTCEntryDecisionSnapshot) -> NQDTCEntryDecision:
    """Evaluate NQDTC A/B/C/fallback entries from raw bars and current state."""

    blocked = _hard_gate(snapshot)
    if blocked:
        return NQDTCEntryDecision(blocked_reasons=(blocked,))

    engine = snapshot.engine
    direction = engine.breakout.direction
    if not engine.breakout.active or direction is Direction.FLAT:
        return NQDTCEntryDecision(blocked_reasons=("no_active_breakout",))
    if C.BLOCK_ETH_SHORTS and engine.session is Session.ETH and direction is Direction.SHORT:
        return NQDTCEntryDecision(blocked_reasons=("eth_short",))

    composite, daily_opposes = _composite(snapshot, direction)
    daily_ready = _daily_history_ready(snapshot)
    if (daily_ready or not snapshot.daily_gate_requires_history) and signals.regime_hard_block(
        snapshot.regime.regime_4h.value,
        snapshot.regime.trend_dir_4h,
        direction,
        daily_opposes,
    ):
        return NQDTCEntryDecision(
            blocked_reasons=("regime_hard_block",),
            composite_regime=composite,
        )
    if (
        (C.BLOCK_NEUTRAL_REGIME and composite is CompositeRegime.NEUTRAL)
        or (C.BLOCK_ALIGNED_REGIME and composite is CompositeRegime.ALIGNED)
        or (C.BLOCK_CAUTION_REGIME and composite is CompositeRegime.CAUTION)
    ):
        return NQDTCEntryDecision(
            blocked_reasons=("regime_composite_block",),
            composite_regime=composite,
        )

    disp_norm = sizing.compute_disp_norm(
        engine.last_disp_metric,
        engine.last_disp_threshold,
        engine.last_disp_threshold * 1.3 if engine.last_disp_threshold > 0 else 1.0,
    )
    quality_mult = sizing.compute_quality_mult(
        composite,
        engine.mode,
        disp_norm,
        es_opposing=snapshot.es_opposing,
    )
    final_risk_pct, _ = sizing.compute_final_risk_pct(quality_mult)
    slope_mult = _slope_multiplier(snapshot, direction)
    continuation_mult = (
        C.CONTINUATION_BREAKOUT_SIZE_MULT
        if snapshot.apply_continuation_size_multiplier
        and engine.breakout.continuation_mode
        and C.CONTINUATION_BREAKOUT_SIZE_MULT < 1.0
        else 1.0
    )
    a_risk_pct = final_risk_pct * slope_mult * continuation_mult
    other_risk_pct = a_risk_pct
    if (
        snapshot.apply_eth_short_size_multiplier
        and engine.session is Session.ETH
        and direction is Direction.SHORT
    ):
        other_risk_pct *= C.ETH_SHORT_SIZE_MULT

    risk_dollars = snapshot.equity * C.RISK_PCT
    if snapshot.friction_gate and not sizing.friction_ok(snapshot.symbol, risk_dollars):
        return NQDTCEntryDecision(
            blocked_reasons=("friction_gate",),
            composite_regime=composite,
        )
    if snapshot.tp1_viability_gate and not sizing.tp1_viable(snapshot.symbol, risk_dollars):
        return NQDTCEntryDecision(
            blocked_reasons=("tp1_fee_viability",),
            composite_regime=composite,
        )
    if snapshot.drawdown_size_multiplier <= 0:
        return NQDTCEntryDecision(
            blocked_reasons=("drawdown_pause",),
            composite_regime=composite,
        )

    closes = np.asarray(snapshot.bars_5m.get("close", ()), dtype=float)
    highs = np.asarray(snapshot.bars_5m.get("high", ()), dtype=float)
    lows = np.asarray(snapshot.bars_5m.get("low", ()), dtype=float)
    if len(closes) < 3 or len(highs) != len(closes) or len(lows) != len(closes):
        return NQDTCEntryDecision(
            blocked_reasons=("insufficient_5m_history",),
            composite_regime=composite,
        )

    close = float(closes[-1])
    plans: list[NQDTCEntryPlan] = []
    blocked_reasons: list[str] = []

    if (
        C.A_ENTRY_ENABLED
        and (snapshot.entry_a_retest or snapshot.entry_a_latch)
        and not _has_active_a_orders(snapshot.working_orders)
    ):
        allowed, reason = signals.a_entry_context_allowed(
            score=engine.last_score,
            box_width=engine.box.box_width,
        )
        if allowed:
            plans.extend(_plans_a(snapshot, direction, quality_mult, disp_norm, a_risk_pct))
        else:
            blocked_reasons.append(reason)

    if snapshot.entry_b_sweep and engine.atr14_30m > 0:
        b_permitted = (
            signals.b_entry_regime_allowed(composite)
            and not engine.breakout.continuation_mode
            and len(engine.disp_hist.data) > 10
            and engine.last_disp_metric
            >= ind.rolling_quantile_past_only(engine.disp_hist.data, C.B_MIN_DISP_Q)
        )
        if b_permitted and signals.entry_b_trigger(
            float(lows[-1]),
            float(highs[-1]),
            close,
            engine.vwap_session.value,
            engine.atr14_30m,
            direction,
        ):
            plan = _plan_b(snapshot, direction, close, quality_mult, disp_norm, other_risk_pct)
            if plan is not None:
                plans.append(plan)

    if len(closes) >= C.C_HOLD_BARS:
        holds, hold_ref = signals.entry_c_hold_check(
            closes,
            lows,
            highs,
            engine.vwap_session.value,
            direction,
            atr14_30m=engine.atr14_30m,
        )
        if holds:
            subtype, reason = _c_subtype(snapshot, composite, disp_norm)
            if reason:
                blocked_reasons.append(reason)
            elif subtype is not None:
                plan = _plan_c(
                    snapshot,
                    direction,
                    subtype,
                    close,
                    hold_ref,
                    quality_mult,
                    disp_norm,
                    other_risk_pct,
                )
                if plan is not None:
                    plans.append(plan)

    consume_fallback = bool(
        C.A_ENTRY_ENABLED
        and snapshot.a_fallback_eligible
        and snapshot.engine.breakout.active
        and not snapshot.working_orders
    )
    if consume_fallback:
        on_breakout_side = (
            direction is Direction.LONG and close > engine.box.box_high
        ) or (
            direction is Direction.SHORT and close < engine.box.box_low
        )
        if on_breakout_side:
            plan = _plan_fallback(
                snapshot,
                direction,
                close,
                quality_mult,
                disp_norm,
                other_risk_pct,
            )
            if plan is not None:
                plans.append(plan)

    return NQDTCEntryDecision(
        plans=tuple(plans),
        blocked_reasons=tuple(blocked_reasons),
        composite_regime=composite,
        consume_fallback=consume_fallback,
    )


def _hard_gate(snapshot: NQDTCEntryDecisionSnapshot) -> str:
    if snapshot.news_blackout:
        return "news_blackout"
    if snapshot.risk_halted:
        return "risk_halt"
    if snapshot.engine.mode.value == "HALT":
        return "chop_halt"
    if snapshot.position_open:
        return "position_open"
    if snapshot.cooldown_bars > 0:
        return "cooldown"
    gap_minutes = float(getattr(C, "MIN_INTER_TRADE_GAP_MINUTES", 0))
    if gap_minutes > 0 and snapshot.last_fill_time is not None:
        elapsed = (snapshot.now - snapshot.last_fill_time).total_seconds() / 60.0
        if elapsed < gap_minutes:
            return "inter_trade_gap"
    ny = snapshot.now.astimezone(_new_york())
    if snapshot.block_04_et and ny.hour == 4:
        return "hour_filter"
    if snapshot.block_05_et and ny.hour == 5 and ny.minute < 30:
        return "hour_filter"
    if snapshot.block_06_et and ny.hour == 6:
        return "hour_filter"
    if snapshot.block_09_et and ny.hour == 9:
        return "hour_filter"
    if snapshot.block_12_et and ny.hour == 12:
        return "hour_filter"
    if snapshot.block_thursday and ny.weekday() == 3:
        return "thursday_filter"
    if C.BLOCK_RTH_DEGRADED and snapshot.engine.session is Session.RTH and snapshot.engine.mode.value == "DEGRADED":
        return "rth_degraded"
    return ""


def _composite(
    snapshot: NQDTCEntryDecisionSnapshot,
    direction: Direction,
) -> tuple[CompositeRegime, bool]:
    ema50 = np.asarray(snapshot.bars_daily.get("ema50", ()), dtype=float)
    atr14 = np.asarray(snapshot.bars_daily.get("atr14", ()), dtype=float)
    _supports, opposes = signals.classify_daily_support(ema50, atr14, direction)
    if not snapshot.recompute_composite:
        return snapshot.regime.composite, opposes
    return (
        signals.compute_composite_regime(
            snapshot.regime.regime_4h.value,
            snapshot.regime.trend_dir_4h,
            direction,
            _supports,
            opposes,
        ),
        opposes,
    )


def _daily_history_ready(snapshot: NQDTCEntryDecisionSnapshot) -> bool:
    return (
        len(np.asarray(snapshot.bars_daily.get("ema50", ()))) > 3
        and len(np.asarray(snapshot.bars_daily.get("atr14", ()))) > 0
    )


def _slope_multiplier(snapshot: NQDTCEntryDecisionSnapshot, direction: Direction) -> float:
    closes = np.asarray(snapshot.bars_15m.get("close", ()), dtype=float)
    needed = C.MACD_SLOW + C.MACD_SIGNAL + C.SLOPE_LOOKBACK
    if not C.SLOPE_FILTER_ENABLED or len(closes) < needed:
        return 1.0
    return C.CONT_SIZE_MULT if signals.slope_supports_breakout(closes, direction) else C.REVERSAL_SIZE_MULT


def _plans_a(
    snapshot: NQDTCEntryDecisionSnapshot,
    direction: Direction,
    quality_mult: float,
    disp_norm: float,
    risk_pct: float,
) -> list[NQDTCEntryPlan]:
    engine = snapshot.engine
    if engine.breakout.breakout_bar_high == 0 and engine.breakout.breakout_bar_low == 0:
        return []
    tick = _tick(snapshot)
    a1_price, a2_price = signals.entry_a_trigger(
        0.0,
        0.0,
        0.0,
        engine.vwap_session.value,
        engine.breakout.breakout_bar_high,
        engine.breakout.breakout_bar_low,
        engine.box.box_high,
        engine.atr14_30m,
        direction,
    )
    a1_price = round_to_tick(a1_price, tick)
    a2_price = round_to_tick(a2_price, tick)
    plans: list[NQDTCEntryPlan] = []
    for subtype, trigger, enabled in (
        (EntrySubtype.A_RETEST, a1_price, snapshot.entry_a_retest),
        (EntrySubtype.A_LATCH, a2_price, snapshot.entry_a_latch),
    ):
        if not enabled:
            continue
        stop_for_risk = stops.compute_initial_stop(
            subtype,
            direction,
            trigger,
            engine.box.box_high,
            engine.box.box_low,
            engine.box.box_mid,
            engine.atr14_30m,
            tick_size=tick,
        )
        qty = _quantity(snapshot, trigger, stop_for_risk, risk_pct)
        if qty <= 0:
            continue
        if subtype is EntrySubtype.A_LATCH:
            sign = 1 if direction is Direction.LONG else -1
            price = round_to_tick(trigger + sign * C.A2_BUFFER_TICKS * tick, tick)
            order_type: Literal["LIMIT", "STOP_LIMIT"] = "STOP_LIMIT"
            stop_price = trigger
            is_limit = False
        else:
            price = trigger
            order_type = "LIMIT"
            stop_price = None
            is_limit = True
        plans.append(
            NQDTCEntryPlan(
                subtype=subtype,
                direction=direction,
                order_type=order_type,
                qty=qty,
                stop_for_risk=stop_for_risk,
                price=price,
                stop_price=stop_price,
                oca_group=snapshot.a_oca_group,
                is_limit=is_limit,
                quality_mult=quality_mult,
                disp_norm=disp_norm,
                ttl_bars=C.A_TTL_5M_BARS,
            )
        )
    return plans


def _plan_b(
    snapshot: NQDTCEntryDecisionSnapshot,
    direction: Direction,
    close: float,
    quality_mult: float,
    disp_norm: float,
    risk_pct: float,
) -> NQDTCEntryPlan | None:
    engine = snapshot.engine
    tick = _tick(snapshot)
    stop_for_risk = stops.compute_initial_stop(
        EntrySubtype.B_SWEEP,
        direction,
        close,
        engine.box.box_high,
        engine.box.box_low,
        engine.box.box_mid,
        engine.atr14_30m,
        tick_size=tick,
    )
    qty = _quantity(snapshot, close, stop_for_risk, risk_pct)
    if qty <= 0:
        return None
    sign = 1 if direction is Direction.LONG else -1
    price = round_to_tick(
        close + sign * C.RESCUE_MAX_SLIP_ATR * engine.atr14_30m,
        tick,
        "up" if sign > 0 else "down",
    )
    return NQDTCEntryPlan(
        subtype=EntrySubtype.B_SWEEP,
        direction=direction,
        order_type="LIMIT",
        qty=qty,
        stop_for_risk=stop_for_risk,
        price=price,
        tif="IOC",
        oca_group=snapshot.entry_oca_group,
        quality_mult=quality_mult,
        disp_norm=disp_norm,
    )


def _c_subtype(
    snapshot: NQDTCEntryDecisionSnapshot,
    composite: CompositeRegime,
    disp_norm: float,
) -> tuple[EntrySubtype | None, str]:
    breakout = snapshot.engine.breakout
    continuation = breakout.continuation_mode and snapshot.continuation_mode
    if continuation:
        if not snapshot.entry_c_continuation or not C.C_CONT_ENTRY_ENABLED:
            return None, "C_CONT_DISABLED"
        subtype = EntrySubtype.C_CONTINUATION
    else:
        if not snapshot.entry_c_standard:
            return None, "C_STANDARD_DISABLED"
        subtype = EntrySubtype.C_STANDARD
    if subtype is EntrySubtype.C_CONTINUATION and breakout.continuation_fills >= 1:
        return None, "C_CONT_MAX_FILLS"
    if subtype is EntrySubtype.C_CONTINUATION and breakout.last_trade_peak_r < C.C_CONT_MFE_GATE_R:
        return None, "C_CONT_MFE_GATE"
    if C.BLOCK_CONT_ALIGNED and subtype is EntrySubtype.C_CONTINUATION and composite is CompositeRegime.ALIGNED:
        return None, "C_CONT_ALIGNED_BLOCK"
    if (
        C.BLOCK_STD_NEUTRAL_LOW_DISP
        and subtype is EntrySubtype.C_STANDARD
        and composite is CompositeRegime.NEUTRAL
        and disp_norm < 0.5
    ):
        return None, "C_STD_NEUTRAL_LOW_DISP"
    return subtype, ""


def _plan_c(
    snapshot: NQDTCEntryDecisionSnapshot,
    direction: Direction,
    subtype: EntrySubtype,
    close: float,
    hold_ref: float,
    quality_mult: float,
    disp_norm: float,
    risk_pct: float,
) -> NQDTCEntryPlan | None:
    engine = snapshot.engine
    tick = _tick(snapshot)
    if subtype is EntrySubtype.C_STANDARD:
        offset = C.C_ENTRY_OFFSET_ATR_STANDARD * engine.atr14_30m
    elif subtype is EntrySubtype.C_CONTINUATION:
        offset = C.C_ENTRY_OFFSET_ATR_CONTINUATION * engine.atr14_30m
    else:
        offset = C.C_ENTRY_OFFSET_ATR * engine.atr14_30m
    if engine.atr14_30m <= 0:
        offset = tick
    entry_price = round_to_tick(
        hold_ref + (offset if direction is Direction.LONG else -offset),
        tick,
    )
    stop_reference = close if snapshot.c_stop_reference == "bar_close" else entry_price
    stop_for_risk = stops.compute_initial_stop(
        subtype,
        direction,
        stop_reference,
        engine.box.box_high,
        engine.box.box_low,
        engine.box.box_mid,
        engine.atr14_30m,
        hold_ref=hold_ref,
        tick_size=tick,
    )
    qty = _quantity(snapshot, entry_price, stop_for_risk, risk_pct)
    if qty <= 0:
        return None
    return NQDTCEntryPlan(
        subtype=subtype,
        direction=direction,
        order_type="LIMIT",
        qty=qty,
        stop_for_risk=stop_for_risk,
        price=entry_price,
        oca_group=snapshot.entry_oca_group,
        is_limit=True,
        quality_mult=quality_mult,
        disp_norm=disp_norm,
        ttl_bars=snapshot.c_ttl_bars,
    )


def _plan_fallback(
    snapshot: NQDTCEntryDecisionSnapshot,
    direction: Direction,
    close: float,
    quality_mult: float,
    disp_norm: float,
    risk_pct: float,
) -> NQDTCEntryPlan | None:
    engine = snapshot.engine
    tick = _tick(snapshot)
    stop_for_risk = stops.compute_initial_stop(
        EntrySubtype.MARKET_FALLBACK,
        direction,
        close,
        engine.box.box_high,
        engine.box.box_low,
        engine.box.box_mid,
        engine.atr14_30m,
        tick_size=tick,
    )
    qty = _quantity(snapshot, close, stop_for_risk, risk_pct)
    if qty <= 0:
        return None
    price = None
    if snapshot.fallback_order_type == "LIMIT":
        sign = 1 if direction is Direction.LONG else -1
        price = round_to_tick(
            close + sign * C.RESCUE_MAX_SLIP_ATR * engine.atr14_30m,
            tick,
            "up" if sign > 0 else "down",
        )
    return NQDTCEntryPlan(
        subtype=EntrySubtype.MARKET_FALLBACK,
        direction=direction,
        order_type=snapshot.fallback_order_type,
        qty=qty,
        stop_for_risk=stop_for_risk,
        price=price,
        tif=snapshot.fallback_tif,
        quality_mult=quality_mult,
        disp_norm=disp_norm,
    )


def _quantity(
    snapshot: NQDTCEntryDecisionSnapshot,
    entry_price: float,
    stop_price: float,
    risk_pct: float,
) -> int:
    qty = snapshot.fixed_quantity
    if qty is None:
        qty = sizing.compute_contracts(
            snapshot.symbol,
            entry_price,
            stop_price,
            snapshot.equity,
            risk_pct,
        )
    if qty <= 0 or snapshot.drawdown_size_multiplier <= 0:
        return 0
    if snapshot.drawdown_size_multiplier < 1.0:
        return max(1, int(qty * snapshot.drawdown_size_multiplier))
    return int(qty)


def _tick(snapshot: NQDTCEntryDecisionSnapshot) -> float:
    return float(C.NQ_SPECS[snapshot.symbol]["tick"])


def _has_active_a_orders(orders: Sequence[WorkingOrder]) -> bool:
    return any(
        order.subtype in {EntrySubtype.A_RETEST, EntrySubtype.A_LATCH}
        for order in orders
    )


def _new_york():
    from zoneinfo import ZoneInfo

    return ZoneInfo("America/New_York")
