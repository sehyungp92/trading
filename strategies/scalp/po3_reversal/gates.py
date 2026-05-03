from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from strategies.scalp._shared.session import ScalpSessionBlock, entries_allowed, get_session_block

from .config import (
    MAX_DAILY_LOSS_PCT,
    MAX_FULL_LOSSES_PER_DAY,
    MAX_TRADES_PER_DAY,
    TradeDirection,
)


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    details: dict[str, float | str | bool] = field(default_factory=dict)


def session_gate(ts: datetime, *, production: bool = True) -> GateResult:
    block = get_session_block(ts)
    allowed = entries_allowed(block, "po3_reversal")
    if not allowed:
        return GateResult(False, ("SESSION_CLOSED",), {"block": block.value})
    if production and block not in {ScalpSessionBlock.IVB_FORMING, ScalpSessionBlock.RTH_PRIME}:
        return GateResult(False, ("OUTSIDE_PRODUCTION_WINDOW",), {"block": block.value})
    return GateResult(True, details={"block": block.value})


def risk_gate(
    *,
    daily_pnl: float,
    equity: float,
    trades_today: int,
    full_losses_today: int,
) -> GateResult:
    reasons: list[str] = []
    if equity > 0 and daily_pnl <= -equity * MAX_DAILY_LOSS_PCT:
        reasons.append("DAILY_LOSS_LIMIT")
    if trades_today >= MAX_TRADES_PER_DAY:
        reasons.append("MAX_TRADES_PER_DAY")
    if full_losses_today >= MAX_FULL_LOSSES_PER_DAY:
        reasons.append("MAX_FULL_LOSSES_PER_DAY")
    return GateResult(
        not reasons,
        tuple(reasons),
        {"daily_pnl": daily_pnl, "equity": equity, "trades_today": trades_today},
    )


def spread_gate(
    *,
    current_spread_ticks: float,
    rolling_75th: float | None,
    rolling_90th: float | None = None,
) -> GateResult:
    if rolling_90th is not None and current_spread_ticks >= rolling_90th:
        return GateResult(False, ("SPREAD_EMERGENCY",), {"spread_ticks": current_spread_ticks})
    if rolling_75th is not None and current_spread_ticks > rolling_75th:
        return GateResult(False, ("SPREAD_ELEVATED",), {"spread_ticks": current_spread_ticks})
    return GateResult(True, details={"spread_ticks": current_spread_ticks})


def volatility_gate(
    *,
    current_range: float,
    median_range_30m: float,
    atr_percentile: float | None = None,
) -> GateResult:
    if median_range_30m > 0 and current_range > 2.5 * median_range_30m:
        return GateResult(False, ("VOLATILITY_SHOCK",), {"range": current_range})
    if atr_percentile is not None and (atr_percentile > 90 or atr_percentile < 20):
        return GateResult(False, ("ATR_REGIME_OUT_OF_BOUNDS",), {"atr_percentile": atr_percentile})
    return GateResult(True)


def direction_gate(
    daily_bias: TradeDirection,
    h4_bias: TradeDirection,
    setup_direction: TradeDirection,
) -> GateResult:
    if h4_bias is TradeDirection.FLAT:
        return GateResult(False, ("NO_H4_BIAS",))
    if setup_direction != h4_bias:
        return GateResult(False, ("SETUP_OPPOSES_H4",))
    if daily_bias not in {TradeDirection.FLAT, h4_bias}:
        return GateResult(False, ("DAILY_CONFLICTS_H4",))
    return GateResult(True)
