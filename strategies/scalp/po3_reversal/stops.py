from __future__ import annotations

from strategies.scalp._shared.nq_contract import compute_contracts, round_to_tick, spec_for

from .config import (
    A_PLUS_RISK_PCT,
    A_RISK_PCT,
    B_RISK_PCT,
    ENTRY_OFFSET_TICKS,
    MIN_RR_A,
    MIN_RR_B,
    STOP_ATR_BUFFER_FRACTION,
    STOP_MIN_BUFFER_TICKS,
    SetupTier,
    TradeDirection,
)
from .models import PriceBar


def compute_entry_price(
    confirmation_bar: PriceBar,
    direction: TradeDirection,
    *,
    tick_size: float = 0.25,
) -> float:
    if direction is TradeDirection.LONG:
        return round_to_tick(confirmation_bar.high + ENTRY_OFFSET_TICKS * tick_size, tick_size, "up")
    if direction is TradeDirection.SHORT:
        return round_to_tick(confirmation_bar.low - ENTRY_OFFSET_TICKS * tick_size, tick_size, "down")
    return 0.0


def compute_stop_price(
    *,
    direction: TradeDirection,
    smt_extreme: float,
    atr_1m: float,
    tick_size: float = 0.25,
) -> float:
    buffer_pts = max(STOP_MIN_BUFFER_TICKS * tick_size, STOP_ATR_BUFFER_FRACTION * atr_1m)
    if direction is TradeDirection.LONG:
        return round_to_tick(smt_extreme - buffer_pts, tick_size, "down")
    if direction is TradeDirection.SHORT:
        return round_to_tick(smt_extreme + buffer_pts, tick_size, "up")
    return 0.0


def reward_to_risk(entry: float, stop: float, target: float, direction: TradeDirection) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = target - entry if direction is TradeDirection.LONG else entry - target
    return max(0.0, reward / risk)


def target_passes_rr(rr: float, tier: SetupTier) -> bool:
    if tier is SetupTier.B:
        return rr >= MIN_RR_B
    if tier in {SetupTier.A, SetupTier.A_PLUS}:
        return rr >= MIN_RR_A
    return False


def risk_pct_for_tier(tier: SetupTier) -> float:
    if tier is SetupTier.A_PLUS:
        return A_PLUS_RISK_PCT
    if tier is SetupTier.A:
        return A_RISK_PCT
    if tier is SetupTier.B:
        return B_RISK_PCT
    return 0.0


def compute_position_size(
    *,
    equity: float,
    tier: SetupTier,
    entry: float,
    stop: float,
    symbol: str = "NQ",
) -> int:
    spec = spec_for(symbol)
    return compute_contracts(
        equity,
        risk_pct_for_tier(tier),
        abs(entry - stop),
        spec.point_value,
    )

