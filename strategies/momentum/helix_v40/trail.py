"""Single-phase chandelier trailing stop.

v4.0 change: removed bar-based phases (wide bars 1-8, decay after 15).
Pure R-based: mult = max(1.5, 3.0 - R/5).
"""
from __future__ import annotations

from .config import (
    PositionState, NQ_POINT_VALUE,
    TRAIL_MULT_MAX, TRAIL_MULT_FLOOR, TRAIL_R_DIVISOR,
    TRAIL_LOOKBACK_1H,
)
from .indicators import BarSeries


def _r_state(pos: PositionState, last_price: float, point_value: float) -> float:
    """Total R including realized partials (inlined to avoid circular import)."""
    pts = (last_price - pos.avg_entry) * pos.direction
    pnl = pts * point_value * pos.contracts
    unr = pnl / max(pos.unit1_risk_usd, 1e-9)
    realized_r = pos.realized_partial_usd / max(pos.unit1_risk_usd, 1e-9)
    return unr + realized_r


def compute_trail(
    pos: PositionState,
    last_price: float,
    h1: BarSeries,
    point_value: float = NQ_POINT_VALUE,
) -> float:
    """Compute chandelier trailing stop price (single-phase).

    mult = max(TRAIL_MULT_FLOOR, TRAIL_MULT_MAX - R / TRAIL_R_DIVISOR)
    """
    R = _r_state(pos, last_price, point_value)

    mult = max(TRAIL_MULT_FLOOR, TRAIL_MULT_MAX - (R / TRAIL_R_DIVISOR))
    pos.trail_mult = mult

    atr1 = h1.current_atr()
    if pos.direction == 1:
        hh = h1.highest_high(TRAIL_LOOKBACK_1H)
        return hh - mult * atr1
    else:
        ll = h1.lowest_low(TRAIL_LOOKBACK_1H)
        return ll + mult * atr1
