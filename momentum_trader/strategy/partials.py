"""Uniform partial exit schedule.

v4.0 change: removed alignment-conditional partial logic.
All setups use the same schedule:
  P1 at +1.0R: 30%
  P2 at +1.5R: 30%
  Runner:      40%
"""
from __future__ import annotations

from .config import (
    PositionState,
    PARTIAL1_R, PARTIAL1_FRAC,
    PARTIAL2_R, PARTIAL2_FRAC,
)


def partial1_due(pos: PositionState, R_total: float) -> tuple[bool, int]:
    """Check if partial 1 is due. Returns (should_exit, qty)."""
    if pos.partial_done:
        return False, 0
    if R_total < PARTIAL1_R:
        return False, 0
    if pos.contracts < 2:
        return False, 0
    entry_qty = pos.entry_contracts if pos.entry_contracts > 0 else pos.contracts
    qty = max(1, int(entry_qty * PARTIAL1_FRAC))
    qty = min(qty, pos.contracts - 1)
    return True, qty


def partial2_due(pos: PositionState, R_total: float) -> tuple[bool, int]:
    """Check if partial 2 is due. Returns (should_exit, qty)."""
    if not pos.partial_done or pos.partial2_done:
        return False, 0
    if R_total < PARTIAL2_R:
        return False, 0
    if pos.contracts < 2:
        return False, 0
    entry_qty = pos.entry_contracts if pos.entry_contracts > 0 else pos.contracts
    qty = max(1, int(entry_qty * PARTIAL2_FRAC))
    qty = min(qty, pos.contracts - 1)
    return True, qty
