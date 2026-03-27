"""Position manager: simplified 9-step management loop.

v4.0 changes:
- Single-phase trail (no bar-based phases)
- Momentum stall exit at bar 8
- Uniform partials (no alignment-conditional logic)
- Tighter MFE ratchet (1.5R activation, 65% floor)
- Single time-decay BE force at bar 12
- Tighter early adverse (4 bars, -0.80R)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .config import (
    PositionState, Setup, SetupClass, SetupState,
    NQ_POINT_VALUE, CATASTROPHIC_STOP_R,
    EARLY_ADVERSE_BARS, EARLY_ADVERSE_R,
    MOMENTUM_STALL_BAR, MOMENTUM_STALL_MIN_R, MOMENTUM_STALL_MIN_MFE,
    TRAIL_ACTIVATION_R,
    MFE_RATCHET_ACTIVATION_R, MFE_RATCHET_FLOOR_PCT,
    TIME_DECAY_BARS, TIME_DECAY_PROGRESS_R,
    EARLY_STALE_BARS, EARLY_STALE_MAX_BARS, STALE_M_BARS, STALE_R_THRESHOLD,
)
from .indicators import BarSeries
from .trail import compute_trail
from .partials import partial1_due, partial2_due
from .risk import RiskEngine
from .session import NewsCalendar
from .signals import alignment_score
from .diagnostics import DiagnosticTracker

logger = logging.getLogger(__name__)


# ── Shared R-calculation utilities ─────────────────────────────────

def unrealized_r(pos: PositionState, last_price: float, point_value: float = NQ_POINT_VALUE) -> float:
    pts = (last_price - pos.avg_entry) * pos.direction
    pnl = pts * point_value * pos.contracts
    return pnl / max(pos.unit1_risk_usd, 1e-9)


def r_state(pos: PositionState, last_price: float, point_value: float = NQ_POINT_VALUE) -> float:
    """Total R including realized partials."""
    unr = unrealized_r(pos, last_price, point_value)
    realized_r = pos.realized_partial_usd / max(pos.unit1_risk_usd, 1e-9)
    return unr + realized_r


class PositionManager:
    """Manages open positions: v4.0 simplified management loop.

    Management flow:
    1. Catastrophic floor (-1.5R)
    1b. Early adverse kill switch (-0.80R within 4 bars)
    2. Momentum stall exit (bar 8, R < 0.30, MFE < 0.80)
    3. Trail activation at +0.55R + BE stop
    3b. +1R milestone tracking
    3c. Uniform partials: P1 +1.0R (30%), P2 +1.5R (30%), runner 40%
    4. Single-phase chandelier trail
    4a. MFE ratchet floor (65% of peak MFE after >= 1.5R)
    4b. Time-decay BE force at bar 12
    5. Early stale (16 bars, adverse structure)
    6. Stale exit (20 bars)
    """

    def __init__(
        self,
        exec_engine,
        risk_engine: RiskEngine,
        signal_engine,
        point_value: float = NQ_POINT_VALUE,
        diagnostics: Optional[DiagnosticTracker] = None,
    ):
        self.exec = exec_engine
        self.risk = risk_engine
        self.signals = signal_engine
        self.pv = point_value
        self.positions: list[PositionState] = []
        self.diag = diagnostics

    def open_position(self, setup: Setup, fill_price: float, qty: int, now_et: datetime) -> PositionState:
        pos = PositionState(
            pos_id=setup.setup_id,
            direction=setup.direction,
            avg_entry=fill_price,
            contracts=qty,
            unit1_risk_usd=setup.unit1_risk_usd,
            origin_class=setup.cls,
            origin_setup_id=setup.setup_id,
            entry_ts=now_et,
            stop_price=setup.stop0,
            alignment_score_at_entry=setup.alignment_score,
            current_alignment_score=setup.alignment_score,
            entry_contracts=qty,
            highest_since_entry=fill_price,
            lowest_since_entry=fill_price,
        )
        self.positions.append(pos)
        return pos

    async def check_profit_target_5m(
        self, high: float, low: float, now_et: Optional[datetime] = None,
    ) -> None:
        """Multi-tranche runner: no hard profit target exit."""
        return

    async def check_catastrophic_5m(
        self, high: float, low: float, now_et: Optional[datetime] = None,
    ) -> None:
        for pos in list(self.positions):
            if pos.contracts <= 0:
                continue
            worst_price = low if pos.direction == 1 else high
            R_at_worst = unrealized_r(pos, worst_price, self.pv)
            if R_at_worst > CATASTROPHIC_STOP_R:
                continue
            exit_price = pos.avg_entry + (
                CATASTROPHIC_STOP_R * pos.unit1_risk_usd
                / (pos.direction * self.pv * pos.contracts)
            )
            await self.exec.flatten(pos, "CATASTROPHIC_STOP")
            self._close_position(pos, "CATASTROPHIC_STOP", exit_price, now_et)

    async def manage_all(
        self, now_et: datetime, last_price: float,
        h1: BarSeries, h4: BarSeries, daily: BarSeries,
        news: NewsCalendar,
        pivots_1h=None,
        bid: Optional[float] = None, ask: Optional[float] = None,
    ) -> None:
        for pos in list(self.positions):
            if pos.contracts <= 0:
                self._close_position(pos, "ZERO_CONTRACTS", last_price, now_et)
                continue

            pos.bars_held_1h += 1
            R_total = r_state(pos, last_price, self.pv)
            R_unr = unrealized_r(pos, last_price, self.pv)
            pos.peak_mfe_r = max(pos.peak_mfe_r, max(0.0, R_total))
            pos.peak_mae_r = max(pos.peak_mae_r, max(0.0, -R_total))
            pos.highest_since_entry = max(pos.highest_since_entry, h1.last_high)
            pos.lowest_since_entry = min(pos.lowest_since_entry, h1.last_low)

            pos.current_alignment_score = alignment_score(pos.direction, daily, h4)

            if self.diag:
                self.diag.record_snapshot(pos, R_total, R_unr, last_price, now_et)

            # 1) Catastrophic floor
            if R_total < CATASTROPHIC_STOP_R:
                await self.exec.flatten(pos, "CATASTROPHIC_STOP")
                self._close_position(pos, "CATASTROPHIC_STOP", last_price, now_et)
                continue

            # 1b) Early adverse: -0.80R within 4 bars (tighter than v3.1)
            if pos.bars_held_1h <= EARLY_ADVERSE_BARS and R_total <= EARLY_ADVERSE_R:
                await self.exec.flatten(pos, "EARLY_ADVERSE")
                self._close_position(pos, "EARLY_ADVERSE", last_price, now_et)
                continue

            # 2) Momentum stall — v7: re-enabled (greedy optimizer found net positive)
            if (pos.bars_held_1h >= MOMENTUM_STALL_BAR
                    and R_total < MOMENTUM_STALL_MIN_R
                    and pos.peak_mfe_r < MOMENTUM_STALL_MIN_MFE
                    and not pos.trailing_active):
                await self.exec.flatten(pos, "MOMENTUM_STALL")
                self._close_position(pos, "MOMENTUM_STALL", last_price, now_et)
                continue

            # 3) Trail activation at TRAIL_ACTIVATION_R + BE stop
            if not pos.trailing_active and R_total >= TRAIL_ACTIVATION_R:
                pos.trailing_active = True
                be_stop = pos.avg_entry
                await self.exec.tighten_stop(pos, be_stop, "TRAIL_ACTIVATION_BE")
                if self.diag:
                    self.diag.record_trail_activation(pos, R_total)

            # 3b) +1R milestone
            if not pos.did_1r and R_total >= 1.0:
                pos.did_1r = True

            # 3c) Uniform partials
            should_p1, p1_qty = partial1_due(pos, R_total)
            if should_p1:
                oms_id = await self.exec.partial_exit(pos, p1_qty, "PARTIAL_1")
                if oms_id:
                    pos.partial_done = True
                    if self.diag:
                        self.diag.record_partial(pos, R_total)
                    await self._apply_partial_bookkeeping(pos, p1_qty, last_price, oms_id)

            should_p2, p2_qty = partial2_due(pos, R_total)
            if should_p2:
                oms_id = await self.exec.partial_exit(pos, p2_qty, "PARTIAL_2")
                if oms_id:
                    pos.partial2_done = True
                    await self._apply_partial_bookkeeping(pos, p2_qty, last_price, oms_id)

            # 4) Single-phase chandelier trail
            if pos.trailing_active and pos.contracts > 0:
                new_stop = compute_trail(pos, last_price, h1, self.pv)
                await self.exec.tighten_stop(pos, new_stop, "TRAIL")

            # 4a) MFE ratchet floor
            if (pos.trailing_active and pos.peak_mfe_r >= MFE_RATCHET_ACTIVATION_R
                    and pos.contracts > 0):
                min_exit_r = pos.peak_mfe_r * MFE_RATCHET_FLOOR_PCT
                r_points = pos.unit1_risk_usd / (self.pv * max(pos.contracts, 1))
                if r_points > 0:
                    floor_price = pos.avg_entry + pos.direction * min_exit_r * r_points
                    await self.exec.tighten_stop(pos, floor_price, "MFE_RATCHET")

            # 4b) Single time-decay: BE force at bar 12
            if not pos.trailing_active and pos.contracts > 0:
                if pos.bars_held_1h >= TIME_DECAY_BARS and R_total < TIME_DECAY_PROGRESS_R:
                    if self.diag:
                        self.diag.record_time_decay(pos)
                    await self.exec.tighten_stop(pos, pos.avg_entry, "TIME_DECAY_BE")

            # 5) Early stale
            if self._should_early_stale(pos, R_total, h1):
                await self.exec.flatten(pos, "EARLY_STALE")
                self._close_position(pos, "EARLY_STALE", last_price, now_et)
                continue

            # 6) Stale exit
            if self._should_stale(pos, R_total):
                await self.exec.flatten(pos, "STALE")
                self._close_position(pos, "STALE", last_price, now_et)
                continue

    def _should_early_stale(self, pos: PositionState, R_total: float, h1: BarSeries = None) -> bool:
        if pos.bars_held_1h < EARLY_STALE_BARS:
            return False
        if pos.trailing_active:
            return False
        if R_total >= 0:
            return False
        if pos.bars_held_1h >= EARLY_STALE_MAX_BARS:
            return True
        if h1 is not None and len(h1.bars) >= 3:
            last_close = h1.last_close
            prev_close = h1.close_n_ago(1)
            prev2_close = h1.close_n_ago(2)
            if pos.direction == 1:
                two_adverse = last_close < prev_close and prev_close < prev2_close
            else:
                two_adverse = last_close > prev_close and prev_close > prev2_close
            if two_adverse and R_total < -0.3:
                return True
            return False
        return True

    def _should_stale(self, pos: PositionState, R_total: float) -> bool:
        if R_total >= STALE_R_THRESHOLD:
            return False
        return pos.bars_held_1h >= STALE_M_BARS

    async def _apply_partial_bookkeeping(
        self, pos: PositionState, partial_qty: int, last_price: float, oms_id: str,
    ) -> None:
        pts_gained = (last_price - pos.avg_entry) * pos.direction
        est_pnl = pts_gained * self.pv * partial_qty
        pos.pending_exit_estimates[oms_id] = est_pnl
        pos.realized_partial_usd += est_pnl
        pos.contracts -= partial_qty
        delta_r = abs(pos.avg_entry - pos.stop_price) * self.pv * partial_qty / max(pos.unit1_risk_usd, 1e-9)
        self.risk.adjust_open_risk(pos.direction, -delta_r)
        pos.current_risk_r = max(0, pos.current_risk_r - delta_r)
        await self.exec.place_stop(pos, pos.stop_price)

    def _close_position(self, pos: PositionState, exit_reason: str = "", last_price: float = 0.0, now_et: Optional[datetime] = None) -> None:
        if self.diag and exit_reason:
            self.diag.finalize(pos, last_price, exit_reason, now_et or pos.entry_ts)
        self.risk.release_open_risk(pos.direction, pos.current_risk_r)
        self.signals.record_exit(pos.direction, exit_reason=exit_reason, bars_held=pos.bars_held_1h)
        if pos in self.positions:
            self.positions.remove(pos)
        logger.info("Position closed: %s", pos.pos_id)

    def has_position(self, direction: Optional[int] = None) -> bool:
        if direction is None:
            return len(self.positions) > 0
        return any(p.direction == direction for p in self.positions)
