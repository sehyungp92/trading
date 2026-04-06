"""BRS position state tracking and management.

Wraps exit checks, trailing stop updates, and scale-out logic into a
stateful per-position object that the engine calls each hourly bar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .config import BRSConfig, BRSSymbolConfig
from .exits import check_exits, check_scale_out, update_trailing_stop
from .models import (
    BRSRegime,
    DailyContext,
    Direction,
    EntrySignal,
    ExitReason,
    HourlyContext,
)

logger = logging.getLogger(__name__)


class PositionAction(Enum):
    """Actions the engine must execute after position management."""
    EXIT = "EXIT"
    STOP_UPDATE = "STOP_UPDATE"
    SCALE_OUT = "SCALE_OUT"


@dataclass
class ActionResult:
    """Instruction from position manager to engine execution layer."""
    action: PositionAction
    exit_reason: Optional[ExitReason] = None
    exit_price: float = 0.0
    new_stop: float = 0.0
    scale_out_qty: int = 0


@dataclass
class BRSPositionState:
    """Full state for a single BRS position."""
    symbol: str
    pos_id: str
    direction: Direction
    entry_price: float
    qty: int
    original_qty: int
    risk_per_unit: float
    stop_price: float
    entry_type: str
    regime_at_entry: BRSRegime
    entry_ts: datetime

    # Management state
    bars_held: int = 0
    mfe_price: float = 0.0
    mfe_r: float = 0.0
    mae_price: float = 0.0
    mae_r: float = 0.0
    be_triggered: bool = False
    chand_bonus: float = 0.0
    quality_score: float = 0.0
    vol_factor: float = 1.0

    # Scale-out state
    tranche_b_open: bool = False
    tranche_b_qty: int = 0

    # Pyramiding
    pyramid_count: int = 0

    # Hourly tracking for chandelier
    hourly_highs: list[float] = field(default_factory=list)
    hourly_lows: list[float] = field(default_factory=list)

    # OMS tracking
    stop_oms_id: Optional[str] = None
    entry_oms_id: Optional[str] = None

    @classmethod
    def from_signal(
        cls,
        symbol: str,
        pos_id: str,
        signal: EntrySignal,
        qty: int,
        entry_ts: datetime,
    ) -> BRSPositionState:
        """Create position state from an entry signal fill."""
        pos = cls(
            symbol=symbol,
            pos_id=pos_id,
            direction=signal.direction,
            entry_price=signal.signal_price,
            qty=qty,
            original_qty=qty,
            risk_per_unit=signal.risk_per_unit,
            stop_price=signal.stop_price,
            entry_type=signal.entry_type.value,
            regime_at_entry=signal.regime_at_entry,
            entry_ts=entry_ts,
            mfe_price=signal.signal_price,
            mae_price=signal.signal_price,
            quality_score=signal.quality_score,
            vol_factor=signal.vol_factor,
        )
        return pos

    def current_r(self, price: float) -> float:
        """Unrealized R-multiple at given price."""
        if self.risk_per_unit <= 0:
            return 0.0
        if self.direction == Direction.SHORT:
            return (self.entry_price - price) / self.risk_per_unit
        else:
            return (price - self.entry_price) / self.risk_per_unit

    def update_mfe_mae(self, bar_high: float, bar_low: float) -> None:
        """Update MFE/MAE tracking from a new bar."""
        if self.direction == Direction.SHORT:
            if bar_low < self.mfe_price or self.mfe_price == self.entry_price:
                self.mfe_price = bar_low
            if bar_high > self.mae_price or self.mae_price == self.entry_price:
                self.mae_price = bar_high
            self.mfe_r = max(self.mfe_r, (self.entry_price - bar_low) / max(self.risk_per_unit, 1e-9))
            self.mae_r = min(self.mae_r, (self.entry_price - bar_high) / max(self.risk_per_unit, 1e-9))
        else:
            if bar_high > self.mfe_price or self.mfe_price == self.entry_price:
                self.mfe_price = bar_high
            if bar_low < self.mae_price or self.mae_price == self.entry_price:
                self.mae_price = bar_low
            self.mfe_r = max(self.mfe_r, (bar_high - self.entry_price) / max(self.risk_per_unit, 1e-9))
            self.mae_r = min(self.mae_r, (bar_low - self.entry_price) / max(self.risk_per_unit, 1e-9))

    def manage(
        self,
        h_ctx: HourlyContext,
        d_ctx: DailyContext,
        cfg: BRSConfig,
        sym_cfg: BRSSymbolConfig,
        prev_regime: BRSRegime,
    ) -> list[ActionResult]:
        """Run full position management for one hourly bar.

        Returns list of actions the engine should execute.
        """
        actions: list[ActionResult] = []
        self.bars_held += 1

        # Track hourly H/L for chandelier (cap at 50, logic uses last 25)
        self.hourly_highs.append(h_ctx.high)
        self.hourly_lows.append(h_ctx.low)
        if len(self.hourly_highs) > 50:
            self.hourly_highs = self.hourly_highs[-50:]
            self.hourly_lows = self.hourly_lows[-50:]

        # Update MFE/MAE
        self.update_mfe_mae(h_ctx.high, h_ctx.low)

        cur_r = self.current_r(h_ctx.close)
        is_long = self.direction == Direction.LONG

        # Check exits
        exit_result = check_exits(
            direction=self.direction,
            entry_price=self.entry_price,
            current_stop=self.stop_price,
            risk_per_unit=self.risk_per_unit,
            bar_high=h_ctx.high,
            bar_low=h_ctx.low,
            bar_close=h_ctx.close,
            atr14_h=h_ctx.atr14_h,
            atr14_d=d_ctx.atr14_d,
            bars_held=self.bars_held,
            mfe_r=self.mfe_r,
            cur_r=cur_r,
            be_triggered=self.be_triggered,
            regime=d_ctx.regime,
            sym_cfg=sym_cfg,
            catastrophic_cap_r=cfg.catastrophic_cap_r,
            be_trigger_r=cfg.be_trigger_r,
            trail_trigger_r=cfg.trail_trigger_r,
            stale_bars=cfg.stale_bars_short if not is_long else cfg.stale_bars_long,
            stale_early_bars=cfg.stale_early_bars,
            time_decay_hours=cfg.time_decay_hours,
            is_long=is_long,
            hourly_highs=self.hourly_highs,
            hourly_lows=self.hourly_lows,
            min_hold_bars=cfg.min_hold_bars,
        )
        if exit_result is not None:
            reason, price = exit_result
            actions.append(ActionResult(
                action=PositionAction.EXIT,
                exit_reason=reason,
                exit_price=price,
            ))
            return actions

        # Scale-out check
        if cfg.scale_out_enabled and self.tranche_b_open:
            if check_scale_out(cur_r, cfg.scale_out_target_r, self.tranche_b_open):
                actions.append(ActionResult(
                    action=PositionAction.SCALE_OUT,
                    scale_out_qty=self.tranche_b_qty,
                    exit_price=h_ctx.close,
                ))
                self.tranche_b_open = False
                self.chand_bonus = cfg.scale_out_trail_bonus

        # Trailing stop update
        new_stop, new_be = update_trailing_stop(
            direction=self.direction,
            current_stop=self.stop_price,
            entry_price=self.entry_price,
            risk_per_unit=self.risk_per_unit,
            mfe_r=self.mfe_r,
            atr14_d=d_ctx.atr14_d,
            atr14_h=h_ctx.atr14_h,
            be_triggered=self.be_triggered,
            regime=d_ctx.regime,
            prev_regime=prev_regime,
            sym_cfg=sym_cfg,
            be_trigger_r=cfg.be_trigger_r,
            trail_trigger_r=cfg.trail_trigger_r,
            profit_floor_scale=cfg.profit_floor_scale,
            hourly_highs=self.hourly_highs,
            hourly_lows=self.hourly_lows,
            chand_bonus=self.chand_bonus,
            trail_regime_scaling=cfg.trail_regime_scaling,
        )

        self.be_triggered = new_be
        if new_stop != self.stop_price:
            self.stop_price = new_stop
            actions.append(ActionResult(
                action=PositionAction.STOP_UPDATE,
                new_stop=new_stop,
            ))

        return actions

    def apply_pyramid(self, add_qty: int, add_price: float) -> None:
        """Blend entry price after pyramid add-on."""
        total_qty = self.qty + add_qty
        self.entry_price = (self.entry_price * self.qty + add_price * add_qty) / total_qty
        self.qty = total_qty
        self.risk_per_unit = abs(self.entry_price - self.stop_price)
        self.pyramid_count += 1

    def setup_scale_out(self, cfg: BRSConfig) -> None:
        """Initialize tranche B for scale-out after entry."""
        if cfg.scale_out_enabled and self.qty >= 2:
            import math
            self.tranche_b_qty = max(1, int(math.floor(self.qty * cfg.scale_out_pct)))
            self.tranche_b_open = True
