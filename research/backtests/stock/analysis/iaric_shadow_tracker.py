"""IARIC shadow tracker -- bar-by-bar simulation of rejected setups.

Tracks what would have happened if rejected IARIC entries had been taken.
IARIC-specific shadow exit rules:
- STOP_HIT: bar_low <= stop_price
- TIME_STOP: bars_held >= time_stop_bars AND bar_close <= entry_price
- PARTIAL_TARGET: bar_high >= entry + partial_r × risk_per_share (runner at EOD)
- EOD_FLATTEN: bars_held >= MAX_SHADOW_BARS with no prior exit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class IARICShadowSetup:
    """A rejected IARIC entry that we simulate forward."""

    symbol: str
    trade_date: date
    rejection_gate: str          # position_cap, timing_blocked, sector_limit, etc.
    entry_price: float
    stop_price: float
    risk_per_share: float = 0.0
    setup_type: str = ""         # PANIC_FLUSH/DRIFT_EXHAUSTION/HOT/WARM
    sector: str = ""
    regime_tier: str = ""
    sponsorship_state: str = ""
    confidence: str = ""
    location_grade: str = ""
    acceptance_count: int = 0
    conviction_multiplier: float = 0.0
    # Updated bar-by-bar
    active: bool = True
    bars_held: int = 0
    max_price: float = 0.0
    min_price: float = 0.0
    # Final result
    simulated_exit: str = ""     # STOP_HIT, TIME_STOP, PARTIAL_TARGET, EOD_FLATTEN
    simulated_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0


_MAX_SHADOW_BARS = 78  # full day of 5m bars
_DEFAULT_TIME_STOP_BARS = 9   # 45min / 5 = 9 bars
_DEFAULT_PARTIAL_R = 1.5


class IARICShadowTracker:
    """Tracks rejected IARIC setups and simulates their outcomes bar-by-bar."""

    def __init__(
        self,
        time_stop_bars: int = _DEFAULT_TIME_STOP_BARS,
        partial_r_multiple: float = _DEFAULT_PARTIAL_R,
    ) -> None:
        self._active_shadows: list[IARICShadowSetup] = []
        self._completed: list[IARICShadowSetup] = []
        self._funnel: dict[str, int] = {}
        self._time_stop_bars = time_stop_bars
        self._partial_r = partial_r_multiple

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_funnel(self, stage: str) -> None:
        """Increment funnel counter."""
        self._funnel[stage] = self._funnel.get(stage, 0) + 1

    def record_rejection(self, setup: IARICShadowSetup) -> None:
        """Record a rejected entry for shadow simulation."""
        rps = abs(setup.entry_price - setup.stop_price)
        setup.risk_per_share = rps if rps > 0 else 1.0
        setup.max_price = setup.entry_price
        setup.min_price = setup.entry_price
        self._active_shadows.append(setup)

    # ------------------------------------------------------------------
    # Bar-by-bar update
    # ------------------------------------------------------------------

    def update_bar(self, symbol: str, bar_high: float, bar_low: float, bar_close: float) -> None:
        """Process one 5m bar for all active shadows on this symbol.

        Simulates stop hit, time stop, partial target, and stale exit.
        """
        for s in self._active_shadows:
            if s.symbol != symbol or not s.active:
                continue

            s.bars_held += 1
            rps = s.risk_per_share
            s.max_price = max(s.max_price, bar_high)
            s.min_price = min(s.min_price, bar_low)

            # Stop hit
            if bar_low <= s.stop_price:
                s.simulated_r = (s.stop_price - s.entry_price) / rps
                s.simulated_exit = "STOP_HIT"
                s.active = False

            # Time stop: after N bars, exit if underwater
            elif (s.bars_held >= self._time_stop_bars
                  and bar_close <= s.entry_price):
                s.simulated_r = (bar_close - s.entry_price) / rps
                s.simulated_exit = "TIME_STOP"
                s.active = False

            # Partial target: price reaches 1.5R -> simulate partial + runner to EOD
            elif bar_high >= s.entry_price + self._partial_r * rps:
                # Simulate: take partial at target, runner at EOD close
                # Use weighted: 50% at target + 50% at current close
                partial_r = self._partial_r
                runner_r = (bar_close - s.entry_price) / rps
                s.simulated_r = 0.5 * partial_r + 0.5 * runner_r
                s.simulated_exit = "PARTIAL_TARGET"
                s.active = False

            # EOD flatten
            elif s.bars_held >= _MAX_SHADOW_BARS:
                s.simulated_r = (bar_close - s.entry_price) / rps
                s.simulated_exit = "EOD_FLATTEN"
                s.active = False

            # Update MFE/MAE
            s.mfe_r = (s.max_price - s.entry_price) / rps
            s.mae_r = (s.entry_price - s.min_price) / rps

        # Move completed shadows
        still_active = []
        for s in self._active_shadows:
            if s.active:
                still_active.append(s)
            else:
                self._completed.append(s)
        self._active_shadows = still_active

    # ------------------------------------------------------------------
    # End-of-day cleanup
    # ------------------------------------------------------------------

    def flush_stale(self) -> None:
        """Force-close any remaining active shadows at end of day."""
        for s in self._active_shadows:
            if s.active:
                s.simulated_exit = "EXPIRED"
                s.simulated_r = 0.0
                s.active = False
                self._completed.append(s)
        self._active_shadows = []

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def completed(self) -> list[IARICShadowSetup]:
        return self._completed

    @property
    def funnel(self) -> dict[str, int]:
        return dict(self._funnel)

    def get_filter_summary(self) -> dict[str, list[IARICShadowSetup]]:
        """Group completed shadows by rejection gate."""
        by_gate: dict[str, list[IARICShadowSetup]] = {}
        for s in self._completed:
            by_gate.setdefault(s.rejection_gate, []).append(s)
        return by_gate

    def funnel_report(self) -> str:
        """Format signal funnel as a text report."""
        if not self._funnel:
            return "  No funnel data recorded."
        lines = ["  Signal Funnel:", "  " + "-" * 50]
        prev = 0
        for stage in [
            "evaluated", "idle_check", "setup_detected", "accepting",
            "ready_to_enter", "regime_gate", "timing_blocked",
            "position_cap", "sector_limit", "heat_cap",
            "confidence_red", "sponsorship_filter", "entered",
        ]:
            count = self._funnel.get(stage, 0)
            if count == 0 and prev == 0:
                continue
            drop = ""
            if prev > 0 and count < prev:
                drop = f"  (-{prev - count}, {count/prev:.0%} pass)"
            lines.append(f"  {stage:<25s} {count:>6}{drop}")
            prev = count
        return "\n".join(lines)
