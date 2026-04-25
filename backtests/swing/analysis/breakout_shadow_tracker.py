"""Shadow trade simulation for Breakout v3.3-ETF rejected candidates.

Tracks what would have happened if a gate had NOT rejected a signal.
Uses the Breakout stop lifecycle: TP1 partial, TP2 partial, runner
trailing stop (4H ATR), BE at TP1, stale exit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from strategies.swing.breakout.config import (
    ENTRY_A_TTL_RTH_HOURS,
    STALE_EXIT_DAYS_MIN,
    STALE_R_THRESH,
    TP1_R_NEUTRAL,
    TP2_R_NEUTRAL,
)
from strategies.swing.breakout.models import Direction

logger = logging.getLogger(__name__)


@dataclass
class BreakoutShadowCandidate:
    """A candidate that was rejected by one or more gates."""

    symbol: str
    direction: int
    filter_names: list[str]
    time: datetime
    entry_price: float
    stop_price: float
    entry_type: str = ""       # A, B, C_standard, C_continuation

    @property
    def filter_name(self) -> str:
        return self.filter_names[0] if self.filter_names else ""


@dataclass
class BreakoutShadowResult:
    """Outcome of simulating a shadow candidate."""

    candidate: BreakoutShadowCandidate
    filled: bool = False
    fill_price: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    reached_tp1: bool = False
    reached_tp2: bool = False


@dataclass
class BreakoutFilterStats:
    """Aggregated stats for one filter/gate."""

    filter_name: str
    rejected_count: int = 0
    simulated_count: int = 0
    filled_count: int = 0
    avg_shadow_r: float = 0.0
    pct_reach_tp1: float = 0.0
    pct_reach_tp2: float = 0.0
    pct_above_1r: float = 0.0
    net_missed_expectancy: float = 0.0
    net_avoided_loss: float = 0.0


class BreakoutShadowTracker:
    """Track rejected Breakout candidates and simulate their outcomes."""

    def __init__(self):
        self.rejections: list[BreakoutShadowCandidate] = []
        self.results: list[BreakoutShadowResult] = []

    def record_rejection(
        self,
        symbol: str,
        direction: int,
        filter_names: str | list[str],
        time: datetime,
        entry_price: float,
        stop_price: float,
        entry_type: str = "",
    ) -> None:
        """Log a rejected candidate with all failed gate names."""
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        self.rejections.append(BreakoutShadowCandidate(
            symbol=symbol, direction=direction, filter_names=filter_names,
            time=time, entry_price=entry_price, stop_price=stop_price,
            entry_type=entry_type,
        ))

    def simulate_shadows(
        self,
        hourly_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        hourly_times: dict[str, np.ndarray],
    ) -> list[BreakoutShadowResult]:
        """Forward-simulate each rejected candidate using Breakout stop lifecycle.

        Uses simplified TP1/TP2/runner/stale exit rules matching the main
        backtest engine.
        """
        self.results.clear()

        for cand in self.rejections:
            sym = cand.symbol
            if sym not in hourly_data or sym not in hourly_times:
                continue

            opens, highs, lows, closes, _ = hourly_data[sym]
            times = hourly_times[sym]

            result = self._simulate_one(cand, opens, highs, lows, closes, times)
            self.results.append(result)

        return self.results

    def _simulate_one(
        self,
        cand: BreakoutShadowCandidate,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray,
    ) -> BreakoutShadowResult:
        """Simulate one shadow candidate with Breakout stop lifecycle.

        Simplified lifecycle:
        1. Check fill within TTL
        2. Track MFE/MAE
        3. TP1 at NEUTRAL R target → stop to BE
        4. TP2 at NEUTRAL R target
        5. Stale exit after STALE_EXIT_DAYS_MIN bars if R < threshold
        6. Protective stop
        """
        cand_ts = (
            np.datetime64(cand.time, "ns")
            if not isinstance(cand.time, np.datetime64)
            else cand.time
        )
        start_idx = int(np.searchsorted(times, cand_ts, side="right"))

        if start_idx >= len(times):
            return BreakoutShadowResult(candidate=cand)

        r_base = abs(cand.entry_price - cand.stop_price)
        if r_base <= 0:
            return BreakoutShadowResult(candidate=cand)

        # TTL: use entry A TTL as default (6 RTH hours)
        ttl = ENTRY_A_TTL_RTH_HOURS

        # Phase 1: check if entry would fill within TTL
        fill_price = 0.0
        fill_idx = -1

        if cand.entry_type == "B":
            # Market order — fills at next bar open
            if start_idx < len(opens):
                fill_price = float(opens[start_idx])
                fill_idx = start_idx
        else:
            # Limit-like entry (A, C) — check if price touches entry level
            for i in range(start_idx, min(start_idx + ttl, len(opens))):
                if cand.direction == Direction.LONG:
                    # Buy limit: fills when price drops to entry_price
                    if lows[i] <= cand.entry_price:
                        fill_price = min(float(opens[i]), cand.entry_price)
                        fill_idx = i
                        break
                else:
                    # Sell limit: fills when price rises to entry_price
                    if highs[i] >= cand.entry_price:
                        fill_price = max(float(opens[i]), cand.entry_price)
                        fill_idx = i
                        break

        if fill_idx < 0:
            return BreakoutShadowResult(candidate=cand, filled=False)

        # Phase 2: full Breakout stop lifecycle
        current_stop = cand.stop_price
        mfe_price = fill_price
        mae_price = fill_price
        mfe_r = 0.0
        mae_r = 0.0
        tp1_reached = False
        tp2_reached = False
        be_triggered = False
        # Stale: use daily bar approximation (7 hourly bars ≈ 1 day)
        stale_bars = STALE_EXIT_DAYS_MIN * 7

        for j in range(fill_idx + 1, len(closes)):
            bars_held = j - fill_idx

            # Update MFE / MAE
            if cand.direction == Direction.LONG:
                if highs[j] > mfe_price:
                    mfe_price = float(highs[j])
                if lows[j] < mae_price:
                    mae_price = float(lows[j])
                cur_mfe = (mfe_price - fill_price) / r_base
                cur_mae = (fill_price - mae_price) / r_base
                cur_r = (float(closes[j]) - fill_price) / r_base
            else:
                if lows[j] < mfe_price:
                    mfe_price = float(lows[j])
                if highs[j] > mae_price:
                    mae_price = float(highs[j])
                cur_mfe = (fill_price - mfe_price) / r_base
                cur_mae = (mae_price - fill_price) / r_base
                cur_r = (fill_price - float(closes[j])) / r_base

            mfe_r = max(mfe_r, cur_mfe)
            mae_r = max(mae_r, cur_mae)

            # Check stop fill first
            if cand.direction == Direction.LONG:
                if lows[j] <= current_stop:
                    pnl_r = (current_stop - fill_price) / r_base
                    return BreakoutShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                        bars_held=bars_held,
                        reached_tp1=tp1_reached, reached_tp2=tp2_reached,
                    )
            else:
                if highs[j] >= current_stop:
                    pnl_r = (fill_price - current_stop) / r_base
                    return BreakoutShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                        bars_held=bars_held,
                        reached_tp1=tp1_reached, reached_tp2=tp2_reached,
                    )

            # TP1 check (using NEUTRAL targets as simplified baseline)
            if not tp1_reached and cur_mfe >= TP1_R_NEUTRAL:
                tp1_reached = True
                # Move stop to breakeven
                if not be_triggered:
                    be_triggered = True
                    current_stop = fill_price

            # TP2 check
            if tp1_reached and not tp2_reached and cur_mfe >= TP2_R_NEUTRAL:
                tp2_reached = True

            # Stale exit
            if bars_held >= stale_bars and cur_r < STALE_R_THRESH:
                exit_px = float(closes[j])
                if cand.direction == Direction.LONG:
                    pnl_r = (exit_px - fill_price) / r_base
                else:
                    pnl_r = (fill_price - exit_px) / r_base
                return BreakoutShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=exit_px, exit_reason="STALE",
                    r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                    bars_held=bars_held,
                    reached_tp1=tp1_reached, reached_tp2=tp2_reached,
                )

        # End of data
        last_close = float(closes[-1])
        if cand.direction == Direction.LONG:
            pnl_r = (last_close - fill_price) / r_base
        else:
            pnl_r = (fill_price - last_close) / r_base

        return BreakoutShadowResult(
            candidate=cand, filled=True, fill_price=fill_price,
            exit_price=last_close, exit_reason="END_OF_DATA",
            r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
            bars_held=len(closes) - fill_idx,
            reached_tp1=tp1_reached, reached_tp2=tp2_reached,
        )

    def get_filter_summary(self) -> dict[str, BreakoutFilterStats]:
        """Compute per-gate stats from simulation results."""
        by_filter: dict[str, list[BreakoutShadowResult]] = {}
        for r in self.results:
            for name in r.candidate.filter_names:
                by_filter.setdefault(name, []).append(r)

        rej_counts: dict[str, int] = {}
        for c in self.rejections:
            for name in c.filter_names:
                rej_counts[name] = rej_counts.get(name, 0) + 1

        summaries: dict[str, BreakoutFilterStats] = {}
        for name, results in by_filter.items():
            filled = [r for r in results if r.filled]
            r_multiples = [r.r_multiple for r in filled]

            stats = BreakoutFilterStats(filter_name=name)
            stats.rejected_count = rej_counts.get(name, len(results))
            stats.simulated_count = len(results)
            stats.filled_count = len(filled)

            if filled:
                arr = np.array(r_multiples)
                stats.avg_shadow_r = float(np.mean(arr))
                stats.pct_above_1r = float(np.mean(arr > 1.0)) * 100
                stats.pct_reach_tp1 = float(np.mean([r.reached_tp1 for r in filled])) * 100
                stats.pct_reach_tp2 = float(np.mean([r.reached_tp2 for r in filled])) * 100
                stats.net_missed_expectancy = float(np.sum(arr[arr > 0]))
                stats.net_avoided_loss = float(np.sum(np.abs(arr[arr < 0])))

            summaries[name] = stats

        return summaries

    def format_summary(self) -> str:
        """Format filter summary as a text report."""
        summaries = self.get_filter_summary()
        if not summaries:
            return "=== Shadow Trade Summary ===\n  No shadow trades simulated."

        lines = ["=== Breakout Shadow Trade Summary ==="]
        header = (
            f"  {'Gate':22s} {'Rej':>5s} {'Filled':>6s} {'AvgR':>7s} "
            f"{'TP1%':>5s} {'TP2%':>5s} {'>1R%':>5s} "
            f"{'Missed':>8s} {'Avoided':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for name in sorted(summaries, key=lambda n: -summaries[n].rejected_count):
            s = summaries[name]
            lines.append(
                f"  {name:22s} {s.rejected_count:5d} {s.filled_count:6d} "
                f"{s.avg_shadow_r:+7.3f} "
                f"{s.pct_reach_tp1:4.0f}% {s.pct_reach_tp2:4.0f}% "
                f"{s.pct_above_1r:4.0f}% "
                f"{s.net_missed_expectancy:+8.1f} {s.net_avoided_loss:8.1f}"
            )

        lines.append("")
        total_rej = sum(s.rejected_count for s in summaries.values())
        total_filled = sum(s.filled_count for s in summaries.values())
        lines.append(f"  Total rejections: {total_rej}  Filled in sim: {total_filled}")

        return "\n".join(lines)
