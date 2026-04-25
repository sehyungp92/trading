"""Shadow trade simulation for Helix v3.1 rejected candidates.

Tracks what would have happened if a gate had NOT rejected a setup.
Uses Helix stop lifecycle: profit target (+0.75R), +1R BE,
chandelier trailing, stale exit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from strategies.momentum.helix_v40.config import (
    STALE_M_BARS, STALE_R_THRESHOLD,
    TRAIL_LOOKBACK_1H, TRAIL_MULT_FLOOR, TRAIL_R_DIVISOR,
    NQ_POINT_VALUE,
    CATCHUP_OVERSHOOT_ATR_FRAC, CATCHUP_OFFSET_PTS,
)

# Shadow-sim parameters (v3.1 legacy constants for shadow trade simulation)
PROFIT_TARGET_R = 2.00
BE_BUFFER_ATR_FRAC = 0.00
TRAIL_MULT_BASE_MAX = 3.0

logger = logging.getLogger(__name__)


@dataclass
class HelixShadowCandidate:
    """A setup rejected by one or more gates."""
    symbol: str
    direction: int
    filter_names: list[str]
    time: datetime
    entry_price: float
    stop_price: float
    setup_class: str = ""
    session_block: str = ""
    alignment_score: int = 0

    @property
    def filter_name(self) -> str:
        return self.filter_names[0] if self.filter_names else ""


@dataclass
class HelixShadowResult:
    """Outcome of simulating a shadow candidate."""
    candidate: HelixShadowCandidate
    filled: bool = False
    fill_price: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    reached_1r: bool = False
    hit_profit_target: bool = False


@dataclass
class HelixFilterStats:
    """Aggregated stats for one filter/gate."""
    filter_name: str
    rejected_count: int = 0
    simulated_count: int = 0
    filled_count: int = 0
    avg_shadow_r: float = 0.0
    median_shadow_r: float = 0.0
    pct_winners: float = 0.0
    pct_reach_1r: float = 0.0
    pct_hit_target: float = 0.0
    net_missed_expectancy: float = 0.0
    net_avoided_loss: float = 0.0


class HelixShadowTracker:
    """Track rejected Helix candidates and simulate their outcomes."""

    def __init__(self):
        self.rejections: list[HelixShadowCandidate] = []
        self.results: list[HelixShadowResult] = []

    def record_rejection(
        self,
        symbol: str,
        direction: int,
        filter_names: str | list[str],
        time: datetime,
        entry_price: float,
        stop_price: float,
        setup_class: str = "",
        session_block: str = "",
        alignment_score: int = 0,
    ) -> None:
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        self.rejections.append(HelixShadowCandidate(
            symbol=symbol, direction=direction, filter_names=filter_names,
            time=time, entry_price=entry_price, stop_price=stop_price,
            setup_class=setup_class, session_block=session_block,
            alignment_score=alignment_score,
        ))

    def simulate_shadows(
        self,
        hourly_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        hourly_times: dict[str, np.ndarray],
        minute_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
        minute_times: dict[str, np.ndarray] | None = None,
    ) -> list[HelixShadowResult]:
        """Forward-simulate each rejected candidate using Helix stop lifecycle."""
        self.results.clear()
        for cand in self.rejections:
            sym = cand.symbol
            if sym not in hourly_data or sym not in hourly_times:
                continue
            opens, highs, lows, closes, _ = hourly_data[sym]
            times = hourly_times[sym]
            m_ohlc = minute_data.get(sym) if minute_data else None
            m_times = minute_times.get(sym) if minute_times else None
            result = self._simulate_one(
                cand, opens, highs, lows, closes, times,
                m_ohlc=m_ohlc, m_times=m_times,
            )
            self.results.append(result)
        return self.results

    def _simulate_one(
        self,
        cand: HelixShadowCandidate,
        opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, times: np.ndarray,
        m_ohlc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
        m_times: np.ndarray | None = None,
    ) -> HelixShadowResult:
        """Simulate one shadow candidate with Helix stop lifecycle."""
        cand_ts = (
            np.datetime64(cand.time, "ns")
            if not isinstance(cand.time, np.datetime64)
            else cand.time
        )
        r_base = abs(cand.entry_price - cand.stop_price)
        if r_base <= 0:
            return HelixShadowResult(candidate=cand)

        # Phase 1: fill detection
        fill_price, fill_ts = self._check_fill(
            cand, cand_ts, m_ohlc, m_times, opens, highs, lows, times)
        if fill_price == 0.0:
            return HelixShadowResult(candidate=cand, filled=False)

        h_start = int(np.searchsorted(times, fill_ts, side="right"))
        if h_start >= len(times):
            return HelixShadowResult(
                candidate=cand, filled=True, fill_price=fill_price,
                exit_price=fill_price, exit_reason="END_OF_DATA")

        # Phase 2: Helix stop lifecycle (hourly bars)
        current_stop = cand.stop_price
        mfe_price = fill_price
        mae_price = fill_price
        mfe_r = 0.0
        mae_r = 0.0
        reached_1r = False
        hit_profit_target = False
        d = cand.direction

        for j in range(h_start, len(closes)):
            bars_held = j - h_start + 1

            # MFE / MAE
            if d == 1:
                mfe_price = max(mfe_price, float(highs[j]))
                mae_price = min(mae_price, float(lows[j]))
                cur_mfe = (mfe_price - fill_price) / r_base
                cur_mae = (fill_price - mae_price) / r_base
                cur_r = (float(closes[j]) - fill_price) / r_base
            else:
                mfe_price = min(mfe_price, float(lows[j]))
                mae_price = max(mae_price, float(highs[j]))
                cur_mfe = (fill_price - mfe_price) / r_base
                cur_mae = (mae_price - fill_price) / r_base
                cur_r = (fill_price - float(closes[j])) / r_base

            mfe_r = max(mfe_r, cur_mfe)
            mae_r = max(mae_r, cur_mae)

            # Stop fill check
            stopped = (d == 1 and lows[j] <= current_stop) or \
                      (d == -1 and highs[j] >= current_stop)
            if stopped:
                if d == 1:
                    pnl_r = (current_stop - fill_price) / r_base
                else:
                    pnl_r = (fill_price - current_stop) / r_base
                return HelixShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=current_stop, exit_reason="STOP",
                    r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                    bars_held=bars_held,
                    reached_1r=reached_1r, hit_profit_target=hit_profit_target,
                )

            # Profit target
            if cur_mfe >= PROFIT_TARGET_R:
                hit_profit_target = True
                if d == 1:
                    target_px = fill_price + PROFIT_TARGET_R * r_base
                else:
                    target_px = fill_price - PROFIT_TARGET_R * r_base
                return HelixShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=target_px, exit_reason="PROFIT_TARGET",
                    r_multiple=PROFIT_TARGET_R, mfe_r=mfe_r, mae_r=mae_r,
                    bars_held=bars_held,
                    reached_1r=reached_1r, hit_profit_target=True,
                )

            # +1R BE
            if not reached_1r and cur_mfe >= 1.0:
                reached_1r = True
                be_buffer = BE_BUFFER_ATR_FRAC * r_base
                if d == 1:
                    current_stop = max(current_stop, fill_price + be_buffer)
                else:
                    current_stop = min(current_stop, fill_price - be_buffer)

            # Chandelier trailing
            if reached_1r and bars_held > 2:
                mult = max(TRAIL_MULT_FLOOR, TRAIL_MULT_BASE_MAX - (cur_r / TRAIL_R_DIVISOR))
                lookback = min(TRAIL_LOOKBACK_1H, bars_held)
                start_lb = max(h_start, j - lookback)
                if d == 1:
                    hh = float(np.max(highs[start_lb:j + 1]))
                    trail = hh - mult * r_base
                    current_stop = max(current_stop, trail)
                else:
                    ll = float(np.min(lows[start_lb:j + 1]))
                    trail = ll + mult * r_base
                    current_stop = min(current_stop, trail)

            # Stale exit
            if bars_held >= STALE_M_BARS and cur_r < STALE_R_THRESHOLD:
                exit_px = float(closes[j])
                if d == 1:
                    pnl_r = (exit_px - fill_price) / r_base
                else:
                    pnl_r = (fill_price - exit_px) / r_base
                return HelixShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=exit_px, exit_reason="STALE",
                    r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                    bars_held=bars_held,
                    reached_1r=reached_1r, hit_profit_target=hit_profit_target,
                )

        # End of data
        last_close = float(closes[-1])
        if d == 1:
            pnl_r = (last_close - fill_price) / r_base
        else:
            pnl_r = (fill_price - last_close) / r_base
        return HelixShadowResult(
            candidate=cand, filled=True, fill_price=fill_price,
            exit_price=last_close, exit_reason="END_OF_DATA",
            r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
            bars_held=len(closes) - h_start + 1,
            reached_1r=reached_1r, hit_profit_target=hit_profit_target,
        )

    @staticmethod
    def _check_fill(
        cand: HelixShadowCandidate,
        cand_ts: np.datetime64,
        m_ohlc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        m_times: np.ndarray | None,
        h_opens: np.ndarray, h_highs: np.ndarray, h_lows: np.ndarray,
        h_times: np.ndarray,
    ) -> tuple[float, np.datetime64]:
        """Detect fill via stop trigger OR catch-up limit (overshoot within cap).

        Returns (fill_price, fill_ts); fill_price==0 means no fill.
        """
        r_base = abs(cand.entry_price - cand.stop_price)
        overshoot_cap = CATCHUP_OVERSHOOT_ATR_FRAC * r_base if r_base > 0 else 0.0

        # Minute-bar path
        if m_ohlc is not None and m_times is not None and len(m_times) > 0:
            m_opens, m_highs, m_lows, _ = m_ohlc
            m_start = int(np.searchsorted(m_times, cand_ts, side="right"))
            end = min(m_start + 72, len(m_opens))  # 6H TTL in 5-min bars
            for i in range(m_start, end):
                if cand.direction == 1:
                    if m_highs[i] >= cand.entry_price:
                        # Primary stop trigger
                        fp = max(float(m_opens[i]), cand.entry_price) if m_opens[i] >= cand.entry_price else cand.entry_price
                        return fp, m_times[i]
                    # Catch-up: price overshot but within cap
                    if overshoot_cap > 0 and float(m_opens[i]) > cand.entry_price:
                        overshoot = float(m_opens[i]) - cand.entry_price
                        if overshoot <= overshoot_cap:
                            fp = float(m_opens[i]) + CATCHUP_OFFSET_PTS
                            return fp, m_times[i]
                else:
                    if m_lows[i] <= cand.entry_price:
                        fp = min(float(m_opens[i]), cand.entry_price) if m_opens[i] <= cand.entry_price else cand.entry_price
                        return fp, m_times[i]
                    if overshoot_cap > 0 and float(m_opens[i]) < cand.entry_price:
                        overshoot = cand.entry_price - float(m_opens[i])
                        if overshoot <= overshoot_cap:
                            fp = float(m_opens[i]) - CATCHUP_OFFSET_PTS
                            return fp, m_times[i]
            return 0.0, cand_ts

        # Hourly fallback (6-bar TTL)
        h_start = int(np.searchsorted(h_times, cand_ts, side="right"))
        for i in range(h_start, min(h_start + 6, len(h_opens))):
            if cand.direction == 1:
                if h_highs[i] >= cand.entry_price:
                    fp = max(float(h_opens[i]), cand.entry_price) if h_opens[i] >= cand.entry_price else cand.entry_price
                    return fp, h_times[i]
            else:
                if h_lows[i] <= cand.entry_price:
                    fp = min(float(h_opens[i]), cand.entry_price) if h_opens[i] <= cand.entry_price else cand.entry_price
                    return fp, h_times[i]
        return 0.0, cand_ts

    def get_filter_summary(self) -> dict[str, HelixFilterStats]:
        """Compute per-gate stats from simulation results."""
        by_filter: dict[str, list[HelixShadowResult]] = {}
        for r in self.results:
            for name in r.candidate.filter_names:
                by_filter.setdefault(name, []).append(r)

        rej_counts: dict[str, int] = {}
        for c in self.rejections:
            for name in c.filter_names:
                rej_counts[name] = rej_counts.get(name, 0) + 1

        summaries: dict[str, HelixFilterStats] = {}
        for name, results in by_filter.items():
            filled = [r for r in results if r.filled]
            stats = HelixFilterStats(filter_name=name)
            stats.rejected_count = rej_counts.get(name, len(results))
            stats.simulated_count = len(results)
            stats.filled_count = len(filled)

            if filled:
                arr = np.array([r.r_multiple for r in filled])
                stats.avg_shadow_r = float(np.mean(arr))
                stats.median_shadow_r = float(np.median(arr))
                stats.pct_winners = float(np.mean(arr > 0)) * 100
                stats.pct_reach_1r = float(np.mean([r.reached_1r for r in filled])) * 100
                stats.pct_hit_target = float(np.mean([r.hit_profit_target for r in filled])) * 100
                stats.net_missed_expectancy = float(np.sum(arr[arr > 0]))
                stats.net_avoided_loss = float(np.sum(np.abs(arr[arr < 0])))

            summaries[name] = stats
        return summaries

    def format_summary(self) -> str:
        summaries = self.get_filter_summary()
        if not summaries:
            return "=== Helix Shadow Trade Summary ===\n  No shadow trades simulated."

        lines = ["=== Helix Shadow Trade Summary ==="]
        header = (
            f"  {'Gate':28s} {'Rej':>5s} {'Filled':>6s} {'AvgR':>7s} {'MedR':>7s} "
            f"{'Win%':>5s} {'1R%':>5s} {'Tgt%':>6s} "
            f"{'Missed':>8s} {'Avoided':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for name in sorted(summaries, key=lambda n: -summaries[n].rejected_count):
            s = summaries[name]
            lines.append(
                f"  {name:28s} {s.rejected_count:5d} {s.filled_count:6d} "
                f"{s.avg_shadow_r:+7.3f} {s.median_shadow_r:+7.3f} "
                f"{s.pct_winners:4.0f}% {s.pct_reach_1r:4.0f}% {s.pct_hit_target:5.0f}% "
                f"{s.net_missed_expectancy:+8.1f} {s.net_avoided_loss:8.1f}"
            )

        total_rej = sum(s.rejected_count for s in summaries.values())
        total_filled = sum(s.filled_count for s in summaries.values())
        lines.append(f"\n  Total rejections: {total_rej}  Filled in sim: {total_filled}")

        # Per-class summary
        class_results: dict[str, list[HelixShadowResult]] = {}
        for r in self.results:
            cls = r.candidate.setup_class
            class_results.setdefault(cls, []).append(r)
        if class_results:
            lines.append("\n  Per-class shadow breakdown:")
            for cls in sorted(class_results):
                filled = [r for r in class_results[cls] if r.filled]
                if filled:
                    avg = np.mean([r.r_multiple for r in filled])
                    wr = np.mean([r.r_multiple > 0 for r in filled]) * 100
                    lines.append(f"    Class {cls}: {len(filled)} filled, avgR={avg:+.3f}, WR={wr:.0f}%")

        return "\n".join(lines)
