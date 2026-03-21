"""Live trade diagnostics: bar-by-bar R curves, stop progression, entry quality, regime context.

Captures enriched per-trade data during live execution for offline weakness analysis.
Complements the DB-backed TradeRecorder with in-memory structured diagnostics.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import (
    PositionState, Setup, SetupClass, SessionBlock,
    NQ_POINT_VALUE,
)

logger = logging.getLogger(__name__)


# ── Per-bar snapshot ──────────────────────────────────────────────────

@dataclass
class TradeSnapshot:
    """State captured at each 1H bar for an open position."""
    bar_index: int              # bars_held_1h at snapshot time
    timestamp: str              # ISO format ET
    r_total: float              # total R (unrealized + realized partials)
    r_unrealized: float         # unrealized R only
    stop_price: float
    last_price: float
    trailing_active: bool
    trail_mult: float
    alignment_score: int


# ── Enriched trade record ────────────────────────────────────────────

@dataclass
class LiveTradeRecord:
    """Enriched trade record with full diagnostic data."""
    # Identity
    pos_id: str = ""
    setup_id: str = ""
    setup_class: str = ""
    direction: int = 0

    # Timing
    entry_ts: str = ""
    exit_ts: str = ""
    bars_held_1h: int = 0

    # Prices
    avg_entry: float = 0.0
    initial_stop: float = 0.0
    exit_price: float = 0.0

    # PnL
    r_multiple: float = 0.0
    pnl_usd: float = 0.0

    # MFE / MAE
    mfe_r: float = 0.0             # max favorable excursion in R
    mae_r: float = 0.0             # max adverse excursion in R (negative)
    mfe_bar: int = 0               # bar index where MFE occurred
    mae_bar: int = 0               # bar index where MAE occurred
    mfe_price: float = 0.0
    mae_price: float = 0.0

    # Exit
    exit_reason: str = ""

    # Stop progression milestones
    trail_activated_at_bar: int = -1      # -1 = never activated
    trail_activated_at_r: float = 0.0
    be_reached_at_bar: int = -1           # -1 = never reached breakeven
    partial_done_at_bar: int = -1
    partial_done_at_r: float = 0.0
    peak_r: float = 0.0                  # highest R reached
    peak_r_at_bar: int = 0
    time_decay_triggered_at_bar: int = -1

    # Entry quality metrics
    entry_alignment_score: int = 0
    entry_vol_pct: float = 0.0
    entry_session: str = ""
    entry_atr_1h: float = 0.0
    entry_atr_daily: float = 0.0
    pullback_depth_atr: float = 0.0       # pb depth / ATR1H
    macd_distance_at_entry: float = 0.0   # |MACD - signal| / ATR at entry
    stop_distance_atr: float = 0.0        # |entry - stop| / ATR1H

    # Slippage (populated on fill if available)
    planned_entry: float = 0.0
    slippage_pts: float = 0.0             # fill_price - planned_entry (directional)
    slippage_r_frac: float = 0.0          # slippage as fraction of unit1_risk

    # Teleport
    teleport_penalty: bool = False

    # R curve (bar-by-bar snapshots)
    snapshots: list[TradeSnapshot] = field(default_factory=list)

    # Regime at entry
    strong_trend: bool = False
    is_reentry: bool = False
    is_extended: bool = False


# ── Diagnostic tracker ───────────────────────────────────────────────

class DiagnosticTracker:
    """Tracks enriched diagnostics for all open positions.

    Usage:
        tracker = DiagnosticTracker(output_dir="logs/diagnostics")
        # On entry fill:
        tracker.start_tracking(pos, setup, context)
        # On each 1H bar (inside manage_all):
        tracker.record_snapshot(pos, r_total, r_unrealized, last_price, now_et)
        # On milestone events:
        tracker.record_trail_activation(pos, bar, r_val)
        tracker.record_partial(pos, bar, r_val)
        tracker.record_time_decay(pos, bar)
        # On exit:
        record = tracker.finalize(pos, exit_price, exit_reason, now_et)
    """

    def __init__(self, output_dir: str = "logs/diagnostics", point_value: float = NQ_POINT_VALUE):
        self._active: dict[str, LiveTradeRecord] = {}  # pos_id -> record
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._completed: list[LiveTradeRecord] = []
        self.pv = point_value

    def start_tracking(
        self,
        pos: PositionState,
        setup: Setup,
        vol_pct: float,
        session_block: str,
        atr_1h: float,
        atr_daily: float,
        macd_distance: float = 0.0,
        pullback_depth_atr: float = 0.0,
    ) -> None:
        """Begin tracking a new position from entry fill."""
        stop_dist = abs(setup.entry_stop - setup.stop0)

        rec = LiveTradeRecord(
            pos_id=pos.pos_id,
            setup_id=setup.setup_id,
            setup_class=setup.cls.value,
            direction=pos.direction,
            entry_ts=pos.entry_ts.isoformat() if pos.entry_ts else "",
            avg_entry=pos.avg_entry,
            initial_stop=setup.stop0,
            entry_alignment_score=setup.alignment_score,
            entry_vol_pct=vol_pct,
            entry_session=session_block,
            entry_atr_1h=atr_1h,
            entry_atr_daily=atr_daily,
            pullback_depth_atr=pullback_depth_atr,
            macd_distance_at_entry=macd_distance,
            stop_distance_atr=stop_dist / max(atr_1h, 1e-9),
            planned_entry=setup.entry_stop,
            slippage_pts=(pos.avg_entry - setup.entry_stop) * pos.direction,
            slippage_r_frac=(
                abs(pos.avg_entry - setup.entry_stop) * self.pv
                / max(setup.unit1_risk_usd, 1e-9)
            ),
            teleport_penalty=pos.teleport_penalty,
            strong_trend=setup.strong_trend,
            is_reentry=setup.is_reentry,
            is_extended=setup.is_extended,
            mfe_price=pos.avg_entry,
            mae_price=pos.avg_entry,
        )
        self._active[pos.pos_id] = rec
        logger.debug("Diagnostics tracking started: %s", pos.pos_id)

    def record_snapshot(
        self,
        pos: PositionState,
        r_total: float,
        r_unrealized: float,
        last_price: float,
        now_et: datetime,
    ) -> None:
        """Record a bar-by-bar snapshot (call on each 1H bar in manage_all)."""
        rec = self._active.get(pos.pos_id)
        if rec is None:
            return

        snap = TradeSnapshot(
            bar_index=pos.bars_held_1h,
            timestamp=now_et.isoformat(),
            r_total=round(r_total, 4),
            r_unrealized=round(r_unrealized, 4),
            stop_price=pos.stop_price,
            last_price=last_price,
            trailing_active=pos.trailing_active,
            trail_mult=pos.trail_mult,
            alignment_score=pos.current_alignment_score,
        )
        rec.snapshots.append(snap)

        # Update MFE / MAE
        if r_total > rec.mfe_r:
            rec.mfe_r = r_total
            rec.mfe_bar = pos.bars_held_1h
            rec.mfe_price = last_price
        if r_total < rec.mae_r:
            rec.mae_r = r_total
            rec.mae_bar = pos.bars_held_1h
            rec.mae_price = last_price

        # Peak R
        if r_total > rec.peak_r:
            rec.peak_r = r_total
            rec.peak_r_at_bar = pos.bars_held_1h

        # BE milestone (first time R >= 0 after being negative)
        if rec.be_reached_at_bar == -1 and rec.mae_r < 0 and r_total >= 0:
            rec.be_reached_at_bar = pos.bars_held_1h

    def record_trail_activation(self, pos: PositionState, r_val: float) -> None:
        """Record when trailing stop activates."""
        rec = self._active.get(pos.pos_id)
        if rec and rec.trail_activated_at_bar == -1:
            rec.trail_activated_at_bar = pos.bars_held_1h
            rec.trail_activated_at_r = r_val

    def record_partial(self, pos: PositionState, r_val: float) -> None:
        """Record when partial exit executes."""
        rec = self._active.get(pos.pos_id)
        if rec and rec.partial_done_at_bar == -1:
            rec.partial_done_at_bar = pos.bars_held_1h
            rec.partial_done_at_r = r_val

    def record_time_decay(self, pos: PositionState) -> None:
        """Record when time-decay stop tightening first triggers."""
        rec = self._active.get(pos.pos_id)
        if rec and rec.time_decay_triggered_at_bar == -1:
            rec.time_decay_triggered_at_bar = pos.bars_held_1h

    def finalize(
        self,
        pos: PositionState,
        exit_price: float,
        exit_reason: str,
        now_et: datetime,
    ) -> Optional[LiveTradeRecord]:
        """Finalize trade record on position close. Returns the completed record."""
        rec = self._active.pop(pos.pos_id, None)
        if rec is None:
            return None

        rec.exit_ts = now_et.isoformat()
        rec.exit_price = exit_price
        rec.exit_reason = exit_reason
        rec.bars_held_1h = pos.bars_held_1h

        # Compute final R
        pts = (exit_price - pos.avg_entry) * pos.direction
        pnl_usd = pts * self.pv * pos.contracts + pos.realized_partial_usd
        rec.pnl_usd = pnl_usd
        rec.r_multiple = pnl_usd / max(pos.unit1_risk_usd, 1e-9)

        self._completed.append(rec)
        self._write_record(rec)
        logger.info(
            "Diagnostics finalized: %s class=%s R=%.3f exit=%s bars=%d",
            pos.pos_id, rec.setup_class, rec.r_multiple, exit_reason, rec.bars_held_1h,
        )
        return rec

    def _write_record(self, rec: LiveTradeRecord) -> None:
        """Append completed record to JSONL file (one per day)."""
        try:
            date_str = rec.exit_ts[:10] if rec.exit_ts else "unknown"
            path = self._output_dir / f"trades_{date_str}.jsonl"
            data = asdict(rec)
            # Convert TradeSnapshot objects to dicts
            data["snapshots"] = [asdict(s) for s in rec.snapshots]
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write diagnostic record: %s", e)

    @property
    def completed_records(self) -> list[LiveTradeRecord]:
        return list(self._completed)

    def active_count(self) -> int:
        return len(self._active)


# ── Offline analysis helpers ──────────────────────────────────────────

def load_records(diagnostics_dir: str) -> list[dict]:
    """Load all completed trade records from JSONL files."""
    records = []
    path = Path(diagnostics_dir)
    for f in sorted(path.glob("trades_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def weakness_report(records: list[dict]) -> dict:
    """Analyze completed records for actionable weaknesses.

    Returns a dict of analysis sections, each with key findings.
    """
    if not records:
        return {"error": "No records to analyze"}

    report: dict = {}

    # 1. R-curve give-back: trades that reached high R then gave it back
    giveback = []
    for r in records:
        peak = r.get("peak_r", 0)
        final = r.get("r_multiple", 0)
        if peak > 0.5 and final < peak * 0.3:
            giveback.append({
                "pos_id": r["pos_id"],
                "peak_r": peak,
                "final_r": final,
                "peak_bar": r.get("peak_r_at_bar", 0),
                "bars_held": r.get("bars_held_1h", 0),
                "exit_reason": r.get("exit_reason", ""),
                "giveback_pct": (peak - final) / peak if peak > 0 else 0,
            })
    report["r_giveback"] = {
        "count": len(giveback),
        "total_trades": len(records),
        "pct": len(giveback) / len(records) if records else 0,
        "worst": sorted(giveback, key=lambda x: -x["giveback_pct"])[:5],
    }

    # 2. Hold-time segmentation
    short_trades = [r for r in records if r.get("bars_held_1h", 0) <= 6]
    mid_trades = [r for r in records if 6 < r.get("bars_held_1h", 0) <= 16]
    long_trades = [r for r in records if r.get("bars_held_1h", 0) > 16]
    for label, group in [("short_0_6h", short_trades), ("mid_7_16h", mid_trades), ("long_17h_plus", long_trades)]:
        if group:
            rs = [r.get("r_multiple", 0) for r in group]
            report[f"holdtime_{label}"] = {
                "count": len(group),
                "avg_r": sum(rs) / len(rs),
                "win_rate": sum(1 for x in rs if x > 0) / len(rs),
                "avg_bars": sum(r.get("bars_held_1h", 0) for r in group) / len(group),
            }

    # 3. Trail activation effectiveness
    trailed = [r for r in records if r.get("trail_activated_at_bar", -1) >= 0]
    not_trailed = [r for r in records if r.get("trail_activated_at_bar", -1) < 0]
    for label, group in [("trailed", trailed), ("not_trailed", not_trailed)]:
        if group:
            rs = [r.get("r_multiple", 0) for r in group]
            report[f"trail_{label}"] = {
                "count": len(group),
                "avg_r": sum(rs) / len(rs),
                "win_rate": sum(1 for x in rs if x > 0) / len(rs),
            }

    # 4. Slippage impact
    slips = [r.get("slippage_r_frac", 0) for r in records if r.get("slippage_r_frac", 0) != 0]
    if slips:
        report["slippage"] = {
            "count": len(slips),
            "avg_r_frac": sum(slips) / len(slips),
            "max_r_frac": max(slips),
            "total_r_cost": sum(slips),
        }

    # 5. Exit reason breakdown
    exit_counts: dict[str, list[float]] = {}
    for r in records:
        reason = r.get("exit_reason", "unknown")
        exit_counts.setdefault(reason, []).append(r.get("r_multiple", 0))
    report["exit_reasons"] = {
        reason: {
            "count": len(rs),
            "avg_r": sum(rs) / len(rs),
            "win_rate": sum(1 for x in rs if x > 0) / len(rs),
        }
        for reason, rs in sorted(exit_counts.items(), key=lambda x: -len(x[1]))
    }

    # 6. Session performance
    session_groups: dict[str, list[float]] = {}
    for r in records:
        sess = r.get("entry_session", "unknown")
        session_groups.setdefault(sess, []).append(r.get("r_multiple", 0))
    report["session_performance"] = {
        sess: {
            "count": len(rs),
            "avg_r": sum(rs) / len(rs),
            "win_rate": sum(1 for x in rs if x > 0) / len(rs),
        }
        for sess, rs in sorted(session_groups.items(), key=lambda x: -len(x[1]))
    }

    # 7. Alignment score performance
    align_groups: dict[int, list[float]] = {}
    for r in records:
        score = r.get("entry_alignment_score", 0)
        align_groups.setdefault(score, []).append(r.get("r_multiple", 0))
    report["alignment_performance"] = {
        str(score): {
            "count": len(rs),
            "avg_r": sum(rs) / len(rs),
            "win_rate": sum(1 for x in rs if x > 0) / len(rs),
        }
        for score, rs in sorted(align_groups.items())
    }

    # 8. Time-decay impact
    decayed = [r for r in records if r.get("time_decay_triggered_at_bar", -1) >= 0]
    if decayed:
        rs = [r.get("r_multiple", 0) for r in decayed]
        report["time_decay_impact"] = {
            "count": len(decayed),
            "avg_r": sum(rs) / len(rs),
            "avg_bar_triggered": sum(r.get("time_decay_triggered_at_bar", 0) for r in decayed) / len(decayed),
            "pct_of_trades": len(decayed) / len(records),
        }

    # 9. Entry quality: stop distance vs outcome
    tight_stop = [r for r in records if r.get("stop_distance_atr", 0) < 0.5]
    wide_stop = [r for r in records if r.get("stop_distance_atr", 0) >= 0.5]
    for label, group in [("tight_stop_lt_0.5atr", tight_stop), ("wide_stop_gte_0.5atr", wide_stop)]:
        if group:
            rs = [r.get("r_multiple", 0) for r in group]
            report[f"stop_distance_{label}"] = {
                "count": len(group),
                "avg_r": sum(rs) / len(rs),
                "win_rate": sum(1 for x in rs if x > 0) / len(rs),
            }

    return report
