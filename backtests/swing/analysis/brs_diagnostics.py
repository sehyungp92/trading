"""BRS diagnostics — downturn-alpha focused analysis.

Focus: does BRS capture alpha during downturns?
1. Regime-conditional metrics
2. Crisis window returns
3. Entry type attribution
4. Bear alpha
5. Conviction correlation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO

import numpy as np

from backtest.engine.backtest_engine import SymbolResult, TradeRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crisis windows (ground truth downturn periods)
# ---------------------------------------------------------------------------

CRISIS_WINDOWS = [
    ("2022 Bear", datetime(2022, 1, 3), datetime(2022, 10, 13)),
    ("SVB", datetime(2023, 3, 8), datetime(2023, 3, 15)),
    ("Aug 2024 Unwind", datetime(2024, 8, 1), datetime(2024, 8, 5)),
    ("Tariff Shock", datetime(2025, 2, 21), datetime(2025, 4, 7)),
    ("Mar 2026 Slow Burn", datetime(2026, 3, 5), datetime(2026, 3, 27)),
]


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------

@dataclass
class RegimeMetrics:
    """Performance breakdown for a single regime bucket."""
    regime: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0
    avg_bars_held: float = 0.0


@dataclass
class EntryTypeMetrics:
    """Performance breakdown for a single entry type."""
    entry_type: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0


@dataclass
class ConvictionBucket:
    """Trades bucketed by conviction score at entry."""
    bucket: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    expectancy: float = 0.0


@dataclass
class CrisisResult:
    """BRS vs benchmark during a crisis window."""
    name: str
    start: datetime
    end: datetime
    brs_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0
    brs_trades: int = 0


@dataclass
class BRSDiagnostics:
    """Complete BRS diagnostics report."""
    regime_metrics: list[RegimeMetrics] = field(default_factory=list)
    entry_type_metrics: list[EntryTypeMetrics] = field(default_factory=list)
    conviction_buckets: list[ConvictionBucket] = field(default_factory=list)
    crisis_results: list[CrisisResult] = field(default_factory=list)
    bear_alpha_pct: float = 0.0
    total_trades: int = 0
    bear_trades: int = 0
    report: str = ""


# ---------------------------------------------------------------------------
# Compute diagnostics
# ---------------------------------------------------------------------------

def compute_brs_diagnostics(
    results: dict[str, SymbolResult],
    initial_equity: float,
    combined_equity: np.ndarray | None = None,
    combined_timestamps: np.ndarray | None = None,
) -> BRSDiagnostics:
    """Compute full BRS diagnostics from backtest results."""
    all_trades: list[TradeRecord] = []
    for sr in results.values():
        all_trades.extend(sr.trades)

    diag = BRSDiagnostics(total_trades=len(all_trades))

    # 1. Regime-conditional metrics
    diag.regime_metrics = _compute_regime_metrics(all_trades)

    # 2. Entry type attribution
    diag.entry_type_metrics = _compute_entry_type_metrics(all_trades)

    # 3. Conviction correlation
    diag.conviction_buckets = _compute_conviction_buckets(all_trades)

    # 4. Crisis window returns
    if combined_equity is not None and combined_timestamps is not None:
        diag.crisis_results = _compute_crisis_returns(
            combined_equity, combined_timestamps, initial_equity,
        )

    # 5. Bear alpha
    bear_trades = [t for t in all_trades if t.regime_entry in ("BEAR_STRONG", "BEAR_TREND", "BEAR_FORMING")]
    diag.bear_trades = len(bear_trades)
    if bear_trades:
        diag.bear_alpha_pct = sum(t.pnl_dollars for t in bear_trades) / initial_equity * 100

    # Build report
    diag.report = _build_report(diag)
    return diag


def _compute_regime_metrics(trades: list[TradeRecord]) -> list[RegimeMetrics]:
    """Split trades by regime_entry, compute metrics per regime."""
    from collections import defaultdict
    buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        buckets[t.regime_entry or "UNKNOWN"].append(t)

    results = []
    for regime, bucket in sorted(buckets.items()):
        n = len(bucket)
        wins = [t for t in bucket if t.r_multiple > 0]
        losses = [t for t in bucket if t.r_multiple <= 0]
        gross_win = sum(t.r_multiple for t in wins) if wins else 0
        gross_loss = abs(sum(t.r_multiple for t in losses)) if losses else 0

        results.append(RegimeMetrics(
            regime=regime,
            trade_count=n,
            win_rate=len(wins) / n * 100 if n > 0 else 0,
            avg_r=sum(t.r_multiple for t in bucket) / n if n > 0 else 0,
            total_r=sum(t.r_multiple for t in bucket),
            profit_factor=gross_win / gross_loss if gross_loss > 0 else 999,
            avg_bars_held=sum(t.bars_held for t in bucket) / n if n > 0 else 0,
        ))
    return results


def _compute_entry_type_metrics(trades: list[TradeRecord]) -> list[EntryTypeMetrics]:
    """Per entry type metrics."""
    from collections import defaultdict
    buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        buckets[t.entry_type or "UNKNOWN"].append(t)

    results = []
    for etype, bucket in sorted(buckets.items()):
        n = len(bucket)
        wins = [t for t in bucket if t.r_multiple > 0]
        losses = [t for t in bucket if t.r_multiple <= 0]
        gross_win = sum(t.r_multiple for t in wins) if wins else 0
        gross_loss = abs(sum(t.r_multiple for t in losses)) if losses else 0

        results.append(EntryTypeMetrics(
            entry_type=etype,
            trade_count=n,
            win_rate=len(wins) / n * 100 if n > 0 else 0,
            avg_r=sum(t.r_multiple for t in bucket) / n if n > 0 else 0,
            total_r=sum(t.r_multiple for t in bucket),
            profit_factor=gross_win / gross_loss if gross_loss > 0 else 999,
        ))
    return results


def _compute_conviction_buckets(trades: list[TradeRecord]) -> list[ConvictionBucket]:
    """Bucket trades by conviction score at entry."""
    buckets_def = [
        ("0-25", 0, 25),
        ("25-50", 25, 50),
        ("50-75", 50, 75),
        ("75-100", 75, 100),
    ]
    results = []
    for label, lo, hi in buckets_def:
        bucket = [t for t in trades if lo <= t.score_entry < hi or (hi == 100 and t.score_entry == 100)]
        n = len(bucket)
        wins = [t for t in bucket if t.r_multiple > 0]
        avg_r = sum(t.r_multiple for t in bucket) / n if n > 0 else 0
        wr = len(wins) / n if n > 0 else 0
        avg_win = sum(t.r_multiple for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.r_multiple for t in bucket if t.r_multiple <= 0) / max(1, n - len(wins)) if n > len(wins) else 0
        expectancy = wr * avg_win + (1 - wr) * avg_loss

        results.append(ConvictionBucket(
            bucket=label,
            trade_count=n,
            win_rate=wr * 100,
            avg_r=avg_r,
            expectancy=expectancy,
        ))
    return results


def _compute_crisis_returns(
    equity: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
) -> list[CrisisResult]:
    """Compare BRS equity vs buy-and-hold during crisis windows."""
    results = []
    for name, start, end in CRISIS_WINDOWS:
        # Find equity curve segment for this window
        mask = np.zeros(len(timestamps), dtype=bool)
        for i, ts in enumerate(timestamps):
            try:
                dt = ts.astype('datetime64[s]').astype(datetime) if hasattr(ts, 'astype') else ts
                if isinstance(dt, (np.datetime64,)):
                    dt = dt.astype('datetime64[s]').item()
                if start <= dt <= end:
                    mask[i] = True
            except (ValueError, TypeError):
                continue

        if not np.any(mask):
            results.append(CrisisResult(name=name, start=start, end=end))
            continue

        indices = np.where(mask)[0]
        start_eq = float(equity[indices[0]])
        end_eq = float(equity[indices[-1]])
        brs_ret = (end_eq - start_eq) / start_eq * 100 if start_eq > 0 else 0

        results.append(CrisisResult(
            name=name,
            start=start,
            end=end,
            brs_return_pct=brs_ret,
            brs_trades=0,  # populated by caller if needed
        ))
    return results


def _build_report(diag: BRSDiagnostics) -> str:
    """Build human-readable diagnostics report."""
    buf = StringIO()
    buf.write("=" * 70 + "\n")
    buf.write("BRS DIAGNOSTICS REPORT\n")
    buf.write("=" * 70 + "\n\n")

    buf.write(f"Total trades: {diag.total_trades}\n")
    buf.write(f"Bear trades:  {diag.bear_trades}\n")
    buf.write(f"Bear alpha:   {diag.bear_alpha_pct:.1f}%\n\n")

    # Regime metrics
    buf.write("--- Regime-Conditional Metrics ---\n")
    buf.write(f"{'Regime':<15} {'Count':>6} {'WR%':>6} {'AvgR':>7} {'TotalR':>8} {'PF':>6} {'AvgBars':>8}\n")
    for rm in diag.regime_metrics:
        buf.write(f"{rm.regime:<15} {rm.trade_count:>6} {rm.win_rate:>5.1f}% {rm.avg_r:>7.2f} "
                 f"{rm.total_r:>8.1f} {rm.profit_factor:>6.2f} {rm.avg_bars_held:>8.0f}\n")
    buf.write("\n")

    # Entry type metrics
    buf.write("--- Entry Type Attribution ---\n")
    buf.write(f"{'Type':<15} {'Count':>6} {'WR%':>6} {'AvgR':>7} {'TotalR':>8} {'PF':>6}\n")
    for em in diag.entry_type_metrics:
        buf.write(f"{em.entry_type:<15} {em.trade_count:>6} {em.win_rate:>5.1f}% {em.avg_r:>7.2f} "
                 f"{em.total_r:>8.1f} {em.profit_factor:>6.2f}\n")
    buf.write("\n")

    # Conviction buckets
    buf.write("--- Conviction Correlation ---\n")
    buf.write(f"{'Bucket':<10} {'Count':>6} {'WR%':>6} {'AvgR':>7} {'Expect':>8}\n")
    for cb in diag.conviction_buckets:
        buf.write(f"{cb.bucket:<10} {cb.trade_count:>6} {cb.win_rate:>5.1f}% "
                 f"{cb.avg_r:>7.2f} {cb.expectancy:>8.3f}\n")
    buf.write("\n")

    # Crisis results
    if diag.crisis_results:
        buf.write("--- Crisis Window Returns ---\n")
        buf.write(f"{'Crisis':<20} {'BRS%':>8} {'Alpha%':>8}\n")
        for cr in diag.crisis_results:
            buf.write(f"{cr.name:<20} {cr.brs_return_pct:>7.1f}% {cr.alpha_pct:>7.1f}%\n")

    return buf.getvalue()
