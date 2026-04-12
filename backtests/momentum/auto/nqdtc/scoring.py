"""NQDTC composite scoring -- 7 orthogonal components targeting diagnosed weaknesses.

Components map to NQDTC structural weaknesses:
  net_profit    (25%): overall profitability (log-scaled)
  pf            (15%): profit factor consistency
  calmar        (15%): return/drawdown ratio
  inv_dd        (10%): inverse max drawdown
  frequency     (15%): trade count (penalizes over-filtering)
  capture       (10%): MFE capture ratio (winners exit_R / MFE)
  entry_quality (10%): WR * clipped avgR -- balanced signal quality

Hard rejects: <15 trades, DD > 35%, PF < 0.80.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NQDTCMetrics:
    """NQDTC-specific performance metrics for scoring and diagnostics."""

    # Core performance
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    net_return_pct: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    avg_r: float = 0.0

    # NQDTC-specific (from trade records)
    capture_ratio: float = 0.0       # mean(R/MFE) for winners with MFE > 0
    burst_trade_pct: float = 0.0     # fraction of trades in bursts (3+ within 4h)
    eth_short_wr: float = 0.0        # ETH shorts win rate
    eth_short_trades: int = 0        # ETH shorts count
    range_regime_pct: float = 0.0    # fraction of trades in Range regime
    tp1_hit_rate: float = 0.0        # TP1 fill rate
    tp2_hit_rate: float = 0.0        # TP2 fill rate
    avg_hold_hours: float = 0.0      # average hold duration in hours
    avg_winner_r: float = 0.0        # average R for winners
    avg_loser_r: float = 0.0         # average R for losers
    avg_mfe_r: float = 0.0           # average MFE in R for all trades


@dataclass(frozen=True)
class NQDTCCompositeScore:
    """Frozen 8-component composite score for NQDTC."""

    net_profit: float = 0.0
    pf: float = 0.0
    calmar: float = 0.0
    inv_dd: float = 0.0
    frequency: float = 0.0
    capture: float = 0.0
    sortino: float = 0.0
    entry_quality: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


BASE_WEIGHTS = {
    "net_profit": 0.20,
    "pf": 0.15,
    "calmar": 0.15,
    "inv_dd": 0.10,
    "frequency": 0.10,
    "capture": 0.10,
    "sortino": 0.10,
    "entry_quality": 0.10,
}


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(
    metrics: NQDTCMetrics,
    weight_overrides: dict[str, float] | None = None,
) -> NQDTCCompositeScore:
    """Compute 7-component composite score for NQDTC."""
    w = dict(BASE_WEIGHTS)
    if weight_overrides:
        w.update(weight_overrides)

    # Hard rejects (immutable across phases -- caller may override via hard_rejects)
    if metrics.total_trades < 15:
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"too_few_trades ({metrics.total_trades})",
        )
    if metrics.max_dd_pct > 0.35:
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"max_dd_exceeded ({metrics.max_dd_pct:.2%})",
        )
    if metrics.profit_factor < 0.80:
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"low_pf ({metrics.profit_factor:.2f})",
        )

    # --- Components ---

    # Net profit: log scale, 1.0 at 800% return (log(9)/log(9))
    net_profit_c = _clip01(math.log(1 + max(metrics.net_return_pct, 0) / 100) / math.log(9))

    # Profit factor: (PF-1)/2, 1.0 at PF=3
    pf_c = _clip01((metrics.profit_factor - 1) / 2.0)

    # Calmar: calmar/18, 1.0 at calmar=18 (baseline 10.7 -> 0.59)
    calmar_c = _clip01(metrics.calmar / 18.0)

    # Inverse drawdown: 1 - DD/0.25, 1.0 at 0% DD
    inv_dd_c = _clip01(1 - metrics.max_dd_pct / 0.25)

    # Frequency: trades/300, 1.0 at 300 trades
    frequency_c = _clip01(metrics.total_trades / 300.0)

    # Capture: mean(winner_R / winner_MFE), direct 0-1 ratio
    capture_c = _clip01(metrics.capture_ratio)

    # Sortino: sortino/8, 1.0 at sortino=8
    sortino_c = _clip01(metrics.sortino / 8.0)

    # Entry quality: WR * min(1+avgR, 2) / 2
    entry_quality_c = _clip01(
        metrics.win_rate * min(1 + max(metrics.avg_r, 0), 2.0) / 2.0
    )

    total = (
        w["net_profit"] * net_profit_c
        + w["pf"] * pf_c
        + w["calmar"] * calmar_c
        + w["inv_dd"] * inv_dd_c
        + w["frequency"] * frequency_c
        + w["capture"] * capture_c
        + w.get("sortino", 0.0) * sortino_c
        + w["entry_quality"] * entry_quality_c
    )

    return NQDTCCompositeScore(
        net_profit=net_profit_c,
        pf=pf_c,
        calmar=calmar_c,
        inv_dd=inv_dd_c,
        frequency=frequency_c,
        capture=capture_c,
        sortino=sortino_c,
        entry_quality=entry_quality_c,
        total=total,
    )


def extract_nqdtc_metrics(
    trades: list,
    equity_curve: list[float],
    timestamps: list,
    initial_equity: float,
) -> NQDTCMetrics:
    """Extract NQDTCMetrics from trade records and equity curve."""
    if not trades:
        return NQDTCMetrics()

    total = len(trades)
    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]
    win_count = len(winners)
    win_rate = win_count / total if total > 0 else 0.0

    gross_profit = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
    gross_loss = abs(sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    avg_r = sum(t.r_multiple for t in trades) / total if total > 0 else 0.0
    avg_winner_r = sum(t.r_multiple for t in winners) / win_count if win_count else 0.0
    avg_loser_r = sum(t.r_multiple for t in losers) / len(losers) if losers else 0.0
    avg_mfe_r = sum(t.mfe_r for t in trades) / total if total > 0 else 0.0

    # Equity curve stats
    eq = equity_curve if len(equity_curve) > 0 else [initial_equity]
    peak = initial_equity
    max_dd = 0.0
    for e in eq:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    final_eq = eq[-1] if len(eq) > 0 else initial_equity
    net_return_pct = ((final_eq - initial_equity) / initial_equity) * 100.0

    calmar = (net_return_pct / 100.0) / max_dd if max_dd > 0 else 99.0

    # Sharpe/Sortino from trade R-multiples
    r_vals = [t.r_multiple for t in trades]
    mean_r = avg_r
    if len(r_vals) >= 2:
        import numpy as np
        std_r = float(np.std(r_vals, ddof=1))
        # Annualized: mean_r * trades_per_year / (std_r * sqrt(trades_per_year))
        if timestamps is not None and len(timestamps) >= 2:
            span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
            trades_per_year = total / (span_days / 365.25) if span_days > 0 else total
        else:
            trades_per_year = total
        sharpe = (mean_r * trades_per_year) / (std_r * trades_per_year ** 0.5) if std_r > 0 else 0.0
        downside = [r for r in r_vals if r < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) >= 2 else std_r
        sortino = (mean_r * trades_per_year) / (downside_std * trades_per_year ** 0.5) if downside_std > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    # Capture ratio: mean(R/MFE) for winners with MFE > 0
    capture_vals = [t.r_multiple / t.mfe_r for t in winners if t.mfe_r > 0.01]
    capture_ratio = sum(capture_vals) / len(capture_vals) if capture_vals else 0.0

    # Burst detection: 3+ trades within 4h -- O(n log n) via bisect
    entry_times = sorted(t.entry_time for t in trades)
    burst_count = 0
    if entry_times:
        from bisect import bisect_right
        from datetime import timedelta
        _4h = timedelta(seconds=14400)
        for i, t_i in enumerate(entry_times):
            if bisect_right(entry_times, t_i + _4h) - i >= 3:
                burst_count += 1
    burst_trade_pct = burst_count / total if total > 0 else 0.0

    # ETH short stats
    eth_shorts = [t for t in trades if t.session == "ETH" and t.direction == -1]
    eth_short_trades = len(eth_shorts)
    eth_short_wins = sum(1 for t in eth_shorts if t.r_multiple > 0)
    eth_short_wr = eth_short_wins / eth_short_trades if eth_short_trades > 0 else 0.0

    # Range regime fraction
    range_trades = sum(1 for t in trades if t.composite_regime == "Range")
    range_regime_pct = range_trades / total if total > 0 else 0.0

    # TP hit rates (use NQDTCTradeRecord bool fields, not exit_reason string)
    tp1_hits = sum(1 for t in trades if getattr(t, 'tp1_hit', False))
    tp2_hits = sum(1 for t in trades if getattr(t, 'tp2_hit', False))
    tp1_hit_rate = tp1_hits / total if total > 0 else 0.0
    tp2_hit_rate = tp2_hits / total if total > 0 else 0.0

    # Average hold hours
    hold_hours = []
    for t in trades:
        if hasattr(t, 'entry_time') and hasattr(t, 'exit_time') and t.exit_time:
            dt = (t.exit_time - t.entry_time).total_seconds() / 3600.0
            hold_hours.append(dt)
    avg_hold_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0

    return NQDTCMetrics(
        total_trades=total,
        win_rate=win_rate,
        profit_factor=pf,
        max_dd_pct=max_dd,
        net_return_pct=net_return_pct,
        calmar=calmar,
        sharpe=sharpe,
        sortino=sortino,
        avg_r=avg_r,
        capture_ratio=capture_ratio,
        burst_trade_pct=burst_trade_pct,
        eth_short_wr=eth_short_wr,
        eth_short_trades=eth_short_trades,
        range_regime_pct=range_regime_pct,
        tp1_hit_rate=tp1_hit_rate,
        tp2_hit_rate=tp2_hit_rate,
        avg_hold_hours=avg_hold_hours,
        avg_winner_r=avg_winner_r,
        avg_loser_r=avg_loser_r,
        avg_mfe_r=avg_mfe_r,
    )
