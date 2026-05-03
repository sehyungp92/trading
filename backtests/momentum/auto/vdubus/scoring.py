"""VdubusNQ immutable Round 2 composite scoring.

The Round 1 diagnostics showed that raw net return alone is not a reliable
guide: the strategy earns money through infrequent, high-MFE trend survivors,
while stale exits and fast deaths leak a lot of expectancy. Round 2 therefore
uses one immutable score in every phase, centered on R/month and trade supply,
with explicit penalties for the diagnosed leaks.

Components:
  r_per_month     (24%): avg R * trades/month, 1.0 at 2.50 R/month
  profit_factor   (18%): net win quality, 1.0 at PF 2.80
  calmar          (14%): return/drawdown balance, 1.0 at calmar 45
  capture_ratio   (12%): winner MFE capture, 1.0 at 65%
  frequency       (12%): trades/month, 1.0 at 8.0
  sharpe          (10%): trade-R Sharpe, 1.0 at 2.20
  inv_dd          (10%): inverse max drawdown, 1.0 at <=12% DD

Penalties discourage stale drift, fast deaths, negative evening flow, and very
low trade counts. Risk-sizing parameters are intentionally not rewarded because
the optimizer evaluates fixed 10 MNQ quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _span_days(timestamps) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    delta = timestamps[-1] - timestamps[0]
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds()) / 86400.0
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s")) / 86400.0
    if isinstance(delta, (int, float, np.integer, np.floating)):
        return float(delta) / 86400.0
    return 0.0


@dataclass
class VdubusMetrics:
    """VdubusNQ-specific performance metrics for scoring and diagnostics."""

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

    # VdubusNQ-specific (from trade records)
    capture_ratio: float = 0.0       # mean(R/MFE) for winners with MFE > 0
    stale_exit_pct: float = 0.0      # fraction of trades exiting via STALE
    multi_session_pct: float = 0.0   # fraction of trades spanning multiple sessions
    trades_per_month: float = 0.0    # annualized trade frequency
    avg_hold_hours: float = 0.0      # average hold duration in hours
    avg_winner_r: float = 0.0        # average R for winners
    avg_loser_r: float = 0.0         # average R for losers
    avg_mfe_r: float = 0.0           # average MFE in R for all trades
    evening_trade_pct: float = 0.0   # fraction of trades in evening session
    evening_avg_r: float = 0.0       # average R for evening trades
    fast_death_pct: float = 0.0      # fraction of trades held <= 4 bars


@dataclass(frozen=True)
class VdubusCompositeScore:
    """Frozen immutable composite score for VdubusNQ."""

    r_per_month: float = 0.0
    pf: float = 0.0
    calmar: float = 0.0
    inv_dd: float = 0.0
    capture: float = 0.0
    frequency: float = 0.0
    sharpe: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


BASE_WEIGHTS = {
    "r_per_month": 0.24,
    "pf": 0.18,
    "calmar": 0.14,
    "inv_dd": 0.10,
    "capture": 0.12,
    "frequency": 0.12,
    "sharpe": 0.10,
}


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(
    metrics: VdubusMetrics,
    weight_overrides: dict[str, float] | None = None,
) -> VdubusCompositeScore:
    """Compute 7-component composite score for VdubusNQ."""
    w = dict(BASE_WEIGHTS)
    if weight_overrides:
        w.update({k: v for k, v in weight_overrides.items() if k in w})
        total_w = sum(w.values())
        if total_w > 0:
            w = {k: v / total_w for k, v in w.items()}

    # Hard rejects (immutable across phases -- caller may override via hard_rejects)
    if metrics.total_trades < 40:
        return VdubusCompositeScore(
            rejected=True, reject_reason=f"too_few_trades ({metrics.total_trades})",
        )
    if metrics.max_dd_pct > 0.30:
        return VdubusCompositeScore(
            rejected=True, reject_reason=f"max_dd_exceeded ({metrics.max_dd_pct:.2%})",
        )
    if metrics.profit_factor < 0.80:
        return VdubusCompositeScore(
            rejected=True, reject_reason=f"low_pf ({metrics.profit_factor:.2f})",
        )

    # --- Components ---

    # R/month: expected R throughput, 1.0 at 2.50R/month.
    r_month = max(metrics.avg_r, 0.0) * max(metrics.trades_per_month, 0.0)
    r_month_c = _clip01(r_month / 2.50)

    # Profit factor: 1.0 at PF=2.80, zero below PF=1.20.
    pf_c = _clip01((metrics.profit_factor - 1.20) / 1.60)

    # Calmar: 1.0 at calmar=45.
    calmar_c = _clip01(metrics.calmar / 45.0)

    # Inverse drawdown: 1.0 at 12% DD or lower, zero at 25% DD.
    inv_dd_c = _clip01((0.25 - metrics.max_dd_pct) / 0.13)

    # Capture: 1.0 at 65% winner MFE capture.
    capture_c = _clip01(metrics.capture_ratio / 0.65)

    # Frequency: 1.0 at 8 trades/month, zero below 3.5 trades/month.
    frequency_c = _clip01((metrics.trades_per_month - 3.5) / 4.5)

    # Sharpe: 1.0 at 2.20.
    sharpe_c = _clip01(metrics.sharpe / 2.20)

    total = (
        w["r_per_month"] * r_month_c
        + w["pf"] * pf_c
        + w["calmar"] * calmar_c
        + w["inv_dd"] * inv_dd_c
        + w["capture"] * capture_c
        + w["frequency"] * frequency_c
        + w["sharpe"] * sharpe_c
    )

    stale_penalty = max(metrics.stale_exit_pct - 0.35, 0.0) * 0.18
    fast_death_penalty = max(metrics.fast_death_pct - 0.18, 0.0) * 0.20
    evening_penalty = 0.0
    if metrics.evening_trade_pct > 0.03 and metrics.evening_avg_r < 0.0:
        evening_penalty = min(0.05, abs(metrics.evening_avg_r) * metrics.evening_trade_pct * 2.0)
    low_frequency_penalty = max(3.5 - metrics.trades_per_month, 0.0) * 0.03

    total = _clip01(total - stale_penalty - fast_death_penalty - evening_penalty - low_frequency_penalty)

    return VdubusCompositeScore(
        r_per_month=r_month_c,
        pf=pf_c,
        calmar=calmar_c,
        inv_dd=inv_dd_c,
        capture=capture_c,
        frequency=frequency_c,
        sharpe=sharpe_c,
        total=total,
    )


def extract_vdubus_metrics(
    trades: list,
    equity_curve: list[float],
    timestamps: list,
    initial_equity: float,
) -> VdubusMetrics:
    """Extract VdubusMetrics from trade records and equity curve."""
    if not trades:
        return VdubusMetrics()

    total = len(trades)
    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]
    win_count = len(winners)
    win_rate = win_count / total if total > 0 else 0.0

    net_pnls = [float(t.pnl_dollars) - float(getattr(t, "commission", 0.0) or 0.0) for t in trades]
    gross_profit = sum(pnl for pnl in net_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in net_pnls if pnl < 0))
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
        std_r = float(np.std(r_vals, ddof=1))
        if timestamps is not None and len(timestamps) >= 2:
            span_days = _span_days(timestamps)
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

    # Stale exit fraction
    stale_count = sum(1 for t in trades if t.exit_reason == "STALE")
    stale_exit_pct = stale_count / total if total > 0 else 0.0

    # Multi-session: trades held > 1 overnight session
    multi_session = sum(1 for t in trades if getattr(t, 'overnight_sessions', 1) > 1)
    multi_session_pct = multi_session / total if total > 0 else 0.0

    # Trades per month
    if timestamps is not None and len(timestamps) >= 2:
        span_days = _span_days(timestamps)
        trades_per_month = total / (span_days / 30.44) if span_days > 0 else 0.0
    else:
        trades_per_month = 0.0

    # Average hold hours
    hold_hours = []
    for t in trades:
        if hasattr(t, 'entry_time') and hasattr(t, 'exit_time') and t.exit_time:
            dt = (t.exit_time - t.entry_time).total_seconds() / 3600.0
            hold_hours.append(dt)
    avg_hold_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0

    # Evening session stats
    evening_trades = [t for t in trades if getattr(t, 'entry_session', '') == 'EVENING'
                      or (hasattr(t, 'sub_window') and getattr(t, 'sub_window', '') == 'EVENING')]
    evening_trade_pct = len(evening_trades) / total if total > 0 else 0.0
    evening_avg_r = sum(t.r_multiple for t in evening_trades) / len(evening_trades) if evening_trades else 0.0

    # Fast deaths (held <= 4 bars)
    fast_deaths = sum(1 for t in trades if t.bars_held_15m <= 4)
    fast_death_pct = fast_deaths / total if total > 0 else 0.0

    return VdubusMetrics(
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
        stale_exit_pct=stale_exit_pct,
        multi_session_pct=multi_session_pct,
        trades_per_month=trades_per_month,
        avg_hold_hours=avg_hold_hours,
        avg_winner_r=avg_winner_r,
        avg_loser_r=avg_loser_r,
        avg_mfe_r=avg_mfe_r,
        evening_trade_pct=evening_trade_pct,
        evening_avg_r=evening_avg_r,
        fast_death_pct=fast_death_pct,
    )
