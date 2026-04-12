"""ATRSS composite scoring -- 7-component score, immutable across all 4 phases.

Components (sum to 1.0):
  - net_r (0.20): log(1 + total_r/100) / log(5)   -- 400R maps to ~1.0
  - profit_factor (0.14): (pf - 1) / 8             -- PF=9 maps to 1.0
  - calmar_r (0.14): calmar_r / 80                  -- 80 maps to 1.0
  - inv_dd (0.10): (1/dd - 10) / 90                 -- dd=1.1% maps to 1.0
  - frequency (0.18): tpm / 6                       -- 6 TPM maps to 1.0
  - mfe_capture (0.12): capture / 0.90              -- 90% maps to 1.0
  - win_rate (0.12): wr / 0.85                      -- 85% maps to 1.0

Scales calibrated for the corrected backtest engine (broker-mediated exits,
full commission accounting). Baseline scores ~0.65, giving the optimizer
headroom to discriminate mutations.

Hard rejects are phase-specific (see phase_scoring.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Weights -- immutable across all phases
# ---------------------------------------------------------------------------
W_NET_R = 0.20
W_PF = 0.14
W_CALMAR_R = 0.14
W_INV_DD = 0.10
W_FREQUENCY = 0.18
W_MFE_CAPTURE = 0.12
W_WIN_RATE = 0.12


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


@dataclass(frozen=True)
class ATRSSMetrics:
    """Metrics extracted from an ATRSS backtest run."""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    net_return_pct: float = 0.0
    total_r: float = 0.0
    max_dd_r: float = 0.0
    calmar_r: float = 0.0
    avg_r: float = 0.0
    mfe_capture: float = 0.0
    trades_per_month: float = 0.0


@dataclass(frozen=True)
class ATRSSCompositeScore:
    """Frozen composite score for ATRSS optimization."""
    net_r_component: float = 0.0
    pf_component: float = 0.0
    calmar_r_component: float = 0.0
    inv_dd_component: float = 0.0
    frequency_component: float = 0.0
    mfe_capture_component: float = 0.0
    win_rate_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


def composite_score(
    metrics: ATRSSMetrics,
    weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> ATRSSCompositeScore:
    """Compute the ATRSS composite score.

    Args:
        metrics: ATRSS metrics from backtest.
        weights: Optional weight overrides (unused -- ATRSS uses fixed weights).
        hard_rejects: Phase-specific hard reject thresholds.

    Returns:
        ATRSSCompositeScore with component values and total.
    """
    rejects = hard_rejects or {}
    min_trades = int(rejects.get("min_trades", 100))
    max_dd = rejects.get("max_dd_pct", 0.07)
    min_pf = rejects.get("min_pf", 2.0)
    min_wr = rejects.get("min_wr", 0.55)

    if metrics.total_trades < min_trades:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"Too few trades: {metrics.total_trades} < {min_trades}",
        )
    if metrics.max_dd_pct > max_dd:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"Max DD too high: {metrics.max_dd_pct:.1%} > {max_dd:.1%}",
        )
    if metrics.profit_factor < min_pf:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"PF too low: {metrics.profit_factor:.2f} < {min_pf:.2f}",
        )
    if metrics.win_rate < min_wr:
        return ATRSSCompositeScore(
            rejected=True,
            reject_reason=f"WR too low: {metrics.win_rate:.1%} < {min_wr:.0%}",
        )

    # --- Component normalization (all clipped to [0, 1]) ---
    # Weights are immutable -- ignore weight overrides for ATRSS
    w_net_r = W_NET_R
    w_pf = W_PF
    w_calmar_r = W_CALMAR_R
    w_inv_dd = W_INV_DD
    w_freq = W_FREQUENCY
    w_capture = W_MFE_CAPTURE
    w_wr = W_WIN_RATE

    # net_r: log(1 + total_r/100) / log(5)  -- 400R maps to ~1.0
    net_r_raw = _clip01(math.log(1.0 + max(metrics.total_r, 0.0) / 100.0) / math.log(5.0))

    # profit_factor: (pf - 1) / 8  -- PF=9 maps to 1.0
    pf_raw = _clip01((metrics.profit_factor - 1.0) / 8.0)

    # calmar_r: calmar_r / 80  -- calmar_r=80 maps to 1.0
    calmar_r_raw = _clip01(metrics.calmar_r / 80.0)

    # inv_dd: (1/dd - 10) / 90  -- dd=1.1% → (91-10)/90 = 1.0
    dd_frac = max(metrics.max_dd_pct, 0.001)
    inv_dd_raw = _clip01(((1.0 / dd_frac) - 10.0) / 90.0)

    # frequency: tpm / 6  -- 6 trades/month maps to 1.0
    freq_raw = _clip01(metrics.trades_per_month / 6.0)

    # mfe_capture: capture / 0.90  -- 90% capture maps to 1.0
    capture_raw = _clip01(metrics.mfe_capture / 0.90)

    # win_rate: wr / 0.85  -- 85% WR maps to 1.0
    wr_raw = _clip01(metrics.win_rate / 0.85)

    total = (
        w_net_r * net_r_raw
        + w_pf * pf_raw
        + w_calmar_r * calmar_r_raw
        + w_inv_dd * inv_dd_raw
        + w_freq * freq_raw
        + w_capture * capture_raw
        + w_wr * wr_raw
    )

    return ATRSSCompositeScore(
        net_r_component=net_r_raw,
        pf_component=pf_raw,
        calmar_r_component=calmar_r_raw,
        inv_dd_component=inv_dd_raw,
        frequency_component=freq_raw,
        mfe_capture_component=capture_raw,
        win_rate_component=wr_raw,
        total=total,
    )


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def extract_atrss_metrics(
    result,  # PortfolioResult
    initial_equity: float,
) -> ATRSSMetrics:
    """Extract ATRSSMetrics from a PortfolioResult.

    Uses TradeRecord fields directly (r_multiple, mfe_r, pnl_dollars)
    and PortfolioResult.combined_equity for equity-based metrics.
    """
    all_trades = []
    for sr in result.symbol_results.values():
        all_trades.extend(sr.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    n_trades = len(all_trades)
    if n_trades == 0:
        return ATRSSMetrics()

    # R-based metrics from TradeRecord.r_multiple
    r_mults = np.array([t.r_multiple for t in all_trades])
    wins_mask = r_mults > 0
    n_wins = int(np.sum(wins_mask))
    win_rate = n_wins / n_trades
    gross_win_r = float(np.sum(r_mults[wins_mask])) if n_wins > 0 else 0.0
    gross_loss_r = abs(float(np.sum(r_mults[~wins_mask])))
    profit_factor = gross_win_r / gross_loss_r if gross_loss_r > 0 else 999.0
    total_r = float(np.sum(r_mults))
    avg_r = float(np.mean(r_mults))

    # Max DD in R
    cum_r = np.cumsum(r_mults)
    peak_r = np.maximum.accumulate(cum_r)
    dd_r = peak_r - cum_r
    max_dd_r = float(np.max(dd_r)) if len(dd_r) > 0 else 0.0
    calmar_r = total_r / max_dd_r if max_dd_r > 0 else 0.0

    # Equity-based metrics from combined_equity
    eq = result.combined_equity
    if len(eq) > 1:
        peak_eq = np.maximum.accumulate(eq)
        dd_pct = (peak_eq - eq) / np.maximum(peak_eq, 1e-9)
        max_dd_pct = float(np.max(dd_pct))
        net_return_pct = (eq[-1] - initial_equity) / initial_equity * 100

        # Sharpe: annualized from bar-level returns
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 7))
        else:
            sharpe = 0.0

        # CAGR-based calmar
        ts = result.combined_timestamps
        if len(ts) > 1:
            delta = ts[-1] - ts[0]
            span_secs = delta.total_seconds() if hasattr(delta, 'total_seconds') else float(delta / np.timedelta64(1, 's'))
            span_years = span_secs / (365.25 * 86400)
            cagr = ((eq[-1] / initial_equity) ** (1 / max(span_years, 0.01)) - 1) if span_years > 0 else 0.0
            calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0
        else:
            calmar = 0.0
    else:
        max_dd_pct = 0.0
        net_return_pct = 0.0
        sharpe = 0.0
        calmar = 0.0

    # MFE capture: avg(r_multiple / mfe_r) for winning trades with mfe > 0
    captures = []
    for t in all_trades:
        if t.mfe_r > 0 and t.r_multiple > 0:
            captures.append(t.r_multiple / t.mfe_r)
    mfe_capture = float(np.mean(captures)) if captures else 0.0

    # Trades per month
    if n_trades >= 2:
        span_days = (all_trades[-1].entry_time - all_trades[0].entry_time).total_seconds() / 86400
        tpm = n_trades / max(span_days / 30.44, 1.0)
    else:
        tpm = 0.0

    return ATRSSMetrics(
        total_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_dd_pct=max_dd_pct,
        calmar=calmar,
        sharpe=sharpe,
        net_return_pct=net_return_pct,
        total_r=total_r,
        max_dd_r=max_dd_r,
        calmar_r=calmar_r,
        avg_r=avg_r,
        mfe_capture=mfe_capture,
        trades_per_month=tpm,
    )
