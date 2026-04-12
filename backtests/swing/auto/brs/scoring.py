"""BRS composite scoring — same immutable score as swing auto-research.

7-component structure (renormalized from swing/auto/scoring.py + bear_alpha + frequency + detection):
  - Net profit (25%): log(1+return)/log(4), equity curve total
  - Profit factor (13%): (PF-1)/2, win quality
  - Calmar ratio (9%): calmar/10, risk-adjusted return
  - Inverse drawdown (8%): 1-DD/0.30, low DD as reward
  - Bear alpha (12%): bear_alpha_pct/20, bear-regime profit contribution
  - Frequency (15%): trades/80, rewards higher trade count (R8: cap raised from 40)
  - Detection quality (18%*): 0.4*bias_latency + 0.3*trade_latency + 0.3*coverage [*R7.2 rebalanced]

Hard rejects: <10 trades, max DD > 30%
PF filtering is phase-specific (see phase_scoring.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class BRSCompositeScore:
    """Frozen composite score — same structure as swing CompositeScore."""
    calmar_component: float = 0.0
    pf_component: float = 0.0
    inv_dd_component: float = 0.0
    net_profit_component: float = 0.0
    bear_alpha_component: float = 0.0
    frequency_component: float = 0.0
    detection_quality_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


# Base weights — sum to ~1.0 (7 components: +bear_alpha +frequency +detection_quality)
# R8: shifted from safety (calmar/dd) → growth (profit/frequency)
W_NET_PROFIT = 0.25
W_PF = 0.13
W_CALMAR = 0.09
W_INV_DD = 0.08
W_BEAR_ALPHA = 0.12
W_FREQUENCY = 0.15
W_DETECTION = 0.18


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


@dataclass
class BRSMetrics:
    """Metrics extracted from a BRS backtest run."""
    total_trades: int = 0
    bear_trades: int = 0
    bear_trade_wr: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    bear_alpha_pct: float = 0.0
    regime_f1: float = 0.0
    exit_efficiency: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    net_return_pct: float = 0.0
    bear_pf: float = 0.0
    detection_latency_days: float = 999.0
    bias_latency_days: float = 999.0   # crisis_start → first SHORT bias confirmation
    bear_capture_ratio: float = 0.0
    crisis_coverage: float = 0.0  # fraction of crisis windows with ≥1 trade


def composite_score(
    metrics: BRSMetrics,
    weights: dict[str, float] | None = None,
) -> BRSCompositeScore:
    """Compute the BRS composite score — same formula as swing/auto/scoring.py.

    Args:
        metrics: BRS metrics from backtest
        weights: Optional weight overrides (for phase-specific scoring)

    Returns:
        BRSCompositeScore with component values and total.
    """
    # Hard rejects
    if metrics.total_trades < 10:
        return BRSCompositeScore(
            rejected=True,
            reject_reason=f"Too few trades: {metrics.total_trades} < 10",
        )
    if metrics.max_dd_pct > 0.30:
        return BRSCompositeScore(
            rejected=True,
            reject_reason=f"Max DD too high: {metrics.max_dd_pct:.1%} > 30%",
        )
    # PF filtering is phase-specific (see phase_scoring.py PHASE_HARD_REJECTS).
    # The PF component already gives 0 credit for PF < 1.0, which is sufficient
    # penalty. A hard reject here would prevent the optimizer from bootstrapping
    # when the strategy is finding its footing (e.g. first bear trades losing money).

    w = weights or {}
    w_np = w.get("net_profit", W_NET_PROFIT)
    w_pf = w.get("pf", W_PF)
    w_ca = w.get("calmar", W_CALMAR)
    w_dd = w.get("inv_dd", W_INV_DD)
    w_ba = w.get("bear_alpha", W_BEAR_ALPHA)
    w_fr = w.get("frequency", W_FREQUENCY)
    w_dq = w.get("detection_quality", W_DETECTION)

    # Component scores — same normalizations as swing/auto/scoring.py
    # Net profit: log scale, 300% (4x money) → 1.0
    np_return = max(metrics.net_return_pct / 100.0, 0.0)
    np_c = _clip01(math.log(1.0 + np_return) / math.log(4.0))
    # Profit factor: PF=3 → 1.0
    pf_c = _clip01((metrics.profit_factor - 1.0) / 2.0)
    # Calmar: calmar=10 → 1.0
    calmar_c = _clip01(metrics.calmar / 10.0)
    # Inverse drawdown: 0% DD → 1.0
    inv_dd_c = _clip01(1.0 - metrics.max_dd_pct / 0.30)
    # Bear alpha: bear_alpha_pct=20% → 1.0
    bear_alpha_c = _clip01(metrics.bear_alpha_pct / 20.0)
    # Frequency: 80 trades → 1.0 (R8: raised from 40 to reward higher trade counts)
    freq_c = _clip01(metrics.total_trades / 80.0)
    # Detection quality: 3-part with bias_latency + trade_latency + coverage
    bias_lat_score = _clip01(1.0 - metrics.bias_latency_days / 20.0)
    trade_lat_score = _clip01(1.0 - metrics.detection_latency_days / 30.0)
    coverage_score = _clip01(metrics.crisis_coverage)
    dq_c = 0.4 * bias_lat_score + 0.3 * trade_lat_score + 0.3 * coverage_score

    total = (
        w_np * np_c
        + w_pf * pf_c
        + w_ca * calmar_c
        + w_dd * inv_dd_c
        + w_ba * bear_alpha_c
        + w_fr * freq_c
        + w_dq * dq_c
    )

    return BRSCompositeScore(
        net_profit_component=np_c,
        pf_component=pf_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        bear_alpha_component=bear_alpha_c,
        frequency_component=freq_c,
        detection_quality_component=dq_c,
        total=total,
    )


def extract_brs_metrics(
    result,
    initial_equity: float,
) -> BRSMetrics:
    """Extract BRSMetrics from a BRS portfolio result.

    Args:
        result: BRSPortfolioResult from run_brs_independent
        initial_equity: Starting equity

    Returns:
        BRSMetrics populated from backtest results
    """
    all_trades = []
    for sr in result.symbol_results.values():
        all_trades.extend(sr.trades)

    if not all_trades:
        return BRSMetrics()

    bear_trades = [t for t in all_trades if t.regime_entry in ("BEAR_STRONG", "BEAR_TREND", "BEAR_FORMING")]
    wins = [t for t in all_trades if t.r_multiple > 0]
    losses = [t for t in all_trades if t.r_multiple <= 0]
    bear_wins = [t for t in bear_trades if t.r_multiple > 0]

    gross_win = sum(t.r_multiple for t in wins) if wins else 0
    gross_loss = abs(sum(t.r_multiple for t in losses)) if losses else 0
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    bear_gw = sum(t.r_multiple for t in bear_wins) if bear_wins else 0
    bear_gl = abs(sum(t.r_multiple for t in bear_trades if t.r_multiple <= 0)) if bear_trades else 0
    bear_pf = bear_gw / bear_gl if bear_gl > 0 else 999

    # Max drawdown
    eq = result.combined_equity
    if len(eq) > 0:
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        net_ret = (eq[-1] - initial_equity) / initial_equity * 100
    else:
        max_dd = 0.0
        net_ret = 0.0

    bear_alpha = sum(t.pnl_dollars for t in bear_trades) / initial_equity * 100 if bear_trades else 0

    # Exit efficiency: sum(r_multiple) / sum(mfe_r) for positive MFE trades
    total_captured = sum(t.r_multiple for t in all_trades)
    total_available = sum(t.mfe_r for t in all_trades if t.mfe_r > 0)
    exit_eff = total_captured / total_available if total_available > 0 else 0.0

    # Sharpe ratio (annualized from hourly equity returns)
    sharpe = 0.0
    if len(eq) > 2:
        hourly_returns = np.diff(eq) / eq[:-1]
        hourly_returns = hourly_returns[~np.isnan(hourly_returns)]
        if len(hourly_returns) > 1:
            mu = float(np.mean(hourly_returns))
            sigma = float(np.std(hourly_returns))
            if sigma > 0:
                # Hourly bars → ~7 bars/day RTH, ~252 trading days/year
                bars_per_day = 7.0
                sharpe = mu / sigma * math.sqrt(252.0 * bars_per_day)

    # Bear capture ratio: bear_period_return / abs(benchmark_bear_return)
    # Simplified: fraction of total return earned during bear regimes
    total_pnl = sum(t.pnl_dollars for t in all_trades)
    bear_pnl = sum(t.pnl_dollars for t in bear_trades)
    bear_capture = bear_pnl / abs(total_pnl) if total_pnl != 0 else 0.0

    # Regime F1: approximate using bear trade accuracy as proxy
    # True F1 requires labeled crisis windows; use bear trade performance as signal quality
    regime_f1 = 0.0
    if bear_trades:
        # Precision: fraction of bear-labeled trades that were profitable
        precision = len(bear_wins) / len(bear_trades) if bear_trades else 0
        # Recall proxy: fraction of total profitable trades that came from bear regimes
        all_wins = [t for t in all_trades if t.r_multiple > 0]
        recall = len(bear_wins) / len(all_wins) if all_wins else 0
        if precision + recall > 0:
            regime_f1 = 2 * precision * recall / (precision + recall)

    # Detection latency: avg days from crisis start to first trade entry
    # Crisis coverage: fraction of crisis windows with ≥1 trade
    detection_latency = 999.0
    bias_latency = 999.0
    crisis_coverage = 0.0
    try:
        from backtests.swing.analysis.brs_diagnostics import CRISIS_WINDOWS
        latency_days = []
        windows_with_trades = 0
        total_windows = len(CRISIS_WINDOWS)
        for _name, crisis_start, crisis_end in CRISIS_WINDOWS:
            crisis_trades = [
                t for t in all_trades
                if getattr(t, "entry_time", None) is not None
                and crisis_start <= t.entry_time.replace(tzinfo=None) <= crisis_end
            ]
            if crisis_trades:
                windows_with_trades += 1
                first_entry = min(t.entry_time.replace(tzinfo=None) for t in crisis_trades)
                delta = (first_entry - crisis_start).total_seconds() / 86400
                latency_days.append(delta)
        if latency_days:
            detection_latency = sum(latency_days) / len(latency_days)
        if total_windows > 0:
            crisis_coverage = windows_with_trades / total_windows

        # Bias confirmation latency: crisis_start → first SHORT bias confirmed
        bias_latency_days_list = []
        all_crisis_logs = []
        for sr in result.symbol_results.values():
            all_crisis_logs.extend(getattr(sr, "crisis_state_log", []))
        for _name, crisis_start, crisis_end in CRISIS_WINDOWS:
            crisis_entries = [e for e in all_crisis_logs if e["crisis"] == _name]
            short_entries = [
                e for e in crisis_entries if e["bias_confirmed"] == "SHORT"
            ]
            if short_entries:
                first_date = min(
                    datetime.strptime(e["date"], "%Y-%m-%d") for e in short_entries
                )
                delta = (first_date - crisis_start).total_seconds() / 86400
                bias_latency_days_list.append(delta)
        if bias_latency_days_list:
            bias_latency = sum(bias_latency_days_list) / len(bias_latency_days_list)
    except Exception:
        pass

    return BRSMetrics(
        total_trades=len(all_trades),
        bear_trades=len(bear_trades),
        bear_trade_wr=len(bear_wins) / len(bear_trades) * 100 if bear_trades else 0,
        profit_factor=pf,
        max_dd_pct=max_dd,
        bear_alpha_pct=bear_alpha,
        regime_f1=regime_f1,
        exit_efficiency=max(0.0, min(1.0, exit_eff)),
        calmar=abs(net_ret / 100) / max_dd if max_dd > 0 else 0,
        sharpe=sharpe,
        net_return_pct=net_ret,
        bear_pf=bear_pf,
        detection_latency_days=detection_latency,
        bias_latency_days=bias_latency,
        bear_capture_ratio=max(0.0, min(1.0, bear_capture)),
        crisis_coverage=crisis_coverage,
    )
