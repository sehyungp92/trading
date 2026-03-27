"""Composite scoring for automated backtesting.

Weights (Phase 6 — replaced frequency with win rate):
  - Net profit (30%): absolute profitability, norm: log(1+R)/log(base) where R=return ratio
  - Profit factor (20%): win quality, norm: log(PF) / log(base)
  - Edge t-stat (15%): statistical significance of R-multiple edge,
    norm: clip((avg_R / std_R) * sqrt(N) / ceiling, 0, 1). Captures both edge quality
    AND frequency — more trades at the same edge improves sqrt(N).
  - Win rate (15%): consistency, norm: win_rate / ceiling
  - Calmar ratio (10%): risk-adjusted return, norm: calmar / ceiling
  - Inverse drawdown (10%): low DD reward, norm: 1.0 - max_dd / ceiling

Normalization ceilings are strategy-specific via ScoreNormalization.
Default ceilings (ALCB): PF 4.0, Calmar 50, DD 30%, return 500%, WR 70%, t-stat 8.
IARIC ceilings: PF 8.0, Calmar 100, DD 5% — rescaled for extreme capital efficiency.

Hard rejects: <25 trades, >35% max DD, PF < 0.8
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from research.backtests.stock.analysis.metrics import PerformanceMetrics, compute_metrics
from research.backtests.stock.models import TradeRecord


# ---------------------------------------------------------------------------
# Normalization ceilings (strategy-specific)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreNormalization:
    """Strategy-specific normalization ceilings for composite scoring.

    Each field defines the metric value that maps to raw=1.0 (perfect).
    Strategies with extreme metrics need wider ceilings to prevent saturation.
    """
    calmar_ceiling: float = 50.0      # Calmar ratio for raw=1.0
    pf_log_base: float = 4.0         # PF for raw=1.0 (log scale)
    dd_ceiling: float = 0.30         # Max DD pct where raw=0.0
    return_log_base: float = 6.0     # (1 + return_ratio) for raw=1.0 (i.e. 500% return)
    wr_ceiling: float = 0.70         # Win rate for raw=1.0
    tstat_ceiling: float = 8.0       # Edge t-stat for raw=1.0


# Default normalization (ALCB and other strategies)
DEFAULT_NORM = ScoreNormalization()

# IARIC normalization — rescaled for extreme capital efficiency
# (Calmar ~50, PF ~3.8, DD ~1.1% saturate the default ceilings)
IARIC_NORM = ScoreNormalization(
    calmar_ceiling=100.0,     # Calmar 50 → raw 0.50 (was 0.998)
    pf_log_base=8.0,          # PF 3.84 → raw 0.65 (was 0.971)
    dd_ceiling=0.05,          # DD 1.13% → raw 0.77 (was 0.962)
)


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositeScore:
    calmar_component: float       # 0.10 weight
    pf_component: float           # 0.20 weight
    inv_dd_component: float       # 0.10 weight
    net_profit_component: float   # 0.30 weight
    wr_component: float = 0.0     # 0.15 weight
    edge_tstat_component: float = 0.0  # 0.15 weight
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


# Weights — Phase 6
_W_CALMAR = 0.10
_W_PF = 0.20
_W_INV_DD = 0.10
_W_NET_PROFIT = 0.30
_W_WR = 0.15
_W_EDGE = 0.15


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(
    metrics: PerformanceMetrics,
    initial_equity: float = 10_000.0,
    r_multiples: np.ndarray | None = None,
    norm: ScoreNormalization | None = None,
) -> CompositeScore:
    """Compute the composite score.

    Args:
        metrics: Standard performance metrics from engine run.
        initial_equity: Starting capital.
        r_multiples: Per-trade R-multiples (pnl / risk). If None, edge t-stat
            component defaults to 0.0 (backwards-compatible).
        norm: Strategy-specific normalization ceilings. None = DEFAULT_NORM.
    """
    if norm is None:
        norm = DEFAULT_NORM

    # Hard rejects
    if metrics.total_trades < 25:
        return CompositeScore(0, 0, 0, 0, rejected=True,
                              reject_reason=f"Too few trades: {metrics.total_trades} < 25")
    if metrics.max_drawdown_pct > 0.35:
        return CompositeScore(0, 0, 0, 0, rejected=True,
                              reject_reason=f"Max DD too high: {metrics.max_drawdown_pct:.1%} > 35%")
    if metrics.profit_factor < 0.8:
        return CompositeScore(0, 0, 0, 0, rejected=True,
                              reject_reason=f"PF too low: {metrics.profit_factor:.2f} < 0.80")

    # Component scores (each clipped to [0, 1])
    calmar_raw = _clip01(metrics.calmar / norm.calmar_ceiling)
    pf_raw = _clip01(math.log(metrics.profit_factor) / math.log(norm.pf_log_base)) if metrics.profit_factor > 1.0 else 0.0
    inv_dd_raw = _clip01(1.0 - metrics.max_drawdown_pct / norm.dd_ceiling)
    return_ratio = metrics.net_profit / initial_equity
    np_raw = _clip01(math.log(1.0 + return_ratio) / math.log(norm.return_log_base))
    wr_raw = _clip01(metrics.win_rate / norm.wr_ceiling)

    # Edge t-statistic: (avg_R / std_R) * sqrt(N)
    edge_raw = 0.0
    if r_multiples is not None and len(r_multiples) > 1:
        std_r = float(np.std(r_multiples, ddof=1))
        if std_r > 0:
            avg_r = float(np.mean(r_multiples))
            t_stat = (avg_r / std_r) * math.sqrt(len(r_multiples))
            edge_raw = _clip01(t_stat / norm.tstat_ceiling)

    total = (
        _W_CALMAR * calmar_raw
        + _W_PF * pf_raw
        + _W_INV_DD * inv_dd_raw
        + _W_NET_PROFIT * np_raw
        + _W_WR * wr_raw
        + _W_EDGE * edge_raw
    )

    return CompositeScore(
        calmar_component=calmar_raw,
        pf_component=pf_raw,
        inv_dd_component=inv_dd_raw,
        net_profit_component=np_raw,
        wr_component=wr_raw,
        edge_tstat_component=edge_raw,
        total=total,
    )


# ---------------------------------------------------------------------------
# Metrics extraction helper
# ---------------------------------------------------------------------------

def extract_metrics(
    trades: list[TradeRecord],
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
) -> PerformanceMetrics:
    """Standard metrics extraction from engine result. Mirrors runner.py pattern."""
    if not trades:
        return PerformanceMetrics()

    pnls = np.array([t.pnl_net for t in trades])
    risks = np.array([t.risk_per_share * t.quantity for t in trades])
    hold_hours = np.array([t.hold_hours for t in trades])
    commissions = np.array([t.commission for t in trades])
    symbols = [t.symbol for t in trades]

    return compute_metrics(
        pnls, risks, hold_hours, commissions,
        equity_curve, timestamps, initial_equity,
        trade_symbols=symbols,
    )


def compute_r_multiples(trades: list[TradeRecord]) -> np.ndarray:
    """Compute per-trade R-multiples (pnl / risk)."""
    if not trades:
        return np.array([])
    pnls = np.array([t.pnl_net for t in trades])
    risks = np.array([t.risk_per_share * t.quantity for t in trades])
    return pnls / np.where(risks == 0, 1.0, risks)
