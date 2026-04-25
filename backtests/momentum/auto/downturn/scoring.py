"""Downturn composite scoring -- 6 orthogonal components, no redundancies.

Each component maps to a structural weakness:
  coverage   (25%): fraction of correction windows traded
  capture    (20%): geomean of MFE capture ratio and exit efficiency
  net_profit (15%): overall profitability (log-scaled)
  edge       (15%): profit factor -- consistency of wins vs losses
  risk       (15%): inverse max drawdown -- don't blow up
  hold       (10%): median hold duration -- penalizes scalping

Hard rejects: <10 trades, negative correction-window PnL, DD > 40%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics


@dataclass(frozen=True)
class DownturnCompositeScore:
    """Frozen composite score."""
    coverage: float = 0.0
    capture: float = 0.0
    net_profit: float = 0.0
    edge: float = 0.0
    risk: float = 0.0
    hold: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""
    # Legacy aliases for phase_scoring/gate compatibility
    exit_quality: float = 0.0
    correction_coverage: float = 0.0
    profit_factor: float = 0.0


BASE_WEIGHTS = {
    "coverage": 0.25,
    "capture": 0.20,
    "net_profit": 0.15,
    "edge": 0.15,
    "risk": 0.15,
    "hold": 0.10,
}


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(
    metrics: DownturnMetrics,
    weight_overrides: dict[str, float] | None = None,
) -> DownturnCompositeScore:
    """Compute 6-component composite score."""
    w = dict(BASE_WEIGHTS)
    if weight_overrides:
        w.update(weight_overrides)

    # Hard rejects
    if metrics.total_trades < 10:
        return DownturnCompositeScore(
            rejected=True, reject_reason=f"too_few_trades ({metrics.total_trades})",
        )
    if metrics.correction_pnl_pct < 0:
        return DownturnCompositeScore(
            rejected=True,
            reject_reason=f"negative_correction_pnl ({metrics.correction_pnl_pct:.2f}%)",
        )
    if metrics.max_dd_pct > 0.40:
        return DownturnCompositeScore(
            rejected=True,
            reject_reason=f"max_dd_exceeded ({metrics.max_dd_pct:.2%})",
        )

    # --- Components ---

    # Coverage: raw 0-1 fraction of correction windows with >= 1 trade. R5 = 0.35
    coverage_c = _clip01(metrics.correction_coverage)

    # Capture: geomean of MFE capture and exit efficiency (both 0-1)
    capture_c = _clip01(math.sqrt(metrics.avg_mfe_capture * metrics.exit_efficiency))

    # Net profit: log scale, 1.0 at ~1000% return. R5 = 726% -> 0.88
    net_profit_c = _clip01(math.log(1 + max(metrics.net_return_pct, 0) / 100) / math.log(11))

    # Edge: (PF - 1) / 5, 1.0 at PF=6. R5 PF=4.05 -> 0.61
    edge_c = _clip01((metrics.profit_factor - 1) / 5.0)

    # Risk: inverse drawdown with 40% ceiling. R5 DD=20.4% -> 0.49
    risk_c = _clip01(1 - metrics.max_dd_pct / 0.40)

    # Hold: median hold in 5m bars / 48 (4 hours). R5 = 19 bars -> 0.40
    hold_c = _clip01(getattr(metrics, "median_hold_5m", 0.0) / 48.0)

    total = (
        w["coverage"] * coverage_c
        + w["capture"] * capture_c
        + w["net_profit"] * net_profit_c
        + w["edge"] * edge_c
        + w["risk"] * risk_c
        + w["hold"] * hold_c
    )

    return DownturnCompositeScore(
        coverage=coverage_c,
        capture=capture_c,
        net_profit=net_profit_c,
        edge=edge_c,
        risk=risk_c,
        hold=hold_c,
        total=total,
        # Legacy aliases
        exit_quality=capture_c,
        correction_coverage=coverage_c,
        profit_factor=edge_c,
    )
