"""Downturn R2 composite scoring -- 7 orthogonal components for correction-capture specialist.

Each component maps to a structural dimension:
  correction_pnl   (25%): PnL from correction windows as % of equity
  edge             (20%): profit factor -- consistency of wins vs losses
  risk             (18%): inverse max drawdown with 30% ceiling
  capture          (15%): geomean of MFE capture ratio and exit efficiency
  regime_purity    (10%): how much non-correction trades dilute alpha
  calmar           (07%): return / max drawdown ratio
  frequency        (05%): total trade count (prevents degenerate configs)

Hard rejects: <10 trades, negative correction-window PnL, DD > 35%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics


@dataclass(frozen=True)
class DownturnCompositeScore:
    """Frozen composite score."""
    correction_pnl: float = 0.0
    edge: float = 0.0
    risk: float = 0.0
    capture: float = 0.0
    regime_purity: float = 0.0
    calmar: float = 0.0
    frequency: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""
    # Legacy aliases for phase_scoring/gate compatibility
    exit_quality: float = 0.0
    correction_coverage: float = 0.0
    profit_factor: float = 0.0


BASE_WEIGHTS = {
    "correction_pnl": 0.25,
    "edge": 0.20,
    "risk": 0.18,
    "capture": 0.15,
    "regime_purity": 0.10,
    "calmar": 0.07,
    "frequency": 0.05,
}


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(
    metrics: DownturnMetrics,
    weight_overrides: dict[str, float] | None = None,
) -> DownturnCompositeScore:
    """Compute 7-component composite score."""
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
    if metrics.max_dd_pct > 0.35:
        return DownturnCompositeScore(
            rejected=True,
            reject_reason=f"max_dd_exceeded ({metrics.max_dd_pct:.2%})",
        )

    # --- Components ---

    # Correction PnL: 1.0 at 80% of equity from correction windows
    correction_pnl_c = _clip01(metrics.correction_pnl_pct / 80.0)

    # Edge: (PF - 1) / 2.0, 1.0 at PF=3.0
    edge_c = _clip01((metrics.profit_factor - 1.0) / 2.0)

    # Risk: inverse drawdown with 30% ceiling
    risk_c = _clip01(1 - metrics.max_dd_pct / 0.30)

    # Capture: geomean of MFE capture and exit efficiency (both 0-1)
    capture_c = _clip01(math.sqrt(max(metrics.avg_mfe_capture * metrics.exit_efficiency, 0.0)))

    # Regime purity: 1.0 when non-correction PnL >= 0; drops as non-corr losses approach correction gains
    corr_pnl_pct = max(metrics.correction_pnl_pct, 1.0)
    regime_purity_c = _clip01(1 + (metrics.net_return_pct - metrics.correction_pnl_pct) / corr_pnl_pct)

    # Calmar: 1.0 at calmar=3.0
    calmar_c = _clip01(metrics.calmar / 3.0)

    # Frequency: 1.0 at 120 trades
    frequency_c = _clip01(metrics.total_trades / 120.0)

    total = (
        w.get("correction_pnl", 0) * correction_pnl_c
        + w.get("edge", 0) * edge_c
        + w.get("risk", 0) * risk_c
        + w.get("capture", 0) * capture_c
        + w.get("regime_purity", 0) * regime_purity_c
        + w.get("calmar", 0) * calmar_c
        + w.get("frequency", 0) * frequency_c
    )

    return DownturnCompositeScore(
        correction_pnl=correction_pnl_c,
        edge=edge_c,
        risk=risk_c,
        capture=capture_c,
        regime_purity=regime_purity_c,
        calmar=calmar_c,
        frequency=frequency_c,
        total=total,
        # Legacy aliases
        exit_quality=capture_c,
        correction_coverage=correction_pnl_c,
        profit_factor=edge_c,
    )
