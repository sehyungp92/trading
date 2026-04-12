"""ATRSS phase gate criteria -- progressive thresholds per phase."""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import ATRSSMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: ATRSSMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*.

    Phase 4 adds no-regression checks against Phase 3 results.
    """
    criteria: list[GateCriterion] = [
        GateCriterion("hard_min_trades", 100.0, float(metrics.total_trades), metrics.total_trades >= 100),
        GateCriterion("hard_max_dd_pct", 0.07, metrics.max_dd_pct, metrics.max_dd_pct <= 0.07),
        GateCriterion("hard_min_pf", 2.0, metrics.profit_factor, metrics.profit_factor >= 2.0),
        GateCriterion("hard_min_wr", 0.55, metrics.win_rate, metrics.win_rate >= 0.55),
    ]

    if phase == 1:
        criteria.extend([
            GateCriterion("profit_factor", 4.0, metrics.profit_factor, metrics.profit_factor >= 4.0),
            GateCriterion("total_r", 130.0, metrics.total_r, metrics.total_r >= 130.0),
            GateCriterion("mfe_capture", 0.35, metrics.mfe_capture, metrics.mfe_capture >= 0.35),
        ])
    elif phase == 2:
        criteria.extend([
            GateCriterion("total_trades", 150.0, float(metrics.total_trades), metrics.total_trades >= 150),
            GateCriterion("win_rate", 0.65, metrics.win_rate, metrics.win_rate >= 0.65),
            GateCriterion("profit_factor", 4.0, metrics.profit_factor, metrics.profit_factor >= 4.0),
        ])
    elif phase == 3:
        criteria.extend([
            GateCriterion("trades_per_month", 4.0, metrics.trades_per_month, metrics.trades_per_month >= 4.0),
            GateCriterion("total_trades", 160.0, float(metrics.total_trades), metrics.total_trades >= 160),
            GateCriterion("profit_factor", 4.5, metrics.profit_factor, metrics.profit_factor >= 4.5),
        ])
    elif phase == 4:
        criteria.extend([
            GateCriterion("calmar_r", 30.0, metrics.calmar_r, metrics.calmar_r >= 30.0),
            GateCriterion("total_r", 150.0, metrics.total_r, metrics.total_r >= 150.0),
            GateCriterion("sharpe", 3.0, metrics.sharpe, metrics.sharpe >= 3.0),
        ])
        # No-regression: no metric >5% below Phase 3
        if prior_phase_metrics:
            for key in ("profit_factor", "total_r", "win_rate", "calmar_r"):
                prior_val = prior_phase_metrics.get(key, 0.0)
                if prior_val > 0:
                    cur_val = getattr(metrics, key, 0.0)
                    floor = prior_val * 0.95
                    criteria.append(
                        GateCriterion(
                            f"no_regress_{key}",
                            floor,
                            cur_val,
                            cur_val >= floor,
                        )
                    )

    return criteria
