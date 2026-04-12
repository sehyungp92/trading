"""VdubusNQ phase gate criteria -- single source of truth for all gate logic."""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion, GateResult

from .scoring import VdubusMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: VdubusMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*.

    Phase 4 uses *prior_phase_metrics* (from phase 3) for 90% no-regression.
    """
    # Hard floors (all phases)
    criteria: list[GateCriterion] = [
        GateCriterion("hard_min_trades", 80.0, float(metrics.total_trades), metrics.total_trades >= 80),
        GateCriterion("hard_max_dd_pct", 0.22, metrics.max_dd_pct, metrics.max_dd_pct <= 0.22),
        GateCriterion("hard_min_pf", 1.5, metrics.profit_factor, metrics.profit_factor >= 1.5),
    ]

    if phase == 1:
        criteria.extend([
            GateCriterion("profit_factor", 2.5, metrics.profit_factor, metrics.profit_factor >= 2.5),
            GateCriterion("capture_ratio", 0.42, metrics.capture_ratio, metrics.capture_ratio >= 0.42),
            GateCriterion("net_return_pct", 1000.0, metrics.net_return_pct, metrics.net_return_pct >= 1000.0),
            GateCriterion("sharpe", 1.0, metrics.sharpe, metrics.sharpe >= 1.0),
        ])
    elif phase == 2:
        criteria.extend([
            GateCriterion("total_trades", 120.0, float(metrics.total_trades), metrics.total_trades >= 120),
            GateCriterion("win_rate", 0.48, metrics.win_rate, metrics.win_rate >= 0.48),
            GateCriterion("profit_factor", 2.5, metrics.profit_factor, metrics.profit_factor >= 2.5),
            GateCriterion("sharpe", 1.2, metrics.sharpe, metrics.sharpe >= 1.2),
        ])
    elif phase == 3:
        criteria.extend([
            GateCriterion("calmar", 8.0, metrics.calmar, metrics.calmar >= 8.0),
            GateCriterion("max_dd_pct", 0.18, metrics.max_dd_pct, metrics.max_dd_pct <= 0.18),
            GateCriterion("net_return_pct", 1000.0, metrics.net_return_pct, metrics.net_return_pct >= 1000.0),
            GateCriterion("capture_ratio", 0.45, metrics.capture_ratio, metrics.capture_ratio >= 0.45),
        ])
    elif phase == 4:
        # 90% no-regression floor on P3 metrics
        if prior_phase_metrics:
            for key in ["profit_factor", "calmar", "sharpe", "net_return_pct", "total_trades", "capture_ratio"]:
                target = float(prior_phase_metrics.get(key, 0.0)) * 0.90
                actual = float(getattr(metrics, key, 0.0))
                criteria.append(GateCriterion(f"no_regress_{key}", target, actual, actual >= target))
        else:
            criteria.append(GateCriterion("phase4_pass", 0.0, 1.0, True))

    return criteria


def check_phase_gate(
    phase: int,
    metrics: VdubusMetrics,
    greedy_result: dict | None = None,
    prior_phase_metrics: dict | None = None,
) -> GateResult:
    """Full gate check with failure categorization and recommendations."""
    criteria = gate_criteria_for_phase(phase, metrics, prior_phase_metrics)
    if all(c.passed for c in criteria):
        return GateResult(passed=True, criteria=tuple(criteria))

    category = _categorize_failure(metrics, criteria, greedy_result)
    recs = _get_recommendations(metrics, criteria, category)
    return GateResult(passed=False, criteria=tuple(criteria), failure_category=category, recommendations=tuple(recs))


def _categorize_failure(metrics: VdubusMetrics, criteria: list[GateCriterion], greedy_result: dict | None) -> str:
    if metrics.total_trades < 80 or metrics.max_dd_pct > 0.22 or metrics.profit_factor < 1.5:
        return "structural_issue"
    if greedy_result and greedy_result.get("total_candidates", 0) > 0 and greedy_result.get("accepted_count", 0) == 0:
        return "candidates_exhausted"
    near_miss = sum(1 for c in criteria if not c.passed and c.target > 0 and c.actual / c.target >= 0.85)
    if near_miss >= 2:
        return "diagnostic_needed"
    return "scoring_ineffective"


def _get_recommendations(metrics: VdubusMetrics, criteria: list[GateCriterion], category: str) -> list[str]:
    recs: list[str] = []
    if category == "structural_issue":
        if metrics.total_trades < 80:
            recs.append("Relax signal gates to increase trade count")
        if metrics.max_dd_pct > 0.22:
            recs.append("Reduce position sizing or tighten stops")
        if metrics.profit_factor < 1.5:
            recs.append("Fundamental edge issue -- review signal quality")
    elif category == "scoring_ineffective":
        recs.append("Adjust scoring weights to better reward gate-relevant metrics")
    elif category == "candidates_exhausted":
        recs.append("All candidates rejected -- expand experiment pool or relax hard rejects")
    elif category == "diagnostic_needed":
        for c in criteria:
            if not c.passed:
                recs.append(f"Near-miss on {c.name}: {c.actual:.3f} vs target {c.target:.3f}")
    return recs
