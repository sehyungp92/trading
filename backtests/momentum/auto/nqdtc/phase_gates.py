"""NQDTC post-audit phase gate criteria -- recalibrated for recovery from breakeven."""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion, GateResult

from .scoring import NQDTCMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: NQDTCMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*.

    Post-audit recalibration: targets scaled for recovery from PF=0.98 baseline.
    Phase 1 (Regime): just need improvement over breakeven
    Phase 2 (Signal): entry quality improvement
    Phase 3 (Timing/Exit): full system tightening
    Phase 4 (Fine-tune): no-regression of Phase 3
    """
    criteria: list[GateCriterion] = []

    if phase == 1:
        # Regime filtering: just need improvement over breakeven
        criteria.extend([
            GateCriterion("hard_min_trades", 10.0, float(metrics.total_trades), metrics.total_trades >= 10),
            GateCriterion("hard_max_dd_pct", 0.45, metrics.max_dd_pct, metrics.max_dd_pct <= 0.45),
            GateCriterion("hard_min_pf", 0.70, metrics.profit_factor, metrics.profit_factor >= 0.70),
            GateCriterion("profit_factor", 1.2, metrics.profit_factor, metrics.profit_factor >= 1.2),
            GateCriterion("max_dd_pct", 0.30, metrics.max_dd_pct, metrics.max_dd_pct <= 0.30),
            GateCriterion("net_return_pct", 10.0, metrics.net_return_pct, metrics.net_return_pct >= 10.0),
        ])
    elif phase == 2:
        # Signal quality: entry quality improvement
        criteria.extend([
            GateCriterion("hard_min_trades", 15.0, float(metrics.total_trades), metrics.total_trades >= 15),
            GateCriterion("hard_max_dd_pct", 0.35, metrics.max_dd_pct, metrics.max_dd_pct <= 0.35),
            GateCriterion("hard_min_pf", 0.90, metrics.profit_factor, metrics.profit_factor >= 0.90),
            GateCriterion("profit_factor", 1.4, metrics.profit_factor, metrics.profit_factor >= 1.4),
            GateCriterion("win_rate", 0.50, metrics.win_rate, metrics.win_rate >= 0.50),
            GateCriterion("net_return_pct", 20.0, metrics.net_return_pct, metrics.net_return_pct >= 20.0),
        ])
    elif phase == 3:
        # Timing & exit: full system tightening
        criteria.extend([
            GateCriterion("hard_min_trades", 15.0, float(metrics.total_trades), metrics.total_trades >= 15),
            GateCriterion("hard_max_dd_pct", 0.30, metrics.max_dd_pct, metrics.max_dd_pct <= 0.30),
            GateCriterion("hard_min_pf", 1.00, metrics.profit_factor, metrics.profit_factor >= 1.00),
            GateCriterion("profit_factor", 1.6, metrics.profit_factor, metrics.profit_factor >= 1.6),
            GateCriterion("max_dd_pct", 0.20, metrics.max_dd_pct, metrics.max_dd_pct <= 0.20),
            GateCriterion("sortino", 1.5, metrics.sortino, metrics.sortino >= 1.5),
            GateCriterion("net_return_pct", 30.0, metrics.net_return_pct, metrics.net_return_pct >= 30.0),
        ])
    elif phase == 4:
        # Fine-tune: no-regression of Phase 3 at 90%
        criteria.extend([
            GateCriterion("hard_min_trades", 15.0, float(metrics.total_trades), metrics.total_trades >= 15),
            GateCriterion("hard_max_dd_pct", 0.25, metrics.max_dd_pct, metrics.max_dd_pct <= 0.25),
            GateCriterion("hard_min_pf", 1.00, metrics.profit_factor, metrics.profit_factor >= 1.00),
        ])
        if prior_phase_metrics:
            for key in ["calmar", "profit_factor", "sharpe", "sortino", "net_return_pct", "total_trades"]:
                target = float(prior_phase_metrics.get(key, 0.0)) * 0.90
                actual = float(getattr(metrics, key, 0.0))
                criteria.append(GateCriterion(f"no_regress_{key}", target, actual, actual >= target))
        else:
            criteria.append(GateCriterion("phase4_pass", 0.0, 1.0, True))

    return criteria


def check_phase_gate(
    phase: int,
    metrics: NQDTCMetrics,
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


def _categorize_failure(metrics: NQDTCMetrics, criteria: list[GateCriterion], greedy_result: dict | None) -> str:
    hard_fails = [c for c in criteria if not c.passed and c.name.startswith("hard_")]
    if hard_fails:
        return "structural_issue"
    if greedy_result and greedy_result.get("total_candidates", 0) > 0 and greedy_result.get("accepted_count", 0) == 0:
        return "candidates_exhausted"
    near_miss = sum(1 for c in criteria if not c.passed and c.target > 0 and c.actual / c.target >= 0.85)
    if near_miss >= 2:
        return "diagnostic_needed"
    return "scoring_ineffective"


def _get_recommendations(metrics: NQDTCMetrics, criteria: list[GateCriterion], category: str) -> list[str]:
    recs: list[str] = []
    if category == "structural_issue":
        if metrics.total_trades < 10:
            recs.append("Relax signal gates to increase trade count")
        if metrics.max_dd_pct > 0.35:
            recs.append("Reduce position sizing or tighten stops")
        if metrics.profit_factor < 0.80:
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
