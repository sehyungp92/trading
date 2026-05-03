"""NQDTC round-2 phase gate criteria.

The score ranks candidates, while these gates keep accepted mutations from
buying frequency with low-quality trades or exit deterioration.
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion, GateResult

from .scoring import NQDTCMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: NQDTCMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*.

    Later rounds start from the previous optimized config, so gates preserve
    the proven edge while probing robust alpha and throughput.
    """
    criteria: list[GateCriterion] = []

    if phase == 1:
        # Session-frequency harvest: extra trades must also improve return.
        criteria.extend([
            GateCriterion("hard_min_trades", 89.0, float(metrics.total_trades), metrics.total_trades >= 89),
            GateCriterion("hard_max_dd_pct", 0.18, metrics.max_dd_pct, metrics.max_dd_pct <= 0.18),
            GateCriterion("hard_min_pf", 2.00, metrics.profit_factor, metrics.profit_factor >= 2.00),
            GateCriterion("total_trades", 94.0, float(metrics.total_trades), metrics.total_trades >= 94),
            GateCriterion("net_return_pct", 300.0, metrics.net_return_pct, metrics.net_return_pct >= 300.0),
            GateCriterion("robust_net_return_pct", 220.0, metrics.robust_net_return_pct, metrics.robust_net_return_pct >= 220.0),
            GateCriterion("largest_win_pnl_share_max", 0.30, metrics.largest_win_pnl_share, metrics.largest_win_pnl_share <= 0.30),
        ])
    elif phase == 2:
        # Robust return protection: prevent one large trade from masking decay.
        criteria.extend([
            GateCriterion("hard_min_trades", 89.0, float(metrics.total_trades), metrics.total_trades >= 89),
            GateCriterion("hard_max_dd_pct", 0.18, metrics.max_dd_pct, metrics.max_dd_pct <= 0.18),
            GateCriterion("hard_min_pf", 2.00, metrics.profit_factor, metrics.profit_factor >= 2.00),
            GateCriterion("avg_r", 0.48, metrics.avg_r, metrics.avg_r >= 0.48),
            GateCriterion("capture_ratio", 0.40, metrics.capture_ratio, metrics.capture_ratio >= 0.40),
            GateCriterion("robust_net_return_pct", 220.0, metrics.robust_net_return_pct, metrics.robust_net_return_pct >= 220.0),
            GateCriterion("largest_win_pnl_share_max", 0.30, metrics.largest_win_pnl_share, metrics.largest_win_pnl_share <= 0.30),
        ])
    elif phase == 3:
        # Selective alpha recovery: existing controls only, no broad quality decay.
        criteria.extend([
            GateCriterion("hard_min_trades", 89.0, float(metrics.total_trades), metrics.total_trades >= 89),
            GateCriterion("hard_max_dd_pct", 0.20, metrics.max_dd_pct, metrics.max_dd_pct <= 0.20),
            GateCriterion("hard_min_pf", 1.90, metrics.profit_factor, metrics.profit_factor >= 1.90),
            GateCriterion("net_return_pct", 296.0, metrics.net_return_pct, metrics.net_return_pct >= 296.0),
            GateCriterion("robust_net_return_pct", 210.0, metrics.robust_net_return_pct, metrics.robust_net_return_pct >= 210.0),
            GateCriterion("avg_r", 0.42, metrics.avg_r, metrics.avg_r >= 0.42),
            GateCriterion("capture_ratio", 0.38, metrics.capture_ratio, metrics.capture_ratio >= 0.38),
        ])
    elif phase == 4:
        # Fine-tune: no material regression from Phase 3 while allowing small
        # tradeoffs for higher expected return/frequency.
        criteria.extend([
            GateCriterion("hard_min_trades", 89.0, float(metrics.total_trades), metrics.total_trades >= 89),
            GateCriterion("hard_max_dd_pct", 0.20, metrics.max_dd_pct, metrics.max_dd_pct <= 0.20),
            GateCriterion("hard_min_pf", 1.90, metrics.profit_factor, metrics.profit_factor >= 1.90),
            GateCriterion("hard_min_capture", 0.38, metrics.capture_ratio, metrics.capture_ratio >= 0.38),
            GateCriterion("hard_min_robust_return", 210.0, metrics.robust_net_return_pct, metrics.robust_net_return_pct >= 210.0),
        ])
        if prior_phase_metrics:
            for key in ["profit_factor", "net_return_pct", "robust_net_return_pct", "total_trades", "avg_r", "capture_ratio"]:
                target = float(prior_phase_metrics.get(key, 0.0)) * 0.90
                actual = float(getattr(metrics, key, 0.0))
                criteria.append(GateCriterion(f"no_regress_{key}", target, actual, actual >= target))
            prior_dd = float(prior_phase_metrics.get("max_dd_pct", metrics.max_dd_pct))
            dd_target = min(0.27, prior_dd * 1.15)
            criteria.append(GateCriterion("no_regress_max_dd_pct", dd_target, metrics.max_dd_pct, metrics.max_dd_pct <= dd_target))
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
        if metrics.total_trades < 89:
            recs.append("Relax signal gates to increase trade count")
        if metrics.max_dd_pct > 0.20:
            recs.append("Tighten stop-width, ATR-stop, or exit protection candidates")
        if metrics.profit_factor < 1.90:
            recs.append("Reject low-quality cohorts before adding frequency")
    elif category == "scoring_ineffective":
        recs.append("Keep score immutable; expand candidates that target the failed gate")
    elif category == "candidates_exhausted":
        recs.append("All candidates rejected -- expand experiment pool or relax hard rejects")
    elif category == "diagnostic_needed":
        for c in criteria:
            if not c.passed:
                recs.append(f"Near-miss on {c.name}: {c.actual:.3f} vs target {c.target:.3f}")
    return recs
