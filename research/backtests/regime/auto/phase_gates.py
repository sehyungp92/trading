"""Phase gate criteria for multi-phase regime optimization.

Each phase has success gates that must pass before advancing.
Gate failure is categorized to guide the decision loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.backtests.regime.analysis.metrics import PortfolioMetrics


@dataclass(frozen=True)
class GateCriterion:
    name: str
    target: float
    actual: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    passed: bool
    criteria: list[GateCriterion]
    failure_category: str | None = None  # "scoring_ineffective" | "candidates_exhausted" | "structural_issue"
    recommendations: list[str] = field(default_factory=list)


def check_phase_gate(
    phase: int,
    metrics: PortfolioMetrics,
    regime_stats: dict,
    greedy_result: dict | None = None,
) -> GateResult:
    """Check whether a phase's success gate is met.

    Args:
        phase: Phase number (1-4).
        metrics: Portfolio metrics from the final optimized run.
        regime_stats: Regime statistics from compute_regime_stats().
        greedy_result: Optional greedy result dict for failure categorization.

    Returns:
        GateResult with pass/fail, criteria details, and recommendations.
    """
    if phase == 1:
        return _gate_phase_1(metrics, regime_stats, greedy_result)
    elif phase == 2:
        return _gate_phase_2(metrics, regime_stats, greedy_result)
    elif phase == 3:
        return _gate_phase_3(metrics, regime_stats, greedy_result)
    elif phase == 4:
        return _gate_phase_4(metrics, regime_stats, greedy_result)
    else:
        raise ValueError(f"Unknown phase: {phase}")


def _gate_phase_1(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 1 gate: >=3 regimes >5%, transition rate >0.008, Sharpe >0.4."""
    # Count regimes with >5% share
    dist = rs.get("dominant_dist", {})
    regimes_above_5 = sum(1 for v in dist.values() if v > 0.05)

    criteria = [
        GateCriterion("regimes_above_5pct", 3.0, float(regimes_above_5),
                       regimes_above_5 >= 3),
        GateCriterion("transition_rate", 0.008, rs.get("transition_rate", 0.0),
                       rs.get("transition_rate", 0.0) > 0.008),
        GateCriterion("sharpe", 0.4, metrics.sharpe, metrics.sharpe > 0.4),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_1_recommendations(criteria, rs),
    )


def _gate_phase_2(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 2 gate: all 4 regimes >3%, Sharpe >0.6."""
    dist = rs.get("dominant_dist", {})
    regimes_above_3 = sum(1 for v in dist.values() if v > 0.03)

    criteria = [
        GateCriterion("all_4_regimes_above_3pct", 4.0, float(regimes_above_3),
                       regimes_above_3 >= 4),
        GateCriterion("sharpe", 0.6, metrics.sharpe, metrics.sharpe > 0.6),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_2_recommendations(criteria, rs),
    )


def _gate_phase_3(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 3 gate: Sharpe >0.8, crisis response >0.4, max DD <20%."""
    criteria = [
        GateCriterion("sharpe", 0.8, metrics.sharpe, metrics.sharpe > 0.8),
        GateCriterion("crisis_response", 0.4, rs.get("crisis_response", 0.0),
                       rs.get("crisis_response", 0.0) > 0.4),
        GateCriterion("max_dd", 0.20, metrics.max_drawdown_pct,
                       metrics.max_drawdown_pct < 0.20),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_3_recommendations(criteria, rs),
    )


def _gate_phase_4(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 4 gate: Sharpe >0.9, historical alignment >0.5."""
    hist_align = rs.get("historical_alignment", 0.0)

    criteria = [
        GateCriterion("sharpe", 0.9, metrics.sharpe, metrics.sharpe > 0.9),
        GateCriterion("historical_alignment", 0.5, hist_align, hist_align > 0.5),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_4_recommendations(criteria, rs),
    )


# ---------------------------------------------------------------------------
# Failure categorization
# ---------------------------------------------------------------------------

def _categorize_failure(
    criteria: list[GateCriterion],
    rs: dict,
    gr: dict | None,
) -> str:
    """Categorize a gate failure for the decision loop."""
    # Structural issue: degenerate HMM
    if rs.get("n_active_regimes", 0) < 2 or rs.get("transition_rate", 0.0) < 0.001:
        return "structural_issue"

    # Check if greedy stopped early (candidates exhausted)
    if gr is not None:
        # Support both full result (rounds list) and summary (n_rounds int)
        n_rounds = gr.get("n_rounds", len(gr.get("rounds", [])))
        max_rounds = gr.get("max_rounds", 20)
        if n_rounds > 0 and n_rounds < max_rounds:
            # Greedy stopped due to no improvement — check how close we are
            close_count = sum(
                1 for c in criteria
                if not c.passed and c.actual >= 0.8 * c.target
            )
            if close_count > 0:
                return "candidates_exhausted"

    return "scoring_ineffective"


# ---------------------------------------------------------------------------
# Per-phase recommendations
# ---------------------------------------------------------------------------

def _phase_1_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "regimes_above_5pct":
            if rs.get("n_active_regimes", 0) < 2:
                recs.append("Only 2 regimes active — sticky prior may need to go even lower (try 2-3)")
            else:
                recs.append("3 regimes exist but not all >5% — try narrower sticky_diag around best value")
        elif c.name == "transition_rate":
            recs.append("Low transition rate — try more aggressive rolling window (5yr) or lower refit_ll_tolerance")
        elif c.name == "sharpe":
            recs.append("Sharpe below target — may be acceptable; focus on regime health first")
    return recs


def _phase_2_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "all_4_regimes_above_3pct":
            dist = rs.get("dominant_dist", {})
            weak = [k for k, v in dist.items() if v <= 0.03]
            recs.append(f"Weak regimes: {weak} — try feature combinations that distinguish these quadrants")
        elif c.name == "sharpe":
            recs.append("Sharpe below target — features may be adding noise; try dropping collinear features")
    return recs


def _phase_3_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "crisis_response":
            recs.append("Low crisis response — investigate ventilator settings or crisis overlay weights")
        elif c.name == "sharpe":
            recs.append("Sharpe below target — try reducing leverage or tightening risk budget")
        elif c.name == "max_dd":
            recs.append("Max DD too high — reduce L_max or increase sigma_floor")
    return recs


def _phase_4_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "historical_alignment":
            recs.append("Low historical alignment — features may be wrong, consider going back to Phase 2")
        elif c.name == "sharpe":
            recs.append("Sharpe below target — fine-tuning may not be enough; revisit Phase 3 parameters")
    return recs
