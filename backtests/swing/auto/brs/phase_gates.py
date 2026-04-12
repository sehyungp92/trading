"""BRS phase gate criteria — success thresholds + 4 failure categories.

Phase structure (Change #2):
  Phase 1: REGIME + ENTRY params + STRUCTURAL → PF >= 1.3, trades >= 20, return > 5%
  Phase 2: SIGNAL_SELECT → PF >= 1.3, trades >= 20, return > 5% (same as P1)
  Phase 3: EXIT + VOLATILITY → PF >= 1.5, return >= 10%, DD <= 22%
  Phase 4: SIZING + FINETUNE → calmar >= 1.0, DD <= 18%, return >= 15%
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backtests.swing.auto.brs.scoring import BRSMetrics


@dataclass(frozen=True)
class GateCriterion:
    """A single gate criterion with target and actual value."""
    name: str
    target: float
    actual: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    """Result of a phase gate check."""
    passed: bool
    criteria: tuple[GateCriterion, ...] = ()
    failure_category: str | None = None  # scoring_ineffective | candidates_exhausted | structural_issue | diagnostic_needed
    recommendations: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Per-phase gate thresholds
# ---------------------------------------------------------------------------

def check_phase_gate(
    phase: int,
    metrics: BRSMetrics,
    greedy_result: dict | None = None,
    prior_phase_metrics: dict | None = None,
) -> GateResult:
    """Check if a phase's gate criteria are met.

    Args:
        phase: Optimization phase (1-4)
        metrics: BRS metrics from backtest
        greedy_result: Optional greedy result dict for failure categorization
        prior_phase_metrics: Optional Phase 3 final_metrics dict (for Phase 4 regression)

    Returns:
        GateResult with pass/fail and recommendations
    """
    if phase == 1:
        return _gate_phase_1(metrics, greedy_result)
    elif phase == 2:
        return _gate_phase_2(metrics, greedy_result)
    elif phase == 3:
        return _gate_phase_3(metrics, greedy_result)
    elif phase == 4:
        return _gate_phase_4(metrics, greedy_result, prior_phase_metrics)
    return GateResult(passed=False, failure_category="structural_issue",
                      recommendations=("Unknown phase",))


def _gate_phase_1(metrics: BRSMetrics, gr: dict | None) -> GateResult:
    """Phase 1: profit_factor >= 1.3, trades >= 20, net_return > 5%."""
    criteria = [
        GateCriterion("profit_factor", 1.3, metrics.profit_factor, metrics.profit_factor >= 1.3),
        GateCriterion("total_trades", 20, metrics.total_trades, metrics.total_trades >= 20),
        GateCriterion("net_return_pct", 5.0, metrics.net_return_pct, metrics.net_return_pct >= 5.0),
    ]

    # Hard rejects
    if metrics.total_trades < 15:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=("Fewer than 15 trades -- fundamental signal issue",),
        )
    if metrics.max_dd_pct > 0.30:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=(f"Max DD {metrics.max_dd_pct:.1%} exceeds 30% hard limit",),
        )

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=tuple(criteria))

    category, recs = _categorize_failure(criteria, gr)
    return GateResult(passed=False, criteria=tuple(criteria),
                      failure_category=category, recommendations=tuple(recs))


def _gate_phase_2(metrics: BRSMetrics, gr: dict | None) -> GateResult:
    """Phase 2 (SIGNAL_SELECT): Same as Phase 1 — PF >= 1.3, trades >= 20, return > 5%."""
    criteria = [
        GateCriterion("profit_factor", 1.3, metrics.profit_factor, metrics.profit_factor >= 1.3),
        GateCriterion("total_trades", 20, metrics.total_trades, metrics.total_trades >= 20),
        GateCriterion("net_return_pct", 5.0, metrics.net_return_pct, metrics.net_return_pct >= 5.0),
    ]

    if metrics.total_trades < 15:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=("Fewer than 15 trades",),
        )
    if metrics.max_dd_pct > 0.30:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=(f"Max DD {metrics.max_dd_pct:.1%} exceeds 30% hard limit",),
        )

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=tuple(criteria))

    category, recs = _categorize_failure(criteria, gr)
    return GateResult(passed=False, criteria=tuple(criteria),
                      failure_category=category, recommendations=tuple(recs))


def _gate_phase_3(metrics: BRSMetrics, gr: dict | None) -> GateResult:
    """Phase 3: profit_factor >= 1.5, net_return >= 10%, max_dd <= 22%."""
    criteria = [
        GateCriterion("profit_factor", 1.5, metrics.profit_factor, metrics.profit_factor >= 1.5),
        GateCriterion("net_return_pct", 10.0, metrics.net_return_pct, metrics.net_return_pct >= 10.0),
        GateCriterion("max_dd_pct", 0.22, metrics.max_dd_pct, metrics.max_dd_pct <= 0.22),
    ]

    if metrics.total_trades < 15:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=("Fewer than 15 trades",),
        )
    if metrics.max_dd_pct > 0.25:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=(f"Max DD {metrics.max_dd_pct:.1%} exceeds 25% hard limit",),
        )

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=tuple(criteria))

    category, recs = _categorize_failure(criteria, gr)
    return GateResult(passed=False, criteria=tuple(criteria),
                      failure_category=category, recommendations=tuple(recs))


def _gate_phase_4(
    metrics: BRSMetrics,
    gr: dict | None,
    prior_metrics: dict | None = None,
) -> GateResult:
    """Phase 4: calmar >= 1.0, max_dd <= 18%, net_return >= 15%, no regression."""
    criteria = [
        GateCriterion("calmar", 1.0, metrics.calmar, metrics.calmar >= 1.0),
        GateCriterion("max_dd_pct", 0.18, metrics.max_dd_pct, metrics.max_dd_pct <= 0.18),
        GateCriterion("net_return_pct", 15.0, metrics.net_return_pct, metrics.net_return_pct >= 15.0),
    ]

    if metrics.max_dd_pct > 0.20:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=(f"Max DD {metrics.max_dd_pct:.1%} exceeds 20% hard limit",),
        )
    if metrics.sharpe < 0.4:
        return GateResult(
            passed=False, criteria=tuple(criteria),
            failure_category="structural_issue",
            recommendations=(f"Sharpe {metrics.sharpe:.2f} below 0.4 floor",),
        )

    # Regression check: compare against Phase 3 final metrics
    if prior_metrics:
        _REGRESSION_CHECKS = {
            "sharpe": ("higher_is_better", 0.10),
            "calmar": ("higher_is_better", 0.10),
            "max_dd_pct": ("lower_is_better", 0.10),
        }
        for metric_name, (direction, threshold) in _REGRESSION_CHECKS.items():
            prior_val = prior_metrics.get(metric_name)
            current_val = getattr(metrics, metric_name, None)
            if prior_val is None or current_val is None or prior_val == 0:
                continue
            if direction == "higher_is_better":
                regressed = current_val < prior_val * (1 - threshold)
            else:  # lower_is_better
                regressed = current_val > prior_val * (1 + threshold)
            if regressed:
                criteria.append(GateCriterion(
                    f"no_regression_{metric_name}",
                    prior_val,
                    current_val,
                    False,
                ))

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=tuple(criteria))

    category, recs = _categorize_failure(criteria, gr)
    return GateResult(passed=False, criteria=tuple(criteria),
                      failure_category=category, recommendations=tuple(recs))


# ---------------------------------------------------------------------------
# Failure categorization (4 categories)
# ---------------------------------------------------------------------------

def _categorize_failure(
    criteria: list[GateCriterion],
    greedy_result: dict | None,
) -> tuple[str, list[str]]:
    """Categorize gate failure into one of 4 types.

    Categories:
      - scoring_ineffective: score improved but gate metrics didn't move
      - candidates_exhausted: all candidates tried, none improved
      - structural_issue: fundamental problem (too few trades, extreme DD)
      - diagnostic_needed: >=2 criteria within 85% of target but failing
    """
    recs: list[str] = []
    failing = [c for c in criteria if not c.passed]

    # Check for diagnostic_needed: criteria close to target
    _LOWER_IS_BETTER = {"max_dd_pct", "no_regression_max_dd_pct"}
    close_to_target = 0
    for c in failing:
        if c.target > 0:
            if c.name in _LOWER_IS_BETTER:
                close = c.actual <= c.target * 1.15  # within 15% above target
            else:
                close = c.actual / c.target >= 0.85  # within 85% of target
            if close:
                close_to_target += 1
                ratio = c.target / c.actual if c.name in _LOWER_IS_BETTER and c.actual > 0 else c.actual / c.target
                recs.append(f"{c.name}: {c.actual:.3f} at {ratio:.0%} of target {c.target:.3f}")

    if close_to_target >= 2:
        recs.insert(0, "Multiple criteria near target -- run deeper diagnostics")
        return "diagnostic_needed", recs

    # Check greedy result for scoring vs candidates issue
    if greedy_result:
        n_rounds = greedy_result.get("n_rounds", 0)
        kept = greedy_result.get("kept_features", [])
        if n_rounds > 0 and not kept:
            recs.append("No candidates improved score -- try different experiment categories")
            return "candidates_exhausted", recs
        if kept and failing:
            recs.append("Score improved but gate metrics lagged -- adjust scoring weights")
            return "scoring_ineffective", recs

    # Default
    for c in failing:
        recs.append(f"{c.name}: actual={c.actual:.3f} vs target={c.target:.3f}")
    return "scoring_ineffective", recs
