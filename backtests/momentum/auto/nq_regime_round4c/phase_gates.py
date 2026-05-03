from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import PHASE_HARD_REJECTS


def gate_criteria_for_phase(phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
    rejects = PHASE_HARD_REJECTS.get(phase, {})
    criteria = [
        _criterion("min_trades", rejects.get("min_trades", 0), metrics.get("total_trades", 0), ">="),
        _criterion("min_pf", rejects.get("min_pf", 0), metrics.get("profit_factor", 0), ">="),
        _criterion("max_dd_pct", rejects.get("max_dd_pct", 1), metrics.get("max_drawdown_pct", 0), "<="),
        _criterion("min_nq3_trades", rejects.get("min_nq3_trades", 0), metrics.get("module_liquidity_reversion_trades", 0), ">="),
        _criterion("min_nq3_avg_r", rejects.get("min_nq3_avg_r", -99), metrics.get("module_liquidity_reversion_avg_r", 0), ">="),
        _criterion("min_nq3_pf", rejects.get("min_nq3_pf", 0), metrics.get("module_liquidity_reversion_profit_factor", 0), ">="),
    ]
    if "max_nq3_positive_mfe_loser_rate" in rejects:
        criteria.append(
            _criterion(
                "max_nq3_positive_mfe_loser_rate",
                rejects["max_nq3_positive_mfe_loser_rate"],
                metrics.get("module_liquidity_reversion_positive_mfe_loser_rate", 0),
                "<=",
            )
        )
    if "min_nq3_capture" in rejects:
        criteria.append(
            _criterion(
                "min_nq3_capture",
                rejects["min_nq3_capture"],
                metrics.get("module_liquidity_reversion_mfe_capture", 0),
                ">=",
            )
        )
    return criteria


def _criterion(name: str, target: float, actual: float, op: str) -> GateCriterion:
    passed = actual <= target if op == "<=" else actual >= target
    return GateCriterion(name, target, actual, passed)
