from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import PHASE_HARD_REJECTS


def gate_criteria_for_phase(phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
    rejects = PHASE_HARD_REJECTS.get(phase, {})
    structural_selected = metrics.get("routing_structural_expansion_selected", 0.0)
    structural_trades = metrics.get("module_structural_expansion_trades", 0.0)
    structural_conversion = structural_trades / structural_selected if structural_selected > 0 else 0.0
    return [
        _criterion("min_trades", rejects.get("min_trades", 0.0), metrics.get("total_trades", 0.0), ">="),
        _criterion("min_pf", rejects.get("min_pf", 0.0), metrics.get("profit_factor", 0.0), ">="),
        _criterion("max_dd_pct", rejects.get("max_dd_pct", 1.0), metrics.get("max_drawdown_pct", 0.0), "<="),
        _criterion("min_avg_r", rejects.get("min_avg_r", -99.0), metrics.get("avg_r", 0.0), ">="),
        _criterion(
            "min_structural_trades",
            rejects.get("min_structural_trades", 0.0),
            structural_trades,
            ">=",
        ),
        _criterion(
            "min_structural_avg_r",
            rejects.get("min_structural_avg_r", -99.0),
            metrics.get("module_structural_expansion_avg_r", 0.0),
            ">=",
        ),
        _criterion(
            "min_structural_total_r",
            rejects.get("min_structural_total_r", -99.0),
            metrics.get("module_structural_expansion_total_r", 0.0),
            ">=",
        ),
        _criterion(
            "min_structural_pf",
            rejects.get("min_structural_pf", 0.0),
            metrics.get("module_structural_expansion_profit_factor", 0.0),
            ">=",
        ),
        _criterion(
            "min_structural_conversion",
            rejects.get("min_structural_conversion", 0.0),
            structural_conversion,
            ">=",
        ),
        _criterion(
            "min_structural_capture",
            rejects.get("min_structural_capture", 0.0),
            metrics.get("module_structural_expansion_mfe_capture", 0.0),
            ">=",
        ),
        _criterion(
            "max_structural_positive_mfe_loser_rate",
            rejects.get("max_structural_positive_mfe_loser_rate", 1.0),
            metrics.get("module_structural_expansion_positive_mfe_loser_rate", 0.0),
            "<=",
        ),
        _criterion(
            "max_structural_top_trade_share",
            rejects.get("max_structural_top_trade_share", 1.0),
            metrics.get("module_structural_expansion_top_trade_share", 0.0),
            "<=",
        ),
    ]


def _criterion(name: str, target: float, actual: float, op: str) -> GateCriterion:
    passed = actual <= target if op == "<=" else actual >= target
    return GateCriterion(name, target, actual, passed)
