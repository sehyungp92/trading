from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import PHASE_HARD_REJECTS


def gate_criteria_for_phase(phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
    rejects = PHASE_HARD_REJECTS.get(phase, {})
    nq1_trades = metrics.get("module_second_wind_trades", 0.0)
    quality_trades = rejects.get("min_nq1_quality_trades", 0.0)
    criteria = [
        _criterion("min_trades", rejects.get("min_trades", 0), metrics.get("total_trades", 0), ">="),
        _criterion("min_pf", rejects.get("min_pf", 0), metrics.get("profit_factor", 0), ">="),
        _criterion("max_dd_pct", rejects.get("max_dd_pct", 1), metrics.get("max_drawdown_pct", 0), "<="),
        _criterion("min_avg_r", rejects.get("min_avg_r", -99), metrics.get("avg_r", 0), ">="),
        _criterion("min_nq1_trades", rejects.get("min_nq1_trades", 0), nq1_trades, ">="),
    ]
    if nq1_trades >= quality_trades:
        criteria.extend(
            [
                _criterion("min_nq1_avg_r", rejects.get("min_nq1_avg_r", -99), metrics.get("module_second_wind_avg_r", 0), ">="),
                _criterion("min_nq1_pf", rejects.get("min_nq1_pf", 0), metrics.get("module_second_wind_profit_factor", 0), ">="),
            ]
        )
        if "min_nq1_total_r_per_month" in rejects:
            criteria.append(
                _criterion(
                    "min_nq1_total_r_per_month",
                    rejects["min_nq1_total_r_per_month"],
                    metrics.get("module_second_wind_total_r_per_month", 0),
                    ">=",
                )
            )
        if "min_nq1_request_to_fill_rate" in rejects:
            criteria.append(
                _criterion(
                    "min_nq1_request_to_fill_rate",
                    rejects["min_nq1_request_to_fill_rate"],
                    metrics.get("routing_second_wind_request_to_fill_rate", 0),
                    ">=",
                )
            )
        if "max_nq1_top_trade_share" in rejects:
            criteria.append(
                _criterion(
                    "max_nq1_top_trade_share",
                    rejects["max_nq1_top_trade_share"],
                    metrics.get("module_second_wind_top_trade_share", 0),
                    "<=",
                )
            )
        if "max_nq1_positive_mfe_loser_rate" in rejects:
            criteria.append(
                _criterion(
                    "max_nq1_positive_mfe_loser_rate",
                    rejects["max_nq1_positive_mfe_loser_rate"],
                    metrics.get("module_second_wind_positive_mfe_loser_rate", 0),
                    "<=",
                )
            )
    return criteria


def _criterion(name: str, target: float, actual: float, op: str) -> GateCriterion:
    passed = actual <= target if op == "<=" else actual >= target
    return GateCriterion(name, target, actual, passed)
