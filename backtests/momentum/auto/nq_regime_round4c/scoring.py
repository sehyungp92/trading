from __future__ import annotations

import math
from dataclasses import dataclass

IMMUTABLE_WEIGHTS: dict[str, float] = {
    "nq3_return": 0.26,
    "nq3_frequency": 0.22,
    "nq3_expectancy": 0.16,
    "nq3_capture": 0.12,
    "nq3_execution": 0.10,
    "nq3_pf": 0.08,
    "drawdown": 0.06,
}

PHASE_WEIGHTS: dict[int, dict[str, float]] = {phase: dict(IMMUTABLE_WEIGHTS) for phase in range(1, 7)}
PHASE_WEIGHTS[7] = {
    "nq3_capture": 0.26,
    "nq3_return": 0.22,
    "nq3_expectancy": 0.18,
    "nq3_pf": 0.12,
    "drawdown": 0.10,
    "nq3_frequency": 0.08,
    "nq3_execution": 0.04,
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 70, "min_pf": 1.20, "max_dd_pct": 0.14, "min_nq3_trades": 60, "min_nq3_avg_r": 0.70, "min_nq3_pf": 3.00},
    2: {"min_trades": 75, "min_pf": 1.30, "max_dd_pct": 0.14, "min_nq3_trades": 70, "min_nq3_avg_r": 0.85, "min_nq3_pf": 3.50},
    3: {"min_trades": 80, "min_pf": 1.35, "max_dd_pct": 0.13, "min_nq3_trades": 75, "min_nq3_avg_r": 0.95, "min_nq3_pf": 4.00},
    4: {"min_trades": 80, "min_pf": 1.40, "max_dd_pct": 0.13, "min_nq3_trades": 75, "min_nq3_avg_r": 1.00, "min_nq3_pf": 4.00},
    5: {"min_trades": 80, "min_pf": 1.40, "max_dd_pct": 0.12, "min_nq3_trades": 75, "min_nq3_avg_r": 1.00, "min_nq3_pf": 4.00},
    6: {"min_trades": 85, "min_pf": 1.45, "max_dd_pct": 0.12, "min_nq3_trades": 80, "min_nq3_avg_r": 1.05, "min_nq3_pf": 4.50},
    7: {
        "min_trades": 85,
        "min_pf": 1.45,
        "max_dd_pct": 0.12,
        "min_nq3_trades": 100,
        "min_nq3_avg_r": 1.05,
        "min_nq3_pf": 4.50,
        "min_nq3_capture": 0.68,
        "max_nq3_positive_mfe_loser_rate": 0.08,
    },
}


@dataclass(frozen=True)
class NqRegimeScore:
    score: float
    rejected: bool = False
    reject_reason: str = ""


def composite_score(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> NqRegimeScore:
    rejects = hard_rejects or {}
    total_trades = metrics.get("total_trades", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    dd = metrics.get("max_drawdown_pct", 0.0)
    nq3_trades = metrics.get("module_liquidity_reversion_trades", 0.0)
    nq3_avg_r = metrics.get("module_liquidity_reversion_avg_r", 0.0)
    nq3_pf = metrics.get("module_liquidity_reversion_profit_factor", 0.0)
    if total_trades < rejects.get("min_trades", 0.0):
        return NqRegimeScore(0.0, True, "min_trades")
    if pf < rejects.get("min_pf", 0.0):
        return NqRegimeScore(0.0, True, "min_pf")
    if dd > rejects.get("max_dd_pct", 1.0):
        return NqRegimeScore(0.0, True, "max_dd_pct")
    if nq3_trades < rejects.get("min_nq3_trades", 0.0):
        return NqRegimeScore(0.0, True, "min_nq3_trades")
    if nq3_avg_r < rejects.get("min_nq3_avg_r", -99.0):
        return NqRegimeScore(0.0, True, "min_nq3_avg_r")
    if nq3_pf < rejects.get("min_nq3_pf", 0.0):
        return NqRegimeScore(0.0, True, "min_nq3_pf")
    if metrics.get("module_liquidity_reversion_mfe_capture", 0.0) < rejects.get("min_nq3_capture", 0.0):
        return NqRegimeScore(0.0, True, "min_nq3_capture")
    if metrics.get("module_liquidity_reversion_positive_mfe_loser_rate", 0.0) > rejects.get("max_nq3_positive_mfe_loser_rate", 1.0):
        return NqRegimeScore(0.0, True, "max_nq3_positive_mfe_loser_rate")

    n_shrink = _shrink(nq3_trades, 75.0)
    pf_shrink = _shrink(nq3_trades, 90.0)
    components = {
        "nq3_return": _clip(metrics.get("module_liquidity_reversion_total_r_per_month", 0.0) * n_shrink / 4.0, -1.0, 1.5),
        "nq3_frequency": _clip(metrics.get("module_liquidity_reversion_trades_per_month", 0.0) / 3.0, 0.0, 1.5),
        "nq3_expectancy": _clip((nq3_avg_r * n_shrink) / 1.40, -1.0, 1.5),
        "nq3_capture": _clip(metrics.get("module_liquidity_reversion_mfe_capture", 0.0) / 0.50, 0.0, 1.5),
        "nq3_execution": _clip(metrics.get("routing_liquidity_reversion_request_to_fill_rate", 0.0) / 0.45, 0.0, 1.5),
        "nq3_pf": _clip((math.log(max(nq3_pf, 0.01)) / math.log(5.0)) * pf_shrink, -0.5, 1.5),
        "drawdown": _clip(1.0 - dd / 0.12, -1.0, 1.0),
    }
    active_weights = weights or IMMUTABLE_WEIGHTS
    total_weight = sum(active_weights.values()) or 1.0
    score = sum(components.get(key, 0.0) * value for key, value in active_weights.items()) / total_weight
    return NqRegimeScore(float(score))


def _shrink(n: float, k: float) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(n / (n + k))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
