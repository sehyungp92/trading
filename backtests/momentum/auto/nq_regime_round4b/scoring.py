from __future__ import annotations

import math
from dataclasses import dataclass

IMMUTABLE_WEIGHTS: dict[str, float] = {
    "structural_return": 0.24,
    "structural_expectancy": 0.24,
    "structural_pf": 0.18,
    "structural_sample": 0.10,
    "structural_conversion": 0.08,
    "structural_capture": 0.10,
    "global_guard": 0.06,
}

PHASE_WEIGHTS: dict[int, dict[str, float]] = {phase: dict(IMMUTABLE_WEIGHTS) for phase in range(1, 7)}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "min_trades": 60,
        "min_pf": 1.05,
        "max_dd_pct": 0.16,
        "min_avg_r": 0.00,
        "min_structural_trades": 25,
        "min_structural_avg_r": -0.10,
        "min_structural_pf": 1.00,
        "min_structural_conversion": 0.04,
    },
    2: {
        "min_trades": 65,
        "min_pf": 1.05,
        "max_dd_pct": 0.16,
        "min_avg_r": 0.00,
        "min_structural_trades": 35,
        "min_structural_avg_r": -0.03,
        "min_structural_pf": 1.05,
        "min_structural_conversion": 0.06,
    },
    3: {
        "min_trades": 65,
        "min_pf": 1.08,
        "max_dd_pct": 0.14,
        "min_avg_r": 0.02,
        "min_structural_trades": 40,
        "min_structural_avg_r": 0.02,
        "min_structural_pf": 1.12,
        "min_structural_conversion": 0.08,
    },
    4: {
        "min_trades": 70,
        "min_pf": 1.08,
        "max_dd_pct": 0.14,
        "min_avg_r": 0.02,
        "min_structural_trades": 50,
        "min_structural_avg_r": 0.06,
        "min_structural_total_r": 4.0,
        "min_structural_pf": 1.25,
        "min_structural_conversion": 0.10,
        "min_structural_capture": 0.40,
        "max_structural_positive_mfe_loser_rate": 0.35,
    },
    5: {
        "min_trades": 70,
        "min_pf": 1.10,
        "max_dd_pct": 0.14,
        "min_avg_r": 0.03,
        "min_structural_trades": 50,
        "min_structural_avg_r": 0.07,
        "min_structural_total_r": 4.0,
        "min_structural_pf": 1.28,
        "min_structural_conversion": 0.10,
        "min_structural_capture": 0.40,
        "max_structural_positive_mfe_loser_rate": 0.35,
    },
    6: {
        "min_trades": 120,
        "min_pf": 1.10,
        "max_dd_pct": 0.12,
        "min_avg_r": 0.04,
        "min_structural_trades": 50,
        "min_structural_avg_r": 0.07,
        "min_structural_total_r": 4.0,
        "min_structural_pf": 1.30,
        "min_structural_conversion": 0.10,
        "min_structural_capture": 0.40,
        "max_structural_positive_mfe_loser_rate": 0.35,
        "max_structural_top_trade_share": 0.25,
    },
}


@dataclass(frozen=True)
class NqRegimeRound4bScore:
    score: float
    rejected: bool = False
    reject_reason: str = ""


def composite_score(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> NqRegimeRound4bScore:
    rejects = hard_rejects or {}
    total_trades = metrics.get("total_trades", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    avg_r = metrics.get("avg_r", 0.0)
    dd = metrics.get("max_drawdown_pct", 0.0)
    structural_trades = metrics.get("module_structural_expansion_trades", 0.0)
    structural_avg_r = metrics.get("module_structural_expansion_avg_r", 0.0)
    structural_total_r = metrics.get("module_structural_expansion_total_r", 0.0)
    structural_pf = metrics.get("module_structural_expansion_profit_factor", 0.0)
    structural_capture = metrics.get("module_structural_expansion_mfe_capture", 0.0)
    structural_positive_mfe_loser = metrics.get("module_structural_expansion_positive_mfe_loser_rate", 0.0)
    structural_top_trade_share = metrics.get("module_structural_expansion_top_trade_share", 0.0)
    structural_selected = metrics.get("routing_structural_expansion_selected", 0.0)
    structural_conversion = structural_trades / structural_selected if structural_selected > 0 else 0.0

    if total_trades < rejects.get("min_trades", 0.0):
        return NqRegimeRound4bScore(0.0, True, "min_trades")
    if pf < rejects.get("min_pf", 0.0):
        return NqRegimeRound4bScore(0.0, True, "min_pf")
    if dd > rejects.get("max_dd_pct", 1.0):
        return NqRegimeRound4bScore(0.0, True, "max_dd_pct")
    if avg_r < rejects.get("min_avg_r", -99.0):
        return NqRegimeRound4bScore(0.0, True, "min_avg_r")
    if structural_trades < rejects.get("min_structural_trades", 0.0):
        return NqRegimeRound4bScore(0.0, True, "min_structural_trades")
    if structural_avg_r < rejects.get("min_structural_avg_r", -99.0):
        return NqRegimeRound4bScore(0.0, True, "min_structural_avg_r")
    if structural_total_r < rejects.get("min_structural_total_r", -99.0):
        return NqRegimeRound4bScore(0.0, True, "min_structural_total_r")
    if structural_pf < rejects.get("min_structural_pf", 0.0):
        return NqRegimeRound4bScore(0.0, True, "min_structural_pf")
    if structural_conversion < rejects.get("min_structural_conversion", 0.0):
        return NqRegimeRound4bScore(0.0, True, "min_structural_conversion")
    if structural_capture < rejects.get("min_structural_capture", 0.0):
        return NqRegimeRound4bScore(0.0, True, "min_structural_capture")
    if structural_positive_mfe_loser > rejects.get("max_structural_positive_mfe_loser_rate", 1.0):
        return NqRegimeRound4bScore(0.0, True, "max_structural_positive_mfe_loser_rate")
    if structural_top_trade_share > rejects.get("max_structural_top_trade_share", 1.0):
        return NqRegimeRound4bScore(0.0, True, "max_structural_top_trade_share")

    months = _months_from_metrics(metrics)
    structural_r_per_month = structural_total_r / months if months > 0 else 0.0

    n_shrink = _shrink(structural_trades, 70.0)
    pf_shrink = _shrink(structural_trades, 90.0)
    sample_floor = _clip((structural_trades - 45.0) / 25.0, 0.0, 1.0)
    overbreadth_penalty = 1.0 - 0.30 * _clip((structural_trades - 120.0) / 80.0, 0.0, 1.0)
    leak_penalty = 1.0 - 0.45 * _clip(structural_positive_mfe_loser, 0.0, 1.0)
    concentration_penalty = 1.0 - 0.40 * _clip((structural_top_trade_share - 0.12) / 0.18, 0.0, 1.0)
    global_guard = _global_guard(metrics)

    components = {
        "structural_return": _clip((structural_r_per_month * n_shrink * concentration_penalty) / 0.30, -1.0, 1.5),
        "structural_expectancy": _clip((structural_avg_r * n_shrink * overbreadth_penalty) / 0.15, -1.0, 1.5),
        "structural_pf": _clip((math.log(max(structural_pf, 0.01)) / math.log(1.60)) * pf_shrink, -0.5, 1.5),
        "structural_sample": sample_floor,
        "structural_conversion": _clip(structural_conversion / 0.35, 0.0, 1.5),
        "structural_capture": _clip((structural_capture / 0.55) * leak_penalty, 0.0, 1.5),
        "global_guard": global_guard,
    }
    active_weights = weights or IMMUTABLE_WEIGHTS
    total_weight = sum(active_weights.values()) or 1.0
    score = sum(components.get(key, 0.0) * value for key, value in active_weights.items()) / total_weight
    return NqRegimeRound4bScore(float(score))


def _global_guard(metrics: dict[str, float]) -> float:
    return_component = _clip(metrics.get("total_r", 0.0) / 150.0, -1.0, 1.0)
    pf_component = _clip((math.log(max(metrics.get("profit_factor", 0.01), 0.01)) / math.log(3.0)), -0.5, 1.0)
    dd_component = _clip(1.0 - metrics.get("max_drawdown_pct", 0.0) / 0.12, -1.0, 1.0)
    coverage_component = _clip(metrics.get("module_coverage", 0.0) / 1.0, 0.0, 1.0)
    return float(0.30 * return_component + 0.25 * pf_component + 0.30 * dd_component + 0.15 * coverage_component)


def _months_from_metrics(metrics: dict[str, float]) -> float:
    total_trades = metrics.get("total_trades", 0.0)
    trades_per_month = metrics.get("trades_per_month", 0.0)
    if total_trades > 0 and trades_per_month > 0:
        return max(total_trades / trades_per_month, 1e-9)
    return 1.0


def _shrink(n: float, k: float) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(n / (n + k))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
