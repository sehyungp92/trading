from __future__ import annotations

import math
from dataclasses import dataclass

IMMUTABLE_WEIGHTS: dict[str, float] = {
    "total_return": 0.18,
    "nq1_return": 0.22,
    "nq1_frequency": 0.18,
    "nq1_quality": 0.16,
    "execution": 0.10,
    "drawdown": 0.08,
    "robustness": 0.08,
}

PHASE_WEIGHTS: dict[int, dict[str, float]] = {phase: dict(IMMUTABLE_WEIGHTS) for phase in range(1, 8)}
PHASE_WEIGHTS[7] = {
    "total_return": 0.12,
    "nq1_return": 0.26,
    "nq1_frequency": 0.20,
    "nq1_quality": 0.20,
    "execution": 0.08,
    "drawdown": 0.06,
    "robustness": 0.08,
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "min_trades": 60,
        "min_pf": 1.10,
        "max_dd_pct": 0.18,
        "min_avg_r": 0.00,
        "min_nq1_trades": 2,
        "min_nq1_quality_trades": 5,
        "min_nq1_avg_r": 0.00,
        "min_nq1_pf": 1.05,
    },
    2: {
        "min_trades": 60,
        "min_pf": 1.10,
        "max_dd_pct": 0.18,
        "min_avg_r": 0.00,
        "min_nq1_trades": 2,
        "min_nq1_quality_trades": 5,
        "min_nq1_avg_r": 0.00,
        "min_nq1_pf": 1.05,
    },
    3: {
        "min_trades": 65,
        "min_pf": 1.12,
        "max_dd_pct": 0.16,
        "min_avg_r": 0.03,
        "min_nq1_trades": 2,
        "min_nq1_quality_trades": 5,
        "min_nq1_avg_r": 0.05,
        "min_nq1_pf": 1.08,
    },
    4: {
        "min_trades": 65,
        "min_pf": 1.12,
        "max_dd_pct": 0.16,
        "min_avg_r": 0.03,
        "min_nq1_trades": 2,
        "min_nq1_quality_trades": 5,
        "min_nq1_avg_r": 0.10,
        "min_nq1_pf": 1.10,
    },
    5: {
        "min_trades": 70,
        "min_pf": 1.15,
        "max_dd_pct": 0.16,
        "min_avg_r": 0.05,
        "min_nq1_trades": 2,
        "min_nq1_quality_trades": 5,
        "min_nq1_avg_r": 0.20,
        "min_nq1_pf": 1.20,
    },
    6: {
        "min_trades": 70,
        "min_pf": 1.15,
        "max_dd_pct": 0.16,
        "min_avg_r": 0.05,
        "min_nq1_trades": 2,
        "min_nq1_quality_trades": 5,
        "min_nq1_avg_r": 0.25,
        "min_nq1_pf": 1.25,
        "max_nq1_top_trade_share": 0.70,
    },
    7: {
        "min_trades": 120,
        "min_pf": 1.20,
        "max_dd_pct": 0.16,
        "min_avg_r": 0.05,
        "min_nq1_trades": 100,
        "min_nq1_quality_trades": 25,
        "min_nq1_avg_r": 0.60,
        "min_nq1_pf": 1.35,
        "min_nq1_total_r_per_month": 2.50,
        "min_nq1_request_to_fill_rate": 0.30,
        "max_nq1_top_trade_share": 0.70,
        "max_nq1_positive_mfe_loser_rate": 0.45,
        "use_nq1_subfamily_quality": 1.0,
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
    avg_r = metrics.get("avg_r", 0.0)
    nq1_trades = metrics.get("module_second_wind_trades", 0.0)
    nq1_avg_r = metrics.get("module_second_wind_avg_r", 0.0)
    nq1_pf = metrics.get("module_second_wind_profit_factor", 0.0)
    nq1_top_share = metrics.get("module_second_wind_top_trade_share", 0.0)
    nq1_total_r_per_month = metrics.get("module_second_wind_total_r_per_month", 0.0)
    nq1_request_to_fill = metrics.get("routing_second_wind_request_to_fill_rate", 0.0)
    nq1_positive_mfe_loser = metrics.get("module_second_wind_positive_mfe_loser_rate", 0.0)

    if total_trades < rejects.get("min_trades", 0.0):
        return NqRegimeScore(0.0, True, "min_trades")
    if pf < rejects.get("min_pf", 0.0):
        return NqRegimeScore(0.0, True, "min_pf")
    if dd > rejects.get("max_dd_pct", 1.0):
        return NqRegimeScore(0.0, True, "max_dd_pct")
    if avg_r < rejects.get("min_avg_r", -99.0):
        return NqRegimeScore(0.0, True, "min_avg_r")
    if nq1_trades < rejects.get("min_nq1_trades", 0.0):
        return NqRegimeScore(0.0, True, "min_nq1_trades")

    quality_trades = rejects.get("min_nq1_quality_trades", 0.0)
    if nq1_trades >= quality_trades:
        if nq1_avg_r < rejects.get("min_nq1_avg_r", -99.0):
            return NqRegimeScore(0.0, True, "min_nq1_avg_r")
        if nq1_pf < rejects.get("min_nq1_pf", 0.0):
            return NqRegimeScore(0.0, True, "min_nq1_pf")
        if nq1_total_r_per_month < rejects.get("min_nq1_total_r_per_month", -99.0):
            return NqRegimeScore(0.0, True, "min_nq1_total_r_per_month")
        if nq1_request_to_fill < rejects.get("min_nq1_request_to_fill_rate", 0.0):
            return NqRegimeScore(0.0, True, "min_nq1_request_to_fill_rate")
        if nq1_top_share > rejects.get("max_nq1_top_trade_share", 1.0):
            return NqRegimeScore(0.0, True, "max_nq1_top_trade_share")
        if nq1_positive_mfe_loser > rejects.get("max_nq1_positive_mfe_loser_rate", 1.0):
            return NqRegimeScore(0.0, True, "max_nq1_positive_mfe_loser_rate")

    n_shrink = _shrink(total_trades, 80.0)
    nq1_shrink = _shrink(nq1_trades, 20.0)
    components = {
        "total_return": _clip(metrics.get("total_r_per_month", 0.0) * n_shrink / 5.0, -1.0, 1.5),
        "nq1_return": _clip(metrics.get("module_second_wind_total_r_per_month", 0.0) * nq1_shrink / 2.5, -1.0, 1.5),
        "nq1_frequency": _clip(metrics.get("module_second_wind_trades_per_month", 0.0) / 2.5, 0.0, 1.5),
        "nq1_quality": _nq1_quality(
            metrics,
            nq1_shrink,
            use_subfamily_quality=bool(rejects.get("use_nq1_subfamily_quality", 0.0)),
        ),
        "execution": _execution_score(metrics),
        "drawdown": _clip(1.0 - dd / 0.10, -1.0, 1.0),
        "robustness": _robustness_score(metrics, n_shrink, nq1_shrink),
    }
    active_weights = weights or IMMUTABLE_WEIGHTS
    total_weight = sum(active_weights.values()) or 1.0
    score = sum(components.get(key, 0.0) * value for key, value in active_weights.items()) / total_weight
    return NqRegimeScore(float(score))


def _nq1_quality(metrics: dict[str, float], shrink: float, *, use_subfamily_quality: bool = False) -> float:
    avg_r = metrics.get("module_second_wind_avg_r", 0.0)
    pf = metrics.get("module_second_wind_profit_factor", 0.0)
    win_rate = metrics.get("module_second_wind_win_rate", 0.0)
    mae = abs(metrics.get("module_second_wind_avg_mae_r", 0.0))
    expectancy = _clip(avg_r / 0.80, -1.0, 1.5)
    pf_score = _clip(math.log(max(pf, 0.01)) / math.log(3.0), -1.0, 1.5)
    win_score = _clip((win_rate - 0.35) / 0.30, -0.5, 1.0)
    mae_score = _clip(1.0 - mae / 0.75, -0.5, 1.0)
    if not use_subfamily_quality:
        return _clip((0.45 * expectancy + 0.30 * pf_score + 0.15 * win_score + 0.10 * mae_score) * shrink, -1.0, 1.5)
    subfamily_score = _nq1_weak_subfamily_score(metrics)
    return _clip(
        (0.40 * expectancy + 0.27 * pf_score + 0.13 * win_score + 0.08 * mae_score + 0.12 * subfamily_score) * shrink,
        -1.0,
        1.5,
    )


def _nq1_weak_subfamily_score(metrics: dict[str, float]) -> float:
    weighted = 0.0
    total_weight = 0.0
    for setup_type in ("pm_vwap_reclaim", "pm_second_leg"):
        prefix = f"module_second_wind_{setup_type}"
        trades = metrics.get(f"{prefix}_trades", 0.0)
        if trades <= 0:
            continue
        shrink = _shrink(trades, 10.0)
        avg_r = metrics.get(f"{prefix}_avg_r", 0.0)
        pf = metrics.get(f"{prefix}_profit_factor", 0.0)
        total_r = metrics.get(f"{prefix}_total_r", 0.0)
        item = (
            0.50 * _clip(avg_r / 0.60, -1.0, 1.5)
            + 0.30 * _clip(math.log(max(pf, 0.01)) / math.log(2.5), -1.0, 1.5)
            + 0.20 * _clip(total_r / 5.0, -1.0, 1.5)
        )
        weighted += item * shrink
        total_weight += shrink
    return _clip(weighted / total_weight if total_weight else 0.20, -1.0, 1.5)


def _execution_score(metrics: dict[str, float]) -> float:
    nq1_fill = metrics.get("routing_second_wind_request_to_fill_rate", 0.0)
    overall = metrics.get("execution_conversion", 0.0)
    selected = metrics.get("routing_second_wind_select_rate_when_valid", 0.0)
    score = (
        0.45 * _clip(nq1_fill / 0.35, 0.0, 1.5)
        + 0.35 * _clip(overall / 0.25, 0.0, 1.5)
        + 0.20 * _clip(selected / 0.50, 0.0, 1.5)
    )
    return _clip(score, 0.0, 1.5)


def _robustness_score(metrics: dict[str, float], n_shrink: float, nq1_shrink: float) -> float:
    top_share = metrics.get("module_second_wind_top_trade_share", 1.0)
    coverage = metrics.get("routing_second_wind_selected_to_fill_rate", 0.0)
    score = (
        0.45 * nq1_shrink
        + 0.25 * n_shrink
        + 0.20 * _clip(1.0 - top_share / 0.70, -0.5, 1.0)
        + 0.10 * _clip(coverage / 0.67, 0.0, 1.5)
    )
    return _clip(score, -1.0, 1.5)


def _shrink(n: float, k: float) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(n / (n + k))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
