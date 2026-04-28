from __future__ import annotations

from backtests.momentum.auto.scoring import CompositeScore, composite_score

PHASE_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"net_profit": 0.25, "pf": 0.35, "calmar": 0.15, "inv_dd": 0.25},
    2: {"net_profit": 0.35, "pf": 0.25, "calmar": 0.20, "inv_dd": 0.20},
    3: {"net_profit": 0.30, "pf": 0.25, "calmar": 0.20, "inv_dd": 0.25},
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 25, "max_dd_pct": 0.35, "min_pf": 0.80},
    2: {"min_trades": 25, "max_dd_pct": 0.30, "min_pf": 0.90},
    3: {"min_trades": 30, "max_dd_pct": 0.25, "min_pf": 1.00},
}


def score_phase_metrics(
    phase: int,
    metrics,
    initial_equity: float,
    *,
    equity_curve=None,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> CompositeScore:
    rejects = dict(PHASE_HARD_REJECTS.get(phase, {}))
    rejects.update(hard_rejects or {})

    min_trades = int(rejects.get("min_trades", 0))
    if metrics.total_trades < min_trades:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True, reject_reason=f"phase{phase}_too_few_trades ({metrics.total_trades} < {min_trades})")

    max_dd_pct = rejects.get("max_dd_pct")
    if max_dd_pct is not None and metrics.max_drawdown_pct > float(max_dd_pct):
        return CompositeScore(0, 0, 0, 0, 0, rejected=True, reject_reason=f"phase{phase}_max_dd ({metrics.max_drawdown_pct:.1%} > {float(max_dd_pct):.1%})")

    min_pf = rejects.get("min_pf")
    if min_pf is not None and metrics.profit_factor < float(min_pf):
        return CompositeScore(0, 0, 0, 0, 0, rejected=True, reject_reason=f"phase{phase}_low_pf ({metrics.profit_factor:.2f} < {float(min_pf):.2f})")

    base_score = composite_score(metrics, initial_equity, strategy="helix", equity_curve=equity_curve)
    if base_score.rejected:
        return base_score

    weights = dict(PHASE_WEIGHTS.get(phase, {}))
    weights.update(weight_overrides or {})
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {key: value / total_weight for key, value in weights.items()}
    else:
        weights = {"net_profit": 0.35, "pf": 0.30, "calmar": 0.20, "inv_dd": 0.15}

    total = (
        weights.get("net_profit", 0.0) * base_score.net_profit_component
        + weights.get("pf", 0.0) * base_score.pf_component
        + weights.get("calmar", 0.0) * base_score.calmar_component
        + weights.get("inv_dd", 0.0) * base_score.inv_dd_component
    )
    return CompositeScore(
        calmar_component=base_score.calmar_component,
        pf_component=base_score.pf_component,
        inv_dd_component=base_score.inv_dd_component,
        net_profit_component=base_score.net_profit_component,
        total=total,
    )
