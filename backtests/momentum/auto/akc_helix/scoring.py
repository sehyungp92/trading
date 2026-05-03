from __future__ import annotations

from backtests.momentum.auto.scoring import CompositeScore

PHASE_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"trade_count": 0.24, "net_profit": 0.22, "expectancy": 0.20, "pf": 0.18, "calmar": 0.10, "inv_dd": 0.06},
    2: {"trade_count": 0.22, "net_profit": 0.26, "expectancy": 0.18, "pf": 0.18, "calmar": 0.10, "inv_dd": 0.06},
    3: {"trade_count": 0.18, "net_profit": 0.24, "expectancy": 0.20, "pf": 0.20, "calmar": 0.10, "inv_dd": 0.08},
    4: {"trade_count": 0.20, "net_profit": 0.24, "expectancy": 0.18, "pf": 0.18, "calmar": 0.10, "inv_dd": 0.10},
    5: {"trade_count": 0.18, "net_profit": 0.26, "expectancy": 0.18, "pf": 0.20, "calmar": 0.10, "inv_dd": 0.08},
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 85, "max_dd_pct": 0.18, "min_pf": 4.00},
    2: {"min_trades": 85, "max_dd_pct": 0.18, "min_pf": 4.00},
    3: {"min_trades": 85, "max_dd_pct": 0.18, "min_pf": 4.00},
    4: {"min_trades": 87, "max_dd_pct": 0.18, "min_pf": 4.00},
    5: {"min_trades": 88, "max_dd_pct": 0.19, "min_pf": 4.00},
}

PHASE_TRADE_TARGETS = {
    1: 88.0,
    2: 92.0,
    3: 94.0,
    4: 98.0,
    5: 100.0,
}

NET_PROFIT_TARGET_USD = 150_000.0
EXPECTANCY_TARGET_USD = 1_750.0
PF_TARGET = 4.50
CALMAR_TARGET = 11.50
MAX_DD_CLIP = 0.26

# ---------------------------------------------------------------------------
# Rebase calibration (corrected NQ daily data -- collapsed baseline)
# ---------------------------------------------------------------------------
REBASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 90, "max_dd_pct": 0.34, "min_pf": 2.10},
    2: {"min_trades": 105, "max_dd_pct": 0.34, "min_pf": 2.00},
    3: {"min_trades": 110, "max_dd_pct": 0.35, "min_pf": 2.00},
    4: {"min_trades": 115, "max_dd_pct": 0.35, "min_pf": 2.00},
    5: {"min_trades": 115, "max_dd_pct": 0.34, "min_pf": 2.00},
    6: {"min_trades": 120, "max_dd_pct": 0.34, "min_pf": 2.10},
    7: {"min_trades": 125, "max_dd_pct": 0.34, "min_pf": 2.20},
}

REBASE_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"trade_count": 0.36, "net_profit": 0.28, "pf": 0.14, "expectancy": 0.08, "calmar": 0.08, "inv_dd": 0.06},
    2: {"trade_count": 0.30, "net_profit": 0.28, "expectancy": 0.14, "pf": 0.14, "calmar": 0.08, "inv_dd": 0.06},
    3: {"trade_count": 0.30, "net_profit": 0.28, "pf": 0.16, "expectancy": 0.10, "calmar": 0.10, "inv_dd": 0.06},
    4: {"trade_count": 0.26, "net_profit": 0.30, "pf": 0.16, "expectancy": 0.10, "calmar": 0.10, "inv_dd": 0.08},
    5: {"net_profit": 0.34, "calmar": 0.18, "inv_dd": 0.14, "pf": 0.14, "expectancy": 0.10, "trade_count": 0.10},
    6: {"net_profit": 0.34, "trade_count": 0.22, "calmar": 0.16, "pf": 0.12, "expectancy": 0.08, "inv_dd": 0.08},
    7: {"net_profit": 0.32, "trade_count": 0.24, "pf": 0.18, "expectancy": 0.10, "calmar": 0.10, "inv_dd": 0.06},
}

REBASE_TRADE_TARGETS = {
    1: 120.0,
    2: 135.0,
    3: 145.0,
    4: 150.0,
    5: 150.0,
    6: 155.0,
    7: 160.0,
}

REBASE_NET_PROFIT_TARGET_USD = 130_000.0
REBASE_EXPECTANCY_TARGET_USD = 700.0
REBASE_PF_TARGET = 2.80
REBASE_CALMAR_TARGET = 6.00
REBASE_MAX_DD_CLIP = 0.35


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def score_phase_metrics(
    phase: int,
    metrics,
    initial_equity: float,
    *,
    equity_curve=None,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
    apply_hard_rejects: bool = True,
    target_overrides: dict[str, float] | None = None,
) -> CompositeScore:
    rejects = dict(PHASE_HARD_REJECTS.get(phase, {}))
    rejects.update(hard_rejects or {})

    if apply_hard_rejects:
        min_trades = int(rejects.get("min_trades", 0))
        if metrics.total_trades < min_trades:
            return CompositeScore(0, 0, 0, 0, 0, rejected=True, reject_reason=f"phase{phase}_too_few_trades ({metrics.total_trades} < {min_trades})")

        max_dd_pct = rejects.get("max_dd_pct")
        if max_dd_pct is not None and metrics.max_drawdown_pct > float(max_dd_pct):
            return CompositeScore(0, 0, 0, 0, 0, rejected=True, reject_reason=f"phase{phase}_max_dd ({metrics.max_drawdown_pct:.1%} > {float(max_dd_pct):.1%})")

        min_pf = rejects.get("min_pf")
        if min_pf is not None and metrics.profit_factor < float(min_pf):
            return CompositeScore(0, 0, 0, 0, 0, rejected=True, reject_reason=f"phase{phase}_low_pf ({metrics.profit_factor:.2f} < {float(min_pf):.2f})")

    weights = dict(PHASE_WEIGHTS.get(phase, {}))
    weights.update(weight_overrides or {})
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {key: value / total_weight for key, value in weights.items()}

    if equity_curve is not None and len(equity_curve) >= 2:
        net_profit = float(equity_curve[-1]) - float(initial_equity)
    else:
        net_profit = float(metrics.net_profit)

    t = target_overrides or {}
    trade_target = t.get("trade_target", float(PHASE_TRADE_TARGETS.get(phase, 100.0)))
    net_profit_target = t.get("net_profit_target", NET_PROFIT_TARGET_USD)
    expectancy_target = t.get("expectancy_target", EXPECTANCY_TARGET_USD)
    pf_target = t.get("pf_target", PF_TARGET)
    calmar_target = t.get("calmar_target", CALMAR_TARGET)
    max_dd_clip = t.get("max_dd_clip", MAX_DD_CLIP)

    net_profit_component = _clip01(net_profit / net_profit_target)
    expectancy_component = _clip01(metrics.expectancy_dollar / expectancy_target)
    pf_component = _clip01(metrics.profit_factor / pf_target)
    calmar_component = _clip01(metrics.calmar / calmar_target)
    inv_dd_component = _clip01(1.0 - metrics.max_drawdown_pct / max_dd_clip)
    trade_count_component = _clip01(metrics.total_trades / trade_target)

    total = (
        weights.get("net_profit", 0.0) * net_profit_component
        + weights.get("expectancy", 0.0) * expectancy_component
        + weights.get("pf", 0.0) * pf_component
        + weights.get("calmar", 0.0) * calmar_component
        + weights.get("trade_count", 0.0) * trade_count_component
        + weights.get("inv_dd", 0.0) * inv_dd_component
    )
    return CompositeScore(
        calmar_component=calmar_component,
        pf_component=pf_component,
        inv_dd_component=inv_dd_component,
        net_profit_component=net_profit_component,
        total=total,
    )
