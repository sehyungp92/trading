from __future__ import annotations

import math

from backtests.swing.auto.scoring import CompositeScore

IMMUTABLE_WEIGHTS: dict[str, float] = {
    "edge_velocity_component": 0.35,
    "return_component": 0.25,
    "trade_component": 0.18,
    "expectancy_component": 0.12,
    "pf_component": 0.05,
    "inv_dd_component": 0.03,
    "mfe_quality_component": 0.02,
}

PHASE_WEIGHTS: dict[int, dict[str, float]] = {
    1: dict(IMMUTABLE_WEIGHTS),
    2: dict(IMMUTABLE_WEIGHTS),
    3: dict(IMMUTABLE_WEIGHTS),
    4: dict(IMMUTABLE_WEIGHTS),
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "min_trades": 1,
        "max_dd_pct": 0.20,
        "min_pf": 1.05,
        "min_net_profit": 0.0,
        "min_expectancy_dollar": 0.0,
        "min_tpm": 0.0,
        "min_edge_velocity": 0.0,
    },
    2: {
        "min_trades": 1,
        "max_dd_pct": 0.20,
        "min_pf": 1.05,
        "min_net_profit": 0.0,
        "min_expectancy_dollar": 0.0,
        "min_tpm": 0.0,
        "min_edge_velocity": 0.0,
    },
    3: {
        "min_trades": 1,
        "max_dd_pct": 0.20,
        "min_pf": 1.05,
        "min_net_profit": 0.0,
        "min_expectancy_dollar": 0.0,
        "min_tpm": 0.0,
        "min_edge_velocity": 0.0,
    },
    4: {
        "min_trades": 1,
        "max_dd_pct": 0.20,
        "min_pf": 1.05,
        "min_net_profit": 0.0,
        "min_expectancy_dollar": 0.0,
        "min_tpm": 0.0,
        "min_edge_velocity": 0.0,
    },
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
    del equity_curve

    edge_velocity = float(metrics.expectancy_dollar * metrics.trades_per_month)
    edge_velocity_component = _scaled_edge_velocity_component(edge_velocity)
    return_component = _scaled_return_component(float(metrics.net_profit), initial_equity)
    trade_component = _scaled_trade_component(float(metrics.trades_per_month))
    expectancy_component = _scaled_expectancy_component(float(metrics.expectancy_dollar))
    pf_component = _scaled_pf_component(float(metrics.profit_factor))
    inv_dd_component = _scaled_inv_dd_component(float(metrics.max_drawdown_pct))
    mfe_quality_component = _scaled_mfe_quality_component(
        float(getattr(metrics, "avg_mfe_r", 0.0)),
        float(getattr(metrics, "winner_capture_ratio", 0.0)),
    )

    weights = dict(PHASE_WEIGHTS.get(phase, IMMUTABLE_WEIGHTS))
    weights.update(weight_overrides or {})
    total_weight = sum(weights.values()) or 1.0
    weights = {key: value / total_weight for key, value in weights.items()}

    total = (
        weights.get("edge_velocity_component", 0.0) * edge_velocity_component
        + weights.get("return_component", 0.0) * return_component
        + weights.get("trade_component", 0.0) * trade_component
        + weights.get("expectancy_component", 0.0) * expectancy_component
        + weights.get("pf_component", 0.0) * pf_component
        + weights.get("inv_dd_component", 0.0) * inv_dd_component
        + weights.get("mfe_quality_component", 0.0) * mfe_quality_component
    )

    reject_reason = _check_hard_rejects(phase, metrics, edge_velocity, hard_rejects)
    return CompositeScore(
        calmar_component=edge_velocity_component,
        pf_component=pf_component,
        inv_dd_component=inv_dd_component,
        net_profit_component=return_component,
        total=total,
        rejected=reject_reason is not None,
        reject_reason=reject_reason or "",
    )


def _check_hard_rejects(
    phase: int,
    metrics,
    edge_velocity: float,
    hard_rejects: dict[str, float] | None,
) -> str | None:
    rejects = dict(PHASE_HARD_REJECTS.get(phase, {}))
    rejects.update(hard_rejects or {})

    min_trades = int(rejects.get("min_trades", 0))
    if metrics.total_trades < min_trades:
        return f"phase{phase}_too_few_trades ({metrics.total_trades} < {min_trades})"

    max_dd_pct = rejects.get("max_dd_pct")
    if max_dd_pct is not None and metrics.max_drawdown_pct > float(max_dd_pct):
        return f"phase{phase}_max_dd ({metrics.max_drawdown_pct:.1%} > {float(max_dd_pct):.1%})"

    min_pf = rejects.get("min_pf")
    if min_pf is not None and metrics.profit_factor < float(min_pf):
        return f"phase{phase}_low_pf ({metrics.profit_factor:.2f} < {float(min_pf):.2f})"

    min_net_profit = rejects.get("min_net_profit")
    if min_net_profit is not None and metrics.net_profit < float(min_net_profit):
        return (
            f"phase{phase}_low_net_profit "
            f"({metrics.net_profit:.2f} < {float(min_net_profit):.2f})"
        )

    min_expectancy_dollar = rejects.get("min_expectancy_dollar")
    if min_expectancy_dollar is not None and metrics.expectancy_dollar < float(min_expectancy_dollar):
        return (
            f"phase{phase}_low_expectancy_dollar "
            f"({metrics.expectancy_dollar:.2f} < {float(min_expectancy_dollar):.2f})"
        )

    min_tpm = rejects.get("min_tpm")
    if min_tpm is not None and metrics.trades_per_month < float(min_tpm):
        return f"phase{phase}_low_tpm ({metrics.trades_per_month:.2f} < {float(min_tpm):.2f})"

    min_edge_velocity = rejects.get("min_edge_velocity")
    if min_edge_velocity is not None and edge_velocity < float(min_edge_velocity):
        return (
            f"phase{phase}_low_edge_velocity "
            f"({edge_velocity:.2f} < {float(min_edge_velocity):.2f})"
        )

    return None


def _scaled_edge_velocity_component(edge_velocity: float) -> float:
    if edge_velocity <= 0:
        return 0.0
    return min(math.log1p(edge_velocity) / math.log1p(110.0), 1.0)


def _scaled_return_component(net_profit: float, initial_equity: float) -> float:
    if initial_equity <= 0:
        return 0.0
    return_pct = max(net_profit / initial_equity, 0.0)
    return min(math.log1p(return_pct) / math.log1p(0.80), 1.0)


def _scaled_trade_component(trades_per_month: float) -> float:
    if trades_per_month <= 0:
        return 0.0
    return min(math.sqrt(trades_per_month / 2.40), 1.0)


def _scaled_expectancy_component(expectancy_dollar: float) -> float:
    if expectancy_dollar <= 0:
        return 0.0
    return min(math.log1p(expectancy_dollar) / math.log1p(60.0), 1.0)


def _scaled_pf_component(profit_factor: float) -> float:
    if not math.isfinite(profit_factor):
        return 1.0
    if profit_factor <= 1.0:
        return 0.0
    return min(math.log(profit_factor) / math.log(5.5), 1.0)


def _scaled_inv_dd_component(max_drawdown_pct: float) -> float:
    return min(max(1.0 - max_drawdown_pct / 0.10, 0.0), 1.0)


def _scaled_mfe_quality_component(avg_mfe_r: float, winner_capture_ratio: float) -> float:
    avg_mfe_component = min(max(avg_mfe_r / 0.35, 0.0), 1.0)
    winner_capture_component = min(max(winner_capture_ratio / 0.60, 0.0), 1.0)
    return 0.60 * avg_mfe_component + 0.40 * winner_capture_component
