"""Helix phase gate criteria -- success thresholds per phase.

Phase 1: trades>=250, PF>=1.5, return>=30%, tail>=0.30
Phase 2: trades>=250, PF>=1.7, return>=40%, exit_eff>=0.20
Phase 3: trades>=250, PF>=1.8, return>=50%, max_r_dd<=20
Phase 4: trades>=250, calmar_r>=5.0, return>=55% + no-regression vs P3
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import HelixMetrics


def gate_criteria_phase_1(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("total_trades", 250.0, float(m.total_trades), m.total_trades >= 250),
        GateCriterion("profit_factor", 1.5, m.profit_factor, m.profit_factor >= 1.5),
        GateCriterion("net_return_pct", 30.0, m.net_return_pct, m.net_return_pct >= 30.0),
        GateCriterion("tail_pct", 0.30, m.tail_pct, m.tail_pct >= 0.30),
    ]


def gate_criteria_phase_2(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("total_trades", 250.0, float(m.total_trades), m.total_trades >= 250),
        GateCriterion("profit_factor", 1.7, m.profit_factor, m.profit_factor >= 1.7),
        GateCriterion("net_return_pct", 40.0, m.net_return_pct, m.net_return_pct >= 40.0),
        GateCriterion("exit_efficiency", 0.20, m.exit_efficiency, m.exit_efficiency >= 0.20),
    ]


def gate_criteria_phase_3(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("total_trades", 250.0, float(m.total_trades), m.total_trades >= 250),
        GateCriterion("profit_factor", 1.8, m.profit_factor, m.profit_factor >= 1.8),
        GateCriterion("net_return_pct", 50.0, m.net_return_pct, m.net_return_pct >= 50.0),
        GateCriterion("max_r_dd", 20.0, m.max_r_dd, m.max_r_dd <= 20.0),
    ]


def gate_criteria_phase_4(
    metrics: dict[str, float],
    prior_phase_metrics: dict[str, float] | None = None,
) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    criteria = [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("total_trades", 250.0, float(m.total_trades), m.total_trades >= 250),
        GateCriterion("calmar_r", 5.0, m.calmar_r, m.calmar_r >= 5.0),
        GateCriterion("net_return_pct", 55.0, m.net_return_pct, m.net_return_pct >= 55.0),
    ]

    # No-regression gates vs Phase 3
    if prior_phase_metrics:
        p3_pf = float(prior_phase_metrics.get("profit_factor", 0.0))
        p3_ee = float(prior_phase_metrics.get("exit_efficiency", 0.0))
        p3_nr = float(prior_phase_metrics.get("net_return_pct", 0.0))
        if p3_pf > 0:
            criteria.append(GateCriterion(
                "no_regression_pf", p3_pf * 0.90, m.profit_factor,
                m.profit_factor >= p3_pf * 0.90,
            ))
        if p3_ee > 0:
            criteria.append(GateCriterion(
                "no_regression_exit_eff", p3_ee * 0.90, m.exit_efficiency,
                m.exit_efficiency >= p3_ee * 0.90,
            ))
        if p3_nr > 0:
            criteria.append(GateCriterion(
                "no_regression_net_return", p3_nr * 0.90, m.net_return_pct,
                m.net_return_pct >= p3_nr * 0.90,
            ))

    return criteria


def _to_metrics(metrics: dict[str, float]) -> HelixMetrics:
    fields = HelixMetrics.__dataclass_fields__
    kwargs = {key: metrics.get(key, 0.0) for key in fields}
    kwargs["total_trades"] = int(kwargs.get("total_trades", 0))
    return HelixMetrics(**kwargs)
