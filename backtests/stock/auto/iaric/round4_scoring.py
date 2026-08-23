"""Immutable scoring for the representative IARIC reversion baseline.

Candidate ranking is limited to discovery and calibration data.  The locked
internal-validation interval and sealed holdout are deliberately absent from
this module, so neither can influence a shortlist through an accidental date
filter or a private score implementation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Any, Iterable

from backtests.stock.auto.iaric.representative_contract import SELECTION_FOLDS
from strategies.stock.iaric.core.lanes import issuer_key


SEGMENTS = SELECTION_FOLDS


def _months(start: str, end: str) -> float:
    return max((date.fromisoformat(end) - date.fromisoformat(start)).days / 30.4375, 1.0)


SELECTION_MONTHS = _months(SEGMENTS[0][1], SEGMENTS[-1][2])
EXECUTABLE_CAPACITY_TRADES_PER_MONTH = 40.0

# Exactly seven components.  Weight and scale changes require a new round;
# neither the runner nor a finalizer is allowed to carry a private copy.
SCORE_SPEC: dict[str, dict[str, float]] = {
    "net_expected_r_per_month": {"weight": 0.26, "scale": 4.0},
    "executable_trades_per_month": {"weight": 0.16, "scale": 0.15},
    "worst_fold_r_per_month": {"weight": 0.18, "scale": 2.0},
    "average_r_and_discrimination": {"weight": 0.14, "scale": 0.10},
    "downside_risk": {"weight": 0.12, "scale": 1.0},
    "issuer_sector_concentration": {"weight": 0.08, "scale": 1.0},
    "cost_and_neighbourhood_robustness": {"weight": 0.06, "scale": 1.0},
}

if len(SCORE_SPEC) != 7:
    raise RuntimeError("IARIC Round 4 score must have exactly seven components")
if not math.isclose(sum(spec["weight"] for spec in SCORE_SPEC.values()), 1.0):
    raise RuntimeError("IARIC Round 4 score weights must sum to one")


def _r_by_issuer(attribution: Iterable[dict[str, Any]]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for trade in attribution:
        totals[issuer_key(str(trade.get("symbol", "")))] += float(trade.get("r", 0.0))
    return dict(totals)


def issuer_diagnostics(attribution: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(attribution)
    totals = _r_by_issuer(rows)
    positive = {key: value for key, value in totals.items() if value > 0.0}
    positive_r = sum(positive.values())
    top_issuer, top_r = max(positive.items(), key=lambda item: item[1], default=("", 0.0))
    shares = [value / positive_r for value in positive.values()] if positive_r > 0.0 else []
    hhi = sum(share * share for share in shares)
    issuer_symbols: defaultdict[str, set[str]] = defaultdict(set)
    for trade in rows:
        symbol = str(trade.get("symbol", "")).upper()
        issuer_symbols[issuer_key(symbol)].add(symbol)
    return {
        "issuer_total_r": totals,
        "top_positive_issuer": top_issuer,
        "top_positive_issuer_r": top_r,
        "positive_issuer_r": positive_r,
        "top_positive_issuer_share": top_r / positive_r if positive_r > 0.0 else 0.0,
        "issuer_neutral_total_r": sum(totals.values()) - top_r,
        "positive_issuer_hhi": hhi,
        "effective_positive_issuers": 1.0 / hhi if hhi > 0.0 else 0.0,
        "unique_issuers": len(totals),
        "multi_share_class_issuers": sorted(
            issuer for issuer, symbols in issuer_symbols.items() if len(symbols) > 1
        ),
    }


def sector_diagnostics(attribution: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals: defaultdict[str, float] = defaultdict(float)
    for trade in attribution:
        sector = str(trade.get("sector", "") or "UNKNOWN").strip().upper()
        totals[sector] += float(trade.get("r", 0.0))
    positive = {key: value for key, value in totals.items() if value > 0.0}
    positive_r = sum(positive.values())
    top_sector, top_r = max(positive.items(), key=lambda item: item[1], default=("", 0.0))
    shares = [value / positive_r for value in positive.values()] if positive_r > 0.0 else []
    hhi = sum(share * share for share in shares)
    return {
        "sector_total_r": dict(totals),
        "top_positive_sector": top_sector,
        "top_positive_sector_r": top_r,
        "positive_sector_r": positive_r,
        "top_positive_sector_share": top_r / positive_r if positive_r > 0.0 else 0.0,
        "positive_sector_hhi": hhi,
        "effective_positive_sectors": 1.0 / hhi if hhi > 0.0 else 0.0,
        "unique_sectors": len(totals),
    }


def _segment_totals(attribution: Iterable[dict[str, Any]]) -> dict[str, float]:
    result = {name: 0.0 for name, _, _ in SEGMENTS}
    for trade in attribution:
        date = str(trade.get("entry_time", ""))[:10]
        value = float(trade.get("r", 0.0))
        for name, start, end in SEGMENTS:
            if start <= date <= end:
                result[name] += value
                break
    return result


def _segment_r_per_month(attribution: Iterable[dict[str, Any]]) -> dict[str, float]:
    totals = _segment_totals(attribution)
    return {
        name: totals[name] / _months(start, end)
        for name, start, end in SEGMENTS
    }


def _positive_sleeve_count(attribution: Iterable[dict[str, Any]]) -> int:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for trade in attribution:
        sleeve = str(
            trade.get("sleeve")
            or trade.get("lane")
            or trade.get("entry_lane_id")
            or "INCUMBENT"
        ).upper()
        grouped[sleeve].append(float(trade.get("r", 0.0)))
    # Twenty executions is still a research threshold, not a production gate;
    # the stronger issuer-day and confidence requirements are enforced by the
    # phase contract.  It prevents one lucky trade from creating breadth score.
    return sum(len(values) >= 20 and sum(values) > 0.0 for values in grouped.values())


def _robustness_value(attribution: Iterable[dict[str, Any]]) -> float:
    """Return a frozen per-trade cost/neighbourhood diagnostic when present."""

    values: list[float] = []
    for trade in attribution:
        raw = trade.get("robustness_score")
        if raw is None:
            metadata = trade.get("metadata") or {}
            if isinstance(metadata, dict):
                raw = metadata.get("robustness_score")
        if raw is not None:
            values.append(min(max(float(raw), 0.0), 1.0))
    return sum(values) / len(values) if values else 0.0


def _capacity_utility(trades_per_month: float) -> float:
    capped = min(max(trades_per_month, 0.0), EXECUTABLE_CAPACITY_TRADES_PER_MONTH)
    return math.log1p(capped) / math.log1p(EXECUTABLE_CAPACITY_TRADES_PER_MONTH)


def _selection_economics(attribution: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = sorted(
        list(attribution),
        key=lambda row: (str(row.get("entry_time", "")), str(row.get("symbol", ""))),
    )
    values = [float(row.get("r", 0.0)) for row in rows]
    total = sum(values)
    gross_profit = sum(value for value in values if value > 0.0)
    gross_loss = -sum(value for value in values if value < 0.0)
    if not values:
        profit_factor = 0.0
    elif gross_loss <= 1e-12:
        profit_factor = 5.0
    else:
        profit_factor = min(gross_profit / gross_loss, 5.0)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    tail_count = max(int(math.ceil(len(values) * 0.10)), 1) if values else 0
    expected_shortfall = (
        sum(sorted(values)[:tail_count]) / tail_count if tail_count else 0.0
    )
    return {
        "total_r": total,
        "trades": float(len(values)),
        "avg_r": total / len(values) if values else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown,
        "expected_shortfall_r": expected_shortfall,
    }


def _quality(economics: dict[str, Any]) -> float:
    expectancy = float(economics.get("avg_r", 0.0) or 0.0)
    profit_factor = max(float(economics.get("profit_factor", 0.0) or 0.0), 1e-6)
    # PF enters logarithmically so an unstable near-zero denominator cannot
    # dominate the more interpretable net expectancy in R units.
    return 0.65 * expectancy + 0.035 * math.log(profit_factor)


def _selection_attribution(
    attribution: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = SEGMENTS[0][1]
    end = SEGMENTS[-1][2]
    return [
        trade
        for trade in attribution
        if start <= str(trade.get("entry_time", ""))[:10] <= end
    ]


def fixed_atlas_recall(
    metrics: dict[str, Any],
    control_metrics: dict[str, Any],
) -> float:
    """Recall on the control's frozen opportunity-atlas denominator."""

    denominator = float(control_metrics.get("entry_oracle_potential_r", 0.0) or 0.0)
    if denominator <= 0.0:
        return float(metrics.get("entry_opportunity_recall", 0.0) or 0.0)
    selected_potential = float(metrics.get("entry_potential_total_r", 0.0) or 0.0)
    return min(max(selected_potential / denominator, 0.0), 1.0)


def score_candidate(
    metrics: dict[str, Any],
    attribution: list[dict[str, Any]],
    control_metrics: dict[str, Any],
    control_attribution: list[dict[str, Any]],
) -> tuple[float, dict[str, float], dict[str, float], dict[str, Any]]:
    """Return score, scaled components, raw values and audit diagnostics."""

    attribution = _selection_attribution(attribution)
    control_attribution = _selection_attribution(control_attribution)
    candidate_segments = _segment_r_per_month(attribution)
    control_segments = _segment_r_per_month(control_attribution)
    segment_deltas = {
        name: candidate_segments[name] - control_segments[name]
        for name, _, _ in SEGMENTS
    }
    candidate_issuer = issuer_diagnostics(attribution)
    control_issuer = issuer_diagnostics(control_attribution)
    candidate_sector = sector_diagnostics(attribution)
    control_sector = sector_diagnostics(control_attribution)
    candidate_economics = _selection_economics(attribution)
    control_economics = _selection_economics(control_attribution)
    candidate_months = SELECTION_MONTHS
    control_months = SELECTION_MONTHS
    candidate_rpm = candidate_economics["total_r"] / candidate_months
    control_rpm = control_economics["total_r"] / control_months
    candidate_tpm = candidate_economics["trades"] / candidate_months
    control_tpm = control_economics["trades"] / control_months
    candidate_breadth = _positive_sleeve_count(attribution)
    control_breadth = _positive_sleeve_count(control_attribution)
    # Cannibalisation must be attributed to selection-window executions.  A
    # run-level metric can include the locked interval and would create a
    # subtle validation leak even though the ordinary trade rows are filtered.
    extra_cannibalization = max(
        sum(float(row.get("cannibalized_r", 0.0) or 0.0) for row in attribution)
        - sum(
            float(row.get("cannibalized_r", 0.0) or 0.0)
            for row in control_attribution
        ),
        0.0,
    )
    issuer_neutral_delta_r_per_month = (
        float(candidate_issuer["issuer_neutral_total_r"]) / candidate_months
        - float(control_issuer["issuer_neutral_total_r"]) / control_months
    )
    issuer_share_improvement = (
        float(control_issuer["top_positive_issuer_share"])
        - float(candidate_issuer["top_positive_issuer_share"])
    )
    sector_share_improvement = (
        float(control_sector["top_positive_sector_share"])
        - float(candidate_sector["top_positive_sector_share"])
    )
    raw = {
        "net_expected_r_per_month": candidate_rpm - control_rpm,
        "executable_trades_per_month": (
            _capacity_utility(candidate_tpm) - _capacity_utility(control_tpm)
        ),
        "worst_fold_r_per_month": min(segment_deltas.values(), default=0.0),
        "average_r_and_discrimination": _quality(candidate_economics)
        - _quality(control_economics),
        "downside_risk": (
            0.70
            * (
                control_economics["max_drawdown_r"]
                - candidate_economics["max_drawdown_r"]
            )
            / 2.0
            + 0.30
            * (
                candidate_economics["expected_shortfall_r"]
                - control_economics["expected_shortfall_r"]
            )
            / 0.25
        ),
        "issuer_sector_concentration": (
            0.50 * issuer_neutral_delta_r_per_month / 2.0
            + 0.30 * issuer_share_improvement / 0.10
            + 0.20 * sector_share_improvement / 0.15
        ),
        "cost_and_neighbourhood_robustness": (
            _robustness_value(attribution)
            - _robustness_value(control_attribution)
            - extra_cannibalization / max(5.0 * candidate_months, 1.0)
        ),
    }
    components = {
        name: 0.5 + 0.5 * math.tanh(raw[name] / spec["scale"])
        for name, spec in SCORE_SPEC.items()
    }
    score = sum(SCORE_SPEC[name]["weight"] * components[name] for name in SCORE_SPEC)
    audit = {
        "candidate_segments_r": candidate_segments,
        "control_segments_r": control_segments,
        "segment_delta_r_per_month": segment_deltas,
        "candidate_issuer": candidate_issuer,
        "control_issuer": control_issuer,
        "candidate_sector": candidate_sector,
        "control_sector": control_sector,
        "candidate_selection_economics": candidate_economics,
        "control_selection_economics": control_economics,
        "economic_rates": {
            "candidate_r_per_month": candidate_rpm,
            "control_r_per_month": control_rpm,
            "candidate_trades_per_month": candidate_tpm,
            "control_trades_per_month": control_tpm,
            "executable_capacity_trades_per_month": EXECUTABLE_CAPACITY_TRADES_PER_MONTH,
        },
        "breadth": {
            "candidate_independently_positive_sleeves": candidate_breadth,
            "control_independently_positive_sleeves": control_breadth,
            "extra_cannibalization_r": extra_cannibalization,
        },
        "fixed_atlas": {
            "oracle_potential_r": float(
                control_metrics.get("entry_oracle_potential_r", 0.0) or 0.0
            ),
            "candidate_recall": fixed_atlas_recall(metrics, control_metrics),
            "control_recall": fixed_atlas_recall(control_metrics, control_metrics),
        },
    }
    return float(score), components, raw, audit
