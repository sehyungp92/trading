"""Executable Phase 4-7 helpers for the IARIC residual programme."""
from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import date, datetime
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CALIBRATION_START,
    DISCOVERY_END,
    DISCOVERY_START,
)
from backtests.stock.auto.runners.run_iaric_daily_residual_discovery import SCORE_SPEC
from backtests.stock.engine.iaric_daily_residual_replay import (
    DailyResidualReplayBundle,
    DailyResidualReplayResult,
    run_daily_residual_replay,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core.lanes import issuer_key


FOLDS = (
    ("discovery", date.fromisoformat(DISCOVERY_START), date.fromisoformat(DISCOVERY_END)),
    ("calibration", date.fromisoformat(CALIBRATION_START), date.fromisoformat(CALIBRATION_END)),
)

# Round 1's legacy score placed 76% of its weight on already-saturated
# transforms.  These fixed centres are economic reference points, not sample
# extrema, so a candidate must improve alpha, breadth or risk to move the score.
# Every component is available on every exact candidate; no neutral placeholder
# can confer a ranking advantage.  The executable signal score remains subject
# to the separate one-to-seven component ceiling in StrategySettings.
ROUND2_SCORE_SPEC: dict[str, dict[str, float]] = {
    "net_expected_r_per_month": {"weight": 0.24, "center": 5.20, "scale": 2.0},
    "executable_trades_per_month": {"weight": 0.14, "center": 21.0, "scale": 5.0},
    "worst_fold_r_per_month": {"weight": 0.16, "center": 2.25, "scale": 1.25},
    "score_discrimination": {"weight": 0.16, "center": 0.10, "scale": 0.15},
    "downside_quality": {"weight": 0.15, "center": 0.0, "scale": 1.0},
    # One-sided 95% winsorisation preserves the ordinary right tail of a
    # positively skewed reversion payoff while preventing a few extreme wins
    # from dominating the score.  Deleting that tail entirely biases the
    # expected-return estimate downward and creates an unstable pass/fail cliff.
    "winner_robust_breadth": {"weight": 0.08, "center": 1.0, "scale": 1.5},
    "issuer_sector_concentration": {"weight": 0.07, "center": 0.75, "scale": 0.15},
}
if len(ROUND2_SCORE_SPEC) != 7 or not math.isclose(
    sum(row["weight"] for row in ROUND2_SCORE_SPEC.values()), 1.0
):
    raise RuntimeError("Round 2 exact score must have seven fixed components")


def _months(start: date, end: date) -> float:
    return max((end - start).days / 30.4375, 1.0)


def _winner_robustness(
    values: Iterable[float],
    *,
    start: date,
    end: date,
) -> dict[str, float]:
    """Return right-tail-robust breadth diagnostics for one selection fold.

    Removing the best five percent of trades is a deliberately downward-biased
    estimator for a long reversion strategy whose legitimate payoff is right
    skewed.  Cap that tail at its empirical 95th percentile instead, then
    independently report median breadth and the tail's share of gross positive
    R.  The three diagnostics distinguish broad alpha from a few lottery wins
    without requiring a symmetric or normally distributed trade outcome.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "median_r_multiple": 0.0,
            "top_5pct_winner_winsorized_r_per_month": 0.0,
            "top_5pct_positive_r_share": 1.0,
        }
    tail_count = max(1, math.ceil(len(ordered) * 0.05))
    upper_cutoff = float(np.quantile(np.asarray(ordered, dtype=float), 0.95))
    winsorized = sum(min(value, upper_cutoff) for value in ordered)
    gross_positive_r = sum(max(value, 0.0) for value in ordered)
    top_tail_positive_r = sum(max(value, 0.0) for value in ordered[-tail_count:])
    return {
        "median_r_multiple": float(np.median(np.asarray(ordered, dtype=float))),
        "top_5pct_winner_winsorized_r_per_month": (
            winsorized / _months(start, end)
        ),
        "top_5pct_positive_r_share": (
            top_tail_positive_r / gross_positive_r
            if gross_positive_r > 0.0
            else 1.0
        ),
    }


def _winner_robustness_passes(folds: Mapping[str, Mapping[str, Any]]) -> bool:
    """Require broad positive alpha without assuming a thin-tailed payoff."""

    return all(
        float(row["top_5pct_winner_winsorized_r_per_month"]) > 0.0
        and float(row["median_r_multiple"]) > 0.0
        and float(row["top_5pct_positive_r_share"]) <= 0.50
        for row in folds.values()
    )


def settings_from_discovery_candidate(
    candidate: Mapping[str, Any],
    *,
    management: Mapping[str, Any] | None = None,
) -> StrategySettings:
    management = dict(management or {})
    holding = int(management.get("maximum_holding_sessions", candidate["holding_sessions"]))
    return StrategySettings(
        strategy_mode="daily_residual_reversion",
        daily_residual_factor_model=str(candidate["factor_model"]),
        daily_residual_formation_sessions=int(candidate["formation_sessions"]),
        daily_residual_minimum_z=float(candidate["residual_z_floor"]),
        daily_residual_minimum_score=float(candidate.get("minimum_score", 0.0)),
        daily_residual_minimum_failed_continuation_r=float(
            candidate.get("minimum_failed_continuation_r", 0.0)
        ),
        daily_residual_lane_id=str(
            candidate.get("lane_id", "daily_residual_generic")
        ),
        daily_residual_minimum_sector_return_5d=float(
            candidate.get("minimum_sector_return_5d", -0.15)
        ),
        daily_residual_minimum_market_trend_z_20d=float(
            candidate.get("minimum_market_trend_z_20d", -8.0)
        ),
        daily_residual_score_components=tuple(candidate["score_components"]),
        daily_residual_ranking_score_components=tuple(
            candidate.get("ranking_score_components", ())
        ),
        daily_residual_max_positions=int(candidate["max_positions"]),
        daily_residual_max_positions_per_sector=int(
            candidate["max_positions_per_sector"]
        ),
        daily_residual_risk_fraction=0.0035,
        daily_residual_maximum_notional_fraction=0.10,
        daily_residual_catastrophic_stop_atr=float(
            management.get("catastrophic_stop_atr", 2.5)
        ),
        daily_residual_catastrophic_stop_residual_r=float(
            management.get(
                "catastrophic_stop_residual_r",
                candidate.get("catastrophic_stop_residual_r", 4.0),
            )
        ),
        daily_residual_partial_normalization_fraction=float(
            management.get("partial_normalization_fraction", 99.0)
        ),
        daily_residual_full_normalization_fraction=float(
            management.get("full_normalization_fraction", 99.0)
        ),
        daily_residual_structural_failure_extension_fraction=float(
            management.get("structural_failure_extension_fraction", 99.0)
        ),
        daily_residual_profit_retention_activation_fraction=float(
            management.get("profit_retention_activation_fraction", 99.0)
        ),
        daily_residual_profit_retention_giveback_fraction=float(
            management.get("profit_retention_giveback_fraction", 99.0)
        ),
        daily_residual_maximum_holding_sessions=holding,
        daily_residual_partial_exit_fraction=float(
            management.get("partial_exit_fraction", 0.0)
        ),
    )


def _fold_payload(result: DailyResidualReplayResult, start: date, end: date) -> dict[str, Any]:
    metrics = result.metrics()
    trades = result.trades
    positive = [trade for trade in trades if trade.r_multiple > 0.0]
    issuer_positive: defaultdict[str, float] = defaultdict(float)
    sector_positive: defaultdict[str, float] = defaultdict(float)
    issuer_entry_risk: defaultdict[str, float] = defaultdict(float)
    sector_entry_risk: defaultdict[str, float] = defaultdict(float)
    total_entry_risk = 0.0
    for trade in trades:
        risk = float(trade.initial_risk_dollars)
        total_entry_risk += risk
        issuer_entry_risk[issuer_key(trade.symbol)] += risk
        sector_entry_risk[trade.sector] += risk
    for trade in positive:
        issuer_positive[issuer_key(trade.symbol)] += trade.r_multiple
        sector_positive[trade.sector] += trade.r_multiple
    total_positive = sum(issuer_positive.values())
    sector_total = sum(sector_positive.values())
    values = sorted(trade.r_multiple for trade in trades)
    overnight = [trade.overnight_return for trade in trades]
    post_open = [trade.open_to_exit_return for trade in trades]
    full_path = [trade.signal_close_to_exit_return for trade in trades]
    tail_count = max(1, math.ceil(len(values) * 0.05)) if values else 0
    top_winner_trimmed = sorted(
        (float(trade.r_multiple) for trade in trades), reverse=True
    )[tail_count:]
    winner_robustness = _winner_robustness(values, start=start, end=end)
    return {
        **metrics,
        **winner_robustness,
        "r_per_month": float(metrics["total_r"]) / _months(start, end),
        "trades_per_month": int(metrics["trades"]) / _months(start, end),
        "expected_shortfall_r_5pct": (
            fmean(values[:tail_count]) if tail_count else 0.0
        ),
        "top_5pct_winner_trimmed_r_per_month": (
            sum(top_winner_trimmed) / _months(start, end)
            if top_winner_trimmed
            else 0.0
        ),
        "top_positive_issuer_share": (
            max(issuer_positive.values(), default=0.0) / total_positive
            if total_positive > 0.0
            else 1.0
        ),
        "top_positive_sector_share": (
            max(sector_positive.values(), default=0.0) / sector_total
            if sector_total > 0.0
            else 1.0
        ),
        "positive_sectors": sum(value > 0.0 for value in sector_positive.values()),
        "top_issuer_entry_risk_share": (
            max(issuer_entry_risk.values(), default=0.0) / total_entry_risk
            if total_entry_risk > 0.0
            else 1.0
        ),
        "top_sector_entry_risk_share": (
            max(sector_entry_risk.values(), default=0.0) / total_entry_risk
            if total_entry_risk > 0.0
            else 1.0
        ),
        "entry_delivery_attribution": {
            "average_signal_close_to_open_return": fmean(overnight) if overnight else 0.0,
            "average_open_to_exit_return": fmean(post_open) if post_open else 0.0,
            "average_signal_close_to_exit_return": fmean(full_path) if full_path else 0.0,
            "positive_post_open_share": (
                sum(value > 0.0 for value in post_open) / len(post_open)
                if post_open
                else 0.0
            ),
        },
    }


def _quintiles(trades) -> dict[str, Any]:
    if len(trades) < 25:
        return {"passed": False, "reason": "fewer_than_25_trades", "values": {}}
    frame = pd.DataFrame(
        {"score": [trade.score for trade in trades], "r": [trade.r_multiple for trade in trades]}
    )
    frame["quintile"] = pd.qcut(
        frame["score"].rank(method="first"), 5, labels=False
    )
    values = {
        f"Q{int(index) + 1}": float(group["r"].mean())
        for index, group in frame.groupby("quintile")
    }
    passed = (
        values.get("Q5", -math.inf) > values.get("Q3", math.inf)
        and values.get("Q5", -math.inf) > values.get("Q1", math.inf) + 0.03
    )
    return {"passed": passed, "values": values}


def _immutable_score(
    folds: Mapping[str, Mapping[str, Any]],
    quintiles: Mapping[str, Any],
    *,
    continuous_metrics: Mapping[str, Any],
    restart_stability: float | None = None,
) -> dict[str, Any]:
    # Expected return and frequency must come from the one economically
    # executable portfolio path.  Summing independently capital-reset folds can
    # reward an artificial boundary liquidation and a fresh capacity stack.
    combined_trades = float(continuous_metrics["trades"])
    combined_r = float(continuous_metrics["total_r"])
    combined_months = sum(_months(start, end) for _name, start, end in FOLDS)
    average_r = combined_r / combined_trades if combined_trades else 0.0
    discrimination_lifts = []
    for row in quintiles.values():
        values = row.get("values", {})
        discrimination_lifts.append(
            float(values.get("Q5", 0.0)) - float(values.get("Q1", 0.0))
        )
    concentration = min(
        1.0
        - 0.5 * float(row["top_issuer_entry_risk_share"])
        - 0.5 * float(row["top_sector_entry_risk_share"])
        for row in folds.values()
    )
    worst_drawdown = max(
        float(continuous_metrics["max_drawdown_pct"]),
        *(float(row["max_drawdown_pct"]) for row in folds.values()),
    )
    worst_es = min(float(row["expected_shortfall_r_5pct"]) for row in folds.values())
    raw = {
        "net_expected_r_per_month": combined_r / combined_months,
        "executable_trades_per_month": combined_trades / combined_months,
        "worst_fold_r_per_month": min(float(row["r_per_month"]) for row in folds.values()),
        "average_r_and_discrimination": (
            0.60 * average_r
            + 0.40 * (min(discrimination_lifts) if discrimination_lifts else 0.0)
        ),
        "downside_risk": -(0.70 * worst_drawdown / 0.08 + 0.30 * abs(min(worst_es, 0.0)) / 0.75),
        "issuer_sector_concentration": max(min(concentration, 1.0), 0.0),
        # Exact finalist runs may add the independent-restart sensitivity
        # diagnostic.  Preliminary candidate screens retain the neutral prior
        # so they require only one continuous replay per candidate.
        "cost_and_neighbourhood_robustness": (
            0.50
            if restart_stability is None
            else 0.25 + 0.50 * min(max(float(restart_stability), 0.0), 1.0)
        ),
    }
    scaled = {
        "net_expected_r_per_month": 0.5
        + 0.5
        * math.tanh(
            raw["net_expected_r_per_month"]
            / SCORE_SPEC["net_expected_r_per_month"]["scale"]
        ),
        "executable_trades_per_month": min(
            max(
                raw["executable_trades_per_month"]
                / SCORE_SPEC["executable_trades_per_month"]["scale"],
                0.0,
            ),
            1.0,
        ),
        "worst_fold_r_per_month": 0.5
        + 0.5
        * math.tanh(
            raw["worst_fold_r_per_month"]
            / SCORE_SPEC["worst_fold_r_per_month"]["scale"]
        ),
        "average_r_and_discrimination": 0.5 + 0.5 * math.tanh(raw["average_r_and_discrimination"] / 0.10),
        "downside_risk": min(max(math.exp(min(raw["downside_risk"], 0.0)), 0.0), 1.0),
        "issuer_sector_concentration": raw["issuer_sector_concentration"],
        "cost_and_neighbourhood_robustness": raw["cost_and_neighbourhood_robustness"],
    }
    score = sum(SCORE_SPEC[name]["weight"] * scaled[name] for name in SCORE_SPEC)
    return {"score": score, "raw": raw, "scaled": scaled, "spec": SCORE_SPEC}


def _round2_immutable_score(
    folds: Mapping[str, Mapping[str, Any]],
    quintiles: Mapping[str, Any],
    *,
    continuous_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the pre-registered, non-saturated Round 2 exact score."""

    combined_months = sum(_months(start, end) for _name, start, end in FOLDS)
    lifts = []
    for row in quintiles.values():
        values = row.get("values", {})
        q5 = float(values.get("Q5", 0.0))
        lifts.append(min(q5 - float(values.get("Q3", 0.0)), q5 - float(values.get("Q1", 0.0))))
    worst_drawdown = max(
        float(continuous_metrics["max_drawdown_pct"]),
        *(float(row["max_drawdown_pct"]) for row in folds.values()),
    )
    worst_es = max(abs(min(float(row["expected_shortfall_r_5pct"]), 0.0)) for row in folds.values())
    concentration = min(
        1.0
        - 0.5 * float(row["top_issuer_entry_risk_share"])
        - 0.5 * float(row["top_sector_entry_risk_share"])
        for row in folds.values()
    )
    raw = {
        "net_expected_r_per_month": float(continuous_metrics["total_r"]) / combined_months,
        "executable_trades_per_month": float(continuous_metrics["trades"]) / combined_months,
        "worst_fold_r_per_month": min(float(row["r_per_month"]) for row in folds.values()),
        "score_discrimination": min(lifts) if lifts else -1.0,
        # This raw term is reported in economically interpretable standardized
        # deltas.  Its final scaled value is the weighted DD/ES transform below.
        "downside_quality": (
            0.70 * (0.15 - worst_drawdown) / 0.04
            + 0.30 * (2.75 - worst_es) / 0.75
        ),
        "winner_robust_breadth": min(
            float(row["top_5pct_winner_winsorized_r_per_month"])
            for row in folds.values()
        ),
        "issuer_sector_concentration": max(min(concentration, 1.0), 0.0),
    }
    scaled = {
        name: 0.5
        + 0.5
        * math.tanh(
            (raw[name] - specification["center"]) / specification["scale"]
        )
        for name, specification in ROUND2_SCORE_SPEC.items()
    }
    scaled["downside_quality"] = (
        0.70 * (0.5 + 0.5 * math.tanh((0.15 - worst_drawdown) / 0.04))
        + 0.30 * (0.5 + 0.5 * math.tanh((2.75 - worst_es) / 0.75))
    )
    if float(continuous_metrics["trades"]) <= 0.0:
        scaled["executable_trades_per_month"] = 0.0
    score = sum(
        ROUND2_SCORE_SPEC[name]["weight"] * scaled[name]
        for name in ROUND2_SCORE_SPEC
    )
    return {
        "contract": "iaric_round2_non_saturated_exact_v2",
        "score": score,
        "raw": raw,
        "scaled": scaled,
        "spec": ROUND2_SCORE_SPEC,
    }


def _continuous_fold_result(
    result: DailyResidualReplayResult,
    *,
    start: date,
    end: date,
) -> tuple[DailyResidualReplayResult, int, int]:
    """Return a purged entry cohort on the unbroken portfolio path.

    A discovery entry whose exit occurs after the discovery boundary is omitted
    from discovery score/discrimination metrics so calibration outcomes cannot
    leak backwards.  The position itself remains in the continuous portfolio,
    consumes capacity, and contributes to period equity exactly as it would in
    production.
    """

    curve = [
        row
        for row in result.equity_curve
        if start <= date.fromisoformat(str(row["date"])) <= end
    ]
    prior_rows = [
        row
        for row in result.equity_curve
        if date.fromisoformat(str(row["date"])) < start
    ]
    initial_equity = (
        float(prior_rows[-1]["mtm_equity"])
        if prior_rows
        else float(result.initial_equity)
    )
    final_equity = (
        float(curve[-1]["mtm_equity"]) if curve else initial_equity
    )
    entered = [
        trade for trade in result.trades if start <= trade.entry_date <= end
    ]
    cohort = [
        trade
        for trade in entered
        if trade.exit_date is not None and trade.exit_date <= end
    ]
    events = []
    for event in result.decision_events:
        value = event.get("ts")
        event_date = value.date() if isinstance(value, datetime) else date.fromisoformat(str(value)[:10])
        if start <= event_date <= end:
            events.append(event)
    return (
        DailyResidualReplayResult(
            initial_equity=initial_equity,
            final_equity=final_equity,
            trades=cohort,
            equity_curve=curve,
            decision_events=events,
            source_fingerprint=result.source_fingerprint,
            factor_model=result.factor_model,
            entry_clock=result.entry_clock,
            shared_core_contract=result.shared_core_contract,
        ),
        len(entered) - len(cohort),
        int(prior_rows[-1]["open_positions"]) if prior_rows else 0,
    )


def _replacement_event_diagnostics(
    results: Mapping[str, DailyResidualReplayResult],
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    for fold, result in results.items():
        events = []
        for event in result.decision_events:
            if event.get("code") != "RESIDUAL_MANAGEMENT_EXIT":
                continue
            details = dict(event.get("details", {}) or {})
            reason = str(details.get("reason", ""))
            if not reason.startswith("capacity_neutral_alpha_replacement"):
                continue
            parts = {
                key: value
                for token in reason.split("|")[1:]
                if "=" in token
                for key, value in (token.split("=", 1),)
            }
            events.append(
                {
                    "incumbent_symbol": str(event.get("symbol", "")),
                    "candidate_symbol": parts.get("candidate", ""),
                    "blocker_kind": parts.get("blocker", ""),
                    "decision_ts": event.get("ts"),
                }
            )
        folds[fold] = {
            "changed_decisions": len(events),
            "sector_capacity_replacements": sum(
                row["blocker_kind"] == "sector_capacity" for row in events
            ),
            "portfolio_capacity_replacements": sum(
                row["blocker_kind"] == "portfolio_capacity" for row in events
            ),
            "events": events,
        }
    return {
        "contract": "actual_shared_core_replacement_decisions_not_shadow_pnl_v1",
        "folds": folds,
        "total_changed_decisions": sum(
            int(row["changed_decisions"]) for row in folds.values()
        ),
    }


def _restart_reconciliation(
    continuous: DailyResidualReplayResult,
    resets: Mapping[str, DailyResidualReplayResult],
) -> dict[str, Any]:
    reset_trades = [
        trade for name, _start, _end in FOLDS for trade in resets[name].trades
    ]
    continuous_by_key = {
        (trade.symbol, trade.entry_date): trade for trade in continuous.trades
    }
    reset_by_key = {
        (trade.symbol, trade.entry_date): trade for trade in reset_trades
    }
    common = set(continuous_by_key) & set(reset_by_key)
    continuous_unique = set(continuous_by_key) - set(reset_by_key)
    reset_unique = set(reset_by_key) - set(continuous_by_key)
    continuous_r = sum(trade.r_multiple for trade in continuous.trades)
    reset_r = sum(trade.r_multiple for trade in reset_trades)
    gap = reset_r - continuous_r
    stability = math.exp(-abs(gap) / max(abs(continuous_r), 20.0))
    return {
        "status": "secondary_non_economic_restart_sensitivity_only",
        "used_for_candidate_expected_return_or_frequency": False,
        "continuous_trades": len(continuous.trades),
        "continuous_total_r": continuous_r,
        "independently_reset_trades": len(reset_trades),
        "independently_reset_total_r": reset_r,
        "reset_minus_continuous_r": gap,
        "restart_stability": stability,
        "common_entry_trades": len(common),
        "common_reset_minus_continuous_r": sum(
            reset_by_key[key].r_multiple - continuous_by_key[key].r_multiple
            for key in common
        ),
        "continuous_unique_trades": len(continuous_unique),
        "continuous_unique_total_r": sum(
            continuous_by_key[key].r_multiple for key in continuous_unique
        ),
        "reset_unique_trades": len(reset_unique),
        "reset_unique_total_r": sum(
            reset_by_key[key].r_multiple for key in reset_unique
        ),
        "intermediate_forced_liquidations": sum(
            trade.exit_reason == "fold_end_marked_liquidation"
            for trade in resets["discovery"].trades
        ),
    }


def run_exact_fold_evaluation(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    round_trip_cost_bps: float = 20.0,
    include_independent_restart_stress: bool = False,
    continuous_result: DailyResidualReplayResult | None = None,
    score_contract: str = "legacy",
) -> dict[str, Any]:
    continuous = continuous_result or run_daily_residual_replay(
        bundle,
        settings,
        start=FOLDS[0][1],
        end=FOLDS[-1][2],
        round_trip_cost_bps=round_trip_cost_bps,
    )
    if continuous.factor_model != bundle.factor_model:
        raise ValueError("continuous result factor model does not match replay bundle")
    if continuous.source_fingerprint != bundle.source_fingerprint:
        raise ValueError("continuous result source fingerprint does not match replay bundle")
    results: dict[str, DailyResidualReplayResult] = {}
    folds: dict[str, Any] = {}
    quintiles: dict[str, Any] = {}
    boundary: dict[str, Any] = {}
    for name, start, end in FOLDS:
        result, purged, carried_in = _continuous_fold_result(
            continuous,
            start=start,
            end=end,
        )
        results[name] = result
        folds[name] = _fold_payload(result, start, end)
        folds[name].update(
            {
                "portfolio_state_contract": "continuous_shared_capital_no_fold_reset",
                "entry_cohort_contract": "entry_and_exit_within_fold_purged_boundary",
                "purged_boundary_entry_count": purged,
                "carried_in_positions": carried_in,
            }
        )
        quintiles[name] = _quintiles(result.trades)
        boundary[name] = {
            "purged_boundary_entry_count": purged,
            "carried_in_positions": carried_in,
        }

    reset_results: dict[str, DailyResidualReplayResult] = {}
    restart = None
    if include_independent_restart_stress:
        for name, start, end in FOLDS:
            reset_results[name] = run_daily_residual_replay(
                bundle,
                settings,
                start=start,
                end=end,
                round_trip_cost_bps=round_trip_cost_bps,
            )
        restart = _restart_reconciliation(continuous, reset_results)
    continuous_metrics = continuous.metrics()
    if score_contract == "round2":
        immutable = _round2_immutable_score(
            folds, quintiles, continuous_metrics=continuous_metrics
        )
    elif score_contract == "legacy":
        immutable = _immutable_score(
            folds,
            quintiles,
            continuous_metrics=continuous_metrics,
            restart_stability=(
                float(restart["restart_stability"]) if restart is not None else None
            ),
        )
    else:
        raise ValueError("score_contract must be 'legacy' or 'round2'")
    gates = {
        "positive_each_fold": all(float(row["total_r"]) > 0.0 for row in folds.values()),
        "positive_continuous_period": float(continuous_metrics["total_r"]) > 0.0,
        "positive_continuous_calibration_equity_return": (
            float(folds["calibration"]["return_pct"]) > 0.0
        ),
        "calibration_pf_gte_1p15": float(folds["calibration"]["profit_factor"]) >= 1.15,
        "calibration_average_r_gte_0p07": float(folds["calibration"]["average_r"]) >= 0.07,
        "at_least_100_trades_each_fold": all(int(row["trades"]) >= 100 for row in folds.values()),
        "score_top_tail_separation_each_fold": all(
            bool(row["passed"]) for row in quintiles.values()
        ),
        "top_issuer_entry_risk_share_lte_15pct": all(
            float(row["top_issuer_entry_risk_share"]) <= 0.15
            for row in folds.values()
        ),
        "top_sector_entry_risk_share_lte_35pct": all(
            float(row["top_sector_entry_risk_share"]) <= 0.35
            for row in folds.values()
        ),
    }
    if score_contract == "round2":
        combined_months = sum(_months(start, end) for _name, start, end in FOLDS)
        gates.update(
            {
                "continuous_max_drawdown_lte_18pct": (
                    float(continuous_metrics["max_drawdown_pct"]) <= 0.18
                ),
                "executable_trades_per_month_gte_17p5": (
                    float(continuous_metrics["trades"]) / combined_months >= 17.5
                ),
                "expected_shortfall_no_worse_than_minus_3p70r": all(
                    float(row["expected_shortfall_r_5pct"]) >= -3.70
                    for row in folds.values()
                ),
                "round2_score_discrimination_each_fold": all(
                    float(row.get("values", {}).get("Q5", -math.inf))
                    - float(row.get("values", {}).get("Q3", math.inf)) >= 0.05
                    and float(row.get("values", {}).get("Q5", -math.inf))
                    - float(row.get("values", {}).get("Q1", math.inf)) >= 0.15
                    for row in quintiles.values()
                ),
                "winner_robust_breadth_each_fold": _winner_robustness_passes(
                    folds
                ),
            }
        )
    return {
        "settings": {
            "factor_model": settings.daily_residual_factor_model,
            "formation_sessions": settings.daily_residual_formation_sessions,
            "minimum_z": settings.daily_residual_minimum_z,
            "minimum_score": settings.daily_residual_minimum_score,
            "minimum_failed_continuation_r": (
                settings.daily_residual_minimum_failed_continuation_r
            ),
            "lane_id": settings.daily_residual_lane_id,
            "minimum_sector_return_5d": (
                settings.daily_residual_minimum_sector_return_5d
            ),
            "minimum_market_trend_z_20d": (
                settings.daily_residual_minimum_market_trend_z_20d
            ),
            "score_components": list(settings.daily_residual_score_components),
            "ranking_score_components": list(
                settings.daily_residual_ranking_score_components
            ),
            "max_positions": settings.daily_residual_max_positions,
            "max_positions_per_sector": settings.daily_residual_max_positions_per_sector,
            "sector_overflow_slots": settings.daily_residual_sector_overflow_slots,
            "sector_overflow_minimum_score": (
                settings.daily_residual_sector_overflow_minimum_score
            ),
            "sector_overflow_minimum_z": (
                settings.daily_residual_sector_overflow_minimum_z
            ),
            "sector_overflow_risk_multiplier": (
                settings.daily_residual_sector_overflow_risk_multiplier
            ),
            "risk_fraction": settings.daily_residual_risk_fraction,
            "maximum_notional_fraction": (
                settings.daily_residual_maximum_notional_fraction
            ),
            "partial_normalization_fraction": settings.daily_residual_partial_normalization_fraction,
            "catastrophic_stop_atr": settings.daily_residual_catastrophic_stop_atr,
            "catastrophic_stop_residual_r": (
                settings.daily_residual_catastrophic_stop_residual_r
            ),
            "full_normalization_fraction": settings.daily_residual_full_normalization_fraction,
            "structural_failure_extension_fraction": settings.daily_residual_structural_failure_extension_fraction,
            "profit_retention_activation_fraction": (
                settings.daily_residual_profit_retention_activation_fraction
            ),
            "profit_retention_giveback_fraction": (
                settings.daily_residual_profit_retention_giveback_fraction
            ),
            "replacement_mode": settings.daily_residual_replacement_mode,
            "replacement_loss_only": settings.daily_residual_replacement_loss_only,
            "replacement_minimum_held_sessions": (
                settings.daily_residual_replacement_minimum_held_sessions
            ),
            "replacement_maximum_normalization_fraction": (
                settings.daily_residual_replacement_maximum_normalization_fraction
            ),
            "replacement_minimum_score_margin": (
                settings.daily_residual_replacement_minimum_score_margin
            ),
            "replacement_max_per_session": (
                settings.daily_residual_replacement_max_per_session
            ),
            "maximum_holding_sessions": settings.daily_residual_maximum_holding_sessions,
            "partial_exit_fraction": settings.daily_residual_partial_exit_fraction,
        },
        "round_trip_cost_bps": round_trip_cost_bps,
        "evaluation_contract": "continuous_shared_capital_with_purged_entry_cohorts_v1",
        "continuous_metrics": continuous_metrics,
        "folds": folds,
        "fold_boundary_diagnostics": boundary,
        "score_quintiles": quintiles,
        "immutable_score": immutable,
        "gates": gates,
        "research_anchor_eligible": all(gates.values()),
        "trades": {
            name: [asdict(trade) for trade in result.trades]
            for name, result in results.items()
        },
        "equity_curves": {
            name: result.equity_curve for name, result in results.items()
        },
        "decision_event_counts": {
            name: len(result.decision_events) for name, result in results.items()
        },
        "capacity_neutral_replacement_diagnostics": (
            _replacement_event_diagnostics(results)
        ),
        "independent_restart_stress": restart,
        "independent_restart_folds": (
            {
                name: _fold_payload(result, start, end)
                for name, start, end in FOLDS
                for result in (reset_results[name],)
            }
            if reset_results
            else None
        ),
        "independent_restart_trades": (
            {
                name: [asdict(trade) for trade in reset_results[name].trades]
                for name, _start, _end in FOLDS
            }
            if reset_results
            else None
        ),
    }


def management_experiment_registry(anchor_holding: int) -> list[dict[str, Any]]:
    """Bounded, economically interpretable management sequence."""

    horizons = sorted({3, 5, 7, 10, int(anchor_holding)})
    rows = [
        {
            "experiment_id": f"fixed_half_life_{holding}",
            "maximum_holding_sessions": holding,
            "partial_normalization_fraction": 99.0,
            "full_normalization_fraction": 99.0,
            "structural_failure_extension_fraction": 99.0,
            "partial_exit_fraction": 0.0,
            "stage": "half_life",
        }
        for holding in horizons
    ]
    rows.extend(
        [
            {
                "experiment_id": "full_anchor_only",
                "maximum_holding_sessions": anchor_holding,
                "partial_normalization_fraction": 99.0,
                "full_normalization_fraction": 1.0,
                "structural_failure_extension_fraction": 0.50,
                "partial_exit_fraction": 0.0,
                "stage": "typed_management",
            },
            {
                "experiment_id": "half_then_full_anchor",
                "maximum_holding_sessions": anchor_holding,
                "partial_normalization_fraction": 0.50,
                "full_normalization_fraction": 1.0,
                "structural_failure_extension_fraction": 0.50,
                "partial_exit_fraction": 0.50,
                "stage": "typed_management",
            },
            {
                "experiment_id": "late_partial_then_anchor",
                "maximum_holding_sessions": anchor_holding,
                "partial_normalization_fraction": 0.75,
                "full_normalization_fraction": 1.0,
                "structural_failure_extension_fraction": 0.50,
                "partial_exit_fraction": 0.50,
                "stage": "typed_management",
            },
            {
                "experiment_id": "early_persistence_failure",
                "maximum_holding_sessions": anchor_holding,
                "partial_normalization_fraction": 0.50,
                "full_normalization_fraction": 1.0,
                "structural_failure_extension_fraction": 0.35,
                "partial_exit_fraction": 0.50,
                "stage": "typed_management",
            },
        ]
    )
    return rows


def final_qualification(
    base: Mapping[str, Any],
    *,
    thirty_bps: Mapping[str, Any],
    forty_bps: Mapping[str, Any],
    neighbourhood_positive_share: float,
) -> dict[str, Any]:
    folds = base["folds"]
    gates = {
        "minimum_100_trades_each_fold": all(
            int(row["trades"]) >= 100 for row in folds.values()
        ),
        "positive_each_fold_after_20bps": all(float(row["total_r"]) > 0.0 for row in folds.values()),
        "positive_each_fold_after_30bps": all(float(row["total_r"]) > 0.0 for row in thirty_bps["folds"].values()),
        "nonnegative_calibration_after_40bps": float(forty_bps["folds"]["calibration"]["total_r"]) >= 0.0,
        "calibration_pf_gte_1p15": float(folds["calibration"]["profit_factor"]) >= 1.15,
        "calibration_average_r_gte_0p07": float(folds["calibration"]["average_r"]) >= 0.07,
        "score_top_tail_separation_each_fold": all(
            bool(row["passed"]) for row in base["score_quintiles"].values()
        ),
        "top_issuer_entry_risk_share_lte_15pct": all(
            float(row["top_issuer_entry_risk_share"]) <= 0.15
            for row in folds.values()
        ),
        "top_sector_entry_risk_share_lte_35pct": all(
            float(row["top_sector_entry_risk_share"]) <= 0.35
            for row in folds.values()
        ),
        "at_least_four_positive_sectors_each_fold": all(int(row["positive_sectors"]) >= 4 for row in folds.values()),
        "positive_neighbourhood_share_gte_50pct": neighbourhood_positive_share >= 0.50,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "neighbourhood_positive_share": neighbourhood_positive_share,
    }


def _evaluate_management_row(
    bundle: DailyResidualReplayBundle,
    candidate: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    score_contract: str = "legacy",
) -> dict[str, Any]:
    settings = settings_from_discovery_candidate(candidate, management=experiment)
    result = run_exact_fold_evaluation(
        replace(bundle, frozen_history_cache={}),
        settings,
        round_trip_cost_bps=20.0,
        score_contract=score_contract,
    )
    return {
        "experiment": dict(experiment),
        "result": result,
        "selection_eligible": bool(result["research_anchor_eligible"]),
    }


def run_management_phase(
    bundle: DailyResidualReplayBundle,
    candidate: Mapping[str, Any],
    *,
    max_workers: int = 2,
    score_contract: str = "legacy",
) -> dict[str, Any]:
    """Run half-life discovery, freeze it, then test typed exits around it."""

    if max_workers != 2:
        raise ValueError("residual management phase is registered for max-workers=2")
    half_life = [
        row
        for row in management_experiment_registry(int(candidate["holding_sessions"]))
        if row["stage"] == "half_life"
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        half_life_rows = list(
            pool.map(
                lambda row: _evaluate_management_row(
                    bundle, candidate, row, score_contract=score_contract
                ),
                half_life,
            )
        )
    eligible_half_life = [row for row in half_life_rows if row["selection_eligible"]]
    if not eligible_half_life:
        return {
            "status": "blocked_no_exact_half_life_anchor",
            "half_life_experiments": half_life_rows,
            "typed_management_experiments": [],
            "selected": None,
        }
    selected_half_life = max(
        eligible_half_life,
        key=lambda row: (
            float(row["result"]["immutable_score"]["score"]),
            -int(row["experiment"]["maximum_holding_sessions"]),
        ),
    )
    frozen_holding = int(
        selected_half_life["experiment"]["maximum_holding_sessions"]
    )
    typed = [
        row
        for row in management_experiment_registry(frozen_holding)
        if row["stage"] == "typed_management"
    ]
    # The frozen half-life is the literal ablation against every typed exit.
    typed.insert(
        0,
        {
            **selected_half_life["experiment"],
            "experiment_id": "frozen_half_life_control",
            "stage": "typed_management_control",
        },
    )
    typed.extend(
        [
            {
                "experiment_id": "frozen_half_life_residual_stop_5r",
                "maximum_holding_sessions": frozen_holding,
                "partial_normalization_fraction": 99.0,
                "full_normalization_fraction": 99.0,
                "structural_failure_extension_fraction": 99.0,
                "partial_exit_fraction": 0.0,
                "catastrophic_stop_residual_r": 5.0,
                "stage": "typed_management_control",
            },
            {
                "experiment_id": "frozen_half_life_residual_stop_6r",
                "maximum_holding_sessions": frozen_holding,
                "partial_normalization_fraction": 99.0,
                "full_normalization_fraction": 99.0,
                "structural_failure_extension_fraction": 99.0,
                "partial_exit_fraction": 0.0,
                "catastrophic_stop_residual_r": 6.0,
                "stage": "typed_management_control",
            },
        ]
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        typed_rows = list(
            pool.map(
                lambda row: _evaluate_management_row(
                    bundle, candidate, row, score_contract=score_contract
                ),
                typed,
            )
        )
    eligible_typed = [row for row in typed_rows if row["selection_eligible"]]
    if not eligible_typed:
        return {
            "status": "blocked_no_exact_typed_management_anchor",
            "half_life_experiments": half_life_rows,
            "selected_half_life": selected_half_life,
            "typed_management_experiments": typed_rows,
            "selected": None,
        }
    selected = max(
        eligible_typed,
        key=lambda row: (
            float(row["result"]["immutable_score"]["score"]),
            row["experiment"]["experiment_id"] == "frozen_half_life_control",
        ),
    )
    control_score = float(
        next(
            row for row in typed_rows
            if row["experiment"]["experiment_id"] == "frozen_half_life_control"
        )["result"]["immutable_score"]["score"]
    )
    selected_score = float(selected["result"]["immutable_score"]["score"])
    return {
        "status": "passed",
        "selection_order": ["half_life", "typed_management"],
        "half_life_experiments": half_life_rows,
        "selected_half_life": selected_half_life,
        "typed_management_experiments": typed_rows,
        "selected": selected,
        "typed_management_value_add_vs_frozen_half_life": selected_score - control_score,
    }


def _leave_one_out(base: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for fold_name, trades in base["trades"].items():
        total = sum(float(row["r_multiple"]) for row in trades)
        by_sector: defaultdict[str, float] = defaultdict(float)
        by_issuer: defaultdict[str, float] = defaultdict(float)
        for row in trades:
            by_sector[str(row["sector"])] += float(row["r_multiple"])
            by_issuer[issuer_key(str(row["symbol"]))] += float(row["r_multiple"])
        sector_results = {
            key: total - value for key, value in sorted(by_sector.items())
        }
        issuer_results = {
            key: total - value for key, value in sorted(by_issuer.items())
        }
        output[fold_name] = {
            "leave_one_sector_total_r": sector_results,
            "leave_one_issuer_worst_total_r": min(issuer_results.values(), default=0.0),
            "all_leave_one_sector_positive": bool(sector_results)
            and all(value > 0.0 for value in sector_results.values()),
            "all_leave_one_issuer_positive": bool(issuer_results)
            and all(value > 0.0 for value in issuer_results.values()),
        }
    return output


def run_final_robustness_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "legacy",
) -> dict[str, Any]:
    """Run bounded local invariance, cost stress and literal concentration ablations."""

    if max_workers != 2:
        raise ValueError("residual robustness phase is registered for max-workers=2")
    variants = {
        "minimum_z_0p90": replace(settings, daily_residual_minimum_z=0.90),
        "minimum_z_1p10": replace(settings, daily_residual_minimum_z=1.10),
        "failed_continuation_0p10": replace(
            settings, daily_residual_minimum_failed_continuation_r=0.10
        ),
        "failed_continuation_0p30": replace(
            settings, daily_residual_minimum_failed_continuation_r=0.30
        ),
        "score_floor_minus_10": replace(
            settings,
            daily_residual_minimum_score=max(
                0.0, settings.daily_residual_minimum_score - 10.0
            ),
        ),
        "score_floor_plus_10": replace(
            settings,
            daily_residual_minimum_score=min(
                100.0, settings.daily_residual_minimum_score + 10.0
            ),
        ),
        "sector_headwind_veto_m3pct": replace(
            settings, daily_residual_minimum_sector_return_5d=-0.03
        ),
        "position_cap_8": replace(settings, daily_residual_max_positions=8),
        "position_cap_12": replace(settings, daily_residual_max_positions=12),
        "sector_cap_1": replace(settings, daily_residual_max_positions_per_sector=1),
    }

    def evaluate(item: tuple[str, StrategySettings]) -> tuple[str, dict[str, Any]]:
        name, variant = item
        return name, run_exact_fold_evaluation(
            replace(bundle, frozen_history_cache={}),
            variant,
            round_trip_cost_bps=20.0,
            score_contract=score_contract,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        neighbourhood = dict(pool.map(evaluate, variants.items()))
    positive_rows = [
        row
        for row in neighbourhood.values()
        if all(float(fold["total_r"]) > 0.0 for fold in row["folds"].values())
    ]
    positive_share = len(positive_rows) / max(len(neighbourhood), 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        base, thirty, forty = list(
            pool.map(
                lambda cost: run_exact_fold_evaluation(
                    replace(bundle, frozen_history_cache={}),
                    settings,
                    round_trip_cost_bps=cost,
                    score_contract=score_contract,
                ),
                (20.0, 30.0, 40.0),
            )
        )
    leave_one_out = _leave_one_out(base)
    qualification = final_qualification(
        base,
        thirty_bps=thirty,
        forty_bps=forty,
        neighbourhood_positive_share=positive_share,
    )
    qualification["gates"]["all_leave_one_sector_positive"] = all(
        bool(row["all_leave_one_sector_positive"])
        for row in leave_one_out.values()
    )
    qualification["gates"]["all_leave_one_issuer_positive"] = all(
        bool(row["all_leave_one_issuer_positive"])
        for row in leave_one_out.values()
    )
    qualification["passed"] = all(qualification["gates"].values())
    qualification["failed_gates"] = [
        name for name, passed in qualification["gates"].items() if not passed
    ]
    return {
        "status": "passed" if qualification["passed"] else "blocked",
        "base_20bps": base,
        "cost_stress_30bps": thirty,
        "cost_stress_40bps": forty,
        "neighbourhood": neighbourhood,
        "neighbourhood_positive_share": positive_share,
        "leave_one_out": leave_one_out,
        "independent_sleeves": {
            "daily_residual_reversion": {
                "enabled": True,
                "qualified": qualification["passed"],
            },
            "intraday_residual_failed_continuation": {
                "enabled": False,
                "qualified": False,
                "reason": "secondary sleeve cannot share the daily alpha gate and requires separately authoritative five-minute inputs",
            },
            "gap_residual_failed_continuation": {
                "enabled": False,
                "qualified": False,
                "reason": "secondary sleeve cannot share the daily alpha gate and requires separately authoritative five-minute inputs",
            },
        },
        "qualification": qualification,
    }


def _run_settings_frontier(
    bundle: DailyResidualReplayBundle,
    variants: Mapping[str, StrategySettings],
    *,
    max_workers: int,
    score_contract: str,
    maximum_selection_drawdown: float,
    prefer_sub_10pct_drawdown: bool = False,
    inherited_control_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exact-replay a small pre-registered frontier and select without leakage."""

    if max_workers != 2:
        raise ValueError("residual settings frontiers are registered for max-workers=2")
    if "control" not in variants:
        raise ValueError("every settings frontier requires a literal control")

    def evaluate(item: tuple[str, StrategySettings]) -> dict[str, Any]:
        experiment_id, settings = item
        if experiment_id == "control" and inherited_control_result is not None:
            result = dict(inherited_control_result)
            inherited_settings = result.get("settings", {})
            expected = {
                "max_positions": settings.daily_residual_max_positions,
                "max_positions_per_sector": (
                    settings.daily_residual_max_positions_per_sector
                ),
                "minimum_z": settings.daily_residual_minimum_z,
                "minimum_score": settings.daily_residual_minimum_score,
            }
            mismatches = {
                name: (inherited_settings.get(name), value)
                for name, value in expected.items()
                if name in inherited_settings
                and inherited_settings.get(name) != value
            }
            if mismatches:
                raise ValueError(
                    "inherited exact control does not match Phase-8 settings: "
                    f"{mismatches}"
                )
            result_source = "inherited_exact_phase6_position_cap12"
        else:
            result = run_exact_fold_evaluation(
                replace(bundle, frozen_history_cache={}),
                settings,
                round_trip_cost_bps=20.0,
                score_contract=score_contract,
            )
            result_source = "new_exact_replay"
        metrics = result["continuous_metrics"]
        aspirational_targets = {
            "total_r_above_100r": float(metrics["total_r"]) > 100.0,
            "mtm_max_drawdown_below_10pct": (
                float(metrics["max_drawdown_pct"]) < 0.10
            ),
        }
        selection_eligible = bool(result["research_anchor_eligible"]) and (
            float(metrics["max_drawdown_pct"]) <= maximum_selection_drawdown
        )
        # Settings frontiers never consume per-trade or per-session arrays
        # after the exact gates and immutable score have been computed.  Drop
        # them inside each worker instead of retaining several full replays at
        # once; this bounds peak memory without changing any selected metric,
        # gate, score or persisted artifact.
        compact_result = {
            key: value
            for key, value in result.items()
            if key not in {"trades", "equity_curves"}
        }
        return {
            "experiment_id": experiment_id,
            "settings": settings,
            "result": compact_result,
            "selection_eligible": selection_eligible,
            "aspirational_targets": aspirational_targets,
            "result_source": result_source,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(evaluate, variants.items()))
    control = next(row for row in rows if row["experiment_id"] == "control")
    eligible = [row for row in rows if row["selection_eligible"]]
    if not eligible:
        return {
            "status": "blocked_no_exact_eligible_candidate",
            "experiments": rows,
            "selected": None,
            "selected_settings": None,
            "control": control,
            "maximum_selection_drawdown": maximum_selection_drawdown,
            "aspirational_targets_are_hard_gates": False,
        }

    def ranking_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        result = row["result"]
        metrics = result["continuous_metrics"]
        below_10pct_drawdown = bool(
            row["aspirational_targets"]["mtm_max_drawdown_below_10pct"]
        )
        return (
            float(result["immutable_score"]["score"]),
            float(below_10pct_drawdown) if prefer_sub_10pct_drawdown else 0.0,
            float(metrics["total_r"]),
            float(metrics["return_pct"]),
            -float(metrics["max_drawdown_pct"]),
        )

    selected = max(eligible, key=ranking_key)
    control_score = float(control["result"]["immutable_score"]["score"])
    selected_score = float(selected["result"]["immutable_score"]["score"])
    if (
        control["selection_eligible"]
        and selected_score <= control_score + 1e-12
    ):
        selected = control
        selected_score = control_score
        status = "complete_control_retained"
    else:
        status = "passed"
    return {
        "status": status,
        "experiments": rows,
        "selected": selected,
        "selected_settings": selected["settings"],
        "control": control,
        "score_delta_vs_control": selected_score - control_score,
        "maximum_selection_drawdown": maximum_selection_drawdown,
        "aspirational_targets_are_hard_gates": False,
    }


def run_selective_sector_overflow_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
    inherited_control_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Test only the unresolved sector displacement mechanism.

    Phase 6 already proved that the twelve-position global cap dominates the
    ten-position control.  Portfolio-cap displacement was negative in both
    folds, whereas sector-cap displacement was positive in both folds.  This
    phase therefore fixes global capacity at twelve and asks whether one
    exceptional third name from a crowded sector should be admitted with a
    quality gate and, where appropriate, reduced marginal risk.
    """

    if settings.daily_residual_max_positions != 12:
        raise ValueError(
            "selective sector-overflow Phase 8 must inherit the exact "
            "twelve-position Phase-6 neighbour"
        )
    if settings.daily_residual_max_positions_per_sector != 2:
        raise ValueError("selective sector-overflow Phase 8 requires sector cap two")

    variants = {
        "control": settings,
        "overflow_1_score_50_z_1p00_full_risk": replace(
            settings,
            daily_residual_sector_overflow_slots=1,
            daily_residual_sector_overflow_minimum_score=50.0,
            daily_residual_sector_overflow_minimum_z=1.0,
            daily_residual_sector_overflow_risk_multiplier=1.0,
        ),
        "overflow_1_score_50_z_1p00_risk_0p75": replace(
            settings,
            daily_residual_sector_overflow_slots=1,
            daily_residual_sector_overflow_minimum_score=50.0,
            daily_residual_sector_overflow_minimum_z=1.0,
            daily_residual_sector_overflow_risk_multiplier=0.75,
        ),
        "overflow_1_score_60_z_1p00_risk_0p75": replace(
            settings,
            daily_residual_sector_overflow_slots=1,
            daily_residual_sector_overflow_minimum_score=60.0,
            daily_residual_sector_overflow_minimum_z=1.0,
            daily_residual_sector_overflow_risk_multiplier=0.75,
        ),
        "overflow_1_score_50_z_1p10_full_risk": replace(
            settings,
            daily_residual_sector_overflow_slots=1,
            daily_residual_sector_overflow_minimum_score=50.0,
            daily_residual_sector_overflow_minimum_z=1.10,
            daily_residual_sector_overflow_risk_multiplier=1.0,
        ),
        "overflow_1_score_50_z_1p10_risk_0p75": replace(
            settings,
            daily_residual_sector_overflow_slots=1,
            daily_residual_sector_overflow_minimum_score=50.0,
            daily_residual_sector_overflow_minimum_z=1.10,
            daily_residual_sector_overflow_risk_multiplier=0.75,
        ),
        "overflow_1_score_60_z_1p10_risk_0p75": replace(
            settings,
            daily_residual_sector_overflow_slots=1,
            daily_residual_sector_overflow_minimum_score=60.0,
            daily_residual_sector_overflow_minimum_z=1.10,
            daily_residual_sector_overflow_risk_multiplier=0.75,
        ),
        "overflow_1_score_60_z_1p10_risk_0p50": replace(
            settings,
            daily_residual_sector_overflow_slots=1,
            daily_residual_sector_overflow_minimum_score=60.0,
            daily_residual_sector_overflow_minimum_z=1.10,
            daily_residual_sector_overflow_risk_multiplier=0.50,
        ),
    }
    result = _run_settings_frontier(
        bundle,
        variants,
        max_workers=max_workers,
        score_contract=score_contract,
        maximum_selection_drawdown=0.14,
        inherited_control_result=inherited_control_result,
    )
    result["contract"] = "exact_selective_sector_overflow_frontier_v1"
    result["phase6_evidence_inherited"] = {
        "global_position_cap": 12,
        "ordinary_sector_cap": 2,
        "global_capacity_replayed": False,
        "portfolio_capacity_displacement_positive_both_folds": False,
        "sector_capacity_displacement_positive_both_folds": True,
    }
    return result


def run_quality_aperture_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
) -> dict[str, Any]:
    """Test admission aperture without turning the score into a large grid."""

    variants = {
        "control": settings,
        "minimum_z_1p05": replace(settings, daily_residual_minimum_z=1.05),
        "minimum_z_1p10": replace(settings, daily_residual_minimum_z=1.10),
        "score_floor_15": replace(settings, daily_residual_minimum_score=15.0),
        "score_floor_20": replace(settings, daily_residual_minimum_score=20.0),
        "score_floor_30": replace(settings, daily_residual_minimum_score=30.0),
        "z_1p05_score_20": replace(
            settings,
            daily_residual_minimum_z=1.05,
            daily_residual_minimum_score=20.0,
        ),
        "z_1p05_score_25": replace(
            settings,
            daily_residual_minimum_z=1.05,
            daily_residual_minimum_score=25.0,
        ),
    }
    result = _run_settings_frontier(
        bundle,
        variants,
        max_workers=max_workers,
        score_contract=score_contract,
        maximum_selection_drawdown=0.14,
    )
    result["contract"] = "exact_quality_aperture_and_discrimination_frontier_v1"
    return result


def run_risk_notional_frontier_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
) -> dict[str, Any]:
    """Prefer sub-ten-percent MTM DD without turning it into a rejection cliff."""

    variants = {
        "control": settings,
        "risk_0p25pct": replace(settings, daily_residual_risk_fraction=0.0025),
        "risk_0p275pct": replace(settings, daily_residual_risk_fraction=0.00275),
        "risk_0p30pct": replace(settings, daily_residual_risk_fraction=0.0030),
        "risk_0p325pct": replace(settings, daily_residual_risk_fraction=0.00325),
        "notional_8pct": replace(
            settings, daily_residual_maximum_notional_fraction=0.08
        ),
        "risk_0p30pct_notional_12pct": replace(
            settings,
            daily_residual_risk_fraction=0.0030,
            daily_residual_maximum_notional_fraction=0.12,
        ),
    }
    result = _run_settings_frontier(
        bundle,
        variants,
        max_workers=max_workers,
        score_contract=score_contract,
        maximum_selection_drawdown=0.12,
        prefer_sub_10pct_drawdown=True,
    )
    result["contract"] = "exact_aggressive_but_bounded_risk_notional_frontier_v1"
    return result


def run_exit_capture_frontier_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
) -> dict[str, Any]:
    """Test a bounded exit family around the proven ten-session control."""

    disabled_anchor = {
        "daily_residual_partial_normalization_fraction": 99.0,
        "daily_residual_full_normalization_fraction": 99.0,
        "daily_residual_structural_failure_extension_fraction": 99.0,
        "daily_residual_partial_exit_fraction": 0.0,
    }
    variants = {
        "control": settings,
        "maximum_hold_9": replace(
            settings, daily_residual_maximum_holding_sessions=9
        ),
        "catastrophic_residual_stop_7r": replace(
            settings, daily_residual_catastrophic_stop_residual_r=7.0
        ),
        "catastrophic_residual_stop_8r": replace(
            settings, daily_residual_catastrophic_stop_residual_r=8.0
        ),
        "atr_only_catastrophic_stop": replace(
            settings, daily_residual_catastrophic_stop_residual_r=0.0
        ),
        "full_normalization_1p25": replace(
            settings,
            **{
                **disabled_anchor,
                "daily_residual_full_normalization_fraction": 1.25,
            },
        ),
        "full_normalization_1p50": replace(
            settings,
            **{
                **disabled_anchor,
                "daily_residual_full_normalization_fraction": 1.50,
            },
        ),
        "late_half_partial_full_1p50": replace(
            settings,
            daily_residual_partial_normalization_fraction=1.0,
            daily_residual_full_normalization_fraction=1.50,
            daily_residual_structural_failure_extension_fraction=99.0,
            daily_residual_partial_exit_fraction=0.50,
        ),
    }
    result = _run_settings_frontier(
        bundle,
        variants,
        max_workers=max_workers,
        score_contract=score_contract,
        maximum_selection_drawdown=0.12,
        prefer_sub_10pct_drawdown=True,
    )
    result["contract"] = "exact_exit_capture_frontier_v1"
    return result


def run_final_alpha_synergy_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
) -> dict[str, Any]:
    """Test bounded interactions without reopening rejected capacity ladders."""

    risk_minus_5pct = max(0.0005, settings.daily_residual_risk_fraction * 0.95)
    floor_minus_5 = max(0.0, settings.daily_residual_minimum_score - 5.0)
    z_plus_005 = min(1.10, settings.daily_residual_minimum_z + 0.05)
    notional_plus_2pct = min(
        0.12, settings.daily_residual_maximum_notional_fraction + 0.02
    )
    variants = {
        "control": settings,
        "floor_minus_5_risk_minus_5pct": replace(
            settings,
            daily_residual_minimum_score=floor_minus_5,
            daily_residual_risk_fraction=risk_minus_5pct,
        ),
        "z_plus_0p05_risk_minus_5pct": replace(
            settings,
            daily_residual_minimum_z=z_plus_005,
            daily_residual_risk_fraction=risk_minus_5pct,
        ),
        "notional_plus_2pct_risk_minus_5pct": replace(
            settings,
            daily_residual_maximum_notional_fraction=notional_plus_2pct,
            daily_residual_risk_fraction=risk_minus_5pct,
        ),
    }
    if settings.daily_residual_sector_overflow_slots > 0:
        overflow_score_plus_5 = min(
            100.0,
            settings.daily_residual_sector_overflow_minimum_score + 5.0,
        )
        overflow_z_plus_005 = min(
            3.0,
            settings.daily_residual_sector_overflow_minimum_z + 0.05,
        )
        overflow_risk_minus_15pct = max(
            0.25,
            settings.daily_residual_sector_overflow_risk_multiplier * 0.85,
        )
        variants.update(
            {
                "overflow_quality_plus_5": replace(
                    settings,
                    daily_residual_sector_overflow_minimum_score=(
                        overflow_score_plus_5
                    ),
                ),
                "overflow_z_plus_0p05": replace(
                    settings,
                    daily_residual_sector_overflow_minimum_z=overflow_z_plus_005,
                ),
                "overflow_risk_minus_15pct": replace(
                    settings,
                    daily_residual_sector_overflow_risk_multiplier=(
                        overflow_risk_minus_15pct
                    ),
                ),
                "floor_minus_5_overflow_risk_minus_15pct": replace(
                    settings,
                    daily_residual_minimum_score=floor_minus_5,
                    daily_residual_sector_overflow_risk_multiplier=(
                        overflow_risk_minus_15pct
                    ),
                ),
            }
        )
    result = _run_settings_frontier(
        bundle,
        variants,
        max_workers=max_workers,
        score_contract=score_contract,
        maximum_selection_drawdown=0.12,
        prefer_sub_10pct_drawdown=True,
    )
    result["contract"] = "exact_mechanism_interaction_synergy_aspirational_v3"
    result["rejected_search_directions_not_reopened"] = {
        "global_capacity_above_12": True,
        "blanket_sector_cap_above_2": True,
    }
    result["aspirational_target_contract"] = {
        "selection_total_r_guidance": 100.0,
        "selection_mtm_max_drawdown_guidance": 0.10,
        "absolute_mtm_max_drawdown_safety_ceiling": 0.12,
        "used_as_hard_rejection_gate": False,
        "selection_requires_immutable_score_improvement_or_control_retention": True,
        "locked_validation_used_for_selection": False,
    }
    return result


def run_path_causal_profit_retention_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
) -> dict[str, Any]:
    """Test a small causal response to diagnosed residual-MFE giveback.

    The trigger observes completed-session residual normalization only and
    exits at the next open through the unchanged shared execution path.  The
    thresholds are coarse mechanism anchors, not a fitted grid.
    """

    variants = {
        "control": settings,
        "activate_0p75_giveback_0p35": replace(
            settings,
            daily_residual_profit_retention_activation_fraction=0.75,
            daily_residual_profit_retention_giveback_fraction=0.35,
        ),
        "activate_0p75_giveback_0p50": replace(
            settings,
            daily_residual_profit_retention_activation_fraction=0.75,
            daily_residual_profit_retention_giveback_fraction=0.50,
        ),
        "activate_1p00_giveback_0p35": replace(
            settings,
            daily_residual_profit_retention_activation_fraction=1.00,
            daily_residual_profit_retention_giveback_fraction=0.35,
        ),
        "activate_1p00_giveback_0p50": replace(
            settings,
            daily_residual_profit_retention_activation_fraction=1.00,
            daily_residual_profit_retention_giveback_fraction=0.50,
        ),
        "activate_1p25_giveback_0p50": replace(
            settings,
            daily_residual_profit_retention_activation_fraction=1.25,
            daily_residual_profit_retention_giveback_fraction=0.50,
        ),
    }
    result = _run_settings_frontier(
        bundle,
        variants,
        max_workers=max_workers,
        score_contract=score_contract,
        maximum_selection_drawdown=0.12,
        prefer_sub_10pct_drawdown=True,
    )
    result["contract"] = "exact_path_causal_residual_profit_retention_v1"
    result["causality_contract"] = {
        "input": "completed_session_frozen_model_residual_path",
        "decision_time": "after_completed_daily_session",
        "earliest_fill": "next_session_open",
        "fill_and_cost_path": "shared_daily_residual_execution_core",
        "threshold_grid_fitted_to_sample": False,
    }
    return result


def run_capacity_neutral_alpha_recycling_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
) -> dict[str, Any]:
    """Test opportunity-cost rotation without adding capacity or score inputs.

    A challenger must improve the actual shared-capital portfolio in both
    selection folds at 20 and 30 bps.  At least twelve changed decisions per
    fold are required for promotion, but an undersized cohort is classified as
    diagnostic-only and simply retains the control; it never blocks the round.
    """

    fixed = {
        "daily_residual_replacement_minimum_held_sessions": 5,
        "daily_residual_replacement_maximum_normalization_fraction": 0.25,
        "daily_residual_replacement_minimum_score_margin": 25.0,
        "daily_residual_replacement_max_per_session": 1,
    }
    variants = {
        "control": replace(
            settings,
            daily_residual_replacement_mode="disabled",
            daily_residual_replacement_loss_only=False,
            **fixed,
        ),
        "same_sector_stale_replacement": replace(
            settings,
            daily_residual_replacement_mode="sector_stale",
            daily_residual_replacement_loss_only=False,
            **fixed,
        ),
        "portfolio_diversifying_stale_replacement": replace(
            settings,
            daily_residual_replacement_mode="portfolio_diversifying",
            daily_residual_replacement_loss_only=False,
            **fixed,
        ),
        "combined_loss_only_replacement": replace(
            settings,
            daily_residual_replacement_mode="combined",
            daily_residual_replacement_loss_only=True,
            **fixed,
        ),
        "combined_stale_replacement": replace(
            settings,
            daily_residual_replacement_mode="combined",
            daily_residual_replacement_loss_only=False,
            **fixed,
        ),
    }
    phase20 = _run_settings_frontier(
        bundle,
        variants,
        max_workers=max_workers,
        score_contract=score_contract,
        maximum_selection_drawdown=0.12,
        prefer_sub_10pct_drawdown=True,
    )
    rows = phase20["experiments"]
    control20 = next(
        row for row in rows if row["experiment_id"] == "control"
    )

    def cost30(item: tuple[str, StrategySettings]) -> tuple[str, dict[str, Any]]:
        experiment_id, trial = item
        result = run_exact_fold_evaluation(
            replace(bundle, frozen_history_cache={}),
            trial,
            round_trip_cost_bps=30.0,
            score_contract=score_contract,
        )
        return (
            experiment_id,
            {
                key: value
                for key, value in result.items()
                if key not in {"trades", "equity_curves"}
            },
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        stress30 = dict(pool.map(cost30, variants.items()))
    control30 = stress30["control"]
    minimum_changed_decisions = 12

    for row in rows:
        experiment_id = row["experiment_id"]
        result20 = row["result"]
        result30 = stress30[experiment_id]
        row["cost_stress_30bps"] = result30
        if experiment_id == "control":
            row["replacement_evidence"] = {
                "role": "literal_no_replacement_control",
                "promotion_eligible": True,
            }
            continue
        diagnostics = result20[
            "capacity_neutral_replacement_diagnostics"
        ]["folds"]
        changed = {
            fold: int(diagnostics[fold]["changed_decisions"])
            for fold, _start, _end in FOLDS
        }
        incremental20 = {
            fold: (
                float(result20["folds"][fold]["total_r"])
                - float(control20["result"]["folds"][fold]["total_r"])
            )
            for fold, _start, _end in FOLDS
        }
        incremental30 = {
            fold: (
                float(result30["folds"][fold]["total_r"])
                - float(control30["folds"][fold]["total_r"])
            )
            for fold, _start, _end in FOLDS
        }
        gates = {
            "at_least_12_actual_replacements_each_fold": all(
                value >= minimum_changed_decisions for value in changed.values()
            ),
            "positive_incremental_r_each_fold_20bps": all(
                value > 0.0 for value in incremental20.values()
            ),
            "positive_incremental_r_each_fold_30bps": all(
                value > 0.0 for value in incremental30.values()
            ),
            "thirty_bps_exact_gates_pass": bool(
                result30["research_anchor_eligible"]
            ),
            "thirty_bps_score_exceeds_control": (
                float(result30["immutable_score"]["score"])
                > float(control30["immutable_score"]["score"]) + 1e-12
            ),
            "thirty_bps_drawdown_lte_12pct": (
                float(result30["continuous_metrics"]["max_drawdown_pct"])
                <= 0.12
            ),
        }
        promotion_eligible = all(gates.values())
        row["selection_eligible"] = bool(
            row["selection_eligible"] and promotion_eligible
        )
        row["replacement_evidence"] = {
            "changed_decisions": changed,
            "incremental_r_vs_control_20bps": incremental20,
            "incremental_r_vs_control_30bps": incremental30,
            "gates": gates,
            "promotion_eligible": promotion_eligible,
            "evidence_class": (
                "exact_both_fold_capacity_neutral_replacement"
                if promotion_eligible
                else (
                    "diagnostic_only_insufficient_changed_decisions"
                    if not gates["at_least_12_actual_replacements_each_fold"]
                    else "exact_rejected_replacement_challenger"
                )
            ),
        }

    eligible = [row for row in rows if row["selection_eligible"]]
    if not eligible:
        raise RuntimeError("capacity-neutral phase lost its eligible control")

    def ranking_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        result = row["result"]
        metrics = result["continuous_metrics"]
        return (
            float(result["immutable_score"]["score"]),
            float(metrics["max_drawdown_pct"] < 0.10),
            float(metrics["total_r"]),
            float(metrics["return_pct"]),
            -float(metrics["max_drawdown_pct"]),
        )

    selected = max(eligible, key=ranking_key)
    control_score = float(control20["result"]["immutable_score"]["score"])
    selected_score = float(selected["result"]["immutable_score"]["score"])
    if selected_score <= control_score + 1e-12:
        selected = control20
        selected_score = control_score
        status = "complete_control_retained"
    else:
        status = "passed"
    return {
        "status": status,
        "contract": "exact_capacity_neutral_alpha_recycling_v1",
        "experiments": rows,
        "selected": selected,
        "selected_settings": selected["settings"],
        "control": control20,
        "score_delta_vs_control": selected_score - control_score,
        "maximum_selection_drawdown": 0.12,
        "aspirational_targets_are_hard_gates": False,
        "minimum_changed_decisions_per_fold_for_promotion": (
            minimum_changed_decisions
        ),
        "small_sample_contract": (
            "undersized_replacement_cohorts_are_diagnostic_only_and_retain_control"
        ),
        "capacity_contract": {
            "global_position_cap_unchanged": settings.daily_residual_max_positions,
            "ordinary_sector_cap_unchanged": (
                settings.daily_residual_max_positions_per_sector
            ),
            "maximum_replacements_per_session": 1,
            "capacity_expansion": False,
        },
        "causality_contract": {
            "inputs": "completed_session_residual_path_entry_score_and_close",
            "decision_time": "after_completed_daily_session",
            "incumbent_exit_and_candidate_entry": "next_session_open",
            "fill_and_cost_path": "shared_daily_residual_execution_core",
            "counterfactual_shadow_pnl_used_for_selection": False,
        },
        "immutable_score_changed": False,
        "score_component_union_ceiling": 7,
        "locked_validation_accessed": False,
        "holdout_accessed": False,
    }


def run_final_robustness_and_target_assessment_phase(
    bundle: DailyResidualReplayBundle,
    settings: StrategySettings,
    *,
    max_workers: int = 2,
    score_contract: str = "round2",
) -> dict[str, Any]:
    """Re-run final invariance checks and report, but do not gate on, aspirations."""

    result = run_final_robustness_phase(
        bundle,
        settings,
        max_workers=max_workers,
        score_contract=score_contract,
    )
    metrics = result["base_20bps"]["continuous_metrics"]
    target_assessment = {
        "selection_total_r_above_100r": float(metrics["total_r"]) > 100.0,
        "selection_mtm_max_drawdown_below_10pct": (
            float(metrics["max_drawdown_pct"]) < 0.10
        ),
    }
    result["qualification"]["gates"][
        "selection_mtm_max_drawdown_lte_12pct_safety_ceiling"
    ] = float(metrics["max_drawdown_pct"]) <= 0.12
    result["qualification"]["passed"] = all(
        result["qualification"]["gates"].values()
    )
    result["qualification"]["failed_gates"] = [
        name
        for name, passed in result["qualification"]["gates"].items()
        if not passed
    ]
    result["status"] = (
        "passed" if result["qualification"]["passed"] else "blocked"
    )
    result["aspirational_target_assessment"] = {
        **target_assessment,
        "both_met": all(target_assessment.values()),
        "used_as_hard_rejection_gate": False,
    }
    result["contract"] = "final_robustness_with_aspirational_target_v2"
    return result


def _drawdown(values: Iterable[float]) -> float:
    peak = -math.inf
    maximum = 0.0
    for value in values:
        peak = max(peak, float(value))
        maximum = max(maximum, (peak - float(value)) / max(abs(peak), 1e-9))
    return maximum


def run_protected_integration_phase(
    bundle: DailyResidualReplayBundle,
    exact: Mapping[str, Any],
    *,
    frozen_control_trades_path: Path,
) -> dict[str, Any]:
    """Literal issuer arbitration against the frozen Round-3 control.

    The control always wins an issuer collision.  Sleeve capacity remains
    separate, as required by the redesign, and dollar PnL is marked daily on a
    shared 100k ledger.  The control is descriptive only because its historical
    selection used a later interval; its outcomes never rank residual variants.
    """

    payload = json.loads(frozen_control_trades_path.read_text(encoding="utf-8"))
    controls: list[dict[str, Any]] = []
    for row in payload:
        entry = datetime.fromisoformat(str(row["entry_time"]))
        exit_time = datetime.fromisoformat(str(row["exit_time"]))
        if entry.date() > date.fromisoformat(CALIBRATION_END):
            continue
        risk_dollars = (
            float(row["pnl_net"]) / float(row["r"])
            if abs(float(row["r"])) > 1e-12
            else 0.0
        )
        qty = max(
            1,
            int(round(abs(risk_dollars) / max(float(row["risk_per_share"]), 1e-9))),
        )
        controls.append(
            {
                **row,
                "entry_dt": entry,
                "exit_dt": exit_time,
                "issuer": issuer_key(str(row["symbol"])),
                "qty": qty,
            }
        )

    fold_rows: dict[str, Any] = {}
    all_combined_equity: list[float] = []
    for fold_name, start, end in FOLDS:
        control_fold = [
            row for row in controls if start <= row["entry_dt"].date() <= end
        ]
        residual_fold = exact["trades"][fold_name]
        accepted: list[Mapping[str, Any]] = []
        collided: list[Mapping[str, Any]] = []
        for row in residual_fold:
            entry = row["entry_time"]
            if not isinstance(entry, datetime):
                entry = datetime.fromisoformat(str(entry))
            issuer = issuer_key(str(row["symbol"]))
            conflict = any(
                control["issuer"] == issuer
                and control["entry_dt"] <= entry <= control["exit_dt"]
                for control in control_fold
            )
            (collided if conflict else accepted).append(row)
        residual_r = sum(float(row["r_multiple"]) for row in accepted)
        collision_r = sum(float(row["r_multiple"]) for row in collided)
        control_r = sum(float(row["r"]) for row in control_fold)
        control_pnl = sum(float(row["pnl_net"]) for row in control_fold)
        residual_pnl = sum(float(row["net_pnl"]) for row in accepted)

        session_dates = sorted(
            stamp.date()
            for stamp in pd.to_datetime(bundle.close.index)
            if start <= stamp.date() <= end
        )
        equity: list[float] = []
        for session in session_dates:
            control_mark = 0.0
            for row in control_fold:
                if session < row["entry_dt"].date():
                    continue
                if session >= row["exit_dt"].date():
                    control_mark += float(row["pnl_net"])
                    continue
                symbol = str(row["symbol"]).replace(" ", ".")
                price = _panel_close(bundle, session, symbol)
                if price is not None:
                    control_mark += (price - float(row["entry_price"])) * int(row["qty"])
            residual_mark = 0.0
            for row in accepted:
                entry_date = _coerce_date(row["entry_date"])
                exit_date = _coerce_date(row["exit_date"])
                if session < entry_date:
                    continue
                if session >= exit_date:
                    residual_mark += float(row["net_pnl"])
                    continue
                price = _panel_close(bundle, session, str(row["symbol"]))
                if price is not None:
                    residual_mark += (
                        (price - float(row["entry_price"])) * int(row["qty_entry"])
                        - float(row["commission"]) * 0.5
                    )
            equity.append(100_000.0 + control_mark + residual_mark)
        all_combined_equity.extend(equity)
        fold_rows[fold_name] = {
            "control_trades": len(control_fold),
            "residual_trades_before_arbitration": len(residual_fold),
            "residual_trades_after_arbitration": len(accepted),
            "issuer_collisions": len(collided),
            "residual_marginal_r_after_collisions": residual_r,
            "residual_r_rejected_by_collisions": collision_r,
            "control_total_r": control_r,
            "combined_total_r": control_r + residual_r,
            "control_total_pnl_usd": control_pnl,
            "residual_marginal_pnl_usd": residual_pnl,
            "combined_final_equity": equity[-1] if equity else 100_000.0,
            "combined_max_drawdown_pct": _drawdown(equity),
        }
    gates = {
        "positive_residual_marginal_r_each_fold": all(
            float(row["residual_marginal_r_after_collisions"]) > 0.0
            for row in fold_rows.values()
        ),
        "residual_retains_at_least_80pct_of_trades_each_fold": all(
            int(row["residual_trades_after_arbitration"])
            >= 0.80 * max(int(row["residual_trades_before_arbitration"]), 1)
            for row in fold_rows.values()
        ),
        "combined_shared_ledger_drawdown_lte_10pct": _drawdown(all_combined_equity)
        <= 0.10,
    }
    return {
        "status": "passed" if all(gates.values()) else "blocked",
        "control_role": "frozen_descriptive_protected_control_not_selection_evidence",
        "capital_contract": "shared_100k_mark_to_market_ledger_with_separate_sleeve_caps",
        "issuer_arbitration": "frozen_round3_control_priority_one_position_per_issuer",
        "folds": fold_rows,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def _panel_close(
    bundle: DailyResidualReplayBundle,
    session: date,
    symbol: str,
) -> float | None:
    stamp = bundle.stamp_by_date.get(session)
    if stamp is None:
        return None
    # The frozen Round-3 control serialises class-share tickers with dots while
    # the IBKR research panel uses spaces.  Resolve only these deterministic
    # aliases; never fall back to fuzzy ticker matching or another issuer.
    aliases = tuple(
        dict.fromkeys(
            (
                symbol,
                symbol.replace(".", " "),
                symbol.replace(" ", "."),
            )
        )
    )
    for alias in aliases:
        if alias not in bundle.close:
            continue
        value = bundle.close.at[stamp, alias]
        return (
            float(value)
            if pd.notna(value) and np.isfinite(float(value))
            else None
        )
    return None
