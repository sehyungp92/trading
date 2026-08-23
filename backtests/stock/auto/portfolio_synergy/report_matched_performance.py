from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from backtests.stock.auto.portfolio_synergy.core.market import CausalPriceBook
from backtests.stock.auto.portfolio_synergy.evaluator import _load_stock_price_bars
from backtests.stock.auto.portfolio_synergy.run_corrected_phased_auto import (
    DATA_DIR,
    IS_END,
    OOS_END,
    OOS_START,
    START,
    StrategyStreams,
    _alpha_model_diagnostics,
    _attach_predictions,
    _evaluate_window,
    _expanding_predictions,
    _filter,
    _generate_streams,
    _no_overlay,
    _write_json,
    baseline_config,
)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float]:
    cleaned = [float(value) for value in values]
    positive = [value for value in cleaned if value > 0.0]
    nonpositive = [value for value in cleaned if value <= 0.0]
    return {
        "count": float(len(cleaned)),
        "positive_count": float(len(positive)),
        "nonpositive_count": float(len(nonpositive)),
        "win_rate": float(len(positive) / len(cleaned)) if cleaned else 0.0,
        "total": float(sum(cleaned)),
        "positive_total": float(sum(positive)),
        "nonpositive_total": float(sum(nonpositive)),
        "average": float(mean(cleaned)) if cleaned else 0.0,
        "median": float(median(cleaned)) if cleaned else 0.0,
        "minimum": min(cleaned, default=0.0),
        "p10": _quantile(cleaned, 0.10),
        "p25": _quantile(cleaned, 0.25),
        "p75": _quantile(cleaned, 0.75),
        "p90": _quantile(cleaned, 0.90),
        "maximum": max(cleaned, default=0.0),
    }


def _candidate_key(candidate: Any) -> tuple[str, str, str]:
    strategy = str(getattr(candidate, "strategy", ""))
    symbol = str(getattr(candidate, "symbol", ""))
    entry_time = getattr(candidate, "entry_time", None)
    return strategy, symbol, entry_time.isoformat() if entry_time is not None else ""


def _blocked_set_delta(optimized_result: Any, control_result: Any) -> dict[str, Any]:
    optimized_counts = Counter(
        _candidate_key(candidate)
        for candidate in optimized_result.state.blocked_candidates
    )
    control_counts = Counter(
        _candidate_key(candidate) for candidate in control_result.state.blocked_candidates
    )

    def exclusive(source: Any, source_counts: Counter, other_counts: Counter) -> list[Any]:
        remaining = source_counts - other_counts
        selected: list[Any] = []
        for candidate in source.state.blocked_candidates:
            key = _candidate_key(candidate)
            if remaining[key] > 0:
                selected.append(candidate)
                remaining[key] -= 1
        return selected

    optimized_only = exclusive(
        optimized_result,
        optimized_counts,
        control_counts,
    )
    control_only = exclusive(control_result, control_counts, optimized_counts)
    common = optimized_counts & control_counts
    optimized_distribution = _distribution(
        [candidate.r_multiple for candidate in optimized_only]
    )
    control_distribution = _distribution(
        [candidate.r_multiple for candidate in control_only]
    )
    return {
        "common_block_count": float(sum(common.values())),
        "optimized_overlay_only": optimized_distribution,
        "no_overlay_only": control_distribution,
        "net_additional_block_count": float(
            len(optimized_result.state.blocked_candidates)
            - len(control_result.state.blocked_candidates)
        ),
        "net_blocked_total_r_delta": float(
            sum(candidate.r_multiple for candidate in optimized_result.state.blocked_candidates)
            - sum(candidate.r_multiple for candidate in control_result.state.blocked_candidates)
        ),
    }


def _daily_mtm_diagnostics(result: Any, *, close, start, end) -> dict[str, Any]:
    positions = list(result.state.accepted_positions)
    strategies = sorted({position.strategy for position in positions})
    timeline = close.loc[start.isoformat() : end.isoformat()].index
    if len(timeline) == 0:
        return {
            "points": 0,
            "max_drawdown_pct": 0.0,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "drawdown_duration_days": 0,
            "drawdown_contribution_by_strategy": {},
            "worst_drawdown_days": [],
        }

    rows: list[dict[str, Any]] = []
    peak_equity = 25_000.0
    peak_index = -1
    max_drawdown = 0.0
    max_peak_index = -1
    max_trough_index = -1
    for timestamp in timeline:
        day = timestamp.date()
        contribution: dict[str, float] = {}
        for strategy in strategies:
            strategy_positions = [
                position for position in positions if position.strategy == strategy
            ]
            realized = sum(
                position.pnl
                for position in strategy_positions
                if position.exit_time.date() <= day
            )
            unrealized = 0.0
            for position in strategy_positions:
                if not (
                    position.entry_time.date() <= day < position.exit_time.date()
                ):
                    continue
                if position.symbol not in close.columns or timestamp not in close.index:
                    continue
                raw_close = float(close.at[timestamp, position.symbol])
                if not math.isfinite(raw_close) or raw_close <= 0.0:
                    continue
                ratio = raw_close / max(float(position.entry_price), 1e-9)
                if ratio > 3.0 or ratio < 1.0 / 3.0:
                    raw_close /= 10.0 ** round(math.log10(ratio))
                unrealized += (
                    (raw_close - float(position.entry_price))
                    * float(position.direction)
                    * float(position.quantity)
                )
            contribution[strategy] = float(realized + unrealized)
        equity = 25_000.0 + sum(contribution.values())
        if equity > peak_equity:
            peak_equity = equity
            peak_index = len(rows)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 0.0
        rows.append(
            {
                "date": day.isoformat(),
                "equity": float(equity),
                "drawdown_pct": float(drawdown),
                "strategy_mtm_pnl": contribution,
            }
        )
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_peak_index = peak_index
            max_trough_index = len(rows) - 1

    if max_peak_index < 0:
        peak_date = start.isoformat()
        peak_contribution = {strategy: 0.0 for strategy in strategies}
        peak_equity_value = 25_000.0
    else:
        peak_date = rows[max_peak_index]["date"]
        peak_contribution = rows[max_peak_index]["strategy_mtm_pnl"]
        peak_equity_value = rows[max_peak_index]["equity"]
    if max_trough_index < 0:
        trough_date = peak_date
        trough_contribution = peak_contribution
        trough_equity_value = peak_equity_value
    else:
        trough_date = rows[max_trough_index]["date"]
        trough_contribution = rows[max_trough_index]["strategy_mtm_pnl"]
        trough_equity_value = rows[max_trough_index]["equity"]
    recovery_date = None
    if max_trough_index >= 0:
        for row in rows[max_trough_index + 1 :]:
            if float(row["equity"]) >= peak_equity_value:
                recovery_date = row["date"]
                break
    contribution_delta = {
        strategy: float(
            trough_contribution.get(strategy, 0.0)
            - peak_contribution.get(strategy, 0.0)
        )
        for strategy in strategies
    }
    worst_days = sorted(rows, key=lambda row: row["drawdown_pct"], reverse=True)[:10]
    return {
        "points": len(rows),
        "max_drawdown_pct": float(max_drawdown),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "drawdown_duration_days": (
            (timeline[max_trough_index].date() - timeline[max_peak_index].date()).days
            if max_peak_index >= 0 and max_trough_index >= 0
            else 0
        ),
        "peak_equity": float(peak_equity_value),
        "trough_equity": float(trough_equity_value),
        "drawdown_contribution_by_strategy": contribution_delta,
        "worst_drawdown_days": [
            {
                "date": row["date"],
                "equity": row["equity"],
                "drawdown_pct": row["drawdown_pct"],
            }
            for row in worst_days
        ],
    }


def _cross_strategy_crowding(streams: StrategyStreams, start, end) -> dict[str, float]:
    alcb = list(_filter(streams.alcb, start, end))
    iaric = list(_filter(streams.iaric, start, end))
    all_trades = [("ALCB_R3", trade) for trade in alcb] + [
        ("IARIC_RESIDUAL_R3", trade) for trade in iaric
    ]
    time_strategies: dict[Any, set[str]] = defaultdict(set)
    time_symbol_strategies: dict[tuple[Any, str], set[str]] = defaultdict(set)
    for strategy, trade in all_trades:
        time_strategies[trade.entry_time].add(strategy)
        time_symbol_strategies[(trade.entry_time, trade.symbol)].add(strategy)
    exact_cross = sum(
        1
        for strategy, trade in all_trades
        if len(time_strategies[trade.entry_time]) > 1
    )
    exact_symbol_cross = sum(
        1
        for strategy, trade in all_trades
        if len(time_symbol_strategies[(trade.entry_time, trade.symbol)]) > 1
    )

    alcb_times = sorted(trade.entry_time for trade in alcb)
    iaric_times = sorted(trade.entry_time for trade in iaric)

    def near_count(source: list[Any], other: list[Any], seconds: float) -> int:
        count = 0
        for timestamp in source:
            index = bisect_left(other, timestamp)
            neighbors = other[max(0, index - 1) : min(len(other), index + 2)]
            if any(abs((candidate - timestamp).total_seconds()) <= seconds for candidate in neighbors):
                count += 1
        return count

    within_one_day = near_count(alcb_times, iaric_times, 86_400.0) + near_count(
        iaric_times, alcb_times, 86_400.0
    )
    total = len(all_trades)
    return {
        "candidate_count": float(total),
        "exact_timestamp_cross_strategy_count": float(exact_cross),
        "exact_timestamp_cross_strategy_rate": float(exact_cross / total) if total else 0.0,
        "exact_timestamp_same_symbol_cross_strategy_count": float(exact_symbol_cross),
        "exact_timestamp_same_symbol_cross_strategy_rate": (
            float(exact_symbol_cross / total) if total else 0.0
        ),
        "within_one_day_cross_strategy_count": float(within_one_day),
        "within_one_day_cross_strategy_rate": float(within_one_day / total) if total else 0.0,
    }


def _result_diagnostics(result: Any, streams: StrategyStreams, *, close, start, end) -> dict[str, Any]:
    accepted = list(result.state.accepted_positions)
    blocked = list(result.state.blocked_candidates)
    strategies = sorted(
        {position.strategy for position in accepted}
        | {candidate.strategy for candidate in blocked}
    )
    by_strategy: dict[str, Any] = {}
    for strategy in strategies:
        accepted_rows = [position for position in accepted if position.strategy == strategy]
        blocked_rows = [candidate for candidate in blocked if candidate.strategy == strategy]
        accepted_r = _distribution([position.r_multiple for position in accepted_rows])
        blocked_r = _distribution([candidate.r_multiple for candidate in blocked_rows])
        accepted_quality = _distribution([position.quality for position in accepted_rows])
        blocked_quality = _distribution([candidate.quality for candidate in blocked_rows])
        candidate_positive = accepted_r["positive_count"] + blocked_r["positive_count"]
        candidate_nonpositive = accepted_r["nonpositive_count"] + blocked_r["nonpositive_count"]
        by_strategy[strategy] = {
            "fired": float(len(accepted_rows) + len(blocked_rows)),
            "accepted": float(len(accepted_rows)),
            "blocked": float(len(blocked_rows)),
            "accept_rate": (
                float(len(accepted_rows) / (len(accepted_rows) + len(blocked_rows)))
                if accepted_rows or blocked_rows
                else 0.0
            ),
            "accepted_r": accepted_r,
            "blocked_r": blocked_r,
            "accepted_quality": accepted_quality,
            "blocked_quality": blocked_quality,
            "positive_trade_block_rate": (
                float(blocked_r["positive_count"] / candidate_positive)
                if candidate_positive
                else 0.0
            ),
            "nonpositive_trade_block_rate": (
                float(blocked_r["nonpositive_count"] / candidate_nonpositive)
                if candidate_nonpositive
                else 0.0
            ),
            "realized_r_discrimination": float(
                accepted_r["average"] - blocked_r["average"]
            ),
            "quality_discrimination": float(
                accepted_quality["average"] - blocked_quality["average"]
            ),
        }

    by_reason: dict[str, list[Any]] = defaultdict(list)
    by_reason_strategy: dict[str, Counter] = defaultdict(Counter)
    for candidate in blocked:
        by_reason[candidate.reason].append(candidate)
        by_reason_strategy[candidate.reason][candidate.strategy] += 1
    block_reasons = {
        reason: {
            "r": _distribution([candidate.r_multiple for candidate in candidates]),
            "quality": _distribution([candidate.quality for candidate in candidates]),
            "strategies": dict(by_reason_strategy[reason]),
            "average_requested_notional": float(
                mean([candidate.requested_notional for candidate in candidates])
            ),
            "average_heat_r": float(mean([candidate.heat_r for candidate in candidates])),
        }
        for reason, candidates in sorted(by_reason.items())
    }

    accepted_r = _distribution([position.r_multiple for position in accepted])
    blocked_r = _distribution([candidate.r_multiple for candidate in blocked])
    accepted_quality = _distribution([position.quality for position in accepted])
    blocked_quality = _distribution([candidate.quality for candidate in blocked])
    candidate_positive = accepted_r["positive_count"] + blocked_r["positive_count"]
    candidate_nonpositive = accepted_r["nonpositive_count"] + blocked_r["nonpositive_count"]
    avoided_loss = -blocked_r["nonpositive_total"]
    forgone_gain = blocked_r["positive_total"]
    open_counts: list[int] = []
    blocked_with_any = 0
    blocked_with_other_strategy = 0
    blocked_with_same_symbol = 0
    blocked_with_same_sector = 0
    for candidate in blocked:
        active = [
            position
            for position in accepted
            if position.entry_time <= candidate.entry_time < position.exit_time
        ]
        open_counts.append(len(active))
        blocked_with_any += int(bool(active))
        blocked_with_other_strategy += int(
            any(position.strategy != candidate.strategy for position in active)
        )
        blocked_with_same_symbol += int(
            any(position.symbol == candidate.symbol for position in active)
        )
        blocked_with_same_sector += int(
            bool(candidate.sector)
            and any(position.sector == candidate.sector for position in active)
        )

    monthly: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "accepted": 0,
            "blocked": 0,
            "accepted_r": 0.0,
            "blocked_r": 0.0,
            "pnl": 0.0,
            "by_strategy": defaultdict(
                lambda: {"accepted": 0, "blocked": 0, "accepted_r": 0.0, "blocked_r": 0.0, "pnl": 0.0}
            ),
        }
    )
    for position in accepted:
        key = position.exit_time.strftime("%Y-%m")
        row = monthly[key]
        row["accepted"] += 1
        row["accepted_r"] += float(position.r_multiple)
        row["pnl"] += float(position.pnl)
        sleeve = row["by_strategy"][position.strategy]
        sleeve["accepted"] += 1
        sleeve["accepted_r"] += float(position.r_multiple)
        sleeve["pnl"] += float(position.pnl)
    for candidate in blocked:
        key = candidate.entry_time.strftime("%Y-%m")
        row = monthly[key]
        row["blocked"] += 1
        row["blocked_r"] += float(candidate.r_multiple)
        sleeve = row["by_strategy"][candidate.strategy]
        sleeve["blocked"] += 1
        sleeve["blocked_r"] += float(candidate.r_multiple)
    monthly_rows = []
    for month, row in sorted(monthly.items()):
        monthly_rows.append(
            {
                "month": month,
                "accepted": row["accepted"],
                "blocked": row["blocked"],
                "accepted_r": row["accepted_r"],
                "blocked_r": row["blocked_r"],
                "pnl": row["pnl"],
                "by_strategy": dict(row["by_strategy"]),
            }
        )

    fired = len(accepted) + len(blocked)
    return {
        "reconciliation": {
            "fired": fired,
            "accepted": len(accepted),
            "blocked": len(blocked),
            "reconciles": fired == int(result.state.candidate_count),
        },
        "accepted_r": accepted_r,
        "blocked_r": blocked_r,
        "accepted_quality": accepted_quality,
        "blocked_quality": blocked_quality,
        "positive_trade_block_rate": (
            float(blocked_r["positive_count"] / candidate_positive)
            if candidate_positive
            else 0.0
        ),
        "nonpositive_trade_block_rate": (
            float(blocked_r["nonpositive_count"] / candidate_nonpositive)
            if candidate_nonpositive
            else 0.0
        ),
        "blocker_precision_nonpositive": float(
            blocked_r["nonpositive_count"] / blocked_r["count"]
        ) if blocked_r["count"] else 0.0,
        "block_efficiency": float(avoided_loss / (avoided_loss + forgone_gain))
        if avoided_loss + forgone_gain > 0.0
        else 0.0,
        "avoided_loss_r": float(avoided_loss),
        "forgone_gain_r": float(forgone_gain),
        "net_block_value_r": float(avoided_loss - forgone_gain),
        "realized_r_discrimination": float(accepted_r["average"] - blocked_r["average"]),
        "quality_discrimination": float(accepted_quality["average"] - blocked_quality["average"]),
        "by_strategy": by_strategy,
        "block_reasons": block_reasons,
        "capacity_context": {
            "blocked_with_any_accepted_position_open_rate": (
                float(blocked_with_any / len(blocked)) if blocked else 0.0
            ),
            "blocked_with_other_strategy_open_rate": (
                float(blocked_with_other_strategy / len(blocked)) if blocked else 0.0
            ),
            "blocked_with_same_symbol_open_rate": (
                float(blocked_with_same_symbol / len(blocked)) if blocked else 0.0
            ),
            "blocked_with_same_sector_open_rate": (
                float(blocked_with_same_sector / len(blocked)) if blocked else 0.0
            ),
            "average_open_positions_at_block": float(mean(open_counts)) if open_counts else 0.0,
            "maximum_open_positions_at_block": max(open_counts, default=0),
        },
        "signal_crowding": _cross_strategy_crowding(streams, start, end),
        "monthly": monthly_rows,
        "daily_mtm_drawdown": _daily_mtm_diagnostics(
            result,
            close=close,
            start=start,
            end=end,
        ),
    }


def _evaluate_set(
    streams: StrategyStreams,
    *,
    close,
    price_book: CausalPriceBook,
    start,
    end,
    pre_config: dict[str, Any],
    post_config: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    def evaluate(
        name: str,
        selected_streams: StrategyStreams,
        config: dict[str, Any],
    ) -> dict[str, float]:
        metrics, result = _evaluate_window(
            selected_streams,
            config,
            close=close,
            price_book=price_book,
            start=start,
            end=end,
        )
        results[name] = (result, selected_streams)
        return metrics

    payload = {
        "window": [start, end],
        "pre_optimization_portfolio": evaluate(
            "pre_optimization_portfolio", streams, pre_config
        ),
        "post_optimization_portfolio": evaluate(
            "post_optimization_portfolio", streams, post_config
        ),
        "post_optimization_no_overlay": evaluate(
            "post_optimization_no_overlay",
            streams,
            _no_overlay(post_config),
        ),
        "alcb_round3_standalone_native_risk": evaluate(
            "alcb_round3_standalone_native_risk",
            StrategyStreams(streams.alcb, ()),
            pre_config,
        ),
        "iaric_round3_standalone_native_risk": evaluate(
            "iaric_round3_standalone_native_risk",
            StrategyStreams((), streams.iaric),
            pre_config,
        ),
        "alcb_standalone_post_risk": evaluate(
            "alcb_standalone_post_risk",
            StrategyStreams(streams.alcb, ()),
            post_config,
        ),
        "iaric_standalone_post_risk": evaluate(
            "iaric_standalone_post_risk",
            StrategyStreams((), streams.iaric),
            post_config,
        ),
    }
    detailed = {
        name: _result_diagnostics(
            result,
            selected_streams,
            close=close,
            start=start,
            end=end,
        )
        for name, (result, selected_streams) in results.items()
    }
    detailed["overlay_block_set_delta"] = _blocked_set_delta(
        results["post_optimization_portfolio"][0],
        results["post_optimization_no_overlay"][0],
    )
    payload["detailed_diagnostics"] = detailed
    return payload


def run(optimized_config: Path, output: Path) -> None:
    post_config = json.loads(optimized_config.read_text(encoding="utf-8"))
    pre_config = baseline_config()
    raw_streams, close, receipt = _generate_streams()
    symbols = {
        trade.symbol for trade in (*raw_streams.alcb, *raw_streams.iaric)
    }
    price_book = CausalPriceBook(
        _load_stock_price_bars(DATA_DIR, symbols),
        daily_close=close,
    )

    is_raw = StrategyStreams(
        _filter(raw_streams.alcb, START, IS_END),
        _filter(raw_streams.iaric, START, IS_END),
    )
    variants = {
        "constant": None,
        "ridge_10": 10.0,
        "ridge_50": 50.0,
        "ridge_200": 200.0,
    }
    alpha_model, _diagnostics, predicted_variants = _alpha_model_diagnostics(
        is_raw,
        variants,
    )
    selected_ridge = variants[alpha_model]
    all_predicted = StrategyStreams(
        _attach_predictions(
            raw_streams.alcb,
            _expanding_predictions(
                raw_streams.alcb,
                ridge=selected_ridge,
                train_end=IS_END,
            ),
        ),
        _attach_predictions(
            raw_streams.iaric,
            _expanding_predictions(
                raw_streams.iaric,
                ridge=selected_ridge,
                train_end=IS_END,
            ),
        ),
    )
    payload = {
        "contract": "same_25000_capital_costs_causal_marks_and_boundaries",
        "optimized_config": str(optimized_config),
        "alpha_model": alpha_model,
        "stream_receipt": receipt,
        "is": _evaluate_set(
            predicted_variants[alpha_model],
            close=close,
            price_book=price_book,
            start=START,
            end=IS_END,
            pre_config=pre_config,
            post_config=post_config,
        ),
        "oos": _evaluate_set(
            all_predicted,
            close=close,
            price_book=price_book,
            start=OOS_START,
            end=OOS_END,
            pre_config=pre_config,
            post_config=post_config,
        ),
    }
    _write_json(output, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.optimized_config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
