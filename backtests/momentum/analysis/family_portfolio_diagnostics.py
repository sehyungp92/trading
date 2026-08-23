from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backtests.momentum.analysis.metrics import (
    compute_cagr,
    compute_max_drawdown,
)
from backtests.momentum.data.cache import load_bars
from backtests.momentum.data.preprocessing import normalize_timezone
from backtests.momentum.engine.family_portfolio_engine import (
    FamilyPortfolioBacktestConfig,
    FamilyPortfolioBacktester,
    FamilyPortfolioTrade,
    MOMENTUM_FAMILY_STRATEGY_IDS,
    family_config_from_dict,
)


STRATEGY_OUTPUTS = {
    "NQDTC_v2.1": ("nqdtc", "round_5"),
    "VdubusNQ_v4": ("vdubus", "round_3"),
    "DownturnDominator_v1": ("downturn", "round_4"),
    "NQ_REGIME": ("nq_regime", "round_6"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build detailed diagnostics for the four-strategy momentum portfolio replay.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--momentum-output-root", default="backtests/output/momentum")
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    diagnostics = build_diagnostics(run_dir, Path(args.momentum_output_root), Path(args.data_dir))
    _write_json(run_dir / "portfolio_diagnostics.json", diagnostics)
    (run_dir / "portfolio_diagnostics.md").write_text(
        render_markdown(diagnostics),
        encoding="utf-8",
    )
    print(f"Wrote {run_dir / 'portfolio_diagnostics.json'}")
    print(f"Wrote {run_dir / 'portfolio_diagnostics.md'}")


def build_diagnostics(
    run_dir: Path,
    momentum_output_root: Path,
    data_dir: Path = Path("backtests/momentum/data/raw"),
) -> dict[str, Any]:
    config = _load_config(run_dir / "optimized_portfolio_config.json")
    run_summary = _load_optional_json(run_dir / "run_summary.json")
    with (run_dir / "strategy_trades.pkl").open("rb") as fh:
        trades_by_strategy = pickle.load(fh)

    result = FamilyPortfolioBacktester(config).run(trades_by_strategy)
    price_bars = _load_mtm_price_bars(data_dir)
    diagnostic_equity = _portfolio_mtm_metrics(config, result, price_bars)
    candidates = sorted(
        [*result.trades, *result.blocked_trades],
        key=lambda trade: (_aware_utc(trade.entry_time), trade.strategy_id),
    )
    candidate_context = _candidate_context(candidates, result.trades)
    block_summary = _block_summary(result.blocked_trades, candidate_context)
    strategy_summary = _strategy_summary(candidates, result.trades, result.blocked_trades)
    outcome_diagnostics = _outcome_diagnostics(
        candidates,
        result.trades,
        result.blocked_trades,
    )
    overlap = _overlap_summary(candidates, result.blocked_trades, candidate_context)
    scenario_comparison = _scenario_comparison(config, trades_by_strategy, price_bars)
    phase_frontier = _phase_frontier(run_dir / "run_summary.json", trades_by_strategy, price_bars)
    headline_metrics = _report_metrics(result.metrics, diagnostic_equity)
    phase_progression = _phase_progression(run_summary)
    score_diagnostics = _score_diagnostics(run_summary, scenario_comparison)
    synergy_assessment = _synergy_assessment(
        headline_metrics=headline_metrics,
        outcome_diagnostics=outcome_diagnostics,
        scenario_comparison=scenario_comparison,
        score_diagnostics=score_diagnostics,
    )

    diagnostics = {
        "schema_version": "momentum_portfolio_synergy_comprehensive_diagnostics.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "round": run_summary.get("round"),
        "round_name": run_summary.get("round_name"),
        "headline": {
            "fired_trades": len(candidates),
            "accepted_trades": len(result.trades),
            "blocked_trades": len(result.blocked_trades),
            "block_rate": _safe_div(len(result.blocked_trades), len(candidates)),
            "net_profit": headline_metrics.get("net_profit", 0.0),
            "net_return_pct": headline_metrics.get("net_return_pct", 0.0),
            "profit_factor": headline_metrics.get("profit_factor", 0.0),
            "win_rate": headline_metrics.get("win_rate", 0.0),
            "max_drawdown_pct": headline_metrics.get("max_drawdown_pct", 0.0),
            "sharpe": headline_metrics.get("sharpe", 0.0),
            "sortino": headline_metrics.get("sortino", 0.0),
            "calmar": headline_metrics.get("calmar", 0.0),
            "trades_per_month": headline_metrics.get("trades_per_month", 0.0),
            "risk_basis": diagnostic_equity.get("risk_basis", "realized_daily"),
            "score": run_summary.get("final_score"),
        },
        "diagnostic_equity": diagnostic_equity,
        "implementation_safeguards": _implementation_safeguards(run_summary, result, diagnostic_equity),
        "config": _config_snapshot(config),
        "strategy_summary": strategy_summary,
        "outcome_diagnostics": outcome_diagnostics,
        "block_summary": block_summary,
        "rule_blocks": result.rule_blocks,
        "overlap_summary": overlap,
        "scenario_comparison": scenario_comparison,
        "phase_frontier": phase_frontier,
        "phase_progression": phase_progression,
        "score_diagnostics": score_diagnostics,
        "synergy_assessment": synergy_assessment,
        "drawdown_path": _drawdown_path_diagnostics(config, result, price_bars),
        "monthly_attribution": _monthly_attribution(
            result.trades,
            result.blocked_trades,
        ),
        "individual_strategy_reference": _individual_reference(
            run_dir,
            momentum_output_root,
        ),
        "interpretation": _interpret(diagnostics_inputs={
            "result": result,
            "strategy_summary": strategy_summary,
            "block_summary": block_summary,
            "overlap": overlap,
            "scenario_comparison": scenario_comparison,
        }),
    }
    return diagnostics


def _candidate_context(
    candidates: list[FamilyPortfolioTrade],
    accepted: list[FamilyPortfolioTrade],
) -> dict[int, dict[str, Any]]:
    accepted_with_ids = [(id(trade), trade) for trade in accepted]
    context: dict[int, dict[str, Any]] = {}
    for trade in candidates:
        entry = _aware_utc(trade.entry_time)
        open_positions = [
            other for other_id, other in accepted_with_ids
            if other_id != id(trade)
            and other.entry_time is not None
            and other.exit_time is not None
            and _aware_utc(other.entry_time) <= entry < _aware_utc(other.exit_time)
        ]
        same_direction = [
            other for other in open_positions if other.direction == trade.direction
        ]
        opposite_direction = [
            other for other in open_positions if other.direction != trade.direction
        ]
        context[id(trade)] = {
            "open_positions": len(open_positions),
            "same_direction_open": len(same_direction),
            "opposite_direction_open": len(opposite_direction),
            "open_strategy_counts": dict(Counter(other.strategy_id for other in open_positions)),
            "open_risk_R": sum(other.normalized_risk_R for other in open_positions),
            "same_direction_risk_R": sum(other.normalized_risk_R for other in same_direction),
            "opposite_direction_risk_R": sum(other.normalized_risk_R for other in opposite_direction),
        }
    return context


def _block_summary(
    blocked: list[FamilyPortfolioTrade],
    candidate_context: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_reason: dict[str, list[FamilyPortfolioTrade]] = defaultdict(list)
    for trade in blocked:
        by_reason[trade.denial_reason or "unknown"].append(trade)

    summary: dict[str, dict[str, Any]] = {}
    for reason, trades in sorted(by_reason.items(), key=lambda item: len(item[1]), reverse=True):
        raw_pnls = [trade.raw_pnl_dollars for trade in trades]
        r_multiples = [trade.r_multiple for trade in trades]
        open_counts = [candidate_context[id(trade)]["open_positions"] for trade in trades]
        contexts = [_portfolio_context(trade) for trade in trades]
        summary[reason] = {
            "count": len(trades),
            "by_strategy": dict(Counter(trade.strategy_id for trade in trades)),
            "raw_pnl_dollars": sum(raw_pnls),
            "raw_win_rate": _safe_div(sum(1 for pnl in raw_pnls if pnl > 0), len(raw_pnls)),
            "avg_r_multiple": _avg([trade.r_multiple for trade in trades]),
            "total_r_multiple": sum(r_multiples),
            "positive_r_total": sum(value for value in r_multiples if value > 0.0),
            "nonpositive_r_total": sum(value for value in r_multiples if value <= 0.0),
            "net_block_value_r": -sum(r_multiples),
            "positive_raw_pnl_count": sum(1 for pnl in raw_pnls if pnl > 0),
            "negative_raw_pnl_count": sum(1 for pnl in raw_pnls if pnl < 0),
            "nonpositive_raw_pnl_count": sum(1 for pnl in raw_pnls if pnl <= 0),
            "avg_open_positions_at_block": _avg(open_counts),
            "pct_with_existing_open_position": _safe_div(
                sum(1 for count in open_counts if count > 0),
                len(open_counts),
            ),
            "avg_base_qty": _avg([ctx.get("base_qty", 0.0) for ctx in contexts]),
            "avg_base_risk_R": _avg([ctx.get("base_risk_R", 0.0) for ctx in contexts]),
            "avg_base_mnq_eq": _avg([ctx.get("base_mnq_eq", 0.0) for ctx in contexts]),
            "avg_current_heat_R": _avg([ctx.get("heat_R", 0.0) for ctx in contexts]),
            "avg_current_family_mnq_eq": _avg([ctx.get("family_mnq_eq", 0.0) for ctx in contexts]),
            "pct_single_order_exceeds_heat_cap": _safe_div(
                sum(
                    1 for ctx in contexts
                    if ctx.get("heat_R", 0.0) <= 1e-9
                    and ctx.get("base_risk_R", 0.0) > ctx.get("heat_cap_R", float("inf"))
                ),
                len(contexts),
            ),
            "pct_single_order_exceeds_family_contract_cap": _safe_div(
                sum(
                    1 for ctx in contexts
                    if ctx.get("family_mnq_eq", 0.0) <= 1e-9
                    and ctx.get("family_contract_cap_mnq_eq", 0.0) > 0
                    and ctx.get("base_mnq_eq", 0.0)
                    > ctx.get("family_contract_cap_mnq_eq", float("inf"))
                ),
                len(contexts),
            ),
        }
    return summary


def _strategy_summary(
    candidates: list[FamilyPortfolioTrade],
    accepted: list[FamilyPortfolioTrade],
    blocked: list[FamilyPortfolioTrade],
) -> dict[str, dict[str, Any]]:
    fired_by_strategy = _group_by_strategy(candidates)
    accepted_by_strategy = _group_by_strategy(accepted)
    blocked_by_strategy = _group_by_strategy(blocked)

    summary: dict[str, dict[str, Any]] = {}
    for strategy_id in MOMENTUM_FAMILY_STRATEGY_IDS:
        fired = fired_by_strategy.get(strategy_id, [])
        acc = accepted_by_strategy.get(strategy_id, [])
        blk = blocked_by_strategy.get(strategy_id, [])
        accepted_raw_pnl = sum(trade.raw_pnl_dollars for trade in acc)
        accepted_contexts = [_portfolio_context(trade) for trade in acc]
        blocked_contexts = [_portfolio_context(trade) for trade in blk]
        summary[strategy_id] = {
            "fired": len(fired),
            "accepted": len(acc),
            "blocked": len(blk),
            "accept_rate": _safe_div(len(acc), len(fired)),
            "blocked_rate": _safe_div(len(blk), len(fired)),
            "accepted_adjusted_pnl": sum(trade.adjusted_pnl for trade in acc),
            "accepted_raw_pnl": accepted_raw_pnl,
            "blocked_raw_pnl": sum(trade.raw_pnl_dollars for trade in blk),
            "fired_raw_pnl": sum(trade.raw_pnl_dollars for trade in fired),
            "accepted_win_rate": _safe_div(sum(1 for trade in acc if trade.adjusted_pnl > 0), len(acc)),
            "blocked_raw_win_rate": _safe_div(sum(1 for trade in blk if trade.raw_pnl_dollars > 0), len(blk)),
            "avg_accepted_r": _avg([trade.r_multiple for trade in acc]),
            "avg_blocked_r": _avg([trade.r_multiple for trade in blk]),
            "accepted_r_distribution": _distribution(
                [trade.r_multiple for trade in acc]
            ),
            "blocked_r_distribution": _distribution(
                [trade.r_multiple for trade in blk]
            ),
            "accepted_raw_pnl_distribution": _distribution(
                [trade.raw_pnl_dollars for trade in acc]
            ),
            "blocked_raw_pnl_distribution": _distribution(
                [trade.raw_pnl_dollars for trade in blk]
            ),
            "positive_trade_block_rate": _safe_div(
                sum(1 for trade in blk if trade.r_multiple > 0.0),
                sum(1 for trade in fired if trade.r_multiple > 0.0),
            ),
            "nonpositive_trade_block_rate": _safe_div(
                sum(1 for trade in blk if trade.r_multiple <= 0.0),
                sum(1 for trade in fired if trade.r_multiple <= 0.0),
            ),
            "realized_r_discrimination": _avg([trade.r_multiple for trade in acc])
            - _avg([trade.r_multiple for trade in blk]),
            "avg_portfolio_qty": _avg([trade.portfolio_qty for trade in acc]),
            "avg_portfolio_risk_R": _avg([trade.normalized_risk_R for trade in acc]),
            "avg_accepted_base_risk_R": _avg([ctx.get("base_risk_R", 0.0) for ctx in accepted_contexts]),
            "avg_blocked_base_risk_R": _avg([ctx.get("base_risk_R", 0.0) for ctx in blocked_contexts]),
            "avg_accepted_base_qty": _avg([ctx.get("base_qty", 0.0) for ctx in accepted_contexts]),
            "avg_blocked_base_qty": _avg([ctx.get("base_qty", 0.0) for ctx in blocked_contexts]),
            "adjusted_to_raw_pnl_ratio": _safe_div(
                sum(trade.adjusted_pnl for trade in acc),
                accepted_raw_pnl,
            ),
            "block_reasons": dict(Counter(trade.denial_reason for trade in blk)),
        }
    return summary


def _distribution(values: list[float]) -> dict[str, float]:
    cleaned = np.asarray([float(value) for value in values], dtype=float)
    if len(cleaned) == 0:
        return {
            "count": 0.0,
            "positive_count": 0.0,
            "nonpositive_count": 0.0,
            "win_rate": 0.0,
            "total": 0.0,
            "positive_total": 0.0,
            "nonpositive_total": 0.0,
            "average": 0.0,
            "median": 0.0,
            "minimum": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "maximum": 0.0,
        }
    positive = cleaned[cleaned > 0.0]
    nonpositive = cleaned[cleaned <= 0.0]
    return {
        "count": float(len(cleaned)),
        "positive_count": float(len(positive)),
        "nonpositive_count": float(len(nonpositive)),
        "win_rate": float(len(positive) / len(cleaned)),
        "total": float(np.sum(cleaned)),
        "positive_total": float(np.sum(positive)) if len(positive) else 0.0,
        "nonpositive_total": float(np.sum(nonpositive)) if len(nonpositive) else 0.0,
        "average": float(np.mean(cleaned)),
        "median": float(np.median(cleaned)),
        "minimum": float(np.min(cleaned)),
        "p10": float(np.quantile(cleaned, 0.10)),
        "p25": float(np.quantile(cleaned, 0.25)),
        "p75": float(np.quantile(cleaned, 0.75)),
        "p90": float(np.quantile(cleaned, 0.90)),
        "maximum": float(np.max(cleaned)),
    }


def _outcome_diagnostics(
    candidates: list[FamilyPortfolioTrade],
    accepted: list[FamilyPortfolioTrade],
    blocked: list[FamilyPortfolioTrade],
) -> dict[str, Any]:
    accepted_r = _distribution([trade.r_multiple for trade in accepted])
    blocked_r = _distribution([trade.r_multiple for trade in blocked])
    accepted_raw_pnl = _distribution([trade.raw_pnl_dollars for trade in accepted])
    blocked_raw_pnl = _distribution([trade.raw_pnl_dollars for trade in blocked])
    candidate_positive = sum(1 for trade in candidates if trade.r_multiple > 0.0)
    candidate_nonpositive = sum(1 for trade in candidates if trade.r_multiple <= 0.0)
    avoided_loss_r = -blocked_r["nonpositive_total"]
    forgone_gain_r = blocked_r["positive_total"]
    avoided_loss_dollars = -blocked_raw_pnl["nonpositive_total"]
    forgone_gain_dollars = blocked_raw_pnl["positive_total"]
    return {
        "reconciliation": {
            "fired": len(candidates),
            "accepted": len(accepted),
            "blocked": len(blocked),
            "reconciles": len(candidates) == len(accepted) + len(blocked),
        },
        "accepted_r": accepted_r,
        "blocked_r": blocked_r,
        "accepted_raw_pnl": accepted_raw_pnl,
        "blocked_raw_pnl": blocked_raw_pnl,
        "positive_trade_block_rate": _safe_div(
            blocked_r["positive_count"],
            candidate_positive,
        ),
        "nonpositive_trade_block_rate": _safe_div(
            blocked_r["nonpositive_count"],
            candidate_nonpositive,
        ),
        "blocker_precision_nonpositive": _safe_div(
            blocked_r["nonpositive_count"],
            blocked_r["count"],
        ),
        "avoided_loss_r": avoided_loss_r,
        "forgone_gain_r": forgone_gain_r,
        "net_block_value_r": avoided_loss_r - forgone_gain_r,
        "block_efficiency_r": _safe_div(
            avoided_loss_r,
            avoided_loss_r + forgone_gain_r,
        ),
        "avoided_loss_raw_dollars": avoided_loss_dollars,
        "forgone_gain_raw_dollars": forgone_gain_dollars,
        "net_block_value_raw_dollars": avoided_loss_dollars
        - forgone_gain_dollars,
        "realized_r_discrimination": accepted_r["average"]
        - blocked_r["average"],
        "accepted_win_rate_uplift_vs_all_candidates": accepted_r["win_rate"]
        - _safe_div(candidate_positive, len(candidates)),
    }


def _overlap_summary(
    candidates: list[FamilyPortfolioTrade],
    blocked: list[FamilyPortfolioTrade],
    candidate_context: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    sorted_candidates = sorted(candidates, key=lambda trade: _aware_utc(trade.entry_time))
    within_15: Counter[str] = Counter()
    within_60: Counter[str] = Counter()
    candidate_near_15: set[int] = set()
    candidate_near_60: set[int] = set()
    blocked_ids = {id(trade) for trade in blocked}
    blocked_near_15: set[int] = set()
    blocked_near_60: set[int] = set()

    for i, left in enumerate(sorted_candidates):
        left_time = _aware_utc(left.entry_time)
        for right in sorted_candidates[i + 1:]:
            delta = _aware_utc(right.entry_time) - left_time
            if delta > timedelta(minutes=60):
                break
            pair = " / ".join(sorted((left.strategy_id, right.strategy_id)))
            candidate_near_60.update((id(left), id(right)))
            within_60[pair] += 1
            if id(left) in blocked_ids:
                blocked_near_60.add(id(left))
            if id(right) in blocked_ids:
                blocked_near_60.add(id(right))
            if delta <= timedelta(minutes=15):
                candidate_near_15.update((id(left), id(right)))
                within_15[pair] += 1
                if id(left) in blocked_ids:
                    blocked_near_15.add(id(left))
                if id(right) in blocked_ids:
                    blocked_near_15.add(id(right))

    exact_groups: dict[str, list[FamilyPortfolioTrade]] = defaultdict(list)
    for trade in candidates:
        exact_groups[_aware_utc(trade.entry_time).isoformat()].append(trade)
    exact_collision_groups = [items for items in exact_groups.values() if len(items) > 1]

    open_counts_blocked = [candidate_context[id(trade)]["open_positions"] for trade in blocked]
    same_direction_blocked = [candidate_context[id(trade)]["same_direction_open"] for trade in blocked]
    opposite_direction_blocked = [candidate_context[id(trade)]["opposite_direction_open"] for trade in blocked]

    return {
        "exact_timestamp_collision_groups": len(exact_collision_groups),
        "candidates_in_exact_timestamp_collisions": sum(len(items) for items in exact_collision_groups),
        "pct_candidates_with_other_signal_within_15m": _safe_div(len(candidate_near_15), len(candidates)),
        "pct_candidates_with_other_signal_within_60m": _safe_div(len(candidate_near_60), len(candidates)),
        "pct_blocked_with_other_signal_within_15m": _safe_div(len(blocked_near_15), len(blocked)),
        "pct_blocked_with_other_signal_within_60m": _safe_div(len(blocked_near_60), len(blocked)),
        "top_strategy_pairs_within_15m": dict(within_15.most_common(12)),
        "top_strategy_pairs_within_60m": dict(within_60.most_common(12)),
        "blocked_avg_existing_open_positions": _avg(open_counts_blocked),
        "blocked_pct_with_existing_open_position": _safe_div(
            sum(1 for count in open_counts_blocked if count > 0),
            len(open_counts_blocked),
        ),
        "blocked_avg_same_direction_open_positions": _avg(same_direction_blocked),
        "blocked_avg_opposite_direction_open_positions": _avg(opposite_direction_blocked),
    }


def _scenario_comparison(
    config: FamilyPortfolioBacktestConfig,
    trades_by_strategy: dict[str, list],
    price_bars: dict[str, Any] | None,
) -> dict[str, Any]:
    scenarios = {
        "optimized_live_rules": config,
        "same_allocations_relaxed_shared_caps": _relaxed_config(config),
        "live_rules_risk_1_5x": _scale_allocations(config, 1.5),
        "live_rules_risk_2_0x": _scale_allocations(config, 2.0),
    }
    comparison: dict[str, Any] = {}
    for name, scenario_config in scenarios.items():
        result = FamilyPortfolioBacktester(scenario_config).run(trades_by_strategy)
        diagnostic_equity = _portfolio_mtm_metrics(scenario_config, result, price_bars)
        candidates = sorted(
            [*result.trades, *result.blocked_trades],
            key=lambda trade: (_aware_utc(trade.entry_time), trade.strategy_id),
        )
        comparison[name] = {
            "metrics": _report_metrics(result.metrics, diagnostic_equity),
            "diagnostic_equity": diagnostic_equity,
            "rule_blocks": result.rule_blocks,
            "strategy_trade_counts": result.strategy_trade_counts,
            "strategy_blocked_counts": result.strategy_blocked_counts,
            "outcome_diagnostics": _outcome_diagnostics(
                candidates,
                result.trades,
                result.blocked_trades,
            ),
            "drawdown_path": _drawdown_path_diagnostics(
                scenario_config,
                result,
                price_bars,
            ),
        }
    return comparison


def _phase_frontier(
    run_summary_path: Path,
    trades_by_strategy: dict[str, list],
    price_bars: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    del trades_by_strategy, price_bars
    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for phase in summary.get("phases", []):
        for item in phase.get("evaluations", []):
            metrics = item.get("metrics", {})
            rows.append({
                "phase": phase.get("phase"),
                "name": item.get("name"),
                "description": item.get("description", ""),
                "phase_accepted": bool(phase.get("accepted")),
                "accepted_candidate": phase.get("accepted_candidate"),
                "selected": item.get("name") == phase.get("accepted_candidate"),
                "score": item.get("score", 0.0),
                "validation_score": item.get("validation_score"),
                "robust_pass": item.get("robust_pass"),
                "robust_reason": item.get("robust_reason", ""),
                "rejected": item.get("rejected", False),
                "reject_reason": item.get("reject_reason", ""),
                "net_profit": metrics.get("net_profit", 0.0),
                "trades_per_month": metrics.get("trades_per_month", 0.0),
                "total_trades": metrics.get("total_trades", 0.0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "calmar": metrics.get("calmar", 0.0),
                "block_rate": metrics.get("block_rate", 0.0),
                "validation_net_profit": item.get("validation_metrics", {}).get(
                    "net_profit"
                ),
                "validation_profit_factor": item.get("validation_metrics", {}).get(
                    "profit_factor"
                ),
                "validation_max_drawdown_pct": item.get(
                    "validation_metrics", {}
                ).get("max_drawdown_pct"),
                "soft_warnings": item.get("soft_warnings", []),
            })
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def _phase_progression(summary: dict[str, Any]) -> list[dict[str, Any]]:
    progression: list[dict[str, Any]] = []
    for phase in summary.get("phases", []):
        evaluations = phase.get("evaluations", [])
        best = max(
            evaluations,
            key=lambda row: float(row.get("score", 0.0) or 0.0),
            default={},
        )
        progression.append(
            {
                "phase": phase.get("phase"),
                "candidate_count": len(evaluations),
                "accepted": bool(phase.get("accepted")),
                "accepted_candidate": phase.get("accepted_candidate"),
                "current_score": float(phase.get("current_score", 0.0) or 0.0),
                "best_candidate": best.get("name"),
                "best_score": float(best.get("score", 0.0) or 0.0),
                "robust_pass_count": sum(
                    bool(row.get("robust_pass")) for row in evaluations
                ),
                "rejected_count": sum(bool(row.get("rejected")) for row in evaluations),
            }
        )
    return progression


def _score_diagnostics(
    run_summary: dict[str, Any],
    scenarios: dict[str, Any],
) -> dict[str, Any]:
    from backtests.momentum.auto.portfolio_synergy.family_phase_auto import (
        SCORE_WEIGHTS,
        TARGETS,
        score_metrics,
    )

    optimized = score_metrics(scenarios["optimized_live_rules"]["metrics"])
    relaxed = score_metrics(
        scenarios["same_allocations_relaxed_shared_caps"]["metrics"]
    )
    return {
        "weights": dict(SCORE_WEIGHTS),
        "targets": dict(TARGETS),
        "optimized": optimized,
        "relaxed_shared_caps": relaxed,
        "final_validation_score": run_summary.get("final_validation_score"),
        "selection_rule": (
            "weighted seven-component score; hard reject for non-positive PnL, "
            "MTM drawdown above 20%, PF below 1.35, or inactive sleeve"
        ),
        "known_gap": (
            "The immutable score penalizes aggregate block rate but has no direct "
            "component for blocker outcome discrimination or forgone winning R."
        ),
    }


def _synergy_assessment(
    *,
    headline_metrics: dict[str, Any],
    outcome_diagnostics: dict[str, Any],
    scenario_comparison: dict[str, Any],
    score_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    optimized = scenario_comparison["optimized_live_rules"]["metrics"]
    relaxed = scenario_comparison["same_allocations_relaxed_shared_caps"]["metrics"]
    profit_capture = _safe_div(
        optimized.get("net_profit", 0.0),
        relaxed.get("net_profit", 0.0),
    )
    trade_capture = _safe_div(
        optimized.get("total_trades", 0.0),
        relaxed.get("total_trades", 0.0),
    )
    criteria = {
        "block_net_value_r_nonnegative": outcome_diagnostics[
            "net_block_value_r"
        ]
        >= 0.0,
        "accepted_average_r_exceeds_blocked": outcome_diagnostics[
            "realized_r_discrimination"
        ]
        > 0.0,
        "nonpositive_block_rate_exceeds_positive_block_rate": outcome_diagnostics[
            "nonpositive_trade_block_rate"
        ]
        > outcome_diagnostics["positive_trade_block_rate"],
        "optimized_drawdown_no_worse_than_relaxed": optimized.get(
            "max_drawdown_pct", 1.0
        )
        <= relaxed.get("max_drawdown_pct", 1.0),
        "optimized_calmar_at_least_relaxed": optimized.get("calmar", 0.0)
        >= relaxed.get("calmar", 0.0),
        "optimized_profit_factor_at_least_relaxed": optimized.get(
            "profit_factor", 0.0
        )
        >= relaxed.get("profit_factor", 0.0),
        "optimized_score_at_least_relaxed": score_diagnostics["optimized"][
            "score"
        ]
        >= score_diagnostics["relaxed_shared_caps"]["score"],
        "trade_capture_at_least_90pct": trade_capture >= 0.90,
        "block_rate_within_15pct_target": optimized.get("block_rate", 1.0)
        <= 0.15,
        "frequency_at_least_40_per_month": optimized.get(
            "trades_per_month", 0.0
        )
        >= 40.0,
        "all_four_strategies_active": headline_metrics.get(
            "active_strategies", 0.0
        )
        >= 4.0,
    }
    blocker_quality_passes = all(
        criteria[key]
        for key in (
            "block_net_value_r_nonnegative",
            "accepted_average_r_exceeds_blocked",
            "nonpositive_block_rate_exceeds_positive_block_rate",
        )
    )
    risk_return_passes = all(
        criteria[key]
        for key in (
            "optimized_drawdown_no_worse_than_relaxed",
            "optimized_calmar_at_least_relaxed",
            "optimized_profit_factor_at_least_relaxed",
        )
    )
    maximized = all(criteria.values())
    if trade_capture < 0.50:
        verdict = "severe_destructive_rule_interference_not_synergistic"
    elif not blocker_quality_passes:
        verdict = "high_opportunity_capture_but_blocker_discrimination_not_synergistic"
    elif not risk_return_passes:
        verdict = "blocker_quality_positive_but_return_drawdown_synergy_not_maximized"
    else:
        verdict = "maximized_among_tested_evidence_not_global" if maximized else "partial_synergy"
    return {
        "verdict": verdict,
        "maximized": maximized,
        "profit_capture_vs_relaxed": profit_capture,
        "trade_capture_vs_relaxed": trade_capture,
        "profit_gap_vs_relaxed": optimized.get("net_profit", 0.0)
        - relaxed.get("net_profit", 0.0),
        "drawdown_delta_vs_relaxed": optimized.get("max_drawdown_pct", 0.0)
        - relaxed.get("max_drawdown_pct", 0.0),
        "calmar_delta_vs_relaxed": optimized.get("calmar", 0.0)
        - relaxed.get("calmar", 0.0),
        "profit_factor_delta_vs_relaxed": optimized.get("profit_factor", 0.0)
        - relaxed.get("profit_factor", 0.0),
        "criteria": criteria,
        "blocker_quality_passes": blocker_quality_passes,
        "risk_return_passes": risk_return_passes,
    }


def _individual_reference(
    run_dir: Path,
    momentum_output_root: Path,
) -> dict[str, Any]:
    manifest = _load_optional_json(run_dir / "strategy_trade_manifest.json")
    repo_root = momentum_output_root.resolve().parents[2]
    reference: dict[str, Any] = {}
    for strategy_id, (folder, round_name) in STRATEGY_OUTPUTS.items():
        manifest_entry = manifest.get(folder, {})
        provenance = manifest_entry.get("source_provenance", {})
        summary_info = provenance.get("run_summary", {})
        diagnostics_info = provenance.get("round_final_diagnostics", {})
        fallback_dir = momentum_output_root / folder / round_name
        run_summary, summary_resolution = _resolve_source_artifact(
            info=summary_info,
            fallback=fallback_dir / "run_summary.json",
            search_root=momentum_output_root / folder,
            repo_root=repo_root,
        )
        evaluation, diagnostics_resolution = _resolve_source_artifact(
            info=diagnostics_info,
            fallback=fallback_dir / "round_final_diagnostics.txt",
            search_root=momentum_output_root / folder,
            repo_root=repo_root,
        )
        if not run_summary.exists():
            continue
        data = json.loads(run_summary.read_text(encoding="utf-8-sig"))
        metrics = data.get("final_metrics", {})
        reference[strategy_id] = {
            "source": str(run_summary),
            "source_round": manifest_entry.get("round", round_name),
            "summary_resolution": summary_resolution,
            "diagnostics_resolution": diagnostics_resolution,
            "expected_summary_sha256": summary_info.get("sha256", ""),
            "total_trades": metrics.get("total_trades"),
            "net_profit": metrics.get("net_profit"),
            "net_return_pct": metrics.get("net_return_pct"),
            "profit_factor": metrics.get("profit_factor"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", metrics.get("max_dd_pct")),
            "trades_per_month": metrics.get("trades_per_month"),
            "win_rate": metrics.get("win_rate"),
            "avg_r": metrics.get("avg_r"),
            "calmar": metrics.get("calmar"),
            "high_value_notes": _extract_high_value_notes(evaluation),
        }
    return reference


def _resolve_source_artifact(
    *,
    info: dict[str, Any],
    fallback: Path,
    search_root: Path,
    repo_root: Path,
) -> tuple[Path, str]:
    expected_sha = str(info.get("sha256", "") or "").lower()
    raw_text = str(info.get("path", "") or "")
    raw_path = Path(raw_text) if raw_text else fallback
    direct = raw_path if raw_path.is_absolute() else repo_root / raw_path
    if raw_text and direct.is_file():
        if not expected_sha or _sha256(direct) == expected_sha:
            return direct, "manifest_direct"
    if expected_sha and search_root.exists():
        filename = raw_path.name or fallback.name
        for candidate in sorted(search_root.rglob(filename)):
            if candidate.is_file() and _sha256(candidate) == expected_sha:
                return candidate, "archived_sha256_match"
    return fallback, "configured_fallback"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mtm_price_bars(data_dir: Path) -> dict[str, Any] | None:
    candidates = (
        data_dir / "NQ_5m.parquet",
        data_dir / "MNQ_5m.parquet",
    )
    for path in candidates:
        if path.exists():
            bars = normalize_timezone(load_bars(path))
            if "close" not in bars.columns:
                return None
            return {
                "bars": bars,
                "source": str(path),
                "timeframe": "5m",
            }
    return None


def _portfolio_mtm_metrics(
    config: FamilyPortfolioBacktestConfig,
    result,
    price_bars: dict[str, Any] | None,
) -> dict[str, Any]:
    realized_daily_max_dd = result.metrics.get("max_drawdown_pct", 0.0)
    realized_final_equity = config.initial_equity + sum(trade.adjusted_pnl for trade in result.trades)
    same_timestamp_trades = [
        trade for trade in result.trades
        if trade.entry_time is not None
        and trade.exit_time is not None
        and _aware_utc(trade.entry_time) == _aware_utc(trade.exit_time)
    ]
    base = {
        "risk_basis": "realized_daily_unavailable_mtm",
        "price_source": "",
        "timeframe": "",
        "final_equity": realized_final_equity,
        "net_return_pct": realized_final_equity / config.initial_equity - 1.0,
        "realized_daily_final_equity": (
            float(result.equity_curve[-1]) if len(result.equity_curve) else realized_final_equity
        ),
        "realized_daily_max_drawdown_pct": realized_daily_max_dd,
        "realized_daily_net_return_pct": result.metrics.get("net_return_pct", 0.0),
        "realized_daily_calmar": result.metrics.get("calmar", 0.0),
        "realized_daily_points": int(len(result.equity_curve)),
        "same_timestamp_trade_count": len(same_timestamp_trades),
        "same_timestamp_adjusted_pnl": sum(trade.adjusted_pnl for trade in same_timestamp_trades),
    }
    if price_bars is None:
        return {
            **base,
            "max_drawdown_pct": realized_daily_max_dd,
            "max_drawdown_dollar": 0.0,
            "cagr": result.metrics.get("cagr", 0.0),
            "calmar": result.metrics.get("calmar", 0.0),
            "points": int(len(result.equity_curve)),
        }

    mtm_curve, mtm_timestamps = _portfolio_mtm_curve(config, result.trades, price_bars["bars"])
    if len(mtm_curve) < 2:
        return {
            **base,
            "max_drawdown_pct": realized_daily_max_dd,
            "max_drawdown_dollar": 0.0,
            "cagr": result.metrics.get("cagr", 0.0),
            "calmar": result.metrics.get("calmar", 0.0),
            "points": int(len(mtm_curve)),
        }

    max_dd_pct, max_dd_dollar = compute_max_drawdown(mtm_curve)
    years = _span_years_list(mtm_timestamps)
    cagr = compute_cagr(config.initial_equity, realized_final_equity, years)
    return {
        **base,
        "risk_basis": "bar_close_mark_to_market",
        "price_source": price_bars.get("source", ""),
        "timeframe": price_bars.get("timeframe", ""),
        "max_drawdown_pct": float(max_dd_pct),
        "max_drawdown_dollar": float(max_dd_dollar),
        "cagr": float(cagr),
        "calmar": float(cagr / max(max_dd_pct, 1e-9)),
        "points": int(len(mtm_curve)),
        "start_time": mtm_timestamps[0].isoformat() if mtm_timestamps else "",
        "end_time": mtm_timestamps[-1].isoformat() if mtm_timestamps else "",
    }


def _portfolio_mtm_curve(
    config: FamilyPortfolioBacktestConfig,
    trades: list[FamilyPortfolioTrade],
    price_bars,
) -> tuple[np.ndarray, list[datetime]]:
    accepted = [
        trade for trade in trades
        if trade.entry_time is not None
        and trade.exit_time is not None
        and trade.portfolio_approved
    ]
    if not accepted:
        return np.asarray([config.initial_equity], dtype=float), []

    start = min(_aware_utc(trade.entry_time) for trade in accepted)
    end = max(_aware_utc(trade.exit_time) for trade in accepted)
    bars = price_bars[(price_bars.index >= start) & (price_bars.index <= end)]
    if len(bars) == 0:
        return np.asarray([config.initial_equity], dtype=float), []

    events: list[tuple[datetime, int, int]] = []
    for idx, trade in enumerate(accepted):
        events.append((_aware_utc(trade.exit_time), 0, idx))
        events.append((_aware_utc(trade.entry_time), 1, idx))
    events.sort(key=lambda item: (item[0], item[1]))

    realized_equity = config.initial_equity
    open_ids: set[int] = set()
    event_idx = 0
    values: list[float] = [config.initial_equity]
    timestamps: list[datetime] = [start]
    close_series = bars["close"]

    for ts, close in close_series.items():
        ts_dt = _aware_utc(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
        while event_idx < len(events) and events[event_idx][0] <= ts_dt:
            _, event_type, trade_idx = events[event_idx]
            trade = accepted[trade_idx]
            if event_type == 0:
                open_ids.discard(trade_idx)
                realized_equity += trade.adjusted_pnl
            else:
                if _aware_utc(trade.exit_time) > ts_dt:
                    open_ids.add(trade_idx)
            event_idx += 1

        close_price = float(close)
        unrealized = sum(
            (close_price - accepted[trade_idx].entry_price)
            * accepted[trade_idx].direction
            * config.point_value
            * accepted[trade_idx].portfolio_qty
            for trade_idx in open_ids
        )
        values.append(realized_equity + unrealized)
        timestamps.append(ts_dt)

    while event_idx < len(events):
        event_time, event_type, trade_idx = events[event_idx]
        trade = accepted[trade_idx]
        if event_type == 0:
            open_ids.discard(trade_idx)
            realized_equity += trade.adjusted_pnl
            values.append(realized_equity)
            timestamps.append(event_time)
        event_idx += 1

    return np.asarray(values, dtype=float), timestamps


def _drawdown_path_diagnostics(
    config: FamilyPortfolioBacktestConfig,
    result,
    price_bars: dict[str, Any] | None,
) -> dict[str, Any]:
    if price_bars is None:
        return {
            "risk_basis": "bar_close_mtm_unavailable",
            "max_drawdown_pct": float(
                result.metrics.get("max_drawdown_pct", 0.0) or 0.0
            ),
            "peak_time": None,
            "trough_time": None,
            "recovery_time": None,
            "duration_hours": 0.0,
            "contribution_by_strategy": {},
        }
    curve, timestamps = _portfolio_mtm_curve(
        config,
        result.trades,
        price_bars["bars"],
    )
    if len(curve) < 2 or len(timestamps) != len(curve):
        return {
            "risk_basis": "bar_close_mark_to_market",
            "max_drawdown_pct": 0.0,
            "peak_time": None,
            "trough_time": None,
            "recovery_time": None,
            "duration_hours": 0.0,
            "contribution_by_strategy": {},
        }
    peaks = np.maximum.accumulate(curve)
    drawdowns = np.divide(
        peaks - curve,
        peaks,
        out=np.zeros_like(curve),
        where=peaks > 0.0,
    )
    trough_index = int(np.argmax(drawdowns))
    peak_index = int(np.argmax(curve[: trough_index + 1]))
    recovery_index = next(
        (
            index
            for index in range(trough_index + 1, len(curve))
            if curve[index] >= curve[peak_index]
        ),
        None,
    )
    peak_time = timestamps[peak_index]
    trough_time = timestamps[trough_index]
    peak_contribution = _mtm_contribution_by_strategy(
        config,
        result.trades,
        price_bars["bars"],
        peak_time,
    )
    trough_contribution = _mtm_contribution_by_strategy(
        config,
        result.trades,
        price_bars["bars"],
        trough_time,
    )
    return {
        "risk_basis": "bar_close_mark_to_market",
        "max_drawdown_pct": float(drawdowns[trough_index]),
        "peak_time": peak_time.isoformat(),
        "trough_time": trough_time.isoformat(),
        "recovery_time": (
            timestamps[recovery_index].isoformat()
            if recovery_index is not None
            else None
        ),
        "peak_equity": float(curve[peak_index]),
        "trough_equity": float(curve[trough_index]),
        "duration_hours": float(
            (trough_time - peak_time).total_seconds() / 3600.0
        ),
        "contribution_by_strategy": {
            strategy_id: float(
                trough_contribution.get(strategy_id, 0.0)
                - peak_contribution.get(strategy_id, 0.0)
            )
            for strategy_id in MOMENTUM_FAMILY_STRATEGY_IDS
        },
        "recovered_in_sample": recovery_index is not None,
    }


def _mtm_contribution_by_strategy(
    config: FamilyPortfolioBacktestConfig,
    trades: list[FamilyPortfolioTrade],
    bars,
    timestamp: datetime,
) -> dict[str, float]:
    available = bars[bars.index <= timestamp]
    close_price = float(available["close"].iloc[-1]) if len(available) else 0.0
    contribution: dict[str, float] = defaultdict(float)
    for trade in trades:
        if (
            not trade.portfolio_approved
            or trade.entry_time is None
            or trade.exit_time is None
        ):
            continue
        entry = _aware_utc(trade.entry_time)
        exit_time = _aware_utc(trade.exit_time)
        if exit_time <= timestamp:
            contribution[trade.strategy_id] += float(trade.adjusted_pnl)
        elif entry <= timestamp < exit_time and close_price > 0.0:
            contribution[trade.strategy_id] += float(
                (close_price - trade.entry_price)
                * trade.direction
                * config.point_value
                * trade.portfolio_qty
            )
    return dict(contribution)


def _monthly_attribution(
    accepted: list[FamilyPortfolioTrade],
    blocked: list[FamilyPortfolioTrade],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "accepted": 0,
            "blocked": 0,
            "adjusted_pnl": 0.0,
            "accepted_r": 0.0,
            "blocked_r": 0.0,
            "blocked_raw_pnl": 0.0,
            "by_strategy": defaultdict(
                lambda: {
                    "accepted": 0,
                    "blocked": 0,
                    "adjusted_pnl": 0.0,
                    "accepted_r": 0.0,
                    "blocked_r": 0.0,
                    "blocked_raw_pnl": 0.0,
                }
            ),
        }
    )
    for trade in accepted:
        if trade.exit_time is None:
            continue
        month = _aware_utc(trade.exit_time).strftime("%Y-%m")
        row = rows[month]
        row["accepted"] += 1
        row["adjusted_pnl"] += float(trade.adjusted_pnl)
        row["accepted_r"] += float(trade.r_multiple)
        sleeve = row["by_strategy"][trade.strategy_id]
        sleeve["accepted"] += 1
        sleeve["adjusted_pnl"] += float(trade.adjusted_pnl)
        sleeve["accepted_r"] += float(trade.r_multiple)
    for trade in blocked:
        if trade.entry_time is None:
            continue
        month = _aware_utc(trade.entry_time).strftime("%Y-%m")
        row = rows[month]
        row["blocked"] += 1
        row["blocked_r"] += float(trade.r_multiple)
        row["blocked_raw_pnl"] += float(trade.raw_pnl_dollars)
        sleeve = row["by_strategy"][trade.strategy_id]
        sleeve["blocked"] += 1
        sleeve["blocked_r"] += float(trade.r_multiple)
        sleeve["blocked_raw_pnl"] += float(trade.raw_pnl_dollars)
    return [
        {
            "month": month,
            "accepted": row["accepted"],
            "blocked": row["blocked"],
            "adjusted_pnl": row["adjusted_pnl"],
            "accepted_r": row["accepted_r"],
            "blocked_r": row["blocked_r"],
            "blocked_raw_pnl": row["blocked_raw_pnl"],
            "by_strategy": dict(row["by_strategy"]),
        }
        for month, row in sorted(rows.items())
    ]


def _report_metrics(metrics: dict[str, float], diagnostic_equity: dict[str, Any]) -> dict[str, float]:
    reported = dict(metrics)
    for key in ("net_return_pct", "max_drawdown_pct", "cagr", "calmar"):
        if key in diagnostic_equity:
            reported[key] = float(diagnostic_equity[key])
    reported["risk_basis"] = diagnostic_equity.get("risk_basis", "realized_daily")
    reported["max_drawdown_pct_mtm"] = float(diagnostic_equity.get("max_drawdown_pct", reported.get("max_drawdown_pct", 0.0)) or 0.0)
    reported["calmar_mtm"] = float(diagnostic_equity.get("calmar", reported.get("calmar", 0.0)) or 0.0)
    reported["max_drawdown_pct_realized"] = float(
        diagnostic_equity.get("realized_daily_max_drawdown_pct", metrics.get("max_drawdown_pct", 0.0)) or 0.0
    )
    reported["calmar_realized"] = float(
        diagnostic_equity.get("realized_daily_calmar", metrics.get("calmar", 0.0)) or 0.0
    )
    return reported


def _extract_high_value_notes(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    notes: list[str] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped in {
            "Signal Extraction / Alpha Capture",
            "Signal Discrimination",
            "Entry Mechanism",
            "Trade Management",
            "Overall Verdict",
        }:
            capture = True
            continue
        if capture and stripped:
            notes.append(stripped)
            capture = False
    return notes[:6]


def _interpret(diagnostics_inputs: dict[str, Any]) -> dict[str, Any]:
    result = diagnostics_inputs["result"]
    strategy_summary = diagnostics_inputs["strategy_summary"]
    block_summary = diagnostics_inputs["block_summary"]
    overlap = diagnostics_inputs["overlap"]
    scenarios = diagnostics_inputs["scenario_comparison"]

    optimized = scenarios["optimized_live_rules"]["metrics"]
    relaxed = scenarios["same_allocations_relaxed_shared_caps"]["metrics"]
    risk_2x = scenarios["live_rules_risk_2_0x"]["metrics"]
    positive_blocked = sum(
        max(0.0, reason.get("raw_pnl_dollars", 0.0))
        for reason in block_summary.values()
    )
    return {
        "is_globally_optimal": False,
        "is_local_objective_best_in_tested_phase_space": True,
        "why_not_equal_to_sum_of_individual_runs": [
            "Individual strategy diagnostics use each strategy's own sizing/account assumptions; the family replay uses one 50,000 equity ledger and per-strategy risk budgets.",
            "The live-rule replay rejected shared-book conflicts instead of allowing each strategy to maintain an independent position stack.",
            "The same-allocation relaxed-cap scenario shows the signal set has more gross opportunity than the live-rule result, but that opportunity requires accepting unrealistic concurrent heat.",
        ],
        "dominant_constraints": [
            reason for reason, _ in Counter(result.rule_blocks).most_common(4)
        ],
        "blocked_positive_raw_pnl_dollars": positive_blocked,
        "optimized_vs_relaxed_net_capture": _safe_div(
            optimized.get("net_profit", 0.0),
            relaxed.get("net_profit", 0.0),
        ),
        "optimized_vs_relaxed_trade_capture": _safe_div(
            optimized.get("total_trades", 0.0),
            relaxed.get("total_trades", 0.0),
        ),
        "risk_2x_net_profit": risk_2x.get("net_profit", 0.0),
        "risk_2x_block_rate": risk_2x.get("block_rate", 0.0),
        "frequency_target_met": optimized.get("trades_per_month", 0.0) >= 24.0,
        "blocked_with_existing_position_pct": overlap.get("blocked_pct_with_existing_open_position", 0.0),
        "strategy_most_blocked": max(
            strategy_summary.items(),
            key=lambda item: item[1].get("blocked", 0),
        )[0],
    }


def render_markdown(diagnostics: dict[str, Any]) -> str:
    headline = diagnostics["headline"]
    diagnostic_equity = diagnostics.get("diagnostic_equity", {})
    outcome = diagnostics["outcome_diagnostics"]
    assessment = diagnostics["synergy_assessment"]
    verdict = (
        "No. Live-rule synergy is not maximized: the rejected trades contain more "
        "realized R than the losses avoided, and the relaxed shared-cap counterfactual "
        "has better return efficiency."
        if not assessment["maximized"]
        else "Yes among the tested evidence, but this is not proof of a global optimum."
    )
    lines = [
        "# Momentum Family Portfolio Diagnostics",
        "",
        "## Executive Read",
        "",
        (
            f"Final local-best tested portfolio fired {headline['fired_trades']} candidates, "
            f"accepted {headline['accepted_trades']}, and blocked {headline['blocked_trades']} "
            f"({headline['block_rate']:.1%} block rate)."
        ),
        (
            f"Net profit was ${headline['net_profit']:,.2f}, return {headline['net_return_pct']:.1%}, "
            f"PF {headline['profit_factor']:.2f}, win rate {headline['win_rate']:.1%}, "
            f"bar-close MTM max DD {headline['max_drawdown_pct']:.2%}, and "
            f"{headline['trades_per_month']:.2f} trades/month."
        ),
        (
            f"Key ratios: Sharpe {headline['sharpe']:.2f}, Sortino {headline['sortino']:.2f}, "
            f"Calmar {headline['calmar']:.2f}."
        ),
        "",
        (
            "Portfolio max DD is reported on a bar-close mark-to-market basis, matching the individual "
            "momentum strategy diagnostics. The prior daily realized-only DD for this same run was "
            f"{diagnostic_equity.get('realized_daily_max_drawdown_pct', 0.0):.2%}."
        ),
        "",
        "This is a local optimum for the tested seven-component portfolio score, not proof of a global optimum.",
        "",
        "## Explicit Synergy Verdict",
        "",
        f"**Is portfolio synergy maximized? {verdict}**",
        "",
        "| Question | Evidence | Answer |",
        "|---|---|---|",
        f"| Are worse trades preferentially blocked? | Accepted-minus-blocked average R {outcome['realized_r_discrimination']:+.3f}R; net block value {outcome['net_block_value_r']:+.2f}R | {'Yes' if assessment['blocker_quality_passes'] else 'No'} |",
        f"| Are winning-trade blocks minimized? | {outcome['positive_trade_block_rate']:.2%} of all eventual winners blocked; {outcome['blocked_r']['positive_count']:.0f} blocked winners | {'Partly' if outcome['positive_trade_block_rate'] < 0.05 else 'No'} |",
        f"| Do live rules preserve opportunity? | {assessment['trade_capture_vs_relaxed']:.1%} of relaxed trades and {assessment['profit_capture_vs_relaxed']:.1%} of relaxed net profit captured | {'Yes' if assessment['trade_capture_vs_relaxed'] >= 0.90 else 'No'} |",
        f"| Do live rules improve max DD? | Optimized minus relaxed MTM DD {assessment['drawdown_delta_vs_relaxed']:+.2%} | {'Yes' if assessment['drawdown_delta_vs_relaxed'] <= 0 else 'No'} |",
        f"| Is the return/DD trade-off superior? | Calmar delta {assessment['calmar_delta_vs_relaxed']:+.2f}; PF delta {assessment['profit_factor_delta_vs_relaxed']:+.2f} | {'Yes' if assessment['risk_return_passes'] else 'No'} |",
        "",
        "The relaxed scenario is an opportunity upper bound, not a deployable recommendation: it removes shared heat, contract, concurrency and stop constraints. Its purpose is to measure how much potentially valuable alpha the live rules discard.",
        "",
        "## Portfolio Risk Basis",
        "",
        "| Basis | Max DD | Final Equity | Net Return | Calmar | Points | Source |",
        "|---|---:|---:|---:|---:|---:|---|",
        (
            f"| Bar-close MTM | {diagnostic_equity.get('max_drawdown_pct', 0.0):.2%} | "
            f"${diagnostic_equity.get('final_equity', 0.0):,.0f} | "
            f"{diagnostic_equity.get('net_return_pct', 0.0):.1%} | "
            f"{diagnostic_equity.get('calmar', 0.0):.2f} | "
            f"{diagnostic_equity.get('points', 0):.0f} | "
            f"{diagnostic_equity.get('price_source', '')} |"
        ),
        (
            f"| Daily realized legacy | {diagnostic_equity.get('realized_daily_max_drawdown_pct', 0.0):.2%} | "
            f"${diagnostic_equity.get('realized_daily_final_equity', 0.0):,.0f} | "
            f"{diagnostic_equity.get('realized_daily_net_return_pct', 0.0):.1%} | "
            f"{diagnostic_equity.get('realized_daily_calmar', 0.0):.2f} | "
            f"{diagnostic_equity.get('realized_daily_points', 0):.0f} | closed-trade daily curve |"
        ),
        "",
        "## Scenario Comparison",
        "",
        "| Scenario | Trades | Blocked | Block Rate | Net Profit | Trades/Mo | Win Rate | PF | MTM Max DD | Sharpe | Sortino | Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in diagnostics["scenario_comparison"].items():
        metrics = data["metrics"]
        lines.append(
            f"| {name} | {metrics.get('total_trades', 0):.0f} | "
            f"{metrics.get('blocked_trades', 0):.0f} | {metrics.get('block_rate', 0):.1%} | "
            f"${metrics.get('net_profit', 0):,.0f} | {metrics.get('trades_per_month', 0):.2f} | "
            f"{metrics.get('win_rate', 0):.1%} | {metrics.get('profit_factor', 0):.2f} | "
            f"{metrics.get('max_drawdown_pct', 0):.2%} | {metrics.get('sharpe', 0):.2f} | "
            f"{metrics.get('sortino', 0):.2f} | {metrics.get('calmar', 0):.2f} |"
        )

    lines.extend([
        "",
        "## Live-Rule Synergy Counterfactual",
        "",
        "| Measure | Optimized live rules | Relaxed shared caps | Optimized delta/capture |",
        "|---|---:|---:|---:|",
        f"| Net profit | ${diagnostics['scenario_comparison']['optimized_live_rules']['metrics'].get('net_profit', 0):,.0f} | ${diagnostics['scenario_comparison']['same_allocations_relaxed_shared_caps']['metrics'].get('net_profit', 0):,.0f} | {assessment['profit_capture_vs_relaxed']:.1%} captured |",
        f"| Accepted trades | {diagnostics['scenario_comparison']['optimized_live_rules']['metrics'].get('total_trades', 0):.0f} | {diagnostics['scenario_comparison']['same_allocations_relaxed_shared_caps']['metrics'].get('total_trades', 0):.0f} | {assessment['trade_capture_vs_relaxed']:.1%} captured |",
        f"| MTM max DD | {diagnostics['scenario_comparison']['optimized_live_rules']['metrics'].get('max_drawdown_pct', 0):.2%} | {diagnostics['scenario_comparison']['same_allocations_relaxed_shared_caps']['metrics'].get('max_drawdown_pct', 0):.2%} | {assessment['drawdown_delta_vs_relaxed']:+.2%} |",
        f"| Calmar | {diagnostics['scenario_comparison']['optimized_live_rules']['metrics'].get('calmar', 0):.2f} | {diagnostics['scenario_comparison']['same_allocations_relaxed_shared_caps']['metrics'].get('calmar', 0):.2f} | {assessment['calmar_delta_vs_relaxed']:+.2f} |",
        f"| Profit factor | {diagnostics['scenario_comparison']['optimized_live_rules']['metrics'].get('profit_factor', 0):.2f} | {diagnostics['scenario_comparison']['same_allocations_relaxed_shared_caps']['metrics'].get('profit_factor', 0):.2f} | {assessment['profit_factor_delta_vs_relaxed']:+.2f} |",
        f"| Immutable score | {diagnostics['score_diagnostics']['optimized']['score']:.4f} | {diagnostics['score_diagnostics']['relaxed_shared_caps']['score']:.4f} | {diagnostics['score_diagnostics']['optimized']['score'] - diagnostics['score_diagnostics']['relaxed_shared_caps']['score']:+.4f} |",
        "",
        "## Overall Blocker Discrimination",
        "",
        "| Population | Count | Win rate | Total R | Avg R | Median R | P10 | P25 | P75 | P90 | Raw PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Accepted | {outcome['accepted_r']['count']:.0f} | {outcome['accepted_r']['win_rate']:.1%} | {outcome['accepted_r']['total']:.2f}R | {outcome['accepted_r']['average']:.3f}R | {outcome['accepted_r']['median']:.3f}R | {outcome['accepted_r']['p10']:.3f} | {outcome['accepted_r']['p25']:.3f} | {outcome['accepted_r']['p75']:.3f} | {outcome['accepted_r']['p90']:.3f} | ${outcome['accepted_raw_pnl']['total']:,.0f} |",
        f"| Blocked | {outcome['blocked_r']['count']:.0f} | {outcome['blocked_r']['win_rate']:.1%} | {outcome['blocked_r']['total']:.2f}R | {outcome['blocked_r']['average']:.3f}R | {outcome['blocked_r']['median']:.3f}R | {outcome['blocked_r']['p10']:.3f} | {outcome['blocked_r']['p25']:.3f} | {outcome['blocked_r']['p75']:.3f} | {outcome['blocked_r']['p90']:.3f} | ${outcome['blocked_raw_pnl']['total']:,.0f} |",
        "",
        "| Blocker-quality measure | Value | Interpretation |",
        "|---|---:|---|",
        f"| Positive-trade block rate | {outcome['positive_trade_block_rate']:.2%} | Share of all eventual winners rejected |",
        f"| Non-positive-trade block rate | {outcome['nonpositive_trade_block_rate']:.2%} | Share of all eventual non-winners rejected |",
        f"| Blocker precision | {outcome['blocker_precision_nonpositive']:.2%} | Share of blocks that were non-positive |",
        f"| Forgone winning R | {outcome['forgone_gain_r']:.2f}R | Ex-post opportunity cost |",
        f"| Avoided losing R | {outcome['avoided_loss_r']:.2f}R | Ex-post protection |",
        f"| Net block value | {outcome['net_block_value_r']:+.2f}R | Positive means losses avoided exceeded winners forgone |",
        f"| Block efficiency | {outcome['block_efficiency_r']:.2%} | Avoided-loss share of gross blocked absolute R |",
        f"| Accepted-minus-blocked average R | {outcome['realized_r_discrimination']:+.3f}R | Positive means accepted outcomes were stronger |",
        f"| Accepted win-rate uplift | {outcome['accepted_win_rate_uplift_vs_all_candidates']:+.2%} | Accepted WR minus fired-candidate WR |",
        f"| Net block value, source raw dollars | ${outcome['net_block_value_raw_dollars']:+,.0f} | Diagnostic only; source quantities are not shared-ledger sizing |",
        "",
        "## Fired, Accepted, Blocked By Strategy",
        "",
        "| Strategy | Fired | Accepted | Blocked | Accept Rate | Accepted WR | Blocked WR | Adjusted PnL | Blocked Raw PnL | Avg Accepted R | Avg Blocked R | Good-trade block rate | Bad-trade block rate | R discrimination |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for strategy_id, row in diagnostics["strategy_summary"].items():
        lines.append(
            f"| {strategy_id} | {row['fired']} | {row['accepted']} | {row['blocked']} | "
            f"{row['accept_rate']:.1%} | {row['accepted_win_rate']:.1%} | "
            f"{row['blocked_raw_win_rate']:.1%} | ${row['accepted_adjusted_pnl']:,.0f} | "
            f"${row['blocked_raw_pnl']:,.0f} | {row['avg_accepted_r']:.2f} | {row['avg_blocked_r']:.2f} | "
            f"{row['positive_trade_block_rate']:.2%} | {row['nonpositive_trade_block_rate']:.2%} | "
            f"{row['realized_r_discrimination']:+.3f}R |"
        )

    lines.extend([
        "",
        "## Block Reasons",
        "",
        "| Reason | Count | Winners | Non-winners | Raw PnL | Raw WR | Total R | Avg R | Net block value R | Avg open positions | Main strategies |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for reason, row in diagnostics["block_summary"].items():
        main = ", ".join(f"{sid}:{count}" for sid, count in sorted(row["by_strategy"].items(), key=lambda item: item[1], reverse=True)[:4])
        lines.append(
            f"| {reason} | {row['count']} | {row['positive_raw_pnl_count']} | "
            f"{row['nonpositive_raw_pnl_count']} | ${row['raw_pnl_dollars']:,.0f} | "
            f"{row['raw_win_rate']:.1%} | {row['total_r_multiple']:.2f}R | "
            f"{row['avg_r_multiple']:.2f} | {row['net_block_value_r']:+.2f}R | "
            f"{row['avg_open_positions_at_block']:.2f} | {main} |"
        )

    overlap = diagnostics["overlap_summary"]
    lines.extend([
        "",
        "## Candidate Size Pressure",
        "",
        "| Reason | Avg Current Heat R | Avg Base Risk R | Avg Current MNQ-eq | Avg Base MNQ-eq | Single Order > Heat Cap | Single Order > Contract Cap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for reason, row in diagnostics["block_summary"].items():
        lines.append(
            f"| {reason} | {row['avg_current_heat_R']:.2f} | {row['avg_base_risk_R']:.2f} | "
            f"{row['avg_current_family_mnq_eq']:.1f} | {row['avg_base_mnq_eq']:.1f} | "
            f"{row['pct_single_order_exceeds_heat_cap']:.1%} | "
            f"{row['pct_single_order_exceeds_family_contract_cap']:.1%} |"
        )

    lines.extend([
        "",
        "## Signal Crowding",
        "",
        f"- Candidates with another family signal within 15m: {overlap['pct_candidates_with_other_signal_within_15m']:.1%}",
        f"- Candidates with another family signal within 60m: {overlap['pct_candidates_with_other_signal_within_60m']:.1%}",
        f"- Blocked candidates with an accepted position already open: {overlap['blocked_pct_with_existing_open_position']:.1%}",
        f"- Average accepted open positions at blocked entry time: {overlap['blocked_avg_existing_open_positions']:.2f}",
        "",
        "Top within-15m strategy pairs:",
    ])
    for pair, count in diagnostics["overlap_summary"]["top_strategy_pairs_within_15m"].items():
        lines.append(f"- {pair}: {count}")

    lines.extend([
        "",
        "## Drawdown Path and Sleeve Attribution",
        "",
        "| Scenario | Max DD | Peak | Trough | Recovery | Peak equity | Trough equity | Duration hours | Peak-to-trough contribution by strategy |",
        "|---|---:|---|---|---|---:|---:|---:|---|",
    ])
    for scenario_name, scenario in diagnostics["scenario_comparison"].items():
        path = scenario.get("drawdown_path", {})
        contribution = ", ".join(
            f"{strategy_id}:${value:+,.0f}"
            for strategy_id, value in path.get("contribution_by_strategy", {}).items()
        )
        lines.append(
            f"| {scenario_name} | {path.get('max_drawdown_pct', 0.0):.2%} | "
            f"{path.get('peak_time') or ''} | {path.get('trough_time') or ''} | "
            f"{path.get('recovery_time') or 'not recovered in sample'} | "
            f"${path.get('peak_equity', 0.0):,.0f} | ${path.get('trough_equity', 0.0):,.0f} | "
            f"{path.get('duration_hours', 0.0):,.1f} | {contribution} |"
        )
    lines.extend([
        "",
        "Negative sleeve contribution identifies MTM PnL lost from the portfolio drawdown peak to trough. It is path attribution, not proof that the sleeve should be removed.",
        "",
        "## Monthly Opportunity and Blocking Path",
        "",
        "| Month | Accepted | Blocked | Adjusted PnL | Accepted R | Blocked R | Blocked source raw PnL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in diagnostics["monthly_attribution"]:
        lines.append(
            f"| {row['month']} | {row['accepted']} | {row['blocked']} | "
            f"${row['adjusted_pnl']:+,.0f} | {row['accepted_r']:.2f}R | "
            f"{row['blocked_r']:.2f}R | ${row['blocked_raw_pnl']:+,.0f} |"
        )

    lines.extend([
        "",
        "## Individual Strategy Reference",
        "",
        "| Strategy | Source round | Resolution | Individual Trades | Individual Return | PF | Max DD | Trades/Mo | High-value diagnostic note |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for strategy_id, row in diagnostics["individual_strategy_reference"].items():
        notes = row.get("high_value_notes", [])
        note = notes[0] if notes else ""
        lines.append(
            f"| {strategy_id} | {row.get('source_round', '')} | {row.get('summary_resolution', '')} | "
            f"{_fmt_num(row.get('total_trades'))} | {_fmt_pct_like(row.get('net_return_pct'))} | "
            f"{_fmt_num(row.get('profit_factor'))} | {_fmt_pct(row.get('max_drawdown_pct'))} | "
            f"{_fmt_num(row.get('trades_per_month'))} | {note} |"
        )

    score_diagnostics = diagnostics["score_diagnostics"]
    optimized_score = score_diagnostics["optimized"]
    lines.extend([
        "",
        "## Immutable Score, Scaling and Known Gap",
        "",
        f"Selection rule: {score_diagnostics['selection_rule']}.",
        "",
        f"**Important:** {score_diagnostics['known_gap']}",
        "",
        "| Component | Weight | Target/scaling anchor | Optimized component | Weighted contribution | Relaxed component |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    target_by_component = {
        "expected_return": ("net_profit", "$"),
        "trade_frequency": ("trades_per_month", ""),
        "drawdown_control": ("max_drawdown_pct", "%"),
        "profit_quality": ("profit_factor", ""),
        "risk_efficiency": ("calmar", ""),
        "strategy_balance": ("min_strategy_trades", ""),
        "live_rule_health": ("max_block_rate", "%"),
    }
    for component, weight in score_diagnostics["weights"].items():
        target_key, suffix = target_by_component[component]
        target = score_diagnostics["targets"][target_key]
        if suffix == "$":
            target_text = f"${target:,.0f}"
        elif suffix == "%":
            target_text = f"{target:.1%}"
        else:
            target_text = f"{target:.2f}"
        value = optimized_score["components"][component]
        relaxed_value = score_diagnostics["relaxed_shared_caps"]["components"][component]
        lines.append(
            f"| {component} | {weight:.2f} | {target_text} | {value:.4f} | "
            f"{weight * value:.4f} | {relaxed_value:.4f} |"
        )
    lines.extend([
        "",
        f"Optimized aggregate score: {optimized_score['score']:.4f}; relaxed shared-cap score: {score_diagnostics['relaxed_shared_caps']['score']:.4f}; final validation score: {_fmt_num(score_diagnostics.get('final_validation_score'))}.",
        "",
        "Because blocker discrimination is absent from the score, the comprehensive verdict uses direct accepted-versus-blocked R diagnostics in addition to the immutable selection score.",
        "",
        "## Phase Progression",
        "",
        "| Phase | Candidates | Robust passes | Rejected | Accepted mutation | Accepted candidate | Current score | Best tested candidate | Best score |",
        "|---:|---:|---:|---:|:---:|---|---:|---|---:|",
    ])
    for row in diagnostics["phase_progression"]:
        lines.append(
            f"| {row['phase']} | {row['candidate_count']} | {row['robust_pass_count']} | "
            f"{row['rejected_count']} | {'yes' if row['accepted'] else 'no'} | "
            f"{row['accepted_candidate'] or ''} | {row['current_score']:.4f} | "
            f"{row['best_candidate'] or ''} | {row['best_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Full Tested Frontier",
        "",
        "| Phase | Candidate | Selected | Score | Validation score | Robust pass | Net Profit | Trades/Mo | Trades | PF | WR | MTM DD | Calmar | Block Rate | Validation PnL | Validation PF | Validation DD | Warnings/reason |",
        "|---:|---|:---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in diagnostics["phase_frontier"]:
        reasons = ", ".join(
            [
                *row.get("soft_warnings", []),
                *([row.get("robust_reason", "")] if row.get("robust_reason") else []),
                *([row.get("reject_reason", "")] if row.get("reject_reason") else []),
            ]
        )
        lines.append(
            f"| {row['phase']} | {row['name']} | {'yes' if row['selected'] else ''} | "
            f"{row['score']:.4f} | {_fmt_num(row.get('validation_score'))} | "
            f"{'yes' if row.get('robust_pass') else 'no'} | ${row['net_profit']:,.0f} | "
            f"{row['trades_per_month']:.2f} | {row['total_trades']:.0f} | "
            f"{row['profit_factor']:.2f} | {row['win_rate']:.1%} | "
            f"{row['max_drawdown_pct']:.2%} | {row['calmar']:.2f} | {row['block_rate']:.1%} | "
            f"{_fmt_num(row.get('validation_net_profit'))} | {_fmt_num(row.get('validation_profit_factor'))} | "
            f"{_fmt_pct(row.get('validation_max_drawdown_pct'))} | {reasons} |"
        )

    safeguards = diagnostics.get("implementation_safeguards", {})
    replay_contract = safeguards.get("replay_contract", {})
    lines.extend([
        "",
        "## Implementation Safeguards",
        "",
        "| Safeguard | Status |",
        "|---|---|",
        f"| Replay contract | {replay_contract.get('version', '')} |",
        f"| Evidence scope | {replay_contract.get('evidence_label', '')} |",
        f"| Live portfolio rules | {_yes_no(safeguards.get('live_portfolio_rule_checker_used'))} |",
        f"| Shared capital ledger | {_yes_no(safeguards.get('shared_capital_ledger_used'))} |",
        f"| Source artifact hashes recorded | {_yes_no(safeguards.get('source_artifact_hashes_recorded'))} |",
        f"| Source artifacts fingerprint | {safeguards.get('source_artifacts_fingerprint', '')} |",
        f"| Headline risk basis | {safeguards.get('headline_risk_basis', '')} |",
        f"| Decision stream status | {safeguards.get('decision_stream_status', '')} |",
        f"| Full source execution simulation | {_yes_no(safeguards.get('source_strategy_execution_simulation'))} |",
        "",
        (
            "The portfolio result is official for shared-capital sizing/routing evidence. "
            "It does not replace source-strategy live/backtest parity tests for fills, "
            "order paths, or intrabar execution."
        ),
    ])

    interp = diagnostics["interpretation"]
    lines.extend([
        "",
        "## Synergy Decision Criteria",
        "",
        "| Criterion | Status |",
        "|---|:---:|",
    ])
    for criterion, passed in assessment["criteria"].items():
        lines.append(f"| {criterion} | {'PASS' if passed else 'FAIL'} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The lower portfolio profit is not mainly because the individual strategies lost their edge. The relaxed shared-cap scenario demonstrates much more gross opportunity, but it requires position stacking that the live engine should not allow.",
        "- The current local optimum is mainly a capital/risk-budget and simultaneous-signal problem: high-value signals cluster, then the live heat, directional, contract, and per-strategy concurrency rules decide which one gets the slot.",
        f"- Optimized live rules captured {interp['optimized_vs_relaxed_net_capture']:.1%} of relaxed-cap net profit and {interp['optimized_vs_relaxed_trade_capture']:.1%} of relaxed-cap trades.",
        f"- The most blocked strategy was {interp['strategy_most_blocked']}.",
    ])
    if interp["frequency_target_met"]:
        lines.append("- Frequency clears the 24 trades/month target; the remaining improvement problem is alpha per accepted slot and reducing avoidable max-concurrent blocks.")
    else:
        lines.append("- Frequency remains below target; pushing it materially higher needs either better signal staggering/ranking or a deliberate increase in allowed shared heat, not independent-account recombination.")
    lines.extend([
        "",
        "## Final Answer to the Portfolio-Synergy Objective",
        "",
        f"1. **Opportunity capture:** live rules retain {assessment['trade_capture_vs_relaxed']:.1%} of relaxed trades and {assessment['profit_capture_vs_relaxed']:.1%} of relaxed net profit. The latter is a diagnostic upper-bound comparison because relaxed compounding violates deployable shared caps.",
        f"2. **Blocking worse trades:** {'passes' if assessment['blocker_quality_passes'] else 'fails'}. Net block value is {outcome['net_block_value_r']:+.2f}R and accepted-minus-blocked average R is {outcome['realized_r_discrimination']:+.3f}R.",
        f"3. **Minimizing good-trade blocks:** {outcome['blocked_r']['positive_count']:.0f} winners were blocked, representing {outcome['positive_trade_block_rate']:.2%} of all eventual winners. Blocker precision is {outcome['blocker_precision_nonpositive']:.2%}.",
        f"4. **Max-drawdown trade-off:** live rules change MTM DD by {assessment['drawdown_delta_vs_relaxed']:+.2%}, but Calmar changes by {assessment['calmar_delta_vs_relaxed']:+.2f}; the return/DD trade-off {'passes' if assessment['risk_return_passes'] else 'does not pass'}.",
        f"5. **Maximized synergy:** {'supported only within the tested evidence, not globally' if assessment['maximized'] else 'not established'}. Verdict: `{assessment['verdict']}`.",
        "",
        "The correct use of this result is as shared-capital sizing/routing evidence. A subsequent optimization should add a frozen blocker-discrimination term to the score, rank simultaneous candidates using causal expected value, and validate that the admitted set has higher R than the rejected set without weakening live heat and contract safeguards.",
    ])
    lines.append("")
    return "\n".join(lines)


def _load_config(path: Path) -> FamilyPortfolioBacktestConfig:
    return family_config_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _implementation_safeguards(
    run_summary: dict[str, Any],
    result,
    diagnostic_equity: dict[str, Any],
) -> dict[str, Any]:
    replay_contract = dict(run_summary.get("replay_contract", {}))
    if not replay_contract:
        replay_contract = dict(result.replay_bundle_metadata.get("replay_contract", {}))
    source_artifacts_fingerprint = str(run_summary.get("source_artifacts_fingerprint", ""))
    risk_basis = diagnostic_equity.get("risk_basis", "realized_daily")
    return {
        "replay_contract": replay_contract,
        "completed_trade_replay_labeled": (
            replay_contract.get("evidence_label")
            == "portfolio_sizing_evidence_not_full_source_execution_simulation"
        ),
        "live_portfolio_rule_checker_used": bool(replay_contract.get("uses_live_portfolio_rules")),
        "shared_capital_ledger_used": bool(replay_contract.get("uses_shared_capital_ledger")),
        "source_artifact_hashes_recorded": bool(source_artifacts_fingerprint),
        "source_artifacts_fingerprint": source_artifacts_fingerprint,
        "headline_risk_basis": risk_basis,
        "mtm_risk_is_headline": risk_basis == "bar_close_mark_to_market",
        "decision_stream_status": replay_contract.get("decision_stream_status", ""),
        "source_strategy_execution_simulation": bool(
            replay_contract.get("source_strategy_execution_simulation")
        ),
    }


def _config_snapshot(config: FamilyPortfolioBacktestConfig) -> dict[str, Any]:
    return {
        "initial_equity": config.initial_equity,
        "reference_unit_risk_dollars": config.reference_unit_risk_dollars,
        "heat_cap_R": config.heat_cap_R,
        "max_total_positions": config.max_total_positions,
        "portfolio_daily_stop_R": config.portfolio_daily_stop_R,
        "portfolio_weekly_stop_R": config.portfolio_weekly_stop_R,
        "directional_cap_long_R": config.rules.directional_cap_long_R,
        "directional_cap_short_R": config.rules.directional_cap_short_R,
        "max_family_contracts_mnq_eq": config.rules.max_family_contracts_mnq_eq,
        "dynamic_risk": {
            "enabled": config.dynamic_risk.enabled,
            "strategy_multipliers": dict(config.dynamic_risk.strategy_multipliers),
            "fit_to_remaining_heat": config.dynamic_risk.fit_to_remaining_heat,
            "fit_to_remaining_directional_cap": config.dynamic_risk.fit_to_remaining_directional_cap,
            "fit_to_remaining_family_cap": config.dynamic_risk.fit_to_remaining_family_cap,
            "min_qty": config.dynamic_risk.min_qty,
            "min_trade_risk_R": config.dynamic_risk.min_trade_risk_R,
            "max_trade_risk_R": config.dynamic_risk.max_trade_risk_R,
            "heat_pressure_threshold": config.dynamic_risk.heat_pressure_threshold,
            "heat_pressure_mult": config.dynamic_risk.heat_pressure_mult,
            "same_direction_pressure_threshold": config.dynamic_risk.same_direction_pressure_threshold,
            "same_direction_pressure_mult": config.dynamic_risk.same_direction_pressure_mult,
            "existing_position_mult": config.dynamic_risk.existing_position_mult,
            "daily_loss_threshold_R": config.dynamic_risk.daily_loss_threshold_R,
            "daily_loss_mult": config.dynamic_risk.daily_loss_mult,
        },
        "allocations": {
            allocation.strategy_id: {
                "base_risk_pct": allocation.base_risk_pct,
                "daily_stop_R": allocation.daily_stop_R,
                "max_concurrent": allocation.max_concurrent,
                "priority": allocation.priority,
            }
            for allocation in config.strategy_allocations
        },
    }


def _relaxed_config(config: FamilyPortfolioBacktestConfig) -> FamilyPortfolioBacktestConfig:
    allocations = tuple(
        replace(allocation, daily_stop_R=999.0, max_concurrent=999)
        for allocation in config.strategy_allocations
    )
    rules = replace(
        config.rules,
        nqdtc_direction_filter_enabled=False,
        directional_cap_R=0.0,
        directional_cap_long_R=0.0,
        directional_cap_short_R=0.0,
        max_family_contracts_mnq_eq=0,
        priority_headroom_R=0.0,
        dd_tiers=((1.0, 1.0),),
    )
    return replace(
        config,
        strategy_allocations=allocations,
        rules=rules,
        heat_cap_R=999.0,
        portfolio_daily_stop_R=999.0,
        portfolio_weekly_stop_R=999.0,
        max_total_positions=999,
    )


def _scale_allocations(
    config: FamilyPortfolioBacktestConfig,
    multiplier: float,
) -> FamilyPortfolioBacktestConfig:
    allocations = tuple(
        replace(allocation, base_risk_pct=allocation.base_risk_pct * multiplier)
        for allocation in config.strategy_allocations
    )
    return replace(config, strategy_allocations=allocations)


def _group_by_strategy(trades: list[FamilyPortfolioTrade]) -> dict[str, list[FamilyPortfolioTrade]]:
    grouped: dict[str, list[FamilyPortfolioTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.strategy_id].append(trade)
    return grouped


def _portfolio_context(trade: FamilyPortfolioTrade) -> dict[str, float]:
    context = trade.metadata.get("portfolio_entry_context", {})
    if not isinstance(context, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in context.items()
        if isinstance(value, (int, float))
    }


def _aware_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _span_years_list(timestamps: list[datetime]) -> float:
    if len(timestamps) < 2:
        return 0.0
    start = _aware_utc(timestamps[0])
    end = _aware_utc(timestamps[-1])
    return max((end - start).total_seconds() / (365.25 * 24 * 60 * 60), 0.0)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _fmt_num(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.1%}"


def _fmt_pct_like(value: Any) -> str:
    if value is None:
        return ""
    value = float(value)
    return f"{value:.1f}%" if abs(value) > 5 else f"{value:.1%}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
