from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.shared.auto.phase_state import _atomic_write_json


PHASE_FILES = (
    (1, "Risk allocation", "phase_1_risk_allocation.json"),
    (2, "Shared capacity", "phase_2_capacity.json"),
    (3, "Alpha admission", "phase_3_alpha_admission.json"),
    (4, "Reserve governors", "phase_4_reserve_governors.json"),
    (5, "Interactions", "phase_5_interactions.json"),
)
STRATEGIES = ("ALCB_R3", "IARIC_RESIDUAL_R3")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _pct(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}%}"


def _num(value: Any, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def _money(value: Any) -> str:
    return f"${float(value):+,.0f}"


def _usd(value: Any) -> str:
    return f"${float(value):,.0f}"


def _bool(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _metric(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, 0.0) or 0.0)


def _performance_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {_pct(_metric(metrics, 'net_return_pct'))} | "
        f"{_money(_metric(metrics, 'net_pnl'))} | {_num(_metric(metrics, 'total_r'))}R | "
        f"{_num(_metric(metrics, 'total_r_per_month'))}R | "
        f"{_num(_metric(metrics, 'active_trades_per_month'), 1)} | "
        f"{int(_metric(metrics, 'total_trades')):,} | "
        f"{_num(_metric(metrics, 'profit_factor'), 3)} | "
        f"{_pct(_metric(metrics, 'win_rate'))} | "
        f"{_pct(_metric(metrics, 'max_drawdown_pct_mtm_daily'))} | "
        f"{_num(_metric(metrics, 'sharpe'))} | {_num(_metric(metrics, 'sortino'))} | "
        f"{_num(_metric(metrics, 'calmar'))} | "
        f"{_num(_metric(metrics, 'certainty_equivalent_growth'), 4)} |"
    )


def _frontier_row(
    phase: int,
    selected: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    aggregate = row["aggregate"]
    components = row["aggregate_score_components"]
    return {
        "phase": phase,
        "name": row["name"],
        "selected": row["name"] == selected,
        "eligible": bool(row["eligible"]),
        "robust_score": float(row["robust_score"]),
        "aggregate_score": float(components["score"]),
        "return": _metric(aggregate, "net_return_pct"),
        "net_pnl": _metric(aggregate, "net_pnl"),
        "total_r_per_month": _metric(aggregate, "total_r_per_month"),
        "trades_per_month": _metric(aggregate, "active_trades_per_month"),
        "profit_factor": _metric(aggregate, "profit_factor"),
        "win_rate": _metric(aggregate, "win_rate"),
        "max_drawdown": _metric(aggregate, "max_drawdown_pct_mtm_daily"),
        "accept_rate": _metric(aggregate, "entry_accept_rate"),
        "block_rate": 1.0 - _metric(aggregate, "entry_accept_rate"),
        "blocked_total_r": _metric(aggregate, "blocked_total_r"),
        "synergy_ce_delta": float(components["synergy_ce_delta"]),
        "median_fold_synergy_ce_delta": float(row["median_synergy_ce_delta"]),
        "negative_synergy_folds": int(row["negative_synergy_folds"]),
        "positive_folds": int(row["positive_folds"]),
        "median_fold_profit_factor": float(row["median_profit_factor"]),
        "worst_fold_drawdown": float(row["worst_fold_mtm_drawdown"]),
    }


def _phase_diagnostics(round_dir: Path) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    selected_folds: list[dict[str, Any]] = []
    selected_score_components: dict[str, float] = {}
    for phase_number, title, filename in PHASE_FILES:
        payload = _load(round_dir / filename)
        rows = payload["results"]
        selected_name = payload["selected"]
        selected = next(row for row in rows if row["name"] == selected_name)
        candidates = [
            _frontier_row(phase_number, selected_name, row) for row in rows
        ]
        candidates.sort(key=lambda row: row["robust_score"], reverse=True)
        frontier.extend(candidates)
        phases.append(
            {
                "phase": phase_number,
                "title": title,
                "candidate_count": len(rows),
                "eligible_count": sum(bool(row["eligible"]) for row in rows),
                "selected": selected_name,
                "winner": _frontier_row(phase_number, selected_name, selected),
            }
        )
        if phase_number == 5:
            selected_score_components = {
                key: float(value)
                for key, value in selected["aggregate_score_components"].items()
            }
            for fold_name, fold in selected["folds"].items():
                metrics = fold["metrics"]
                no_overlay = fold["no_overlay_metrics"]
                components = fold["score_components"]
                selected_folds.append(
                    {
                        "fold": fold_name,
                        "return": _metric(metrics, "net_return_pct"),
                        "net_pnl": _metric(metrics, "net_pnl"),
                        "total_r_per_month": _metric(metrics, "total_r_per_month"),
                        "trades_per_month": _metric(metrics, "active_trades_per_month"),
                        "profit_factor": _metric(metrics, "profit_factor"),
                        "max_drawdown": _metric(
                            metrics,
                            "max_drawdown_pct_mtm_daily",
                        ),
                        "accept_rate": _metric(metrics, "entry_accept_rate"),
                        "no_overlay_return": _metric(no_overlay, "net_return_pct"),
                        "no_overlay_max_drawdown": _metric(
                            no_overlay,
                            "max_drawdown_pct_mtm_daily",
                        ),
                        "synergy_ce_delta": float(components["synergy_ce_delta"]),
                        "score": float(components["score"]),
                    }
                )
    return {
        "phases": phases,
        "tested_candidate_count": len(frontier),
        "frontier": frontier,
        "selected_folds": selected_folds,
        "selected_score_components": selected_score_components,
    }


def _window_synergy(window: dict[str, Any]) -> dict[str, float]:
    post = window["post_optimization_portfolio"]
    control = window["post_optimization_no_overlay"]
    pre = window["pre_optimization_portfolio"]
    alcb_native = window["alcb_round3_standalone_native_risk"]
    iaric_native = window["iaric_round3_standalone_native_risk"]
    alcb_post = window["alcb_standalone_post_risk"]
    iaric_post = window["iaric_standalone_post_risk"]
    native_sum_r = _metric(alcb_native, "total_r") + _metric(iaric_native, "total_r")
    post_sum_r = _metric(alcb_post, "total_r") + _metric(iaric_post, "total_r")
    native_sum_return = _metric(alcb_native, "net_return_pct") + _metric(
        iaric_native,
        "net_return_pct",
    )
    post_sum_return = _metric(alcb_post, "net_return_pct") + _metric(
        iaric_post,
        "net_return_pct",
    )
    return {
        "portfolio_r_capture_vs_native_standalones": _safe_div(
            _metric(post, "total_r"),
            native_sum_r,
        ),
        "portfolio_r_capture_vs_post_risk_standalones": _safe_div(
            _metric(post, "total_r"),
            post_sum_r,
        ),
        "portfolio_return_minus_native_standalone_sum": _metric(
            post,
            "net_return_pct",
        )
        - native_sum_return,
        "portfolio_return_minus_post_risk_standalone_sum": _metric(
            post,
            "net_return_pct",
        )
        - post_sum_return,
        "overlay_return_delta": _metric(post, "net_return_pct")
        - _metric(control, "net_return_pct"),
        "overlay_total_r_delta": _metric(post, "total_r")
        - _metric(control, "total_r"),
        "overlay_profit_factor_delta": _metric(post, "profit_factor")
        - _metric(control, "profit_factor"),
        "overlay_win_rate_delta": _metric(post, "win_rate")
        - _metric(control, "win_rate"),
        "overlay_drawdown_delta": _metric(post, "max_drawdown_pct_mtm_daily")
        - _metric(control, "max_drawdown_pct_mtm_daily"),
        "overlay_ce_delta": _metric(post, "certainty_equivalent_growth")
        - _metric(control, "certainty_equivalent_growth"),
        "post_minus_pre_return": _metric(post, "net_return_pct")
        - _metric(pre, "net_return_pct"),
        "post_minus_pre_drawdown": _metric(post, "max_drawdown_pct_mtm_daily")
        - _metric(pre, "max_drawdown_pct_mtm_daily"),
        "post_minus_pre_ce": _metric(post, "certainty_equivalent_growth")
        - _metric(pre, "certainty_equivalent_growth"),
        "post_drawdown_minus_worst_native_standalone": _metric(
            post,
            "max_drawdown_pct_mtm_daily",
        )
        - max(
            _metric(alcb_native, "max_drawdown_pct_mtm_daily"),
            _metric(iaric_native, "max_drawdown_pct_mtm_daily"),
        ),
    }


def _synergy_assessment(
    matched: dict[str, Any],
    phase_zero: dict[str, Any],
    phase_diagnostics: dict[str, Any],
    alpha: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    is_synergy = _window_synergy(matched["is"])
    oos_synergy = _window_synergy(matched["oos"])
    is_detail = matched["is"].get("detailed_diagnostics", {}).get(
        "post_optimization_portfolio",
        {},
    )
    oos_detail = matched["oos"].get("detailed_diagnostics", {}).get(
        "post_optimization_portfolio",
        {},
    )
    alpha_selected = alpha["selected"]
    alpha_stable = bool(alpha["variants"][alpha_selected]["stable"])
    criteria = {
        "is_overlay_improves_certainty_equivalent_growth": is_synergy[
            "overlay_ce_delta"
        ]
        > 0.0,
        "oos_overlay_improves_certainty_equivalent_growth": oos_synergy[
            "overlay_ce_delta"
        ]
        > 0.0,
        "is_overlay_reduces_drawdown": is_synergy["overlay_drawdown_delta"] < 0.0,
        "oos_overlay_reduces_drawdown": oos_synergy["overlay_drawdown_delta"] < 0.0,
        "is_blocking_has_positive_net_value": float(
            is_detail.get("net_block_value_r", 0.0)
        )
        > 0.0,
        "oos_blocking_has_nonnegative_net_value": float(
            oos_detail.get("net_block_value_r", 0.0)
        )
        >= 0.0,
        "is_accepted_trades_outperform_blocked": float(
            is_detail.get("realized_r_discrimination", 0.0)
        )
        > 0.0,
        "oos_accepted_trades_outperform_blocked": float(
            oos_detail.get("realized_r_discrimination", 0.0)
        )
        > 0.0,
        "is_post_drawdown_no_worse_than_pre": is_synergy[
            "post_minus_pre_drawdown"
        ]
        <= 0.0,
        "oos_post_drawdown_no_worse_than_pre": oos_synergy[
            "post_minus_pre_drawdown"
        ]
        <= 0.0,
        "alpha_rank_model_stable": alpha_stable,
        "interaction_selection_pbo_within_limit": float(
            selection["cscv_pbo"]["probability_backtest_overfit"]
        )
        <= float(selection["maximum_probability_backtest_overfit"]),
        "all_selected_folds_profitable": all(
            float(row["net_pnl"]) > 0.0
            for row in phase_diagnostics["selected_folds"]
        ),
    }
    strong = all(criteria.values())
    is_support = (
        criteria["is_overlay_improves_certainty_equivalent_growth"]
        and criteria["is_overlay_reduces_drawdown"]
        and criteria["is_blocking_has_positive_net_value"]
        and criteria["is_accepted_trades_outperform_blocked"]
    )
    verdict = (
        "maximized_among_tested_candidates_not_global"
        if strong
        else "modest_is_synergy_not_oos_validated_or_maximized"
        if is_support
        else "synergy_not_demonstrated"
    )
    return {
        "verdict": verdict,
        "maximized": strong,
        "is": is_synergy,
        "oos": oos_synergy,
        "criteria": criteria,
        "daily_r_correlation": float(
            phase_zero["daily_and_weekly_return_correlation"]["daily_R_correlation"]
        ),
        "weekly_r_correlation": float(
            phase_zero["daily_and_weekly_return_correlation"]["weekly_R_correlation"]
        ),
        "interpretation": (
            "The sleeves are close to uncorrelated, so combining them preserves broad "
            "opportunity. The overlay adds modest IS value and reduces matched-risk IS "
            "drawdown, but produces no OOS increment; the only OOS capacity block rejects "
            "a winner. Post risk also has higher drawdown than the native-risk baseline. "
            "High interaction PBO and an unstable alpha rank model prevent a claim that "
            "routing has maximized synergy."
        ),
    }


def enrich_summary(round_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    matched = summary["matched_performance"]
    phase_zero = _load(round_dir / "phase_0_matched_baselines.json")
    phases = _phase_diagnostics(round_dir)
    alpha = _load(round_dir / "alpha_calibration.json")
    run_spec = _load(round_dir / "run_spec.json")
    selection = summary["selection"]
    summary["schema"] = "stock_portfolio_synergy_final_diagnostics_v2"
    summary["comprehensive_diagnostics"] = {
        "evidence_scope": {
            "headline_contract": matched["contract"],
            "decision_stream": "completed_source_trade_replay",
            "raw_signal_cosimulation": False,
            "point_in_time_universe": False,
            "global_optimum_proven": False,
            "causal_marks": True,
            "shared_account": True,
        },
        "immutable_score": run_spec["immutable_score"],
        "alpha_calibration": alpha,
        "phase_zero": phase_zero,
        "phase_diagnostics": phases,
        "window_diagnostics": {
            "is": matched["is"].get("detailed_diagnostics", {}),
            "oos": matched["oos"].get("detailed_diagnostics", {}),
        },
        "synergy_assessment": _synergy_assessment(
            matched,
            phase_zero,
            phases,
            alpha,
            selection,
        ),
    }
    return summary


def _append_performance_table(
    lines: list[str],
    rows: list[tuple[str, dict[str, Any]]],
) -> None:
    lines.extend(
        [
            "| Scenario | Return | Net PnL | Total R | R/month | Trades/month | Trades | PF | Win rate | Daily-MTM DD | Sharpe | Sortino | Calmar | CE growth |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_performance_row(label, metrics) for label, metrics in rows)


def _append_blocking_section(
    lines: list[str],
    window_name: str,
    detail: dict[str, Any],
) -> None:
    if not detail:
        lines.extend([f"### {window_name}", "", "Detailed replay diagnostics unavailable.", ""])
        return
    accepted = detail["accepted_r"]
    blocked = detail["blocked_r"]
    lines.extend(
        [
            f"### {window_name}",
            "",
            "| Population | Count | Win rate | Total R | Avg R | Median R | P10 | P25 | P75 | P90 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| Accepted | {int(accepted['count']):,} | {_pct(accepted['win_rate'])} | {_num(accepted['total'], 3)}R | {_num(accepted['average'], 3)}R | {_num(accepted['median'], 3)}R | {_num(accepted['p10'], 3)} | {_num(accepted['p25'], 3)} | {_num(accepted['p75'], 3)} | {_num(accepted['p90'], 3)} |",
            f"| Blocked | {int(blocked['count']):,} | {_pct(blocked['win_rate'])} | {_num(blocked['total'], 3)}R | {_num(blocked['average'], 3)}R | {_num(blocked['median'], 3)}R | {_num(blocked['p10'], 3)} | {_num(blocked['p25'], 3)} | {_num(blocked['p75'], 3)} | {_num(blocked['p90'], 3)} |",
            "",
            "| Blocker-quality measure | Value | Interpretation |",
            "|---|---:|---|",
            f"| Positive-trade block rate | {_pct(detail['positive_trade_block_rate'])} | Share of all eventual winners rejected |",
            f"| Non-positive-trade block rate | {_pct(detail['nonpositive_trade_block_rate'])} | Share of all eventual non-winners rejected |",
            f"| Blocker precision | {_pct(detail['blocker_precision_nonpositive'])} | Share of blocked trades that were non-positive |",
            f"| Forgone winning R | {_num(detail['forgone_gain_r'], 3)}R | Ex-post opportunity cost |",
            f"| Avoided losing R | {_num(detail['avoided_loss_r'], 3)}R | Ex-post protection |",
            f"| Net block value | {_num(detail['net_block_value_r'], 3)}R | Positive means losses avoided exceeded winners forgone |",
            f"| Block efficiency | {_pct(detail['block_efficiency'])} | Avoided-loss share of gross blocked absolute R |",
            f"| Accepted-minus-blocked avg R | {_num(detail['realized_r_discrimination'], 3)}R | Positive means realized outcomes were better among accepted trades |",
            f"| Accepted-minus-blocked ex-ante quality | {_num(detail['quality_discrimination'], 4)} | Positive means the decision-time rank favored accepted trades |",
            "",
            "#### Fired, accepted and blocked by strategy",
            "",
            "| Strategy | Fired | Accepted | Blocked | Accept rate | Accepted WR | Blocked WR | Accepted avg R | Blocked avg R | Good-trade block rate | Bad-trade block rate | R discrimination | Quality discrimination |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy, row in detail["by_strategy"].items():
        lines.append(
            f"| {strategy} | {int(row['fired']):,} | {int(row['accepted']):,} | {int(row['blocked']):,} | {_pct(row['accept_rate'])} | {_pct(row['accepted_r']['win_rate'])} | {_pct(row['blocked_r']['win_rate'])} | {_num(row['accepted_r']['average'], 3)}R | {_num(row['blocked_r']['average'], 3)}R | {_pct(row['positive_trade_block_rate'])} | {_pct(row['nonpositive_trade_block_rate'])} | {_num(row['realized_r_discrimination'], 3)}R | {_num(row['quality_discrimination'], 4)} |"
        )
    lines.extend(
        [
            "",
            "#### Block reasons",
            "",
            "| Reason | Count | Win rate | Positive | Non-positive | Total R | Avg R | Avg ex-ante quality | Avg requested notional | Avg heat R | Strategy counts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for reason, row in detail["block_reasons"].items():
        r_dist = row["r"]
        strategies = ", ".join(
            f"{key}:{value}" for key, value in row["strategies"].items()
        )
        lines.append(
            f"| `{reason}` | {int(r_dist['count']):,} | {_pct(r_dist['win_rate'])} | {int(r_dist['positive_count']):,} | {int(r_dist['nonpositive_count']):,} | {_num(r_dist['total'], 3)}R | {_num(r_dist['average'], 3)}R | {_num(row['quality']['average'], 4)} | {_usd(row['average_requested_notional'])} | {_num(row['average_heat_r'], 2)} | {strategies} |"
        )
    capacity = detail["capacity_context"]
    crowding = detail["signal_crowding"]
    lines.extend(
        [
            "",
            "#### Capacity and signal crowding",
            "",
            f"- Blocked with any accepted position open: {_pct(capacity['blocked_with_any_accepted_position_open_rate'])}; with the other strategy open: {_pct(capacity['blocked_with_other_strategy_open_rate'])}.",
            f"- Same-symbol / same-sector open at block: {_pct(capacity['blocked_with_same_symbol_open_rate'])} / {_pct(capacity['blocked_with_same_sector_open_rate'])}.",
            f"- Average / maximum open positions at a block: {_num(capacity['average_open_positions_at_block'], 2)} / {int(capacity['maximum_open_positions_at_block'])}.",
            f"- Cross-strategy candidates at the exact same timestamp: {int(crowding['exact_timestamp_cross_strategy_count']):,} ({_pct(crowding['exact_timestamp_cross_strategy_rate'])}); same-symbol exact collisions: {int(crowding['exact_timestamp_same_symbol_cross_strategy_count']):,} ({_pct(crowding['exact_timestamp_same_symbol_cross_strategy_rate'])}).",
            f"- Candidates with an other-strategy entry within one day: {int(crowding['within_one_day_cross_strategy_count']):,} ({_pct(crowding['within_one_day_cross_strategy_rate'])}).",
            "",
        ]
    )


def _append_monthly(
    lines: list[str],
    window_name: str,
    detail: dict[str, Any],
) -> None:
    monthly = detail.get("monthly", [])
    lines.extend(
        [
            f"### {window_name}",
            "",
            "| Month | Accepted | Blocked | Accepted R | Blocked R | PnL | ALCB accepted / R / PnL | IARIC accepted / R / PnL |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in monthly:
        alcb = row["by_strategy"].get("ALCB_R3", {})
        iaric = row["by_strategy"].get("IARIC_RESIDUAL_R3", {})

        def sleeve(value: dict[str, Any]) -> str:
            return (
                f"{int(value.get('accepted', 0))} / "
                f"{_num(value.get('accepted_r', 0.0), 2)}R / "
                f"{_money(value.get('pnl', 0.0))}"
            )

        lines.append(
            f"| {row['month']} | {int(row['accepted'])} | {int(row['blocked'])} | {_num(row['accepted_r'], 2)}R | {_num(row['blocked_r'], 2)}R | {_money(row['pnl'])} | {sleeve(alcb)} | {sleeve(iaric)} |"
        )
    lines.append("")


def render_diagnostics(summary: dict[str, Any]) -> str:
    matched = summary["matched_performance"]
    comprehensive = summary["comprehensive_diagnostics"]
    assessment = comprehensive["synergy_assessment"]
    is_data = matched["is"]
    oos_data = matched["oos"]
    is_detail = comprehensive["window_diagnostics"]["is"].get(
        "post_optimization_portfolio",
        {},
    )
    oos_detail = comprehensive["window_diagnostics"]["oos"].get(
        "post_optimization_portfolio",
        {},
    )
    selection = summary["selection"]
    config = summary["optimized_config"]
    account = config["account_rules"]
    rules = config["portfolio_rules"]
    cross = config["cross_strategy_rules"]
    phases = comprehensive["phase_diagnostics"]
    alpha = comprehensive["alpha_calibration"]
    robustness = summary["robustness"]
    gates = summary["promotion_decision"]["gates"]
    receipt = summary["stream_receipt"]
    post_is = is_data["post_optimization_portfolio"]
    post_oos = oos_data["post_optimization_portfolio"]
    is_synergy = assessment["is"]
    oos_synergy = assessment["oos"]

    verdict_text = (
        "No. The retained configuration is a realistic local research baseline with "
        "modest positive IS overlay synergy, but synergy is not demonstrated OOS and "
        "has not been maximized."
        if not assessment["maximized"]
        else "Yes among the tested candidate set, but this is not proof of a global optimum."
    )
    lines = [
        "# Comprehensive final stock portfolio-synergy diagnostics — Round 1",
        "",
        "Status: **active realistic research baseline; production activation is not approved**.",
        "",
        "## Executive verdict",
        "",
        f"**Is strategy synergy maximized? {verdict_text}**",
        "",
        assessment["interpretation"],
        "",
        "| Question | IS evidence | OOS evidence | Answer |",
        "|---|---|---|---|",
        f"| Does the overlay add risk-adjusted value? | CE {_num(is_synergy['overlay_ce_delta'], 4)}, return {_pct(is_synergy['overlay_return_delta'])}, DD {_pct(is_synergy['overlay_drawdown_delta'])} | CE {_num(oos_synergy['overlay_ce_delta'], 4)}, return {_pct(oos_synergy['overlay_return_delta'])}, DD {_pct(oos_synergy['overlay_drawdown_delta'])} | Modest IS benefit; no OOS increment |",
        f"| Does blocking reject worse trades? | Net block value {_num(is_detail.get('net_block_value_r', 0.0), 3)}R; accepted-minus-blocked {_num(is_detail.get('realized_r_discrimination', 0.0), 3)}R | Net block value {_num(oos_detail.get('net_block_value_r', 0.0), 3)}R; accepted-minus-blocked {_num(oos_detail.get('realized_r_discrimination', 0.0), 3)}R | IS yes in aggregate; OOS no |",
        f"| Are good blocks minimized? | {_pct(is_detail.get('positive_trade_block_rate', 0.0))} of all winners blocked; {int(is_detail.get('blocked_r', {}).get('positive_count', 0))} blocked winners | {_pct(oos_detail.get('positive_trade_block_rate', 0.0))} of all winners blocked; {int(oos_detail.get('blocked_r', {}).get('positive_count', 0))} blocked winner | Low rate IS, but not eliminated; OOS block was harmful |",
        f"| Is max DD minimized? | Overlay improves matched-risk DD by {_pct(-is_synergy['overlay_drawdown_delta'])}, but post exceeds pre by {_pct(is_synergy['post_minus_pre_drawdown'])} | Overlay changes DD by {_pct(oos_synergy['overlay_drawdown_delta'])}; post exceeds pre by {_pct(oos_synergy['post_minus_pre_drawdown'])} | No |",
        f"| Is the interaction optimum stable? | CSCV PBO {_pct(selection['cscv_pbo']['probability_backtest_overfit'])} vs {_pct(selection['maximum_probability_backtest_overfit'])} limit | Frozen OOS used once | No; stability fallback retained |",
        "",
        "The word *maximized* is deliberately reserved: finite phased search can establish a local best among tested configurations, not a global optimum. Here, even that local interaction challenger was rejected by the stability guard.",
        "",
        "## Evidence contract, lineage and limits",
        "",
        f"- IS: `{summary['is_window'][0]}` through `{summary['is_window'][1]}`.",
        f"- OOS: `{summary['oos_window'][0]}` through `{summary['oos_window'][1]}`; evaluated after configuration freeze.",
        f"- Initial capital: `${summary['initial_equity']:,.0f}` shared across both sleeves.",
        f"- Frozen config SHA-256: `{summary['config_sha256']}`.",
        f"- Selected candidate: `{selection['selected']}`; unconstrained interaction winner: `{selection['unconstrained_winner']}`.",
        f"- ALCB Round-3 regeneration parity: `{receipt['alcb']['parity']['passed']}`; IARIC Round-3 parity: `{receipt['iaric']['parity']['passed']}`.",
        "- Evidence is completed-trade replay with live-aligned portfolio admission, causal marks and a shared cash/NLV/margin ledger. It is not raw-signal co-simulation or a full source execution/fill simulation.",
        "- The universe is not point-in-time survivorship controlled and the retained data cache is diagnostic-only.",
        "",
        "## Headline post-optimization performance",
        "",
    ]
    _append_performance_table(
        lines,
        [("IS optimized portfolio", post_is), ("OOS optimized portfolio", post_oos)],
    )
    lines.extend(["", "## Pre- versus post-optimization", ""])
    _append_performance_table(
        lines,
        [
            ("IS pre", is_data["pre_optimization_portfolio"]),
            ("IS post", post_is),
            ("OOS pre", oos_data["pre_optimization_portfolio"]),
            ("OOS post", post_oos),
        ],
    )
    lines.extend(
        [
            "",
            f"Post adds {_pct(is_synergy['post_minus_pre_return'])} IS return but changes DD by {_pct(is_synergy['post_minus_pre_drawdown'])} and CE by {_num(is_synergy['post_minus_pre_ce'], 4)}. OOS adds {_pct(oos_synergy['post_minus_pre_return'])} return while DD changes by {_pct(oos_synergy['post_minus_pre_drawdown'])}. This is a more aggressive risk allocation, not an across-the-board risk-adjusted dominance result.",
            "",
            "## Portfolio versus Round-3 standalones",
            "",
            "All comparisons use the same $25,000 capital, causal marks, costs and window boundaries.",
            "",
        ]
    )
    _append_performance_table(
        lines,
        [
            ("IS ALCB R3 native risk", is_data["alcb_round3_standalone_native_risk"]),
            ("IS IARIC R3 native risk", is_data["iaric_round3_standalone_native_risk"]),
            ("IS portfolio", post_is),
            ("OOS ALCB R3 native risk", oos_data["alcb_round3_standalone_native_risk"]),
            ("OOS IARIC R3 native risk", oos_data["iaric_round3_standalone_native_risk"]),
            ("OOS portfolio", post_oos),
        ],
    )
    lines.extend(
        [
            "",
            "### Synergy decomposition",
            "",
            "| Measure | IS | OOS | Interpretation |",
            "|---|---:|---:|---|",
            f"| Daily / weekly sleeve R correlation | {_num(assessment['daily_r_correlation'], 4)} / {_num(assessment['weekly_r_correlation'], 4)} | Not separately recalibrated | Near-zero IS correlation supports complementarity |",
            f"| Portfolio R capture vs native standalone R sum | {_pct(is_synergy['portfolio_r_capture_vs_native_standalones'])} | {_pct(oos_synergy['portfolio_r_capture_vs_native_standalones'])} | Below 100% means some standalone R was not captured |",
            f"| Portfolio R capture vs post-risk standalone R sum | {_pct(is_synergy['portfolio_r_capture_vs_post_risk_standalones'])} | {_pct(oos_synergy['portfolio_r_capture_vs_post_risk_standalones'])} | Matched allocation comparison |",
            f"| Portfolio return minus native standalone return sum | {_pct(is_synergy['portfolio_return_minus_native_standalone_sum'])} | {_pct(oos_synergy['portfolio_return_minus_native_standalone_sum'])} | Shared-NLV cross-compounding affects return percentages |",
            f"| Overlay return delta | {_pct(is_synergy['overlay_return_delta'])} | {_pct(oos_synergy['overlay_return_delta'])} | Incremental routing/governor value |",
            f"| Overlay total-R delta | {_num(is_synergy['overlay_total_r_delta'], 3)}R | {_num(oos_synergy['overlay_total_r_delta'], 3)}R | Alpha added after matching risk |",
            f"| Overlay DD delta | {_pct(is_synergy['overlay_drawdown_delta'])} | {_pct(oos_synergy['overlay_drawdown_delta'])} | Negative is better |",
            f"| Overlay CE-growth delta | {_num(is_synergy['overlay_ce_delta'], 4)} | {_num(oos_synergy['overlay_ce_delta'], 4)} | Immutable score's direct synergy input |",
            f"| Portfolio DD minus worst native standalone DD | {_pct(is_synergy['post_drawdown_minus_worst_native_standalone'])} | {_pct(oos_synergy['post_drawdown_minus_worst_native_standalone'])} | Positive means portfolio DD exceeded both standalones |",
            "",
            "Portfolio percentage return exceeding the sum of standalone percentage returns is possible because both sleeves compound through the same NLV. Total R and the shared-ledger reconciliation are the cleaner opportunity-capture checks.",
            "",
            "## Optimized overlay versus the same allocation without the overlay",
            "",
        ]
    )
    _append_performance_table(
        lines,
        [
            ("IS no overlay", is_data["post_optimization_no_overlay"]),
            ("IS optimized overlay", post_is),
            ("OOS no overlay", oos_data["post_optimization_no_overlay"]),
            ("OOS optimized overlay", post_oos),
        ],
    )
    for window_name, window in (("IS", "is"), ("OOS", "oos")):
        delta = comprehensive["window_diagnostics"][window].get(
            "overlay_block_set_delta",
            {},
        )
        if not delta:
            continue
        optimized_only = delta["optimized_overlay_only"]
        control_only = delta["no_overlay_only"]
        lines.extend(
            [
                "",
                f"- {window_name}: {int(delta['common_block_count'])} blocks were common to both configurations. The overlay uniquely blocked {int(optimized_only['count'])} trades totalling {_num(optimized_only['total'], 3)}R ({int(optimized_only['positive_count'])} winners), while no-overlay uniquely blocked {int(control_only['count'])} trades totalling {_num(control_only['total'], 3)}R.",
            ]
        )
    lines.extend(["", "## Blocking discrimination and good-trade preservation", ""])
    _append_blocking_section(lines, "IS", is_detail)
    _append_blocking_section(lines, "OOS", oos_detail)
    lines.extend(
        [
            "The IS blocker result is net positive because avoided loser magnitude slightly exceeds forgone winner magnitude, despite more than half of blocked trades being eventual winners. OOS contains only one block, so it is statistically uninformative and directionally adverse. Ex-post discrimination must not be confused with ex-ante alpha ranking.",
            "",
            "## Sleeve contribution and balance",
            "",
            "| Window | Sleeve | Trades | Trade share | PnL | PnL share | Risk share |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for window_name, metrics in (("IS", post_is), ("OOS", post_oos)):
        total_pnl = _metric(metrics, "net_pnl")
        total_trades = _metric(metrics, "total_trades")
        sleeve_pnl_total = sum(
            _metric(metrics, f"pnl_{strategy}") for strategy in STRATEGIES
        )
        for strategy in STRATEGIES:
            pnl = _metric(metrics, f"pnl_{strategy}")
            trades = _metric(metrics, f"trades_{strategy}")
            lines.append(
                f"| {window_name} | {strategy} | {int(trades):,} | {_pct(_safe_div(trades, total_trades))} | {_money(pnl)} | {_pct(_safe_div(pnl, sleeve_pnl_total))} | {_pct(_metric(metrics, f'risk_share_{strategy}'))} |"
            )
        financing = _metric(metrics, "financing_cost")
        lines.append(
            f"| {window_name} | Financing/ledger cost | 0 | 0.00% | {_money(-financing)} | n/a | n/a |"
        )
        lines.append(
            f"| {window_name} | Reconciled portfolio net PnL | {int(total_trades):,} | 100.00% | {_money(total_pnl)} | n/a | 100.00% |"
        )
    lines.extend(["", "## Drawdown path and attribution", ""])
    lines.extend(
        [
            "| Window/scenario | Max DD | Peak date | Trough date | Recovery date | Peak equity | Trough equity | Duration | Drawdown PnL contribution |",
            "|---|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for window_name, key, scenario in (
        ("IS", "is", "pre_optimization_portfolio"),
        ("IS", "is", "post_optimization_no_overlay"),
        ("IS", "is", "post_optimization_portfolio"),
        ("OOS", "oos", "pre_optimization_portfolio"),
        ("OOS", "oos", "post_optimization_no_overlay"),
        ("OOS", "oos", "post_optimization_portfolio"),
    ):
        dd = comprehensive["window_diagnostics"][key].get(scenario, {}).get(
            "daily_mtm_drawdown",
            {},
        )
        if not dd:
            continue
        contribution = ", ".join(
            f"{strategy}:{_money(value)}"
            for strategy, value in dd["drawdown_contribution_by_strategy"].items()
        )
        lines.append(
            f"| {window_name} {scenario} | {_pct(dd['max_drawdown_pct'])} | {dd['peak_date']} | {dd['trough_date']} | {dd['recovery_date'] or 'not recovered in window'} | {_usd(dd.get('peak_equity', 0.0))} | {_usd(dd.get('trough_equity', 0.0))} | {int(dd['drawdown_duration_days'])}d | {contribution} |"
        )
    lines.extend(
        [
            "",
            "Negative contribution identifies the sleeve losing MTM PnL from the drawdown peak to trough; it is path attribution, not causal blame. The post portfolio does not minimize DD versus the lower-risk pre baseline or either OOS standalone.",
            "",
            "## Monthly opportunity and blocker path",
            "",
        ]
    )
    _append_monthly(lines, "IS", is_detail)
    _append_monthly(lines, "OOS", oos_detail)
    lines.extend(
        [
            "## Immutable score and scaling",
            "",
            "The score was frozen before candidate testing:",
            "",
            "`score = 0.45*tanh(annual_log_growth/0.50) + 0.25*tanh(R_per_month/15) + 0.10*tanh(trades_per_month/100) + 0.10*tanh(blocked_value_R_per_month/2) + 0.10*tanh(CE_synergy/0.10) - drawdown_penalty - expected_shortfall_penalty - realism_penalty`",
            "",
            "Robust fold aggregation: `0.55*median + 0.30*q25 + 0.15*worst - 0.10*IQR`.",
            "",
            "| Component | Selected IS contribution | Meaning |",
            "|---|---:|---|",
        ]
    )
    score_components = phases["selected_score_components"]
    meanings = {
        "growth": "Expected-return growth term",
        "alpha_throughput": "R delivered per month",
        "frequency": "Accepted trading frequency",
        "blocker_value": "Net ex-post R value of rejected trades",
        "matched_risk_synergy": "CE growth over the same allocation without overlay",
        "drawdown_penalty": "Penalty above 10% daily-MTM DD",
        "expected_shortfall_penalty": "Penalty above 2.5% daily expected shortfall",
        "realism_penalty": "Marks, leverage and margin constraint penalty",
        "score": "Aggregate IS score before robust fold aggregation",
    }
    for key in meanings:
        if key in score_components:
            lines.append(
                f"| `{key}` | {_num(score_components[key], 6)} | {meanings[key]} |"
            )
    lines.extend(
        [
            "",
            "The score intentionally rewards return and frequency, but only 10% directly rewards matched-risk synergy and 10% blocker value. Consequently, a high score alone cannot establish optimal blocker discrimination; the direct diagnostics above are necessary.",
            "",
            "## Phase progression",
            "",
            "| Phase | Objective | Candidates | Eligible | Selected | Robust score | Return | R/month | Trades/month | PF | DD | Accept rate | IS CE synergy | Median fold synergy | Negative synergy folds |",
            "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for phase in phases["phases"]:
        winner = phase["winner"]
        lines.append(
            f"| {phase['phase']} | {phase['title']} | {phase['candidate_count']} | {phase['eligible_count']} | `{phase['selected']}` | {_num(winner['robust_score'], 6)} | {_pct(winner['return'])} | {_num(winner['total_r_per_month'])} | {_num(winner['trades_per_month'], 1)} | {_num(winner['profit_factor'], 3)} | {_pct(winner['max_drawdown'])} | {_pct(winner['accept_rate'])} | {_num(winner['synergy_ce_delta'], 4)} | {_num(winner['median_fold_synergy_ce_delta'], 4)} | {winner['negative_synergy_folds']} |"
        )
    lines.extend(
        [
            "",
            f"The search tested {phases['tested_candidate_count']} candidates. Phase 4 generated the retained benefit. Phase 5's unconstrained winner gained only {_num(selection['robust_score_gain'], 6)} robust-score units, below the {_num(selection['minimum_robust_score_gain'], 3)} requirement, while CSCV PBO was {_pct(selection['cscv_pbo']['probability_backtest_overfit'])}; therefore the simpler phase-4 incumbent was retained.",
            "",
            "### Full tested frontier",
            "",
            "| Phase | Candidate | Selected | Eligible | Robust score | Aggregate score | Return | R/month | Trades/month | PF | WR | DD | Accept | Blocked R | CE synergy | Median fold synergy | Negative folds | Worst fold DD |",
            "|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in phases["frontier"]:
        lines.append(
            f"| {row['phase']} | `{row['name']}` | {'yes' if row['selected'] else ''} | {'yes' if row['eligible'] else 'no'} | {_num(row['robust_score'], 6)} | {_num(row['aggregate_score'], 6)} | {_pct(row['return'])} | {_num(row['total_r_per_month'])} | {_num(row['trades_per_month'], 1)} | {_num(row['profit_factor'], 3)} | {_pct(row['win_rate'])} | {_pct(row['max_drawdown'])} | {_pct(row['accept_rate'])} | {_num(row['blocked_total_r'], 2)}R | {_num(row['synergy_ce_delta'], 4)} | {_num(row['median_fold_synergy_ce_delta'], 4)} | {row['negative_synergy_folds']} | {_pct(row['worst_fold_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "### Selected configuration fold stability",
            "",
            "| Fold | Return | PnL | R/month | Trades/month | PF | DD | Accept | No-overlay return | No-overlay DD | CE synergy | Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in phases["selected_folds"]:
        lines.append(
            f"| {row['fold']} | {_pct(row['return'])} | {_money(row['net_pnl'])} | {_num(row['total_r_per_month'])} | {_num(row['trades_per_month'], 1)} | {_num(row['profit_factor'], 3)} | {_pct(row['max_drawdown'])} | {_pct(row['accept_rate'])} | {_pct(row['no_overlay_return'])} | {_pct(row['no_overlay_max_drawdown'])} | {_num(row['synergy_ce_delta'], 4)} | {_num(row['score'], 5)} |"
        )
    lines.extend(
        [
            "",
            "## Alpha-ranking calibration",
            "",
            f"Selected model: `{alpha['selected']}`. Stable: `{alpha['variants'][alpha['selected']]['stable']}`. The final config requests `candidate_rank_mode={cross['candidate_rank_mode']}`, but the model stability gate failed and alpha admission is disabled. Capacity rules, not a validated alpha ranker, drive most rejections.",
            "",
            "| Variant | Stable | Comparisons | Positive lifts | Median top-bottom lift R | Lower-quartile lift R | Selection score |",
            "|---|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in alpha["variants"].items():
        lines.append(
            f"| `{name}` | {'yes' if row['stable'] else 'no'} | {row['comparisons']} | {row['positive_lifts']} | {_num(row['median_top_bottom_lift_R'], 4)} | {_num(row['lower_quartile_lift_R'], 4)} | {_num(row['selection_score'], 5)} |"
        )
    lines.extend(
        [
            "",
            "### Selected model lift by fold and strategy",
            "",
            "| Fold | Strategy | Count | Top-quartile R | Bottom-quartile R | Top-minus-bottom R |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in alpha["variants"][alpha["selected"]]["folds"]:
        lines.append(
            f"| {row['fold']} | {row['strategy']} | {row['count']} | {_num(row['top_quartile_R'], 4)} | {_num(row['bottom_quartile_R'], 4)} | {_num(row['top_bottom_lift_R'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Robustness, perturbations and costs",
            "",
            "### Local parameter perturbations",
            "",
            "| Parameter | Multiplier | Return | PnL | R/month | Trades/month | PF | DD | Accept rate | Blocked R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in robustness["local_perturbations"]:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['path']}` | {row['multiplier']:.2f}x | {_pct(_metric(metrics, 'net_return_pct'))} | {_money(_metric(metrics, 'net_pnl'))} | {_num(_metric(metrics, 'total_r_per_month'))} | {_num(_metric(metrics, 'active_trades_per_month'), 1)} | {_num(_metric(metrics, 'profit_factor'), 3)} | {_pct(_metric(metrics, 'max_drawdown_pct_mtm_daily'))} | {_pct(_metric(metrics, 'entry_accept_rate'))} | {_num(_metric(metrics, 'blocked_total_r'), 2)}R |"
        )
    lines.extend(
        [
            "",
            "### Incremental round-trip cost stress",
            "",
            "| Extra cost | Return | PnL | PF | DD | R/month |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in robustness["incremental_cost_stress"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['extra_round_trip_bps']:.0f} bps | {_pct(_metric(metrics, 'net_return_pct'))} | {_money(_metric(metrics, 'net_pnl'))} | {_num(_metric(metrics, 'profit_factor'), 3)} | {_pct(_metric(metrics, 'max_drawdown_pct_mtm_daily'))} | {_num(_metric(metrics, 'total_r_per_month'))}R |"
        )
    bootstrap = robustness["weekly_block_bootstrap"]
    lines.extend(
        [
            "",
            f"- Weekly block bootstrap: {bootstrap['samples']:,} samples; probability total PnL is positive {_pct(bootstrap['probability_total_pnl_positive'])}; 95% total-PnL interval {_money(bootstrap['ci_95_total_pnl'][0])} to {_money(bootstrap['ci_95_total_pnl'][1])}.",
            f"- CSCV: {robustness['cscv_pbo']['splits']} splits; PBO {_pct(robustness['cscv_pbo']['probability_backtest_overfit'])}; median test-rank logit {_num(robustness['cscv_pbo']['median_test_rank_logit'], 4)}.",
            "",
            "## Optimized configuration and binding controls",
            "",
            "| Control | Value |",
            "|---|---:|",
            f"| Risk stance | `{config['risk_stance']}` |",
            f"| ALCB unit risk | {_pct(config['strategy_allocations']['ALCB_R3']['unit_risk_pct'], 4)} |",
            f"| IARIC unit risk | {_pct(config['strategy_allocations']['IARIC_RESIDUAL_R3']['unit_risk_pct'], 4)} |",
            f"| Gross / net / overnight notional cap | {account['max_gross_notional_pct']:.2f}x / {account['max_net_notional_pct']:.2f}x / {account['max_overnight_gross_notional_pct']:.2f}x |",
            f"| Position / symbol notional cap | {_pct(account['max_position_notional_pct'])} / {_pct(account['max_symbol_notional_pct'])} |",
            f"| Portfolio heat / long heat / symbol heat | {rules['heat_cap_R']:.1f}R / {rules['max_long_heat_R']:.1f}R / {rules['max_symbol_heat_R']:.1f}R |",
            f"| Maximum active positions | {rules['max_total_active_positions']} |",
            f"| Daily / weekly stop | {rules['portfolio_daily_stop_R']:.1f}R / {rules['portfolio_weekly_stop_R']:.1f}R |",
            f"| Alpha admission | `{cross['alpha_admission_enabled']}` |",
            f"| Rank mode / capacity action | `{cross['candidate_rank_mode']}` / `{cross['capacity_action']}` |",
            f"| Minimum capacity multiplier | {cross['minimum_capacity_size_mult']:.2f} |",
            "",
            "### Shared-account realism",
            "",
            f"- IS peak gross / net / overnight leverage: {post_is['gross_leverage_peak']:.2f}x / {post_is['net_leverage_peak_abs']:.2f}x / {post_is['overnight_gross_leverage_peak']:.2f}x; OOS: {post_oos['gross_leverage_peak']:.2f}x / {post_oos['net_leverage_peak_abs']:.2f}x / {post_oos['overnight_gross_leverage_peak']:.2f}x.",
            f"- Mark coverage IS/OOS: {_pct(post_is['mark_coverage_ratio'])} / {_pct(post_oos['mark_coverage_ratio'])}; margin breaches: {int(post_is['margin_breach_count'])} / {int(post_oos['margin_breach_count'])}.",
            f"- Financing cost IS/OOS: {_money(post_is['financing_cost'])} / {_money(post_oos['financing_cost'])}; minimum margin buffer: {_pct(post_is['minimum_margin_buffer_pct'])} / {_pct(post_oos['minimum_margin_buffer_pct'])}.",
            "- Whole-share quantities are floored without forcing a minimum share. Debit-cash financing and Reg-T admission are applied. Shared buying power prevents duplicated-capital inflation.",
            "",
            "## Synergy decision criteria",
            "",
            "| Criterion | Status |",
            "|---|:---:|",
        ]
    )
    for key, value in assessment["criteria"].items():
        lines.append(f"| `{key}` | {_bool(value)} |")
    lines.extend(["", "## Promotion gates and research restrictions", ""])
    for key, value in gates.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Production activation remains disallowed because:", ""])
    for restriction in summary["promotion_decision"]["research_restrictions"]:
        lines.append(f"- {restriction}.")
    lines.extend(
        [
            "",
            "## Final answer to the portfolio-synergy objective",
            "",
            "1. **Expected return and trade frequency:** preserved well. IS captures approximately the full native standalone R sum and OOS captures most of it, with high acceptance and both sleeves profitable.",
            "2. **Blocking worse trades:** demonstrated only in aggregate IS. Losses avoided slightly exceed winners forgone and accepted average R is higher. This is not validated OOS, where the only block is a +1.010R winner.",
            "3. **Minimizing good-trade blocks:** partly achieved IS because the positive-trade block rate is low relative to all winners, but the blocker precision is weak and many blocked trades are winners. The report therefore does not call routing alpha-selective.",
            "4. **Minimizing max drawdown:** the overlay reduces IS DD relative to the same aggressive allocation without overlay, but post DD is higher than the native-risk pre baseline and OOS overlay provides no DD benefit.",
            "5. **Maximized synergy:** not established. The system is a stable fallback local baseline, not the stable interaction winner. The 90% PBO, unstable alpha calibration, absent raw-signal co-simulation, IARIC parity failure and single harmful OOS block are decisive limitations.",
            "",
            "The correct operational conclusion is **retain as a realistic research baseline, not as proof of maximized synergy or as a production-ready portfolio**. The next valid optimization round must improve a causal ex-ante ranker and test blocker discrimination on raw fired signals under point-in-time data, while keeping the shared-account ledger and frozen OOS protocol unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_round(round_dir: Path) -> dict[str, Any]:
    round_dir = round_dir.resolve()
    summary_path = round_dir / "diagnostics_summary.json"
    summary = _load(summary_path)
    summary["matched_performance"] = _load(round_dir / "matched_performance.json")
    summary["diagnostics_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary = enrich_summary(round_dir, summary)
    report = render_diagnostics(summary)
    _atomic_write_json(summary, summary_path)
    _atomic_write_json(
        summary["comprehensive_diagnostics"],
        round_dir / "comprehensive_synergy_diagnostics.json",
    )
    for name in ("round_final_diagnostics.md", "round_final_diagnostics.txt"):
        (round_dir / name).write_text(report, encoding="utf-8")

    run_summary_path = round_dir / "run_summary.json"
    run_summary = _load(run_summary_path)
    run_summary["diagnostic_schema"] = summary["schema"]
    run_summary["synergy_assessment"] = summary["comprehensive_diagnostics"][
        "synergy_assessment"
    ]
    _atomic_write_json(run_summary, run_summary_path)
    (round_dir / "round_evaluation.txt").write_text(
        "Round 1 remains the canonical realistic research baseline. It shows modest "
        "IS overlay synergy, but OOS routing synergy and maximized blocker discrimination "
        "are not demonstrated; production activation remains disallowed. See "
        "round_final_diagnostics.md.\n",
        encoding="utf-8",
    )

    artifacts = {
        path.name: _sha256(path)
        for path in sorted(round_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest = _load(round_dir / "artifact_manifest.json")
    manifest["diagnostic_schema"] = summary["schema"]
    manifest["artifact_count"] = len(artifacts)
    manifest["artifacts"] = artifacts
    _atomic_write_json(manifest, round_dir / "artifact_manifest.json")
    return {
        "round_dir": str(round_dir),
        "schema": summary["schema"],
        "verdict": summary["comprehensive_diagnostics"]["synergy_assessment"][
            "verdict"
        ],
        "report_lines": len(report.splitlines()),
        "report_bytes": len(report.encode("utf-8")),
        "tested_candidates": summary["comprehensive_diagnostics"][
            "phase_diagnostics"
        ]["tested_candidate_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-dir",
        type=Path,
        default=Path("backtests/output/stock/portfolio_synergy/round_1"),
    )
    args = parser.parse_args()
    print(json.dumps(refresh_round(args.round_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
