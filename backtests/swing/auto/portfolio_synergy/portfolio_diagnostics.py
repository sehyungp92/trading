from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backtests.swing.analysis.metrics import compute_metrics
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import load_unified_data, run_unified

from .evaluator import (
    ACTIVE_STRATEGIES,
    _collect_unified_trades,
    _headline_equity_curve,
    _months,
    _positive_years,
    _realized_equity_curve,
    _risk_basis,
    _total_r,
    _trade_commission,
    _trade_hold_hours,
    _trade_pnl,
    _trade_risk,
    _trade_symbol,
    build_effective_portfolio_config,
    build_replay_config,
)
from .phase_candidates import INITIAL_EQUITY, STRATEGY_ORDER


STRATEGY_OUTPUTS = {
    "ATRSS": ("atrss", "round_3"),
    "AKC_HELIX": ("helix", "round_2"),
    "BRS_R9": ("brs", "round_1"),
    "SWING_BREAKOUT_V3": ("breakout", "round_5"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build detailed diagnostics for the unified swing portfolio synergy replay.",
    )
    parser.add_argument("--run-dir", default="backtests/output/swing/portfolio_synergy/round_1")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--swing-output-root", default="backtests/output/swing")
    parser.add_argument("--equity", type=float, default=INITIAL_EQUITY)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    diagnostics = build_diagnostics(
        run_dir=run_dir,
        data_dir=Path(args.data_dir),
        swing_output_root=Path(args.swing_output_root),
        initial_equity=float(args.equity),
    )
    _write_json(run_dir / "portfolio_diagnostics.json", diagnostics)
    (run_dir / "portfolio_diagnostics.md").write_text(render_markdown(diagnostics), encoding="utf-8")
    print(f"Wrote {run_dir / 'portfolio_diagnostics.json'}")
    print(f"Wrote {run_dir / 'portfolio_diagnostics.md'}")


def build_diagnostics(
    *,
    run_dir: Path,
    data_dir: Path,
    swing_output_root: Path,
    initial_equity: float = INITIAL_EQUITY,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    mutations = _load_mutations(run_dir)
    effective = build_effective_portfolio_config(mutations, initial_equity=initial_equity)
    replay_config = build_replay_config(effective, Path(data_dir), initial_equity)
    data = load_unified_data(replay_config)
    result = run_unified(data, replay_config)

    all_trades = _collect_unified_trades(result)
    trades_by_strategy = _trades_by_strategy(result)
    metrics = _portfolio_metrics(result, all_trades, initial_equity)
    summary_metrics = _load_summary_metrics(run_dir)
    if summary_metrics:
        for key, value in summary_metrics.items():
            if value is not None and (key.startswith("score_") or key == "score_total" or key not in metrics):
                metrics[key] = value
    metrics.update(_entry_flow_totals(result))

    entry_events = list(getattr(result, "entry_events", []) or [])
    strategy_summary = _strategy_summary(result, entry_events, trades_by_strategy)
    block_summary = _block_summary(entry_events)
    shadow_summary = _blocked_shadow_summary(entry_events, data)
    crowding = _crowding_summary(entry_events)
    monthly_flow = _monthly_flow(entry_events)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "headline": _headline(metrics),
        "config": _config_snapshot(effective, replay_config),
        "strategy_summary": strategy_summary,
        "block_summary": block_summary,
        "blocked_shadow_summary": shadow_summary,
        "monthly_flow": monthly_flow,
        "crowding_summary": crowding,
        "overlay": _overlay_summary(result, replay_config, effective, initial_equity),
        "coordination": {
            "tighten_events": int(getattr(result, "coordination_tighten_count", 0)),
            "boost_events": int(getattr(result, "coordination_boost_count", 0)),
            "portfolio_daily_stop_activations": int(getattr(result, "portfolio_daily_stop_activations", 0)),
        },
        "heat": {
            "avg_heat_R": float(getattr(result.heat_stats, "avg_heat_pct", 0.0)),
            "max_heat_R": float(getattr(result.heat_stats, "max_heat_pct", 0.0)),
            "pct_time_at_cap": float(getattr(result.heat_stats, "pct_time_at_cap", 0.0)),
        },
        "individual_strategy_reference": _individual_reference(Path(swing_output_root)),
        "interpretation": _interpret(metrics, strategy_summary, block_summary, crowding),
        "notes": {
            "blocked_outcome_backfill": (
                "Blocked portfolio entries are counted with exact block reason and risk context. "
                "A conservative shadow replay estimates their later outcome, but the labels are diagnostic "
                "only and should be treated as small-sample evidence."
            ),
            "fired_definition": (
                "Fired means an entry or add-on reached the unified portfolio risk gate. "
                "Accepted means the portfolio gate allowed it. Closed trades are accepted "
                "positions/orders that later produced a completed trade record."
            ),
        },
    }


def render_markdown(diagnostics: dict[str, Any]) -> str:
    headline = diagnostics["headline"]
    lines = [
        "# Swing Portfolio Synergy Diagnostics",
        "",
        "## Executive Read",
        "",
        (
            f"Unified replay fired {headline['fired_entries']:.0f} portfolio entry opportunities, "
            f"accepted {headline['accepted_entries']:.0f}, and blocked {headline['blocked_entries']:.0f} "
            f"({headline['block_rate']:.1%} block rate)."
        ),
        (
            f"Closed trades: {headline['closed_trades']:.0f}; final equity ${headline['final_equity']:,.2f}; "
            f"net return {headline['net_return_pct']:.1%}; PF {headline['profit_factor']:.2f}; "
            f"win rate {headline['win_rate']:.1%}; max DD {headline['max_drawdown_pct']:.2%}."
        ),
        (
            f"Key ratios: Sharpe {headline['sharpe']:.2f}, Sortino {headline['sortino']:.2f}, "
            f"Calmar {headline['calmar']:.2f}; score {headline['score_total']:.4f}."
        ),
        "",
        diagnostics["notes"]["blocked_outcome_backfill"],
        "",
        "## Risk Allocation Snapshot",
        "",
    ]
    if headline.get("risk_basis"):
        lines.insert(
            10,
            (
                f"Risk basis: {headline['risk_basis']}; MTM max DD "
                f"{headline.get('max_drawdown_pct_mtm', headline['max_drawdown_pct']):.2%} "
                f"vs realized-only {headline.get('max_drawdown_pct_realized', 0.0):.2%}."
            ),
        )
    config = diagnostics.get("config", {})
    rules = config.get("portfolio_rules", {})
    replay = config.get("replay", {})
    symbol_mults = config.get("symbol_risk_multipliers", {})
    lines.extend(
        [
            (
                f"Dynamic risk is {'enabled' if rules.get('dynamic_risk_enabled') else 'disabled'} with "
                f"{replay.get('heat_cap_R', 0.0):.2f}R portfolio heat, "
                f"{rules.get('portfolio_daily_stop_R', 0.0):.2f}R daily stop, and "
                f"{replay.get('overlay_max_pct', 0.0):.0%} overlay max equity."
            ),
            f"Drawdown risk tiers: {_tier_text(rules.get('drawdown_tiers', []))}.",
            "",
            "| Strategy | Unit Risk | Max Heat | Daily Stop | Priority |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGY_ORDER:
        alloc = config.get("strategy_allocations", {}).get(strategy, {})
        if strategy == "OVERLAY":
            continue
        lines.append(
            f"| {strategy} | {float(alloc.get('unit_risk_pct', 0.0)):.2%} | "
            f"{float(alloc.get('max_heat_R', 0.0)):.2f}R | "
            f"{float(alloc.get('daily_stop_R', 0.0)):.2f}R | "
            f"{int(alloc.get('priority', 0))} |"
        )
    if symbol_mults:
        lines.extend(["", f"Symbol risk multipliers: {_counter_text(symbol_mults)}."])
    lines.extend(
        [
        "",
        "## Fired, Accepted, Blocked By Sleeve",
        "",
        "| Strategy | Fired | Accepted | Blocked | Accept Rate | Closed Trades | Closed WR | PnL | Total R | Avg R | Top Block Reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in ACTIVE_STRATEGIES:
        row = diagnostics["strategy_summary"].get(strategy, {})
        lines.append(
            f"| {strategy} | {row.get('fired', 0):.0f} | {row.get('accepted', 0):.0f} | "
            f"{row.get('blocked', 0):.0f} | {row.get('accept_rate', 0.0):.1%} | "
            f"{row.get('closed_trades', 0):.0f} | {row.get('closed_win_rate', 0.0):.1%} | "
            f"${row.get('pnl', 0.0):,.2f} | {row.get('total_r', 0.0):.2f} | "
            f"{row.get('avg_r', 0.0):.2f} | {row.get('top_block_reason', '')} |"
        )

    overlay = diagnostics["overlay"]
    lines.extend(
        [
            "",
            "## Overlay Idle-Cash Sleeve",
            "",
            (
                f"Overlay is {'enabled' if overlay['enabled'] else 'disabled'} in {overlay['mode']} mode with "
                f"a {overlay['max_equity_pct']:.0%} max equity cap, "
                f"{overlay['min_alloc_pct']:.0%}-{overlay['max_alloc_pct']:.0%} adaptive sizing, "
                f"and weights {_weight_text(overlay['weights'])}."
            ),
            (
                f"Overlay net PnL: ${overlay.get('pnl_net', overlay['pnl']):,.2f} "
                f"({overlay['return_pct']:.1%} of starting equity); transaction costs "
                f"${overlay.get('commission', 0.0):,.2f}; per-symbol PnL: "
                f"{_money_counter_text(overlay['per_symbol_pnl'])}."
            ),
            (
                "Fired/accepted/blocked counts are not applicable to overlay because it is a daily "
                "idle-capital allocation, not a discrete swing entry that passes through the portfolio heat gate."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Block Reasons",
            "",
            "| Reason Bucket | Count | Main Sleeves | Symbols | Stages | Avg Portfolio Open R | Avg Request R | Avg Strategy After R | Raw Examples |",
            "|---|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for reason, row in diagnostics["block_summary"].items():
        lines.append(
            f"| {reason} | {row['count']} | {_counter_text(row['by_strategy'])} | "
            f"{_counter_text(row['by_symbol'])} | {_counter_text(row['by_stage'])} | "
            f"{row['avg_portfolio_open_R']:.2f} | {row['avg_portfolio_request_R']:.2f} | "
            f"{row['avg_strategy_after_R']:.2f} | {_example_text(row.get('raw_reason_examples', []))} |"
        )

    shadow = diagnostics.get("blocked_shadow_summary", {})
    if shadow:
        lines.extend(
            [
                "",
                "## Blocked Candidate Shadow Outcomes",
                "",
                (
                    f"Shadow replay evaluated {shadow.get('evaluated', 0)} blocked candidates over "
                    f"{shadow.get('horizon_bars', 0)} future hourly bars with a conservative stop-first rule. "
                    f"Filled shadow candidates had {shadow.get('filled_win_rate', 0.0):.1%} positive outcomes "
                    f"and average outcome {shadow.get('avg_outcome_R', 0.0):+.2f}R."
                ),
                "These labels are diagnostic only; they are not fed directly into the optimizer as hindsight targets.",
                "",
                "| Bucket | Count | Filled | Positive | Win Rate | Avg Outcome R | Avg MFE R | Avg MAE R |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for bucket, row in shadow.get("by_strategy", {}).items():
            lines.append(_shadow_row(f"Strategy {bucket}", row))
        for bucket, row in shadow.get("by_reason", {}).items():
            lines.append(_shadow_row(f"Reason {bucket}", row))
        for item in shadow.get("broad_discriminators", []):
            lines.append(_shadow_row(item["label"], item))

    lines.extend(
        [
            "",
            "## Monthly Entry Flow",
            "",
            "| Month | Fired | Accepted | Blocked | Block Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for month, row in diagnostics["monthly_flow"].items():
        lines.append(
            f"| {month} | {row['fired']} | {row['accepted']} | {row['blocked']} | {row['block_rate']:.1%} |"
        )

    crowding = diagnostics["crowding_summary"]
    lines.extend(
        [
            "",
            "## Signal Crowding And Priority Pressure",
            "",
            f"- Entry opportunities with another sleeve firing within 4h: {crowding['pct_events_with_other_signal_within_4h']:.1%}",
            f"- Blocked opportunities with another sleeve firing within 4h: {crowding['pct_blocked_with_other_signal_within_4h']:.1%}",
            f"- Blocked opportunities with portfolio open risk above 70% of cap: {crowding['pct_blocked_when_portfolio_heat_above_70pct']:.1%}",
            f"- Average portfolio open R before blocked entry: {crowding['avg_portfolio_open_R_before_block']:.2f}",
            "",
            "Top within-4h sleeve pairs:",
        ]
    )
    for pair, count in crowding["top_pairs_within_4h"].items():
        lines.append(f"- {pair}: {count}")

    heat = diagnostics["heat"]
    coord = diagnostics["coordination"]
    lines.extend(
        [
            "",
            "## Heat And Coordination",
            "",
            f"- Avg heat used: {heat['avg_heat_R']:.2f}R",
            f"- Max heat reached: {heat['max_heat_R']:.2f}R",
            f"- Time at heat cap: {heat['pct_time_at_cap']:.1f}%",
            f"- ATRSS to Helix tighten events: {coord['tighten_events']}",
            f"- ATRSS to Helix boost events: {coord['boost_events']}",
            f"- Portfolio daily stop activations: {coord['portfolio_daily_stop_activations']}",
        ]
    )
    if heat["max_heat_R"] > replay.get("heat_cap_R", 0.0) > 0.0:
        lines.append(
            "- Max heat can exceed the entry cap after a dynamic risk-scale throttle because open positions are not forcibly liquidated."
        )
    lines.extend(
        [
            "",
            "## Individual Strategy Reference",
            "",
            "| Strategy | Source | Trades | Return | PF | Max DD | Trades/Mo | High-value note |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for strategy, row in diagnostics["individual_strategy_reference"].items():
        note = (row.get("high_value_notes") or [""])[0]
        lines.append(
            f"| {strategy} | {row.get('source_label', '')} | {_fmt_num(row.get('total_trades'))} | "
            f"{_fmt_pct_like(row.get('net_return_pct'))} | {_fmt_num(row.get('profit_factor'))} | "
            f"{_fmt_pct(row.get('max_drawdown_pct'))} | {_fmt_num(row.get('trades_per_month'))} | {note} |"
        )

    interp = diagnostics["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The dominant block reason is `{interp['dominant_block_reason']}`.",
            f"- The most blocked sleeve is `{interp['most_blocked_strategy']}`.",
            f"- Fired-to-closed conversion is {interp['fired_to_closed_conversion']:.1%}; this separates portfolio acceptance from eventual trade completion.",
            "- Blocked-candidate shadow labels are included above, but broad discriminator splits are only reported when the sample is large enough to reduce hindsight overfit.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_mutations(run_dir: Path) -> dict[str, Any]:
    optimized = run_dir / "optimized_config.json"
    if optimized.exists():
        return json.loads(optimized.read_text(encoding="utf-8"))
    state = run_dir / "phase_state.json"
    if state.exists():
        return json.loads(state.read_text(encoding="utf-8")).get("cumulative_mutations", {})
    return {}


def _portfolio_metrics(result, all_trades: list[Any], initial_equity: float) -> dict[str, Any]:
    equity = _headline_equity_curve(result)
    realized_equity = _realized_equity_curve(result)
    timestamps = result.combined_timestamps
    metrics = compute_metrics(
        np.array([_trade_pnl(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_risk(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_hold_hours(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_commission(trade) for trade in all_trades], dtype=np.float64),
        equity,
        timestamps,
        initial_equity,
        trade_symbols=[_trade_symbol(trade) for trade in all_trades],
    )
    realized_metrics = compute_metrics(
        np.array([_trade_pnl(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_risk(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_hold_hours(trade) for trade in all_trades], dtype=np.float64),
        np.array([_trade_commission(trade) for trade in all_trades], dtype=np.float64),
        realized_equity,
        timestamps,
        initial_equity,
        trade_symbols=[_trade_symbol(trade) for trade in all_trades],
    )
    final_equity = float(equity[-1]) if len(equity) else float(initial_equity)
    final_equity_realized = float(realized_equity[-1]) if len(realized_equity) else float(initial_equity)
    total_trades = len(all_trades)
    return {
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "net_pnl": final_equity - initial_equity,
        "net_return_pct": (final_equity - initial_equity) / initial_equity if initial_equity else 0.0,
        "final_equity_realized": final_equity_realized,
        "net_pnl_realized": final_equity_realized - initial_equity,
        "net_return_pct_realized": (
            (final_equity_realized - initial_equity) / initial_equity if initial_equity else 0.0
        ),
        "closed_trades": float(total_trades),
        "active_trades_per_month": total_trades / _months(timestamps) if len(timestamps) else 0.0,
        "total_r": _total_r(all_trades),
        "total_r_per_month": _total_r(all_trades) / _months(timestamps) if len(timestamps) else 0.0,
        "profit_factor": float(metrics.profit_factor),
        "win_rate": float(metrics.win_rate),
        "expectancy_r": float(metrics.expectancy),
        "sharpe": float(metrics.sharpe),
        "sortino": float(metrics.sortino),
        "calmar": float(metrics.calmar),
        "max_drawdown_pct": float(metrics.max_drawdown_pct),
        "max_drawdown_dollar": float(metrics.max_drawdown_dollar),
        "risk_basis": _risk_basis(result),
        "calmar_mtm": float(metrics.calmar),
        "max_drawdown_pct_mtm": float(metrics.max_drawdown_pct),
        "max_drawdown_dollar_mtm": float(metrics.max_drawdown_dollar),
        "calmar_realized": float(realized_metrics.calmar),
        "max_drawdown_pct_realized": float(realized_metrics.max_drawdown_pct),
        "max_drawdown_dollar_realized": float(realized_metrics.max_drawdown_dollar),
        "overlay_pnl": float(getattr(result, "overlay_pnl", 0.0) or 0.0),
        "overlay_pnl_net": float(getattr(result, "overlay_pnl", 0.0) or 0.0),
        "overlay_commission": float(getattr(result, "overlay_commission", 0.0) or 0.0),
        "positive_years": float(_positive_years(equity, timestamps)),
    }


def _load_summary_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("final_metrics", {})
    except json.JSONDecodeError:
        return {}


def _entry_flow_totals(result) -> dict[str, float]:
    fired = sum(getattr(sr, "entry_signals_fired", 0) for sr in result.strategy_results.values())
    accepted = sum(getattr(sr, "entries_accepted_by_portfolio", 0) for sr in result.strategy_results.values())
    blocked = sum(getattr(sr, "entries_blocked_by_heat", 0) for sr in result.strategy_results.values())
    return {
        "fired_entries": float(fired),
        "accepted_entries": float(accepted),
        "blocked_entries": float(blocked),
        "block_rate": _safe_div(blocked, fired),
        "entry_accept_rate": _safe_div(accepted, fired),
    }


def _overlay_summary(
    result,
    config: UnifiedBacktestConfig,
    effective: dict[str, Any],
    initial_equity: float,
) -> dict[str, Any]:
    overlay_alloc = effective.get("strategy_allocations", {}).get("OVERLAY", {})
    pnl = float(getattr(result, "overlay_pnl", 0.0) or 0.0)
    commission = float(getattr(result, "overlay_commission", 0.0) or 0.0)
    return {
        "enabled": bool(getattr(config, "overlay_enabled", False)),
        "mode": str(getattr(config, "overlay_mode", "")),
        "symbols": list(getattr(config, "overlay_symbols", []) or []),
        "max_equity_pct": float(getattr(config, "overlay_max_pct", overlay_alloc.get("max_equity_pct", 0.0)) or 0.0),
        "min_alloc_pct": float(getattr(config, "overlay_min_alloc_pct", overlay_alloc.get("min_alloc_pct", 0.0)) or 0.0),
        "max_alloc_pct": float(getattr(config, "overlay_max_alloc_pct", overlay_alloc.get("max_alloc_pct", 0.0)) or 0.0),
        "weights": dict(getattr(config, "overlay_weights", None) or overlay_alloc.get("weights", {}) or {}),
        "pnl": pnl,
        "pnl_net": pnl,
        "commission": commission,
        "return_pct": pnl / initial_equity if initial_equity > 0 else 0.0,
        "per_symbol_pnl": dict(getattr(result, "overlay_per_symbol_pnl", {}) or {}),
        "entry_gate_counts_applicable": False,
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "fired_entries": float(metrics.get("fired_entries", metrics.get("entry_signals_fired", 0.0)) or 0.0),
        "accepted_entries": float(metrics.get("accepted_entries", metrics.get("entries_accepted_by_portfolio", 0.0)) or 0.0),
        "blocked_entries": float(metrics.get("blocked_entries", metrics.get("entries_blocked_by_portfolio", 0.0)) or 0.0),
        "block_rate": float(metrics.get("block_rate", metrics.get("positive_alpha_block_rate", 0.0)) or 0.0),
        "closed_trades": float(metrics.get("closed_trades", metrics.get("total_trades", 0.0)) or 0.0),
        "final_equity": float(metrics.get("final_equity", 0.0) or 0.0),
        "net_return_pct": float(metrics.get("net_return_pct", 0.0) or 0.0),
        "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
        "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
        "sharpe": float(metrics.get("sharpe", 0.0) or 0.0),
        "sortino": float(metrics.get("sortino", 0.0) or 0.0),
        "calmar": float(metrics.get("calmar", 0.0) or 0.0),
        "risk_basis": str(metrics.get("risk_basis", "") or ""),
        "max_drawdown_pct_mtm": float(metrics.get("max_drawdown_pct_mtm", metrics.get("max_drawdown_pct", 0.0)) or 0.0),
        "calmar_mtm": float(metrics.get("calmar_mtm", metrics.get("calmar", 0.0)) or 0.0),
        "max_drawdown_pct_realized": float(metrics.get("max_drawdown_pct_realized", 0.0) or 0.0),
        "calmar_realized": float(metrics.get("calmar_realized", 0.0) or 0.0),
        "overlay_pnl_net": float(metrics.get("overlay_pnl_net", metrics.get("overlay_pnl", 0.0)) or 0.0),
        "overlay_commission": float(metrics.get("overlay_commission", 0.0) or 0.0),
        "score_total": float(metrics.get("score_total", 0.0) or 0.0),
    }


def _trades_by_strategy(result) -> dict[str, list[Any]]:
    return {
        "ATRSS": list(getattr(result, "atrss_trades", []) or []),
        "AKC_HELIX": list(getattr(result, "helix_trades", []) or []),
        "BRS_R9": list(getattr(result, "brs_trades", []) or []),
        "SWING_BREAKOUT_V3": list(getattr(result, "breakout_trades", []) or []),
    }


def _strategy_summary(result, entry_events: list[dict], trades_by_strategy: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    blocked_by_strategy = _events_by_strategy(entry_events, status="blocked")
    accepted_by_strategy = _events_by_strategy(entry_events, status="accepted")
    summary: dict[str, dict[str, Any]] = {}
    for strategy in ACTIVE_STRATEGIES:
        sr = result.strategy_results.get(strategy)
        trades = trades_by_strategy.get(strategy, [])
        fired = int(getattr(sr, "entry_signals_fired", 0) if sr is not None else 0)
        accepted = int(getattr(sr, "entries_accepted_by_portfolio", 0) if sr is not None else 0)
        blocked = int(getattr(sr, "entries_blocked_by_heat", 0) if sr is not None else 0)
        closed = int(getattr(sr, "total_trades", len(trades)) if sr is not None else len(trades))
        wins = int(getattr(sr, "winning_trades", sum(1 for trade in trades if _trade_pnl(trade) > 0)) if sr is not None else 0)
        total_r = float(getattr(sr, "total_r", sum(getattr(trade, "r_multiple", 0.0) for trade in trades)) if sr is not None else 0.0)
        pnl = float(sum(_trade_pnl(trade) - _trade_commission(trade) for trade in trades))
        raw_reasons = Counter(event.get("reason", "unknown") or "unknown" for event in blocked_by_strategy.get(strategy, []))
        reasons = Counter(_reason_bucket(reason) for reason in raw_reasons.elements())
        summary[strategy] = {
            "fired": fired,
            "accepted": accepted,
            "blocked": blocked,
            "accept_rate": _safe_div(accepted, fired),
            "block_rate": _safe_div(blocked, fired),
            "closed_trades": closed,
            "accepted_to_closed_rate": _safe_div(closed, accepted),
            "closed_win_rate": _safe_div(wins, closed),
            "pnl": pnl,
            "total_r": total_r,
            "avg_r": _safe_div(total_r, closed),
            "avg_accepted_request_R": _avg(_event_context_value(accepted_by_strategy.get(strategy, []), "portfolio_request_risk_R")),
            "avg_blocked_request_R": _avg(_event_context_value(blocked_by_strategy.get(strategy, []), "portfolio_request_risk_R")),
            "block_reasons": dict(reasons),
            "raw_block_reasons": dict(raw_reasons),
            "top_block_reason": reasons.most_common(1)[0][0] if reasons else "",
        }
    return summary


def _block_summary(entry_events: list[dict]) -> dict[str, dict[str, Any]]:
    blocked = [event for event in entry_events if event.get("status") == "blocked"]
    by_reason: dict[str, list[dict]] = defaultdict(list)
    for event in blocked:
        by_reason[_reason_bucket(event.get("reason", "unknown"))].append(event)

    summary: dict[str, dict[str, Any]] = {}
    for reason, events in sorted(by_reason.items(), key=lambda item: len(item[1]), reverse=True):
        raw_reasons = Counter(event.get("reason", "unknown") or "unknown" for event in events)
        summary[reason] = {
            "count": len(events),
            "by_strategy": dict(Counter(event.get("strategy", "unknown") for event in events)),
            "by_symbol": dict(Counter(event.get("symbol", "unknown") for event in events)),
            "by_stage": dict(Counter(event.get("stage", "entry") for event in events)),
            "raw_reasons": dict(raw_reasons),
            "raw_reason_examples": [raw_reason for raw_reason, _ in raw_reasons.most_common(3)],
            "avg_risk_dollars": _avg([float(event.get("risk_dollars", 0.0) or 0.0) for event in events]),
            "avg_portfolio_open_R": _avg(_event_context_value(events, "portfolio_open_risk_R")),
            "avg_portfolio_request_R": _avg(_event_context_value(events, "portfolio_request_risk_R")),
            "avg_portfolio_after_R": _avg(_event_context_value(events, "portfolio_after_request_R")),
            "avg_strategy_open_R": _avg(_event_context_value(events, "strategy_open_risk_R")),
            "avg_strategy_request_R": _avg(_event_context_value(events, "strategy_request_risk_R")),
            "avg_strategy_after_R": _avg(_event_context_value(events, "strategy_after_request_R")),
            "avg_strategy_daily_realized_R": _avg(_event_context_value(events, "strategy_daily_realized_R")),
            "outcome_backfilled": False,
        }
    return summary


def _reason_bucket(reason: Any) -> str:
    text = str(reason or "unknown").strip()
    lower = text.lower()
    if not text:
        return "unknown"
    if "portfolio daily stop" in lower:
        return "Portfolio daily stop"
    if "daily stop" in lower:
        return "Strategy daily stop"
    if "portfolio heat cap" in lower:
        return "Portfolio heat cap"
    if "heat reserved" in lower:
        return "Priority heat reserve"
    if "heat ceiling" in lower:
        prefix = text.split(" heat ceiling", 1)[0].strip()
        return f"{prefix} heat ceiling" if prefix else "Strategy heat ceiling"
    if "unknown strategy" in lower:
        return "Unknown strategy"
    return text


def _blocked_shadow_summary(entry_events: list[dict], data, *, horizon_bars: int = 120, fill_window_bars: int = 24) -> dict[str, Any]:
    rows = []
    for event in entry_events:
        if event.get("status") != "blocked":
            continue
        outcome = _shadow_event_outcome(event, data, horizon_bars=horizon_bars, fill_window_bars=fill_window_bars)
        if outcome is not None:
            rows.append(outcome)
    if not rows:
        return {
            "evaluated": 0,
            "horizon_bars": horizon_bars,
            "fill_window_bars": fill_window_bars,
            "filled_win_rate": 0.0,
            "avg_outcome_R": 0.0,
            "by_strategy": {},
            "by_reason": {},
            "broad_discriminators": [],
        }
    filled = [row for row in rows if row["filled"]]
    return {
        "evaluated": len(rows),
        "horizon_bars": horizon_bars,
        "fill_window_bars": fill_window_bars,
        "filled": len(filled),
        "unfilled": len(rows) - len(filled),
        "filled_win_rate": _safe_div(sum(1 for row in filled if row["outcome_R"] > 0.0), len(filled)),
        "avg_outcome_R": _avg([row["outcome_R"] for row in filled]),
        "avg_mfe_R": _avg([row["mfe_R"] for row in filled]),
        "avg_mae_R": _avg([row["mae_R"] for row in filled]),
        "by_strategy": _shadow_group(rows, "strategy"),
        "by_reason": _shadow_group(rows, "reason_bucket"),
        "by_signal_type": _shadow_group(rows, "signal_type"),
        "broad_discriminators": _shadow_broad_discriminators(rows),
        "method": (
            "Uses event-time direction, entry, and stop recorded before/at the risk gate. "
            "ATRSS stop-entry candidates must touch entry within the fill window; already-filled "
            "candidate types start immediately. If +1R and stop are both touched in a bar, the "
            "stop is counted first."
        ),
    }


def _shadow_event_outcome(
    event: dict,
    data,
    *,
    horizon_bars: int,
    fill_window_bars: int,
) -> dict[str, Any] | None:
    try:
        direction = int(event.get("direction"))
        entry = float(event.get("entry_price"))
        stop = float(event.get("stop_price"))
    except (TypeError, ValueError):
        return None
    if direction == 0 or entry <= 0 or stop <= 0:
        return None
    r_price = abs(entry - stop)
    if r_price <= 0:
        return None
    bars = _shadow_bars_for_event(event, data)
    if bars is None or len(bars.closes) == 0:
        return None

    event_time = np.datetime64(_aware_utc(event.get("time")).replace(tzinfo=None), "ns")
    start_idx = int(np.searchsorted(bars.times.astype("datetime64[ns]"), event_time, side="right"))
    if start_idx >= len(bars.closes):
        return None

    filled = bool(event.get("entry_already_filled", False))
    fill_idx = start_idx
    if not filled:
        fill_end = min(len(bars.closes), start_idx + max(1, fill_window_bars))
        for idx in range(start_idx, fill_end):
            if _entry_touched(direction, entry, bars.highs[idx], bars.lows[idx]):
                fill_idx = idx
                filled = True
                break
    if not filled:
        return _shadow_base_row(event, filled=False, outcome_R=0.0, mfe_R=0.0, mae_R=0.0)

    end_idx = min(len(bars.closes), fill_idx + max(1, horizon_bars))
    highs = bars.highs[fill_idx:end_idx]
    lows = bars.lows[fill_idx:end_idx]
    closes = bars.closes[fill_idx:end_idx]
    if len(closes) == 0:
        return None

    target = entry + direction * r_price
    outcome_r: float | None = None
    for high, low in zip(highs, lows, strict=False):
        if direction > 0:
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target
        if stop_hit:
            outcome_r = -1.0
            break
        if target_hit:
            outcome_r = 1.0
            break
    if outcome_r is None:
        outcome_r = direction * (float(closes[-1]) - entry) / r_price

    if direction > 0:
        mfe_r = (float(np.nanmax(highs)) - entry) / r_price
        mae_r = (entry - float(np.nanmin(lows))) / r_price
    else:
        mfe_r = (entry - float(np.nanmin(lows))) / r_price
        mae_r = (float(np.nanmax(highs)) - entry) / r_price
    return _shadow_base_row(
        event,
        filled=True,
        outcome_R=float(outcome_r),
        mfe_R=float(mfe_r),
        mae_R=float(mae_r),
    )


def _shadow_base_row(event: dict, *, filled: bool, outcome_R: float, mfe_R: float, mae_R: float) -> dict[str, Any]:
    context = event.get("risk_context", {}) if isinstance(event.get("risk_context"), dict) else {}
    return {
        "strategy": str(event.get("strategy", "unknown")),
        "symbol": str(event.get("symbol", "unknown")),
        "reason_bucket": _reason_bucket(event.get("reason", "unknown")),
        "signal_type": str(event.get("signal_type", "unknown") or "unknown"),
        "filled": filled,
        "outcome_R": outcome_R,
        "positive": filled and outcome_R > 0.0,
        "mfe_R": mfe_R,
        "mae_R": mae_R,
        "quality_score": event.get("quality_score"),
        "portfolio_open_R": float(context.get("portfolio_open_risk_R", 0.0) or 0.0),
        "portfolio_request_R": float(context.get("portfolio_request_risk_R", 0.0) or 0.0),
        "strategy_after_R": float(context.get("strategy_after_request_R", 0.0) or 0.0),
    }


def _shadow_bars_for_event(event: dict, data):
    strategy = event.get("strategy")
    symbol = str(event.get("symbol", ""))
    if strategy == "ATRSS":
        return data.atrss_hourly.get(symbol)
    if strategy == "BRS_R9":
        return data.brs_hourly.get(symbol)
    if strategy == "SWING_BREAKOUT_V3":
        return data.breakout_hourly.get(symbol)
    return data.hourly.get(symbol)


def _entry_touched(direction: int, entry: float, high: float, low: float) -> bool:
    return bool(high >= entry) if direction > 0 else bool(low <= entry)


def _shadow_group(rows: list[dict[str, Any]], key: str, *, min_count: int = 1) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        bucket: _shadow_stats(items)
        for bucket, items in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
        if len(items) >= min_count
    }


def _shadow_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    filled = [row for row in rows if row["filled"]]
    return {
        "count": len(rows),
        "filled": len(filled),
        "positive": sum(1 for row in filled if row["outcome_R"] > 0.0),
        "win_rate": _safe_div(sum(1 for row in filled if row["outcome_R"] > 0.0), len(filled)),
        "avg_outcome_R": _avg([row["outcome_R"] for row in filled]),
        "avg_mfe_R": _avg([row["mfe_R"] for row in filled]),
        "avg_mae_R": _avg([row["mae_R"] for row in filled]),
    }


def _shadow_broad_discriminators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    discriminators: list[dict[str, Any]] = []
    for strategy in sorted({row["strategy"] for row in rows}):
        subset = [row for row in rows if row["strategy"] == strategy and row["filled"]]
        if len(subset) < 50:
            continue
        _append_quantile_discriminator(discriminators, subset, strategy, "quality_score", "quality")
        _append_quantile_discriminator(discriminators, subset, strategy, "portfolio_open_R", "portfolio open R")
        _append_quantile_discriminator(discriminators, subset, strategy, "strategy_after_R", "strategy after-request R")
    return discriminators


def _append_quantile_discriminator(
    out: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    strategy: str,
    field: str,
    label: str,
) -> None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    if len(values) < 50:
        return
    threshold = float(np.nanmedian(values))
    high = [row for row in rows if isinstance(row.get(field), (int, float)) and float(row[field]) >= threshold]
    low = [row for row in rows if isinstance(row.get(field), (int, float)) and float(row[field]) < threshold]
    if len(high) < 25 or len(low) < 25:
        return
    high_stats = _shadow_stats(high)
    low_stats = _shadow_stats(low)
    edge = high_stats["avg_outcome_R"] - low_stats["avg_outcome_R"]
    if abs(edge) < 0.15:
        return
    chosen = high_stats if edge > 0 else low_stats
    side = "high" if edge > 0 else "low"
    out.append(
        {
            "label": f"{strategy} {side} {label} split",
            **chosen,
            "threshold": threshold,
            "comparison_edge_R": edge,
        }
    )


def _monthly_flow(entry_events: list[dict]) -> dict[str, dict[str, Any]]:
    monthly: dict[str, Counter] = defaultdict(Counter)
    for event in entry_events:
        event_time = _aware_utc(event.get("time"))
        key = event_time.strftime("%Y-%m")
        monthly[key]["fired"] += 1
        monthly[key][str(event.get("status", "unknown"))] += 1
    return {
        month: {
            "fired": int(counts.get("fired", 0)),
            "accepted": int(counts.get("accepted", 0)),
            "blocked": int(counts.get("blocked", 0)),
            "block_rate": _safe_div(counts.get("blocked", 0), counts.get("fired", 0)),
        }
        for month, counts in sorted(monthly.items())
    }


def _crowding_summary(entry_events: list[dict]) -> dict[str, Any]:
    events = sorted(entry_events, key=lambda event: _aware_utc(event.get("time")))
    if not events:
        return {
            "pct_events_with_other_signal_within_4h": 0.0,
            "pct_blocked_with_other_signal_within_4h": 0.0,
            "pct_blocked_when_portfolio_heat_above_70pct": 0.0,
            "avg_portfolio_open_R_before_block": 0.0,
            "top_pairs_within_4h": {},
        }
    near_any: set[int] = set()
    near_blocked: set[int] = set()
    pair_counts: Counter[str] = Counter()
    blocked_ids = {index for index, event in enumerate(events) if event.get("status") == "blocked"}
    for i, left in enumerate(events):
        left_time = _aware_utc(left.get("time"))
        for j in range(i + 1, len(events)):
            right = events[j]
            delta = _aware_utc(right.get("time")) - left_time
            if delta > timedelta(hours=4):
                break
            near_any.update((i, j))
            if i in blocked_ids:
                near_blocked.add(i)
            if j in blocked_ids:
                near_blocked.add(j)
            pair = " / ".join(sorted((str(left.get("strategy", "")), str(right.get("strategy", "")))))
            pair_counts[pair] += 1

    blocked_events = [event for event in events if event.get("status") == "blocked"]
    blocked_open_r = _event_context_value(blocked_events, "portfolio_open_risk_R")
    blocked_caps = _event_context_value(blocked_events, "portfolio_heat_cap_R")
    hot_blocks = sum(
        1
        for open_r, cap_r in zip(blocked_open_r, blocked_caps, strict=False)
        if cap_r > 0 and open_r / cap_r >= 0.70
    )
    return {
        "pct_events_with_other_signal_within_4h": _safe_div(len(near_any), len(events)),
        "pct_blocked_with_other_signal_within_4h": _safe_div(len(near_blocked), len(blocked_ids)),
        "pct_blocked_when_portfolio_heat_above_70pct": _safe_div(hot_blocks, len(blocked_events)),
        "avg_portfolio_open_R_before_block": _avg(blocked_open_r),
        "top_pairs_within_4h": dict(pair_counts.most_common(10)),
    }


def _config_snapshot(effective: dict[str, Any], replay_config: UnifiedBacktestConfig) -> dict[str, Any]:
    return {
        "risk_stance": effective.get("risk_stance"),
        "initial_equity": float(effective.get("initial_equity", replay_config.initial_equity)),
        "portfolio_rules": effective.get("portfolio_rules", {}),
        "strategy_allocations": effective.get("strategy_allocations", {}),
        "symbol_risk_multipliers": effective.get("symbol_risk_multipliers", {}),
        "replay": {
            "heat_cap_R": replay_config.heat_cap_R,
            "portfolio_daily_stop_R": replay_config.portfolio_daily_stop_R,
            "overlay_enabled": replay_config.overlay_enabled,
            "overlay_max_pct": replay_config.overlay_max_pct,
            "overlay_bear_mode_max_pct": effective.get("strategy_allocations", {})
            .get("OVERLAY", {})
            .get("bear_mode_max_equity_pct"),
        },
    }


def _individual_reference(output_root: Path) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    for strategy, (folder, round_name) in STRATEGY_OUTPUTS.items():
        run_summary = output_root / folder / round_name / "run_summary.json"
        evaluation = output_root / folder / round_name / "round_evaluation.txt"
        diagnostics = output_root / folder / round_name / "round_final_diagnostics.txt"
        if not run_summary.exists():
            continue
        data = json.loads(run_summary.read_text(encoding="utf-8"))
        metrics = data.get("final_metrics", {})
        reference[strategy] = {
            "source": str(run_summary),
            "source_label": f"{folder}/{round_name}",
            "total_trades": metrics.get("total_trades"),
            "net_return_pct": metrics.get("net_return_pct"),
            "profit_factor": metrics.get("profit_factor"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", metrics.get("max_dd_pct")),
            "trades_per_month": metrics.get("trades_per_month", metrics.get("active_trades_per_month")),
            "win_rate": metrics.get("win_rate"),
            "avg_r": metrics.get("avg_r", metrics.get("expectancy_r")),
            "calmar": metrics.get("calmar"),
            "high_value_notes": _extract_high_value_notes(evaluation, diagnostics),
        }
    return reference


def _extract_high_value_notes(*paths: Path) -> list[str]:
    headings = {
        "Signal Extraction / Alpha Capture",
        "Signal Discrimination",
        "Entry Mechanism",
        "Trade Management",
        "Exit Mechanism",
        "Overall Verdict",
    }
    notes: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        capture = False
        for line in lines:
            stripped = line.strip()
            if stripped in headings:
                capture = True
                continue
            if capture and stripped:
                if "no report provided" not in stripped.lower():
                    _append_note(notes, stripped)
                capture = False
            if stripped.startswith("- "):
                note = stripped[2:].strip()
                lower = note.lower()
                if any(
                    token in lower
                    for token in (
                        "best symbol",
                        "best setup",
                        "best entry",
                        "best exit",
                        "worst",
                        "primary drag",
                        "losses are concentrated",
                        "positive fee-net pnl",
                    )
                ):
                    _append_note(notes, note)
            elif "primary drag" in stripped.lower():
                _append_note(notes, stripped.strip("* "))
            if len(notes) >= 6:
                return notes
    return notes


def _append_note(notes: list[str], note: str) -> None:
    if note and note not in notes:
        notes.append(note)


def _interpret(
    metrics: dict[str, Any],
    strategy_summary: dict[str, dict[str, Any]],
    block_summary: dict[str, dict[str, Any]],
    crowding: dict[str, Any],
) -> dict[str, Any]:
    dominant = max(block_summary.items(), key=lambda item: item[1].get("count", 0))[0] if block_summary else "none"
    most_blocked = max(strategy_summary.items(), key=lambda item: item[1].get("blocked", 0))[0] if strategy_summary else "none"
    fired = float(metrics.get("fired_entries", 0.0) or 0.0)
    closed = float(metrics.get("closed_trades", metrics.get("total_trades", 0.0)) or 0.0)
    return {
        "dominant_block_reason": dominant,
        "most_blocked_strategy": most_blocked,
        "fired_to_closed_conversion": _safe_div(closed, fired),
        "block_rate": float(metrics.get("block_rate", 0.0) or 0.0),
        "crowded_block_rate": crowding.get("pct_blocked_with_other_signal_within_4h", 0.0),
        "needs_blocked_outcome_backfill": False,
    }


def _events_by_strategy(entry_events: list[dict], *, status: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in entry_events:
        if event.get("status") == status:
            grouped[str(event.get("strategy", ""))].append(event)
    return grouped


def _event_context_value(events: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for event in events:
        context = event.get("risk_context", {})
        if not isinstance(context, dict):
            continue
        value = context.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _aware_utc(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    try:
        import pandas as pd

        dt = pd.Timestamp(value).to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _counter_text(raw: dict[str, int], *, limit: int = 3) -> str:
    items = sorted(raw.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def _tier_text(tiers: Any) -> str:
    if not tiers:
        return "none"
    parts = []
    for tier in tiers:
        try:
            drawdown, scale = tier
        except (TypeError, ValueError):
            continue
        parts.append(f"{float(drawdown):.0%}->{float(scale):.0%}")
    return ", ".join(parts) if parts else "none"


def _money_counter_text(raw: dict[str, float], *, limit: int = 4) -> str:
    items = sorted(raw.items(), key=lambda item: abs(float(item[1])), reverse=True)[:limit]
    return ", ".join(f"{key}:${float(value):+,.2f}" for key, value in items) if items else "none"


def _shadow_row(label: str, row: dict[str, Any]) -> str:
    return (
        f"| {label} | {row.get('count', 0)} | {row.get('filled', 0)} | {row.get('positive', 0)} | "
        f"{row.get('win_rate', 0.0):.1%} | {row.get('avg_outcome_R', 0.0):+.2f} | "
        f"{row.get('avg_mfe_R', 0.0):.2f} | {row.get('avg_mae_R', 0.0):.2f} |"
    )


def _weight_text(raw: dict[str, float]) -> str:
    return ", ".join(f"{key}:{float(value):.0%}" for key, value in sorted(raw.items())) if raw else "default"


def _example_text(raw: list[str], *, limit: int = 2) -> str:
    return "; ".join(str(item).replace("|", "/") for item in raw[:limit])


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _fmt_num(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.2f}"


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
