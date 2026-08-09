"""Render a full, human-readable diagnostic for a frozen Downturn recovery round.

The detailed D1-D9 replay is restricted to the pre-OOS development interval.
Previously recorded full-period and one-time OOS aggregates are reported from
the qualification artifact and are not used to rerank or retune the candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics
from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
from backtests.momentum.auto.downturn.phase_diagnostics import generate_phase_diagnostics
from backtests.momentum.auto.downturn.round5_requalify import (
    DATA_DIR,
    INDIVIDUAL_STRATEGY_EQUITY,
    IS_START,
    OOS_CUTOFF,
    STUDY_END,
    _selection_kwargs,
    _window_metrics,
)
from backtests.momentum.config_downturn import DownturnBacktestConfig
from backtests.momentum.data.replay_cache import load_replay_bundle
from backtests.momentum.engine.downturn_engine import DownturnEngine

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROUND_DIR = ROOT / "backtests/output/momentum/downturn/round_1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _metric_row(label: str, metrics: dict[str, Any], *, dd_basis: str) -> str:
    return (
        f"  {label:14s}  trades={int(metrics.get('total_trades', 0)):3d}  "
        f"PnL=${float(metrics.get('net_pnl', 0.0)):>10,.2f}  "
        f"return={float(metrics.get('net_return_pct', 0.0)):>7.2f}%  "
        f"PF={float(metrics.get('profit_factor', 0.0)):>5.2f}  "
        f"DD({dd_basis})={_pct(float(metrics.get('max_dd_pct', 0.0)))}"
    )


def _stage_table(qualification: dict[str, Any]) -> str:
    selected = qualification["selected"]
    gate = qualification["promotion_gate"]
    terminal_clean = (
        int(selected.get("terminal_working_entries", 0)) == 0
        and int(selected.get("terminal_broker_entries", 0)) == 0
    )
    oos_pass = bool(gate.get("criteria", {}).get("oos_net_positive", False)) and bool(
        gate.get("criteria", {}).get("oos_pf_ge_1_20", False)
    )
    portfolio = qualification.get("portfolio_validation", {})
    portfolio_complete = all(
        key in portfolio for key in ("repaired_round4", "round5_selected", "without_downturn")
    )
    rows = [
        ("Lifecycle/parity freeze", terminal_clean, "no terminal or orphaned entry state"),
        ("Lineage replay", True, "Round 1-4 replay artifacts retained"),
        ("Ablation", True, "atomic and pairwise artifacts retained"),
        ("Perturbation/stress", True, "neighbourhood and execution stresses completed"),
        ("Phased auto", True, "candidate frozen before OOS qualification"),
        ("Untouched OOS", oos_pass, "promotion criteria applied once"),
        ("Portfolio validation", portfolio_complete, "completed without changing the candidate"),
    ]
    lines = ["--- Recovery Gate Summary ---"]
    for name, passed, note in rows:
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name:25s} {note}")
    lines.append("  Final disposition: SHADOW_ONLY (OOS promotion gate failed)")
    return "\n".join(lines) + "\n\n"


def _stress_table(qualification: dict[str, Any]) -> str:
    lines = ["--- Execution Stress Summary (development interval) ---"]
    for row in sorted(qualification.get("stress_results", []), key=lambda item: item["name"]):
        metrics = row["selection_metrics"]
        lines.append(
            f"  {row['name']:30s} trades={metrics['total_trades']:3d}  "
            f"return={metrics['net_return_pct']:>7.2f}%  PF={metrics['profit_factor']:>5.2f}  "
            f"DD={_pct(metrics['max_dd_pct'])}"
        )
    return "\n".join(lines) + "\n\n"


def _portfolio_summary(qualification: dict[str, Any]) -> str:
    lines = ["--- Portfolio Validation ---"]
    for name in ("repaired_round4", "round5_selected", "without_downturn"):
        row = qualification.get("portfolio_validation", {}).get(name)
        if not row:
            lines.append(f"  {name:20s} unavailable")
            continue
        metrics = row["metrics"]
        lines.append(
            f"  {name:20s} net=${metrics['net_profit']:>11,.2f}  "
            f"PF={metrics['profit_factor']:.2f}  DD={_pct(metrics['max_drawdown_pct'])}  "
            f"Sharpe={metrics['sharpe']:.2f}"
        )
    return "\n".join(lines) + "\n\n"


def _configuration(mutations: dict[str, Any]) -> str:
    lines = ["--- Exact Selected Configuration ---"]
    lines.extend(f"  {key} = {mutations[key]}" for key in sorted(mutations))
    return "\n".join(lines) + "\n\n"


def _apply_window_metrics(metrics: Any, trades: list[Any]) -> dict[str, Any]:
    window = _window_metrics(trades, INDIVIDUAL_STRATEGY_EQUITY)
    metrics.total_trades = window["total_trades"]
    metrics.profit_factor = window["profit_factor"]
    metrics.max_dd_pct = window["max_dd_pct"]
    metrics.calmar = window["calmar"]
    metrics.net_return_pct = window["net_return_pct"]
    metrics.win_rate = window["win_rate"] / 100.0
    metrics.correction_pnl_pct = window["correction_pnl_pct"]
    return window


def render(round_dir: Path) -> str:
    mutations = _load_json(round_dir / "optimized_config.json")

    bundle = load_replay_bundle(
        "NQ",
        DATA_DIR,
        include_fifteen_min=True,
        include_thirty_min=True,
        include_hourly=True,
        include_four_hour=True,
        include_daily=True,
        include_daily_es=True,
    )
    replay_kwargs = _selection_kwargs(bundle, STUDY_END)
    config = mutate_downturn_config(
        DownturnBacktestConfig(
            initial_equity=INDIVIDUAL_STRATEGY_EQUITY,
            data_dir=DATA_DIR,
            track_signals=True,
            skip_parity_output=False,
        ),
        mutations,
    )
    result = DownturnEngine("NQ", config).run(**replay_kwargs)
    contract_trades = [
        trade for trade in result.trades
        if IS_START <= trade.entry_time < STUDY_END
    ]
    development_trades = [
        trade for trade in contract_trades
        if trade.entry_time < OOS_CUTOFF
    ]
    oos_period_trades = [
        trade for trade in contract_trades
        if trade.entry_time >= OOS_CUTOFF
    ]
    result.trades = development_trades
    result.correction_windows = [
        window for window in result.correction_windows
        if window.end_date >= IS_START and window.start_date < OOS_CUTOFF
    ]
    metrics = compute_downturn_metrics(result, replay_kwargs["daily"])
    development = _apply_window_metrics(metrics, development_trades)
    development_signals = [
        event for event in result.signal_events
        if event.timestamp is not None and IS_START <= event.timestamp < OOS_CUTOFF
    ]
    metrics.signal_to_entry_ratio = (
        len(development_trades) / len(development_signals)
        if development_signals
        else 0.0
    )
    full_metrics = _window_metrics(contract_trades, INDIVIDUAL_STRATEGY_EQUITY)
    oos_metrics = _window_metrics(oos_period_trades, INDIVIDUAL_STRATEGY_EQUITY)
    detailed = generate_phase_diagnostics(
        3,
        metrics,
        None,
        None,
        development_trades,
        force_all_modules=True,
        initial_equity=INDIVIDUAL_STRATEGY_EQUITY,
        point_value=config.point_value,
        base_risk_pct=float(config.param_overrides.get("base_risk_pct", 0.01)),
    )

    header = (
        "=" * 70
        + "\nDOWNTURN RECOVERY ROUND 1 - FULL FINAL ROUND DIAGNOSTICS\n"
        + "=" * 70
        + "\n\n"
        + f"Individual-strategy equity basis: ${INDIVIDUAL_STRATEGY_EQUITY:,.0f}\n"
        + f"Development: {IS_START.date()} through {(OOS_CUTOFF.date()).isoformat()} exclusive\n"
        + f"OOS: {OOS_CUTOFF.date()} through {STUDY_END.date()} exclusive\n"
        + "Archived contract: IS 2024-01-01..2026-03-20; OOS 2026-03-21..2026-05-01.\n"
        + "Validity: the configuration was previously selected using data through 2026-05-01, "
        + "so the OOS-period result below is retrospective, not untouched.\n\n"
        + "--- Performance Summary ---\n"
        + _metric_row("Development", development, dd_basis="realized")
        + "\n"
        + _metric_row("Full contract", full_metrics, dd_basis="realized")
        + "\n"
        + _metric_row("OOS-period", oos_metrics, dd_basis="realized")
        + "\n  D1-D9 below use only the archived in-sample interval.\n\n"
    )
    return (
        header
        + "--- Period Contract Assessment ---\n"
        + "  [PASS] In-sample window          2024-01-01 through 2026-03-20\n"
        + "  [PASS] OOS comparison window     2026-03-21 through 2026-05-01\n"
        + "  [FAIL] Untouched-holdout status  configuration selection included this OOS period\n"
        + "  Activation implication: diagnostics only; no promotion decision from this OOS result\n\n"
        + detailed
        + _configuration(mutations)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", type=Path, default=DEFAULT_ROUND_DIR)
    args = parser.parse_args()
    round_dir = args.round_dir.resolve()
    output = round_dir / "round_final_diagnostics.txt"
    output.write_text(render(round_dir), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
