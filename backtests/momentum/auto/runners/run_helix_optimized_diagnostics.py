"""Run Helix v4.0 with greedy-optimized config and full diagnostics.

Applies the 3 optimal mutations from greedy optimization:
  - use_momentum_stall: True  (+65% score)
  - use_drawdown_throttle: False  (+8% score)
  - vol_50_80_sizing_mult: 0.85  (+1.6% score)

Generates 34-section diagnostic report including 8 new deep strategy diagnostics.

Usage:
    cd trading
    python -u backtests/momentum/auto/runners/run_helix_optimized_diagnostics.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.momentum._aliases import install
install()

from backtests.momentum.analysis.helix_diagnostics import helix_full_diagnostic
from backtests.momentum.auto.config_mutator import mutate_helix_config
from backtests.momentum.auto.scoring import composite_score, extract_metrics
from backtests.momentum.cli import _load_helix_data_cached
from backtests.momentum.config_helix import Helix4BacktestConfig
from backtests.momentum.engine.helix_engine import Helix4Engine
from backtests.diagnostic_snapshot import build_group_snapshot

EQUITY = 10_000.0
POINT_VALUE = 2.0  # MNQ
DATA_DIR = ROOT / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "momentum" / "auto" / "output"

# Greedy-optimized mutations (from greedy_strategy_helix.json)
OPTIMIZED_MUTATIONS = {
    "flags.use_momentum_stall": True,
    "flags.use_drawdown_throttle": False,
    "flags.vol_50_80_sizing_mult": 0.85,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "helix_optimized_diagnostics.txt"))
    parser.add_argument("--json-output", default=str(OUTPUT_DIR / "helix_optimized.json"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 72)
    print("  HELIX v4.0 OPTIMIZED - FULL DIAGNOSTICS")
    print("=" * 72)

    # ── Load data ──
    print("\n  Loading bar data...")
    data = _load_helix_data_cached("NQ", DATA_DIR)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ── Build config with optimized mutations ──
    base_cfg = Helix4BacktestConfig(initial_equity=EQUITY, fixed_qty=10)
    cfg = mutate_helix_config(base_cfg, OPTIMIZED_MUTATIONS)

    print(f"\n  Applied mutations:")
    for k, v in OPTIMIZED_MUTATIONS.items():
        print(f"    {k} = {v}")

    # ── Run engine ──
    print(f"\n  Running Helix4 engine...")
    t1 = time.time()
    engine = Helix4Engine(symbol="NQ", bt_config=cfg)
    result = engine.run(
        data["minute_bars"], data["hourly"], data["four_hour"],
        data["daily"], data["hourly_idx_map"],
        data["four_hour_idx_map"], data["daily_idx_map"],
    )
    engine_elapsed = time.time() - t1
    print(f"  Helix4: {len(result.trades)} trades in {engine_elapsed:.1f}s")

    # ── Compute metrics and score ──
    ts_numeric = np.array([])
    if result.timestamps is not None and len(result.timestamps) > 0:
        ts_arr = result.timestamps
        if hasattr(ts_arr[0], "timestamp"):
            ts_numeric = np.array([dt.timestamp() for dt in ts_arr])
        else:
            ts_numeric = ts_arr

    metrics = extract_metrics(result.trades, result.equity_curve, ts_numeric, EQUITY)
    score = composite_score(metrics, EQUITY, strategy="helix", equity_curve=result.equity_curve)

    # ── Build report ──
    report_sections = [
        build_group_snapshot(
            "Momentum Helix Strength / Weakness Snapshot",
            result.trades,
            [
                ("setup class", lambda trade: getattr(trade, "setup_class", None)),
                ("session", lambda trade: getattr(trade, "session_at_entry", None)),
                ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
            ],
            min_count=5,
        )
    ]

    # Header
    header_lines = [
        "=" * 72,
        "  HELIX v4.0 OPTIMIZED DIAGNOSTICS",
        "=" * 72,
        "",
        "  Config: Greedy-optimized (3 mutations)",
        f"    use_momentum_stall = True   (+65.4% score)",
        f"    use_drawdown_throttle = False   (+8.0% score)",
        f"    vol_50_80_sizing_mult = 0.85   (+1.6% score)",
        "",
        "  SCORING",
        "  " + "-" * 55,
        f"    Composite Score:     {score.total:.4f}",
        f"      Net Profit (0.35): {score.net_profit_component:.4f}",
        f"      PF (0.30):         {score.pf_component:.4f}",
        f"      Calmar (0.20):     {score.calmar_component:.4f}",
        f"      Inv DD (0.15):     {score.inv_dd_component:.4f}",
    ]
    if score.rejected:
        header_lines.append(f"    REJECTED: {score.reject_reason}")
    header_lines.extend([
        "",
        "  PERFORMANCE SUMMARY",
        "  " + "-" * 55,
        f"    Total trades:        {metrics.total_trades}",
        f"    Win rate:            {metrics.win_rate:.1%}",
        f"    Profit factor:       {metrics.profit_factor:.2f}",
        f"    Net profit:          ${metrics.net_profit:,.0f}",
        f"    Max drawdown:        {metrics.max_drawdown_pct:.1%}",
        f"    Max drawdown ($):    ${metrics.max_drawdown_dollar:,.0f}",
        f"    Sharpe ratio:        {metrics.sharpe:.2f}",
        f"    Sortino ratio:       {metrics.sortino:.2f}",
        f"    Calmar ratio:        {metrics.calmar:.2f}",
        f"    Expectancy ($/trade): ${metrics.expectancy_dollar:,.0f}",
        f"    Trades/month:        {metrics.trades_per_month:.1f}",
        f"    Avg hold (hours):    {metrics.avg_hold_hours:.1f}",
        f"    Total commissions:   ${metrics.total_commissions:,.0f}",
        "",
        "  FUNNEL",
        "  " + "-" * 55,
        f"    Setups detected:     {result.setups_detected}",
        f"    Gates blocked:       {result.gates_blocked}",
        f"    Entries placed:      {result.entries_placed}",
        f"    Entries filled:      {result.entries_filled}",
    ])
    report_sections.append("\n".join(header_lines))

    # Full diagnostics (26 existing + 8 new deep strategy sections)
    report_sections.append(helix_full_diagnostic(
        result.trades,
        setup_log=result.setup_log,
        gate_log=result.gate_log,
        entry_tracking=result.entry_tracking,
        equity_curve=result.equity_curve,
        timestamps=result.timestamps,
    ))

    # Shadow trade report
    if result.shadow_summary:
        report_sections.append(result.shadow_summary)

    # Join and save
    full_report = "\n\n".join(report_sections)

    output_path = Path(args.output)
    output_path.write_text(full_report, encoding="utf-8")

    # Also save structured JSON
    json_data = {
        "mutations": OPTIMIZED_MUTATIONS,
        "score": score.total,
        "score_components": {
            "net_profit": score.net_profit_component,
            "pf": score.pf_component,
            "calmar": score.calmar_component,
            "inv_dd": score.inv_dd_component,
        },
        "metrics": {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "net_profit": metrics.net_profit,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "calmar": metrics.calmar,
            "expectancy_dollar": metrics.expectancy_dollar,
            "trades_per_month": metrics.trades_per_month,
        },
        "funnel": {
            "setups_detected": result.setups_detected,
            "gates_blocked": result.gates_blocked,
            "entries_placed": result.entries_placed,
            "entries_filled": result.entries_filled,
        },
    }
    json_path = Path(args.json_output)
    json_path.write_text(json.dumps(json_data, indent=2, default=str))

    try:
        print(f"\n{full_report}")
    except UnicodeEncodeError:
        print(full_report.encode("ascii", errors="replace").decode("ascii"))

    elapsed = time.time() - t0
    print(f"\n  Report saved to: {output_path}")
    print(f"  JSON saved to: {json_path}")
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
