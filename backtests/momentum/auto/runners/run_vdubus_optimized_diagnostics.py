"""Run VdubusNQ with R1-optimized config defaults and full diagnostics.

Config.py now contains all R1 accepted mutations as defaults:
  MOM_N=65, MAX_POSITION_BARS_15M=64, EARLY_KILL_R=-0.40,
  PLUS_1R_PARTIAL_ENABLED=False, CHOP_THRESHOLD=40, BASE_RISK_PCT=0.02

Usage:
    cd trading
    python -u backtests/momentum/auto/runners/run_vdubus_optimized_diagnostics.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.momentum._aliases import install
install()

from backtests.momentum.analysis.metrics import compute_metrics
from backtests.momentum.analysis.reports import format_summary
from backtests.momentum.analysis.vdubus_diagnostics import vdubus_full_diagnostic
from backtests.momentum.analysis.vdubus_filter_attribution import (
    vdubus_filter_attribution_report,
)
from backtests.momentum.auto.scoring import composite_score, extract_metrics
from backtests.momentum.cli import _load_vdubus_data
from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
from backtests.momentum.engine.vdubus_engine import VdubusEngine

from strategy_3 import config as C

EQUITY = 10_000.0
POINT_VALUE = 2.0  # MNQ
DATA_DIR = ROOT / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "momentum" / "auto" / "vdubus" / "output"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 72)
    print("  VDUBUS R1-OPTIMIZED CONFIG -- FULL DIAGNOSTICS")
    print("=" * 72)

    # Show active config defaults (R1 mutations now baked in)
    r1_params = {
        "MOM_N": C.MOM_N,
        "MAX_POSITION_BARS_15M": C.MAX_POSITION_BARS_15M,
        "EARLY_KILL_R": C.EARLY_KILL_R,
        "PLUS_1R_PARTIAL_ENABLED": C.PLUS_1R_PARTIAL_ENABLED,
        "CHOP_THRESHOLD": C.CHOP_THRESHOLD,
        "BASE_RISK_PCT": C.BASE_RISK_PCT,
    }
    print("\n  R1-optimized config defaults:")
    for k, v in r1_params.items():
        print(f"    {k}: {v}")
    print()

    # Patch strategy_3 config for MNQ (micro contract)
    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = 0.50
    C.NQ_SPEC["point_value"] = 2.0
    C.RT_COMM_FEES = 1.24

    try:
        # Build config with defaults -- no mutations needed
        config = VdubusBacktestConfig(
            initial_equity=EQUITY,
            data_dir=DATA_DIR,
            fixed_qty=10,
            flags=VdubusAblationFlags(
                heat_cap=False,
                viability_filter=False,
                plus_1r_partial=False,  # R1 mutation: disable +1R partial
            ),
        )

        # Load data
        print("  Loading NQ bar data...")
        vdubus_data = _load_vdubus_data("NQ", DATA_DIR)
        print(f"  Data loaded in {time.time() - t0:.1f}s")

        # Run engine
        print("  Running VdubusNQ engine...")
        t1 = time.time()
        engine = VdubusEngine("NQ", config)
        result = engine.run(
            vdubus_data["bars_15m"],
            vdubus_data.get("bars_5m"),
            vdubus_data["hourly"],
            vdubus_data["daily_es"],
            vdubus_data["hourly_idx_map"],
            vdubus_data["daily_es_idx_map"],
            vdubus_data.get("five_to_15_idx_map"),
        )
        print(f"  VdubusNQ: {len(result.trades)} trades in {time.time() - t1:.1f}s")

    finally:
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    if not result.trades:
        print("  ERROR: No trades produced. Aborting.")
        return

    # Compute metrics
    pnls = np.array([t.pnl_dollars for t in result.trades])
    risks = np.array([
        abs(t.entry_price - t.initial_stop) * POINT_VALUE * t.qty
        for t in result.trades
    ])
    holds = np.array([t.bars_held_15m for t in result.trades])
    comms = np.array([t.commission for t in result.trades])

    metrics = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=holds,
        trade_commissions=comms,
        equity_curve=result.equity_curve,
        timestamps=result.time_series,
        initial_equity=EQUITY,
    )

    # Composite score
    score = composite_score(extract_metrics(
        result.trades, result.equity_curve, result.time_series, EQUITY,
    ))
    print(f"\n  Composite Score: {score.total:.4f}")
    print(f"    Calmar={score.calmar_component:.4f}  PF={score.pf_component:.4f}  "
          f"InvDD={score.inv_dd_component:.4f}  Net={score.net_profit_component:.4f}")

    # Build report sections
    report_sections: list[str] = []

    report_sections.append("=" * 72)
    report_sections.append("  VDUBUS R1-OPTIMIZED DEFAULTS -- FULL DIAGNOSTICS")
    report_sections.append(f"  Equity: ${EQUITY:,.0f}  Fixed qty: 10 MNQ")
    report_sections.append(f"  Config: {json.dumps(r1_params, default=str)}")
    report_sections.append(f"  Score: {score.total:.4f}")
    report_sections.append("=" * 72)

    # Performance summary
    perf_lines = [
        f"=== VdubusNQ v4.0 Performance Report: NQ ===",
        f"Total trades:       {metrics.total_trades}",
        f"Win rate:           {metrics.win_rate:.1%}",
        f"Profit factor:      {metrics.profit_factor:.2f}",
        f"Expectancy (R):     {metrics.expectancy:+.3f}",
        f"Expectancy ($):     {metrics.expectancy_dollar:+,.2f}",
        f"Net profit:         ${metrics.net_profit:+,.2f}",
        f"CAGR:               {metrics.cagr:.1%}",
        f"Sharpe:             {metrics.sharpe:.2f}",
        f"Sortino:            {metrics.sortino:.2f}",
        f"Calmar:             {metrics.calmar:.2f}",
        f"Max drawdown:       {metrics.max_drawdown_pct:.1%} (${metrics.max_drawdown_dollar:,.2f})",
        f"Avg hold (15m bars):{metrics.avg_hold_hours:.1f}",
        f"Trades/month:       {metrics.trades_per_month:.1f}",
        f"Total commissions:  ${metrics.total_commissions:,.2f}",
    ]
    report_sections.append("\n".join(perf_lines))

    # Signal funnel
    funnel = [
        f"=== VdubusNQ Signal Funnel ===",
        f"  15m evaluations:  {result.evaluations}",
        f"  Regime passed:    {result.regime_passed}",
        f"  Signals found:    {result.signals_found}",
        f"  Entries placed:   {result.entries_placed}",
        f"  Entries filled:   {result.entries_filled}",
        f"  Trades completed: {len(result.trades)}",
    ]
    report_sections.append("\n".join(funnel))

    # Summary
    report_sections.append(format_summary(metrics))

    # Full diagnostics (all sections)
    report_sections.append(vdubus_full_diagnostic(
        result.trades,
        signal_events=result.signal_events,
        equity_curve=result.equity_curve,
        time_series=result.time_series,
    ))

    # Gating attribution
    if result.signal_events:
        report_sections.append(vdubus_filter_attribution_report(
            result.signal_events, result.trades,
            shadow_tracker=result.shadow_tracker,
        ))

    # Shadow trade report
    if result.shadow_summary:
        report_sections.append(result.shadow_summary)

    # Join and save
    full_report = "\n\n".join(report_sections)
    output_path = OUTPUT_DIR / "r1_final_diagnostics.txt"
    output_path.write_text(full_report, encoding="utf-8")

    try:
        print(f"\n{full_report}")
    except UnicodeEncodeError:
        print(full_report.encode("ascii", errors="replace").decode("ascii"))

    elapsed = time.time() - t0
    print(f"\n  Report saved to: {output_path}")
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
