"""Run NQDTC with greedy-optimized config and full diagnostics.

Applies the optimal mutations from greedy optimization (max_loss_cap,
max_stop_width) and generates the complete diagnostic report including
the new structural diagnostics sections.

Usage:
    cd trading
    python -u backtests/momentum/auto/runners/run_nqdtc_optimized_diagnostics.py
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

from backtests.momentum.analysis.metrics import compute_metrics
from backtests.momentum.analysis.reports import (
    format_summary,
    nqdtc_behavior_report,
    nqdtc_diagnostic_report,
    nqdtc_performance_report,
)
from backtests.momentum.analysis.nqdtc_diagnostics import nqdtc_full_diagnostic
from backtests.momentum.analysis.nqdtc_filter_attribution import (
    nqdtc_filter_attribution_report,
)
from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
from backtests.momentum.auto.scoring import composite_score, extract_metrics
from backtests.momentum.cli import _load_nqdtc_data
from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
from backtests.diagnostic_snapshot import build_group_snapshot

EQUITY = 10_000.0
POINT_VALUE = 2.0  # MNQ
DATA_DIR = ROOT / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "momentum" / "auto" / "output"
AUTO_OUTPUT = ROOT / "backtests" / "momentum" / "auto" / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-result",
        help="Load mutations from backtests/momentum/auto/nqdtc/output/phase_state.json using this phase id.",
    )
    parser.add_argument(
        "--mutation-kind",
        choices=["base", "final", "current"],
        default="final",
        help="When --phase-result is set, choose base_mutations, final_mutations, or cumulative_mutations.",
    )
    parser.add_argument("--output", help="Optional output path override.")
    parser.add_argument("--label", help="Optional report label override.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 72)
    print("  NQDTC OPTIMIZED CONFIG -- FULL DIAGNOSTICS")
    print("=" * 72)

    # Load optimal mutations -- prefer explicit phase selection, otherwise exit-opt > ablation > greedy
    nqdtc_out = ROOT / "backtests" / "momentum" / "auto" / "nqdtc" / "output"
    exit_opt_path = nqdtc_out / "exit_opt_result.json"
    ablation_path = nqdtc_out / "ablation_greedy_result.json"
    greedy_path = AUTO_OUTPUT / "greedy_strategy_nqdtc.json"
    phase_state_path = nqdtc_out / "phase_state.json"

    if args.phase_result:
        if not phase_state_path.exists():
            print(f"  ERROR: Missing phase state: {phase_state_path}")
            return
        phase_state = json.loads(phase_state_path.read_text())
        if args.mutation_kind == "current":
            mutations = phase_state.get("cumulative_mutations", {})
            greedy_data = {"baseline_score": 0.0, "final_score": 0.0}
            source_label = f"phase_{args.phase_result}_current"
        else:
            phase_result = phase_state.get("phase_results", {}).get(str(args.phase_result))
            if not phase_result:
                print(f"  ERROR: Phase {args.phase_result} not found in {phase_state_path}")
                return
            key = "base_mutations" if args.mutation_kind == "base" else "final_mutations"
            mutations = phase_result.get(key, {})
            greedy_data = {
                "baseline_score": phase_result.get("base_score", 0.0),
                "final_score": phase_result.get("final_score", 0.0),
            }
            source_label = f"phase_{args.phase_result}_{args.mutation_kind}"
        print(f"\n  Source: phase_state.json ({source_label})")
    elif exit_opt_path.exists():
        opt_data = json.loads(exit_opt_path.read_text())
        mutations = opt_data["final_mutations"]
        source_label = "exit-opt"
        print(f"\n  Source: exit alpha optimization result")
        print(f"  Base score:  {opt_data['base_score']:.4f}")
        print(f"  Final score: {opt_data['final_score']:.4f}")
        print(f"  Accepted:    {opt_data['accepted_count']} mutations")
        greedy_data = {"baseline_score": opt_data["base_score"], "final_score": opt_data["final_score"]}
    elif ablation_path.exists():
        opt_data = json.loads(ablation_path.read_text())
        mutations = opt_data["final_mutations"]
        source_label = "ablation"
        print(f"\n  Source: ablation greedy result")
        print(f"  Base score:  {opt_data['base_score']:.4f}")
        print(f"  Final score: {opt_data['final_score']:.4f}")
        print(f"  Accepted:    {opt_data['accepted_count']} mutations")
        greedy_data = {"baseline_score": opt_data["base_score"], "final_score": opt_data["final_score"]}
    elif greedy_path.exists():
        greedy_data = json.loads(greedy_path.read_text())
        mutations = greedy_data["accepted_mutations"]
        source_label = "greedy"
        print(f"\n  Source: greedy strategy result")
        print(f"  Baseline score: {greedy_data['baseline_score']:.4f}")
        print(f"  Final score:    {greedy_data['final_score']:.4f}")
    else:
        print("  ERROR: No optimization result found.")
        return

    print(f"\n  Applying {len(mutations)} optimal mutations:")
    for k, v in sorted(mutations.items()):
        print(f"    {k}: {v}")
    print()

    # Build config with mutations
    config = NQDTCBacktestConfig(
        initial_equity=EQUITY,
        data_dir=DATA_DIR,
        fixed_qty=10,
    )
    config = mutate_nqdtc_config(config, mutations)

    # Confirm flags
    print("  Active flags after mutation:")
    for f_name in sorted(vars(config.flags)):
        if not f_name.startswith("_"):
            print(f"    {f_name}: {getattr(config.flags, f_name)}")
    print()

    # Load data
    print("  Loading NQ bar data...")
    nqdtc_data = _load_nqdtc_data("NQ", DATA_DIR)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # Run engine
    print("  Running NQDTC engine...")
    t1 = time.time()
    engine = NQDTCEngine("MNQ", config)
    result = engine.run(
        nqdtc_data["five_min_bars"],
        nqdtc_data["thirty_min"],
        nqdtc_data["hourly"],
        nqdtc_data["four_hour"],
        nqdtc_data["daily"],
        nqdtc_data["thirty_min_idx_map"],
        nqdtc_data["hourly_idx_map"],
        nqdtc_data["four_hour_idx_map"],
        nqdtc_data["daily_idx_map"],
        daily_es=nqdtc_data.get("daily_es"),
        daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
    )
    print(f"  NQDTC: {len(result.trades)} trades in {time.time() - t1:.1f}s")

    if not result.trades:
        print("  ERROR: No trades produced. Aborting.")
        return

    # Compute metrics
    pnls = np.array([t.pnl_dollars for t in result.trades])
    risks = np.array([
        abs(t.entry_price - t.initial_stop) * POINT_VALUE * t.qty
        for t in result.trades
    ])
    holds = np.array([t.bars_held_30m for t in result.trades])
    comms = np.array([t.commission for t in result.trades])

    metrics = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=holds,
        trade_commissions=comms,
        equity_curve=result.equity_curve,
        timestamps=result.timestamps,
        initial_equity=EQUITY,
    )

    # Composite score
    score = composite_score(extract_metrics(result.trades, result.equity_curve, result.timestamps, EQUITY))
    print(f"\n  Composite Score: {score.total:.4f}")
    print(f"    Calmar={score.calmar_component:.4f}  PF={score.pf_component:.4f}  "
          f"InvDD={score.inv_dd_component:.4f}  Net={score.net_profit_component:.4f}")
    snapshot = build_group_snapshot(
        "NQDTC Strength / Weakness Snapshot",
        result.trades,
        [
            ("entry subtype", lambda trade: getattr(trade, "entry_subtype", None)),
            ("session", lambda trade: getattr(trade, "session", None)),
            ("regime", lambda trade: getattr(trade, "composite_regime", None)),
            ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
        ],
        min_count=5,
    )

    # Build report sections
    report_sections: list[str] = [snapshot]

    report_sections.append("=" * 72)
    report_sections.append(f"  {args.label or f'NQDTC OPTIMIZED CONFIG ({source_label})'}")
    report_sections.append(f"  Equity: ${EQUITY:,.0f}  Fixed qty: 10 MNQ")
    report_sections.append(f"  Mutations: {json.dumps(mutations, default=str)}")
    report_sections.append(f"  Score: {score.total:.4f} (baseline: {greedy_data.get('baseline_score', greedy_data.get('base_score', 0)):.4f})")
    report_sections.append("=" * 72)

    # Standard reports
    report_sections.append(nqdtc_performance_report("MNQ", metrics))
    report_sections.append(nqdtc_behavior_report(result.trades))
    report_sections.append(nqdtc_diagnostic_report(result))
    report_sections.append(format_summary(metrics))

    # Full diagnostics (29 existing + 4 new structural sections)
    report_sections.append(nqdtc_full_diagnostic(
        result.trades,
        signal_events=result.signal_events,
        equity_curve=result.equity_curve,
        initial_equity=EQUITY,
        point_value=POINT_VALUE,
    ))

    # Gating attribution
    if result.signal_events:
        report_sections.append(nqdtc_filter_attribution_report(
            result.signal_events, result.trades,
        ))

    # Shadow trade report
    if result.shadow_summary:
        report_sections.append(result.shadow_summary)

    # Join and save
    full_report = "\n\n".join(report_sections)
    nqdtc_output = ROOT / "backtests" / "momentum" / "auto" / "nqdtc" / "output"
    nqdtc_output.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else nqdtc_output / f"{source_label}_full_diagnostics.txt"
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
