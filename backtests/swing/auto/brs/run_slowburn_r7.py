"""Round 7: Slow-burn regime detection optimization.

Single-phase greedy from R6 baseline with targeted candidates.
Goal: reduce detection_latency_days and increase crisis_coverage
without breaching R6 performance thresholds.

Usage:
    python research/backtests/swing/auto/brs/run_slowburn_r7.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Aliases must be installed before any backtest imports
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from research.backtests.swing._aliases import install
install()

from research.backtests.swing.auto.brs.experiment_categories import get_category_experiments
from research.backtests.swing.auto.brs.greedy_optimize import run_greedy, save_greedy_result
from research.backtests.swing.auto.brs.scoring import extract_brs_metrics

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PHASE_STATE_PATH = OUTPUT_DIR / "phase_state.json"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Candidate name patterns targeting slow-burn regime detection
SLOWBURN_PATTERNS = [
    r"regime_adx_strong_\d+",
    r"regime_adx_on_\w+_\d+",
    r"regime_ema_fast_\d+",
    r"regime_ema_slow_\d+",
    r"struct_bear_min_\d+",
    r"struct_peak_drop_.*",
    r"struct_return_only_.*",
    r"struct_4h_accel_.*",
    r"struct_forming_.*",
    r"struct_peak_return_combo",
    r"struct_regime_all_three",
    r"struct_crash_return_.*",
    r"struct_crash_atr_ratio_.*",
    r"struct_cum_return_.*",
]

# R6 must-maintain thresholds
R6_THRESHOLDS = {
    "pf_min": 4.0,
    "dd_max_pct": 0.02,
    "return_min_pct": 25.0,
    "sharpe_min": 1.0,
}


def load_r6_mutations() -> dict:
    """Load R6 cumulative_mutations from phase_state.json."""
    with open(PHASE_STATE_PATH) as f:
        state = json.load(f)
    return state["cumulative_mutations"]


def filter_slowburn_candidates(
    candidates: list[tuple[str, dict]],
) -> list[tuple[str, dict]]:
    """Filter to slow-burn relevant experiments only."""
    compiled = [re.compile(p) for p in SLOWBURN_PATTERNS]
    filtered = []
    for name, muts in candidates:
        if any(pat.fullmatch(name) for pat in compiled):
            filtered.append((name, muts))
    return filtered


def run_final_diagnostics(mutations: dict, label: str) -> dict:
    """Run backtest with given mutations and return metrics dict."""
    from backtest.config_brs import BRSConfig
    from backtest.engine.brs_portfolio_engine import load_brs_data, run_brs_independent
    from research.backtests.swing.auto.brs.config_mutator import mutate_brs_config

    equity = 100_000.0
    config = BRSConfig(initial_equity=equity, data_dir=DATA_DIR)
    config = mutate_brs_config(config, mutations)
    data = load_brs_data(config)
    result = run_brs_independent(data, config)
    metrics = extract_brs_metrics(result, equity)
    return asdict(metrics)


def print_comparison(r6_metrics: dict, r7_metrics: dict) -> str:
    """Print side-by-side R6 vs R7 comparison. Returns report string."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("ROUND 7 vs ROUND 6 COMPARISON")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Metric':<30} {'R6':>12} {'R7':>12} {'Delta':>12} {'Status':>8}")
    lines.append("-" * 74)

    comparisons = [
        ("Total Trades", "total_trades", None, "higher"),
        ("Bear Trades", "bear_trades", None, "higher"),
        ("Profit Factor", "profit_factor", R6_THRESHOLDS["pf_min"], "higher"),
        ("Max DD %", "max_dd_pct", R6_THRESHOLDS["dd_max_pct"], "lower"),
        ("Net Return %", "net_return_pct", R6_THRESHOLDS["return_min_pct"], "higher"),
        ("Sharpe", "sharpe", R6_THRESHOLDS["sharpe_min"], "higher"),
        ("Calmar", "calmar", None, "higher"),
        ("Bear Alpha %", "bear_alpha_pct", None, "higher"),
        ("Detection Latency (days)", "detection_latency_days", 10.0, "lower"),
        ("Bias Latency (days)", "bias_latency_days", 8.0, "lower"),
        ("Crisis Coverage", "crisis_coverage", 0.5, "higher"),
        ("Bear Capture Ratio", "bear_capture_ratio", None, "higher"),
        ("Exit Efficiency", "exit_efficiency", None, "higher"),
    ]

    for label, key, threshold, direction in comparisons:
        r6_val = r6_metrics.get(key, 0)
        r7_val = r7_metrics.get(key, 0)

        if isinstance(r6_val, (int,)):
            delta_str = f"{r7_val - r6_val:+d}"
            r6_str = f"{r6_val:d}"
            r7_str = f"{r7_val:d}"
        else:
            delta_str = f"{r7_val - r6_val:+.3f}"
            r6_str = f"{r6_val:.3f}"
            r7_str = f"{r7_val:.3f}"

        if threshold is not None:
            if direction == "higher":
                ok = r7_val >= threshold
            else:
                ok = r7_val <= threshold
            status = "OK" if ok else "FAIL"
        else:
            if direction == "higher":
                status = "+" if r7_val > r6_val else ("=" if r7_val == r6_val else "-")
            else:
                status = "+" if r7_val < r6_val else ("=" if r7_val == r6_val else "-")

        lines.append(f"{label:<30} {r6_str:>12} {r7_str:>12} {delta_str:>12} {status:>8}")

    lines.append("")

    # Must-maintain check
    pf_ok = r7_metrics["profit_factor"] >= R6_THRESHOLDS["pf_min"]
    dd_ok = r7_metrics["max_dd_pct"] <= R6_THRESHOLDS["dd_max_pct"]
    ret_ok = r7_metrics["net_return_pct"] >= R6_THRESHOLDS["return_min_pct"]
    sh_ok = r7_metrics["sharpe"] >= R6_THRESHOLDS["sharpe_min"]
    lat_ok = r7_metrics["detection_latency_days"] < 10.0
    bias_ok = r7_metrics["bias_latency_days"] < 8.0
    cov_ok = r7_metrics["crisis_coverage"] >= 4 / 5

    lines.append("MUST-MAINTAIN THRESHOLDS:")
    lines.append(f"  PF >= 4.0:       {'PASS' if pf_ok else 'FAIL'} ({r7_metrics['profit_factor']:.2f})")
    lines.append(f"  DD <= 2.0%:      {'PASS' if dd_ok else 'FAIL'} ({r7_metrics['max_dd_pct']:.2%})")
    lines.append(f"  Return >= 25%:   {'PASS' if ret_ok else 'FAIL'} ({r7_metrics['net_return_pct']:.1f}%)")
    lines.append(f"  Sharpe >= 1.0:   {'PASS' if sh_ok else 'FAIL'} ({r7_metrics['sharpe']:.2f})")
    lines.append("")
    lines.append("MUST-IMPROVE TARGETS:")
    lines.append(f"  Trade Lat < 10d: {'PASS' if lat_ok else 'FAIL'} ({r7_metrics['detection_latency_days']:.1f}d)")
    lines.append(f"  Bias Lat < 8d:   {'PASS' if bias_ok else 'FAIL'} ({r7_metrics['bias_latency_days']:.1f}d)")
    lines.append(f"  Coverage >= 4/5: {'PASS' if cov_ok else 'FAIL'} ({r7_metrics['crisis_coverage']:.2f})")
    lines.append("")

    all_pass = pf_ok and dd_ok and ret_ok and sh_ok
    improved = lat_ok or bias_ok or cov_ok
    if all_pass and improved:
        lines.append(">>> ROUND 7 SUCCESS: thresholds maintained + detection improved <<<")
    elif all_pass:
        lines.append(">>> ROUND 7 PARTIAL: thresholds maintained but detection not improved <<<")
    else:
        lines.append(">>> ROUND 7 REGRESSION: one or more must-maintain thresholds breached <<<")

    report = "\n".join(lines)
    print(report)
    return report


def main():
    start = time.time()

    # 1. Load R6 baseline
    print("Loading R6 cumulative mutations...")
    r6_mutations = load_r6_mutations()
    print(f"  R6 mutations: {json.dumps(r6_mutations, indent=2)}")

    # 2. Backup phase_state.json
    backup_path = OUTPUT_DIR / "round6_phase_state_backup.json"
    if not backup_path.exists():
        shutil.copy2(PHASE_STATE_PATH, backup_path)
        print(f"  Backed up phase_state.json -> {backup_path.name}")
    else:
        print(f"  Backup already exists: {backup_path.name}")

    # 3. Build targeted candidate list
    all_candidates = get_category_experiments(["REGIME", "STRUCTURAL"])
    candidates = filter_slowburn_candidates(all_candidates)
    print(f"\nFiltered {len(candidates)} slow-burn candidates from {len(all_candidates)} REGIME+STRUCTURAL experiments:")
    for name, muts in candidates:
        print(f"  {name}: {muts}")

    # 4. R6 baseline metrics
    print("\nComputing R6 baseline metrics...")
    r6_metrics = run_final_diagnostics(r6_mutations, "R6")
    print(f"  R6: {r6_metrics['total_trades']} trades, PF={r6_metrics['profit_factor']:.2f}, "
          f"DD={r6_metrics['max_dd_pct']:.2%}, Return={r6_metrics['net_return_pct']:.1f}%, "
          f"TradeLat={r6_metrics['detection_latency_days']:.1f}d, "
          f"BiasLat={r6_metrics['bias_latency_days']:.1f}d, "
          f"Coverage={r6_metrics['crisis_coverage']:.2f}")

    # 5. Run greedy optimization
    print("\nStarting single-phase greedy optimization...")
    greedy_result = run_greedy(
        data_dir=DATA_DIR,
        candidates=candidates,
        initial_equity=100_000.0,
        base_mutations=r6_mutations,
        phase=1,
        verbose=True,
        min_delta=0.001,
        checkpoint_dir=OUTPUT_DIR,
    )

    # 6. Save result
    result_path = OUTPUT_DIR / "round7_slowburn_greedy.json"
    save_greedy_result(greedy_result, result_path)
    print(f"\nGreedy result saved to {result_path.name}")

    # 7. Final diagnostics
    print("\nComputing R7 final metrics...")
    r7_metrics = run_final_diagnostics(greedy_result.final_mutations, "R7")

    # 8. Comparison
    report = print_comparison(r6_metrics, r7_metrics)

    # Save diagnostics
    diag_path = OUTPUT_DIR / "round7_diagnostics.txt"
    with open(diag_path, "w") as f:
        f.write(f"Round 7 Slow-Burn Regime Detection Optimization\n")
        f.write(f"Elapsed: {time.time() - start:.0f}s\n")
        f.write(f"Kept features: {greedy_result.kept_features}\n")
        f.write(f"Final mutations: {json.dumps(greedy_result.final_mutations, indent=2)}\n\n")
        f.write(f"R6 metrics: {json.dumps(r6_metrics, indent=2)}\n\n")
        f.write(f"R7 metrics: {json.dumps(r7_metrics, indent=2)}\n\n")
        f.write(report)
    print(f"\nDiagnostics saved to {diag_path.name}")

    print(f"\nTotal elapsed: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
