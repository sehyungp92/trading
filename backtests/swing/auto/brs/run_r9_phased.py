"""Round 9: Position management + final fine-tuning -- 4-phase greedy optimization.

Targets R8's quality gap (PF 7.41->4.70, DD 0.76%->3.47%) by adding:
  - Pyramiding: add to winning positions on new signals
  - Regime-adaptive sizing: reduce RANGE_CHOP, boost BEAR_TREND
  - Configurable quality multipliers: optimizer tunes chop/persistence penalties
  - Fine-tuning: narrow sweeps around R8-selected values

Starts from R8 locked baseline and lets the optimizer decide which
features improve the composite score.

Usage:
    python backtests/swing/auto/brs/run_r9_phased.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Aliases must be installed before any backtest imports
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from backtests.swing._aliases import install
install()

from backtests.swing.auto.brs.greedy_optimize import run_greedy, save_greedy_result
from backtests.swing.auto.brs.phase_candidates import get_phase_candidates
from backtests.swing.auto.brs.phase_gates import check_phase_gate
from backtests.swing.auto.brs.phase_scoring import PHASE_WEIGHTS
from backtests.swing.auto.brs.phase_state import (
    PhaseState, save_phase_state, load_phase_state,
)
from backtests.swing.auto.brs.scoring import extract_brs_metrics

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PHASE_STATE_PATH = OUTPUT_DIR / "phase_state.json"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# R8 baseline mutations (locked -- starting point for R9)
R8_BASELINE = {
    "disable_s1": False,
    "symbol_configs.GLD.adx_on": 16,
    "symbol_configs.GLD.adx_off": 14,
    "scale_out_enabled": True,
    "scale_out_pct": 0.33,
    "bd_donchian_period": 10,
    "lh_swing_lookback": 5,
    "scale_out_target_r": 3.6,
    "symbol_configs.QQQ.stop_floor_atr": 0.6,
    "symbol_configs.GLD.stop_buffer_atr": 0.18,
    "profit_floor_scale": 3.456,
    "peak_drop_enabled": True,
    "peak_drop_pct": -0.02,
    "peak_drop_lookback": 15,
    "bd_arm_bars": 24,
    "bias_4h_accel_enabled": True,
    "chop_short_entry_enabled": True,
    "persistence_override_bars": 5,
    "adx_strong": 25,
    "lh_arm_bars": 22,
    "bt_volume_mult": 2.0,
    "symbol_configs.QQQ.stop_buffer_atr": 0.3,
}

NUM_PHASES = 4
INITIAL_EQUITY = 100_000.0


def run_backtest_metrics(mutations: dict) -> dict:
    """Run backtest and return metrics dict."""
    from backtest.config_brs import BRSConfig
    from backtest.engine.brs_portfolio_engine import load_brs_data, run_brs_synchronized
    from backtests.swing.auto.brs.config_mutator import mutate_brs_config

    config = BRSConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR)
    config = mutate_brs_config(config, mutations)
    data = load_brs_data(config)
    result = run_brs_synchronized(data, config)
    metrics = extract_brs_metrics(result, INITIAL_EQUITY)
    return asdict(metrics)


def main():
    start = time.time()
    run_id = f"r9_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print("=" * 70)
    print("BRS R9 POSITION MANAGEMENT + FINE-TUNING -- PHASED OPTIMIZATION")
    print(f"Run ID: {run_id}")
    print(f"New features: pyramiding, regime_adaptive_sizing, configurable_quality_mults")
    print(f"Baseline: R8 cumulative mutations ({len(R8_BASELINE)} params)")
    print("=" * 70)

    # Backup current phase_state.json
    backup_path = OUTPUT_DIR / "phase_state_pre_r9_backup.json"
    if PHASE_STATE_PATH.exists() and not backup_path.exists():
        shutil.copy2(PHASE_STATE_PATH, backup_path)
        print(f"\nBacked up phase_state.json -> {backup_path.name}")

    # Initialize fresh phase state from R8 baseline
    state = PhaseState(
        current_phase=0,
        completed_phases=[],
        cumulative_mutations=dict(R8_BASELINE),
    )

    cumulative = dict(R8_BASELINE)
    phase_logs = []

    for phase in range(1, NUM_PHASES + 1):
        phase_start = time.time()
        print(f"\n{'='*70}")
        print(f"PHASE {phase}/4")
        print(f"{'='*70}")

        # Get candidates for this phase
        candidates = get_phase_candidates(phase, cumulative)
        print(f"  Candidates: {len(candidates)}")

        # Phase-specific weight overrides
        weights = PHASE_WEIGHTS.get(phase)

        # Run greedy optimization
        print(f"  Running greedy search...")
        greedy_result = run_greedy(
            data_dir=DATA_DIR,
            candidates=candidates,
            initial_equity=INITIAL_EQUITY,
            base_mutations=cumulative,
            phase=phase,
            verbose=True,
            min_delta=0.001,
            checkpoint_dir=OUTPUT_DIR,
            scoring_weight_overrides=weights,
        )

        # Update cumulative mutations
        cumulative = dict(greedy_result.final_mutations)

        # Save greedy result
        greedy_path = OUTPUT_DIR / f"r9_phase{phase}_greedy.json"
        save_greedy_result(greedy_result, greedy_path)

        # Compute final metrics for gate check
        metrics_dict = asdict(extract_brs_metrics.__wrapped__(greedy_result, INITIAL_EQUITY)) if hasattr(extract_brs_metrics, '__wrapped__') else run_backtest_metrics(cumulative)

        # Reconstruct BRSMetrics for gate check
        from backtests.swing.auto.brs.scoring import BRSMetrics
        gate_metrics = BRSMetrics(**{
            k: v for k, v in greedy_result.final_metrics.items()
            if k in BRSMetrics.__dataclass_fields__
        }) if greedy_result.final_metrics else BRSMetrics(**{
            k: v for k, v in metrics_dict.items()
            if k in BRSMetrics.__dataclass_fields__
        })

        # Check phase gate
        prior_metrics = state.phase_results.get(phase - 1, {}).get("final_metrics") if phase > 1 else None
        gate = check_phase_gate(phase, gate_metrics, asdict(greedy_result) if hasattr(greedy_result, '__dataclass_fields__') else {
            "n_rounds": greedy_result.n_rounds,
            "kept_features": greedy_result.kept_features,
        }, prior_metrics)

        phase_elapsed = time.time() - phase_start

        # Record phase
        phase_result = {
            "score_before": greedy_result.base_score,
            "score_after": greedy_result.final_score,
            "kept_features": greedy_result.kept_features,
            "n_candidates": len(candidates),
            "n_rounds": greedy_result.n_rounds,
            "elapsed_s": phase_elapsed,
            "final_metrics": greedy_result.final_metrics,
        }
        state.advance_phase(phase, cumulative, phase_result)
        state.record_gate(phase, {
            "passed": gate.passed,
            "failure_category": gate.failure_category,
            "criteria": [(c.name, c.target, c.actual, c.passed) for c in gate.criteria],
        })
        state.phase_timestamps[phase] = {
            "start": datetime.fromtimestamp(phase_start, tz=timezone.utc).isoformat(),
            "end": datetime.now(timezone.utc).isoformat(),
        }

        # Save state after each phase (resumable)
        save_phase_state(state, PHASE_STATE_PATH)

        # Log summary
        summary = (
            f"  Phase {phase} complete ({phase_elapsed:.0f}s):\n"
            f"    Score: {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f} "
            f"(+{greedy_result.final_score - greedy_result.base_score:.4f})\n"
            f"    Kept: {greedy_result.kept_features or '(none)'}\n"
            f"    Trades={greedy_result.final_trades}, PF={greedy_result.final_pf:.2f}, "
            f"DD={greedy_result.final_dd_pct:.2%}, Return={greedy_result.final_return_pct:.1f}%\n"
            f"    Gate: {'PASS' if gate.passed else 'FAIL'}"
            f"{f' ({gate.failure_category})' if not gate.passed else ''}"
        )
        print(summary)
        phase_logs.append(summary)

        if not gate.passed:
            print(f"\n  WARNING: Phase {phase} gate FAILED -- continuing anyway (informational)")
            if gate.recommendations:
                for r in gate.recommendations:
                    print(f"    - {r}")

    # Final diagnostics
    total_elapsed = time.time() - start
    print(f"\n{'='*70}")
    print("R9 POSITION MANAGEMENT + FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"\nCumulative mutations ({len(cumulative)}):")
    for k, v in sorted(cumulative.items()):
        changed = k not in R8_BASELINE or R8_BASELINE[k] != v
        marker = " [NEW]" if k not in R8_BASELINE else (" [CHANGED]" if changed else "")
        print(f"  {k}: {v}{marker}")

    new_mutations = {k: v for k, v in cumulative.items() if k not in R8_BASELINE or R8_BASELINE[k] != v}
    print(f"\nNew/changed mutations: {len(new_mutations)}")
    for k, v in sorted(new_mutations.items()):
        old = R8_BASELINE.get(k, "(new)")
        print(f"  {k}: {old} -> {v}")

    # Run final comparison
    print("\nComputing R8 baseline vs R9 final metrics...")
    r8_metrics = run_backtest_metrics(R8_BASELINE)
    r9_metrics = run_backtest_metrics(cumulative)

    comparison_keys = [
        ("Total Trades", "total_trades", None, "higher"),
        ("Profit Factor", "profit_factor", 4.0, "higher"),
        ("Max DD %", "max_dd_pct", 0.02, "lower"),
        ("Net Return %", "net_return_pct", 25.0, "higher"),
        ("Sharpe", "sharpe", 1.0, "higher"),
        ("Calmar", "calmar", None, "higher"),
        ("Bear Alpha %", "bear_alpha_pct", None, "higher"),
        ("Detection Latency (days)", "detection_latency_days", 10.0, "lower"),
        ("Bias Latency (days)", "bias_latency_days", 8.0, "lower"),
        ("Crisis Coverage", "crisis_coverage", 0.5, "higher"),
    ]

    print(f"\n{'Metric':<30} {'R8':>12} {'R9':>12} {'Delta':>12}")
    print("-" * 68)
    for label, key, thresh, direction in comparison_keys:
        r8v = r8_metrics.get(key, 0)
        r9v = r9_metrics.get(key, 0)
        if isinstance(r8v, int):
            print(f"{label:<30} {r8v:>12d} {r9v:>12d} {r9v - r8v:>+12d}")
        else:
            print(f"{label:<30} {r8v:>12.3f} {r9v:>12.3f} {r9v - r8v:>+12.3f}")

    # Save full report
    report_path = OUTPUT_DIR / "r9_full_diagnostics.txt"
    with open(report_path, "w") as f:
        f.write(f"BRS R9 Position Management + Fine-Tuning\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Elapsed: {total_elapsed:.0f}s\n")
        f.write(f"Features: pyramiding, regime_adaptive_sizing, configurable_quality_mults\n\n")
        for log in phase_logs:
            f.write(log + "\n")
        f.write(f"\nCumulative mutations: {json.dumps(cumulative, indent=2)}\n")
        f.write(f"\nR8 metrics: {json.dumps(r8_metrics, indent=2)}\n")
        f.write(f"\nR9 metrics: {json.dumps(r9_metrics, indent=2)}\n")
    print(f"\nDiagnostics saved to {report_path.name}")


if __name__ == "__main__":
    main()
