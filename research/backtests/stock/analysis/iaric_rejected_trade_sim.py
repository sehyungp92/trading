"""Priority 4: Rejected Trade Simulation Analysis.

Analyzes shadow tracker results to determine if the 18.8% rejection
rate is optimal and which gates are providing the most value.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from research.backtests.stock.analysis.iaric_shadow_tracker import IARICShadowTracker
from research.backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def iaric_rejected_trade_sim(
    trades: list[TradeRecord],
    shadow_tracker: IARICShadowTracker | None = None,
) -> str:
    """Analyze rejected trades to determine gate effectiveness."""
    lines = [_hdr("REJ-1  Rejected Trade Simulation")]

    if shadow_tracker is None or not shadow_tracker.completed:
        lines.append("  (no shadow data -- run with --shadow flag)")
        return "\n".join(lines)

    completed = list(shadow_tracker.completed)
    if not completed:
        lines.append("  No rejected setups were simulated.")
        return "\n".join(lines)

    # Overall rejection stats
    actual_wr = sum(1 for t in trades if t.is_winner) / len(trades) if trades else 0
    n_actual = len(trades)
    n_rejected = len(completed)
    total_candidates = n_actual + n_rejected

    lines.append(f"  Candidates: {total_candidates}")
    lines.append(f"  Accepted: {n_actual} ({n_actual/total_candidates:.1%})")
    lines.append(f"  Rejected: {n_rejected} ({n_rejected/total_candidates:.1%})")
    lines.append(f"  Actual WR: {actual_wr:.1%}")

    # Hypothetical P&L of rejected trades
    shadow_rs = [s.simulated_r for s in completed]
    shadow_winners = sum(1 for r in shadow_rs if r > 0)
    shadow_wr = shadow_winners / n_rejected if n_rejected else 0
    shadow_total_r = sum(shadow_rs)
    shadow_mean_r = np.mean(shadow_rs) if shadow_rs else 0

    lines.append(f"\n  Hypothetical Performance of Rejected Trades:")
    lines.append(f"    Shadow WR: {shadow_wr:.1%} (vs actual {actual_wr:.1%})")
    lines.append(f"    Shadow Mean R: {shadow_mean_r:+.3f}")
    lines.append(f"    Shadow Total R: {shadow_total_r:+.2f}")

    verdict = "CORRECT" if shadow_wr < actual_wr else "REVIEW"
    lines.append(f"    Verdict: Gates are {'effectively filtering' if verdict == 'CORRECT' else 'possibly over-filtering'} ({verdict})")

    # --- Per-gate attribution ---
    lines.append(f"\n  Per-Gate Rejection Analysis:")
    lines.append(f"    {'Gate':<25s} {'n':>5s} {'Hyp WR':>8s} {'Mean R':>8s} {'Total R':>9s} {'Verdict':>8s}")
    lines.append(f"    {'-'*67}")

    by_gate = shadow_tracker.get_filter_summary()
    for gate_name in sorted(by_gate, key=lambda g: len(by_gate[g]), reverse=True):
        setups = by_gate[gate_name]
        n = len(setups)
        if n == 0:
            continue
        gate_rs = [s.simulated_r for s in setups]
        gate_wr = sum(1 for r in gate_rs if r > 0) / n
        gate_mean_r = np.mean(gate_rs)
        gate_total_r = sum(gate_rs)
        gate_verdict = "KEEP" if gate_wr < actual_wr else "REVIEW"
        lines.append(
            f"    {gate_name:<25s} {n:>5d} {gate_wr:>7.1%} "
            f"{gate_mean_r:>+7.3f} {gate_total_r:>+8.2f} {gate_verdict:>8s}"
        )

    # --- False positive rate per gate ---
    lines.append(f"\n  False Positive Rate (% of blocked trades that would have been winners):")
    lines.append(f"    {'Gate':<25s} {'Blocked':>8s} {'Would Win':>10s} {'FP Rate':>8s}")
    lines.append(f"    {'-'*55}")
    for gate_name in sorted(by_gate, key=lambda g: len(by_gate[g]), reverse=True):
        setups = by_gate[gate_name]
        n = len(setups)
        if n == 0:
            continue
        would_win = sum(1 for s in setups if s.simulated_r > 0)
        fp_rate = would_win / n
        lines.append(f"    {gate_name:<25s} {n:>8d} {would_win:>10d} {fp_rate:>7.1%}")

    # --- Exit type distribution of shadow trades ---
    lines.append(f"\n  Shadow Exit Type Distribution:")
    exit_counts: dict[str, list[float]] = defaultdict(list)
    for s in completed:
        exit_counts[s.simulated_exit].append(s.simulated_r)
    for exit_type in sorted(exit_counts, key=lambda e: len(exit_counts[e]), reverse=True):
        rs = exit_counts[exit_type]
        n = len(rs)
        wr = sum(1 for r in rs if r > 0) / n
        lines.append(f"    {exit_type:<20s}: n={n}, WR={wr:.1%}, Mean R={np.mean(rs):+.3f}")

    # --- MFE/MAE of shadow trades ---
    lines.append(f"\n  Shadow Trade MFE/MAE:")
    mfes = [s.mfe_r for s in completed if s.mfe_r > 0]
    maes = [s.mae_r for s in completed if s.mae_r > 0]
    if mfes:
        lines.append(f"    MFE: Mean={np.mean(mfes):.3f}R, Median={np.median(mfes):.3f}R, P90={np.percentile(mfes, 90):.3f}R")
    if maes:
        lines.append(f"    MAE: Mean={np.mean(maes):.3f}R, Median={np.median(maes):.3f}R, P90={np.percentile(maes, 90):.3f}R")

    # Funnel report
    funnel = shadow_tracker.funnel_report()
    if funnel:
        lines.append(f"\n  Signal Funnel:")
        for funnel_line in funnel.strip().split("\n"):
            lines.append(f"    {funnel_line}")

    return "\n".join(lines)
