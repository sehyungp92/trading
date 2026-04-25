"""Helix v4.0 gating attribution — formal KEEP/REVIEW verdicts.

Extends helix_gate_opportunity_cost() with shadow trade simulation,
missed EV computation, and actionable filter verdicts.
"""
from __future__ import annotations

import numpy as np
from collections import Counter

from backtests.momentum.analysis.helix_shadow_tracker import HelixFilterStats, HelixShadowTracker


def helix_filter_attribution_report(
    gate_log: list[dict],
    setup_log: list[dict],
    trades: list,
    shadow_tracker: HelixShadowTracker | None = None,
) -> str:
    """Generate the Helix gating attribution report with verdicts.

    Args:
        gate_log: List of gate rejection dicts from engine.
        setup_log: List of detected setup dicts from engine.
        trades: Completed trade records.
        shadow_tracker: If provided, enables shadow trade simulation verdicts.
    """
    lines = ["=" * 60]
    lines.append("  HELIX v4.0 GATING ATTRIBUTION REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Per-gate rejection counts (gate_log entries are dataclass objects)
    blocked_log = [g for g in gate_log if getattr(g, 'decision', '') == 'blocked']
    placed_log = [g for g in gate_log if getattr(g, 'decision', '') == 'placed']

    # Summary counts
    total_detected = len(setup_log)
    total_blocked = len(blocked_log)
    total_passed = len(placed_log)
    lines.append(f"  Setups detected:           {total_detected}")
    lines.append(f"  Passed all gates:          {total_passed}")
    lines.append(f"  Blocked by gates:          {total_blocked}")
    lines.append(f"  Completed trades:          {len(trades)}")
    lines.append("")

    reasons = Counter()
    for entry in blocked_log:
        for reason in getattr(entry, 'block_reasons', []):
            reasons[reason] += 1

    if shadow_tracker and shadow_tracker.results:
        summaries = shadow_tracker.get_filter_summary()

        lines.append("  Per-Gate Attribution (with shadow trade simulation):")
        lines.append("")
        header = (
            f"  {'Gate':28s} {'Blocked':>7s} {'Filled':>7s} {'WR%':>6s} "
            f"{'AvgR':>7s} {'>1R%':>5s} {'>2R%':>5s} {'MissEV':>8s} "
            f"{'AvdLoss':>8s} {'NetEV':>8s} {'Verdict':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        all_gates = sorted(
            set(list(reasons.keys()) + list(summaries.keys())),
            key=lambda g: -reasons.get(g, 0),
        )

        for gate in all_gates:
            count = reasons.get(gate, 0)
            s = summaries.get(gate)
            if s and s.filled_count > 0:
                net_ev = s.net_missed_expectancy - s.net_avoided_loss
                verdict = "KEEP" if net_ev < 0 else "REVIEW"
                # Compute shadow win rate
                shadow_wr = 0.0
                filled_results = [
                    r for r in shadow_tracker.results
                    if r.filled and gate in r.candidate.filter_names
                ]
                if filled_results:
                    shadow_wr = sum(1 for r in filled_results if r.r_multiple > 0) / len(filled_results) * 100
                lines.append(
                    f"  {gate:28s} {count:7d} {s.filled_count:7d} "
                    f"{shadow_wr:5.1f}% {s.avg_shadow_r:+7.3f} "
                    f"{s.pct_reach_1r:4.0f}% {s.pct_hit_target:4.0f}% "
                    f"{s.net_missed_expectancy:+8.1f} {s.net_avoided_loss:8.1f} "
                    f"{net_ev:+8.1f} {verdict:>8s}"
                )
            else:
                lines.append(
                    f"  {gate:28s} {count:7d}     N/A    N/A     N/A"
                    f"   N/A   N/A      N/A      N/A      N/A"
                )

        lines.append("")
        lines.append("  Verdict interpretation:")
        lines.append("    KEEP   = gate prevents more loss than it misses (net EV < 0)")
        lines.append("    REVIEW = gate may be blocking profitable trades (net EV >= 0)")

    else:
        # Without shadow trades, just show rejection counts
        lines.append("  Per-Gate Rejection Counts:")
        lines.append("")
        header = f"  {'Gate':28s} {'Blocked':>7s} {'% of Total':>10s}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        total = len(gate_log)
        for gate, count in reasons.most_common():
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {gate:28s} {count:7d} {pct:9.1f}%")

        lines.append("")
        lines.append("  (Enable shadow tracking for full virtual R analysis)")

    # Setup class breakdown of rejections — join with setup_log by setup_id
    lines.append("")
    lines.append("  Rejections by Setup Class:")
    setup_map = {getattr(s, 'setup_id', ''): s for s in setup_log}
    cls_counts = Counter()
    for entry in blocked_log:
        sid = getattr(entry, 'setup_id', '')
        setup = setup_map.get(sid)
        cls_counts[getattr(setup, 'setup_class', 'unknown') if setup else 'unknown'] += 1
    for cls, cnt in cls_counts.most_common():
        pct = cnt / len(blocked_log) * 100 if blocked_log else 0
        lines.append(f"    {cls}: {cnt} ({pct:.1f}%)")

    # Direction breakdown
    lines.append("")
    lines.append("  Rejections by Direction:")
    dir_counts = Counter()
    for entry in blocked_log:
        sid = getattr(entry, 'setup_id', '')
        setup = setup_map.get(sid)
        d = getattr(setup, 'direction', 0) if setup else 0
        label = "LONG" if d == 1 else "SHORT" if d == -1 else str(d)
        dir_counts[label] += 1
    for d_label, cnt in dir_counts.most_common():
        lines.append(f"    {d_label}: {cnt}")

    # High-alignment rejections (most costly potential misses)
    lines.append("")
    lines.append("  High-Alignment Rejections (score >= 1):")
    high_align = []
    for entry in blocked_log:
        sid = getattr(entry, 'setup_id', '')
        setup = setup_map.get(sid)
        if setup and getattr(setup, 'alignment_score', 0) >= 1:
            high_align.append(entry)
    if high_align:
        ha_reasons = Counter()
        for e in high_align:
            for reason in getattr(e, 'block_reasons', []):
                ha_reasons[reason] += 1
        for gate, cnt in ha_reasons.most_common(5):
            lines.append(f"    {gate}: {cnt}")
    else:
        lines.append("    (none)")

    return "\n".join(lines)
