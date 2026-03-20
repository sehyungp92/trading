"""Filter effectiveness analysis for Breakout v3.3-ETF strategy backtests.

Consumes SignalEvent logs from BreakoutEngine to produce:
- Signal funnel report (daily evaluations → fills)
- Blocked reason table (per-gate rejection stats with virtual outcomes)
- Pending effectiveness (conversion rate, pending vs immediate performance)
- Filter cost analysis (per-ablation impact)

All functions accept list[SignalEvent] and/or list[BreakoutTradeRecord] and return str.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# 1. Signal funnel report
# ---------------------------------------------------------------------------

def breakout_signal_funnel(signal_events: list, trades: list) -> str:
    """Full pipeline funnel from evaluations to filled trades.

    Stages: evaluations → entry selected → allowed → filled
    """
    if not signal_events:
        return "=== Signal Funnel ===\n  No signal events recorded."

    lines = ["=== Breakout Signal Funnel ==="]

    total = len(signal_events)
    with_entry = [e for e in signal_events if e.entry_type_selected]
    allowed = [e for e in signal_events if e.allowed]
    blocked = [e for e in signal_events if not e.allowed and e.entry_type_selected]
    no_entry = [e for e in signal_events if not e.entry_type_selected]

    n_filled = sum(1 for t in trades if t.entry_type != "ADD") if trades else 0

    lines.append(f"  Hourly evaluations:  {total:6d}")
    lines.append(f"  No entry available:  {len(no_entry):6d} ({100 * len(no_entry) / total:.0f}%)")
    lines.append(f"  Entry type selected: {len(with_entry):6d} ({100 * len(with_entry) / total:.0f}%)")
    lines.append(f"    Allowed:           {len(allowed):6d} ({100 * len(allowed) / total:.0f}%)")
    lines.append(f"    Blocked:           {len(blocked):6d} ({100 * len(blocked) / total:.0f}%)")
    lines.append(f"  Orders filled:       {n_filled:6d}")

    # Per entry type breakdown
    lines.append("")
    for etype in ["A", "B", "C_standard", "C_continuation"]:
        et_events = [e for e in with_entry if e.entry_type_selected == etype]
        et_allowed = [e for e in et_events if e.allowed]
        et_blocked = [e for e in et_events if not e.allowed]
        if et_events:
            lines.append(
                f"  {etype:18s}: {len(et_events):5d} selected, "
                f"{len(et_allowed):5d} allowed, {len(et_blocked):5d} blocked"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Blocked reason table
# ---------------------------------------------------------------------------

def breakout_blocked_reason_table(signal_events: list) -> str:
    """Per-gate rejection stats: count, avg score, avg quality_mult."""
    blocked = [e for e in signal_events if not e.allowed and e.blocked_reason]
    if not blocked:
        return "=== Blocked Reason Table ===\n  No blocked signals."

    lines = ["=== Breakout Blocked Reason Table ==="]
    header = (
        f"  {'Reason':22s} {'Count':>6s} {'%':>6s} {'AvgScore':>9s} "
        f"{'AvgQMult':>9s} {'AvgRVOL_H':>9s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    total_blocked = len(blocked)
    reason_groups: dict[str, list] = defaultdict(list)
    for e in blocked:
        reason_groups[e.blocked_reason].append(e)

    # Sort by count descending
    for reason, events in sorted(reason_groups.items(), key=lambda kv: -len(kv[1])):
        count = len(events)
        pct = 100 * count / total_blocked
        avg_score = np.mean([e.score_total for e in events])
        avg_qmult = np.mean([e.quality_mult for e in events])
        avg_rvol = np.mean([e.rvol_h for e in events])
        lines.append(
            f"  {reason:22s} {count:6d} {pct:5.0f}% {avg_score:9.1f} "
            f"{avg_qmult:9.3f} {avg_rvol:9.2f}"
        )

    lines.append(f"\n  Total blocked: {total_blocked}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Pending effectiveness
# ---------------------------------------------------------------------------

def breakout_pending_effectiveness(signal_events: list, trades: list) -> str:
    """Pending mechanism analysis: conversion rate, pending vs immediate."""
    if not signal_events:
        return "=== Pending Effectiveness ===\n  No signal events."

    lines = ["=== Breakout Pending Effectiveness ==="]

    # Count blocked by PENDING gate vs allowed
    pending_blocked = [e for e in signal_events if e.blocked_reason == "PENDING"]
    if not pending_blocked:
        lines.append("  No pending-blocked signals found.")
        return "\n".join(lines)

    lines.append(f"  Pending-blocked signals: {len(pending_blocked)}")

    # Trades from continuation entries (likely converted from pending)
    if trades:
        cont_trades = [t for t in trades if t.continuation_at_entry]
        non_cont = [t for t in trades if not t.continuation_at_entry and t.entry_type != "ADD"]

        lines.append(f"\n  Continuation trades: {len(cont_trades)}")
        if cont_trades:
            lines.append(
                f"    Avg R:    {np.mean([t.r_multiple for t in cont_trades]):+.3f}"
            )
            lines.append(
                f"    Win rate: {np.mean([t.r_multiple > 0 for t in cont_trades]) * 100:.0f}%"
            )
        lines.append(f"  Non-continuation trades: {len(non_cont)}")
        if non_cont:
            lines.append(
                f"    Avg R:    {np.mean([t.r_multiple for t in non_cont]):+.3f}"
            )
            lines.append(
                f"    Win rate: {np.mean([t.r_multiple > 0 for t in non_cont]) * 100:.0f}%"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Chop mode gate analysis
# ---------------------------------------------------------------------------

def breakout_chop_gate_analysis(signal_events: list) -> str:
    """Break down signals by chop mode at evaluation time."""
    if not signal_events:
        return "=== Chop Gate Analysis ===\n  No signal events."

    lines = ["=== Breakout Chop Gate Analysis ==="]
    header = f"  {'ChopMode':10s} {'Total':>6s} {'Allowed':>8s} {'Blocked':>8s} {'Block%':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for mode in ["NORMAL", "DEGRADED", "HALT"]:
        events = [e for e in signal_events if e.chop_mode == mode and e.entry_type_selected]
        if not events:
            lines.append(f"  {mode:10s} {'0':>6s}")
            continue
        allowed = sum(1 for e in events if e.allowed)
        blocked = len(events) - allowed
        block_pct = 100 * blocked / len(events) if events else 0
        lines.append(
            f"  {mode:10s} {len(events):6d} {allowed:8d} {blocked:8d} {block_pct:6.0f}%"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Regime gate analysis
# ---------------------------------------------------------------------------

def breakout_regime_gate_analysis(signal_events: list) -> str:
    """Break down signals by 4H regime at evaluation time."""
    if not signal_events:
        return "=== Regime Gate Analysis ===\n  No signal events."

    lines = ["=== Breakout Regime Gate Analysis ==="]
    header = f"  {'Regime':14s} {'Total':>6s} {'Allowed':>8s} {'AvgScore':>9s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    regimes = sorted(set(e.regime_4h for e in signal_events if e.entry_type_selected))
    for regime in regimes:
        events = [e for e in signal_events if e.regime_4h == regime and e.entry_type_selected]
        if not events:
            continue
        allowed = sum(1 for e in events if e.allowed)
        avg_score = np.mean([e.score_total for e in events])
        lines.append(
            f"  {regime:14s} {len(events):6d} {allowed:8d} {avg_score:9.1f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Multiplier distribution
# ---------------------------------------------------------------------------

def breakout_multiplier_summary(signal_events: list) -> str:
    """Summary statistics for quality/expiry/displacement multipliers."""
    with_entry = [e for e in signal_events if e.entry_type_selected]
    if not with_entry:
        return "=== Multiplier Summary ===\n  No entry-selected signals."

    lines = ["=== Breakout Multiplier Summary ==="]

    for name, attr in [
        ("Quality", "quality_mult"),
        ("Expiry", "expiry_mult"),
        ("Displacement", "disp_mult"),
    ]:
        vals = [getattr(e, attr) for e in with_entry]
        arr = np.array(vals)
        lines.append(
            f"  {name:14s}: mean={np.mean(arr):.3f}  "
            f"med={np.median(arr):.3f}  min={np.min(arr):.3f}  max={np.max(arr):.3f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Score distribution
# ---------------------------------------------------------------------------

def breakout_score_distribution(signal_events: list) -> str:
    """Evidence score distribution for entry-selected signals."""
    with_entry = [e for e in signal_events if e.entry_type_selected]
    if not with_entry:
        return "=== Score Distribution ===\n  No entry-selected signals."

    lines = ["=== Breakout Score Distribution ==="]

    scores = [e.score_total for e in with_entry]
    arr = np.array(scores)
    lines.append(f"  Mean:   {np.mean(arr):.1f}")
    lines.append(f"  Median: {np.median(arr):.0f}")
    lines.append(f"  Min:    {np.min(arr)}")
    lines.append(f"  Max:    {np.max(arr)}")

    # Histogram by bins
    lines.append("")
    allowed_scores = [e.score_total for e in with_entry if e.allowed]
    blocked_scores = [e.score_total for e in with_entry if not e.allowed]

    all_scores = sorted(set(scores))
    header = f"  {'Score':>6s} {'Total':>6s} {'Allowed':>8s} {'Blocked':>8s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in all_scores:
        n_total = scores.count(s)
        n_allowed = allowed_scores.count(s)
        n_blocked = blocked_scores.count(s)
        lines.append(f"  {s:6d} {n_total:6d} {n_allowed:8d} {n_blocked:8d}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. First-fail attribution
# ---------------------------------------------------------------------------

# Gate precedence order matching breakout_engine._detect_and_place_entry()
_GATE_PRECEDENCE = [
    "CHOP_HALT",
    "HARD_BLOCK",
    "SCORE_THRESHOLD",
    "BREAKOUT_QUALITY",
    "MICRO_GUARD",
    "PENDING",
    "ENTRY_B_RVOL",
    "FRICTION_GATE",
    "DISPLACEMENT",
    "CORR_HEAT",
    "REENTRY_GATE",
]


def breakout_first_fail_attribution(signal_events: list) -> str:
    """Attribute each blocked signal to the first gate in precedence order.

    Standard get_filter_summary double-counts because one signal can be
    blocked by multiple gates.  This view assigns each to only the
    highest-precedence gate that blocked it.
    """
    blocked = [e for e in signal_events if not e.allowed and e.blocked_reason]
    if not blocked:
        return "=== First-Fail Attribution ===\n  No blocked signals."

    lines = ["=== Breakout First-Fail Attribution ==="]

    precedence_map = {g: i for i, g in enumerate(_GATE_PRECEDENCE)}

    first_fail_counts: dict[str, int] = defaultdict(int)
    for e in blocked:
        reason = e.blocked_reason
        first_fail_counts[reason] += 1

    header = f"  {'Gate':22s} {'Count':>6s} {'%':>6s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    total = len(blocked)
    for gate in sorted(first_fail_counts, key=lambda g: precedence_map.get(g, 999)):
        count = first_fail_counts[gate]
        lines.append(f"  {gate:22s} {count:6d} {100 * count / total:5.0f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Combined filter net-value report (ablation + shadow)
# ---------------------------------------------------------------------------

def breakout_combined_filter_value(
    shadow_summary: dict | None,
    ablation_deltas: dict | None = None,
) -> str:
    """Combined per-filter report merging shadow outcomes with ablation deltas.

    Parameters
    ----------
    shadow_summary : dict[str, BreakoutFilterStats]
        From BreakoutShadowTracker.get_filter_summary().
    ablation_deltas : dict[str, dict], optional
        Per-filter ablation results. Keys are filter names, values are dicts
        with optional keys: delta_trades, delta_expectancy, delta_sharpe,
        delta_max_dd.
    """
    if not shadow_summary:
        return "=== Combined Filter Value ===\n  No shadow data."

    lines = ["=== Breakout Combined Filter Net Value ==="]
    header = (
        f"  {'Gate':22s} {'Rej':>5s} {'ShadR':>7s} "
        f"{'Missed':>8s} {'Avoided':>8s} {'NetVal':>8s}"
    )
    if ablation_deltas:
        header += f" {'dTrades':>8s} {'dE[R]':>7s} {'dMaxDD':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for name in sorted(shadow_summary, key=lambda n: -shadow_summary[n].rejected_count):
        s = shadow_summary[name]
        net_val = s.net_avoided_loss - s.net_missed_expectancy
        row = (
            f"  {name:22s} {s.rejected_count:5d} {s.avg_shadow_r:+7.3f} "
            f"{s.net_missed_expectancy:+8.1f} {s.net_avoided_loss:8.1f} {net_val:+8.1f}"
        )
        if ablation_deltas and name in ablation_deltas:
            d = ablation_deltas[name]
            row += (
                f" {d.get('delta_trades', 0):+8d} "
                f"{d.get('delta_expectancy', 0):+7.3f} "
                f"{d.get('delta_max_dd', 0):+7.3f}"
            )
        lines.append(row)

    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    NetVal > 0  →  filter is net PROTECTIVE (keep)")
    lines.append("    NetVal < 0  →  filter costs more than it saves (investigate)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined filter attribution report
# ---------------------------------------------------------------------------

def breakout_filter_attribution_report(
    signal_events: list,
    trades: list,
    shadow_summary: dict | None = None,
    ablation_deltas: dict | None = None,
) -> str:
    """Generate full filter attribution report."""
    sections = [
        breakout_signal_funnel(signal_events, trades),
        breakout_blocked_reason_table(signal_events),
        breakout_first_fail_attribution(signal_events),
        breakout_pending_effectiveness(signal_events, trades),
        breakout_chop_gate_analysis(signal_events),
        breakout_regime_gate_analysis(signal_events),
        breakout_multiplier_summary(signal_events),
        breakout_score_distribution(signal_events),
    ]
    if shadow_summary:
        sections.append(breakout_combined_filter_value(shadow_summary, ablation_deltas))
    return "\n\n".join(sections)
