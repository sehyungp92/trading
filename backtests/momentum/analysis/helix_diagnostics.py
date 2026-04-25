"""Diagnostic reports for Helix v4.0 NQ strategy backtests.

All functions accept lists of HelixTradeRecord and related logs, returning str.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# 1. Trade frequency by class x direction x session
# ---------------------------------------------------------------------------

def helix_trade_frequency(trades: list) -> str:
    if not trades:
        return "=== Trade Frequency ===\n  No trades."

    lines = ["=== Trade Frequency (class x direction x session) ==="]
    dir_map = {1: "LONG", -1: "SHORT"}
    header = f"  {'Class':6s} {'Dir':6s} {'Session':18s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    classes = sorted(set(t.setup_class for t in trades))
    sessions = sorted(set(t.session_at_entry for t in trades))

    for cls in classes:
        for d_val, d_label in dir_map.items():
            for sess in sessions:
                ct = [t for t in trades
                      if t.setup_class == cls and t.direction == d_val
                      and t.session_at_entry == sess]
                if not ct:
                    continue
                avg_r = np.mean([t.r_multiple for t in ct])
                wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
                lines.append(
                    f"  {cls:6s} {d_label:6s} {sess:18s} {len(ct):6d} {avg_r:+7.3f} {wr:5.0f}%"
                )

    lines.append(f"\n  Total trades: {len(trades)}")
    # Trades per week estimate
    if len(trades) >= 2:
        first_ts = min(t.entry_time for t in trades if t.entry_time)
        last_ts = max(t.exit_time for t in trades if t.exit_time)
        if first_ts and last_ts:
            weeks = max(1, (last_ts - first_ts).total_seconds() / (7 * 86400))
            lines.append(f"  Trades/week: {len(trades) / weeks:.1f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. R-distribution by class
# ---------------------------------------------------------------------------

def helix_r_distribution(trades: list) -> str:
    if not trades:
        return "=== R-Distribution ===\n  No trades."

    lines = ["=== R-Distribution by Class ==="]
    header = (
        f"  {'Class':6s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'MedR':>7s} "
        f"{'P&L':>10s} {'MFE':>6s} {'MAE':>6s} {'PF':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for cls in sorted(set(t.setup_class for t in trades)):
        ct = [t for t in trades if t.setup_class == cls]
        n = len(ct)
        rs = np.array([t.r_multiple for t in ct])
        wr = np.mean(rs > 0) * 100
        avg_r = float(np.mean(rs))
        med_r = float(np.median(rs))
        pnl = sum(t.pnl_dollars for t in ct)
        mfe = np.mean([t.mfe_r for t in ct])
        mae = np.mean([t.mae_r for t in ct])
        gross_win = float(np.sum(rs[rs > 0]))
        gross_loss = float(np.sum(np.abs(rs[rs < 0])))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        lines.append(
            f"  {cls:6s} {n:6d} {wr:5.0f}% {avg_r:+7.3f} {med_r:+7.3f} "
            f"{pnl:+10,.0f} {mfe:6.2f} {mae:6.2f} {pf:6.2f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Gate block rates
# ---------------------------------------------------------------------------

def helix_gate_blocks(gate_log: list) -> str:
    if not gate_log:
        return "=== Gate Blocks ===\n  No gate decisions logged."

    lines = ["=== Gate Block Rates ==="]
    total = len(gate_log)
    blocked = [g for g in gate_log if g.decision == "blocked"]
    placed = [g for g in gate_log if g.decision == "placed"]

    lines.append(f"  Total gate decisions: {total}")
    lines.append(f"  Placed: {len(placed)} ({100 * len(placed) / total:.0f}%)")
    lines.append(f"  Blocked: {len(blocked)} ({100 * len(blocked) / total:.0f}%)")

    if blocked:
        lines.append("\n  Block reasons:")
        reason_counts: dict[str, int] = {}
        for g in blocked:
            for reason in g.block_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason:32s} {count:5d} ({100 * count / len(blocked):.0f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Milestone progression
# ---------------------------------------------------------------------------

def helix_milestone_progression(trades: list) -> str:
    if not trades:
        return "=== Milestone Progression ===\n  No trades."

    lines = ["=== Milestone Progression ==="]
    n = len(trades)
    hit_1r = sum(1 for t in trades if t.hit_1r)

    lines.append(f"  Total trades: {n}")
    lines.append(f"  +1R BE: {hit_1r:5d} ({100 * hit_1r / n:.0f}%)")

    # Exit reason as milestone proxy
    profit_target = sum(1 for t in trades if t.exit_reason == "PROFIT_TARGET")
    lines.append(f"  Profit target exits: {profit_target:5d} ({100 * profit_target / n:.0f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Exit reason breakdown
# ---------------------------------------------------------------------------

def helix_exit_breakdown(trades: list) -> str:
    if not trades:
        return "=== Exit Reason Breakdown ===\n  No trades."

    lines = ["=== Exit Reason Breakdown ==="]
    header = f"  {'Reason':22s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for reason in sorted(set(t.exit_reason for t in trades)):
        ct = [t for t in trades if t.exit_reason == reason]
        n = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(t.pnl_dollars for t in ct)
        lines.append(
            f"  {reason:22s} {n:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Regime slicing (vol_pct, alignment score, session)
# ---------------------------------------------------------------------------

def helix_regime_slicing(trades: list) -> str:
    if not trades:
        return "=== Regime Slicing ===\n  No trades."

    lines = ["=== Regime Slicing ==="]

    # Vol percentile buckets
    lines.append("\n  By vol_pct bucket:")
    buckets = [(0, 20), (20, 50), (50, 80), (80, 95), (95, 101)]
    for lo, hi in buckets:
        ct = [t for t in trades if lo <= t.vol_pct_at_entry < hi]
        if not ct:
            continue
        avg_r = np.mean([t.r_multiple for t in ct])
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        lines.append(
            f"    vol {lo:2d}-{hi:2d}: {len(ct):4d} trades, avgR={avg_r:+.3f}, WR={wr:.0f}%"
        )

    # Alignment score
    lines.append("\n  By alignment score:")
    for score in sorted(set(t.alignment_at_entry for t in trades)):
        ct = [t for t in trades if t.alignment_at_entry == score]
        avg_r = np.mean([t.r_multiple for t in ct])
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        lines.append(
            f"    score={score}: {len(ct):4d} trades, avgR={avg_r:+.3f}, WR={wr:.0f}%"
        )

    # Session block
    lines.append("\n  By session block:")
    for sess in sorted(set(t.session_at_entry for t in trades)):
        ct = [t for t in trades if t.session_at_entry == sess]
        avg_r = np.mean([t.r_multiple for t in ct])
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        lines.append(
            f"    {sess:18s}: {len(ct):4d} trades, avgR={avg_r:+.3f}, WR={wr:.0f}%"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Heat budget analysis
# ---------------------------------------------------------------------------

def helix_heat_budget(entry_tracking: list) -> str:
    """Analyze risk_r per entry and heat cap saturation."""
    if not entry_tracking:
        return "=== Heat Budget Analysis ===\n  No entry tracking data."

    lines = ["=== Heat Budget Analysis ==="]

    risk_rs = np.array([e.risk_r for e in entry_tracking])
    lines.append(f"  Entries placed: {len(entry_tracking)}")
    lines.append(f"  Risk-R per entry:  min={risk_rs.min():.3f}  "
                 f"median={np.median(risk_rs):.3f}  "
                 f"mean={risk_rs.mean():.3f}  max={risk_rs.max():.3f}")

    # Stop distance in points
    stop_dists = np.array([abs(e.entry_stop - e.stop0) for e in entry_tracking])
    lines.append(f"  Stop distance (pts): min={stop_dists.min():.1f}  "
                 f"median={np.median(stop_dists):.1f}  "
                 f"mean={stop_dists.mean():.1f}  max={stop_dists.max():.1f}")

    # Unit1 risk
    unit1s = np.array([e.unit1_risk for e in entry_tracking])
    lines.append(f"  Unit1 risk ($):  min={unit1s.min():.0f}  "
                 f"median={np.median(unit1s):.0f}  "
                 f"mean={unit1s.mean():.0f}  max={unit1s.max():.0f}")

    # Contracts
    ctrs = np.array([e.contracts for e in entry_tracking])
    lines.append(f"  Contracts:       min={ctrs.min():.0f}  "
                 f"median={np.median(ctrs):.0f}  "
                 f"mean={ctrs.mean():.1f}  max={ctrs.max():.0f}")

    # Heat state at arm time
    heat_totals = np.array([e.heat_total_r for e in entry_tracking])
    heat_dirs = np.array([e.heat_dir_r for e in entry_tracking])
    lines.append(f"\n  Heat at arm time:")
    lines.append(f"    Total heat (R): min={heat_totals.min():.3f}  "
                 f"median={np.median(heat_totals):.3f}  max={heat_totals.max():.3f}")
    lines.append(f"    Dir heat (R):   min={heat_dirs.min():.3f}  "
                 f"median={np.median(heat_dirs):.3f}  max={heat_dirs.max():.3f}")

    # Breakdown by class
    lines.append(f"\n  By class:")
    for cls in sorted(set(e.setup_class for e in entry_tracking)):
        sub = [e for e in entry_tracking if e.setup_class == cls]
        r_arr = np.array([e.risk_r for e in sub])
        sd_arr = np.array([abs(e.entry_stop - e.stop0) for e in sub])
        filled = sum(1 for e in sub if e.filled)
        lines.append(
            f"    {cls}: {len(sub)} placed, {filled} filled ({100 * filled / len(sub):.0f}%), "
            f"avgRiskR={r_arr.mean():.3f}, avgStop={sd_arr.mean():.1f}pts"
        )

    # Breakdown by direction
    dir_map = {1: "LONG", -1: "SHORT"}
    lines.append(f"\n  By direction:")
    for d_val, d_label in dir_map.items():
        sub = [e for e in entry_tracking if e.direction == d_val]
        if not sub:
            continue
        r_arr = np.array([e.risk_r for e in sub])
        filled = sum(1 for e in sub if e.filled)
        lines.append(
            f"    {d_label}: {len(sub)} placed, {filled} filled ({100 * filled / len(sub):.0f}%), "
            f"avgRiskR={r_arr.mean():.3f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Entry fill-rate analysis
# ---------------------------------------------------------------------------

def helix_fill_rate_analysis(entry_tracking: list) -> str:
    """Analyze why placed entries don't fill: distance to trigger, closest approach."""
    if not entry_tracking:
        return "=== Entry Fill-Rate Analysis ===\n  No entry tracking data."

    lines = ["=== Entry Fill-Rate Analysis ==="]

    filled = [e for e in entry_tracking if e.filled]
    unfilled = [e for e in entry_tracking if not e.filled]
    total = len(entry_tracking)

    lines.append(f"  Total placed: {total}  Filled: {len(filled)} ({100 * len(filled) / total:.1f}%)  "
                 f"Unfilled: {len(unfilled)} ({100 * len(unfilled) / total:.1f}%)")

    # Distance from arm price to entry trigger at placement time
    if entry_tracking:
        trigger_dists = []
        for e in entry_tracking:
            if e.direction == 1:
                dist = e.entry_stop - e.arm_price  # positive = trigger above price
            else:
                dist = e.arm_price - e.entry_stop  # positive = trigger below price
            trigger_dists.append(dist)
        trigger_dists = np.array(trigger_dists)
        lines.append(f"\n  Distance to trigger at arm time (pts):")
        lines.append(f"    All:      min={trigger_dists.min():+.1f}  "
                     f"median={np.median(trigger_dists):+.1f}  "
                     f"mean={trigger_dists.mean():+.1f}  max={trigger_dists.max():+.1f}")
        if filled:
            fd = []
            for e in filled:
                fd.append(e.entry_stop - e.arm_price if e.direction == 1
                          else e.arm_price - e.entry_stop)
            fd = np.array(fd)
            lines.append(f"    Filled:   min={fd.min():+.1f}  "
                         f"median={np.median(fd):+.1f}  mean={fd.mean():+.1f}")
        if unfilled:
            ud = []
            for e in unfilled:
                ud.append(e.entry_stop - e.arm_price if e.direction == 1
                          else e.arm_price - e.entry_stop)
            ud = np.array(ud)
            lines.append(f"    Unfilled: min={ud.min():+.1f}  "
                         f"median={np.median(ud):+.1f}  mean={ud.mean():+.1f}")

    # Closest price approach for unfilled entries
    if unfilled:
        gaps = []
        for e in unfilled:
            if e.direction == 1:
                gap = e.entry_stop - e.closest_price  # positive = never reached
            else:
                gap = e.closest_price - e.entry_stop  # positive = never reached
            gaps.append(gap)
        gaps = np.array(gaps)
        lines.append(f"\n  Unfilled entry gap (pts remaining to trigger):")
        lines.append(f"    min={gaps.min():+.1f}  median={np.median(gaps):+.1f}  "
                     f"mean={gaps.mean():+.1f}  max={gaps.max():+.1f}")

        # How many came within N points
        for threshold in [1.0, 5.0, 10.0, 25.0, 50.0]:
            near = sum(1 for g in gaps if g <= threshold)
            lines.append(f"    Within {threshold:5.1f}pts: {near:4d} ({100 * near / len(gaps):.0f}%)")

        # Negative gap = price DID cross trigger but order didn't fill (race condition / TTL)
        crossed = sum(1 for g in gaps if g <= 0)
        if crossed > 0:
            lines.append(f"    Price crossed trigger but no fill: {crossed} "
                         f"({100 * crossed / len(gaps):.0f}%)")

    # Fill rate by class
    lines.append(f"\n  Fill rate by class:")
    for cls in sorted(set(e.setup_class for e in entry_tracking)):
        sub = [e for e in entry_tracking if e.setup_class == cls]
        f_count = sum(1 for e in sub if e.filled)
        lines.append(f"    {cls}: {f_count}/{len(sub)} ({100 * f_count / len(sub):.0f}%)")

    # Fill rate by session
    lines.append(f"\n  Fill rate by session:")
    for sess in sorted(set(e.session_block for e in entry_tracking)):
        sub = [e for e in entry_tracking if e.session_block == sess]
        f_count = sum(1 for e in sub if e.filled)
        lines.append(f"    {sess:18s}: {f_count}/{len(sub)} ({100 * f_count / len(sub):.0f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Setup-to-fill funnel
# ---------------------------------------------------------------------------

def helix_setup_funnel(setup_log: list, gate_log: list, entry_tracking: list) -> str:
    """Full funnel: detected → gate pass → placed → filled."""
    if not setup_log:
        return "=== Setup-to-Fill Funnel ===\n  No setup data."

    lines = ["=== Setup-to-Fill Funnel ==="]

    detected = len(setup_log)
    placed_decisions = [g for g in gate_log if g.decision == "placed"] if gate_log else []
    blocked_decisions = [g for g in gate_log if g.decision == "blocked"] if gate_log else []
    placed = len(placed_decisions)
    blocked = len(blocked_decisions)
    filled = sum(1 for e in entry_tracking if e.filled) if entry_tracking else 0
    unfilled = sum(1 for e in entry_tracking if not e.filled) if entry_tracking else 0

    lines.append(f"  Setups detected:  {detected:5d}")
    lines.append(f"  Gates passed:     {placed:5d} ({100 * placed / max(detected, 1):.0f}%)")
    lines.append(f"  Gates blocked:    {blocked:5d} ({100 * blocked / max(detected, 1):.0f}%)")
    lines.append(f"  Entries placed:   {len(entry_tracking):5d}")
    lines.append(f"  Entries filled:   {filled:5d} ({100 * filled / max(len(entry_tracking), 1):.0f}% of placed)")
    lines.append(f"  Entries unfilled: {unfilled:5d} ({100 * unfilled / max(len(entry_tracking), 1):.0f}% of placed)")
    lines.append(f"  Overall conversion: {100 * filled / max(detected, 1):.1f}% (detected → filled)")

    # Funnel by class
    lines.append(f"\n  Funnel by class:")
    for cls in sorted(set(s.setup_class for s in setup_log)):
        det = sum(1 for s in setup_log if s.setup_class == cls)
        plc = sum(1 for e in entry_tracking if e.setup_class == cls) if entry_tracking else 0
        fil = sum(1 for e in entry_tracking if e.setup_class == cls and e.filled) if entry_tracking else 0
        blk = sum(1 for g in blocked_decisions
                  if any(s.setup_id == g.setup_id and s.setup_class == cls for s in setup_log)
                  ) if gate_log else 0
        lines.append(
            f"    {cls}: {det} detected → {plc} placed → {fil} filled "
            f"({100 * fil / max(det, 1):.0f}% overall)"
        )

    # Funnel by direction
    dir_map = {1: "LONG", -1: "SHORT"}
    lines.append(f"\n  Funnel by direction:")
    for d_val, d_label in dir_map.items():
        det = sum(1 for s in setup_log if s.direction == d_val)
        plc = sum(1 for e in entry_tracking if e.direction == d_val) if entry_tracking else 0
        fil = sum(1 for e in entry_tracking if e.direction == d_val and e.filled) if entry_tracking else 0
        lines.append(
            f"    {d_label}: {det} detected → {plc} placed → {fil} filled "
            f"({100 * fil / max(det, 1):.0f}% overall)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. Gate block analysis by class and direction
# ---------------------------------------------------------------------------

def helix_gate_block_detail(setup_log: list, gate_log: list) -> str:
    """Detailed gate analysis: which gates block which classes/directions most."""
    if not gate_log:
        return "=== Gate Block Detail ===\n  No gate data."

    blocked = [g for g in gate_log if g.decision == "blocked"]
    if not blocked:
        return "=== Gate Block Detail ===\n  No blocked setups."

    # Build setup_id -> setup_log lookup
    setup_map = {}
    if setup_log:
        for s in setup_log:
            setup_map[s.setup_id] = s

    lines = ["=== Gate Block Detail (by class x reason) ==="]
    dir_map = {1: "LONG", -1: "SHORT"}

    # Aggregate: (class, direction, reason_prefix) -> count
    breakdown: dict[tuple[str, str, str], int] = {}
    for g in blocked:
        s = setup_map.get(g.setup_id)
        cls = s.setup_class if s else "?"
        d_label = dir_map.get(s.direction, "?") if s else "?"
        for reason in g.block_reasons:
            # Group similar reasons (e.g., heat_dir_1.70>1.20 -> heat_dir)
            prefix = reason.split("_")[0] + "_" + reason.split("_")[1] if "_" in reason else reason
            # Trim numeric suffixes for grouping
            for known in ("heat_dir", "heat_total", "spread", "spike_filter",
                          "min_stop", "corridor_cap", "extension_blocked",
                          "news_blocked", "extreme_vol", "session", "4h_structure"):
                if reason.startswith(known):
                    prefix = known
                    break
            key = (cls, d_label, prefix)
            breakdown[key] = breakdown.get(key, 0) + 1

    header = f"  {'Class':6s} {'Dir':6s} {'Gate':24s} {'Count':>6s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for (cls, d_label, reason), count in sorted(breakdown.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:6s} {d_label:6s} {reason:24s} {count:6d}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. Entry placement distance distribution
# ---------------------------------------------------------------------------

def helix_entry_distance_distribution(entry_tracking: list) -> str:
    """Histogram of trigger distance from current price at arm time."""
    if not entry_tracking:
        return "=== Entry Trigger Distance Distribution ===\n  No entry data."

    lines = ["=== Entry Trigger Distance Distribution ==="]

    # Compute directional distance: how far trigger is from current price
    # Positive = trigger ahead (needs move in direction), Negative = already past trigger
    dists = []
    for e in entry_tracking:
        if e.direction == 1:
            d = e.entry_stop - e.arm_price
        else:
            d = e.arm_price - e.entry_stop
        dists.append(d)
    dists = np.array(dists)

    # Bucket distribution
    buckets = [
        ("Already past trigger (<=0)", lambda x: x <= 0),
        ("0 to 10 pts", lambda x: 0 < x <= 10),
        ("10 to 25 pts", lambda x: 10 < x <= 25),
        ("25 to 50 pts", lambda x: 25 < x <= 50),
        ("50 to 100 pts", lambda x: 50 < x <= 100),
        ("100 to 200 pts", lambda x: 100 < x <= 200),
        ("200+ pts", lambda x: x > 200),
    ]

    lines.append(f"  {'Bucket':30s} {'Count':>6s} {'Fill%':>6s}")
    lines.append("  " + "-" * 44)

    for label, cond in buckets:
        indices = [i for i, d in enumerate(dists) if cond(d)]
        count = len(indices)
        if count == 0:
            continue
        filled_in_bucket = sum(1 for i in indices if entry_tracking[i].filled)
        fill_pct = 100 * filled_in_bucket / count if count > 0 else 0
        lines.append(f"  {label:30s} {count:6d} {fill_pct:5.0f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 13. R-multiple distribution with cumulative thresholds
# ---------------------------------------------------------------------------

def helix_r_distribution_cumulative(trades: list) -> str:
    if not trades:
        return "=== R-Multiple Distribution ===\n  No trades."

    lines = ["=== R-Multiple Distribution ==="]
    rs = np.array([t.r_multiple for t in trades])
    n = len(rs)

    lines.append(f"  Mean:   {np.mean(rs):+.3f}")
    lines.append(f"  Median: {np.median(rs):+.3f}")
    lines.append(f"  Std:    {np.std(rs):.3f}")

    if np.mean(rs) > np.median(rs):
        lines.append("  Skew:   POSITIVE (mean > median --letting winners run)")
    elif np.mean(rs) < np.median(rs):
        lines.append("  Skew:   NEGATIVE (mean < median --fat left tail)")
    else:
        lines.append("  Skew:   SYMMETRIC")

    lines.append(f"\n  Cumulative distribution:")
    for thresh in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0]:
        count = int(np.sum(rs >= thresh))
        pct = 100 * count / n
        lines.append(f"    >= {thresh:+.0f}R:  {count:5d} ({pct:5.1f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 14. Direction breakdown (long vs short)
# ---------------------------------------------------------------------------

def helix_direction_breakdown(trades: list) -> str:
    if not trades:
        return "=== Direction Breakdown ===\n  No trades."

    lines = ["=== Direction Breakdown ==="]
    header = f"  {'Direction':10s} {'N':>5s} {'WinR':>6s} {'AvgR':>7s} {'PF':>6s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, dir_val in [("Long", 1), ("Short", -1)]:
        group = [t for t in trades if t.direction == dir_val]
        if not group:
            continue
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100
        avg_r = np.mean(rs)
        total_pnl = sum(pnls)
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        pf_str = f"{pf:6.2f}" if pf < 100 else "   inf"
        lines.append(
            f"  {label:10s} {n:5d} {win_rate:5.0f}% {avg_r:+7.3f} {pf_str} ${total_pnl:+9.0f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 15. Monthly P&L
# ---------------------------------------------------------------------------

def helix_monthly_pnl(trades: list) -> str:
    if not trades:
        return "=== Monthly P&L ===\n  No trades."

    dated = [t for t in trades if t.entry_time]
    if not dated:
        return "=== Monthly P&L ===\n  No dated trades."

    lines = ["=== Monthly P&L ==="]
    header = f"  {'Month':10s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'NetPnL':>10s} {'CumPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    by_month: dict[str, list] = {}
    for t in dated:
        key = t.entry_time.strftime("%Y-%m")
        by_month.setdefault(key, []).append(t)

    cum_pnl = 0.0
    for month in sorted(by_month):
        group = by_month[month]
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        net = sum(t.pnl_dollars for t in group)
        cum_pnl += net
        lines.append(
            f"  {month:10s} {n:5d} {wr:5.0f}% {avg_r:+7.3f} ${net:+9.0f} ${cum_pnl:+9.0f}"
        )

    monthly_pnls = [sum(t.pnl_dollars for t in by_month[m]) for m in sorted(by_month)]
    win_months = sum(1 for p in monthly_pnls if p > 0)
    total_months = len(monthly_pnls)
    if total_months > 0:
        lines.append(f"\n  Winning months: {win_months}/{total_months} "
                     f"({100 * win_months / total_months:.0f}%)")

    if monthly_pnls:
        best_idx = int(np.argmax(monthly_pnls))
        worst_idx = int(np.argmin(monthly_pnls))
        months_sorted = sorted(by_month)
        lines.append(f"  Best month:  {months_sorted[best_idx]} (${monthly_pnls[best_idx]:+,.0f})")
        lines.append(f"  Worst month: {months_sorted[worst_idx]} (${monthly_pnls[worst_idx]:+,.0f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 16. Streak analysis
# ---------------------------------------------------------------------------

def helix_streak_analysis(trades: list) -> str:
    if not trades:
        return "=== Streak Analysis ===\n  No trades."

    lines = ["=== Streak Analysis ==="]
    max_win = max_loss = cur_win = cur_loss = 0
    worst_loss_streak_pnl = 0.0
    worst_loss_streak_r = 0.0
    cur_loss_pnl = 0.0
    cur_loss_r = 0.0
    win_streaks: list[int] = []
    loss_streaks: list[int] = []

    for t in trades:
        if t.r_multiple > 0:
            cur_win += 1
            if cur_loss > 0:
                loss_streaks.append(cur_loss)
            cur_loss = 0
            cur_loss_pnl = 0.0
            cur_loss_r = 0.0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_loss_pnl += t.pnl_dollars
            cur_loss_r += t.r_multiple
            if cur_win > 0:
                win_streaks.append(cur_win)
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
            worst_loss_streak_pnl = min(worst_loss_streak_pnl, cur_loss_pnl)
            worst_loss_streak_r = min(worst_loss_streak_r, cur_loss_r)

    if cur_win > 0:
        win_streaks.append(cur_win)
    if cur_loss > 0:
        loss_streaks.append(cur_loss)

    lines.append(f"  Max consecutive wins:   {max_win}")
    lines.append(f"  Max consecutive losses: {max_loss}")
    lines.append(f"  Worst consecutive-loss P&L: ${worst_loss_streak_pnl:+,.0f}")
    lines.append(f"  Worst consecutive-loss R:   {worst_loss_streak_r:+.2f}R")

    avg_win_streak = np.mean(win_streaks) if win_streaks else 0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0
    lines.append(f"  Avg win streak length:  {avg_win_streak:.1f}")
    lines.append(f"  Avg loss streak length: {avg_loss_streak:.1f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 17. Drawdown profile
# ---------------------------------------------------------------------------

def helix_drawdown_profile(
    equity_curve: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> str:
    if equity_curve is None or len(equity_curve) == 0:
        return "=== Drawdown Profile ===\n  No equity curve data available."

    lines = ["=== Drawdown Profile ==="]
    eq = np.asarray(equity_curve, dtype=float)

    peak = np.maximum.accumulate(eq)
    dd_pct = np.where(peak > 0, (eq - peak) / peak * 100, 0.0)

    max_dd_idx = int(np.argmin(dd_pct))
    max_dd = dd_pct[max_dd_idx]
    peak_idx = int(np.argmax(eq[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    dd_duration_bars = max_dd_idx - peak_idx

    lines.append(f"  Max drawdown: {max_dd:.2f}%")

    if timestamps is not None and len(timestamps) > max(max_dd_idx, peak_idx):
        peak_date = str(timestamps[peak_idx])[:10]
        trough_date = str(timestamps[max_dd_idx])[:10]
        lines.append(f"    Peak:   bar {peak_idx} ({peak_date})")
        lines.append(f"    Trough: bar {max_dd_idx} ({trough_date})")
        lines.append(f"    Duration: {dd_duration_bars} bars (~{dd_duration_bars:.0f} hours)")
    else:
        lines.append(f"    Duration: {dd_duration_bars} bars")

    lines.append(f"\n  Drawdown episode counts:")
    for thresh in [-0.5, -1.0, -2.0, -5.0]:
        in_dd = dd_pct < thresh
        episodes = 0
        was_in = False
        for v in in_dd:
            if v and not was_in:
                episodes += 1
            was_in = bool(v)
        lines.append(f"    Exceeding {thresh:+.1f}%: {episodes} episodes")

    underwater = np.sum(eq < peak)
    pct_underwater = 100 * underwater / len(eq) if len(eq) > 0 else 0
    lines.append(f"\n  Time underwater (below HWM): {pct_underwater:.1f}% of bars")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 18. Day-of-week analysis
# ---------------------------------------------------------------------------

def helix_day_of_week(trades: list) -> str:
    if not trades:
        return "=== Day-of-Week Analysis ===\n  No trades."

    dated = [t for t in trades if t.entry_time]
    if not dated:
        return "=== Day-of-Week Analysis ===\n  No dated trades."

    lines = ["=== Day-of-Week Analysis ==="]
    header = f"  {'Day':10s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day: dict[int, list] = {}
    for t in dated:
        dow = t.entry_time.weekday()
        by_day.setdefault(dow, []).append(t)

    for dow in sorted(by_day):
        group = by_day[dow]
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        pnl = sum(t.pnl_dollars for t in group)
        lines.append(f"  {day_names[dow]:10s} {n:5d} {wr:5.0f}% {avg_r:+7.3f} ${pnl:+9.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 19. Hourly performance
# ---------------------------------------------------------------------------

def helix_hourly_performance(trades: list) -> str:
    if not trades:
        return "=== Hourly Performance ===\n  No trades."

    dated = [t for t in trades if t.entry_time]
    if not dated:
        return "=== Hourly Performance ===\n  No dated trades."

    lines = ["=== Hourly Performance (ET) ==="]
    header = f"  {'Hour':6s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    by_hour: dict[int, list] = {}
    for t in dated:
        h = t.entry_time.hour
        by_hour.setdefault(h, []).append(t)

    for hour in sorted(by_hour):
        group = by_hour[hour]
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        pnl = sum(t.pnl_dollars for t in group)
        lines.append(f"  {hour:02d}:00 {n:5d} {wr:5.0f}% {avg_r:+7.3f} ${pnl:+9.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 20. Rolling expectancy
# ---------------------------------------------------------------------------

def helix_rolling_expectancy(trades: list, window: int = 10) -> str:
    if not trades or len(trades) < window:
        return f"=== Rolling Expectancy (window={window}) ===\n  Insufficient trades (need >= {window})."

    lines = [f"=== Rolling Expectancy (window={window}) ==="]
    sorted_trades = sorted(
        [t for t in trades if t.entry_time],
        key=lambda t: t.entry_time,
    )
    if len(sorted_trades) < window:
        lines.append(f"  Insufficient dated trades (need >= {window}).")
        return "\n".join(lines)

    rs = [t.r_multiple for t in sorted_trades]
    rolling = []
    for i in range(len(rs) - window + 1):
        rolling.append(np.mean(rs[i:i + window]))

    rolling = np.array(rolling)
    lines.append(f"  Windows computed: {len(rolling)}")
    lines.append(f"  Rolling E[R]:  min={np.min(rolling):+.3f}  max={np.max(rolling):+.3f}  "
                 f"current={rolling[-1]:+.3f}")

    neg_windows = int(np.sum(rolling < 0))
    lines.append(f"  Negative-expectancy windows: {neg_windows} ({100 * neg_windows / len(rolling):.0f}%)")

    mid = len(rolling) // 2
    if mid > 0:
        first_half = np.mean(rolling[:mid])
        second_half = np.mean(rolling[mid:])
        delta = second_half - first_half
        trend = "IMPROVING" if delta > 0.05 else "DEGRADING" if delta < -0.05 else "STABLE"
        lines.append(f"  Trend: {trend} (1st half={first_half:+.3f}, 2nd half={second_half:+.3f}, "
                     f"delta={delta:+.3f})")

    worst_idx = int(np.argmin(rolling))
    worst_start = sorted_trades[worst_idx].entry_time
    worst_end = sorted_trades[worst_idx + window - 1].entry_time
    lines.append(f"  Worst window: E[R]={rolling[worst_idx]:+.3f}  "
                 f"({worst_start.strftime('%Y-%m-%d')} to {worst_end.strftime('%Y-%m-%d')})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 21. Trade gap / dry spell analysis (frequency diagnostic)
# ---------------------------------------------------------------------------

def helix_trade_gaps(trades: list) -> str:
    """Analyze gaps between trades to identify frequency bottlenecks."""
    dated = sorted([t for t in trades if t.entry_time], key=lambda t: t.entry_time)
    if len(dated) < 2:
        return "=== Trade Gap Analysis ===\n  Insufficient trades."

    lines = ["=== Trade Gap Analysis (frequency diagnostic) ==="]

    # Compute inter-trade gaps in calendar days and hours
    gaps_hours = []
    gap_details = []
    for i in range(1, len(dated)):
        delta = dated[i].entry_time - dated[i - 1].exit_time
        hours = delta.total_seconds() / 3600
        gaps_hours.append(hours)
        gap_details.append({
            "hours": hours,
            "days": hours / 24,
            "start": dated[i - 1].exit_time,
            "end": dated[i].entry_time,
        })

    gaps = np.array(gaps_hours)
    lines.append(f"  Inter-trade gaps (exit → next entry):")
    lines.append(f"    Count:  {len(gaps)}")
    lines.append(f"    Mean:   {np.mean(gaps):.1f}h ({np.mean(gaps) / 24:.1f} days)")
    lines.append(f"    Median: {np.median(gaps):.1f}h ({np.median(gaps) / 24:.1f} days)")
    lines.append(f"    Min:    {np.min(gaps):.1f}h")
    lines.append(f"    Max:    {np.max(gaps):.1f}h ({np.max(gaps) / 24:.1f} days)")

    # Percentile distribution
    lines.append(f"\n  Gap percentiles (hours):")
    for p in [25, 50, 75, 90, 95]:
        val = np.percentile(gaps, p)
        lines.append(f"    P{p:2d}: {val:6.1f}h ({val / 24:.1f} days)")

    # Long dry spells (> 5 calendar days)
    long_gaps = [(i, g) for i, g in enumerate(gap_details) if g["days"] > 5]
    if long_gaps:
        lines.append(f"\n  Dry spells (>5 calendar days): {len(long_gaps)}")
        for i, g in sorted(long_gaps, key=lambda x: -x[1]["days"])[:8]:
            lines.append(f"    {g['days']:5.1f}d: {g['start'].strftime('%Y-%m-%d %H:%M')} → "
                         f"{g['end'].strftime('%Y-%m-%d %H:%M')}")
    else:
        lines.append(f"\n  Dry spells (>5 calendar days): 0")

    # Weekly trade density
    first_ts = dated[0].entry_time
    last_ts = dated[-1].entry_time
    total_weeks = max(1, (last_ts - first_ts).total_seconds() / (7 * 86400))

    # Calendar weeks with 0 trades
    from collections import defaultdict
    weekly_counts: dict[str, int] = defaultdict(int)
    # Generate all weeks in range
    import datetime as _dt
    start_monday = first_ts - _dt.timedelta(days=first_ts.weekday())
    week_cursor = start_monday
    all_weeks = []
    while week_cursor <= last_ts:
        wk = week_cursor.strftime("%Y-W%W")
        all_weeks.append(wk)
        week_cursor += _dt.timedelta(weeks=1)

    for t in dated:
        wk = t.entry_time.strftime("%Y-W%W")
        weekly_counts[wk] += 1

    zero_weeks = [w for w in all_weeks if weekly_counts.get(w, 0) == 0]
    lines.append(f"\n  Weekly trade density:")
    lines.append(f"    Total weeks in sample: {len(all_weeks)}")
    lines.append(f"    Weeks with 0 trades:   {len(zero_weeks)} ({100 * len(zero_weeks) / max(len(all_weeks), 1):.0f}%)")

    wk_counts = np.array([weekly_counts.get(w, 0) for w in all_weeks])
    lines.append(f"    Trades/week: min={wk_counts.min()}, median={np.median(wk_counts):.0f}, "
                 f"max={wk_counts.max()}, mean={wk_counts.mean():.1f}")

    # Longest consecutive zero-trade weeks
    max_consec = 0
    cur_consec = 0
    for w in all_weeks:
        if weekly_counts.get(w, 0) == 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0
    lines.append(f"    Longest zero-trade streak: {max_consec} consecutive weeks")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 22. Gate opportunity cost (which gates are most expensive?)
# ---------------------------------------------------------------------------

def helix_gate_opportunity_cost(setup_log: list, gate_log: list, shadow_results: list | None = None) -> str:
    """For each gate reason, show how many setups it blocks and estimated opportunity cost.

    If shadow_results are available, use actual simulated R; otherwise show counts only.
    """
    if not gate_log:
        return "=== Gate Opportunity Cost ===\n  No gate data."

    blocked = [g for g in gate_log if g.decision == "blocked"]
    if not blocked:
        return "=== Gate Opportunity Cost ===\n  No blocked setups."

    setup_map = {s.setup_id: s for s in setup_log} if setup_log else {}

    lines = ["=== Gate Opportunity Cost ==="]
    lines.append("  Which gates block the most setups and at what cost?\n")

    # Aggregate by normalized gate reason
    gate_groups: dict[str, list] = {}
    for g in blocked:
        for reason in g.block_reasons:
            # Normalize: strip numeric suffixes for grouping
            norm = reason
            for known in ("heat_dir", "heat_total", "spread", "spike_filter",
                          "min_stop", "corridor_cap", "extension_blocked",
                          "news_blocked", "extreme_vol", "high_vol",
                          "chop_zone", "session", "duplicate"):
                if reason.startswith(known):
                    norm = known
                    break
            gate_groups.setdefault(norm, []).append(g)

    # Sort by count descending
    header = (f"  {'Gate':28s} {'Blocked':>8s} {'% of all':>8s} "
              f"{'Unique setups':>14s} {'Avg align':>10s}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    total_blocked = len(blocked)
    for gate, decisions in sorted(gate_groups.items(), key=lambda x: -len(x[1])):
        n = len(decisions)
        pct = 100 * n / total_blocked
        unique_ids = set(d.setup_id for d in decisions)
        # Get alignment scores for blocked setups
        aligns = [setup_map[sid].alignment_score for sid in unique_ids
                  if sid in setup_map]
        avg_align = np.mean(aligns) if aligns else 0
        lines.append(
            f"  {gate:28s} {n:8d} {pct:7.0f}% {len(unique_ids):14d} {avg_align:10.1f}"
        )

    # Which gates block high-alignment (score>=1) setups?
    lines.append(f"\n  Gates blocking high-alignment setups (score >= 1):")
    for gate, decisions in sorted(gate_groups.items(), key=lambda x: -len(x[1])):
        high_align_ids = set()
        for d in decisions:
            s = setup_map.get(d.setup_id)
            if s and s.alignment_score >= 1:
                high_align_ids.add(d.setup_id)
        if high_align_ids:
            lines.append(f"    {gate:28s} blocks {len(high_align_ids)} high-alignment setups")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 23. MFE capture efficiency
# ---------------------------------------------------------------------------

def helix_mfe_capture(trades: list) -> str:
    """How much of the available favorable excursion does each trade capture?"""
    if not trades:
        return "=== MFE Capture Efficiency ===\n  No trades."

    lines = ["=== MFE Capture Efficiency ==="]
    lines.append("  How much of the peak unrealized profit (MFE) is captured at exit?\n")

    # Capture ratio = final R / MFE R (for winners, how much of peak is kept)
    winners = [t for t in trades if t.r_multiple > 0 and t.mfe_r > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    if winners:
        capture_ratios = [t.r_multiple / t.mfe_r for t in winners]
        cap = np.array(capture_ratios)
        lines.append(f"  Winners (n={len(winners)}):")
        lines.append(f"    MFE capture ratio (exitR / mfeR):")
        lines.append(f"      Mean:   {np.mean(cap):.1%}")
        lines.append(f"      Median: {np.median(cap):.1%}")
        lines.append(f"      Min:    {np.min(cap):.1%}  (most MFE given back)")
        lines.append(f"      Max:    {np.max(cap):.1%}  (most MFE kept)")

        # How many capture >= 50% of MFE?
        for thresh in [0.25, 0.50, 0.75]:
            count = sum(1 for c in cap if c >= thresh)
            lines.append(f"      >= {thresh:.0%}: {count} ({100 * count / len(cap):.0f}%)")

        # Avg MFE and avg captured
        avg_mfe = np.mean([t.mfe_r for t in winners])
        avg_exit_r = np.mean([t.r_multiple for t in winners])
        avg_left = avg_mfe - avg_exit_r
        lines.append(f"    Avg MFE: {avg_mfe:+.3f}R → Avg exit: {avg_exit_r:+.3f}R "
                     f"→ Left on table: {avg_left:.3f}R")

    if losers:
        lines.append(f"\n  Losers (n={len(losers)}):")
        mfe_losers = np.array([t.mfe_r for t in losers])
        lines.append(f"    Avg MFE before reversal: {np.mean(mfe_losers):+.3f}R")
        lines.append(f"    Median MFE before reversal: {np.median(mfe_losers):+.3f}R")
        # Losers that reached +1R before losing
        was_positive = sum(1 for t in losers if t.mfe_r >= 1.0)
        lines.append(f"    Reached +1R before losing: {was_positive} ({100 * was_positive / len(losers):.0f}%)")
        was_any_positive = sum(1 for t in losers if t.mfe_r > 0.1)
        lines.append(f"    Reached +0.1R before losing: {was_any_positive} ({100 * was_any_positive / len(losers):.0f}%)")

    # MFE distribution overall
    lines.append(f"\n  MFE distribution (all trades):")
    mfe_all = np.array([t.mfe_r for t in trades])
    for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
        count = sum(1 for m in mfe_all if m >= thresh)
        lines.append(f"    MFE >= {thresh:+.1f}R: {count} ({100 * count / len(trades):.0f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 24. Hold duration vs outcome
# ---------------------------------------------------------------------------

def helix_hold_duration_analysis(trades: list) -> str:
    """Analyze relationship between hold time and trade outcome."""
    if not trades:
        return "=== Hold Duration Analysis ===\n  No trades."

    lines = ["=== Hold Duration Analysis ==="]

    # Bucket by hold duration
    buckets = [
        ("1-3 bars", 1, 3),
        ("4-6 bars", 4, 6),
        ("7-10 bars", 7, 10),
        ("11-15 bars", 11, 15),
        ("16-20 bars", 16, 20),
        ("21+ bars", 21, 999),
    ]

    header = f"  {'Duration':12s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'AvgMFE':>7s} {'AvgMAE':>7s} {'P&L':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, lo, hi in buckets:
        ct = [t for t in trades if lo <= t.bars_held_1h <= hi]
        if not ct:
            continue
        n = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        avg_mfe = np.mean([t.mfe_r for t in ct])
        avg_mae = np.mean([t.mae_r for t in ct])
        pnl = sum(t.pnl_dollars for t in ct)
        lines.append(
            f"  {label:12s} {n:5d} {wr:5.0f}% {avg_r:+7.3f} {avg_mfe:+7.3f} {avg_mae:+7.3f} {pnl:+10,.0f}"
        )

    # Optimal hold range: which bucket has best avgR?
    best_label = ""
    best_r = -999
    for label, lo, hi in buckets:
        ct = [t for t in trades if lo <= t.bars_held_1h <= hi]
        if len(ct) >= 3:
            avg_r = np.mean([t.r_multiple for t in ct])
            if avg_r > best_r:
                best_r = avg_r
                best_label = label
    if best_label:
        lines.append(f"\n  Best hold range: {best_label} (avgR={best_r:+.3f})")

    # Correlation: bars_held vs r_multiple
    bars = np.array([t.bars_held_1h for t in trades], dtype=float)
    rs = np.array([t.r_multiple for t in trades])
    if len(bars) > 3:
        corr = np.corrcoef(bars, rs)[0, 1]
        lines.append(f"  Hold-time ↔ R correlation: {corr:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 25. Partial exit effectiveness
# ---------------------------------------------------------------------------

def helix_partial_exit_analysis(trades: list) -> str:
    """Analyze multi-tranche partial exit performance."""
    if not trades:
        return "=== Partial Exit Analysis ===\n  No trades."

    lines = ["=== Partial Exit Analysis ==="]

    n = len(trades)
    p1_done = [t for t in trades if t.partial_done]
    p2_done = [t for t in trades if getattr(t, "partial2_done", False)]
    neither = [t for t in trades if not t.partial_done]

    lines.append(f"  Total trades: {n}")
    lines.append(f"  Partial1 triggered: {len(p1_done)} ({100 * len(p1_done) / n:.0f}%)")
    lines.append(f"  Partial2 triggered: {len(p2_done)} ({100 * len(p2_done) / n:.0f}%)")
    lines.append(f"  No partial:         {len(neither)} ({100 * len(neither) / n:.0f}%)")

    # Performance by partial milestone reached
    groups = [
        ("No partial", neither),
        ("Partial1 only", [t for t in p1_done if not getattr(t, "partial2_done", False)]),
        ("Both partials", p2_done),
    ]

    lines.append(f"\n  Performance by partial milestone:")
    header = f"  {'Milestone':16s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'MedR':>7s} {'P&L':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, group in groups:
        if not group:
            continue
        gn = len(group)
        wr = np.mean([t.r_multiple > 0 for t in group]) * 100
        avg_r = np.mean([t.r_multiple for t in group])
        med_r = np.median([t.r_multiple for t in group])
        pnl = sum(t.pnl_dollars for t in group)
        lines.append(
            f"  {label:16s} {gn:5d} {wr:5.0f}% {avg_r:+7.3f} {med_r:+7.3f} {pnl:+10,.0f}"
        )

    # Runner analysis: for trades with partials, how many contracts exited as runner?
    runner_trades = [t for t in p1_done
                     if getattr(t, "entry_contracts", 0) > 0 and getattr(t, "exit_contracts", 0) > 0]
    if runner_trades:
        lines.append(f"\n  Runner analysis (partial trades with exit qty data):")
        for t in runner_trades:
            runner_frac = t.exit_contracts / t.entry_contracts
            lines.append(
                f"    {t.setup_class} {t.entry_time.strftime('%Y-%m-%d') if t.entry_time else '?'}: "
                f"entry={t.entry_contracts} exit_runner={t.exit_contracts} "
                f"({runner_frac:.0%}) R={t.r_multiple:+.3f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 26. Setup detection density and regime gaps
# ---------------------------------------------------------------------------

def helix_setup_density(setup_log: list, gate_log: list | None = None) -> str:
    """When are setups detected and when are there detection droughts?"""
    if not setup_log:
        return "=== Setup Detection Density ===\n  No setup data."

    lines = ["=== Setup Detection Density ==="]

    # Setups per calendar week
    from collections import defaultdict
    import datetime as _dt

    timestamps = sorted([s.timestamp for s in setup_log])
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # Weekly detection counts
    weekly: dict[str, int] = defaultdict(int)
    weekly_placed: dict[str, int] = defaultdict(int)

    for s in setup_log:
        wk = s.timestamp.strftime("%Y-W%W")
        weekly[wk] += 1

    placed_ids = set()
    if gate_log:
        placed_ids = {g.setup_id for g in gate_log if g.decision == "placed"}
        for s in setup_log:
            if s.setup_id in placed_ids:
                wk = s.timestamp.strftime("%Y-W%W")
                weekly_placed[wk] += 1

    # Generate all weeks
    start_monday = first_ts - _dt.timedelta(days=first_ts.weekday())
    all_weeks = []
    cursor = start_monday
    while cursor <= last_ts:
        all_weeks.append(cursor.strftime("%Y-W%W"))
        cursor += _dt.timedelta(weeks=1)

    det_counts = np.array([weekly.get(w, 0) for w in all_weeks])
    lines.append(f"  Total setups detected: {len(setup_log)}")
    lines.append(f"  Weeks in sample: {len(all_weeks)}")
    lines.append(f"  Detections/week: min={det_counts.min()}, median={np.median(det_counts):.0f}, "
                 f"mean={det_counts.mean():.1f}, max={det_counts.max()}")

    zero_detect_weeks = sum(1 for c in det_counts if c == 0)
    lines.append(f"  Weeks with 0 detections: {zero_detect_weeks}")

    # Detection by regime
    lines.append(f"\n  Detection rate by vol regime:")
    vol_buckets = [(0, 20, "low"), (20, 50, "chop"), (50, 80, "normal"), (80, 95, "high"), (95, 101, "extreme")]
    for lo, hi, label in vol_buckets:
        ct = [s for s in setup_log if lo <= s.vol_pct < hi]
        placed_ct = [s for s in ct if s.setup_id in placed_ids]
        if ct:
            lines.append(f"    vol {lo:2d}-{hi:3d} ({label:7s}): {len(ct):4d} detected, "
                         f"{len(placed_ct):4d} placed ({100 * len(placed_ct) / len(ct):.0f}% pass rate)")

    # Detection by session
    lines.append(f"\n  Detection rate by session:")
    for sess in sorted(set(s.session_block for s in setup_log)):
        ct = [s for s in setup_log if s.session_block == sess]
        placed_ct = [s for s in ct if s.setup_id in placed_ids]
        lines.append(f"    {sess:18s}: {len(ct):4d} detected, {len(placed_ct):4d} placed "
                     f"({100 * len(placed_ct) / len(ct):.0f}% pass rate)")

    # Detection by class
    lines.append(f"\n  Detection rate by class:")
    for cls in sorted(set(s.setup_class for s in setup_log)):
        ct = [s for s in setup_log if s.setup_class == cls]
        placed_ct = [s for s in ct if s.setup_id in placed_ids]
        lines.append(f"    {cls}: {len(ct):4d} detected, {len(placed_ct):4d} placed "
                     f"({100 * len(placed_ct) / len(ct):.0f}% pass rate)")

    # Cross-tab: class x direction gate pass rate
    lines.append(f"\n  Gate pass rate (class x direction):")
    dir_map = {1: "LONG", -1: "SHORT"}
    for cls in sorted(set(s.setup_class for s in setup_log)):
        for dval, dlabel in dir_map.items():
            ct = [s for s in setup_log if s.setup_class == cls and s.direction == dval]
            placed_ct = [s for s in ct if s.setup_id in placed_ids]
            if ct:
                lines.append(f"    {cls} {dlabel}: {len(ct)} detected → {len(placed_ct)} placed "
                             f"({100 * len(placed_ct) / len(ct):.0f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 27. Momentum stall deep dive
# ---------------------------------------------------------------------------

def helix_momentum_stall_deep_dive(trades: list) -> str:
    """Deep analysis of the momentum stall exit — the single biggest optimization mutation.

    Quantifies what stall catches: trades that never develop momentum and would
    have become losers. Compares stall exits vs other exits to measure false-positive cost.
    """
    if not trades:
        return "=== Momentum Stall Deep Dive ===\n  No trades."

    lines = ["=== Momentum Stall Deep Dive ==="]
    lines.append("  Momentum stall exits at bar 8 when R < 0.30 and MFE < 0.80R.\n")

    stall = [t for t in trades if t.exit_reason == "MOMENTUM_STALL"]
    non_stall = [t for t in trades if t.exit_reason != "MOMENTUM_STALL"]
    n = len(trades)

    lines.append(f"  Stall exits:     {len(stall):4d} ({100 * len(stall) / n:.0f}% of trades)")
    lines.append(f"  Non-stall exits: {len(non_stall):4d}")

    if stall:
        rs = np.array([t.r_multiple for t in stall])
        mfes = np.array([t.mfe_r for t in stall])
        maes = np.array([t.mae_r for t in stall])
        pnl = sum(t.pnl_dollars for t in stall)
        wr = np.mean(rs > 0) * 100

        lines.append(f"\n  Stall exit profile:")
        lines.append(f"    Avg R:   {np.mean(rs):+.3f}  (median: {np.median(rs):+.3f})")
        lines.append(f"    Avg MFE: {np.mean(mfes):+.3f}  (these never gained traction)")
        lines.append(f"    Avg MAE: {np.mean(maes):+.3f}")
        lines.append(f"    WR:      {wr:.0f}%  (low WR = stall correctly catching dead trades)")
        lines.append(f"    Net P&L: ${pnl:+,.0f}")

        # R-distribution of stall exits
        lines.append(f"\n  Stall R-distribution:")
        for lo, hi, label in [(-2.0, -1.0, "-2 to -1R"), (-1.0, -0.5, "-1 to -0.5R"),
                               (-0.5, 0.0, "-0.5 to 0R"), (0.0, 0.3, "0 to +0.3R"),
                               (0.3, 1.0, "+0.3 to +1R")]:
            ct = sum(1 for r in rs if lo <= r < hi)
            if ct:
                lines.append(f"    {label:14s}: {ct:4d} ({100 * ct / len(stall):.0f}%)")

        # Losses avoided: stall trades that were already negative (correct catches)
        correct_catches = sum(1 for r in rs if r < 0)
        false_positives = sum(1 for r in rs if r >= 0)
        lines.append(f"\n  Catch quality:")
        lines.append(f"    Correct catches (R<0): {correct_catches} ({100 * correct_catches / len(stall):.0f}%)")
        lines.append(f"    False positives (R>=0): {false_positives} ({100 * false_positives / len(stall):.0f}%)")
        if false_positives > 0:
            fp_avg_r = np.mean([t.r_multiple for t in stall if t.r_multiple >= 0])
            lines.append(f"    False positive avg R: {fp_avg_r:+.3f} (opportunity cost per false positive)")

    # Compare: what do non-stall trades look like at bar 8?
    early_non_stall = [t for t in non_stall if t.bars_held_1h >= 8]
    if early_non_stall:
        lines.append(f"\n  Trades that survived past bar 8 (n={len(early_non_stall)}):")
        en_rs = np.array([t.r_multiple for t in early_non_stall])
        lines.append(f"    Avg R: {np.mean(en_rs):+.3f}  WR: {np.mean(en_rs > 0) * 100:.0f}%")
        lines.append(f"    P&L:   ${sum(t.pnl_dollars for t in early_non_stall):+,.0f}")

    # Stall by class
    if stall:
        lines.append(f"\n  Stall exits by class:")
        for cls in sorted(set(t.setup_class for t in stall)):
            ct = [t for t in stall if t.setup_class == cls]
            avg_r = np.mean([t.r_multiple for t in ct])
            lines.append(f"    {cls}: {len(ct)} exits, avg R={avg_r:+.3f}")

    # Stall by direction
    if stall:
        lines.append(f"\n  Stall exits by direction:")
        for d_val, d_label in [(1, "LONG"), (-1, "SHORT")]:
            ct = [t for t in stall if t.direction == d_val]
            if ct:
                avg_r = np.mean([t.r_multiple for t in ct])
                lines.append(f"    {d_label}: {len(ct)} exits, avg R={avg_r:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 28. Trail effectiveness analysis
# ---------------------------------------------------------------------------

def helix_trail_effectiveness(trades: list) -> str:
    """Analyze the trailing stop system: activation rate, capture, and value-add."""
    if not trades:
        return "=== Trail Effectiveness ===\n  No trades."

    lines = ["=== Trail Effectiveness ==="]
    lines.append("  Trail activates at +0.55R (BE stop). Chandelier trail: mult = max(1.5, 3.0 - R/5).\n")

    trail_exits = [t for t in trades if t.exit_reason == "TRAILING_STOP"]
    initial_exits = [t for t in trades if t.exit_reason == "INITIAL_STOP"]
    hit_1r = [t for t in trades if t.hit_1r]
    n = len(trades)

    lines.append(f"  Trail activation (+1R milestone): {len(hit_1r)}/{n} ({100 * len(hit_1r) / n:.0f}%)")
    lines.append(f"  Trailing stop exits:  {len(trail_exits):4d} ({100 * len(trail_exits) / n:.0f}%)")
    lines.append(f"  Initial stop exits:   {len(initial_exits):4d} ({100 * len(initial_exits) / n:.0f}%)")

    if trail_exits:
        rs = np.array([t.r_multiple for t in trail_exits])
        mfes = np.array([t.mfe_r for t in trail_exits])
        capture = rs / np.where(mfes > 0, mfes, 1.0)

        lines.append(f"\n  Trailing stop exit quality:")
        lines.append(f"    Avg R at exit:   {np.mean(rs):+.3f}")
        lines.append(f"    Avg MFE:         {np.mean(mfes):+.3f}")
        lines.append(f"    Avg capture:     {np.mean(capture):.1%} of MFE retained")
        lines.append(f"    Median capture:  {np.median(capture):.1%}")
        lines.append(f"    P&L:             ${sum(t.pnl_dollars for t in trail_exits):+,.0f}")
        lines.append(f"    Avg bars held:   {np.mean([t.bars_held_1h for t in trail_exits]):.1f}")

        # R at trail exit distribution
        lines.append(f"\n  Trail exit R-distribution:")
        for lo, hi, label in [(0.0, 0.5, "0 to +0.5R"), (0.5, 1.0, "+0.5 to +1R"),
                               (1.0, 2.0, "+1 to +2R"), (2.0, 3.0, "+2 to +3R"),
                               (3.0, 100.0, "+3R+")]:
            ct = sum(1 for r in rs if lo <= r < hi)
            if ct:
                lines.append(f"    {label:14s}: {ct:4d} ({100 * ct / len(trail_exits):.0f}%)")

        # Negative trail exits (stopped at BE)
        neg_trail = sum(1 for r in rs if r <= 0)
        if neg_trail:
            lines.append(f"    At or below BE: {neg_trail} ({100 * neg_trail / len(trail_exits):.0f}%)")

    if initial_exits:
        lines.append(f"\n  Initial stop exit profile:")
        init_rs = np.array([t.r_multiple for t in initial_exits])
        lines.append(f"    Avg R: {np.mean(init_rs):+.3f}  (expected: ~-1.0R)")
        lines.append(f"    P&L:   ${sum(t.pnl_dollars for t in initial_exits):+,.0f}")

    # Value of trailing: compare winners with trail vs without
    winners = [t for t in trades if t.r_multiple > 0]
    if winners:
        w_trail = [t for t in winners if t.exit_reason == "TRAILING_STOP"]
        w_other = [t for t in winners if t.exit_reason != "TRAILING_STOP"]
        if w_trail and w_other:
            lines.append(f"\n  Trail value-add (winners only):")
            lines.append(f"    Trail exit avg R: {np.mean([t.r_multiple for t in w_trail]):+.3f} (n={len(w_trail)})")
            lines.append(f"    Other exit avg R: {np.mean([t.r_multiple for t in w_other]):+.3f} (n={len(w_other)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 29. Drawdown episode anatomy
# ---------------------------------------------------------------------------

def helix_drawdown_anatomy(trades: list) -> str:
    """Deep dive into drawdown episodes: attribution by class, session, vol regime."""
    if not trades:
        return "=== Drawdown Episode Anatomy ===\n  No trades."

    dated = sorted([t for t in trades if t.entry_time], key=lambda t: t.entry_time)
    if not dated:
        return "=== Drawdown Episode Anatomy ===\n  No dated trades."

    lines = ["=== Drawdown Episode Anatomy ==="]
    lines.append("  Identifies worst drawdown episodes and attributes them to specific contexts.\n")

    # Build cumulative PnL series
    cum_pnl = np.cumsum([t.pnl_dollars for t in dated])
    peak_pnl = np.maximum.accumulate(cum_pnl)
    dd_dollars = cum_pnl - peak_pnl

    # Find top 5 drawdown episodes (defined by trough)
    episodes = []
    in_dd = False
    ep_start = 0
    for i in range(len(dd_dollars)):
        if dd_dollars[i] < 0 and not in_dd:
            in_dd = True
            ep_start = i
        elif dd_dollars[i] >= 0 and in_dd:
            in_dd = False
            trough_idx = ep_start + int(np.argmin(dd_dollars[ep_start:i]))
            episodes.append({
                "start": ep_start,
                "end": i,
                "trough": trough_idx,
                "depth": float(dd_dollars[trough_idx]),
                "trades_in_dd": i - ep_start,
            })
    # Handle open drawdown at end
    if in_dd:
        trough_idx = ep_start + int(np.argmin(dd_dollars[ep_start:]))
        episodes.append({
            "start": ep_start,
            "end": len(dd_dollars) - 1,
            "trough": trough_idx,
            "depth": float(dd_dollars[trough_idx]),
            "trades_in_dd": len(dd_dollars) - ep_start,
        })

    episodes.sort(key=lambda e: e["depth"])
    top_episodes = episodes[:5]

    if not top_episodes:
        lines.append("  No drawdown episodes found.")
        return "\n".join(lines)

    for rank, ep in enumerate(top_episodes, 1):
        ep_trades = dated[ep["start"]:ep["end"] + 1]
        start_date = ep_trades[0].entry_time.strftime("%Y-%m-%d") if ep_trades[0].entry_time else "?"
        end_date = ep_trades[-1].exit_time.strftime("%Y-%m-%d") if ep_trades[-1].exit_time else "?"
        trough_date = dated[ep["trough"]].entry_time.strftime("%Y-%m-%d") if dated[ep["trough"]].entry_time else "?"

        lines.append(f"  Episode #{rank}: ${ep['depth']:+,.0f} ({start_date} -- {end_date})")
        lines.append(f"    Trough: {trough_date}, trades: {ep['trades_in_dd']}")

        # Attribution breakdown
        class_pnl: dict[str, float] = defaultdict(float)
        dir_pnl: dict[str, float] = defaultdict(float)
        session_pnl: dict[str, float] = defaultdict(float)
        exit_pnl: dict[str, float] = defaultdict(float)
        for t in ep_trades:
            class_pnl[t.setup_class] += t.pnl_dollars
            dir_pnl["LONG" if t.direction == 1 else "SHORT"] += t.pnl_dollars
            session_pnl[t.session_at_entry] += t.pnl_dollars
            exit_pnl[t.exit_reason] += t.pnl_dollars

        # Show top contributor to loss
        worst_class = min(class_pnl.items(), key=lambda x: x[1])
        worst_dir = min(dir_pnl.items(), key=lambda x: x[1])
        worst_session = min(session_pnl.items(), key=lambda x: x[1])
        lines.append(f"    Worst class:   {worst_class[0]} (${worst_class[1]:+,.0f})")
        lines.append(f"    Worst dir:     {worst_dir[0]} (${worst_dir[1]:+,.0f})")
        lines.append(f"    Worst session: {worst_session[0]} (${worst_session[1]:+,.0f})")
        lines.append(f"    Exit reasons:  {', '.join(f'{k}={v:+.0f}' for k, v in sorted(exit_pnl.items(), key=lambda x: x[1]))}")
        lines.append("")

    # Aggregate drawdown stats
    all_dd_trades = []
    for ep in top_episodes:
        all_dd_trades.extend(dated[ep["start"]:ep["end"] + 1])

    if all_dd_trades:
        lines.append(f"  Aggregate across top {len(top_episodes)} episodes ({len(all_dd_trades)} trades):")
        vol_buckets = defaultdict(list)
        for t in all_dd_trades:
            if t.vol_pct_at_entry < 20:
                vol_buckets["low (<20)"].append(t.pnl_dollars)
            elif t.vol_pct_at_entry < 50:
                vol_buckets["chop (20-50)"].append(t.pnl_dollars)
            elif t.vol_pct_at_entry < 80:
                vol_buckets["normal (50-80)"].append(t.pnl_dollars)
            else:
                vol_buckets["high (80+)"].append(t.pnl_dollars)
        for regime, pnls in sorted(vol_buckets.items()):
            lines.append(f"    Vol {regime}: {len(pnls)} trades, ${sum(pnls):+,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 30. Winner vs loser entry profile comparison
# ---------------------------------------------------------------------------

def helix_winner_loser_profile(trades: list) -> str:
    """Compare entry conditions of winners vs losers to surface predictive features."""
    if not trades:
        return "=== Winner vs Loser Entry Profile ===\n  No trades."

    lines = ["=== Winner vs Loser Entry Profile ==="]
    lines.append("  Compares entry context to identify features predictive of outcome.\n")

    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    if not winners or not losers:
        lines.append("  Insufficient winners or losers for comparison.")
        return "\n".join(lines)

    header = f"  {'Feature':24s} {'Winners':>10s} {'Losers':>10s} {'Delta':>10s} {'Signal':>8s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    # Alignment score
    w_align = np.mean([t.alignment_at_entry for t in winners])
    l_align = np.mean([t.alignment_at_entry for t in losers])
    sig = "STRONG" if abs(w_align - l_align) > 0.3 else "weak"
    lines.append(f"  {'Alignment score':24s} {w_align:10.2f} {l_align:10.2f} {w_align - l_align:+10.2f} {sig:>8s}")

    # Vol percentile
    w_vol = np.mean([t.vol_pct_at_entry for t in winners])
    l_vol = np.mean([t.vol_pct_at_entry for t in losers])
    sig = "STRONG" if abs(w_vol - l_vol) > 10 else "weak"
    lines.append(f"  {'Vol percentile':24s} {w_vol:10.1f} {l_vol:10.1f} {w_vol - l_vol:+10.1f} {sig:>8s}")

    # Session distribution
    for sess in sorted(set(t.session_at_entry for t in trades)):
        w_pct = 100 * sum(1 for t in winners if t.session_at_entry == sess) / len(winners)
        l_pct = 100 * sum(1 for t in losers if t.session_at_entry == sess) / len(losers)
        sig = "STRONG" if abs(w_pct - l_pct) > 10 else "weak"
        lines.append(f"  {'  ' + sess:24s} {w_pct:9.0f}% {l_pct:9.0f}% {w_pct - l_pct:+9.0f}% {sig:>8s}")

    # Class distribution
    for cls in sorted(set(t.setup_class for t in trades)):
        w_pct = 100 * sum(1 for t in winners if t.setup_class == cls) / len(winners)
        l_pct = 100 * sum(1 for t in losers if t.setup_class == cls) / len(losers)
        sig = "STRONG" if abs(w_pct - l_pct) > 10 else "weak"
        lines.append(f"  {'  Class ' + cls:24s} {w_pct:9.0f}% {l_pct:9.0f}% {w_pct - l_pct:+9.0f}% {sig:>8s}")

    # Direction
    w_long_pct = 100 * sum(1 for t in winners if t.direction == 1) / len(winners)
    l_long_pct = 100 * sum(1 for t in losers if t.direction == 1) / len(losers)
    sig = "STRONG" if abs(w_long_pct - l_long_pct) > 10 else "weak"
    lines.append(f"  {'  Long %':24s} {w_long_pct:9.0f}% {l_long_pct:9.0f}% {w_long_pct - l_long_pct:+9.0f}% {sig:>8s}")

    # Bars held
    w_bars = np.mean([t.bars_held_1h for t in winners])
    l_bars = np.mean([t.bars_held_1h for t in losers])
    sig = "STRONG" if abs(w_bars - l_bars) > 3 else "weak"
    lines.append(f"  {'Bars held (avg)':24s} {w_bars:10.1f} {l_bars:10.1f} {w_bars - l_bars:+10.1f} {sig:>8s}")

    # Teleport rate
    w_tele = 100 * sum(1 for t in winners if t.teleport) / len(winners)
    l_tele = 100 * sum(1 for t in losers if t.teleport) / len(losers)
    sig = "STRONG" if abs(w_tele - l_tele) > 5 else "weak"
    lines.append(f"  {'Teleport %':24s} {w_tele:9.0f}% {l_tele:9.0f}% {w_tele - l_tele:+9.0f}% {sig:>8s}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 31. R-per-bar efficiency
# ---------------------------------------------------------------------------

def helix_r_per_bar_efficiency(trades: list) -> str:
    """Capital efficiency metric: R earned per bar held."""
    if not trades:
        return "=== R-per-Bar Efficiency ===\n  No trades."

    lines = ["=== R-per-Bar Efficiency ==="]
    lines.append("  Measures capital efficiency: how much R is generated per unit of time.\n")

    valid = [t for t in trades if t.bars_held_1h > 0]
    if not valid:
        lines.append("  No trades with positive hold time.")
        return "\n".join(lines)

    rpb = np.array([t.r_multiple / t.bars_held_1h for t in valid])
    lines.append(f"  Overall R/bar: {np.mean(rpb):+.4f}  (median: {np.median(rpb):+.4f})")

    # By hold duration bucket
    buckets = [
        ("1-4 bars (fast)", 1, 4),
        ("5-8 bars (medium)", 5, 8),
        ("9-14 bars (developing)", 9, 14),
        ("15-20 bars (mature)", 15, 20),
        ("21+ bars (extended)", 21, 999),
    ]

    header = f"  {'Duration':28s} {'N':>5s} {'AvgR':>7s} {'R/bar':>8s} {'TotalR':>8s} {'P&L':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, lo, hi in buckets:
        ct = [t for t in valid if lo <= t.bars_held_1h <= hi]
        if not ct:
            continue
        n = len(ct)
        avg_r = np.mean([t.r_multiple for t in ct])
        total_r = sum(t.r_multiple for t in ct)
        r_per_bar = np.mean([t.r_multiple / t.bars_held_1h for t in ct])
        pnl = sum(t.pnl_dollars for t in ct)
        lines.append(
            f"  {label:28s} {n:5d} {avg_r:+7.3f} {r_per_bar:+8.4f} {total_r:+8.1f} {pnl:+10,.0f}"
        )

    # Best R/bar bucket insight
    best_rpb = -999.0
    best_label = ""
    for label, lo, hi in buckets:
        ct = [t for t in valid if lo <= t.bars_held_1h <= hi]
        if len(ct) >= 3:
            val = np.mean([t.r_multiple / t.bars_held_1h for t in ct])
            if val > best_rpb:
                best_rpb = val
                best_label = label
    if best_label:
        lines.append(f"\n  Most capital-efficient bucket: {best_label} (R/bar={best_rpb:+.4f})")

    # By exit reason
    lines.append(f"\n  R/bar by exit reason:")
    for reason in sorted(set(t.exit_reason for t in valid)):
        ct = [t for t in valid if t.exit_reason == reason]
        if len(ct) >= 2:
            rpb_reason = np.mean([t.r_multiple / t.bars_held_1h for t in ct])
            avg_bars = np.mean([t.bars_held_1h for t in ct])
            lines.append(f"    {reason:22s}: R/bar={rpb_reason:+.4f}  avg_bars={avg_bars:.1f}  n={len(ct)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 32. Post-partial runner analysis
# ---------------------------------------------------------------------------

def helix_post_partial_runner(trades: list) -> str:
    """Deep dive into runner performance after P1/P2 partial exits."""
    if not trades:
        return "=== Post-Partial Runner Analysis ===\n  No trades."

    lines = ["=== Post-Partial Runner Analysis ==="]
    lines.append("  P1: +1.0R/30%, P2: +1.5R/30%, Runner: 40% remaining.\n")

    p1_only = [t for t in trades if t.partial_done and not getattr(t, "partial2_done", False)]
    both = [t for t in trades if getattr(t, "partial2_done", False)]
    no_partial = [t for t in trades if not t.partial_done]
    n = len(trades)

    lines.append(f"  No partial:     {len(no_partial):4d} ({100 * len(no_partial) / n:.0f}%)")
    lines.append(f"  P1 only:        {len(p1_only):4d} ({100 * len(p1_only) / n:.0f}%)")
    lines.append(f"  P1 + P2:        {len(both):4d} ({100 * len(both) / n:.0f}%)")

    # Key question: does the runner after P1 add value?
    if p1_only:
        lines.append(f"\n  P1-only runner outcomes (n={len(p1_only)}):")
        rs = np.array([t.r_multiple for t in p1_only])
        mfes = np.array([t.mfe_r for t in p1_only])
        lines.append(f"    Avg final R:  {np.mean(rs):+.3f}")
        lines.append(f"    Avg peak MFE: {np.mean(mfes):+.3f}")
        # Runner typically exits via trail or stale after P1 at +1R
        trail_after_p1 = [t for t in p1_only if t.exit_reason == "TRAILING_STOP"]
        stale_after_p1 = [t for t in p1_only if t.exit_reason in ("STALE", "EARLY_STALE")]
        if trail_after_p1:
            lines.append(f"    Trail exits: {len(trail_after_p1)}, avg R={np.mean([t.r_multiple for t in trail_after_p1]):+.3f}")
        if stale_after_p1:
            lines.append(f"    Stale exits: {len(stale_after_p1)}, avg R={np.mean([t.r_multiple for t in stale_after_p1]):+.3f}")

    if both:
        lines.append(f"\n  P1+P2 runner outcomes (n={len(both)}):")
        rs = np.array([t.r_multiple for t in both])
        mfes = np.array([t.mfe_r for t in both])
        lines.append(f"    Avg final R:  {np.mean(rs):+.3f}")
        lines.append(f"    Avg peak MFE: {np.mean(mfes):+.3f}")
        # These reached +1.5R minimum; where do they end up?
        above_2r = sum(1 for r in rs if r >= 2.0)
        above_3r = sum(1 for r in rs if r >= 3.0)
        lines.append(f"    Finished >= +2R: {above_2r} ({100 * above_2r / len(both):.0f}%)")
        lines.append(f"    Finished >= +3R: {above_3r} ({100 * above_3r / len(both):.0f}%)")

        # MFE left on table
        left_on_table = np.array([t.mfe_r - t.r_multiple for t in both])
        lines.append(f"    Avg MFE left on table: {np.mean(left_on_table):.3f}R")

    # Partial vs no-partial P&L comparison
    partial_pnl = sum(t.pnl_dollars for t in p1_only) + sum(t.pnl_dollars for t in both)
    no_partial_pnl = sum(t.pnl_dollars for t in no_partial)
    lines.append(f"\n  P&L summary:")
    lines.append(f"    Partial trades:    ${partial_pnl:+,.0f} (n={len(p1_only) + len(both)})")
    lines.append(f"    No-partial trades: ${no_partial_pnl:+,.0f} (n={len(no_partial)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 33. Volatility regime interaction crosstab
# ---------------------------------------------------------------------------

def helix_vol_regime_crosstab(trades: list) -> str:
    """Cross-tab vol regime x class x direction to find toxic combinations."""
    if not trades:
        return "=== Vol Regime Interaction Crosstab ===\n  No trades."

    lines = ["=== Vol Regime Interaction Crosstab ==="]
    lines.append("  Finds specific vol x class x direction combinations that underperform.\n")

    vol_labels = [
        (0, 20, "low(0-20)"),
        (20, 50, "chop(20-50)"),
        (50, 80, "norm(50-80)"),
        (80, 101, "high(80+)"),
    ]
    dir_map = {1: "L", -1: "S"}

    header = f"  {'Vol Regime':14s} {'Class':5s} {'Dir':3s} {'N':>4s} {'WR':>5s} {'AvgR':>7s} {'PF':>6s} {'P&L':>9s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    toxic = []
    strong = []

    for lo, hi, vol_label in vol_labels:
        for cls in sorted(set(t.setup_class for t in trades)):
            for d_val, d_str in dir_map.items():
                ct = [t for t in trades
                      if lo <= t.vol_pct_at_entry < hi
                      and t.setup_class == cls
                      and t.direction == d_val]
                if len(ct) < 3:
                    continue
                n = len(ct)
                rs = np.array([t.r_multiple for t in ct])
                wr = np.mean(rs > 0) * 100
                avg_r = float(np.mean(rs))
                pnl = sum(t.pnl_dollars for t in ct)
                gw = float(np.sum(rs[rs > 0]))
                gl = float(np.sum(np.abs(rs[rs < 0])))
                pf = gw / gl if gl > 0 else float('inf')
                pf_str = f"{pf:6.2f}" if pf < 100 else "   inf"

                lines.append(
                    f"  {vol_label:14s} {cls:5s} {d_str:3s} {n:4d} {wr:4.0f}% {avg_r:+7.3f} {pf_str} ${pnl:+8,.0f}"
                )

                if avg_r < -0.15 and n >= 3:
                    toxic.append((vol_label, cls, d_str, n, avg_r, pnl))
                elif avg_r > 0.3 and n >= 5:
                    strong.append((vol_label, cls, d_str, n, avg_r, pnl))

    if toxic:
        lines.append(f"\n  TOXIC combinations (avg R < -0.15, n>=3):")
        for vol, cls, d, n, avg_r, pnl in sorted(toxic, key=lambda x: x[4]):
            lines.append(f"    {vol} {cls} {d}: n={n}, avgR={avg_r:+.3f}, P&L=${pnl:+,.0f}")

    if strong:
        lines.append(f"\n  STRONG combinations (avg R > +0.30, n>=5):")
        for vol, cls, d, n, avg_r, pnl in sorted(strong, key=lambda x: -x[4]):
            lines.append(f"    {vol} {cls} {d}: n={n}, avgR={avg_r:+.3f}, P&L=${pnl:+,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 34. Stop distance calibration
# ---------------------------------------------------------------------------

def helix_stop_distance_calibration(trades: list, entry_tracking: list | None = None) -> str:
    """Analyze initial stop distance vs outcome to assess stop sizing quality."""
    if not trades:
        return "=== Stop Distance Calibration ===\n  No trades."

    lines = ["=== Stop Distance Calibration ==="]
    lines.append("  Are stops well-sized? Too tight → stopped out on noise. Too wide → excess risk per trade.\n")

    # Compute stop distance in R-units (always ~1R by construction) and in points
    # Use MAE to assess if stops are reached on winning trades (near-miss stops)
    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    if winners:
        w_mae = np.array([t.mae_r for t in winners])
        lines.append(f"  Winner adverse excursion (how close to stop before recovering):")
        lines.append(f"    Avg MAE:    {np.mean(w_mae):.3f}R")
        lines.append(f"    Median MAE: {np.median(w_mae):.3f}R")
        near_stop = sum(1 for m in w_mae if m > 0.7)
        lines.append(f"    Near-stop (MAE > 0.7R): {near_stop}/{len(winners)} "
                     f"({100 * near_stop / len(winners):.0f}%) -- winners that almost hit stop")

    if losers:
        # Initial stop vs other exit reasons
        init_stop = [t for t in losers if t.exit_reason == "INITIAL_STOP"]
        other_loss = [t for t in losers if t.exit_reason != "INITIAL_STOP"]

        lines.append(f"\n  Loser classification:")
        lines.append(f"    Initial stop hits: {len(init_stop)}/{len(losers)} ({100 * len(init_stop) / len(losers):.0f}%)")
        lines.append(f"    Other loss exits:  {len(other_loss)}/{len(losers)} ({100 * len(other_loss) / len(losers):.0f}%)")

        if init_stop:
            is_mae = np.array([t.mae_r for t in init_stop])
            is_mfe = np.array([t.mfe_r for t in init_stop])
            lines.append(f"\n  Initial stop losers:")
            lines.append(f"    Avg MAE:  {np.mean(is_mae):.3f}R  (should be ~1.0R)")
            lines.append(f"    Avg MFE before stop: {np.mean(is_mfe):.3f}R")
            # Were any of these profitable at some point?
            was_positive = sum(1 for m in is_mfe if m > 0.3)
            lines.append(f"    Reached +0.3R before stopping: {was_positive}/{len(init_stop)} "
                         f"({100 * was_positive / len(init_stop):.0f}%)")

        if other_loss:
            lines.append(f"\n  Non-stop losers (stale, early adverse, momentum stall):")
            for reason in sorted(set(t.exit_reason for t in other_loss)):
                ct = [t for t in other_loss if t.exit_reason == reason]
                avg_r = np.mean([t.r_multiple for t in ct])
                avg_bars = np.mean([t.bars_held_1h for t in ct])
                lines.append(f"    {reason:22s}: n={len(ct)}, avgR={avg_r:+.3f}, avg_bars={avg_bars:.1f}")

    # Entry tracking: stop distance in points
    if entry_tracking:
        stop_dists = np.array([abs(e.entry_stop - e.stop0) for e in entry_tracking])
        filled = [e for e in entry_tracking if e.filled]
        unfilled = [e for e in entry_tracking if not e.filled]

        lines.append(f"\n  Stop distance distribution (points):")
        lines.append(f"    All entries: min={stop_dists.min():.1f}  "
                     f"median={np.median(stop_dists):.1f}  "
                     f"mean={stop_dists.mean():.1f}  max={stop_dists.max():.1f}")
        if filled:
            fd = np.array([abs(e.entry_stop - e.stop0) for e in filled])
            lines.append(f"    Filled:   median={np.median(fd):.1f}  mean={fd.mean():.1f}")
        if unfilled:
            ud = np.array([abs(e.entry_stop - e.stop0) for e in unfilled])
            lines.append(f"    Unfilled: median={np.median(ud):.1f}  mean={ud.mean():.1f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full diagnostic report
# ---------------------------------------------------------------------------

def helix_full_diagnostic(
    trades: list,
    setup_log: list | None = None,
    gate_log: list | None = None,
    entry_tracking: list | None = None,
    equity_curve: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> str:
    """Combine all diagnostic sections into one report."""
    sections = [
        helix_trade_frequency(trades),
        helix_r_distribution(trades),
        helix_r_distribution_cumulative(trades),
        helix_direction_breakdown(trades),
        helix_milestone_progression(trades),
        helix_exit_breakdown(trades),
        helix_regime_slicing(trades),
        helix_monthly_pnl(trades),
        helix_day_of_week(trades),
        helix_hourly_performance(trades),
        helix_streak_analysis(trades),
        helix_rolling_expectancy(trades),
        # Frequency & weakness diagnostics
        helix_trade_gaps(trades),
        helix_mfe_capture(trades),
        helix_hold_duration_analysis(trades),
        helix_partial_exit_analysis(trades),
        # Deep strategy diagnostics (v4.0)
        helix_momentum_stall_deep_dive(trades),
        helix_trail_effectiveness(trades),
        helix_drawdown_anatomy(trades),
        helix_winner_loser_profile(trades),
        helix_r_per_bar_efficiency(trades),
        helix_post_partial_runner(trades),
        helix_vol_regime_crosstab(trades),
        helix_stop_distance_calibration(trades, entry_tracking),
    ]
    if gate_log:
        sections.insert(2, helix_gate_blocks(gate_log))
    if setup_log and gate_log:
        sections.append(helix_setup_funnel(setup_log, gate_log, entry_tracking or []))
        sections.append(helix_gate_block_detail(setup_log, gate_log))
        sections.append(helix_gate_opportunity_cost(setup_log, gate_log))
        sections.append(helix_setup_density(setup_log, gate_log))
    if entry_tracking:
        sections.append(helix_heat_budget(entry_tracking))
        sections.append(helix_fill_rate_analysis(entry_tracking))
        sections.append(helix_entry_distance_distribution(entry_tracking))
    if equity_curve is not None:
        sections.append(helix_drawdown_profile(equity_curve, timestamps))
    return "\n\n".join(sections)
