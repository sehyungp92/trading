"""Extended diagnostic reports for Breakout v3.3-ETF strategy backtests.

All functions accept list[BreakoutTradeRecord] (duck-typed) and return str.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def _trade_net_pnl(trade) -> float:
    return float(trade.pnl_dollars) - float(getattr(trade, "commission", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# 1. Entry type drill-down
# ---------------------------------------------------------------------------

def breakout_entry_drilldown(trades: list) -> str:
    """Per-entry-type (A/B/C_standard/C_continuation/ADD) table."""
    if not trades:
        return "No trades for entry type drilldown."

    lines = ["=== Breakout Entry Type Drilldown ==="]
    header = (
        f"  {'Type':18s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} "
        f"{'MFE':>6s} {'MAE':>6s} {'Hold':>6s} {'Long':>5s} {'Short':>5s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for etype in ["A", "B", "C_standard", "C_continuation", "ADD"]:
        ct = [t for t in trades if t.entry_type == etype]
        if not ct:
            lines.append(f"  {etype:18s} {'0':>6s}")
            continue

        count = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(_trade_net_pnl(t) for t in ct)
        mfe = np.mean([t.mfe_r for t in ct])
        mae = np.mean([t.mae_r for t in ct])
        hold = np.mean([t.bars_held for t in ct])
        n_long = sum(1 for t in ct if t.direction == 1)
        n_short = count - n_long

        lines.append(
            f"  {etype:18s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} "
            f"{mfe:6.2f} {mae:6.2f} {hold:6.1f} {n_long:5d} {n_short:5d}"
        )

    count = len(trades)
    wr = np.mean([t.r_multiple > 0 for t in trades]) * 100
    avg_r = np.mean([t.r_multiple for t in trades])
    pnl = sum(_trade_net_pnl(t) for t in trades)
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(f"  {'ALL':18s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f}")

    # Flag worst entry type
    type_avg_r = {}
    for etype in ["A", "B", "C_standard", "C_continuation"]:
        ct = [t for t in trades if t.entry_type == etype]
        if ct:
            type_avg_r[etype] = np.mean([t.r_multiple for t in ct])
    if type_avg_r:
        worst = min(type_avg_r, key=type_avg_r.get)
        if type_avg_r[worst] < 0:
            lines.append(f"\n  ** Entry {worst} is the primary drag (avg R = {type_avg_r[worst]:+.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Exit tier drill-down
# ---------------------------------------------------------------------------

def breakout_exit_tier_drilldown(trades: list) -> str:
    """Per-exit-tier (ALIGNED/NEUTRAL/CAUTION) breakdown."""
    if not trades:
        return "No trades for exit tier drilldown."

    lines = ["=== Breakout Exit Tier Drilldown ==="]
    header = (
        f"  {'Tier':10s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} "
        f"{'TP1%':>6s} {'TP2%':>6s} {'Runner%':>8s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for tier in ["ALIGNED", "NEUTRAL", "CAUTION"]:
        ct = [t for t in trades if t.exit_tier == tier]
        if not ct:
            lines.append(f"  {tier:10s} {'0':>6s}")
            continue

        count = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(_trade_net_pnl(t) for t in ct)
        tp1_pct = np.mean([t.tp1_done for t in ct]) * 100
        tp2_pct = np.mean([t.tp2_done for t in ct]) * 100
        runner_pct = np.mean([t.runner_active for t in ct]) * 100

        lines.append(
            f"  {tier:10s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} "
            f"{tp1_pct:5.0f}% {tp2_pct:5.0f}% {runner_pct:7.0f}%"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Regime breakdown
# ---------------------------------------------------------------------------

def breakout_regime_breakdown(trades: list) -> str:
    """Per-regime (BULL_TREND/BEAR_TREND/RANGE_CHOP) breakdown."""
    if not trades:
        return "No trades for regime breakdown."

    lines = ["=== Breakout Regime Breakdown ==="]
    header = (
        f"  {'Regime':12s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} "
        f"{'Long':>5s} {'Short':>5s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for regime in ["BULL_TREND", "BEAR_TREND", "RANGE_CHOP"]:
        ct = [t for t in trades if t.regime_at_entry == regime]
        if not ct:
            lines.append(f"  {regime:12s} {'0':>6s}")
            continue

        count = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(_trade_net_pnl(t) for t in ct)
        n_long = sum(1 for t in ct if t.direction == 1)
        n_short = count - n_long

        lines.append(
            f"  {regime:12s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} "
            f"{n_long:5d} {n_short:5d}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Exit reason breakdown
# ---------------------------------------------------------------------------

def breakout_exit_reason_breakdown(trades: list) -> str:
    """Breakdown by exit reason."""
    if not trades:
        return "No trades for exit reason breakdown."

    lines = ["=== Breakout Exit Reason Breakdown ==="]
    header = f"  {'Reason':18s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    reasons = sorted(set(t.exit_reason for t in trades))
    for reason in reasons:
        ct = [t for t in trades if t.exit_reason == reason]
        count = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(_trade_net_pnl(t) for t in ct)
        lines.append(
            f"  {reason:18s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Partial fill analysis
# ---------------------------------------------------------------------------

def breakout_partial_fill_analysis(trades: list) -> str:
    """TP1/TP2/runner analysis."""
    if not trades:
        return "No trades for partial fill analysis."

    lines = ["=== Breakout Partial Fill Analysis ==="]

    total = len(trades)
    tp1_trades = [t for t in trades if t.tp1_done]
    tp2_trades = [t for t in trades if t.tp2_done]
    runner_trades = [t for t in trades if t.runner_active]

    lines.append(f"  Total trades: {total}")
    lines.append(f"  Reached TP1:  {len(tp1_trades):5d} ({100 * len(tp1_trades) / total:.0f}%)")
    lines.append(f"  Reached TP2:  {len(tp2_trades):5d} ({100 * len(tp2_trades) / total:.0f}%)")
    lines.append(f"  Runner active:{len(runner_trades):5d} ({100 * len(runner_trades) / total:.0f}%)")

    if tp1_trades:
        lines.append(f"\n  TP1 trades avg R:    {np.mean([t.r_multiple for t in tp1_trades]):+.3f}")
    if tp2_trades:
        lines.append(f"  TP2 trades avg R:    {np.mean([t.r_multiple for t in tp2_trades]):+.3f}")
    if runner_trades:
        lines.append(f"  Runner trades avg R: {np.mean([t.r_multiple for t in runner_trades]):+.3f}")

    # Trades that never hit TP1
    no_tp1 = [t for t in trades if not t.tp1_done]
    if no_tp1:
        lines.append(f"\n  No-TP1 trades:  {len(no_tp1):5d} avg R = {np.mean([t.r_multiple for t in no_tp1]):+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Gap-through-stop report
# ---------------------------------------------------------------------------

def breakout_gap_stop_report(trades: list) -> str:
    """Gap-through-stop event analysis."""
    gap_trades = [t for t in trades if t.gap_stop_event]
    if not gap_trades:
        return "=== Gap-Through-Stop Report ===\n  No gap stop events."

    lines = ["=== Gap-Through-Stop Report ==="]
    lines.append(f"  Count: {len(gap_trades)}")
    lines.append(f"  Avg R: {np.mean([t.r_multiple for t in gap_trades]):+.3f}")
    lines.append(f"  Total P&L: {sum(_trade_net_pnl(t) for t in gap_trades):+,.0f}")

    slippages = []
    for t in gap_trades:
        if t.gap_stop_fill_price > 0 and t.stop_price > 0:
            slip = abs(t.gap_stop_fill_price - t.stop_price)
            slippages.append(slip)
    if slippages:
        lines.append(f"  Avg gap slippage: ${np.mean(slippages):.4f}")
        lines.append(f"  Max gap slippage: ${np.max(slippages):.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Add contribution analysis
# ---------------------------------------------------------------------------

def breakout_add_analysis(trades: list) -> str:
    """Analyze add contribution to trade outcomes."""
    if not trades:
        return "No trades for add analysis."

    lines = ["=== Breakout Add Contribution ==="]
    add_trades = [t for t in trades if t.add_count > 0]
    no_add = [t for t in trades if t.add_count == 0]

    lines.append(f"  Trades with adds:    {len(add_trades)}")
    lines.append(f"  Trades without adds: {len(no_add)}")

    if add_trades:
        lines.append(f"  With-add avg R:      {np.mean([t.r_multiple for t in add_trades]):+.3f}")
    if no_add:
        lines.append(f"  Without-add avg R:   {np.mean([t.r_multiple for t in no_add]):+.3f}")

    for n_adds in [1, 2]:
        ct = [t for t in trades if t.add_count == n_adds]
        if ct:
            lines.append(f"  {n_adds}-add trades:      {len(ct):5d} avg R = {np.mean([t.r_multiple for t in ct]):+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Chop mode impact
# ---------------------------------------------------------------------------

def breakout_chop_impact(trades: list) -> str:
    """Analyze chop mode impact on trade outcomes."""
    if not trades:
        return "No trades for chop analysis."

    lines = ["=== Breakout Chop Mode Impact ==="]
    for mode in ["NORMAL", "DEGRADED", "HALT"]:
        ct = [t for t in trades if t.chop_mode_at_entry == mode]
        if not ct:
            lines.append(f"  {mode:10s}: 0 trades")
            continue
        count = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(_trade_net_pnl(t) for t in ct)
        lines.append(
            f"  {mode:10s}: {count:5d} trades  WR={wr:5.0f}%  avgR={avg_r:+.3f}  P&L={pnl:+,.0f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Stale exit analysis
# ---------------------------------------------------------------------------

def breakout_stale_analysis(trades: list) -> str:
    """Analyze stale exit trades."""
    stale = [t for t in trades if t.exit_reason == "STALE"]
    if not stale:
        return "=== Stale Exit Analysis ===\n  No stale exits."

    lines = ["=== Stale Exit Analysis ==="]
    lines.append(f"  Count: {len(stale)}")
    lines.append(f"  Avg R: {np.mean([t.r_multiple for t in stale]):+.3f}")
    lines.append(f"  Avg bars held: {np.mean([t.bars_held for t in stale]):.0f}")
    lines.append(f"  Avg days held: {np.mean([t.days_held for t in stale]):.0f}")
    lines.append(f"  Total P&L: {sum(_trade_net_pnl(t) for t in stale):+,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Streak analysis
# ---------------------------------------------------------------------------

def breakout_streak_analysis(trades: list) -> str:
    """Win/loss streak analysis."""
    if not trades:
        return "No trades for streak analysis."

    lines = ["=== Breakout Streak Analysis ==="]
    wins = [1 if t.r_multiple > 0 else 0 for t in trades]

    max_win_streak = max_loss_streak = 0
    current_win = current_loss = 0
    for w in wins:
        if w:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)

    lines.append(f"  Max win streak:  {max_win_streak}")
    lines.append(f"  Max loss streak: {max_loss_streak}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. Continuation mode analysis
# ---------------------------------------------------------------------------

def breakout_continuation_analysis(trades: list) -> str:
    """Compare continuation vs non-continuation trades."""
    if not trades:
        return "No trades for continuation analysis."

    lines = ["=== Breakout Continuation Analysis ==="]
    cont = [t for t in trades if t.continuation_at_entry]
    non_cont = [t for t in trades if not t.continuation_at_entry]

    if cont:
        lines.append(f"  Continuation:     {len(cont):5d} trades  avgR={np.mean([t.r_multiple for t in cont]):+.3f}")
    else:
        lines.append("  Continuation:     0 trades")
    if non_cont:
        lines.append(f"  Non-continuation: {len(non_cont):5d} trades  avgR={np.mean([t.r_multiple for t in non_cont]):+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. Loser classification
# ---------------------------------------------------------------------------

def breakout_loser_classification(trades: list) -> str:
    """Classify losers as 'right-then-stopped' vs 'immediately wrong'."""
    if not trades:
        return "No trades for loser classification."

    lines = ["=== Breakout Loser Classification ==="]

    losers = [t for t in trades if t.r_multiple <= 0]
    winners = [t for t in trades if t.r_multiple > 0]
    if not losers:
        lines.append("  No losing trades.")
        return "\n".join(lines)

    right_then_stopped = [t for t in losers if t.mfe_r >= 0.5]
    immediately_wrong = [t for t in losers if t.mfe_r < 0.5]

    lines.append(f"  Total losers: {len(losers)}")
    lines.append("")

    for label, cohort in [("Right-then-stopped (MFE >= 0.5R)", right_then_stopped),
                          ("Immediately wrong  (MFE <  0.5R)", immediately_wrong)]:
        if not cohort:
            lines.append(f"  {label}: 0 trades")
            continue
        n = len(cohort)
        pct = 100 * n / len(losers)
        avg_mfe = np.mean([t.mfe_r for t in cohort])
        avg_r = np.mean([t.r_multiple for t in cohort])
        avg_mae = np.mean([t.mae_r for t in cohort])
        avg_hold = np.mean([t.bars_held for t in cohort])
        lines.append(f"  {label}")
        lines.append(f"    Count: {n} ({pct:.0f}%)  avgMFE={avg_mfe:.2f}  avgR={avg_r:+.3f}  avgMAE={avg_mae:.2f}  avgHold={avg_hold:.0f}")

        # Sub-split by entry type
        etypes = sorted(set(t.entry_type for t in cohort))
        if etypes:
            parts = []
            for et in etypes:
                et_ct = sum(1 for t in cohort if t.entry_type == et)
                parts.append(f"{et}={et_ct}")
            lines.append(f"    Entry types: {', '.join(parts)}")

    # MFE capture ratio for winners
    if winners:
        captures = [t.r_multiple / t.mfe_r for t in winners if t.mfe_r > 0]
        if captures:
            lines.append(f"\n  Winner MFE capture ratio: {np.mean(captures):.2f} (R / MFE)")

    # Actionable flag
    if len(right_then_stopped) > 0.6 * len(losers):
        lines.append("\n  ** EXIT MANAGEMENT is the primary drag (>60% right-then-stopped)")
    elif len(immediately_wrong) > 0.6 * len(losers):
        lines.append("\n  ** ENTRY TIMING is the primary drag (>60% immediately wrong)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 13. Exit reason x Entry type cross-tab
# ---------------------------------------------------------------------------

def breakout_exit_entry_crosstab(trades: list) -> str:
    """Matrix of exit_reason x entry_type with count, avg R, win rate."""
    if not trades:
        return "No trades for exit-entry cross-tab."

    lines = ["=== Exit Reason x Entry Type Cross-Tab ==="]

    entry_types = sorted(set(t.entry_type for t in trades))
    exit_reasons = sorted(set(t.exit_reason for t in trades))

    # Header
    et_header = "  ".join(f"{et:>14s}" for et in entry_types)
    lines.append(f"  {'Exit Reason':18s}  {et_header}")
    lines.append("  " + "-" * (20 + 16 * len(entry_types)))

    problem_pairs = []
    sweet_spots = []

    for reason in exit_reasons:
        cells = []
        for et in entry_types:
            ct = [t for t in trades if t.exit_reason == reason and t.entry_type == et]
            if not ct:
                cells.append(f"{'--':>14s}")
                continue
            n = len(ct)
            avg_r = np.mean([t.r_multiple for t in ct])
            wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
            cells.append(f"{n:3d} {avg_r:+.2f} {wr:3.0f}%")
            if n >= 3 and avg_r < -0.5:
                problem_pairs.append((reason, et, n, avg_r))
            if n >= 3 and avg_r > 1.0:
                sweet_spots.append((reason, et, n, avg_r))
        lines.append(f"  {reason:18s}  {'  '.join(cells)}")

    if problem_pairs:
        lines.append("\n  Problem pairs (n>=3, avgR < -0.5):")
        for reason, et, n, avg_r in problem_pairs:
            lines.append(f"    {reason} x {et}: n={n}, avgR={avg_r:+.3f}")
    if sweet_spots:
        lines.append("\n  Sweet spots (n>=3, avgR > 1.0):")
        for reason, et, n, avg_r in sweet_spots:
            lines.append(f"    {reason} x {et}: n={n}, avgR={avg_r:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 14. Regime x Direction grid
# ---------------------------------------------------------------------------

def breakout_regime_direction_grid(trades: list) -> str:
    """3x2 grid: regime_at_entry x direction with alignment labels."""
    if not trades:
        return "No trades for regime-direction grid."

    lines = ["=== Regime x Direction Grid ==="]

    # Direction mapping: 1=LONG, -1=SHORT (or int direction)
    dir_labels = {1: "LONG", -1: "SHORT"}
    regimes = ["BULL_TREND", "BEAR_TREND", "RANGE_CHOP"]

    header = (
        f"  {'Regime':12s} {'Dir':6s} {'Align':8s} {'Count':>6s} {'WR':>6s} "
        f"{'AvgR':>7s} {'P&L':>10s} {'QMult':>6s} {'MFE':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    aligned_trades = []
    counter_trades = []

    for regime in regimes:
        for d_val, d_label in dir_labels.items():
            ct = [t for t in trades if t.regime_at_entry == regime and t.direction == d_val]
            if not ct:
                continue

            # Determine alignment
            if (d_label == "LONG" and regime == "BULL_TREND") or \
               (d_label == "SHORT" and regime == "BEAR_TREND"):
                align = "ALIGNED"
                aligned_trades.extend(ct)
            elif (d_label == "LONG" and regime == "BEAR_TREND") or \
                 (d_label == "SHORT" and regime == "BULL_TREND"):
                align = "COUNTER"
                counter_trades.extend(ct)
            else:
                align = "NEUTRAL"

            n = len(ct)
            wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
            avg_r = np.mean([t.r_multiple for t in ct])
            pnl = sum(_trade_net_pnl(t) for t in ct)
            avg_qm = np.mean([t.quality_mult_at_entry for t in ct])
            avg_mfe = np.mean([t.mfe_r for t in ct])

            lines.append(
                f"  {regime:12s} {d_label:6s} {align:8s} {n:6d} {wr:5.0f}% "
                f"{avg_r:+7.3f} {pnl:+10,.0f} {avg_qm:6.2f} {avg_mfe:6.2f}"
            )

    # Summary comparison
    lines.append("")
    if aligned_trades:
        avg_r_a = np.mean([t.r_multiple for t in aligned_trades])
        lines.append(f"  Aligned  trades: {len(aligned_trades):5d}  avgR={avg_r_a:+.3f}")
    if counter_trades:
        avg_r_c = np.mean([t.r_multiple for t in counter_trades])
        lines.append(f"  Counter  trades: {len(counter_trades):5d}  avgR={avg_r_c:+.3f}")
        if avg_r_c < 0:
            lines.append("  ** Counter-regime trades have negative avg R — consider tighter filtering")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 15. MFE cohort segmentation
# ---------------------------------------------------------------------------

def breakout_mfe_cohort_segmentation(trades: list) -> str:
    """Compare entry-time features between developed and undeveloped trades."""
    if not trades:
        return "No trades for MFE cohort segmentation."

    lines = ["=== MFE Cohort Segmentation ==="]

    developed = [t for t in trades if t.mfe_r >= 1.0]
    undeveloped = [t for t in trades if t.mfe_r < 1.0]

    lines.append(f"  Developed   (MFE >= 1.0R): {len(developed):5d}")
    lines.append(f"  Undeveloped (MFE <  1.0R): {len(undeveloped):5d}")

    if not developed or not undeveloped:
        lines.append("  (Need both cohorts for comparison)")
        return "\n".join(lines)

    # Per-cohort summary
    lines.append("")
    for label, cohort in [("Developed", developed), ("Undeveloped", undeveloped)]:
        n = len(cohort)
        wr = np.mean([t.r_multiple > 0 for t in cohort]) * 100
        avg_r = np.mean([t.r_multiple for t in cohort])
        avg_mfe = np.mean([t.mfe_r for t in cohort])
        avg_hold = np.mean([t.bars_held for t in cohort])
        lines.append(f"  {label:12s}: n={n}  WR={wr:.0f}%  avgR={avg_r:+.3f}  avgMFE={avg_mfe:.2f}  avgHold={avg_hold:.0f}")

    # Feature comparison
    lines.append("\n  Feature comparison (Developed vs Undeveloped):")
    header = f"    {'Feature':22s} {'Developed':>10s} {'Undeveloped':>12s} {'Delta':>8s} {'Flag':>5s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    features = [
        ("score_at_entry", "score_at_entry", False),
        ("quality_mult_at_entry", "quality_mult_at_entry", False),
        ("disp_at_entry", "disp_at_entry", False),
        ("rvol_d_at_entry", "rvol_d_at_entry", False),
        ("sq_good_at_entry", "sq_good_at_entry", True),
        ("continuation_at_entry", "continuation_at_entry", True),
    ]

    for label, attr, is_bool in features:
        if is_bool:
            dev_val = np.mean([getattr(t, attr, False) for t in developed]) * 100
            und_val = np.mean([getattr(t, attr, False) for t in undeveloped]) * 100
            delta = dev_val - und_val
            flag = " ***" if abs(delta) > 15 else ""
            lines.append(f"    {label:22s} {dev_val:9.0f}% {und_val:11.0f}% {delta:+7.0f}%{flag}")
        else:
            dev_val = np.mean([getattr(t, attr, 0) for t in developed])
            und_val = np.mean([getattr(t, attr, 0) for t in undeveloped])
            delta_pct = 100 * (dev_val - und_val) / abs(und_val) if und_val != 0 else 0
            flag = " ***" if abs(delta_pct) > 15 else ""
            lines.append(f"    {label:22s} {dev_val:10.3f} {und_val:12.3f} {delta_pct:+7.0f}%{flag}")

    # Regime distribution per cohort
    lines.append("\n  Regime distribution:")
    for label, cohort in [("Developed", developed), ("Undeveloped", undeveloped)]:
        n = len(cohort)
        regimes = Counter(t.regime_at_entry for t in cohort)
        parts = [f"{r}={100 * regimes.get(r, 0) / n:.0f}%" for r in ["BULL_TREND", "BEAR_TREND", "RANGE_CHOP"]]
        lines.append(f"    {label:12s}: {', '.join(parts)}")

    # Entry type distribution per cohort
    lines.append("\n  Entry type distribution:")
    all_etypes = sorted(set(t.entry_type for t in trades))
    for label, cohort in [("Developed", developed), ("Undeveloped", undeveloped)]:
        n = len(cohort)
        etypes = Counter(t.entry_type for t in cohort)
        parts = [f"{et}={100 * etypes.get(et, 0) / n:.0f}%" for et in all_etypes]
        lines.append(f"    {label:12s}: {', '.join(parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 16. Position occupancy
# ---------------------------------------------------------------------------

def breakout_position_occupancy(trades: list) -> str:
    """Quantify capital deployment efficiency."""
    if not trades:
        return "No trades for position occupancy."

    lines = ["=== Position Occupancy ==="]

    # Hold time stats
    holds = [t.bars_held for t in trades]
    lines.append(f"  Hold time (bars): mean={np.mean(holds):.1f}  median={np.median(holds):.0f}  "
                 f"min={np.min(holds)}  max={np.max(holds)}")

    if hasattr(trades[0], 'days_held'):
        days = [t.days_held for t in trades]
        lines.append(f"  Hold time (days): mean={np.mean(days):.1f}  median={np.median(days):.0f}  "
                     f"min={np.min(days)}  max={np.max(days)}")

    # Sort trades by entry time for gap analysis
    sorted_trades = sorted(
        [t for t in trades if t.entry_time is not None and t.exit_time is not None],
        key=lambda t: t.entry_time,
    )

    if len(sorted_trades) >= 2:
        gaps_bars = []
        gaps_td = []
        for i in range(1, len(sorted_trades)):
            prev_exit = sorted_trades[i - 1].exit_time
            curr_entry = sorted_trades[i].entry_time
            if prev_exit is not None and curr_entry is not None and curr_entry > prev_exit:
                td = curr_entry - prev_exit
                gaps_td.append(td)

        if gaps_td:
            gap_hours = [g.total_seconds() / 3600 for g in gaps_td]
            gap_days = [g.total_seconds() / 86400 for g in gaps_td]
            lines.append(f"\n  Gap between trades (hours): mean={np.mean(gap_hours):.1f}  "
                         f"median={np.median(gap_hours):.1f}  max={np.max(gap_hours):.0f}")
            lines.append(f"  Gap between trades (days):  mean={np.mean(gap_days):.1f}  "
                         f"median={np.median(gap_days):.1f}  max={np.max(gap_days):.0f}")

            # Longest dry spell
            longest_idx = int(np.argmax(gap_hours))
            dry_start = sorted_trades[longest_idx].exit_time
            dry_end = sorted_trades[longest_idx + 1].entry_time
            lines.append(f"  Longest dry spell: {dry_start} to {dry_end}")

    # Trades per month / per week
    if sorted_trades:
        first_entry = sorted_trades[0].entry_time
        last_exit = sorted_trades[-1].exit_time
        if first_entry and last_exit and last_exit > first_entry:
            span_days = (last_exit - first_entry).total_seconds() / 86400
            if span_days > 0:
                trades_per_month = len(sorted_trades) / (span_days / 30.44)
                trades_per_week = len(sorted_trades) / (span_days / 7)
                lines.append(f"\n  Trades/month: {trades_per_month:.1f}")
                lines.append(f"  Trades/week:  {trades_per_week:.2f}")

    # Overlapping bars from adds
    add_trades = [t for t in trades if t.add_count > 0]
    no_add = [t for t in trades if t.add_count == 0]
    if add_trades:
        lines.append(f"\n  Trades with adds (overlapping): {len(add_trades)} ({100*len(add_trades)/len(trades):.0f}%)")
        lines.append(f"  Single-position trades:         {len(no_add)} ({100*len(no_add)/len(trades):.0f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 17. Score component analysis (Phase 2 — uses score sub-component fields)
# ---------------------------------------------------------------------------

def breakout_score_component_analysis(trades: list) -> str:
    """Per-component predictive power of the evidence score."""
    if not trades:
        return "No trades for score component analysis."

    # Check if score sub-components are populated
    if not hasattr(trades[0], 'score_vol_at_entry'):
        return "=== Score Component Analysis ===\n  (score sub-components not captured — run with Phase 2 engine)"

    lines = ["=== Score Component Analysis ==="]

    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    components = [
        ("volume", "score_vol_at_entry"),
        ("squeeze", "score_squeeze_at_entry"),
        ("regime", "score_regime_at_entry"),
        ("consec_closes", "score_consec_at_entry"),
        ("atr_expanding", "score_atr_at_entry"),
    ]

    # Per-component: mean for winners vs losers
    lines.append("\n  Mean contribution by outcome:")
    header = f"    {'Component':16s} {'Winners':>8s} {'Losers':>8s} {'Delta':>8s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for label, attr in components:
        w_mean = np.mean([getattr(t, attr, 0) for t in winners]) if winners else 0
        l_mean = np.mean([getattr(t, attr, 0) for t in losers]) if losers else 0
        delta = w_mean - l_mean
        lines.append(f"    {label:16s} {w_mean:+8.2f} {l_mean:+8.2f} {delta:+8.2f}")

    # Frequency of +1, 0, -1 per component
    lines.append("\n  Value frequency distribution:")
    header2 = f"    {'Component':16s} {'  +1':>6s} {'   0':>6s} {'  -1':>6s}"
    lines.append(header2)
    lines.append("    " + "-" * (len(header2) - 4))

    for label, attr in components:
        vals = [getattr(t, attr, 0) for t in trades]
        n_pos = sum(1 for v in vals if v > 0)
        n_zero = sum(1 for v in vals if v == 0)
        n_neg = sum(1 for v in vals if v < 0)
        total = len(vals)
        lines.append(
            f"    {label:16s} {100*n_pos/total:5.0f}% {100*n_zero/total:5.0f}% {100*n_neg/total:5.0f}%"
        )

    # Avg R when component == +1 vs -1
    lines.append("\n  Avg R by component value:")
    header3 = f"    {'Component':16s} {'R|+1':>8s} {'n|+1':>6s} {'R|-1':>8s} {'n|-1':>6s} {'Pred?':>6s}"
    lines.append(header3)
    lines.append("    " + "-" * (len(header3) - 4))

    for label, attr in components:
        pos_trades = [t for t in trades if getattr(t, attr, 0) > 0]
        neg_trades = [t for t in trades if getattr(t, attr, 0) < 0]
        r_pos = np.mean([t.r_multiple for t in pos_trades]) if pos_trades else 0
        r_neg = np.mean([t.r_multiple for t in neg_trades]) if neg_trades else 0
        predictive = "YES" if (r_pos - r_neg) > 0.1 and pos_trades and neg_trades else "no"
        if not pos_trades and not neg_trades:
            predictive = "n/a"
        lines.append(
            f"    {label:16s} {r_pos:+8.3f} {len(pos_trades):6d} "
            f"{r_neg:+8.3f} {len(neg_trades):6d} {predictive:>6s}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 18. Quality mult decomposition (Phase 2)
# ---------------------------------------------------------------------------

def breakout_quality_mult_decomposition(trades: list) -> str:
    """Decompose quality_mult into sub-components and identify binding constraint."""
    if not trades:
        return "No trades for quality mult decomposition."

    if not hasattr(trades[0], 'regime_mult_at_entry'):
        return "=== Quality Mult Decomposition ===\n  (sub-components not captured — run with Phase 2 engine)"

    lines = ["=== Quality Mult Decomposition ==="]

    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    sub_components = [
        ("regime_mult", "regime_mult_at_entry"),
        ("disp_mult", "disp_mult_at_entry"),
        ("squeeze_mult", "squeeze_mult_at_entry"),
        ("corr_mult", "corr_mult_at_entry"),
    ]

    # Stats per sub-component
    lines.append("\n  Sub-component stats (all trades):")
    header = f"    {'Component':14s} {'Mean':>7s} {'Median':>7s} {'Min':>7s} {'Max':>7s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    means_all = {}
    for label, attr in sub_components:
        vals = [getattr(t, attr, 1.0) for t in trades]
        m = np.mean(vals)
        means_all[label] = m
        lines.append(
            f"    {label:14s} {m:7.3f} {np.median(vals):7.3f} {np.min(vals):7.3f} {np.max(vals):7.3f}"
        )

    # Binding constraint
    binding = min(means_all, key=means_all.get)
    lines.append(f"\n  Binding constraint (lowest mean): {binding} = {means_all[binding]:.3f}")

    # Winners vs losers
    if winners and losers:
        lines.append("\n  Winners vs Losers:")
        header2 = f"    {'Component':14s} {'Winners':>8s} {'Losers':>8s} {'Delta':>8s}"
        lines.append(header2)
        lines.append("    " + "-" * (len(header2) - 4))
        for label, attr in sub_components:
            w_mean = np.mean([getattr(t, attr, 1.0) for t in winners])
            l_mean = np.mean([getattr(t, attr, 1.0) for t in losers])
            lines.append(f"    {label:14s} {w_mean:8.3f} {l_mean:8.3f} {w_mean - l_mean:+8.3f}")

    # Cross-tab: entry_type x avg of each sub-component
    entry_types = sorted(set(t.entry_type for t in trades))
    if entry_types:
        lines.append("\n  By entry type:")
        et_header = "    " + f"{'Entry':10s}" + "".join(f" {label:>12s}" for label, _ in sub_components)
        lines.append(et_header)
        lines.append("    " + "-" * (len(et_header) - 4))
        for et in entry_types:
            ct = [t for t in trades if t.entry_type == et]
            if not ct:
                continue
            vals_str = "".join(
                f" {np.mean([getattr(t, attr, 1.0) for t in ct]):12.3f}"
                for _, attr in sub_components
            )
            lines.append(f"    {et:10s}{vals_str}")

    # Developed vs undeveloped cohort
    developed = [t for t in trades if t.mfe_r >= 1.0]
    undeveloped = [t for t in trades if t.mfe_r < 1.0]
    if developed and undeveloped:
        lines.append("\n  Developed vs Undeveloped:")
        header3 = f"    {'Component':14s} {'Developed':>10s} {'Undeveloped':>12s}"
        lines.append(header3)
        lines.append("    " + "-" * (len(header3) - 4))
        for label, attr in sub_components:
            d_mean = np.mean([getattr(t, attr, 1.0) for t in developed])
            u_mean = np.mean([getattr(t, attr, 1.0) for t in undeveloped])
            lines.append(f"    {label:14s} {d_mean:10.3f} {u_mean:12.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 19. Partial fill timing analysis (Phase 2)
# ---------------------------------------------------------------------------

def breakout_partial_timing_analysis(trades: list) -> str:
    """Timing analysis for TP1, TP2, and BE stop move."""
    if not trades:
        return "No trades for partial timing analysis."

    if not hasattr(trades[0], 'tp1_bar'):
        return "=== Partial Fill Timing ===\n  (timing fields not captured — run with Phase 2 engine)"

    lines = ["=== Partial Fill Timing ==="]

    tp1_trades = [t for t in trades if t.tp1_bar > 0]
    tp2_trades = [t for t in trades if t.tp2_bar > 0]
    be_trades = [t for t in trades if t.be_bar > 0]

    if tp1_trades:
        tp1_bars = [t.tp1_bar for t in tp1_trades]
        lines.append(f"  TP1 reached: {len(tp1_trades)} trades")
        lines.append(f"    Bars to TP1: mean={np.mean(tp1_bars):.1f}  median={np.median(tp1_bars):.0f}  "
                     f"min={np.min(tp1_bars)}  max={np.max(tp1_bars)}")
    else:
        lines.append("  TP1 reached: 0 trades")

    if tp2_trades:
        tp2_bars = [t.tp2_bar for t in tp2_trades]
        lines.append(f"\n  TP2 reached: {len(tp2_trades)} trades")
        lines.append(f"    Bars to TP2: mean={np.mean(tp2_bars):.1f}  median={np.median(tp2_bars):.0f}  "
                     f"min={np.min(tp2_bars)}  max={np.max(tp2_bars)}")

        # Time gap TP1 → TP2
        both = [t for t in trades if t.tp1_bar > 0 and t.tp2_bar > 0 and t.tp2_bar > t.tp1_bar]
        if both:
            gaps = [t.tp2_bar - t.tp1_bar for t in both]
            lines.append(f"    TP1→TP2 gap: mean={np.mean(gaps):.1f}  median={np.median(gaps):.0f}")
    else:
        lines.append("\n  TP2 reached: 0 trades")

    if be_trades:
        be_bars = [t.be_bar for t in be_trades]
        lines.append(f"\n  BE stop moved: {len(be_trades)} trades")
        lines.append(f"    Bars to BE: mean={np.mean(be_bars):.1f}  median={np.median(be_bars):.0f}")
    else:
        lines.append("\n  BE stop moved: 0 trades")

    # Split by exit tier
    tiers = sorted(set(t.exit_tier for t in trades if t.exit_tier))
    if tiers and tp1_trades:
        lines.append("\n  TP1 timing by exit tier:")
        for tier in tiers:
            tier_tp1 = [t for t in tp1_trades if t.exit_tier == tier]
            if tier_tp1:
                bars = [t.tp1_bar for t in tier_tp1]
                lines.append(f"    {tier:10s}: n={len(tier_tp1):4d}  meanBars={np.mean(bars):.1f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 20. Stop evolution (Phase 2)
# ---------------------------------------------------------------------------

def breakout_stop_evolution(trades: list) -> str:
    """Analyze stop tightening and final stop positioning."""
    if not trades:
        return "No trades for stop evolution."

    if not hasattr(trades[0], 'stop_tightens'):
        return "=== Stop Evolution ===\n  (stop evolution fields not captured — run with Phase 2 engine)"

    lines = ["=== Stop Evolution ==="]

    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    # Stop tighten stats
    lines.append("  Stop tighten counts:")
    for label, cohort in [("All", trades), ("Winners", winners), ("Losers", losers)]:
        if not cohort:
            continue
        tightens = [t.stop_tightens for t in cohort]
        lines.append(
            f"    {label:8s}: mean={np.mean(tightens):.1f}  median={np.median(tightens):.0f}  max={np.max(tightens)}"
        )

    # Stop movement in R-units (initial_stop - final_stop) normalized by r_price
    lines.append("\n  Stop movement (initial→final, in R-units):")
    for label, cohort in [("Winners", winners), ("Losers", losers)]:
        if not cohort:
            continue
        movements = []
        for t in cohort:
            if t.stop_price > 0 and t.final_stop_at_exit > 0:
                r_price = abs(t.entry_price - t.stop_price)
                if r_price > 0:
                    if t.direction == 1:  # LONG
                        move_r = (t.final_stop_at_exit - t.stop_price) / r_price
                    else:  # SHORT
                        move_r = (t.stop_price - t.final_stop_at_exit) / r_price
                    movements.append(move_r)
        if movements:
            lines.append(f"    {label:8s}: mean={np.mean(movements):+.2f}R  median={np.median(movements):+.2f}R")

    # Losers: hit at initial stop vs adjusted stop
    if losers:
        initial_stop_losers = [t for t in losers if t.stop_tightens == 0]
        adjusted_stop_losers = [t for t in losers if t.stop_tightens > 0]
        lines.append(f"\n  Losers stopped at initial level: {len(initial_stop_losers)}")
        lines.append(f"  Losers stopped at adjusted level: {len(adjusted_stop_losers)}")
        if initial_stop_losers:
            lines.append(f"    Initial-stop losers avgR: {np.mean([t.r_multiple for t in initial_stop_losers]):+.3f}")
        if adjusted_stop_losers:
            lines.append(f"    Adjusted-stop losers avgR: {np.mean([t.r_multiple for t in adjusted_stop_losers]):+.3f}")

    # Runners: mean tightens and final stop distance
    runners = [t for t in trades if t.runner_active]
    if runners:
        r_tightens = [t.stop_tightens for t in runners]
        lines.append(f"\n  Runners ({len(runners)} trades):")
        lines.append(f"    Mean tightens: {np.mean(r_tightens):.1f}")
        exit_distances = []
        for t in runners:
            r_price = abs(t.entry_price - t.stop_price)
            if r_price > 0 and t.final_stop_at_exit > 0:
                if t.direction == 1:
                    dist = abs(t.exit_price - t.final_stop_at_exit) / r_price
                else:
                    dist = abs(t.final_stop_at_exit - t.exit_price) / r_price
                exit_distances.append(dist)
        if exit_distances:
            lines.append(f"    Exit-to-final-stop distance: mean={np.mean(exit_distances):.2f}R")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 21. Regime transition matrix (Phase 2 — addition to existing regime breakdown)
# ---------------------------------------------------------------------------

def breakout_regime_transition(trades: list) -> str:
    """Regime at entry vs regime at exit transition matrix."""
    if not trades:
        return "No trades for regime transition."

    if not hasattr(trades[0], 'regime_at_exit'):
        return "=== Regime Transition Matrix ===\n  (regime_at_exit not captured — run with Phase 2 engine)"

    # Filter trades with both regime fields populated
    valid = [t for t in trades if t.regime_at_entry and t.regime_at_exit]
    if not valid:
        return "=== Regime Transition Matrix ===\n  No trades with regime data."

    lines = ["=== Regime Transition Matrix ==="]
    regimes = ["BULL_TREND", "BEAR_TREND", "RANGE_CHOP"]

    header = f"  {'Entry\\Exit':14s}" + "".join(f" {r:>12s}" for r in regimes)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    shifted_trades = []
    for r_entry in regimes:
        cells = []
        for r_exit in regimes:
            ct = [t for t in valid if t.regime_at_entry == r_entry and t.regime_at_exit == r_exit]
            if ct:
                avg_r = np.mean([t.r_multiple for t in ct])
                cells.append(f"{len(ct):4d} {avg_r:+.2f}")
                if r_entry != r_exit:
                    shifted_trades.extend(ct)
            else:
                cells.append(f"{'--':>12s}")
        lines.append(f"  {r_entry:14s}" + "".join(f" {c:>12s}" for c in cells))

    # Summary of regime-shifted trades
    stable = [t for t in valid if t.regime_at_entry == t.regime_at_exit]
    lines.append(f"\n  Regime stable:  {len(stable):5d} trades  avgR={np.mean([t.r_multiple for t in stable]):+.3f}" if stable else "")
    if shifted_trades:
        lines.append(f"  Regime shifted: {len(shifted_trades):5d} trades  avgR={np.mean([t.r_multiple for t in shifted_trades]):+.3f}")

        # Flag aligned→counter shifts
        bad_shifts = []
        for t in shifted_trades:
            entry_aligned = (t.direction == 1 and t.regime_at_entry == "BULL_TREND") or \
                           (t.direction == -1 and t.regime_at_entry == "BEAR_TREND")
            exit_counter = (t.direction == 1 and t.regime_at_exit == "BEAR_TREND") or \
                          (t.direction == -1 and t.regime_at_exit == "BULL_TREND")
            if entry_aligned and exit_counter:
                bad_shifts.append(t)
        if bad_shifts:
            lines.append(f"  Aligned→Counter shifts: {len(bad_shifts)} trades  "
                         f"avgR={np.mean([t.r_multiple for t in bad_shifts]):+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 22. Monthly returns heatmap
# ---------------------------------------------------------------------------

def breakout_monthly_returns(trades: list) -> str:
    """Calendar heatmap of monthly R returns -- reveals edge stability."""
    if not trades:
        return "No trades for monthly returns."

    lines = ["=== Monthly Returns (R) ==="]

    # Group trades by year-month
    monthly: dict[tuple[int, int], list] = defaultdict(list)
    for t in trades:
        if t.entry_time is not None:
            dt = t.entry_time
            if hasattr(dt, 'year'):
                monthly[(dt.year, dt.month)].append(t)

    if not monthly:
        return "=== Monthly Returns (R) ===\n  No trades with timestamps."

    years = sorted(set(y for y, m in monthly))
    months = range(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Header
    header = f"  {'Year':>6s}" + "".join(f" {mn:>7s}" for mn in month_names) + f" {'Total':>8s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    year_totals = {}
    positive_months = 0
    total_months = 0

    for year in years:
        cells = []
        year_r = 0.0
        for m in months:
            key = (year, m)
            if key in monthly:
                month_r = sum(t.r_multiple for t in monthly[key])
                n = len(monthly[key])
                year_r += month_r
                total_months += 1
                if month_r > 0:
                    positive_months += 1
                cells.append(f"{month_r:+7.2f}")
            else:
                cells.append(f"{'--':>7s}")
        year_totals[year] = year_r
        lines.append(f"  {year:6d}" + "".join(f" {c}" for c in cells) + f" {year_r:+8.2f}")

    # Summary stats
    lines.append("")
    if total_months > 0:
        lines.append(f"  Positive months: {positive_months}/{total_months} ({100*positive_months/total_months:.0f}%)")
    all_monthly_r = [sum(t.r_multiple for t in tl) for tl in monthly.values()]
    if all_monthly_r:
        lines.append(f"  Best month:  {max(all_monthly_r):+.2f}R")
        lines.append(f"  Worst month: {min(all_monthly_r):+.2f}R")
        lines.append(f"  Avg month:   {np.mean(all_monthly_r):+.2f}R")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 23. Rolling edge (trailing N-trade avg R)
# ---------------------------------------------------------------------------

def breakout_rolling_edge(trades: list) -> str:
    """Trailing 20/50 trade average R -- detects edge decay or improvement."""
    if len(trades) < 20:
        return "=== Rolling Edge ===\n  Fewer than 20 trades -- insufficient for rolling analysis."

    lines = ["=== Rolling Edge (Trailing Avg R) ==="]

    r_vals = np.array([t.r_multiple for t in trades])
    n = len(r_vals)

    for window in [20, 50]:
        if n < window:
            continue
        rolling = np.convolve(r_vals, np.ones(window) / window, mode='valid')
        lines.append(f"\n  Trailing {window}-trade avg R:")
        lines.append(f"    Current:  {rolling[-1]:+.3f}")
        lines.append(f"    Peak:     {np.max(rolling):+.3f} (trade #{np.argmax(rolling) + window})")
        lines.append(f"    Trough:   {np.min(rolling):+.3f} (trade #{np.argmin(rolling) + window})")
        lines.append(f"    Mean:     {np.mean(rolling):+.3f}")
        lines.append(f"    Std:      {np.std(rolling):.3f}")

        # Is edge currently above or below historical mean?
        mean_r = np.mean(rolling)
        std_r = np.std(rolling)
        current = rolling[-1]
        if std_r > 0:
            z = (current - mean_r) / std_r
            if z < -1.0:
                lines.append(f"    ** BELOW mean by {abs(z):.1f} sigma -- potential edge decay")
            elif z > 1.0:
                lines.append(f"    ** ABOVE mean by {z:.1f} sigma -- edge is strong")

    # First half vs second half comparison
    mid = n // 2
    first_half_r = np.mean(r_vals[:mid])
    second_half_r = np.mean(r_vals[mid:])
    lines.append(f"\n  Half-split comparison:")
    lines.append(f"    First half  (trades 1-{mid}):     avg R = {first_half_r:+.3f}")
    lines.append(f"    Second half (trades {mid+1}-{n}): avg R = {second_half_r:+.3f}")
    delta = second_half_r - first_half_r
    lines.append(f"    Delta: {delta:+.3f}")
    if delta < -0.1:
        lines.append("    ** Edge appears to be DECAYING over time")
    elif delta > 0.1:
        lines.append("    ** Edge appears to be IMPROVING over time")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 24. Cumulative R curve analysis
# ---------------------------------------------------------------------------

def breakout_r_curve(trades: list) -> str:
    """Cumulative R curve with drawdown and recovery analysis."""
    if not trades:
        return "No trades for R curve analysis."

    lines = ["=== Cumulative R Curve ==="]

    r_vals = [t.r_multiple for t in trades]
    cum_r = np.cumsum(r_vals)
    n = len(r_vals)

    lines.append(f"  Total trades: {n}")
    lines.append(f"  Final R:      {cum_r[-1]:+.2f}")
    lines.append(f"  Peak R:       {np.max(cum_r):+.2f} (trade #{np.argmax(cum_r) + 1})")

    # R drawdown (peak-to-trough in R-space)
    running_max = np.maximum.accumulate(cum_r)
    r_drawdown = cum_r - running_max
    max_r_dd = np.min(r_drawdown)
    max_r_dd_idx = np.argmin(r_drawdown)

    lines.append(f"  Max R drawdown: {max_r_dd:+.2f}R (at trade #{max_r_dd_idx + 1})")

    # Find the peak before the max drawdown
    peak_before = np.argmax(cum_r[:max_r_dd_idx + 1]) if max_r_dd_idx > 0 else 0
    lines.append(f"    Drawdown from trade #{peak_before + 1} to #{max_r_dd_idx + 1} "
                 f"({max_r_dd_idx - peak_before} trades)")

    # Recovery: did we recover from max dd?
    if max_r_dd_idx < n - 1:
        peak_r = cum_r[peak_before]
        recovered = False
        recovery_trade = None
        for i in range(max_r_dd_idx + 1, n):
            if cum_r[i] >= peak_r:
                recovered = True
                recovery_trade = i + 1
                break
        if recovered:
            lines.append(f"    Recovery at trade #{recovery_trade} "
                         f"({recovery_trade - max_r_dd_idx - 1} trades to recover)")
        else:
            lines.append(f"    ** Not yet recovered from max drawdown")

    # R per trade by quintile (shows if big wins cluster)
    quintile_size = n // 5
    if quintile_size > 0:
        lines.append(f"\n  R by chronological quintile:")
        for q in range(5):
            start = q * quintile_size
            end = start + quintile_size if q < 4 else n
            q_r = sum(r_vals[start:end])
            q_avg = np.mean(r_vals[start:end])
            lines.append(f"    Q{q+1} (trades {start+1}-{end}): {q_r:+.2f}R total, {q_avg:+.3f} avg")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 25. Crisis window analysis
# ---------------------------------------------------------------------------

def breakout_crisis_window_analysis(trades: list) -> str:
    """Performance during known market stress periods."""
    if not trades:
        return "No trades for crisis window analysis."

    from datetime import datetime as dt

    CRISIS_WINDOWS = [
        ("2022 Bear Market", dt(2022, 1, 3), dt(2022, 10, 13)),
        ("SVB Collapse", dt(2023, 3, 8), dt(2023, 3, 15)),
        ("Aug 2024 Unwind", dt(2024, 8, 1), dt(2024, 8, 5)),
        ("Tariff Shock", dt(2025, 2, 1), dt(2025, 4, 30)),
    ]

    lines = ["=== Crisis Window Analysis ==="]

    for name, start, end in CRISIS_WINDOWS:
        crisis_trades = []
        for t in trades:
            if t.entry_time is None:
                continue
            entry = t.entry_time
            if hasattr(entry, 'replace'):
                entry = entry.replace(tzinfo=None)
            if start <= entry <= end:
                crisis_trades.append(t)

        if not crisis_trades:
            lines.append(f"\n  {name} ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}): no trades")
            continue

        n = len(crisis_trades)
        total_r = sum(t.r_multiple for t in crisis_trades)
        wr = np.mean([t.r_multiple > 0 for t in crisis_trades]) * 100
        avg_r = np.mean([t.r_multiple for t in crisis_trades])
        pnl = sum(_trade_net_pnl(t) for t in crisis_trades)

        lines.append(f"\n  {name} ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}):")
        lines.append(f"    Trades: {n}  WR: {wr:.0f}%  Avg R: {avg_r:+.3f}  "
                     f"Total R: {total_r:+.2f}  PnL: ${pnl:+,.0f}")

        # Direction breakdown
        longs = [t for t in crisis_trades if t.direction == 1]
        shorts = [t for t in crisis_trades if t.direction == -1]
        if longs:
            lines.append(f"    Longs:  {len(longs)} trades, avgR={np.mean([t.r_multiple for t in longs]):+.3f}")
        if shorts:
            lines.append(f"    Shorts: {len(shorts)} trades, avgR={np.mean([t.r_multiple for t in shorts]):+.3f}")

    # Non-crisis performance for comparison
    all_crisis_trades = set()
    for name, start, end in CRISIS_WINDOWS:
        for t in trades:
            if t.entry_time is None:
                continue
            entry = t.entry_time
            if hasattr(entry, 'replace'):
                entry = entry.replace(tzinfo=None)
            if start <= entry <= end:
                all_crisis_trades.add(id(t))

    normal_trades = [t for t in trades if id(t) not in all_crisis_trades]
    if normal_trades and all_crisis_trades:
        lines.append(f"\n  Normal periods: {len(normal_trades)} trades  "
                     f"avgR={np.mean([t.r_multiple for t in normal_trades]):+.3f}")
        lines.append(f"  Crisis periods: {len(all_crisis_trades)} trades  "
                     f"avgR={np.mean([t.r_multiple for t in trades if id(t) in all_crisis_trades]):+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 26. Profit concentration
# ---------------------------------------------------------------------------

def breakout_profit_concentration(trades: list) -> str:
    """Measure dependency on outlier winners -- fragility indicator."""
    if not trades:
        return "No trades for profit concentration."

    lines = ["=== Profit Concentration ==="]

    r_vals = sorted([t.r_multiple for t in trades], reverse=True)
    total_r = sum(r_vals)
    n = len(r_vals)

    if total_r <= 0:
        lines.append("  Total R is non-positive -- no profit to concentrate.")
        return "\n".join(lines)

    # Top N% contribution
    for pct in [5, 10, 20]:
        top_n = max(1, n * pct // 100)
        top_r = sum(r_vals[:top_n])
        lines.append(f"  Top {pct:2d}% ({top_n:3d} trades): {top_r:+.2f}R = {100*top_r/total_r:.0f}% of total profit")

    # Gini coefficient of positive R trades
    winners = sorted([t.r_multiple for t in trades if t.r_multiple > 0])
    if len(winners) >= 2:
        n_w = len(winners)
        cum = np.cumsum(winners)
        gini = 1 - 2 * np.sum(cum) / (n_w * cum[-1]) + 1 / n_w
        lines.append(f"\n  Winner Gini coefficient: {gini:.3f}")
        if gini > 0.5:
            lines.append("    ** HIGH concentration -- profits depend on few large winners")
        elif gini < 0.3:
            lines.append("    ** LOW concentration -- profits well distributed")

    # Single trade dependency
    best_trade_r = max(t.r_multiple for t in trades)
    lines.append(f"\n  Best single trade: {best_trade_r:+.2f}R ({100*best_trade_r/total_r:.0f}% of total)")
    worst_trade_r = min(t.r_multiple for t in trades)
    lines.append(f"  Worst single trade: {worst_trade_r:+.2f}R")

    # What if we remove the best 3 trades?
    if n > 3:
        r_without_best3 = sum(r_vals[3:])
        lines.append(f"\n  Without best 3 trades: {r_without_best3:+.2f}R "
                     f"({100*r_without_best3/total_r:.0f}% of total)")
        if r_without_best3 <= 0:
            lines.append("    ** FRAGILE -- strategy is unprofitable without top 3 trades")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 27. Exit efficiency (MFE capture)
# ---------------------------------------------------------------------------

def breakout_exit_efficiency(trades: list) -> str:
    """Measure how much of MFE (peak favorable excursion) is captured at exit."""
    if not trades:
        return "No trades for exit efficiency."

    lines = ["=== Exit Efficiency (MFE Capture) ==="]

    # MFE capture ratio: exit R / MFE R (1.0 = perfect, 0 = gave it all back)
    trades_with_mfe = [t for t in trades if t.mfe_r > 0]
    if not trades_with_mfe:
        lines.append("  No trades with positive MFE.")
        return "\n".join(lines)

    capture_ratios = [t.r_multiple / t.mfe_r for t in trades_with_mfe]
    # Clamp to [-2, 1] for stats (avoid extreme outliers from tiny MFE)
    capture_clamped = [max(-2.0, min(1.0, c)) for c in capture_ratios]

    lines.append(f"  Trades with MFE > 0: {len(trades_with_mfe)}")
    lines.append(f"  Mean capture ratio:  {np.mean(capture_clamped):.3f}")
    lines.append(f"  Median capture:      {np.median(capture_clamped):.3f}")

    # Breakdown by outcome
    winners = [t for t in trades_with_mfe if t.r_multiple > 0]
    losers = [t for t in trades_with_mfe if t.r_multiple <= 0]

    if winners:
        w_capture = [t.r_multiple / t.mfe_r for t in winners]
        lines.append(f"\n  Winners ({len(winners)}):  mean capture = {np.mean(w_capture):.3f}")
    if losers:
        l_capture = [t.r_multiple / t.mfe_r for t in losers]
        lines.append(f"  Losers  ({len(losers)}):  mean capture = {np.mean(l_capture):.3f}")

    # By MFE bucket
    lines.append(f"\n  Capture by MFE bucket:")
    header = f"    {'MFE Range':14s} {'N':>5s} {'AvgCapture':>11s} {'AvgR':>7s} {'AvgMFE':>7s} {'WR':>5s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    buckets = [
        ("0.0 - 0.5R", 0.0, 0.5),
        ("0.5 - 1.0R", 0.5, 1.0),
        ("1.0 - 2.0R", 1.0, 2.0),
        ("2.0 - 3.0R", 2.0, 3.0),
        ("3.0R+", 3.0, float("inf")),
    ]

    for label, lo, hi in buckets:
        bucket = [t for t in trades_with_mfe if lo <= t.mfe_r < hi]
        if not bucket:
            continue
        n = len(bucket)
        caps = [max(-2.0, min(1.0, t.r_multiple / t.mfe_r)) for t in bucket]
        avg_r = np.mean([t.r_multiple for t in bucket])
        avg_mfe = np.mean([t.mfe_r for t in bucket])
        wr = 100 * np.mean([t.r_multiple > 0 for t in bucket])
        lines.append(f"    {label:14s} {n:5d} {np.mean(caps):11.3f} {avg_r:+7.3f} {avg_mfe:7.2f} {wr:4.0f}%")

    # By exit tier
    tiers = sorted(set(t.exit_tier for t in trades_with_mfe if t.exit_tier))
    if tiers:
        lines.append(f"\n  Capture by exit tier:")
        for tier in tiers:
            tier_trades = [t for t in trades_with_mfe if t.exit_tier == tier]
            if tier_trades:
                caps = [max(-2.0, min(1.0, t.r_multiple / t.mfe_r)) for t in tier_trades]
                lines.append(f"    {tier:10s}: n={len(tier_trades):4d}  capture={np.mean(caps):.3f}")

    # By entry type
    etypes = sorted(set(t.entry_type for t in trades_with_mfe))
    if etypes:
        lines.append(f"\n  Capture by entry type:")
        for et in etypes:
            et_trades = [t for t in trades_with_mfe if t.entry_type == et]
            if et_trades:
                caps = [max(-2.0, min(1.0, t.r_multiple / t.mfe_r)) for t in et_trades]
                lines.append(f"    {et:18s}: n={len(et_trades):4d}  capture={np.mean(caps):.3f}")

    # Actionable insight
    overall_capture = np.mean(capture_clamped)
    if overall_capture < 0.3:
        lines.append("\n  ** LOW capture (<0.3) -- exits are leaving significant profit on the table")
    elif overall_capture > 0.6:
        lines.append("\n  ** HIGH capture (>0.6) -- exit management is strong")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def breakout_full_diagnostic(trades: list) -> str:
    """Generate full breakout diagnostic report (27 sections)."""
    sections = [
        breakout_entry_drilldown(trades),
        breakout_exit_tier_drilldown(trades),
        breakout_regime_breakdown(trades),
        breakout_exit_reason_breakdown(trades),
        breakout_partial_fill_analysis(trades),
        breakout_gap_stop_report(trades),
        breakout_add_analysis(trades),
        breakout_chop_impact(trades),
        breakout_stale_analysis(trades),
        breakout_streak_analysis(trades),
        breakout_continuation_analysis(trades),
        breakout_loser_classification(trades),
        breakout_exit_entry_crosstab(trades),
        breakout_regime_direction_grid(trades),
        breakout_mfe_cohort_segmentation(trades),
        breakout_position_occupancy(trades),
        breakout_score_component_analysis(trades),
        breakout_quality_mult_decomposition(trades),
        breakout_partial_timing_analysis(trades),
        breakout_stop_evolution(trades),
        breakout_regime_transition(trades),
        # Advanced diagnostics (22-27)
        breakout_monthly_returns(trades),
        breakout_rolling_edge(trades),
        breakout_r_curve(trades),
        breakout_crisis_window_analysis(trades),
        breakout_profit_concentration(trades),
        breakout_exit_efficiency(trades),
    ]
    return "\n\n".join(sections)
