"""S5 (Keltner Momentum) diagnostics — Strategy-5 weakness analysis.

15-section report covering signal quality, entry conditions, risk management,
and edge stability for S5_PB and S5_DUAL strategies.

All functions accept list[S5TradeRecord] (duck-typed) and return str.
"""
from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def s5_full_diagnostic(trades: list, strategy: str = "s5") -> str:
    """Run all 15 diagnostic sections and return combined report."""
    if not trades:
        return f"No {strategy.upper()} trades to analyze."

    sections = [
        s5_overview(trades, strategy),
        s5_entry_mode_breakdown(trades),
        s5_direction_breakdown(trades),
        s5_symbol_breakdown(trades),
        s5_rsi_analysis(trades),
        s5_keltner_position(trades),
        s5_volume_filter_impact(trades),
        s5_stop_analysis(trades),
        s5_exit_reason_breakdown(trades),
        s5_mfe_mae_analysis(trades),
        s5_hold_duration(trades),
        s5_monthly_pnl(trades),
        s5_day_of_week(trades),
        s5_streak_analysis(trades),
        s5_rolling_expectancy(trades),
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _wr(trades: list) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if getattr(t, "r_multiple", 0) > 0)
    return 100.0 * wins / len(trades)


def _pf(r_arr: np.ndarray) -> float:
    gross_win = float(np.sum(r_arr[r_arr > 0])) if np.any(r_arr > 0) else 0.0
    gross_loss = float(np.abs(np.sum(r_arr[r_arr <= 0]))) if np.any(r_arr <= 0) else 0.0
    return _safe_div(gross_win, gross_loss, default=float("inf"))


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------

def s5_overview(trades: list, strategy: str = "s5") -> str:
    """Trade count, WR%, mean R, median R, PF, total PnL, avg hold, commission."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"  {strategy.upper()} OVERVIEW")
    lines.append("=" * 60)

    r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in trades])
    pnl_arr = np.array([getattr(t, "pnl_dollars", 0.0) for t in trades])
    hold_arr = np.array([getattr(t, "bars_held", 0) for t in trades])
    comm_arr = np.array([getattr(t, "commission", 0.0) for t in trades])

    n = len(trades)
    wr = _wr(trades)
    mean_r = float(np.mean(r_arr))
    med_r = float(np.median(r_arr))
    pf = _pf(r_arr)
    total_pnl = float(np.sum(pnl_arr))
    avg_hold = float(np.mean(hold_arr))
    total_comm = float(np.sum(comm_arr))

    lines.append(f"  Trades:       {n}")
    lines.append(f"  Win Rate:     {wr:.1f}%")
    lines.append(f"  Mean R:       {mean_r:+.3f}")
    lines.append(f"  Median R:     {med_r:+.3f}")
    lines.append(f"  Profit Factor:{pf:>8.2f}")
    lines.append(f"  Total PnL $:  {total_pnl:>+12,.2f}")
    lines.append(f"  Avg Hold (d): {avg_hold:.1f}")
    lines.append(f"  Total Comm:   {total_comm:>10,.2f}")

    # Best / worst trade
    best_idx = int(np.argmax(r_arr))
    worst_idx = int(np.argmin(r_arr))
    bt = trades[best_idx]
    wt = trades[worst_idx]
    lines.append(f"\n  Best trade:   {getattr(bt, 'symbol', '?')} "
                 f"{getattr(bt, 'entry_time', '?')} => {getattr(bt, 'r_multiple', 0):+.2f}R "
                 f"(${getattr(bt, 'pnl_dollars', 0):+,.2f})")
    lines.append(f"  Worst trade:  {getattr(wt, 'symbol', '?')} "
                 f"{getattr(wt, 'entry_time', '?')} => {getattr(wt, 'r_multiple', 0):+.2f}R "
                 f"(${getattr(wt, 'pnl_dollars', 0):+,.2f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Entry Mode Breakdown
# ---------------------------------------------------------------------------

def s5_entry_mode_breakdown(trades: list) -> str:
    """Per entry_mode table: count, WR%, avg R, total R, PnL, avg MFE, avg MAE."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 ENTRY MODE BREAKDOWN")
    lines.append("=" * 60)

    modes = sorted(set(getattr(t, "entry_mode", "unknown") for t in trades))
    if len(modes) <= 1:
        lines.append("  All trades share the same entry mode — skipping breakdown.")
        return "\n".join(lines)

    header = (
        f"  {'Mode':12s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s} "
        f"{'TotalR':>8s} {'PnL$':>10s} {'MFE':>6s} {'MAE':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for mode in modes:
        ct = [t for t in trades if getattr(t, "entry_mode", "unknown") == mode]
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in ct)
        mfe = np.mean([getattr(t, "mfe_r", 0.0) for t in ct])
        mae = np.mean([getattr(t, "mae_r", 0.0) for t in ct])
        lines.append(
            f"  {mode:12s} {len(ct):6d} {_wr(ct):6.1f} {float(np.mean(r_arr)):+7.3f} "
            f"{float(np.sum(r_arr)):+8.2f} {pnl:>+10,.2f} {mfe:6.2f} {mae:6.2f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Direction Breakdown
# ---------------------------------------------------------------------------

def s5_direction_breakdown(trades: list) -> str:
    """LONG vs SHORT: count, WR%, avg R, total R, total PnL, avg MFE, avg hold."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 DIRECTION BREAKDOWN")
    lines.append("=" * 60)

    header = (
        f"  {'Dir':6s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s} "
        f"{'TotalR':>8s} {'PnL$':>10s} {'MFE':>6s} {'Hold':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for dir_val, dir_label in [(1, "LONG"), (-1, "SHORT")]:
        ct = [t for t in trades if getattr(t, "direction", 0) == dir_val]
        if not ct:
            continue
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in ct)
        mfe = np.mean([getattr(t, "mfe_r", 0.0) for t in ct])
        hold = np.mean([getattr(t, "bars_held", 0) for t in ct])
        lines.append(
            f"  {dir_label:6s} {len(ct):6d} {_wr(ct):6.1f} {float(np.mean(r_arr)):+7.3f} "
            f"{float(np.sum(r_arr)):+8.2f} {pnl:>+10,.2f} {mfe:6.2f} {hold:6.1f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Symbol Breakdown
# ---------------------------------------------------------------------------

def s5_symbol_breakdown(trades: list) -> str:
    """Per-symbol table: count, WR%, avg R, total R, PnL, PF, avg hold."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 SYMBOL BREAKDOWN")
    lines.append("=" * 60)

    header = (
        f"  {'Symbol':8s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s} "
        f"{'TotalR':>8s} {'PnL$':>10s} {'PF':>6s} {'Hold':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    symbols = sorted(set(getattr(t, "symbol", "?") for t in trades))
    for sym in symbols:
        ct = [t for t in trades if getattr(t, "symbol", "?") == sym]
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in ct)
        pf = _pf(r_arr)
        hold = np.mean([getattr(t, "bars_held", 0) for t in ct])
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        lines.append(
            f"  {sym:8s} {len(ct):6d} {_wr(ct):6.1f} {float(np.mean(r_arr)):+7.3f} "
            f"{float(np.sum(r_arr)):+8.2f} {pnl:>+10,.2f} {pf_str:>6s} {hold:6.1f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. RSI Analysis
# ---------------------------------------------------------------------------

def s5_rsi_analysis(trades: list) -> str:
    """Bucket trades by rsi_at_entry and find optimal RSI zone."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 RSI AT ENTRY ANALYSIS")
    lines.append("=" * 60)

    buckets = [
        ("20-30", 20, 30),
        ("30-40", 30, 40),
        ("40-50", 40, 50),
        ("50-60", 50, 60),
        ("60-70", 60, 70),
        ("70-80", 70, 80),
    ]

    header = f"  {'RSI':8s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s} {'TotalR':>8s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    best_bucket = None
    best_avg_r = -999.0

    for label, lo, hi in buckets:
        ct = [t for t in trades
              if lo <= getattr(t, "rsi_at_entry", -1) < hi]
        if not ct:
            lines.append(f"  {label:8s} {0:6d}      -       -        -")
            continue
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        avg_r = float(np.mean(r_arr))
        lines.append(
            f"  {label:8s} {len(ct):6d} {_wr(ct):6.1f} {avg_r:+7.3f} "
            f"{float(np.sum(r_arr)):+8.2f}"
        )
        if avg_r > best_avg_r:
            best_avg_r = avg_r
            best_bucket = label

    # Trades outside 20-80 range
    outside = [t for t in trades
               if getattr(t, "rsi_at_entry", -1) < 20
               or getattr(t, "rsi_at_entry", -1) >= 80]
    if outside:
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in outside])
        lines.append(
            f"  {'<20/>80':8s} {len(outside):6d} {_wr(outside):6.1f} "
            f"{float(np.mean(r_arr)):+7.3f} {float(np.sum(r_arr)):+8.2f}"
        )

    if best_bucket:
        lines.append(f"\n  Verdict: Optimal RSI zone: {best_bucket} (avg R {best_avg_r:+.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Keltner Channel Position
# ---------------------------------------------------------------------------

def s5_keltner_position(trades: list) -> str:
    """Bucket kelt_position (0-1) into 5 bands and show performance."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 KELTNER CHANNEL POSITION")
    lines.append("=" * 60)

    buckets = [
        ("0.0-0.2 (lower)", 0.0, 0.2),
        ("0.2-0.4",         0.2, 0.4),
        ("0.4-0.6 (mid)",   0.4, 0.6),
        ("0.6-0.8",         0.6, 0.8),
        ("0.8-1.0 (upper)", 0.8, 1.01),  # 1.01 to include 1.0
    ]

    header = f"  {'Band':18s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s} {'TotalR':>8s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, lo, hi in buckets:
        ct = [t for t in trades
              if lo <= getattr(t, "kelt_position", -1) < hi]
        if not ct:
            lines.append(f"  {label:18s} {0:6d}      -       -        -")
            continue
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        lines.append(
            f"  {label:18s} {len(ct):6d} {_wr(ct):6.1f} "
            f"{float(np.mean(r_arr)):+7.3f} {float(np.sum(r_arr)):+8.2f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Volume Filter Impact
# ---------------------------------------------------------------------------

def s5_volume_filter_impact(trades: list) -> str:
    """Compare above-average volume (ratio>1) vs below-average."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 VOLUME FILTER IMPACT")
    lines.append("=" * 60)

    above = [t for t in trades if getattr(t, "volume_ratio", 0.0) > 1.0]
    below = [t for t in trades if getattr(t, "volume_ratio", 0.0) <= 1.0]

    header = f"  {'Group':14s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s} {'AvgMFE':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    above_wr, above_r, below_wr, below_r = 0.0, 0.0, 0.0, 0.0

    for label, group in [("Above Avg (>1)", above), ("Below Avg (<=1)", below)]:
        if not group:
            lines.append(f"  {label:14s} {0:6d}      -       -       -")
            continue
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in group])
        mfe_arr = np.array([getattr(t, "mfe_r", 0.0) for t in group])
        wr = _wr(group)
        avg_r = float(np.mean(r_arr))
        avg_mfe = float(np.mean(mfe_arr))
        lines.append(
            f"  {label:14s} {len(group):6d} {wr:6.1f} {avg_r:+7.3f} {avg_mfe:7.3f}"
        )
        if label.startswith("Above"):
            above_wr, above_r = wr, avg_r
        else:
            below_wr, below_r = wr, avg_r

    if above and below:
        helpful = above_wr > below_wr and above_r > below_r
        verdict = "HELPFUL" if helpful else "REVIEW"
        lines.append(f"\n  Verdict: Volume filter is {verdict}")
        lines.append(f"    Above-avg WR delta: {above_wr - below_wr:+.1f}pp, "
                     f"R delta: {above_r - below_r:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Stop Analysis
# ---------------------------------------------------------------------------

def s5_stop_analysis(trades: list) -> str:
    """ATR-based stop statistics and stop-out efficiency."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 STOP ANALYSIS")
    lines.append("=" * 60)

    atr_arr = np.array([getattr(t, "atr_at_entry", 0.0) for t in trades])
    stop_dist = np.array([
        abs(getattr(t, "entry_price", 0.0) - getattr(t, "initial_stop", 0.0))
        for t in trades
    ])
    risk_dollars = np.array([
        abs(getattr(t, "entry_price", 0.0) - getattr(t, "initial_stop", 0.0))
        * getattr(t, "qty", 0)
        for t in trades
    ])

    lines.append(f"  Avg ATR at entry:   {float(np.mean(atr_arr)):.4f}")
    lines.append(f"  Avg stop distance:  {float(np.mean(stop_dist)):.4f}")
    lines.append(f"  Avg initial risk $: {float(np.mean(risk_dollars)):,.2f}")

    stopped = [t for t in trades
               if getattr(t, "exit_reason", "").upper() in ("STOP", "STOP_LOSS", "STOPPED")]
    pct_stopped = 100.0 * len(stopped) / len(trades) if trades else 0.0
    lines.append(f"  % stopped out:      {pct_stopped:.1f}% ({len(stopped)}/{len(trades)})")

    if stopped:
        stopped_mfe = np.array([getattr(t, "mfe_r", 0.0) for t in stopped])
        stopped_r = np.array([getattr(t, "r_multiple", 0.0) for t in stopped])
        lines.append(f"  Avg MFE of stopped: {float(np.mean(stopped_mfe)):+.3f}R")
        lines.append(f"  Avg R of stopped:   {float(np.mean(stopped_r)):+.3f}R")
        high_mfe_stops = sum(1 for t in stopped if getattr(t, "mfe_r", 0.0) > 1.0)
        lines.append(f"  Stopped with MFE>1R:{high_mfe_stops:4d} "
                     f"({100.0 * high_mfe_stops / len(stopped):.1f}%) — potential trail issue")

    # MFE capture ratio for winners
    winners = [t for t in trades if getattr(t, "r_multiple", 0.0) > 0]
    if winners:
        w_r = np.mean([getattr(t, "r_multiple", 0.0) for t in winners])
        w_mfe = np.mean([getattr(t, "mfe_r", 0.0) for t in winners])
        capture = _safe_div(w_r, w_mfe)
        lines.append(f"\n  Winner MFE capture: {capture:.1%} (avg_r/avg_mfe = "
                     f"{w_r:.3f}/{w_mfe:.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Exit Reason Breakdown
# ---------------------------------------------------------------------------

def s5_exit_reason_breakdown(trades: list) -> str:
    """Per exit_reason: count, %, avg R, total R, avg MFE, avg hold."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 EXIT REASON BREAKDOWN")
    lines.append("=" * 60)

    header = (
        f"  {'Reason':16s} {'Count':>6s} {'%':>6s} {'AvgR':>7s} "
        f"{'TotalR':>8s} {'MFE':>6s} {'Hold':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    reasons = sorted(set(getattr(t, "exit_reason", "UNKNOWN") for t in trades))
    for reason in reasons:
        ct = [t for t in trades if getattr(t, "exit_reason", "UNKNOWN") == reason]
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        mfe = np.mean([getattr(t, "mfe_r", 0.0) for t in ct])
        hold = np.mean([getattr(t, "bars_held", 0) for t in ct])
        pct = 100.0 * len(ct) / len(trades)
        lines.append(
            f"  {reason:16s} {len(ct):6d} {pct:5.1f}% {float(np.mean(r_arr)):+7.3f} "
            f"{float(np.sum(r_arr)):+8.2f} {mfe:6.2f} {hold:6.1f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. MFE / MAE Analysis
# ---------------------------------------------------------------------------

def s5_mfe_mae_analysis(trades: list) -> str:
    """Winners vs losers MFE/MAE, capture efficiency, MFE cohort buckets."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 MFE / MAE ANALYSIS")
    lines.append("=" * 60)

    winners = [t for t in trades if getattr(t, "r_multiple", 0.0) > 0]
    losers = [t for t in trades if getattr(t, "r_multiple", 0.0) <= 0]

    for label, group in [("Winners", winners), ("Losers", losers)]:
        if not group:
            lines.append(f"\n  {label}: (none)")
            continue
        mfe_arr = np.array([getattr(t, "mfe_r", 0.0) for t in group])
        mae_arr = np.array([getattr(t, "mae_r", 0.0) for t in group])
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in group])
        avg_mfe = float(np.mean(mfe_arr))
        avg_mae = float(np.mean(mae_arr))
        avg_r = float(np.mean(r_arr))

        lines.append(f"\n  {label} (n={len(group)}):")
        lines.append(f"    Avg MFE-R: {avg_mfe:.3f}")
        lines.append(f"    Avg MAE-R: {avg_mae:.3f}")

        if label == "Winners" and avg_mfe != 0:
            capture = avg_r / avg_mfe
            lines.append(f"    Capture efficiency (avg_r/avg_mfe): {capture:.1%}")
        elif label == "Losers" and avg_r != 0:
            pain = _safe_div(avg_mae, abs(avg_r))
            lines.append(f"    Pain ratio (avg_mae/|avg_r|): {pain:.2f}")

    # MFE cohort buckets
    lines.append(f"\n  MFE Cohort Breakdown:")
    mfe_buckets = [
        ("0-0.5R",  0.0, 0.5),
        ("0.5-1R",  0.5, 1.0),
        ("1-2R",    1.0, 2.0),
        ("2-3R",    2.0, 3.0),
        ("3+R",     3.0, 999.0),
    ]
    header = f"    {'MFE':8s} {'Count':>6s} {'AvgRealR':>9s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for label, lo, hi in mfe_buckets:
        ct = [t for t in trades
              if lo <= getattr(t, "mfe_r", 0.0) < hi]
        if not ct:
            lines.append(f"    {label:8s} {0:6d}         -")
            continue
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        lines.append(f"    {label:8s} {len(ct):6d} {float(np.mean(r_arr)):+9.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. Hold Duration
# ---------------------------------------------------------------------------

def s5_hold_duration(trades: list) -> str:
    """Bucket bars_held (daily bars = days) and show performance per bucket."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 HOLD DURATION ANALYSIS")
    lines.append("=" * 60)

    buckets = [
        ("1-2d",   1,  3),
        ("3-5d",   3,  6),
        ("6-10d",  6, 11),
        ("11-20d", 11, 21),
        ("21+d",   21, 99999),
    ]

    header = f"  {'Duration':10s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s} {'TotalR':>8s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, lo, hi in buckets:
        ct = [t for t in trades
              if lo <= getattr(t, "bars_held", 0) < hi]
        if not ct:
            lines.append(f"  {label:10s} {0:6d}      -       -        -")
            continue
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        lines.append(
            f"  {label:10s} {len(ct):6d} {_wr(ct):6.1f} "
            f"{float(np.mean(r_arr)):+7.3f} {float(np.sum(r_arr)):+8.2f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. Monthly PnL
# ---------------------------------------------------------------------------

def s5_monthly_pnl(trades: list) -> str:
    """Group by YYYY-MM from exit_time, show monthly stats and cumulative R."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 MONTHLY PnL")
    lines.append("=" * 60)

    monthly: dict[str, list] = defaultdict(list)
    for t in trades:
        exit_t = getattr(t, "exit_time", None)
        if exit_t is None:
            continue
        key = exit_t.strftime("%Y-%m") if hasattr(exit_t, "strftime") else str(exit_t)[:7]
        monthly[key].append(t)

    header = (
        f"  {'Month':8s} {'Trades':>6s} {'WR%':>6s} {'TotalR':>8s} "
        f"{'PnL$':>10s} {'CumR':>8s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    cum_r = 0.0
    for month in sorted(monthly.keys()):
        ct = monthly[month]
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in ct)
        total_r = float(np.sum(r_arr))
        cum_r += total_r
        lines.append(
            f"  {month:8s} {len(ct):6d} {_wr(ct):6.1f} {total_r:+8.2f} "
            f"{pnl:>+10,.2f} {cum_r:+8.2f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 13. Day of Week
# ---------------------------------------------------------------------------

def s5_day_of_week(trades: list) -> str:
    """Group by entry_time weekday (Mon-Fri) and show performance."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 DAY OF WEEK ANALYSIS")
    lines.append("=" * 60)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day: dict[int, list] = defaultdict(list)

    for t in trades:
        entry_t = getattr(t, "entry_time", None)
        if entry_t is None or not hasattr(entry_t, "weekday"):
            continue
        by_day[entry_t.weekday()].append(t)

    header = f"  {'Day':5s} {'Count':>6s} {'WR%':>6s} {'AvgR':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for d in range(7):
        ct = by_day.get(d, [])
        if not ct:
            continue
        r_arr = np.array([getattr(t, "r_multiple", 0.0) for t in ct])
        lines.append(
            f"  {day_names[d]:5s} {len(ct):6d} {_wr(ct):6.1f} "
            f"{float(np.mean(r_arr)):+7.3f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 14. Streak Analysis
# ---------------------------------------------------------------------------

def s5_streak_analysis(trades: list) -> str:
    """Win/loss streak statistics and worst consecutive R drawdown."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 STREAK ANALYSIS")
    lines.append("=" * 60)

    results = [getattr(t, "r_multiple", 0.0) for t in trades]
    if not results:
        lines.append("  No trades.")
        return "\n".join(lines)

    # Build streaks
    win_streaks: list[int] = []
    loss_streaks: list[int] = []
    current_type: str | None = None
    current_len = 0

    for r in results:
        is_win = r > 0
        tag = "W" if is_win else "L"
        if tag == current_type:
            current_len += 1
        else:
            if current_type == "W" and current_len > 0:
                win_streaks.append(current_len)
            elif current_type == "L" and current_len > 0:
                loss_streaks.append(current_len)
            current_type = tag
            current_len = 1

    # Flush last streak
    if current_type == "W" and current_len > 0:
        win_streaks.append(current_len)
    elif current_type == "L" and current_len > 0:
        loss_streaks.append(current_len)

    max_ws = max(win_streaks) if win_streaks else 0
    max_ls = max(loss_streaks) if loss_streaks else 0
    avg_ws = float(np.mean(win_streaks)) if win_streaks else 0.0
    avg_ls = float(np.mean(loss_streaks)) if loss_streaks else 0.0

    lines.append(f"  Max win streak:       {max_ws}")
    lines.append(f"  Max loss streak:      {max_ls}")
    lines.append(f"  Avg win streak:       {avg_ws:.1f}")
    lines.append(f"  Avg loss streak:      {avg_ls:.1f}")

    # Worst consecutive loss in R
    worst_consec_r = 0.0
    current_loss_r = 0.0
    for r in results:
        if r <= 0:
            current_loss_r += r
            worst_consec_r = min(worst_consec_r, current_loss_r)
        else:
            current_loss_r = 0.0
    lines.append(f"  Worst consec loss R:  {worst_consec_r:+.2f}")

    # Current streak
    if results:
        last_type = "W" if results[-1] > 0 else "L"
        streak = 0
        for r in reversed(results):
            if (r > 0 and last_type == "W") or (r <= 0 and last_type == "L"):
                streak += 1
            else:
                break
        streak_label = "WIN" if last_type == "W" else "LOSS"
        lines.append(f"  Current streak:       {streak} {streak_label}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 15. Rolling Expectancy
# ---------------------------------------------------------------------------

def s5_rolling_expectancy(trades: list) -> str:
    """10-trade rolling window avg R with trend detection."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  S5 ROLLING EXPECTANCY (10-trade window)")
    lines.append("=" * 60)

    window = 10
    r_list = [getattr(t, "r_multiple", 0.0) for t in trades]

    if len(r_list) < window:
        lines.append(f"  Not enough trades ({len(r_list)}) for {window}-trade window.")
        return "\n".join(lines)

    rolling = []
    for i in range(len(r_list) - window + 1):
        chunk = r_list[i: i + window]
        rolling.append(float(np.mean(chunk)))

    rolling_arr = np.array(rolling)

    # Show last 5 windows
    n_show = min(5, len(rolling))
    lines.append(f"  Last {n_show} rolling windows:")
    for j in range(len(rolling) - n_show, len(rolling)):
        start_idx = j
        end_idx = j + window - 1
        lines.append(
            f"    Trades {start_idx + 1:>4d}-{end_idx + 1:<4d}  avg R: {rolling[j]:+.3f}"
        )

    # Linear regression slope on rolling expectancy
    x = np.arange(len(rolling), dtype=float)
    if len(rolling) >= 2:
        slope, intercept = np.polyfit(x, rolling_arr, 1)
    else:
        slope = 0.0

    lines.append(f"\n  Regression slope: {slope:+.4f}")
    lines.append(f"  Current window:   {rolling[-1]:+.3f}")
    lines.append(f"  Overall mean:     {float(np.mean(rolling_arr)):+.3f}")

    if abs(slope) < 0.01:
        verdict = "STABLE"
    elif slope > 0.01:
        verdict = "IMPROVING"
    else:
        verdict = "DEGRADING"

    lines.append(f"\n  Verdict: Edge is {verdict} (slope threshold: +/-0.01)")

    return "\n".join(lines)
