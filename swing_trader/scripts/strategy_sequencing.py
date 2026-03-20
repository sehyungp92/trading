"""Analyze temporal sequencing patterns between the 3 strategies.

Questions:
  1. When ATRSS/Breakout enters within N days of a Helix trade, is Helix better/worse?
  2. When Helix precedes/follows ATRSS, is ATRSS better/worse?
  3. Concurrent position performance vs solo performance
  4. Cluster analysis - do trades bunch together across strategies?
  5. Lead/lag: which strategy tends to enter first in a cluster?
"""
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path("data/raw")
SYMBOLS = ["QQQ", "USO", "GLD", "IBIT"]
FIXED_QTY = 10
EQUITY = 100_000.0


def _strip_tz(dt):
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def run_atrss():
    from backtest.config import BacktestConfig, SlippageConfig
    from backtest.engine.portfolio_engine import PortfolioData, run_independent
    from backtest.data.cache import load_bars
    from backtest.data.preprocessing import (
        align_daily_to_hourly, build_numpy_arrays, filter_rth, normalize_timezone,
    )
    data = PortfolioData()
    for sym in SYMBOLS:
        h_df = normalize_timezone(load_bars(DATA_DIR / f"{sym}_1h.parquet"))
        h_df = filter_rth(h_df)
        d_df = normalize_timezone(load_bars(DATA_DIR / f"{sym}_1d.parquet"))
        data.hourly[sym] = build_numpy_arrays(h_df)
        data.daily[sym] = build_numpy_arrays(d_df)
        data.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)
    slippage = SlippageConfig(commission_per_contract=1.00)
    config = BacktestConfig(symbols=SYMBOLS, initial_equity=EQUITY,
                            data_dir=DATA_DIR, slippage=slippage, fixed_qty=FIXED_QTY)
    result = run_independent(data, config)
    trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            trades.append({
                "strategy": "ATRSS", "symbol": t.symbol or sym,
                "entry": _strip_tz(t.entry_time), "exit": _strip_tz(t.exit_time),
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "R": t.r_multiple, "class": t.entry_type,
                "bars": t.bars_held, "pnl": t.pnl_dollars,
            })
    return trades


def run_helix():
    from backtest.config import SlippageConfig
    from backtest.config_helix import HelixBacktestConfig
    from backtest.engine.helix_portfolio_engine import load_helix_data, run_helix_independent
    data = load_helix_data(SYMBOLS, DATA_DIR)
    slippage = SlippageConfig(commission_per_contract=1.00)
    config = HelixBacktestConfig(symbols=SYMBOLS, initial_equity=EQUITY,
                                 data_dir=DATA_DIR, slippage=slippage, fixed_qty=FIXED_QTY)
    result = run_helix_independent(data, config)
    trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            trades.append({
                "strategy": "Helix", "symbol": t.symbol or sym,
                "entry": _strip_tz(t.entry_time), "exit": _strip_tz(t.exit_time),
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "R": t.r_multiple, "class": t.setup_class,
                "bars": t.bars_held, "pnl": t.pnl_dollars,
            })
    return trades


def run_breakout():
    from backtest.config import SlippageConfig
    from backtest.config_breakout import BreakoutBacktestConfig
    from backtest.engine.breakout_portfolio_engine import load_breakout_data, run_breakout_independent
    data = load_breakout_data(SYMBOLS, DATA_DIR)
    slippage = SlippageConfig(commission_per_contract=1.00)
    config = BreakoutBacktestConfig(symbols=SYMBOLS, initial_equity=EQUITY,
                                    data_dir=DATA_DIR, slippage=slippage, fixed_qty=FIXED_QTY)
    result = run_breakout_independent(data, config)
    trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            trades.append({
                "strategy": "Breakout", "symbol": t.symbol or sym,
                "entry": _strip_tz(t.entry_time), "exit": _strip_tz(t.exit_time),
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "R": t.r_multiple, "class": t.entry_type,
                "bars": t.bars_held, "pnl": t.pnl_dollars,
            })
    return trades


def overlaps(t1, t2):
    """Check if two trades have overlapping holding periods."""
    if not all([t1["entry"], t1["exit"], t2["entry"], t2["exit"]]):
        return False
    return t1["entry"] <= t2["exit"] and t2["entry"] <= t1["exit"]


def hours_between(dt1, dt2):
    """Signed hours from dt1 to dt2 (positive = dt2 is later)."""
    if dt1 is None or dt2 is None:
        return None
    return (dt2 - dt1).total_seconds() / 3600


def main():
    print("Running all 3 backtests...")
    print("  [1/3] ATRSS...", end=" ", flush=True)
    atrss = run_atrss()
    print(f"{len(atrss)}")
    print("  [2/3] Helix...", end=" ", flush=True)
    helix = run_helix()
    print(f"{len(helix)}")
    print("  [3/3] Breakout...", end=" ", flush=True)
    breakout = run_breakout()
    print(f"{len(breakout)}")

    all_strats = {"ATRSS": atrss, "Helix": helix, "Breakout": breakout}

    print()
    print("=" * 90)
    print("STRATEGY SEQUENCING & INTERACTION ANALYSIS")
    print("=" * 90)

    # ===================================================================
    # 1. HELIX PERFORMANCE: WITH VS WITHOUT CONCURRENT ATRSS POSITION
    # ===================================================================
    print("\n" + "=" * 90)
    print("1. HELIX PERFORMANCE: WITH vs WITHOUT CONCURRENT ATRSS POSITION")
    print("=" * 90)

    for sym in SYMBOLS + ["ALL"]:
        h_trades = [t for t in helix if (sym == "ALL" or t["symbol"] == sym)]
        a_trades = [t for t in atrss if (sym == "ALL" or t["symbol"] == sym)]

        with_atrss = []
        without_atrss = []
        with_atrss_same_dir = []
        with_atrss_opp_dir = []

        for ht in h_trades:
            concurrent = False
            same_dir = False
            for at in a_trades:
                if ht["symbol"] == at["symbol"] and overlaps(ht, at):
                    concurrent = True
                    if ht["direction"] == at["direction"]:
                        same_dir = True
            if concurrent:
                with_atrss.append(ht)
                if same_dir:
                    with_atrss_same_dir.append(ht)
                else:
                    with_atrss_opp_dir.append(ht)
            else:
                without_atrss.append(ht)

        if not with_atrss and not without_atrss:
            continue

        def _stats(trades):
            if not trades:
                return 0, 0, 0, 0
            rs = [t["R"] for t in trades]
            wr = sum(1 for r in rs if r > 0) / len(rs) * 100
            return len(trades), np.mean(rs), np.sum(rs), wr

        n_w, avg_w, tot_w, wr_w = _stats(with_atrss)
        n_wo, avg_wo, tot_wo, wr_wo = _stats(without_atrss)
        n_sd, avg_sd, tot_sd, wr_sd = _stats(with_atrss_same_dir)
        n_od, avg_od, tot_od, wr_od = _stats(with_atrss_opp_dir)

        print(f"\n  {sym}:")
        print(f"    Helix w/ concurrent ATRSS:     n={n_w:3d}  avgR={avg_w:+.3f}  "
              f"totR={tot_w:+7.1f}  WR={wr_w:.0f}%")
        print(f"    Helix w/o concurrent ATRSS:    n={n_wo:3d}  avgR={avg_wo:+.3f}  "
              f"totR={tot_wo:+7.1f}  WR={wr_wo:.0f}%")
        if n_sd:
            print(f"      same direction:              n={n_sd:3d}  avgR={avg_sd:+.3f}  "
                  f"totR={tot_sd:+7.1f}  WR={wr_sd:.0f}%")
        if n_od:
            print(f"      opposite direction:           n={n_od:3d}  avgR={avg_od:+.3f}  "
                  f"totR={tot_od:+7.1f}  WR={wr_od:.0f}%")

    # ===================================================================
    # 2. ATRSS PERFORMANCE: WITH VS WITHOUT CONCURRENT HELIX POSITION
    # ===================================================================
    print("\n" + "=" * 90)
    print("2. ATRSS PERFORMANCE: WITH vs WITHOUT CONCURRENT HELIX POSITION")
    print("=" * 90)

    for sym in SYMBOLS + ["ALL"]:
        a_trades = [t for t in atrss if (sym == "ALL" or t["symbol"] == sym)]
        h_trades = [t for t in helix if (sym == "ALL" or t["symbol"] == sym)]

        with_helix = []
        without_helix = []

        for at in a_trades:
            concurrent = any(
                at["symbol"] == ht["symbol"] and overlaps(at, ht) for ht in h_trades
            )
            if concurrent:
                with_helix.append(at)
            else:
                without_helix.append(at)

        def _stats(trades):
            if not trades:
                return 0, 0, 0, 0
            rs = [t["R"] for t in trades]
            wr = sum(1 for r in rs if r > 0) / len(rs) * 100
            return len(trades), np.mean(rs), np.sum(rs), wr

        n_w, avg_w, tot_w, wr_w = _stats(with_helix)
        n_wo, avg_wo, tot_wo, wr_wo = _stats(without_helix)

        if not n_w and not n_wo:
            continue
        print(f"\n  {sym}:")
        print(f"    ATRSS w/ concurrent Helix:     n={n_w:3d}  avgR={avg_w:+.3f}  "
              f"totR={tot_w:+7.1f}  WR={wr_w:.0f}%")
        print(f"    ATRSS w/o concurrent Helix:    n={n_wo:3d}  avgR={avg_wo:+.3f}  "
              f"totR={tot_wo:+7.1f}  WR={wr_wo:.0f}%")

    # ===================================================================
    # 3. LEAD/LAG: WHEN ATRSS ENTERS, WHAT HAPPENS TO HELIX NEARBY?
    # ===================================================================
    print("\n" + "=" * 90)
    print("3. LEAD/LAG: HELIX TRADES WITHIN N DAYS OF ATRSS ENTRY (same symbol)")
    print("=" * 90)

    windows = [1, 3, 5, 7, 14]
    print(f"\n  {'Window':>8s}  {'n_helix':>8s}  {'avgR':>7s}  {'WR':>5s}  {'totR':>8s}  "
          f"{'baseline_avgR':>13s}  {'delta':>7s}")
    print("  " + "-" * 70)

    helix_baseline_r = np.mean([t["R"] for t in helix])

    for w in windows:
        # Helix trades that started within w days AFTER an ATRSS entry (same symbol)
        nearby_helix = []
        for at in atrss:
            if at["entry"] is None:
                continue
            for ht in helix:
                if ht["entry"] is None or ht["symbol"] != at["symbol"]:
                    continue
                gap_hours = hours_between(at["entry"], ht["entry"])
                if gap_hours is not None and 0 <= gap_hours <= w * 24:
                    nearby_helix.append(ht)

        # Deduplicate (a helix trade may match multiple ATRSS entries)
        seen = set()
        unique = []
        for ht in nearby_helix:
            key = (ht["symbol"], ht["entry"])
            if key not in seen:
                seen.add(key)
                unique.append(ht)

        if unique:
            avg = np.mean([t["R"] for t in unique])
            wr = sum(1 for t in unique if t["R"] > 0) / len(unique) * 100
            tot = sum(t["R"] for t in unique)
            delta = avg - helix_baseline_r
            print(f"  {w:>5d}d    {len(unique):8d}  {avg:+7.3f}  {wr:4.0f}%  {tot:+8.1f}  "
                  f"{helix_baseline_r:+13.3f}  {delta:+7.3f}")

    # Same but BEFORE
    print(f"\n  Helix trades within N days BEFORE an ATRSS entry:")
    print(f"  {'Window':>8s}  {'n_helix':>8s}  {'avgR':>7s}  {'WR':>5s}  {'totR':>8s}  "
          f"{'baseline_avgR':>13s}  {'delta':>7s}")
    print("  " + "-" * 70)

    for w in windows:
        nearby_helix = []
        for at in atrss:
            if at["entry"] is None:
                continue
            for ht in helix:
                if ht["entry"] is None or ht["symbol"] != at["symbol"]:
                    continue
                gap_hours = hours_between(ht["entry"], at["entry"])
                if gap_hours is not None and 0 <= gap_hours <= w * 24:
                    nearby_helix.append(ht)
        seen = set()
        unique = []
        for ht in nearby_helix:
            key = (ht["symbol"], ht["entry"])
            if key not in seen:
                seen.add(key)
                unique.append(ht)
        if unique:
            avg = np.mean([t["R"] for t in unique])
            wr = sum(1 for t in unique if t["R"] > 0) / len(unique) * 100
            tot = sum(t["R"] for t in unique)
            delta = avg - helix_baseline_r
            print(f"  {w:>5d}d    {len(unique):8d}  {avg:+7.3f}  {wr:4.0f}%  {tot:+8.1f}  "
                  f"{helix_baseline_r:+13.3f}  {delta:+7.3f}")

    # ===================================================================
    # 4. LEAD/LAG: WHEN HELIX ENTERS, WHAT HAPPENS TO ATRSS NEARBY?
    # ===================================================================
    print("\n" + "=" * 90)
    print("4. LEAD/LAG: ATRSS TRADES WITHIN N DAYS OF HELIX ENTRY (same symbol)")
    print("=" * 90)

    atrss_baseline_r = np.mean([t["R"] for t in atrss])
    print(f"\n  ATRSS trades that enter within N days AFTER a Helix entry:")
    print(f"  {'Window':>8s}  {'n_atrss':>8s}  {'avgR':>7s}  {'WR':>5s}  {'totR':>8s}  "
          f"{'baseline_avgR':>13s}  {'delta':>7s}")
    print("  " + "-" * 70)

    for w in windows:
        nearby = []
        for ht in helix:
            if ht["entry"] is None:
                continue
            for at in atrss:
                if at["entry"] is None or at["symbol"] != ht["symbol"]:
                    continue
                gap = hours_between(ht["entry"], at["entry"])
                if gap is not None and 0 <= gap <= w * 24:
                    nearby.append(at)
        seen = set()
        unique = []
        for t in nearby:
            key = (t["symbol"], t["entry"])
            if key not in seen:
                seen.add(key)
                unique.append(t)
        if unique:
            avg = np.mean([t["R"] for t in unique])
            wr = sum(1 for t in unique if t["R"] > 0) / len(unique) * 100
            tot = sum(t["R"] for t in unique)
            delta = avg - atrss_baseline_r
            print(f"  {w:>5d}d    {len(unique):8d}  {avg:+7.3f}  {wr:4.0f}%  {tot:+8.1f}  "
                  f"{atrss_baseline_r:+13.3f}  {delta:+7.3f}")

    print(f"\n  ATRSS trades that enter within N days BEFORE a Helix entry:")
    print(f"  {'Window':>8s}  {'n_atrss':>8s}  {'avgR':>7s}  {'WR':>5s}  {'totR':>8s}  "
          f"{'baseline_avgR':>13s}  {'delta':>7s}")
    print("  " + "-" * 70)

    for w in windows:
        nearby = []
        for ht in helix:
            if ht["entry"] is None:
                continue
            for at in atrss:
                if at["entry"] is None or at["symbol"] != ht["symbol"]:
                    continue
                gap = hours_between(at["entry"], ht["entry"])
                if gap is not None and 0 <= gap <= w * 24:
                    nearby.append(at)
        seen = set()
        unique = []
        for t in nearby:
            key = (t["symbol"], t["entry"])
            if key not in seen:
                seen.add(key)
                unique.append(t)
        if unique:
            avg = np.mean([t["R"] for t in unique])
            wr = sum(1 for t in unique if t["R"] > 0) / len(unique) * 100
            tot = sum(t["R"] for t in unique)
            delta = avg - atrss_baseline_r
            print(f"  {w:>5d}d    {len(unique):8d}  {avg:+7.3f}  {wr:4.0f}%  {tot:+8.1f}  "
                  f"{atrss_baseline_r:+13.3f}  {delta:+7.3f}")

    # ===================================================================
    # 5. BREAKOUT AS A SIGNAL: HELIX/ATRSS PERFORMANCE NEAR BREAKOUT ENTRIES
    # ===================================================================
    print("\n" + "=" * 90)
    print("5. BREAKOUT AS LEADING INDICATOR: OTHER STRATEGIES WITHIN 7d AFTER BREAKOUT")
    print("=" * 90)

    for other_name, other_trades in [("Helix", helix), ("ATRSS", atrss)]:
        baseline_r = np.mean([t["R"] for t in other_trades])
        nearby = []
        for bt in breakout:
            if bt["entry"] is None:
                continue
            for ot in other_trades:
                if ot["entry"] is None or ot["symbol"] != bt["symbol"]:
                    continue
                gap = hours_between(bt["entry"], ot["entry"])
                if gap is not None and 0 <= gap <= 7 * 24:
                    nearby.append(ot)
        seen = set()
        unique = []
        for t in nearby:
            key = (t["symbol"], t["entry"])
            if key not in seen:
                seen.add(key)
                unique.append(t)

        if unique:
            avg = np.mean([t["R"] for t in unique])
            wr = sum(1 for t in unique if t["R"] > 0) / len(unique) * 100
            delta = avg - baseline_r
            print(f"\n  {other_name} trades within 7d after Breakout entry:")
            print(f"    n={len(unique)}  avgR={avg:+.3f}  WR={wr:.0f}%  "
                  f"(baseline: {baseline_r:+.3f}, delta: {delta:+.3f})")
        else:
            print(f"\n  {other_name} trades within 7d after Breakout entry: none")

    # ===================================================================
    # 6. CLUSTER ANALYSIS: Periods with multi-strategy activity
    # ===================================================================
    print("\n" + "=" * 90)
    print("6. TRADE CLUSTERS: 7-Day Windows with 2+ Strategy Entries (same symbol)")
    print("=" * 90)

    # Build (symbol, date) -> strategy map
    all_trades = atrss + helix + breakout
    entry_map = defaultdict(lambda: defaultdict(list))  # sym -> date -> [trades]
    for t in all_trades:
        if t["entry"]:
            entry_map[t["symbol"]][t["entry"].date()].append(t)

    # Find 7-day clusters with entries from 2+ strategies
    clusters = []
    for sym in SYMBOLS:
        dates = sorted(entry_map[sym].keys())
        i = 0
        while i < len(dates):
            window_start = dates[i]
            window_end = window_start + timedelta(days=7)
            cluster_trades = []
            j = i
            while j < len(dates) and dates[j] <= window_end:
                cluster_trades.extend(entry_map[sym][dates[j]])
                j += 1
            strats_in_cluster = set(t["strategy"] for t in cluster_trades)
            if len(strats_in_cluster) >= 2:
                clusters.append({
                    "symbol": sym,
                    "start": window_start,
                    "end": min(window_end, dates[j-1] if j > i else window_start),
                    "trades": cluster_trades,
                    "strategies": strats_in_cluster,
                    "n": len(cluster_trades),
                    "avg_r": np.mean([t["R"] for t in cluster_trades]),
                    "total_r": sum(t["R"] for t in cluster_trades),
                })
            i = j if j > i else i + 1

    print(f"\n  Found {len(clusters)} clusters with 2+ strategies in 7-day window\n")

    # Summarize cluster performance
    cluster_rs = [c["avg_r"] for c in clusters]
    cluster_tots = [c["total_r"] for c in clusters]
    if clusters:
        print(f"  Cluster avg R (per trade):  {np.mean(cluster_rs):+.3f}")
        print(f"  Cluster total R:            {sum(cluster_tots):+.1f}")
        print(f"  Cluster avg total R:        {np.mean(cluster_tots):+.3f}")

    # Show best and worst clusters
    clusters_sorted = sorted(clusters, key=lambda c: c["total_r"], reverse=True)
    print(f"\n  Top 10 clusters:")
    print(f"    {'Sym':5s} {'Start':>12s}  {'Strats':20s} {'n':>3s} {'TotR':>7s} {'AvgR':>7s}")
    print("    " + "-" * 60)
    for c in clusters_sorted[:10]:
        strats = "+".join(sorted(c["strategies"]))
        print(f"    {c['symbol']:5s} {str(c['start']):>12s}  {strats:20s} {c['n']:3d} "
              f"{c['total_r']:+7.1f} {c['avg_r']:+7.3f}")

    print(f"\n  Bottom 10 clusters:")
    print(f"    {'Sym':5s} {'Start':>12s}  {'Strats':20s} {'n':>3s} {'TotR':>7s} {'AvgR':>7s}")
    print("    " + "-" * 60)
    for c in clusters_sorted[-10:]:
        strats = "+".join(sorted(c["strategies"]))
        print(f"    {c['symbol']:5s} {str(c['start']):>12s}  {strats:20s} {c['n']:3d} "
              f"{c['total_r']:+7.1f} {c['avg_r']:+7.3f}")

    # ===================================================================
    # 7. DIRECTION CONFLUENCE: Same symbol, same direction, nearby entries
    # ===================================================================
    print("\n" + "=" * 90)
    print("7. DIRECTION CONFLUENCE: Helix R when entering same dir as recent ATRSS")
    print("=" * 90)

    # For each Helix trade, check if ATRSS had a same-direction trade in the prior 7 days
    confirmed = []
    contra = []
    solo = []

    for ht in helix:
        if ht["entry"] is None:
            continue
        recent_atrss_dirs = set()
        for at in atrss:
            if at["entry"] is None or at["symbol"] != ht["symbol"]:
                continue
            # ATRSS entered in the last 14 days and position is still open OR just entered
            gap = hours_between(at["entry"], ht["entry"])
            if gap is not None and 0 <= gap <= 14 * 24:
                # Check if ATRSS position was still open at Helix entry
                if at["exit"] and at["exit"] >= ht["entry"]:
                    recent_atrss_dirs.add(at["direction"])

        if not recent_atrss_dirs:
            solo.append(ht)
        elif ht["direction"] in recent_atrss_dirs:
            confirmed.append(ht)
        else:
            contra.append(ht)

    def _stats(trades, label):
        if not trades:
            print(f"    {label:35s}  n=  0")
            return
        rs = [t["R"] for t in trades]
        wr = sum(1 for r in rs if r > 0) / len(rs) * 100
        print(f"    {label:35s}  n={len(trades):3d}  avgR={np.mean(rs):+.3f}  "
              f"WR={wr:.0f}%  totR={sum(rs):+.1f}")

    _stats(confirmed, "Helix CONFIRMED by open ATRSS")
    _stats(contra, "Helix CONTRA to open ATRSS")
    _stats(solo, "Helix SOLO (no open ATRSS)")

    # ===================================================================
    # 8. ATRSS wins/losses when Helix recently won/lost
    # ===================================================================
    print("\n" + "=" * 90)
    print("8. CROSS-STRATEGY MOMENTUM: ATRSS performance after recent Helix win/loss")
    print("=" * 90)

    # For each ATRSS trade, check the last Helix trade in same symbol that CLOSED before entry
    after_helix_win = []
    after_helix_loss = []
    no_recent_helix = []

    for at in atrss:
        if at["entry"] is None:
            continue
        # Find most recent Helix trade that closed before this ATRSS entry
        recent_ht = None
        for ht in helix:
            if (ht["symbol"] == at["symbol"] and ht["exit"] and
                    ht["exit"] < at["entry"]):
                gap = hours_between(ht["exit"], at["entry"])
                if gap is not None and gap <= 7 * 24:
                    if recent_ht is None or ht["exit"] > recent_ht["exit"]:
                        recent_ht = ht

        if recent_ht is None:
            no_recent_helix.append(at)
        elif recent_ht["R"] > 0:
            after_helix_win.append(at)
        else:
            after_helix_loss.append(at)

    _stats(after_helix_win, "ATRSS after Helix WIN (7d)")
    _stats(after_helix_loss, "ATRSS after Helix LOSS (7d)")
    _stats(no_recent_helix, "ATRSS with no recent Helix")

    # Reverse: Helix after ATRSS result
    print(f"\n  Reverse: Helix performance after recent ATRSS close:")
    after_atrss_win = []
    after_atrss_loss = []
    no_recent_atrss = []

    for ht in helix:
        if ht["entry"] is None:
            continue
        recent_at = None
        for at in atrss:
            if (at["symbol"] == ht["symbol"] and at["exit"] and
                    at["exit"] < ht["entry"]):
                gap = hours_between(at["exit"], ht["entry"])
                if gap is not None and gap <= 7 * 24:
                    if recent_at is None or at["exit"] > recent_at["exit"]:
                        recent_at = at

        if recent_at is None:
            no_recent_atrss.append(ht)
        elif recent_at["R"] > 0:
            after_atrss_win.append(ht)
        else:
            after_atrss_loss.append(ht)

    _stats(after_atrss_win, "Helix after ATRSS WIN (7d)")
    _stats(after_atrss_loss, "Helix after ATRSS LOSS (7d)")
    _stats(no_recent_atrss, "Helix with no recent ATRSS")

    # ===================================================================
    # 9. DRY SPELL COVERAGE
    # ===================================================================
    print("\n" + "=" * 90)
    print("9. DRY SPELL COVERAGE: Does one strategy fill gaps when another is inactive?")
    print("=" * 90)

    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        # Find gaps in ATRSS (periods with no open ATRSS position)
        a_sym = sorted([t for t in atrss if t["symbol"] == sym and t["entry"]], key=lambda t: t["entry"])
        h_sym = sorted([t for t in helix if t["symbol"] == sym and t["entry"]], key=lambda t: t["entry"])
        b_sym = sorted([t for t in breakout if t["symbol"] == sym and t["entry"]], key=lambda t: t["entry"])

        if not a_sym:
            print("    No ATRSS trades")
            continue

        # Find ATRSS dry spells > 14 days
        gaps = []
        for i in range(len(a_sym) - 1):
            gap_start = a_sym[i]["exit"]
            gap_end = a_sym[i+1]["entry"]
            if gap_start and gap_end:
                gap_days = (gap_end - gap_start).days
                if gap_days > 14:
                    # Count Helix trades during this gap
                    h_during = [t for t in h_sym if t["entry"] and gap_start <= t["entry"] <= gap_end]
                    b_during = [t for t in b_sym if t["entry"] and gap_start <= t["entry"] <= gap_end]
                    h_r = sum(t["R"] for t in h_during)
                    b_r = sum(t["R"] for t in b_during)
                    gaps.append({
                        "start": gap_start, "end": gap_end, "days": gap_days,
                        "helix_n": len(h_during), "helix_r": h_r,
                        "breakout_n": len(b_during), "breakout_r": b_r,
                    })

        if gaps:
            print(f"    ATRSS dry spells > 14 days: {len(gaps)}")
            total_h_n = sum(g["helix_n"] for g in gaps)
            total_h_r = sum(g["helix_r"] for g in gaps)
            total_b_n = sum(g["breakout_n"] for g in gaps)
            total_b_r = sum(g["breakout_r"] for g in gaps)
            total_days = sum(g["days"] for g in gaps)
            print(f"    Total gap days: {total_days}")
            print(f"    Helix during gaps:    {total_h_n:3d} trades, {total_h_r:+7.1f} R")
            print(f"    Breakout during gaps: {total_b_n:3d} trades, {total_b_r:+7.1f} R")
            # Show the longest gaps
            gaps_sorted = sorted(gaps, key=lambda g: g["days"], reverse=True)
            for g in gaps_sorted[:5]:
                print(f"      {g['start'].date()} to {g['end'].date()} ({g['days']:3d}d): "
                      f"H={g['helix_n']}t/{g['helix_r']:+.1f}R  B={g['breakout_n']}t/{g['breakout_r']:+.1f}R")

    print()


if __name__ == "__main__":
    main()
