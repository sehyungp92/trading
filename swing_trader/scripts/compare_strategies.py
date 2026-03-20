"""Compare trade timing across all 3 strategies to evaluate complementarity.

Runs all three backtests, extracts trade-level data, and analyzes:
  1. Same-day entry overlap
  2. Concurrent position overlap (time-in-market)
  3. Monthly R correlation
  4. Calendar coverage
  5. Direction agreement
"""
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path("data/raw")
SYMBOLS = ["QQQ", "USO", "GLD", "IBIT"]
FIXED_QTY = 10
EQUITY = 100_000.0


def run_atrss():
    """Run ATRSS backtest, return list of trade dicts."""
    from backtest.config import BacktestConfig, SlippageConfig
    from backtest.engine.portfolio_engine import PortfolioData, run_independent
    from backtest.data.cache import load_bars
    from backtest.data.preprocessing import (
        align_daily_to_hourly, build_numpy_arrays, filter_rth, normalize_timezone,
    )
    from strategy.config import SYMBOL_CONFIGS

    data = PortfolioData()
    for sym in SYMBOLS:
        h_df = normalize_timezone(load_bars(DATA_DIR / f"{sym}_1h.parquet"))
        h_df = filter_rth(h_df)
        d_df = normalize_timezone(load_bars(DATA_DIR / f"{sym}_1d.parquet"))
        data.hourly[sym] = build_numpy_arrays(h_df)
        data.daily[sym] = build_numpy_arrays(d_df)
        data.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)

    slippage = SlippageConfig(commission_per_contract=1.00)
    config = BacktestConfig(
        symbols=SYMBOLS, initial_equity=EQUITY,
        data_dir=DATA_DIR, slippage=slippage, fixed_qty=FIXED_QTY,
    )
    result = run_independent(data, config)

    trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            trades.append({
                "strategy": "ATRSS",
                "symbol": t.symbol or sym,
                "entry": t.entry_time,
                "exit": t.exit_time,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "R": t.r_multiple,
                "class": t.entry_type,
                "bars": t.bars_held,
                "pnl": t.pnl_dollars,
            })
    return trades


def run_helix():
    """Run Helix backtest, return list of trade dicts."""
    from backtest.config import SlippageConfig
    from backtest.config_helix import HelixBacktestConfig
    from backtest.engine.helix_portfolio_engine import load_helix_data, run_helix_independent
    from strategy_2.config import SYMBOL_CONFIGS

    data = load_helix_data(SYMBOLS, DATA_DIR)
    slippage = SlippageConfig(commission_per_contract=1.00)
    config = HelixBacktestConfig(
        symbols=SYMBOLS, initial_equity=EQUITY,
        data_dir=DATA_DIR, slippage=slippage, fixed_qty=FIXED_QTY,
    )
    result = run_helix_independent(data, config)

    trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            trades.append({
                "strategy": "Helix",
                "symbol": t.symbol or sym,
                "entry": t.entry_time,
                "exit": t.exit_time,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "R": t.r_multiple,
                "class": t.setup_class,
                "bars": t.bars_held,
                "pnl": t.pnl_dollars,
            })
    return trades


def run_breakout():
    """Run Breakout backtest, return list of trade dicts."""
    from backtest.config import SlippageConfig
    from backtest.config_breakout import BreakoutBacktestConfig
    from backtest.engine.breakout_portfolio_engine import load_breakout_data, run_breakout_independent
    from strategy_3.config import SYMBOL_CONFIGS

    data = load_breakout_data(SYMBOLS, DATA_DIR)
    slippage = SlippageConfig(commission_per_contract=1.00)
    config = BreakoutBacktestConfig(
        symbols=SYMBOLS, initial_equity=EQUITY,
        data_dir=DATA_DIR, slippage=slippage, fixed_qty=FIXED_QTY,
    )
    result = run_breakout_independent(data, config)

    trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            trades.append({
                "strategy": "Breakout",
                "symbol": t.symbol or sym,
                "entry": t.entry_time,
                "exit": t.exit_time,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "R": t.r_multiple,
                "class": t.entry_type,
                "bars": t.bars_held,
                "pnl": t.pnl_dollars,
            })
    return trades


def _strip_tz(dt):
    """Strip timezone info for comparison."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def date_range(dt1, dt2):
    """Return set of dates between two datetimes (inclusive)."""
    if dt1 is None or dt2 is None:
        return set()
    d1, d2 = _strip_tz(dt1).date(), _strip_tz(dt2).date()
    days = set()
    current = d1
    while current <= d2:
        days.add(current)
        current += timedelta(days=1)
    return days


def pearson_corr(xs, ys):
    if len(xs) < 3:
        return float('nan')
    mx, my = np.mean(xs), np.mean(ys)
    cov = np.mean([(x - mx) * (y - my) for x, y in zip(xs, ys)])
    sx = np.std(xs)
    sy = np.std(ys)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def main():
    print("Running all 3 backtests (fixed_qty=10)...")
    print()

    print("  [1/3] ATRSS...", end=" ", flush=True)
    atrss_trades = run_atrss()
    print(f"{len(atrss_trades)} trades")

    print("  [2/3] Helix...", end=" ", flush=True)
    helix_trades = run_helix()
    print(f"{len(helix_trades)} trades")

    print("  [3/3] Breakout...", end=" ", flush=True)
    breakout_trades = run_breakout()
    print(f"{len(breakout_trades)} trades")

    # Normalize all timestamps to naive (strip tz)
    for trades in (atrss_trades, helix_trades, breakout_trades):
        for t in trades:
            t["entry"] = _strip_tz(t["entry"])
            t["exit"] = _strip_tz(t["exit"])

    all_strats = {"ATRSS": atrss_trades, "Helix": helix_trades, "Breakout": breakout_trades}

    print()
    print("=" * 90)
    print("STRATEGY COMPLEMENTARITY ANALYSIS")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # 1. Basic counts per symbol
    # -----------------------------------------------------------------------
    print("\n--- Trade Counts by Symbol ---")
    print(f"  {'Strategy':10s}  {'Total':>6s}  {'QQQ':>5s}  {'USO':>5s}  {'GLD':>5s}  {'IBIT':>5s}  "
          f"{'AvgR':>7s}  {'TotalR':>8s}  {'Net$':>10s}")
    print("  " + "-" * 82)
    for name, trades in all_strats.items():
        by_sym = defaultdict(list)
        for t in trades:
            by_sym[t["symbol"]].append(t)
        total_r = sum(t["R"] for t in trades)
        total_pnl = sum(t["pnl"] for t in trades)
        avg_r = total_r / len(trades) if trades else 0
        counts = {s: len(by_sym.get(s, [])) for s in SYMBOLS}
        print(f"  {name:10s}  {len(trades):6d}  {counts['QQQ']:5d}  {counts['USO']:5d}  "
              f"{counts['GLD']:5d}  {counts['IBIT']:5d}  {avg_r:+7.3f}  {total_r:+8.1f}  "
              f"${total_pnl:+10,.0f}")

    # -----------------------------------------------------------------------
    # 2. Same-day entry overlap (all symbols combined)
    # -----------------------------------------------------------------------
    print("\n--- Same-Day Entry Overlap (all symbols) ---")
    entry_dates = {}
    for name, trades in all_strats.items():
        entry_dates[name] = set()
        for t in trades:
            if t["entry"]:
                entry_dates[name].add((t["symbol"], t["entry"].date()))

    pairs = [("ATRSS", "Helix"), ("ATRSS", "Breakout"), ("Helix", "Breakout")]
    for a, b in pairs:
        overlap = entry_dates[a] & entry_dates[b]
        total_unique = len(entry_dates[a] | entry_dates[b])
        pct = len(overlap) / total_unique * 100 if total_unique else 0
        print(f"  {a:10s} vs {b:10s}: {len(overlap):3d} same (sym,date) pairs "
              f"out of {total_unique} ({pct:.1f}%)")
        if overlap:
            by_sym = defaultdict(list)
            for sym, d in overlap:
                by_sym[sym].append(d)
            for sym in sorted(by_sym):
                dates_str = ", ".join(str(d) for d in sorted(by_sym[sym])[:5])
                extra = f" (+{len(by_sym[sym])-5} more)" if len(by_sym[sym]) > 5 else ""
                print(f"    {sym}: {dates_str}{extra}")

    triple = entry_dates["ATRSS"] & entry_dates["Helix"] & entry_dates["Breakout"]
    print(f"  All three:              {len(triple):3d} same (sym,date) pairs")

    # -----------------------------------------------------------------------
    # 3. Concurrent position overlap
    # -----------------------------------------------------------------------
    print("\n--- Concurrent Position Overlap (all symbols) ---")
    for a, b in pairs:
        concurrent = 0
        same_dir = 0
        opp_dir = 0
        for t1 in all_strats[a]:
            for t2 in all_strats[b]:
                if t1["symbol"] != t2["symbol"]:
                    continue
                if t1["entry"] and t2["entry"] and t1["exit"] and t2["exit"]:
                    if t1["entry"] <= t2["exit"] and t2["entry"] <= t1["exit"]:
                        concurrent += 1
                        if t1["direction"] == t2["direction"]:
                            same_dir += 1
                        else:
                            opp_dir += 1
        print(f"  {a:10s} vs {b:10s}: {concurrent:3d} concurrent positions "
              f"(same dir: {same_dir}, opposite: {opp_dir})")

    # -----------------------------------------------------------------------
    # 4. Calendar coverage by symbol
    # -----------------------------------------------------------------------
    print("\n--- Calendar Coverage (days with open position) ---")
    print(f"  {'Symbol':6s}  {'ATRSS':>6s}  {'Helix':>6s}  {'Brk':>6s}  "
          f"{'Union':>6s}  {'2+':>5s}  {'3':>5s}")
    print("  " + "-" * 52)
    for sym in SYMBOLS:
        coverage = {}
        for name, trades in all_strats.items():
            days = set()
            for t in trades:
                if t["symbol"] == sym and t["entry"] and t["exit"]:
                    days |= date_range(t["entry"], t["exit"])
            coverage[name] = days
        union = set()
        for d in coverage.values():
            union |= d
        two_plus = sum(1 for d in union if sum(1 for v in coverage.values() if d in v) >= 2)
        three = sum(1 for d in union if sum(1 for v in coverage.values() if d in v) >= 3)
        print(f"  {sym:6s}  {len(coverage['ATRSS']):6d}  {len(coverage['Helix']):6d}  "
              f"{len(coverage['Breakout']):6d}  {len(union):6d}  {two_plus:5d}  {three:5d}")

    # -----------------------------------------------------------------------
    # 5. Monthly trade distribution
    # -----------------------------------------------------------------------
    print("\n--- Monthly Trade Distribution (all symbols combined) ---")
    monthly_counts = defaultdict(lambda: defaultdict(int))
    monthly_r = defaultdict(lambda: defaultdict(float))
    for name, trades in all_strats.items():
        for t in trades:
            if t["entry"]:
                m = t["entry"].strftime("%Y-%m")
                monthly_counts[m][name] += 1
                monthly_r[m][name] += t["R"]

    months = sorted(monthly_counts.keys())
    print(f"  {'Month':8s}  {'ATRSS':>5s}  {'Helix':>5s}  {'Brk':>5s}  "
          f"{'Total':>5s}  {'A_R':>7s}  {'H_R':>7s}  {'B_R':>7s}")
    print("  " + "-" * 62)
    for m in months:
        ac = monthly_counts[m].get("ATRSS", 0)
        hc = monthly_counts[m].get("Helix", 0)
        bc = monthly_counts[m].get("Breakout", 0)
        ar = monthly_r[m].get("ATRSS", 0)
        hr = monthly_r[m].get("Helix", 0)
        br = monthly_r[m].get("Breakout", 0)
        print(f"  {m:8s}  {ac:5d}  {hc:5d}  {bc:5d}  {ac+hc+bc:5d}  "
              f"{ar:+7.1f}  {hr:+7.1f}  {br:+7.1f}")

    # -----------------------------------------------------------------------
    # 6. Monthly R correlation
    # -----------------------------------------------------------------------
    print("\n--- Monthly R Correlation ---")
    for a, b in pairs:
        common = [m for m in months
                  if monthly_r[m].get(a, 0) != 0 or monthly_r[m].get(b, 0) != 0]
        if len(common) < 5:
            print(f"  {a:10s} vs {b:10s}: insufficient data (n={len(common)})")
            continue
        va = [monthly_r[m].get(a, 0) for m in common]
        vb = [monthly_r[m].get(b, 0) for m in common]
        r = pearson_corr(va, vb)
        print(f"  {a:10s} vs {b:10s}: r = {r:+.3f}  (n={len(common)} months)")

    # -----------------------------------------------------------------------
    # 7. Holding period overlap
    # -----------------------------------------------------------------------
    print("\n--- Average Hold Times (hours) ---")
    for name, trades in all_strats.items():
        holds = []
        for t in trades:
            if t["entry"] and t["exit"]:
                h = (t["exit"] - t["entry"]).total_seconds() / 3600
                holds.append(h)
        if holds:
            print(f"  {name:10s}: mean={np.mean(holds):.1f}h  median={np.median(holds):.1f}h  "
                  f"min={min(holds):.0f}h  max={max(holds):.0f}h")

    # -----------------------------------------------------------------------
    # 8. Direction breakdown
    # -----------------------------------------------------------------------
    print("\n--- Direction Breakdown ---")
    for name, trades in all_strats.items():
        longs = sum(1 for t in trades if t["direction"] == "LONG")
        shorts = len(trades) - longs
        print(f"  {name:10s}: {longs:4d} long ({longs/len(trades)*100:.0f}%)  "
              f"{shorts:4d} short ({shorts/len(trades)*100:.0f}%)")

    # -----------------------------------------------------------------------
    # 9. Regime-phase analysis: when each strategy profits/loses
    # -----------------------------------------------------------------------
    print("\n--- Yearly R Breakdown ---")
    yearly_r = defaultdict(lambda: defaultdict(float))
    yearly_n = defaultdict(lambda: defaultdict(int))
    for name, trades in all_strats.items():
        for t in trades:
            if t["entry"]:
                y = t["entry"].year
                yearly_r[y][name] += t["R"]
                yearly_n[y][name] += 1

    years = sorted(yearly_r.keys())
    print(f"  {'Year':6s}  {'ATRSS_R':>8s} (n)  {'Helix_R':>8s} (n)  {'Brk_R':>8s} (n)  {'Combined':>9s}")
    print("  " + "-" * 72)
    for y in years:
        ar = yearly_r[y].get("ATRSS", 0)
        hr = yearly_r[y].get("Helix", 0)
        br = yearly_r[y].get("Breakout", 0)
        an = yearly_n[y].get("ATRSS", 0)
        hn = yearly_n[y].get("Helix", 0)
        bn = yearly_n[y].get("Breakout", 0)
        print(f"  {y:6d}  {ar:+8.1f} ({an:2d})  {hr:+8.1f} ({hn:2d})  "
              f"{br:+8.1f} ({bn:2d})  {ar+hr+br:+9.1f}")

    # -----------------------------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Count same-day overlaps
    ah_overlap = len(entry_dates["ATRSS"] & entry_dates["Helix"])
    ab_overlap = len(entry_dates["ATRSS"] & entry_dates["Breakout"])
    hb_overlap = len(entry_dates["Helix"] & entry_dates["Breakout"])

    total_entries = sum(len(v) for v in entry_dates.values())
    print(f"\n  Total entry (sym,date) pairs: {total_entries}")
    print(f"  ATRSS-Helix same-day:    {ah_overlap:3d} ({ah_overlap/total_entries*100:.1f}%)")
    print(f"  ATRSS-Breakout same-day: {ab_overlap:3d} ({ab_overlap/total_entries*100:.1f}%)")
    print(f"  Helix-Breakout same-day: {hb_overlap:3d} ({hb_overlap/total_entries*100:.1f}%)")

    # Monthly R correlation summary
    for a, b in pairs:
        common = [m for m in months
                  if monthly_r[m].get(a, 0) != 0 or monthly_r[m].get(b, 0) != 0]
        if len(common) >= 5:
            va = [monthly_r[m].get(a, 0) for m in common]
            vb = [monthly_r[m].get(b, 0) for m in common]
            r = pearson_corr(va, vb)
            verdict = ("REDUNDANT (high +corr)" if r > 0.5
                       else "COMPLEMENTARY (low/negative corr)" if r < 0.3
                       else "MODERATE overlap")
            print(f"  {a:10s} vs {b:10s} monthly R corr: {r:+.3f} → {verdict}")


if __name__ == "__main__":
    main()
