"""Analyze the pre-ATRSS Helix loser pattern to find actionable signals.

Questions:
  1. Can we detect the ATRSS "arming" condition and use it as a Helix filter/tighten?
  2. When Helix loses, does it predict an imminent ATRSS entry?
  3. What Helix class/direction/setup characteristics mark the pre-ATRSS losers?
  4. Could Helix use a tighter stop when ATRSS conditions are forming?
  5. If we skip/tighten Helix trades that precede ATRSS, what's the R impact?
"""
import logging
from collections import defaultdict
from datetime import timedelta
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
                "entry_price": t.entry_price,
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
                "origin_tf": t.origin_tf,
                "bars": t.bars_held, "pnl": t.pnl_dollars,
                "exit_reason": t.exit_reason,
                "entry_price": t.entry_price,
                "adx": t.adx_at_entry,
                "regime": t.regime_at_entry,
            })
    return trades


def hours_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt2 - dt1).total_seconds() / 3600


def _stats(trades, label=""):
    if not trades:
        return {"n": 0, "avg_r": 0, "wr": 0, "tot_r": 0}
    rs = [t["R"] for t in trades]
    wr = sum(1 for r in rs if r > 0) / len(rs) * 100
    return {"n": len(trades), "avg_r": np.mean(rs), "wr": wr, "tot_r": sum(rs)}


def _print_stats(trades, label):
    s = _stats(trades)
    if s["n"] == 0:
        print(f"    {label:45s}  n=  0")
    else:
        print(f"    {label:45s}  n={s['n']:3d}  avgR={s['avg_r']:+.3f}  "
              f"WR={s['wr']:.0f}%  totR={s['tot_r']:+.1f}")


def main():
    print("Running backtests...")
    print("  ATRSS...", end=" ", flush=True)
    atrss = run_atrss()
    print(f"{len(atrss)}")
    print("  Helix...", end=" ", flush=True)
    helix = run_helix()
    print(f"{len(helix)}")

    print()
    print("=" * 90)
    print("PRE-ATRSS HELIX LOSER PATTERN -- DEEP ANALYSIS")
    print("=" * 90)

    # ===================================================================
    # 1. Profile the pre-ATRSS Helix losers: what makes them different?
    # ===================================================================
    print("\n" + "=" * 90)
    print("1. PROFILE: Helix trades 1-5 days BEFORE an ATRSS entry (same symbol)")
    print("=" * 90)

    pre_atrss = []      # Helix entered 1-5d before ATRSS
    not_pre_atrss = []  # All other Helix trades

    for ht in helix:
        if ht["entry"] is None:
            continue
        is_pre = False
        for at in atrss:
            if at["entry"] is None or at["symbol"] != ht["symbol"]:
                continue
            gap = hours_between(ht["entry"], at["entry"])
            if gap is not None and 0 < gap <= 5 * 24:
                is_pre = True
                break
        if is_pre:
            pre_atrss.append(ht)
        else:
            not_pre_atrss.append(ht)

    _print_stats(pre_atrss, "Helix PRE-ATRSS (1-5d before)")
    _print_stats(not_pre_atrss, "Helix NOT pre-ATRSS")

    # Break down by class
    print("\n  By setup class:")
    for cls in ["A", "C", "D"]:
        pre = [t for t in pre_atrss if t["class"] == cls]
        other = [t for t in not_pre_atrss if t["class"] == cls]
        if pre or other:
            _print_stats(pre, f"  PRE-ATRSS Class {cls}")
            _print_stats(other, f"  Other Class {cls}")

    # By direction
    print("\n  By direction:")
    for d in ["LONG", "SHORT"]:
        pre = [t for t in pre_atrss if t["direction"] == d]
        other = [t for t in not_pre_atrss if t["direction"] == d]
        if pre or other:
            _print_stats(pre, f"  PRE-ATRSS {d}")
            _print_stats(other, f"  Other {d}")

    # By origin timeframe
    print("\n  By origin TF:")
    for tf in ["1H", "4H"]:
        pre = [t for t in pre_atrss if t.get("origin_tf") == tf]
        other = [t for t in not_pre_atrss if t.get("origin_tf") == tf]
        if pre or other:
            _print_stats(pre, f"  PRE-ATRSS {tf}")
            _print_stats(other, f"  Other {tf}")

    # By exit reason
    print("\n  By exit reason:")
    reasons = set(t.get("exit_reason", "") for t in pre_atrss)
    for r in sorted(reasons):
        pre = [t for t in pre_atrss if t.get("exit_reason") == r]
        other = [t for t in not_pre_atrss if t.get("exit_reason") == r]
        _print_stats(pre, f"  PRE-ATRSS exit={r}")
        _print_stats(other, f"  Other exit={r}")

    # By regime
    print("\n  By regime at entry:")
    regimes = set(t.get("regime", "") for t in pre_atrss)
    for r in sorted(regimes):
        if not r:
            continue
        pre = [t for t in pre_atrss if t.get("regime") == r]
        other = [t for t in not_pre_atrss if t.get("regime") == r]
        _print_stats(pre, f"  PRE-ATRSS regime={r}")
        _print_stats(other, f"  Other regime={r}")

    # By symbol
    print("\n  By symbol:")
    for sym in SYMBOLS:
        pre = [t for t in pre_atrss if t["symbol"] == sym]
        other = [t for t in not_pre_atrss if t["symbol"] == sym]
        _print_stats(pre, f"  PRE-ATRSS {sym}")
        _print_stats(other, f"  Other {sym}")

    # ===================================================================
    # 2. Reverse: When Helix LOSES, does an ATRSS entry follow?
    # ===================================================================
    print("\n" + "=" * 90)
    print("2. REVERSE: After Helix LOSS, how often does ATRSS enter within N days?")
    print("=" * 90)

    helix_losses = [t for t in helix if t["R"] < 0 and t["exit"]]
    helix_wins = [t for t in helix if t["R"] > 0 and t["exit"]]

    for window_days in [1, 3, 5, 7]:
        loss_followed = 0
        win_followed = 0
        for ht in helix_losses:
            for at in atrss:
                if at["entry"] and at["symbol"] == ht["symbol"]:
                    gap = hours_between(ht["exit"], at["entry"])
                    if gap is not None and 0 <= gap <= window_days * 24:
                        loss_followed += 1
                        break
        for ht in helix_wins:
            for at in atrss:
                if at["entry"] and at["symbol"] == ht["symbol"]:
                    gap = hours_between(ht["exit"], at["entry"])
                    if gap is not None and 0 <= gap <= window_days * 24:
                        win_followed += 1
                        break
        loss_pct = loss_followed / len(helix_losses) * 100 if helix_losses else 0
        win_pct = win_followed / len(helix_wins) * 100 if helix_wins else 0
        print(f"  Within {window_days}d: Helix LOSS -> ATRSS entry: "
              f"{loss_followed}/{len(helix_losses)} ({loss_pct:.1f}%)   |   "
              f"Helix WIN -> ATRSS entry: "
              f"{win_followed}/{len(helix_wins)} ({win_pct:.1f}%)")

    # ===================================================================
    # 3. Consecutive Helix losses as ATRSS predictor
    # ===================================================================
    print("\n" + "=" * 90)
    print("3. CONSECUTIVE HELIX LOSSES as ATRSS entry predictor")
    print("=" * 90)

    for sym in SYMBOLS:
        h_sym = sorted([t for t in helix if t["symbol"] == sym and t["exit"]],
                       key=lambda t: t["exit"])
        a_sym = [t for t in atrss if t["symbol"] == sym and t["entry"]]

        # Find runs of consecutive Helix losses
        streaks = []
        current_streak = 0
        streak_end = None
        for i, ht in enumerate(h_sym):
            if ht["R"] < 0:
                current_streak += 1
                streak_end = ht["exit"]
            else:
                if current_streak >= 2:
                    streaks.append({"length": current_streak, "end": streak_end})
                current_streak = 0
        if current_streak >= 2:
            streaks.append({"length": current_streak, "end": streak_end})

        if not streaks:
            continue

        # For each streak of 2+ losses, did ATRSS enter within 5d?
        followed = 0
        not_followed = 0
        for s in streaks:
            found = False
            for at in a_sym:
                gap = hours_between(s["end"], at["entry"])
                if gap is not None and 0 <= gap <= 5 * 24:
                    found = True
                    break
            if found:
                followed += 1
            else:
                not_followed += 1
        print(f"  {sym}: {len(streaks)} streaks of 2+ Helix losses, "
              f"{followed} followed by ATRSS within 5d "
              f"({followed/len(streaks)*100:.0f}%)")

    # ===================================================================
    # 4. Helix direction vs ATRSS direction for the pre-ATRSS losers
    # ===================================================================
    print("\n" + "=" * 90)
    print("4. PRE-ATRSS Helix: Same vs opposite direction as upcoming ATRSS")
    print("=" * 90)

    same_dir_pre = []
    opp_dir_pre = []

    for ht in helix:
        if ht["entry"] is None:
            continue
        for at in atrss:
            if at["entry"] is None or at["symbol"] != ht["symbol"]:
                continue
            gap = hours_between(ht["entry"], at["entry"])
            if gap is not None and 0 < gap <= 5 * 24:
                if ht["direction"] == at["direction"]:
                    same_dir_pre.append(ht)
                else:
                    opp_dir_pre.append(ht)
                break

    _print_stats(same_dir_pre, "PRE-ATRSS Helix SAME direction")
    _print_stats(opp_dir_pre, "PRE-ATRSS Helix OPPOSITE direction")

    # ===================================================================
    # 5. Can we detect "ATRSS is arming" from Helix's perspective?
    # ===================================================================
    print("\n" + "=" * 90)
    print("5. ATRSS ARMING DETECTION: Helix during ATRSS pullback phase")
    print("=" * 90)
    print("""
  ATRSS enters on pullback-to-EMA bounces. Before entry, price is retracing
  toward the EMA. This pullback phase creates counter-trend chop that stops
  out Helix momentum trades.

  Potential detection methods:
  a) ATRSS has a signal pipeline that detects pullback conditions before entry.
     If exposed, Helix could check "is ATRSS pullback arming?" before entering.
  b) Simpler proxy: check if price is within X% of a key EMA level.
  c) Use Helix's own stop-out as the signal: if Helix just got stopped out,
     check if an ATRSS entry follows.
    """)

    # ===================================================================
    # 6. Actionable: If Helix tightens stop after N consecutive losses,
    #    what's the impact?
    # ===================================================================
    print("=" * 90)
    print("6. SIMULATION: Helix with stop tightening after consecutive losses")
    print("=" * 90)

    for sym in SYMBOLS:
        h_sym = sorted([t for t in helix if t["symbol"] == sym and t["entry"]],
                       key=lambda t: t["entry"])
        if not h_sym:
            continue

        baseline_r = sum(t["R"] for t in h_sym)
        baseline_n = len(h_sym)

        # Simulate: after 2 consecutive losses, skip the next trade
        consec_losses = 0
        skipped = []
        kept = []
        for ht in h_sym:
            if consec_losses >= 2:
                skipped.append(ht)
                # Reset streak tracking — we skipped, so this doesn't count
                if ht["R"] < 0:
                    consec_losses += 1  # would have been another loss
                else:
                    consec_losses = 0
            else:
                kept.append(ht)
                if ht["R"] < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0

        skip_r = sum(t["R"] for t in kept)
        skipped_r = sum(t["R"] for t in skipped)
        print(f"\n  {sym}: Skip after 2 consecutive losses:")
        print(f"    Baseline:  n={baseline_n:3d}  totR={baseline_r:+7.1f}")
        print(f"    Kept:      n={len(kept):3d}  totR={skip_r:+7.1f}")
        print(f"    Skipped:   n={len(skipped):3d}  totR={skipped_r:+7.1f}  "
              f"avgR={np.mean([t['R'] for t in skipped]):+.3f}" if skipped else
              f"    Skipped:   n=  0")

    # Same but skip after 3 consecutive losses
    print()
    for sym in SYMBOLS:
        h_sym = sorted([t for t in helix if t["symbol"] == sym and t["entry"]],
                       key=lambda t: t["entry"])
        if not h_sym:
            continue

        baseline_r = sum(t["R"] for t in h_sym)

        consec_losses = 0
        skipped = []
        kept = []
        for ht in h_sym:
            if consec_losses >= 3:
                skipped.append(ht)
                if ht["R"] < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0
            else:
                kept.append(ht)
                if ht["R"] < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0

        skip_r = sum(t["R"] for t in kept)
        skipped_r = sum(t["R"] for t in skipped)
        print(f"  {sym}: Skip after 3 consecutive losses:")
        print(f"    Baseline:  n={len(h_sym):3d}  totR={baseline_r:+7.1f}")
        print(f"    Kept:      n={len(kept):3d}  totR={skip_r:+7.1f}")
        print(f"    Skipped:   n={len(skipped):3d}  totR={skipped_r:+7.1f}  "
              f"avgR={np.mean([t['R'] for t in skipped]):+.3f}" if skipped else
              f"    Skipped:   n=  0")

    # ===================================================================
    # 7. Helix stop-out as ATRSS trigger: if Helix just got stopped,
    #    should ATRSS entries that follow get a size boost?
    # ===================================================================
    print("\n" + "=" * 90)
    print("7. HELIX STOP-OUT AS ATRSS BOOSTER: ATRSS entries after Helix stop-out")
    print("=" * 90)

    helix_stopped = [t for t in helix if t.get("exit_reason") == "STOP" and t["R"] < 0 and t["exit"]]

    atrss_after_hstop = []
    atrss_not_after_hstop = []

    for at in atrss:
        if at["entry"] is None:
            continue
        found = False
        for ht in helix_stopped:
            if ht["symbol"] != at["symbol"]:
                continue
            gap = hours_between(ht["exit"], at["entry"])
            if gap is not None and 0 <= gap <= 3 * 24:
                found = True
                break
        if found:
            atrss_after_hstop.append(at)
        else:
            atrss_not_after_hstop.append(at)

    _print_stats(atrss_after_hstop, "ATRSS within 3d after Helix STOP-OUT")
    _print_stats(atrss_not_after_hstop, "ATRSS not after Helix STOP-OUT")

    # Same with 5d window
    atrss_after_hstop_5d = []
    atrss_not_after_hstop_5d = []
    for at in atrss:
        if at["entry"] is None:
            continue
        found = False
        for ht in helix_stopped:
            if ht["symbol"] != at["symbol"]:
                continue
            gap = hours_between(ht["exit"], at["entry"])
            if gap is not None and 0 <= gap <= 5 * 24:
                found = True
                break
        if found:
            atrss_after_hstop_5d.append(at)
        else:
            atrss_not_after_hstop_5d.append(at)

    _print_stats(atrss_after_hstop_5d, "ATRSS within 5d after Helix STOP-OUT")
    _print_stats(atrss_not_after_hstop_5d, "ATRSS not after Helix STOP-OUT")

    # By symbol
    print("\n  By symbol (3d window):")
    for sym in SYMBOLS:
        after = [t for t in atrss_after_hstop if t["symbol"] == sym]
        other = [t for t in atrss_not_after_hstop if t["symbol"] == sym]
        if after:
            _print_stats(after, f"  {sym} ATRSS after Helix stop")
            _print_stats(other, f"  {sym} ATRSS other")

    # ===================================================================
    # 8. ATRSS entry as Helix trailing stop trigger
    # ===================================================================
    print("\n" + "=" * 90)
    print("8. ATRSS ENTRY AS HELIX STOP TRIGGER: If ATRSS enters while Helix")
    print("   is in a losing position, would flattening Helix at that moment help?")
    print("=" * 90)

    # Find Helix trades that were open AND losing when ATRSS entered
    # We can't know the exact intra-trade P&L, but we know the final R
    # and whether the trade was still open at ATRSS entry time
    helix_open_at_atrss = []
    for at in atrss:
        if at["entry"] is None:
            continue
        for ht in helix:
            if (ht["symbol"] == at["symbol"] and ht["entry"] and ht["exit"] and
                    ht["entry"] < at["entry"] < ht["exit"]):
                hours_in = hours_between(ht["entry"], at["entry"])
                hours_total = hours_between(ht["entry"], ht["exit"])
                pct_through = hours_in / hours_total if hours_total > 0 else 0
                helix_open_at_atrss.append({
                    **ht,
                    "atrss_entry": at["entry"],
                    "atrss_dir": at["direction"],
                    "pct_through": pct_through,
                })

    # Of these, which ended as losers?
    losers = [t for t in helix_open_at_atrss if t["R"] < 0]
    winners = [t for t in helix_open_at_atrss if t["R"] > 0]

    print(f"\n  Helix trades open when ATRSS entered: {len(helix_open_at_atrss)}")
    print(f"    Eventually won:  {len(winners)}  avgR={np.mean([t['R'] for t in winners]):+.3f}" if winners else "")
    print(f"    Eventually lost: {len(losers)}  avgR={np.mean([t['R'] for t in losers]):+.3f}" if losers else "")

    if losers:
        # How early did ATRSS signal come? (% through the trade)
        early = [t for t in losers if t["pct_through"] < 0.5]
        late = [t for t in losers if t["pct_through"] >= 0.5]
        print(f"\n    Losers where ATRSS entered in first half of Helix trade:")
        _print_stats(early, "    Early ATRSS signal (< 50% through)")
        _print_stats(late, "    Late ATRSS signal (>= 50% through)")

        # Same direction vs opposite
        same = [t for t in losers if t["direction"] == t["atrss_dir"]]
        opp = [t for t in losers if t["direction"] != t["atrss_dir"]]
        print(f"\n    Direction match:")
        _print_stats(same, "    Helix same dir as ATRSS")
        _print_stats(opp, "    Helix opposite dir to ATRSS")

    print()


if __name__ == "__main__":
    main()
