"""Diagnose remaining drags in VdubusNQ v4.0 backtest results.

Dumps trade-level detail for each drag pattern to understand root causes.
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

MNQ_POINT_VALUE = 2.0
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
MNQ_COMMISSION = 0.62
MNQ_RT_COMM_FEES = 1.24
FIXED_QTY = 10
DATA_DIR = Path("backtest/data/raw")
EQUITY = 100_000.0


def _to_et(dt):
    from zoneinfo import ZoneInfo
    return dt.astimezone(ZoneInfo("America/New_York"))


def main():
    from backtest.cli import _load_vdubus_data
    from backtest.config import SlippageConfig
    from backtest.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtest.engine.vdubus_engine import VdubusEngine
    from strategy_3 import config as C

    symbol = "NQ"
    vdubus_data = _load_vdubus_data(symbol, DATA_DIR)

    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = MNQ_TICK_VALUE
    C.NQ_SPEC["point_value"] = MNQ_POINT_VALUE
    C.RT_COMM_FEES = MNQ_RT_COMM_FEES

    try:
        flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )
        config = VdubusBacktestConfig(
            symbols=[symbol],
            initial_equity=EQUITY,
            data_dir=DATA_DIR,
            slippage=SlippageConfig(commission_per_contract=MNQ_COMMISSION),
            fixed_qty=FIXED_QTY,
            tick_size=MNQ_TICK_SIZE,
            point_value=MNQ_POINT_VALUE,
            flags=flags,
        )
        engine = VdubusEngine(symbol, config)
        result = engine.run(
            vdubus_data["bars_15m"],
            vdubus_data.get("bars_5m"),
            vdubus_data["hourly"],
            vdubus_data["daily_es"],
            vdubus_data["hourly_idx_map"],
            vdubus_data["daily_es_idx_map"],
            vdubus_data.get("five_to_15_idx_map"),
        )
    finally:
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    trades = result.trades
    out = []

    def p(s=""):
        out.append(s)

    def fmt_dt(dt):
        if dt is None:
            return "N/A"
        et = _to_et(dt)
        return et.strftime("%Y-%m-%d %H:%M ET")

    def trade_line(t, idx=None):
        prefix = f"  [{idx:2d}]" if idx is not None else "  "
        et_entry = _to_et(t.entry_time) if t.entry_time else None
        hour_str = et_entry.strftime("%H:%M") if et_entry else "?"
        day_str = et_entry.strftime("%a") if et_entry else "?"
        return (
            f"{prefix} {fmt_dt(t.entry_time):20s}  {t.entry_type:1s}  "
            f"{t.sub_window:7s}  {day_str:3s} {hour_str}  "
            f"R={t.r_multiple:+6.2f}  MFE={t.mfe_r:+5.2f}  MAE={t.mae_r:+5.2f}  "
            f"Bars={t.bars_held_15m:3d}  Exit={t.exit_reason:12s}  "
            f"PnL=${t.pnl_dollars:+8.0f}  "
            f"VWAP_d={t.vwap_used_at_entry:6.1f}"
        )

    # ─── OVERVIEW ──────────────────────────────────────────────────────
    p("=" * 90)
    p("  REMAINING DRAGS — DEEP INVESTIGATION")
    p("=" * 90)

    total_pnl = sum(t.pnl_dollars for t in trades)
    winners = [t for t in trades if t.pnl_dollars > 0]
    losers = [t for t in trades if t.pnl_dollars <= 0]
    p(f"\nTotal: {len(trades)} trades, ${total_pnl:+,.0f}")
    p(f"Winners: {len(winners)}, Losers: {len(losers)}")

    # ─── DRAG 1: TYPE B LEAKAGE ────────────────────────────────────────
    p("\n" + "=" * 90)
    p("  DRAG 1: TYPE B LEAKAGE (type_b_enabled flag defaults True in backtest)")
    p("=" * 90)
    type_b = [t for t in trades if t.entry_type == "B"]
    p(f"\nType B trades: {len(type_b)}")
    p(f"The backtest uses flags.type_b_enabled (defaults True), NOT C.USE_TYPE_B.")
    p(f"The runner script does not set type_b_enabled=False in the flags.")
    if type_b:
        p(f"Total Type B PnL: ${sum(t.pnl_dollars for t in type_b):+,.0f}")
        p(f"Avg R: {np.mean([t.r_multiple for t in type_b]):+.3f}")
        for i, t in enumerate(type_b):
            p(trade_line(t, i))
        p(f"\nFIX: Set type_b_enabled=False in the runner's VdubusAblationFlags.")
        p(f"Impact: Removes {len(type_b)} trades, recovers ${-sum(t.pnl_dollars for t in type_b):+,.0f}")

    # ─── DRAG 2: EVENING SESSION ────────────────────────────────────────
    p("\n" + "=" * 90)
    p("  DRAG 2: EVENING SESSION (12 trades, 33% WR, -$6,141)")
    p("=" * 90)
    evening = [t for t in trades if t.sub_window == "EVENING"]
    eve_w = [t for t in evening if t.pnl_dollars > 0]
    eve_l = [t for t in evening if t.pnl_dollars <= 0]
    p(f"\nEvening trades: {len(evening)}, Winners: {len(eve_w)}, Losers: {len(eve_l)}")
    p(f"Total PnL: ${sum(t.pnl_dollars for t in evening):+,.0f}")
    p(f"Avg R: {np.mean([t.r_multiple for t in evening]):+.3f}")
    p(f"Edge ratio: {np.mean([t.mfe_r for t in evening]) / max(0.001, np.mean([t.mae_r for t in evening])):.2f}")
    p()

    # Group by exit reason
    eve_by_exit = defaultdict(list)
    for t in evening:
        eve_by_exit[t.exit_reason].append(t)
    p("  Evening by exit reason:")
    for reason, group in sorted(eve_by_exit.items(), key=lambda x: sum(t.pnl_dollars for t in x[1])):
        pnl = sum(t.pnl_dollars for t in group)
        avg_r = np.mean([t.r_multiple for t in group])
        wr = sum(1 for t in group if t.pnl_dollars > 0) / len(group)
        p(f"    {reason:12s}  N={len(group):2d}  WR={wr:4.0%}  AvgR={avg_r:+6.2f}  PnL=${pnl:+8,.0f}")

    # Group by entry hour
    eve_by_hour = defaultdict(list)
    for t in evening:
        et = _to_et(t.entry_time) if t.entry_time else None
        hour = et.hour if et else -1
        eve_by_hour[hour].append(t)
    p("\n  Evening by entry hour (ET):")
    for hour, group in sorted(eve_by_hour.items()):
        pnl = sum(t.pnl_dollars for t in group)
        avg_r = np.mean([t.r_multiple for t in group])
        wr = sum(1 for t in group if t.pnl_dollars > 0) / len(group)
        avg_mfe = np.mean([t.mfe_r for t in group])
        avg_mae = np.mean([t.mae_r for t in group])
        p(f"    {hour:02d}:00  N={len(group):2d}  WR={wr:4.0%}  AvgR={avg_r:+6.2f}  "
          f"MFE={avg_mfe:+5.2f}  MAE={avg_mae:+5.2f}  PnL=${pnl:+8,.0f}")

    # Group by day of week
    eve_by_day = defaultdict(list)
    for t in evening:
        et = _to_et(t.entry_time) if t.entry_time else None
        day = et.strftime("%a") if et else "?"
        eve_by_day[day].append(t)
    p("\n  Evening by day of week:")
    for day, group in sorted(eve_by_day.items()):
        pnl = sum(t.pnl_dollars for t in group)
        avg_r = np.mean([t.r_multiple for t in group])
        wr = sum(1 for t in group if t.pnl_dollars > 0) / len(group)
        p(f"    {day:3s}  N={len(group):2d}  WR={wr:4.0%}  AvgR={avg_r:+6.2f}  PnL=${pnl:+8,.0f}")

    p("\n  All evening trades (chronological):")
    for i, t in enumerate(evening):
        p(trade_line(t, i))

    # ─── DRAG 3: VWAP_FAIL EXIT (all sessions) ────────────────────────
    p("\n" + "=" * 90)
    p("  DRAG 3: VWAP_FAIL EXITS (8 trades, -0.977R avg)")
    p("=" * 90)
    vwap_fail = [t for t in trades if t.exit_reason == "VWAP_FAIL"]
    p(f"\nVWAP_FAIL trades: {len(vwap_fail)}")
    p(f"Total PnL: ${sum(t.pnl_dollars for t in vwap_fail):+,.0f}")
    p(f"Avg R: {np.mean([t.r_multiple for t in vwap_fail]):+.3f}")
    p(f"Avg bars held: {np.mean([t.bars_held_15m for t in vwap_fail]):.1f}")
    p(f"Avg MFE: {np.mean([t.mfe_r for t in vwap_fail]):+.3f}R")

    vf_by_sw = defaultdict(list)
    for t in vwap_fail:
        vf_by_sw[t.sub_window].append(t)
    p("\n  VWAP_FAIL by sub-window:")
    for sw, group in sorted(vf_by_sw.items(), key=lambda x: sum(t.pnl_dollars for t in x[1])):
        pnl = sum(t.pnl_dollars for t in group)
        avg_r = np.mean([t.r_multiple for t in group])
        avg_bars = np.mean([t.bars_held_15m for t in group])
        avg_mfe = np.mean([t.mfe_r for t in group])
        p(f"    {sw:7s}  N={len(group):2d}  AvgR={avg_r:+6.2f}  AvgBars={avg_bars:4.1f}  "
          f"MFE={avg_mfe:+5.2f}  PnL=${pnl:+8,.0f}")

    p("\n  All VWAP_FAIL trades:")
    for i, t in enumerate(vwap_fail):
        p(trade_line(t, i))

    # ─── DRAG 4: EARLY_KILL ANALYSIS ──────────────────────────────────
    p("\n" + "=" * 90)
    p("  DRAG 4: EARLY_KILL EFFECTIVENESS")
    p("=" * 90)
    early_kills = [t for t in trades if t.exit_reason == "EARLY_KILL"]
    p(f"\nEarly kills: {len(early_kills)}")
    p(f"Total PnL: ${sum(t.pnl_dollars for t in early_kills):+,.0f}")
    p(f"Avg R: {np.mean([t.r_multiple for t in early_kills]):+.3f}")

    p("\n  All early kills:")
    for i, t in enumerate(early_kills):
        p(trade_line(t, i))

    # Compare: fast deaths NOT caught by early kill
    fast_deaths_not_ek = [t for t in trades if t.bars_held_15m <= 4
                          and t.pnl_dollars <= 0 and t.exit_reason != "EARLY_KILL"]
    p(f"\n  Fast deaths (1-4 bars) NOT caught by early kill: {len(fast_deaths_not_ek)}")
    if fast_deaths_not_ek:
        p(f"  Total PnL: ${sum(t.pnl_dollars for t in fast_deaths_not_ek):+,.0f}")
        p(f"  Avg R: {np.mean([t.r_multiple for t in fast_deaths_not_ek]):+.3f}")
        p(f"  Exit reasons: {', '.join(t.exit_reason for t in fast_deaths_not_ek)}")
        for i, t in enumerate(fast_deaths_not_ek):
            p(trade_line(t, i))
        p(f"\n  Why early kill missed these:")
        p(f"    Early kill triggers when: bars<=3, unrealR<-0.30, peakMFE<0.30")
        for t in fast_deaths_not_ek:
            if t.bars_held_15m > 3:
                p(f"    {fmt_dt(t.entry_time)}: bars={t.bars_held_15m} > 3 (died on bar 4)")
            elif t.mfe_r >= 0.30:
                p(f"    {fmt_dt(t.entry_time)}: MFE={t.mfe_r:+.2f}R >= 0.30 (showed promise before dying)")
            elif t.r_multiple >= -0.30:
                p(f"    {fmt_dt(t.entry_time)}: R={t.r_multiple:+.2f} >= -0.30 (not deep enough)")
            else:
                p(f"    {fmt_dt(t.entry_time)}: bars={t.bars_held_15m}, R={t.r_multiple:+.2f}, MFE={t.mfe_r:+.2f} (unknown)")

    # ─── DRAG 5: SLOW DEATHS (5+ bars, reached profit then failed) ───
    p("\n" + "=" * 90)
    p("  DRAG 5: SLOW DEATHS (5+ bars losers, 80% reached 0.5R+ MFE)")
    p("=" * 90)
    slow_deaths = [t for t in trades if t.bars_held_15m >= 5 and t.pnl_dollars <= 0]
    p(f"\nSlow deaths: {len(slow_deaths)}")
    p(f"Total PnL: ${sum(t.pnl_dollars for t in slow_deaths):+,.0f}")
    p(f"Avg R: {np.mean([t.r_multiple for t in slow_deaths]):+.3f}")
    p(f"Avg MFE: {np.mean([t.mfe_r for t in slow_deaths]):+.3f}R")
    reached_profit = [t for t in slow_deaths if t.mfe_r >= 0.30]
    p(f"Reached 0.30R+ MFE: {len(reached_profit)}/{len(slow_deaths)}")
    reached_half = [t for t in slow_deaths if t.mfe_r >= 0.50]
    p(f"Reached 0.50R+ MFE: {len(reached_half)}/{len(slow_deaths)}")

    sd_by_exit = defaultdict(list)
    for t in slow_deaths:
        sd_by_exit[t.exit_reason].append(t)
    p("\n  Slow deaths by exit reason:")
    for reason, group in sorted(sd_by_exit.items(), key=lambda x: sum(t.pnl_dollars for t in x[1])):
        pnl = sum(t.pnl_dollars for t in group)
        avg_r = np.mean([t.r_multiple for t in group])
        avg_mfe = np.mean([t.mfe_r for t in group])
        p(f"    {reason:12s}  N={len(group):2d}  AvgR={avg_r:+6.2f}  MFE={avg_mfe:+5.2f}  PnL=${pnl:+8,.0f}")

    sd_by_sw = defaultdict(list)
    for t in slow_deaths:
        sd_by_sw[t.sub_window].append(t)
    p("\n  Slow deaths by sub-window:")
    for sw, group in sorted(sd_by_sw.items(), key=lambda x: sum(t.pnl_dollars for t in x[1])):
        pnl = sum(t.pnl_dollars for t in group)
        avg_r = np.mean([t.r_multiple for t in group])
        p(f"    {sw:7s}  N={len(group):2d}  AvgR={avg_r:+6.2f}  PnL=${pnl:+8,.0f}")

    p("\n  All slow deaths:")
    for i, t in enumerate(slow_deaths):
        p(trade_line(t, i))

    # ─── DRAG 6: DECEMBER 2025 ─────────────────────────────────────────
    p("\n" + "=" * 90)
    p("  DRAG 6: DECEMBER 2025 (11 trades, 27% WR, -$6,960)")
    p("=" * 90)
    dec_trades = [t for t in trades if t.entry_time and _to_et(t.entry_time).month == 12
                  and _to_et(t.entry_time).year == 2025]
    if dec_trades:
        p(f"\nDecember trades: {len(dec_trades)}")
        p(f"Total PnL: ${sum(t.pnl_dollars for t in dec_trades):+,.0f}")
        p(f"WR: {sum(1 for t in dec_trades if t.pnl_dollars > 0)/len(dec_trades):.0%}")
        p(f"Avg R: {np.mean([t.r_multiple for t in dec_trades]):+.3f}")

        dec_by_sw = defaultdict(list)
        for t in dec_trades:
            dec_by_sw[t.sub_window].append(t)
        p("\n  December by sub-window:")
        for sw, group in sorted(dec_by_sw.items(), key=lambda x: sum(t.pnl_dollars for t in x[1])):
            pnl = sum(t.pnl_dollars for t in group)
            avg_r = np.mean([t.r_multiple for t in group])
            wr = sum(1 for t in group if t.pnl_dollars > 0) / len(group)
            p(f"    {sw:7s}  N={len(group):2d}  WR={wr:4.0%}  AvgR={avg_r:+6.2f}  PnL=${pnl:+8,.0f}")

        dec_by_exit = defaultdict(list)
        for t in dec_trades:
            dec_by_exit[t.exit_reason].append(t)
        p("\n  December by exit reason:")
        for reason, group in sorted(dec_by_exit.items(), key=lambda x: sum(t.pnl_dollars for t in x[1])):
            pnl = sum(t.pnl_dollars for t in group)
            avg_r = np.mean([t.r_multiple for t in group])
            p(f"    {reason:12s}  N={len(group):2d}  AvgR={avg_r:+6.2f}  PnL=${pnl:+8,.0f}")

        p("\n  All December trades:")
        for i, t in enumerate(dec_trades):
            p(trade_line(t, i))

    # ─── DRAG 7: 0.4-0.6R VWAP DISTANCE BUCKET ──────────────────────
    p("\n" + "=" * 90)
    p("  DRAG 7: VWAP DISTANCE 0.4-0.6R BUCKET (EdgeR=1.02)")
    p("=" * 90)
    # Compute VWAP distance in R for each trade
    vwap_04_06 = []
    for t in trades:
        if t.vwap_used_at_entry != 0 and abs(t.entry_price - t.initial_stop) > 0:
            r_pts = abs(t.entry_price - t.initial_stop)
            vwap_dist_r = abs(t.entry_price - t.vwap_used_at_entry) / r_pts
            if 0.4 <= vwap_dist_r < 0.6:
                vwap_04_06.append((t, vwap_dist_r))
    p(f"\nTrades in 0.4-0.6R VWAP distance: {len(vwap_04_06)}")
    if vwap_04_06:
        ts = [x[0] for x in vwap_04_06]
        p(f"Total PnL: ${sum(t.pnl_dollars for t in ts):+,.0f}")
        p(f"Avg R: {np.mean([t.r_multiple for t in ts]):+.3f}")
        p(f"WR: {sum(1 for t in ts if t.pnl_dollars > 0)/len(ts):.0%}")
        p(f"\nNote: VWAP_CAP_CORE=0.40 caps distance in ATR units, not R units.")
        p(f"These trades passed the ATR cap but have 0.4-0.6R distance because")
        p(f"R (stop distance) differs from ATR.")

        p("\n  By sub-window:")
        sw_groups = defaultdict(list)
        for t, d in vwap_04_06:
            sw_groups[t.sub_window].append(t)
        for sw, group in sorted(sw_groups.items()):
            pnl = sum(t.pnl_dollars for t in group)
            avg_r = np.mean([t.r_multiple for t in group])
            p(f"    {sw:7s}  N={len(group):2d}  AvgR={avg_r:+6.2f}  PnL=${pnl:+8,.0f}")

        p("\n  All 0.4-0.6R VWAP trades:")
        for i, (t, d) in enumerate(vwap_04_06):
            p(f"  [{i:2d}] {trade_line(t)[4:]}  VWAP_dist_R={d:.3f}")

    # ─── DRAG 8: TUESDAY + THURSDAY + FRIDAY ────────────────────────
    p("\n" + "=" * 90)
    p("  DRAG 8: WEAK DAYS (Tue -$6,541, Thu -$3,233, Fri -$3,428)")
    p("=" * 90)
    for day_name in ["Tuesday", "Thursday", "Friday"]:
        day_trades = [t for t in trades if t.entry_time and
                      _to_et(t.entry_time).strftime("%A") == day_name]
        if not day_trades:
            continue
        pnl = sum(t.pnl_dollars for t in day_trades)
        wr = sum(1 for t in day_trades if t.pnl_dollars > 0) / len(day_trades)
        avg_r = np.mean([t.r_multiple for t in day_trades])
        p(f"\n  {day_name}: {len(day_trades)} trades, WR={wr:.0%}, AvgR={avg_r:+.3f}, PnL=${pnl:+,.0f}")

        day_by_sw = defaultdict(list)
        for t in day_trades:
            day_by_sw[t.sub_window].append(t)
        for sw, group in sorted(day_by_sw.items(), key=lambda x: sum(t.pnl_dollars for t in x[1])):
            gpnl = sum(t.pnl_dollars for t in group)
            gavg = np.mean([t.r_multiple for t in group])
            gwr = sum(1 for t in group if t.pnl_dollars > 0) / len(group)
            p(f"      {sw:7s}  N={len(group):2d}  WR={gwr:4.0%}  AvgR={gavg:+6.2f}  PnL=${gpnl:+8,.0f}")

    # ─── DRAG 9: 10:00 HOUR (24 trades, -$1,248 despite 58% WR) ─────
    p("\n" + "=" * 90)
    p("  DRAG 9: 10:00 HOUR — 58% WR BUT STILL LOSING (-$1,248)")
    p("=" * 90)
    h10 = [t for t in trades if t.entry_time and _to_et(t.entry_time).hour == 10]
    if h10:
        h10_w = [t for t in h10 if t.pnl_dollars > 0]
        h10_l = [t for t in h10 if t.pnl_dollars <= 0]
        p(f"\n  10:00 trades: {len(h10)}")
        p(f"  Winners: {len(h10_w)}, avg R={np.mean([t.r_multiple for t in h10_w]):+.3f}, avg PnL=${np.mean([t.pnl_dollars for t in h10_w]):+,.0f}")
        p(f"  Losers:  {len(h10_l)}, avg R={np.mean([t.r_multiple for t in h10_l]):+.3f}, avg PnL=${np.mean([t.pnl_dollars for t in h10_l]):+,.0f}")
        p(f"  Winner avg MFE: {np.mean([t.mfe_r for t in h10_w]):+.3f}R, capture: {np.mean([t.r_multiple / max(0.01, t.mfe_r) for t in h10_w]):.2f}")
        p(f"  Loser avg MFE: {np.mean([t.mfe_r for t in h10_l]):+.3f}R")
        p(f"\n  Asymmetry check:")
        p(f"    Avg winner: ${np.mean([t.pnl_dollars for t in h10_w]):+,.0f}")
        p(f"    Avg loser:  ${np.mean([t.pnl_dollars for t in h10_l]):+,.0f}")
        p(f"    Win/loss ratio: {abs(np.mean([t.pnl_dollars for t in h10_w])) / max(1, abs(np.mean([t.pnl_dollars for t in h10_l]))):.2f}")

    # ─── SUMMARY: P&L DECOMPOSITION ─────────────────────────────────
    p("\n" + "=" * 90)
    p("  P&L DECOMPOSITION — WHERE DOES EACH DOLLAR COME FROM?")
    p("=" * 90)

    p("\n  By sub-window:")
    for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]:
        group = [t for t in trades if t.sub_window == sw]
        if group:
            pnl = sum(t.pnl_dollars for t in group)
            p(f"    {sw:7s}  N={len(group):2d}  PnL=${pnl:+8,.0f}  ({pnl/abs(total_pnl)*100 if total_pnl else 0:+5.1f}% of total loss)")

    p("\n  By exit reason:")
    for reason in ["EARLY_KILL", "VWAP_FAIL", "STOP", "VWAP_A_FAIL", "STALE", "MAX_DURATION"]:
        group = [t for t in trades if t.exit_reason == reason]
        if group:
            pnl = sum(t.pnl_dollars for t in group)
            p(f"    {reason:12s}  N={len(group):2d}  PnL=${pnl:+8,.0f}")

    p("\n  By entry type:")
    for etype in ["A", "B"]:
        group = [t for t in trades if t.entry_type == etype]
        if group:
            pnl = sum(t.pnl_dollars for t in group)
            p(f"    Type {etype:1s}  N={len(group):2d}  PnL=${pnl:+8,.0f}")

    # ─── ACTIONABLE SUMMARY ─────────────────────────────────────────
    p("\n" + "=" * 90)
    p("  ACTIONABLE FINDINGS")
    p("=" * 90)

    type_b_pnl = sum(t.pnl_dollars for t in type_b)
    eve_vwap_fail = [t for t in trades if t.sub_window == "EVENING" and t.exit_reason == "VWAP_FAIL"]
    eve_vf_pnl = sum(t.pnl_dollars for t in eve_vwap_fail)
    ek_pnl = sum(t.pnl_dollars for t in early_kills)

    p(f"""
  1. TYPE B LEAKAGE: {len(type_b)} trades, ${type_b_pnl:+,.0f}
     Root cause: backtest uses flags.type_b_enabled (True) not C.USE_TYPE_B (False)
     Fix: Set type_b_enabled=False in runner script flags

  2. EVENING VWAP_FAIL: {len(eve_vwap_fail)} trades, ${eve_vf_pnl:+,.0f}
     This is the SINGLE BIGGEST DRAG. Evening VWAP reclaim fails
     because session VWAP is stale/unreliable after RTH close.
     These trades enter near VWAP but VWAP drifts, triggering failure exit.

  3. EVENING SESSION OVERALL: {len(evening)} trades, ${sum(t.pnl_dollars for t in evening):+,.0f}
     Winners (+${sum(t.pnl_dollars for t in eve_w):+,.0f}) don't compensate for
     losers (${sum(t.pnl_dollars for t in eve_l):+,.0f}). Edge ratio only 1.12.

  4. EARLY KILL: {len(early_kills)} trades caught at avg {np.mean([t.r_multiple for t in early_kills]):+.3f}R
     {len(fast_deaths_not_ek)} additional fast deaths escaped (bars=4, or MFE>=0.30)
     The feature is working but could be tuned wider.

  5. SLOW DEATHS: {len(slow_deaths)} trades that reached profit then failed
     {len(reached_half)}/{len(slow_deaths)} reached 0.50R+ MFE before dying.
     Exit timing / trailing stop issue for pre-partial positions.
""")

    report = "\n".join(out)
    output_path = Path("backtest/output/drag_diagnosis.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
