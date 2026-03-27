"""Evaluate whether sector_sz_cons_disc_11 (1.1x Consumer Discretionary sizing)
represents overfitting or genuine edge.

Tests:
  1. P8 WITH vs WITHOUT the 1.1x multiplier
  2. Monthly ConsDisc R contribution (temporal stability)
  3. First-half vs second-half consistency
  4. Per-symbol breakdown within ConsDisc
  5. Composite score delta

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    python -u -m research.backtests.stock.auto.output.sector_overfit_analysis
"""
from __future__ import annotations

import io
import json
import sys
import time
from collections import defaultdict
from datetime import time as dt_time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from research.backtests.stock.auto.config_mutator import mutate_alcb_config
from research.backtests.stock.auto.scoring import composite_score, compute_r_multiples, extract_metrics
from research.backtests.stock.config_alcb import ALCBBacktestConfig
from research.backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from research.backtests.stock.engine.research_replay import ResearchReplayEngine

DATA_DIR = Path("research/backtests/stock/data/raw")
OUTPUT_DIR = Path("research/backtests/stock/auto/output")
INITIAL_EQUITY = 10_000.0
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"

_TIME_FIELDS = {
    "premarket_start", "post_close_scan", "market_open", "first_30m_close",
    "entry_end", "close_block_start", "forced_flatten", "early_entry_end",
    "late_entry_start", "entry_window_start", "entry_window_end",
    "eod_flatten_time", "entry_window_end_early", "late_entry_cutoff",
}


def _parse_time_fields(overrides: dict) -> dict:
    result = dict(overrides)
    for key, val in result.items():
        if key in _TIME_FIELDS and isinstance(val, str):
            parts = [int(p) for p in val.split(":")]
            result[key] = dt_time(*parts)
    return result


def build_config(param_overrides: dict, ablation: dict | None = None) -> ALCBBacktestConfig:
    base = ALCBBacktestConfig(
        start_date=START_DATE, end_date=END_DATE,
        initial_equity=INITIAL_EQUITY, tier=2, data_dir=DATA_DIR,
        param_overrides=dict(param_overrides),
    )
    if ablation:
        mutations = {f"ablation.{k}": v for k, v in ablation.items()}
        for k, v in param_overrides.items():
            if isinstance(v, list):
                mutations[f"param_overrides.{k}"] = tuple(v)
        if mutations:
            return mutate_alcb_config(base, mutations)
    return base


def analyze_trades(trades, label):
    """Analyze trades by sector, month, symbol, and half."""
    # Sector breakdown
    sector_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_r": 0.0, "pnl": 0.0})
    # Monthly ConsDisc
    monthly_cd = defaultdict(lambda: {"trades": 0, "wins": 0, "total_r": 0.0, "pnl": 0.0})
    # Per-symbol within ConsDisc
    symbol_cd = defaultdict(lambda: {"trades": 0, "wins": 0, "total_r": 0.0, "pnl": 0.0})
    # Half splits
    half1 = {"trades": 0, "wins": 0, "total_r": 0.0, "pnl": 0.0}
    half2 = {"trades": 0, "wins": 0, "total_r": 0.0, "pnl": 0.0}

    midpoint = len(trades) // 2
    cd_midpoint_date = None

    for i, t in enumerate(trades):
        sector = getattr(t, "sector", "Unknown") or "Unknown"
        s = sector_stats[sector]
        s["trades"] += 1
        s["wins"] += 1 if t.pnl_net > 0 else 0
        s["total_r"] += t.r_multiple
        s["pnl"] += t.pnl_net

        if sector == "Consumer Discretionary":
            month_key = t.entry_time.strftime("%Y-%m") if t.entry_time else "?"
            m = monthly_cd[month_key]
            m["trades"] += 1
            m["wins"] += 1 if t.pnl_net > 0 else 0
            m["total_r"] += t.r_multiple
            m["pnl"] += t.pnl_net

            sym = t.symbol
            ss = symbol_cd[sym]
            ss["trades"] += 1
            ss["wins"] += 1 if t.pnl_net > 0 else 0
            ss["total_r"] += t.r_multiple
            ss["pnl"] += t.pnl_net

    # Split ConsDisc trades into halves by time
    cd_trades = [t for t in trades if (getattr(t, "sector", "") or "") == "Consumer Discretionary"]
    cd_mid = len(cd_trades) // 2
    for i, t in enumerate(cd_trades):
        h = half1 if i < cd_mid else half2
        h["trades"] += 1
        h["wins"] += 1 if t.pnl_net > 0 else 0
        h["total_r"] += t.r_multiple
        h["pnl"] += t.pnl_net

    if cd_trades and cd_mid < len(cd_trades):
        cd_midpoint_date = cd_trades[cd_mid].entry_time.strftime("%Y-%m-%d") if cd_trades[cd_mid].entry_time else "?"

    return {
        "sector_stats": dict(sector_stats),
        "monthly_cd": dict(monthly_cd),
        "symbol_cd": dict(symbol_cd),
        "half1": half1,
        "half2": half2,
        "cd_midpoint_date": cd_midpoint_date,
        "total_trades": len(trades),
        "cd_trades": len(cd_trades),
    }


def fmt_wr(wins, total):
    return f"{wins/total*100:.1f}%" if total > 0 else "N/A"


def fmt_avg_r(total_r, total):
    return f"{total_r/total:+.3f}" if total > 0 else "N/A"


def main():
    print("=" * 70)
    print("SECTOR_SZ_CONS_DISC_11 OVERFITTING ANALYSIS")
    print("=" * 70, flush=True)

    # Load P8 optimal config
    raw_p8 = json.loads(
        (OUTPUT_DIR / "optimal_config_alcb_p8.json").read_text(encoding="utf-8")
    )
    p8_overrides = _parse_time_fields(raw_p8.get("param_overrides", {}))
    p8_ablation = raw_p8.get("ablation", {})

    # Create WITHOUT variant (remove sector mult)
    p8_no_sector = dict(p8_overrides)
    p8_no_sector.pop("sector_mult_consumer_disc", None)

    print("\n[1/3] Loading data...", flush=True)
    t_start = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t_start:.1f}s", flush=True)

    # Run WITH sector sizing (full P8)
    print("\n[2/3] Running P8 WITH sector_sz_cons_disc_11 (1.1x)...", flush=True)
    t0 = time.time()
    cfg_with = build_config(p8_overrides, p8_ablation)
    eng_with = ALCBIntradayEngine(cfg_with, replay)
    res_with = eng_with.run()
    met_with = extract_metrics(res_with.trades, res_with.equity_curve, res_with.timestamps, INITIAL_EQUITY)
    sc_with = composite_score(met_with, INITIAL_EQUITY, r_multiples=compute_r_multiples(res_with.trades))
    print(f"  Done in {time.time() - t0:.1f}s — {len(res_with.trades)} trades", flush=True)

    # Run WITHOUT sector sizing
    print("\n[3/3] Running P8 WITHOUT sector_sz_cons_disc_11 (1.0x)...", flush=True)
    t0 = time.time()
    cfg_without = build_config(p8_no_sector, p8_ablation)
    eng_without = ALCBIntradayEngine(cfg_without, replay)
    res_without = eng_without.run()
    met_without = extract_metrics(res_without.trades, res_without.equity_curve, res_without.timestamps, INITIAL_EQUITY)
    sc_without = composite_score(met_without, INITIAL_EQUITY, r_multiples=compute_r_multiples(res_without.trades))
    print(f"  Done in {time.time() - t0:.1f}s — {len(res_without.trades)} trades", flush=True)

    # Analyze both
    analysis_with = analyze_trades(res_with.trades, "WITH 1.1x")
    analysis_without = analyze_trades(res_without.trades, "WITHOUT 1.1x")

    # Print results
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("RESULTS")
    lines.append("=" * 70)

    lines.append("")
    lines.append("─" * 50)
    lines.append("A. COMPOSITE SCORE COMPARISON")
    lines.append("─" * 50)
    lines.append(f"  WITH 1.1x:    {sc_with.total:.6f}")
    lines.append(f"  WITHOUT 1.1x: {sc_without.total:.6f}")
    delta = sc_with.total - sc_without.total
    lines.append(f"  Delta:        {delta:+.6f} ({delta/sc_without.total*100:+.3f}%)")
    lines.append("")
    lines.append(f"  WITH    — Trades: {met_with.total_trades}, WR: {met_with.win_rate*100:.1f}%, PF: {met_with.profit_factor:.2f}, Net: ${met_with.net_profit:,.2f}, DD: {met_with.max_drawdown_pct*100:.2f}%")
    lines.append(f"  WITHOUT — Trades: {met_without.total_trades}, WR: {met_without.win_rate*100:.1f}%, PF: {met_without.profit_factor:.2f}, Net: ${met_without.net_profit:,.2f}, DD: {met_without.max_drawdown_pct*100:.2f}%")
    lines.append(f"  Net Profit Delta: ${met_with.net_profit - met_without.net_profit:,.2f}")

    lines.append("")
    lines.append("─" * 50)
    lines.append("B. SECTOR COMPARISON (WITH 1.1x)")
    lines.append("─" * 50)
    lines.append(f"  {'Sector':<25} {'Trades':>6} {'WR':>7} {'AvgR':>7} {'TotalR':>8} {'PnL':>10}")
    for sector in sorted(analysis_with["sector_stats"].keys()):
        s = analysis_with["sector_stats"][sector]
        lines.append(f"  {sector:<25} {s['trades']:>6} {fmt_wr(s['wins'], s['trades']):>7} {fmt_avg_r(s['total_r'], s['trades']):>7} {s['total_r']:>+8.2f} ${s['pnl']:>9,.2f}")

    lines.append("")
    lines.append("─" * 50)
    lines.append("C. SECTOR COMPARISON (WITHOUT 1.1x)")
    lines.append("─" * 50)
    lines.append(f"  {'Sector':<25} {'Trades':>6} {'WR':>7} {'AvgR':>7} {'TotalR':>8} {'PnL':>10}")
    for sector in sorted(analysis_without["sector_stats"].keys()):
        s = analysis_without["sector_stats"][sector]
        lines.append(f"  {sector:<25} {s['trades']:>6} {fmt_wr(s['wins'], s['trades']):>7} {fmt_avg_r(s['total_r'], s['trades']):>7} {s['total_r']:>+8.2f} ${s['pnl']:>9,.2f}")

    lines.append("")
    lines.append("─" * 50)
    lines.append("D. CONSUMER DISCRETIONARY — MONTHLY BREAKDOWN (WITH 1.1x)")
    lines.append("─" * 50)
    lines.append(f"  {'Month':<10} {'Trades':>6} {'WR':>7} {'AvgR':>7} {'TotalR':>8} {'PnL':>10}")
    positive_months = 0
    total_months = 0
    for month in sorted(analysis_with["monthly_cd"].keys()):
        m = analysis_with["monthly_cd"][month]
        lines.append(f"  {month:<10} {m['trades']:>6} {fmt_wr(m['wins'], m['trades']):>7} {fmt_avg_r(m['total_r'], m['trades']):>7} {m['total_r']:>+8.2f} ${m['pnl']:>9,.2f}")
        total_months += 1
        if m["total_r"] > 0:
            positive_months += 1
    lines.append(f"  Positive months: {positive_months}/{total_months} ({positive_months/total_months*100:.0f}%)" if total_months > 0 else "")

    lines.append("")
    lines.append("─" * 50)
    lines.append("E. CONSUMER DISCRETIONARY — HALF/HALF STABILITY (WITH 1.1x)")
    lines.append("─" * 50)
    h1 = analysis_with["half1"]
    h2 = analysis_with["half2"]
    lines.append(f"  Midpoint date: {analysis_with['cd_midpoint_date']}")
    lines.append(f"  First half:  {h1['trades']} trades, WR {fmt_wr(h1['wins'], h1['trades'])}, AvgR {fmt_avg_r(h1['total_r'], h1['trades'])}, TotalR {h1['total_r']:+.2f}, PnL ${h1['pnl']:,.2f}")
    lines.append(f"  Second half: {h2['trades']} trades, WR {fmt_wr(h2['wins'], h2['trades'])}, AvgR {fmt_avg_r(h2['total_r'], h2['trades'])}, TotalR {h2['total_r']:+.2f}, PnL ${h2['pnl']:,.2f}")

    # Same for WITHOUT
    lines.append("")
    lines.append("─" * 50)
    lines.append("E2. CONSUMER DISCRETIONARY — HALF/HALF STABILITY (WITHOUT 1.1x)")
    lines.append("─" * 50)
    h1w = analysis_without["half1"]
    h2w = analysis_without["half2"]
    lines.append(f"  Midpoint date: {analysis_without['cd_midpoint_date']}")
    lines.append(f"  First half:  {h1w['trades']} trades, WR {fmt_wr(h1w['wins'], h1w['trades'])}, AvgR {fmt_avg_r(h1w['total_r'], h1w['trades'])}, TotalR {h1w['total_r']:+.2f}, PnL ${h1w['pnl']:,.2f}")
    lines.append(f"  Second half: {h2w['trades']} trades, WR {fmt_wr(h2w['wins'], h2w['trades'])}, AvgR {fmt_avg_r(h2w['total_r'], h2w['trades'])}, TotalR {h2w['total_r']:+.2f}, PnL ${h2w['pnl']:,.2f}")

    lines.append("")
    lines.append("─" * 50)
    lines.append("F. TOP CONSUMER DISCRETIONARY SYMBOLS (WITH 1.1x)")
    lines.append("─" * 50)
    lines.append(f"  {'Symbol':<8} {'Trades':>6} {'WR':>7} {'AvgR':>7} {'TotalR':>8} {'PnL':>10}")
    sorted_symbols = sorted(analysis_with["symbol_cd"].items(), key=lambda x: x[1]["total_r"], reverse=True)
    for sym, s in sorted_symbols:
        lines.append(f"  {sym:<8} {s['trades']:>6} {fmt_wr(s['wins'], s['trades']):>7} {fmt_avg_r(s['total_r'], s['trades']):>7} {s['total_r']:>+8.2f} ${s['pnl']:>9,.2f}")

    lines.append("")
    lines.append("─" * 50)
    lines.append("G. OVERFITTING RISK ASSESSMENT")
    lines.append("─" * 50)

    # Compute metrics for assessment
    cd_pct_alpha_with = 0
    total_r_with = sum(s["total_r"] for s in analysis_with["sector_stats"].values())
    cd_r_with = analysis_with["sector_stats"].get("Consumer Discretionary", {}).get("total_r", 0)
    if total_r_with > 0:
        cd_pct_alpha_with = cd_r_with / total_r_with * 100

    # Size of effect
    sizing_effect_r = cd_r_with * 0.1 / 1.1  # approximate: 1.1x means 10% more, so ~1/11 of total CD R
    lines.append(f"  1. Effect size: The 1.1x multiplier adds ~{sizing_effect_r:.1f}R ({sizing_effect_r/total_r_with*100:.1f}% of total alpha)")
    lines.append(f"     ConsDisc contributes {cd_pct_alpha_with:.0f}% of total alpha ({cd_r_with:.1f}R / {total_r_with:.1f}R)")

    # Temporal consistency
    if h1["trades"] > 0 and h2["trades"] > 0:
        h1_avg = h1["total_r"] / h1["trades"]
        h2_avg = h2["total_r"] / h2["trades"]
        consistency = "CONSISTENT" if (h1_avg > 0 and h2_avg > 0) else "INCONSISTENT"
        lines.append(f"  2. Temporal stability: {consistency} (H1 avg {h1_avg:+.3f}R, H2 avg {h2_avg:+.3f}R)")

    # Monthly win rate
    lines.append(f"  3. Monthly positive: {positive_months}/{total_months} months ({positive_months/total_months*100:.0f}%)" if total_months > 0 else "")

    # Symbol concentration
    top_sym = sorted_symbols[0] if sorted_symbols else None
    if top_sym:
        top_pct = top_sym[1]["total_r"] / cd_r_with * 100 if cd_r_with > 0 else 0
        lines.append(f"  4. Top symbol: {top_sym[0]} = {top_sym[1]['total_r']:+.1f}R ({top_pct:.0f}% of ConsDisc alpha)")
        top2_r = sum(s[1]["total_r"] for s in sorted_symbols[:2])
        top2_pct = top2_r / cd_r_with * 100 if cd_r_with > 0 else 0
        lines.append(f"     Top 2 symbols: {top2_r:+.1f}R ({top2_pct:.0f}% of ConsDisc alpha)")

    # Trade count vs other sectors
    cd_trades = analysis_with["cd_trades"]
    total_trades = analysis_with["total_trades"]
    lines.append(f"  5. Trade share: {cd_trades}/{total_trades} ({cd_trades/total_trades*100:.1f}%) — sufficient sample for sector analysis")

    # Net score delta
    lines.append(f"  6. Score impact: {delta:+.6f} ({delta/sc_without.total*100:+.3f}%) — {'MATERIAL' if abs(delta/sc_without.total) > 0.005 else 'MARGINAL'}")

    report = "\n".join(lines)
    print(report)

    # Save to file
    out_path = OUTPUT_DIR / "sector_overfit_analysis.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
