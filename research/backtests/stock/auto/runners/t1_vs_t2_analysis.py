"""Deep T1 vs T2v3 side-by-side comparison with full diagnostic coverage.

Sections:
  1. Headline comparison (T1 daily vs T1 FSM-only vs T2v3)
  2. Entry trigger breakdown (FSM_ENTRY vs FSM_IMPROVED vs PM_REENTRY)
  3. Entry price comparison on overlapping sym-dates
  4. Stop distance comparison (structural vs trailed)
  5. MFE analysis (max favorable excursion before exit)
  6. Exit reason breakdown (per-exit-type WR, PnL, count)
  7. Carry analysis (carry rate, carry PnL, flow reversal PnL, days distribution)
  8. Hold duration distribution (same-day vs overnight vs multi-day)
  9. Symbol-date overlap (shared, unique, leakage)
 10. Monthly equity divergence
 11. Risk-normalized comparison

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    python -u -m research.backtests.stock.auto.output.t1_vs_t2_analysis
"""
from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from research.backtests.stock.auto.config_mutator import mutate_iaric_config
from research.backtests.stock.auto.scoring import composite_score, compute_r_multiples, extract_metrics
from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine
from research.backtests.stock.engine.iaric_engine import IARICIntradayEngine
from research.backtests.stock.engine.iaric_intraday_engine_v2 import IARICIntradayEngineV2
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.models import TradeRecord

DATA_DIR = Path("research/backtests/stock/data/raw")
OUTPUT_DIR = Path("research/backtests/stock/auto/output")
START = "2024-03-22"
END = "2026-03-01"
EQ = 10_000.0


def _sym_date_key(t: TradeRecord) -> str:
    d = t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time
    return f"{t.symbol}_{d}"


def _wr(trades: list[TradeRecord]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl_net > 0) / len(trades) * 100


def _total_pnl(trades: list[TradeRecord]) -> float:
    return sum(t.pnl_net for t in trades)


def _avg_r(trades: list[TradeRecord]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t.r_multiple for t in trades]))


def _print_metric_row(label: str, v1, v2, v3=None, fmt="s"):
    if v3 is not None:
        if fmt == "pct":
            print(f"  {label:24} {v1:>11.1f}% {v2:>11.1f}% {v3:>11.1f}%")
        elif fmt == "dollar":
            print(f"  {label:24} ${v1:>10.2f} ${v2:>10.2f} ${v3:>10.2f}")
        elif fmt == "int":
            print(f"  {label:24} {v1:>12} {v2:>12} {v3:>12}")
        elif fmt == "float":
            print(f"  {label:24} {v1:>12.3f} {v2:>12.3f} {v3:>12.3f}")
        else:
            print(f"  {label:24} {v1:>12} {v2:>12} {v3:>12}")
    else:
        if fmt == "pct":
            print(f"  {label:24} {v1:>11.1f}% {v2:>11.1f}%")
        elif fmt == "dollar":
            print(f"  {label:24} ${v1:>10.2f} ${v2:>10.2f}")
        elif fmt == "int":
            print(f"  {label:24} {v1:>12} {v2:>12}")
        else:
            print(f"  {label:24} {v1:>12} {v2:>12}")


def _group_by_exit(trades: list[TradeRecord]) -> dict[str, list[TradeRecord]]:
    groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        groups[t.exit_reason].append(t)
    return dict(groups)


def _group_by_entry(trades: list[TradeRecord]) -> dict[str, list[TradeRecord]]:
    groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        trigger = t.metadata.get("entry_trigger", t.entry_type) if t.metadata else t.entry_type
        groups[trigger].append(t)
    return dict(groups)


def _monthly_equity(trades: list[TradeRecord], initial: float) -> dict[str, float]:
    """Return cumulative equity by YYYY-MM."""
    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        key = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        monthly[key] += t.pnl_net
    cum = initial
    result = {}
    for k in sorted(monthly.keys()):
        cum += monthly[k]
        result[k] = cum
    return result


def main():
    print("Loading data...", flush=True)
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()

    # ── Run T1 daily optimal ──
    print("Running T1 Daily...", flush=True)
    t1_muts_path = OUTPUT_DIR / "greedy_optimal_iaric_t1.json"
    t1_muts = {}
    if t1_muts_path.exists():
        t1_muts = json.loads(t1_muts_path.read_text())["final_mutations"]
    t1_cfg = mutate_iaric_config(
        IARICBacktestConfig(start_date=START, end_date=END, initial_equity=EQ,
                            tier=1, data_dir=DATA_DIR),
        t1_muts,
    )
    t1_engine = IARICDailyEngine(t1_cfg, replay)
    t1_result = t1_engine.run()
    t1_m = extract_metrics(t1_result.trades, t1_result.equity_curve, t1_result.timestamps, EQ)

    # ── Run T1 FSM-only (tier=2 engine, no v3 flags) ──
    print("Running T1 FSM...", flush=True)
    fsm_cfg = mutate_iaric_config(
        IARICBacktestConfig(start_date=START, end_date=END, initial_equity=EQ,
                            tier=2, data_dir=DATA_DIR),
        t1_muts,
    )
    fsm_engine = IARICIntradayEngine(fsm_cfg, replay)
    fsm_result = fsm_engine.run()
    fsm_m = extract_metrics(fsm_result.trades, fsm_result.equity_curve, fsm_result.timestamps, EQ)

    # ── Run T2v3 optimal ──
    print("Running T2v3...", flush=True)
    v3_muts_path = OUTPUT_DIR / "greedy_optimal_iaric_t2_v3.json"
    v3_muts = {}
    if v3_muts_path.exists():
        v3_muts = json.loads(v3_muts_path.read_text())["final_mutations"]
    else:
        fallback = OUTPUT_DIR / "greedy_optimal_iaric_t2.json"
        if fallback.exists():
            v3_muts = json.loads(fallback.read_text())["final_mutations"]
    v3_cfg = mutate_iaric_config(
        IARICBacktestConfig(start_date=START, end_date=END, initial_equity=EQ,
                            tier=2, data_dir=DATA_DIR),
        v3_muts,
    )
    v3_engine = IARICIntradayEngineV2(v3_cfg, replay)
    v3_result = v3_engine.run()
    v3_m = extract_metrics(v3_result.trades, v3_result.equity_curve, v3_result.timestamps, EQ)

    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("T1 DAILY vs T1 FSM vs T2v3 — FULL DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    # ── 1. Headline Comparison ──
    print(f"\n{'':26} {'T1 Daily':>12} {'T1 FSM':>12} {'T2v3':>12}")
    print("-" * 64)
    _print_metric_row("Trades", t1_m.total_trades, fsm_m.total_trades, v3_m.total_trades, "int")
    _print_metric_row("Win Rate", t1_m.win_rate * 100, fsm_m.win_rate * 100, v3_m.win_rate * 100, "pct")
    _print_metric_row("Net Profit", t1_m.net_profit, fsm_m.net_profit, v3_m.net_profit, "dollar")
    _print_metric_row("Profit Factor", t1_m.profit_factor, fsm_m.profit_factor, v3_m.profit_factor, "float")
    _print_metric_row("Max DD", t1_m.max_drawdown_pct * 100, fsm_m.max_drawdown_pct * 100,
                       v3_m.max_drawdown_pct * 100, "pct")
    _print_metric_row("Sharpe", t1_m.sharpe, fsm_m.sharpe, v3_m.sharpe, "float")
    _print_metric_row("CAGR", t1_m.cagr * 100, fsm_m.cagr * 100, v3_m.cagr * 100, "pct")

    # ── 2. Entry Trigger Breakdown (v3 only) ──
    print("\n\n── 2. ENTRY TRIGGER BREAKDOWN (T2v3) ──")
    v3_by_entry = _group_by_entry(v3_result.trades)
    print(f"  {'Trigger':20} {'Count':>6} {'WR':>7} {'PnL':>10} {'Avg R':>7}")
    for trigger in sorted(v3_by_entry.keys()):
        grp = v3_by_entry[trigger]
        print(f"  {trigger:20} {len(grp):>6} {_wr(grp):>6.1f}% ${_total_pnl(grp):>9.2f} {_avg_r(grp):>6.2f}")

    # ── 3. Entry Price Comparison (overlapping sym-dates) ──
    print("\n\n── 3. ENTRY PRICE COMPARISON (overlapping sym-dates) ──")
    t1_keys = {_sym_date_key(t): t for t in t1_result.trades}
    v3_keys = {_sym_date_key(t): t for t in v3_result.trades}
    overlap = set(t1_keys.keys()) & set(v3_keys.keys())
    if overlap:
        price_diffs = []
        for k in overlap:
            t1_t = t1_keys[k]
            v3_t = v3_keys[k]
            diff_pct = (v3_t.entry_price - t1_t.entry_price) / t1_t.entry_price * 100
            price_diffs.append(diff_pct)
        print(f"  Overlapping sym-dates: {len(overlap)}")
        print(f"  T2v3 entry price vs T1: {np.mean(price_diffs):+.3f}% avg  "
              f"({np.median(price_diffs):+.3f}% median)")
        better = sum(1 for d in price_diffs if d < 0)
        print(f"  Better price (lower): {better}/{len(overlap)} ({better / len(overlap) * 100:.0f}%)")
    else:
        print("  No overlapping sym-dates found")

    # ── 4. Stop Distance Comparison ──
    print("\n\n── 4. STOP DISTANCE COMPARISON ──")
    for label, trd in [("T1 Daily", t1_result.trades), ("T1 FSM", fsm_result.trades),
                        ("T2v3", v3_result.trades)]:
        risks = [t.risk_per_share / t.entry_price * 100 for t in trd if t.entry_price > 0]
        if risks:
            print(f"  {label:12} avg stop dist: {np.mean(risks):.2f}%  "
                  f"median: {np.median(risks):.2f}%  max: {np.max(risks):.2f}%")

    # ── 5. MFE Analysis ──
    print("\n\n── 5. MFE ANALYSIS (max favorable excursion) ──")
    for label, trd in [("T1 Daily", t1_result.trades), ("T2v3", v3_result.trades)]:
        mfe_rs = [t.metadata.get("mfe_r", 0) for t in trd if t.metadata]
        if mfe_rs:
            avg_mfe = float(np.mean(mfe_rs))
            avg_realized = _avg_r(trd)
            capture = avg_realized / max(avg_mfe, 0.01) * 100
            print(f"  {label:12} avg MFE: {avg_mfe:.2f}R  "
                  f"median: {float(np.median(mfe_rs)):.2f}R  "
                  f"captured: {avg_realized:.2f}R ({capture:.0f}%)")

    # ── 6. Exit Reason Breakdown ──
    print("\n\n── 6. EXIT REASON BREAKDOWN ──")
    for label, trd in [("T1 Daily", t1_result.trades), ("T2v3", v3_result.trades)]:
        print(f"\n  {label}:")
        by_exit = _group_by_exit(trd)
        print(f"    {'Reason':24} {'Count':>6} {'WR':>7} {'PnL':>10} {'Avg R':>7}")
        for reason in sorted(by_exit.keys()):
            grp = by_exit[reason]
            print(f"    {reason:24} {len(grp):>6} {_wr(grp):>6.1f}% ${_total_pnl(grp):>9.2f} {_avg_r(grp):>6.2f}")

    # ── 7. Carry Analysis ──
    print("\n\n── 7. CARRY ANALYSIS ──")
    v3_carry = [t for t in v3_result.trades if t.metadata and t.metadata.get("carry_days", 0) > 0]
    v3_same_day = [t for t in v3_result.trades if not (t.metadata and t.metadata.get("carry_days", 0) > 0)]
    v3_flow_rev = [t for t in v3_result.trades if t.exit_reason == "FLOW_REVERSAL"]
    print(f"  Total v3 trades: {len(v3_result.trades)}")
    print(f"  Same-day:  {len(v3_same_day):>4}  WR={_wr(v3_same_day):.1f}%  PnL=${_total_pnl(v3_same_day):.2f}")
    print(f"  Carried:   {len(v3_carry):>4}  WR={_wr(v3_carry):.1f}%  PnL=${_total_pnl(v3_carry):.2f}")
    print(f"  Flow Rev:  {len(v3_flow_rev):>4}  PnL=${_total_pnl(v3_flow_rev):.2f}")
    if v3_carry:
        carry_days = [t.metadata.get("carry_days", 0) for t in v3_carry if t.metadata]
        if carry_days:
            print(f"  Carry days: avg={np.mean(carry_days):.1f}  max={max(carry_days)}")
        carry_pnl_pct = _total_pnl(v3_carry) / max(_total_pnl(v3_result.trades), 0.01) * 100
        print(f"  Carry PnL share: {carry_pnl_pct:.1f}% of total")

    # T1 carry for comparison
    t1_carry = [t for t in t1_result.trades if t.exit_reason in ("FLOW_REVERSAL", "CARRY_EXIT", "CARRY_MAX_DAYS")]
    t1_same = [t for t in t1_result.trades if t.exit_reason not in ("FLOW_REVERSAL", "CARRY_EXIT", "CARRY_MAX_DAYS")]
    print(f"\n  T1 Daily comparison:")
    print(f"  Same-day:  {len(t1_same):>4}  WR={_wr(t1_same):.1f}%  PnL=${_total_pnl(t1_same):.2f}")
    print(f"  Carried:   {len(t1_carry):>4}  WR={_wr(t1_carry):.1f}%  PnL=${_total_pnl(t1_carry):.2f}")

    # ── 8. Hold Duration Distribution ──
    print("\n\n── 8. HOLD DURATION DISTRIBUTION ──")
    for label, trd in [("T1 Daily", t1_result.trades), ("T2v3", v3_result.trades)]:
        same_day = [t for t in trd if t.hold_hours < 8]
        overnight = [t for t in trd if 8 <= t.hold_hours < 24]
        multi_day = [t for t in trd if t.hold_hours >= 24]
        print(f"  {label:12} same-day={len(same_day)}  overnight={len(overnight)}  multi-day={len(multi_day)}")

    # ── 9. Symbol-Date Overlap ──
    print("\n\n── 9. SYMBOL-DATE OVERLAP ──")
    t1_sd = set(t1_keys.keys())
    v3_sd = set(v3_keys.keys())
    shared = t1_sd & v3_sd
    t1_only = t1_sd - v3_sd
    v3_only = v3_sd - t1_sd
    print(f"  Shared:    {len(shared):>4}")
    print(f"  T1 only:   {len(t1_only):>4}  (T1 trades not in v3 — potential leakage)")
    print(f"  v3 only:   {len(v3_only):>4}  (v3 unique — FSM timing advantage)")

    if t1_only:
        t1_only_trades = [t1_keys[k] for k in t1_only]
        print(f"  T1-only PnL: ${_total_pnl(t1_only_trades):.2f}  WR={_wr(t1_only_trades):.1f}%")
    if v3_only:
        v3_only_trades = [v3_keys[k] for k in v3_only]
        print(f"  v3-only PnL: ${_total_pnl(v3_only_trades):.2f}  WR={_wr(v3_only_trades):.1f}%")

    if shared:
        shared_t1_pnl = sum(t1_keys[k].pnl_net for k in shared)
        shared_v3_pnl = sum(v3_keys[k].pnl_net for k in shared)
        print(f"\n  Shared trades PnL:")
        print(f"    T1: ${shared_t1_pnl:.2f}   v3: ${shared_v3_pnl:.2f}   "
              f"delta: ${shared_v3_pnl - shared_t1_pnl:.2f}")

    # ── 10. Monthly Equity Divergence ──
    print("\n\n── 10. MONTHLY EQUITY DIVERGENCE ──")
    t1_monthly = _monthly_equity(t1_result.trades, EQ)
    v3_monthly = _monthly_equity(v3_result.trades, EQ)
    all_months = sorted(set(t1_monthly.keys()) | set(v3_monthly.keys()))
    print(f"  {'Month':>8}  {'T1 Equity':>12}  {'v3 Equity':>12}  {'Delta':>10}")
    for m in all_months:
        t1_eq = t1_monthly.get(m, EQ)
        v3_eq = v3_monthly.get(m, EQ)
        print(f"  {m:>8}  ${t1_eq:>11.2f}  ${v3_eq:>11.2f}  ${v3_eq - t1_eq:>+9.2f}")

    # ── 11. Risk-Normalized Comparison ──
    print("\n\n── 11. RISK-NORMALIZED COMPARISON ──")
    for label, trd in [("T1 Daily", t1_result.trades), ("T1 FSM", fsm_result.trades),
                        ("T2v3", v3_result.trades)]:
        risks = [t.risk_per_share * t.quantity for t in trd]
        sizes = [t.entry_price * t.quantity for t in trd]
        if risks and sizes:
            print(f"  {label:12} avg risk/trade: ${np.mean(risks):.2f}  "
                  f"avg position: ${np.mean(sizes):.0f}  "
                  f"PnL/risk: {_total_pnl(trd) / max(sum(risks), 1):.2f}x")

    # ── Scoring Summary ──
    print("\n\n── SCORING SUMMARY ──")
    for label, m, trd in [("T1 Daily", t1_m, t1_result.trades),
                            ("T1 FSM", fsm_m, fsm_result.trades),
                            ("T2v3", v3_m, v3_result.trades)]:
        r_mult = compute_r_multiples(trd)
        score = composite_score(m, EQ, r_multiples=r_mult)
        print(f"  {label:12} score={score.total:.4f}  "
              f"(net={score.net_profit_component:.4f} calmar={score.calmar_component:.4f} "
              f"edge={score.edge_tstat_component:.4f} wr={score.wr_component:.4f})")

    print("\n" + "=" * 70)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
