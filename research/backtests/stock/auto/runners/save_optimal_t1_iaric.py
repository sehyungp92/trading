"""Run IARIC T1 optimal config and save full diagnostics.

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    python -u -m research.backtests.stock.auto.output.save_optimal_t1_iaric
"""
from __future__ import annotations

import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from research.backtests.stock.auto.config_mutator import mutate_iaric_config
from research.backtests.stock.auto.scoring import (
    IARIC_NORM, composite_score, compute_r_multiples, extract_metrics,
)
from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.iaric_daily_engine import IARICDailyEngine
from research.backtests.stock.engine.research_replay import ResearchReplayEngine

DATA_DIR = Path("research/backtests/stock/data/raw")
OUTPUT_DIR = Path("research/backtests/stock/auto/output")
INITIAL_EQUITY = 10_000.0
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"

OPTIMAL_PATH = OUTPUT_DIR / "greedy_optimal_iaric_t1_p3.json"

# P1 safety params (match config defaults, kept for explicit documentation)
P1_SAFETY_PARAMS: dict = {
    "param_overrides.flow_reversal_lookback": 1,
    "param_overrides.regime_b_carry_mult": 0.6,
    "param_overrides.min_carry_r": 0.5,
}


def build_config(mutations: dict) -> IARICBacktestConfig:
    base = IARICBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_equity=INITIAL_EQUITY,
        tier=1,
        data_dir=DATA_DIR,
    )
    return mutate_iaric_config(base, mutations)


def fmt_metrics(m, score, label: str) -> str:
    lines = [
        f"  {'─' * 50}",
        f"  {label}",
        f"  {'─' * 50}",
        f"  Trades:          {m.total_trades}  (W:{m.winning_trades} / L:{m.losing_trades})",
        f"  Win Rate:        {m.win_rate * 100:.1f}%",
        f"  Gross Profit:    ${m.gross_profit:,.2f}",
        f"  Gross Loss:      ${m.gross_loss:,.2f}",
        f"  Net Profit:      ${m.net_profit:,.2f}",
        f"  Profit Factor:   {m.profit_factor:.2f}",
        f"  Expectancy (R):  {m.expectancy:.3f}",
        f"  Expectancy ($):  ${m.expectancy_dollar:.2f}",
        f"  CAGR:            {m.cagr * 100:.2f}%",
        f"  Sharpe:          {m.sharpe:.3f}",
        f"  Sortino:         {m.sortino:.3f}",
        f"  Calmar:          {m.calmar:.3f}",
        f"  Max DD (%):      {m.max_drawdown_pct * 100:.2f}%",
        f"  Max DD ($):      ${m.max_drawdown_dollar:,.2f}",
        f"  Avg Hold (hrs):  {m.avg_hold_hours:.1f}",
        f"  Trades/Month:    {m.trades_per_month:.2f}",
        f"  Commissions:     ${m.total_commissions:,.2f}",
        "",
        f"  Composite Score: {score.total:.6f}",
        f"    Calmar (25%):    {score.calmar_component:.6f}",
        f"    PF     (20%):    {score.pf_component:.6f}",
        f"    Inv DD (20%):    {score.inv_dd_component:.6f}",
        f"    Net P  (20%):    {score.net_profit_component:.6f}",
        f"    WR     (15%):    {score.wr_component:.6f}",
    ]
    if score.rejected:
        lines.append(f"    REJECTED:        {score.reject_reason}")
    return "\n".join(lines)


def exit_breakdown(trades) -> str:
    by_reason: dict[str, list] = defaultdict(list)
    for t in trades:
        by_reason[t.exit_reason].append(t)
    lines = []
    for reason in sorted(by_reason, key=lambda r: -len(by_reason[r])):
        tt = by_reason[reason]
        wins = sum(1 for t in tt if t.pnl_net > 0)
        pnl = sum(t.pnl_net for t in tt)
        avg_r = sum(t.r_multiple for t in tt) / len(tt) if tt else 0
        wr = wins / len(tt) * 100 if tt else 0
        lines.append(
            f"  {reason:<25} {len(tt):>4} trades  WR={wr:5.1f}%  "
            f"PnL=${pnl:>9,.2f}  AvgR={avg_r:>+6.3f}"
        )
    return "\n".join(lines)


def sector_breakdown(trades) -> str:
    by_sector: dict[str, list] = defaultdict(list)
    for t in trades:
        by_sector[t.sector].append(t)
    lines = []
    for sector in sorted(by_sector, key=lambda s: -sum(t.pnl_net for t in by_sector[s])):
        tt = by_sector[sector]
        wins = sum(1 for t in tt if t.pnl_net > 0)
        pnl = sum(t.pnl_net for t in tt)
        wr = wins / len(tt) * 100 if tt else 0
        lines.append(
            f"  {sector:<28} {len(tt):>3} trades  WR={wr:5.1f}%  PnL=${pnl:>9,.2f}"
        )
    return "\n".join(lines)


def regime_breakdown(trades) -> str:
    by_regime: dict[str, list] = defaultdict(list)
    for t in trades:
        by_regime[t.regime_tier].append(t)
    lines = []
    for tier in sorted(by_regime):
        tt = by_regime[tier]
        wins = sum(1 for t in tt if t.pnl_net > 0)
        pnl = sum(t.pnl_net for t in tt)
        avg_r = sum(t.r_multiple for t in tt) / len(tt) if tt else 0
        wr = wins / len(tt) * 100 if tt else 0
        lines.append(
            f"  Tier {tier:<4}  {len(tt):>4} trades  WR={wr:5.1f}%  "
            f"PnL=${pnl:>9,.2f}  AvgR={avg_r:>+6.3f}"
        )
    return "\n".join(lines)


def monthly_breakdown(trades) -> str:
    by_month: dict[str, list] = defaultdict(list)
    for t in trades:
        if t.entry_time:
            key = t.entry_time.strftime("%Y-%m")
            by_month[key].append(t)
    lines = []
    for month in sorted(by_month):
        tt = by_month[month]
        wins = sum(1 for t in tt if t.pnl_net > 0)
        pnl = sum(t.pnl_net for t in tt)
        wr = wins / len(tt) * 100 if tt else 0
        lines.append(
            f"  {month}  {len(tt):>3} trades  WR={wr:5.1f}%  PnL=${pnl:>9,.2f}"
        )
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("IARIC T1 Greedy-Optimal — Full Diagnostic Report")
    print("=" * 70, flush=True)

    # Load optimal config
    raw = json.loads(OPTIMAL_PATH.read_text(encoding="utf-8"))
    mutations = {**raw["final_mutations"], **P1_SAFETY_PARAMS}
    kept_features = raw["kept_features"] + ["(P1 safety: flow_lb_1, regime_b_carry_06, carry_r_050)"]

    # Load data
    print("\n[1/3] Loading data...", flush=True)
    t_start = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t_start:.1f}s", flush=True)

    # Run optimal
    print("\n[2/3] Running optimal T1 config...", flush=True)
    t0 = time.time()
    opt_cfg = build_config(mutations)
    opt_engine = IARICDailyEngine(opt_cfg, replay)
    opt_result = opt_engine.run()
    r_mults = compute_r_multiples(opt_result.trades)
    opt_metrics = extract_metrics(
        opt_result.trades, opt_result.equity_curve, opt_result.timestamps, INITIAL_EQUITY,
    )
    opt_score = composite_score(opt_metrics, INITIAL_EQUITY, r_multiples=r_mults, norm=IARIC_NORM)
    print(f"  Done in {time.time() - t0:.1f}s — {len(opt_result.trades)} trades", flush=True)

    # Run default baseline
    print("\n[3/3] Running default baseline...", flush=True)
    t0 = time.time()
    # Baseline: original defaults before any optimization deployments
    bl_mutations = {
        "param_overrides.t1_entry_flow_gate": False,
        "param_overrides.base_risk_fraction": 0.005,
        "param_overrides.dow_friday_mult": 1.0,
        "param_overrides.max_carry_days": 5,
    }
    bl_cfg = build_config(bl_mutations)
    bl_engine = IARICDailyEngine(bl_cfg, replay)
    bl_result = bl_engine.run()
    bl_r_mults = compute_r_multiples(bl_result.trades)
    bl_metrics = extract_metrics(
        bl_result.trades, bl_result.equity_curve, bl_result.timestamps, INITIAL_EQUITY,
    )
    bl_score = composite_score(bl_metrics, INITIAL_EQUITY, r_multiples=bl_r_mults, norm=IARIC_NORM)
    print(f"  Done in {time.time() - t0:.1f}s — {len(bl_result.trades)} trades", flush=True)

    # Generate full diagnostics
    print("\n[+] Running diagnostics...", flush=True)
    from research.backtests.stock.analysis.iaric_diagnostics import iaric_full_diagnostic
    diag = iaric_full_diagnostic(
        opt_result.trades,
        daily_selections=opt_result.daily_selections,
    )

    # Build report
    report = [
        "=" * 70,
        "IARIC T1 Greedy-Optimal — Full Diagnostic Report",
        "=" * 70,
        f"Period: {START_DATE} -> {END_DATE}",
        f"Equity: ${INITIAL_EQUITY:,.0f}  |  Tier: 1 (daily bars)",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 70,
        "CONFIGURATION",
        "=" * 70,
        "",
        "Greedy-selected features:",
        *[f"  + {f}" for f in kept_features],
        "",
        "Final mutations:",
        json.dumps(mutations, indent=2, sort_keys=True),
        "",
        "=" * 70,
        "PERFORMANCE COMPARISON",
        "=" * 70,
        "",
        fmt_metrics(bl_metrics, bl_score, "DEFAULT BASELINE"),
        "",
        fmt_metrics(opt_metrics, opt_score, "GREEDY OPTIMAL"),
        "",
        f"  {'─' * 50}",
        f"  IMPROVEMENT",
        f"  {'─' * 50}",
        f"  Score:       {bl_score.total:.6f} -> {opt_score.total:.6f}  "
        f"({(opt_score.total - bl_score.total) / max(bl_score.total, 1e-9) * 100:+.2f}%)",
        f"  Trades:      {bl_metrics.total_trades} -> {opt_metrics.total_trades}",
        f"  Win Rate:    {bl_metrics.win_rate * 100:.1f}% -> {opt_metrics.win_rate * 100:.1f}%",
        f"  PF:          {bl_metrics.profit_factor:.2f} -> {opt_metrics.profit_factor:.2f}",
        f"  Net:         ${bl_metrics.net_profit:,.2f} -> ${opt_metrics.net_profit:,.2f}",
        f"  Max DD:      {bl_metrics.max_drawdown_pct * 100:.2f}% -> {opt_metrics.max_drawdown_pct * 100:.2f}%",
        f"  Sharpe:      {bl_metrics.sharpe:.3f} -> {opt_metrics.sharpe:.3f}",
        f"  Calmar:      {bl_metrics.calmar:.3f} -> {opt_metrics.calmar:.3f}",
        "",
        "=" * 70,
        "EXIT REASON BREAKDOWN",
        "=" * 70,
        "",
        exit_breakdown(opt_result.trades),
        "",
        "=" * 70,
        "SECTOR BREAKDOWN",
        "=" * 70,
        "",
        sector_breakdown(opt_result.trades),
        "",
        "=" * 70,
        "REGIME BREAKDOWN",
        "=" * 70,
        "",
        regime_breakdown(opt_result.trades),
        "",
        "=" * 70,
        "MONTHLY PERFORMANCE",
        "=" * 70,
        "",
        monthly_breakdown(opt_result.trades),
        "",
        "=" * 70,
        "FULL DIAGNOSTICS (28+ sections)",
        "=" * 70,
        "",
        diag,
        "",
    ]

    total_time = time.time() - t_start
    report.append(f"Total generation time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    out_path = OUTPUT_DIR / "optimal_t1_diagnostics_iaric_v3.txt"
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n  Saved: {out_path} ({len(report)} lines)")
    print(f"  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
