"""Run the P9 optimal config and save full diagnostics with extended analysis.

Baseline = P8 optimal config (with sector_mult_consumer_disc removed).
Optimal  = P9 optimal config (from greedy_optimal_alcb_p9.json).

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    PYTHONUNBUFFERED=1 python -u -m research.backtests.stock.auto.output.save_optimal_diagnostic_p9
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import time as dt_time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from research.backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
from research.backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis
from research.backtests.stock.analysis.alcb_score_attribution import score_component_attribution
from research.backtests.stock.analysis.alcb_short_hold_analysis import short_hold_deep_dive
from research.backtests.stock.analysis.alcb_winner_prediction import winner_prediction_analysis
from research.backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
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
        f"  Tail Loss (%):   {m.tail_loss_pct * 100:.2f}%",
        f"  Tail Loss (R):   {m.tail_loss_r:.3f}",
        "",
        f"  Composite Score: {score.total:.6f}",
        f"    Net P  (30%):    {score.net_profit_component:.6f}",
        f"    PF     (20%):    {score.pf_component:.6f}",
        f"    Edge   (15%):    {score.edge_tstat_component:.6f}",
        f"    WR     (15%):    {score.wr_component:.6f}",
        f"    Calmar (10%):    {score.calmar_component:.6f}",
        f"    Inv DD (10%):    {score.inv_dd_component:.6f}",
    ]
    if score.rejected:
        lines.append(f"    REJECTED:        {score.reject_reason}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("ALCB T2 Phase 9 Optimal — Full Diagnostic Report")
    print("=" * 70, flush=True)

    # Load P9 optimal config
    p9_config_path = OUTPUT_DIR / "optimal_config_alcb_p9.json"
    if not p9_config_path.exists():
        print(f"ERROR: {p9_config_path} not found — run P9 greedy first.")
        sys.exit(1)

    raw_p9 = json.loads(p9_config_path.read_text(encoding="utf-8"))
    p9_overrides = _parse_time_fields(raw_p9.get("param_overrides", {}))
    p9_ablation = raw_p9.get("ablation", {})

    # P8 baseline (= P9 base, i.e. P8 optimal minus sector_mult)
    raw_p8 = json.loads(
        (OUTPUT_DIR / "optimal_config_alcb_p8.json").read_text(encoding="utf-8")
    )
    # Remove overfitted sector_mult_consumer_disc from P8 baseline
    p8_params = raw_p8.get("param_overrides", {})
    p8_params.pop("sector_mult_consumer_disc", None)
    p8_overrides = _parse_time_fields(p8_params)
    p8_ablation = raw_p8.get("ablation", {})

    print("\n[1/6] Loading data...", flush=True)
    t_start = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t_start:.1f}s", flush=True)

    # Run P8 baseline (minus sector_mult)
    print("\n[2/6] Running P8 baseline (sans sector_mult)...", flush=True)
    t0 = time.time()
    bl_config = build_config(p8_overrides, p8_ablation)
    bl_engine = ALCBIntradayEngine(bl_config, replay)
    bl_result = bl_engine.run()
    bl_metrics = extract_metrics(
        bl_result.trades, bl_result.equity_curve, bl_result.timestamps, INITIAL_EQUITY,
    )
    bl_score = composite_score(bl_metrics, INITIAL_EQUITY, r_multiples=compute_r_multiples(bl_result.trades))
    print(f"  P8 done in {time.time() - t0:.1f}s — {len(bl_result.trades)} trades", flush=True)

    # Run P9 optimal with shadow tracker
    print("\n[3/6] Running P9 optimal with shadow tracker...", flush=True)
    t0 = time.time()
    opt_config = build_config(p9_overrides, p9_ablation)
    shadow = ALCBShadowTracker()
    opt_engine = ALCBIntradayEngine(opt_config, replay)
    opt_engine.shadow_tracker = shadow
    opt_result = opt_engine.run()
    opt_metrics = extract_metrics(
        opt_result.trades, opt_result.equity_curve, opt_result.timestamps, INITIAL_EQUITY,
    )
    opt_score = composite_score(opt_metrics, INITIAL_EQUITY, r_multiples=compute_r_multiples(opt_result.trades))
    print(f"  P9 done in {time.time() - t0:.1f}s — {len(opt_result.trades)} trades", flush=True)

    # Generate full diagnostic
    print("\n[4/6] Generating full diagnostic report...", flush=True)
    diagnostic_text = alcb_full_diagnostic(
        opt_result.trades,
        shadow_tracker=shadow,
        daily_selections=opt_result.daily_selections,
    )

    # Extended analysis scripts
    print("\n[5/6] Running extended analysis...", flush=True)
    t0 = time.time()
    score_attr_text = score_component_attribution(opt_result.trades)
    print(f"  Score attribution done ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    short_hold_text = short_hold_deep_dive(opt_result.trades)
    print(f"  Short hold analysis done ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    winner_pred_text = winner_prediction_analysis(opt_result.trades)
    print(f"  Winner prediction done ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    max_pos = p9_overrides.get("max_positions", 5)
    qe_replacement_text = qe_replacement_analysis(opt_result.trades, max_positions=max_pos)
    print(f"  QE replacement analysis done ({time.time() - t0:.1f}s)", flush=True)

    # Identify P9 mutations vs P8
    print("\n[6/6] Assembling report...", flush=True)
    p9_new_params = {}
    for k, v in p9_overrides.items():
        if k not in p8_overrides or p8_overrides[k] != v:
            p9_new_params[k] = v
    p9_new_ablation = {}
    for k, v in p9_ablation.items():
        if k not in p8_ablation or p8_ablation[k] != v:
            p9_new_ablation[k] = v

    # Clean P8 baseline JSON for display (without sector_mult)
    raw_p8_display = dict(raw_p8)
    if "param_overrides" in raw_p8_display:
        p8_display_params = dict(raw_p8_display["param_overrides"])
        p8_display_params.pop("sector_mult_consumer_disc", None)
        raw_p8_display["param_overrides"] = p8_display_params

    # ── Assemble report: important diagnostics first, appendices last ──

    # --- Header + Performance Comparison (most important) ---
    report_lines = [
        "=" * 70,
        "ALCB T2 Phase 9 Optimal — Full Diagnostic Report",
        "=" * 70,
        f"Period: {START_DATE} → {END_DATE}",
        f"Equity: ${INITIAL_EQUITY:,.0f}  |  Tier: 2 (5m momentum)",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  PERFORMANCE COMPARISON                                             ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        fmt_metrics(bl_metrics, bl_score, "P8 BASELINE (sans sector_mult)"),
        "",
        fmt_metrics(opt_metrics, opt_score, "P9 OPTIMAL"),
        "",
        f"  {'─' * 50}",
        f"  P8 → P9 IMPROVEMENT",
        f"  {'─' * 50}",
        f"  Score:       {bl_score.total:.6f} → {opt_score.total:.6f}  "
        f"({(opt_score.total - bl_score.total) / bl_score.total * 100:+.2f}%)" if bl_score.total > 0 else "",
        f"  Trades:      {bl_metrics.total_trades} → {opt_metrics.total_trades}",
        f"  Win Rate:    {bl_metrics.win_rate * 100:.1f}% → {opt_metrics.win_rate * 100:.1f}%",
        f"  PF:          {bl_metrics.profit_factor:.2f} → {opt_metrics.profit_factor:.2f}",
        f"  Net:         ${bl_metrics.net_profit:,.2f} → ${opt_metrics.net_profit:,.2f}",
        f"  Max DD:      {bl_metrics.max_drawdown_pct * 100:.2f}% → {opt_metrics.max_drawdown_pct * 100:.2f}%",
        f"  Sharpe:      {bl_metrics.sharpe:.3f} → {opt_metrics.sharpe:.3f}",
        f"  Sortino:     {bl_metrics.sortino:.3f} → {opt_metrics.sortino:.3f}",
        f"  Calmar:      {bl_metrics.calmar:.3f} → {opt_metrics.calmar:.3f}",
        "",
    ]

    # --- P9 Config Diff (concise: just what changed) ---
    report_lines.extend([
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  P9 CONFIG CHANGES vs P8                                            ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
    ])
    for k, v in sorted(p9_new_params.items()):
        old = p8_overrides.get(k, "(default)")
        report_lines.append(f"  {k}: {old} → {v}")
    for k, v in sorted(p9_new_ablation.items()):
        old = p8_ablation.get(k, "(default)")
        report_lines.append(f"  ablation.{k}: {old} → {v}")
    if not p9_new_params and not p9_new_ablation:
        report_lines.append("  (none — P9 greedy found no improvement over P8)")

    # --- Extended analysis (actionable insights) ---
    report_lines.extend([
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  QUICK EXIT REPLACEMENT TRADE ANALYSIS                              ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        qe_replacement_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  SHORT HOLD ANALYSIS                                                ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        short_hold_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  MOMENTUM SCORE COMPONENT ATTRIBUTION                               ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        score_attr_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  WINNER vs LOSER PREDICTION                                         ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        winner_pred_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  FULL 30-SECTION DIAGNOSTIC                                         ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        diagnostic_text,
        "",
    ])

    # --- Appendices (reference material) ---
    report_lines.extend([
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  APPENDIX A: FULL CONFIGURATION                                     ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        "P8 Baseline Config (sans sector_mult_consumer_disc):",
        json.dumps(raw_p8_display, indent=2, sort_keys=True, default=str),
        "",
        "P9 Optimal Config:",
        json.dumps(raw_p9, indent=2, sort_keys=True, default=str),
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  APPENDIX B: TRADE LIST                                             ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
    ])

    header = (
        f"{'#':>3}  {'Symbol':<6}  {'Dir':<5}  {'Entry Date':<12}  {'Exit Date':<12}  "
        f"{'Entry$':>8}  {'Exit$':>8}  {'Qty':>4}  {'P&L':>9}  {'R':>6}  "
        f"{'Exit Reason':<14}  {'Hold(h)':>7}"
    )
    report_lines.append(header)
    report_lines.append("─" * len(header))

    for i, t in enumerate(opt_result.trades, 1):
        entry_dt = t.entry_time.strftime("%Y-%m-%d") if t.entry_time else "?"
        exit_dt = t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "?"
        report_lines.append(
            f"{i:>3}  {t.symbol:<6}  {t.direction:<5}  {entry_dt:<12}  {exit_dt:<12}  "
            f"{t.entry_price:>8.2f}  {t.exit_price:>8.2f}  {t.quantity:>4}  "
            f"${t.pnl_net:>8.2f}  {t.r_multiple:>+5.2f}R  "
            f"{t.exit_reason:<14}  {t.hold_hours:>6.1f}h"
        )
    report_lines.append("")

    total_time = time.time() - t_start
    report_lines.append(f"Total generation time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "optimal_baseline_p9.txt"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\n  Saved: {out_path} ({len(report_lines)} lines)", flush=True)
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
