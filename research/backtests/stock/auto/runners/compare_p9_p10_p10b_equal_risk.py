"""Compare P9, P10, P10b on equal risk basis (all at 0.5% base_risk_fraction).

P9 normally uses 2% risk — this forces it to 0.5% for a true like-for-like
comparison of the signal/filter quality independent of risk sizing.

Usage:
    cd C:/Users/sehyu/Documents/Other/Projects/trading
    PYTHONUNBUFFERED=1 python -u -m research.backtests.stock.auto.output.compare_p9_p10_p10b_equal_risk
"""
from __future__ import annotations

import io
import json
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
EQUAL_RISK = 0.005  # Force all configs to 0.5%

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
        f"  {'─' * 55}",
        f"  {label}",
        f"  {'─' * 55}",
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


def load_config_json(path: Path) -> tuple[dict, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    overrides = _parse_time_fields(raw.get("param_overrides", {}))
    ablation = raw.get("ablation", {})
    return overrides, ablation


def run_config(label: str, overrides: dict, ablation: dict, replay, with_shadow: bool = False):
    """Run a single config and return (result, metrics, score, shadow)."""
    print(f"\n  Running {label}...", flush=True)
    t0 = time.time()
    config = build_config(overrides, ablation)
    shadow = ALCBShadowTracker() if with_shadow else None
    engine = ALCBIntradayEngine(config, replay)
    if shadow:
        engine.shadow_tracker = shadow
    result = engine.run()
    metrics = extract_metrics(
        result.trades, result.equity_curve, result.timestamps, INITIAL_EQUITY,
    )
    score = composite_score(
        metrics, INITIAL_EQUITY, r_multiples=compute_r_multiples(result.trades),
    )
    print(f"  {label} done in {time.time() - t0:.1f}s — {len(result.trades)} trades", flush=True)
    return result, metrics, score, shadow


def main():
    print("=" * 70)
    print("ALCB T2 — P9 vs P10 vs P10b EQUAL RISK COMPARISON")
    print(f"All configs forced to base_risk_fraction = {EQUAL_RISK} (0.5%)")
    print("=" * 70, flush=True)

    # Load configs
    p9_overrides, p9_ablation = load_config_json(OUTPUT_DIR / "optimal_config_alcb_p9.json")
    p10_overrides, p10_ablation = load_config_json(OUTPUT_DIR / "optimal_config_alcb_p10.json")
    p10b_overrides, p10b_ablation = load_config_json(OUTPUT_DIR / "optimal_config_alcb_p10b.json")

    # Force equal risk on all configs
    p9_overrides["base_risk_fraction"] = EQUAL_RISK
    p10_overrides["base_risk_fraction"] = EQUAL_RISK
    p10b_overrides["base_risk_fraction"] = EQUAL_RISK

    print(f"\nP9 original risk: 2.0% → forced to {EQUAL_RISK*100:.1f}%")
    print(f"P10 original risk: 0.5% → unchanged at {EQUAL_RISK*100:.1f}%")
    print(f"P10b original risk: 0.5% → unchanged at {EQUAL_RISK*100:.1f}%")

    # Load data
    print("\n[1/5] Loading data...", flush=True)
    t_start = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Loaded in {time.time() - t_start:.1f}s", flush=True)

    # Run all 3
    print("\n[2/5] Running all configs at equal 0.5% risk...", flush=True)
    p9_result, p9_metrics, p9_score, _ = run_config(
        "P9 (0.5% risk)", p9_overrides, p9_ablation, replay,
    )
    p10_result, p10_metrics, p10_score, _ = run_config(
        "P10 raw (0.5% risk)", p10_overrides, p10_ablation, replay,
    )
    p10b_result, p10b_metrics, p10b_score, p10b_shadow = run_config(
        "P10b curated (0.5% risk)", p10b_overrides, p10b_ablation, replay,
        with_shadow=True,
    )

    # Extended analysis on P10b (the recommended config)
    print("\n[3/5] Running extended analysis on P10b...", flush=True)
    t0 = time.time()
    diagnostic_text = alcb_full_diagnostic(
        p10b_result.trades,
        shadow_tracker=p10b_shadow,
        daily_selections=p10b_result.daily_selections,
    )
    print(f"  Full diagnostic done ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    score_attr_text = score_component_attribution(p10b_result.trades)
    short_hold_text = short_hold_deep_dive(p10b_result.trades)
    winner_pred_text = winner_prediction_analysis(p10b_result.trades)
    max_pos = p10b_overrides.get("max_positions", 5)
    qe_text = qe_replacement_analysis(p10b_result.trades, max_positions=max_pos)
    print(f"  Extended analysis done ({time.time() - t0:.1f}s)", flush=True)

    # Assemble report
    print("\n[4/5] Assembling report...", flush=True)

    report_lines = [
        "=" * 70,
        "ALCB T2 — P9 vs P10 vs P10b EQUAL RISK COMPARISON",
        "=" * 70,
        f"Period: {START_DATE} → {END_DATE}",
        f"Equity: ${INITIAL_EQUITY:,.0f}  |  Tier: 2 (5m momentum)  |  Leverage: 2:1",
        f"ALL configs forced to base_risk_fraction = {EQUAL_RISK} (0.5%)",
        f"Scoring: Calmar /50.0, Net Profit log(6), new normalization",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This comparison isolates signal/filter quality from risk sizing.",
        "P9 normally uses 2% risk (4x higher). Here all are at 0.5%.",
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  EQUAL-RISK PERFORMANCE COMPARISON                                  ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        fmt_metrics(p9_metrics, p9_score, "P9 @ 0.5% RISK (signal quality only)"),
        "",
        fmt_metrics(p10_metrics, p10_score, "P10 RAW @ 0.5% RISK"),
        "",
        fmt_metrics(p10b_metrics, p10b_score, "P10b CURATED @ 0.5% RISK (recommended)"),
        "",
    ]

    # Side-by-side summary table
    report_lines.extend([
        f"  {'─' * 55}",
        f"  SIDE-BY-SIDE SUMMARY (all at 0.5% risk)",
        f"  {'─' * 55}",
        f"  {'Metric':<20s} {'P9':>12s} {'P10':>12s} {'P10b':>12s}  {'Best':>6s}",
        f"  {'─' * 65}",
    ])

    def _row(label, p9v, p10v, p10bv, fmt=".2f", higher_better=True):
        vals = [p9v, p10v, p10bv]
        names = ["P9", "P10", "P10b"]
        best_idx = vals.index(max(vals)) if higher_better else vals.index(min(vals))
        best = names[best_idx]
        return f"  {label:<20s} {p9v:>12{fmt}} {p10v:>12{fmt}} {p10bv:>12{fmt}}  {best:>6s}"

    report_lines.extend([
        _row("Score", p9_score.total, p10_score.total, p10b_score.total, ".6f"),
        _row("Trades", p9_metrics.total_trades, p10_metrics.total_trades, p10b_metrics.total_trades, "d"),
        _row("Win Rate %", p9_metrics.win_rate * 100, p10_metrics.win_rate * 100, p10b_metrics.win_rate * 100, ".1f"),
        _row("Profit Factor", p9_metrics.profit_factor, p10_metrics.profit_factor, p10b_metrics.profit_factor),
        _row("Net Profit $", p9_metrics.net_profit, p10_metrics.net_profit, p10b_metrics.net_profit, ",.2f"),
        _row("Expectancy (R)", p9_metrics.expectancy, p10_metrics.expectancy, p10b_metrics.expectancy, ".3f"),
        _row("CAGR %", p9_metrics.cagr * 100, p10_metrics.cagr * 100, p10b_metrics.cagr * 100, ".2f"),
        _row("Max DD %", p9_metrics.max_drawdown_pct * 100, p10_metrics.max_drawdown_pct * 100, p10b_metrics.max_drawdown_pct * 100, ".2f", higher_better=False),
        _row("Sharpe", p9_metrics.sharpe, p10_metrics.sharpe, p10b_metrics.sharpe, ".3f"),
        _row("Sortino", p9_metrics.sortino, p10_metrics.sortino, p10b_metrics.sortino, ".3f"),
        _row("Calmar", p9_metrics.calmar, p10_metrics.calmar, p10b_metrics.calmar, ".3f"),
        _row("Avg Hold (h)", p9_metrics.avg_hold_hours, p10_metrics.avg_hold_hours, p10b_metrics.avg_hold_hours, ".1f", higher_better=False),
        _row("Trades/Month", p9_metrics.trades_per_month, p10_metrics.trades_per_month, p10b_metrics.trades_per_month, ".1f"),
        _row("Commissions $", p9_metrics.total_commissions, p10_metrics.total_commissions, p10b_metrics.total_commissions, ",.2f", higher_better=False),
        _row("Tail Loss (R)", p9_metrics.tail_loss_r, p10_metrics.tail_loss_r, p10b_metrics.tail_loss_r, ".3f", higher_better=False),
        "",
    ])

    # Score component comparison
    report_lines.extend([
        f"  {'─' * 55}",
        f"  SCORE COMPONENTS (all at 0.5% risk)",
        f"  {'─' * 55}",
        f"  {'Component':<20s} {'P9':>12s} {'P10':>12s} {'P10b':>12s}",
        f"  {'─' * 55}",
        f"  {'Net Profit (30%)':<20s} {p9_score.net_profit_component:>12.6f} {p10_score.net_profit_component:>12.6f} {p10b_score.net_profit_component:>12.6f}",
        f"  {'PF (20%)':<20s} {p9_score.pf_component:>12.6f} {p10_score.pf_component:>12.6f} {p10b_score.pf_component:>12.6f}",
        f"  {'Edge t-stat (15%)':<20s} {p9_score.edge_tstat_component:>12.6f} {p10_score.edge_tstat_component:>12.6f} {p10b_score.edge_tstat_component:>12.6f}",
        f"  {'Win Rate (15%)':<20s} {p9_score.wr_component:>12.6f} {p10_score.wr_component:>12.6f} {p10b_score.wr_component:>12.6f}",
        f"  {'Calmar (10%)':<20s} {p9_score.calmar_component:>12.6f} {p10_score.calmar_component:>12.6f} {p10b_score.calmar_component:>12.6f}",
        f"  {'Inv DD (10%)':<20s} {p9_score.inv_dd_component:>12.6f} {p10_score.inv_dd_component:>12.6f} {p10b_score.inv_dd_component:>12.6f}",
        "",
    ])

    # P10b extended analysis
    report_lines.extend([
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  P10b QUICK EXIT REPLACEMENT ANALYSIS                               ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        qe_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  P10b SHORT HOLD ANALYSIS                                           ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        short_hold_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  P10b MOMENTUM SCORE COMPONENT ATTRIBUTION                          ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        score_attr_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  P10b WINNER vs LOSER PREDICTION                                    ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        winner_pred_text,
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  P10b FULL 30-SECTION DIAGNOSTIC                                    ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        diagnostic_text,
        "",
    ])

    # Appendix: configs
    report_lines.extend([
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  APPENDIX A: CONFIGURATIONS (with forced 0.5% risk)                 ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        "P9 Config (risk forced to 0.5%):",
        json.dumps({"ablation": p9_ablation, "param_overrides": {k: str(v) if isinstance(v, dt_time) else v for k, v in p9_overrides.items()}}, indent=2, sort_keys=True, default=str),
        "",
        "P10 Config (risk unchanged at 0.5%):",
        json.dumps({"ablation": p10_ablation, "param_overrides": {k: str(v) if isinstance(v, dt_time) else v for k, v in p10_overrides.items()}}, indent=2, sort_keys=True, default=str),
        "",
        "P10b Config (risk unchanged at 0.5%):",
        json.dumps({"ablation": p10b_ablation, "param_overrides": {k: str(v) if isinstance(v, dt_time) else v for k, v in p10b_overrides.items()}}, indent=2, sort_keys=True, default=str),
        "",
    ])

    # Trade list for P10b
    report_lines.extend([
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  APPENDIX B: P10b TRADE LIST                                        ║",
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
    for i, t in enumerate(p10b_result.trades, 1):
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

    # Save
    print("\n[5/5] Saving report...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "equal_risk_comparison_p9_p10_p10b.txt"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\n  Saved: {out_path} ({len(report_lines)} lines)", flush=True)
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
