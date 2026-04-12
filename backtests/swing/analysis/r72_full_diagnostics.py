"""Round 7.2 BRS — Full Diagnostics + R6 vs R7.2 Comparison.

Runs both R6 baseline and R7.2 optimized configs side-by-side.
Includes crisis state log analysis, trade timeline, and scoring breakdown.

Run from project root:
    python backtests/swing/analysis/r72_full_diagnostics.py
"""
from __future__ import annotations
import json, sys, math, datetime as dt
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import asdict

# --- bootstrap aliases -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing._aliases import install; install()

from backtests.swing.config_brs import BRSConfig
from backtests.swing.engine.brs_portfolio_engine import (
    load_brs_data, run_brs_independent,
)
from backtests.swing.analysis.brs_diagnostics import (
    compute_brs_diagnostics, CRISIS_WINDOWS,
)
from backtests.swing.auto.brs.scoring import (
    extract_brs_metrics, composite_score, BRSMetrics,
    W_NET_PROFIT, W_PF, W_CALMAR, W_INV_DD, W_BEAR_ALPHA, W_FREQUENCY, W_DETECTION,
)
from backtests.swing.auto.brs.config_mutator import mutate_brs_config
import numpy as np

DATA_DIR = Path("backtests/swing/data/raw")
OUTPUT_DIR = Path("backtests/swing/auto/brs/output")

# R6 baseline mutations
R6_MUTATIONS = {
    "disable_s1": False,
    "symbol_configs.GLD.adx_on": 16,
    "symbol_configs.GLD.adx_off": 14,
    "scale_out_enabled": True,
    "scale_out_pct": 0.33,
    "bd_donchian_period": 10,
    "lh_swing_lookback": 3,
    "scale_out_target_r": 3.0,
    "symbol_configs.QQQ.stop_floor_atr": 0.6,
    "symbol_configs.GLD.stop_buffer_atr": 0.2,
    "profit_floor_scale": 2.4,
}

# R7.2 mutations (R6 + 4 new from optimizer)
R72_MUTATIONS = {
    **R6_MUTATIONS,
    "profit_floor_scale": 2.88,       # CHANGED: 2.4 -> 2.88
    "peak_drop_enabled": True,         # NEW
    "peak_drop_pct": -0.05,            # NEW
    "peak_drop_lookback": 15,          # NEW
    "bd_arm_bars": 32,                 # NEW
}


def build_config(mutations: dict) -> BRSConfig:
    cfg = BRSConfig(data_dir=DATA_DIR)
    cfg = mutate_brs_config(cfg, mutations)
    return cfg


def run_full(cfg: BRSConfig):
    data = load_brs_data(cfg)
    result = run_brs_independent(data, cfg)
    diag = compute_brs_diagnostics(
        result.symbol_results,
        cfg.initial_equity,
        result.combined_equity,
        result.combined_timestamps,
    )
    metrics = extract_brs_metrics(result, cfg.initial_equity)
    score = composite_score(metrics)
    return result, diag, metrics, score


def scoring_breakdown(metrics: BRSMetrics, label: str) -> str:
    """Detailed scoring component breakdown."""
    lines = []
    lines.append(f"\n--- Scoring Breakdown ({label}) ---")

    # Reproduce component calculations
    np_return = max(metrics.net_return_pct / 100.0, 0.0)
    np_c = min(max(math.log(1.0 + np_return) / math.log(4.0), 0), 1)
    pf_c = min(max((metrics.profit_factor - 1.0) / 2.0, 0), 1)
    calmar_c = min(max(metrics.calmar / 10.0, 0), 1)
    inv_dd_c = min(max(1.0 - metrics.max_dd_pct / 0.30, 0), 1)
    bear_alpha_c = min(max(metrics.bear_alpha_pct / 20.0, 0), 1)
    freq_c = min(max(metrics.total_trades / 40.0, 0), 1)
    bias_lat_c = min(max(1.0 - metrics.bias_latency_days / 20.0, 0), 1)
    trade_lat_c = min(max(1.0 - metrics.detection_latency_days / 30.0, 0), 1)
    coverage_c = min(max(metrics.crisis_coverage, 0), 1)
    dq_c = 0.4 * bias_lat_c + 0.3 * trade_lat_c + 0.3 * coverage_c

    score = composite_score(metrics)

    lines.append(f"  {'Component':<25} {'Raw':>8} {'Norm':>8} {'Weight':>8} {'Weighted':>10}")
    lines.append(f"  {'-'*59}")
    components = [
        ("Net Profit", f"{metrics.net_return_pct:.1f}%", np_c, W_NET_PROFIT),
        ("Profit Factor", f"{metrics.profit_factor:.2f}", pf_c, W_PF),
        ("Calmar", f"{metrics.calmar:.1f}", calmar_c, W_CALMAR),
        ("Inverse DD", f"{metrics.max_dd_pct:.2%}", inv_dd_c, W_INV_DD),
        ("Bear Alpha", f"{metrics.bear_alpha_pct:.1f}%", bear_alpha_c, W_BEAR_ALPHA),
        ("Frequency", f"{metrics.total_trades}", freq_c, W_FREQUENCY),
        ("Detection Quality", f"composite", dq_c, W_DETECTION),
    ]
    total = 0
    for name, raw, norm, weight in components:
        weighted = norm * weight
        total += weighted
        saturated = " [SAT]" if norm >= 0.99 else ""
        lines.append(f"  {name:<25} {raw:>8} {norm:>8.3f} {weight:>8.2f} {weighted:>10.4f}{saturated}")
    lines.append(f"  {'-'*59}")
    lines.append(f"  {'TOTAL':<25} {'':>8} {'':>8} {'1.00':>8} {total:>10.4f}")

    # Detection sub-components
    lines.append(f"\n  Detection Quality Sub-Components:")
    lines.append(f"    Bias Latency:   {metrics.bias_latency_days:.1f}d -> score {bias_lat_c:.3f} (w=0.4, contrib={0.4*bias_lat_c:.4f})")
    lines.append(f"    Trade Latency:  {metrics.detection_latency_days:.1f}d -> score {trade_lat_c:.3f} (w=0.3, contrib={0.3*trade_lat_c:.4f})")
    lines.append(f"    Coverage:       {metrics.crisis_coverage:.2f} -> score {coverage_c:.3f} (w=0.3, contrib={0.3*coverage_c:.4f})")

    return "\n".join(lines)


def analyse_crisis_state_logs(result) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("CRISIS STATE LOG ANALYSIS")
    lines.append("=" * 70)

    for sym, sr in result.symbol_results.items():
        crisis_log = getattr(sr, "crisis_state_log", [])
        if not crisis_log:
            lines.append(f"\n  [{sym}] No crisis state log entries")
            continue

        lines.append(f"\n  [{sym}] {len(crisis_log)} crisis state entries")

        by_crisis = defaultdict(list)
        for entry in crisis_log:
            by_crisis[entry["crisis"]].append(entry)

        for crisis_name, entries in by_crisis.items():
            lines.append(f"\n  --- {crisis_name} ({len(entries)} daily bars) ---")
            lines.append(
                f"  {'Date':<12} {'Regime':<14} {'ADX':>5} {'Bias':>7} "
                f"{'Hold':>4} {'DlyRet%':>7} {'CumRet%':>7} {'ATR_r':>5} "
                f"{'CrashOvr':>8} {'PeakOvr':>7} {'CumOvr':>6} {'InPos':>5}"
            )
            for e in entries:
                cum_ret = e.get("cum_return", "n/a")
                cum_ovr = e.get("cum_return_override", False)
                lines.append(
                    f"  {e['date']:<12} {e['regime']:<14} {e['adx']:>5.1f} "
                    f"{e['bias_confirmed']:>7} {e['hold_count']:>4} "
                    f"{e['daily_return']:>+6.2f}% "
                    f"{cum_ret if isinstance(cum_ret, str) else f'{cum_ret:>+6.2f}%':>7} "
                    f"{e['atr_ratio']:>5.2f} "
                    f"{'Y' if e['crash_override'] else 'N':>8} "
                    f"{'Y' if e['peak_drop_override'] else 'N':>7} "
                    f"{'Y' if cum_ovr else 'N':>6} "
                    f"{'Y' if e['in_position'] else 'N':>5}"
                )

            # Detection latency for this crisis
            first_trade_in_window = None
            for t in sorted(sr.trades, key=lambda t: t.entry_time):
                t_naive = t.entry_time.replace(tzinfo=None) if t.entry_time.tzinfo else t.entry_time
                cw = next((c for c in CRISIS_WINDOWS if c[0] == crisis_name), None)
                if cw and cw[1] <= t_naive <= cw[2]:
                    first_trade_in_window = t
                    break

            if first_trade_in_window:
                cw = next(c for c in CRISIS_WINDOWS if c[0] == crisis_name)
                latency = (first_trade_in_window.entry_time.replace(tzinfo=None) - cw[1]).total_seconds() / 86400
                lines.append(f"  First trade: {first_trade_in_window.entry_time} ({latency:.1f}d latency)")
                lines.append(f"    Type: {first_trade_in_window.entry_type}, R: {first_trade_in_window.r_multiple:+.2f}")
            else:
                lines.append(f"  No trades during this crisis window")

            # Bias latency for this crisis
            bias_short = [e for e in entries if e["bias_confirmed"] == "SHORT"]
            if bias_short:
                cw = next((c for c in CRISIS_WINDOWS if c[0] == crisis_name), None)
                if cw:
                    from datetime import datetime
                    first_short_date = min(datetime.strptime(e["date"], "%Y-%m-%d") for e in bias_short)
                    bias_lat = (first_short_date - cw[1]).total_seconds() / 86400
                    lines.append(f"  First SHORT bias: {first_short_date.date()} ({bias_lat:.1f}d bias latency)")
            else:
                lines.append(f"  No SHORT bias confirmed during this crisis")

    return "\n".join(lines)


def analyse_trades(result) -> str:
    lines = []
    all_trades = []
    for sr in result.symbol_results.values():
        all_trades.extend(sr.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    if not all_trades:
        return "\nNo trades."

    # All trades
    lines.append("\n" + "=" * 70)
    lines.append("FULL TRADE LIST")
    lines.append("=" * 70)
    lines.append(
        f"  {'#':<4} {'Sym':<5} {'Type':<20} {'Entry':<22} {'Exit':<22} "
        f"{'R':>6} {'MFE':>5} {'Regime':<15} {'ExitReason'}"
    )
    for i, t in enumerate(all_trades, 1):
        lines.append(
            f"  {i:<4} {t.symbol:<5} {t.entry_type:<20} "
            f"{str(t.entry_time)[:19]:<22} {str(t.exit_time)[:19]:<22} "
            f"{t.r_multiple:>+5.2f} {t.mfe_r:>5.2f} {t.regime_entry:<15} {t.exit_reason}"
        )

    # Per-crisis trade counts
    lines.append(f"\n--- Trades Per Crisis Window ---")
    for cname, cstart, cend in CRISIS_WINDOWS:
        crisis_trades = [
            t for t in all_trades
            if cstart <= (t.entry_time.replace(tzinfo=None) if t.entry_time.tzinfo else t.entry_time) <= cend
        ]
        total_r = sum(t.r_multiple for t in crisis_trades)
        lines.append(f"  {cname}: {len(crisis_trades)} trades, Total R={total_r:+.2f}")
        for t in crisis_trades:
            lines.append(f"    {t.symbol} {t.entry_type} {str(t.entry_time)[:19]} R={t.r_multiple:+.2f} ({t.exit_reason})")

    # Trade distribution
    lines.append(f"\n--- Trade Distribution by Year ---")
    year_counts = Counter(t.entry_time.year for t in all_trades)
    for y in sorted(year_counts):
        yr_trades = [t for t in all_trades if t.entry_time.year == y]
        yr_r = sum(t.r_multiple for t in yr_trades)
        yr_wr = sum(1 for t in yr_trades if t.r_multiple > 0) / len(yr_trades) * 100
        lines.append(f"  {y}: {year_counts[y]} trades, WR={yr_wr:.0f}%, TotalR={yr_r:+.1f}")

    lines.append(f"\n--- Trade Distribution by Quarter ---")
    q_counts = Counter(f"{t.entry_time.year}-Q{(t.entry_time.month-1)//3+1}" for t in all_trades)
    for q in sorted(q_counts):
        lines.append(f"  {q}: {q_counts[q]} trades")

    # Entry type breakdown
    lines.append(f"\n--- Entry Type Breakdown ---")
    by_type = defaultdict(list)
    for t in all_trades:
        by_type[t.entry_type].append(t)
    for etype in sorted(by_type):
        tlist = by_type[etype]
        wr = sum(1 for t in tlist if t.r_multiple > 0) / len(tlist) * 100
        avg_r = np.mean([t.r_multiple for t in tlist])
        total_r = sum(t.r_multiple for t in tlist)
        avg_mfe = np.mean([t.mfe_r for t in tlist])
        lines.append(f"  {etype}: {len(tlist)} trades, WR={wr:.0f}%, AvgR={avg_r:+.2f}, TotalR={total_r:+.1f}, AvgMFE={avg_mfe:.2f}")

    # Exit reason breakdown
    lines.append(f"\n--- Exit Reason Breakdown ---")
    by_exit = Counter(t.exit_reason for t in all_trades)
    for reason, count in by_exit.most_common():
        exits = [t for t in all_trades if t.exit_reason == reason]
        avg_r = np.mean([t.r_multiple for t in exits])
        lines.append(f"  {reason}: {count} trades, AvgR={avg_r:+.2f}")

    # Regime breakdown
    lines.append(f"\n--- Regime at Entry Breakdown ---")
    by_regime = defaultdict(list)
    for t in all_trades:
        by_regime[t.regime_entry].append(t)
    for regime in sorted(by_regime):
        tlist = by_regime[regime]
        wr = sum(1 for t in tlist if t.r_multiple > 0) / len(tlist) * 100
        avg_r = np.mean([t.r_multiple for t in tlist])
        lines.append(f"  {regime}: {len(tlist)} trades, WR={wr:.0f}%, AvgR={avg_r:+.2f}")

    return "\n".join(lines)


def comparison_table(r6_m: BRSMetrics, r72_m: BRSMetrics) -> str:
    """Side-by-side R6 vs R7.2 comparison."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("R6 vs R7.2 COMPARISON")
    lines.append("=" * 70)
    lines.append(f"  {'Metric':<30} {'R6':>12} {'R7.2':>12} {'Delta':>12} {'Verdict':>8}")
    lines.append(f"  {'-'*74}")

    comparisons = [
        ("Total Trades", r6_m.total_trades, r72_m.total_trades, "higher"),
        ("Bear Trades", r6_m.bear_trades, r72_m.bear_trades, "higher"),
        ("Bear Trade WR %", r6_m.bear_trade_wr, r72_m.bear_trade_wr, "higher"),
        ("Profit Factor", r6_m.profit_factor, r72_m.profit_factor, "higher"),
        ("Max DD %", r6_m.max_dd_pct * 100, r72_m.max_dd_pct * 100, "lower"),
        ("Net Return %", r6_m.net_return_pct, r72_m.net_return_pct, "higher"),
        ("Sharpe", r6_m.sharpe, r72_m.sharpe, "higher"),
        ("Calmar", r6_m.calmar, r72_m.calmar, "higher"),
        ("Bear Alpha %", r6_m.bear_alpha_pct, r72_m.bear_alpha_pct, "higher"),
        ("Bear PF", r6_m.bear_pf, r72_m.bear_pf, "higher"),
        ("Bear Capture", r6_m.bear_capture_ratio, r72_m.bear_capture_ratio, "higher"),
        ("Exit Efficiency", r6_m.exit_efficiency, r72_m.exit_efficiency, "higher"),
        ("Trade Latency (d)", r6_m.detection_latency_days, r72_m.detection_latency_days, "lower"),
        ("Bias Latency (d)", r6_m.bias_latency_days, r72_m.bias_latency_days, "lower"),
        ("Crisis Coverage", r6_m.crisis_coverage, r72_m.crisis_coverage, "higher"),
        ("Regime F1", r6_m.regime_f1, r72_m.regime_f1, "higher"),
    ]

    for label, r6v, r72v, direction in comparisons:
        if isinstance(r6v, int):
            delta = r72v - r6v
            r6s, r72s, ds = f"{r6v:d}", f"{r72v:d}", f"{delta:+d}"
        else:
            delta = r72v - r6v
            r6s, r72s, ds = f"{r6v:.3f}", f"{r72v:.3f}", f"{delta:+.3f}"

        if direction == "higher":
            verdict = "+" if r72v > r6v else ("=" if r72v == r6v else "-")
        else:
            verdict = "+" if r72v < r6v else ("=" if r72v == r6v else "-")

        lines.append(f"  {label:<30} {r6s:>12} {r72s:>12} {ds:>12} {verdict:>8}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Running R6 baseline backtest...")
    r6_cfg = build_config(R6_MUTATIONS)
    r6_result, r6_diag, r6_metrics, r6_score = run_full(r6_cfg)

    print("Running R7.2 optimized backtest...")
    r72_cfg = build_config(R72_MUTATIONS)
    r72_result, r72_diag, r72_metrics, r72_score = run_full(r72_cfg)

    # Build full report
    lines = []
    lines.append("=" * 70)
    lines.append("ROUND 7.2 BRS — FULL DIAGNOSTICS")
    lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(f"Scoring: R7.2 rebalanced (detection 18%, bias_latency)")
    lines.append("=" * 70)

    # R7.2 config
    lines.append("\n--- R7.2 Configuration ---")
    lines.append(f"  Inherited from R6: {len(R6_MUTATIONS)} params")
    lines.append(f"  New/changed in R7.2:")
    lines.append(f"    peak_drop_enabled: True (NEW)")
    lines.append(f"    peak_drop_pct: -0.05 (NEW)")
    lines.append(f"    peak_drop_lookback: 15 (NEW)")
    lines.append(f"    bd_arm_bars: 32 (NEW)")
    lines.append(f"    profit_floor_scale: 2.4 -> 2.88 (CHANGED)")

    # Crisis windows
    lines.append("\n--- Crisis Windows ---")
    for cname, cstart, cend in CRISIS_WINDOWS:
        lines.append(f"  {cname}: {cstart.date()} to {cend.date()}")

    # Key metrics - R7.2
    lines.append("\n--- R7.2 Key Metrics ---")
    lines.append(f"  Total Trades:    {r72_metrics.total_trades}")
    lines.append(f"  Bear Trades:     {r72_metrics.bear_trades}")
    lines.append(f"  Profit Factor:   {r72_metrics.profit_factor:.2f}")
    lines.append(f"  Max DD:          {r72_metrics.max_dd_pct:.2%}")
    lines.append(f"  Net Return:      {r72_metrics.net_return_pct:.1f}%")
    lines.append(f"  Sharpe:          {r72_metrics.sharpe:.2f}")
    lines.append(f"  Calmar:          {r72_metrics.calmar:.1f}")
    lines.append(f"  Bear Alpha:      {r72_metrics.bear_alpha_pct:.1f}%")
    lines.append(f"  Trade Latency:   {r72_metrics.detection_latency_days:.1f}d")
    lines.append(f"  Bias Latency:    {r72_metrics.bias_latency_days:.1f}d")
    lines.append(f"  Crisis Coverage: {r72_metrics.crisis_coverage:.2f}")
    lines.append(f"  Exit Efficiency: {r72_metrics.exit_efficiency:.3f}")
    lines.append(f"  Composite Score: {r72_score.total:.4f}")

    # Comparison table
    lines.append(comparison_table(r6_metrics, r72_metrics))

    # Scoring breakdowns
    lines.append(scoring_breakdown(r6_metrics, "R6"))
    lines.append(scoring_breakdown(r72_metrics, "R7.2"))

    # Score comparison
    r6_total = composite_score(r6_metrics)
    lines.append(f"\n--- Composite Score ---")
    lines.append(f"  R6:   {r6_total.total:.4f}")
    lines.append(f"  R7.2: {r72_score.total:.4f}")
    lines.append(f"  Delta: {r72_score.total - r6_total.total:+.4f}")

    # Standard diagnostics report (R7.2)
    lines.append("\n" + "=" * 70)
    lines.append("R7.2 STANDARD DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append(r72_diag.report)

    # Per-symbol breakdown
    lines.append("\n--- Per-Symbol Breakdown (R7.2) ---")
    for sym, sr in r72_result.symbol_results.items():
        trades = sr.trades
        wins = [t for t in trades if t.r_multiple > 0]
        total_r = sum(t.r_multiple for t in trades)
        wr = len(wins) / len(trades) * 100 if trades else 0
        lines.append(f"\n  [{sym}] {len(trades)} trades, WR={wr:.1f}%, Total R={total_r:.1f}")
        if trades:
            by_type = {}
            for t in trades:
                by_type.setdefault(t.entry_type, []).append(t)
            for etype, tlist in sorted(by_type.items()):
                wr_e = sum(1 for t in tlist if t.r_multiple > 0) / len(tlist) * 100
                avg_r = np.mean([t.r_multiple for t in tlist])
                lines.append(f"    {etype}: {len(tlist)} trades, WR={wr_e:.0f}%, AvgR={avg_r:.2f}")

    # Crisis state log analysis (R7.2)
    lines.append(analyse_crisis_state_logs(r72_result))

    # Trade list (R7.2)
    lines.append(analyse_trades(r72_result))

    full_report = "\n".join(lines)

    # Save
    out_path = OUTPUT_DIR / "r72_full_diagnostics_detailed.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_report, encoding="utf-8")

    print(full_report)
    print(f"\n\nSaved to {out_path}")
