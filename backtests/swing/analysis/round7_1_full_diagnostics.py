"""Round 7.1 BRS — Full Diagnostics + Crisis State Analysis.

Runs R6 baseline config (no new mutations kept by R7.1 optimizer)
with COVID removed from CRISIS_WINDOWS. Includes crisis state log
analysis to show why Path G cumulative return didn't fire.

Run from project root:
    python backtests/swing/analysis/round7_1_full_diagnostics.py
"""
from __future__ import annotations
import json, sys, datetime as dt
from pathlib import Path
from collections import Counter, defaultdict

# --- bootstrap aliases -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing.config_brs import BRSConfig
from backtests.swing.engine.brs_portfolio_engine import (
    load_brs_data, run_brs_synchronized,
)
from backtests.swing.analysis.brs_diagnostics import (
    compute_brs_diagnostics, CRISIS_WINDOWS,
)
from backtests.swing.auto.brs.scoring import extract_brs_metrics
import numpy as np

DATA_DIR = Path("backtests/swing/data/raw")
OUTPUT_DIR = Path("backtests/swing/auto/brs/output")


# ---------------------------------------------------------------------------
# 1. Build Round 6 config (= R7.1 since no new mutations kept)
# ---------------------------------------------------------------------------
def build_config() -> BRSConfig:
    cfg = BRSConfig(data_dir=DATA_DIR)
    # Cumulative mutations from R6 phase_state.json (unchanged by R7.1)
    cfg.disable_s1 = False
    cfg.scale_out_enabled = True
    cfg.scale_out_pct = 0.33
    cfg.bd_donchian_period = 10
    cfg.lh_swing_lookback = 3
    cfg.scale_out_target_r = 3.0
    cfg.profit_floor_scale = 2.4
    cfg.symbol_configs["GLD"].adx_on = 16
    cfg.symbol_configs["GLD"].adx_off = 14
    cfg.symbol_configs["QQQ"].stop_floor_atr = 0.6
    cfg.symbol_configs["GLD"].stop_buffer_atr = 0.2
    return cfg


# ---------------------------------------------------------------------------
# 2. Run backtest + diagnostics
# ---------------------------------------------------------------------------
def run_full(cfg: BRSConfig):
    data = load_brs_data(cfg)
    result = run_brs_synchronized(data, cfg)
    diag = compute_brs_diagnostics(
        result.symbol_results,
        cfg.initial_equity,
        result.combined_equity,
        result.combined_timestamps,
    )
    metrics = extract_brs_metrics(result, cfg.initial_equity)
    return result, diag, data, metrics


# ---------------------------------------------------------------------------
# 3. Crisis state log analysis
# ---------------------------------------------------------------------------
def analyse_crisis_state_logs(result, cfg) -> str:
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

        # Group by crisis
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

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Path G analysis — why it didn't fire
# ---------------------------------------------------------------------------
def analyse_path_g_potential(result, cfg) -> str:
    """Compute what cum_return values looked like during crisis windows
    to understand why Path G wasn't selected."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("PATH G CUMULATIVE RETURN ANALYSIS")
    lines.append("=" * 70)

    for sym, sr in result.symbol_results.items():
        crisis_log = getattr(sr, "crisis_state_log", [])
        if not crisis_log:
            continue

        lines.append(f"\n  [{sym}]")

        by_crisis = defaultdict(list)
        for entry in crisis_log:
            by_crisis[entry["crisis"]].append(entry)

        for crisis_name, entries in by_crisis.items():
            lines.append(f"\n  --- {crisis_name} ---")

            # Check which entries would have triggered Path G at various thresholds
            for lookback_label, thresh_candidates in [
                ("Default (5d, -3%)", -0.03),
                ("Tight (3d, -2%)", -0.02),
                ("Loose (7d, -4%)", -0.04),
            ]:
                cum_vals = [e.get("cum_return", None) for e in entries]
                cum_vals_valid = [v for v in cum_vals if isinstance(v, (int, float))]
                if cum_vals_valid:
                    min_cum = min(cum_vals_valid)
                    max_cum = max(cum_vals_valid)
                    triggered = sum(1 for v in cum_vals_valid if v / 100 <= thresh_candidates)
                    lines.append(
                        f"  {lookback_label}: cum_return range [{min_cum:+.2f}%, {max_cum:+.2f}%], "
                        f"would trigger {triggered}/{len(cum_vals_valid)} days"
                    )

            # Check bias state — Path G needs raw_direction=SHORT + hold_count >= 1 + close < ema_fast
            bias_short_days = sum(1 for e in entries if e["bias_confirmed"] == "SHORT")
            bias_flat_days = sum(1 for e in entries if e["bias_confirmed"] == "FLAT")
            raw_short_days = sum(1 for e in entries if e.get("bias_raw", "") == "SHORT")
            below_ema_days = sum(1 for e in entries if e["close"] < e["ema_fast"])
            lines.append(
                f"  Bias: {bias_short_days} SHORT, {bias_flat_days} FLAT | "
                f"Raw SHORT: {raw_short_days} | Close < EMA_fast: {below_ema_days}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Recent gap + trade distribution
# ---------------------------------------------------------------------------
def analyse_trades(result, cfg) -> str:
    lines = []
    all_trades = []
    for sr in result.symbol_results.values():
        all_trades.extend(sr.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    if not all_trades:
        return "\nNo trades."

    # Last 15 trades
    lines.append("\n" + "=" * 70)
    lines.append("TRADE TIMELINE (last 15)")
    lines.append("=" * 70)
    lines.append(
        f"{'Sym':<5} {'Type':<20} {'Entry':<22} {'Exit':<22} "
        f"{'R':>6} {'Regime':<15} {'ExitReason'}"
    )
    for t in all_trades[-15:]:
        lines.append(
            f"{t.symbol:<5} {t.entry_type:<20} "
            f"{str(t.entry_time)[:19]:<22} {str(t.exit_time)[:19]:<22} "
            f"{t.r_multiple:>+5.2f}  {t.regime_entry:<15} {t.exit_reason}"
        )

    # Per-crisis trade counts
    lines.append(f"\n--- Trades Per Crisis Window ---")
    for cname, cstart, cend in CRISIS_WINDOWS:
        crisis_trades = [
            t for t in all_trades
            if cstart <= (t.entry_time.replace(tzinfo=None) if t.entry_time.tzinfo else t.entry_time) <= cend
        ]
        lines.append(f"  {cname}: {len(crisis_trades)} trades")
        for t in crisis_trades:
            lines.append(f"    {t.symbol} {t.entry_type} {str(t.entry_time)[:19]} R={t.r_multiple:+.2f}")

    # Trade distribution
    lines.append(f"\n--- Trade Distribution by Year ---")
    year_counts = Counter(t.entry_time.year for t in all_trades)
    for y in sorted(year_counts):
        lines.append(f"  {y}: {year_counts[y]} trades")

    lines.append(f"\n--- Trade Distribution by Quarter ---")
    q_counts = Counter(f"{t.entry_time.year}-Q{(t.entry_time.month-1)//3+1}" for t in all_trades)
    for q in sorted(q_counts):
        lines.append(f"  {q}: {q_counts[q]} trades")

    # Gap analysis
    last_trade = all_trades[-1]
    lines.append(f"\n--- Last Trade ---")
    lines.append(f"  {last_trade.symbol} {last_trade.entry_type}")
    lines.append(f"  Entry: {last_trade.entry_time}  Exit: {last_trade.exit_time}")
    lines.append(f"  R: {last_trade.r_multiple:+.2f}  Regime: {last_trade.regime_entry}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = build_config()
    print("Running R7.1 full backtest...")
    result, diag, data, metrics = run_full(cfg)

    # Build full report
    lines = []
    lines.append("=" * 70)
    lines.append("ROUND 7.1 BRS — FULL DIAGNOSTICS")
    lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(f"Config: R6 baseline (no new R7.1 mutations kept)")
    lines.append(f"Change: COVID removed from CRISIS_WINDOWS (data starts 2021)")
    lines.append("=" * 70)

    # Metrics summary
    lines.append("\n--- Key Metrics ---")
    lines.append(f"  Total Trades:    {metrics.total_trades}")
    lines.append(f"  Bear Trades:     {metrics.bear_trades}")
    lines.append(f"  Profit Factor:   {metrics.profit_factor:.2f}")
    lines.append(f"  Max DD:          {metrics.max_dd_pct:.2%}")
    lines.append(f"  Net Return:      {metrics.net_return_pct:.1f}%")
    lines.append(f"  Sharpe:          {metrics.sharpe:.2f}")
    lines.append(f"  Calmar:          {metrics.calmar:.1f}")
    lines.append(f"  Bear Alpha:      {metrics.bear_alpha_pct:.1f}%")
    lines.append(f"  Trade Latency:   {metrics.detection_latency_days:.1f}d")
    lines.append(f"  Bias Latency:    {metrics.bias_latency_days:.1f}d")
    lines.append(f"  Crisis Coverage: {metrics.crisis_coverage:.2f}")
    lines.append(f"  Exit Efficiency: {metrics.exit_efficiency:.3f}")

    # Config summary
    lines.append("\n--- R6 Optimised Parameters (unchanged) ---")
    lines.append(f"  S1 Pullback: ENABLED")
    lines.append(f"  Scale-out: pct={cfg.scale_out_pct}, target_r={cfg.scale_out_target_r}")
    lines.append(f"  BD Donchian: {cfg.bd_donchian_period}")
    lines.append(f"  LH Swing Lookback: {cfg.lh_swing_lookback}")
    lines.append(f"  Profit Floor Scale: {cfg.profit_floor_scale}")
    lines.append(f"  QQQ stop_floor_atr: {cfg.symbol_configs['QQQ'].stop_floor_atr}")
    lines.append(f"  GLD adx_on/off: {cfg.symbol_configs['GLD'].adx_on}/{cfg.symbol_configs['GLD'].adx_off}")
    lines.append(f"  GLD stop_buffer_atr: {cfg.symbol_configs['GLD'].stop_buffer_atr}")

    # CRISIS_WINDOWS
    lines.append("\n--- Crisis Windows (post-COVID removal) ---")
    for cname, cstart, cend in CRISIS_WINDOWS:
        lines.append(f"  {cname}: {cstart.date()} to {cend.date()}")

    # Standard diagnostics report
    lines.append("\n" + diag.report)

    # Per-symbol breakdown
    lines.append("\n--- Per-Symbol Breakdown ---")
    for sym, sr in result.symbol_results.items():
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

    # Crisis state log analysis
    crisis_analysis = analyse_crisis_state_logs(result, cfg)
    lines.append(crisis_analysis)

    # Path G analysis
    path_g_analysis = analyse_path_g_potential(result, cfg)
    lines.append(path_g_analysis)

    # Trade timeline
    trade_analysis = analyse_trades(result, cfg)
    lines.append(trade_analysis)

    full_report = "\n".join(lines)

    # Save
    out_path = OUTPUT_DIR / "round7_1_full_diagnostics.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_report, encoding="utf-8")

    print(full_report)
    print(f"\n\nSaved to {out_path}")
