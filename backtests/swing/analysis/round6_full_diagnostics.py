"""Round 6 Optimized BRS — Full Diagnostics + Recent No-Trade Analysis.

Run from project root:
    python research/backtests/swing/analysis/round6_full_diagnostics.py
"""
from __future__ import annotations
import sys, datetime as dt
from pathlib import Path

# --- bootstrap aliases -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.backtests.swing._aliases import install; install()

from research.backtests.swing.config_brs import BRSConfig
from research.backtests.swing.engine.brs_portfolio_engine import (
    load_brs_data, run_brs_independent,
)
from research.backtests.swing.analysis.brs_diagnostics import compute_brs_diagnostics
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Build Round 6 optimised config
# ---------------------------------------------------------------------------
def build_round6_config() -> BRSConfig:
    cfg = BRSConfig(data_dir=Path("research/backtests/swing/data/raw"))
    # Cumulative mutations from phase_state.json
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
    result = run_brs_independent(data, cfg)
    diag = compute_brs_diagnostics(
        result.symbol_results,
        cfg.initial_equity,
        result.combined_equity,
        result.combined_timestamps,
    )
    return result, diag, data


# ---------------------------------------------------------------------------
# 3. Analyse recent no-trade gap
# ---------------------------------------------------------------------------
def analyse_recent_gap(result, data, cfg):
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("RECENT NO-TRADE GAP ANALYSIS (last 2-3 weeks)")
    lines.append("=" * 70)

    # Collect all trades sorted by entry_time
    all_trades = []
    for sym, sr in result.symbol_results.items():
        for t in sr.trades:
            all_trades.append(t)
    all_trades.sort(key=lambda t: t.entry_time)

    if not all_trades:
        lines.append("No trades found at all.")
        return "\n".join(lines)

    last_trade = all_trades[-1]
    lines.append(f"\nLast trade:  {last_trade.symbol} {last_trade.entry_type}")
    lines.append(f"  Entry: {last_trade.entry_time}  Exit: {last_trade.exit_time}")
    lines.append(f"  R: {last_trade.r_multiple:+.2f}  Regime: {last_trade.regime_entry}")
    lines.append(f"  Exit reason: {last_trade.exit_reason}")

    # Show last 10 trades timeline
    lines.append(f"\n--- Last 15 trades timeline ---")
    lines.append(f"{'Sym':<5} {'Type':<20} {'Entry':<22} {'Exit':<22} {'R':>6} {'Regime':<15} {'ExitReason'}")
    for t in all_trades[-15:]:
        lines.append(
            f"{t.symbol:<5} {t.entry_type:<20} "
            f"{str(t.entry_time)[:19]:<22} {str(t.exit_time)[:19]:<22} "
            f"{t.r_multiple:>+5.2f}  {t.regime_entry:<15} {t.exit_reason}"
        )

    # Now analyze the regime state during the recent period
    lines.append(f"\n--- Regime State During Recent Period (Mar 1 - Mar 27, 2026) ---")

    for sym in cfg.symbols:
        sym_data = data.get(sym)
        if sym_data is None:
            continue

        lines.append(f"\n  [{sym}]")

        # Walk through the daily bars to get regime info
        d_times = sym_data.daily.times
        d_closes = sym_data.daily.closes
        d_highs = sym_data.daily.highs
        d_lows = sym_data.daily.lows

        h_times = sym_data.hourly.times
        h_closes = sym_data.hourly.closes
        h_highs = sym_data.hourly.highs
        h_lows = sym_data.hourly.lows

        # Find daily bars from Mar 1 onwards
        cutoff = np.datetime64("2026-03-01")
        recent_daily_idx = [i for i, t in enumerate(d_times) if t >= cutoff]

        if not recent_daily_idx:
            lines.append("    No daily bars in recent period")
            continue

        # Show daily price action
        lines.append(f"    Daily bars from {str(d_times[recent_daily_idx[0]])[:10]} to {str(d_times[recent_daily_idx[-1]])[:10]}:")
        lines.append(f"    {'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Chg%':>7}")
        prev_close = d_closes[recent_daily_idx[0] - 1] if recent_daily_idx[0] > 0 else d_closes[recent_daily_idx[0]]
        for i in recent_daily_idx:
            chg = (d_closes[i] / prev_close - 1) * 100
            lines.append(
                f"    {str(d_times[i])[:10]:<12} "
                f"{sym_data.daily.opens[i]:>8.2f} {d_highs[i]:>8.2f} "
                f"{d_lows[i]:>8.2f} {d_closes[i]:>8.2f} {chg:>+6.2f}%"
            )
            prev_close = d_closes[i]

        # Check regime conditions at each daily bar
        lines.append(f"\n    --- Regime Conditions Check (recent daily bars) ---")

        # Compute EMAs for regime
        ema_fast_period = cfg.ema_fast_period
        ema_slow_period = cfg.ema_slow_period
        adx_strong = cfg.adx_strong
        min_conds = cfg.regime_bear_min_conditions

        # Compute full EMA series
        def ema(arr, period):
            result = np.empty_like(arr, dtype=float)
            result[0] = arr[0]
            alpha = 2.0 / (period + 1)
            for j in range(1, len(arr)):
                result[j] = alpha * arr[j] + (1 - alpha) * result[j - 1]
            return result

        ema_f = ema(d_closes, ema_fast_period)
        ema_s = ema(d_closes, ema_slow_period)

        # Compute ADX (simplified 14-period)
        def compute_adx(highs, lows, closes, period=14):
            n = len(closes)
            atr = np.zeros(n)
            plus_dm = np.zeros(n)
            minus_dm = np.zeros(n)
            for j in range(1, n):
                tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
                atr[j] = tr
                up = highs[j] - highs[j-1]
                down = lows[j-1] - lows[j]
                plus_dm[j] = up if (up > down and up > 0) else 0
                minus_dm[j] = down if (down > up and down > 0) else 0

            # Smooth
            smoothed_atr = np.zeros(n)
            smoothed_pdm = np.zeros(n)
            smoothed_mdm = np.zeros(n)
            smoothed_atr[period] = np.sum(atr[1:period+1])
            smoothed_pdm[period] = np.sum(plus_dm[1:period+1])
            smoothed_mdm[period] = np.sum(minus_dm[1:period+1])
            for j in range(period+1, n):
                smoothed_atr[j] = smoothed_atr[j-1] - smoothed_atr[j-1]/period + atr[j]
                smoothed_pdm[j] = smoothed_pdm[j-1] - smoothed_pdm[j-1]/period + plus_dm[j]
                smoothed_mdm[j] = smoothed_mdm[j-1] - smoothed_mdm[j-1]/period + minus_dm[j]

            adx = np.zeros(n)
            dx = np.zeros(n)
            for j in range(period, n):
                if smoothed_atr[j] == 0:
                    continue
                pdi = 100 * smoothed_pdm[j] / smoothed_atr[j]
                mdi = 100 * smoothed_mdm[j] / smoothed_atr[j]
                if pdi + mdi == 0:
                    continue
                dx[j] = 100 * abs(pdi - mdi) / (pdi + mdi)
            # ADX = EMA of DX
            adx[2*period-1] = np.mean(dx[period:2*period])
            for j in range(2*period, n):
                adx[j] = (adx[j-1] * (period - 1) + dx[j]) / period
            return adx

        adx_arr = compute_adx(d_highs, d_lows, d_closes)

        # ATR-14 and ATR-50 for crash detection
        def compute_atr(highs, lows, closes, period):
            n = len(closes)
            tr = np.zeros(n)
            for j in range(1, n):
                tr[j] = max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1]))
            atr = np.zeros(n)
            atr[period] = np.mean(tr[1:period+1])
            for j in range(period+1, n):
                atr[j] = (atr[j-1] * (period-1) + tr[j]) / period
            return atr

        atr14 = compute_atr(d_highs, d_lows, d_closes, 14)
        atr50 = compute_atr(d_highs, d_lows, d_closes, 50)

        lines.append(f"    {'Date':<12} {'Close':>8} {'EMA'+str(ema_fast_period):>8} {'EMA'+str(ema_slow_period):>8} "
                     f"{'ADX':>6} {'ADX>'+str(adx_strong):>7} {'C<Ef':>5} {'Ef<Es':>5} {'DlyRet':>7} "
                     f"{'ATR14/50':>8} {'#Conds':>6} {'Regime'}")

        for i in recent_daily_idx:
            c = d_closes[i]
            ef = ema_f[i]
            es = ema_s[i]
            adx_val = adx_arr[i]
            daily_ret = (c / d_closes[i-1] - 1) if i > 0 else 0
            atr_ratio = atr14[i] / atr50[i] if atr50[i] > 0 else 0

            # Count bear conditions
            conds = 0
            c_below_ef = c < ef
            ef_below_es = ef < es
            adx_above = adx_val > adx_strong
            price_below_es = c < es

            if c_below_ef: conds += 1
            if ef_below_es: conds += 1
            if adx_above: conds += 1
            if price_below_es: conds += 1

            # Determine regime
            regime = "FLAT"
            if conds >= min_conds:
                regime = "BEAR_STRONG" if adx_above else "BEAR_TREND"

            # Check fast crash override
            crash_note = ""
            if cfg.fast_crash_enabled and daily_ret <= cfg.fast_crash_return_thresh and atr_ratio > cfg.fast_crash_atr_ratio:
                crash_note = " +CRASH_BIAS"
            if cfg.crash_override_enabled and daily_ret <= cfg.crash_override_return and atr_ratio > cfg.crash_override_atr_ratio:
                crash_note = " +CRASH_OVERRIDE"

            # Per-symbol ADX check (for entry eligibility)
            adx_on = cfg.symbol_configs[sym].adx_on
            adx_off = cfg.symbol_configs[sym].adx_off

            lines.append(
                f"    {str(d_times[i])[:10]:<12} {c:>8.2f} {ef:>8.2f} {es:>8.2f} "
                f"{adx_val:>6.1f} {'Y' if adx_above else 'N':>7} "
                f"{'Y' if c_below_ef else 'N':>5} {'Y' if ef_below_es else 'N':>5} "
                f"{daily_ret:>+6.2%} {atr_ratio:>8.2f} "
                f"{conds:>6} {regime}{crash_note}"
            )

        # Check hourly data for the recent period too
        lines.append(f"\n    --- Hourly Bias State (last 2 weeks) ---")
        h_cutoff = np.datetime64("2026-03-10")
        recent_h_idx = [i for i, t in enumerate(h_times) if t >= h_cutoff]

        if recent_h_idx:
            # Check 4H bars
            fh_times = sym_data.four_hour.times
            fh_closes = sym_data.four_hour.closes
            fh_cutoff_idx = [i for i, t in enumerate(fh_times) if t >= h_cutoff]

            if fh_cutoff_idx:
                # Compute 4H EMAs
                fh_ema_f = ema(fh_closes, ema_fast_period)
                fh_ema_s = ema(fh_closes, ema_slow_period)

                # Show daily summary of 4H regime
                from collections import defaultdict
                day_regimes = defaultdict(list)
                for i in fh_cutoff_idx:
                    fc = fh_closes[i]
                    fef = fh_ema_f[i]
                    fes = fh_ema_s[i]
                    bias_4h = "BEAR" if fc < fef and fef < fes else ("BULL" if fc > fef and fef > fes else "NEUTRAL")
                    day = str(fh_times[i])[:10]
                    day_regimes[day].append(bias_4h)

                lines.append(f"    {'Date':<12} {'4H Bias States'}")
                for day in sorted(day_regimes.keys()):
                    states = day_regimes[day]
                    lines.append(f"    {str(day):<12} {' → '.join(states)}")

    # Trade gap summary
    lines.append(f"\n--- Gap Diagnosis Summary ---")
    last_exit = all_trades[-1].exit_time
    data_end = dt.datetime(2026, 3, 27, tzinfo=dt.timezone.utc)
    gap_days = (data_end - last_exit).days
    lines.append(f"Last exit: {last_exit}")
    lines.append(f"Data end:  {data_end.date()}")
    lines.append(f"Gap: ~{gap_days} calendar days")

    # Count trades per year
    lines.append(f"\n--- Trade Distribution by Year ---")
    from collections import Counter
    year_counts = Counter(t.entry_time.year for t in all_trades)
    for y in sorted(year_counts):
        lines.append(f"  {y}: {year_counts[y]} trades")

    # Count trades per quarter
    lines.append(f"\n--- Trade Distribution by Quarter ---")
    q_counts = Counter(f"{t.entry_time.year}-Q{(t.entry_time.month-1)//3+1}" for t in all_trades)
    for q in sorted(q_counts):
        lines.append(f"  {q}: {q_counts[q]} trades")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = build_round6_config()
    result, diag, data = run_full(cfg)

    # Build full report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("ROUND 6 OPTIMISED BRS — FULL DIAGNOSTICS")
    report_lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    report_lines.append("=" * 70)

    # Config summary
    report_lines.append("\n--- Round 6 Optimised Parameters ---")
    report_lines.append(f"  S1 Pullback: ENABLED (was disabled)")
    report_lines.append(f"  Scale-out: pct={cfg.scale_out_pct}, target_r={cfg.scale_out_target_r}")
    report_lines.append(f"  BD Donchian: {cfg.bd_donchian_period} (was 20)")
    report_lines.append(f"  LH Swing Lookback: {cfg.lh_swing_lookback} (was 5)")
    report_lines.append(f"  Profit Floor Scale: {cfg.profit_floor_scale} (was 1.0)")
    report_lines.append(f"  QQQ stop_floor_atr: {cfg.symbol_configs['QQQ'].stop_floor_atr} (was 1.0)")
    report_lines.append(f"  GLD adx_on/off: {cfg.symbol_configs['GLD'].adx_on}/{cfg.symbol_configs['GLD'].adx_off} (was 18/16)")
    report_lines.append(f"  GLD stop_buffer_atr: {cfg.symbol_configs['GLD'].stop_buffer_atr} (was 0.27)")

    # Standard diagnostics
    report_lines.append("\n" + diag.report)

    # Per-symbol summary
    report_lines.append("\n--- Per-Symbol Breakdown ---")
    for sym, sr in result.symbol_results.items():
        trades = sr.trades
        wins = [t for t in trades if t.r_multiple > 0]
        losses = [t for t in trades if t.r_multiple <= 0]
        total_r = sum(t.r_multiple for t in trades)
        wr = len(wins) / len(trades) * 100 if trades else 0
        report_lines.append(f"\n  [{sym}] {len(trades)} trades, WR={wr:.1f}%, Total R={total_r:.1f}")
        report_lines.append(f"    Bias days — Short: {sr.bias_days_short}, Long: {sr.bias_days_long}, Flat: {sr.bias_days_flat}")
        if trades:
            by_type = {}
            for t in trades:
                by_type.setdefault(t.entry_type, []).append(t)
            for etype, tlist in sorted(by_type.items()):
                wr_e = sum(1 for t in tlist if t.r_multiple > 0) / len(tlist) * 100
                avg_r = np.mean([t.r_multiple for t in tlist])
                report_lines.append(f"    {etype}: {len(tlist)} trades, WR={wr_e:.0f}%, AvgR={avg_r:.2f}")

    # Recent gap analysis
    gap_report = analyse_recent_gap(result, data, cfg)
    report_lines.append(gap_report)

    full_report = "\n".join(report_lines)

    # Save to file
    out_path = Path("research/backtests/swing/auto/brs/output/round6_diagnostics.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_report, encoding="utf-8")

    # Print to stdout (safe for Windows consoles)
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(full_report)
    print(f"\n\nSaved to {out_path}")
