"""Portfolio Optimized v2 -- 4-way equal-weight allocation.

25% Active strategies (ATRSS + Helix + Breakout, no overlay)
25% S6 EMA Crossover (ema_8_21_tight)
25% S5 Keltner Momentum (pb_k10_roc5_s15)
25% S5 Keltner Momentum (lo_dual_k15)

Usage:
    python run_portfolio_v2.py [--equity 10000]
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- Active strategies (unified portfolio engine) ---
from backtest.config_unified import UnifiedBacktestConfig
from backtest.engine.unified_portfolio_engine import (
    load_unified_data,
    run_unified,
)

# --- S5 Keltner Momentum ---
from backtest.config_s5 import S5BacktestConfig
from backtest.run_s5 import load_daily_data as load_s5_data, run_variant as run_s5_variant

# --- S6 EMA Crossover ---
from backtest.config_s6 import S6BacktestConfig
from backtest.run_s6 import load_daily_data as load_s6_data, run_variant as run_s6_variant

logger = logging.getLogger(__name__)


def _compute_sharpe_daily(equity: np.ndarray) -> float:
    """Annualized Sharpe from daily equity curve."""
    if len(equity) < 2:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def _compute_sharpe_hourly(equity: np.ndarray) -> float:
    """Annualized Sharpe from hourly equity curve."""
    if len(equity) < 2:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 7))


def _compute_max_dd(equity: np.ndarray) -> tuple[float, float]:
    """Returns (max_dd_pct, max_dd_dollars)."""
    if len(equity) < 2:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100
    dd_dollars = equity - peak
    return float(np.min(dd_pct)), float(np.min(dd_dollars))


def _sample_hourly_to_daily(
    equity: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, list]:
    """Sample hourly equity curve at end-of-day boundaries."""
    daily_eq = []
    daily_dates = []
    prev_date = None
    last_eq = equity[0]

    for i in range(len(timestamps)):
        ts = timestamps[i]
        dt = pd.Timestamp(ts)
        d = dt.date()
        if prev_date is not None and d != prev_date:
            daily_eq.append(last_eq)
            daily_dates.append(prev_date)
        last_eq = equity[i]
        prev_date = d

    # Final day
    if prev_date is not None:
        daily_eq.append(last_eq)
        daily_dates.append(prev_date)

    return np.array(daily_eq), daily_dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Optimized v2 backtest")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--data-dir", type=str, default="backtest/data/raw")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    total_equity = args.equity
    slot_equity = total_equity / 4.0
    data_dir = Path(args.data_dir)

    print(f"\n{'=' * 70}")
    print(f"PORTFOLIO OPTIMIZED v2 -- 4-Way Equal-Weight Allocation")
    print(f"{'=' * 70}")
    print(f"Total Equity:    ${total_equity:,.2f}")
    print(f"Per-Slot Equity: ${slot_equity:,.2f}")
    print(f"  Slot 1: Active Strategies (ATRSS + Helix + Breakout, no overlay)")
    print(f"  Slot 2: S6 EMA Crossover (ema_8_21_tight)")
    print(f"  Slot 3: S5 Keltner (pb_k10_roc5_s15)")
    print(f"  Slot 4: S5 Keltner (lo_dual_k15)")
    print(f"{'=' * 70}\n")

    # =========================================================================
    # SLOT 1: Active strategies (unified portfolio, overlay disabled)
    # =========================================================================
    print("=" * 50)
    print("SLOT 1: Active Strategies (ATRSS + Helix + Breakout)")
    print("=" * 50)
    t0 = time.perf_counter()

    active_config = UnifiedBacktestConfig(
        initial_equity=slot_equity,
        overlay_enabled=False,  # No EMA overlay - replaced by S5/S6
        data_dir=data_dir,
    )

    active_data = load_unified_data(active_config)
    active_result = run_unified(active_data, active_config)

    active_elapsed = time.perf_counter() - t0
    active_pnl = sum(sr.total_pnl for sr in active_result.strategy_results.values())
    active_trades = sum(sr.total_trades for sr in active_result.strategy_results.values())
    active_sharpe = _compute_sharpe_hourly(active_result.combined_equity)
    active_dd_pct, active_dd_dollars = _compute_max_dd(active_result.combined_equity)

    print(f"  Trades: {active_trades}, PnL: ${active_pnl:+,.2f}, "
          f"Sharpe: {active_sharpe:.2f}, MaxDD: {active_dd_pct:.2f}% ({active_elapsed:.1f}s)")

    for sid in ["ATRSS", "AKC_HELIX", "SWING_BREAKOUT_V3"]:
        sr = active_result.strategy_results.get(sid)
        if sr and sr.total_trades > 0:
            wr = sr.winning_trades / sr.total_trades * 100
            print(f"    {sid:<22} {sr.total_trades:>4} trades, "
                  f"WR {wr:.1f}%, PnL ${sr.total_pnl:+,.2f}, R {sr.total_r:+.1f}")

    # =========================================================================
    # SLOT 2: S6 EMA Crossover (ema_8_21_tight)
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("SLOT 2: S6 EMA Crossover (ema_8_21_tight)")
    print("=" * 50)
    t0 = time.perf_counter()

    s6_data = load_s6_data(["QQQ", "GLD", "IBIT"], data_dir)
    s6_config = S6BacktestConfig(initial_equity=slot_equity)
    s6_result = run_s6_variant("ema_8_21_tight", s6_config, s6_data)

    s6_elapsed = time.perf_counter() - t0
    print(f"  Trades: {s6_result.total_trades}, PnL: ${s6_result.total_pnl:+,.2f}, "
          f"Sharpe: {s6_result.sharpe:.2f}, WR: {s6_result.win_rate:.1f}%, "
          f"MaxDD: {s6_result.max_drawdown_pct:.2f}% ({s6_elapsed:.1f}s)")
    for sym, stats in sorted(s6_result.per_symbol.items()):
        print(f"    {sym:<6} {stats['trades']:>3} trades, WR {stats['win_rate']:.1f}%, "
              f"PnL ${stats['pnl']:+,.2f}")

    # =========================================================================
    # SLOT 3: S5 Keltner (pb_k10_roc5_s15)
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("SLOT 3: S5 Keltner (pb_k10_roc5_s15)")
    print("=" * 50)
    t0 = time.perf_counter()

    s5_data = load_s5_data(["QQQ", "GLD", "IBIT"], data_dir)
    s5_pb_config = S5BacktestConfig(
        initial_equity=slot_equity,
        entry_mode="pullback",
        kelt_ema_period=10,
        roc_period=5,
        atr_stop_mult=1.5,
    )
    s5_pb_result = run_s5_variant("pb_k10_roc5_s15", s5_pb_config, s5_data)

    s5_pb_elapsed = time.perf_counter() - t0
    print(f"  Trades: {s5_pb_result.total_trades}, PnL: ${s5_pb_result.total_pnl:+,.2f}, "
          f"Sharpe: {s5_pb_result.sharpe:.2f}, WR: {s5_pb_result.win_rate:.1f}%, "
          f"MaxDD: {s5_pb_result.max_drawdown_pct:.2f}% ({s5_pb_elapsed:.1f}s)")
    for sym, stats in sorted(s5_pb_result.per_symbol.items()):
        print(f"    {sym:<6} {stats['trades']:>3} trades, WR {stats['win_rate']:.1f}%, "
              f"PnL ${stats['pnl']:+,.2f}")

    # =========================================================================
    # SLOT 4: S5 Keltner (lo_dual_k15)
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("SLOT 4: S5 Keltner (lo_dual_k15)")
    print("=" * 50)
    t0 = time.perf_counter()

    s5_dual_config = S5BacktestConfig(
        initial_equity=slot_equity,
        entry_mode="dual",
        kelt_ema_period=15,
        shorts_enabled=False,
        rsi_entry_long=45.0,
    )
    s5_dual_result = run_s5_variant("lo_dual_k15", s5_dual_config, s5_data)

    s5_dual_elapsed = time.perf_counter() - t0
    print(f"  Trades: {s5_dual_result.total_trades}, PnL: ${s5_dual_result.total_pnl:+,.2f}, "
          f"Sharpe: {s5_dual_result.sharpe:.2f}, WR: {s5_dual_result.win_rate:.1f}%, "
          f"MaxDD: {s5_dual_result.max_drawdown_pct:.2f}% ({s5_dual_elapsed:.1f}s)")
    for sym, stats in sorted(s5_dual_result.per_symbol.items()):
        print(f"    {sym:<6} {stats['trades']:>3} trades, WR {stats['win_rate']:.1f}%, "
              f"PnL ${stats['pnl']:+,.2f}")

    # =========================================================================
    # COMBINED PORTFOLIO - merge equity curves at daily resolution
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("COMBINED PORTFOLIO RESULTS")
    print("=" * 70)

    # Sample active hourly equity to daily
    active_daily_eq, active_daily_dates = _sample_hourly_to_daily(
        active_result.combined_equity,
        active_result.combined_timestamps,
    )

    # S5/S6 equity curves are already daily; build from engine equity curves
    # We need to build a date-aligned combined equity curve
    # Strategy: use the daily bars timestamps from the data as the common timeline

    # Get daily timestamps from the data (use QQQ as reference)
    daily_times = s5_data["QQQ"].times
    n_daily = len(daily_times)
    daily_dates = [pd.Timestamp(t).date() if hasattr(t, 'item')
                   else pd.Timestamp(t).date() for t in daily_times]

    # Build per-slot daily equity arrays aligned to QQQ daily bars
    # Active: map from active_daily_dates to our daily_dates
    active_date_to_eq = dict(zip(active_daily_dates, active_daily_eq))
    active_aligned = np.full(n_daily, np.nan)
    for i, d in enumerate(daily_dates):
        if d in active_date_to_eq:
            active_aligned[i] = active_date_to_eq[d]

    # Forward-fill NaN gaps (weekends where hourly data has bars but daily doesn't,
    # or vice versa)
    last_valid = slot_equity
    for i in range(n_daily):
        if np.isnan(active_aligned[i]):
            active_aligned[i] = last_valid
        else:
            last_valid = active_aligned[i]

    # S6: reconstruct daily equity curve from the S6 engine
    # S6 engine equity curve is indexed by bar position
    s6_eq_curve = np.array([slot_equity])  # start
    for t in s6_result.all_trades:
        pass  # trades are already in total_pnl

    # Simpler approach: use the equity curves from the engines directly
    # The S5/S6 run_variant returns final_equity - we need the curve
    # Let me reconstruct from the S5Engine and S6Engine internal equity curves
    # Actually, let's re-run the engines to capture equity curves

    # Re-run S6 to capture equity curve
    from backtest.engine.s6_engine import S6Engine
    from strategy_6.config import SYMBOL_CONFIGS as S6_SYMBOL_CONFIGS

    s6_per_sym_curves = {}
    for sym in ["QQQ", "GLD", "IBIT"]:
        if sym not in s6_data:
            continue
        cfg = S6_SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        engine = S6Engine(symbol=sym, cfg=cfg, bt_config=s6_config, point_value=cfg.multiplier)
        engine.run(s6_data[sym])
        s6_per_sym_curves[sym] = np.array(engine.equity_curve)

    # Combine S6 equity curves (same approach as run_s6.py)
    s6_combined = None
    for sym, curve in s6_per_sym_curves.items():
        deltas = curve - slot_equity
        if s6_combined is None:
            s6_combined = slot_equity + deltas
        else:
            min_len = min(len(s6_combined), len(deltas))
            s6_combined = s6_combined[:min_len] + deltas[:min_len]

    # Re-run S5 pb to capture equity curve
    from backtest.engine.s5_engine import S5Engine
    from strategy_5.config import SYMBOL_CONFIGS as S5_SYMBOL_CONFIGS

    s5_pb_per_sym_curves = {}
    for sym in ["QQQ", "GLD", "IBIT"]:
        if sym not in s5_data:
            continue
        cfg = S5_SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        engine = S5Engine(symbol=sym, cfg=cfg, bt_config=s5_pb_config, point_value=cfg.multiplier)
        engine.run(s5_data[sym])
        s5_pb_per_sym_curves[sym] = np.array(engine.equity_curve)

    s5_pb_combined = None
    for sym, curve in s5_pb_per_sym_curves.items():
        deltas = curve - slot_equity
        if s5_pb_combined is None:
            s5_pb_combined = slot_equity + deltas
        else:
            min_len = min(len(s5_pb_combined), len(deltas))
            s5_pb_combined = s5_pb_combined[:min_len] + deltas[:min_len]

    # Re-run S5 dual to capture equity curve
    s5_dual_per_sym_curves = {}
    for sym in ["QQQ", "GLD", "IBIT"]:
        if sym not in s5_data:
            continue
        cfg = S5_SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        engine = S5Engine(symbol=sym, cfg=cfg, bt_config=s5_dual_config, point_value=cfg.multiplier)
        engine.run(s5_data[sym])
        s5_dual_per_sym_curves[sym] = np.array(engine.equity_curve)

    s5_dual_combined = None
    for sym, curve in s5_dual_per_sym_curves.items():
        deltas = curve - slot_equity
        if s5_dual_combined is None:
            s5_dual_combined = slot_equity + deltas
        else:
            min_len = min(len(s5_dual_combined), len(deltas))
            s5_dual_combined = s5_dual_combined[:min_len] + deltas[:min_len]

    # All S5/S6 curves are aligned to daily bars (same timestamps as QQQ daily)
    # Combine all 4 slots into a single portfolio equity curve at daily resolution
    # Use the shortest common length
    lens = [len(active_aligned)]
    if s6_combined is not None:
        lens.append(len(s6_combined))
    if s5_pb_combined is not None:
        lens.append(len(s5_pb_combined))
    if s5_dual_combined is not None:
        lens.append(len(s5_dual_combined))
    min_len = min(lens)

    # Portfolio equity = sum of (each slot's equity - slot_equity) + total_equity
    # This correctly accounts for each slot starting at slot_equity
    portfolio_eq = np.full(min_len, total_equity, dtype=float)
    portfolio_eq += (active_aligned[:min_len] - slot_equity)
    if s6_combined is not None:
        portfolio_eq += (s6_combined[:min_len] - slot_equity)
    if s5_pb_combined is not None:
        portfolio_eq += (s5_pb_combined[:min_len] - slot_equity)
    if s5_dual_combined is not None:
        portfolio_eq += (s5_dual_combined[:min_len] - slot_equity)

    # Portfolio metrics
    final_eq = portfolio_eq[-1]
    total_pnl = final_eq - total_equity
    total_return = (final_eq - total_equity) / total_equity * 100
    sharpe = _compute_sharpe_daily(portfolio_eq)
    max_dd_pct, max_dd_dollars = _compute_max_dd(portfolio_eq)
    total_trades = (active_trades + s6_result.total_trades +
                    s5_pb_result.total_trades + s5_dual_result.total_trades)

    # Profit factor across all trades
    all_trade_pnls = []
    for sr in active_result.strategy_results.values():
        pass  # We don't have individual trade PnLs from the unified result easily
    # Use the per-slot profit factors as a weighted average
    gross_profit = 0.0
    gross_loss = 0.0
    for result_obj in [s6_result, s5_pb_result, s5_dual_result]:
        for t in result_obj.all_trades:
            if t.pnl_dollars > 0:
                gross_profit += t.pnl_dollars
            else:
                gross_loss += abs(t.pnl_dollars)
    # Add active strategy trades
    for trades_list in [active_result.atrss_trades, active_result.helix_trades, active_result.breakout_trades]:
        for t in trades_list:
            if t.pnl_dollars > 0:
                gross_profit += t.pnl_dollars
            else:
                gross_loss += abs(t.pnl_dollars)
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate across all trades
    total_wins = (
        sum(sr.winning_trades for sr in active_result.strategy_results.values()) +
        s6_result.winning_trades + s5_pb_result.winning_trades + s5_dual_result.winning_trades
    )
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    # =========================================================================
    # Print combined report
    # =========================================================================
    print(f"Initial Equity:  ${total_equity:,.2f}")
    print(f"Final Equity:    ${final_eq:,.2f}")
    print(f"Total Return:    {total_return:+.2f}%")
    print(f"Total PnL:       ${total_pnl:+,.2f}")
    print(f"Max Drawdown:    {max_dd_pct:.2f}% (${max_dd_dollars:,.2f})")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.1f}%")
    print(f"Timeline:        {min_len:,} daily bars")

    # Per-slot breakdown
    print(f"\nPer-Slot Breakdown (${slot_equity:,.0f} each)")
    print(f"{'Slot':<40} {'Trades':>7} {'Win%':>6} {'PnL':>12} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print("-" * 95)

    # Active
    active_return = (active_aligned[min_len-1] - slot_equity) / slot_equity * 100
    active_wr = 0
    active_wins = sum(sr.winning_trades for sr in active_result.strategy_results.values())
    if active_trades > 0:
        active_wr = active_wins / active_trades * 100
    print(f"{'Active (ATRSS+Helix+Breakout)':<40} {active_trades:>7} {active_wr:>5.1f}% "
          f"${active_pnl:>10,.2f} {active_return:>7.2f}% {active_sharpe:>7.2f} {active_dd_pct:>6.2f}%")

    # S6
    s6_return = s6_result.total_return_pct
    s6_sharpe = s6_result.sharpe
    s6_dd = s6_result.max_drawdown_pct
    print(f"{'S6 EMA Crossover (ema_8_21_tight)':<40} {s6_result.total_trades:>7} {s6_result.win_rate:>5.1f}% "
          f"${s6_result.total_pnl:>10,.2f} {s6_return:>7.2f}% {s6_sharpe:>7.2f} {s6_dd:>6.2f}%")

    # S5 pb
    print(f"{'S5 Keltner (pb_k10_roc5_s15)':<40} {s5_pb_result.total_trades:>7} {s5_pb_result.win_rate:>5.1f}% "
          f"${s5_pb_result.total_pnl:>10,.2f} {s5_pb_result.total_return_pct:>7.2f}% "
          f"{s5_pb_result.sharpe:>7.2f} {s5_pb_result.max_drawdown_pct:>6.2f}%")

    # S5 dual
    print(f"{'S5 Keltner (lo_dual_k15)':<40} {s5_dual_result.total_trades:>7} {s5_dual_result.win_rate:>5.1f}% "
          f"${s5_dual_result.total_pnl:>10,.2f} {s5_dual_result.total_return_pct:>7.2f}% "
          f"{s5_dual_result.sharpe:>7.2f} {s5_dual_result.max_drawdown_pct:>6.2f}%")
    print("-" * 95)

    # Active strategies sub-breakdown
    print(f"\nActive Strategies Sub-Breakdown")
    print(f"{'Strategy':<22} {'Trades':>7} {'Win%':>6} {'PnL':>12} {'Total R':>8} {'Blocked':>8}")
    print("-" * 65)
    for sid in ["ATRSS", "AKC_HELIX", "SWING_BREAKOUT_V3"]:
        sr = active_result.strategy_results.get(sid)
        if sr is None:
            continue
        wr = sr.winning_trades / sr.total_trades * 100 if sr.total_trades > 0 else 0
        print(f"{sid:<22} {sr.total_trades:>7} {wr:>5.1f}% ${sr.total_pnl:>9,.2f} "
              f"{sr.total_r:>7.1f}R {sr.entries_blocked_by_heat:>8}")

    # Comparison with v1
    print(f"\n{'=' * 70}")
    print("COMPARISON: v2 (4-Way) vs v1 (Active + EMA Overlay)")
    print("=" * 70)
    print(f"{'Metric':<20} {'v2 (this run)':>15} {'v1 (reference)':>15}")
    print("-" * 55)
    print(f"{'Total PnL':<20} ${total_pnl:>+13,.2f} ${17521.88:>+13,.2f}")
    print(f"{'Return':<20} {total_return:>+14.2f}% {172.32:>+14.2f}%")
    print(f"{'Sharpe':<20} {sharpe:>15.2f} {1.15:>15.2f}")
    print(f"{'Max DD':<20} {max_dd_pct:>14.2f}% {-12.32:>14.2f}%")
    print(f"{'Trades':<20} {total_trades:>15} {286:>15}")
    print(f"{'Win Rate':<20} {win_rate:>14.1f}% {'N/A':>15}")
    print(f"\nv1 note: Active PnL $+4,213, Overlay PnL $+13,309 (85% equity EMA 10/21+13/21)")
    print("=" * 70)

    # =========================================================================
    # Save results to file
    # =========================================================================
    output_path = Path("backtest/output/portfolio_optimized_v2.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts_str} Portfolio Optimized v2 -- 4-Way Equal-Weight\n\n")

        f.write(f"{'=' * 70}\n")
        f.write(f"PORTFOLIO OPTIMIZED v2 -- COMBINED RESULTS\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Total Equity:    ${total_equity:,.2f}\n")
        f.write(f"Per-Slot Equity: ${slot_equity:,.2f}\n")
        f.write(f"Final Equity:    ${final_eq:,.2f}\n")
        f.write(f"Total Return:    {total_return:+.2f}%\n")
        f.write(f"Total PnL:       ${total_pnl:+,.2f}\n")
        f.write(f"Max Drawdown:    {max_dd_pct:.2f}% (${max_dd_dollars:,.2f})\n")
        f.write(f"Sharpe Ratio:    {sharpe:.2f}\n")
        f.write(f"Profit Factor:   {pf:.2f}\n")
        f.write(f"Total Trades:    {total_trades}\n")
        f.write(f"Win Rate:        {win_rate:.1f}%\n")
        f.write(f"Timeline:        {min_len:,} daily bars\n")

        f.write(f"\nAllocation: 25% Active / 25% S6 / 25% S5-PB / 25% S5-Dual\n")

        f.write(f"\nPer-Slot Breakdown (${slot_equity:,.0f} each)\n")
        f.write(f"{'Slot':<40} {'Trades':>7} {'Win%':>6} {'PnL':>12} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7}\n")
        f.write(f"{'-' * 95}\n")

        f.write(f"{'Active (ATRSS+Helix+Breakout)':<40} {active_trades:>7} {active_wr:>5.1f}% "
                f"${active_pnl:>10,.2f} {active_return:>7.2f}% {active_sharpe:>7.2f} {active_dd_pct:>6.2f}%\n")
        f.write(f"{'S6 EMA Crossover (ema_8_21_tight)':<40} {s6_result.total_trades:>7} {s6_result.win_rate:>5.1f}% "
                f"${s6_result.total_pnl:>10,.2f} {s6_return:>7.2f}% {s6_sharpe:>7.2f} {s6_dd:>6.2f}%\n")
        f.write(f"{'S5 Keltner (pb_k10_roc5_s15)':<40} {s5_pb_result.total_trades:>7} {s5_pb_result.win_rate:>5.1f}% "
                f"${s5_pb_result.total_pnl:>10,.2f} {s5_pb_result.total_return_pct:>7.2f}% "
                f"{s5_pb_result.sharpe:>7.2f} {s5_pb_result.max_drawdown_pct:>6.2f}%\n")
        f.write(f"{'S5 Keltner (lo_dual_k15)':<40} {s5_dual_result.total_trades:>7} {s5_dual_result.win_rate:>5.1f}% "
                f"${s5_dual_result.total_pnl:>10,.2f} {s5_dual_result.total_return_pct:>7.2f}% "
                f"{s5_dual_result.sharpe:>7.2f} {s5_dual_result.max_drawdown_pct:>6.2f}%\n")
        f.write(f"{'-' * 95}\n")

        f.write(f"\nActive Strategies Sub-Breakdown\n")
        f.write(f"{'Strategy':<22} {'Trades':>7} {'Win%':>6} {'PnL':>12} {'Total R':>8} {'Blocked':>8}\n")
        f.write(f"{'-' * 65}\n")
        for sid in ["ATRSS", "AKC_HELIX", "SWING_BREAKOUT_V3"]:
            sr = active_result.strategy_results.get(sid)
            if sr is None:
                continue
            wr = sr.winning_trades / sr.total_trades * 100 if sr.total_trades > 0 else 0
            f.write(f"{sid:<22} {sr.total_trades:>7} {wr:>5.1f}% ${sr.total_pnl:>9,.2f} "
                    f"{sr.total_r:>7.1f}R {sr.entries_blocked_by_heat:>8}\n")

        f.write(f"\n{'=' * 70}\n")
        f.write(f"COMPARISON: v2 vs v1\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"{'Metric':<20} {'v2 (4-Way)':>15} {'v1 (Overlay)':>15}\n")
        f.write(f"{'-' * 55}\n")
        f.write(f"{'Total PnL':<20} ${total_pnl:>+13,.2f} ${17521.88:>+13,.2f}\n")
        f.write(f"{'Return':<20} {total_return:>+14.2f}% {172.32:>+14.2f}%\n")
        f.write(f"{'Sharpe':<20} {sharpe:>15.2f} {1.15:>15.2f}\n")
        f.write(f"{'Max DD':<20} {max_dd_pct:>14.2f}% {-12.32:>14.2f}%\n")
        f.write(f"{'Trades':<20} {total_trades:>15} {286:>15}\n")
        f.write(f"{'Win Rate':<20} {win_rate:>14.1f}% {'N/A':>15}\n")
        f.write(f"{'=' * 70}\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
