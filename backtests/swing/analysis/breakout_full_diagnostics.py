"""Breakout v3.3-ETF Full Diagnostics -- comprehensive analysis with optimized config.

Runs the Breakout backtest with greedy-portfolio-optimized config and generates
a complete diagnostic report using all diagnostic functions.

Optimized mutations (from greedy_portfolio_optimal.json):
  - disable_regime_chop_block: True  (round 5)
  - disable_score_threshold: True    (round 7)

Usage:
    python -u backtests/swing/analysis/breakout_full_diagnostics.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Setup path and aliases
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing.analysis.breakout_diagnostics import (
    breakout_add_analysis,
    breakout_chop_impact,
    breakout_continuation_analysis,
    breakout_crisis_window_analysis,
    breakout_entry_drilldown,
    breakout_exit_entry_crosstab,
    breakout_exit_reason_breakdown,
    breakout_exit_tier_drilldown,
    breakout_gap_stop_report,
    breakout_loser_classification,
    breakout_mfe_cohort_segmentation,
    breakout_monthly_returns,
    breakout_partial_fill_analysis,
    breakout_partial_timing_analysis,
    breakout_position_occupancy,
    breakout_profit_concentration,
    breakout_quality_mult_decomposition,
    breakout_r_curve,
    breakout_regime_breakdown,
    breakout_regime_direction_grid,
    breakout_regime_transition,
    breakout_rolling_edge,
    breakout_score_component_analysis,
    breakout_stale_analysis,
    breakout_stop_evolution,
    breakout_streak_analysis,
    breakout_exit_efficiency,
)
from backtests.swing.analysis.metrics import (
    compute_buy_and_hold,
    compute_max_drawdown,
    compute_metrics,
    compute_sharpe,
    compute_sortino,
)
from backtests.swing.analysis.reports import (
    breakout_behavior_report,
    breakout_diagnostic_report,
    breakout_performance_report,
    buy_and_hold_report,
)
from backtests.swing.config import SlippageConfig
from backtests.swing.config_breakout import BreakoutAblationFlags, BreakoutBacktestConfig
from backtests.swing.engine.breakout_portfolio_engine import (
    load_breakout_data,
    run_breakout_synchronized,
)
from backtests.diagnostic_snapshot import build_group_snapshot
from backtests.swing.auto.config_mutator import mutate_breakout_config
from strategies.swing.breakout.config import SYMBOL_CONFIGS

import numpy as np

DATA_DIR = Path("backtests/swing/data/raw")
INITIAL_EQUITY = 10_000.0
DEFAULT_OUTPUT = Path("backtests/swing/auto/output/breakout_full_diagnostics.txt")

# Greedy-portfolio-optimized mutations (breakout-specific only)
OPTIMIZED_MUTATIONS = {
    "flags.disable_regime_chop_block": True,
    "flags.disable_score_threshold": True,
}


def _out(text: str, f=None) -> None:
    print(text)
    if f is not None:
        f.write(text + "\n")


def _trade_net_pnl(trade) -> float:
    return float(trade.pnl_dollars) - float(getattr(trade, "commission", 0.0) or 0.0)


def _net_profit_factor(trades: list) -> float:
    gross_profit = sum(_trade_net_pnl(t) for t in trades if _trade_net_pnl(t) > 0)
    gross_loss = abs(sum(_trade_net_pnl(t) for t in trades if _trade_net_pnl(t) < 0))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _path_signature(path: Path) -> str:
    if not path.exists():
        return f"{path.resolve()}|missing"
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"


def _cache_fingerprint(
    mutations: dict,
    equity: float,
    fixed_qty: int,
    symbols: list[str],
    data_dir: Path,
) -> str:
    """Compute a deterministic fingerprint for result caching."""
    import hashlib
    source_paths = [
        Path(__file__),
        Path("backtests/swing/analysis/metrics.py"),
        Path("backtests/swing/engine/breakout_engine.py"),
        Path("backtests/swing/engine/breakout_portfolio_engine.py"),
        Path("strategies/swing/breakout/engine.py"),
    ]
    data_paths = []
    for sym in symbols:
        data_paths.extend([data_dir / f"{sym}_1h.parquet", data_dir / f"{sym}_1d.parquet"])
    source_sig = "|".join(_path_signature(path) for path in source_paths)
    data_sig = "|".join(_path_signature(path) for path in data_paths)
    key = (
        f"synchronized|{sorted(mutations.items())}|{equity}|{fixed_qty}|{sorted(symbols)}|"
        f"{source_sig}|{data_sig}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:16]


def main():
    import pickle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Base config
    base_config = BreakoutBacktestConfig(
        symbols=["QQQ", "GLD"],
        initial_equity=INITIAL_EQUITY,
        data_dir=DATA_DIR,
        slippage=SlippageConfig(commission_per_contract=1.00),
        fixed_qty=10,
        flags=BreakoutAblationFlags(),
    )

    # Apply optimized mutations
    config = mutate_breakout_config(base_config, OPTIMIZED_MUTATIONS)

    print(f"Running Breakout v3.3-ETF backtest: symbols={config.symbols}, "
          f"equity=${config.initial_equity:,.0f}, fixed_qty={config.fixed_qty}")
    print(f"Optimized mutations ({len(OPTIMIZED_MUTATIONS)}):")
    for k, v in OPTIMIZED_MUTATIONS.items():
        print(f"  {k} = {v}")

    # Check result cache
    cache_dir = output_path.parent
    fingerprint = _cache_fingerprint(
        OPTIMIZED_MUTATIONS,
        INITIAL_EQUITY,
        10,
        config.symbols,
        DATA_DIR,
    )
    result_cache_path = cache_dir / "breakout_result_cache.pkl"
    result = None
    all_trades = None

    if result_cache_path.exists():
        try:
            with open(result_cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                result = cached["result"]
                all_trades = cached["all_trades"]
                print(f"[CACHED] Loaded result from cache")
        except Exception:
            pass

    if result is None:
        print(f"\nLoading data from {DATA_DIR}...")
        data = load_breakout_data(config.symbols, config.data_dir)
        load_elapsed = time.time() - t0
        print(f"Data loaded in {load_elapsed:.1f}s")

        print("Running backtest...")
        t1 = time.time()
        result = run_breakout_synchronized(data, config)
        engine_elapsed = time.time() - t1
        print(f"Engine completed in {engine_elapsed:.1f}s")

        # Collect all trades
        all_trades = []
        for sym, sr in result.symbol_results.items():
            for t in sr.trades:
                t.symbol = sym
                all_trades.append(t)
        all_trades.sort(key=lambda t: t.entry_time or datetime.min)

        # Cache result
        try:
            with open(result_cache_path, "wb") as f:
                pickle.dump({"fingerprint": fingerprint, "result": result,
                             "all_trades": all_trades}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    else:
        load_elapsed = 0.0
        engine_elapsed = 0.0
        # Load data for buy-and-hold comparison (uses pickle cache, fast)
        data = load_breakout_data(config.symbols, config.data_dir)

    print(f"Total trades: {len(all_trades)}")
    snapshot = build_group_snapshot(
        "Breakout Strength / Weakness Snapshot",
        all_trades,
        [
            ("symbol", lambda trade: getattr(trade, "symbol", None)),
            ("direction", lambda trade: "LONG" if getattr(trade, "direction", 0) == 1 else "SHORT"),
            ("exit reason", lambda trade: getattr(trade, "exit_reason", None)),
        ],
        min_count=3,
        width=80,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        _out("=" * 80, f)
        _out("BREAKOUT v3.3-ETF FULL DIAGNOSTICS (GREEDY-PORTFOLIO OPTIMIZED)", f)
        _out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f)
        _out(f"Initial Equity: ${INITIAL_EQUITY:,.0f}", f)
        _out(f"Symbols: {', '.join(config.symbols)}", f)
        _out(f"Commission: $1.00/contract, fixed_qty=10", f)
        _out(f"Data load: {load_elapsed:.1f}s, Engine: {engine_elapsed:.1f}s", f)
        _out("", f)
        _out("Optimized mutations:", f)
        for mk, mv in sorted(OPTIMIZED_MUTATIONS.items()):
            _out(f"  {mk}: {mv}", f)
        _out("=" * 80, f)
        _out("", f)
        _out(snapshot, f)
        _out("", f)

        # ==================================================================
        # AGGREGATE SUMMARY
        # ==================================================================
        if all_trades:
            total_pnl = sum(_trade_net_pnl(t) for t in all_trades)
            total_r = sum(t.r_multiple for t in all_trades)
            n_wins = sum(1 for t in all_trades if t.r_multiple > 0)
            wr = n_wins / len(all_trades) * 100

            eq = result.combined_equity

            pf = _net_profit_factor(all_trades)
            max_dd_pct, max_dd_dollar = compute_max_drawdown(eq) if len(eq) > 1 else (0, 0)

            sharpe = compute_sharpe(eq) if len(eq) > 1 else 0.0
            sortino = compute_sortino(eq) if len(eq) > 1 else 0.0

            return_pct = 100 * (eq[-1] - eq[0]) / eq[0] if len(eq) > 1 and eq[0] > 0 else 0.0

            _out("--- AGGREGATE SUMMARY ---", f)
            _out(f"  Total trades:    {len(all_trades)}", f)
            _out(f"  Win rate:        {wr:.1f}%", f)
            _out(f"  Profit factor:   {pf:.2f}", f)
            _out(f"  Total R:         {total_r:+.2f}", f)
            _out(f"  Total PnL:       ${total_pnl:+,.2f}", f)
            _out(f"  Return:          {return_pct:+.1f}%", f)
            _out(f"  Max drawdown:    {max_dd_pct:.2%} (${max_dd_dollar:,.2f})", f)
            _out(f"  Sharpe:          {sharpe:.2f}", f)
            _out(f"  Sortino:         {sortino:.2f}", f)
            _out(f"  Avg R:           {np.mean([t.r_multiple for t in all_trades]):+.3f}", f)
            _out(f"  Avg hold (bars): {np.mean([t.bars_held for t in all_trades]):.1f}", f)
            _out("", f)

        # ==================================================================
        # PER-SYMBOL SUMMARY TABLE
        # ==================================================================
        _out("=" * 80, f)
        _out("PER-SYMBOL SUMMARY", f)
        _out("=" * 80, f)
        _out(f"  {'Symbol':>6s} {'N':>5s} {'WR':>5s} {'PF':>6s} {'AvgR':>7s} "
             f"{'TotR':>8s} {'PnL':>10s} {'MaxDD':>7s} {'Sharpe':>7s}", f)
        _out("  " + "-" * 70, f)

        for sym in sorted(result.symbol_results):
            sr = result.symbol_results[sym]
            trades = sr.trades
            if not trades:
                _out(f"  {sym:>6s}    -- no trades --", f)
                continue
            n = len(trades)
            wr_s = sum(1 for t in trades if t.r_multiple > 0) / n * 100
            pf_s = _net_profit_factor(trades)
            avg_r = np.mean([t.r_multiple for t in trades])
            tot_r = sum(t.r_multiple for t in trades)
            pnl_s = sum(_trade_net_pnl(t) for t in trades)
            dd_pct, _ = compute_max_drawdown(sr.equity_curve) if len(sr.equity_curve) > 1 else (0, 0)
            sh = compute_sharpe(sr.equity_curve) if len(sr.equity_curve) > 1 else 0.0
            _out(f"  {sym:>6s} {n:5d} {wr_s:4.0f}% {min(pf_s, 99.99):6.2f} {avg_r:+7.3f} "
                 f"{tot_r:+8.2f} ${pnl_s:+9,.0f} {dd_pct:6.2%} {sh:7.2f}", f)
        _out("", f)

        # ==================================================================
        # PER-SYMBOL DETAILED REPORTS
        # ==================================================================
        for sym in sorted(result.symbol_results):
            sr = result.symbol_results[sym]
            if not sr.trades:
                continue

            _out("=" * 80, f)
            _out(f"DETAILED REPORT: {sym}", f)
            _out("=" * 80, f)

            cfg = SYMBOL_CONFIGS.get(sym)
            multiplier = cfg.multiplier if cfg else 1.0

            pnls = np.array([t.pnl_dollars for t in sr.trades])
            risks = np.array([abs(t.entry_price - t.stop_price) * multiplier * t.qty
                              for t in sr.trades])
            holds = np.array([t.bars_held for t in sr.trades])
            comms = np.array([t.commission for t in sr.trades])

            metrics = compute_metrics(
                pnls, risks, holds, comms,
                sr.equity_curve, sr.timestamps, config.initial_equity,
            )

            _out(breakout_performance_report(sym, metrics), f)
            _out(breakout_behavior_report(sr.trades), f)
            _out(breakout_diagnostic_report(sr), f)

            # Buy & hold comparison
            if sym in data.daily:
                daily_closes = data.daily[sym].closes
                if len(sr.timestamps) >= 2 and len(daily_closes) >= 2:
                    delta = sr.timestamps[-1] - sr.timestamps[0]
                    if hasattr(delta, 'total_seconds'):
                        years = delta.total_seconds() / (365.25 * 24 * 3600)
                    elif isinstance(delta, np.timedelta64):
                        years = float(delta / np.timedelta64(1, 's')) / (365.25 * 24 * 3600)
                    else:
                        years = float(delta) / (365.25 * 24 * 3600)
                    qty = config.fixed_qty or 1
                    bh = compute_buy_and_hold(
                        sym, daily_closes, years,
                        qty=qty,
                        multiplier=multiplier,
                        initial_equity=config.initial_equity,
                    )
                    _out(buy_and_hold_report(bh, metrics), f)

            # Signal events funnel (per-symbol)
            if sr.signal_events:
                total_signals = len(sr.signal_events)
                allowed = sum(1 for s in sr.signal_events if s.allowed)
                blocked = total_signals - allowed
                _out(f"\n  Signal funnel ({sym}): {total_signals} evaluated -> "
                     f"{allowed} allowed ({100*allowed/total_signals:.0f}%) / "
                     f"{blocked} blocked ({100*blocked/total_signals:.0f}%)", f)
                if blocked:
                    reasons = Counter(s.blocked_reason for s in sr.signal_events if not s.allowed)
                    for reason, cnt in reasons.most_common(10):
                        _out(f"    {reason:30s}: {cnt:5d} ({100*cnt/blocked:.0f}%)", f)

            _out("", f)

        # ==================================================================
        # AGGREGATE DIAGNOSTICS (all trades pooled)
        # ==================================================================
        _out("=" * 80, f)
        _out("AGGREGATE DIAGNOSTICS (all symbols pooled)", f)
        _out("=" * 80, f)
        _out("", f)

        _out(breakout_entry_drilldown(all_trades), f)
        _out("", f)
        _out(breakout_exit_tier_drilldown(all_trades), f)
        _out("", f)
        _out(breakout_regime_breakdown(all_trades), f)
        _out("", f)
        _out(breakout_exit_reason_breakdown(all_trades), f)
        _out("", f)
        _out(breakout_partial_fill_analysis(all_trades), f)
        _out("", f)
        _out(breakout_gap_stop_report(all_trades), f)
        _out("", f)
        _out(breakout_add_analysis(all_trades), f)
        _out("", f)
        _out(breakout_chop_impact(all_trades), f)
        _out("", f)
        _out(breakout_stale_analysis(all_trades), f)
        _out("", f)
        _out(breakout_streak_analysis(all_trades), f)
        _out("", f)
        _out(breakout_continuation_analysis(all_trades), f)
        _out("", f)
        _out(breakout_loser_classification(all_trades), f)
        _out("", f)
        _out(breakout_exit_entry_crosstab(all_trades), f)
        _out("", f)
        _out(breakout_regime_direction_grid(all_trades), f)
        _out("", f)
        _out(breakout_mfe_cohort_segmentation(all_trades), f)
        _out("", f)
        _out(breakout_position_occupancy(all_trades), f)
        _out("", f)
        _out(breakout_score_component_analysis(all_trades), f)
        _out("", f)
        _out(breakout_quality_mult_decomposition(all_trades), f)
        _out("", f)
        _out(breakout_partial_timing_analysis(all_trades), f)
        _out("", f)
        _out(breakout_stop_evolution(all_trades), f)
        _out("", f)
        _out(breakout_regime_transition(all_trades), f)
        _out("", f)

        # ==================================================================
        # ADVANCED DIAGNOSTICS (new functions)
        # ==================================================================
        _out("=" * 80, f)
        _out("ADVANCED DIAGNOSTICS", f)
        _out("=" * 80, f)
        _out("", f)

        _out(breakout_monthly_returns(all_trades), f)
        _out("", f)
        _out(breakout_rolling_edge(all_trades), f)
        _out("", f)
        _out(breakout_r_curve(all_trades), f)
        _out("", f)
        _out(breakout_crisis_window_analysis(all_trades), f)
        _out("", f)
        _out(breakout_profit_concentration(all_trades), f)
        _out("", f)
        _out(breakout_exit_efficiency(all_trades), f)
        _out("", f)

        # ==================================================================
        _out("=" * 80, f)
        _out("BREAKOUT DIAGNOSTICS COMPLETE", f)
        _out("=" * 80, f)

    total_elapsed = time.time() - t0
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")
    print(f"Saved to {output_path}")

    # Save JSON summary
    json_path = output_path.with_suffix(".json")
    if all_trades:
        eq = result.combined_equity
        max_dd_pct, _ = compute_max_drawdown(eq) if len(eq) > 1 else (0, 0)
        summary = {
            "strategy": "breakout_v3.3_etf",
            "mutations": OPTIMIZED_MUTATIONS,
            "initial_equity": INITIAL_EQUITY,
            "total_trades": len(all_trades),
            "win_rate": sum(1 for t in all_trades if t.r_multiple > 0) / len(all_trades),
            "profit_factor": float(_net_profit_factor(all_trades)),
            "total_r": sum(t.r_multiple for t in all_trades),
            "total_pnl": sum(_trade_net_pnl(t) for t in all_trades),
            "return_pct": float(100 * (eq[-1] - eq[0]) / eq[0]) if len(eq) > 1 else 0,
            "max_drawdown_pct": float(max_dd_pct),
            "sharpe": float(compute_sharpe(eq)) if len(eq) > 1 else 0,
            "sortino": float(compute_sortino(eq)) if len(eq) > 1 else 0,
            "avg_r": float(np.mean([t.r_multiple for t in all_trades])),
            "avg_hold_bars": float(np.mean([t.bars_held for t in all_trades])),
            "per_symbol": {},
        }
        for sym, sr in result.symbol_results.items():
            if sr.trades:
                summary["per_symbol"][sym] = {
                    "trades": len(sr.trades),
                    "total_r": sum(t.r_multiple for t in sr.trades),
                    "total_pnl": sum(_trade_net_pnl(t) for t in sr.trades),
                }
        with open(json_path, "w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
