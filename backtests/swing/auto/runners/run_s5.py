"""Strategy 5 (Keltner Momentum Breakout) backtest runner with variant sweep.

Usage:
    python -m backtest.run_s5 --run-all                    # all variants
    python -m backtest.run_s5 --group A                    # breakout + dual
    python -m backtest.run_s5 --group B                    # pullback
    python -m backtest.run_s5 --group C                    # momentum + tuned
    python -m backtest.run_s5 --variant bo_k20_r14_s20     # single variant
"""
from __future__ import annotations

from backtests.swing._aliases import install; install()

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.config_s5 import S5BacktestConfig
from backtest.data.preprocessing import NumpyBars, build_numpy_arrays
from backtest.engine.s5_engine import S5Engine, S5TradeRecord
from strategy_4.config import SYMBOL_CONFIGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GROUP A: Breakout + Dual entry (12 variants)
# ---------------------------------------------------------------------------

def make_bo_k10_r7_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=10, rsi_period=7, atr_stop_mult=2.0)
def make_bo_k10_r14_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=10, rsi_period=14, atr_stop_mult=2.0)
def make_bo_k20_r7_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=20, rsi_period=7, atr_stop_mult=2.0)
def make_bo_k20_r14_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=20, rsi_period=14, atr_stop_mult=2.0)
def make_bo_k10_wide(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=10, kelt_atr_mult=2.5, atr_stop_mult=2.5)
def make_bo_k20_wide(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=20, kelt_atr_mult=2.5, atr_stop_mult=2.5)
def make_dual_k10_r7(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=10, rsi_period=7, rsi_entry_long=45.0, rsi_entry_short=55.0)
def make_dual_k15_r7(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=15, rsi_period=7, rsi_entry_long=45.0, rsi_entry_short=55.0)
def make_dual_k20_r7(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=20, rsi_period=7, rsi_entry_long=45.0, rsi_entry_short=55.0)
def make_dual_k10_r14(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=10, rsi_period=14, rsi_entry_long=45.0, rsi_entry_short=55.0)
def make_dual_k15_r14(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=15, rsi_period=14, rsi_entry_long=45.0, rsi_entry_short=55.0)
def make_dual_k20_r14(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=20, rsi_period=14, rsi_entry_long=45.0, rsi_entry_short=55.0)


# ---------------------------------------------------------------------------
# GROUP B: Pullback entry (12 variants)
# ---------------------------------------------------------------------------

def make_pb_k10_roc5_s15(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=10, roc_period=5, atr_stop_mult=1.5)
def make_pb_k10_roc5_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=10, roc_period=5, atr_stop_mult=2.0)
def make_pb_k10_roc10_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=10, roc_period=10, atr_stop_mult=2.0)
def make_pb_k20_roc5_s15(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=20, roc_period=5, atr_stop_mult=1.5)
def make_pb_k20_roc5_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=20, roc_period=5, atr_stop_mult=2.0)
def make_pb_k20_roc10_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=20, roc_period=10, atr_stop_mult=2.0)
def make_pb_k30_roc5_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=30, roc_period=5, atr_stop_mult=2.0)
def make_pb_k30_roc10_s20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=30, roc_period=10, atr_stop_mult=2.0)
def make_pb_k10_rsi40(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=10, rsi_entry_long=40.0, rsi_entry_short=60.0)
def make_pb_k20_rsi40(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=20, rsi_entry_long=40.0, rsi_entry_short=60.0)
def make_pb_k10_novol(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=10, volume_filter=False)
def make_pb_k20_novol(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="pullback", kelt_ema_period=20, volume_filter=False)


# ---------------------------------------------------------------------------
# GROUP C: Momentum + Long-only + Exit-mode experiments (12 variants)
# ---------------------------------------------------------------------------

def make_mo_k10_r7_roc5(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="momentum", kelt_ema_period=10, rsi_period=7, roc_period=5)
def make_mo_k10_r14_roc10(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="momentum", kelt_ema_period=10, rsi_period=14, roc_period=10)
def make_mo_k20_r7_roc5(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="momentum", kelt_ema_period=20, rsi_period=7, roc_period=5)
def make_mo_k20_r14_roc10(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="momentum", kelt_ema_period=20, rsi_period=14, roc_period=10)
def make_mo_k15_r10_roc7(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="momentum", kelt_ema_period=15, rsi_period=10, roc_period=7)
def make_mo_k15_r10_lo(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="momentum", kelt_ema_period=15, rsi_period=10, roc_period=7, shorts_enabled=False)
def make_rev_k10_bo(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=10, exit_mode="reversal")
def make_rev_k20_bo(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=20, exit_mode="reversal")
def make_mid_k15_dual(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=15, exit_mode="midline")
def make_lo_bo_k10(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=10, shorts_enabled=False)
def make_lo_bo_k20(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="breakout", kelt_ema_period=20, shorts_enabled=False)
def make_lo_dual_k15(eq): return S5BacktestConfig(initial_equity=eq, entry_mode="dual", kelt_ema_period=15, shorts_enabled=False, rsi_entry_long=45.0)


VARIANTS_A: dict[str, callable] = {
    "bo_k10_r7_s20": make_bo_k10_r7_s20,
    "bo_k10_r14_s20": make_bo_k10_r14_s20,
    "bo_k20_r7_s20": make_bo_k20_r7_s20,
    "bo_k20_r14_s20": make_bo_k20_r14_s20,
    "bo_k10_wide": make_bo_k10_wide,
    "bo_k20_wide": make_bo_k20_wide,
    "dual_k10_r7": make_dual_k10_r7,
    "dual_k15_r7": make_dual_k15_r7,
    "dual_k20_r7": make_dual_k20_r7,
    "dual_k10_r14": make_dual_k10_r14,
    "dual_k15_r14": make_dual_k15_r14,
    "dual_k20_r14": make_dual_k20_r14,
}

VARIANTS_B: dict[str, callable] = {
    "pb_k10_roc5_s15": make_pb_k10_roc5_s15,
    "pb_k10_roc5_s20": make_pb_k10_roc5_s20,
    "pb_k10_roc10_s20": make_pb_k10_roc10_s20,
    "pb_k20_roc5_s15": make_pb_k20_roc5_s15,
    "pb_k20_roc5_s20": make_pb_k20_roc5_s20,
    "pb_k20_roc10_s20": make_pb_k20_roc10_s20,
    "pb_k30_roc5_s20": make_pb_k30_roc5_s20,
    "pb_k30_roc10_s20": make_pb_k30_roc10_s20,
    "pb_k10_rsi40": make_pb_k10_rsi40,
    "pb_k20_rsi40": make_pb_k20_rsi40,
    "pb_k10_novol": make_pb_k10_novol,
    "pb_k20_novol": make_pb_k20_novol,
}

VARIANTS_C: dict[str, callable] = {
    "mo_k10_r7_roc5": make_mo_k10_r7_roc5,
    "mo_k10_r14_roc10": make_mo_k10_r14_roc10,
    "mo_k20_r7_roc5": make_mo_k20_r7_roc5,
    "mo_k20_r14_roc10": make_mo_k20_r14_roc10,
    "mo_k15_r10_roc7": make_mo_k15_r10_roc7,
    "mo_k15_r10_lo": make_mo_k15_r10_lo,
    "rev_k10_bo": make_rev_k10_bo,
    "rev_k20_bo": make_rev_k20_bo,
    "mid_k15_dual": make_mid_k15_dual,
    "lo_bo_k10": make_lo_bo_k10,
    "lo_bo_k20": make_lo_bo_k20,
    "lo_dual_k15": make_lo_dual_k15,
}

VARIANTS: dict[str, callable] = {**VARIANTS_A, **VARIANTS_B, **VARIANTS_C}

GROUP_MAP = {"A": VARIANTS_A, "B": VARIANTS_B, "C": VARIANTS_C}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_data(
    symbols: list[str],
    data_dir: Path,
) -> dict[str, NumpyBars]:
    data: dict[str, NumpyBars] = {}
    for sym in symbols:
        path = data_dir / f"{sym}_1d.parquet"
        if not path.exists():
            logger.warning("Missing daily data for %s at %s", sym, path)
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex(df.index)
        df.columns = df.columns.str.lower()
        data[sym] = build_numpy_arrays(df)
        logger.info("Loaded %s: %d daily bars", sym, len(data[sym].times))
    return data


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    variant_name: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    total_r: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    final_equity: float = 0.0
    initial_equity: float = 0.0
    total_return_pct: float = 0.0
    avg_bars_held: float = 0.0
    per_symbol: dict[str, dict] = None
    all_trades: list[S5TradeRecord] = None

    def __post_init__(self):
        if self.per_symbol is None:
            self.per_symbol = {}
        if self.all_trades is None:
            self.all_trades = []


def run_variant(
    variant_name: str,
    config: S5BacktestConfig,
    data: dict[str, NumpyBars],
) -> VariantResult:
    all_trades: list[S5TradeRecord] = []
    all_equity_curves: dict[str, list[float]] = {}
    per_symbol: dict[str, dict] = {}
    init_eq = config.initial_equity

    for sym in config.symbols:
        if sym not in data:
            continue
        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue

        engine = S5Engine(
            symbol=sym, cfg=cfg, bt_config=config, point_value=cfg.multiplier,
        )
        engine.run(data[sym])

        sym_trades = engine.trades
        all_trades.extend(sym_trades)

        sym_pnl = sum(t.pnl_dollars for t in sym_trades)
        sym_wins = sum(1 for t in sym_trades if t.pnl_dollars > 0)
        sym_total = len(sym_trades)
        sym_wr = sym_wins / sym_total * 100 if sym_total > 0 else 0
        sym_r = sum(t.r_multiple for t in sym_trades)
        sym_bars = sum(t.bars_held for t in sym_trades) / sym_total if sym_total > 0 else 0

        per_symbol[sym] = {
            "trades": sym_total,
            "pnl": sym_pnl,
            "win_rate": sym_wr,
            "total_r": sym_r,
            "avg_r": sym_r / sym_total if sym_total > 0 else 0,
            "avg_bars_held": sym_bars,
        }
        all_equity_curves[sym] = engine.equity_curve

    combined_eq = None
    for sym, curve in all_equity_curves.items():
        arr = np.array(curve)
        deltas = arr - init_eq
        if combined_eq is None:
            combined_eq = init_eq + deltas
        else:
            min_len = min(len(combined_eq), len(deltas))
            combined_eq = combined_eq[:min_len] + deltas[:min_len]

    if combined_eq is None or len(combined_eq) == 0:
        return VariantResult(variant_name=variant_name, initial_equity=init_eq)

    final_eq = combined_eq[-1]
    total_pnl = sum(t.pnl_dollars for t in all_trades)
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t.pnl_dollars > 0)
    total_r = sum(t.r_multiple for t in all_trades)
    avg_bars = sum(t.bars_held for t in all_trades) / total_trades if total_trades > 0 else 0

    peak = np.maximum.accumulate(combined_eq)
    dd = (combined_eq - peak) / peak * 100
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    returns = np.diff(combined_eq) / combined_eq[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    gross_profit = sum(t.pnl_dollars for t in all_trades if t.pnl_dollars > 0)
    gross_loss = abs(sum(t.pnl_dollars for t in all_trades if t.pnl_dollars < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_r = total_r / total_trades if total_trades > 0 else 0
    total_return = (final_eq - init_eq) / init_eq * 100

    return VariantResult(
        variant_name=variant_name, total_trades=total_trades, winning_trades=wins,
        total_pnl=total_pnl, total_r=total_r, max_drawdown_pct=max_dd,
        sharpe=sharpe, win_rate=win_rate, avg_r=avg_r, profit_factor=pf,
        final_equity=final_eq, initial_equity=init_eq, total_return_pct=total_return,
        avg_bars_held=avg_bars, per_symbol=per_symbol, all_trades=all_trades,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_variant_report(result: VariantResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"STRATEGY 5 (KELTNER MOMENTUM) - Variant: {result.variant_name}")
    print(f"{'=' * 70}")
    print(f"Initial Equity:  ${result.initial_equity:,.2f}")
    print(f"Final Equity:    ${result.final_equity:,.2f}")
    print(f"Total Return:    {result.total_return_pct:+.2f}%")
    print(f"Total PnL:       ${result.total_pnl:+,.2f}")
    print(f"Max Drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:    {result.sharpe:.2f}")
    print(f"Profit Factor:   {result.profit_factor:.2f}")
    print(f"Total Trades:    {result.total_trades}")
    print(f"Win Rate:        {result.win_rate:.1f}%")
    print(f"Avg R-Multiple:  {result.avg_r:+.2f}R")
    print(f"Total R:         {result.total_r:+.1f}R")
    print(f"Avg Bars Held:   {result.avg_bars_held:.1f}")

    if result.per_symbol:
        print(f"\nPer-Symbol Breakdown:")
        print(f"{'Symbol':<8} {'Trades':>7} {'Win%':>6} {'PnL':>12} {'Total R':>8} {'Avg R':>7} {'AvgBars':>8}")
        print("-" * 60)
        for sym, stats in sorted(result.per_symbol.items()):
            print(f"{sym:<8} {stats['trades']:>7} {stats['win_rate']:>5.1f}% "
                  f"${stats['pnl']:>10,.2f} {stats['total_r']:>7.1f}R "
                  f"{stats['avg_r']:>6.2f}R {stats['avg_bars_held']:>7.1f}")
    print(f"{'=' * 70}")


def print_comparison_table(results: list[VariantResult]) -> None:
    print(f"\n{'=' * 125}")
    print("STRATEGY 5 KELTNER MOMENTUM VARIANT COMPARISON")
    print(f"{'=' * 125}")
    print(f"{'Variant':<22} {'Trades':>7} {'Win%':>6} {'PnL':>12} {'Return':>8} "
          f"{'Sharpe':>7} {'MaxDD':>7} {'PF':>6} {'AvgR':>6} {'TotalR':>8} {'AvgBars':>8}")
    print("-" * 125)

    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)
    for r in sorted_results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
        print(f"{r.variant_name:<22} {r.total_trades:>7} {r.win_rate:>5.1f}% "
              f"${r.total_pnl:>10,.2f} {r.total_return_pct:>7.2f}% "
              f"{r.sharpe:>7.2f} {r.max_drawdown_pct:>6.2f}% "
              f"{pf_str:>6} {r.avg_r:>5.2f}R {r.total_r:>7.1f}R {r.avg_bars_held:>7.1f}")
    print("-" * 125)

    if sorted_results:
        best = sorted_results[0]
        print(f"\n  Best Sharpe: {best.variant_name} ({best.sharpe:.2f})")
        print(f"\n  === Best Variant ({best.variant_name}) Per-Symbol ===")
        for sym, stats in sorted(best.per_symbol.items()):
            print(f"    {sym}: {stats['trades']} trades, "
                  f"WR {stats['win_rate']:.1f}%, "
                  f"PnL ${stats['pnl']:+,.2f}, "
                  f"R {stats['total_r']:+.1f}, "
                  f"AvgBars {stats['avg_bars_held']:.1f}")
    print(f"{'=' * 125}")


def save_results(results, best, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        rows.append({
            "variant": r.variant_name,
            "trades": r.total_trades,
            "win_rate": round(r.win_rate, 1),
            "pnl": round(r.total_pnl, 2),
            "return_pct": round(r.total_return_pct, 2),
            "sharpe": round(r.sharpe, 2),
            "max_dd_pct": round(r.max_drawdown_pct, 2),
            "profit_factor": round(r.profit_factor, 2) if r.profit_factor < 100 else "inf",
            "avg_r": round(r.avg_r, 2),
            "total_r": round(r.total_r, 1),
            "avg_bars_held": round(r.avg_bars_held, 1),
            "per_symbol": {
                sym: {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
                for sym, stats in r.per_symbol.items()
            },
        })
    trade_rows = []
    for t in best.all_trades:
        trade_rows.append({
            "symbol": t.symbol,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_time": t.entry_time.isoformat() if t.entry_time else "",
            "exit_time": t.exit_time.isoformat() if t.exit_time else "",
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "qty": t.qty,
            "pnl": round(t.pnl_dollars, 2),
            "r_multiple": round(t.r_multiple, 2),
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
        })
    output = {
        "run_timestamp": ts,
        "best_variant": best.variant_name,
        "best_sharpe": round(best.sharpe, 2),
        "best_pnl": round(best.total_pnl, 2),
        "best_return_pct": round(best.total_return_pct, 2),
        "comparison": rows,
        "best_trades": trade_rows,
    }
    out_path = output_dir / f"s5_sweep_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy 5 Keltner Momentum backtest runner")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--group", type=str, default=None, choices=["A", "B", "C"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--data-dir", type=str, default="backtest/data/raw")
    parser.add_argument("--save-best", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = Path(args.data_dir)
    symbols = ["QQQ", "GLD", "IBIT"]

    print(f"Loading daily data for {symbols}...")
    data = load_daily_data(symbols, data_dir)
    if not data:
        print("ERROR: No data loaded.")
        return

    for sym, bars in data.items():
        print(f"  {sym}: {len(bars.times)} daily bars")

    # Select variants to run
    if args.group:
        variants = GROUP_MAP[args.group]
        print(f"\nRunning Group {args.group} ({len(variants)} variants)...")
    elif args.run_all:
        variants = VARIANTS
        print(f"\nRunning ALL variants ({len(variants)})...")
    elif args.variant:
        if args.variant not in VARIANTS:
            print(f"ERROR: Unknown variant '{args.variant}'")
            print(f"Available: {', '.join(sorted(VARIANTS.keys()))}")
            return
        variants = {args.variant: VARIANTS[args.variant]}
    else:
        # Default: run all
        variants = VARIANTS
        print(f"\nRunning ALL variants ({len(variants)})...")

    results = []
    for name, factory in variants.items():
        print(f"\nRunning variant: {name}...")
        config = factory(args.equity)
        result = run_variant(name, config, data)
        results.append(result)
        print(f"  -> {result.total_trades} trades, "
              f"PnL ${result.total_pnl:+,.2f}, Sharpe {result.sharpe:.2f}, "
              f"AvgBars {result.avg_bars_held:.1f}")

    if len(results) > 1:
        print_comparison_table(results)

    best = max(results, key=lambda r: r.sharpe) if results else None
    if best:
        print_variant_report(best)
        if args.save_best:
            save_results(results, best, Path(args.save_best))


if __name__ == "__main__":
    main()
