"""Sweep TOUCH_LOOKBACK_15M with corrected VWAP-A anchor mapping.

Tests lookback values [8, 12, 16, 20, 24, 32] and prints a comparison table.
Requires the anchor fix in vdubus_engine.py (timestamp-based mapping).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("backtest/data/raw")
EQUITY = 100_000.0
LOOKBACK_VALUES = [8, 12, 16, 20, 24, 32]


def run_single(lookback: int) -> dict:
    """Run one full backtest with the given TOUCH_LOOKBACK_15M value."""
    from backtest.analysis.metrics import compute_metrics
    from backtest.cli import _load_vdubus_data
    from backtest.config import SlippageConfig
    from backtest.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtest.engine.vdubus_engine import VdubusEngine
    from strategy_3 import config as C
    from strategy_3 import signals as sig

    symbol = "NQ"
    vdubus_data = _load_vdubus_data(symbol, DATA_DIR)

    # Patch MNQ config (same as CLI defaults)
    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = 0.50
    C.NQ_SPEC["point_value"] = 2.0
    C.RT_COMM_FEES = 1.24

    # Patch both the config constant AND the function default (bound at import)
    orig_lookback = C.TOUCH_LOOKBACK_15M
    orig_defaults = sig.choose_vwap_used.__defaults__
    C.TOUCH_LOOKBACK_15M = lookback
    sig.choose_vwap_used.__defaults__ = (lookback,)

    try:
        flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )
        config = VdubusBacktestConfig(
            symbols=[symbol],
            initial_equity=EQUITY,
            data_dir=DATA_DIR,
            slippage=SlippageConfig(),
            fixed_qty=10,
            flags=flags,
        )

        engine = VdubusEngine(symbol, config)
        result = engine.run(
            vdubus_data["bars_15m"],
            vdubus_data.get("bars_5m"),
            vdubus_data["hourly"],
            vdubus_data["daily_es"],
            vdubus_data["hourly_idx_map"],
            vdubus_data["daily_es_idx_map"],
            vdubus_data.get("five_to_15_idx_map"),
        )
    finally:
        C.TOUCH_LOOKBACK_15M = orig_lookback
        sig.choose_vwap_used.__defaults__ = orig_defaults
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    trades = result.trades
    n_trades = len(trades)

    if n_trades == 0:
        return {
            "lookback": lookback,
            "trades": 0,
            "win_rate": 0.0,
            "expectancy_R": 0.0,
            "net_pnl": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "profit_factor": 0.0,
            "eod_trades": 0,
            "net_pnl_ex_eod": 0.0,
        }

    pnls = np.array([t.pnl_dollars for t in trades])
    risks = np.array([
        abs(t.entry_price - t.initial_stop) * config.point_value
        for t in trades
    ])
    holds = np.array([t.bars_held_15m for t in trades])
    comms = np.array([t.commission for t in trades])

    metrics = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=holds,
        trade_commissions=comms,
        equity_curve=result.equity_curve,
        timestamps=result.time_series,
        initial_equity=config.initial_equity,
    )

    # Compute stats excluding END_OF_DATA trades
    eod_trades = [t for t in trades if t.exit_reason == "END_OF_DATA"]
    non_eod_pnls = np.array([
        t.pnl_dollars for t in trades if t.exit_reason != "END_OF_DATA"
    ])
    net_pnl_ex_eod = float(non_eod_pnls.sum()) if len(non_eod_pnls) > 0 else 0.0

    return {
        "lookback": lookback,
        "trades": n_trades,
        "win_rate": metrics.win_rate,
        "expectancy_R": metrics.expectancy,
        "net_pnl": metrics.net_profit,
        "sharpe": metrics.sharpe,
        "max_dd_pct": metrics.max_drawdown_pct,
        "profit_factor": metrics.profit_factor,
        "eod_trades": len(eod_trades),
        "net_pnl_ex_eod": net_pnl_ex_eod,
    }


def main():
    print("=" * 90)
    print("VWAP-A TOUCH_LOOKBACK_15M Sweep (with corrected timestamp-based anchor)")
    print("=" * 90)
    print()

    results = []
    for lb in LOOKBACK_VALUES:
        print(f"  Running lookback={lb}...", end=" ", flush=True)
        row = run_single(lb)
        results.append(row)
        print(
            f"{row['trades']} trades, WR={row['win_rate']:.1%}, "
            f"PnL=${row['net_pnl']:+,.0f}, PF={row['profit_factor']:.2f}"
        )

    # Print comparison table
    print()
    print("-" * 90)
    header = (
        f"{'LB':>4s}  {'Trades':>6s}  {'WinRate':>7s}  {'Exp(R)':>7s}  "
        f"{'NetPnL':>10s}  {'Sharpe':>6s}  {'MaxDD%':>7s}  {'PF':>6s}  "
        f"{'EOD':>3s}  {'PnL ex-EOD':>11s}"
    )
    print(header)
    print("-" * 90)
    for r in results:
        line = (
            f"{r['lookback']:>4d}  {r['trades']:>6d}  {r['win_rate']:>7.1%}  "
            f"{r['expectancy_R']:>+7.3f}  {r['net_pnl']:>+10,.0f}  "
            f"{r['sharpe']:>6.2f}  {r['max_dd_pct']:>7.1%}  "
            f"{r['profit_factor']:>6.2f}  {r['eod_trades']:>3d}  "
            f"{r['net_pnl_ex_eod']:>+11,.0f}"
        )
        print(line)
    print("-" * 90)
    print()
    print("Notes:")
    print("  - 'EOD' = END_OF_DATA trades (position open when data ends)")
    print("  - 'PnL ex-EOD' = net PnL excluding END_OF_DATA exits")
    print("  - Anchor mapping uses timestamp matching (not pivot.idx * 4)")


if __name__ == "__main__":
    main()
