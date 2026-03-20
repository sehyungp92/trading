"""Sweep MID L values to find optimal session-scaled lookback.

Run: python sweep_L.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.disable(logging.INFO)

from backtest.cli import _load_nqdtc_data
from backtest.analysis.metrics import compute_metrics
from backtest.config_nqdtc import NQDTCBacktestConfig
from backtest.engine.nqdtc_engine import NQDTCEngine
from strategy_2 import config as C
from strategy_2 import models as M

# Save original BoxEngineState class for restoration
_OrigBoxEngineState = M.BoxEngineState


def run_with_L(mid_L: int, nqdtc_data: dict, config: NQDTCBacktestConfig) -> dict:
    old_adaptive = C.ADAPTIVE_L.copy()

    C.ADAPTIVE_L["MID"] = mid_L

    # Monkey-patch BoxEngineState so new instances default to mid_L.
    # Dataclass __init__ bakes defaults into function signature, so we wrap it.
    _orig_init = _OrigBoxEngineState.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("L", mid_L)
        kwargs.setdefault("L_used", mid_L)
        _orig_init(self, *args, **kwargs)

    M.BoxEngineState.__init__ = _patched_init

    try:
        engine = NQDTCEngine("MNQ", config)
        result = engine.run(
            nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"],
            nqdtc_data["hourly"], nqdtc_data["four_hour"], nqdtc_data["daily"],
            nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
            nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
        )

        # Extract arrays from trades (matches cli.py NQDTC pattern)
        pnls = np.array([t.pnl_dollars for t in result.trades], dtype=np.float64)
        risks = np.array([
            abs(t.entry_price - t.initial_stop) * config.point_value * t.qty
            for t in result.trades
        ], dtype=np.float64)
        holds = np.array([t.bars_held_30m for t in result.trades], dtype=np.float64)
        comms = np.array([t.commission for t in result.trades], dtype=np.float64)

        metrics = compute_metrics(
            pnls, risks, holds, comms,
            result.equity_curve, result.timestamps,
            config.initial_equity,
        )
        return {
            "L": mid_L,
            "trades": metrics.total_trades,
            "wr": metrics.win_rate * 100,
            "pf": metrics.profit_factor,
            "net": metrics.net_profit,
            "sharpe": metrics.sharpe,
            "er": metrics.expectancy,
            "maxdd": metrics.max_drawdown_pct * 100,
            "cagr": metrics.cagr * 100,
        }
    finally:
        C.ADAPTIVE_L.update(old_adaptive)
        M.BoxEngineState.__init__ = _orig_init


def main():
    data_dir = Path("backtest/data/raw")
    nqdtc_data = _load_nqdtc_data("NQ", data_dir)
    config = NQDTCBacktestConfig(symbols=["MNQ"], data_dir=data_dir, fixed_qty=10)

    L_values = [16, 18, 20, 22, 24, 26, 28, 30, 32]

    print(f"{'L':>4}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  {'E[R]':>7}  "
          f"{'Net':>10}  {'Sharpe':>7}  {'CAGR':>6}  {'MaxDD':>6}")
    print("-" * 78)

    results = []
    for L in L_values:
        r = run_with_L(L, nqdtc_data, config)
        results.append(r)
        print(f"{r['L']:>4}  {r['trades']:>6}  {r['wr']:>5.1f}%  {r['pf']:>6.2f}  "
              f"{r['er']:>+7.3f}  ${r['net']:>+9,.0f}  {r['sharpe']:>7.2f}  "
              f"{r['cagr']:>5.1f}%  {r['maxdd']:>5.1f}%")

    best_pf = max(results, key=lambda x: x["pf"])
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    best_net = max(results, key=lambda x: x["net"])
    print(f"\nBest PF:     L={best_pf['L']} (PF={best_pf['pf']:.2f})")
    print(f"Best Sharpe: L={best_sharpe['L']} (Sharpe={best_sharpe['sharpe']:.2f})")
    print(f"Best Net:    L={best_net['L']} (Net=${best_net['net']:+,.0f})")


if __name__ == "__main__":
    main()
