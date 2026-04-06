"""Run greedy forward selection for momentum portfolio optimization.

Usage:
    cd trading
    PYTHONUNBUFFERED=1 python -u research/backtests/momentum/auto/run_greedy.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

# Install momentum aliases before any backtest imports
from research.backtests.momentum._aliases import install
install()

from research.backtests.momentum.auto.greedy_optimize import (
    PORTFOLIO_CANDIDATES,
    run_greedy,
    save_result,
)

EQUITY = 10_000.0
DATA_DIR = ROOT / "research" / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = ROOT / "research" / "backtests" / "momentum" / "auto" / "output"


def main():
    print("Starting momentum greedy portfolio optimization...")
    t0 = time.time()

    result = run_greedy(
        candidates=PORTFOLIO_CANDIDATES,
        initial_equity=EQUITY,
        data_dir=DATA_DIR,
        max_workers=3,
        verbose=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_result(result, OUTPUT_DIR / "greedy_portfolio_optimal.json")

    # Print per-strategy breakdown
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
