#!/usr/bin/env python
"""Run Strategy 5 Group B backtest."""
import sys
import os
from pathlib import Path
import logging

os.chdir(str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

from backtest.run_s5 import (
    load_daily_data, run_variant, VARIANTS_B,
    print_comparison_table, print_variant_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

data_dir = Path("backtest/data/raw")
symbols = ["QQQ", "GLD", "IBIT"]
equity = 10000.0

print(f"Loading daily data for {symbols}...")
data = load_daily_data(symbols, data_dir)

if not data:
    print("ERROR: No data loaded.")
    sys.exit(1)

for sym, bars in data.items():
    print(f"  {sym}: {len(bars.times)} daily bars")

print(f"\nRunning Group B ({len(VARIANTS_B)} pullback variants)...")
results = []

for name, factory in VARIANTS_B.items():
    print(f"\nRunning variant: {name}...")
    config = factory(equity)
    result = run_variant(name, config, data)
    results.append(result)
    print(f"  -> {result.total_trades} trades, PnL ${result.total_pnl:+,.2f}, Sharpe {result.sharpe:.2f}, AvgBars {result.avg_bars_held:.1f}")

print("\n" + "="*150)
print_comparison_table(results)

best = max(results, key=lambda r: r.sharpe) if results else None
if best:
    print_variant_report(best)

print("\nBacktest completed successfully!")
