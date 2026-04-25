"""Generate v7 portfolio report using greedy-optimized configs.

Produces the same report format as portfolio_10k_v6.txt for like-for-like comparison.

Usage:
    cd trading
    python -u backtests/momentum/auto/run_v7_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.momentum.analysis.portfolio_reports import portfolio_full_report
from backtests.momentum.auto.config_mutator import (
    extract_passthrough_mutations,
    mutate_helix_config,
    mutate_nqdtc_config,
    mutate_portfolio_config,
    mutate_vdubus_config,
)
from backtests.momentum.cli import _load_helix_data, _load_nqdtc_data, _load_vdubus_data
from backtests.momentum.config_helix import Helix4BacktestConfig
from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.config_portfolio import PortfolioBacktestConfig
from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
from backtests.momentum.engine.helix_engine import Helix4Engine
from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
from backtests.momentum.engine.portfolio_engine import PortfolioBacktester
from backtests.momentum.engine.vdubus_engine import VdubusEngine

EQUITY = 10_000.0
DATA_DIR = ROOT / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "momentum" / "output"
AUTO_OUTPUT = ROOT / "backtests" / "momentum" / "auto" / "output"


def main():
    # Load optimal mutations from greedy result
    greedy_path = AUTO_OUTPUT / "greedy_portfolio_optimal.json"
    greedy_data = json.loads(greedy_path.read_text())
    mutations = greedy_data["accepted_mutations"]

    print(f"Loaded {len(mutations)} optimal mutations:")
    for k, v in sorted(mutations.items()):
        print(f"  {k}: {v}")
    print()

    # Build configs with optimal mutations
    portfolio_cfg = PortfolioBacktestConfig()
    portfolio_cfg = mutate_portfolio_config(portfolio_cfg, mutations)

    helix_muts = extract_passthrough_mutations(mutations, "helix")
    nqdtc_muts = extract_passthrough_mutations(mutations, "nqdtc")
    vdubus_muts = extract_passthrough_mutations(mutations, "vdubus")

    # Load data
    print("Loading bar data...")
    helix_data = _load_helix_data("NQ", DATA_DIR)
    nqdtc_data = _load_nqdtc_data("NQ", DATA_DIR)
    vdubus_data = _load_vdubus_data("NQ", DATA_DIR)

    # Run Helix
    print("Running Helix...")
    helix_cfg = Helix4BacktestConfig(initial_equity=EQUITY, fixed_qty=10)
    if helix_muts:
        helix_cfg = mutate_helix_config(helix_cfg, helix_muts)
    engine = Helix4Engine(symbol="NQ", bt_config=helix_cfg)
    helix_result = engine.run(
        helix_data["minute_bars"], helix_data["hourly"], helix_data["four_hour"],
        helix_data["daily"], helix_data["hourly_idx_map"],
        helix_data["four_hour_idx_map"], helix_data["daily_idx_map"],
    )
    helix_trades = helix_result.trades

    # Run NQDTC
    print("Running NQDTC...")
    nqdtc_cfg = NQDTCBacktestConfig(symbols=["MNQ"], initial_equity=EQUITY, fixed_qty=10)
    if nqdtc_muts:
        nqdtc_cfg = mutate_nqdtc_config(nqdtc_cfg, nqdtc_muts)
    engine = NQDTCEngine(symbol="MNQ", bt_config=nqdtc_cfg)
    nqdtc_result = engine.run(
        nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"], nqdtc_data["hourly"],
        nqdtc_data["four_hour"], nqdtc_data["daily"],
        nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
        nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
        daily_es=nqdtc_data.get("daily_es"),
        daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
    )
    nqdtc_trades = nqdtc_result.trades

    # Run Vdubus
    print("Running Vdubus...")
    vdubus_cfg = VdubusBacktestConfig(
        initial_equity=EQUITY, fixed_qty=10,
        flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
    )
    if vdubus_muts:
        vdubus_cfg = mutate_vdubus_config(vdubus_cfg, vdubus_muts)
    engine = VdubusEngine(symbol="NQ", bt_config=vdubus_cfg)
    vdubus_result = engine.run(
        vdubus_data["bars_15m"], vdubus_data.get("bars_5m"), vdubus_data["hourly"],
        vdubus_data["daily_es"], vdubus_data["hourly_idx_map"],
        vdubus_data["daily_es_idx_map"], vdubus_data.get("five_to_15_idx_map"),
    )
    vdubus_trades = vdubus_result.trades

    # Compute independent R-multiples (sum-of-independent baseline)
    independent_pnl = {
        "Helix": sum(t.r_multiple for t in helix_trades),
        "NQDTC": sum(t.r_multiple for t in nqdtc_trades),
        "Vdubus": sum(t.r_multiple for t in vdubus_trades),
    }
    print(f"\nIndependent R: Helix={independent_pnl['Helix']:+.1f}R, "
          f"NQDTC={independent_pnl['NQDTC']:+.1f}R, "
          f"Vdubus={independent_pnl['Vdubus']:+.1f}R")

    # Run portfolio simulation
    print("Running portfolio simulation...")
    backtester = PortfolioBacktester(portfolio_cfg)
    result = backtester.run(helix_trades, nqdtc_trades, vdubus_trades)

    # Generate report (same format as v6)
    report = portfolio_full_report(result, independent_pnl=independent_pnl)

    # Save
    output_path = OUTPUT_DIR / "portfolio_10k_v7.txt"
    output_path.write_text(report, encoding="utf-8")

    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
