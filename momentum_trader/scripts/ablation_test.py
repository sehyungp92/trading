"""Systematic ablation test for Apex v2.2 changes.

Instrument: MNQ (Micro E-mini Nasdaq, $2/pt, same price as NQ)
Fixed qty: 10 contracts per trade
Benchmark: 10 MNQ contracts buy-and-hold for the entire period

Establishes current config as baseline, then reverts one change at a time.
If reverting a change IMPROVES performance, that change is kept reverted.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.analysis.metrics import PerformanceMetrics, compute_metrics
from backtest.cli import _load_apex_data
from backtest.config_apex import ApexAblationFlags, ApexBacktestConfig
from backtest.engine.apex_engine import ApexEngine

SYMBOL = "NQ"            # data file symbol (NQ_5m.parquet)
FIXED_QTY = 10           # MNQ contracts per trade
INITIAL_EQUITY = 100_000.0
MNQ_POINT_VALUE = 2.0    # $2 per point (vs NQ $20)


# ---------------------------------------------------------------------------
# Ablation definitions: each entry reverts ONE change group to old values
# ---------------------------------------------------------------------------

@dataclass
class AblationCase:
    name: str
    description: str
    param_overrides: dict[str, float] = field(default_factory=dict)
    flag_overrides: dict[str, bool] = field(default_factory=dict)


ABLATION_CASES = [
    AblationCase(
        name="offset_ticks_std",
        description="Revert OFFSET_TICKS_STD from 4 -> 12 (old)",
        param_overrides={"offset_ticks_std": 12},
    ),
    AblationCase(
        name="offset_ticks_hivol",
        description="Revert OFFSET_TICKS_HIVOL from 8 -> 20 (old)",
        param_overrides={"offset_ticks_hivol": 20},
    ),
    AblationCase(
        name="bos_ttl_1h",
        description="Revert BOS_TTL_1H_HOURS from 12 -> 8 (old)",
        param_overrides={"bos_ttl_1h_hours": 8},
    ),
    AblationCase(
        name="bos_ttl_4h",
        description="Revert BOS_TTL_4H_HOURS from 18 -> 12 (old)",
        param_overrides={"bos_ttl_4h_hours": 12},
    ),
    AblationCase(
        name="heat_cap",
        description="Revert HEAT_CAP_R from 3.0 -> 2.0 (old)",
        param_overrides={"heat_cap_r": 2.0},
    ),
    AblationCase(
        name="low_vol_sizing",
        description="Revert LOW_VOL_SIZING_THRESHOLD from 50 -> 40 (old)",
        param_overrides={"low_vol_sizing_threshold": 40},
    ),
    AblationCase(
        name="rescue_tier",
        description="Disable rescue tier (ablation flag) + revert TTL/distance",
        param_overrides={
            "rescue_ttl_minutes": 2,
            "rescue_max_distance_ticks": 40,
        },
        flag_overrides={"rescue_enabled": False},
    ),
    AblationCase(
        name="be_buffer",
        description="Revert PLUS_1R_BE_BUFFER_ATR_FRAC from 0.05 -> 0.10 (old)",
        param_overrides={"plus_1r_be_buffer_atr_frac": 0.10},
    ),
    AblationCase(
        name="partial_thresholds",
        description="Revert PARTIAL_R=2.5, PARTIAL_PCT=0.40, VIRTUAL_PARTIAL_R=2.5 (old)",
        param_overrides={
            "partial_r_threshold": 2.5,
            "partial_pct": 0.40,
            "virtual_partial_r_threshold": 2.5,
        },
    ),
    AblationCase(
        name="trail_mult_min",
        description="Revert TRAIL_MULT_MIN from 1.2 -> 1.5 (old)",
        param_overrides={"trail_mult_min": 1.5},
    ),
    AblationCase(
        name="4h_offset_differentiation",
        description="Revert 4H offsets to match 1H (disable differentiation)",
        param_overrides={
            "offset_ticks_4h_std": 4,    # same as 1H STD
            "offset_ticks_4h_hivol": 8,  # same as 1H HIVOL
        },
    ),
    AblationCase(
        name="low_vol_entry_gate",
        description="Disable low-vol entry gate (ablation flag)",
        flag_overrides={"low_vol_entry_gate": False},
    ),
]


def run_backtest(
    apex_data: dict,
    param_overrides: dict[str, float] | None = None,
    flag_overrides: dict[str, bool] | None = None,
) -> tuple[PerformanceMetrics, int, np.ndarray, np.ndarray]:
    """Run a single backtest. Returns (metrics, n_trades, equity_curve, timestamps)."""
    flags = ApexAblationFlags()
    if flag_overrides:
        for k, v in flag_overrides.items():
            setattr(flags, k, v)

    config = ApexBacktestConfig(
        symbols=[SYMBOL],
        initial_equity=INITIAL_EQUITY,
        fixed_qty=FIXED_QTY,
        flags=flags,
        param_overrides=param_overrides or {},
        track_signals=False,
        track_shadows=False,
    )

    engine = ApexEngine(SYMBOL, config)
    result = engine.run(
        apex_data["minute_bars"],
        apex_data["hourly"],
        apex_data["four_hour"],
        apex_data["daily"],
        apex_data["hourly_idx_map"],
        apex_data["four_hour_idx_map"],
        apex_data["daily_idx_map"],
    )

    n_trades = len(result.trades)
    if n_trades == 0:
        return PerformanceMetrics(), 0, result.equity_curve, result.timestamps

    pnls = np.array([t.pnl_dollars for t in result.trades])
    risks = np.array([
        abs(t.avg_entry - t.initial_stop) * config.point_value
        for t in result.trades
    ])
    holds = np.array([t.bars_held_1h for t in result.trades])
    comms = np.array([t.commission for t in result.trades])

    metrics = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=holds,
        trade_commissions=comms,
        equity_curve=result.equity_curve,
        timestamps=result.timestamps,
        initial_equity=INITIAL_EQUITY,
    )

    return metrics, n_trades, result.equity_curve, result.timestamps


def compute_buy_and_hold(apex_data: dict) -> dict:
    """Compute buy-and-hold of 10 MNQ contracts for the full period."""
    daily = apex_data["daily"]
    closes = daily.closes[np.isfinite(daily.closes)]
    if len(closes) < 2:
        return {"net": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0}

    start_price = float(closes[0])
    end_price = float(closes[-1])
    pnl = (end_price - start_price) * MNQ_POINT_VALUE * FIXED_QTY
    ret_pct = (end_price / start_price - 1.0) * 100.0

    # Max drawdown
    equity = INITIAL_EQUITY + (closes - start_price) * MNQ_POINT_VALUE * FIXED_QTY
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd_pct = float(np.min(dd)) * 100.0  # negative

    return {
        "start_price": start_price,
        "end_price": end_price,
        "net": pnl,
        "return_pct": ret_pct,
        "max_dd_pct": max_dd_pct,
        "final_equity": INITIAL_EQUITY + pnl,
    }


def score(m: PerformanceMetrics) -> float:
    """Composite score: higher is better.

    Weighted blend emphasizing risk-adjusted returns and trade frequency.
    """
    if m.total_trades == 0:
        return -999.0

    profit_score = m.net_profit / INITIAL_EQUITY
    expectancy_score = m.expectancy
    sharpe_score = m.sharpe * 0.5
    frequency_bonus = min(m.trades_per_month / 5.0, 1.0) * 0.2
    dd_penalty = m.max_drawdown_pct * 0.5

    return profit_score + expectancy_score + sharpe_score + frequency_bonus - dd_penalty


def fmt_metrics(m: PerformanceMetrics, n_trades: int) -> str:
    """Format key metrics in a single line."""
    return (
        f"Trades={n_trades:3d}  "
        f"Net=${m.net_profit:+,.0f}  "
        f"WR={m.win_rate:.1%}  "
        f"Exp={m.expectancy:+.3f}R  "
        f"Sharpe={m.sharpe:.2f}  "
        f"MaxDD={m.max_drawdown_pct:.2%}  "
        f"Trades/mo={m.trades_per_month:.1f}  "
        f"PF={m.profit_factor:.2f}"
    )


def main():
    data_dir = Path("backtest/data/raw")

    print(f"Loading data (symbol={SYMBOL}, instrument=MNQ, pv=${MNQ_POINT_VALUE}/pt)...")
    apex_data = _load_apex_data(SYMBOL, data_dir)
    print("Data loaded.\n")

    # ---------------------------------------------------------------
    # Buy-and-hold benchmark
    # ---------------------------------------------------------------
    bh = compute_buy_and_hold(apex_data)
    print("=" * 80)
    print(f"BUY & HOLD BENCHMARK: {FIXED_QTY} MNQ contracts")
    print("=" * 80)
    print(f"  Entry: {bh['start_price']:.2f}  Exit: {bh['end_price']:.2f}")
    print(f"  Net P&L: ${bh['net']:+,.0f}  Return: {bh['return_pct']:+.1f}%")
    print(f"  Final equity: ${bh['final_equity']:,.0f}  MaxDD: {bh['max_dd_pct']:.2f}%")
    print()

    # ---------------------------------------------------------------
    # Step 1: Run baseline (all current changes applied)
    # ---------------------------------------------------------------
    print("=" * 80)
    print(f"BASELINE (all v2.2 changes, fixed_qty={FIXED_QTY} MNQ)")
    print("=" * 80)
    baseline_metrics, baseline_n, _, _ = run_backtest(apex_data)
    baseline_score = score(baseline_metrics)
    print(f"  {fmt_metrics(baseline_metrics, baseline_n)}")
    print(f"  Score: {baseline_score:.4f}")
    print()

    # ---------------------------------------------------------------
    # Step 2: Ablate each change one at a time
    # ---------------------------------------------------------------
    removals: list[str] = []
    cumulative_overrides: dict[str, float] = {}
    cumulative_flag_overrides: dict[str, bool] = {}

    for case in ABLATION_CASES:
        print("-" * 80)
        print(f"ABLATING: {case.name}")
        print(f"  {case.description}")
        print("-" * 80)

        test_overrides = {**cumulative_overrides, **case.param_overrides}
        test_flags = {**cumulative_flag_overrides, **case.flag_overrides}

        ablated_metrics, ablated_n, _, _ = run_backtest(
            apex_data,
            param_overrides=test_overrides,
            flag_overrides=test_flags,
        )
        ablated_score = score(ablated_metrics)

        print(f"  {fmt_metrics(ablated_metrics, ablated_n)}")
        print(f"  Score: {ablated_score:.4f}  (baseline: {baseline_score:.4f}  delta: {ablated_score - baseline_score:+.4f})")

        if ablated_score > baseline_score:
            print(f"  >>> REMOVING {case.name} IMPROVES performance. Keeping it removed.")
            removals.append(case.name)
            cumulative_overrides.update(case.param_overrides)
            cumulative_flag_overrides.update(case.flag_overrides)
            baseline_score = ablated_score
        else:
            print(f"  >>> Keeping {case.name} (removing it hurts performance).")

        print()

    # ---------------------------------------------------------------
    # Step 3: Final result
    # ---------------------------------------------------------------
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if removals:
        print(f"Changes removed: {', '.join(removals)}")
        final_metrics, final_n, _, _ = run_backtest(
            apex_data,
            param_overrides=cumulative_overrides,
            flag_overrides=cumulative_flag_overrides,
        )
        final_sc = score(final_metrics)
        print(f"  {fmt_metrics(final_metrics, final_n)}")
        print(f"  Score: {final_sc:.4f}")
    else:
        print("No changes removed -- all v2.2 changes contribute positively.")
        final_metrics, final_n = baseline_metrics, baseline_n
        print(f"  {fmt_metrics(final_metrics, final_n)}")
        print(f"  Score: {baseline_score:.4f}")

    print()
    print(f"  vs Buy & Hold: strategy ${final_metrics.net_profit:+,.0f} vs B&H ${bh['net']:+,.0f}")
    print()
    print("Change disposition:")
    for case in ABLATION_CASES:
        status = "REMOVED" if case.name in removals else "KEPT"
        print(f"  [{status}] {case.name}: {case.description}")

    if cumulative_overrides or cumulative_flag_overrides:
        print()
        print("=" * 80)
        print("OVERRIDES TO APPLY PERMANENTLY IN config.py:")
        print("=" * 80)
        for k, v in cumulative_overrides.items():
            print(f"  {k.upper()} = {v}")
        for k, v in cumulative_flag_overrides.items():
            print(f"  [ablation flag] {k} = {v}")


if __name__ == "__main__":
    main()
