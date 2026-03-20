"""Run VdubusNQ v4.0 baseline backtest with MNQ (Micro E-mini Nasdaq 100).

Fixed 10 contracts per trade, full diagnostics, buy-and-hold comparison.
Output: backtest/output/vdubus_baseline.txt
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
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- MNQ contract specs ---
MNQ_POINT_VALUE = 2.0       # $2 per point (1/10th of NQ)
MNQ_TICK_SIZE = 0.25        # Same tick size as NQ
MNQ_TICK_VALUE = 0.50       # $0.50 per tick
MNQ_COMMISSION = 0.62       # Per-side commission (IBKR micros)
MNQ_RT_COMM_FEES = 1.24     # Round-trip commission + fees
FIXED_QTY = 10
DATA_DIR = Path("backtest/data/raw")
EQUITY = 100_000.0
OUTPUT = Path("backtest/output/vdubus_baseline.txt")


def main():
    from backtest.analysis.metrics import compute_metrics
    from backtest.analysis.reports import format_summary
    from backtest.analysis.vdubus_diagnostics import vdubus_full_diagnostic
    from backtest.analysis.vdubus_filter_attribution import (
        vdubus_filter_attribution_report,
    )
    from backtest.cli import _load_vdubus_data
    from backtest.config import SlippageConfig
    from backtest.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtest.data.cache import load_bars
    from backtest.data.preprocessing import build_numpy_arrays, normalize_timezone
    from backtest.engine.vdubus_engine import VdubusEngine
    from strategy_3 import config as C

    # Load data (NQ price data — MNQ tracks NQ identically)
    symbol = "NQ"
    logger.info("Loading VdubusNQ data for MNQ baseline...")
    vdubus_data = _load_vdubus_data(symbol, DATA_DIR)

    # --- Patch strategy_3 config for MNQ ---
    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = MNQ_TICK_VALUE
    C.NQ_SPEC["point_value"] = MNQ_POINT_VALUE
    C.RT_COMM_FEES = MNQ_RT_COMM_FEES

    try:
        # Disable heat_cap and viability_filter since fixed-qty bypasses
        # risk-based sizing — these gates are calibrated for dynamic sizing
        # and would incorrectly block fixed-size trades.
        flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )

        config = VdubusBacktestConfig(
            symbols=[symbol],
            initial_equity=EQUITY,
            data_dir=DATA_DIR,
            slippage=SlippageConfig(commission_per_contract=MNQ_COMMISSION),
            fixed_qty=FIXED_QTY,
            tick_size=MNQ_TICK_SIZE,
            point_value=MNQ_POINT_VALUE,
            flags=flags,
        )

        logger.info(
            "Running VdubusNQ v4.0 backtest: MNQ, fixed_qty=%d, pv=$%.2f, comm=$%.2f/side",
            FIXED_QTY, MNQ_POINT_VALUE, MNQ_COMMISSION,
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
        # Restore original config
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    if not result.trades:
        print("VdubusNQ v4.0 (MNQ): No trades generated.")
        return

    logger.info("Backtest complete: %d trades", len(result.trades))

    # --- Compute metrics ---
    pnls = np.array([t.pnl_dollars for t in result.trades])
    risks = np.array([
        abs(t.entry_price - t.initial_stop) * config.point_value
        for t in result.trades
    ])
    holds = np.array([t.bars_held_15m for t in result.trades])
    comms = np.array([t.commission for t in result.trades])

    metrics = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=holds,
        trade_commissions=comms,
        equity_curve=result.equity_curve,
        timestamps=result.time_series,
        initial_equity=config.initial_equity,
    )

    # --- Build report sections ---
    sections: list[str] = []

    # 1. Header
    sections.append(
        f"=== VdubusNQ v4.0 Baseline: MNQ (Micro E-mini Nasdaq 100) ===\n"
        f"Instrument:         MNQ\n"
        f"Point value:        ${MNQ_POINT_VALUE:.2f}\n"
        f"Tick size:          {MNQ_TICK_SIZE}\n"
        f"Tick value:         ${MNQ_TICK_VALUE:.2f}\n"
        f"Commission/side:    ${MNQ_COMMISSION:.2f}\n"
        f"Fixed qty:          {FIXED_QTY} contracts\n"
        f"Initial equity:     ${EQUITY:,.0f}\n"
        f"Disabled gates:     heat_cap, viability_filter (N/A for fixed-qty)"
    )

    # 2. Performance summary
    perf_lines = [
        f"=== Performance Summary ===",
        f"Total trades:       {metrics.total_trades}",
        f"Win rate:           {metrics.win_rate:.1%}",
        f"Profit factor:      {metrics.profit_factor:.2f}",
        f"Expectancy (R):     {metrics.expectancy:+.3f}",
        f"Expectancy ($):     {metrics.expectancy_dollar:+,.2f}",
        f"Net profit:         ${metrics.net_profit:+,.2f}",
        f"CAGR:               {metrics.cagr:.1%}",
        f"Sharpe:             {metrics.sharpe:.2f}",
        f"Sortino:            {metrics.sortino:.2f}",
        f"Calmar:             {metrics.calmar:.2f}",
        f"Max drawdown:       {metrics.max_drawdown_pct:.1%} (${metrics.max_drawdown_dollar:,.2f})",
        f"Avg hold (15m bars):{metrics.avg_hold_hours:.1f}",
        f"Trades/month:       {metrics.trades_per_month:.1f}",
        f"Total commissions:  ${metrics.total_commissions:,.2f}",
    ]
    sections.append("\n".join(perf_lines))

    # 3. Signal funnel
    funnel_lines = [
        f"=== Signal Funnel ===",
        f"  15m evaluations:  {result.evaluations}",
        f"  Regime passed:    {result.regime_passed}",
        f"  Signals found:    {result.signals_found}",
        f"  Entries placed:   {result.entries_placed}",
        f"  Entries filled:   {result.entries_filled}",
        f"  Trades completed: {len(result.trades)}",
    ]
    if result.entries_placed > 0:
        funnel_lines.append(
            f"  Fill rate:        {result.entries_filled / result.entries_placed:.1%}"
        )
    sections.append("\n".join(funnel_lines))

    # 4. Buy & Hold comparison (10 MNQ held for entire period)
    nq_daily_path = DATA_DIR / "NQ_1d.parquet"
    if nq_daily_path.exists():
        daily_df = normalize_timezone(load_bars(nq_daily_path))
        daily_bars = build_numpy_arrays(daily_df)
        closes = daily_bars.closes[np.isfinite(daily_bars.closes)]
        if len(closes) >= 2:
            start_price = float(closes[0])
            end_price = float(closes[-1])
            price_change = end_price - start_price
            bh_pnl = price_change * MNQ_POINT_VALUE * FIXED_QTY
            bh_return_pct = (end_price / start_price - 1.0) * 100

            # B&H equity curve for max drawdown
            bh_equity = EQUITY + (closes - start_price) * MNQ_POINT_VALUE * FIXED_QTY
            bh_peak = np.maximum.accumulate(bh_equity)
            bh_dd_series = (bh_equity - bh_peak) / bh_peak
            bh_max_dd_pct = float(np.min(bh_dd_series)) * 100

            strat_return_pct = metrics.net_profit / EQUITY * 100

            bh_lines = [
                f"=== Buy & Hold Comparison: {FIXED_QTY} MNQ ===",
                f"{'':22s} {'Buy & Hold':>14s} {'Strategy':>14s}",
                f"{'-'*52}",
                f"{'Net P&L ($)':22s} ${bh_pnl:>+13,.0f} ${metrics.net_profit:>+13,.0f}",
                f"{'Return (%)':22s} {bh_return_pct:>+13.1f}% {strat_return_pct:>+13.1f}%",
                f"{'Max Drawdown (%)':22s} {bh_max_dd_pct:>+13.1f}% {-metrics.max_drawdown_pct * 100:>+13.1f}%",
                f"{'Start Price':22s} {start_price:>14.2f} {'':>14s}",
                f"{'End Price':22s} {end_price:>14.2f} {'':>14s}",
                f"{'Price Change (pts)':22s} {price_change:>+14.2f} {'':>14s}",
            ]
            sections.append("\n".join(bh_lines))

    # 5. Compact summary line
    sections.append(format_summary(metrics))

    # 6. Full diagnostics (22 sections)
    logger.info("Generating full diagnostics...")
    diag = vdubus_full_diagnostic(
        result.trades,
        signal_events=result.signal_events,
        equity_curve=result.equity_curve,
        time_series=result.time_series,
    )
    sections.append(diag)

    # 7. Filter attribution
    if result.signal_events:
        logger.info("Generating filter attribution report...")
        attr = vdubus_filter_attribution_report(
            result.signal_events, result.trades,
            shadow_tracker=result.shadow_tracker,
        )
        sections.append(attr)

    # 8. Shadow trade report
    if result.shadow_summary:
        sections.append(result.shadow_summary)

    # --- Save report ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n\n".join(sections) + "\n"
    OUTPUT.write_text(report_text, encoding="utf-8")

    # Print to console (handle Windows encoding)
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))
    logger.info("Report saved to: %s (%d bytes)", OUTPUT, len(report_text))


if __name__ == "__main__":
    main()
