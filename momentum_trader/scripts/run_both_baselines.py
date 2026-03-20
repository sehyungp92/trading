"""Run NQDTC v2.0 + VdubusNQ v4.4 full diagnostics (no charts).

Produces:
  output/nqdtc_baseline.txt       – NQDTC full diagnostic report
  output/vdubus_baseline.txt      – VdubusNQ full diagnostic report
  output/strategy_comparison.txt  – side-by-side comparison table

Both use 10 MNQ contracts (fixed_qty) for apples-to-apples comparison.
Standardized: same commission ($0.62/side), same data period, same equity.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────
MNQ_POINT_VALUE = 2.0       # $2 per point (1/10 of NQ)
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
MNQ_COMMISSION = 0.62       # per-side (IBKR micros — conservative)
MNQ_RT_COMM_FEES = 1.24
FIXED_QTY = 10
INITIAL_EQUITY = 100_000.0

DATA_DIR = ROOT / "backtest" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtest" / "output"

# ======================================================================
# 1. NQDTC v2.0
# ======================================================================

def run_nqdtc():
    from backtest.analysis.metrics import compute_max_drawdown, compute_metrics
    from backtest.analysis.nqdtc_diagnostics import nqdtc_full_diagnostic
    from backtest.analysis.reports import (
        format_summary,
        nqdtc_behavior_report,
        nqdtc_diagnostic_report,
        nqdtc_performance_report,
    )
    from backtest.config import SlippageConfig
    from backtest.config_nqdtc import NQDTCBacktestConfig
    from backtest.data.cache import load_bars
    from backtest.data.preprocessing import (
        align_higher_tf_to_5m,
        build_numpy_arrays,
        filter_eth,
        normalize_timezone,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )
    from backtest.engine.nqdtc_engine import NQDTCEngine

    report_path = OUTPUT_DIR / "nqdtc_baseline.txt"

    logger.info("=" * 60)
    logger.info("  NQDTC v2.0 Full Diagnostics – MNQ x %d fixed units", FIXED_QTY)
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    five_min_path = DATA_DIR / "NQ_5m.parquet"
    daily_path = DATA_DIR / "NQ_1d.parquet"
    if not five_min_path.exists():
        raise FileNotFoundError(f"Missing 5-min data: {five_min_path}")

    m5_df = normalize_timezone(load_bars(five_min_path))
    m5_df = filter_eth(m5_df)

    thirty_min_df = resample_5m_to_30m(m5_df)
    hourly_df = resample_5m_to_1h(m5_df)
    four_hour_df = resample_5m_to_4h(m5_df)

    if daily_path.exists():
        daily_df = normalize_timezone(load_bars(daily_path))
    else:
        daily_df = resample_5m_to_daily(m5_df)

    five_min_bars = build_numpy_arrays(m5_df)
    thirty_min = build_numpy_arrays(thirty_min_df)
    hourly = build_numpy_arrays(hourly_df)
    four_hour = build_numpy_arrays(four_hour_df)
    daily = build_numpy_arrays(daily_df)

    thirty_min_idx_map = align_higher_tf_to_5m(m5_df, thirty_min_df)
    hourly_idx_map = align_higher_tf_to_5m(m5_df, hourly_df)
    four_hour_idx_map = align_higher_tf_to_5m(m5_df, four_hour_df)
    daily_idx_map = align_higher_tf_to_5m(m5_df, daily_df)

    logger.info(
        "Loaded: %d 5m, %d 30m, %d 1H, %d 4H, %d daily bars",
        len(m5_df), len(thirty_min_df), len(hourly_df),
        len(four_hour_df), len(daily_df),
    )

    # ── Configure & run ───────────────────────────────────────────
    config = NQDTCBacktestConfig(
        symbols=["MNQ"],
        initial_equity=INITIAL_EQUITY,
        slippage=SlippageConfig(commission_per_contract=MNQ_COMMISSION),
        data_dir=DATA_DIR,
        fixed_qty=FIXED_QTY,
        tick_size=MNQ_TICK_SIZE,
        point_value=MNQ_POINT_VALUE,
    )

    logger.info("Running NQDTC v2.0 backtest engine...")
    engine = NQDTCEngine("MNQ", config)
    result = engine.run(
        five_min_bars, thirty_min, hourly, four_hour, daily,
        thirty_min_idx_map, hourly_idx_map, four_hour_idx_map, daily_idx_map,
    )

    # ── Build report ──────────────────────────────────────────────
    sections: list[str] = []

    sections.append(
        f"=== NQDTC v2.0 Backtest: MNQ ({FIXED_QTY} units fixed) ===\n"
        f"  Trades: {len(result.trades)}  "
        f"Breakouts evaluated: {result.breakouts_evaluated}  "
        f"Breakouts qualified: {result.breakouts_qualified}\n"
        f"  Entries placed: {result.entries_placed}  "
        f"Entries filled: {result.entries_filled}  "
        f"Gates blocked: {result.gates_blocked}"
    )

    metrics = None
    if result.trades:
        pnls = np.array([t.pnl_dollars for t in result.trades])
        risks = np.array([
            abs(t.entry_price - t.initial_stop) * config.point_value * t.qty
            for t in result.trades
        ])
        holds = np.array([t.bars_held_30m for t in result.trades])
        comms = np.array([t.commission for t in result.trades])

        metrics = compute_metrics(
            trade_pnls=pnls,
            trade_risks=risks,
            trade_hold_hours=holds,
            trade_commissions=comms,
            equity_curve=result.equity_curve,
            timestamps=result.timestamps,
            initial_equity=config.initial_equity,
        )

        sections.append(nqdtc_performance_report("MNQ", metrics))
        sections.append(format_summary(metrics))
        sections.append(nqdtc_diagnostic_report(result))
        sections.append(nqdtc_behavior_report(result.trades))
        sections.append(nqdtc_full_diagnostic(
            result.trades,
            signal_events=result.signal_events,
            equity_curve=result.equity_curve,
            initial_equity=config.initial_equity,
            point_value=config.point_value,
        ))
    else:
        sections.append("WARNING: No trades generated!")

    if result.shadow_summary:
        sections.append(result.shadow_summary)

    # Buy & hold baseline
    bh_closes = hourly.closes[np.isfinite(hourly.closes)]
    if len(bh_closes) >= 2:
        sp, ep = float(bh_closes[0]), float(bh_closes[-1])
        bh_pnl = (ep - sp) * MNQ_POINT_VALUE * FIXED_QTY
        bh_equity = INITIAL_EQUITY + (bh_closes - sp) * MNQ_POINT_VALUE * FIXED_QTY
        dd_pct, dd_dollar = compute_max_drawdown(bh_equity)
        valid_ts = hourly.times[:len(bh_closes)]
        if len(valid_ts) >= 2:
            delta = valid_ts[-1] - valid_ts[0]
            span_s = float(delta / np.timedelta64(1, "s")) if hasattr(delta, "astype") else delta.total_seconds()
            years = span_s / (365.25 * 24 * 3600)
        else:
            years = 1.0
        cagr = ((INITIAL_EQUITY + bh_pnl) / INITIAL_EQUITY) ** (1.0 / max(years, 0.01)) - 1.0
        sections.append(
            f"=== Buy & Hold Baseline ({FIXED_QTY} MNQ) ===\n"
            f"  Start price:      {sp:,.2f}\n"
            f"  End price:        {ep:,.2f}\n"
            f"  Price change:     {ep - sp:+,.2f} ({(ep/sp - 1)*100:+.1f}%)\n"
            f"  PnL ({FIXED_QTY} MNQ):     ${bh_pnl:+,.2f}\n"
            f"  Return on equity: {bh_pnl/INITIAL_EQUITY*100:+.1f}%\n"
            f"  CAGR:             {cagr:.1%}\n"
            f"  Max drawdown:     {dd_pct:.1%} (${dd_dollar:,.2f})\n"
            f"  Period:           {years:.2f} years"
        )

    # ── Save ──────────────────────────────────────────────────────
    report_text = "\n\n".join(sections) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))

    logger.info("NQDTC report saved: %s (%d bytes)", report_path, len(report_text))
    return metrics


# ======================================================================
# 2. VdubusNQ v4.4
# ======================================================================

def run_vdubus():
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

    report_path = OUTPUT_DIR / "vdubus_baseline.txt"
    symbol = "NQ"

    logger.info("=" * 60)
    logger.info("  VdubusNQ v4.4 Full Diagnostics – MNQ x %d fixed units", FIXED_QTY)
    logger.info("=" * 60)

    vdubus_data = _load_vdubus_data(symbol, DATA_DIR)

    # Patch strategy_3 config for MNQ
    orig_nq_spec = dict(C.NQ_SPEC)
    orig_rt_comm = C.RT_COMM_FEES
    C.NQ_SPEC["tick_value"] = MNQ_TICK_VALUE
    C.NQ_SPEC["point_value"] = MNQ_POINT_VALUE
    C.RT_COMM_FEES = MNQ_RT_COMM_FEES

    try:
        flags = VdubusAblationFlags(
            heat_cap=False,
            viability_filter=False,
        )

        config = VdubusBacktestConfig(
            symbols=[symbol],
            initial_equity=INITIAL_EQUITY,
            data_dir=DATA_DIR,
            slippage=SlippageConfig(commission_per_contract=MNQ_COMMISSION),
            fixed_qty=FIXED_QTY,
            tick_size=MNQ_TICK_SIZE,
            point_value=MNQ_POINT_VALUE,
            flags=flags,
        )

        logger.info(
            "Running VdubusNQ v4.4: MNQ, fixed_qty=%d, pv=$%.2f, comm=$%.2f/side",
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
        C.NQ_SPEC.update(orig_nq_spec)
        C.RT_COMM_FEES = orig_rt_comm

    if not result.trades:
        print("VdubusNQ v4.4 (MNQ): No trades generated.")
        return

    logger.info("Backtest complete: %d trades", len(result.trades))

    # ── Metrics ───────────────────────────────────────────────────
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

    # ── Build report ──────────────────────────────────────────────
    sections: list[str] = []

    sections.append(
        f"=== VdubusNQ v4.4 Baseline: MNQ (Micro E-mini Nasdaq 100) ===\n"
        f"Instrument:         MNQ\n"
        f"Point value:        ${MNQ_POINT_VALUE:.2f}\n"
        f"Tick size:          {MNQ_TICK_SIZE}\n"
        f"Tick value:         ${MNQ_TICK_VALUE:.2f}\n"
        f"Commission/side:    ${MNQ_COMMISSION:.2f}\n"
        f"Fixed qty:          {FIXED_QTY} contracts\n"
        f"Initial equity:     ${INITIAL_EQUITY:,.0f}\n"
        f"Disabled gates:     heat_cap, viability_filter (N/A for fixed-qty)"
    )

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

    # Buy & Hold comparison
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

            bh_equity = INITIAL_EQUITY + (closes - start_price) * MNQ_POINT_VALUE * FIXED_QTY
            bh_peak = np.maximum.accumulate(bh_equity)
            bh_dd_series = (bh_equity - bh_peak) / bh_peak
            bh_max_dd_pct = float(np.min(bh_dd_series)) * 100

            strat_return_pct = metrics.net_profit / INITIAL_EQUITY * 100

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

    sections.append(format_summary(metrics))

    logger.info("Generating full diagnostics...")
    diag = vdubus_full_diagnostic(
        result.trades,
        signal_events=result.signal_events,
        equity_curve=result.equity_curve,
        time_series=result.time_series,
    )
    sections.append(diag)

    if result.signal_events:
        logger.info("Generating filter attribution report...")
        attr = vdubus_filter_attribution_report(
            result.signal_events, result.trades,
            shadow_tracker=result.shadow_tracker,
        )
        sections.append(attr)

    if result.shadow_summary:
        sections.append(result.shadow_summary)

    # ── Save ──────────────────────────────────────────────────────
    report_text = "\n\n".join(sections) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))

    logger.info("VdubusNQ report saved: %s (%d bytes)", report_path, len(report_text))
    return metrics


# ======================================================================
# Cross-strategy comparison
# ======================================================================

def _build_comparison(nqdtc_metrics, vdubus_metrics) -> str:
    """Build a side-by-side comparison table for the two strategies."""
    lines = [
        "=" * 68,
        "  STRATEGY COMPARISON: NQDTC v2.0 vs VdubusNQ v4.4",
        "=" * 68,
        "",
        "Standardized assumptions:",
        f"  Instrument:       MNQ (Micro E-mini Nasdaq-100, $2/point)",
        f"  Fixed qty:        {FIXED_QTY} contracts per trade",
        f"  Commission:       ${MNQ_COMMISSION:.2f}/side (IBKR micros)",
        f"  Initial equity:   ${INITIAL_EQUITY:,.0f}",
        f"  Data period:      ~11 months (2025-03 to 2026-02)",
        "",
    ]

    n = nqdtc_metrics
    v = vdubus_metrics

    # Header
    hdr = f"{'Metric':28s} {'NQDTC v2.0':>16s} {'VdubusNQ v4.4':>16s}"
    sep = "-" * 62

    rows = [
        hdr, sep,
        f"{'Total trades':28s} {n.total_trades:>16d} {v.total_trades:>16d}",
        f"{'Win rate':28s} {n.win_rate:>15.1%} {v.win_rate:>15.1%}",
        f"{'Profit factor':28s} {n.profit_factor:>16.2f} {v.profit_factor:>16.2f}",
        f"{'Expectancy (R)':28s} {n.expectancy:>+16.3f} {v.expectancy:>+16.3f}",
        f"{'Expectancy ($)':28s} ${n.expectancy_dollar:>+14,.2f} ${v.expectancy_dollar:>+14,.2f}",
        f"{'Net profit':28s} ${n.net_profit:>+14,.2f} ${v.net_profit:>+14,.2f}",
        f"{'CAGR':28s} {n.cagr:>15.1%} {v.cagr:>15.1%}",
        f"{'Sharpe':28s} {n.sharpe:>16.2f} {v.sharpe:>16.2f}",
        f"{'Sortino':28s} {n.sortino:>16.2f} {v.sortino:>16.2f}",
        f"{'Calmar':28s} {n.calmar:>16.2f} {v.calmar:>16.2f}",
        f"{'Max DD (%)':28s} {n.max_drawdown_pct:>15.1%} {v.max_drawdown_pct:>15.1%}",
        f"{'Max DD ($)':28s} ${n.max_drawdown_dollar:>14,.2f} ${v.max_drawdown_dollar:>14,.2f}",
        f"{'Trades/month':28s} {n.trades_per_month:>16.1f} {v.trades_per_month:>16.1f}",
        f"{'Total commissions':28s} ${n.total_commissions:>14,.2f} ${v.total_commissions:>14,.2f}",
    ]

    # Risk-adjusted net profit: net profit / max drawdown dollar
    n_risk_adj = n.net_profit / n.max_drawdown_dollar if n.max_drawdown_dollar > 0 else 0
    v_risk_adj = v.net_profit / v.max_drawdown_dollar if v.max_drawdown_dollar > 0 else 0
    rows.append(sep)
    rows.append(f"{'Net profit / Max DD':28s} {n_risk_adj:>16.2f} {v_risk_adj:>16.2f}")

    lines.extend(rows)
    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"  Running both baselines: FIXED_QTY = {FIXED_QTY} MNQ")
    print(f"  Initial equity: ${INITIAL_EQUITY:,.0f}")
    print(f"  Commission/side: ${MNQ_COMMISSION:.2f}")
    print("=" * 60)

    print("\n\n>>> [1/2] NQDTC v2.0 <<<\n")
    nqdtc_metrics = run_nqdtc()

    print("\n\n>>> [2/2] VdubusNQ v4.4 <<<\n")
    vdubus_metrics = run_vdubus()

    # Cross-strategy comparison
    if nqdtc_metrics and vdubus_metrics:
        comparison = _build_comparison(nqdtc_metrics, vdubus_metrics)
        comp_path = OUTPUT_DIR / "strategy_comparison.txt"
        comp_path.write_text(comparison + "\n", encoding="utf-8")
        print(f"\n\n{comparison}")
        logger.info("Comparison saved: %s", comp_path)

    print("\n" + "=" * 60)
    print("  Both baselines complete.")
    print(f"  → output/nqdtc_baseline.txt")
    print(f"  → output/vdubus_baseline.txt")
    print(f"  → output/strategy_comparison.txt")
    print("=" * 60)
