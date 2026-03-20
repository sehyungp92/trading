"""Run NQDTC v2.0 full diagnostics for MNQ (10 fixed units).

Reusable diagnostic script — run anytime to get the current state of strategy_2.

Produces:
  output/nqdtc_v2.txt  – full diagnostic report (all sections)

Metrics note:
  - CAGR, Sharpe, Sortino, MaxDD, Calmar are derived from the equity curve
    (ground truth — tracks actual fills including commissions).
  - Trade-level PnL and R-multiples use initial_stop_price for the risk
    denominator (fixed in engine._close_position). Prior to the fix,
    post-BE stop migration inflated R by ~0.7 on average.
  - Profit factor and expectancy use recorded trade_pnls (pre-commission).
  - An equity reconciliation section cross-checks recorded vs equity-curve PnL.
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

# ── Instrument constants (MNQ) ──────────────────────────────────────
MNQ_POINT_VALUE = 2.0       # $2 per point (1/10 of NQ)
MNQ_TICK_SIZE = 0.25
MNQ_COMMISSION = 0.62       # per-side (IBKR micros — conservative)
FIXED_QTY = 10
INITIAL_EQUITY = 100_000.0

DATA_DIR = ROOT / "backtest" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtest" / "output"
REPORT_PATH = OUTPUT_DIR / "nqdtc_v2.txt"


# ── Data loading ─────────────────────────────────────────────────────

def _load_data(data_dir: Path) -> dict:
    """Load NQ 5m data and resample to higher timeframes."""
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

    five_min_path = data_dir / "NQ_5m.parquet"
    daily_path = data_dir / "NQ_1d.parquet"

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

    return {
        "five_min_bars": five_min_bars,
        "thirty_min": thirty_min,
        "hourly": hourly,
        "four_hour": four_hour,
        "daily": daily,
        "thirty_min_idx_map": thirty_min_idx_map,
        "hourly_idx_map": hourly_idx_map,
        "four_hour_idx_map": four_hour_idx_map,
        "daily_idx_map": daily_idx_map,
    }


# ── Buy-and-hold baseline ───────────────────────────────────────────

def _buy_and_hold_section(
    hourly_closes: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
) -> str:
    """Buy-and-hold report for 10 MNQ contracts."""
    from backtest.analysis.metrics import compute_max_drawdown

    closes = hourly_closes[np.isfinite(hourly_closes)]
    if len(closes) < 2:
        return "Buy & hold: insufficient data"

    sp, ep = float(closes[0]), float(closes[-1])
    pnl = (ep - sp) * MNQ_POINT_VALUE * FIXED_QTY

    bh_equity = initial_equity + (closes - sp) * MNQ_POINT_VALUE * FIXED_QTY
    dd_pct, dd_dollar = compute_max_drawdown(bh_equity)

    valid_ts = timestamps[:len(closes)]
    if len(valid_ts) >= 2:
        delta = valid_ts[-1] - valid_ts[0]
        span_s = (
            float(delta / np.timedelta64(1, "s"))
            if hasattr(delta, "astype")
            else delta.total_seconds()
        )
        years = span_s / (365.25 * 24 * 3600)
    else:
        years = 1.0

    cagr = (
        ((initial_equity + pnl) / initial_equity) ** (1.0 / max(years, 0.01)) - 1.0
    )

    return "\n".join([
        f"=== Buy & Hold Baseline ({FIXED_QTY} MNQ) ===",
        f"  Start price:      {sp:,.2f}",
        f"  End price:        {ep:,.2f}",
        f"  Price change:     {ep - sp:+,.2f} ({(ep/sp - 1)*100:+.1f}%)",
        f"  PnL ({FIXED_QTY} MNQ):     ${pnl:+,.2f}",
        f"  Return on equity: {pnl/initial_equity*100:+.1f}%",
        f"  CAGR:             {cagr:.1%}",
        f"  Max drawdown:     {dd_pct:.1%} (${dd_dollar:,.2f})",
        f"  Period:           {years:.2f} years",
    ])


# ── Main ─────────────────────────────────────────────────────────────

def main():
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
    from backtest.engine.nqdtc_engine import NQDTCEngine

    logger.info("=" * 60)
    logger.info("  NQDTC v2.0 Full Diagnostics – MNQ x %d fixed units", FIXED_QTY)
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    data = _load_data(DATA_DIR)

    # ── Configure & run ──────────────────────────────────────────
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
        data["five_min_bars"],
        data["thirty_min"],
        data["hourly"],
        data["four_hour"],
        data["daily"],
        data["thirty_min_idx_map"],
        data["hourly_idx_map"],
        data["four_hour_idx_map"],
        data["daily_idx_map"],
    )

    # ── Build report sections ────────────────────────────────────
    sections: list[str] = []

    # Header with instrument metadata
    sections.append(
        f"=== NQDTC v2.0 Backtest: MNQ ({FIXED_QTY} units fixed) ===\n"
        f"Instrument:         MNQ (Micro E-mini Nasdaq-100)\n"
        f"Point value:        ${MNQ_POINT_VALUE:.2f}\n"
        f"Tick size:          {MNQ_TICK_SIZE}\n"
        f"Commission/side:    ${MNQ_COMMISSION:.2f}\n"
        f"Fixed qty:          {FIXED_QTY} contracts\n"
        f"Initial equity:     ${INITIAL_EQUITY:,.0f}\n"
        f"Primary feed:       5-minute bars\n"
        f"\n"
        f"  Trades: {len(result.trades)}  "
        f"Breakouts evaluated: {result.breakouts_evaluated}  "
        f"Breakouts qualified: {result.breakouts_qualified}\n"
        f"  Entries placed: {result.entries_placed}  "
        f"Entries filled: {result.entries_filled}  "
        f"Gates blocked: {result.gates_blocked}"
    )

    metrics = None
    if result.trades:
        # Risk uses initial_stop (frozen at entry) — correct after R-fix
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

        # Performance summary (equity-curve-based metrics)
        sections.append(nqdtc_performance_report("MNQ", metrics))
        sections.append(format_summary(metrics))

        # Signal pipeline, regime distribution
        sections.append(nqdtc_diagnostic_report(result))

        # Entry subtypes, exits, sessions, TP rates, R-distribution
        sections.append(nqdtc_behavior_report(result.trades))

        # Full strategy-specific diagnostics (28+ sections)
        sections.append(nqdtc_full_diagnostic(
            result.trades,
            signal_events=result.signal_events,
            equity_curve=result.equity_curve,
            initial_equity=config.initial_equity,
            point_value=config.point_value,
        ))
    else:
        sections.append("WARNING: No trades generated!")

    # Shadow simulation summary
    if result.shadow_summary:
        sections.append(result.shadow_summary)

    # Buy-and-hold baseline
    hourly = data["hourly"]
    sections.append(_buy_and_hold_section(
        hourly.closes, hourly.times, config.initial_equity,
    ))

    # ── Save report ──────────────────────────────────────────────
    report_text = "\n\n".join(sections) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))

    logger.info("Report saved: %s (%d bytes)", REPORT_PATH, len(report_text))
    return metrics


if __name__ == "__main__":
    main()
