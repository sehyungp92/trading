"""BRS multi-symbol portfolio engine — 3-timeframe data loader + orchestrator.

Follows breakout_portfolio_engine.py pattern for 3-timeframe loading
(daily + hourly + 4H) with resample_1h_to_4h + alignment maps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backtest.config_brs import BRSConfig, BRS_SYMBOL_DEFAULTS
from backtest.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    build_numpy_arrays,
    filter_rth,
    normalize_timezone,
    resample_1h_to_4h,
)
from backtest.engine.backtest_engine import SymbolResult
from backtest.engine.brs_engine import BRSEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class BRSSymbolData:
    """Pre-loaded 3-timeframe data for one symbol."""
    daily: NumpyBars = field(default_factory=lambda: NumpyBars(
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])))
    hourly: NumpyBars = field(default_factory=lambda: NumpyBars(
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])))
    four_hour: NumpyBars = field(default_factory=lambda: NumpyBars(
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])))
    daily_idx_map: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    four_hour_idx_map: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


@dataclass
class BRSPortfolioResult:
    """Combined results across all BRS symbols."""
    symbol_results: dict[str, SymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_brs_data(config: BRSConfig) -> dict[str, BRSSymbolData]:
    """Load and prepare 3-timeframe data for all BRS symbols.

    Pattern: load parquet → normalize_timezone → filter_rth →
             resample_1h_to_4h → build_numpy_arrays → align maps
    """
    from backtest.data.cache import load_bars

    data: dict[str, BRSSymbolData] = {}

    for sym in config.symbols:
        try:
            # Load raw parquet
            h_path = config.data_dir / f"{sym}_1h.parquet"
            d_path = config.data_dir / f"{sym}_1d.parquet"

            if not h_path.exists() or not d_path.exists():
                logger.warning("Missing data for %s (need %s and %s), skipping", sym, h_path, d_path)
                continue

            h_df = normalize_timezone(load_bars(h_path))
            d_df = normalize_timezone(load_bars(d_path))

            # Filter to date range
            if config.start_date:
                h_df = h_df[h_df.index >= config.start_date]
                d_df = d_df[d_df.index >= config.start_date]
            if config.end_date:
                h_df = h_df[h_df.index <= config.end_date]
                d_df = d_df[d_df.index <= config.end_date]

            # Filter RTH
            h_df = filter_rth(h_df)

            if len(h_df) == 0 or len(d_df) == 0:
                logger.warning("No data for %s after filtering, skipping", sym)
                continue

            # Resample 1H → 4H
            four_h_df = resample_1h_to_4h(h_df)

            # Build numpy arrays
            sym_data = BRSSymbolData(
                daily=build_numpy_arrays(d_df),
                hourly=build_numpy_arrays(h_df),
                four_hour=build_numpy_arrays(four_h_df),
                daily_idx_map=align_daily_to_hourly(h_df, d_df),
                four_hour_idx_map=align_4h_to_hourly(h_df, four_h_df),
            )
            data[sym] = sym_data
            logger.info("Loaded %s: %d daily, %d hourly, %d 4H bars",
                       sym, len(d_df), len(h_df), len(four_h_df))

        except Exception as e:
            logger.error("Failed to load %s: %s", sym, e)
            continue

    return data


# ---------------------------------------------------------------------------
# Independent run (no cross-symbol coordination except QQQ regime)
# ---------------------------------------------------------------------------

def run_brs_independent(
    data: dict[str, BRSSymbolData],
    config: BRSConfig,
) -> BRSPortfolioResult:
    """Run each symbol independently with BRS engine.

    Runs QQQ first to extract regime timeline for GLD's L1 cross-symbol logic.
    """
    results: dict[str, SymbolResult] = {}
    qqq_regimes: list | None = None

    # Run QQQ first if present (needed for cross-symbol GLD logic)
    run_order = sorted(data.keys(), key=lambda s: (0 if s == "QQQ" else 1, s))

    for sym in run_order:
        sym_data = data[sym]
        sym_cfg = config.get_symbol_config(sym)

        engine = BRSEngine(
            symbol=sym,
            sym_cfg=sym_cfg,
            cfg=config,
        )

        # Inject QQQ regime timeline for GLD
        if sym == "GLD" and qqq_regimes is not None:
            engine._qqq_regimes = qqq_regimes

        result = engine.run(
            daily=sym_data.daily,
            hourly=sym_data.hourly,
            four_hour=sym_data.four_hour,
            daily_idx_map=sym_data.daily_idx_map,
            four_hour_idx_map=sym_data.four_hour_idx_map,
        )
        results[sym] = result

        # Extract QQQ regime timeline for GLD cross-symbol logic
        if sym == "QQQ":
            qqq_regimes = engine._regime_history

        logger.info("%s: %d trades, final equity delta: $%.0f",
                   sym, len(result.trades),
                   result.equity_curve[-1] - config.initial_equity if len(result.equity_curve) > 0 else 0)

    # Combine equity curves
    combined_eq, combined_ts = _combine_equity_curves(results, config.initial_equity)

    return BRSPortfolioResult(
        symbol_results=results,
        combined_equity=combined_eq,
        combined_timestamps=combined_ts,
    )


def _combine_equity_curves(
    results: dict[str, SymbolResult],
    initial_equity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-symbol equity curves into portfolio-level."""
    if not results:
        return np.array([initial_equity]), np.array([])

    max_len = max(len(r.equity_curve) for r in results.values())
    combined = np.full(max_len, initial_equity, dtype=np.float64)

    for r in results.values():
        n = len(r.equity_curve)
        if n == 0:
            continue
        padded = np.full(max_len, r.equity_curve[-1] if n > 0 else initial_equity)
        padded[:n] = r.equity_curve
        combined += (padded - initial_equity)

    # Use timestamps from longest series
    longest_sym = max(results, key=lambda s: len(results[s].timestamps))
    combined_ts = results[longest_sym].timestamps

    return combined, combined_ts
