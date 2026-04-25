"""BRS multi-symbol portfolio engine with synchronized shared-capital support."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtests.swing.config_brs import BRSConfig
from backtests.swing.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    build_numpy_arrays,
    filter_rth,
    normalize_timezone,
    resample_1h_to_4h,
)
from backtests.swing.engine.backtest_engine import SymbolResult
from backtests.swing.engine.brs_engine import BRSEngine

logger = logging.getLogger(__name__)


@dataclass
class BRSSymbolData:
    """Pre-loaded 3-timeframe data for one symbol."""

    daily: NumpyBars = field(
        default_factory=lambda: NumpyBars(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    )
    hourly: NumpyBars = field(
        default_factory=lambda: NumpyBars(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    )
    four_hour: NumpyBars = field(
        default_factory=lambda: NumpyBars(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    )
    daily_idx_map: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    four_hour_idx_map: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


@dataclass
class BRSPortfolioResult:
    """Combined results across all BRS symbols."""

    symbol_results: dict[str, SymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def timestamps(self) -> np.ndarray:
        return self.combined_timestamps


def load_brs_data(config: BRSConfig) -> dict[str, BRSSymbolData]:
    """Load and prepare 3-timeframe data for all BRS symbols."""
    from backtests.swing.data.cache import load_bars

    data: dict[str, BRSSymbolData] = {}

    for sym in config.symbols:
        try:
            h_path = config.data_dir / f"{sym}_1h.parquet"
            d_path = config.data_dir / f"{sym}_1d.parquet"

            if not h_path.exists() or not d_path.exists():
                logger.warning("Missing data for %s (need %s and %s), skipping", sym, h_path, d_path)
                continue

            h_df = normalize_timezone(load_bars(h_path))
            d_df = normalize_timezone(load_bars(d_path))

            start_bound = _normalize_bound(config.start_date)
            end_bound = _normalize_bound(config.end_date)
            if start_bound is not None:
                h_df = h_df[h_df.index >= start_bound]
                d_df = d_df[d_df.index >= start_bound]
            if end_bound is not None:
                h_df = h_df[h_df.index <= end_bound]
                d_df = d_df[d_df.index <= end_bound]

            h_df = filter_rth(h_df)
            if len(h_df) == 0 or len(d_df) == 0:
                logger.warning("No data for %s after filtering, skipping", sym)
                continue

            four_h_df = resample_1h_to_4h(h_df)
            data[sym] = BRSSymbolData(
                daily=build_numpy_arrays(d_df),
                hourly=build_numpy_arrays(h_df),
                four_hour=build_numpy_arrays(four_h_df),
                daily_idx_map=align_daily_to_hourly(h_df, d_df),
                four_hour_idx_map=align_4h_to_hourly(h_df, four_h_df),
            )
            logger.info(
                "Loaded %s: %d daily, %d hourly, %d 4H bars",
                sym,
                len(d_df),
                len(h_df),
                len(four_h_df),
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", sym, exc)
            continue

    return data


def run_brs_independent(
    data: dict[str, BRSSymbolData],
    config: BRSConfig,
) -> BRSPortfolioResult:
    """Run each symbol independently with BRS engine."""
    results: dict[str, SymbolResult] = {}
    qqq_regimes: list | None = None

    for sym in _run_order(data):
        sym_data = data[sym]
        engine = BRSEngine(symbol=sym, sym_cfg=config.get_symbol_config(sym), cfg=config)
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
        if sym == "QQQ":
            qqq_regimes = engine._regime_history

    combined_eq, combined_ts = _combine_equity_curves(results, config.initial_equity)
    return BRSPortfolioResult(
        symbol_results=results,
        combined_equity=combined_eq,
        combined_timestamps=combined_ts,
    )


def run_brs_synchronized(
    data: dict[str, BRSSymbolData],
    config: BRSConfig,
) -> BRSPortfolioResult:
    """Run symbols on a unified hourly timeline with shared capital and heat."""
    engines: dict[str, BRSEngine] = {}
    time_maps: dict[str, dict] = {}
    all_times: set = set()

    for sym in _run_order(data):
        sym_data = data[sym]
        engine = BRSEngine(
            symbol=sym,
            sym_cfg=config.get_symbol_config(sym),
            cfg=config,
            starting_equity=0.0,
        )
        if sym == "GLD" and "QQQ" in engines:
            engine._qqq_regimes = engines["QQQ"]._regime_history
        engine._prepare_run_context(
            daily=sym_data.daily,
            hourly=sym_data.hourly,
            four_hour=sym_data.four_hour,
            daily_idx_map=sym_data.daily_idx_map,
            four_hour_idx_map=sym_data.four_hour_idx_map,
        )
        engines[sym] = engine

        mapping = {}
        for idx, ts in enumerate(sym_data.hourly.times):
            mapping[ts] = idx
        time_maps[sym] = mapping
        all_times.update(mapping.keys())

    if not engines:
        return BRSPortfolioResult()

    combined_equity: list[float] = []
    combined_timestamps: list = []
    run_order = [sym for sym in _run_order(data) if sym in engines]

    for ts in sorted(all_times):
        for sym in run_order:
            bar_idx = time_maps[sym].get(ts)
            if bar_idx is None:
                continue

            portfolio_equity = _portfolio_equity(engines, config.initial_equity)
            reserved_risk = sum(engine.reserved_risk_dollars() for engine in engines.values())
            reserved_slots = sum(engine.reserved_position_slots() for engine in engines.values())

            engines[sym]._step_bar(
                bar_idx,
                sizing_equity=portfolio_equity,
                shared_reserved_risk_dollars=reserved_risk,
                shared_concurrent_positions=reserved_slots,
            )

        combined_equity.append(_portfolio_equity(engines, config.initial_equity))
        combined_timestamps.append(ts)

    for engine in engines.values():
        engine._finalize_end_of_data()
        if engine.equity_curve:
            engine.equity_curve[-1] = engine._mtm_equity()

    if combined_equity:
        combined_equity[-1] = _portfolio_equity(engines, config.initial_equity)

    symbol_results: dict[str, SymbolResult] = {}
    for sym, engine in engines.items():
        symbol_results[sym] = engine._build_result(equity_offset=config.initial_equity)

    return BRSPortfolioResult(
        symbol_results=symbol_results,
        combined_equity=np.array(combined_equity, dtype=np.float64),
        combined_timestamps=np.array(combined_timestamps),
    )


def _run_order(data: dict[str, BRSSymbolData]) -> list[str]:
    return sorted(data.keys(), key=lambda sym: (0 if sym == "QQQ" else 1, sym))


def _normalize_bound(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _portfolio_equity(engines: dict[str, BRSEngine], initial_equity: float) -> float:
    return initial_equity + sum(engine.equity for engine in engines.values()) + sum(
        engine.unrealized_pnl() for engine in engines.values()
    )


def _combine_equity_curves(
    results: dict[str, SymbolResult],
    initial_equity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-symbol equity curves into a portfolio curve."""
    if not results:
        return np.array([initial_equity]), np.array([])

    max_len = max(len(result.equity_curve) for result in results.values())
    combined = np.full(max_len, initial_equity, dtype=np.float64)

    for result in results.values():
        n = len(result.equity_curve)
        if n == 0:
            continue
        padded = np.full(max_len, result.equity_curve[-1])
        padded[:n] = result.equity_curve
        combined += padded - initial_equity

    longest_symbol = max(results, key=lambda sym: len(results[sym].timestamps))
    return combined, results[longest_symbol].timestamps
