"""Multi-symbol Breakout v3.3-ETF portfolio backtesting engine.

Two modes:
- run_breakout_independent: Each symbol runs its own BreakoutEngine (fast, for optimization)
- run_breakout_synchronized: All symbols step together with cross-symbol allocation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from strategies.swing.breakout.config import (
    CORR_LOOKBACK_BARS,
    MAX_PORTFOLIO_HEAT,
    SYMBOL_CONFIGS,
    SymbolConfig,
)
from strategies.swing.breakout.indicators import rolling_correlation
from strategies.swing.breakout.models import (
    CircuitBreakerState,
    Direction,
    PositionState,
)

from backtests.swing.config_breakout import BreakoutBacktestConfig
from backtests.swing.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    build_numpy_arrays,
    filter_rth,
    resample_1h_to_4h,
)
from backtests.swing.engine.breakout_engine import (
    _AblationPatch,
    BreakoutEngine,
    BreakoutSymbolResult,
)

logger = logging.getLogger(__name__)


@dataclass
class BreakoutPortfolioData:
    """Pre-loaded data for all symbols including 4H bars."""

    daily: dict[str, NumpyBars] = field(default_factory=dict)
    hourly: dict[str, NumpyBars] = field(default_factory=dict)
    four_hour: dict[str, NumpyBars] = field(default_factory=dict)
    daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    four_hour_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BreakoutHeatStats:
    """Portfolio heat utilization statistics."""

    avg_heat_pct: float = 0.0
    max_heat_pct: float = 0.0
    pct_time_at_limit: float = 0.0


@dataclass
class BreakoutPortfolioResult:
    """Combined results across all symbols."""

    symbol_results: dict[str, BreakoutSymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_stats: BreakoutHeatStats = field(default_factory=BreakoutHeatStats)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_breakout_data(
    symbols: list[str],
    data_dir,
) -> BreakoutPortfolioData:
    """Load 1H + 1D parquets, resample 1H->4H, build NumpyBars + idx maps.

    Caches preprocessed data as a pickle keyed on source file mtime+size
    to avoid redundant reprocessing across diagnostic/optimization runs.
    """
    import hashlib
    import pickle
    from pathlib import Path
    data_dir = Path(data_dir)

    # Build cache key from source file stats
    key_parts = []
    for sym in sorted(symbols):
        for suffix in ("_1h.parquet", "_1d.parquet"):
            p = data_dir / f"{sym}{suffix}"
            if p.exists():
                st = p.stat()
                key_parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
    cache_hash = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
    cache_path = data_dir / f".cache_breakout_{cache_hash}.pkl"

    # Try cache hit
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                portfolio = pickle.load(f)
            logger.info("Loaded cached breakout data from %s", cache_path)
            return portfolio
        except Exception:
            logger.warning("Cache load failed, rebuilding")

    # Cache miss -- build from scratch
    portfolio = BreakoutPortfolioData()

    for sym in symbols:
        hourly_path = data_dir / f"{sym}_1h.parquet"
        daily_path = data_dir / f"{sym}_1d.parquet"

        if not hourly_path.exists() or not daily_path.exists():
            logger.warning("Missing data for %s, skipping", sym)
            continue

        hourly_df = pd.read_parquet(hourly_path)
        daily_df = pd.read_parquet(daily_path)

        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            hourly_df.index = pd.DatetimeIndex(hourly_df.index)
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            daily_df.index = pd.DatetimeIndex(daily_df.index)

        # Convert to ET and filter to RTH — engine expects ET timestamps
        # for entry window checks (is_entry_window_open compares .time()
        # against ET entry_window_start/end).
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        if hourly_df.index.tz is not None:
            hourly_df = hourly_df.tz_convert(et)
        else:
            hourly_df = hourly_df.tz_localize("UTC").tz_convert(et)
        hourly_df = filter_rth(hourly_df)

        four_hour_df = resample_1h_to_4h(hourly_df)

        portfolio.hourly[sym] = build_numpy_arrays(hourly_df)
        portfolio.daily[sym] = build_numpy_arrays(daily_df)
        portfolio.four_hour[sym] = build_numpy_arrays(four_hour_df)

        portfolio.daily_idx_maps[sym] = align_daily_to_hourly(hourly_df, daily_df)
        portfolio.four_hour_idx_maps[sym] = align_4h_to_hourly(hourly_df, four_hour_df)

    # Write cache (clean stale caches first)
    for stale in data_dir.glob(".cache_breakout_*.pkl"):
        if stale != cache_path:
            try:
                stale.unlink()
            except OSError:
                pass
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(portfolio, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached breakout data to %s", cache_path)
    except Exception:
        logger.warning("Failed to write cache")

    return portfolio


# ---------------------------------------------------------------------------
# Independent mode (fast, for optimization)
# ---------------------------------------------------------------------------

def run_breakout_independent(
    data: BreakoutPortfolioData,
    bt_config: BreakoutBacktestConfig,
) -> BreakoutPortfolioResult:
    """Run each symbol independently (fast path for optimization)."""
    results: dict[str, BreakoutSymbolResult] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily or sym not in data.four_hour:
            logger.warning("No data for %s, skipping", sym)
            continue

        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)

        engine = BreakoutEngine(symbol=sym, cfg=cfg, bt_config=bt_config,
                                point_value=cfg.multiplier)
        results[sym] = engine.run(
            daily=data.daily[sym],
            hourly=data.hourly[sym],
            four_hour=data.four_hour[sym],
            daily_idx_map=data.daily_idx_maps[sym],
            four_hour_idx_map=data.four_hour_idx_maps[sym],
        )

    combined_equity, combined_ts = _combine_equity_curves(results, bt_config.initial_equity)
    return BreakoutPortfolioResult(
        symbol_results=results,
        combined_equity=combined_equity,
        combined_timestamps=combined_ts,
    )


# ---------------------------------------------------------------------------
# Synchronized mode (cross-symbol allocation)
# ---------------------------------------------------------------------------

def run_breakout_synchronized(
    data: BreakoutPortfolioData,
    bt_config: BreakoutBacktestConfig,
) -> BreakoutPortfolioResult:
    """Run all symbols stepping through time with cross-symbol constraints.

    Shares PositionState, CircuitBreakerState, and correlation map
    across all per-symbol BreakoutEngines. On each unified hourly bar:
    1. Each symbol steps (fills, daily boundary, 4H, hourly, position mgmt)
    2. Portfolio equity is tracked across symbols
    3. Correlation map is updated at daily boundaries
    """
    # Shared state
    shared_positions: dict[str, PositionState] = {}
    shared_cb = CircuitBreakerState()
    shared_corr: dict[tuple[str, str], float] = {}

    engines: dict[str, BreakoutEngine] = {}
    configs: dict[str, SymbolConfig] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily or sym not in data.four_hour:
            continue
        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)
        configs[sym] = cfg

        eng = BreakoutEngine(
            symbol=sym,
            cfg=cfg,
            bt_config=bt_config,
            point_value=cfg.multiplier,
            external_positions=shared_positions,
            external_circuit_breaker=shared_cb,
            external_correlation_map=shared_corr,
        )
        eng._init_equity_arrays(len(data.hourly[sym]), data.hourly[sym].times.dtype)
        engines[sym] = eng

    if not engines:
        return BreakoutPortfolioResult()

    # Build unified timestamp index
    all_times_set: set = set()
    time_sets: dict[str, dict] = {}
    for sym in engines:
        times = data.hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            key = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[key] = i
        time_sets[sym] = mapping
        all_times_set.update(mapping.keys())

    unified_ts = sorted(all_times_set)
    init_eq = bt_config.initial_equity
    prev_sym_equity: dict[str, float] = {sym: init_eq for sym in engines}
    portfolio_equity = init_eq

    equity_curve: list[float] = []
    timestamps: list = []
    heat_samples: list[float] = []
    warmup_d = bt_config.warmup_daily
    warmup_h = bt_config.warmup_hourly
    warmup_4h = bt_config.warmup_4h

    prev_daily_idxs: dict[str, int] = {sym: -1 for sym in engines}

    with _AblationPatch(bt_config.flags, bt_config.param_overrides):
        for sym, engine in engines.items():
            engine._precompute_indicators(data.daily[sym], data.hourly[sym], data.four_hour[sym])
            engine._init_histories(data.daily[sym], warmup_d)
            engine._init_slot_medians(data.hourly[sym], warmup_h)

        for ts in unified_ts:
            # Step each symbol
            for sym, engine in engines.items():
                bar_idx = time_sets[sym].get(ts)
                if bar_idx is None:
                    continue

                engine.sizing_equity = portfolio_equity
                engine._step_bar(
                    data.daily[sym], data.hourly[sym], data.four_hour[sym],
                    data.daily_idx_maps[sym], data.four_hour_idx_maps[sym],
                    bar_idx,
                    warmup_d, warmup_h, warmup_4h,
                )

                # Detect daily boundary for correlation update
                d_idx = int(data.daily_idx_maps[sym][bar_idx])
                if d_idx != prev_daily_idxs[sym]:
                    prev_daily_idxs[sym] = d_idx

            # Portfolio equity from per-symbol deltas
            for sym, eng in engines.items():
                delta = eng.equity - prev_sym_equity[sym]
                portfolio_equity += delta
                prev_sym_equity[sym] = eng.equity
            equity_curve.append(portfolio_equity)
            timestamps.append(ts)

            # Update correlations periodically (every 20 bars)
            if len(timestamps) % 20 == 0:
                _update_correlations(engines, shared_corr)

            # Track portfolio heat
            total_heat = 0.0
            for sym, eng in engines.items():
                pos = eng.active_position
                if pos is not None and pos.qty_open > 0:
                    risk_dollars = abs(pos.fill_price - pos.current_stop) * pos.qty_open
                    total_heat += risk_dollars
            heat_pct = total_heat / portfolio_equity if portfolio_equity > 0 else 0.0
            heat_samples.append(heat_pct)

        for sym, engine in engines.items():
            if engine.active_position is None:
                continue
            hourly = data.hourly[sym]
            engine._flatten_at_end_of_data(
                hourly.closes[-1],
                engine._to_datetime(hourly.times[-1]),
            )
            delta = engine.equity - prev_sym_equity[sym]
            portfolio_equity += delta
            prev_sym_equity[sym] = engine.equity
        if equity_curve:
            equity_curve[-1] = portfolio_equity

    # Compute heat stats
    heat_arr = np.array(heat_samples) if heat_samples else np.array([0.0])
    heat = BreakoutHeatStats(
        avg_heat_pct=float(np.mean(heat_arr)),
        max_heat_pct=float(np.max(heat_arr)),
        pct_time_at_limit=float(np.mean(heat_arr >= MAX_PORTFOLIO_HEAT)) * 100,
    )

    # Build per-symbol results
    results: dict[str, BreakoutSymbolResult] = {}
    for sym, engine in engines.items():
        results[sym] = BreakoutSymbolResult(
            symbol=sym,
            trades=engine.trades,
            signal_events=engine.signal_events,
            equity_curve=engine._eq_arr[:engine._bar_idx].copy(),
            timestamps=engine._ts_arr[:engine._bar_idx].copy(),
            total_commission=engine.total_commission,
            entries_placed=engine.entries_placed,
            entries_filled=engine.entries_filled,
            entries_expired=engine.entries_expired,
            entries_rejected=engine.entries_rejected,
            entries_blocked=engine.entries_blocked,
            adds_placed=engine.adds_placed,
            adds_filled=engine.adds_filled,
            campaigns_activated=engine.campaigns_activated,
            breakouts_qualified=engine.breakouts_qualified,
            dirty_episodes=engine.dirty_episodes,
            continuations_entered=engine.continuations_entered,
            regime_bars_bull=engine.regime_bars_bull,
            regime_bars_bear=engine.regime_bars_bear,
            regime_bars_chop=engine.regime_bars_chop,
        )

    return BreakoutPortfolioResult(
        symbol_results=results,
        combined_equity=np.array(equity_curve),
        combined_timestamps=np.array(timestamps),
        heat_stats=heat,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combine_equity_curves(
    results: dict[str, BreakoutSymbolResult],
    initial_equity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-symbol equity curves into a portfolio curve."""
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

    longest_sym = max(results, key=lambda s: len(results[s].timestamps))
    combined_ts = results[longest_sym].timestamps
    return combined, combined_ts


def _update_correlations(
    engines: dict[str, BreakoutEngine],
    corr_map: dict[tuple[str, str], float],
) -> None:
    """Update rolling correlations between all symbol pairs."""
    symbols = list(engines.keys())
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sym_a, sym_b = symbols[i], symbols[j]
            hs_a = engines[sym_a].hourly_state
            hs_b = engines[sym_b].hourly_state
            if len(hs_a.closes) < 20 or len(hs_b.closes) < 20:
                continue
            c_a = np.array(hs_a.closes[-CORR_LOOKBACK_BARS * 4:])
            c_b = np.array(hs_b.closes[-CORR_LOOKBACK_BARS * 4:])
            if len(c_a) > 4 and len(c_b) > 4:
                ret_a = np.diff(c_a[::4]) / c_a[::4][:-1]
                ret_b = np.diff(c_b[::4]) / c_b[::4][:-1]
                corr = rolling_correlation(ret_a, ret_b, CORR_LOOKBACK_BARS)
                pair = tuple(sorted([sym_a, sym_b]))
                corr_map[pair] = corr


def _apply_overrides(cfg: SymbolConfig, overrides: dict[str, float]) -> SymbolConfig:
    """Create a new SymbolConfig with parameter overrides applied."""
    if not overrides:
        return cfg

    changes: dict[str, object] = {}
    for key, value in overrides.items():
        suffix = f"_{cfg.symbol}"
        field_name = key[:-len(suffix)] if key.endswith(suffix) else key
        if hasattr(cfg, field_name):
            current = getattr(cfg, field_name)
            changes[field_name] = int(round(value)) if isinstance(current, int) else float(value)

    if not changes:
        return cfg

    from dataclasses import asdict
    d = asdict(cfg)
    d.update(changes)
    return SymbolConfig(**d)
