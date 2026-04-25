"""Multi-symbol portfolio backtesting engine.

Two modes:
- run_independent: Each symbol runs its own BacktestEngine (fast, for optimization)
- run_synchronized: All symbols step together with portfolio allocation (accurate)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from strategies.swing.atrss import allocator
from strategies.swing.atrss.config import SYMBOL_CONFIGS, SymbolConfig
from strategies.swing.atrss.models import Direction

from backtests.swing.analysis.shadow_tracker import FilterStats, ShadowTracker
from backtests.swing.config import BacktestConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.backtest_engine import BacktestEngine, SymbolResult, _AblationPatch

logger = logging.getLogger(__name__)


@dataclass
class PortfolioData:
    """Pre-loaded data for all symbols."""

    daily: dict[str, NumpyBars] = field(default_factory=dict)
    hourly: dict[str, NumpyBars] = field(default_factory=dict)
    daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class HeatStats:
    """Portfolio heat utilization statistics."""

    avg_heat_pct: float = 0.0
    max_heat_pct: float = 0.0
    pct_time_at_limit: float = 0.0  # % of bars where heat >= MAX_PORTFOLIO_HEAT


@dataclass
class PortfolioResult:
    """Combined results across all symbols."""

    symbol_results: dict[str, SymbolResult] = field(default_factory=dict)
    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    filter_summary: dict[str, FilterStats] = field(default_factory=dict)
    heat_stats: HeatStats = field(default_factory=HeatStats)


def _get_point_value(symbol: str) -> float:
    cfg = SYMBOL_CONFIGS.get(symbol)
    return cfg.multiplier if cfg else 1.0


def run_independent(
    data: PortfolioData,
    bt_config: BacktestConfig,
) -> PortfolioResult:
    """Run each symbol independently (fast path for optimization)."""
    results: dict[str, SymbolResult] = {}
    engines: dict[str, BacktestEngine] = {}
    shadow = ShadowTracker() if bt_config.track_shadows else None
    configs: dict[str, SymbolConfig] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily:
            logger.warning("No data for %s, skipping", sym)
            continue

        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)
        configs[sym] = cfg

        engine = BacktestEngine(
            symbol=sym, cfg=cfg, bt_config=bt_config,
            point_value=_get_point_value(sym),
        )
        if shadow:
            engine.on_rejection = shadow.record_rejection
        engines[sym] = engine
        results[sym] = engine.run(
            daily=data.daily[sym],
            hourly=data.hourly[sym],
            daily_idx_map=data.daily_idx_maps[sym],
        )

    filter_summary = _run_shadow_sim(shadow, engines, configs, data, bt_config)

    combined_equity, combined_ts = _combine_equity_curves(results, bt_config.initial_equity)
    return PortfolioResult(
        symbol_results=results,
        combined_equity=combined_equity,
        combined_timestamps=combined_ts,
        filter_summary=filter_summary,
    )


def run_synchronized(
    data: PortfolioData,
    bt_config: BacktestConfig,
) -> PortfolioResult:
    """Run all symbols stepping through time with cross-symbol allocation.

    Steps through a unified hourly timestamp index. On each bar:
    1. Each symbol updates state, processes fills, manages positions
    2. Candidates from all symbols are collected
    3. allocator.allocate() ranks and filters with portfolio heat caps
    4. Only accepted candidates are submitted
    """
    engines: dict[str, BacktestEngine] = {}
    configs: dict[str, SymbolConfig] = {}

    for sym in bt_config.symbols:
        if sym not in data.hourly or sym not in data.daily:
            continue
        cfg = SYMBOL_CONFIGS.get(sym)
        if cfg is None:
            continue
        cfg = _apply_overrides(cfg, bt_config.param_overrides)
        configs[sym] = cfg
        engines[sym] = BacktestEngine(
            symbol=sym, cfg=cfg, bt_config=bt_config,
            point_value=_get_point_value(sym),
        )

    if not engines:
        return PortfolioResult()

    shadow = ShadowTracker() if bt_config.track_shadows else None
    if shadow:
        for eng in engines.values():
            eng.on_rejection = shadow.record_rejection

    # Mock instruments for allocator (needs .point_value)
    instruments = {sym: SimpleNamespace(point_value=_get_point_value(sym)) for sym in engines}

    # Build unified timestamp index from all symbols
    time_sets: dict[str, dict] = {}
    all_times_set: set = set()
    for sym in engines:
        times = data.hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            key = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[key] = i
        time_sets[sym] = mapping
        all_times_set.update(mapping.keys())

    unified_ts = sorted(all_times_set)
    warmup_d = bt_config.warmup_daily
    warmup_h = bt_config.warmup_hourly
    init_eq = bt_config.initial_equity
    prev_sym_equity: dict[str, float] = {sym: init_eq for sym in engines}
    portfolio_equity = init_eq

    equity_curve: list[float] = []
    timestamps: list = []
    heat_samples: list[float] = []

    from strategies.swing.atrss.config import MAX_PORTFOLIO_HEAT

    with _AblationPatch(bt_config.flags, bt_config.param_overrides):
        for t in unified_ts:
            all_candidates = []

            for sym, engine in engines.items():
                bar_idx = time_sets[sym].get(t)
                if bar_idx is None:
                    continue

                engine.sizing_equity = portfolio_equity

                candidates = engine.step_bar(
                    data.daily[sym], data.hourly[sym],
                    data.daily_idx_maps[sym], bar_idx,
                    warmup_d, warmup_h,
                )
                all_candidates.extend(candidates)

            if all_candidates:
                positions = {
                    sym: eng.position for sym, eng in engines.items()
                    if eng.position.direction != Direction.FLAT
                }
                daily_states = {
                    sym: eng.daily_state for sym, eng in engines.items()
                    if eng.daily_state is not None
                }
                hourly_states = {
                    sym: eng.hourly_state for sym, eng in engines.items()
                    if eng.hourly_state is not None
                }

                accepted = allocator.allocate(
                    all_candidates, positions, daily_states,
                    portfolio_equity, instruments, hourly_states,
                )

                bar_time = engines[next(iter(engines))]._to_datetime(
                    t if not hasattr(t, 'item') else t
                )
                for cand in accepted:
                    engines[cand.symbol].submit_candidate(cand, bar_time)

            # Portfolio equity = sum of per-symbol PnL deltas
            for sym, eng in engines.items():
                delta = eng.equity - prev_sym_equity[sym]
                portfolio_equity += delta
                prev_sym_equity[sym] = eng.equity
            equity_curve.append(portfolio_equity)
            timestamps.append(t)

            # Track portfolio heat utilization
            total_heat = 0.0
            for sym, eng in engines.items():
                pos = eng.position
                if pos.direction != Direction.FLAT and pos.base_leg is not None:
                    risk_dollars = abs(pos.base_leg.entry_price - pos.current_stop) * _get_point_value(sym) * pos.total_qty
                    total_heat += risk_dollars
            heat_pct = total_heat / portfolio_equity if portfolio_equity > 0 else 0.0
            heat_samples.append(heat_pct)

    # Compute heat utilization stats
    heat_arr = np.array(heat_samples) if heat_samples else np.array([0.0])
    heat = HeatStats(
        avg_heat_pct=float(np.mean(heat_arr)),
        max_heat_pct=float(np.max(heat_arr)),
        pct_time_at_limit=float(np.mean(heat_arr >= MAX_PORTFOLIO_HEAT)) * 100,
    )

    # Build per-symbol results
    results: dict[str, SymbolResult] = {}
    for sym, engine in engines.items():
        results[sym] = SymbolResult(
            symbol=sym,
            trades=engine.trades,
            equity_curve=np.array(engine.equity_curve),
            timestamps=np.array(engine.timestamps),
            total_commission=engine.total_commission,
            bias_days_long=engine._bias_days_long,
            bias_days_short=engine._bias_days_short,
            bias_days_flat=engine._bias_days_flat,
            funnel=engine._funnel,
            order_metadata=engine._order_metadata,
        )

    filter_summary = _run_shadow_sim(shadow, engines, configs, data, bt_config)

    return PortfolioResult(
        symbol_results=results,
        combined_equity=np.array(equity_curve),
        combined_timestamps=np.array(timestamps),
        filter_summary=filter_summary,
        heat_stats=heat,
    )


def _combine_equity_curves(
    results: dict[str, SymbolResult],
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


def _run_shadow_sim(
    shadow: ShadowTracker | None,
    engines: dict[str, BacktestEngine],
    configs: dict[str, SymbolConfig],
    data: PortfolioData,
    bt_config: BacktestConfig,
) -> dict[str, FilterStats]:
    """Run shadow simulation on rejections collected during a backtest run."""
    if not shadow or not shadow.rejections:
        return {}
    syms = list(engines)
    # Run within patch context so shadow sim uses same overrides as main run
    with _AblationPatch(bt_config.flags, bt_config.param_overrides):
        shadow.simulate_shadows(
            hourly_data={s: (data.hourly[s].opens, data.hourly[s].highs, data.hourly[s].lows,
                             data.hourly[s].closes, data.hourly[s].volumes) for s in syms},
            hourly_times={s: data.hourly[s].times for s in syms},
            configs=configs,
            point_values={s: _get_point_value(s) for s in syms},
            daily_states={s: engines[s]._daily_state_by_idx for s in syms},
            daily_idx_maps={s: data.daily_idx_maps[s] for s in syms},
        )
    return shadow.get_filter_summary()


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
