from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_paths
from backtests.shared.auto.replay_bundle import ReplayBundle

_REPLAY_CACHE: dict[str, ReplayBundle[Any]] = {}


def load_brs_replay_bundle(config) -> ReplayBundle[dict[str, Any]]:
    from backtests.swing.engine.brs_portfolio_engine import load_brs_data

    base_dir = Path(config.data_dir)
    source_paths = [
        base_dir / f"{symbol}_1h.parquet"
        for symbol in config.symbols
    ] + [
        base_dir / f"{symbol}_1d.parquet"
        for symbol in config.symbols
    ]
    return _build_bundle(
        "swing.brs.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "symbols": tuple(config.symbols),
            "start_date": config.start_date,
            "end_date": config.end_date,
        },
        loader=lambda: load_brs_data(config),
    )


def load_atrss_replay_bundle(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
) -> ReplayBundle[Any]:
    from backtests.swing.data.cache import load_bars
    from backtests.swing.data.preprocessing import (
        align_daily_to_hourly,
        build_numpy_arrays,
        filter_rth,
        normalize_timezone,
    )
    from backtests.swing.engine.portfolio_engine import PortfolioData

    base_dir = Path(data_dir)
    source_paths = [
        base_dir / f"{symbol}_1h.parquet"
        for symbol in symbols
    ] + [
        base_dir / f"{symbol}_1d.parquet"
        for symbol in symbols
    ]

    def _load() -> Any:
        data = PortfolioData()
        for symbol in symbols:
            hourly_df = normalize_timezone(load_bars(base_dir / f"{symbol}_1h.parquet"))
            hourly_df = filter_rth(hourly_df)
            daily_df = normalize_timezone(load_bars(base_dir / f"{symbol}_1d.parquet"))
            data.hourly[symbol] = build_numpy_arrays(hourly_df)
            data.daily[symbol] = build_numpy_arrays(daily_df)
            data.daily_idx_maps[symbol] = align_daily_to_hourly(hourly_df, daily_df)
        return data

    return _build_bundle(
        "swing.atrss.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={"symbols": symbols},
        loader=_load,
    )


def load_helix_replay_bundle(symbols: list[str], data_dir: Path) -> ReplayBundle[Any]:
    from backtests.swing.engine.helix_portfolio_engine import load_helix_data

    base_dir = Path(data_dir)
    source_paths = [
        base_dir / f"{symbol}_1h.parquet"
        for symbol in symbols
    ] + [
        base_dir / f"{symbol}_1d.parquet"
        for symbol in symbols
    ]
    return _build_bundle(
        "swing.helix.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={"symbols": tuple(symbols)},
        loader=lambda: load_helix_data(symbols, base_dir),
    )


def load_breakout_replay_bundle(symbols: list[str], data_dir: Path) -> ReplayBundle[Any]:
    from backtests.swing.engine.breakout_portfolio_engine import load_breakout_data

    base_dir = Path(data_dir)
    source_paths = [
        base_dir / f"{symbol}_1h.parquet"
        for symbol in symbols
    ] + [
        base_dir / f"{symbol}_1d.parquet"
        for symbol in symbols
    ]
    return _build_bundle(
        "swing.breakout.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={"symbols": tuple(symbols)},
        loader=lambda: load_breakout_data(symbols, base_dir),
    )


def load_unified_portfolio_replay_bundle(config) -> ReplayBundle[Any]:
    """Load the all-swing portfolio replay data behind a source-fingerprinted bundle."""

    from backtests.swing.engine.unified_portfolio_engine import load_unified_data

    base_dir = Path(config.data_dir)
    overlay_symbols = tuple(config.overlay_symbols if config.overlay_enabled else ())
    all_symbols = tuple(
        sorted(
            set(config.atrss_symbols)
            | set(config.helix_symbols)
            | set(config.brs_symbols)
            | set(config.breakout_symbols)
            | set(overlay_symbols)
        )
    )
    source_paths = [
        base_dir / f"{symbol}_{timeframe}.parquet"
        for symbol in all_symbols
        for timeframe in ("1h", "1d")
    ]
    return _build_bundle(
        "swing.portfolio_synergy.unified_replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "atrss_symbols": tuple(config.atrss_symbols),
            "helix_symbols": tuple(config.helix_symbols),
            "brs_symbols": tuple(config.brs_symbols),
            "breakout_symbols": tuple(config.breakout_symbols),
            "overlay_symbols": overlay_symbols,
            "overlay_enabled": bool(config.overlay_enabled),
        },
        loader=lambda: load_unified_data(config),
    )


def _build_bundle(
    namespace: str,
    *,
    source_paths: list[Path],
    root: Path,
    extra: dict[str, Any],
    loader,
) -> ReplayBundle[Any]:
    source_fingerprint = fingerprint_paths(source_paths, root=root)
    cache_key = build_cache_key(
        namespace,
        source_fingerprint=source_fingerprint,
        extra={
            "data_dir": str(root.resolve()),
            **extra,
        },
    )
    cached = _REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bundle = ReplayBundle(
        data=loader(),
        cache_key=cache_key,
        cache_source_fingerprint=source_fingerprint,
    )
    _REPLAY_CACHE[cache_key] = bundle
    return bundle
