from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_paths
from backtests.shared.auto.replay_bundle import ReplayBundle

_REPLAY_CACHE: dict[str, ReplayBundle[Any]] = {}


def tpc_replay_source_artifacts(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
) -> dict[str, Path]:
    """Return the complete ETF-only TPC replay surface."""
    base_dir = Path(data_dir)
    return {
        f"{symbol}_{timeframe}": base_dir / f"{symbol}_{timeframe}.parquet"
        for symbol in symbols
        for timeframe in ("15m", "1h", "1d")
    }


def load_atrss_replay_bundle(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
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

    start_ts = _coerce_utc_timestamp(start_date)
    end_ts = _coerce_utc_timestamp(end_date, end_of_day=True)

    def _load() -> Any:
        data = PortfolioData()
        for symbol in symbols:
            hourly_df = normalize_timezone(load_bars(base_dir / f"{symbol}_1h.parquet"))
            hourly_df = filter_rth(hourly_df)
            daily_df = normalize_timezone(load_bars(base_dir / f"{symbol}_1d.parquet"))
            hourly_df = _slice_timestamp_index(hourly_df, start_ts, end_ts)
            daily_df = _slice_timestamp_index(daily_df, start_ts, end_ts)
            data.hourly[symbol] = build_numpy_arrays(hourly_df)
            data.daily[symbol] = build_numpy_arrays(daily_df)
            data.daily_idx_maps[symbol] = align_daily_to_hourly(hourly_df, daily_df)
        return data

    return _build_bundle(
        "swing.atrss.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "symbols": symbols,
            "start_date": start_ts.isoformat() if start_ts is not None else None,
            "end_date": end_ts.isoformat() if end_ts is not None else None,
        },
        loader=_load,
    )


def _coerce_utc_timestamp(
    value: str | pd.Timestamp | None,
    *,
    end_of_day: bool = False,
) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if end_of_day and ts == ts.normalize():
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def _slice_timestamp_index(
    df: pd.DataFrame,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if start_ts is not None:
        df = df.loc[df.index >= start_ts]
    if end_ts is not None:
        df = df.loc[df.index <= end_ts]
    return df


def load_helix_replay_bundle(
    symbols: list[str],
    data_dir: Path,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> ReplayBundle[Any]:
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
        extra={
            "symbols": tuple(symbols),
            "start_date": _coerce_utc_timestamp(start_date).isoformat() if start_date is not None else None,
            "end_date": _coerce_utc_timestamp(end_date, end_of_day=True).isoformat() if end_date is not None else None,
        },
        loader=lambda: load_helix_data(symbols, base_dir, start_date=start_date, end_date=end_date),
    )


def load_tpc_replay_bundle(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> ReplayBundle[Any]:
    return _load_etf_15m_bundle(
        "swing.tpc.replay_bundle",
        data_dir,
        symbols,
        start_date,
        end_date,
        pullback_timeframe="1h",
    )


def load_tpc_pb30_replay_bundle(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> ReplayBundle[Any]:
    """Load TPC data with the pullback window backed by completed 30m bars.

    The TPC core still receives the compatibility key ``bars_1h`` because the
    pullback detector is shared. The bundle cache key includes
    ``pullback_timeframe`` so indicator arrays from the canonical 1h view
    cannot leak into the 30m research view.
    """

    return _load_etf_15m_bundle(
        "swing.tpc.pb30_replay_bundle",
        data_dir,
        symbols,
        start_date,
        end_date,
        pullback_timeframe="30m",
    )


def _load_etf_15m_bundle(
    namespace: str,
    data_dir: Path,
    symbols: tuple[str, ...],
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
    *,
    pullback_timeframe: str = "1h",
) -> ReplayBundle[Any]:
    from backtests.swing.data.cache import load_bars
    from backtests.swing.data.multitimeframe import (
        align_15m_to_30m,
        align_15m_to_1h,
        align_15m_to_4h,
        align_daily_to_15m,
        resample_15m_to_30m,
        resample_1h_to_4h,
    )
    from backtests.swing.data.preprocessing import build_numpy_arrays, normalize_timezone

    base_dir = Path(data_dir)
    source_paths = [
        base_dir / f"{symbol}_{timeframe}.parquet"
        for symbol in symbols
        for timeframe in ("15m", "1h", "1d")
    ]
    start_ts = _coerce_utc_timestamp(start_date)
    end_ts = _coerce_utc_timestamp(end_date, end_of_day=True)

    def _load() -> dict[str, dict[str, Any]]:
        data: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            df15 = normalize_timezone(load_bars(base_dir / f"{symbol}_15m.parquet"))
            df1h = normalize_timezone(load_bars(base_dir / f"{symbol}_1h.parquet"))
            dfd = normalize_timezone(load_bars(base_dir / f"{symbol}_1d.parquet"))
            df15 = _slice_timestamp_index(df15, start_ts, end_ts)
            df1h = _slice_timestamp_index(df1h, start_ts, end_ts)
            dfd = _slice_timestamp_index(dfd, start_ts, end_ts)
            df30 = resample_15m_to_30m(df15)
            df4h = resample_1h_to_4h(df1h)
            if pullback_timeframe == "1h":
                pullback_df = df1h
                idx_pullback = align_15m_to_1h(df15, df1h)
            elif pullback_timeframe == "30m":
                pullback_df = df30
                idx_pullback = align_15m_to_30m(df15, df30)
            else:
                raise ValueError(f"Unsupported TPC pullback_timeframe={pullback_timeframe!r}")
            data[symbol] = {
                "bars_15m": build_numpy_arrays(df15),
                "bars_30m": build_numpy_arrays(df30),
                "bars_1h": build_numpy_arrays(pullback_df),
                "bars_4h": build_numpy_arrays(df4h),
                "bars_daily": build_numpy_arrays(dfd),
                "idx_30m": align_15m_to_30m(df15, df30),
                "idx_1h": idx_pullback,
                "idx_4h": align_15m_to_4h(df15, df4h),
                "idx_daily": align_daily_to_15m(df15, dfd),
            }
        return data

    # Phase 4: Keep the shared ETF namespace stable for TPC replay data.
    return _build_bundle(
        "swing.etf_15m_data",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "symbols": tuple(symbols),
            "start_date": start_ts.isoformat() if start_ts is not None else None,
            "end_date": end_ts.isoformat() if end_ts is not None else None,
            "pullback_timeframe": pullback_timeframe,
            "market_data_scope": "traded_etfs_only",
        },
        loader=_load,
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
            | set(getattr(config, "tpc_symbols", ()))
            | set(overlay_symbols)
        )
    )
    source_paths = [
        base_dir / f"{symbol}_{timeframe}.parquet"
        for symbol in all_symbols
        for timeframe in ("1h", "1d")
    ] + [
        base_dir / f"{symbol}_15m.parquet"
        for symbol in sorted(
            set(getattr(config, "tpc_symbols", ()))
        )
    ]
    return _build_bundle(
        "swing.unified.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "atrss_symbols": tuple(config.atrss_symbols),
            "helix_symbols": tuple(config.helix_symbols),
            "overlay_symbols": overlay_symbols,
            "overlay_enabled": bool(config.overlay_enabled),
            "tpc_market_data_scope": "traded_etfs_only",
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
