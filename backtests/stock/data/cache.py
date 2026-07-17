"""Parquet-based bar data caching."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtests.stock.data.authority import normalize_bar_frame


def save_bars(df: pd.DataFrame, path: Path) -> None:
    """Write a bar DataFrame to Parquet with timestamp index preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)


def load_bars(path: Path) -> pd.DataFrame:
    """Read a Parquet bar file, returning a DataFrame with DatetimeIndex."""
    return normalize_bar_frame(pd.read_parquet(path, engine="pyarrow"))


def bar_path(
    data_dir: Path,
    symbol: str,
    timeframe: str,
    *,
    session_policy: str | None = None,
) -> Path:
    """Return a legacy path or an explicitly session-qualified compatibility path."""
    if session_policy:
        safe_policy = "".join(character if character.isalnum() or character in "-_" else "_" for character in session_policy)
        return data_dir / f"session={safe_policy}" / f"{symbol}_{timeframe}.parquet"
    return data_dir / f"{symbol}_{timeframe}.parquet"


def load_or_download(
    symbol: str,
    timeframe: str,
    data_dir: Path,
    *,
    session_policy: str | None = None,
) -> pd.DataFrame | None:
    """Load from cache if it exists, otherwise return None.

    Actual download is handled separately via the async downloader.
    """
    path = bar_path(data_dir, symbol, timeframe, session_policy=session_policy)
    if path.exists():
        return load_bars(path)
    return None
