"""Data pipeline: FRED + yfinance download, IBIT proxy, cache as parquet."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs (market-priced or non-revised — safe for standard API)
_FRED_SERIES = {
    "VIX": "VIXCLS",
    "SPREAD": "BAMLH0A0HYM2",
    "SLOPE_10Y2Y": "T10Y2Y",
    "INFLATION": "T10YIE",
    "REAL_RATE_10Y": "DFII10",  # 10-Year Real Interest Rate (TIPS yield)
}

# ICSA requires ALFRED vintage discipline (point-in-time first-release values)
_ICSA_SERIES_ID = "ICSA"

# ETF tickers for yfinance
_ETF_TICKERS = ["SPY", "EFA", "TLT", "GLD", "BIL"]
_FEATURE_ETF_TICKERS = ["DBC"]  # Invesco DB Commodity Index (inception 2006-02)

# IBIT proxy dates
_BTC_START = pd.Timestamp("2014-09-17")
_IBIT_START = pd.Timestamp("2024-01-11")
_BTC_FEE_DRAG_ANNUAL = 0.0025


def _require_fred_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Set FRED_API_KEY env var (free: https://fred.stlouisfed.org/docs/api/api_key.html)"
        )
    return key


def _download_fred(start: str = "2002-01-01") -> pd.DataFrame:
    """Download all FRED series and return a combined daily DataFrame."""
    from fredapi import Fred

    key = _require_fred_key()
    fred = Fred(api_key=key)
    frames = {}
    for col, series_id in _FRED_SERIES.items():
        logger.info("Downloading FRED %s (%s)", col, series_id)
        s = fred.get_series(series_id, observation_start=start)
        s.name = col
        frames[col] = s

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    return combined


def _download_icsa_vintage(start: str = "2002-01-01") -> pd.Series:
    """Download ICSA using ALFRED vintage API for point-in-time first-release values.

    Standard FRED get_series() returns latest-revised data, which introduces
    look-ahead bias. ALFRED provides every vintage, so we take the earliest
    release for each observation date — exactly what was known at the time.
    """
    from fredapi import Fred

    key = _require_fred_key()
    fred = Fred(api_key=key)

    logger.info("Downloading ICSA via ALFRED (point-in-time vintages)")
    releases = fred.get_series_all_releases(_ICSA_SERIES_ID)
    # releases is a DataFrame with columns: realtime_start, date, value
    # or a MultiIndex Series — normalize to DataFrame
    if isinstance(releases, pd.Series):
        releases = releases.reset_index()
        releases.columns = ["realtime_start", "date", "value"]
    elif "realtime_start" not in releases.columns:
        releases = releases.reset_index()

    releases["date"] = pd.to_datetime(releases["date"])
    releases["realtime_start"] = pd.to_datetime(releases["realtime_start"])
    releases["value"] = pd.to_numeric(releases["value"], errors="coerce")

    # Filter to start date
    releases = releases[releases["date"] >= start]

    # For each observation date, take the first release (earliest realtime_start)
    releases = releases.sort_values(["date", "realtime_start"])
    first_release = releases.groupby("date").first()["value"]
    first_release.index.name = "date"
    first_release.name = "GROWTH_RAW"

    logger.info(
        "ICSA vintage: %d observation dates, range %s to %s",
        len(first_release),
        first_release.index.min().strftime("%Y-%m-%d"),
        first_release.index.max().strftime("%Y-%m-%d"),
    )
    return first_release


def _download_etf_prices(start: str = "2002-01-01") -> pd.DataFrame:
    """Download adjusted close prices for ETFs via yfinance."""
    import yfinance as yf

    logger.info("Downloading ETF prices: %s", _ETF_TICKERS)
    data = yf.download(_ETF_TICKERS, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    # Remove timezone info if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices


def _download_btc_prices(start: str = "2014-09-01") -> pd.Series:
    """Download BTC-USD daily prices via yfinance."""
    import yfinance as yf

    logger.info("Downloading BTC-USD prices")
    data = yf.download("BTC-USD", start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].squeeze()
    else:
        prices = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.name = "BTC"
    return prices


def _download_ibit_prices() -> pd.Series:
    """Download IBIT ETF prices (post Jan 2024)."""
    import yfinance as yf

    logger.info("Downloading IBIT prices")
    data = yf.download("IBIT", start="2024-01-01", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].squeeze()
    else:
        prices = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.name = "IBIT"
    return prices


def _download_feature_etfs(start: str = "2002-01-01") -> pd.DataFrame:
    """Download feature-only ETF prices (DBC etc.) via yfinance."""
    import yfinance as yf

    if not _FEATURE_ETF_TICKERS:
        return pd.DataFrame()

    logger.info("Downloading feature ETF prices: %s", _FEATURE_ETF_TICKERS)
    data = yf.download(_FEATURE_ETF_TICKERS, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=_FEATURE_ETF_TICKERS[0])
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices


def _build_ibit_returns(
    etf_returns: pd.DataFrame,
    btc_prices: pd.Series,
    ibit_prices: pd.Series,
) -> pd.Series:
    """Build IBIT return series with proxy logic.

    Pre 2014-09-17:  IBIT returns = CASH (BIL) returns
    2014-09 to 2024-01:  BTC-USD log returns - 0.25% annual fee drag
    Post 2024-01:  Actual IBIT ETF log returns
    """
    full_idx = etf_returns.index
    ibit_ret = pd.Series(np.nan, index=full_idx, name="IBIT")

    # Phase 1: pre-BTC → use CASH returns
    pre_btc = full_idx < _BTC_START
    if "CASH" in etf_returns.columns:
        ibit_ret.loc[pre_btc] = etf_returns.loc[pre_btc, "CASH"]
    else:
        ibit_ret.loc[pre_btc] = 0.0

    # Phase 2: BTC proxy (2014-09 to IBIT launch)
    btc_log_ret = np.log(btc_prices / btc_prices.shift(1)).dropna()
    fee_daily = _BTC_FEE_DRAG_ANNUAL / 252.0
    btc_adj_ret = btc_log_ret - fee_daily

    # Find IBIT splice date (first available IBIT trading day)
    ibit_splice = _IBIT_START
    if len(ibit_prices) > 1:
        ibit_splice = ibit_prices.index[1]  # first day with a return

    btc_mask = (full_idx >= _BTC_START) & (full_idx < ibit_splice)
    btc_adj_ret_aligned = btc_adj_ret.reindex(full_idx)
    ibit_ret.loc[btc_mask] = btc_adj_ret_aligned.loc[btc_mask]

    # Phase 3: actual IBIT returns
    ibit_log_ret = np.log(ibit_prices / ibit_prices.shift(1)).dropna()
    ibit_log_ret_aligned = ibit_log_ret.reindex(full_idx)
    post_ibit = full_idx >= ibit_splice
    ibit_ret.loc[post_ibit] = ibit_log_ret_aligned.loc[post_ibit]

    # Forward-fill any remaining gaps
    ibit_ret = ibit_ret.ffill().fillna(0.0)
    return ibit_ret


def build_all_data(
    data_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download all data, build the three DataFrames, cache as parquet.

    Returns:
        macro_df: columns [GROWTH, INFLATION]
        market_df: columns [VIX, SPREAD, SLOPE_10Y2Y, REAL_RATE_10Y, DBC]
        strat_ret_df: columns [SPY, EFA, TLT, GLD, IBIT, CASH]
    """
    if data_dir is None:
        data_dir = Path("backtests/regime/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download raw data
    fred_df = _download_fred()
    icsa_vintage = _download_icsa_vintage()
    etf_prices = _download_etf_prices()
    btc_prices = _download_btc_prices()
    ibit_prices = _download_ibit_prices()
    feature_etf_prices = _download_feature_etfs()

    # -- Build market_df --
    market_df = fred_df[["VIX", "SPREAD", "SLOPE_10Y2Y"]].copy()
    # Add REAL_RATE_10Y from FRED (DFII10)
    if "REAL_RATE_10Y" in fred_df.columns:
        market_df["REAL_RATE_10Y"] = fred_df["REAL_RATE_10Y"]
    # Add DBC log returns as a feature column (not an allocation target)
    # DBC inception is 2006-02 — pre-inception dates get 0.0 (no return data)
    if "DBC" in feature_etf_prices.columns:
        dbc_log_ret = np.log(feature_etf_prices["DBC"] / feature_etf_prices["DBC"].shift(1))
        market_df["DBC"] = dbc_log_ret.fillna(0.0)
    market_df = market_df.ffill()

    # -- Build macro_df --
    # Growth: negate ICSA (lower claims = stronger growth)
    # Uses ALFRED first-release values — point-in-time, no look-ahead bias
    # ICSA dates are Saturdays (week-ending) — reindex to daily and forward-fill
    growth_raw = -icsa_vintage
    growth_raw.name = "GROWTH"

    inflation = fred_df["INFLATION"].ffill()
    inflation.name = "INFLATION"

    # Build on INFLATION's daily index, then merge GROWTH with ffill
    macro_df = pd.DataFrame({"INFLATION": inflation})
    macro_df["GROWTH"] = growth_raw.reindex(macro_df.index, method="ffill")
    macro_df = macro_df[["GROWTH", "INFLATION"]]

    # -- Build strat_ret_df --
    # Compute daily log returns from ETF prices
    log_returns = np.log(etf_prices / etf_prices.shift(1))

    strat_ret_df = pd.DataFrame(index=log_returns.index)
    for col in ["SPY", "EFA", "TLT", "GLD"]:
        if col in log_returns.columns:
            strat_ret_df[col] = log_returns[col]

    # CASH = BIL returns
    if "BIL" in log_returns.columns:
        strat_ret_df["CASH"] = log_returns["BIL"]
    else:
        strat_ret_df["CASH"] = 0.0

    # IBIT proxy
    ibit_ret = _build_ibit_returns(strat_ret_df, btc_prices, ibit_prices)
    strat_ret_df["IBIT"] = ibit_ret

    # Drop rows before both macro features are available
    # T10YIE starts Jan 2003; ICSA first-release may have a few leading NaN
    common_start = macro_df.dropna(how="any").index.min()
    if common_start is not None:
        macro_df = macro_df.loc[common_start:]
        market_df = market_df.loc[common_start:]
        strat_ret_df = strat_ret_df.loc[common_start:]

    # Drop any all-NaN leading rows
    strat_ret_df = strat_ret_df.dropna(how="all")
    macro_df = macro_df.reindex(strat_ret_df.index).ffill()
    market_df = market_df.reindex(strat_ret_df.index).ffill()

    # Cache
    macro_df.to_parquet(data_dir / "macro_df.parquet")
    market_df.to_parquet(data_dir / "market_df.parquet")
    strat_ret_df.to_parquet(data_dir / "strat_ret_df.parquet")

    logger.info("Saved 3 parquet files to %s", data_dir)
    return macro_df, market_df, strat_ret_df


def load_cached_data(
    data_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached parquet files. Raises FileNotFoundError if not cached."""
    if data_dir is None:
        data_dir = Path("backtests/regime/data/raw")

    macro_df = pd.read_parquet(data_dir / "macro_df.parquet")
    market_df = pd.read_parquet(data_dir / "market_df.parquet")
    strat_ret_df = pd.read_parquet(data_dir / "strat_ret_df.parquet")
    return macro_df, market_df, strat_ret_df
