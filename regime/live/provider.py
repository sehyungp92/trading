"""LiveDataProvider: assembles macro_df, market_df, strat_ret_df from IBKR + FRED + cached parquets."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series and ETF symbols for live data overlay
_FRED_SERIES = {
    "VIX": "VIXCLS",
    "SPREAD": "BAMLH0A0HYM2",
    "SLOPE_10Y2Y": "T10Y2Y",
    "INFLATION": "T10YIE",
    "REAL_RATE_10Y": "DFII10",
}

_ETF_SYMBOLS = ["SPY", "EFA", "TLT", "GLD", "IBIT", "BIL", "DBC"]

_MARKET_FRED_COLS = ["VIX", "SPREAD", "SLOPE_10Y2Y", "REAL_RATE_10Y"]
_MACRO_GROWTH_SERIES = "ICSA"


class LiveDataProvider:
    """Assembles macro_df, market_df, strat_ret_df from IBKR + FRED + cached parquets."""

    def __init__(self, ib_session: Any, data_dir: Path) -> None:
        self._session = ib_session
        self._data_dir = data_dir
        self._contracts: dict[str, Any] = {}

    async def qualify_contracts(self) -> None:
        """Qualify ETF contracts once at startup."""
        from ib_async import Stock

        for sym in _ETF_SYMBOLS:
            try:
                contract = Stock(sym, "SMART", "USD")
                qualified = await self._session.ib.qualifyContractsAsync(contract)
                if qualified:
                    self._contracts[sym] = qualified[0]
            except Exception as e:
                logger.warning("Regime: could not qualify %s: %s", sym, e)

        logger.info("Regime: qualified %d/%d ETF contracts", len(self._contracts), len(_ETF_SYMBOLS))

    async def build_live_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached parquets, overlay fresh IBKR + FRED data, return 3 DataFrames."""
        # 1. Load cached baseline
        macro_df, market_df, strat_ret_df = self._load_cached()

        # 2. Fetch IBKR daily bars and overlay
        ibkr_prices = await self._fetch_ibkr_bars()
        if ibkr_prices is not None and not ibkr_prices.empty:
            strat_ret_df, market_df = self._overlay_ibkr(
                ibkr_prices, strat_ret_df, market_df,
            )

        # 3. Fetch FRED macro data and overlay
        loop = asyncio.get_running_loop()
        try:
            fred_data = await loop.run_in_executor(None, self._fetch_fred)
            if fred_data is not None:
                fred_df, icsa_raw = fred_data
                macro_df, market_df = self._overlay_fred(
                    fred_df, icsa_raw, macro_df, market_df,
                )
        except Exception:
            logger.warning("Regime: FRED fetch failed, using cached macro data", exc_info=True)

        # 4. Align all three DataFrames to union of dates (IBKR/FRED extend beyond cache)
        all_dates = macro_df.index.union(market_df.index).union(strat_ret_df.index)
        macro_df = macro_df.reindex(all_dates).ffill()
        market_df = market_df.reindex(all_dates).ffill()
        strat_ret_df = strat_ret_df.reindex(all_dates).fillna(0.0)

        # Drop leading rows where macro features are NaN
        first_valid = macro_df.dropna(how="any").index.min()
        if first_valid is not None:
            macro_df = macro_df.loc[first_valid:]
            market_df = market_df.loc[first_valid:]
            strat_ret_df = strat_ret_df.loc[first_valid:]

        # 5. Save updated parquets back to data_dir
        try:
            macro_df.to_parquet(self._data_dir / "macro_df.parquet")
            market_df.to_parquet(self._data_dir / "market_df.parquet")
            strat_ret_df.to_parquet(self._data_dir / "strat_ret_df.parquet")
            logger.info("Regime: updated cached parquets in %s", self._data_dir)
        except Exception:
            logger.warning("Regime: failed to update parquet cache", exc_info=True)

        logger.info(
            "Regime: data assembled -- %d rows, range %s to %s",
            len(strat_ret_df),
            strat_ret_df.index.min().strftime("%Y-%m-%d") if len(strat_ret_df) else "N/A",
            strat_ret_df.index.max().strftime("%Y-%m-%d") if len(strat_ret_df) else "N/A",
        )
        return macro_df.copy(), market_df.copy(), strat_ret_df.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cached(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached parquets from data_dir. Raises FileNotFoundError if missing."""
        names = ("macro_df.parquet", "market_df.parquet", "strat_ret_df.parquet")
        for name in names:
            if not (self._data_dir / name).exists():
                raise FileNotFoundError(
                    f"Regime cached parquet '{name}' not found in {self._data_dir}. "
                    "Seed data via Dockerfile COPY or run "
                    "`FRED_API_KEY=<key> python -m backtests.regime.cli download --data-dir data/regime/raw`."
                )
        return (
            pd.read_parquet(self._data_dir / "macro_df.parquet").copy(),
            pd.read_parquet(self._data_dir / "market_df.parquet").copy(),
            pd.read_parquet(self._data_dir / "strat_ret_df.parquet").copy(),
        )

    async def _fetch_ibkr_bars(self) -> pd.DataFrame | None:
        """Fetch 1Y daily bars for all qualified ETF contracts."""
        if not self._contracts:
            logger.warning("Regime: no IBKR contracts qualified, skipping IBKR fetch")
            return None

        all_prices: dict[str, pd.Series] = {}
        for sym, contract in self._contracts.items():
            try:
                bars = await self._session.ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr="1 Y",
                    barSizeSetting="1 day", whatToShow="TRADES",
                    useRTH=True, formatDate=1,
                )
                if bars:
                    dates = pd.to_datetime([b.date for b in bars])
                    closes = pd.Series([b.close for b in bars], index=dates, name=sym, dtype=float)
                    all_prices[sym] = closes
                    logger.debug("Regime: fetched %d bars for %s", len(bars), sym)
                else:
                    logger.warning("Regime: empty bars for %s", sym)
            except Exception:
                logger.warning("Regime: failed to fetch IBKR bars for %s", sym, exc_info=True)

        if not all_prices:
            return None

        prices_df = pd.DataFrame(all_prices)
        prices_df.index.name = "date"
        # Remove timezone info if present
        if prices_df.index.tz is not None:
            prices_df.index = prices_df.index.tz_localize(None)
        return prices_df

    def _overlay_ibkr(
        self,
        ibkr_prices: pd.DataFrame,
        strat_ret_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Overlay IBKR-derived returns onto cached DataFrames."""
        ibkr_log_ret = np.log(ibkr_prices / ibkr_prices.shift(1))

        # Overlay strat_ret_df columns
        col_map = {"SPY": "SPY", "EFA": "EFA", "TLT": "TLT", "GLD": "GLD", "IBIT": "IBIT", "BIL": "CASH"}
        for ibkr_col, ret_col in col_map.items():
            if ibkr_col in ibkr_log_ret.columns and ret_col in strat_ret_df.columns:
                fresh = ibkr_log_ret[ibkr_col].dropna()
                overlap = fresh.index.intersection(strat_ret_df.index)
                if len(overlap) > 0:
                    strat_ret_df.loc[overlap, ret_col] = fresh.loc[overlap]
                # Extend with new dates beyond cached range
                new_dates = fresh.index.difference(strat_ret_df.index)
                if len(new_dates) > 0:
                    extension = pd.DataFrame(0.0, index=new_dates, columns=strat_ret_df.columns)
                    extension[ret_col] = fresh.loc[new_dates]
                    strat_ret_df = pd.concat([strat_ret_df, extension]).sort_index()

        # Overlay DBC log returns into market_df (overlay + extend)
        if "DBC" in ibkr_log_ret.columns and "DBC" in market_df.columns:
            fresh_dbc = ibkr_log_ret["DBC"].dropna()
            overlap = fresh_dbc.index.intersection(market_df.index)
            if len(overlap) > 0:
                market_df.loc[overlap, "DBC"] = fresh_dbc.loc[overlap]
            new_dbc_dates = fresh_dbc.index.difference(market_df.index)
            if len(new_dbc_dates) > 0:
                extension = pd.DataFrame(index=new_dbc_dates, columns=market_df.columns, dtype=float)
                extension["DBC"] = fresh_dbc.loc[new_dbc_dates]
                market_df = pd.concat([market_df, extension]).sort_index()

        logger.info(
            "Regime: IBKR overlay applied -- %d symbols, strat_ret range to %s",
            len(ibkr_log_ret.columns),
            strat_ret_df.index.max().strftime("%Y-%m-%d") if len(strat_ret_df) else "N/A",
        )
        return strat_ret_df, market_df

    def _fetch_fred(self) -> tuple[pd.DataFrame, pd.Series] | None:
        """Fetch recent FRED data (blocking -- run in executor)."""
        import os

        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            logger.warning("Regime: FRED_API_KEY not set, skipping FRED fetch")
            return None

        from fredapi import Fred

        fred = Fred(api_key=key)

        # Only need recent data to overlay on cached baseline
        from datetime import datetime, timedelta
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        # Fetch 5 market series
        frames: dict[str, pd.Series] = {}
        for col, series_id in _FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start)
                s.name = col
                frames[col] = s
            except Exception as exc:
                logger.warning("Regime: FRED fetch failed for %s (%s): %s", col, series_id, exc)

        fred_df = pd.DataFrame(frames)
        fred_df.index = pd.to_datetime(fred_df.index)
        fred_df.index.name = "date"

        # Fetch ICSA (growth proxy) -- standard API, not ALFRED vintage
        icsa_raw = pd.Series(dtype=float)
        try:
            icsa = fred.get_series(_MACRO_GROWTH_SERIES, observation_start=start)
            icsa.index = pd.to_datetime(icsa.index)
            icsa_raw = icsa
        except Exception as exc:
            logger.warning("Regime: FRED fetch failed for ICSA: %s", exc)

        return fred_df, icsa_raw

    def _overlay_fred(
        self,
        fred_df: pd.DataFrame,
        icsa_raw: pd.Series,
        macro_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Overlay fresh FRED data onto cached DataFrames (overlay + extend)."""
        # Overlay + extend market_df columns (VIX, SPREAD, SLOPE_10Y2Y, REAL_RATE_10Y)
        for col in _MARKET_FRED_COLS:
            if col in fred_df.columns and col in market_df.columns:
                fresh = fred_df[col].dropna()
                overlap = fresh.index.intersection(market_df.index)
                if len(overlap) > 0:
                    market_df.loc[overlap, col] = fresh.loc[overlap]
                new_dates = fresh.index.difference(market_df.index)
                if len(new_dates) > 0:
                    extension = pd.DataFrame(index=new_dates, columns=market_df.columns, dtype=float)
                    extension[col] = fresh.loc[new_dates]
                    market_df = pd.concat([market_df, extension]).sort_index()

        # Overlay + extend INFLATION in macro_df (before GROWTH so macro_df index is extended)
        if "INFLATION" in fred_df.columns and "INFLATION" in macro_df.columns:
            fresh_infl = fred_df["INFLATION"].dropna()
            overlap = fresh_infl.index.intersection(macro_df.index)
            if len(overlap) > 0:
                macro_df.loc[overlap, "INFLATION"] = fresh_infl.loc[overlap]
            new_dates = fresh_infl.index.difference(macro_df.index)
            if len(new_dates) > 0:
                extension = pd.DataFrame(index=new_dates, columns=macro_df.columns, dtype=float)
                extension["INFLATION"] = fresh_infl.loc[new_dates]
                macro_df = pd.concat([macro_df, extension]).sort_index()

        # Overlay GROWTH in macro_df (negated ICSA, daily-reindexed with ffill onto extended index)
        if len(icsa_raw) > 0 and "GROWTH" in macro_df.columns:
            growth_fresh = -icsa_raw
            growth_daily = growth_fresh.reindex(macro_df.index, method="ffill").copy()
            valid = growth_daily.dropna()
            if len(valid) > 0:
                macro_df.loc[valid.index, "GROWTH"] = valid

        # Forward-fill gaps
        macro_df = macro_df.ffill()
        market_df = market_df.ffill()

        logger.info("Regime: FRED overlay applied")
        return macro_df, market_df
