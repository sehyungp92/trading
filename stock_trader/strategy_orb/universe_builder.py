"""Live universe builder — resolves symbols from IB on demand."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time as _time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import fmean

from ib_async import IB, ScannerSubscription, Stock

from .models import CachedSymbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known ETFs to exclude from ORB universe
# ---------------------------------------------------------------------------

KNOWN_ETFS: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV",
    "XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI",
    "XLB", "XLU", "XLRE", "XLC", "GLD", "SLV", "HYG",
    "TLT", "IEF", "LQD", "EFA", "EEM", "VEA", "VWO",
    "ARKK", "SOXL", "TQQQ", "SQQQ", "XBI", "SMH",
})

# ---------------------------------------------------------------------------
# Intraday volume fractions (U-shaped curve, 09:35 – 11:15)
# ---------------------------------------------------------------------------

def _build_intraday_fractions() -> dict[str, float]:
    """Declining U-shaped volume curve from 09:35 to 11:15.

    f(t) = 0.006 + 0.019 * exp(-t/20)  where t = minutes since 09:35.
    """
    start = 9 * 60 + 35
    end = 11 * 60 + 15
    return {
        f"{(start + t) // 60:02d}:{(start + t) % 60:02d}": round(
            0.006 + 0.019 * math.exp(-t / 20.0), 6
        )
        for t in range(end - start + 1)
    }


_INTRADAY_FRACTIONS: dict[str, float] = _build_intraday_fractions()


# ---------------------------------------------------------------------------
# Rate limiter (adapted from IARIC)
# ---------------------------------------------------------------------------

@dataclass
class _RateBudget:
    rate_per_second: float = 2.0
    burst: float = 4.0
    _tokens: float = 4.0
    _updated_at: float = 0.0

    def __post_init__(self) -> None:
        self._updated_at = _time.monotonic()

    def _refill(self) -> None:
        now = _time.monotonic()
        elapsed = now - self._updated_at
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_second)
        self._updated_at = now

    async def wait_for(self, cost: float = 1.0) -> None:
        while True:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return
            await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Sector mapping
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    "Computers": "Technology",
    "Semiconductors": "Technology",
    "Software": "Technology",
    "Internet": "Technology",
    "Electronics": "Technology",
    "Telecommunications": "Technology",
    "Pharmaceuticals": "Healthcare",
    "Biotechnology": "Healthcare",
    "Medical": "Healthcare",
    "Health": "Healthcare",
    "Banks": "Financials",
    "Insurance": "Financials",
    "Financial": "Financials",
    "Investment": "Financials",
    "Oil": "Energy",
    "Gas": "Energy",
    "Coal": "Energy",
    "Energy": "Energy",
    "Retail": "Consumer Discretionary",
    "Consumer": "Consumer Discretionary",
    "Automotive": "Consumer Discretionary",
    "Food": "Consumer Staples",
    "Beverage": "Consumer Staples",
    "Tobacco": "Consumer Staples",
    "Aerospace": "Industrials",
    "Defense": "Industrials",
    "Industrial": "Industrials",
    "Transportation": "Industrials",
    "Mining": "Materials",
    "Chemicals": "Materials",
    "Steel": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}


def _map_sector(category: str, subcategory: str) -> str:
    for token, sector in _SECTOR_MAP.items():
        if token in category or token in subcategory:
            return sector
    return "Unknown"


# ---------------------------------------------------------------------------
# Disk caching helpers (adapted from IARIC)
# ---------------------------------------------------------------------------

def _bars_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / "daily_bars" / f"{symbol}.json"


def _contract_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / "contract_details" / f"{symbol}.json"


def _load_cached_bars(cache_dir: Path, symbol: str) -> tuple[list[dict], str | None]:
    path = _bars_cache_path(cache_dir, symbol)
    if not path.exists():
        return [], None
    with open(path) as f:
        data = json.load(f)
    return data.get("bars", []), data.get("last_updated")


def _save_cached_bars(cache_dir: Path, symbol: str, bars: list[dict], last_updated: str) -> None:
    path = _bars_cache_path(cache_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"symbol": symbol, "last_updated": last_updated, "bars": bars}, f)
    tmp.replace(path)


def _load_cached_contract(cache_dir: Path, symbol: str) -> dict | None:
    path = _contract_cache_path(cache_dir, symbol)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_cached_contract(cache_dir: Path, symbol: str, data: dict) -> None:
    path = _contract_cache_path(cache_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# SMA slope (numpy-free linear regression)
# ---------------------------------------------------------------------------

def _sma_slope(closes: list[float]) -> float:
    """Slope of linear regression over the given closes."""
    n = len(closes)
    if n < 2:
        return 0.0
    x_bar = (n - 1) / 2.0
    y_bar = fmean(closes)
    numerator = 0.0
    denominator = 0.0
    for i, y in enumerate(closes):
        numerator += (i - x_bar) * (y - y_bar)
        denominator += (i - x_bar) ** 2
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# ATR computation
# ---------------------------------------------------------------------------

def _compute_atr(bars: list, period: int = 14) -> float:
    """ATR from bar objects (ib_async BarData with .high/.low/.close attrs)."""
    sample = bars[-period:]
    if len(sample) < 2:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(sample)):
        h = float(sample[i].high)
        l = float(sample[i].low)
        pc = float(sample[i - 1].close)
        true_ranges.append(max(h - l, abs(h - pc), abs(l - pc)))
    return fmean(true_ranges) if true_ranges else 0.0


# ---------------------------------------------------------------------------
# LiveUniverseBuilder
# ---------------------------------------------------------------------------

class LiveUniverseBuilder:
    """Resolves symbols from IB on demand, populating a shared cache."""

    def __init__(
        self,
        ib: IB,
        cache: dict[str, CachedSymbol],
        cache_dir: Path,
    ) -> None:
        self._ib = ib
        self._cache = cache
        self._cache_dir = cache_dir
        self._rate = _RateBudget(rate_per_second=2.0, burst=4.0)
        self._sem = asyncio.Semaphore(6)
        self._resolving: set[str] = set()
        self._failed: set[str] = set()

    # -- Public API ---------------------------------------------------------

    def resolve_batch(self, symbols: list[str]) -> None:
        """Launch background resolution tasks for new symbols (non-blocking)."""
        for sym in symbols:
            sym = sym.upper()
            if sym in self._cache or sym in self._resolving or sym in self._failed or sym in KNOWN_ETFS:
                continue
            self._resolving.add(sym)
            asyncio.create_task(self._resolve_one(sym))

    async def resolve_batch_blocking(self, symbols: list[str]) -> None:
        """Resolve symbols and wait for all to complete."""
        tasks: list[asyncio.Task] = []
        for sym in symbols:
            sym = sym.upper()
            if sym in self._cache or sym in self._resolving or sym in self._failed or sym in KNOWN_ETFS:
                continue
            self._resolving.add(sym)
            tasks.append(asyncio.create_task(self._resolve_one(sym)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def pre_market_warmup(self) -> None:
        """Run one-shot scanners and resolve results before market open."""
        logger.info("Pre-market warmup: running scanners...")
        symbols: set[str] = set()

        for scan_code in ("TOP_PERC_GAIN", "HOT_BY_VOLUME"):
            try:
                await self._rate.wait_for()
                sub = ScannerSubscription(
                    numberOfRows=40,
                    instrument="STK",
                    locationCode="STK.US.MAJOR",
                    scanCode=scan_code,
                    abovePrice=5.0,
                    aboveVolume=250_000,
                )
                results = await asyncio.wait_for(
                    self._ib.reqScannerDataAsync(sub), timeout=15.0,
                )
                for r in (results or []):
                    if r.contractDetails and r.contractDetails.contract:
                        sym = r.contractDetails.contract.symbol.upper()
                        if sym not in KNOWN_ETFS:
                            symbols.add(sym)
            except Exception:
                logger.exception("Warmup scanner %s failed", scan_code)

        logger.info("Warmup: resolving %d symbols from scanners", len(symbols))
        await self.resolve_batch_blocking(list(symbols))
        logger.info("Warmup complete: %d symbols in cache", len(self._cache))

    # -- Internal -----------------------------------------------------------

    async def _resolve_one(self, symbol: str) -> None:
        """Full resolution pipeline for a single symbol."""
        try:
            today_str = date.today().isoformat()

            # 1. Contract details
            cd = await self._fetch_contract_details(symbol)
            if cd is None:
                self._failed.add(symbol)
                self._resolving.discard(symbol)
                return

            category = cd.get("category", "")
            subcategory = cd.get("subcategory", "")
            sector = _map_sector(category, subcategory)
            tech_tag = sector == "Technology"

            # 2. Daily bars
            daily_bars = await self._fetch_daily_bars(symbol, cd, today_str)
            if not daily_bars or len(daily_bars) < 5:
                logger.warning("Insufficient daily bars for %s (%d)", symbol, len(daily_bars))
                self._failed.add(symbol)
                self._resolving.discard(symbol)
                return

            # Exclude today's partial bar so prior_close / SMA use completed days only
            history = [b for b in daily_bars if b["trade_date"] != today_str]
            if len(history) < 5:
                history = daily_bars

            closes = [b["close"] for b in history]
            volumes = [b["volume"] for b in history]

            prior_close = closes[-1]
            sma20 = fmean(closes[-20:]) if len(closes) >= 20 else fmean(closes)
            sma60 = fmean(closes[-60:]) if len(closes) >= 60 else fmean(closes)
            sma20_slope = _sma_slope(closes[-20:]) if len(closes) >= 20 else _sma_slope(closes)
            adv20_values = [
                closes[i] * volumes[i]
                for i in range(max(0, len(closes) - 20), len(closes))
            ]
            adv20 = fmean(adv20_values) if adv20_values else 0.0

            # 3. 1-min bars for ATR
            atr1m14 = await self._fetch_1min_atr(symbol, cd)

            # 4. Baseline estimation
            warmup_keys = [k for k in _INTRADAY_FRACTIONS if "09:35" <= k <= "09:49"]
            opening_value15_baseline = adv20 * sum(_INTRADAY_FRACTIONS[k] for k in warmup_keys)

            share_volume_daily = adv20 / prior_close if prior_close > 0 else 0.0
            minute_volume_baseline: dict[str, float] = {
                key: share_volume_daily * frac for key, frac in _INTRADAY_FRACTIONS.items()
            }

            # 5. Build CachedSymbol
            cached = CachedSymbol(
                symbol=symbol,
                exchange=cd.get("exchange", "SMART"),
                primary_exchange=cd.get("primary_exchange", ""),
                currency="USD",
                tick_size=cd.get("min_tick", 0.01),
                point_value=1.0,
                adv20=adv20,
                prior_close=prior_close,
                sma20=sma20,
                sma60=sma60,
                sma20_slope=sma20_slope,
                atr1m14=atr1m14,
                opening_value15_baseline_0935_0950=opening_value15_baseline,
                minute_volume_baseline_0935_1115=minute_volume_baseline,
                sector=sector,
                float_shares=None,
                float_bucket="",
                catalyst_tag="",
                luld_tier="tier_1",
                tech_tag=tech_tag,
                secondary_universe=adv20 < 20_000_000,
            )
            self._cache[symbol] = cached
            logger.info("Resolved %s (sector=%s, adv20=%.0f, atr1m=%.4f)", symbol, sector, adv20, atr1m14)

        except Exception:
            logger.exception("Failed to resolve %s", symbol)
            self._failed.add(symbol)
        finally:
            self._resolving.discard(symbol)

    @staticmethod
    def _qualified_contract(symbol: str, cd: dict) -> Stock:
        """Build a Stock contract pre-filled with conId & primaryExchange."""
        contract = Stock(symbol, "SMART", "USD")
        pex = cd.get("primary_exchange", "")
        if pex:
            contract.primaryExchange = pex
        con_id = cd.get("con_id")
        if con_id:
            contract.conId = con_id
        return contract

    async def _fetch_contract_details(self, symbol: str) -> dict | None:
        """Fetch and cache IB contract details."""
        cached = _load_cached_contract(self._cache_dir, symbol)
        if cached:
            return cached

        async with self._sem:
            await self._rate.wait_for()
            contract = Stock(symbol, "SMART", "USD")
            try:
                details_list = await asyncio.wait_for(
                    self._ib.reqContractDetailsAsync(contract), timeout=15.0,
                )
            except Exception:
                logger.exception("reqContractDetails failed for %s", symbol)
                return None

            if not details_list:
                logger.warning("No contract details for %s", symbol)
                return None

            cd = details_list[0]
            result = {
                "con_id": cd.contract.conId,
                "symbol": cd.contract.symbol,
                "exchange": cd.contract.exchange or "SMART",
                "primary_exchange": cd.contract.primaryExchange or "",
                "min_tick": cd.minTick or 0.01,
                "category": cd.category or "",
                "subcategory": cd.subcategory or "",
                "long_name": cd.longName or "",
            }
            _save_cached_contract(self._cache_dir, symbol, result)
            return result

    async def _fetch_daily_bars(self, symbol: str, cd: dict, today_str: str) -> list[dict]:
        """Fetch daily bars with incremental caching."""
        cached_bars, last_updated = _load_cached_bars(self._cache_dir, symbol)

        if last_updated == today_str and len(cached_bars) >= 60:
            return cached_bars

        async with self._sem:
            await self._rate.wait_for()
            contract = self._qualified_contract(symbol, cd)
            duration = "5 D" if (cached_bars and last_updated) else "120 D"

            try:
                bars = await asyncio.wait_for(
                    self._ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting="1 day",
                        whatToShow="TRADES",
                        useRTH=True,
                        keepUpToDate=False,
                    ),
                    timeout=30.0,
                )
            except Exception:
                logger.exception("reqHistoricalData (daily) failed for %s", symbol)
                return cached_bars or []

            new_bars = [
                {
                    "trade_date": str(b.date),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(getattr(b, "volume", 0)),
                }
                for b in (bars or [])
            ]

            if cached_bars and duration == "5 D":
                existing_dates = {b["trade_date"] for b in cached_bars}
                for nb in new_bars:
                    if nb["trade_date"] not in existing_dates:
                        cached_bars.append(nb)
                merged = cached_bars[-120:]
            else:
                merged = new_bars[-120:]

            _save_cached_bars(self._cache_dir, symbol, merged, today_str)
            return merged

    async def _fetch_1min_atr(self, symbol: str, cd: dict) -> float:
        """Fetch 1-min bars and compute 14-period ATR."""
        async with self._sem:
            await self._rate.wait_for()
            contract = self._qualified_contract(symbol, cd)

            try:
                bars = await asyncio.wait_for(
                    self._ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr="1 D",
                        barSizeSetting="1 min",
                        whatToShow="TRADES",
                        useRTH=True,
                        keepUpToDate=False,
                    ),
                    timeout=30.0,
                )
            except Exception:
                logger.exception("reqHistoricalData (1min) failed for %s", symbol)
                return 0.0

            if not bars or len(bars) < 2:
                return 0.0
            return _compute_atr(bars, period=14)
