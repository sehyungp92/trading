"""IBKR historical data download via ib_async.

Downloads in backward-walking chunks for resume-on-interrupt support.
Includes retry logic with pacing violation detection and timestamp
deduplication across chunk boundaries.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .cache import bar_path, save_bars

logger = logging.getLogger(__name__)

# IBKR pacing: max 2000 bars per request, 60s between identical requests
_MAX_BARS_PER_REQUEST = 2000
_PACING_DELAY = 1.0   # seconds between requests
_PACING_SLEEP = 60    # seconds to wait on pacing violation
_MAX_RETRIES = 3
_CLIENT_ID = 100

# Chunk durations per timeframe (stay well under 2000 bars per chunk)
_CHUNK_DURATION: dict[str, str] = {
    "1m": "1 D",    # ~1440 bars/day
    "5m": "1 W",    # ~2016 bars/week (288/day × 7)
    "15m": "3 W",   # ~2016 bars/3 weeks (96/day × 21)
    "1h": "2 M",    # ~1440 hourly bars per chunk
    "1d": "1 Y",    # ~252 daily bars per chunk
}


def _timeframe_to_ibkr(timeframe: str) -> str:
    """Map our timeframe labels to IBKR barSizeSetting strings."""
    mapping = {
        "1m": "1 min",
        "5m": "5 mins",
        "15m": "15 mins",
        "1h": "1 hour",
        "1d": "1 day",
    }
    return mapping.get(timeframe, timeframe)


def _duration_to_days(duration: str) -> int:
    """Parse an IBKR duration string to approximate days."""
    parts = duration.strip().split()
    num = int(parts[0])
    unit = parts[1].upper()
    if unit == "Y":
        return num * 365
    if unit == "M":
        return int(num * 30.44)
    if unit == "W":
        return num * 7
    if unit == "D":
        return num
    raise ValueError(f"Unknown duration unit: {duration}")


def _chunk_step(duration: str) -> timedelta:
    """Convert a chunk duration string to a timedelta."""
    parts = duration.strip().split()
    num = int(parts[0])
    unit = parts[1].upper()
    if unit == "Y":
        return timedelta(days=365 * num)
    if unit == "M":
        return timedelta(days=30 * num)
    if unit == "W":
        return timedelta(weeks=num)
    if unit == "D":
        return timedelta(days=num)
    raise ValueError(f"Unknown duration unit: {duration}")


def _chunk_filename(symbol: str, timeframe: str, end_dt: datetime) -> str:
    """Deterministic chunk filename from symbol + timeframe + endDateTime."""
    ts = end_dt.strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{timeframe}_{ts}.parquet"


def _bars_to_df(bars) -> pd.DataFrame:
    """Convert ib_async bar objects to a DataFrame with DatetimeIndex."""
    records = []
    for b in bars:
        records.append({
            "time": b.date if isinstance(b.date, datetime) else pd.Timestamp(b.date),
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": int(b.volume),
        })
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df


def _ensure_utc(dt: datetime) -> datetime:
    """Guard against timezone-naive datetimes from pd.Timestamp conversion."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _build_stock(symbol: str, exchange: str, currency: str = "USD"):
    """Build a Stock contract for ETF data."""
    from ib_async import Stock

    return Stock(symbol=symbol, exchange=exchange, currency=currency)


def _build_cont_future(ib, symbol: str, exchange: str, trading_class: str):
    """Build a ContFuture contract for continuous back-adjusted data."""
    from ib_async import ContFuture

    contract = ContFuture(
        symbol=symbol,
        exchange=exchange,
        tradingClass=trading_class or symbol,
    )
    return contract


async def _resolve_chunked_contract(
    ib, symbol: str, exchange: str, trading_class: str,
):
    """Qualify ContFuture and return a Future(conId=...) for explicit endDateTime.

    IBKR error 10339 prevents setting endDateTime on ContFuture directly.
    Workaround: qualify ContFuture to get the conId, then use a plain Future.
    """
    from ib_async import Future

    cont = _build_cont_future(ib, symbol, exchange, trading_class)
    qualified = await ib.qualifyContractsAsync(cont)
    if not qualified:
        raise ValueError(f"Could not qualify contract for {symbol}")
    resolved = qualified[0]

    contract = Future(conId=resolved.conId, exchange=exchange)
    await ib.qualifyContractsAsync(contract)
    return contract


async def _reconnect(ib, host: str = "127.0.0.1", port: int = 7496,
                     client_id: int = _CLIENT_ID, timeout: int = 30) -> None:
    """Reconnect a disconnected IB instance."""
    try:
        ib.disconnect()
    except Exception:
        pass
    await asyncio.sleep(5)
    logger.info("Reconnecting to %s:%d ...", host, port)
    await ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
    logger.info("Reconnected successfully")


async def _request_with_retry(
    ib,
    contract,
    end_dt: datetime | str,
    duration: str,
    bar_size: str,
    use_rth: bool,
) -> list:
    """Request historical data with retry logic and pacing violation handling.

    Retries up to _MAX_RETRIES times. Pacing violations (error 162) trigger
    a 60-second cooldown. Disconnections trigger a reconnect attempt.
    Other errors use a shorter backoff.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if not ib.isConnected():
                await _reconnect(ib)
            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_dt,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=2,
                    timeout=0,  # disable ib_async internal timeout
                ),
                timeout=60,  # 1-minute timeout
            )
            return bars or []
        except Exception as e:
            msg = str(e).lower()
            if "pacing" in msg or "162" in msg:
                logger.warning(
                    "Pacing violation (attempt %d/%d), sleeping %ds...",
                    attempt, _MAX_RETRIES, _PACING_SLEEP,
                )
                await asyncio.sleep(_PACING_SLEEP)
            elif "disconnect" in msg or "not connected" in msg or "winerror" in msg:
                logger.warning(
                    "Connection lost (attempt %d/%d): %s. Reconnecting...",
                    attempt, _MAX_RETRIES, e,
                )
                try:
                    await _reconnect(ib)
                except Exception as re_err:
                    logger.error("Reconnect failed: %s", re_err)
                    if attempt >= _MAX_RETRIES:
                        return []
                await asyncio.sleep(_PACING_DELAY * 3)
            elif attempt < _MAX_RETRIES:
                logger.warning(
                    "Request error (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, e,
                )
                await asyncio.sleep(_PACING_DELAY * 3)
            else:
                logger.error(
                    "Failed after %d attempts: %s", _MAX_RETRIES, e,
                )
                return []
    return []


async def download_historical(
    ib,
    symbol: str,
    timeframe: str,
    duration: str,
    exchange: str,
    trading_class: str = "",
    rth_only: bool = False,
    output_dir: Path = Path("backtest/data/raw"),
    sec_type: str = "FUT",
    primary_exchange: str = "",
) -> pd.DataFrame:
    """Download historical bars using ContFuture or Stock contract.

    For futures (sec_type="FUT"): Uses ContFuture with endDateTime='' and
    year-sized chunks for continuous back-adjusted data.

    For stocks/ETFs (sec_type="STK"): Uses Stock contract with a single
    request per timeframe.

    Args:
        ib: Connected ib_async.IB instance
        symbol: Root symbol (e.g. "NQ" or "QQQ")
        timeframe: "1h" or "1d"
        duration: Total IBKR duration string (e.g. "5 Y")
        exchange: Exchange name (e.g. "CME" or "SMART")
        trading_class: Contract trading class (futures only)
        rth_only: True for daily (RTH only), False for hourly (all hours)
        output_dir: Base directory for chunk cache and final output
        sec_type: Security type ("FUT" or "STK")
        primary_exchange: Primary exchange for STK contracts

    Returns:
        DataFrame with columns [open, high, low, close, volume] and DatetimeIndex
    """
    bar_size = _timeframe_to_ibkr(timeframe)
    total_days = _duration_to_days(duration)

    # --- Stock / ETF path ---
    if sec_type == "STK":
        stock = _build_stock(symbol, exchange)
        qualified = await ib.qualifyContractsAsync(stock)
        if not qualified:
            raise ValueError(f"Could not qualify Stock for {symbol}")
        stock = qualified[0]

        all_chunks: list[pd.DataFrame] = []

        if timeframe == "1d":
            logger.info("Downloading %s %s via Stock (duration=%s)...", symbol, timeframe, duration)
            bars = await _request_with_retry(
                ib, stock, "", duration, bar_size, rth_only,
            )
            if bars:
                df = _bars_to_df(bars)
                all_chunks.append(df)
                logger.info(
                    "[ok] %s %s: %d bars (%s -> %s)",
                    symbol, timeframe, len(df), df.index[0], df.index[-1],
                )
        else:
            years_needed = (total_days // 365) + 1
            for yr in range(1, years_needed + 1):
                chunk_dur = f"{yr} Y"
                logger.info(
                    "Downloading %s %s via Stock (duration=%s)...",
                    symbol, timeframe, chunk_dur,
                )
                bars = await _request_with_retry(
                    ib, stock, "", chunk_dur, bar_size, rth_only,
                )
                if bars:
                    df = _bars_to_df(bars)
                    all_chunks.append(df)
                    logger.info(
                        "[ok] %s %s: %d bars (%s -> %s)",
                        symbol, timeframe, len(df), df.index[0], df.index[-1],
                    )
                    earliest = _ensure_utc(df.index[0].to_pydatetime())
                    now = datetime.now(timezone.utc)
                    if (now - earliest).days >= total_days:
                        break
                else:
                    logger.info("[empty] %s %s duration=%s", symbol, timeframe, chunk_dur)
                    break
                await asyncio.sleep(_PACING_DELAY)

        if not all_chunks:
            raise ValueError(f"No data returned for {symbol} {timeframe}")

        combined = pd.concat(all_chunks)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        logger.info(
            "Downloaded %d bars for %s %s (%s -> %s)",
            len(combined), symbol, timeframe,
            combined.index[0], combined.index[-1],
        )
        return combined

    # --- Futures path ---
    all_chunks: list[pd.DataFrame] = []

    # For daily bars, use ContFuture with endDateTime='' (full duration in one shot)
    if timeframe == "1d":
        cont_contract = _build_cont_future(ib, symbol, exchange, trading_class)
        qualified = await ib.qualifyContractsAsync(cont_contract)
        if not qualified:
            raise ValueError(f"Could not qualify ContFuture for {symbol}")
        cont_contract = qualified[0]

        logger.info("Downloading %s %s via ContFuture (duration=%s)...", symbol, timeframe, duration)
        bars = await _request_with_retry(
            ib, cont_contract, "", duration, bar_size, rth_only,
        )
        if bars:
            df = _bars_to_df(bars)
            all_chunks.append(df)
            logger.info(
                "[ok] %s %s: %d bars (%s -> %s)",
                symbol, timeframe, len(df), df.index[0], df.index[-1],
            )

    elif timeframe == "1h":
        # For hourly bars, ContFuture with progressive year-sized durations
        cont_contract = _build_cont_future(ib, symbol, exchange, trading_class)
        qualified = await ib.qualifyContractsAsync(cont_contract)
        if not qualified:
            raise ValueError(f"Could not qualify ContFuture for {symbol}")
        cont_contract = qualified[0]

        years_needed = (total_days // 365) + 1
        for yr in range(1, years_needed + 1):
            chunk_dur = f"{yr} Y"
            logger.info(
                "Downloading %s %s via ContFuture (duration=%s)...",
                symbol, timeframe, chunk_dur,
            )
            bars = await _request_with_retry(
                ib, cont_contract, "", chunk_dur, bar_size, rth_only,
            )
            if bars:
                df = _bars_to_df(bars)
                all_chunks.append(df)
                logger.info(
                    "[ok] %s %s: %d bars (%s -> %s)",
                    symbol, timeframe, len(df), df.index[0], df.index[-1],
                )
                earliest = _ensure_utc(df.index[0].to_pydatetime())
                now = datetime.now(timezone.utc)
                if (now - earliest).days >= total_days:
                    break
            else:
                logger.info("[limit] %s %s max reached at %s", symbol, timeframe, chunk_dur)
                break
            await asyncio.sleep(_PACING_DELAY)

    else:
        # For 5m, 15m, 1m: IBKR limits per-request duration.
        # First grab max via ContFuture (endDateTime=''), then walk backward
        # with Future(conId=...) using 1D chunks for older data.
        #
        # Tested max durations with ContFuture endDateTime='':
        #   1m  -> 1 W
        #   5m  -> 1 M
        #   15m -> 1 M (conservative)
        _MAX_CONTFUT_DURATION = {"1m": "1 W", "5m": "1 M", "15m": "1 M"}
        contfut_dur = _MAX_CONTFUT_DURATION.get(timeframe, "1 W")

        cont_contract = _build_cont_future(ib, symbol, exchange, trading_class)
        qualified = await ib.qualifyContractsAsync(cont_contract)
        if not qualified:
            raise ValueError(f"Could not qualify ContFuture for {symbol}")
        cont_contract = qualified[0]

        # Phase 1: Get the latest chunk via ContFuture
        logger.info("Downloading %s %s via ContFuture (duration=%s)...",
                     symbol, timeframe, contfut_dur)
        bars = await _request_with_retry(
            ib, cont_contract, "", contfut_dur, bar_size, rth_only,
        )
        if bars:
            df = _bars_to_df(bars)
            all_chunks.append(df)
            earliest_so_far = _ensure_utc(df.index[0].to_pydatetime())
            logger.info(
                "[ok] %s %s: %d bars (%s -> %s)",
                symbol, timeframe, len(df), df.index[0], df.index[-1],
            )
        else:
            earliest_so_far = datetime.now(timezone.utc)

        # Phase 2: Walk backward with Future(conId=...) + explicit endDateTime
        cutoff = datetime.now(timezone.utc) - timedelta(days=total_days)
        if earliest_so_far > cutoff:
            chunk_dur = "1 D"
            step = _chunk_step(chunk_dur)
            contract = await _resolve_chunked_contract(
                ib, symbol, exchange, trading_class,
            )

            end_dt = earliest_so_far - timedelta(seconds=1)
            empty_streak = 0
            max_empty = 5

            logger.info(
                "Walking backward %s %s in %s chunks (to %s)...",
                symbol, timeframe, chunk_dur, cutoff.strftime("%Y-%m-%d"),
            )

            chunk_num = 0
            while end_dt > cutoff and empty_streak < max_empty:
                chunk_num += 1
                end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")
                bars = await _request_with_retry(
                    ib, contract, end_str, chunk_dur, bar_size, rth_only,
                )
                if bars:
                    df = _bars_to_df(bars)
                    all_chunks.append(df)
                    empty_streak = 0
                    earliest = _ensure_utc(df.index[0].to_pydatetime())
                    if chunk_num % 50 == 0:
                        logger.info(
                            "[ok] %s %s chunk %d: %d bars (back to %s)",
                            symbol, timeframe, chunk_num, len(df),
                            earliest.strftime("%Y-%m-%d"),
                        )
                    end_dt = earliest - timedelta(seconds=1)
                else:
                    empty_streak += 1
                    end_dt -= step
                await asyncio.sleep(_PACING_DELAY)

            if chunk_num > 0:
                logger.info(
                    "Completed %s %s backward walk: %d chunks", symbol, timeframe, chunk_num,
                )

    if not all_chunks:
        raise ValueError(f"No data returned for {symbol} {timeframe}")

    # Stitch chunks, deduplicate overlapping boundaries, sort
    combined = pd.concat(all_chunks)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    logger.info(
        "Downloaded %d bars for %s %s (%s -> %s)",
        len(combined), symbol, timeframe,
        combined.index[0], combined.index[-1],
    )
    return combined


async def download_all_symbols(
    symbols: list[str],
    configs: dict,
    duration: str = "5 Y",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, dict[str, Path]]:
    """Download hourly + daily data for all symbols.

    Args:
        symbols: List of symbol names
        configs: Dict of SymbolConfig keyed by symbol
        duration: IBKR duration string
        output_dir: Directory to save parquet files

    Returns:
        Nested dict: {symbol: {timeframe: path}}
    """
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=_CLIENT_ID, timeout=30)

    result: dict[str, dict[str, Path]] = {}

    try:
        for sym in symbols:
            cfg = configs[sym]
            result[sym] = {}

            for tf, rth in [("1h", False), ("1d", True)]:
                logger.info("Downloading %s %s ...", sym, tf)
                df = await download_historical(
                    ib, sym, tf, duration,
                    exchange=cfg.exchange,
                    trading_class=cfg.trading_class,
                    rth_only=rth,
                    output_dir=output_dir,
                    sec_type=cfg.sec_type,
                    primary_exchange=cfg.primary_exchange,
                )
                path = bar_path(output_dir, sym, tf)
                save_bars(df, path)
                result[sym][tf] = path
                logger.info("Saved %s %s -> %s (%d bars)", sym, tf, path, len(df))

                await asyncio.sleep(_PACING_DELAY)

    finally:
        ib.disconnect()

    return result


async def download_apex_data(
    symbol: str = "NQ",
    duration: str = "5 Y",
    exchange: str = "CME",
    trading_class: str = "NQ",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, Path]:
    """Download 5-minute + daily data for NQ futures backtesting.

    IBKR provides several years of 5-minute data for futures (daily goes to 5Y+).
    1H and 4H bars are resampled from 5-min in the preprocessing step.

    Returns:
        Dict mapping timeframe to saved path, e.g. {"5m": Path(...), "1d": Path(...)}.
    """
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=_CLIENT_ID, timeout=30)

    result: dict[str, Path] = {}

    try:
        for tf, rth in [("5m", False), ("1d", True)]:
            logger.info("Downloading %s %s ...", symbol, tf)
            df = await download_historical(
                ib, symbol, tf, duration,
                exchange=exchange,
                trading_class=trading_class,
                rth_only=rth,
                output_dir=output_dir,
                sec_type="FUT",
            )
            path = bar_path(output_dir, symbol, tf)
            save_bars(df, path)
            result[tf] = path
            logger.info("Saved %s %s -> %s (%d bars)", symbol, tf, path, len(df))

            await asyncio.sleep(_PACING_DELAY)

        available_days = 0
        if "5m" in result:
            five_min_df = pd.read_parquet(result["5m"], engine="pyarrow")
            span = five_min_df.index[-1] - five_min_df.index[0]
            available_days = span.days
            requested_days = _duration_to_days(duration)
            if available_days < requested_days:
                logger.warning(
                    "IBKR returned %d days of 5-min data (requested %d).",
                    available_days, requested_days,
                )
    finally:
        ib.disconnect()

    return result


async def download_nqdtc_data(
    symbol: str = "NQ",
    duration: str = "5 Y",
    exchange: str = "CME",
    trading_class: str = "NQ",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, Path]:
    """Download 5-minute + daily data for NQ futures (NQDTC strategy).

    Returns:
        Dict mapping timeframe to saved path, e.g. {"5m": Path(...), "1d": Path(...)}.
    """
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=_CLIENT_ID, timeout=30)

    result: dict[str, Path] = {}

    try:
        for tf, rth in [("5m", False), ("1d", True)]:
            path = bar_path(output_dir, symbol, tf)
            if tf == "1d" and path.exists():
                logger.info("Skipping %s %s (already exists at %s)", symbol, tf, path)
                result[tf] = path
                continue

            logger.info("Downloading %s %s ...", symbol, tf)
            df = await download_historical(
                ib, symbol, tf, duration,
                exchange=exchange,
                trading_class=trading_class,
                rth_only=rth,
                output_dir=output_dir,
                sec_type="FUT",
            )
            path = bar_path(output_dir, symbol, tf)
            save_bars(df, path)
            result[tf] = path
            logger.info("Saved %s %s -> %s (%d bars)", symbol, tf, path, len(df))

            await asyncio.sleep(_PACING_DELAY)

    finally:
        ib.disconnect()

    return result


async def download_vdubus_data(
    symbol: str = "NQ",
    duration: str = "5 Y",
    exchange: str = "CME",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, Path]:
    """Download NQ 15-minute + ES daily data for VdubusNQ strategy.

    Returns:
        Dict mapping key to saved path, e.g. {"15m": Path(...), "ES_1d": Path(...)}.
    """
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=_CLIENT_ID, timeout=30)

    result: dict[str, Path] = {}

    try:
        # NQ 15m bars
        logger.info("Downloading %s 15m ...", symbol)
        df_15m = await download_historical(
            ib, symbol, "15m", duration,
            exchange=exchange,
            trading_class=symbol,
            rth_only=False,
            output_dir=output_dir,
            sec_type="FUT",
        )
        path_15m = bar_path(output_dir, symbol, "15m")
        save_bars(df_15m, path_15m)
        result["15m"] = path_15m
        logger.info("Saved %s 15m -> %s (%d bars)", symbol, path_15m, len(df_15m))

        await asyncio.sleep(_PACING_DELAY)

        # ES daily bars for regime filter
        es_daily_path = bar_path(output_dir, "ES", "1d")
        if es_daily_path.exists():
            logger.info("Skipping ES 1d (already exists at %s)", es_daily_path)
            result["ES_1d"] = es_daily_path
        else:
            logger.info("Downloading ES 1d ...")
            df_es = await download_historical(
                ib, "ES", "1d", duration,
                exchange=exchange,
                trading_class="ES",
                rth_only=True,
                output_dir=output_dir,
                sec_type="FUT",
            )
            save_bars(df_es, es_daily_path)
            result["ES_1d"] = es_daily_path
            logger.info("Saved ES 1d -> %s (%d bars)", es_daily_path, len(df_es))

    finally:
        ib.disconnect()

    return result
