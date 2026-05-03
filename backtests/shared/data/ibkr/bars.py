"""Shared IBKR historical bar downloader primitives."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import FuturesContractSpec
from .models import BarDownloadRequest, ConnectionSettings, DownloadResult, DownloadWindow
from .pacing import RequestPacer, request_weight
from .store import detect_large_gaps, ensure_utc_index, merge_frames, read_parquet_if_exists, write_parquet_atomic

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
PACING_VIOLATION_SLEEP_SECONDS = 65

IB_BAR_SIZES: dict[str, str] = {
    "1s": "1 secs",
    "5s": "5 secs",
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "daily": "1 day",
}

CHUNK_DURATIONS: dict[str, str] = {
    "1s": "1800 S",
    "5s": "2 H",
    "1m": "1 D",
    "5m": "1 W",
    "15m": "3 W",
    "30m": "1 M",
    "1h": "2 M",
    "4h": "6 M",
    "1d": "1 Y",
    "daily": "1 Y",
}


def timeframe_to_ibkr(timeframe: str) -> str:
    return IB_BAR_SIZES.get(timeframe, timeframe)


def duration_to_timedelta(duration: str) -> timedelta:
    number_text, unit = duration.strip().split(maxsplit=1)
    number = int(number_text)
    unit = unit.upper()
    if unit == "S":
        return timedelta(seconds=number)
    if unit == "H":
        return timedelta(hours=number)
    if unit == "D":
        return timedelta(days=number)
    if unit == "W":
        return timedelta(weeks=number)
    if unit == "M":
        return timedelta(days=30 * number)
    if unit == "Y":
        return timedelta(days=365 * number)
    raise ValueError(f"Unsupported IBKR duration unit: {duration!r}")


def duration_to_days(duration: str) -> int:
    return max(1, duration_to_timedelta(duration).days)


def plan_bar_windows(start: datetime, end: datetime, timeframe: str) -> list[DownloadWindow]:
    start = _ensure_utc_dt(start)
    end = _ensure_utc_dt(end)
    if end <= start:
        return []
    duration = CHUNK_DURATIONS.get(timeframe, "1 W")
    step = duration_to_timedelta(duration)
    windows: list[DownloadWindow] = []
    cursor = end
    while cursor > start:
        window_start = max(start, cursor - step)
        windows.append(DownloadWindow(start=window_start, end=cursor, duration=duration))
        cursor = window_start - timedelta(seconds=1)
    return windows


async def connect_ib(settings: ConnectionSettings):
    from ib_async import IB

    ib = IB()
    await ib.connectAsync(settings.host, settings.port, clientId=settings.client_id, timeout=settings.timeout)
    return ib


async def download_contract_bars(
    ib: Any,
    contract_spec: FuturesContractSpec,
    *,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    what_to_show: str = "TRADES",
    use_rth: bool = False,
    pacer: RequestPacer | None = None,
    dry_run: bool = False,
    latest_only: bool = False,
) -> DownloadResult:
    start = _ensure_utc_dt(start)
    end = _ensure_utc_dt(end)
    existing = read_parquet_if_exists(output_path)
    effective_start = start
    if latest_only and not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(timeframe, "1 D"))
        effective_start = max(start, existing.index[-1].to_pydatetime() - overlap)

    windows = _plan_windows_with_existing_gaps(
        existing,
        start=effective_start,
        end=end,
        timeframe=timeframe,
        include_existing_gaps=latest_only,
        hard_start=start,
    )
    if dry_run:
        return DownloadResult(
            symbol=contract_spec.symbol,
            timeframe=timeframe,
            what_to_show=what_to_show,
            dry_run=True,
            paths=[output_path],
            messages=[
                f"{contract_spec.local_symbol} {timeframe} {what_to_show}: {len(windows)} IBKR bar requests"
            ],
        )
    if not windows:
        return DownloadResult(
            symbol=contract_spec.symbol,
            timeframe=timeframe,
            what_to_show=what_to_show,
            rows=len(existing),
            start=_frame_start(existing),
            end=_frame_end(existing),
            paths=[output_path],
            messages=[f"{contract_spec.local_symbol} {timeframe}: up to date"],
        )

    pacer = pacer or RequestPacer()
    contract = await build_future_contract(ib, contract_spec)
    chunks: list[pd.DataFrame] = []
    empty_streak = 0
    previous_earliest: datetime | None = None
    for window in windows:
        bars = await request_bars_with_retry(
            ib,
            contract,
            end_dt=window.end,
            duration=window.duration,
            timeframe=timeframe,
            what_to_show=what_to_show,
            use_rth=use_rth,
            pacer=pacer,
        )
        if bars:
            frame = bars_to_frame(bars)
            if not frame.empty:
                earliest = frame.index[0].to_pydatetime()
                if previous_earliest is not None and earliest >= previous_earliest:
                    logger.info("%s %s stale progress at %s", contract_spec.local_symbol, timeframe, earliest)
                    break
                previous_earliest = earliest
                chunks.append(frame)
                empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 3 and not latest_only:
                break

    merged = merge_frames(existing, *chunks)
    if not merged.empty:
        merged = merged[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]
        write_parquet_atomic(merged, output_path)

    return DownloadResult(
        symbol=contract_spec.symbol,
        timeframe=timeframe,
        what_to_show=what_to_show,
        rows=len(merged),
        start=_frame_start(merged),
        end=_frame_end(merged),
        paths=[output_path],
        messages=[f"{contract_spec.local_symbol} {timeframe} {what_to_show}: {len(merged)} rows"],
    )


async def download_historical_bars(
    ib: Any,
    request: BarDownloadRequest,
    *,
    output_path: Path | None = None,
    pacer: RequestPacer | None = None,
    dry_run: bool = False,
    latest_only: bool = False,
) -> DownloadResult | pd.DataFrame:
    """Generic stock/continuous-future downloader used by legacy wrappers."""
    end = request.end or datetime.now(timezone.utc)
    start = request.start or (end - duration_to_timedelta(request.duration))
    target_path = output_path or request.output_dir / f"{request.symbol}_{request.timeframe}.parquet"
    existing = read_parquet_if_exists(target_path) if latest_only else pd.DataFrame()
    effective_start = start
    if latest_only and not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(request.timeframe, "1 D"))
        effective_start = max(start, existing.index[-1].to_pydatetime() - overlap)
    windows = _plan_windows_with_existing_gaps(
        existing,
        start=effective_start,
        end=end,
        timeframe=request.timeframe,
        include_existing_gaps=latest_only,
        hard_start=start,
    )
    if dry_run:
        return DownloadResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            dry_run=True,
            paths=[target_path],
            messages=[f"{request.symbol} {request.timeframe}: {len(windows)} IBKR bar requests"],
        )

    pacer = pacer or RequestPacer()
    if request.sec_type.upper() == "FUT":
        downloaded = await _download_continuous_future(ib, request, start=effective_start, end=end, pacer=pacer)
    else:
        contract = await build_generic_contract(ib, request)
        chunks: list[pd.DataFrame] = []
        for window in windows:
            bars = await request_bars_with_retry(
                ib,
                contract,
                end_dt=window.end,
                duration=window.duration,
                timeframe=request.timeframe,
                what_to_show=request.what_to_show,
                use_rth=request.use_rth,
                pacer=pacer,
            )
            if bars:
                chunks.append(bars_to_frame(bars))
        downloaded = merge_frames(*chunks)
    merged = merge_frames(existing, downloaded) if latest_only else downloaded
    if output_path is not None and not merged.empty:
        write_parquet_atomic(merged, output_path)
        return DownloadResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            rows=len(merged),
            start=_frame_start(merged),
            end=_frame_end(merged),
            paths=[output_path],
        )
    return merged


async def _download_continuous_future(
    ib: Any,
    request: BarDownloadRequest,
    *,
    start: datetime,
    end: datetime,
    pacer: RequestPacer | None,
) -> pd.DataFrame:
    cont_contract = await build_generic_contract(ib, request)
    chunks: list[pd.DataFrame] = []

    if request.timeframe in {"1d", "daily"}:
        bars = await request_bars_with_retry(
            ib,
            cont_contract,
            end_dt="",
            duration=request.duration,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            use_rth=request.use_rth,
            pacer=pacer,
        )
        if bars:
            chunks.append(bars_to_frame(bars))
        return merge_frames(*chunks)

    latest_duration = {
        "1m": "1 W",
        "5m": "1 M",
        "15m": "1 M",
        "30m": "2 M",
        "1h": "1 Y",
        "4h": "2 Y",
    }.get(request.timeframe, CHUNK_DURATIONS.get(request.timeframe, "1 W"))
    bars = await request_bars_with_retry(
        ib,
        cont_contract,
        end_dt="",
        duration=latest_duration,
        timeframe=request.timeframe,
        what_to_show=request.what_to_show,
        use_rth=request.use_rth,
        pacer=pacer,
    )
    earliest = end
    if bars:
        latest_frame = bars_to_frame(bars)
        chunks.append(latest_frame)
        earliest = latest_frame.index[0].to_pydatetime()

    if earliest <= start:
        return merge_frames(*chunks)

    chunk_contract = await build_chunked_continuous_future(ib, request)
    cursor = earliest - timedelta(seconds=1)
    empty_streak = 0
    previous_earliest: datetime | None = None
    duration = CHUNK_DURATIONS.get(request.timeframe, "1 W")
    while cursor > start and empty_streak < 5:
        bars = await request_bars_with_retry(
            ib,
            chunk_contract,
            end_dt=cursor,
            duration=duration,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            use_rth=request.use_rth,
            pacer=pacer,
        )
        if bars:
            frame = bars_to_frame(bars)
            chunks.append(frame)
            empty_streak = 0
            frame_earliest = frame.index[0].to_pydatetime()
            if previous_earliest is not None and frame_earliest >= previous_earliest:
                break
            previous_earliest = frame_earliest
            cursor = frame_earliest - timedelta(seconds=1)
        else:
            empty_streak += 1
            cursor = cursor - duration_to_timedelta(duration)
    merged = merge_frames(*chunks)
    if merged.empty:
        return merged
    return merged[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]


async def request_bars_with_retry(
    ib: Any,
    contract: Any,
    *,
    end_dt: datetime | str,
    duration: str,
    timeframe: str,
    what_to_show: str,
    use_rth: bool,
    pacer: RequestPacer | None = None,
    timeout: int = 120,
) -> list[Any]:
    bar_size = timeframe_to_ibkr(timeframe)
    end_str = _format_ib_end(end_dt)
    signature = (
        getattr(contract, "conId", None),
        getattr(contract, "symbol", None),
        getattr(contract, "lastTradeDateOrContractMonth", None),
        end_str,
        duration,
        bar_size,
        what_to_show,
        use_rth,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if pacer is not None:
                await pacer.wait(signature, weight=request_weight(what_to_show))
            return await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_str,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=2,
                    timeout=0,
                ),
                timeout=timeout,
            ) or []
        except Exception as exc:
            message = str(exc).lower()
            if "pacing" in message or "162" in message:
                logger.warning("IBKR pacing violation on %s %s; sleeping %ss", getattr(contract, "symbol", "?"), timeframe, PACING_VIOLATION_SLEEP_SECONDS)
                await asyncio.sleep(PACING_VIOLATION_SLEEP_SECONDS)
            elif "no data" in message or "HMDS query returned no data".lower() in message:
                return []
            elif attempt >= MAX_RETRIES:
                logger.warning("IBKR request failed after %d attempts: %s", MAX_RETRIES, exc)
                return []
            else:
                await asyncio.sleep(5 * attempt)
    return []


async def build_future_contract(ib: Any, spec: FuturesContractSpec):
    from ib_async import Future

    contract = Future(
        symbol=spec.symbol,
        exchange=spec.exchange,
        tradingClass=spec.ib_trading_class,
        lastTradeDateOrContractMonth=spec.yyyymm,
        includeExpired=True,
    )
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise ValueError(f"Could not qualify {spec.local_symbol}")
    contract = qualified[0]
    contract.includeExpired = True
    return contract


async def build_generic_contract(ib: Any, request: BarDownloadRequest):
    if request.sec_type.upper() == "STK":
        from ib_async import Stock

        contract = Stock(symbol=request.symbol, exchange=request.exchange, currency=request.currency)
    elif request.sec_type.upper() == "FUT":
        from ib_async import ContFuture

        contract = ContFuture(
            symbol=request.symbol,
            exchange=request.exchange,
            tradingClass=request.ib_trading_class,
        )
    else:
        raise ValueError(f"Unsupported sec_type: {request.sec_type}")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise ValueError(f"Could not qualify {request.sec_type} {request.symbol}")
    return qualified[0]


async def build_chunked_continuous_future(ib: Any, request: BarDownloadRequest):
    from ib_async import Future

    cont = await build_generic_contract(ib, request)
    contract = Future(conId=cont.conId, exchange=request.exchange)
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise ValueError(f"Could not resolve chunked continuous future for {request.symbol}")
    return qualified[0]


def bars_to_frame(bars: list[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for bar in bars:
        row = {
            "time": bar.date if isinstance(bar.date, datetime) else pd.Timestamp(bar.date),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": int(getattr(bar, "volume", 0) or 0),
        }
        if hasattr(bar, "barCount"):
            row["bar_count"] = int(getattr(bar, "barCount") or 0)
        if hasattr(bar, "average"):
            row["wap"] = float(getattr(bar, "average") or 0.0)
        records.append(row)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return ensure_utc_index(frame.set_index("time"))


def _plan_windows_with_existing_gaps(
    existing: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
    include_existing_gaps: bool,
    hard_start: datetime,
) -> list[DownloadWindow]:
    windows = plan_bar_windows(start, end, timeframe)
    if include_existing_gaps and not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(timeframe, "1 D"))
        gap_windows: list[DownloadWindow] = []
        for gap in detect_large_gaps(existing, timeframe):
            gap_start = max(hard_start, gap.start - overlap)
            gap_end = min(end, gap.end + overlap)
            gap_windows.extend(plan_bar_windows(gap_start, gap_end, timeframe))
        windows = gap_windows + windows
    return _dedupe_windows(windows)


def _dedupe_windows(windows: list[DownloadWindow]) -> list[DownloadWindow]:
    seen: set[tuple[datetime, datetime, str]] = set()
    unique: list[DownloadWindow] = []
    for window in sorted(windows, key=lambda item: item.end, reverse=True):
        key = (window.start, window.end, window.duration)
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
    return unique


def _ensure_utc_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_ib_end(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    value = _ensure_utc_dt(value)
    return value.strftime("%Y%m%d %H:%M:%S UTC")


def _frame_start(df: pd.DataFrame) -> datetime | None:
    if df.empty:
        return None
    return df.index[0].to_pydatetime()


def _frame_end(df: pd.DataFrame) -> datetime | None:
    if df.empty:
        return None
    return df.index[-1].to_pydatetime()
