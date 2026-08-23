"""Acquire immutable, session-qualified stock bars from IBKR.

The default is a fresh direct-RTH acquisition for 1d, 30m, and 5m. ``--latest``
increments only a previously accepted immutable parent. Extended-hours data, when
requested, is written under a separate dataset identity and can never alias RTH.

Usage:
    python -m backtests.stock.data.update_intraday                  # all, direct RTH
    python -m backtests.stock.data.update_intraday --timeframe 1d   # daily only
    python -m backtests.stock.data.update_intraday --timeframe 30m
    python -m backtests.stock.data.update_intraday --timeframe 5m
    python -m backtests.stock.data.update_intraday --session extended --timeframe 5m
"""
from __future__ import annotations

import asyncio
import argparse
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from backtests.shared.data.ibkr.bars import connect_ib
from backtests.shared.data.ibkr.models import BarDownloadRequest, ConnectionSettings
from backtests.shared.data.ibkr.pacing import RequestPacer
from backtests.stock.data.authoritative_downloader import download_authoritative_stock_bars
from backtests.stock.data.authority import DEFAULT_AUTHORITY_ROOT
from backtests.stock.data.calendar import is_trading_day
from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# One canonical universe; callers copy it before adding daily reference symbols.
BACKTESTED_SYMBOLS = list(BACKTESTED_INTRADAY_STOCK_SYMBOLS)

AUTHORITY_ROOT = DEFAULT_AUTHORITY_ROOT

# Reference symbols needed for regime/sector computation in stock backtests
REFERENCE_SYMBOLS = [
    "SPY", "VIX", "HYG",
    "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLB", "XLI", "XLU", "XLRE", "XLC",
]
IARIC_RESIDUAL_DAILY_REFERENCES = [
    "SPY", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE",
    "XLB", "XLI", "XLU", "XLRE", "XLC",
]

PRIMARY_EXCHANGE_BY_SYMBOL = {symbol: exchange for symbol, _sector, exchange in SP500_CONSTITUENTS}


async def update_stock_data(
    timeframes: list[str],
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 113,
    timeout_seconds: int = 60,
    *,
    session: str = "rth",
    start: datetime | None = None,
    end: datetime | None = None,
    latest_only: bool = False,
    repo_root: Path = Path("."),
    authority_root: Path = AUTHORITY_ROOT,
    profile: str = "all",
    daily_what_to_show: str = "TRADES",
) -> None:
    """Acquire session-qualified immutable data for the canonical stock universe."""
    if session not in {"rth", "extended", "both"}:
        raise ValueError("session must be rth, extended, or both")
    if profile not in {"all", "iaric-residual"}:
        raise ValueError("profile must be all or iaric-residual")
    if daily_what_to_show not in {"TRADES", "ADJUSTED_LAST"}:
        raise ValueError("daily what-to-show must be TRADES or ADJUSTED_LAST")
    if profile == "iaric-residual" and set(timeframes) != {"1d"}:
        raise ValueError("the IARIC residual acquisition profile is daily-only")
    start = start or datetime(
        2023 if profile == "iaric-residual" else 2025,
        6 if profile == "iaric-residual" else 3,
        1 if profile == "iaric-residual" else 21,
        tzinfo=timezone.utc,
    )
    end = end or (
        datetime(2026, 3, 1, 23, 59, 59, tzinfo=timezone.utc)
        if profile == "iaric-residual"
        else _last_completed_session_end()
    )
    settings = ConnectionSettings(
        host=host,
        port=port,
        client_id=client_id,
        timeout=timeout_seconds,
    )
    pacer = RequestPacer()
    ib = await connect_ib(settings)
    failures: list[str] = []

    try:
        for tf in timeframes:
            # For 1d, also include reference symbols (SPY, HYG, sector ETFs)
            if tf == "1d":
                seen = set(BACKTESTED_SYMBOLS)
                symbols = list(BACKTESTED_SYMBOLS)
                references = (
                    IARIC_RESIDUAL_DAILY_REFERENCES
                    if profile == "iaric-residual"
                    else REFERENCE_SYMBOLS
                )
                for ref in references:
                    if ref not in seen:
                        symbols.append(ref)
                        seen.add(ref)
            else:
                symbols = list(BACKTESTED_SYMBOLS)

            sessions = ["rth"] if tf == "1d" else (["rth", "extended"] if session == "both" else [session])
            for requested_session in sessions:
                use_rth = requested_session == "rth"
                logger.info(
                    "[stock %s/%s] %d symbols [%s .. %s]",
                    tf,
                    requested_session,
                    len(symbols),
                    start.isoformat(),
                    end.isoformat(),
                )
                for i, sym in enumerate(symbols, 1):
                    is_vix = sym == "VIX"
                    request = BarDownloadRequest(
                        symbol=sym,
                        timeframe=tf,
                        sec_type="IND" if is_vix else "STK",
                        exchange="CBOE" if is_vix else "SMART",
                        primary_exchange="CBOE" if is_vix else PRIMARY_EXCHANGE_BY_SYMBOL.get(sym, ""),
                        what_to_show=(daily_what_to_show if tf == "1d" else "TRADES"),
                        use_rth=use_rth,
                        duration="2 Y",
                        start=start,
                        end=end,
                        output_dir=authority_root,
                        family="stock",
                        adjustment_policy=(
                            "ibkr_adjusted_last_split_dividend_adjusted_v1"
                            if tf == "1d" and daily_what_to_show == "ADJUSTED_LAST"
                            else "ibkr_trades_split_adjusted_not_dividend_v1"
                        ),
                    )
                    logger.info("[stock %s/%s] (%d/%d) %s", tf, requested_session, i, len(symbols), sym)
                    try:
                        result = await download_authoritative_stock_bars(
                            ib,
                            request,
                            repo_root=repo_root,
                            authority_root=authority_root,
                            pacer=pacer,
                            dry_run=False,
                            latest_only=latest_only,
                        )
                        accepted = bool(result.metadata.get("accepted"))
                        logger.info(
                            "  -> %s %s/%s: %d rows [%s .. %s] accepted=%s receipt=%s",
                            result.symbol,
                            result.timeframe,
                            requested_session,
                            result.rows,
                            result.start,
                            result.end,
                            accepted,
                            result.metadata.get("receipt_id", ""),
                        )
                        if not accepted:
                            failures.append(f"{sym} {tf} {requested_session}: validation blocked")
                    except Exception as exc:
                        logger.error("  -> %s %s/%s FAILED: %s", sym, tf, requested_session, exc)
                        failures.append(f"{sym} {tf} {requested_session}: {exc}")
    finally:
        ib.disconnect()
    if failures:
        raise RuntimeError(
            f"authoritative acquisition blocked for {len(failures)} datasets; first failures: "
            + "; ".join(failures[:10])
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", action="append", choices=["1d", "30m", "5m"])
    parser.add_argument("--session", choices=["rth", "extended", "both"], default="rth")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=114)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--authority-root", default=str(AUTHORITY_ROOT))
    parser.add_argument("--profile", choices=["all", "iaric-residual"], default="all")
    parser.add_argument(
        "--daily-what-to-show",
        choices=["TRADES", "ADJUSTED_LAST"],
        default=None,
    )
    args = parser.parse_args()
    start = (
        datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        if args.start
        else None
    )
    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(microseconds=1)
        if args.end
        else None
    )
    asyncio.run(
        update_stock_data(
            args.timeframe
            or (["1d"] if args.profile == "iaric-residual" else ["1d", "30m", "5m"]),
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            timeout_seconds=args.timeout,
            session=args.session,
            start=start,
            end=end,
            latest_only=(args.latest or args.profile == "iaric-residual"),
            repo_root=Path("."),
            authority_root=Path(args.authority_root),
            profile=args.profile,
            daily_what_to_show=(
                args.daily_what_to_show
                or (
                    "ADJUSTED_LAST"
                    if args.profile == "iaric-residual"
                    else "TRADES"
                )
            ),
        )
    )


def _last_completed_session_end() -> datetime:
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    day = now_et.date()
    # Use the previous trading date unless the complete extended session has closed.
    if not is_trading_day(day) or now_et.time() < time(20, 0):
        day -= timedelta(days=1)
        while not is_trading_day(day):
            day -= timedelta(days=1)
    return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)


if __name__ == "__main__":
    main()
