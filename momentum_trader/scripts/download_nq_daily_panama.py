"""Download individual NQ quarterly contracts (daily bars) and stitch with Panama adjustment.

Downloads each quarterly contract from NQH4 through NQH6, stitches them on
the Monday of expiration week (liquidity roll date), and applies a backward
Panama price adjustment for a seamless continuous series.

Output: NQ_daily_panama.parquet + NQ_daily_panama.csv in backtest/data/raw/

Usage:
    python scripts/download_nq_daily_panama.py [--host HOST] [--port PORT]
                                                [--client-id CID]

Requires: IB Gateway or TWS running and accepting API connections.
"""
import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_nq_daily_panama.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("backtest/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PACING_DELAY = 12  # seconds between requests
PACING_VIOLATION_SLEEP = 65  # seconds after error 162
MAX_RETRIES = 4
CONNECT_TIMEOUT = 180
CHUNK_DURATION = "1 Y"  # daily bars: 1-year chunks (~252 bars, well under IBKR limit)

# Hardcoded contract list with liquidity roll dates (Monday of expiry week).
# These are the actual dates where front-month volume migrates to the next contract.
CONTRACTS = [
    {"yyyymm": "202403", "code": "H", "local_sym": "NQH4", "expiry": date(2024, 3, 15), "roll_date": date(2024, 3, 11)},
    {"yyyymm": "202406", "code": "M", "local_sym": "NQM4", "expiry": date(2024, 6, 21), "roll_date": date(2024, 6, 10)},
    {"yyyymm": "202409", "code": "U", "local_sym": "NQU4", "expiry": date(2024, 9, 20), "roll_date": date(2024, 9, 16)},
    {"yyyymm": "202412", "code": "Z", "local_sym": "NQZ4", "expiry": date(2024, 12, 20), "roll_date": date(2024, 12, 16)},
    {"yyyymm": "202503", "code": "H", "local_sym": "NQH5", "expiry": date(2025, 3, 21), "roll_date": date(2025, 3, 17)},
    {"yyyymm": "202506", "code": "M", "local_sym": "NQM5", "expiry": date(2025, 6, 20), "roll_date": date(2025, 6, 16)},
    {"yyyymm": "202509", "code": "U", "local_sym": "NQU5", "expiry": date(2025, 9, 19), "roll_date": date(2025, 9, 15)},
    {"yyyymm": "202512", "code": "Z", "local_sym": "NQZ5", "expiry": date(2025, 12, 19), "roll_date": date(2025, 12, 15)},
    {"yyyymm": "202603", "code": "H", "local_sym": "NQH6", "expiry": date(2026, 3, 20), "roll_date": date(2026, 3, 16)},
]


def get_roll_schedule() -> list[tuple[date, str, str]]:
    """Return [(roll_date, old_yyyymm, new_yyyymm), ...] sorted chronologically."""
    rolls = []
    for i in range(len(CONTRACTS) - 1):
        old = CONTRACTS[i]
        new = CONTRACTS[i + 1]
        rolls.append((old["roll_date"], old["yyyymm"], new["yyyymm"]))
    return rolls


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

async def connect_ib(host: str, port: int, client_id: int) -> "IB":
    from ib_async import IB
    ib = IB()
    for attempt in range(1, 4):
        try:
            await ib.connectAsync(host, port, clientId=client_id,
                                  timeout=CONNECT_TIMEOUT)
            logger.info("Connected to %s:%d (clientId=%d)", host, port, client_id)
            return ib
        except Exception as e:
            logger.warning("Connect attempt %d/3 failed: %s", attempt, e)
            try:
                ib.disconnect()
            except Exception:
                pass
            await asyncio.sleep(5)
    raise RuntimeError(f"Could not connect to IBKR at {host}:{port}")


async def ensure_connected(ib, host: str, port: int, client_id: int):
    if ib.isConnected():
        return ib
    logger.info("Reconnecting...")
    try:
        ib.disconnect()
    except Exception:
        pass
    await asyncio.sleep(5)
    from ib_async import IB
    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id,
                          timeout=CONNECT_TIMEOUT)
    logger.info("Reconnected")
    return ib


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def bars_to_df(bars) -> pd.DataFrame:
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
    return df.set_index("time").sort_index()


# ---------------------------------------------------------------------------
# Request with retry + pacing violation handling
# ---------------------------------------------------------------------------

async def request_bars(ib, contract, end_dt_str: str, duration: str,
                       max_retries: int = MAX_RETRIES) -> list:
    """Request daily historical bars with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            if not ib.isConnected():
                raise ConnectionError("Not connected")

            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_dt_str,
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,
                    timeout=0,
                ),
                timeout=120,
            )
            return bars or []

        except Exception as e:
            msg = str(e).lower()
            if "pacing" in msg or "162" in msg:
                logger.warning(
                    "  Pacing violation (attempt %d/%d), waiting %ds...",
                    attempt, max_retries, PACING_VIOLATION_SLEEP,
                )
                await asyncio.sleep(PACING_VIOLATION_SLEEP)
            elif "disconnect" in msg or "not connected" in msg or "winerror" in msg:
                logger.warning("  Disconnected (attempt %d/%d): %s", attempt, max_retries, e)
                raise
            elif "no data" in msg or "no market data" in msg:
                logger.info("  No data available for this period")
                return []
            elif attempt < max_retries:
                wait = attempt * 10
                logger.warning(
                    "  Request error (attempt %d/%d): %s -- waiting %ds",
                    attempt, max_retries, e, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("  Failed after %d attempts: %s", max_retries, e)
                return []
    return []


# ---------------------------------------------------------------------------
# Per-contract download
# ---------------------------------------------------------------------------

async def download_contract_data(
    ib, contract_spec: dict, host: str, port: int, client_id: int,
) -> tuple[pd.DataFrame | None, "IB"]:
    """Download all available daily data for a single quarterly contract.

    Walks backward from expiry in 1Y chunks until 3 consecutive empty
    responses or stale progress.
    """
    from ib_async import Future

    yyyymm = contract_spec["yyyymm"]
    local_sym = contract_spec["local_sym"]

    try:
        ib = await ensure_connected(ib, host, port, client_id)
        fut = Future(symbol="NQ", exchange="CME", tradingClass="NQ",
                     lastTradeDateOrContractMonth=yyyymm,
                     includeExpired=True)
        qualified = await ib.qualifyContractsAsync(fut)
        if not qualified:
            logger.warning("  Could not qualify %s, skipping", local_sym)
            return None, ib
        fut = qualified[0]
        fut.includeExpired = True
        logger.info("  Qualified %s (conId=%d, expiry=%s)",
                    fut.localSymbol, fut.conId, contract_spec["expiry"])
    except Exception as e:
        logger.warning("  Error qualifying %s: %s, skipping", local_sym, e)
        ib = await ensure_connected(ib, host, port, client_id)
        return None, ib

    # Walk backward from expiry
    end_dt = datetime.combine(
        contract_spec["expiry"], datetime.min.time()
    ).replace(tzinfo=timezone.utc)

    all_chunks: list[pd.DataFrame] = []
    empty_streak = 0
    chunk_num = 0
    prev_earliest = None

    while empty_streak < 3:
        chunk_num += 1
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC"

        try:
            ib = await ensure_connected(ib, host, port, client_id)
            bars = await request_bars(ib, fut, end_str, CHUNK_DURATION)
        except Exception as e:
            logger.warning("  %s chunk %d disconnect: %s, reconnecting...",
                           local_sym, chunk_num, e)
            ib = await ensure_connected(ib, host, port, client_id)
            await asyncio.sleep(PACING_DELAY)
            try:
                bars = await request_bars(ib, fut, end_str, CHUNK_DURATION)
            except Exception as e2:
                logger.warning("  %s chunk %d retry failed: %s", local_sym, chunk_num, e2)
                empty_streak += 1
                end_dt -= timedelta(days=60)
                await asyncio.sleep(PACING_DELAY)
                continue

        if bars:
            df = bars_to_df(bars)
            earliest = df.index[0].to_pydatetime()
            if earliest.tzinfo is None:
                earliest = earliest.replace(tzinfo=timezone.utc)

            # Stale progress detection
            if prev_earliest is not None and earliest >= prev_earliest:
                logger.info("  %s chunk %d: stale (earliest=%s), stopping",
                            local_sym, chunk_num, earliest.strftime("%Y-%m-%d"))
                break

            all_chunks.append(df)
            empty_streak = 0
            prev_earliest = earliest
            end_dt = earliest - timedelta(seconds=1)

            total = sum(len(d) for d in all_chunks)
            logger.info("  %s chunk %d: +%d bars, back to %s (%d total)",
                        local_sym, chunk_num, len(df),
                        earliest.strftime("%Y-%m-%d"), total)
        else:
            empty_streak += 1
            end_dt -= timedelta(days=30)
            logger.info("  %s chunk %d: empty, stepping back", local_sym, chunk_num)

        await asyncio.sleep(PACING_DELAY)

    if not all_chunks:
        logger.warning("  %s: no data collected", local_sym)
        return None, ib

    combined = pd.concat(all_chunks)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    logger.info("  %s: %d daily bars (%s -> %s)",
                local_sym, len(combined),
                combined.index[0].strftime("%Y-%m-%d"),
                combined.index[-1].strftime("%Y-%m-%d"))
    return combined, ib


# ---------------------------------------------------------------------------
# Panama stitching
# ---------------------------------------------------------------------------

def round_to_tick(value: float, tick: float = 0.25) -> float:
    """Round a price adjustment to the nearest NQ tick size."""
    return round(value / tick) * tick


def stitch_and_adjust(
    contract_data: dict[str, pd.DataFrame],
    roll_schedule: list[tuple[date, str, str]],
) -> pd.DataFrame:
    """Stitch individual contract DataFrames with Panama backward adjustment.

    Uses the Monday liquidity roll dates (not 3rd Friday expiry).
    Adjustments are rounded to NQ tick size (0.25).

    Args:
        contract_data: {yyyymm: DataFrame} for each downloaded contract.
        roll_schedule: [(roll_date, old_yyyymm, new_yyyymm), ...] chronological.

    Returns:
        Single stitched DataFrame with Panama-adjusted OHLC prices.
    """
    if not contract_data:
        raise ValueError("No contract data to stitch")

    # Filter to rolls where both contracts have data
    valid_rolls = [
        (rd, old, new) for rd, old, new in roll_schedule
        if old in contract_data and new in contract_data
    ]

    if not valid_rolls:
        logger.warning("No valid roll pairs found, returning single contract data")
        dfs = list(contract_data.values())
        return pd.concat(dfs).sort_index()

    valid_rolls.sort(key=lambda x: x[0])

    # Build ordered list of yyyymm codes involved
    all_yyyymm = []
    for rd, old, new in valid_rolls:
        if old not in all_yyyymm:
            all_yyyymm.append(old)
        if new not in all_yyyymm:
            all_yyyymm.append(new)

    # --- Phase A: Segment assignment ---
    segments: list[tuple[str, pd.DataFrame]] = []

    for yyyymm in all_yyyymm:
        if yyyymm not in contract_data:
            continue
        df = contract_data[yyyymm]

        # Upper bound: roll date where this contract is OLD
        upper = None
        for rd, old, new in valid_rolls:
            if old == yyyymm:
                upper = pd.Timestamp(datetime.combine(rd, datetime.min.time()), tz="UTC")
                break

        # Lower bound: roll date where this contract is NEW
        lower = None
        for rd, old, new in valid_rolls:
            if new == yyyymm:
                lower = pd.Timestamp(datetime.combine(rd, datetime.min.time()), tz="UTC")
                break

        # Slice to owned window
        if lower is not None and upper is not None:
            seg = df[(df.index >= lower) & (df.index < upper)]
        elif lower is not None:
            seg = df[df.index >= lower]
        elif upper is not None:
            seg = df[df.index < upper]
        else:
            seg = df

        if len(seg) > 0:
            segments.append((yyyymm, seg))
            logger.info("  Segment %s: %d bars (%s -> %s)",
                        yyyymm, len(seg),
                        seg.index[0].strftime("%Y-%m-%d"),
                        seg.index[-1].strftime("%Y-%m-%d"))

    # --- Phase B: Compute Panama gaps (newest to oldest) ---
    adjustments: dict[str, float] = {}
    cumulative_adjustment = 0.0

    # Newest contract gets 0 adjustment
    if all_yyyymm:
        adjustments[all_yyyymm[-1]] = 0.0

    for rd, old_yyyymm, new_yyyymm in reversed(valid_rolls):
        roll_ts = pd.Timestamp(datetime.combine(rd, datetime.min.time()), tz="UTC")

        old_df = contract_data.get(old_yyyymm)
        new_df = contract_data.get(new_yyyymm)

        if old_df is None or new_df is None:
            logger.warning("  Roll %s: missing data for %s or %s, gap=0",
                           rd, old_yyyymm, new_yyyymm)
            adjustments[old_yyyymm] = cumulative_adjustment
            continue

        # Last bar of OLD contract before roll date
        old_before_roll = old_df[old_df.index < roll_ts]
        # First bar of NEW contract on or after roll date
        new_after_roll = new_df[new_df.index >= roll_ts]

        if len(old_before_roll) == 0 or len(new_after_roll) == 0:
            logger.warning("  Roll %s: no bars at boundary for %s/%s, gap=0",
                           rd, old_yyyymm, new_yyyymm)
            adjustments[old_yyyymm] = cumulative_adjustment
            continue

        old_close = old_before_roll.iloc[-1]["close"]
        new_open = new_after_roll.iloc[0]["open"]
        gap = round_to_tick(new_open - old_close)
        cumulative_adjustment += gap

        adjustments[old_yyyymm] = cumulative_adjustment
        logger.info("  Roll %s: %s->%s, old_close=%.2f, new_open=%.2f, "
                    "gap=%.2f, cumulative=%.2f",
                    rd, old_yyyymm, new_yyyymm, old_close, new_open,
                    gap, cumulative_adjustment)

    # --- Phase C: Apply adjustments and concatenate ---
    adjusted_segments: list[pd.DataFrame] = []

    for yyyymm, seg in segments:
        adj = adjustments.get(yyyymm, 0.0)
        if adj != 0.0:
            seg = seg.copy()
            seg["open"] = seg["open"] - adj
            seg["high"] = seg["high"] - adj
            seg["low"] = seg["low"] - adj
            seg["close"] = seg["close"] - adj
        adjusted_segments.append(seg)

    result = pd.concat(adjusted_segments)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def print_analysis(df: pd.DataFrame, name: str):
    """Print summary of the stitched Panama-adjusted daily data."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total bars:  {len(df):>10,}")
    print(f"  Start:       {df.index[0]}")
    print(f"  End:         {df.index[-1]}")
    span = df.index[-1] - df.index[0]
    print(f"  Span:        {span.days} days ({span.days/365:.2f} years)")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    if df["close"].min() < 0:
        neg_bars = (df["close"] < 0).sum()
        print(f"  WARNING: {neg_bars} bars with negative close prices (over-adjustment)")

    # Monthly bar counts
    monthly = df.resample("ME").size()
    print(f"\n  Monthly coverage (~22 trading days/month):")
    for dt, count in monthly.items():
        pct = count / 22 * 100
        status = "OK" if pct > 80 else "LOW" if pct > 40 else "GAP"
        print(f"    {dt.strftime('%Y-%m')}: {count:>4} bars ({pct:5.1f}%)  {status}")

    # Gaps > 5 calendar days (possible missing weeks)
    diffs = df.index.to_series().diff().dropna()
    real_gaps = diffs[diffs > pd.Timedelta(days=5)]
    print(f"\n  Gaps > 5 days: {len(real_gaps)}")
    if len(real_gaps) > 0:
        for ts, gap in real_gaps.head(10).items():
            print(f"    {ts.strftime('%Y-%m-%d')} -- gap of {gap}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Download NQ daily bars (quarterly contracts) with Panama adjustment"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="IBKR host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7496,
                        help="IBKR port (default: 7496 for TWS)")
    parser.add_argument("--client-id", type=int, default=106,
                        help="IBKR client ID (default: 106)")
    args = parser.parse_args()

    roll_schedule = get_roll_schedule()

    logger.info("Contracts to download: %d", len(CONTRACTS))
    for c in CONTRACTS:
        logger.info("  %s (%s) expiry=%s roll=%s",
                    c["local_sym"], c["yyyymm"], c["expiry"], c["roll_date"])
    logger.info("Roll schedule:")
    for rd, old, new in roll_schedule:
        logger.info("  %s: %s -> %s", rd, old, new)

    # Connect
    ib = await connect_ib(args.host, args.port, args.client_id)

    try:
        logger.info("=" * 60)
        logger.info("Downloading daily data for %d contracts", len(CONTRACTS))
        logger.info("=" * 60)

        contract_data: dict[str, pd.DataFrame] = {}

        for cinfo in CONTRACTS:
            logger.info("Downloading %s...", cinfo["local_sym"])
            df, ib = await download_contract_data(
                ib, cinfo, args.host, args.port, args.client_id,
            )
            if df is not None and len(df) > 0:
                contract_data[cinfo["yyyymm"]] = df
            await asyncio.sleep(PACING_DELAY)

        logger.info("Downloaded %d/%d contracts with data",
                    len(contract_data), len(CONTRACTS))

        if not contract_data:
            logger.error("No data downloaded, exiting")
            return

        # Stitch and adjust
        logger.info("Stitching with Panama adjustment (Monday liquidity rolls)...")
        stitched = stitch_and_adjust(contract_data, roll_schedule)

        # Save parquet
        parquet_path = DATA_DIR / "NQ_daily_panama.parquet"
        stitched.to_parquet(parquet_path, engine="pyarrow")
        logger.info("Saved %s: %d bars", parquet_path, len(stitched))

        # Save CSV with Adjusted_Close column (per spec)
        csv_df = stitched.copy()
        csv_df.index.name = "Date"
        csv_df = csv_df.rename(columns={"close": "Adjusted_Close"})
        csv_path = DATA_DIR / "NQ_daily_panama.csv"
        csv_df.to_csv(csv_path)
        logger.info("Saved %s: %d rows", csv_path, len(csv_df))

        # Analysis
        print_analysis(stitched, "NQ_daily_panama")

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
