"""Download individual NQ quarterly contracts and stitch with Panama adjustment.

Unlike download_nq_2y.py which uses IBKR's ContFuture (back-adjusted by IBKR),
this script downloads each quarterly contract separately, stitches them on
3rd-Friday roll dates, and applies a backward Panama price adjustment — giving
full control over roll logic and adjustment method.

Output: NQ_5m_panama.parquet, NQ_15m_panama.parquet in backtest/data/raw/

Usage:
    python scripts/download_nq_panama.py [--5m-only] [--15m-only]
                                          [--host HOST] [--port PORT]
                                          [--client-id CID]

Requires: IB Gateway or TWS running and accepting API connections.
"""
import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_nq_panama.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("backtest/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PACING_DELAY = 12  # seconds between historical data requests
PACING_VIOLATION_SLEEP = 65  # seconds after error 162
MAX_RETRIES = 4
CONNECT_TIMEOUT = 180
CHUNK_DURATION = "6 M"  # download chunks per request
NUM_CONTRACTS = 8  # ~2 years of quarterly contracts

BAR_SIZES = {
    "5m": "5 mins",
    "15m": "15 mins",
}


# ---------------------------------------------------------------------------
# Contract generation
# ---------------------------------------------------------------------------

def generate_nq_contracts(n: int = NUM_CONTRACTS) -> list[dict]:
    """Generate the N most recent NQ quarterly contract specs.

    Returns list of dicts with keys: yyyymm, expiry, code, local_sym,
    sorted newest-first.
    """
    today = date.today()
    contracts = []
    quarters = [("03", "H"), ("06", "M"), ("09", "U"), ("12", "Z")]

    for y in range(today.year + 1, today.year - 4, -1):
        for month_str, code in reversed(quarters):
            month = int(month_str)
            # 3rd Friday of the delivery month
            first_day = date(y, month, 1)
            days_to_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_friday)
            third_friday = first_friday + timedelta(weeks=2)

            # Skip contracts >90 days in the future
            if third_friday > today + timedelta(days=90):
                continue

            contracts.append({
                "yyyymm": f"{y}{month_str}",
                "expiry": third_friday,
                "roll_date": third_friday - timedelta(days=4),  # Monday of expiry week
                "code": code,
                "local_sym": f"NQ{code}{y % 10}",
                "year": y,
            })

    contracts.sort(key=lambda x: x["expiry"], reverse=True)
    return contracts[:n]


# ---------------------------------------------------------------------------
# Roll date computation
# ---------------------------------------------------------------------------

def compute_roll_dates(contracts: list[dict]) -> list[tuple[date, str, str]]:
    """Compute roll dates from contract list.

    Returns [(roll_date, old_yyyymm, new_yyyymm), ...] sorted chronologically.
    Roll date = Monday of old contract's expiry week (liquidity roll),
    matching the daily script's convention.
    """
    # Sort by expiry ascending for pairing
    sorted_contracts = sorted(contracts, key=lambda c: c["expiry"])
    roll_dates = []

    for i in range(len(sorted_contracts) - 1):
        old = sorted_contracts[i]
        new = sorted_contracts[i + 1]
        roll_dates.append((old["roll_date"], old["yyyymm"], new["yyyymm"]))

    return roll_dates


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
                       bar_size: str, max_retries: int = MAX_RETRIES) -> list:
    """Request historical bars with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            if not ib.isConnected():
                raise ConnectionError("Not connected")

            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_dt_str,
                    durationStr=duration,
                    barSizeSetting=bar_size,
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
                raise  # Let caller handle reconnection
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
    ib, contract_spec: dict, bar_size: str, host: str, port: int, client_id: int,
) -> tuple[pd.DataFrame | None, "IB"]:
    """Download all available data for a single quarterly contract.

    Walks backward from expiry in 6M chunks until 3 consecutive empty
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
            bars = await request_bars(ib, fut, end_str, CHUNK_DURATION, bar_size)
        except Exception as e:
            logger.warning("  %s chunk %d disconnect: %s, reconnecting...",
                           local_sym, chunk_num, e)
            ib = await ensure_connected(ib, host, port, client_id)
            await asyncio.sleep(PACING_DELAY)
            try:
                bars = await request_bars(ib, fut, end_str, CHUNK_DURATION, bar_size)
            except Exception as e2:
                logger.warning("  %s chunk %d retry failed: %s", local_sym, chunk_num, e2)
                empty_streak += 1
                end_dt -= timedelta(days=20)
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

            if chunk_num % 3 == 0 or chunk_num == 1:
                total = sum(len(d) for d in all_chunks)
                logger.info("  %s chunk %d: +%d bars, back to %s (%d total)",
                            local_sym, chunk_num, len(df),
                            earliest.strftime("%Y-%m-%d"), total)
        else:
            empty_streak += 1
            end_dt -= timedelta(days=10)
            logger.info("  %s chunk %d: empty, stepping back", local_sym, chunk_num)

        await asyncio.sleep(PACING_DELAY)

    if not all_chunks:
        logger.warning("  %s: no data collected", local_sym)
        return None, ib

    combined = pd.concat(all_chunks)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    logger.info("  %s: %d bars total (%s -> %s)",
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
    roll_dates: list[tuple[date, str, str]],
) -> pd.DataFrame:
    """Stitch individual contract DataFrames with Panama backward adjustment.

    Args:
        contract_data: {yyyymm: DataFrame} for each downloaded contract.
        roll_dates: [(roll_date, old_yyyymm, new_yyyymm), ...] chronological.

    Returns:
        Single stitched DataFrame with Panama-adjusted OHLC prices.
    """
    if not contract_data:
        raise ValueError("No contract data to stitch")

    # Filter roll_dates to only those where both contracts have data
    valid_rolls = [
        (rd, old, new) for rd, old, new in roll_dates
        if old in contract_data and new in contract_data
    ]

    if not valid_rolls:
        # Only one contract has data — return it as-is
        logger.warning("No valid roll pairs found, returning single contract data")
        dfs = list(contract_data.values())
        return pd.concat(dfs).sort_index()

    # Sort roll dates chronologically
    valid_rolls.sort(key=lambda x: x[0])

    # All yyyymm codes involved, sorted by expiry (chronological)
    all_yyyymm = []
    for rd, old, new in valid_rolls:
        if old not in all_yyyymm:
            all_yyyymm.append(old)
        if new not in all_yyyymm:
            all_yyyymm.append(new)

    # --- Phase A: Segment assignment ---
    # Each contract owns a time window:
    #   oldest contract: (-inf, first_roll_date)
    #   middle contracts: [prev_roll, next_roll)
    #   newest contract: [last_roll_date, +inf)
    segments: list[tuple[str, pd.DataFrame]] = []

    for i, yyyymm in enumerate(all_yyyymm):
        if yyyymm not in contract_data:
            continue
        df = contract_data[yyyymm]

        # Determine time window for this contract
        # Find the roll date where this contract is the OLD (upper bound)
        upper = None
        for rd, old, new in valid_rolls:
            if old == yyyymm:
                upper = pd.Timestamp(datetime.combine(rd, datetime.min.time()),
                                     tz="UTC")
                break

        # Find the roll date where this contract is the NEW (lower bound)
        lower = None
        for rd, old, new in valid_rolls:
            if new == yyyymm:
                lower = pd.Timestamp(datetime.combine(rd, datetime.min.time()),
                                     tz="UTC")
                break

        # Slice
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

    # Walk roll dates in reverse chronological order
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
# RTH flag
# ---------------------------------------------------------------------------

def add_rth_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_RTH column: True for 09:30-15:59 ET on weekdays."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    df["is_RTH"] = (minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def print_analysis(df: pd.DataFrame, name: str):
    """Print summary of the stitched Panama-adjusted data."""
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

    if "is_RTH" in df.columns:
        rth_count = df["is_RTH"].sum()
        eth_count = len(df) - rth_count
        print(f"  RTH bars:    {rth_count:>10,}")
        print(f"  ETH bars:    {eth_count:>10,}")

    # Large gaps
    diffs = df.index.to_series().diff().dropna()
    real_gaps = diffs[diffs > pd.Timedelta(hours=72)]
    print(f"  Gaps > 72h:  {len(real_gaps)}")
    if len(real_gaps) > 0:
        for ts, gap in real_gaps.head(10).items():
            print(f"    {ts.strftime('%Y-%m-%d %H:%M')} -- gap of {gap}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Download NQ quarterly contracts and stitch with Panama adjustment"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="IBKR host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7496,
                        help="IBKR port (default: 7496 for TWS)")
    parser.add_argument("--client-id", type=int, default=105,
                        help="IBKR client ID (default: 105)")
    parser.add_argument("--5m-only", action="store_true",
                        help="Only download 5-minute data")
    parser.add_argument("--15m-only", action="store_true",
                        help="Only download 15-minute data")
    args = parser.parse_args()

    # Determine timeframes
    timeframes = []
    if args.__dict__.get("5m_only"):
        timeframes = ["5m"]
    elif args.__dict__.get("15m_only"):
        timeframes = ["15m"]
    else:
        timeframes = ["5m", "15m"]

    # Generate contracts and roll dates
    contracts = generate_nq_contracts()
    roll_dates = compute_roll_dates(contracts)

    logger.info("Generated %d contracts:", len(contracts))
    for c in contracts:
        logger.info("  %s (expiry %s)", c["local_sym"], c["expiry"])
    logger.info("Roll dates:")
    for rd, old, new in roll_dates:
        logger.info("  %s: %s -> %s", rd, old, new)

    # Connect
    ib = await connect_ib(args.host, args.port, args.client_id)

    try:
        for tf in timeframes:
            bar_size = BAR_SIZES[tf]
            name = f"NQ_{tf}_panama"

            logger.info("=" * 60)
            logger.info("Downloading %s data for %d contracts", tf, len(contracts))
            logger.info("=" * 60)

            # Download all contracts
            contract_data: dict[str, pd.DataFrame] = {}

            for cinfo in contracts:
                logger.info("Downloading %s (%s)...", cinfo["local_sym"], tf)
                df, ib = await download_contract_data(
                    ib, cinfo, bar_size, args.host, args.port, args.client_id,
                )
                if df is not None and len(df) > 0:
                    contract_data[cinfo["yyyymm"]] = df
                await asyncio.sleep(PACING_DELAY)

            logger.info("Downloaded %d/%d contracts with data",
                        len(contract_data), len(contracts))

            if not contract_data:
                logger.error("No data downloaded for %s, skipping", tf)
                continue

            # Stitch and adjust
            logger.info("Stitching with Panama adjustment...")
            stitched = stitch_and_adjust(contract_data, roll_dates)

            # Add RTH flag
            stitched = add_rth_flag(stitched)

            # Save
            out_path = DATA_DIR / f"{name}.parquet"
            stitched.to_parquet(out_path, engine="pyarrow")
            logger.info("Saved %s: %d bars", out_path, len(stitched))

            # Print analysis
            print_analysis(stitched, name)

            if len(timeframes) > 1:
                logger.info("Waiting %ds before next timeframe...", PACING_DELAY * 2)
                await asyncio.sleep(PACING_DELAY * 2)

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
