"""Generate IARIC/ALCB artifacts using a temporary IB connection."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator

from ib_async import IB

logger = logging.getLogger(__name__)

RESEARCH_CLIENT_ID = int(os.environ.get("STOCK_RESEARCH_CLIENT_ID", "35"))


@asynccontextmanager
async def _research_ib(
    host: str = "127.0.0.1",
    port: int = 4002,
) -> AsyncIterator[IB]:
    """Temporary IB connection for research data fetching."""
    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=RESEARCH_CLIENT_ID, timeout=60)
        logger.info("Research IB connected (client_id=%d)", RESEARCH_CLIENT_ID)
        yield ib
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Research IB disconnected")


async def generate_iaric(trade_date: date, ib: IB) -> bool:
    """Run IARIC research + selection. Returns True on success."""
    from strategies.stock.iaric.research_generator import generate_research_snapshot
    from strategies.stock.iaric.research import run_daily_selection

    try:
        await generate_research_snapshot(trade_date, ib)
        run_daily_selection(trade_date)
        logger.info("IARIC artifact generated for %s", trade_date)
        return True
    except Exception:
        logger.error("IARIC artifact generation failed", exc_info=True)
        return False


async def generate_alcb(trade_date: date, ib: IB) -> bool:
    """Run ALCB research + selection. Returns True on success."""
    from strategies.stock.alcb.research_generator import generate_research_snapshot
    from strategies.stock.alcb.research import run_daily_selection

    try:
        await generate_research_snapshot(trade_date, ib=ib)
        run_daily_selection(trade_date)
        logger.info("ALCB artifact generated for %s", trade_date)
        return True
    except Exception:
        logger.error("ALCB artifact generation failed", exc_info=True)
        return False


async def ensure_artifacts(
    trade_date: date,
    missing: list[str],
    host: str = "127.0.0.1",
    port: int = 4002,
) -> dict[str, bool]:
    """Generate artifacts for missing strategies. Returns {sid: success}."""
    results: dict[str, bool] = {}
    async with _research_ib(host=host, port=port) as ib:
        if "IARIC_v1" in missing:
            results["IARIC_v1"] = await generate_iaric(trade_date, ib)
        if "ALCB_v1" in missing:
            results["ALCB_v1"] = await generate_alcb(trade_date, ib)
    return results
