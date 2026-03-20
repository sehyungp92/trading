"""Paper trading dynamic equity tracker.

Maintains a single-row DB table that tracks simulated equity for paper mode.
Three async functions (no class) — matches project's pure-function style.

Only used when ALGO_TRADER_ENV=paper. Zero impact on live/backtest.
"""
from __future__ import annotations

import logging
import os

import asyncpg

logger = logging.getLogger(__name__)

_DEFAULT_INITIAL_EQUITY = 10_000.0


def _get_initial_equity() -> float:
    """Read initial equity from env, falling back to default."""
    raw = os.getenv("PAPER_INITIAL_EQUITY", "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid PAPER_INITIAL_EQUITY=%r, using default", raw)
    return _DEFAULT_INITIAL_EQUITY


async def load_paper_equity(pool: asyncpg.Pool, initial: float | None = None) -> float:
    """Load current paper equity from DB. Seed row if missing.

    Args:
        pool: asyncpg connection pool
        initial: Override initial equity (defaults to env/10k)

    Returns:
        Current equity value
    """
    if initial is None:
        initial = _get_initial_equity()

    row = await pool.fetchrow("SELECT equity FROM paper_equity WHERE id = 1")
    if row is not None:
        equity = float(row["equity"])
        logger.info("Loaded paper equity: $%.2f", equity)
        return equity

    # Seed the singleton row
    await pool.execute(
        """
        INSERT INTO paper_equity (id, equity, initial_equity)
        VALUES (1, $1, $1)
        ON CONFLICT (id) DO NOTHING
        """,
        initial,
    )
    logger.info("Seeded paper equity: $%.2f", initial)
    return initial


async def apply_paper_pnl(
    pool: asyncpg.Pool, pnl: float, commission: float = 0.0,
) -> float:
    """Atomically apply realized P&L and commission to paper equity.

    Args:
        pool: asyncpg connection pool
        pnl: Realized P&L (positive = profit)
        commission: Commission to deduct (always positive)

    Returns:
        New equity after update
    """
    row = await pool.fetchrow(
        """
        UPDATE paper_equity
        SET equity = equity + $1 - $2,
            total_pnl = total_pnl + $1,
            total_commission = total_commission + $2,
            trade_count = trade_count + CASE WHEN $1 != 0 THEN 1 ELSE 0 END,
            last_update_at = now()
        WHERE id = 1
        RETURNING equity
        """,
        pnl,
        commission,
    )
    if row is None:
        raise RuntimeError("paper_equity row missing — call load_paper_equity first")
    return float(row["equity"])


async def reset_paper_equity(
    pool: asyncpg.Pool, new_equity: float | None = None,
) -> None:
    """Reset paper equity to a fresh start.

    Args:
        pool: asyncpg connection pool
        new_equity: New equity value (defaults to env/10k)
    """
    if new_equity is None:
        new_equity = _get_initial_equity()

    await pool.execute(
        """
        UPDATE paper_equity
        SET equity = $1,
            initial_equity = $1,
            total_pnl = 0.0,
            total_commission = 0.0,
            trade_count = 0,
            last_update_at = now()
        WHERE id = 1
        """,
        new_equity,
    )
    logger.info("Paper equity reset to $%.2f", new_equity)
