"""Cross-family account-level risk gate using PostgreSQL advisory locks.

Serializes entry approval across all families (swing, momentum, stock)
to enforce account-global heat cap and daily stop limits.

Operates in DOLLARS internally to avoid mixing R-unit bases across
families with different unit_risk_dollars.  R-based caps are converted
via ``account_urd`` (the dollar value of one "account R").

Usage:
    gate = AccountRiskGate(pool, heat_cap_R=2.5, daily_stop_R=3.0, account_urd=200.0)
    approved, reason = await gate.check_entry("momentum", risk_dollars=120.0)
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

# Stable advisory lock key derived from a fixed string
_LOCK_KEY = int.from_bytes(hashlib.sha256(b"account_entry_gate").digest()[:8], "big", signed=True)


@dataclass(frozen=True)
class EntryDecision:
    approved: bool
    reason: Optional[str] = None


class AccountRiskGate:
    """Atomic cross-family entry approval using PostgreSQL advisory locks.

    On check_entry:
    1. Acquires pg_advisory_xact_lock (released on COMMIT/ROLLBACK)
    2. Reads aggregate open risk DOLLARS from positions table
    3. Reads aggregate daily realized DOLLARS from risk_daily_portfolio
    4. Converts to account-R via account_urd for comparison against caps
    5. Returns approval decision
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        heat_cap_R: float = 2.5,
        daily_stop_R: float = 3.0,
        account_urd: float = 200.0,
    ) -> None:
        self._pool = pool
        self._heat_cap_R = heat_cap_R
        self._daily_stop_R = daily_stop_R
        self._account_urd = account_urd

    async def check_entry(
        self,
        family_id: str,
        risk_dollars: float,
        conn: Any = None,
    ) -> EntryDecision:
        """Check if an entry is allowed at the account level.

        Args:
            family_id: The family requesting entry ('swing', 'momentum', 'stock').
            risk_dollars: The risk in DOLLARS for the proposed entry.

        Returns:
            EntryDecision with approved=True if allowed, else reason for denial.
        """
        urd = self._account_urd
        if urd <= 0:
            logger.error("AccountRiskGate: account_urd must be > 0, denying entry")
            return EntryDecision(approved=False, reason="Invalid account_urd configuration")
        if not math.isfinite(risk_dollars) or risk_dollars <= 0:
            logger.error(
                "AccountRiskGate: proposed risk must be positive, denying entry: %s",
                risk_dollars,
            )
            return EntryDecision(approved=False, reason="Invalid proposed risk dollars")

        try:
            if conn is not None:
                return await self._check_entry_on_conn(conn, family_id, risk_dollars, urd)
            async with self._pool.acquire() as acquired:
                async with acquired.transaction():
                    return await self._check_entry_on_conn(acquired, family_id, risk_dollars, urd)
        except Exception:
            logger.exception("AccountRiskGate DB error [%s] -- denying entry as safety fallback", family_id)
            return EntryDecision(approved=False, reason="Account gate unavailable (DB error)")

    async def _check_entry_on_conn(
        self,
        conn: Any,
        family_id: str,
        risk_dollars: float,
        urd: float,
    ) -> EntryDecision:
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _LOCK_KEY)

        pos_row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(GREATEST(open_risk_dollars, 0)), 0) AS total_open_risk_dollars
            FROM positions
            WHERE net_qty != 0
            """
        )
        total_open_risk_dollars = float(pos_row["total_open_risk_dollars"])

        pending_row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(
                CASE
                    WHEN status = 'PARTIALLY_FILLED' AND qty > 0
                        THEN GREATEST(COALESCE((risk_context->>'risk_dollars')::numeric, 0), 0)
                             * GREATEST(remaining_qty, 0) / qty
                    ELSE GREATEST(COALESCE((risk_context->>'risk_dollars')::numeric, 0), 0)
                END
            ), 0) AS pending_entry_risk_dollars
            FROM orders
            WHERE role = 'ENTRY'
              AND status IN ('RISK_APPROVED', 'ROUTED', 'ACKED', 'WORKING', 'PARTIALLY_FILLED')
              AND risk_context IS NOT NULL
            """
        )
        pending_entry_risk_dollars = float(pending_row["pending_entry_risk_dollars"])

        pnl_row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(daily_realized_usd), 0) AS total_daily_realized_dollars
            FROM risk_daily_portfolio
            WHERE trade_date = (now() AT TIME ZONE 'America/New_York')::date
            """
        )
        total_daily_realized_dollars = float(pnl_row["total_daily_realized_dollars"])

        total_reserved_risk_dollars = total_open_risk_dollars + pending_entry_risk_dollars
        total_reserved_risk_R = total_reserved_risk_dollars / urd
        total_daily_realized_R = total_daily_realized_dollars / urd
        new_risk_R = risk_dollars / urd

        if total_daily_realized_R <= -self._daily_stop_R:
            reason = (
                f"Account daily stop: realized {total_daily_realized_R:.2f}R "
                f"(${total_daily_realized_dollars:,.0f}) <= -{self._daily_stop_R}R"
            )
            logger.warning("AccountRiskGate DENIED [%s]: %s", family_id, reason)
            return EntryDecision(approved=False, reason=reason)

        projected_risk_R = total_reserved_risk_R + new_risk_R
        if projected_risk_R > self._heat_cap_R:
            reason = (
                f"Account heat cap: reserved {total_reserved_risk_R:.2f}R + "
                f"new {new_risk_R:.2f}R = {projected_risk_R:.2f}R > "
                f"cap {self._heat_cap_R}R "
                f"(${total_reserved_risk_dollars:,.0f} + ${risk_dollars:,.0f})"
            )
            logger.warning("AccountRiskGate DENIED [%s]: %s", family_id, reason)
            return EntryDecision(approved=False, reason=reason)

        logger.info(
            "AccountRiskGate APPROVED [%s]: +%.2fR/$%.0f "
            "(reserved=%.2fR/$%.0f, realized=%.2fR/$%.0f)",
            family_id, new_risk_R, risk_dollars,
            total_reserved_risk_R, total_reserved_risk_dollars,
            total_daily_realized_R, total_daily_realized_dollars,
        )
        return EntryDecision(approved=True)
