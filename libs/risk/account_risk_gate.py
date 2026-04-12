"""Cross-family account-level risk gate using PostgreSQL advisory locks.

Serializes entry approval across all families (swing, momentum, stock)
to enforce account-global heat cap and daily stop limits.

Usage:
    gate = AccountRiskGate(pool, heat_cap_R=2.5, daily_stop_R=3.0)
    approved, reason = await gate.check_entry("momentum", risk_r=0.5)
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

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
    2. Reads aggregate open risk and daily realized across ALL families
    3. Checks against account-global heat_cap_R and daily_stop_R
    4. Returns approval decision
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        heat_cap_R: float = 2.5,
        daily_stop_R: float = 3.0,
    ) -> None:
        self._pool = pool
        self._heat_cap_R = heat_cap_R
        self._daily_stop_R = daily_stop_R

    async def check_entry(
        self,
        family_id: str,
        risk_r: float,
    ) -> EntryDecision:
        """Check if an entry is allowed at the account level.

        Args:
            family_id: The family requesting entry ('swing', 'momentum', 'stock').
            risk_r: The risk in R-units for the proposed entry.

        Returns:
            EntryDecision with approved=True if allowed, else reason for denial.
        """
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Acquire advisory lock (auto-released on transaction end)
                    await conn.execute("SELECT pg_advisory_xact_lock($1)", _LOCK_KEY)

                    # 2. Read aggregate open risk across ALL families
                    # Use explicit ET trade date — not CURRENT_DATE which depends on DB timezone setting
                    row = await conn.fetchrow(
                        """
                        SELECT
                            COALESCE(SUM(portfolio_open_risk_r), 0) AS total_open_risk_r,
                            COALESCE(SUM(daily_realized_r), 0) AS total_daily_realized_r
                        FROM risk_daily_portfolio
                        WHERE trade_date = (now() AT TIME ZONE 'America/New_York')::date
                        """
                    )
                    total_open_risk = float(row["total_open_risk_r"])
                    total_daily_realized = float(row["total_daily_realized_r"])

                    # 3. Check account-level daily stop
                    if total_daily_realized <= -self._daily_stop_R:
                        reason = (
                            f"Account daily stop: realized {total_daily_realized:.2f}R "
                            f"<= -{self._daily_stop_R}R"
                        )
                        logger.warning("AccountRiskGate DENIED [%s]: %s", family_id, reason)
                        return EntryDecision(approved=False, reason=reason)

                    # 4. Check account-level heat cap
                    projected_risk = total_open_risk + risk_r
                    if projected_risk > self._heat_cap_R:
                        reason = (
                            f"Account heat cap: open {total_open_risk:.2f}R + "
                            f"new {risk_r:.2f}R = {projected_risk:.2f}R > "
                            f"cap {self._heat_cap_R}R"
                        )
                        logger.warning("AccountRiskGate DENIED [%s]: %s", family_id, reason)
                        return EntryDecision(approved=False, reason=reason)

                    logger.info(
                        "AccountRiskGate APPROVED [%s]: +%.2fR (open=%.2fR, realized=%.2fR)",
                        family_id, risk_r, total_open_risk, total_daily_realized,
                    )
                    return EntryDecision(approved=True)
        except Exception:
            logger.exception("AccountRiskGate DB error [%s] — denying entry as safety fallback", family_id)
            return EntryDecision(approved=False, reason="Account gate unavailable (DB error)")
