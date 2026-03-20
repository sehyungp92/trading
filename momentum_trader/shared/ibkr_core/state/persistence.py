"""Persistence interface for restart-safe state."""
import json
from datetime import datetime
from typing import Optional, Protocol


class IBPersistence(Protocol):
    """Minimal persistence interface for restart-safe state."""

    async def save_next_valid_id(self, next_id: int) -> None: ...
    async def load_next_valid_id(self) -> Optional[int]: ...
    async def save_con_id(self, symbol: str, expiry: str, con_id: int) -> None: ...
    async def load_con_ids(self) -> dict[tuple[str, str], int]: ...
    async def save_recon_watermark(self, last_exec_time: datetime) -> None: ...
    async def load_recon_watermark(self) -> Optional[datetime]: ...


class PostgresIBPersistence:
    """Postgres implementation of IBPersistence.

    Table: ib_state (key TEXT PRIMARY KEY, value JSONB, updated_at TIMESTAMPTZ)
    """

    def __init__(self, pool):
        self._pool = pool

    async def save_next_valid_id(self, next_id: int) -> None:
        await self._upsert("next_valid_id", {"value": next_id})

    async def load_next_valid_id(self) -> Optional[int]:
        row = await self._get("next_valid_id")
        return row["value"] if row else None

    async def save_con_id(self, symbol: str, expiry: str, con_id: int) -> None:
        key = f"con_id:{symbol}:{expiry}"
        await self._upsert(key, {"con_id": con_id})

    async def load_con_ids(self) -> dict[tuple[str, str], int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM ib_state WHERE key LIKE 'con_id:%'"
            )
        result = {}
        for row in rows:
            parts = row["key"].split(":")
            if len(parts) == 3:
                symbol, expiry = parts[1], parts[2]
                data = json.loads(row["value"])
                result[(symbol, expiry)] = data["con_id"]
        return result

    async def save_recon_watermark(self, last_exec_time: datetime) -> None:
        await self._upsert("recon_watermark", {"ts": last_exec_time.isoformat()})

    async def load_recon_watermark(self) -> Optional[datetime]:
        row = await self._get("recon_watermark")
        if row and "ts" in row:
            return datetime.fromisoformat(row["ts"])
        return None

    async def _upsert(self, key: str, value: dict) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ib_state (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
                """,
                key,
                json.dumps(value),
            )

    async def _get(self, key: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT value FROM ib_state WHERE key = $1", key)
        if row:
            return json.loads(row["value"])
        return None
