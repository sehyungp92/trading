"""SQLite event store for the relay service."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class EventStore:
    """Simple SQLite-backed event buffer."""

    def __init__(self, db_path: str = "data/relay.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA_PATH.read_text())
        # Enable WAL mode for concurrent read/write performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # Migration: add priority column to existing DBs
        try:
            conn.execute("ALTER TABLE events ADD COLUMN priority INTEGER NOT NULL DEFAULT 3")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority)")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def insert_events(self, events: list[dict]) -> dict[str, int]:
        """Insert events, skipping duplicates. Returns accepted/duplicate counts."""
        accepted = 0
        duplicates = 0
        conn = self._connect()
        try:
            for event in events:
                try:
                    conn.execute(
                        """INSERT INTO events (event_id, bot_id, event_type, payload, exchange_timestamp, received_at, priority)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            event["event_id"],
                            event["bot_id"],
                            event.get("event_type", "unknown"),
                            event.get("payload", "{}"),
                            event.get("exchange_timestamp", ""),
                            datetime.now(timezone.utc).isoformat(),
                            event.get("priority", 3),
                        ),
                    )
                    accepted += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
            conn.commit()
        finally:
            conn.close()
        return {"accepted": accepted, "duplicates": duplicates}

    def get_events(
        self,
        since: str | None = None,
        limit: int = 100,
        bot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch un-acked events, optionally after a watermark event_id."""
        conn = self._connect()
        try:
            if since:
                # Find the row id for the watermark
                row = conn.execute(
                    "SELECT id FROM events WHERE event_id = ?", (since,)
                ).fetchone()
                min_id = row["id"] if row else 0
            else:
                min_id = 0

            query = "SELECT * FROM events WHERE acked = 0 AND id > ?"
            params: list[Any] = [min_id]

            if bot_id:
                query += " AND bot_id = ?"
                params.append(bot_id)

            query += " ORDER BY priority ASC, id ASC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def ack_up_to(self, watermark_event_id: str) -> int:
        """Mark all events up to and including the watermark as acked."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM events WHERE event_id = ?", (watermark_event_id,)
            ).fetchone()
            if not row:
                return 0
            cursor = conn.execute(
                "UPDATE events SET acked = 1 WHERE id <= ? AND acked = 0",
                (row["id"],),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def count_pending(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM events WHERE acked = 0").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Return enriched health stats: per-bot pending, last event times, oldest age, DB size."""
        conn = self._connect()
        try:
            # Per-bot pending counts
            rows = conn.execute(
                "SELECT bot_id, COUNT(*) as cnt FROM events WHERE acked = 0 GROUP BY bot_id"
            ).fetchall()
            per_bot_pending = {r["bot_id"]: r["cnt"] for r in rows}

            # Last event timestamp per bot (all events, not just pending)
            rows = conn.execute(
                "SELECT bot_id, MAX(received_at) as last_ts FROM events GROUP BY bot_id"
            ).fetchall()
            last_event_per_bot = {r["bot_id"]: r["last_ts"] for r in rows}

            # Oldest pending event age in seconds
            row = conn.execute(
                "SELECT MIN(received_at) as oldest FROM events WHERE acked = 0"
            ).fetchone()
            if row["oldest"]:
                oldest_dt = datetime.fromisoformat(row["oldest"])
                oldest_age = (datetime.now(timezone.utc) - oldest_dt).total_seconds()
            else:
                oldest_age = 0.0

            # DB file size
            try:
                db_size = os.path.getsize(self.db_path)
            except OSError:
                db_size = 0

            return {
                "per_bot_pending": per_bot_pending,
                "last_event_per_bot": last_event_per_bot,
                "oldest_pending_age_seconds": round(oldest_age, 1),
                "db_size_bytes": db_size,
            }
        finally:
            conn.close()

    def vacuum(self) -> None:
        """Reclaim disk space after bulk deletes."""
        conn = self._connect()
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    def purge_acked(self, days: int = 7, vacuum: bool = True) -> int:
        """Delete acked events older than N days."""
        conn = self._connect()
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            cursor = conn.execute(
                "DELETE FROM events WHERE acked = 1 AND received_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                if vacuum:
                    conn.execute("VACUUM")
                logger.info("Purged %d acked events older than %d days", deleted, days)
            return deleted
        finally:
            conn.close()

    def start_periodic_purge(self, interval_hours: float = 6.0) -> None:
        """Schedule periodic purge of acked and stale unacked events."""
        self._purge_stop = threading.Event()

        def _run_purge():
            try:
                acked = self.purge_acked(days=7, vacuum=False)
                stale = self.purge_stale_unacked(days=3, vacuum=False)
                if acked > 0 or stale > 0:
                    self.vacuum()
            except Exception:
                logger.warning("Periodic purge failed", exc_info=True)

        def _purge_loop():
            _run_purge()  # immediate first run
            while not self._purge_stop.wait(timeout=interval_hours * 3600):
                _run_purge()

        self._purge_thread = threading.Thread(
            target=_purge_loop, daemon=True, name="relay-purge",
        )
        self._purge_thread.start()
        logger.info("Periodic purge scheduled every %.1fh", interval_hours)

    def stop_periodic_purge(self) -> None:
        """Stop the periodic purge thread."""
        if hasattr(self, "_purge_stop"):
            self._purge_stop.set()

    def purge_stale_unacked(self, days: int = 3, vacuum: bool = True) -> int:
        """Delete unacked events older than N days (no consumer draining them)."""
        conn = self._connect()
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            cursor = conn.execute(
                "DELETE FROM events WHERE acked = 0 AND received_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                if vacuum:
                    conn.execute("VACUUM")
                logger.info("Purged %d stale unacked events older than %d days", deleted, days)
            return deleted
        finally:
            conn.close()
