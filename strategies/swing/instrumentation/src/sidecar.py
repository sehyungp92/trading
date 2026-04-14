"""Sidecar Forwarder — reads local events and forwards to the central relay.

Handles offline periods, network failures, and duplicate delivery gracefully.
Every payload is HMAC-SHA256 signed with canonicalized JSON (sort_keys=True).
"""
from __future__ import annotations

import gzip
import hashlib
import hmac as hmac_mod
import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger("instrumentation.sidecar")

_PRIORITY_MAP = {
    "error": 1,
    "trade": 3,
    "missed_opportunity": 3,
    "daily_snapshot": 3,
    "order": 3,
    "market_snapshot": 4,
    "process_quality": 4,
    "post_exit": 4,
    "coordinator_action": 4,
    "heartbeat": 4,
    # Phase 2B enriched event types
    "indicator_snapshot": 5,
    "filter_decision": 5,
    "orderbook_context": 4,
    "parameter_change": 3,
    "portfolio_rule_check": 3,
    "stop_adjustment": 3,
}

_DIR_TO_EVENT_TYPE = {
    "trades": "trade",
    "missed": "missed_opportunity",
    "errors": "error",
    "scores": "process_quality",
    "daily": "daily_snapshot",
    "snapshots": "market_snapshot",
    "post_exit": "post_exit",
    "coordination_events": "coordinator_action",
    "orders": "order",
    "heartbeat": "heartbeat",
    # Phase 2B enriched event types
    "indicators": "indicator_snapshot",
    "filter_decisions": "filter_decision",
    "orderbook": "orderbook_context",
    "config_changes": "parameter_change",
    "portfolio_rules": "portfolio_rule_check",
    "stop_adjustments": "stop_adjustment",
}


class Sidecar:
    """Forwards events from local JSONL files to the central relay.

    Usage::

        sidecar = Sidecar(config)
        sidecar.start()       # background thread
        # ...
        sidecar.stop()
    """

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"])

        sc = config.get("sidecar", {})
        raw_url = os.environ.get("INSTRUMENTATION_RELAY_URL") or sc.get("relay_url", "")
        # Ensure relay_url points to the /events ingest endpoint
        self.relay_url = raw_url.rstrip("/") + "/events" if raw_url and not raw_url.rstrip("/").endswith("/events") else raw_url
        self.batch_size = sc.get("batch_size", 50)
        self.retry_max = sc.get("retry_max", 5)
        self.retry_backoff_base = sc.get("retry_backoff_base_seconds", 10)
        self.buffer_dir = Path(sc.get("buffer_dir", str(self.data_dir / ".sidecar_buffer")))
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        hmac_env = sc.get("hmac_secret_env", "INSTRUMENTATION_HMAC_SECRET")
        self.hmac_secret = os.environ.get(hmac_env, "").encode()
        if not self.hmac_secret:
            logger.warning("HMAC secret not set in %s — events will be unsigned", hmac_env)

        # Event type filtering — only forward types the brain handles.
        # Default: all types the brain currently routes to real handlers.
        self.forward_event_types: set[str] | None = None
        fwd = sc.get("forward_event_types")
        if fwd is not None:
            self.forward_event_types = set(fwd)

        self.watermark_file = self.buffer_dir / "watermark.json"
        self.watermarks = self._load_watermarks()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.poll_interval = sc.get("poll_interval_seconds", 60)
        self.heartbeat_every_n = sc.get("heartbeat_every_n_polls", 10)

        # Heartbeat state
        self._relay_reachable: Optional[bool] = None
        self._last_successful_forward_at: Optional[str] = None
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._poll_count = 0

    # --- Watermarks ---

    def _load_watermarks(self) -> Dict[str, int]:
        if self.watermark_file.exists():
            try:
                return json.loads(self.watermark_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_watermarks(self) -> None:
        try:
            self.watermark_file.write_text(json.dumps(self.watermarks, indent=2))
        except OSError as e:
            logger.warning("Failed to save watermarks: %s", e)

    # --- Event collection ---

    def _get_event_files(self) -> List[tuple]:
        files: List[tuple] = []
        for subdir, event_type in _DIR_TO_EVENT_TYPE.items():
            dir_path = self.data_dir / subdir
            if not dir_path.exists():
                continue
            if subdir == "daily":
                for f in sorted(dir_path.glob("daily_*.json")):
                    files.append((f, event_type))
            else:
                for f in sorted(dir_path.glob("*.jsonl")):
                    files.append((f, event_type))
        return files

    def _read_unsent_events(self, filepath: Path, event_type: str) -> List[dict]:
        key = str(filepath)
        last_sent = self.watermarks.get(key, 0)
        events: List[dict] = []
        try:
            if filepath.suffix == ".jsonl":
                with open(filepath, "r") as fh:
                    for i, line in enumerate(fh):
                        if i < last_sent:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                            wrapped = self._wrap_event(raw, event_type)
                            wrapped["_source_file"] = key
                            wrapped["_line_number"] = i
                            events.append(wrapped)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning("Skipping bad line %d in %s: %s", i, filepath, e)
            elif filepath.suffix == ".json":
                if last_sent == 0:
                    raw = json.loads(filepath.read_text())
                    wrapped = self._wrap_event(raw, event_type)
                    wrapped["_source_file"] = key
                    wrapped["_line_number"] = 1
                    events.append(wrapped)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read %s: %s", filepath, e)
        return events

    def _wrap_event(self, raw_event: dict, event_type: str) -> dict:
        forwarded_event_type = self._forward_event_type(raw_event, event_type)
        metadata = raw_event.get("event_metadata", {})
        event_id = metadata.get("event_id", "")

        exchange_ts = (
            metadata.get("exchange_timestamp", "")
            or raw_event.get("entry_time", "")
            or raw_event.get("timestamp", "")
            or datetime.now(timezone.utc).isoformat()
        )

        if not event_id:
            key = raw_event.get("trade_id", raw_event.get("date", raw_event.get("snapshot_id", "")))
            raw_str = f"{self.bot_id}|{exchange_ts}|{forwarded_event_type}|{key}"
            event_id = hashlib.sha256(raw_str.encode()).hexdigest()[:16]

        # Compute priority: trade exits elevated to 2, errors to 1
        priority = _PRIORITY_MAP.get(forwarded_event_type, _PRIORITY_MAP.get(event_type, 3))
        if forwarded_event_type == "trade" and raw_event.get("stage") == "exit":
            priority = 2
        elif event_type == "order" and raw_event.get("coordinator_triggered", False):
            priority = 2

        return {
            "event_id": event_id,
            "bot_id": self.bot_id,
            "event_type": forwarded_event_type,
            "payload": json.dumps(raw_event, default=str),
            "exchange_timestamp": exchange_ts,
            "priority": priority,
        }

    @staticmethod
    def _forward_event_type(raw_event: dict, event_type: str) -> str:
        """Keep the assistant's canonical trade feed limited to completed trades."""
        if event_type == "trade":
            stage = str(raw_event.get("stage", "")).strip().lower()
            if stage and stage != "exit":
                return f"trade_{stage}"
        return event_type

    # --- Signing ---

    def _sign_payload(self, canonical_json: str) -> str:
        """HMAC-SHA256 of the canonicalized (sort_keys=True) JSON."""
        if not self.hmac_secret:
            return ""
        return hmac_mod.new(self.hmac_secret, canonical_json.encode(), hashlib.sha256).hexdigest()

    # --- Sending ---

    def _send_batch(self, events: List[dict]) -> bool:
        if not self.relay_url:
            logger.warning("No relay_url configured — skipping send")
            return False
        if requests is None:
            logger.error("requests library not installed — cannot forward events")
            return False

        clean_events = [{k: v for k, v in e.items() if not k.startswith("_")} for e in events]
        envelope = {"bot_id": self.bot_id, "events": clean_events}

        canonical = json.dumps(envelope, sort_keys=True)
        signature = self._sign_payload(canonical)

        # gzip compress if it saves bytes
        raw_bytes = canonical.encode()
        compressed = gzip.compress(raw_bytes)
        if len(compressed) < len(raw_bytes):
            send_bytes = compressed
            use_gzip = True
        else:
            send_bytes = raw_bytes
            use_gzip = False

        headers = {
            "Content-Type": "application/json",
            "X-Bot-ID": self.bot_id,
            "X-Signature": signature,
        }
        if use_gzip:
            headers["Content-Encoding"] = "gzip"

        for attempt in range(self.retry_max):
            try:
                response = requests.post(
                    self.relay_url,
                    data=send_bytes,
                    headers=headers,
                    timeout=30,
                )
                if response.status_code == 200:
                    self._relay_reachable = True
                    self._last_successful_forward_at = datetime.now(timezone.utc).isoformat()
                    return True
                elif response.status_code == 409:
                    self._relay_reachable = True
                    self._last_successful_forward_at = datetime.now(timezone.utc).isoformat()
                    return True  # duplicate, treat as success
                elif response.status_code == 401:
                    logger.error("Authentication failed — check HMAC secret")
                    return False
                elif response.status_code == 429:
                    logger.warning("Rate limited by relay — backing off")
                else:
                    logger.warning("Relay returned %d (attempt %d/%d)",
                                   response.status_code, attempt + 1, self.retry_max)
            except Exception as e:
                logger.warning("Send failed (attempt %d/%d): %s", attempt + 1, self.retry_max, e)

            backoff = min(self.retry_backoff_base * (2 ** attempt), 300)
            self._interruptible_sleep(backoff)

        self._relay_reachable = False
        return False

    # --- Main loop ---

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in 1-second chunks so stop() isn't blocked."""
        remaining = seconds
        while remaining > 0 and self._running:
            chunk = min(remaining, 1.0)
            time.sleep(chunk)
            remaining -= chunk

    def run_once(self) -> None:
        all_files = self._get_event_files()
        total_sent = 0

        for filepath, event_type in all_files:
            if self.forward_event_types is not None and event_type not in self.forward_event_types:
                continue
            unsent = self._read_unsent_events(filepath, event_type)
            if not unsent:
                continue

            for i in range(0, len(unsent), self.batch_size):
                batch = unsent[i:i + self.batch_size]
                if self._send_batch(batch):
                    key = str(filepath)
                    max_line = max(e["_line_number"] for e in batch)
                    self.watermarks[key] = max_line + 1
                    self._save_watermarks()
                    total_sent += len(batch)
                else:
                    logger.warning("Failed to send batch from %s, will retry", filepath)
                    break

        # Periodically clean up watermarks for rotated/deleted files
        if self._poll_count > 0 and self._poll_count % self.heartbeat_every_n == 0:
            try:
                self.cleanup_old_watermarks()
            except Exception as e:
                logger.debug("Watermark cleanup failed: %s", e)

        if total_sent > 0:
            logger.info("Forwarded %d events to relay", total_sent)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Sidecar started (poll every %ds)", self.poll_interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _run_loop(self) -> None:
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error("Sidecar run_once failed: %s", e)
            self._poll_count += 1
            if self._poll_count % self.heartbeat_every_n == 0:
                try:
                    self._emit_heartbeat()
                except Exception as e:
                    logger.debug("Heartbeat emission failed: %s", e)
            time.sleep(self.poll_interval)

    def _compute_buffer_depth(self) -> int:
        """Count unsent lines across all event files."""
        total = 0
        for filepath, event_type in self._get_event_files():
            key = str(filepath)
            last_sent = self.watermarks.get(key, 0)
            try:
                if filepath.suffix == ".jsonl":
                    with open(filepath, "r") as fh:
                        line_count = sum(1 for line in fh if line.strip())
                    total += max(0, line_count - last_sent)
                elif filepath.suffix == ".json" and last_sent == 0:
                    total += 1
            except OSError:
                pass
        return total

    def _emit_heartbeat(self) -> None:
        """Synthesize and send a heartbeat event."""
        if not self.relay_url or requests is None:
            return

        heartbeat = {
            "event_id": hashlib.sha256(
                f"{self.bot_id}|heartbeat|{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:16],
            "bot_id": self.bot_id,
            "event_type": "heartbeat",
            "payload": json.dumps({
                "source": "sidecar",
                "relay_reachable": self._relay_reachable,
                "last_successful_forward_at": self._last_successful_forward_at,
                "buffer_depth": self._compute_buffer_depth(),
                "uptime_since": self._start_time,
                "poll_count": self._poll_count,
            }),
            "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": _PRIORITY_MAP.get("heartbeat", 4),
        }

        self._send_batch([heartbeat])

    def cleanup_old_watermarks(self) -> None:
        to_remove = [key for key in self.watermarks if not Path(key).exists()]
        for key in to_remove:
            del self.watermarks[key]
        if to_remove:
            self._save_watermarks()
