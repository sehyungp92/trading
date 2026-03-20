import gzip
import hashlib
import hmac
import json
import os
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("instrumentation.sidecar")

_DIR_TO_EVENT_TYPE = {
    "trades": "trade",
    "missed": "missed_opportunity",
    "errors": "error",
    "scores": "process_quality",
    "daily": "daily_snapshot",
    "orders": "order",
    "heartbeats": "heartbeat",
    "portfolio_rules": "portfolio_rule",
    # Phase 2B event types
    "indicators": "indicator_snapshot",
    "filter_decisions": "filter_decision",
    "orderbook": "orderbook_context",
    "config_changes": "parameter_change",
    "snapshots": "market_snapshot",
    "post_exit": "post_exit",
}

# Event priority for sorting (#25): lower number = higher priority
_EVENT_PRIORITY = {
    "error": 0,
    "daily_snapshot": 1,
    "trade": 2,
    "missed_opportunity": 3,
    "order": 3,
    "portfolio_rule": 3,
    "parameter_change": 3,
    "process_quality": 4,
    "indicator_snapshot": 4,
    "filter_decision": 4,
    "orderbook_context": 4,
    "market_snapshot": 4,
    "post_exit": 4,
    "heartbeat": 5,
}


class Sidecar:
    """
    Forwards events from local JSONL files to the central relay.
    Runs as a background thread or standalone process.

    Usage:
        sidecar = Sidecar(config)
        sidecar.start()  # background thread
        # or
        sidecar.run_once()  # one-shot forward
    """

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"])

        sidecar_config = config.get("sidecar", {})
        raw_url = os.environ.get("INSTRUMENTATION_RELAY_URL") or sidecar_config.get("relay_url", "")
        self.relay_url = raw_url.rstrip("/") + "/events" if raw_url and not raw_url.rstrip("/").endswith("/events") else raw_url
        self.batch_size = sidecar_config.get("batch_size", 50)
        self.retry_max = sidecar_config.get("retry_max", 5)
        self.retry_backoff_base = sidecar_config.get("retry_backoff_base_seconds", 10)
        self.buffer_dir = Path(sidecar_config.get("buffer_dir", str(self.data_dir / ".sidecar_buffer")))
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        hmac_env = sidecar_config.get("hmac_secret_env", "INSTRUMENTATION_HMAC_SECRET")
        self.hmac_secret = os.environ.get(hmac_env, "").encode()
        if not self.hmac_secret:
            logger.warning("HMAC secret not set in %s — events will be unsigned", hmac_env)

        self.watermark_file = self.buffer_dir / "watermark.json"
        self.watermarks = self._load_watermarks()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.poll_interval = sidecar_config.get("poll_interval_seconds", 60)

        # gzip compression (#26)
        self.use_gzip = sidecar_config.get("use_gzip", False)

        # Diagnostics state (#24)
        self._buffer_depth = 0
        self._relay_reachable = False
        self._last_successful_forward_at: Optional[str] = None
        self._total_forwarded = 0
        self._last_error: Optional[str] = None

    def get_diagnostics(self) -> dict:
        """Return snapshot of sidecar health state (#24)."""
        return {
            "sidecar_buffer_depth": self._buffer_depth,
            "relay_reachable": self._relay_reachable,
            "last_successful_forward_at": self._last_successful_forward_at,
            "total_forwarded": self._total_forwarded,
            "last_error": self._last_error,
        }

    def _load_watermarks(self) -> Dict[str, int]:
        if self.watermark_file.exists():
            try:
                return json.loads(self.watermark_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_watermarks(self):
        try:
            self.watermark_file.write_text(json.dumps(self.watermarks, indent=2))
        except OSError as e:
            logger.warning("Failed to save watermarks: %s", e)

    def _get_event_files(self) -> List[Tuple[Path, str]]:
        files: List[Tuple[Path, str]] = []
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

        events = []
        try:
            if filepath.suffix == ".jsonl":
                with open(filepath, "r", encoding="utf-8") as handle:
                    for i, line in enumerate(handle):
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
        metadata = raw_event.get("event_metadata", {})
        event_id = metadata.get("event_id", "")

        exchange_ts = (
            metadata.get("exchange_timestamp", "")
            or raw_event.get("entry_time", "")
            or raw_event.get("timestamp", "")
            or raw_event.get("date", "")
            or datetime.now(timezone.utc).isoformat()
        )

        if not event_id:
            key = raw_event.get("trade_id", raw_event.get("date", raw_event.get("snapshot_id", "")))
            raw_str = f"{self.bot_id}|{exchange_ts}|{event_type}|{key}"
            event_id = hashlib.sha256(raw_str.encode()).hexdigest()[:16]

        return {
            "event_id": event_id,
            "bot_id": self.bot_id,
            "event_type": event_type,
            "priority": _EVENT_PRIORITY.get(event_type, 99),
            "payload": json.dumps(raw_event, default=str),
            "exchange_timestamp": exchange_ts,
        }

    def _sign_payload(self, canonical_json: str) -> str:
        """HMAC-SHA256 signature of the canonicalized JSON payload.

        CRITICAL: The relay verifies against json.dumps(data, sort_keys=True).
        The input MUST be the sort_keys=True serialization.
        """
        if not self.hmac_secret:
            return ""
        return hmac.new(self.hmac_secret, canonical_json.encode(), hashlib.sha256).hexdigest()

    def _send_batch(self, events: List[dict]) -> bool:
        if not self.relay_url:
            logger.warning("No relay_url configured — skipping send")
            return False

        if requests is None:
            logger.error("requests library not installed — cannot forward events")
            return False

        clean_events = []
        for e in events:
            clean = {k: v for k, v in e.items() if not k.startswith("_")}
            clean_events.append(clean)

        envelope = {
            "bot_id": self.bot_id,
            "events": clean_events,
        }

        canonical = json.dumps(envelope, sort_keys=True)
        signature = self._sign_payload(canonical)

        headers = {
            "Content-Type": "application/json",
            "X-Bot-ID": self.bot_id,
            "X-Signature": signature,
        }

        # gzip compression (#26): compress payload, HMAC on uncompressed
        body = canonical.encode()
        if self.use_gzip:
            body = gzip.compress(body)
            headers["Content-Encoding"] = "gzip"

        for attempt in range(self.retry_max):
            try:
                response = requests.post(
                    self.relay_url,
                    data=body,
                    headers=headers,
                    timeout=30,
                )
                if response.status_code in (200, 409):
                    self._relay_reachable = True
                    self._last_successful_forward_at = datetime.now(timezone.utc).isoformat()
                    self._total_forwarded += len(events)
                    self._last_error = None
                    return True
                elif response.status_code == 401:
                    logger.error("Authentication failed — check HMAC secret")
                    self._last_error = "auth_failed"
                    return False
                elif response.status_code == 415 and self.use_gzip:
                    # Relay doesn't support gzip — auto-disable and retry uncompressed
                    logger.warning("Relay returned 415 — disabling gzip compression")
                    self.use_gzip = False
                    body = canonical.encode()
                    headers.pop("Content-Encoding", None)
                    continue
                elif response.status_code == 429:
                    logger.warning("Rate limited by relay — backing off")
                else:
                    logger.warning(
                        "Relay returned %d (attempt %d/%d)",
                        response.status_code, attempt + 1, self.retry_max,
                    )
            except Exception as e:
                logger.warning("Send failed (attempt %d/%d): %s", attempt + 1, self.retry_max, e)

            backoff = self.retry_backoff_base * (2 ** attempt)
            time.sleep(min(backoff, 300))

        self._relay_reachable = False
        self._last_error = "all_retries_failed"
        return False

    def run_once(self):
        """Collect and forward all unsent events, sorted by priority (#25)."""
        self.cleanup_old_watermarks()
        all_files = self._get_event_files()

        # Collect ALL unsent events across all files first (#25)
        all_unsent: List[dict] = []
        for filepath, event_type in all_files:
            unsent = self._read_unsent_events(filepath, event_type)
            all_unsent.extend(unsent)

        # Update buffer depth diagnostic (#24)
        self._buffer_depth = len(all_unsent)

        if not all_unsent:
            return

        # Sort by priority: errors first, then snapshots, trades, missed, scores (#25)
        all_unsent.sort(key=lambda e: e.get("priority", 99))

        total_sent = 0
        for i in range(0, len(all_unsent), self.batch_size):
            batch = all_unsent[i:i + self.batch_size]
            if self._send_batch(batch):
                # Update watermarks per-event using _source_file/_line_number
                for evt in batch:
                    src = evt["_source_file"]
                    line = evt["_line_number"]
                    current = self.watermarks.get(src, 0)
                    if line + 1 > current:
                        self.watermarks[src] = line + 1
                self._save_watermarks()
                total_sent += len(batch)
            else:
                logger.warning("Failed to send batch, will retry")
                break

        # Update buffer depth after sending
        self._buffer_depth = max(0, self._buffer_depth - total_sent)

        if total_sent > 0:
            logger.info("Forwarded %d events to relay", total_sent)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Sidecar started (poll every %ds)", self.poll_interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _run_loop(self):
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error("Sidecar run_once failed: %s", e)
            time.sleep(self.poll_interval)

    def cleanup_old_watermarks(self):
        to_remove = [key for key in self.watermarks if not Path(key).exists()]
        for key in to_remove:
            del self.watermarks[key]
        if to_remove:
            self._save_watermarks()
