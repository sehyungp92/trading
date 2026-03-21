"""Parameter change detection — emits ParameterChangeEvent when config constants change."""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config_snapshot import snapshot_config_module

logger = logging.getLogger("instrumentation.config_watcher")


_SAFETY_CRITICAL_PARAMS = {
    "risk_per_trade", "max_position_size", "kill_switch_enabled",
    "trailing_stop_pct", "max_drawdown_pct", "leverage_limit",
    "HEAT_CAP_R", "DAILY_STOP_R", "WEEKLY_STOP_R", "MAX_POSITION_R",
}


class ConfigWatcher:
    """Monitors strategy config modules for parameter changes.

    Compares successive snapshots and emits change events when constants differ.
    Reuses existing config_snapshot infrastructure.

    Usage:
        watcher = ConfigWatcher(
            bot_id="momentum_nq_01",
            config_modules=["strategy.config", "strategy_2.config", "strategy_3.config"],
            data_dir="instrumentation/data",
        )
        # Call periodically (e.g., every heartbeat cycle):
        watcher.check()
    """

    def __init__(
        self,
        bot_id: str,
        config_modules: list[str],
        data_dir: str | Path,
    ) -> None:
        self._bot_id = bot_id
        self._config_modules = config_modules
        self._data_dir = Path(data_dir) / "config_changes"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._previous: dict[str, dict] = {}

        # Take initial snapshot
        for mod_name in self._config_modules:
            self._previous[mod_name] = self._snapshot_module(mod_name)

    def _snapshot_module(self, module_name: str) -> dict:
        """Use existing config_snapshot infrastructure."""
        try:
            mod = importlib.import_module(module_name)
            try:
                importlib.reload(mod)
            except Exception:
                pass  # reload may fail for synthetic/non-file modules; use cached version
            return snapshot_config_module(mod)
        except Exception:
            return {}

    def check(self) -> list[dict]:
        """Compare current config against previous snapshot, emit change events.

        Returns list of change event dicts (empty if no changes).
        """
        changes: list[dict] = []
        for mod_name in self._config_modules:
            current = self._snapshot_module(mod_name)
            previous = self._previous.get(mod_name, {})

            for key in set(current.keys()) | set(previous.keys()):
                old_val = previous.get(key)
                new_val = current.get(key)
                if old_val != new_val:
                    event = self._make_event(mod_name, key, old_val, new_val)
                    changes.append(event)
                    self._write_event(event)

            self._previous[mod_name] = current

        return changes

    def _make_event(
        self,
        module_name: str,
        param_name: str,
        old_value,
        new_value,
    ) -> dict:
        ts = datetime.now(timezone.utc).isoformat()
        raw = f"{self._bot_id}|{ts}|parameter_change|{module_name}:{param_name}"
        event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

        return {
            "bot_id": self._bot_id,
            "timestamp": ts,
            "event_id": event_id,
            "event_type": "parameter_change",
            "module": module_name,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "change_source": "hot_reload",
            "is_safety_critical": param_name in _SAFETY_CRITICAL_PARAMS,
        }

    def _write_event(self, event: dict) -> None:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self._data_dir / f"config_changes_{today}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write parameter change event: %s", e)
