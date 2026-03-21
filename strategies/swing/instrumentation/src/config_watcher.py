"""Config watcher — detects parameter changes and emits ParameterChangeEvents.

Snapshots Python config modules and YAML files, then diffs on each check()
to produce events for any changed values.  Handles frozen dataclass configs
by snapshotting class-level defaults.
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .event_metadata import create_event_metadata

logger = logging.getLogger("instrumentation.config_watcher")


@dataclass
class ParameterChangeEvent:
    """A single parameter change detection."""

    bot_id: str
    param_name: str
    old_value: Any
    new_value: Any
    change_source: str = "pr_merge"   # "pr_merge", "experiment", "manual"
    timestamp: str = ""
    config_file: str = ""
    commit_sha: Optional[str] = None
    pr_url: Optional[str] = None
    event_id: str = ""
    event_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ConfigWatcher:
    """Watches config modules and YAML files for parameter changes.

    Usage::

        watcher = ConfigWatcher(config, config_modules=["strategy.config"])
        watcher.take_baseline()

        # Periodically from main loop:
        watcher.check()
    """

    def __init__(
        self,
        config: dict,
        config_modules: Optional[list[str]] = None,
        yaml_paths: Optional[list[Path]] = None,
    ):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "config_changes"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_source_id = config.get("data_source_id", "ibkr_execution")

        self.config_modules = config_modules or ["strategy.config"]
        self.yaml_paths = yaml_paths or []

        self._baseline: dict[str, dict[str, Any]] = {}
        self._yaml_baseline: dict[str, dict[str, Any]] = {}

    def take_baseline(self) -> None:
        """Snapshot current state of all watched configs."""
        for mod_name in self.config_modules:
            self._baseline[mod_name] = self._snapshot_module(mod_name)
        for path in self.yaml_paths:
            self._yaml_baseline[str(path)] = self._snapshot_yaml(path)

    def check(self) -> list[ParameterChangeEvent]:
        """Compare current config against baseline, emit events for changes."""
        events: list[ParameterChangeEvent] = []

        for mod_name in self.config_modules:
            current = self._snapshot_module(mod_name)
            baseline = self._baseline.get(mod_name, {})

            for key, new_val in current.items():
                old_val = baseline.get(key)
                if old_val != new_val:
                    event = self._emit_change(
                        param_name=key,
                        old_value=old_val,
                        new_value=new_val,
                        config_file=mod_name,
                        change_source="pr_merge",
                    )
                    events.append(event)

            # Detect removed keys
            for key in baseline:
                if key not in current:
                    event = self._emit_change(
                        param_name=key,
                        old_value=baseline[key],
                        new_value=None,
                        config_file=mod_name,
                        change_source="pr_merge",
                    )
                    events.append(event)

            self._baseline[mod_name] = current

        # Check YAML files
        for path in self.yaml_paths:
            path_key = str(path)
            current = self._snapshot_yaml(path)
            baseline = self._yaml_baseline.get(path_key, {})

            for key, new_val in current.items():
                old_val = baseline.get(key)
                if old_val != new_val:
                    event = self._emit_change(
                        param_name=key,
                        old_value=old_val,
                        new_value=new_val,
                        config_file=path_key,
                        change_source="experiment",
                    )
                    events.append(event)

            for key in baseline:
                if key not in current:
                    event = self._emit_change(
                        param_name=key,
                        old_value=baseline[key],
                        new_value=None,
                        config_file=path_key,
                        change_source="experiment",
                    )
                    events.append(event)

            self._yaml_baseline[path_key] = current

        return events

    def _snapshot_module(self, module_name: str) -> dict[str, Any]:
        """Snapshot a config module's uppercase constants and dataclass defaults."""
        try:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)
            result: dict[str, Any] = {}

            # Extract uppercase module constants
            for name in dir(mod):
                if not name.isupper() or name.startswith("_"):
                    continue
                val = getattr(mod, name)
                if callable(val) and not isinstance(val, type):
                    continue
                result[name] = self._make_json_safe(val)

            # Snapshot SymbolConfig defaults if present
            if hasattr(mod, "SymbolConfig"):
                defaults: dict[str, Any] = {}
                for f in mod.SymbolConfig.__dataclass_fields__.values():
                    if f.default is not f.default_factory:
                        defaults[f.name] = self._make_json_safe(f.default)
                result["__SymbolConfig_defaults__"] = defaults

            return result
        except Exception as e:
            logger.debug("Failed to snapshot %s: %s", module_name, e)
            return {}

    def _snapshot_yaml(self, path: Path) -> dict[str, Any]:
        """Snapshot a YAML config file."""
        if not path.exists():
            return {}
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return self._make_json_safe(data)
        except Exception as e:
            logger.debug("Failed to snapshot %s: %s", path, e)
            return {}

    def _make_json_safe(self, val: Any) -> Any:
        """Convert a value to a JSON-serializable form for diffing."""
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        if isinstance(val, (list, tuple)):
            return [self._make_json_safe(v) for v in val]
        if isinstance(val, dict):
            return {str(k): self._make_json_safe(v) for k, v in val.items()}
        if isinstance(val, set):
            return sorted(self._make_json_safe(v) for v in val)
        # Fallback: repr
        return repr(val)

    def _emit_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        config_file: str,
        change_source: str,
    ) -> ParameterChangeEvent:
        """Create and write a ParameterChangeEvent."""
        now = datetime.now(timezone.utc)

        meta = create_event_metadata(
            bot_id=self.bot_id,
            event_type="parameter_change",
            payload_key=f"{config_file}:{param_name}:{now.isoformat()}",
            exchange_timestamp=now,
            data_source_id=self.data_source_id,
        )

        event = ParameterChangeEvent(
            bot_id=self.bot_id,
            param_name=param_name,
            old_value=old_value,
            new_value=new_value,
            change_source=change_source,
            timestamp=now.isoformat(),
            config_file=config_file,
            event_id=meta.event_id,
            event_metadata=meta.to_dict(),
        )

        self._write_event(event)
        return event

    def _write_event(self, event: ParameterChangeEvent) -> None:
        try:
            date_str = event.timestamp[:10] if event.timestamp else (
                datetime.now(timezone.utc).strftime("%Y-%m-%d")
            )
            filepath = self.data_dir / f"config_changes_{date_str}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception:
            logger.exception("Failed to write ParameterChangeEvent")
