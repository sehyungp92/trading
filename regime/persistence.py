"""Load/save RegimeContext to JSON for live runtime consumption."""
from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from regime.context import RegimeContext

logger = logging.getLogger(__name__)

REGIME_CONTEXT_PATH = Path("data/regime/latest_context.json")

RECOVERY_DEFAULT = RegimeContext(
    regime="G",
    regime_confidence=0.5,
    stress_level=0.0,
    stress_onset=False,
    shift_velocity=0.0,
    suggested_leverage_mult=1.0,
    regime_allocations={
        "SPY": 0.278, "TLT": 0.024, "GLD": 0.031,
        "IBIT": 0.25, "EFA": 0.10, "CASH": 0.144,
    },
    computed_at="",
)


def load_regime_context(path: Path = REGIME_CONTEXT_PATH) -> RegimeContext:
    """Load from JSON, return RECOVERY_DEFAULT on any failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ctx = RegimeContext(**data)

        # Staleness check: warn if >10 days old but still use
        computed_at = ctx.computed_at
        if computed_at:
            try:
                ts = datetime.fromisoformat(computed_at)
                age = datetime.now(timezone.utc) - ts
                if age.days > 10:
                    logger.warning(
                        "Regime context is %d days old (computed_at=%s)",
                        age.days, computed_at,
                    )
            except (ValueError, TypeError):
                logger.warning("Cannot parse computed_at=%r for staleness check", computed_at)

        return ctx

    except FileNotFoundError:
        logger.warning("Regime context file not found (%s), using Recovery default", path)
        return RECOVERY_DEFAULT
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Regime context file corrupt (%s): %s, using Recovery default", path, exc)
        return RECOVERY_DEFAULT


def save_regime_context(ctx: RegimeContext, path: Path = REGIME_CONTEXT_PATH) -> None:
    """Persist to JSON atomically. Creates data/regime/ directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(ctx)
    # Atomic write: temp file + rename prevents partial reads
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
