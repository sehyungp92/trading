"""Fail-closed engine router keyed by the immutable nightly artifact."""
from __future__ import annotations

from .core.daily_residual import DAILY_RESIDUAL_SLEEVE
from .engine import IARICEngine as LegacyIARICEngine
from .residual_engine import IARICDailyResidualEngine


class IARICEngineRouter:
    def __new__(cls, *args, **kwargs):
        artifact = kwargs.get("artifact")
        if artifact is None and len(args) >= 2:
            artifact = args[1]
        if artifact is None:
            raise ValueError("IARIC engine routing requires an artifact")
        mode = str(getattr(artifact, "strategy_mode", "legacy_pullback"))
        if mode == DAILY_RESIDUAL_SLEEVE:
            return IARICDailyResidualEngine(*args, **kwargs)
        if mode == "legacy_pullback":
            return LegacyIARICEngine(*args, **kwargs)
        raise ValueError(f"unsupported IARIC artifact strategy mode: {mode}")
