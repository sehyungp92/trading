"""Plugin adapter for IARIC intraday stock strategy."""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from strategies.contracts import RuntimeContext
from strategies.core.capital import resolve_plugin_nav
from .artifact_store import coerce_intraday_state_snapshot
from .config import StrategySettings
from .diagnostics import JsonlDiagnostics
from .router import IARICEngineRouter

logger = logging.getLogger(__name__)


class IARICPlugin:
    strategy_id = "IARIC_v1"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        manifest = ctx.manifest
        settings = StrategySettings()

        # account_id from the connection group tied to this strategy
        conn_group = ctx.registry.connection_groups[manifest.connection_group]
        account_id = conn_group.account_id or ""

        nav = resolve_plugin_nav(ctx, self.strategy_id)

        # Artifact will be supplied by the family coordinator before start().
        # Store a sentinel so the coordinator can inject it.
        self._artifact: Any = None

        trade_recorder = getattr(ctx.instrumentation, "trade_recorder", None)
        diagnostics = JsonlDiagnostics(settings.diagnostics_dir, enabled=True)

        self._settings = settings
        self._account_id = account_id
        self._nav = nav
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics
        self._instrumentation = ctx.instrumentation
        self._engine: Any | None = None
        self._pending_snapshot: Any | None = None

    # -- lifecycle --------------------------------------------------------

    def _build_engine(self) -> Any:
        if self._artifact is None:
            raise RuntimeError(
                f"{self.strategy_id}: artifact must be set before start(). "
                "The family coordinator should call plugin._artifact = artifact."
            )
        settings = self._settings
        if getattr(self._artifact, "strategy_mode", "") == "daily_residual_reversion":
            parameters = dict(self._artifact.strategy_parameters)
            settings = dataclasses.replace(
                settings,
                strategy_mode="daily_residual_reversion",
                daily_residual_factor_model=str(parameters["factor_model"]),
                daily_residual_formation_sessions=int(parameters["formation_sessions"]),
                daily_residual_minimum_z=float(parameters["minimum_z"]),
                daily_residual_minimum_score=float(
                    parameters.get("minimum_score", 0.0)
                ),
                daily_residual_minimum_failed_continuation_r=float(
                    parameters.get("minimum_failed_continuation_r", 0.0)
                ),
                daily_residual_lane_id=str(
                    parameters.get("lane_id", "daily_residual_generic")
                ),
                daily_residual_minimum_sector_return_5d=float(
                    parameters.get("minimum_sector_return_5d", -0.15)
                ),
                daily_residual_score_components=tuple(parameters["score_components"]),
                daily_residual_ranking_score_components=tuple(
                    parameters.get("ranking_score_components", ())
                ),
                daily_residual_max_positions=int(parameters["max_positions"]),
                daily_residual_max_positions_per_sector=int(
                    parameters["max_positions_per_sector"]
                ),
                daily_residual_sector_overflow_slots=int(
                    parameters.get("sector_overflow_slots", 0)
                ),
                daily_residual_sector_overflow_minimum_score=float(
                    parameters.get("sector_overflow_minimum_score", 50.0)
                ),
                daily_residual_sector_overflow_minimum_z=float(
                    parameters.get("sector_overflow_minimum_z", 1.0)
                ),
                daily_residual_sector_overflow_risk_multiplier=float(
                    parameters.get("sector_overflow_risk_multiplier", 1.0)
                ),
                daily_residual_risk_fraction=float(parameters["risk_fraction"]),
                daily_residual_maximum_notional_fraction=float(
                    parameters["maximum_notional_fraction"]
                ),
                daily_residual_catastrophic_stop_atr=float(
                    parameters.get("catastrophic_stop_atr", 2.5)
                ),
                daily_residual_catastrophic_stop_residual_r=float(
                    parameters.get("catastrophic_stop_residual_r", 4.0)
                ),
                daily_residual_partial_normalization_fraction=float(
                    parameters["partial_normalization_fraction"]
                ),
                daily_residual_full_normalization_fraction=float(
                    parameters["full_normalization_fraction"]
                ),
                daily_residual_structural_failure_extension_fraction=float(
                    parameters["structural_failure_extension_fraction"]
                ),
                daily_residual_profit_retention_activation_fraction=float(
                    parameters.get("profit_retention_activation_fraction", 99.0)
                ),
                daily_residual_profit_retention_giveback_fraction=float(
                    parameters.get("profit_retention_giveback_fraction", 99.0)
                ),
                daily_residual_maximum_holding_sessions=int(
                    parameters["maximum_holding_sessions"]
                ),
                daily_residual_partial_exit_fraction=float(
                    parameters["partial_exit_fraction"]
                ),
            )
        return IARICEngineRouter(
            oms_service=self._ctx.oms,
            artifact=self._artifact,
            account_id=self._account_id,
            nav=self._nav,
            settings=settings,
            trade_recorder=self._trade_recorder,
            diagnostics=self._diagnostics,
            instrumentation=self._instrumentation,
        )

    async def start(self) -> None:
        self._engine = self._build_engine()
        if self._pending_snapshot is not None:
            self._engine.hydrate_state(
                coerce_intraday_state_snapshot(self._pending_snapshot)
            )
        await self._engine.start()

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.stop()

    def health_status(self) -> dict[str, Any]:
        if self._engine is not None:
            return self._engine.health_status()
        return {
            "strategy_id": self.strategy_id,
            "running": False,
            "has_artifact": self._artifact is not None,
        }

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        self._pending_snapshot = coerce_intraday_state_snapshot(snapshot)
        if self._engine is not None:
            self._engine.hydrate_state(self._pending_snapshot)

    def snapshot_state(self) -> dict[str, Any]:
        if self._engine is not None and hasattr(self._engine, "snapshot_state"):
            state = self._engine.snapshot_state()
            if dataclasses.is_dataclass(state):
                return dataclasses.asdict(state)
            return state
        if self._pending_snapshot is not None:
            if dataclasses.is_dataclass(self._pending_snapshot):
                return dataclasses.asdict(self._pending_snapshot)
            return self._pending_snapshot
        return {"strategy_id": self.strategy_id}

    async def on_market_data(self, event: Any) -> None:
        pass

    async def on_order_event(self, event: Any) -> None:
        pass

    async def on_fill_event(self, event: Any) -> None:
        pass
