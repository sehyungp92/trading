"""Runtime shell and preflight checks for the monorepo scaffold."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from libs.broker_ibkr.session import UnifiedIBSession
from libs.config.capital_allocation import resolve_strategy_capital_allocation
from libs.config.loader import (
    load_contracts,
    load_event_calendar,
    load_portfolio_config,
    load_routes,
    load_strategy_registry,
)
from libs.config.registry import build_registry_artifact
from libs.oms.persistence.db_config import get_environment
from strategies.contracts import RuntimeContext

logger = logging.getLogger(__name__)

# Family coordinator registry (lazy imports to avoid circular deps)
_FAMILY_COORDINATORS: dict[str, str] = {
    "swing": "strategies.swing.coordinator.SwingFamilyCoordinator",
    "momentum": "strategies.momentum.coordinator.MomentumFamilyCoordinator",
    "stock": "strategies.stock.coordinator.StockFamilyCoordinator",
}


def _import_coordinator(family: str) -> type:
    """Dynamically import a family coordinator class."""
    dotted = _FAMILY_COORDINATORS[family]
    module_path, class_name = dotted.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


class RuntimeShell:
    """Loads monorepo runtime metadata and optionally starts IB connectivity."""

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.registry = None
        self.portfolio = None
        self.contracts = None
        self.routes = None
        self.event_calendar = None
        self.session: UnifiedIBSession | None = None

    def load(self) -> None:
        self.registry = load_strategy_registry(self.config_dir)
        self.portfolio = load_portfolio_config(self.config_dir)
        self.contracts = load_contracts(self.config_dir)
        self.routes = load_routes(self.config_dir)
        self.event_calendar = load_event_calendar(self.config_dir)

    def _require_loaded(self) -> None:
        """Verify config was loaded. Raises RuntimeError instead of assert."""
        for attr in ("registry", "portfolio", "contracts", "routes", "event_calendar"):
            if getattr(self, attr) is None:
                raise RuntimeError(f"config not loaded — call load() first (missing: {attr})")

    def run_preflight(self) -> list[PreflightCheck]:
        self.load()
        self._require_loaded()

        checks: list[PreflightCheck] = []
        enabled = self.registry.enabled_strategies()
        checks.append(
            PreflightCheck(
                name="registry-load",
                ok=True,
                detail=f"Loaded {len(self.registry.strategies)} strategies across {len(self.registry.connection_groups)} groups",
            )
        )
        checks.append(
            PreflightCheck(
                name="enabled-strategies",
                ok=bool(enabled),
                detail=f"{len(enabled)} strategies enabled",
            )
        )

        missing_contracts: list[str] = []
        missing_routes: list[str] = []
        for manifest in enabled:
            for symbol in manifest.symbols:
                if symbol not in self.contracts:
                    missing_contracts.append(f"{manifest.strategy_id}:{symbol}")
                if symbol not in self.routes:
                    missing_routes.append(f"{manifest.strategy_id}:{symbol}")
        checks.append(
            PreflightCheck(
                name="contract-coverage",
                ok=not missing_contracts,
                detail="all manifest symbols resolved"
                if not missing_contracts
                else ", ".join(missing_contracts),
            )
        )
        checks.append(
            PreflightCheck(
                name="route-coverage",
                ok=not missing_routes,
                detail="all manifest symbols routed"
                if not missing_routes
                else ", ".join(missing_routes),
            )
        )

        family_total = sum(self.portfolio.capital.family_allocations.values())
        checks.append(
            PreflightCheck(
                name="family-allocation-sum",
                ok=math.isclose(family_total, 1.0, abs_tol=1e-9),
                detail=f"family allocation total={family_total:.6f}",
            )
        )

        # Dynamic per-family allocation check for families with explicit strategy_allocations
        families_with_explicit: dict[str, list[str]] = {}
        for manifest in enabled:
            if manifest.strategy_id in self.portfolio.capital.strategy_allocations:
                families_with_explicit.setdefault(manifest.family, []).append(manifest.strategy_id)
        for family, strategy_ids in families_with_explicit.items():
            family_total = sum(
                self.portfolio.capital.strategy_allocations.get(sid, 0.0)
                for sid in strategy_ids
            )
            checks.append(
                PreflightCheck(
                    name=f"family-allocation-sum:{family}",
                    ok=math.isclose(family_total, 1.0, abs_tol=1e-9),
                    detail=f"{family} enabled strategy allocation total={family_total:.6f} ({', '.join(strategy_ids)})",
                )
            )

        for manifest in enabled:
            allocation = resolve_strategy_capital_allocation(
                manifest.strategy_id,
                raw_nav=self.portfolio.capital.initial_equity,
                registry=self.registry,
                portfolio=self.portfolio,
            )
            checks.append(
                PreflightCheck(
                    name=f"allocation:{manifest.strategy_id}",
                    ok=allocation.allocated_nav > 0,
                    detail=f"allocated_nav={allocation.allocated_nav:.2f}",
                )
            )

        artifact = build_registry_artifact(self.registry)
        checks.append(
            PreflightCheck(
                name="registry-artifact",
                ok=len(artifact["strategies"]) == len(self.registry.strategies),
                detail=f"artifact strategies={len(artifact['strategies'])}",
            )
        )

        # Family cross-validation: every enabled strategy's family must have a family_allocation entry
        families_used = {m.family for m in enabled}
        families_configured = set(self.portfolio.capital.family_allocations.keys())
        missing_families = families_used - families_configured
        extra_families = families_configured - families_used
        family_detail_parts: list[str] = []
        if missing_families:
            family_detail_parts.append(f"missing allocations for: {sorted(missing_families)}")
        if extra_families:
            family_detail_parts.append(f"unreferenced families: {sorted(extra_families)}")
        checks.append(
            PreflightCheck(
                name="family-allocation-coverage",
                ok=not missing_families,
                detail="; ".join(family_detail_parts) if family_detail_parts else "all families covered",
            )
        )

        checks.append(
            PreflightCheck(
                name="event-calendar",
                ok=True,
                detail=f"{len(self.event_calendar.windows)} blackout windows configured",
            )
        )
        return checks

    async def _run_async_preflight(
        self,
        connect_ib: bool,
        families: set[str],
    ) -> list[PreflightCheck]:
        """Run async preflight checks before heavy startup.

        Checks:
          1a. Coordinator imports (CRITICAL)
          1b. Database connectivity (CRITICAL in paper/live)
          1c. IB Gateway reachability (CRITICAL when connect_ib=True)
          1d. Instrumentation config parsability (WARNING only)
        """
        checks: list[PreflightCheck] = []

        # 1a. Coordinator imports
        for family in sorted(families):
            if family not in _FAMILY_COORDINATORS:
                checks.append(PreflightCheck(
                    name=f"import:{family}",
                    ok=False,
                    detail=f"No coordinator registered for family '{family}'",
                ))
                continue
            try:
                _import_coordinator(family)
                checks.append(PreflightCheck(
                    name=f"import:{family}",
                    ok=True,
                    detail=f"Coordinator for '{family}' imported successfully",
                ))
            except Exception as exc:
                checks.append(PreflightCheck(
                    name=f"import:{family}",
                    ok=False,
                    detail=f"Coordinator import failed: {exc}",
                ))

        # 1b. Database connectivity
        try:
            from libs.oms.persistence.db_config import DBConfig
            db_config = DBConfig.from_env()
            if db_config is not None:
                import asyncpg
                conn = await asyncio.wait_for(
                    asyncpg.connect(dsn=db_config.to_dsn()),
                    timeout=5.0,
                )
                await conn.execute("SELECT 1")
                await conn.close()
                checks.append(PreflightCheck(
                    name="database",
                    ok=True,
                    detail="Database reachable",
                ))
            else:
                env = get_environment()
                db_required = env in ("paper", "live")
                checks.append(PreflightCheck(
                    name="database",
                    ok=not db_required,
                    detail=f"No DB config (env={env})"
                    + (" -- required for paper/live" if db_required else " -- OK for dev/backtest"),
                ))
        except Exception as exc:
            checks.append(PreflightCheck(
                name="database",
                ok=False,
                detail=f"Database unreachable: {exc}",
            ))

        # 1c. IB Gateway reachability (async to avoid blocking the event loop)
        if connect_ib and self.registry is not None:
            for group_name, group_cfg in self.registry.connection_groups.items():
                host = getattr(group_cfg, "host", "127.0.0.1")
                port = getattr(group_cfg, "port", 4002)
                try:
                    _reader, _writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port),
                        timeout=5.0,
                    )
                    _writer.close()
                    await _writer.wait_closed()
                    checks.append(PreflightCheck(
                        name=f"ib-gateway:{group_name}",
                        ok=True,
                        detail=f"IB Gateway reachable at {host}:{port}",
                    ))
                except Exception as exc:
                    checks.append(PreflightCheck(
                        name=f"ib-gateway:{group_name}",
                        ok=False,
                        detail=f"IB Gateway unreachable at {host}:{port}: {exc}",
                    ))

        # 1d. Instrumentation config parsability (WARNING only)
        for family in sorted(families):
            config_path = (
                Path(__file__).resolve().parent.parent.parent
                / "strategies" / family / "instrumentation" / "config" / "instrumentation_config.yaml"
            )
            if not config_path.exists():
                continue  # missing is OK -- strategies use defaults
            try:
                import yaml
                with open(config_path) as f:
                    yaml.safe_load(f)
                checks.append(PreflightCheck(
                    name=f"instr-config:{family}",
                    ok=True,
                    detail="Instrumentation config parsed OK",
                ))
            except Exception as exc:
                checks.append(PreflightCheck(
                    name=f"instr-config:{family}",
                    ok=True,  # WARNING only, don't block startup
                    detail=f"Instrumentation config parse error (non-fatal): {exc}",
                ))
                logger.warning("Instrumentation config for %s unparseable: %s", family, exc)

        return checks

    async def run(
        self,
        shadow: bool = False,
        connect_ib: bool = False,
        once: bool = False,
        family_filter: str | None = None,
        allow_no_db: bool = False,
    ) -> None:
        self.load()
        self._require_loaded()

        runtime_env = get_environment()
        enabled = self.registry.enabled_strategies(live=runtime_env == "live")
        logger.info(
            "Runtime shell loaded %d enabled strategies across %d connection groups (env=%s)%s",
            len(enabled),
            len(self.registry.connection_groups),
            runtime_env,
            " in shadow mode" if shadow else "",
        )

        # ------------------------------------------------------------------
        # 0. Filter by family if requested (before preflight)
        # ------------------------------------------------------------------
        if family_filter:
            enabled = [m for m in enabled if m.family == family_filter]
            if not enabled:
                raise RuntimeError(f"No enabled strategies for family={family_filter!r}")
            logger.info("Family filter active: running %d strategies for '%s'", len(enabled), family_filter)

        # ------------------------------------------------------------------
        # 1. Async preflight (fail-fast before heavy startup)
        # ------------------------------------------------------------------
        enabled_families = {m.family for m in enabled}
        checks = await self._run_async_preflight(
            connect_ib=connect_ib,
            families=enabled_families,
        )
        for c in checks:
            lvl = logging.INFO if c.ok else logging.WARNING
            logger.log(lvl, "PREFLIGHT %s: %s -- %s", "OK" if c.ok else "FAIL", c.name, c.detail)
        critical_failures = [
            c for c in checks
            if not c.ok and c.name.split(":")[0] in ("import", "database", "ib-gateway")
        ]
        if critical_failures:
            for c in critical_failures:
                logger.error("PREFLIGHT FAIL: %s -- %s", c.name, c.detail)
            raise RuntimeError(f"Preflight failed: {len(critical_failures)} critical check(s)")

        # ------------------------------------------------------------------
        # 2. Connect broker
        # ------------------------------------------------------------------
        if connect_ib:
            strategy_group_map = {
                manifest.strategy_id: manifest.connection_group for manifest in enabled
            }
            self.session = UnifiedIBSession(self.registry.connection_groups, strategy_group_map)
            await self.session.start()
            await self.session.wait_ready()
            logger.info("Unified IB session connected for all configured groups")
            await self.session.verify_streaming_data()

        if once:
            return

        # ------------------------------------------------------------------
        # 3. Bootstrap database
        # ------------------------------------------------------------------
        db_pool = None
        account_gate = None
        trade_recorder = None
        heartbeat = None
        try:
            from libs.services.bootstrap import bootstrap_database
            bootstrap_ctx = await bootstrap_database()
            db_pool = bootstrap_ctx.pool
            trade_recorder = bootstrap_ctx.trade_recorder
            heartbeat = bootstrap_ctx.heartbeat
            logger.info("Database bootstrapped")
        except Exception as exc:
            if allow_no_db:
                logger.warning("Database bootstrap failed (--allow-no-db): %s", exc)
            else:
                raise RuntimeError(
                    f"Database bootstrap failed (portfolio rules require DB). "
                    f"Use --allow-no-db to start without DB. Error: {exc}"
                ) from exc

        if db_pool is not None:
            try:
                from libs.risk.account_risk_gate import AccountRiskGate
                _account_urd = float(os.environ.get("ACCOUNT_UNIT_RISK_DOLLARS", "200"))
                account_gate = AccountRiskGate(db_pool, account_urd=_account_urd)
            except Exception as exc:
                logger.warning("AccountRiskGate init failed (non-fatal): %s", exc)

        # ------------------------------------------------------------------
        # 3.5  Regime service (non-fatal)
        # ------------------------------------------------------------------
        regime_service = None
        if connect_ib and self.session:
            try:
                from regime.live import RegimeService
                from libs.config.market_calendar import MarketCalendar
                regime_service = RegimeService(
                    ib_session=self.session,
                    market_calendar=MarketCalendar(),
                )
                await regime_service.start()
                logger.info("Regime service started: %s", regime_service.get_context())
            except Exception as exc:
                logger.warning("Regime service init failed (non-fatal): %s", exc)

        # ------------------------------------------------------------------
        # 4. Group strategies by family and build coordinators
        # ------------------------------------------------------------------
        families: dict[str, list] = {}
        for manifest in enabled:
            families.setdefault(manifest.family, []).append(manifest)

        coordinators: list[Any] = []
        for family, manifests in families.items():
            if family not in _FAMILY_COORDINATORS:
                logger.error("No coordinator registered for family '%s', skipping", family)
                continue

            # Build RuntimeContext for this family
            ctx = RuntimeContext(
                manifest=manifests[0],  # primary manifest (coordinator reads all from registry)
                registry=self.registry,
                portfolio=self.portfolio,
                session=self.session,
                market_data=None,
                oms=None,  # coordinators build their own OMS
                state_store=None,
                instrumentation=None,
                contracts=self.contracts,
                health={},
                logger=logging.getLogger(f"runtime.{family}"),
                clock=None,
                db_pool=db_pool,
                account_gate=account_gate,
                family_coordinator=None,
                regime_service=regime_service,
                trade_recorder=trade_recorder,
                heartbeat=heartbeat,
            )

            try:
                coordinator_cls = _import_coordinator(family)
                coordinator = coordinator_cls(ctx)
                coordinators.append(coordinator)
                logger.info(
                    "Coordinator created for family '%s' (%d strategies)",
                    family, len(manifests),
                )
            except Exception as exc:
                logger.error("Failed to create coordinator for '%s': %s", family, exc, exc_info=True)

        # ------------------------------------------------------------------
        # 5. Start all coordinators (must run before regime apply so that
        #    _base_portfolio_rules is initialised inside start())
        # ------------------------------------------------------------------
        started_coordinators: list[Any] = []
        for coordinator in coordinators:
            try:
                await coordinator.start()
                started_coordinators.append(coordinator)
                logger.info("Family '%s' coordinator started", coordinator.family_id)
            except Exception as exc:
                logger.error(
                    "Coordinator '%s' failed to start: %s",
                    getattr(coordinator, "family_id", "?"), exc, exc_info=True,
                )
        coordinators = started_coordinators

        # ------------------------------------------------------------------
        # 5b. Load and apply initial regime context AFTER coordinators started
        # ------------------------------------------------------------------
        regime_task: asyncio.Task | None = None
        try:
            regime_ctx = None
            if regime_service is not None:
                regime_ctx = regime_service.get_context()
            if regime_ctx is None:
                from regime.persistence import load_regime_context
                regime_ctx = load_regime_context()
            for coordinator in coordinators:
                if hasattr(coordinator, "apply_regime"):
                    try:
                        coordinator.apply_regime(regime_ctx)
                    except Exception as exc:
                        logger.error("Initial regime apply failed for %s: %s",
                                    getattr(coordinator, "family_id", "?"), exc)
            logger.info("Regime context applied: regime=%s, confidence=%.3f, computed_at=%s",
                        regime_ctx.regime, regime_ctx.regime_confidence, regime_ctx.computed_at or "unknown")
        except Exception as exc:
            logger.warning("Regime context load failed (non-fatal): %s", exc)

        if not coordinators:
            logger.error("No coordinators started successfully — shutting down")
            if db_pool is not None:
                await db_pool.close()
            if self.session is not None:
                await self.session.stop()
            return

        # ------------------------------------------------------------------
        # 6. Run until shutdown signal
        # ------------------------------------------------------------------
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass  # Windows

        active_families = [getattr(c, "family_id", "?") for c in coordinators]
        logger.info("Runtime active — families: %s — press Ctrl+C to stop", active_families)

        # 6b. Start weekly regime refresh task
        async def _regime_refresh_loop() -> None:
            """Reload regime context weekly. Checks hourly, refreshes Friday 17:00+ ET."""
            from zoneinfo import ZoneInfo
            ET = ZoneInfo("America/New_York")
            last_refresh_date = None
            while True:
                await asyncio.sleep(3600)
                if stop_event.is_set():
                    break
                now_et = datetime.now(ET)
                today = now_et.date()
                if now_et.weekday() != 4 or now_et.hour < 17:
                    continue
                if last_refresh_date == today:
                    continue
                last_refresh_date = today
                try:
                    # Prefer live service context over disk
                    ctx = None
                    if regime_service is not None:
                        ctx = regime_service.get_context()
                    if ctx is None:
                        from regime.persistence import load_regime_context
                        ctx = load_regime_context()

                    # Staleness circuit breaker: escalate to ERROR if >14 days old
                    if ctx.computed_at:
                        try:
                            from datetime import timezone as _tz
                            age = datetime.now(_tz.utc) - datetime.fromisoformat(ctx.computed_at)
                            if age.days > 14:
                                logger.error(
                                    "Regime context is %d days stale (computed_at=%s) -- "
                                    "check RegimeService or data pipeline",
                                    age.days, ctx.computed_at,
                                )
                        except (ValueError, TypeError):
                            pass

                    for coordinator in coordinators:
                        if hasattr(coordinator, "apply_regime"):
                            try:
                                coordinator.apply_regime(ctx)
                            except Exception as exc:
                                logger.error("Regime refresh failed for %s: %s",
                                            getattr(coordinator, "family_id", "?"), exc)
                    logger.info("Weekly regime refresh: %s (confidence=%.3f, computed_at=%s)",
                                ctx.regime, ctx.regime_confidence, ctx.computed_at or "unknown")
                except Exception as exc:
                    logger.error("Regime refresh loop error: %s", exc)

        regime_task = asyncio.create_task(_regime_refresh_loop(), name="regime_refresh")

        try:
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

        # ------------------------------------------------------------------
        # 7. Graceful shutdown (reverse order)
        # ------------------------------------------------------------------
        logger.info("Shutting down ...")

        if regime_task is not None:
            regime_task.cancel()
            with suppress(asyncio.CancelledError):
                await regime_task

        if regime_service is not None:
            try:
                await regime_service.stop()
                logger.info("Regime service stopped")
            except Exception as exc:
                logger.warning("Regime service stop error: %s", exc)

        for coordinator in reversed(coordinators):
            try:
                await coordinator.stop()
                logger.info("Family '%s' coordinator stopped", getattr(coordinator, "family_id", "?"))
            except Exception as exc:
                logger.warning("Coordinator stop error: %s", exc, exc_info=True)

        if db_pool is not None:
            try:
                await db_pool.close()
                logger.info("Database pool closed")
            except Exception as exc:
                logger.warning("DB pool close error: %s", exc)

        if self.session is not None:
            await self.session.stop()
            logger.info("IB session disconnected")

        logger.info("Runtime shutdown complete")
