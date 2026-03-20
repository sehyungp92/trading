"""Live bootstrap for the U.S. ORB runtime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from datetime import date, datetime, time, timezone
from pathlib import Path

from instrumentation.src.strategy_data_providers import USORBInstrumentationDataProvider
from shared.services.heartbeat import emit_heartbeat

from .config import ET, STRATEGY_ID, StrategySettings
from .data import IBMarketDataSource, IBScannerSource
from .diagnostics import JsonlDiagnostics
from .engine import USORBEngine
from .session import bootstrap_runtime, shutdown_runtime
from .universe_builder import LiveUniverseBuilder

logger = logging.getLogger(__name__)


async def run_intraday_session(
    trading_date: date | None = None,
    settings: StrategySettings | None = None,
) -> None:
    settings = settings or StrategySettings()
    config_dir = Path(__file__).resolve().parents[1] / "config"
    services = await bootstrap_runtime(config_dir=config_dir, settings=settings)
    cache: dict = {}
    builder = LiveUniverseBuilder(
        ib=services.session.ib, cache=cache, cache_dir=settings.cache_dir,
    )
    diagnostics = JsonlDiagnostics(settings.diagnostics_dir)

    # Optional pre-market warmup
    et_now = datetime.now(ET)
    if et_now.time() < settings.scanner_start:
        try:
            await builder.pre_market_warmup()
        except Exception:
            logger.exception("Pre-market warmup failed — continuing with empty cache")

    engine = USORBEngine(
        oms_service=services.oms,
        cache=cache,
        account_id=services.account_id,
        nav=services.allocated_nav,
        settings=settings,
        trade_recorder=services.trade_recorder,
        diagnostics=diagnostics,
        instrumentation=services.instrumentation_kit,
    )
    # Reconcile OMS vs broker positions on startup to catch orphans from prior crashes
    try:
        await engine._reconcile_after_reconnect()
    except Exception:
        logger.exception("Startup position reconciliation failed — continuing")

    scanner = IBScannerSource(services.session.ib, settings.scanner)
    market_data = IBMarketDataSource(
        ib=services.session.ib,
        contract_factory=services.contract_factory,
        on_quote=engine.on_quote,
        on_bar=engine.on_bar,
    )

    async def _on_reconnect() -> None:
        await engine._reconcile_after_reconnect()
        logger.info("Restarting scanner subscriptions after reconnect")
        await scanner.restart()
        market_data.invalidate_subscriptions()
        logger.info("Market data subscriptions invalidated — scanner_loop will re-subscribe")

    services.session.set_reconnect_callback(_on_reconnect)
    stop_event = asyncio.Event()
    session_date = trading_date or datetime.now(ET).date()
    started_at = datetime.now(timezone.utc)

    def _record_runtime_error(error_type: str, exc: Exception, *, severity: str = "medium") -> None:
        if services.instrumentation_kit is None:
            return
        services.instrumentation_kit.log_error(
            error_type=error_type,
            message=str(exc),
            severity=severity,
            category="dependency",
            context={"component": "strategy_orb.main"},
            exc=exc,
        )

    async def scanner_loop() -> None:
        while not stop_event.is_set():
            now_et = datetime.now(ET)
            if now_et.date() != session_date or now_et.time() >= time(16, 1):
                logger.info("Date boundary reached — ending ORB session")
                stop_event.set()
                break
            await services.session.wait_until_ready()
            try:
                symbols = await asyncio.wait_for(scanner.next_update(), timeout=5.0)
                builder.resolve_batch(symbols)
                engine.update_scanner_symbols(symbols, datetime.now(timezone.utc))
            except asyncio.TimeoutError:
                symbols = scanner.latest_symbols()
                if symbols:
                    builder.resolve_batch(symbols)
                    engine.update_scanner_symbols(symbols, datetime.now(timezone.utc))
            except Exception as exc:
                logger.exception("Scanner loop failed")
                _record_runtime_error("scanner_loop_error", exc, severity="high")
                await asyncio.sleep(5)
                continue
            try:
                await market_data.ensure_symbols(engine.subscription_instruments())
            except Exception as exc:
                logger.exception("Market data subscription refresh failed")
                _record_runtime_error("market_data_subscription_error", exc, severity="high")
                await asyncio.sleep(5)

    async def heartbeat_loop() -> None:
        while True:
            try:
                risk_state = await services.oms.get_strategy_risk(STRATEGY_ID)
                heat_r = float(getattr(risk_state, "open_risk_R", 0.0))
                daily_pnl_r = float(getattr(risk_state, "daily_realized_R", 0.0))
                sidecar_diagnostics = (
                    services.instrumentation.get_sidecar_diagnostics()
                    if services.instrumentation is not None
                    else None
                )
                if services.bootstrap.has_db:
                    await emit_heartbeat(
                        services.bootstrap.pg_store,
                        STRATEGY_ID,
                        heat_r=heat_r,
                        daily_pnl_r=daily_pnl_r,
                        mode="RUNNING",
                        sidecar_diagnostics=sidecar_diagnostics,
                    )
                if services.instrumentation_kit is not None:
                    positions = engine.get_position_snapshot()
                    gross_notional = sum(
                        abs(float(pos.get("entry_price", 0.0) or 0.0) * float(pos.get("qty", 0.0) or 0.0))
                        for pos in positions
                    )
                    exposure_pct = (
                        (gross_notional / services.allocated_nav) * 100.0
                        if services.allocated_nav > 0
                        else 0.0
                    )
                    services.instrumentation_kit.emit_heartbeat(
                        active_positions=len(positions),
                        open_orders=engine.open_order_count(),
                        uptime_s=(datetime.now(timezone.utc) - started_at).total_seconds(),
                        error_count_1h=(
                            services.instrumentation.recent_error_count_1h()
                            if services.instrumentation is not None
                            else 0
                        ),
                        positions=positions,
                        portfolio_exposure={
                            "total_positions": len(positions),
                            "open_orders": engine.open_order_count(),
                            "heat_r": heat_r,
                            "daily_pnl_r": daily_pnl_r,
                            "gross_notional": round(gross_notional, 2),
                            "allocated_nav": round(services.allocated_nav, 2),
                            "exposure_pct": round(exposure_pct, 4),
                            "by_strategy": {
                                STRATEGY_ID: {
                                    "positions": len(positions),
                                    "open_orders": engine.open_order_count(),
                                }
                            },
                        },
                    )
            except Exception as exc:
                logger.exception("Heartbeat failed")
                _record_runtime_error("heartbeat_loop_error", exc)
            await asyncio.sleep(30)

    scanner_task: asyncio.Task | None = None
    heartbeat_task: asyncio.Task | None = None

    def _handle_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_stop)

    try:
        if services.instrumentation is not None:
            services.instrumentation.attach_data_provider(USORBInstrumentationDataProvider(engine))
            await services.instrumentation.start()

        await engine.start()
        await market_data.start()
        await market_data.ensure_symbols(engine.subscription_instruments())
        await scanner.start()
        scanner_task = asyncio.create_task(scanner_loop())
        heartbeat_task = asyncio.create_task(heartbeat_loop())

        while not stop_event.is_set():
            now_et = datetime.now(ET)
            if now_et.date() != session_date or now_et.time() >= time(16, 1):
                logger.info("Session date boundary — shutting down")
                stop_event.set()
                break
            await asyncio.sleep(5)
    finally:
        if scanner_task is not None:
            scanner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scanner_task
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        await scanner.stop()
        await market_data.stop()
        await engine.stop()
        await shutdown_runtime(services)


async def main() -> None:
    settings = StrategySettings()
    logger.info("ORB strategy started — scanner %s, entry %s–%s, flatten %s ET",
                settings.scanner_start, settings.entry_start,
                settings.entry_end, settings.forced_flatten)
    while True:
        now = datetime.now(ET)
        today = now.date()
        try:
            if settings.scanner_start <= now.time() < time(16, 1):
                await run_intraday_session(trading_date=today, settings=settings)
        except Exception:
            logger.exception("ORB main loop failed")
        await asyncio.sleep(5)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
