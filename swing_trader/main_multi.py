"""Unified multi-strategy launcher — runs all 5 strategies in one process.

Strategies (by priority): ATRSS(0), S5_PB(1), S5_DUAL(2), Breakout(3), Helix(4).
Shares a single IBKR adapter, OMS, and StrategyCoordinator across all strategies.
Cross-strategy coordination rules (tighten Helix stop on ATRSS entry, size boost)
are implemented via the shared coordinator.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("multi_strategy")


async def main() -> None:
    """Wire up shared IB session, multi-strategy OMS, and start all engines."""
    from shared.ibkr_core.config.loader import IBKRConfig
    from shared.ibkr_core.client.session import IBSession
    from shared.ibkr_core.mapping.contract_factory import ContractFactory
    from shared.ibkr_core.adapters.execution_adapter import IBKRExecutionAdapter
    from shared.oms.services.factory import build_multi_strategy_oms
    from shared.oms.risk.calculator import RiskCalculator
    from shared.services.bootstrap import bootstrap_database
    from shared.market_calendar import MarketCalendar

    # Strategy imports
    from strategy.config import (
        STRATEGY_ID as ATRSS_ID,
        SYMBOL_CONFIGS as ATRSS_CONFIGS,
        build_instruments as atrss_build_instruments,
    )
    from strategy.engine import ATRSSEngine

    from strategy_2.config import (
        STRATEGY_ID as HELIX_ID,
        SYMBOL_CONFIGS as HELIX_CONFIGS,
        build_instruments as helix_build_instruments,
    )
    from strategy_2.engine import HelixEngine

    from strategy_3.config import (
        STRATEGY_ID as BREAKOUT_ID,
        SYMBOL_CONFIGS as BREAKOUT_CONFIGS,
        build_instruments as breakout_build_instruments,
    )
    from strategy_3.engine import BreakoutEngine

    from strategy_4.config import (
        S5_PB_STRATEGY_ID, S5_DUAL_STRATEGY_ID,
        S5_PB_CONFIGS, S5_DUAL_CONFIGS,
        build_instruments as s5_build_instruments,
    )
    from strategy_4.engine import KeltnerEngine

    from shared.overlay.config import OverlayConfig
    from shared.overlay.engine import OverlayEngine

    # -------------------------------------------------------------------
    # 1. Load IBKR configuration
    # -------------------------------------------------------------------
    config_dir = Path(__file__).resolve().parent / "config"
    ibkr_config = IBKRConfig(config_dir)
    logger.info(
        "Loaded IBKR config: host=%s port=%d",
        ibkr_config.profile.host, ibkr_config.profile.port,
    )

    # -------------------------------------------------------------------
    # 2. Create and connect IB session (shared)
    # -------------------------------------------------------------------
    session = IBSession(ibkr_config)
    await session.start()
    await session.wait_ready()
    logger.info("IB session connected")

    # -------------------------------------------------------------------
    # 3. Create execution adapter (shared)
    # -------------------------------------------------------------------
    contract_factory = ContractFactory(
        ib=session.ib,
        templates=ibkr_config.contracts,
        routes=ibkr_config.routes,
    )
    adapter = IBKRExecutionAdapter(
        session=session,
        contract_factory=contract_factory,
        account=ibkr_config.profile.account_id,
    )

    # -------------------------------------------------------------------
    # 4. Bootstrap database (graceful degradation)
    # -------------------------------------------------------------------
    bootstrap_ctx = await bootstrap_database()
    trade_recorder = bootstrap_ctx.trade_recorder

    # -------------------------------------------------------------------
    # 4b. Bootstrap instrumentation (per-strategy kits)
    # -------------------------------------------------------------------
    instrumentation_ctx = None
    atrss_kit = None
    helix_kit = None
    breakout_kit = None
    s5_pb_kit = None
    s5_dual_kit = None
    try:
        from instrumentation.src.bootstrap import bootstrap_instrumentation, bootstrap_kit

        # Shared context for overlay engine and periodic tasks (daily snapshots, backfill)
        all_symbols = sorted(set(
            list(ATRSS_CONFIGS) + list(HELIX_CONFIGS) + list(BREAKOUT_CONFIGS)
            + list(S5_PB_CONFIGS) + list(S5_DUAL_CONFIGS)
        ))
        instrumentation_ctx = bootstrap_instrumentation(symbols=all_symbols)
        logger.info("Instrumentation bootstrapped for %s", all_symbols)

        # Per-strategy InstrumentationKits — share the parent context's services
        # (snapshot, sidecar, regime classifier, etc.) but get own TradeLogger/MissedLogger
        atrss_kit = bootstrap_kit(
            strategy_id=ATRSS_ID,
            shared_ctx=instrumentation_ctx,
        )
        logger.info("ATRSS InstrumentationKit bootstrapped")

        helix_kit = bootstrap_kit(
            strategy_id=HELIX_ID,
            shared_ctx=instrumentation_ctx,
        )
        logger.info("AKC_HELIX InstrumentationKit bootstrapped")

        breakout_kit = bootstrap_kit(
            strategy_id=BREAKOUT_ID,
            shared_ctx=instrumentation_ctx,
        )
        logger.info("SWING_BREAKOUT_V3 InstrumentationKit bootstrapped")

        s5_pb_kit = bootstrap_kit(
            strategy_id=S5_PB_STRATEGY_ID,
            shared_ctx=instrumentation_ctx,
        )
        logger.info("S5_PB InstrumentationKit bootstrapped")

        s5_dual_kit = bootstrap_kit(
            strategy_id=S5_DUAL_STRATEGY_ID,
            shared_ctx=instrumentation_ctx,
        )
        logger.info("S5_DUAL InstrumentationKit bootstrapped")
    except Exception:
        logger.warning("Instrumentation bootstrap failed — running without instrumentation", exc_info=True)

    # -------------------------------------------------------------------
    # 5. Register instruments from all strategies (union)
    # -------------------------------------------------------------------
    atrss_instruments = atrss_build_instruments()
    helix_instruments = helix_build_instruments()
    breakout_instruments = breakout_build_instruments()
    s5_pb_instruments = s5_build_instruments(S5_PB_CONFIGS)
    s5_dual_instruments = s5_build_instruments(S5_DUAL_CONFIGS)
    # Merge all instruments (InstrumentRegistry is global singleton)
    all_instruments = {**atrss_instruments, **helix_instruments,
                       **breakout_instruments, **s5_pb_instruments,
                       **s5_dual_instruments}
    logger.info(
        "Registered instruments: ATRSS=%s, Helix=%s, Breakout=%s, S5_PB=%s, S5_DUAL=%s",
        list(atrss_instruments), list(helix_instruments), list(breakout_instruments),
        list(s5_pb_instruments), list(s5_dual_instruments),
    )

    # -------------------------------------------------------------------
    # 6. Fetch account equity
    # -------------------------------------------------------------------
    equity = 100_000.0  # fallback default
    actual_net_liq = equity
    try:
        accounts = session.ib.managedAccounts()
        if accounts:
            summary = await session.ib.accountSummaryAsync(accounts[0])
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    actual_net_liq = float(item.value)
                    equity = actual_net_liq
                    logger.info("Account equity from IB: $%.2f", equity)
                    break
    except Exception:
        logger.warning("Could not fetch equity from IB, using default $%.2f", equity)

    # Paper capital: use virtual equity if PAPER_CAPITAL is set.
    # Virtual equity = PAPER_CAPITAL + (actual_net_liq - baseline).
    # Baseline is the actual IBKR net liq when paper mode first started,
    # persisted so that equity grows/shrinks with real P&L across restarts.
    paper_equity_offset = 0.0
    _paper_capital_raw = os.environ.get("PAPER_CAPITAL", "")
    if _paper_capital_raw:
        _env_mode = os.environ.get("SWING_TRADER_ENV", "").lower()
        if _env_mode not in ("paper", "dev"):
            logger.error(
                "PAPER_CAPITAL is set but SWING_TRADER_ENV=%s (not paper/dev) "
                "— ignoring to protect live account",
                _env_mode,
            )
        elif float(_paper_capital_raw) <= 0:
            logger.warning("PAPER_CAPITAL must be > 0 (got %s), ignoring", _paper_capital_raw)
        else:
            paper_capital = float(_paper_capital_raw)
            state_file = config_dir / "paper_capital_state.json"
            baseline = None
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        _state = json.load(f)
                    baseline = _state["baseline_net_liq"]
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.warning(
                        "Corrupt paper_capital_state.json (%s), re-creating baseline", e,
                    )
            if baseline is None:
                baseline = actual_net_liq
                state_file.write_text(json.dumps({
                    "baseline_net_liq": baseline,
                    "paper_capital": paper_capital,
                }, indent=2))
                logger.info(
                    "Paper capital baseline recorded: $%.2f (actual account)",
                    baseline,
                )
            equity = paper_capital + (actual_net_liq - baseline)
            paper_equity_offset = equity - actual_net_liq
            logger.info(
                "Paper capital mode: equity=$%.2f "
                "(start=$%.2f, actual=$%.2f, baseline=$%.2f, offset=%+.2f)",
                equity, paper_capital, actual_net_liq, baseline, paper_equity_offset,
            )

    # -------------------------------------------------------------------
    # 6b. Overlay configuration
    # -------------------------------------------------------------------
    overlay_config = OverlayConfig(
        enabled=True,
        symbols=["QQQ", "GLD"],
        max_equity_pct=0.85,
        ema_fast=13,
        ema_slow=48,
        ema_overrides={"QQQ": (10, 21), "GLD": (13, 21)},
        weights=None,
        state_file=str(Path(__file__).resolve().parent / "overlay_state.json"),
    )

    # -------------------------------------------------------------------
    # 7. Compute per-strategy unit_risk_dollars
    # -------------------------------------------------------------------
    atrss_urd = RiskCalculator.compute_unit_risk_dollars(
        nav=equity, unit_risk_pct=0.012,  # 1.2% base risk (optimized_v1)
    )
    helix_urd = RiskCalculator.compute_unit_risk_dollars(
        nav=equity, unit_risk_pct=0.005,  # 0.50% base risk per Helix spec
    )
    breakout_urd = RiskCalculator.compute_unit_risk_dollars(
        nav=equity, unit_risk_pct=0.005,  # 0.50% base risk per Breakout spec
    )
    s5_pb_urd = RiskCalculator.compute_unit_risk_dollars(
        nav=equity, unit_risk_pct=0.008,  # 0.80% base risk per S5_PB spec
    )
    s5_dual_urd = RiskCalculator.compute_unit_risk_dollars(
        nav=equity, unit_risk_pct=0.008,  # 0.80% base risk per S5_DUAL spec
    )

    # -------------------------------------------------------------------
    # 8. Build shared multi-strategy OMS
    # -------------------------------------------------------------------
    market_cal = MarketCalendar()

    oms, coordinator = await build_multi_strategy_oms(
        adapter=adapter,
        strategies=[
            {
                "id": ATRSS_ID,
                "unit_risk_dollars": atrss_urd,
                "daily_stop_R": 2.0,
                "priority": 0,       # highest expectancy
                "max_heat_R": 1.00,
                "max_working_orders": 4,
            },
            {
                "id": S5_PB_STRATEGY_ID,
                "unit_risk_dollars": s5_pb_urd,
                "daily_stop_R": 2.0,
                "priority": 1,       # 80% WR on IBIT (optimized_v2)
                "max_heat_R": 1.50,
                "max_working_orders": 2,
            },
            {
                "id": S5_DUAL_STRATEGY_ID,
                "unit_risk_dollars": s5_dual_urd,
                "daily_stop_R": 2.0,
                "priority": 2,       # 70.7% WR on GLD+IBIT (optimized_v2)
                "max_heat_R": 1.50,
                "max_working_orders": 2,
            },
            {
                "id": BREAKOUT_ID,
                "unit_risk_dollars": breakout_urd,
                "daily_stop_R": 2.0,
                "priority": 3,       # rare signals (3 trades), priority barely matters
                "max_heat_R": 0.65,
                "max_working_orders": 2,
            },
            {
                "id": HELIX_ID,
                "unit_risk_dollars": helix_urd,
                "daily_stop_R": 2.5,
                "priority": 4,       # 34% WR, high stale-exit rate — lowest priority
                "max_heat_R": 0.85,
                "max_working_orders": 4,
            },
        ],
        heat_cap_R=2.0,  # expanded (optimized_v2)
        portfolio_daily_stop_R=3.0,
        db_pool=bootstrap_ctx.pool,
        market_calendar=market_cal,
    )

    # -------------------------------------------------------------------
    # 8b. Wire coordinator action logger
    # -------------------------------------------------------------------
    if instrumentation_ctx and getattr(instrumentation_ctx, 'coordination_logger', None):
        coordinator.set_action_logger(instrumentation_ctx.coordination_logger.log_action)
        logger.info("Coordinator action logger wired")

    # -------------------------------------------------------------------
    # 9. Start OMS
    # -------------------------------------------------------------------
    await oms.start()
    logger.info("Multi-strategy OMS started")

    # Wire post-reconnect OMS reconciliation
    session.set_reconnect_callback(oms._reconciler.on_reconnect_reconciliation)
    logger.info("Post-reconnect reconciliation callback wired")

    # Start instrumentation sidecar (background thread)
    if instrumentation_ctx is not None:
        try:
            instrumentation_ctx.start()
        except Exception:
            logger.warning("Instrumentation start failed", exc_info=True)

    # Per-strategy kits share the parent context's sidecar — no separate start needed

    # -------------------------------------------------------------------
    # 10. Create strategy engines (shared OMS, coordinator)
    # -------------------------------------------------------------------
    atrss_engine = ATRSSEngine(
        ib_session=session,
        oms_service=oms,
        instruments=atrss_instruments,
        config=ATRSS_CONFIGS,
        trade_recorder=trade_recorder,
        equity=equity,
        market_calendar=market_cal,
        kit=atrss_kit,
        equity_offset=paper_equity_offset,
    )

    helix_engine = HelixEngine(
        ib_session=session,
        oms_service=oms,
        instruments=helix_instruments,
        config=HELIX_CONFIGS,
        trade_recorder=trade_recorder,
        equity=equity,
        coordinator=coordinator,
        market_calendar=market_cal,
        instrumentation_kit=helix_kit,
        equity_offset=paper_equity_offset,
    )

    breakout_engine = BreakoutEngine(
        ib_session=session,
        oms_service=oms,
        instruments=breakout_instruments,
        config=BREAKOUT_CONFIGS,
        trade_recorder=trade_recorder,
        equity=equity,
        market_calendar=market_cal,
        instrumentation=breakout_kit,
        equity_offset=paper_equity_offset,
    )

    s5_pb_engine = KeltnerEngine(
        strategy_id=S5_PB_STRATEGY_ID,
        ib_session=session,
        oms_service=oms,
        instruments=s5_pb_instruments,
        config=S5_PB_CONFIGS,
        trade_recorder=trade_recorder,
        equity=equity,
        market_calendar=market_cal,
        kit=s5_pb_kit,
        equity_offset=paper_equity_offset,
    )

    s5_dual_engine = KeltnerEngine(
        strategy_id=S5_DUAL_STRATEGY_ID,
        ib_session=session,
        oms_service=oms,
        instruments=s5_dual_instruments,
        config=S5_DUAL_CONFIGS,
        trade_recorder=trade_recorder,
        equity=equity,
        market_calendar=market_cal,
        kit=s5_dual_kit,
        equity_offset=paper_equity_offset,
    )

    overlay_engine = OverlayEngine(
        ib_session=session,
        equity=equity,
        config=overlay_config,
        market_calendar=market_cal,
        instrumentation=instrumentation_ctx,
        equity_offset=paper_equity_offset,
    )

    # -------------------------------------------------------------------
    # 10b. Wire overlay state provider to all kits
    # -------------------------------------------------------------------
    if overlay_config.enabled:
        overlay_state_fn = overlay_engine.get_signals
        if instrumentation_ctx is not None:
            instrumentation_ctx.overlay_state_provider = overlay_state_fn
        for kit_obj in [atrss_kit, helix_kit, breakout_kit, s5_pb_kit, s5_dual_kit]:
            if kit_obj is not None:
                kit_obj.ctx.overlay_state_provider = overlay_state_fn

    # -------------------------------------------------------------------
    # 11. Start all engines
    # -------------------------------------------------------------------
    await atrss_engine.start()
    logger.info("ATRSS engine started (priority 0, symbols: %s)", list(ATRSS_CONFIGS))
    await s5_pb_engine.start()
    logger.info("S5_PB engine started (priority 1, symbols: %s)", list(S5_PB_CONFIGS))
    await s5_dual_engine.start()
    logger.info("S5_DUAL engine started (priority 2, symbols: %s)", list(S5_DUAL_CONFIGS))
    await breakout_engine.start()
    logger.info("Breakout engine started (priority 3, symbols: %s)", list(BREAKOUT_CONFIGS))
    await helix_engine.start()
    logger.info("Helix engine started (priority 4, symbols: %s)", list(HELIX_CONFIGS))

    if overlay_config.enabled:
        await overlay_engine.start()
        logger.info("Overlay engine started (symbols: %s, max_pct: %.0f%%)",
                     overlay_config.symbols, overlay_config.max_equity_pct * 100)

    # -------------------------------------------------------------------
    # 11b. Launch instrumentation periodic tasks
    # -------------------------------------------------------------------
    _daily_snapshot_task = None
    _backfill_task = None
    _heartbeat_task = None
    _config_check_task = None
    _post_exit_task = None

    if instrumentation_ctx is not None:
        async def _run_daily_snapshot() -> None:
            """Build + save daily snapshot at 16:05 ET each trading day."""
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            while True:
                try:
                    now_et = datetime.now(timezone.utc).astimezone(et)
                    # Next 16:05 ET
                    target = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
                    if target <= now_et:
                        target += timedelta(days=1)
                    # Skip weekends
                    while target.weekday() >= 5:
                        target += timedelta(days=1)
                    delay = (target - now_et).total_seconds()
                    await asyncio.sleep(delay)
                    try:
                        snap = instrumentation_ctx.daily_builder.build()
                        instrumentation_ctx.daily_builder.save(snap)
                        logger.info("Daily instrumentation snapshot saved")
                    except Exception:
                        logger.warning("Daily snapshot build failed", exc_info=True)
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.warning("Daily snapshot task error", exc_info=True)
                    await asyncio.sleep(300)

        # Create IBKR historical provider for backfill operations
        _ibkr_provider = None
        try:
            from instrumentation.src.ibkr_provider import IBKRHistoricalProvider
            _ibkr_provider = IBKRHistoricalProvider(
                ib=session.ib,
                contract_factory=contract_factory,
                loop=asyncio.get_running_loop(),
            )
        except Exception:
            logger.warning("IBKR historical provider init failed — backfills degraded")

        async def _run_backfill() -> None:
            """Run missed-opportunity backfill every 5 minutes."""
            loop = asyncio.get_running_loop()
            while True:
                try:
                    await asyncio.sleep(300)
                    try:
                        await loop.run_in_executor(
                            None,
                            instrumentation_ctx.missed_logger.run_backfill,
                            _ibkr_provider,
                        )
                    except Exception:
                        logger.debug("Backfill cycle error", exc_info=True)
                except asyncio.CancelledError:
                    break

        async def _run_heartbeat() -> None:
            """Emit per-strategy heartbeats every 60 seconds.

            Each strategy kit emits its own heartbeat so TA monitoring sees
            all 5 strategies as alive (not just ATRSS).
            """
            import time as _time
            start_time = _time.monotonic()
            error_counter = 0

            # Map each kit to its engine(s) for position/order counting
            _kit_engines = [
                (atrss_kit, {
                    "positions": [("positions", atrss_engine)],
                    "orders": [("pending_orders", atrss_engine)],
                }),
                (helix_kit, {
                    "positions": [("active_setups", helix_engine)],
                    "orders": [("pending_setups", helix_engine)],
                }),
                (breakout_kit, {
                    "positions": [("active_setups", breakout_engine)],
                    "orders": [],
                }),
                (s5_pb_kit, {
                    "positions": [("positions", s5_pb_engine)],
                    "orders": [("_pending_entry", s5_pb_engine)],
                }),
                (s5_dual_kit, {
                    "positions": [("positions", s5_dual_engine)],
                    "orders": [("_pending_entry", s5_dual_engine)],
                }),
            ]

            while True:
                try:
                    await asyncio.sleep(60)
                    uptime = _time.monotonic() - start_time

                    for kit_obj, engine_map in _kit_engines:
                        if kit_obj is None:
                            continue
                        n_pos = sum(
                            len(getattr(eng, attr, {}))
                            for attr, eng in engine_map["positions"]
                        )
                        n_orders = sum(
                            len(getattr(eng, attr, {}))
                            for attr, eng in engine_map["orders"]
                        )
                        kit_obj.emit_heartbeat(
                            active_positions=n_pos,
                            open_orders=n_orders,
                            uptime_s=uptime,
                            error_count_1h=error_counter,
                        )
                except asyncio.CancelledError:
                    break
                except Exception:
                    error_counter += 1
                    await asyncio.sleep(60)

        async def _run_config_check() -> None:
            """Check for parameter changes every 5 minutes."""
            while True:
                try:
                    await asyncio.sleep(300)
                    for kit_obj in [atrss_kit, helix_kit, breakout_kit, s5_pb_kit, s5_dual_kit]:
                        if kit_obj is not None:
                            kit_obj.check_config_changes()

                    # Emit coordinator-level portfolio snapshot + filter decisions
                    if atrss_kit is not None:
                        try:
                            _pos_counts = {
                                "atrss": len(getattr(atrss_engine, "positions", {})),
                                "helix": len(getattr(helix_engine, "active_setups", {})),
                                "breakout": len(getattr(breakout_engine, "active_setups", {})),
                                "s5_pb": len(getattr(s5_pb_engine, "positions", {})),
                                "s5_dual": len(getattr(s5_dual_engine, "positions", {})),
                            }
                            n_positions = sum(_pos_counts.values())

                            atrss_kit.on_indicator_snapshot(
                                pair="PORTFOLIO",
                                indicators={
                                    "concurrent_positions": float(n_positions),
                                    "atrss_positions": float(_pos_counts["atrss"]),
                                    "helix_setups": float(_pos_counts["helix"]),
                                    "breakout_setups": float(_pos_counts["breakout"]),
                                    "s5_pb_positions": float(_pos_counts["s5_pb"]),
                                    "s5_dual_positions": float(_pos_counts["s5_dual"]),
                                },
                                signal_name="coordinator_risk_check",
                                signal_strength=0.0,
                                decision="skip" if n_positions >= 10 else "enter",
                                strategy_id="COORDINATOR",
                            )

                            # Coordinator filter decisions
                            # heat_cap_R=2.0 is OMS-internal; use position count as proxy
                            _POSITION_CAP = 10
                            atrss_kit.on_filter_decision(
                                pair="PORTFOLIO",
                                filter_name="portfolio_heat_cap",
                                passed=n_positions < _POSITION_CAP,
                                threshold=float(_POSITION_CAP),
                                actual_value=float(n_positions),
                                signal_name="coordinator_entry_check",
                                coordinator_triggered=True,
                                strategy_id="COORDINATOR",
                            )

                            # Per-strategy heat checks
                            _STRATEGY_CAPS = {
                                "atrss": 4, "helix": 4, "breakout": 2,
                                "s5_pb": 2, "s5_dual": 2,
                            }
                            for _sid, _count in _pos_counts.items():
                                _cap = _STRATEGY_CAPS.get(_sid, 4)
                                atrss_kit.on_filter_decision(
                                    pair="PORTFOLIO",
                                    filter_name=f"strategy_heat_cap_{_sid}",
                                    passed=_count < _cap,
                                    threshold=float(_cap),
                                    actual_value=float(_count),
                                    signal_name="coordinator_entry_check",
                                    coordinator_triggered=True,
                                    strategy_id="COORDINATOR",
                                )
                        except Exception:
                            pass  # coordinator emission is best-effort
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.debug("Config check task error", exc_info=True)
                    await asyncio.sleep(300)

        async def _run_post_exit_backfill() -> None:
            """Backfill post-exit prices every 30 minutes."""
            from instrumentation.src.post_exit_tracker import PostExitTracker
            if _ibkr_provider is None:
                logger.warning("No IBKR provider — post-exit backfill disabled")
                return
            tracker = PostExitTracker(instrumentation_ctx.data_dir, _ibkr_provider)
            loop = asyncio.get_running_loop()
            while True:
                try:
                    await asyncio.sleep(1800)
                    try:
                        await loop.run_in_executor(None, tracker.run_backfill)
                    except Exception:
                        logger.debug("Post-exit backfill error", exc_info=True)
                except asyncio.CancelledError:
                    break

        _daily_snapshot_task = asyncio.create_task(_run_daily_snapshot())
        _backfill_task = asyncio.create_task(_run_backfill())
        _heartbeat_task = asyncio.create_task(_run_heartbeat())
        _config_check_task = asyncio.create_task(_run_config_check())
        _post_exit_task = asyncio.create_task(_run_post_exit_backfill())
        logger.info("Instrumentation periodic tasks started")

    # -------------------------------------------------------------------
    # 12. Run until interrupted
    # -------------------------------------------------------------------
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows does not support add_signal_handler
            pass

    logger.info(
        "Multi-strategy runner active — ATRSS + S5_PB + S5_DUAL + Breakout + Helix + Overlay — press Ctrl+C to stop"
    )

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # -------------------------------------------------------------------
    # 13. Graceful shutdown
    # M7: Correct ordering: engines → OMS → database → broker connection
    # Engines must stop first (cancel pending orders, stop scheduling).
    # OMS stops next (drain queues, flush state).
    # Database closes after OMS has flushed.
    # Broker connection closes last.
    # -------------------------------------------------------------------
    logger.info("Shutting down …")

    # 0. Cancel instrumentation periodic tasks
    if _daily_snapshot_task is not None:
        _daily_snapshot_task.cancel()
    if _backfill_task is not None:
        _backfill_task.cancel()
    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
    if _config_check_task is not None:
        _config_check_task.cancel()
    if _post_exit_task is not None:
        _post_exit_task.cancel()

    # 0b. Build final daily snapshot before engines stop
    if instrumentation_ctx is not None:
        try:
            snap = instrumentation_ctx.daily_builder.build()
            instrumentation_ctx.daily_builder.save(snap)
            logger.info("Final daily instrumentation snapshot saved")
        except Exception:
            logger.debug("Final daily snapshot failed", exc_info=True)

    # 1. Stop overlay engine (independent of OMS, stop before strategy engines)
    if overlay_config.enabled:
        await overlay_engine.stop()
        logger.info("Overlay engine stopped")

    # 1b. Stop strategy engines (highest level — stop generating intents)
    await breakout_engine.stop()
    await helix_engine.stop()
    await s5_dual_engine.stop()
    await s5_pb_engine.stop()
    await atrss_engine.stop()
    logger.info("All strategy engines stopped")

    # 2. Stop OMS (drain execution queue, flush pending state)
    await oms.stop()
    logger.info("OMS stopped")

    # 2b. Stop instrumentation (sidecar thread)
    if instrumentation_ctx is not None:
        try:
            instrumentation_ctx.stop()
            logger.info("Instrumentation stopped")
        except Exception:
            logger.debug("Instrumentation stop failed", exc_info=True)

    # Per-strategy kits share the parent sidecar — no separate stop needed

    # 3. Close database (after OMS has flushed all state)
    if bootstrap_ctx.has_db:
        from shared.services.bootstrap import shutdown_database
        await shutdown_database(bootstrap_ctx)
        logger.info("Database shutdown complete")

    # 4. Disconnect broker (last — no more messages to send/receive)
    await session.stop()
    logger.info("IB session disconnected")

    logger.info("Multi-strategy shutdown complete")


def _setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    _setup_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
