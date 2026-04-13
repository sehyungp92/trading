"""Swing family coordinator — orchestrates 6 strategies sharing one OMS.

Strategies by priority: ATRSS(0), S5_PB(1), S5_DUAL(2), Breakout(3), Helix(4), BRS_R9(5).
Shares a single IBKR adapter, multi-strategy OMS, StrategyCoordinator, and
OverlayEngine across all engines.

Cross-strategy coordination:
  - ATRSS entry fill on symbol X -> tighten Helix stop to breakeven on X
  - has_atrss_position() -> Helix 1.25x size boost when ATRSS confirms direction
  - BRS_R9 SHORT entry on symbol X -> tighten all LONG stops on X to breakeven

Extracted from swing_trader/main_multi.py (826 lines) into a clean coordinator
that receives its dependencies via RuntimeContext.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from typing import Any

from strategies.contracts import RuntimeContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-strategy risk parameters (unit_risk_pct passed to RiskCalculator)
# ---------------------------------------------------------------------------
_RISK_PARAMS: dict[str, dict[str, Any]] = {
    "ATRSS": {
        "unit_risk_pct": 0.018,   # 1.8% base risk (P1 heat-unlock optimized)
        "daily_stop_R": 2.0,
        "priority": 0,            # highest expectancy
        "max_heat_R": 1.50,
        "max_working_orders": 4,
    },
    "S5_PB": {
        "unit_risk_pct": 0.012,   # 1.2% base risk (P1 heat-unlock optimized)
        "daily_stop_R": 2.0,
        "priority": 1,            # 80% WR on IBIT (optimized_v2)
        "max_heat_R": 2.00,       # greedy v4: was 1.50
        "max_working_orders": 2,
    },
    "S5_DUAL": {
        "unit_risk_pct": 0.012,   # 1.2% base risk (P1 heat-unlock optimized)
        "daily_stop_R": 2.0,
        "priority": 2,            # 70.7% WR on GLD+IBIT (optimized_v2)
        "max_heat_R": 2.00,       # greedy v4: was 1.50
        "max_working_orders": 2,
    },
    "SWING_BREAKOUT_V3": {
        "unit_risk_pct": 0.008,   # 0.8% base risk (P1 heat-unlock optimized)
        "daily_stop_R": 2.0,
        "priority": 3,            # rare signals, priority barely matters
        "max_heat_R": 1.50,       # greedy v4: was 1.00
        "max_working_orders": 2,
    },
    "AKC_HELIX": {
        "unit_risk_pct": 0.008,   # 0.8% base risk (P1 heat-unlock optimized)
        "daily_stop_R": 2.5,
        "priority": 4,            # 34% WR, high stale-exit rate
        "max_heat_R": 1.20,
        "max_working_orders": 4,
    },
    "BRS_R9": {
        "unit_risk_pct": 0.003,   # 0.3% base risk (GLD rate, higher of the two)
        "daily_stop_R": 2.0,
        "priority": 5,            # lowest -- episodic bear specialist
        "max_heat_R": 1.25,       # tighter than standalone 1.80 for shared portfolio
        "max_working_orders": 2,
    },
}

# Portfolio-level risk caps (P1 heat-unlock optimized)
_HEAT_CAP_R = 3.0
_PORTFOLIO_DAILY_STOP_R = 4.0


class SwingFamilyCoordinator:
    """Orchestrates 6 swing strategies sharing one OMS instance.

    Lifecycle:
        coordinator = SwingFamilyCoordinator(ctx)
        await coordinator.start()
        ...
        await coordinator.stop()
    """

    family_id = "swing"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engines: list[tuple[str, Any]] = []  # (strategy_id, engine)
        self._oms: Any = None
        self._coordinator: Any = None  # StrategyCoordinator
        self._overlay_engine: Any = None
        self._instrumentation_ctx: Any = None
        self._kits: dict[str, Any] = {}
        self._portfolio_checker: Any = None
        self._base_portfolio_rules: Any = None
        self._regime_ctx: Any = None
        self._heartbeat_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build shared OMS, create engines, wire overlay, and start."""
        # -- Lazy imports (keep module-level lightweight) ------------------
        from libs.oms.services.factory import build_multi_strategy_oms
        from libs.oms.risk.calculator import RiskCalculator
        from libs.config.capital_bootstrap import bootstrap_capital
        from libs.risk import AccountRiskGate

        from strategies.swing.atrss.config import (
            STRATEGY_ID as ATRSS_ID,
            SYMBOL_CONFIGS as ATRSS_CONFIGS,
            build_instruments as atrss_build_instruments,
        )
        from strategies.swing.atrss.engine import ATRSSEngine

        from strategies.swing.akc_helix.config import (
            STRATEGY_ID as HELIX_ID,
            SYMBOL_CONFIGS as HELIX_CONFIGS,
            build_instruments as helix_build_instruments,
        )
        from strategies.swing.akc_helix.engine import HelixEngine

        from strategies.swing.breakout.config import (
            STRATEGY_ID as BREAKOUT_ID,
            SYMBOL_CONFIGS as BREAKOUT_CONFIGS,
            build_instruments as breakout_build_instruments,
        )
        from strategies.swing.breakout.engine import BreakoutEngine

        from strategies.swing.keltner.config import (
            S5_PB_STRATEGY_ID,
            S5_DUAL_STRATEGY_ID,
            S5_PB_CONFIGS,
            S5_DUAL_CONFIGS,
            build_instruments as s5_build_instruments,
        )
        from strategies.swing.keltner.engine import KeltnerEngine

        from strategies.swing.brs.config import (
            STRATEGY_ID as BRS_ID,
            BRS_SYMBOL_DEFAULTS as BRS_CONFIGS,
            build_instruments as brs_build_instruments,
        )
        from strategies.swing.brs.engine import BRSLiveEngine
        from strategies.swing.brs.models import BRSRegime

        ctx = self._ctx
        session = ctx.session
        db_pool = getattr(ctx, "db_pool", None)

        # -- Config dir (needed for adapter + capital bootstrap) -----------
        config_dir = Path(
            os.environ.get(
                "CONFIG_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "config"),
            )
        )

        # -- Build execution adapter (same pattern as momentum/stock) ------
        adapter = None
        contract_factory = None
        if session is not None:
            from libs.broker_ibkr.config.loader import IBKRConfig
            from libs.broker_ibkr.mapping.contract_factory import ContractFactory
            from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter

            try:
                ibkr_config = IBKRConfig(config_dir)
            except Exception as exc:
                logger.error(
                    "IBKRConfig load failed (%s) — swing adapter unavailable. "
                    "Ensure config/ibkr_profiles.yaml exists.",
                    exc,
                )
                ibkr_config = None

            if ibkr_config is not None:
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
                logger.info("Swing execution adapter created (account=%s)", ibkr_config.profile.account_id)
            else:
                logger.warning("IBKRConfig unavailable — swing running without execution adapter")
        else:
            logger.warning("No IB session -- swing adapter is None (shadow/test mode)")

        # -- Market calendar -----------------------------------------------
        from libs.config.market_calendar import MarketCalendar
        market_cal = MarketCalendar()

        # -- Equity & paper capital ----------------------------------------
        equity: float = getattr(ctx.portfolio, "allocated_nav", None) or getattr(
            ctx.portfolio, "nav", 100_000.0
        )
        paper_equity_offset: float = getattr(ctx, "paper_equity_offset", 0.0)

        # -- Capital allocation per strategy -------------------------------
        allocs: dict[str, Any] | None = None
        try:
            allocs = bootstrap_capital(equity, config_dir)
            swing_nav = {
                sid: allocs[sid].allocated_nav
                for sid in allocs
                if getattr(allocs[sid], "family", "") == "swing"
            }
            logger.info(
                "Capital allocation (swing family): %s",
                {k: f"${v:,.2f}" for k, v in swing_nav.items()},
            )
        except Exception as exc:
            logger.warning("bootstrap_capital failed (%s), falling back to full equity", exc)

        def _nav_for(strategy_id: str) -> float:
            if allocs and strategy_id in allocs:
                return allocs[strategy_id].allocated_nav
            return equity

        # -- Compute unit risk dollars per strategy ------------------------
        strategy_ids = [ATRSS_ID, S5_PB_STRATEGY_ID, S5_DUAL_STRATEGY_ID, BREAKOUT_ID, HELIX_ID, BRS_ID]
        urds: dict[str, float] = {}
        for sid in strategy_ids:
            params = _RISK_PARAMS[sid]
            urds[sid] = RiskCalculator.compute_unit_risk_dollars(
                nav=_nav_for(sid),
                unit_risk_pct=params["unit_risk_pct"],
            )

        # -- Build shared multi-strategy OMS -------------------------------
        account_gate = AccountRiskGate(db_pool) if db_pool else None

        # Portfolio rules: directional cap + symbol collision for swing family
        from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
        portfolio_rules = PortfolioRulesConfig(
            directional_cap_R=6.0,
            initial_equity=equity,
            family_strategy_ids=tuple(strategy_ids),
            symbol_collision_action="half_size",
            helix_nqdtc_cooldown_minutes=0,  # disable momentum-specific rules
            nqdtc_direction_filter_enabled=False,
        )

        self._live_equity = [equity]
        self._oms, self._coordinator = await build_multi_strategy_oms(
            adapter=adapter,
            strategies=[
                {
                    "id": sid,
                    "unit_risk_dollars": urds[sid],
                    "daily_stop_R": _RISK_PARAMS[sid]["daily_stop_R"],
                    "priority": _RISK_PARAMS[sid]["priority"],
                    "max_heat_R": _RISK_PARAMS[sid]["max_heat_R"],
                    "max_working_orders": _RISK_PARAMS[sid]["max_working_orders"],
                }
                for sid in strategy_ids
            ],
            heat_cap_R=_HEAT_CAP_R,
            portfolio_daily_stop_R=_PORTFOLIO_DAILY_STOP_R,
            db_pool=db_pool,
            market_calendar=market_cal,
            family_id=self.family_id,
            account_gate=account_gate,
            portfolio_rules_config=portfolio_rules,
            get_current_equity=lambda: self._live_equity[0],
            live_equity=self._live_equity,
        )
        self._portfolio_checker = getattr(self._oms, '_portfolio_checker', None)
        self._base_portfolio_rules = portfolio_rules

        # -- Wire coordinator action logger --------------------------------
        instrumentation_ctx = getattr(ctx, "instrumentation", None)
        self._instrumentation_ctx = instrumentation_ctx
        if instrumentation_ctx and getattr(instrumentation_ctx, "coordination_logger", None):
            self._coordinator.set_action_logger(
                instrumentation_ctx.coordination_logger.log_action
            )
            logger.info("Coordinator action logger wired")

        # -- Start OMS -----------------------------------------------------
        await self._oms.start()
        logger.info("Multi-strategy OMS started")

        # Wire post-reconnect reconciliation
        if hasattr(session, "set_reconnect_callback") and hasattr(self._oms, "_reconciler"):
            session.set_reconnect_callback(self._oms._reconciler.on_reconnect_reconciliation)
            logger.info("Post-reconnect reconciliation callback wired")

        # -- Bootstrap instrumentation kits --------------------------------
        _data_provider = None
        try:
            import asyncio as _asyncio
            from .instrumentation.src.ibkr_provider import IBKRHistoricalProvider
            _ib = getattr(session, "ib", None)
            _loop = _asyncio.get_running_loop()
            if _ib is not None:
                _data_provider = IBKRHistoricalProvider(
                    ib=_ib,
                    contract_factory=contract_factory,
                    loop=_loop,
                )
                logger.info("IBKRHistoricalProvider created for post-exit backfill")
        except Exception:
            logger.debug("IBKRHistoricalProvider creation skipped", exc_info=True)

        self._kits = self._bootstrap_instrumentation_kits(
            strategy_ids,
            {
                ATRSS_ID: ATRSS_CONFIGS,
                HELIX_ID: HELIX_CONFIGS,
                BREAKOUT_ID: BREAKOUT_CONFIGS,
                S5_PB_STRATEGY_ID: S5_PB_CONFIGS,
                S5_DUAL_STRATEGY_ID: S5_DUAL_CONFIGS,
                BRS_ID: BRS_CONFIGS,
            },
            data_provider=_data_provider,
        )

        # -- Build instruments per strategy --------------------------------
        atrss_instruments = atrss_build_instruments()
        helix_instruments = helix_build_instruments()
        breakout_instruments = breakout_build_instruments()
        s5_pb_instruments = s5_build_instruments(S5_PB_STRATEGY_ID)
        s5_dual_instruments = s5_build_instruments(S5_DUAL_STRATEGY_ID)

        # -- Trade recorder (from bootstrap context) -----------------------
        trade_recorder = getattr(ctx, "trade_recorder", None)
        if trade_recorder is None and instrumentation_ctx is not None:
            trade_recorder = getattr(instrumentation_ctx, "trade_recorder", None)

        # -- Create strategy engines (all sharing single OMS) --------------
        oms = self._oms
        coordinator = self._coordinator

        atrss_engine = ATRSSEngine(
            ib_session=session,
            oms_service=oms,
            instruments=atrss_instruments,
            config=ATRSS_CONFIGS,
            trade_recorder=trade_recorder,
            equity=equity,
            market_calendar=market_cal,
            kit=self._kits.get(ATRSS_ID),
            equity_offset=paper_equity_offset,
        )

        helix_engine = HelixEngine(
            ib_session=session,
            oms_service=oms,
            instruments=helix_instruments,
            config=HELIX_CONFIGS,
            trade_recorder=trade_recorder,
            equity=equity,
            coordinator=coordinator,  # enables ATRSS->Helix cross-strategy rules
            market_calendar=market_cal,
            instrumentation_kit=self._kits.get(HELIX_ID),
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
            instrumentation=self._kits.get(BREAKOUT_ID),
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
            kit=self._kits.get(S5_PB_STRATEGY_ID),
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
            kit=self._kits.get(S5_DUAL_STRATEGY_ID),
            equity_offset=paper_equity_offset,
        )

        brs_instruments = brs_build_instruments()

        brs_engine = BRSLiveEngine(
            ib_session=session,
            oms_service=oms,
            instruments=brs_instruments,
            trade_recorder=trade_recorder,
            equity=equity,
            instrumentation=self._kits.get(BRS_ID),
        )

        # Store engines in priority order (ATRSS first, BRS last)
        self._engines = [
            (ATRSS_ID, atrss_engine),
            (S5_PB_STRATEGY_ID, s5_pb_engine),
            (S5_DUAL_STRATEGY_ID, s5_dual_engine),
            (BREAKOUT_ID, breakout_engine),
            (HELIX_ID, helix_engine),
            (BRS_ID, brs_engine),
        ]

        # -- Create OverlayEngine (idle-capital EMA crossover) -------------
        self._overlay_engine = self._create_overlay_engine(
            session=session,
            equity=equity,
            market_cal=market_cal,
            paper_equity_offset=paper_equity_offset,
        )

        # -- Wire BRS bear regime check to overlay engine -------------------
        if self._overlay_engine is not None:
            def _brs_bear_strong() -> bool:
                try:
                    for sym in ("QQQ", "GLD"):
                        dctx = brs_engine._daily_ctx.get(sym)
                        if dctx and dctx.regime == BRSRegime.BEAR_STRONG:
                            return True
                except Exception:
                    pass  # engine not ready yet — default to non-bear
                return False
            self._overlay_engine.set_bear_regime_check(_brs_bear_strong)

        # -- Wire overlay state provider to all instrumentation kits -------
        if self._overlay_engine is not None:
            overlay_state_fn = self._overlay_engine.get_signals
            if instrumentation_ctx is not None:
                instrumentation_ctx.overlay_state_provider = overlay_state_fn
            for kit in self._kits.values():
                if kit is not None and hasattr(kit, "ctx"):
                    kit.ctx.overlay_state_provider = overlay_state_fn

        # -- Start engines in priority order (ATRSS first, BRS last) --------
        for sid, engine in self._engines:
            await engine.start()
            logger.info("%s engine started (priority %d)", sid, _RISK_PARAMS[sid]["priority"])

        # -- Start overlay engine ------------------------------------------
        if self._overlay_engine is not None:
            await self._overlay_engine.start()
            logger.info("Overlay engine started")

        # -- Heartbeat background task --------------------------------------
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            "Swing family coordinator active -- %d engines + overlay",
            len(self._engines),
        )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Graceful shutdown: overlay -> engines (reverse priority) -> OMS."""
        # 0. Stop heartbeat loop
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        # 1. Stop overlay engine first (independent of OMS)
        if self._overlay_engine is not None:
            try:
                await self._overlay_engine.stop()
                logger.info("Overlay engine stopped")
            except Exception:
                logger.debug("Overlay engine stop failed", exc_info=True)

        # 2. Stop strategy engines in reverse priority (BRS -> ... -> ATRSS)
        for sid, engine in reversed(self._engines):
            try:
                await engine.stop()
                logger.info("%s engine stopped", sid)
            except Exception:
                logger.warning("Failed to stop %s engine", sid, exc_info=True)
        logger.info("All strategy engines stopped")

        # 3. Stop instrumentation sidecar
        if self._instrumentation_ctx is not None:
            try:
                self._instrumentation_ctx.stop()
                logger.info("Instrumentation stopped")
            except Exception:
                logger.debug("Instrumentation stop failed", exc_info=True)

        # 4. Stop OMS last (drain execution queue, flush pending state)
        if self._oms is not None:
            try:
                await self._oms.stop()
                logger.info("OMS stopped")
            except Exception:
                logger.warning("OMS stop failed", exc_info=True)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_status(self) -> dict[str, Any]:
        """Return health of all engines + OMS + overlay."""
        status: dict[str, Any] = {
            "family_id": self.family_id,
            "oms_running": getattr(self._oms, "_running", False) if self._oms else False,
            "overlay_running": (
                getattr(self._overlay_engine, "_running", False)
                if self._overlay_engine
                else False
            ),
            "engines": {},
        }
        for sid, engine in self._engines:
            if hasattr(engine, "health_status"):
                status["engines"][sid] = engine.health_status()
            else:
                status["engines"][sid] = {
                    "strategy_id": sid,
                    "running": getattr(engine, "_running", False),
                }
        return status

    def apply_regime(self, ctx: "RegimeContext") -> None:
        """Apply regime context to swing portfolio rules and overlay weights."""
        from regime.integration import build_swing_rules, OVERLAY_WEIGHTS

        if self._base_portfolio_rules is None:
            logger.warning("apply_regime called before start() — skipping")
            return

        prev_regime = getattr(self._regime_ctx, "regime", None)
        self._regime_ctx = ctx

        # Tier 1: portfolio rules
        if self._portfolio_checker is not None:
            new_rules = build_swing_rules(ctx, self._base_portfolio_rules)
            self._portfolio_checker.update_config(new_rules)

        # Overlay QQQ/GLD capital split from regime
        from regime.integration import _validated_regime
        regime = _validated_regime(ctx.regime)
        if self._overlay_engine is not None:
            self._overlay_engine._config.weights = dict(OVERLAY_WEIGHTS[regime])

        changed = f" (was {prev_regime})" if prev_regime and prev_regime != ctx.regime else ""
        logger.info("Swing regime applied: %s%s (cap=%.1fR, risk=%.2fx, overlay=%s)",
                    ctx.regime, changed,
                    self._portfolio_checker._cfg.directional_cap_R if self._portfolio_checker else 0,
                    self._portfolio_checker._cfg.regime_unit_risk_mult if self._portfolio_checker else 1,
                    OVERLAY_WEIGHTS[regime])

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        heartbeat = getattr(self._ctx, "heartbeat", None)
        if heartbeat is None:
            return
        session = self._ctx.session
        while True:
            try:
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                return
            rs = getattr(self._oms, "_portfolio_risk_state", None)
            srs = getattr(self._oms, "_strategy_risk_states", {})
            for sid, _engine in self._engines:
                try:
                    sr = srs.get(sid)
                    await heartbeat.strategy_heartbeat(
                        strategy_id=sid,
                        heat_r=Decimal(str(sr.open_risk_R)) if sr else Decimal("0"),
                        daily_pnl_r=Decimal(str(sr.daily_realized_R)) if sr else Decimal("0"),
                        mode="HALTED" if (rs and rs.halted) else "RUNNING",
                    )
                except Exception:
                    logger.debug("Heartbeat failed for %s", sid, exc_info=True)
            try:
                connected = session.ib.isConnected() if session else False
                await heartbeat.adapter_heartbeat(self.family_id, connected=connected)
            except Exception:
                logger.debug("Adapter heartbeat failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap_instrumentation_kits(
        self,
        strategy_ids: list[str],
        config_maps: dict[str, dict],
        data_provider=None,
    ) -> dict[str, Any]:
        """Create per-strategy InstrumentationKits (graceful degradation)."""
        kits: dict[str, Any] = {}
        try:
            from .instrumentation.src.bootstrap import (
                bootstrap_instrumentation,
                bootstrap_kit,
            )

            all_symbols = sorted({
                sym for configs in config_maps.values() for sym in configs
            })
            self._instrumentation_ctx = bootstrap_instrumentation(
                symbols=all_symbols, data_provider=data_provider,
                get_regime_ctx=lambda: self._regime_ctx,
                get_applied_config=lambda: self._portfolio_checker._cfg if self._portfolio_checker else None,
            )
            logger.info("Instrumentation bootstrapped for %s", all_symbols)

            for sid in strategy_ids:
                kits[sid] = bootstrap_kit(
                    strategy_id=sid,
                    shared_ctx=self._instrumentation_ctx,
                )
                logger.info("%s InstrumentationKit bootstrapped", sid)

            # Start instrumentation sidecar (background thread)
            try:
                self._instrumentation_ctx.start()
            except Exception:
                logger.debug("Instrumentation sidecar start failed", exc_info=True)

        except ImportError:
            logger.info("Instrumentation not available — running without kits")
        except Exception:
            logger.warning("Instrumentation bootstrap failed", exc_info=True)

        return kits

    def _get_swing_deployed_capital(self) -> float:
        """Return approximate notional capital deployed by swing strategies."""
        if not hasattr(self, "_oms") or self._oms is None:
            return 0.0
        try:
            risk_state = getattr(self._oms, "_portfolio_risk_state", None)
            if risk_state is not None:
                return risk_state.open_risk_dollars
        except Exception:
            pass
        return 0.0

    def _create_overlay_engine(
        self,
        session: Any,
        equity: float,
        market_cal: Any,
        paper_equity_offset: float,
    ) -> Any | None:
        """Create OverlayEngine for idle-capital EMA crossover (QQQ, GLD)."""
        try:
            from strategies.swing.overlay.config import OverlayConfig
            from strategies.swing.overlay.engine import OverlayEngine

            overlay_config = OverlayConfig()

            if not overlay_config.enabled:
                return None

            engine = OverlayEngine(
                ib_session=session,
                equity=equity,
                config=overlay_config,
                market_calendar=market_cal,
                equity_offset=paper_equity_offset,
                get_deployed_capital=self._get_swing_deployed_capital,
            )
            return engine

        except ImportError:
            logger.info("OverlayEngine not available — overlay disabled")
            return None
        except Exception:
            logger.warning("OverlayEngine creation failed", exc_info=True)
            return None
