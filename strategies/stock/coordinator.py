"""Stock family coordinator — each strategy gets its own OMS instance.

Like momentum (per-strategy OMS), unlike swing (shared OMS).
Three strategies:
  - IARIC_v1   (IARICEngine)   — WatchlistArtifact
  - US_ORB_v1  (USORBEngine)   — cache dict
  - ALCB_v1    (ALCBT2Engine)  — CandidateArtifact

Stock-specific differences from momentum:
  - portfolio_rules with family-scoped directional cap + symbol collision guard
  - Paper equity NAV tracking via resolve_paper_nav / capital_bootstrap
  - Artifacts or cache dicts instead of instrument dicts
  - IBMarketDataSource per engine, wired via on_bar/on_quote callbacks
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from libs.oms.persistence.db_config import get_environment

from strategies.contracts import RuntimeContext

logger = logging.getLogger(__name__)


class StockFamilyCoordinator:
    """Lifecycle manager for the three stock strategies.

    Each strategy receives its own OMS built via ``build_oms_service``.
    Shared across all three: *db_pool*, *AccountRiskGate*, *family_id*.
    """

    family_id = "stock"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engines: list = []
        self._oms_services: list = []
        self._instrumentations: list = []
        self._strategy_ids: list[str] = []
        self._market_data_sources: list = []
        self._market_data_task: asyncio.Task | None = None
        self._contract_factory = None
        self._portfolio_checkers: list = []
        self._engine_map: dict[str, Any] = {}
        self._base_portfolio_rules: Any = None
        self._regime_ctx: Any = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Import, build OMS, and start all three stock engines."""
        from libs.oms.services.factory import build_oms_service
        from libs.oms.risk.calculator import RiskCalculator
        from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
        from libs.config.capital_bootstrap import bootstrap_capital

        ctx = self._ctx
        session = ctx.session
        db_pool = ctx.db_pool
        account_gate = ctx.account_gate

        # Resolve paper-mode equity
        paper_mode = get_environment() == "paper"
        equity: float

        if paper_mode:
            from libs.persistence.paper_equity import PaperEquityManager
            pem = PaperEquityManager(db_pool, account_scope=self.family_id, initial_equity=100_000.0)
            equity = await pem.load()
            logger.info("Paper mode equity for stock family: $%.2f", equity)
        else:
            equity = 100_000.0
            try:
                accounts = session.ib.managedAccounts()
                if accounts:
                    summary = await session.ib.accountSummaryAsync(accounts[0])
                    for item in summary:
                        if item.tag == "NetLiquidation" and item.currency == "USD":
                            equity = float(item.value)
                            logger.info("Account equity: $%.2f", equity)
                            break
            except Exception:
                logger.warning("Using default equity $%.2f", equity)

        # Capital allocation per strategy
        config_dir = Path(
            os.environ.get(
                "CONFIG_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "config"),
            )
        )

        try:
            allocs = bootstrap_capital(equity, config_dir)
        except Exception as exc:
            logger.warning("bootstrap_capital failed (%s), using equal split", exc)
            allocs = {}

        # Load artifacts for strategies that need them
        artifacts = self._load_artifacts()

        # ── Strategy descriptors ─────────────────────────────────────
        _strategies = self._build_strategy_descriptors(artifacts)

        # Portfolio rules: drawdown tiers + family-scoped directional cap + symbol collision
        all_strategy_ids = tuple(d["strategy_id"] for d in _strategies)
        logger.info(
            "Stock portfolio rules: directional_cap=%.1fR, collision=%s, strategies=%s",
            12.0, "half_size", all_strategy_ids,
        )

        # Log dollar-equivalent directional cap per strategy for monitoring
        for desc in _strategies:
            _alloc = allocs.get(desc["strategy_id"])
            _nav = _alloc.allocated_nav if _alloc else equity
            _unit = RiskCalculator.compute_unit_risk_dollars(nav=_nav, unit_risk_pct=desc["base_risk_pct"])
            logger.info(
                "Directional cap dollar-equiv for %s: 1R=$%.0f, cap=12R=$%.0f, heat=%sR=$%.0f",
                desc["strategy_id"], _unit, 12.0 * _unit,
                desc["heat_cap_R"], desc["heat_cap_R"] * _unit,
            )

        for desc in _strategies:
            sid = desc["strategy_id"]
            self._strategy_ids.append(sid)

            # Resolve allocated NAV
            alloc = allocs.get(sid)
            allocated_nav = alloc.allocated_nav if alloc else equity

            # Per-strategy portfolio rules — initial_equity must match the
            # get_current_equity callback (allocated_nav), otherwise drawdown
            # tiers see a phantom 67% DD and halt every entry.
            portfolio_rules = PortfolioRulesConfig(
                directional_cap_R=12.0,
                initial_equity=allocated_nav,
                family_strategy_ids=all_strategy_ids,
                symbol_collision_action="half_size",
                strategy_priorities=(
                    ("IARIC_v1", 0),    # highest: daily alpha, multi-position (5R heat)
                    ("ALCB_v1", 1),     # second: proven backtest edge, broader alpha
                    ("US_ORB_v1", 2),   # lowest: unproven, pure OR overlap with ALCB
                ),
                priority_headroom_R=4.0,       # reserve last 4R for IARIC + ALCB
                priority_reserve_threshold=1,  # priority 0-1 can use reserved headroom
            )
            # Save first portfolio_rules as base template for regime updates
            if self._base_portfolio_rules is None:
                self._base_portfolio_rules = portfolio_rules

            if alloc:
                logger.info(
                    "Capital allocation: %s -> $%.2f (%.1f%% of $%.2f)",
                    sid, allocated_nav, alloc.capital_pct, equity,
                )
            else:
                logger.warning(
                    "Strategy %s not in unified config, using full equity", sid,
                )

            # Risk parameters
            unit_risk = RiskCalculator.compute_unit_risk_dollars(
                nav=allocated_nav, unit_risk_pct=desc["base_risk_pct"],
            )
            _live_equity = [allocated_nav]

            # Build per-strategy OMS with portfolio rules
            oms = await build_oms_service(
                adapter=desc["adapter"](session),
                strategy_id=sid,
                unit_risk_dollars=unit_risk,
                daily_stop_R=desc["daily_stop_R"],
                heat_cap_R=desc["heat_cap_R"],
                portfolio_daily_stop_R=desc["portfolio_daily_stop_R"],
                db_pool=db_pool,
                portfolio_rules_config=portfolio_rules,
                get_current_equity=lambda eq=_live_equity: eq[0],
                paper_equity_pool=db_pool if paper_mode else None,
                family_id=self.family_id,
                account_gate=account_gate,
            )
            await oms.start()
            logger.info("OMS started for %s", sid)
            self._oms_services.append(oms)
            self._portfolio_checkers.append(getattr(oms, '_portfolio_checker', None))

            # Instrumentation (non-fatal)
            instr = None
            try:
                from .instrumentation.src.bootstrap import InstrumentationManager
                instr = InstrumentationManager(
                    oms, sid, strategy_type=desc["instr_type"],
                )
                await instr.start()
            except Exception as exc:
                logger.warning(
                    "Instrumentation init failed for %s (non-fatal): %s", sid, exc,
                )
            self._instrumentations.append(instr)

            # Trade recorder
            trade_recorder = desc["trade_recorder"]
            try:
                from .instrumentation.src.facade import InstrumentationKit
                from .instrumentation.src.pg_bridge import InstrumentedTradeRecorder
                if instr is not None:
                    kit = InstrumentationKit(instr, strategy_type=desc["instr_type"])
                    trade_recorder = InstrumentedTradeRecorder(
                            trade_recorder,
                            kit,
                            strategy_id=sid,
                            strategy_type=desc["instr_type"],
                        )
            except Exception as exc:
                logger.warning(
                    "InstrumentedTradeRecorder setup failed for %s (non-fatal): %s",
                    sid, exc,
                )

            # Diagnostics
            diagnostics = desc["diagnostics_factory"]()

            # Build engine
            engine_cls = desc["engine_cls"]()
            engine_kwargs = dict(
                oms_service=oms,
                account_id=desc["account_id"],
                nav=allocated_nav,
                settings=desc["settings"],
                trade_recorder=trade_recorder,
                diagnostics=diagnostics,
                instrumentation=instr,
            )
            # Strategy-specific data source (artifact or cache)
            engine_kwargs[desc["data_key"]] = desc["data_value"]

            engine = engine_cls(**engine_kwargs)
            await engine.start()
            logger.info("Engine started for %s", sid)
            self._engines.append(engine)
            self._engine_map[sid] = engine

            # ── Market data source per engine ──────────────────────────
            md_source = None
            if self._contract_factory is not None:
                try:
                    MarketDataCls = desc["market_data_cls"]()
                    md_source = MarketDataCls(
                        ib=session.ib,
                        contract_factory=self._contract_factory,
                        on_quote=engine.on_quote,
                        on_bar=engine.on_bar,
                    )
                    await md_source.start()
                    # Initial subscription setup
                    if hasattr(engine, "subscription_instruments"):
                        await md_source.ensure_hot_symbols(engine.subscription_instruments())
                    if hasattr(engine, "polling_instruments"):
                        await md_source.poll_due_bars(engine.polling_instruments())
                    logger.info("Market data started for %s", sid)
                except Exception as exc:
                    logger.error("Market data init failed for %s: %s", sid, exc, exc_info=exc)
            else:
                logger.warning("No contract factory — market data NOT wired for %s", sid)
            self._market_data_sources.append(md_source)

        # ── Reconnect callback ─────────────────────────────────────
        async def _on_reconnect() -> None:
            for i, eng in enumerate(self._engines):
                if hasattr(eng, "_reconcile_after_reconnect"):
                    try:
                        await eng._reconcile_after_reconnect()
                    except Exception as exc:
                        logger.error("Reconnect reconciliation failed for %s: %s",
                                     self._strategy_ids[i], exc)
                md = self._market_data_sources[i] if i < len(self._market_data_sources) else None
                if md is not None and hasattr(md, "invalidate_subscriptions"):
                    md.invalidate_subscriptions()
            logger.info("Post-reconnect: reconciliation + subscription invalidation complete")

        if hasattr(session, "set_reconnect_callback"):
            session.set_reconnect_callback(_on_reconnect)

        # ── Background market data refresh loop ────────────────────
        self._market_data_task = asyncio.create_task(self._market_data_loop())

        logger.info(
            "StockFamilyCoordinator started %d strategies with market data", len(self._engines),
        )

    async def stop(self) -> None:
        """Stop market data, engines, and OMS instances in reverse order."""
        # Stop background market data loop
        if self._market_data_task is not None:
            self._market_data_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._market_data_task
            self._market_data_task = None

        for i in reversed(range(len(self._engines))):
            sid = self._strategy_ids[i]

            # Stop market data source
            if i < len(self._market_data_sources):
                md = self._market_data_sources[i]
                if md is not None:
                    try:
                        await md.stop()
                        logger.info("Market data stopped for %s", sid)
                    except Exception as exc:
                        logger.warning("Error stopping market data %s: %s", sid, exc)

            # Stop engine
            try:
                await self._engines[i].stop()
                logger.info("Engine stopped for %s", sid)
            except Exception as exc:
                logger.error("Error stopping engine %s: %s", sid, exc)

            # Stop instrumentation
            try:
                instr = self._instrumentations[i]
                if instr is not None:
                    await instr.stop()
            except Exception as exc:
                logger.warning("Error stopping instrumentation %s: %s", sid, exc)

            # Stop OMS
            try:
                await self._oms_services[i].stop()
                logger.info("OMS stopped for %s", sid)
            except Exception as exc:
                logger.error("Error stopping OMS %s: %s", sid, exc)

        self._engines.clear()
        self._oms_services.clear()
        self._instrumentations.clear()
        self._strategy_ids.clear()
        self._market_data_sources.clear()
        logger.info("StockFamilyCoordinator stopped")

    def health_status(self) -> dict[str, Any]:
        """Return health of all three stock engines."""
        result: dict[str, Any] = {"family": self.family_id, "strategies": {}}
        for i, engine in enumerate(self._engines):
            sid = self._strategy_ids[i]
            try:
                result["strategies"][sid] = engine.health_status()
            except Exception as exc:
                result["strategies"][sid] = {"error": str(exc)}
        return result

    def apply_regime(self, ctx: "RegimeContext") -> None:
        """Apply regime context to all stock portfolio rules and engine configs."""
        import dataclasses
        from regime.integration import build_stock_rules, STOCK_PROFILES

        if self._base_portfolio_rules is None:
            logger.warning("apply_regime called before start() — skipping")
            return

        prev_regime = getattr(self._regime_ctx, "regime", None)
        self._regime_ctx = ctx
        new_rules = build_stock_rules(ctx, self._base_portfolio_rules)

        # Tier 1: update PortfolioRulesConfig on each checker
        for checker in self._portfolio_checkers:
            if checker is not None:
                checker.update_config(dataclasses.replace(
                    new_rules, initial_equity=checker._cfg.initial_equity,
                ))

        # Tier 2: engine position limit updates
        from regime.integration import _validated_regime
        regime = _validated_regime(ctx.regime)
        profile = STOCK_PROFILES[regime]
        for sid, engine in self._engine_map.items():
            settings = getattr(engine, '_settings', None)
            if settings is None:
                continue
            if sid == "ALCB_v1" and hasattr(settings, 'max_positions'):
                object.__setattr__(settings, 'max_positions', profile["alcb_max_positions"])
            elif sid == "IARIC_v1" and hasattr(settings, 'pb_max_positions'):
                object.__setattr__(settings, 'pb_max_positions', profile["iaric_pb_max_positions"])

        changed = f" (was {prev_regime})" if prev_regime and prev_regime != ctx.regime else ""
        logger.info("Stock regime applied: %s%s (cap=%.1fR, risk=%.2fx, disabled=%s)",
                    ctx.regime, changed, new_rules.directional_cap_R,
                    new_rules.regime_unit_risk_mult, new_rules.disabled_strategies or "none")

    async def _market_data_loop(self) -> None:
        """Periodically refresh market data subscriptions for all engines."""
        while True:
            try:
                for i, engine in enumerate(self._engines):
                    md = self._market_data_sources[i] if i < len(self._market_data_sources) else None
                    if md is None:
                        continue
                    try:
                        if hasattr(engine, "subscription_instruments"):
                            await md.ensure_hot_symbols(engine.subscription_instruments())
                        if hasattr(engine, "polling_instruments"):
                            polling = engine.polling_instruments()
                            if polling:
                                await md.poll_due_bars(
                                    polling, now=datetime.now(timezone.utc),
                                )
                    except Exception as exc:
                        logger.warning(
                            "Market data refresh failed for %s: %s",
                            self._strategy_ids[i], exc,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Market data loop error: %s", exc, exc_info=exc)
            await asyncio.sleep(5.0)

    # ── internal ─────────────────────────────────────────────────────

    def _load_artifacts(self) -> dict[str, Any]:
        """Attempt to load artifacts from disk for strategies that need them.

        Returns a dict keyed by strategy ID.  If an artifact cannot be loaded
        (e.g. research hasn't run yet) the value is ``None``; engines handle
        missing artifacts gracefully.
        """
        today = date.today()
        artifacts: dict[str, Any] = {}

        # IARIC — WatchlistArtifact
        try:
            from strategies.stock.iaric.artifact_store import load_watchlist_artifact
            from strategies.stock.iaric.config import StrategySettings as IARICSettings
            artifacts["IARIC_v1"] = load_watchlist_artifact(today, settings=IARICSettings())
            logger.info("Loaded IARIC watchlist artifact for %s", today)
        except Exception as exc:
            logger.warning("IARIC artifact not available (%s), engine will handle gracefully", exc)
            artifacts["IARIC_v1"] = None

        # ALCB — CandidateArtifact
        try:
            from strategies.stock.alcb.artifact_store import load_candidate_artifact
            from strategies.stock.alcb.config import StrategySettings as ALCBSettings
            artifacts["ALCB_v1"] = load_candidate_artifact(today, settings=ALCBSettings())
            logger.info("Loaded ALCB candidate artifact for %s", today)
        except Exception as exc:
            logger.warning("ALCB artifact not available (%s), engine will handle gracefully", exc)
            artifacts["ALCB_v1"] = None

        return artifacts

    def _build_strategy_descriptors(
        self, artifacts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return per-strategy wiring descriptors.

        Imports are deferred into lambdas / callables so that engine modules
        are only loaded when ``start()`` actually runs.
        """
        ctx = self._ctx

        # --- IBKR config (loaded once, shared across strategies) --------------
        from libs.broker_ibkr.config.loader import IBKRConfig
        from libs.broker_ibkr.mapping.contract_factory import ContractFactory
        from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter

        config_dir = Path(os.environ.get("CONFIG_DIR", str(Path(__file__).resolve().parent.parent.parent / "config")))
        try:
            ibkr_config = IBKRConfig(config_dir)
            account_id = ibkr_config.profile.account_id
        except Exception:
            ibkr_config = None
            account_id = ""

        # Build ContractFactory once, reuse for adapters and market data sources
        session = ctx.session
        if ibkr_config is not None:
            self._contract_factory = ContractFactory(
                ib=session.ib,
                templates=ibkr_config.contracts,
                routes=ibkr_config.routes,
            )

        def _make_adapter(session: Any) -> Any:
            """Build an IBKRExecutionAdapter from the IB session."""
            if self._contract_factory is None:
                raise RuntimeError("IBKRConfig not available")
            return IBKRExecutionAdapter(
                session=session,
                contract_factory=self._contract_factory,
                account=account_id,
            )

        # Shared trade recorder from bootstrap
        trade_recorder = getattr(ctx, "trade_recorder", None)

        # --- IARIC v1 --------------------------------------------------------
        from strategies.stock.iaric.config import (
            STRATEGY_ID as IARIC_ID,
            StrategySettings as IARICSettings,
        )
        iaric_settings = IARICSettings()

        # --- US_ORB v1 -------------------------------------------------------
        from strategies.stock.us_orb.config import (
            STRATEGY_ID as ORB_ID,
            StrategySettings as ORBSettings,
        )
        orb_settings = ORBSettings()

        # --- ALCB v1 ---------------------------------------------------------
        from strategies.stock.alcb.config import (
            STRATEGY_ID as ALCB_ID,
            StrategySettings as ALCBSettings,
        )
        alcb_settings = ALCBSettings()

        return [
            # ── IARIC_v1 ──────────────────────────────────────────────
            {
                "strategy_id": IARIC_ID,
                "base_risk_pct": iaric_settings.base_risk_fraction,
                "daily_stop_R": iaric_settings.daily_stop_r,
                "heat_cap_R": iaric_settings.heat_cap_r,
                "portfolio_daily_stop_R": iaric_settings.portfolio_daily_stop_r,
                "adapter": _make_adapter,
                "engine_cls": lambda: _import_iaric_engine(),
                "market_data_cls": lambda: _import_iaric_market_data(),
                "instr_type": "strategy_iaric",
                "trade_recorder": trade_recorder,
                "account_id": account_id,
                "settings": iaric_settings,
                "data_key": "artifact",
                "data_value": artifacts.get(IARIC_ID),
                "diagnostics_factory": lambda s=iaric_settings: _make_diagnostics(
                    "strategies.stock.iaric.diagnostics", s.diagnostics_dir,
                ),
            },
            # ── US_ORB_v1 ─────────────────────────────────────────────
            {
                "strategy_id": ORB_ID,
                "base_risk_pct": orb_settings.base_risk_pct,
                "daily_stop_R": orb_settings.daily_stop_r,
                "heat_cap_R": orb_settings.heat_cap_r,
                "portfolio_daily_stop_R": orb_settings.portfolio_daily_stop_r,
                "adapter": _make_adapter,
                "engine_cls": lambda: _import_orb_engine(),
                "market_data_cls": lambda: _import_orb_market_data(),
                "instr_type": "strategy_orb",
                "trade_recorder": trade_recorder,
                "account_id": account_id,
                "settings": orb_settings,
                "data_key": "cache",
                "data_value": {},
                "diagnostics_factory": lambda s=orb_settings: _make_diagnostics(
                    "strategies.stock.us_orb.diagnostics", s.diagnostics_dir,
                ),
            },
            # ── ALCB_v1 ───────────────────────────────────────────────
            {
                "strategy_id": ALCB_ID,
                "base_risk_pct": alcb_settings.base_risk_fraction,
                "daily_stop_R": alcb_settings.daily_stop_r,
                "heat_cap_R": alcb_settings.heat_cap_r,
                "portfolio_daily_stop_R": alcb_settings.portfolio_daily_stop_r,
                "adapter": _make_adapter,
                "engine_cls": lambda: _import_alcb_engine(),
                "market_data_cls": lambda: _import_alcb_market_data(),
                "instr_type": "strategy_alcb",
                "trade_recorder": trade_recorder,
                "account_id": account_id,
                "settings": alcb_settings,
                "data_key": "artifact",
                "data_value": artifacts.get(ALCB_ID),
                "diagnostics_factory": lambda s=alcb_settings: _make_diagnostics(
                    "strategies.stock.alcb.diagnostics", s.diagnostics_dir,
                ),
            },
        ]


# ── Deferred engine imports ──────────────────────────────────────────

def _import_iaric_engine():
    from strategies.stock.iaric.engine import IARICEngine
    return IARICEngine


def _import_orb_engine():
    from strategies.stock.us_orb.engine import USORBEngine
    return USORBEngine


def _import_alcb_engine():
    from strategies.stock.alcb.engine import ALCBT2Engine
    return ALCBT2Engine


def _import_iaric_market_data():
    from strategies.stock.iaric.data import IBMarketDataSource
    return IBMarketDataSource


def _import_orb_market_data():
    from strategies.stock.us_orb.data import IBMarketDataSource
    return IBMarketDataSource


def _import_alcb_market_data():
    from strategies.stock.alcb.data import IBMarketDataSource
    return IBMarketDataSource


def _make_diagnostics(module_path: str, diagnostics_dir: Path) -> Any:
    """Instantiate JsonlDiagnostics from the given strategy module."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        return mod.JsonlDiagnostics(root=diagnostics_dir)
    except Exception as exc:
        logger.warning("Diagnostics init failed for %s: %s", module_path, exc)
        return None
