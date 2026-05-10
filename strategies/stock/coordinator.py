"""Stock family coordinator — each enabled stock strategy gets its own OMS instance.

Like momentum (per-strategy OMS), unlike swing (shared OMS).
Supported stock strategies:
  - IARIC_v1   (IARICEngine)   — WatchlistArtifact
  - ALCB_v1    (ALCBT2Engine)  — CandidateArtifact

Stock-specific differences from momentum:
  - portfolio_rules with family-scoped directional cap + symbol collision guard
  - Paper equity NAV tracking via resolve_paper_nav / capital_bootstrap
  - Artifacts or cache dicts instead of instrument dicts
  - IBMarketDataSource per engine, wired via on_bar/on_quote callbacks
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import suppress
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from libs.oms.persistence.db_config import get_environment
from libs.services.heartbeat import emit_family_heartbeats

from strategies.contracts import RuntimeContext
from strategies.stock.readiness import validate_stock_readiness

logger = logging.getLogger(__name__)

_STOCK_SYMBOL_COLLISION_PAIRS: tuple[tuple[str, str, str], ...] = ()
_STOCK_STRATEGY_PRIORITIES: tuple[tuple[str, int], ...] = (
    ("IARIC_v1", 0),
    ("ALCB_v1", 1),
)
_STOCK_DIRECTIONAL_CAP_R = 6.5
_STOCK_DIRECTIONAL_LONG_CAP_R = 6.25
_STOCK_PRIORITY_HEADROOM_R = 1.15
_STOCK_REFERENCE_UNIT_RISK_DOLLARS = 162.0
_STOCK_PORTFOLIO_WEEKLY_STOP_R = 8.0
_STOCK_DD_TIERS = (
    (0.04, 1.00),
    (0.07, 0.75),
    (0.10, 0.40),
    (0.13, 0.00),
)


class StockFamilyCoordinator:
    """Lifecycle manager for the configured stock strategies.

    Each strategy receives its own OMS built via ``build_oms_service``.
    Shared across the stock family: *db_pool*, *AccountRiskGate*, *family_id*.
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
        self._regime_adjusted_rules: Any = None  # stored after apply_regime for crisis overlay
        self._regime_stock_profile: dict | None = None  # stored Tier 2 profile
        self._crisis_ctx: Any = None
        self._heartbeat_task: asyncio.Task | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Import, build OMS, and start all enabled stock engines."""
        from libs.oms.services.factory import build_oms_service
        from libs.oms.risk.calculator import RiskCalculator
        from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
        from libs.config.capital_bootstrap import bootstrap_capital

        active_strategy_ids = self._enabled_stock_strategy_ids()
        if not active_strategy_ids:
            logger.warning("No enabled stock strategies remain after registry filtering")
            return

        artifacts, readiness_failures = validate_stock_readiness(
            self._ctx.registry,
            live=get_environment() == "live",
            strategy_ids=active_strategy_ids,
        )
        if readiness_failures:
            detail = "; ".join(
                f"{failure.check_name}={failure.detail}" for failure in readiness_failures
            )
            raise RuntimeError(f"Stock family readiness failed: {detail}")

        ctx = self._ctx
        session = ctx.session
        db_pool = ctx.db_pool
        account_gate = ctx.account_gate

        config_dir = Path(
            os.environ.get(
                "CONFIG_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "config"),
            )
        )

        # Resolve paper-mode equity
        runtime_env = get_environment()
        paper_mode = runtime_env == "paper"
        strict_market_data = runtime_env in {"paper", "live"}
        _env = os.getenv("PAPER_INITIAL_EQUITY")
        _paper_seed = (
            float(_env) if _env else ctx.portfolio.capital.paper_initial_equity
        )

        if paper_mode:
            base_equity = _paper_seed
        else:
            # EQUITY-1: hard-fail in live mode if NetLiquidation can't be
            # resolved, using the configured IBKR profile account. The helper polls
            # accountValues() up to its timeout to handle the async populate
            # race between ib.connectAsync and the first accountValues read.
            from libs.services.equity import resolve_live_nlv
            from libs.broker_ibkr.config.loader import IBKRConfig

            ibkr_config = IBKRConfig(config_dir)
            if not ibkr_config.profile.account_id:
                raise RuntimeError("Stock live equity requires configured IBKR account_id")
            base_equity = await resolve_live_nlv(
                session,
                account_id=ibkr_config.profile.account_id,
            )

        # Capital allocation per strategy
        try:
            allocs = bootstrap_capital(base_equity, config_dir)
        except Exception as exc:
            logger.warning("bootstrap_capital failed (%s), using equal split", exc)
            allocs = {}

        # ── Strategy descriptors ─────────────────────────────────────
        _strategies = self._build_strategy_descriptors(artifacts, active_strategy_ids)
        if not _strategies:
            logger.warning("No enabled stock strategies remain after registry filtering")
            return
        fallback_nav = base_equity / max(1, len(_strategies))

        paper_nav_by_strategy: dict[str, float] = {}
        paper_initial_nav_by_strategy: dict[str, float] = {}
        equity = base_equity
        if paper_mode:
            from libs.persistence.paper_equity import PaperEquityManager

            for desc in _strategies:
                sid = desc["strategy_id"]
                alloc = allocs.get(sid)
                initial_nav = alloc.allocated_nav if alloc else fallback_nav
                paper_initial_nav_by_strategy[sid] = initial_nav
                pem = PaperEquityManager(
                    db_pool,
                    account_scope=sid,
                    initial_equity=initial_nav,
                )
                paper_nav_by_strategy[sid] = await pem.load()
            if paper_nav_by_strategy:
                equity = sum(paper_nav_by_strategy.values())
            logger.info(
                "Paper mode equity for stock strategies: total=$%.2f scopes=%s",
                equity,
                {k: round(v, 2) for k, v in paper_nav_by_strategy.items()},
            )

        # Portfolio rules: drawdown tiers + family-scoped directional cap + symbol collision
        rule_inputs = self._portfolio_rule_inputs(
            tuple(d["strategy_id"] for d in _strategies)
        )
        all_strategy_ids = rule_inputs["family_strategy_ids"]
        collision_pairs = rule_inputs["symbol_collision_pairs"]
        strategy_priorities = rule_inputs["strategy_priorities"]
        logger.info(
            "Stock portfolio rules: directional_cap=%.1fR, collision=%s, "
            "collision_pairs=%s, headroom=%.1fR, strategies=%s",
            _STOCK_DIRECTIONAL_CAP_R, "half_size",
            [f"{h}->{r}:{a}" for h, r, a in collision_pairs],
            _STOCK_PRIORITY_HEADROOM_R, all_strategy_ids,
        )

        # Log dollar-equivalent directional cap per strategy for monitoring
        for desc in _strategies:
            _alloc = allocs.get(desc["strategy_id"])
            _nav = (
                paper_nav_by_strategy.get(desc["strategy_id"])
                if paper_mode
                else None
            )
            if _nav is None:
                _nav = _alloc.allocated_nav if _alloc else fallback_nav
            _unit = RiskCalculator.compute_unit_risk_dollars(nav=_nav, unit_risk_pct=desc["base_risk_pct"])
            logger.info(
                "Directional cap dollar-equiv for %s: 1R=$%.0f, cap=%.2fR=$%.0f, heat=%sR=$%.0f",
                desc["strategy_id"], _unit, _STOCK_DIRECTIONAL_CAP_R, _STOCK_DIRECTIONAL_CAP_R * _unit,
                desc["heat_cap_R"], desc["heat_cap_R"] * _unit,
            )

        _shared_sidecar = None  # one sidecar per family (Finding 8)
        for desc in _strategies:
            sid = desc["strategy_id"]

            if desc.get("data_key") == "artifact" and desc.get("data_value") is None:
                raise RuntimeError(
                    f"Stock artifact missing after readiness validation for {sid}"
                )

            self._strategy_ids.append(sid)

            # Resolve allocated NAV
            alloc = allocs.get(sid)
            initial_nav = (
                paper_initial_nav_by_strategy.get(sid)
                if paper_mode
                else None
            )
            if initial_nav is None:
                initial_nav = alloc.allocated_nav if alloc else fallback_nav
            allocated_nav = (
                paper_nav_by_strategy.get(sid, initial_nav)
                if paper_mode
                else (alloc.allocated_nav if alloc else fallback_nav)
            )

            # Per-strategy portfolio rules — initial_equity must match the
            # get_current_equity callback (allocated_nav), otherwise drawdown
            # tiers see a phantom 67% DD and halt every entry.
            portfolio_rules = PortfolioRulesConfig(
                directional_cap_R=_STOCK_DIRECTIONAL_CAP_R,
                directional_cap_long_R=_STOCK_DIRECTIONAL_LONG_CAP_R,
                initial_equity=allocated_nav,
                family_strategy_ids=all_strategy_ids,
                symbol_collision_action="half_size",
                symbol_collision_pairs=collision_pairs,
                strategy_priorities=strategy_priorities,
                priority_headroom_R=_STOCK_PRIORITY_HEADROOM_R,
                priority_reserve_threshold=1,  # priority 0-1 can use reserved headroom
                reference_unit_risk_dollars=_STOCK_REFERENCE_UNIT_RISK_DOLLARS,
                dd_tiers=_STOCK_DD_TIERS,
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
                    "Strategy %s not in unified config, using equal fallback NAV %.2f",
                    sid, fallback_nav,
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
                portfolio_weekly_stop_R=_STOCK_PORTFOLIO_WEEKLY_STOP_R,
                db_pool=db_pool,
                portfolio_rules_config=portfolio_rules,
                get_current_equity=lambda eq=_live_equity: eq[0],
                paper_equity_pool=db_pool if paper_mode else None,
                paper_equity_scope=sid,
                paper_initial_equity=initial_nav,
                live_equity=_live_equity if not paper_mode else None,
                family_id=self.family_id,
                account_gate=account_gate,
                family_strategy_ids=list(all_strategy_ids),
            )
            await oms.start()
            logger.info("OMS started for %s", sid)
            self._oms_services.append(oms)
            self._portfolio_checkers.append(getattr(oms, '_portfolio_checker', None))

            # Instrumentation (non-fatal) — share ONE sidecar across all strategies
            instr = None
            try:
                from libs.oms.persistence.postgres import PgStore
                from .instrumentation.src.bootstrap import InstrumentationManager
                _pg_store = PgStore(db_pool) if db_pool is not None else None
                instr = InstrumentationManager(
                    oms, sid, strategy_type=desc["instr_type"],
                    pg_store=_pg_store,
                    family_strategy_ids=list(all_strategy_ids),
                    get_regime_ctx=lambda: self._regime_ctx,
                    get_applied_config=lambda: self._portfolio_checkers[0]._cfg if self._portfolio_checkers and self._portfolio_checkers[0] else None,
                )
                if _shared_sidecar is None:
                    _shared_sidecar = instr.sidecar
                else:
                    instr.sidecar = _shared_sidecar
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
                        historical_requester=getattr(session, "req_historical_data", None),
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
                    if strict_market_data:
                        await self.stop()
                        raise RuntimeError(
                            f"Stock market data init failed for {sid} in {runtime_env} mode"
                        ) from exc
            else:
                logger.warning("No contract factory — market data NOT wired for %s", sid)
            if self._contract_factory is None and strict_market_data:
                await self.stop()
                raise RuntimeError(
                    f"Stock market data not wired for {sid} in {runtime_env} mode"
                )
            self._market_data_sources.append(md_source)

        # ── Reconnect callback ─────────────────────────────────────
        # CONN-1: also drive OMS reconciliation on each per-strategy OMS.
        # The previous version only did per-engine reconcile + subscription
        # invalidation, leaving OMS<->broker drift undetected.
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
            for i, oms in enumerate(self._oms_services):
                reconciler = getattr(oms, "_reconciler", None)
                if reconciler is not None:
                    try:
                        await reconciler.on_reconnect_reconciliation()
                    except Exception as exc:
                        logger.error(
                            "Stock OMS reconnect reconciliation failed for %s: %s",
                            self._strategy_ids[i], exc,
                        )
            logger.info(
                "Post-reconnect: per-engine + OMS reconciliation + "
                "subscription invalidation complete"
            )

        if hasattr(session, "add_reconnect_callback"):
            session.add_reconnect_callback(_on_reconnect)

        # ── Background market data refresh loop ────────────────────
        self._market_data_task = asyncio.create_task(self._market_data_loop())

        # ── Heartbeat background task ──────────────────────────────────
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            "StockFamilyCoordinator started %d strategies with market data", len(self._engines),
        )

    async def stop(self) -> None:
        """Stop market data, engines, and OMS instances in reverse order."""
        # Stop heartbeat loop
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

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
        self._regime_adjusted_rules = new_rules  # Store for crisis overlay

        # Tier 2: engine position limit updates
        from regime.integration import _validated_regime
        regime = _validated_regime(ctx.regime)
        profile = STOCK_PROFILES[regime]
        self._regime_stock_profile = dict(profile)  # Store for crisis overlay
        for sid, engine in self._engine_map.items():
            settings = getattr(engine, '_settings', None)
            if settings is None:
                continue
            if sid == "ALCB_v1" and hasattr(settings, 'max_positions'):
                object.__setattr__(settings, 'max_positions', profile["alcb_max_positions"])
            elif sid == "IARIC_v1" and hasattr(settings, 'pb_max_positions'):
                object.__setattr__(settings, 'pb_max_positions', profile["iaric_pb_max_positions"])

        # Apply crisis overlay if active, including pre-action stress formation.
        if self._crisis_ctx is not None:
            from regime.crisis.integration import apply_crisis_overlay
            new_rules = apply_crisis_overlay(
                new_rules,
                self._crisis_ctx,
                self.family_id,
                regime=ctx.regime,
            )

        # Tier 1: update PortfolioRulesConfig on each checker
        for checker in self._portfolio_checkers:
            if checker is not None:
                checker.update_config(dataclasses.replace(
                    new_rules, initial_equity=checker._cfg.initial_equity,
                ))

        changed = f" (was {prev_regime})" if prev_regime and prev_regime != ctx.regime else ""
        logger.info("Stock regime applied: %s%s (cap=%.1fR, risk=%.2fx, disabled=%s)",
                    ctx.regime, changed, new_rules.directional_cap_R,
                    new_rules.regime_unit_risk_mult, new_rules.disabled_strategies or "none")

        # Emit structured regime→rules event for TA pipeline
        self._emit_regime_event({
            "family": "stock",
            "regime": str(ctx.regime),
            "prev_regime": str(prev_regime) if prev_regime else None,
            "rules_applied": {
                "directional_cap_R": new_rules.directional_cap_R,
                "regime_unit_risk_mult": new_rules.regime_unit_risk_mult,
                "disabled_strategies": new_rules.disabled_strategies or [],
                "alcb_max_positions": profile.get("alcb_max_positions"),
                "iaric_pb_max_positions": profile.get("iaric_pb_max_positions"),
            },
        })

    def apply_crisis(self, ctx) -> None:
        """Apply crisis context overlay on top of regime-adjusted rules.

        Handles both Tier 1 (PortfolioRulesConfig) and Tier 2 (engine settings).
        Always starts from _regime_adjusted_rules to prevent compounding.
        """
        import dataclasses
        from regime.crisis.actions import resolve_crisis_action
        from regime.crisis.integration import apply_crisis_overlay

        prev_level = getattr(self._crisis_ctx, "alert_level", "NORMAL") if self._crisis_ctx else "NORMAL"
        self._crisis_ctx = ctx

        if self._regime_adjusted_rules is None:
            logger.warning("apply_crisis called before apply_regime — skipping")
            return

        regime = getattr(self._regime_ctx, "regime", None)
        action = resolve_crisis_action(ctx, self.family_id, regime=regime)
        if action.is_no_action():
            # NORMAL or WATCH: revert to regime-only rules
            for checker in self._portfolio_checkers:
                if checker is not None:
                    checker.update_config(dataclasses.replace(
                        self._regime_adjusted_rules,
                        initial_equity=checker._cfg.initial_equity,
                    ))
            # Restore regime Tier 2 settings
            if self._regime_stock_profile is not None:
                self._apply_tier2_settings(self._regime_stock_profile)
            if prev_level not in ("NORMAL", "WATCH"):
                logger.info("Stock crisis overlay removed (level=%s)", ctx.alert_level)
            return

        tightened = apply_crisis_overlay(
            self._regime_adjusted_rules,
            ctx,
            self.family_id,
            regime=regime,
        )
        for checker in self._portfolio_checkers:
            if checker is not None:
                checker.update_config(dataclasses.replace(
                    tightened, initial_equity=checker._cfg.initial_equity,
                ))

        # Tier 2 crisis tightening: reduce engine position limits further
        if self._regime_stock_profile is not None and action.alert_level_int >= 2:
            crisis_profile = {
                "alcb_max_positions": max(
                    1, int(self._regime_stock_profile.get("alcb_max_positions", 6) * action.position_limit_multiplier)
                ),
                "iaric_pb_max_positions": max(
                    1, int(self._regime_stock_profile.get("iaric_pb_max_positions", 10) * action.position_limit_multiplier)
                ),
            }
            self._apply_tier2_settings(crisis_profile)

        changed = f" (was {prev_level})" if prev_level != ctx.alert_level else ""
        logger.info(
            "Stock crisis applied: %s%s (risk_mult=%.2f, dd_mult=%.2f, "
            "provenance=%s, dominant=%s)",
            ctx.alert_level, changed, action.risk_multiplier,
            action.dd_tier_multiplier, action.action_provenance, ctx.dominant_channel,
        )

        self._emit_crisis_event({
            "family": "stock",
            "alert_level": ctx.alert_level,
            "prev_level": prev_level,
            "risk_multiplier": ctx.risk_multiplier,
            "dd_tier_multiplier": ctx.dd_tier_multiplier,
            "dominant_channel": ctx.dominant_channel,
            "action_policy": action.to_dict(),
        })

    def _apply_tier2_settings(self, profile: dict) -> None:
        """Apply Tier 2 engine position limit settings."""
        for sid, engine in self._engine_map.items():
            settings = getattr(engine, '_settings', None)
            if settings is None:
                continue
            if sid == "ALCB_v1" and hasattr(settings, 'max_positions'):
                object.__setattr__(settings, 'max_positions', profile["alcb_max_positions"])
            elif sid == "IARIC_v1" and hasattr(settings, 'pb_max_positions'):
                object.__setattr__(settings, 'pb_max_positions', profile["iaric_pb_max_positions"])

    def _emit_crisis_event(self, payload: dict) -> None:
        """Write a crisis event to each strategy's data_dir for TA pipeline."""
        now = datetime.now(timezone.utc)
        record = {"timestamp": now.isoformat(), "event_type": "crisis_alert_change", **payload}
        for instr in self._instrumentations:
            try:
                data_dir = getattr(instr, "_config", {}).get("data_dir")
                if not data_dir:
                    continue
                out_dir = Path(data_dir) / "coordination_events"
                out_dir.mkdir(parents=True, exist_ok=True)
                date_str = now.strftime("%Y-%m-%d")
                with open(out_dir / f"{date_str}.jsonl", "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass

    def _emit_regime_event(self, payload: dict) -> None:
        """Write a regime→rules event to each strategy's data_dir for TA pipeline."""
        now = datetime.now(timezone.utc)
        record = {"timestamp": now.isoformat(), "event_type": "regime_rules_change", **payload}
        for instr in self._instrumentations:
            try:
                # InstrumentationManager stores config as self._config dict,
                # NOT as self._data_dir attribute.
                data_dir = getattr(instr, "_config", {}).get("data_dir")
                if not data_dir:
                    continue
                out_dir = Path(data_dir) / "coordination_events"
                out_dir.mkdir(parents=True, exist_ok=True)
                date_str = now.strftime("%Y-%m-%d")
                with open(out_dir / f"{date_str}.jsonl", "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass

    async def _heartbeat_loop(self) -> None:
        heartbeat = getattr(self._ctx, "heartbeat", None)
        if heartbeat is None:
            return
        session = self._ctx.session
        # HB-1: any exception in payload construction or emission must not
        # kill the loop. A dead heartbeat task wouldn't be observed until the
        # watchdog stale_threshold elapsed, masking a live runtime as silent.
        while True:
            try:
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                return
            try:
                await self._heartbeat_iteration(heartbeat, session)
            except Exception:
                logger.exception("Stock heartbeat iteration failed; continuing")

    async def _heartbeat_iteration(self, heartbeat, session) -> None:
        payloads = []
        for i, sid in enumerate(self._strategy_ids):
            # Use per-strategy state (not portfolio state) for correct per-strategy metrics
            srs = getattr(self._oms_services[i], "_strategy_risk_states", {})
            sr = srs.get(sid)
            # Fall back to portfolio state for halted check
            prs = getattr(self._oms_services[i], "_portfolio_risk_state", None)
            payload = {
                "strategy_id": sid,
                "heat_r": Decimal(str(sr.open_risk_R)) if sr else Decimal("0"),
                "daily_pnl_r": Decimal(str(sr.daily_realized_R)) if sr else Decimal("0"),
                "mode": "HALTED" if (prs and prs.halted) else "RUNNING",
            }
            # Diagnostic pulse fields
            engine = self._engines[i]
            if hasattr(engine, "_last_decision_code"):
                payload["last_decision_code"] = engine._last_decision_code
                payload["last_decision_details"] = getattr(engine, "_last_decision_details", None)
                payload["last_seen_bar_ts"] = getattr(engine, "_last_bar_ts", None)
            oms = self._oms_services[i]
            if hasattr(engine, "liveness_payload") or hasattr(oms, "_intents_submitted"):
                details = payload.get("last_decision_details") or {}
                if hasattr(engine, "liveness_payload"):
                    details["liveness"] = engine.liveness_payload()
                if hasattr(oms, "_intents_submitted"):
                    details["oms_health"] = {
                        "submitted": oms._intents_submitted,
                        "accepted": oms._intents_accepted,
                        "denied": oms._intents_denied,
                        "consecutive_denials": oms._consecutive_denials,
                    }
                payload["last_decision_details"] = details
            payloads.append(payload)
        connected = session.ib.isConnected() if session else False
        # Enrich with IB farm status for diagnostic context
        if session:
            farm_statuses = {}
            for group in session.groups.values():
                if group.farm_monitor:
                    farm_statuses.update(group.farm_monitor.all_statuses())
            if farm_statuses:
                for p in payloads:
                    details = p.get("last_decision_details") or {}
                    if isinstance(details, dict):
                        details["ib_farm_status"] = farm_statuses
                        p["last_decision_details"] = details
        await emit_family_heartbeats(
            heartbeat, self.family_id, payloads, adapter_connected=connected,
        )

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

    def _enabled_stock_strategy_ids(self) -> tuple[str, ...]:
        """Return enabled stock strategy IDs for the current runtime environment."""
        registry = getattr(self._ctx, "registry", None)
        if registry is None:
            return ()
        live = get_environment() == "live"
        return tuple(
            manifest.strategy_id
            for manifest in registry.enabled_strategies(live=live)
            if manifest.family == self.family_id
        )

    def _portfolio_rule_inputs(self, strategy_ids: tuple[str, ...]) -> dict[str, Any]:
        """Build stock-family rule inputs for the active strategy set."""
        active_ids = set(strategy_ids)
        return {
            "family_strategy_ids": strategy_ids,
            "symbol_collision_pairs": tuple(
                (holder_id, requester_id, action)
                for holder_id, requester_id, action in _STOCK_SYMBOL_COLLISION_PAIRS
                if holder_id in active_ids and requester_id in active_ids
            ),
            "strategy_priorities": tuple(
                (strategy_id, priority)
                for strategy_id, priority in _STOCK_STRATEGY_PRIORITIES
                if strategy_id in active_ids
            ),
        }

    def _build_strategy_descriptors(
        self,
        artifacts: dict[str, Any],
        strategy_ids: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Return per-strategy wiring descriptors.

        Strategy modules are imported only for enabled stock strategies so a
        removed disabled package cannot break coordinator startup.
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

        def _build_iaric_descriptor() -> dict[str, Any]:
            from strategies.stock.iaric.config import (
                STRATEGY_ID as IARIC_ID,
                StrategySettings as IARICSettings,
            )

            iaric_settings = IARICSettings()
            return {
                "strategy_id": IARIC_ID,
                "base_risk_pct": iaric_settings.base_risk_fraction,
                "daily_stop_R": iaric_settings.daily_stop_r,
                "heat_cap_R": iaric_settings.heat_cap_r,
                "portfolio_daily_stop_R": iaric_settings.portfolio_daily_stop_r,
                "adapter": _make_adapter,
                "engine_cls": _import_iaric_engine,
                "market_data_cls": _import_iaric_market_data,
                "instr_type": "strategy_iaric",
                "trade_recorder": trade_recorder,
                "account_id": account_id,
                "settings": iaric_settings,
                "data_key": "artifact",
                "data_value": artifacts.get(IARIC_ID),
                "diagnostics_factory": lambda s=iaric_settings: _make_diagnostics(
                    "strategies.stock.iaric.diagnostics", s.diagnostics_dir,
                ),
            }

        def _build_alcb_descriptor() -> dict[str, Any]:
            from strategies.stock.alcb.config import (
                STRATEGY_ID as ALCB_ID,
                StrategySettings as ALCBSettings,
            )

            alcb_settings = ALCBSettings()
            return {
                "strategy_id": ALCB_ID,
                "base_risk_pct": alcb_settings.base_risk_fraction,
                "daily_stop_R": alcb_settings.daily_stop_r,
                "heat_cap_R": alcb_settings.heat_cap_r,
                "portfolio_daily_stop_R": alcb_settings.portfolio_daily_stop_r,
                "adapter": _make_adapter,
                "engine_cls": _import_alcb_engine,
                "market_data_cls": _import_alcb_market_data,
                "instr_type": "strategy_alcb",
                "trade_recorder": trade_recorder,
                "account_id": account_id,
                "settings": alcb_settings,
                "data_key": "artifact",
                "data_value": artifacts.get(ALCB_ID),
                "diagnostics_factory": lambda s=alcb_settings: _make_diagnostics(
                    "strategies.stock.alcb.diagnostics", s.diagnostics_dir,
                ),
            }

        descriptor_builders = {
            "IARIC_v1": _build_iaric_descriptor,
            "ALCB_v1": _build_alcb_descriptor,
        }
        descriptors: list[dict[str, Any]] = []
        unsupported: list[str] = []
        for strategy_id in strategy_ids:
            builder = descriptor_builders.get(strategy_id)
            if builder is None:
                unsupported.append(strategy_id)
                continue
            descriptors.append(builder())
        if unsupported:
            logger.warning(
                "Skipping unsupported stock strategies with no wiring descriptor: %s",
                unsupported,
            )
        return descriptors


# ── Deferred engine imports ──────────────────────────────────────────

def _import_iaric_engine():
    from strategies.stock.iaric.engine import IARICEngine
    return IARICEngine


def _import_alcb_engine():
    from strategies.stock.alcb.engine import ALCBT2Engine
    return ALCBT2Engine


def _import_iaric_market_data():
    from strategies.stock.iaric.data import IBMarketDataSource
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
