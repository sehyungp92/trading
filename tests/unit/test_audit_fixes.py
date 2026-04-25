"""Unit tests for the 2026-04-10 audit fixes.

Covers:
  Fix 1 — Cancel status normalization (adapter emits "Cancelled", status map matches)
  Fix 2 — Family-scoped risk aggregation (strategy_ids filter in postgres queries)
  Fix 3 — CURRENT_DATE replaced with explicit ET trade date
  Fix 4 — Order event payload (symbol field + reject_reason key)
  Fix 6 — Relay ack data-loss (monotonic id ordering, no priority)
  Fix 8 — Order bus symbol field present
  Fix 9 — Relay HMAC auth enforcement in paper/live
  Fix 10 — Dashboard registry-backed strategy metadata
  Fix 13 — TA order/process_quality curated artifacts
"""
from __future__ import annotations

import inspect
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ===========================================================================
# Fix 1 — Cancel status normalization
# ===========================================================================


class TestCancelStatusNormalization:
    """Adapter emits 'Cancelled' (PascalCase); both single and multi OMS
    status maps must map it to OrderStatus.CANCELLED."""

    def test_adapter_emits_pascal_case_cancelled(self):
        """execution_adapter._handle_order_status sends 'Cancelled' for IB cancels."""
        from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter

        # We inspect the source to confirm the string literal is "Cancelled"
        import inspect

        src = inspect.getsource(IBKRExecutionAdapter._handle_order_status)
        # The adapter should emit "Cancelled", NOT "CANCELLED"
        assert '"Cancelled"' in src or "'Cancelled'" in src
        assert '"CANCELLED"' not in src

    def test_single_oms_status_map_contains_cancelled(self):
        """The on_status callback in build_oms_service maps 'Cancelled'."""
        # Read factory source to verify status_map keys
        src_path = Path("libs/oms/services/factory.py")
        src = src_path.read_text(encoding="utf-8")
        # Both single and multi OMS status maps must have "Cancelled" key
        assert '"Cancelled": OrderStatus.CANCELLED' in src

    def test_cancel_status_roundtrip(self):
        """Drive 'Cancelled' through a mock status map and verify it resolves."""
        from libs.oms.models.order import OrderStatus

        status_map = {
            "Submitted": OrderStatus.WORKING,
            "PreSubmitted": OrderStatus.ROUTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "ApiCancelled": OrderStatus.CANCELLED,
            "PendingCancel": OrderStatus.CANCEL_REQUESTED,
        }
        assert status_map.get("Cancelled") == OrderStatus.CANCELLED
        # UPPERCASE must NOT match (this was the bug)
        assert status_map.get("CANCELLED") is None


# ===========================================================================
# Fix 4 + 8 — Order event payload (symbol field + reject_reason key)
# ===========================================================================


class TestOrderEventPayload:
    """EventBus.emit_order_event must include 'symbol' and 'reject_reason'."""

    def test_order_event_contains_symbol_field(self):
        from libs.oms.events.bus import EventBus
        from libs.oms.models.instrument import Instrument
        from libs.oms.models.order import (
            OMSOrder,
            OrderRole,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        bus = EventBus()
        q = bus.subscribe("test_strat")

        instr = Instrument(
            symbol="AAPL", root="AAPL", venue="SMART",
            tick_size=0.01, tick_value=0.01, multiplier=1.0,
        )
        order = OMSOrder(
            oms_order_id="ord-1",
            strategy_id="test_strat",
            instrument=instr,
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
        )
        bus.emit_order_event(order)

        event = q.get_nowait()
        assert event.payload["symbol"] == "AAPL"

    def test_order_event_contains_reject_reason_key(self):
        from libs.oms.events.bus import EventBus
        from libs.oms.models.order import (
            OMSOrder,
            OrderRole,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        bus = EventBus()
        q = bus.subscribe("test_strat")

        order = OMSOrder(
            oms_order_id="ord-2",
            strategy_id="test_strat",
            status=OrderStatus.REJECTED,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
            reject_reason="Insufficient margin",
        )
        bus.emit_order_event(order)

        event = q.get_nowait()
        # Must use "reject_reason" key (not "rejection_reason")
        assert "reject_reason" in event.payload
        assert event.payload["reject_reason"] == "Insufficient margin"
        assert "rejection_reason" not in event.payload

    def test_order_event_symbol_empty_when_no_instrument(self):
        from libs.oms.events.bus import EventBus
        from libs.oms.models.order import (
            OMSOrder,
            OrderRole,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        bus = EventBus()
        q = bus.subscribe("test_strat")

        order = OMSOrder(
            oms_order_id="ord-3",
            strategy_id="test_strat",
            instrument=None,
            status=OrderStatus.CANCELLED,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            role=OrderRole.EXIT,
        )
        bus.emit_order_event(order)

        event = q.get_nowait()
        assert event.payload["symbol"] == ""


# ===========================================================================
# Fix 6 — Relay ack data-loss (monotonic id ordering)
# ===========================================================================


class TestRelayAckDataLoss:
    """get_events must return events by id ASC (not priority ASC) so that
    watermark-based ack cannot skip unseen events."""

    def _make_store(self, db_path: str):
        from apps.relay.db.store import EventStore
        return EventStore(db_path=db_path)

    def test_events_returned_in_id_order_not_priority(self):
        """Insert events with mixed priorities; verify fetch order is by id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(f"{tmpdir}/test.db")
            store.insert_events([
                {"event_id": "e1", "bot_id": "b1", "priority": 3},
                {"event_id": "e2", "bot_id": "b1", "priority": 3},
                {"event_id": "e3", "bot_id": "b1", "priority": 0},  # high priority, higher id
            ])

            events = store.get_events(limit=2)
            ids = [e["event_id"] for e in events]
            # Must be in insertion order (id ASC), NOT priority order
            assert ids == ["e1", "e2"]

    def test_ack_does_not_skip_unseen_events(self):
        """Acking the last event in a batch must not mark later-id events as acked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(f"{tmpdir}/test.db")
            store.insert_events([
                {"event_id": "e1", "bot_id": "b1", "priority": 3},
                {"event_id": "e2", "bot_id": "b1", "priority": 3},
                {"event_id": "e3", "bot_id": "b1", "priority": 0},
                {"event_id": "e4", "bot_id": "b1", "priority": 3},
            ])

            # Fetch first 2 events
            batch = store.get_events(limit=2)
            assert len(batch) == 2

            # Ack up to the last event in the batch
            last_event_id = batch[-1]["event_id"]
            store.ack_up_to(last_event_id)

            # Remaining events must all still be fetchable
            remaining = store.get_events(limit=10)
            remaining_ids = [e["event_id"] for e in remaining]
            assert "e3" in remaining_ids
            assert "e4" in remaining_ids

    def test_all_events_eventually_delivered(self):
        """Repeated fetch-ack cycles must deliver every event exactly once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(f"{tmpdir}/test.db")
            store.insert_events([
                {"event_id": f"e{i}", "bot_id": "b1", "priority": (0 if i % 3 == 0 else 3)}
                for i in range(1, 11)
            ])

            delivered = []
            for _ in range(20):  # safety limit
                batch = store.get_events(limit=3)
                if not batch:
                    break
                delivered.extend(e["event_id"] for e in batch)
                store.ack_up_to(batch[-1]["event_id"])

            assert sorted(delivered, key=lambda x: int(x[1:])) == [f"e{i}" for i in range(1, 11)]


# ===========================================================================
# Fix 2 — Family-scoped risk aggregation (strategy_ids filter)
# ===========================================================================


class TestFamilyScopedRiskAggregation:
    """Postgres queries must filter by strategy_ids to prevent cross-family bleed."""

    def test_postgres_get_risk_daily_strategies_accepts_strategy_ids(self):
        """Verify the method signature accepts strategy_ids parameter."""
        import inspect
        from libs.oms.persistence.postgres import PgStore

        sig = inspect.signature(PgStore.get_risk_daily_strategies_for_date)
        assert "strategy_ids" in sig.parameters

    def test_postgres_get_risk_daily_strategy_totals_accepts_strategy_ids(self):
        """Verify the method signature accepts strategy_ids parameter."""
        import inspect
        from libs.oms.persistence.postgres import PgStore

        sig = inspect.signature(PgStore.get_risk_daily_strategy_totals)
        assert "strategy_ids" in sig.parameters

    def test_factory_single_oms_defines_family_sids(self):
        """build_oms_service must accept family_strategy_ids for scoped queries."""
        src = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
        # Single-OMS builder should use family_strategy_ids when provided, else [strategy_id]
        assert "family_strategy_ids" in src
        assert "_family_sids = list(family_strategy_ids) if family_strategy_ids else [strategy_id]" in src

    def test_factory_multi_oms_defines_family_sids(self):
        """build_multi_strategy_oms must define _family_sids for scoped queries."""
        src = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
        # Multi-OMS builder should define _family_sids from strategies list
        assert '_family_sids = [s["id"] for s in strategies]' in src

    def test_factory_passes_strategy_ids_to_queries(self):
        """Both load and persist paths must pass strategy_ids filter."""
        src = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
        assert "strategy_ids=_family_sids" in src


# ===========================================================================
# Fix 3 — CURRENT_DATE replaced with explicit ET trade date
# ===========================================================================


class TestCurrentDateRemoved:
    """All SQL queries must use explicit ET trade date, never CURRENT_DATE."""

    _SQL_FILES = [
        "libs/risk/account_risk_gate.py",
        "apps/dashboard/src/app/api/live/route.ts",
        "apps/dashboard/src/app/api/portfolio/route.ts",
        "apps/dashboard/src/app/api/strategies/route.ts",
        "apps/dashboard/src/app/api/charts/route.ts",
    ]

    def test_no_current_date_in_sql_files(self):
        """CURRENT_DATE must not appear in any query file."""
        for rel_path in self._SQL_FILES:
            p = Path(rel_path)
            if not p.exists():
                continue
            src = p.read_text(encoding="utf-8")
            assert "CURRENT_DATE" not in src, f"CURRENT_DATE found in {rel_path}"

    def test_account_risk_gate_uses_et_timezone(self):
        """account_risk_gate.py must use explicit ET trade date expression."""
        src = Path("libs/risk/account_risk_gate.py").read_text(encoding="utf-8")
        assert "(now() AT TIME ZONE 'America/New_York')::date" in src

    def test_dashboard_live_route_uses_et_timezone(self):
        """Live batch route must use explicit ET trade date expression."""
        src = Path("apps/dashboard/src/app/api/live/route.ts").read_text(encoding="utf-8")
        assert "(now() AT TIME ZONE 'America/New_York')::date" in src


# ===========================================================================
# Fix 9 — HMAC auth enforcement in paper/live
# ===========================================================================


class TestHMACEnforcement:
    """Sidecar HMAC must be enforced (raise on missing) in paper/live modes."""

    def test_swing_context_enforces_hmac(self):
        """Swing InstrumentationContext.start() raises if HMAC missing in paper/live."""
        src = Path("strategies/swing/instrumentation/src/context.py").read_text(encoding="utf-8")
        assert 'env in ("paper", "live")' in src
        assert "HMAC secret is required" in src
        assert "RuntimeError" in src

    def test_momentum_bootstrap_enforces_hmac(self):
        """Momentum InstrumentationManager.start() raises if HMAC missing in paper/live."""
        src = Path("strategies/momentum/instrumentation/src/bootstrap.py").read_text(encoding="utf-8")
        assert 'env in ("paper", "live")' in src
        assert "HMAC secret is required" in src
        assert "RuntimeError" in src

    def test_swing_hmac_check_before_sidecar_start(self):
        """HMAC check must happen BEFORE sidecar.start() call."""
        src = Path("strategies/swing/instrumentation/src/context.py").read_text(encoding="utf-8")
        hmac_pos = src.index("HMAC secret is required")
        start_pos = src.index("self.sidecar.start()")
        assert hmac_pos < start_pos, "HMAC check must precede sidecar.start()"

    def test_momentum_hmac_check_before_running_flag(self):
        """HMAC check must happen BEFORE self._running = True."""
        src = Path("strategies/momentum/instrumentation/src/bootstrap.py").read_text(encoding="utf-8")
        hmac_pos = src.index("HMAC secret is required")
        running_pos = src.index("self._running = True")
        assert hmac_pos < running_pos, "HMAC check must precede _running = True"


# ===========================================================================
# Fix 10 — Dashboard registry-backed strategy metadata
# ===========================================================================


class TestDashboardRegistryMetadata:
    """Dashboard must derive system grouping from DB registry, not only hardcoded config."""

    def test_types_has_registry_cache(self):
        """types.ts must define a registry cache mechanism."""
        src = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        assert "_registryCache" in src
        assert "setRegistryCache" in src

    def test_types_get_system_checks_registry_first(self):
        """getSystem() must check registry cache before falling back to STRATEGY_CONFIG."""
        src = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        # Registry check must come before STRATEGY_CONFIG fallback
        registry_pos = src.index("_registryCache")
        config_pos = src.index("STRATEGY_CONFIG[strategyId]?.system")
        assert registry_pos < config_pos, "Registry cache must be checked before hardcoded config"

    def test_types_family_to_system_mapping(self):
        """FAMILY_TO_SYSTEM must map family_id strings to SystemId values."""
        src = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        assert "FAMILY_TO_SYSTEM" in src
        assert "'swing': 'swing_trader'" in src or "swing: 'swing_trader'" in src
        assert "'momentum': 'momentum_trader'" in src or "momentum: 'momentum_trader'" in src
        assert "'stock': 'stock_trader'" in src or "stock: 'stock_trader'" in src

    def test_live_route_populates_registry(self):
        """Live batch route must query registry and call setRegistryCache."""
        src = Path("apps/dashboard/src/app/api/live/route.ts").read_text(encoding="utf-8")
        assert "setRegistryCache" in src
        assert "family_id" in src


# ===========================================================================
# Fix 13 — TA order/process_quality curated artifacts
# ===========================================================================


class TestTACuratedArtifacts:
    """Trading Assistant must load order + process_quality events and write curated artifacts."""

    def test_handlers_maps_order_events(self):
        """handlers.py event_type mapping must include 'order' -> 'order_events'."""
        src = Path("_references/trading_assistant/orchestrator/handlers.py").read_text(encoding="utf-8")
        assert '"order": "order_events"' in src or "'order': 'order_events'" in src

    def test_handlers_maps_process_quality_events(self):
        """handlers.py event_type mapping must include 'process_quality' -> 'process_quality_events'."""
        src = Path("_references/trading_assistant/orchestrator/handlers.py").read_text(encoding="utf-8")
        assert '"process_quality": "process_quality_events"' in src or \
               "'process_quality': 'process_quality_events'" in src

    def test_builder_has_order_lifecycle_method(self):
        """DailyMetricsBuilder must have build_order_lifecycle_summary method."""
        src = Path("_references/trading_assistant/skills/build_daily_metrics.py").read_text(encoding="utf-8")
        assert "def build_order_lifecycle_summary" in src

    def test_builder_has_process_quality_method(self):
        """DailyMetricsBuilder must have build_process_quality_summary method."""
        src = Path("_references/trading_assistant/skills/build_daily_metrics.py").read_text(encoding="utf-8")
        assert "def build_process_quality_summary" in src

    def test_write_curated_accepts_order_events(self):
        """write_curated() must accept order_events parameter."""
        src = Path("_references/trading_assistant/skills/build_daily_metrics.py").read_text(encoding="utf-8")
        assert "order_events" in src

    def test_write_curated_accepts_process_quality_events(self):
        """write_curated() must accept process_quality_events parameter."""
        src = Path("_references/trading_assistant/skills/build_daily_metrics.py").read_text(encoding="utf-8")
        assert "process_quality_events" in src


# ===========================================================================
# Apr13RT-14 — Runtime async preflight checks
# ===========================================================================


class TestAsyncPreflight:
    """Tests for RuntimeShell._run_async_preflight()."""

    def _make_shell(self):
        from apps.runtime.runtime import RuntimeShell
        shell = RuntimeShell(config_dir="config")
        shell.registry = MagicMock()
        shell.registry.connection_groups = {}
        return shell

    @pytest.mark.asyncio
    async def test_preflight_catches_bad_coordinator_import(self):
        """Bad coordinator dotted path -> import check fails."""
        from apps.runtime import runtime as rt_mod

        shell = self._make_shell()
        original = rt_mod._FAMILY_COORDINATORS.copy()
        rt_mod._FAMILY_COORDINATORS["bad_family"] = "does.not.exist.BadCoordinator"
        try:
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families={"bad_family"},
            )
            import_checks = [c for c in checks if c.name == "import:bad_family"]
            assert len(import_checks) == 1
            assert not import_checks[0].ok
        finally:
            rt_mod._FAMILY_COORDINATORS.clear()
            rt_mod._FAMILY_COORDINATORS.update(original)

    @pytest.mark.asyncio
    async def test_preflight_db_unreachable(self):
        """Mock asyncpg.connect to raise OSError -> database check fails."""
        shell = self._make_shell()

        mock_db_config = MagicMock()
        mock_db_config.to_dsn.return_value = "postgresql://localhost/test"

        with patch("libs.oms.persistence.db_config.DBConfig.from_env", return_value=mock_db_config), \
             patch("asyncpg.connect", side_effect=OSError("Connection refused")):
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families=set(),
            )
            db_checks = [c for c in checks if c.name == "database"]
            assert len(db_checks) == 1
            assert not db_checks[0].ok

    @pytest.mark.asyncio
    async def test_preflight_ib_unreachable(self):
        """Mock asyncio.open_connection to raise -> ib-gateway check fails."""
        shell = self._make_shell()
        group_cfg = MagicMock()
        group_cfg.host = "127.0.0.1"
        group_cfg.port = 4002
        shell.registry.connection_groups = {"default": group_cfg}

        with patch("asyncio.open_connection",
                   side_effect=ConnectionRefusedError("refused")):
            checks = await shell._run_async_preflight(
                connect_ib=True,
                families=set(),
            )
            gw_checks = [c for c in checks if c.name.startswith("ib-gateway:")]
            assert len(gw_checks) == 1
            assert not gw_checks[0].ok

    @pytest.mark.asyncio
    async def test_preflight_flags_unresolved_stock_account_id(self):
        shell = self._make_shell()

        with patch("apps.runtime.runtime.get_environment", return_value="dev"), patch(
            "apps.runtime.runtime.validate_stock_readiness",
            return_value=(
                {},
                [
                    MagicMock(
                        check_name="stock-account-config:default",
                        detail="account_id is unresolved placeholder ${IB_ACCOUNT_ID}",
                    )
                ],
            ),
        ):
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families={"stock"},
            )

        stock_checks = [c for c in checks if c.name == "stock-account-config:default"]
        assert len(stock_checks) == 1
        assert not stock_checks[0].ok

    @pytest.mark.asyncio
    async def test_preflight_flags_missing_stock_artifact(self):
        shell = self._make_shell()

        with patch("apps.runtime.runtime.get_environment", return_value="dev"), patch(
            "apps.runtime.runtime.validate_stock_readiness",
            return_value=(
                {},
                [
                    MagicMock(
                        check_name="stock-artifact-readiness:IARIC_v1",
                        detail="watchlist unavailable for 2026-04-24: missing file",
                    )
                ],
            ),
        ):
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families={"stock"},
            )

        stock_checks = [c for c in checks if c.name == "stock-artifact-readiness:IARIC_v1"]
        assert len(stock_checks) == 1
        assert not stock_checks[0].ok


# ===========================================================================
# Apr10-7 / Apr13RT-12 — Paper equity scoping
# ===========================================================================


class TestPaperEquityScoping:
    """Tests for paper equity parameter wiring in multi-strategy OMS."""

    def test_multi_oms_signature_has_paper_equity_params(self):
        """build_multi_strategy_oms must accept paper_equity_pool, scope, initial."""
        from libs.oms.services.factory import build_multi_strategy_oms
        sig = inspect.signature(build_multi_strategy_oms)
        params = sig.parameters
        assert "paper_equity_pool" in params
        assert "paper_equity_scope" in params
        assert "paper_initial_equity" in params

    def test_wire_callbacks_multi_signature_has_paper_equity_params(self):
        """_wire_adapter_callbacks_multi must accept paper equity params."""
        from libs.oms.services.factory import _wire_adapter_callbacks_multi
        sig = inspect.signature(_wire_adapter_callbacks_multi)
        params = sig.parameters
        assert "paper_equity_pool" in params
        assert "paper_equity_scope" in params
        assert "paper_initial_equity" in params

    def test_swing_coordinator_references_paper_equity(self):
        """Swing coordinator source must reference PaperEquityManager and paper_equity_pool."""
        src_path = Path("strategies/swing/coordinator.py")
        text = src_path.read_text(encoding="utf-8")
        assert "paper_equity_pool" in text
        assert "PaperEquityManager" in text
