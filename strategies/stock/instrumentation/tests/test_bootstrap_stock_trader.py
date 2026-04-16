import asyncio
from unittest.mock import MagicMock

import pytest

from strategies.stock.instrumentation.src.bootstrap import InstrumentationManager, _load_config


class _DummyEventBus:
    def unsubscribe_all(self, queue) -> None:
        return None


class _DummyOMS:
    def __init__(self) -> None:
        self.event_bus = _DummyEventBus()

    def stream_all_events(self):
        return asyncio.Queue()


def test_load_config_applies_stock_trader_defaults():
    config = _load_config("IARIC_v1", "strategy_iaric")

    assert config["bot_id"] == "stock_trader"
    assert config["strategy_id"] == "iaric"
    assert config["strategy_type"] == "strategy_iaric"
    assert config["data_source_id"] == "ibkr_us_equities"
    assert config["heartbeat_interval_seconds"] == 30
    assert config["daily_snapshot_checkpoint_interval_seconds"] == 300
    assert config["market_snapshots"]["symbols"] == ["SPY", "QQQ", "IWM"]
    assert config["sidecar"]["hmac_secret_env"] == "INSTRUMENTATION_HMAC_SECRET"


def test_load_config_applies_alcb_strategy_identity():
    config = _load_config("ALCB_v1", "strategy_alcb")

    assert config["bot_id"] == "stock_trader"
    assert config["strategy_id"] == "alcb"
    assert config["strategy_type"] == "strategy_alcb"
    assert config["data_source_id"] == "ibkr_us_equities"
    assert config["sidecar"]["hmac_secret_env"] == "INSTRUMENTATION_HMAC_SECRET"


def test_manager_maps_config_modules_and_attachs_provider():
    manager = InstrumentationManager(
        oms=_DummyOMS(),
        strategy_id="US_ORB_v1",
        strategy_type="strategy_orb",
    )
    provider = object()

    manager.attach_data_provider(provider)

    assert manager.bot_id == "stock_trader"
    assert manager.config["strategy_id"] == "us_orb"
    assert manager.config_watcher is not None
    assert manager.config_watcher._config_modules == ["strategy_orb.config"]
    assert manager.snapshot_service._data_provider is provider
    assert manager.regime_classifier.data_provider is provider
    assert manager.config["sidecar"]["hmac_secret_env"] == "INSTRUMENTATION_HMAC_SECRET"


def test_manager_maps_alcb_config_module():
    manager = InstrumentationManager(
        oms=_DummyOMS(),
        strategy_id="ALCB_v1",
        strategy_type="strategy_alcb",
    )

    assert manager.bot_id == "stock_trader"
    assert manager.config["strategy_id"] == "alcb"
    assert manager.config_watcher is not None
    assert manager.config_watcher._config_modules == ["strategy_alcb.config"]


def test_periodic_loop_runs_trade_and_missed_backfills():
    async def _run():
        manager = InstrumentationManager(
            oms=_DummyOMS(),
            strategy_id="IARIC_v1",
            strategy_type="strategy_iaric",
        )
        provider = object()
        manager.attach_data_provider(provider)
        manager.snapshot_service.run_periodic = MagicMock()
        manager.trade_logger.run_post_exit_backfill = MagicMock()
        manager.missed_logger.run_backfill = MagicMock()
        manager.config_watcher = MagicMock()
        manager.daily_builder.build = MagicMock(return_value=MagicMock())
        manager.daily_builder.save = MagicMock()
        manager.config["daily_snapshot_checkpoint_interval_seconds"] = 0
        manager._running = True

        task = asyncio.create_task(manager._periodic_snapshot_loop(0.01))
        await asyncio.sleep(0.03)
        manager._running = False
        await task

        manager.trade_logger.run_post_exit_backfill.assert_called_with(provider)
        manager.missed_logger.run_backfill.assert_called_with(provider)

    asyncio.run(_run())


def test_periodic_loop_checkpoints_daily_snapshot():
    async def _run():
        manager = InstrumentationManager(
            oms=_DummyOMS(),
            strategy_id="US_ORB_v1",
            strategy_type="strategy_orb",
        )
        manager.snapshot_service.run_periodic = MagicMock()
        manager.trade_logger.run_post_exit_backfill = MagicMock()
        manager.missed_logger.run_backfill = MagicMock()
        manager.config_watcher = MagicMock()
        manager.daily_builder.build = MagicMock(return_value=MagicMock())
        manager.daily_builder.save = MagicMock()
        manager.config["daily_snapshot_checkpoint_interval_seconds"] = 1
        manager._last_snapshot_checkpoint_at = 0.0
        manager._running = True

        task = asyncio.create_task(manager._periodic_snapshot_loop(0.01))
        await asyncio.sleep(0.03)
        manager._running = False
        await task

        manager.daily_builder.build.assert_called()
        manager.daily_builder.save.assert_called()

    asyncio.run(_run())


def test_start_requires_sidecar_auth_in_paper(monkeypatch):
    async def _run():
        monkeypatch.setenv("TRADING_MODE", "paper")
        monkeypatch.delenv("INSTRUMENTATION_HMAC_SECRET", raising=False)

        manager = InstrumentationManager(
            oms=_DummyOMS(),
            strategy_id="US_ORB_v1",
            strategy_type="strategy_orb",
        )

        with pytest.raises(RuntimeError, match="HMAC secret"):
            await manager.start()
        assert manager._running is False

    asyncio.run(_run())
