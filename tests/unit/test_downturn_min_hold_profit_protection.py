from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest

from backtests.momentum.config_downturn import DownturnBacktestConfig
from backtests.momentum.data.preprocessing import NumpyBars
from backtests.momentum.engine.downturn_engine import DownturnEngine, _ActivePosition
from strategies.momentum.downturn.bt_models import CompositeRegime, EngineTag, VolState


def _hourly_stub() -> NumpyBars:
    values = np.asarray([100.0], dtype=float)
    return NumpyBars(
        opens=values,
        highs=values,
        lows=values,
        closes=values,
        volumes=np.asarray([1.0]),
        times=np.asarray([np.datetime64("2026-03-23T10:00")]),
    )


@pytest.mark.parametrize(
    ("protection_enabled", "expected_updates"),
    [(False, 0), (True, 1)],
)
def test_min_hold_profit_protection_only_ratchets_earned_stop(
    protection_enabled: bool,
    expected_updates: int,
) -> None:
    config = DownturnBacktestConfig(
        flags=replace(
            DownturnBacktestConfig().flags,
            min_hold_period=True,
            min_hold_profit_protection=protection_enabled,
            profit_floor_trail=True,
            chandelier_trailing=True,
        ),
        param_overrides={
            "min_hold_bars": 13,
            "profit_floor_r_threshold": 1.0,
            "profit_floor_lock_pct": 0.4,
            "be_trigger_r": 0.9,
            "be_stop_buffer_mult": 0.08,
        },
    )
    engine = DownturnEngine("NQ", config)
    position = _ActivePosition(
        engine_tag=EngineTag.FADE,
        signal_class="vwap_rejection",
        entry_price=100.0,
        stop0=110.0,
        qty=1,
        entry_time=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
        entry_bar_idx=0,
        composite_regime=CompositeRegime.NEUTRAL,
        vol_state=VolState.HIGH,
        in_correction=False,
        predator=False,
        tp_schedule=[(1.8, 0.35)],
    )
    position.hold_bars_5m = 5
    engine._position = position
    engine._atr_1h = 2.0
    engine._update_protective_stop = Mock()

    engine._manage_position(
        5,
        datetime(2026, 3, 23, 10, 25, tzinfo=timezone.utc),
        100.0,
        87.5,
        88.0,
        _hourly_stub(),
        0,
    )

    assert engine._update_protective_stop.call_count == expected_updates
    if protection_enabled:
        assert position.exit_trigger == "profit_floor"
        assert position.chandelier_stop < position.entry_price
    else:
        assert position.exit_trigger == ""
        assert position.chandelier_stop == position.stop0
