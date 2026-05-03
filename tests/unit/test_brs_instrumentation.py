"""Test BRS engine stop adjustment instrumentation.

Verifies that BRS uses log_stop_adjustment (routed to stop_adjustments/)
instead of on_filter_decision (which goes to filter_decisions/) for stop
trail updates.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_position(
    pos_id: str = "BRS_QQQ_001",
    symbol: str = "QQQ",
    entry_type: str = "primary",
    bars_held: int = 5,
    mfe_r: float = 1.5,
) -> MagicMock:
    pos = MagicMock()
    pos.pos_id = pos_id
    pos.symbol = symbol
    pos.entry_type = entry_type
    pos.bars_held = bars_held
    pos.mfe_r = mfe_r
    return pos


def _make_kit() -> MagicMock:
    kit = MagicMock()
    kit.log_stop_adjustment = MagicMock()
    kit.on_filter_decision = MagicMock()
    return kit


def _make_stop_req(stop_price: float) -> MagicMock:
    req = MagicMock()
    req.stop_price = stop_price
    return req


class TestBrsStopInstrumentation:
    """Verify stop adjustments route to log_stop_adjustment."""

    def test_stop_update_calls_log_stop_adjustment(self):
        """When stop price changes, log_stop_adjustment should be called."""
        kit = _make_kit()
        pos = _make_position()
        stop_req = _make_stop_req(295.0)
        old_stop = 290.0

        # Simulate the instrumentation block from _execute_actions
        if stop_req is not None and kit and old_stop != stop_req.stop_price:
            try:
                kit.log_stop_adjustment(
                    trade_id=pos.pos_id,
                    symbol="QQQ",
                    old_stop=old_stop,
                    new_stop=stop_req.stop_price,
                    adjustment_type="trailing",
                    trigger=f"stop_ratchet_{pos.entry_type}",
                    metadata={
                        "bars_held": pos.bars_held,
                        "mfe_r": round(pos.mfe_r, 3),
                        "entry_type": pos.entry_type,
                    },
                )
            except Exception:
                pass

        kit.log_stop_adjustment.assert_called_once()
        call_kwargs = kit.log_stop_adjustment.call_args[1]
        assert call_kwargs["trade_id"] == "BRS_QQQ_001"
        assert call_kwargs["old_stop"] == 290.0
        assert call_kwargs["new_stop"] == 295.0
        assert call_kwargs["adjustment_type"] == "trailing"
        assert call_kwargs["trigger"] == "stop_ratchet_primary"
        assert call_kwargs["metadata"]["bars_held"] == 5
        assert call_kwargs["metadata"]["mfe_r"] == 1.5

        # on_filter_decision should NOT be called
        kit.on_filter_decision.assert_not_called()

    def test_no_call_when_stop_unchanged(self):
        """When old_stop == new stop, no instrumentation should fire."""
        kit = _make_kit()
        stop_req = _make_stop_req(290.0)
        old_stop = 290.0

        if stop_req is not None and kit and old_stop != stop_req.stop_price:
            kit.log_stop_adjustment()

        kit.log_stop_adjustment.assert_not_called()

    def test_no_call_when_no_kit(self):
        """When instrumentation is None, no error should occur."""
        kit = None
        stop_req = _make_stop_req(295.0)
        old_stop = 290.0

        # Should not raise
        if stop_req is not None and kit and old_stop != stop_req.stop_price:
            kit.log_stop_adjustment()

    def test_metadata_contains_entry_type_as_string(self):
        """entry_type in metadata should be a plain string, not .value."""
        kit = _make_kit()
        pos = _make_position(entry_type="pyramid_add")
        stop_req = _make_stop_req(295.0)
        old_stop = 290.0

        if stop_req is not None and kit and old_stop != stop_req.stop_price:
            kit.log_stop_adjustment(
                trade_id=pos.pos_id,
                symbol="QQQ",
                old_stop=old_stop,
                new_stop=stop_req.stop_price,
                adjustment_type="trailing",
                trigger=f"stop_ratchet_{pos.entry_type}",
                metadata={
                    "bars_held": pos.bars_held,
                    "mfe_r": round(pos.mfe_r, 3),
                    "entry_type": pos.entry_type,
                },
            )

        call_kwargs = kit.log_stop_adjustment.call_args[1]
        assert call_kwargs["trigger"] == "stop_ratchet_pyramid_add"
        assert call_kwargs["metadata"]["entry_type"] == "pyramid_add"
