"""Tests for FarmMonitor — farm connectivity detection and recovery callbacks."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from shared.ibkr_core.client.farm_monitor import FarmMonitor, FarmStatus


class _MockIB:
    """Minimal mock matching ib_async.IB errorEvent interface."""

    class _Event:
        def __init__(self):
            self._handlers: list = []

        def __iadd__(self, handler):
            self._handlers.append(handler)
            return self

        def __isub__(self, handler):
            self._handlers.remove(handler)
            return self

    def __init__(self):
        self.errorEvent = self._Event()

    def fire_error(self, reqId: int, errorCode: int, errorString: str, contract: object = None):
        for h in list(self.errorEvent._handlers):
            h(reqId, errorCode, errorString, contract)


@pytest.fixture
def mock_ib():
    return _MockIB()


@pytest.fixture
def monitor(mock_ib):
    fm = FarmMonitor(mock_ib)
    fm.start()
    yield fm
    fm.stop()


class TestParseFarmName:
    def test_extracts_usfarm(self):
        assert FarmMonitor._parse_farm_name(
            "Market data farm connection is broken:usfarm"
        ) == "usfarm"

    def test_extracts_with_trailing_space(self):
        assert FarmMonitor._parse_farm_name(
            "Market data farm connection is OK:hfarm "
        ) == "hfarm"

    def test_returns_empty_on_no_match(self):
        assert FarmMonitor._parse_farm_name("Some other error") == ""


class TestFarmStateTransitions:
    def test_broken_to_ok_fires_callback(self, mock_ib, monitor):
        cb = MagicMock()
        monitor.on_farm_recovered = cb

        mock_ib.fire_error(-1, 2103, "Market data farm connection is broken:usfarm")
        cb.assert_not_called()
        assert monitor.get_status("usfarm") == FarmStatus.BROKEN

        mock_ib.fire_error(-1, 2104, "Market data farm connection is OK:usfarm")
        cb.assert_called_once_with("usfarm")
        assert monitor.get_status("usfarm") == FarmStatus.OK

    def test_repeated_same_status_deduplicates(self, mock_ib, monitor):
        cb = MagicMock()
        monitor.on_farm_recovered = cb

        mock_ib.fire_error(-1, 2108, "Market data farm connection is inactive:usfarm")
        mock_ib.fire_error(-1, 2108, "Market data farm connection is inactive:usfarm")
        mock_ib.fire_error(-1, 2108, "Market data farm connection is inactive:usfarm")

        cb.assert_not_called()
        assert monitor.get_status("usfarm") == FarmStatus.INACTIVE

    def test_full_sequence_fires_once(self, mock_ib, monitor):
        cb = MagicMock()
        monitor.on_farm_recovered = cb

        mock_ib.fire_error(-1, 2108, "Market data farm connection is inactive:usfarm")
        mock_ib.fire_error(-1, 2119, "Market data farm is connecting:usfarm")
        mock_ib.fire_error(-1, 2104, "Market data farm connection is OK:usfarm")

        cb.assert_called_once_with("usfarm")

    def test_first_ever_ok_fires_callback(self, mock_ib, monitor):
        """Handles startup mid-recovery — first event is 2104."""
        cb = MagicMock()
        monitor.on_farm_recovered = cb

        mock_ib.fire_error(-1, 2104, "Market data farm connection is OK:usfarm")
        cb.assert_called_once_with("usfarm")

    def test_non_farm_codes_ignored(self, mock_ib, monitor):
        cb = MagicMock()
        monitor.on_farm_recovered = cb

        mock_ib.fire_error(1, 200, "No security definition found")
        cb.assert_not_called()
        assert monitor.get_status("usfarm") is None


class TestMultipleCallbacks:
    def test_ibsession_dispatches_all(self):
        """Simulate IBSession dispatching to multiple strategy callbacks."""
        mock_ib = _MockIB()
        monitor = FarmMonitor(mock_ib)
        monitor.start()

        cb1 = MagicMock()
        cb2 = MagicMock()
        callbacks = [cb1, cb2]

        def dispatch(farm_name: str):
            for cb in callbacks:
                try:
                    cb(farm_name)
                except Exception:
                    pass

        monitor.on_farm_recovered = dispatch

        mock_ib.fire_error(-1, 2104, "Market data farm connection is OK:usfarm")
        cb1.assert_called_once_with("usfarm")
        cb2.assert_called_once_with("usfarm")

        monitor.stop()

    def test_exception_in_one_callback_doesnt_block_others(self):
        """One failing callback must not prevent others from firing."""
        mock_ib = _MockIB()
        monitor = FarmMonitor(mock_ib)
        monitor.start()

        cb1 = MagicMock(side_effect=RuntimeError("boom"))
        cb2 = MagicMock()
        callbacks = [cb1, cb2]

        def dispatch(farm_name: str):
            for cb in callbacks:
                try:
                    cb(farm_name)
                except Exception:
                    pass

        monitor.on_farm_recovered = dispatch

        mock_ib.fire_error(-1, 2104, "Market data farm connection is OK:hfarm")
        cb1.assert_called_once_with("hfarm")
        cb2.assert_called_once_with("hfarm")

        monitor.stop()


class TestGetStatus:
    def test_unknown_farm_returns_none(self, monitor):
        assert monitor.get_status("nonexistent") is None

    def test_returns_current_status(self, mock_ib, monitor):
        mock_ib.fire_error(-1, 2119, "Market data farm is connecting:usfarm")
        assert monitor.get_status("usfarm") == FarmStatus.CONNECTING
