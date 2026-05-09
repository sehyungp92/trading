"""Tests for live crisis-detector defaults."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from regime.crisis import config as C


def test_live_crisis_defaults_match_latest_optimized_config() -> None:
    optimized_path = Path("backtests/output/regime/crisis/round_9/optimized_config.json")
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))

    for key, expected in optimized.items():
        actual = getattr(C, key)
        if isinstance(expected, float):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected


def test_watch_is_internal_buffer_only() -> None:
    assert C.RISK_MULT_WATCH == 1.0
    assert C.DD_TIER_MULT_WATCH == 1.0
    assert C.ADVISORY_WATCH_MIN_PRIMARY > C.WATCH_MIN_PRIMARY
    assert C.ADVISORY_WATCH_MIN_WARNING >= C.WARNING_MIN_PRIMARY
