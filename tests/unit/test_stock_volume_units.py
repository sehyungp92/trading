"""Unit convention for IBKR US-equity volume across the stock family.

``reqHistoricalData(whatToShow="TRADES")`` returns volume in 100-share lots.
Both the live research generators and the backtest replay bundle draw on that
API, so raw ``volume`` is lot-scale everywhere and acquires its true units only
where it is converted into dollars or into a share count.  These tests pin the
three consumer classes so the convention cannot silently drift again.
"""
from __future__ import annotations

import numpy as np
import pytest

from strategies.stock.volume_units import (
    IBKR_SHARE_VOLUME_MULTIPLIER,
    dollar_volume,
    to_shares,
)


def test_multiplier_is_the_ibkr_lot_size() -> None:
    assert IBKR_SHARE_VOLUME_MULTIPLIER == 100.0


def test_to_shares_and_dollar_volume_agree_on_scalars_and_arrays() -> None:
    assert to_shares(7_617.0) == 761_700.0
    assert dollar_volume(200.0, 493_607.0) == pytest.approx(200.0 * 49_360_700.0)

    closes = np.array([10.0, 20.0])
    volumes = np.array([1_000.0, 2_000.0])
    np.testing.assert_allclose(
        dollar_volume(closes, volumes), closes * to_shares(volumes)
    )


def test_research_replay_reuses_the_shared_definition() -> None:
    """The backtest must not carry a second, drifting copy of the constant."""
    from backtests.stock.engine import research_replay

    assert research_replay.IBKR_SHARE_VOLUME_MULTIPLIER is IBKR_SHARE_VOLUME_MULTIPLIER
    assert research_replay.dollar_volume is dollar_volume


def test_adv_floors_are_aligned_across_the_stock_family() -> None:
    """Both screens express the same genuine $1bn floor.

    Value equivalence to the pre-fix regime is pinned separately in
    ``test_stock_adv_threshold_equivalence.py``.
    """
    from strategies.stock.alcb.config import StrategySettings as ALCBSettings
    from strategies.stock.iaric.config import StrategySettings as IARICSettings

    assert ALCBSettings().min_adv_usd == IARICSettings().min_adv_usd == 1_000_000_000.0


def test_participation_cap_is_expressed_in_shares_not_lots() -> None:
    """max_qty caps a share count, so the 30m lot volume must be converted."""
    from strategies.stock.alcb import risk as alcb_risk

    lots = 5_000.0
    participation = 0.01
    # A cap computed on raw lots would allow 50 shares; on shares, 5,000.
    naive = int(lots * participation)
    correct = int(to_shares(lots) * participation)
    assert correct == naive * 100 == 5_000

    src = alcb_risk.__dict__["add_position_quantity"].__code__.co_names
    assert "to_shares" in src, "sizing helper must route the cap through to_shares"


def test_volume_ratios_are_scale_invariant() -> None:
    """RVOL-style consumers divide volume by volume and must NOT be converted."""
    from strategies.stock.alcb.signals import compute_bar_rvol

    raw = compute_bar_rvol(1_500.0, 1_000.0)
    scaled = compute_bar_rvol(to_shares(1_500.0), to_shares(1_000.0))
    assert raw == pytest.approx(scaled)
