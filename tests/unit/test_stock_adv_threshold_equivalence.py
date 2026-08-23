"""The lot-unit fix must be behaviour-neutral for ALCB.

IBKR volume arrives in 100-share lots (see ``strategies.stock.volume_units``),
and the pre-fix code compared that lot-scale figure against dollar thresholds
and multiplied it directly by participation fractions.  ALCB's promoted
round_3/round_4 config was fitted under that regime -- and so was live, which
reads the same API, so the two stayed consistent with each other.

Correcting the units therefore must not move ALCB.  Every threshold compared
against dollars scales up by 100; every fraction applied to a share count
scales down by 100.  These tests pin that equivalence so a later "tidy-up" of
the odd-looking constants cannot silently re-fit the strategy.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from strategies.stock.alcb.config import StrategySettings as ALCBSettings
from strategies.stock.alcb.risk import estimate_cost_buffer_per_share
from strategies.stock.iaric.config import StrategySettings as IARICSettings
from strategies.stock.volume_units import IBKR_SHARE_VOLUME_MULTIPLIER, to_shares

LOT = IBKR_SHARE_VOLUME_MULTIPLIER

# The thresholds the promoted config was actually fitted against, expressed in
# the lot-scale dollars the pre-fix code compared them to.
LEGACY_MIN_ADV = 10_000_000.0
LEGACY_SLIPPAGE_TIERS = (50_000_000.0, 20_000_000.0)
LEGACY_MAX_PARTICIPATION = 0.01
LEGACY_THIN_PARTICIPATION = 0.005


def _legacy_slippage(lot_scale_adv: float) -> float:
    """The pre-fix tier ladder, verbatim."""
    if lot_scale_adv >= LEGACY_SLIPPAGE_TIERS[0]:
        return 0.01
    if lot_scale_adv >= LEGACY_SLIPPAGE_TIERS[1]:
        return 0.02
    return 0.03


@pytest.mark.parametrize(
    "true_adv_usd",
    [4.5e7, 4.9e9, 5.0e9, 5.1e9, 1.9e9, 2.0e9, 2.1e9, 1.5e10, 6.0e8],
)
def test_slippage_tier_is_unchanged_for_every_symbol(true_adv_usd: float) -> None:
    """A name must land in the same cost tier as it did before the fix."""
    item = SimpleNamespace(adv20_usd=true_adv_usd, median_spread_pct=0.0, tick_size=0.01)
    # Pre-fix, this same name presented as true_adv/100 and the spread term was
    # identical, so any difference in the buffer is the slippage tier moving.
    expected = _legacy_slippage(true_adv_usd / LOT)
    assert estimate_cost_buffer_per_share(item, entry_price=100.0) == pytest.approx(
        max(0.0, 0.01) + expected
    )


def test_universe_floor_is_the_same_set_of_symbols() -> None:
    assert ALCBSettings().min_adv_usd == LEGACY_MIN_ADV * LOT == 1_000_000_000.0


def test_iaric_shares_alcbs_effective_floor() -> None:
    assert IARICSettings().min_adv_usd == ALCBSettings().min_adv_usd


@pytest.mark.parametrize("lot_volume", [1_000.0, 5_000.0, 123_456.0])
def test_participation_cap_resolves_to_the_same_share_count(lot_volume: float) -> None:
    """to_shares() scales up by 100, so the fraction must scale down by 100."""
    settings = ALCBSettings()
    for new, legacy in (
        (settings.max_participation_30m, LEGACY_MAX_PARTICIPATION),
        (settings.thin_participation_30m, LEGACY_THIN_PARTICIPATION),
    ):
        assert int(to_shares(lot_volume) * new) == int(lot_volume * legacy)


def test_live_sector_participation_divisor_tracks_the_dollar_scale() -> None:
    """research_generator normalises ADV by a dollar constant; it must scale too."""
    import inspect

    from strategies.stock.alcb import research_generator

    src = inspect.getsource(research_generator._sector_metrics)
    assert "10_000_000_000.0" in src, "sector participation divisor must be 100x $100M"
