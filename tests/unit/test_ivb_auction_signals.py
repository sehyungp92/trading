from __future__ import annotations

from strategies.scalp.ivb_auction.signals import score_signal
from strategies.scalp.ivb_auction.config import TradeDirection
from strategies.scalp.ivb_auction.target_model import reclaim_targets
from strategies.scalp._shared.levels import IVBLevels


def test_ivb_signal_renormalizes_when_footprint_missing() -> None:
    score = score_signal(
        regime_quality=1.0,
        retest_quality=1.0,
        target_quality=1.0,
        volatility_quality=1.0,
        time_quality=1.0,
        absorption_quality=None,
        delta_confirmation=None,
    )

    assert score.total == 100.0
    assert not score.footprint_available
    assert "absorption" not in score.available_components


def test_ivb_reclaim_targets_rotate_back_through_auction_value() -> None:
    ivb = IVBLevels.from_bounds(120.0, 80.0, poc=100.0, vah=110.0, val=90.0)

    failed_upside = reclaim_targets(entry_price=110.0, direction=TradeDirection.SHORT, ivb=ivb)
    failed_downside = reclaim_targets(entry_price=90.0, direction=TradeDirection.LONG, ivb=ivb)

    assert failed_upside.tp1 == 100.0
    assert failed_upside.tp2 == 80.0
    assert failed_downside.tp1 == 100.0
    assert failed_downside.tp2 == 120.0
