from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.scalp.ivb_auction.footprint import FootprintBuilder
from strategies.scalp.ivb_auction.models import ScalpTick


def test_footprint_classifies_bid_ask_aggression() -> None:
    builder = FootprintBuilder(bar_seconds=30)
    ts = datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc)
    builder.on_tick(ScalpTick(ts=ts, price=100.25, size=3, bid=100.0, ask=100.25))
    builder.on_tick(ScalpTick(ts=ts + timedelta(seconds=1), price=100.0, size=2, bid=100.0, ask=100.25))
    bar = builder.flush()

    assert bar is not None
    assert bar.ask_volume == 3
    assert bar.bid_volume == 2
    assert bar.delta == 1

