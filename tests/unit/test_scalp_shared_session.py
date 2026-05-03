from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from strategies.scalp._shared.session import ScalpSessionBlock, entries_allowed, get_session_block, must_flatten


ET = ZoneInfo("America/New_York")


def test_scalp_session_blocks_and_entry_permissions() -> None:
    assert get_session_block(datetime(2026, 4, 29, 9, 0, tzinfo=ET)) is ScalpSessionBlock.PRE_MARKET
    assert get_session_block(datetime(2026, 4, 29, 9, 45, tzinfo=ET)) is ScalpSessionBlock.IVB_FORMING
    assert get_session_block(datetime(2026, 4, 29, 10, 15, tzinfo=ET)) is ScalpSessionBlock.RTH_PRIME
    assert entries_allowed(ScalpSessionBlock.IVB_FORMING, "po3_reversal")
    assert not entries_allowed(ScalpSessionBlock.IVB_FORMING, "ivb_auction")
    assert must_flatten(datetime(2026, 4, 29, 15, 55, tzinfo=ET))

