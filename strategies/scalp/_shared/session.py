from __future__ import annotations

from datetime import datetime, time
from enum import Enum

from .time_utils import et_time


class ScalpSessionBlock(Enum):
    PRE_MARKET = "PRE_MARKET"
    IVB_FORMING = "IVB_FORMING"
    RTH_PRIME = "RTH_PRIME"
    RTH_ACTIVE = "RTH_ACTIVE"
    RTH_WIND_DOWN = "RTH_WIND_DOWN"
    FLATTEN_ZONE = "FLATTEN_ZONE"


RTH_OPEN = time(9, 30)
IVB_END = time(10, 0)
PRIME_END = time(10, 45)
NO_NEW_ENTRIES = time(15, 30)
FLATTEN_START = time(15, 55)


def get_session_block(now_et: datetime) -> ScalpSessionBlock:
    current = et_time(now_et)
    if current < RTH_OPEN:
        return ScalpSessionBlock.PRE_MARKET
    if current < IVB_END:
        return ScalpSessionBlock.IVB_FORMING
    if current < PRIME_END:
        return ScalpSessionBlock.RTH_PRIME
    if current < NO_NEW_ENTRIES:
        return ScalpSessionBlock.RTH_ACTIVE
    if current < FLATTEN_START:
        return ScalpSessionBlock.RTH_WIND_DOWN
    return ScalpSessionBlock.FLATTEN_ZONE


def entries_allowed(block: ScalpSessionBlock, strategy: str) -> bool:
    key = strategy.lower()
    if block is ScalpSessionBlock.FLATTEN_ZONE:
        return False
    if key in {"po3", "po3_reversal", "strategy_b"}:
        return block in {ScalpSessionBlock.IVB_FORMING, ScalpSessionBlock.RTH_PRIME}
    if key in {"ivb", "ivb_auction", "strategy_a"}:
        return block in {ScalpSessionBlock.RTH_PRIME, ScalpSessionBlock.RTH_ACTIVE}
    return block in {
        ScalpSessionBlock.IVB_FORMING,
        ScalpSessionBlock.RTH_PRIME,
        ScalpSessionBlock.RTH_ACTIVE,
    }


def must_flatten(now_et: datetime) -> bool:
    return et_time(now_et) >= FLATTEN_START

