"""Session classification, news guard, and time helpers (all times ET).

v4.0 changes:
- RTH_DEAD narrowed to 11:30-12:30 (opens RTH_PRIME2 at 12:30)
- ETH_EUROPE entries blocked (confirmed losers: WR=14%, avgR=-0.436 even with regime filters)
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

from .config import (
    SessionBlock, SESSION_SCHEDULE, SESSION_SIZE_MULT, MAX_SPREAD_BY_SESSION,
    NEWS_WINDOWS, SetupClass,
)


def get_session_block(now_et: datetime) -> SessionBlock:
    t = now_et.time()
    for start, end, block in SESSION_SCHEDULE:
        if start <= end:
            if start <= t < end:
                return block
        else:
            if t >= start or t < end:
                return block
    return SessionBlock.DAILY_HALT


def session_size_mult(block: SessionBlock) -> float:
    return SESSION_SIZE_MULT.get(block, 0.0)


def entries_allowed(block: SessionBlock) -> bool:
    return block in {
        SessionBlock.RTH_PRIME1, SessionBlock.RTH_PRIME2,
        SessionBlock.RTH_DEAD,
        SessionBlock.ETH_QUALITY_AM, SessionBlock.ETH_QUALITY_PM,
        SessionBlock.ETH_OVERNIGHT,
    }


def adds_allowed(block: SessionBlock, now_et: datetime) -> bool:
    if block not in {
        SessionBlock.RTH_PRIME1, SessionBlock.RTH_PRIME2,
        SessionBlock.ETH_EUROPE, SessionBlock.ETH_QUALITY_AM,
    }:
        return False
    if block in {SessionBlock.ETH_EUROPE, SessionBlock.ETH_QUALITY_AM}:
        return True
    return now_et.time() < time(15, 0)


def is_trend_only(block: SessionBlock) -> bool:
    return block in {SessionBlock.ETH_EUROPE, SessionBlock.ETH_OVERNIGHT}


def is_overnight_4h_only(block: SessionBlock) -> bool:
    return block == SessionBlock.ETH_OVERNIGHT


def is_halt(now_et: datetime) -> bool:
    return time(17, 0) <= now_et.time() < time(18, 0)


def is_reopen_dead(now_et: datetime) -> bool:
    return time(18, 0) <= now_et.time() < time(20, 0)


def is_pre_halt_cancel(now_et: datetime) -> bool:
    return now_et.time() >= time(16, 55) and now_et.time() < time(17, 0)


def is_pre_halt_add_control(now_et: datetime) -> bool:
    return now_et.time() >= time(16, 50) and now_et.time() < time(17, 0)


def max_spread_for_session(block: SessionBlock) -> float:
    return MAX_SPREAD_BY_SESSION.get(block, 1.00)


def is_rth(block: SessionBlock) -> bool:
    return block in {SessionBlock.RTH_PRIME1, SessionBlock.RTH_PRIME2}


def class_allowed_in_session(cls: SetupClass, block: SessionBlock) -> bool:
    if block in {SessionBlock.DAILY_HALT, SessionBlock.REOPEN_DEAD}:
        return False
    if block == SessionBlock.ETH_OVERNIGHT and cls != SetupClass.M:
        return False
    # Class T: allowed in RTH_PRIME1, RTH_PRIME2, RTH_DEAD, ETH_QUALITY_AM, ETH_QUALITY_PM
    # Blocked: ETH_EUROPE, ETH_OVERNIGHT
    if cls == SetupClass.T and block in {SessionBlock.ETH_EUROPE, SessionBlock.ETH_OVERNIGHT}:
        return False
    return True


def is_sunday_reopen(now_et: datetime) -> bool:
    return now_et.weekday() == 6 and now_et.time() >= time(18, 0)


class NewsCalendar:
    def __init__(self):
        self.events: list[tuple[datetime, str]] = []

    def add_event(self, event_time: datetime, event_type: str) -> None:
        self.events.append((event_time, event_type))

    def is_blocked(self, now_et: datetime) -> bool:
        for event_time, event_type in self.events:
            window = NEWS_WINDOWS.get(event_type)
            if window is None:
                continue
            pre_min, post_min = window
            block_start = event_time - timedelta(minutes=pre_min)
            block_end = event_time + timedelta(minutes=post_min)
            if block_start <= now_et <= block_end:
                return True
        return False

    def clear_past(self, now_et: datetime) -> None:
        cutoff = now_et - timedelta(hours=2)
        self.events = [(t, e) for t, e in self.events if t > cutoff]
