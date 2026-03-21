"""Gate engine: simplified gate chain for v4.0.

Removed:
- Corridor cap (only applied to A/R, both removed)
- Extension gate for R (removed)
- Chop zone hard gate (replaced by sizing penalty)
- Three-tier heat cap (single tier now)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .config import (
    Setup, SetupClass, SessionBlock,
    MIN_STOP_NORMAL, MIN_STOP_ATR_FRAC,
    MIN_STOP_EXTREME, MIN_STOP_ATR_FRAC_EXTREME,
    SPIKE_FILTER_ATR_MULT, EXTENSION_ATR_MULT,
    HIGH_VOL_M_THRESHOLD,
    HEAT_CAP_R, HEAT_CAP_DIR_R,
    ETH_EUROPE_REGIME_MIN_TREND, ETH_EUROPE_REGIME_MAX_VOL,
)
from .indicators import BarSeries, VolEngine
from .session import (
    get_session_block, entries_allowed, is_halt, is_reopen_dead,
    max_spread_for_session, class_allowed_in_session, NewsCalendar,
)

logger = logging.getLogger(__name__)


class GateResult:
    __slots__ = ("passed", "reason", "all_reasons")

    def __init__(self, passed: bool, reason: str = "ok", all_reasons: list[str] | None = None):
        self.passed = passed
        self.reason = reason
        self.all_reasons: list[str] = all_reasons or ([] if passed else [reason])

    def __bool__(self):
        return self.passed


def check_gates(
    setup: Setup,
    now_et: datetime,
    h1: BarSeries,
    daily: BarSeries,
    vol: VolEngine,
    news: NewsCalendar,
    bid: Optional[float],
    ask: Optional[float],
    open_risk_r: float,
    pending_risk_r: float,
    dir_risk_r: float,
    heat_cap_r: float,
    heat_cap_dir_r: float,
    collect_all: bool = False,
) -> GateResult:
    """Run all pre-arm gates. Simplified for v4.0 (M + F only)."""
    block = get_session_block(now_et)
    failures: list[str] = []

    # Session block
    if not class_allowed_in_session(setup.cls, block):
        reason = f"session_{block.value}_class_{setup.cls.value}"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)

    if not entries_allowed(block):
        reason = f"session_no_entries_{block.value}"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)

    # ETH_EUROPE: only longs with alignment >= 1
    if block == SessionBlock.ETH_EUROPE:
        if setup.direction == -1:
            reason = "eth_europe_short_blocked"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)
        elif setup.alignment_score < 1:
            reason = "eth_europe_long_score0"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)

    # ETH_EUROPE regime expansion: strict trend/vol/alignment/class gates
    # These always apply to ETH_EUROPE longs — the engine controls whether
    # ETH_EUROPE entries reach the gate chain at all.
    if block == SessionBlock.ETH_EUROPE and setup.direction == 1:
        if setup.cls != SetupClass.M:
            reason = f"eth_europe_regime_class_{setup.cls.value}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)
        if setup.alignment_score < 2:
            reason = f"eth_europe_regime_score_low_{setup.alignment_score}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)
        from .signals import trend_strength
        ts = trend_strength(daily)
        if ts < ETH_EUROPE_REGIME_MIN_TREND:
            reason = f"eth_europe_regime_trend_low_{ts:.2f}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)
        if vol.vol_pct > ETH_EUROPE_REGIME_MAX_VOL:
            reason = f"eth_europe_regime_vol_high_{vol.vol_pct:.0f}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)

    # M SHORT in RTH_PRIME1: 25% WR, avgR=-0.483
    if (setup.cls == SetupClass.M and setup.direction == -1
            and block == SessionBlock.RTH_PRIME1):
        reason = "session_M_short_RTH_PRIME1"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)

    # News
    if news.is_blocked(now_et):
        reason = "news_blocked"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)

    # Extension (blocks M and T, not F)
    if setup.cls in {SetupClass.M, SetupClass.T}:
        atr_d_ext = daily.current_atr()
        if atr_d_ext > 0:
            ext_fail = False
            if setup.direction == 1 and daily.last_close > daily.ema_fast() + EXTENSION_ATR_MULT * atr_d_ext:
                ext_fail = True
            if setup.direction == -1 and daily.last_close < daily.ema_fast() - EXTENSION_ATR_MULT * atr_d_ext:
                ext_fail = True
            if ext_fail:
                if not collect_all:
                    return GateResult(False, "extension_blocked")
                failures.append("extension_blocked")

    # High vol blocks M and T
    if vol.vol_pct > HIGH_VOL_M_THRESHOLD and setup.cls in {SetupClass.M, SetupClass.T}:
        reason = f"high_vol_M_{vol.vol_pct:.0f}"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)

    # Min stop distance
    atr1 = h1.current_atr()
    if vol.extreme_vol:
        min_stop = max(MIN_STOP_EXTREME, MIN_STOP_ATR_FRAC_EXTREME * atr1)
    else:
        min_stop = max(MIN_STOP_NORMAL, MIN_STOP_ATR_FRAC * atr1)
    stop_dist = abs(setup.entry_stop - setup.stop0)
    if stop_dist < min_stop:
        reason = f"min_stop_{stop_dist:.2f}<{min_stop:.2f}"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)

    # Spike filter (Class M only; Class T has its own compression check)
    if setup.cls == SetupClass.M:
        bar_rng = h1.bar_range()
        if bar_rng > SPIKE_FILTER_ATR_MULT * atr1:
            reason = f"spike_filter_{bar_rng:.1f}>{SPIKE_FILTER_ATR_MULT * atr1:.1f}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)

    # Spread gate
    if bid is None or ask is None:
        reason = "spread_missing"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)
    else:
        spread = ask - bid
        max_spread = max_spread_for_session(block)
        if spread > max_spread:
            reason = f"spread_{spread:.2f}>{max_spread:.2f}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)

    # Heat caps (single tier — no alignment-extended caps)
    total_risk = open_risk_r + pending_risk_r
    if total_risk > heat_cap_r:
        reason = f"heat_total_{total_risk:.2f}>{heat_cap_r:.2f}"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)
    if dir_risk_r > heat_cap_dir_r:
        reason = f"heat_dir_{dir_risk_r:.2f}>{heat_cap_dir_r:.2f}"
        if not collect_all:
            return GateResult(False, reason)
        failures.append(reason)

    if failures:
        return GateResult(False, failures[0], failures)

    logger.info("Gates PASS for %s %s dir=%d score=%d",
                setup.cls.value, setup.setup_id, setup.direction, setup.alignment_score)
    return GateResult(True)
