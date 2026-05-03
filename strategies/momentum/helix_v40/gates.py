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
    CHOP_ZONE_VOL_LOW, CHOP_ZONE_VOL_HIGH,
    CHOP_LONG_QUALITY_GATE, CHOP_LONG_MIN_SCORE, CHOP_LONG_REQUIRE_STRONG_TREND,
    NOON_QUALITY_GATE, NOON_QUALITY_HOUR_ET, NOON_QUALITY_MIN_SCORE, NOON_QUALITY_REQUIRE_STRONG_TREND,
    HIGH_VOL_CONTINUATION_REOPEN_ENABLED, HIGH_VOL_CONTINUATION_MAX_VOL_PCT,
    HIGH_VOL_CONTINUATION_MIN_ALIGNMENT_SCORE, HIGH_VOL_CONTINUATION_REQUIRE_STRONG_TREND,
    HIGH_VOL_RETEST_ENABLED, HIGH_VOL_RETEST_MIN_VOL_PCT, HIGH_VOL_RETEST_MAX_VOL_PCT,
    HIGH_VOL_M_CONT_ENABLED, HIGH_VOL_M_CONT_MIN_VOL_PCT, HIGH_VOL_M_CONT_MAX_VOL_PCT,
    NORM_VOL_M_SHORT_ENABLED, NORM_VOL_M_SHORT_MIN_VOL_PCT, NORM_VOL_M_SHORT_MAX_VOL_PCT,
    NORM_VOL_M_SHORT_ALLOW_RTH_DEAD, NORM_VOL_M_SHORT_ALLOW_RTH_PRIME2,
    NORM_VOL_M_SHORT_ALLOW_ETH_QUALITY_PM,
    SHORT_SCORE0_REOPEN_ENABLED, SHORT_SCORE0_ALLOW_RTH_DEAD,
    SHORT_SCORE0_ALLOW_RTH_PRIME2, SHORT_SCORE0_ALLOW_ETH_QUALITY_PM,
    SHORT_SCORE0_MIN_VOL_PCT, SHORT_SCORE0_MAX_VOL_PCT,
    FAILED_RECLAIM_SHORT_ENABLED, FAILED_RECLAIM_SHORT_MIN_VOL_PCT,
    FAILED_RECLAIM_SHORT_MAX_VOL_PCT, FAILED_RECLAIM_SHORT_ALLOW_RTH_DEAD,
    FAILED_RECLAIM_SHORT_ALLOW_RTH_PRIME2, FAILED_RECLAIM_SHORT_ALLOW_ETH_QUALITY_PM,
    FAILED_RECLAIM_SHORT_MIN_CONTEXT_SCORE,
    MICRO_TREND_RETEST_ENABLED, MICRO_TREND_RETEST_MIN_VOL_PCT,
    MICRO_TREND_RETEST_MAX_VOL_PCT, MICRO_TREND_RETEST_ALLOW_RTH_PRIME1,
    MICRO_TREND_RETEST_ALLOW_RTH_PRIME2, MICRO_TREND_RETEST_ALLOW_RTH_DEAD,
    MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_AM, MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_PM,
    MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME1,
    MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME2,
    MICRO_TREND_RETEST_ALLOW_SHORT_RTH_DEAD,
    MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_AM,
    MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_PM,
    MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT,
    M15_TREND_RETEST_ENABLED, M15_TREND_RETEST_MIN_VOL_PCT,
    M15_TREND_RETEST_MAX_VOL_PCT, M15_TREND_RETEST_ALLOW_RTH_PRIME1,
    M15_TREND_RETEST_ALLOW_RTH_PRIME2, M15_TREND_RETEST_ALLOW_RTH_DEAD,
    M15_TREND_RETEST_ALLOW_ETH_QUALITY_AM, M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM,
    M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT,
    CLASS_F_MIN_VOL_PCT, CLASS_F_BLOCK_ETH_QUALITY_AM, CLASS_F_BLOCK_RTH_PRIME1,
    HEAT_CAP_R, HEAT_CAP_DIR_R,
    ETH_EUROPE_REGIME_MIN_TREND, ETH_EUROPE_REGIME_MAX_VOL,
)
from .context import context_quality_score, meets_context_quality
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


def allow_selective_short_score0_reopen(
    setup: Setup,
    block: SessionBlock,
    vol_pct: float,
) -> bool:
    if not SHORT_SCORE0_REOPEN_ENABLED:
        return False
    if setup.cls != SetupClass.M or setup.direction != -1 or setup.alignment_score != 0:
        return False
    if vol_pct < SHORT_SCORE0_MIN_VOL_PCT or vol_pct > SHORT_SCORE0_MAX_VOL_PCT:
        return False

    return (
        (block == SessionBlock.RTH_DEAD and SHORT_SCORE0_ALLOW_RTH_DEAD)
        or (block == SessionBlock.RTH_PRIME2 and SHORT_SCORE0_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and SHORT_SCORE0_ALLOW_ETH_QUALITY_PM)
    )


def allow_high_vol_continuation_reopen(setup: Setup, vol_pct: float) -> bool:
    if not HIGH_VOL_CONTINUATION_REOPEN_ENABLED:
        return False
    if setup.direction != 1 or setup.cls not in {SetupClass.M, SetupClass.T}:
        return False
    if vol_pct <= HIGH_VOL_M_THRESHOLD or vol_pct > HIGH_VOL_CONTINUATION_MAX_VOL_PCT:
        return False
    if setup.alignment_score < HIGH_VOL_CONTINUATION_MIN_ALIGNMENT_SCORE:
        return False
    if HIGH_VOL_CONTINUATION_REQUIRE_STRONG_TREND and not setup.strong_trend:
        return False
    return True


def allow_high_vol_structural_retest(setup: Setup, vol_pct: float) -> bool:
    if not HIGH_VOL_RETEST_ENABLED:
        return False
    if setup.signal_variant != "high_vol_retest":
        return False
    return HIGH_VOL_RETEST_MIN_VOL_PCT <= vol_pct <= HIGH_VOL_RETEST_MAX_VOL_PCT


def allow_high_vol_structural_continuation(setup: Setup, vol_pct: float) -> bool:
    if not HIGH_VOL_M_CONT_ENABLED:
        return False
    if setup.signal_variant != "high_vol_m_cont":
        return False
    return HIGH_VOL_M_CONT_MIN_VOL_PCT <= vol_pct <= HIGH_VOL_M_CONT_MAX_VOL_PCT


def allow_normal_vol_m_short_variant(
    setup: Setup,
    block: SessionBlock,
    vol_pct: float,
) -> bool:
    if not NORM_VOL_M_SHORT_ENABLED:
        return False
    if setup.signal_variant != "normal_vol_m_short":
        return False
    if vol_pct < NORM_VOL_M_SHORT_MIN_VOL_PCT or vol_pct > NORM_VOL_M_SHORT_MAX_VOL_PCT:
        return False

    session_ok = (
        (block == SessionBlock.RTH_DEAD and NORM_VOL_M_SHORT_ALLOW_RTH_DEAD)
        or (block == SessionBlock.RTH_PRIME2 and NORM_VOL_M_SHORT_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and NORM_VOL_M_SHORT_ALLOW_ETH_QUALITY_PM)
    )
    if not session_ok:
        return False

    return meets_context_quality(
        0,
        context_quality_score(setup, block, vol_pct),
    )


def allow_failed_reclaim_short_variant(
    setup: Setup,
    block: SessionBlock,
    vol_pct: float,
) -> bool:
    if not FAILED_RECLAIM_SHORT_ENABLED:
        return False
    if setup.signal_variant != "failed_reclaim_short":
        return False
    if vol_pct < FAILED_RECLAIM_SHORT_MIN_VOL_PCT or vol_pct > FAILED_RECLAIM_SHORT_MAX_VOL_PCT:
        return False

    session_ok = (
        (block == SessionBlock.RTH_DEAD and FAILED_RECLAIM_SHORT_ALLOW_RTH_DEAD)
        or (block == SessionBlock.RTH_PRIME2 and FAILED_RECLAIM_SHORT_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.ETH_QUALITY_PM and FAILED_RECLAIM_SHORT_ALLOW_ETH_QUALITY_PM)
    )
    if not session_ok:
        return False

    return meets_context_quality(
        FAILED_RECLAIM_SHORT_MIN_CONTEXT_SCORE,
        context_quality_score(setup, block, vol_pct),
    )


def _fallback_flag(value: object, default: bool) -> bool:
    return default if value is None else bool(value)


def _micro_short_session_allowed(block: SessionBlock) -> bool:
    shared = {
        SessionBlock.RTH_PRIME1: MICRO_TREND_RETEST_ALLOW_RTH_PRIME1,
        SessionBlock.RTH_PRIME2: MICRO_TREND_RETEST_ALLOW_RTH_PRIME2,
        SessionBlock.RTH_DEAD: MICRO_TREND_RETEST_ALLOW_RTH_DEAD,
        SessionBlock.ETH_QUALITY_AM: MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_AM,
        SessionBlock.ETH_QUALITY_PM: MICRO_TREND_RETEST_ALLOW_ETH_QUALITY_PM,
    }
    short_specific = {
        SessionBlock.RTH_PRIME1: MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME1,
        SessionBlock.RTH_PRIME2: MICRO_TREND_RETEST_ALLOW_SHORT_RTH_PRIME2,
        SessionBlock.RTH_DEAD: MICRO_TREND_RETEST_ALLOW_SHORT_RTH_DEAD,
        SessionBlock.ETH_QUALITY_AM: MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_AM,
        SessionBlock.ETH_QUALITY_PM: MICRO_TREND_RETEST_ALLOW_SHORT_ETH_QUALITY_PM,
    }
    return _fallback_flag(short_specific.get(block), bool(shared.get(block, False)))


def _m15_session_allowed(block: SessionBlock) -> bool:
    return (
        (block == SessionBlock.RTH_PRIME1 and M15_TREND_RETEST_ALLOW_RTH_PRIME1)
        or (block == SessionBlock.RTH_PRIME2 and M15_TREND_RETEST_ALLOW_RTH_PRIME2)
        or (block == SessionBlock.RTH_DEAD and M15_TREND_RETEST_ALLOW_RTH_DEAD)
        or (block == SessionBlock.ETH_QUALITY_AM and M15_TREND_RETEST_ALLOW_ETH_QUALITY_AM)
        or (block == SessionBlock.ETH_QUALITY_PM and M15_TREND_RETEST_ALLOW_ETH_QUALITY_PM)
    )


def allow_micro_trend_retest_short_variant(
    setup: Setup,
    block: SessionBlock,
    vol_pct: float,
) -> bool:
    if not MICRO_TREND_RETEST_ENABLED:
        return False
    if setup.signal_variant != "micro_trend_retest" or setup.direction != -1:
        return False
    if vol_pct < MICRO_TREND_RETEST_MIN_VOL_PCT or vol_pct > MICRO_TREND_RETEST_MAX_VOL_PCT:
        return False

    if not _micro_short_session_allowed(block):
        return False

    return meets_context_quality(
        MICRO_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT,
        context_quality_score(setup, block, vol_pct),
    )


def allow_m15_trend_retest_short_variant(
    setup: Setup,
    block: SessionBlock,
    vol_pct: float,
) -> bool:
    if not M15_TREND_RETEST_ENABLED:
        return False
    if setup.signal_variant != "m15_trend_retest" or setup.direction != -1:
        return False
    if vol_pct < M15_TREND_RETEST_MIN_VOL_PCT or vol_pct > M15_TREND_RETEST_MAX_VOL_PCT:
        return False
    if not _m15_session_allowed(block):
        return False

    return meets_context_quality(
        M15_TREND_RETEST_MIN_CONTEXT_SCORE_SHORT,
        context_quality_score(setup, block, vol_pct),
    )


def allow_short_below_min_score(
    setup: Setup,
    block: SessionBlock,
    vol_pct: float,
    min_score: int,
) -> bool:
    if setup.direction != -1:
        return True
    if setup.alignment_score >= min_score:
        return True
    if allow_selective_short_score0_reopen(setup, block, vol_pct):
        return True
    if allow_normal_vol_m_short_variant(setup, block, vol_pct):
        return True
    if allow_failed_reclaim_short_variant(setup, block, vol_pct):
        return True
    if allow_micro_trend_retest_short_variant(setup, block, vol_pct):
        return True
    if allow_m15_trend_retest_short_variant(setup, block, vol_pct):
        return True
    return False


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

    if setup.cls == SetupClass.F:
        if vol.vol_pct < CLASS_F_MIN_VOL_PCT:
            reason = f"class_f_low_vol_{vol.vol_pct:.0f}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)
        if CLASS_F_BLOCK_ETH_QUALITY_AM and block == SessionBlock.ETH_QUALITY_AM:
            reason = "class_f_eth_quality_am_block"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)
        if CLASS_F_BLOCK_RTH_PRIME1 and block == SessionBlock.RTH_PRIME1:
            reason = "class_f_rth_prime1_block"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)

    if (CHOP_LONG_QUALITY_GATE
            and setup.cls == SetupClass.M
            and setup.direction == 1
            and CHOP_ZONE_VOL_LOW < vol.vol_pct < CHOP_ZONE_VOL_HIGH):
        score_ok = setup.alignment_score >= CHOP_LONG_MIN_SCORE
        trend_ok = (not CHOP_LONG_REQUIRE_STRONG_TREND) or setup.strong_trend
        if not (score_ok and trend_ok):
            reason = f"chop_long_quality_score_{setup.alignment_score}"
            if not collect_all:
                return GateResult(False, reason)
            failures.append(reason)

    if (NOON_QUALITY_GATE
            and now_et.hour == NOON_QUALITY_HOUR_ET
            and setup.direction == 1
            and setup.cls in {SetupClass.M, SetupClass.T}):
        score_ok = setup.alignment_score >= NOON_QUALITY_MIN_SCORE
        trend_ok = (not NOON_QUALITY_REQUIRE_STRONG_TREND) or setup.strong_trend
        if not (score_ok and trend_ok):
            reason = f"noon_quality_score_{setup.alignment_score}"
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
    if (
            vol.vol_pct > HIGH_VOL_M_THRESHOLD
            and setup.cls in {SetupClass.M, SetupClass.T}
            and not allow_high_vol_continuation_reopen(setup, vol.vol_pct)
            and not allow_high_vol_structural_retest(setup, vol.vol_pct)
            and not allow_high_vol_structural_continuation(setup, vol.vol_pct)
    ):
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
