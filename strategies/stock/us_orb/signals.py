"""Signal, ranking, sizing, and exit logic."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .config import ET, StrategySettings
from .indicators import adaptive_trail_stop, atr_stop, entry_buffer, structure_stop, support_level
from .models import DangerSnapshot, DangerState, PortfolioSnapshot, RegimeSnapshot, State, SymbolContext


def quality_multiplier(score: float, settings: StrategySettings) -> float:
    if score < settings.minimum_quality_score:
        return 0.0
    anchors = (
        (settings.minimum_quality_score, 0.0),
        (settings.caution_score_floor, 0.5),
        (0.5 * (settings.caution_score_floor + settings.top_quality_score), 1.0),
        (0.5 * (settings.top_quality_score + 100.0), settings.top_quality_multiplier),
    )
    if score <= anchors[0][0]:
        return 0.0
    for i in range(1, len(anchors)):
        if score <= anchors[i][0]:
            x0, y0 = anchors[i - 1]
            x1, y1 = anchors[i]
            return y0 + (score - x0) * (y1 - y0) / (x1 - x0)
    return anchors[-1][1]


def flow_multiplier(flow_state: str, settings: StrategySettings) -> float:
    mapping = {
        "strong_inflow": settings.strong_inflow_mult,
        "mixed": settings.mixed_flow_mult,
        "outflow": settings.outflow_mult,
    }
    return mapping.get(flow_state, settings.mixed_flow_mult)


def time_decay(now: datetime, flow_regime: str = "mixed") -> tuple[float, float, int]:
    et = now.astimezone(ET)
    minutes_since_0951 = max(0, ((et.hour * 60) + et.minute) - ((9 * 60) + 51))
    if flow_regime == "strong_inflow":
        surge_rate, decay_rate, floor = 0.030, 0.009, 0.55
    elif flow_regime == "outflow":
        surge_rate, decay_rate, floor = 0.060, 0.020, 0.30
    else:
        surge_rate, decay_rate, floor = 0.045, 0.0135, 0.40
    min_surge = 3.0 + (surge_rate * minutes_since_0951)
    size_mult = max(floor, 1.0 - (decay_rate * minutes_since_0951))
    return min_surge, size_mult, minutes_since_0951


def apply_gap_policy(gap_pct: float, pre_score: float, spread_pct: float) -> tuple[bool, float, str]:
    if gap_pct >= 0.12:
        return False, 0.0, "gap_ge_12pct"
    if gap_pct <= -0.05:
        return False, 0.0, "gap_le_minus_5pct"
    if 0.08 <= gap_pct < 0.12:
        return pre_score >= 80.0, 0.65, "gap_8_to_12pct"
    if 0.05 <= gap_pct < 0.08:
        return spread_pct <= 0.0030, 0.80, "gap_5_to_8pct"
    return True, 1.0, "gap_ok"


def spread_score(spread_pct: float) -> float:
    if spread_pct <= 0.0015:
        return 10.0
    if spread_pct <= 0.0025:
        return 7.0
    if spread_pct <= 0.0035:
        return 4.0
    return 0.0


def relative_strength_score(rs_value: float) -> float:
    if rs_value >= 0.010:
        return 8.0
    if rs_value >= 0.004:
        return 6.0
    if rs_value >= 0.0:
        return 4.0
    if rs_value >= -0.005:
        return 1.0
    return 0.0


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _recent_event(events, now: datetime, seconds: int) -> bool:
    cutoff = now.timestamp() - seconds
    return any(event.timestamp() >= cutoff for event in events)


def _count_recent(events, now: datetime, seconds: int) -> int:
    cutoff = now.timestamp() - seconds
    return sum(1 for event in events if event.timestamp() >= cutoff)


def _atr(ctx: SymbolContext) -> float:
    return max(ctx.cached.atr1m14, ctx.cached.tick_size, 0.01)


def _last_bar_range(ctx: SymbolContext) -> float:
    if not ctx.bars:
        return 0.0
    bar = ctx.bars[-1]
    return max(0.0, bar.high - bar.low)


def _price_extension_atr(price: float | None, anchor: float | None, atr: float) -> float:
    if price is None or anchor is None or atr <= 0:
        return 0.0
    return max(0.0, (price - anchor) / atr)


def _pre_acceptance(ctx: SymbolContext) -> bool:
    return ctx.state in {
        State.CANDIDATE,
        State.ARMED,
        State.WAIT_BREAK,
        State.WAIT_ACCEPTANCE,
        State.READY,
        State.ORDER_SENT,
    }


def volatility_danger_snapshot(ctx: SymbolContext, now: datetime, settings: StrategySettings) -> DangerSnapshot:
    quote = ctx.quote
    last_price = ctx.last_price
    atr = _atr(ctx)
    last_bar_range = _last_bar_range(ctx)
    spread_pct = ctx.spread_pct

    current_halt = bool(quote and quote.is_halted)
    current_past_limit = bool(quote and quote.past_limit)
    current_ask_past_high = bool(quote and quote.ask_past_high)
    current_bid_past_low = bool(quote and quote.bid_past_low)

    recent_halt = _recent_event(ctx.halt_events, now, settings.recent_halt_window_s)
    recent_resume = _recent_event(ctx.resume_events, now, settings.recent_halt_window_s)
    recent_halt_or_resume = recent_halt or recent_resume
    recent_ask_past_high = _recent_event(ctx.ask_past_high_events, now, settings.direct_flag_window_s)
    recent_bid_past_low = _recent_event(ctx.bid_past_low_events, now, settings.direct_flag_window_s)
    past_limit_count = _count_recent(ctx.past_limit_events, now, settings.past_limit_window_s)

    price_extension_vwap = _price_extension_atr(last_price, ctx.vwap, atr)
    price_extension_or = _price_extension_atr(last_price, ctx.or_high, atr) if _pre_acceptance(ctx) else 0.0
    quote_gap_abnormal = bool(quote and quote.quote_gap_pct > settings.quote_gap_abnormal_pct)
    midpoint_velocity_high = bool(quote and quote.midpoint_velocity > settings.midpoint_velocity_threshold_pct_per_s)
    quote_expansion = bool(quote and quote.quote_expansion_streak >= 3)

    blocked_reasons: list[str] = []
    if current_halt:
        blocked_reasons.append("is_halted")
    if current_past_limit:
        blocked_reasons.append("past_limit")
    if current_ask_past_high:
        blocked_reasons.append("ask_past_high")
    if recent_halt_or_resume:
        blocked_reasons.append("recent_halt_or_resume")
    if past_limit_count >= settings.recent_past_limit_block_count:
        blocked_reasons.append("repeated_past_limit")
    if spread_pct > settings.blocked_spread_pct:
        blocked_reasons.append("spread_gt_60bp")
    if last_bar_range > (settings.blocked_range_atr * atr) and spread_pct > settings.blocked_range_spread_pct:
        blocked_reasons.append("range_spread_extreme")

    score = 0
    reasons: list[str] = []
    if current_past_limit or past_limit_count > 0:
        score += 100
        reasons.append("past_limit_seen")
    if recent_halt_or_resume or bool(quote and quote.resumed_from_halt):
        score += 100
        reasons.append("recent_halt_or_resume")
    if current_ask_past_high:
        score += 80
        reasons.append("ask_past_high_current")
    if current_bid_past_low:
        score += 60
        reasons.append("bid_past_low_current")
    if spread_pct > settings.blocked_spread_pct:
        score += 80
        reasons.append("spread_gt_60bp")
    elif spread_pct > settings.danger_spread_pct:
        score += 45
        reasons.append("spread_gt_45bp")
    elif spread_pct > settings.caution_spread_pct:
        score += 20
        reasons.append("spread_gt_35bp")
    if last_bar_range > (settings.blocked_range_atr * atr):
        score += 50
        reasons.append("range_gt_2atr")
    elif last_bar_range > (settings.caution_range_atr * atr):
        score += 20
        reasons.append("range_gt_1p4atr")
    if price_extension_vwap > settings.price_extension_vwap_danger_atr:
        score += 35
        reasons.append("extended_from_vwap")
    if quote_gap_abnormal:
        score += 35
        reasons.append("quote_gap_abnormal")

    danger_reasons: list[str] = []
    if spread_pct > settings.danger_spread_pct:
        danger_reasons.append("spread_danger")
    if recent_ask_past_high:
        danger_reasons.append("recent_ask_past_high")
    if recent_bid_past_low:
        danger_reasons.append("recent_bid_past_low")
    if price_extension_vwap > settings.price_extension_vwap_danger_atr:
        danger_reasons.append("extended_from_vwap")
    if price_extension_or > settings.price_extension_or_danger_atr:
        danger_reasons.append("extended_from_or_high")
    if quote_gap_abnormal:
        danger_reasons.append("quote_gap_abnormal")

    caution_reasons: list[str] = []
    if spread_pct > settings.caution_spread_pct:
        caution_reasons.append("spread_caution")
    if last_bar_range > (settings.caution_range_atr * atr):
        caution_reasons.append("range_caution")
    if midpoint_velocity_high:
        caution_reasons.append("midpoint_velocity")
    if quote_expansion:
        caution_reasons.append("quote_expansion")

    if blocked_reasons or score >= 100:
        state = DangerState.BLOCKED
    elif score >= 50 or danger_reasons:
        state = DangerState.DANGER
    elif score >= 20 or caution_reasons:
        state = DangerState.CAUTION
    else:
        state = DangerState.SAFE

    cooldown_seconds = 0
    if current_halt or current_past_limit or recent_halt_or_resume:
        cooldown_seconds = settings.halt_cooldown_s
    elif current_ask_past_high or recent_ask_past_high:
        cooldown_seconds = settings.ask_past_high_cooldown_s
    elif state in (DangerState.DANGER, DangerState.BLOCKED):
        cooldown_seconds = settings.danger_cooldown_s

    return DangerSnapshot(
        state=state,
        score=score,
        reasons=_dedupe([*blocked_reasons, *reasons, *danger_reasons, *caution_reasons]),
        cooldown_seconds=cooldown_seconds,
    )


def candidate_pre_score(ctx: SymbolContext, market: RegimeSnapshot) -> float:
    score = 0.0
    score += min(28.0, max(0.0, (ctx.surge - 3.0) * 9.0))
    score += min(18.0, max(0.0, (ctx.rvol_1m - 2.2) * 8.0))
    if ctx.tick_flow_available:
        score += (
            18.0 if ctx.imbalance_90s >= 0.30
            else 12.0 if ctx.imbalance_90s >= 0.15
            else 6.0 if ctx.imbalance_90s >= 0
            else 0.0
        )
    score += relative_strength_score(ctx.relative_strength_5m)
    score += spread_score(ctx.spread_pct)
    score += 6.0 if not market.risk_off else 0.0
    score += 5.0 if ctx.cached.catalyst_tag else 0.0
    if ctx.cached.sector and ctx.cached.sector not in ("", "unknown"):
        score += 5.0
    if ctx.vdm.state == DangerState.SAFE:
        score += 6.0
    elif ctx.vdm.state == DangerState.CAUTION:
        score += 2.0
    elif ctx.vdm.state == DangerState.DANGER:
        score -= 8.0
    return max(0.0, min(100.0, score))


def quality_score(ctx: SymbolContext, market: RegimeSnapshot, sector_penalty: bool = False) -> float:
    if ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
        return 0.0

    score = 0.0
    score += min(18.0, max(0.0, (ctx.surge - 3.0) * 6.0))
    score += min(14.0, max(0.0, (ctx.rvol_1m - 2.2) * 7.0))

    if ctx.tick_flow_available:
        if ctx.imbalance_90s >= 0.30:
            score += 20.0
        elif ctx.imbalance_90s >= 0.15:
            score += 15.0
        elif ctx.imbalance_90s >= 0.0:
            score += 8.0

    score += 5.0 if ctx.acceptance.pulled_back else 0.0
    score += 5.0 if ctx.acceptance.held_support else 0.0
    score += 5.0 if ctx.acceptance.reclaimed else 0.0
    score += 10.0 if market.regime_ok else 0.0
    score += spread_score(ctx.spread_pct)
    score += relative_strength_score(ctx.relative_strength_5m)
    score += 5.0 if ctx.cached.catalyst_tag else 0.0

    if sector_penalty:
        score -= 10.0
    if ctx.cached.float_shares is not None and ctx.cached.float_shares < 20_000_000:
        score -= 8.0
    if ctx.vdm.state == DangerState.CAUTION:
        score -= 10.0

    return max(0.0, min(100.0, score))


def live_gate_pass(ctx: SymbolContext, market: RegimeSnapshot, now: datetime, settings: StrategySettings) -> bool:
    min_surge, _, _ = time_decay(now, flow_regime=market.flow_regime)
    last_price = ctx.last_price
    atr = _atr(ctx)
    recent_ask_past_high = _recent_event(ctx.ask_past_high_events, now, settings.direct_flag_window_s)
    recent_halt_or_resume = _recent_event(ctx.halt_events, now, settings.recent_halt_window_s) or _recent_event(
        ctx.resume_events, now, settings.recent_halt_window_s
    )
    return (
        ctx.tick_flow_available
        and
        ctx.cached.trend_ok
        and ctx.surge >= min_surge
        and ctx.or_pct is not None
        and settings.min_or_pct <= ctx.or_pct <= settings.max_or_pct
        and ctx.rvol_1m >= settings.rvol_threshold
        and last_price is not None
        and ctx.vwap is not None
        and last_price >= ctx.vwap
        and ctx.spread_pct <= settings.spread_limit_pct
        and _last_bar_range(ctx) <= (settings.entry_range_atr * atr)
        and _price_extension_atr(last_price, ctx.vwap, atr) <= settings.price_extension_vwap_entry_atr
        and not recent_ask_past_high
        and not recent_halt_or_resume
        and ctx.vdm.state in (DangerState.SAFE, DangerState.CAUTION)
        and market.regime_ok
    )


def breakout_triggered(ctx: SymbolContext) -> bool:
    if ctx.or_high is None or ctx.last_price is None or ctx.vwap is None:
        return False
    return (
        ctx.vdm.state in (DangerState.SAFE, DangerState.CAUTION)
        and ctx.last_price > (ctx.or_high + 0.01)
        and ctx.last_price > ctx.vwap
        and ctx.spread_pct <= 0.0035
    )


def update_acceptance(ctx: SymbolContext) -> None:
    if ctx.acceptance.deadline is None or ctx.or_high is None or ctx.last_price is None:
        return
    support = support_level(ctx.or_high, ctx.vwap)
    if ctx.last_price <= ctx.or_high:
        ctx.acceptance.pulled_back = True
        if ctx.acceptance.retest_low is None or ctx.last_price < ctx.acceptance.retest_low:
            ctx.acceptance.retest_low = ctx.last_price
    if ctx.acceptance.pulled_back and ctx.last_price >= support:
        ctx.acceptance.held_support = True
    if ctx.acceptance.pulled_back and ctx.acceptance.held_support and ctx.last_price >= ctx.or_high:
        ctx.acceptance.reclaimed = True


def acceptance_passed(ctx: SymbolContext, now: datetime) -> bool:
    if ctx.acceptance.deadline is None or now > ctx.acceptance.deadline:
        return False
    return (
        ctx.tick_flow_available
        and
        ctx.vdm.state in (DangerState.SAFE, DangerState.CAUTION)
        and ctx.acceptance.pulled_back
        and ctx.acceptance.held_support
        and ctx.acceptance.reclaimed
        and ctx.imbalance_90s >= 0.0
        and ctx.spread_pct <= 0.0035
    )


def compute_order_plan(
    ctx: SymbolContext,
    portfolio: PortfolioSnapshot,
    market: RegimeSnapshot,
    now: datetime,
    settings: StrategySettings,
) -> tuple[float, float, float, int]:
    if ctx.or_high is None or ctx.last_price is None:
        return 0.0, 0.0, 0.0, 0
    if ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
        return 0.0, 0.0, 0.0, 0
    if ctx.vdm.state == DangerState.CAUTION and ctx.quality_score < settings.caution_quality_min:
        return 0.0, 0.0, 0.0, 0

    planned_entry = ctx.or_high + 0.01
    buffer_value = entry_buffer(ctx.last_price, ctx.spread, ctx.cached.atr1m14)
    planned_limit = planned_entry + buffer_value

    structure = structure_stop(ctx.or_high, ctx.vwap, ctx.acceptance.retest_low, buffer_value)
    atr_based = atr_stop(planned_entry, ctx.cached.atr1m14)
    final_stop = min(structure, atr_based)
    risk_per_share = planned_entry - final_stop
    if risk_per_share <= 0:
        return planned_entry, planned_limit, final_stop, 0

    _, time_mult, _ = time_decay(now, flow_regime=market.flow_regime)
    qty = (portfolio.nav * settings.base_risk_pct) / risk_per_share
    qty *= quality_multiplier(ctx.quality_score, settings)
    qty *= time_mult
    qty *= flow_multiplier(market.flow_regime, settings)
    qty *= ctx.size_penalty
    if ctx.cached.secondary_universe:
        qty *= settings.secondary_size_penalty
    if ctx.vdm.state == DangerState.CAUTION:
        qty *= settings.caution_size_penalty

    max_qty_liquidity = (ctx.last5m_value * settings.max_last5m_value_pct) / planned_entry if planned_entry > 0 else qty
    max_qty_nav = (portfolio.nav * settings.max_notional_nav_pct) / planned_entry if planned_entry > 0 else qty
    max_qty_depth = (ctx.quote.ask_size * settings.max_displayed_ask_pct) if ctx.quote and ctx.quote.ask_size > 0 else qty

    capped = min(qty, max_qty_liquidity or qty, max_qty_nav or qty, max_qty_depth or qty)
    return planned_entry, planned_limit, final_stop, max(0, int(capped))


def exit_signal(
    ctx: SymbolContext,
    market: RegimeSnapshot,
    now: datetime,
    settings: StrategySettings,
) -> tuple[str | None, float | None]:
    position = ctx.position
    last_price = ctx.last_price
    if position is None or last_price is None:
        return None, None

    position.max_favorable_price = max(position.max_favorable_price, last_price)

    if market.risk_off:
        return "exit", last_price
    if ctx.quote and ctx.quote.past_limit:
        return "exit", last_price
    if ctx.quote and ctx.quote.ask_past_high and ctx.imbalance_90s < 0:
        return "exit", last_price
    if ctx.quote and ctx.quote.resumed_from_halt and ((ctx.vwap is not None and last_price < ctx.vwap) or ctx.imbalance_90s < 0):
        return "exit", last_price
    if last_price <= position.current_stop:
        return "exit", position.current_stop

    held_minutes = max(0, int((now - position.entry_time).total_seconds() / 60))
    if held_minutes <= 15 and ctx.or_high is not None and ctx.vwap is not None:
        if last_price < ctx.or_high and last_price < ctx.vwap:
            return "exit", last_price

    if held_minutes >= settings.scratch_after_minutes:
        unrealized_r = (last_price - position.entry_price) / position.initial_risk_per_share if position.initial_risk_per_share > 0 else 0.0
        if unrealized_r < settings.scratch_min_r:
            return "exit", last_price

    if not position.partial_taken and last_price >= position.entry_price + (settings.partial_r_multiple * position.initial_risk_per_share):
        return "partial", last_price

    desired_trail = adaptive_trail_stop(
        position=position,
        minutes_held=held_minutes,
        flow_regime=market.flow_regime,
        imbalance_90s=ctx.imbalance_90s,
        spread_pct=ctx.spread_pct,
        vdm_state=ctx.vdm.state,
        settings=settings,
    )
    if desired_trail > position.current_stop:
        return "trail", desired_trail
    return None, None
