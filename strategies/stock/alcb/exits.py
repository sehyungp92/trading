"""Trade management helpers for ALCB."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from statistics import fmean

from .config import StrategySettings
from .data import StrategyDataStore
from .models import Bar, Campaign, CampaignState, CandidateItem, Direction, PositionState, ResearchDailyBar
from .signals import adx_from_bars, atr_from_bars, close_location_value, compute_campaign_avwap, compute_weekly_vwap, detect_compression_box, ema


def business_days_between(start: date, end: date) -> int:
    if end < start:
        return 0
    days = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return max(0, days - 1)


def _directional_progress(direction: Direction, reference: float, price: float) -> float:
    return (price - reference) if direction == Direction.LONG else (reference - price)


def tp1_hit(position: PositionState, last_price: float) -> bool:
    if position.partial_taken:
        return False
    if position.direction == Direction.LONG:
        return last_price >= position.tp1_price
    return last_price <= position.tp1_price


def tp2_hit(position: PositionState, last_price: float) -> bool:
    if position.tp2_taken:
        return False
    if position.direction == Direction.LONG:
        return last_price >= position.tp2_price
    return last_price <= position.tp2_price


def breakeven_plus_buffer(position: PositionState, daily_atr: float, settings: StrategySettings) -> float:
    buffer = settings.breakeven_buffer_atr * daily_atr
    if position.direction == Direction.LONG:
        return position.entry_price + buffer
    return position.entry_price - buffer


def ratchet_runner_stop(position: PositionState, bars_4h: list[Bar], settings: StrategySettings, *, avwap: float = 0.0) -> tuple[float, dict]:
    if not bars_4h:
        return position.current_stop, {"binding": "unchanged"}
    atr14 = atr_from_bars(bars_4h, 14)
    atr50 = atr_from_bars(bars_4h, 50)
    adx_val = adx_from_bars(bars_4h[-30:], 14)
    recent = bars_4h[-3:]
    latest = bars_4h[-1]
    closes_4h = [bar.close for bar in bars_4h]
    ema50_4h_vals = ema(closes_4h, min(50, len(closes_4h)))
    ema50_4h = ema50_4h_vals[-1] if ema50_4h_vals else 0.0
    vol_sample = bars_4h[-20:] if len(bars_4h) >= 20 else bars_4h
    avg_vol = fmean([bar.volume for bar in vol_sample]) if vol_sample else 0.0
    bar_range = latest.high - latest.low
    bar_rvol = latest.volume / avg_vol if avg_vol > 0 else 0.0
    clv = close_location_value(latest)
    # Adaptive trend strength detection
    atr_ratio = atr14 / atr50 if atr50 > 0 else 1.0
    strong_trend = adx_val >= settings.runner_adx_strong_threshold and atr_ratio >= settings.runner_atr_expansion_threshold
    if strong_trend:
        blend = min(max((adx_val - settings.runner_adx_strong_threshold) / 10.0, 0.0), 1.0)
        structure_mult = settings.runner_structure_atr_base + blend * (settings.runner_structure_atr_strong - settings.runner_structure_atr_base)
        ema_buffer = 0.50
        profit_ratchet_r = settings.runner_profit_ratchet_r_base + blend * (settings.runner_profit_ratchet_r_strong - settings.runner_profit_ratchet_r_base)
    else:
        structure_mult = settings.runner_structure_atr_base
        ema_buffer = 0.25
        profit_ratchet_r = settings.runner_profit_ratchet_r_base
    is_long = position.direction == Direction.LONG
    if is_long:
        structure_val = min(bar.low for bar in recent)
        structure_trail = structure_val - (structure_mult * atr14)
    else:
        structure_val = max(bar.high for bar in recent)
        structure_trail = structure_val + (structure_mult * atr14)
    trails: dict[str, float] = {"structure": structure_trail}
    if ema50_4h > 0 and len(bars_4h) >= 20:
        ema50_trail = (ema50_4h - ema_buffer * atr14) if is_long else (ema50_4h + ema_buffer * atr14)
        trails["ema50"] = ema50_trail
    if bar_range > 2 * atr14 and bar_rvol > 1.5:
        if (is_long and clv < 0.30) or (not is_long and clv > 0.70):
            expansion_trail = (latest.close - 0.5 * atr14) if is_long else (latest.close + 0.5 * atr14)
            trails["expansion"] = expansion_trail
    if avwap > 0 and atr14 > 0:
        dist = ((latest.close - avwap) if is_long else (avwap - latest.close)) / atr14
        if dist > 2.0:
            avwap_trail = (latest.close - 0.75 * atr14) if is_long else (latest.close + 0.75 * atr14)
            trails["avwap"] = avwap_trail
    if position.unrealized_r(latest.close) >= profit_ratchet_r:
        profit_trail = (latest.close - atr14) if is_long else (latest.close + atr14)
        trails["profit_ratchet"] = profit_trail
    if is_long:
        trail = max(trails.values())
        new_stop = max(position.current_stop, trail)
    else:
        trail = min(trails.values())
        new_stop = min(position.current_stop, trail)
    binding = max(trails, key=trails.get) if is_long else min(trails, key=trails.get)  # type: ignore[arg-type]
    detail = {k: round(v, 6) for k, v in trails.items()}
    detail["binding"] = binding
    detail["strong_trend"] = strong_trend
    return new_stop, detail


def stale_exit_needed(position: PositionState, now: datetime, last_price: float, settings: StrategySettings) -> bool:
    trade_days = business_days_between(position.entry_time.date(), now.date()) + 1
    if trade_days < settings.stale_warn_days:
        return False
    if trade_days < settings.stale_exit_days:
        return False
    r_threshold = settings.stale_exit_runner_r_threshold if position.partial_taken else 0.5
    return position.unrealized_r(last_price) < r_threshold


def update_dirty_state(
    campaign: Campaign,
    latest_close: float,
    settings: StrategySettings,
    as_of: date,
    *,
    daily_bars: list[ResearchDailyBar] | None = None,
) -> None:
    if campaign.breakout is None or campaign.box is None:
        return
    failed = latest_close < campaign.box.high if campaign.breakout.direction == Direction.LONG else latest_close > campaign.box.low
    breakout_date = date.fromisoformat(campaign.breakout.breakout_date)
    if failed and business_days_between(breakout_date, as_of) <= settings.dirty_break_fail_days:
        campaign.state = CampaignState.DIRTY
        campaign.reentry_block_same_direction = True
        campaign.reentry_block_opposite_enhanced = True
        campaign.dirty_since = as_of.isoformat()
    elif campaign.state == CampaignState.DIRTY and campaign.dirty_since:
        dirty_since = date.fromisoformat(campaign.dirty_since)
        min_reset_days = max(
            settings.dirty_reset_days,
            max(1, campaign.box.L_used // 2),
        )
        if business_days_between(dirty_since, as_of) >= min_reset_days:
            if daily_bars is not None:
                new_box = detect_compression_box(daily_bars, settings)
                if new_box is None:
                    return
                campaign.box = new_box
            campaign.reentry_block_same_direction = False
            campaign.reentry_block_opposite_enhanced = False
            campaign.state = CampaignState.COMPRESSION


def maybe_enable_continuation(campaign: Campaign, latest_close: float, daily_atr: float, settings: StrategySettings) -> None:
    if campaign.box is None or campaign.breakout is None:
        return
    if campaign.breakout.direction == Direction.LONG:
        measured_move = campaign.box.high + (settings.continuation_box_mult * campaign.box.height)
        r_proxy = (latest_close - campaign.box.high) / max(daily_atr, 1e-9)
        triggered = latest_close >= measured_move or r_proxy >= settings.continuation_r_mult
    else:
        measured_move = campaign.box.low - (settings.continuation_box_mult * campaign.box.height)
        r_proxy = (campaign.box.low - latest_close) / max(daily_atr, 1e-9)
        triggered = latest_close <= measured_move or r_proxy >= settings.continuation_r_mult
    if triggered:
        campaign.continuation_enabled = True
        campaign.state = CampaignState.CONTINUATION


def gap_through_stop(position: PositionState, opening_price: float) -> bool:
    if position.direction == Direction.LONG:
        return opening_price <= position.current_stop
    return opening_price >= position.current_stop


def add_trigger_price(item: CandidateItem, campaign: Campaign, store: StrategyDataStore) -> float | None:
    bars = store.bars_30m(item.symbol)
    if len(bars) < 2 or campaign.breakout is None:
        return None
    latest = bars[-1]
    ref_candidates = [
        compute_weekly_vwap(bars),
        compute_campaign_avwap(bars, campaign.avwap_anchor_ts),
    ]
    closes = [bar.close for bar in bars]
    ema20 = ema(closes, min(20, len(closes)))
    if ema20:
        ref_candidates.append(ema20[-1])
    direction = campaign.breakout.direction
    for ref in ref_candidates:
        if ref <= 0:
            continue
        touched = latest.low <= ref if direction == Direction.LONG else latest.high >= ref
        reclaimed = latest.close > ref if direction == Direction.LONG else latest.close < ref
        strong_close = close_location_value(latest) >= 0.55 if direction == Direction.LONG else close_location_value(latest) <= 0.45
        if touched and reclaimed and strong_close:
            return float(latest.close)
    return None


def can_add(item: CandidateItem, campaign: Campaign, state_position: PositionState | None, store: StrategyDataStore, settings: StrategySettings) -> bool:
    if state_position is None or campaign.breakout is None:
        return False
    if not campaign.profit_funded or campaign.add_count >= settings.max_adds:
        return False
    if campaign.box is None:
        return False
    bars = store.bars_30m(item.symbol)
    if not bars:
        return False
    avwap = compute_campaign_avwap(bars, campaign.avwap_anchor_ts)
    last_price = bars[-1].close
    daily_atr = atr_from_bars(item.daily_bars, 14)
    extension = _directional_progress(campaign.breakout.direction, avwap, last_price) / max(daily_atr, 1e-9)
    if extension > 2.0:
        return False
    return add_trigger_price(item, campaign, store) is not None
