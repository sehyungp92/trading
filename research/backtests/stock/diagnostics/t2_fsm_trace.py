"""Trace FSM progression for dates with known drift setups.

Simulates the T2 FSM to find exactly where setups fail to become trades.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.models import Bar, MarketSnapshot, SymbolIntradayState
from strategies.stock.iaric.signals import (
    compute_location_grade,
    compute_micropressure_proxy,
    compute_required_acceptance,
    cooldown_expired,
    lock_setup,
    reset_setup_state,
    resolve_confidence,
    update_acceptance,
)
from strategies.stock.iaric.risk import timing_gate_allows_entry

logging.basicConfig(level=logging.WARNING)
ET = ZoneInfo("America/New_York")
_MKT_OPEN = time(9, 30)
_MKT_CLOSE = time(16, 0)

_SPONSORSHIP_TO_SIGNAL = {
    "STRONG": "STRONG",
    "NEUTRAL": "NEUTRAL",
    "WEAK": "WEAK",
    "BREAKDOWN": "WEAK",
}


def trace_date(replay, trade_date, settings, cfg, verbose=True):
    """Trace full FSM for all tradable symbols on one date."""
    artifact = replay.iaric_selection_for_date(trade_date, settings)
    if artifact.regime.tier == "C":
        if verbose:
            print(f"  [{trade_date}] Regime C — skipped")
        return {"regime_c": 1}

    tradable_map = {item.symbol: item for item in artifact.tradable}
    if not tradable_map:
        return {"no_tradable": 1}

    stats = Counter()
    stats["dates_active"] = 1
    stats["tradable"] = len(tradable_map)

    for sym, item in tradable_map.items():
        bars = replay.get_5m_bar_objects_for_date(sym, trade_date)
        if not bars:
            stats["no_5m_data"] += 1
            continue
        rth_bars = [b for b in bars if _MKT_OPEN <= b.start_time.astimezone(ET).time() < _MKT_CLOSE]
        if not rth_bars:
            stats["no_rth_bars"] += 1
            continue
        stats["have_bars"] += 1

        # Wire flow proxy
        flow_signal = "UNAVAILABLE"
        flow_data = replay.get_flow_proxy_last_n(sym, trade_date, settings.flow_reversal_lookback)
        if flow_data is not None:
            if all(v < 0 for v in flow_data):
                flow_signal = "WEAK"
            elif all(v > 0 for v in flow_data):
                flow_signal = "STRONG"
            else:
                flow_signal = "NEUTRAL"

        sym_state = SymbolIntradayState(
            symbol=sym,
            fsm_state="IDLE",
            setup_time=None,
            setup_low=0.0,
            reclaim_level=0.0,
            stop_level=0.0,
            acceptance_count=0,
            required_acceptance_count=0,
            confidence="YELLOW",
            micropressure_signal="NEUTRAL",
            flowproxy_signal=flow_signal,
            micropressure_mode="LIVE",
            invalidated_at=None,
            tier="WARM",
            sponsorship_signal=_SPONSORSHIP_TO_SIGNAL.get(item.sponsorship_state, "NEUTRAL"),
        )
        market = MarketSnapshot(
            symbol=sym,
            last_price=item.avwap_ref,
            session_high=0.0,
            session_low=float("inf"),
            session_vwap=0.0,
            avwap_live=item.avwap_ref,
        )

        atr_5m_pct = item.intraday_atr_seed if item.intraday_atr_seed > 0 else 0.01
        avwap_lo = item.avwap_band_lower
        avwap_hi = item.avwap_band_upper
        mwis = artifact.market_wide_institutional_selling

        sym_hod_idx = 0
        cum_vol = 0.0
        cum_pv = 0.0
        expected_vol = item.expected_5m_volume if item.expected_5m_volume > 0 else 0.0
        median_vol = item.average_30m_volume if item.average_30m_volume > 0 else 0.0

        for bar_idx, bar in enumerate(rth_bars):
            now = bar.end_time

            # Update market
            if bar.high > (market.session_high or 0):
                market.session_high = bar.high
                sym_hod_idx = bar_idx
            market.last_price = bar.close
            if market.session_low == float("inf"):
                market.session_low = bar.low
            else:
                market.session_low = min(market.session_low, bar.low)
            cum_vol += bar.volume
            cum_pv += bar.close * bar.volume
            if cum_vol > 0:
                market.session_vwap = cum_pv / cum_vol
            market.bars_5m.append(bar)
            market.last_5m_bar = bar

            # Global invalidation checks
            if sym_state.fsm_state in ("SETUP_DETECTED", "ACCEPTING"):
                if sym_state.stop_level is not None and bar.low <= sym_state.stop_level:
                    prev = sym_state.fsm_state
                    sym_state.fsm_state = "INVALIDATED"
                    sym_state.invalidated_at = now
                    stats[f"invalidated_stop_{prev}"] += 1
                    if verbose:
                        print(f"    [{sym}] bar {bar_idx}: {prev} → INVALIDATED (stop breach @ {bar.low:.2f} <= {sym_state.stop_level:.2f})")
                elif sym_state.setup_time:
                    elapsed = (now - sym_state.setup_time).total_seconds() / 60
                    if elapsed > settings.setup_stale_minutes:
                        prev = sym_state.fsm_state
                        sym_state.fsm_state = "INVALIDATED"
                        sym_state.invalidated_at = now
                        stats[f"invalidated_stale_{prev}"] += 1
                        if verbose:
                            print(f"    [{sym}] bar {bar_idx}: {prev} → INVALIDATED (stale {elapsed:.0f}min > {settings.setup_stale_minutes}min)")

            # FSM states
            if sym_state.fsm_state == "IDLE":
                in_band = (avwap_lo <= bar.low <= avwap_hi) or (avwap_lo <= bar.close <= avwap_hi)
                if not in_band:
                    continue

                stats["bars_in_band"] += 1
                session_high = market.session_high
                drop_from_hod = (session_high - bar.close) / session_high if session_high > 0 else 0
                minutes_since_hod = (bar_idx - sym_hod_idx) * 5

                setup_type = None
                if drop_from_hod >= settings.panic_flush_drop_pct and minutes_since_hod <= settings.panic_flush_minutes:
                    setup_type = "PANIC_FLUSH"
                elif drop_from_hod >= settings.drift_exhaustion_drop_pct and minutes_since_hod >= settings.drift_exhaustion_minutes:
                    setup_type = "DRIFT_EXHAUSTION"

                if setup_type:
                    sym_state.setup_type = setup_type
                    lock_setup(sym_state, bar, atr_5m_pct, reason=setup_type)
                    sym_state.location_grade = compute_location_grade(item, market)
                    stats[f"setup_{setup_type}"] += 1
                    if verbose:
                        print(f"    [{sym}] bar {bar_idx} ({now.astimezone(ET).strftime('%H:%M')}): IDLE → SETUP_DETECTED ({setup_type})")
                        print(f"      setup_low={sym_state.setup_low:.2f} reclaim={sym_state.reclaim_level:.2f} stop={sym_state.stop_level:.2f}")
                        print(f"      drop={drop_from_hod:.4f} min_since_hod={minutes_since_hod}")

            elif sym_state.fsm_state == "SETUP_DETECTED":
                reclaim_touched = False
                if sym_state.reclaim_level is not None:
                    if bar.high >= sym_state.reclaim_level:
                        reclaim_touched = True
                    elif market.last_price is not None and market.last_price >= sym_state.reclaim_level:
                        reclaim_touched = True

                if reclaim_touched:
                    sym_state.fsm_state = "ACCEPTING"
                    required, adders = compute_required_acceptance(
                        item=item, sym=sym_state, now=now,
                        settings=settings,
                        market_wide_institutional_selling=mwis,
                    )
                    sym_state.required_acceptance_count = required
                    stats["reclaim_reached"] += 1
                    if verbose:
                        print(f"    [{sym}] bar {bar_idx} ({now.astimezone(ET).strftime('%H:%M')}): SETUP_DETECTED → ACCEPTING")
                        print(f"      required_acceptance={required} adders={adders}")

            elif sym_state.fsm_state == "ACCEPTING":
                update_acceptance(sym_state, bar)
                vol_for_proxy = expected_vol if expected_vol > 0 else bar.volume
                med_for_proxy = median_vol if median_vol > 0 else bar.volume
                if sym_state.reclaim_level:
                    mp = compute_micropressure_proxy(bar, vol_for_proxy, med_for_proxy, sym_state.reclaim_level)
                    sym_state.micropressure_signal = mp
                sym_state.confidence = resolve_confidence(sym_state)

                if (sym_state.acceptance_count >= sym_state.required_acceptance_count
                        and sym_state.confidence != "RED"):
                    sym_state.fsm_state = "READY_TO_ENTER"
                    stats["ready_to_enter"] += 1
                    if verbose:
                        print(f"    [{sym}] bar {bar_idx} ({now.astimezone(ET).strftime('%H:%M')}): ACCEPTING → READY_TO_ENTER")
                        print(f"      acceptance={sym_state.acceptance_count}/{sym_state.required_acceptance_count} conf={sym_state.confidence}")

            elif sym_state.fsm_state == "INVALIDATED":
                if cooldown_expired(sym_state, now, settings):
                    reset_setup_state(sym_state)
                    stats["cooldown_reset"] += 1

            # Entry check
            if sym_state.fsm_state == "READY_TO_ENTER":
                entry_blocked = False
                entry_block_reason = ""
                if not timing_gate_allows_entry(now, settings):
                    entry_blocked = True
                    entry_block_reason = "timing"
                elif item.sponsorship_state in ("WEAK", "BREAKDOWN"):
                    entry_blocked = True
                    entry_block_reason = "sponsorship"

                if entry_blocked:
                    stats[f"entry_blocked_{entry_block_reason}"] += 1
                    if verbose:
                        print(f"    [{sym}] bar {bar_idx} ({now.astimezone(ET).strftime('%H:%M')}): READY_TO_ENTER blocked: {entry_block_reason}")
                    continue

                stats["entries"] += 1
                if verbose:
                    print(f"    [{sym}] bar {bar_idx} ({now.astimezone(ET).strftime('%H:%M')}): *** ENTRY *** at {bar.close:.2f}")
                reset_setup_state(sym_state)

    return stats


def main():
    cfg = IARICBacktestConfig()
    settings = StrategySettings()
    if cfg.param_overrides:
        settings = replace(settings, **cfg.param_overrides)

    replay = ResearchReplayEngine(data_dir=cfg.data_dir, universe_config=cfg.universe)
    replay.load_all_data()

    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    trading_dates = replay.tradable_dates(start, end)

    # Phase 1: Full scan (non-verbose) to count all failure modes
    print("=" * 70)
    print("FULL SCAN: Tracing FSM for all trading dates")
    print("=" * 70)
    total = Counter()
    for td in trading_dates:
        s = trace_date(replay, td, settings, cfg, verbose=False)
        total.update(s)

    print(f"\n{'='*50}")
    print("AGGREGATE FSM FUNNEL")
    print(f"{'='*50}")
    for key in sorted(total.keys()):
        print(f"  {key}: {total[key]}")

    # Phase 2: Verbose trace for a few dates with known setups
    print(f"\n{'='*70}")
    print("VERBOSE TRACE: Dates with drift setups")
    print(f"{'='*70}")
    replay.clear_selection_cache()
    count = 0
    for td in trading_dates:
        s = trace_date(replay, td, settings, cfg, verbose=False)
        if s.get("setup_DRIFT_EXHAUSTION", 0) > 0:
            if count < 3:  # Show first 3 dates with setups
                print(f"\n--- {td} ---")
                replay.clear_selection_cache()
                trace_date(replay, td, settings, cfg, verbose=True)
                count += 1


if __name__ == "__main__":
    main()
