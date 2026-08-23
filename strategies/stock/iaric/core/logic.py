from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from math import ceil
from typing import Any, Sequence

from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitMarketExit,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.stock.iaric.execution import build_position_from_fill
from strategies.stock.iaric.exits import partial_remainder_stop_after_fill
from strategies.stock.iaric.models import Bar, MarketSnapshot, PBSymbolState, PositionState, VWAPLedger, WatchlistItem
from strategies.stock.iaric.signals import compute_micropressure_proxy

from .opportunity import (
    REVERSION_FAMILIES,
    DailyOpportunityContext,
    OpportunityEvent,
    detect_completed_bar_opportunities,
)
from .lanes import (
    consumption_token,
    event_is_consumed,
    family_event_caps,
    lane_id_for_route,
    management_override,
    REARMABLE_FAMILIES,
    rearm_cooldown_bars,
    score_from_components,
    score_profile_name,
)

from .state import (
    IARICBarInput,
    IARICCoreState,
    IARICEntryRequest,
    IARICEntryAcceptance,
    IARICFill,
    IARICFlattenRequest,
    IARICOrderUpdate,
    IARICPartialExitRequest,
    IARICRouteStep,
    IARICStopUpdateRequest,
)

# ---------------------------------------------------------------------------
# Canonical trigger vocabulary
#
# The live signal generator and the replay trigger evaluator historically used
# two disjoint name sets for the same seven daily conditions, and every gate in
# this module matched only the live spelling.  Any policy that filtered on
# trigger identity therefore admitted nothing in replay while binding fully in
# live.  All trigger identity comparisons must now go through
# ``normalize_trigger_types`` so both paths resolve to the canonical names.
# ---------------------------------------------------------------------------
CANONICAL_TRIGGERS = (
    "RSI2",
    "RSI5_CDD",
    "DEPTH",
    "BB_PCTB",
    "VOL_CLIMAX",
    "ROC5_DROP",
    "RS_STRONG",
    "GAP_FILL",
)

_TRIGGER_ALIASES = {
    # replay spelling -> canonical
    "DEEP_RSI": "RSI2",
    "MOD_RSI": "RSI5_CDD",
    "ATR_DEPTH": "DEPTH",
    "BB_EXTREME": "BB_PCTB",
    "VOL_CAPITULATION": "VOL_CLIMAX",
    "RS_DIP": "ROC5_DROP",
}

# Conditions that represent a genuine price dislocation (as opposed to a
# relative-strength or gap context flag).
DISLOCATION_TRIGGERS = frozenset(
    {"RSI2", "RSI5_CDD", "DEPTH", "BB_PCTB", "VOL_CLIMAX", "ROC5_DROP"}
)
OVERSOLD_TRIGGERS = frozenset({"RSI2", "RSI5_CDD", "BB_PCTB"})


def normalize_trigger_types(triggers: Any) -> set[str]:
    """Map any accepted trigger spelling onto the canonical vocabulary."""

    out: set[str] = set()
    for trigger in triggers or []:
        name = str(trigger).strip().upper()
        if not name:
            continue
        out.add(_TRIGGER_ALIASES.get(name, name))
    return out


def assert_trigger_vocabulary(triggers: Any) -> set[str]:
    """Normalize and reject unknown trigger names.

    Live/replay parity depends on both emitters staying inside the canonical
    vocabulary.  A new trigger added on one side only will now fail loudly
    instead of silently disabling every identity-based gate.
    """

    canonical = normalize_trigger_types(triggers)
    unknown = canonical - set(CANONICAL_TRIGGERS)
    if unknown:
        raise ValueError(
            f"unknown IARIC trigger name(s) {sorted(unknown)}; "
            f"expected a subset of {list(CANONICAL_TRIGGERS)}"
        )
    return canonical


_TERMINAL_STATUSES = {
    "cancelled",
    "expired",
    "rejected",
    "order_cancelled",
    "order_expired",
    "order_rejected",
}


def build_core_state(engine) -> IARICCoreState:
    return IARICCoreState(
        trade_date=engine._artifact.trade_date,
        saved_at=datetime.now(timezone.utc),
        symbols=deepcopy(list(engine._symbols.values())),
        last_decision_code=engine._last_decision_code,
        meta={
            "active_symbols": sorted(engine._active_symbols),
            "order_index": deepcopy(engine._order_index),
            "pending_entry_risk": deepcopy(engine._portfolio.pending_entry_risk),
            "account_equity": engine._portfolio.account_equity,
            "base_risk_fraction": engine._portfolio.base_risk_fraction,
            "regime_allows_no_new_entries": engine._portfolio.regime_allows_no_new_entries,
            "expected_stop_cancels": sorted(engine._expected_stop_cancels),
            "aperture_family_counts": deepcopy(
                getattr(engine, "_aperture_family_counts", {})
            ),
            "daily_entry_symbols": list(getattr(engine, "_daily_entry_symbols", [])),
            "rescue_entry_count": int(getattr(engine, "_rescue_entry_count", 0)),
            "lane_entry_counts": deepcopy(getattr(engine, "_lane_entry_counts", {})),
            "last_decision_details": deepcopy(engine._last_decision_details),
            "last_bar_ts": engine._last_bar_ts,
        },
    )


def apply_core_state(engine, state: IARICCoreState) -> None:
    restored = {symbol_state.symbol: deepcopy(symbol_state) for symbol_state in state.symbols}
    for symbol, symbol_state in restored.items():
        engine._symbols[symbol] = symbol_state
        engine._markets.setdefault(symbol, MarketSnapshot(symbol=symbol))
        engine._session_vwap.setdefault(symbol, VWAPLedger())

    meta = state.meta if isinstance(state.meta, dict) else {}
    engine._active_symbols = set(meta.get("active_symbols", engine._active_symbols))
    engine._order_index = {
        str(order_id): _coerce_order_index_entry(value)
        for order_id, value in dict(meta.get("order_index", {})).items()
    }
    engine._portfolio.pending_entry_risk = {
        str(symbol): float(risk)
        for symbol, risk in dict(meta.get("pending_entry_risk", {})).items()
    }
    if "account_equity" in meta:
        engine._portfolio.account_equity = float(meta["account_equity"])
    if "base_risk_fraction" in meta:
        engine._portfolio.base_risk_fraction = float(meta["base_risk_fraction"])
    if "regime_allows_no_new_entries" in meta:
        engine._portfolio.regime_allows_no_new_entries = bool(meta["regime_allows_no_new_entries"])
    engine._expected_stop_cancels = {
        str(order_id) for order_id in meta.get("expected_stop_cancels", [])
    }
    engine._aperture_family_counts = {
        str(family): int(count)
        for family, count in dict(meta.get("aperture_family_counts", {})).items()
    }
    engine._daily_entry_symbols = [
        str(symbol) for symbol in meta.get("daily_entry_symbols", [])
    ]
    engine._rescue_entry_count = int(meta.get("rescue_entry_count", 0))
    engine._lane_entry_counts = {
        str(lane): int(count)
        for lane, count in dict(meta.get("lane_entry_counts", {})).items()
    }
    engine._portfolio.open_positions = {
        symbol_state.symbol: deepcopy(symbol_state.position)
        for symbol_state in engine._symbols.values()
        if symbol_state.position is not None and symbol_state.in_position
    }
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = dict(meta.get("last_decision_details", {}))
    engine._last_bar_ts = _coerce_datetime(meta.get("last_bar_ts"))


def active_symbols(state: IARICCoreState) -> list[str]:
    meta = state.meta if isinstance(state.meta, dict) else {}
    return list(meta.get("active_symbols", []))


def route_prefix(route_family: str) -> str:
    route = str(route_family or "").upper()
    if route.startswith("APERTURE_"):
        return "pb_aperture"
    return {
        "OPEN_SCORED_ENTRY": "pb_open_scored",
        "OPEN_SCORED_RESCUE_ENTRY": "pb_open_scored",
        "OPEN_SCORED_RETEST": "pb_open_scored",
        "OPEN_SCORED_RETRACE_LIMIT": "pb_open_scored",
        "DELAYED_CONFIRM": "pb_delayed_confirm",
        "OPENING_RECLAIM": "pb_opening_reclaim",
    }.get(route, "pb_opening_reclaim")


def is_open_scored_route(route_family: str) -> bool:
    return str(route_family or "").upper() in {
        "OPEN_SCORED_ENTRY",
        "OPEN_SCORED_RESCUE_ENTRY",
        "OPEN_SCORED_RETEST",
        "OPEN_SCORED_RETRACE_LIMIT",
    }


def is_aperture_route(route_family: str) -> bool:
    return str(route_family or "").upper().startswith("APERTURE_")


def is_retrace_limit_route(route_family: str) -> bool:
    return str(route_family or "").upper().endswith("_RETRACE_LIMIT")


def open_scored_transition(settings: Any) -> str:
    transition = str(
        getattr(settings, "pb_open_scored_transition", "next_bar") or "next_bar"
    ).lower()
    if transition not in {
        "next_bar",
        "confirmed_retest",
        "resting_retrace",
        "reclaim_or_limit",
    }:
        raise ValueError(
            "pb_open_scored_transition must be 'next_bar', 'confirmed_retest', "
            "'resting_retrace', or 'reclaim_or_limit'"
        )
    return transition


def estimate_session_atr(item: Any, bars: Sequence[Any], daily_atr: float = 0.0) -> float:
    """Return the same causal 5-minute ATR estimate in live and replay paths.

    Before three completed bars are available, use the nightly dimensionless
    seed on the current session's price basis. Once enough bars exist, use
    only completed intraday true ranges.
    """
    ref_price = float(bars[0].open) if bars else max(float(getattr(item, "avwap_ref", 0.0)), 1.0)
    floor = ref_price * 0.0025
    if len(bars) >= 3:
        true_ranges = [
            max(
                float(bars[idx].high) - float(bars[idx].low),
                abs(float(bars[idx].high) - float(bars[idx - 1].close)),
                abs(float(bars[idx].low) - float(bars[idx - 1].close)),
            )
            for idx in range(1, len(bars))
        ]
        if true_ranges:
            return max(sum(true_ranges) / len(true_ranges), floor)

    intraday_seed = float(getattr(item, "intraday_atr_seed", 0.0) or 0.0)
    if intraday_seed > 0:
        return max(intraday_seed * ref_price, floor)
    fallback_daily_atr = max(float(daily_atr or 0.0), float(getattr(item, "daily_atr_estimate", 0.0) or 0.0))
    if fallback_daily_atr > 0:
        return max(fallback_daily_atr * 0.25, floor)
    return ref_price * 0.01


def route_enabled(settings: Any, route_family: str) -> bool:
    route_key = str(route_family or "").upper()
    v2 = bool(getattr(settings, "pb_v2_enabled", False))
    if is_open_scored_route(route_key):
        attr = "pb_v2_open_scored_enabled" if v2 else "pb_open_scored_enabled"
        return bool(getattr(settings, attr, True))
    if is_aperture_route(route_key):
        return bool(getattr(settings, "pb_aperture_enabled", False))
    if route_key == "DELAYED_CONFIRM":
        return bool(getattr(settings, "pb_delayed_confirm_enabled", True))
    if route_key == "OPENING_RECLAIM":
        return bool(getattr(settings, "pb_opening_reclaim_enabled", True))
    if route_key == "VWAP_BOUNCE":
        return v2 and bool(getattr(settings, "pb_v2_vwap_bounce_enabled", True))
    if route_key == "AFTERNOON_RETEST":
        return v2 and bool(getattr(settings, "pb_v2_afternoon_retest_enabled", True))
    return True


def route_setting(settings: Any, route_family: str, suffix: str, fallback_suffix: str | None = None) -> Any:
    override = management_override(settings, route_family, suffix)
    if override is not None:
        return override
    prefix = route_prefix(route_family)
    attr = f"{prefix}_{suffix}"
    if hasattr(settings, attr):
        return getattr(settings, attr)
    if fallback_suffix is not None and hasattr(settings, fallback_suffix):
        return getattr(settings, fallback_suffix)
    raise AttributeError(f"Missing route setting for {route_family}:{suffix}")


def route_carry_profile(route_family: str) -> str:
    return route_prefix(route_family).replace("pb_", "").upper()


def route_min_daily_signal_score(settings: Any, route_family: str) -> float:
    route_key = str(route_family or "").upper()
    if is_aperture_route(route_key):
        return float(getattr(settings, "pb_aperture_event_score_min", 70.0))
    if is_open_scored_route(route_key):
        if route_key == "OPEN_SCORED_RESCUE_ENTRY" and bool(
            getattr(settings, "pb_rescue_event_lane_enabled", False)
        ):
            return float(getattr(settings, "pb_rescue_event_daily_score_min", 60.0))
        v2 = bool(getattr(settings, "pb_v2_enabled", False))
        attr = "pb_v2_open_scored_min_score" if v2 else "pb_open_scored_min_score"
        return float(getattr(settings, attr, 0.0))
    if route_key == "DELAYED_CONFIRM":
        return float(
            getattr(
                settings,
                "pb_delayed_confirm_min_daily_signal_score",
                getattr(settings, "pb_daily_signal_min_score", 0.0),
            )
        )
    if route_key == "OPENING_RECLAIM":
        return float(
            getattr(
                settings,
                "pb_opening_reclaim_min_daily_signal_score",
                getattr(settings, "pb_daily_signal_min_score", 0.0),
            )
        )
    if route_key == "AFTERNOON_RETEST":
        return float(
            getattr(
                settings,
                "pb_v2_afternoon_retest_min_score",
                getattr(settings, "pb_daily_signal_min_score", 0.0),
            )
        )
    return float(getattr(settings, "pb_daily_signal_min_score", 0.0))


def open_scored_eligible(settings: Any, payload: dict[str, Any] | None) -> bool:
    if not route_enabled(settings, "OPEN_SCORED_ENTRY"):
        return False
    source = payload or {}
    if "APERTURE" in normalize_trigger_types(source.get("trigger_types")):
        return False
    score = float(source.get("daily_signal_score") or 0.0)
    rank_pct = float(source.get("daily_signal_rank_pct") or 100.0)
    v2 = bool(getattr(settings, "pb_v2_enabled", False))
    min_score = route_min_daily_signal_score(settings, "OPEN_SCORED_ENTRY")
    max_score = float(getattr(settings, "pb_v2_open_scored_max_score", 100.0))
    max_rank_attr = "pb_v2_open_scored_rank_pct_max" if v2 else "pb_open_scored_rank_pct_max"
    max_rank_pct = float(getattr(settings, max_rank_attr, 100.0))
    rescue_candidate = bool(source.get("rescue_flow_candidate", False))
    rescue_allowed = not v2 or bool(getattr(settings, "pb_v2_open_scored_allow_rescue", False))
    trigger_policy = str(
        getattr(settings, "pb_v2_open_scored_trigger_policy", "any") or "any"
    ).lower()
    if rescue_candidate and bool(getattr(settings, "pb_rescue_event_lane_enabled", False)):
        # A rescue is a new causal lane, never permission to use the historical
        # same-open adapter or a non-isolated OPEN_SCORED transition.
        if (
            open_scored_transition(settings) != "next_bar"
            or str(getattr(settings, "pb_open_scored_fill_timing", "next_5m_open")).lower()
            != "next_5m_open"
        ):
            return False
        rescue_allowed = True
        min_score = max(
            min_score,
            float(getattr(settings, "pb_rescue_event_daily_score_min", 60.0)),
        )
        trigger_policy = str(
            getattr(settings, "pb_rescue_event_trigger_policy", "oversold_or_multi")
            or "oversold_or_multi"
        ).lower()
    if trigger_policy not in {"any", "dislocation", "oversold", "multi_dislocation", "oversold_or_multi"}:
        raise ValueError(
            "pb_v2_open_scored_trigger_policy must be 'any', 'dislocation', "
            "'oversold', 'multi_dislocation', or 'oversold_or_multi'"
        )
    # Normalize before any identity comparison: live and replay spell the same
    # seven daily conditions differently, and matching raw strings silently
    # disabled every trigger policy on the replay path.
    triggers = assert_trigger_vocabulary(source.get("trigger_types"))
    dislocation = triggers.intersection(DISLOCATION_TRIGGERS)
    oversold = triggers.intersection(OVERSOLD_TRIGGERS)
    trigger_ok = (
        trigger_policy == "any"
        or (trigger_policy == "dislocation" and bool(dislocation))
        or (trigger_policy == "oversold" and bool(oversold))
        or (trigger_policy == "multi_dislocation" and len(dislocation) >= 2)
        or (trigger_policy == "oversold_or_multi" and (bool(oversold) or len(dislocation) >= 2))
    )
    return (
        score >= min_score
        and score <= max_score
        and rank_pct <= max_rank_pct
        and (not rescue_candidate or rescue_allowed)
        and trigger_ok
    )


def open_scored_entry_score_eligible(settings: Any, score: float) -> bool:
    """Return whether a computed OPEN_SCORED route score is admissible.

    The lower threshold remains the shared route entry threshold.  The upper
    bound is an opt-in research control for a non-monotonic score tail and is
    deliberately disabled at 100 by default.
    """

    return float(score) <= float(
        getattr(settings, "pb_v2_open_scored_max_entry_score", 100.0)
    )


def open_scored_bar_confirmed(
    settings: Any,
    bar: Bar,
    market: MarketSnapshot,
) -> bool:
    """Apply the shared completed-bar confirmation for direct OPEN_SCORED.

    The decision uses only the completed signal bar.  A successful decision is
    still submitted/fillable no earlier than the following bar.
    """

    policy = str(
        getattr(settings, "pb_v2_open_scored_confirmation_policy", "any") or "any"
    ).lower()
    if policy == "any":
        return True
    if policy == "bullish_close":
        return bool(bar.close > bar.open)
    if policy == "bullish_vwap":
        return bool(bar.close > bar.open and market.session_vwap is not None and bar.close >= market.session_vwap)
    if policy == "vwap_reclaim":
        # Nightly oversold/dislocation is context, not an executable trigger.
        # Require a completed prior bar below the VWAP that was available at
        # that prior close, followed by a completed bullish close back above
        # the now-current VWAP.  This is the mean-reversion analogue of an
        # armed completed-bar breakout and is identical in live and replay.
        bars = list(market.bars_5m)
        if len(bars) < 2 or market.session_vwap is None:
            return False
        prior_bars = bars[:-1]
        prior_volume = sum(max(float(sample.volume), 0.0) for sample in prior_bars)
        if prior_volume <= 0:
            return False
        prior_vwap = sum(
            float(sample.typical_price) * max(float(sample.volume), 0.0)
            for sample in prior_bars
        ) / prior_volume
        prior = prior_bars[-1]
        return bool(
            prior.close < prior_vwap
            and bar.close > bar.open
            and bar.close >= market.session_vwap
        )
    if policy == "band_reclaim":
        return band_reclaim_confirmed(settings, bar, market, state=None)
    raise ValueError(
        "pb_v2_open_scored_confirmation_policy must be 'any', "
        "'bullish_close', 'bullish_vwap', 'vwap_reclaim', or 'band_reclaim'"
    )


def dislocation_band(settings: Any, state: Any, market: MarketSnapshot | None = None) -> float:
    """Return the daily-anchored price level that defines "dislocated".

    The thesis is a *daily* pullback, so the band is anchored on prior-session
    information rather than on the current session's own opening range.  Using
    the session's own range would make the band a function of the move we are
    trying to detect.
    """

    prev_close = float(getattr(state, "prev_close", 0.0) or 0.0)
    prev_low = float(getattr(state, "prev_low", 0.0) or 0.0)
    daily_atr = float(getattr(state, "daily_atr", 0.0) or 0.0)
    atr_frac = float(getattr(settings, "pb_v2_dislocation_band_atr", 0.35) or 0.0)

    levels: list[float] = []
    if prev_close > 0 and daily_atr > 0:
        levels.append(prev_close - atr_frac * daily_atr)
    if prev_low > 0 and bool(getattr(settings, "pb_v2_dislocation_use_prev_low", True)):
        levels.append(prev_low)
    if not levels:
        return 0.0
    # The shallower of the two anchors: a candidate only has to breach one
    # recognised dislocation reference, not both.
    return float(max(levels))


def band_reclaim_confirmed(
    settings: Any,
    bar: Bar,
    market: MarketSnapshot,
    *,
    state: Any = None,
) -> bool:
    """Completed-bar dislocation -> reclaim event.

    This is the mean-reversion analogue of a completed-bar breakout: the session
    must first trade *below* a daily-anchored dislocation band, and a later
    completed bar must close back *above* it.  Both halves use only information
    available at that bar's close, and the caller may submit no earlier than the
    following bar.

    Unlike a score threshold, this is a discrete event: it either happened or it
    did not.  The score is then only a ranker over events.
    """

    if state is None:
        state = getattr(market, "_iaric_state", None)
    if state is None:
        return False
    band = dislocation_band(settings, state, market)
    if band <= 0:
        return False

    bars = list(market.bars_5m) or [bar]
    # The dislocation must be established on completed bars strictly before the
    # reclaim bar, otherwise a single wide bar would satisfy both halves.
    prior = bars[:-1] if bars and bars[-1] is bar else bars
    if not prior:
        return False
    dislocated = any(float(sample.low) <= band for sample in prior)
    if not dislocated:
        return False
    if float(bar.close) <= band:
        return False
    # Require the reclaim bar itself to be constructive; a gap-through with a
    # weak close is not a reclaim.
    if float(bar.close) <= float(bar.open):
        return False

    rvol_min = float(getattr(settings, "pb_v2_open_scored_rvol_min", 0.0) or 0.0)
    if rvol_min > 0:
        item = getattr(state, "item", None)
        rvol = compute_rvol(bar, item, len(bars) - 1)
        if rvol < rvol_min:
            return False
    return True


def open_scored_slot_cap(settings: Any, available_slots: int, *, has_intraday_candidates: bool) -> int:
    """Return the executable OPEN_SCORED capacity shared by live and replay.

    OPEN_SCORED is an early fallback route.  It may use only its named route cap
    and must leave the configured reserve available for later reclaim/confirm
    decisions when such candidates exist.
    """

    slots = max(int(available_slots), 0)
    reserve = 0
    if has_intraday_candidates:
        reserve = min(max(int(getattr(settings, "pb_intraday_priority_reserve_slots", 0)), 0), slots)
    route_cap = (
        int(getattr(settings, "pb_v2_open_scored_max_slots", 0))
        if bool(getattr(settings, "pb_v2_enabled", False))
        else int(ceil(slots * float(getattr(settings, "pb_open_scored_max_share", 0.0))))
    )
    return max(min(route_cap, slots - reserve), 0)


def route_priority_value(settings: Any, route_family: str, score: float) -> float:
    """Return the shared ascending sort value for live/replay entry priority."""

    numeric = float(score)
    if not is_open_scored_route(route_family):
        return -numeric
    priority = str(
        getattr(settings, "pb_open_scored_priority", "high_score") or "high_score"
    ).lower()
    if priority not in {"high_score", "low_score"}:
        raise ValueError(
            "pb_open_scored_priority must be 'high_score' or 'low_score'"
        )
    return numeric if priority == "low_score" else -numeric


def opening_gap_eligible(settings: Any, previous_close: float, opening_price: float) -> bool:
    """Apply the trade-day gap gate only once the opening price is observable."""

    previous = float(previous_close)
    opening = float(opening_price)
    if previous <= 0 or opening <= 0:
        return True
    gap_pct = (opening - previous) / previous * 100.0
    if bool(getattr(settings, "pb_v2_enabled", False)):
        return (
            float(getattr(settings, "pb_v2_gap_min_pct", -99.0))
            <= gap_pct
            <= float(getattr(settings, "pb_v2_gap_max_pct", 99.0))
        )
    return (
        float(getattr(settings, "pb_gap_min_pct", -99.0))
        <= gap_pct
        <= float(getattr(settings, "pb_gap_max_pct", 99.0))
    )


def entry_threshold(settings: Any, state: Any) -> float:
    if bool(getattr(state, "rescue_flow_candidate", False)):
        if (
            bool(getattr(settings, "pb_rescue_event_lane_enabled", False))
            and str(getattr(state, "route_family", "")).upper()
            == "OPEN_SCORED_RESCUE_ENTRY"
        ):
            return float(
                max(
                    getattr(settings, "pb_rescue_event_entry_score_min", 65.0),
                    getattr(settings, "pb_entry_score_min", 0.0),
                )
            )
        return float(max(getattr(settings, "pb_rescue_min_score", 0.0), getattr(settings, "pb_entry_score_min", 0.0)))
    if getattr(state, "intraday_setup_type", "") == "DELAYED_CONFIRM":
        return float(min(getattr(settings, "pb_entry_score_min", 0.0), getattr(settings, "pb_delayed_confirm_score_min", 0.0)))
    return float(getattr(settings, "pb_entry_score_min", 0.0))


def compute_volume_ratio(bar: Bar, item: WatchlistItem | None) -> float:
    if item is None:
        return 1.0
    expected = float(item.expected_5m_volume)
    if expected <= 0 and item.average_30m_volume > 0:
        expected = float(item.average_30m_volume) / 6.0
    return float(bar.volume / max(expected, 1.0))


def compute_rvol(bar: Bar, item: WatchlistItem | None, bar_idx: int) -> float:
    """Time-of-day-matched relative volume.

    ``compute_volume_ratio`` divides by a single flat session average.  Intraday
    volume is strongly U-shaped, so against a flat baseline the 09:30-09:35 bar
    is always an extreme multiple and every downstream transform saturates --
    the ratio carries no information at the one bar the route actually used.

    This compares each bar against the prior session's volume at *the same bar
    index*, so a value near 1.0 means "normal for this time of day" and the
    measure stays discriminating across the whole session.
    """

    if item is None:
        return 1.0
    profile = tuple(getattr(item, "expected_5m_profile", ()) or ())
    expected = 0.0
    if profile and 0 <= bar_idx < len(profile):
        expected = float(profile[bar_idx])
    if expected <= 0:
        # No time-of-day baseline available: fall back to the flat estimate but
        # keep the same units so thresholds remain comparable.
        return compute_volume_ratio(bar, item)
    return float(bar.volume) / max(expected, 1.0)


def rvol_score(rvol: float) -> float:
    """Map RVOL onto [0, 1] without saturating at ordinary opening volume.

    1.0x -> 0.35, 1.5x -> ~0.55, 3.0x -> ~0.79, 6.0x -> ~0.95.  A log transform
    keeps the whole realistic range discriminating instead of clipping every
    opening bar to the same value.
    """

    from math import log

    value = max(float(rvol), 0.01)
    return min(max(0.35 + 0.30 * log(value), 0.0), 1.0)


def micropressure_label(
    bars: Sequence[Bar],
    bar_idx: int,
    reclaim_level: float,
    item: WatchlistItem,
    *,
    lookback_bars: int = 3,
) -> str:
    if bar_idx < 0 or bar_idx >= len(bars):
        return "NEUTRAL"
    span = max(int(lookback_bars), 1)
    recent = list(bars[max(0, bar_idx - (span - 1)) : bar_idx + 1])
    bullish = 0
    for sample in recent:
        label = compute_micropressure_proxy(
            sample,
            expected_volume=max(item.expected_5m_volume, 1.0),
            median20_volume=max(item.average_30m_volume / 6.0, 1.0),
            reclaim_level=reclaim_level,
        )
        if label == "ACCUMULATE":
            bullish += 1
    if bullish >= max(1, len(recent) - 1):
        return "ACCUMULATE"
    if bullish == 0 and recent and recent[-1].close < recent[-1].open:
        return "DISTRIBUTE"
    return "NEUTRAL"


def thirty_min_context_bonus(market: MarketSnapshot, *, weight: float) -> float:
    bar = market.last_30m_bar
    if bar is None:
        return 0.0
    close_pct = _close_in_range_pct(bar.high, bar.low, bar.close)
    bonus = (close_pct - 0.5) * weight
    if bar.close > bar.open:
        bonus += weight * 0.35
    elif bar.close < bar.open:
        bonus -= weight * 0.20
    return float(min(max(bonus, -weight), weight))


def compute_initial_stop(settings: Any, setup_low: float, daily_atr: float, session_atr: float) -> float:
    daily_cap = float(getattr(settings, "pb_stop_daily_atr_cap", 0.0)) * max(float(daily_atr), 0.0)
    session_buffer = float(getattr(settings, "pb_stop_session_atr_mult", 0.0)) * float(session_atr)
    buffer = min(session_buffer, daily_cap) if daily_cap > 0 else session_buffer
    return max(float(setup_low) - max(buffer, 0.01), 0.01)


def event_stop_anchor(
    settings: Any,
    state: Any,
    bar: Bar,
    session_low: float,
    daily_atr: float,
) -> float:
    """Stop anchor for a dislocation->reclaim entry.

    The session low is the wrong anchor for an event entry.  After a dislocation
    the session low is far below the reclaim bar, so risk-per-share is large, the
    reversion target is only a small fraction of R, and winners cap out near
    +0.3R however well the entry is timed.  The structural invalidation of a
    reclaim is price falling back through the bar that produced it, so that is
    what the stop should reference.

    A floor of ``pb_v2_event_stop_min_atr`` * daily ATR keeps the stop off the
    noise band; the session low remains available as an opt-out.
    """

    anchor_mode = str(
        getattr(settings, "pb_v2_event_stop_anchor", "session_low") or "session_low"
    ).lower()
    if anchor_mode not in {"session_low", "reclaim_bar"}:
        raise ValueError(
            "pb_v2_event_stop_anchor must be 'session_low' or 'reclaim_bar'"
        )
    if anchor_mode == "session_low":
        return float(session_low)

    band = dislocation_band(settings, state)
    candidates = [float(bar.low)]
    if band > 0:
        candidates.append(band)
    anchor = min(candidates)

    min_atr = float(getattr(settings, "pb_v2_event_stop_min_atr", 0.25) or 0.0)
    if min_atr > 0 and daily_atr > 0:
        # Never tighter than a minimum fraction of daily ATR below the entry.
        anchor = min(anchor, float(bar.close) - min_atr * float(daily_atr))
    # Never wider than the session low would have been.
    return float(max(anchor, min(float(session_low), float(bar.low))))


def _close_in_range_pct(high: float, low: float, close: float) -> float:
    high_f = float(high)
    low_f = float(low)
    if high_f <= low_f:
        return 1.0
    return float(min(max((float(close) - low_f) / (high_f - low_f), 0.0), 1.0))


def compute_route_entry_score_bundle(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    *,
    bars: Sequence[Bar] | None = None,
    volume_ratio: float | None = None,
    micropressure: str | None = None,
    context_bonus: float | None = None,
) -> dict[str, float]:
    def _clip01(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)

    def _peak_score(value: float, *, target: float, width: float) -> float:
        width = max(float(width), 1e-6)
        return _clip01(1.0 - abs(float(value) - float(target)) / width)

    route_family_name = getattr(state, "route_family", "") or (
        "DELAYED_CONFIRM" if getattr(state, "intraday_setup_type", "") == "DELAYED_CONFIRM" else "OPENING_RECLAIM"
    )
    score_family = str(getattr(settings, "pb_entry_score_family", "meanrev_sweetspot_v1") or "meanrev_sweetspot_v1").lower()
    daily_signal = min(max(float(getattr(state, "daily_signal_score", 0.0)) / 100.0, 0.0), 1.0)
    reclaim_score = 0.0
    if float(getattr(state, "stop_level", 0.0)) > 0 and bar.close > float(getattr(state, "reclaim_level", 0.0)):
        reclaim_score = min(
            max(
                (bar.close - float(getattr(state, "reclaim_level", 0.0)))
                / max(bar.close - float(getattr(state, "stop_level", 0.0)), 0.01),
                0.0,
            ),
            1.5,
        ) / 1.5
    if volume_ratio is None:
        volume_ratio = compute_volume_ratio(bar, item)
    volume_score = min(max(float(volume_ratio) / max(float(getattr(settings, "pb_ready_min_volume_ratio", 0.25)), 0.25), 0.0), 1.25) / 1.25
    vwap = market.session_vwap or bar.close
    vwap_score = 0.0
    daily_atr = float(getattr(state, "daily_atr", 0.0))
    if daily_atr > 0:
        vwap_score = min(max((bar.close - vwap) / max(daily_atr * 0.75, 0.01), 0.0), 1.0)
    cpr_score = min(max(bar.cpr, 0.0), 1.0)
    if micropressure is None:
        series = list(bars) if bars is not None else [bar]
        series_idx = bar_idx if bars is not None else len(series) - 1
        micropressure = micropressure_label(series, series_idx, float(getattr(state, "reclaim_level", 0.0)), item)
    reclaim_bars = max(bar_idx - int(getattr(state, "flush_bar_idx", 0)) + 1, 1)
    speed_score = min(max(1.0 - (reclaim_bars - 1) / 8.0, 0.0), 1.0)
    if context_bonus is None:
        context_bonus = thirty_min_context_bonus(market, weight=4.0)
    route_flag = 0.0 if route_family_name == "OPENING_RECLAIM" else 1.0

    micropressure_policy = str(
        getattr(settings, "pb_entry_micropressure_policy", "score_penalty")
        or "score_penalty"
    ).lower()
    if micropressure_policy not in {"score_penalty", "block_distribute"}:
        raise ValueError(
            "pb_entry_micropressure_policy must be 'score_penalty' or 'block_distribute'"
        )
    min_room_atr = max(
        float(getattr(settings, "pb_entry_min_reversion_room_atr", 0.0) or 0.0),
        0.0,
    )
    prev_close_for_room = float(getattr(state, "prev_close", 0.0) or 0.0)
    room_atr = (
        (prev_close_for_room - float(bar.close)) / daily_atr
        if prev_close_for_room > 0 and daily_atr > 0
        else 0.0
    )
    if (
        micropressure_policy == "block_distribute"
        and str(micropressure or "").upper() == "DISTRIBUTE"
    ) or (min_room_atr > 0 and room_atr < min_room_atr):
        return {
            "route_family": route_flag,
            "quality_adjustment": -100.0,
            "score": 0.0,
        }

    if score_family == "reversion_event_v1":
        # Ranker over confirmed dislocation->reclaim events, not a trigger.
        #
        # The legacy families scored an OPEN_SCORED candidate by how strong its
        # first five-minute bar was: `reclaim`, `vwap_hold` and `cpr` are three
        # collinear measures of the same extension, and all three were measured
        # anti-correlated with remaining favourable excursion.  `volume` and
        # `speed` were arithmetically constant on that route.  This family keeps
        # only inputs whose sign is defensible for a reversion entry.
        prev_close = float(getattr(state, "prev_close", 0.0) or 0.0)
        atr = max(float(getattr(state, "daily_atr", 0.0) or 0.0), 1e-9)

        # 1. Remaining room back to the daily reference.  Positive means price is
        #    still below where it is reverting toward -- the opposite sign to the
        #    legacy `reclaim`/`vwap_hold` components.
        residual = 0.0
        if prev_close > 0:
            residual = _clip01((prev_close - float(bar.close)) / (atr * 1.25))

        # 2. Genuine time-of-day-matched relative volume on the reclaim bar.
        series_for_rvol = list(bars) if bars is not None else list(market.bars_5m)
        rvol_idx = bar_idx if bars is not None else max(len(series_for_rvol) - 1, 0)
        rvol = compute_rvol(bar, item, rvol_idx)
        rvol_component = rvol_score(rvol)

        # 3. Reclaim speed.  Meaningful only because the event route records the
        #    real dislocation bar index instead of a hardcoded zero.
        span = max(bar_idx - int(getattr(state, "flush_bar_idx", 0)), 0)
        speed_component = _clip01(1.0 - span / 8.0)

        # 4. Overextension penalty: the further the reclaim bar has already run
        #    above the band, the less of the move is left to capture.
        band = dislocation_band(settings, state, market)
        overextension = 0.0
        if band > 0:
            overextension = _clip01((float(bar.close) - band) / (atr * 0.75))

        weights = {
            "daily_signal": 46.0,
            "residual_dislocation": 22.0,
            "rvol": 16.0,
            "speed": 8.0,
        }
        overextension_penalty = -overextension * 14.0
        rescue_penalty = -8.0 if bool(getattr(state, "rescue_flow_candidate", False)) else 0.0
        total = (
            daily_signal * weights["daily_signal"]
            + residual * weights["residual_dislocation"]
            + rvol_component * weights["rvol"]
            + speed_component * weights["speed"]
            + overextension_penalty
            + rescue_penalty
        )
        return {
            "route_family": route_flag,
            "daily_signal": float(daily_signal * weights["daily_signal"]),
            "residual_dislocation": float(residual * weights["residual_dislocation"]),
            "rvol": float(rvol_component * weights["rvol"]),
            "speed": float(speed_component * weights["speed"]),
            "overextension_penalty": float(overextension_penalty),
            "quality_adjustment": float(overextension_penalty + rescue_penalty),
            "rvol_raw": float(rvol),
            "score": float(max(total, 0.0)),
        }

    def _bundle(
        *,
        daily_weight: float,
        reclaim_weight: float,
        volume_weight: float,
        vwap_weight: float,
        cpr_weight: float,
        speed_weight: float,
        context_low: float,
        context_high: float,
        distribute_penalty: float,
        neutral_penalty: float,
        weak_vwap_penalty_value: float,
        rescue_penalty_value: float,
        reclaim_input: float = reclaim_score,
        vwap_input: float = vwap_score,
        cpr_input: float = cpr_score,
        extension_penalty: float = 0.0,
    ) -> dict[str, float]:
        context_adjust = min(max(float(context_bonus), context_low), context_high)
        micro_penalty = distribute_penalty if micropressure == "DISTRIBUTE" else neutral_penalty if micropressure == "NEUTRAL" else 0.0
        weak_vwap_penalty = weak_vwap_penalty_value if bar.close < vwap else 0.0
        rescue_penalty = rescue_penalty_value if bool(getattr(state, "rescue_flow_candidate", False)) else 0.0
        total = (
            daily_signal * daily_weight
            + reclaim_input * reclaim_weight
            + volume_score * volume_weight
            + vwap_input * vwap_weight
            + cpr_input * cpr_weight
            + speed_score * speed_weight
            + context_adjust
            + micro_penalty
            + weak_vwap_penalty
            + rescue_penalty
            + extension_penalty
        )
        quality_adjustment = (
            context_adjust
            + micro_penalty
            + weak_vwap_penalty
            + rescue_penalty
            + extension_penalty
        )
        return {
            "route_family": route_flag,
            "daily_signal": float(daily_signal * daily_weight),
            "reclaim": float(reclaim_input * reclaim_weight),
            "volume": float(volume_score * volume_weight),
            "vwap_hold": float(vwap_input * vwap_weight),
            "cpr": float(cpr_input * cpr_weight),
            "speed": float(speed_score * speed_weight),
            "quality_adjustment": float(quality_adjustment),
            "score": float(max(total, 0.0)),
        }

    if score_family == "route_momentum_v1":
        return _bundle(
            daily_weight=45.0,
            reclaim_weight=18.0,
            volume_weight=12.0,
            vwap_weight=10.0,
            cpr_weight=10.0,
            speed_weight=8.0,
            context_low=-6.0,
            context_high=3.0,
            distribute_penalty=-12.0,
            neutral_penalty=-4.0,
            weak_vwap_penalty_value=-8.0,
            rescue_penalty_value=-8.0,
        )
    if score_family == "route_quality_v1":
        return _bundle(
            daily_weight=40.0,
            reclaim_weight=10.0,
            volume_weight=16.0,
            vwap_weight=10.0,
            cpr_weight=10.0,
            speed_weight=8.0,
            context_low=-4.0,
            context_high=2.0,
            distribute_penalty=-14.0,
            neutral_penalty=-6.0,
            weak_vwap_penalty_value=-12.0,
            rescue_penalty_value=-10.0,
        )
    if score_family == "route_early_reversal_v1":
        return _bundle(
            daily_weight=36.0,
            reclaim_weight=14.0,
            volume_weight=14.0,
            vwap_weight=12.0,
            cpr_weight=10.0,
            speed_weight=12.0,
            context_low=-4.0,
            context_high=2.0,
            distribute_penalty=-12.0,
            neutral_penalty=-5.0,
            weak_vwap_penalty_value=-10.0,
            rescue_penalty_value=-8.0,
        )

    reclaim_target = 0.55 if route_family_name == "OPENING_RECLAIM" else 0.45
    vwap_target = 0.28 if route_family_name == "OPENING_RECLAIM" else 0.20
    cpr_target = 0.68 if route_family_name == "OPENING_RECLAIM" else 0.62
    reclaim_component = _peak_score(reclaim_score, target=reclaim_target, width=0.45)
    vwap_component = _peak_score(vwap_score, target=vwap_target, width=0.28)
    cpr_component = _peak_score(cpr_score, target=cpr_target, width=0.28)
    extension_penalty = 0.0
    if reclaim_score > 0.85:
        extension_penalty -= _clip01((reclaim_score - 0.85) / 0.15) * 4.0
    if vwap_score > 0.60:
        extension_penalty -= _clip01((vwap_score - 0.60) / 0.40) * 6.0
    if cpr_score > 0.85:
        extension_penalty -= _clip01((cpr_score - 0.85) / 0.15) * 6.0
    return _bundle(
        daily_weight=54.0,
        reclaim_weight=8.0,
        volume_weight=12.0,
        vwap_weight=5.0,
        cpr_weight=6.0,
        speed_weight=8.0,
        context_low=-4.0,
        context_high=2.0,
        distribute_penalty=-12.0,
        neutral_penalty=-5.0,
        weak_vwap_penalty_value=-10.0,
        rescue_penalty_value=-8.0,
        reclaim_input=reclaim_component,
        vwap_input=vwap_component,
        cpr_input=cpr_component,
        extension_penalty=extension_penalty,
    )


def _dislocation_bar_index(
    settings: Any,
    state: Any,
    series: Sequence[Bar],
    bar_idx: int,
) -> int:
    """Index of the last completed bar that traded below the dislocation band.

    Falls back to the session low bar when no band is available so the speed
    component still measures a real span rather than a constant.
    """

    band = dislocation_band(settings, state)
    upper = min(int(bar_idx), len(series) - 1)
    if upper < 0:
        return 0
    if band > 0:
        for idx in range(upper, -1, -1):
            if float(series[idx].low) <= band:
                return idx
    lows = [float(series[idx].low) for idx in range(upper + 1)]
    if not lows:
        return 0
    return int(lows.index(min(lows)))


def activate_open_scored_direct_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    """Accept the first causal direct OPEN_SCORED event on a completed bar.

    The nightly signal is admission context.  This transition owns the actual
    completed-bar confirmation and score in both live and replay; adapters may
    submit/fill only on the following bar.  A failed bar leaves the symbol in
    WATCHING so later confirmations inside the entry window remain eligible.
    """

    if open_scored_transition(settings) not in {"next_bar", "reclaim_or_limit"}:
        return None
    if getattr(state, "stage", "") != "WATCHING":
        return None
    after_bar = (
        int(getattr(settings, "pb_v2_open_scored_after_bar", 0))
        if bool(getattr(settings, "pb_v2_enabled", False))
        else 0
    )
    if bar_idx < after_bar:
        return None
    if not open_scored_eligible(
        settings,
        {
            "daily_signal_score": getattr(state, "daily_signal_score", 0.0),
            "daily_signal_rank_pct": getattr(
                state,
                "daily_signal_rank_pct",
                getattr(state, "entry_rank_pct", 100.0),
            ),
            "rescue_flow_candidate": getattr(state, "rescue_flow_candidate", False),
            "trigger_types": list(getattr(state, "trigger_types", []) or []),
        },
    ):
        return None
    confirmation_policy = str(
        getattr(settings, "pb_v2_open_scored_confirmation_policy", "any") or "any"
    ).lower()
    entry_window_bars = int(
        getattr(settings, "pb_v2_open_scored_entry_window_bars", 0) or 0
    )
    if entry_window_bars > 0 and bar_idx >= after_bar + entry_window_bars:
        return None

    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]

    if confirmation_policy == "band_reclaim":
        if not band_reclaim_confirmed(settings, bar, market, state=state):
            return None
    elif not open_scored_bar_confirmed(settings, bar, market):
        return None

    session_low = _state_or_market_session_low(state, market, bar)
    rescue_lane = bool(
        getattr(state, "rescue_flow_candidate", False)
        and getattr(settings, "pb_rescue_event_lane_enabled", False)
    )
    route_family = "OPEN_SCORED_RESCUE_ENTRY" if rescue_lane else "OPEN_SCORED_ENTRY"
    state.route_family = route_family
    state.intraday_setup_type = route_family
    daily_atr_value = float(getattr(state, "daily_atr", 0.0))
    stop_anchor = (
        event_stop_anchor(settings, state, bar, session_low, daily_atr_value)
        if confirmation_policy == "band_reclaim"
        else session_low
    )
    state.setup_low = stop_anchor
    state.reclaim_level = float(bar.open)
    state.stop_level = compute_initial_stop(
        settings,
        stop_anchor,
        daily_atr_value,
        session_atr,
    )
    # Record the bar on which the dislocation was actually established so the
    # reclaim-speed component measures a real span instead of being pinned to a
    # hardcoded zero (which made it arithmetically constant on this route).
    state.flush_bar_idx = _dislocation_bar_index(settings, state, series, bar_idx)
    score_bundle = compute_route_entry_score_bundle(
        settings,
        state,
        item,
        bar,
        market,
        bar_idx,
        bars=series,
    )
    score = float(score_bundle["score"])
    if score < entry_threshold(settings, state) or not open_scored_entry_score_eligible(
        settings,
        score,
    ):
        # Preserve WATCHING/retry semantics without leaking a failed bar's
        # route geometry into a later decision.
        reset_route_state(state)
        return None

    prior = str(getattr(state, "stage", "WATCHING"))
    state.stage = "READY"
    state.intraday_score = score
    state.score_components = dict(score_bundle)
    state.ready_bar_idx = int(bar_idx)
    state.ready_timestamp = bar.end_time
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.accepted_session_atr = float(session_atr)
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "open_scored_completed_bar_confirmed"
    acceptance = IARICEntryAcceptance(
        accepted_bar_idx=int(bar_idx),
        accepted_timestamp=bar.end_time,
        accepted_entry_price=float(bar.close),
        entry_trigger=route_family,
        route_family=route_family,
        score=score,
        session_atr=float(session_atr),
        score_components=dict(score_bundle),
        lane_id=lane_id_for_route(route_family),
    )
    return IARICRouteStep(
        prior_stage=prior,
        stage="READY",
        reason="next_bar_open_fill",
        score=score,
        entry_feasible=True,
        acceptance=acceptance,
    )


def arm_open_scored_retest_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    """Arm a causal pullback/recovery transition from one completed signal bar.

    The unchanged seven-component OPEN_SCORED score is an admission check, not
    an immediate fill instruction.  A later completed bar must retrace toward
    the signal-session low and recover before the adapter may submit an entry
    for the following bar.
    """

    if open_scored_transition(settings) != "confirmed_retest":
        return None
    if getattr(state, "stage", "") != "WATCHING":
        return None
    if not open_scored_eligible(
        settings,
        {
            "daily_signal_score": getattr(state, "daily_signal_score", 0.0),
            "daily_signal_rank_pct": getattr(
                state,
                "daily_signal_rank_pct",
                getattr(state, "entry_rank_pct", 100.0),
            ),
            "rescue_flow_candidate": getattr(state, "rescue_flow_candidate", False),
            "trigger_types": list(getattr(state, "trigger_types", []) or []),
        },
    ):
        return None

    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    session_atr = max(float(session_atr), 0.01)
    session_low = _state_or_market_session_low(state, market, bar)
    impulse = max(float(bar.close) - session_low, 0.0)
    min_impulse = max(
        float(getattr(settings, "pb_open_scored_retest_min_impulse_atr", 0.0)),
        0.0,
    ) * session_atr
    if impulse < min_impulse:
        return None

    prior = str(getattr(state, "stage", "WATCHING"))
    state.route_family = "OPEN_SCORED_RETEST"
    state.intraday_setup_type = "OPEN_SCORED_RETEST"
    state.setup_low = session_low
    # Preserve the repaired immediate route's score definition: the signal
    # bar open is the time-available reversal reference.
    state.reclaim_level = float(bar.open)
    state.stop_level = compute_initial_stop(
        settings,
        session_low,
        float(getattr(state, "daily_atr", 0.0)),
        session_atr,
    )
    state.flush_bar_idx = int(bar_idx)
    score_bundle = compute_route_entry_score_bundle(
        settings,
        state,
        item,
        bar,
        market,
        bar_idx,
        bars=series,
    )
    score = float(score_bundle["score"])
    if score < entry_threshold(settings, state) or not open_scored_entry_score_eligible(settings, score):
        reset_route_state(state)
        return None

    retrace_fraction = min(
        max(
            float(getattr(settings, "pb_open_scored_retest_retrace_frac", 0.35)),
            0.0,
        ),
        1.0,
    )
    target = float(bar.close) - retrace_fraction * impulse
    target = max(target, float(state.stop_level) + 0.01)
    window = max(
        int(getattr(settings, "pb_open_scored_retest_window_bars", 1)),
        1,
    )
    state.stage = "RETEST_ARMED"
    state.intraday_score = score
    state.score_components = dict(score_bundle)
    state.target_entry_price = float(target)
    # Once armed, reclaim_level stores the signal close so the confirmation
    # path can reject a renewed extension/chase without adding a score input.
    state.reclaim_level = float(bar.close)
    state.ready_bar_idx = int(bar_idx)
    state.improvement_expires = int(bar_idx + window)
    state.acceptance_count = 0
    state.required_acceptance = 1
    state.ready_timestamp = bar.end_time
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "open_scored_retest_armed"
    return IARICRouteStep(
        prior_stage=prior,
        stage="RETEST_ARMED",
        reason="open_scored_retest_armed",
        score=score,
    )


def arm_open_scored_retrace_limit_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    """Create a causal resting pullback order from one completed signal bar.

    Selection uses the unchanged OPEN_SCORED seven-component bundle.  The
    returned limit becomes actionable only after this completed bar, so a
    replay adapter must never fill it on ``bar_idx`` itself.
    """

    transition = open_scored_transition(settings)
    if transition not in {"resting_retrace", "reclaim_or_limit"}:
        return None
    if getattr(state, "stage", "") != "WATCHING":
        return None
    if transition == "reclaim_or_limit":
        # Additive, not alternative: the resting bid is armed only after the
        # reclaim route has been given its window on this symbol today, so the
        # two mechanisms cover disjoint sessions instead of competing.
        arm_bar = int(getattr(settings, "pb_open_scored_limit_arm_bar", 3) or 0)
        if int(bar_idx) != arm_bar:
            return None
    if not open_scored_eligible(
        settings,
        {
            "daily_signal_score": getattr(state, "daily_signal_score", 0.0),
            "daily_signal_rank_pct": getattr(
                state,
                "daily_signal_rank_pct",
                getattr(state, "entry_rank_pct", 100.0),
            ),
            "rescue_flow_candidate": getattr(state, "rescue_flow_candidate", False),
            "trigger_types": list(getattr(state, "trigger_types", []) or []),
        },
    ):
        return None

    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    session_low = _state_or_market_session_low(state, market, bar)
    prior = str(getattr(state, "stage", "WATCHING"))
    state.route_family = "OPEN_SCORED_RETRACE_LIMIT"
    state.intraday_setup_type = "OPEN_SCORED_RETRACE_LIMIT"
    state.setup_low = session_low
    state.reclaim_level = float(bar.open)
    state.stop_level = compute_initial_stop(
        settings,
        session_low,
        float(getattr(state, "daily_atr", 0.0)),
        session_atr,
    )
    state.flush_bar_idx = int(bar_idx)
    score_bundle = compute_route_entry_score_bundle(
        settings,
        state,
        item,
        bar,
        market,
        bar_idx,
        bars=series,
    )
    score = float(score_bundle["score"])
    if score < entry_threshold(settings, state) or not open_scored_entry_score_eligible(settings, score):
        reset_route_state(state)
        return None

    tick = max(float(getattr(item, "tick_size", 0.01) or 0.01), 0.01)
    anchor_mode = str(
        getattr(settings, "pb_open_scored_limit_anchor", "impulse_retrace")
        or "impulse_retrace"
    ).lower()
    if anchor_mode not in {"impulse_retrace", "daily_atr"}:
        raise ValueError(
            "pb_open_scored_limit_anchor must be 'impulse_retrace' or 'daily_atr'"
        )

    if anchor_mode == "daily_atr":
        # Independent entry mechanism: rest a bid a fixed fraction of daily ATR
        # *below the session open* and let a continuing decline come to it.
        #
        # This is deliberately not the impulse retrace.  That construct requires
        # a bounce to have already happened and then buys a pullback within it,
        # so it only fires on days the reclaim route already covers.  A
        # below-open bid instead adds the days where price never reclaims, which
        # is where a reversion book is otherwise blind.
        session_open = float(getattr(market, "session_open", 0.0) or 0.0)
        if session_open <= 0:
            session_open = float(series[0].open)
        atr = float(getattr(state, "daily_atr", 0.0) or 0.0)
        frac = max(float(getattr(settings, "pb_open_scored_limit_atr_frac", 0.25) or 0.0), 0.0)
        if atr <= 0 or frac <= 0:
            reset_route_state(state)
            return None
        target = session_open - frac * atr
        if target <= 0 or target >= float(bar.close):
            # Already trading at or below the bid: no improvement available, and
            # filling at the current price would silently become a market entry.
            reset_route_state(state)
            return None
        # The structural stop must sit below the resting bid, otherwise the fill
        # would arrive already through its own stop.
        state.stop_level = min(
            float(state.stop_level),
            target - max(float(session_atr) * 0.5, tick),
        )
        state.stop_level = max(state.stop_level, 0.01)
    else:
        retrace_fraction = min(
            max(
                float(
                    getattr(
                        settings,
                        "pb_open_scored_retrace_limit_fraction",
                        0.35,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        impulse = max(float(bar.close) - session_low, 0.0)
        target = float(bar.close) - retrace_fraction * impulse
    target = max(target, float(state.stop_level) + tick)
    window = max(
        int(getattr(settings, "pb_open_scored_retrace_limit_window_bars", 12)),
        1,
    )
    state.stage = "READY"
    state.intraday_score = score
    state.score_components = dict(score_bundle)
    state.target_entry_price = float(target)
    state.ready_bar_idx = int(bar_idx)
    state.improvement_expires = int(bar_idx + window)
    state.ready_timestamp = bar.end_time
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "open_scored_retrace_limit_armed"
    acceptance = IARICEntryAcceptance(
        accepted_bar_idx=int(bar_idx),
        accepted_timestamp=bar.end_time,
        accepted_entry_price=float(target),
        entry_trigger="OPEN_SCORED_RETRACE_LIMIT",
        route_family="OPEN_SCORED_RETRACE_LIMIT",
        score=score,
        session_atr=float(session_atr),
        score_components=dict(score_bundle),
    )
    return IARICRouteStep(
        prior_stage=prior,
        stage="READY",
        reason="resting_retrace_limit",
        score=score,
        entry_feasible=True,
        acceptance=acceptance,
    )


def advance_open_scored_retest_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    """Advance an armed retest using completed data; accept for next-bar fill."""

    if getattr(state, "stage", "") != "RETEST_ARMED":
        return None
    armed_bar_idx = int(getattr(state, "ready_bar_idx", -1))
    if bar_idx <= armed_bar_idx:
        return None
    if bar_idx > int(getattr(state, "improvement_expires", armed_bar_idx)):
        return invalidate_route_state(state, "open_scored_retest_expired", bar_idx + 1_000)
    if bar.low <= float(getattr(state, "stop_level", 0.0)):
        return invalidate_route_state(state, "open_scored_retest_failed", bar_idx + 1_000)

    target = float(getattr(state, "target_entry_price", 0.0))
    if target <= 0:
        return invalidate_route_state(state, "open_scored_retest_missing_target", bar_idx + 1_000)
    if bar.low <= target:
        state.acceptance_count = 1

    close_pct = _close_in_range_pct(bar.high, bar.low, bar.close)
    max_extension = max(
        float(getattr(settings, "pb_open_scored_retest_max_extension_atr", 0.35)),
        0.0,
    ) * max(float(session_atr), 0.01)
    signal_close = float(getattr(state, "reclaim_level", bar.close))
    confirmed = (
        int(getattr(state, "acceptance_count", 0)) >= 1
        and bar.close > bar.open
        and close_pct
        >= float(getattr(settings, "pb_open_scored_retest_min_close_pct", 0.55))
        and bar.close >= target
        and bar.close <= signal_close + max_extension
    )
    if not confirmed:
        return None

    prior = str(getattr(state, "stage", "RETEST_ARMED"))
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    state.ready_timestamp = bar.end_time
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "open_scored_retest_confirmed"
    acceptance = IARICEntryAcceptance(
        accepted_bar_idx=int(bar_idx),
        accepted_timestamp=bar.end_time,
        accepted_entry_price=float(bar.close),
        entry_trigger="OPEN_SCORED_RETEST",
        route_family="OPEN_SCORED_RETEST",
        score=float(getattr(state, "intraday_score", 0.0)),
        session_atr=float(session_atr),
        score_components=dict(getattr(state, "score_components", {}) or {}),
    )
    return IARICRouteStep(
        prior_stage=prior,
        stage="READY",
        reason="next_bar_open_fill",
        score=acceptance.score,
        entry_feasible=True,
        acceptance=acceptance,
    )


def apply_entry_acceptance(state: Any, acceptance: IARICEntryAcceptance) -> None:
    state.accepted_bar_idx = int(acceptance.accepted_bar_idx)
    state.accepted_timestamp = acceptance.accepted_timestamp
    state.accepted_entry_price = float(acceptance.accepted_entry_price)
    state.accepted_entry_trigger = str(acceptance.entry_trigger)
    state.accepted_route_family = str(acceptance.route_family)
    state.accepted_score = float(acceptance.score)
    state.accepted_session_atr = float(acceptance.session_atr)
    state.accepted_score_components = dict(acceptance.score_components)
    if hasattr(state, "accepted_lane_id"):
        state.accepted_lane_id = str(
            acceptance.lane_id
            or lane_id_for_route(
                acceptance.route_family,
                rescue_candidate=bool(getattr(state, "rescue_flow_candidate", False)),
            )
        )
    for state_name, value in (
        ("accepted_event_id", str(acceptance.event_id)),
        ("accepted_reversion_anchor", float(acceptance.reversion_anchor)),
        ("accepted_stop_anchor", float(acceptance.stop_anchor)),
        ("accepted_remaining_room_atr", float(acceptance.remaining_room_atr)),
        ("accepted_prospective_reward_risk", float(acceptance.prospective_reward_risk)),
    ):
        if hasattr(state, state_name):
            setattr(state, state_name, value)


def reset_route_state(state: Any) -> None:
    reset_for_watch = getattr(state, "reset_for_watch", None)
    if callable(reset_for_watch):
        reset_for_watch()
        if hasattr(state, "last_transition_reason"):
            state.last_transition_reason = ""
        return
    state.stage = "WATCHING"
    state.intraday_setup_type = ""
    state.route_family = ""
    state.setup_low = 0.0
    state.reclaim_level = 0.0
    state.stop_level = 0.0
    state.flush_bar_idx = 0
    state.ready_bar_idx = -1
    state.acceptance_count = 0
    state.required_acceptance = 0
    state.intraday_score = 0.0
    state.target_entry_price = 0.0
    state.improvement_expires = 0
    state.invalid_reason = ""
    state.invalid_reset_bar = 0
    state.score_components = {}
    state.ready_cpr = 0.0
    state.ready_volume_ratio = 0.0
    state.ready_timestamp = None
    state.accepted_bar_idx = -1
    state.accepted_timestamp = None
    state.accepted_entry_price = 0.0
    state.accepted_entry_trigger = ""
    state.accepted_route_family = ""
    state.accepted_score = 0.0
    state.accepted_session_atr = 0.0
    state.accepted_score_components = {}
    if hasattr(state, "accepted_lane_id"):
        state.accepted_lane_id = ""
    for state_name, value in (
        ("accepted_event_id", ""),
        ("accepted_reversion_anchor", 0.0),
        ("accepted_stop_anchor", 0.0),
        ("accepted_remaining_room_atr", 0.0),
        ("accepted_prospective_reward_risk", 0.0),
        ("opportunity_event_id", ""),
        ("opportunity_reversion_anchor", 0.0),
        ("opportunity_stop_anchor", 0.0),
        ("opportunity_remaining_room_atr", 0.0),
        ("opportunity_prospective_reward_risk", 0.0),
    ):
        if hasattr(state, state_name):
            setattr(state, state_name, value)
    if hasattr(state, "opportunity_family"):
        state.opportunity_family = ""
    if hasattr(state, "opportunity_signal_bar_idx"):
        state.opportunity_signal_bar_idx = -1
    if hasattr(state, "opportunity_signal_close"):
        state.opportunity_signal_close = 0.0
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = ""


def maybe_reset_invalidated_state(state: Any, bar_idx: int) -> bool:
    if getattr(state, "stage", "") != "INVALIDATED":
        return False
    if bar_idx < int(getattr(state, "invalid_reset_bar", 0)):
        return False
    reset_route_state(state)
    return True


def invalidate_route_state(state: Any, reason: str, reset_bar: int) -> IARICRouteStep:
    prior = str(getattr(state, "stage", "WATCHING"))
    state.stage = "INVALIDATED"
    state.invalid_reason = reason
    state.invalid_reset_bar = int(reset_bar)
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = reason
    return IARICRouteStep(prior_stage=prior, stage="INVALIDATED", reason=reason)


def _state_or_market_session_low(state: Any, market: MarketSnapshot, bar: Bar) -> float:
    state_session_low = float(getattr(state, "session_low", 0.0))
    market_session_low = float(market.session_low) if market.session_low is not None else 0.0
    base_low = state_session_low if state_session_low > 0 else market_session_low if market_session_low > 0 else bar.low
    return min(base_low, bar.low)


def advance_opening_reclaim_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if float(getattr(state, "daily_signal_score", 0.0)) < route_min_daily_signal_score(settings, "OPENING_RECLAIM"):
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    session_low = _state_or_market_session_low(state, market, bar)
    if state.stage == "WATCHING":
        first_bar_open = series[0].open if series else bar.open
        flush_distance = (first_bar_open - session_low) / max(session_atr, 0.01)
        flush_bar = (
            bar_idx < int(getattr(settings, "pb_flush_window_bars", 0))
            and flush_distance >= float(getattr(settings, "pb_flush_min_atr", 0.0))
            and bar.cpr <= float(getattr(settings, "pb_flush_cpr_max", 0.0))
        )
        micro = micropressure_label(series, min(bar_idx, len(series) - 1), bar.close, item)
        pm_reentry_signal = (
            bool(getattr(state, "stopped_out_today", False))
            and bool(getattr(settings, "pb_pm_reentry", False))
            and bar_idx >= int(getattr(settings, "pb_pm_reentry_after_bar", 0))
            and bar.close > bar.open
            and market.session_vwap is not None
            and bar.close >= market.session_vwap
            and micro == "ACCUMULATE"
        )
        if not (flush_bar or pm_reentry_signal):
            return None
        prior = state.stage
        state.stage = "FLUSH_LOCKED"
        if hasattr(state, "intraday_setup_type"):
            state.intraday_setup_type = (
                "PM_REENTRY"
                if pm_reentry_signal
                else "OPENING_FLUSH" if bar_idx < int(getattr(settings, "pb_opening_range_bars", 0)) else "SESSION_FLUSH"
            )
        state.route_family = "OPENING_RECLAIM"
        state.setup_low = session_low
        reclaim_anchor = max(
            bar.high - float(getattr(settings, "pb_reclaim_offset_atr", 0.0)) * session_atr,
            (market.session_vwap or bar.close) - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr,
        )
        state.reclaim_level = max(reclaim_anchor, session_low + session_atr * 0.25)
        state.stop_level = compute_initial_stop(settings, state.setup_low, float(getattr(state, "daily_atr", 0.0)), session_atr)
        state.flush_bar_idx = int(bar_idx)
        if hasattr(state, "last_transition_reason"):
            state.last_transition_reason = "flush_detected"
        return IARICRouteStep(prior_stage=prior, stage="FLUSH_LOCKED", reason=str(getattr(state, "intraday_setup_type", "flush_detected")))

    if state.stage == "FLUSH_LOCKED":
        state.setup_low = min(float(getattr(state, "setup_low", bar.low)), bar.low)
        reclaim_anchor = max(
            bar.high - float(getattr(settings, "pb_reclaim_offset_atr", 0.0)) * session_atr,
            (market.session_vwap or bar.close) - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr,
        )
        state.reclaim_level = max(reclaim_anchor, float(getattr(state, "setup_low", bar.low)) + session_atr * 0.25)
        state.stop_level = compute_initial_stop(settings, state.setup_low, float(getattr(state, "daily_atr", 0.0)), session_atr)
        if bar.close >= float(getattr(state, "reclaim_level", 0.0)) or bar.high >= float(getattr(state, "reclaim_level", 0.0)):
            prior = state.stage
            state.stage = "RECLAIMING"
            state.required_acceptance = max(1, int(getattr(settings, "pb_ready_acceptance_bars", 1)))
            state.acceptance_count = 0
            if hasattr(state, "last_transition_reason"):
                state.last_transition_reason = "reclaim_hit"
            return IARICRouteStep(prior_stage=prior, stage="RECLAIMING", reason="reclaim_hit")
        if bar_idx >= int(getattr(settings, "pb_flush_window_bars", 0)) + int(getattr(settings, "pb_ready_acceptance_bars", 0)):
            return invalidate_route_state(
                state,
                "flush_stale",
                max(bar_idx + 1, int(getattr(settings, "pb_delayed_confirm_after_bar", 0))),
            )
        return None

    if state.stage != "RECLAIMING":
        return None
    if bar.low <= float(getattr(state, "stop_level", 0.0)) or bar.close < float(getattr(state, "setup_low", 0.0)):
        reset_bar = max(
            bar_idx + 2,
            int(getattr(settings, "pb_pm_reentry_after_bar", 0)) if bool(getattr(state, "stopped_out_today", False)) else bar_idx + 2,
        )
        return invalidate_route_state(state, "reclaim_failed", reset_bar)
    micro = micropressure_label(series, min(bar_idx, len(series) - 1), float(getattr(state, "reclaim_level", 0.0)), item)
    volume_ok = compute_volume_ratio(bar, item) >= float(getattr(settings, "pb_ready_min_volume_ratio", 0.0))
    cpr_ok = bar.cpr >= float(getattr(settings, "pb_ready_min_cpr", 0.0))
    vwap_ok = market.session_vwap is None or bar.close >= market.session_vwap - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr
    if bar.close >= float(getattr(state, "reclaim_level", 0.0)) and bar.close > bar.open and cpr_ok and volume_ok and vwap_ok and micro != "DISTRIBUTE":
        state.acceptance_count = int(getattr(state, "acceptance_count", 0)) + 1
    elif bar.close < float(getattr(state, "reclaim_level", 0.0)):
        state.acceptance_count = max(int(getattr(state, "acceptance_count", 0)) - 1, 0)
    if int(getattr(state, "acceptance_count", 0)) < max(1, int(getattr(state, "required_acceptance", 1))):
        return None
    prior = state.stage
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series, micropressure=micro)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = max(
        float(getattr(state, "reclaim_level", 0.0)),
        bar.close * (1.0 - float(getattr(settings, "pb_improvement_discount_pct", 0.0))),
    )
    state.improvement_expires = bar_idx + max(0, int(getattr(settings, "pb_improvement_window_bars", 0)))
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "acceptance_complete"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="acceptance_complete", score=state.intraday_score)


def activate_delayed_confirm_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if bool(getattr(state, "stopped_out_today", False)):
        return None
    if bool(getattr(state, "rescue_flow_candidate", False)) and not bool(getattr(settings, "pb_v2_delayed_confirm_allow_rescue", False)):
        return None
    if not route_enabled(settings, "DELAYED_CONFIRM"):
        return None
    if float(getattr(state, "daily_signal_score", 0.0)) < route_min_daily_signal_score(settings, "DELAYED_CONFIRM"):
        return None
    if bar_idx < int(getattr(settings, "pb_delayed_confirm_after_bar", 0)):
        return None
    if getattr(state, "stage", "") != "WATCHING":
        return None
    vwap = market.session_vwap
    if vwap is None:
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    session_low = _state_or_market_session_low(state, market, bar)
    close_pct = _close_in_range_pct(bar.high, bar.low, bar.close)
    micro = micropressure_label(series, min(bar_idx, len(series) - 1), vwap, item)
    if bool(getattr(settings, "pb_v2_enabled", False)):
        min_close_pct = float(getattr(settings, "pb_v2_delayed_confirm_min_close_pct", 0.0))
        vol_ratio_min = float(getattr(settings, "pb_v2_delayed_confirm_vol_ratio", 0.0))
        volume_ok = compute_volume_ratio(bar, item) >= vol_ratio_min
        vwap_ok = bar.close >= vwap - 0.50 * session_atr
        if bar.close <= bar.open or close_pct < min_close_pct or not volume_ok or not vwap_ok or micro == "DISTRIBUTE":
            return None
    else:
        volume_ok = compute_volume_ratio(bar, item) >= max(float(getattr(settings, "pb_ready_min_volume_ratio", 0.0)) * 0.75, 0.5)
        vwap_ok = bar.close >= vwap - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr
        retest_depth = (series[0].open - session_low) / max(session_atr, 0.01)
        bounce_strength = (bar.close - session_low) / max(session_atr, 0.01)
        if (
            bar.close <= bar.open
            or close_pct < float(getattr(settings, "pb_delayed_confirm_min_close_pct", 0.0))
            or not volume_ok
            or not vwap_ok
            or micro == "DISTRIBUTE"
            or retest_depth < 0.05
            or bounce_strength < 0.20
        ):
            return None
    state.intraday_setup_type = "DELAYED_CONFIRM"
    state.route_family = "DELAYED_CONFIRM"
    state.setup_low = session_low
    state.reclaim_level = max(vwap, session_low + session_atr * 0.35)
    state.stop_level = compute_initial_stop(settings, state.setup_low, float(getattr(state, "daily_atr", 0.0)), session_atr)
    state.flush_bar_idx = max(0, bar_idx - int(getattr(settings, "pb_delayed_confirm_after_bar", 0)) + 1)
    state.acceptance_count = 1
    state.required_acceptance = 1
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series, micropressure=micro)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    if state.intraday_score < float(getattr(settings, "pb_delayed_confirm_score_min", 0.0)):
        state.intraday_setup_type = ""
        state.setup_low = 0.0
        state.reclaim_level = 0.0
        state.stop_level = 0.0
        state.flush_bar_idx = 0
        state.acceptance_count = 0
        state.required_acceptance = 0
        state.intraday_score = 0.0
        state.score_components = {}
        return None
    prior = "WATCHING"
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = max(
        float(getattr(state, "reclaim_level", 0.0)),
        bar.close * (1.0 - float(getattr(settings, "pb_improvement_discount_pct", 0.0)) * 0.5),
    )
    state.improvement_expires = bar_idx + max(0, int(getattr(settings, "pb_improvement_window_bars", 0)))
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "delayed_confirm"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="delayed_confirm", score=state.intraday_score)


def activate_vwap_bounce_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if not bool(getattr(settings, "pb_v2_enabled", False)) or not bool(getattr(settings, "pb_v2_vwap_bounce_enabled", False)):
        return None
    if bool(getattr(state, "stopped_out_today", False)):
        return None
    if bool(getattr(state, "rescue_flow_candidate", False)) and not bool(getattr(settings, "pb_v2_vwap_bounce_allow_rescue", False)):
        return None
    if getattr(state, "stage", "") != "WATCHING" or bar_idx < int(getattr(settings, "pb_v2_vwap_bounce_after_bar", 0)):
        return None
    vwap = market.session_vwap
    if vwap is None or session_atr <= 0:
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    touched_below = any(sample.low < vwap for sample in series[: min(12, max(bar_idx, 0))])
    if not touched_below:
        return None
    if bar.close <= vwap or bar.close <= bar.open:
        return None
    if compute_volume_ratio(bar, item) < float(getattr(settings, "pb_v2_vwap_bounce_vol_ratio", 0.0)):
        return None
    micro = micropressure_label(series, min(bar_idx, len(series) - 1), vwap, item)
    if micro == "DISTRIBUTE":
        return None
    session_low = _state_or_market_session_low(state, market, bar)
    state.intraday_setup_type = "VWAP_BOUNCE"
    state.route_family = "VWAP_BOUNCE"
    state.setup_low = session_low
    state.reclaim_level = vwap
    state.stop_level = max(session_low - 0.25 * session_atr, 0.01)
    state.flush_bar_idx = 0
    state.acceptance_count = 1
    state.required_acceptance = 1
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series, micropressure=micro)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    prior = "WATCHING"
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = bar.close
    state.improvement_expires = bar_idx + 2
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "vwap_bounce"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="vwap_bounce", score=state.intraday_score)


def activate_afternoon_retest_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if not bool(getattr(settings, "pb_v2_enabled", False)) or not bool(getattr(settings, "pb_v2_afternoon_retest_enabled", False)):
        return None
    if bool(getattr(state, "rescue_flow_candidate", False)) and not bool(getattr(settings, "pb_v2_afternoon_retest_allow_rescue", False)):
        return None
    if getattr(state, "stage", "") != "WATCHING" or bar_idx < int(getattr(settings, "pb_v2_afternoon_retest_after_bar", 0)):
        return None
    if float(getattr(state, "daily_signal_score", 0.0)) < float(getattr(settings, "pb_v2_afternoon_retest_min_score", 0.0)):
        return None
    vwap = market.session_vwap
    if vwap is None or session_atr <= 0:
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    session_low = _state_or_market_session_low(state, market, bar)
    if bar.low < 0.95 * session_low:
        return None
    if bar.close <= vwap:
        return None
    avg_vol = float(sum(sample.volume for sample in series[: bar_idx + 1]) / max(bar_idx + 1, 1)) if series else 0.0
    if avg_vol > 0 and bar.volume > 1.5 * avg_vol:
        return None
    state.intraday_setup_type = "AFTERNOON_RETEST"
    state.route_family = "AFTERNOON_RETEST"
    state.setup_low = session_low
    state.reclaim_level = vwap
    state.stop_level = max(session_low - 0.40 * session_atr, 0.01)
    state.flush_bar_idx = 0
    state.acceptance_count = 1
    state.required_acceptance = 1
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    prior = "WATCHING"
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = bar.close
    state.improvement_expires = bar_idx + 2
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "afternoon_retest"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="afternoon_retest", score=state.intraday_score)


def _aperture_families(settings: Any) -> set[str]:
    raw = getattr(settings, "pb_aperture_families", "")
    values = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
    return {str(value).strip().upper() for value in values if str(value).strip()}


def _aperture_family_mapping(settings: Any, setting: str) -> dict[str, str]:
    """Parse and validate a frozen family-policy mapping.

    Centralizing this parser keeps the live engine, replay engine, optimizer,
    and snapshot-restored settings on one contract. Unknown families are hard
    errors instead of silently creating a backtest-only policy.
    """

    raw = str(getattr(settings, setting, "") or "")
    mappings: dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        separator = ":" if ":" in token else "=" if "=" in token else ""
        if not separator:
            raise ValueError(f"{setting} entries must use FAMILY:value")
        key, value = token.split(separator, 1)
        family = key.strip().upper()
        if family not in REVERSION_FAMILIES:
            raise ValueError(f"{setting} contains unknown reversion family {family!r}")
        if family in mappings:
            raise ValueError(f"{setting} contains duplicate family {family!r}")
        mappings[family] = value.strip().lower()
    return mappings


def aperture_family_score_floor(settings: Any, family: str) -> float:
    family_key = str(family or "").strip().upper()
    raw = _aperture_family_mapping(settings, "pb_aperture_family_score_floors").get(
        family_key
    )
    if raw is None:
        value = float(getattr(settings, "pb_aperture_event_score_min", 70.0))
        if not 0.0 <= value <= 100.0:
            raise ValueError("pb_aperture_event_score_min must be between 0 and 100")
        return value
    value = float(raw)
    if not 0.0 <= value <= 100.0:
        raise ValueError("aperture family score floors must be between 0 and 100")
    return value


def aperture_family_filter(settings: Any, family: str) -> str:
    family_key = str(family or "").strip().upper()
    policy = _aperture_family_mapping(settings, "pb_aperture_family_filters").get(
        family_key,
        "none",
    )
    if policy not in {
        "none",
        "geometry",
        "participation",
        "deep_reclaim",
        "residual_reclaim",
        "room_reclaim",
        "quiet_deep_room",
        "relative_exhaustion",
    }:
        raise ValueError(
            "aperture family filters must be 'geometry', 'participation', "
            "'deep_reclaim', 'residual_reclaim', 'room_reclaim', "
            "'quiet_deep_room', or 'relative_exhaustion'"
        )
    return policy


def _aperture_quality_policy_passes(event: OpportunityEvent, policy: str) -> bool:
    """Evaluate structural gates in their documented economic units.

    Score components are normalized ranking features, not ATR/volume values.
    Comparing an ATR threshold to a square-root transformed component created
    contradictory slivers (notably the inert residual route).  Current events
    always carry raw geometry; component fallbacks retain snapshot hydration
    compatibility only.
    """

    components = event.score_components
    raw_geometry_available = all(
        hasattr(event, name)
        for name in (
            "dislocation_atr",
            "reclaim_atr",
            "relative_volume",
            "residual_dislocation_atr",
            "reversion_room_atr",
        )
    )
    if not raw_geometry_available:
        # Hydrated pre-schema snapshots retain their historical transformed
        # policy semantics. They are compatibility inputs only and can never
        # establish the repaired representative baseline.
        if policy == "none":
            return True
        if policy == "geometry":
            return (
                float(event.score) >= 40.0
                and float(components.get("reclaim", 0.0)) >= 0.40
                and float(components.get("close_quality", 0.0)) >= 0.60
            )
        if policy == "participation":
            return (
                float(event.score) >= 40.0
                and float(components.get("relative_volume", 0.0)) >= 0.25
            )
        if policy == "deep_reclaim":
            return (
                float(components.get("dislocation", 0.0)) >= 0.40
                and float(components.get("reclaim", 0.0)) >= 0.35
                and float(components.get("close_quality", 0.0)) >= 0.55
            )
        if policy == "residual_reclaim":
            return (
                float(components.get("residual_dislocation", 0.0)) >= 0.35
                and float(components.get("reclaim", 0.0)) >= 0.35
                and float(components.get("close_quality", 0.0)) >= 0.55
            )
        if policy == "room_reclaim":
            return (
                float(components.get("reversion_room", 0.0)) >= 0.30
                and float(components.get("reclaim", 0.0)) >= 0.35
                and float(components.get("close_quality", 0.0)) >= 0.55
            )
        if policy == "quiet_deep_room":
            return (
                float(event.score) >= 40.0
                and float(components.get("relative_volume", 0.0)) <= 0.50
                and float(components.get("reversion_room", 0.0)) >= 0.50
            )
        if policy == "relative_exhaustion":
            return (
                float(event.score) >= 40.0
                and float(components.get("residual_dislocation", 0.0)) >= 0.75
                and float(components.get("reversion_room", 0.0)) <= 0.25
            )
        raise ValueError(f"unknown aperture quality policy {policy!r}")
    component_dislocation = float(components.get("dislocation", 0.0))
    component_residual = float(components.get("residual_dislocation", 0.0))
    component_room = float(components.get("reversion_room", 0.0))
    dislocation_atr = abs(float(getattr(
        event,
        "dislocation_atr",
        2.0 * component_dislocation * component_dislocation,
    )))
    reclaim_atr = max(float(getattr(
        event,
        "reclaim_atr",
        float(components.get("reclaim", 0.0)) * max(dislocation_atr, 0.20),
    )), 0.0)
    close_quality = float(getattr(event, "close_in_range", components.get("close_quality", 0.0)))
    relative_volume = max(float(getattr(
        event,
        "relative_volume",
        components.get("relative_volume", 0.0),
    )), 0.0)
    residual_atr = abs(min(float(getattr(
        event,
        "residual_dislocation_atr",
        -2.0 * component_residual * component_residual,
    )), 0.0))
    room_atr = max(float(getattr(
        event,
        "reversion_room_atr",
        2.0 * component_room * component_room,
    )), 0.0)
    if policy == "none":
        return True
    if policy == "geometry":
        return (
            float(event.score) >= 40.0
            and reclaim_atr >= 0.20
            and close_quality >= 0.60
        )
    if policy == "participation":
        return (
            float(event.score) >= 40.0
            and relative_volume >= 0.75
        )
    if policy == "deep_reclaim":
        return (
            dislocation_atr >= 0.40
            and reclaim_atr >= 0.20
            and close_quality >= 0.55
        )
    if policy == "residual_reclaim":
        return (
            residual_atr >= 0.35
            and reclaim_atr >= 0.20
            and close_quality >= 0.55
        )
    if policy == "room_reclaim":
        return (
            room_atr >= 0.30
            and reclaim_atr >= 0.20
            and close_quality >= 0.55
        )
    if policy == "quiet_deep_room":
        # Portable trend-pullback interaction from the pre-existing atlas:
        # genuine room remains, while participation is orderly rather than a
        # high-volume liquidation. Alphabet was excluded from discovery.
        return (
            float(event.score) >= 40.0
            and 0.50 <= relative_volume <= 1.50
            and room_atr >= 0.50
        )
    if policy == "relative_exhaustion":
        # A large idiosyncratic shock must still retain enough tradable room;
        # the former transformed-component ceiling admitted only roughly
        # 0.10-0.125 raw ATR after the hard room veto.
        return (
            float(event.score) >= 40.0
            and residual_atr >= 0.75
            and room_atr >= 0.25
            and reclaim_atr >= 0.15
            and close_quality >= 0.55
        )
    raise ValueError(f"unknown aperture quality policy {policy!r}")


def aperture_event_admitted(settings: Any, event: OpportunityEvent) -> bool:
    """Apply the frozen route-specific breadth/quality policy to one event."""

    return aperture_event_admission_reason(settings, event) == "admitted"


def aperture_event_admission_reason(settings: Any, event: OpportunityEvent) -> str:
    """Return a stable lane-funnel outcome for one completed-bar event."""

    components = getattr(event, "score_components", {})
    # Older snapshots/tests predate the explicit anchor geometry fields. Their
    # seven-component score remains a supported hydration boundary; live and
    # current replay events always provide the raw values.
    reversion_room_atr = float(
        getattr(
            event,
            "reversion_room_atr",
            float(components.get("reversion_room", 0.80)) * 1.25,
        )
    )
    prospective_reward_risk = float(
        getattr(event, "prospective_reward_risk", float("inf"))
    )
    residual_dislocation_atr = float(
        getattr(
            event,
            "residual_dislocation_atr",
            -float(components.get("residual_dislocation", 0.0)),
        )
    )
    if reversion_room_atr <= 0.0:
        return "target_already_reached"
    if reversion_room_atr < float(
        getattr(settings, "pb_aperture_min_remaining_room_atr", 0.10)
    ):
        return "insufficient_remaining_room"
    if prospective_reward_risk < float(
        getattr(settings, "pb_aperture_min_prospective_rr", 0.60)
    ):
        return "insufficient_prospective_rr"
    if (
        str(event.family).upper() == "MARKET_SECTOR_RESIDUAL_RECLAIM"
        and residual_dislocation_atr >= 0.0
    ):
        return "missing_causal_residual"
    if aperture_event_score(settings, event) < aperture_family_score_floor(settings, event.family):
        return "score_rejected"
    policy = aperture_family_filter(settings, event.family)
    if not _aperture_quality_policy_passes(event, policy):
        return "structural_policy_rejected"
    return "admitted"


def aperture_event_score(settings: Any, event: OpportunityEvent) -> float:
    """Score one event with a fixed seven-component family profile."""

    family = str(event.family or "").strip().upper()
    if family not in _aperture_family_mapping(
        settings,
        "pb_aperture_family_score_profiles",
    ):
        # Preserve the exact stored score—including compatibility fixtures and
        # historical serialized events—unless a family profile is explicit.
        return float(event.score)
    return score_from_components(
        event.score_components,
        score_profile_name(settings, family),
    )


def aperture_hybrid_uses_next_bar(settings: Any, event: OpportunityEvent) -> bool:
    """Choose the causal leg of a quality-hybrid entry on the signal bar."""

    family_key = str(event.family or "").strip().upper()
    policy = _aperture_family_mapping(
        settings, "pb_aperture_family_hybrid_next_policies"
    ).get(family_key)
    if policy not in {"deep_reclaim", "residual_reclaim", "room_reclaim"}:
        raise ValueError(
            "quality_hybrid requires a family hybrid-next policy of "
            "'deep_reclaim', 'residual_reclaim', or 'room_reclaim'"
        )
    return _aperture_quality_policy_passes(event, policy)


def aperture_family_from_route(route_family: str) -> str:
    route = str(route_family or "").strip().upper()
    if not is_aperture_route(route):
        return ""
    if route == "APERTURE_MULTIDAY_CONFIRM":
        return "MULTIDAY_HIGHER_LOW_RECLAIM"
    if route == "APERTURE_PRIOR_DAY_LOW_RETRACE_LIMIT":
        return "PRIOR_DAY_LOW_RECLAIM"
    body = route.removeprefix("APERTURE_")
    for suffix in ("_RETRACE_LIMIT", "_CONFIRM", "_ENTRY"):
        if body.endswith(suffix):
            body = body[: -len(suffix)]
            break
    return body if body in REVERSION_FAMILIES else ""


def aperture_family_daily_cap(settings: Any, route_or_family: str) -> int | None:
    family = aperture_family_from_route(route_or_family) or str(route_or_family).strip().upper()
    raw = _aperture_family_mapping(settings, "pb_aperture_family_daily_caps").get(
        family
    )
    if raw is None:
        return None
    value = int(raw)
    if value not in {1, 2}:
        raise ValueError("aperture family daily caps must use 1 or 2")
    return value


def aperture_family_max_bar(settings: Any, family: str) -> int:
    """Return a shared, causal time-of-day cutoff for one aperture family."""

    family_key = str(family or "").strip().upper()
    raw = _aperture_family_mapping(settings, "pb_aperture_family_max_bars").get(
        family_key
    )
    if raw is not None:
        value = int(raw)
        if value not in {6, 12, 24, 48}:
            raise ValueError(
                "aperture family max bars must use the pre-registered values 6, 12, 24, or 48"
            )
        return value
    if family_key == "PRIOR_DAY_LOW_RECLAIM":
        return int(getattr(settings, "pb_aperture_prior_low_max_bar", 48))
    if family_key == "MULTIDAY_HIGHER_LOW_RECLAIM":
        return int(getattr(settings, "pb_aperture_multiday_max_bar", 6))
    return int(getattr(settings, "pb_aperture_default_max_bar", 48))


def aperture_family_transition(settings: Any, family: str) -> str:
    """Return a causal entry mechanism for one aperture family.

    Explicit family mappings take precedence.  With no mapping, the two
    Research-2 routes retain their registered mechanisms and every other route
    retains the historical next-bar behaviour.  This makes the new search
    opt-in and keeps live/replay default parity.
    """

    family_key = str(family or "").strip().upper()
    raw = str(getattr(settings, "pb_aperture_family_transitions", "") or "")
    mappings: dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        separator = ":" if ":" in token else "=" if "=" in token else ""
        if not separator:
            raise ValueError(
                "pb_aperture_family_transitions entries must use FAMILY:transition"
            )
        key, value = token.split(separator, 1)
        mappings[key.strip().upper()] = value.strip().lower()
    transition = mappings.get(family_key)
    if transition is None:
        if family_key == "PRIOR_DAY_LOW_RECLAIM":
            transition = str(
                getattr(settings, "pb_aperture_prior_low_transition", "retrace")
                or "retrace"
            ).lower()
        elif family_key == "MULTIDAY_HIGHER_LOW_RECLAIM":
            transition = str(
                getattr(settings, "pb_aperture_multiday_transition", "confirm")
                or "confirm"
            ).lower()
        else:
            transition = "next_bar"
    if transition not in {"next_bar", "confirm", "retrace", "quality_hybrid"}:
        raise ValueError(
            "aperture family transition must be 'next_bar', 'confirm', 'retrace', "
            "or 'quality_hybrid'"
        )
    return transition


def _aperture_route_name(family: str, transition: str) -> str:
    family_key = str(family or "").upper()
    if family_key == "MULTIDAY_HIGHER_LOW_RECLAIM" and transition == "confirm":
        return "APERTURE_MULTIDAY_CONFIRM"
    if family_key == "PRIOR_DAY_LOW_RECLAIM" and transition == "retrace":
        return "APERTURE_PRIOR_DAY_LOW_RETRACE_LIMIT"
    suffix = "CONFIRM" if transition == "confirm" else "RETRACE_LIMIT"
    return f"APERTURE_{family_key}_{suffix}"


def advance_aperture_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
    relative_dislocation_atr: Sequence[float] | None = None,
) -> IARICRouteStep | None:
    """Advance a route-neutral, completed-bar reversion satellite.

    The nightly aperture only buys observation capacity.  Entry permission is
    created here by a discrete opportunity event.  The two initially enabled
    families use the pre-registered Research-2 transitions: a resting retrace
    after a prior-day-low reclaim, and one additional confirmation bar after an
    early multiday higher-low reclaim.  Other registered families fall back to
    a causal next-bar entry so future rounds can isolate them without changing
    the live/backtest contract.
    """

    if not bool(getattr(settings, "pb_aperture_enabled", False)):
        return None
    if not bool(getattr(item, "aperture_candidate", False)):
        return None
    if bool(getattr(settings, "pb_aperture_require_information_state", False)):
        if not bool(getattr(item, "information_state_available", False)):
            return None
        if bool(getattr(item, "earnings_risk_flag", False)):
            return None
    enabled = _aperture_families(settings)
    if not enabled:
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    series = series[: int(bar_idx) + 1]
    if not series or int(bar_idx) != len(series) - 1:
        return None

    # A confirmation route's event bar is deliberately not its entry signal.
    # Observe one more completed bar, then nominate the following bar only if
    # confirmation remains constructive.
    if str(getattr(state, "stage", "")) == "APERTURE_CONFIRM_ARMED":
        signal_idx = int(getattr(state, "opportunity_signal_bar_idx", -1))
        if int(bar_idx) != signal_idx + 1:
            if int(bar_idx) > signal_idx + 1:
                reset_route_state(state)
            return None
        width = max(float(bar.high) - float(bar.low), 1e-9)
        cpr = (float(bar.close) - float(bar.low)) / width
        confirmed = (
            float(bar.close) > float(bar.open)
            and float(bar.close) > float(getattr(state, "opportunity_signal_close", 0.0))
            and cpr >= 0.55
        )
        if not confirmed:
            reset_route_state(state)
            return IARICRouteStep(
                prior_stage="APERTURE_CONFIRM_ARMED",
                stage="WATCHING",
                reason="aperture_confirmation_failed",
            )
        family = str(getattr(state, "opportunity_family", "") or "").upper()
        state.stage = "READY"
        state.route_family = _aperture_route_name(family, "confirm")
        state.setup_low = min(float(series[signal_idx].low), float(bar.low))
        state.stop_level = max(
            float(getattr(state, "opportunity_stop_anchor", 0.0)),
            compute_initial_stop(
                settings,
                state.setup_low,
                float(getattr(state, "daily_atr", 0.0)),
                session_atr,
            ),
        )
        acceptance = IARICEntryAcceptance(
            accepted_bar_idx=int(bar_idx),
            accepted_timestamp=bar.end_time,
            accepted_entry_price=float(bar.close),
            entry_trigger=f"{family}_CONFIRM",
            route_family=state.route_family,
            score=float(getattr(state, "intraday_score", 0.0)),
            session_atr=float(session_atr),
            score_components=dict(getattr(state, "score_components", {})),
            lane_id=lane_id_for_route(state.route_family),
            event_id=str(getattr(state, "opportunity_event_id", "")),
            reversion_anchor=float(getattr(state, "opportunity_reversion_anchor", 0.0)),
            stop_anchor=float(getattr(state, "opportunity_stop_anchor", 0.0)),
            remaining_room_atr=float(getattr(state, "opportunity_remaining_room_atr", 0.0)),
            prospective_reward_risk=float(
                getattr(state, "opportunity_prospective_reward_risk", 0.0)
            ),
        )
        return IARICRouteStep(
            prior_stage="APERTURE_CONFIRM_ARMED",
            stage="READY",
            reason="aperture_confirmation_complete",
            score=acceptance.score,
            entry_feasible=True,
            acceptance=acceptance,
            lane_id=lane_id_for_route(state.route_family),
        )

    if str(getattr(state, "stage", "WATCHING")) != "WATCHING":
        return None
    context = DailyOpportunityContext(
        prev_close=float(getattr(item, "previous_close", 0.0)),
        prev_high=float(getattr(item, "previous_high", 0.0)),
        prev_low=float(getattr(item, "previous_low", 0.0)),
        daily_atr=float(getattr(item, "daily_atr_estimate", 0.0)),
        consecutive_down_days=int(getattr(item, "cdd_value", 0)),
        expected_5m_volume=float(getattr(item, "expected_5m_volume", 0.0)),
        expected_5m_profile=tuple(getattr(item, "expected_5m_profile", ()) or ()),
        five_day_return=float(getattr(item, "five_day_return", 0.0)),
        sma20_slope_atr=float(getattr(item, "sma20_slope_atr", 0.0)),
    )
    event_caps = family_event_caps(settings, enabled)
    # Detection supply is independent of entry capacity. A rejected first
    # episode must not monopolize the family for the rest of the session.
    # Re-armable families may expose a second reset episode even when their
    # entry cap is one; consumption still enforces the configured entry cap.
    detection_caps = {
        family: (2 if family in REARMABLE_FAMILIES else 1)
        for family in enabled
    }
    events = detect_completed_bar_opportunities(
        series,
        context,
        relative_dislocation_atr=(
            list(relative_dislocation_atr)[: len(series)]
            if relative_dislocation_atr is not None
            else None
        ),
        opening_range_bars=int(getattr(settings, "pb_opening_range_bars", 8)),
        require_entry_bar=False,
        max_events_per_family=detection_caps,
        min_event_separation_bars=(
            rearm_cooldown_bars(settings)
            if any(cap > 1 for cap in event_caps.values())
            else 0
        ),
        allow_episode_updates=True,
    )
    consumed = list(getattr(state, "opportunity_consumed_families", []))
    observed = [
        event for event in events
        if event.signal_bar_index == int(bar_idx)
        and event.family in enabled
        and not event_is_consumed(
            event.family,
            event.signal_bar_index,
            event_caps.get(event.family, 1),
            consumed,
            event.episode_start_bar_index,
        )
    ]
    audit_events = [
        {
            "event_id": event.event_id,
            "family": event.family,
            "lane_id": lane_id_for_route(f"APERTURE_{event.family}_ENTRY"),
            "score": aperture_event_score(settings, event),
            "reason": aperture_event_admission_reason(settings, event),
            "reversion_anchor": event.reversion_anchor,
            "stop_anchor": event.stop_anchor,
            "remaining_room_atr": event.reversion_room_atr,
            "prospective_reward_risk": event.prospective_reward_risk,
            "episode_sequence": event.episode_sequence,
        }
        for event in observed
    ]
    if hasattr(state, "opportunity_audit_bar_idx"):
        state.opportunity_audit_bar_idx = int(bar_idx)
        state.opportunity_audit_events = audit_events
    candidates = [
        event
        for event in observed
        if aperture_event_admission_reason(settings, event) == "admitted"
    ]
    if not candidates:
        return None
    event = max(
        candidates,
        key=lambda value: (aperture_event_score(settings, value), value.family),
    )
    event_score = aperture_event_score(settings, event)
    token = consumption_token(
        event.family,
        event.signal_bar_index,
        event_caps.get(event.family, 1),
        event.episode_start_bar_index,
    )
    selected_audit = next(
        (row for row in audit_events if row["event_id"] == event.event_id),
        None,
    )
    max_bar = aperture_family_max_bar(settings, event.family)
    if int(bar_idx) > max_bar:
        if selected_audit is not None:
            selected_audit["reason"] = "time_window_rejected"
        consumed.append(token)
        state.opportunity_consumed_families = consumed
        return None

    consumed.append(token)
    state.opportunity_consumed_families = consumed
    state.opportunity_family = event.family
    state.opportunity_signal_bar_idx = int(bar_idx)
    state.opportunity_signal_close = float(bar.close)
    state.opportunity_event_id = event.event_id
    state.opportunity_reversion_anchor = float(event.reversion_anchor)
    state.opportunity_stop_anchor = float(event.stop_anchor)
    state.opportunity_remaining_room_atr = float(event.reversion_room_atr)
    state.opportunity_prospective_reward_risk = float(event.prospective_reward_risk)
    state.intraday_setup_type = event.family
    state.intraday_score = float(event_score)
    state.daily_signal_score = float(event_score)
    state.score_components = dict(event.score_components)
    state.setup_low = float(bar.low)
    state.reclaim_level = float(bar.close)

    transition = aperture_family_transition(settings, event.family)
    if transition == "quality_hybrid":
        transition = (
            "next_bar"
            if aperture_hybrid_uses_next_bar(settings, event)
            else "confirm"
        )
    if transition == "confirm":
        if selected_audit is not None:
            selected_audit["reason"] = "transition_armed"
        state.stage = "APERTURE_CONFIRM_ARMED"
        state.route_family = _aperture_route_name(event.family, "confirm")
        return IARICRouteStep(
            prior_stage="WATCHING",
            stage="APERTURE_CONFIRM_ARMED",
            reason="aperture_confirmation_armed",
            score=event_score,
            lane_id=lane_id_for_route(state.route_family),
        )

    if transition == "retrace":
        if selected_audit is not None:
            selected_audit["reason"] = "retrace_armed"
        state.stage = "READY"
        state.route_family = _aperture_route_name(event.family, "retrace")
        retrace = max(
            float(event.reclaim_atr)
            * float(context.daily_atr)
            * float(getattr(settings, "pb_aperture_limit_retrace_fraction", 0.25)),
            max(float(getattr(item, "tick_size", 0.01)), 0.01) * 2.0,
        )
        state.target_entry_price = max(float(bar.close) - retrace, 0.01)
        state.improvement_expires = int(bar_idx) + max(
            int(getattr(settings, "pb_aperture_limit_window_bars", 3)), 1
        )
        state.stop_level = max(event.stop_anchor, compute_initial_stop(
            settings,
            state.setup_low,
            float(context.daily_atr),
            session_atr,
        ))
        acceptance = IARICEntryAcceptance(
            accepted_bar_idx=int(bar_idx),
            accepted_timestamp=bar.end_time,
            accepted_entry_price=float(state.target_entry_price),
            entry_trigger=event.family,
            route_family=state.route_family,
            score=float(event_score),
            session_atr=float(session_atr),
            score_components=dict(event.score_components),
            lane_id=lane_id_for_route(state.route_family),
            event_id=event.event_id,
            reversion_anchor=event.reversion_anchor,
            stop_anchor=event.stop_anchor,
            remaining_room_atr=event.reversion_room_atr,
            prospective_reward_risk=event.prospective_reward_risk,
        )
        return IARICRouteStep(
            prior_stage="WATCHING",
            stage="READY",
            reason="aperture_retrace_limit_armed",
            score=event_score,
            entry_feasible=True,
            acceptance=acceptance,
            lane_id=lane_id_for_route(state.route_family),
        )

    if selected_audit is not None:
        selected_audit["reason"] = "next_bar_ready"
    state.stage = "READY"
    state.route_family = f"APERTURE_{event.family}_ENTRY"
    state.stop_level = max(event.stop_anchor, compute_initial_stop(
        settings,
        state.setup_low,
        float(context.daily_atr),
        session_atr,
    ))
    acceptance = IARICEntryAcceptance(
        accepted_bar_idx=int(bar_idx),
        accepted_timestamp=bar.end_time,
        accepted_entry_price=float(bar.close),
        entry_trigger=event.family,
        route_family=state.route_family,
        score=float(event_score),
        session_atr=float(session_atr),
        score_components=dict(event.score_components),
        lane_id=lane_id_for_route(state.route_family),
        event_id=event.event_id,
        reversion_anchor=event.reversion_anchor,
        stop_anchor=event.stop_anchor,
        remaining_room_atr=event.reversion_room_atr,
        prospective_reward_risk=event.prospective_reward_risk,
    )
    return IARICRouteStep(
        prior_stage="WATCHING",
        stage="READY",
        reason="aperture_event_complete",
        score=event_score,
        entry_feasible=True,
        acceptance=acceptance,
        lane_id=lane_id_for_route(state.route_family),
    )


def evaluate_ready_entry(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if getattr(state, "stage", "") != "READY":
        return None
    if bar.low <= float(getattr(state, "stop_level", 0.0)):
        reset_bar = max(
            bar_idx + 2,
            int(getattr(settings, "pb_pm_reentry_after_bar", 0)) if bool(getattr(state, "stopped_out_today", False)) else bar_idx + 2,
        )
        return invalidate_route_state(state, "ready_stop_breach", reset_bar)
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    route_family_name = getattr(state, "route_family", "") or (
        "DELAYED_CONFIRM" if getattr(state, "intraday_setup_type", "") == "DELAYED_CONFIRM" else "OPENING_RECLAIM"
    )
    desired_entry = 0.0
    entry_trigger = ""
    if bar_idx > int(getattr(state, "ready_bar_idx", -1)) and bar_idx <= int(getattr(state, "improvement_expires", 0)) and bar.low <= float(getattr(state, "target_entry_price", 0.0)) <= bar.high:
        desired_entry = float(getattr(state, "target_entry_price", 0.0))
        entry_trigger = route_family_name
    elif (
        bar_idx > int(getattr(state, "ready_bar_idx", -1))
        and (
            bar_idx >= int(getattr(state, "improvement_expires", 0))
            or bar.close >= float(getattr(state, "reclaim_level", 0.0)) + session_atr * 0.25
        )
    ):
        desired_entry = max(bar.close, float(getattr(state, "reclaim_level", 0.0)))
        entry_trigger = route_family_name
    feasible = desired_entry > 0 and bool(entry_trigger)
    step = IARICRouteStep(prior_stage="READY", stage="READY", score=state.intraday_score, entry_feasible=feasible)
    if state.intraday_score < entry_threshold(settings, state) or not feasible:
        return step if feasible else None
    acceptance = IARICEntryAcceptance(
        accepted_bar_idx=int(bar_idx),
        accepted_timestamp=bar.end_time,
        accepted_entry_price=float(desired_entry),
        entry_trigger=str(entry_trigger),
        route_family=str(route_family_name),
        score=float(state.intraday_score),
        session_atr=float(session_atr),
        score_components=dict(state.score_components),
    )
    step.reason = "next_bar_open_fill"
    step.acceptance = acceptance
    return step


def on_bar(
    state: IARICCoreState,
    payload: IARICBarInput | None = None,
    *,
    bar_ts: datetime | None = None,
    entry_request: IARICEntryRequest | None = None,
    stop_update: IARICStopUpdateRequest | None = None,
    partial_exit_request: IARICPartialExitRequest | None = None,
    flatten_request: IARICFlattenRequest | None = None,
) -> tuple[
    IARICCoreState,
    list[SubmitEntry | ReplaceProtectiveStop | SubmitMarketExit | FlattenPosition | CancelAction],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | ReplaceProtectiveStop | SubmitMarketExit | FlattenPosition | CancelAction] = []
    events: list[DecisionEvent] = []

    if payload is not None and all(
        request is None
        for request in (entry_request, stop_update, partial_exit_request, flatten_request)
    ):
        events = _legacy_bar_events(payload)
        if payload.bar_ts is not None:
            _meta(next_state)["last_bar_ts"] = payload.bar_ts
        _update_last_decision(next_state, events)
        return next_state, [], events

    if bar_ts is not None:
        _meta(next_state)["last_bar_ts"] = bar_ts
    event_ts = bar_ts or datetime.now(timezone.utc)

    if entry_request is not None:
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=entry_request.symbol,
                side="BUY",
                qty=entry_request.qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                limit_price=entry_request.limit_price,
                role="entry",
                risk_context={
                    "stop_for_risk": entry_request.stop_price,
                    "planned_entry_price": entry_request.limit_price,
                },
                metadata={
                    **entry_request.metadata,
                    "route": entry_request.route,
                },
            )
        )
        events.append(
            _event(
                code="ENTRY_REQUESTED",
                ts=event_ts,
                symbol=entry_request.symbol,
                details={
                    "qty": entry_request.qty,
                    "limit_price": entry_request.limit_price,
                    "stop_price": entry_request.stop_price,
                    "route": entry_request.route,
                },
            )
        )

    if stop_update is not None:
        symbol_state = _symbol_state(next_state, stop_update.symbol)
        position = symbol_state.position if symbol_state is not None else None
        if symbol_state is not None and position is not None and position.stop_order_id:
            symbol_state.stop_level = stop_update.stop_price
            position.current_stop = stop_update.stop_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=position.stop_order_id,
                    side="SELL",
                    stop_price=stop_update.stop_price,
                    qty=min(stop_update.qty, position.qty_open),
                    reason=stop_update.reason,
                )
            )
            events.append(
                _event(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=stop_update.symbol,
                    details={
                        "stop_price": stop_update.stop_price,
                        "qty": min(stop_update.qty, position.qty_open),
                        "reason": stop_update.reason,
                    },
                )
            )

    if partial_exit_request is not None:
        symbol_state = _symbol_state(next_state, partial_exit_request.symbol)
        position = symbol_state.position if symbol_state is not None else None
        if (
            position is not None
            and position.qty_open > 0
            and position.pending_partial_stop <= 0
        ):
            position.pending_partial_stop = max(float(partial_exit_request.remainder_stop_price), 0.0)
            position.pending_partial_stop_buffer = max(float(partial_exit_request.execution_buffer), 0.0)
            actions.append(
                SubmitMarketExit(
                    client_order_id=partial_exit_request.client_order_id,
                    symbol=partial_exit_request.symbol,
                    side="SELL",
                    qty=min(partial_exit_request.qty, position.qty_open),
                    role="tp",
                    metadata={"reason": partial_exit_request.reason},
                )
            )
            events.append(
                _event(
                    code="PARTIAL_EXIT_REQUESTED",
                    ts=event_ts,
                    symbol=partial_exit_request.symbol,
                    details={
                        "qty": min(partial_exit_request.qty, position.qty_open),
                        "reason": partial_exit_request.reason,
                    },
                )
            )

    if flatten_request is not None:
        symbol_state = _symbol_state(next_state, flatten_request.symbol)
        position = symbol_state.position if symbol_state is not None else None
        if symbol_state is not None and position is not None and position.qty_open > 0:
            symbol_state.last_transition_reason = flatten_request.reason
            if symbol_state.exit_order is not None:
                if symbol_state.exit_order.role == "EXIT":
                    events.append(
                        _event(
                            code="FLATTEN_ALREADY_IN_FLIGHT",
                            ts=event_ts,
                            symbol=flatten_request.symbol,
                            details={"reason": flatten_request.reason},
                        )
                    )
                else:
                    symbol_state.pending_hard_exit = True
                    if not symbol_state.exit_order.cancel_requested:
                        symbol_state.exit_order.cancel_requested = True
                        actions.append(
                            CancelAction(
                                symbol=flatten_request.symbol,
                                target_order_id=symbol_state.exit_order.oms_order_id,
                                reason="hard_exit",
                            )
                        )
                    events.append(
                        _event(
                            code="FLATTEN_QUEUED_AFTER_CANCEL",
                            ts=event_ts,
                            symbol=flatten_request.symbol,
                            details={"reason": flatten_request.reason},
                        )
                    )
            else:
                actions.append(
                    FlattenPosition(
                        symbol=flatten_request.symbol,
                        reason=flatten_request.reason,
                        side="SELL",
                        qty=flatten_request.qty or position.qty_open,
                    )
                )
                events.append(
                    _event(
                        code="FLATTEN_REQUESTED",
                        ts=event_ts,
                        symbol=flatten_request.symbol,
                        details={
                            "reason": flatten_request.reason,
                            "qty": flatten_request.qty or position.qty_open,
                        },
                    )
                )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_order_update(
    state: IARICCoreState,
    update: IARICOrderUpdate,
) -> tuple[
    IARICCoreState,
    list[SubmitProtectiveStop | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitProtectiveStop | FlattenPosition] = []
    status = update.status.lower()
    symbol_state, symbol = _resolve_symbol_state(next_state, update.symbol, update.oms_order_id)
    role = _resolve_role(symbol_state, update.order_role, update.oms_order_id)
    event_ts = update.timestamp or datetime.now(timezone.utc)
    events: list[DecisionEvent] = []

    if not symbol:
        if update.decision_code:
            events.append(
                _event(
                    code=update.decision_code,
                    ts=event_ts,
                    symbol=update.symbol,
                    details=update.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if update.oms_order_id:
        _order_index(next_state).pop(update.oms_order_id, None)

    if status in _TERMINAL_STATUSES and symbol_state is not None:
        if role == "ENTRY":
            symbol_state.entry_order = None
            symbol_state.active_order_id = None
            _pending_entry_risk(next_state).pop(symbol, None)
            if not symbol_state.in_position:
                symbol_state.stage = "INVALIDATED"
            symbol_state.last_transition_reason = update.reason or "entry_terminal"
            events.append(
                _event(
                    code="ENTRY_TERMINAL",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status},
                )
            )
        elif role in {"TP", "EXIT"}:
            position = symbol_state.position
            if symbol_state.exit_order is not None and symbol_state.exit_order.oms_order_id == update.oms_order_id:
                symbol_state.exit_order = None
            pending_hard_exit = symbol_state.pending_hard_exit
            symbol_state.pending_hard_exit = False
            if role == "TP" and position is not None:
                position.pending_partial_stop = 0.0
                position.pending_partial_stop_buffer = 0.0
            if pending_hard_exit and position is not None and position.qty_open > 0:
                actions.append(
                    FlattenPosition(
                        symbol=symbol,
                        reason=symbol_state.last_transition_reason or "hard_exit",
                        side="SELL",
                        qty=position.qty_open,
                    )
                )
            elif (
                role == "EXIT"
                and position is not None
                and position.qty_open > 0
                and not position.stop_order_id
            ):
                actions.append(_submit_stop_action(symbol, position))
            events.append(
                _event(
                    code=f"{role}_TERMINAL",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status},
                )
            )
        elif role == "STOP":
            position = symbol_state.position
            expected = _expected_stop_cancels(next_state)
            was_expected = update.oms_order_id in expected
            if was_expected:
                expected.discard(update.oms_order_id)
                _set_expected_stop_cancels(next_state, expected)
            if position is not None and position.stop_order_id == update.oms_order_id:
                position.stop_order_id = ""
            if not was_expected and position is not None and position.qty_open > 0:
                actions.append(
                    FlattenPosition(
                        symbol=symbol,
                        reason="stop_terminal",
                        side="SELL",
                        qty=position.qty_open,
                    )
                )
            events.append(
                _event(
                    code="STOP_TERMINAL",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status, "expected_cancel": was_expected},
                )
            )

    if not events and update.decision_code:
        events.append(
            _event(
                code=update.decision_code,
                ts=event_ts,
                symbol=symbol,
                details=update.decision_details,
            )
        )
    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: IARICCoreState,
    fill: IARICFill,
) -> tuple[
    IARICCoreState,
    list[SubmitProtectiveStop | ReplaceProtectiveStop | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitProtectiveStop | ReplaceProtectiveStop | FlattenPosition] = []
    symbol_state, symbol = _resolve_symbol_state(next_state, fill.symbol, fill.oms_order_id)
    role = _resolve_role(symbol_state, fill.order_role, fill.oms_order_id)
    event_ts = fill.fill_time or datetime.now(timezone.utc)
    events: list[DecisionEvent] = []

    if not symbol or symbol_state is None:
        if fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=fill.symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if fill.oms_order_id:
        _order_index(next_state).pop(fill.oms_order_id, None)

    if role == "ENTRY":
        symbol_state.entry_order = None
        symbol_state.active_order_id = None
        _pending_entry_risk(next_state).pop(symbol, None)
        stop_price = max(symbol_state.stop_level, 0.01)
        position = build_position_from_fill(
            fill_price=fill.fill_price,
            fill_qty=fill.fill_qty,
            stop_price=stop_price,
            fill_time=event_ts,
            setup_tag=f"PB_{symbol_state.route_family}",
        )
        position.opportunity_event_id = str(
            getattr(symbol_state, "accepted_event_id", "")
        )
        position.reversion_anchor = float(
            getattr(symbol_state, "accepted_reversion_anchor", 0.0)
        )
        position.structural_stop_anchor = float(
            getattr(symbol_state, "accepted_stop_anchor", 0.0)
        )
        position.initial_remaining_room_atr = float(
            getattr(symbol_state, "accepted_remaining_room_atr", 0.0)
        )
        position.prospective_reward_risk = float(
            getattr(symbol_state, "accepted_prospective_reward_risk", 0.0)
        )
        position.entry_commission = fill.commission
        symbol_state.position = position
        symbol_state.in_position = True
        symbol_state.stage = "IN_POSITION"
        symbol_state.risk_per_share = max(fill.fill_price - stop_price, 0.01)
        actions.append(_submit_stop_action(symbol, position))
        events.append(
            _event(
                code="ENTRY_FILLED",
                ts=event_ts,
                symbol=symbol,
                details={
                    "qty": fill.fill_qty,
                    "price": fill.fill_price,
                    "route": symbol_state.route_family,
                    "stop_price": stop_price,
                },
            )
        )
        if not events and fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    position = symbol_state.position
    if position is None or fill.fill_qty <= 0:
        if not events and fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if symbol_state.exit_order is not None and symbol_state.exit_order.oms_order_id == fill.oms_order_id:
        symbol_state.exit_order = None

    position.max_favorable_price = max(position.max_favorable_price, fill.fill_price)
    position.max_adverse_price = min(position.max_adverse_price, fill.fill_price)
    exit_qty = min(fill.fill_qty, position.qty_open)
    if exit_qty <= 0:
        _update_last_decision(next_state, events)
        return next_state, actions, events

    position.exit_commission += fill.commission
    position.realized_pnl_usd += (fill.fill_price - position.entry_price) * exit_qty
    position.qty_open = max(0, position.qty_open - exit_qty)

    if role == "TP":
        position.partial_taken = True
        symbol_state.v2_partial_taken = True
        if position.pending_partial_stop > 0:
            position.current_stop = partial_remainder_stop_after_fill(
                current_stop=position.current_stop,
                requested_stop=position.pending_partial_stop,
                fill_price=fill.fill_price,
                execution_buffer=position.pending_partial_stop_buffer,
            )
            symbol_state.stop_level = position.current_stop
        position.pending_partial_stop = 0.0
        position.pending_partial_stop_buffer = 0.0
        if position.qty_open > 0 and position.stop_order_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=symbol,
                    target_order_id=position.stop_order_id,
                    side="SELL",
                    stop_price=position.current_stop,
                    qty=position.qty_open,
                    reason="partial_resize",
                )
            )
        if symbol_state.pending_hard_exit and position.qty_open > 0:
            symbol_state.pending_hard_exit = False
            actions.append(
                FlattenPosition(
                    symbol=symbol,
                    reason=symbol_state.last_transition_reason or "hard_exit",
                    side="SELL",
                    qty=position.qty_open,
                )
            )
        events.append(
            _event(
                code="PARTIAL_EXIT_FILLED" if position.qty_open > 0 else "EXIT_FILLED",
                ts=event_ts,
                symbol=symbol,
                details={
                    "qty": exit_qty,
                    "price": fill.fill_price,
                    "reason": fill.exit_type or "TP",
                },
            )
        )
    elif role == "EXIT":
        if position.qty_open > 0 and not position.stop_order_id:
            actions.append(_submit_stop_action(symbol, position))
            events.append(
                _event(
                    code="EXIT_PARTIALLY_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    details={"qty": exit_qty, "price": fill.fill_price},
                )
            )
        else:
            events.append(
                _event(
                    code="EXIT_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    details={
                        "qty": exit_qty,
                        "price": fill.fill_price,
                        "reason": symbol_state.last_transition_reason or fill.exit_type or "EXIT",
                    },
                )
            )
    elif role == "STOP":
        position.stop_order_id = ""
        if position.qty_open > 0:
            actions.append(
                FlattenPosition(
                    symbol=symbol,
                    reason="stop_unprotected",
                    side="SELL",
                    qty=position.qty_open,
                )
            )
        events.append(
            _event(
                code="STOP_FILLED",
                ts=event_ts,
                symbol=symbol,
                details={"qty": exit_qty, "price": fill.fill_price},
            )
        )

    if position.qty_open <= 0:
        reason = symbol_state.last_transition_reason or fill.exit_type or role
        if role == "STOP" or "STOP" in reason.upper():
            symbol_state.stopped_out_today = True
        symbol_state.position = None
        symbol_state.in_position = False
        symbol_state.stage = "INVALIDATED"
        symbol_state.exit_order = None
        symbol_state.pending_hard_exit = False

    if not events and fill.decision_code:
        events.append(
            _event(
                code=fill.decision_code,
                ts=event_ts,
                symbol=symbol,
                details=fill.decision_details,
            )
        )
    _update_last_decision(next_state, events)
    return next_state, actions, events


def _legacy_bar_events(payload: IARICBarInput) -> list[DecisionEvent]:
    if not payload.decision_code:
        return []
    return [
        _event(
            code=payload.decision_code,
            ts=payload.bar_ts or datetime.now(timezone.utc),
            symbol=payload.symbol,
            details=payload.decision_details,
        )
    ]


def _symbol_state(state: IARICCoreState, symbol: str) -> PBSymbolState | None:
    for symbol_state in state.symbols:
        if symbol_state.symbol == symbol:
            return symbol_state
    return None


def _resolve_symbol_state(
    state: IARICCoreState,
    symbol: str,
    oms_order_id: str,
) -> tuple[PBSymbolState | None, str]:
    if symbol:
        symbol_state = _symbol_state(state, symbol)
        if symbol_state is not None:
            return symbol_state, symbol
    if oms_order_id:
        for symbol_state in state.symbols:
            if symbol_state.entry_order and symbol_state.entry_order.oms_order_id == oms_order_id:
                return symbol_state, symbol_state.symbol
            if symbol_state.exit_order and symbol_state.exit_order.oms_order_id == oms_order_id:
                return symbol_state, symbol_state.symbol
            if (
                symbol_state.position is not None
                and symbol_state.position.stop_order_id == oms_order_id
            ):
                return symbol_state, symbol_state.symbol
    return None, ""


def _resolve_role(
    symbol_state: PBSymbolState | None,
    explicit_role: str,
    oms_order_id: str,
) -> str:
    if explicit_role and explicit_role != "UNKNOWN":
        return explicit_role
    if symbol_state is None:
        return "UNKNOWN"
    if symbol_state.entry_order and symbol_state.entry_order.oms_order_id == oms_order_id:
        return "ENTRY"
    if symbol_state.exit_order and symbol_state.exit_order.oms_order_id == oms_order_id:
        return str(symbol_state.exit_order.role or "EXIT").upper()
    if symbol_state.position and symbol_state.position.stop_order_id == oms_order_id:
        return "STOP"
    return "UNKNOWN"


def _event(
    *,
    code: str,
    ts: datetime,
    symbol: str,
    details: dict[str, Any],
) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts, symbol=symbol, timeframe="5m", details=dict(details))


def _meta(state: IARICCoreState) -> dict[str, Any]:
    if not isinstance(state.meta, dict):
        state.meta = {}
    return state.meta


def _order_index(state: IARICCoreState) -> dict[str, tuple[str, str]]:
    raw = dict(_meta(state).get("order_index", {}))
    normalized = {
        str(order_id): _coerce_order_index_entry(value)
        for order_id, value in raw.items()
    }
    _meta(state)["order_index"] = normalized
    return normalized


def _pending_entry_risk(state: IARICCoreState) -> dict[str, float]:
    raw = dict(_meta(state).get("pending_entry_risk", {}))
    normalized = {str(symbol): float(risk) for symbol, risk in raw.items()}
    _meta(state)["pending_entry_risk"] = normalized
    return normalized


def _expected_stop_cancels(state: IARICCoreState) -> set[str]:
    return {str(order_id) for order_id in _meta(state).get("expected_stop_cancels", [])}


def _set_expected_stop_cancels(state: IARICCoreState, order_ids: set[str]) -> None:
    _meta(state)["expected_stop_cancels"] = sorted(order_ids)


def _submit_stop_action(symbol: str, position: PositionState) -> SubmitProtectiveStop:
    return SubmitProtectiveStop(
        client_order_id=f"{symbol}-stop-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        symbol=symbol,
        side="SELL",
        qty=position.qty_open,
        stop_price=position.current_stop,
    )


def _update_last_decision(state: IARICCoreState, events: list[DecisionEvent]) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    meta = _meta(state)
    meta["last_decision_details"] = dict(latest.details)
    if latest.ts is not None:
        meta["last_bar_ts"] = meta.get("last_bar_ts", latest.ts)


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _coerce_order_index_entry(value: Any) -> tuple[str, str]:
    if isinstance(value, tuple) and len(value) == 2:
        return str(value[0]), str(value[1])
    if isinstance(value, list) and len(value) == 2:
        return str(value[0]), str(value[1])
    return "", ""
