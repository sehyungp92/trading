"""Prospective shared contracts for additive IARIC reversion lanes.

This module is intentionally dependency-light so live and replay adapters can
consume the same issuer, lane, score-profile, management-profile and event-ID
rules.  All new behaviour is opt-in at the settings layer; importing this
module keeps lane experiments explicit; issuer event deduplication is a
portfolio-identity invariant and is enabled by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCORE_COMPONENTS = (
    "dislocation",
    "reclaim",
    "close_quality",
    "relative_volume",
    "residual_dislocation",
    "prior_down_sequence",
    "reversion_room",
)

# These are economic hypotheses fixed before replay.  Profiles reuse exactly
# the registered seven features; they do not add optimizer-only information.
SCORE_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "dislocation": 0.17,
        "reclaim": 0.18,
        "close_quality": 0.13,
        "relative_volume": 0.10,
        "residual_dislocation": 0.14,
        "prior_down_sequence": 0.10,
        "reversion_room": 0.18,
    },
    "shock_exhaustion": {
        "dislocation": 0.22,
        "reclaim": 0.18,
        "close_quality": 0.15,
        "relative_volume": 0.20,
        "residual_dislocation": 0.10,
        "prior_down_sequence": 0.05,
        "reversion_room": 0.10,
    },
    "level_reclaim": {
        "dislocation": 0.18,
        "reclaim": 0.25,
        "close_quality": 0.20,
        "relative_volume": 0.08,
        "residual_dislocation": 0.07,
        "prior_down_sequence": 0.07,
        "reversion_room": 0.15,
    },
    "relative_shock": {
        "dislocation": 0.10,
        "reclaim": 0.15,
        "close_quality": 0.15,
        "relative_volume": 0.10,
        "residual_dislocation": 0.30,
        "prior_down_sequence": 0.05,
        "reversion_room": 0.15,
    },
    "trend_pullback": {
        "dislocation": 0.20,
        "reclaim": 0.25,
        "close_quality": 0.20,
        "relative_volume": 0.10,
        "residual_dislocation": 0.05,
        "prior_down_sequence": 0.10,
        "reversion_room": 0.10,
    },
}

for _profile_name, _weights in SCORE_PROFILES.items():
    if tuple(_weights) != SCORE_COMPONENTS:
        raise RuntimeError(f"{_profile_name} must use exactly the seven registered score components")
    if abs(sum(_weights.values()) - 1.0) > 1e-12:
        raise RuntimeError(f"{_profile_name} score weights must sum to one")


FAMILY_ARCHETYPES = {
    "GAP_EXHAUSTION_RECLAIM": "SHOCK_EXHAUSTION",
    "GAP_FILL_RECLAIM": "SHOCK_EXHAUSTION",
    "GAP_PARTIAL_RECLAIM": "SHOCK_EXHAUSTION",
    "OPENING_FLUSH_RECLAIM": "SHOCK_EXHAUSTION",
    "VOLUME_CLIMAX_RECLAIM": "SHOCK_EXHAUSTION",
    "PRIOR_DAY_LOW_RECLAIM": "LEVEL_RECLAIM",
    "OPENING_RANGE_LOW_RECLAIM": "LEVEL_RECLAIM",
    "FAILED_BREAKDOWN_RECLAIM": "LEVEL_RECLAIM",
    "VWAP_DEVIATION_RECLAIM": "LEVEL_RECLAIM",
    "MARKET_SECTOR_RESIDUAL_RECLAIM": "RELATIVE_SHOCK",
    "MULTIDAY_HIGHER_LOW_RECLAIM": "TREND_PULLBACK",
    "UPTREND_PULLBACK_RECLAIM": "TREND_PULLBACK",
}

# A second signal is permitted only for families whose causal condition can
# genuinely reset intraday.  Session-opening and gap definitions remain
# one-shot even if a malformed configuration asks for more events.
REARMABLE_FAMILIES = frozenset({
    "PRIOR_DAY_LOW_RECLAIM",
    "OPENING_RANGE_LOW_RECLAIM",
    "FAILED_BREAKDOWN_RECLAIM",
    "VWAP_DEVIATION_RECLAIM",
    "MARKET_SECTOR_RESIDUAL_RECLAIM",
    "VOLUME_CLIMAX_RECLAIM",
})

CAPPED_ADDITIVE_LANES = frozenset({
    "RESCUE_EVENT",
    "APERTURE_SHOCK_EXHAUSTION",
    "APERTURE_LEVEL_RECLAIM",
    "APERTURE_RELATIVE_SHOCK",
    "APERTURE_TREND_PULLBACK",
})


# Profiles change route management as a coherent mechanism.  Missing suffixes
# fall through to the historical route setting rather than silently becoming
# zero.  The baseline profile is deliberately empty.
MANAGEMENT_PROFILES: dict[str, dict[str, float]] = {
    "baseline": {},
    "tail_capture": {
        "max_hold_days": 5.0,
        "rsi_exit": 62.0,
        "quick_exit_loss_r": 0.50,
        "stale_exit_bars": 0.0,
    },
    "fast_snapback": {
        "max_hold_days": 2.0,
        "rsi_exit": 58.0,
        "quick_exit_loss_r": 0.35,
        "stale_exit_bars": 6.0,
    },
}


ISSUER_ALIASES = {
    "GOOG": "ALPHABET",
    "GOOGL": "ALPHABET",
    "BRK.A": "BERKSHIRE_HATHAWAY",
    "BRK.B": "BERKSHIRE_HATHAWAY",
    "FOX": "FOX_CORPORATION",
    "FOXA": "FOX_CORPORATION",
    "NWS": "NEWS_CORPORATION",
    "NWSA": "NEWS_CORPORATION",
}


def _tokens(raw: Any) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
    return [str(value).strip() for value in values if str(value).strip()]


def parse_mapping(
    raw: Any,
    *,
    setting: str,
    allowed_keys: Iterable[str] | None = None,
    allowed_values: Iterable[str] | None = None,
) -> dict[str, str]:
    """Parse a deterministic ``KEY:value`` mapping and fail closed."""

    keys = {str(value).upper() for value in allowed_keys} if allowed_keys is not None else None
    values = {str(value).lower() for value in allowed_values} if allowed_values is not None else None
    result: dict[str, str] = {}
    for token in _tokens(raw):
        separator = ":" if ":" in token else "=" if "=" in token else ""
        if not separator:
            raise ValueError(f"{setting} entries must use KEY:value")
        key, value = token.split(separator, 1)
        key = key.strip().upper()
        value = value.strip().lower()
        if keys is not None and key not in keys:
            raise ValueError(f"{setting} contains unknown key {key!r}")
        if values is not None and value not in values:
            raise ValueError(f"{setting} contains unknown value {value!r}")
        if key in result:
            raise ValueError(f"{setting} contains duplicate key {key!r}")
        result[key] = value
    return result


def normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper().replace("-", ".")
    if value.startswith("BRK "):
        value = value.replace("BRK ", "BRK.", 1)
    return value


def issuer_key(symbol: str, overrides: Any = "") -> str:
    normalized = normalize_symbol(symbol)
    custom = parse_mapping(overrides, setting="pb_issuer_aliases")
    return custom.get(normalized, ISSUER_ALIASES.get(normalized, normalized))


def is_aperture_only_item(item: Any) -> bool:
    """Return whether an item belongs only to the broad aperture sleeve.

    Dual-eligible incumbent items deliberately carry ``aperture_candidate`` so
    the shared completed-bar route can observe them. They must nevertheless
    stay in the incumbent ranking/capacity cohort. Pure aperture items are
    identified by their typed allocation tier, not replay-only metadata.
    """

    return bool(getattr(item, "aperture_candidate", False)) and str(
        getattr(item, "trigger_tier", "") or ""
    ).strip().upper() == "APERTURE"


@dataclass(frozen=True, slots=True)
class IssuerExposureDecision:
    allowed: bool
    issuer: str
    reason: str = ""
    active_count: int = 0
    daily_entry_count: int = 0


@dataclass(frozen=True, slots=True)
class IssuerEntryCandidate:
    """Minimal same-batch signal identity used by both execution adapters."""

    symbol: str
    route_family: str
    score: float
    stable_rank: int = 0


@dataclass(frozen=True, slots=True)
class IssuerBatchArbitration:
    selected_symbols: frozenset[str]
    rejected_by_winner: Mapping[str, str]


def issuer_batch_arbitration(
    settings: Any,
    candidates: Iterable[IssuerEntryCandidate],
) -> IssuerBatchArbitration:
    """Deduplicate simultaneous share-class/economic-lane signals.

    This is signal arbitration rather than a blunt portfolio issuer cap.  It
    keeps the best causal score from a duplicated issuer event across routes
    while allowing a later, independently reset episode to compete normally.
    """

    rows = list(candidates)
    if not bool(getattr(settings, "pb_issuer_event_dedupe_enabled", True)):
        return IssuerBatchArbitration(
            frozenset(row.symbol for row in rows),
            {},
        )
    aliases = getattr(settings, "pb_issuer_aliases", "")
    grouped: dict[str, list[IssuerEntryCandidate]] = {}
    for row in rows:
        group = issuer_key(row.symbol, aliases)
        grouped.setdefault(group, []).append(row)

    selected: set[str] = set()
    rejected: dict[str, str] = {}
    for group_rows in grouped.values():
        winner = min(
            group_rows,
            key=lambda row: (-float(row.score), int(row.stable_rank), normalize_symbol(row.symbol)),
        )
        selected.add(winner.symbol)
        for row in group_rows:
            if row.symbol != winner.symbol:
                rejected[row.symbol] = winner.symbol
    return IssuerBatchArbitration(frozenset(selected), rejected)


def issuer_exposure_decision(
    settings: Any,
    symbol: str,
    *,
    active_symbols: Iterable[str] = (),
    daily_entry_symbols: Iterable[str] = (),
) -> IssuerExposureDecision:
    """Apply prospective issuer caps to active/pending and same-day entries."""

    aliases = getattr(settings, "pb_issuer_aliases", "")
    issuer = issuer_key(symbol, aliases)
    active_count = sum(issuer_key(value, aliases) == issuer for value in active_symbols)
    daily_count = sum(issuer_key(value, aliases) == issuer for value in daily_entry_symbols)
    position_cap = max(int(getattr(settings, "pb_issuer_position_cap", 0) or 0), 0)
    daily_cap = max(int(getattr(settings, "pb_issuer_daily_entry_cap", 0) or 0), 0)
    if position_cap and active_count >= position_cap:
        return IssuerExposureDecision(False, issuer, "issuer_position_cap", active_count, daily_count)
    if daily_cap and daily_count >= daily_cap:
        return IssuerExposureDecision(False, issuer, "issuer_daily_entry_cap", active_count, daily_count)
    return IssuerExposureDecision(True, issuer, "", active_count, daily_count)


def aperture_family_from_route(route_family: str) -> str:
    route = str(route_family or "").strip().upper()
    if not route.startswith("APERTURE_"):
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
    return body if body in FAMILY_ARCHETYPES else ""


def lane_id_for_route(route_family: str, *, rescue_candidate: bool = False) -> str:
    route = str(route_family or "").strip().upper()
    if rescue_candidate or route == "OPEN_SCORED_RESCUE_ENTRY":
        return "RESCUE_EVENT"
    family = aperture_family_from_route(route)
    if family:
        return f"APERTURE_{FAMILY_ARCHETYPES[family]}"
    if route.startswith("OPEN_SCORED"):
        return "OPEN_SCORED_ANCHOR"
    return route or "UNCLASSIFIED"


def event_id(family: str, signal_bar_index: int) -> str:
    return f"{str(family).strip().upper()}@{int(signal_bar_index)}"


def family_event_caps(settings: Any, enabled_families: Iterable[str]) -> dict[str, int]:
    """Return validated one- or two-episode caps for enabled families."""

    enabled = {str(value).strip().upper() for value in enabled_families}
    mapping = parse_mapping(
        getattr(settings, "pb_aperture_family_max_events", ""),
        setting="pb_aperture_family_max_events",
        allowed_keys=FAMILY_ARCHETYPES,
        allowed_values={"1", "2"},
    )
    caps: dict[str, int] = {}
    for family in enabled:
        requested = int(mapping.get(family, "1"))
        caps[family] = requested if family in REARMABLE_FAMILIES else 1
    return caps


def rearm_cooldown_bars(settings: Any) -> int:
    value = int(getattr(settings, "pb_aperture_rearm_cooldown_bars", 12) or 12)
    if value not in {6, 12, 24}:
        raise ValueError("pb_aperture_rearm_cooldown_bars must be 6, 12, or 24")
    return value


def consumption_token(
    family: str,
    signal_bar_index: int,
    family_cap: int,
    episode_start_bar_index: int | None = None,
) -> str:
    """Consume one economic episode, not every improving observation of it."""

    episode_index = (
        int(signal_bar_index)
        if episode_start_bar_index is None
        else int(episode_start_bar_index)
    )
    return event_id(family, episode_index) if int(family_cap) > 1 else str(family).upper()


def event_is_consumed(
    family: str,
    signal_bar_index: int,
    family_cap: int,
    consumed: Iterable[str],
    episode_start_bar_index: int | None = None,
) -> bool:
    return consumption_token(
        family,
        signal_bar_index,
        family_cap,
        episode_start_bar_index,
    ) in set(consumed)


def score_profile_name(settings: Any, family: str) -> str:
    mapping = parse_mapping(
        getattr(settings, "pb_aperture_family_score_profiles", ""),
        setting="pb_aperture_family_score_profiles",
        allowed_keys=FAMILY_ARCHETYPES,
        allowed_values=SCORE_PROFILES,
    )
    return mapping.get(str(family).strip().upper(), "balanced")


def score_from_components(components: Mapping[str, float], profile: str) -> float:
    if set(components) != set(SCORE_COMPONENTS):
        raise ValueError("reversion event score must contain exactly seven registered components")
    if profile not in SCORE_PROFILES:
        raise ValueError(f"unknown reversion score profile {profile!r}")
    return 100.0 * sum(
        SCORE_PROFILES[profile][name] * min(max(float(components[name]), 0.0), 1.0)
        for name in SCORE_COMPONENTS
    )


def management_profile_name(settings: Any, route_family: str) -> str:
    family = aperture_family_from_route(route_family)
    if not family:
        return "baseline"
    mapping = parse_mapping(
        getattr(settings, "pb_aperture_family_management_profiles", ""),
        setting="pb_aperture_family_management_profiles",
        allowed_keys=FAMILY_ARCHETYPES,
        allowed_values=MANAGEMENT_PROFILES,
    )
    return mapping.get(family, "baseline")


def anchor_exit_enabled(settings: Any, route_family: str) -> bool:
    """Return whether this family explicitly owns a full normalization exit.

    The global setting is only a master switch.  A family must additionally
    use the pre-registered fast-snapback profile; this prevents an architecture
    migration from converting incumbent trend-pullback tail trades into short
    normalization trades merely because they now carry an anchor.
    """

    if not bool(getattr(settings, "pb_aperture_anchor_exit_enabled", False)):
        return False
    return management_profile_name(settings, route_family) == "fast_snapback"


def management_override(settings: Any, route_family: str, suffix: str) -> float | None:
    profile = management_profile_name(settings, route_family)
    return MANAGEMENT_PROFILES[profile].get(str(suffix))


def lane_counter_key(lane_id: str, stage: str) -> str:
    lane = "_".join(str(lane_id).strip().lower().split())
    state = "_".join(str(stage).strip().lower().split())
    return f"lane__{lane}__{state}"


def lane_daily_cap(settings: Any, lane_id: str) -> int | None:
    """Return the independent cap for one additive economic lane."""

    lane = str(lane_id or "").strip().upper()
    mapping = parse_mapping(
        getattr(settings, "pb_reversion_lane_daily_caps", ""),
        setting="pb_reversion_lane_daily_caps",
        allowed_keys=CAPPED_ADDITIVE_LANES,
        allowed_values={"1", "2", "4"},
    )
    raw = mapping.get(lane)
    return int(raw) if raw is not None else None
