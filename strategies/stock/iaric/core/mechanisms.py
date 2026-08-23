"""Mechanism-pure contracts for price/volume residual reversion.

News state is intentionally absent from admission and scoring.  The strategy
does not claim to identify why a price moved; it ranks residual dislocations
and uses only causally observable price/volume paths to distinguish likely
normalization from continued repricing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class InformationState(str, Enum):
    UNKNOWN = "unknown"
    VERIFIED_NO_EVENT = "verified_no_event"
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    ANALYST = "analyst"
    MATERIAL_COMPANY_NEWS = "material_company_news"
    CORPORATE_ACTION = "corporate_action"
    SECTOR_OR_MACRO_EVENT = "sector_or_macro_event"


@dataclass(frozen=True)
class MechanismSleeveSpec:
    name: str
    role: str
    score_weights: Mapping[str, float]
    hard_vetoes: tuple[str, ...]
    entry_mechanisms: tuple[str, ...]
    management_mechanisms: tuple[str, ...]
    diagnostic_legs: tuple[str, ...] = ("long",)
    default_max_capital_fraction: float = 0.20

    @property
    def score_components(self) -> tuple[str, ...]:
        return tuple(self.score_weights)


SLEEVE_SPECS: dict[str, MechanismSleeveSpec] = {
    "daily_residual_reversion": MechanismSleeveSpec(
        name="daily_residual_reversion",
        role="reversion",
        score_weights={
            "residual_extremeness": 0.14,
            "shock_freshness": 0.10,
            "price_rejection_recovery": 0.14,
            "volume_transition": 0.10,
            "volume_exhaustion_quality": 0.18,
            "regime_execution_quality": 0.10,
            "failed_continuation": 0.24,
        },
        hard_vetoes=(
            "missing_authoritative_input",
            "noncausal_universe_observation",
            "uncertified_price_basis",
            "incomplete_session",
            "suspected_unhandled_corporate_action",
            "nonpositive_remaining_room",
            "residual_continuation_not_failed",
            "cost_or_capacity_infeasible",
        ),
        entry_mechanisms=(
            "next_session_open",
            "causal_opening_schedule",
            "preexisting_resting_retrace",
            "completed_five_minute_residual_recovery",
        ),
        management_mechanisms=(
            "partial_fifty_percent_residual_normalization",
            "frozen_full_residual_anchor",
            "three_five_seven_session_half_life",
            "residual_persistence_structural_failure",
        ),
        diagnostic_legs=("long_loser", "short_winner", "dollar_neutral_spread"),
        default_max_capital_fraction=0.35,
    ),
    "intraday_residual_failed_continuation": MechanismSleeveSpec(
        name="intraday_residual_failed_continuation",
        role="reversion",
        score_weights={
            "intraday_residual_extremeness": 0.23,
            "failed_continuation": 0.23,
            "residual_recovery": 0.18,
            "volume_deceleration": 0.12,
            "residual_normalization_room": 0.14,
            "execution_regime_quality": 0.10,
        },
        hard_vetoes=(
            "missing_authoritative_input",
            "noncausal_universe_observation",
            "uncertified_price_basis",
            "incomplete_five_minute_bar",
            "residual_low_not_rejected",
            "nonpositive_remaining_room",
            "cost_or_capacity_infeasible",
        ),
        entry_mechanisms=(
            "next_five_minute_open",
            "preexisting_residual_retrace_limit",
            "completed_bar_residual_recovery",
        ),
        management_mechanisms=(
            "partial_intraday_residual_anchor",
            "full_intraday_residual_anchor",
            "preestimated_half_life_time_stop",
            "residual_extension_structural_failure",
            "session_flat_default",
        ),
        default_max_capital_fraction=0.20,
    ),
    "gap_residual_failed_continuation": MechanismSleeveSpec(
        name="gap_residual_failed_continuation",
        role="reversion",
        score_weights={
            "residual_gap_extremeness": 0.23,
            "failed_continuation": 0.23,
            "gap_recovery": 0.18,
            "volume_transition": 0.12,
            "residual_normalization_room": 0.14,
            "execution_regime_quality": 0.10,
        },
        hard_vetoes=(
            "missing_authoritative_input",
            "noncausal_universe_observation",
            "uncertified_price_basis",
            "incomplete_five_minute_bar",
            "gap_continuation_not_rejected",
            "nonpositive_remaining_room",
            "cost_or_capacity_infeasible",
        ),
        entry_mechanisms=(
            "next_five_minute_open",
            "preexisting_gap_retrace_limit",
            "completed_bar_gap_recovery",
        ),
        management_mechanisms=(
            "partial_previous_close_or_residual_anchor",
            "full_gap_residual_anchor",
            "preestimated_half_life_time_stop",
            "gap_extension_structural_failure",
            "session_flat_default",
        ),
        default_max_capital_fraction=0.20,
    ),
    "trend_pullback_tail_control": MechanismSleeveSpec(
        name="trend_pullback_tail_control",
        role="control",
        score_weights={
            "trend_quality": 0.20,
            "pullback_depth": 0.18,
            "reclaim_quality": 0.16,
            "relative_strength": 0.14,
            "participation_quality": 0.10,
            "remaining_room": 0.12,
            "regime_context": 0.10,
        },
        hard_vetoes=(
            "missing_authoritative_input",
            "not_point_in_time_member",
            "uncertified_price_basis",
            "nonpositive_remaining_room",
            "cost_or_capacity_infeasible",
        ),
        entry_mechanisms=("frozen_round3_causal_entry",),
        management_mechanisms=("frozen_round3_tail_management",),
        default_max_capital_fraction=0.45,
    ),
}


def validate_sleeve_specs() -> dict[str, Any]:
    failures: list[str] = []
    for name, spec in SLEEVE_SPECS.items():
        if name != spec.name:
            failures.append(f"sleeve key/name mismatch: {name}/{spec.name}")
        if not 1 <= len(spec.score_components) <= 7:
            failures.append(f"{name} must use between one and seven score components")
        if abs(sum(float(value) for value in spec.score_weights.values()) - 1.0) > 1e-12:
            failures.append(f"{name} score weights must sum to one")
        if not spec.hard_vetoes:
            failures.append(f"{name} has no hard veto contract")
        if not spec.entry_mechanisms or not spec.management_mechanisms:
            failures.append(f"{name} lacks typed entry or management mechanisms")
    reversion_component_sets = {
        tuple(spec.score_components)
        for spec in SLEEVE_SPECS.values()
        if spec.role == "reversion"
    }
    if len(reversion_component_sets) != sum(
        spec.role == "reversion" for spec in SLEEVE_SPECS.values()
    ):
        failures.append("reversion sleeves must not share one global component set")
    return {
        "passed": not failures,
        "failures": failures,
        "sleeve_count": len(SLEEVE_SPECS),
        "max_score_components": max(
            (len(spec.score_components) for spec in SLEEVE_SPECS.values()),
            default=0,
        ),
    }


def score_mechanism_components(
    sleeve: str,
    components: Mapping[str, float],
) -> float:
    """Score already transformed [0, 1] components for one sleeve only."""

    spec = SLEEVE_SPECS[sleeve]
    if set(components) != set(spec.score_components):
        raise ValueError(
            f"{sleeve} score requires exactly {spec.score_components!r}; "
            f"received {tuple(components)!r}"
        )
    return 100.0 * sum(
        float(spec.score_weights[name]) * min(max(float(components[name]), 0.0), 1.0)
        for name in spec.score_components
    )


def information_state_veto(sleeve: str, state: InformationState | str) -> str | None:
    """Compatibility hook: information state never gates price/volume sleeves.

    Corporate actions are handled by the certified price-basis contract, not
    by an unreliable news-state proxy.  Keeping this no-op avoids divergent
    live and replay behaviour while older adapters are retired.
    """

    if sleeve not in SLEEVE_SPECS:
        raise KeyError(sleeve)
    InformationState(str(state)) if not isinstance(state, InformationState) else state
    return None


_VALIDATION = validate_sleeve_specs()
if not _VALIDATION["passed"]:
    raise RuntimeError("; ".join(_VALIDATION["failures"]))
