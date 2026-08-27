"""Shared decision primitives for the IARIC daily residual sleeve.

The module is deliberately independent of pandas, a broker and the research
runner.  Live and replay adapters must supply the same completed-session
features and consume the same entry and management decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import Counter
from datetime import date, datetime
from math import floor
from typing import Any, Iterable, Literal, Mapping, Sequence

from strategies.core.actions import (
    CancelAction,
    NeutralAction,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitMarketExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent

from .mechanisms import SLEEVE_SPECS, score_mechanism_components
from .lanes import issuer_key


DAILY_RESIDUAL_SLEEVE = "daily_residual_reversion"
DAILY_RESIDUAL_SCORE_WEIGHTS = dict(
    SLEEVE_SPECS[DAILY_RESIDUAL_SLEEVE].score_weights
)


@dataclass(frozen=True, slots=True)
class DailyResidualFeatures:
    """Causal normalized inputs; any executable score uses at most seven."""

    residual_extremeness: float
    shock_freshness: float
    price_rejection_recovery: float
    volume_transition: float
    volume_exhaustion_quality: float
    residual_normalization_room: float
    regime_execution_quality: float
    failed_continuation: float = 0.0

    def as_mapping(self) -> dict[str, float]:
        return {
            "residual_extremeness": self.residual_extremeness,
            "shock_freshness": self.shock_freshness,
            "price_rejection_recovery": self.price_rejection_recovery,
            "volume_transition": self.volume_transition,
            "volume_exhaustion_quality": self.volume_exhaustion_quality,
            "residual_normalization_room": self.residual_normalization_room,
            "regime_execution_quality": self.regime_execution_quality,
            "failed_continuation": self.failed_continuation,
        }


@dataclass(frozen=True, slots=True)
class DailyResidualEntryDecision:
    eligible: bool
    score: float
    reason: str
    side: Literal["long", "short"]
    entry_clock: str = "next_session_open"
    # Admission and capacity ranking normally share one score.  A frozen
    # two-stage selector may admit on a broader mechanism score and rank the
    # admitted set on a more stable subset; preserve both for parity audits.
    admission_score: float | None = None


@dataclass(frozen=True, slots=True)
class DailyResidualOpportunity:
    """Adapter-neutral input to the shared cross-sectional selector."""

    symbol: str
    issuer: str
    sector: str
    side: Literal["long", "short"]
    residual_z: float
    remaining_room_r: float
    features: DailyResidualFeatures
    failed_continuation_r: float = 0.0
    sector_return_5d: float = 0.0
    regime_feasible: bool = True
    cost_feasible: bool = True
    data_ready: bool = True


@dataclass(frozen=True, slots=True)
class RankedDailyResidualOpportunity:
    opportunity: DailyResidualOpportunity
    decision: DailyResidualEntryDecision
    rank: int
    sector_overflow: bool = False


@dataclass(frozen=True, slots=True)
class DailyResidualReplacementIncumbent:
    """Causal completed-session state used for capacity-neutral replacement."""

    symbol: str
    issuer: str
    sector: str
    entry_score: float
    held_sessions: int
    normalization_fraction: float
    unrealized_r: float


@dataclass(frozen=True, slots=True)
class DailyResidualReplacementDecision:
    incumbent_symbol: str
    candidate_symbol: str
    blocker_kind: Literal["sector_capacity", "portfolio_capacity"]
    incumbent_entry_score: float
    candidate_score: float
    score_margin: float
    incumbent_held_sessions: int
    incumbent_normalization_fraction: float
    incumbent_unrealized_r: float


@dataclass(frozen=True, slots=True)
class ResidualManagementState:
    """Formation-frozen state; the residual anchor never moves after entry."""

    initial_dislocation_r: float
    cumulative_normalization_r: float = 0.0
    peak_normalization_r: float = 0.0
    held_sessions: int = 0
    partial_taken: bool = False


@dataclass(frozen=True, slots=True)
class ResidualManagementPolicy:
    partial_normalization_fraction: float = 0.50
    full_normalization_fraction: float = 1.00
    structural_failure_extension_fraction: float = 0.50
    profit_retention_activation_fraction: float = 99.0
    profit_retention_giveback_fraction: float = 99.0
    maximum_holding_sessions: int = 7
    partial_exit_fraction: float = 0.50


@dataclass(frozen=True, slots=True)
class ResidualManagementDecision:
    action: Literal["hold", "partial_exit", "full_exit"]
    reason: str
    exit_fraction: float
    normalization_fraction: float
    next_state: ResidualManagementState


@dataclass(slots=True)
class DailyResidualExecutionPosition:
    symbol: str
    issuer: str
    sector: str
    qty_entry: int
    qty_open: int
    entry_price: float
    entry_time: datetime
    initial_risk_per_share: float
    catastrophic_stop_distance: float
    residual_factor_model: str
    residual_formation_sessions: int
    residual_volatility: float
    management: ResidualManagementState
    residual_lane_id: str = ""
    residual_model_contract_version: str = ""
    residual_model_intercept: float = 0.0
    residual_factor_names: tuple[str, ...] = ()
    residual_factor_betas: tuple[float, ...] = ()
    residual_peer_symbols: tuple[str, ...] = ()
    residual_model_estimation_session: date | None = None
    last_processed_session: date | None = None
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    realized_pnl_usd: float = 0.0
    entry_score: float = 0.0
    trade_id: str = ""


@dataclass(slots=True)
class DailyResidualSymbolState:
    symbol: str
    issuer: str
    sector: str
    exchange: str
    primary_exchange: str
    currency: str
    tick_size: float
    point_value: float
    sleeve_id: str = DAILY_RESIDUAL_SLEEVE
    factor_model: str = ""
    formation_sessions: int = 0
    residual_z: float = 0.0
    residual_volatility: float = 0.0
    initial_dislocation_r: float = 0.0
    failed_continuation_r: float = 0.0
    sector_return_5d: float = 0.0
    residual_lane_id: str = ""
    residual_model_contract_version: str = ""
    residual_model_intercept: float = 0.0
    residual_factor_names: tuple[str, ...] = ()
    residual_factor_betas: tuple[float, ...] = ()
    residual_peer_symbols: tuple[str, ...] = ()
    residual_model_estimation_session: date | None = None
    last_processed_session: date | None = None
    planned_entry_price: float = 0.0
    planned_stop_price: float = 0.0
    planned_initial_risk_per_share: float = 0.0
    planned_qty: int = 0
    entry_score: float = 0.0
    pending_client_order_id: str = ""
    pending_role: str = ""
    pending_remaining_qty: int = 0
    pending_management_action: str = "hold"
    pending_management_reason: str = ""
    pending_exit_fraction: float = 0.0
    protective_stop_client_order_id: str = ""
    protective_stop_price: float = 0.0
    protective_stop_qty: int = 0
    entry_skipped_reason: str = ""
    position: DailyResidualExecutionPosition | None = None


@dataclass(slots=True)
class DailyResidualExecutionState:
    trade_date: date
    nav: float
    symbols: dict[str, DailyResidualSymbolState] = field(default_factory=dict)
    session_orders_planned: bool = False
    entry_orders_staged: bool = False
    entry_orders_planned: bool = False
    exit_orders_planned: bool = False
    last_decision_code: str = "IDLE"
    schema_version: str = "iaric_daily_residual_execution_v2"


@dataclass(frozen=True, slots=True)
class DailyResidualFill:
    client_order_id: str
    symbol: str
    role: Literal["ENTRY", "PARTIAL_EXIT", "EXIT", "STOP"]
    qty: int
    price: float
    ts: datetime
    commission: float = 0.0


def _unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def score_daily_residual_features(
    features: DailyResidualFeatures | Mapping[str, float],
    *,
    score_weights: Mapping[str, float] | None = None,
) -> float:
    """Return a frozen component-subset score on a 0-100 scale.

    The mechanism registry supplies the seven-component ceiling. A phased
    research run may freeze a smaller empirically qualified subset, but it may
    not introduce unregistered components or exceed that ceiling.
    """

    components = (
        features.as_mapping()
        if isinstance(features, DailyResidualFeatures)
        else {str(name): _unit(value) for name, value in features.items()}
    )
    if score_weights is None:
        registered = SLEEVE_SPECS[DAILY_RESIDUAL_SLEEVE].score_components
        return score_mechanism_components(
            DAILY_RESIDUAL_SLEEVE,
            {name: components[name] for name in registered},
        )
    weights = {str(name): float(value) for name, value in score_weights.items()}
    if not 1 <= len(weights) <= 7:
        raise ValueError("qualified daily residual score requires one to seven components")
    if not set(weights).issubset(DAILY_RESIDUAL_SCORE_WEIGHTS):
        raise ValueError("qualified daily residual score contains an unregistered component")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("qualified daily residual score weights must be positive")
    return 100.0 * sum(
        weights[name] / total * _unit(components[name]) for name in weights
    )


def decide_daily_residual_entry(
    features: DailyResidualFeatures | Mapping[str, float],
    *,
    side: Literal["long", "short"],
    residual_z: float,
    remaining_room_r: float,
    failed_continuation_r: float = 0.0,
    regime_feasible: bool = True,
    cost_feasible: bool,
    data_ready: bool,
    minimum_residual_z: float = 1.0,
    minimum_score: float = 0.0,
    minimum_failed_continuation_r: float = 0.0,
    score_weights: Mapping[str, float] | None = None,
) -> DailyResidualEntryDecision:
    """Apply economic hard gates before cross-sectional capacity ranking.

    A zero score floor preserves rank-only selection.  A positive, frozen
    floor rejects weak events before capacity ranking; this is useful when the
    diagnostics show that low-quality signals are not merely lower ranked but
    have negative expectancy.
    """

    score = score_daily_residual_features(features, score_weights=score_weights)
    directionally_extreme = (
        residual_z <= -abs(float(minimum_residual_z))
        if side == "long"
        else residual_z >= abs(float(minimum_residual_z))
    )
    if not data_ready:
        reason = "missing_authoritative_input"
    elif not regime_feasible:
        reason = "adverse_sector_regime"
    elif not directionally_extreme:
        reason = "residual_not_extreme"
    elif remaining_room_r <= 0.0:
        reason = "nonpositive_remaining_room"
    elif failed_continuation_r < float(minimum_failed_continuation_r):
        reason = "residual_continuation_not_failed"
    elif score < float(minimum_score):
        reason = "score_below_floor"
    elif not cost_feasible:
        reason = "cost_or_capacity_infeasible"
    else:
        return DailyResidualEntryDecision(
            True,
            score,
            "eligible_top_rank",
            side,
            admission_score=score,
        )
    return DailyResidualEntryDecision(
        False,
        score,
        reason,
        side,
        admission_score=score,
    )


def rank_daily_residual_opportunities(
    opportunities: Iterable[DailyResidualOpportunity],
    *,
    max_positions: int = 10,
    max_positions_per_sector: int = 2,
    sector_overflow_slots: int = 0,
    sector_overflow_minimum_score: float = 50.0,
    sector_overflow_minimum_z: float = 1.0,
    active_issuers: Sequence[str] = (),
    active_sectors: Sequence[str] = (),
    blocked_issuers: Sequence[str] = (),
    minimum_residual_z: float = 1.0,
    minimum_score: float = 0.0,
    minimum_failed_continuation_r: float = 0.0,
    score_weights: Mapping[str, float] | None = None,
    ranking_score_weights: Mapping[str, float] | None = None,
    required_symbols: Sequence[str] = (),
) -> tuple[RankedDailyResidualOpportunity, ...]:
    """Apply shared economic gates and deterministic issuer/sector caps.

    ``required_symbols`` reserves capacity for a candidate that has already
    passed the causal capacity-neutral replacement decision.  It is deliberately
    empty for ordinary ranking.  Without the reservation, releasing an
    incumbent can expose a different opportunity that consumes the freed slot,
    leaving the paired replacement exit without its promised entry.
    """

    if max_positions < 1:
        raise ValueError("max_positions must be positive")
    if max_positions_per_sector < 1:
        raise ValueError("max_positions_per_sector must be positive")
    if not 0 <= int(sector_overflow_slots) <= 2:
        raise ValueError("sector_overflow_slots must be in [0, 2]")
    if score_weights is not None and ranking_score_weights is not None:
        if len(set(score_weights) | set(ranking_score_weights)) > 7:
            raise ValueError(
                "daily residual admission/ranking score union exceeds seven components"
            )
    active_issuer_set = {issuer_key(str(value)) for value in active_issuers}
    # ``blocked_issuers`` do not consume portfolio capacity.  This is used for
    # positions whose causal full-exit order will execute before new entries at
    # the same open: their slot is available, but immediately selling and
    # rebuying the same issuer would be churn rather than a new opportunity.
    used_issuers = active_issuer_set | {
        issuer_key(str(value)) for value in blocked_issuers
    }
    sector_counts = Counter(str(value) for value in active_sectors)
    remaining = max(0, int(max_positions) - len(active_issuer_set))
    eligible: list[tuple[DailyResidualOpportunity, DailyResidualEntryDecision]] = []
    for opportunity in opportunities:
        decision = decide_daily_residual_entry(
            opportunity.features,
            side=opportunity.side,
            residual_z=opportunity.residual_z,
            remaining_room_r=opportunity.remaining_room_r,
            cost_feasible=opportunity.cost_feasible,
            data_ready=opportunity.data_ready,
            minimum_residual_z=minimum_residual_z,
            minimum_score=minimum_score,
            minimum_failed_continuation_r=minimum_failed_continuation_r,
            failed_continuation_r=opportunity.failed_continuation_r,
            regime_feasible=opportunity.regime_feasible,
            score_weights=score_weights,
        )
        if decision.eligible:
            ranking_score = score_daily_residual_features(
                opportunity.features,
                score_weights=(ranking_score_weights or score_weights),
            )
            decision = replace(
                decision,
                score=ranking_score,
                admission_score=decision.score,
            )
            eligible.append((opportunity, decision))
    eligible.sort(
        key=lambda item: (
            -float(item[1].score),
            -abs(float(item[0].residual_z)),
            str(item[0].symbol),
        )
    )
    required = {str(symbol) for symbol in required_symbols}
    if len(required) != len(tuple(required_symbols)):
        raise ValueError("required_symbols must be unique")
    eligible_symbols = {item[0].symbol for item in eligible}
    missing_required = sorted(required - eligible_symbols)
    if missing_required:
        raise ValueError(
            "required capacity-neutral candidates are not economically eligible: "
            f"{missing_required}"
        )
    if required:
        eligible.sort(key=lambda item: item[0].symbol not in required)

    selected: list[RankedDailyResidualOpportunity] = []
    for opportunity, decision in eligible:
        if remaining <= 0:
            break
        opportunity_issuer = issuer_key(
            opportunity.issuer or opportunity.symbol
        )
        if opportunity_issuer in used_issuers:
            continue
        sector_count = sector_counts[opportunity.sector]
        sector_overflow = sector_count >= int(max_positions_per_sector)
        if sector_overflow:
            if sector_count >= int(max_positions_per_sector) + int(
                sector_overflow_slots
            ):
                continue
            admission_score = float(
                decision.admission_score
                if decision.admission_score is not None
                else decision.score
            )
            if admission_score < float(sector_overflow_minimum_score):
                continue
            if abs(float(opportunity.residual_z)) < float(
                sector_overflow_minimum_z
            ):
                continue
        used_issuers.add(opportunity_issuer)
        sector_counts[opportunity.sector] += 1
        selected.append(
            RankedDailyResidualOpportunity(
                opportunity,
                decision,
                len(selected) + 1,
                sector_overflow=sector_overflow,
            )
        )
        remaining -= 1
    return tuple(selected)


def choose_capacity_neutral_replacements(
    opportunities: Iterable[DailyResidualOpportunity],
    incumbents: Sequence[DailyResidualReplacementIncumbent],
    selected: Sequence[RankedDailyResidualOpportunity],
    *,
    mode: str,
    loss_only: bool,
    minimum_held_sessions: int,
    maximum_normalization_fraction: float,
    minimum_score_margin: float,
    maximum_replacements: int,
    max_positions: int,
    max_positions_per_sector: int,
    minimum_residual_z: float,
    minimum_score: float,
    minimum_failed_continuation_r: float,
    score_weights: Mapping[str, float] | None,
    ranking_score_weights: Mapping[str, float] | None,
    blocked_issuers: Sequence[str] = (),
) -> tuple[DailyResidualReplacementDecision, ...]:
    """Choose at most one causal, capacity-neutral incumbent replacement.

    This is deliberately not an overflow path.  The incoming opportunity must
    already pass the frozen economic gates, and it may enter only by releasing
    a stale incumbent at the same following open.  Sector replacements keep
    the ordinary sector count unchanged; portfolio replacements must reduce
    concentration.  No future incumbent or candidate outcome is consulted.
    """

    supported_modes = {
        "disabled",
        "sector_stale",
        "portfolio_diversifying",
        "combined",
    }
    if mode not in supported_modes:
        raise ValueError(f"unsupported capacity-neutral replacement mode: {mode}")
    if mode == "disabled" or maximum_replacements <= 0:
        return ()
    if maximum_replacements != 1:
        raise ValueError("capacity-neutral replacement is bounded to one per session")

    active = list(incumbents)
    selected_issuers = {
        issuer_key(row.opportunity.issuer or row.opportunity.symbol)
        for row in selected
    }
    selected_symbols = {
        row.opportunity.symbol for row in selected
    }
    active_issuers = {
        issuer_key(row.issuer or row.symbol) for row in active
    }
    blocked_issuer_set = {
        issuer_key(str(value)) for value in blocked_issuers
    }
    sector_counts = Counter(row.sector for row in active)
    sector_counts.update(row.opportunity.sector for row in selected)
    portfolio_count = len(active) + len(selected)

    eligible: list[
        tuple[DailyResidualOpportunity, DailyResidualEntryDecision]
    ] = []
    for opportunity in opportunities:
        opportunity_issuer = issuer_key(
            opportunity.issuer or opportunity.symbol
        )
        if (
            opportunity.symbol in selected_symbols
            or opportunity_issuer in selected_issuers
            or opportunity_issuer in active_issuers
            or opportunity_issuer in blocked_issuer_set
        ):
            continue
        decision = decide_daily_residual_entry(
            opportunity.features,
            side=opportunity.side,
            residual_z=opportunity.residual_z,
            remaining_room_r=opportunity.remaining_room_r,
            failed_continuation_r=opportunity.failed_continuation_r,
            regime_feasible=opportunity.regime_feasible,
            cost_feasible=opportunity.cost_feasible,
            data_ready=opportunity.data_ready,
            minimum_residual_z=minimum_residual_z,
            minimum_score=minimum_score,
            minimum_failed_continuation_r=minimum_failed_continuation_r,
            score_weights=score_weights,
        )
        if not decision.eligible:
            continue
        ranking_score = score_daily_residual_features(
            opportunity.features,
            score_weights=(ranking_score_weights or score_weights),
        )
        eligible.append(
            (
                opportunity,
                replace(
                    decision,
                    score=ranking_score,
                    admission_score=decision.score,
                ),
            )
        )
    eligible.sort(
        key=lambda item: (
            -float(item[1].score),
            -abs(float(item[0].residual_z)),
            str(item[0].symbol),
        )
    )

    def stale_pool(
        opportunity: DailyResidualOpportunity,
        decision: DailyResidualEntryDecision,
        *,
        blocker_kind: Literal["sector_capacity", "portfolio_capacity"],
    ) -> list[DailyResidualReplacementIncumbent]:
        rows: list[DailyResidualReplacementIncumbent] = []
        for incumbent in active:
            if incumbent.held_sessions < int(minimum_held_sessions):
                continue
            if (
                incumbent.normalization_fraction
                > float(maximum_normalization_fraction)
            ):
                continue
            if loss_only and incumbent.unrealized_r > 0.0:
                continue
            if (
                float(decision.score) - float(incumbent.entry_score)
                < float(minimum_score_margin)
            ):
                continue
            if blocker_kind == "sector_capacity":
                if incumbent.sector != opportunity.sector:
                    continue
            else:
                candidate_sector_count = sector_counts[opportunity.sector]
                if incumbent.sector == opportunity.sector:
                    continue
                if sector_counts[incumbent.sector] <= candidate_sector_count:
                    continue
            rows.append(incumbent)
        rows.sort(
            key=lambda row: (
                float(row.entry_score),
                float(row.unrealized_r),
                -int(row.held_sessions),
                str(row.symbol),
            )
        )
        return rows

    for opportunity, decision in eligible:
        sector_full = (
            sector_counts[opportunity.sector]
            >= int(max_positions_per_sector)
        )
        portfolio_full = portfolio_count >= int(max_positions)
        route: Literal["sector_capacity", "portfolio_capacity"] | None = None
        pool: list[DailyResidualReplacementIncumbent] = []
        if sector_full and mode in {"sector_stale", "combined"}:
            route = "sector_capacity"
            pool = stale_pool(opportunity, decision, blocker_kind=route)
        if (
            not pool
            and portfolio_full
            and not sector_full
            and mode in {"portfolio_diversifying", "combined"}
        ):
            route = "portfolio_capacity"
            pool = stale_pool(opportunity, decision, blocker_kind=route)
        if not pool or route is None:
            continue
        incumbent = pool[0]
        return (
            DailyResidualReplacementDecision(
                incumbent_symbol=incumbent.symbol,
                candidate_symbol=opportunity.symbol,
                blocker_kind=route,
                incumbent_entry_score=float(incumbent.entry_score),
                candidate_score=float(decision.score),
                score_margin=(
                    float(decision.score) - float(incumbent.entry_score)
                ),
                incumbent_held_sessions=int(incumbent.held_sessions),
                incumbent_normalization_fraction=float(
                    incumbent.normalization_fraction
                ),
                incumbent_unrealized_r=float(incumbent.unrealized_r),
            ),
        )
    return ()


def advance_residual_management(
    state: ResidualManagementState,
    *,
    incremental_normalization_r: float,
    policy: ResidualManagementPolicy = ResidualManagementPolicy(),
) -> ResidualManagementDecision:
    """Advance one completed session using a formation-frozen residual anchor.

    Positive incremental values normalize the original dislocation; negative
    values extend it.  Decisions are emitted after the completed session and
    therefore execute through an adapter no earlier than the next session.
    """

    dislocation = abs(float(state.initial_dislocation_r))
    if dislocation <= 0.0:
        raise ValueError("initial_dislocation_r must be non-zero")
    cumulative = float(state.cumulative_normalization_r) + float(
        incremental_normalization_r
    )
    peak = max(float(state.peak_normalization_r), cumulative)
    held = int(state.held_sessions) + 1
    fraction = cumulative / dislocation
    next_state = ResidualManagementState(
        initial_dislocation_r=state.initial_dislocation_r,
        cumulative_normalization_r=cumulative,
        peak_normalization_r=peak,
        held_sessions=held,
        partial_taken=state.partial_taken,
    )

    if fraction <= -abs(policy.structural_failure_extension_fraction):
        return ResidualManagementDecision(
            "full_exit", "residual_structural_failure", 1.0, fraction, next_state
        )
    if fraction >= policy.full_normalization_fraction:
        return ResidualManagementDecision(
            "full_exit", "full_residual_normalization", 1.0, fraction, next_state
        )
    peak_fraction = peak / dislocation
    giveback_fraction = (peak - cumulative) / dislocation
    if (
        peak_fraction >= policy.profit_retention_activation_fraction
        and giveback_fraction >= policy.profit_retention_giveback_fraction
    ):
        return ResidualManagementDecision(
            "full_exit",
            "residual_profit_retention_giveback",
            1.0,
            fraction,
            next_state,
        )
    if (
        not state.partial_taken
        and fraction >= policy.partial_normalization_fraction
        and 0.0 < policy.partial_exit_fraction < 1.0
    ):
        partial_state = ResidualManagementState(
            initial_dislocation_r=state.initial_dislocation_r,
            cumulative_normalization_r=cumulative,
            peak_normalization_r=peak,
            held_sessions=held,
            partial_taken=True,
        )
        return ResidualManagementDecision(
            "partial_exit",
            "partial_residual_normalization",
            policy.partial_exit_fraction,
            fraction,
            partial_state,
        )
    if held >= int(policy.maximum_holding_sessions):
        return ResidualManagementDecision(
            "full_exit", "residual_half_life_time_stop", 1.0, fraction, next_state
        )
    return ResidualManagementDecision("hold", "await_residual_normalization", 0.0, fraction, next_state)


def build_daily_residual_execution_state(
    artifact: Any,
    *,
    nav: float,
    catastrophic_stop_atr: float,
    catastrophic_stop_residual_r: float = 4.0,
) -> DailyResidualExecutionState:
    """Create typed executable state from one immutable nightly artifact."""

    if getattr(artifact, "strategy_mode", "") != DAILY_RESIDUAL_SLEEVE:
        raise ValueError("daily residual execution requires a residual artifact")
    state = DailyResidualExecutionState(
        trade_date=artifact.trade_date,
        nav=float(nav),
    )
    held_symbols: set[str] = set()
    for held in artifact.held_positions:
        if held.sleeve_id != DAILY_RESIDUAL_SLEEVE:
            continue
        held_symbols.add(held.symbol)
        management = ResidualManagementState(
            initial_dislocation_r=float(held.residual_initial_dislocation_r),
            cumulative_normalization_r=float(
                held.residual_cumulative_normalization_r
            ),
            peak_normalization_r=float(
                getattr(
                    held,
                    "residual_peak_normalization_r",
                    max(0.0, held.residual_cumulative_normalization_r),
                )
            ),
            held_sessions=int(held.residual_held_sessions),
            partial_taken=bool(held.residual_partial_taken),
        )
        state.symbols[held.symbol] = DailyResidualSymbolState(
            symbol=held.symbol,
            issuer=issuer_key(held.issuer or held.symbol),
            sector=held.sector,
            exchange=held.exchange,
            primary_exchange=held.primary_exchange,
            currency=held.currency,
            tick_size=float(held.tick_size),
            point_value=float(held.point_value),
            factor_model=held.residual_factor_model,
            formation_sessions=int(held.residual_formation_sessions),
            residual_volatility=float(held.residual_volatility),
            initial_dislocation_r=management.initial_dislocation_r,
            residual_lane_id=held.residual_lane_id,
            residual_model_contract_version=held.residual_model_contract_version,
            residual_model_intercept=float(held.residual_model_intercept),
            residual_factor_names=tuple(held.residual_factor_names),
            residual_factor_betas=tuple(held.residual_factor_betas),
            residual_peer_symbols=tuple(held.residual_peer_symbols),
            residual_model_estimation_session=(
                held.residual_model_estimation_session
            ),
            last_processed_session=held.residual_last_processed_session,
            entry_score=float(held.residual_entry_score),
            pending_management_action=held.residual_pending_action,
            pending_management_reason=held.residual_pending_reason,
            pending_exit_fraction=float(held.residual_pending_exit_fraction),
            protective_stop_client_order_id=str(
                held.residual_protective_stop_client_order_id
            ),
            protective_stop_price=float(
                held.residual_protective_stop_price or held.stop
            ),
            protective_stop_qty=int(
                held.residual_protective_stop_qty or held.size
            ),
            position=DailyResidualExecutionPosition(
                symbol=held.symbol,
                issuer=issuer_key(held.issuer or held.symbol),
                sector=held.sector,
                qty_entry=int(held.residual_qty_entry or held.size),
                qty_open=int(held.size),
                entry_price=float(held.entry_price),
                entry_time=held.entry_time,
                initial_risk_per_share=max(float(held.initial_r), float(held.tick_size)),
                catastrophic_stop_distance=max(
                    float(held.entry_price) - float(held.stop),
                    float(held.tick_size),
                ),
                residual_factor_model=held.residual_factor_model,
                residual_formation_sessions=int(held.residual_formation_sessions),
                residual_volatility=float(held.residual_volatility),
                management=management,
                residual_lane_id=held.residual_lane_id,
                residual_model_contract_version=(
                    held.residual_model_contract_version
                ),
                residual_model_intercept=float(held.residual_model_intercept),
                residual_factor_names=tuple(held.residual_factor_names),
                residual_factor_betas=tuple(held.residual_factor_betas),
                residual_peer_symbols=tuple(held.residual_peer_symbols),
                residual_model_estimation_session=(
                    held.residual_model_estimation_session
                ),
                last_processed_session=held.residual_last_processed_session,
                entry_commission=float(held.residual_entry_commission),
                exit_commission=float(held.residual_exit_commission),
                realized_pnl_usd=float(held.residual_realized_pnl_usd),
                entry_score=float(held.residual_entry_score),
                trade_id=str(held.residual_trade_id),
            ),
        )

    parameters = dict(getattr(artifact, "strategy_parameters", {}) or {})
    risk_fraction = float(parameters.get("risk_fraction", 0.0035))
    notional_fraction = float(parameters.get("maximum_notional_fraction", 0.10))
    for item in artifact.tradable:
        if item.symbol in held_symbols:
            continue
        planned_price = float(item.previous_close)
        catastrophic_stop_distance = max(
            float(item.daily_atr_estimate) * float(catastrophic_stop_atr),
            planned_price
            * float(item.residual_volatility)
            * float(catastrophic_stop_residual_r),
            float(item.tick_size),
        )
        # One economic R is the prospective residual-horizon volatility that
        # defines both the signal and its discovery diagnostic.  The wider
        # catastrophe stop is tail protection; it must not redefine R or
        # silently shrink every position to the stop distance.
        residual_risk_per_share = max(
            planned_price
            * float(item.residual_volatility)
            * float(parameters.get("maximum_holding_sessions", 1)) ** 0.5,
            float(item.tick_size),
        )
        # The nightly artifact may deliberately reduce risk for a selective
        # sector-overflow admission.  Standard residual items carry the same
        # value as ``risk_fraction``, preserving historical sizing exactly.
        # Consuming the item-level value here keeps replay and live execution
        # identical and ensures an overflow-risk experiment is not a no-op.
        item_risk_fraction = float(
            getattr(item, "recommended_risk_r", risk_fraction)
        )
        if item_risk_fraction <= 0.0:
            item_risk_fraction = risk_fraction
        risk_qty = floor(
            float(nav) * item_risk_fraction / residual_risk_per_share
        )
        notional_qty = floor(float(nav) * notional_fraction / planned_price) if planned_price > 0 else 0
        qty = max(0, min(risk_qty, notional_qty))
        state.symbols[item.symbol] = DailyResidualSymbolState(
            symbol=item.symbol,
            issuer=issuer_key(item.symbol),
            sector=item.sector,
            exchange=item.exchange,
            primary_exchange=item.primary_exchange,
            currency=item.currency,
            tick_size=float(item.tick_size),
            point_value=float(item.point_value),
            factor_model=item.residual_factor_model,
            formation_sessions=int(item.residual_formation_sessions),
            residual_z=float(item.residual_z),
            residual_volatility=float(item.residual_volatility),
            initial_dislocation_r=float(item.residual_initial_dislocation_r),
            failed_continuation_r=float(item.residual_failed_continuation_r),
            sector_return_5d=float(item.residual_sector_return_5d),
            residual_lane_id=item.residual_lane_id,
            residual_model_contract_version=item.residual_model_contract_version,
            residual_model_intercept=float(item.residual_model_intercept),
            residual_factor_names=tuple(item.residual_factor_names),
            residual_factor_betas=tuple(item.residual_factor_betas),
            residual_peer_symbols=tuple(item.residual_peer_symbols),
            residual_model_estimation_session=(
                item.residual_model_estimation_session
            ),
            last_processed_session=item.anchor_date,
            planned_entry_price=planned_price,
            planned_stop_price=max(
                planned_price - catastrophic_stop_distance,
                float(item.tick_size),
            ),
            planned_initial_risk_per_share=residual_risk_per_share,
            planned_qty=qty,
            entry_score=float(item.daily_signal_score),
            entry_skipped_reason="nonpositive_executable_size" if qty <= 0 else "",
        )
    return state


def _daily_residual_full_exit_blockers(
    state: DailyResidualExecutionState,
) -> tuple[str, ...]:
    """Return positions that must be flat before staged entries may be sent."""

    return tuple(
        symbol
        for symbol in sorted(state.symbols)
        if (
            state.symbols[symbol].pending_management_action == "full_exit"
            and state.symbols[symbol].position is not None
            and state.symbols[symbol].position.qty_open > 0
        )
    )


def plan_daily_residual_session_orders(
    state: DailyResidualExecutionState,
    *,
    ts: datetime,
    allow_entries: bool,
) -> tuple[DailyResidualExecutionState, tuple[NeutralAction, ...], tuple[DecisionEvent, ...]]:
    """Plan next-open orders, holding entries behind actual full-exit fills."""

    actions: list[NeutralAction] = []
    events: list[DecisionEvent] = []
    if not state.exit_orders_planned:
        for symbol in sorted(state.symbols):
            symbol_state = state.symbols[symbol]
            position = symbol_state.position
            if position is None or position.qty_open <= 0:
                continue
            requested_action = symbol_state.pending_management_action
            if requested_action not in {"partial_exit", "full_exit"}:
                continue
            if requested_action == "partial_exit":
                qty = min(
                    position.qty_open,
                    max(1, int(round(position.qty_open * symbol_state.pending_exit_fraction))),
                )
                role = "PARTIAL_EXIT"
                action: NeutralAction = SubmitPartialExit(
                    client_order_id=f"IARIC-RES-{state.trade_date}-{symbol}-PARTIAL",
                    symbol=symbol,
                    side="SELL",
                    qty=qty,
                    route=DAILY_RESIDUAL_SLEEVE,
                    session="NEXT_OPEN",
                    oca_group=f"IARIC-RES-{symbol}-EXIT-OCA",
                    metadata={"reason": symbol_state.pending_management_reason},
                )
            else:
                qty = position.qty_open
                role = "EXIT"
                action = SubmitMarketExit(
                    client_order_id=f"IARIC-RES-{state.trade_date}-{symbol}-EXIT",
                    symbol=symbol,
                    side="SELL",
                    qty=qty,
                    route=DAILY_RESIDUAL_SLEEVE,
                    session="NEXT_OPEN",
                    oca_group=f"IARIC-RES-{symbol}-EXIT-OCA",
                    metadata={"reason": symbol_state.pending_management_reason},
                )
            if symbol_state.protective_stop_client_order_id:
                actions.append(
                    CancelAction(
                        symbol=symbol,
                        target_order_id=symbol_state.protective_stop_client_order_id,
                        reason="residual_management_exit",
                        route=DAILY_RESIDUAL_SLEEVE,
                        session="NEXT_OPEN",
                    )
                )
            symbol_state.pending_client_order_id = action.client_order_id
            symbol_state.pending_role = role
            symbol_state.pending_remaining_qty = qty
            actions.append(action)
            events.append(
                DecisionEvent(
                    code="RESIDUAL_MANAGEMENT_EXIT",
                    ts=ts,
                    symbol=symbol,
                    timeframe="1d",
                    strategy_id="IARIC_v1",
                    decision_kind="exit",
                    emitted_actions=(type(action).__name__,),
                    details={
                        "reason": symbol_state.pending_management_reason,
                        "qty": qty,
                        "role": role,
                    },
                )
            )
        state.exit_orders_planned = True

    if not state.entry_orders_planned:
        entry_symbols = tuple(
            symbol
            for symbol in sorted(state.symbols)
            if (
                state.symbols[symbol].position is None
                and state.symbols[symbol].planned_qty > 0
            )
        )
        staged_now = False
        if not entry_symbols:
            state.entry_orders_staged = True
            state.entry_orders_planned = True
        elif not state.entry_orders_staged:
            if allow_entries:
                state.entry_orders_staged = True
                staged_now = True
            else:
                for symbol in entry_symbols:
                    symbol_state = state.symbols[symbol]
                    symbol_state.entry_skipped_reason = (
                        "missed_live_next_open_staging_cutoff"
                    )
                    events.append(
                        DecisionEvent(
                            code="RESIDUAL_ENTRY_SKIPPED",
                            ts=ts,
                            symbol=symbol,
                            timeframe="1d",
                            strategy_id="IARIC_v1",
                            decision_kind="entry",
                            details={"reason": symbol_state.entry_skipped_reason},
                        )
                    )
                state.entry_orders_planned = True

        full_exit_blockers = _daily_residual_full_exit_blockers(state)
        if (
            state.entry_orders_staged
            and not state.entry_orders_planned
            and full_exit_blockers
        ):
            if staged_now:
                for symbol in entry_symbols:
                    events.append(
                        DecisionEvent(
                            code="RESIDUAL_ENTRY_DEFERRED",
                            ts=ts,
                            symbol=symbol,
                            timeframe="1d",
                            strategy_id="IARIC_v1",
                            decision_kind="entry",
                            details={
                                "reason": "awaiting_full_exit_fills",
                                "blocking_symbols": full_exit_blockers,
                            },
                        )
                    )
        elif state.entry_orders_staged and not state.entry_orders_planned:
            for symbol in entry_symbols:
                symbol_state = state.symbols[symbol]
                client_order_id = f"IARIC-RES-{state.trade_date}-{symbol}-ENTRY"
                action = SubmitEntry(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side="BUY",
                    qty=symbol_state.planned_qty,
                    order_type="MARKET",
                    route=DAILY_RESIDUAL_SLEEVE,
                    session="NEXT_OPEN",
                    risk_context={
                        "planned_entry_price": symbol_state.planned_entry_price,
                        "initial_residual_risk_per_share": (
                            symbol_state.planned_initial_risk_per_share
                        ),
                        "catastrophic_stop_price": symbol_state.planned_stop_price,
                        # OMS order-risk validation receives the actual hard stop.
                        # Economic R and sizing remain the separately supplied
                        # residual-horizon risk so catastrophe distance is visible
                        # as a multi-R tail exposure rather than relabelled as 1R.
                        "stop_for_risk": symbol_state.planned_stop_price,
                        "initial_dislocation_r": symbol_state.initial_dislocation_r,
                    },
                    metadata={
                        "factor_model": symbol_state.factor_model,
                        "formation_sessions": symbol_state.formation_sessions,
                        "lane_id": symbol_state.residual_lane_id,
                        "residual_model_contract_version": (
                            symbol_state.residual_model_contract_version
                        ),
                        "entry_clock": "next_session_open",
                    },
                )
                symbol_state.pending_client_order_id = client_order_id
                symbol_state.pending_role = "ENTRY"
                symbol_state.pending_remaining_qty = symbol_state.planned_qty
                actions.append(action)
                events.append(
                    DecisionEvent(
                        code="RESIDUAL_ENTRY_SELECTED",
                        ts=ts,
                        symbol=symbol,
                        timeframe="1d",
                        strategy_id="IARIC_v1",
                        decision_kind="entry",
                        emitted_actions=(type(action).__name__,),
                        details={
                            "qty": symbol_state.planned_qty,
                            "residual_z": symbol_state.residual_z,
                            "factor_model": symbol_state.factor_model,
                        },
                    )
                )
            state.entry_orders_planned = True
    state.session_orders_planned = state.entry_orders_planned and state.exit_orders_planned
    if events:
        state.last_decision_code = events[-1].code
    return state, tuple(actions), tuple(events)


def plan_daily_residual_forced_exit(
    state: DailyResidualExecutionState,
    *,
    symbol: str,
    ts: datetime,
    reason: str,
) -> tuple[DailyResidualExecutionState, SubmitMarketExit, DecisionEvent]:
    """Create an adapter-neutral forced exit without bypassing fill state.

    Normal management exits are formed by the previous completed session and
    execute at the following open.  Replay fold boundaries and live emergency
    handling are different operational events, but their fills must still pass
    through the same reducer.  This helper makes that exception explicit and
    auditable instead of allowing an adapter to mutate positions directly.
    """

    symbol_state = state.symbols.get(symbol)
    position = symbol_state.position if symbol_state is not None else None
    if symbol_state is None or position is None or position.qty_open <= 0:
        raise ValueError(f"cannot force-exit a flat residual symbol: {symbol}")
    if symbol_state.pending_remaining_qty > 0:
        raise ValueError(f"cannot force-exit residual symbol with a pending order: {symbol}")
    client_order_id = f"IARIC-RES-{state.trade_date}-{symbol}-FORCED-EXIT"
    action = SubmitMarketExit(
        client_order_id=client_order_id,
        symbol=symbol,
        side="SELL",
        qty=position.qty_open,
        route=DAILY_RESIDUAL_SLEEVE,
        session="IMMEDIATE",
        oca_group=f"IARIC-RES-{symbol}-EXIT-OCA",
        metadata={"reason": reason, "forced": True},
    )
    symbol_state.pending_client_order_id = client_order_id
    symbol_state.pending_role = "EXIT"
    symbol_state.pending_remaining_qty = position.qty_open
    symbol_state.pending_management_action = "full_exit"
    symbol_state.pending_management_reason = reason
    symbol_state.pending_exit_fraction = 1.0
    state.last_decision_code = "RESIDUAL_FORCED_EXIT"
    event = DecisionEvent(
        code="RESIDUAL_FORCED_EXIT",
        ts=ts,
        symbol=symbol,
        timeframe="operational",
        strategy_id="IARIC_v1",
        decision_kind="exit",
        emitted_actions=(type(action).__name__,),
        details={"reason": reason, "qty": position.qty_open},
    )
    return state, action, event


def apply_daily_residual_fill(
    state: DailyResidualExecutionState,
    fill: DailyResidualFill,
) -> tuple[
    DailyResidualExecutionState,
    tuple[NeutralAction, ...],
    tuple[DecisionEvent, ...],
]:
    """Apply one matched live or simulated fill to shared typed state."""

    symbol_state = state.symbols.get(fill.symbol)
    if symbol_state is None:
        raise ValueError(f"unmatched residual fill symbol: {fill.symbol}")
    if fill.qty <= 0 or fill.price <= 0.0:
        raise ValueError("residual fill must have positive quantity and price")
    if (
        symbol_state.pending_client_order_id
        and fill.client_order_id != symbol_state.pending_client_order_id
        and fill.role != "STOP"
    ):
        raise ValueError(f"unmatched residual client order id: {fill.client_order_id}")
    if fill.role == "STOP" and fill.client_order_id != (
        symbol_state.protective_stop_client_order_id
    ):
        raise ValueError(f"unmatched residual protective stop id: {fill.client_order_id}")
    if fill.role != "STOP" and fill.role != symbol_state.pending_role:
        raise ValueError(
            f"residual fill role {fill.role} does not match {symbol_state.pending_role}"
        )

    emitted_actions: list[NeutralAction] = []
    if fill.role == "ENTRY":
        existing = symbol_state.position
        if existing is None:
            catastrophic_stop_distance = max(
                symbol_state.planned_entry_price
                - symbol_state.planned_stop_price,
                symbol_state.tick_size,
            )
            risk_per_share = max(
                symbol_state.planned_initial_risk_per_share,
                symbol_state.tick_size,
            )
            position = DailyResidualExecutionPosition(
                symbol=fill.symbol,
                issuer=symbol_state.issuer,
                sector=symbol_state.sector,
                qty_entry=fill.qty,
                qty_open=fill.qty,
                entry_price=fill.price,
                entry_time=fill.ts,
                initial_risk_per_share=risk_per_share,
                catastrophic_stop_distance=catastrophic_stop_distance,
                residual_factor_model=symbol_state.factor_model,
                residual_formation_sessions=symbol_state.formation_sessions,
                residual_volatility=symbol_state.residual_volatility,
                management=ResidualManagementState(
                    initial_dislocation_r=symbol_state.initial_dislocation_r
                ),
                residual_lane_id=symbol_state.residual_lane_id,
                residual_model_contract_version=(
                    symbol_state.residual_model_contract_version
                ),
                residual_model_intercept=symbol_state.residual_model_intercept,
                residual_factor_names=symbol_state.residual_factor_names,
                residual_factor_betas=symbol_state.residual_factor_betas,
                residual_peer_symbols=symbol_state.residual_peer_symbols,
                residual_model_estimation_session=(
                    symbol_state.residual_model_estimation_session
                ),
                last_processed_session=symbol_state.last_processed_session,
                entry_commission=float(fill.commission),
                entry_score=float(symbol_state.entry_score),
            )
            symbol_state.position = position
        else:
            combined = existing.qty_entry + fill.qty
            existing.entry_price = (
                existing.entry_price * existing.qty_entry + fill.price * fill.qty
            ) / combined
            existing.qty_entry = combined
            existing.qty_open += fill.qty
            existing.entry_commission += float(fill.commission)
        position = symbol_state.position
        if position is None:
            raise RuntimeError("entry fill did not construct a residual position")
        stop_price = max(
            position.entry_price - position.catastrophic_stop_distance,
            symbol_state.tick_size,
        )
        stop_qty = position.qty_open
        if not symbol_state.protective_stop_client_order_id:
            stop_client_id = (
                f"IARIC-RES-{state.trade_date}-{fill.symbol}-CATASTOP"
            )
            emitted_actions.append(
                SubmitProtectiveStop(
                    client_order_id=stop_client_id,
                    symbol=fill.symbol,
                    side="SELL",
                    qty=stop_qty,
                    stop_price=stop_price,
                    route=DAILY_RESIDUAL_SLEEVE,
                    oca_group=f"IARIC-RES-{fill.symbol}-EXIT-OCA",
                    risk_context={"catastrophic_only": True},
                    metadata={"management": "residual_close_to_close"},
                )
            )
            symbol_state.protective_stop_client_order_id = stop_client_id
        else:
            emitted_actions.append(
                ReplaceProtectiveStop(
                    symbol=fill.symbol,
                    target_order_id=symbol_state.protective_stop_client_order_id,
                    side="SELL",
                    stop_price=stop_price,
                    qty=stop_qty,
                    reason="entry_partial_fill_resize",
                    route=DAILY_RESIDUAL_SLEEVE,
                )
            )
        symbol_state.protective_stop_price = stop_price
        symbol_state.protective_stop_qty = stop_qty
        code = "RESIDUAL_ENTRY_FILLED"
    else:
        position = symbol_state.position
        if position is None or fill.qty > position.qty_open:
            raise ValueError("residual exit fill exceeds the open position")
        gross = (fill.price - position.entry_price) * fill.qty * symbol_state.point_value
        position.qty_open -= fill.qty
        position.realized_pnl_usd += gross - float(fill.commission)
        position.exit_commission += float(fill.commission)
        if fill.role == "PARTIAL_EXIT" and position.qty_open > 0:
            code = "RESIDUAL_PARTIAL_EXIT_FILLED"
        elif fill.role == "STOP":
            code = "RESIDUAL_CATASTROPHIC_STOP_FILLED"
        else:
            code = "RESIDUAL_EXIT_FILLED"
        if position.qty_open > 0 and fill.role == "PARTIAL_EXIT":
            stop_client_id = (
                f"IARIC-RES-{state.trade_date}-{fill.symbol}-"
                f"CATASTOP-P{position.management.held_sessions}"
            )
            emitted_actions.append(
                SubmitProtectiveStop(
                    client_order_id=stop_client_id,
                    symbol=fill.symbol,
                    side="SELL",
                    qty=position.qty_open,
                    stop_price=symbol_state.protective_stop_price,
                    route=DAILY_RESIDUAL_SLEEVE,
                    oca_group=f"IARIC-RES-{fill.symbol}-EXIT-OCA",
                    risk_context={"catastrophic_only": True},
                    metadata={"reason": "resize_after_partial_exit"},
                )
            )
            symbol_state.protective_stop_client_order_id = stop_client_id
            symbol_state.protective_stop_qty = position.qty_open
        else:
            symbol_state.protective_stop_client_order_id = ""
            symbol_state.protective_stop_qty = 0

        if fill.role == "STOP":
            # The stop wins any race with a staged management exit.  The
            # adapter cancels that broker order; shared state must become flat
            # immediately so a late exit fill is quarantined, not double-sold.
            symbol_state.pending_client_order_id = ""
            symbol_state.pending_role = ""
            symbol_state.pending_remaining_qty = 0

    symbol_state.pending_remaining_qty = max(
        0, symbol_state.pending_remaining_qty - fill.qty
    )
    if fill.role != "STOP" and symbol_state.pending_remaining_qty == 0:
        symbol_state.pending_client_order_id = ""
        symbol_state.pending_role = ""
    state.last_decision_code = code
    event = DecisionEvent(
        code=code,
        ts=fill.ts,
        symbol=fill.symbol,
        timeframe="fill",
        strategy_id="IARIC_v1",
        decision_kind="fill",
        details={
            "qty": fill.qty,
            "price": fill.price,
            "commission": fill.commission,
            "role": fill.role,
        },
    )
    emitted_events = [event]
    if fill.role in {"EXIT", "STOP"} and not state.entry_orders_planned:
        state, released_entries, release_events = plan_daily_residual_session_orders(
            state,
            ts=fill.ts,
            # Only a prior pre-open staging decision can authorize entries once
            # the actual full-exit fills have released their capacity.
            allow_entries=False,
        )
        emitted_actions.extend(released_entries)
        emitted_events.extend(release_events)
    return state, tuple(emitted_actions), tuple(emitted_events)


def hydrate_daily_residual_symbol_state(
    payload: Mapping[str, Any],
) -> DailyResidualSymbolState:
    """Hydrate a persisted typed symbol state without broker dependencies."""

    def _date(value: Any) -> date | None:
        if value in (None, ""):
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        return date.fromisoformat(str(value)[:10])

    def _datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

    position_payload = payload.get("position")
    position = None
    if isinstance(position_payload, Mapping):
        management_payload = position_payload.get("management", {})
        position = DailyResidualExecutionPosition(
            symbol=str(position_payload["symbol"]),
            issuer=issuer_key(
                str(position_payload.get("issuer", position_payload["symbol"]))
            ),
            sector=str(position_payload.get("sector", "")),
            qty_entry=int(position_payload["qty_entry"]),
            qty_open=int(position_payload["qty_open"]),
            entry_price=float(position_payload["entry_price"]),
            entry_time=_datetime(position_payload["entry_time"]),
            initial_risk_per_share=float(position_payload["initial_risk_per_share"]),
            catastrophic_stop_distance=float(
                position_payload.get(
                    "catastrophic_stop_distance",
                    position_payload["initial_risk_per_share"],
                )
            ),
            residual_factor_model=str(
                position_payload.get("residual_factor_model", "")
            ),
            residual_formation_sessions=int(
                position_payload.get("residual_formation_sessions", 0)
            ),
            residual_volatility=float(position_payload.get("residual_volatility", 0.0)),
            management=ResidualManagementState(
                initial_dislocation_r=float(
                    management_payload["initial_dislocation_r"]
                ),
                cumulative_normalization_r=float(
                    management_payload.get("cumulative_normalization_r", 0.0)
                ),
                peak_normalization_r=float(
                    management_payload.get(
                        "peak_normalization_r",
                        max(
                            0.0,
                            float(
                                management_payload.get(
                                    "cumulative_normalization_r", 0.0
                                )
                            ),
                        ),
                    )
                ),
                held_sessions=int(management_payload.get("held_sessions", 0)),
                partial_taken=bool(management_payload.get("partial_taken", False)),
            ),
            residual_lane_id=str(position_payload.get("residual_lane_id", "")),
            residual_model_contract_version=str(
                position_payload.get("residual_model_contract_version", "")
            ),
            residual_model_intercept=float(
                position_payload.get("residual_model_intercept", 0.0)
            ),
            residual_factor_names=tuple(
                str(value)
                for value in position_payload.get("residual_factor_names", ())
            ),
            residual_factor_betas=tuple(
                float(value)
                for value in position_payload.get("residual_factor_betas", ())
            ),
            residual_peer_symbols=tuple(
                str(value)
                for value in position_payload.get("residual_peer_symbols", ())
            ),
            residual_model_estimation_session=_date(
                position_payload.get("residual_model_estimation_session")
            ),
            last_processed_session=_date(
                position_payload.get("last_processed_session")
            ),
            entry_commission=float(position_payload.get("entry_commission", 0.0)),
            exit_commission=float(position_payload.get("exit_commission", 0.0)),
            realized_pnl_usd=float(position_payload.get("realized_pnl_usd", 0.0)),
            entry_score=float(position_payload.get("entry_score", 0.0)),
            trade_id=str(position_payload.get("trade_id", "")),
        )
    return DailyResidualSymbolState(
        symbol=str(payload["symbol"]),
        issuer=issuer_key(str(payload.get("issuer", payload["symbol"]))),
        sector=str(payload.get("sector", "")),
        exchange=str(payload.get("exchange", "SMART")),
        primary_exchange=str(payload.get("primary_exchange", "")),
        currency=str(payload.get("currency", "USD")),
        tick_size=float(payload.get("tick_size", 0.01)),
        point_value=float(payload.get("point_value", 1.0)),
        sleeve_id=str(payload.get("sleeve_id", DAILY_RESIDUAL_SLEEVE)),
        factor_model=str(payload.get("factor_model", "")),
        formation_sessions=int(payload.get("formation_sessions", 0)),
        residual_z=float(payload.get("residual_z", 0.0)),
        residual_volatility=float(payload.get("residual_volatility", 0.0)),
        initial_dislocation_r=float(payload.get("initial_dislocation_r", 0.0)),
        failed_continuation_r=float(payload.get("failed_continuation_r", 0.0)),
        sector_return_5d=float(payload.get("sector_return_5d", 0.0)),
        residual_lane_id=str(payload.get("residual_lane_id", "")),
        residual_model_contract_version=str(
            payload.get("residual_model_contract_version", "")
        ),
        residual_model_intercept=float(payload.get("residual_model_intercept", 0.0)),
        residual_factor_names=tuple(
            str(value) for value in payload.get("residual_factor_names", ())
        ),
        residual_factor_betas=tuple(
            float(value) for value in payload.get("residual_factor_betas", ())
        ),
        residual_peer_symbols=tuple(
            str(value) for value in payload.get("residual_peer_symbols", ())
        ),
        residual_model_estimation_session=_date(
            payload.get("residual_model_estimation_session")
        ),
        last_processed_session=_date(payload.get("last_processed_session")),
        planned_entry_price=float(payload.get("planned_entry_price", 0.0)),
        planned_stop_price=float(payload.get("planned_stop_price", 0.0)),
        planned_initial_risk_per_share=float(
            payload.get("planned_initial_risk_per_share", 0.0)
        ),
        planned_qty=int(payload.get("planned_qty", 0)),
        entry_score=float(payload.get("entry_score", 0.0)),
        pending_client_order_id=str(payload.get("pending_client_order_id", "")),
        pending_role=str(payload.get("pending_role", "")),
        pending_remaining_qty=int(payload.get("pending_remaining_qty", 0)),
        pending_management_action=str(
            payload.get("pending_management_action", "hold")
        ),
        pending_management_reason=str(payload.get("pending_management_reason", "")),
        pending_exit_fraction=float(payload.get("pending_exit_fraction", 0.0)),
        protective_stop_client_order_id=str(
            payload.get("protective_stop_client_order_id", "")
        ),
        protective_stop_price=float(payload.get("protective_stop_price", 0.0)),
        protective_stop_qty=int(payload.get("protective_stop_qty", 0)),
        entry_skipped_reason=str(payload.get("entry_skipped_reason", "")),
        position=position,
    )


if len(SLEEVE_SPECS[DAILY_RESIDUAL_SLEEVE].score_components) > 7:
    raise RuntimeError("daily residual sleeve must keep no more than seven score components")
