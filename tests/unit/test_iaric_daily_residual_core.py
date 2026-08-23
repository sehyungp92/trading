from __future__ import annotations

import pytest

from strategies.stock.iaric.core.daily_residual import (
    DailyResidualFeatures,
    DailyResidualOpportunity,
    DailyResidualReplacementIncumbent,
    ResidualManagementPolicy,
    ResidualManagementState,
    advance_residual_management,
    choose_capacity_neutral_replacements,
    decide_daily_residual_entry,
    rank_daily_residual_opportunities,
    score_daily_residual_features,
)


def _features(value: float = 0.5) -> DailyResidualFeatures:
    return DailyResidualFeatures(
        residual_extremeness=value,
        shock_freshness=value,
        price_rejection_recovery=value,
        volume_transition=value,
        volume_exhaustion_quality=value,
        residual_normalization_room=value,
        regime_execution_quality=value,
        failed_continuation=value,
    )


def test_daily_residual_score_has_seven_fixed_components() -> None:
    assert score_daily_residual_features(_features()) == pytest.approx(50.0)
    assert score_daily_residual_features(_features(1.0)) == pytest.approx(100.0)


def test_entry_has_no_news_gate_or_score_floor() -> None:
    weak_but_valid = decide_daily_residual_entry(
        _features(0.1),
        side="long",
        residual_z=-1.2,
        remaining_room_r=0.5,
        cost_feasible=True,
        data_ready=True,
    )
    assert weak_but_valid.eligible is True
    assert weak_but_valid.reason == "eligible_top_rank"


def test_entry_can_reject_weak_scores_before_capacity_when_floor_is_frozen() -> None:
    decision = decide_daily_residual_entry(
        _features(0.24),
        side="long",
        residual_z=-1.2,
        remaining_room_r=0.5,
        minimum_score=25.0,
        cost_feasible=True,
        data_ready=True,
    )
    assert decision.eligible is False
    assert decision.score == pytest.approx(24.0)
    assert decision.reason == "score_below_floor"


def test_entry_rejects_wrong_direction_and_nonpositive_room_before_capacity() -> None:
    assert decide_daily_residual_entry(
        _features(),
        side="long",
        residual_z=1.5,
        remaining_room_r=1.0,
        cost_feasible=True,
        data_ready=True,
    ).reason == "residual_not_extreme"
    assert decide_daily_residual_entry(
        _features(),
        side="short",
        residual_z=1.5,
        remaining_room_r=0.0,
        cost_feasible=True,
        data_ready=True,
    ).reason == "nonpositive_remaining_room"


def test_entry_rejects_continuation_before_capacity() -> None:
    decision = decide_daily_residual_entry(
        _features(),
        side="long",
        residual_z=-1.5,
        remaining_room_r=1.0,
        failed_continuation_r=0.19,
        minimum_failed_continuation_r=0.20,
        cost_feasible=True,
        data_ready=True,
    )
    assert decision.eligible is False
    assert decision.reason == "residual_continuation_not_failed"


def test_entry_rejects_adverse_sector_regime_before_capacity() -> None:
    decision = decide_daily_residual_entry(
        _features(),
        side="long",
        residual_z=-1.5,
        remaining_room_r=1.0,
        failed_continuation_r=0.5,
        regime_feasible=False,
        cost_feasible=True,
        data_ready=True,
    )
    assert decision.eligible is False
    assert decision.reason == "adverse_sector_regime"


def test_management_uses_frozen_anchor_partial_then_full_exit() -> None:
    policy = ResidualManagementPolicy(maximum_holding_sessions=7)
    first = advance_residual_management(
        ResidualManagementState(initial_dislocation_r=-2.0),
        incremental_normalization_r=1.1,
        policy=policy,
    )
    assert first.action == "partial_exit"
    assert first.next_state.partial_taken is True
    second = advance_residual_management(
        first.next_state,
        incremental_normalization_r=1.0,
        policy=policy,
    )
    assert second.action == "full_exit"
    assert second.reason == "full_residual_normalization"


def test_management_exits_structural_failure_before_time_stop() -> None:
    decision = advance_residual_management(
        ResidualManagementState(
            initial_dislocation_r=-2.0,
            cumulative_normalization_r=-0.8,
            held_sessions=6,
        ),
        incremental_normalization_r=-0.3,
        policy=ResidualManagementPolicy(maximum_holding_sessions=7),
    )
    assert decision.action == "full_exit"
    assert decision.reason == "residual_structural_failure"


def test_management_profit_retention_uses_completed_residual_peak_and_giveback() -> None:
    policy = ResidualManagementPolicy(
        maximum_holding_sessions=10,
        partial_normalization_fraction=99.0,
        full_normalization_fraction=99.0,
        structural_failure_extension_fraction=99.0,
        profit_retention_activation_fraction=0.75,
        profit_retention_giveback_fraction=0.35,
        partial_exit_fraction=0.0,
    )
    armed = advance_residual_management(
        ResidualManagementState(initial_dislocation_r=-2.0),
        incremental_normalization_r=1.6,
        policy=policy,
    )
    assert armed.action == "hold"
    assert armed.next_state.peak_normalization_r == pytest.approx(1.6)

    exit_decision = advance_residual_management(
        armed.next_state,
        incremental_normalization_r=-0.8,
        policy=policy,
    )
    assert exit_decision.action == "full_exit"
    assert exit_decision.reason == "residual_profit_retention_giveback"


def test_shared_ranker_enforces_issuer_sector_and_total_caps_deterministically() -> None:
    opportunities = []
    for symbol, issuer, sector, quality in (
        ("AAA", "AAA", "Technology", 0.95),
        ("AAA.A", "AAA", "Technology", 0.90),
        ("BBB", "BBB", "Technology", 0.85),
        ("CCC", "CCC", "Technology", 0.80),
        ("DDD", "DDD", "Health Care", 0.75),
    ):
        opportunities.append(
            DailyResidualOpportunity(
                symbol=symbol,
                issuer=issuer,
                sector=sector,
                side="long",
                residual_z=-2.0,
                remaining_room_r=1.0,
                features=_features(quality),
            )
        )
    selected = rank_daily_residual_opportunities(
        reversed(opportunities), max_positions=3, max_positions_per_sector=2
    )
    assert [row.opportunity.symbol for row in selected] == ["AAA", "BBB", "DDD"]
    assert [row.rank for row in selected] == [1, 2, 3]


def test_selective_sector_overflow_admits_only_exceptional_third_sector_name() -> None:
    opportunities = [
        DailyResidualOpportunity(
            symbol=symbol,
            issuer=symbol,
            sector=sector,
            side="long",
            residual_z=residual_z,
            remaining_room_r=1.0,
            features=_features(quality),
        )
        for symbol, sector, residual_z, quality in (
            ("TECH1", "Technology", -2.0, 0.95),
            ("TECH2", "Technology", -1.8, 0.85),
            ("TECH3", "Technology", -1.3, 0.65),
            ("TECH_WEAK", "Technology", -1.4, 0.55),
            ("HEALTH", "Health Care", -1.2, 0.60),
        )
    ]

    selected = rank_daily_residual_opportunities(
        opportunities,
        max_positions=4,
        max_positions_per_sector=2,
        sector_overflow_slots=1,
        sector_overflow_minimum_score=60.0,
        sector_overflow_minimum_z=1.10,
    )

    assert [row.opportunity.symbol for row in selected] == [
        "TECH1",
        "TECH2",
        "TECH3",
        "HEALTH",
    ]
    assert [row.sector_overflow for row in selected] == [False, False, True, False]


def test_selective_sector_overflow_preserves_ordinary_cap_when_quality_fails() -> None:
    opportunities = [
        DailyResidualOpportunity(
            symbol=f"TECH{index}",
            issuer=f"TECH{index}",
            sector="Technology",
            side="long",
            residual_z=-1.05 if index == 3 else -2.0,
            remaining_room_r=1.0,
            features=_features(0.95 - index * 0.10),
        )
        for index in range(1, 4)
    ]

    selected = rank_daily_residual_opportunities(
        opportunities,
        max_positions=4,
        max_positions_per_sector=2,
        sector_overflow_slots=1,
        sector_overflow_minimum_score=60.0,
        sector_overflow_minimum_z=1.10,
    )

    assert [row.opportunity.symbol for row in selected] == ["TECH1", "TECH2"]
    assert not any(row.sector_overflow for row in selected)


def test_capacity_neutral_sector_replacement_chooses_weakest_stale_incumbent() -> None:
    candidate = DailyResidualOpportunity(
        symbol="TECH_NEW",
        issuer="TECH_NEW",
        sector="Technology",
        side="long",
        residual_z=-2.0,
        remaining_room_r=1.0,
        features=_features(0.90),
    )
    incumbents = (
        DailyResidualReplacementIncumbent(
            "TECH_WEAK", "TECH_WEAK", "Technology", 40.0, 6, 0.10, -0.2
        ),
        DailyResidualReplacementIncumbent(
            "TECH_STRONG", "TECH_STRONG", "Technology", 60.0, 7, 0.05, 0.1
        ),
    )

    replacements = choose_capacity_neutral_replacements(
        [candidate],
        incumbents,
        (),
        mode="sector_stale",
        loss_only=False,
        minimum_held_sessions=5,
        maximum_normalization_fraction=0.25,
        minimum_score_margin=25.0,
        maximum_replacements=1,
        max_positions=12,
        max_positions_per_sector=2,
        minimum_residual_z=1.0,
        minimum_score=25.0,
        minimum_failed_continuation_r=0.0,
        score_weights={"volume_transition": 1.0},
        ranking_score_weights={"volume_transition": 1.0},
    )

    assert len(replacements) == 1
    assert replacements[0].incumbent_symbol == "TECH_WEAK"
    assert replacements[0].candidate_symbol == "TECH_NEW"
    assert replacements[0].blocker_kind == "sector_capacity"
    assert replacements[0].score_margin == pytest.approx(50.0)


def test_capacity_neutral_loss_only_and_staleness_rules_block_churn() -> None:
    candidate = DailyResidualOpportunity(
        symbol="NEW",
        issuer="NEW",
        sector="Energy",
        side="long",
        residual_z=-2.0,
        remaining_room_r=1.0,
        features=_features(0.90),
    )
    profitable_or_young = (
        DailyResidualReplacementIncumbent(
            "TECH", "TECH", "Technology", 30.0, 6, 0.10, 0.2
        ),
        DailyResidualReplacementIncumbent(
            "HEALTH", "HEALTH", "Health Care", 30.0, 4, 0.10, -0.2
        ),
    )

    replacements = choose_capacity_neutral_replacements(
        [candidate],
        profitable_or_young,
        (),
        mode="portfolio_diversifying",
        loss_only=True,
        minimum_held_sessions=5,
        maximum_normalization_fraction=0.25,
        minimum_score_margin=25.0,
        maximum_replacements=1,
        max_positions=2,
        max_positions_per_sector=2,
        minimum_residual_z=1.0,
        minimum_score=25.0,
        minimum_failed_continuation_r=0.0,
        score_weights={"volume_transition": 1.0},
        ranking_score_weights={"volume_transition": 1.0},
    )

    assert replacements == ()


def test_capacity_neutral_replacement_excludes_preplanned_exit_issuer_alias() -> None:
    opportunities = (
        DailyResidualOpportunity(
            symbol="GOOGL",
            issuer="ALPHABET",
            sector="Communication Services",
            side="long",
            residual_z=-2.5,
            remaining_room_r=1.0,
            features=_features(0.95),
        ),
        DailyResidualOpportunity(
            symbol="META",
            issuer="META",
            sector="Communication Services",
            side="long",
            residual_z=-2.0,
            remaining_room_r=1.0,
            features=_features(0.90),
        ),
    )
    incumbents = (
        DailyResidualReplacementIncumbent(
            "WEAK", "WEAK", "Communication Services", 30.0, 6, 0.10, -0.2
        ),
    )

    replacements = choose_capacity_neutral_replacements(
        opportunities,
        incumbents,
        (),
        mode="sector_stale",
        loss_only=False,
        minimum_held_sessions=5,
        maximum_normalization_fraction=0.25,
        minimum_score_margin=25.0,
        maximum_replacements=1,
        max_positions=12,
        max_positions_per_sector=1,
        minimum_residual_z=1.0,
        minimum_score=25.0,
        minimum_failed_continuation_r=0.0,
        score_weights={"volume_transition": 1.0},
        ranking_score_weights={"volume_transition": 1.0},
        blocked_issuers=("GOOG",),
    )

    assert len(replacements) == 1
    assert replacements[0].candidate_symbol == "META"


def test_capacity_neutral_replacement_canonicalizes_raw_incumbent_issuer() -> None:
    candidate = DailyResidualOpportunity(
        symbol="GOOGL",
        issuer="ALPHABET",
        sector="Communication Services",
        side="long",
        residual_z=-2.5,
        remaining_room_r=1.0,
        features=_features(0.95),
    )
    raw_alias_incumbent = DailyResidualReplacementIncumbent(
        "GOOG", "GOOG", "Communication Services", 30.0, 6, 0.10, -0.2
    )

    replacements = choose_capacity_neutral_replacements(
        [candidate],
        [raw_alias_incumbent],
        (),
        mode="sector_stale",
        loss_only=False,
        minimum_held_sessions=5,
        maximum_normalization_fraction=0.25,
        minimum_score_margin=25.0,
        maximum_replacements=1,
        max_positions=12,
        max_positions_per_sector=1,
        minimum_residual_z=1.0,
        minimum_score=25.0,
        minimum_failed_continuation_r=0.0,
        score_weights={"volume_transition": 1.0},
        ranking_score_weights={"volume_transition": 1.0},
    )

    assert replacements == ()


def test_capacity_neutral_required_candidate_reserves_released_slot() -> None:
    higher_ranked = DailyResidualOpportunity(
        symbol="HIGHER",
        issuer="HIGHER",
        sector="Technology",
        side="long",
        residual_z=-2.5,
        remaining_room_r=1.0,
        features=_features(0.95),
    )
    paired_replacement = DailyResidualOpportunity(
        symbol="PAIRED",
        issuer="PAIRED",
        sector="Communication Services",
        side="long",
        residual_z=-2.0,
        remaining_room_r=1.0,
        features=_features(0.90),
    )

    selected = rank_daily_residual_opportunities(
        [higher_ranked, paired_replacement],
        max_positions=1,
        max_positions_per_sector=1,
        minimum_residual_z=1.0,
        minimum_score=25.0,
        score_weights={"volume_transition": 1.0},
        ranking_score_weights={"volume_transition": 1.0},
        required_symbols=["PAIRED"],
    )

    assert [row.opportunity.symbol for row in selected] == ["PAIRED"]


def test_two_stage_selector_admits_on_broad_score_and_ranks_on_stable_subset() -> None:
    first = DailyResidualOpportunity(
        symbol="FIRST",
        issuer="FIRST",
        sector="Technology",
        side="long",
        residual_z=-2.0,
        remaining_room_r=1.0,
        features=DailyResidualFeatures(
            **{
                **_features(0.5).as_mapping(),
                "volume_transition": 0.9,
                "price_rejection_recovery": 0.9,
                "failed_continuation": 0.1,
            }
        ),
    )
    second = DailyResidualOpportunity(
        symbol="SECOND",
        issuer="SECOND",
        sector="Technology",
        side="long",
        residual_z=-2.0,
        remaining_room_r=1.0,
        features=DailyResidualFeatures(
            **{
                **_features(0.5).as_mapping(),
                "volume_transition": 0.7,
                "price_rejection_recovery": 0.7,
                "failed_continuation": 1.0,
            }
        ),
    )
    selected = rank_daily_residual_opportunities(
        [second, first],
        max_positions=1,
        minimum_score=25.0,
        score_weights={
            "volume_transition": 1.0,
            "price_rejection_recovery": 1.0,
            "failed_continuation": 1.0,
        },
        ranking_score_weights={
            "volume_transition": 1.0,
            "price_rejection_recovery": 1.0,
        },
    )
    assert [row.opportunity.symbol for row in selected] == ["FIRST"]
    assert selected[0].decision.admission_score == pytest.approx(63.333333)
    assert selected[0].decision.score == pytest.approx(90.0)


def test_ranker_releases_full_exit_capacity_without_reentering_same_issuer() -> None:
    opportunities = [
        DailyResidualOpportunity(
            symbol=symbol,
            issuer=issuer,
            sector="Technology",
            side="long",
            residual_z=-2.0,
            remaining_room_r=1.0,
            features=_features(quality),
        )
        for symbol, issuer, quality in (
            ("EXIT", "EXIT", 0.99),
            ("NEW", "NEW", 0.90),
        )
    ]
    selected = rank_daily_residual_opportunities(
        opportunities,
        max_positions=2,
        max_positions_per_sector=2,
        active_issuers=("HELD",),
        active_sectors=("Technology",),
        blocked_issuers=("EXIT",),
    )
    assert [row.opportunity.symbol for row in selected] == ["NEW"]
