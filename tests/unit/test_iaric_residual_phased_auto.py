from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.auto.runners import run_iaric_residual_phased_auto as phased
from backtests.stock.auto.runners import (
    run_iaric_residual_phase8_continuation as continuation,
)
from backtests.stock.auto.iaric.residual_phases import (
    ROUND2_SCORE_SPEC,
    _continuous_fold_result,
    _winner_robustness,
    _winner_robustness_passes,
    settings_from_discovery_candidate,
)
from backtests.stock.auto.iaric import residual_phases
from backtests.stock.auto.iaric.representative_contract import (
    HOLDOUT_START,
    LOCKED_VALIDATION_END,
)
from backtests.stock.engine.iaric_daily_residual_replay import (
    DailyResidualReplayResult,
    DailyResidualReplayTrade,
)


def _trade(symbol: str, entry: date, exit_: date, r: float) -> DailyResidualReplayTrade:
    return DailyResidualReplayTrade(
        symbol=symbol,
        sector="Technology",
        entry_date=entry,
        entry_time=datetime.combine(entry, datetime.min.time(), tzinfo=timezone.utc),
        entry_price=100.0,
        qty_entry=1,
        initial_risk_dollars=10.0,
        factor_model="market_sector_peer",
        formation_sessions=1,
        score=50.0,
        exit_date=exit_,
        exit_time=datetime.combine(exit_, datetime.min.time(), tzinfo=timezone.utc),
        exit_price=100.0 + 10.0 * r,
        r_multiple=r,
    )


def test_continuous_fold_view_preserves_carry_and_purges_boundary_labels() -> None:
    result = DailyResidualReplayResult(
        initial_equity=100_000.0,
        final_equity=101_000.0,
        trades=[
            _trade("DONE", date(2024, 11, 1), date(2024, 11, 15), 1.0),
            _trade("CARRY", date(2024, 11, 20), date(2024, 12, 10), 2.0),
            _trade("CAL", date(2024, 12, 3), date(2024, 12, 20), -0.5),
        ],
        equity_curve=[
            {"date": "2024-11-29", "mtm_equity": 100_500.0, "open_positions": 1},
            {"date": "2024-12-02", "mtm_equity": 100_600.0, "open_positions": 1},
            {"date": "2024-12-31", "mtm_equity": 101_000.0, "open_positions": 0},
        ],
        decision_events=[],
        source_fingerprint="test",
        factor_model="market_sector_peer",
    )
    discovery, purged, carried = _continuous_fold_result(
        result,
        start=date(2024, 11, 1),
        end=date(2024, 11, 30),
    )
    calibration, cal_purged, cal_carried = _continuous_fold_result(
        result,
        start=date(2024, 12, 1),
        end=date(2024, 12, 31),
    )
    assert [trade.symbol for trade in discovery.trades] == ["DONE"]
    assert purged == 1
    assert carried == 0
    assert [trade.symbol for trade in calibration.trades] == ["CAL"]
    assert cal_purged == 0
    assert cal_carried == 1
    assert calibration.initial_equity == 100_500.0
    assert calibration.final_equity == 101_000.0


def test_missing_authority_stops_before_any_price_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("price loader must not run before authority passes")

    monkeypatch.setattr(
        discovery,
        "_load_daily_panel_from_authoritative_bundle",
        forbidden,
    )
    output = tmp_path / "output"
    result = phased.run(
        output,
        tmp_path / "unused-data",
        max_workers=2,
        authority_manifest=tmp_path / "missing-authority.json",
        data_contract=phased.PRODUCTION_AUTHORITY,
    )
    assert result == 2
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked_phase_0_missing_authoritative_price_contract"
    assert summary["optimizer_started"] is False
    assert summary["locked_validation_accessed"] is False
    assert summary["holdout_accessed"] is False


def test_authoritative_loader_cannot_expose_locked_or_holdout_without_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="one-shot"):
        discovery._load_daily_panel_from_authoritative_bundle(
            tmp_path / "missing-bundle.json",
            selection_end=LOCKED_VALIDATION_END,
        )
    with pytest.raises(ValueError, match="sealed holdout"):
        discovery._load_daily_panel_from_authoritative_bundle(
            tmp_path / "missing-bundle.json",
            selection_end=HOLDOUT_START,
            allow_locked_validation=True,
        )


def test_phase_registry_and_immutable_scores_are_bounded() -> None:
    assert len(discovery.SCORE_SPEC) == 7
    assert len(ROUND2_SCORE_SPEC) == 7
    assert sum(row["weight"] for row in ROUND2_SCORE_SPEC.values()) == pytest.approx(1.0)
    assert len(phased.PHASE_ORDER) == 20
    assert phased.PHASE_ORDER[11] == (
        "phase_8_selective_sector_overflow_and_displacement_quality"
    )
    assert phased.PHASE_ORDER[16] == "phase_13_path_causal_profit_retention"
    assert phased.PHASE_ORDER[17] == "phase_14_capacity_neutral_alpha_recycling"
    assert phased.PHASE_ORDER[-1] == "phase_16_locked_chronological_validation"


@dataclass(frozen=True)
class _FakeFrontierBundle:
    frozen_history_cache: dict


def _frontier_settings():
    candidate = discovery.Candidate(
        candidate_id="frontier_control",
        residual_z_floor=1.0,
        holding_sessions=10,
        max_positions=10,
        max_positions_per_sector=2,
        formation_sessions=1,
        factor_model="market_sector_peer",
        score_components=("volume_transition", "price_rejection_recovery"),
        minimum_score=25.0,
        catastrophic_stop_residual_r=6.0,
    )
    return settings_from_discovery_candidate(candidate.__dict__)


def test_latest_compact_frozen_candidate_is_a_valid_starting_baseline(
    tmp_path: Path,
) -> None:
    settings = _frontier_settings()
    path = tmp_path / "frozen_selection_candidate.json"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "latest_optimized",
                "settings": phased._settings_payload(settings),
                "settings_sha256": "declared-settings-hash",
            }
        ),
        encoding="utf-8",
    )

    candidate, lineage = phased._load_round2_baseline(path)

    assert candidate.candidate_id == "latest_optimized"
    assert candidate.minimum_score == 25.0
    assert candidate.score_components == (
        "volume_transition",
        "price_rejection_recovery",
    )
    assert lineage["configuration_role"] == "frozen_selection_candidate"
    assert lineage["declared_baseline_sha256"] == "declared-settings-hash"


def test_phase9_resume_hydrates_the_exact_phase8_compact_settings() -> None:
    original = replace(
        _frontier_settings(),
        daily_residual_max_positions=12,
        daily_residual_sector_overflow_slots=1,
        daily_residual_sector_overflow_minimum_score=60.0,
        daily_residual_sector_overflow_minimum_z=1.10,
        daily_residual_sector_overflow_risk_multiplier=0.75,
        daily_residual_profit_retention_activation_fraction=1.0,
        daily_residual_profit_retention_giveback_fraction=0.35,
        daily_residual_replacement_mode="combined",
        daily_residual_replacement_loss_only=True,
    )

    resumed = continuation._settings_from_compact(
        phased._settings_payload(original)
    )

    assert phased._settings_payload(resumed) == phased._settings_payload(original)


def _fake_exact_result(*, total_r: float, drawdown: float, score: float) -> dict:
    return {
        "research_anchor_eligible": True,
        "continuous_metrics": {
            "total_r": total_r,
            "max_drawdown_pct": drawdown,
            "return_pct": total_r / 100.0,
            "trades": 300,
        },
        "immutable_score": {"score": score},
    }


def test_risk_frontier_never_restores_an_ineligible_control_on_a_score_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _frontier_settings()

    def fake_exact(_bundle, trial, **_kwargs):
        drawdown = (
            0.121 if trial.daily_residual_risk_fraction >= 0.0035 else 0.09
        )
        return _fake_exact_result(total_r=95.0, drawdown=drawdown, score=0.7)

    monkeypatch.setattr(residual_phases, "run_exact_fold_evaluation", fake_exact)
    result = residual_phases.run_risk_notional_frontier_phase(
        _FakeFrontierBundle(frozen_history_cache={}),
        settings,
        max_workers=2,
    )

    assert result["control"]["selection_eligible"] is False
    assert result["selected"]["selection_eligible"] is True
    assert result["selected"]["experiment_id"] != "control"
    assert result["selected_settings"].daily_residual_risk_fraction < 0.0035


def test_final_synergy_accepts_99r_improvement_below_10pct_dd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _frontier_settings()

    def fake_exact(_bundle, trial, **_kwargs):
        improved = (
            trial.daily_residual_minimum_score < 25.0
            and trial.daily_residual_risk_fraction < 0.0035
        )
        return _fake_exact_result(
            total_r=99.0 if improved else 90.0,
            drawdown=0.099 if improved else 0.101,
            score=0.8 if improved else 0.7,
        )

    monkeypatch.setattr(residual_phases, "run_exact_fold_evaluation", fake_exact)
    result = residual_phases.run_final_alpha_synergy_phase(
        _FakeFrontierBundle(frozen_history_cache={}),
        settings,
        max_workers=2,
    )

    assert result["status"] == "passed"
    assert result["selected"]["aspirational_targets"] == {
        "total_r_above_100r": False,
        "mtm_max_drawdown_below_10pct": True,
    }
    assert result["selected_settings"].daily_residual_max_positions == 10
    assert result["selected_settings"].daily_residual_minimum_score == 20.0
    assert result["aspirational_targets_are_hard_gates"] is False


def test_final_synergy_does_not_reject_improvement_on_guidance_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _frontier_settings()

    def fake_exact(_bundle, trial, **_kwargs):
        improved = (
            trial.daily_residual_minimum_score < 25.0
            and trial.daily_residual_risk_fraction < 0.0035
        )
        return _fake_exact_result(
            total_r=100.0 if improved else 90.0,
            drawdown=0.10,
            score=0.8 if improved else 0.7,
        )

    monkeypatch.setattr(residual_phases, "run_exact_fold_evaluation", fake_exact)
    result = residual_phases.run_final_alpha_synergy_phase(
        _FakeFrontierBundle(frozen_history_cache={}),
        settings,
        max_workers=2,
    )

    assert result["status"] == "passed"
    assert result["selected_settings"].daily_residual_max_positions == 10
    assert result["selected_settings"].daily_residual_minimum_score == 20.0
    assert result["selected"]["aspirational_targets"] == {
        "total_r_above_100r": False,
        "mtm_max_drawdown_below_10pct": False,
    }


def test_phase8_inherits_exact_cap12_control_without_replaying_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(_frontier_settings(), daily_residual_max_positions=12)
    inherited = {
        **_fake_exact_result(total_r=97.49, drawdown=0.121, score=0.7773),
        "settings": {
            "max_positions": 12,
            "max_positions_per_sector": 2,
            "minimum_z": 1.0,
            "minimum_score": 25.0,
        },
    }
    calls = 0

    def fake_exact(_bundle, trial, **_kwargs):
        nonlocal calls
        calls += 1
        assert trial.daily_residual_sector_overflow_slots == 1
        return _fake_exact_result(total_r=98.0, drawdown=0.115, score=0.76)

    monkeypatch.setattr(residual_phases, "run_exact_fold_evaluation", fake_exact)
    result = residual_phases.run_selective_sector_overflow_phase(
        _FakeFrontierBundle(frozen_history_cache={}),
        settings,
        max_workers=2,
        inherited_control_result=inherited,
    )

    assert calls == 7
    assert result["control"]["result_source"] == (
        "inherited_exact_phase6_position_cap12"
    )
    assert result["selected"]["experiment_id"] == "control"
    assert result["phase6_evidence_inherited"]["global_capacity_replayed"] is False


def test_frontier_discards_large_replay_arrays_after_exact_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _frontier_settings()

    def fake_exact(_bundle, _trial, **_kwargs):
        return {
            **_fake_exact_result(total_r=90.0, drawdown=0.09, score=0.7),
            "trades": {"discovery": [{"large": "payload"}]},
            "equity_curves": {"discovery": [{"large": "payload"}]},
        }

    monkeypatch.setattr(residual_phases, "run_exact_fold_evaluation", fake_exact)
    result = residual_phases.run_risk_notional_frontier_phase(
        _FakeFrontierBundle(frozen_history_cache={}),
        settings,
        max_workers=2,
    )

    assert all("trades" not in row["result"] for row in result["experiments"])
    assert all(
        "equity_curves" not in row["result"] for row in result["experiments"]
    )


@pytest.mark.parametrize(
    ("changed_decisions", "expected_selected"),
    ((15, "combined_stale_replacement"), (5, "control")),
)
def test_capacity_neutral_phase_promotes_only_with_adequate_both_fold_evidence(
    monkeypatch: pytest.MonkeyPatch,
    changed_decisions: int,
    expected_selected: str,
) -> None:
    settings = replace(_frontier_settings(), daily_residual_max_positions=12)

    def payload(trial, *, cost_bps: float) -> dict:
        improved = trial.daily_residual_replacement_mode == "combined" and not (
            trial.daily_residual_replacement_loss_only
        )
        fold_r = 48.0 if improved else 45.0
        score = 0.80 if improved else 0.70
        changed = changed_decisions if trial.daily_residual_replacement_mode != "disabled" else 0
        return {
            "research_anchor_eligible": True,
            "continuous_metrics": {
                "total_r": 2.0 * fold_r,
                "max_drawdown_pct": 0.09,
                "return_pct": 0.25,
                "trades": 340,
            },
            "folds": {
                "discovery": {"total_r": fold_r},
                "calibration": {"total_r": fold_r},
            },
            "immutable_score": {"score": score - (0.01 if cost_bps == 30.0 else 0.0)},
            "capacity_neutral_replacement_diagnostics": {
                "folds": {
                    "discovery": {"changed_decisions": changed},
                    "calibration": {"changed_decisions": changed},
                }
            },
        }

    def fake_frontier(_bundle, variants, **_kwargs):
        rows = [
            {
                "experiment_id": name,
                "settings": trial,
                "result": payload(trial, cost_bps=20.0),
                "selection_eligible": True,
                "aspirational_targets": {},
                "result_source": "fake_exact",
            }
            for name, trial in variants.items()
        ]
        control = next(row for row in rows if row["experiment_id"] == "control")
        return {"experiments": rows, "control": control}

    monkeypatch.setattr(residual_phases, "_run_settings_frontier", fake_frontier)
    monkeypatch.setattr(
        residual_phases,
        "run_exact_fold_evaluation",
        lambda _bundle, trial, **_kwargs: payload(trial, cost_bps=30.0),
    )

    result = residual_phases.run_capacity_neutral_alpha_recycling_phase(
        _FakeFrontierBundle(frozen_history_cache={}),
        settings,
        max_workers=2,
    )

    assert result["selected"]["experiment_id"] == expected_selected
    assert result["capacity_contract"]["capacity_expansion"] is False
    assert result["score_component_union_ceiling"] == 7


def test_winner_robustness_caps_extremes_without_deleting_valid_right_tail() -> None:
    # Six ordinary top-tail outcomes make the empirical 95th percentile 1R.
    # Removing five of them makes the residual total negative even though the
    # median, winsorized result and gross-positive contribution are all broad.
    values = [-0.2] * 44 + [0.1] * 50 + [1.0] * 6
    diagnostics = _winner_robustness(
        values,
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
    )
    assert sum(sorted(values)[:-5]) < 0.0
    assert diagnostics["top_5pct_winner_winsorized_r_per_month"] > 0.0
    assert diagnostics["median_r_multiple"] > 0.0
    assert diagnostics["top_5pct_positive_r_share"] < 0.50
    assert _winner_robustness_passes(
        {"discovery": diagnostics, "calibration": diagnostics}
    )


def test_winner_robustness_rejects_true_lottery_ticket_concentration() -> None:
    values = [-0.1] * 95 + [3.0] * 5
    diagnostics = _winner_robustness(
        values,
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
    )
    assert diagnostics["top_5pct_positive_r_share"] == pytest.approx(1.0)
    assert not _winner_robustness_passes(
        {"discovery": diagnostics, "calibration": diagnostics}
    )


def test_round2_candidate_preserves_quality_floor_and_frozen_six_r_stop() -> None:
    candidate = discovery.Candidate(
        candidate_id="round2_test",
        residual_z_floor=1.0,
        holding_sessions=10,
        max_positions=10,
        max_positions_per_sector=2,
        formation_sessions=1,
        factor_model="market_sector_peer",
        score_components=("volume_transition", "price_rejection_recovery"),
        ranking_score_components=("volume_transition",),
        minimum_score=25.0,
        catastrophic_stop_residual_r=6.0,
    )
    settings = settings_from_discovery_candidate(candidate.__dict__)
    assert settings.daily_residual_minimum_score == 25.0
    assert settings.daily_residual_catastrophic_stop_residual_r == 6.0
    assert len(settings.daily_residual_score_components) == 2
    assert settings.daily_residual_ranking_score_components == (
        "volume_transition",
    )


def test_official_local_snapshot_needs_no_acquisition_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phased, "REPO_ROOT", tmp_path)
    required = {
        *phased.BACKTESTED_INTRADAY_STOCK_SYMBOLS,
        "SPY",
        *discovery.SECTOR_ETFS.values(),
    }
    assert len(required) == 110
    for symbol in required:
        (tmp_path / f"{symbol}_1d.parquet").write_bytes(symbol.encode("utf-8"))

    authority = phased._attest_retained_local_research_snapshot(tmp_path)

    assert authority["research_snapshot_certified"] is True
    assert authority["production_promotion_eligible"] is True
    assert authority["acquisition_receipts_required"] is False
    assert authority["broker_connection_required"] is False
    assert authority["fingerprinted_daily_dataset_count"] == 110
