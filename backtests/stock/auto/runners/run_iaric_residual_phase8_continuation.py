"""Continue the frozen IARIC residual round at Phase 8 without rerunning 1-7."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backtests.stock.auto.iaric.final_diagnostics import (
    write_blocked_round_final_diagnostics,
    write_round_final_diagnostics,
)
from backtests.stock.auto.iaric.residual_phases import (
    run_capacity_neutral_alpha_recycling_phase,
    run_exit_capture_frontier_phase,
    run_final_alpha_synergy_phase,
    run_final_robustness_and_target_assessment_phase,
    run_path_causal_profit_retention_phase,
    run_quality_aperture_phase,
    run_risk_notional_frontier_phase,
    run_selective_sector_overflow_phase,
)
from backtests.stock.auto.iaric.representative_contract import (
    HOLDOUT_START,
    LOCKED_VALIDATION_START,
    PHASE_ORDER,
    assess_input_authority,
)
from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.auto.runners import run_iaric_residual_phased_auto as base
from backtests.stock.engine.iaric_daily_residual_replay import (
    build_daily_residual_replay_bundle,
)
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS


DEFAULT_SOURCE = (
    base.REPO_ROOT
    / "backtests/output/stock/iaric/round_2/phased_auto_alpha_v3_robust_breadth"
)
DEFAULT_OUTPUT = (
    base.REPO_ROOT
    / "backtests/output/stock/iaric/round_2/phased_auto_alpha_v5_selective_sector_overflow"
)

PRIOR_ARTIFACTS = (
    "round_2_baseline_lineage.json",
    "phase_0_price_data_integrity_and_parity.json",
    "phase_1_residual_model_and_horizon_atlas.json",
    "phase_2_executable_candidate_registry.json",
    "phase_2_control_leg_registry.json",
    "phase_2_feature_profiles.json",
    "phase_3_selection_contract_robustness.json",
    "phase_4_causal_entry_delivery.json",
    "phase_4a_exact_screen_completion_and_pareto.json",
    "phase_4b_mechanism_aware_rejection_and_capacity_attribution.json",
    "phase_4c_two_stage_admission_and_ranking.json",
    "phase_5_residual_anchor_and_half_life_management.json",
    "phase_6_independent_sleeve_qualification_and_final_robustness.json",
    "phase_7_protected_integration_and_literal_ablation.json",
    "best_diagnostic_candidate.json",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_prior_lineage(source: Path, output: Path) -> None:
    missing = [name for name in PRIOR_ARTIFACTS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase 0-7 source artifacts: {missing}")
    for name in PRIOR_ARTIFACTS:
        shutil.copy2(source / name, output / name)


def _settings_from_compact(payload: dict[str, Any]) -> StrategySettings:
    """Hydrate the exact persisted frontier settings for an audited resume."""

    aliases = {
        "factor_model": "daily_residual_factor_model",
        "formation_sessions": "daily_residual_formation_sessions",
        "minimum_z": "daily_residual_minimum_z",
        "minimum_score": "daily_residual_minimum_score",
        "minimum_failed_continuation_r": (
            "daily_residual_minimum_failed_continuation_r"
        ),
        "lane_id": "daily_residual_lane_id",
        "minimum_sector_return_5d": "daily_residual_minimum_sector_return_5d",
        "minimum_market_trend_z_20d": (
            "daily_residual_minimum_market_trend_z_20d"
        ),
        "score_components": "daily_residual_score_components",
        "ranking_score_components": "daily_residual_ranking_score_components",
        "max_positions": "daily_residual_max_positions",
        "max_positions_per_sector": "daily_residual_max_positions_per_sector",
        "sector_overflow_slots": "daily_residual_sector_overflow_slots",
        "sector_overflow_minimum_score": (
            "daily_residual_sector_overflow_minimum_score"
        ),
        "sector_overflow_minimum_z": "daily_residual_sector_overflow_minimum_z",
        "sector_overflow_risk_multiplier": (
            "daily_residual_sector_overflow_risk_multiplier"
        ),
        "risk_fraction": "daily_residual_risk_fraction",
        "maximum_notional_fraction": "daily_residual_maximum_notional_fraction",
        "catastrophic_stop_atr": "daily_residual_catastrophic_stop_atr",
        "catastrophic_stop_residual_r": (
            "daily_residual_catastrophic_stop_residual_r"
        ),
        "partial_normalization_fraction": (
            "daily_residual_partial_normalization_fraction"
        ),
        "full_normalization_fraction": "daily_residual_full_normalization_fraction",
        "structural_failure_extension_fraction": (
            "daily_residual_structural_failure_extension_fraction"
        ),
        "profit_retention_activation_fraction": (
            "daily_residual_profit_retention_activation_fraction"
        ),
        "profit_retention_giveback_fraction": (
            "daily_residual_profit_retention_giveback_fraction"
        ),
        "replacement_mode": "daily_residual_replacement_mode",
        "replacement_loss_only": "daily_residual_replacement_loss_only",
        "replacement_minimum_held_sessions": (
            "daily_residual_replacement_minimum_held_sessions"
        ),
        "replacement_maximum_normalization_fraction": (
            "daily_residual_replacement_maximum_normalization_fraction"
        ),
        "replacement_minimum_score_margin": (
            "daily_residual_replacement_minimum_score_margin"
        ),
        "replacement_max_per_session": (
            "daily_residual_replacement_max_per_session"
        ),
        "maximum_holding_sessions": "daily_residual_maximum_holding_sessions",
        "partial_exit_fraction": "daily_residual_partial_exit_fraction",
    }
    values = {
        strategy_name: payload[compact_name]
        for compact_name, strategy_name in aliases.items()
        if compact_name in payload
    }
    for name in (
        "daily_residual_score_components",
        "daily_residual_ranking_score_components",
    ):
        if name in values:
            values[name] = tuple(values[name])
    return replace(
        StrategySettings(),
        strategy_mode="daily_residual_reversion",
        **values,
    )


def _block(
    output: Path,
    *,
    status: str,
    phase_index: int,
    blocker: Any,
    frozen_candidate: dict[str, Any],
    lineage: dict[str, Any],
) -> int:
    summary = {
        "status": status,
        "representative_reversion_baseline_eligible": True,
        "research_baseline_eligible": True,
        "phase_order": list(PHASE_ORDER),
        "last_completed_phase": PHASE_ORDER[phase_index],
        "current_phase": PHASE_ORDER[phase_index],
        "blocker": blocker,
        "frozen_candidate": frozen_candidate,
        "phase_8_continuation_lineage": lineage,
        "locked_validation_accessed": False,
        "holdout_accessed": False,
        "max_workers": 2,
    }
    base._write_json(output / "run_summary.json", summary)
    write_blocked_round_final_diagnostics(output)
    base._status(output, status, blocker=blocker)
    return 2


def run(
    output: Path,
    source: Path,
    data_dir: Path,
    *,
    max_workers: int = 2,
    resume_after_phase8: bool = False,
    resume_after_phase: int | None = None,
) -> int:
    if max_workers != 2:
        raise ValueError("IARIC Phase-8 continuation requires max-workers=2")
    source = source.resolve()
    output = output.resolve()
    if output == source:
        raise ValueError("continuation output must differ from its frozen source")
    output.mkdir(parents=True, exist_ok=True)
    if resume_after_phase8 and resume_after_phase is not None:
        raise ValueError("use either --resume-after-phase8 or --resume-after-phase")
    completed_phase = 8 if resume_after_phase8 else resume_after_phase
    if completed_phase is not None and completed_phase not in range(8, 15):
        raise ValueError("resume phase must be between 8 and 14")
    is_resume = completed_phase is not None
    phase_artifacts = {
        8: "phase_8_selective_sector_overflow_and_displacement_quality.json",
        9: "phase_9_quality_aperture_and_discrimination.json",
        10: "phase_10_risk_and_notional_frontier.json",
        11: "phase_11_exit_capture_frontier.json",
        12: "phase_12_final_alpha_frequency_synergy.json",
        13: "phase_13_path_causal_profit_retention.json",
        14: "phase_14_capacity_neutral_alpha_recycling.json",
    }
    phase8_path = output / phase_artifacts[8]
    if is_resume:
        completed_path = output / phase_artifacts[int(completed_phase)]
        if not completed_path.is_file():
            raise FileNotFoundError(
                f"resume requires completed Phase-{completed_phase} artifact"
            )
        next_artifact = phase_artifacts.get(int(completed_phase) + 1)
        if next_artifact and (output / next_artifact).exists():
            raise FileExistsError(
                f"Phase-{int(completed_phase) + 1} output already exists; "
                "refusing ambiguous resume"
            )
    if not is_resume and phase8_path.exists():
        raise FileExistsError(f"Phase 8 output already exists: {output}")

    if not is_resume:
        _copy_prior_lineage(source, output)
    source_summary = _load(source / "run_summary.json")
    source_phase0 = _load(source / "phase_0_price_data_integrity_and_parity.json")
    source_phase7 = _load(
        source / "phase_7_protected_integration_and_literal_ablation.json"
    )
    source_phase6 = _load(
        source
        / "phase_6_independent_sleeve_qualification_and_final_robustness.json"
    )
    if not source_phase7.get("passed"):
        raise ValueError("source Phase 7 did not pass")

    frozen_path = source / "frozen_selection_candidate.json"
    _candidate, normalized_lineage = base._load_round2_baseline(frozen_path)
    final_settings = StrategySettings(**normalized_lineage["settings"])
    frozen_candidate = _load(frozen_path)
    # Reconcile the predecessor against its literal historical payload.  New
    # behavior-preserving fields introduced for Phase 8+ must not invalidate a
    # candidate frozen before those fields existed.
    settings_sha = base._payload_sha256(frozen_candidate["settings"])
    if settings_sha != frozen_candidate.get("settings_sha256"):
        raise ValueError("frozen Phase-7 settings checksum does not reconcile")
    if not is_resume:
        shutil.copy2(frozen_path, output / "frozen_selection_candidate.json")

    inherited_phase6_cap12 = source_phase6.get("neighbourhood", {}).get(
        "position_cap_12"
    )
    source_phase6_control = source_phase6.get("base_20bps") or {}
    if not inherited_phase6_cap12 or not bool(
        inherited_phase6_cap12.get("research_anchor_eligible")
    ):
        raise ValueError(
            "continuation requires the exact eligible Phase-6 position_cap_12 result"
        )
    inherited_score = float(
        inherited_phase6_cap12["immutable_score"]["score"]
    )
    source_control_score = float(source_phase6_control["immutable_score"]["score"])
    if inherited_score <= source_control_score:
        raise ValueError(
            "Phase-6 position_cap_12 did not strictly improve the immutable score"
        )
    final_settings = replace(
        final_settings,
        daily_residual_max_positions=12,
        daily_residual_max_positions_per_sector=2,
        daily_residual_sector_overflow_slots=0,
    )
    resumed_phase: dict[str, Any] | None = None
    if is_resume:
        completed_path = output / phase_artifacts[int(completed_phase)]
        resumed_phase = _load(completed_path)
        selected = resumed_phase.get("selected")
        if not selected or resumed_phase.get("status") not in {
            "passed",
            "complete_control_retained",
        }:
            raise ValueError(
                f"Phase-{completed_phase} artifact is not a completed frontier"
            )
        final_settings = _settings_from_compact(dict(selected["settings"]))

    base._status(
        output,
        (
            f"attesting_phase_{int(completed_phase) + 1}_resume_inputs"
            if is_resume
            else "attesting_phase_8_continuation_inputs"
        ),
        source_output=str(source),
        locked_validation_accessed=False,
        holdout_accessed=False,
    )
    authority = base._attest_retained_local_research_snapshot(data_dir)
    assessment = assess_input_authority(authority["input_authority"])
    if not assessment["representative_reversion_baseline_eligible"]:
        blocker = "retained research snapshot no longer passes input authority"
        lineage = {"source_output": str(source), "data_reconciled": False}
        return _block(
            output,
            status="blocked_phase_8_input_authority",
            phase_index=11,
            blocker=blocker,
            frozen_candidate=frozen_candidate,
            lineage=lineage,
        )

    close, open_, high, low, volume, sectors, paths = base._load_research_panel(
        data_contract=base.RETAINED_LOCAL_RESEARCH,
        data_dir=data_dir,
        selection_bundle_path=None,
    )
    data_fingerprint, _fingerprint_rows = discovery._selection_data_fingerprint(
        close, open_, high, low, volume, paths
    )
    if data_fingerprint != source_phase0["data_fingerprint"]:
        blocker = {
            "message": "selection data changed since the frozen Phase-7 candidate",
            "source": source_phase0["data_fingerprint"],
            "current": data_fingerprint,
        }
        lineage = {"source_output": str(source), "data_reconciled": False}
        return _block(
            output,
            status="blocked_phase_8_data_fingerprint_drift",
            phase_index=11,
            blocker=blocker,
            frozen_candidate=frozen_candidate,
            lineage=lineage,
        )

    bundle = build_daily_residual_replay_bundle(
        close,
        open_,
        high,
        low,
        volume,
        sectors,
        factor_model=final_settings.daily_residual_factor_model,
        source_fingerprint=data_fingerprint,
    )
    code_fingerprint = base._sha256_files(
        [
            Path(__file__),
            Path(base.__file__),
            base.REPO_ROOT / "backtests/stock/auto/iaric/residual_phases.py",
            base.REPO_ROOT / "backtests/stock/auto/iaric/representative_contract.py",
            base.REPO_ROOT / "backtests/stock/engine/iaric_daily_residual_replay.py",
            base.REPO_ROOT / "strategies/stock/iaric/config.py",
            base.REPO_ROOT / "strategies/stock/iaric/core/daily_residual.py",
        ]
    )
    lineage = {
        "contract": "phase6_exact_cap12_inheritance_then_novel_phase8_v2",
        "source_output": str(source),
        "source_status": source_summary.get("status"),
        "source_frozen_candidate_sha256": base._sha256_path(frozen_path),
        "source_settings_sha256": settings_sha,
        "phase6_inherited_experiment": "position_cap_12",
        "phase6_inherited_result_path": (
            "phase_6_independent_sleeve_qualification_and_final_robustness.json"
            "#/neighbourhood/position_cap_12"
        ),
        "phase6_inherited_immutable_score": inherited_score,
        "phase6_source_control_immutable_score": source_control_score,
        "starting_settings": base._settings_payload(final_settings),
        "phase6_cap12_replayed_in_phase8": False,
        "selection_data_fingerprint": data_fingerprint,
        "data_reconciled": True,
        "starting_phase": PHASE_ORDER[11],
        "phases_1_through_7_rerun": False,
        "source_locked_validation_previously_accessed": bool(
            source_summary.get("locked_validation_accessed")
        ),
        "source_locked_metrics_used_for_phase_8_through_13_selection": False,
        "locked_validation_accessed": False,
        "holdout_accessed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if is_resume:
        original_lineage_path = output / "phase_8_continuation_lineage.json"
        original_lineage = _load(original_lineage_path)
        resume_receipt = {
            "contract": "resume_after_completed_frontier_phase_v2",
            "reason": "load_new_registered_phase_without_repeating_completed_work",
            "completed_phase": int(completed_phase),
            "completed_artifact": phase_artifacts[int(completed_phase)],
            "completed_artifact_sha256": base._sha256_path(
                output / phase_artifacts[int(completed_phase)]
            ),
            "selected_experiment": resumed_phase["selected"][
                "experiment_id"
            ],
            "selected_settings_sha256": base._payload_sha256(
                base._settings_payload(final_settings)
            ),
            "completed_phases_rerun": False,
            "resume_starting_phase": PHASE_ORDER[11 + int(completed_phase) - 7],
            "max_workers": max_workers,
            "memory_bounding_contract": (
                "frontier_workers_discard_trade_and_equity_arrays_after_exact_scoring"
            ),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        resume_path = output / f"phase_{int(completed_phase) + 1}_resume_lineage.json"
        base._write_json(resume_path, resume_receipt)
        lineage = {
            **original_lineage,
            f"phase_{int(completed_phase) + 1}_resume": resume_receipt,
        }
    else:
        base._write_json(output / "phase_8_continuation_lineage.json", lineage)

    frontier_specs: tuple[
        tuple[int, str, str, Callable[..., dict[str, Any]]], ...
    ] = (
        (
            8,
            "selective_sector_overflow_and_displacement_quality",
            "phase_8_selective_sector_overflow_and_displacement_quality.json",
            run_selective_sector_overflow_phase,
        ),
        (
            9,
            "quality_aperture_and_discrimination",
            "phase_9_quality_aperture_and_discrimination.json",
            run_quality_aperture_phase,
        ),
        (
            10,
            "risk_and_notional_frontier",
            "phase_10_risk_and_notional_frontier.json",
            run_risk_notional_frontier_phase,
        ),
        (
            11,
            "exit_capture_frontier",
            "phase_11_exit_capture_frontier.json",
            run_exit_capture_frontier_phase,
        ),
        (
            12,
            "final_alpha_frequency_synergy",
            "phase_12_final_alpha_frequency_synergy.json",
            run_final_alpha_synergy_phase,
        ),
        (
            13,
            "path_causal_profit_retention",
            "phase_13_path_causal_profit_retention.json",
            run_path_causal_profit_retention_phase,
        ),
        (
            14,
            "capacity_neutral_alpha_recycling",
            "phase_14_capacity_neutral_alpha_recycling.json",
            run_capacity_neutral_alpha_recycling_phase,
        ),
    )
    if is_resume:
        frontier_specs = tuple(
            spec for spec in frontier_specs if spec[0] > int(completed_phase)
        )
    post_phase7_lineage: dict[str, str] = {}
    if is_resume:
        for phase_number in range(8, int(completed_phase) + 1):
            artifact = _load(output / phase_artifacts[phase_number])
            post_phase7_lineage[f"phase_{phase_number}"] = artifact[
                "selected"
            ]["experiment_id"]
    aspirational_target_contract: dict[str, Any] | None = None
    if is_resume and int(completed_phase) >= 12:
        phase12_artifact = _load(output / phase_artifacts[12])
        aspirational_target_contract = dict(
            phase12_artifact["aspirational_target_contract"]
        )
    for phase_number, phase_name, filename, phase_function in frontier_specs:
        phase_index = 11 + (phase_number - 8)
        base._status(output, f"running_phase_{phase_number}_{phase_name}")
        phase_kwargs: dict[str, Any] = {}
        if phase_number == 8:
            phase_kwargs["inherited_control_result"] = inherited_phase6_cap12
        result = phase_function(
            bundle,
            final_settings,
            max_workers=max_workers,
            score_contract="round2",
            **phase_kwargs,
        )
        base._write_json(
            output / filename,
            {
                **base._compact_settings_frontier(result),
                "locked_validation_accessed": False,
                "holdout_accessed": False,
            },
        )
        if result.get("selected_settings") is None:
            blocker = {
                "message": f"No exact eligible candidate remained after Phase {phase_number}.",
                "maximum_selection_drawdown": result.get(
                    "maximum_selection_drawdown"
                ),
                "aspirational_targets_are_hard_gates": False,
            }
            return _block(
                output,
                status=f"blocked_phase_{phase_number}_no_exact_eligible_candidate",
                phase_index=phase_index,
                blocker=blocker,
                frozen_candidate=frozen_candidate,
                lineage=lineage,
            )
        post_phase7_lineage[f"phase_{phase_number}"] = result["selected"][
            "experiment_id"
        ]
        if phase_number == 12:
            aspirational_target_contract = dict(
                result["aspirational_target_contract"]
            )
        final_settings = result["selected_settings"]
        del result
        gc.collect()
    if aspirational_target_contract is None:
        raise RuntimeError("Phase 12 did not publish its aspirational target contract")
    frozen_candidate.update(
        {
            "settings": base._settings_payload(final_settings),
            "settings_sha256": base._payload_sha256(
                base._settings_payload(final_settings)
            ),
            "post_phase_7_experiment_lineage": post_phase7_lineage,
            "aspirational_target_contract": aspirational_target_contract,
            "phase_8_continuation_lineage": lineage,
        }
    )
    base._write_json(output / "frozen_selection_candidate.json", frozen_candidate)

    base._status(
        output, "running_phase_15_final_robustness_and_target_assessment"
    )
    phase15 = run_final_robustness_and_target_assessment_phase(
        bundle,
        final_settings,
        max_workers=max_workers,
        score_contract="round2",
    )
    base._write_json(
        output / "phase_15_final_robustness_and_target_assessment.json",
        {
            **phase15,
            "base_20bps": base._compact_exact(phase15["base_20bps"]),
            "cost_stress_30bps": base._compact_exact(
                phase15["cost_stress_30bps"]
            ),
            "cost_stress_40bps": base._compact_exact(
                phase15["cost_stress_40bps"]
            ),
            "neighbourhood": {
                name: base._compact_exact(row)
                for name, row in phase15["neighbourhood"].items()
            },
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if not phase15["qualification"]["passed"]:
        blocker = {
            "message": "Final robustness or the 12% MTM drawdown safety ceiling failed.",
            "failed_gates": phase15["qualification"]["failed_gates"],
            "aspirational_target_assessment": phase15[
                "aspirational_target_assessment"
            ],
        }
        return _block(
            output,
            status="blocked_phase_15_final_robustness",
            phase_index=18,
            blocker=blocker,
            frozen_candidate=frozen_candidate,
            lineage=lineage,
        )

    base._status(
        output,
        "running_phase_16_chronological_validation_replay",
        locked_validation_accessed=True,
        holdout_accessed=False,
    )
    phase16 = base._run_locked_validation_once(
        output=output,
        data_contract=base.RETAINED_LOCAL_RESEARCH,
        data_dir=data_dir,
        selection_bundle_path=None,
        settings=final_settings,
    )
    phase16.update(
        {
            "evidence_class": (
                "chronological_validation_replay_after_predecessor_baseline_"
                "window_was_consumed"
            ),
            "untouched_validation": False,
            "candidate_mutations_selected_on_validation": False,
            "sealed_holdout_used": False,
        }
    )
    base._write_json(
        output / "phase_16_locked_chronological_validation.json", phase16
    )
    final_status = (
        "complete_chronological_validation_replay"
        if phase16["passed"]
        else "failed_chronological_validation_replay"
    )
    summary = {
        "status": final_status,
        "representative_reversion_baseline_eligible": bool(phase16["passed"]),
        "research_baseline_eligible": True,
        "production_promotion_eligible": False,
        "data_contract": base.RETAINED_LOCAL_RESEARCH,
        "data_authority_class": authority.get(
            "authority_class", "project_official_local_snapshot"
        ),
        "optimizer_class": "gated_price_volume_residual_phase8_continuation_v2",
        "terminal_artifact": "round_final_diagnostics.txt",
        "phase_order": list(PHASE_ORDER),
        "last_completed_phase": PHASE_ORDER[-1],
        "current_phase": None,
        "blocker": None if phase16["passed"] else phase16["failed_gates"],
        "frozen_candidate": frozen_candidate,
        "phase_8_continuation_lineage": lineage,
        "selection_fold_metrics": phase15["base_20bps"]["folds"],
        "selection_metrics": phase15["base_20bps"]["continuous_metrics"],
        "aspirational_target_met": bool(
            phase15["aspirational_target_assessment"]["both_met"]
        ),
        "aspirational_target_assessment": phase15[
            "aspirational_target_assessment"
        ],
        "locked_validation_metrics": phase16["metrics"],
        "residual_estimation_stock_symbols": len(sectors),
        "non_traded_explanatory_reference_symbols": 1
        + len(set(discovery.SECTOR_ETFS.values())),
        "tradable_execution_symbols": len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        "data_fingerprint": data_fingerprint,
        "code_fingerprint": code_fingerprint,
        "cache_hits": 0,
        "cache_entries": 0,
        "max_workers": max_workers,
        "locked_validation_start": LOCKED_VALIDATION_START,
        "locked_validation_accessed": True,
        "locked_validation_untouched": False,
        "holdout_start": HOLDOUT_START,
        "holdout_accessed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    base._write_json(output / "run_summary.json", summary)
    write_round_final_diagnostics(output)
    base._status(
        output,
        final_status,
        representative_reversion_baseline_eligible=summary[
            "representative_reversion_baseline_eligible"
        ],
        locked_validation_accessed=True,
        holdout_accessed=False,
    )
    return 0 if phase16["passed"] else 2


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=discovery.DEFAULT_DATA_DIR)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--resume-after-phase8",
        action="store_true",
        help="resume the same audited output at Phase 9 after Phase 8 committed",
    )
    parser.add_argument(
        "--resume-after-phase",
        type=int,
        choices=range(8, 15),
        help=(
            "resume after an exact completed Phase 8-14 artifact without "
            "replaying earlier phases"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    return run(
        args.output_dir,
        args.source_output,
        args.data_dir,
        max_workers=args.max_workers,
        resume_after_phase8=args.resume_after_phase8,
        resume_after_phase=args.resume_after_phase,
    )


if __name__ == "__main__":
    raise SystemExit(main())
