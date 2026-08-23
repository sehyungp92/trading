"""Reconcile and replace the IARIC baseline using one continuous portfolio path.

The prior exact selector added two independently reset fold results.  That is a
useful restart sensitivity test, but not an executable economic estimate for a
capacity-constrained overlapping-position strategy.  This bounded follow-up
replays only previously registered candidates on shared capital from discovery
through calibration, uses purged continuous-state cohorts for chronological
gates, and runs independent restart sensitivity only for the selected winner's
full diagnostics.

Locked validation and the sealed holdout are never loaded.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.iaric.residual_phases import (
    run_exact_fold_evaluation,
    settings_from_discovery_candidate,
)
from backtests.stock.auto.runners import run_iaric_daily_residual_discovery as discovery
from backtests.stock.auto.runners.run_iaric_residual_baseline_diagnostics import (
    run as run_full_diagnostics,
)
from backtests.stock.auto.runners.run_iaric_residual_exact_followup import (
    registered_followups,
)
from backtests.stock.engine.iaric_daily_residual_replay import (
    build_daily_residual_replay_bundle,
    run_daily_residual_replay,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests/output/stock/iaric/continuous_baseline_reconciliation_v1"
)
CONTRACT_ID = "iaric_residual_continuous_reconciled_exact98_v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _status(output: Path, status: str, **details: Any) -> None:
    _write_json(
        output / "background_status.json",
        {
            "status": status,
            "max_workers": 2,
            "evaluation_contract": (
                "continuous_shared_capital_with_purged_entry_cohorts_v1"
            ),
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _candidate(
    candidate_id: str,
    *,
    factor_model: str = "market_sector_peer",
    formation: int = 1,
    holding: int = 10,
    components: tuple[str, ...] = ("volume_transition",),
    minimum_failed_continuation_r: float = 0.0,
    market_floor: float = -8.0,
    stop_residual_r: float = 4.0,
) -> tuple[dict[str, Any], float]:
    return (
        {
            "candidate_id": candidate_id,
            "residual_z_floor": 1.0,
            "holding_sessions": holding,
            "max_positions": 10,
            "max_positions_per_sector": 2,
            "round_trip_cost_bps": 20.0,
            "formation_sessions": formation,
            "diagnostic_leg": "long_loser",
            "factor_model": factor_model,
            "score_components": list(components),
            "lane_id": (
                f"continuous_{factor_model}_residual_{formation}d"
            ),
            "minimum_failed_continuation_r": minimum_failed_continuation_r,
            "minimum_sector_return_5d": -0.15,
            "minimum_market_trend_z_20d": market_floor,
        },
        stop_residual_r,
    )


def registered_candidates() -> list[tuple[dict[str, Any], float]]:
    """Previously registered exact finalists and mechanism follow-ups only."""

    rows = [
        _candidate("official_anchor_volume_f1_h10"),
        _candidate(
            "volume_exhaustion_regime_f1_h10",
            components=("volume_exhaustion_quality", "regime_execution_quality"),
        ),
        _candidate(
            "volume_exhaustion_rejection_f1_h10",
            components=("volume_exhaustion_quality", "price_rejection_recovery"),
        ),
        _candidate(
            "market_sector_peer_failed_f1_h10",
            components=("failed_continuation",),
            minimum_failed_continuation_r=0.20,
        ),
        _candidate(
            "peer_demeaned_failed_f1_h10",
            factor_model="peer_demeaned",
            components=("failed_continuation",),
            minimum_failed_continuation_r=0.20,
        ),
        _candidate(
            "peer_demeaned_failed_f5_h10",
            factor_model="peer_demeaned",
            formation=5,
            components=("failed_continuation",),
            minimum_failed_continuation_r=0.20,
        ),
    ]
    rows.extend(registered_followups())
    deduplicated: dict[str, tuple[dict[str, Any], float]] = {}
    for candidate, stop in rows:
        deduplicated[str(candidate["candidate_id"])] = (candidate, stop)
    return list(deduplicated.values())


def _baseline_gates(exact: dict[str, Any]) -> dict[str, bool]:
    folds = exact["folds"]
    return {
        "positive_continuous_period": (
            float(exact["continuous_metrics"]["total_r"]) > 0.0
        ),
        "positive_purged_entry_cohort_each_fold": all(
            float(row["total_r"]) > 0.0 for row in folds.values()
        ),
        "positive_continuous_calibration_equity_return": (
            float(folds["calibration"]["return_pct"]) > 0.0
        ),
        "minimum_100_purged_entry_trades_each_fold": all(
            int(row["trades"]) >= 100 for row in folds.values()
        ),
        "issuer_and_sector_entry_risk_caps": all(
            float(row["top_issuer_entry_risk_share"]) <= 0.15
            and float(row["top_sector_entry_risk_share"]) <= 0.35
            for row in folds.values()
        ),
    }


def _compact_exact(exact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in exact.items()
        if key
        not in {
            "trades",
            "equity_curves",
            "independent_restart_trades",
        }
    }


def run(output: Path, data_dir: Path, *, max_workers: int = 2) -> int:
    if max_workers != 2:
        raise ValueError("continuous baseline reconciliation requires max-workers=2")
    output.mkdir(parents=True, exist_ok=True)
    _status(output, "loading_exact98_selection_panel")
    close, open_, high, low, volume, sectors, paths = discovery._load_daily_panel(
        data_dir
    )
    fingerprint, fingerprint_rows = discovery._selection_data_fingerprint(
        close, open_, high, low, volume, paths
    )
    registry = registered_candidates()
    factor_models = sorted({candidate["factor_model"] for candidate, _stop in registry})
    bundles = {}
    for index, factor_model in enumerate(factor_models, start=1):
        _status(
            output,
            "building_frozen_factor_bundles",
            current_factor_model=factor_model,
            completed_factor_models=index - 1,
            total_factor_models=len(factor_models),
        )
        bundles[factor_model] = build_daily_residual_replay_bundle(
            close,
            open_,
            high,
            low,
            volume,
            sectors,
            factor_model=factor_model,
            source_fingerprint=fingerprint,
        )
    _write_json(
        output / "candidate_registry.json",
        [
            {"candidate": candidate, "catastrophic_stop_residual_r": stop}
            for candidate, stop in registry
        ],
    )
    _status(output, "running_continuous_exact_candidates", candidates=len(registry))

    def evaluate(row: tuple[dict[str, Any], float]) -> dict[str, Any]:
        candidate, stop = row
        settings = replace(
            settings_from_discovery_candidate(candidate),
            daily_residual_catastrophic_stop_residual_r=float(stop),
        )
        bundle = replace(
            bundles[candidate["factor_model"]], frozen_history_cache={}
        )
        continuous = run_daily_residual_replay(
            bundle,
            settings,
            start=date.fromisoformat("2024-03-25"),
            end=date.fromisoformat("2025-07-31"),
            round_trip_cost_bps=20.0,
        )
        exact = run_exact_fold_evaluation(
            bundle,
            settings,
            round_trip_cost_bps=20.0,
            continuous_result=continuous,
        )
        return {
            "candidate": candidate,
            "catastrophic_stop_residual_r": stop,
            "settings_object": settings,
            "continuous_result": continuous,
            "exact": exact,
            "baseline_gates": _baseline_gates(exact),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(evaluate, row): row[0]["candidate_id"] for row in registry}
        for future in as_completed(futures):
            rows.append(future.result())
            _write_json(
                output / "continuous_candidate_results.partial.json",
                [
                    {
                        "candidate": completed["candidate"],
                        "catastrophic_stop_residual_r": completed[
                            "catastrophic_stop_residual_r"
                        ],
                        "baseline_gates": completed["baseline_gates"],
                        "exact": _compact_exact(completed["exact"]),
                    }
                    for completed in rows
                ],
            )
            _status(
                output,
                "running_continuous_exact_candidates",
                candidates=len(registry),
                candidates_completed=len(rows),
                last_completed_candidate_id=futures[future],
            )
    rows.sort(
        key=lambda row: (
            not all(row["baseline_gates"].values()),
            -float(row["exact"]["immutable_score"]["score"]),
            -float(row["exact"]["continuous_metrics"]["total_r"]),
            row["candidate"]["candidate_id"],
        )
    )
    _write_json(
        output / "continuous_candidate_results.json",
        [
            {
                "candidate": row["candidate"],
                "catastrophic_stop_residual_r": row[
                    "catastrophic_stop_residual_r"
                ],
                "baseline_gates": row["baseline_gates"],
                "exact": _compact_exact(row["exact"]),
            }
            for row in rows
        ],
    )

    # Independent fold restarts are a secondary sensitivity test, not an
    # executable candidate-selection objective.  Running them for three
    # finalists added six redundant replays.  The selected continuous winner's
    # full diagnostics performs this test once and records the gap explicitly.
    eligible = [row for row in rows if all(row["baseline_gates"].values())]
    if not eligible:
        _status(
            output,
            "complete_no_continuous_representative_baseline",
            best_candidate_id=rows[0]["candidate"]["candidate_id"],
            failed_baseline_gates=[
                name
                for name, passed in rows[0]["baseline_gates"].items()
                if not passed
            ],
        )
        return 2

    selected = eligible[0]
    _write_json(
        output / "selected_baseline_config.json",
        {
            "contract_id": CONTRACT_ID,
            "candidate": selected["candidate"],
            "catastrophic_stop_residual_r": selected[
                "catastrophic_stop_residual_r"
            ],
            "baseline_gates": selected["baseline_gates"],
            "selection_score": selected["exact"]["immutable_score"],
            "continuous_metrics": selected["exact"]["continuous_metrics"],
            "data_fingerprint": fingerprint,
            "fingerprinted_inputs": fingerprint_rows,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    _status(
        output,
        "materializing_full_continuous_representative_baseline",
        candidate_id=selected["candidate"]["candidate_id"],
    )
    run_full_diagnostics(
        output / "representative_baseline",
        data_dir,
        settings=selected["settings_object"],
        contract_id=CONTRACT_ID,
    )
    _status(
        output,
        "complete_continuous_representative_baseline",
        candidate_id=selected["candidate"]["candidate_id"],
        diagnostics_output=str((output / "representative_baseline").resolve()),
    )
    return 0


def finalize_existing_continuous_results(output: Path, data_dir: Path) -> int:
    """Materialize diagnostics without repeating the completed candidate pass."""

    results_path = output / "continuous_candidate_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    rows.sort(
        key=lambda row: (
            not all(row["baseline_gates"].values()),
            -float(row["exact"]["immutable_score"]["score"]),
            -float(row["exact"]["continuous_metrics"]["total_r"]),
            row["candidate"]["candidate_id"],
        )
    )
    eligible = [row for row in rows if all(row["baseline_gates"].values())]
    if not eligible:
        _status(
            output,
            "complete_no_continuous_representative_baseline",
            best_candidate_id=rows[0]["candidate"]["candidate_id"],
        )
        return 2
    selected = eligible[0]
    settings = replace(
        settings_from_discovery_candidate(selected["candidate"]),
        daily_residual_catastrophic_stop_residual_r=float(
            selected["catastrophic_stop_residual_r"]
        ),
    )
    _write_json(
        output / "selected_baseline_config.json",
        {
            "contract_id": CONTRACT_ID,
            "candidate": selected["candidate"],
            "catastrophic_stop_residual_r": selected[
                "catastrophic_stop_residual_r"
            ],
            "baseline_gates": selected["baseline_gates"],
            "selection_score": selected["exact"]["immutable_score"],
            "continuous_metrics": selected["exact"]["continuous_metrics"],
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    _status(
        output,
        "materializing_full_continuous_representative_baseline",
        candidate_id=selected["candidate"]["candidate_id"],
        resumed_from_completed_continuous_comparison=True,
    )
    run_full_diagnostics(
        output / "representative_baseline",
        data_dir,
        settings=settings,
        contract_id=CONTRACT_ID,
    )
    _status(
        output,
        "complete_continuous_representative_baseline",
        candidate_id=selected["candidate"]["candidate_id"],
        diagnostics_output=str((output / "representative_baseline").resolve()),
        resumed_from_completed_continuous_comparison=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=discovery.DEFAULT_DATA_DIR)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="skip candidate replays and finalize continuous_candidate_results.json",
    )
    args = parser.parse_args()
    if args.finalize_existing:
        return finalize_existing_continuous_results(
            args.output_dir.resolve(), args.data_dir.resolve()
        )
    return run(
        args.output_dir.resolve(),
        args.data_dir.resolve(),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
