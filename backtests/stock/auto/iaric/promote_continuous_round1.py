"""Safely promote a verified continuous-state IARIC baseline to Round 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.stock.auto.runners.run_iaric_residual_baseline_diagnostics import (
    finalize_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
IARIC_ROOT = REPO_ROOT / "backtests/output/stock/iaric"
DEFAULT_SOURCE = IARIC_ROOT / "continuous_baseline_reconciliation_v1/representative_baseline"


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _verify_source(source: Path) -> dict[str, Any]:
    required = (
        "artifact_manifest.json",
        "baseline_config.json",
        "baseline_data_contract.json",
        "final_metrics.json",
        "final_fold_metrics.json",
        "final_trades.json",
        "round_final_diagnostics.txt",
        "run_summary.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"source baseline is incomplete: {missing}")
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    if not bool(summary.get("representative_alpha_baseline")):
        raise RuntimeError("source is not classified as a representative alpha baseline")
    if bool(summary.get("promotion_ready")):
        raise RuntimeError("starting baseline must not be mislabeled promotion-ready")
    if bool(summary.get("locked_validation_accessed")) or bool(summary.get("holdout_accessed")):
        raise RuntimeError("source accessed protected data")
    if int(summary.get("traded_universe_count", 0)) != 98:
        raise RuntimeError("source does not use the exact 98-name universe")
    manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in manifest["artifacts"].items():
        path = source / name
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"source artifact hash mismatches: {mismatches}")
    return summary


def promote(source: Path, *, archive_name: str | None = None) -> Path:
    source = source.resolve()
    iaric_root = IARIC_ROOT.resolve()
    round_dir = (iaric_root / "round_1").resolve()
    archive_root = (iaric_root / "archive").resolve()
    if not _inside(source, iaric_root):
        raise RuntimeError("source is outside the IARIC output root")
    if not _inside(round_dir, iaric_root) or not _inside(archive_root, iaric_root):
        raise RuntimeError("promotion targets are outside the IARIC output root")
    summary = _verify_source(source)
    stamp = archive_name or (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_pre_continuous_reconciled_round1"
    )
    archive_dir = (archive_root / stamp).resolve()
    if not _inside(archive_dir, archive_root):
        raise RuntimeError("archive target is outside the archive root")
    if archive_dir.exists():
        raise FileExistsError(archive_dir)
    archive_dir.mkdir(parents=True)
    if round_dir.exists():
        shutil.move(str(round_dir), str(archive_dir / "round_1"))
    root_manifest_path = iaric_root / "rounds_manifest.json"
    if root_manifest_path.is_file():
        shutil.copy2(root_manifest_path, archive_dir / "rounds_manifest.json")
    shutil.copytree(source, round_dir)

    selection_receipt = source.parent / "selected_baseline_config.json"
    if selection_receipt.is_file():
        shutil.copy2(selection_receipt, round_dir / "selection_receipt.json")
    baseline_config = json.loads((round_dir / "baseline_config.json").read_text(encoding="utf-8"))
    _write_json(
        round_dir / "optimized_config.json",
        {
            **baseline_config,
            "configuration_role": "continuous_reconciled_representative_starting_baseline",
            "promotion_ready": False,
        },
    )
    folds = summary["fold_metrics"]
    restart = summary["independent_restart_stress"]
    metrics = summary["metrics"]
    run_spec = {
        "contract_id": summary["contract_id"],
        "cost_bps_round_trip": 20.0,
        "evaluation_contract": summary["fold_evaluation_contract"],
        "family": "stock",
        "holdout_accessed": False,
        "locked_validation_accessed": False,
        "round": 1,
        "selection_window": summary["window"],
        "strategy": "iaric",
        "traded_universe_count": 98,
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    artifact_names = {
        "artifact_manifest": "round_1/artifact_manifest.json",
        "baseline_config": "round_1/baseline_config.json",
        "data_contract": "round_1/baseline_data_contract.json",
        "final_concentration": "round_1/final_concentration.json",
        "final_cost_stress": "round_1/final_cost_stress.json",
        "final_equity_curve": "round_1/final_equity_curve.json",
        "final_exits": "round_1/final_exits.json",
        "final_fold_metrics": "round_1/final_fold_metrics.json",
        "final_metrics": "round_1/final_metrics.json",
        "final_monthly": "round_1/final_monthly.json",
        "final_score_diagnostics": "round_1/final_score_diagnostics.json",
        "final_symbols": "round_1/final_symbols.json",
        "final_trades": "round_1/final_trades.json",
        "full_final_diagnostics": "round_1/round_final_diagnostics.txt",
        "optimized_config": "round_1/optimized_config.json",
        "run_spec": "round_1/run_spec.json",
        "run_summary": "round_1/run_summary.json",
        "selection_receipt": "round_1/selection_receipt.json",
    }
    round_manifest = {
        "active": True,
        "artifacts": artifact_names,
        "baseline_eligible": True,
        "configuration_role": "continuous_reconciled_representative_starting_baseline",
        "contract_id": summary["contract_id"],
        "data_authority": "project_official_local_snapshot",
        "execution_contract": metrics["shared_core_contract"],
        "family": "stock",
        "headline": (
            f"{metrics['trades']} trades, {metrics['total_r']:+.2f}R, "
            f"PF {metrics['profit_factor']:.2f}, {metrics['trades_per_month']:.2f} trades/month"
        ),
        "known_optimization_targets": [
            "discovery_fold_top_tail_score_ordering",
            "catastrophic_stop_value_destruction",
            "profitable_mfe_surrender",
            "february_march_2025_regime_drawdown",
        ],
        "metrics": {
            "average_r": metrics["average_r"],
            "max_drawdown_fraction": metrics["max_drawdown_pct"],
            "profit_factor": metrics["profit_factor"],
            "total_r": metrics["total_r"],
            "total_trades": metrics["trades"],
            "trades_per_month": metrics["trades_per_month"],
        },
        "official": True,
        "promotion_ready": False,
        "representative_alpha_baseline": True,
        "restart_sensitivity": restart,
        "round": 1,
        "sealed_holdout": {"accessed": False, "start": "2026-03-02"},
        "selection_folds": {
            name: {
                "average_r": row["average_r"],
                "portfolio_return": row["return_pct"],
                "profit_factor": row["profit_factor"],
                "total_r": row["total_r"],
                "trades": row["trades"],
            }
            for name, row in folds.items()
        },
        "selection_fold_qualification_complete": bool(
            summary["selection_fold_qualification_complete"]
        ),
        "starting_baseline": True,
        "status": "complete_round1_continuous_representative_starting_baseline",
        "strategy": "iaric",
        "training_window": summary["window"],
        "validation": {
            "continuous_shared_capital": True,
            "costs_bps": 20.0,
            "locked_validation_accessed": False,
            "positive_continuous_calibration_equity_return": True,
            "positive_in_both_purged_selection_cohorts": True,
            "score_top_tail_separation_passed_each_fold": False,
        },
    }
    _write_json(round_dir / "round_manifest.json", round_manifest)
    finalize_artifact_manifest(round_dir, contract_id=str(summary["contract_id"]))

    now = datetime.now(timezone.utc).isoformat()
    hashes = {
        "artifact_manifest": _sha256(round_dir / "artifact_manifest.json"),
        "data_contract": _sha256(round_dir / "baseline_data_contract.json"),
        "final_metrics": _sha256(round_dir / "final_metrics.json"),
        "full_final_diagnostics": _sha256(round_dir / "round_final_diagnostics.txt"),
        "optimized_config": _sha256(round_dir / "optimized_config.json"),
        "round_manifest": _sha256(round_dir / "round_manifest.json"),
        "run_summary": _sha256(round_dir / "run_summary.json"),
    }
    root_manifest = {
        "active_round": 1,
        "family": "stock",
        "generated_at_utc": now,
        "latest_archive": f"archive/{stamp}",
        "rounds": [
            {
                "active": True,
                "artifacts": {**artifact_names, "round_manifest": "round_1/round_manifest.json"},
                "artifact_sha256": hashes,
                "average_r": metrics["average_r"],
                "baseline_eligible": True,
                "configuration_role": round_manifest["configuration_role"],
                "contract_id": summary["contract_id"],
                "data_authority": "project_official_local_snapshot",
                "headline": round_manifest["headline"],
                "immutable_score": summary["immutable_score"]["score"],
                "immutable_score_component_count": 7,
                "max_drawdown_fraction": metrics["max_drawdown_pct"],
                "official": True,
                "profit_factor": metrics["profit_factor"],
                "promotion_allowed": False,
                "representative_alpha_baseline": True,
                "restart_sensitivity": restart,
                "round": 1,
                "sealed_holdout": round_manifest["sealed_holdout"],
                "status": round_manifest["status"],
                "strategy_score_component_count": len(
                    summary["baseline_config"]["daily_residual_score_components"]
                ),
                "timestamp": now,
                "total_r": metrics["total_r"],
                "total_trades": metrics["trades"],
                "trades_per_month": metrics["trades_per_month"],
                "training_window": summary["window"],
                "validation_contract": round_manifest["validation"],
            }
        ],
        "strategy": "iaric",
    }
    _write_json(root_manifest_path, root_manifest)
    return archive_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--archive-name")
    args = parser.parse_args()
    archive = promote(args.source, archive_name=args.archive_name)
    print(json.dumps({"status": "promoted", "archive": str(archive)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
