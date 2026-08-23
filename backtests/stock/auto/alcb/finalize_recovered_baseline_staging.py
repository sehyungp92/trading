"""Finish metadata for an already replay-validated recovered Round 1 staging bundle."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.stock.auto.alcb.plugin import ALCBP16Plugin
from backtests.stock.auto.alcb.promote_recovered_baseline import _sha256, _write_json


ALCB_ROOT = (REPO_ROOT / "backtests/output/stock/alcb").resolve()
STAGING_DIR = ALCB_ROOT / ".round_1_recovered_staging"
DATA_DIR = (REPO_ROOT / "backtests/stock/data/raw").resolve()


def _generated_at_from_diagnostics(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines()[:10]:
        if line.startswith("Generated: "):
            return line.removeprefix("Generated: ").strip()
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_valid_staging(staging_dir: Path) -> dict[str, Any]:
    required = (
        "optimized_config.json",
        "final_metrics.json",
        "final_analysis.json",
        "final_trades.json",
        "final_monthly.json",
        "final_symbols.json",
        "final_exits.json",
        "final_entry_types.json",
        "round_final_diagnostics.txt",
        "round_evaluation.txt",
        "promotion_validation.json",
        "recovery_evidence/final_recovery_manifest.json",
    )
    missing = [name for name in required if not (staging_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete staging bundle: {missing}")
    validation = json.loads((staging_dir / "promotion_validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or not validation.get("fresh_reproduction"):
        raise RuntimeError("Staging bundle has not passed fresh reproduction")
    if validation.get("trade_rows") != int(json.loads(
        (staging_dir / "final_metrics.json").read_text(encoding="utf-8")
    )["total_trades"]):
        raise RuntimeError("Trade-row validation does not match final metrics")
    hash_checks = {
        "optimized_config_sha256": staging_dir / "optimized_config.json",
        "round_final_diagnostics_sha256": staging_dir / "round_final_diagnostics.txt",
    }
    for key, path in hash_checks.items():
        actual = _sha256(path)
        if validation.get(key) != actual:
            raise RuntimeError(f"Staging hash mismatch for {path.name}")
    return validation


def main() -> None:
    staging_dir = STAGING_DIR.resolve()
    if staging_dir.parent != ALCB_ROOT or staging_dir.name != STAGING_DIR.name:
        raise ValueError(f"Unsafe staging path: {staging_dir}")
    validation = _require_valid_staging(staging_dir)

    config = json.loads((staging_dir / "optimized_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((staging_dir / "final_metrics.json").read_text(encoding="utf-8"))
    recovery_manifest = json.loads(
        (staging_dir / "recovery_evidence/final_recovery_manifest.json").read_text(encoding="utf-8")
    )
    generated_at = _generated_at_from_diagnostics(staging_dir / "round_final_diagnostics.txt")
    archive_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_pre_recovered_round1")
    final_round_dir = ALCB_ROOT / "round_1"
    archive_dir = ALCB_ROOT / "archive" / archive_name
    diagnostics_sha = validation["round_final_diagnostics_sha256"]
    config_sha = validation["optimized_config_sha256"]

    plugin = ALCBP16Plugin(
        DATA_DIR,
        start_date="2024-03-25",
        end_date="2026-03-01",
        initial_equity=10_000.0,
        max_workers=1,
    )
    provenance = plugin.build_provenance()

    run_spec = {
        "family": "stock",
        "strategy": "alcb",
        "round": 1,
        "description": "Recovered RTH baseline promoted as the clean starting point for phased auto optimization",
        "generated_at_utc": generated_at,
        "execution_context": {
            "data_dir": str(DATA_DIR),
            "start_date": "2024-03-25",
            "end_date": "2026-03-01",
            "initial_equity": 10_000.0,
            "session_policy": config["intraday_session_policy"],
            "excluded_period_start": "2026-03-02",
            "excluded_period_accessed": False,
        },
        "baseline_mutations": {},
        "baseline_mutation_count": 0,
        "promoted_mutations": config,
        "promoted_mutation_count": len(config),
        "selected_candidate": recovery_manifest["selected"]["id"],
        "immutable_score": recovery_manifest["immutable_score"],
        "provenance": provenance,
        "provenance_status": recovery_manifest["status"],
        "data_authority": recovery_manifest["data_authority"],
    }
    _write_json(staging_dir / "run_spec.json", run_spec)

    artifact_names = (
        "optimized_config.json",
        "final_metrics.json",
        "final_analysis.json",
        "final_trades.json",
        "final_monthly.json",
        "final_symbols.json",
        "final_exits.json",
        "final_entry_types.json",
        "round_final_diagnostics.txt",
        "round_evaluation.txt",
        "promotion_validation.json",
        "run_spec.json",
    )
    artifacts = {name: str(final_round_dir / name) for name in artifact_names}
    artifacts["recovery_evidence"] = str(final_round_dir / "recovery_evidence")
    headline = {
        "total_trades": int(metrics["total_trades"]),
        "trades_per_month": float(metrics["trades_per_month"]),
        "win_rate": float(metrics["win_rate"]) * 100.0,
        "expected_total_r": float(metrics["expected_total_r"]),
        "net_profit": float(metrics["net_profit"]),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown_pct": float(metrics["max_drawdown_pct"]) * 100.0,
        "sharpe_ratio": float(metrics["sharpe"]),
        "calmar_ratio": float(metrics["calmar"]),
    }
    run_summary = {
        "family": "stock",
        "strategy": "alcb",
        "round": 1,
        "generated_at_utc": generated_at,
        "completed_phases": [],
        "recovery_round": True,
        "selected_candidate": recovery_manifest["selected"]["id"],
        "cumulative_mutations": config,
        "mutation_count": len(config),
        "final_metrics": metrics,
        "headline_metrics": headline,
        "artifacts": artifacts,
        "provenance": provenance,
        "provenance_status": recovery_manifest["status"],
        "data_authority": recovery_manifest["data_authority"],
        "archive_of_previous_active_results": str(archive_dir),
        "promotion_validation": validation,
    }
    _write_json(staging_dir / "run_summary.json", run_summary)

    round_entry = {
        "round": 1,
        "timestamp": generated_at,
        "mutations": config,
        "mutations_count": len(config),
        **headline,
        "net_return_pct": None,
        "selected_candidate": recovery_manifest["selected"]["id"],
        "selection_fingerprint": recovery_manifest["data_source_fingerprint"],
        "diagnostics_fingerprint": diagnostics_sha,
        "optimized_config_sha256": config_sha,
        "provenance_schema_version": 1,
        "provenance_status": recovery_manifest["status"],
        "data_authority": recovery_manifest["data_authority"],
        "round_dir": str(final_round_dir),
        "source_recovery_manifest": str(final_round_dir / "recovery_evidence/final_recovery_manifest.json"),
        "archive_of_previous_active_results": str(archive_dir),
        "excluded_period": recovery_manifest["excluded_period"],
    }
    root_manifest = {"family": "stock", "rounds": [round_entry], "strategy": "alcb"}
    _write_json(staging_dir / "round_manifest_entry.json", round_entry)
    _write_json(staging_dir / "rounds_manifest_snapshot.json", root_manifest)

    promotion_plan = {
        "status": "ready",
        "alcb_root": str(ALCB_ROOT),
        "staging_dir": str(staging_dir),
        "final_round_dir": str(final_round_dir),
        "archive_dir": str(archive_dir),
        "archive_targets": [
            str(ALCB_ROOT / name)
            for name in (
                "round_1",
                "round_2",
                "round_3",
                "round_4",
                "phase_0_validity_20260816",
                "baseline_recovery_rth_20260816",
                "rounds_manifest.json",
            )
        ],
        "new_manifest_source": str(staging_dir / "rounds_manifest_snapshot.json"),
        "validation": validation,
    }
    _write_json(staging_dir / "promotion_plan.json", promotion_plan)
    print(f"FINALIZED verified staging metadata at {staging_dir}")
    print(f"ARCHIVE target reserved as {archive_dir}")


if __name__ == "__main__":
    main()
