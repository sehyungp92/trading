"""Recoverably relabel the final ALCB optimization as Round 3.

The 2026-07-23 final optimization was initially promoted over Round 2. This
script preserves that promoted state as Round 3, restores the pre-promotion
Round 2 artifacts, and rebuilds the manifest with internally consistent
metrics, mutations, provenance, and round numbers.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "backtests" / "output" / "stock" / "alcb"
ROUND2_DIR = STRATEGY_DIR / "round_2"
ROUND3_DIR = STRATEGY_DIR / "round_3"
MANIFEST_PATH = STRATEGY_DIR / "rounds_manifest.json"
RESTORE_ARCHIVE = (
    STRATEGY_DIR
    / "archived_rounds"
    / "20260723T180854Z_pre_final_risk_optimization_round2"
)
RESTORE_ROUND2 = RESTORE_ARCHIVE / "round_2"
RESTORE_MANIFEST = RESTORE_ARCHIVE / "rounds_manifest.json"

RESTORED_CORE_FILES = (
    "optimized_config.json",
    "run_spec.json",
    "run_summary.json",
    "round_final_diagnostics.txt",
    "round_evaluation.txt",
    "phase_state.json",
    "final_metrics.json",
    "final_trades.json",
    "final_monthly.json",
    "final_symbols.json",
    "final_exits.json",
)

ROUND3_ONLY_ROUND2_PATHS = (
    "final_candidate_comparison.json",
    "final_optimization_summary.json",
    "final_optimization_20260723",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _assert_within(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(parent.resolve())


def _replace_round2_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_round2_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_round2_paths(item) for item in value]
    if not isinstance(value, str):
        return value

    old_abs = str(ROUND2_DIR.resolve())
    new_abs = str(ROUND3_DIR.resolve())
    replaced = value.replace(old_abs, new_abs)
    replaced = replaced.replace(old_abs.replace("\\", "/"), new_abs.replace("\\", "/"))
    replaced = replaced.replace(
        "backtests/output/stock/alcb/round_2/",
        "backtests/output/stock/alcb/round_3/",
    )
    replaced = replaced.replace(
        "backtests\\output\\stock\\alcb\\round_2\\",
        "backtests\\output\\stock\\alcb\\round_3\\",
    )
    if replaced.startswith("round_2_final_"):
        replaced = "round_3_final_" + replaced.removeprefix("round_2_final_")
    return replaced


def _rewrite_round3_metadata() -> None:
    for path in ROUND3_DIR.rglob("*.json"):
        payload = _replace_round2_paths(_load(path))
        if path == ROUND3_DIR / "run_spec.json":
            payload["round"] = 3
        elif path == ROUND3_DIR / "run_summary.json":
            payload["round"] = 3
        elif path == ROUND3_DIR / "phase_state.json":
            payload["round_name"] = "round_3_final_risk_optimization_20260723"
        elif path.name == "completion.json" and path.parent.name == "final_optimization_20260723":
            payload["round_dir"] = str(ROUND3_DIR.resolve())
        _write_json(path, payload)

    text_replacements = {
        ROUND3_DIR / "round_final_diagnostics.txt": (
            ("ALCB ROUND 2 FINAL RISK/QUALITY OPTIMIZATION", "ALCB ROUND 3 FINAL RISK/QUALITY OPTIMIZATION"),
        ),
        ROUND3_DIR / "round_evaluation.txt": (
            ("ALCB ROUND 2 FINAL RISK/QUALITY OPTIMIZATION", "ALCB ROUND 3 FINAL RISK/QUALITY OPTIMIZATION"),
        ),
        ROUND3_DIR / "final_optimization_20260723" / "report.md": (
            ("# ALCB Round 2 final optimization", "# ALCB Round 3 final optimization"),
            ("saved as a Round-2 candidate", "saved as a Round-3 candidate"),
        ),
    }
    for path, replacements in text_replacements.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in replacements:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def _canonical_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    def percent(key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        number = float(value)
        return number * 100.0 if abs(number) <= 1.0 else number

    return {
        "total_trades": int(metrics["total_trades"]),
        "win_rate": percent("win_rate"),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown_pct": percent("max_drawdown_pct"),
        "net_return_pct": percent("net_return_pct"),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", metrics.get("sharpe"))),
        "calmar_ratio": float(metrics.get("calmar_ratio", metrics.get("calmar"))),
    }


def _restored_round2_entry(
    archived_manifest: dict[str, Any],
    migration_backup: Path,
) -> dict[str, Any]:
    original = copy.deepcopy(
        next(entry for entry in archived_manifest["rounds"] if int(entry["round"]) == 2)
    )
    metrics = _load(ROUND2_DIR / "final_metrics.json")
    summary = _load(ROUND2_DIR / "run_summary.json")
    provenance = summary["provenance"]
    original.update(
        {
            **_canonical_metrics(metrics),
            "timestamp": summary["generated_at_utc"],
            "mutations": _load(ROUND2_DIR / "optimized_config.json"),
            "mutations_count": len(_load(ROUND2_DIR / "optimized_config.json")),
            "selection_fingerprint": provenance["selection_fingerprint"],
            "diagnostics_fingerprint": provenance["diagnostics_fingerprint"],
            "provenance_schema_version": provenance["schema_version"],
            "provenance_status": summary["provenance_status"],
            "restored_at_utc": datetime.now(timezone.utc),
            "restored_from_archive": str(RESTORE_ARCHIVE.resolve()),
            "migration_backup": str(migration_backup.resolve()),
        }
    )
    return original


def _rebuild_manifest(migration_backup: Path) -> None:
    current = _load(MANIFEST_PATH)
    archived = _load(RESTORE_MANIFEST)
    round1 = copy.deepcopy(
        next(entry for entry in current["rounds"] if int(entry["round"]) == 1)
    )
    promoted = copy.deepcopy(
        next(
            entry
            for entry in current["rounds"]
            if int(entry["round"]) == 2 and not entry.get("archived")
        )
    )
    round2 = _restored_round2_entry(archived, migration_backup)
    round3 = _replace_round2_paths(promoted)
    round3.update(
        {
            "round": 3,
            "migrated_at_utc": datetime.now(timezone.utc),
            "migrated_from_round": 2,
            "migration_backup": str(migration_backup.resolve()),
        }
    )
    manifest = {
        "family": "stock",
        "strategy": "alcb",
        "rounds": [round1, round2, round3],
    }
    _write_json(MANIFEST_PATH, manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the migration after all safety checks pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute:
        raise SystemExit("Pass --execute to perform the recoverable migration.")

    for path in (ROUND2_DIR, ROUND3_DIR, RESTORE_ARCHIVE, RESTORE_ROUND2, MANIFEST_PATH):
        _assert_within(path, STRATEGY_DIR)
    if not ROUND2_DIR.is_dir():
        raise FileNotFoundError(f"Missing promoted Round 2: {ROUND2_DIR}")
    if ROUND3_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Round 3: {ROUND3_DIR}")
    if not RESTORE_ROUND2.is_dir() or not RESTORE_MANIFEST.is_file():
        raise FileNotFoundError(f"Incomplete restore archive: {RESTORE_ARCHIVE}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    migration_backup = (
        STRATEGY_DIR
        / "archived_rounds"
        / f"{timestamp}_pre_round3_lineage_correction"
    )
    _assert_within(migration_backup, STRATEGY_DIR)
    migration_backup.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROUND2_DIR, migration_backup / "round_2_promoted_snapshot")
    shutil.copy2(MANIFEST_PATH, migration_backup / "rounds_manifest.json")

    shutil.copytree(ROUND2_DIR, ROUND3_DIR)
    _rewrite_round3_metadata()

    for name in RESTORED_CORE_FILES:
        source = RESTORE_ROUND2 / name
        if not source.exists():
            raise FileNotFoundError(f"Restore artifact missing: {source}")
        shutil.copy2(source, ROUND2_DIR / name)

    for name in ROUND3_ONLY_ROUND2_PATHS:
        target = ROUND2_DIR / name
        _assert_within(target, ROUND2_DIR)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    _rebuild_manifest(migration_backup)

    if _load(ROUND2_DIR / "optimized_config.json") != _load(
        RESTORE_ROUND2 / "optimized_config.json"
    ):
        raise RuntimeError("Round 2 configuration restoration did not verify.")
    round3_config = _load(ROUND3_DIR / "optimized_config.json")
    manifest = _load(MANIFEST_PATH)
    round3_entry = next(entry for entry in manifest["rounds"] if int(entry["round"]) == 3)
    if round3_config != round3_entry["mutations"]:
        raise RuntimeError("Round 3 manifest/config consistency check failed.")

    print(f"restored_round_2={ROUND2_DIR}")
    print(f"created_round_3={ROUND3_DIR}")
    print(f"migration_backup={migration_backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
