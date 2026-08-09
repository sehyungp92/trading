"""Reclassify the promoted ALCB candidate artifacts as Round 4.

This is intentionally non-destructive toward Rounds 1-3. It updates the
already-preserved Round 4 copy, repairs internal paths/round metadata, and
appends (or replaces) only the Round 4 manifest entry.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "backtests" / "output" / "stock" / "alcb"
ROUND2_DIR = STRATEGY_DIR / "round_2"
ROUND4_DIR = STRATEGY_DIR / "round_4"
MANIFEST_PATH = STRATEGY_DIR / "rounds_manifest.json"
RECLASSIFICATION_ARCHIVE = (
    STRATEGY_DIR
    / "archived_rounds"
    / "20260723T180854Z_reclassified_to_round4"
)
RESTORED_ROUND2_COMMIT = "6ce8cea8a96490640b9d30d64d40e4f1a58cd1a2"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _replace_round2_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_round2_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_round2_paths(item) for item in value]
    if not isinstance(value, str):
        return value

    old_abs = str(ROUND2_DIR.resolve())
    new_abs = str(ROUND4_DIR.resolve())
    replaced = value.replace(old_abs, new_abs)
    replaced = replaced.replace(old_abs.replace("\\", "/"), new_abs.replace("\\", "/"))
    replaced = replaced.replace(
        "backtests/output/stock/alcb/round_2/",
        "backtests/output/stock/alcb/round_4/",
    )
    replaced = replaced.replace(
        "backtests\\output\\stock\\alcb\\round_2\\",
        "backtests\\output\\stock\\alcb\\round_4\\",
    )
    if replaced.startswith("round_2_final_"):
        replaced = "round_4_final_" + replaced.removeprefix("round_2_final_")
    return replaced


def _rewrite_metadata(reclassified_at: datetime) -> None:
    for path in ROUND4_DIR.rglob("*.json"):
        payload = _load(path)
        rewritten = _replace_round2_paths(payload)
        if path == ROUND4_DIR / "run_spec.json":
            rewritten["round"] = 4
            rewritten["reclassified_from_round"] = 2
            rewritten["reclassified_at_utc"] = reclassified_at
        elif path == ROUND4_DIR / "run_summary.json":
            rewritten["round"] = 4
            rewritten["reclassified_from_round"] = 2
            rewritten["reclassified_at_utc"] = reclassified_at
        elif path == ROUND4_DIR / "phase_state.json":
            rewritten["round_name"] = "round_4_final_risk_optimization_20260723"
            rewritten["reclassified_from_round"] = 2
            rewritten["reclassified_at_utc"] = reclassified_at
        elif path == ROUND4_DIR / "final_optimization_summary.json":
            rewritten["reclassified_from_round"] = 2
            rewritten["reclassified_at_utc"] = reclassified_at
        elif (
            path.name == "completion.json"
            and path.parent.name == "final_optimization_20260723"
        ):
            rewritten["round_dir"] = str(ROUND4_DIR.resolve())
            rewritten["reclassified_from_round"] = 2
            rewritten["reclassified_at_utc"] = reclassified_at
        if rewritten != payload:
            _write_json(path, rewritten)

    diagnostics = ROUND4_DIR / "round_final_diagnostics.txt"
    diagnostics.write_text(
        diagnostics.read_text(encoding="utf-8").replace(
            "ALCB ROUND 2 FINAL RISK/QUALITY OPTIMIZATION",
            "ALCB ROUND 4 FINAL RISK/QUALITY OPTIMIZATION",
            1,
        ),
        encoding="utf-8",
    )

    evaluation = ROUND4_DIR / "round_evaluation.txt"
    evaluation_text = evaluation.read_text(encoding="utf-8").replace(
        "ALCB ROUND 2 FINAL RISK/QUALITY OPTIMIZATION",
        "ALCB ROUND 4 FINAL RISK/QUALITY OPTIMIZATION",
        1,
    )
    marker = "=" * 72 + "\n"
    lineage = (
        "\n"
        f"Reclassified from promoted round_2 as round_4 at UTC: {reclassified_at.isoformat()}\n"
        f"Restored round_2 commit: {RESTORED_ROUND2_COMMIT}\n"
        f"Round 4 directory: {ROUND4_DIR.resolve()}\n"
    )
    if "Reclassified from promoted round_2 as round_4" not in evaluation_text:
        evaluation_text = evaluation_text.replace(marker, marker + lineage, 1)
    evaluation.write_text(evaluation_text, encoding="utf-8")

    report = ROUND4_DIR / "final_optimization_20260723" / "report.md"
    report_text = report.read_text(encoding="utf-8")
    report_text = report_text.replace(
        "# ALCB Round 2 final optimization",
        "# ALCB Round 4 final optimization",
        1,
    )
    report_text = report_text.replace(
        "saved as a Round-2 candidate",
        "saved as a Round-4 candidate",
    )
    report.write_text(report_text, encoding="utf-8")


def _percent(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    number = float(value)
    return number * 100.0 if abs(number) <= 1.0 else number


def _manifest_entry(reclassified_at: datetime) -> dict[str, Any]:
    mutations = _load(ROUND4_DIR / "optimized_config.json")
    metrics = _load(ROUND4_DIR / "final_metrics.json")
    summary = _load(ROUND4_DIR / "run_summary.json")
    optimization = _load(ROUND4_DIR / "final_optimization_summary.json")
    completion = _load(
        ROUND4_DIR / "final_optimization_20260723" / "completion.json"
    )
    provenance = summary["provenance"]
    return {
        "round": 4,
        "timestamp": reclassified_at,
        "mutations_count": len(mutations),
        "mutations": mutations,
        "total_trades": int(metrics["total_trades"]),
        "win_rate": _percent(metrics, "win_rate"),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown_pct": _percent(metrics, "max_drawdown_pct"),
        "net_return_pct": _percent(metrics, "net_return_pct"),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", metrics["sharpe"])),
        "calmar_ratio": float(metrics.get("calmar_ratio", metrics["calmar"])),
        "selection_fingerprint": provenance["selection_fingerprint"],
        "diagnostics_fingerprint": provenance["diagnostics_fingerprint"],
        "provenance_schema_version": provenance["schema_version"],
        "provenance_status": summary["provenance_status"],
        "selected_candidate": completion["selected_candidate"],
        "research_report": str(
            (ROUND4_DIR / "final_optimization_20260723" / "report.md").resolve()
        ),
        "balanced_comparison": _load(
            ROUND4_DIR / "final_candidate_comparison.json"
        ),
        "data_authority": optimization["data_authority"],
        "oos_status": optimization["oos_status"],
        "reclassified_from_round": 2,
        "reclassified_at_utc": reclassified_at,
        "restored_round2_commit": RESTORED_ROUND2_COMMIT,
        "reclassification_archive": str(RECLASSIFICATION_ARCHIVE.resolve()),
        "round_dir": str(ROUND4_DIR.resolve()),
    }


def _update_manifest(reclassified_at: datetime) -> None:
    manifest = _load(MANIFEST_PATH)
    rounds = [
        entry for entry in manifest["rounds"] if int(entry.get("round", 0)) != 4
    ]
    rounds.append(_manifest_entry(reclassified_at))
    manifest["rounds"] = sorted(rounds, key=lambda entry: int(entry["round"]))
    _write_json(MANIFEST_PATH, manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute:
        raise SystemExit("Pass --execute to reclassify the preserved candidate.")
    if not ROUND2_DIR.is_dir() or not ROUND4_DIR.is_dir():
        raise FileNotFoundError("Round 2 and Round 4 directories must both exist.")
    ROUND4_DIR.resolve().relative_to(STRATEGY_DIR.resolve())
    MANIFEST_PATH.resolve().relative_to(STRATEGY_DIR.resolve())

    reclassified_at = datetime.now(timezone.utc)
    _rewrite_metadata(reclassified_at)
    _update_manifest(reclassified_at)

    manifest = _load(MANIFEST_PATH)
    round4_entry = next(
        entry for entry in manifest["rounds"] if int(entry["round"]) == 4
    )
    if round4_entry["mutations"] != _load(ROUND4_DIR / "optimized_config.json"):
        raise RuntimeError("Round 4 manifest/config consistency check failed.")
    if int(_load(ROUND4_DIR / "run_spec.json")["round"]) != 4:
        raise RuntimeError("Round 4 run-spec correction failed.")
    if int(_load(ROUND4_DIR / "run_summary.json")["round"]) != 4:
        raise RuntimeError("Round 4 run-summary correction failed.")

    print(f"round_2_restored={ROUND2_DIR}")
    print(f"round_4_reclassified={ROUND4_DIR}")
    print(f"manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
