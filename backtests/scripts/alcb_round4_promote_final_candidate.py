"""Promote the validated final ALCB candidate into the active Round-4 artifacts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from backtests.scripts.alcb_round2_oos_robustness import (  # noqa: E402
    INITIAL_EQUITY,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    REPO_ROOT,
    _trade_to_dict,
    _write_json,
)
from backtests.shared.auto.round_manager import RoundManager  # noqa: E402
from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic  # noqa: E402
from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis  # noqa: E402
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin  # noqa: E402


ROUND_DIR = REPO_ROOT / "backtests" / "output" / "stock" / "alcb" / "round_4"
DEFAULT_RESEARCH = ROUND_DIR / "final_optimization_20260723"
DATA_DIR = REPO_ROOT / "backtests" / "stock" / "data" / "raw"
ROUND_MANAGER = RoundManager("stock", "alcb")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_context(mutations: dict[str, Any], start: str, end: str) -> tuple[dict[str, Any], Any]:
    plugin = ALCBP16Plugin(
        DATA_DIR,
        start_date=start,
        end_date=end,
        initial_equity=INITIAL_EQUITY,
        max_workers=1,
    )
    context = plugin._run_config(mutations, store_context=True, collect_diagnostics=True)
    return context, plugin


def _group(
    rows: list[dict[str, Any]],
    label: str,
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        rs = [float(row.get("r_multiple", 0.0) or 0.0) for row in values]
        pnls = [float(row.get("pnl_net", 0.0) or 0.0) for row in values]
        output.append(
            {
                label: key,
                "trades": len(values),
                "share": len(values) / len(rows) if rows else 0.0,
                "win_rate": sum(value > 0 for value in pnls) / len(values),
                "avg_r": mean(rs),
                "total_r": sum(rs),
                "pnl_net": sum(pnls),
            }
        )
    if label == "month":
        return sorted(output, key=lambda row: row[label])
    if label == "exit_reason":
        return sorted(output, key=lambda row: (-row["trades"], row[label]))
    return sorted(output, key=lambda row: (-row["pnl_net"], row[label]))


def _pct_delta(new: float, old: float) -> float:
    return (new - old) / abs(old) if abs(old) > 1e-12 else 0.0


def _comparison(selected: dict[str, Any], balanced: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "expected_total_r",
        "net_profit",
        "trades_per_month",
        "profit_factor",
        "max_drawdown_pct",
        "win_rate",
    )
    output: dict[str, Any] = {}
    for window in ("is", "oos"):
        output[window] = {}
        for key in keys:
            old = float(balanced[f"{window}_{key}"])
            new = float(selected[f"{window}_{key}"])
            output[window][key] = {
                "balanced": old,
                "selected": new,
                "delta": new - old,
                "delta_pct": _pct_delta(new, old),
            }
    return output


def _archive_current(round_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = (
        round_dir.parent
        / "archived_rounds"
        / f"{timestamp}_pre_final_risk_optimization_round4"
        / "round_4"
    )
    archive.mkdir(parents=True, exist_ok=False)
    names = (
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
    for name in names:
        source = round_dir / name
        if source.exists():
            shutil.copy2(source, archive / name)
    manifest = round_dir.parent / "rounds_manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, archive.parent / "rounds_manifest.json")
    return archive.parent


def _update_phase_state(
    path: Path,
    mutations: dict[str, Any],
    metrics: dict[str, Any],
    selected_name: str,
) -> None:
    payload = _load(path) if path.exists() else {}
    payload["current_phase"] = 8
    payload["completed_phases"] = list(range(1, 9))
    payload["cumulative_mutations"] = mutations
    payload["round_name"] = "round_4_final_risk_optimization_20260723"
    phase_results = payload.setdefault("phase_results", {})
    final = phase_results.setdefault("8", {})
    final.update(
        {
            "focus": "Final causal drawdown/PF optimization",
            "base_mutations": mutations,
            "final_mutations": mutations,
            "final_metrics": metrics,
            "attempted_final_mutations": mutations,
            "attempted_final_metrics": metrics,
            "selected_candidate": selected_name,
            "adoption_reason": "final_robustness_guardrails_passed",
            "applied_phase_mutations": True,
        }
    )
    _write_json(path, payload)


def _evaluation_text(
    selected: dict[str, Any],
    balanced: dict[str, Any],
    comparison: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    lines = [
        "ALCB ROUND 2 FINAL RISK/QUALITY OPTIMIZATION",
        "=" * 72,
        f"Selected candidate: {selected['name']}",
        f"Promotion guardrails passed: {recommendation['promotion_eligible']}",
        "Data authority: repaired legacy cache; authoritative frozen direct-RTH bundle unavailable.",
        "OOS status: consumed development data.",
        "",
        "BALANCED VS SELECTED",
        "-" * 72,
    ]
    for window in ("is", "oos"):
        lines.append(window.upper())
        for key, values in comparison[window].items():
            lines.append(
                f"  {key}: {values['balanced']:.6f} -> {values['selected']:.6f} "
                f"({values['delta_pct']:+.2%})"
            )
    lines.extend(
        [
            "",
            "ROBUSTNESS",
            "-" * 72,
            f"Early IS R retention: {selected['early_r_ratio']:.3f}",
            f"Late IS R retention: {selected['late_r_ratio']:.3f}",
            f"Stress R delta: {selected['stress_r_delta']:+.2f}R",
            f"Stress DD ratio: {selected['stress_dd_ratio']:.3f}",
            f"Paired bootstrap P(R uplift > 0): "
            f"{recommendation['bootstrap']['r_delta']['probability_positive']:.1%}",
            f"Paired bootstrap P(net uplift > 0): "
            f"{recommendation['bootstrap']['pnl_delta']['probability_positive']:.1%}",
            "",
            f"Balanced reference: {balanced['name']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.allow_legacy_data:
        raise SystemExit("Pass --allow-legacy-data to acknowledge the repaired legacy cache.")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    research = args.research_dir.resolve()
    recommendation = _load(research / "recommendation.json")
    completion = _load(research / "completion.json")
    if not completion.get("selected_candidate"):
        raise RuntimeError("Final optimization is incomplete.")
    if not recommendation.get("promotion_eligible"):
        raise RuntimeError("Selected candidate did not pass final promotion guardrails.")

    selected = recommendation["candidate"]
    balanced = recommendation["balanced"]
    mutations = recommendation["full_mutations"]
    base_mutations = _load(research / "base_round2_snapshot.json")
    balanced_mutations = {**base_mutations, **balanced["patch"]}
    comparison = _comparison(selected, balanced)

    context, plugin = _run_context(mutations, IS_START, IS_END)
    try:
        metrics = dict(context["metrics"])
        trades = list(context["trades"])
        rows = [_trade_to_dict(trade) for trade in trades]
        provenance = plugin.build_provenance()
        diagnostics = "\n\n".join(
            [
                _evaluation_text(selected, balanced, comparison, recommendation),
                alcb_full_diagnostic(
                    trades,
                    shadow_tracker=context.get("shadow_tracker"),
                    daily_selections=context.get("daily_selections"),
                ),
                qe_replacement_analysis(
                    trades,
                    max_positions=int(
                        context["config"].param_overrides.get("max_positions", 10)
                    ),
                ),
            ]
        )
    finally:
        plugin.close_pool()

    archive = _archive_current(ROUND_DIR)
    _write_json(ROUND_DIR / "optimized_config.json", mutations)
    _write_json(ROUND_DIR / "final_metrics.json", metrics)
    _write_json(ROUND_DIR / "final_trades.json", rows)
    _write_json(
        ROUND_DIR / "final_monthly.json",
        _group(rows, "month", lambda row: str(row["exit_time"])[:7]),
    )
    _write_json(
        ROUND_DIR / "final_symbols.json",
        _group(rows, "symbol", lambda row: str(row.get("symbol") or "UNKNOWN")),
    )
    _write_json(
        ROUND_DIR / "final_exits.json",
        _group(rows, "exit_reason", lambda row: str(row.get("exit_reason") or "UNKNOWN")),
    )
    (ROUND_DIR / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    evaluation = _evaluation_text(selected, balanced, comparison, recommendation)
    (ROUND_DIR / "round_evaluation.txt").write_text(evaluation, encoding="utf-8")
    _write_json(ROUND_DIR / "final_candidate_comparison.json", comparison)
    _write_json(
        ROUND_DIR / "final_optimization_summary.json",
        {
            "promoted_at_utc": datetime.now(timezone.utc),
            "selected_candidate": selected,
            "balanced_candidate": balanced,
            "comparison": comparison,
            "recommendation_source": str(research / "recommendation.json"),
            "research_report": str(research / "report.md"),
            "archive": str(archive),
            "data_authority": recommendation["data_authority"],
            "oos_status": recommendation["oos_status"],
        },
    )
    _update_phase_state(
        ROUND_DIR / "phase_state.json",
        mutations,
        metrics,
        selected["name"],
    )

    ROUND_MANAGER.write_run_spec(
        ROUND_DIR,
        4,
        "alcb",
        description="Final causal drawdown and profit-factor optimization",
        baseline_mutations=balanced_mutations,
        baseline_source=research / "base_round2_snapshot.json",
        scoring_weights={
            "is_return_and_net": 0.49,
            "is_pf_and_drawdown": 0.41,
            "is_frequency": 0.10,
            "oos_confirmation": "consumed development comparison only",
        },
        execution_context={
            "data_dir": str(DATA_DIR.resolve()),
            "initial_equity": INITIAL_EQUITY,
            "start_date": IS_START,
            "end_date": IS_END,
            "oos_start": OOS_START,
            "oos_end": OOS_END,
            "research_dir": str(research),
        },
        provenance=provenance,
        provenance_status="promoted_recomputed_legacy_cache_consumed_oos",
        overwrite=True,
    )
    ROUND_MANAGER.write_run_summary(
        ROUND_DIR,
        mutations,
        metrics,
        list(range(1, 9)),
        round_num=4,
        source_diagnostics=ROUND_DIR / "round_final_diagnostics.txt",
        source_phase_state=ROUND_DIR / "phase_state.json",
        provenance=provenance,
        provenance_status="promoted_recomputed_legacy_cache_consumed_oos",
        provenance_validation={
            "valid": True,
            "status": "final_targeted_optimization",
            "selection_drift": False,
            "diagnostics_drift": False,
            "message": (
                "Candidate passed aggregate IS, early/late IS, historical stress, "
                "paired resampling, and execution-cost diagnostics; OOS is consumed."
            ),
        },
    )
    summary_path = ROUND_DIR / "run_summary.json"
    summary = _load(summary_path)
    summary.update(
        {
            "final_optimization": {
                "selected_candidate": selected,
                "balanced_candidate": balanced,
                "comparison": comparison,
                "research_dir": str(research),
                "archive": str(archive),
                "bootstrap": recommendation["bootstrap"],
                "leave_one_month_out": recommendation["leave_one_month_out"],
                "leave_one_sector_out": recommendation["leave_one_sector_out"],
            },
            "artifacts": {
                "round_final_diagnostics": str((ROUND_DIR / "round_final_diagnostics.txt").resolve()),
                "round_evaluation": str((ROUND_DIR / "round_evaluation.txt").resolve()),
                "optimized_config": str((ROUND_DIR / "optimized_config.json").resolve()),
                "final_metrics": str((ROUND_DIR / "final_metrics.json").resolve()),
                "final_trades": str((ROUND_DIR / "final_trades.json").resolve()),
                "final_monthly": str((ROUND_DIR / "final_monthly.json").resolve()),
                "final_symbols": str((ROUND_DIR / "final_symbols.json").resolve()),
                "final_exits": str((ROUND_DIR / "final_exits.json").resolve()),
                "final_candidate_comparison": str(
                    (ROUND_DIR / "final_candidate_comparison.json").resolve()
                ),
                "final_optimization_summary": str(
                    (ROUND_DIR / "final_optimization_summary.json").resolve()
                ),
                "research_report": str((research / "report.md").resolve()),
            },
        }
    )
    _write_json(summary_path, summary)

    manifest_path = ROUND_MANAGER.append_to_manifest(
        4,
        mutations,
        metrics,
        provenance=provenance,
        provenance_status="promoted_recomputed_legacy_cache_consumed_oos",
    )
    manifest = _load(manifest_path)
    for entry in manifest.get("rounds", []):
        if int(entry.get("round", 0)) == 4 and not entry.get("archived"):
            entry.update(
                {
                    "selected_candidate": selected["name"],
                    "research_report": str((research / "report.md").resolve()),
                    "balanced_comparison": comparison,
                    "data_authority": recommendation["data_authority"],
                    "oos_status": recommendation["oos_status"],
                    "archive": str(archive),
                }
            )
    _write_json(manifest_path, manifest)

    completion["promotion_written"] = True
    completion["promoted_at_utc"] = datetime.now(timezone.utc)
    completion["round_dir"] = str(ROUND_DIR)
    completion["archive"] = str(archive)
    _write_json(research / "completion.json", completion)
    print(f"promoted {selected['name']} to {ROUND_DIR}")
    print(f"archive={archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
