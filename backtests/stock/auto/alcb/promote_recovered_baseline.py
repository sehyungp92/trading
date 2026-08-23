"""Build a verified Round 1 package from the recovered ALCB RTH baseline.

This command intentionally stages the package without moving active results.  The
caller can inspect ``promotion_plan.json`` and then perform the archive/promotion
with literal, pre-validated paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin
from backtests.stock.auto.alcb.run_phase0_validity import (
    _analysis,
    _group_summary,
    _serialize_trade,
)


ALCB_ROOT = REPO_ROOT / "backtests/output/stock/alcb"
RECOVERY_DIR = ALCB_ROOT / "baseline_recovery_rth_20260816"
STAGING_DIR = ALCB_ROOT / ".round_1_recovered_staging"
DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"
CORE_REPRODUCTION_KEYS = (
    "total_trades",
    "trades_per_month",
    "win_rate",
    "expectancy",
    "expected_total_r",
    "net_profit",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe",
)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    try:
        return value.item()
    except AttributeError as exc:
        raise TypeError(f"not JSON serializable: {type(value).__name__}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_delta(fresh: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for key in CORE_REPRODUCTION_KEYS:
        actual = float(fresh[key])
        target = float(expected["avg_r"] if key == "expectancy" else expected[key])
        deltas[key] = {
            "fresh": actual,
            "recovery_manifest": target,
            "absolute_error": actual - target,
        }
    return deltas


def _assert_reproduced(deltas: dict[str, dict[str, float]]) -> None:
    failures: list[str] = []
    for key, values in deltas.items():
        tolerance = 0.0 if key == "total_trades" else (0.005 if key == "net_profit" else 1e-9)
        if abs(values["absolute_error"]) > tolerance:
            failures.append(
                f"{key}: fresh={values['fresh']!r}, "
                f"expected={values['recovery_manifest']!r}, "
                f"error={values['absolute_error']!r}"
            )
    if failures:
        raise RuntimeError("Recovered baseline did not reproduce:\n" + "\n".join(failures))


def _diagnostics_header(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    generated_at: str,
) -> str:
    selected = manifest["selected"]
    costs = selected["costs"]
    validation = selected["validation"]
    lines = [
        "ALCB RECOVERED BASELINE — PROMOTED ROUND 1",
        "=" * 72,
        f"Generated: {generated_at}",
        f"Selected candidate: {selected['id']}",
        f"Mutation signature: {selected['signature']}",
        f"Training window: {START_DATE} through {END_DATE}",
        f"Session policy: us_equity_rth_0930_exchange_close_v1",
        f"Data authority: {manifest['data_authority']}",
        f"Status: {manifest['status']}",
        "Post-2026-03-01 excluded period accessed: false",
        "",
        "FRESH REPRODUCTION",
        "-" * 72,
        f"Trades: {int(metrics['total_trades'])}",
        f"Trades/month: {metrics['trades_per_month']:.4f}",
        f"Win rate: {metrics['win_rate']:.2%}",
        f"Average R: {metrics['expectancy']:+.6f}",
        f"Total R: {metrics['expected_total_r']:+.6f}",
        f"Net profit: ${metrics['net_profit']:+,.2f}",
        f"Profit factor: {metrics['profit_factor']:.6f}",
        f"Max drawdown: {metrics['max_drawdown_pct']:.4%}",
        f"Sharpe: {metrics['sharpe']:.6f}",
        "",
        "CHRONOLOGICAL ROBUSTNESS",
        "-" * 72,
        f"Positive folds: {validation['positive_fold_count']}/4",
        f"Worst-fold average R: {validation['worst_fold_avg_r']:+.6f}",
        f"Minimum fold PF: {validation['minimum_fold_profit_factor']:.6f}",
        f"Maximum fold DD: {validation['maximum_fold_drawdown_pct']:.4%}",
    ]
    for fold in validation["folds"]:
        lines.append(
            f"{fold['fold']}: n={int(fold['total_trades'])}, "
            f"avgR={fold['avg_r']:+.5f}, totalR={fold['expected_total_r']:+.2f}, "
            f"PF={fold['profit_factor']:.3f}, DD={fold['max_drawdown_pct']:.2%}"
        )
    lines.extend([
        "",
        "EXECUTION-COST ROBUSTNESS",
        "-" * 72,
    ])
    for label in ("7.5", "10.0"):
        row = costs[label]
        lines.append(
            f"{label} bps: n={int(row['total_trades'])}, "
            f"avgR={row['avg_r']:+.5f}, totalR={row['expected_total_r']:+.2f}, "
            f"net=${row['net_profit']:+,.2f}, PF={row['profit_factor']:.3f}, "
            f"DD={row['max_drawdown_pct']:.2%}, Sharpe={row['sharpe']:.2f}"
        )
    lines.extend([
        "",
        "INTERPRETATION BOUNDARY",
        "-" * 72,
        "This is the recovered diagnostic RTH baseline and the correct starting point",
        "for a new phased optimization round. It is not an untouched holdout result.",
        "Authoritative direct-RTH/frozen-data confirmation remains required before",
        "production promotion; the post-2026-03-01 excluded interval remains unaccessed.",
    ])
    return "\n".join(lines)


def _round_evaluation(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    selected = manifest["selected"]
    folds = selected["validation"]
    return "\n".join([
        "ALCB RECOVERED BASELINE ROUND 1 EVALUATION",
        "=" * 72,
        "Decision: PROMOTE as the new phased-auto starting baseline.",
        f"Candidate: {selected['id']}",
        f"5 bps: {int(metrics['total_trades'])} trades, "
        f"{metrics['trades_per_month']:.2f}/month, {metrics['expected_total_r']:+.2f}R, "
        f"${metrics['net_profit']:+,.2f}, PF {metrics['profit_factor']:.3f}, "
        f"DD {metrics['max_drawdown_pct']:.2%}.",
        f"Chronology: {folds['positive_fold_count']}/4 positive folds; "
        f"worst fold {folds['worst_fold_avg_r']:+.4f} average R.",
        f"Costs: 7.5 bps {selected['costs']['7.5']['expected_total_r']:+.2f}R; "
        f"10 bps {selected['costs']['10.0']['expected_total_r']:+.2f}R.",
        "Risk note: the baseline is provisional until authoritative direct-RTH replay.",
        "Holdout note: no data after 2026-03-01 was accessed.",
        "",
    ])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-dir", type=Path, default=RECOVERY_DIR)
    parser.add_argument("--staging-dir", type=Path, default=STAGING_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    recovery_dir = args.recovery_dir.resolve()
    staging_dir = args.staging_dir.resolve()
    data_dir = args.data_dir.resolve()
    alcb_root = ALCB_ROOT.resolve()

    if staging_dir.parent != alcb_root or staging_dir.name != STAGING_DIR.name:
        raise ValueError(f"Unsafe staging path: {staging_dir}")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    for required in ("optimized_config.json", "final_recovery_manifest.json"):
        if not (recovery_dir / required).is_file():
            raise FileNotFoundError(recovery_dir / required)

    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "0"
    config = json.loads((recovery_dir / "optimized_config.json").read_text(encoding="utf-8"))
    recovery_manifest = json.loads(
        (recovery_dir / "final_recovery_manifest.json").read_text(encoding="utf-8")
    )
    if config.get("intraday_session_policy") != "us_equity_rth_0930_exchange_close_v1":
        raise ValueError("Recovered config is not pinned to the RTH session policy")

    print("START final recovered baseline replay with full diagnostics", flush=True)
    started = datetime.now(timezone.utc)
    plugin = ALCBP16Plugin(
        data_dir,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_equity=10_000.0,
        max_workers=1,
    )
    context = plugin._run_config(
        config,
        start_date=START_DATE,
        end_date=END_DATE,
        store_context=True,
        collect_diagnostics=True,
    )
    metrics = context["metrics"]
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    deltas = _metric_delta(metrics, recovery_manifest["selected"]["metrics"])
    _assert_reproduced(deltas)
    print(
        f"DONE final replay: n={int(metrics['total_trades'])}, "
        f"R={metrics['expected_total_r']:+.2f}, net=${metrics['net_profit']:+,.2f}, "
        f"PF={metrics['profit_factor']:.3f}, DD={metrics['max_drawdown_pct']:.2%} "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    archive_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_pre_recovered_round1")
    final_round_dir = alcb_root / "round_1"
    archive_dir = alcb_root / "archive" / archive_name
    staging_dir.mkdir(parents=False)

    trades = context["trades"]
    analysis = _analysis(context, START_DATE, END_DATE)
    diagnostics = "\n\n".join([
        _diagnostics_header(recovery_manifest, metrics, generated_at),
        alcb_full_diagnostic(
            trades,
            shadow_tracker=context.get("shadow_tracker"),
            daily_selections=context.get("daily_selections"),
        ),
        qe_replacement_analysis(
            trades,
            max_positions=int(context["config"].param_overrides.get("max_positions", 10)),
        ),
    ]) + "\n"

    _write_json(staging_dir / "optimized_config.json", config)
    _write_json(staging_dir / "final_metrics.json", metrics)
    _write_json(staging_dir / "final_analysis.json", analysis)
    _write_json(staging_dir / "final_trades.json", [_serialize_trade(trade) for trade in trades])
    _write_json(staging_dir / "final_monthly.json", _group_summary(
        trades, lambda trade: trade.entry_time.strftime("%Y-%m")
    ))
    _write_json(staging_dir / "final_symbols.json", analysis["by_symbol"])
    _write_json(staging_dir / "final_exits.json", analysis["by_exit_reason"])
    _write_json(staging_dir / "final_entry_types.json", analysis["by_entry_type"])
    (staging_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    (staging_dir / "round_evaluation.txt").write_text(
        _round_evaluation(recovery_manifest, metrics), encoding="utf-8"
    )

    evidence_dir = staging_dir / "recovery_evidence"
    evidence_dir.mkdir()
    for source in sorted(recovery_dir.iterdir()):
        if source.is_file():
            shutil.copy2(source, evidence_dir / source.name)

    config_sha = _sha256(staging_dir / "optimized_config.json")
    diagnostics_sha = _sha256(staging_dir / "round_final_diagnostics.txt")
    provenance = plugin.build_provenance()
    validation = {
        "status": "passed",
        "fresh_reproduction": True,
        "metric_deltas": deltas,
        "optimized_config_sha256": config_sha,
        "round_final_diagnostics_sha256": diagnostics_sha,
        "diagnostics_bytes": (staging_dir / "round_final_diagnostics.txt").stat().st_size,
        "diagnostics_lines": len(diagnostics.splitlines()),
        "trade_rows": len(trades),
        "replay_elapsed_seconds": elapsed,
    }
    _write_json(staging_dir / "promotion_validation.json", validation)

    run_spec = {
        "family": "stock",
        "strategy": "alcb",
        "round": 1,
        "description": "Recovered RTH baseline promoted as the clean starting point for phased auto optimization",
        "generated_at_utc": generated_at,
        "execution_context": {
            "data_dir": str(data_dir),
            "start_date": START_DATE,
            "end_date": END_DATE,
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

    artifacts = {
        name: str(final_round_dir / name)
        for name in (
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
    }
    artifacts["recovery_evidence"] = str(final_round_dir / "recovery_evidence")
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
        "headline_metrics": {
            "total_trades": int(metrics["total_trades"]),
            "trades_per_month": float(metrics["trades_per_month"]),
            "win_rate": float(metrics["win_rate"]) * 100.0,
            "expected_total_r": float(metrics["expected_total_r"]),
            "net_profit": float(metrics["net_profit"]),
            "profit_factor": float(metrics["profit_factor"]),
            "max_drawdown_pct": float(metrics["max_drawdown_pct"]) * 100.0,
            "sharpe_ratio": float(metrics["sharpe"]),
            "calmar_ratio": float(metrics["calmar"]),
        },
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
        "total_trades": int(metrics["total_trades"]),
        "trades_per_month": float(metrics["trades_per_month"]),
        "win_rate": float(metrics["win_rate"]) * 100.0,
        "expected_total_r": float(metrics["expected_total_r"]),
        "net_profit": float(metrics["net_profit"]),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown_pct": float(metrics["max_drawdown_pct"]) * 100.0,
        "sharpe_ratio": float(metrics["sharpe"]),
        "calmar_ratio": float(metrics["calmar"]),
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
        "alcb_root": str(alcb_root),
        "staging_dir": str(staging_dir),
        "final_round_dir": str(final_round_dir),
        "archive_dir": str(archive_dir),
        "archive_targets": [
            str(alcb_root / name)
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
    print(f"STAGED verified Round 1 package at {staging_dir}", flush=True)
    print(f"ARCHIVE target reserved as {archive_dir}", flush=True)


if __name__ == "__main__":
    main()
