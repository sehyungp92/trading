from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.shared.auto.phase_state import _atomic_write_json


ROOT = Path(__file__).resolve().parents[4]
STRATEGY_DIR = ROOT / "backtests" / "output" / "stock" / "portfolio_synergy"
STABLE_SOURCE_NAME = "round_6_realism_stable"
ANALYSIS_SOURCE_NAME = "round_6_realism_stable_analysis"
ARCHIVE_REASON = "pre_realism_round1_reset"

REQUIRED_STABLE_FILES = (
    "artifact_manifest.json",
    "final_robustness.json",
    "freeze_receipt.json",
    "oos_validation.json",
    "optimized_config.json",
    "phase_0_matched_baselines.json",
    "phase_5_interactions.json",
    "promotion_decision.json",
    "run_spec.json",
    "stream_receipt.json",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_child(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Unsafe path outside intended parent: {resolved}")
    return resolved


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "net_return_pct",
        "total_r",
        "profit_factor",
        "win_rate",
        "max_drawdown_pct_mtm_daily",
        "total_trades",
        "entries_blocked_by_portfolio",
    )
    return {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in keys
    }


def _pct(value: Any) -> str:
    return f"{float(value):.2%}"


def _num(value: Any, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def _performance_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {_pct(metrics['net_return_pct'])} | "
        f"{_num(metrics['total_r'])}R | {_num(metrics['profit_factor'], 3)} | "
        f"{_pct(metrics['win_rate'])} | "
        f"{_pct(metrics['max_drawdown_pct_mtm_daily'])} | "
        f"{int(metrics['total_trades']):,} | "
        f"{int(metrics['entries_blocked_by_portfolio']):,} |"
    )


def _render_diagnostics(summary: dict[str, Any]) -> str:
    if "comprehensive_diagnostics" in summary:
        from backtests.stock.auto.portfolio_synergy.comprehensive_diagnostics import (
            render_diagnostics,
        )

        return render_diagnostics(summary)
    comparison = summary["matched_performance"]
    is_data = comparison["is"]
    oos_data = comparison["oos"]
    post_is = is_data["post_optimization_portfolio"]
    post_oos = oos_data["post_optimization_portfolio"]
    no_overlay_is = is_data["post_optimization_no_overlay"]
    no_overlay_oos = oos_data["post_optimization_no_overlay"]
    overlay = summary["overlay_effect"]
    config = summary["optimized_config"]
    account = config["account_rules"]
    allocations = config["strategy_allocations"]
    robustness = summary["robustness"]
    selection = summary["selection"]
    gates = summary["promotion_decision"]["gates"]
    receipt = summary["stream_receipt"]

    lines = [
        "# Final stock portfolio-synergy diagnostics — Round 1",
        "",
        "Status: **saved as the active research baseline; production activation is not approved**.",
        "",
        "## Contract and lineage",
        "",
        f"- IS: `{summary['is_window'][0]}` through `{summary['is_window'][1]}`.",
        f"- OOS: `{summary['oos_window'][0]}` through `{summary['oos_window'][1]}`; evaluated after the configuration freeze.",
        f"- Initial equity: `${summary['initial_equity']:,.0f}` shared by both strategies.",
        f"- Frozen config SHA-256: `{summary['config_sha256']}`.",
        f"- Selected candidate: `{selection['selected']}`.",
        f"- The unconstrained interaction winner `{selection['unconstrained_winner']}` was rejected: CSCV PBO was {_pct(selection['cscv_pbo']['probability_backtest_overfit'])} versus a {_pct(selection['maximum_probability_backtest_overfit'])} maximum, and robust-score gain was {selection['robust_score_gain']:.6f} versus a {selection['minimum_robust_score_gain']:.3f} minimum.",
        f"- ALCB Round-3 regeneration parity: `{receipt['alcb']['parity']['passed']}`; IARIC Round-3 regeneration parity: `{receipt['iaric']['parity']['passed']}`.",
        "",
        "## Pre- versus post-optimization",
        "",
        "| Window/version | Return | Total R | PF | Win rate | Daily-MTM max DD | Trades | Blocked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _performance_row("IS pre", is_data["pre_optimization_portfolio"]),
        _performance_row("IS post", post_is),
        _performance_row("OOS pre", oos_data["pre_optimization_portfolio"]),
        _performance_row("OOS post", post_oos),
        "",
        "Post-optimization increases return modestly, but the deliberately more aggressive sizing raises daily-MTM drawdown versus the native-risk baseline. Therefore it does not dominate the pre-optimization portfolio on every risk-adjusted metric.",
        "",
        "## Post portfolio versus Round-3 standalones",
        "",
        "All rows use the same $25,000 capital, costs, causal marks and boundary rules.",
        "",
        "| Window/system | Return | Total R | PF | Win rate | Daily-MTM max DD | Trades | Blocked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _performance_row("IS ALCB R3", is_data["alcb_round3_standalone_native_risk"]),
        _performance_row("IS IARIC R3", is_data["iaric_round3_standalone_native_risk"]),
        _performance_row("IS portfolio", post_is),
        _performance_row("OOS ALCB R3", oos_data["alcb_round3_standalone_native_risk"]),
        _performance_row("OOS IARIC R3", oos_data["iaric_round3_standalone_native_risk"]),
        _performance_row("OOS portfolio", post_oos),
        "",
        "Portfolio return can exceed the sum of standalone percentage returns because both sleeves compound through one shared NLV. The shared ledger prevents duplicated buying power; this is cross-compounding, not two independent accounts being added together.",
        "",
        "## Overlay and blocking diagnostics",
        "",
        "| Window/version | Return | Total R | PF | Win rate | Daily-MTM max DD | Trades | Blocked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _performance_row("IS no overlay", no_overlay_is),
        _performance_row("IS optimized overlay", post_is),
        _performance_row("OOS no overlay", no_overlay_oos),
        _performance_row("OOS optimized overlay", post_oos),
        "",
        f"- IS overlay added {int(overlay['is']['additional_blocks'])} blocks and removed {overlay['is']['incremental_blocked_total_r']:.3f}R of candidate outcomes; because this value is negative, those additional blocks were beneficial in aggregate.",
        f"- IS overlay changed return by {overlay['is']['return_delta']:.2%}, PF by {overlay['is']['profit_factor_delta']:+.3f}, win rate by {overlay['is']['win_rate_delta']:+.3%}, and daily-MTM max DD by {overlay['is']['max_drawdown_delta']:+.2%}.",
        f"- All {int(post_is['entries_blocked_by_portfolio'])} post-IS blocks totalled {post_is['blocked_total_r']:.3f}R; accepted trades averaged {post_is['accepted_avg_r']:.3f}R.",
        f"- {int(post_is['blocked_positive_count'])} blocked IS trades were positive and {int(post_is['blocked_nonpositive_count'])} were non-positive. Positive-trade blocking was {_pct(post_is['positive_alpha_block_rate'])} of positive candidates.",
        f"- OOS overlay and no-overlay results were identical. The single OOS capacity block was a positive {post_oos['blocked_total_r']:.3f}R trade, so improved OOS routing synergy was not demonstrated.",
        "",
        "## Shared-account realism",
        "",
        f"- Gross/net/overnight notional caps: {account['max_gross_notional_pct']:.2f}x / {account['max_net_notional_pct']:.2f}x / {account['max_overnight_gross_notional_pct']:.2f}x NLV.",
        f"- Symbol/position caps: {_pct(account['max_symbol_notional_pct'])} / {_pct(account['max_position_notional_pct'])}; IARIC position cap {_pct(allocations['IARIC_RESIDUAL_R3']['max_position_notional_pct'])}.",
        f"- Initial long margin {_pct(account['initial_margin_long_pct'])}; minimum admission buffer {_pct(account['minimum_margin_buffer_pct'])}; annual debit rate {_pct(account['annual_margin_interest_rate'])}.",
        f"- IS peak gross/overnight leverage: {post_is['gross_leverage_peak']:.2f}x / {post_is['overnight_gross_leverage_peak']:.2f}x; OOS: {post_oos['gross_leverage_peak']:.2f}x / {post_oos['overnight_gross_leverage_peak']:.2f}x.",
        f"- Mark coverage: IS {_pct(post_is['mark_coverage_ratio'])}, OOS {_pct(post_oos['mark_coverage_ratio'])}; margin breaches: IS {int(post_is['margin_breach_count'])}, OOS {int(post_oos['margin_breach_count'])}.",
        "- Position sizing uses causal mark-to-market NLV; whole-share quantities are floored with no forced one-share minimum; debit-cash financing is included.",
        "",
        "## Robustness and costs",
        "",
        "| Extra round-trip cost | IS return | PF | Daily-MTM max DD |",
        "|---:|---:|---:|---:|",
    ]
    for row in robustness["incremental_cost_stress"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['extra_round_trip_bps']:.0f} bps | {_pct(metrics['net_return_pct'])} | "
            f"{metrics['profit_factor']:.3f} | {_pct(metrics['max_drawdown_pct_mtm_daily'])} |"
        )
    lines.extend(
        [
            "",
            f"- All local perturbations retained positive IS PnL: `{gates['local_perturbations_positive']}`.",
            f"- Weekly block bootstrap probability of positive total PnL: {_pct(robustness['weekly_block_bootstrap']['probability_total_pnl_positive'])}.",
            f"- Complex interaction-surface CSCV PBO: {_pct(robustness['cscv_pbo']['probability_backtest_overfit'])}; the stability guard retained the simpler phase-4 incumbent.",
            "",
            "## Promotion gates and restrictions",
            "",
        ]
    )
    for key, value in gates.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "Production activation remains disallowed because:",
            "",
        ]
    )
    for restriction in summary["promotion_decision"]["research_restrictions"]:
        lines.append(f"- {restriction}.")
    lines.extend(
        [
            "",
            "## Final assessment",
            "",
            "Round 1 is the canonical realistic research baseline. The shared-account implementation removes the former duplicated-capital inflation and the IS governor improves matched-risk return and drawdown versus no overlay. However, the post configuration does not reduce drawdown versus the lower-risk pre baseline, OOS routing adds no incremental benefit, and the one OOS block rejected a winning trade. The result is suitable for further raw-signal, point-in-time and shadow-parity work, not live production.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_summary(staging: Path, matched: dict[str, Any]) -> dict[str, Any]:
    run_spec = _load(staging / "run_spec.json")
    config = _load(staging / "optimized_config.json")
    freeze = _load(staging / "freeze_receipt.json")
    robustness = _load(staging / "final_robustness.json")
    promotion = _load(staging / "promotion_decision.json")
    receipt = _load(staging / "stream_receipt.json")
    selection = freeze["interaction_selection"]
    is_data = matched["is"]
    oos_data = matched["oos"]
    post_is = is_data["post_optimization_portfolio"]
    post_oos = oos_data["post_optimization_portfolio"]
    no_overlay_is = is_data["post_optimization_no_overlay"]
    no_overlay_oos = oos_data["post_optimization_no_overlay"]

    overlay_effect = {
        "is": {
            "additional_blocks": float(post_is["entries_blocked_by_portfolio"])
            - float(no_overlay_is["entries_blocked_by_portfolio"]),
            "incremental_blocked_total_r": float(post_is["blocked_total_r"])
            - float(no_overlay_is["blocked_total_r"]),
            "return_delta": float(post_is["net_return_pct"])
            - float(no_overlay_is["net_return_pct"]),
            "profit_factor_delta": float(post_is["profit_factor"])
            - float(no_overlay_is["profit_factor"]),
            "win_rate_delta": float(post_is["win_rate"])
            - float(no_overlay_is["win_rate"]),
            "max_drawdown_delta": float(post_is["max_drawdown_pct_mtm_daily"])
            - float(no_overlay_is["max_drawdown_pct_mtm_daily"]),
        },
        "oos": {
            "additional_blocks": float(post_oos["entries_blocked_by_portfolio"])
            - float(no_overlay_oos["entries_blocked_by_portfolio"]),
            "incremental_blocked_total_r": float(post_oos["blocked_total_r"])
            - float(no_overlay_oos["blocked_total_r"]),
            "return_delta": float(post_oos["net_return_pct"])
            - float(no_overlay_oos["net_return_pct"]),
            "profit_factor_delta": float(post_oos["profit_factor"])
            - float(no_overlay_oos["profit_factor"]),
            "win_rate_delta": float(post_oos["win_rate"])
            - float(no_overlay_oos["win_rate"]),
            "max_drawdown_delta": float(post_oos["max_drawdown_pct_mtm_daily"])
            - float(no_overlay_oos["max_drawdown_pct_mtm_daily"]),
        },
    }
    return {
        "schema": "stock_portfolio_synergy_final_diagnostics_v1",
        "round": 1,
        "status": "active_research_baseline_not_production_approved",
        "initial_equity": float(run_spec["initial_equity"]),
        "is_window": run_spec["is_window"],
        "oos_window": run_spec["oos_window"],
        "config_sha256": freeze["config_sha256"],
        "selection": selection,
        "optimized_config": config,
        "matched_performance": matched,
        "pre_post_delta": {
            "is": _metric_delta(
                is_data["pre_optimization_portfolio"],
                post_is,
            ),
            "oos": _metric_delta(
                oos_data["pre_optimization_portfolio"],
                post_oos,
            ),
        },
        "overlay_effect": overlay_effect,
        "robustness": robustness,
        "promotion_decision": promotion,
        "stream_receipt": receipt,
    }


def _validate_sources(stable: Path, analysis: Path) -> None:
    for name in REQUIRED_STABLE_FILES:
        path = stable / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing stable artifact: {path}")
    matched = analysis / "matched_performance.json"
    if not matched.is_file():
        raise FileNotFoundError(f"Missing matched analysis: {matched}")

    freeze = _load(stable / "freeze_receipt.json")
    config_sha = _sha256(stable / "optimized_config.json")
    if freeze.get("config_sha256") != config_sha:
        raise ValueError("Frozen config hash does not match optimized_config.json")
    oos = _load(stable / "oos_validation.json")["metrics"]
    matched_oos = _load(matched)["oos"]["post_optimization_portfolio"]
    for key in (
        "net_return_pct",
        "total_r",
        "profit_factor",
        "win_rate",
        "max_drawdown_pct_mtm_daily",
        "total_trades",
        "entries_blocked_by_portfolio",
    ):
        if matched_oos.get(key) != oos.get(key):
            raise ValueError(f"Matched OOS mismatch for {key}")


def _write_round_artifacts(
    staging: Path,
    *,
    matched: dict[str, Any],
    archive_relative: str,
    generated_at: str,
) -> dict[str, Any]:
    shutil.copy2(
        STRATEGY_DIR / ANALYSIS_SOURCE_NAME / "matched_performance.json",
        staging / "matched_performance.json",
    )
    for source_name, target_name in (
        ("run.log", "matched_performance_run.log"),
        ("run.err.log", "matched_performance_run.err.log"),
    ):
        source = STRATEGY_DIR / ANALYSIS_SOURCE_NAME / source_name
        if source.exists():
            shutil.copy2(source, staging / target_name)

    run_spec_path = staging / "run_spec.json"
    run_spec = _load(run_spec_path)
    run_spec.update(
        {
            "round": 1,
            "canonical_round": "round_1",
            "canonicalized_at_utc": generated_at,
            "canonicalization_source": STABLE_SOURCE_NAME,
            "archive_lineage": archive_relative,
            "activation_scope": "research_only",
            "production_eligible": False,
        }
    )
    _atomic_write_json(run_spec, run_spec_path)

    promotion_path = staging / "promotion_decision.json"
    promotion = _load(promotion_path)
    promotion.update(
        {
            "saved_as_round": 1,
            "active_research_baseline": True,
            "production_activation_approved": False,
            "canonicalized_at_utc": generated_at,
        }
    )
    _atomic_write_json(promotion, promotion_path)

    summary = _build_summary(staging, matched)
    from backtests.stock.auto.portfolio_synergy.comprehensive_diagnostics import (
        enrich_summary,
    )

    summary = enrich_summary(staging, summary)
    summary["archive_lineage"] = archive_relative
    summary["generated_at_utc"] = generated_at
    _atomic_write_json(summary, staging / "diagnostics_summary.json")
    _atomic_write_json(
        summary["comprehensive_diagnostics"],
        staging / "comprehensive_synergy_diagnostics.json",
    )
    _atomic_write_json(
        {
            "round": 1,
            "status": summary["status"],
            "generated_at_utc": generated_at,
            "config_sha256": summary["config_sha256"],
            "is": summary["matched_performance"]["is"][
                "post_optimization_portfolio"
            ],
            "oos": summary["matched_performance"]["oos"][
                "post_optimization_portfolio"
            ],
        },
        staging / "final_metrics.json",
    )
    _atomic_write_json(
        {
            "family": "stock",
            "strategy": "portfolio_synergy",
            "round": 1,
            "status": summary["status"],
            "generated_at_utc": generated_at,
            "config_sha256": summary["config_sha256"],
            "selected_candidate": summary["selection"]["selected"],
            "is_window": summary["is_window"],
            "oos_window": summary["oos_window"],
            "final_metrics": {
                "is": summary["matched_performance"]["is"][
                    "post_optimization_portfolio"
                ],
                "oos": summary["matched_performance"]["oos"][
                    "post_optimization_portfolio"
                ],
            },
            "pre_post_delta": summary["pre_post_delta"],
            "overlay_effect": summary["overlay_effect"],
            "diagnostic_schema": summary["schema"],
            "synergy_assessment": summary["comprehensive_diagnostics"][
                "synergy_assessment"
            ],
            "promotion_decision": summary["promotion_decision"],
            "archive_lineage": archive_relative,
        },
        staging / "run_summary.json",
    )
    diagnostics = _render_diagnostics(summary)
    (staging / "round_final_diagnostics.md").write_text(
        diagnostics,
        encoding="utf-8",
    )
    (staging / "round_final_diagnostics.txt").write_text(
        diagnostics,
        encoding="utf-8",
    )
    (staging / "round_evaluation.txt").write_text(
        "Round 1 is retained as the canonical realistic research baseline. "
        "It passes shared-account and OOS performance gates but is not production-approved; "
        "see round_final_diagnostics.md for the full assessment.\n",
        encoding="utf-8",
    )
    return summary


def _write_artifact_manifest(round_dir: Path, summary: dict[str, Any]) -> None:
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(round_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _atomic_write_json(
        {
            "round": 1,
            "status": summary["status"],
            "active_research_baseline": True,
            "production_active": False,
            "config_sha256": summary["config_sha256"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
        round_dir / "artifact_manifest.json",
    )


def _archive_targets() -> list[Path]:
    targets = [
        path
        for path in STRATEGY_DIR.iterdir()
        if path.is_dir() and path.name.startswith("round_")
    ]
    return sorted((_safe_child(path, STRATEGY_DIR) for path in targets))


def finalize(*, execute: bool) -> dict[str, Any]:
    strategy_root = STRATEGY_DIR.resolve()
    stable = _safe_child(STRATEGY_DIR / STABLE_SOURCE_NAME, strategy_root)
    analysis = _safe_child(STRATEGY_DIR / ANALYSIS_SOURCE_NAME, strategy_root)
    _validate_sources(stable, analysis)
    targets = _archive_targets()
    target_names = [path.name for path in targets]
    if STABLE_SOURCE_NAME not in target_names or ANALYSIS_SOURCE_NAME not in target_names:
        raise ValueError("Stable source and matched analysis must both be archived")

    now = _utc_now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    generated_at = now.isoformat()
    archive_dir = _safe_child(
        STRATEGY_DIR / "archive" / f"{timestamp}_{ARCHIVE_REASON}",
        STRATEGY_DIR / "archive",
    )
    round_dir = _safe_child(STRATEGY_DIR / "round_1", strategy_root)
    staging = _safe_child(STRATEGY_DIR / f".round_1_staging_{timestamp}", strategy_root)
    archive_relative = str(archive_dir.relative_to(ROOT.resolve()))
    plan = {
        "execute": execute,
        "strategy_dir": str(strategy_root),
        "archive_dir": str(archive_dir),
        "archive_targets": [str(path) for path in targets],
        "old_manifest": str(STRATEGY_DIR / "rounds_manifest.json"),
        "source": str(stable),
        "analysis": str(analysis),
        "destination": str(round_dir),
    }
    if not execute:
        return plan
    if archive_dir.exists() or staging.exists():
        raise FileExistsError("Archive or staging destination already exists")

    shutil.copytree(stable, staging)
    matched = _load(analysis / "matched_performance.json")
    summary = _write_round_artifacts(
        staging,
        matched=matched,
        archive_relative=archive_relative,
        generated_at=generated_at,
    )

    archive_dir.mkdir(parents=True, exist_ok=False)
    for source in targets:
        target = _safe_child(archive_dir / source.name, archive_dir)
        if target.exists():
            raise FileExistsError(f"Archive target exists: {target}")
        shutil.move(str(source), str(target))
    old_manifest = STRATEGY_DIR / "rounds_manifest.json"
    if old_manifest.exists():
        shutil.move(
            str(old_manifest),
            str(archive_dir / "rounds_manifest_pre_reset.json"),
        )

    if round_dir.exists():
        raise FileExistsError(f"Canonical round destination still exists: {round_dir}")
    staging.replace(round_dir)
    archive_receipt = {
        "archived_at_utc": generated_at,
        "reason": ARCHIVE_REASON,
        "archive_dir": archive_relative,
        "archived_entries": target_names,
        "canonical_source": STABLE_SOURCE_NAME,
        "canonical_analysis": ANALYSIS_SOURCE_NAME,
        "canonical_destination": "backtests/output/stock/portfolio_synergy/round_1",
        "recoverable": True,
    }
    _atomic_write_json(archive_receipt, archive_dir / "archive_receipt.json")
    _atomic_write_json(archive_receipt, round_dir / "archive_receipt.json")
    _write_artifact_manifest(round_dir, summary)

    root_manifest = {
        "family": "stock",
        "strategy": "portfolio_synergy",
        "latest_round": 1,
        "active_round": 1,
        "active_scope": "research_only",
        "production_active_round": None,
        "generated_at_utc": generated_at,
        "archive_lineage": archive_relative,
        "rounds": [
            {
                "round": 1,
                "path": "backtests/output/stock/portfolio_synergy/round_1",
                "status": summary["status"],
                "archived": False,
                "active_research_baseline": True,
                "production_active": False,
                "config_sha256": summary["config_sha256"],
                "selected_candidate": summary["selection"]["selected"],
                "is_window": summary["is_window"],
                "oos_window": summary["oos_window"],
            }
        ],
    }
    _atomic_write_json(root_manifest, STRATEGY_DIR / "rounds_manifest.json")
    plan["archive_receipt"] = archive_receipt
    plan["config_sha256"] = summary["config_sha256"]
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Archive existing results and install the stable result as round_1.",
    )
    args = parser.parse_args()
    print(json.dumps(finalize(execute=args.execute), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
