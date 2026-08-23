"""Finalize the corrected-split IARIC residual winner as research Round 3.

This is a packaging/final-diagnostics step.  It does not run another search and
does not upgrade selection-informed March-May 2026 evidence into an untouched
validation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
IARIC_ROOT = REPO_ROOT / "backtests/output/stock/iaric"
DEFAULT_SOURCE = IARIC_ROOT / "round_2/corrected_split_targeted_extension_20260823"
DEFAULT_ROUND = IARIC_ROOT / "round_3"
DEFAULT_MANIFEST = IARIC_ROOT / "rounds_manifest.json"
CONTRACT_ID = "iaric_round3_corrected_split_residual_exact98_v1"
WINNER_SHA = "d21590d167f8365a886eedb77aa5cca0493af8b91d965f8649838ad5b892821f"
CONTROL_SHA = "304bd330cc40487dcd9595eda9cc34eb0d10c5c70982910c3a79a9e1e5280e79"
WINNER_NAME = "target_score_floor_20p0__failed_continuation_0p3"
IS_WINDOW = {"start": "2024-03-25", "end": "2026-03-01"}
OOS_WINDOW = {"start": "2026-03-02", "end": "2026-05-01"}
INITIAL_EQUITY = 100_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--round-dir", default=str(DEFAULT_ROUND))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    return parser.parse_args()


def _load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _months(window: Mapping[str, str]) -> float:
    start = date.fromisoformat(window["start"])
    end = date.fromisoformat(window["end"])
    return ((end - start).days + 1) / 30.4375


def _finite(value: float | None) -> float | None:
    return value if value is None or math.isfinite(value) else None


def _stats(trades: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(trades)
    rs = [float(row["r_multiple"]) for row in rows]
    pnls = [float(row["net_pnl"]) for row in rows]
    gains = sum(value for value in rs if value > 0.0)
    losses = abs(sum(value for value in rs if value < 0.0))
    return {
        "trades": len(rows),
        "total_r": sum(rs),
        "average_r": fmean(rs) if rs else 0.0,
        "median_r": _percentile(rs, 0.5),
        "net_pnl": sum(pnls),
        "win_rate": sum(value > 0.0 for value in rs) / len(rs) if rs else 0.0,
        "profit_factor": _finite(gains / losses if losses else None),
        "best_r": max(rs) if rs else 0.0,
        "worst_r": min(rs) if rs else 0.0,
    }


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x, mean_y = fmean(xs), fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numerator / math.sqrt(denom_x * denom_y)


def _grouped(
    trades: list[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], str]
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in trades:
        groups[str(key(trade))].append(trade)
    total_r = sum(float(row["r_multiple"]) for row in trades)
    output = []
    for name, rows in groups.items():
        item = {"group": name, **_stats(rows)}
        item["total_r_share"] = item["total_r"] / total_r if total_r else None
        output.append(item)
    return sorted(output, key=lambda row: (-int(row["trades"]), str(row["group"])))


def _score_diagnostics(trades: list[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: float(row.get("score", 0.0)))
    buckets: list[dict[str, Any]] = []
    for index in range(5):
        lo = round(index * len(ordered) / 5)
        hi = round((index + 1) * len(ordered) / 5)
        rows = ordered[lo:hi]
        scores = [float(row.get("score", 0.0)) for row in rows]
        buckets.append(
            {
                "quintile": index + 1,
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                **_stats(rows),
            }
        )
    scores = [float(row.get("score", 0.0)) for row in trades]
    failed = [float(row.get("failed_continuation_r", 0.0)) for row in trades]
    outcomes = [float(row["r_multiple"]) for row in trades]
    return {
        "contract": "selected_trade_score_diagnostics_not_rejected_opportunity_diagnostics",
        "score_outcome_pearson": _pearson(scores, outcomes),
        "failed_continuation_outcome_pearson": _pearson(failed, outcomes),
        "score_quintiles": buckets,
    }


def _realized_curve(trades: list[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: (str(row["exit_time"]), str(row["symbol"])))
    equity = INITIAL_EQUITY
    peak = equity
    maximum_drawdown = 0.0
    points = [{"time": None, "equity": equity, "drawdown_fraction": 0.0}]
    for trade in ordered:
        equity += float(trade["net_pnl"])
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        maximum_drawdown = max(maximum_drawdown, drawdown)
        points.append(
            {
                "time": trade["exit_time"],
                "equity": equity,
                "drawdown_fraction": drawdown,
                "symbol": trade["symbol"],
            }
        )
    return {
        "contract": "trade_exit_realization_curve_not_engine_mark_to_market_curve",
        "initial_equity": INITIAL_EQUITY,
        "final_equity": equity,
        "maximum_realized_drawdown_fraction": maximum_drawdown,
        "points": points,
    }


def _tail_diagnostics(trades: list[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: float(row["r_multiple"]))
    total_r = sum(float(row["r_multiple"]) for row in rows) if (rows := ordered) else 0.0
    output: dict[str, Any] = {
        "worst_trades": ordered[:10],
        "best_trades": list(reversed(ordered[-10:])),
    }
    for count in (1, 3, 5, 10):
        worst = sum(float(row["r_multiple"]) for row in ordered[:count])
        best = sum(float(row["r_multiple"]) for row in ordered[-count:])
        output[f"remove_worst_{count}"] = {
            "remaining_total_r": total_r - worst,
            "removed_total_r": worst,
        }
        output[f"remove_best_{count}"] = {
            "remaining_total_r": total_r - best,
            "removed_total_r": best,
        }
    return output


def _paired_attribution(
    control: list[Mapping[str, Any]], winner: list[Mapping[str, Any]]
) -> dict[str, Any]:
    def trade_key(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row["symbol"]), str(row["entry_time"])

    control_map = {trade_key(row): row for row in control}
    winner_map = {trade_key(row): row for row in winner}
    common = sorted(control_map.keys() & winner_map.keys())
    control_only = [control_map[key] for key in sorted(control_map.keys() - winner_map.keys())]
    winner_only = [winner_map[key] for key in sorted(winner_map.keys() - control_map.keys())]
    common_control_r = sum(float(control_map[key]["r_multiple"]) for key in common)
    common_winner_r = sum(float(winner_map[key]["r_multiple"]) for key in common)
    return {
        "trade_identity": ["symbol", "entry_time"],
        "common": {
            "trades": len(common),
            "control_total_r": common_control_r,
            "winner_total_r": common_winner_r,
            "winner_minus_control_r": common_winner_r - common_control_r,
        },
        "control_only": _stats(control_only),
        "winner_only": _stats(winner_only),
        "reconciliation_delta_r": (
            common_winner_r
            - common_control_r
            + sum(float(row["r_multiple"]) for row in winner_only)
            - sum(float(row["r_multiple"]) for row in control_only)
        ),
    }


def _settings_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    missing = "<not_set>"
    rows = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key, missing), after.get(key, missing)
        if old != new:
            rows.append({"setting": key, "before": old, "after": new})
    return rows


def _cache(source: Path, sha: str, split: str, cost: int = 20) -> dict[str, Any]:
    return _load(source / "cache" / f"{sha[:16]}__{split}__{cost}bps.json")


def _artifact_paths() -> dict[str, str]:
    names = {
        "artifact_manifest": "artifact_manifest.json",
        "baseline_config": "baseline_config.json",
        "data_contract": "data_contract.json",
        "final_candidate_comparison": "final_candidate_comparison.json",
        "final_concentration": "final_concentration.json",
        "final_cost_stress": "final_cost_stress.json",
        "final_decision_events": "final_decision_events.json",
        "final_drawdown_diagnostics": "final_drawdown_diagnostics.json",
        "final_equity_curve": "final_equity_curve.json",
        "final_exits": "final_exits.json",
        "final_fold_metrics": "final_fold_metrics.json",
        "final_metrics": "final_metrics.json",
        "final_monthly": "final_monthly.json",
        "final_robustness": "final_robustness.json",
        "final_score_diagnostics": "final_score_diagnostics.json",
        "final_sectors": "final_sectors.json",
        "final_symbols": "final_symbols.json",
        "final_trades": "final_trades.json",
        "full_final_diagnostics": "round_final_diagnostics.txt",
        "mutation_lineage": "mutation_lineage.json",
        "optimized_config": "optimized_config.json",
        "round_manifest": "round_manifest.json",
        "run_spec": "run_spec.json",
        "run_summary": "run_summary.json",
        "selection_receipt": "selection_receipt.json",
    }
    return {key: f"round_3/{value}" for key, value in names.items()}


def _diagnostic_text(summary: Mapping[str, Any]) -> str:
    control = summary["control"]
    winner = summary["winner"]
    delta = summary["delta"]
    robustness = summary["robustness"]
    slices = summary["diagnostic_slices"]
    attribution = slices["mutation_trade_attribution"]["oos"]
    tail = slices["oos_tail_sensitivity"]
    worst_sector = min(slices["oos_sectors"], key=lambda row: float(row["total_r"]))
    worst_month = min(slices["oos_monthly"], key=lambda row: float(row["total_r"]))
    best_month = max(slices["oos_monthly"], key=lambda row: float(row["total_r"]))
    catastrophic = next(
        row for row in slices["oos_exits"] if row["group"] == "catastrophic_stop"
    )
    score = slices["oos_score_diagnostics"]
    top_score_quintile = score["score_quintiles"][-1]
    lines = [
        "=" * 96,
        "IARIC RESIDUAL REVERSION — ROUND 3 FULL FINAL DIAGNOSTICS",
        "=" * 96,
        "",
        "STATUS AND EVIDENCE",
        f"Status: {summary['status']}",
        "Official active research round: yes",
        "Production/capital promotion ready: no",
        "Reason: 2026-03-02–2026-05-01 OOS informed candidate selection; its result is research evidence, not untouched validation.",
        "Next required validation: unchanged replay on chronological data after 2026-05-01.",
        "",
        "CORRECT CHRONOLOGY",
        "IS: 2024-03-25–2026-03-01",
        "OOS: 2026-03-02–2026-05-01",
        "Early OOS stress: 2026-03-02–2026-03-20",
        "Later OOS recovery: 2026-03-21–2026-05-01",
        "",
        "SELECTED CONFIGURATION",
        f"Candidate: {WINNER_NAME}",
        "Round-3 mutations vs Round-2 research control: minimum score 25 -> 20; minimum failed-continuation R 0.0 -> 0.3.",
        "All other cumulative settings remain frozen.",
        "",
        "EXACT 20 BPS RESULTS",
    ]
    for split in ("is", "oos"):
        c, w, d = control[split], winner[split], delta[split]
        lines.extend(
            [
                f"{split.upper()} control: {c['trades']} trades | {c['total_r']:+.2f}R | avg {c['average_r']:+.3f}R | PF {c['profit_factor']:.3f} | WR {100*c['win_rate']:.2f}% | engine MTM DD {100*c['engine_mtm_max_drawdown_fraction']:.2f}%",
                f"{split.upper()} winner:  {w['trades']} trades | {w['total_r']:+.2f}R | avg {w['average_r']:+.3f}R | PF {w['profit_factor']:.3f} | WR {100*w['win_rate']:.2f}% | engine MTM DD {100*w['engine_mtm_max_drawdown_fraction']:.2f}%",
                f"{split.upper()} delta:   {d['trades']:+d} trades | {d['total_r']:+.2f}R | avg {d['average_r']:+.3f}R | PF {d['profit_factor']:+.3f} | WR {100*d['win_rate']:+.2f} points | engine MTM DD {100*d['engine_mtm_max_drawdown_fraction']:+.2f} points",
            ]
        )
    lines.extend(
        [
            "",
            "OOS REGIME DISCREPANCY",
            f"Early OOS control: {control['early_oos']['trades']} trades, {control['early_oos']['total_r']:+.2f}R; winner: {winner['early_oos']['trades']} trades, {winner['early_oos']['total_r']:+.2f}R.",
            f"Later OOS control: {control['latest_oos']['trades']} trades, {control['latest_oos']['total_r']:+.2f}R; winner: {winner['latest_oos']['trades']} trades, {winner['latest_oos']['total_r']:+.2f}R.",
            "The mutation mitigates but does not eliminate the early-OOS regime loss. The headline uplift is not evidence that the initial OOS weakness has been fully repaired.",
            "",
            "MUTATION-LEVEL CAUSAL ATTRIBUTION",
            f"Common OOS entries: {attribution['common']['trades']} trades; their winner-minus-control result was {attribution['common']['winner_minus_control_r']:+.6f}R (numerically zero).",
            f"The mutation removed {attribution['control_only']['trades']} control-only trades worth {attribution['control_only']['total_r']:+.2f}R and admitted {attribution['winner_only']['trades']} winner-only trades worth {attribution['winner_only']['total_r']:+.2f}R, reconciling to the {attribution['reconciliation_delta_r']:+.2f}R uplift.",
            "The uplift therefore comes from a better replacement/admission set, not altered exits on common trades. Lowering the score floor alone is not the mechanism; its interaction with the 0.3R failed-continuation filter is material.",
            "",
            "EDGE-CASE AND CONCENTRATION TEST",
            f"The best one/three/five OOS trades contributed {tail['remove_best_1']['removed_total_r']:+.2f}R, {tail['remove_best_3']['removed_total_r']:+.2f}R and {tail['remove_best_5']['removed_total_r']:+.2f}R. Removing the best five leaves only {tail['remove_best_5']['remaining_total_r']:+.2f}R.",
            f"The worst five contributed {tail['remove_worst_5']['removed_total_r']:+.2f}R; removing them raises OOS to {tail['remove_worst_5']['remaining_total_r']:+.2f}R. Catastrophic stops were {catastrophic['trades']} trades and {catastrophic['total_r']:+.2f}R.",
            "This is positive but highly winner-concentrated OOS performance. The main fragility is dependence on a handful of large winners alongside a small catastrophic-stop loss cluster—not one isolated loss that can safely be patched away.",
            "",
            "REGIME, SECTOR, AND SCORE WEAKNESS",
            f"Worst month: {worst_month['group']} at {worst_month['total_r']:+.2f}R; best month: {best_month['group']} at {best_month['total_r']:+.2f}R. Worst sector: {worst_sector['group']} with {worst_sector['trades']} trades and {worst_sector['total_r']:+.2f}R.",
            f"Selected-trade score/outcome correlation was {score['score_outcome_pearson']:+.3f}; the highest score quintile returned {top_score_quintile['total_r']:+.2f}R across {top_score_quintile['trades']} trades.",
            "The OOS score ordering is non-monotonic and the Healthcare/early-March weakness is economically important. This supports the score-floor ablation result and argues against treating higher composite score as universally safer.",
            "",
            "ROBUSTNESS",
            f"30 bps OOS: {robustness['oos_cost_30bps']['trades']} trades, {robustness['oos_cost_30bps']['total_r']:+.2f}R.",
            f"40 bps OOS: {robustness['oos_cost_40bps']['trades']} trades, {robustness['oos_cost_40bps']['total_r']:+.2f}R.",
            f"Paired block bootstrap: P(delta > 0)={100*robustness['paired_oos_bootstrap']['probability_positive']:.2f}%; 95% CI per entry-day [{robustness['paired_oos_bootstrap']['ci_95'][0]:+.3f}, {robustness['paired_oos_bootstrap']['ci_95'][1]:+.3f}]R.",
            "The cost result is positive, but the bootstrap interval crosses zero and the OOS sample is only 46 trades.",
            "",
            "SEARCH COVERAGE",
            f"149 prior candidates corrected-rescored; 20 existing exact finalists; 120 new targeted candidates; 138 unique exact settings; {summary['eligible_candidates']} candidates passed every registered gate; four received final cost/bootstrap robustness.",
            "The previously recommended z=1.10/score=20 candidate remained eligible for selection and was explicitly retested; it was not the corrected winner.",
            "",
            "DRAWDOWN DEFINITIONS",
            "Engine MTM drawdown is the authoritative within-position portfolio drawdown.",
            "Close-to-close trade drawdown is the eligibility statistic used in the targeted comparison.",
            "The serialized equity curve is exit-realized only and is labelled accordingly; it must not be substituted for engine MTM drawdown.",
            "",
            "FINAL VERDICT",
            "Round 3 is the best gate-qualified research reference in the completed corrected-split search. It improves IS total R and average R while reducing frequency modestly; it materially improves the observed OOS total/average R, win rate, profit factor and drawdown. It is not a deployment-ready promotion because OOS informed selection, the early-OOS loss remains material, and paired uncertainty includes zero.",
            "",
            "=" * 96,
        ]
    )
    return "\n".join(lines) + "\n"


def finalize(source: Path, round_dir: Path, manifest_path: Path) -> dict[str, Any]:
    source = source.resolve()
    round_dir = round_dir.resolve()
    manifest_path = manifest_path.resolve()
    if round_dir.exists() and any(round_dir.iterdir()):
        existing = round_dir / "round_manifest.json"
        if not existing.is_file() or _load(existing).get("contract_id") != CONTRACT_ID:
            raise RuntimeError(f"refusing to overwrite unrelated non-empty round directory: {round_dir}")
    round_dir.mkdir(parents=True, exist_ok=True)

    recommendation = _load(source / "recommended_research_config.json")
    results = _load(source / "corrected_targeted_results.json")
    catalog = _load(source / "exact_candidate_catalog.json")
    receipt = _load(source / "data_consumption_receipt.json")
    source_spec = _load(source / "run_spec.json")
    if recommendation["candidate_name"] != WINNER_NAME or recommendation["settings_sha256"] != WINNER_SHA:
        raise ValueError("source recommendation does not match the registered Round-3 winner")
    winner_row = next(row for row in results["comparison"] if row["name"] == WINNER_NAME)
    if not winner_row["eligibility"]["passed"]:
        raise ValueError("registered Round-3 winner no longer passes the exact gates")

    caches = {
        "control_is": _cache(source, CONTROL_SHA, "is"),
        "control_oos": _cache(source, CONTROL_SHA, "oos"),
        "winner_is": _cache(source, WINNER_SHA, "is"),
        "winner_oos": _cache(source, WINNER_SHA, "oos"),
        "winner_oos_30bps": _cache(source, WINNER_SHA, "oos", 30),
        "winner_oos_40bps": _cache(source, WINNER_SHA, "oos", 40),
    }
    winner_settings = recommendation["settings"]
    control_settings = next(row["settings"] for row in catalog if row["name"] == "current")
    round1_config = _load(IARIC_ROOT / "round_1/optimized_config.json")

    control_metrics: dict[str, Any] = {
        "is": {**results["control"]["is"], "engine_mtm_max_drawdown_fraction": caches["control_is"]["metrics"]["max_drawdown_pct"], "return_fraction": caches["control_is"]["metrics"]["return_pct"]},
        "oos": {**results["control"]["oos"]["oos"], "engine_mtm_max_drawdown_fraction": caches["control_oos"]["metrics"]["max_drawdown_pct"], "return_fraction": caches["control_oos"]["metrics"]["return_pct"]},
        "early_oos": results["control"]["oos"]["early_oos"],
        "latest_oos": results["control"]["oos"]["latest_oos"],
    }
    winner_metrics: dict[str, Any] = {
        "is": {**winner_row["is"], "engine_mtm_max_drawdown_fraction": caches["winner_is"]["metrics"]["max_drawdown_pct"], "return_fraction": caches["winner_is"]["metrics"]["return_pct"]},
        "oos": {**winner_row["oos"]["oos"], "engine_mtm_max_drawdown_fraction": caches["winner_oos"]["metrics"]["max_drawdown_pct"], "return_fraction": caches["winner_oos"]["metrics"]["return_pct"]},
        "early_oos": winner_row["oos"]["early_oos"],
        "latest_oos": winner_row["oos"]["latest_oos"],
    }
    metric_keys = (
        "trades", "total_r", "average_r", "profit_factor", "win_rate", "net_pnl",
        "close_to_close_trade_drawdown_pct", "engine_mtm_max_drawdown_fraction", "return_fraction",
    )
    deltas = {
        split: {key: winner_metrics[split][key] - control_metrics[split][key] for key in metric_keys}
        for split in ("is", "oos")
    }
    selected_robustness = results["selected_research_candidate"]
    robustness = {
        "oos_cost_20bps": winner_metrics["oos"],
        "oos_cost_30bps": selected_robustness["oos_cost30"]["oos"],
        "oos_cost_40bps": selected_robustness["oos_cost40"]["oos"],
        "paired_oos_bootstrap": selected_robustness["paired_oos_bootstrap"],
        "finalists": results["finalist_robustness"],
    }
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    optimized = {
        "configuration_role": "round3_corrected_split_joint_is_oos_research_reference",
        "contract_id": CONTRACT_ID,
        "official_round": 3,
        "promotion_ready": False,
        "research_only": True,
        "selection_informed_oos": True,
        "candidate_name": WINNER_NAME,
        "patch_vs_round2_research_control": recommendation["patch_vs_round2"],
        "settings": winner_settings,
        "sha256": WINNER_SHA,
        "validation_required": "untouched chronological data after 2026-05-01",
    }
    baseline = {
        "configuration_role": "round2_research_control_not_canonical_round",
        "settings": control_settings,
        "sha256": CONTROL_SHA,
        "source": "round_2 phased research control",
    }
    lineage = {
        "contract": "all_cumulative_accepted_mutations_round1_through_round3_v1",
        "round_1_official_to_round_2_research_control": _settings_diff(round1_config["settings"], control_settings),
        "round_2_research_control_to_round_3": _settings_diff(control_settings, winner_settings),
        "round_3_accepted_mutations": recommendation["patch_vs_round2"],
        "round_2_canonical_status": "research_only_not_added_as_official_manifest_round",
        "full_round_3_settings": winner_settings,
    }
    selection_receipt = {
        "contract": "selection_informed_oos_research_receipt_v1",
        "selected_candidate": WINNER_NAME,
        "selection_windows": {"is": IS_WINDOW, "oos": OOS_WINDOW},
        "oos_accessed": True,
        "oos_used_for_selection": True,
        "promotion_ready": False,
        "candidate_counts": {
            "previous_candidates_corrected_rescored": source_spec["existing_candidates_rescored"],
            "existing_exact_shortlist": source_spec["exact_existing_shortlist"],
            "new_targeted_candidates": source_spec["genuinely_new_targeted_candidates"],
            "unique_exact_settings": len(catalog),
            "eligible": results["eligible_count"],
            "robustness_finalists": len(results["finalist_robustness"]),
        },
        "next_untouched_validation_start": "2026-05-02",
        "selection_caveat": results["selection_caveat"],
    }
    data_contract = {
        **receipt,
        "authority": "project_official_local_snapshot_consumed_for_research",
        "is_window": IS_WINDOW,
        "oos_window": OOS_WINDOW,
    }
    run_spec = {
        "contract_id": CONTRACT_ID,
        "family": "stock",
        "strategy": "iaric",
        "round": 3,
        "execution_contract": "iaric_daily_residual_execution_v2",
        "costs_bps": 20.0,
        "is_window": IS_WINDOW,
        "oos_window": OOS_WINDOW,
        "early_oos_window": {"start": "2026-03-02", "end": "2026-03-20"},
        "latest_oos_window": {"start": "2026-03-21", "end": "2026-05-01"},
        "research_only": True,
        "promotion_ready": False,
        "source_research": str(source.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_contract": source_spec,
        "settings_sha256": WINNER_SHA,
        "generated_at_utc": now,
    }

    winner_is_trades = caches["winner_is"]["trades"]
    winner_oos_trades = caches["winner_oos"]["trades"]
    control_is_trades = caches["control_is"]["trades"]
    control_oos_trades = caches["control_oos"]["trades"]
    tagged_trades = [
        {"split": split, **trade}
        for split, rows in (("is", winner_is_trades), ("oos", winner_oos_trades))
        for trade in rows
    ]
    tagged_events = [
        {"split": split, **event}
        for split, rows in (("is", caches["winner_is"]["decision_events"]), ("oos", caches["winner_oos"]["decision_events"]))
        for event in rows
    ]
    by_split = {"is": winner_is_trades, "oos": winner_oos_trades}
    concentration = {
        split: _tail_diagnostics(rows)
        for split, rows in by_split.items()
    }
    concentration["mutation_trade_attribution"] = {
        "is": _paired_attribution(control_is_trades, winner_is_trades),
        "oos": _paired_attribution(control_oos_trades, winner_oos_trades),
    }
    exits = {split: _grouped(rows, lambda row: str(row["exit_reason"])) for split, rows in by_split.items()}
    monthly = {split: _grouped(rows, lambda row: str(row["entry_date"])[:7]) for split, rows in by_split.items()}
    symbols = {split: _grouped(rows, lambda row: str(row["symbol"])) for split, rows in by_split.items()}
    sectors = {split: _grouped(rows, lambda row: str(row["sector"])) for split, rows in by_split.items()}
    scores = {split: _score_diagnostics(rows) for split, rows in by_split.items()}
    curves = {split: _realized_curve(rows) for split, rows in by_split.items()}
    drawdowns = {
        split: {
            "engine_mtm_max_drawdown_fraction": winner_metrics[split]["engine_mtm_max_drawdown_fraction"],
            "close_to_close_trade_drawdown_fraction": winner_metrics[split]["close_to_close_trade_drawdown_pct"],
            "realized_exit_equity_max_drawdown_fraction": curves[split]["maximum_realized_drawdown_fraction"],
            "definitions_are_not_interchangeable": True,
        }
        for split in ("is", "oos")
    }
    comparison = {
        "contract": results["contract"],
        "selected_candidate": WINNER_NAME,
        "eligible_count": results["eligible_count"],
        "unique_exact_settings": len(catalog),
        "comparison": results["comparison"],
        "candidate_catalog": catalog,
    }
    final_metrics = {
        "control": control_metrics,
        "winner": winner_metrics,
        "winner_minus_control": deltas,
        "headline": {
            "is": f"{winner_metrics['is']['trades']} trades, {winner_metrics['is']['total_r']:+.2f}R, PF {winner_metrics['is']['profit_factor']:.2f}",
            "oos": f"{winner_metrics['oos']['trades']} trades, {winner_metrics['oos']['total_r']:+.2f}R, PF {winner_metrics['oos']['profit_factor']:.2f}",
        },
    }
    summary = {
        "family": "stock",
        "strategy": "iaric",
        "round": 3,
        "status": "complete_round3_corrected_split_research_reference",
        "official_active_research_round": True,
        "promotion_ready": False,
        "research_only": True,
        "generated_at_utc": now,
        "candidate": WINNER_NAME,
        "settings_sha256": WINNER_SHA,
        "control": control_metrics,
        "winner": winner_metrics,
        "delta": deltas,
        "robustness": robustness,
        "eligible_candidates": results["eligible_count"],
        "candidate_counts": selection_receipt["candidate_counts"],
        "mutation_lineage": lineage,
        "selection_receipt": selection_receipt,
        "diagnostic_verdict": {
            "oos_underperformance_fully_eliminated": False,
            "early_oos_regime_loss_mitigated_not_removed": True,
            "winner_is_best_completed_gate_qualified_candidate": True,
            "bootstrap_ci_excludes_zero": False,
            "untouched_forward_validation_required": True,
        },
        "diagnostic_slices": {
            "mutation_trade_attribution": concentration["mutation_trade_attribution"],
            "oos_tail_sensitivity": {
                key: value
                for key, value in concentration["oos"].items()
                if key.startswith("remove_")
            },
            "oos_monthly": monthly["oos"],
            "oos_sectors": sectors["oos"],
            "oos_exits": exits["oos"],
            "oos_score_diagnostics": scores["oos"],
        },
    }

    _write(round_dir / "optimized_config.json", optimized)
    _write(round_dir / "baseline_config.json", baseline)
    _write(round_dir / "mutation_lineage.json", lineage)
    _write(round_dir / "selection_receipt.json", selection_receipt)
    _write(round_dir / "data_contract.json", data_contract)
    _write(round_dir / "run_spec.json", run_spec)
    _write(round_dir / "final_metrics.json", final_metrics)
    _write(round_dir / "final_fold_metrics.json", {"control": control_metrics, "winner": winner_metrics, "delta": deltas})
    _write(round_dir / "final_cost_stress.json", robustness)
    _write(round_dir / "final_robustness.json", robustness)
    _write(round_dir / "final_concentration.json", concentration)
    _write(round_dir / "final_exits.json", exits)
    _write(round_dir / "final_monthly.json", monthly)
    _write(round_dir / "final_symbols.json", symbols)
    _write(round_dir / "final_sectors.json", sectors)
    _write(round_dir / "final_score_diagnostics.json", scores)
    _write(round_dir / "final_drawdown_diagnostics.json", drawdowns)
    _write(round_dir / "final_equity_curve.json", curves)
    _write(round_dir / "final_trades.json", tagged_trades)
    _write(round_dir / "final_decision_events.json", tagged_events)
    _write(round_dir / "final_candidate_comparison.json", comparison)
    _write(round_dir / "run_summary.json", summary)
    (round_dir / "round_final_diagnostics.txt").write_text(_diagnostic_text(summary), encoding="utf-8")

    artifacts = _artifact_paths()
    round_manifest = {
        "active": True,
        "artifacts": artifacts,
        "baseline_eligible": False,
        "configuration_role": optimized["configuration_role"],
        "contract_id": CONTRACT_ID,
        "data_authority": data_contract["authority"],
        "execution_contract": run_spec["execution_contract"],
        "family": "stock",
        "headline": final_metrics["headline"],
        "metrics": {"is": winner_metrics["is"], "oos": winner_metrics["oos"]},
        "official": True,
        "promotion_ready": False,
        "research_only": True,
        "representative_alpha_baseline": False,
        "round": 3,
        "round_2_lineage": "research_only_not_canonicalized",
        "selection_informed_oos": True,
        "status": summary["status"],
        "strategy": "iaric",
        "training_window": IS_WINDOW,
        "validation": {
            "costs_bps": 20.0,
            "evidence_class": "selection_informed_oos_research",
            "oos_window": OOS_WINDOW,
            "oos_accessed": True,
            "oos_used_for_selection": True,
            "paired_bootstrap_ci_excludes_zero": False,
            "untouched_validation_required_after": "2026-05-01",
        },
    }
    _write(round_dir / "round_manifest.json", round_manifest)

    artifact_manifest: dict[str, Any] = {
        "contract": "sha256_immutable_round_artifact_manifest_v1",
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "files": {},
    }
    for path in sorted(round_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_manifest["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write(round_dir / "artifact_manifest.json", artifact_manifest)

    root_manifest = _load(manifest_path)
    existing_round3 = [row for row in root_manifest.get("rounds", []) if int(row.get("round", 0)) == 3]
    if existing_round3 and any(row.get("contract_id") != CONTRACT_ID for row in existing_round3):
        raise RuntimeError("refusing to replace an unrelated Round-3 manifest entry")
    for row in root_manifest.get("rounds", []):
        row["active"] = False
    root_manifest["rounds"] = [row for row in root_manifest.get("rounds", []) if int(row.get("round", 0)) != 3]
    artifact_sha = {
        key: _sha256(round_dir / Path(relative).name)
        for key, relative in artifacts.items()
        if (round_dir / Path(relative).name).is_file()
    }
    root_entry = {
        "active": True,
        "artifact_sha256": artifact_sha,
        "artifacts": artifacts,
        "average_r": winner_metrics["is"]["average_r"],
        "baseline_eligible": False,
        "configuration_role": optimized["configuration_role"],
        "contract_id": CONTRACT_ID,
        "data_authority": data_contract["authority"],
        "headline": f"IS {winner_metrics['is']['trades']} trades/{winner_metrics['is']['total_r']:+.2f}R; OOS {winner_metrics['oos']['trades']} trades/{winner_metrics['oos']['total_r']:+.2f}R",
        "max_drawdown_fraction": winner_metrics["is"]["engine_mtm_max_drawdown_fraction"],
        "official": True,
        "profit_factor": winner_metrics["is"]["profit_factor"],
        "promotion_allowed": False,
        "promotion_ready": False,
        "representative_alpha_baseline": False,
        "research_only": True,
        "round": 3,
        "round_2_lineage": "research_only_not_canonicalized",
        "sealed_holdout": {"accessed": True, "start": OOS_WINDOW["start"], "used_for_selection": True},
        "status": summary["status"],
        "timestamp": now,
        "total_r": winner_metrics["is"]["total_r"],
        "total_trades": winner_metrics["is"]["trades"],
        "trades_per_month": winner_metrics["is"]["trades_per_month"],
        "training_window": IS_WINDOW,
        "validation_contract": round_manifest["validation"],
        "oos_metrics": winner_metrics["oos"],
    }
    root_manifest["rounds"].append(root_entry)
    root_manifest["rounds"].sort(key=lambda row: int(row.get("round", 0)))
    root_manifest["active_round"] = 3
    root_manifest["generated_at_utc"] = now
    root_manifest["round_sequence_note"] = "Round 2 remained a research workspace and was not canonicalized; the corrected-split result is the next official artifact package."
    _write(manifest_path, root_manifest)
    return {"round_dir": str(round_dir), "manifest": str(manifest_path), "headline": root_entry["headline"]}


def main() -> None:
    args = _parse_args()
    result = finalize(Path(args.source), Path(args.round_dir), Path(args.manifest))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
