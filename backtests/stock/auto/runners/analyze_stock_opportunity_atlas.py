"""Walk-forward aperture, entry, and horizon audit for the opportunity atlas.

Selection is hierarchical to control multiplicity: choose an economically
interpretable aperture on the early fold using the unconditional next-open /
bar-12 outcome, choose a causal entry inside that aperture, then choose a
holding horizon. Middle and latest folds confirm the frozen choice, its local
neighbours, and its ability to reject lower-quality signals.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ATLAS = REPO_ROOT / "backtests/output/stock/opportunity_atlas/round_1"
DEFAULT_OUTPUT = DEFAULT_ATLAS / "walk_forward"
HORIZONS = ("bar_3", "bar_6", "bar_12", "bar_24", "bar_48", "eod")
ENTRY_VARIANTS = ("next_bar_open", "one_bar_confirmation", "resting_25pct_retrace")


def _policy_all(record: dict[str, Any]) -> bool:
    return True


def _policy_score_40(record: dict[str, Any]) -> bool:
    return float(record["score"]) >= 40.0


def _policy_score_50(record: dict[str, Any]) -> bool:
    return float(record["score"]) >= 50.0


def _policy_score_60(record: dict[str, Any]) -> bool:
    return float(record["score"]) >= 60.0


def _policy_score_70(record: dict[str, Any]) -> bool:
    return float(record["score"]) >= 70.0


def _policy_geometry(record: dict[str, Any]) -> bool:
    components = record["score_components"]
    return (
        float(record["score"]) >= 40.0
        and float(components["reclaim"]) >= 0.40
        and float(components["close_quality"]) >= 0.60
    )


def _policy_participation(record: dict[str, Any]) -> bool:
    components = record["score_components"]
    return (
        float(record["score"]) >= 40.0
        and float(components["relative_volume"]) >= 0.25
    )


# These apertures reuse the immutable seven-component score. The two
# conditional routes test whether a lower floor can retain breadth while an
# orthogonal geometry/participation condition rejects poor events.
APERTURES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "all_events": _policy_all,
    "score_40": _policy_score_40,
    "score_50": _policy_score_50,
    "score_60": _policy_score_60,
    "score_70": _policy_score_70,
    "score_40_geometry": _policy_geometry,
    "score_40_participation": _policy_participation,
}

APERTURE_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "all_events": ("score_40",),
    "score_40": ("all_events", "score_50"),
    "score_50": ("score_40", "score_60"),
    "score_60": ("score_50", "score_70"),
    "score_70": ("score_60",),
    "score_40_geometry": ("score_40",),
    "score_40_participation": ("score_40",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", default=str(DEFAULT_ATLAS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--wait-for-pid", type=int, default=0)
    parser.add_argument("--bootstrap-simulations", type=int, default=5000)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wait_for_pid(pid: int) -> None:
    if pid <= 0:
        return
    print(f"queued behind PID {pid}", flush=True)
    if os.name == "nt":
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", f"Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue"],
            check=False,
        )
        return
    import time
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(10.0)


def _records_for_entry_variant(
    records: list[dict[str, Any]], entry_variant: str,
) -> list[dict[str, Any]]:
    """Overlay a causal entry outcome while preserving signal attributes."""

    result: list[dict[str, Any]] = []
    for record in records:
        if entry_variant == "next_bar_open" and not record.get("entry_variants"):
            result.append(record)
            continue
        outcome = record.get("entry_variants", {}).get(entry_variant)
        if not isinstance(outcome, dict):
            continue
        overlaid = deepcopy(record)
        for key in (
            "entry_price", "risk_per_share", "cost_r", "stop_target_r",
            "bars_to_terminal", "mfe_r", "mae_r", "horizon_r",
        ):
            overlaid[key] = deepcopy(outcome[key])
        overlaid["selected_entry_variant"] = entry_variant
        result.append(overlaid)
    return result


def _profit_factor(values: list[float]) -> float:
    gains = sum(max(value, 0.0) for value in values)
    losses = -sum(min(value, 0.0) for value in values)
    return gains / losses if losses > 0.0 else (float("inf") if gains > 0.0 else 0.0)


def _metrics(records: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    values = [float(record["horizon_r"][horizon]) for record in records]
    if not values:
        return {"events": 0, "avg_r": 0.0, "total_r": 0.0, "profit_factor": 0.0}
    return {
        "events": len(values),
        "avg_r": fmean(values),
        "total_r": sum(values),
        "profit_factor": _profit_factor(values),
    }


def _bootstrap_probability_positive(
    records: list[dict[str, Any]], horizon: str, simulations: int,
) -> float:
    by_day: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_day[str(record["date"])].append(float(record["horizon_r"][horizon]))
    daily = [fmean(by_day[key]) for key in sorted(by_day)]
    if not daily:
        return 0.0
    rng = random.Random(20260820)
    positive = 0
    for _ in range(simulations):
        sample = [daily[rng.randrange(len(daily))] for _ in daily]
        positive += fmean(sample) > 0.0
    return positive / simulations


def _select_aperture(family_records: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    early = [record for record in family_records if record["fold"] == "early"]
    development: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for name, predicate in APERTURES.items():
        samples = [record for record in early if predicate(record)]
        metrics = _metrics(samples, "bar_12")
        metrics["development_objective"] = metrics["avg_r"] * math.sqrt(metrics["events"])
        metrics["eligible"] = metrics["events"] >= 30 and metrics["avg_r"] > 0.0
        development[name] = metrics
        if metrics["eligible"]:
            eligible.append((float(metrics["development_objective"]), name))
    if not eligible:
        return None, development
    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return eligible[0][1], development


def _select_entry_variant(
    selected_records: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    early = [record for record in selected_records if record["fold"] == "early"]
    development: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for variant in ENTRY_VARIANTS:
        samples = _records_for_entry_variant(early, variant)
        metrics = _metrics(samples, "bar_12")
        metrics["fill_rate"] = len(samples) / len(early) if early else 0.0
        metrics["development_objective"] = metrics["avg_r"] * math.sqrt(metrics["events"])
        metrics["eligible"] = (
            metrics["events"] >= 30
            and metrics["avg_r"] > 0.0
            and metrics["fill_rate"] >= 0.35
        )
        development[variant] = metrics
        if metrics["eligible"]:
            eligible.append((float(metrics["development_objective"]), variant))
    if not eligible:
        return None, development
    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return eligible[0][1], development


def _select_horizon(selected_early: list[dict[str, Any]]) -> str:
    ranked = sorted(
        ((_metrics(selected_early, horizon)["avg_r"], -index, horizon)
         for index, horizon in enumerate(HORIZONS)),
        reverse=True,
    )
    return ranked[0][2]


def _selection_surface(records: list[dict[str, Any]]) -> dict[str, Any]:
    surface: dict[str, Any] = {}
    for aperture, predicate in APERTURES.items():
        aperture_records = [record for record in records if predicate(record)]
        surface[aperture] = {}
        for variant in ENTRY_VARIANTS:
            variant_records = _records_for_entry_variant(aperture_records, variant)
            surface[aperture][variant] = {
                "fill_rate": len(variant_records) / len(aperture_records) if aperture_records else 0.0,
                "horizons": {
                    horizon: {
                        fold: _metrics(
                            [record for record in variant_records if record["fold"] == fold], horizon,
                        )
                        for fold in ("early", "middle", "latest")
                    }
                    for horizon in HORIZONS
                },
            }
    return surface


def _alpha_funnel(
    records: list[dict[str, Any]], aperture: str, variant: str, horizon: str,
) -> dict[str, Any]:
    validation = [record for record in records if record["fold"] in {"middle", "latest"}]
    filled = _records_for_entry_variant(validation, variant)
    predicate = APERTURES[aperture]
    accepted = [record for record in filled if predicate(record)]
    rejected = [record for record in filled if not predicate(record)]
    accepted_metrics = _metrics(accepted, horizon)
    rejected_metrics = _metrics(rejected, horizon)
    accepted_positive_r = sum(max(float(row["horizon_r"][horizon]), 0.0) for row in accepted)
    rejected_positive_r = sum(max(float(row["horizon_r"][horizon]), 0.0) for row in rejected)
    total_positive_r = accepted_positive_r + rejected_positive_r
    return {
        "accepted": accepted_metrics,
        "rejected": rejected_metrics,
        "accepted_positive_event_precision": (
            sum(float(row["horizon_r"][horizon]) > 0.0 for row in accepted) / len(accepted)
            if accepted else 0.0
        ),
        "accepted_positive_r_recall": accepted_positive_r / total_positive_r if total_positive_r > 0.0 else 0.0,
        "rejected_positive_r_share": rejected_positive_r / total_positive_r if total_positive_r > 0.0 else 0.0,
        "discrimination_lift_r": accepted_metrics["avg_r"] - rejected_metrics["avg_r"],
        "rejected_sample_available": bool(rejected),
    }


def _neighbour_stability(
    records: list[dict[str, Any]], aperture: str, variant: str, horizon: str,
) -> dict[str, Any]:
    horizon_index = HORIZONS.index(horizon)
    horizon_names = tuple(dict.fromkeys(
        HORIZONS[index]
        for index in (max(0, horizon_index - 1), min(len(HORIZONS) - 1, horizon_index + 1))
        if index != horizon_index
    ))
    predicate = APERTURES[aperture]
    selected = _records_for_entry_variant([row for row in records if predicate(row)], variant)
    horizon_rows = {
        neighbour: {
            fold: _metrics([row for row in selected if row["fold"] == fold], neighbour)
            for fold in ("middle", "latest")
        }
        for neighbour in horizon_names
    }
    aperture_rows = {}
    for neighbour in APERTURE_NEIGHBOURS[aperture]:
        neighbour_records = _records_for_entry_variant(
            [row for row in records if APERTURES[neighbour](row)], variant,
        )
        aperture_rows[neighbour] = {
            fold: _metrics([row for row in neighbour_records if row["fold"] == fold], horizon)
            for fold in ("middle", "latest")
        }
    def stable(rows: dict[str, Any]) -> bool:
        for fold in ("middle", "latest"):
            populated = [row[fold] for row in rows.values() if row[fold]["events"] >= 20]
            if not populated or any(sample["avg_r"] <= 0.0 for sample in populated):
                return False
        return True

    return {
        "horizon_neighbours": horizon_rows,
        "aperture_neighbours": aperture_rows,
        "horizon_neighbours_positive": stable(horizon_rows),
        "aperture_neighbours_positive": stable(aperture_rows),
    }


def _audit_family(records: list[dict[str, Any]], simulations: int) -> dict[str, Any]:
    aperture, aperture_development = _select_aperture(records)
    surface = _selection_surface(records)
    if aperture is None:
        return {
            "selected_aperture": None,
            "aperture_development": aperture_development,
            "selection_surface": surface,
            "route_ready_for_portfolio_replay": False,
            "failed_checks": ["no_positive_early_aperture_with_30_events"],
        }
    aperture_records = [record for record in records if APERTURES[aperture](record)]
    variant, entry_development = _select_entry_variant(aperture_records)
    if variant is None:
        return {
            "selected_aperture": aperture,
            "selected_entry_variant": None,
            "aperture_development": aperture_development,
            "entry_development": entry_development,
            "selection_surface": surface,
            "route_ready_for_portfolio_replay": False,
            "failed_checks": ["no_positive_causal_entry_with_30_events_and_35pct_fill"],
        }
    selected = _records_for_entry_variant(aperture_records, variant)
    selected_early = [record for record in selected if record["fold"] == "early"]
    horizon = _select_horizon(selected_early)
    folds = {
        fold: _metrics([record for record in selected if record["fold"] == fold], horizon)
        for fold in ("early", "middle", "latest")
    }
    validation = [record for record in selected if record["fold"] in {"middle", "latest"}]
    probability = _bootstrap_probability_positive(validation, horizon, simulations)
    fill_rate = len(selected) / len(aperture_records) if aperture_records else 0.0
    stability = _neighbour_stability(records, aperture, variant, horizon)
    funnel = _alpha_funnel(records, aperture, variant, horizon)
    checks = {
        "middle_at_least_25_filled_events": folds["middle"]["events"] >= 25,
        "latest_at_least_25_filled_events": folds["latest"]["events"] >= 25,
        "middle_positive": folds["middle"]["avg_r"] > 0.0,
        "latest_positive": folds["latest"]["avg_r"] > 0.0,
        "entry_fill_rate_at_least_35pct": fill_rate >= 0.35,
        "validation_bootstrap_probability_at_least_90pct": probability >= 0.90,
        "horizon_neighbours_positive": stability["horizon_neighbours_positive"],
        "aperture_neighbours_positive": stability["aperture_neighbours_positive"],
        "rejected_sample_available": funnel["rejected_sample_available"],
        "accepted_alpha_positive": funnel["accepted"]["avg_r"] > 0.0,
        "positive_discrimination_lift": funnel["discrimination_lift_r"] > 0.0,
    }
    return {
        "selected_aperture": aperture,
        "selected_entry_variant": variant,
        "selected_horizon": horizon,
        "aperture_development": aperture_development,
        "entry_development": entry_development,
        "selection_surface": surface,
        "fill_rate": fill_rate,
        "folds": folds,
        "validation_bootstrap_probability_positive": probability,
        "neighbour_stability": stability,
        "alpha_funnel": funnel,
        "checks": checks,
        "route_ready_for_portfolio_replay": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "boundary": "exact shared-core portfolio replay only; no production promotion and no holdout access",
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Opportunity Atlas Walk-Forward Audit",
        "",
        "Aperture, entry, and horizon are chosen sequentially on the early fold. Middle and latest are "
        "confirmation folds and must also support neighbouring choices and positive accepted-versus-rejected "
        "discrimination. The sealed holdout is not accessed.",
        "",
        "| Family | Aperture | Entry | Horizon | Middle avg R | Latest avg R | Rejected alpha share | Replay? |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for family, row in sorted(summary["families"].items()):
        if row["selected_aperture"] is None or row.get("selected_entry_variant") is None:
            lines.append(f"| {family} | none | - | - | - | - | - | no |")
            continue
        folds = row["folds"]
        lines.append(
            f"| {family} | {row['selected_aperture']} | {row['selected_entry_variant']} | "
            f"{row['selected_horizon']} | {folds['middle']['avg_r']:+.3f} | "
            f"{folds['latest']['avg_r']:+.3f} | "
            f"{row['alpha_funnel']['rejected_positive_r_share']:.1%} | "
            f"{'yes' if row['route_ready_for_portfolio_replay'] else 'no'} |"
        )
    lines.extend([
        "",
        "A passing row is next tested through the shared strategy core with shared capital, mark-to-market "
        "drawdown, realistic execution stress, route-specific entries/exits, and parity tests. Event-level "
        "results are never presented as portfolio performance.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    atlas_dir = Path(args.atlas_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_for_pid > 0:
        _write_json(output_dir / "queue_status.json", {
            "status": "queued",
            "waiting_for_pid": args.wait_for_pid,
            "queued_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        _wait_for_pid(args.wait_for_pid)
    summary_path = atlas_dir / "atlas_summary.json"
    events_path = atlas_dir / "events.jsonl"
    if not summary_path.exists() or not events_path.exists():
        raise FileNotFoundError("completed atlas_summary.json and events.jsonl are required")
    atlas_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if atlas_summary.get("holdout_accessed") is not False:
        raise ValueError("atlas does not prove that the sealed holdout remained untouched")
    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    families = sorted({str(record["family"]) for record in records})
    result = {
        "status": "complete_research_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "atlas_dir": str(atlas_dir),
        "holdout_accessed": False,
        "selection_protocol": "early_aperture_then_entry_then_horizon; middle_latest_confirmation_and_neighbour_stability",
        "apertures": list(APERTURES),
        "entry_variants": list(ENTRY_VARIANTS),
        "horizons": list(HORIZONS),
        "score_component_count": 7,
        "hypothesis_coverage": atlas_summary.get("hypothesis_coverage", {}),
        "families": {
            family: _audit_family(
                [record for record in records if record["family"] == family],
                args.bootstrap_simulations,
            )
            for family in families
        },
    }
    _write_json(output_dir / "walk_forward_summary.json", result)
    (output_dir / "report.md").write_text(_render_report(result), encoding="utf-8")
    _write_json(output_dir / "queue_status.json", {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    survivors = sum(row["route_ready_for_portfolio_replay"] for row in result["families"].values())
    print(f"walk-forward audit complete: {survivors} route(s) ready for portfolio replay", flush=True)


if __name__ == "__main__":
    main()
