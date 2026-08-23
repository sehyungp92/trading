"""Gate and stage the unified IARIC Round-3 result after both research streams.

The process waits for the branched-aperture run and the atlas walk-forward run,
verifies matching pre-holdout data authority, and applies a fixed seven-part
portfolio score to the executable incumbent.  Atlas survivors are never mapped
onto merely similar legacy routes: any survivor without both adapters produces
an exact typed implementation/replay catalog and blocks false promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BRANCHED = REPO_ROOT / "backtests/output/stock/iaric/round_3"
DEFAULT_ATLAS = REPO_ROOT / "backtests/output/stock/opportunity_atlas/round_1"
DEFAULT_OUTPUT = DEFAULT_BRANCHED / "unified"
HOLDOUT_START = "2026-03-02"

# Economic anchors are pre-registered before the full atlas result is known.
# Net profit is deliberately absent because it duplicates total R and embeds a
# capital-size assumption.
UNIFIED_SCORE_SPEC: dict[str, dict[str, float]] = {
    "expected_total_r": {"weight": 0.30, "center": 30.0, "scale": 15.0},
    "total_trades": {"weight": 0.18, "center": 102.0, "scale": 50.0},
    "inverse_mtm_drawdown": {"weight": 0.17, "center": 0.045, "scale": 0.025},
    "avg_r": {"weight": 0.12, "center": 0.12, "scale": 0.12},
    "worst_fold_avg_r": {"weight": 0.10, "center": 0.0, "scale": 0.10},
    "profit_factor": {"weight": 0.08, "center": 1.45, "scale": 0.35},
    "entry_realized_discrimination_lift_r": {"weight": 0.05, "center": 0.0, "scale": 0.12},
}

# Detection is shared, causal core code.  Portfolio/live adapter support stays
# false until the route is expressed as neutral actions and replayed through
# the common execution path.  This registry prevents approximate mappings.
ROUTE_READINESS: dict[str, dict[str, Any]] = {
    "GAP_EXHAUSTION_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "GAP_PARTIAL_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "VWAP_DEVIATION_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "FAILED_BREAKDOWN_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "MARKET_SECTOR_RESIDUAL_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["authoritative_live_sector_residual_feed", "typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "MULTIDAY_HIGHER_LOW_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_multi_session_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter", "snapshot_hydration_parity"],
    },
    "VOLUME_CLIMAX_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "GAP_FILL_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "OPENING_FLUSH_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "OPENING_RANGE_LOW_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "PRIOR_DAY_LOW_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_route_state", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter"],
    },
    "UPTREND_PULLBACK_RECLAIM": {
        "detector": "core.opportunity.detect_completed_bar_opportunities",
        "missing": ["typed_daily_trend_context", "neutral_entry_action", "live_adapter", "portfolio_replay_adapter", "snapshot_hydration_parity"],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branched-dir", default=str(DEFAULT_BRANCHED))
    parser.add_argument("--atlas-dir", default=str(DEFAULT_ATLAS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--wait-for-pid", action="append", type=int, default=[])
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "strategies/stock/iaric/core/opportunity.py",
        REPO_ROOT / "backtests/stock/auto/runners/run_iaric_branched_aperture.py",
        REPO_ROOT / "backtests/stock/auto/runners/analyze_stock_opportunity_atlas.py",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _score(selected: dict[str, Any]) -> tuple[float, dict[str, float], dict[str, float]]:
    metrics = selected.get("metrics", {})
    folds = selected.get("validation", {}).get("folds", [])
    raw = {
        "expected_total_r": float(metrics.get("expected_total_r", 0.0)),
        "total_trades": float(metrics.get("total_trades", 0.0)),
        "inverse_mtm_drawdown": float(metrics.get("max_drawdown_pct", 1.0)),
        "avg_r": float(metrics.get("avg_r", 0.0)),
        "worst_fold_avg_r": min((float(fold.get("avg_r", 0.0)) for fold in folds), default=0.0),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "entry_realized_discrimination_lift_r": float(
            metrics.get("entry_realized_discrimination_lift_r", 0.0)
        ),
    }
    components: dict[str, float] = {}
    for name, spec in UNIFIED_SCORE_SPEC.items():
        if name == "inverse_mtm_drawdown":
            z_value = (spec["center"] - raw[name]) / spec["scale"]
        else:
            z_value = (raw[name] - spec["center"]) / spec["scale"]
        components[name] = min(max(0.5 + 0.5 * math.tanh(z_value), 0.0), 1.0)
    score = sum(UNIFIED_SCORE_SPEC[name]["weight"] * components[name] for name in components)
    return float(score), components, raw


def _validate_inputs(
    branched_dir: Path, atlas_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    branched_progress = _load_json(branched_dir / "progress.json")
    if branched_progress.get("stage") != "complete" or branched_progress.get("status") == "running":
        raise RuntimeError("branched Round3 did not complete successfully")
    branched_selection = _load_json(branched_dir / "final_selection.json")
    branched_spec = _load_json(branched_dir / "run_spec.json")
    atlas_summary = _load_json(atlas_dir / "atlas_summary.json")
    atlas_spec = _load_json(atlas_dir / "run_spec.json")
    walk = _load_json(atlas_dir / "walk_forward/walk_forward_summary.json")
    if branched_selection.get("holdout_accessed") is not False:
        raise ValueError("branched result does not prove holdout exclusion")
    if atlas_summary.get("holdout_accessed") is not False or walk.get("holdout_accessed") is not False:
        raise ValueError("atlas result does not prove holdout exclusion")
    if int(branched_spec.get("score_component_count", 0)) > 7 or int(walk.get("score_component_count", 0)) > 7:
        raise ValueError("an upstream score exceeds seven components")
    branched_window = branched_spec.get("training_window", {})
    atlas_window = atlas_spec.get("window", {})
    if branched_window != atlas_window:
        raise ValueError(f"training-window mismatch: {branched_window} != {atlas_window}")
    if str(branched_window.get("end", "")) >= HOLDOUT_START:
        raise ValueError("unified inputs overlap the sealed holdout")
    if branched_spec.get("data_fingerprint") != atlas_spec.get("data_fingerprint"):
        raise ValueError("research streams used different source fingerprints")
    return branched_selection, branched_spec, atlas_summary, atlas_spec, walk


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# IARIC Unified Round 3",
        "",
        f"Status: {result['status']}",
        "Holdout accessed: no.",
        f"Branched incumbent: {result['incumbent']['id']}",
        f"Unified immutable score: {result['incumbent']['unified_score']:.4f}",
        f"Atlas walk-forward survivors: {len(result['atlas_survivors'])}",
        "",
    ]
    if result["atlas_survivors"]:
        lines.extend([
            "No atlas survivor was silently translated into a similar legacy route. The generated "
            "implementation catalog is the mandatory next gate: shared typed state, neutral actions, "
            "live/backtest adapters, shared-capital MTM replay, and parity tests.",
        ])
    else:
        lines.extend([
            "No atlas family survived the pre-registered walk-forward gate. The branched incumbent is "
            "therefore retained as the unified research result; no structural route was manufactured.",
        ])
    lines.extend([
        "",
        "This remains research-only because both inputs use the legacy cache. Production promotion "
        "requires unchanged frozen-data replay and the sealed holdout decision.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    branched_dir = Path(args.branched_dir).resolve()
    atlas_dir = Path(args.atlas_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    waiting_for = sorted(set(pid for pid in args.wait_for_pid if pid > 0))
    _write_json(output_dir / "queue_status.json", {
        "status": "queued" if waiting_for else "starting",
        "waiting_for_pids": waiting_for,
        "queued_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    for pid in waiting_for:
        _wait_for_pid(pid)
    _write_json(output_dir / "queue_status.json", {
        "status": "running",
        "waiting_for_pids": waiting_for,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    branched_selection, branched_spec, atlas_summary, atlas_spec, walk = _validate_inputs(
        branched_dir, atlas_dir,
    )
    selected = branched_selection["selected"]
    score, components, raw = _score(selected)
    all_walk_survivors = {
        family: row
        for family, row in walk.get("families", {}).items()
        if bool(row.get("route_ready_for_portfolio_replay"))
    }
    survivors = {
        family: row for family, row in all_walk_survivors.items()
        if family in ROUTE_READINESS
    }
    reference_survivors = sorted(set(all_walk_survivors) - set(survivors))
    implementation_catalog = {
        family: {
            "family": family,
            "selected_aperture": row.get("selected_aperture"),
            "selected_horizon": row.get("selected_horizon"),
            "folds": row.get("folds", {}),
            "readiness": ROUTE_READINESS.get(family, {
                "detector": None,
                "missing": ["registered_exact_route_contract"],
            }),
            "approximate_mapping_allowed": False,
            "next_experiment": (
                "isolated exact route -> incumbent plus route -> at most one low-overlap two-route combination"
            ),
        }
        for family, row in survivors.items()
    }
    status = (
        "typed_route_implementation_required"
        if survivors else "incumbent_retained_no_atlas_survivors"
    )
    result = {
        "status": status,
        "research_only": True,
        "holdout_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_fingerprint": _fingerprint(),
        "data_fingerprint": branched_spec["data_fingerprint"],
        "training_window": branched_spec["training_window"],
        "unified_score_spec": UNIFIED_SCORE_SPEC,
        "score_component_count": len(UNIFIED_SCORE_SPEC),
        "incumbent": {
            "id": selected["id"],
            "source_status": branched_selection["status"],
            "unified_score": score,
            "unified_score_components": components,
            "unified_score_raw": raw,
            "metrics": selected.get("metrics", {}),
            "validation": selected.get("validation", {}),
        },
        "atlas_survivors": sorted(survivors),
        "reference_survivors_not_iaric_routes": reference_survivors,
        "implementation_catalog": implementation_catalog,
        "source_fingerprints": {
            "branched_code": branched_spec.get("code_fingerprint"),
            "atlas_code": atlas_spec.get("code_fingerprint"),
        },
        "promotion_allowed": False,
        "promotion_blockers": (
            ["typed route and adapter implementation", "shared-capital MTM portfolio replay", "live/backtest parity"]
            if survivors else ["legacy data authority", "frozen-data replay", "sealed holdout decision"]
        ),
    }
    _write_json(output_dir / "unified_selection.json", result)
    _write_json(output_dir / "implementation_catalog.json", implementation_catalog)
    config_name = "implementation_gated_incumbent_config.json" if survivors else "provisional_config.json"
    _write_json(output_dir / config_name, dict(sorted(selected["mutations"].items())))
    (output_dir / "report.md").write_text(_render_report(result), encoding="utf-8")
    _write_json(output_dir / "queue_status.json", {
        "status": "complete",
        "result_status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(
        f"unified Round3 complete: {status}; incumbent={selected['id']}; "
        f"atlas_survivors={len(survivors)}; holdout accessed=no",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failed_args = _parse_args()
        failed_output = Path(failed_args.output_dir).resolve()
        _write_json(failed_output / "queue_status.json", {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        raise
