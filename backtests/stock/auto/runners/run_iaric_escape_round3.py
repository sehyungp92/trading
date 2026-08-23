"""Find a causal IARIC Round 3 that escapes the incumbent local maximum.

The incumbent remains an anchor sleeve.  This runner searches executable,
route-neutral reversion satellites in the only safe order: family isolation,
aperture/composition, discrimination and entry geometry, management, then
chronological verification.  The sealed holdout is excluded by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.stock.auto.iaric.worker import evaluate_candidate_attribution
from backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery import (
    HOLDOUT_START,
    _evaluate_batch,
    _replay_source_fingerprint,
    _signature,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
IARIC_DIR = REPO_ROOT / "backtests/output/stock/iaric"
DEFAULT_BASELINE = IARIC_DIR / "round_1/optimized_config.json"
DEFAULT_OUTPUT = IARIC_DIR / "round_3/escape_round"
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"
FOLDS = (
    ("early", "2024-03-25", "2024-11-30"),
    ("middle", "2024-12-01", "2025-07-31"),
    ("latest", "2025-08-01", END_DATE),
)
FAMILIES = (
    "GAP_EXHAUSTION_RECLAIM",
    "GAP_FILL_RECLAIM",
    "OPENING_FLUSH_RECLAIM",
    "OPENING_RANGE_LOW_RECLAIM",
    "PRIOR_DAY_LOW_RECLAIM",
    "VWAP_DEVIATION_RECLAIM",
    "FAILED_BREAKDOWN_RECLAIM",
    "MARKET_SECTOR_RESIDUAL_RECLAIM",
    "MULTIDAY_HIGHER_LOW_RECLAIM",
    "UPTREND_PULLBACK_RECLAIM",
    "VOLUME_CLIMAX_RECLAIM",
)

# Exactly seven immutable, baseline-relative components.  The scales are fixed
# economic materiality units, not estimates learned from this small sample.
SCORE_SPEC = {
    "incremental_total_r": {"weight": 0.30, "scale": 12.0},
    "incremental_trades": {"weight": 0.22, "scale": 40.0},
    "marginal_avg_r": {"weight": 0.12, "scale": 0.20},
    "profit_factor": {"weight": 0.07, "center": 1.0, "scale": 0.50},
    "discrimination_lift": {"weight": 0.05, "center": 0.0, "scale": 0.15},
    "inverse_drawdown": {"weight": 0.12, "center": 0.05, "scale": 0.03},
    "robust_avg_r": {"weight": 0.12, "center": 0.0, "scale": 0.15},
}

# Fields produced by the replay evaluator are authoritative and must never be
# overwritten by a prior full-period finalist when that finalist is replayed
# on a chronological fold. Candidate dictionaries may carry these fields after
# an earlier phase, so metadata reattachment must explicitly exclude them.
EVALUATION_OWNED_FIELDS = frozenset({
    "all_gates_pass",
    "aperture",
    "economic_score",
    "end_date",
    "error",
    "escape_score",
    "escape_score_components",
    "escape_score_raw",
    "folds",
    "funnel_counters",
    "gates",
    "metrics",
    "mutations",
    "score_components",
    "signature",
    "start_date",
    "trade_attribution",
    "validation_contract",
})


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _code_fingerprint() -> str:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "strategies/stock/iaric/core/opportunity.py",
        REPO_ROOT / "strategies/stock/iaric/core/logic.py",
        REPO_ROOT / "strategies/stock/iaric/research.py",
        REPO_ROOT / "strategies/stock/iaric/models.py",
        REPO_ROOT / "strategies/stock/iaric/config.py",
        REPO_ROOT / "strategies/stock/iaric/risk.py",
        REPO_ROOT / "strategies/stock/iaric/entry_request.py",
        REPO_ROOT / "strategies/stock/iaric/exits.py",
        REPO_ROOT / "backtests/stock/engine/iaric_pullback_intraday_hybrid_engine.py",
        REPO_ROOT / "backtests/stock/auto/iaric/worker.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _candidate(candidate_id: str, mutations: dict[str, Any], **meta: Any) -> dict[str, Any]:
    return {"id": candidate_id, "mutations": dict(sorted(mutations.items())), **meta}


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return orchestration metadata without any replay-owned result fields."""

    return {
        key: deepcopy(value)
        for key, value in candidate.items()
        if key not in EVALUATION_OWNED_FIELDS and key not in {"id", "mutations"}
    }


def _replay_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize a result-bearing finalist back into an immutable candidate."""

    return _candidate(
        str(candidate["id"]),
        candidate["mutations"],
        **_candidate_metadata(candidate),
    )


def _dedupe(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(_signature(candidate["mutations"]), candidate)
    return list(unique.values())


def _route_stats(attribution: list[dict[str, Any]]) -> dict[str, Any]:
    routes: dict[str, list[float]] = {}
    symbols: dict[str, float] = {}
    for trade in attribution:
        route = str(trade.get("route", ""))
        if not route.startswith("APERTURE_"):
            continue
        value = float(trade.get("r", 0.0))
        routes.setdefault(route, []).append(value)
        symbols[str(trade.get("symbol", ""))] = symbols.get(str(trade.get("symbol", "")), 0.0) + value
    flat = [value for values in routes.values() for value in values]
    wins = sum(value for value in flat if value > 0)
    losses = abs(sum(value for value in flat if value < 0))
    positive_symbol_r = sum(max(value, 0.0) for value in symbols.values())
    return {
        "trades": len(flat),
        "total_r": sum(flat),
        "avg_r": sum(flat) / len(flat) if flat else 0.0,
        "profit_factor": wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0),
        "max_positive_symbol_share": (
            max((max(value, 0.0) for value in symbols.values()), default=0.0) / positive_symbol_r
            if positive_symbol_r > 0 else 0.0
        ),
        "routes": {
            route: {
                "trades": len(values),
                "total_r": sum(values),
                "avg_r": sum(values) / len(values),
            }
            for route, values in sorted(routes.items())
        },
    }


def _anchor_trade_keys(
    attribution: list[dict[str, Any]],
    *,
    before: str | None = None,
) -> list[tuple[Any, ...]]:
    """Canonical incumbent trades for an anchor-isolation comparison."""
    keys = []
    for trade in attribution:
        route = str(trade.get("route", ""))
        entry_time = str(trade.get("entry_time", ""))
        if route.startswith("APERTURE_") or (before is not None and entry_time >= before):
            continue
        keys.append((
            entry_time,
            str(trade.get("symbol", "")),
            route,
            round(float(trade.get("entry_price", 0.0)), 6),
            round(float(trade.get("daily_signal_rank_pct", 100.0)), 6),
        ))
    return sorted(keys)


def _verify_anchor_isolation(rows: list[dict[str, Any]], output: Path) -> None:
    """Prove satellites do not alter the anchor before capital interaction."""
    control = next(row for row in rows if row["id"] == "incumbent_control")
    control_trades = list(control.get("trade_attribution", []))
    report: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in rows:
        if row["id"] == "incumbent_control":
            continue
        satellite_times = sorted(
            str(trade.get("entry_time", ""))
            for trade in row.get("trade_attribution", [])
            if str(trade.get("route", "")).startswith("APERTURE_")
        )
        first_satellite = satellite_times[0] if satellite_times else None
        expected = _anchor_trade_keys(control_trades, before=first_satellite)
        actual = _anchor_trade_keys(
            list(row.get("trade_attribution", [])),
            before=first_satellite,
        )
        passed = actual == expected
        report.append({
            "id": row["id"],
            "passed": passed,
            "first_satellite_entry": first_satellite,
            "control_anchor_trades_compared": len(expected),
            "candidate_anchor_trades_compared": len(actual),
        })
        if not passed:
            failures.append(str(row["id"]))
    _write_json(output / "anchor_isolation_verification.json", report)
    if failures:
        raise RuntimeError(
            "Aperture changed incumbent entries before any shared-capital interaction: "
            + ", ".join(failures)
        )


def _score(metrics: dict[str, Any], control: dict[str, Any]) -> tuple[float, dict[str, float], dict[str, float]]:
    delta_r = float(metrics.get("expected_total_r", 0.0)) - float(control.get("expected_total_r", 0.0))
    delta_trades = float(metrics.get("total_trades", 0.0)) - float(control.get("total_trades", 0.0))
    marginal = delta_r / max(delta_trades, 1.0) if delta_trades > 0 else delta_r
    raw = {
        "incremental_total_r": delta_r,
        "incremental_trades": delta_trades,
        "marginal_avg_r": marginal,
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "discrimination_lift": float(metrics.get("entry_realized_discrimination_lift_r", 0.0)),
        "inverse_drawdown": float(metrics.get("max_drawdown_pct", 1.0)),
        "robust_avg_r": float(metrics.get("robust_avg_r", 0.0)),
    }
    components: dict[str, float] = {}
    for name, spec in SCORE_SPEC.items():
        value = raw[name]
        if name == "inverse_drawdown":
            z_value = (float(spec["center"]) - value) / float(spec["scale"])
        elif "center" in spec:
            z_value = (value - float(spec["center"])) / float(spec["scale"])
        else:
            z_value = value / float(spec["scale"])
        components[name] = 0.5 + 0.5 * math.tanh(z_value)
    score = sum(float(SCORE_SPEC[name]["weight"]) * components[name] for name in SCORE_SPEC)
    return float(score), components, raw


def _evaluate(
    stage: str,
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output: Path,
    source_fingerprint: str,
    code_fingerprint: str,
    control_metrics: dict[str, Any] | None,
    evaluation_fn: Any = evaluate_candidate_attribution,
    cache_filename: str = "evaluation_cache.json",
    score_metrics_fn: Any | None = None,
) -> list[dict[str, Any]]:
    deduped_candidates = _dedupe(candidates)
    candidate_by_signature = {
        _signature(candidate["mutations"]): candidate
        for candidate in deduped_candidates
    }
    rows = _evaluate_batch(
        deduped_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=min(max(int(args.max_workers), 1), 2),
        cache_path=output / cache_filename,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        evaluation_fn=evaluation_fn,
        require_bundle=False,
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        _write_json(output / f"{stage}_errors.json", errors)
        raise RuntimeError(f"{len(errors)} {stage} evaluations failed")
    control = control_metrics or next(row["metrics"] for row in rows if row["id"] == "incumbent_control")
    for row in rows:
        # The shared cache layer intentionally retains only generic candidate
        # fields.  Escape-round orchestration also needs structural metadata
        # (especially ``families``) after both fresh and cached evaluations.
        # Reattach it by immutable mutation signature before constructing the
        # next phase; otherwise a valid isolation becomes an empty-family
        # composition that silently evaluates as the incumbent control.
        candidate = candidate_by_signature.get(str(row.get("signature", "")))
        if candidate is None:
            raise RuntimeError(
                f"Evaluation has no matching candidate metadata: {row.get('signature', '')}"
            )
        # Reattach only orchestration metadata. Replay-owned dates, metrics,
        # attribution and diagnostics remain authoritative for this exact run.
        row["id"] = str(candidate["id"])
        for key, value in _candidate_metadata(candidate).items():
            row[key] = value
        row["aperture"] = _route_stats(row.get("trade_attribution", []))
        score_metrics = score_metrics_fn(row) if score_metrics_fn is not None else row["metrics"]
        score, components, raw = _score(score_metrics, control)
        row["escape_score"] = score
        row["escape_score_components"] = components
        row["escape_score_raw"] = raw
    rows.sort(
        key=lambda row: (
            float(row["escape_score"]),
            float(row["metrics"].get("expected_total_r", -1e9)),
            float(row["metrics"].get("total_trades", 0.0)),
            -float(row["metrics"].get("max_drawdown_pct", 1.0)),
        ),
        reverse=True,
    )
    _write_json(output / f"{stage}_results.json", rows)
    _write_json(output / "progress.json", {
        "status": "running",
        "last_completed_stage": stage,
        "evaluated": len(rows),
        "best_id": rows[0]["id"] if rows else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return rows


def _common_aperture(baseline: dict[str, Any], families: Iterable[str]) -> dict[str, Any]:
    result = deepcopy(baseline)
    result.update({
        "param_overrides.pb_aperture_enabled": True,
        "param_overrides.pb_aperture_max_symbols": 120,
        "param_overrides.pb_aperture_families": ",".join(sorted(families)),
        "param_overrides.pb_aperture_event_score_min": 70.0,
        "param_overrides.pb_aperture_sizing_mult": 0.55,
        "param_overrides.pb_aperture_prior_low_transition": "retrace",
        "param_overrides.pb_aperture_multiday_transition": "confirm",
    })
    changed = {
        key for key in result
        if result.get(key) != baseline.get(key)
    }
    invalid = sorted(
        key for key in changed
        if not key.startswith("param_overrides.pb_aperture_")
    )
    if invalid:
        raise RuntimeError(
            "Satellite construction mutated incumbent configuration: "
            + ", ".join(invalid)
        )
    return result


def _phase0(baseline: dict[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    families = ("PRIOR_DAY_LOW_RECLAIM", "MULTIDAY_HIGHER_LOW_RECLAIM") if smoke else FAMILIES
    rows = [_candidate("incumbent_control", baseline, stage="control", families=[])]
    for family in families:
        rows.append(_candidate(
            f"isolate__{family.lower()}",
            _common_aperture(baseline, [family]),
            stage="route_isolation",
            families=[family],
        ))
    return rows


def _viable_isolations(rows: list[dict[str, Any]], control: dict[str, Any]) -> list[dict[str, Any]]:
    base_r = float(control["metrics"].get("expected_total_r", 0.0))
    base_dd = float(control["metrics"].get("max_drawdown_pct", 1.0))
    viable = [
        row for row in rows
        if row["id"] != "incumbent_control"
        and int(row["aperture"]["trades"]) >= 3
        and float(row["aperture"]["total_r"]) > 0.0
        and float(row["aperture"]["profit_factor"]) >= 1.05
        and float(row["metrics"].get("expected_total_r", -99.0)) >= base_r - 5.0
        # Isolation is baseline-relative.  An absolute cap below the measured
        # incumbent DD can reject every value-creating satellite before the
        # management phase has a chance to reduce its risk.
        and float(row["metrics"].get("max_drawdown_pct", 1.0))
        <= base_dd + 0.010
    ]
    return sorted(viable, key=lambda row: (float(row["aperture"]["total_r"]), row["escape_score"]), reverse=True)[:4]


def _composition_family_sets(isolations: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    family_sets = {tuple(row.get("families", [])) for row in isolations}
    singles = sorted({family for values in family_sets for family in values})
    for size in (2, 3):
        for combo in itertools.combinations(singles, size):
            family_sets.add(tuple(sorted(combo)))
    return sorted(values for values in family_sets if values)


def _composition_center_candidates(
    baseline: dict[str, Any],
    isolations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Screen structural family sets once at the neutral 120-name aperture."""

    candidates: list[dict[str, Any]] = []
    for families in _composition_family_sets(isolations):
        mutations = _common_aperture(baseline, families)
        mutations["param_overrides.pb_aperture_max_symbols"] = 120
        candidates.append(_candidate(
            f"compose__{'_'.join(value.lower() for value in families)}__n120",
            mutations,
            stage="composition_center",
            families=list(families),
        ))
    return candidates


def _aperture_expansion_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand aperture only around structurally viable family sets."""

    candidates: list[dict[str, Any]] = []
    for parent in parents:
        for aperture in (60, 120, 180):
            mutations = dict(parent["mutations"])
            mutations["param_overrides.pb_aperture_max_symbols"] = aperture
            candidates.append(_candidate(
                f"{parent['id'].rsplit('__n', 1)[0]}__n{aperture}",
                mutations,
                stage="aperture_expansion",
                families=parent.get("families", []),
            ))
    return candidates


def _family_transition_mutation(
    base: dict[str, Any],
    family: str,
    transition: str,
) -> dict[str, Any]:
    raw = str(base.get("param_overrides.pb_aperture_family_transitions", "") or "")
    mappings: dict[str, str] = {}
    for token in raw.split(","):
        if not token.strip():
            continue
        separator = ":" if ":" in token else "="
        key, value = token.split(separator, 1)
        mappings[key.strip().upper()] = value.strip().lower()
    mappings[str(family).upper()] = str(transition).lower()
    mutations = dict(base)
    mutations["param_overrides.pb_aperture_family_transitions"] = ",".join(
        f"{key}:{value}" for key, value in sorted(mappings.items())
    )
    return mutations


def _quality_entry_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Test only mechanisms that can affect the families in each parent."""

    candidates: list[dict[str, Any]] = []
    for parent in parents:
        base = parent["mutations"]
        families = sorted(set(parent.get("families", [])))
        candidates.append(_candidate(
            parent["id"] + "__quality_control",
            base,
            stage="quality_entry",
            families=families,
        ))
        for name, delta in (
            ("floor65", {"param_overrides.pb_aperture_event_score_min": 65.0}),
            ("floor75", {"param_overrides.pb_aperture_event_score_min": 75.0}),
        ):
            mutations = dict(base)
            mutations.update(delta)
            candidates.append(_candidate(
                parent["id"] + "__" + name,
                mutations,
                stage="quality_entry",
                families=families,
            ))
        for family in families:
            default_transition = (
                "retrace" if family == "PRIOR_DAY_LOW_RECLAIM"
                else "confirm" if family == "MULTIDAY_HIGHER_LOW_RECLAIM"
                else "next_bar"
            )
            for transition in ("next_bar", "confirm", "retrace"):
                if transition == default_transition:
                    continue
                candidates.append(_candidate(
                    parent["id"]
                    + "__"
                    + family.lower()
                    + "_"
                    + transition,
                    _family_transition_mutation(base, family, transition),
                    stage="quality_entry",
                    families=families,
                ))
    return candidates


def _management_candidates(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    changes = (
        ("management_control", {}),
        ("size40", {"param_overrides.pb_aperture_sizing_mult": 0.40}),
        ("size70", {"param_overrides.pb_aperture_sizing_mult": 0.70}),
        ("no_carry_floor", {"param_overrides.pb_aperture_carry_min_r": 0.0}),
        ("carry_floor40", {"param_overrides.pb_aperture_carry_min_r": 0.40}),
        ("hold2", {"param_overrides.pb_aperture_max_hold_days": 2}),
        ("hold5", {"param_overrides.pb_aperture_max_hold_days": 5}),
        ("stale4", {"param_overrides.pb_aperture_stale_exit_bars": 4}),
        ("stale_off", {"param_overrides.pb_aperture_stale_exit_bars": 0}),
    )
    for parent in parents:
        for name, delta in changes:
            mutations = dict(parent["mutations"])
            mutations.update(delta)
            candidates.append(_candidate(
                parent["id"] + "__" + name,
                mutations,
                stage="management",
                families=parent.get("families", []),
            ))
    return candidates


def _shortlist(
    rows: list[dict[str, Any]],
    limit: int,
    control: dict[str, Any],
) -> list[dict[str, Any]]:
    base_dd = float(control["metrics"].get("max_drawdown_pct", 1.0))
    viable = [
        row for row in rows
        if float(row["metrics"].get("avg_r", -99.0)) >= 0.10
        and float(row["metrics"].get("profit_factor", 0.0)) >= 1.25
        and float(row["metrics"].get("max_drawdown_pct", 1.0))
        <= base_dd + 0.015
        and int(row["aperture"]["trades"]) >= 3
        and float(row["aperture"]["total_r"]) > 0.0
    ]
    return sorted(viable, key=lambda row: row["escape_score"], reverse=True)[:limit]


def _diverse_structure_shortlist(
    rows: list[dict[str, Any]],
    limit: int,
    control: dict[str, Any],
) -> list[dict[str, Any]]:
    ranked = _shortlist(rows, len(rows), control)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in ranked:
        key = tuple(sorted(row.get("families", [])))
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _fold_validate(
    finalists: list[dict[str, Any]],
    control: dict[str, Any],
    *,
    args: argparse.Namespace,
    output: Path,
    source_fingerprint: str,
    code_fingerprint: str,
) -> None:
    candidates = [
        _candidate("incumbent_control", control["mutations"], stage="control", families=[])
    ] + [_replay_candidate(finalist) for finalist in finalists]
    expected_signatures = {_signature(candidate["mutations"]) for candidate in candidates}
    if len(expected_signatures) != len(candidates):
        raise RuntimeError("Chronological validation candidates are not signature-unique")
    for fold_name, start, end in FOLDS:
        fold_args = argparse.Namespace(**vars(args))
        fold_args.start_date = start
        fold_args.end_date = end
        rows = _evaluate(
            f"validation_{fold_name}",
            candidates,
            args=fold_args,
            output=output,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
            control_metrics=None,
        )
        actual_signatures = {_signature(row["mutations"]) for row in rows}
        if actual_signatures != expected_signatures or len(rows) != len(candidates):
            raise RuntimeError(
                f"{fold_name} fold returned a different immutable candidate set"
            )
        for row in rows:
            if row.get("start_date") != start or row.get("end_date") != end:
                raise RuntimeError(
                    f"{fold_name} fold date contract violated for {row.get('id')}: "
                    f"{row.get('start_date')}..{row.get('end_date')} != {start}..{end}"
                )
        by_sig = {_signature(row["mutations"]): row for row in rows}
        fold_control = next(row for row in rows if row["id"] == "incumbent_control")
        for finalist in finalists:
            match = by_sig[_signature(finalist["mutations"])]
            finalist.setdefault("folds", []).append({
                "fold": fold_name,
                "start_date": start,
                "end_date": end,
                "metrics": match["metrics"],
                "aperture": match["aperture"],
                "delta_total_r": float(match["metrics"].get("expected_total_r", 0.0)) - float(fold_control["metrics"].get("expected_total_r", 0.0)),
                "delta_trades": float(match["metrics"].get("total_trades", 0.0)) - float(fold_control["metrics"].get("total_trades", 0.0)),
            })
    expected_fold_contract = [
        {"fold": fold, "start_date": start, "end_date": end}
        for fold, start, end in FOLDS
    ]
    for finalist in finalists:
        actual = [
            {
                "fold": fold.get("fold"),
                "start_date": fold.get("start_date"),
                "end_date": fold.get("end_date"),
            }
            for fold in finalist.get("folds", [])
        ]
        passed = actual == expected_fold_contract
        finalist["validation_contract"] = {
            "passed": passed,
            "folds": actual,
            "holdout_accessed": False,
        }
        if not passed:
            raise RuntimeError(f"Incomplete chronological validation for {finalist['id']}")


def _gates(row: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    metrics, base = row["metrics"], control["metrics"]
    folds = row.get("folds", [])
    delta_r = float(metrics.get("expected_total_r", 0.0)) - float(base.get("expected_total_r", 0.0))
    delta_n = float(metrics.get("total_trades", 0.0)) - float(base.get("total_trades", 0.0))
    return {
        "fold_integrity": bool(row.get("validation_contract", {}).get("passed")),
        "frequency_uplift_15pct": float(metrics.get("total_trades", 0.0)) >= 1.15 * float(base.get("total_trades", 0.0)),
        "total_r_created": delta_r >= 2.0,
        "positive_marginal_expectancy": delta_n > 0 and delta_r / delta_n > 0.0,
        "portfolio_avg_r": float(metrics.get("avg_r", 0.0)) >= 0.12,
        "portfolio_pf": float(metrics.get("profit_factor", 0.0)) >= 1.35,
        "bounded_drawdown": float(metrics.get("max_drawdown_pct", 1.0)) <= max(float(base.get("max_drawdown_pct", 0.0)) + 0.015, 0.045),
        "satellite_positive": int(row["aperture"]["trades"]) >= 10 and float(row["aperture"]["total_r"]) > 0.0 and float(row["aperture"]["profit_factor"]) >= 1.10,
        "satellite_not_single_symbol": float(row["aperture"].get("max_positive_symbol_share", 1.0)) <= 0.35,
        "chronological_consistency": sum(float(fold["delta_total_r"]) > 0.0 for fold in folds) >= 2 and min((float(fold["delta_total_r"]) for fold in folds), default=-99.0) >= -3.0,
    }


def _diagnostics(selected: dict[str, Any], control: dict[str, Any], status: str) -> str:
    sm, cm = selected["metrics"], control["metrics"]
    gates = selected["gates"]
    lines = [
        "IARIC ROUND 3 — LOCAL-MAXIMUM ESCAPE FINAL DIAGNOSTICS",
        "=" * 72,
        f"Status: {status}",
        f"Selected: {selected['id']}",
        f"Training authority: {START_DATE} through {END_DATE}; holdout from {HOLDOUT_START} excluded",
        "",
        "OUTCOME",
        f"  Trades: {cm.get('total_trades', 0):.0f} -> {sm.get('total_trades', 0):.0f}",
        f"  Expected total R: {cm.get('expected_total_r', 0):+.3f} -> {sm.get('expected_total_r', 0):+.3f}",
        f"  Avg R: {cm.get('avg_r', 0):+.4f} -> {sm.get('avg_r', 0):+.4f}",
        f"  PF: {cm.get('profit_factor', 0):.3f} -> {sm.get('profit_factor', 0):.3f}",
        f"  Max DD: {cm.get('max_drawdown_pct', 0):.3%} -> {sm.get('max_drawdown_pct', 0):.3%}",
        f"  Aperture sleeve: n={selected['aperture']['trades']}, totalR={selected['aperture']['total_r']:+.3f}, avgR={selected['aperture']['avg_r']:+.4f}, PF={selected['aperture']['profit_factor']:.3f}",
        "",
        "VALUE / REAL-ALPHA GATES",
    ]
    lines.extend(f"  [{'PASS' if passed else 'FAIL'}] {name}" for name, passed in gates.items())
    lines += ["", "CHRONOLOGICAL FOLDS"]
    for fold in selected.get("folds", []):
        lines.append(
            f"  {fold['fold']}: deltaR={fold['delta_total_r']:+.3f}, delta trades={fold['delta_trades']:+.0f}, satelliteR={fold['aperture']['total_r']:+.3f}"
        )
    lines += [
        "",
        "STRUCTURAL INTERPRETATION",
        "  Round 3 is an anchor-plus-satellite platform. The incumbent sleeve is retained;",
        "  route-neutral nightly aperture, discrete completed-bar family events, route-specific",
        "  transitions, shared capital, and route management are independently optimizable.",
        "  Search allocation is adaptive: family sets are screened at aperture 120, only robust",
        "  structures receive 60/180 expansion, and entry mechanisms are generated only for",
        "  families actually present. This removes behaviourally inert full replays.",
        "  This escapes the former threshold basin without claiming that unconditional stock",
        "  reversion is positive; the event discriminator remains mandatory.",
        "",
        "IMMUTABLE SCORE (7 COMPONENTS)",
    ]
    lines.extend(
        f"  {name}: weight={spec['weight']:.2f}, scale={spec['scale']:.4g}"
        for name, spec in SCORE_SPEC.items()
    )
    lines += ["", "SELECTED MUTATIONS", json.dumps(selected["mutations"], indent=2, sort_keys=True)]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    if len(SCORE_SPEC) != 7:
        raise RuntimeError("IARIC escape score must contain exactly seven components")
    if args.end_date >= HOLDOUT_START:
        raise ValueError(f"end-date must precede sealed holdout {HOLDOUT_START}")
    if args.max_workers > 2:
        raise ValueError("IARIC escape round is capped at max-workers=2")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(args.baseline_config).read_text(encoding="utf-8"))
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()
    _write_json(output / "run_spec.json", {
        "objective": "escape the incumbent local maximum with causal route-neutral reversion satellites",
        "baseline": str(Path(args.baseline_config).resolve()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "holdout_start": HOLDOUT_START,
        "holdout_accessed": False,
        "max_workers": args.max_workers,
        "families": list(FAMILIES),
        "score_spec": SCORE_SPEC,
        "adaptive_search": {
            "composition_center_aperture": 120,
            "composition_center_limit": 3,
            "aperture_expansion": [60, 120, 180],
            "aperture_diverse_limit": 3,
            "entry_parent_limit": 3,
            "management_parent_limit": 3,
            "validation_finalist_limit": 4,
            "family_aware_entry_mechanisms": ["next_bar", "confirm", "retrace"],
        },
        "source_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
    })

    phase0 = _evaluate(
        "phase_0_route_isolation",
        _phase0(baseline, smoke=args.smoke),
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=None,
    )
    control = next(row for row in phase0 if row["id"] == "incumbent_control")
    _verify_anchor_isolation(phase0, output)
    if args.smoke:
        _write_json(output / "smoke_summary.json", phase0)
        return 0
    isolations = _viable_isolations(phase0, control)
    _write_json(output / "phase_0_survivors.json", isolations)
    if not isolations:
        raise RuntimeError("No isolated aperture route produced positive executable satellite alpha")

    phase1_center = _evaluate(
        "phase_1a_composition_center",
        _composition_center_candidates(baseline, isolations),
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=control["metrics"],
    )
    phase1_center_short = _diverse_structure_shortlist(
        phase1_center,
        3,
        control,
    )
    if not phase1_center_short:
        raise RuntimeError("No centered composition retained positive alpha and bounded quality")
    phase1 = _evaluate(
        "phase_1b_aperture_expansion",
        _aperture_expansion_candidates(phase1_center_short),
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=control["metrics"],
    )
    phase1_short = _diverse_structure_shortlist(phase1, 3, control)
    if not phase1_short:
        raise RuntimeError("No aperture expansion retained positive alpha and bounded quality")
    phase2 = _evaluate(
        "phase_2_discrimination_entry",
        _quality_entry_candidates(phase1_short),
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=control["metrics"],
    )
    phase2_short = _shortlist(phase2, 3, control)
    if not phase2_short:
        # Every entry family includes its unchanged control, so reaching this
        # branch means the stage violated its own parent-preservation contract.
        raise RuntimeError("Entry refinement lost every viable parent control")
    phase3 = _evaluate(
        "phase_3_management",
        _management_candidates(phase2_short),
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
        control_metrics=control["metrics"],
    )
    finalists = _shortlist(phase3, 4, control)
    if not finalists:
        raise RuntimeError("No management finalist retained positive satellite alpha")
    _fold_validate(
        finalists,
        control,
        args=args,
        output=output,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    for row in finalists:
        row["gates"] = _gates(row, control)
        row["all_gates_pass"] = all(row["gates"].values())
    finalists.sort(key=lambda row: (row["all_gates_pass"], row["escape_score"]), reverse=True)
    selected = finalists[0]
    status = "complete_value_verified" if selected["all_gates_pass"] else "blocked_value_verification"
    _write_json(output / "validated_finalists.json", finalists)
    _write_json(output / "final_selection.json", {"status": status, "selected": selected, "control": control})
    (output / "round_final_diagnostics.txt").write_text(_diagnostics(selected, control, status), encoding="utf-8")

    # A blocked research result is preserved but never made canonical.
    if selected["all_gates_pass"]:
        round3 = IARIC_DIR / "round_3"
        _write_json(round3 / "optimized_config.json", selected["mutations"])
        _write_json(round3 / "run_summary.json", {
            "status": status,
            "selected_id": selected["id"],
            "metrics": selected["metrics"],
            "aperture": selected["aperture"],
            "gates": selected["gates"],
            "holdout_accessed": False,
            "escape_round": "round_3/escape_round/final_selection.json",
        })
        (round3 / "round_final_diagnostics.txt").write_text(_diagnostics(selected, control, status), encoding="utf-8")
        manifest_path = IARIC_DIR / "rounds_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["active_round"] = 3
        manifest.pop("pending_round_3", None)
        manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest.setdefault("rounds", []).append({
            "round": 3,
            "status": status,
            "configuration_role": "local_maximum_escape_anchor_plus_satellites",
            "mutations": selected["mutations"],
            "metrics": selected["metrics"],
            "aperture": selected["aperture"],
            "score_component_count": 7,
            "sealed_holdout": {"start": HOLDOUT_START, "used": False},
            "artifacts": {
                "optimized_config": "round_3/optimized_config.json",
                "full_final_diagnostics": "round_3/round_final_diagnostics.txt",
                "selection": "round_3/escape_round/final_selection.json",
            },
        })
        _write_json(manifest_path, manifest)
    _write_json(output / "progress.json", {
        "status": status,
        "selected_id": selected["id"],
        "all_gates_pass": selected["all_gates_pass"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return 0 if selected["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
