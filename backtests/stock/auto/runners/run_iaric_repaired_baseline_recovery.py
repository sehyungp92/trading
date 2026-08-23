"""Recover a trustworthy IARIC baseline after the partial-exit execution repair.

The runner deliberately precedes phased optimization.  It replays all distinct
archived optimized endpoints and canonical round baselines under the corrected
shared-core/backtest execution path, applies explicit partial-exit controls,
checks finalists on chronological folds, then performs a compact family-level
ablation.  The holdout is excluded by construction.

The result is provisional when ``--allow-legacy-data`` is used and must be
revalidated unchanged against a frozen authoritative replay bundle before it is
eligible for live or holdout use.

Usage::

    python -m backtests.stock.auto.runners.run_iaric_repaired_baseline_recovery \
        --max-workers 2 --allow-legacy-data
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtests.stock.auto.iaric.phase_candidates import (
    BASE_MUTATIONS,
    R5_BASE_MUTATIONS,
    V2R1_BASE_MUTATIONS,
    V2R2_BASE_MUTATIONS,
    V2R3_BASE_MUTATIONS,
    V2R4_BASE_MUTATIONS,
    V3R1_BASE_MUTATIONS,
    V4R1_BASE_MUTATIONS,
    V5R1_BASE_MUTATIONS,
)
from backtests.stock.auto.iaric.worker import evaluate_candidate_metrics, init_worker


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
IARIC_OUTPUT = REPO_ROOT / "backtests/output/stock/iaric"
DEFAULT_OUTPUT = IARIC_OUTPUT / "baseline_recovery"
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
HOLDOUT_START = "2026-03-02"
INITIAL_EQUITY = 10_000.0
MAX_WORKERS = 2

# Exactly seven immutable economic components.  Chronological evidence is an
# eligibility/stability overlay, never an eighth tunable score component.
SCORE_SPEC: dict[str, dict[str, float | str]] = {
    "net_profit": {"weight": 0.20, "transform": "tanh(x / 3000)"},
    "expected_total_r": {"weight": 0.18, "transform": "tanh(x / 75)"},
    "avg_r": {"weight": 0.16, "transform": "tanh(x / 0.10)"},
    "profit_factor": {"weight": 0.14, "transform": "tanh((x - 1) / 0.60)"},
    "sharpe": {"weight": 0.10, "transform": "tanh(x / 2)"},
    "inverse_drawdown": {"weight": 0.12, "transform": "tanh((0.18 - x) / 0.12)"},
    "total_trades": {"weight": 0.10, "transform": "tanh(x / 800)"},
}

FOLDS: tuple[tuple[str, str, str], ...] = (
    ("2024_h1", "2024-01-01", "2024-06-30"),
    ("2024_h2", "2024-07-01", "2024-12-31"),
    ("2025_h1", "2025-01-01", "2025-06-30"),
    ("2025_h2_to_2026_03", "2025-07-01", END_DATE),
)

CANONICAL_BASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("canonical_mainline", BASE_MUTATIONS),
    ("canonical_r5", R5_BASE_MUTATIONS),
    ("canonical_v2r1", V2R1_BASE_MUTATIONS),
    ("canonical_v2r2", V2R2_BASE_MUTATIONS),
    ("canonical_v2r3", V2R3_BASE_MUTATIONS),
    ("canonical_v2r4", V2R4_BASE_MUTATIONS),
    ("canonical_v3r1", V3R1_BASE_MUTATIONS),
    ("canonical_v4r1", V4R1_BASE_MUTATIONS),
    ("canonical_v5r1", V5R1_BASE_MUTATIONS),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--allow-legacy-data", action="store_true")
    parser.add_argument("--top-fold-finalists", type=int, default=4)
    return parser.parse_args()


def _signature(mutations: dict[str, Any]) -> str:
    raw = json.dumps(mutations, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _replay_source_fingerprint() -> str:
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    return ResearchReplayEngine(DATA_DIR, require_bundle=False).data_fingerprint()


def _code_fingerprint() -> str:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "backtests/stock/auto/iaric/worker.py",
        REPO_ROOT / "backtests/stock/auto/iaric/phase_candidates.py",
        REPO_ROOT / "backtests/stock/auto/config_mutator.py",
        REPO_ROOT / "backtests/stock/engine/iaric_pullback_engine.py",
        REPO_ROOT / "backtests/stock/engine/iaric_pullback_intraday_hybrid_engine.py",
        REPO_ROOT / "strategies/stock/iaric/core/logic.py",
        REPO_ROOT / "strategies/stock/iaric/exits.py",
        REPO_ROOT / "strategies/stock/iaric/config.py",
        REPO_ROOT / "strategies/stock/iaric/bar_policy.py",
        REPO_ROOT / "strategies/stock/iaric/models.py",
        REPO_ROOT / "strategies/stock/iaric/research.py",
        REPO_ROOT / "strategies/stock/iaric/signals.py",
        REPO_ROOT / "backtests/stock/engine/research_replay.py",
        REPO_ROOT / "backtests/stock/data/price_basis.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _normalize_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(mutations)
    # A completed-bar signal cannot fill on the same bar in live trading.
    normalized["param_overrides.pb_open_scored_fill_timing"] = "next_5m_open"
    return normalized


def _load_lineage_candidates() -> list[dict[str, Any]]:
    by_signature: dict[str, dict[str, Any]] = {}

    paths = sorted(IARIC_OUTPUT.rglob("optimized_config.json"))
    for path in paths:
        # Recovery outputs are derived candidates, not historical lineage.  A
        # rerun must not recursively promote its own prior selected artifact.
        if "baseline_recovery" in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            continue
        mutations = _normalize_mutations(payload)
        sig = _signature(mutations)
        record = by_signature.setdefault(
            sig,
            {
                "id": f"archive_{sig[:10]}",
                "mutations": mutations,
                "sources": [],
                "kind": "archived_endpoint",
            },
        )
        record["sources"].append(str(path.relative_to(REPO_ROOT)))

    for name, payload in CANONICAL_BASES:
        mutations = _normalize_mutations(payload)
        sig = _signature(mutations)
        record = by_signature.setdefault(
            sig,
            {
                "id": name,
                "mutations": mutations,
                "sources": [],
                "kind": "canonical_round_baseline",
            },
        )
        record["sources"].append(f"phase_candidates.py:{name}")

    return sorted(by_signature.values(), key=lambda item: item["id"])


def _partial_controls(lineage: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    controls = (
        ("repaired_stored", {}),
        (
            "partial_off",
            {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.0,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.0,
            },
        ),
        (
            "partial_neutral_050",
            {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.50,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.0,
            },
        ),
    )
    for record in lineage:
        for control_name, overrides in controls:
            mutations = dict(record["mutations"])
            mutations.update(overrides)
            sig = _signature(mutations)
            candidate = candidates.setdefault(
                sig,
                {
                    "id": f"{record['id']}__{control_name}",
                    "mutations": mutations,
                    "family": "lineage_partial_control",
                    "sources": [],
                },
            )
            candidate["sources"].extend(record["sources"])
    return sorted(candidates.values(), key=lambda item: item["id"])


def _economic_components(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "net_profit": math.tanh(float(metrics.get("net_profit", 0.0)) / 3000.0),
        "expected_total_r": math.tanh(float(metrics.get("expected_total_r", 0.0)) / 75.0),
        "avg_r": math.tanh(float(metrics.get("avg_r", 0.0)) / 0.10),
        "profit_factor": math.tanh((float(metrics.get("profit_factor", 0.0)) - 1.0) / 0.60),
        "sharpe": math.tanh(float(metrics.get("sharpe", 0.0)) / 2.0),
        "inverse_drawdown": math.tanh(
            (0.18 - float(metrics.get("max_drawdown_pct", 1.0))) / 0.12
        ),
        "total_trades": math.tanh(float(metrics.get("total_trades", 0.0)) / 800.0),
    }


def _economic_score(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    components = _economic_components(metrics)
    score = sum(float(SCORE_SPEC[key]["weight"]) * value for key, value in components.items())
    return float(score), components


def _read_cache(
    path: Path,
    source_fingerprint: str,
    code_fingerprint: str,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "source_fingerprint": source_fingerprint,
            "code_fingerprint": code_fingerprint,
            "evaluations": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("evaluations", {})
    previous_fingerprint = payload.get("source_fingerprint")
    if previous_fingerprint not in (None, source_fingerprint):
        return {
            "source_fingerprint": source_fingerprint,
            "code_fingerprint": code_fingerprint,
            "invalidated_previous_source_fingerprint": previous_fingerprint,
            "evaluations": {},
        }
    # Migrate the first repaired recovery run, which was completed against the
    # same working-tree data immediately before fingerprint namespacing landed.
    # Future source changes invalidate the entire cache above.
    if previous_fingerprint is None:
        payload["evaluations"] = {
            f"{source_fingerprint}|{key}": value
            for key, value in payload["evaluations"].items()
        }
    payload["source_fingerprint"] = source_fingerprint

    previous_code_fingerprint = payload.get("code_fingerprint")
    if previous_code_fingerprint not in (None, code_fingerprint):
        return {
            "source_fingerprint": source_fingerprint,
            "code_fingerprint": code_fingerprint,
            "invalidated_previous_code_fingerprint": previous_code_fingerprint,
            "evaluations": {},
        }
    if previous_code_fingerprint is None:
        prefix = f"{source_fingerprint}|"
        payload["evaluations"] = {
            (
                f"{source_fingerprint}|{code_fingerprint}|{key[len(prefix):]}"
                if key.startswith(prefix)
                else f"{source_fingerprint}|{code_fingerprint}|{key}"
            ): value
            for key, value in payload["evaluations"].items()
        }
        payload["code_fingerprint_migration"] = (
            "Existing recovery evaluations used collect_diagnostics=False. "
            "The migrated code change made the diagnostics ledger observer-only "
            "without changing that lean execution path; full-history Phase 0 "
            "confirmed identical selected-config economics."
        )
    payload["code_fingerprint"] = code_fingerprint
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Attribution caches are large enough that building a second monolithic
    # string with ``json.dumps`` can exhaust RAM while two replay workers still
    # retain their data bundles. Stream to a same-directory temporary and then
    # atomically replace the target so readers cannot observe a partial cache.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_evaluation_with_candidate(
    evaluation: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Attach identity/provenance without overwriting period-specific results."""

    result = {
        key: candidate[key]
        for key in ("id", "family", "sources")
        if key in candidate
    }
    result.update(evaluation)
    result["mutations"] = candidate["mutations"]
    return result


def _evaluate_batch(
    candidates: list[dict[str, Any]],
    *,
    start_date: str,
    end_date: str,
    max_workers: int,
    cache_path: Path,
    source_fingerprint: str,
    code_fingerprint: str,
    evaluation_fn: Any = evaluate_candidate_metrics,
    bundle_path: Path | None = None,
    require_bundle: bool | None = None,
) -> list[dict[str, Any]]:
    cache = _read_cache(cache_path, source_fingerprint, code_fingerprint)
    # Persist fingerprint migration/invalidation even when every requested
    # evaluation is already cached and no worker completion triggers a write.
    _write_json(cache_path, cache)
    evaluations: dict[str, Any] = cache["evaluations"]
    pending: list[tuple[str, dict[str, Any], str]] = []
    results: list[dict[str, Any]] = []
    cached_retry_keys: set[str] = set()

    def retryable_error(record: dict[str, Any]) -> bool:
        error = str(record.get("error", ""))
        return any(
            marker in error
            for marker in ("MemoryError", "ArrowMemoryError", "std::bad_alloc")
        )

    for candidate in candidates:
        sig = _signature(candidate["mutations"])
        cache_key = f"{source_fingerprint}|{code_fingerprint}|{start_date}|{end_date}|{sig}"
        if cache_key in evaluations:
            cached = evaluations[cache_key]
            if retryable_error(cached):
                # A resource failure is not an economic result. Never make it
                # sticky across supervisor restarts; retry it alone in a fresh
                # process so allocator high-water marks cannot accumulate.
                del evaluations[cache_key]
                cached_retry_keys.add(cache_key)
                pending.append((cache_key, candidate, sig))
            else:
                results.append(_merge_evaluation_with_candidate(cached, candidate))
        else:
            pending.append((cache_key, candidate, sig))

    if cached_retry_keys:
        cache["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json(cache_path, cache)

    if pending:
        print(
            f"Evaluating {len(pending)} candidates: {start_date} -> {end_date} "
            f"with {max_workers} workers",
            flush=True,
        )
        # Long-lived IARIC replay workers can retain allocator high-water marks
        # across heterogeneous configurations.  With the supported two-worker
        # ceiling, a two-item chunk means each process handles exactly one
        # candidate before recycling.  Retry a broken chunk with one worker.
        queue = list(pending)
        retry_counts: dict[str, int] = {key: 1 for key in cached_retry_keys}
        completed = 0
        while queue:
            # Retryable resource failures always receive a fresh, single-task
            # process. Ordinary candidates retain the supported two-worker
            # throughput.
            chunk_size = 1 if retry_counts.get(queue[0][0], 0) else 2
            chunk = queue[:chunk_size]
            queue = queue[chunk_size:]
            chunk_workers = 1 if chunk_size == 1 else max_workers
            failed: list[tuple[str, dict[str, Any], str]] = []
            with ProcessPoolExecutor(
                max_workers=chunk_workers,
                initializer=init_worker,
                initargs=(
                    str(DATA_DIR),
                    start_date,
                    end_date,
                    INITIAL_EQUITY,
                    0,
                    {},
                    {},
                    "v5r2",
                    str(bundle_path) if bundle_path is not None else None,
                    require_bundle,
                ),
            ) as pool:
                future_map = {
                    pool.submit(
                        evaluation_fn,
                        (candidate["id"], candidate["mutations"], {}),
                    ): (cache_key, candidate, sig)
                    for cache_key, candidate, sig in chunk
                }
                for future in as_completed(future_map):
                    cache_key, candidate, sig = future_map[future]
                    try:
                        worker_result = future.result()
                    except BrokenProcessPool:
                        failed.append((cache_key, candidate, sig))
                        continue
                    if retryable_error(worker_result):
                        failed.append((cache_key, candidate, sig))
                        print(
                            f"  [retry] {candidate['id']}: transient resource error",
                            flush=True,
                        )
                        continue
                    completed += 1
                    record = {
                        "signature": sig,
                        # Keep the immutable configuration beside its digest.
                        # Structural runs may safely re-key legacy evaluations
                        # when opt-in settings are added, but must fail closed
                        # once a cache contains an evaluation of those settings.
                        "mutations": dict(candidate["mutations"]),
                        "start_date": start_date,
                        "end_date": end_date,
                        "metrics": worker_result.get("metrics", {}),
                        "error": worker_result.get("error", ""),
                    }
                    for extra_key in ("trade_attribution", "funnel_counters"):
                        if extra_key in worker_result:
                            record[extra_key] = worker_result[extra_key]
                    if not record["error"]:
                        score, components = _economic_score(record["metrics"])
                        record["economic_score"] = score
                        record["score_components"] = components
                    evaluations[cache_key] = record
                    cache["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    _write_json(cache_path, cache)
                    results.append(_merge_evaluation_with_candidate(record, candidate))
                    status = "ERROR" if record["error"] else (
                        f"score={record['economic_score']:+.4f} "
                        f"net=${record['metrics'].get('net_profit', 0.0):,.0f} "
                        f"PF={record['metrics'].get('profit_factor', 0.0):.2f}"
                    )
                    print(
                        f"  [{completed:02d}/{len(pending):02d}] {candidate['id']}: {status}",
                        flush=True,
                    )
            if failed:
                for item in failed:
                    retry_counts[item[0]] = retry_counts.get(item[0], 0) + 1
                    if retry_counts[item[0]] > 2:
                        raise RuntimeError(
                            f"Worker pool repeatedly failed while evaluating {item[1]['id']}"
                        )
                print(
                    f"Worker pool recycled after an abrupt exit; retrying {len(failed)} "
                    "unfinished candidates with one worker.",
                    flush=True,
                )
                queue = failed + queue

    return sorted(results, key=lambda item: item["id"])


def _full_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = record.get("metrics", {})
    safety = (
        float(metrics.get("total_trades", 0.0)) >= 200
        and float(metrics.get("max_drawdown_pct", 1.0)) <= 0.25
    )
    return (
        1.0 if safety else 0.0,
        float(record.get("economic_score", -99.0)),
        float(metrics.get("expected_total_r", -1e9)),
        -float(metrics.get("max_drawdown_pct", 1.0)),
    )


def _fold_summary(
    candidate: dict[str, Any],
    fold_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    sig = _signature(candidate["mutations"])
    rows = []
    for fold_name, _, _ in FOLDS:
        match = next((row for row in fold_results[fold_name] if row["signature"] == sig), None)
        if match is not None and not match.get("error"):
            rows.append({"fold": fold_name, **_compact_metrics(match["metrics"])})
    avg_rs = [float(row.get("avg_r", 0.0)) for row in rows]
    pfs = [float(row.get("profit_factor", 0.0)) for row in rows]
    dds = [float(row.get("max_drawdown_pct", 1.0)) for row in rows]
    positive = sum(value > 0.0 for value in avg_rs)
    fold_count = len(rows)
    median_avg_r = sorted(avg_rs)[fold_count // 2] if fold_count else -99.0
    worst_avg_r = min(avg_rs, default=-99.0)
    max_fold_dd = max(dds, default=1.0)
    min_fold_pf = min(pfs, default=0.0)
    # The stability overlay penalizes dispersion and bad tails.  It is a fixed
    # validation rule, not part of the seven-component optimizer objective.
    instability_penalty = (
        0.10 * max(0.0, -worst_avg_r / 0.10)
        + 0.08 * max(0.0, max_fold_dd - 0.15) / 0.10
        + 0.04 * max(0.0, 2 - positive)
    )
    robust_eligible = (
        fold_count == len(FOLDS)
        and positive >= 2
        and worst_avg_r >= -0.12
        and max_fold_dd <= 0.25
    )
    return {
        "folds": rows,
        "positive_fold_count": positive,
        "median_fold_avg_r": median_avg_r,
        "worst_fold_avg_r": worst_avg_r,
        "min_fold_profit_factor": min_fold_pf,
        "max_fold_drawdown_pct": max_fold_dd,
        "instability_penalty": instability_penalty,
        "robust_eligible": robust_eligible,
        "validated_score": float(candidate.get("economic_score", -99.0)) - instability_penalty,
    }


def _validated_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    validation = record.get("validation", {})
    metrics = record.get("metrics", {})
    return (
        1.0 if validation.get("robust_eligible") else 0.0,
        float(validation.get("validated_score", -99.0)),
        float(validation.get("positive_fold_count", 0.0)),
        float(metrics.get("expected_total_r", -1e9)),
    )


def _family_ablation_candidates(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    base = deepcopy(baseline["mutations"])
    get = base.get

    floor = float(get("param_overrides.pb_v2_signal_floor", 72.0))
    daily = float(get("param_overrides.pb_daily_signal_min_score", 54.0))
    rescue = float(get("param_overrides.pb_daily_rescue_min_score", 52.0))
    delayed = float(get("param_overrides.pb_delayed_confirm_score_min", 52.0))

    families: tuple[tuple[str, dict[str, Any]], ...] = (
        ("unchanged_finalist", {}),
        (
            "signal_selectivity_stricter",
            {
                "param_overrides.pb_v2_signal_floor": min(84.0, floor + 6.0),
                "param_overrides.pb_daily_signal_min_score": min(70.0, daily + 4.0),
                "param_overrides.pb_daily_rescue_min_score": min(68.0, rescue + 4.0),
                "param_overrides.pb_delayed_confirm_score_min": min(72.0, delayed + 6.0),
            },
        ),
        (
            "signal_selectivity_looser",
            {
                "param_overrides.pb_v2_signal_floor": max(54.0, floor - 6.0),
                "param_overrides.pb_daily_signal_min_score": max(46.0, daily - 4.0),
                "param_overrides.pb_daily_rescue_min_score": max(44.0, rescue - 4.0),
                "param_overrides.pb_delayed_confirm_score_min": max(46.0, delayed - 6.0),
            },
        ),
        (
            "context_selective",
            {
                "param_overrides.pb_v2_gap_max_pct": 1.0,
                "param_overrides.pb_cdd_max": 4,
            },
        ),
        (
            "context_broad",
            {
                "param_overrides.pb_v2_gap_max_pct": 3.0,
                "param_overrides.pb_cdd_max": 9,
            },
        ),
        (
            "entry_core_only",
            {
                "param_overrides.pb_v2_vwap_bounce_enabled": False,
                "param_overrides.pb_v2_afternoon_retest_enabled": False,
                "param_overrides.pb_opening_reclaim_enabled": False,
            },
        ),
        (
            "entry_multiroute",
            {
                "param_overrides.pb_v2_vwap_bounce_enabled": True,
                "param_overrides.pb_v2_afternoon_retest_enabled": True,
                "param_overrides.pb_opening_reclaim_enabled": True,
            },
        ),
        ("entry_no_delayed_confirm", {"param_overrides.pb_delayed_confirm_enabled": False}),
        (
            "exit_partial_off",
            {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.0,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.0,
            },
        ),
        (
            "exit_patient",
            {
                "param_overrides.pb_atr_stop_mult": 1.5,
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.75,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.10,
                "param_overrides.pb_v2_stale_mfe_thresh": 0.10,
                "param_overrides.pb_v2_stale_bars": 6,
            },
        ),
        (
            "exit_defensive",
            {
                "param_overrides.pb_atr_stop_mult": 1.0,
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.30,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.0,
                "param_overrides.pb_v2_mfe_stage1_trigger": 0.35,
                "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
                "param_overrides.pb_v2_stale_bars": 3,
            },
        ),
        (
            "carry_intraday_only",
            {
                "param_overrides.pb_max_hold_days": 1,
                "param_overrides.pb_open_scored_max_hold_days": 1,
            },
        ),
        (
            "carry_selective",
            {
                "param_overrides.pb_carry_close_pct_min": 0.25,
                "param_overrides.pb_carry_mfe_gate_r": 0.20,
                "param_overrides.pb_open_scored_carry_close_pct_min": 0.25,
                "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
                "param_overrides.pb_max_hold_days": 2,
                "param_overrides.pb_open_scored_max_hold_days": 2,
            },
        ),
        ("flow_hard_gate", {"param_overrides.pb_flow_policy": "hard_gate"}),
        ("flow_soft_rescue", {"param_overrides.pb_flow_policy": "soft_penalty_rescue"}),
        (
            "capacity_concentrated",
            {
                "param_overrides.pb_max_positions": 6,
                "param_overrides.max_positions_per_sector": 3,
            },
        ),
        (
            "capacity_aggressive",
            {
                "param_overrides.pb_max_positions": 12,
                "param_overrides.max_positions_per_sector": 5,
            },
        ),
    )

    deduped: dict[str, dict[str, Any]] = {}
    for family, changes in families:
        mutations = dict(base)
        mutations.update(changes)
        sig = _signature(mutations)
        deduped.setdefault(
            sig,
            {
                "id": f"ablation__{family}",
                "mutations": mutations,
                "family": family,
                "sources": [baseline["id"]],
            },
        )
    return list(deduped.values())


def _compact_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = (
        "net_profit",
        "total_trades",
        "avg_r",
        "expected_total_r",
        "profit_factor",
        "sharpe",
        "max_drawdown_pct",
    )
    return {key: float(metrics.get(key, 0.0)) for key in keys}


def _report_table(records: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    table = []
    for rank, record in enumerate(records[:limit], 1):
        table.append(
            {
                "rank": rank,
                "id": record["id"],
                "economic_score": float(record.get("economic_score", -99.0)),
                "metrics": _compact_metrics(record.get("metrics", {})),
                "validation": record.get("validation", {}),
            }
        )
    return table


def main() -> None:
    args = _parse_args()
    if args.max_workers < 1 or args.max_workers > MAX_WORKERS:
        raise ValueError(f"Recovery must use between 1 and {MAX_WORKERS} workers.")
    if args.end_date >= HOLDOUT_START:
        raise ValueError(
            f"End date {args.end_date} overlaps the sealed holdout beginning {HOLDOUT_START}."
        )
    if args.allow_legacy_data:
        os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "evaluation_cache.json"
    source_fingerprint = _replay_source_fingerprint()
    code_fingerprint = _code_fingerprint()

    lineage = _load_lineage_candidates()
    bakeoff_candidates = _partial_controls(lineage)
    print("=" * 76)
    print("IARIC EXECUTION-CORRECTED BASELINE RECOVERY")
    print("=" * 76)
    print(f"Distinct lineage configurations: {len(lineage)}")
    print(f"Lineage/partial-control candidates: {len(bakeoff_candidates)}")
    print(f"Training window: {args.start_date} -> {args.end_date}")
    print(f"Sealed holdout begins: {HOLDOUT_START}")
    print(f"Data authority: {'legacy_diagnostic_only' if args.allow_legacy_data else 'frozen_required'}")
    print(f"Workers: {args.max_workers}")
    print(f"Immutable score components ({len(SCORE_SPEC)}): {', '.join(SCORE_SPEC)}", flush=True)

    full_results = _evaluate_batch(
        bakeoff_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    errors = [row for row in full_results if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} lineage evaluations failed; see errors.json")
    full_ranked = sorted(full_results, key=_full_rank_key, reverse=True)
    finalists = full_ranked[: max(1, args.top_fold_finalists)]

    fold_results: dict[str, list[dict[str, Any]]] = {}
    for fold_name, fold_start, fold_end in FOLDS:
        fold_results[fold_name] = _evaluate_batch(
            finalists,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
        )
    for finalist in finalists:
        finalist["validation"] = _fold_summary(finalist, fold_results)
    validated_finalists = sorted(finalists, key=_validated_rank_key, reverse=True)
    lineage_winner = validated_finalists[0]

    ablation_candidates = _family_ablation_candidates(lineage_winner)
    ablation_full = _evaluate_batch(
        ablation_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source_fingerprint=source_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    errors = [row for row in ablation_full if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} ablation evaluations failed; see errors.json")
    ablation_ranked = sorted(ablation_full, key=_full_rank_key, reverse=True)

    # Validate the best ablations plus the unchanged lineage winner.  Signature
    # deduplication prevents the control from consuming another replay.
    ablation_finalists_by_sig: dict[str, dict[str, Any]] = {}
    for candidate in ablation_ranked[: max(4, args.top_fold_finalists)]:
        ablation_finalists_by_sig.setdefault(_signature(candidate["mutations"]), candidate)
    ablation_finalists_by_sig.setdefault(_signature(lineage_winner["mutations"]), lineage_winner)
    ablation_finalists = list(ablation_finalists_by_sig.values())
    ablation_fold_results: dict[str, list[dict[str, Any]]] = {}
    for fold_name, fold_start, fold_end in FOLDS:
        ablation_fold_results[fold_name] = _evaluate_batch(
            ablation_finalists,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source_fingerprint=source_fingerprint,
            code_fingerprint=code_fingerprint,
        )
    for candidate in ablation_finalists:
        candidate["validation"] = _fold_summary(candidate, ablation_fold_results)
    validated_ablation = sorted(ablation_finalists, key=_validated_rank_key, reverse=True)
    winner = validated_ablation[0]

    # Do not promote a family mutation unless its stability-adjusted result
    # beats the recovered lineage endpoint.  This is the anti-overfit guardrail.
    lineage_key = _validated_rank_key(lineage_winner)
    if _validated_rank_key(winner) < lineage_key:
        winner = lineage_winner

    status = (
        "provisional_legacy_data_revalidation_required"
        if args.allow_legacy_data
        else "recovered_training_baseline_holdout_still_sealed"
    )
    selected_config = dict(sorted(winner["mutations"].items()))
    _write_json(output_dir / "optimized_config.json", selected_config)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "purpose": "execution-corrected robust baseline for the next phased auto round",
        "data_authority": "legacy_diagnostic_only" if args.allow_legacy_data else "frozen_bundle",
        "data_source_fingerprint": source_fingerprint,
        "code_fingerprint": code_fingerprint,
        "training_window": {"start": args.start_date, "end": args.end_date},
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
        "max_workers": args.max_workers,
        "worker_policy": (
            "two-worker ceiling with one-candidate process isolation; memory-pressure "
            "recovery may continue unfinished chunks with one worker"
        ),
        "score_spec": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "chronological_folds": [
            {"name": name, "start": start, "end": end} for name, start, end in FOLDS
        ],
        "lineage_inventory_count": len(lineage),
        "bakeoff_candidate_count": len(bakeoff_candidates),
        "ablation_candidate_count": len(ablation_candidates),
        "lineage_winner": {
            "id": lineage_winner["id"],
            "sources": lineage_winner.get("sources", []),
            "economic_score": lineage_winner.get("economic_score"),
            "metrics": _compact_metrics(lineage_winner["metrics"]),
            "validation": lineage_winner["validation"],
        },
        "selected": {
            "id": winner["id"],
            "family": winner.get("family", "lineage"),
            "sources": winner.get("sources", []),
            "signature": _signature(selected_config),
            "economic_score": winner.get("economic_score"),
            "metrics": _compact_metrics(winner["metrics"]),
            "validation": winner["validation"],
        },
        "promotion_policy": (
            "Seven-component economic score, full-period safety gate, four chronological "
            "folds, and stability penalty; family ablations cannot displace the lineage "
            "winner without a better fixed validation rank."
        ),
        "next_command": (
            "python -m backtests.stock.auto.runners.run_v5r2_targeted_alpha "
            f"--baseline-config {output_dir / 'optimized_config.json'} --max-workers 2"
        ),
    }
    _write_json(output_dir / "recovery_manifest.json", manifest)
    _write_json(output_dir / "lineage_inventory.json", lineage)
    _write_json(output_dir / "lineage_ranking.json", _report_table(full_ranked, limit=len(full_ranked)))
    _write_json(output_dir / "validated_lineage_finalists.json", _report_table(validated_finalists))
    _write_json(output_dir / "ablation_ranking.json", _report_table(ablation_ranked, limit=len(ablation_ranked)))
    _write_json(output_dir / "validated_ablation_finalists.json", _report_table(validated_ablation))

    summary = [
        "=" * 76,
        "IARIC REPAIRED BASELINE RECOVERY COMPLETE",
        "=" * 76,
        f"Status: {status}",
        f"Selected: {winner['id']} ({winner.get('family', 'lineage')})",
        f"Signature: {_signature(selected_config)}",
        f"Economic score: {float(winner.get('economic_score', 0.0)):+.6f}",
        (
            f"Trades={winner['metrics'].get('total_trades', 0.0):.0f} "
            f"Net=${winner['metrics'].get('net_profit', 0.0):,.2f} "
            f"ExpR={winner['metrics'].get('expected_total_r', 0.0):.2f} "
            f"AvgR={winner['metrics'].get('avg_r', 0.0):+.4f}"
        ),
        (
            f"PF={winner['metrics'].get('profit_factor', 0.0):.3f} "
            f"Sharpe={winner['metrics'].get('sharpe', 0.0):+.3f} "
            f"DD={winner['metrics'].get('max_drawdown_pct', 0.0):.2%}"
        ),
        (
            f"Positive folds={winner['validation']['positive_fold_count']}/{len(FOLDS)} "
            f"Worst fold AvgR={winner['validation']['worst_fold_avg_r']:+.4f} "
            f"Max fold DD={winner['validation']['max_fold_drawdown_pct']:.2%}"
        ),
        f"Config: {output_dir / 'optimized_config.json'}",
        "Holdout was not accessed.",
    ]
    (output_dir / "recovery_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
