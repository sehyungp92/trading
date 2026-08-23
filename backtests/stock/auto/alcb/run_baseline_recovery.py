"""Reconstruct a robust corrected-RTH ALCB baseline before phased auto.

The runner replays every distinct historical optimized endpoint with its era's
implicit defaults materialized, adds a clean pre-auto anchor, validates the
best lineages on chronological folds, and then runs broad family
counterfactuals.  Finalists must also survive real 7.5/10 bps replays.  The
already-consumed post-2026-03-01 period is never opened or used for selection.

The projected-RTH cache is diagnostic rather than authoritative, so this run
requires ``--allow-projected-rth-data`` and can only establish a provisional
research baseline pending an unchanged direct-RTH replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.stock.auto.alcb.worker import evaluate_candidate_metrics, init_worker
from backtests.stock.data.calendar import RTH_SESSION_POLICY


DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
ALCB_OUTPUT = REPO_ROOT / "backtests/output/stock/alcb"
DEFAULT_OUTPUT = ALCB_OUTPUT / "baseline_recovery_rth_20260816"
START_DATE = "2024-03-25"
END_DATE = "2026-03-01"
CONSUMED_START = "2026-03-02"
INITIAL_EQUITY = 10_000.0
MAX_WORKERS = 2

# The first completed run fingerprinted this orchestration/report file as well
# as the economic evaluation path.  The only subsequent change restored a
# dropped parent-signature field while joining already-computed cost records;
# worker, engine, data, mutations, dates, and metrics were unchanged.  Migrate
# that cache once into the evaluation-only fingerprint namespace below.
_BOOKKEEPING_COMPATIBLE_CACHE_FINGERPRINTS = {
    "246f1254353eeb750fe820b42f3d1e7302279547bf1467902c15910ceb3951e7",
}

FOLDS: tuple[tuple[str, str, str], ...] = (
    ("fold_1", "2024-03-25", "2024-09-30"),
    ("fold_2", "2024-10-01", "2025-03-31"),
    ("fold_3", "2025-04-01", "2025-09-30"),
    ("fold_4", "2025-10-01", "2026-03-01"),
)

# Fixed before looking at recovery results.  Return and throughput receive 53%
# of the weight, which is aggressive-leaning; risk-adjusted quality receives
# 25%, and drawdown/Sharpe 22%.  Scales are economic targets, not sample maxima.
SCORE_SPEC: dict[str, dict[str, float | str]] = {
    "expected_total_r": {"weight": 0.22, "transform": "tanh(x / 80R)"},
    "net_profit": {"weight": 0.16, "transform": "tanh(x / $3000)"},
    "avg_r": {"weight": 0.13, "transform": "tanh(x / 0.10R)"},
    "profit_factor": {"weight": 0.12, "transform": "tanh((x - 1) / 0.50)"},
    "trades_per_month": {"weight": 0.15, "transform": "tanh(x / 35)"},
    "inverse_drawdown": {"weight": 0.12, "transform": "tanh((0.08 - x) / 0.06)"},
    "sharpe": {"weight": 0.10, "transform": "tanh(x / 2.5)"},
}

# StrategySettings changed after several saved endpoints.  Explicitly applying
# the era snapshot prevents omitted keys from inheriting today's baked values.
PRE_AUTO_DEFAULTS: dict[str, Any] = {
    "param_overrides.base_risk_fraction": 0.0065,
    "param_overrides.daily_stop_r": 2.0,
    "param_overrides.heat_cap_r": 7.0,
    "param_overrides.portfolio_daily_stop_r": 5.0,
    "param_overrides.max_positions": 8,
    "param_overrides.opening_range_bars": 6,
    "param_overrides.rvol_threshold": 2.0,
    "param_overrides.stop_atr_multiple": 1.0,
    "param_overrides.flow_reversal_min_hold_bars": 12,
    "param_overrides.breakout_distance_cap_r": 0.0,
    "param_overrides.sector_mult_financials": 0.50,
    "param_overrides.qe_stage1_bars": 0,
    "param_overrides.adaptive_trail_late_activate_r": 0.25,
    "param_overrides.adaptive_trail_late_distance_r": 0.20,
    "param_overrides.entry_score_blocklist": [],
    "param_overrides.entry_score_size_mults": {},
    "param_overrides.entry_detail_size_mults": {},
    "param_overrides.failure_stop_bars": 0,
    "param_overrides.failure_stop_mfe_max_r": 0.0,
    "param_overrides.failure_stop_current_r_max": -999.0,
    "param_overrides.failure_stop_to_r": -1.0,
    "param_overrides.orb_entry_range_cap_r": 0.0,
    "param_overrides.selection_long_count": 20,
    "ablation.use_daily_stop": False,
}

MAY_DEFAULTS: dict[str, Any] = {
    **PRE_AUTO_DEFAULTS,
    "param_overrides.base_risk_fraction": 0.00702,
    "param_overrides.daily_stop_r": 2.35,
    "param_overrides.heat_cap_r": 4.0,
    "param_overrides.portfolio_daily_stop_r": 3.5,
    "param_overrides.max_positions": 6,
    "param_overrides.stop_atr_multiple": 0.8,
    "param_overrides.breakout_distance_cap_r": 1.0,
    "param_overrides.sector_mult_financials": 0.65,
    "param_overrides.qe_stage1_bars": 10,
    "param_overrides.adaptive_trail_late_activate_r": 0.22,
    "param_overrides.adaptive_trail_late_distance_r": 0.12,
    "param_overrides.entry_score_blocklist": ["COMBINED_BREAKOUT:5"],
    "param_overrides.entry_score_size_mults": {
        "OR_BREAKOUT:5": 0.75,
        "COMBINED_BREAKOUT:7": 1.15,
        "PDH_BREAKOUT:6": 0.5,
    },
    "param_overrides.entry_detail_size_mults": {
        "OR_BREAKOUT:5:!bar_vol_surge": 0.55,
    },
    "param_overrides.failure_stop_bars": 10,
    "param_overrides.failure_stop_mfe_max_r": 0.2,
    "param_overrides.failure_stop_current_r_max": 0.0,
    "param_overrides.failure_stop_to_r": -0.25,
    "param_overrides.orb_entry_range_cap_r": 1.1,
}

CURRENT_DEFAULTS: dict[str, Any] = {
    **MAY_DEFAULTS,
    "param_overrides.opening_range_bars": 9,
    "param_overrides.rvol_threshold": 1.7,
    "param_overrides.flow_reversal_min_hold_bars": 8,
    "param_overrides.adaptive_trail_late_distance_r": 0.04,
    "param_overrides.orb_entry_range_cap_r": 1.25,
    "param_overrides.selection_long_count": 30,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--top-fold-finalists", type=int, default=4)
    parser.add_argument("--top-cost-finalists", type=int, default=4)
    parser.add_argument("--allow-projected-rth-data", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _signature(mutations: dict[str, Any]) -> str:
    raw = json.dumps(_json_safe(mutations), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_fingerprint() -> str:
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    return ResearchReplayEngine(DATA_DIR, require_bundle=False).data_fingerprint()


def _code_fingerprint() -> str:
    paths = (
        REPO_ROOT / "backtests/stock/auto/alcb/worker.py",
        REPO_ROOT / "backtests/stock/auto/alcb/phase_scoring.py",
        REPO_ROOT / "backtests/stock/auto/config_mutator.py",
        REPO_ROOT / "backtests/stock/config_alcb.py",
        REPO_ROOT / "backtests/stock/engine/alcb_engine.py",
        REPO_ROOT / "backtests/stock/engine/research_replay.py",
        REPO_ROOT / "strategies/stock/alcb/config.py",
        REPO_ROOT / "strategies/stock/alcb/risk.py",
        REPO_ROOT / "strategies/stock/alcb/signals.py",
        REPO_ROOT / "strategies/stock/alcb/exits.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _era_defaults(path: Path) -> tuple[str, dict[str, Any]]:
    normalized = str(path).replace("\\", "/")
    if "/round_4/" in normalized or "20260723_1839_pre_r3_relabel" in normalized:
        return "current_round4", CURRENT_DEFAULTS
    if "260429_r1_pre_sanitized" in normalized or "round_2_pre_rerun_20260429" in normalized:
        return "pre_auto_april", PRE_AUTO_DEFAULTS
    return "may_rounds", MAY_DEFAULTS


def _normalize_endpoint(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    mutations = deepcopy(defaults)
    mutations.update(raw)
    mutations["intraday_session_policy"] = RTH_SESSION_POLICY
    # Some old artifacts serialized a one-item blocklist as a scalar string.
    for key in (
        "param_overrides.entry_score_blocklist",
        "param_overrides.entry_detail_blocklist",
        "param_overrides.block_entry_bars",
        "param_overrides.entry_type_bar_blocklist",
    ):
        value = mutations.get(key)
        if isinstance(value, str):
            mutations[key] = [value]
    return mutations


def _load_lineage() -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    clean = _normalize_endpoint({}, PRE_AUTO_DEFAULTS)
    candidates[_signature(clean)] = {
        "id": "clean_pre_auto_78b6398",
        "family": "clean_anchor",
        "era": "pre_auto_april",
        "mutations": clean,
        "sources": ["git:78b6398 StrategySettings/control snapshot"],
    }

    for path in sorted(ALCB_OUTPUT.rglob("optimized_config.json")):
        normalized_path = str(path).replace("\\", "/")
        if "baseline_recovery" in normalized_path or "oos_robustness" in normalized_path:
            continue
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict) or not raw:
            continue
        era, defaults = _era_defaults(path)
        mutations = _normalize_endpoint(raw, defaults)
        sig = _signature(mutations)
        record = candidates.setdefault(
            sig,
            {
                "id": f"lineage_{sig[:10]}",
                "family": "historical_endpoint",
                "era": era,
                "mutations": mutations,
                "sources": [],
            },
        )
        record["sources"].append(str(path.relative_to(REPO_ROOT)))

    return sorted(candidates.values(), key=lambda row: row["id"])


def _components(metrics: dict[str, float]) -> dict[str, float]:
    avg_r = float(metrics.get("avg_r", metrics.get("expectancy", 0.0)))
    return {
        "expected_total_r": math.tanh(float(metrics.get("expected_total_r", 0.0)) / 80.0),
        "net_profit": math.tanh(float(metrics.get("net_profit", 0.0)) / 3000.0),
        "avg_r": math.tanh(avg_r / 0.10),
        "profit_factor": math.tanh((float(metrics.get("profit_factor", 0.0)) - 1.0) / 0.50),
        "trades_per_month": math.tanh(float(metrics.get("trades_per_month", 0.0)) / 35.0),
        "inverse_drawdown": math.tanh(
            (0.08 - float(metrics.get("max_drawdown_pct", 1.0))) / 0.06
        ),
        "sharpe": math.tanh(float(metrics.get("sharpe", 0.0)) / 2.5),
    }


def _economic_score(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    components = _components(metrics)
    score = sum(
        float(SCORE_SPEC[key]["weight"]) * value
        for key, value in components.items()
    )
    return float(score), components


def _read_cache(path: Path, source: str, code: str) -> dict[str, Any]:
    empty = {"source_fingerprint": source, "code_fingerprint": code, "evaluations": {}}
    if not path.exists():
        return empty
    payload = json.loads(path.read_text(encoding="utf-8"))
    previous_source = payload.get("source_fingerprint")
    previous_code = payload.get("code_fingerprint")
    if previous_source != source:
        empty["invalidated_previous"] = {
            "source_fingerprint": previous_source,
            "code_fingerprint": previous_code,
        }
        return empty
    payload.setdefault("evaluations", {})
    if previous_code != code:
        if previous_code not in _BOOKKEEPING_COMPATIBLE_CACHE_FINGERPRINTS:
            empty["invalidated_previous"] = {
                "source_fingerprint": previous_source,
                "code_fingerprint": previous_code,
            }
            return empty
        old_token = f"|{previous_code}|"
        new_token = f"|{code}|"
        payload["evaluations"] = {
            key.replace(old_token, new_token, 1): value
            for key, value in payload["evaluations"].items()
        }
        payload["code_fingerprint"] = code
        payload["cache_migration"] = (
            "Migrated completed economics after an observer-only parent-signature "
            "join repair; no engine evaluation input or output changed."
        )
    return payload


def _merge(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: candidate.get(key)
        for key in ("id", "family", "era", "sources", "base_signature")
        if key in candidate
    }
    output.update(record)
    output["mutations"] = candidate["mutations"]
    return output


def _evaluate_batch(
    candidates: list[dict[str, Any]],
    *,
    start_date: str,
    end_date: str,
    max_workers: int,
    cache_path: Path,
    source: str,
    code: str,
) -> list[dict[str, Any]]:
    cache = _read_cache(cache_path, source, code)
    _write_json(cache_path, cache)
    evaluations = cache["evaluations"]
    results: list[dict[str, Any]] = []
    pending: list[tuple[str, str, dict[str, Any]]] = []

    for candidate in candidates:
        sig = _signature(candidate["mutations"])
        key = f"{source}|{code}|{start_date}|{end_date}|{sig}"
        cached = evaluations.get(key)
        if cached is not None:
            results.append(_merge(cached, candidate))
        else:
            pending.append((key, sig, candidate))

    if pending:
        print(
            f"Evaluating {len(pending)} candidate(s): {start_date} -> {end_date} "
            f"with {max_workers} worker(s)",
            flush=True,
        )
        completed = 0
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(str(DATA_DIR), start_date, end_date, INITIAL_EQUITY, 0, {}, {}),
        ) as pool:
            future_map = {
                pool.submit(
                    evaluate_candidate_metrics,
                    (candidate["id"], candidate["mutations"], {}),
                ): (key, sig, candidate)
                for key, sig, candidate in pending
            }
            for future in as_completed(future_map):
                key, sig, candidate = future_map[future]
                try:
                    worker = future.result()
                except Exception as exc:  # process failures are persisted for diagnosis
                    worker = {"metrics": {}, "error": repr(exc)}
                completed += 1
                record = {
                    "signature": sig,
                    "start_date": start_date,
                    "end_date": end_date,
                    "metrics": worker.get("metrics", {}),
                    "error": worker.get("error", ""),
                }
                if not record["error"]:
                    score, score_components = _economic_score(record["metrics"])
                    record["economic_score"] = score
                    record["score_components"] = score_components
                evaluations[key] = record
                cache["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                _write_json(cache_path, cache)
                results.append(_merge(record, candidate))
                if record["error"]:
                    status = "ERROR"
                else:
                    metrics = record["metrics"]
                    status = (
                        f"score={record['economic_score']:+.4f} "
                        f"R={metrics.get('expected_total_r', 0.0):+.1f} "
                        f"TPM={metrics.get('trades_per_month', 0.0):.1f} "
                        f"PF={metrics.get('profit_factor', 0.0):.2f} "
                        f"DD={metrics.get('max_drawdown_pct', 0.0):.1%}"
                    )
                print(
                    f"  [{completed:02d}/{len(pending):02d}] {candidate['id']}: {status}",
                    flush=True,
                )

    return sorted(results, key=lambda row: row["id"])


def _safety_gate(metrics: dict[str, float]) -> bool:
    return bool(
        float(metrics.get("expected_total_r", 0.0)) > 0.0
        and float(metrics.get("net_profit", 0.0)) > 0.0
        and float(metrics.get("trades_per_month", 0.0)) >= 20.0
        and float(metrics.get("profit_factor", 0.0)) >= 1.10
        and float(metrics.get("max_drawdown_pct", 1.0)) <= 0.08
    )


def _full_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = record.get("metrics", {})
    return (
        1.0 if _safety_gate(metrics) else 0.0,
        float(record.get("economic_score", -99.0)),
        float(metrics.get("expected_total_r", -1e9)),
        float(metrics.get("trades_per_month", 0.0)),
        -float(metrics.get("max_drawdown_pct", 1.0)),
    )


def _fold_summary(
    candidate: dict[str, Any], fold_results: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    sig = _signature(candidate["mutations"])
    rows: list[dict[str, Any]] = []
    for name, _, _ in FOLDS:
        match = next((row for row in fold_results[name] if row["signature"] == sig), None)
        if match is None or match.get("error"):
            continue
        metrics = match["metrics"]
        rows.append({"fold": name, **_compact_metrics(metrics)})
    avg_rs = [float(row["avg_r"]) for row in rows]
    pfs = [float(row["profit_factor"]) for row in rows]
    dds = [float(row["max_drawdown_pct"]) for row in rows]
    positive = sum(value > 0.0 for value in avg_rs)
    worst = min(avg_rs, default=-99.0)
    dispersion = pstdev(avg_rs) if len(avg_rs) > 1 else 1.0
    max_dd = max(dds, default=1.0)
    min_pf = min(pfs, default=0.0)
    penalty = (
        0.12 * min(1.0, dispersion / 0.10)
        + 0.10 * min(1.0, max(0.0, -worst) / 0.05)
        + 0.08 * max(0, 3 - positive)
        + 0.08 * min(1.0, max(0.0, max_dd - 0.07) / 0.05)
    )
    robust = bool(
        len(rows) == len(FOLDS)
        and positive >= 3
        and worst >= -0.02
        and max_dd <= 0.07
        and min_pf >= 0.95
    )
    return {
        "folds": rows,
        "positive_fold_count": positive,
        "worst_fold_avg_r": worst,
        "avg_r_dispersion": dispersion,
        "minimum_fold_profit_factor": min_pf,
        "maximum_fold_drawdown_pct": max_dd,
        "stability_penalty": penalty,
        "robust_eligible": robust,
        "validated_score": float(candidate.get("economic_score", -99.0)) - penalty,
    }


def _validated_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    validation = record.get("validation", {})
    return (
        1.0 if validation.get("robust_eligible") else 0.0,
        float(validation.get("validated_score", -99.0)),
        float(validation.get("positive_fold_count", 0.0)),
        float(record.get("economic_score", -99.0)),
    )


def _family_candidates(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    base = deepcopy(baseline["mutations"])
    rows: list[tuple[str, str, dict[str, Any]]] = [
        ("unchanged", "control", {}),
        (
            "equal_risk_signals",
            "sizing",
            {
                "param_overrides.entry_score_size_mults": {},
                "param_overrides.entry_detail_size_mults": {},
                "param_overrides.pdh_size_mult": 1.0,
                "param_overrides.regime_mult_b": 1.0,
                "param_overrides.momentum_size_mult_score_3": 1.0,
                "param_overrides.momentum_size_mult_score_4": 1.0,
                "param_overrides.momentum_size_mult_score_5": 1.0,
                "param_overrides.momentum_size_mult_score_6": 1.0,
                "param_overrides.momentum_size_mult_score_7_plus": 1.0,
            },
        ),
        (
            "remove_score_detail_sizing",
            "sizing",
            {
                "param_overrides.entry_score_size_mults": {},
                "param_overrides.entry_detail_size_mults": {},
            },
        ),
        (
            "loose_entry_geometry",
            "entry_geometry",
            {
                "ablation.use_or_width_min": False,
                "ablation.use_orb_entry_range_gate": False,
                "param_overrides.pdh_avwap_cap_pct": 0.0,
                "param_overrides.combined_avwap_cap_pct": 0.0,
            },
        ),
        (
            "portable_session_geometry",
            "entry_geometry",
            {
                "param_overrides.opening_range_bars": 6,
                "param_overrides.rvol_threshold": 2.0,
                "param_overrides.selection_long_count": 20,
                "ablation.use_orb_entry_range_gate": False,
            },
        ),
        (
            "frequency_rvol_1p5",
            "signal_frequency",
            {"param_overrides.rvol_threshold": 1.5},
        ),
        (
            "frequency_rvol_1p7",
            "signal_frequency",
            {"param_overrides.rvol_threshold": 1.7},
        ),
        (
            "selective_rvol_2p0",
            "signal_selectivity",
            {"param_overrides.rvol_threshold": 2.0},
        ),
        (
            "combined_gate_off",
            "signal_selectivity",
            {
                "ablation.use_combined_quality_gate": False,
                "param_overrides.block_combined_regime_b": False,
            },
        ),
        (
            "failure_stop_off",
            "early_failure",
            {"param_overrides.failure_stop_bars": 0},
        ),
        (
            "mfe_conviction_off",
            "early_failure",
            {"ablation.use_mfe_conviction_exit": False},
        ),
        (
            "flow_reversal_off",
            "exit_management",
            {"ablation.use_flow_reversal_exit": False},
        ),
        (
            "adaptive_trail_off",
            "winner_management",
            {"ablation.use_adaptive_trail": False},
        ),
        (
            "adaptive_trail_relaxed",
            "winner_management",
            {
                "param_overrides.adaptive_trail_late_activate_r": 0.25,
                "param_overrides.adaptive_trail_late_distance_r": 0.12,
            },
        ),
        (
            "daily_stop_off",
            "portfolio_risk",
            {"ablation.use_daily_stop": False},
        ),
    ]
    candidates: dict[str, dict[str, Any]] = {}
    for name, family, changes in rows:
        mutations = dict(base)
        mutations.update(changes)
        sig = _signature(mutations)
        candidates.setdefault(
            sig,
            {
                "id": f"family__{name}",
                "family": family,
                "era": "recovery_counterfactual",
                "mutations": mutations,
                "sources": [baseline["id"]],
            },
        )
    return list(candidates.values())


def _cost_candidates(candidates: Iterable[dict[str, Any]], slip_bps: float) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        mutations = dict(candidate["mutations"])
        mutations["slippage.slip_bps_normal"] = slip_bps
        rows.append(
            {
                **{key: candidate.get(key) for key in ("id", "family", "era", "sources")},
                "id": f"{candidate['id']}__cost_{str(slip_bps).replace('.', 'p')}bps",
                "mutations": mutations,
                "base_signature": _signature(candidate["mutations"]),
            }
        )
    return rows


def _cost_summary(
    candidate: dict[str, Any], cost_results: dict[float, list[dict[str, Any]]]
) -> dict[str, Any]:
    sig = _signature(candidate["mutations"])
    output: dict[str, Any] = {}
    for cost, rows in cost_results.items():
        match = next((row for row in rows if row.get("base_signature") == sig), None)
        if match and not match.get("error"):
            output[str(cost)] = _compact_metrics(match["metrics"])
    seven = output.get("7.5", {})
    ten = output.get("10.0", {})
    output["seven_five_gate"] = bool(
        float(seven.get("expected_total_r", -1.0)) >= 0.0
        and float(seven.get("profit_factor", 0.0)) >= 1.02
    )
    output["ten_gate"] = bool(
        float(ten.get("expected_total_r", -1.0)) >= 0.0
        and float(ten.get("profit_factor", 0.0)) >= 1.00
    )
    cost_penalty = (
        0.10 * min(1.0, max(0.0, -float(seven.get("expected_total_r", -100.0))) / 40.0)
        + 0.08 * min(1.0, max(0.0, -float(ten.get("expected_total_r", -100.0))) / 60.0)
    )
    output["cost_penalty"] = cost_penalty
    return output


def _final_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    validation = record.get("validation", {})
    costs = record.get("costs", {})
    return (
        1.0 if validation.get("robust_eligible") else 0.0,
        1.0 if costs.get("seven_five_gate") else 0.0,
        1.0 if costs.get("ten_gate") else 0.0,
        float(validation.get("validated_score", -99.0)) - float(costs.get("cost_penalty", 1.0)),
        float(record.get("metrics", {}).get("expected_total_r", -1e9)),
    )


def _compact_metrics(metrics: dict[str, float]) -> dict[str, float]:
    avg_r = float(metrics.get("avg_r", metrics.get("expectancy", 0.0)))
    return {
        "total_trades": float(metrics.get("total_trades", 0.0)),
        "trades_per_month": float(metrics.get("trades_per_month", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "avg_r": avg_r,
        "expected_total_r": float(metrics.get("expected_total_r", 0.0)),
        "net_profit": float(metrics.get("net_profit", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
    }


def _report_table(records: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    output = []
    for rank, row in enumerate(records[:limit] if limit else records, 1):
        output.append(
            {
                "rank": rank,
                "id": row["id"],
                "family": row.get("family"),
                "era": row.get("era"),
                "sources": row.get("sources", []),
                "economic_score": row.get("economic_score"),
                "safety_gate": _safety_gate(row.get("metrics", {})),
                "metrics": _compact_metrics(row.get("metrics", {})),
                "validation": row.get("validation", {}),
                "costs": row.get("costs", {}),
            }
        )
    return output


def _render_report(
    manifest: dict[str, Any], lineage: list[dict[str, Any]], families: list[dict[str, Any]]
) -> str:
    selected = manifest["selected"]
    metrics = selected["metrics"]
    validation = selected["validation"]
    costs = selected["costs"]
    lineage_top = lineage[:5]
    family_top = families[:8]
    lines = [
        "# ALCB corrected-RTH baseline recovery",
        "",
        f"Status: **{manifest['status']}**",
        "",
        (
            f"Selected `{selected['id']}`: {metrics['total_trades']:.0f} trades "
            f"({metrics['trades_per_month']:.2f}/month), {metrics['expected_total_r']:+.2f}R, "
            f"${metrics['net_profit']:+,.2f}, PF {metrics['profit_factor']:.3f}, "
            f"AvgR {metrics['avg_r']:+.4f}, DD {metrics['max_drawdown_pct']:.2%}."
        ),
        "",
        "## Decision",
        "",
        (
            f"The baseline is {'eligible' if manifest['research_eligible'] else 'not eligible'} "
            "for the next phased research round under the fixed fold/cost policy. "
            "It is never live-promotable from this cache: an unchanged direct-RTH replay and a fresh lockbox are still required."
        ),
        "",
        "## Immutable score",
        "",
        "The score was fixed before the recovery results were inspected. It is aggressive-leaning: "
        "53% return/throughput, 25% risk-adjusted signal economics, and 22% drawdown/Sharpe. "
        "Chronological stability and execution costs are separate eligibility overlays, so they cannot be traded away inside the optimizer.",
        "",
        "| Component | Weight | Scaling |",
        "|---|---:|---|",
    ]
    for key, spec in SCORE_SPEC.items():
        lines.append(f"| {key} | {float(spec['weight']):.0%} | `{spec['transform']}` |")
    lines.extend([
        "",
        "## Top lineage anchors",
        "",
        "| Rank | Candidate | R | TPM | PF | AvgR | DD | Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(lineage_top, 1):
        m = row["metrics"]
        lines.append(
            f"| {rank} | {row['id']} | {m.get('expected_total_r', 0):+.2f} | "
            f"{m.get('trades_per_month', 0):.2f} | {m.get('profit_factor', 0):.3f} | "
            f"{m.get('avg_r', m.get('expectancy', 0)):+.4f} | "
            f"{m.get('max_drawdown_pct', 0):.2%} | {row.get('economic_score', 0):+.4f} |"
        )
    lines.extend([
        "",
        "## Top family counterfactuals",
        "",
        "| Rank | Candidate | Family | R | TPM | PF | DD | Validated score |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(family_top, 1):
        m = row["metrics"]
        lines.append(
            f"| {rank} | {row['id']} | {row.get('family', '')} | "
            f"{m.get('expected_total_r', 0):+.2f} | {m.get('trades_per_month', 0):.2f} | "
            f"{m.get('profit_factor', 0):.3f} | {m.get('max_drawdown_pct', 0):.2%} | "
            f"{row.get('validation', {}).get('validated_score', -99):+.4f} |"
        )
    lines.extend([
        "",
        "## Selected validation",
        "",
        (
            f"Positive folds: {validation['positive_fold_count']}/4; worst fold AvgR "
            f"{validation['worst_fold_avg_r']:+.4f}; maximum fold DD "
            f"{validation['maximum_fold_drawdown_pct']:.2%}."
        ),
        "",
        "| Cost | R | PF | Net | DD |",
        "|---:|---:|---:|---:|---:|",
    ])
    for cost in ("7.5", "10.0"):
        row = costs.get(cost, {})
        lines.append(
            f"| {cost} bps | {row.get('expected_total_r', 0):+.2f} | "
            f"{row.get('profit_factor', 0):.3f} | ${row.get('net_profit', 0):+,.2f} | "
            f"{row.get('max_drawdown_pct', 0):.2%} |"
        )
    lines.extend([
        "",
        "## Interpretation and next round",
        "",
        "Use the selected config unchanged as the Phase 1 parent. Do not resume the old cumulative Round 4 state. "
        "The next round should first optimize continuous signal discrimination and sizing, then entry geometry/frequency, "
        "then early-failure logic, winner management, and finally portfolio risk. Keep the score and gates in this manifest immutable.",
        "",
        "The period beginning 2026-03-02 was not accessed by this recovery run. It was previously consumed elsewhere, "
        "so it remains supporting evidence only and is not a fresh lockbox.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    if not args.allow_projected_rth_data:
        raise RuntimeError(
            "Direct frozen RTH data are unavailable; pass --allow-projected-rth-data "
            "to produce a provisional diagnostic baseline."
        )
    if args.max_workers < 1 or args.max_workers > MAX_WORKERS:
        raise ValueError(f"Use between 1 and {MAX_WORKERS} workers.")
    if args.end_date >= CONSUMED_START:
        raise ValueError(
            f"Recovery end {args.end_date} overlaps the excluded period beginning {CONSUMED_START}."
        )
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "evaluation_cache.json"
    source = _source_fingerprint()
    code = _code_fingerprint()
    lineage = _load_lineage()

    print("=" * 78)
    print("ALCB CORRECTED-RTH BASELINE RECOVERY")
    print("=" * 78)
    print(f"Lineage/clean anchors: {len(lineage)}")
    print(f"Training only: {args.start_date} -> {args.end_date}")
    print(f"Excluded consumed period begins: {CONSUMED_START}")
    print(f"Workers: {args.max_workers}")
    print(f"Immutable score components: {len(SCORE_SPEC)}", flush=True)

    lineage_full = _evaluate_batch(
        lineage,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source=source,
        code=code,
    )
    errors = [row for row in lineage_full if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} lineage evaluations failed; see errors.json")
    lineage_ranked = sorted(lineage_full, key=_full_rank_key, reverse=True)
    lineage_finalists = lineage_ranked[: max(1, args.top_fold_finalists)]

    lineage_folds: dict[str, list[dict[str, Any]]] = {}
    for name, fold_start, fold_end in FOLDS:
        lineage_folds[name] = _evaluate_batch(
            lineage_finalists,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
    for candidate in lineage_finalists:
        candidate["validation"] = _fold_summary(candidate, lineage_folds)
    lineage_validated = sorted(lineage_finalists, key=_validated_rank_key, reverse=True)
    lineage_winner = lineage_validated[0]

    families = _family_candidates(lineage_winner)
    family_full = _evaluate_batch(
        families,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        cache_path=cache_path,
        source=source,
        code=code,
    )
    errors = [row for row in family_full if row.get("error")]
    if errors:
        _write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"{len(errors)} family evaluations failed; see errors.json")
    family_ranked = sorted(family_full, key=_full_rank_key, reverse=True)
    family_finalists_by_sig: dict[str, dict[str, Any]] = {}
    for candidate in family_ranked[: max(4, args.top_fold_finalists)]:
        family_finalists_by_sig.setdefault(_signature(candidate["mutations"]), candidate)
    family_finalists_by_sig.setdefault(_signature(lineage_winner["mutations"]), lineage_winner)
    family_finalists = list(family_finalists_by_sig.values())

    family_folds: dict[str, list[dict[str, Any]]] = {}
    for name, fold_start, fold_end in FOLDS:
        family_folds[name] = _evaluate_batch(
            family_finalists,
            start_date=fold_start,
            end_date=fold_end,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
    for candidate in family_finalists:
        candidate["validation"] = _fold_summary(candidate, family_folds)
    family_validated = sorted(family_finalists, key=_validated_rank_key, reverse=True)

    cost_finalists = family_validated[: max(1, args.top_cost_finalists)]
    if all(_signature(row["mutations"]) != _signature(lineage_winner["mutations"]) for row in cost_finalists):
        cost_finalists.append(lineage_winner)
    cost_results: dict[float, list[dict[str, Any]]] = {}
    for cost in (7.5, 10.0):
        cost_results[cost] = _evaluate_batch(
            _cost_candidates(cost_finalists, cost),
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=args.max_workers,
            cache_path=cache_path,
            source=source,
            code=code,
        )
    for candidate in cost_finalists:
        candidate["costs"] = _cost_summary(candidate, cost_results)
    final_ranked = sorted(cost_finalists, key=_final_rank_key, reverse=True)
    winner = final_ranked[0]

    research_eligible = bool(
        _safety_gate(winner["metrics"])
        and winner["validation"]["robust_eligible"]
        and winner["costs"]["seven_five_gate"]
    )
    status = (
        "provisional_recovered_baseline_direct_rth_revalidation_required"
        if research_eligible
        else "recovery_incomplete_no_candidate_passed_all_research_gates"
    )
    selected_config = dict(sorted(winner["mutations"].items()))
    _write_json(output_dir / "optimized_config.json", selected_config)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "research_eligible": research_eligible,
        "purpose": "corrected-RTH robust baseline for the next ALCB phased-auto round",
        "data_authority": "derived_legacy_extended_cache_filtered_to_versioned_rth_diagnostic_only",
        "data_source_fingerprint": source,
        "code_fingerprint": code,
        "training_window": {"start": args.start_date, "end": args.end_date},
        "excluded_period": {
            "start": CONSUMED_START,
            "accessed": False,
            "note": "Previously consumed elsewhere; not a fresh holdout.",
        },
        "score_spec": SCORE_SPEC,
        "score_component_count": len(SCORE_SPEC),
        "score_policy": (
            "Fixed absolute economic scaling; chronology and costs are separate "
            "promotion overlays and never additional tunable components."
        ),
        "hard_research_gates": {
            "full_period": {
                "expected_total_r": ">0",
                "net_profit": ">0",
                "trades_per_month": ">=20",
                "profit_factor": ">=1.10",
                "max_drawdown_pct": "<=0.08",
            },
            "folds": {
                "positive_folds": ">=3/4",
                "worst_fold_avg_r": ">=-0.02",
                "maximum_fold_drawdown_pct": "<=0.07",
                "minimum_fold_profit_factor": ">=0.95",
            },
            "cost_7p5bps": {"expected_total_r": ">=0", "profit_factor": ">=1.02"},
            "cost_10bps": {
                "expected_total_r": ">=0",
                "profit_factor": ">=1.00",
                "role": "preferred final robustness gate; reported even if not passed",
            },
        },
        "lineage_count": len(lineage),
        "family_count": len(families),
        "lineage_winner": {
            "id": lineage_winner["id"],
            "sources": lineage_winner.get("sources", []),
            "metrics": _compact_metrics(lineage_winner["metrics"]),
            "economic_score": lineage_winner.get("economic_score"),
            "validation": lineage_winner["validation"],
        },
        "selected": {
            "id": winner["id"],
            "family": winner.get("family"),
            "sources": winner.get("sources", []),
            "signature": _signature(selected_config),
            "metrics": _compact_metrics(winner["metrics"]),
            "economic_score": winner.get("economic_score"),
            "validation": winner["validation"],
            "costs": winner["costs"],
        },
        "promotion_policy": (
            "A family mutation displaces the best historical lineage only through fixed "
            "full-period economics, four chronological folds, and real 7.5/10 bps replays."
        ),
        "next_round_order": [
            "continuous signal discrimination and causal sizing",
            "entry geometry and quality-preserved frequency recovery",
            "early-failure management",
            "winner management and exits",
            "portfolio risk synthesis",
        ],
    }
    _write_json(output_dir / "recovery_manifest.json", manifest)
    _write_json(output_dir / "lineage_inventory.json", lineage)
    _write_json(output_dir / "lineage_ranking.json", _report_table(lineage_ranked))
    _write_json(output_dir / "validated_lineage_finalists.json", _report_table(lineage_validated))
    _write_json(output_dir / "family_ranking.json", _report_table(family_ranked))
    _write_json(output_dir / "validated_family_finalists.json", _report_table(family_validated))
    _write_json(output_dir / "cost_validated_finalists.json", _report_table(final_ranked))
    (output_dir / "recovery_report.md").write_text(
        _render_report(manifest, lineage_ranked, family_validated), encoding="utf-8"
    )

    m = manifest["selected"]["metrics"]
    v = manifest["selected"]["validation"]
    c = manifest["selected"]["costs"]
    summary = [
        "=" * 78,
        "ALCB CORRECTED-RTH BASELINE RECOVERY COMPLETE",
        "=" * 78,
        f"Status: {status}",
        f"Selected: {manifest['selected']['id']} ({manifest['selected']['family']})",
        f"Signature: {manifest['selected']['signature']}",
        (
            f"Trades={m['total_trades']:.0f} TPM={m['trades_per_month']:.2f} "
            f"R={m['expected_total_r']:+.2f} Net=${m['net_profit']:+,.2f} "
            f"AvgR={m['avg_r']:+.4f} PF={m['profit_factor']:.3f} DD={m['max_drawdown_pct']:.2%}"
        ),
        (
            f"Folds positive={v['positive_fold_count']}/4 worst AvgR={v['worst_fold_avg_r']:+.4f} "
            f"max fold DD={v['maximum_fold_drawdown_pct']:.2%}"
        ),
        (
            f"7.5bps R={c.get('7.5', {}).get('expected_total_r', 0):+.2f} "
            f"PF={c.get('7.5', {}).get('profit_factor', 0):.3f}; "
            f"10bps R={c.get('10.0', {}).get('expected_total_r', 0):+.2f} "
            f"PF={c.get('10.0', {}).get('profit_factor', 0):.3f}"
        ),
        f"Config: {output_dir / 'optimized_config.json'}",
        "The post-2026-03-01 period was not accessed.",
    ]
    (output_dir / "recovery_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
