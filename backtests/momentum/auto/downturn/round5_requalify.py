"""Repair-aware Downturn lineage, robustness, phase-auto, and OOS qualification.

Selection uses the archived Round 4 contract: 2024-01-01 through 2026-03-20.
Candidate workers receive a replay bundle truncated before the 2026-03-21 OOS
boundary. The 2026-03-21 through 2026-05-01 holdout is loaded only by final
qualification after an optimized configuration has been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
from backtests.momentum.auto.downturn.phase_candidates import get_phase_candidates
from backtests.momentum.config_downturn import DownturnBacktestConfig
from backtests.momentum.data.preprocessing import NumpyBars

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "backtests/momentum/data/raw"
ARCHIVED_ROUNDS_DIR = (
    ROOT
    / "backtests/output/momentum/downturn/archive/2026-08-08_pre_recovery_reset"
)
ROUND4_DIR = ARCHIVED_ROUNDS_DIR / "round_4"
OUTPUT_DIR = ROOT / "backtests/output/momentum/downturn/round_5_requalification_aligned_split"
IS_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
OOS_CUTOFF = datetime(2026, 3, 21, tzinfo=timezone.utc)
STUDY_END = datetime(2026, 5, 2, tzinfo=timezone.utc)
INDIVIDUAL_STRATEGY_EQUITY = 10_000.0
INITIAL_EQUITY = INDIVIDUAL_STRATEGY_EQUITY
MAX_WORKERS = 3
MIN_PROMOTION_DELTA = 0.002

_WORKER_KWARGS: dict[str, Any] | None = None
_WORKER_DAILY = None
_WORKER_FULL = False
_WORKER_INITIAL_EQUITY = INITIAL_EQUITY
_WORKER_IS_START = IS_START
_WORKER_OOS_CUTOFF = OOS_CUTOFF
_WORKER_STUDY_END = STUDY_END


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(mutations: dict[str, Any]) -> str:
    raw = json.dumps(mutations, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_dt(value: Any) -> datetime:
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype("int64")
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _slice_bars(bars: NumpyBars, cutoff: datetime) -> NumpyBars:
    cutoff64 = np.datetime64(cutoff.replace(tzinfo=None))
    times = np.asarray(bars.times)
    mask = times < cutoff64
    return NumpyBars(
        opens=bars.opens[mask],
        highs=bars.highs[mask],
        lows=bars.lows[mask],
        closes=bars.closes[mask],
        volumes=bars.volumes[mask],
        times=bars.times[mask],
    )


def _selection_kwargs(bundle, cutoff: datetime) -> dict[str, Any]:
    data = dict(bundle.data)
    five = _slice_bars(data["five_min"], cutoff)
    count = len(five)
    data["five_min"] = five
    for key in list(data):
        if key.endswith("_idx_map") and data[key] is not None:
            data[key] = data[key][:count]
    for key in ("daily", "daily_es"):
        if data.get(key) is not None:
            data[key] = _slice_bars(data[key], cutoff)
    return data


def _init_worker(
    data_dir: str,
    initial_equity: float,
    is_start_iso: str,
    cutoff_iso: str,
    study_end_iso: str,
    full_data: bool,
) -> None:
    global _WORKER_KWARGS, _WORKER_DAILY, _WORKER_FULL, _WORKER_INITIAL_EQUITY
    global _WORKER_IS_START, _WORKER_OOS_CUTOFF, _WORKER_STUDY_END
    from backtests.momentum.data.replay_cache import load_replay_bundle

    is_start = datetime.fromisoformat(is_start_iso)
    cutoff = datetime.fromisoformat(cutoff_iso)
    study_end = datetime.fromisoformat(study_end_iso)
    bundle = load_replay_bundle(
        "NQ",
        Path(data_dir),
        include_fifteen_min=True,
        include_thirty_min=True,
        include_hourly=True,
        include_four_hour=True,
        include_daily=True,
        include_daily_es=True,
    )
    _WORKER_FULL = full_data
    _WORKER_INITIAL_EQUITY = initial_equity
    _WORKER_IS_START = is_start
    _WORKER_OOS_CUTOFF = cutoff
    _WORKER_STUDY_END = study_end
    if full_data:
        _WORKER_KWARGS = _selection_kwargs(bundle, study_end)
    else:
        _WORKER_KWARGS = _selection_kwargs(bundle, cutoff)
    _WORKER_DAILY = _WORKER_KWARGS["daily"]


def _trade_time(trade: Any) -> datetime:
    return _normalize_dt(trade.entry_time)


def _window_metrics(trades: Iterable[Any], initial_equity: float) -> dict[str, Any]:
    ordered = sorted(list(trades), key=_trade_time)
    pnls = [float(trade.pnl) for trade in ordered]
    rs = [float(trade.r_multiple) for trade in ordered]
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    net_pnl = sum(pnls)
    net_return_pct = net_pnl / initial_equity * 100.0
    if len(ordered) >= 2:
        span_days = max((_trade_time(ordered[-1]) - _trade_time(ordered[0])).days, 30)
    else:
        span_days = 365
    annual_return_pct = net_return_pct * 365.25 / span_days
    calmar = annual_return_pct / (max_dd * 100.0) if max_dd > 0 else 99.0
    correction_pnl = sum(
        float(trade.pnl) for trade in ordered if bool(trade.in_correction_window)
    )
    return {
        "total_trades": len(ordered),
        "net_pnl": net_pnl,
        "net_return_pct": net_return_pct,
        "profit_factor": profit_factor,
        "max_dd_pct": max_dd,
        "calmar": calmar,
        "win_rate": (sum(value > 0 for value in pnls) / len(pnls) * 100.0) if pnls else 0.0,
        "net_r": sum(rs),
        "avg_r": statistics.fmean(rs) if rs else 0.0,
        "correction_pnl_pct": correction_pnl / initial_equity * 100.0,
        "correction_trade_share": (
            sum(bool(trade.in_correction_window) for trade in ordered) / len(ordered)
            if ordered
            else 0.0
        ),
    }


def _fold_metrics(trades: list[Any], initial_equity: float) -> list[dict[str, Any]]:
    boundaries = [
        IS_START,
        datetime(2024, 10, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        OOS_CUTOFF,
    ]
    output: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        selected = [trade for trade in trades if start <= _trade_time(trade) < end]
        output.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                **_window_metrics(selected, initial_equity),
            }
        )
    return output


def _robust_score(
    metrics: dict[str, Any],
    folds: list[dict[str, Any]],
    complexity: int,
) -> tuple[float, str]:
    if metrics["total_trades"] < 60:
        return -99.0, "too_few_trades"
    if metrics["profit_factor"] < 1.10:
        return -99.0, "profit_factor_below_1.10"
    if metrics["max_dd_pct"] > 0.25:
        return -99.0, "drawdown_above_25pct"
    if metrics["correction_pnl_pct"] < 0:
        return -99.0, "negative_correction_pnl"

    active_folds = [fold for fold in folds if fold["total_trades"] >= 8]
    positive_fold_share = (
        sum(fold["net_pnl"] > 0 for fold in active_folds) / len(active_folds)
        if active_folds
        else 0.0
    )
    fold_returns = [fold["net_return_pct"] for fold in active_folds]
    fold_dispersion = statistics.pstdev(fold_returns) if len(fold_returns) > 1 else 0.0
    pf = min(float(metrics["profit_factor"]), 4.0)
    score = (
        0.26 * math.tanh(float(metrics["net_return_pct"]) / 80.0)
        + 0.24 * math.tanh((pf - 1.0) / 1.2)
        + 0.14 * math.tanh(float(metrics["calmar"]) / 4.0)
        + 0.12 * math.tanh(float(metrics["net_r"]) / 15.0)
        + 0.10 * math.tanh(float(metrics["correction_pnl_pct"]) / 60.0)
        + 0.10 * positive_fold_share
        - 0.03 * math.tanh(fold_dispersion / 30.0)
        - 0.001 * complexity
    )
    return score, ""


def _evaluate_worker(task: tuple[str, dict[str, Any], bool]) -> dict[str, Any]:
    name, mutations, detail = task
    from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics
    from backtests.momentum.engine.downturn_engine import DownturnEngine

    assert _WORKER_KWARGS is not None
    config = mutate_downturn_config(
        DownturnBacktestConfig(
            initial_equity=_WORKER_INITIAL_EQUITY,
            data_dir=DATA_DIR,
            track_signals=detail,
            skip_parity_output=True,
            max_dd_abort=0.50,
        ),
        mutations,
    )
    started = time.time()
    engine = DownturnEngine("NQ", config)
    result = engine.run(**_WORKER_KWARGS)
    selected_trades = [
        trade
        for trade in result.trades
        if _WORKER_IS_START <= _trade_time(trade) < (
            _WORKER_STUDY_END if _WORKER_FULL else _WORKER_OOS_CUTOFF
        )
    ]
    if _WORKER_FULL:
        is_trades = [
            trade for trade in selected_trades
            if _trade_time(trade) < _WORKER_OOS_CUTOFF
        ]
        oos_trades = [
            trade for trade in selected_trades
            if _trade_time(trade) >= _WORKER_OOS_CUTOFF
        ]
        selection_metrics = _window_metrics(is_trades, _WORKER_INITIAL_EQUITY)
    else:
        is_trades = selected_trades
        oos_trades = []
        selection_metrics = _window_metrics(selected_trades, _WORKER_INITIAL_EQUITY)
    folds = _fold_metrics(is_trades, _WORKER_INITIAL_EQUITY)
    score, reject_reason = _robust_score(selection_metrics, folds, len(mutations))
    payload: dict[str, Any] = {
        "name": name,
        "signature": _signature(mutations),
        "mutations": mutations,
        "score": score,
        "rejected": bool(reject_reason),
        "reject_reason": reject_reason,
        "selection_metrics": selection_metrics,
        "folds": folds,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if _WORKER_FULL:
        full_window = _window_metrics(selected_trades, _WORKER_INITIAL_EQUITY)
        full_metrics = asdict(compute_downturn_metrics(result, _WORKER_DAILY))
        full_metrics.update(full_window)
        payload["full_metrics"] = full_metrics
        payload["oos_metrics"] = _window_metrics(oos_trades, _WORKER_INITIAL_EQUITY)
        payload["terminal_working_entries"] = len(engine._core_state.working_entries)
        payload["terminal_broker_entries"] = sum(
            order.tag == "entry" for order in engine.broker.pending_orders
        )
        if detail:
            payload["trades"] = selected_trades
    return payload


def _evaluate_batch(
    candidates: list[tuple[str, dict[str, Any]]],
    *,
    full_data: bool = False,
    detail_names: set[str] | None = None,
    max_workers: int = MAX_WORKERS,
) -> list[dict[str, Any]]:
    unique: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, mutations in candidates:
        unique.setdefault(_signature(mutations), (name, mutations))
    tasks = [
        (name, mutations, bool(detail_names and name in detail_names))
        for name, mutations in unique.values()
    ]
    output: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(
            str(DATA_DIR),
            INITIAL_EQUITY,
            IS_START.isoformat(),
            OOS_CUTOFF.isoformat(),
            STUDY_END.isoformat(),
            full_data,
        ),
    ) as pool:
        futures = {pool.submit(_evaluate_worker, task): task[0] for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "name": name,
                    "score": -99.0,
                    "rejected": True,
                    "reject_reason": f"error:{exc}",
                }
            output.append(row)
            print(
                f"[{completed}/{len(tasks)}] {name}: score={row.get('score', -99):+.4f} "
                f"PF={row.get('selection_metrics', {}).get('profit_factor', 0):.2f} "
                f"ret={row.get('selection_metrics', {}).get('net_return_pct', 0):+.1f}%",
                flush=True,
            )
    return sorted(output, key=_selection_key, reverse=True)


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = row.get("selection_metrics", {})
    return (
        float(row.get("score", -99.0)),
        float(metrics.get("profit_factor", 0.0)),
        float(metrics.get("net_return_pct", -999.0)),
        -float(metrics.get("max_dd_pct", 1.0)),
        -len(row.get("mutations", {})),
    )


def _round_mutations() -> dict[str, dict[str, Any]]:
    manifest = _load_json(ROUND4_DIR.parent / "rounds_manifest.json")
    output = {"default": {}}
    for row in manifest["rounds"]:
        output[f"round_{row['round']}"] = dict(row["mutations"])
    return output


def _round4() -> dict[str, Any]:
    return dict(_load_json(ROUND4_DIR / "optimized_config.json"))


def _best_viable(rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [row for row in rows if not row.get("rejected")]
    if not viable:
        raise RuntimeError("No viable Downturn candidate remained")
    return max(viable, key=_selection_key)


def stage_lineage() -> None:
    rows = _evaluate_batch(list(_round_mutations().items()))
    _write_json(
        OUTPUT_DIR / "lineage_replay.json",
        {"results": rows, "best": _best_viable(rows)},
    )


def stage_light() -> None:
    base = _round4()
    candidates: list[tuple[str, dict[str, Any]]] = [("round4_repaired", base)]
    for ttl in (12, 24, 48, 72, 96, 144):
        candidates.append((f"ttl_{ttl}", {**base, "param_overrides.entry_ttl_bars": ttl}))
    for value in (0, 1, 2, 3):
        candidates.append((f"entry_buffer_{value}", {**base, "param_overrides.entry_buffer_ticks": value}))
    for value in (1, 2, 3):
        candidates.append((f"trigger_buffer_{value}", {**base, "param_overrides.trigger_low_buffer_ticks": value}))
    for value in (2, 4, 6, 8):
        candidates.append((f"limit_offset_{value}", {**base, "param_overrides.entry_limit_offset_ticks": value}))
    candidates.append(("cancel_replace", {**base, "flags.cancel_replace_entry": True}))

    lineage = _load_best("lineage_replay.json")
    lineage_base = dict(lineage["mutations"])
    if _signature(lineage_base) != _signature(base):
        candidates.append((f"{lineage['name']}_repaired", lineage_base))
        for ttl in (24, 48, 72, 96):
            candidates.append(
                (f"{lineage['name']}_ttl_{ttl}", {**lineage_base, "param_overrides.entry_ttl_bars": ttl})
            )
        for value in (0, 1, 2):
            candidates.append(
                (f"{lineage['name']}_entry_buffer_{value}", {**lineage_base, "param_overrides.entry_buffer_ticks": value})
            )
        for value in (2, 4, 6):
            candidates.append(
                (f"{lineage['name']}_limit_offset_{value}", {**lineage_base, "param_overrides.entry_limit_offset_ticks": value})
            )
        candidates.append(
            (f"{lineage['name']}_cancel_replace", {**lineage_base, "flags.cancel_replace_entry": True})
        )
    rows = _evaluate_batch(candidates)
    _write_json(
        OUTPUT_DIR / "light_recovery.json",
        {"results": rows, "best": _best_viable(rows)},
    )


def _ablation_groups() -> dict[str, tuple[str, ...]]:
    return {
        "regime": ("regime", "adx", "ema", "drawdown", "progressive", "counter"),
        "signal": ("engine", "momentum", "divergence", "vwap_cap"),
        "entry": ("entry", "friction", "vol_percentile"),
        "exit": ("chandelier", "profit_floor", "be_", "hold", "tp1"),
        "risk": ("risk_pct", "sizing", "mult_"),
    }


def stage_ablation() -> None:
    base = _round4()
    historical = _round_mutations()
    candidates: list[tuple[str, dict[str, Any]]] = [("round4_repaired", base)]
    for key in sorted(base):
        reverted = dict(base)
        reverted.pop(key)
        candidates.append((f"ablate:{key}", reverted))
        prior_values = []
        for mutations in historical.values():
            if key in mutations and mutations[key] != base[key] and mutations[key] not in prior_values:
                prior_values.append(mutations[key])
        for value in prior_values:
            candidates.append((f"rollback:{key}:{value}", {**base, key: value}))
    for group, patterns in _ablation_groups().items():
        reverted = {
            key: value
            for key, value in base.items()
            if not any(pattern in key for pattern in patterns)
        }
        candidates.append((f"group_ablate:{group}", reverted))

    first_pass = _evaluate_batch(candidates)
    singles = [row for row in first_pass if row["name"].startswith("ablate:") and not row.get("rejected")]
    top_singles = singles[:6]
    pair_candidates: list[tuple[str, dict[str, Any]]] = []
    for left_idx, left in enumerate(top_singles):
        left_key = left["name"].split(":", 1)[1]
        for right in top_singles[left_idx + 1 :]:
            right_key = right["name"].split(":", 1)[1]
            reverted = dict(base)
            reverted.pop(left_key, None)
            reverted.pop(right_key, None)
            pair_candidates.append((f"pair_ablate:{left_key}|{right_key}", reverted))
    pairs = _evaluate_batch(pair_candidates) if pair_candidates else []
    rows = sorted([*first_pass, *pairs], key=_selection_key, reverse=True)
    _write_json(
        OUTPUT_DIR / "ablation.json",
        {"results": rows, "best": _best_viable(rows)},
    )


def _load_best(path: str) -> dict[str, Any]:
    payload = _load_json(OUTPUT_DIR / path)
    return dict(payload.get("best") or _best_viable(payload["results"]))


def stage_perturbation() -> None:
    light = _load_best("light_recovery.json")
    ablation = _load_best("ablation.json")
    lineage = _load_best("lineage_replay.json")
    seed_row = max((lineage, light, ablation), key=_selection_key)
    seed = dict(seed_row["mutations"])
    ablation_rows = _load_json(OUTPUT_DIR / "ablation.json")["results"]
    sensitive_keys = []
    for row in ablation_rows:
        if row["name"].startswith("ablate:"):
            key = row["name"].split(":", 1)[1]
            if key in seed and isinstance(seed[key], (int, float)) and not isinstance(seed[key], bool):
                sensitive_keys.append(key)
        if len(sensitive_keys) >= 8:
            break
    for key in (
        "param_overrides.entry_ttl_bars",
        "param_overrides.entry_buffer_ticks",
        "param_overrides.entry_limit_offset_ticks",
    ):
        if key not in sensitive_keys:
            sensitive_keys.append(key)

    candidates: list[tuple[str, dict[str, Any]]] = [("robust_seed", seed)]
    defaults = {
        "param_overrides.entry_ttl_bars": 72.0,
        "param_overrides.entry_buffer_ticks": 2.0,
        "param_overrides.entry_limit_offset_ticks": 4.0,
    }
    for key in sensitive_keys:
        value = seed.get(key, defaults.get(key))
        if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        for label, factor in (("m20", 0.8), ("m10", 0.9), ("p10", 1.1), ("p20", 1.2)):
            varied = value * factor
            if isinstance(value, int) or key.endswith(("_bars", "_lookback")):
                varied = max(1, int(round(varied)))
            else:
                varied = round(float(varied), 6)
            candidates.append((f"perturb:{key}:{label}", {**seed, key: varied}))
    first_pass = _evaluate_batch(candidates)
    top = [row for row in first_pass if row["name"].startswith("perturb:") and not row.get("rejected")][:5]
    pair_candidates: list[tuple[str, dict[str, Any]]] = []
    for left_idx, left in enumerate(top):
        for right in top[left_idx + 1 :]:
            combined = dict(seed)
            for source in (left["mutations"], right["mutations"]):
                for key, value in source.items():
                    if seed.get(key) != value:
                        combined[key] = value
            if combined != left["mutations"] and combined != right["mutations"]:
                pair_candidates.append((f"interaction:{left['name']}+{right['name']}", combined))
    interactions = _evaluate_batch(pair_candidates) if pair_candidates else []
    rows = sorted([*first_pass, *interactions], key=_selection_key, reverse=True)
    _write_json(
        OUTPUT_DIR / "perturbation.json",
        {
            "seed": seed_row,
            "sensitive_keys": sensitive_keys,
            "results": rows,
            "best": _best_viable(rows),
        },
    )


def _overlay(base: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    return {**base, **mutation}


def stage_phased_auto() -> None:
    current = dict(_load_best("perturbation.json")["mutations"])
    current_eval = _evaluate_batch([("phase_seed", current)])[0]
    phase_records = []
    for phase in (1, 2, 3):
        specs = get_phase_candidates(phase, current)
        candidates = [(name, _overlay(current, mutation)) for name, mutation in specs]
        first_pass = _evaluate_batch(candidates)
        best = _best_viable([current_eval, *first_pass])
        accepted = best["signature"] != current_eval["signature"] and best["score"] > current_eval["score"] + MIN_PROMOTION_DELTA
        accepted_names: list[str] = []
        if accepted:
            current = dict(best["mutations"])
            current_eval = best
            accepted_names.append(best["name"])

            remaining = [
                (name, _overlay(current, mutation))
                for name, mutation in specs
                if name != best["name"]
            ]
            # Re-test the strongest first-pass neighbourhood after the first
            # accepted mutation so interactions are selected causally.
            strongest_names = {row["name"] for row in first_pass[:12]}
            second_candidates = [item for item in remaining if item[0] in strongest_names]
            second_pass = _evaluate_batch(second_candidates) if second_candidates else []
            second_best = _best_viable([current_eval, *second_pass])
            if (
                second_best["signature"] != current_eval["signature"]
                and second_best["score"] > current_eval["score"] + MIN_PROMOTION_DELTA
            ):
                current = dict(second_best["mutations"])
                current_eval = second_best
                accepted_names.append(second_best["name"])
        else:
            second_pass = []
        record = {
            "phase": phase,
            "accepted": accepted_names,
            "current": current_eval,
            "first_pass": first_pass,
            "second_pass": second_pass,
        }
        phase_records.append(record)
        _write_json(OUTPUT_DIR / f"phase_{phase}.json", record)
    _write_json(OUTPUT_DIR / "optimized_config.json", current)
    _write_json(
        OUTPUT_DIR / "phase_auto_summary.json",
        {"phases": phase_records, "final": current_eval, "optimized_config": current},
    )


def _bootstrap_oos(trades: list[Any], samples: int = 5_000) -> dict[str, Any]:
    oos = [
        trade for trade in trades
        if OOS_CUTOFF <= _trade_time(trade) < STUDY_END
    ]
    if not oos:
        return {"samples": samples, "trades": 0, "net_r_ci95": [0.0, 0.0]}
    values = np.asarray([float(trade.r_multiple) for trade in oos], dtype=float)
    rng = np.random.default_rng(20260808)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).sum(axis=1)
    return {
        "samples": samples,
        "trades": len(oos),
        "net_r_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive_net_r": float(np.mean(draws > 0)),
    }


def _promotion_gate(row: dict[str, Any]) -> dict[str, Any]:
    full = row["full_metrics"]
    oos = row["oos_metrics"]
    criteria = {
        "full_return_ge_55": float(full["net_return_pct"]) >= 55.0,
        "full_pf_ge_1_70": float(full["profit_factor"]) >= 1.70,
        "full_dd_le_15pct": float(full["max_dd_pct"]) <= 0.15,
        "full_calmar_ge_1_50": float(full["calmar"]) >= 1.50,
        "correction_pnl_ge_45": float(full["correction_pnl_pct"]) >= 45.0,
        "trades_ge_90": int(full["total_trades"]) >= 90,
        "oos_trades_ge_15": int(oos["total_trades"]) >= 15,
        "oos_net_positive": float(oos["net_pnl"]) > 0,
        "oos_pf_ge_1_20": float(oos["profit_factor"]) >= 1.20,
        "no_terminal_entry": not row["terminal_working_entries"] and not row["terminal_broker_entries"],
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def _portfolio_validation(selected_trades: list[Any]) -> dict[str, Any]:
    from backtests.momentum.auto.portfolio_synergy.family_phase_auto import (
        evaluate_portfolio_config,
        load_or_build_latest_strategy_trades,
    )
    from backtests.momentum.engine.family_portfolio_engine import (
        build_family_replay_bundle,
        family_config_from_dict,
    )

    portfolio_dir = OUTPUT_DIR / "portfolio_cache"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    trades_by_strategy = load_or_build_latest_strategy_trades(
        data_dir=DATA_DIR,
        output_dir=portfolio_dir,
        initial_equity=INDIVIDUAL_STRATEGY_EQUITY,
        force=True,
    )
    config = family_config_from_dict(
        _load_json(ROOT / "backtests/output/momentum/portfolio_synergy/round_2/optimized_portfolio_config.json")
    )

    baseline = evaluate_portfolio_config(
        "repaired_round4",
        config,
        build_family_replay_bundle(trades_by_strategy),
    )
    selected_streams = dict(trades_by_strategy)
    selected_streams["DownturnDominator_v1"] = selected_trades
    selected = evaluate_portfolio_config(
        "round5_selected",
        config,
        build_family_replay_bundle(selected_streams),
    )
    without_streams = dict(trades_by_strategy)
    without_streams["DownturnDominator_v1"] = []
    without = evaluate_portfolio_config(
        "without_downturn",
        config,
        build_family_replay_bundle(without_streams),
    )
    return {
        "repaired_round4": _portfolio_row(baseline),
        "round5_selected": _portfolio_row(selected),
        "without_downturn": _portfolio_row(without),
    }


def _portfolio_row(evaluation: Any) -> dict[str, Any]:
    return {
        "score": evaluation.score,
        "rejected": evaluation.rejected,
        "reject_reason": evaluation.reject_reason,
        "metrics": evaluation.metrics,
    }


def stage_final() -> None:
    selected = dict(_load_json(OUTPUT_DIR / "optimized_config.json"))
    round4 = _round4()
    stress_candidates = [
        ("round4_repaired", round4),
        ("round5_selected", selected),
        ("stress_commission_1_5x", {**selected, "slippage.commission_per_contract": 0.93}),
        ("stress_commission_2x", {**selected, "slippage.commission_per_contract": 1.24}),
        ("stress_slippage_2ticks", {**selected, "slippage.slip_ticks_normal": 2}),
        ("stress_spread_1bp", {**selected, "slippage.spread_bps": 1.0}),
        ("stress_entry_latency_1bar", {**selected, "entry_latency_bars": 1}),
        (
            "stress_combined",
            {
                **selected,
                "slippage.commission_per_contract": 1.24,
                "slippage.slip_ticks_normal": 2,
                "slippage.spread_bps": 1.0,
                "entry_latency_bars": 1,
            },
        ),
    ]
    rows = _evaluate_batch(
        stress_candidates,
        full_data=True,
        detail_names={"round5_selected"},
    )
    selected_row = next(row for row in rows if row["name"] == "round5_selected")
    selected_trades = selected_row.pop("trades")
    bootstrap = _bootstrap_oos(selected_trades)
    gate = _promotion_gate(selected_row)
    pre_portfolio = {
        "selection_cutoff": OOS_CUTOFF.isoformat(),
        "selected": selected_row,
        "stress_results": rows,
        "oos_bootstrap": bootstrap,
        "promotion_gate": gate,
        "disposition": "PROMOTE" if gate["passed"] else "SHADOW_ONLY",
    }
    _write_json(OUTPUT_DIR / "final_stress_oos.json", pre_portfolio)
    portfolio = _portfolio_validation(selected_trades)
    summary = {
        **pre_portfolio,
        "portfolio_validation": portfolio,
    }
    _write_json(OUTPUT_DIR / "final_qualification.json", summary)
    print(json.dumps({"disposition": summary["disposition"], "gate": gate}, indent=2))


def write_run_spec() -> None:
    from backtests.momentum.auto.downturn.plugin import DownturnPlugin

    provenance = DownturnPlugin(DATA_DIR, initial_equity=INITIAL_EQUITY, max_workers=1).build_provenance()
    payload = {
        "strategy": "downturn",
        "round": 5,
        "purpose": "lifecycle repair and validation-aware requalification",
        "is_start": IS_START.isoformat(),
        "selection_cutoff": OOS_CUTOFF.isoformat(),
        "study_end_exclusive": STUDY_END.isoformat(),
        "oos_policy": "candidate workers receive pre-cutoff market arrays only",
        "initial_equity": INITIAL_EQUITY,
        "max_workers": MAX_WORKERS,
        "source_data": {
            "nq_manifest": _load_json(DATA_DIR / "NQ_5m.manifest.json"),
            "es_manifest": _load_json(DATA_DIR / "ES_1d.manifest.json"),
        },
        "provenance": asdict(provenance),
    }
    _write_json(OUTPUT_DIR / "run_spec.json", payload)


def main() -> None:
    global INITIAL_EQUITY, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("lineage", "light", "ablation", "perturbation", "phased", "final", "all"),
        default="all",
    )
    parser.add_argument("--initial-equity", type=float, default=INITIAL_EQUITY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.initial_equity <= 0:
        parser.error("--initial-equity must be positive")
    if not math.isclose(args.initial_equity, INDIVIDUAL_STRATEGY_EQUITY):
        parser.error(
            "Downturn is an individual-strategy run and must use the canonical "
            f"${INDIVIDUAL_STRATEGY_EQUITY:,.0f} equity basis"
        )
    INITIAL_EQUITY = float(args.initial_equity)
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_run_spec()
    stages = {
        "lineage": stage_lineage,
        "light": stage_light,
        "ablation": stage_ablation,
        "perturbation": stage_perturbation,
        "phased": stage_phased_auto,
        "final": stage_final,
    }
    selected = list(stages) if args.stage == "all" else [args.stage]
    for name in selected:
        print(f"=== {name.upper()} ===", flush=True)
        stages[name]()


if __name__ == "__main__":
    main()
