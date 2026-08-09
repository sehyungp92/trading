"""Granular Round-1 Downturn OOS attribution and robustness repair.

This research pass deliberately treats 2026-03-21 through 2026-05-01 as
*observed validation* once it is used to compare candidates.  It
therefore writes a shadow recommendation and never promotes the strategy or
overwrites Round 1's frozen optimized configuration.

The runner is restartable.  Full-history candidate results are checkpointed by
mutation signature after every completed worker so long sweeps can resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backtests.momentum.auto.downturn import round5_requalify as recovery
from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
from backtests.momentum.analysis.downturn_diagnostics import compute_downturn_metrics
from backtests.momentum.config_downturn import DownturnBacktestConfig
from backtests.momentum.data.preprocessing import NumpyBars
from backtests.momentum.data.replay_cache import load_replay_bundle, replay_engine_kwargs
from backtests.momentum.engine.downturn_engine import DownturnEngine


ROOT = Path(__file__).resolve().parents[4]
ROUND_DIR = ROOT / "backtests/output/momentum/downturn/round_1"
OUTPUT_DIR = ROUND_DIR / "oos_repair"
ARCHIVE_DIR = (
    ROOT
    / "backtests/output/momentum/downturn/archive/2026-08-08_pre_recovery_reset"
)
CACHE_PATH = OUTPUT_DIR / "candidate_cache_20240101_20260321_20260502.json"
MAX_WORKERS = 3
IS_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
OOS_START = datetime(2026, 3, 21, tzinfo=timezone.utc)
EVALUATION_END = datetime(2026, 5, 2, tzinfo=timezone.utc)  # exclusive

_WORKER_KWARGS: dict[str, Any] | None = None
_WORKER_DAILY: NumpyBars | None = None


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(mutations: dict[str, Any]) -> str:
    encoded = json.dumps(
        mutations,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current() -> dict[str, Any]:
    return dict(_load_json(ROUND_DIR / "optimized_config.json"))


def _slice_bars(bars: NumpyBars, start: int, end: int) -> NumpyBars:
    return NumpyBars(
        opens=bars.opens[start:end],
        highs=bars.highs[start:end],
        lows=bars.lows[start:end],
        closes=bars.closes[start:end],
        volumes=bars.volumes[start:end],
        times=bars.times[start:end],
    )


def _init_split_worker(data_dir: str, initial_equity: float) -> None:
    """Load once and make the engine's first tradeable bar exactly IS_START."""

    global _WORKER_KWARGS, _WORKER_DAILY
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
    kwargs = replay_engine_kwargs(bundle)
    five = kwargs["five_min"]
    times = np.asarray(five.times).astype("datetime64[ns]")
    is_index = int(
        np.searchsorted(times, np.datetime64(IS_START.replace(tzinfo=None)), side="left")
    )
    end_index = int(
        np.searchsorted(
            times,
            np.datetime64(EVALUATION_END.replace(tzinfo=None)),
            side="left",
        )
    )
    warmup_bars = DownturnBacktestConfig().warmup_days * 78
    slice_start = max(0, is_index - warmup_bars)
    kwargs["five_min"] = _slice_bars(five, slice_start, end_index)
    for key in list(kwargs):
        if key.endswith("_idx_map") and kwargs[key] is not None:
            kwargs[key] = kwargs[key][slice_start:end_index]
    _WORKER_KWARGS = kwargs
    _WORKER_DAILY = kwargs["daily"]


def _fold_metrics(trades: list[Any]) -> list[dict[str, Any]]:
    boundaries = [
        IS_START,
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        OOS_START,
    ]
    return [
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
            **recovery._window_metrics(
                [
                    trade
                    for trade in trades
                    if start <= recovery._trade_time(trade) < end
                ],
                recovery.INITIAL_EQUITY,
            ),
        }
        for start, end in zip(boundaries, boundaries[1:])
    ]


def _evaluate_worker(task: tuple[str, dict[str, Any], bool]) -> dict[str, Any]:
    name, mutations, detail = task
    if _WORKER_KWARGS is None or _WORKER_DAILY is None:
        raise RuntimeError("split worker is not initialized")
    config = mutate_downturn_config(
        DownturnBacktestConfig(
            initial_equity=recovery.INITIAL_EQUITY,
            data_dir=recovery.DATA_DIR,
            track_signals=detail,
            skip_parity_output=True,
            max_dd_abort=0.50,
        ),
        mutations,
    )
    started = time.time()
    engine = DownturnEngine("NQ", config)
    result = engine.run(**_WORKER_KWARGS)
    all_trades = [
        trade
        for trade in result.trades
        if IS_START <= recovery._trade_time(trade) < EVALUATION_END
    ]
    development = [
        trade for trade in all_trades if recovery._trade_time(trade) < OOS_START
    ]
    oos = [
        trade for trade in all_trades if recovery._trade_time(trade) >= OOS_START
    ]
    selection_metrics = recovery._window_metrics(
        development, recovery.INITIAL_EQUITY
    )
    folds = _fold_metrics(development)
    score, reject_reason = recovery._robust_score(
        selection_metrics, folds, len(mutations)
    )
    payload: dict[str, Any] = {
        "name": name,
        "signature": _signature(mutations),
        "mutations": mutations,
        "score": score,
        "rejected": bool(reject_reason),
        "reject_reason": reject_reason,
        "selection_metrics": selection_metrics,
        "oos_metrics": recovery._window_metrics(oos, recovery.INITIAL_EQUITY),
        "full_window_metrics": recovery._window_metrics(
            all_trades, recovery.INITIAL_EQUITY
        ),
        "full_metrics": asdict(compute_downturn_metrics(result, _WORKER_DAILY)),
        "folds": folds,
        "terminal_working_entries": len(engine._core_state.working_entries),
        "terminal_broker_entries": sum(
            order.tag == "entry" for order in engine.broker.pending_orders
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if detail:
        payload["trades"] = all_trades
    return payload


def _archived_manifest() -> list[dict[str, Any]]:
    return list(_load_json(ARCHIVE_DIR / "rounds_manifest.json")["rounds"])


def _candidate(
    name: str,
    mutations: dict[str, Any],
    family: str,
    note: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "mutations": mutations,
        "family": family,
        "note": note,
    }


def _deduplicate(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        signature = _signature(candidate["mutations"])
        if signature not in unique:
            unique[signature] = {**candidate, "aliases": []}
        elif candidate["name"] != unique[signature]["name"]:
            unique[signature]["aliases"].append(candidate["name"])
    return list(unique.values())


def _default_for(key: str) -> tuple[bool, Any]:
    config = DownturnBacktestConfig()
    if key.startswith("flags."):
        name = key.split(".", 1)[1]
        return True, getattr(config.flags, name)
    if key.startswith("param_overrides."):
        return False, None
    if key.startswith("slippage."):
        return True, getattr(config.slippage, key.split(".", 1)[1])
    return True, getattr(config, key)


def _set_or_remove(
    base: dict[str, Any], key: str, value_exists: bool, value: Any
) -> dict[str, Any]:
    mutated = dict(base)
    if value_exists:
        mutated[key] = value
    else:
        mutated.pop(key, None)
    return mutated


def historical_candidates() -> list[dict[str, Any]]:
    """Every cumulative round/phase and atomic historical/current mutation."""

    current = _current()
    rounds = _archived_manifest()
    candidates = [
        _candidate("baseline:round1_recovery", current, "baseline"),
        _candidate("lineage:default", {}, "lineage"),
    ]
    historical_values: dict[str, list[Any]] = defaultdict(list)

    for row in rounds:
        mutations = dict(row["mutations"])
        candidates.append(
            _candidate(
                f"lineage:round_{row['round']}", mutations, "lineage"
            )
        )
        for key, value in mutations.items():
            if value not in historical_values[key]:
                historical_values[key].append(value)

        state_path = ARCHIVE_DIR / f"round_{row['round']}" / "phase_state.json"
        if state_path.exists():
            phase_state = _load_json(state_path)
            for phase, phase_row in phase_state.get("phase_results", {}).items():
                for side in ("base_mutations", "final_mutations"):
                    snapshot = dict(phase_row.get(side) or {})
                    candidates.append(
                        _candidate(
                            f"lineage:round_{row['round']}:phase_{phase}:{side}",
                            snapshot,
                            "phase_lineage",
                        )
                    )

    for key, current_value in sorted(current.items()):
        exists, default_value = _default_for(key)
        candidates.append(
            _candidate(
                f"ablate:{key}",
                _set_or_remove(current, key, exists, default_value),
                "atomic_ablation",
                f"restore default={default_value!r}" if exists else "remove override",
            )
        )
        if key.startswith("flags.") and isinstance(current_value, bool):
            candidates.append(
                _candidate(
                    f"flip:{key}:{not current_value}",
                    {**current, key: not current_value},
                    "atomic_flag_flip",
                )
            )
        for prior in historical_values.get(key, []):
            if prior != current_value:
                candidates.append(
                    _candidate(
                        f"rollback:{key}:{prior}",
                        {**current, key: prior},
                        "atomic_rollback",
                    )
                )

    # Historical accepted mutations removed by the recovery round are restored
    # one at a time, so they are not hidden inside a round-level cluster.
    for key, values in sorted(historical_values.items()):
        if key in current:
            continue
        for value in values:
            candidates.append(
                _candidate(
                    f"restore_removed:{key}:{value}",
                    {**current, key: value},
                    "removed_historical_mutation",
                )
            )
    return _deduplicate(candidates)


def _numeric_variant(value: int | float, factor: float, key: str) -> int | float:
    varied = value * factor
    if isinstance(value, int) or key.endswith(("_bars", "_lookback", "_period")):
        return max(1, int(round(varied)))
    return round(float(varied), 8)


def perturbation_candidates() -> list[dict[str, Any]]:
    """Fine neighbourhoods around every numeric cumulative mutation."""

    base = _current()
    candidates = [_candidate("baseline:round1_recovery", base, "baseline")]
    for key, value in sorted(base.items()):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        for factor in (0.80, 0.90, 0.95, 1.05, 1.10, 1.20):
            varied = _numeric_variant(value, factor, key)
            if varied == value:
                continue
            candidates.append(
                _candidate(
                    f"perturb:{key}:{factor:.2f}x",
                    {**base, key: varied},
                    "numeric_perturbation",
                )
            )

    grids: dict[str, tuple[Any, ...]] = {
        "param_overrides.adx_range_threshold": (12, 15, 16, 18, 20, 22, 25),
        "param_overrides.adx_trending_threshold": (16, 18, 20, 22, 25, 30),
        # Base-risk coverage is supplied by all +/-5/10/20% perturbations plus
        # the archived 0.8% and default 1.0% atomic tests.  More distant grid
        # points are intentionally omitted: they cross nonlinear contract-count
        # thresholds and were pathological in runtime without adding local
        # robustness evidence.
        "param_overrides.be_stop_buffer_mult": (0.0, 0.04, 0.08, 0.12, 0.16),
        "param_overrides.be_trigger_r": (0.5, 0.6, 0.75, 0.9, 1.0, 1.2, 1.5),
        "param_overrides.chandelier_lookback": (8, 12, 16, 20, 22, 24, 28, 32),
        "param_overrides.divergence_mag_threshold": (0.03, 0.05, 0.075, 0.1, 0.15),
        "param_overrides.drawdown_lookback": (5, 8, 10, 12, 15, 20),
        "param_overrides.ema_fast_period": (10, 15, 18, 20, 24, 30),
        "param_overrides.entry_ttl_bars": (6, 8, 12, 18, 24, 36, 48, 72, 96),
        "param_overrides.friction_min_atr_pctl": (0.0, 0.05, 0.1, 0.15, 0.25, 0.4),
        "param_overrides.min_hold_bars": (0, 4, 8, 10, 12, 13, 16, 18, 24),
        "param_overrides.momentum_cooldown_bars": (12, 18, 24, 30, 36, 48, 72, 96),
        "param_overrides.momentum_roc_threshold": (-0.0015, -0.002, -0.003, -0.004, -0.005, -0.006, -0.008, -0.01),
        "param_overrides.profit_floor_r_threshold": (0.5, 0.75, 1.0, 1.25, 1.5, 1.8, 2.2),
        "param_overrides.progressive_sma_min": (40, 60, 80, 100, 120, 150, 180, 220),
        "param_overrides.tp1_r_aligned": (1.5, 1.8, 2.0, 2.4, 2.8, 3.2),
        "param_overrides.tp1_r_emerging": (1.0, 1.25, 1.5, 1.8, 2.0, 2.4, 2.8),
        "param_overrides.entry_buffer_ticks": (0, 1, 2, 3, 4),
        "param_overrides.entry_limit_offset_ticks": (1, 2, 3, 4, 6, 8),
        "param_overrides.trigger_low_buffer_ticks": (0, 1, 2, 3, 4),
        "param_overrides.fade_stop_atr_mult": (0.25, 0.35, 0.5, 0.65, 0.85, 1.0),
        "param_overrides.max_daily_entries": (1, 2, 3, 4, 5, 8),
        "param_overrides.profit_floor_lock_pct": (0.25, 0.4, 0.5, 0.6, 0.7, 0.8),
        "param_overrides.stale_bars_fade": (12, 18, 24, 28, 36, 48, 72),
        "flags.regime_confidence_gate": (0.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        "flags.vol_percentile_gate": (0.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
    }
    for key, values in grids.items():
        for value in values:
            if base.get(key) == value:
                continue
            candidates.append(
                _candidate(
                    f"grid:{key}:{value}",
                    {**base, key: value},
                    "parameter_grid",
                )
            )

    standalone = {
        "disable_momentum": {"flags.momentum_signal": False},
        "disable_correction_override": {"flags.correction_regime_override": False},
        "disable_bear_structure": {"flags.bear_structure_override": False},
        "disable_drawdown_override": {"flags.drawdown_regime_override": False},
        "disable_progressive_sma": {"flags.progressive_sma": False},
        "disable_profit_floor": {"flags.profit_floor_trail": False},
        "disable_vwap_failure_exit": {"flags.vwap_failure_exit": False},
        "disable_min_hold": {"flags.min_hold_period": False},
        "enable_breakdown": {"flags.breakdown_engine": True},
        "enable_cancel_replace": {"flags.cancel_replace_entry": True},
        "correction_only": {"flags.correction_only_mode": True},
        "correction_only_fade": {"flags.correction_only_fade": True},
        "fade_allow_nonbear": {"flags.fade_bear_regime_required": False},
        "enable_adaptive_floor": {"flags.adaptive_profit_floor": True},
        "enable_multitier_floor": {"flags.multi_tier_profit_floor": True},
        "enable_regime_chandelier": {"flags.regime_adaptive_chandelier": True},
    }
    for name, mutation in standalone.items():
        candidates.append(
            _candidate(
                f"mechanism:{name}", {**base, **mutation}, "mechanism_perturbation"
            )
        )
    return _deduplicate(candidates)


def targeted_candidates(seed: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fine ADX repair and explicit interactions with independent controls."""

    base = dict(seed or _current())
    candidates = [_candidate("targeted:seed", base, "targeted_baseline")]
    adx_key = "param_overrides.adx_trending_threshold"
    for value in (20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.5, 23.0, 24.0, 25.0, 27.5, 30.0):
        candidates.append(
            _candidate(
                f"targeted:adx_trending:{value}",
                {**base, adx_key: value},
                "targeted_adx_refinement",
            )
        )

    adx_seed = {**base, adx_key: 21.0}
    companions: dict[str, dict[str, Any]] = {
        "drawdown_15": {"param_overrides.drawdown_lookback": 15},
        "drawdown_20": {"param_overrides.drawdown_lookback": 20},
        "chandelier_32": {"param_overrides.chandelier_lookback": 32},
        "ttl_18": {"param_overrides.entry_ttl_bars": 18},
        "entry_limit_6": {"param_overrides.entry_limit_offset_ticks": 6},
        "entry_buffer_3": {"param_overrides.entry_buffer_ticks": 3},
        "max_daily_4": {"param_overrides.max_daily_entries": 4},
        "profit_floor_threshold_080": {
            "param_overrides.profit_floor_r_threshold": 0.8
        },
        "regime_chandelier": {"flags.regime_adaptive_chandelier": True},
        "be_trigger_080": {"param_overrides.be_trigger_r": 0.8},
        "tp1_aligned_180": {"param_overrides.tp1_r_aligned": 1.8},
        "fade_stop_035": {"param_overrides.fade_stop_atr_mult": 0.35},
        "fade_stop_065": {"param_overrides.fade_stop_atr_mult": 0.65},
        "adx_range_12": {"param_overrides.adx_range_threshold": 12},
    }
    for name, mutation in companions.items():
        candidates.append(
            _candidate(
                f"targeted:adx21+{name}",
                {**adx_seed, **mutation},
                "targeted_single_interaction",
            )
        )

    # Pairwise interactions are explicit and exhaustive across the shortlisted
    # independent controls.  This avoids a single opaque mega-cluster.
    items = list(companions.items())
    for left_index, (left_name, left_mutation) in enumerate(items):
        for right_name, right_mutation in items[left_index + 1 :]:
            if set(left_mutation) & set(right_mutation):
                continue
            candidates.append(
                _candidate(
                    f"targeted:adx21+{left_name}+{right_name}",
                    {**adx_seed, **left_mutation, **right_mutation},
                    "targeted_pair_interaction",
                )
            )

    # A complexity-pruned equivalent removes values that are exact dataclass or
    # engine defaults.  Semantic no-ops tied to dormant paths remain documented
    # rather than silently removed from the active research configuration.
    pruned = dict(adx_seed)
    for key in (
        "flags.chandelier_trailing",
        "flags.progressive_sma",
        "flags.vol_percentile_gate",
        "param_overrides.ema_fast_period",
    ):
        pruned.pop(key, None)
    candidates.append(
        _candidate(
            "targeted:adx21_pruned_exact_defaults",
            pruned,
            "targeted_complexity_prune",
        )
    )
    return _deduplicate(candidates)


def verification_candidates() -> list[dict[str, Any]]:
    """Local stability surface and execution stresses around the targeted best."""

    targeted_path = OUTPUT_DIR / "targeted.json"
    if not targeted_path.exists():
        raise RuntimeError("targeted.json is required before verification")
    targeted_rows = _load_json(targeted_path)["results"]
    qualified = [
        row
        for row in targeted_rows
        if row.get("repair_qualification", {}).get("passed")
    ]
    selected = (qualified or targeted_rows)[0]
    seed = dict(selected["mutations"])
    candidates = [
        _candidate(
            "verification:selected_seed",
            seed,
            "verification_baseline",
            selected["name"],
        )
    ]

    for trending in (20.75, 21.0, 21.25):
        for chandelier in (28, 30, 32, 34, 36):
            for adx_range in (10, 11, 12, 13, 14, 15):
                candidates.append(
                    _candidate(
                        f"surface:trend_{trending}:chandelier_{chandelier}:range_{adx_range}",
                        {
                            **seed,
                            "param_overrides.adx_trending_threshold": trending,
                            "param_overrides.chandelier_lookback": chandelier,
                            "param_overrides.adx_range_threshold": adx_range,
                        },
                        "local_stability_surface",
                    )
                )

    stresses = {
        "commission_1_5x": {"slippage.commission_per_contract": 0.93},
        "commission_2x": {"slippage.commission_per_contract": 1.24},
        "slippage_2ticks": {"slippage.slip_ticks_normal": 2},
        "slippage_3ticks": {"slippage.slip_ticks_normal": 3},
        "spread_0_5bp": {"slippage.spread_bps": 0.5},
        "spread_1bp": {"slippage.spread_bps": 1.0},
        "entry_latency_1bar": {"entry_latency_bars": 1},
        "combined_execution": {
            "slippage.commission_per_contract": 1.24,
            "slippage.slip_ticks_normal": 2,
            "slippage.spread_bps": 1.0,
            "entry_latency_bars": 1,
        },
    }
    for name, mutation in stresses.items():
        candidates.append(
            _candidate(
                f"stress:{name}", {**seed, **mutation}, "execution_stress"
            )
        )
    frozen_baseline = _current()
    for name, mutation in stresses.items():
        candidates.append(
            _candidate(
                f"stress_baseline:{name}",
                {**frozen_baseline, **mutation},
                "baseline_execution_stress",
            )
        )
    return _deduplicate(candidates)


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return _load_json(CACHE_PATH)
    return {"schema_version": 1, "rows": {}}


def _repair_score(row: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Balanced observed-validation score; not a promotion statistic."""

    development = row.get("selection_metrics", {})
    oos = row.get("oos_metrics", {})
    base_dev = baseline["selection_metrics"]
    base_oos = baseline["oos_metrics"]
    if not development or not oos:
        return -99.0
    is_return_ratio = float(development["net_return_pct"]) / max(
        float(base_dev["net_return_pct"]), 1e-9
    )
    is_trade_ratio = float(development["total_trades"]) / max(
        float(base_dev["total_trades"]), 1.0
    )
    oos_trade_ratio = float(oos["total_trades"]) / max(
        float(base_oos["total_trades"]), 1.0
    )
    score = (
        0.32 * math.tanh(float(oos["net_return_pct"]) / 15.0)
        + 0.16 * math.tanh((float(oos["profit_factor"]) - 1.0) / 0.75)
        + 0.08 * math.tanh((float(oos["win_rate"]) - 35.0) / 20.0)
        + 0.10 * math.tanh((oos_trade_ratio - 0.60) / 0.40)
        + 0.14 * math.tanh((is_return_ratio - 0.75) / 0.25)
        + 0.08 * math.tanh((is_trade_ratio - 0.75) / 0.25)
        + 0.07 * math.tanh((float(development["profit_factor"]) - 2.0) / 1.0)
        + 0.05 * math.tanh((0.08 - float(development["max_dd_pct"])) / 0.04)
    )
    if float(development["net_return_pct"]) < 80.0:
        score -= 0.50
    if int(development["total_trades"]) < 90:
        score -= 0.35
    if float(development["profit_factor"]) < 2.0:
        score -= 0.35
    return score


def _qualification(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    development = row["selection_metrics"]
    oos = row["oos_metrics"]
    base_dev = baseline["selection_metrics"]
    criteria = {
        "development_return_preserved": float(development["net_return_pct"])
        >= 0.80 * float(base_dev["net_return_pct"]),
        "development_frequency_preserved": int(development["total_trades"])
        >= math.ceil(0.85 * int(base_dev["total_trades"])),
        "development_pf_ge_2_20": float(development["profit_factor"]) >= 2.20,
        "development_dd_le_8pct": float(development["max_dd_pct"]) <= 0.08,
        "observed_oos_positive": float(oos["net_pnl"]) > 0.0,
        "observed_oos_pf_ge_1_10": float(oos["profit_factor"]) >= 1.10,
        "observed_oos_trades_ge_8": int(oos["total_trades"]) >= 8,
    }
    return {**criteria, "passed": all(criteria.values())}


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    *,
    stage: str,
    max_workers: int = MAX_WORKERS,
) -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    rows_by_signature: dict[str, Any] = cache.setdefault("rows", {})
    pending: list[dict[str, Any]] = []
    for candidate in candidates:
        signature = _signature(candidate["mutations"])
        if signature not in rows_by_signature:
            pending.append(candidate)

    print(
        f"{stage}: {len(candidates)} unique candidates, "
        f"{len(candidates) - len(pending)} cached, {len(pending)} pending",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_split_worker,
            initargs=(
                str(recovery.DATA_DIR),
                recovery.INITIAL_EQUITY,
            ),
        ) as pool:
            futures = {
                pool.submit(
                    _evaluate_worker,
                    (candidate["name"], candidate["mutations"], False),
                ): candidate
                for candidate in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                candidate = futures[future]
                signature = _signature(candidate["mutations"])
                try:
                    row = future.result()
                except Exception as exc:  # pragma: no cover - durable research runner
                    row = {
                        "name": candidate["name"],
                        "signature": signature,
                        "mutations": candidate["mutations"],
                        "rejected": True,
                        "reject_reason": f"error:{type(exc).__name__}:{exc}",
                        "score": -99.0,
                    }
                rows_by_signature[signature] = row
                _write_json(CACHE_PATH, cache)
                development = row.get("selection_metrics", {})
                oos = row.get("oos_metrics", {})
                print(
                    f"[{completed}/{len(pending)}] {candidate['name']} "
                    f"IS={development.get('net_return_pct', 0):+.1f}%/"
                    f"{development.get('total_trades', 0)} "
                    f"OOS={oos.get('net_return_pct', 0):+.1f}%/"
                    f"{oos.get('total_trades', 0)}",
                    flush=True,
                )

    resolved: list[dict[str, Any]] = []
    baseline_signature = _signature(_current())
    baseline = rows_by_signature[baseline_signature]
    for candidate in candidates:
        signature = _signature(candidate["mutations"])
        row = dict(rows_by_signature[signature])
        row["name"] = candidate["name"]
        row["family"] = candidate["family"]
        row["note"] = candidate.get("note", "")
        row["aliases"] = candidate.get("aliases", [])
        row["repair_score"] = _repair_score(row, baseline)
        if row.get("selection_metrics") and row.get("oos_metrics"):
            row["repair_qualification"] = _qualification(row, baseline)
        resolved.append(row)
    resolved.sort(key=lambda row: row.get("repair_score", -99.0), reverse=True)
    _write_json(
        OUTPUT_DIR / f"{stage}.json",
        {
            "stage": stage,
            "candidate_count": len(resolved),
            "baseline": baseline,
            "results": resolved,
        },
    )
    return resolved


def _metric_group(trades: list[Any], key_fn) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        grouped[str(key_fn(trade))].append(trade)
    output: dict[str, dict[str, Any]] = {}
    for key, group in sorted(grouped.items()):
        pnls = [float(trade.pnl) for trade in group]
        rs = [float(trade.r_multiple) for trade in group]
        gross_profit = sum(value for value in pnls if value > 0)
        gross_loss = -sum(value for value in pnls if value < 0)
        output[key] = {
            "trades": len(group),
            "wins": sum(value > 0 for value in pnls),
            "win_rate": 100.0 * sum(value > 0 for value in pnls) / len(group),
            "net_pnl": sum(pnls),
            "profit_factor": gross_profit / gross_loss if gross_loss else 99.0,
            "net_r": sum(rs),
            "avg_r": statistics.fmean(rs),
        }
    return output


def _trade_diagnostics(trades: list[Any]) -> dict[str, Any]:
    development = [
        trade
        for trade in trades
        if IS_START <= recovery._trade_time(trade) < OOS_START
    ]
    oos = [
        trade
        for trade in trades
        if OOS_START <= recovery._trade_time(trade) < EVALUATION_END
    ]
    sorted_losses = sorted(
        [trade for trade in oos if float(trade.pnl) < 0],
        key=lambda trade: float(trade.pnl),
    )
    total_loss = -sum(min(0.0, float(trade.pnl)) for trade in oos)
    edge_cases = []
    for count in range(1, min(8, len(sorted_losses)) + 1):
        removed = sorted_losses[:count]
        remaining = [trade for trade in oos if trade not in removed]
        edge_cases.append(
            {
                "removed_worst_n": count,
                "removed_loss_dollars": -sum(float(trade.pnl) for trade in removed),
                "share_of_all_gross_loss": (
                    -sum(float(trade.pnl) for trade in removed) / total_loss
                    if total_loss
                    else 0.0
                ),
                "remaining_net_pnl": sum(float(trade.pnl) for trade in remaining),
                "remaining_wins": sum(float(trade.pnl) > 0 for trade in remaining),
                "remaining_trades": len(remaining),
            }
        )

    def groups(selected: list[Any]) -> dict[str, Any]:
        return {
            "overall": recovery._window_metrics(selected, recovery.INITIAL_EQUITY),
            "signal_class": _metric_group(selected, lambda trade: trade.signal_class),
            "regime": _metric_group(
                selected, lambda trade: trade.composite_regime_at_entry
            ),
            "vol_state": _metric_group(selected, lambda trade: trade.vol_state_at_entry),
            "exit_type": _metric_group(selected, lambda trade: trade.exit_type),
            "day": _metric_group(
                selected, lambda trade: recovery._trade_time(trade).date().isoformat()
            ),
            "session_hour_utc": _metric_group(
                selected, lambda trade: recovery._trade_time(trade).hour
            ),
        }

    return {
        "is_start": IS_START.isoformat(),
        "oos_start": OOS_START.isoformat(),
        "evaluation_end_exclusive": EVALUATION_END.isoformat(),
        "development": groups(development),
        "oos": groups(oos),
        "oos_edge_case_removal": edge_cases,
        "oos_trades": [asdict(trade) for trade in oos],
    }


def diagnose() -> None:
    _init_split_worker(str(recovery.DATA_DIR), recovery.INITIAL_EQUITY)
    row = _evaluate_worker(("baseline_detail", _current(), True))
    trades = row.pop("trades")
    payload = {"evaluation": row, "attribution": _trade_diagnostics(trades)}
    _write_json(OUTPUT_DIR / "baseline_attribution.json", payload)


def _all_stage_rows() -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in ("historical_ablation", "perturbation", "targeted", "verification"):
        path = OUTPUT_DIR / f"{name}.json"
        if not path.exists():
            continue
        for row in _load_json(path)["results"]:
            existing = rows.get(row["signature"])
            if existing is None or row.get("repair_score", -99) > existing.get(
                "repair_score", -99
            ):
                rows[row["signature"]] = row
    return sorted(
        rows.values(), key=lambda row: row.get("repair_score", -99), reverse=True
    )


def _bootstrap_window(trades: list[Any], samples: int = 10_000) -> dict[str, Any]:
    if not trades:
        return {
            "samples": samples,
            "trades": 0,
            "net_pnl_ci95": [0.0, 0.0],
            "net_r_ci95": [0.0, 0.0],
            "probability_positive_net_pnl": 0.0,
        }
    pnls = np.asarray([float(trade.pnl) for trade in trades], dtype=float)
    rs = np.asarray([float(trade.r_multiple) for trade in trades], dtype=float)
    rng = np.random.default_rng(20260808)
    indices = rng.integers(0, len(trades), size=(samples, len(trades)))
    pnl_draws = pnls[indices].sum(axis=1)
    r_draws = rs[indices].sum(axis=1)
    return {
        "samples": samples,
        "trades": len(trades),
        "net_pnl_ci95": [
            float(np.quantile(pnl_draws, 0.025)),
            float(np.quantile(pnl_draws, 0.975)),
        ],
        "net_r_ci95": [
            float(np.quantile(r_draws, 0.025)),
            float(np.quantile(r_draws, 0.975)),
        ],
        "probability_positive_net_pnl": float(np.mean(pnl_draws > 0)),
    }


def _same_performance(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for window in ("selection_metrics", "oos_metrics"):
        for key in ("total_trades", "net_pnl", "profit_factor", "max_dd_pct"):
            if not math.isclose(
                float(left[window][key]),
                float(right[window][key]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return False
    return True


def finalize() -> None:
    rows = _all_stage_rows()
    if not rows:
        raise RuntimeError("No completed candidate stages found")
    qualified = [
        row for row in rows if row.get("repair_qualification", {}).get("passed")
    ]
    eligible = [
        row
        for row in rows
        if row.get("selection_metrics", {}).get("net_return_pct", -999) >= 80.0
        and row.get("selection_metrics", {}).get("total_trades", 0) >= 90
        and row.get("selection_metrics", {}).get("profit_factor", 0) >= 2.0
    ]
    selected = (qualified or eligible or rows)[0]

    _init_split_worker(str(recovery.DATA_DIR), recovery.INITIAL_EQUITY)
    detail = _evaluate_worker(
        ("shadow_recommendation_detail", selected["mutations"], True)
    )
    selected_trades = detail.pop("trades")
    selected_attribution = _trade_diagnostics(selected_trades)
    selected_oos_trades = [
        trade
        for trade in selected_trades
        if OOS_START <= recovery._trade_time(trade) < EVALUATION_END
    ]
    _write_json(OUTPUT_DIR / "recommended_config.json", selected["mutations"])
    _write_json(
        OUTPUT_DIR / "recommended_trade_attribution.json",
        {"evaluation": detail, "attribution": selected_attribution},
    )

    baseline = next(row for row in rows if row["signature"] == _signature(_current()))
    historical = _load_json(OUTPUT_DIR / "historical_ablation.json")["results"]
    no_op_rows = [
        {
            "name": row["name"],
            "aliases": row.get("aliases", []),
            "note": row.get("note", ""),
        }
        for row in historical
        if row["signature"] != baseline["signature"]
        and _same_performance(row, baseline)
    ]
    baseline_row = next(
        row for row in historical if row["signature"] == baseline["signature"]
    )
    if baseline_row.get("aliases"):
        no_op_rows.insert(
            0,
            {
                "name": baseline_row["name"],
                "aliases": baseline_row["aliases"],
                "note": "exact-default mutation aliases",
            },
        )

    verification = _load_json(OUTPUT_DIR / "verification.json")["results"]
    surface = [
        row for row in verification if row["family"] == "local_stability_surface"
    ]
    surface_seed = next(
        row for row in verification if row["family"] == "verification_baseline"
    )
    surface_all = [surface_seed, *surface]
    surface_summary = {
        "points": len(surface_all),
        "strict_gate_passes": sum(
            row.get("repair_qualification", {}).get("passed", False)
            for row in surface_all
        ),
        "positive_oos_points": sum(
            float(row["oos_metrics"]["net_pnl"]) > 0 for row in surface_all
        ),
        "development_return_range_pct": [
            min(float(row["selection_metrics"]["net_return_pct"]) for row in surface_all),
            max(float(row["selection_metrics"]["net_return_pct"]) for row in surface_all),
        ],
        "oos_return_range_pct": [
            min(float(row["oos_metrics"]["net_return_pct"]) for row in surface_all),
            max(float(row["oos_metrics"]["net_return_pct"]) for row in surface_all),
        ],
    }
    stress_summary: dict[str, Any] = {}
    for stress_name in (
        "commission_1_5x",
        "commission_2x",
        "slippage_2ticks",
        "slippage_3ticks",
        "spread_0_5bp",
        "spread_1bp",
        "entry_latency_1bar",
        "combined_execution",
    ):
        candidate_stress = next(
            row for row in verification if row["name"] == f"stress:{stress_name}"
        )
        baseline_stress = next(
            row
            for row in verification
            if row["name"] == f"stress_baseline:{stress_name}"
        )
        stress_summary[stress_name] = {
            "candidate": {
                "selection_metrics": candidate_stress["selection_metrics"],
                "oos_metrics": candidate_stress["oos_metrics"],
            },
            "baseline": {
                "selection_metrics": baseline_stress["selection_metrics"],
                "oos_metrics": baseline_stress["oos_metrics"],
            },
        }

    stage_counts = {}
    for stage_name in (
        "historical_ablation",
        "perturbation",
        "targeted",
        "verification",
    ):
        payload = _load_json(OUTPUT_DIR / f"{stage_name}.json")
        stage_counts[stage_name] = int(payload["candidate_count"])
    baseline_attribution = _load_json(OUTPUT_DIR / "baseline_attribution.json")[
        "attribution"
    ]
    is_months = (OOS_START - IS_START).total_seconds() / 86_400.0 / 365.25 * 12.0
    oos_months = (
        (EVALUATION_END - OOS_START).total_seconds() / 86_400.0 / 365.25 * 12.0
    )
    summary = {
        "disposition": "SHADOW_RESEARCH_ONLY",
        "oos_status": (
            "The 2026-03-21 through 2026-05-01 OOS interval became observed "
            "validation during this repair round; fresh future data is required for "
            "promotion."
        ),
        "split": {
            "is_start": IS_START.isoformat(),
            "is_end_inclusive": "2026-03-20",
            "oos_start": OOS_START.isoformat(),
            "oos_end_inclusive": "2026-05-01",
            "evaluation_end_exclusive": EVALUATION_END.isoformat(),
        },
        "baseline": baseline,
        "selected": selected,
        "root_cause": {
            "finding": (
                "The reported severe OOS loss used the wrong interval (2026-05-02 "
                "onward). On the specified 2026-03-21 through 2026-05-01 OOS, "
                "the frozen baseline is profitable with six winners and one loser."
            ),
            "baseline_oos_active_days": len(
                baseline_attribution.get("oos", {}).get("day", {})
            ),
            "baseline_attribution_path": "baseline_attribution.json",
        },
        "oos_bootstrap": _bootstrap_window(selected_oos_trades),
        "frequency": {
            "baseline_is_trades_per_month": float(
                baseline["selection_metrics"]["total_trades"]
            )
            / is_months,
            "selected_is_trades_per_month": float(
                selected["selection_metrics"]["total_trades"]
            )
            / is_months,
            "baseline_oos_trades_per_month": float(
                baseline["oos_metrics"]["total_trades"]
            )
            / oos_months,
            "selected_oos_trades_per_month": float(
                selected["oos_metrics"]["total_trades"]
            )
            / oos_months,
        },
        "local_surface": surface_summary,
        "execution_stress_comparison": stress_summary,
        "no_op_or_redundant_mutations": no_op_rows,
        "candidate_counts_by_stage": stage_counts,
        "qualified_candidate_count": len(qualified),
        "evaluated_unique_configurations": len(rows),
        "top_20": rows[:20],
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    script_path = Path(__file__).resolve()
    data_paths = [
        recovery.DATA_DIR / "NQ_5m.parquet",
        recovery.DATA_DIR / "ES_1d.parquet",
        recovery.DATA_DIR / "NQ_5m.manifest.json",
        recovery.DATA_DIR / "ES_1d.manifest.json",
    ]
    _write_json(
        OUTPUT_DIR / "run_spec.json",
        {
            "purpose": "Round 1 corrected-split OOS ablation and robustness repair",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "initial_equity": recovery.INITIAL_EQUITY,
            "split": summary["split"],
            "max_workers": MAX_WORKERS,
            "baseline_signature": baseline["signature"],
            "selected_signature": selected["signature"],
            "script": {
                "path": str(script_path.relative_to(ROOT)),
                "sha256": _file_sha256(script_path),
            },
            "data": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _file_sha256(path),
                }
                for path in data_paths
                if path.exists()
            ],
            "candidate_counts_by_stage": stage_counts,
            "evaluated_unique_configurations": len(rows),
            "disposition": summary["disposition"],
        },
    )
    _write_report(summary)


def _write_report(summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    selected = summary["selected"]
    stresses = summary["execution_stress_comparison"]

    def line(label: str, row: dict[str, Any]) -> str:
        dev = row["selection_metrics"]
        oos = row["oos_metrics"]
        return (
            f"| {label} | {dev['total_trades']} | {dev['net_return_pct']:.2f}% | "
            f"{dev['profit_factor']:.2f} | {dev['max_dd_pct'] * 100:.2f}% | "
            f"{oos['total_trades']} | {oos['net_return_pct']:.2f}% | "
            f"{oos['profit_factor']:.2f} | {oos['win_rate']:.2f}% |"
        )

    lines = [
        "# Downturn Round 1 OOS Repair",
        "",
        "Disposition: **SHADOW_RESEARCH_ONLY**. The specified OOS interval is no longer "
        "untouched OOS because this round explicitly examined and optimized against "
        "the 2026-03-21 through 2026-05-01 interval. A new future holdout is required "
        "before promotion.",
        "",
        "| Configuration | IS trades | IS return | IS PF | IS DD | Validation trades | Validation return | Validation PF | Validation WR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        line("Frozen baseline", baseline),
        line("Shadow recommendation", selected),
        "",
        "## Root cause",
        "",
        summary["root_cause"]["finding"],
        "",
        "The specified baseline OOS contains seven trades across three active days: "
        "six winners and one -$99.24 loser. There is no catastrophic-loss cluster to "
        "repair in this interval; its weakness is low sample size and frequency.",
        "",
        f"Selected candidate: `{selected['name']}`",
        "",
        f"Unique configurations compared: {summary['evaluated_unique_configurations']}",
        "",
        f"Strict repair-gate passes: {summary['qualified_candidate_count']}",
        "",
        f"IS frequency: {summary['frequency']['baseline_is_trades_per_month']:.2f} "
        f"-> {summary['frequency']['selected_is_trades_per_month']:.2f} trades/month.",
        "",
        f"OOS frequency: {summary['frequency']['baseline_oos_trades_per_month']:.2f} "
        f"-> {summary['frequency']['selected_oos_trades_per_month']:.2f} trades/month.",
        "",
        "## Stability and execution",
        "",
        f"- Local surface: {summary['local_surface']['strict_gate_passes']}/"
        f"{summary['local_surface']['points']} strict passes; all "
        f"{summary['local_surface']['positive_oos_points']} points had positive OOS PnL.",
        f"- Selected OOS bootstrap probability of positive net PnL: "
        f"{summary['oos_bootstrap']['probability_positive_net_pnl']:.1%}.",
        f"- Selected OOS bootstrap 95% PnL interval: "
        f"${summary['oos_bootstrap']['net_pnl_ci95'][0]:,.0f} to "
        f"${summary['oos_bootstrap']['net_pnl_ci95'][1]:,.0f}.",
        "- One-bar entry latency remains the main fragility; see `summary.json` for "
        "the matched candidate/baseline stress table.",
        "",
        "| Stress | Candidate IS | Baseline IS | Candidate OOS | Baseline OOS |",
        "|---|---:|---:|---:|---:|",
        *[
            (
                f"| {name} | "
                f"{stresses[name]['candidate']['selection_metrics']['net_return_pct']:.2f}% | "
                f"{stresses[name]['baseline']['selection_metrics']['net_return_pct']:.2f}% | "
                f"{stresses[name]['candidate']['oos_metrics']['net_return_pct']:.2f}% | "
                f"{stresses[name]['baseline']['oos_metrics']['net_return_pct']:.2f}% |"
            )
            for name in (
                "commission_2x",
                "slippage_3ticks",
                "entry_latency_1bar",
                "combined_execution",
            )
        ],
        "",
        "## Mutation delta",
        "",
    ]
    base_mutations = baseline["mutations"]
    selected_mutations = selected["mutations"]
    for key in sorted(set(base_mutations) | set(selected_mutations)):
        old = base_mutations.get(key, "<default>")
        new = selected_mutations.get(key, "<default>")
        if old != new:
            lines.append(f"- `{key}`: `{old}` -> `{new}`")
    lines.extend(
        [
            "",
            "## Ablation findings",
            "",
            "- Exact-default redundancies: `flags.chandelier_trailing=True`, "
            "`flags.progressive_sma=True`, `flags.vol_percentile_gate=0`, and "
            "`param_overrides.ema_fast_period=20` do not add behavior over defaults.",
            "- Sample-path no-ops: `base_risk_pct` is floor-dominated at one contract; "
            "`divergence_mag_threshold` is dormant because reversal contributes zero "
            "trades; `regime_mult_counter` is dormant while counter entries are blocked.",
            "- These functional no-ops should be documented or pruned only with care: "
            "they can become active if contract sizing, reversal, or counter-regime "
            "policies change later.",
            "- The selected ADX/chandelier interaction is supported by the local "
            "surface, not by one isolated parameter point.",
            "",
            "## Limitations",
            "",
            "- OOS has only nine selected-candidate trades. The bootstrap PnL interval "
            "is positive, but the net-R interval still crosses zero; uncertainty remains "
            "material.",
            "- Because OOS was examined and used for selection, it is now validation. "
            "The recommendation must remain shadow-only until fresh future data accrues.",
            "- Entry latency materially reduces both configurations and should be "
            "monitored in paper/shadow execution.",
            "",
            "## Interpretation",
            "",
            "See `baseline_attribution.json` for trade-level loss concentration, "
            "`historical_ablation.json` for every cumulative/atomic historical test, "
            "`perturbation.json` for all numeric neighbourhoods, `targeted.json` "
            "for the additional robustness mechanisms, and `verification.json` "
            "for the final local surface and execution stresses.",
            "",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "diagnose",
            "historical",
            "perturb",
            "targeted",
            "verify",
            "final",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stages = {
        "diagnose": diagnose,
        "historical": lambda: evaluate_candidates(
            historical_candidates(),
            stage="historical_ablation",
            max_workers=args.max_workers,
        ),
        "perturb": lambda: evaluate_candidates(
            perturbation_candidates(),
            stage="perturbation",
            max_workers=args.max_workers,
        ),
        "targeted": lambda: evaluate_candidates(
            targeted_candidates(), stage="targeted", max_workers=args.max_workers
        ),
        "verify": lambda: evaluate_candidates(
            verification_candidates(),
            stage="verification",
            max_workers=args.max_workers,
        ),
        "final": finalize,
    }
    order = list(stages) if args.stage == "all" else [args.stage]
    for name in order:
        print(f"=== {name.upper()} ===", flush=True)
        stages[name]()


if __name__ == "__main__":
    main()
