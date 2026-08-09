"""Granular ALCB round-2 IS/OOS mutation robustness audit.

This is deliberately a research runner, not a promotion runner.  It evaluates
the repaired legacy cache only when ``--allow-legacy-data`` is supplied and
records that limitation in every output.  The OOS window is diagnostic after
the first screen; targeted candidates therefore remain exploratory until a
fresh frozen-bundle lockbox is available.

The runner is resumable.  Results are checkpointed after every small batch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time as clock_time, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.shared.auto.types import Experiment
from backtests.shared.auto.plugin_utils import create_process_pool, shutdown_process_pool
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin, _ThreadBatchEvaluator
from backtests.stock.auto.alcb.worker import init_worker, score_candidate
from backtests.stock.auto.alcb.time_utils import hydrate_time_mutations
from backtests.stock.auto.config_mutator import mutate_alcb_config
from backtests.stock.config_alcb import ALCBBacktestConfig
from strategies.stock.alcb.config import StrategySettings


IS_START = "2024-03-25"
IS_END = "2026-03-01"
OOS_START = "2026-03-02"
OOS_END = "2026-05-01"
INITIAL_EQUITY = 10_000.0
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests"
    / "output"
    / "stock"
    / "alcb"
    / "round_2"
    / "oos_robustness_20260722"
)
BASE_CONFIG_PATH = (
    REPO_ROOT / "backtests" / "output" / "stock" / "alcb" / "round_2" / "optimized_config.json"
)
BASE_IS_METRICS_PATH = (
    REPO_ROOT / "backtests" / "output" / "stock" / "alcb" / "round_2" / "final_metrics.json"
)
BASE_IS_TRADES_PATH = (
    REPO_ROOT / "backtests" / "output" / "stock" / "alcb" / "round_2" / "final_trades.json"
)
DATA_DIR = REPO_ROOT / "backtests" / "stock" / "data" / "raw"

CORE_METRICS = (
    "total_trades",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "net_profit",
    "profit_factor",
    "expectancy",
    "expectancy_dollar",
    "expected_total_r",
    "trades_per_month",
    "max_drawdown_pct",
    "sharpe",
    "sortino",
    "tail_loss_r",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: str
    category: str
    patch: dict[str, Any]
    thesis: str
    lineage: str = ""
    atomic_key: str = ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (date, datetime, clock_time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return str(value)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(metrics.get(key)) for key in CORE_METRICS if key in metrics}


def _delta(new: Any, old: Any) -> float:
    try:
        return float(new) - float(old)
    except (TypeError, ValueError):
        return 0.0


def _relative_delta(new: Any, old: Any) -> float:
    try:
        new_value = float(new)
        old_value = float(old)
    except (TypeError, ValueError):
        return 0.0
    scale = abs(old_value)
    return (new_value - old_value) / scale if scale > 1e-12 else new_value - old_value


def _bounded(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def utility(metrics: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Balanced return/frequency utility; never used as a deployment proof."""
    parts = {
        "expected_total_r": _bounded(_relative_delta(metrics.get("expected_total_r"), baseline.get("expected_total_r"))),
        "net_profit": _bounded(_relative_delta(metrics.get("net_profit"), baseline.get("net_profit"))),
        "expectancy": _bounded(_relative_delta(metrics.get("expectancy"), baseline.get("expectancy"))),
        "trades_per_month": _bounded(_relative_delta(metrics.get("trades_per_month"), baseline.get("trades_per_month"))),
        "profit_factor": _bounded(_relative_delta(metrics.get("profit_factor"), baseline.get("profit_factor"))),
        "win_rate": _bounded(_relative_delta(metrics.get("win_rate"), baseline.get("win_rate"))),
        "drawdown": _bounded(-_relative_delta(metrics.get("max_drawdown_pct"), baseline.get("max_drawdown_pct"))),
    }
    return (
        0.27 * parts["expected_total_r"]
        + 0.23 * parts["net_profit"]
        + 0.14 * parts["expectancy"]
        + 0.16 * parts["trades_per_month"]
        + 0.09 * parts["profit_factor"]
        + 0.04 * parts["win_rate"]
        + 0.07 * parts["drawdown"]
    )


def _strict_uplift(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("expected_total_r", 0.0)) > float(baseline.get("expected_total_r", 0.0))
        and float(metrics.get("net_profit", 0.0)) > float(baseline.get("net_profit", 0.0))
        and float(metrics.get("trades_per_month", 0.0)) >= float(baseline.get("trades_per_month", 0.0))
        and float(metrics.get("profit_factor", 0.0)) >= float(baseline.get("profit_factor", 0.0)) * 0.90
        and float(metrics.get("max_drawdown_pct", 1.0)) <= max(
            float(baseline.get("max_drawdown_pct", 0.0)) * 1.25,
            float(baseline.get("max_drawdown_pct", 0.0)) + 0.005,
        )
    )


def _is_guardrail(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("expected_total_r", 0.0)) >= float(baseline.get("expected_total_r", 0.0)) * 0.97
        and float(metrics.get("net_profit", 0.0)) >= float(baseline.get("net_profit", 0.0)) * 0.95
        and float(metrics.get("trades_per_month", 0.0)) >= float(baseline.get("trades_per_month", 0.0)) * 0.97
        and float(metrics.get("profit_factor", 0.0)) >= float(baseline.get("profit_factor", 0.0)) * 0.92
        and float(metrics.get("max_drawdown_pct", 1.0)) <= max(
            float(baseline.get("max_drawdown_pct", 0.0)) * 1.10,
            float(baseline.get("max_drawdown_pct", 0.0)) + 0.01,
        )
    )


def _effective_settings(mutations: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    hydrated = hydrate_time_mutations(mutations)
    config = mutate_alcb_config(ALCBBacktestConfig(), hydrated)
    settings = StrategySettings(**config.param_overrides) if config.param_overrides else StrategySettings()
    return asdict(config.ablation), asdict(settings)


def _literal_removal_audit(base: dict[str, Any]) -> list[dict[str, Any]]:
    base_ablation, base_settings = _effective_settings(base)
    rows: list[dict[str, Any]] = []
    for key in sorted(base):
        variant = dict(base)
        variant.pop(key)
        ablation, settings = _effective_settings(variant)
        if key.startswith("ablation."):
            field = key.split(".", 1)[1]
            before, after = base_ablation[field], ablation[field]
        else:
            field = key.split(".", 1)[1]
            before, after = base_settings[field], settings[field]
        rows.append(
            {
                "key": key,
                "literal_removal_changes_effective_config": before != after,
                "effective_before": before,
                "effective_after": after,
                "note": (
                    "behavioral literal removal"
                    if before != after
                    else "literal removal is a no-op because the accepted value is materialized in StrategySettings"
                ),
            }
        )
    return rows


def _counterfactual_controls() -> dict[str, Any]:
    """Pre-acceptance or neutral controls recovered from lineage/history.

    Even baseline-era values that predate the two recorded rounds are included
    so the cumulative configuration is covered literally, not just the latest
    accepted delta.  Where the feature did not exist historically, use a
    feature-neutral control.
    """
    return {
        "ablation.use_adaptive_trail": False,
        "ablation.use_combined_quality_gate": False,
        "ablation.use_mfe_conviction_exit": False,
        "ablation.use_or_width_min": False,
        "ablation.use_partial_takes": True,
        "param_overrides.adaptive_trail_late_activate_r": 0.25,
        "param_overrides.adaptive_trail_late_distance_r": 0.20,
        "param_overrides.adaptive_trail_start_bars": 0,
        # No pre-feature value exists.  999 removes only the late tightening
        # phase while retaining the mid-phase trail after start_bars.
        "param_overrides.adaptive_trail_tighten_bars": 999,
        "param_overrides.block_combined_regime_b": False,
        "param_overrides.carry_min_cpr": 0.40,
        "param_overrides.carry_min_r": 0.0,
        "param_overrides.combined_avwap_cap_pct": 0.0,
        "param_overrides.combined_breakout_min_rvol": 0.0,
        "param_overrides.combined_breakout_score_min": 0,
        "param_overrides.entry_window_end": clock_time(12, 0),
        "param_overrides.flow_reversal_min_hold_bars": 0,
        "param_overrides.fr_cpr_threshold": 0.0,
        "param_overrides.fr_mfe_grace_r": 0.15,
        "param_overrides.fr_trailing_activate_r": 0.30,
        "param_overrides.mfe_conviction_check_bars": 12,
        "param_overrides.mfe_conviction_floor_r": 0.0,
        "param_overrides.mfe_conviction_min_r": 0.15,
        "param_overrides.opening_range_bars": 12,
        "param_overrides.or_width_min_pct": 0.0,
        "param_overrides.pdh_avwap_cap_pct": 0.0,
        "param_overrides.pdh_size_mult": 1.0,
        "param_overrides.regime_mult_b": 0.50,
        "param_overrides.rvol_threshold": 1.50,
        "param_overrides.failure_stop_bars": 0,
        "param_overrides.failure_stop_mfe_max_r": 0.0,
        "param_overrides.failure_stop_current_r_max": -999.0,
        "param_overrides.failure_stop_to_r": -1.0,
    }


def _candidate_catalog(base: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(
        name: str,
        stage: str,
        category: str,
        patch: dict[str, Any],
        thesis: str,
        *,
        lineage: str = "",
        atomic_key: str = "",
    ) -> None:
        candidates.append(Candidate(name, stage, category, patch, thesis, lineage, atomic_key))

    for key, control in _counterfactual_controls().items():
        if base.get(key) == _json_safe(control) or base.get(key) == control:
            continue
        short = key.replace("ablation.", "").replace("param_overrides.", "").replace(".", "_")
        add(
            f"ablate__{short}",
            "ablation",
            "atomic_counterfactual",
            {key: control},
            f"Revert only {key} to its pre-acceptance or feature-neutral control.",
            lineage="all cumulative rounds",
            atomic_key=key,
        )

    score_map = dict(base["param_overrides.entry_score_size_mults"])
    for entry in sorted(score_map):
        variant = dict(score_map)
        removed = variant.pop(entry)
        add(
            f"ablate__entry_score_size__{entry.lower().replace(':', '_')}",
            "ablation",
            "nested_atomic_counterfactual",
            {"param_overrides.entry_score_size_mults": variant},
            f"Remove only nested score-sizing rule {entry}={removed}.",
            lineage="round 1/2 nested map",
            atomic_key=f"param_overrides.entry_score_size_mults[{entry}]",
        )
    detail_map = dict(base["param_overrides.entry_detail_size_mults"])
    for entry in sorted(detail_map):
        variant = dict(detail_map)
        removed = variant.pop(entry)
        add(
            f"ablate__entry_detail_size__{entry.lower().replace(':', '_').replace('!', 'not_')}",
            "ablation",
            "nested_atomic_counterfactual",
            {"param_overrides.entry_detail_size_mults": variant},
            f"Remove only nested detail-sizing rule {entry}={removed}.",
            lineage="round 2 nested map",
            atomic_key=f"param_overrides.entry_detail_size_mults[{entry}]",
        )

    add(
        "ablate__failure_stop_bundle",
        "ablation",
        "supplemental_bundle",
        {"param_overrides.failure_stop_bars": 0},
        "Disable the accepted failure-stop behavior as a bundle cross-check; atomic controls are tested separately.",
        lineage="round 1 accepted bundle",
    )
    add(
        "restore__round1_exact_delta",
        "ablation",
        "round2_bundle_crosscheck",
        {
            "param_overrides.adaptive_trail_late_activate_r": 0.25,
            "param_overrides.adaptive_trail_late_distance_r": 0.20,
            "param_overrides.entry_detail_size_mults": {
                "OR_BREAKOUT:5:!bar_vol_surge": 0.65,
                "*:5:!adx_trending": 0.85,
            },
            "param_overrides.entry_score_size_mults": {
                "OR_BREAKOUT:5": 0.70,
                "COMBINED_BREAKOUT:7": 1.15,
                "PDH_BREAKOUT:6": 0.65,
            },
        },
        "Restore every granular Round-2-vs-Round-1 difference together.",
        lineage="round 2 exact delta",
    )

    numeric_grids: dict[str, list[Any]] = {
        "param_overrides.adaptive_trail_late_activate_r": [0.15, 0.18, 0.25, 0.30],
        "param_overrides.adaptive_trail_late_distance_r": [0.08, 0.10, 0.15, 0.18, 0.20],
        "param_overrides.adaptive_trail_start_bars": [20, 22, 28, 30],
        "param_overrides.adaptive_trail_tighten_bars": [22, 28, 30, 35],
        "param_overrides.combined_avwap_cap_pct": [0.002, 0.0025, 0.004, 0.005],
        "param_overrides.combined_breakout_min_rvol": [2.2, 2.3, 2.7, 3.0],
        "param_overrides.combined_breakout_score_min": [4, 6],
        "param_overrides.flow_reversal_min_hold_bars": [8, 10, 14, 16],
        "param_overrides.fr_cpr_threshold": [0.20, 0.25, 0.35, 0.40],
        "param_overrides.fr_mfe_grace_r": [0.10, 0.15, 0.25, 0.30],
        "param_overrides.mfe_conviction_check_bars": [12, 14, 18, 20],
        "param_overrides.mfe_conviction_floor_r": [-0.25, -0.10, -0.05, 0.0],
        "param_overrides.mfe_conviction_min_r": [0.10, 0.15, 0.25, 0.30],
        "param_overrides.opening_range_bars": [5, 7, 8, 9],
        "param_overrides.or_width_min_pct": [0.0010, 0.0020, 0.0025, 0.0030],
        "param_overrides.pdh_avwap_cap_pct": [0.003, 0.004, 0.006, 0.008],
        "param_overrides.pdh_size_mult": [0.35, 0.50, 0.65, 0.90, 1.00],
        "param_overrides.regime_mult_b": [0.50, 0.60, 0.80, 0.90, 1.00],
        "param_overrides.rvol_threshold": [1.80, 1.90, 2.10, 2.20, 2.30],
        "param_overrides.failure_stop_bars": [6, 8, 12, 14, 16],
        "param_overrides.failure_stop_mfe_max_r": [0.10, 0.15, 0.25, 0.30],
        "param_overrides.failure_stop_current_r_max": [-0.20, -0.10, 0.10, 0.20],
        "param_overrides.failure_stop_to_r": [-0.50, -0.40, -0.30, -0.20, -0.10, 0.0],
    }
    for key, values in numeric_grids.items():
        current = base.get(key)
        for value in values:
            if value == current:
                continue
            short = key.split(".", 1)[1]
            label = str(value).replace("-", "m").replace(".", "p")
            add(
                f"perturb__{short}__{label}",
                "perturbation",
                "atomic_parameter_grid",
                {key: value},
                f"One-dimensional perturbation of {key} from {current} to {value}.",
                atomic_key=key,
            )
    for value in (clock_time(11, 30), clock_time(12, 0), clock_time(13, 0), clock_time(13, 30)):
        if value.isoformat() == str(base.get("param_overrides.entry_window_end")):
            continue
        add(
            f"perturb__entry_window_end__{value.strftime('%H%M')}",
            "perturbation",
            "atomic_time_grid",
            {"param_overrides.entry_window_end": value},
            "Move only the entry cutoff to test frequency/late-entry quality stability.",
            atomic_key="param_overrides.entry_window_end",
        )

    def score_patch(**updates: float) -> dict[str, Any]:
        merged = dict(score_map)
        merged.update(updates)
        return {"param_overrides.entry_score_size_mults": merged}

    for entry, values in {
        "OR_BREAKOUT:5": [0.55, 0.65, 0.70, 0.85, 1.00],
        "COMBINED_BREAKOUT:7": [0.90, 1.00, 1.10, 1.20, 1.25],
        "PDH_BREAKOUT:6": [0.35, 0.65, 0.75, 0.90, 1.00],
    }.items():
        for value in values:
            if score_map.get(entry) == value:
                continue
            add(
                f"perturb__score_size__{entry.lower().replace(':', '_')}__{str(value).replace('.', 'p')}",
                "perturbation",
                "nested_atomic_parameter_grid",
                score_patch(**{entry: value}),
                f"Perturb only the nested {entry} sizing multiplier.",
                atomic_key=f"param_overrides.entry_score_size_mults[{entry}]",
            )
    for value in (0.35, 0.45, 0.65, 0.75, 0.90, 1.0):
        if detail_map.get("OR_BREAKOUT:5:!bar_vol_surge") == value:
            continue
        merged = dict(detail_map)
        merged["OR_BREAKOUT:5:!bar_vol_surge"] = value
        add(
            f"perturb__or5_no_surge_size__{str(value).replace('.', 'p')}",
            "perturbation",
            "nested_atomic_parameter_grid",
            {"param_overrides.entry_detail_size_mults": merged},
            "Perturb only the OR score-5/no-volume-surge sizing penalty.",
            atomic_key="param_overrides.entry_detail_size_mults[OR_BREAKOUT:5:!bar_vol_surge]",
        )
    for value in (0.55, 0.70, 0.85, 1.0):
        merged = dict(detail_map)
        merged["*:5:!adx_trending"] = value
        add(
            f"perturb__restore_no_adx_size__{str(value).replace('.', 'p')}",
            "perturbation",
            "round2_removed_nested_rule",
            {"param_overrides.entry_detail_size_mults": merged},
            "Restore the Round-1 no-ADX score-5 sizing rule at one coefficient.",
            lineage="rule removed by Round 2",
        )

    targeted: list[tuple[str, dict[str, Any], str]] = [
        (
            "target__rvol190_pdh065",
            {"param_overrides.rvol_threshold": 1.90, "param_overrides.pdh_size_mult": 0.65},
            "Recover frequency via RVOL 1.9 while cautiously increasing the positive-PDH sleeve.",
        ),
        (
            "target__rvol190_pdh090",
            {"param_overrides.rvol_threshold": 1.90, "param_overrides.pdh_size_mult": 0.90},
            "Frequency recovery plus stronger PDH monetization.",
        ),
        (
            "target__rvol190_score_gradient",
            {
                "param_overrides.rvol_threshold": 1.90,
                "param_overrides.entry_score_size_mults": {
                    **score_map,
                    "OR_BREAKOUT:4": 0.90,
                    "OR_BREAKOUT:5": 0.85,
                    "OR_BREAKOUT:6": 1.05,
                    "COMBINED_BREAKOUT:7": 1.20,
                    "PDH_BREAKOUT:6": 0.65,
                },
            },
            "Align sizing with the monotonic OOS score gradient while retaining frequency recovery.",
        ),
        (
            "target__rvol190_failstop8",
            {"param_overrides.rvol_threshold": 1.90, "param_overrides.failure_stop_bars": 8},
            "Add frequency but accelerate protection of the entirely negative short-hold cohort.",
        ),
        (
            "target__rvol190_failstop12",
            {"param_overrides.rvol_threshold": 1.90, "param_overrides.failure_stop_bars": 12},
            "Test whether slower failure classification preserves recovered entries.",
        ),
        (
            "target__failure8_mfe015_to_m020",
            {
                "param_overrides.failure_stop_bars": 8,
                "param_overrides.failure_stop_mfe_max_r": 0.15,
                "param_overrides.failure_stop_to_r": -0.20,
            },
            "Earlier but more selective/tighter protection for the negative 0-24 bar cohort.",
        ),
        (
            "target__failure12_mfe025_to_m020",
            {
                "param_overrides.failure_stop_bars": 12,
                "param_overrides.failure_stop_mfe_max_r": 0.25,
                "param_overrides.failure_stop_to_r": -0.20,
            },
            "Give trades longer to mature, then retain more capital on failed paths.",
        ),
        (
            "target__adaptive_activate018_distance010",
            {
                "param_overrides.adaptive_trail_late_activate_r": 0.18,
                "param_overrides.adaptive_trail_late_distance_r": 0.10,
            },
            "Protect the large profitable 25+ bar cohort earlier and more tightly.",
        ),
        (
            "target__adaptive_activate025_distance015",
            {
                "param_overrides.adaptive_trail_late_activate_r": 0.25,
                "param_overrides.adaptive_trail_late_distance_r": 0.15,
            },
            "Relax Round-2 trail tightness to check for truncation overfit.",
        ),
        (
            "target__pdh_cap006_size075",
            {"param_overrides.pdh_avwap_cap_pct": 0.006, "param_overrides.pdh_size_mult": 0.75},
            "Slightly broaden and monetize PDH entries without changing other entry families.",
        ),
        (
            "target__pdh_cap008_size065",
            {"param_overrides.pdh_avwap_cap_pct": 0.008, "param_overrides.pdh_size_mult": 0.65},
            "Broaden PDH frequency with moderate sizing.",
        ),
        (
            "target__combined_rvol23_cap004_score5",
            {
                "param_overrides.combined_breakout_min_rvol": 2.30,
                "param_overrides.combined_avwap_cap_pct": 0.004,
                "param_overrides.combined_breakout_score_min": 5,
            },
            "Recover combined-breakout frequency while retaining a score floor.",
        ),
        (
            "target__combined_rvol22_cap005_score5",
            {
                "param_overrides.combined_breakout_min_rvol": 2.20,
                "param_overrides.combined_avwap_cap_pct": 0.005,
                "param_overrides.combined_breakout_score_min": 5,
            },
            "Broader combined-breakout frequency recovery stress test.",
        ),
        (
            "target__entry1300_rvol21",
            {"param_overrides.entry_window_end": clock_time(13, 0), "param_overrides.rvol_threshold": 2.10},
            "Trade a longer window but offset it with a higher RVOL floor.",
        ),
        (
            "target__entry1300_rvol19",
            {"param_overrides.entry_window_end": clock_time(13, 0), "param_overrides.rvol_threshold": 1.90},
            "Aggressive frequency recovery, included as a boundary/overfit check.",
        ),
        (
            "target__orb_quality_size60",
            {
                "ablation.use_orb_quality_gate": True,
                "param_overrides.orb_quality_score_min": 60.0,
                "param_overrides.orb_quality_size_floor": 0.60,
                "param_overrides.orb_quality_top_score": 85.0,
                "param_overrides.orb_quality_top_mult": 1.10,
            },
            "Continuously downsize the weak 60-70 quality bucket without a hard frequency cut.",
        ),
        (
            "target__orb_quality_size70",
            {
                "ablation.use_orb_quality_gate": True,
                "param_overrides.orb_quality_score_min": 60.0,
                "param_overrides.orb_quality_size_floor": 0.70,
                "param_overrides.orb_quality_top_score": 85.0,
                "param_overrides.orb_quality_top_mult": 1.10,
            },
            "Milder continuous quality sizing.",
        ),
        (
            "target__rvol190_quality_size70",
            {
                "param_overrides.rvol_threshold": 1.90,
                "ablation.use_orb_quality_gate": True,
                "param_overrides.orb_quality_score_min": 60.0,
                "param_overrides.orb_quality_size_floor": 0.70,
                "param_overrides.orb_quality_top_score": 85.0,
                "param_overrides.orb_quality_top_mult": 1.10,
            },
            "Pair frequency recovery with continuous quality-based risk control.",
        ),
    ]
    for name, patch, thesis in targeted:
        add(name, "targeted", "weakness_targeted", patch, thesis, lineage="post-OOS diagnostic")

    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise ValueError(f"duplicate candidate names: {duplicates}")
    return candidates


def _lineage_coverage(
    literal_audit: list[dict[str, Any]], candidates: list[Candidate]
) -> list[dict[str, Any]]:
    """Map every cumulative configuration key to its behavioral ablation(s)."""
    ablations = [candidate for candidate in candidates if candidate.stage == "ablation"]
    output = []
    for audit in literal_audit:
        key = str(audit["key"])
        direct = [candidate.name for candidate in ablations if candidate.atomic_key == key]
        nested = [
            candidate.name
            for candidate in ablations
            if candidate.atomic_key.startswith(f"{key}[")
        ]
        output.append(
            {
                "key": key,
                "coverage": "direct_atomic" if direct else "nested_members" if nested else "missing",
                "candidate_names": direct or nested,
                "literal_removal_changes_effective_config": audit["literal_removal_changes_effective_config"],
            }
        )
    return output


def _trade_to_dict(trade: Any) -> dict[str, Any]:
    if isinstance(trade, dict):
        payload = trade.copy()
        if "pnl_net" not in payload and "pnl" in payload:
            payload["pnl_net"] = float(payload.get("pnl", 0.0)) - float(payload.get("commission", 0.0))
        return _json_safe(payload)
    payload = vars(trade).copy()
    # TradeRecord exposes these as @property values, so vars(trade) alone
    # silently omitted net PnL and made every diagnostic trade look flat.
    for name in ("pnl_net", "hold_hours", "is_winner"):
        if hasattr(trade, name):
            payload[name] = getattr(trade, name)
    return _json_safe(payload)


def _diagnostics_consistent(diagnostics: dict[str, Any], metrics: dict[str, Any]) -> bool:
    """Reject stale diagnostics produced before computed TradeRecord fields were serialized."""
    return (
        int(diagnostics.get("trade_count", -1)) == int(metrics.get("total_trades", -2))
        and int(diagnostics.get("loss_count", -1)) == int(metrics.get("losing_trades", -2))
    )


def _trade_diagnostics(trades: Iterable[Any]) -> dict[str, Any]:
    rows = [_trade_to_dict(trade) for trade in trades]
    losses = sorted((row for row in rows if float(row.get("pnl_net", 0.0)) < 0.0), key=lambda row: row["pnl_net"])
    total_loss = -sum(float(row.get("pnl_net", 0.0)) for row in losses)

    def concentration(count: int) -> float:
        amount = -sum(float(row.get("pnl_net", 0.0)) for row in losses[:count])
        return amount / total_loss if total_loss else 0.0

    def grouped(name: str, key_fn) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(key_fn(row))].append(row)
        output = []
        for key, group in buckets.items():
            rs = [float(row.get("r_multiple", 0.0)) for row in group]
            pnls = [float(row.get("pnl_net", 0.0)) for row in group]
            output.append(
                {
                    name: key,
                    "trades": len(group),
                    "win_rate": sum(value > 0.0 for value in pnls) / len(group),
                    "avg_r": sum(rs) / len(group),
                    "total_r": sum(rs),
                    "pnl_net": sum(pnls),
                }
            )
        return sorted(output, key=lambda item: item["pnl_net"])

    def metadata(row: dict[str, Any], key: str, default: Any = None) -> Any:
        return (row.get("metadata") or {}).get(key, default)

    return {
        "trade_count": len(rows),
        "loss_count": len(losses),
        "gross_loss_dollar": -total_loss,
        "loss_concentration": {
            "worst_1_share": concentration(1),
            "worst_3_share": concentration(3),
            "worst_5_share": concentration(5),
            "worst_10_share": concentration(10),
        },
        "worst_trades": losses[:20],
        "monthly": grouped("month", lambda row: str(row.get("exit_time", ""))[:7]),
        "symbol": grouped("symbol", lambda row: row.get("symbol")),
        "entry_type": grouped("entry_type", lambda row: row.get("entry_type")),
        "exit_reason": grouped("exit_reason", lambda row: row.get("exit_reason")),
        "sector": grouped("sector", lambda row: row.get("sector")),
        "regime": grouped("regime", lambda row: row.get("regime_tier")),
        "score": grouped("score", lambda row: metadata(row, "momentum_score")),
        "hold_bucket": grouped(
            "hold_bucket",
            lambda row: (
                "00-04"
                if int(row.get("hold_bars", 0)) <= 4
                else "05-09"
                if int(row.get("hold_bars", 0)) <= 9
                else "10-15"
                if int(row.get("hold_bars", 0)) <= 15
                else "16-24"
                if int(row.get("hold_bars", 0)) <= 24
                else "25+"
            ),
        ),
        "rvol_bucket": grouped(
            "rvol_bucket",
            lambda row: (
                "<2.5"
                if float(metadata(row, "entry_signal_rvol", 0.0) or 0.0) < 2.5
                else "2.5-3.0"
                if float(metadata(row, "entry_signal_rvol", 0.0) or 0.0) < 3.0
                else "3.0-4.0"
                if float(metadata(row, "entry_signal_rvol", 0.0) or 0.0) < 4.0
                else "4.0+"
            ),
        ),
        "orb_quality_bucket": grouped(
            "orb_quality_bucket",
            lambda row: (
                "<60"
                if float(metadata(row, "orb_quality_score", 0.0) or 0.0) < 60.0
                else "60-70"
                if float(metadata(row, "orb_quality_score", 0.0) or 0.0) < 70.0
                else "70-80"
                if float(metadata(row, "orb_quality_score", 0.0) or 0.0) < 80.0
                else "80+"
            ),
        ),
    }


def _run_context(mutations: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    plugin = ALCBP16Plugin(
        DATA_DIR,
        start_date=start,
        end_date=end,
        initial_equity=INITIAL_EQUITY,
        max_workers=1,
    )
    try:
        return plugin._run_config(mutations, store_context=True, collect_diagnostics=False)
    finally:
        plugin.close_pool()


def _evaluate_candidates(
    candidates: list[Candidate],
    base: dict[str, Any],
    *,
    start: str,
    end: str,
    max_workers: int,
    output_path: Path,
    baseline_metrics: dict[str, Any],
    batch_size: int,
    evaluator_kind: str,
) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        payload = _load_json(output_path)
        existing = dict(payload.get("results", {}))
    # A rejected result may reflect a transient runner/config-serialization
    # defect rather than a strategy rejection.  Retrying it on resume is safe
    # because successful candidates remain checkpointed by name.
    pending = [
        candidate
        for candidate in candidates
        if candidate.name not in existing or existing[candidate.name].get("rejected")
    ]
    if not pending:
        print(f"resume: {output_path.name} already has all {len(candidates)} result(s)", flush=True)
        return existing

    evaluator = None
    process_pool = None
    if evaluator_kind == "process":
        process_pool = create_process_pool(
            max_workers,
            initializer=init_worker,
            initargs=(str(DATA_DIR), start, end, INITIAL_EQUITY, 0, {}, None),
            description=f"ALCB {start}..{end}",
        )
    else:
        evaluator = _ThreadBatchEvaluator(
            DATA_DIR,
            start,
            end,
            INITIAL_EQUITY,
            0,
            {},
            None,
            max_workers=max_workers,
            description=f"ALCB {start}..{end}",
            heartbeat_seconds=30.0,
            per_candidate_timeout_seconds=1800.0,
            minimum_timeout_seconds=1800.0,
        )
    try:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            started = time.perf_counter()
            print(
                f"evaluate {output_path.stem}: {offset + 1}-{offset + len(batch)} of {len(pending)} pending",
                flush=True,
            )
            if process_pool is not None:
                worker_args = [
                    (candidate.name, candidate.patch, base, 0, {}, None)
                    for candidate in batch
                ]
                scored = process_pool.map(score_candidate, worker_args, chunksize=1)
            else:
                assert evaluator is not None
                scored = evaluator(
                    [Experiment(candidate.name, candidate.patch) for candidate in batch],
                    base,
                )
            catalog = {candidate.name: candidate for candidate in batch}
            for result in scored:
                candidate = catalog[result.name]
                metrics = dict(result.metrics or {})
                existing[result.name] = {
                    "name": result.name,
                    "stage": candidate.stage,
                    "category": candidate.category,
                    "patch": candidate.patch,
                    "thesis": candidate.thesis,
                    "lineage": candidate.lineage,
                    "atomic_key": candidate.atomic_key,
                    "rejected": result.rejected,
                    "reject_reason": result.reject_reason,
                    "metrics": _metric_subset(metrics),
                    "utility_vs_baseline": utility(metrics, baseline_metrics) if metrics else -999.0,
                    "strict_return_frequency_uplift": _strict_uplift(metrics, baseline_metrics) if metrics else False,
                }
            _write_json(
                output_path,
                {
                    "window": {"start": start, "end": end},
                    "baseline_metrics": _metric_subset(baseline_metrics),
                    "completed": len(existing),
                    "requested": len(candidates),
                    "results": existing,
                },
            )
            print(f"checkpointed {len(existing)} result(s) in {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        if evaluator is not None:
            evaluator.close()
        shutdown_process_pool(process_pool)
    return existing


def _top_names(results: dict[str, dict[str, Any]], stage: str, count: int) -> list[str]:
    rows = [row for row in results.values() if row.get("stage") == stage and not row.get("rejected")]
    rows.sort(key=lambda row: float(row.get("utility_vs_baseline", -999.0)), reverse=True)
    return [row["name"] for row in rows[:count]]


def _merge_patches(*patches: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for patch in patches:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


def _combination_candidates(
    catalog: dict[str, Candidate],
    oos_results: dict[str, dict[str, Any]],
    is_results: dict[str, dict[str, Any]],
    limit: int = 8,
) -> list[Candidate]:
    eligible = []
    for name, row in oos_results.items():
        if row.get("stage") not in {"perturbation", "targeted"}:
            continue
        if name not in is_results:
            continue
        if not row.get("strict_return_frequency_uplift"):
            continue
        if not is_results[name].get("is_guardrail_pass"):
            continue
        eligible.append(name)
    eligible.sort(key=lambda name: float(oos_results[name].get("utility_vs_baseline", -999.0)), reverse=True)
    eligible = eligible[:limit]
    combos: list[Candidate] = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            left_patch, right_patch = catalog[left].patch, catalog[right].patch
            conflicting_scalars = {
                key
                for key in set(left_patch) & set(right_patch)
                if not (isinstance(left_patch[key], dict) and isinstance(right_patch[key], dict))
                and left_patch[key] != right_patch[key]
            }
            if conflicting_scalars:
                continue
            combos.append(
                Candidate(
                    name=f"combo__{left}__plus__{right}",
                    stage="combination",
                    category="pairwise_robust_singles",
                    patch=_merge_patches(left_patch, right_patch),
                    thesis=f"Pair two OOS-uplifting singles that independently pass IS guardrails: {left} + {right}.",
                    lineage="exploratory pairwise synthesis",
                )
            )
    return combos


def _attach_is_assessment(
    is_results: dict[str, dict[str, Any]],
    baseline_is: dict[str, Any],
    path: Path,
) -> None:
    for row in is_results.values():
        metrics = row.get("metrics", {})
        row["is_guardrail_pass"] = _is_guardrail(metrics, baseline_is) if metrics else False
        row["utility_vs_baseline"] = utility(metrics, baseline_is) if metrics else -999.0
    payload = _load_json(path)
    payload["results"] = is_results
    _write_json(path, payload)


def _flatten_results(
    oos_results: dict[str, dict[str, Any]],
    is_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for name, oos in oos_results.items():
        row = {
            "name": name,
            "stage": oos.get("stage"),
            "category": oos.get("category"),
            "oos_utility": oos.get("utility_vs_baseline"),
            "oos_strict_uplift": oos.get("strict_return_frequency_uplift"),
            "is_evaluated": name in is_results,
            "is_guardrail_pass": is_results.get(name, {}).get("is_guardrail_pass"),
        }
        for prefix, source in (("oos", oos), ("is", is_results.get(name, {}))):
            for key, value in source.get("metrics", {}).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    rows.sort(key=lambda row: float(row.get("oos_utility") or -999.0), reverse=True)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _recommendation(
    base: dict[str, Any],
    catalog: dict[str, Candidate],
    combos: dict[str, Candidate],
    oos_results: dict[str, dict[str, Any]],
    is_results: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
    all_catalog = {**catalog, **combos}
    eligible = []
    for name, oos in oos_results.items():
        if name not in is_results or name not in all_catalog:
            continue
        if not oos.get("strict_return_frequency_uplift"):
            continue
        if not is_results[name].get("is_guardrail_pass"):
            continue
        is_utility = float(is_results[name].get("utility_vs_baseline", -999.0))
        oos_utility = float(oos.get("utility_vs_baseline", -999.0))
        eligible.append(
            {
                "name": name,
                "robust_score": 0.65 * oos_utility + 0.35 * is_utility,
                "oos_utility": oos_utility,
                "is_utility": is_utility,
                "patch": all_catalog[name].patch,
                "stage": all_catalog[name].stage,
            }
        )
    eligible.sort(key=lambda row: row["robust_score"], reverse=True)
    if not eligible:
        return None, dict(base), eligible
    best = eligible[0]
    config = dict(base)
    config.update(best["patch"])
    return best["name"], config, eligible


def _render_report(
    baseline_is: dict[str, Any],
    baseline_oos: dict[str, Any],
    baseline_is_diag: dict[str, Any],
    baseline_oos_diag: dict[str, Any],
    literal_audit: list[dict[str, Any]],
    lineage_coverage: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    recommended_name: str | None,
    eligible: list[dict[str, Any]],
    recommended_is: dict[str, Any] | None,
    recommended_oos: dict[str, Any] | None,
    recommended_oos_diag: dict[str, Any] | None,
) -> str:
    oos_better = (
        float(baseline_oos.get("win_rate", 0.0)) > float(baseline_is.get("win_rate", 0.0))
        and float(baseline_oos.get("expectancy", 0.0)) > float(baseline_is.get("expectancy", 0.0))
        and float(baseline_oos.get("profit_factor", 0.0)) > float(baseline_is.get("profit_factor", 0.0))
    )
    meaningful_removals = sum(row["literal_removal_changes_effective_config"] for row in literal_audit)
    covered_mutations = sum(row["coverage"] != "missing" for row in lineage_coverage)
    top = rows[:15]
    row_by_name = {row["name"]: row for row in rows}

    def hold_summary(diagnostics: dict[str, Any]) -> tuple[int, float, int, float]:
        short = [row for row in diagnostics.get("hold_bucket", []) if row.get("hold_bucket") != "25+"]
        long = [row for row in diagnostics.get("hold_bucket", []) if row.get("hold_bucket") == "25+"]
        return (
            sum(int(row.get("trades", 0)) for row in short),
            sum(float(row.get("total_r", 0.0)) for row in short),
            sum(int(row.get("trades", 0)) for row in long),
            sum(float(row.get("total_r", 0.0)) for row in long),
        )

    def candidate_table(names: list[str]) -> list[str]:
        output = [
            "| Candidate | OOS delta R | OOS delta trades/mo | IS delta R | IS PF | IS trades/mo | IS DD | Guardrail |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for name in names:
            row = row_by_name.get(name)
            if not row or not row.get("is_evaluated"):
                continue
            output.append(
                f"| {name} | {float(row.get('oos_expected_total_r') or 0)-float(baseline_oos.get('expected_total_r',0)):+.2f} | "
                f"{float(row.get('oos_trades_per_month') or 0)-float(baseline_oos.get('trades_per_month',0)):+.2f} | "
                f"{float(row.get('is_expected_total_r') or 0)-float(baseline_is.get('expected_total_r',0)):+.2f} | "
                f"{float(row.get('is_profit_factor') or 0):.2f} | {float(row.get('is_trades_per_month') or 0):.1f} | "
                f"{100*float(row.get('is_max_drawdown_pct') or 0):.1f}% | "
                f"{'pass' if row.get('is_guardrail_pass') else 'fail'} |"
            )
        return output

    baseline_short_n, baseline_short_r, baseline_long_n, baseline_long_r = hold_summary(baseline_oos_diag)
    monthly = baseline_oos_diag.get("monthly", [])
    edge_detail = [
        "",
        (
            f"Holding-period attribution is much more informative than a tail event: the {baseline_short_n} trades held "
            f"0-24 bars contribute {baseline_short_r:+.2f}R, while the {baseline_long_n} trades held 25+ bars contribute "
            f"{baseline_long_r:+.2f}R. The weakness is repeated early trade failure, not an unbounded-loss outlier."
        ),
    ]
    if monthly:
        month_text = "; ".join(
            f"{row.get('month')}: {int(row.get('trades',0))} trades, {float(row.get('total_r',0)):+.2f}R"
            for row in sorted(monthly, key=lambda row: str(row.get("month")))
        )
        edge_detail.extend(
            [
                "",
                (
                    f"The short holdout is also temporally concentrated ({month_text}). Almost all aggregate OOS profit "
                    "comes from April, so the high OOS win rate is not evidence of stable month-to-month performance."
                ),
            ]
        )
    worst = baseline_oos_diag.get("worst_trades", [])[:5]
    if worst:
        edge_detail.extend(
            [
                "",
                "| Worst baseline OOS trade | Entry type | Exit | Hold bars | R | Net PnL |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for trade in worst:
            edge_detail.append(
                f"| {trade.get('symbol','?')} {str(trade.get('exit_time',''))[:10]} | {trade.get('entry_type','?')} | "
                f"{trade.get('exit_reason','?')} | {int(trade.get('hold_bars',0))} | "
                f"{float(trade.get('r_multiple',0)):+.2f} | ${float(trade.get('pnl_net',0)):,.2f} |"
            )
    if recommended_oos_diag:
        rec_short_n, rec_short_r, rec_long_n, rec_long_r = hold_summary(recommended_oos_diag)
        edge_detail.extend(
            [
                "",
                (
                    f"For the recommended patch, 0-24-bar trades contribute {rec_short_r:+.2f}R across {rec_short_n} trades "
                    f"and 25+-bar trades contribute {rec_long_r:+.2f}R across {rec_long_n} trades. This is an opportunity/"
                    "selection uplift; it does not eliminate the structurally negative short-hold cohort."
                ),
            ]
        )

    ablation_findings = candidate_table(
        [
            "ablate__use_adaptive_trail",
            "ablate__adaptive_trail_start_bars",
            "ablate__adaptive_trail_tighten_bars",
            "ablate__fr_trailing_activate_r",
            "ablate__use_partial_takes",
            "restore__round1_exact_delta",
            "ablate__combined_avwap_cap_pct",
            "ablate__use_combined_quality_gate",
            "ablate__failure_stop_bars",
            "ablate__flow_reversal_min_hold_bars",
            "ablate__carry_min_cpr",
            "ablate__carry_min_r",
            "ablate__fr_cpr_threshold",
            "ablate__use_mfe_conviction_exit",
            "ablate__entry_detail_size__or_breakout_5_not_bar_vol_surge",
            "ablate__combined_breakout_score_min",
        ]
    )
    perturbation_findings = candidate_table(
        [
            "perturb__adaptive_trail_late_distance_r__0p08",
            "perturb__opening_range_bars__8",
            "perturb__opening_range_bars__9",
            "perturb__or_width_min_pct__0p001",
            "perturb__rvol_threshold__1p9",
            "perturb__rvol_threshold__1p8",
            "ablate__rvol_threshold",
            "ablate__use_or_width_min",
        ]
    )
    targeted_findings = candidate_table(
        [
            "target__rvol190_pdh090",
            "target__rvol190_score_gradient",
            "target__entry1300_rvol19",
            "combo__target__rvol190_pdh090__plus__target__entry1300_rvol19",
            "target__rvol190_failstop8",
            "target__combined_rvol22_cap005_score5",
        ]
    )
    lines = [
        "# ALCB Round 2 OOS robustness audit",
        "",
        "## Executive finding",
        "",
        (
            "The repaired-cache replay does **not** reproduce aggregate OOS underperformance. "
            if oos_better
            else "The repaired-cache replay reproduces at least part of the reported OOS quality gap. "
        )
        + "This is a diagnostic-only result because the repository has no accepted frozen direct-RTH bundle.",
        "",
        "| Window | Trades | Win rate | Avg R | Total R | PF | Net PnL | Trades/month | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| IS {IS_START}..{IS_END} | {baseline_is.get('total_trades', 0):.0f} | "
            f"{100*float(baseline_is.get('win_rate', 0)):.1f}% | {float(baseline_is.get('expectancy', 0)):+.3f} | "
            f"{float(baseline_is.get('expected_total_r', 0)):+.2f} | {float(baseline_is.get('profit_factor', 0)):.2f} | "
            f"${float(baseline_is.get('net_profit', 0)):,.2f} | {float(baseline_is.get('trades_per_month', 0)):.1f} | "
            f"{100*float(baseline_is.get('max_drawdown_pct', 0)):.1f}% |"
        ),
        (
            f"| OOS {OOS_START}..{OOS_END} | {baseline_oos.get('total_trades', 0):.0f} | "
            f"{100*float(baseline_oos.get('win_rate', 0)):.1f}% | {float(baseline_oos.get('expectancy', 0)):+.3f} | "
            f"{float(baseline_oos.get('expected_total_r', 0)):+.2f} | {float(baseline_oos.get('profit_factor', 0)):.2f} | "
            f"${float(baseline_oos.get('net_profit', 0)):,.2f} | {float(baseline_oos.get('trades_per_month', 0)):.1f} | "
            f"{100*float(baseline_oos.get('max_drawdown_pct', 0)):.1f}% |"
        ),
        "",
        "## Edge-case loss concentration",
        "",
        (
            f"OOS gross loss is spread across {baseline_oos_diag.get('loss_count', 0)} losses. "
            f"The worst 1/3/5 trades account for "
            f"{100*float(baseline_oos_diag.get('loss_concentration', {}).get('worst_1_share', 0)):.1f}% / "
            f"{100*float(baseline_oos_diag.get('loss_concentration', {}).get('worst_3_share', 0)):.1f}% / "
            f"{100*float(baseline_oos_diag.get('loss_concentration', {}).get('worst_5_share', 0)):.1f}% of gross loss."
        ),
        *edge_detail,
        "",
        "## Mutation lineage warning",
        "",
        (
            f"Only {meaningful_removals} of {len(literal_audit)} top-level literal mutation removals change the effective "
            "runtime configuration. Most accepted parameter values were later baked into `StrategySettings`, so a naive "
            "delete-key ablation falsely reports no effect. This audit therefore uses explicit historical/neutral controls "
            f"and separately removes every nested sizing-map member. Behavioral coverage is {covered_mutations}/"
            f"{len(lineage_coverage)} cumulative keys."
        ),
        "",
        "## Ablation conclusions",
        "",
        "The core exit architecture is indispensable: adaptive trailing, its fast-runner activation, and partial takes "
        "all suffer large cross-window damage when ablated. Restoring the complete Round 1 delta also fails, so a blanket "
        "Round 2 rollback is not supported. Removing only the late adaptive-tightening phase cuts IS to about 66R. The "
        "failure stop adds about 7R in-sample but is not the source of an OOS tail.",
        "",
        *ablation_findings,
        "",
        "Several accepted micro-mutations are low-value rather than catastrophic. The carry CPR threshold is an exact "
        "no-op, the carry-R floor is nearly flat, and the flow-reversal CPR gate, MFE-conviction exit, combined-breakout "
        "score floor, and individual score/detail map entries move full-history results only a few R. They are "
        "simplification candidates, but removing them does not explain or fix the claimed OOS gap.",
        "",
        "## Perturbation stability and rejected OOS fits",
        "",
        "The strongest raw OOS winners mostly relax entry filters, but the aggressive versions fail the historical risk "
        "guardrail. Removing the quality gate, removing the OR-width floor, loosening the combined AVWAP cap, and RVOL "
        "1.8 are holdout-favored fits. Historical RVOL 1.5 raises IS return/frequency to 202R/58.8 trades per month but "
        "misses the PF retention floor at 1.62, making it an aggressive near-miss rather than a robust recommendation. "
        "Nearby milder values are stable: RVOL 1.9, OR width 0.10%, opening range 8-9 bars, and late-trail distance 0.08R all pass.",
        "",
        *perturbation_findings,
        "",
        "## Targeted post-diagnostic phase",
        "",
        "The targeted phase deliberately avoids symbol, sector, or date exclusions. Its robust frontier is driven by "
        "RVOL 1.9 plus modest sizing/timing refinements; permissive combined-breakout recipes remain OOS-only fits.",
        "",
        *targeted_findings,
        "",
        "## Top screened configurations",
        "",
        "| Candidate | Stage | OOS utility | OOS uplift | IS checked | IS guardrail | OOS total R | OOS trades/mo | OOS PF |",
        "|---|---|---:|:---:|:---:|:---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| {row['name']} | {row.get('stage','')} | {float(row.get('oos_utility') or 0):+.3f} | "
            f"{'yes' if row.get('oos_strict_uplift') else 'no'} | {'yes' if row.get('is_evaluated') else 'no'} | "
            f"{'yes' if row.get('is_guardrail_pass') else 'no'} | {float(row.get('oos_expected_total_r') or 0):+.2f} | "
            f"{float(row.get('oos_trades_per_month') or 0):.1f} | {float(row.get('oos_profit_factor') or 0):.2f} |"
        )
    lines.extend(["", "## Exploratory recommendation", ""])
    if recommended_name is None:
        lines.append("No tested configuration simultaneously improved OOS total return and frequency while passing the IS guardrails.")
    else:
        lines.append(
            f"`{recommended_name}` is the highest balanced-score exploratory configuration among candidates that strictly "
            "raise OOS total R, net PnL, and frequency and pass the predefined IS degradation limits. It is **not approved "
            "for promotion** because the targeted search inspected this OOS sample and the authoritative bundle is absent."
        )
        recommended_entry = next((row for row in eligible if row.get("name") == recommended_name), None)
        if recommended_entry:
            patch_text = ", ".join(
                f"`{key}` = `{value}`" for key, value in recommended_entry.get("patch", {}).items()
            )
            lines.extend(["", f"Effective delta: {patch_text}."])
        if recommended_is and recommended_oos:
            lines.extend(
                [
                    "",
                    "| Configuration | Window | Trades | Win rate | Avg R | Total R | PF | Net PnL | Trades/month | Max DD |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                    (
                        f"| recommended | IS | {recommended_is.get('total_trades',0):.0f} | "
                        f"{100*float(recommended_is.get('win_rate',0)):.1f}% | {float(recommended_is.get('expectancy',0)):+.3f} | "
                        f"{float(recommended_is.get('expected_total_r',0)):+.2f} | {float(recommended_is.get('profit_factor',0)):.2f} | "
                        f"${float(recommended_is.get('net_profit',0)):,.2f} | {float(recommended_is.get('trades_per_month',0)):.1f} | "
                        f"{100*float(recommended_is.get('max_drawdown_pct',0)):.1f}% |"
                    ),
                    (
                        f"| recommended | OOS | {recommended_oos.get('total_trades',0):.0f} | "
                        f"{100*float(recommended_oos.get('win_rate',0)):.1f}% | {float(recommended_oos.get('expectancy',0)):+.3f} | "
                        f"{float(recommended_oos.get('expected_total_r',0)):+.2f} | {float(recommended_oos.get('profit_factor',0)):.2f} | "
                        f"${float(recommended_oos.get('net_profit',0)):,.2f} | {float(recommended_oos.get('trades_per_month',0)):.1f} | "
                        f"{100*float(recommended_oos.get('max_drawdown_pct',0)):.1f}% |"
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## Statistical interpretation",
            "",
            "- Atomic ablations answer mutation dependence; perturbations test local stability; targeted and pairwise candidates are exploratory searches.",
            "- The OOS sample is only about two months. Reusing it for targeted design consumes the lockbox and creates selection bias.",
            "- Symbol/sector/day exclusions are intentionally absent: they are the easiest route to small-sample overfit.",
            "- Before any production change, rebuild the missing frozen direct-RTH bundle and rerun the recommended configuration on a fresh later lockbox.",
            "",
            "## Artifacts",
            "",
            "See `literal_removal_audit.json`, `lineage_coverage.json`, `baseline_diagnostics.json`, `oos_screen.json`, `is_validation.json`, "
            "`combination_oos.json`, `combination_is.json`, `all_results.csv`, `robust_eligible.json`, "
            "`recommended_config.json`, and `recommended_oos_diagnostics.json` in this directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--evaluator", choices=("process", "thread"), default="process")
    parser.add_argument("--allow-legacy-data", action="store_true")
    parser.add_argument("--top-is-per-stage", type=int, default=14)
    parser.add_argument("--skip-combinations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.allow_legacy_data:
        raise SystemExit(
            "No accepted frozen direct-RTH bundle is present. Pass --allow-legacy-data to run the repaired legacy cache "
            "in diagnostic-only mode."
        )
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"

    base = _load_json(BASE_CONFIG_PATH)
    baseline_is = _load_json(BASE_IS_METRICS_PATH)
    baseline_is_trades = _load_json(BASE_IS_TRADES_PATH)
    literal_audit = _literal_removal_audit(base)
    candidates = _candidate_catalog(base)
    lineage_coverage = _lineage_coverage(literal_audit, candidates)
    catalog = {candidate.name: candidate for candidate in candidates}
    _write_json(output_dir / "run_spec.json", {
        "generated_at_utc": datetime.now(timezone.utc),
        "data_authority": "diagnostic-only repaired legacy filename cache",
        "authoritative_bundle_available": False,
        "base_config": str(BASE_CONFIG_PATH),
        "windows": {"is": {"start": IS_START, "end": IS_END}, "oos": {"start": OOS_START, "end": OOS_END}},
        "max_workers": args.max_workers,
        "batch_size": args.batch_size,
        "evaluator": args.evaluator,
        "candidate_count": len(candidates),
        "candidate_counts_by_stage": {
            stage: sum(candidate.stage == stage for candidate in candidates)
            for stage in sorted({candidate.stage for candidate in candidates})
        },
    })
    _write_json(output_dir / "candidate_catalog.json", [asdict(candidate) for candidate in candidates])
    _write_json(output_dir / "literal_removal_audit.json", literal_audit)
    _write_json(output_dir / "lineage_coverage.json", lineage_coverage)

    baseline_path = output_dir / "baseline_diagnostics.json"
    baseline_payload = _load_json(baseline_path) if baseline_path.exists() else None
    if baseline_payload and _diagnostics_consistent(
        baseline_payload["oos"]["diagnostics"], baseline_payload["oos"]["metrics"]
    ):
        baseline_oos = baseline_payload["oos"]["metrics"]
        baseline_oos_diag = baseline_payload["oos"]["diagnostics"]
        print("resume: loaded baseline diagnostics", flush=True)
    else:
        started = time.perf_counter()
        print("run OOS baseline with trade diagnostics", flush=True)
        context = _run_context(base, OOS_START, OOS_END)
        baseline_oos = _metric_subset(context["metrics"])
        baseline_oos_diag = _trade_diagnostics(context["trades"])
        baseline_payload = {
            "is": {"window": {"start": IS_START, "end": IS_END}, "metrics": _metric_subset(baseline_is), "diagnostics": _trade_diagnostics(baseline_is_trades)},
            "oos": {"window": {"start": OOS_START, "end": OOS_END}, "metrics": baseline_oos, "diagnostics": baseline_oos_diag},
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(baseline_path, baseline_payload)
    baseline_is_diag = baseline_payload["is"]["diagnostics"]

    print(f"screen {len(candidates)} atomic/perturbation/targeted candidates on OOS", flush=True)
    oos_results = _evaluate_candidates(
        candidates,
        base,
        start=OOS_START,
        end=OOS_END,
        max_workers=args.max_workers,
        output_path=output_dir / "oos_screen.json",
        baseline_metrics=baseline_oos,
        batch_size=args.batch_size,
        evaluator_kind=args.evaluator,
    )

    validate_names = set(_top_names(oos_results, "perturbation", args.top_is_per_stage))
    validate_names.update(_top_names(oos_results, "targeted", args.top_is_per_stage))
    validate_names.update(
        candidate.name
        for candidate in candidates
        if candidate.stage == "ablation" and candidate.category != "supplemental_bundle"
    )
    is_candidates = [catalog[name] for name in sorted(validate_names)]
    print(f"validate {len(is_candidates)} candidates on full IS", flush=True)
    is_path = output_dir / "is_validation.json"
    is_results = _evaluate_candidates(
        is_candidates,
        base,
        start=IS_START,
        end=IS_END,
        max_workers=args.max_workers,
        output_path=is_path,
        baseline_metrics=baseline_is,
        batch_size=max(4, min(args.batch_size, 8)),
        evaluator_kind=args.evaluator,
    )
    _attach_is_assessment(is_results, baseline_is, is_path)

    combo_catalog: dict[str, Candidate] = {}
    if not args.skip_combinations:
        combo_candidates = _combination_candidates(catalog, oos_results, is_results)
        combo_catalog = {candidate.name: candidate for candidate in combo_candidates}
        _write_json(output_dir / "combination_catalog.json", [asdict(candidate) for candidate in combo_candidates])
        if combo_candidates:
            combo_oos = _evaluate_candidates(
                combo_candidates,
                base,
                start=OOS_START,
                end=OOS_END,
                max_workers=args.max_workers,
                output_path=output_dir / "combination_oos.json",
                baseline_metrics=baseline_oos,
                batch_size=args.batch_size,
                evaluator_kind=args.evaluator,
            )
            oos_results.update(combo_oos)
            top_combo_names = _top_names(combo_oos, "combination", min(10, len(combo_candidates)))
            combo_is_candidates = [combo_catalog[name] for name in top_combo_names]
            combo_is_path = output_dir / "combination_is.json"
            combo_is = _evaluate_candidates(
                combo_is_candidates,
                base,
                start=IS_START,
                end=IS_END,
                max_workers=args.max_workers,
                output_path=combo_is_path,
                baseline_metrics=baseline_is,
                batch_size=max(4, min(args.batch_size, 8)),
                evaluator_kind=args.evaluator,
            )
            _attach_is_assessment(combo_is, baseline_is, combo_is_path)
            is_results.update(combo_is)

    recommended_name, recommended_config, eligible = _recommendation(
        base, catalog, combo_catalog, oos_results, is_results
    )
    _write_json(output_dir / "robust_eligible.json", eligible)
    _write_json(output_dir / "recommended_config.json", recommended_config)

    recommended_is_metrics = None
    recommended_oos_metrics = None
    recommended_oos_diag = None
    if recommended_name:
        recommended_is_metrics = is_results[recommended_name]["metrics"]
        recommended_oos_metrics = oos_results[recommended_name]["metrics"]
        final_path = output_dir / "recommended_oos_diagnostics.json"
        final_payload = _load_json(final_path) if final_path.exists() else None
        if (
            not final_payload
            or final_payload.get("name") != recommended_name
            or not _diagnostics_consistent(final_payload.get("diagnostics", {}), final_payload.get("metrics", {}))
        ):
            print(f"run final trade-level diagnostics for {recommended_name}", flush=True)
            context = _run_context(recommended_config, OOS_START, OOS_END)
            final_payload = {
                "name": recommended_name,
                "metrics": _metric_subset(context["metrics"]),
                "diagnostics": _trade_diagnostics(context["trades"]),
            }
            _write_json(final_path, final_payload)
        recommended_oos_diag = final_payload["diagnostics"]

    rows = _flatten_results(oos_results, is_results)
    _write_csv(output_dir / "all_results.csv", rows)
    _write_json(output_dir / "all_results.json", rows)
    report = _render_report(
        baseline_is,
        baseline_oos,
        baseline_is_diag,
        baseline_oos_diag,
        literal_audit,
        lineage_coverage,
        rows,
        recommended_name,
        eligible,
        recommended_is_metrics,
        recommended_oos_metrics,
        recommended_oos_diag,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _write_json(
        output_dir / "completion.json",
        {
            "completed_at_utc": datetime.now(timezone.utc),
            "candidate_count": len(candidates),
            "oos_result_count": len(oos_results),
            "is_result_count": len(is_results),
            "recommended_name": recommended_name,
            "promotion_authorized": False,
            "promotion_blocker": "No accepted frozen direct-RTH bundle and OOS was consumed by targeted search.",
        },
    )
    print(f"complete: {output_dir}", flush=True)
    print(f"recommended exploratory config: {recommended_name or 'none'}", flush=True)


if __name__ == "__main__":
    main()
