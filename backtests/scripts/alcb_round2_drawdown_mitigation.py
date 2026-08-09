"""Targeted ALCB Round-2 drawdown mitigation and alpha-recovery research.

The script starts from the balanced post-audit candidate (RVOL 1.70, OR 9,
late trail 0.04), retains every accepted mutation in the Round-2 optimized
configuration, and applies one granular change at a time.  It then builds a
small set of orthogonal combinations from the strongest atomic mechanisms and
validates finalists on both in-sample halves and the historical max-drawdown
descent.

All outputs are diagnostic-only.  The repaired legacy cache is not an
authoritative frozen bundle, and the OOS interval has already been consumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from backtests.scripts.alcb_round2_drawdown_diagnostics import BALANCED_PATCH
from backtests.scripts.alcb_round2_oos_robustness import (
    BASE_CONFIG_PATH,
    Candidate,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    REPO_ROOT,
    _evaluate_candidates,
    _load_json,
    _write_json,
)


PRIOR_REVIEW = (
    REPO_ROOT
    / "backtests"
    / "output"
    / "stock"
    / "alcb"
    / "round_2"
    / "oos_robustness_20260722"
    / "recommendation_review_20260722"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests"
    / "output"
    / "stock"
    / "alcb"
    / "round_2"
    / "oos_robustness_20260722"
    / "drawdown_mitigation_20260723"
)
BALANCED_NAME = "control__balanced_rvol170_or9_trail004"
BALANCED_REVIEW_NAME = "boundary__rvol1p7__or9__trail0p04"
EARLY_IS = ("2024-03-25", "2025-03-24")
LATE_IS = ("2025-03-25", "2026-03-01")
MAX_DD_DESCENT = ("2024-07-19", "2024-09-30")

BASE_SCORE_MULTS = {
    "OR_BREAKOUT:5": 0.75,
    "COMBINED_BREAKOUT:7": 1.15,
    "PDH_BREAKOUT:6": 0.50,
}


def _candidate_catalog() -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(
        name: str,
        mutation: dict[str, Any],
        thesis: str,
        category: str,
    ) -> None:
        candidates.append(
            Candidate(
                name=name,
                stage="drawdown_mitigation_atomic",
                category=category,
                patch={**BALANCED_PATCH, **mutation},
                thesis=thesis,
                lineage="balanced Round-2 candidate plus one granular targeted mechanism",
            )
        )

    add(BALANCED_NAME, {}, "Balanced post-audit candidate.", "control")

    # Delayed entry confirmation: first isolate delay, then each causal check.
    for bars in (1, 2, 3):
        add(
            f"confirm__bars{bars}__delay_only",
            {"param_overrides.entry_confirmation_bars": bars},
            "Isolate the effect of waiting for completed bars before filling.",
            "entry_confirmation",
        )
    for current_r in (-0.10, 0.00, 0.10):
        add(
            f"confirm__bars1__current_{str(current_r).replace('-', 'm').replace('.', 'p')}",
            {
                "param_overrides.entry_confirmation_bars": 1,
                "param_overrides.entry_confirmation_min_current_r": current_r,
            },
            "Require the first completed post-signal bar to retain price progress.",
            "entry_confirmation",
        )
    for mfe_r in (0.05, 0.10, 0.20):
        add(
            f"confirm__bars1__mfe_{str(mfe_r).replace('.', 'p')}",
            {
                "param_overrides.entry_confirmation_bars": 1,
                "param_overrides.entry_confirmation_min_mfe_r": mfe_r,
            },
            "Require early favorable excursion before accepting the breakout.",
            "entry_confirmation",
        )
    for mae_r in (0.20, 0.35, 0.50, 0.65, 0.80):
        add(
            f"confirm__bars1__mae_{str(mae_r).replace('.', 'p')}",
            {
                "param_overrides.entry_confirmation_bars": 1,
                "param_overrides.entry_confirmation_max_mae_r": mae_r,
            },
            "Reject signals that immediately suffer excessive adverse excursion.",
            "entry_confirmation",
        )
    for suffix, extra in (
        ("above_breakout", {"param_overrides.entry_confirmation_require_above_breakout": True}),
        ("above_avwap", {"param_overrides.entry_confirmation_require_above_avwap": True}),
        (
            "above_breakout_avwap",
            {
                "param_overrides.entry_confirmation_require_above_breakout": True,
                "param_overrides.entry_confirmation_require_above_avwap": True,
            },
        ),
        ("rvol_ratio050", {"param_overrides.entry_confirmation_min_rvol_ratio": 0.50}),
        ("rvol_ratio075", {"param_overrides.entry_confirmation_min_rvol_ratio": 0.75}),
        ("rvol_ratio100", {"param_overrides.entry_confirmation_min_rvol_ratio": 1.00}),
    ):
        add(
            f"confirm__bars1__{suffix}",
            {"param_overrides.entry_confirmation_bars": 1, **extra},
            "Test one post-signal persistence discriminator.",
            "entry_confirmation",
        )
    for suffix, extra in (
        (
            "current0_breakout",
            {
                "param_overrides.entry_confirmation_min_current_r": 0.0,
                "param_overrides.entry_confirmation_require_above_breakout": True,
            },
        ),
        (
            "mfe010_mae035",
            {
                "param_overrides.entry_confirmation_min_mfe_r": 0.10,
                "param_overrides.entry_confirmation_max_mae_r": 0.35,
            },
        ),
        (
            "current0_breakout_rvol050",
            {
                "param_overrides.entry_confirmation_min_current_r": 0.0,
                "param_overrides.entry_confirmation_require_above_breakout": True,
                "param_overrides.entry_confirmation_min_rvol_ratio": 0.50,
            },
        ),
        (
            "mae050_size110",
            {
                "param_overrides.entry_confirmation_max_mae_r": 0.50,
                "param_overrides.entry_confirmation_size_mult": 1.10,
            },
        ),
        (
            "mae065_size105",
            {
                "param_overrides.entry_confirmation_max_mae_r": 0.65,
                "param_overrides.entry_confirmation_size_mult": 1.05,
            },
        ),
    ):
        add(
            f"confirm__bars1__{suffix}",
            {"param_overrides.entry_confirmation_bars": 1, **extra},
            "Test a compact conjunction of independently interpretable confirmation checks.",
            "entry_confirmation_composite",
        )

    # Post-entry maturation: tighten, rather than reject, when evidence fails.
    for bars in (2, 4, 6, 8):
        add(
            f"mature__bars{bars}__current0",
            {
                "param_overrides.maturation_stop_bars": bars,
                "param_overrides.maturation_stop_min_current_r": 0.0,
                "param_overrides.maturation_stop_to_r": -0.10,
            },
            "Tighten the stop when the trade has not made price progress.",
            "maturation",
        )
    for suffix, extra in (
        ("mfe010", {"param_overrides.maturation_stop_min_mfe_r": 0.10}),
        ("mfe020", {"param_overrides.maturation_stop_min_mfe_r": 0.20}),
        ("mae035", {"param_overrides.maturation_stop_max_mae_r": 0.35}),
        ("mae050", {"param_overrides.maturation_stop_max_mae_r": 0.50}),
        ("above_breakout", {"param_overrides.maturation_stop_require_above_breakout": True}),
        ("above_avwap", {"param_overrides.maturation_stop_require_above_avwap": True}),
        (
            "current0_breakout_fail2",
            {
                "param_overrides.maturation_stop_min_current_r": 0.0,
                "param_overrides.maturation_stop_require_above_breakout": True,
                "param_overrides.maturation_stop_min_failed_checks": 2,
            },
        ),
    ):
        add(
            f"mature__bars4__{suffix}",
            {
                "param_overrides.maturation_stop_bars": 4,
                "param_overrides.maturation_stop_to_r": -0.10,
                **extra,
            },
            "Isolate one causal maturation-failure check after four bars.",
            "maturation",
        )
    for stop_r in (-0.25, -0.40):
        add(
            f"mature__bars4__mfe010__stop_{str(stop_r).replace('-', 'm').replace('.', 'p')}",
            {
                "param_overrides.maturation_stop_bars": 4,
                "param_overrides.maturation_stop_min_mfe_r": 0.10,
                "param_overrides.maturation_stop_to_r": stop_r,
            },
            "Perturb the only comparatively viable maturation check with a less aggressive stop.",
            "maturation",
        )

    # Signal discrimination: quality gradients, geometry, time, and ADX contribution.
    for score_min in (55.0, 60.0, 62.5, 65.0, 67.5, 70.0):
        add(
            f"quality__hard_min_{str(score_min).replace('.', 'p')}",
            {
                "ablation.use_orb_quality_gate": True,
                "param_overrides.orb_quality_score_min": score_min,
            },
            "Hard-test the monotonic ORB quality relationship.",
            "signal_quality",
        )
    for floor in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
        add(
            f"quality__size_floor_{str(floor).replace('.', 'p')}",
            {
                "param_overrides.orb_quality_score_min": 55.0,
                "param_overrides.orb_quality_size_floor": floor,
                "param_overrides.orb_quality_top_score": 82.5,
                "param_overrides.orb_quality_top_mult": 1.10,
            },
            "Downweight low-quality signals continuously instead of deleting them.",
            "signal_quality_sizing",
        )
    add(
        "quality__size_floor_0p75__top105",
        {
            "param_overrides.orb_quality_score_min": 55.0,
            "param_overrides.orb_quality_size_floor": 0.75,
            "param_overrides.orb_quality_top_score": 82.5,
            "param_overrides.orb_quality_top_mult": 1.05,
        },
        "Perturb the high-quality uplift around the viable 0.75 sizing floor.",
        "signal_quality_sizing",
    )
    for risk_fraction in (0.00723, 0.00737):
        add(
            f"quality__size_floor_0p7__risk_{str(risk_fraction).replace('.', 'p')}",
            {
                "param_overrides.orb_quality_score_min": 55.0,
                "param_overrides.orb_quality_size_floor": 0.70,
                "param_overrides.orb_quality_top_score": 82.5,
                "param_overrides.orb_quality_top_mult": 1.10,
                "param_overrides.base_risk_fraction": risk_fraction,
            },
            "Restore a small portion of risk budget after robust continuous quality sizing.",
            "quality_risk_restore",
        )
    for cap in (0.75, 1.00, 1.15, 1.25, 1.35):
        add(
            f"geometry__entry_range_cap_{str(cap).replace('.', 'p')}",
            {
                "ablation.use_orb_entry_range_gate": True,
                "param_overrides.orb_entry_range_cap_r": cap,
            },
            "Reject unusually wide signal bars with poor stop geometry.",
            "signal_geometry",
        )
    for cap in (0.50, 0.75, 1.00):
        add(
            f"geometry__breakout_cap_{str(cap).replace('.', 'p')}",
            {
                "ablation.use_breakout_distance_cap": True,
                "param_overrides.breakout_distance_cap_r": cap,
            },
            "Reject entries too extended beyond their breakout reference.",
            "signal_geometry",
        )
    for cap in (0.0075, 0.0100, 0.0150):
        add(
            f"geometry__avwap_cap_{str(cap).replace('.', 'p')}",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": cap,
            },
            "Reject entries excessively extended above session value.",
            "signal_geometry",
        )
    for threshold in (12.5, 15.0, 17.5, 25.0, 30.0):
        add(
            f"score__adx_threshold_{int(threshold)}",
            {"param_overrides.adx_threshold": threshold},
            "Perturb the weak ADX score contribution without adding a hard gate.",
            "score_discrimination",
        )
    for mult in (0.80, 0.90, 1.00):
        add(
            f"score__six_plus_size_{str(mult).replace('.', 'p')}",
            {
                "param_overrides.entry_score_size_mults": {
                    **BASE_SCORE_MULTS,
                    "*:6": mult,
                    "*:7": mult,
                }
            },
            "Test whether high score sizing, rather than acceptance, amplifies bad regimes.",
            "score_discrimination",
        )
    for suffix, mutation in (
        (
            "late1030_size075",
            {
                "param_overrides.late_entry_cutoff": clock_time(10, 30),
                "param_overrides.late_entry_size_mult": 0.75,
            },
        ),
        (
            "late1030_score5",
            {
                "param_overrides.late_entry_cutoff": clock_time(10, 30),
                "param_overrides.late_entry_score_min": 5,
            },
        ),
        (
            "late1030_rvol_add010",
            {
                "param_overrides.orb_time_decay_start": clock_time(10, 30),
                "param_overrides.orb_late_rvol_add_per_30m": 0.10,
            },
        ),
    ):
        add(
            f"time__{suffix}",
            mutation,
            "Test discrimination in the weak post-10:30 signal cohort.",
            "time_discrimination",
        )

    # Additional entry mechanisms: retest/reclaim plus broader selection/capacity.
    for mode in ("or", "or_avwap", "or_pdh", "or_pdh_avwap"):
        add(
            f"reclaim__{mode}__default",
            {"param_overrides.reclaim_entry_mode": mode},
            "Add causal retest/reclaim entries while preserving direct breakout fallback.",
            "entry_expansion",
        )
    for min_rvol in (1.70, 2.25):
        add(
            f"reclaim__or_avwap__rvol_{str(min_rvol).replace('.', 'p')}",
            {
                "param_overrides.reclaim_entry_mode": "or_avwap",
                "param_overrides.reclaim_min_rvol": min_rvol,
            },
            "Perturb reclaim-specific RVOL independently.",
            "entry_expansion",
        )
    for cpr in (0.55, 0.60, 0.65):
        add(
            f"reclaim__or_avwap__cpr_{str(cpr).replace('.', 'p')}",
            {
                "param_overrides.reclaim_entry_mode": "or_avwap",
                "param_overrides.reclaim_cpr_threshold": cpr,
            },
            "Perturb reclaim bar closing-location quality.",
            "entry_expansion",
        )
    add(
        "reclaim__or_avwap__structure_stop",
        {
            "param_overrides.reclaim_entry_mode": "or_avwap",
            "param_overrides.orb_structure_stop_mode": "reclaim",
        },
        "Pair reclaim entries with their natural retest support stop.",
        "entry_expansion",
    )
    for min_risk in (0.75, 0.90):
        add(
            f"reclaim__or_avwap__structure_stop__minrisk_{str(min_risk).replace('.', 'p')}",
            {
                "param_overrides.reclaim_entry_mode": "or_avwap",
                "param_overrides.orb_structure_stop_mode": "reclaim",
                "param_overrides.orb_structure_min_risk_pct": min_risk,
            },
            "Widen the retest-aware stop to test whether tight normalized risk causes reclaim churn.",
            "entry_expansion",
        )
    add(
        "reclaim__or_avwap__structure_stop__buffer_0p0025",
        {
            "param_overrides.reclaim_entry_mode": "or_avwap",
            "param_overrides.orb_structure_stop_mode": "reclaim",
            "param_overrides.orb_structure_stop_buffer_pct": 0.0025,
        },
        "Place the reclaim stop farther beyond support to reduce noise exits.",
        "entry_expansion",
    )
    for count in (25, 30):
        add(
            f"selection__long_count_{count}",
            {"param_overrides.selection_long_count": count},
            "Expand the daily long candidate set to test scanner alpha left on the table.",
            "selection_expansion",
        )

    # Trade management and exits.
    add(
        "exit__flow_reversal_off",
        {"ablation.use_flow_reversal_exit": False},
        "Ablate the loss-only flow-reversal exit and let normal stops manage those trades.",
        "exit_logic",
    )
    for bars in (6, 8, 10, 16, 24):
        add(
            f"exit__flow_hold_{bars}",
            {"param_overrides.flow_reversal_min_hold_bars": bars},
            "Perturb the flow-reversal grace period.",
            "exit_logic",
        )
    add(
        "exit__flow_require_below_entry",
        {"param_overrides.flow_reversal_require_below_entry": True},
        "Require actual loss of entry before the flow-reversal exit can fire.",
        "exit_logic",
    )
    for bars, minimum in ((4, -0.25), (6, -0.25), (6, 0.00), (8, 0.00)):
        add(
            f"exit__quick_{bars}_min_{str(minimum).replace('-', 'm').replace('.', 'p')}",
            {
                "ablation.use_time_based_quick_exit": True,
                "param_overrides.quick_exit_max_bars": bars,
                "param_overrides.quick_exit_min_r": minimum,
            },
            "Exit breakouts that fail to establish progress within a fixed early window.",
            "early_exit",
        )
    for bars in (12, 20):
        add(
            f"exit__mfe_check_{bars}",
            {"param_overrides.mfe_conviction_check_bars": bars},
            "Perturb the existing MFE conviction checkpoint.",
            "exit_logic",
        )
    add(
        "exit__mfe_conviction_off",
        {"ablation.use_mfe_conviction_exit": False},
        "Ablate the existing MFE conviction exit.",
        "exit_logic",
    )
    for start in (10, 15, 20):
        add(
            f"exit__retrace_start_{start}",
            {
                "ablation.use_orb_retracement_trail": True,
                "param_overrides.orb_retracement_trail_start_bars": start,
                "param_overrides.orb_retracement_trail_tighten_bars": 25,
            },
            "Preserve a fraction of established MFE before the current late trail.",
            "exit_logic",
        )
    for tighten in (20, 30):
        add(
            f"exit__adaptive_tighten_{tighten}",
            {"param_overrides.adaptive_trail_tighten_bars": tighten},
            "Perturb when the accepted tight late trail begins.",
            "exit_logic",
        )

    # Portfolio constraints and sizing: explicitly test the positive rejected-shadow cohorts.
    for positions in (7, 8):
        add(
            f"capacity__max_positions_{positions}",
            {"param_overrides.max_positions": positions},
            "Allow one or two additional concurrent positions.",
            "capacity",
        )
    add(
        "capacity__sector_limit_4",
        {"param_overrides.max_positions_per_sector": 4},
        "Relax the sector slot limit by one.",
        "capacity",
    )
    for leverage in (2.50, 3.00):
        add(
            f"capacity__leverage_{str(leverage).replace('.', 'p')}",
            {"param_overrides.intraday_leverage": leverage},
            "Relax buying-power capacity while retaining heat and position limits.",
            "capacity",
        )
    for mult in (0.75, 0.85):
        add(
            f"risk__thursday_mult_{str(mult).replace('.', 'p')}",
            {"param_overrides.thursday_sizing_mult": mult},
            "Downweight the weak Thursday cohort without blocking it.",
            "calendar_risk",
        )
    add(
        "risk__industrials_mult_025",
        {"param_overrides.sector_mult_industrials": 0.25},
        "Further reduce the only negative aggregate sector.",
        "sector_risk",
    )

    return candidates


def _baseline_metrics(prior_rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    row = next(item for item in prior_rows if item["name"] == BALANCED_REVIEW_NAME)
    return {
        "total_trades": row[f"{prefix}_total_trades"],
        "win_rate": row[f"{prefix}_win_rate"],
        "profit_factor": row[f"{prefix}_profit_factor"],
        "net_profit": row[f"{prefix}_net_profit"],
        "expectancy": row[f"{prefix}_expectancy"],
        "expected_total_r": row[f"{prefix}_expected_total_r"],
        "trades_per_month": row[f"{prefix}_trades_per_month"],
        "max_drawdown_pct": row[f"{prefix}_max_drawdown_pct"],
    }


def _metrics(results: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return dict(results.get(name, {}).get("metrics") or {})


def _ratio(value: Any, baseline: Any) -> float:
    try:
        denominator = float(baseline)
        return float(value) / denominator if abs(denominator) > 1e-12 else 1.0
    except (TypeError, ValueError):
        return 0.0


def _preliminary_score(
    is_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    baseline_is: dict[str, Any],
    baseline_oos: dict[str, Any],
) -> float:
    return (
        0.24 * _ratio(oos_metrics.get("expected_total_r"), baseline_oos.get("expected_total_r"))
        + 0.16 * _ratio(oos_metrics.get("net_profit"), baseline_oos.get("net_profit"))
        + 0.13 * _ratio(oos_metrics.get("trades_per_month"), baseline_oos.get("trades_per_month"))
        + 0.10 * _ratio(oos_metrics.get("profit_factor"), baseline_oos.get("profit_factor"))
        + 0.14 * _ratio(is_metrics.get("expected_total_r"), baseline_is.get("expected_total_r"))
        + 0.09 * _ratio(is_metrics.get("net_profit"), baseline_is.get("net_profit"))
        + 0.07 * _ratio(is_metrics.get("trades_per_month"), baseline_is.get("trades_per_month"))
        + 0.07 * _ratio(is_metrics.get("profit_factor"), baseline_is.get("profit_factor"))
        - 0.10
        * max(
            0.0,
            _ratio(is_metrics.get("max_drawdown_pct"), baseline_is.get("max_drawdown_pct"))
            - 1.0,
        )
    )


def _screen_rows(
    candidates: list[Candidate],
    is_results: dict[str, dict[str, Any]],
    oos_results: dict[str, dict[str, Any]],
    baseline_is: dict[str, Any],
    baseline_oos: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        ism = _metrics(is_results, candidate.name)
        oosm = _metrics(oos_results, candidate.name)
        if not ism or not oosm:
            continue
        is_guardrail = (
            _ratio(ism.get("expected_total_r"), baseline_is.get("expected_total_r")) >= 0.90
            and _ratio(ism.get("net_profit"), baseline_is.get("net_profit")) >= 0.90
            and _ratio(ism.get("trades_per_month"), baseline_is.get("trades_per_month")) >= 0.85
            and _ratio(ism.get("profit_factor"), baseline_is.get("profit_factor")) >= 0.90
            and _ratio(ism.get("max_drawdown_pct"), baseline_is.get("max_drawdown_pct")) <= 1.05
        )
        oos_floor = (
            _ratio(oosm.get("expected_total_r"), baseline_oos.get("expected_total_r")) >= 0.85
            and _ratio(oosm.get("trades_per_month"), baseline_oos.get("trades_per_month")) >= 0.80
            and _ratio(oosm.get("profit_factor"), baseline_oos.get("profit_factor")) >= 0.85
        )
        strict_joint_uplift = (
            ism.get("expected_total_r", 0) >= baseline_is.get("expected_total_r", 0)
            and ism.get("net_profit", 0) >= baseline_is.get("net_profit", 0)
            and ism.get("trades_per_month", 0) >= baseline_is.get("trades_per_month", 0)
            and oosm.get("expected_total_r", 0) >= baseline_oos.get("expected_total_r", 0)
            and oosm.get("net_profit", 0) >= baseline_oos.get("net_profit", 0)
            and oosm.get("trades_per_month", 0) >= baseline_oos.get("trades_per_month", 0)
        )
        rows.append(
            {
                "name": candidate.name,
                "category": candidate.category,
                "thesis": candidate.thesis,
                "patch": candidate.patch,
                "is_expected_total_r": ism.get("expected_total_r"),
                "is_net_profit": ism.get("net_profit"),
                "is_trades_per_month": ism.get("trades_per_month"),
                "is_profit_factor": ism.get("profit_factor"),
                "is_max_drawdown_pct": ism.get("max_drawdown_pct"),
                "oos_expected_total_r": oosm.get("expected_total_r"),
                "oos_net_profit": oosm.get("net_profit"),
                "oos_trades_per_month": oosm.get("trades_per_month"),
                "oos_profit_factor": oosm.get("profit_factor"),
                "oos_max_drawdown_pct": oosm.get("max_drawdown_pct"),
                "is_guardrail": is_guardrail,
                "oos_floor": oos_floor,
                "strict_joint_uplift": strict_joint_uplift,
                "preliminary_score": _preliminary_score(ism, oosm, baseline_is, baseline_oos),
            }
        )
    return sorted(rows, key=lambda row: row["preliminary_score"], reverse=True)


def _build_combinations(
    screen: list[dict[str, Any]],
    catalog: dict[str, Candidate],
) -> list[Candidate]:
    eligible = [
        row
        for row in screen
        if row["name"] != BALANCED_NAME and row["is_guardrail"] and row["oos_floor"]
    ]
    category_best: dict[str, dict[str, Any]] = {}
    for row in eligible:
        category_best.setdefault(row["category"], row)

    pairs = (
        ("signal_quality_sizing", "entry_confirmation"),
        ("signal_quality", "entry_confirmation"),
        ("signal_geometry", "entry_confirmation"),
        ("signal_geometry", "signal_quality_sizing"),
        ("signal_geometry", "quality_risk_restore"),
        ("signal_geometry", "exit_logic"),
        ("signal_geometry", "score_discrimination"),
        ("signal_geometry", "selection_expansion"),
        ("maturation", "exit_logic"),
        ("entry_expansion", "signal_quality_sizing"),
        ("entry_expansion", "signal_geometry"),
        ("entry_expansion", "exit_logic"),
        ("entry_expansion", "capacity"),
        ("selection_expansion", "signal_quality_sizing"),
        ("selection_expansion", "signal_geometry"),
        ("capacity", "signal_quality_sizing"),
        ("signal_quality_sizing", "exit_logic"),
        ("score_discrimination", "entry_confirmation"),
        ("time_discrimination", "entry_confirmation"),
    )
    output: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for left_category, right_category in pairs:
        left = category_best.get(left_category)
        right = category_best.get(right_category)
        if not left or not right or left["name"] == right["name"]:
            continue
        pair_key = tuple(sorted((left["name"], right["name"])))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        left_candidate = catalog[left["name"]]
        right_candidate = catalog[right["name"]]
        merged = {**left_candidate.patch, **right_candidate.patch}
        output.append(
            Candidate(
                name=f"combo__{left_category}__{right_category}",
                stage="drawdown_mitigation_combination",
                category="orthogonal_combination",
                patch=merged,
                thesis=f"Combine {left['name']} with {right['name']}.",
                lineage=f"{left['name']} + {right['name']}",
            )
        )
    triples = (
        ("signal_geometry", "signal_quality_sizing", "exit_logic"),
        ("signal_geometry", "quality_risk_restore", "exit_logic"),
        ("signal_geometry", "signal_quality_sizing", "score_discrimination"),
        ("signal_geometry", "signal_quality_sizing", "entry_expansion"),
        ("signal_geometry", "signal_quality_sizing", "selection_expansion"),
    )
    for categories in triples:
        rows = [category_best.get(category) for category in categories]
        if any(row is None for row in rows):
            continue
        assert all(row is not None for row in rows)
        names = [row["name"] for row in rows]
        if len(set(names)) != len(names):
            continue
        merged: dict[str, Any] = {}
        for name in names:
            merged.update(catalog[name].patch)
        output.append(
            Candidate(
                name=f"combo__{'__'.join(categories)}",
                stage="drawdown_mitigation_combination",
                category="orthogonal_combination",
                patch=merged,
                thesis=f"Combine {' + '.join(names)}.",
                lineage=" + ".join(names),
            )
        )
    balanced_specs = (
        (
            "combo_balanced__geom125__quality070",
            ("geometry__entry_range_cap_1p25", "quality__size_floor_0p7"),
        ),
        (
            "combo_balanced__geom125__adx15",
            ("geometry__entry_range_cap_1p25", "score__adx_threshold_15"),
        ),
        (
            "combo_balanced__geom125__flow8",
            ("geometry__entry_range_cap_1p25", "exit__flow_hold_8"),
        ),
        (
            "combo_balanced__geom125__selection30",
            ("geometry__entry_range_cap_1p25", "selection__long_count_30"),
        ),
        (
            "combo_balanced__geom125__quality070__selection30",
            (
                "geometry__entry_range_cap_1p25",
                "quality__size_floor_0p7",
                "selection__long_count_30",
            ),
        ),
        (
            "combo_balanced__geom125__quality070__flow8",
            (
                "geometry__entry_range_cap_1p25",
                "quality__size_floor_0p7",
                "exit__flow_hold_8",
            ),
        ),
        (
            "combo_balanced__geom125__selection30__flow8",
            (
                "geometry__entry_range_cap_1p25",
                "selection__long_count_30",
                "exit__flow_hold_8",
            ),
        ),
        (
            "combo_balanced__geom115__selection30__flow8",
            (
                "geometry__entry_range_cap_1p15",
                "selection__long_count_30",
                "exit__flow_hold_8",
            ),
        ),
        (
            "combo_balanced__selection30__flow8",
            (
                "selection__long_count_30",
                "exit__flow_hold_8",
            ),
        ),
        (
            "combo_balanced__geom125__selection30__mfeoff",
            (
                "geometry__entry_range_cap_1p25",
                "selection__long_count_30",
                "exit__mfe_conviction_off",
            ),
        ),
        (
            "combo_balanced__geom125__selection30__flow8__mfeoff",
            (
                "geometry__entry_range_cap_1p25",
                "selection__long_count_30",
                "exit__flow_hold_8",
                "exit__mfe_conviction_off",
            ),
        ),
        (
            "combo_balanced__geom115__selection30__flow8__mfeoff",
            (
                "geometry__entry_range_cap_1p15",
                "selection__long_count_30",
                "exit__flow_hold_8",
                "exit__mfe_conviction_off",
            ),
        ),
    )
    for name, lineage_names in balanced_specs:
        if any(lineage_name not in catalog for lineage_name in lineage_names):
            continue
        merged: dict[str, Any] = {}
        for lineage_name in lineage_names:
            merged.update(catalog[lineage_name].patch)
        output.append(
            Candidate(
                name=name,
                stage="drawdown_mitigation_balanced_combination",
                category="balanced_geometry_combination",
                patch=merged,
                thesis="Combine the stable 1.25R geometry anchor with validated orthogonal mechanisms.",
                lineage=" + ".join(lineage_names),
            )
        )
    return output


def _validation_candidates(
    screen: list[dict[str, Any]],
    catalog: dict[str, Candidate],
    limit: int,
) -> list[Candidate]:
    selected_names: list[str] = [BALANCED_NAME]
    eligible = [
        row
        for row in screen
        if row["name"] != BALANCED_NAME and row["is_guardrail"] and row["oos_floor"]
    ]
    selected_names.extend(row["name"] for row in eligible[:limit])
    selected_names.extend(
        row["name"]
        for row in eligible
        if catalog[row["name"]].category == "balanced_geometry_combination"
    )
    category_counts: dict[str, int] = {}
    for row in eligible:
        count = category_counts.get(row["category"], 0)
        if count < 2:
            selected_names.append(row["name"])
            category_counts[row["category"]] = count + 1
    unique = list(dict.fromkeys(selected_names))
    return [catalog[name] for name in unique if name in catalog]


def _validated_rows(
    candidates: list[Candidate],
    screen_by_name: dict[str, dict[str, Any]],
    early: dict[str, dict[str, Any]],
    late: dict[str, dict[str, Any]],
    stress: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_early = _metrics(early, BALANCED_NAME)
    baseline_late = _metrics(late, BALANCED_NAME)
    baseline_stress = _metrics(stress, BALANCED_NAME)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        base = dict(screen_by_name.get(candidate.name, {}))
        em = _metrics(early, candidate.name)
        lm = _metrics(late, candidate.name)
        sm = _metrics(stress, candidate.name)
        if not base or not em or not lm or not sm:
            continue
        early_r = _ratio(em.get("expected_total_r"), baseline_early.get("expected_total_r"))
        late_r = _ratio(lm.get("expected_total_r"), baseline_late.get("expected_total_r"))
        stress_r_delta = float(sm.get("expected_total_r", 0)) - float(
            baseline_stress.get("expected_total_r", 0)
        )
        stress_dd_ratio = _ratio(
            sm.get("max_drawdown_pct"), baseline_stress.get("max_drawdown_pct")
        )
        segment_floor = min(early_r, late_r)
        validation_score = (
            base["preliminary_score"]
            + 0.10 * segment_floor
            + 0.05 * (early_r + late_r)
            + 0.04 * stress_r_delta
            - 0.08 * max(0.0, stress_dd_ratio - 1.0)
        )
        base.update(
            {
                "early_expected_total_r": em.get("expected_total_r"),
                "early_trades_per_month": em.get("trades_per_month"),
                "early_r_ratio": early_r,
                "late_expected_total_r": lm.get("expected_total_r"),
                "late_trades_per_month": lm.get("trades_per_month"),
                "late_r_ratio": late_r,
                "segment_floor": segment_floor,
                "stress_expected_total_r": sm.get("expected_total_r"),
                "stress_profit_factor": sm.get("profit_factor"),
                "stress_max_drawdown_pct": sm.get("max_drawdown_pct"),
                "stress_r_delta": stress_r_delta,
                "stress_dd_ratio": stress_dd_ratio,
                "both_is_halves_90pct": segment_floor >= 0.90,
                "stress_improved": stress_r_delta > 0 and stress_dd_ratio <= 1.0,
                "validation_score": validation_score,
            }
        )
        rows.append(base)
    return sorted(rows, key=lambda row: row["validation_score"], reverse=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [key for key in rows[0] if key not in {"patch", "thesis"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report(
    screen: list[dict[str, Any]],
    validated: list[dict[str, Any]],
    atomic_count: int,
    combination_count: int,
) -> str:
    lines = [
        "# ALCB Round 2 drawdown mitigation and alpha recovery",
        "",
        "Diagnostic-only repaired-cache research. OOS has been consumed and is not a fresh lockbox.",
        "",
        f"- Atomic candidates: {atomic_count}",
        f"- Orthogonal follow-up combinations: {combination_count}",
        "- Baseline: RVOL 1.70 / OR 9 / late trail distance 0.04.",
        "",
        "## Leading aggregate candidates",
        "",
        "| Candidate | Category | IS R | IS freq | IS PF | IS DD | OOS R | OOS freq | OOS PF | OOS DD | Guardrails |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in screen[:30]:
        lines.append(
            f"| {row['name']} | {row['category']} | {row['is_expected_total_r']:.2f} | "
            f"{row['is_trades_per_month']:.2f} | {row['is_profit_factor']:.2f} | "
            f"{100*row['is_max_drawdown_pct']:.2f}% | {row['oos_expected_total_r']:.2f} | "
            f"{row['oos_trades_per_month']:.2f} | {row['oos_profit_factor']:.2f} | "
            f"{100*row['oos_max_drawdown_pct']:.2f}% | "
            f"{'pass' if row['is_guardrail'] and row['oos_floor'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Segment and drawdown-stress validation",
            "",
            "| Candidate | Early R ratio | Late R ratio | Stress R delta | Stress DD ratio | OOS R | OOS freq |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validated[:30]:
        lines.append(
            f"| {row['name']} | {row['early_r_ratio']:.3f} | {row['late_r_ratio']:.3f} | "
            f"{row['stress_r_delta']:+.2f} | {row['stress_dd_ratio']:.3f} | "
            f"{row['oos_expected_total_r']:.2f} | {row['oos_trades_per_month']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Automated scores are triage aids, not promotion decisions. Mechanism coherence, local smoothness, "
            "segment stability, and complexity are reviewed separately.",
            "",
            "No production configuration was modified.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prior-review", type=Path, default=PRIOR_REVIEW)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.allow_legacy_data:
        raise SystemExit(
            "Pass --allow-legacy-data; no authoritative frozen direct-RTH bundle is available."
        )
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = _load_json(BASE_CONFIG_PATH)
    prior_rows = _load_json(args.prior_review.resolve() / "all_results.json")
    baseline_is = _baseline_metrics(prior_rows, "is")
    baseline_oos = _baseline_metrics(prior_rows, "oos")

    atomic = _candidate_catalog()
    atomic_catalog = {candidate.name: candidate for candidate in atomic}
    _write_json(output / "atomic_candidate_catalog.json", [asdict(row) for row in atomic])
    _write_json(
        output / "run_spec.json",
        {
            "generated_at_utc": datetime.now(timezone.utc),
            "atomic_candidate_count": len(atomic),
            "windows": {
                "is": [IS_START, IS_END],
                "oos": [OOS_START, OOS_END],
                "early_is": EARLY_IS,
                "late_is": LATE_IS,
                "max_drawdown_descent": MAX_DD_DESCENT,
            },
            "base_config": str(BASE_CONFIG_PATH),
            "balanced_patch": BALANCED_PATCH,
            "data_authority": "diagnostic-only repaired legacy filename cache",
            "promotion_authorized": False,
        },
    )

    atomic_oos = _evaluate_candidates(
        atomic,
        base,
        start=OOS_START,
        end=OOS_END,
        max_workers=args.max_workers,
        output_path=output / "atomic_oos_results.json",
        baseline_metrics=baseline_oos,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    atomic_is = _evaluate_candidates(
        atomic,
        base,
        start=IS_START,
        end=IS_END,
        max_workers=args.max_workers,
        output_path=output / "atomic_is_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    atomic_screen = _screen_rows(atomic, atomic_is, atomic_oos, baseline_is, baseline_oos)
    _write_json(output / "atomic_screen.json", atomic_screen)

    combinations = _build_combinations(atomic_screen, atomic_catalog)
    _write_json(output / "combination_candidate_catalog.json", [asdict(row) for row in combinations])
    combination_oos = _evaluate_candidates(
        combinations,
        base,
        start=OOS_START,
        end=OOS_END,
        max_workers=args.max_workers,
        output_path=output / "combination_oos_results.json",
        baseline_metrics=baseline_oos,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    combination_is = _evaluate_candidates(
        combinations,
        base,
        start=IS_START,
        end=IS_END,
        max_workers=args.max_workers,
        output_path=output / "combination_is_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    combined_screen = _screen_rows(
        combinations, combination_is, combination_oos, baseline_is, baseline_oos
    )
    screen = sorted(
        [*atomic_screen, *combined_screen],
        key=lambda row: row["preliminary_score"],
        reverse=True,
    )
    _write_json(output / "aggregate_screen.json", screen)
    _write_csv(output / "aggregate_screen.csv", screen)

    catalog = {**atomic_catalog, **{candidate.name: candidate for candidate in combinations}}
    validation_candidates = _validation_candidates(screen, catalog, args.validation_limit)
    _write_json(
        output / "validation_candidate_catalog.json",
        [asdict(row) for row in validation_candidates],
    )
    early = _evaluate_candidates(
        validation_candidates,
        base,
        start=EARLY_IS[0],
        end=EARLY_IS[1],
        max_workers=args.max_workers,
        output_path=output / "early_is_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    late = _evaluate_candidates(
        validation_candidates,
        base,
        start=LATE_IS[0],
        end=LATE_IS[1],
        max_workers=args.max_workers,
        output_path=output / "late_is_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    stress = _evaluate_candidates(
        validation_candidates,
        base,
        start=MAX_DD_DESCENT[0],
        end=MAX_DD_DESCENT[1],
        max_workers=args.max_workers,
        output_path=output / "max_dd_stress_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    validated = _validated_rows(
        validation_candidates,
        {row["name"]: row for row in screen},
        early,
        late,
        stress,
    )
    _write_json(output / "validated_results.json", validated)
    _write_csv(output / "validated_results.csv", validated)
    (output / "report.md").write_text(
        _report(screen, validated, len(atomic), len(combinations)),
        encoding="utf-8",
    )
    _write_json(
        output / "completion.json",
        {
            "completed_at_utc": datetime.now(timezone.utc),
            "atomic_candidate_count": len(atomic),
            "combination_candidate_count": len(combinations),
            "validation_candidate_count": len(validation_candidates),
            "promotion_authorized": False,
            "promotion_blocker": (
                "OOS was reused for targeted research and authoritative frozen data is unavailable."
            ),
        },
    )
    print(f"complete: {output}", flush=True)
    print(
        f"atomic={len(atomic)} combinations={len(combinations)} "
        f"validated={len(validation_candidates)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
