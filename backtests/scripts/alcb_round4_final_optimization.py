"""Final ALCB Round-4 risk/quality optimization and robustness validation.

The search is anchored to the previously selected balanced control and targets
the remaining weaknesses: high IS drawdown, low IS profit factor, backtest/live
daily-stop parity, causal failure-cluster throttling, and hard-versus-continuous
entry geometry.  The March-May 2026 OOS window is consumed development data;
results are retained for comparison but are not treated as a fresh lockbox.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from backtests.scripts.alcb_round2_drawdown_diagnostics import (  # noqa: E402
    BALANCED_PATCH,
    _daily_equity,
    _drawdown_episodes,
    _loss_concentration,
)
from backtests.scripts.alcb_round2_oos_robustness import (  # noqa: E402
    BASE_CONFIG_PATH,
    Candidate,
    INITIAL_EQUITY,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    REPO_ROOT,
    _evaluate_candidates,
    _load_json,
    _metric_subset,
    _trade_to_dict,
    _write_json,
)
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "backtests"
    / "output"
    / "stock"
    / "alcb"
    / "round_4"
    / "final_optimization_20260723"
)
PRIOR_VALIDATED = (
    REPO_ROOT
    / "backtests"
    / "output"
    / "stock"
    / "alcb"
    / "round_4"
    / "oos_robustness_20260722"
    / "drawdown_mitigation_20260723"
    / "validated_results.json"
)
DATA_DIR = REPO_ROOT / "backtests" / "stock" / "data" / "raw"
EARLY_IS = ("2024-03-25", "2025-03-24")
LATE_IS = ("2025-03-25", "2026-03-01")
STRESS = ("2024-07-19", "2024-09-30")
BALANCED_NAME = "control__balanced"
PREVIOUS_NAME = "control__previous_balanced_candidate"


def _slug(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _candidate_catalog() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()

    def add(name: str, patch: dict[str, Any], category: str, thesis: str) -> None:
        if name in seen:
            return
        seen.add(name)
        candidates.append(
            Candidate(
                name=name,
                stage="round4_final_optimization",
                category=category,
                patch={**BALANCED_PATCH, **patch},
                thesis=thesis,
                lineage="full accepted Round-2 configuration + balanced control + targeted causal change",
            )
        )

    add(BALANCED_NAME, {}, "control", "Previously selected balanced RVOL/OR/trail control.")
    add(
        PREVIOUS_NAME,
        {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 1.25,
            "param_overrides.selection_long_count": 30,
            "param_overrides.flow_reversal_min_hold_bars": 8,
        },
        "control",
        "Balanced drawdown candidate from the preceding phase.",
    )

    # Hard geometry surface, including scanner and flow interactions.
    for cap in (0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25):
        for selection in (20, 30):
            base = {
                "ablation.use_orb_entry_range_gate": True,
                "param_overrides.orb_entry_range_cap_r": cap,
                "param_overrides.selection_long_count": selection,
            }
            add(
                f"hard__cap{_slug(cap)}__sel{selection}",
                base,
                "hard_geometry",
                "Map the smooth risk/alpha response of completed signal-bar range geometry.",
            )
            add(
                f"hard__cap{_slug(cap)}__sel{selection}__flow8",
                {**base, "param_overrides.flow_reversal_min_hold_bars": 8},
                "hard_geometry_flow",
                "Combine range discrimination with the locally stable eight-bar flow grace.",
            )
    for cap in (0.90, 1.00, 1.10):
        for selection in (25, 35, 40):
            add(
                f"hard__cap{_slug(cap)}__sel{selection}",
                {
                    "ablation.use_orb_entry_range_gate": True,
                    "param_overrides.orb_entry_range_cap_r": cap,
                    "param_overrides.selection_long_count": selection,
                },
                "hard_geometry_selection",
                "Test whether wider scanner extraction restores alpha removed by geometry gating.",
            )

    # Continuous geometry sizing avoids relying on a single hard boundary.
    for start, end in ((0.65, 1.15), (0.75, 1.15), (0.75, 1.25), (0.85, 1.25), (0.95, 1.35)):
        for floor in (0.40, 0.60, 0.80):
            add(
                f"taper__s{_slug(start)}__e{_slug(end)}__f{_slug(floor)}__sel30",
                {
                    "param_overrides.orb_entry_range_taper_start_r": start,
                    "param_overrides.orb_entry_range_taper_end_r": end,
                    "param_overrides.orb_entry_range_taper_floor": floor,
                    "param_overrides.selection_long_count": 30,
                },
                "continuous_geometry",
                "Downweight extended completed-bar entries continuously instead of deleting them.",
            )

    anchors = {
        "balanced": {},
        "cap075_sel20_flow8": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 0.75,
            "param_overrides.selection_long_count": 20,
            "param_overrides.flow_reversal_min_hold_bars": 8,
        },
        "cap085_sel30": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 0.85,
            "param_overrides.selection_long_count": 30,
        },
        "cap085_sel30_flow8": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 0.85,
            "param_overrides.selection_long_count": 30,
            "param_overrides.flow_reversal_min_hold_bars": 8,
        },
        "cap090_sel30": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 0.90,
            "param_overrides.selection_long_count": 30,
        },
        "cap100_sel30": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 1.00,
            "param_overrides.selection_long_count": 30,
        },
        "cap100_sel30_flow8": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 1.00,
            "param_overrides.selection_long_count": 30,
            "param_overrides.flow_reversal_min_hold_bars": 8,
        },
        "cap115_sel30_flow8": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 1.15,
            "param_overrides.selection_long_count": 30,
            "param_overrides.flow_reversal_min_hold_bars": 8,
        },
        "cap125_sel30_flow8": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 1.25,
            "param_overrides.selection_long_count": 30,
            "param_overrides.flow_reversal_min_hold_bars": 8,
        },
        "cap125_sel30": {
            "ablation.use_orb_entry_range_gate": True,
            "param_overrides.orb_entry_range_cap_r": 1.25,
            "param_overrides.selection_long_count": 30,
        },
    }

    # Exact causal daily-stop parity, evaluated both alone and with geometry.
    for anchor_name, anchor in anchors.items():
        for stop in (2.00, 2.35, 2.75, 3.00, 3.50):
            add(
                f"daily__{anchor_name}__stop{_slug(stop)}",
                {
                    **anchor,
                    "ablation.use_daily_stop": True,
                    "param_overrides.daily_stop_r": stop,
                },
                "daily_stop",
                "Block only subsequent entries after causally realized daily losses reach the threshold.",
            )

    # Rolling failure density uses only previously closed trades.
    density_specs = (
        (12, 0.50, 0.50),
        (12, 0.58, 0.70),
        (12, 0.66, 0.85),
        (20, 0.50, 0.50),
        (20, 0.58, 0.70),
        (20, 0.66, 0.85),
        (30, 0.50, 0.50),
        (30, 0.58, 0.70),
        (30, 0.66, 0.85),
    )
    for anchor_name in ("balanced", "cap100_sel30", "cap115_sel30_flow8"):
        anchor = anchors[anchor_name]
        for lookback, trigger, mult in density_specs:
            add(
                f"density__{anchor_name}__l{lookback}__t{_slug(trigger)}__m{_slug(mult)}",
                {
                    **anchor,
                    "param_overrides.failure_density_lookback_trades": lookback,
                    "param_overrides.failure_density_min_observations": min(8, lookback),
                    "param_overrides.failure_density_mfe_threshold_r": 0.20,
                    "param_overrides.failure_density_trigger_pct": trigger,
                    "param_overrides.failure_density_size_mult": mult,
                },
                "failure_density",
                "Reduce risk after a causal rolling cluster of losing low-MFE breakouts.",
            )

    # Continuous quality sizing with tighter geometry, plus isolated unlevered reclaim.
    for cap in (0.90, 1.00, 1.10):
        for floor in (0.60, 0.70, 0.80):
            add(
                f"quality__cap{_slug(cap)}__floor{_slug(floor)}__sel30",
                {
                    "ablation.use_orb_entry_range_gate": True,
                    "param_overrides.orb_entry_range_cap_r": cap,
                    "param_overrides.selection_long_count": 30,
                    "param_overrides.orb_quality_score_min": 55.0,
                    "param_overrides.orb_quality_size_floor": floor,
                    "param_overrides.orb_quality_top_score": 82.5,
                    "param_overrides.orb_quality_top_mult": 1.10,
                },
                "quality_geometry",
                "Use geometry for rejection and the existing composite only for continuous sizing.",
            )
    for mode in ("or", "or_avwap", "or_pdh_avwap"):
        add(
            f"reclaim__{mode}__unlevered",
            {
                "param_overrides.reclaim_entry_mode": mode,
                "param_overrides.intraday_leverage": 2.0,
            },
            "reclaim_isolation",
            "Retest causal reclaim alpha without leverage or capacity expansion.",
        )
    retained = {
        BALANCED_NAME,
        PREVIOUS_NAME,
        "hard__cap0p75__sel20",
        "hard__cap0p75__sel20__flow8",
        "hard__cap0p75__sel30",
        "hard__cap0p75__sel30__flow8",
        "hard__cap0p85__sel20",
        "hard__cap0p85__sel20__flow8",
        "hard__cap0p85__sel30",
        "hard__cap0p85__sel30__flow8",
        "hard__cap0p9__sel20",
        "hard__cap0p9__sel20__flow8",
        "hard__cap0p9__sel30",
        "hard__cap0p9__sel30__flow8",
        "hard__cap0p95__sel30",
        "hard__cap0p95__sel30__flow8",
        "hard__cap1p0__sel30",
        "hard__cap1p0__sel30__flow8",
        "taper__s0p65__e1p15__f0p4__sel30",
        "taper__s0p65__e1p15__f0p6__sel30",
        "taper__s0p75__e1p25__f0p4__sel30",
        "taper__s0p75__e1p25__f0p6__sel30",
        "daily__balanced__stop2p35",
        "daily__balanced__stop3p0",
        "daily__cap075_sel20_flow8__stop2p35",
        "daily__cap085_sel30__stop2p35",
        "daily__cap085_sel30_flow8__stop2p35",
        "daily__cap090_sel30__stop2p0",
        "daily__cap090_sel30__stop2p35",
        "daily__cap090_sel30__stop2p75",
        "daily__cap090_sel30__stop3p0",
        "daily__cap100_sel30__stop2p35",
        "daily__cap100_sel30_flow8__stop2p35",
        "daily__cap125_sel30__stop2p35",
        "daily__cap125_sel30_flow8__stop2p35",
        "density__balanced__l12__t0p5__m0p5",
        "density__balanced__l20__t0p58__m0p7",
        "density__cap100_sel30__l12__t0p5__m0p5",
        "density__cap100_sel30__l20__t0p58__m0p7",
        "quality__cap0p9__floor0p6__sel30",
        "quality__cap0p9__floor0p7__sel30",
        "reclaim__or_avwap__unlevered",
    }
    return [candidate for candidate in candidates if candidate.name in retained]


def _prior_baselines() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _load_json(PRIOR_VALIDATED)
    row = next(item for item in rows if item["name"] == "control__balanced_rvol170_or9_trail004")
    is_metrics = {
        "expected_total_r": row["is_expected_total_r"],
        "net_profit": row["is_net_profit"],
        "trades_per_month": row["is_trades_per_month"],
        "profit_factor": row["is_profit_factor"],
        "max_drawdown_pct": row["is_max_drawdown_pct"],
    }
    oos_metrics = {
        "expected_total_r": row["oos_expected_total_r"],
        "net_profit": row["oos_net_profit"],
        "trades_per_month": row["oos_trades_per_month"],
        "profit_factor": row["oos_profit_factor"],
        "max_drawdown_pct": row["oos_max_drawdown_pct"],
    }
    return is_metrics, oos_metrics


def _metrics(results: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return dict((results.get(name) or {}).get("metrics") or {})


def _ratio(value: Any, baseline: Any) -> float:
    base = float(baseline or 0.0)
    return float(value or 0.0) / base if abs(base) > 1e-12 else 0.0


def _is_score(metrics: dict[str, Any], baseline: dict[str, Any]) -> float:
    r = _ratio(metrics.get("expected_total_r"), baseline.get("expected_total_r"))
    net = _ratio(metrics.get("net_profit"), baseline.get("net_profit"))
    freq = _ratio(metrics.get("trades_per_month"), baseline.get("trades_per_month"))
    pf = _ratio(metrics.get("profit_factor"), baseline.get("profit_factor"))
    dd = _ratio(metrics.get("max_drawdown_pct"), baseline.get("max_drawdown_pct"))
    score = 0.25 * r + 0.24 * net + 0.10 * freq + 0.19 * pf + 0.22 * (2.0 - dd)
    if float(metrics.get("profit_factor", 0.0)) < 1.90:
        score -= 0.15
    if float(metrics.get("max_drawdown_pct", 1.0)) > 0.10:
        score -= 0.15
    return score


def _is_viable(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        _ratio(metrics.get("expected_total_r"), baseline.get("expected_total_r")) >= 0.84
        and _ratio(metrics.get("net_profit"), baseline.get("net_profit")) >= 0.86
        and _ratio(metrics.get("trades_per_month"), baseline.get("trades_per_month")) >= 0.84
        and float(metrics.get("profit_factor", 0.0)) >= 1.88
        and float(metrics.get("max_drawdown_pct", 1.0)) <= 0.108
    )


def _select_oos(
    candidates: list[Candidate],
    is_results: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    limit: int,
) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in candidates}
    ranked = sorted(
        (
            (_is_score(_metrics(is_results, candidate.name), baseline), candidate)
            for candidate in candidates
            if _is_viable(_metrics(is_results, candidate.name), baseline)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    names = [BALANCED_NAME, PREVIOUS_NAME]
    names.extend(candidate.name for _, candidate in ranked[:limit])
    family_best: dict[str, tuple[float, Candidate]] = {}
    for score, candidate in ranked:
        family_best.setdefault(candidate.category, (score, candidate))
    names.extend(candidate.name for _, candidate in family_best.values())
    return [by_name[name] for name in dict.fromkeys(names) if name in by_name]


def _joint_rows(
    candidates: Iterable[Candidate],
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
        is_score = _is_score(ism, baseline_is)
        oos_r = _ratio(oosm.get("expected_total_r"), baseline_oos.get("expected_total_r"))
        oos_net = _ratio(oosm.get("net_profit"), baseline_oos.get("net_profit"))
        oos_freq = _ratio(oosm.get("trades_per_month"), baseline_oos.get("trades_per_month"))
        oos_pf = _ratio(oosm.get("profit_factor"), baseline_oos.get("profit_factor"))
        oos_dd = _ratio(oosm.get("max_drawdown_pct"), baseline_oos.get("max_drawdown_pct"))
        joint = (
            0.52 * is_score
            + 0.15 * oos_r
            + 0.12 * oos_net
            + 0.05 * oos_freq
            + 0.08 * oos_pf
            + 0.08 * (2.0 - oos_dd)
        )
        rows.append(
            {
                "name": candidate.name,
                "category": candidate.category,
                "patch": candidate.patch,
                "thesis": candidate.thesis,
                "is_score": is_score,
                "joint_score": joint,
                **{f"is_{key}": value for key, value in _metric_subset(ism).items()},
                **{f"oos_{key}": value for key, value in _metric_subset(oosm).items()},
            }
        )
    return sorted(rows, key=lambda row: row["joint_score"], reverse=True)


def _validation_candidates(
    joint: list[dict[str, Any]],
    catalog: dict[str, Candidate],
    limit: int,
) -> list[Candidate]:
    names = [
        BALANCED_NAME,
        PREVIOUS_NAME,
        "daily__cap125_sel30__stop2p35",
        "daily__cap125_sel30_flow8__stop2p35",
    ]
    names.extend(row["name"] for row in joint[:limit])
    family_seen: set[str] = set()
    for row in joint:
        if row["category"] not in family_seen:
            names.append(row["name"])
            family_seen.add(row["category"])
    return [catalog[name] for name in dict.fromkeys(names) if name in catalog]


def _validated_rows(
    joint_by_name: dict[str, dict[str, Any]],
    candidates: Iterable[Candidate],
    early: dict[str, dict[str, Any]],
    late: dict[str, dict[str, Any]],
    stress: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    be = _metrics(early, BALANCED_NAME)
    bl = _metrics(late, BALANCED_NAME)
    bs = _metrics(stress, BALANCED_NAME)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        base = dict(joint_by_name.get(candidate.name) or {})
        em = _metrics(early, candidate.name)
        lm = _metrics(late, candidate.name)
        sm = _metrics(stress, candidate.name)
        if not base or not em or not lm or not sm:
            continue
        early_r = _ratio(em.get("expected_total_r"), be.get("expected_total_r"))
        late_r = _ratio(lm.get("expected_total_r"), bl.get("expected_total_r"))
        stress_r_delta = float(sm.get("expected_total_r", 0.0)) - float(bs.get("expected_total_r", 0.0))
        stress_dd_ratio = _ratio(sm.get("max_drawdown_pct"), bs.get("max_drawdown_pct"))
        robustness = (
            base["joint_score"]
            + 0.10 * min(early_r, late_r)
            + 0.04 * (early_r + late_r)
            + 0.03 * stress_r_delta
            + 0.08 * (1.0 - stress_dd_ratio)
        )
        base.update(
            {
                "early_expected_total_r": em.get("expected_total_r"),
                "early_profit_factor": em.get("profit_factor"),
                "early_max_drawdown_pct": em.get("max_drawdown_pct"),
                "early_r_ratio": early_r,
                "late_expected_total_r": lm.get("expected_total_r"),
                "late_profit_factor": lm.get("profit_factor"),
                "late_max_drawdown_pct": lm.get("max_drawdown_pct"),
                "late_r_ratio": late_r,
                "segment_floor": min(early_r, late_r),
                "stress_expected_total_r": sm.get("expected_total_r"),
                "stress_profit_factor": sm.get("profit_factor"),
                "stress_max_drawdown_pct": sm.get("max_drawdown_pct"),
                "stress_r_delta": stress_r_delta,
                "stress_dd_ratio": stress_dd_ratio,
                "robustness_score": robustness,
            }
        )
        rows.append(base)
    return sorted(rows, key=lambda row: row["robustness_score"], reverse=True)


def _promotion_eligible(row: dict[str, Any], balanced: dict[str, Any]) -> bool:
    return bool(
        _ratio(row.get("is_expected_total_r"), balanced.get("is_expected_total_r")) >= 0.84
        and _ratio(row.get("is_net_profit"), balanced.get("is_net_profit")) >= 0.88
        and _ratio(row.get("is_trades_per_month"), balanced.get("is_trades_per_month")) >= 0.84
        and float(row.get("is_profit_factor", 0.0)) >= 1.90
        and float(row.get("is_max_drawdown_pct", 1.0)) <= 0.1025
        and _ratio(row.get("oos_expected_total_r"), balanced.get("oos_expected_total_r")) >= 0.90
        and _ratio(row.get("oos_net_profit"), balanced.get("oos_net_profit")) >= 0.88
        and _ratio(row.get("oos_trades_per_month"), balanced.get("oos_trades_per_month")) >= 0.84
        and float(row.get("oos_profit_factor", 0.0)) >= 3.25
        and float(row.get("segment_floor", 0.0)) >= 0.84
        and float(row.get("stress_dd_ratio", 2.0)) <= 1.0
    )


def _return_frequency_score(
    row: dict[str, Any],
    balanced: dict[str, Any],
) -> float:
    """Rank robust candidates on the requested return/frequency objective.

    Risk and path stability are handled by the promotion guardrails first. The
    short, consumed OOS interval receives less weight than IS, while still
    requiring the selected candidate to preserve its return and activity.
    """
    return (
        0.28 * _ratio(row.get("is_expected_total_r"), balanced.get("is_expected_total_r"))
        + 0.26 * _ratio(row.get("is_net_profit"), balanced.get("is_net_profit"))
        + 0.12 * _ratio(row.get("is_trades_per_month"), balanced.get("is_trades_per_month"))
        + 0.14 * _ratio(row.get("oos_expected_total_r"), balanced.get("oos_expected_total_r"))
        + 0.12 * _ratio(row.get("oos_net_profit"), balanced.get("oos_net_profit"))
        + 0.08 * _ratio(row.get("oos_trades_per_month"), balanced.get("oos_trades_per_month"))
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [key for key in rows[0] if key not in {"patch", "thesis"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_context(mutations: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    plugin = ALCBP16Plugin(
        DATA_DIR,
        start_date=start,
        end_date=end,
        initial_equity=INITIAL_EQUITY,
        max_workers=1,
    )
    try:
        return plugin._run_config(mutations, store_context=True, collect_diagnostics=True)
    finally:
        plugin.close_pool()


def _paired_day_bootstrap(
    balanced_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    iterations: int = 5000,
    seed: int = 240723,
) -> dict[str, Any]:
    def daily(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
        output: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in rows:
            key = str(row["exit_time"])[:10]
            output[key][0] += float(row.get("r_multiple", 0.0) or 0.0)
            output[key][1] += float(row.get("pnl_net", 0.0) or 0.0)
        return {key: (values[0], values[1]) for key, values in output.items()}

    left = daily(balanced_rows)
    right = daily(candidate_rows)
    days = sorted(set(left) | set(right))
    rng = random.Random(seed)
    r_deltas: list[float] = []
    pnl_deltas: list[float] = []
    for _ in range(iterations):
        sampled = [days[rng.randrange(len(days))] for _ in days]
        r_deltas.append(sum(right.get(day, (0.0, 0.0))[0] - left.get(day, (0.0, 0.0))[0] for day in sampled))
        pnl_deltas.append(sum(right.get(day, (0.0, 0.0))[1] - left.get(day, (0.0, 0.0))[1] for day in sampled))

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean": mean(values),
            "p05": ordered[int(0.05 * (len(ordered) - 1))],
            "median": ordered[len(ordered) // 2],
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "probability_positive": sum(value > 0 for value in values) / len(values),
        }

    return {
        "seed": seed,
        "iterations": iterations,
        "block": "exit_day paired resampling",
        "r_delta": summary(r_deltas),
        "pnl_delta": summary(pnl_deltas),
    }


def _leave_one_group_out(
    balanced_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    group_key,
    label: str,
) -> dict[str, Any]:
    groups = sorted({group_key(row) for row in [*balanced_rows, *candidate_rows]})
    output = []
    for omitted in groups:
        b = [row for row in balanced_rows if group_key(row) != omitted]
        c = [row for row in candidate_rows if group_key(row) != omitted]
        output.append(
            {
                f"omitted_{label}": omitted,
                "balanced_r": sum(float(row.get("r_multiple", 0.0) or 0.0) for row in b),
                "candidate_r": sum(float(row.get("r_multiple", 0.0) or 0.0) for row in c),
                "balanced_net": sum(float(row.get("pnl_net", 0.0) or 0.0) for row in b),
                "candidate_net": sum(float(row.get("pnl_net", 0.0) or 0.0) for row in c),
            }
        )
    for row in output:
        row["delta_r"] = row["candidate_r"] - row["balanced_r"]
        row["delta_net"] = row["candidate_net"] - row["balanced_net"]
    return {
        "group": label,
        "rows": output,
        "all_r_uplifts_positive": all(row["delta_r"] > 0 for row in output),
        "all_net_uplifts_positive": all(row["delta_net"] > 0 for row in output),
        "min_delta_r": min((row["delta_r"] for row in output), default=0.0),
        "min_delta_net": min((row["delta_net"] for row in output), default=0.0),
    }


def _context_diagnostics(rows: list[dict[str, Any]], start: str, end: str) -> dict[str, Any]:
    daily = _daily_equity(rows, start, end)
    episodes = _drawdown_episodes(daily, rows)
    return {
        "loss_concentration": _loss_concentration(rows),
        "daily_equity": daily,
        "drawdown_episodes": episodes,
    }


def _report(
    recommendation: dict[str, Any],
    validated: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    costs: list[dict[str, Any]],
    candidate_count: int,
) -> str:
    balanced = next(row for row in validated if row["name"] == BALANCED_NAME)
    winner = recommendation["candidate"]
    lines = [
        "# ALCB Round 4 final optimization",
        "",
        "The March-May 2026 interval is consumed development data. Selection is anchored to "
        "IS, early/late IS, the historical drawdown stress interval, paired resampling, and cost sensitivity.",
        "",
        f"- Candidates screened on IS: {candidate_count}",
        f"- Selected candidate: `{winner['name']}`",
        f"- Promotion eligible under the predeclared final guardrails: {recommendation['promotion_eligible']}",
        "",
        "## Balanced versus selected",
        "",
        "| Metric | Balanced IS | Selected IS | Balanced OOS | Selected OOS |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, pct in (
        ("Expected total R", "expected_total_r", False),
        ("Net profit", "net_profit", False),
        ("Trades/month", "trades_per_month", False),
        ("Profit factor", "profit_factor", False),
        ("Max drawdown", "max_drawdown_pct", True),
    ):
        fmt = (lambda value: f"{100*float(value):.2f}%") if pct else (lambda value: f"{float(value):.2f}")
        lines.append(
            f"| {label} | {fmt(balanced[f'is_{key}'])} | {fmt(winner[f'is_{key}'])} | "
            f"{fmt(balanced[f'oos_{key}'])} | {fmt(winner[f'oos_{key}'])} |"
        )
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            f"- Early/late IS R retention: {winner['early_r_ratio']:.3f} / {winner['late_r_ratio']:.3f}.",
            f"- Stress R delta: {winner['stress_r_delta']:+.2f}R.",
            f"- Stress DD ratio: {winner['stress_dd_ratio']:.3f}.",
            f"- Paired day-block bootstrap probability of positive R uplift: "
            f"{bootstrap['r_delta']['probability_positive']:.1%}.",
            f"- Paired day-block bootstrap probability of positive dollar uplift: "
            f"{bootstrap['pnl_delta']['probability_positive']:.1%}.",
            "",
            "## Cost sensitivity",
            "",
            "| Candidate | Window | Slip bps | Commission/share | R | Net | PF | DD |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in costs:
        lines.append(
            f"| {row['candidate']} | {row['window']} | {row['slip_bps']:.1f} | "
            f"{row['commission_per_share']:.4f} | {row['expected_total_r']:.2f} | "
            f"{row['net_profit']:.2f} | {row['profit_factor']:.3f} | "
            f"{100*row['max_drawdown_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The result is saved as a Round-4 candidate only after artifact promotion. "
            "The unavailable authoritative frozen direct-RTH bundle remains a provenance limitation.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--oos-limit", type=int, default=55)
    parser.add_argument("--validation-limit", type=int, default=28)
    parser.add_argument("--allow-legacy-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.allow_legacy_data:
        raise SystemExit("Pass --allow-legacy-data; the authoritative frozen direct-RTH bundle is absent.")
    os.environ["TRADING_REQUIRE_FROZEN_DATA"] = "false"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_snapshot = output / "base_round2_snapshot.json"
    if base_snapshot.exists():
        base = _load_json(base_snapshot)
    else:
        base = _load_json(BASE_CONFIG_PATH)
        _write_json(base_snapshot, base)

    baseline_is_prior, baseline_oos_prior = _prior_baselines()
    candidates = _candidate_catalog()
    catalog = {candidate.name: candidate for candidate in candidates}
    _write_json(output / "candidate_catalog.json", [asdict(candidate) for candidate in candidates])
    _write_json(
        output / "run_spec.json",
        {
            "generated_at_utc": datetime.now(timezone.utc),
            "windows": {
                "is": [IS_START, IS_END],
                "oos": [OOS_START, OOS_END],
                "early_is": EARLY_IS,
                "late_is": LATE_IS,
                "stress": STRESS,
            },
            "candidate_count": len(candidates),
            "base_config_snapshot": str(base_snapshot),
            "balanced_patch": BALANCED_PATCH,
            "data_authority": "repaired legacy cache; diagnostic provenance limitation",
            "oos_status": "consumed development data",
        },
    )

    is_results = _evaluate_candidates(
        candidates,
        base,
        start=IS_START,
        end=IS_END,
        max_workers=args.max_workers,
        output_path=output / "is_results.json",
        baseline_metrics=baseline_is_prior,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    baseline_is = _metrics(is_results, BALANCED_NAME)
    if not baseline_is:
        raise RuntimeError("Balanced IS control did not complete.")
    oos_candidates = _select_oos(candidates, is_results, baseline_is, args.oos_limit)
    _write_json(output / "oos_candidate_catalog.json", [asdict(candidate) for candidate in oos_candidates])

    oos_results = _evaluate_candidates(
        oos_candidates,
        base,
        start=OOS_START,
        end=OOS_END,
        max_workers=args.max_workers,
        output_path=output / "oos_results.json",
        baseline_metrics=baseline_oos_prior,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    baseline_oos = _metrics(oos_results, BALANCED_NAME)
    joint = _joint_rows(oos_candidates, is_results, oos_results, baseline_is, baseline_oos)
    _write_json(output / "joint_screen.json", joint)
    _write_csv(output / "joint_screen.csv", joint)

    validation_candidates = _validation_candidates(joint, catalog, args.validation_limit)
    _write_json(
        output / "validation_candidate_catalog.json",
        [asdict(candidate) for candidate in validation_candidates],
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
        start=STRESS[0],
        end=STRESS[1],
        max_workers=args.max_workers,
        output_path=output / "stress_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    validated = _validated_rows(
        {row["name"]: row for row in joint},
        validation_candidates,
        early,
        late,
        stress,
    )
    balanced_row = next(row for row in validated if row["name"] == BALANCED_NAME)
    for row in validated:
        row["promotion_eligible"] = _promotion_eligible(row, balanced_row)
        row["return_frequency_score"] = _return_frequency_score(row, balanced_row)
    eligible = [row for row in validated if row["promotion_eligible"] and row["name"] != BALANCED_NAME]
    winner = max(eligible, key=lambda row: row["return_frequency_score"]) if eligible else balanced_row
    _write_json(output / "validated_results.json", validated)
    _write_csv(output / "validated_results.csv", validated)

    context_names = list(dict.fromkeys([BALANCED_NAME, PREVIOUS_NAME, winner["name"]]))
    context_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name in context_names:
        mutation = {**base, **catalog[name].patch}
        context_rows[name] = {}
        for label, window in (("is", (IS_START, IS_END)), ("oos", (OOS_START, OOS_END))):
            ctx = _run_context(mutation, window[0], window[1])
            rows = [_trade_to_dict(trade) for trade in ctx["trades"]]
            context_rows[name][label] = rows
            _write_json(output / f"context_{name}_{label}_trades.json", rows)
            _write_json(
                output / f"context_{name}_{label}_diagnostics.json",
                {
                    "metrics": _metric_subset(ctx["metrics"]),
                    **_context_diagnostics(rows, window[0], window[1]),
                },
            )

    bootstrap = _paired_day_bootstrap(
        context_rows[BALANCED_NAME]["is"],
        context_rows[winner["name"]]["is"],
    )
    _write_json(output / "paired_day_bootstrap.json", bootstrap)
    leave_month = _leave_one_group_out(
        context_rows[BALANCED_NAME]["is"],
        context_rows[winner["name"]]["is"],
        lambda row: str(row["exit_time"])[:7],
        "month",
    )
    leave_sector = _leave_one_group_out(
        context_rows[BALANCED_NAME]["is"],
        context_rows[winner["name"]]["is"],
        lambda row: str(row.get("sector") or "UNKNOWN"),
        "sector",
    )
    _write_json(output / "leave_one_month_out.json", leave_month)
    _write_json(output / "leave_one_sector_out.json", leave_sector)

    cost_candidates: list[Candidate] = []
    for name in (BALANCED_NAME, winner["name"]):
        for slip, commission in ((7.5, 0.0075), (10.0, 0.0100), (15.0, 0.0100)):
            cost_candidates.append(
                Candidate(
                    name=f"cost__{name}__s{_slug(slip)}__c{_slug(commission)}",
                    stage="execution_cost_sensitivity",
                    category="cost",
                    patch={
                        **catalog[name].patch,
                        "slippage.slip_bps_normal": slip,
                        "slippage.commission_per_share": commission,
                    },
                    thesis="Execution-cost perturbation.",
                )
            )
    cost_is = _evaluate_candidates(
        cost_candidates,
        base,
        start=IS_START,
        end=IS_END,
        max_workers=args.max_workers,
        output_path=output / "cost_is_results.json",
        baseline_metrics=baseline_is,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    cost_oos = _evaluate_candidates(
        cost_candidates,
        base,
        start=OOS_START,
        end=OOS_END,
        max_workers=args.max_workers,
        output_path=output / "cost_oos_results.json",
        baseline_metrics=baseline_oos,
        batch_size=args.batch_size,
        evaluator_kind="process",
    )
    costs: list[dict[str, Any]] = []
    for candidate in cost_candidates:
        source_name = BALANCED_NAME if candidate.name.startswith(f"cost__{BALANCED_NAME}") else winner["name"]
        slip = float(candidate.patch["slippage.slip_bps_normal"])
        commission = float(candidate.patch["slippage.commission_per_share"])
        for label, results in (("is", cost_is), ("oos", cost_oos)):
            metrics = _metrics(results, candidate.name)
            costs.append(
                {
                    "candidate": source_name,
                    "window": label,
                    "slip_bps": slip,
                    "commission_per_share": commission,
                    **_metric_subset(metrics),
                }
            )
    _write_json(output / "cost_sensitivity.json", costs)

    recommendation = {
        "generated_at_utc": datetime.now(timezone.utc),
        "candidate": winner,
        "full_mutations": {**base, **catalog[winner["name"]].patch},
        "balanced": balanced_row,
        "previous_balanced_candidate": next(row for row in validated if row["name"] == PREVIOUS_NAME),
        "promotion_eligible": bool(winner.get("promotion_eligible")),
        "selection_rule": (
            "highest IS/OOS return-and-frequency score among candidates passing "
            "the final PF, drawdown, segment, and stress guardrails"
        ),
        "data_authority": "repaired legacy cache; authoritative frozen direct-RTH bundle unavailable",
        "oos_status": "consumed development data",
        "bootstrap": bootstrap,
        "leave_one_month_out": {
            key: value for key, value in leave_month.items() if key != "rows"
        },
        "leave_one_sector_out": {
            key: value for key, value in leave_sector.items() if key != "rows"
        },
    }
    _write_json(output / "recommendation.json", recommendation)
    (output / "report.md").write_text(
        _report(recommendation, validated, bootstrap, costs, len(candidates)),
        encoding="utf-8",
    )
    _write_json(
        output / "completion.json",
        {
            "completed_at_utc": datetime.now(timezone.utc),
            "candidate_count": len(candidates),
            "oos_candidate_count": len(oos_candidates),
            "validation_candidate_count": len(validation_candidates),
            "selected_candidate": winner["name"],
            "promotion_eligible": bool(winner.get("promotion_eligible")),
            "promotion_written": False,
        },
    )
    print(f"complete: {output}", flush=True)
    print(f"selected={winner['name']} eligible={winner.get('promotion_eligible')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
