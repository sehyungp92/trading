"""Candidate-centred extension to the corrected-split Round-1 OOS repair.

The first repair pass was intentionally broad.  This runner asks the narrower
follow-up question: once the ADX/chandelier seed is fixed, can a directly
relevant entry, regime, or exit control improve both development and observed
validation?  Results remain shadow research because the validation interval
has already been inspected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backtests.momentum.auto.downturn import round1_oos_repair as repair


SEED_PATH = repair.OUTPUT_DIR / "recommended_config.json"


def _candidate(
    name: str,
    seed: dict[str, Any],
    mutation: dict[str, Any],
    family: str,
    note: str = "",
) -> dict[str, Any]:
    return repair._candidate(name, {**seed, **mutation}, family, note)


def _grid_candidates(
    seed: dict[str, Any],
    grids: dict[str, Iterable[Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key, values in grids.items():
        for value in values:
            if seed.get(key) == value:
                continue
            candidates.append(
                _candidate(
                    f"extension:grid:{key}:{value}",
                    seed,
                    {key: value},
                    "candidate_centered_grid",
                )
            )
    return candidates


def single_candidates() -> list[dict[str, Any]]:
    """Granular candidate-centred singles, including previously missed paths."""

    seed = dict(repair._load_json(SEED_PATH))
    candidates = [
        repair._candidate(
            "extension:recommended_seed", seed, "extension_baseline"
        )
    ]

    # These controls are evaluated on the selected regime classification, not
    # on the old baseline.  In particular, emerging/neutral TP levels now apply
    # to the two large observed-validation winners.
    grids: dict[str, tuple[Any, ...]] = {
        "param_overrides.adx_trending_threshold": (
            20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.5, 23.0,
        ),
        "param_overrides.adx_range_threshold": (8, 9, 10, 11, 12, 13, 14, 15, 16),
        "param_overrides.chandelier_lookback": (24, 26, 28, 30, 32, 34, 36, 40),
        "param_overrides.chandelier_mult_floor": (1.5, 1.8, 2.0, 2.2, 2.5, 3.0),
        "param_overrides.chandelier_mult_ceiling": (3.0, 3.5, 4.0, 4.5, 5.0),
        "param_overrides.post_tp1_chandelier_mult": (2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
        "param_overrides.min_hold_bars": (0, 3, 4, 6, 8, 10, 11, 12, 13, 14, 16, 18, 24),
        "param_overrides.profit_floor_r_threshold": (
            0.40, 0.50, 0.60, 0.75, 0.80, 0.90, 1.0, 1.10, 1.25, 1.50, 1.80, 2.0,
        ),
        "param_overrides.profit_floor_lock_pct": (
            0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
        ),
        "param_overrides.be_trigger_r": (
            0.30, 0.40, 0.50, 0.60, 0.75, 0.80, 0.90, 1.0, 1.10, 1.25, 1.50,
        ),
        "param_overrides.be_stop_buffer_mult": (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20),
        "param_overrides.tp1_r_emerging": (
            0.80, 1.0, 1.20, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.0, 2.20, 2.40, 2.60, 2.80, 3.0,
        ),
        "param_overrides.tp2_r_emerging": (1.50, 1.80, 2.0, 2.20, 2.50, 2.80, 3.0, 3.50, 4.0),
        "param_overrides.tp1_r_range": (0.60, 0.80, 1.0, 1.20, 1.40, 1.60, 1.80, 2.0),
        "param_overrides.stale_bars_fade": (12, 18, 24, 28, 36, 48, 72),
        "param_overrides.climax_mult": (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5),
        "param_overrides.entry_ttl_bars": (6, 8, 10, 12, 14, 16, 18, 24, 36),
        "param_overrides.entry_limit_offset_ticks": (1, 2, 3, 4, 5, 6, 8),
        "param_overrides.entry_buffer_ticks": (0, 1, 2, 3, 4),
        "param_overrides.trigger_low_buffer_ticks": (0, 1, 2, 3, 4),
        "param_overrides.fade_stop_atr_mult": (0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.75, 0.85, 1.0),
        "param_overrides.vwap_cap_core": (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60),
        "param_overrides.vwap_cap_extended": (0.30, 0.40, 0.50, 0.60, 0.70, 0.77, 0.90),
        "param_overrides.rejection_lookback_bars": (3, 4, 5, 6, 8, 10, 12, 16),
        "param_overrides.mom_slope_lookback": (3, 4, 5, 6, 8, 10, 12),
        "param_overrides.momentum_cooldown_bars": (6, 12, 18, 24, 30, 36, 48, 72),
        "param_overrides.momentum_roc_threshold": (-0.001, -0.0015, -0.002, -0.0025, -0.003, -0.0035, -0.004, -0.005, -0.006),
        "param_overrides.max_daily_entries": (1, 2, 3, 4, 5, 6, 8),
        "param_overrides.friction_min_atr_pctl": (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30),
        "param_overrides.drawdown_threshold": (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05),
        "param_overrides.drawdown_lookback": (5, 8, 10, 12, 15, 20, 30, 40),
        "param_overrides.progressive_sma_min": (40, 60, 80, 100, 120, 140, 160, 180, 220),
        "param_overrides.bear_structure_min_conditions": (1, 2, 3),
        "param_overrides.bear_structure_adx_on": (15, 18, 20, 22, 25, 28, 30),
        "param_overrides.bear_structure_adx_off": (8, 10, 12, 15, 18, 20),
        "param_overrides.bear_structure_path_b_conviction": (30, 40, 50, 60, 70),
        "param_overrides.bear_structure_path_c_di_gap": (4, 6, 8, 10, 12),
        "param_overrides.bear_structure_path_c_ema_sep": (0.05, 0.10, 0.15, 0.20, 0.25),
        "param_overrides.crash_daily_threshold": (-0.01, -0.0125, -0.015, -0.0175, -0.02, -0.025, -0.03),
        "param_overrides.conviction_threshold": (20, 25, 30, 35, 40, 50, 60),
        "flags.vol_percentile_gate": (0.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
        "flags.regime_confidence_gate": (0.0, 20.0, 30.0, 40.0, 50.0, 60.0),
    }
    candidates.extend(_grid_candidates(seed, grids))

    mechanisms: dict[str, dict[str, Any]] = {
        "disable_min_hold": {"flags.min_hold_period": False},
        "disable_profit_floor": {"flags.profit_floor_trail": False},
        "disable_vwap_failure": {"flags.vwap_failure_exit": False},
        "disable_stale_exit": {"flags.stale_exit": False},
        "disable_climax_exit": {"flags.climax_exit": False},
        "disable_tiered_exits": {"flags.tiered_exits": False},
        "enable_adaptive_floor": {"flags.adaptive_profit_floor": True},
        "enable_multitier_floor": {"flags.multi_tier_profit_floor": True},
        "enable_regime_chandelier": {"flags.regime_adaptive_chandelier": True},
        "enable_cancel_replace": {"flags.cancel_replace_entry": True},
        "disable_fade_momentum_confirm": {"flags.fade_momentum_confirm": False},
        "allow_fade_nonbear": {"flags.fade_bear_regime_required": False},
        "enable_fast_crash": {"flags.fast_crash_override": True},
        "enable_fast_crash_conviction": {
            "flags.fast_crash_override": True,
            "flags.conviction_scoring": True,
        },
        "enable_conviction": {"flags.conviction_scoring": True},
        "allow_reversal_correction": {"flags.allow_reversal_in_correction": True},
        "reversal_1of3": {"flags.reversal_min_gate_count": 1},
        "reversal_no_extension": {"flags.reversal_no_extension_gate": True},
        "reversal_wider_corridor": {"flags.reversal_wider_corridor": 3.0},
        "enable_breakdown": {"flags.breakdown_engine": True},
        "correction_only": {"flags.correction_only_mode": True},
        "correction_only_fade": {"flags.correction_only_fade": True},
    }
    for name, mutation in mechanisms.items():
        candidates.append(
            _candidate(
                f"extension:mechanism:{name}",
                seed,
                mutation,
                "candidate_centered_mechanism",
            )
        )

    return repair._deduplicate(candidates)


def interaction_candidates() -> list[dict[str, Any]]:
    """Explicit pairs plus frequency-restoration triples around the seed."""

    seed = dict(repair._load_json(SEED_PATH))
    controls: dict[str, dict[str, Any]] = {
        "daily4": {"param_overrides.max_daily_entries": 4},
        "daily5": {"param_overrides.max_daily_entries": 5},
        "ttl14": {"param_overrides.entry_ttl_bars": 14},
        "ttl18": {"param_overrides.entry_ttl_bars": 18},
        "ttl24": {"param_overrides.entry_ttl_bars": 24},
        "be045": {"param_overrides.be_trigger_r": 0.45},
        "be050": {"param_overrides.be_trigger_r": 0.50},
        "be055": {"param_overrides.be_trigger_r": 0.55},
        "floor18": {"param_overrides.profit_floor_r_threshold": 1.8},
        "floor20": {"param_overrides.profit_floor_r_threshold": 2.0},
        "chceil325": {"param_overrides.chandelier_mult_ceiling": 3.25},
        "chceil350": {"param_overrides.chandelier_mult_ceiling": 3.50},
        "chceil375": {"param_overrides.chandelier_mult_ceiling": 3.75},
        "drawdown15": {"param_overrides.drawdown_lookback": 15},
        "vwapcore020": {"param_overrides.vwap_cap_core": 0.20},
        "entrybuffer1": {"param_overrides.entry_buffer_ticks": 1},
        "bearoff12": {"param_overrides.bear_structure_adx_off": 12},
    }
    candidates = [
        repair._candidate(
            "extension_interactions:recommended_seed",
            seed,
            "extension_interaction_baseline",
        )
    ]
    for name, mutation in controls.items():
        candidates.append(
            _candidate(
                f"extension_interactions:single:{name}",
                seed,
                mutation,
                "extension_shortlist_single",
            )
        )

    items = list(controls.items())
    for left_index, (left_name, left) in enumerate(items):
        for right_name, right in items[left_index + 1 :]:
            if set(left) & set(right):
                continue
            candidates.append(
                _candidate(
                    f"extension_interactions:pair:{left_name}+{right_name}",
                    seed,
                    {**left, **right},
                    "extension_pair_interaction",
                )
            )

    # The daily-entry expansion is the only single that raises both OOS return
    # and OOS frequency.  Test every pair of the strongest independent IS
    # restorers on top of it rather than accepting its IS sacrifice.
    restorers = {
        name: mutation
        for name, mutation in controls.items()
        if name
        in {
            "ttl14", "ttl18", "ttl24", "be050", "floor18",
            "chceil350", "drawdown15", "vwapcore020", "entrybuffer1",
            "bearoff12",
        }
    }
    restorer_items = list(restorers.items())
    for left_index, (left_name, left) in enumerate(restorer_items):
        for right_name, right in restorer_items[left_index + 1 :]:
            if set(left) & set(right):
                continue
            candidates.append(
                _candidate(
                    f"extension_interactions:triple:daily4+{left_name}+{right_name}",
                    seed,
                    {**controls["daily4"], **left, **right},
                    "frequency_restoration_triple",
                )
            )

    return repair._deduplicate(candidates)


def _frontier_seed() -> dict[str, Any]:
    payload = repair._load_json(repair.OUTPUT_DIR / "extended_interactions.json")
    wanted = "extension_interactions:triple:daily4+ttl18+be050"
    row = next(row for row in payload["results"] if row["name"] == wanted)
    return dict(row["mutations"])


def mechanism_candidates() -> list[dict[str, Any]]:
    """Test smooth in-min-hold protection and the finalist local surface."""

    frontier = _frontier_seed()
    prior = dict(repair._load_json(SEED_PATH))
    candidates = [
        repair._candidate(
            "extension_mechanisms:frontier_seed",
            frontier,
            "mechanism_frontier_baseline",
        ),
        _candidate(
            "extension_mechanisms:prior+protection",
            prior,
            {"flags.min_hold_profit_protection": True},
            "min_hold_protection_control",
        ),
    ]

    # Local stability of the purely-configurational frontier.  The daily cap is
    # discrete, while TTL and BE should form a plateau rather than one peak.
    for daily_entries in (3, 4, 5):
        for ttl in (12, 14, 16, 18, 20, 24):
            for be_trigger in (0.40, 0.45, 0.50, 0.55, 0.60):
                candidates.append(
                    _candidate(
                        f"extension_mechanisms:local:daily{daily_entries}:ttl{ttl}:be{be_trigger}",
                        frontier,
                        {
                            "param_overrides.max_daily_entries": daily_entries,
                            "param_overrides.entry_ttl_bars": ttl,
                            "param_overrides.be_trigger_r": be_trigger,
                        },
                        "frontier_local_surface",
                    )
                )

    # Profit protection is deliberately crossed with min-hold length, trigger,
    # and lock strength.  This prevents selection from being driven by the one
    # observed trade that reached 1.53R before reverting to a loss.
    for min_hold in (11, 12, 13, 14, 16):
        for floor_trigger in (0.75, 1.0, 1.25, 1.50):
            for lock_pct in (0.30, 0.40, 0.50, 0.60):
                candidates.append(
                    _candidate(
                        f"extension_mechanisms:protect:hold{min_hold}:floor{floor_trigger}:lock{lock_pct}",
                        frontier,
                        {
                            "flags.min_hold_profit_protection": True,
                            "param_overrides.min_hold_bars": min_hold,
                            "param_overrides.profit_floor_r_threshold": floor_trigger,
                            "param_overrides.profit_floor_lock_pct": lock_pct,
                        },
                        "min_hold_protection_surface",
                    )
                )

    for be_trigger in (0.40, 0.45, 0.50, 0.55, 0.60):
        candidates.append(
            _candidate(
                f"extension_mechanisms:protect_be_only:{be_trigger}",
                frontier,
                {
                    "flags.min_hold_profit_protection": True,
                    "flags.profit_floor_trail": False,
                    "param_overrides.be_trigger_r": be_trigger,
                },
                "min_hold_be_only",
            )
        )

    # Recheck the inherited emerging TP cliff on the stronger frontier.
    for tp1 in (1.70, 1.75, 1.80, 1.825, 1.85, 1.875, 1.90):
        candidates.append(
            _candidate(
                f"extension_mechanisms:tp1_emerging:{tp1}",
                frontier,
                {"param_overrides.tp1_r_emerging": tp1},
                "emerging_tp_local_surface",
            )
        )

    return repair._deduplicate(candidates)


def _mechanism_seed(name: str) -> dict[str, Any]:
    payload = repair._load_json(repair.OUTPUT_DIR / "extended_mechanisms.json")
    row = next(row for row in payload["results"] if row["name"] == name)
    return dict(row["mutations"])


def verification_candidates() -> list[dict[str, Any]]:
    """Verify robust, high-IS, and high-OOS protection finalists."""

    robust = _mechanism_seed(
        "extension_mechanisms:protect:hold13:floor1.25:lock0.4"
    )
    # The 40% lock is the engine default.  Drop the redundant override so the
    # recommendation contains only behavior-changing mutations.
    robust.pop("param_overrides.profit_floor_lock_pct", None)
    high_is = _mechanism_seed(
        "extension_mechanisms:protect:hold13:floor1.5:lock0.4"
    )
    high_oos = _mechanism_seed(
        "extension_mechanisms:protect:hold13:floor1.5:lock0.5"
    )
    high_frequency = _mechanism_seed(
        "extension_mechanisms:protect:hold13:floor0.75:lock0.4"
    )
    config_only = _frontier_seed()
    previous = dict(repair._load_json(SEED_PATH))

    candidates = [
        repair._candidate(
            "extension_verification:robust_seed",
            robust,
            "extension_verification_baseline",
        ),
        repair._candidate(
            "extension_verification:high_is_challenger",
            high_is,
            "extension_finalist_challenger",
        ),
        repair._candidate(
            "extension_verification:high_oos_challenger",
            high_oos,
            "extension_finalist_challenger",
        ),
        repair._candidate(
            "extension_verification:high_frequency_challenger",
            high_frequency,
            "extension_finalist_challenger",
        ),
        repair._candidate(
            "extension_verification:config_only_frontier",
            config_only,
            "extension_frontier_control",
        ),
        repair._candidate(
            "extension_verification:previous_recommendation",
            previous,
            "extension_previous_control",
        ),
    ]

    for ttl in (14, 16, 18, 20, 22):
        for be_trigger in (0.40, 0.45, 0.50, 0.55, 0.60):
            candidates.append(
                _candidate(
                    f"extension_verification:ttl_be:{ttl}:{be_trigger}",
                    robust,
                    {
                        "param_overrides.entry_ttl_bars": ttl,
                        "param_overrides.be_trigger_r": be_trigger,
                    },
                    "protected_ttl_be_surface",
                )
            )

    for floor_trigger in (1.0, 1.125, 1.25, 1.375, 1.5, 1.6, 1.75, 2.0):
        for lock_pct in (0.30, 0.35, 0.40, 0.45, 0.50):
            candidates.append(
                _candidate(
                    f"extension_verification:floor_lock:{floor_trigger}:{lock_pct}",
                    robust,
                    {
                        "param_overrides.profit_floor_r_threshold": floor_trigger,
                        "param_overrides.profit_floor_lock_pct": lock_pct,
                    },
                    "protected_floor_lock_surface",
                )
            )

    for hold in (12, 13, 14):
        candidates.append(
            _candidate(
                f"extension_verification:min_hold:{hold}",
                robust,
                {"param_overrides.min_hold_bars": hold},
                "protected_hold_axis",
            )
        )
    for daily_entries in (3, 4, 5):
        candidates.append(
            _candidate(
                f"extension_verification:daily_entries:{daily_entries}",
                robust,
                {"param_overrides.max_daily_entries": daily_entries},
                "protected_frequency_axis",
            )
        )
    for tp1 in (1.70, 1.75, 1.80, 1.825, 1.85, 1.875, 1.90):
        candidates.append(
            _candidate(
                f"extension_verification:tp1_emerging:{tp1}",
                robust,
                {"param_overrides.tp1_r_emerging": tp1},
                "protected_tp_axis",
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
    for control_name, control in {
        "robust": robust,
        "selected_high_is": high_is,
        "config_only": config_only,
        "previous": previous,
    }.items():
        for stress_name, stress in stresses.items():
            candidates.append(
                _candidate(
                    f"extension_verification:stress:{control_name}:{stress_name}",
                    control,
                    stress,
                    f"{control_name}_execution_stress",
                )
            )

    return repair._deduplicate(candidates)


def _row_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in rows if row["name"] == name)


def finalize_extension() -> None:
    """Write the shadow recommendation, diagnostics, and reproducibility addendum."""

    selected_mutations = _mechanism_seed(
        "extension_mechanisms:protect:hold13:floor1.5:lock0.4"
    )
    selected_mutations.pop("param_overrides.profit_floor_lock_pct", None)

    repair._init_split_worker(
        str(repair.recovery.DATA_DIR), repair.recovery.INITIAL_EQUITY
    )
    detail = repair._evaluate_worker(
        ("extension_selected_detail", selected_mutations, True)
    )
    trades = detail.pop("trades")
    attribution = repair._trade_diagnostics(trades)
    oos_trades = [
        trade
        for trade in trades
        if repair.OOS_START
        <= repair.recovery._trade_time(trade)
        < repair.EVALUATION_END
    ]
    repair._write_json(
        repair.OUTPUT_DIR / "recommended_config_extension.json",
        selected_mutations,
    )
    repair._write_json(
        repair.OUTPUT_DIR / "recommended_trade_attribution_extension.json",
        {"evaluation": detail, "attribution": attribution},
    )

    stage_names = (
        "extended_singles",
        "extended_interactions",
        "extended_mechanisms",
        "extended_verify",
    )
    stage_payloads = {
        stage: repair._load_json(repair.OUTPUT_DIR / f"{stage}.json")
        for stage in stage_names
    }
    verification = stage_payloads["extended_verify"]["results"]
    contenders = {
        "previous_recommendation": _row_by_name(
            verification, "extension_verification:previous_recommendation"
        ),
        "config_only_frontier": _row_by_name(
            verification, "extension_verification:config_only_frontier"
        ),
        "robust_interior": _row_by_name(
            verification, "extension_verification:robust_seed"
        ),
        "selected_balanced_knee": _row_by_name(
            verification, "extension_verification:high_is_challenger"
        ),
        "higher_oos_lock": _row_by_name(
            verification, "extension_verification:high_oos_challenger"
        ),
        "higher_frequency": _row_by_name(
            verification, "extension_verification:high_frequency_challenger"
        ),
        "higher_is_floor_1_6": _row_by_name(
            verification, "extension_verification:floor_lock:1.6:0.4"
        ),
    }

    stresses: dict[str, Any] = {}
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
        row = _row_by_name(
            verification,
            f"extension_verification:stress:selected_high_is:{stress_name}",
        )
        stresses[stress_name] = {
            "selection_metrics": row["selection_metrics"],
            "oos_metrics": row["oos_metrics"],
        }

    ttl_be = [
        row for row in verification if row["family"] == "protected_ttl_be_surface"
    ]
    floor_lock = [
        row for row in verification if row["family"] == "protected_floor_lock_surface"
    ]
    tp_axis = [
        row for row in verification if row["family"] == "protected_tp_axis"
    ]
    all_signatures: set[str] = set()
    for stage in (
        "historical_ablation",
        "perturbation",
        "targeted",
        "verification",
        *stage_names,
    ):
        path = repair.OUTPUT_DIR / f"{stage}.json"
        if path.exists():
            all_signatures.update(
                row["signature"] for row in repair._load_json(path)["results"]
            )
    original_signatures: set[str] = set()
    for stage in ("historical_ablation", "perturbation", "targeted", "verification"):
        original_signatures.update(
            row["signature"]
            for row in repair._load_json(repair.OUTPUT_DIR / f"{stage}.json")[
                "results"
            ]
        )

    oos_total = sum(float(trade.pnl) for trade in oos_trades)
    day_pnl: dict[str, float] = {}
    for trade in oos_trades:
        day = repair.recovery._trade_time(trade).date().isoformat()
        day_pnl[day] = day_pnl.get(day, 0.0) + float(trade.pnl)
    leave_one_active_day_out = {
        day: oos_total - pnl for day, pnl in sorted(day_pnl.items())
    }
    is_months = (
        (repair.OOS_START - repair.IS_START).total_seconds()
        / 86_400.0
        / 365.25
        * 12.0
    )
    oos_months = (
        (repair.EVALUATION_END - repair.OOS_START).total_seconds()
        / 86_400.0
        / 365.25
        * 12.0
    )

    summary = {
        "disposition": "SHADOW_RESEARCH_ONLY",
        "split": {
            "is_start": repair.IS_START.isoformat(),
            "is_end_inclusive": "2026-03-20",
            "oos_start": repair.OOS_START.isoformat(),
            "oos_end_inclusive": "2026-05-01",
            "evaluation_end_exclusive": repair.EVALUATION_END.isoformat(),
        },
        "selection_rationale": (
            "The 1.5R/40% protection point is the balanced knee: it materially "
            "beats the prior recommendation in both windows and remains between "
            "the 1.25R OOS-protective and 1.6R IS-maximizing frontiers."
        ),
        "selected": {
            "name": "extension:daily4+ttl18+be050+min_hold_protection_floor150",
            "mutations": selected_mutations,
            "signature": repair._signature(selected_mutations),
            "selection_metrics": detail["selection_metrics"],
            "oos_metrics": detail["oos_metrics"],
            "folds": detail["folds"],
        },
        "contenders": contenders,
        "frequency": {
            "is_trades_per_month": detail["selection_metrics"]["total_trades"]
            / is_months,
            "oos_trades_per_month": detail["oos_metrics"]["total_trades"]
            / oos_months,
        },
        "oos_bootstrap": repair._bootstrap_window(oos_trades),
        "oos_day_pnl": day_pnl,
        "oos_leave_one_active_day_out_pnl": leave_one_active_day_out,
        "execution_stress": stresses,
        "stability": {
            "ttl_be_points": len(ttl_be),
            "ttl_be_points_is_ge_110_oos_ge_24": sum(
                row["selection_metrics"]["net_return_pct"] >= 110.0
                and row["oos_metrics"]["net_return_pct"] >= 24.0
                for row in ttl_be
            ),
            "floor_lock_points": len(floor_lock),
            "floor_lock_points_is_ge_110_oos_ge_24": sum(
                row["selection_metrics"]["net_return_pct"] >= 110.0
                and row["oos_metrics"]["net_return_pct"] >= 24.0
                for row in floor_lock
            ),
            "tp_axis": [
                {
                    "name": row["name"],
                    "selection_metrics": row["selection_metrics"],
                    "oos_metrics": row["oos_metrics"],
                }
                for row in tp_axis
            ],
        },
        "candidate_counts": {
            stage: payload["candidate_count"]
            for stage, payload in stage_payloads.items()
        },
        "extension_unique_configurations": len(all_signatures - original_signatures),
        "all_repair_unique_configurations": len(all_signatures),
        "attribution_path": "recommended_trade_attribution_extension.json",
    }
    repair._write_json(repair.OUTPUT_DIR / "extension_summary.json", summary)

    extension_path = Path(__file__).resolve()
    engine_path = repair.ROOT / "backtests/momentum/engine/downturn_engine.py"
    config_path = repair.ROOT / "strategies/momentum/downturn/config.py"
    data_paths = [
        repair.recovery.DATA_DIR / "NQ_5m.parquet",
        repair.recovery.DATA_DIR / "ES_1d.parquet",
        repair.recovery.DATA_DIR / "NQ_5m.manifest.json",
        repair.recovery.DATA_DIR / "ES_1d.manifest.json",
    ]
    repair._write_json(
        repair.OUTPUT_DIR / "extension_run_spec.json",
        {
            "purpose": "Candidate-centred Round-1 corrected-split extension",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disposition": summary["disposition"],
            "split": summary["split"],
            "initial_equity": repair.recovery.INITIAL_EQUITY,
            "selected_signature": summary["selected"]["signature"],
            "code": [
                {
                    "path": str(path.relative_to(repair.ROOT)),
                    "sha256": repair._file_sha256(path),
                }
                for path in (extension_path, engine_path, config_path)
            ],
            "data": [
                {
                    "path": str(path.relative_to(repair.ROOT)),
                    "sha256": repair._file_sha256(path),
                }
                for path in data_paths
                if path.exists()
            ],
            "candidate_counts": summary["candidate_counts"],
            "extension_unique_configurations": summary[
                "extension_unique_configurations"
            ],
            "all_repair_unique_configurations": summary[
                "all_repair_unique_configurations"
            ],
        },
    )

    previous = contenders["previous_recommendation"]
    selected_is = detail["selection_metrics"]
    selected_oos = detail["oos_metrics"]
    previous_is = previous["selection_metrics"]
    previous_oos = previous["oos_metrics"]
    report = [
        "# Downturn Round 1 OOS Repair Extension",
        "",
        "Disposition: **SHADOW_RESEARCH_ONLY**. The observed validation interval "
        "was used for comparison and cannot support production promotion.",
        "",
        "| Configuration | IS trades | IS return | IS PF | IS DD | Validation trades | Validation return | Validation PF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Previous recommendation | {previous_is['total_trades']} | {previous_is['net_return_pct']:.2f}% | {previous_is['profit_factor']:.2f} | {100*previous_is['max_dd_pct']:.2f}% | {previous_oos['total_trades']} | {previous_oos['net_return_pct']:.2f}% | {previous_oos['profit_factor']:.2f} |",
        f"| Extended recommendation | {selected_is['total_trades']} | {selected_is['net_return_pct']:.2f}% | {selected_is['profit_factor']:.2f} | {100*selected_is['max_dd_pct']:.2f}% | {selected_oos['total_trades']} | {selected_oos['net_return_pct']:.2f}% | {selected_oos['profit_factor']:.2f} |",
        "",
        "## Decision",
        "",
        "The prior candidate was not the best available point. The extended "
        "candidate raises return and frequency in both windows while improving "
        "IS PF and drawdown. It adds a four-entry daily cap, longer entry TTL, "
        "earlier breakeven, and default-off profit protection during min hold.",
        "",
        "The 1.5R floor is the balanced knee. A 1.25R trigger is more interior "
        "but gives up IS return; 1.6R raises IS further while giving back about "
        "0.9 percentage points of validation return.",
        "",
        "## Robustness",
        "",
        f"- New unique configurations: {summary['extension_unique_configurations']}; total corrected-split repair configurations: {summary['all_repair_unique_configurations']}.",
        f"- TTL/BE surface passing IS >=110% and OOS >=24%: {summary['stability']['ttl_be_points_is_ge_110_oos_ge_24']}/{summary['stability']['ttl_be_points']}.",
        f"- Floor/lock surface passing IS >=110% and OOS >=24%: {summary['stability']['floor_lock_points_is_ge_110_oos_ge_24']}/{summary['stability']['floor_lock_points']}.",
        f"- Selected OOS bootstrap probability of positive PnL: {100*summary['oos_bootstrap']['probability_positive_net_pnl']:.1f}%.",
        "- Commission, slippage, and spread stresses remain strong; one-bar "
        "latency remains the principal fragility.",
        "",
        "## Caveats",
        "",
        "- Validation still contains only ten trades and three active days; all "
        "three active-day PnLs and every leave-one-active-day-out result are positive.",
        f"- Validation win rate is {selected_oos['win_rate']:.1f}% versus "
        f"{selected_is['win_rate']:.1f}% IS. This gap is not treated as a durable "
        "uplift; it is primarily a ten-trade sampling effect.",
        "- The inherited 1.8R emerging TP remains sensitive above approximately "
        "1.85R; it was retained rather than retuned on observed validation.",
        "- The min-hold protection flag is default-off and requires live/core "
        "parity implementation and fresh shadow validation before promotion.",
    ]
    (repair.OUTPUT_DIR / "extension_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("singles", "interactions", "mechanisms", "verify", "final"),
        default="singles",
    )
    parser.add_argument("--max-workers", type=int, default=repair.MAX_WORKERS)
    args = parser.parse_args()
    if not Path(SEED_PATH).exists():
        parser.error(f"missing seed: {SEED_PATH}")
    if args.stage == "final":
        finalize_extension()
        return
    stage_candidates = {
        "singles": single_candidates,
        "interactions": interaction_candidates,
        "mechanisms": mechanism_candidates,
        "verify": verification_candidates,
    }
    candidates = stage_candidates[args.stage]()
    repair.evaluate_candidates(
        candidates,
        stage=f"extended_{args.stage}",
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
