"""VdubusNQ R1 phased auto-optimization experiment candidates.

Phase 1: Trail rehabilitation & exit management (24 experiments)
Phase 2: Entry precision & filter tuning (20 experiments)
Phase 3: Signal discrimination & regime (18 experiments)
Phase 4: Fine-tune & interaction (~20+ dynamic experiments)
"""
from __future__ import annotations

from typing import Any


def get_phase_candidates(
    phase: int,
    prior_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Return (name, mutations) pairs for the given phase."""
    prior = prior_mutations or {}
    candidates: list[tuple[str, dict]] = []

    # Prepend suggested experiments (deduped)
    if suggested_experiments:
        seen = set()
        for name, muts in suggested_experiments:
            if name not in seen:
                seen.add(name)
                candidates.append((name, muts))

    if phase == 1:
        candidates.extend(_phase_1_trail(prior))
    elif phase == 2:
        candidates.extend(_phase_2_entry(prior))
    elif phase == 3:
        candidates.extend(_phase_3_signal(prior))
    elif phase == 4:
        candidates.extend(_phase_4_finetune(prior))

    # Deduplicate by name
    seen_names: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, muts in candidates:
        if name not in seen_names:
            seen_names.add(name)
            unique.append((name, muts))
    return unique


def _phase_1_trail(prior: dict) -> list[tuple[str, dict]]:
    """Phase 1: Trail rehabilitation & exit management (~24 candidates)."""
    c: list[tuple[str, dict]] = []

    # Strategy A: Re-enable BE + Wider Trail
    c.append(("be_default_trail", {
        "flags.plus_1r_partial": True,
    }))
    c.append(("be_wide_trail_35", {
        "flags.plus_1r_partial": True,
        "param_overrides.TRAIL_MULT_BASE": 3.5,
        "param_overrides.TRAIL_MULT_MIN": 2.0,
    }))
    c.append(("be_wide_trail_40", {
        "flags.plus_1r_partial": True,
        "param_overrides.TRAIL_MULT_BASE": 4.0,
        "param_overrides.TRAIL_MULT_MIN": 2.5,
    }))
    c.append(("be_ultra_wide", {
        "flags.plus_1r_partial": True,
        "param_overrides.TRAIL_MULT_BASE": 5.0,
        "param_overrides.TRAIL_MULT_MIN": 3.0,
    }))
    c.append(("be_no_core_tighten", {
        "flags.plus_1r_partial": True,
        "param_overrides.TRAIL_MULT_BASE": 3.5,
        "param_overrides.TRAIL_CORE_TRANSITION_REDUCTION": 1.0,
    }))
    c.append(("be_wide_evening_trail", {
        "flags.plus_1r_partial": True,
        "param_overrides.TRAIL_MULT_BASE": 3.5,
        "param_overrides.TRAIL_WINDOW_MULT": {"OPEN": 1.0, "CORE": 0.60, "CLOSE": 1.0, "EVENING": 1.0},
    }))

    # Strategy B: Keep No-Trail, Add MFE Protection
    c.append(("stale_mfe_exempt_050", {
        "flags.stale_mfe_exempt": True,
    }))
    c.append(("stale_mfe_exempt_075", {
        "flags.stale_mfe_exempt": True,
        "param_overrides.STALE_MFE_EXEMPT_R": 0.75,
    }))
    c.append(("stale_mfe_exempt_030", {
        "flags.stale_mfe_exempt": True,
        "param_overrides.STALE_MFE_EXEMPT_R": 0.30,
    }))
    c.append(("mfe_ratchet_low", {
        "flags.mfe_ratchet": True,
        "param_overrides.MFE_RATCHET_TIERS": [(0.50, 0.10), (1.00, 0.30), (2.00, 0.80), (3.00, 1.50)],
    }))
    c.append(("mfe_ratchet_aggressive", {
        "flags.mfe_ratchet": True,
        "param_overrides.MFE_RATCHET_TIERS": [(0.40, 0.05), (0.75, 0.20), (1.50, 0.60), (2.50, 1.30)],
    }))

    # Strategy C: Stale/Free-Ride Parameter Tuning
    c.append(("stale_bars_6", {"param_overrides.STALE_BARS_15M": 6}))
    c.append(("stale_bars_12", {"param_overrides.STALE_BARS_15M": 12}))
    c.append(("stale_r_015", {"param_overrides.STALE_R": 0.15}))
    c.append(("stale_r_050", {"param_overrides.STALE_R": 0.50}))
    c.append(("hold_weekday_075", {"param_overrides.HOLD_WEEKDAY_R": 0.75}))
    c.append(("hold_weekday_050", {"param_overrides.HOLD_WEEKDAY_R": 0.50}))
    c.append(("hold_friday_100", {"param_overrides.HOLD_FRIDAY_R": 1.0}))
    c.append(("overnight_atr_03", {"param_overrides.OVERNIGHT_ATR_MULT": 0.3}))
    c.append(("overnight_atr_07", {"param_overrides.OVERNIGHT_ATR_MULT": 0.7}))
    c.append(("max_duration_64", {"param_overrides.MAX_POSITION_BARS_15M": 64}))
    c.append(("max_duration_192", {"param_overrides.MAX_POSITION_BARS_15M": 192}))

    # Combination experiments
    c.append(("be_wide_mfe_exempt", {
        "flags.plus_1r_partial": True,
        "flags.stale_mfe_exempt": True,
        "param_overrides.TRAIL_MULT_BASE": 3.5,
    }))
    c.append(("mfe_exempt_hold_low", {
        "flags.stale_mfe_exempt": True,
        "param_overrides.HOLD_WEEKDAY_R": 0.50,
    }))

    return c


def _phase_2_entry(prior: dict) -> list[tuple[str, dict]]:
    """Phase 2: Entry precision & filter tuning (~20 candidates)."""
    c: list[tuple[str, dict]] = []

    # Entry quality experiments
    c.append(("disable_early_kill", {"flags.early_kill": False}))
    c.append(("early_kill_r_015", {"param_overrides.EARLY_KILL_R": -0.15}))
    c.append(("early_kill_r_040", {"param_overrides.EARLY_KILL_R": -0.40}))
    c.append(("early_kill_bars_6", {"param_overrides.EARLY_KILL_BARS": 6}))
    c.append(("vwap_cap_core_070", {"param_overrides.VWAP_CAP_CORE": 0.70}))
    c.append(("vwap_cap_core_060", {"param_overrides.VWAP_CAP_CORE": 0.60}))
    c.append(("vwap_cap_open_055", {"param_overrides.VWAP_CAP_OPEN_EVE": 0.55}))
    c.append(("extension_skip_09", {"param_overrides.EXTENSION_SKIP_ATR": 0.9}))
    c.append(("touch_lb_4", {"param_overrides.TOUCH_LOOKBACK_15M": 4}))
    c.append(("touch_lb_12", {"param_overrides.TOUCH_LOOKBACK_15M": 12}))

    # Filter relaxation experiments
    c.append(("relax_hourly_mult", {"param_overrides.HOURLY_NEUTRAL_MULT": 0.80}))
    c.append(("relax_hourly_mult_100", {"param_overrides.HOURLY_NEUTRAL_MULT": 1.00}))
    c.append(("disable_slope_gate", {"flags.slope_gate": False}))
    c.append(("relax_momentum_floor", {"param_overrides.FLOOR_PCT": 0.15}))
    c.append(("relax_momentum_floor_010", {"param_overrides.FLOOR_PCT": 0.10}))

    # Other entry tuning
    c.append(("mom_n_35", {"param_overrides.MOM_N": 35}))
    c.append(("mom_n_65", {"param_overrides.MOM_N": 65}))
    c.append(("max_stop_150", {"param_overrides.MAX_STOP_POINTS": 150}))
    c.append(("atr_stop_12", {"param_overrides.ATR_STOP_MULT": 1.2}))
    c.append(("combo_tight_precise", {
        "param_overrides.VWAP_CAP_CORE": 0.65,
        "param_overrides.EXTENSION_SKIP_ATR": 0.9,
        "param_overrides.TOUCH_LOOKBACK_15M": 4,
    }))

    return c


def _phase_3_signal(prior: dict) -> list[tuple[str, dict]]:
    """Phase 3: Signal discrimination & regime (~18 candidates)."""
    c: list[tuple[str, dict]] = []

    # Regime & sizing
    c.append(("chop_35", {"param_overrides.CHOP_THRESHOLD": 35}))
    c.append(("chop_50", {"param_overrides.CHOP_THRESHOLD": 50}))
    c.append(("max_longs_1", {"param_overrides.MAX_LONGS_PER_DAY": 1}))
    c.append(("max_longs_3", {"param_overrides.MAX_LONGS_PER_DAY": 3}))
    c.append(("nopred_050", {"param_overrides.CLASS_MULT_NOPRED": 0.50}))
    c.append(("nopred_100", {"param_overrides.CLASS_MULT_NOPRED": 1.00}))
    c.append(("evening_block", {"param_overrides.VWAP_CAP_EVENING": 0.01}))
    c.append(("evening_tight", {"param_overrides.VWAP_CAP_EVENING": 0.35}))
    c.append(("vol_high_050", {
        "param_overrides.VOL_FACTOR": {"Normal": 1.0, "High": 0.50, "Shock": 0.0},
    }))
    c.append(("vol_high_040", {
        "param_overrides.VOL_FACTOR": {"Normal": 1.0, "High": 0.40, "Shock": 0.0},
    }))

    # Structural entry types
    c.append(("enable_type_b", {"param_overrides.USE_TYPE_B": True}))
    c.append(("enable_late_shoulder", {"param_overrides.USE_LATE_SHOULDER": True}))
    c.append(("enable_vwap_fail_evening", {"flags.vwap_fail_evening": True}))

    # Other signal experiments
    c.append(("disable_event_blocking", {"flags.event_blocking": False}))
    c.append(("drawdown_throttle_off", {"flags.drawdown_throttle": False}))
    c.append(("combo_selective_evening_block", {
        "param_overrides.MAX_LONGS_PER_DAY": 1,
        "param_overrides.VWAP_CAP_EVENING": 0.01,
    }))
    c.append(("combo_full_size_vol_cut", {
        "param_overrides.CLASS_MULT_NOPRED": 1.00,
        "param_overrides.VOL_FACTOR": {"Normal": 1.0, "High": 0.40, "Shock": 0.0},
    }))
    c.append(("enable_entry_quality_gate", {
        "flags.entry_quality_gate": True,
        "param_overrides.EQS_MIN_RTH": 2,
    }))

    return c


def _phase_4_finetune(prior: dict) -> list[tuple[str, dict]]:
    """Phase 4: Fine-tune & interaction (~20+ dynamic candidates)."""
    c: list[tuple[str, dict]] = []

    # Static candidates
    c.append(("base_risk_008", {"param_overrides.BASE_RISK_PCT": 0.008}))
    c.append(("base_risk_012", {"param_overrides.BASE_RISK_PCT": 0.012}))

    # Dynamic candidates: +/-10/20% variants of accepted numeric mutations
    numeric_keys = []
    for key, val in prior.items():
        if key.startswith("param_overrides.") and isinstance(val, (int, float)):
            numeric_keys.append((key, val))

    for key, val in numeric_keys:
        short = key.split(".")[-1].lower()
        for mult, label in [(0.80, "m20"), (0.90, "m10"), (1.10, "p10"), (1.20, "p20")]:
            new_val = round(val * mult, 6)
            if new_val != val:
                c.append((f"finetune_{short}_{label}", {key: new_val}))

    # Cross-phase combos: combine strongest P1 + P2 + P3 mutations
    # (these are already in cumulative_mutations, so adding individual
    # finetune variants covers cross-phase interaction testing)

    return c
