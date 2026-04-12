"""ATRSS per-phase candidate experiments.

Phase 1: Exit Cleanup (26 candidates)
Phase 2: Signal & Filtering (20 candidates)
Phase 3: Entry & Fill Optimization (20 candidates)
Phase 4: Sizing & Fine-tune (16 static + ~14 dynamic)
"""
from __future__ import annotations


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Get experiment candidates for a specific phase."""
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates()
    elif phase == 4:
        candidates = _phase_4_candidates(prior_mutations or {})
    else:
        candidates = []

    if suggested_experiments:
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


# ---------------------------------------------------------------------------
# Phase 1: Exit Cleanup (26 candidates)
# ---------------------------------------------------------------------------

def _phase_1_candidates() -> list[tuple[str, dict]]:
    candidates = []

    # Structural (2)
    candidates.append(("disable_early_stall", {"flags.early_stall_exit": False}))
    candidates.append(("reenable_full_stall", {"flags.stall_exit": True}))

    # TP levels (8)
    candidates.append(("tp1_r_075", {"param_overrides.tp1_r": 0.75}))
    candidates.append(("tp1_r_125", {"param_overrides.tp1_r": 1.25}))
    candidates.append(("tp1_r_150", {"param_overrides.tp1_r": 1.50}))
    candidates.append(("tp1_frac_025", {"param_overrides.tp1_frac": 0.25}))
    candidates.append(("tp1_frac_050", {"param_overrides.tp1_frac": 0.50}))
    candidates.append(("tp2_r_150", {"param_overrides.tp2_r": 1.50}))
    candidates.append(("tp2_r_250", {"param_overrides.tp2_r": 2.50}))
    candidates.append(("tp2_frac_050", {"param_overrides.tp2_frac": 0.50}))

    # BE & trailing (5)
    candidates.append(("be_trigger_075r", {"param_overrides.be_trigger_r": 0.75}))
    candidates.append(("be_trigger_100r", {"param_overrides.be_trigger_r": 1.0}))
    candidates.append(("be_trigger_150r", {"param_overrides.be_trigger_r": 1.5}))
    candidates.append(("be_offset_005", {"param_overrides.be_atr_offset": 0.05}))
    candidates.append(("chand_trigger_075r", {"param_overrides.chandelier_trigger_r": 0.75}))

    # Time decay (5)
    candidates.append(("max_hold_240h", {"param_overrides.max_hold_hours": 240}))
    candidates.append(("max_hold_360h", {"param_overrides.max_hold_hours": 360}))
    candidates.append(("early_stall_8h", {"param_overrides.early_stall_check_hours": 8}))
    candidates.append(("early_stall_mfe_020", {"param_overrides.early_stall_mfe_threshold": 0.20}))
    candidates.append(("early_stall_mfe_050", {"param_overrides.early_stall_mfe_threshold": 0.50}))

    # Profit floor (4) -- dict type
    candidates.append(("floor_tight_low", {
        "param_overrides.profit_floor": {1.0: 0.1, 1.5: 0.35, 2.0: 0.85, 3.0: 1.6, 4.0: 2.6},
    }))
    candidates.append(("floor_tight_all", {
        "param_overrides.profit_floor": {1.0: 0.2, 1.5: 0.50, 2.0: 1.0, 3.0: 1.8, 4.0: 2.8},
    }))
    candidates.append(("floor_loose", {
        "param_overrides.profit_floor": {1.0: -0.25, 1.5: 0.15, 2.0: 0.60, 3.0: 1.3, 4.0: 2.3},
    }))
    candidates.append(("floor_short_tight", {
        "param_overrides.profit_floor_short": {0.75: 0.20, 1.0: 0.60, 1.5: 1.1, 2.0: 1.35},
    }))

    # Stop tightening (2)
    candidates.append(("trend_stop_tight_075", {"param_overrides.trend_stop_tightening": 0.75}))
    candidates.append(("trend_stop_tight_095", {"param_overrides.trend_stop_tightening": 0.95}))

    return candidates


# ---------------------------------------------------------------------------
# Phase 2: Signal & Filtering (20 candidates)
# ---------------------------------------------------------------------------

def _phase_2_candidates() -> list[tuple[str, dict]]:
    candidates = []

    # Structural toggles (5)
    candidates.append(("disable_breakout", {"flags.breakout_entries": False}))
    candidates.append(("disable_momentum_filter", {"flags.momentum_filter": False}))
    candidates.append(("disable_reset_requirement", {"flags.reset_requirement": False}))
    candidates.append(("disable_prior_high", {"flags.prior_high_confirm": False}))
    candidates.append(("disable_hysteresis_gap", {"flags.hysteresis_gap": False}))

    # Quality gate (3)
    candidates.append(("enable_quality_30", {
        "flags.quality_gate": True,
        "param_overrides.quality_gate_threshold": 3.0,
    }))
    candidates.append(("enable_quality_40", {
        "flags.quality_gate": True,
        "param_overrides.quality_gate_threshold": 4.0,
    }))
    candidates.append(("enable_quality_50", {
        "flags.quality_gate": True,
        "param_overrides.quality_gate_threshold": 5.0,
    }))

    # Confirmation tuning (5)
    candidates.append(("fast_confirm_score_45", {"param_overrides.fast_confirm_score": 45}))
    candidates.append(("fast_confirm_score_65", {"param_overrides.fast_confirm_score": 65}))
    candidates.append(("fast_confirm_adx_18", {"param_overrides.fast_confirm_adx": 18}))
    candidates.append(("confirm_days_0", {"param_overrides.confirm_days_normal": 0}))
    candidates.append(("adx_strong_25", {"param_overrides.adx_strong": 25}))

    # Regime sensitivity (3)
    candidates.append(("adx_on_15", {"param_overrides.adx_on": 15}))
    candidates.append(("adx_on_17", {"param_overrides.adx_on": 17}))
    candidates.append(("adx_off_14", {"param_overrides.adx_off": 14}))

    # Momentum/pullback tuning (4)
    candidates.append(("momentum_tol_005", {"param_overrides.momentum_tolerance_atr": 0.05}))
    candidates.append(("momentum_tol_020", {"param_overrides.momentum_tolerance_atr": 0.20}))
    candidates.append(("recovery_strong_070", {"param_overrides.recovery_tolerance_atr_strong": 0.70}))
    candidates.append(("recovery_trend_035", {"param_overrides.recovery_tolerance_atr_trend": 0.35}))

    return candidates


# ---------------------------------------------------------------------------
# Phase 3: Entry & Fill Optimization (20 candidates)
# ---------------------------------------------------------------------------

def _phase_3_candidates() -> list[tuple[str, dict]]:
    candidates = []

    # Order management (4)
    candidates.append(("expiry_6h", {"param_overrides.order_expiry_hours": 6}))
    candidates.append(("expiry_12h", {"param_overrides.order_expiry_hours": 12}))
    candidates.append(("expiry_24h", {"param_overrides.order_expiry_hours": 24}))
    candidates.append(("expiry_36h", {"param_overrides.order_expiry_hours": 36}))

    # Entry slip tolerance (2)
    candidates.append(("disable_slippage_abort", {"flags.slippage_abort": False}))
    candidates.append(("slip_atr_050", {"param_overrides.max_entry_slip_atr": 0.50}))

    # Recovery tolerance (4)
    candidates.append(("recov_strong_060", {"param_overrides.recovery_tolerance_atr_strong": 0.60}))
    candidates.append(("recov_strong_095", {"param_overrides.recovery_tolerance_atr_strong": 0.95}))
    candidates.append(("recov_trend_050", {"param_overrides.recovery_tolerance_atr_trend": 0.50}))
    candidates.append(("recov_base_030", {"param_overrides.recovery_tolerance_atr": 0.30}))

    # Cooldown tuning (4)
    candidates.append(("cd_range_2h", {"param_overrides.cooldown_range": 2}))
    candidates.append(("cd_range_6h", {"param_overrides.cooldown_range": 6}))
    candidates.append(("cd_trend_0h", {"param_overrides.cooldown_trend": 0}))
    candidates.append(("cd_strong_0h", {"param_overrides.cooldown_strong": 0}))

    # Voucher tuning (4)
    candidates.append(("voucher_12h", {"param_overrides.voucher_valid_hours": 12}))
    candidates.append(("voucher_36h", {"param_overrides.voucher_valid_hours": 36}))
    candidates.append(("voucher_48h", {"param_overrides.voucher_valid_hours": 48}))
    candidates.append(("disable_voucher", {"flags.voucher_system": False}))

    # Pullback & EMA (2)
    candidates.append(("ema_pull_normal_35", {"param_overrides.ema_pull_normal": 35}))
    candidates.append(("ema_pull_normal_55", {"param_overrides.ema_pull_normal": 55}))

    return candidates


# ---------------------------------------------------------------------------
# Phase 4: Sizing & Fine-tune (16 static + ~14 dynamic)
# ---------------------------------------------------------------------------

def _phase_4_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    candidates = []

    # Pyramiding (6)
    candidates.append(("addon_a_100r", {"param_overrides.addon_a_r": 1.00}))
    candidates.append(("addon_a_150r", {"param_overrides.addon_a_r": 1.50}))
    candidates.append(("addon_b_150r", {"param_overrides.addon_b_r": 1.50}))
    candidates.append(("addon_b_100r", {"param_overrides.addon_b_r": 1.00}))
    candidates.append(("addon_a_size_050", {"param_overrides.addon_a_size_mult": 0.50}))
    candidates.append(("addon_b_size_025", {"param_overrides.addon_b_size_mult": 0.25}))

    # Portfolio heat (4)
    candidates.append(("heat_4pct", {"param_overrides.max_portfolio_heat": 0.04}))
    candidates.append(("heat_5pct", {"param_overrides.max_portfolio_heat": 0.05}))
    candidates.append(("heat_8pct", {"param_overrides.max_portfolio_heat": 0.08}))
    candidates.append(("heat_10pct", {"param_overrides.max_portfolio_heat": 0.10}))

    # Per-symbol stop & chandelier (4)
    candidates.append(("chand_mult_25", {"param_overrides.chand_mult": 2.5}))
    candidates.append(("chand_mult_35", {"param_overrides.chand_mult": 3.5}))
    candidates.append(("base_risk_008", {"param_overrides.base_risk_pct": 0.008}))
    candidates.append(("base_risk_005", {"param_overrides.base_risk_pct": 0.005}))

    # ADX slope gate (2)
    candidates.append(("adx_slope_neg3", {"param_overrides.adx_slope_gate": -3.0}))
    candidates.append(("adx_slope_neg1", {"param_overrides.adx_slope_gate": -1.0}))

    # Dynamic fine-tuning from Phase 1-3 winners (+-10%, +-20%)
    candidates.extend(_dynamic_finetune(prior_mutations))

    return candidates


def _dynamic_finetune(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Generate +-10% and +-20% variants for top numeric param_override winners."""
    finetune = []
    numeric_winners = {}

    for key, value in prior_mutations.items():
        if key.startswith("param_overrides.") and isinstance(value, (int, float)):
            numeric_winners[key] = value

    # Take up to 4 numeric winners for fine-tuning (14-16 experiments)
    for key, base_val in list(numeric_winners.items())[:4]:
        short_name = key.split(".", 1)[1]
        for pct_label, factor in [("plus10", 1.10), ("plus20", 1.20),
                                   ("minus10", 0.90), ("minus20", 0.80)]:
            new_val = base_val * factor
            # Preserve int type if original was int
            if isinstance(base_val, int):
                new_val = int(round(new_val))
            name = f"{short_name}_{pct_label}"
            finetune.append((name, {key: new_val}))

    return finetune
