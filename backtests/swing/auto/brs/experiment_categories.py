"""BRS experiment definitions — ~120 experiments in 6 named categories.

Each experiment is a (name, mutations_dict) tuple.
Categories: REGIME, ENTRY, EXIT, SIZING, VOLATILITY
"""
from __future__ import annotations


def _regime_experiments() -> list[tuple[str, dict]]:
    """~20 REGIME experiments: ADX thresholds, EMA periods, conviction."""
    exps = []

    # ADX on/off sweeps per symbol
    for sym in ["QQQ", "GLD"]:
        for on_val in [14, 16, 18, 20]:
            exps.append((
                f"regime_adx_on_{sym}_{on_val}",
                {f"symbol_configs.{sym}.adx_on": on_val, f"symbol_configs.{sym}.adx_off": on_val - 2},
            ))

    # ADX strong threshold
    for val in [25, 26, 28, 30, 32, 35]:
        exps.append((f"regime_adx_strong_{val}", {"adx_strong": val}))

    # EMA periods
    for fast in [15, 20, 25]:
        exps.append((f"regime_ema_fast_{fast}", {"ema_fast_period": fast}))
    for slow in [40, 50, 60]:
        exps.append((f"regime_ema_slow_{slow}", {"ema_slow_period": slow}))

    return exps


def _entry_experiments() -> list[tuple[str, dict]]:
    """~40 ENTRY experiments: LH/BD parameter sweeps, ablations."""
    exps = []

    # LH Rejection params
    for val in [3, 5, 7]:
        exps.append((f"entry_lh_swing_lookback_{val}", {"lh_swing_lookback": val}))
    for val in [20, 30, 40, 60]:
        exps.append((f"entry_lh_arm_bars_{val}", {"lh_arm_bars": val}))
    for val in [2.0, 2.5, 3.0, 3.5]:
        exps.append((f"entry_lh_max_stop_atr_{val}", {"lh_max_stop_atr": val}))

    # V-reversal filter sweep
    for val in [1.0, 1.25, 1.5, 2.0, 2.5]:
        exps.append((f"entry_v_reversal_max_{val}", {"v_reversal_max_rally": val}))

    # BEAR_TREND volume filter for LHR
    for val in [1.0, 1.2, 1.5, 2.0]:
        exps.append((f"entry_bt_volume_mult_{val}", {"bt_volume_mult": val}))

    # BD Continuation params
    for val in [1.0, 1.2, 1.5, 2.0]:
        exps.append((f"entry_bd_volume_mult_{val}", {"bd_volume_mult": val}))
    for val in [0.20, 0.30, 0.40, 0.50]:
        exps.append((f"entry_bd_close_quality_{val}", {"bd_close_quality": val}))
    for val in [12, 16, 24, 32]:
        exps.append((f"entry_bd_arm_bars_{val}", {"bd_arm_bars": val}))
    # BD Donchian period (shorter = more breaks)
    for val in [10, 15, 20, 30]:
        exps.append((f"entry_bd_donchian_{val}", {"bd_donchian_period": val}))
    # BD ablation: restrict to BEAR_STRONG only (default allows BEAR_TREND too)
    exps.append(("entry_bd_no_bear_trend", {"bd_allow_bear_trend": False}))

    # TOD filter ablation + window variants
    exps.append(("entry_tod_filter_off", {"tod_filter_enabled": False}))
    exps.append(("entry_tod_narrow", {"tod_filter_start": 870, "tod_filter_end": 900}))  # 14:30-15:00
    exps.append(("entry_tod_wide", {"tod_filter_start": 810, "tod_filter_end": 960}))    # 13:30-16:00

    # Signal ablations
    exps.append(("entry_abl_no_lh", {"disable_lh": True}))
    exps.append(("entry_abl_no_bd", {"disable_bd": True}))
    # Re-enable legacy S1 for comparison
    exps.append(("entry_reenable_s1", {"disable_s1": False}))
    exps.append(("entry_reenable_s1_only", {"disable_s1": False, "disable_lh": True, "disable_bd": True}))

    return exps


def _exit_experiments() -> list[tuple[str, dict]]:
    """~20 EXIT experiments: stop multipliers, trailing, stale, time decay."""
    exps = []

    # Per-symbol daily_mult / hourly_mult / chand_mult
    for sym in ["QQQ", "GLD"]:
        for dm in [1.8, 2.0, 2.3, 2.5, 2.8]:
            exps.append((
                f"exit_daily_mult_{sym}_{dm}",
                {f"symbol_configs.{sym}.daily_mult": dm},
            ))

    # BE trigger
    for val in [0.5, 0.75, 1.0, 1.25, 1.5]:
        exps.append((f"exit_be_trigger_{val}", {"be_trigger_r": val}))

    # Trail trigger
    for val in [0.75, 1.0, 1.25, 1.5, 2.0]:
        exps.append((f"exit_trail_trigger_{val}", {"trail_trigger_r": val}))

    # Stale bars
    for val in [30, 40, 50, 60]:
        exps.append((f"exit_stale_short_{val}", {"stale_bars_short": val}))
    for val in [20, 25, 30, 40]:
        exps.append((f"exit_stale_long_{val}", {"stale_bars_long": val}))

    # Catastrophic cap
    for val in [1.5, 1.75, 2.0, 2.5]:
        exps.append((f"exit_catastrophic_{val}", {"catastrophic_cap_r": val}))

    # Time decay
    for val in [240, 300, 360, 480]:
        exps.append((f"exit_time_decay_{val}", {"time_decay_hours": val}))

    # Stale early bars
    for val in [20, 25, 35, 45, 55]:
        exps.append((f"exit_stale_early_{val}", {"stale_early_bars": val}))

    # Stop buffer ATR per symbol
    for sym in ["QQQ", "GLD"]:
        for val in [0.2, 0.3, 0.5, 0.6]:
            exps.append((f"exit_stop_buffer_{sym}_{val}",
                         {f"symbol_configs.{sym}.stop_buffer_atr": val}))

    # Stop floor ATR per symbol
    for sym in ["QQQ", "GLD"]:
        for val in [0.6, 0.8, 1.2, 1.5]:
            exps.append((f"exit_stop_floor_{sym}_{val}",
                         {f"symbol_configs.{sym}.stop_floor_atr": val}))

    # Chandelier multiplier per symbol (controls trail tightness)
    for sym in ["QQQ", "GLD"]:
        for val in [2.5, 3.0, 3.5, 4.0, 4.5]:
            exps.append((f"exit_chand_mult_{sym}_{val}",
                         {f"symbol_configs.{sym}.chand_mult": val}))

    # Profit floor scale (>1 = lock more profit at each MFE tier)
    for val in [0.5, 0.75, 1.0, 1.5, 2.0]:
        exps.append((f"exit_profit_floor_{val}", {"profit_floor_scale": val}))

    return exps


def _sizing_experiments() -> list[tuple[str, dict]]:
    """~15 SIZING experiments: risk pct, heat caps, max concurrent."""
    exps = []

    # Per-symbol base_risk_pct
    for sym in ["QQQ", "GLD"]:
        for risk in [0.003, 0.004, 0.005, 0.006]:
            exps.append((
                f"sizing_risk_{sym}_{risk}",
                {f"symbol_configs.{sym}.base_risk_pct": risk},
            ))

    # Heat caps
    for val in [1.25, 1.50, 1.80, 2.00, 2.25]:
        exps.append((f"sizing_heat_cap_{val}", {"heat_cap_r": val}))

    # Crisis heat cap
    for val in [0.80, 1.00, 1.25, 1.50]:
        exps.append((f"sizing_crisis_heat_{val}", {"crisis_heat_cap_r": val}))

    # Max concurrent
    for val in [2, 3, 4]:
        exps.append((f"sizing_max_concurrent_{val}", {"max_concurrent": val}))

    # --- Round 9: Pyramiding ---
    exps.append(("sizing_pyramid_on", {"pyramid_enabled": True}))
    for min_r in [0.5, 1.0, 1.5, 2.0]:
        exps.append((f"sizing_pyramid_min_r_{min_r}", {
            "pyramid_enabled": True, "pyramid_min_r": min_r}))
    for scale in [0.3, 0.5, 0.7, 1.0]:
        exps.append((f"sizing_pyramid_scale_{scale}", {
            "pyramid_enabled": True, "pyramid_scale": scale}))
    exps.append(("sizing_pyramid_2_adds", {
        "pyramid_enabled": True, "max_pyramid_adds": 2}))
    exps.append(("sizing_pyramid_conservative", {
        "pyramid_enabled": True, "pyramid_min_r": 1.5, "pyramid_scale": 0.3}))
    exps.append(("sizing_pyramid_aggressive", {
        "pyramid_enabled": True, "pyramid_min_r": 0.5, "pyramid_scale": 1.0}))

    # --- Round 9: Regime-adaptive sizing ---
    for val in [0.50, 0.65, 0.75, 0.85]:
        exps.append((f"sizing_chop_mult_{val}", {"size_mult_range_chop": val}))
    for val in [0.70, 0.80, 0.90]:
        exps.append((f"sizing_bs_mult_{val}", {"size_mult_bear_strong": val}))
    for val in [1.10, 1.20, 1.30]:
        exps.append((f"sizing_bt_mult_{val}", {"size_mult_bear_trend": val}))
    exps.append(("sizing_regime_adaptive_balanced", {
        "size_mult_bear_trend": 1.2, "size_mult_range_chop": 0.65,
        "size_mult_bear_strong": 0.85}))
    exps.append(("sizing_regime_adaptive_aggressive", {
        "size_mult_bear_trend": 1.3, "size_mult_range_chop": 0.50,
        "size_mult_bear_strong": 0.75}))

    return exps


def _volatility_experiments() -> list[tuple[str, dict]]:
    """~10 VOLATILITY experiments: ATR periods, VolFactor ranges."""
    exps = []

    # These are applied as param_overrides that the engine reads
    # VolFactor clamp range — via param_overrides
    for lo in [0.25, 0.35, 0.45]:
        exps.append((f"vol_vf_min_{lo}", {"param_overrides.vf_clamp_min": lo}))
    for hi in [1.2, 1.5, 1.8]:
        exps.append((f"vol_vf_max_{hi}", {"param_overrides.vf_clamp_max": hi}))

    # Extreme vol threshold
    for val in [90, 93, 95, 97]:
        exps.append((f"vol_extreme_thresh_{val}", {"param_overrides.extreme_vol_pct": val}))

    return exps


def _structural_experiments() -> list[tuple[str, dict]]:
    """~10 STRUCTURAL experiments: regime relaxation, quality gates, IBIT add-back."""
    exps = []
    # Regime bear min conditions (only 2 is meaningful; 3 is a no-op since
    # bear_conds==3 already returns BEAR_TREND before the BEAR_FORMING check)
    exps.append(("struct_bear_min_2", {"regime_bear_min_conditions": 2}))
    # Fast crash ablation (now enabled by default — Change #9)
    exps.append(("struct_fast_crash_off", {"fast_crash_enabled": False}))
    # Quality score gates
    for val in [0.50, 0.65, 0.75, 0.85]:
        exps.append((f"struct_quality_gate_{val}", {"min_quality_score": val}))
    # Key combinations (fast_crash already True by default)
    exps.append(("struct_bear2_crash", {
        "regime_bear_min_conditions": 2,
    }))
    # IBIT add-back experiment (test if new signals make IBIT viable)
    exps.append(("struct_add_ibit", {"symbols": ["QQQ", "GLD", "IBIT"]}))

    # Minimum hold time sweep (Change #4)
    for val in [0, 2, 3, 4, 5]:
        exps.append((f"struct_min_hold_{val}", {"min_hold_bars": val}))

    # Regime-adaptive cooldown sweeps (Change #5)
    for val in [1, 2, 3, 4]:
        exps.append((f"struct_cd_bear_strong_{val}", {"cooldown_bear_strong": val}))
    for val in [2, 3, 4, 6]:
        exps.append((f"struct_cd_bear_trend_{val}", {"cooldown_bear_trend": val}))

    # Wider BEAR_STRONG stop floor sweep (Change #7)
    for val in [1.0, 1.15, 1.3, 1.5]:
        exps.append((f"struct_stop_floor_bs_mult_{val}", {"stop_floor_bear_strong_mult": val}))

    # Trail regime scaling ablation (Change #8)
    exps.append(("struct_trail_regime_off", {"trail_regime_scaling": False}))

    # Scale-out experiments (Change #6)
    exps.append(("struct_scale_out_on", {"scale_out_enabled": True}))
    for val in [1.5, 2.0, 2.5, 3.0]:
        exps.append((f"struct_scale_out_target_{val}", {
            "scale_out_enabled": True, "scale_out_target_r": val,
        }))
    for val in [0.33, 0.50]:
        exps.append((f"struct_scale_out_pct_{val}", {
            "scale_out_enabled": True, "scale_out_pct": val,
        }))

    # Crash override experiments (Round 5)
    exps.append(("struct_crash_override_off", {"crash_override_enabled": False}))
    for val in [-0.02, -0.03, -0.04, -0.05]:
        exps.append((f"struct_crash_return_{val}", {"crash_override_return": val}))
    for val in [1.3, 1.5, 1.8, 2.0, 2.5]:
        exps.append((f"struct_crash_atr_ratio_{val}", {"crash_override_atr_ratio": val}))

    # BD BEAR_STRONG restriction (Round 5 — 25% WR in BEAR_STRONG)
    exps.append(("struct_bd_no_bear_strong", {"bd_allow_bear_strong": False}))

    # BEAR_FORMING entry with vol gate (Round 5)
    exps.append(("struct_forming_entry_on", {"forming_entry_enabled": True}))
    for val in [1.2, 1.5, 1.8, 2.0]:
        exps.append((f"struct_forming_vol_gate_{val}", {
            "forming_entry_enabled": True, "forming_vol_gate": val,
        }))

    # --- Round 7: Cumulative return Path G ---
    exps.append(("struct_cum_return_on", {"cum_return_enabled": True}))
    for val in [3, 5, 7, 10]:
        exps.append((f"struct_cum_return_lookback_{val}", {
            "cum_return_enabled": True, "cum_return_lookback": val}))
    for val in [-0.015, -0.02, -0.025, -0.03, -0.04, -0.05]:
        exps.append((f"struct_cum_return_thresh_{val}", {
            "cum_return_enabled": True, "cum_return_thresh": val}))
    exps.append(("struct_cum_return_3d_tight", {
        "cum_return_enabled": True, "cum_return_lookback": 3, "cum_return_thresh": -0.02}))
    exps.append(("struct_cum_return_7d_loose", {
        "cum_return_enabled": True, "cum_return_lookback": 7, "cum_return_thresh": -0.04}))
    exps.append(("struct_cum_return_10d_loose", {
        "cum_return_enabled": True, "cum_return_lookback": 10, "cum_return_thresh": -0.05}))
    exps.append(("struct_cum_return_plus_return_only", {
        "cum_return_enabled": True, "return_only_enabled": True}))
    exps.append(("struct_cum_return_plus_peak", {
        "cum_return_enabled": True, "peak_drop_enabled": True}))

    # --- Round 6: Structural regime detection ---

    # Peak drawdown trigger (Part A)
    exps.append(("struct_peak_drop_on", {"peak_drop_enabled": True}))
    for val in [10, 15, 20, 30, 40]:
        exps.append((f"struct_peak_drop_lookback_{val}", {
            "peak_drop_enabled": True, "peak_drop_lookback": val,
        }))
    for val in [-0.02, -0.03, -0.04, -0.05, -0.07]:
        exps.append((f"struct_peak_drop_pct_{val}", {
            "peak_drop_enabled": True, "peak_drop_pct": val,
        }))

    # Return-only bias confirmation (Part B)
    exps.append(("struct_return_only_on", {"return_only_enabled": True}))
    for val in [-0.010, -0.015, -0.020, -0.025, -0.030]:
        exps.append((f"struct_return_only_{val}", {
            "return_only_enabled": True, "return_only_thresh": val,
        }))

    # 4H bias accelerator (Part C)
    exps.append(("struct_4h_accel_on", {"bias_4h_accel_enabled": True}))
    exps.append(("struct_4h_accel_red_2", {
        "bias_4h_accel_enabled": True, "bias_4h_accel_reduction": 2,
    }))

    # Combinations
    exps.append(("struct_peak_return_combo", {
        "peak_drop_enabled": True, "return_only_enabled": True,
    }))
    exps.append(("struct_regime_all_three", {
        "peak_drop_enabled": True, "return_only_enabled": True,
        "bias_4h_accel_enabled": True,
    }))

    # --- Round 8: Signal coverage expansion ---

    # RANGE_CHOP+SHORT entry override
    exps.append(("struct_chop_short_on", {"chop_short_entry_enabled": True}))
    # (struct_chop_short_peak_combo removed — redundant with R7.2 baseline peak_drop=True)

    # Churn bridge
    for val in [1, 2, 3, 5]:
        exps.append((f"struct_churn_bridge_{val}", {"churn_bridge_bars": val}))

    # GLD cross-symbol override
    exps.append(("struct_gld_qqq_override_on", {"gld_qqq_override_enabled": True}))
    exps.append(("struct_gld_qqq_plus_low_adx", {
        "gld_qqq_override_enabled": True,
        "symbol_configs.GLD.adx_on": 14,
        "symbol_configs.GLD.adx_off": 12,
    }))
    exps.append(("struct_gld_qqq_bear_trend", {
        "gld_qqq_override_enabled": True, "gld_qqq_override_bear_trend": True}))

    # Combinations of R8 features
    exps.append(("struct_r8_chop_churn", {
        "chop_short_entry_enabled": True, "churn_bridge_bars": 3}))
    exps.append(("struct_r8_all_three", {
        "chop_short_entry_enabled": True, "churn_bridge_bars": 3,
        "gld_qqq_override_enabled": True}))
    exps.append(("struct_r8_all_four", {
        "chop_short_entry_enabled": True, "churn_bridge_bars": 3,
        "gld_qqq_override_enabled": True, "persistence_override_bars": 10}))

    # R8 + previously disabled feature combos
    exps.append(("struct_forming_chop_combo", {
        "forming_entry_enabled": True, "chop_short_entry_enabled": True}))
    exps.append(("struct_4h_accel_chop_combo", {
        "bias_4h_accel_enabled": True, "chop_short_entry_enabled": True}))

    # Persistence override: relax signal structure during SHORT-bias droughts
    for val in [5, 8, 10, 15]:
        exps.append((f"struct_persistence_{val}", {"persistence_override_bars": val}))
    exps.append(("struct_persistence_chop", {
        "persistence_override_bars": 10, "chop_short_entry_enabled": True}))
    exps.append(("struct_persistence_churn", {
        "persistence_override_bars": 10, "churn_bridge_bars": 3}))

    # --- Round 9: Fine-tuning around R8 optimals ---

    # Persistence override: narrow sweep around R8's selected 5
    for val in [3, 4, 6, 7]:
        exps.append((f"struct_persistence_ft_{val}", {"persistence_override_bars": val}))

    # Peak drop sensitivity: narrow sweep around R8's -0.02
    for val in [-0.01, -0.015, -0.025, -0.03]:
        exps.append((f"struct_peak_drop_pct_ft_{val}", {
            "peak_drop_enabled": True, "peak_drop_pct": val}))

    # Peak drop lookback: test alternatives to 15
    for val in [10, 12, 20]:
        exps.append((f"struct_peak_drop_lookback_ft_{val}", {
            "peak_drop_enabled": True, "peak_drop_lookback": val}))

    # BD close quality: sweep around default 0.45
    for val in [0.35, 0.40, 0.50, 0.55]:
        exps.append((f"entry_bd_close_quality_ft_{val}", {"bd_close_quality": val}))

    # BD max stop ATR: tighter/looser than default 3.0
    for val in [2.0, 2.5, 3.5]:
        exps.append((f"entry_bd_max_stop_ft_{val}", {"bd_max_stop_atr": val}))

    # Chop quality multiplier: sweep
    for val in [0.45, 0.50, 0.55, 0.65, 0.70]:
        exps.append((f"struct_chop_quality_{val}", {"chop_quality_mult": val}))

    # LH persistence quality: sweep around 0.50
    for val in [0.35, 0.40, 0.45, 0.55, 0.60]:
        exps.append((f"struct_persist_lh_quality_{val}", {"persist_quality_mult_lh": val}))

    # BD persistence quality: sweep (NEW parameter)
    for val in [0.40, 0.50, 0.60, 0.70, 0.80]:
        exps.append((f"struct_persist_bd_quality_{val}", {"persist_quality_mult_bd": val}))

    return exps


def _signal_select_experiments() -> list[tuple[str, dict]]:
    """Signal combination experiments — tested AFTER params are tuned (Change #2)."""
    return [
        ("select_all_signals", {"disable_s1": False, "disable_lh": False, "disable_bd": False}),
        ("select_s1_lh", {"disable_s1": False, "disable_lh": False, "disable_bd": True}),
        ("select_s1_only", {"disable_s1": False, "disable_lh": True, "disable_bd": True}),
        ("select_lh_only", {"disable_s1": True, "disable_lh": False, "disable_bd": True}),
        ("select_lh_bd", {"disable_s1": True, "disable_lh": False, "disable_bd": False}),
        ("select_s1_bd", {"disable_s1": False, "disable_lh": True, "disable_bd": False}),
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXPERIMENT_CATEGORIES: dict[str, list[tuple[str, dict]]] = {
    "REGIME": _regime_experiments(),
    "ENTRY": _entry_experiments(),
    "EXIT": _exit_experiments(),
    "SIZING": _sizing_experiments(),
    "VOLATILITY": _volatility_experiments(),
    "STRUCTURAL": _structural_experiments(),
    "SIGNAL_SELECT": _signal_select_experiments(),
}


def get_all_experiments() -> list[tuple[str, dict]]:
    """Return all experiments across all categories."""
    all_exps = []
    for cat_exps in EXPERIMENT_CATEGORIES.values():
        all_exps.extend(cat_exps)
    return all_exps


def get_category_experiments(categories: list[str]) -> list[tuple[str, dict]]:
    """Return experiments for specific categories."""
    exps = []
    for cat in categories:
        exps.extend(EXPERIMENT_CATEGORIES.get(cat, []))
    return exps
