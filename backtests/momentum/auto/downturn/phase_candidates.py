"""Downturn per-phase candidate selection from experiment categories.

R6 Rev2: fixes structural failures from R6 Rev1 (zero acceptances).

Phase 1: Regime/signal (~17) -- DRAWDOWN_OVERRIDE + PROGRESSIVE_SMA (V2) + MOMENTUM_SIGNAL (V2)
         Fixed: progressive SMA now has direct regime override, momentum has cooldown gate
Phase 2: Exits/trail (~18) -- TRAIL_REDESIGN + EXIT_ADAPTIVE (subset)
         New: chandelier as primary trail + profit floor as safety net
Phase 3: Interaction (~6) -- combined regime+exit experiments
Phase 4: Adaptive +/-10-20% around prior optima (includes R5 + R6 params)
Phase 5: Exit management optimization -- profit floor, adaptive lock, chandelier, breakeven, scale-out
"""
from __future__ import annotations

from backtests.momentum.auto.downturn.experiment_categories import (
    get_category_experiments,
)


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Get experiment candidates for a specific phase.

    Args:
        phase: Optimization phase (1-5)
        prior_mutations: Cumulative mutations from prior phases (for phase 4 finetune, phase 5 exits)
        suggested_experiments: Analyzer-suggested experiments (prepended)
    """
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates()
    elif phase == 4:
        candidates = _phase_4_candidates(prior_mutations or {})
    elif phase == 5:
        candidates = _phase_5_candidates()
    else:
        candidates = []

    # Prepend suggested experiments from analyzer
    if suggested_experiments:
        # Deduplicate by name
        existing_names = {name for name, _ in candidates}
        new = [(n, m) for n, m in suggested_experiments if n not in existing_names]
        candidates = new + candidates

    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """R6 Rev2 regime/signal -- fixed progressive SMA + momentum cooldown."""
    return get_category_experiments([
        "DRAWDOWN_OVERRIDE", "PROGRESSIVE_SMA", "MOMENTUM_SIGNAL",
    ])


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """R6 Rev2 exits/trail -- chandelier trail + adaptive profit floor."""
    candidates = get_category_experiments(["TRAIL_REDESIGN"])
    # Add subset of EXIT_ADAPTIVE (most promising: threshold + aggressive variants)
    ea_exps = get_category_experiments(["EXIT_ADAPTIVE"])
    ea_subset = [e for e in ea_exps if any(k in e[0] for k in ["thresh", "aggressive", "high_cap", "base"])]
    candidates.extend(ea_subset)
    return candidates


def _phase_3_candidates() -> list[tuple[str, dict]]:
    """Interaction -- combined regime+exit experiments."""
    exps: list[tuple[str, dict]] = []
    # Best regime features + best trail features (interaction test)
    for cd in [24, 36]:
        exps.append((f"interact_ms_trail_cd{cd}", {
            "flags.momentum_signal": True,
            "flags.chandelier_trailing": True,
            "param_overrides.momentum_cooldown_bars": cd,
            "param_overrides.profit_floor_r_threshold": 1.5,
        }))
    exps.append(("interact_psma_trail", {
        "flags.progressive_sma": True,
        "flags.chandelier_trailing": True,
        "flags.adaptive_profit_floor": True,
        "param_overrides.profit_floor_r_threshold": 1.5,
    }))
    exps.append(("interact_dd_trail", {
        "flags.drawdown_regime_override": True,
        "flags.chandelier_trailing": True,
        "param_overrides.profit_floor_r_threshold": 1.5,
    }))
    exps.append(("interact_all_regime_trail", {
        "flags.momentum_signal": True,
        "flags.progressive_sma": True,
        "flags.drawdown_regime_override": True,
        "flags.chandelier_trailing": True,
        "flags.adaptive_profit_floor": True,
        "param_overrides.momentum_cooldown_bars": 36,
        "param_overrides.profit_floor_r_threshold": 1.5,
    }))
    exps.append(("interact_all_full", {
        "flags.momentum_signal": True,
        "flags.progressive_sma": True,
        "flags.drawdown_regime_override": True,
        "flags.chandelier_trailing": True,
        "flags.adaptive_profit_floor": True,
        "flags.min_hold_period": True,
        "flags.regime_adaptive_chandelier": True,
        "param_overrides.momentum_cooldown_bars": 36,
        "param_overrides.min_hold_bars": 12,
        "param_overrides.profit_floor_r_threshold": 1.5,
    }))
    return exps


def _phase_4_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Adaptive ±10-20% finetune around accepted numeric mutations."""
    candidates: list[tuple[str, dict]] = []

    if not prior_mutations:
        # Fallback: sample from all categories
        all_exps = get_category_experiments([
            "REGIME", "REVERSAL_SIGNAL", "BREAKDOWN_SIGNAL",
            "FADE_SIGNAL", "EXIT", "SIZING",
        ])
        return all_exps[:30]

    for key, value in prior_mutations.items():
        if not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue

        for pct_label, pct in [("m20", 0.80), ("m10", 0.90), ("p10", 1.10), ("p20", 1.20)]:
            new_val = value * pct
            if isinstance(value, int):
                new_val = int(round(new_val))
                if new_val == value:
                    continue
            else:
                new_val = round(new_val, 6)

            name = f"finetune_{key.replace('.', '_')}_{pct_label}"
            candidates.append((name, {key: new_val}))

    return candidates


def _phase_5_candidates() -> list[tuple[str, dict]]:
    """Phase 5: Exit management optimization -- reduce give-back, improve capture.

    ~34 candidates in 9 categories:
    A. Profit floor threshold  B. Profit floor lock fraction  C. Min hold period
    D. Adaptive lock tiers  E. Chandelier trailing  F. Breakeven stop
    G. Scale-out  H. Momentum pruning  I. Interaction combos
    """
    exps: list[tuple[str, dict]] = []

    # A. Profit floor threshold -- lower from 1.35R to capture moderate winners
    exps.append(("exit_pf_threshold_080", {"param_overrides.profit_floor_r_threshold": 0.80}))
    exps.append(("exit_pf_threshold_100", {"param_overrides.profit_floor_r_threshold": 1.00}))
    exps.append(("exit_pf_threshold_120", {"param_overrides.profit_floor_r_threshold": 1.20}))

    # B. Profit floor lock fraction -- pair with threshold changes
    exps.append(("exit_pf_lock_040", {"param_overrides.profit_floor_lock_pct": 0.40}))
    exps.append(("exit_pf_lock_060", {"param_overrides.profit_floor_lock_pct": 0.60}))

    # C. Min hold period -- shorter hold lets profit floor activate on flash crashes
    exps.append(("exit_min_hold_006", {"param_overrides.min_hold_bars": 6}))
    exps.append(("exit_min_hold_012", {"param_overrides.min_hold_bars": 12}))
    exps.append(("exit_min_hold_018", {"param_overrides.min_hold_bars": 18}))

    # D. Adaptive lock tiers -- more aggressive profit locking on big winners
    exps.append(("exit_adapt_t1_150", {"param_overrides.adaptive_lock_t1": 1.5}))
    exps.append(("exit_adapt_t1_300", {"param_overrides.adaptive_lock_t1": 3.0}))
    exps.append(("exit_adapt_bonus1_015", {"param_overrides.adaptive_lock_bonus_1": 0.15}))
    exps.append(("exit_adapt_bonus1_020", {"param_overrides.adaptive_lock_bonus_1": 0.20}))
    exps.append(("exit_adapt_bonus2_020", {"param_overrides.adaptive_lock_bonus_2": 0.20}))
    exps.append(("exit_adapt_bonus3_035", {"param_overrides.adaptive_lock_bonus_3": 0.35}))

    # E. Chandelier trailing -- tighten/loosen trailing stop width
    exps.append(("exit_chand_floor_180", {"param_overrides.chandelier_mult_floor": 1.8}))
    exps.append(("exit_chand_floor_200", {"param_overrides.chandelier_mult_floor": 2.0}))
    exps.append(("exit_chand_floor_250", {"param_overrides.chandelier_mult_floor": 2.5}))
    exps.append(("exit_chand_ceil_350", {"param_overrides.chandelier_mult_ceiling": 3.5}))
    exps.append(("exit_chand_ceil_450", {"param_overrides.chandelier_mult_ceiling": 4.5}))
    exps.append(("exit_chand_other_065", {"param_overrides.chandelier_regime_mult_other": 0.65}))

    # F. Breakeven stop -- faster capital protection
    exps.append(("exit_be_trigger_050", {"param_overrides.be_trigger_r": 0.5}))
    exps.append(("exit_be_trigger_075", {"param_overrides.be_trigger_r": 0.75}))
    exps.append(("exit_be_buffer_010", {"param_overrides.be_stop_buffer_mult": 0.10}))

    # G. Scale-out -- partial profit lock on big winners
    exps.append(("exit_scaleout_2r", {
        "flags.scale_out_enabled": True, "param_overrides.scale_out_target_r": 2.0,
    }))
    exps.append(("exit_scaleout_3r", {
        "flags.scale_out_enabled": True, "param_overrides.scale_out_target_r": 3.0,
    }))
    exps.append(("exit_scaleout_4r", {
        "flags.scale_out_enabled": True, "param_overrides.scale_out_target_r": 4.0,
    }))

    # H. Momentum pruning -- tighten/remove weak sub-engine
    exps.append(("signal_mom_cooldown_36", {"param_overrides.momentum_cooldown_bars": 36}))
    exps.append(("signal_mom_cooldown_48", {"param_overrides.momentum_cooldown_bars": 48}))
    exps.append(("signal_mom_disable", {"flags.momentum_signal": False}))

    # I. Interaction combos -- coherent multi-param changes
    exps.append(("combo_low_thresh_low_hold", {
        "param_overrides.profit_floor_r_threshold": 0.80,
        "param_overrides.min_hold_bars": 12,
    }))
    exps.append(("combo_low_thresh_tight_chand", {
        "param_overrides.profit_floor_r_threshold": 1.00,
        "param_overrides.chandelier_mult_floor": 2.0,
    }))
    exps.append(("combo_scaleout_adapt", {
        "flags.scale_out_enabled": True,
        "param_overrides.scale_out_target_r": 3.0,
        "param_overrides.adaptive_lock_bonus_1": 0.15,
    }))
    exps.append(("combo_early_be_low_hold", {
        "param_overrides.be_trigger_r": 0.5,
        "param_overrides.min_hold_bars": 12,
    }))
    exps.append(("combo_aggressive_capture", {
        "param_overrides.profit_floor_r_threshold": 0.80,
        "param_overrides.min_hold_bars": 12,
        "param_overrides.adaptive_lock_bonus_1": 0.20,
        "param_overrides.chandelier_mult_floor": 2.0,
    }))

    return exps
