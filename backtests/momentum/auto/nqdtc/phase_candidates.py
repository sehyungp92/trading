"""NQDTC phased auto-optimization experiment candidates.

Phase 1: Exit optimization (22 experiments) -- address 33% MFE capture
Phase 2: Signal discrimination (18 experiments) -- recalibrate quality/sizing
Phase 3: Timing & clustering (16 experiments) -- eliminate burst drag
Phase 4: Fine-tune & interaction (~22 dynamic experiments)
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
        candidates.extend(_phase_1_exit(prior))
    elif phase == 2:
        candidates.extend(_phase_2_signal(prior))
    elif phase == 3:
        candidates.extend(_phase_3_timing(prior))
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


def _phase_1_exit(prior: dict) -> list[tuple[str, dict]]:
    """Exit optimization: address 33% MFE capture ratio."""
    return [
        # TP1 R-target sweeps
        ("tp1_r_0.8", {"param_overrides.TP1_R": 0.8}),
        ("tp1_r_1.0", {"param_overrides.TP1_R": 1.0}),
        ("tp1_r_2.0", {"param_overrides.TP1_R": 2.0}),

        # TP1 partial close fraction
        ("tp1_partial_40", {"param_overrides.TP1_R": 1.5, "param_overrides.TP1_PARTIAL_PCT": 0.40}),
        ("tp1_partial_60", {"param_overrides.TP1_R": 1.5, "param_overrides.TP1_PARTIAL_PCT": 0.60}),
        ("tp1_partial_75", {"param_overrides.TP1_R": 1.5, "param_overrides.TP1_PARTIAL_PCT": 0.75}),

        # Re-enable TP2
        ("tp2_at_2.5R", {"param_overrides.TP2_R": 2.5, "param_overrides.TP2_PARTIAL_PCT": 0.25}),
        ("tp2_at_3.5R", {"param_overrides.TP2_R": 3.5, "param_overrides.TP2_PARTIAL_PCT": 0.25}),
        ("tp2_at_5.0R", {"param_overrides.TP2_R": 5.0, "param_overrides.TP2_PARTIAL_PCT": 0.25}),

        # Chandelier ATR mult sweeps
        ("chandelier_mult_1.5", {"param_overrides.CHANDELIER_ATR_MULT": 1.5}),
        ("chandelier_mult_2.0", {"param_overrides.CHANDELIER_ATR_MULT": 2.0}),
        ("chandelier_mult_3.5", {"param_overrides.CHANDELIER_ATR_MULT": 3.5}),

        # Chandelier lookback sweeps
        ("chandelier_lb_8", {"param_overrides.CHANDELIER_LOOKBACK": 8}),
        ("chandelier_lb_20", {"param_overrides.CHANDELIER_LOOKBACK": 20}),

        # Profit-funded BE adjustments
        ("profit_be_0.5R", {"param_overrides.PROFIT_BE_R": 0.5}),
        ("profit_be_1.0R", {"param_overrides.PROFIT_BE_R": 1.0}),

        # Early BE
        ("early_be_enable", {"flags.early_be": True}),

        # Stale exit adjustments
        ("stale_bars_12", {"param_overrides.STALE_BARS_NORMAL": 12}),
        ("stale_bars_25", {"param_overrides.STALE_BARS_NORMAL": 25}),

        # Combinations
        ("tight_trail_high_partial", {
            "param_overrides.CHANDELIER_ATR_MULT": 1.5,
            "param_overrides.TP1_R": 1.5, "param_overrides.TP1_PARTIAL_PCT": 0.60,
        }),
        ("wide_trail_tp2", {
            "param_overrides.CHANDELIER_ATR_MULT": 3.0,
            "param_overrides.TP2_R": 3.5, "param_overrides.TP2_PARTIAL_PCT": 0.25,
        }),
        ("early_tp1_keep_runner", {
            "param_overrides.TP1_R": 0.8, "param_overrides.TP1_PARTIAL_PCT": 0.40,
            "param_overrides.CHANDELIER_ATR_MULT": 3.0,
        }),
    ]


def _phase_2_signal(prior: dict) -> list[tuple[str, dict]]:
    """Signal discrimination: recalibrate non-discriminatory quality/sizing."""
    return [
        # Displacement gate
        ("disable_disp_gate", {"flags.displacement_threshold": False}),
        ("disp_q_0.05", {"param_overrides.Q_DISP": 0.05}),
        ("disp_q_0.20", {"param_overrides.Q_DISP": 0.20}),

        # Score gate
        ("disable_score_gate", {"flags.score_threshold": False}),
        ("score_normal_3.0", {"param_overrides.SCORE_NORMAL": 3.0}),
        ("score_normal_0.5", {"param_overrides.SCORE_NORMAL": 0.5}),

        # Regime mult overrides
        ("regime_mult_uniform", {
            "param_overrides.regime_mult_aligned": 1.0,
            "param_overrides.regime_mult_neutral": 1.0,
            "param_overrides.regime_mult_range": 1.0,
            "param_overrides.regime_mult_caution": 1.0,
        }),
        ("regime_mult_range_boost", {"param_overrides.regime_mult_range": 1.0}),
        ("regime_mult_neutral_cut", {"param_overrides.regime_mult_neutral": 0.35}),

        # Box width filters (NEW)
        ("min_box_180", {"param_overrides.MIN_BOX_WIDTH": 180}),
        ("max_box_420", {"param_overrides.MAX_BOX_WIDTH": 420}),
        ("box_filter_both", {"param_overrides.MIN_BOX_WIDTH": 180, "param_overrides.MAX_BOX_WIDTH": 420}),

        # Chop/quality ablations
        ("disable_chop_halt", {"flags.chop_halt": False}),
        ("disable_chop_degraded", {"flags.chop_degraded": False}),
        ("disable_quality_reject", {"flags.breakout_quality_reject": False}),
        ("disable_friction_gate", {"flags.friction_gate": False}),

        # Combinations
        ("loosen_funnel", {"flags.displacement_threshold": False, "flags.score_threshold": False}),
        ("regime_data_driven", {
            "param_overrides.regime_mult_range": 1.0,
            "param_overrides.regime_mult_neutral": 0.40,
            "flags.chop_degraded": False,
        }),
    ]


def _phase_3_timing(prior: dict) -> list[tuple[str, dict]]:
    """Timing & clustering: eliminate burst drag, exploit session-direction asymmetry."""
    return [
        # Inter-trade cooldown (NEW)
        ("cooldown_30m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 30}),
        ("cooldown_60m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 60}),
        ("cooldown_120m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 120}),
        ("cooldown_240m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 240}),

        # Session-direction filter (NEW)
        ("block_eth_shorts", {"flags.block_eth_shorts": True}),
        ("eth_short_half_size", {"param_overrides.ETH_SHORT_SIZE_MULT": 0.50}),

        # Daily stop relaxation
        ("daily_stop_r_minus_3.5", {"param_overrides.DAILY_STOP_R": -3.5}),
        ("daily_stop_r_minus_5.0", {"param_overrides.DAILY_STOP_R": -5.0}),
        ("disable_daily_stop", {"flags.daily_stop": False}),

        # Loss streak parameterization
        ("loss_streak_2", {"param_overrides.LOSS_STREAK_THRESHOLD": 2}),
        ("loss_streak_skip_12bars", {"param_overrides.LOSS_STREAK_SKIP_BARS": 12}),

        # Re-enable entry types
        ("enable_entry_a", {"param_overrides.A_ENTRY_ENABLED": True}),

        # Hour block ablations
        ("unblock_09_et", {"flags.block_09_et": False}),
        ("unblock_06_et", {"flags.block_06_et": False}),

        # Combinations
        ("cooldown_60m_block_eth_shorts", {
            "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 60,
            "flags.block_eth_shorts": True,
        }),
        ("cooldown_60m_relax_daily", {
            "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 60,
            "param_overrides.DAILY_STOP_R": -5.0,
        }),
    ]


def _phase_4_finetune(prior: dict) -> list[tuple[str, dict]]:
    """Fine-tune: +/-10/20% adaptive adjustment of prior numeric mutations + cross-phase combos."""
    candidates: list[tuple[str, dict]] = []

    # Extract numeric mutations from prior phases for adaptive adjustment
    for key, value in prior.items():
        if not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue
        base_name = key.replace("param_overrides.", "").replace("flags.", "")
        for factor, label in [(0.80, "m20"), (0.90, "m10"), (1.10, "p10"), (1.20, "p20")]:
            adjusted = value * factor
            # Round integers
            if isinstance(value, int):
                adjusted = int(round(adjusted))
            candidates.append((f"finetune_{base_name}_{label}", {key: adjusted}))

    # Cross-phase interactions: combine strongest exit + signal + timing mutations
    # These are generated dynamically from accepted mutations
    exit_muts = {k: v for k, v in prior.items() if any(
        x in k for x in ['TP1', 'TP2', 'CHANDELIER', 'STALE', 'PROFIT_BE']
    )}
    signal_muts = {k: v for k, v in prior.items() if any(
        x in k for x in ['displacement', 'score', 'regime_mult', 'BOX_WIDTH', 'chop']
    )}
    timing_muts = {k: v for k, v in prior.items() if any(
        x in k for x in ['INTER_TRADE', 'block_eth', 'DAILY_STOP', 'LOSS_STREAK']
    )}

    if exit_muts and signal_muts:
        candidates.append(("cross_exit_signal", {**exit_muts, **signal_muts}))
    if exit_muts and timing_muts:
        candidates.append(("cross_exit_timing", {**exit_muts, **timing_muts}))
    if signal_muts and timing_muts:
        candidates.append(("cross_signal_timing", {**signal_muts, **timing_muts}))
    if exit_muts and signal_muts and timing_muts:
        candidates.append(("cross_all_three", {**exit_muts, **signal_muts, **timing_muts}))

    # If no prior numeric mutations, provide fallback experiments
    if not candidates:
        candidates.extend([
            ("fallback_conservative", {
                "param_overrides.BASE_RISK_PCT": 0.006,
                "param_overrides.DAILY_STOP_R": -2.0,
            }),
            ("fallback_aggressive", {
                "param_overrides.BASE_RISK_PCT": 0.010,
                "param_overrides.regime_mult_range": 0.80,
            }),
        ])

    return candidates
