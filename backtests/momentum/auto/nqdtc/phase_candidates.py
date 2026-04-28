"""NQDTC post-audit phased auto-optimization experiment candidates.

Phase 1: Regime filtering (~19 experiments) -- eliminate losing regimes
Phase 2: Signal quality (~17 experiments) -- tighten entry discrimination
Phase 3: Timing & exit (~17 experiments) -- address burst clustering + exit mgmt
Phase 4: Fine-tune & interaction (~dynamic experiments)
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
        candidates.extend(_phase_1_regime(prior))
    elif phase == 2:
        candidates.extend(_phase_2_signal(prior))
    elif phase == 3:
        candidates.extend(_phase_3_timing_exit(prior))
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


def _phase_1_regime(prior: dict) -> list[tuple[str, dict]]:
    """Regime filtering: eliminate losing regimes (Neutral PF=0.53, Aligned avgR=-0.15)."""
    return [
        # Single regime blocks
        ("block_neutral", {"param_overrides.BLOCK_NEUTRAL_REGIME": True}),
        ("block_aligned", {"param_overrides.BLOCK_ALIGNED_REGIME": True}),
        ("block_caution", {"param_overrides.BLOCK_CAUTION_REGIME": True}),

        # Combined blocks
        ("block_neutral_aligned", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": True,
            "param_overrides.BLOCK_ALIGNED_REGIME": True,
        }),
        ("block_neutral_caution", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": True,
            "param_overrides.BLOCK_CAUTION_REGIME": True,
        }),
        ("range_only", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": True,
            "param_overrides.BLOCK_ALIGNED_REGIME": True,
            "param_overrides.BLOCK_CAUTION_REGIME": True,
        }),

        # Sizing reduction instead of blocking
        ("regime_mult_neutral_0.35", {"param_overrides.regime_mult_neutral": 0.35}),
        ("regime_mult_neutral_0.20", {"param_overrides.regime_mult_neutral": 0.20}),
        ("regime_mult_aligned_0.50", {"param_overrides.regime_mult_aligned": 0.50}),
        ("regime_mult_aligned_0.35", {"param_overrides.regime_mult_aligned": 0.35}),

        # Score threshold multiplier for non-Range
        ("score_non_range_2x", {"param_overrides.SCORE_NON_RANGE_MULT": 2.0}),
        ("score_non_range_3x", {"param_overrides.SCORE_NON_RANGE_MULT": 3.0}),

        # Combo: block + reduce
        ("block_neutral_reduce_aligned", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": True,
            "param_overrides.regime_mult_aligned": 0.35,
        }),
        ("block_neutral_boost_range", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": True,
            "param_overrides.regime_mult_range": 1.0,
        }),
        ("block_neutral_aligned_boost_range", {
            "param_overrides.BLOCK_NEUTRAL_REGIME": True,
            "param_overrides.BLOCK_ALIGNED_REGIME": True,
            "param_overrides.regime_mult_range": 1.0,
        }),

        # Ablate existing subtype x regime blocks (both default True in config)
        ("disable_std_neutral_low_disp", {"param_overrides.BLOCK_STD_NEUTRAL_LOW_DISP": False}),
        ("disable_cont_aligned", {"param_overrides.BLOCK_CONT_ALIGNED": False}),

        # Chop mode interaction
        ("disable_chop_degraded", {"flags.chop_degraded": False}),
        ("disable_chop_halt", {"flags.chop_halt": False}),
    ]


def _phase_2_signal(prior: dict) -> list[tuple[str, dict]]:
    """Signal quality: tighten entry discrimination after regime filtering."""
    return [
        # Score gate sweeps
        ("score_normal_1.5", {"param_overrides.SCORE_NORMAL": 1.5}),
        ("score_normal_2.0", {"param_overrides.SCORE_NORMAL": 2.0}),
        ("score_normal_2.5", {"param_overrides.SCORE_NORMAL": 2.5}),

        # Displacement gate
        ("disp_q_0.15", {"param_overrides.Q_DISP": 0.15}),
        ("disp_q_0.20", {"param_overrides.Q_DISP": 0.20}),
        ("disable_disp_gate", {"flags.displacement_threshold": False}),

        # Box width filters
        ("min_box_150", {"param_overrides.MIN_BOX_WIDTH": 150}),
        ("min_box_200", {"param_overrides.MIN_BOX_WIDTH": 200}),
        ("min_box_250", {"param_overrides.MIN_BOX_WIDTH": 250}),

        # Quality ablations
        ("disable_quality_reject", {"flags.breakout_quality_reject": False}),
        ("disable_entry_b", {"flags.entry_b_sweep": False}),
        ("disable_entry_a_latch", {"flags.entry_a_latch": False}),
        ("disable_friction_gate", {"flags.friction_gate": False}),

        # RVOL threshold
        ("rvol_thresh_1.5", {"param_overrides.RVOL_SCORE_THRESH": 1.5}),

        # Combos
        ("score_2.0_disp_0.15", {
            "param_overrides.SCORE_NORMAL": 2.0,
            "param_overrides.Q_DISP": 0.15,
        }),
        ("score_1.5_min_box_200", {
            "param_overrides.SCORE_NORMAL": 1.5,
            "param_overrides.MIN_BOX_WIDTH": 200,
        }),
        ("tighten_all_quality", {
            "param_overrides.SCORE_NORMAL": 2.0,
            "param_overrides.Q_DISP": 0.15,
            "param_overrides.MIN_BOX_WIDTH": 200,
        }),
    ]


def _phase_3_timing_exit(prior: dict) -> list[tuple[str, dict]]:
    """Timing & exit: address burst clustering and exit management."""
    return [
        # Inter-trade cooldown
        ("cooldown_30m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 30}),
        ("cooldown_60m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 60}),
        ("cooldown_120m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 120}),

        # Session-direction
        ("block_eth_shorts", {"flags.block_eth_shorts": True}),
        ("eth_short_half_size", {"param_overrides.ETH_SHORT_SIZE_MULT": 0.50}),

        # TP1 sweeps
        ("tp1_r_0.8", {"param_overrides.TP1_R": 0.8}),
        ("tp1_r_1.2", {"param_overrides.TP1_R": 1.2}),
        ("tp1_r_1.5", {"param_overrides.TP1_R": 1.5}),
        ("tp1_partial_0.33", {"param_overrides.TP1_PARTIAL_PCT": 0.33}),
        ("tp1_partial_0.50", {"param_overrides.TP1_PARTIAL_PCT": 0.50}),

        # Chandelier trailing
        ("chandelier_mult_1.0", {"param_overrides.CHANDELIER_ATR_MULT": 1.0}),
        ("chandelier_mult_1.5", {"param_overrides.CHANDELIER_ATR_MULT": 1.5}),
        ("chandelier_mult_2.0", {"param_overrides.CHANDELIER_ATR_MULT": 2.0}),

        # Loss streak / daily stop
        ("loss_streak_2", {"param_overrides.LOSS_STREAK_THRESHOLD": 2}),
        ("daily_stop_minus_1.5R", {"param_overrides.DAILY_STOP_R": -1.5}),

        # TP2
        ("tp2_at_3.5R", {"param_overrides.TP2_R": 3.5, "param_overrides.TP2_PARTIAL_PCT": 0.25}),
        ("tp2_at_5.0R", {"param_overrides.TP2_R": 5.0, "param_overrides.TP2_PARTIAL_PCT": 0.25}),
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

    # Cross-phase interactions: combine strongest regime + signal + timing mutations
    regime_muts = {k: v for k, v in prior.items() if any(
        x in k for x in ['BLOCK_NEUTRAL', 'BLOCK_ALIGNED', 'BLOCK_CAUTION',
                          'SCORE_NON_RANGE', 'regime_mult']
    )}
    signal_muts = {k: v for k, v in prior.items() if any(
        x in k for x in ['SCORE_NORMAL', 'Q_DISP', 'MIN_BOX_WIDTH', 'quality_reject',
                          'displacement', 'RVOL_SCORE']
    )}
    timing_muts = {k: v for k, v in prior.items() if any(
        x in k for x in ['INTER_TRADE', 'block_eth', 'ETH_SHORT', 'TP1', 'TP2',
                          'CHANDELIER', 'DAILY_STOP', 'LOSS_STREAK']
    )}

    if regime_muts and signal_muts:
        candidates.append(("cross_regime_signal", {**regime_muts, **signal_muts}))
    if regime_muts and timing_muts:
        candidates.append(("cross_regime_timing", {**regime_muts, **timing_muts}))
    if signal_muts and timing_muts:
        candidates.append(("cross_signal_timing", {**signal_muts, **timing_muts}))
    if regime_muts and signal_muts and timing_muts:
        candidates.append(("cross_all_three", {**regime_muts, **signal_muts, **timing_muts}))

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
