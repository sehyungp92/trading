"""NQDTC targeted phased auto-optimization candidates.

The current baseline is the previous optimized config.  These candidates are
kept inside existing parity-safe controls: session blocks, score/box thresholds,
existing entry toggles, and existing exit-management parameters.  No backtest-
only structural entry logic is introduced here.
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

    if suggested_experiments:
        seen = set()
        for name, muts in suggested_experiments:
            if name in seen:
                continue
            seen.add(name)
            candidates.append((name, muts))

    if phase == 1:
        candidates.extend(_phase_1_session_harvest(prior))
    elif phase == 2:
        candidates.extend(_phase_2_robust_protection(prior))
    elif phase == 3:
        candidates.extend(_phase_3_selective_recovery(prior))
    elif phase == 4:
        candidates.extend(_phase_4_interactions(prior))

    seen_names: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, muts in candidates:
        if name in seen_names:
            continue
        seen_names.add(name)
        unique.append((name, muts))
    return unique


def _phase_1_session_harvest(prior: dict) -> list[tuple[str, dict]]:
    """Recover session-specific Range trades without reopening weak regimes."""
    return [
        ("allow_05_et", {"flags.block_05_et": False}),
        ("allow_05_score_1.75", {
            "flags.block_05_et": False,
            "param_overrides.SCORE_NORMAL": 1.75,
        }),
        ("allow_05_score_2.0", {
            "flags.block_05_et": False,
            "param_overrides.SCORE_NORMAL": 2.0,
        }),
        ("allow_05_rvol_1.75", {
            "flags.block_05_et": False,
            "param_overrides.RVOL_SCORE_THRESH": 1.75,
        }),
        ("allow_05_max_box_350", {
            "flags.block_05_et": False,
            "param_overrides.MAX_BOX_WIDTH": 350,
        }),
        ("allow_05_max_box_450", {
            "flags.block_05_et": False,
            "param_overrides.MAX_BOX_WIDTH": 450,
        }),
        ("allow_05_max_stop_175", {
            "flags.block_05_et": False,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175,
        }),
        ("allow_05_max_stop_225", {
            "flags.block_05_et": False,
            "param_overrides.MAX_STOP_WIDTH_PTS": 225,
        }),
        ("allow_05_loss_streak_3", {
            "flags.block_05_et": False,
            "param_overrides.LOSS_STREAK_THRESHOLD": 3,
        }),
        ("allow_05_tp1_1.35", {
            "flags.block_05_et": False,
            "param_overrides.TP1_R": 1.35,
        }),
        ("allow_05_tp1_1.45", {
            "flags.block_05_et": False,
            "param_overrides.TP1_R": 1.45,
        }),
        ("allow_09_et", {"flags.block_09_et": False}),
        ("allow_05_09_et", {
            "flags.block_05_et": False,
            "flags.block_09_et": False,
        }),
        ("allow_05_09_score_1.75", {
            "flags.block_05_et": False,
            "flags.block_09_et": False,
            "param_overrides.SCORE_NORMAL": 1.75,
        }),
        ("allow_05_09_rvol_1.75", {
            "flags.block_05_et": False,
            "flags.block_09_et": False,
            "param_overrides.RVOL_SCORE_THRESH": 1.75,
        }),
        ("allow_05_cooldown_15", {
            "flags.block_05_et": False,
            "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 15,
        }),
    ]


def _phase_2_robust_protection(prior: dict) -> list[tuple[str, dict]]:
    """Improve return robustness and protect quality after session harvest."""
    return [
        ("tp1_1.35", {"param_overrides.TP1_R": 1.35}),
        ("tp1_1.45", {"param_overrides.TP1_R": 1.45}),
        ("tp1_1.50", {"param_overrides.TP1_R": 1.50}),
        ("tp1_partial_0.45", {"param_overrides.TP1_PARTIAL_PCT": 0.45}),
        ("tp1_partial_0.55", {"param_overrides.TP1_PARTIAL_PCT": 0.55}),
        ("tp1_1.45_p45", {
            "param_overrides.TP1_R": 1.45,
            "param_overrides.TP1_PARTIAL_PCT": 0.45,
        }),
        ("tp1_1.35_p55", {
            "param_overrides.TP1_R": 1.35,
            "param_overrides.TP1_PARTIAL_PCT": 0.55,
        }),
        ("ratchet_lock_0.45", {"param_overrides.RATCHET_LOCK_PCT": 0.45}),
        ("ratchet_lock_0.55", {"param_overrides.RATCHET_LOCK_PCT": 0.55}),
        ("ratchet_threshold_0.75", {"param_overrides.RATCHET_THRESHOLD_R": 0.75}),
        ("ratchet_0.45_at_0.75", {
            "param_overrides.RATCHET_LOCK_PCT": 0.45,
            "param_overrides.RATCHET_THRESHOLD_R": 0.75,
        }),
        ("chandelier_tighter_mid", {
            "param_overrides.CHANDELIER_TIER1_MULT": 1.8,
            "param_overrides.CHANDELIER_TIER2_MULT": 1.3,
            "param_overrides.CHANDELIER_TIER3_MULT": 1.0,
            "param_overrides.CHANDELIER_TIER4_MULT": 0.8,
        }),
        ("chandelier_grace_1", {"param_overrides.CHANDELIER_GRACE_BARS_30M": 1}),
        ("post_tp1_decay_0.05", {
            "param_overrides.CHANDELIER_POST_TP1_MULT_DECAY": 0.05,
            "param_overrides.CHANDELIER_POST_TP1_FLOOR_MULT": 1.2,
        }),
        ("max_stop_width_175", {"param_overrides.MAX_STOP_WIDTH_PTS": 175}),
        ("max_stop_width_225", {"param_overrides.MAX_STOP_WIDTH_PTS": 225}),
    ]


def _phase_3_selective_recovery(prior: dict) -> list[tuple[str, dict]]:
    """Test limited recovery levers that previously looked plausible."""
    return [
        ("rvol_1.35", {"param_overrides.RVOL_SCORE_THRESH": 1.35}),
        ("rvol_1.75", {"param_overrides.RVOL_SCORE_THRESH": 1.75}),
        ("score_1.25_rvol_1.35", {
            "param_overrides.SCORE_NORMAL": 1.25,
            "param_overrides.RVOL_SCORE_THRESH": 1.35,
        }),
        ("score_1.75_rvol_1.75", {
            "param_overrides.SCORE_NORMAL": 1.75,
            "param_overrides.RVOL_SCORE_THRESH": 1.75,
        }),
        ("min_box_125", {"param_overrides.MIN_BOX_WIDTH": 125}),
        ("min_box_175", {"param_overrides.MIN_BOX_WIDTH": 175}),
        ("cooldown_15", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 15}),
        ("cooldown_45", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 45}),
        ("loss_streak_3", {"param_overrides.LOSS_STREAK_THRESHOLD": 3}),
        ("loss_streak_skip_3", {"param_overrides.LOSS_STREAK_SKIP_BARS": 3}),
        ("allow_06_et_guarded", {
            "flags.block_06_et": False,
            "param_overrides.SCORE_NORMAL": 1.75,
        }),
        ("allow_12_et_guarded", {
            "flags.block_12_et": False,
            "param_overrides.SCORE_NORMAL": 1.75,
        }),
        ("aligned_nonrange_2.5", {
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 2.5,
        }),
        ("aligned_nonrange_3.0", {
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
        }),
        ("aligned_nonrange_3.0_score_1.75", {
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
            "param_overrides.SCORE_NORMAL": 1.75,
        }),
        ("aligned_nonrange_3.0_rvol_1.75", {
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
            "param_overrides.RVOL_SCORE_THRESH": 1.75,
        }),
    ]


def _phase_4_interactions(prior: dict) -> list[tuple[str, dict]]:
    """Fine-tune accepted session/quality mutations and curated interactions."""
    candidates: list[tuple[str, dict]] = []

    ranges = {
        "param_overrides.SCORE_NORMAL": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.RVOL_SCORE_THRESH": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.MIN_BOX_WIDTH": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": [(0.50, "m50"), (1.50, "p50")],
        "param_overrides.TP1_R": [(0.95, "m05"), (1.05, "p05")],
        "param_overrides.TP1_PARTIAL_PCT": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.LOSS_STREAK_THRESHOLD": [(1.50, "p50")],
        "param_overrides.RATCHET_LOCK_PCT": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.RATCHET_THRESHOLD_R": [(0.90, "m10"), (1.10, "p10")],
    }
    for key, adjustments in ranges.items():
        value = prior.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        base_name = key.replace("param_overrides.", "")
        for factor, label in adjustments:
            adjusted = value * factor
            if isinstance(value, int):
                adjusted = int(round(adjusted))
            candidates.append((f"finetune_{base_name}_{label}", {key: adjusted}))

    candidates.extend([
        ("combo_05_09", {
            "flags.block_05_et": False,
            "flags.block_09_et": False,
        }),
        ("combo_05_rvol_1.35", {
            "flags.block_05_et": False,
            "param_overrides.RVOL_SCORE_THRESH": 1.35,
        }),
        ("combo_05_score_1.75", {
            "flags.block_05_et": False,
            "param_overrides.SCORE_NORMAL": 1.75,
        }),
        ("combo_05_ratchet", {
            "flags.block_05_et": False,
            "param_overrides.RATCHET_LOCK_PCT": 0.45,
            "param_overrides.RATCHET_THRESHOLD_R": 0.75,
        }),
        ("combo_05_tp1_1.45", {
            "flags.block_05_et": False,
            "param_overrides.TP1_R": 1.45,
        }),
        ("combo_05_max_stop_175", {
            "flags.block_05_et": False,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175,
        }),
        ("combo_rvol_1.35_09", {
            "flags.block_09_et": False,
            "param_overrides.RVOL_SCORE_THRESH": 1.35,
        }),
    ])

    return candidates
