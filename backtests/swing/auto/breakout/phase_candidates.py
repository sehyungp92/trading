from __future__ import annotations

from dataclasses import fields

from backtests.swing.config_breakout import BreakoutAblationFlags

BASE_MUTATIONS = {
    "flags.disable_regime_chop_block": True,
    "flags.disable_score_threshold": True,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("ABLATION", ["profit_factor", "total_trades", "max_drawdown_pct"]),
    2: ("ENTRY + THRESHOLDS", ["profit_factor", "net_profit", "max_drawdown_pct"]),
    3: ("EXIT + REENTRY + SIZING", ["net_profit", "calmar", "profit_factor"]),
}

_PHASE_2_SINGLE_SWEEPS: dict[str, list[float | int]] = {
    "param_overrides.ENTRYB_RVOLH_MIN": [0.50, 0.70, 0.90],
    "param_overrides.ENTRYB_NEUTRAL_QUALITY_MIN": [0.55, 0.65, 0.75],
    "param_overrides.DIRTY_BOX_SHIFT_ATR_MULT": [0.20, 0.30, 0.40],
    "param_overrides.DIRTY_DURATION_L_FRAC": [0.25, 0.40, 0.60],
    "param_overrides.HARD_BLOCK_SLOPE_MULT": [0.08, 0.12, 0.16],
}

_PHASE_3_SINGLE_SWEEPS: dict[str, list[float | int]] = {
    "param_overrides.ADD_RISK_MULT": [0.35, 0.50, 0.65],
    "param_overrides.BE_BUFFER_ATR_MULT": [0.05, 0.10, 0.20],
    "param_overrides.TRAIL_4H_ATR_MULT": [0.30, 0.40, 0.60],
    "param_overrides.MAX_REENTRY_PER_BOX_VERSION": [1, 2, 3],
    "param_overrides.REENTRY_COOLDOWN_DAYS": [0, 1, 2],
    "param_overrides.REENTRY_MIN_REALIZED_R": [-2.0, -1.0, -0.5],
}


def get_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates()
    else:
        candidates = []

    if suggested_experiments:
        existing = {name for name, _ in candidates}
        for name, mutations in suggested_experiments:
            if name not in existing:
                candidates.append((name, dict(mutations)))
                existing.add(name)
    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    for flag_field in fields(BreakoutAblationFlags):
        flag_name = flag_field.name
        key = f"flags.{flag_name}"
        baseline_value = bool(BASE_MUTATIONS.get(key, flag_field.default))
        candidate_value = not baseline_value
        action = "reenable" if baseline_value else "disable"
        short_name = flag_name.removeprefix("disable_")
        candidates.append((f"abl_breakout_{action}_{short_name}", {key: candidate_value}))
    return candidates


def _phase_2_candidates() -> list[tuple[str, dict]]:
    candidates = _single_value_candidates("thr_breakout", _PHASE_2_SINGLE_SWEEPS)

    for key in (
        "param_overrides.SCORE_THRESHOLD_NORMAL",
        "param_overrides.SCORE_THRESHOLD_DEGRADED",
        "param_overrides.SCORE_THRESHOLD_RANGE",
    ):
        suffix = key.rsplit(".", 1)[-1].lower()
        for value in (1, 2, 3):
            candidates.append((
                f"thr_breakout_{suffix}_{value}",
                {
                    "flags.disable_score_threshold": False,
                    key: value,
                },
            ))

    for value in (2, 3, 4):
        candidates.append((
            f"thr_breakout_regime_chop_score_override_{value}",
            {
                "flags.disable_regime_chop_block": False,
                "param_overrides.REGIME_CHOP_SCORE_OVERRIDE": value,
            },
        ))

    candidates.extend([
        (
            "thr_breakout_entry_b_relaxed",
            {
                "param_overrides.ENTRYB_RVOLH_MIN": 0.50,
                "param_overrides.ENTRYB_NEUTRAL_QUALITY_MIN": 0.55,
            },
        ),
        (
            "thr_breakout_entry_b_tight",
            {
                "param_overrides.ENTRYB_RVOLH_MIN": 0.90,
                "param_overrides.ENTRYB_NEUTRAL_QUALITY_MIN": 0.75,
            },
        ),
        (
            "thr_breakout_dirty_looser",
            {
                "param_overrides.DIRTY_BOX_SHIFT_ATR_MULT": 0.20,
                "param_overrides.DIRTY_DURATION_L_FRAC": 0.25,
            },
        ),
    ])
    return candidates


def _phase_3_candidates() -> list[tuple[str, dict]]:
    candidates = _single_value_candidates("exit_breakout", _PHASE_3_SINGLE_SWEEPS)
    candidates.extend([
        (
            "exit_breakout_fast_trail",
            {
                "param_overrides.BE_BUFFER_ATR_MULT": 0.05,
                "param_overrides.TRAIL_4H_ATR_MULT": 0.30,
            },
        ),
        (
            "exit_breakout_wide_trail",
            {
                "param_overrides.BE_BUFFER_ATR_MULT": 0.20,
                "param_overrides.TRAIL_4H_ATR_MULT": 0.60,
            },
        ),
        (
            "exit_breakout_reentry_looser",
            {
                "param_overrides.MAX_REENTRY_PER_BOX_VERSION": 3,
                "param_overrides.REENTRY_COOLDOWN_DAYS": 0,
                "param_overrides.REENTRY_MIN_REALIZED_R": -2.0,
            },
        ),
        (
            "exit_breakout_reentry_tighter",
            {
                "param_overrides.MAX_REENTRY_PER_BOX_VERSION": 1,
                "param_overrides.REENTRY_COOLDOWN_DAYS": 2,
                "param_overrides.REENTRY_MIN_REALIZED_R": -0.5,
            },
        ),
        (
            "exit_breakout_add_risk_high",
            {
                "param_overrides.ADD_RISK_MULT": 0.65,
                "param_overrides.TRAIL_4H_ATR_MULT": 0.50,
            },
        ),
    ])
    return candidates


def _single_value_candidates(
    prefix: str,
    sweeps: dict[str, list[float | int]],
) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    for key, values in sweeps.items():
        short_name = key.rsplit(".", 1)[-1].lower()
        for value in values:
            value_name = str(value).replace("-", "neg").replace(".", "p")
            candidates.append((f"{prefix}_{short_name}_{value_name}", {key: value}))
    return candidates
