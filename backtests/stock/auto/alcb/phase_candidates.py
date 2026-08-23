"""ALCB round-2 experiments for broad, parity-safe residual alpha extraction.

The round is seeded from the recovered round-1 optimized configuration by the
CLI.  Candidates use only completed-bar inputs and controls implemented on both
the live and replay paths.  Symbol, weekday, and narrow sector fits are excluded.
"""
from __future__ import annotations

from typing import Any


BASE_MUTATIONS: dict[str, Any] = {}


def sanitize_round2_seed(mutations: dict[str, Any]) -> dict[str, Any]:
    """Preserve the previous optimized configuration exactly as the seed."""
    return dict(mutations)


SMALL_SAMPLE_OVERFIT_MUTATION_KEYS: frozenset[str] = frozenset({
    "param_overrides.sector_entry_blocklist",
    "param_overrides.sector_entry_size_mults",
    "param_overrides.monday_sizing_mult",
    "param_overrides.tuesday_sizing_mult",
    "param_overrides.wednesday_sizing_mult",
    "param_overrides.thursday_sizing_mult",
    "param_overrides.friday_sizing_mult",
})


def is_small_sample_overfit_candidate(name: str, mutations: dict[str, Any]) -> bool:
    """Reject candidate families that encode narrow calendar/sector cohorts."""
    del name
    return any(key in SMALL_SAMPLE_OVERFIT_MUTATION_KEYS for key in mutations)


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Execution-cost discrimination and seven-factor score allocation",
        ["expected_total_r", "net_profit", "trades_per_month", "expectancy", "profit_factor", "max_drawdown_pct", "sharpe"],
    ),
    2: (
        "Broad signal-gate alpha recovery",
        ["expected_total_r", "trades_per_month", "expectancy", "profit_factor", "signal_quality", "rvol_selectivity", "max_drawdown_pct"],
    ),
    3: (
        "Causal entry timing and geometry",
        ["expected_total_r", "trades_per_month", "expectancy", "profit_factor", "timing_quality", "late_entry_quality", "max_drawdown_pct"],
    ),
    4: (
        "Route and regime risk allocation",
        ["expected_total_r", "net_profit", "expectancy", "profit_factor", "sizing_alignment", "trades_per_month", "max_drawdown_pct"],
    ),
    5: (
        "Loser containment without clipping runners",
        ["expected_total_r", "net_profit", "expectancy", "profit_factor", "profit_protection", "mfe_capture_efficiency", "max_drawdown_pct"],
    ),
    6: (
        "Aggressive-leaning portfolio and exit synthesis",
        ["expected_total_r", "net_profit", "trades_per_month", "expectancy", "profit_factor", "max_drawdown_pct", "sharpe"],
    ),
}


PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        ("r2_friction_gate_010", {
            "param_overrides.use_momentum_friction_gate": True,
            "param_overrides.max_friction_to_risk": 0.10,
        }),
        ("r2_friction_gate_015", {
            "param_overrides.use_momentum_friction_gate": True,
            "param_overrides.max_friction_to_risk": 0.15,
        }),
        ("r2_friction_gate_020", {
            "param_overrides.use_momentum_friction_gate": True,
            "param_overrides.max_friction_to_risk": 0.20,
        }),
        ("r2_remove_or5_no_volume_penalty", {
            "param_overrides.entry_detail_size_mults": {},
        }),
        ("r2_score5_global_defensive", {
            "param_overrides.momentum_size_mult_score_5": 0.90,
        }),
    ],
    2: [
        ("r2_combined_rvol_220", {
            "param_overrides.combined_breakout_min_rvol": 2.20,
        }),
        ("r2_combined_rvol_200", {
            "param_overrides.combined_breakout_min_rvol": 2.00,
        }),
        ("r2_global_rvol_140", {
            "param_overrides.rvol_threshold": 1.40,
        }),
        ("r2_global_rvol_135", {
            "param_overrides.rvol_threshold": 1.35,
        }),
    ],
    3: [
        ("r2_orb_range_cap_150", {
            "param_overrides.orb_entry_range_cap_r": 1.50,
        }),
        ("r2_opening_range_8", {
            "param_overrides.opening_range_bars": 8,
        }),
        ("r2_opening_range_10", {
            "param_overrides.opening_range_bars": 10,
        }),
        ("r2_selective_1300_extension", {
            "param_overrides.entry_window_end": "13:00:00",
            "param_overrides.late_entry_cutoff": "11:30:00",
            "param_overrides.late_entry_score_min": 6,
            "param_overrides.late_entry_size_mult": 0.75,
        }),
    ],
    4: [
        ("r2_regime_b_085", {
            "param_overrides.regime_mult_b": 0.85,
        }),
        ("r2_regime_b_100", {
            "param_overrides.regime_mult_b": 1.00,
        }),
        ("r2_pdh_size_090", {
            "param_overrides.pdh_size_mult": 0.90,
        }),
        ("r2_pdh_size_100", {
            "param_overrides.pdh_size_mult": 1.00,
        }),
    ],
    5: [
        ("r2_flow_require_below_entry", {
            "param_overrides.flow_reversal_require_below_entry": True,
        }),
        ("r2_flow_hold10_grace030", {
            "param_overrides.flow_reversal_min_hold_bars": 10,
            "param_overrides.fr_mfe_grace_r": 0.30,
            "param_overrides.fr_cpr_threshold": 0.25,
        }),
        ("r2_flow_hold12_grace035", {
            "param_overrides.flow_reversal_min_hold_bars": 12,
            "param_overrides.fr_mfe_grace_r": 0.35,
            "param_overrides.fr_cpr_threshold": 0.20,
        }),
        ("r2_disable_mfe_conviction_exit", {
            "param_overrides.mfe_conviction_check_bars": 0,
        }),
    ],
    6: [
        ("r2_base_risk_00750", {
            "param_overrides.base_risk_fraction": 0.00750,
        }),
        ("r2_base_risk_00800", {
            "param_overrides.base_risk_fraction": 0.00800,
        }),
        ("r2_capacity_7_heat_450", {
            "param_overrides.max_positions": 7,
            "param_overrides.heat_cap_r": 4.50,
        }),
        ("r2_adaptive_trail_distance_008", {
            "param_overrides.adaptive_trail_late_activate_r": 0.24,
            "param_overrides.adaptive_trail_late_distance_r": 0.08,
        }),
    ],
}


def get_phase_candidates(
    phase: int,
    *,
    experiment_filter: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    experiments = [
        (name, mutations)
        for name, mutations in PHASE_CANDIDATES.get(phase, [])
        if not is_small_sample_overfit_candidate(name, mutations)
    ]
    if experiment_filter:
        experiments = [
            (name, mutations)
            for name, mutations in experiments
            if name in experiment_filter
        ]
    return experiments
