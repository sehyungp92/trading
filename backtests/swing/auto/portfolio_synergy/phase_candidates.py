from __future__ import annotations

from copy import deepcopy
from typing import Any

INITIAL_EQUITY = 25_000.0
ROUND_NAME = "round_2_dynamic_risk_blocked_alpha"
RISK_STANCE = "aggressive_controlled"

STRATEGY_ORDER = (
    "ATRSS",
    "AKC_HELIX",
    "BRS_R9",
    "SWING_BREAKOUT_V3",
    "OVERLAY",
)

SCORE_WEIGHTS: dict[str, float] = {
    "alpha_return": 0.26,
    "trade_frequency": 0.22,
    "drawdown_control": 0.20,
    "profit_factor_quality": 0.12,
    "strategy_balance": 0.08,
    "idle_cash_efficiency": 0.07,
    "robustness": 0.05,
}

ROUND_TARGETS: dict[str, float] = {
    "initial_equity": INITIAL_EQUITY,
    "min_active_trades_per_month": 11.0,
    "min_total_r_per_month": 6.5,
    "min_profit_factor": 2.25,
    "target_max_drawdown_pct": 0.10,
    "hard_max_drawdown_pct": 0.12,
    "min_active_strategies": 5.0,
    "max_single_strategy_risk_share": 0.45,
    "max_overlay_equity_pct": 0.70,
}

SEED_PORTFOLIO_CONFIG: dict[str, Any] = {
    "initial_equity": INITIAL_EQUITY,
    "risk_stance": RISK_STANCE,
    "portfolio_rules": {
        "heat_cap_R": 3.75,
        "dynamic_risk_enabled": False,
        "max_total_active_positions": 5,
        "max_total_positions_including_overlay": 7,
        "portfolio_daily_stop_R": 3.25,
        "portfolio_weekly_stop_R": 7.0,
        "priority_headroom_R": 0.75,
        "downturn_reserve_R": 1.15,
        "max_symbol_directional_heat_R": 2.25,
        "max_long_heat_R": 3.00,
        "max_short_heat_R": 3.25,
        "max_single_strategy_risk_share": 0.45,
        "drawdown_tiers": (
            (0.06, 1.00),
            (0.09, 0.70),
            (0.12, 0.40),
            (0.15, 0.00),
        ),
    },
    "strategy_allocations": {
        "ATRSS": {
            "unit_risk_pct": 0.016,
            "max_heat_R": 1.60,
            "daily_stop_R": 2.25,
            "max_working_orders": 4,
            "priority": 0,
            "role": "long pullback frequency core",
        },
        "AKC_HELIX": {
            "unit_risk_pct": 0.009,
            "max_heat_R": 1.35,
            "daily_stop_R": 2.50,
            "max_working_orders": 4,
            "priority": 1,
            "role": "convex trend and add-on alpha engine",
        },
        "BRS_R9": {
            "unit_risk_pct": 0.007,
            "max_heat_R": 1.25,
            "daily_stop_R": 2.00,
            "max_working_orders": 3,
            "priority": 2,
            "role": "bear-regime and drawdown-offset sleeve",
        },
        "SWING_BREAKOUT_V3": {
            "unit_risk_pct": 0.005,
            "max_heat_R": 0.75,
            "daily_stop_R": 1.50,
            "max_working_orders": 2,
            "priority": 3,
            "role": "low-drawdown opportunistic breakout sleeve",
        },
        "OVERLAY": {
            "max_equity_pct": 0.70,
            "bear_mode_max_equity_pct": 0.35,
            "min_alloc_pct": 0.25,
            "max_alloc_pct": 1.00,
            "weights": {"QQQ": 0.45, "GLD": 0.55},
            "priority": 4,
            "role": "idle-cash deployment after active heat reservation",
        },
    },
    "cross_strategy_rules": {
        "priority_order": list(STRATEGY_ORDER),
        "all_five_required_for_baseline": True,
        "atrss_helix_alignment": {
            "enabled": True,
            "same_symbol_same_direction_boost": 1.15,
            "tighten_helix_to_be_on_atrss_entry": True,
            "max_combined_symbol_heat_R": 2.10,
        },
        "brs_downturn_reserve": {
            "enabled": True,
            "reserve_R": 1.15,
            "trigger": "BRS bear trend or bear strong regime",
            "overlay_bear_cap_pct": 0.35,
            "long_entry_size_mult": 0.60,
        },
        "breakout_opportunistic_slot": {
            "enabled": True,
            "reserved_heat_R": 0.35,
            "release_if_no_campaign_hours": 96,
        },
        "overlay_idle_cash_policy": {
            "rebalance_after_active_heat": True,
            "do_not_use_margin": True,
            "cut_when_active_heat_above_R": 2.75,
        },
    },
}

PHASE_FOCUS: dict[int, str] = {
    1: "Diagnostic ingestion, five-strategy baseline, and BRS integration",
    2: "Dynamic live-style unit-risk recalibration and frequency unlock",
    3: "Synergistic actual-risk allocation across sleeves and symbols",
    4: "Overlay idle-cash efficiency and bear-mode throttling",
    5: "Strategy-specific frequency unlocks with quality protection",
    6: "Blocked-candidate discrimination and cross-strategy routing",
    7: "Drawdown governors, final blend, and robustness gates",
}

PHASE_GATES: dict[int, dict[str, float]] = {
    1: {
        "min_active_strategies": 5.0,
        "min_active_trades_per_month": 8.5,
        "hard_max_drawdown_pct": 0.12,
    },
    2: {
        "min_active_trades_per_month": 11.0,
        "min_total_r_per_month": 6.0,
        "hard_max_drawdown_pct": 0.12,
    },
    3: {
        "min_total_r_per_month": 6.5,
        "max_single_strategy_risk_share": 0.45,
        "target_max_drawdown_pct": 0.10,
    },
    4: {
        "max_overlay_equity_pct": 0.70,
        "max_overlay_score_drag_pct": 0.03,
        "min_idle_cash_deployment_share": 0.35,
    },
    5: {
        "min_active_trades_per_month": 11.5,
        "min_profit_factor": 2.20,
        "max_strategy_quality_regression_pct": 0.08,
    },
    6: {
        "max_positive_alpha_block_rate": 0.18,
        "max_symbol_directional_heat_R": 2.25,
        "min_trade_capture_ratio": 0.82,
    },
    7: {
        "min_active_strategies": 5.0,
        "min_active_trades_per_month": 11.0,
        "min_total_r_per_month": 6.5,
        "min_profit_factor": 2.25,
        "hard_max_drawdown_pct": 0.12,
        "min_positive_years": 4.0,
    },
}


def get_phase_candidates(phase: int) -> list[dict[str, Any]]:
    candidates = _PHASE_CANDIDATES.get(phase)
    if candidates is None:
        raise ValueError(f"Unknown swing portfolio synergy phase: {phase}")
    return deepcopy(candidates)


def phase_summary() -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "focus": PHASE_FOCUS[phase],
            "gate": PHASE_GATES[phase],
            "candidate_count": len(_PHASE_CANDIDATES[phase]),
            "candidate_names": [candidate["name"] for candidate in _PHASE_CANDIDATES[phase]],
        }
        for phase in sorted(PHASE_FOCUS)
    ]


_PHASE_CANDIDATES: dict[int, list[dict[str, Any]]] = {
    1: [
        {
            "name": "seed_aggressive_controlled_25k",
            "mutations": SEED_PORTFOLIO_CONFIG,
            "rationale": "Start all four active swing engines plus overlay at $25k with controlled-aggressive heat.",
        },
        {
            "name": "brs_integrated_bear_sleeve",
            "mutations": {
                "strategy_allocations.BRS_R9.enabled": True,
                "cross_strategy_rules.brs_downturn_reserve.enabled": True,
            },
            "rationale": "BRS r10 has 81.2% bear alpha and 7.59 bear PF; it should be present from the baseline.",
        },
        {
            "name": "overlay_enabled_but_capped",
            "mutations": {
                "strategy_allocations.OVERLAY.max_equity_pct": 0.70,
                "cross_strategy_rules.overlay_idle_cash_policy.rebalance_after_active_heat": True,
            },
            "rationale": "Keep overlay as the fifth strategy while respecting older evidence that uncapped overlay drags score.",
        },
    ],
    2: [
        {
            "name": "dynamic_unit_risk_recalibration",
            "mutations": {"portfolio_rules.dynamic_risk_enabled": True},
            "rationale": "Match live-style NAV recalibration so growing equity does not create artificial late-period heat starvation.",
        },
        {
            "name": "dynamic_drawdown_guard_6_9_12_15",
            "mutations": {
                "portfolio_rules.dynamic_risk_enabled": True,
                "portfolio_rules.drawdown_tiers": (
                    (0.06, 1.00),
                    (0.09, 0.70),
                    (0.12, 0.40),
                    (0.15, 0.00),
                ),
            },
            "rationale": "Press when near highs, but throttle risk before the controlled-aggressive DD ceiling is threatened.",
        },
        {
            "name": "heat_cap_3_75_seed",
            "mutations": {"portfolio_rules.heat_cap_R": 3.75},
            "rationale": "Above live-parity 3.0R, but below an unconstrained sum of strategy demand.",
        },
        {
            "name": "dynamic_heat_cap_4_00_probe",
            "mutations": {
                "portfolio_rules.dynamic_risk_enabled": True,
                "portfolio_rules.heat_cap_R": 4.00,
            },
            "rationale": "Modest controlled-aggressive heat expansion only under dynamic drawdown throttling.",
        },
        {
            "name": "heat_cap_4_25_probe",
            "mutations": {"portfolio_rules.heat_cap_R": 4.25},
            "rationale": "Probe upper alpha extraction only if DD and alpha-blocking stay inside gates.",
        },
        {
            "name": "positions_6_probe",
            "mutations": {"portfolio_rules.max_total_active_positions": 6},
            "rationale": "Allow ATRSS and Helix to overlap while leaving room for BRS/Breakout opportunism.",
        },
        {
            "name": "short_room_for_brs_helix",
            "mutations": {
                "portfolio_rules.max_short_heat_R": 3.50,
                "portfolio_rules.max_long_heat_R": 2.85,
            },
            "rationale": "Lean into downturn diversification without letting long QQQ crowding dominate.",
        },
        {
            "name": "dynamic_strategy_heat_unlock",
            "mutations": {
                "portfolio_rules.dynamic_risk_enabled": True,
                "strategy_allocations.ATRSS.max_heat_R": 1.85,
                "strategy_allocations.AKC_HELIX.max_heat_R": 1.50,
                "strategy_allocations.BRS_R9.max_heat_R": 1.40,
            },
            "rationale": "Most blocks are own-sleeve heat ceilings; unlock them together so no single sleeve monopolizes routing.",
        },
    ],
    3: [
        {
            "name": "balanced_actual_risk_multipliers",
            "mutations": {
                "symbol_risk_multipliers": {
                    "ATRSS:QQQ": 0.90,
                    "ATRSS:GLD": 0.90,
                    "AKC_HELIX:QQQ": 1.05,
                    "AKC_HELIX:GLD": 1.05,
                    "BRS_R9:QQQ": 1.10,
                    "BRS_R9:GLD": 1.05,
                    "SWING_BREAKOUT_V3:QQQ": 1.15,
                    "SWING_BREAKOUT_V3:GLD": 1.10,
                }
            },
            "rationale": "Reduce ATRSS crowding slightly and reallocate actual risk toward Helix/BRS/Breakout without creating a one-sleeve book.",
        },
        {
            "name": "atrss_smaller_risk_more_capacity",
            "mutations": {
                "strategy_allocations.ATRSS.max_heat_R": 1.85,
                "symbol_risk_multipliers": {"ATRSS:QQQ": 0.85, "ATRSS:GLD": 0.85},
            },
            "rationale": "Test whether ATRSS extracts more real alpha from more accepted entries at smaller per-entry risk.",
        },
        {
            "name": "helix_brs_convex_tilt",
            "mutations": {
                "strategy_allocations.AKC_HELIX.max_heat_R": 1.50,
                "strategy_allocations.BRS_R9.max_heat_R": 1.40,
                "symbol_risk_multipliers": {
                    "AKC_HELIX:QQQ": 1.08,
                    "AKC_HELIX:GLD": 1.08,
                    "BRS_R9:QQQ": 1.12,
                    "BRS_R9:GLD": 1.08,
                },
            },
            "rationale": "Lean into Helix convexity and BRS drawdown-offset alpha while staying below broad portfolio DD caps.",
        },
        {
            "name": "risk_atrss_018",
            "mutations": {"strategy_allocations.ATRSS.unit_risk_pct": 0.018},
            "rationale": "ATRSS has 274 trades, 79.6% WR, and 2.29% DD; it earns the first active risk probe.",
        },
        {
            "name": "risk_helix_010_with_leakage_guard",
            "mutations": {
                "strategy_allocations.AKC_HELIX.unit_risk_pct": 0.010,
                "strategy_filters.AKC_HELIX.require_leakage_control": True,
            },
            "rationale": "Helix exit-alpha is the largest return engine, but right-then-stopped leakage must be guarded.",
        },
        {
            "name": "risk_brs_008_bear_only",
            "mutations": {"strategy_allocations.BRS_R9.unit_risk_pct": 0.008},
            "rationale": "BRS is already a bearish sleeve in this replay; test whether its live risk should be lifted modestly.",
        },
        {
            "name": "risk_breakout_006_low_dd",
            "mutations": {"strategy_allocations.SWING_BREAKOUT_V3.unit_risk_pct": 0.006},
            "rationale": "Breakout has only 0.32% DD and PF 21.2, but low AvgR keeps this a modest sleeve.",
        },
        {
            "name": "single_strategy_share_cap_45",
            "mutations": {"portfolio_rules.max_single_strategy_risk_share": 0.45},
            "rationale": "Prevent the optimizer from becoming a disguised Helix or ATRSS-only round.",
        },
    ],
    4: [
        {
            "name": "overlay_max_70",
            "mutations": {"strategy_allocations.OVERLAY.max_equity_pct": 0.70},
            "rationale": "Older portfolio experiments showed overlay_max_70 improved score while preserving overlay participation.",
        },
        {
            "name": "overlay_max_55_defensive",
            "mutations": {"strategy_allocations.OVERLAY.max_equity_pct": 0.55},
            "rationale": "If active signals are frequent, cap idle deployment harder to avoid passive risk drag.",
        },
        {
            "name": "overlay_bear_cap_35",
            "mutations": {
                "strategy_allocations.OVERLAY.bear_mode_max_equity_pct": 0.35,
                "cross_strategy_rules.brs_downturn_reserve.overlay_bear_cap_pct": 0.35,
            },
            "rationale": "Do not let long overlay fight BRS during bear regimes.",
        },
        {
            "name": "overlay_gld_tilt_55",
            "mutations": {"strategy_allocations.OVERLAY.weights": {"QQQ": 0.45, "GLD": 0.55}},
            "rationale": "Offset QQQ-heavy active books with a modest GLD idle-cash tilt.",
        },
        {
            "name": "overlay_off_control_only",
            "mutations": {"strategy_allocations.OVERLAY.enabled": False},
            "rationale": "Diagnostic control because older results favored overlay_off; it should not be adopted unless the user drops the overlay requirement.",
        },
    ],
    5: [
        {
            "name": "atrss_breakout_reactivation_probe",
            "mutations": {
                "strategy_filters.ATRSS.breakout_reactivation_probe": True,
                "strategy_filters.ATRSS.max_breakout_share": 0.20,
            },
            "rationale": "R9 has zero ATRSS breakout conversions; probe frequency only with a share cap and quality gate.",
        },
        {
            "name": "breakout_entry_window_widen_1h",
            "mutations": {"strategy_filters.SWING_BREAKOUT_V3.entry_window_extension_hours": 1},
            "rationale": "Breakout diagnostics show outside-entry-window blocks dominate rejected signals.",
        },
        {
            "name": "breakout_continuation_small_size",
            "mutations": {
                "strategy_filters.SWING_BREAKOUT_V3.enable_continuation_probe": True,
                "strategy_allocations.SWING_BREAKOUT_V3.continuation_size_mult": 0.50,
            },
            "rationale": "Continuations are currently absent; test them small to increase frequency without wrecking DD.",
        },
        {
            "name": "helix_stale_bail_tighter",
            "mutations": {"strategy_filters.AKC_HELIX.stale_exit_tighten": True},
            "rationale": "Helix stale exits are a clear negative bucket: 61 trades, PF 0.02.",
        },
        {
            "name": "brs_lh_bear_strong_block",
            "mutations": {"strategy_filters.BRS_R9.block_lh_rejection_bear_strong": True},
            "rationale": "BRS LH_REJECTION in BEAR_STRONG is negative; block it unless BD confirmation also fires.",
        },
    ],
    6: [
        {
            "name": "atrss_rank_score_per_risk",
            "mutations": {"strategy_filters.ATRSS.rank_mode": "score_per_risk"},
            "rationale": "Use a live-known candidate ranking that prefers conviction per unit stop distance when heat is scarce.",
        },
        {
            "name": "atrss_rank_stop_first_control",
            "mutations": {"strategy_filters.ATRSS.rank_mode": "stop_first"},
            "rationale": "Control probe for whether blocked-candidate rescue comes from lower stop distance rather than true conviction.",
        },
        {
            "name": "atrss_rank_gld_first_control",
            "mutations": {"strategy_filters.ATRSS.rank_mode": "gld_first"},
            "rationale": "Diagnostic control for the GLD-heavy blocked set; should only survive if broad portfolio metrics improve.",
        },
        {
            "name": "atrss_helix_same_symbol_cap",
            "mutations": {
                "cross_strategy_rules.atrss_helix_alignment.max_combined_symbol_heat_R": 2.10,
                "cross_strategy_rules.atrss_helix_alignment.same_symbol_same_direction_boost": 1.15,
            },
            "rationale": "Keep the useful ATRSS/Helix synergy without stacking too much one-symbol heat.",
        },
        {
            "name": "brs_downturn_reserve_125",
            "mutations": {"cross_strategy_rules.brs_downturn_reserve.reserve_R": 1.25},
            "rationale": "Give BRS enough capacity to fire when bear alpha is available.",
        },
        {
            "name": "breakout_reserved_slot_035",
            "mutations": {"cross_strategy_rules.breakout_opportunistic_slot.reserved_heat_R": 0.35},
            "rationale": "Protect low-frequency Breakout from always losing the heat race to ATRSS/Helix.",
        },
        {
            "name": "overlay_cut_when_heat_high",
            "mutations": {"cross_strategy_rules.overlay_idle_cash_policy.cut_when_active_heat_above_R": 2.75},
            "rationale": "Keep overlay from silently increasing gross exposure during active-signal clusters.",
        },
    ],
    7: [
        {
            "name": "daily_stop_325",
            "mutations": {"portfolio_rules.portfolio_daily_stop_R": 3.25},
            "rationale": "Aggressive enough for frequent ETF swings, but clips cascade days.",
        },
        {
            "name": "daily_stop_375_probe",
            "mutations": {"portfolio_rules.portfolio_daily_stop_R": 3.75},
            "rationale": "Only acceptable if the optimizer materially lifts return/frequency without DD regression.",
        },
        {
            "name": "drawdown_tiers_aggressive_controlled",
            "mutations": {
                "portfolio_rules.drawdown_tiers": (
                    (0.06, 1.00),
                    (0.09, 0.70),
                    (0.12, 0.40),
                    (0.15, 0.00),
                )
            },
            "rationale": "Press near highs, then force de-risking before drawdown reaches blow-up territory.",
        },
        {
            "name": "final_all_five_contribution_gate",
            "mutations": {
                "validation.min_active_strategies": 5,
                "validation.max_single_strategy_risk_share": 0.45,
            },
            "rationale": "The round is only valid if the final portfolio remains a five-strategy blend.",
        },
        {
            "name": "final_walk_forward_and_live_parity",
            "mutations": {
                "validation.walk_forward_required": True,
                "validation.completed_bar_and_fee_parity_required": True,
                "validation.min_positive_years": 4,
            },
            "rationale": "Reject alpha that depends on overfit periods, incomplete bars, or optimistic fees/fills.",
        },
    ],
}
