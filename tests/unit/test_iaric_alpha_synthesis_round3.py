from __future__ import annotations

from backtests.stock.auto.runners.run_iaric_alpha_synthesis_round3 import (
    _broad_unified_shortlist,
    _candidate_gates,
    _credible_research2_candidate,
    _survivor_incremental_audit,
    _value_creation_verification,
)


def _pool_candidate(candidate_id: str, family: str, trades: int, total_r: float) -> dict:
    return {
        "id": candidate_id,
        "root_family": family,
        "mutations": {"candidate": candidate_id},
        "metrics": {
            "total_trades": trades,
            "expected_total_r": total_r,
            "avg_r": total_r / trades,
            "profit_factor": 1.6,
            "max_drawdown_pct": 0.04,
            "robust_avg_r": 0.05,
        },
    }


def test_broad_shortlist_recovers_full_pool_diversity_under_unified_score() -> None:
    pool = [
        _pool_candidate("control_oversold", "control", 90, 27.0),
        _pool_candidate("old_pruned_frequency", "multi", 150, 24.0),
        _pool_candidate("old_pruned_total_r", "floor", 100, 35.0),
        _pool_candidate("orthogonal_route", "opening", 95, 25.0),
    ]

    selected, catalog = _broad_unified_shortlist(pool, cap=8)

    assert {row["root_family"] for row in selected} == {"control", "multi", "floor", "opening"}
    assert catalog["preservation_gate_passed"] is True
    assert catalog["source_candidates"] == 4


def test_alpha_synthesis_accepts_frequency_expansion_with_noninferior_total_r() -> None:
    baseline = {
        "id": "control_oversold",
        "metrics": {
            "total_trades": 88,
            "expected_total_r": 30.9,
            "avg_r": 0.35,
            "profit_factor": 2.29,
            "max_drawdown_pct": 0.025,
            "robust_avg_r": 0.20,
        },
    }
    candidate = {
        "id": "expanded",
        "metrics": {
            "total_trades": 116,
            "expected_total_r": 30.2,
            "avg_r": 0.26,
            "profit_factor": 2.0,
            "max_drawdown_pct": 0.033,
            "robust_avg_r": 0.12,
        },
        "validation": {
            "folds": [
                {"total_r": 8.0},
                {"total_r": 9.0},
                {"total_r": 13.2},
            ],
        },
    }
    attribution = {
        "ex_top3_total_r": 20.0,
        "ex_top3_profit_factor": 1.5,
        "max_single_symbol_net_share": 0.20,
    }

    gates = _candidate_gates(
        candidate,
        baseline,
        attribution,
        {"probability_positive": 0.80},
    )

    assert all(gates.values())


def test_final_value_verification_requires_real_uplift_and_robustness() -> None:
    baseline = {
        "id": "control_oversold",
        "unified_score": 0.50,
        "metrics": {
            "total_trades": 90,
            "expected_total_r": 27.0,
            "avg_r": 0.30,
            "profit_factor": 2.0,
            "max_drawdown_pct": 0.03,
        },
    }
    selected = {
        "id": "expanded",
        "unified_score": 0.56,
        "mutations": {"route": "multi"},
        "metrics": {
            "total_trades": 110,
            "expected_total_r": 30.0,
            "avg_r": 0.27,
            "profit_factor": 1.8,
            "max_drawdown_pct": 0.04,
            "robust_avg_r": 0.15,
            "entry_realized_discrimination_lift_r": 0.10,
        },
        "validation": {"folds": [
            {"total_r": 8.0}, {"total_r": 9.0}, {"total_r": 13.0},
        ]},
        "attribution": {
            "ex_top3_total_r": 20.0,
            "ex_top3_profit_factor": 1.4,
            "max_single_symbol_net_share": 0.20,
        },
        "paired_bootstrap": {"observed_delta_r": 3.0, "probability_positive": 0.80},
        "alpha_synthesis_eligible": True,
    }

    result = _value_creation_verification(
        selected,
        [selected, baseline],
        {"route": "oversold"},
        {"gate_passed": True},
    )

    assert result["gate_passed"] is True
    assert result["deltas"]["expected_total_r"] == 3.0


def test_positive_research2_near_miss_remains_eligible_for_incremental_audit() -> None:
    row = {
        "selected_aperture": "score_50",
        "selected_entry_variant": "next_bar_open",
        "folds": {
            "middle": {"events": 25, "avg_r": 0.10},
            "latest": {"events": 25, "avg_r": 0.08},
        },
        "validation_bootstrap_probability_positive": 0.85,
        "route_ready_for_portfolio_replay": False,
    }

    assert _credible_research2_candidate(row) is True


def test_structural_route_requires_positive_unique_alpha_in_validation_folds() -> None:
    events = []
    for fold_index, fold in enumerate(("early", "middle", "latest")):
        for index in range(20):
            events.append({
                "family": "GAP_EXHAUSTION_RECLAIM",
                "symbol": f"S{index:02d}",
                "date": f"202{4 + fold_index}-01-{(index % 20) + 1:02d}",
                "fold": fold,
                "score": 70.0,
                "horizon_r": {"bar_12": 0.20},
            })

    result = _survivor_incremental_audit(
        "GAP_EXHAUSTION_RECLAIM",
        {"selected_aperture": "all_events", "selected_horizon": "bar_12"},
        events,
        incumbent_keys=set(),
        simulations=50,
    )

    assert result["unique_events"] == 60
    assert result["admitted_for_exact_route_replay"] is True
    assert all(result["gates"].values())
