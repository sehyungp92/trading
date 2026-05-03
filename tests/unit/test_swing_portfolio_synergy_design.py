from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.plugin_utils import mutation_signature
from backtests.shared.auto.phase_state import PhaseState
from backtests.swing.auto.portfolio_synergy.phase_candidates import (
    ROUND_TARGETS,
    SCORE_WEIGHTS,
    SEED_PORTFOLIO_CONFIG,
    STRATEGY_ORDER,
    get_phase_candidates,
    phase_summary,
)
from backtests.swing.auto.portfolio_synergy.evaluator import (
    _headline_equity_curve,
    _realized_equity_curve,
    build_effective_portfolio_config,
    build_replay_config,
)
from backtests.swing.auto.portfolio_synergy.plugin import PortfolioSynergyPlugin
from backtests.swing.auto.portfolio_synergy.portfolio_diagnostics import _block_summary, render_markdown
from backtests.swing.auto.portfolio_synergy.round_design import build_round_design
from backtests.swing.auto.portfolio_synergy.scoring import SCORE_COMPONENTS, score_portfolio_metrics
from backtests.swing.engine.unified_portfolio_engine import UnifiedPortfolioResult, _combined_child_mtm_equity


def test_swing_portfolio_synergy_design_and_seed_are_controlled_aggressive() -> None:
    design = build_round_design(Path("."))
    rules = SEED_PORTFOLIO_CONFIG["portfolio_rules"]
    allocations = SEED_PORTFOLIO_CONFIG["strategy_allocations"]

    assert design["initial_equity"] == 25_000.0
    assert design["risk_stance"] == "aggressive_controlled"
    assert [item["strategy_id"] for item in design["diagnostic_assessments"]] == list(STRATEGY_ORDER)
    assert len(design["seed_portfolio_config"]["strategy_allocations"]) == 5
    assert len(design["scoring_weights"]) <= 7
    assert abs(sum(design["scoring_weights"].values()) - 1.0) < 1e-9
    assert rules["heat_cap_R"] > 3.0
    assert rules["heat_cap_R"] <= 4.0
    assert rules["drawdown_tiers"][-1] == (0.15, 0.00)
    assert allocations["ATRSS"]["priority"] == 0
    assert allocations["AKC_HELIX"]["unit_risk_pct"] < allocations["ATRSS"]["unit_risk_pct"]
    assert allocations["OVERLAY"]["max_equity_pct"] <= ROUND_TARGETS["max_overlay_equity_pct"]


def test_swing_phase_space_prioritizes_alpha_frequency_with_bounded_score() -> None:
    summaries = phase_summary()

    assert len(summaries) == 7
    assert SCORE_WEIGHTS["alpha_return"] > SCORE_WEIGHTS["drawdown_control"]
    assert SCORE_WEIGHTS["trade_frequency"] > SCORE_WEIGHTS["profit_factor_quality"]
    assert len(SCORE_COMPONENTS) == 7
    assert tuple(SCORE_WEIGHTS) == SCORE_COMPONENTS
    assert ROUND_TARGETS["hard_max_drawdown_pct"] == 0.12
    assert any(candidate["name"] == "heat_cap_4_25_probe" for candidate in get_phase_candidates(2))
    assert any(candidate["name"] == "overlay_off_control_only" for candidate in get_phase_candidates(4))
    assert any(candidate["name"] == "drawdown_tiers_aggressive_controlled" for candidate in get_phase_candidates(7))

    score = score_portfolio_metrics(
        {
            "net_return_pct": 1.0,
            "active_trades_per_month": 12.0,
            "max_drawdown_pct": 0.08,
            "profit_factor": 2.6,
            "max_strategy_trade_share": 0.40,
            "active_strategy_count": 5.0,
            "overlay_return_pct": 0.05,
            "overlay_max_equity_pct": 0.70,
            "sharpe": 1.8,
            "positive_years": 4.0,
            "overlay_enabled": 1.0,
        }
    )

    assert not score.rejected
    assert set(score.components) == set(SCORE_COMPONENTS)


def test_swing_portfolio_synergy_plugin_caps_workers_and_namespaces_cache(monkeypatch) -> None:
    plugin = PortfolioSynergyPlugin(Path("backtests/swing/data/raw"), max_workers=8)
    monkeypatch.setattr(
        plugin,
        "_ensure_bundle",
        lambda: type("Bundle", (), {"cache_source_fingerprint": "swing-portfolio-fp"})(),
    )

    spec = plugin.get_phase_spec(1, state=PhaseState())
    evaluator = plugin.create_evaluate_batch(2, {})

    assert plugin.max_workers == 2
    assert len(spec.scoring_weights or {}) == 7
    assert evaluator._signature_prefix == build_cache_key(
        "swing.portfolio_synergy.evaluation",
        source_fingerprint="swing-portfolio-fp",
        extra={
            "phase": 2,
            "score_components": list(SCORE_COMPONENTS),
            "scoring_weights": {},
            "hard_rejects": {},
            "initial_equity": 25_000.0,
        },
    )


def test_swing_portfolio_synergy_final_metrics_add_total_score_from_cache() -> None:
    plugin = PortfolioSynergyPlugin(Path("backtests/swing/data/raw"), max_workers=2)
    mutations = {"risk_stance": "aggressive_controlled"}
    cached_metrics = {
        "net_return_pct": 1.0,
        "active_trades_per_month": 12.0,
        "max_drawdown_pct": 0.08,
        "profit_factor": 2.6,
        "max_strategy_trade_share": 0.40,
        "active_strategy_count": 5.0,
        "overlay_return_pct": 0.05,
        "overlay_max_equity_pct": 0.55,
        "sharpe": 1.8,
        "positive_years": 4.0,
        "overlay_enabled": 1.0,
    }
    plugin._metrics_cache[mutation_signature(mutations)] = dict(cached_metrics)

    metrics = plugin.compute_final_metrics(mutations)

    expected = score_portfolio_metrics(cached_metrics, scoring_weights=SCORE_WEIGHTS)
    assert metrics["score_total"] == expected.total
    assert set(metrics) >= {f"score_{key}" for key in SCORE_COMPONENTS}


def test_swing_portfolio_mtm_curve_is_headline_and_keeps_realized_curve() -> None:
    result = UnifiedPortfolioResult(
        combined_equity=np.array([25_000.0, 24_850.0]),
        combined_equity_mtm=np.array([25_000.0, 24_850.0]),
        combined_equity_realized=np.array([25_000.0, 25_020.0]),
        overlay_pnl=-25.0,
        overlay_commission=5.0,
    )

    assert _headline_equity_curve(result)[-1] == 24_850.0
    assert _realized_equity_curve(result)[-1] == 25_020.0


def test_swing_combined_mtm_includes_child_marks_and_overlay_costs() -> None:
    initial_equity = 25_000.0
    atrss = SimpleNamespace(equity_curve=[24_900.0], equity=25_000.0)
    brs = SimpleNamespace(equity_curve=[-50.0], equity=0.0)

    combined = _combined_child_mtm_equity(
        initial_equity,
        [("atrss", {"QQQ": atrss}), ("brs", {"GLD": brs})],
        overlay_pnl=-25.0,
    )

    assert combined == 24_825.0


def test_swing_portfolio_synergy_diagnostics_include_win_rate_ratios_and_blocks() -> None:
    plugin = PortfolioSynergyPlugin(Path("backtests/swing/data/raw"), max_workers=2)
    score = score_portfolio_metrics(
        {
            "net_return_pct": 1.0,
            "active_trades_per_month": 12.0,
            "max_drawdown_pct": 0.08,
            "profit_factor": 2.6,
            "max_strategy_trade_share": 0.40,
            "active_strategy_count": 5.0,
            "overlay_return_pct": 0.05,
            "overlay_max_equity_pct": 0.55,
            "sharpe": 1.8,
            "positive_years": 4.0,
            "overlay_enabled": 1.0,
        },
        scoring_weights=SCORE_WEIGHTS,
    )
    metrics = {
        "final_equity": 50_000.0,
        "net_pnl": 25_000.0,
        "net_return_pct": 1.0,
        "active_trades_per_month": 12.0,
        "total_r_per_month": 6.5,
        "profit_factor": 2.6,
        "win_rate": 0.61,
        "max_drawdown_pct": 0.08,
        "sharpe": 1.8,
        "sortino": 1.1,
        "calmar": 3.2,
        "active_strategy_count": 5.0,
        "max_strategy_trade_share": 0.40,
        "trade_capture_ratio": 0.72,
        "positive_alpha_block_rate": 0.18,
        "overlay_max_equity_pct": 0.55,
        "overlay_pnl": 5_000.0,
        **{f"score_{key}": value for key, value in score.components.items()},
        "score_total": score.total,
    }

    text = plugin._format_diagnostics("TEST", metrics, None)

    assert "Win rate: 61.00%" in text
    assert "Sharpe: 1.80" in text
    assert "Sortino: 1.10" in text
    assert "Calmar: 3.20" in text
    assert "Positive-alpha block rate: 18.00%" in text


def test_swing_portfolio_detailed_diagnostics_report_block_reasons() -> None:
    events = [
        {
            "strategy": "AKC_HELIX",
            "symbol": "QQQ",
            "stage": "entry",
            "status": "blocked",
            "reason": "Portfolio heat cap (3.50R + 0.50R > 3.75R)",
            "risk_dollars": 125.0,
            "risk_context": {
                "portfolio_open_risk_R": 3.50,
                "portfolio_request_risk_R": 0.50,
                "strategy_after_request_R": 1.40,
            },
        },
        {
            "strategy": "ATRSS",
            "symbol": "GLD",
            "stage": "entry",
            "status": "accepted",
            "reason": "",
            "risk_dollars": 100.0,
            "risk_context": {"portfolio_open_risk_R": 1.0},
        },
    ]

    summary = _block_summary(events)

    assert summary["Portfolio heat cap"]["count"] == 1
    assert summary["Portfolio heat cap"]["by_strategy"] == {"AKC_HELIX": 1}
    assert summary["Portfolio heat cap"]["raw_reason_examples"] == [
        "Portfolio heat cap (3.50R + 0.50R > 3.75R)"
    ]

    text = render_markdown(
        {
            "headline": {
                "fired_entries": 2.0,
                "accepted_entries": 1.0,
                "blocked_entries": 1.0,
                "block_rate": 0.5,
                "closed_trades": 1.0,
                "final_equity": 26000.0,
                "net_return_pct": 0.04,
                "profit_factor": 2.0,
                "win_rate": 0.6,
                "max_drawdown_pct": 0.05,
                "sharpe": 1.2,
                "sortino": 0.8,
                "calmar": 2.5,
                "score_total": 0.7,
            },
            "notes": {"blocked_outcome_backfill": "Blocked outcomes are not backfilled."},
            "strategy_summary": {
                "ATRSS": {
                    "fired": 1,
                    "accepted": 1,
                    "blocked": 0,
                    "accept_rate": 1.0,
                    "closed_trades": 1,
                    "closed_win_rate": 1.0,
                    "pnl": 100.0,
                    "total_r": 1.0,
                    "avg_r": 1.0,
                    "top_block_reason": "",
                },
                "AKC_HELIX": {
                    "fired": 1,
                    "accepted": 0,
                    "blocked": 1,
                    "accept_rate": 0.0,
                    "closed_trades": 0,
                    "closed_win_rate": 0.0,
                    "pnl": 0.0,
                    "total_r": 0.0,
                    "avg_r": 0.0,
                    "top_block_reason": "Portfolio heat cap",
                },
                "BRS_R9": {},
                "SWING_BREAKOUT_V3": {},
            },
            "block_summary": summary,
            "monthly_flow": {"2026-01": {"fired": 2, "accepted": 1, "blocked": 1, "block_rate": 0.5}},
            "overlay": {
                "enabled": True,
                "mode": "multi",
                "max_equity_pct": 0.55,
                "min_alloc_pct": 0.25,
                "max_alloc_pct": 1.0,
                "weights": {"QQQ": 0.45, "GLD": 0.55},
                "pnl": 100.0,
                "return_pct": 0.004,
                "per_symbol_pnl": {"QQQ": 60.0, "GLD": 40.0},
            },
            "crowding_summary": {
                "pct_events_with_other_signal_within_4h": 0.5,
                "pct_blocked_with_other_signal_within_4h": 1.0,
                "pct_blocked_when_portfolio_heat_above_70pct": 1.0,
                "avg_portfolio_open_R_before_block": 3.5,
                "top_pairs_within_4h": {"AKC_HELIX / ATRSS": 1},
            },
            "heat": {"avg_heat_R": 1.0, "max_heat_R": 3.5, "pct_time_at_cap": 0.0},
            "coordination": {"tighten_events": 0, "boost_events": 0, "portfolio_daily_stop_activations": 0},
            "individual_strategy_reference": {},
            "interpretation": {
                "dominant_block_reason": "Portfolio heat cap",
                "most_blocked_strategy": "AKC_HELIX",
                "fired_to_closed_conversion": 0.5,
            },
        }
    )

    assert "Fired, Accepted, Blocked By Sleeve" in text
    assert "Overlay Idle-Cash Sleeve" in text
    assert "Portfolio heat cap" in text
    assert "Blocked outcomes are not backfilled." in text


def test_swing_portfolio_synergy_builds_unified_brs_runtime_config() -> None:
    effective = build_effective_portfolio_config(SEED_PORTFOLIO_CONFIG)
    cfg = build_replay_config(effective, Path("backtests/swing/data/raw"), 25_000.0)

    assert cfg.brs.strategy_id == "BRS_R9"
    assert cfg.brs.priority == SEED_PORTFOLIO_CONFIG["strategy_allocations"]["BRS_R9"]["priority"]
    assert cfg.brs_runtime_config is not None
    assert cfg.brs_runtime_config.heat_cap_r == cfg.brs.max_heat_R
    assert cfg.brs_runtime_config.max_concurrent == cfg.brs.max_working_orders
