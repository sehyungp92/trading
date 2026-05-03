from __future__ import annotations

from types import SimpleNamespace

from backtests.momentum.auto.akc_helix.phase_candidates import get_phase_candidates
from backtests.momentum.auto.akc_helix.plugin import AKCHelixPlugin
from backtests.momentum.auto.akc_helix.scoring import score_phase_metrics
from backtests.momentum.engine.helix_engine import _ablation_patch
from backtests.shared.auto.phase_state import PhaseState
from strategies.momentum.helix_v40 import config as helix_config
from strategies.momentum.helix_v40 import gates as helix_gates
from strategies.momentum.helix_v40 import signals as helix_signals


def test_helix_ablation_patch_updates_new_structural_entry_constants() -> None:
    original_rearm = helix_signals.SECOND_CHANCE_REARM_ENABLED
    original_high_vol = helix_signals.HIGH_VOL_RETEST_ENABLED
    original_high_vol_m_cont = helix_signals.HIGH_VOL_M_CONT_ENABLED
    original_reclaim = helix_signals.CLASS_F_RECLAIM_ENABLED

    with _ablation_patch(
        {
            "SECOND_CHANCE_REARM_ENABLED": True,
            "HIGH_VOL_RETEST_ENABLED": True,
            "HIGH_VOL_M_CONT_ENABLED": True,
            "CLASS_F_RECLAIM_ENABLED": True,
        }
    ):
        assert helix_signals.SECOND_CHANCE_REARM_ENABLED is True
        assert helix_signals.HIGH_VOL_RETEST_ENABLED is True
        assert helix_signals.HIGH_VOL_M_CONT_ENABLED is True
        assert helix_signals.CLASS_F_RECLAIM_ENABLED is True
        assert helix_config.SECOND_CHANCE_REARM_ENABLED is True

    assert helix_signals.SECOND_CHANCE_REARM_ENABLED == original_rearm
    assert helix_signals.HIGH_VOL_RETEST_ENABLED == original_high_vol
    assert helix_signals.HIGH_VOL_M_CONT_ENABLED == original_high_vol_m_cont
    assert helix_signals.CLASS_F_RECLAIM_ENABLED == original_reclaim


def test_helix_phase_candidates_include_round7_contextual_entry_families() -> None:
    phase1 = {name for name, _ in get_phase_candidates(1)}
    phase2 = {name for name, _ in get_phase_candidates(2)}
    phase3 = {name for name, _ in get_phase_candidates(3)}
    phase4 = {name for name, _ in get_phase_candidates(4)}
    phase5 = {name for name, _ in get_phase_candidates(5)}

    assert "t_secondary_eth_pm_context" in phase1
    assert "normvol_short_prime2_context" in phase2
    assert "highvol_long_prime1_context" in phase3
    assert "classf_context_reclaim_stack" in phase4
    assert "final_context_stack_balanced" in phase5


def test_helix_phase_scoring_prefers_clean_structural_progress_over_soft_frequency() -> None:
    clean_structural_progress = SimpleNamespace(
        total_trades=85,
        max_drawdown_pct=0.153,
        profit_factor=4.20,
        calmar=11.0,
        net_profit=140_632.0,
        expectancy_dollar=1654.0,
    )
    softer_frequency_push = SimpleNamespace(
        total_trades=92,
        max_drawdown_pct=0.196,
        profit_factor=3.21,
        calmar=7.2,
        net_profit=121_900.0,
        expectancy_dollar=1325.0,
    )

    clean_score = score_phase_metrics(1, clean_structural_progress, 10_000.0)
    soft_score = score_phase_metrics(4, softer_frequency_push, 10_000.0)

    assert not clean_score.rejected
    assert soft_score.rejected
    assert clean_score.total > soft_score.total


def test_phase1_dynamic_trade_floor_preserves_seeded_baseline_score() -> None:
    plugin = AKCHelixPlugin(data_dir="backtests/momentum/data/raw", max_workers=1)
    state = PhaseState(cumulative_mutations={"baseline.seed": True})
    baseline_metrics = {
        "total_trades": 84,
        "net_profit": 142_012.3,
        "expectancy_dollar": 1690.62,
        "profit_factor": 4.332,
        "calmar": 11.15,
        "max_drawdown_pct": 0.1534,
    }
    plugin._metrics_for_mutations = lambda _mutations: baseline_metrics  # type: ignore[method-assign]

    spec = plugin.get_phase_spec(1, state)

    assert spec.hard_rejects["min_trades"] == 84


def test_phase1_gate_rejects_one_trade_gain_when_quality_leaks_from_baseline() -> None:
    plugin = AKCHelixPlugin(data_dir="backtests/momentum/data/raw", max_workers=1)
    state = PhaseState(cumulative_mutations={"baseline.seed": True})
    baseline_metrics = {
        "total_trades": 84,
        "net_profit": 142_012.3,
        "expectancy_dollar": 1690.62,
        "profit_factor": 4.332,
        "calmar": 11.15,
        "max_drawdown_pct": 0.1534,
    }
    plugin._metrics_for_mutations = lambda _mutations: baseline_metrics  # type: ignore[method-assign]

    criteria = plugin._gate_criteria(1, {
        "total_trades": 85,
        "net_profit": 140_632.42,
        "expectancy_dollar": 1654.50,
        "profit_factor": 4.1968,
        "calmar": 11.09,
        "max_drawdown_pct": 0.1534,
    }, state)

    by_name = {criterion.name: criterion for criterion in criteria}
    assert by_name["no_regress_net_profit"].passed is False
    assert by_name["no_regress_expectancy_dollar"].passed is False
    assert by_name["no_regress_profit_factor"].passed is False


def test_descriptive_phase_score_can_be_computed_without_hard_rejects() -> None:
    seeded_baseline = SimpleNamespace(
        total_trades=84,
        max_drawdown_pct=0.1534,
        profit_factor=4.332,
        calmar=11.15,
        net_profit=142_012.3,
        expectancy_dollar=1690.62,
    )

    score = score_phase_metrics(5, seeded_baseline, 10_000.0, apply_hard_rejects=False)

    assert not score.rejected
    assert score.total > 0.0


def test_phase5_gate_rejects_final_push_when_expectancy_and_pf_fall_too_far() -> None:
    plugin = AKCHelixPlugin(data_dir="backtests/momentum/data/raw", max_workers=1)
    state = PhaseState()
    state.phase_results[4] = {
        "final_metrics": {
            "total_trades": 90,
            "net_profit": 136_875.0,
            "expectancy_dollar": 1573.0,
            "profit_factor": 3.87,
            "calmar": 10.5,
            "max_drawdown_pct": 0.153,
        }
    }

    criteria = plugin._gate_criteria(5, {
        "total_trades": 93,
        "net_profit": 121_312.0,
        "expectancy_dollar": 1304.0,
        "profit_factor": 3.18,
        "calmar": 7.1,
        "max_drawdown_pct": 0.201,
    }, state)

    by_name = {criterion.name: criterion for criterion in criteria}
    assert by_name["floor_expectancy_dollar"].passed is False
    assert by_name["floor_profit_factor"].passed is False
    assert by_name["no_regress_net_profit"].passed is False


def test_structural_high_vol_gate_only_opens_variant_specific_retests() -> None:
    structural_retest = SimpleNamespace(
        cls=helix_config.SetupClass.T,
        direction=1,
        alignment_score=2,
        strong_trend=True,
        signal_variant="high_vol_retest",
    )
    generic_t = SimpleNamespace(
        cls=helix_config.SetupClass.T,
        direction=1,
        alignment_score=2,
        strong_trend=True,
        signal_variant="",
    )

    with _ablation_patch(
        {
            "HIGH_VOL_RETEST_ENABLED": True,
            "HIGH_VOL_RETEST_MIN_VOL_PCT": 90,
            "HIGH_VOL_RETEST_MAX_VOL_PCT": 101,
            "HIGH_VOL_CONTINUATION_REOPEN_ENABLED": False,
        }
    ):
        assert helix_gates.allow_high_vol_structural_retest(structural_retest, 95)
        assert not helix_gates.allow_high_vol_structural_retest(generic_t, 95)


def test_structural_high_vol_gate_only_opens_variant_specific_m_continuations() -> None:
    structural_m_cont = SimpleNamespace(
        cls=helix_config.SetupClass.M,
        direction=-1,
        alignment_score=1,
        strong_trend=False,
        signal_variant="high_vol_m_cont",
    )
    generic_m = SimpleNamespace(
        cls=helix_config.SetupClass.M,
        direction=-1,
        alignment_score=1,
        strong_trend=False,
        signal_variant="",
    )

    with _ablation_patch(
        {
            "HIGH_VOL_M_CONT_ENABLED": True,
            "HIGH_VOL_M_CONT_MIN_VOL_PCT": 94,
            "HIGH_VOL_M_CONT_MAX_VOL_PCT": 96,
        }
    ):
        assert helix_gates.allow_high_vol_structural_continuation(structural_m_cont, 95)
        assert not helix_gates.allow_high_vol_structural_continuation(generic_m, 95)


def test_class_f_reclaim_stacks_on_top_of_base_f_when_enabled(monkeypatch) -> None:
    engine = helix_signals.SignalEngine()
    base_setup = SimpleNamespace(direction=1)
    reclaim_setup = SimpleNamespace(direction=-1)

    monkeypatch.setattr(
        helix_signals.SignalEngine,
        "_detect_class_F_base",
        lambda self, h1, h4, daily, now_et, vol_pct=None: [base_setup],
    )
    monkeypatch.setattr(
        helix_signals.SignalEngine,
        "_detect_class_F_reclaim",
        lambda self, h1, h4, daily, now_et, pivots_1h, vol_pct, blocked_directions=None: [reclaim_setup],
    )

    with _ablation_patch({"CLASS_F_RECLAIM_ENABLED": True}):
        setups = engine.detect_class_F(None, None, None, None)  # type: ignore[arg-type]

    assert setups == [base_setup, reclaim_setup]


def test_normal_vol_short_variant_can_bypass_scalar_short_floor() -> None:
    setup = SimpleNamespace(
        cls=helix_config.SetupClass.M,
        direction=-1,
        alignment_score=0,
        strong_trend=False,
        signal_variant="normal_vol_m_short",
    )

    with _ablation_patch(
        {
            "NORM_VOL_M_SHORT_ENABLED": True,
            "NORM_VOL_M_SHORT_ALLOW_RTH_PRIME2": True,
            "NORM_VOL_M_SHORT_MIN_VOL_PCT": 50,
            "NORM_VOL_M_SHORT_MAX_VOL_PCT": 80,
        }
    ):
        assert helix_gates.allow_short_below_min_score(
            setup,
            helix_config.SessionBlock.RTH_PRIME2,
            65,
            1,
        )
