from __future__ import annotations

from pathlib import Path

from backtests.shared.auto.plugin_utils import mutation_signature
from backtests.shared.auto.types import ScoredCandidate
from backtests.stock.auto.alcb_p16_phase import plugin as alcb_plugin_mod
from backtests.stock.auto.iaric_pullback import plugin as iaric_plugin_mod


class _DummyBatchEvaluator:
    def __call__(self, candidates, current_mutations):
        del candidates, current_mutations
        return []

    def close(self) -> None:
        return None


def test_alcb_phase_batch_seeds_metrics_cache_with_raw_mutation_signature(monkeypatch, tmp_path: Path) -> None:
    plugin = alcb_plugin_mod.ALCBP16Plugin(tmp_path, max_workers=1)
    mutations = {"param_overrides.example": 1.0}
    metrics = {
        "net_profit": 1.0,
        "expectancy_dollar": 1.0,
        "expected_total_r": 1.0,
        "trades_per_month": 1.0,
        "profit_factor": 1.0,
        "max_drawdown_pct": 0.01,
    }

    monkeypatch.setattr(alcb_plugin_mod, "_LocalBatchEvaluator", lambda *args, **kwargs: _DummyBatchEvaluator())
    monkeypatch.setattr(plugin, "_run_config", lambda muts, **kwargs: {"metrics": dict(metrics)})
    monkeypatch.setattr(plugin, "_resolve_phase_hard_rejects", lambda phase, base_metrics, hard_rejects: dict(hard_rejects))
    monkeypatch.setattr(
        plugin,
        "_seed_result_for_metrics",
        lambda name, phase, base_metrics, hard_rejects, scoring_weights: ScoredCandidate(
            name=name,
            score=1.0,
            metrics=dict(base_metrics),
        ),
    )
    monkeypatch.setattr(plugin, "_replay_data_fingerprint", lambda: "fingerprint")

    plugin.create_evaluate_batch(1, mutations)

    assert plugin._metrics_cache == {mutation_signature(mutations): metrics}


def test_iaric_phase_batch_seeds_metrics_cache_with_raw_mutation_signature(monkeypatch, tmp_path: Path) -> None:
    plugin = iaric_plugin_mod.IARICPullbackPlugin(tmp_path, max_workers=1)
    mutations = {"param_overrides.example": 1.0}
    metrics = {
        "total_trades": 12.0,
        "avg_r": 0.2,
        "profit_factor": 2.0,
        "max_drawdown_pct": 0.01,
        "sharpe": 1.0,
    }

    monkeypatch.setattr(iaric_plugin_mod, "_LocalBatchEvaluator", lambda *args, **kwargs: _DummyBatchEvaluator())
    monkeypatch.setattr(plugin, "_run_config", lambda muts, **kwargs: {"metrics": dict(metrics)})
    monkeypatch.setattr(plugin, "_replay_data_fingerprint", lambda: "fingerprint")

    plugin.create_evaluate_batch(1, mutations)

    assert plugin._metrics_cache == {mutation_signature(mutations): metrics}
