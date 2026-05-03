from __future__ import annotations

import json
from pathlib import Path

from backtests.momentum.auto.nq_regime_round4c.plugin import NqRegimePlugin
from backtests.momentum.auto.nq_regime_round4c.scoring import IMMUTABLE_WEIGHTS
from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.phase_state import _utc_now_iso


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    output_root = repo_root / "backtests" / "output" / "momentum" / "nq_regime"
    baseline_path = output_root / "round_3" / "optimized_config.json"
    output_dir = output_root / "round_4c"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    plugin = NqRegimePlugin(
        data_dir=repo_root / "backtests" / "momentum" / "data" / "raw",
        initial_equity=10_000.0,
        max_workers=2,
        analysis_symbol="NQ",
        trade_symbol="MNQ",
    )
    plugin.initial_mutations = dict(baseline)
    source_fingerprint = plugin.source_fingerprint()
    source_fingerprint_parts = plugin.source_fingerprint_parts()

    run_spec = {
        "family": "momentum",
        "strategy": "nq_regime",
        "strategy_name": "nq_regime",
        "round": "4c",
        "description": "round_4c isolated nq_3-first alpha/frequency optimization with dedicated exit-capture phase",
        "generated_at_utc": _utc_now_iso(),
        "baseline_source": str(baseline_path.resolve()),
        "baseline_mutation_count": len(baseline),
        "baseline_mutations": baseline,
        "scoring_weights": dict(IMMUTABLE_WEIGHTS),
        "max_workers": 2,
        "source_fingerprint": source_fingerprint,
        "source_fingerprint_parts": source_fingerprint_parts,
        "replay_command": "python -m backtests.momentum.auto.nq_regime_round4c.run",
    }
    (output_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2), encoding="utf-8")

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="round_4c",
        max_rounds=8,
        min_delta=0.003,
        max_retries=0,
        max_diagnostic_retries=0,
    )
    state = runner.run_all_phases()
    final_metrics = plugin.compute_final_metrics(state.cumulative_mutations)
    (output_dir / "optimized_config.json").write_text(json.dumps(state.cumulative_mutations, indent=2), encoding="utf-8")
    summary = {
        "family": "momentum",
        "strategy": "nq_regime",
        "round": "4c",
        "generated_at_utc": _utc_now_iso(),
        "completed_phases": state.completed_phases,
        "mutation_count": len(state.cumulative_mutations),
        "cumulative_mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "source_fingerprint": plugin.source_fingerprint(),
        "source_fingerprint_parts": plugin.source_fingerprint_parts(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    reproducibility_manifest = {
        "family": "momentum",
        "strategy": "nq_regime",
        "round": "4c",
        "generated_at_utc": summary["generated_at_utc"],
        "source_fingerprint": summary["source_fingerprint"],
        "source_fingerprint_parts": summary["source_fingerprint_parts"],
        "baseline_source": str(baseline_path.resolve()),
        "optimized_config_path": str((output_dir / "optimized_config.json").resolve()),
        "run_summary_path": str((output_dir / "run_summary.json").resolve()),
        "replay_command": run_spec["replay_command"],
        "final_metrics": final_metrics,
    }
    (output_dir / "reproducibility_manifest.json").write_text(json.dumps(reproducibility_manifest, indent=2), encoding="utf-8")
    plugin.close_pool()


if __name__ == "__main__":
    main()
