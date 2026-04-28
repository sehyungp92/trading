from __future__ import annotations

import json

from backtests.shared.auto.phase_state import PhaseState, save_phase_state
from backtests.shared.auto.round_manager import RoundManager, canonicalize_metrics


def test_bootstrap_round_1_writes_canonical_artifacts(tmp_path) -> None:
    diagnostics_src = tmp_path / "legacy_full_diagnostics.txt"
    diagnostics_src.write_text("diagnostics\n", encoding="utf-8")

    phase_state_src = tmp_path / "legacy_phase_state.json"
    save_phase_state(
        PhaseState(
            current_phase=2,
            completed_phases=[1, 2],
            cumulative_mutations={"flags.enabled": True},
            phase_results={2: {"final_metrics": {"total_trades": 11}}},
        ),
        phase_state_src,
    )

    diagnostics_summary_src = tmp_path / "legacy_summary.json"
    diagnostics_summary_src.write_text('{"ok": true}\n', encoding="utf-8")

    round_dir = RoundManager.bootstrap_round_1(
        "momentum",
        "sample",
        {"flags.enabled": True},
        diagnostics_src,
        phase_state_src,
        diagnostics_summary_src_path=diagnostics_summary_src,
        base_dir=tmp_path / "output",
        final_metrics={"total_trades": 11, "win_rate": 0.5},
        completed_phases=[1, 2],
    )

    assert (round_dir / "round_final_diagnostics.txt").read_text(encoding="utf-8") == "diagnostics\n"
    assert (round_dir / "phase_state.json").read_text(encoding="utf-8") == phase_state_src.read_text(encoding="utf-8")
    assert (round_dir / "diagnostics_summary.json").read_text(encoding="utf-8") == '{"ok": true}\n'
    assert (round_dir / "run_spec.json").exists()
    assert (round_dir / "run_summary.json").exists()
    assert json.loads((round_dir / "optimized_config.json").read_text(encoding="utf-8")) == {"flags.enabled": True}


def test_resolve_round_reuses_in_progress_then_advances_when_complete(tmp_path) -> None:
    manager = RoundManager("swing", "sample", base_dir=tmp_path / "output")

    round_num, round_dir = manager.resolve_round(None, for_write=True, expected_phases=2)
    assert round_num == 1
    assert round_dir.name == "round_1"

    save_phase_state(PhaseState(current_phase=1, completed_phases=[1]), manager.phase_state_path(round_dir))
    round_num, round_dir = manager.resolve_round(None, for_write=True, expected_phases=2)
    assert round_num == 1

    complete_state = PhaseState(
        current_phase=2,
        completed_phases=[1, 2],
        cumulative_mutations={"flags.enabled": True},
        phase_results={2: {"final_metrics": {"total_trades": 12}}},
    )
    save_phase_state(complete_state, manager.phase_state_path(round_dir))
    manager.write_run_summary(round_dir, complete_state.cumulative_mutations, {"total_trades": 12}, [1, 2], round_num=1)
    manager.write_optimized_config(round_dir, complete_state.cumulative_mutations)
    manager.append_to_manifest(1, complete_state.cumulative_mutations, {"total_trades": 12})

    next_round_num, next_round_dir = manager.resolve_round(None, for_write=True, expected_phases=2)
    assert next_round_num == 2
    assert next_round_dir.name == "round_2"


def test_canonicalize_metrics_normalizes_aliases_and_percent_units() -> None:
    metrics = canonicalize_metrics(
        {
            "trades": 9,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "max_dd_pct": 0.12,
            "return_pct": 0.34,
            "sharpe": 1.25,
            "calmar": 2.5,
        }
    )

    assert metrics == {
        "total_trades": 9,
        "win_rate": 55.00000000000001,
        "profit_factor": 1.8,
        "max_drawdown_pct": 12.0,
        "net_return_pct": 34.0,
        "sharpe_ratio": 1.25,
        "calmar_ratio": 2.5,
    }


def test_resolve_round_rejects_reusing_completed_rounds_and_skipping_numbers(tmp_path) -> None:
    manager = RoundManager("stock", "sample", base_dir=tmp_path / "output")
    round_dir = manager.get_round_dir(1)
    complete_state = PhaseState(
        current_phase=2,
        completed_phases=[1, 2],
        cumulative_mutations={"flags.enabled": True},
        phase_results={2: {"final_metrics": {"total_trades": 12}}},
    )
    save_phase_state(complete_state, manager.phase_state_path(round_dir))
    manager.write_run_summary(round_dir, complete_state.cumulative_mutations, {"total_trades": 12}, [1, 2], round_num=1)
    manager.write_optimized_config(round_dir, complete_state.cumulative_mutations)
    manager.append_to_manifest(1, complete_state.cumulative_mutations, {"total_trades": 12})

    try:
        manager.resolve_round(1, for_write=True, expected_phases=2)
        assert False, "Expected FileExistsError when reusing a completed round."
    except FileExistsError:
        pass

    try:
        manager.resolve_round(3, for_write=True, expected_phases=2)
        assert False, "Expected ValueError when skipping round numbers."
    except ValueError:
        pass

    next_round, next_dir = manager.resolve_round(2, for_write=True, expected_phases=2)
    assert next_round == 2
    assert next_dir.name == "round_2"


def test_resolve_round_detects_missing_directory_for_manifest_round(tmp_path) -> None:
    manager = RoundManager("momentum", "sample", base_dir=tmp_path / "output")
    manager.append_to_manifest(1, {"flags.enabled": True}, {"total_trades": 8})

    try:
        manager.resolve_round(None, for_write=False)
        assert False, "Expected FileNotFoundError when latest manifest round has no directory."
    except FileNotFoundError:
        pass
