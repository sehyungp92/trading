from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.shared.auto.phase_state import load_phase_state
from backtests.stock.auto.alcb_p16_phase.plugin import ALCBP16Plugin


DEFAULT_OUTPUT_DIR = Path("backtests/stock/auto/alcb_p16_phase/output")
DEFAULT_PHASE_STATE = DEFAULT_OUTPUT_DIR / "phase_state.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "round_final_diagnostics.txt"

FLOOR_NAME_MAP = {
    "expectancy_dollar": "min_expectancy_dollar",
    "expected_total_r": "min_expected_total_r",
    "inv_dd": "min_inv_dd",
    "max_drawdown_pct": "max_dd_pct",
    "net_profit": "min_net_profit",
    "profit_factor": "min_pf",
    "trades_per_month": "min_trades_per_month",
}


def _hydrate_final_phase_runtime_context(plugin: ALCBP16Plugin, state) -> int:
    final_phase = max(state.completed_phases) if state.completed_phases else plugin.num_phases
    # Warm replay data first because the initial source-fingerprint load clears
    # transient runtime context caches, including phase diagnostics metadata.
    plugin._replay_bundle()
    phase_result = state.phase_results.get(final_phase, {})
    phase_gate = state.phase_gate_results.get(final_phase, {})
    plugin._phase_runtime_context[final_phase] = {
        "base_metrics": dict(phase_result.get("final_metrics", {})),
        "hard_rejects": {
            FLOOR_NAME_MAP.get(criterion["name"], criterion["name"]): criterion["target"]
            for criterion in phase_gate.get("criteria", [])
            if isinstance(criterion, dict) and "name" in criterion and "target" in criterion
        },
    }
    return final_phase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate the ALCB P16 final diagnostics from saved phase state.")
    parser.add_argument("--phase-state", default=str(DEFAULT_PHASE_STATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--data-dir", default="backtests/stock/data/raw")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    phase_state_path = Path(args.phase_state)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_phase_state(phase_state_path)
    plugin = ALCBP16Plugin(
        Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=float(args.equity),
        max_workers=max(1, int(args.max_workers)),
    )
    _hydrate_final_phase_runtime_context(plugin, state)
    diagnostics_text = plugin.render_final_diagnostics_text(state)
    output_path.write_text(diagnostics_text, encoding="utf-8")
    print(f"Saved final diagnostics to {output_path}")


if __name__ == "__main__":
    main()
