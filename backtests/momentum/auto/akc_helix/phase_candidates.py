from __future__ import annotations

from backtests.momentum.auto.experiments import build_experiment_queue

BASE_MUTATIONS = {
    "flags.use_momentum_stall": True,
    "flags.use_drawdown_throttle": False,
    "flags.vol_50_80_sizing_mult": 0.85,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("ABLATION", ["total_trades", "profit_factor", "max_drawdown_pct"]),
    2: ("PARAMETER SWEEPS", ["net_profit", "calmar", "profit_factor"]),
    3: ("INTERACTIONS", ["net_profit", "calmar", "max_drawdown_pct"]),
}

_PHASE_TYPES = {
    1: {"ABLATION"},
    2: {"PARAM_SWEEP"},
    3: {"INTERACTION"},
}


def get_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    allowed_types = _PHASE_TYPES.get(phase, set())
    candidates = [
        (experiment.id, dict(experiment.mutations))
        for experiment in build_experiment_queue("helix")
        if experiment.type in allowed_types
    ]

    if suggested_experiments:
        existing = {name for name, _ in candidates}
        for name, mutations in suggested_experiments:
            if name not in existing:
                candidates.append((name, dict(mutations)))
                existing.add(name)
    return candidates
