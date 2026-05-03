from __future__ import annotations

from backtests.momentum.analysis.regime_diagnostics import generate_regime_diagnostics


def generate_phase_diagnostics(
    phase: int,
    trades,
    metrics: dict[str, float],
    *,
    signal_events=None,
    equity_curve=None,
    timestamps=None,
) -> str:
    return generate_regime_diagnostics(
        list(trades or []),
        metrics,
        signal_events=list(signal_events) if signal_events is not None else [],
        equity_curve=equity_curve,
        timestamps=timestamps,
        title=f"NQ_REGIME Phase {phase} Diagnostics",
    )
