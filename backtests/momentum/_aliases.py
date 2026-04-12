"""Momentum family import aliases for backtest compatibility."""
from __future__ import annotations

ALIASES: dict[str, str] = {
    "strategy": "strategies.momentum.helix_v40",
    "strategy_2": "strategies.momentum.nqdtc",
    "strategy_3": "strategies.momentum.vdub",
    "backtest": "backtests.momentum",
}


def install() -> None:
    """Install the momentum alias redirector."""
    from backtests._import_hook import install as _install
    _install(ALIASES)
