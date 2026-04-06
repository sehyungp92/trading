"""Swing family import aliases for backtest compatibility."""
from __future__ import annotations

ALIASES: dict[str, str] = {
    "strategy": "strategies.swing.atrss",
    "strategy_2": "strategies.swing.akc_helix",
    "strategy_3": "strategies.swing.breakout",
    "strategy_4": "strategies.swing.keltner",
    "backtest": "research.backtests.swing",
}


def install() -> None:
    """Install the swing alias redirector."""
    from research.backtests._import_hook import install as _install
    _install(ALIASES)
