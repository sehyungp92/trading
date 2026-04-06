"""Stock family import aliases for backtest compatibility."""
from __future__ import annotations

ALIASES: dict[str, str] = {
    "strategy": "strategies.stock.alcb",
    "strategy_2": "strategies.stock.iaric",
    "backtest": "research.backtests.stock",
}


def install() -> None:
    """Install the stock alias redirector."""
    from research.backtests._import_hook import install as _install
    _install(ALIASES)
