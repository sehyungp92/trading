"""Downturn Dominator v1 -- short-only regime-adaptive MNQ scalping.

Installs backtest import aliases so that ``from backtest.engine.downturn_*``
resolves to ``research.backtests.momentum.engine.downturn_*``.
"""
from research.backtests.momentum._aliases import install as _install_aliases

_install_aliases()
