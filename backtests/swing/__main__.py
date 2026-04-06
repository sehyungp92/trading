"""Allow running as: python -m research.backtests.swing"""
from research.backtests.swing._aliases import install; install()

from backtest.cli import main

main()
