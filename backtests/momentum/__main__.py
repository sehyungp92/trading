"""Allow running as: python -m research.backtests.momentum"""
from research.backtests.momentum._aliases import install; install()

from backtest.cli import main

main()
