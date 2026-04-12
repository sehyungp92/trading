"""Allow running as: python -m backtests.momentum"""
from backtests.momentum._aliases import install; install()

from backtest.cli import main

main()
