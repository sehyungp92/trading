"""Allow running as: python -m backtests.swing"""
from backtests.swing._aliases import install; install()

from backtest.cli import main

main()
