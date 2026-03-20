#!/usr/bin/env python
"""Execute backtest and print all output in real-time."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "run_backtest.py"],
    cwd="/c/Users/sehyu/Documents/Other/Projects/swing_trader",
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
