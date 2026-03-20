#!/usr/bin/env python
"""Test script to run S4 backtest sweep."""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run
from backtest.run_s4 import main

if __name__ == "__main__":
    main()
