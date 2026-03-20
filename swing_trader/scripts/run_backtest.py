#!/usr/bin/env python
"""Execute S4 backtest sweep with all 41 variants."""
import sys
import os
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent
os.chdir(str(project_root))
sys.path.insert(0, str(project_root))

# Now run the backtest
if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--run-all",
        "--save-best",
        "backtest/output"
    ]
    from backtest.run_s4 import main
    main()
