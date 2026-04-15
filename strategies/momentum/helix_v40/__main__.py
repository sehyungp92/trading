"""Allow running as: python -m strategies.momentum.helix_v40"""
from strategies.momentum.helix_v40.main import _setup_logging, main
import asyncio

_setup_logging()
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
