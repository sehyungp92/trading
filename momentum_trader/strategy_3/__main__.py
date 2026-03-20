"""Allow running as: python -m strategy_3"""
from strategy_3.main import _setup_logging, main
import asyncio

_setup_logging()
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
