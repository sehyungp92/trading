"""Allow running as: python -m strategy_2"""
from strategy_2.main import _setup_logging, main
import asyncio

_setup_logging()
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
