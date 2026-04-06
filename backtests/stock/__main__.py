"""Entry point for ``python -m research.backtests.stock``."""
import multiprocessing

multiprocessing.freeze_support()

from research.backtests.stock._aliases import install

install()

from research.backtests.stock.cli import main  # noqa: E402

main()
