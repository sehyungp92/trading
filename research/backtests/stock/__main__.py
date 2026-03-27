"""Entry point for ``python -m research.backtests.stock``."""
from research.backtests.stock._aliases import install

install()

from research.backtests.stock.cli import main  # noqa: E402

main()
