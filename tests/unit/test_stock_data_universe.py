from backtests.stock.data.update_intraday import BACKTESTED_SYMBOLS
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS


def test_intraday_updater_uses_exact_canonical_98_symbol_universe() -> None:
    assert len(BACKTESTED_INTRADAY_STOCK_SYMBOLS) == 98
    assert len(set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)) == 98
    assert BACKTESTED_SYMBOLS == list(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
    assert "BRK B" in BACKTESTED_SYMBOLS
