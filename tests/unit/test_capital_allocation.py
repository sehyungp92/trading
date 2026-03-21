from pathlib import Path

from libs.config.capital_allocation import resolve_strategy_capital_allocation
from libs.config.loader import load_portfolio_config, load_strategy_registry


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_family_default_split_for_swing_strategy() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    portfolio = load_portfolio_config(CONFIG_DIR)

    allocation = resolve_strategy_capital_allocation("ATRSS", 100_000.0, registry, portfolio)

    assert allocation.family == "swing"
    assert allocation.family_fraction == 0.3334
    assert allocation.strategy_fraction_within_family == 0.20
    assert allocation.allocated_nav == 6_668.0


def test_explicit_stock_strategy_allocation_is_used() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    portfolio = load_portfolio_config(CONFIG_DIR)

    allocation = resolve_strategy_capital_allocation("IARIC_v1", 100_000.0, registry, portfolio)

    assert allocation.family == "stock"
    assert allocation.family_fraction == 0.3333
    assert allocation.strategy_fraction_within_family == 0.34
    assert allocation.allocated_nav == 11_332.2

