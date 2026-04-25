from pathlib import Path

import pytest

from libs.config.capital_allocation import resolve_strategy_capital_allocation
from libs.config.loader import load_portfolio_config, load_strategy_registry


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_family_default_split_for_swing_strategy() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    portfolio = load_portfolio_config(CONFIG_DIR)

    allocation = resolve_strategy_capital_allocation("ATRSS", 100_000.0, registry, portfolio)

    assert allocation.family == "swing"
    assert allocation.family_fraction == 0.3334
    # Swing has 4 enabled strategies, so the fallback split is 1/4.
    assert allocation.strategy_fraction_within_family == pytest.approx(1 / 4, rel=1e-6)
    assert allocation.allocated_nav == pytest.approx(100_000.0 * 0.3334 / 4, rel=1e-6)


def test_explicit_stock_strategy_allocations_are_used() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    portfolio = load_portfolio_config(CONFIG_DIR)

    iaric = resolve_strategy_capital_allocation("IARIC_v1", 100_000.0, registry, portfolio)
    alcb = resolve_strategy_capital_allocation("ALCB_v1", 100_000.0, registry, portfolio)

    assert iaric.family == "stock"
    assert iaric.family_fraction == 0.3333
    assert iaric.strategy_fraction_within_family == 0.55
    assert iaric.allocated_nav == pytest.approx(100_000.0 * 0.3333 * 0.55, rel=1e-6)

    assert alcb.family == "stock"
    assert alcb.family_fraction == 0.3333
    assert alcb.strategy_fraction_within_family == 0.45
    assert alcb.allocated_nav == pytest.approx(100_000.0 * 0.3333 * 0.45, rel=1e-6)
