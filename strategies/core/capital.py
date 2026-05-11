"""Runtime capital helpers for legacy strategy plugins."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from libs.config.capital_allocation import resolve_strategy_capital_allocation
from libs.oms.persistence.db_config import get_environment


def resolve_plugin_nav(ctx: Any, strategy_id: str) -> float:
    """Resolve NAV for standalone plugin adapters without live account fallbacks."""
    portfolio = getattr(ctx, "portfolio", None)
    explicit = getattr(portfolio, "allocation", None)
    if explicit is not None:
        return _positive_nav(explicit, "portfolio.allocation")

    capital = getattr(portfolio, "capital", None)
    strategy_navs = getattr(capital, "strategy_navs", None)
    if isinstance(strategy_navs, Mapping) and strategy_id in strategy_navs:
        return _positive_nav(strategy_navs[strategy_id], f"strategy_navs[{strategy_id!r}]")

    if get_environment() != "live":
        paper_nav = _paper_allocated_nav(ctx, strategy_id)
        if paper_nav is not None:
            return paper_nav

    raise RuntimeError(
        f"{strategy_id}: no explicit runtime NAV supplied. Live sizing must use "
        "the family coordinator with broker NetLiquidation; standalone plugins "
        "require ctx.portfolio.allocation or ctx.portfolio.capital.strategy_navs."
    )


def _paper_allocated_nav(ctx: Any, strategy_id: str) -> float | None:
    portfolio = getattr(ctx, "portfolio", None)
    capital = getattr(portfolio, "capital", None)
    paper_initial_equity = getattr(capital, "paper_initial_equity", None)
    registry = getattr(ctx, "registry", None)
    if paper_initial_equity is None or registry is None:
        return None
    if strategy_id not in getattr(registry, "strategies", {}):
        return None
    allocation = resolve_strategy_capital_allocation(
        strategy_id,
        raw_nav=float(paper_initial_equity),
        registry=registry,
        portfolio=portfolio,
    )
    return _positive_nav(allocation.allocated_nav, f"paper allocation for {strategy_id}")


def _positive_nav(value: Any, label: str) -> float:
    nav = float(value)
    if not math.isfinite(nav) or nav <= 0:
        raise RuntimeError(f"{label} must be a positive finite NAV; got {value!r}")
    return nav
