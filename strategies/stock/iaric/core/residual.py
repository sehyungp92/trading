"""Causal cross-sectional residuals shared by IARIC live and replay.

Every value is formed from completed bars carrying the same close timestamp.
The stock itself is excluded from its sector and market reference medians so a
large move cannot manufacture its own benchmark.  Sparse sectors fall back to
the broader market reference rather than emitting a misleading zero residual.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, nan
from typing import Mapping, Sequence

from strategies.stock.iaric.models import Bar


@dataclass(frozen=True, slots=True)
class FrozenResidualModel:
    """Entry-time factor contract used for both selection and management.

    ``factor_names`` and ``factor_betas`` are deliberately persisted rather
    than inferred again while a position is open.  A correlated-peer factor
    additionally freezes the exact peer symbols that formed the factor.  This
    makes a residual anchor an immutable economic target instead of a target
    that drifts when rolling betas or peer membership are refreshed.
    """

    factor_names: tuple[str, ...]
    intercept: float
    factor_betas: tuple[float, ...]
    peer_symbols: tuple[str, ...] = ()
    estimation_session: object | None = None
    contract_version: str = "frozen_residual_model_v2"

    def __post_init__(self) -> None:
        if len(self.factor_names) != len(self.factor_betas):
            raise ValueError("frozen residual factor names and betas must align")
        if not isfinite(float(self.intercept)):
            raise ValueError("frozen residual intercept must be finite")
        if not all(isfinite(float(value)) for value in self.factor_betas):
            raise ValueError("frozen residual betas must be finite")

    def expected_return(self, factors: Mapping[str, float]) -> float:
        values = [float(factors.get(name, nan)) for name in self.factor_names]
        if not all(isfinite(value) for value in values):
            return nan
        return float(self.intercept) + sum(
            float(beta) * value for beta, value in zip(self.factor_betas, values)
        )

    def residual_return(
        self,
        stock_return: float,
        factors: Mapping[str, float],
    ) -> float:
        expected = self.expected_return(factors)
        stock_value = float(stock_return)
        if not isfinite(stock_value) or not isfinite(expected):
            return nan
        return stock_value - expected


def _solve_linear_system(matrix: list[list[float]], target: list[float]) -> list[float]:
    """Solve a small dense system with deterministic partial pivoting."""

    size = len(target)
    augmented = [list(matrix[row]) + [float(target[row])] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            raise ValueError("factor design is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == 0.0:
                continue
            augmented[row] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def causal_rolling_factor_contracts(
    stock_returns: Mapping[object, float],
    factor_returns: Mapping[object, Mapping[str, float]],
    *,
    factor_names: Sequence[str] | None = None,
    window: int = 120,
    min_observations: int = 60,
    ridge: float = 1e-6,
) -> tuple[dict[object, float], dict[object, FrozenResidualModel]]:
    """Return causal residuals and the model actually used for each session.

    The return for session ``t`` is predicted using coefficients fit only on
    sessions strictly before ``t``.  The intercept is not ridge-penalised;
    factor coefficients receive a small fixed penalty to make sparse sector
    and style designs fail stably rather than manufacture extreme betas.
    Missing or insufficient histories emit NaN and therefore fail closed.
    """

    if window <= 0 or min_observations <= 0 or min_observations > window:
        raise ValueError("factor window must be positive and cover min_observations")
    if factor_names is None:
        names = sorted({name for row in factor_returns.values() for name in row})
    else:
        names = [str(name) for name in factor_names]
    if not names:
        raise ValueError("at least one factor is required")
    dates = sorted(set(stock_returns) & set(factor_returns))
    result: dict[object, float] = {session: nan for session in dates}
    contracts: dict[object, FrozenResidualModel] = {}
    history: list[tuple[list[float], float]] = []
    width = len(names) + 1
    penalty = max(float(ridge), 0.0)
    for session in dates:
        stock_value = float(stock_returns[session])
        factor_row = factor_returns[session]
        current = [1.0] + [float(factor_row.get(name, nan)) for name in names]
        current_valid = isfinite(stock_value) and all(isfinite(value) for value in current)
        training = history[-int(window):]
        if current_valid and len(training) >= int(min_observations):
            gram = [[0.0] * width for _ in range(width)]
            rhs = [0.0] * width
            for vector, response in training:
                for row in range(width):
                    rhs[row] += vector[row] * response
                    for column in range(width):
                        gram[row][column] += vector[row] * vector[column]
            for index in range(1, width):
                gram[index][index] += penalty
            try:
                coefficients = _solve_linear_system(gram, rhs)
            except ValueError:
                coefficients = []
            if coefficients:
                prediction = sum(value * beta for value, beta in zip(current, coefficients))
                result[session] = stock_value - prediction
                contracts[session] = FrozenResidualModel(
                    factor_names=tuple(names),
                    intercept=float(coefficients[0]),
                    factor_betas=tuple(float(value) for value in coefficients[1:]),
                    estimation_session=session,
                )
        if current_valid:
            history.append((current, stock_value))
    return result, contracts


def causal_rolling_factor_residuals(
    stock_returns: Mapping[object, float],
    factor_returns: Mapping[object, Mapping[str, float]],
    *,
    factor_names: Sequence[str] | None = None,
    window: int = 120,
    min_observations: int = 60,
    ridge: float = 1e-6,
) -> dict[object, float]:
    """Compatibility wrapper returning only point-in-time residual values."""

    residuals, _contracts = causal_rolling_factor_contracts(
        stock_returns,
        factor_returns,
        factor_names=factor_names,
        window=window,
        min_observations=min_observations,
        ridge=ridge,
    )
    return residuals


def align_values_by_date(
    target_dates: Sequence[object],
    source_dates: Sequence[object],
    source_values: Sequence[float],
) -> list[float]:
    """Align a causal reference series by explicit session label.

    Positional truncation silently pairs different trading sessions whenever
    one security has a missing history row.  Missing reference sessions remain
    NaN so downstream indicators fail closed instead of borrowing an adjacent
    day's benchmark value.
    """

    if len(source_dates) != len(source_values):
        raise ValueError("source dates and values must have identical lengths")
    lookup = {
        session: float(value)
        for session, value in zip(source_dates, source_values)
    }
    return [lookup.get(session, nan) for session in target_dates]


def _median_sorted(values: Sequence[float]) -> float:
    size = len(values)
    if size <= 0:
        return 0.0
    middle = size // 2
    if size % 2:
        return float(values[middle])
    return (float(values[middle - 1]) + float(values[middle])) / 2.0


def _median_without_index(values: Sequence[float], removed_index: int) -> float:
    """Return the exact median after removing one item from sorted values."""

    size = len(values)
    if size <= 1:
        return _median_sorted(values)
    remaining = size - 1

    def original_index(filtered_index: int) -> int:
        return filtered_index if filtered_index < removed_index else filtered_index + 1

    middle = remaining // 2
    if remaining % 2:
        return float(values[original_index(middle)])
    lower = float(values[original_index(middle - 1)])
    upper = float(values[original_index(middle)])
    return (lower + upper) / 2.0


def causal_relative_dislocation_atr(
    bars_by_symbol: Mapping[str, Sequence[Bar]],
    sector_by_symbol: Mapping[str, str],
    daily_atr_by_symbol: Mapping[str, float],
    *,
    sector_weight: float = 0.65,
    min_sector_peers: int = 2,
) -> dict[str, list[float]]:
    """Return point-in-time stock-minus-sector/market moves in daily ATR units.

    The computation may be run over a complete historical day: element ``i``
    still uses only the completed bar at element ``i`` and peers with the same
    completion timestamp.  It therefore has the same information set as a
    synchronized live completed-bar batch.
    """

    weight = min(max(float(sector_weight), 0.0), 1.0)
    peers_required = max(int(min_sector_peers), 1)
    result = {str(symbol): [0.0] * len(bars) for symbol, bars in bars_by_symbol.items()}
    observations: dict[object, list[tuple[str, int, float, str]]] = defaultdict(list)

    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        first_open = float(bars[0].open)
        if first_open <= 0:
            continue
        sector = str(sector_by_symbol.get(symbol, "") or "UNKNOWN").upper()
        for index, bar in enumerate(bars):
            move = (float(bar.close) - first_open) / first_open
            observations[bar.end_time].append((str(symbol), index, move, sector))

    for rows in observations.values():
        # Pre-sort each synchronized cross-section once.  Exact leave-one-out
        # medians are then O(1) per symbol rather than rebuilding and sorting a
        # peer list for every symbol.  This matters in both full replay and the
        # live completed-bar batch without changing the information set.
        ordered_market = sorted(
            ((move, symbol) for symbol, _index, move, _sector in rows),
            key=lambda item: (item[0], item[1]),
        )
        market_values = [move for move, _symbol in ordered_market]
        market_rank = {symbol: rank for rank, (_move, symbol) in enumerate(ordered_market)}
        sector_moves: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for symbol, _index, move, sector in rows:
            sector_moves[sector].append((move, symbol))
        sector_values: dict[str, list[float]] = {}
        sector_ranks: dict[str, dict[str, int]] = {}
        for sector, values in sector_moves.items():
            ordered = sorted(values, key=lambda item: (item[0], item[1]))
            sector_values[sector] = [move for move, _symbol in ordered]
            sector_ranks[sector] = {
                peer_symbol: rank
                for rank, (_move, peer_symbol) in enumerate(ordered)
            }

        for symbol, index, stock_move, sector in rows:
            market_move = _median_without_index(
                market_values,
                market_rank[symbol],
            )
            same_sector_values = sector_values[sector]
            same_sector_count = len(same_sector_values) - 1
            if same_sector_count >= peers_required:
                sector_move = _median_without_index(
                    same_sector_values,
                    sector_ranks[sector][symbol],
                )
                reference_move = weight * sector_move + (1.0 - weight) * market_move
            else:
                reference_move = market_move

            bars = bars_by_symbol[symbol]
            first_open = float(bars[0].open)
            daily_atr = float(daily_atr_by_symbol.get(symbol, 0.0) or 0.0)
            if daily_atr > 0:
                result[symbol][index] = (stock_move - reference_move) * first_open / daily_atr

    return result
