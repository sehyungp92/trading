from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median

import pytest

from strategies.stock.iaric.core.residual import (
    _median_without_index,
    align_values_by_date,
    causal_rolling_factor_residuals,
    causal_rolling_factor_contracts,
    causal_relative_dislocation_atr,
)
from strategies.stock.iaric.models import Bar


def _bars(symbol: str, closes: list[float]) -> list[Bar]:
    start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    return [
        Bar(
            symbol=symbol,
            start_time=start + timedelta(minutes=5 * index),
            end_time=start + timedelta(minutes=5 * (index + 1)),
            open=100.0 if index == 0 else closes[index - 1],
            high=max(100.0, close),
            low=min(100.0, close),
            close=close,
            volume=1_000.0,
        )
        for index, close in enumerate(closes)
    ]


def test_residual_is_causal_cross_sectional_and_excludes_the_stock_itself() -> None:
    bars = {
        "AAA": _bars("AAA", [98.0, 99.0]),
        "BBB": _bars("BBB", [100.0, 100.5]),
        "CCC": _bars("CCC", [100.0, 101.0]),
        "DDD": _bars("DDD", [100.0, 100.0]),
    }
    result = causal_relative_dislocation_atr(
        bars,
        {symbol: "TECH" for symbol in bars},
        {symbol: 2.0 for symbol in bars},
    )
    assert result["AAA"][0] < -0.9
    assert result["AAA"][1] > result["AAA"][0]

    extended = dict(bars)
    extended["AAA"] = bars["AAA"] + _bars("AAA", [50.0])[0:1]
    # Appending later information cannot alter the already completed values.
    repeated = causal_relative_dislocation_atr(
        extended,
        {symbol: "TECH" for symbol in extended},
        {symbol: 2.0 for symbol in extended},
    )
    assert repeated["AAA"][:2] == pytest.approx(result["AAA"])


def test_fast_leave_one_out_median_matches_literal_peer_median() -> None:
    for size in range(1, 13):
        values = sorted(float(((index * 7) % 5) - 2) for index in range(size))
        for removed_index in range(size):
            peers = values[:removed_index] + values[removed_index + 1 :]
            expected = median(peers) if peers else median(values)
            assert _median_without_index(values, removed_index) == pytest.approx(expected)


def test_reference_values_align_by_session_instead_of_positional_index() -> None:
    target = ["2026-01-02", "2026-01-05", "2026-01-06"]
    source = ["2026-01-02", "2026-01-06"]
    aligned = align_values_by_date(target, source, [100.0, 102.0])
    assert aligned[0] == 100.0
    assert aligned[1] != aligned[1]  # missing session remains NaN
    assert aligned[2] == 102.0


def test_factor_residual_uses_only_strictly_prior_sessions() -> None:
    sessions = [f"2026-01-{day:02d}" for day in range(1, 13)]
    factors = {
        session: {
            "market": 0.001 * index,
            "sector": (-1.0 if index % 2 else 1.0) * 0.002,
        }
        for index, session in enumerate(sessions)
    }
    stock = {
        session: 0.0005 + 1.4 * row["market"] + 0.6 * row["sector"]
        for session, row in factors.items()
    }
    stock[sessions[8]] += 0.025
    baseline = causal_rolling_factor_residuals(
        stock,
        factors,
        factor_names=("market", "sector"),
        window=8,
        min_observations=5,
    )

    revised = dict(stock)
    revised[sessions[-1]] = -0.90
    repeated = causal_rolling_factor_residuals(
        revised,
        factors,
        factor_names=("market", "sector"),
        window=8,
        min_observations=5,
    )

    # Future/current outcomes cannot alter any earlier residual. The injected
    # idiosyncratic move remains visible after removing the causal factor fit.
    assert [repeated[key] for key in sessions[:-1]] == pytest.approx(
        [baseline[key] for key in sessions[:-1]], nan_ok=True
    )
    assert baseline[sessions[8]] > 0.02
    assert baseline[sessions[3]] != baseline[sessions[3]]


def test_factor_residual_fails_closed_on_missing_current_factor() -> None:
    sessions = list(range(7))
    factors = {session: {"market": float(session) / 100.0} for session in sessions}
    stock = {session: 0.001 + 0.8 * factors[session]["market"] for session in sessions}
    factors[6] = {}
    result = causal_rolling_factor_residuals(
        stock,
        factors,
        factor_names=("market",),
        window=6,
        min_observations=4,
    )
    assert result[6] != result[6]


def test_factor_contract_freezes_the_entry_time_coefficients() -> None:
    sessions = list(range(10))
    factors = {session: {"market": session / 100.0} for session in sessions}
    stock = {
        session: 0.002 + 1.25 * factors[session]["market"]
        for session in sessions
    }
    residuals, contracts = causal_rolling_factor_contracts(
        stock,
        factors,
        factor_names=("market",),
        window=8,
        min_observations=5,
    )
    model = contracts[8]
    assert model.contract_version == "frozen_residual_model_v2"
    assert model.factor_names == ("market",)
    assert model.residual_return(stock[8], factors[8]) == pytest.approx(
        residuals[8]
    )
    # Later observations do not mutate the contract stored for entry session 8.
    revised = dict(stock)
    revised[9] = -1.0
    _revised_residuals, revised_contracts = causal_rolling_factor_contracts(
        revised,
        factors,
        factor_names=("market",),
        window=8,
        min_observations=5,
    )
    assert revised_contracts[8] == model
