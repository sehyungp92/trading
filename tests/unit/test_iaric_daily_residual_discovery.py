from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests.stock.auto.runners.run_iaric_daily_residual_discovery import (
    FACTOR_MODELS,
    SECTOR_ETFS,
    TRADABLE_EXECUTION_SYMBOLS,
    SCORE_SPEC,
    Candidate,
    _causal_correlated_peer_returns,
    _causal_factor_residual,
    _immutable_score_components,
    _select_candidate,
    evaluate_candidate,
    registered_candidates,
)
from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS


def _atlas_row(
    date: str,
    symbol: str,
    issuer: str,
    sector: str,
    residual_z: float,
    *,
    return_value: float = 0.01,
) -> dict[str, object]:
    formation = pd.Timestamp(date)
    entry = formation + pd.offsets.BDay(1)
    row = {
        "formation_date": formation,
        "symbol": symbol,
        "issuer": issuer,
        "sector": sector,
        "residual_return": -0.02,
        "residual_volatility": 0.02,
        "residual_z": residual_z,
        "residual_z_h1": residual_z,
        "residual_z_h3": residual_z,
        "residual_z_h5": residual_z,
        "residual_z_h20": residual_z,
        "adv_dollars": 1_000_000_000.0,
        "loser_percentile": 0.01,
        "residual_percentile_h1": 0.01 if residual_z < 0 else 0.99,
        "residual_percentile_h3": 0.01 if residual_z < 0 else 0.99,
        "residual_percentile_h5": 0.01 if residual_z < 0 else 0.99,
        "residual_percentile_h20": 0.01 if residual_z < 0 else 0.99,
        "normalization_room_long_h1": 0.8,
        "normalization_room_long_h3": 0.8,
        "normalization_room_long_h5": 0.8,
        "normalization_room_long_h20": 0.8,
        "normalization_room_short_h1": 0.8,
        "normalization_room_short_h3": 0.8,
        "normalization_room_short_h5": 0.8,
        "normalization_room_short_h20": 0.8,
        "daily_score_long_h1": 80.0 + abs(residual_z),
        "daily_score_long_h3": 80.0 + abs(residual_z),
        "daily_score_long_h5": 80.0 + abs(residual_z),
        "daily_score_long_h20": 80.0 + abs(residual_z),
        "daily_score_short_h1": 80.0 + abs(residual_z),
        "daily_score_short_h3": 80.0 + abs(residual_z),
        "daily_score_short_h5": 80.0 + abs(residual_z),
        "daily_score_short_h20": 80.0 + abs(residual_z),
        "suspicious_price_jump_h1": False,
        "suspicious_price_jump_h3": False,
        "suspicious_price_jump_h5": False,
        "suspicious_price_jump_h20": False,
        "entry_price_h1": 100.0,
        "exit_price_h1": 100.0 * (1.0 + return_value),
        "entry_date_h1": entry,
        "exit_date_h1": entry,
        "entry_price_h2": 100.0,
        "exit_price_h2": 100.0 * (1.0 + return_value),
        "entry_date_h2": entry,
        "exit_date_h2": entry + pd.offsets.BDay(1),
        "entry_price_h5": 100.0,
        "exit_price_h5": 100.0 * (1.0 + return_value),
        "entry_date_h5": entry,
        "exit_date_h5": entry + pd.offsets.BDay(4),
        "entry_price_h3": 100.0,
        "exit_price_h3": 100.0 * (1.0 + return_value),
        "entry_date_h3": entry,
        "exit_date_h3": entry + pd.offsets.BDay(2),
        "entry_price_h4": 100.0,
        "exit_price_h4": 100.0 * (1.0 + return_value),
        "entry_date_h4": entry,
        "exit_date_h4": entry + pd.offsets.BDay(3),
        "entry_price_h7": 100.0,
        "exit_price_h7": 100.0 * (1.0 + return_value),
        "entry_date_h7": entry,
        "exit_date_h7": entry + pd.offsets.BDay(6),
        "entry_price_h6": 100.0,
        "exit_price_h6": 100.0 * (1.0 + return_value),
        "entry_date_h6": entry,
        "exit_date_h6": entry + pd.offsets.BDay(5),
        "entry_price_h8": 100.0,
        "exit_price_h8": 100.0 * (1.0 + return_value),
        "entry_date_h8": entry,
        "exit_date_h8": entry + pd.offsets.BDay(7),
        "entry_price_h9": 100.0,
        "exit_price_h9": 100.0 * (1.0 + return_value),
        "entry_date_h9": entry,
        "exit_date_h9": entry + pd.offsets.BDay(8),
        "entry_price_h10": 100.0,
        "exit_price_h10": 100.0 * (1.0 + return_value),
        "entry_date_h10": entry,
        "exit_date_h10": entry + pd.offsets.BDay(9),
    }
    for horizon in (1, 3, 5, 20):
        for side in ("long", "short"):
            row[f"residual_extremeness_{side}_h{horizon}"] = 0.8
            row[f"shock_freshness_{side}_h{horizon}"] = 0.7
    row.update(
        {
            "price_rejection_long": 0.6,
            "price_rejection_short": 0.6,
            "volume_transition": 0.5,
            "volume_exhaustion_quality": 1.0,
            "regime_execution_quality": 0.7,
            "market_trend_z_20d": 0.0,
        }
    )
    return row


def test_registered_search_is_bounded_and_score_has_exactly_seven_components() -> None:
    candidates = registered_candidates()
    assert len(candidates) == 72
    assert len(SCORE_SPEC) == 7
    assert sum(row["weight"] for row in SCORE_SPEC.values()) == pytest.approx(1.0)
    assert {candidate.holding_sessions for candidate in candidates} == {1, 2, 3, 5, 7, 10}
    assert {candidate.formation_sessions for candidate in candidates} == {1, 3, 5, 20}
    assert {candidate.diagnostic_leg for candidate in candidates} == {
        "long_loser",
        "short_winner",
        "dollar_neutral_spread",
    }


def test_execution_universe_is_frozen_to_the_98_intraday_names() -> None:
    assert len(TRADABLE_EXECUTION_SYMBOLS) == 98
    assert {"AAPL", "GOOG", "GOOGL", "TSLA"} <= TRADABLE_EXECUTION_SYMBOLS
    assert FACTOR_MODELS == (
        "market_only",
        "market_sector",
        "market_sector_peer",
        "peer_demeaned",
    )
    assert SECTOR_ETFS["Healthcare"] == "XLV"
    assert SECTOR_ETFS["Health Care"] == "XLV"
    assert len(set(SECTOR_ETFS.values())) == 11
    sector_by_symbol = {symbol: sector for symbol, sector, _ in SP500_CONSTITUENTS}
    assert {
        sector_by_symbol[symbol] for symbol in TRADABLE_EXECUTION_SYMBOLS
    } <= set(SECTOR_ETFS)


def test_immutable_score_scaling_does_not_reward_zero_frequency() -> None:
    components = _immutable_score_components(
        {
            "net_expected_r_per_month": 0.0,
            "executable_trades_per_month": 0.0,
            "worst_fold_r_per_month": 0.0,
            "average_r_and_discrimination": 0.0,
            "downside_risk": 0.0,
            "issuer_sector_concentration": 0.5,
            "cost_and_neighbourhood_robustness": 0.0,
        }
    )
    assert components["executable_trades_per_month"] == 0.0
    assert components["net_expected_r_per_month"] == pytest.approx(0.5)
    assert components["downside_risk"] == pytest.approx(1.0)


def test_factor_fit_is_strictly_causal() -> None:
    market = np.linspace(-0.01, 0.01, 100)
    sector = np.sin(np.arange(100)) / 100.0
    stock = 0.001 + 1.2 * market + 0.5 * sector
    stock[80] += 0.04
    baseline = _causal_factor_residual(
        stock, market, sector, window=60, min_observations=40
    )
    changed = stock.copy()
    changed[99] = -0.90
    repeated = _causal_factor_residual(
        changed, market, sector, window=60, min_observations=40
    )
    assert repeated[:99] == pytest.approx(baseline[:99], nan_ok=True)
    assert baseline[80] > 0.035


def test_singleton_sector_uses_causal_cross_universe_peer_fallback() -> None:
    index = pd.bdate_range("2024-01-02", periods=90)
    base = np.sin(np.arange(90) / 7.0) / 100.0
    returns = pd.DataFrame(
        {
            "CAT": base,
            "MSFT": base * 0.95 + 0.0001,
            "AAPL": -base * 0.50,
        },
        index=index,
    )
    peers = _causal_correlated_peer_returns(
        returns,
        {"CAT": "Industrials", "MSFT": "Technology", "AAPL": "Technology"},
        min_observations=40,
        peer_count=1,
    )
    assert peers["CAT"].notna().sum() > 0
    assert peers["CAT"].iloc[-1] == pytest.approx(returns["MSFT"].iloc[-1])


def test_fast_factor_fit_matches_strict_prior_normal_equations() -> None:
    generator = np.random.default_rng(7)
    market = generator.normal(0.0, 0.01, 90)
    sector = generator.normal(0.0, 0.008, 90)
    peer = generator.normal(0.0, 0.009, 90)
    stock = 0.0005 + 1.1 * market + 0.4 * sector + 0.3 * peer
    observed = _causal_factor_residual(
        stock,
        market,
        sector,
        peer,
        window=40,
        min_observations=25,
        ridge=1e-5,
    )
    index = 70
    design = np.column_stack(
        [np.ones(40), market[index - 40:index], sector[index - 40:index], peer[index - 40:index]]
    )
    gram = design.T @ design
    gram[1:, 1:] += np.eye(3) * 1e-5
    beta = np.linalg.solve(gram, design.T @ stock[index - 40:index])
    expected = stock[index] - np.array([1.0, market[index], sector[index], peer[index]]) @ beta
    assert observed[index] == pytest.approx(expected)


def test_selection_enforces_issuer_sector_and_total_position_caps() -> None:
    atlas = pd.DataFrame(
        [
            _atlas_row("2025-01-06", "GOOG", "ALPHABET", "Technology", -3.0),
            _atlas_row("2025-01-06", "GOOGL", "ALPHABET", "Technology", -2.9),
            _atlas_row("2025-01-06", "MSFT", "MICROSOFT", "Technology", -2.8),
            _atlas_row("2025-01-06", "JPM", "JPM", "Financials", -2.7),
        ]
    )
    candidate = Candidate("caps", 1.0, 1, 2, 1)
    selected = _select_candidate(atlas, candidate)
    assert len(selected) == 2
    assert selected["issuer"].nunique() == 2
    assert selected.groupby("sector").size().max() == 1


def test_short_and_hedged_legs_have_explicit_direction_and_balanced_weight() -> None:
    atlas = pd.DataFrame(
        [
            _atlas_row("2025-01-06", "LOSER", "LOSER", "Technology", -2.0, return_value=0.02),
            _atlas_row("2025-01-06", "WINNER", "WINNER", "Financials", 2.0, return_value=-0.02),
        ]
    )
    short = _select_candidate(
        atlas,
        Candidate("short", 1.0, 1, 10, 2, diagnostic_leg="short_winner"),
    )
    hedged = _select_candidate(
        atlas,
        Candidate("spread", 1.0, 1, 10, 2, diagnostic_leg="dollar_neutral_spread"),
    )
    assert short["trade_side"].tolist() == ["short"]
    assert short["r"].iloc[0] > 0.0
    assert set(hedged["trade_side"]) == {"long", "short"}
    assert set(hedged["leg_weight"]) == {0.5}


def test_locked_rows_cannot_change_candidate_score_or_metrics() -> None:
    rows = []
    for fold_start in (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-01-02")):
        for index in range(110):
            date = fold_start + pd.offsets.BDay(index)
            rows.append(
                _atlas_row(
                    date.strftime("%Y-%m-%d"),
                    f"S{index % 20:02d}",
                    f"I{index % 20:02d}",
                    f"Sector{index % 5}",
                    -2.0 - index / 1000.0,
                    return_value=0.02,
                )
            )
    atlas = pd.DataFrame(rows)
    candidate = Candidate("causal", 1.0, 1, 20, 4)
    baseline = evaluate_candidate(atlas, candidate)
    locked = pd.concat(
        [
            atlas,
            pd.DataFrame(
                [
                    _atlas_row(
                        "2025-10-01",
                        "LEAK",
                        "LEAK",
                        "LeakSector",
                        -10.0,
                        return_value=10.0,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    repeated = evaluate_candidate(locked, candidate)
    assert repeated["score"] == pytest.approx(baseline["score"])
    assert repeated["metrics"] == pytest.approx(baseline["metrics"])


def test_positive_but_economically_weak_calibration_does_not_qualify() -> None:
    rows = []
    for fold_start, return_value in (
        (pd.Timestamp("2024-04-01"), 0.02),
        (pd.Timestamp("2025-01-02"), 0.0025),
    ):
        for index in range(110):
            date = fold_start + pd.offsets.BDay(index)
            rows.append(
                _atlas_row(
                    date.strftime("%Y-%m-%d"),
                    f"S{index % 20:02d}",
                    f"I{index % 20:02d}",
                    f"Sector{index % 5}",
                    -2.0,
                    return_value=return_value,
                )
            )
    result = evaluate_candidate(
        pd.DataFrame(rows), Candidate("weak", 1.0, 1, 20, 4)
    )
    assert result["fold_metrics"]["calibration"]["total_r"] > 0.0
    assert result["gates"]["calibration_avg_r_gte_0p07"] is False
    assert result["qualified_discovery_candidate"] is False
