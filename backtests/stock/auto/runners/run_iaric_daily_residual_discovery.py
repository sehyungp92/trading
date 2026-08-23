"""Frozen-universe price/volume-only residual-reversion discovery for IARIC.

This selection-only runner tests causal market, sector and correlated-peer
residual normalization.  It never reads news or historical quotes and never
treats their absence as a veto.  The phased runner treats the project-designated
local stock Parquets as official once bounded content integrity, source
fingerprinting, completed-session semantics and live/replay parity pass; broker
acquisition receipts are optional provenance rather than an alpha gate.

Only the registered discovery and calibration folds are loaded.  The locked
internal-validation interval and sealed holdout are neither read nor scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from backtests.stock.auto.iaric.representative_contract import (
    CALIBRATION_END,
    CALIBRATION_START,
    DISCOVERY_END,
    DISCOVERY_START,
    HOLDOUT_START,
    LOCKED_VALIDATION_END,
)
from backtests.stock.data.bundle import FrozenBundleResolver
from backtests.stock.data.calendar import RTH_SESSION_POLICY
from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS
from strategies.stock.iaric.core.daily_residual import (
    DAILY_RESIDUAL_SCORE_WEIGHTS,
    DailyResidualFeatures,
    DailyResidualOpportunity,
    rank_daily_residual_opportunities,
)
from strategies.stock.iaric.core.lanes import issuer_key
from strategies.stock.iaric.daily_residual_selection import SECTOR_REFERENCE
from strategies.stock.volume_units import IBKR_SHARE_VOLUME_MULTIPLIER


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "backtests/stock/data/raw"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "backtests/output/stock/iaric/round_4/daily_residual_discovery_v3"
)
WARMUP_START = "2023-06-01"
FACTOR_MODELS: tuple[str, ...] = (
    "market_only",
    "market_sector",
    "market_sector_peer",
    "peer_demeaned",
)
TRADABLE_EXECUTION_SYMBOLS = frozenset(BACKTESTED_INTRADAY_STOCK_SYMBOLS)

# One shared live/replay spelling map.  The retained constituent metadata uses
# ``Healthcare`` while some live-universe sources use ``Health Care``; both
# must resolve to XLV or an entire sector silently disappears from discovery.
SECTOR_ETFS: dict[str, str] = dict(SECTOR_REFERENCE)

FOLDS: tuple[tuple[str, str, str], ...] = (
    ("discovery", DISCOVERY_START, DISCOVERY_END),
    ("calibration", CALIBRATION_START, CALIBRATION_END),
)

SCORE_SPEC: dict[str, dict[str, float]] = {
    "net_expected_r_per_month": {"weight": 0.26, "scale": 2.0},
    "executable_trades_per_month": {"weight": 0.18, "scale": 20.0},
    "worst_fold_r_per_month": {"weight": 0.18, "scale": 1.0},
    "average_r_and_discrimination": {"weight": 0.14, "scale": 0.10},
    "downside_risk": {"weight": 0.10, "scale": 1.0},
    "issuer_sector_concentration": {"weight": 0.07, "scale": 1.0},
    "cost_and_neighbourhood_robustness": {"weight": 0.07, "scale": 1.0},
}

if len(SCORE_SPEC) != 7 or not math.isclose(
    sum(item["weight"] for item in SCORE_SPEC.values()), 1.0
):
    raise RuntimeError("daily residual discovery score must have seven fixed components")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    residual_z_floor: float
    holding_sessions: int
    max_positions: int
    max_positions_per_sector: int
    round_trip_cost_bps: float = 20.0
    formation_sessions: int = 1
    diagnostic_leg: str = "long_loser"
    factor_model: str = "market_sector_peer"
    score_components: tuple[str, ...] = ()
    lane_id: str = "daily_residual_generic"
    minimum_failed_continuation_r: float = 0.0
    minimum_sector_return_5d: float = -0.15
    minimum_market_trend_z_20d: float = -8.0
    minimum_score: float = 0.0
    catastrophic_stop_residual_r: float = 4.0
    ranking_score_components: tuple[str, ...] = ()


def registered_candidates() -> list[Candidate]:
    """Return a bounded economic grid, not an outcome-fitted search surface."""

    rows: list[Candidate] = []
    # Capacity and the economic residual floor are frozen.  Frequency comes
    # from orthogonal formation horizons and overlapping positions, not from a
    # ladder of progressively weaker score floors.  Twenty sessions is an
    # explicitly labelled monthly-reversal control rather than a primary lane.
    for formation in (1, 3, 5, 20):
        for leg in ("long_loser", "short_winner", "dollar_neutral_spread"):
            for holding in (1, 2, 3, 5, 7, 10):
                candidate_id = f"f{formation}_{leg}_z1p0_h{holding}_p10_s2_c20"
                rows.append(
                    Candidate(
                        candidate_id,
                        1.0,
                        holding,
                        10,
                        2,
                        formation_sessions=formation,
                        diagnostic_leg=leg,
                    )
                )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _selection_data_fingerprint(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    paths: Iterable[Path],
) -> tuple[str, list[dict[str, Any]]]:
    """Fingerprint only the bounded selection view, never later outcomes."""

    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    ordered_paths = list(paths)
    path_by_symbol = {path.stem[:-3]: path for path in ordered_paths}
    if set(path_by_symbol) != set(close.columns):
        if len(ordered_paths) != len(close.columns):
            raise ValueError(
                "daily panel paths cannot be mapped one-to-one to panel symbols"
            )
        # Content-addressed authority objects are all named ``bars.parquet``;
        # the authoritative loader returns them in exact panel-column order.
        path_by_symbol = dict(zip(close.columns, ordered_paths))
    for symbol in sorted(close.columns):
        frame = pd.DataFrame(
            {
                "open": open_[symbol],
                "high": high[symbol],
                "low": low[symbol],
                "close": close[symbol],
                "volume": volume[symbol],
            }
        ).dropna(how="all")
        hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
        relative = path_by_symbol[symbol].resolve().relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(hashed.tobytes())
        rows.append(
            {
                "path": relative,
                "selection_rows": len(frame),
                "selection_start": str(frame.index.min().date()) if not frame.empty else None,
                "selection_end": str(frame.index.max().date()) if not frame.empty else None,
            }
        )
    return digest.hexdigest(), rows


def _load_daily_panel(
    data_dir: Path,
    *,
    selection_end: str = CALIBRATION_END,
    allow_locked_validation: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    list[Path],
]:
    if selection_end >= HOLDOUT_START:
        raise ValueError("daily panel loader may never expose the sealed holdout")
    if selection_end > CALIBRATION_END and not allow_locked_validation:
        raise ValueError(
            "locked-validation rows require the explicit one-shot loader contract"
        )
    if selection_end > LOCKED_VALIDATION_END:
        raise ValueError("daily panel end exceeds the registered locked validation")
    metadata = {symbol: sector for symbol, sector, _exchange in SP500_CONSTITUENTS}
    required_references = {"SPY", *SECTOR_ETFS.values()}
    available = {path.stem[:-3]: path for path in data_dir.glob("*_1d.parquet")}
    symbols = sorted(set(metadata) & set(available) & TRADABLE_EXECUTION_SYMBOLS)
    missing_references = sorted(required_references - set(available))
    if missing_references:
        raise RuntimeError("missing daily factor files: " + ", ".join(missing_references))
    if set(symbols) != TRADABLE_EXECUTION_SYMBOLS:
        missing = sorted(TRADABLE_EXECUTION_SYMBOLS - set(symbols))
        raise RuntimeError(
            "residual discovery requires the frozen 98-name execution universe; "
            f"missing {missing}"
        )

    paths = [available[symbol] for symbol in [*symbols, *sorted(required_references)]]
    close_series: dict[str, pd.Series] = {}
    open_series: dict[str, pd.Series] = {}
    high_series: dict[str, pd.Series] = {}
    low_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}
    for symbol in [*symbols, *sorted(required_references)]:
        # Predicate pushdown ensures post-calibration rows are not materialized
        # merely because the legacy Parquet also contains later observations.
        frame = pd.read_parquet(
            available[symbol],
            columns=["open", "high", "low", "close", "volume"],
            filters=[
                ("time", ">=", pd.Timestamp(WARMUP_START, tz="UTC")),
                ("time", "<=", pd.Timestamp(selection_end + " 23:59:59", tz="UTC")),
            ],
        )
        index = pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)
        frame = frame.set_axis(index).sort_index()
        frame = frame.loc[WARMUP_START:selection_end]
        close_series[symbol] = frame["close"].astype(float)
        open_series[symbol] = frame["open"].astype(float)
        high_series[symbol] = frame["high"].astype(float)
        low_series[symbol] = frame["low"].astype(float)
        volume_series[symbol] = frame["volume"].astype(float)
    close = pd.DataFrame(close_series).sort_index()
    open_ = pd.DataFrame(open_series).reindex(close.index)
    high = pd.DataFrame(high_series).reindex(close.index)
    low = pd.DataFrame(low_series).reindex(close.index)
    volume = pd.DataFrame(volume_series).reindex(close.index)
    return (
        close,
        open_,
        high,
        low,
        volume,
        {symbol: metadata[symbol] for symbol in symbols},
        paths,
    )


def _load_daily_panel_from_authoritative_bundle(
    bundle_path: Path,
    *,
    repository_root: Path = REPO_ROOT,
    selection_end: str = CALIBRATION_END,
    allow_locked_validation: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    list[Path],
]:
    """Load the exact receipt-backed ADJUSTED_LAST daily selection view."""

    if selection_end >= HOLDOUT_START:
        raise ValueError("authoritative daily loader may never expose the sealed holdout")
    if selection_end > CALIBRATION_END and not allow_locked_validation:
        raise ValueError(
            "locked-validation rows require the explicit one-shot loader contract"
        )
    if selection_end > LOCKED_VALIDATION_END:
        raise ValueError("authoritative daily end exceeds locked validation")
    resolver = FrozenBundleResolver.load(
        bundle_path,
        repo_root=repository_root,
        require_clean=False,
        expected_universe=list(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        expected_session_policy_by_timeframe={"1d": RTH_SESSION_POLICY},
        expected_what_to_show_by_timeframe={"1d": "ADJUSTED_LAST"},
    )
    metadata = {symbol: sector for symbol, sector, _exchange in SP500_CONSTITUENTS}
    stock_symbols = sorted(TRADABLE_EXECUTION_SYMBOLS)
    references = sorted({"SPY", *SECTOR_ETFS.values()})
    requested = [*stock_symbols, *references]
    paths = [resolver.bar_path(symbol, "1d") for symbol in requested]
    series: dict[str, dict[str, pd.Series]] = {
        field: {} for field in ("open", "high", "low", "close", "volume")
    }
    for symbol, path in zip(requested, paths):
        frame = pd.read_parquet(
            path,
            columns=["open", "high", "low", "close", "volume"],
            filters=[
                ("time", ">=", pd.Timestamp(WARMUP_START, tz="UTC")),
                ("time", "<=", pd.Timestamp(selection_end + " 23:59:59", tz="UTC")),
            ],
        )
        index = pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)
        frame = frame.set_axis(index).sort_index().loc[WARMUP_START:selection_end]
        for field in series:
            series[field][symbol] = frame[field].astype(float)
    close = pd.DataFrame(series["close"]).sort_index()
    open_ = pd.DataFrame(series["open"]).reindex(close.index)
    high = pd.DataFrame(series["high"]).reindex(close.index)
    low = pd.DataFrame(series["low"]).reindex(close.index)
    volume = pd.DataFrame(series["volume"]).reindex(close.index)
    return (
        close,
        open_,
        high,
        low,
        volume,
        {symbol: metadata[symbol] for symbol in stock_symbols},
        paths,
    )


def _price_data_integrity(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    sector_by_symbol: Mapping[str, str],
    *,
    authority_certified: bool = False,
) -> dict[str, Any]:
    stock_symbols = list(sector_by_symbol)
    finite = (
        np.isfinite(open_[stock_symbols])
        & np.isfinite(high[stock_symbols])
        & np.isfinite(low[stock_symbols])
        & np.isfinite(close[stock_symbols])
    )
    positive_prices = (
        (open_[stock_symbols] > 0.0)
        & (high[stock_symbols] > 0.0)
        & (low[stock_symbols] > 0.0)
        & (close[stock_symbols] > 0.0)
    )
    invalid_envelope = finite & (
        (high[stock_symbols] < open_[stock_symbols])
        | (high[stock_symbols] < close[stock_symbols])
        | (low[stock_symbols] > open_[stock_symbols])
        | (low[stock_symbols] > close[stock_symbols])
    )
    negative_volume = (volume[stock_symbols] < 0.0).fillna(False)
    checks = {
        "monotonic_unique_session_index": bool(
            close.index.is_monotonic_increasing and close.index.is_unique
        ),
        "no_nonpositive_finite_ohlc": not bool((finite & ~positive_prices).to_numpy().any()),
        "valid_ohlc_envelopes": not bool(invalid_envelope.to_numpy().any()),
        "nonnegative_volume": not bool(negative_volume.to_numpy().any()),
        "selection_view_ends_at_calibration": str(close.index.max().date()) <= CALIBRATION_END,
        "frozen_98_name_execution_universe_available": (
            set(stock_symbols) == TRADABLE_EXECUTION_SYMBOLS
        ),
        "market_and_sector_references_available": all(
            symbol in close for symbol in {"SPY", *SECTOR_ETFS.values()}
        ),
    }
    return {
        "passed_structural_checks": all(checks.values()),
        "checks": checks,
        "input_scope": "price_volume_only",
        "news_or_quotes_accessed": False,
        "native_volume_unit": "ibkr_100_share_lots",
        "share_volume_multiplier": IBKR_SHARE_VOLUME_MULTIPLIER,
        "universe_contract": "frozen_98_intraday_symbols_only",
        "known_authority_limitations": (
            []
            if authority_certified
            else [
                "diagnostic inventory may contain survivorship bias",
                "corporate-action-consistent signal and executable price bases are not certified",
                "historical/live adapter parity is not certified",
            ]
        ),
        "research_scope_limitations": [
            "results estimate performance conditional on the predeclared frozen 98-name execution universe and do not claim index-wide generality"
        ],
    }


def _causal_factor_residual(
    stock_return: np.ndarray,
    market_return: np.ndarray,
    sector_return: np.ndarray | None,
    peer_return: np.ndarray | None = None,
    *,
    window: int = 120,
    min_observations: int = 60,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Rolling residual whose fit always ends strictly before ``t``.

    Cumulative sufficient statistics preserve the exact rolling normal
    equations while avoiding an O(window) history rebuild for every session.
    """

    result = np.full(len(stock_return), np.nan, dtype=float)
    factors = [market_return]
    if sector_return is not None:
        factors.append(sector_return)
    if peer_return is not None:
        factors.append(peer_return)
    design = np.column_stack([np.ones(len(stock_return)), *factors])
    valid = np.isfinite(stock_return) & np.isfinite(design).all(axis=1)
    clean_x = np.where(valid[:, None], design, 0.0)
    clean_y = np.where(valid, stock_return, 0.0)
    cumulative_gram = np.concatenate(
        [
            np.zeros((1, design.shape[1], design.shape[1]), dtype=float),
            np.cumsum(clean_x[:, :, None] * clean_x[:, None, :], axis=0),
        ],
        axis=0,
    )
    cumulative_rhs = np.concatenate(
        [
            np.zeros((1, design.shape[1]), dtype=float),
            np.cumsum(clean_x * clean_y[:, None], axis=0),
        ],
        axis=0,
    )
    cumulative_count = np.concatenate([[0], np.cumsum(valid.astype(int))])
    eligible: list[int] = []
    grams: list[np.ndarray] = []
    right_sides: list[np.ndarray] = []
    for index in range(len(stock_return)):
        start = max(0, index - window)
        count = int(cumulative_count[index] - cumulative_count[start])
        if not valid[index] or count < min_observations:
            continue
        gram = cumulative_gram[index] - cumulative_gram[start]
        gram[1:, 1:] += np.eye(len(factors)) * ridge
        grams.append(gram)
        right_sides.append(cumulative_rhs[index] - cumulative_rhs[start])
        eligible.append(index)
    if eligible:
        try:
            coefficients = np.linalg.solve(np.stack(grams), np.stack(right_sides))
        except np.linalg.LinAlgError:
            coefficients = np.full((len(eligible), design.shape[1]), np.nan)
            for position, (gram, rhs) in enumerate(zip(grams, right_sides)):
                try:
                    coefficients[position] = np.linalg.solve(gram, rhs)
                except np.linalg.LinAlgError:
                    continue
        indices = np.asarray(eligible, dtype=int)
        predictions = np.einsum("ij,ij->i", design[indices], coefficients)
        valid_predictions = np.isfinite(predictions)
        result[indices[valid_predictions]] = (
            stock_return[indices[valid_predictions]] - predictions[valid_predictions]
        )
    return result


def _causal_correlated_peer_returns(
    returns: pd.DataFrame,
    sector_by_symbol: Mapping[str, str],
    *,
    lookback: int = 120,
    min_observations: int = 60,
    peer_count: int = 5,
    rebalance_sessions: int = 21,
) -> pd.DataFrame:
    """Return a causal price-only industry-peer proxy for every stock.

    Peer identities are selected only from prior completed returns, within the
    broad sector, and are held between scheduled rebalances.  If the frozen
    execution universe contains no other member of a stock's sector, the
    fallback is the most correlated stocks in the full frozen universe.  This
    preserves coverage for singleton sectors without using future data or an
    external classification feed.  The stock is always excluded from its own
    peer factor.
    """

    symbols = [symbol for symbol in sector_by_symbol if symbol in returns]
    result = pd.DataFrame(np.nan, index=returns.index, columns=symbols, dtype=float)
    peers: dict[str, tuple[str, ...]] = {}
    by_sector: dict[str, list[str]] = defaultdict(list)
    for symbol in symbols:
        by_sector[str(sector_by_symbol[symbol])].append(symbol)

    for index in range(len(returns.index)):
        if index >= min_observations and (
            not peers or (index - min_observations) % rebalance_sessions == 0
        ):
            start = max(0, index - lookback)
            history = returns.iloc[start:index]
            refreshed: dict[str, tuple[str, ...]] = {}
            correlations = history[symbols].corr(
                min_periods=min_observations
            )
            for symbol in symbols:
                sector_candidates = [
                    peer
                    for peer in by_sector[str(sector_by_symbol[symbol])]
                    if peer != symbol
                ]
                candidates = sector_candidates or [
                    peer for peer in symbols if peer != symbol
                ]
                ranked = (
                    correlations[symbol]
                    .reindex(candidates)
                    .dropna()
                    .sort_values(ascending=False, kind="mergesort")
                )
                selected = tuple(
                    str(name) for name in ranked.head(peer_count).index
                )
                if selected:
                    refreshed[symbol] = selected
            peers = refreshed
        current = returns.iloc[index]
        for symbol, selected in peers.items():
            values = current.reindex(selected).dropna()
            if len(values) >= min(2, len(selected)):
                result.iat[index, result.columns.get_loc(symbol)] = float(values.median())
    return result


def build_opportunity_atlas(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    sector_by_symbol: Mapping[str, str],
    *,
    factor_model: str = "market_sector_peer",
    precomputed_peer_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build every causal residual observation before portfolio caps."""

    if factor_model not in FACTOR_MODELS:
        raise ValueError(f"unsupported residual factor model: {factor_model}")

    unmapped_sectors = sorted(set(sector_by_symbol.values()) - set(SECTOR_ETFS))
    if unmapped_sectors:
        raise ValueError(
            "every residual stock sector must map to an explanatory ETF; unmapped: "
            + ", ".join(unmapped_sectors)
        )

    # Additive log returns keep multi-session residual formation and frozen
    # anchor reconstruction internally consistent with the executable core.
    returns = np.log(close / close.shift(1))
    market = returns["SPY"].to_numpy(dtype=float)
    market_volatility_20d = returns["SPY"].rolling(20, min_periods=20).std()
    market_trend_z_20d = (
        returns["SPY"].rolling(20, min_periods=20).sum()
        / (market_volatility_20d * math.sqrt(20.0)).replace(0.0, np.nan)
    )
    peer_returns = (
        precomputed_peer_returns
        if precomputed_peer_returns is not None
        else _causal_correlated_peer_returns(returns, sector_by_symbol)
    )
    residual_by_symbol: dict[str, pd.Series] = {}
    for symbol, sector in sector_by_symbol.items():
        sector_etf = SECTOR_ETFS.get(sector)
        if not sector_etf or sector_etf not in returns:
            raise ValueError(f"missing sector reference for {symbol}: {sector}")
        stock_values = returns[symbol].to_numpy(dtype=float)
        sector_values = returns[sector_etf].to_numpy(dtype=float)
        peer_values = (
            peer_returns[symbol].to_numpy(dtype=float)
            if symbol in peer_returns
            else np.full(len(returns), np.nan, dtype=float)
        )
        if factor_model == "market_only":
            residual_values = _causal_factor_residual(stock_values, market, None, None)
        elif factor_model == "market_sector":
            residual_values = _causal_factor_residual(
                stock_values, market, sector_values, None
            )
        elif factor_model == "market_sector_peer":
            if symbol not in peer_returns:
                continue
            residual_values = _causal_factor_residual(
                stock_values, market, sector_values, peer_values
            )
        else:
            if symbol not in peer_returns:
                continue
            residual_values = stock_values - peer_values
        residual_by_symbol[symbol] = pd.Series(residual_values, index=close.index)
    residual_panel = pd.DataFrame(residual_by_symbol, index=close.index)
    cross_sectional_dispersion = residual_panel.std(axis=1, skipna=True)
    dispersion_reference = (
        cross_sectional_dispersion.shift(1).rolling(120, min_periods=60).median()
    )
    records: list[pd.DataFrame] = []
    for symbol, sector in sector_by_symbol.items():
        if symbol not in residual_panel:
            continue
        residual_series = residual_panel[symbol]
        sector_return_5d = returns[SECTOR_ETFS[sector]].rolling(
            5, min_periods=5
        ).sum()
        residual_volatility = residual_series.shift(1).rolling(60, min_periods=40).std()
        adv_dollars = (
            close[symbol] * volume[symbol] * 100.0
        ).shift(1).rolling(20, min_periods=15).mean()
        suspicious_jump = returns[symbol].abs() >= 0.35
        previous_close = close[symbol].shift(1)
        true_range = pd.concat(
            [
                high[symbol] - low[symbol],
                (high[symbol] - previous_close).abs(),
                (low[symbol] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_fraction = (
            true_range.shift(1).rolling(20, min_periods=15).mean()
            / previous_close.replace(0.0, np.nan)
        )
        session_range = (high[symbol] - low[symbol]).replace(0.0, np.nan)
        close_location = ((close[symbol] - low[symbol]) / session_range).clip(0.0, 1.0)
        lower_wick = (
            (pd.concat([open_[symbol], close[symbol]], axis=1).min(axis=1) - low[symbol])
            / session_range
        ).clip(0.0, 1.0)
        upper_wick = (
            (high[symbol] - pd.concat([open_[symbol], close[symbol]], axis=1).max(axis=1))
            / session_range
        ).clip(0.0, 1.0)
        price_rejection_long = (0.65 * close_location + 0.35 * lower_wick).clip(0.0, 1.0)
        price_rejection_short = (
            0.65 * (1.0 - close_location) + 0.35 * upper_wick
        ).clip(0.0, 1.0)
        lagged_volume_median = volume[symbol].shift(1).rolling(20, min_periods=15).median()
        relative_volume = volume[symbol] / lagged_volume_median.replace(0.0, np.nan)
        prior_relative_volume = relative_volume.shift(1).rolling(3, min_periods=1).mean()
        # An isolated pressure transition is observable at the formation close;
        # future volume deceleration is reserved for a causal opening entry.
        isolated_pressure = (
            (relative_volume / 2.0).clip(0.0, 1.0)
            * (1.0 - (prior_relative_volume / 3.0).clip(0.0, 1.0) * 0.50)
        ).clip(0.0, 1.0)
        volume_exhaustion_quality = (
            1.0 - (isolated_pressure - 0.55).abs() / 0.35
        ).clip(0.0, 1.0)
        adv_quality = (
            (np.log10(adv_dollars.clip(lower=50_000_000.0)) - math.log10(50_000_000.0))
            / (math.log10(1_000_000_000.0) - math.log10(50_000_000.0))
        ).clip(0.0, 1.0)
        volatility_quality = (1.0 - (residual_volatility / 0.06).clip(0.0, 1.0)).clip(0.0, 1.0)
        dispersion_quality = (
            cross_sectional_dispersion
            / dispersion_reference.replace(0.0, np.nan)
        ).clip(0.50, 2.0)
        dispersion_quality = (1.0 - (dispersion_quality - 1.0).abs() / 1.5).clip(0.0, 1.0)
        regime_execution_quality = (
            0.45 * adv_quality + 0.30 * volatility_quality + 0.25 * dispersion_quality
        ).clip(0.0, 1.0)
        prior_residual_trend = residual_series.shift(1).rolling(5, min_periods=3).sum()
        record: dict[str, Any] = {
            "formation_date": close.index,
            "symbol": symbol,
            "issuer": issuer_key(symbol),
            "sector": sector,
            "factor_model": factor_model,
            "tradable_execution_universe": symbol in TRADABLE_EXECUTION_SYMBOLS,
            "residual_return": residual_series.to_numpy(),
            "residual_volatility": residual_volatility.to_numpy(),
            "adv_dollars": adv_dollars.to_numpy(),
            "suspicious_price_jump": suspicious_jump.to_numpy(),
            "price_rejection_long": price_rejection_long.to_numpy(),
            "price_rejection_short": price_rejection_short.to_numpy(),
            "volume_transition": isolated_pressure.to_numpy(),
            "volume_exhaustion_quality": volume_exhaustion_quality.to_numpy(),
            "regime_execution_quality": regime_execution_quality.to_numpy(),
            "sector_return_5d": sector_return_5d.to_numpy(),
            "market_trend_z_20d": market_trend_z_20d.to_numpy(),
        }
        for formation in (1, 3, 5, 20):
            cumulative = residual_series.rolling(
                formation, min_periods=formation
            ).sum()
            scaled_volatility = residual_volatility * math.sqrt(float(formation))
            record[f"residual_z_h{formation}"] = (
                cumulative / scaled_volatility.replace(0.0, np.nan)
            ).to_numpy()
            record[f"suspicious_price_jump_h{formation}"] = (
                suspicious_jump.rolling(formation, min_periods=1).max().astype(bool).to_numpy()
            )
            long_shock = (-cumulative).clip(lower=0.0)
            short_shock = cumulative.clip(lower=0.0)
            prior_long = (-prior_residual_trend).clip(lower=0.0)
            prior_short = prior_residual_trend.clip(lower=0.0)
            record[f"shock_freshness_long_h{formation}"] = (
                long_shock / (long_shock + prior_long + 1e-12)
            ).clip(0.0, 1.0).to_numpy()
            record[f"shock_freshness_short_h{formation}"] = (
                short_shock / (short_shock + prior_short + 1e-12)
            ).clip(0.0, 1.0).to_numpy()
            room_scale = atr_fraction.replace(0.0, np.nan) * 3.0
            record[f"normalization_room_long_h{formation}"] = (
                long_shock / room_scale
            ).clip(0.0, 1.0).to_numpy()
            record[f"normalization_room_short_h{formation}"] = (
                short_shock / room_scale
            ).clip(0.0, 1.0).to_numpy()
            if formation == 1:
                failed_long = (
                    np.log(
                        close[symbol].clip(lower=1e-12)
                        / low[symbol].clip(lower=1e-12)
                    )
                    / residual_volatility.replace(0.0, np.nan)
                )
                failed_short = (
                    np.log(
                        high[symbol].clip(lower=1e-12)
                        / close[symbol].clip(lower=1e-12)
                    )
                    / residual_volatility.replace(0.0, np.nan)
                )
            else:
                failed_long = residual_series / residual_volatility.replace(
                    0.0, np.nan
                )
                failed_short = -residual_series / residual_volatility.replace(
                    0.0, np.nan
                )
            record[f"failed_continuation_long_r_h{formation}"] = (
                failed_long.to_numpy()
            )
            record[f"failed_continuation_short_r_h{formation}"] = (
                failed_short.to_numpy()
            )
            record[f"failed_continuation_long_h{formation}"] = (
                failed_long.clip(0.0, 1.0).to_numpy()
            )
            record[f"failed_continuation_short_h{formation}"] = (
                failed_short.clip(0.0, 1.0).to_numpy()
            )
        records.append(pd.DataFrame(record))
    atlas = pd.concat(records, ignore_index=True)
    atlas = atlas[
        atlas["formation_date"].between(DISCOVERY_START, CALIBRATION_END)
        & np.isfinite(atlas["residual_z_h1"])
        & (atlas["adv_dollars"] >= 50_000_000.0)
        & ~atlas["suspicious_price_jump"]
    ].copy()
    for formation in (1, 3, 5, 20):
        signal = f"residual_z_h{formation}"
        atlas[f"residual_percentile_h{formation}"] = atlas.groupby(
            "formation_date"
        )[signal].rank(method="first", pct=True, ascending=True)
        percentile = atlas[f"residual_percentile_h{formation}"]
        z = atlas[signal]
        atlas[f"residual_extremeness_long_h{formation}"] = (
            0.50 * (1.0 - percentile) + 0.50 * (-z / 3.0).clip(0.0, 1.0)
        )
        atlas[f"residual_extremeness_short_h{formation}"] = (
            0.50 * percentile + 0.50 * (z / 3.0).clip(0.0, 1.0)
        )
        for side in ("long", "short"):
            score_column = f"daily_score_{side}_h{formation}"
            component_columns = {
                "residual_extremeness": f"residual_extremeness_{side}_h{formation}",
                "shock_freshness": f"shock_freshness_{side}_h{formation}",
                "price_rejection_recovery": f"price_rejection_{side}",
                "volume_transition": "volume_transition",
                "volume_exhaustion_quality": "volume_exhaustion_quality",
                "regime_execution_quality": "regime_execution_quality",
                "failed_continuation": f"failed_continuation_{side}_h{formation}",
            }
            atlas[score_column] = 100.0 * sum(
                float(DAILY_RESIDUAL_SCORE_WEIGHTS[name])
                * atlas[column].clip(0.0, 1.0)
                for name, column in component_columns.items()
            )

    calendar = close.index
    positions = {timestamp: index for index, timestamp in enumerate(calendar)}
    # Retain every intermediate mark for shared-capital MTM accounting.  Only
    # (1, 2, 3, 5, 7, 10) are registered exit decisions.
    for holding in range(1, 11):
        entry_prices: list[float] = []
        exit_prices: list[float] = []
        entry_dates: list[pd.Timestamp | pd.NaT] = []
        exit_dates: list[pd.Timestamp | pd.NaT] = []
        for row in atlas.itertuples(index=False):
            formation_index = positions.get(pd.Timestamp(row.formation_date), -1)
            entry_index = formation_index + 1
            exit_index = entry_index + holding - 1
            if entry_index >= len(calendar) or exit_index >= len(calendar):
                entry_prices.append(np.nan)
                exit_prices.append(np.nan)
                entry_dates.append(pd.NaT)
                exit_dates.append(pd.NaT)
                continue
            entry_prices.append(float(open_.at[calendar[entry_index], row.symbol]))
            exit_prices.append(float(close.at[calendar[exit_index], row.symbol]))
            entry_dates.append(calendar[entry_index])
            exit_dates.append(calendar[exit_index])
        atlas[f"entry_price_h{holding}"] = entry_prices
        atlas[f"exit_price_h{holding}"] = exit_prices
        atlas[f"entry_date_h{holding}"] = entry_dates
        atlas[f"exit_date_h{holding}"] = exit_dates
    return atlas.reset_index(drop=True)


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    maximum = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def _immutable_score_components(raw: Mapping[str, float]) -> dict[str, float]:
    """Map economic units to stable [0, 1] values before fixed weighting.

    Zero alpha is neutral (0.5), zero executable frequency is zero, and the
    bounded quality terms retain their literal meaning.  This prevents a
    no-trade candidate receiving half of the frequency allocation and avoids
    outcome-dependent min/max scaling across the candidate registry.
    """

    return {
        "net_expected_r_per_month": 0.5
        + 0.5
        * math.tanh(
            float(raw["net_expected_r_per_month"])
            / SCORE_SPEC["net_expected_r_per_month"]["scale"]
        ),
        "executable_trades_per_month": min(
            max(
                float(raw["executable_trades_per_month"])
                / SCORE_SPEC["executable_trades_per_month"]["scale"],
                0.0,
            ),
            1.0,
        ),
        "worst_fold_r_per_month": 0.5
        + 0.5
        * math.tanh(
            float(raw["worst_fold_r_per_month"])
            / SCORE_SPEC["worst_fold_r_per_month"]["scale"]
        ),
        "average_r_and_discrimination": 0.5
        + 0.5
        * math.tanh(
            float(raw["average_r_and_discrimination"])
            / SCORE_SPEC["average_r_and_discrimination"]["scale"]
        ),
        "downside_risk": min(
            max(math.exp(min(float(raw["downside_risk"]), 0.0)), 0.0), 1.0
        ),
        "issuer_sector_concentration": min(
            max(float(raw["issuer_sector_concentration"]), 0.0), 1.0
        ),
        # Unknown until the complete registered neighbourhood is available.
        "cost_and_neighbourhood_robustness": 0.5,
    }


def _months(start: str, end: str) -> float:
    return max((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4375, 1.0)


def _fold_metrics(
    trades: pd.DataFrame,
    start: str,
    end: str,
    candidate: Candidate | None = None,
) -> dict[str, float]:
    fold = trades[trades["formation_date"].between(start, end)]
    values = fold["r"].astype(float).tolist()
    daily_values = (
        _mark_to_market_daily_r(fold, candidate)
        if candidate is not None and not fold.empty
        else fold.groupby("exit_date")["r"].sum().astype(float).tolist()
        if not fold.empty
        else []
    )
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    gross_loss = -sum(losses)
    return {
        "trades": float(len(values)),
        "total_r": float(sum(values)),
        "avg_r": float(np.mean(values)) if values else 0.0,
        "r_per_month": float(sum(values)) / _months(start, end),
        "trades_per_month": float(len(values)) / _months(start, end),
        "win_rate": float(len(wins) / len(values)) if values else 0.0,
        "profit_factor": min(sum(wins) / gross_loss, 5.0) if gross_loss > 0 else (5.0 if wins else 0.0),
        "max_drawdown_r": _max_drawdown(daily_values),
    }


def _cost_stress_metrics(
    trades: pd.DataFrame,
    candidate: Candidate,
    *,
    cost_bps: float,
    start: str,
    end: str,
) -> dict[str, float]:
    fold = trades[trades["formation_date"].between(start, end)].copy()
    if fold.empty:
        return {"trades": 0.0, "total_r": 0.0, "avg_r": 0.0, "profit_factor": 0.0}
    risk = fold["residual_volatility"] * math.sqrt(float(candidate.holding_sessions))
    additional_cost = max(float(cost_bps) - float(candidate.round_trip_cost_bps), 0.0) / 10_000.0
    stressed = fold["r"] - fold["leg_weight"] * additional_cost / risk.replace(0.0, np.nan)
    stressed = stressed.replace([np.inf, -np.inf], np.nan).dropna().clip(-5.0, 5.0)
    wins = stressed[stressed > 0.0]
    losses = stressed[stressed < 0.0]
    gross_loss = -float(losses.sum())
    return {
        "trades": float(len(stressed)),
        "total_r": float(stressed.sum()),
        "avg_r": float(stressed.mean()) if len(stressed) else 0.0,
        "profit_factor": min(float(wins.sum()) / gross_loss, 5.0)
        if gross_loss > 0.0
        else (5.0 if len(wins) else 0.0),
    }


def _shared_capital_daily_returns(
    trades: pd.DataFrame,
    candidate: Candidate,
    *,
    ordinary_risk_fraction: float = 0.0035,
    maximum_notional_fraction: float = 0.10,
) -> list[float]:
    """Conservative fixed-size shared-capital MTM return changes.

    Ten simultaneous ordinary positions can consume at most 100% gross
    notional.  Residual-volatility sizing may use less capital, so this is a
    risk-budgeted replay rather than independent-account recombination.
    """

    by_date: defaultdict[pd.Timestamp, float] = defaultdict(float)
    holding = int(candidate.holding_sessions)
    cost = float(candidate.round_trip_cost_bps) / 10_000.0
    for row in trades.itertuples(index=False):
        risk = float(row.residual_volatility) * math.sqrt(float(holding))
        if not math.isfinite(risk) or risk <= 0.0:
            continue
        leg_weight = float(row.leg_weight)
        notional = min(ordinary_risk_fraction / risk, maximum_notional_fraction) * leg_weight
        direction = 1.0 if row.trade_side == "long" else -1.0
        prior_cumulative = 0.0
        for day in range(1, holding + 1):
            mark = float(getattr(row, f"exit_price_h{day}"))
            mark_date = pd.Timestamp(getattr(row, f"exit_date_h{day}"))
            if not math.isfinite(mark) or pd.isna(mark_date):
                break
            cumulative = notional * (
                direction * (mark / float(row.entry_price) - 1.0) - cost
            )
            by_date[mark_date] += cumulative - prior_cumulative
            prior_cumulative = cumulative
    return [by_date[date] for date in sorted(by_date)]


def _mark_to_market_daily_r(
    trades: pd.DataFrame,
    candidate: Candidate,
) -> list[float]:
    """Return shared-calendar daily R changes including every open position."""

    by_date: defaultdict[pd.Timestamp, float] = defaultdict(float)
    holding = int(candidate.holding_sessions)
    cost = float(candidate.round_trip_cost_bps) / 10_000.0
    for row in trades.itertuples(index=False):
        risk = float(row.residual_volatility) * math.sqrt(float(holding))
        if not math.isfinite(risk) or risk <= 0.0:
            continue
        direction = 1.0 if row.trade_side == "long" else -1.0
        prior_cumulative = 0.0
        for day in range(1, holding + 1):
            mark = float(getattr(row, f"exit_price_h{day}"))
            mark_date = pd.Timestamp(getattr(row, f"exit_date_h{day}"))
            if not math.isfinite(mark) or pd.isna(mark_date):
                break
            cumulative = float(row.leg_weight) * np.clip(
                (direction * (mark / float(row.entry_price) - 1.0) - cost) / risk,
                -5.0,
                5.0,
            )
            by_date[mark_date] += cumulative - prior_cumulative
            prior_cumulative = cumulative
    return [by_date[date] for date in sorted(by_date)]


def _cluster_bootstrap_lower_mean(
    values: Iterable[float],
    *,
    seed: int,
    repetitions: int = 500,
) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 20:
        return float("-inf")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.05))


def _cluster_bootstrap_summary(
    values: Iterable[float],
    *,
    seed: int,
    repetitions: int = 1_000,
) -> dict[str, float]:
    """Return uncertainty evidence without pretending a short fold is huge.

    A one-sided 95% lower bound is a production-certification standard, not a
    sensible discovery veto for roughly thirty independent weeks.  Selection
    instead requires a positive point estimate and at least 75% bootstrap
    probability of a positive clustered mean. Locked validation remains the
    one-shot promotion test.
    """

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 20:
        return {
            "mean_r": float(array.mean()) if len(array) else 0.0,
            "lower_80pct_mean_r": float("-inf"),
            "probability_mean_positive": 0.0,
        }
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean_r": float(array.mean()),
        "lower_80pct_mean_r": float(np.quantile(means, 0.10)),
        "probability_mean_positive": float(np.mean(means > 0.0)),
    }


def _robustness_diagnostics(
    trades: pd.DataFrame,
    candidate: Candidate,
) -> dict[str, Any]:
    seed = int(hashlib.sha256(candidate.candidate_id.encode("utf-8")).hexdigest()[:8], 16)
    folds: dict[str, Any] = {}
    for fold_index, (name, start, end) in enumerate(FOLDS):
        fold = trades[trades["formation_date"].between(start, end)]
        daily = fold.groupby("formation_date")["r"].sum()
        weekly = fold.assign(
            formation_week=pd.to_datetime(fold["formation_date"]).dt.to_period("W-FRI")
        ).groupby("formation_week")["r"].sum()
        issuer = fold.groupby("issuer")["r"].sum()
        sector = fold.groupby("sector")["r"].sum()
        total = float(fold["r"].sum())
        leave_one_sector = {
            str(label): total - float(value) for label, value in sector.items()
        }
        date_bootstrap = _cluster_bootstrap_summary(
            daily, seed=seed + fold_index * 2
        )
        issuer_bootstrap = _cluster_bootstrap_summary(
            issuer, seed=seed + fold_index * 2 + 1
        )
        week_bootstrap = _cluster_bootstrap_summary(
            weekly, seed=seed + fold_index * 2 + 101
        )
        folds[name] = {
            "effective_dates": int(len(daily)),
            "available_business_dates": int(len(pd.bdate_range(start, end))),
            "effective_date_share": float(len(daily) / max(len(pd.bdate_range(start, end)), 1)),
            "effective_weeks": int(len(weekly)),
            "effective_issuers": int(len(issuer)),
            "date_cluster_bootstrap_lower_mean_r": _cluster_bootstrap_lower_mean(
                daily, seed=seed + fold_index * 2
            ),
            "issuer_cluster_bootstrap_lower_mean_r": _cluster_bootstrap_lower_mean(
                issuer, seed=seed + fold_index * 2 + 1
            ),
            "moving_week_bootstrap_lower_mean_r": _cluster_bootstrap_lower_mean(
                weekly, seed=seed + fold_index * 2 + 101
            ),
            "date_cluster_bootstrap": date_bootstrap,
            "issuer_cluster_bootstrap": issuer_bootstrap,
            "moving_week_cluster_bootstrap": week_bootstrap,
            "leave_one_sector_total_r": leave_one_sector,
            "all_leave_one_sector_positive": bool(leave_one_sector)
            and all(value > 0.0 for value in leave_one_sector.values()),
            "adv_strata_total_r": {
                str(int(floor / 1_000_000)): float(
                    fold.loc[fold["adv_dollars"] >= floor, "r"].sum()
                )
                for floor in (50_000_000.0, 100_000_000.0, 250_000_000.0)
            },
        }
    return folds


def _capacity_stress_diagnostics(
    atlas: pd.DataFrame,
    candidate: Candidate,
    *,
    enabled: bool,
) -> dict[str, Any]:
    variants = {
        "positions_5": replace(candidate, max_positions=5),
        "positions_15": replace(candidate, max_positions=15),
        "sector_cap_1": replace(candidate, max_positions_per_sector=1),
        "sector_cap_3": replace(candidate, max_positions_per_sector=3),
    }
    if not enabled:
        return {"tested": False, "variants": {}, "positive_calibration_share": 0.0}
    rows: dict[str, Any] = {}
    positive = 0
    for name, variant in variants.items():
        trades = _select_candidate(atlas, variant)
        metrics = _fold_metrics(
            trades, CALIBRATION_START, CALIBRATION_END, variant
        )
        rows[name] = metrics
        positive += int(metrics["total_r"] > 0.0 and metrics["profit_factor"] > 1.0)
    return {
        "tested": True,
        "variants": rows,
        "positive_calibration_share": positive / len(variants),
    }


def _select_candidate(
    atlas: pd.DataFrame,
    candidate: Candidate,
    *,
    apply_capacity: bool = True,
    rejection_cohort: str | None = None,
) -> pd.DataFrame:
    holding = candidate.holding_sessions
    formation = candidate.formation_sessions
    signal = f"residual_z_h{formation}"
    bounded = atlas[atlas["formation_date"].between(DISCOVERY_START, CALIBRATION_END)]
    if "tradable_execution_universe" in bounded:
        bounded = bounded[bounded["tradable_execution_universe"]]

    def signal_pool(side: str) -> pd.DataFrame:
        score_column = f"daily_score_{side}_h{formation}"
        failed_raw_column = f"failed_continuation_{side}_r_h{formation}"
        failed_component_column = f"failed_continuation_{side}_h{formation}"
        failed_raw = bounded.get(
            failed_raw_column, pd.Series(0.0, index=bounded.index)
        )
        sector_return_5d = bounded.get(
            "sector_return_5d", pd.Series(0.0, index=bounded.index)
        )
        market_trend_z_20d = bounded.get(
            "market_trend_z_20d", pd.Series(0.0, index=bounded.index)
        )
        if rejection_cohort not in {None, "low_extremeness", "continuation"}:
            raise ValueError(f"unsupported rejection cohort: {rejection_cohort}")
        if side == "long":
            extreme = bounded[signal] <= -candidate.residual_z_floor
            if rejection_cohort == "low_extremeness":
                extreme = bounded[signal].between(
                    -candidate.residual_z_floor, -0.50, inclusive="neither"
                )
            failed = failed_raw >= candidate.minimum_failed_continuation_r
            if rejection_cohort == "continuation":
                failed = failed_raw < candidate.minimum_failed_continuation_r
            mask = (
                extreme
                & failed
                & (
                    sector_return_5d
                    >= candidate.minimum_sector_return_5d
                )
                & (
                    market_trend_z_20d
                    >= candidate.minimum_market_trend_z_20d
                )
                & (bounded[f"normalization_room_long_h{formation}"] > 0.0)
                & np.isfinite(bounded[score_column])
                & ~bounded[f"suspicious_price_jump_h{formation}"]
            )
        else:
            extreme = bounded[signal] >= candidate.residual_z_floor
            if rejection_cohort == "low_extremeness":
                extreme = bounded[signal].between(
                    0.50, candidate.residual_z_floor, inclusive="neither"
                )
            failed = failed_raw >= candidate.minimum_failed_continuation_r
            if rejection_cohort == "continuation":
                failed = failed_raw < candidate.minimum_failed_continuation_r
            mask = (
                extreme
                & failed
                & (
                    sector_return_5d
                    >= candidate.minimum_sector_return_5d
                )
                & (
                    market_trend_z_20d
                    >= candidate.minimum_market_trend_z_20d
                )
                & (bounded[f"normalization_room_short_h{formation}"] > 0.0)
                & np.isfinite(bounded[score_column])
                & ~bounded[f"suspicious_price_jump_h{formation}"]
            )
        pool = bounded[mask].copy()
        pool["residual_z"] = pool[signal]
        pool["daily_score"] = pool[score_column]
        pool["trade_side"] = side
        pool["component_residual_extremeness"] = pool[
            f"residual_extremeness_{side}_h{formation}"
        ]
        pool["component_shock_freshness"] = pool[
            f"shock_freshness_{side}_h{formation}"
        ]
        pool["component_price_rejection_recovery"] = pool[
            f"price_rejection_{side}"
        ]
        pool["component_volume_transition"] = pool["volume_transition"]
        pool["component_volume_exhaustion_quality"] = pool.get(
            "volume_exhaustion_quality",
            (1.0 - (pool["volume_transition"] - 0.55).abs() / 0.35).clip(
                0.0, 1.0
            ),
        )
        pool["component_residual_normalization_room"] = pool[
            f"normalization_room_{side}_h{formation}"
        ]
        pool["component_regime_execution_quality"] = pool[
            "regime_execution_quality"
        ]
        pool["failed_continuation_r"] = failed_raw.reindex(pool.index).fillna(0.0)
        pool["component_failed_continuation"] = bounded.get(
            failed_component_column, pd.Series(0.0, index=bounded.index)
        ).reindex(pool.index).fillna(0.0)
        pool["sector_return_5d"] = sector_return_5d.reindex(pool.index)
        if candidate.score_components:
            selected_columns = [
                f"component_{name}" for name in candidate.score_components
            ]
            if not 1 <= len(selected_columns) <= 7:
                raise ValueError("candidate score must contain one to seven components")
            missing = [name for name in selected_columns if name not in pool]
            if missing:
                raise ValueError(f"candidate score components are unavailable: {missing}")
            pool["admission_score"] = 100.0 * pool[selected_columns].mean(axis=1)
        else:
            pool["admission_score"] = pool["daily_score"]
        ranking_components = (
            candidate.ranking_score_components or candidate.score_components
        )
        if ranking_components:
            ranking_columns = [
                f"component_{name}" for name in ranking_components
            ]
            missing = [name for name in ranking_columns if name not in pool]
            if missing:
                raise ValueError(
                    f"candidate ranking score components are unavailable: {missing}"
                )
            pool["ranking_score"] = 100.0 * pool[ranking_columns].mean(axis=1)
        else:
            pool["ranking_score"] = pool["admission_score"]
        # Downstream discrimination evaluates the score that actually controls
        # capacity priority.  Admission quality remains explicit for rejection
        # attribution and score-floor gates.
        pool["daily_score"] = pool["ranking_score"]
        pool["entry_date"] = pool[f"entry_date_h{holding}"]
        pool["exit_date"] = pool[f"exit_date_h{holding}"]
        pool["entry_price"] = pool[f"entry_price_h{holding}"]
        pool["exit_price"] = pool[f"exit_price_h{holding}"]
        pool = pool.dropna(
            subset=["entry_date", "exit_date", "entry_price", "exit_price"]
        )
        pool = pool[(pool["entry_price"] > 0.0) & (pool["exit_price"] > 0.0)]
        return pool.sort_values(
            ["formation_date", "daily_score", "residual_z", "symbol"],
            ascending=[True, False, side != "long", True],
        )

    def cap_pool(pool: pd.DataFrame) -> pd.DataFrame:
        accepted: list[int] = []
        active: list[tuple[pd.Timestamp, str, str]] = []
        for _formation_date, group in pool.groupby("formation_date", sort=True):
            entry_date = pd.Timestamp(group["entry_date"].iloc[0])
            active = [item for item in active if item[0] >= entry_date]
            opportunities = [
                DailyResidualOpportunity(
                    symbol=str(row.symbol),
                    issuer=str(row.issuer),
                    sector=str(row.sector),
                    side=str(row.trade_side),
                    residual_z=float(row.residual_z),
                    remaining_room_r=float(
                        row.component_residual_normalization_room
                    ),
                    features=DailyResidualFeatures(
                        residual_extremeness=float(row.component_residual_extremeness),
                        shock_freshness=float(row.component_shock_freshness),
                        price_rejection_recovery=float(
                            row.component_price_rejection_recovery
                        ),
                        volume_transition=float(row.component_volume_transition),
                        volume_exhaustion_quality=float(
                            row.component_volume_exhaustion_quality
                        ),
                        residual_normalization_room=float(
                            row.component_residual_normalization_room
                        ),
                        regime_execution_quality=float(
                            row.component_regime_execution_quality
                        ),
                        failed_continuation=float(
                            row.component_failed_continuation
                        ),
                    ),
                    failed_continuation_r=float(row.failed_continuation_r),
                    sector_return_5d=float(row.sector_return_5d),
                )
                for row in group.itertuples(index=False)
            ]
            score_weights = (
                {name: 1.0 for name in candidate.score_components}
                if candidate.score_components
                else None
            )
            ranking_score_weights = (
                {name: 1.0 for name in candidate.ranking_score_components}
                if candidate.ranking_score_components
                else score_weights
            )
            selected = rank_daily_residual_opportunities(
                opportunities,
                max_positions=candidate.max_positions,
                max_positions_per_sector=candidate.max_positions_per_sector,
                active_issuers=[item[1] for item in active],
                active_sectors=[item[2] for item in active],
                minimum_residual_z=(
                    0.50
                    if rejection_cohort == "low_extremeness"
                    else candidate.residual_z_floor
                ),
                minimum_score=(
                    0.0 if rejection_cohort is not None else candidate.minimum_score
                ),
                minimum_failed_continuation_r=(
                    -1_000_000.0
                    if rejection_cohort == "continuation"
                    else candidate.minimum_failed_continuation_r
                ),
                score_weights=score_weights,
                ranking_score_weights=ranking_score_weights,
            )
            index_by_symbol = {
                str(row["symbol"]): index for index, row in group.iterrows()
            }
            row_by_symbol = {
                str(row["symbol"]): row for _index, row in group.iterrows()
            }
            for ranked in selected:
                symbol = ranked.opportunity.symbol
                index = index_by_symbol[symbol]
                row = row_by_symbol[symbol]
                accepted.append(index)
                active.append(
                    (
                        pd.Timestamp(row["exit_date"]),
                        str(row["issuer"]),
                        str(row["sector"]),
                    )
                )
        return pool.loc[accepted].copy() if accepted else pool.iloc[0:0].copy()

    if candidate.diagnostic_leg == "long_loser":
        pool = signal_pool("long")
        trades = cap_pool(pool) if apply_capacity else pool
        trades["leg_weight"] = 1.0
    elif candidate.diagnostic_leg == "short_winner":
        pool = signal_pool("short")
        trades = cap_pool(pool) if apply_capacity else pool
        trades["leg_weight"] = 1.0
    elif candidate.diagnostic_leg == "dollar_neutral_spread":
        long_pool = signal_pool("long")
        short_pool = signal_pool("short")
        trades = pd.concat(
            [
                cap_pool(long_pool) if apply_capacity else long_pool,
                cap_pool(short_pool) if apply_capacity else short_pool,
            ],
            ignore_index=True,
        )
        trades["leg_weight"] = 0.5
    else:
        raise ValueError(f"unsupported diagnostic leg: {candidate.diagnostic_leg}")

    gross_long_return = trades["exit_price"] / trades["entry_price"] - 1.0
    direction = trades["trade_side"].map({"long": 1.0, "short": -1.0})
    trades["net_return"] = (
        direction * gross_long_return - candidate.round_trip_cost_bps / 10_000.0
    )
    risk = trades["residual_volatility"] * math.sqrt(float(holding))
    trades["r"] = (
        trades["leg_weight"]
        * (trades["net_return"] / risk.replace(0.0, np.nan)).clip(-5.0, 5.0)
    )
    return trades.dropna(subset=["r"]).sort_values(["entry_date", "symbol"])


def _discrimination_metrics(
    observations: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Measure the score's actual within-date capacity-ranking function."""

    fold = observations[observations["formation_date"].between(start, end)].copy()
    if len(fold) < 100 or fold["daily_score"].nunique() < 5:
        return {
            "observations": len(fold),
            "quintile_avg_r": [],
            "top_minus_bottom_r": 0.0,
            "top_minus_middle_r": 0.0,
            "rank_correlation": 0.0,
            "time_shift_placebo_lift_r": 0.0,
            "placebo_lifts_r": [],
            "maximum_absolute_placebo_lift_r": 0.0,
            "placebo_absolute_lift_q95_r": 0.0,
            "placebo_p_value": 1.0,
            "components": {},
            "passed": False,
        }
    fold["selection_score_rank"] = fold.groupby("formation_date")[
        "daily_score"
    ].rank(method="first", pct=True)
    ranks = fold["selection_score_rank"].rank(method="first")
    fold["score_quintile"] = pd.qcut(ranks, 5, labels=False) + 1
    quintiles = fold.groupby("score_quintile")["r"].mean().reindex(range(1, 6))
    values = [float(value) for value in quintiles]
    top_minus_bottom = values[4] - values[0]
    top_minus_middle = values[4] - values[2]
    rank_correlation = float(
        fold[["selection_score_rank", "r"]].corr(method="spearman").iloc[0, 1]
    )

    shifted = fold.sort_values(["formation_date", "sector", "symbol"]).copy()

    def placebo_lift(scores: np.ndarray) -> float:
        placebo = shifted.assign(placebo_score=scores)
        within_date = placebo.groupby("formation_date")["placebo_score"].rank(
            method="first", pct=True
        )
        placebo_rank = within_date.rank(method="first")
        buckets = pd.qcut(placebo_rank, 5, labels=False) + 1
        means = shifted.assign(placebo_quintile=buckets).groupby(
            "placebo_quintile"
        )["r"].mean().reindex(range(1, 6))
        return float(means.iloc[-1] - means.iloc[0])

    placebo_lifts: list[float] = []
    original_scores = shifted["daily_score"].to_numpy(dtype=float)
    for seed in range(32):
        generator = np.random.default_rng(10_003 + seed)
        permuted = shifted.groupby(
            ["formation_date", "sector"], sort=False
        )["daily_score"].transform(
            lambda values: generator.permutation(values.to_numpy(dtype=float))
        )
        placebo_lifts.append(placebo_lift(permuted.to_numpy(dtype=float)))
    for offset in (5, 10, 15, 20, 30, 40, 60, 80):
        placebo_lifts.append(placebo_lift(np.roll(original_scores, offset)))
    placebo_lift_value = placebo_lifts[0]
    max_abs_placebo_lift = max(abs(value) for value in placebo_lifts)
    placebo_abs_q95 = float(np.quantile(np.abs(placebo_lifts), 0.95))
    placebo_p_value = float(
        (1 + sum(value >= top_minus_bottom for value in placebo_lifts))
        / (1 + len(placebo_lifts))
    )
    passed = (
        top_minus_bottom >= 0.05
        and top_minus_middle > 0.0
        and rank_correlation > 0.0
        and values[4] > 0.0
        and top_minus_bottom >= placebo_abs_q95 + 0.01
        and placebo_p_value <= 0.10
    )
    component_diagnostics: dict[str, Any] = {}
    for column in sorted(
        name for name in fold.columns if name.startswith("component_")
    ):
        valid_component = fold[["formation_date", column, "r"]].dropna()
        if len(valid_component) < 100 or valid_component[column].nunique() < 5:
            continue
        valid_component["within_date_rank"] = valid_component.groupby(
            "formation_date"
        )[column].rank(method="first", pct=True)
        component_rank = valid_component["within_date_rank"].rank(method="first")
        buckets = pd.qcut(component_rank, 5, labels=False) + 1
        means = valid_component.assign(bucket=buckets).groupby("bucket")["r"].mean().reindex(range(1, 6))
        component_diagnostics[column.removeprefix("component_")] = {
            "quintile_avg_r": [float(value) for value in means],
            "top_minus_bottom_r": float(means.iloc[-1] - means.iloc[0]),
            "rank_correlation": float(
                valid_component[["within_date_rank", "r"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
        }
    return {
        "observations": len(fold),
        "quintile_avg_r": values,
        "top_minus_bottom_r": top_minus_bottom,
        "top_minus_middle_r": top_minus_middle,
        "rank_correlation": rank_correlation,
        "time_shift_placebo_lift_r": placebo_lift_value,
        "placebo_lifts_r": placebo_lifts,
        "maximum_absolute_placebo_lift_r": max_abs_placebo_lift,
        "placebo_absolute_lift_q95_r": placebo_abs_q95,
        "placebo_p_value": placebo_p_value,
        "components": component_diagnostics,
        "passed": passed,
    }


def qualify_score_components(
    atlas: pd.DataFrame,
    *,
    factor_model: str,
    formation_sessions: int = 3,
    qualification_holding_sessions: int = 5,
    minimum_lift_r: float = 0.015,
    maximum_pairwise_correlation: float = 0.75,
    minimum_failed_continuation_r: float = 0.20,
) -> dict[str, Any]:
    """Freeze a minimal stable component subset before candidate search."""

    probe = Candidate(
        candidate_id=f"{factor_model}_component_probe",
        residual_z_floor=1.0,
        holding_sessions=qualification_holding_sessions,
        max_positions=10,
        max_positions_per_sector=2,
        formation_sessions=formation_sessions,
        diagnostic_leg="long_loser",
        factor_model=factor_model,
        lane_id="component_diagnostic_after_failed_continuation",
        minimum_failed_continuation_r=minimum_failed_continuation_r,
    )
    observations = _select_candidate(atlas, probe, apply_capacity=False)
    diagnostics = {
        name: _discrimination_metrics(observations, start, end)
        for name, start, end in FOLDS
    }
    component_names = sorted(
        set(diagnostics["discovery"].get("components", {}))
        & set(diagnostics["calibration"].get("components", {}))
    )
    stable: list[tuple[str, float]] = []
    for name in component_names:
        rows = [diagnostics[fold]["components"][name] for fold, *_ in FOLDS]
        minimum_lift = min(float(row["top_minus_bottom_r"]) for row in rows)
        minimum_rho = min(float(row["rank_correlation"]) for row in rows)
        if minimum_lift >= minimum_lift_r and minimum_rho > 0.0:
            stable.append((name, minimum_lift))
    stable.sort(key=lambda item: (-item[1], item[0]))

    selected: list[str] = []
    component_frame = observations[
        [f"component_{name}" for name, _lift in stable]
    ].copy() if stable else pd.DataFrame(index=observations.index)
    for name, _lift in stable:
        column = f"component_{name}"
        if any(
            abs(float(component_frame[[column, f"component_{other}"]].corr().iloc[0, 1]))
            > maximum_pairwise_correlation
            for other in selected
        ):
            continue
        selected.append(name)
        if len(selected) >= 5:
            break
    fallback_used = not selected
    if fallback_used:
        selected = ["residual_extremeness"]
    return {
        "factor_model": factor_model,
        "qualification_formation_sessions": formation_sessions,
        "qualification_holding_sessions": qualification_holding_sessions,
        "minimum_component_lift_r": minimum_lift_r,
        "maximum_pairwise_correlation": maximum_pairwise_correlation,
        "minimum_failed_continuation_r": minimum_failed_continuation_r,
        "selected_components": selected,
        "equal_weight": 1.0 / len(selected),
        "stable_component_candidates": [name for name, _lift in stable],
        "fallback_residual_only_used": fallback_used,
        "fold_diagnostics": diagnostics,
    }


def negative_rejection_diagnostics(
    atlas: pd.DataFrame,
    candidate: Candidate,
) -> dict[str, Any]:
    """Compare equally capacity-capped accepted and rejected opportunities.

    The prior diagnostic compared ten selected positions with every rejected
    observation, which answered a different question and penalized capacity
    strategies for having a broad raw aperture. Each counterfactual is now
    ranked with the same score, issuer rule, position cap and sector cap.
    """

    accepted = _select_candidate(atlas, candidate, apply_capacity=True)
    low_extremeness = _select_candidate(
        atlas,
        candidate,
        apply_capacity=True,
        rejection_cohort="low_extremeness",
    )
    persistent = _select_candidate(
        atlas,
        candidate,
        apply_capacity=True,
        rejection_cohort="continuation",
    )
    output: dict[str, Any] = {}
    passes = True
    continuation_rejection_active = (
        float(candidate.minimum_failed_continuation_r) > 0.0
    )
    for name, start, end in FOLDS:
        accepted_fold = accepted[accepted["formation_date"].between(start, end)]
        low_fold = low_extremeness[low_extremeness["formation_date"].between(start, end)]
        persistent_fold = persistent[persistent["formation_date"].between(start, end)]
        accepted_avg = float(accepted_fold["r"].mean()) if len(accepted_fold) else 0.0
        low_avg = float(low_fold["r"].mean()) if len(low_fold) else 0.0
        persistent_avg = float(persistent_fold["r"].mean()) if len(persistent_fold) else 0.0
        fold_pass = (
            len(accepted_fold) >= 100
            and len(low_fold) >= 50
            and accepted_avg >= low_avg + 0.02
            and (
                not continuation_rejection_active
                or (
                    len(persistent_fold) >= 30
                    and accepted_avg >= persistent_avg + 0.02
                )
            )
        )
        passes = passes and fold_pass
        output[name] = {
            "accepted_observations": len(accepted_fold),
            "accepted_avg_r": accepted_avg,
            "low_extremeness_rejected_observations": len(low_fold),
            "low_extremeness_rejected_avg_r": low_avg,
            "persistent_continuation_rejected_observations": len(persistent_fold),
            "persistent_continuation_rejected_avg_r": persistent_avg,
            "persistent_continuation_rejection_status": (
                "active" if continuation_rejection_active else "not_applicable"
            ),
            "passed": fold_pass,
        }
    return {
        "folds": output,
        "passed_each_fold": passes,
        "inactive_mechanisms_are_not_failures": True,
    }


def evaluate_candidate(atlas: pd.DataFrame, candidate: Candidate) -> dict[str, Any]:
    trades = _select_candidate(atlas, candidate)
    uncapped = _select_candidate(atlas, candidate, apply_capacity=False)
    fold_metrics = {
        name: _fold_metrics(trades, start, end, candidate)
        for name, start, end in FOLDS
    }
    values = trades["r"].astype(float).tolist()
    daily_values = _mark_to_market_daily_r(trades, candidate)
    portfolio_daily_returns = _shared_capital_daily_returns(trades, candidate)
    portfolio_drawdown = _max_drawdown(portfolio_daily_returns)
    portfolio_tail_count = max(
        1, int(math.ceil(len(portfolio_daily_returns) * 0.05))
    ) if portfolio_daily_returns else 0
    portfolio_expected_shortfall = (
        float(np.mean(sorted(portfolio_daily_returns)[:portfolio_tail_count]))
        if portfolio_tail_count
        else 0.0
    )
    issuer_r = trades.groupby("issuer")["r"].sum().to_dict() if not trades.empty else {}
    sector_r = trades.groupby("sector")["r"].sum().to_dict() if not trades.empty else {}
    positive_issuer = {key: value for key, value in issuer_r.items() if value > 0.0}
    positive_sector = {key: value for key, value in sector_r.items() if value > 0.0}
    positive_issuer_total = sum(positive_issuer.values())
    positive_sector_total = sum(positive_sector.values())
    top_issuer = max(positive_issuer.values(), default=0.0)
    top_sector = max(positive_sector.values(), default=0.0)
    issuer_neutral_total = sum(values) - top_issuer
    tail_count = max(1, int(math.ceil(len(values) * 0.10))) if values else 0
    expected_shortfall = float(np.mean(sorted(values)[:tail_count])) if tail_count else 0.0
    selection_months = _months(DISCOVERY_START, CALIBRATION_END)
    total_r_per_month = sum(values) / selection_months
    trades_per_month = len(values) / selection_months
    avg_r = float(np.mean(values)) if values else 0.0
    minimum_fold_rpm = min(row["r_per_month"] for row in fold_metrics.values())
    discrimination = {
        name: _discrimination_metrics(uncapped, start, end)
        for name, start, end in FOLDS
    }
    rejection = negative_rejection_diagnostics(atlas, candidate)
    robustness = _robustness_diagnostics(trades, candidate)
    capacity_stress = _capacity_stress_diagnostics(
        atlas,
        candidate,
        enabled=all(row["total_r"] > 0.0 for row in fold_metrics.values()),
    )
    cost_stress = {
        str(int(cost_bps)): {
            name: _cost_stress_metrics(
                trades,
                candidate,
                cost_bps=cost_bps,
                start=start,
                end=end,
            )
            for name, start, end in FOLDS
        }
        for cost_bps in (20.0, 30.0, 40.0)
    }
    minimum_discrimination_lift = min(
        row["top_minus_bottom_r"] for row in discrimination.values()
    )
    positive_sector_count = sum(value > 0.0 for value in sector_r.values())
    top_issuer_share = top_issuer / positive_issuer_total if positive_issuer_total > 0 else 1.0
    top_sector_share = top_sector / positive_sector_total if positive_sector_total > 0 else 1.0
    raw = {
        "net_expected_r_per_month": total_r_per_month,
        "executable_trades_per_month": trades_per_month,
        "worst_fold_r_per_month": minimum_fold_rpm,
        "average_r_and_discrimination": 0.60 * avg_r + 0.40 * minimum_discrimination_lift,
        "downside_risk": -(
            0.70 * portfolio_drawdown / 0.08
            + 0.30 * abs(min(portfolio_expected_shortfall, 0.0)) / 0.015
        ),
        "issuer_sector_concentration": (
            0.40 * (1.0 - top_issuer_share)
            + 0.30 * (1.0 - top_sector_share)
            + 0.30
            * min(
                min(row["effective_date_share"] for row in robustness.values()) / 0.40,
                1.0,
            )
        ),
        # Filled from the complete candidate registry before final ranking.
        "cost_and_neighbourhood_robustness": 0.0,
    }
    components = _immutable_score_components(raw)
    score = sum(SCORE_SPEC[name]["weight"] * components[name] for name in SCORE_SPEC)
    gates = {
        "positive_each_fold": all(row["total_r"] > 0.0 for row in fold_metrics.values()),
        # The 250-per-fold threshold belonged to the rejected 352-name panel.
        # On the exact 98-name execution universe, 100 issuer trades per fold
        # preserves meaningful breadth without making qualification impossible
        # by construction.
        "minimum_100_trades_each_fold": all(
            row["trades"] >= 100 for row in fold_metrics.values()
        ),
        "calibration_avg_r_gte_0p07": fold_metrics["calibration"]["avg_r"] >= 0.07,
        "calibration_profit_factor_gte_1p15": fold_metrics["calibration"]["profit_factor"] >= 1.15,
        "calibration_retains_25pct_discovery_rpm": (
            fold_metrics["discovery"]["r_per_month"] > 0.0
            and fold_metrics["calibration"]["r_per_month"]
            >= 0.25 * fold_metrics["discovery"]["r_per_month"]
        ),
        "positive_issuer_neutral_total_r": issuer_neutral_total > 0.0,
        "top_positive_issuer_share_lte_15pct": top_issuer_share <= 0.15,
        "top_positive_sector_share_lte_35pct": top_sector_share <= 0.35,
        "at_least_four_positive_sectors": positive_sector_count >= 4,
        "score_discrimination_passes_each_fold": all(
            row["passed"] for row in discrimination.values()
        ),
        "accepted_beats_fixed_rejected_cohorts_each_fold": rejection[
            "passed_each_fold"
        ],
        "date_and_issuer_cluster_positive_probability_gte_75pct": all(
            row["date_cluster_bootstrap"]["mean_r"] > 0.0
            and row["issuer_cluster_bootstrap"]["mean_r"] > 0.0
            and row["date_cluster_bootstrap"]["probability_mean_positive"]
            >= 0.75
            and row["issuer_cluster_bootstrap"]["probability_mean_positive"]
            >= 0.75
            for row in robustness.values()
        ),
        "minimum_independent_formation_dates_each_fold": all(
            row["effective_dates"] >= 40 and row["effective_date_share"] >= 0.20
            for row in robustness.values()
        ),
        "moving_week_cluster_positive_probability_gte_70pct": all(
            row["moving_week_cluster_bootstrap"]["mean_r"] > 0.0
            and row["moving_week_cluster_bootstrap"][
                "probability_mean_positive"
            ]
            >= 0.70
            for row in robustness.values()
        ),
        "positive_each_fold_after_30bps": all(
            row["total_r"] > 0.0 for row in cost_stress["30"].values()
        ),
        "nonnegative_calibration_after_40bps": (
            cost_stress["40"]["calibration"]["total_r"] >= 0.0
        ),
        "leave_one_sector_out_positive_each_fold": all(
            row["all_leave_one_sector_positive"] for row in robustness.values()
        ),
        "adv_100m_stratum_positive_each_fold": all(
            row["adv_strata_total_r"]["100"] > 0.0
            for row in robustness.values()
        ),
        "capacity_and_sector_cap_stress_pass": (
            capacity_stress["tested"]
            and capacity_stress["positive_calibration_share"] >= 0.75
        ),
        "primary_formation_horizon": candidate.formation_sessions in {1, 3, 5},
        "executable_long_loser_core": candidate.diagnostic_leg == "long_loser",
    }
    compact_trades = [
        {
            "formation_date": row.formation_date.strftime("%Y-%m-%d"),
            "entry_date": row.entry_date.strftime("%Y-%m-%d"),
            "exit_date": row.exit_date.strftime("%Y-%m-%d"),
            "symbol": row.symbol,
            "issuer": row.issuer,
            "sector": row.sector,
            "residual_z": float(row.residual_z),
            "daily_score": float(row.daily_score),
            "trade_side": row.trade_side,
            "net_return": float(row.net_return),
            "r": float(row.r),
        }
        for row in trades.itertuples(index=False)
    ]
    return {
        "candidate": asdict(candidate),
        "score": score,
        "score_components": components,
        "score_raw": raw,
        "cost_stress": cost_stress,
        "portfolio_risk": {
            "basis": "fixed_size_shared_capital_mtm_v1",
            "ordinary_risk_fraction": 0.0035,
            "maximum_notional_fraction": 0.10,
            "max_drawdown_fraction": portfolio_drawdown,
            "daily_expected_shortfall_5pct": portfolio_expected_shortfall,
        },
        "fold_metrics": fold_metrics,
        "metrics": {
            "trades": len(values),
            "total_r": float(sum(values)),
            "avg_r": avg_r,
            "r_per_month": total_r_per_month,
            "trades_per_month": trades_per_month,
            "max_drawdown_r": _max_drawdown(daily_values),
            "expected_shortfall_r": expected_shortfall,
            "issuer_neutral_total_r": issuer_neutral_total,
            "top_positive_issuer_share": top_issuer_share,
            "top_positive_sector_share": top_sector_share,
            "positive_sector_count": positive_sector_count,
        },
        "gates": gates,
        "discrimination": discrimination,
        "negative_rejection": rejection,
        "robustness": robustness,
        "capacity_stress": capacity_stress,
        "qualified_discovery_candidate": all(gates.values()),
        "trades": compact_trades,
    }


_WORKER_ATLAS: pd.DataFrame | None = None


def _init_worker(atlas: pd.DataFrame) -> None:
    global _WORKER_ATLAS
    _WORKER_ATLAS = atlas


def _worker(candidate: Candidate) -> dict[str, Any]:
    if _WORKER_ATLAS is None:
        raise RuntimeError("worker atlas was not initialized")
    return evaluate_candidate(_WORKER_ATLAS, candidate)


def _apply_neighbourhood_robustness(results: list[dict[str, Any]]) -> None:
    """Attach pre-registered adjacent-horizon stability before final ranking."""

    lookup = {
        (
            str(row["candidate"].get("factor_model", "market_sector_peer")),
            tuple(row["candidate"].get("score_components", ())),
            tuple(row["candidate"].get("ranking_score_components", ())),
            str(row["candidate"]["diagnostic_leg"]),
            int(row["candidate"]["formation_sessions"]),
            int(row["candidate"]["holding_sessions"]),
            float(row["candidate"].get("minimum_failed_continuation_r", 0.0)),
            float(row["candidate"].get("minimum_score", 0.0)),
            float(row["candidate"].get("minimum_market_trend_z_20d", -8.0)),
        ): row
        for row in results
    }
    formations = (1, 3, 5)
    holdings = (1, 2, 3, 5, 7, 10)
    for row in results:
        candidate = row["candidate"]
        factor_model = str(candidate.get("factor_model", "market_sector_peer"))
        score_components = tuple(candidate.get("score_components", ()))
        ranking_components = tuple(
            candidate.get("ranking_score_components", ())
        )
        leg = str(candidate["diagnostic_leg"])
        formation = int(candidate["formation_sessions"])
        holding = int(candidate["holding_sessions"])
        failed_floor = float(candidate.get("minimum_failed_continuation_r", 0.0))
        minimum_score = float(candidate.get("minimum_score", 0.0))
        market_floor = float(candidate.get("minimum_market_trend_z_20d", -8.0))
        neighbour_keys: list[tuple[Any, ...]] = []
        if formation in formations:
            formation_index = formations.index(formation)
            for offset in (-1, 1):
                adjacent = formation_index + offset
                if 0 <= adjacent < len(formations):
                    neighbour_keys.append(
                        (
                            factor_model,
                            score_components,
                            ranking_components,
                            leg,
                            formations[adjacent],
                            holding,
                            failed_floor,
                            minimum_score,
                            market_floor,
                        )
                    )
        holding_index = holdings.index(holding)
        for offset in (-1, 1):
            adjacent = holding_index + offset
            if 0 <= adjacent < len(holdings):
                neighbour_keys.append(
                    (
                        factor_model,
                        score_components,
                        ranking_components,
                        leg,
                        formation,
                        holdings[adjacent],
                        failed_floor,
                        minimum_score,
                        market_floor,
                    )
                )
        threshold_grid = (0.20, 0.30, 0.40)
        if failed_floor in threshold_grid:
            threshold_index = threshold_grid.index(failed_floor)
            for offset in (-1, 1):
                adjacent = threshold_index + offset
                if 0 <= adjacent < len(threshold_grid):
                    neighbour_keys.append(
                        (
                            factor_model,
                            score_components,
                            ranking_components,
                            leg,
                            formation,
                            holding,
                            threshold_grid[adjacent],
                            minimum_score,
                            market_floor,
                        )
                    )
        neighbours = [lookup[key] for key in neighbour_keys if key in lookup]
        passing = [
            neighbour
            for neighbour in neighbours
            if float(neighbour["fold_metrics"]["calibration"]["avg_r"]) > 0.0
            and float(neighbour["fold_metrics"]["calibration"]["profit_factor"]) > 1.0
        ]
        share = len(passing) / len(neighbours) if neighbours else 0.0
        thirty_pass = all(
            float(value["total_r"]) > 0.0
            for value in row["cost_stress"]["30"].values()
        )
        forty_pass = float(
            row["cost_stress"]["40"]["calibration"]["total_r"]
        ) >= 0.0
        cost_strength = 0.50 * float(thirty_pass) + 0.50 * float(forty_pass)
        pass_rate = 0.50 * cost_strength + 0.50 * share
        raw_value = pass_rate
        name = "cost_and_neighbourhood_robustness"
        row["score_raw"][name] = raw_value
        row["score_components"][name] = raw_value
        row["score"] = sum(
            SCORE_SPEC[component]["weight"] * row["score_components"][component]
            for component in SCORE_SPEC
        )
        row["neighbourhood"] = {
            "neighbour_ids": [item["candidate"]["candidate_id"] for item in neighbours],
            "positive_calibration_neighbour_ids": [
                item["candidate"]["candidate_id"] for item in passing
            ],
            "positive_calibration_share": share,
            "base_cost_bps": float(candidate["round_trip_cost_bps"]),
            "thirty_bps_positive_each_fold": thirty_pass,
            "forty_bps_nonnegative_calibration": forty_pass,
            "robustness_pass_rate": raw_value,
        }
        row["gates"]["positive_neighbourhood_share_gte_50pct"] = share >= 0.50
        row["qualified_discovery_candidate"] = all(row["gates"].values())


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.max_workers != 2:
        raise ValueError("this registered discovery run requires max_workers=2")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "progress.json",
        {
            "status": "loading_broad_daily_panel",
            "max_workers": 2,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "optimizer_class": "non_promotable_price_volume_residual_discovery",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        output / "background_status.json",
        {
            "status": "running_loading_broad_daily_panel",
            "runner_pid": os.getpid(),
            "max_workers": 2,
            "optimizer_started": True,
            "optimizer_class": "non_promotable_price_volume_residual_discovery",
            "representative_reversion_baseline_eligible": False,
            "promotion_eligible": False,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    close, open_, high, low, volume, sector_by_symbol, paths = _load_daily_panel(
        Path(args.data_dir).resolve()
    )
    fingerprint, fingerprint_rows = _selection_data_fingerprint(
        close, open_, high, low, volume, paths
    )
    integrity = _price_data_integrity(
        close, open_, high, low, volume, sector_by_symbol
    )
    _write_json(
        output / "phase_0_price_data_integrity.json",
        {
            **integrity,
            "data_fingerprint": fingerprint,
            "fingerprinted_inputs": fingerprint_rows,
            "window": {"start": DISCOVERY_START, "end": CALIBRATION_END},
            "locked_validation_accessed": False,
            "holdout_accessed": False,
        },
    )
    if not integrity["passed_structural_checks"]:
        raise RuntimeError("price data failed structural integrity checks")
    _write_json(
        output / "progress.json",
        {
            "status": "building_causal_market_sector_peer_residual_atlas",
            "max_workers": 2,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    atlas = build_opportunity_atlas(
        close, open_, high, low, volume, sector_by_symbol
    )
    _write_json(
        output / "phase_1_opportunity_atlas_summary.json",
        {
            "status": "complete",
            "rows": len(atlas),
            "symbols": int(atlas["symbol"].nunique()),
            "issuers": int(atlas["issuer"].nunique()),
            "sectors": int(atlas["sector"].nunique()),
            "window": {"start": DISCOVERY_START, "end": CALIBRATION_END},
            "factor_model": "causal_rolling_spy_sector_correlated_peer_ridge_v2",
            "peer_model": "prior_return_correlation_within_sector_top5_rebalanced_21_sessions",
            "formation_horizons_sessions": [1, 3, 5, 20],
            "primary_formation_horizons_sessions": [1, 3, 5],
            "monthly_control_formation_horizon_sessions": 20,
            "diagnostic_legs": [
                "long_loser",
                "short_winner",
                "dollar_neutral_spread",
            ],
            "score_components": [
                "residual_extremeness",
                "shock_freshness",
                "price_rejection_recovery",
                "volume_transition",
                "volume_exhaustion_quality",
                "regime_execution_quality",
                "failed_continuation",
            ],
            "entry_clock": "formation_close_then_next_session_open",
            "forward_horizons_sessions": [1, 2, 3, 5, 7, 10],
            "round_trip_cost_bps": 20.0,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "data_fingerprint": fingerprint,
        },
    )
    _write_json(
        output / "progress.json",
        {
            "status": "running_bounded_candidate_search",
            "candidate_count": len(registered_candidates()),
            "max_workers": 2,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        output / "background_status.json",
        {
            "status": "running_bounded_candidate_search",
            "runner_pid": os.getpid(),
            "candidate_count": len(registered_candidates()),
            "max_workers": 2,
            "optimizer_started": True,
            "optimizer_class": "non_promotable_price_volume_residual_discovery",
            "representative_reversion_baseline_eligible": False,
            "promotion_eligible": False,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=_init_worker,
        initargs=(atlas,),
    ) as pool:
        results = list(pool.map(_worker, registered_candidates()))
    _apply_neighbourhood_robustness(results)
    results.sort(key=lambda row: (-float(row["score"]), row["candidate"]["candidate_id"]))
    qualified = [row for row in results if row["qualified_discovery_candidate"]]
    winner = qualified[0] if qualified else results[0]
    registry = [{key: value for key, value in row.items() if key != "trades"} for row in results]
    _write_json(output / "phase_2_candidate_registry.json", registry)
    _write_json(output / "best_diagnostic_candidate.json", winner)
    _write_json(
        output / "run_summary.json",
        {
            "status": "complete_discovery_only",
            "optimizer_started": True,
            "optimizer_completed": True,
            "optimizer_class": "non_promotable_price_volume_residual_discovery",
            "representative_reversion_baseline_eligible": False,
            "promotion_eligible": False,
            "promotion_blockers": [
                "unknown legacy price acquisition provenance",
                "no certified causal historical/delisted universe inventory",
                "no certified corporate-action-consistent price basis",
                "no historical/live price-volume parity certificate",
                "shared-capital source execution and entry-delivery phases remain incomplete",
            ],
            "candidate_count": len(results),
            "qualified_discovery_candidate_count": len(qualified),
            "best_candidate_id": winner["candidate"]["candidate_id"],
            "best_score": winner["score"],
            "best_metrics": winner["metrics"],
            "best_fold_metrics": winner["fold_metrics"],
            "best_gates": winner["gates"],
            "score_spec": SCORE_SPEC,
            "max_workers": 2,
            "window": {"start": DISCOVERY_START, "end": CALIBRATION_END},
            "locked_validation_start": "2025-08-01",
            "locked_validation_accessed": False,
            "holdout_start": HOLDOUT_START,
            "holdout_accessed": False,
            "data_fingerprint": fingerprint,
            "fingerprinted_input_count": len(fingerprint_rows),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        output / "progress.json",
        {
            "status": "complete_discovery_only",
            "optimizer_started": True,
            "optimizer_completed": True,
            "best_candidate_id": winner["candidate"]["candidate_id"],
            "max_workers": 2,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        output / "background_status.json",
        {
            "status": "complete_discovery_only",
            "runner_pid": os.getpid(),
            "optimizer_started": True,
            "optimizer_completed": True,
            "optimizer_class": "non_promotable_price_volume_residual_discovery",
            "representative_reversion_baseline_eligible": False,
            "promotion_eligible": False,
            "best_candidate_id": winner["candidate"]["candidate_id"],
            "best_metrics": winner["metrics"],
            "qualified_discovery_candidate_count": len(qualified),
            "max_workers": 2,
            "locked_validation_accessed": False,
            "holdout_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        parsed = _args()
        failed_output = Path(parsed.output_dir).resolve()
        _write_json(
            failed_output / "background_status.json",
            {
                "status": "failed_non_promotable_residual_discovery",
                "runner_pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "max_workers": parsed.max_workers,
                "representative_reversion_baseline_eligible": False,
                "promotion_eligible": False,
                "locked_validation_accessed": False,
                "holdout_accessed": False,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
