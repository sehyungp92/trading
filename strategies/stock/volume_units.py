"""Single source of truth for IBKR US-equity volume units.

``reqHistoricalData(whatToShow="TRADES")`` returns US equity volume in
100-share lots, not shares.  Both the live research generators and the
backtest replay bundle are fed from that same API, so every array and every
``Bar.volume`` in the stock family carries lot units.

Verified against the stored bundle (2025-06-02 daily bars, x100):
AAPL 493,607 -> 49.4m shares, NVDA 1,298,411 -> 130m, META 200,298 -> 20.0m,
JNJ 32,838 -> 3.3m -- all matching published share volume.  Intraday 30m bars
sum to the same daily figure (ratio 0.5-1.5, ETH included), so they carry the
identical lot scale.

Consumers fall into three classes:

* **ratios** (RVOL, volume climax, expected-vs-actual) divide volume by volume
  and are scale-invariant -- they must NOT be converted;
* **dollar values** (ADV20, flow proxies) -- use :func:`dollar_volume`;
* **share counts** (participation caps, max_qty) -- use :func:`to_shares`.

Raw arrays and ``Bar.volume`` deliberately stay on the native IBKR lot scale so
that stored research artifacts keep a single, stable convention; conversion
happens at the point where volume acquires dollar or share semantics.
"""

from __future__ import annotations

from typing import TypeVar

__all__ = ["IBKR_SHARE_VOLUME_MULTIPLIER", "to_shares", "dollar_volume"]

IBKR_SHARE_VOLUME_MULTIPLIER = 100.0

_T = TypeVar("_T")


def to_shares(volume: _T) -> _T:
    """Convert IBKR lot-scale *volume* into a share count.

    Accepts scalars or numpy arrays (arithmetic is elementwise either way).
    """
    return volume * IBKR_SHARE_VOLUME_MULTIPLIER


def dollar_volume(closes: _T, volumes: _T) -> _T:
    """Per-bar traded dollar value from lot-scale *volumes*."""
    return closes * volumes * IBKR_SHARE_VOLUME_MULTIPLIER
