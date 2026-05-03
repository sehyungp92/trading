from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .nq_contract import round_to_tick


@dataclass(frozen=True, slots=True)
class VolumeProfileResult:
    poc: float
    vah: float
    val: float
    total_volume: float
    profile: dict[float, float] = field(default_factory=dict)


def compute_volume_profile(
    prices: Iterable[float],
    volumes: Iterable[float],
    *,
    tick_size: float = 0.25,
    value_area_pct: float = 0.70,
) -> VolumeProfileResult:
    bins: dict[float, float] = defaultdict(float)
    for price, volume in zip(prices, volumes, strict=False):
        if volume <= 0:
            continue
        bins[round_to_tick(float(price), tick_size)] += float(volume)
    return _profile_from_bins(dict(bins), tick_size=tick_size, value_area_pct=value_area_pct)


def compute_volume_profile_from_ticks(
    trade_prices: Iterable[float],
    trade_sizes: Iterable[float],
    *,
    tick_size: float = 0.25,
    value_area_pct: float = 0.70,
) -> VolumeProfileResult:
    return compute_volume_profile(
        trade_prices,
        trade_sizes,
        tick_size=tick_size,
        value_area_pct=value_area_pct,
    )


def _profile_from_bins(
    profile: dict[float, float],
    *,
    tick_size: float,
    value_area_pct: float,
) -> VolumeProfileResult:
    if not profile:
        return VolumeProfileResult(poc=0.0, vah=0.0, val=0.0, total_volume=0.0, profile={})

    ordered_prices = sorted(profile)
    total_volume = float(sum(profile.values()))
    poc = max(ordered_prices, key=lambda price: (profile[price], -abs(price)))
    included = {poc}
    included_volume = profile[poc]
    target_volume = total_volume * max(0.0, min(1.0, value_area_pct))

    poc_idx = ordered_prices.index(poc)
    lower_idx = poc_idx - 1
    upper_idx = poc_idx + 1
    while included_volume < target_volume and (lower_idx >= 0 or upper_idx < len(ordered_prices)):
        lower_volume = profile[ordered_prices[lower_idx]] if lower_idx >= 0 else -1.0
        upper_volume = profile[ordered_prices[upper_idx]] if upper_idx < len(ordered_prices) else -1.0
        if upper_volume > lower_volume:
            price = ordered_prices[upper_idx]
            upper_idx += 1
        else:
            price = ordered_prices[lower_idx]
            lower_idx -= 1
        included.add(price)
        included_volume += profile[price]

    return VolumeProfileResult(
        poc=poc,
        vah=max(included),
        val=min(included),
        total_volume=total_volume,
        profile={price: profile[price] for price in ordered_prices},
    )

