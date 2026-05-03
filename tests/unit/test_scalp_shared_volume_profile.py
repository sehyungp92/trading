from __future__ import annotations

from strategies.scalp._shared.volume_profile import compute_volume_profile


def test_volume_profile_expands_value_area_from_poc() -> None:
    result = compute_volume_profile(
        [100.0, 100.0, 100.25, 100.5, 100.75],
        [10, 10, 5, 4, 1],
        tick_size=0.25,
        value_area_pct=0.70,
    )

    assert result.poc == 100.0
    assert result.val == 100.0
    assert result.vah == 100.25
    assert result.total_volume == 30

