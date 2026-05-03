from __future__ import annotations

from strategies.scalp._shared.levels import detect_fractal_pivots


def test_fractal_pivots_expose_confirmation_index() -> None:
    highs, lows = detect_fractal_pivots(
        highs=[1, 2, 5, 2, 1, 3, 2],
        lows=[5, 4, 3, 4, 1, 4, 5],
        left_n=2,
        right_n=2,
    )

    assert highs[0].index == 2
    assert highs[0].confirmed_at_index == 4
    assert lows[0].index == 4
    assert lows[0].confirmed_at_index == 6

