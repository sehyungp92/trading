from __future__ import annotations

import pandas as pd
import pytest

from backtests.stock.data.price_basis import align_intraday_to_daily_price_basis


def test_align_intraday_to_daily_price_basis_corrects_clear_split_scale() -> None:
    daily = pd.DataFrame(
        {"open": [91.0], "high": [92.0], "low": [90.0], "close": [91.5], "volume": [1_000.0]},
        index=pd.DatetimeIndex(["2025-08-07T00:00:00Z"]),
    )
    intraday = pd.DataFrame(
        {
            "open": [910.0, 915.0], "high": [920.0, 925.0], "low": [905.0, 912.0],
            "close": [915.0, 920.0], "volume": [100.0, 120.0], "wap": [912.0, 918.0],
        },
        index=pd.DatetimeIndex(["2025-08-07T13:30:00Z", "2025-08-07T13:35:00Z"]),
    )

    aligned, factors = align_intraday_to_daily_price_basis(intraday, daily)

    assert factors == {pd.Timestamp("2025-08-07").date(): 10.0}
    assert aligned.iloc[0].open == pytest.approx(91.0)
    assert aligned.iloc[1].close == pytest.approx(92.0)
    assert aligned.iloc[0].volume == pytest.approx(1_000.0)
    assert aligned.iloc[0].wap == pytest.approx(91.2)


def test_align_intraday_to_daily_price_basis_leaves_normal_open_gap_untouched() -> None:
    daily = pd.DataFrame(
        {"open": [103.0], "close": [104.0]},
        index=pd.DatetimeIndex(["2025-08-07T00:00:00Z"]),
    )
    intraday = pd.DataFrame(
        {"open": [103.1], "high": [104.0], "low": [102.5], "close": [103.8], "volume": [500.0]},
        index=pd.DatetimeIndex(["2025-08-07T13:30:00Z"]),
    )

    aligned, factors = align_intraday_to_daily_price_basis(intraday, daily)

    assert factors == {}
    pd.testing.assert_frame_equal(aligned, intraday)
