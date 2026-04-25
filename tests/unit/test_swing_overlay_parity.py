from __future__ import annotations

import numpy as np

from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import _overlay_ema
from strategies.swing.overlay.config import OverlayConfig
from strategies.swing.overlay.engine import _compute_ema


def test_overlay_config_defaults_match_unified_backtest_defaults() -> None:
    live = OverlayConfig()
    backtest = UnifiedBacktestConfig()

    assert live.enabled == backtest.overlay_enabled
    assert live.symbols == backtest.overlay_symbols
    assert live.max_equity_pct == backtest.overlay_max_pct
    assert live.ema_fast == backtest.overlay_ema_fast
    assert live.ema_slow == backtest.overlay_ema_slow
    assert live.ema_overrides == backtest.overlay_ema_overrides
    assert live.weights == backtest.overlay_weights


def test_overlay_ema_matches_backtest_reference() -> None:
    closes = np.array([100.0, 101.0, 100.5, 102.5, 103.0, 104.0, 103.5, 105.0])

    live = _compute_ema(closes, 3)
    backtest = _overlay_ema(closes, 3)

    assert np.allclose(live, backtest, equal_nan=True)
