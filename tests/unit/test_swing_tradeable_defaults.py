from pathlib import Path

from backtests.swing._aliases import install

install()

from backtests.swing.auto.experiments import build_experiment_queue
from backtests.swing.config import BacktestConfig
from backtests.swing.config_breakout import BreakoutBacktestConfig
from backtests.swing.config_brs import BRSConfig, BRS_SYMBOL_DEFAULTS
from backtests.swing.config_helix import HelixBacktestConfig
from backtests.swing.config_regime import RegimeConfig
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.optimization.breakout_param_space import _breakout_symbols
from backtests.swing.optimization.helix_param_space import _BTC_SYMS as HELIX_BTC_SYMS
from backtests.swing.optimization.helix_param_space import _helix_symbols
from backtests.swing.optimization.param_space import _BTC_SYMS as ATRSS_BTC_SYMS
from libs.config.loader import load_strategy_registry
from strategies.swing.akc_helix.config import SYMBOLS as HELIX_SYMBOLS
from strategy.config import SYMBOLS as ATRSS_SYMBOLS


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_swing_registry_excludes_s5_and_ibit() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    assert "S5_PB" not in registry.strategies
    assert "S5_DUAL" not in registry.strategies
    assert "IBIT" not in list(registry.strategies["AKC_HELIX"].symbols)


def test_live_swing_defaults_exclude_ibit() -> None:
    assert "IBIT" not in HELIX_SYMBOLS


def test_backtest_swing_defaults_exclude_ibit() -> None:
    assert "IBIT" not in BacktestConfig().symbols
    assert "IBIT" not in HelixBacktestConfig().symbols
    assert "IBIT" not in BreakoutBacktestConfig().symbols
    assert "IBIT" not in RegimeConfig().symbols
    assert "IBIT" not in BRSConfig().symbols
    assert "IBIT" not in BRS_SYMBOL_DEFAULTS
    unified = UnifiedBacktestConfig()
    assert "IBIT" not in unified.atrss_symbols
    assert "IBIT" not in unified.helix_symbols
    assert "IBIT" not in unified.breakout_symbols


def test_unified_backtest_priorities_match_live_subset() -> None:
    unified = UnifiedBacktestConfig()

    assert [
        unified.atrss.strategy_id,
        unified.breakout.strategy_id,
        unified.helix.strategy_id,
    ] == ["ATRSS", "SWING_BREAKOUT_V3", "AKC_HELIX"]
    assert unified.atrss.priority < unified.breakout.priority < unified.helix.priority


def test_swing_auto_experiments_exclude_s5_and_ibit_overlay() -> None:
    experiments = build_experiment_queue("all")

    assert {exp.strategy for exp in experiments}.issubset(
        {"atrss", "helix", "breakout", "portfolio"},
    )
    assert all("s5" not in exp.id for exp in experiments)
    assert all(
        not any(key.startswith(("s5_pb", "s5_dual")) for key in exp.mutations)
        for exp in experiments
    )
    assert all(exp.mutations.get("overlay_symbols") != ["QQQ", "GLD", "IBIT"] for exp in experiments)


def test_swing_optimization_defaults_exclude_ibit() -> None:
    assert "IBIT" not in _breakout_symbols()
    assert "IBIT" not in _helix_symbols()
    assert "IBIT" not in ATRSS_SYMBOLS
    assert "IBIT" not in HELIX_BTC_SYMS
    assert "IBIT" not in ATRSS_BTC_SYMS
