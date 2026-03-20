from backtest.config_unified import UnifiedBacktestConfig, StrategySlot
from backtest.engine.unified_portfolio_engine import load_unified_data, run_unified, print_unified_report

config = UnifiedBacktestConfig(
    initial_equity=10000,
    s5_pb=StrategySlot('S5_PB', priority=1, unit_risk_pct=0.008, max_heat_R=1.50, daily_stop_R=2.0, max_working_orders=2),
    s5_dual=StrategySlot('S5_DUAL', priority=2, unit_risk_pct=0.008, max_heat_R=1.50, daily_stop_R=2.0, max_working_orders=2),
    breakout=StrategySlot('SWING_BREAKOUT_V3', priority=3, unit_risk_pct=0.005, max_heat_R=0.65, daily_stop_R=2.0, max_working_orders=2),
    helix=StrategySlot('AKC_HELIX', priority=4, unit_risk_pct=0.005, max_heat_R=0.85, daily_stop_R=2.5, max_working_orders=4),
)
data = load_unified_data(config)
result = run_unified(data, config)
print_unified_report(result, config)
