"""Full ablation study: vanilla -> R9 in logical groups."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtests.swing._aliases import install
install()

from backtests.swing.config_brs import BRSConfig
from backtests.swing.engine.brs_portfolio_engine import load_brs_data, run_brs_synchronized
from backtests.swing.auto.brs.scoring import extract_brs_metrics
from backtests.swing.auto.brs.config_mutator import mutate_brs_config

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EQ = 10_000.0

# -----------------------------------------------------------------------
# Original spec defaults (pre-R1) -- revert EVERYTHING from current R9
# -----------------------------------------------------------------------
REVERT_TO_VANILLA = {
    # BRSConfig-level
    "adx_strong": 30,
    "peak_drop_enabled": False,
    "peak_drop_lookback": 20,
    "peak_drop_pct": -0.04,
    "bias_4h_accel_enabled": False,
    "chop_short_entry_enabled": False,
    "persistence_override_bars": 0,
    "pyramid_enabled": False,
    "size_mult_bear_trend": 1.0,
    "chop_quality_mult": 0.60,
    "persist_quality_mult_bd": 0.60,
    "bd_arm_bars": 40,
    "bd_donchian_period": 20,
    "bd_max_stop_atr": 3.0,
    "bt_volume_mult": 1.2,
    "lh_arm_bars": 40,
    "lh_swing_lookback": 10,
    "be_trigger_r": 0.5,
    "profit_floor_scale": 1.0,
    "min_hold_bars": 3,
    "stop_floor_bear_strong_mult": 1.3,
    "scale_out_enabled": False,
    "scale_out_pct": 0.50,
    "scale_out_target_r": 2.0,
    "disable_s1": True,
    # Per-symbol reverts
    "symbol_configs.QQQ.stop_buffer_atr": 0.4,
    "symbol_configs.QQQ.stop_floor_atr": 1.0,
    "symbol_configs.QQQ.base_risk_pct": 0.005,
    "symbol_configs.QQQ.chand_mult": 3.2,
    "symbol_configs.GLD.adx_on": 16,
    "symbol_configs.GLD.adx_off": 14,
    "symbol_configs.GLD.daily_mult": 2.3,
    "symbol_configs.GLD.hourly_mult": 2.9,
    "symbol_configs.GLD.chand_mult": 3.2,
    "symbol_configs.GLD.stop_buffer_atr": 0.4,
    "symbol_configs.GLD.stop_floor_atr": 1.0,
    "symbol_configs.GLD.base_risk_pct": 0.005,
    "param_overrides.extreme_vol_pct": 95,
}


def run(mutations):
    cfg = BRSConfig(initial_equity=EQ, data_dir=DATA_DIR)
    if mutations:
        cfg = mutate_brs_config(cfg, mutations)
    data = load_brs_data(cfg)
    res = run_brs_synchronized(data, cfg)
    m = extract_brs_metrics(res, EQ)
    # PnL concentration
    trades = []
    for sym, sr in res.symbol_results.items():
        for t in sr.trades:
            trades.append(getattr(t, "pnl_dollars", 0))
    trades.sort(reverse=True)
    total = sum(trades)
    top1_pct = trades[0] / total * 100 if total > 0 and trades else 0
    return m, top1_pct


def main():
    # Build scenarios as cumulative layers
    scenarios = []

    # 0) Vanilla (all reverts)
    scenarios.append(("0) Vanilla (original spec)", dict(REVERT_TO_VANILLA)))

    # 1) + Signal enablement (S1, chop_short, persistence)
    s1 = dict(REVERT_TO_VANILLA)
    s1.update({
        "disable_s1": False,
        "chop_short_entry_enabled": True,
        "persistence_override_bars": 7,
    })
    scenarios.append(("1) + Signal enablement", s1))

    # 2) + Regime detection (ADX, peak_drop, 4H accel)
    s2 = dict(s1)
    s2.update({
        "adx_strong": 25,
        "peak_drop_enabled": True,
        "peak_drop_pct": -0.02,
        "peak_drop_lookback": 16,
        "bias_4h_accel_enabled": True,
        "symbol_configs.GLD.adx_on": 14,
        "symbol_configs.GLD.adx_off": 12,
        "bt_volume_mult": 2.0,
    })
    scenarios.append(("2) + Regime detection", s2))

    # 3) + Entry tuning (arming, donchian, swing lookback)
    s3 = dict(s2)
    s3.update({
        "lh_arm_bars": 26,
        "lh_swing_lookback": 5,
        "bd_arm_bars": 24,
        "bd_donchian_period": 10,
        "bd_max_stop_atr": 4.2,
    })
    scenarios.append(("3) + Entry tuning", s3))

    # 4) + Stop/risk tuning (buffers, floors, min hold)
    s4 = dict(s3)
    s4.update({
        "symbol_configs.QQQ.stop_buffer_atr": 0.3,
        "symbol_configs.QQQ.stop_floor_atr": 0.6,
        "symbol_configs.GLD.stop_buffer_atr": 0.18,
        "symbol_configs.GLD.stop_floor_atr": 0.6,
        "stop_floor_bear_strong_mult": 1.5,
        "min_hold_bars": 5,
        "symbol_configs.QQQ.chand_mult": 2.5,
        "symbol_configs.GLD.daily_mult": 2.0,
        "symbol_configs.GLD.hourly_mult": 2.6,
        "symbol_configs.GLD.chand_mult": 3.6,
    })
    scenarios.append(("4) + Stop/risk tuning", s4))

    # 5) + Quality multipliers
    s5 = dict(s4)
    s5.update({
        "chop_quality_mult": 0.84,
        "persist_quality_mult_bd": 0.96,
    })
    scenarios.append(("5) + Quality multipliers", s5))

    # 6) + Exit management (scale-out, BE, profit floor)
    s6 = dict(s5)
    s6.update({
        "scale_out_enabled": True,
        "scale_out_pct": 0.33,
        "scale_out_target_r": 3.6,
        "be_trigger_r": 0.75,
        "profit_floor_scale": 4.1472,
        "param_overrides.extreme_vol_pct": 90,
    })
    scenarios.append(("6) + Exit management", s6))

    # 7) + Risk sizing (leverage-capped)
    s7 = dict(s6)
    s7.update({
        "symbol_configs.QQQ.base_risk_pct": 0.002,
        "symbol_configs.GLD.base_risk_pct": 0.003,
    })
    scenarios.append(("7) + Risk sizing (lev-cap)", s7))

    # 8) + Regime sizing (BT 1.3x)
    s8 = dict(s7)
    s8.update({"size_mult_bear_trend": 1.3})
    scenarios.append(("8) + BT sizing 1.3x", s8))

    # 9) + Pyramiding (0.5x) — full R9
    s9 = dict(s8)
    s9.update({"pyramid_enabled": True})
    scenarios.append(("9) + Pyramid 0.5x (full R9)", s9))

    # Run all
    hdr = f"{'Scenario':<35} {'N':>4} {'PF':>6} {'DD%':>6} {'Ret%':>7} {'Shrp':>5} {'Calmr':>7} {'Top1%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for label, muts in scenarios:
        m, top1 = run(muts)
        print(f"{label:<35} {m.total_trades:>4} {m.profit_factor:>6.2f} {m.max_dd_pct:>5.2%} {m.net_return_pct:>6.1f}% {m.sharpe:>5.2f} {m.calmar:>7.1f} {top1:>5.1f}%")


if __name__ == "__main__":
    main()
