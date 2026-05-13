"""In-sample smoke grid for TPC round 8.

This is a research artifact for the round_8 output folder, not a strategy
module. It evaluates small mutation probes against the latest optimized config
using the train-only round-8 plugin and writes compact CSV/JSON reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.shared.auto.types import Experiment
from backtests.swing.auto.tpc.round8_plugin import (
    ROUND8_HARD_REJECTS,
    ROUND8_SCORING_WEIGHTS,
    Round8TPCPlugin,
    _round8_composite_score,
)

DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
ROUND_DIR = ROOT / "backtests" / "output" / "swing" / "tpc" / "round_8"
CONFIG_PATH = ROUND_DIR / "optimized_config.json"
TRAIN_END = "2025-11-01"
INITIAL_EQUITY = 100_000.0

KEY_METRICS = (
    "net_return_pct",
    "avg_r",
    "total_r",
    "total_trades",
    "trades_per_month",
    "dollar_profit_factor",
    "profit_factor",
    "max_dd_pct",
    "mfe_capture",
    "avg_mfe_r",
    "avg_mae_r",
    "never_worked_rate",
    "low_mfe_loss_rate",
    "right_then_lost_rate",
    "excellent_trades",
    "excellent_rate",
    "two_r_plus_rate",
    "top5_winner_share",
    "dollar_top5_winner_share",
    "gld_trade_count",
    "qqq_trade_count",
    "gld_excellent_rate",
    "qqq_excellent_rate",
    "additive_trade_count",
    "additive_avg_r",
    "additive_total_r",
    "pb30_plain_trade_count",
    "pb30_plain_avg_r",
)


@dataclass(frozen=True)
class Probe:
    category: str
    name: str
    mutations: dict[str, Any]
    thesis: str


def probes() -> list[Probe]:
    out: list[Probe] = []

    def add(category: str, name: str, mutations: dict[str, Any], thesis: str) -> None:
        out.append(Probe(category, name, mutations, thesis))

    # Contribution/fragility probes.
    add("ablation", "disable_pb30_additive", {"all.pb30_pullback_enabled": False}, "Measure whether the round-8 30m additive lane is genuinely additive.")
    add("ablation", "disable_type_c_reentry", {"all.type_c_enabled": False}, "Check whether real re-entry supply is carrying robust expectancy.")
    add("ablation", "disable_qqq_type_b", {"QQQ.type_b_enabled": False}, "Test QQQ shallow Type-B contribution.")
    add("ablation", "disable_qqq_shorts", {"QQQ.shorts_enabled": False}, "Check whether QQQ short expansion is adding or diluting edge.")
    add("ablation", "disable_asset_context", {"all.asset_context_enabled": False}, "Check whether context scoring adds discrimination.")
    add("ablation", "disable_mfe_giveback", {"all.mfe_giveback_trigger_r": 0.0}, "Measure exit-giveback contribution.")
    add("ablation", "disable_structure_trail", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False}, "Measure post-T1 trail contribution.")
    add("ablation", "disable_stall_exit", {"all.stall_exit_bars_15m": 0}, "Check whether stall exits improve realised R.")
    add("ablation", "disable_time_stop", {"all.max_hold_bars_15m": 0}, "Check whether the low-MFE time stop is protective or premature.")
    add("ablation", "disable_midday_avoid", {"all.avoid_windows_et": ()}, "Test if midday avoidance is filtering negative expectancy.")

    # Signal extraction and additive lane supply.
    add("signal", "pb30_relax_orderly", {"all.pb30_pullback_orderly_required": False}, "See whether orderly filter is leaving usable 30m alpha on the table.")
    add("signal", "pb30_confirm2", {"all.pb30_confirmation_required": 2}, "Demand more 15m proof on 30m pullbacks.")
    add("signal", "pb30_confirm_any", {"all.pb30_confirmation_combo_mode": "any"}, "Test broader 30m confirmation supply.")
    add("signal", "pb30_duration_5_18", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18}, "Shift 30m pullback duration shorter.")
    add("signal", "pb30_duration_7_22", {"all.pb30_pullback_min_bars_30m": 7, "all.pb30_pullback_max_bars_30m": 22}, "Shift 30m pullback duration longer.")
    add("signal", "pb30_value_hits2", {"all.pb30_type_a_value_hits_min": 2}, "Require stronger value interaction on 30m pullbacks.")
    add("signal", "pb30_fib_38_70", {"all.pb30_fib_a_low": 0.38, "all.pb30_fib_a_high": 0.70}, "Constrain 30m depth to classic value.")
    add("signal", "pb30_fib_42_72", {"all.pb30_fib_a_low": 0.42, "all.pb30_fib_a_high": 0.72}, "Test slightly deeper controlled 30m pullbacks.")
    add("signal", "pb30_ma_fast_slope", {"all.pb30_ma_transition_enabled": True, "all.pb30_ma_transition_mode": "fast_slope", "all.pb30_ma_transition_lookback_bars_30m": 6, "all.pb30_ma_transition_min_slope_atr": 0.02}, "Filter 30m pullbacks by true fast MA slope.")
    add("signal", "pb30_ma_slope_stack", {"all.pb30_ma_transition_enabled": True, "all.pb30_ma_transition_mode": "slope_and_stack", "all.pb30_ma_transition_lookback_bars_30m": 6}, "Require 30m slope plus fast/slow stack.")
    add("signal", "pb30_ma_transition", {"all.pb30_ma_transition_enabled": True, "all.pb30_ma_transition_mode": "transition", "all.pb30_ma_transition_lookback_bars_30m": 6, "all.pb30_ma_transition_window_bars_30m": 12}, "Require an actual 30m MA transition.")
    add("signal", "pb30_ema20_touch_context", {"all.pb30_ema20_context_enabled": True, "all.pb30_ema20_context_mode": "touch", "all.pb30_ema20_context_distance_atr": 0.15}, "Qualify 30m pullbacks by EMA20 value touch.")
    add("signal", "pb30_ema20_reclaim_context", {"all.pb30_ema20_context_enabled": True, "all.pb30_ema20_context_mode": "reclaim", "all.pb30_ema20_context_distance_atr": 0.15}, "Require 30m EMA20 reclaim geometry.")
    add("signal", "pb30_value_touch_market", {"all.pb30_ema20_value_touch_enabled": True, "all.pb30_ema20_value_touch_entry_order_model": "ema20_30m_value_touch_market", "all.pb30_confirmation_required": 1}, "Add a 30m EMA20 value-touch entry route.")
    add("signal", "pb30_value_touch_limit", {"all.pb30_ema20_value_touch_enabled": True, "all.pb30_ema20_value_touch_entry_order_model": "ema20_30m_value_touch_limit", "all.pb30_confirmation_required": 1}, "Try passive 30m EMA20 value-touch entries.")
    add("signal", "ema20_value_touch_market", {"all.ema20_value_touch_entry_enabled": True, "all.ema20_value_touch_entry_order_model": "ema20_value_touch_market", "all.ema20_value_touch_confirmation_required": 1, "all.ema20_value_touch_confirmation_combo_mode": "structure_or_vwap"}, "Add a standalone 1h EMA20 value-touch route.")
    add("signal", "ema20_value_touch_limit", {"all.ema20_value_touch_entry_enabled": True, "all.ema20_value_touch_entry_order_model": "ema20_value_touch_limit", "all.ema20_value_touch_confirmation_required": 1, "all.ema20_value_touch_confirmation_combo_mode": "structure_or_vwap"}, "Try passive standalone 1h EMA20 value-touch entries.")
    add("signal", "enable_gld_type_b", {"GLD.type_b_enabled": True, "GLD.type_b_requires_a_plus": True}, "Check if GLD has suppressed shallow pullback alpha.")
    add("signal", "enable_gld_type_b_not_aplus", {"GLD.type_b_enabled": True, "GLD.type_b_requires_a_plus": False}, "Broaden GLD shallow pullbacks without A+ requirement.")
    add("signal", "all_type_b_not_aplus", {"all.type_b_enabled": True, "all.type_b_requires_a_plus": False}, "Broad shallow-pullback supply stress test.")
    add("signal", "type_c_aplus_only", {"all.type_c_requires_a_plus": True}, "Check if Type-C needs higher regime quality.")
    add("signal", "second_entry_score14", {"all.second_entry_score_min": 14, "all.second_entry_min_source_score": 15.0}, "Broaden second-entry supply modestly.")
    add("signal", "second_entry_score16", {"all.second_entry_score_min": 16, "all.second_entry_min_source_score": 17.0}, "Tighten second-entry source quality.")

    # Discrimination and false-positive rejection.
    add("discrimination", "score_all_plus1", {"all.score_a_min": 11, "all.score_b_min": 10, "all.score_a_plus_min": 14}, "Raise global setup score thresholds.")
    add("discrimination", "score_all_plus2", {"all.score_a_min": 12, "all.score_b_min": 11, "all.score_a_plus_min": 15}, "Stress stricter global setup scoring.")
    add("discrimination", "score_gld_plus1", {"GLD.score_a_min": 12, "GLD.score_b_min": 11}, "Target GLD false positives.")
    add("discrimination", "score_gld_plus2", {"GLD.score_a_min": 13, "GLD.score_b_min": 12}, "Stress stricter GLD setup scoring.")
    add("discrimination", "score_qqq_plus1", {"QQQ.score_a_min": 11, "QQQ.score_b_min": 10}, "Check whether QQQ supply can be cleaned without starving it.")
    add("discrimination", "all_confirm2", {"all.confirmation_required": 2}, "Require two confirmations globally.")
    add("discrimination", "combo_structure_vwap", {"all.confirmation_combo_mode": "structure_vwap"}, "Demand both structure and VWAP confirmation.")
    add("discrimination", "combo_micro_vwap", {"all.confirmation_combo_mode": "micro_vwap"}, "Demand micro-break plus VWAP.")
    add("discrimination", "combo_preferred", {"all.confirmation_combo_mode": "preferred"}, "Demand the preferred confirmation stack.")
    add("discrimination", "require_vwap", {"all.require_vwap_confirmation": True}, "Probe VWAP as a hard filter.")
    add("discrimination", "require_structure", {"all.require_structure_confirmation": True}, "Probe market structure as a hard filter.")
    add("discrimination", "require_micro_break", {"all.require_micro_break_confirmation": True}, "Probe micro-break as a hard filter.")
    add("discrimination", "type_a_value_hits2", {"all.type_a_value_hits_min": 2}, "Require more value hits for Type-A.")
    add("discrimination", "daily_room_all_25", {"all.daily_room_min_r": 2.5}, "Demand more daily room globally.")
    add("discrimination", "daily_room_all_30", {"all.daily_room_min_r": 3.0}, "Stress daily-room filter globally.")
    add("discrimination", "daily_room_gld_35", {"GLD.daily_room_min_r": 3.5}, "Tighten GLD room.")
    add("discrimination", "daily_room_qqq_20", {"QQQ.daily_room_min_r": 2.0}, "Tighten QQQ room modestly.")
    add("discrimination", "max_extension_all_175", {"all.max_extension_atr_mult": 1.75}, "Reject late/extended entries globally.")
    add("discrimination", "max_extension_gld_125", {"GLD.max_extension_atr_mult": 1.25}, "Reject extended GLD entries.")
    add("discrimination", "max_extension_qqq_175", {"QQQ.max_extension_atr_mult": 1.75}, "Reject extended QQQ entries.")
    add("discrimination", "ma100_slope_all_008", {"all.min_ma100_slope_atr_4h": 0.08}, "Demand stronger 4h slow MA slope.")
    add("discrimination", "ma50_slope_all_004", {"all.min_ma50_slope_atr_4h": 0.04}, "Add 4h fast MA slope filter.")
    add("discrimination", "qqq_require_di_alignment", {"QQQ.require_di_alignment": True}, "Check if QQQ needs DI alignment.")
    add("discrimination", "gld_ma_slopes_strict", {"GLD.min_ma50_slope_atr_4h": 0.06, "GLD.min_ma100_slope_atr_4h": 0.08}, "Tighten GLD trend quality.")
    add("discrimination", "pullback_orderly_all", {"all.pullback_orderly_required": True}, "Require orderly 1h pullbacks as well as PB30.")
    add("discrimination", "pullback_volume_contract_09", {"all.pullback_volume_contract_max": 0.9}, "Require pullback volume contraction.")
    add("discrimination", "asset_context_min_00", {"all.asset_context_enabled": True, "all.asset_context_min_score": 0.0}, "Raise context permission floor to neutral.")
    add("discrimination", "asset_context_min_02", {"all.asset_context_enabled": True, "all.asset_context_min_score": 0.2}, "Stress positive context permission.")

    # Entry model and fill quality.
    add("entry", "entry_market_next_bar", {"all.entry_order_model": "market_next_bar", "GLD.entry_order_model": "market_next_bar"}, "Test simple next-bar market entries against structure-stop baseline.")
    add("entry", "entry_structure_stop_market", {"all.entry_order_model": "structure_stop_market", "GLD.entry_order_model": "structure_stop_market"}, "Use stop trigger without limit cap.")
    add("entry", "entry_adaptive_structure_stop", {"all.entry_order_model": "adaptive_structure_stop", "GLD.entry_order_model": "adaptive_structure_stop"}, "Adapt stop-limit width to signal bar range.")
    add("entry", "entry_vwap_retest_limit", {"all.entry_order_model": "vwap_retest_limit", "GLD.entry_order_model": "vwap_retest_limit"}, "Try confirmation-filtered VWAP retest entries.")
    add("entry", "entry_midpoint_retest_limit", {"all.entry_order_model": "midpoint_retest_limit", "GLD.entry_order_model": "midpoint_retest_limit"}, "Try confirmation-bar midpoint retest entries.")
    add("entry", "gld_structure_stop", {"GLD.entry_order_model": "structure_stop"}, "Check whether GLD should also require structure-stop activation.")
    add("entry", "qqq_market_next_bar", {"QQQ.entry_order_model": "market_next_bar"}, "Check whether QQQ structure-stop is too selective.")
    add("entry", "entry_ttl_2h", {"all.entry_order_ttl_hours": 2.0}, "Shorten pending structure-stop TTL.")
    add("entry", "entry_ttl_6h", {"all.entry_order_ttl_hours": 6.0}, "Lengthen pending structure-stop TTL modestly.")
    add("entry", "entry_limit_004", {"all.entry_stop_limit_atr_mult": 0.04}, "Tighten stop-limit slippage cap.")
    add("entry", "entry_limit_012", {"all.entry_stop_limit_atr_mult": 0.12}, "Relax stop-limit slippage cap.")
    add("entry", "entry_limit_020", {"all.entry_stop_limit_atr_mult": 0.20}, "Stress wider stop-limit cap.")
    add("entry", "pb30_entry_market", {"all.pb30_entry_order_model": "market_next_bar"}, "Force PB30 lane to market entry.")
    add("entry", "pb30_entry_structure_stop", {"all.pb30_entry_order_model": "structure_stop"}, "Force PB30 lane to structure-stop entry.")
    add("entry", "pb30_entry_vwap_retest", {"all.pb30_entry_order_model": "vwap_retest_limit"}, "Use PB30 confirmation as permission for VWAP retest.")

    # Trade management, risk, and exits.
    add("management", "t2_175", {"all.t2_r": 1.75}, "Take T2 earlier to increase conversion.")
    add("management", "t2_225", {"all.t2_r": 2.25}, "Slightly extend T2.")
    add("management", "t2_250", {"all.t2_r": 2.50}, "Extend T2 if MFE supports it.")
    add("management", "t2_300", {"all.t2_r": 3.00}, "Stress train-only larger target.")
    add("management", "qqq_t1_080", {"QQQ.t1_r": 0.80}, "Improve QQQ first-target conversion.")
    add("management", "qqq_t1_100", {"QQQ.t1_r": 1.00}, "Delay QQQ T1 modestly.")
    add("management", "gld_t1_100", {"GLD.t1_r": 1.00}, "Improve GLD first-target conversion.")
    add("management", "gld_t1_120", {"GLD.t1_r": 1.20}, "Delay GLD T1 modestly.")
    add("management", "t1_stop_025", {"all.t1_stop_r": 0.25}, "Reduce post-T1 stop lock.")
    add("management", "t1_stop_050", {"all.t1_stop_r": 0.50}, "Increase post-T1 stop lock.")
    add("management", "t1_stop_060", {"all.t1_stop_r": 0.60}, "Stress more aggressive post-T1 locking.")
    add("management", "qqq_partial_060", {"QQQ.t1_partial_pct": 0.60}, "Let QQQ runners keep more size.")
    add("management", "qqq_partial_080", {"QQQ.t1_partial_pct": 0.80}, "Realise more QQQ size at T1.")
    add("management", "mfe_giveback_175_04_05", {"all.mfe_giveback_trigger_r": 1.75, "all.mfe_giveback_retain_frac": 0.4, "all.mfe_giveback_lock_r": 0.5, "all.mfe_giveback_after_t1_only": False}, "Tighten giveback trigger.")
    add("management", "mfe_giveback_225_045_075", {"all.mfe_giveback_trigger_r": 2.25, "all.mfe_giveback_retain_frac": 0.45, "all.mfe_giveback_lock_r": 0.75, "all.mfe_giveback_after_t1_only": False}, "Let winners breathe but lock more.")
    add("management", "mfe_giveback_250_050_075", {"all.mfe_giveback_trigger_r": 2.50, "all.mfe_giveback_retain_frac": 0.50, "all.mfe_giveback_lock_r": 0.75, "all.mfe_giveback_after_t1_only": False}, "Stress looser giveback.")
    add("management", "mfe_giveback_after_t1_only", {"all.mfe_giveback_after_t1_only": True}, "Avoid premature giveback flattening before T1.")
    add("management", "trail_after_t1_4", {"all.trail_after_t1_30m_bars": 4, "all.trail_use_vwap_after_t1": True}, "Tighten structure/VWAP trail.")
    add("management", "trail_after_t1_6", {"all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "Slightly tighten structure/VWAP trail.")
    add("management", "trail_after_t1_12", {"all.trail_after_t1_30m_bars": 12, "all.trail_use_vwap_after_t1": True}, "Let runners breathe longer.")
    add("management", "trail_no_vwap", {"all.trail_use_vwap_after_t1": False}, "Check whether VWAP trail is over-tight.")
    add("management", "max_hold_24", {"all.max_hold_bars_15m": 24, "all.time_stop_min_mfe_r": 0.5}, "Cut never-worked trades sooner.")
    add("management", "max_hold_48", {"all.max_hold_bars_15m": 48, "all.time_stop_min_mfe_r": 0.5}, "Give slow starters more time.")
    add("management", "time_stop_min_mfe_075", {"all.time_stop_min_mfe_r": 0.75}, "Only hold if it has shown more progress.")
    add("management", "stall_exit_40", {"all.stall_exit_bars_15m": 40, "all.stall_exit_min_mfe_r": 1.0, "all.stall_exit_max_current_r": 0.2}, "Exit stalled trades sooner.")
    add("management", "stall_exit_64", {"all.stall_exit_bars_15m": 64, "all.stall_exit_min_mfe_r": 1.0, "all.stall_exit_max_current_r": 0.2}, "Let stalled winners breathe longer.")
    add("management", "profit_floor_ladder", {"all.profit_floor_ladder": ((1.25, 0.15), (1.75, 0.50), (2.50, 1.00))}, "Replace binary giveback with a profit-floor ladder.")
    add("management", "addon_cautious", {"all.addon_enabled": True, "all.addon_trigger_r": 1.50, "all.addon_size_mult": 0.15, "all.addon_min_score": 17, "all.addon_requires_t1": True, "all.addon_require_vwap_hold": True, "all.addon_require_structure_hold": True, "all.addon_max_total_risk_pct": 0.03}, "Test small adds only after proven continuation.")
    add("management", "addon_score15", {"all.addon_enabled": True, "all.addon_trigger_r": 1.50, "all.addon_size_mult": 0.20, "all.addon_min_score": 15, "all.addon_requires_t1": True, "all.addon_require_vwap_hold": True, "all.addon_require_structure_hold": True, "all.addon_max_total_risk_pct": 0.03}, "Broaden add-ons but keep confirmation holds.")
    add("management", "risk_temper_020", {"all.max_risk_pct": 0.020, "all.max_position_notional_pct": 5.0, "all.risk_a_pct": 0.013, "all.risk_a_plus_pct": 0.020, "all.risk_b_pct": 0.0085}, "Reduce drawdown and concentration via lower risk.")
    add("management", "dynamic_risk_mild", {"all.dynamic_risk_enabled": True, "all.dynamic_risk_score_floor": 10.0, "all.dynamic_risk_score_ceiling": 17.0, "all.dynamic_risk_min_mult": 0.75, "all.dynamic_risk_max_mult": 1.25}, "Shift risk toward higher setup score.")

    # Cross-dimensional probes that target the diagnosed weaknesses directly.
    add("combo", "gld_quality_exit_t2_25", {"GLD.score_a_min": 12, "GLD.score_b_min": 11, "GLD.daily_room_min_r": 3.5, "all.t2_r": 2.50}, "Tighten GLD false positives while extending proven winners.")
    add("combo", "pb30_more_supply_tight_exit", {"all.pb30_pullback_orderly_required": False, "all.pb30_confirmation_required": 2, "all.mfe_giveback_trigger_r": 1.75}, "Recover 30m supply but demand more entry proof and faster giveback control.")
    add("combo", "entry_adaptive_t2_25", {"all.entry_order_model": "adaptive_structure_stop", "GLD.entry_order_model": "adaptive_structure_stop", "all.t2_r": 2.50}, "Combine cleaner activation with more upside capture.")
    add("combo", "strict_discrimination_keep_frequency", {"GLD.score_a_min": 12, "GLD.score_b_min": 11, "QQQ.score_a_min": 10, "QQQ.score_b_min": 9, "all.type_c_requires_a_plus": True}, "Target GLD/type-C false positives while preserving QQQ supply.")
    add("combo", "capture_ladder_t2_25", {"all.t2_r": 2.50, "all.profit_floor_ladder": ((1.25, 0.15), (1.75, 0.50), (2.50, 1.00)), "all.mfe_giveback_trigger_r": 2.25, "all.mfe_giveback_retain_frac": 0.45, "all.mfe_giveback_lock_r": 0.75}, "Improve MFE capture with target extension plus floors.")

    # Follow-up probes after the first smoke pass: combine the strongest clean
    # single-factor leads and try to tame the high-return/no-trail pocket.
    add("followup", "pb30_5_18_plus_qqq_t1_100", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00}, "Combine the best signal-duration and QQQ target leads.")
    add("followup", "pb30_5_18_plus_no_mfe_giveback", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "all.mfe_giveback_trigger_r": 0.0}, "Combine PB30 duration with removal of MFE giveback.")
    add("followup", "pb30_5_18_plus_trail6", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "Combine PB30 duration with slightly tighter post-T1 trail.")
    add("followup", "pb30_5_18_plus_qqq_ext175", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.max_extension_atr_mult": 1.75}, "Combine PB30 duration with QQQ extension rejection.")
    add("followup", "qqq_t1_100_plus_no_mfe_giveback", {"QQQ.t1_r": 1.00, "all.mfe_giveback_trigger_r": 0.0}, "Combine QQQ target lead with no MFE giveback.")
    add("followup", "qqq_t1_100_plus_trail6", {"QQQ.t1_r": 1.00, "all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "Combine QQQ target lead with tighter post-T1 trail.")
    add("followup", "top3_pb30_qqq_no_mfe", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.mfe_giveback_trigger_r": 0.0}, "Stack the three cleanest first-pass leads.")
    add("followup", "top3_pb30_qqq_trail6", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.trail_after_t1_30m_bars": 6, "all.trail_use_vwap_after_t1": True}, "Stack PB30 duration, QQQ target, and tighter trail.")
    add("followup", "top4_pb30_qqq_no_mfe_ext", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "QQQ.t1_r": 1.00, "all.mfe_giveback_trigger_r": 0.0, "QQQ.max_extension_atr_mult": 1.75}, "Stack clean leads while adding QQQ extension control.")
    add("followup", "no_trail_risk_temper", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False, "all.max_risk_pct": 0.020, "all.max_position_notional_pct": 5.0, "all.risk_a_pct": 0.013, "all.risk_a_plus_pct": 0.020, "all.risk_b_pct": 0.0085}, "Test whether no-trail upside survives lower risk and DD control.")
    add("followup", "no_trail_profit_floor", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False, "all.profit_floor_ladder": ((1.25, 0.15), (1.75, 0.50), (2.50, 1.00))}, "Replace structure trail with profit floors.")
    add("followup", "no_trail_mfe175", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False, "all.mfe_giveback_trigger_r": 1.75, "all.mfe_giveback_retain_frac": 0.4, "all.mfe_giveback_lock_r": 0.5, "all.mfe_giveback_after_t1_only": False}, "Try to tame no-trail giveback with earlier MFE control.")
    add("followup", "no_trail_qqq_t1_100", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False, "QQQ.t1_r": 1.00}, "Check if QQQ target improvement stacks with no-trail upside.")
    add("followup", "no_trail_t1_stop050", {"all.trail_after_t1_30m_bars": 0, "all.trail_use_vwap_after_t1": False, "all.t1_stop_r": 0.50}, "Use stronger post-T1 lock instead of structure trail.")
    add("followup", "pb30_value_touch_limit_plus_duration", {"all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18, "all.pb30_ema20_value_touch_enabled": True, "all.pb30_ema20_value_touch_entry_order_model": "ema20_30m_value_touch_limit"}, "Check if value-touch limit adds anything once PB30 duration is improved.")

    names = set()
    dupes = [probe.name for probe in out if probe.name in names or names.add(probe.name)]
    if dupes:
        raise ValueError(f"duplicate probe names: {dupes}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="all", help="One category, comma list, or all.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or ROUND_DIR / f"in_sample_smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize_existing(output_dir)
        return

    selected = select_probes(args.category)
    started = time.time()
    base = read_json(CONFIG_PATH)
    plugin = Round8TPCPlugin(DATA_DIR, initial_equity=INITIAL_EQUITY, max_workers=args.max_workers, end_date=TRAIN_END)
    try:
        baseline_metrics = plugin.compute_train_metrics(base)
        baseline_score = _round8_composite_score(baseline_metrics, ROUND8_HARD_REJECTS, ROUND8_SCORING_WEIGHTS)
        batch = plugin.create_evaluate_batch(
            phase=99,
            cumulative_mutations={},
            scoring_weights=ROUND8_SCORING_WEIGHTS,
            hard_rejects=ROUND8_HARD_REJECTS,
        )
        experiments = [Experiment(name=probe.name, mutations=probe.mutations) for probe in selected]
        print(
            f"[smoke] category={args.category} probes={len(experiments)} max_workers={args.max_workers} train_end={TRAIN_END}",
            flush=True,
        )
        scored = batch(experiments, base)
    finally:
        plugin.close_pool()

    by_name = {item.name: item for item in scored}
    rows = [baseline_row(baseline_metrics, baseline_score.total)]
    for probe in selected:
        item = by_name[probe.name]
        rows.append(result_row(probe, item.metrics, item.score, item.rejected, item.reject_reason, baseline_metrics, baseline_score.total))

    category_slug = safe_slug(args.category)
    write_csv(output_dir / f"smoke_results_{category_slug}.csv", rows)
    write_json(output_dir / f"smoke_results_{category_slug}.json", rows)
    write_json(
        output_dir / f"smoke_meta_{category_slug}.json",
        {
            "category": args.category,
            "output_dir": str(output_dir.resolve()),
            "config_path": str(CONFIG_PATH.resolve()),
            "train_end": TRAIN_END,
            "max_workers": args.max_workers,
            "probe_count": len(selected),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(format_category_summary(args.category, rows), flush=True)
    print(f"[smoke] wrote {output_dir.resolve()}", flush=True)


def select_probes(category: str) -> list[Probe]:
    all_probes = probes()
    if category == "all":
        return all_probes
    wanted = {item.strip() for item in category.split(",") if item.strip()}
    selected = [probe for probe in all_probes if probe.category in wanted]
    if not selected:
        known = sorted({probe.category for probe in all_probes})
        raise SystemExit(f"No probes selected for {category!r}; known categories={known}")
    return selected


def baseline_row(metrics: dict[str, float], score: float) -> dict[str, Any]:
    row = {
        "category": "baseline",
        "name": "round8_optimized_config",
        "thesis": "Latest optimized round-8 train-only baseline.",
        "score": score,
        "rejected": False,
        "reject_reason": "",
        "smoke_promotable": False,
        "objective": objective(metrics),
    }
    for key in KEY_METRICS:
        row[key] = metrics.get(key, 0.0)
        row[f"delta_{key}"] = 0.0
    return row


def result_row(
    probe: Probe,
    metrics: dict[str, float],
    score: float,
    rejected: bool,
    reject_reason: str,
    baseline: dict[str, float],
    baseline_score: float,
) -> dict[str, Any]:
    row = {
        "category": probe.category,
        "name": probe.name,
        "thesis": probe.thesis,
        "score": score,
        "delta_score": score - baseline_score,
        "rejected": bool(rejected),
        "reject_reason": reject_reason,
        "objective": objective(metrics),
        "delta_objective": objective(metrics) - objective(baseline),
        "smoke_promotable": smoke_promotable(metrics, baseline, rejected),
    }
    for key in KEY_METRICS:
        value = metrics.get(key, 0.0)
        row[key] = value
        row[f"delta_{key}"] = value - baseline.get(key, 0.0)
    return row


def smoke_promotable(metrics: dict[str, float], baseline: dict[str, float], rejected: bool) -> bool:
    if rejected:
        return False
    trade_floor = max(100.0, baseline.get("total_trades", 0.0) * 0.92)
    alpha_ok = (
        metrics.get("net_return_pct", 0.0) >= baseline.get("net_return_pct", 0.0) - 2.0
        and metrics.get("avg_r", 0.0) >= baseline.get("avg_r", 0.0) - 0.025
        and metrics.get("total_r", 0.0) >= baseline.get("total_r", 0.0) - 3.0
    )
    risk_ok = (
        metrics.get("max_dd_pct", 99.0) <= min(15.0, baseline.get("max_dd_pct", 99.0) + 0.75)
        and metrics.get("low_mfe_loss_rate", 1.0) <= baseline.get("low_mfe_loss_rate", 1.0) + 0.015
        and metrics.get("dollar_top5_winner_share", 1.0) <= baseline.get("dollar_top5_winner_share", 1.0) + 0.06
    )
    freq_ok = metrics.get("total_trades", 0.0) >= trade_floor
    return bool(alpha_ok and risk_ok and freq_ok)


def objective(metrics: dict[str, float]) -> float:
    # A transparent smoke objective, separate from the round-8 score.
    return (
        metrics.get("net_return_pct", 0.0)
        + 12.0 * metrics.get("trades_per_month", 0.0)
        + 28.0 * metrics.get("avg_r", 0.0)
        + 8.0 * metrics.get("mfe_capture", 0.0)
        - 1.6 * metrics.get("max_dd_pct", 0.0)
        - 18.0 * metrics.get("low_mfe_loss_rate", 0.0)
        - 10.0 * metrics.get("right_then_lost_rate", 0.0)
        - 6.0 * max(metrics.get("dollar_top5_winner_share", 0.0) - 0.25, 0.0)
    )


def format_category_summary(category: str, rows: list[dict[str, Any]]) -> str:
    variants = [row for row in rows if row["category"] != "baseline"]
    top_score = sorted(variants, key=lambda row: float(row["score"]), reverse=True)[:5]
    top_obj = sorted(variants, key=lambda row: float(row["delta_objective"]), reverse=True)[:5]
    kept = [row for row in variants if row.get("smoke_promotable")]
    lines = [
        f"[smoke] {category}: {len(variants)} variants, {len(kept)} passed smoke_promotable guardrails",
        "Top score:",
        *[
            (
                f"  {row['name']}: score={row['score']:.2f} dScore={row.get('delta_score', 0):+.2f} "
                f"net={row['net_return_pct']:+.2f}% dNet={row['delta_net_return_pct']:+.2f}% "
                f"avgR={row['avg_r']:+.3f} trades={row['total_trades']:.0f} DD={row['max_dd_pct']:.2f}% "
                f"lowMFE={row['low_mfe_loss_rate']:.1%} reject={row['reject_reason'] or '-'}"
            )
            for row in top_score
        ],
        "Top objective delta:",
        *[
            (
                f"  {row['name']}: dObj={row['delta_objective']:+.2f} "
                f"net={row['net_return_pct']:+.2f}% avgR={row['avg_r']:+.3f} "
                f"trades={row['total_trades']:.0f} DD={row['max_dd_pct']:.2f}%"
            )
            for row in top_obj
        ],
    ]
    return "\n".join(lines)


def summarize_existing(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("smoke_results_*.json")):
        if path.name == "smoke_results_combined.json":
            continue
        for row in read_json(path):
            if row["category"] == "baseline" and any(existing["category"] == "baseline" for existing in rows):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No smoke_results_*.json files found in {output_dir}")
    write_csv(output_dir / "smoke_results_combined.csv", rows)
    write_json(output_dir / "smoke_results_combined.json", rows)
    report = format_report(rows, output_dir)
    (output_dir / "smoke_report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[smoke] combined report: {output_dir.resolve()}", flush=True)


def format_report(rows: list[dict[str, Any]], output_dir: Path) -> str:
    baseline = next(row for row in rows if row["category"] == "baseline")
    variants = [row for row in rows if row["category"] != "baseline"]
    promotable = [row for row in variants if row.get("smoke_promotable")]
    top_score = sorted(variants, key=lambda row: float(row["score"]), reverse=True)[:12]
    top_obj = sorted(variants, key=lambda row: float(row["delta_objective"]), reverse=True)[:12]
    top_net = sorted(variants, key=lambda row: float(row["delta_net_return_pct"]), reverse=True)[:10]
    best_by_category = []
    for category in sorted({row["category"] for row in variants}):
        items = [row for row in variants if row["category"] == category]
        best_by_category.append(max(items, key=lambda row: float(row["score"])))

    lines = [
        "# TPC Round 8 In-Sample Smoke Tests",
        "",
        f"Output: `{output_dir.resolve()}`",
        f"Train-only window end: `{TRAIN_END}`",
        f"Baseline: net {baseline['net_return_pct']:+.2f}%, avgR {baseline['avg_r']:+.3f}, totalR {baseline['total_r']:+.2f}, trades {baseline['total_trades']:.0f}, trades/month {baseline['trades_per_month']:.2f}, PF$ {baseline['dollar_profit_factor']:.2f}, DD {baseline['max_dd_pct']:.2f}%, low-MFE losses {baseline['low_mfe_loss_rate']:.1%}, MFE capture {baseline['mfe_capture']:.1%}.",
        f"Variants tested: {len(variants)}. Smoke-promotable guardrail passes: {len(promotable)}.",
        "",
        "## Best By Category",
        "| Category | Variant | Score | dScore | dNet | dAvgR | Trades | DD | Low-MFE | Reject |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in best_by_category:
        lines.append(summary_table_row(row))
    lines.extend(["", "## Top Round-8 Score", "| Variant | Category | Score | dScore | dNet | dAvgR | Trades | DD | Low-MFE | Reject |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in top_score:
        lines.append(score_table_row(row))
    lines.extend(["", "## Top Smoke Objective Delta", "| Variant | Category | dObj | dNet | dAvgR | Trades | DD | Low-MFE | Reject |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in top_obj:
        lines.append(objective_table_row(row))
    lines.extend(["", "## Largest Net Return Deltas", "| Variant | Category | dNet | Net | AvgR | Trades | DD | Low-MFE | Reject |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in top_net:
        lines.append(net_table_row(row))
    if promotable:
        lines.extend(["", "## Smoke-Promotable Guardrail Passes", "| Variant | Category | Score | dNet | dAvgR | Trades | DD | Low-MFE | Thesis |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
        for row in sorted(promotable, key=lambda r: float(r["score"]), reverse=True):
            lines.append(
                f"| {row['name']} | {row['category']} | {row['score']:.2f} | {row['delta_net_return_pct']:+.2f}% | "
                f"{row['delta_avg_r']:+.3f} | {row['total_trades']:.0f} | {row['max_dd_pct']:.2f}% | "
                f"{row['low_mfe_loss_rate']:.1%} | {row['thesis']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "- This report is in-sample only by request; it is useful for smoke-screening mechanisms, not promotion.",
            "- The smoke-promotable flag requires no hard reject, preserved trade count, preserved alpha, controlled drawdown, low-MFE loss, and concentration.",
            "- Variants that improve net only by increasing concentration, increasing drawdown, or starving frequency should be treated as research leads at most.",
        ]
    )
    return "\n".join(lines) + "\n"


def summary_table_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['category']} | {row['name']} | {row['score']:.2f} | {row.get('delta_score', 0):+.2f} | "
        f"{row['delta_net_return_pct']:+.2f}% | {row['delta_avg_r']:+.3f} | {row['total_trades']:.0f} | "
        f"{row['max_dd_pct']:.2f}% | {row['low_mfe_loss_rate']:.1%} | {row['reject_reason'] or '-'} |"
    )


def score_table_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['name']} | {row['category']} | {row['score']:.2f} | {row.get('delta_score', 0):+.2f} | "
        f"{row['delta_net_return_pct']:+.2f}% | {row['delta_avg_r']:+.3f} | {row['total_trades']:.0f} | "
        f"{row['max_dd_pct']:.2f}% | {row['low_mfe_loss_rate']:.1%} | {row['reject_reason'] or '-'} |"
    )


def objective_table_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['name']} | {row['category']} | {row['delta_objective']:+.2f} | "
        f"{row['delta_net_return_pct']:+.2f}% | {row['delta_avg_r']:+.3f} | {row['total_trades']:.0f} | "
        f"{row['max_dd_pct']:.2f}% | {row['low_mfe_loss_rate']:.1%} | {row['reject_reason'] or '-'} |"
    )


def net_table_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['name']} | {row['category']} | {row['delta_net_return_pct']:+.2f}% | "
        f"{row['net_return_pct']:+.2f}% | {row['avg_r']:+.3f} | {row['total_trades']:.0f} | "
        f"{row['max_dd_pct']:.2f}% | {row['low_mfe_loss_rate']:.1%} | {row['reject_reason'] or '-'} |"
    )


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "all"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(normalize(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(normalize(rows))


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


if __name__ == "__main__":
    main()
