"""Experiment definitions and priority queue for auto backtesting.

~197 experiments across 5 strategies:
  - Priority 1: Ablation (54)
  - Priority 2: Parameter sweeps (99)
  - Priority 3: Interactions (18)
  - Priority 4: Portfolio-level (26)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    id: str                       # e.g. "abl_atrss_stall_exit"
    type: str                     # ABLATION | PARAM_SWEEP | INTERACTION | PORTFOLIO
    strategy: str                 # "atrss" | "helix" | "breakout" | "s5_pb" | "s5_dual" | "portfolio"
    description: str
    hypothesis: str
    priority: int                 # 1=highest
    mutations: dict = field(default_factory=dict)


def build_experiment_queue(strategy_filter: str = "all") -> list[Experiment]:
    """Build priority-ordered experiment queue.

    Args:
        strategy_filter: "atrss", "helix", "breakout", "s5_pb", "s5_dual", "portfolio", or "all"
    """
    experiments: list[Experiment] = []
    _all = strategy_filter == "all"

    # Priority 1: Ablations
    if _all or strategy_filter == "atrss":
        experiments.extend(_atrss_ablation())
    if _all or strategy_filter == "helix":
        experiments.extend(_helix_ablation())
    if _all or strategy_filter == "breakout":
        experiments.extend(_breakout_ablation())
    if _all or strategy_filter == "s5_pb":
        experiments.extend(_s5_pb_ablation())
    if _all or strategy_filter == "s5_dual":
        experiments.extend(_s5_dual_ablation())

    # Priority 2: Param sweeps
    if _all or strategy_filter == "atrss":
        experiments.extend(_atrss_param_sweeps())
    if _all or strategy_filter == "helix":
        experiments.extend(_helix_param_sweeps())
    if _all or strategy_filter == "breakout":
        experiments.extend(_breakout_param_sweeps())
    if _all or strategy_filter in ("s5_pb", "s5_dual"):
        s5_sweeps = _s5_param_sweeps()
        if not _all and strategy_filter in ("s5_pb", "s5_dual"):
            s5_sweeps = [e for e in s5_sweeps if e.strategy == strategy_filter]
        experiments.extend(s5_sweeps)

    # Priority 3: Interactions
    if _all or strategy_filter == "atrss":
        experiments.extend(_atrss_interactions())
    if _all or strategy_filter == "helix":
        experiments.extend(_helix_interactions())
    if _all or strategy_filter == "breakout":
        experiments.extend(_breakout_interactions())

    # Priority 4: Portfolio
    if _all or strategy_filter == "portfolio":
        experiments.extend(_portfolio_experiments())

    return sorted(experiments, key=lambda e: (e.priority, e.id))


# ---------------------------------------------------------------------------
# Priority 1: Ablation experiments
# ---------------------------------------------------------------------------

def _atrss_ablation() -> list[Experiment]:
    """ATRSS: 15 wired + 2 unwired = 17 experiments."""
    E = Experiment
    return [
        # Priority 1a — key weaknesses
        E("abl_atrss_stall_exit", "ABLATION", "atrss",
          "Disable stall exit", "May cut winners short before momentum develops",
          1, {"flags.stall_exit": False}),
        E("abl_atrss_early_stall", "ABLATION", "atrss",
          "Disable early stall exit", "Premature partial exit from trades needing time",
          1, {"flags.early_stall_exit": False}),
        E("abl_atrss_time_decay", "ABLATION", "atrss",
          "Disable time decay", "MAX_HOLD_HOURS may flatten trades before breakout",
          1, {"flags.time_decay": False}),
        E("abl_atrss_breakout_entries", "ABLATION", "atrss",
          "Disable breakout entries", "Is Candidate B net positive vs pure pullback?",
          1, {"flags.breakout_entries": False}),

        # Priority 1b — filters
        E("abl_atrss_momentum", "ABLATION", "atrss",
          "Disable momentum filter", "May be too strict on breakout/reverse entries",
          1, {"flags.momentum_filter": False}),
        E("abl_atrss_conviction", "ABLATION", "atrss",
          "Disable conviction gating", "SCORE_REVERSE_MIN may reject valid reverses",
          1, {"flags.conviction_gating": False}),
        E("abl_atrss_fast_confirm", "ABLATION", "atrss",
          "Disable fast confirm", "FAST_CONFIRM_SCORE may delay entries",
          1, {"flags.fast_confirm": False}),
        E("abl_atrss_reset_req", "ABLATION", "atrss",
          "Disable reset requirement", "EMA_pull reset gate may prevent re-entries",
          1, {"flags.reset_requirement": False}),
        E("abl_atrss_voucher", "ABLATION", "atrss",
          "Disable voucher system", "Breakout arm TTL may expire before optimal trigger",
          1, {"flags.voucher_system": False}),
        E("abl_atrss_cooldown", "ABLATION", "atrss",
          "Disable cooldown", "Time gate between same-dir trades",
          1, {"flags.cooldown": False}),

        # Priority 1c — secondary
        E("abl_atrss_slippage_abort", "ABLATION", "atrss",
          "Disable slippage abort", "Bad fill rejection may be too conservative",
          1, {"flags.slippage_abort": False}),
        E("abl_atrss_short_safety", "ABLATION", "atrss",
          "Disable short safety", "Short filter may block valid QQQ/GLD shorts",
          1, {"flags.short_safety": False}),
        E("abl_atrss_addon_a", "ABLATION", "atrss",
          "Disable add-on A", "Pullback add-on at +1.5R MFE",
          1, {"flags.addon_a": False}),
        E("abl_atrss_addon_b", "ABLATION", "atrss",
          "Disable add-on B", "Continuation add-on at +2R in STRONG_TREND",
          1, {"flags.addon_b": False}),
        E("abl_atrss_quality_gate", "ABLATION", "atrss",
          "Enable quality gate", "Currently disabled — test if enabling helps",
          1, {"flags.quality_gate": True}),

        # Priority 1d — unwired (expect delta=0)
        E("abl_atrss_prior_high", "ABLATION", "atrss",
          "Disable prior high confirm", "UNWIRED: flag not checked in engine",
          1, {"flags.prior_high_confirm": False}),
        E("abl_atrss_hysteresis", "ABLATION", "atrss",
          "Disable hysteresis gap", "UNWIRED: flag not checked in engine",
          1, {"flags.hysteresis_gap": False}),
    ]


def _helix_ablation() -> list[Experiment]:
    """Helix: 10 wired + 3 unwired = 13 experiments."""
    E = Experiment
    return [
        # Priority 1a — signal class value
        E("abl_helix_class_c", "ABLATION", "helix",
          "Disable Class C", "Classic divergence reversal may drag with false reversals",
          1, {"flags.disable_class_c": True}),
        E("abl_helix_class_d", "ABLATION", "helix",
          "Disable Class D", "1H momentum-only is weakest class — may generate noise",
          1, {"flags.disable_class_d": True}),
        E("abl_helix_class_a", "ABLATION", "helix",
          "Disable Class A", "4H hidden div is highest quality — delta should be large negative",
          1, {"flags.disable_class_a": True}),
        E("abl_helix_class_b", "ABLATION", "helix",
          "Disable Class B", "1H hidden div — higher frequency but lower per-trade edge",
          1, {"flags.disable_class_b": True}),

        # Priority 1b — position management
        E("abl_helix_chandelier", "ABLATION", "helix",
          "Disable chandelier trailing", "1.5 ATR mult may be too tight, cutting runners",
          1, {"flags.disable_chandelier_trailing": True}),
        E("abl_helix_partial_2p5r", "ABLATION", "helix",
          "Disable partial at +2.5R", "Taking partial may cap best trades",
          1, {"flags.disable_partial_2p5r": True}),
        E("abl_helix_partial_5r", "ABLATION", "helix",
          "Disable partial at +5R", "Only affects big winners",
          1, {"flags.disable_partial_5r": True}),
        E("abl_helix_addons", "ABLATION", "helix",
          "Disable add-ons", "4H/1H add-ons may increase size at wrong time",
          1, {"flags.disable_add_ons": True}),
        E("abl_helix_circuit_breaker", "ABLATION", "helix",
          "Disable circuit breaker", "Consecutive loss halving may be too aggressive",
          1, {"flags.disable_circuit_breaker": True}),

        # Priority 1c
        E("abl_helix_corridor_cap", "ABLATION", "helix",
          "Disable corridor cap", "Entry-to-stop cap may reject wide-stop setups",
          1, {"flags.disable_corridor_cap": True}),

        # Priority 1d — unwired
        E("abl_helix_spread_gate", "ABLATION", "helix",
          "Disable spread gate", "UNWIRED: flag not checked in engine",
          1, {"flags.disable_spread_gate": True}),
        E("abl_helix_basket_rule", "ABLATION", "helix",
          "Disable basket rule", "UNWIRED: flag not checked in engine",
          1, {"flags.disable_basket_rule": True}),
        E("abl_helix_extreme_vol", "ABLATION", "helix",
          "Disable extreme vol gate", "UNWIRED: flag not checked in engine",
          1, {"flags.disable_extreme_vol_gate": True}),
    ]


def _breakout_ablation() -> list[Experiment]:
    """Breakout: 15 wired + 2 unwired = 17 experiments."""
    E = Experiment
    return [
        # Priority 1a — key weaknesses
        E("abl_brk_stale_exit", "ABLATION", "breakout",
          "Disable stale exit", "May cut winners consolidating before continuation",
          1, {"flags.disable_stale_exit": True}),
        E("abl_brk_chop_mode", "ABLATION", "breakout",
          "Disable chop mode", "ChopMode blocking may reject valid setups in small ETF universe",
          1, {"flags.disable_chop_mode": True}),
        E("abl_brk_regime_chop", "ABLATION", "breakout",
          "Disable regime chop block", "Double gating on chop compounds with chop_mode",
          1, {"flags.disable_regime_chop_block": True}),

        # Priority 1b
        E("abl_brk_score_threshold", "ABLATION", "breakout",
          "Disable score threshold", "Score threshold may reject valid breakouts",
          1, {"flags.disable_score_threshold": True}),
        E("abl_brk_displacement", "ABLATION", "breakout",
          "Disable displacement", "May delay entries until breakout is extended",
          1, {"flags.disable_displacement": True}),
        E("abl_brk_dirty", "ABLATION", "breakout",
          "Disable dirty state", "May block legitimate structural re-entries",
          1, {"flags.disable_dirty": True}),
        E("abl_brk_quality_reject", "ABLATION", "breakout",
          "Disable breakout quality reject", "Additional quality filter on score threshold",
          1, {"flags.disable_breakout_quality_reject": True}),

        # Priority 1c
        E("abl_brk_hard_block", "ABLATION", "breakout",
          "Disable hard block", "Hard block conditions",
          1, {"flags.disable_hard_block": True}),
        E("abl_brk_continuation", "ABLATION", "breakout",
          "Disable continuation", "Continuation entry logic",
          1, {"flags.disable_continuation": True}),
        E("abl_brk_c_continuation", "ABLATION", "breakout",
          "Disable C continuation entry", "Weaker continuation variant",
          1, {"flags.disable_c_continuation_entry": True}),
        E("abl_brk_reentry_gate", "ABLATION", "breakout",
          "Disable reentry gate", "May miss quick re-breakouts",
          1, {"flags.disable_reentry_gate": True}),
        E("abl_brk_micro_guard", "ABLATION", "breakout",
          "Disable micro guard", "Micro structure protection — conservative",
          1, {"flags.disable_micro_guard": True}),
        E("abl_brk_friction_gate", "ABLATION", "breakout",
          "Disable friction gate", "Friction/slippage gate — may reject viable ETF trades",
          1, {"flags.disable_friction_gate": True}),
        E("abl_brk_pending", "ABLATION", "breakout",
          "Disable pending entry queue", "Pending entry queue management",
          1, {"flags.disable_pending": True}),
        E("abl_brk_add_rvol", "ABLATION", "breakout",
          "Disable add RVOL acceptance", "Volume confirmation for position building",
          1, {"flags.disable_add_rvol_acceptance": True}),

        # Priority 1d — unwired
        E("abl_brk_corr_heat", "ABLATION", "breakout",
          "Disable correlation heat penalty", "UNWIRED: flag not checked in engine",
          1, {"flags.disable_corr_heat_penalty": True}),
        E("abl_brk_rvol_gating", "ABLATION", "breakout",
          "Disable entry B RVOL gating", "UNWIRED: flag not checked in engine",
          1, {"flags.disable_entry_b_rvol_gating": True}),
    ]


def _s5_pb_ablation() -> list[Experiment]:
    """S5_PB: 3 experiments via config field mutations."""
    E = Experiment
    return [
        E("abl_s5pb_no_volume", "ABLATION", "s5_pb",
          "Disable volume filter", "Volume SMA filter may reject valid pullback entries in IBIT",
          1, {"volume_filter": False}),
        E("abl_s5pb_enable_shorts", "ABLATION", "s5_pb",
          "Enable shorts", "IBIT shorts during drawdowns — currently disabled",
          1, {"shorts_enabled": True}),
        E("abl_s5pb_exit_midline", "ABLATION", "s5_pb",
          "Exit mode midline", "Faster midline exit vs trail_only",
          1, {"exit_mode": "midline"}),
    ]


def _s5_dual_ablation() -> list[Experiment]:
    """S5_DUAL: 4 experiments via config field mutations."""
    E = Experiment
    return [
        E("abl_s5dual_no_volume", "ABLATION", "s5_dual",
          "Disable volume filter", "Same volume filter concern",
          1, {"volume_filter": False}),
        E("abl_s5dual_enable_shorts", "ABLATION", "s5_dual",
          "Enable shorts", "GLD/IBIT shorts — currently disabled",
          1, {"shorts_enabled": True}),
        E("abl_s5dual_pullback_only", "ABLATION", "s5_dual",
          "Pullback only mode", "Is breakout mode adding value?",
          1, {"entry_mode": "pullback"}),
        E("abl_s5dual_exit_midline", "ABLATION", "s5_dual",
          "Exit mode midline", "Faster exit for daily-cycle strategy",
          1, {"exit_mode": "midline"}),
    ]


# ---------------------------------------------------------------------------
# Priority 2: Parameter sweeps
# ---------------------------------------------------------------------------

def _atrss_param_sweeps() -> list[Experiment]:
    """ATRSS: 28 parameter sweep experiments."""
    E = Experiment
    exps = []

    for val in [0.5, 1, 2, 4]:
        exps.append(E(f"ps_atrss_cd_strong_{val}", "PARAM_SWEEP", "atrss",
                      f"Cooldown STRONG_TREND={val}h", "STRONG_TREND re-entry timing",
                      2, {"param_overrides.cooldown_strong": val}))

    for val in [1, 2, 4, 8]:
        exps.append(E(f"ps_atrss_cd_trend_{val}", "PARAM_SWEEP", "atrss",
                      f"Cooldown TREND={val}h", "TREND re-entry timing",
                      2, {"param_overrides.cooldown_trend": val}))

    for val in [12, 24, 48, 72]:
        exps.append(E(f"ps_atrss_voucher_{val}", "PARAM_SWEEP", "atrss",
                      f"Voucher valid hours={val}", "How long breakout arm stays armed",
                      2, {"param_overrides.voucher_valid_hours": val}))

    for val in [1, 2, 3, 5]:
        exps.append(E(f"ps_atrss_confirm_{val}", "PARAM_SWEEP", "atrss",
                      f"Confirm days normal={val}", "Entry confirmation period",
                      2, {"param_overrides.confirm_days_normal": val}))

    for val in [20, 25, 30, 35]:
        exps.append(E(f"ps_atrss_adx_{val}", "PARAM_SWEEP", "atrss",
                      f"ADX strong={val}", "ADX threshold for STRONG_TREND",
                      2, {"param_overrides.adx_strong": val}))

    for val in [0, 0.3, 0.5, 0.7]:
        exps.append(E(f"ps_atrss_rev_min_{val}", "PARAM_SWEEP", "atrss",
                      f"Score reverse min={val}", "Min conviction for reverse entries",
                      2, {"param_overrides.score_reverse_min": val}))

    for val in [0.4, 0.5, 0.6, 999]:
        label = "disabled" if val == 999 else str(val)
        exps.append(E(f"ps_atrss_fast_conf_{label}", "PARAM_SWEEP", "atrss",
                      f"Fast confirm score={label}", "Fast bar confirmation threshold",
                      2, {"param_overrides.fast_confirm_score": val}))

    return exps


def _helix_param_sweeps() -> list[Experiment]:
    """Helix: 20 parameter sweep experiments."""
    E = Experiment
    exps = []

    for val in [1.0, 1.25, 1.5, 2.0]:
        exps.append(E(f"ps_helix_chand_{val}", "PARAM_SWEEP", "helix",
                      f"Chandelier mult={val}", "Trailing stop tightness",
                      2, {"param_overrides.TRAIL_PROFIT_MULT": val}))

    for val in [0.25, 0.33, 0.50]:
        exps.append(E(f"ps_helix_p2p5_frac_{val}", "PARAM_SWEEP", "helix",
                      f"Partial 2.5R frac={val}", "Partial exit fraction at +2.5R",
                      2, {"param_overrides.PARTIAL_2P5_FRAC": val}))

    # Stale bars ±20% of typical values (typical ~8-12 for 1H, ~3-4 for 4H)
    for val in [6, 8, 10, 14]:
        exps.append(E(f"ps_helix_stale_1h_{val}", "PARAM_SWEEP", "helix",
                      f"Stale 1H bars={val}", "1H stale detection sensitivity",
                      2, {"param_overrides.STALE_1H_BARS": val}))

    for val in [2, 3, 4, 5]:
        exps.append(E(f"ps_helix_stale_4h_{val}", "PARAM_SWEEP", "helix",
                      f"Stale 4H bars={val}", "4H stale detection sensitivity",
                      2, {"param_overrides.STALE_4H_BARS": val}))

    for val in [2, 3, 4, 5]:
        exps.append(E(f"ps_helix_cb_halve_{val}", "PARAM_SWEEP", "helix",
                      f"Circuit breaker halve={val}", "Consecutive stops before halving",
                      2, {"param_overrides.CONSEC_STOPS_HALVE": val}))

    # CLASS_B_MIN_ADX (±20% of typical, 3 values)
    for val in [16, 20, 24]:
        exps.append(E(f"ps_helix_b_adx_{val}", "PARAM_SWEEP", "helix",
                      f"Class B min ADX={val}", "ADX filter for Class B setups",
                      2, {"param_overrides.CLASS_B_MIN_ADX": val}))

    return exps


def _breakout_param_sweeps() -> list[Experiment]:
    """Breakout: 20 parameter sweep experiments."""
    E = Experiment
    exps = []

    for val in [1, 2, 3]:
        exps.append(E(f"ps_brk_score_{val}", "PARAM_SWEEP", "breakout",
                      f"Score threshold normal={val}", "Entry score gate",
                      2, {"param_overrides.SCORE_THRESHOLD_NORMAL": val}))

    for val in [0.5, 0.67, 0.80, 1.0]:
        exps.append(E(f"ps_brk_chop_mult_{val}", "PARAM_SWEEP", "breakout",
                      f"Chop degraded size mult={val}", "Sizing in degraded regime",
                      2, {"param_overrides.CHOP_DEGRADED_SIZE_MULT": val}))

    for val in [-3, -2, -1, 0]:
        exps.append(E(f"ps_brk_chop_stale_{val}", "PARAM_SWEEP", "breakout",
                      f"Chop degraded stale adj={val}", "Stale exit days added in degraded",
                      2, {"param_overrides.CHOP_DEGRADED_STALE_ADJ": val}))

    for val in [4, 6, 8, 12]:
        exps.append(E(f"ps_brk_pending_{val}", "PARAM_SWEEP", "breakout",
                      f"Pending max RTH hours={val}", "Pending entry queue timeout",
                      2, {"param_overrides.PENDING_MAX_RTH_HOURS": val}))

    # TP1/TP2 sweeps for aligned tier (±20% of defaults)
    for mult in [0.8, 1.0, 1.2]:
        exps.append(E(f"ps_brk_tp1_aligned_{mult}", "PARAM_SWEEP", "breakout",
                      f"TP1 R aligned ×{mult}", "Take-profit level for aligned tier",
                      2, {"param_overrides.TP1_R_ALIGNED_MULT": mult}))

    for mult in [0.8, 1.0, 1.2]:
        exps.append(E(f"ps_brk_tp2_aligned_{mult}", "PARAM_SWEEP", "breakout",
                      f"TP2 R aligned ×{mult}", "Second take-profit level",
                      2, {"param_overrides.TP2_R_ALIGNED_MULT": mult}))

    return exps


def _s5_param_sweeps() -> list[Experiment]:
    """S5: 28 experiments applied to both S5_PB and S5_DUAL."""
    E = Experiment
    exps = []

    for strategy, label in [("s5_pb", "pb"), ("s5_dual", "dual")]:
        for val in [1.5, 2.0, 2.5, 3.0]:
            exps.append(E(f"ps_s5{label}_kelt_{val}", "PARAM_SWEEP", strategy,
                          f"Keltner ATR mult={val}", "Keltner channel width",
                          2, {"kelt_atr_mult": val}))

        for val in [40, 45, 50, 55]:
            exps.append(E(f"ps_s5{label}_rsi_{val}", "PARAM_SWEEP", strategy,
                          f"RSI entry long={val}", "RSI threshold for long entries",
                          2, {"rsi_entry_long": float(val)}))

        for val in [1.5, 2.0, 2.5]:
            exps.append(E(f"ps_s5{label}_atr_stop_{val}", "PARAM_SWEEP", strategy,
                          f"ATR stop mult={val}", "Initial stop distance",
                          2, {"atr_stop_mult": val}))

        for val in [1.0, 1.5, 2.0]:
            exps.append(E(f"ps_s5{label}_trail_{val}", "PARAM_SWEEP", strategy,
                          f"Trail ATR mult={val}", "Trailing stop multiplier",
                          2, {"trail_atr_mult": val}))

    return exps


# ---------------------------------------------------------------------------
# Priority 3: Interaction experiments
# ---------------------------------------------------------------------------

def _atrss_interactions() -> list[Experiment]:
    """ATRSS: 9 interaction experiments (3 pairs × 3 combos)."""
    E = Experiment
    exps = []

    # Pair 1: stall_exit × addon_a
    exps.append(E("int_atrss_stall_addon_both", "INTERACTION", "atrss",
                  "Disable stall_exit + addon_a",
                  "Stall exit timing interacts with add-on success",
                  3, {"flags.stall_exit": False, "flags.addon_a": False}))
    exps.append(E("int_atrss_stall_only", "INTERACTION", "atrss",
                  "Disable stall_exit only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.stall_exit": False}))
    exps.append(E("int_atrss_addon_only", "INTERACTION", "atrss",
                  "Disable addon_a only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.addon_a": False}))

    # Pair 2: breakout_entries × momentum_filter
    exps.append(E("int_atrss_brk_mom_both", "INTERACTION", "atrss",
                  "Disable breakout + momentum",
                  "Disabling both changes ATRSS to pure pullback",
                  3, {"flags.breakout_entries": False, "flags.momentum_filter": False}))
    exps.append(E("int_atrss_brk_only", "INTERACTION", "atrss",
                  "Disable breakout_entries only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.breakout_entries": False}))
    exps.append(E("int_atrss_mom_only", "INTERACTION", "atrss",
                  "Disable momentum only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.momentum_filter": False}))

    # Pair 3: time_decay × early_stall_exit
    exps.append(E("int_atrss_td_es_both", "INTERACTION", "atrss",
                  "Disable time_decay + early_stall_exit",
                  "Multiple time-based exits may compound",
                  3, {"flags.time_decay": False, "flags.early_stall_exit": False}))
    exps.append(E("int_atrss_td_only", "INTERACTION", "atrss",
                  "Disable time_decay only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.time_decay": False}))
    exps.append(E("int_atrss_es_only", "INTERACTION", "atrss",
                  "Disable early_stall_exit only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.early_stall_exit": False}))

    return exps


def _helix_interactions() -> list[Experiment]:
    """Helix: 6 interaction experiments (2 pairs × 3 combos)."""
    E = Experiment
    exps = []

    # Pair 1: class_c × class_d
    exps.append(E("int_helix_cd_both", "INTERACTION", "helix",
                  "Disable Class C + Class D",
                  "Both low-quality classes — disabling both may be strictly better",
                  3, {"flags.disable_class_c": True, "flags.disable_class_d": True}))
    exps.append(E("int_helix_c_only", "INTERACTION", "helix",
                  "Disable Class C only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.disable_class_c": True}))
    exps.append(E("int_helix_d_only", "INTERACTION", "helix",
                  "Disable Class D only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.disable_class_d": True}))

    # Pair 2: chandelier × partial_2p5r
    exps.append(E("int_helix_chand_p2p5_both", "INTERACTION", "helix",
                  "Disable chandelier + partial 2.5R",
                  "Trailing stop interacts with partial profit taking",
                  3, {"flags.disable_chandelier_trailing": True, "flags.disable_partial_2p5r": True}))
    exps.append(E("int_helix_chand_only", "INTERACTION", "helix",
                  "Disable chandelier only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.disable_chandelier_trailing": True}))
    exps.append(E("int_helix_p2p5_only", "INTERACTION", "helix",
                  "Disable partial 2.5R only (interaction ref)",
                  "Reference for interaction effect",
                  3, {"flags.disable_partial_2p5r": True}))

    return exps


def _breakout_interactions() -> list[Experiment]:
    """Breakout: 3 interaction experiments (1 pair × 3 combos)."""
    E = Experiment
    return [
        E("int_brk_chop_regime_both", "INTERACTION", "breakout",
          "Disable chop_mode + regime_chop_block",
          "Double chop gating — may be redundant",
          3, {"flags.disable_chop_mode": True, "flags.disable_regime_chop_block": True}),
        E("int_brk_chop_only", "INTERACTION", "breakout",
          "Disable chop_mode only (interaction ref)",
          "Reference for interaction effect",
          3, {"flags.disable_chop_mode": True}),
        E("int_brk_regime_only", "INTERACTION", "breakout",
          "Disable regime_chop_block only (interaction ref)",
          "Reference for interaction effect",
          3, {"flags.disable_regime_chop_block": True}),
    ]


# ---------------------------------------------------------------------------
# Priority 4: Portfolio-level experiments
# ---------------------------------------------------------------------------

def _portfolio_experiments() -> list[Experiment]:
    """Portfolio: 26 experiments using UnifiedBacktestConfig."""
    E = Experiment
    exps = []

    # Cross-strategy coordination (4)
    exps.extend([
        E("pf_no_tighten", "PORTFOLIO", "portfolio",
          "Disable ATRSS→Helix tighten", "May cause premature Helix exits",
          4, {"enable_atrss_helix_tighten": False}),
        E("pf_no_boost", "PORTFOLIO", "portfolio",
          "Disable ATRSS→Helix size boost", "1.25x Helix size increases correlated risk",
          4, {"enable_atrss_helix_size_boost": False}),
        E("pf_no_coordination", "PORTFOLIO", "portfolio",
          "Disable all coordination", "Net coordination impact",
          4, {"enable_atrss_helix_tighten": False, "enable_atrss_helix_size_boost": False}),
        E("pf_boost_only", "PORTFOLIO", "portfolio",
          "Boost without tighten", "Test if boost alone is better",
          4, {"enable_atrss_helix_tighten": False, "enable_atrss_helix_size_boost": True}),
    ])

    # Heat cap sweeps (5)
    for val in [2.0, 2.5, 3.5, 4.0, 5.0]:
        exps.append(E(f"pf_heat_{val}", "PORTFOLIO", "portfolio",
                      f"Heat cap R={val}", f"Portfolio heat cap at {val}R",
                      4, {"heat_cap_R": val}))

    # Daily stop sweeps (3)
    for val in [3.0, 5.0, 6.0]:
        exps.append(E(f"pf_daily_{val}", "PORTFOLIO", "portfolio",
                      f"Portfolio daily stop R={val}", f"Daily loss limit at {val}R",
                      4, {"portfolio_daily_stop_R": val}))

    # Priority ordering (4)
    exps.extend([
        E("pf_helix_pri1", "PORTFOLIO", "portfolio",
          "Helix priority 1, Breakout 4", "Old ordering: Helix before Breakout",
          4, {"helix.priority": 1, "breakout.priority": 4}),
        E("pf_s5pb_pri0", "PORTFOLIO", "portfolio",
          "S5_PB priority 0, ATRSS 1", "S5_PB has 80% WR — first fill?",
          4, {"s5_pb.priority": 0, "atrss.priority": 1}),
        E("pf_equal_pri", "PORTFOLIO", "portfolio",
          "Equal priority", "Pure time-of-signal ordering",
          4, {"atrss.priority": 0, "s5_pb.priority": 0, "s5_dual.priority": 0,
              "breakout.priority": 0, "helix.priority": 0}),
        E("pf_expectancy", "PORTFOLIO", "portfolio",
          "Expectancy-ordered priority", "Re-order by estimated expectancy",
          4, {"atrss.priority": 0, "s5_pb.priority": 1, "helix.priority": 2,
              "breakout.priority": 3, "s5_dual.priority": 4}),
    ])

    # Risk allocation (6)
    exps.extend([
        E("pf_atrss_risk_2.0", "PORTFOLIO", "portfolio",
          "ATRSS risk 2.0%", "ATRSS highest expectancy — more allocation",
          4, {"atrss.unit_risk_pct": 0.020}),
        E("pf_atrss_risk_2.5", "PORTFOLIO", "portfolio",
          "ATRSS risk 2.5%", "Aggressive ATRSS tilt",
          4, {"atrss.unit_risk_pct": 0.025}),
        E("pf_helix_risk_1.0", "PORTFOLIO", "portfolio",
          "Helix risk 1.0%", "Helix biggest heat beneficiary — give more",
          4, {"helix.unit_risk_pct": 0.010}),
        E("pf_helix_risk_1.2", "PORTFOLIO", "portfolio",
          "Helix risk 1.2%", "Moderate Helix increase",
          4, {"helix.unit_risk_pct": 0.012}),
        E("pf_brk_heat_1.5", "PORTFOLIO", "portfolio",
          "Breakout max heat 1.5R", "Expand Breakout ceiling",
          4, {"breakout.max_heat_R": 1.5}),
        E("pf_s5_heat_2.0", "PORTFOLIO", "portfolio",
          "S5 max heat 2.0R", "S5 strategies expand from 1.5R",
          4, {"s5_pb.max_heat_R": 2.0, "s5_dual.max_heat_R": 2.0}),
    ])

    # Overlay experiments (4)
    exps.extend([
        E("pf_overlay_off", "PORTFOLIO", "portfolio",
          "Disable overlay", "Critical: overlay is 28-42% of PnL",
          4, {"overlay_enabled": False}),
        E("pf_overlay_max_70", "PORTFOLIO", "portfolio",
          "Overlay max 70%", "Reduce idle deployment from 85% to 70%",
          4, {"overlay_max_pct": 0.70}),
        E("pf_overlay_multi", "PORTFOLIO", "portfolio",
          "Overlay multi mode", "Multi-indicator vs EMA-only",
          4, {"overlay_mode": "multi"}),
        E("pf_overlay_3sym", "PORTFOLIO", "portfolio",
          "Overlay 3 symbols", "Add IBIT to overlay universe",
          4, {"overlay_symbols": ["QQQ", "GLD", "IBIT"]}),
    ])

    return exps
