"""Immutable contract for price/volume-only IARIC residual reversion.

The representative strategy is deliberately unconditional with respect to
news.  It ranks causal market/sector/peer residual dislocations and asks the
subsequent price path to distinguish normalization from continuation.  News,
earnings, historical quotes and order imbalance are neither inputs nor vetoes.

The frozen 98-name daily residual sleeve is sufficient to establish a representative
baseline.  Five-minute failed-continuation sleeves are optional and must
qualify independently; their absence cannot suppress a valid daily strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONTRACT_VERSION = "iaric_price_volume_residual_reversion_v4"
AUTHORITY_MANIFEST_VERSION = "iaric_price_volume_authority_v2"
DOWNSTREAM_EXECUTION_CONTRACT = "price_volume_residual_shared_core_v2"

DISCOVERY_START = "2024-03-25"
DISCOVERY_END = "2024-11-30"
CALIBRATION_START = "2024-12-01"
CALIBRATION_END = "2025-07-31"
LOCKED_VALIDATION_START = "2025-08-01"
LOCKED_VALIDATION_END = "2026-03-01"
HOLDOUT_START = "2026-03-02"

SELECTION_FOLDS: tuple[tuple[str, str, str], ...] = (
    ("discovery", DISCOVERY_START, DISCOVERY_END),
    ("calibration", CALIBRATION_START, CALIBRATION_END),
)
LOCKED_VALIDATION_FOLD = (
    "locked_internal_validation",
    LOCKED_VALIDATION_START,
    LOCKED_VALIDATION_END,
)

PHASE_ORDER: tuple[str, ...] = (
    "phase_0_price_data_integrity_and_parity",
    "phase_1_residual_model_and_horizon_atlas",
    "phase_2_feature_qualification_and_discrimination",
    "phase_3_selection_contract_robustness",
    "phase_4_causal_entry_delivery",
    "phase_4a_exact_screen_completion_and_pareto",
    "phase_4b_mechanism_aware_rejection_and_capacity_attribution",
    "phase_4c_two_stage_admission_and_ranking",
    "phase_5_residual_anchor_and_half_life_management",
    "phase_6_independent_sleeve_qualification_and_final_robustness",
    "phase_7_protected_integration_and_literal_ablation",
    "phase_8_selective_sector_overflow_and_displacement_quality",
    "phase_9_quality_aperture_and_discrimination",
    "phase_10_risk_and_notional_frontier",
    "phase_11_exit_capture_frontier",
    "phase_12_final_alpha_frequency_synergy",
    "phase_13_path_causal_profit_retention",
    "phase_14_capacity_neutral_alpha_recycling",
    "phase_15_final_robustness_and_target_assessment",
    "phase_16_locked_chronological_validation",
)

# Pre-registered experiment families. These are hypotheses, not grids; each
# phase freezes its winning contract before the next phase can consume it.
EXPERIMENT_REGISTRY: dict[str, dict[str, Any]] = {
    PHASE_ORDER[0]: {
        "experiments": (
            "source checksums and calibration-bounded selection views",
            "split/dividend basis and executable-price reconciliation",
            "volume units, completed-session timestamps and factor availability",
            "causal data-availability universe and historical/live adapter parity",
        ),
        "gate": "price/volume inputs are causal, internally consistent and parity-certified for each enabled sleeve",
    },
    PHASE_ORDER[1]: {
        "experiments": (
            "one, three and five-session residual formations plus a twenty-session control",
            "forward residual and executable-price paths through ten sessions",
            "long-loser executable atlas with short-winner and neutral-spread controls kept separate",
            "market-only, market-sector, market-sector-peer and direct peer-demeaned residual ablations",
        ),
        "gate": "fixed causal opportunity denominator, adequate issuer-day breadth and no locked outcomes",
    },
    PHASE_ORDER[2]: {
        "experiments": (
            "component-level sign, nonlinear response and redundancy qualification",
            "matched low-extremeness and persistent-continuation rejected cohorts",
            "minimal equal-weight discriminator versus residual-rank-only baseline",
            "within-date-sector, weekly-block, sign-flipped and component-ablation placebos",
        ),
        "gate": "a no-more-than-seven-component frozen discriminator beats matched rejected cohorts in both folds",
    },
    PHASE_ORDER[3]: {
        "experiments": (
            "moving-week and two-way date/issuer clustered bootstrap",
            "twenty, thirty and forty-basis-point cost stress",
            "ADV, residual-model, position-cap and sector-cap invariance",
            "formation-date breadth and parameter-neighbourhood stability",
        ),
        "gate": "positive calibration economics survive independent-date, cost, model and capacity stress",
    },
    PHASE_ORDER[4]: {
        "experiments": (
            "next-session open baseline and overnight versus post-open attribution",
            "causal opening schedule or first-thirty-minute VWAP implementation",
            "pre-existing resting retrace limit with missed-fill accounting",
            "completed-five-minute residual recovery confirmation on a frozen intersection",
        ),
        "gate": "net delivery value survives conservative fills, costs, misses and adverse-selection attribution",
    },
    PHASE_ORDER[5]: {
        "experiments": (
            "exact shared-core replay for every economically screened candidate",
            "Pareto dominance on return, executable frequency and MTM drawdown",
            "frozen baseline and residual-model controls regardless of approximate rank",
        ),
        "gate": "approximate diagnostics may reject but never rank or truncate exact finalists",
    },
    PHASE_ORDER[6]: {
        "experiments": (
            "score-floor rejected opportunity outcomes",
            "portfolio, sector and issuer capacity displacement",
            "same-date-sector residual-magnitude matching",
            "mechanism-aware not-applicable rejection cohorts",
        ),
        "gate": "every standardized opportunity is reconciled without claiming counterfactual portfolio PnL",
    },
    PHASE_ORDER[7]: {
        "experiments": (
            "three-component admission with two-component capacity priority",
            "single-score literal comparators",
            "shared live/replay selector with a seven-component union ceiling",
        ),
        "gate": "two-stage selector must pass the unchanged exact economic and discrimination gates",
    },
    PHASE_ORDER[8]: {
        "experiments": (
            "fifty-percent residual normalization partial",
            "full frozen residual-anchor normalization",
            "three, five, seven and ten-session half-life controls",
            "residual persistence structural-failure exit",
        ),
        "gate": "management improves causal net capture without converting reversion into unlimited carry",
    },
    PHASE_ORDER[9]: {
        "experiments": (
            "daily residual reversion",
            "five-minute residual failed continuation",
            "gap residual failed continuation",
            "frozen Round-3 trend-pullback tail control",
            "leave-one-issuer, leave-one-sector and volatility/liquidity strata",
        ),
        "gate": "each sleeve is positive independently after costs, clustering and concentration controls",
    },
    PHASE_ORDER[10]: {
        "experiments": (
            "shared-capital mark-to-market replay",
            "issuer and sector arbitration with one position per issuer",
            "protected integration against the frozen Round-3 control",
            "literal leave-one-sleeve-out ablation and cannibalization",
        ),
        "gate": "daily residual adds positive marginal value after slot conflicts, costs and MTM risk",
    },
    PHASE_ORDER[11]: {
        "experiments": (
            "inherit the exact Phase-6 twelve-position neighbour without replay",
            "one exceptional third-sector slot gated at neutral or strong score quality",
            "residual-z 1.00/1.10 overflow gates with full, three-quarter or half marginal risk",
        ),
        "gate": "selective sector overflow must improve the immutable score under exact both-fold replay; global capacity remains fixed at the already-proven twelve-position baseline",
    },
    PHASE_ORDER[12]: {
        "experiments": (
            "residual z at 1.00, 1.05 and 1.10",
            "absolute score floors at fifteen, twenty, twenty-five and thirty",
            "small pre-registered z and score-floor interactions",
        ),
        "gate": "admission aperture improves exact score while retaining both-fold discrimination and breadth",
    },
    PHASE_ORDER[13]: {
        "experiments": (
            "risk fractions from 0.25 to 0.35 percent",
            "eight, ten and twelve-percent per-position notional caps",
            "risk/notional interaction under shared capital",
        ),
        "gate": "immutable-score improvement governs selection; sub-ten-percent MTM drawdown is an explicit aspiration and tie-break, with a twelve-percent safety ceiling",
    },
    PHASE_ORDER[14]: {
        "experiments": (
            "nine versus ten-session maximum hold",
            "six, seven, eight-R and ATR-only catastrophic-stop controls",
            "late full normalization and partial-plus-full residual capture",
        ),
        "gate": "exit changes must improve exact score without weakening both-fold alpha or exceeding twelve-percent MTM drawdown",
    },
    PHASE_ORDER[15]: {
        "experiments": (
            "small score-floor or residual-z changes with five-percent risk relief",
            "bounded notional/risk interaction",
            "only if selected: overflow quality, z and marginal-risk interactions",
        ),
        "gate": "promote only exact robust immutable-score improvement, otherwise retain control; do not reopen global-capacity or blanket-sector-cap directions rejected by prior attribution",
    },
    PHASE_ORDER[16]: {
        "experiments": (
            "completed-session residual normalization arms at 0.75, 1.00 or 1.25 of the frozen dislocation",
            "coarse 0.35 or 0.50 residual giveback exits at the following open",
            "literal disabled profit-retention control",
        ),
        "gate": "path-causal profit retention must improve exact score without exceeding the twelve-percent MTM drawdown ceiling; otherwise retain the fixed/normalization control",
    },
    PHASE_ORDER[17]: {
        "experiments": (
            "same-sector replacement of a stale low-score incumbent",
            "diversifying portfolio replacement under the unchanged global cap",
            "loss-only combined replacement and a broader stale combined control",
            "literal no-replacement control plus thirty-basis-point confirmation",
        ),
        "gate": "promotion requires exact immutable-score improvement, at least twelve actual replacements and positive incremental R in each fold at twenty and thirty basis points; undersized cohorts remain diagnostic-only",
    },
    PHASE_ORDER[18]: {
        "experiments": (
            "twenty, thirty and forty-basis-point costs",
            "local z, score-floor, capacity and sector perturbations",
            "leave-one-issuer and leave-one-sector robustness",
        ),
        "gate": "every final robustness gate and the twelve-percent MTM drawdown safety ceiling pass; aspiration attainment is reported but does not block validation",
    },
    PHASE_ORDER[19]: {
        "experiments": (
            "one-shot locked internal validation",
            "shared-capital cost and capacity stress",
            "decision-stream and economic-input parity recheck",
        ),
        "gate": "positive locked marginal R with every non-negotiable gate passing",
    },
}


# The optional receipt-backed deployment preflight begins fail-closed.  The
# phased runner may independently certify the project-designated official local
# snapshot through bounded content validation and deterministic fingerprints;
# original acquisition receipts are not required for that route.
CURRENT_INPUT_AUTHORITY: dict[str, Any] = {
    "five_minute_ohlcv": False,
    "daily_ohlcv": False,
    "causal_universe_definition": False,
    "corporate_action_consistent_price_basis": False,
    "volume_unit_semantics": False,
    "completed_session_timestamps": False,
    "historical_live_price_volume_parity": False,
}


@dataclass(frozen=True)
class SleeveRequirement:
    name: str
    required_inputs: tuple[str, ...]
    role: str = "reversion"
    priority: int = 100


SLEEVE_REQUIREMENTS: tuple[SleeveRequirement, ...] = (
    SleeveRequirement(
        "daily_residual_reversion",
        (
            "daily_ohlcv",
            "causal_universe_definition",
            "corporate_action_consistent_price_basis",
            "volume_unit_semantics",
            "completed_session_timestamps",
            "historical_live_price_volume_parity",
        ),
        priority=1,
    ),
    SleeveRequirement(
        "intraday_residual_failed_continuation",
        (
            "daily_ohlcv",
            "five_minute_ohlcv",
            "causal_universe_definition",
            "corporate_action_consistent_price_basis",
            "volume_unit_semantics",
            "completed_session_timestamps",
            "historical_live_price_volume_parity",
        ),
        priority=2,
    ),
    SleeveRequirement(
        "gap_residual_failed_continuation",
        (
            "daily_ohlcv",
            "five_minute_ohlcv",
            "causal_universe_definition",
            "corporate_action_consistent_price_basis",
            "volume_unit_semantics",
            "completed_session_timestamps",
            "historical_live_price_volume_parity",
        ),
        priority=3,
    ),
    SleeveRequirement(
        "trend_pullback_tail_control",
        (
            "daily_ohlcv",
            "five_minute_ohlcv",
            "causal_universe_definition",
            "corporate_action_consistent_price_basis",
            "volume_unit_semantics",
            "completed_session_timestamps",
            "historical_live_price_volume_parity",
        ),
        role="control",
        priority=4,
    ),
)

REVERSION_SLEEVES = tuple(
    sleeve.name for sleeve in SLEEVE_REQUIREMENTS if sleeve.role == "reversion"
)
CONTROL_SLEEVES = tuple(
    sleeve.name for sleeve in SLEEVE_REQUIREMENTS if sleeve.role == "control"
)
ANCHOR_REVERSION_SLEEVE = "daily_residual_reversion"
MIN_REPRESENTATIVE_REVERSION_SLEEVES = 1


def assess_input_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    """Return per-sleeve readiness and programme-level launch eligibility."""

    sleeve_readiness: dict[str, dict[str, Any]] = {}
    ready_reversion: list[str] = []
    ready_controls: list[str] = []
    for sleeve in sorted(SLEEVE_REQUIREMENTS, key=lambda value: value.priority):
        missing = [key for key in sleeve.required_inputs if not bool(authority.get(key))]
        ready = not missing
        sleeve_readiness[sleeve.name] = {
            "ready": ready,
            "role": sleeve.role,
            "priority": sleeve.priority,
            "required_inputs": list(sleeve.required_inputs),
            "missing_inputs": missing,
        }
        if ready and sleeve.role == "reversion":
            ready_reversion.append(sleeve.name)
        elif ready and sleeve.role == "control":
            ready_controls.append(sleeve.name)

    representative_ready = (
        ANCHOR_REVERSION_SLEEVE in ready_reversion
        and len(ready_reversion) >= MIN_REPRESENTATIVE_REVERSION_SLEEVES
    )
    sleeve_blockers = [
        f"{name}: missing {', '.join(row['missing_inputs'])}"
        for name, row in sleeve_readiness.items()
        if row["missing_inputs"]
    ]
    programme_blockers: list[str] = []
    if ANCHOR_REVERSION_SLEEVE not in ready_reversion:
        programme_blockers.append(
            f"required anchor sleeve not ready: {ANCHOR_REVERSION_SLEEVE}"
        )
    if len(ready_reversion) < MIN_REPRESENTATIVE_REVERSION_SLEEVES:
        programme_blockers.append(
            "representative launch requires at least "
            f"{MIN_REPRESENTATIVE_REVERSION_SLEEVES} independently data-ready "
            f"reversion sleeves; ready={len(ready_reversion)}"
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "representative_reversion_baseline_eligible": representative_ready,
        "mechanism_discovery_eligible": bool(ready_reversion),
        "ready_reversion_sleeves": ready_reversion,
        "ready_control_sleeves": ready_controls,
        "disabled_sleeves": [
            name for name, row in sleeve_readiness.items() if not row["ready"]
        ],
        "minimum_representative_reversion_sleeves": MIN_REPRESENTATIVE_REVERSION_SLEEVES,
        "anchor_reversion_sleeve": ANCHOR_REVERSION_SLEEVE,
        "sleeve_readiness": sleeve_readiness,
        "programme_blockers": programme_blockers,
        "blockers": [*programme_blockers, *sleeve_blockers],
    }


def _qualified_sleeves(atlas: Mapping[str, Any]) -> list[str]:
    raw = atlas.get("qualified_sleeves", ())
    if isinstance(raw, Mapping):
        return sorted(str(name) for name, row in raw.items() if bool(row))
    if isinstance(raw, (list, tuple, set, frozenset)):
        return sorted(str(name) for name in raw)
    return []


def assess_atlas_for_optimization(atlas: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a mechanism atlas may launch the phased optimizer."""

    authority = assess_input_authority(atlas.get("input_authority", {}))
    qualified = _qualified_sleeves(atlas)
    qualified_reversion = [name for name in qualified if name in REVERSION_SLEEVES]
    ready = set(authority["ready_reversion_sleeves"])
    phase_order = tuple(atlas.get("phase_order", ()))
    parity = atlas.get("economic_input_parity") or {}
    parity_sleeves = parity.get("passed_sleeves", ())
    if isinstance(parity_sleeves, Mapping):
        parity_sleeves = [name for name, passed in parity_sleeves.items() if passed]
    parity_set = {str(name) for name in parity_sleeves}
    checks = {
        "authority_contract_passed": bool(
            authority["representative_reversion_baseline_eligible"]
        ),
        "atlas_contract_version_matches": (
            str(atlas.get("representative_contract_version", "")) == CONTRACT_VERSION
        ),
        "mechanism_atlas_complete": bool(atlas.get("mechanism_atlas_complete")),
        "mechanism_candidate_registry_complete": bool(
            atlas.get("mechanism_candidate_registry_complete")
        ),
        "representative_qualified_sleeves": (
            ANCHOR_REVERSION_SLEEVE in qualified_reversion
            and len(qualified_reversion) >= MIN_REPRESENTATIVE_REVERSION_SLEEVES
        ),
        "qualified_sleeves_have_authority": set(qualified_reversion).issubset(ready),
        "economic_input_parity_passed": bool(parity.get("passed"))
        and set(qualified_reversion).issubset(parity_set),
        "phase_order_matches": phase_order == PHASE_ORDER,
        "selection_window_only": (
            bool((atlas.get("window") or {}).get("end"))
            and str((atlas.get("window") or {}).get("end", "")) <= CALIBRATION_END
        ),
        "holdout_not_accessed": not bool(atlas.get("holdout_accessed")),
        "downstream_execution_contract_matches": (
            str(atlas.get("downstream_execution_contract", ""))
            == DOWNSTREAM_EXECUTION_CONTRACT
        ),
    }
    blockers = list(authority["programme_blockers"])
    blockers.extend(name for name, passed in checks.items() if not passed)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "input_authority": authority,
        "qualified_sleeves": qualified,
        "qualified_reversion_sleeves": qualified_reversion,
        "blockers": blockers,
    }


def chronology_contract() -> dict[str, Any]:
    return {
        "selection_folds": [
            {"name": name, "start": start, "end": end}
            for name, start, end in SELECTION_FOLDS
        ],
        "locked_internal_validation": {
            "name": LOCKED_VALIDATION_FOLD[0],
            "start": LOCKED_VALIDATION_FOLD[1],
            "end": LOCKED_VALIDATION_FOLD[2],
            "used_for_candidate_ranking": False,
            "one_shot": True,
        },
        "sealed_holdout": {"start": HOLDOUT_START, "accessed": False},
    }
