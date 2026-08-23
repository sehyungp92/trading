from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from math import floor, isfinite
from typing import Any

import numpy as np

from backtests.stock.analysis.metrics import compute_metrics
from backtests.stock.models import Direction, TradeRecord
from libs.oms.risk.portfolio_rules import PortfolioRuleChecker, PortfolioRulesConfig

from ..phase_candidates import INITIAL_EQUITY
from .state import (
    BlockedCandidate,
    DecisionEvent,
    PortfolioAction,
    PortfolioActionType,
    PortfolioCoreState,
    PortfolioPosition,
    PortfolioReplayResult,
    ReplayCandidate,
    TradeOutcome,
)


CURRENT_IARIC_ID = "IARIC_RESIDUAL_R3"
CURRENT_ALCB_ID = "ALCB_R3"
MarkPriceProvider = Callable[[str, datetime], float | None]


def _strategy_ids(effective: dict[str, Any]) -> tuple[str, str]:
    """Use current strategy IDs while retaining archived-config compatibility."""

    allocations = effective.get("strategy_allocations", {})
    if CURRENT_IARIC_ID in allocations:
        return CURRENT_IARIC_ID, CURRENT_ALCB_ID
    return "IARIC_V5R1", "ALCB_R3"


def replay_trade_streams(
    alcb_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    iaric_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    effective: dict[str, Any],
    *,
    mark_price_provider: MarkPriceProvider | None = None,
) -> dict[str, float]:
    return run_portfolio_replay(
        alcb_trades,
        iaric_trades,
        effective,
        mark_price_provider=mark_price_provider,
    ).metrics


def run_portfolio_replay(
    alcb_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    iaric_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    effective: dict[str, Any],
    *,
    mark_price_provider: MarkPriceProvider | None = None,
) -> PortfolioReplayResult:
    return asyncio.run(
        _run_portfolio_replay_async(
            alcb_trades,
            iaric_trades,
            effective,
            mark_price_provider=mark_price_provider,
        )
    )


async def _run_portfolio_replay_async(
    alcb_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    iaric_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    effective: dict[str, Any],
    *,
    mark_price_provider: MarkPriceProvider | None = None,
) -> PortfolioReplayResult:
    initial_equity = float(effective.get("initial_equity", INITIAL_EQUITY))
    rules = effective["portfolio_rules"]
    reference_risk_pct = float(rules.get("reference_risk_pct", 0.006) or 0.006)
    iaric_id, alcb_id = _strategy_ids(effective)
    strategy_order = (iaric_id, alcb_id)
    lookback = int(effective.get("dynamic_allocation", {}).get("lookback_trades", 60) or 60)
    state = PortfolioCoreState.initial(
        initial_equity=initial_equity,
        reference_risk_pct=reference_risk_pct,
        lookback_trades=lookback,
        strategy_ids=strategy_order,
    )

    entries: list[tuple[datetime, str, TradeRecord]] = []
    entries.extend((trade.entry_time, alcb_id, trade) for trade in alcb_trades)
    entries.extend((trade.entry_time, iaric_id, trade) for trade in iaric_trades)
    entries.sort(key=lambda item: item[0])
    live_rule_adapter = _StockPortfolioLiveRuleReplayAdapter(
        state=state,
        effective=effective,
        symbol_sector_map=_symbol_sector_map(entries),
        strategy_order=strategy_order,
    )

    actions: list[PortfolioAction] = []
    decisions: list[DecisionEvent] = []
    trade_outcomes: list[TradeOutcome] = []

    i = 0
    while i < len(entries):
        ts = entries[i][0]
        _close_positions(
            state,
            before=ts,
            actions=actions,
            trade_outcomes=trade_outcomes,
            effective=effective,
            mark_price_provider=mark_price_provider,
        )
        _refresh_account_state(
            state,
            ts,
            effective,
            mark_price_provider=mark_price_provider,
        )

        batch: list[tuple[str, TradeRecord]] = []
        while i < len(entries) and entries[i][0] == ts:
            batch.append((entries[i][1], entries[i][2]))
            i += 1
        candidates = [
            _build_candidate(
                strategy,
                trade,
                effective,
                state.net_liquidation_value,
                state.peak_net_liquidation_value,
                state.strategy_recent,
            )
            for strategy, trade in batch
        ]
        candidates = [candidate for candidate in candidates if candidate is not None]
        candidates.sort(key=lambda candidate: _rank_candidate(candidate, effective), reverse=True)

        for candidate in candidates:
            state.candidate_count += 1
            reason = await live_rule_adapter.check_entry(candidate)
            if reason:
                action = PortfolioAction(
                    action_type=PortfolioActionType.BLOCK_ENTRY,
                    timestamp=candidate.trade.entry_time,
                    strategy_id=candidate.strategy,
                    symbol=candidate.trade.symbol,
                    reason=reason,
                    risk_dollars=candidate.risk_dollars,
                    metadata=_candidate_metadata(candidate),
                )
                actions.append(action)
                decisions.append(
                    _decision_event(
                        state,
                        candidate,
                        decision_code="BLOCK_ENTRY",
                        reason=reason,
                        action=action,
                    )
                )
                state.blocked_candidates.append(
                    BlockedCandidate(
                        strategy=candidate.strategy,
                        symbol=candidate.trade.symbol,
                        sector=candidate.trade.sector,
                        entry_time=candidate.trade.entry_time,
                        r_multiple=candidate.r_multiple,
                        reason=reason,
                        quality=candidate.quality,
                        heat_r=candidate.heat_r,
                        requested_quantity=candidate.requested_quantity,
                        requested_notional=candidate.requested_notional,
                    )
                )
                continue

            requested_qty = int(candidate.requested_quantity)
            approved_qty = int(floor(requested_qty * float(candidate.portfolio_size_mult)))
            if requested_qty < 1 or approved_qty < 1:
                raise RuntimeError("approved portfolio quantity must be positive")
            qty_ratio = approved_qty / requested_qty if requested_qty > 0 else 0.0
            approved_risk_dollars = candidate.risk_dollars * qty_ratio
            approved_pnl = candidate.pnl * qty_ratio
            price_scale = approved_qty / float(candidate.trade.quantity) if candidate.trade.quantity else 0.0
            scaled_commission = candidate.trade.commission * price_scale
            metadata = dict(candidate.trade.metadata or {})
            metadata.update(
                {
                    "portfolio_requested_qty": requested_qty,
                    "portfolio_approved_qty": approved_qty,
                    "portfolio_size_mult": float(candidate.portfolio_size_mult),
                }
            )
            position = PortfolioPosition(
                strategy=candidate.strategy,
                symbol=candidate.trade.symbol,
                sector=candidate.trade.sector,
                direction=candidate.trade.direction,
                entry_time=candidate.trade.entry_time,
                decision_time=candidate.trade.entry_time,
                fill_time=candidate.trade.fill_time or candidate.trade.entry_time,
                exit_time=candidate.trade.exit_time,
                risk_dollars=approved_risk_dollars,
                pnl=approved_pnl,
                r_multiple=candidate.r_multiple,
                quality=candidate.quality,
                entry_price=float(candidate.trade.entry_price),
                exit_price=float(candidate.trade.exit_price),
                quantity=float(approved_qty),
                entry_notional=float(candidate.trade.entry_price) * approved_qty,
                current_mark=float(candidate.trade.entry_price),
                mark_price_scale=_mark_price_scale(
                    candidate.trade,
                    mark_price_provider,
                ),
                last_mark_time=candidate.trade.entry_time,
                price_scale=price_scale,
                commission=scaled_commission,
                exit_reason=candidate.trade.exit_reason,
                entry_type=candidate.trade.entry_type,
                metadata=metadata,
            )
            action = PortfolioAction(
                action_type=PortfolioActionType.SUBMIT_ENTRY,
                timestamp=position.decision_time,
                strategy_id=position.strategy,
                symbol=position.symbol,
                risk_dollars=position.risk_dollars,
                metadata=_candidate_metadata(candidate),
            )
            actions.append(action)
            decisions.append(
                _decision_event(
                    state,
                    candidate,
                    decision_code="ACCEPT_ENTRY",
                    reason="accepted",
                    action=action,
                )
            )
            state.active_positions.append(position)
            state.accepted_positions.append(position)
            state.cash -= (
                float(position.direction) * position.entry_notional
            )
            state.risk_by_strategy[position.strategy] = (
                state.risk_by_strategy.get(position.strategy, 0.0) + position.risk_dollars
            )
            _refresh_account_state(
                state,
                position.entry_time,
                effective,
                mark_price_provider=mark_price_provider,
            )

    if state.active_positions:
        final_exit = max(position.exit_time for position in state.active_positions)
        _close_positions(
            state,
            before=final_exit,
            actions=actions,
            trade_outcomes=trade_outcomes,
            effective=effective,
            mark_price_provider=mark_price_provider,
        )
    metrics = _compute_replay_metrics(state, entries, initial_equity, strategy_order)
    return PortfolioReplayResult(
        metrics=metrics,
        state=state,
        decisions=tuple(decisions),
        actions=tuple(actions),
        trade_outcomes=tuple(trade_outcomes),
        replay_architecture="stock_portfolio_core_live_rule_adapter",
    )


class _StockPortfolioLiveRuleReplayAdapter:
    """Replay adapter that delegates portfolio admission to the live rule checker."""

    def __init__(
        self,
        *,
        state: PortfolioCoreState,
        effective: dict[str, Any],
        symbol_sector_map: tuple[tuple[str, str], ...],
        strategy_order: tuple[str, str],
    ) -> None:
        self._state = state
        self._effective = effective
        self._strategy_order = strategy_order
        self._current_time: datetime | None = None
        self._base_config = self._portfolio_rules_config(symbol_sector_map)
        self._checker = PortfolioRuleChecker(
            config=self._base_config,
            get_strategy_signal=self._get_strategy_signal,
            get_directional_risk_R=self._get_directional_risk_R,
            get_current_equity=lambda: float(self._state.net_liquidation_value),
            get_directional_risk_R_for_strategies=self._get_directional_risk_R_for_strategies,
            get_sibling_positions_for_symbol=self._get_sibling_positions_for_symbol,
            get_directional_risk_dollars_for_strategies=(
                self._get_directional_risk_dollars_for_strategies
            ),
            get_open_position_count_for_strategies=self._get_open_position_count_for_strategies,
            get_symbol_open_risk_dollars_for_strategies=(
                self._get_symbol_open_risk_dollars_for_strategies
            ),
            get_symbols_open_risk_dollars_for_strategies=(
                self._get_symbols_open_risk_dollars_for_strategies
            ),
            get_active_risk_dollars_for_strategies=self._get_active_risk_dollars_for_strategies,
            get_completed_trade_counts_for_strategies=self._get_completed_trade_counts_for_strategies,
            get_recent_strategy_r_multiples=self._get_recent_strategy_r_multiples,
            now_provider=lambda: self._current_time or datetime.utcnow(),
        )

    async def check_entry(self, candidate: ReplayCandidate) -> str:
        self._current_time = candidate.trade.entry_time
        if candidate.requested_quantity < 1:
            return "quantity_below_one"
        if _drawdown_mult(
            self._state.net_liquidation_value,
            self._state.peak_net_liquidation_value,
            self._effective,
        ) <= 0.0:
            return "drawdown_halt"
        alpha_reason = self._alpha_admission_reason(candidate)
        if alpha_reason:
            return alpha_reason
        capacity_mult = self._capacity_size_multiplier(candidate)
        if capacity_mult <= 0.0:
            return candidate.capacity_reason or "capacity_below_min_size"
        prospective_quantity = int(
            floor(candidate.requested_quantity * capacity_mult)
        )
        if prospective_quantity < 1:
            return candidate.capacity_reason or "capacity_below_min_size"
        self._checker.update_config(
            replace(
                self._base_config,
                reference_unit_risk_dollars=self._reference_risk_dollars(),
                initial_equity=float(self._state.net_liquidation_value),
            )
        )
        result = await self._checker.check_entry(
            strategy_id=candidate.strategy,
            direction=_direction_text(candidate.trade.direction),
            new_risk_R=candidate.heat_r * capacity_mult,
            symbol=candidate.trade.symbol,
            new_qty=prospective_quantity,
            new_risk_dollars=candidate.risk_dollars * capacity_mult,
        )
        if not result.approved:
            return _legacy_block_reason(result.denial_reason or "")
        candidate.portfolio_size_mult = capacity_mult * float(result.size_multiplier)
        if int(floor(candidate.requested_quantity * candidate.portfolio_size_mult)) < 1:
            return "capacity_below_min_size"
        return self._custom_replay_block_reason(candidate)

    def _alpha_admission_reason(self, candidate: ReplayCandidate) -> str:
        cross = self._effective.get("cross_strategy_rules", {})
        if not bool(cross.get("alpha_admission_enabled", False)):
            return ""
        expected_r = _expected_candidate_r(candidate.trade)
        uncertainty = max(
            _meta_float(candidate.trade.metadata or {}, "portfolio_expected_r_uncertainty", 0.0),
            0.0,
        )
        penalty = float(cross.get("alpha_uncertainty_penalty", 0.0) or 0.0)
        expected_lcb = expected_r - penalty * uncertainty
        floor = float(cross.get("minimum_expected_r", 0.0) or 0.0)
        return "alpha_floor" if expected_lcb < floor else ""

    def _capacity_size_multiplier(self, candidate: ReplayCandidate) -> float:
        cross = self._effective.get("cross_strategy_rules", {})
        account_multiplier, account_reason = self._account_capacity_multiplier(candidate)
        if account_multiplier < 1.0:
            candidate.capacity_reason = account_reason
            account = self._effective.get("account_rules", {})
            if str(account.get("oversize_action", "resize")) == "block":
                return 0.0

        if str(cross.get("capacity_action", "block")) != "resize":
            return account_multiplier

        rules = self._effective.get("portfolio_rules", {})
        allocations = self._effective.get("strategy_allocations", {})
        ref = self._reference_risk_dollars()
        candidate_risk = max(float(candidate.risk_dollars), 1e-9)
        active = self._state.active_positions
        available: list[float] = []

        def add_available(cap_r: float, used_dollars: float) -> None:
            if cap_r > 0:
                available.append((cap_r * ref - used_dollars) / candidate_risk)

        add_available(
            float(rules.get("heat_cap_R", 0.0) or 0.0),
            sum(position.risk_dollars for position in active),
        )
        if candidate.trade.direction == Direction.LONG:
            add_available(
                float(rules.get("max_long_heat_R", 0.0) or 0.0),
                sum(
                    position.risk_dollars
                    for position in active
                    if position.direction == Direction.LONG
                ),
            )
        add_available(
            float(rules.get("max_symbol_heat_R", 0.0) or 0.0),
            sum(
                position.risk_dollars
                for position in active
                if position.symbol == candidate.trade.symbol
            ),
        )
        sector_cap = float(cross.get("same_sector_heat_cap_R", 0.0) or 0.0)
        if candidate.trade.sector:
            add_available(
                sector_cap,
                sum(
                    position.risk_dollars
                    for position in active
                    if position.sector == candidate.trade.sector
                ),
            )
        if bool(cross.get("apply_duplicate_native_limits", True)):
            allocation = allocations.get(candidate.strategy, {})
            add_available(
                float(allocation.get("max_heat_R", 0.0) or 0.0),
                sum(
                    position.risk_dollars
                    for position in active
                    if position.strategy == candidate.strategy
                ),
            )

        multiplier = min([account_multiplier, *available]) if available else account_multiplier
        minimum = float(cross.get("minimum_capacity_size_mult", 0.35) or 0.35)
        if multiplier + 1e-12 < minimum:
            if not candidate.capacity_reason:
                candidate.capacity_reason = "capacity_below_min_size"
            return 0.0
        return min(max(multiplier, 0.0), 1.0)

    def _account_capacity_multiplier(
        self,
        candidate: ReplayCandidate,
    ) -> tuple[float, str]:
        account = self._effective.get("account_rules", {})
        if not bool(account.get("enforce_shared_buying_power", False)):
            return 1.0, ""

        nlv = max(float(self._state.net_liquidation_value), 0.0)
        candidate_notional = max(float(candidate.requested_notional), 0.0)
        if nlv <= 0.0 or candidate_notional <= 0.0:
            return 0.0, "invalid_account_equity"

        active = self._state.active_positions
        notionals = [
            abs(float(position.current_mark or position.entry_price) * position.quantity)
            for position in active
        ]
        gross = sum(notionals)
        net = sum(
            float(position.direction)
            * float(position.current_mark or position.entry_price)
            * position.quantity
            for position in active
        )
        symbol_gross = sum(
            notional
            for position, notional in zip(active, notionals)
            if position.symbol == candidate.trade.symbol
        )
        overnight_gross = sum(
            notional
            for position, notional in zip(active, notionals)
            if position.exit_time.date() > position.entry_time.date()
        )

        limits: list[tuple[float, str]] = []

        def add_limit(available_notional: float, reason: str) -> None:
            limits.append((available_notional / candidate_notional, reason))

        max_position = float(account.get("max_position_notional_pct", 0.0) or 0.0)
        allocation = self._effective.get("strategy_allocations", {}).get(
            candidate.strategy, {}
        )
        strategy_position = float(
            allocation.get("max_position_notional_pct", 0.0) or 0.0
        )
        position_cap = min(
            [value for value in (max_position, strategy_position) if value > 0.0],
            default=0.0,
        )
        if position_cap > 0.0:
            add_limit(position_cap * nlv, "position_notional_cap")

        gross_cap = float(account.get("max_gross_notional_pct", 0.0) or 0.0)
        if gross_cap > 0.0:
            add_limit(gross_cap * nlv - gross, "gross_notional_cap")

        net_cap = float(account.get("max_net_notional_pct", 0.0) or 0.0)
        if net_cap > 0.0:
            direction = float(candidate.trade.direction)
            if direction >= 0.0:
                add_limit(net_cap * nlv - net, "net_notional_cap")
            else:
                add_limit(net_cap * nlv + net, "net_notional_cap")

        symbol_cap = float(account.get("max_symbol_notional_pct", 0.0) or 0.0)
        if symbol_cap > 0.0:
            add_limit(symbol_cap * nlv - symbol_gross, "symbol_notional_cap")

        is_overnight = candidate.trade.exit_time.date() > candidate.trade.entry_time.date()
        overnight_cap = float(
            account.get("max_overnight_gross_notional_pct", 0.0) or 0.0
        )
        if is_overnight and overnight_cap > 0.0:
            add_limit(
                overnight_cap * nlv - overnight_gross,
                "overnight_notional_cap",
            )

        initial_margin = sum(
            notional * _initial_margin_rate(position.direction, account)
            for position, notional in zip(active, notionals)
        )
        candidate_margin_rate = _initial_margin_rate(candidate.trade.direction, account)
        minimum_buffer = float(account.get("minimum_margin_buffer_pct", 0.0) or 0.0)
        margin_available = nlv * max(1.0 - minimum_buffer, 0.0) - initial_margin
        if candidate_margin_rate > 0.0:
            add_limit(
                margin_available / candidate_margin_rate,
                "initial_margin_cap",
            )

        if not limits:
            return 1.0, ""
        multiplier, reason = min(limits, key=lambda item: item[0])
        return min(max(float(multiplier), 0.0), 1.0), reason

    def _portfolio_rules_config(
        self,
        symbol_sector_map: tuple[tuple[str, str], ...],
    ) -> PortfolioRulesConfig:
        rules = self._effective["portfolio_rules"]
        allocations = self._effective["strategy_allocations"]
        cross = self._effective.get("cross_strategy_rules", {})
        strategy_priorities = tuple(
            (strategy, int(allocations[strategy].get("priority", index)))
            for index, strategy in enumerate(self._strategy_order)
            if strategy in allocations
        )
        same_symbol_policy = str(cross.get("same_symbol_policy", "half_size"))
        collision_action = same_symbol_policy if same_symbol_policy in {"none", "block", "half_size"} else "none"
        apply_native_limits = bool(cross.get("apply_duplicate_native_limits", True))
        return PortfolioRulesConfig(
            nqdtc_direction_filter_enabled=False,
            directional_cap_R=0.0,
            directional_cap_long_R=float(rules.get("max_long_heat_R", 0.0) or 0.0),
            directional_cap_short_R=0.0,
            dd_tiers=((1.0, 1.0),),
            initial_equity=float(self._state.net_liquidation_value),
            family_strategy_ids=tuple(self._strategy_order),
            symbol_collision_action=collision_action,
            symbol_collision_pairs=tuple(
                (str(row[0]), str(row[1]), str(row[2]))
                for row in cross.get("symbol_collision_pairs", ()) or ()
                if len(row) >= 3
            ),
            strategy_priorities=strategy_priorities,
            priority_headroom_R=0.0,
            reference_unit_risk_dollars=self._reference_risk_dollars(),
            reference_unit_risk_pct=float(rules.get("reference_risk_pct", 0.006) or 0.006),
            max_total_active_positions=int(rules.get("max_total_active_positions", 0) or 0),
            max_symbol_heat_R=float(rules.get("max_symbol_heat_R", 0.0) or 0.0),
            same_sector_heat_cap_R=float(cross.get("same_sector_heat_cap_R", 0.0) or 0.0),
            symbol_sector_map=symbol_sector_map,
            max_single_strategy_trade_share=float(
                rules.get("max_single_strategy_trade_share", 1.0) or 1.0
            ),
            strategy_trade_share_min_total=50,
            dynamic_allocation_enabled=False,
            portfolio_heat_cap_R=float(rules.get("heat_cap_R", 0.0) or 0.0),
            max_strategy_active_positions=tuple(
                (strategy, int(allocations[strategy].get("max_concurrent", 0) or 0))
                for strategy in self._strategy_order
                if strategy in allocations
            ) if apply_native_limits else (),
            max_strategy_heat_R=tuple(
                (strategy, float(allocations[strategy].get("max_heat_R", 0.0) or 0.0))
                for strategy in self._strategy_order
                if strategy in allocations
            ) if apply_native_limits else (),
        )

    def _reference_risk_dollars(self) -> float:
        return max(
            float(self._state.net_liquidation_value)
            * float(self._effective["portfolio_rules"].get("reference_risk_pct", 0.006) or 0.006),
            1.0,
        )

    def _custom_replay_block_reason(self, candidate: ReplayCandidate) -> str:
        stop_reason = self._loss_stop_reason(candidate)
        if stop_reason:
            return stop_reason
        reserve_reason = self._intraday_reserve_reason(candidate)
        if reserve_reason:
            return reserve_reason
        cross = self._effective.get("cross_strategy_rules", {})
        same_symbol_policy = str(cross.get("same_symbol_policy", "half_size"))
        if same_symbol_policy != "best_rank_only":
            return ""
        same_symbol_active = [
            position
            for position in self._state.active_positions
            if position.symbol == candidate.trade.symbol
        ]
        if not same_symbol_active:
            return ""
        best_active_quality = max(position.quality for position in same_symbol_active)
        return "same_symbol_lower_rank" if candidate.quality <= best_active_quality else ""

    def _loss_stop_reason(self, candidate: ReplayCandidate) -> str:
        rules = self._effective.get("portfolio_rules", {})
        day_key = candidate.trade.entry_time.date().isoformat()
        iso = candidate.trade.entry_time.isocalendar()
        week_key = f"{iso.year:04d}-W{iso.week:02d}"
        daily_cap = float(rules.get("portfolio_daily_stop_R", 0.0) or 0.0)
        weekly_cap = float(rules.get("portfolio_weekly_stop_R", 0.0) or 0.0)
        if daily_cap > 0 and self._state.daily_realized_r.get(day_key, 0.0) <= -daily_cap:
            return "portfolio_daily_stop"
        if weekly_cap > 0 and self._state.weekly_realized_r.get(week_key, 0.0) <= -weekly_cap:
            return "portfolio_weekly_stop"
        cross = self._effective.get("cross_strategy_rules", {})
        strategy_cap = (
            float(
                self._effective["strategy_allocations"][candidate.strategy].get(
                    "daily_stop_R", 0.0
                )
                or 0.0
            )
            if bool(cross.get("apply_duplicate_native_limits", True))
            else 0.0
        )
        strategy_day_key = f"{candidate.strategy}|{day_key}"
        if (
            strategy_cap > 0
            and self._state.strategy_daily_realized_r.get(strategy_day_key, 0.0)
            <= -strategy_cap
        ):
            return "strategy_daily_stop"
        return ""

    def _intraday_reserve_reason(self, candidate: ReplayCandidate) -> str:
        iaric_id, _alcb_id = self._strategy_order
        if candidate.strategy != iaric_id:
            return ""
        rules = self._effective.get("portfolio_rules", {})
        cross = self._effective.get("cross_strategy_rules", {})
        reserve_slots = int(cross.get("intraday_reserved_slots", 0) or 0)
        total_slots = int(rules.get("max_total_active_positions", 0) or 0)
        if reserve_slots > 0 and total_slots > 0:
            if len(self._state.active_positions) + 1 > max(total_slots - reserve_slots, 0):
                return "intraday_reserved_slots"
        reserve_heat = float(cross.get("intraday_reserved_heat_R", 0.0) or 0.0)
        total_heat = float(rules.get("heat_cap_R", 0.0) or 0.0)
        if reserve_heat > 0 and total_heat > 0:
            ref = self._reference_risk_dollars()
            current_heat = sum(p.risk_dollars for p in self._state.active_positions) / ref
            candidate_heat = candidate.heat_r * float(candidate.portfolio_size_mult)
            if current_heat + candidate_heat > max(total_heat - reserve_heat, 0.0):
                return "intraday_reserved_heat"
        return ""

    async def _get_strategy_signal(self, strategy_id: str) -> None:
        del strategy_id
        return None

    async def _get_directional_risk_R(self, direction: str) -> float:
        return await self._get_directional_risk_R_for_strategies(
            direction, list(self._strategy_order)
        )

    async def _get_directional_risk_R_for_strategies(
        self,
        direction: str,
        strategy_ids: list[str],
    ) -> float:
        ref = self._reference_risk_dollars()
        return await self._get_directional_risk_dollars_for_strategies(direction, strategy_ids) / ref

    async def _get_directional_risk_dollars_for_strategies(
        self,
        direction: str,
        strategy_ids: list[str],
    ) -> float:
        ids = set(strategy_ids)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids
                and _direction_text(position.direction) == direction.upper()
            )
        )

    async def _get_open_position_count_for_strategies(self, strategy_ids: list[str]) -> int:
        ids = set(strategy_ids)
        return sum(1 for position in self._state.active_positions if position.strategy in ids)

    async def _get_sibling_positions_for_symbol(self, strategy_ids: list[str], symbol: str) -> bool:
        ids = set(strategy_ids)
        return any(
            position.strategy in ids and position.symbol == symbol
            for position in self._state.active_positions
        )

    async def _get_symbol_open_risk_dollars_for_strategies(
        self,
        strategy_ids: list[str],
        symbol: str,
    ) -> float:
        ids = set(strategy_ids)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids and position.symbol == symbol
            )
        )

    async def _get_symbols_open_risk_dollars_for_strategies(
        self,
        strategy_ids: list[str],
        symbols: list[str],
    ) -> float:
        ids = set(strategy_ids)
        symbol_set = set(symbols)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids and position.symbol in symbol_set
            )
        )

    async def _get_active_risk_dollars_for_strategies(self, strategy_ids: list[str]) -> float:
        ids = set(strategy_ids)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids
            )
        )

    async def _get_completed_trade_counts_for_strategies(
        self,
        strategy_ids: list[str],
    ) -> dict[str, int]:
        ids = set(strategy_ids)
        counts: dict[str, int] = {strategy: 0 for strategy in strategy_ids}
        for position in self._state.accepted_positions:
            if position.strategy in ids:
                counts[position.strategy] = counts.get(position.strategy, 0) + 1
        return counts

    async def _get_recent_strategy_r_multiples(self, strategy_id: str, lookback: int) -> list[float]:
        recent = self._state.strategy_recent.get(strategy_id)
        if not recent:
            return []
        return list(recent)[-max(1, int(lookback)):]


def _symbol_sector_map(entries: list[tuple[datetime, str, TradeRecord]]) -> tuple[tuple[str, str], ...]:
    mapping: dict[str, str] = {}
    for _, _, trade in entries:
        symbol = str(getattr(trade, "symbol", "") or "")
        sector = str(getattr(trade, "sector", "") or "")
        if symbol and sector:
            mapping[symbol] = sector
    return tuple(sorted(mapping.items()))


def _direction_text(direction: Direction) -> str:
    return "LONG" if direction == Direction.LONG or int(direction) > 0 else "SHORT"


def _legacy_block_reason(denial_reason: str) -> str:
    reason = denial_reason or "portfolio_rule_block"
    if reason.startswith("max_total_active_positions"):
        return "max_total_active_positions"
    if reason.startswith("max_strategy_active_positions"):
        return "strategy_max_concurrent"
    if reason.startswith("portfolio_heat_cap"):
        return "portfolio_heat_cap"
    if reason.startswith("strategy_heat_cap"):
        return "strategy_heat_cap"
    if reason.startswith("symbol_heat_cap"):
        return "symbol_heat_cap"
    if reason.startswith("sector_heat_cap"):
        return "sector_heat_cap"
    if reason.startswith("directional_cap"):
        return "long_heat_cap"
    if reason.startswith("strategy_trade_share_cap"):
        return "strategy_trade_share_cap"
    return reason.split(":", 1)[0]


def _decision_event(
    state: PortfolioCoreState,
    candidate: ReplayCandidate,
    *,
    decision_code: str,
    reason: str,
    action: PortfolioAction,
) -> DecisionEvent:
    state.decision_seq += 1
    return DecisionEvent(
        timestamp=candidate.trade.entry_time,
        strategy_id=candidate.strategy,
        symbol=candidate.trade.symbol,
        decision_code=decision_code,
        reason=reason,
        state_snapshot_ref=f"stock_portfolio_core:{state.decision_seq}",
        actions_emitted=(action,),
        details=_candidate_metadata(candidate),
    )


def _candidate_metadata(candidate: ReplayCandidate) -> dict[str, Any]:
    expected_r = _expected_candidate_r(candidate.trade)
    uncertainty = _meta_float(
        candidate.trade.metadata or {},
        "portfolio_expected_r_uncertainty",
        0.0,
    )
    return {
        "entry_type": candidate.trade.entry_type,
        "sector": candidate.trade.sector,
        "heat_r": float(candidate.heat_r),
        "quality": float(candidate.quality),
        "r_multiple": float(candidate.r_multiple),
        "size_mult": float(candidate.size_mult),
        "requested_quantity": int(candidate.requested_quantity),
        "requested_notional": float(candidate.requested_notional),
        "portfolio_size_mult": float(candidate.portfolio_size_mult),
        "capacity_reason": candidate.capacity_reason,
        "portfolio_expected_r": float(expected_r),
        "portfolio_expected_r_uncertainty": float(uncertainty),
    }


def _compute_replay_metrics(
    state: PortfolioCoreState,
    entries: list[tuple[datetime, str, TradeRecord]],
    initial_equity: float,
    strategy_order: tuple[str, str],
) -> dict[str, float]:
    accepted = state.accepted_positions
    blocked = state.blocked_candidates
    pnl_by_strategy: dict[str, float] = defaultdict(float)
    for position in accepted:
        pnl_by_strategy[position.strategy] += position.pnl

    pnls = np.array([position.pnl for position in accepted], dtype=np.float64)
    risks = np.array([position.risk_dollars for position in accepted], dtype=np.float64)
    hold_hours = np.array(
        [
            max((position.exit_time - position.entry_time).total_seconds() / 3600.0, 0.0)
            for position in accepted
        ],
        dtype=np.float64,
    )
    commissions = np.array([p.commission for p in accepted], dtype=np.float64)
    timestamps = np.array(state.nlv_times, dtype="datetime64[ns]")
    equity_curve = np.array(state.nlv_points, dtype=np.float64)
    if len(timestamps) + 1 == len(equity_curve):
        equity_for_metrics = equity_curve
    else:
        equity_for_metrics = np.array([initial_equity, state.equity], dtype=np.float64)
        timestamps = np.array([], dtype="datetime64[ns]")

    perf = compute_metrics(
        pnls,
        risks,
        hold_hours,
        commissions,
        equity_for_metrics,
        timestamps,
        initial_equity,
        trade_symbols=[position.symbol for position in accepted],
    )

    months = _months_from_positions(accepted, blocked)
    total_r = float(np.sum(np.divide(pnls, np.where(risks > 0, risks, 1.0)))) if len(pnls) else 0.0
    strategy_counts = {
        strategy: sum(1 for position in accepted if position.strategy == strategy)
        for strategy in strategy_order
    }
    active_strategy_count = sum(1 for count in strategy_counts.values() if count > 0)
    total_risk = sum(state.risk_by_strategy.values())
    max_strategy_risk_share = (
        max(state.risk_by_strategy.values()) / total_risk
        if total_risk > 0 and state.risk_by_strategy
        else 0.0
    )
    max_strategy_trade_share = max(strategy_counts.values()) / len(accepted) if accepted else 0.0
    positive_candidates = sum(
        1
        for _, _, trade in entries
        if float(getattr(trade, "r_multiple", 0.0) or 0.0) > 0
    )
    positive_blocks = [candidate for candidate in blocked if candidate.r_multiple > 0]
    nonpositive_blocks = [candidate for candidate in blocked if candidate.r_multiple <= 0]
    blocked_r = [candidate.r_multiple for candidate in blocked]
    accepted_r = [position.r_multiple for position in accepted]
    blocked_positive_r = [candidate.r_multiple for candidate in positive_blocks]
    candidate_discrimination = _candidate_discrimination(accepted_r, blocked_r)
    daily_losses = [min(0.0, value) for value in state.daily_realized_r.values()]
    weekly_losses = [min(0.0, value) for value in state.weekly_realized_r.values()]
    max_daily_loss_r = abs(min(daily_losses)) if daily_losses else 0.0
    max_weekly_loss_r = abs(min(weekly_losses)) if weekly_losses else 0.0

    metrics = {
        "initial_equity": initial_equity,
        "final_equity": state.equity,
        "net_pnl": state.equity - initial_equity,
        "net_return_pct": (state.equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0,
        "total_trades": float(len(accepted)),
        "entry_signals_fired": float(state.candidate_count),
        "entries_accepted_by_portfolio": float(len(accepted)),
        "entries_blocked_by_portfolio": float(len(blocked)),
        "entry_accept_rate": float(len(accepted) / state.candidate_count) if state.candidate_count else 0.0,
        "active_trades_per_month": float(len(accepted) / months) if months > 0 else 0.0,
        "total_r": total_r,
        "total_r_per_month": total_r / months if months > 0 else 0.0,
        "profit_factor": float(perf.profit_factor),
        "win_rate": float(perf.win_rate),
        "expectancy_r": float(perf.expectancy),
        "sharpe": float(perf.sharpe),
        "sortino": float(perf.sortino),
        "calmar": float(perf.calmar),
        "max_drawdown_pct": float(perf.max_drawdown_pct),
        "max_drawdown_dollar": float(perf.max_drawdown_dollar),
        "active_strategy_count": float(active_strategy_count),
        "max_strategy_trade_share": float(max_strategy_trade_share),
        "max_strategy_risk_share": float(max_strategy_risk_share),
        "trade_capture_ratio": float(len(accepted) / state.candidate_count) if state.candidate_count else 0.0,
        "positive_alpha_block_rate": (
            float(len(positive_blocks) / positive_candidates) if positive_candidates > 0 else 0.0
        ),
        "blocked_positive_count": float(len(positive_blocks)),
        "blocked_nonpositive_count": float(len(nonpositive_blocks)),
        "blocked_positive_fraction": float(len(positive_blocks) / len(blocked)) if blocked else 0.0,
        "blocked_avg_r": float(np.mean(blocked_r)) if blocked_r else 0.0,
        "accepted_avg_r": float(np.mean(accepted_r)) if accepted_r else 0.0,
        "blocked_positive_avg_r": float(np.mean(blocked_positive_r)) if blocked_positive_r else 0.0,
        "candidate_discrimination": float(candidate_discrimination),
        "positive_slices": float(_positive_slices(accepted)),
        "max_daily_loss_R": float(max_daily_loss_r),
        "max_weekly_loss_R": float(max_weekly_loss_r),
        "financing_cost": float(state.financing_cost),
        "gross_notional_peak": float(state.gross_notional_peak),
        "gross_leverage_peak": float(state.gross_leverage_peak),
        "net_notional_peak_abs": float(state.net_notional_peak_abs),
        "net_leverage_peak_abs": float(state.net_leverage_peak_abs),
        "overnight_gross_leverage_peak": float(
            state.overnight_gross_leverage_peak
        ),
        "initial_margin_peak": float(state.initial_margin_peak),
        "maintenance_margin_peak": float(state.maintenance_margin_peak),
        "minimum_margin_buffer_pct": float(state.minimum_margin_buffer_pct),
        "margin_breach_count": float(state.margin_breach_count),
        "mark_coverage_ratio": float(
            1.0 - state.missing_mark_count / state.mark_observation_count
        )
        if state.mark_observation_count
        else 1.0,
        "missing_mark_count": float(state.missing_mark_count),
        **{f"trades_{strategy}": float(count) for strategy, count in strategy_counts.items()},
        **{
            f"pnl_{strategy}": float(pnl_by_strategy.get(strategy, 0.0))
            for strategy in strategy_order
        },
        **{
            f"risk_share_{strategy}": (
                float(state.risk_by_strategy.get(strategy, 0.0) / total_risk) if total_risk else 0.0
            )
            for strategy in strategy_order
        },
        **_block_reason_metrics(blocked),
    }
    return metrics


def _build_candidate(
    strategy: str,
    trade: TradeRecord,
    effective: dict[str, Any],
    equity: float,
    peak_equity: float,
    strategy_recent: dict[str, deque[float]],
) -> ReplayCandidate | None:
    allocations = effective["strategy_allocations"]
    if strategy not in allocations:
        return None
    allocation = allocations[strategy]
    if allocation.get("enabled", True) is False:
        return None

    risk_per_share = float(trade.risk_per_share)
    source_quantity = float(trade.quantity)
    source_risk = risk_per_share * source_quantity
    if source_risk <= 0 or risk_per_share <= 0 or float(trade.entry_price) <= 0:
        return None

    drawdown_mult = _drawdown_mult(equity, peak_equity, effective)
    quality = _candidate_quality(strategy, trade, effective)
    size_mult = _candidate_size_mult(strategy, trade, effective)
    dynamic_mult = _dynamic_mult(strategy, strategy_recent, effective)
    unit_risk_pct = float(allocation.get("unit_risk_pct", 0.006) or 0.006)
    # A zero drawdown multiplier is an explicit admission denial, not a missing
    # signal.  Preserve the candidate at its pre-halt risk so frequency and
    # blocker diagnostics remain invariant under stress.
    sizing_drawdown_mult = drawdown_mult if drawdown_mult > 0.0 else 1.0
    target_risk = equity * unit_risk_pct * sizing_drawdown_mult * dynamic_mult * size_mult
    if target_risk <= 0:
        return None
    quantity_float = target_risk / risk_per_share
    requested_quantity = int(floor(quantity_float + 1e-12))
    sized_risk = (
        risk_per_share * requested_quantity
        if requested_quantity > 0
        else target_risk
    )
    scale = requested_quantity / source_quantity if source_quantity > 0 else 0.0
    r_multiple = float(trade.r_multiple or 0.0)
    pnl = r_multiple * sized_risk
    heat_r = sized_risk / max(
        equity
        * float(
            effective["portfolio_rules"].get("reference_risk_pct", 0.006)
        ),
        1.0,
    )
    return ReplayCandidate(
        strategy=strategy,
        trade=trade,
        risk_dollars=sized_risk,
        pnl=pnl,
        r_multiple=r_multiple,
        heat_r=heat_r,
        quality=quality,
        size_mult=scale,
        requested_quantity=requested_quantity,
        requested_notional=(
            float(trade.entry_price) * max(requested_quantity, 0)
        ),
    )


def _block_reason(
    candidate: ReplayCandidate,
    active: list[PortfolioPosition],
    accepted: list[PortfolioPosition],
    effective: dict[str, Any],
    equity: float,
    reference_risk_pct: float,
) -> str:
    rules = effective["portfolio_rules"]
    allocation = effective["strategy_allocations"][candidate.strategy]
    cross = effective.get("cross_strategy_rules", {})
    reference_risk = max(equity * reference_risk_pct, 1.0)
    current_heat = sum(position.risk_dollars for position in active) / reference_risk
    strategy_heat = sum(position.risk_dollars for position in active if position.strategy == candidate.strategy) / reference_risk
    symbol_heat = sum(position.risk_dollars for position in active if position.symbol == candidate.trade.symbol) / reference_risk
    sector_heat = sum(position.risk_dollars for position in active if position.sector == candidate.trade.sector) / reference_risk
    long_heat = sum(
        position.risk_dollars
        for position in active
        if position.direction == Direction.LONG
    ) / reference_risk
    strategy_open = sum(1 for position in active if position.strategy == candidate.strategy)

    if len(active) >= int(rules.get("max_total_active_positions", 999)):
        return "max_total_active_positions"
    if strategy_open >= int(allocation.get("max_concurrent", 999)):
        return "strategy_max_concurrent"
    if current_heat + candidate.heat_r > float(rules.get("heat_cap_R", 999.0)):
        return "portfolio_heat_cap"
    if strategy_heat + candidate.heat_r > float(allocation.get("max_heat_R", 999.0)):
        return "strategy_heat_cap"
    if symbol_heat + candidate.heat_r > float(rules.get("max_symbol_heat_R", 999.0)):
        return "symbol_heat_cap"
    if candidate.trade.direction == Direction.LONG and long_heat + candidate.heat_r > float(rules.get("max_long_heat_R", 999.0)):
        return "long_heat_cap"
    if sector_heat + candidate.heat_r > float(cross.get("same_sector_heat_cap_R", 999.0)):
        return "sector_heat_cap"

    same_symbol_active = [position for position in active if position.symbol == candidate.trade.symbol]
    same_symbol_policy = str(cross.get("same_symbol_policy", "half_size"))
    if same_symbol_policy == "best_rank_only" and same_symbol_active:
        best_active_quality = max(position.quality for position in same_symbol_active)
        if candidate.quality <= best_active_quality:
            return "same_symbol_lower_rank"

    max_share = float(rules.get("max_single_strategy_trade_share", 1.0))
    if accepted and max_share < 1.0:
        future_count = sum(1 for position in accepted if position.strategy == candidate.strategy) + 1
        future_total = len(accepted) + 1
        if future_count / future_total > max_share and future_total > 50:
            return "strategy_trade_share_cap"

    return ""


def _initial_margin_rate(direction: Direction, account: dict[str, Any]) -> float:
    key = (
        "initial_margin_long_pct"
        if direction == Direction.LONG
        else "initial_margin_short_pct"
    )
    fallback = 0.50 if direction == Direction.LONG else 0.60
    return max(float(account.get(key, fallback) or fallback), 0.0)


def _maintenance_margin_rate(
    direction: Direction,
    account: dict[str, Any],
) -> float:
    key = (
        "maintenance_margin_long_pct"
        if direction == Direction.LONG
        else "maintenance_margin_short_pct"
    )
    fallback = 0.25 if direction == Direction.LONG else 0.30
    return max(float(account.get(key, fallback) or fallback), 0.0)


def _mark_price_scale(
    trade: TradeRecord,
    mark_price_provider: MarkPriceProvider | None,
) -> float:
    if mark_price_provider is None:
        return 1.0
    raw = mark_price_provider(trade.symbol, trade.entry_time)
    if raw is None or not isfinite(float(raw)) or float(raw) <= 0.0:
        return 1.0
    ratio = float(raw) / max(float(trade.entry_price), 1e-9)
    if 1.0 / 3.0 <= ratio <= 3.0:
        return 1.0
    return 10.0 ** (-round(float(np.log10(ratio))))


def _accrue_financing(
    state: PortfolioCoreState,
    at: datetime,
    effective: dict[str, Any],
) -> None:
    at_cmp = _naive_dt(at)
    if state.last_account_time is None:
        state.last_account_time = at
        return
    prior_cmp = _naive_dt(state.last_account_time)
    elapsed_seconds = max((at_cmp - prior_cmp).total_seconds(), 0.0)
    state.last_account_time = at
    if elapsed_seconds <= 0.0:
        return

    account = effective.get("account_rules", {})
    debit_rate = max(float(account.get("annual_margin_interest_rate", 0.0) or 0.0), 0.0)
    credit_rate = max(float(account.get("annual_cash_interest_rate", 0.0) or 0.0), 0.0)
    years = elapsed_seconds / (365.25 * 24.0 * 3600.0)
    debit_cost = max(-float(state.cash), 0.0) * debit_rate * years
    credit_income = max(float(state.cash), 0.0) * credit_rate * years
    net_cost = debit_cost - credit_income
    if abs(net_cost) <= 1e-12:
        return
    state.cash -= net_cost
    state.equity -= net_cost
    state.financing_cost += net_cost
    state.equity_points.append(float(state.equity))
    state.equity_times.append(at_cmp)


def _refresh_account_state(
    state: PortfolioCoreState,
    at: datetime,
    effective: dict[str, Any],
    *,
    mark_price_provider: MarkPriceProvider | None,
) -> None:
    account = effective.get("account_rules", {})
    unrealized = 0.0
    gross_notional = 0.0
    net_notional = 0.0
    initial_margin = 0.0
    maintenance_margin = 0.0
    overnight_gross_notional = 0.0
    for position in state.active_positions:
        state.mark_observation_count += 1
        raw_mark = (
            mark_price_provider(position.symbol, at)
            if mark_price_provider is not None
            else None
        )
        if raw_mark is None or not isfinite(float(raw_mark)) or float(raw_mark) <= 0.0:
            state.missing_mark_count += 1
            mark = float(position.current_mark or position.entry_price)
        else:
            mark = float(raw_mark) * float(position.mark_price_scale)
        position.current_mark = mark
        position.last_mark_time = at
        notional = abs(mark * position.quantity)
        gross_notional += notional
        if position.exit_time.date() > position.entry_time.date():
            overnight_gross_notional += notional
        net_notional += float(position.direction) * mark * position.quantity
        unrealized += (
            (mark - position.entry_price)
            * float(position.direction)
            * position.quantity
        )
        initial_margin += notional * _initial_margin_rate(position.direction, account)
        maintenance_margin += notional * _maintenance_margin_rate(
            position.direction,
            account,
        )

    nlv = float(state.equity) + unrealized
    state.net_liquidation_value = nlv
    state.peak_net_liquidation_value = max(state.peak_net_liquidation_value, nlv)
    state.peak_equity = max(state.peak_equity, nlv)
    state.gross_notional_peak = max(state.gross_notional_peak, gross_notional)
    state.net_notional_peak_abs = max(state.net_notional_peak_abs, abs(net_notional))
    state.initial_margin_peak = max(state.initial_margin_peak, initial_margin)
    state.maintenance_margin_peak = max(
        state.maintenance_margin_peak,
        maintenance_margin,
    )
    leverage = gross_notional / max(nlv, 1.0)
    state.gross_leverage_peak = max(state.gross_leverage_peak, leverage)
    state.net_leverage_peak_abs = max(
        state.net_leverage_peak_abs,
        abs(net_notional) / max(nlv, 1.0),
    )
    state.overnight_gross_leverage_peak = max(
        state.overnight_gross_leverage_peak,
        overnight_gross_notional / max(nlv, 1.0),
    )
    margin_buffer = (nlv - maintenance_margin) / max(nlv, 1.0)
    state.minimum_margin_buffer_pct = min(
        state.minimum_margin_buffer_pct,
        margin_buffer,
    )
    breached = bool(account.get("enforce_shared_buying_power", False)) and (
        margin_buffer < 0.0
    )
    if breached and not state.in_margin_breach:
        state.margin_breach_count += 1
    state.in_margin_breach = breached

    at_cmp = _naive_dt(at)
    if not state.nlv_times or state.nlv_times[-1] != at_cmp:
        state.nlv_times.append(at_cmp)
        state.nlv_points.append(nlv)
    else:
        state.nlv_points[-1] = nlv


def _close_positions(
    state: PortfolioCoreState,
    *,
    before: datetime,
    actions: list[PortfolioAction],
    trade_outcomes: list[TradeOutcome],
    effective: dict[str, Any],
    mark_price_provider: MarkPriceProvider | None,
) -> None:
    before_cmp = _naive_dt(before)
    closing = sorted(
        (position for position in state.active_positions if _naive_dt(position.exit_time) <= before_cmp),
        key=lambda item: item.exit_time,
    )
    for position in closing:
        _accrue_financing(state, position.exit_time, effective)
        _refresh_account_state(
            state,
            position.exit_time,
            effective,
            mark_price_provider=mark_price_provider,
        )
        state.cash += (
            float(position.direction) * position.entry_notional + position.pnl
        )
        state.equity = float(state.equity) + position.pnl
        state.active_positions.remove(position)
        state.peak_equity = max(float(state.peak_equity), state.equity)
        state.equity_points.append(state.equity)
        state.equity_times.append(_naive_dt(position.exit_time))
        reference_risk = max(state.equity * float(state.reference_risk_pct), 1.0)
        realized_r = position.pnl / reference_risk
        state.daily_realized_r[position.exit_time.date().isoformat()] = (
            state.daily_realized_r.get(position.exit_time.date().isoformat(), 0.0) + realized_r
        )
        iso = position.exit_time.isocalendar()
        weekly_key = f"{iso.year:04d}-W{iso.week:02d}"
        state.weekly_realized_r[weekly_key] = state.weekly_realized_r.get(weekly_key, 0.0) + realized_r
        strategy_day_key = f"{position.strategy}|{position.exit_time.date().isoformat()}"
        native_r = position.pnl / max(position.risk_dollars, 1e-9)
        state.strategy_daily_realized_r[strategy_day_key] = (
            state.strategy_daily_realized_r.get(strategy_day_key, 0.0) + native_r
        )
        state.strategy_recent[position.strategy].append(position.r_multiple)

        action = PortfolioAction(
            action_type=PortfolioActionType.SUBMIT_EXIT,
            timestamp=position.exit_time,
            strategy_id=position.strategy,
            symbol=position.symbol,
            reason=position.exit_reason,
            risk_dollars=position.risk_dollars,
            metadata={
                "entry_type": position.entry_type,
                "r_multiple": float(position.r_multiple),
                "quality": float(position.quality),
            },
        )
        actions.append(action)
        trade_outcomes.append(
            TradeOutcome(
                strategy_id=position.strategy,
                symbol=position.symbol,
                entry_time=position.entry_time,
                decision_time=position.decision_time,
                fill_time=position.fill_time,
                exit_time=position.exit_time,
                gross_pnl=position.pnl + position.commission,
                commission=position.commission,
                net_pnl=position.pnl,
                r_multiple=position.r_multiple,
                risk_dollars=position.risk_dollars,
                exit_reason=position.exit_reason,
                route=position.entry_type,
                metadata=dict(position.metadata),
            )
        )
    _accrue_financing(state, before, effective)
    _refresh_account_state(
        state,
        before,
        effective,
        mark_price_provider=mark_price_provider,
    )


def _naive_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.replace(tzinfo=None)


def _rank_candidate(candidate: ReplayCandidate, effective: dict[str, Any]) -> float:
    mode = str(effective.get("cross_strategy_rules", {}).get("candidate_rank_mode", "diagnostic_alpha_score"))
    priority = float(effective["strategy_allocations"][candidate.strategy].get("priority", 5))
    priority_score = 1.0 / (1.0 + priority)
    if mode == "frequency_first":
        return 0.70 * priority_score + 0.30 * candidate.quality
    if mode == "expected_alpha_per_heat":
        return _expected_candidate_r(candidate.trade) / max(candidate.heat_r, 0.10)
    if mode == "expected_net_r":
        uncertainty = max(
            _meta_float(
                candidate.trade.metadata or {},
                "portfolio_expected_r_uncertainty",
                0.0,
            ),
            0.0,
        )
        penalty = float(
            effective.get("cross_strategy_rules", {}).get(
                "alpha_uncertainty_penalty", 0.0
            )
            or 0.0
        )
        return _expected_candidate_r(candidate.trade) - penalty * uncertainty
    if mode == "strategy_priority":
        return priority_score
    return 0.55 * candidate.quality + 0.45 * priority_score


def _candidate_quality(strategy: str, trade: TradeRecord, effective: dict[str, Any]) -> float:
    meta = trade.metadata or {}
    if "portfolio_expected_r" in meta:
        uncertainty = max(
            _meta_float(meta, "portfolio_expected_r_uncertainty", 0.0),
            0.0,
        )
        penalty = float(
            effective.get("cross_strategy_rules", {}).get(
                "alpha_uncertainty_penalty", 0.0
            )
            or 0.0
        )
        return _expected_candidate_r(trade) - penalty * uncertainty
    quality = 0.55
    if strategy == CURRENT_IARIC_ID:
        score = _meta_float(meta, "residual_score", _meta_float(meta, "score", 0.0))
        failed = _meta_float(meta, "failed_continuation_r", 0.0)
        quality = 0.55 + min(max(score, 0.0), 100.0) / 250.0
        quality += min(max(failed, 0.0), 2.0) * 0.08
    elif strategy.startswith("IARIC"):
        quality = 0.72
        route = (trade.entry_type or "").upper()
        if route == "OPEN_SCORED_ENTRY":
            quality += 0.08
        elif route == "DELAYED_CONFIRM":
            quality += 0.03
        elif route in {"VWAP_BOUNCE", "AFTERNOON_RETEST"}:
            quality -= 0.06
        quality += _scaled_meta(meta, "daily_signal_score", 72.0, 92.0, 0.12)
        quality += _scaled_meta(meta, "intraday_score", 72.0, 92.0, 0.10)
        gap = _meta_float(meta, "entry_gap_pct", 0.0)
        if gap < -0.5:
            quality += 0.08
        elif gap > 0.5:
            quality -= 0.08
    elif not strategy.startswith("IARIC"):
        quality = 0.66
        entry_type = (trade.entry_type or "").upper()
        if entry_type == "PDH_BREAKOUT":
            quality += 0.12
        elif entry_type == "OR_BREAKOUT":
            quality += 0.05
        elif entry_type == "COMBINED_BREAKOUT":
            quality += 0.02
        score = _meta_float(meta, "momentum_score", _meta_float(meta, "score", 5.0))
        quality += max(min((score - 5.0) * 0.04, 0.12), -0.12)
        rvol = _meta_float(meta, "rvol", _meta_float(meta, "entry_rvol", 2.5))
        if rvol >= 3.0:
            quality += 0.08
        if trade.sector == "Financials":
            quality -= 0.12
    return max(0.05, min(1.25, quality))


def _expected_candidate_r(trade: TradeRecord) -> float:
    return _meta_float(trade.metadata or {}, "portfolio_expected_r", 0.0)


def _candidate_size_mult(strategy: str, trade: TradeRecord, effective: dict[str, Any]) -> float:
    filters = effective.get("strategy_filters", {}).get(strategy, {})
    mult = 1.0
    if strategy == "ALCB_R3":
        if trade.sector == "Financials":
            mult *= float(filters.get("financials_size_mult", 1.0) or 1.0)
        if (trade.entry_type or "").upper() == "PDH_BREAKOUT":
            mult *= float(filters.get("pdh_size_mult", 1.0) or 1.0)
        score = _meta_float(trade.metadata or {}, "momentum_score", _meta_float(trade.metadata or {}, "score", 5.0))
        has_surge = bool((trade.metadata or {}).get("bar_vol_surge", False))
        if int(round(score)) == 5 and not has_surge:
            mult *= float(filters.get("score5_no_surge_mult", 1.0) or 1.0)
    elif strategy != CURRENT_IARIC_ID:
        gap = _meta_float(trade.metadata or {}, "entry_gap_pct", 0.0)
        if gap > 0:
            mult *= float(filters.get("gap_up_size_mult", 1.0) or 1.0)
        if "CARRY" in (trade.exit_reason or "").upper():
            mult *= float(filters.get("carry_route_size_mult", 1.0) or 1.0)
    return max(0.0, min(1.5, mult))


def _dynamic_mult(strategy: str, strategy_recent: dict[str, deque[float]], effective: dict[str, Any]) -> float:
    dynamic = effective.get("dynamic_allocation", {})
    if not dynamic.get("enabled", False):
        return 1.0
    recent = strategy_recent.get(strategy)
    if not recent or len(recent) < 20:
        return 1.0
    avg_r = float(np.mean(recent))
    win_rate = sum(1 for value in recent if value > 0) / len(recent)
    mult = 1.0
    if avg_r > 0.20 and win_rate > 0.58:
        mult += float(dynamic.get("positive_expectancy_boost", 0.10) or 0.10)
    elif avg_r < 0.0 or win_rate < 0.45:
        mult -= float(dynamic.get("negative_expectancy_cut", 0.18) or 0.18)
    return max(float(dynamic.get("min_mult", 0.65)), min(float(dynamic.get("max_mult", 1.20)), mult))


def _drawdown_mult(equity: float, peak_equity: float, effective: dict[str, Any]) -> float:
    if peak_equity <= 0:
        return 1.0
    dd = max(0.0, (peak_equity - equity) / peak_equity)
    tiers = effective.get("portfolio_rules", {}).get("drawdown_tiers", ())
    mult = 1.0
    for threshold, tier_mult in tiers:
        if dd >= float(threshold):
            mult = float(tier_mult)
    return max(0.0, mult)


def _candidate_discrimination(accepted_r: list[float], blocked_r: list[float]) -> float:
    if not accepted_r:
        return 0.0
    if not blocked_r:
        return 1.0
    delta = float(np.mean(accepted_r) - np.mean(blocked_r))
    return max(0.0, min(1.0, 0.50 + delta / 0.60))


def _positive_slices(positions: list[PortfolioPosition]) -> int:
    if not positions:
        return 0
    ordered = sorted(positions, key=lambda item: item.entry_time)
    chunks = [chunk for chunk in np.array_split(np.array(ordered, dtype=object), 4) if len(chunk)]
    return int(sum(1 for chunk in chunks if sum(position.pnl for position in chunk) > 0))


def _months_from_positions(
    accepted: list[PortfolioPosition],
    blocked: list[BlockedCandidate],
) -> float:
    dates = [position.entry_time for position in accepted]
    dates.extend(candidate.entry_time for candidate in blocked)
    if len(dates) < 2:
        return 1.0
    start = min(dates)
    end = max(dates)
    span_days = max((end - start).total_seconds() / 86400.0, 1.0)
    return span_days / 30.4375


def _block_reason_metrics(blocked: list[BlockedCandidate]) -> dict[str, float]:
    total = len(blocked)
    if total <= 0:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for candidate in blocked:
        counts[candidate.reason] += 1
    return {f"blocked_reason_{reason}": float(count) for reason, count in counts.items()}


def _scaled_meta(meta: dict[str, Any], key: str, low: float, high: float, weight: float) -> float:
    value = _meta_float(meta, key, low)
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low))) * weight


def _meta_float(meta: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default
