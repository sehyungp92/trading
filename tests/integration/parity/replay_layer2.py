from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from tests.integration.parity.replay_candidates import (
    ReplayDecisionTimeline,
    broker_event_key as _broker_event_key,
)
from tests.integration.parity.source_inputs import (
    iaric_artifact,
    iaric_minute_bars,
    iaric_quote,
    iaric_state_snapshot,
    nq_bar_data,
    nq_daily_context,
    nq_live_context,
    nqdtc_r1b_market_input,
    parse_time,
    source_bars,
    tpc_bar_input,
    tpc_symbol_config,
    vdub_r1b_market_input,
)


def _replay_tpc(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> None:
    from strategies.swing.tpc.core import logic
    from strategies.swing.tpc.core.serializers import restore_state, snapshot_state

    state = restore_state((fixture.get("initial_strategy_state", {}) or {}).get("TPC", {}))
    symbols = {str(row["symbol"]) for row in fixture.get("bars", []) if str(row.get("timeframe", "")).lower() == "15m"}
    for symbol in sorted(symbols):
        cfg = tpc_symbol_config(fixture, symbol)
        replay = run_replay(
            state,
            steps=[ReplayStep(bar_input=tpc_bar_input(fixture, symbol))],
            on_bar=lambda current, bar_input, cfg=cfg: logic.on_bar(current, bar_input, cfg),
            on_order_update=logic.on_order_update,
            on_fill=logic.on_fill,
        )
        state = replay.state
        out.record_actions("TPC", replay.actions)
    out.strategy_state["TPC"] = {
        "setups": sorted(snapshot_state(state).get("setups", {}).keys()),
        "positions": sorted(snapshot_state(state).get("positions", {}).keys()),
        "pending_count": len(snapshot_state(state).get("pending_orders", {}) or {}),
    }


def _replay_nq_regime(
    fixture: Mapping[str, Any],
    out: ReplayDecisionTimeline,
    *,
    causal_authorization: bool = False,
    portfolio_authorizer=None,
    market_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    from libs.oms.risk.portfolio_rules import (
        PortfolioRulesConfig,
        adjusted_entry_quantity,
        evaluate_static_portfolio_entry,
        require_static_portfolio_config,
    )
    from strategies.core.actions import SubmitEntry
    from strategies.momentum.nq_regime.config import StrategyRuntimeSettings
    from strategies.momentum.nq_regime.core.data_policy import CompletedBarPolicy
    from strategies.momentum.nq_regime.core.logic import on_authorization, on_bar, on_fill
    from strategies.momentum.nq_regime.core.serializers import hydrate_state, snapshot_state
    from strategies.momentum.nq_regime.core.state import AuthorizationEvent, FillEvent
    from tests.integration.parity.portfolio_rules import portfolio_rules_config_from_fixture

    settings = StrategyRuntimeSettings(
        initial_equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
        max_contracts=int(((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {}).get("max_contracts", 5)),
        enable_liquidity_reversion=False,
        enable_second_wind=False,
    )
    state = hydrate_state((fixture.get("initial_strategy_state", {}) or {}).get("NQ_REGIME", {}))
    policy = CompletedBarPolicy()
    portfolio_rules = None
    if causal_authorization:
        family_strategies = (fixture.get("family_config", {}) or {}).get("strategies", []) or []
        portfolio_rules = (
            portfolio_rules_config_from_fixture(fixture)
            if family_strategies
            else PortfolioRulesConfig(
                initial_equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
                nqdtc_direction_filter_enabled=False,
                directional_cap_R=0.0,
                dd_tiers=((1.0, 1.0),),
            )
        )
        if portfolio_rules is None:
            raise AssertionError("NQ_REGIME R1A fixture must materialize portfolio rules")
        if portfolio_authorizer is None:
            require_static_portfolio_config(portfolio_rules)
    current_equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    entry_sequence = 0
    rows = market_rows or (
        source_bars(fixture, "NQ", "5m")
        or source_bars(fixture, "MNQ", "5m")
    )
    for row in rows:
        bar = nq_bar_data(row)
        step = ReplayStep(
            bar_input=policy.build_event(
                bar_5m=bar,
                recent_5m=[*state.bars_5m, bar],
                daily_context=nq_daily_context(fixture),
                live_context=nq_live_context(fixture),
            )
        )
        replay = run_replay(
            state,
            steps=[step],
            on_bar=lambda current, event: on_bar(
                current,
                event,
                scheduled_news=[],
                settings=settings,
                authorization_required=causal_authorization,
            ),
            on_order_update=lambda current, update: (current, [], []),
            on_fill=lambda current, fill: on_fill(current, fill, settings=settings),
        )
        state = replay.state
        if not causal_authorization:
            out.record_actions("NQ_REGIME", replay.actions)
            continue
        for action in replay.actions:
            if not isinstance(action, SubmitEntry):
                out.record_actions("NQ_REGIME", [action])
                continue
            entry_sequence += 1
            assert portfolio_rules is not None
            if portfolio_authorizer is not None:
                authorization = portfolio_authorizer(
                    "NQ_REGIME",
                    action,
                    timestamp=bar.ts,
                )
                approved = bool(authorization["approved"])
                approved_qty = int(authorization["approved_qty"])
                denial_reason = str(authorization.get("denial_reason", ""))
                portfolio_decision_ref = str(
                    authorization.get("portfolio_decision_ref", "")
                )
                family_surface = str(authorization.get("family_surface", ""))
            else:
                portfolio_decision = evaluate_static_portfolio_entry(
                    portfolio_rules,
                    strategy_id="NQ_REGIME",
                    direction="LONG" if action.side == "BUY" else "SHORT",
                    current_equity=current_equity,
                )
                approved = portfolio_decision.approved
                approved_qty = (
                    adjusted_entry_quantity(action.qty, portfolio_decision.size_multiplier)
                    if approved
                    else 0
                )
                denial_reason = portfolio_decision.denial_reason or ""
                portfolio_decision_ref = ""
                family_surface = "nq_regime_r1a_static_portfolio"
            state, submissions, _authorization_events = on_authorization(
                state,
                AuthorizationEvent(
                    client_order_id=action.client_order_id,
                    approved=approved,
                    approved_qty=approved_qty,
                    requested_qty=action.qty,
                    timestamp=bar.ts,
                    symbol=action.symbol,
                    denial_reason=denial_reason,
                    portfolio_decision_ref=portfolio_decision_ref,
                ),
                settings=settings,
            )
            status = (
                "rejected"
                if approved_qty <= 0
                else "accepted"
                if approved_qty == action.qty
                else "reduced"
            )
            decision = {
                "strategy_id": "NQ_REGIME",
                "symbol": action.symbol,
                "side": action.side,
                "role": "ENTRY",
                "status": status,
                "reason": (
                    f"Portfolio rule: {denial_reason}"
                    if denial_reason
                    else ""
                ),
                "family_surface": family_surface,
                "candidate_key": f"NQ_REGIME|{action.symbol}|ENTRY|{action.side}|{entry_sequence}",
                "sequence": entry_sequence,
                "original_qty": action.qty,
                "approved_qty": approved_qty,
                "order_match": {
                    "strategy_id": "NQ_REGIME",
                    "symbol": action.symbol,
                    "role": "ENTRY",
                    "side": action.side,
                    "sequence": entry_sequence,
                },
            }
            if submissions:
                out.record_actions("NQ_REGIME", submissions, decision=decision)
            else:
                out.record_family_rejection("NQ_REGIME", action, decision)
    for event in fixture.get("broker_event_script", []):
        if str((event.get("order_match", {}) or {}).get("strategy_id")) != "NQ_REGIME":
            continue
        key = _broker_event_key(event)
        if key in out._applied:
            continue
        order = out._match_order(event.get("order_match", {}))
        if order is None:
            continue
        out.note_broker_event(order, event)
        out._applied.add(key)
        fill = FillEvent(
            oms_order_id=str(order["client_tag"]),
            fill_price=float(event.get("price", order.get("limit_price") or 0.0)),
            fill_qty=int(float(event.get("qty", order["qty"]))),
            fill_time=parse_time(event.get("timestamp")),
            symbol=str(order["symbol"]),
            commission=float(event.get("commission", 0.0)),
            order_role="entry",
        )
        state, actions, _events = on_fill(state, fill, settings=settings)
        out.record_actions("NQ_REGIME", actions)
    snap = snapshot_state(state)
    out.strategy_state["NQ_REGIME"] = {
        "position_side": str(snap.get("position_side", "")),
        "entry_price": snap.get("entry_price", 0.0),
        "stop_price": snap.get("stop_price", 0.0),
        "qty_open": snap.get("qty_open", 0),
        "daily_trades": snap.get("daily_trades", 0),
        "last_decision_code": snap.get("last_decision_code", ""),
    }


def _replay_nqdtc_r1b(
    fixture: Mapping[str, Any],
    out: ReplayDecisionTimeline,
    *,
    portfolio_authorizer,
) -> None:
    """Replay the bounded raw NQDTC decision and lifecycle on the R1B timeline."""

    from strategies.core.actions import SubmitEntry
    from strategies.momentum.nqdtc.core.entry_decision import (
        NQDTCEntryDecisionSnapshot,
        evaluate_entry_decision,
    )
    from strategies.momentum.nqdtc.core.logic import (
        on_authorization,
        on_bar,
        on_fill,
        on_order_update,
    )
    from strategies.momentum.nqdtc.core.serializers import restore_state, snapshot_state
    from strategies.momentum.nqdtc.core.state import (
        NQDTCAuthorization,
        NQDTCEntryFillContext,
        NQDTCEntryRequest,
        NQDTCFill,
        NQDTCOrderUpdate,
    )
    from strategies.momentum.nqdtc.models import (
        Direction,
        ExitTier,
        Regime4H,
        RegimeState,
        Session,
        SessionEngineState,
    )

    market_input = nqdtc_r1b_market_input(fixture)
    rows = market_input["bars"]
    state_input = market_input["decision_state"]
    session = Session[str(state_input.get("session", "RTH")).upper()]
    direction = Direction[str(state_input.get("direction", "LONG")).upper()]
    session_state = SessionEngineState(session=session)
    session_state.breakout.active = True
    session_state.breakout.direction = direction
    session_state.breakout.breakout_bar_high = float(
        state_input.get("breakout_bar_high", rows[-1]["high"])
    )
    session_state.breakout.breakout_bar_low = float(
        state_input.get("breakout_bar_low", rows[-1]["low"])
    )
    session_state.box.box_high = float(state_input.get("box_high", rows[-1]["high"]))
    session_state.box.box_low = float(state_input.get("box_low", rows[-1]["low"]))
    session_state.box.box_mid = float(
        state_input.get(
            "box_mid",
            (session_state.box.box_high + session_state.box.box_low) / 2.0,
        )
    )
    session_state.box.box_width = session_state.box.box_high - session_state.box.box_low
    session_state.atr14_30m = float(state_input.get("atr14_30m", 20.0))
    session_state.last_score = float(state_input.get("score", 3.0))
    session_state.last_disp_metric = float(state_input.get("disp_metric", 2.0))
    session_state.last_disp_threshold = float(state_input.get("disp_threshold", 1.0))
    session_state.disp_hist.data = [
        float(value) for value in state_input.get("disp_history", [1.0] * 12)
    ]
    vwap = float(state_input.get("vwap", rows[-1]["close"]))
    session_state.vwap_session.cum_tpv = vwap
    session_state.vwap_session.cum_vol = 1.0
    regime = RegimeState(
        regime_4h=Regime4H[str(state_input.get("regime_4h", "TRANSITIONAL")).upper()],
        trend_dir_4h=Direction[str(state_input.get("trend_dir_4h", "FLAT")).upper()],
    )
    bars_5m = {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in ("open", "high", "low", "close", "volume")
    }
    bars_15m_rows = market_input.get("bars_15m") or rows
    bars_15m = {
        "close": np.asarray(
            [float(row["close"]) for row in bars_15m_rows],
            dtype=float,
        )
    }
    decision = evaluate_entry_decision(
        NQDTCEntryDecisionSnapshot(
            now=market_input["timestamp"],
            symbol=market_input["symbol"],
            equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
            engine=session_state,
            regime=regime,
            bars_5m=bars_5m,
            bars_15m=bars_15m,
            bars_daily={"ema50": np.array([]), "atr14": np.array([])},
            entry_oca_group="R1B_ENTRY",
            a_oca_group="R1B_A_OCO",
            entry_a_retest=True,
            entry_a_latch=False,
            entry_b_sweep=True,
            entry_c_standard=True,
            entry_c_continuation=False,
            continuation_mode=True,
            friction_gate=True,
            tp1_viability_gate=True,
            recompute_composite=True,
            c_stop_reference="entry_price",
            fallback_order_type="LIMIT",
            fallback_tif="IOC",
        )
    )
    core_state = restore_state(
        ((fixture.get("initial_strategy_state", {}) or {}).get("NQDTC_v2.1", {}) or {}).get(
            "core",
            {},
        )
    )
    core_state.bar_count_5m += len(rows)
    accepted_requests: dict[str, NQDTCEntryRequest] = {}
    entry_sequence = 0

    for plan in decision.plans:
        entry_sequence += 1
        request = NQDTCEntryRequest(
            client_order_id=(
                f"NQDTC_v2.1:{plan.subtype.value}:"
                f"{core_state.bar_count_5m}:{len(core_state.working_orders)}"
            ),
            symbol=market_input["symbol"],
            subtype=plan.subtype,
            direction=plan.direction,
            qty=plan.qty,
            stop_for_risk=plan.stop_for_risk,
            tif=plan.tif,
            order_type=plan.order_type,
            price=plan.price,
            limit_price=plan.price,
            stop_price=plan.stop_price,
            oca_group=plan.oca_group,
            is_limit=plan.is_limit,
            quality_mult=plan.quality_mult,
            submitted_bar_idx=core_state.bar_count_5m,
            ttl_bars=plan.ttl_bars,
        )
        core_state, proposals, _events = on_bar(
            core_state,
            bar_count_5m=core_state.bar_count_5m,
            bar_ts=market_input["timestamp"],
            entry_request=request,
            authorization_required=True,
        )
        proposal = next(action for action in proposals if isinstance(action, SubmitEntry))
        authorization = portfolio_authorizer(
            "NQDTC_v2.1",
            proposal,
            timestamp=market_input["timestamp"],
        )
        denial_reason = str(authorization.get("denial_reason", ""))
        strategy_denial_reason = (
            f"Portfolio rule: {denial_reason}" if denial_reason else ""
        )
        core_state, submissions, _events = on_authorization(
            core_state,
            NQDTCAuthorization(
                client_order_id=request.client_order_id,
                approved=bool(authorization["approved"]),
                approved_qty=int(authorization["approved_qty"]),
                requested_qty=request.qty,
                timestamp=market_input["timestamp"],
                symbol=request.symbol,
                denial_reason=strategy_denial_reason,
                portfolio_decision_ref=str(
                    authorization.get("portfolio_decision_ref", "")
                ),
            ),
        )
        approved_qty = int(authorization["approved_qty"])
        status = (
            "rejected"
            if approved_qty <= 0
            else "accepted"
            if approved_qty == request.qty
            else "reduced"
        )
        family_decision = {
            "strategy_id": "NQDTC_v2.1",
            "symbol": proposal.symbol,
            "side": proposal.side,
            "role": "ENTRY",
            "status": status,
            "reason": (
                strategy_denial_reason
            ),
            "family_surface": str(authorization.get("family_surface", "")),
            "candidate_key": (
                f"NQDTC_v2.1|{proposal.symbol}|ENTRY|{proposal.side}|{entry_sequence}"
            ),
            "sequence": entry_sequence,
            "original_qty": request.qty,
            "approved_qty": approved_qty,
            "order_match": {
                "strategy_id": "NQDTC_v2.1",
                "symbol": proposal.symbol,
                "role": "ENTRY",
                "side": proposal.side,
                "sequence": entry_sequence,
            },
        }
        if submissions:
            approved_request = replace(request, qty=approved_qty)
            accepted_requests[request.client_order_id] = approved_request
            core_state, _, _events = on_order_update(
                core_state,
                NQDTCOrderUpdate(
                    oms_order_id=request.client_order_id,
                    status="accepted",
                    timestamp=market_input["timestamp"],
                    order_role="entry",
                    accepted_entry=approved_request,
                ),
            )
            out.record_actions("NQDTC_v2.1", submissions, decision=family_decision)
        else:
            out.record_family_rejection("NQDTC_v2.1", proposal, family_decision)

    for event in fixture.get("broker_event_script", []) or []:
        match = event.get("order_match", {}) or {}
        if str(match.get("strategy_id")) != "NQDTC_v2.1":
            continue
        key = _broker_event_key(event)
        order = out._match_order(match)
        if order is None:
            continue
        out.note_broker_event(order, event)
        out._applied.add(key)
        request = accepted_requests[str(order["client_tag"])]
        core_state, actions, _events = on_fill(
            core_state,
            NQDTCFill(
                oms_order_id=str(order["client_tag"]),
                fill_price=float(event.get("price", order.get("limit_price") or 0.0)),
                fill_qty=int(float(event.get("qty", order["qty"]))),
                fill_time=parse_time(event.get("timestamp")),
                entry_context=NQDTCEntryFillContext(
                    exit_tier=ExitTier.NEUTRAL,
                    tp_levels=[],
                    mm_level=session_state.breakout.mm_level,
                    mm_reached=session_state.breakout.mm_reached,
                    box_high_at_entry=session_state.box.box_high,
                    box_low_at_entry=session_state.box.box_low,
                    box_mid_at_entry=session_state.box.box_mid,
                    entry_session=session,
                    tp1_only_cap=False,
                    r_dollars=(
                        abs(
                            float(request.price or request.stop_price or 0.0)
                            - request.stop_for_risk
                        )
                        * request.qty
                    ),
                ),
            ),
        )
        out.record_actions("NQDTC_v2.1", actions)
        if actions:
            core_state, _, _events = on_order_update(
                core_state,
                NQDTCOrderUpdate(
                    oms_order_id=f"{order['client_tag']}:protective_stop",
                    status="accepted",
                    timestamp=parse_time(event.get("timestamp")),
                    order_role="stop",
                ),
            )

    snapshot = snapshot_state(core_state)
    out.strategy_state["NQDTC_v2.1"] = {
        "strategy_id": "NQDTC_v2.1",
        "last_bar_ts": snapshot.get("last_bar_ts"),
        "last_decision_code": snapshot.get("last_decision_code", "IDLE"),
        "last_decision_details": snapshot.get("last_decision_details", {}),
        "bar_count_5m": int(snapshot.get("bar_count_5m", 0) or 0),
        "working_order_count": len(snapshot.get("working_orders", []) or []),
        "position_open": bool((snapshot.get("position", {}) or {}).get("open", False)),
    }


def _replay_vdub_r1b(
    fixture: Mapping[str, Any],
    out: ReplayDecisionTimeline,
    *,
    portfolio_authorizer,
) -> None:
    """Replay Vdub raw input through its strategy-owned causal core."""

    from strategies.core.actions import SubmitEntry
    from strategies.momentum.vdub import config as C
    from strategies.momentum.vdub.core.entry_decision import (
        build_entry_proposal,
        evaluate_proposal_gates,
        select_entry_signal,
    )
    from strategies.momentum.vdub.core.logic import (
        on_authorization,
        on_bar,
        on_fill,
    )
    from strategies.momentum.vdub.core.serializers import restore_state, snapshot_state
    from strategies.momentum.vdub.core.state import (
        VdubAuthorization,
        VdubEntryFillContext,
        VdubFill,
    )
    from strategies.momentum.vdub.models import (
        DayCounters,
        Direction,
        RegimeState,
        SessionWindow,
        SubWindow,
        VolState,
    )

    market_input = vdub_r1b_market_input(fixture)
    state_input = market_input["decision_state"]
    point_value = float(
        state_input.get("point_value", C.NQ_SPECS["MNQ"]["point_value"])
    )
    C.NQ_SPEC["point_value"] = point_value
    C.NQ_SPEC["tick_value"] = C.NQ_SPEC["tick"] * point_value
    rows = market_input["bars_15m"]
    rows_1h = market_input.get("bars_1h", [])
    closes = np.asarray([float(row["close"]) for row in rows], dtype=float)
    lows = np.asarray([float(row["low"]) for row in rows], dtype=float)
    highs = np.asarray([float(row["high"]) for row in rows], dtype=float)
    direction = Direction[str(state_input.get("direction", "LONG")).upper()]
    session = SessionWindow[str(state_input.get("session", "RTH")).upper()]
    sub_window = SubWindow[str(state_input.get("sub_window", "OPEN")).upper()]
    regime = RegimeState(
        daily_trend=int(state_input.get("daily_trend", 1)),
        trend_1h=int(state_input.get("trend_1h", 1)),
        choppiness=float(state_input.get("choppiness", 10.0)),
        vol_state=VolState(str(state_input.get("vol_state", VolState.NORMAL.value))),
    )
    counters = DayCounters()
    atr15_values = np.asarray(state_input.get("atr15", []), dtype=float)
    atr1h_values = np.asarray(state_input.get("atr1h", []), dtype=float)
    selection = select_entry_signal(
        closes_15m=closes,
        lows_15m=lows,
        highs_15m=highs,
        svwap=np.asarray(state_input.get("svwap", []), dtype=float),
        vwap_a=np.asarray(state_input.get("vwap_a", []), dtype=float),
        pivots_1h=[],
        n_1h_bars=len(rows_1h),
        atr15=float(atr15_values[-1]),
        direction=direction,
        sub_window=sub_window,
        trend_1h=regime.trend_1h,
        type_a_enabled=True,
        type_b_enabled=True,
        type_c_enabled=False,
    )
    if selection is None:
        raise AssertionError("bounded Vdub raw input did not reach a shared signal")
    class_mult = float(state_input.get("class_mult", C.CLASS_MULT_NOPRED))
    proposal, reason = build_entry_proposal(
        selection=selection,
        symbol=market_input["symbol"],
        direction=direction,
        session=session,
        sub_window=sub_window,
        now=market_input["timestamp"],
        bar_idx=len(rows),
        bar_high=float(highs[-1]),
        bar_low=float(lows[-1]),
        close_price=float(closes[-1]),
        atr15=float(atr15_values[-1]),
        atr1h=float(atr1h_values[-1]),
        pivots_1h=[],
        regime=regime,
        counters=counters,
        positions=[],
        equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
        is_flip=False,
        class_mult=class_mult,
        session_mult=C.SESSION_MULT["RTH" if session == SessionWindow.RTH else "EVENING"],
        hourly_mult=C.HOURLY_ALIGNED_MULT,
        point_value=point_value,
        tick_size=C.NQ_SPEC["tick"],
        signal_id=f"{selection.entry_type.value}_{direction.name}_{len(rows)}",
        bar_id=f"{market_input['symbol']}:15m:{market_input['timestamp'].isoformat()}",
    )
    if proposal is None:
        raise AssertionError(f"bounded Vdub proposal was rejected: {reason}")
    approved, reason = evaluate_proposal_gates(
        proposal,
        counters=counters,
        open_risk=0.0,
    )
    if not approved:
        raise AssertionError(f"bounded Vdub strategy gate rejected proposal: {reason}")

    initial = (fixture.get("initial_strategy_state", {}) or {}).get(
        "VdubusNQ_v4",
        {},
    ) or {}
    core_state = restore_state(initial.get("core", initial))
    core_state.bar_idx += len(rows)
    core_state.regime = regime
    core_state.counters = counters
    core_state, proposal_actions, _events = on_bar(
        core_state,
        bar_ts=market_input["timestamp"],
        entry_proposal=proposal,
    )
    action = next(item for item in proposal_actions if isinstance(item, SubmitEntry))
    authorization = portfolio_authorizer(
        "VdubusNQ_v4",
        action,
        timestamp=market_input["timestamp"],
    )
    denial_reason = str(authorization.get("denial_reason", ""))
    strategy_reason = f"Portfolio rule: {denial_reason}" if denial_reason else ""
    approved_qty = int(authorization["approved_qty"])
    core_state, submissions, _events = on_authorization(
        core_state,
        VdubAuthorization(
            client_order_id=proposal.client_order_id,
            approved=bool(authorization["approved"]),
            approved_qty=approved_qty,
            requested_qty=proposal.qty,
            timestamp=market_input["timestamp"],
            oms_order_id=proposal.client_order_id,
            denial_reason=strategy_reason,
            portfolio_decision_ref=str(
                authorization.get("portfolio_decision_ref", "")
            ),
        ),
    )
    status = (
        "rejected"
        if approved_qty <= 0
        else "accepted"
        if approved_qty == proposal.qty
        else "reduced"
    )
    family_decision = {
        "strategy_id": "VdubusNQ_v4",
        "symbol": action.symbol,
        "side": action.side,
        "role": "ENTRY",
        "status": status,
        "reason": strategy_reason,
        "family_surface": str(authorization.get("family_surface", "")),
        "candidate_key": f"VdubusNQ_v4|{action.symbol}|ENTRY|{action.side}|1",
        "sequence": 1,
        "original_qty": proposal.qty,
        "approved_qty": approved_qty,
        "order_match": {
            "strategy_id": "VdubusNQ_v4",
            "symbol": action.symbol,
            "role": "ENTRY",
            "side": action.side,
            "sequence": 1,
        },
    }
    if submissions:
        out.record_actions("VdubusNQ_v4", submissions, decision=family_decision)
    else:
        out.record_family_rejection("VdubusNQ_v4", action, family_decision)

    for event in fixture.get("broker_event_script", []) or []:
        match = event.get("order_match", {}) or {}
        if str(match.get("strategy_id")) != "VdubusNQ_v4":
            continue
        order = out._match_order(match)
        if order is None:
            continue
        key = _broker_event_key(event)
        out.note_broker_event(order, event)
        out._applied.add(key)
        working = core_state.working_entries[str(order["client_tag"])]
        core_state, actions, _events = on_fill(
            core_state,
            VdubFill(
                oms_order_id=str(order["client_tag"]),
                fill_price=float(event.get("price", order.get("stop_price") or 0.0)),
                fill_qty=int(float(event.get("qty", order["qty"]))),
                fill_time=parse_time(event.get("timestamp")),
                point_value=point_value,
                commission=float(event.get("commission", 0.0)),
                entry_context=VdubEntryFillContext(working_entry=working),
            ),
        )
        out.record_actions("VdubusNQ_v4", actions)

    snapshot = snapshot_state(core_state)
    out.strategy_state["VdubusNQ_v4"] = {
        "strategy_id": "VdubusNQ_v4",
        "last_bar_ts": snapshot.get("last_bar_ts"),
        "last_decision_code": snapshot.get("last_decision_code", "IDLE"),
        "last_decision_details": snapshot.get("last_decision_details", {}),
        "bar_idx": int(snapshot.get("bar_idx", 0) or 0),
        "position_count": len(snapshot.get("positions", []) or []),
        "working_entry_count": len(snapshot.get("working_entries", {}) or {}),
    }


def _replay_iaric(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> None:
    from strategies.stock.iaric.config import StrategySettings
    from strategies.stock.iaric.core import logic as iaric_logic
    from strategies.stock.iaric.core.state import IARICFill
    from strategies.stock.iaric.entry_request import build_ready_entry_request
    from strategies.stock.iaric.models import PortfolioState

    settings = StrategySettings(
        base_risk_fraction=float(((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {}).get("base_risk_fraction", StrategySettings().base_risk_fraction))
    )
    artifact = iaric_artifact(fixture)
    state = iaric_state_snapshot(fixture, "IARIC_v1")
    symbols = [symbol_state.symbol for symbol_state in state.symbols]
    for symbol in symbols:
        bars = iaric_minute_bars(fixture, symbol)
        if not bars:
            continue
        bar_5m = _aggregate_5m(bars[-5:])
        symbol_state = next(item for item in state.symbols if item.symbol == symbol)
        item = artifact.by_symbol[symbol]
        quote = iaric_quote(fixture, symbol)
        market = type("ReplayMarket", (), {})()
        market.bars_5m = [bar_5m]
        market.last_5m_bar = bar_5m
        market.last_30m_bar = bar_5m
        market.session_vwap = bar_5m.close
        market.session_low = min(symbol_state.session_low or bar_5m.low, bar_5m.low)
        market.session_high = max(symbol_state.session_high or bar_5m.high, bar_5m.high)
        market.last_price = bar_5m.close
        market.ask = quote.ask
        market.bid = quote.bid
        market.spread_pct = quote.spread_pct
        step = iaric_logic.evaluate_ready_entry(
            settings,
            symbol_state,
            item,
            bar_5m,
            market,
            max(int(symbol_state.bars_seen_today), 0),
            max(float(symbol_state.daily_atr), 0.01),
            bars=[bar_5m],
        )
        if step is None or step.acceptance is None:
            continue
        iaric_logic.apply_entry_acceptance(symbol_state, step.acceptance)
        portfolio = PortfolioState(
            account_equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
            base_risk_fraction=settings.base_risk_fraction,
        )
        request_build = build_ready_entry_request(
            symbol=symbol,
            state=symbol_state,
            item=item,
            market=market,
            portfolio=portfolio,
            symbol_to_sector={symbol: item.sector},
            settings=settings,
            now=bar_5m.end_time,
            route=str(symbol_state.route_family or "OPENING_RECLAIM"),
        )
        if request_build.entry_request is None:
            continue
        symbol_state.risk_per_share = max(request_build.entry_price - float(symbol_state.stop_level), 0.01)
        entry_request = request_build.entry_request
        state, actions, _events = iaric_logic.on_bar(state, bar_ts=bar_5m.end_time, entry_request=entry_request)
        out.record_actions("IARIC_v1", actions)
    for event in fixture.get("broker_event_script", []):
        if str((event.get("order_match", {}) or {}).get("strategy_id")) != "IARIC_v1":
            continue
        key = _broker_event_key(event)
        if key in out._applied:
            continue
        order = out._match_order(event.get("order_match", {}))
        if order is None:
            continue
        out.note_broker_event(order, event)
        out._applied.add(key)
        state, actions, _events = iaric_logic.on_fill(
            state,
            IARICFill(
                oms_order_id=str(order["client_tag"]),
                fill_price=float(event.get("price", order.get("limit_price") or 0.0)),
                fill_qty=int(float(event.get("qty", order["qty"]))),
                fill_time=parse_time(event.get("timestamp")),
                commission=float(event.get("commission", 0.0)),
                symbol=str(order["symbol"]),
                order_role="ENTRY",
            ),
        )
        out.record_actions("IARIC_v1", actions)
    out.strategy_state["IARIC_v1"] = _compact_iaric_state(state)


def _aggregate_5m(bars: list[Any]) -> Any:
    from strategies.stock.iaric.models import Bar

    return Bar(
        symbol=bars[0].symbol,
        start_time=bars[0].start_time,
        end_time=bars[-1].end_time,
        open=bars[0].open,
        high=max(bar.high for bar in bars),
        low=min(bar.low for bar in bars),
        close=bars[-1].close,
        volume=sum(bar.volume for bar in bars),
    )


def _compact_iaric_state(state: Any) -> dict[str, Any]:
    symbols = {}
    for symbol_state in sorted(state.symbols, key=lambda item: item.symbol):
        position = getattr(symbol_state, "position", None)
        symbols[symbol_state.symbol] = {
            "stage": getattr(symbol_state, "stage", ""),
            "route_family": getattr(symbol_state, "route_family", ""),
            "in_position": bool(getattr(symbol_state, "in_position", False)),
            "risk_per_share": getattr(symbol_state, "risk_per_share", 0.0),
            "position": (
                {
                    "qty_open": getattr(position, "qty_open", 0),
                    "entry_price": getattr(position, "entry_price", 0.0),
                    "current_stop": getattr(position, "current_stop", 0.0),
                }
                if position is not None
                else None
            ),
        }
    return {"symbols": symbols, "last_decision_code": getattr(state, "last_decision_code", "")}


replay_tpc = _replay_tpc
replay_nq_regime = _replay_nq_regime
replay_nqdtc_r1b = _replay_nqdtc_r1b
replay_vdub_r1b = _replay_vdub_r1b
replay_iaric = _replay_iaric
