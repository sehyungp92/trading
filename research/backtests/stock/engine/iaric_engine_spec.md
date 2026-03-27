# IARIC Intraday Engine (T2) -- Complete Specification

Extracted from `research/backtests/stock/engine/iaric_engine.py` (998 lines).

---

## 1. Class Name and Constructor Signature

```python
class IARICIntradayEngine:
    def __init__(
        self,
        config: IARICBacktestConfig,
        replay: ResearchReplayEngine,
        settings: StrategySettings | None = None,
        shadow_tracker: IARICShadowTracker | None = None,
    ):
```

**Constructor body:**
- `self._config = config`
- `self._replay = replay`
- `self._settings = settings or StrategySettings()`
- If `config.param_overrides`: applies via `replace(self._settings, **config.param_overrides)`
- `self._slippage = config.slippage`
- `self._sector_map = {sym: sector for sym, sector, _ in replay._universe}`
- `self._shadow = shadow_tracker`

---

## 2. Result Dataclass

```python
@dataclass
class IARICIntradayResult:
    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, WatchlistArtifact]
    fsm_log: list[dict] = field(default_factory=list)
    rejection_log: list[dict] = field(default_factory=list)
```

---

## 3. Position Dataclass (Internal)

```python
@dataclass
class _IARICPosition:
    """Backtest-internal IARIC Tier 2 position."""

    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    risk_per_share: float
    stop: float
    sector: str
    regime_tier: str
    conviction_multiplier: float
    sponsorship_state: str
    qty_original: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    hold_bars: int = 0
    partial_taken: bool = False
    partial_qty_exited: int = 0
    realized_partial_pnl: float = 0.0
    realized_partial_commission: float = 0.0
    # Enriched diagnostic fields
    confidence: str = "YELLOW"
    location_grade: str = "B"
    acceptance_count: int = 0
    required_acceptance_count: int = 0
    micropressure_signal: str = "NEUTRAL"
    timing_window: str = ""
    timing_mult: float = 1.0
    setup_type: str = ""
    drop_from_hod_pct: float = 0.0
    risk_unit_final: float = 0.0
    conviction_adders: tuple = ()
    time_stop_deadline: datetime | None = None
    setup_tag: str = "UNCLASSIFIED"
```

**Methods on _IARICPosition:**

```python
def unrealized_r(self, price: float) -> float:
    if self.risk_per_share <= 0:
        return 0.0
    return (price - self.entry_price) / self.risk_per_share

def to_position_state(self) -> PositionState:
    return PositionState(
        entry_price=self.entry_price,
        entry_time=self.entry_time,
        qty_entry=self.qty_original or self.quantity,
        qty_open=self.quantity - self.partial_qty_exited,
        final_stop=self.stop,
        current_stop=self.stop,
        initial_risk_per_share=self.risk_per_share,
        max_favorable_price=self.max_favorable,
        max_adverse_price=self.max_adverse if self.max_adverse > 0 else self.entry_price,
        partial_taken=self.partial_taken,
        setup_tag=self.setup_tag,
        time_stop_deadline=self.time_stop_deadline,
    )

def build_metadata(self, item: WatchlistItem, regime_risk_mult: float) -> dict:
    rps = max(self.risk_per_share, 0.01)
    oq = self.qty_original or (self.quantity + self.partial_qty_exited)
    return {
        "conviction_bucket": item.conviction_bucket,
        "conviction_multiplier": self.conviction_multiplier,
        "confidence": self.confidence,
        "location_grade": self.location_grade,
        "acceptance_count": self.acceptance_count,
        "required_acceptance_count": self.required_acceptance_count,
        "conviction_adders": list(self.conviction_adders),
        "micropressure_signal": self.micropressure_signal,
        "sponsorship_state": self.sponsorship_state,
        "timing_window": self.timing_window,
        "timing_multiplier": self.timing_mult,
        "setup_type": self.setup_type,
        "drop_from_hod_pct": round(self.drop_from_hod_pct, 4),
        "risk_unit_final": round(self.risk_unit_final, 4),
        "partial_taken": self.partial_taken,
        "partial_qty_fraction": self.partial_qty_exited / oq if oq else 0,
        "regime_risk_multiplier": regime_risk_mult,
        "setup_tag": self.setup_tag,
        "mfe_r": round((self.max_favorable - self.entry_price) / rps, 4) if self.max_favorable > 0 else 0,
        "mae_r": round((self.entry_price - self.max_adverse) / rps, 4) if self.max_adverse > 0 and self.max_adverse < self.entry_price else 0,
    }
```

---

## 4. SymbolIntradayState Dataclass (from `strategies/stock/iaric/models.py`)

```python
@dataclass(slots=True)
class SymbolIntradayState:
    symbol: str
    tier: str = "COLD"
    fsm_state: str = "IDLE"
    in_position: bool = False
    position_qty: int = 0
    avg_price: float | None = None
    setup_type: str | None = None
    setup_low: float | None = None
    reclaim_level: float | None = None
    stop_level: float | None = None
    setup_time: datetime | None = None
    invalidated_at: datetime | None = None
    acceptance_count: int = 0
    required_acceptance_count: int = 0
    location_grade: str | None = None
    session_vwap: float | None = None
    avwap_live: float | None = None
    sponsorship_signal: str = "NEUTRAL"
    micropressure_signal: str = "NEUTRAL"
    micropressure_mode: str = "PROXY"
    flowproxy_signal: str = "UNAVAILABLE"
    confidence: str | None = None
    last_1m_bar_time: datetime | None = None
    last_5m_bar_time: datetime | None = None
    active_order_id: str | None = None
    time_stop_deadline: datetime | None = None
    setup_tag: str | None = None
    expected_volume_pct: float = 0.0
    average_30m_volume: float = 0.0
    last_transition_reason: str = ""
    entry_order: PendingOrderState | None = None
    position: PositionState | None = None
    exit_order: PendingOrderState | None = None
    pending_hard_exit: bool = False
```

---

## 5. MarketSnapshot Dataclass (from `strategies/stock/iaric/models.py`)

```python
@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    last_price: float | None = None
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0
    session_high: float | None = None
    session_low: float | None = None
    session_vwap: float | None = None
    avwap_live: float | None = None
    last_quote: QuoteSnapshot | None = None
    last_1m_bar: Bar | None = None
    last_5m_bar: Bar | None = None
    last_30m_bar: Bar | None = None
    minute_bars: deque[Bar] = field(default_factory=lambda: deque(maxlen=390))
    bars_5m: deque[Bar] = field(default_factory=lambda: deque(maxlen=120))
    bars_30m: deque[Bar] = field(default_factory=lambda: deque(maxlen=40))
    tick_pressure_window: deque[tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=512))
```

**Properties on MarketSnapshot:**
- `minutes_since_hod -> int`: iterates reversed minute_bars to find bar with high >= session_high
- `drop_from_hod -> float`: `max(0.0, (session_high - last_price) / session_high)`

**How the engine populates MarketSnapshot per bar:**
```python
if bar.high > (market.session_high or 0):
    market.session_high = bar.high
    sym_hod_idx = sym_bar_idx
market.last_price = bar.close
if market.session_low == float("inf"):
    market.session_low = bar.low
else:
    market.session_low = min(market.session_low, bar.low)
cum_vol += bar.volume
cum_pv += bar.close * bar.volume
if cum_vol > 0:
    market.session_vwap = cum_pv / cum_vol
market.bars_5m.append(bar)
market.last_5m_bar = bar
```

---

## 6. run() Method Structure

### Main Flow:
1. Parse `start_date`, `end_date` from config
2. Get `trading_dates = self._replay.tradable_dates(start, end)`
3. Initialize: `equity = cfg.initial_equity`, empty lists for trades/equity_history/ts_history/fsm_log/rejection_log
4. Pre-extract ablation flags to locals: `use_regime_gate`, `use_time_stop`, `use_avwap_breakdown`, `use_partial_take`, `use_carry`, `use_sector_limit`, `use_sponsorship_filter`
5. Pre-extract slippage: `slip_bps = self._slippage.slip_bps_normal / 10_000`, `comm_per_share`
6. Pre-extract settings: `panic_drop`, `panic_minutes`, `drift_drop`, `drift_minutes`, `stale_minutes`, `base_risk_frac`, `ts_minutes`, `max_per_sector`

### Per Day Loop (`for day_idx, trade_date in enumerate(trading_dates)`):
1. `artifact = self._replay.iaric_selection_for_date(trade_date, settings)`
2. Regime gate: if `use_regime_gate and artifact.regime.tier == "C"` -> skip day
3. `max_pos = max_positions_for_regime(artifact.regime.tier, settings)`
4. `tradable_map = {item.symbol: item for item in artifact.items}`
5. Initialize per-day dicts: `positions`, `fsm_states`, `market_snapshots`, `sector_counts`, `hod_bar_index`, `bar_index`, `bar_30m_accum`, `vwap_cum_vol`, `vwap_cum_pv`
6. Initialize FSM for each tradable symbol (SymbolIntradayState with IDLE, flow signal wired)
7. Initialize MarketSnapshot per symbol (last_price=avwap_ref, session_high=0.0, session_low=inf, avwap_live=avwap_ref)

### Per Symbol Loop (`for sym in list(tradable_map.keys())`):
1. `bars = self._replay.get_5m_bar_objects_for_date(sym, trade_date)`
2. Filter to regular hours: `_MKT_OPEN <= bar.start_time.astimezone(ET).time() < _MKT_CLOSE`
3. Pre-extract per-symbol constants: `atr_5m_pct`, AVWAP band (lo/hi), `expected_vol`, `median_vol`

### Per Bar Loop (`for bar in bars`):
1. `now = bar.end_time`
2. Update market snapshot (session_high, last_price, session_low, VWAP)
3. Track 5m bars on market, aggregate 30m bars (every 6 5m bars)
4. Shadow tracker: `shadow.update_bar(sym, bar.high, bar.low, bar.close)`
5. **If sym in positions** -> manage position (exit cascade)
6. **Else** -> FSM state machine + entry attempt

### Post-bar Loop (EOD):
1. Flatten remaining positions using `self._replay.get_daily_close(sym, trade_date)`
2. Shadow tracker: `shadow.flush_stale()`
3. Append equity to history

---

## 7. FSM State Transitions

### States: `IDLE`, `SETUP_DETECTED`, `ACCEPTING`, `READY_TO_ENTER`, `IN_POSITION`, `INVALIDATED`

### Transition Map:

```
IDLE -> SETUP_DETECTED
    Condition: bar is within AVWAP band (avwap_band_lo..avwap_band_hi)
               AND (PANIC_FLUSH: drop_from_hod >= panic_drop AND minutes_since_hod <= panic_minutes)
                OR (DRIFT_EXHAUSTION: drop_from_hod >= drift_drop AND minutes_since_hod >= drift_minutes)
    Action: lock_setup(sym_state, bar, atr_5m_pct, reason=setup_type)
            compute_location_grade(item, market)

SETUP_DETECTED -> ACCEPTING
    Condition: reclaim_level touched (bar.high >= reclaim_level OR last_price >= reclaim_level)
    Action: compute_required_acceptance() -> sets required_acceptance_count

SETUP_DETECTED -> INVALIDATED
    Condition: bar.low <= stop_level (stop breach)
           OR elapsed since setup_time > stale_minutes (staleness)

ACCEPTING -> READY_TO_ENTER
    Condition: acceptance_count >= required_acceptance_count AND confidence != "RED"
    Per-bar actions: update_acceptance(sym_state, bar)
                     compute_micropressure_proxy(bar, ...)
                     resolve_confidence(sym_state)

ACCEPTING -> INVALIDATED
    Condition: bar.low <= stop_level (stop breach)
           OR elapsed since setup_time > stale_minutes (staleness)

READY_TO_ENTER -> IN_POSITION
    Condition: NOT entry_blocked AND passes sponsorship filter AND risk_unit > 0 AND qty >= 1
    Action: create _IARICPosition, increment sector_counts

INVALIDATED -> IDLE
    Condition: cooldown_expired(sym_state, now, settings)
    Action: reset_setup_state(sym_state)
```

### Global Invalidation (checked BEFORE state-specific logic):
Applied when `fsm_state in ("SETUP_DETECTED", "ACCEPTING")`:
1. **Stop breach**: `bar.low <= sym_state.stop_level` -> INVALIDATED
2. **Stale timeout**: `elapsed > stale_minutes` -> INVALIDATED

---

## 8. Entry Logic

```python
# Entry price = bar.close
entry_price = bar.close
stop_level = sym_state.stop_level
if not stop_level or stop_level <= 0:
    stop_level = entry_price * 0.98  # fallback 2% stop

risk_per_share = abs(entry_price - stop_level)
if risk_per_share <= 0:
    reset_setup_state(sym_state)
    continue

# 6-factor sizing
risk_unit = compute_final_risk_unit(item, sym_state, now, settings)
if risk_unit <= 0:
    reset_setup_state(sym_state)
    continue

# Position sizing
risk_dollars = equity * base_risk_frac * risk_unit
qty = int(floor(risk_dollars / risk_per_share))
if qty < 1:
    reset_setup_state(sym_state)
    continue

# Slippage (adverse, long-only: add to entry)
slip = entry_price * slip_bps
fill_price = round(entry_price + slip, 2)
commission = comm_per_share * qty

# Time stop deadline
ts_deadline = now + timedelta(minutes=ts_minutes)
```

### Entry gates (checked BEFORE entry, do NOT block FSM progression):
1. `len(positions) >= max_pos` -> "position_cap"
2. `not timing_gate_allows_entry(now, settings)` -> "timing_blocked"
3. `use_sector_limit and sector_counts.get(sec, 0) >= max_per_sector` -> "sector_limit"
4. `use_sponsorship_filter and item.sponsorship_state in ("WEAK", "BREAKDOWN")` -> "sponsorship_filter"

---

## 9. Exit Cascade (exact order, for IN_POSITION)

```python
# 1. Track MFE/MAE
pos.max_favorable = max(pos.max_favorable, bar.high)
pos.max_adverse = min(pos.max_adverse, bar.low) if pos.max_adverse > 0 else bar.low

exit_price = None
exit_reason = ""

# 2. STOP HIT (highest priority)
if bar.low <= pos.stop:
    exit_price = pos.stop
    exit_reason = "STOP_HIT"

# 3. TIME STOP
if exit_price is None and use_time_stop:
    pos_state = pos.to_position_state()
    if should_exit_for_time_stop(pos_state, now, bar.close):
        exit_price = bar.close
        exit_reason = "TIME_STOP"

# 4. AVWAP BREAKDOWN (30m bar required)
if (exit_price is None and use_avwap_breakdown
        and bar_30m is not None and market.avwap_live):
    if should_exit_for_avwap_breakdown(
        bar_30m, market.avwap_live,
        item.average_30m_volume, settings,
    ):
        exit_price = bar.close
        exit_reason = "AVWAP_BREAKDOWN"

# 5. PARTIAL TAKE (non-exclusive -- can coexist with runner)
if exit_price is None and not pos.partial_taken and use_partial_take:
    pos_state = pos.to_position_state()
    take, fraction = should_take_partial(
        pos_state, bar.close, settings,
        regime_multiplier=regime_risk_mult,
    )
    if take:
        partial_qty = max(1, int(pos.quantity * fraction))
        pnl = (bar.close - pos.entry_price) * partial_qty
        comm = comm_per_share * partial_qty
        equity += pnl - comm
        pos.quantity -= partial_qty
        pos.partial_taken = True
        pos.partial_qty_exited = partial_qty
        pos.realized_partial_pnl += pnl
        pos.realized_partial_commission += comm
        if pos.quantity <= 0:
            exit_price = bar.close
            exit_reason = "PARTIAL_FULL"

# 6. EOD FLATTEN (15:55 ET check, within bar loop)
if exit_price is None:
    bar_et = now.astimezone(ET) if now.tzinfo else now
    if bar_et.time() >= time(15, 55):
        pos_state = pos.to_position_state()
        pos.setup_tag = classify_trade(market, pos_state)

        if use_carry:
            elig, reason = carry_eligible(
                item, market, pos_state,
                flow_reversal_flag=False,
            )
            if not elig:
                exit_price = bar.close
                exit_reason = "EOD_FLATTEN"
        else:
            exit_price = bar.close
            exit_reason = "EOD_FLATTEN"
```

### Exit fill processing:
```python
if exit_price is not None and pos.quantity > 0:
    slip = exit_price * slip_bps
    fill = round(exit_price - slip, 2)  # adverse slippage on exit (long-only)
    commission = comm_per_share * pos.quantity
    runner_pnl = (fill - pos.entry_price) * pos.quantity
    equity += runner_pnl - commission

    total_pnl = runner_pnl + pos.realized_partial_pnl
    total_commission = pos.commission_entry + commission + pos.realized_partial_commission
    total_qty = pos.quantity + pos.partial_qty_exited
    orig_risk = pos.risk_per_share * (pos.qty_original or total_qty)
    r_mult = total_pnl / orig_risk if orig_risk > 0 else 0.0
```

---

## 10. EOD Handling (Post bar-loop)

After the per-bar loop for all symbols completes, any remaining positions are force-flattened:

```python
for sym, pos in list(positions.items()):
    close_price = self._replay.get_daily_close(sym, trade_date)
    if close_price is None:
        continue
    item = tradable_map.get(sym)
    ts_close = datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=timezone.utc)
    slip = close_price * slip_bps
    fill = round(close_price - slip, 2)
    commission = comm_per_share * pos.quantity
    runner_pnl = (fill - pos.entry_price) * pos.quantity
    equity += runner_pnl - commission

    total_pnl = runner_pnl + pos.realized_partial_pnl
    total_commission = pos.commission_entry + commission + pos.realized_partial_commission
    total_qty = pos.quantity + pos.partial_qty_exited
    orig_risk = pos.risk_per_share * (pos.qty_original or total_qty)
    r_mult = total_pnl / orig_risk if orig_risk > 0 else 0.0

    meta = pos.build_metadata(item, regime_risk_mult) if item else {"conviction": "N/A"}

    trades.append(TradeRecord(
        strategy="IARIC",
        symbol=sym,
        direction=BTDirection.LONG,
        entry_time=pos.entry_time,
        exit_time=ts_close,
        entry_price=pos.entry_price,
        exit_price=fill,
        quantity=total_qty,
        pnl=total_pnl,
        r_multiple=r_mult,
        risk_per_share=pos.risk_per_share,
        commission=total_commission,
        slippage=pos.slippage_entry + slip * pos.quantity,
        entry_type="FSM_ENTRY",
        exit_reason="EOD_FLATTEN",
        sector=pos.sector,
        regime_tier=pos.regime_tier,
        hold_bars=pos.hold_bars,
        max_favorable=pos.max_favorable,
        max_adverse=pos.max_adverse,
        metadata=meta,
    ))
```

Note: Positions that pass the carry_eligible check at 15:55 ET within the bar loop are NOT exited there, but they ARE force-flattened in this post-loop EOD section. This means **carry does NOT actually hold overnight** -- it just avoids the early 15:55 exit and gets closed at 16:00 instead.

---

## 11. All Imports from signals.py and exits.py

### From `strategies.stock.iaric.exits`:
```python
carry_eligible          # (item, market, pos_state, flow_reversal_flag) -> (bool, str)
classify_trade          # (market, pos_state) -> str (setup_tag)
should_exit_for_avwap_breakdown  # (bar_30m, avwap_live, avg_30m_volume, settings) -> bool
should_exit_for_time_stop        # (pos_state, now, market_price) -> bool
should_take_partial              # (pos_state, price, settings, regime_multiplier) -> (bool, float)
```

### From `strategies.stock.iaric.signals`:
```python
compute_location_grade          # (item, market) -> str ("A"/"B"/"C")
compute_micropressure_proxy     # (bar, expected_vol, median_vol, reclaim_level) -> str
compute_required_acceptance     # (item, sym, now, settings, mwis) -> (int, list[str])
cooldown_expired                # (sym_state, now, settings) -> bool
lock_setup                      # (sym_state, bar, atr_5m_pct, reason) -> None
reset_setup_state               # (sym_state) -> None (resets to IDLE)
resolve_confidence              # (sym_state) -> str ("RED"/"YELLOW"/"GREEN")
update_acceptance               # (sym_state, bar) -> None
```

### From `strategies.stock.iaric.risk`:
```python
compute_final_risk_unit         # (item, sym_state, now, settings) -> float
max_positions_for_regime        # (tier, settings) -> int
timing_gate_allows_entry        # (now, settings) -> bool
timing_multiplier               # (now, settings) -> float
```

### Other imports:
```python
from research.backtests.stock.config import SlippageConfig
from research.backtests.stock.config_iaric import IARICBacktestConfig
from research.backtests.stock.engine.research_replay import ResearchReplayEngine
from research.backtests.stock.models import Direction as BTDirection, TradeRecord
from strategies.stock.iaric.config import ET, StrategySettings
from strategies.stock.iaric.models import Bar, MarketSnapshot, PositionState, SymbolIntradayState, WatchlistArtifact, WatchlistItem
```

---

## 12. How Replay/Artifact Data Is Accessed

```python
# Get tradable dates in range
self._replay.tradable_dates(start, end) -> list[date]

# Get daily selection artifact (calls research pipeline)
self._replay.iaric_selection_for_date(trade_date, settings) -> WatchlistArtifact

# Get pre-built 5m Bar objects for a symbol on a date
self._replay.get_5m_bar_objects_for_date(sym, trade_date) -> list[Bar]

# Get daily close price
self._replay.get_daily_close(sym, trade_date) -> float | None

# Get flow proxy data (last N daily values)
self._replay.get_flow_proxy_last_n(sym, trade_date, lookback) -> list[float] | None
```

---

## 13. Shadow Tracking

### Shadow tracker type: `IARICShadowTracker` (TYPE_CHECKING import only)

### Usage in engine:
```python
# Per bar, for all symbols:
shadow.update_bar(sym, bar.high, bar.low, bar.close)

# Funnel stages (progressive):
shadow.record_funnel("evaluated")      # every bar for non-positioned symbols
shadow.record_funnel("idle_check")     # when in IDLE state
shadow.record_funnel("setup_detected") # on IDLE -> SETUP_DETECTED
shadow.record_funnel("accepting")      # on SETUP_DETECTED -> ACCEPTING
shadow.record_funnel("ready_to_enter") # on ACCEPTING -> READY_TO_ENTER
shadow.record_funnel("entered")        # on actual entry

# Rejection recording:
shadow.record_rejection(IARICShadowSetup(...))  # via _record_shadow_rejection helper

# End of day:
shadow.flush_stale()
```

### `_record_shadow_rejection` helper signature:
```python
def _record_shadow_rejection(
    self, symbol: str, trade_date: date, gate: str,
    entry_price: float, stop_price: float,
    setup_type: str = "", sector: str = "", regime_tier: str = "",
    sponsorship_state: str = "", confidence: str = "",
    location_grade: str = "", acceptance_count: int = 0,
    conviction_multiplier: float = 0.0, risk_per_share: float = 0.0,
) -> None:
```

Creates `IARICShadowSetup` with all fields and calls `self._shadow.record_rejection(...)`.

---

## 14. Day-Level Initialization

Per trading day, the following state is set up:

```python
positions: dict[str, _IARICPosition] = {}
fsm_states: dict[str, SymbolIntradayState] = {}
market_snapshots: dict[str, MarketSnapshot] = {}
sector_counts: dict[str, int] = {}
hod_bar_index: dict[str, int] = {}
bar_index: dict[str, int] = {}
bar_30m_accum: dict[str, list[Bar]] = {}
vwap_cum_vol: dict[str, float] = {}
vwap_cum_pv: dict[str, float] = {}
```

### Per-symbol initialization within day:

**Flow signal derivation:**
```python
flow_signal = "UNAVAILABLE"
flow_data = self._replay.get_flow_proxy_last_n(sym, trade_date, settings.flow_reversal_lookback)
if flow_data is not None:
    if all(v < 0 for v in flow_data):
        flow_signal = "WEAK"
    elif all(v > 0 for v in flow_data):
        flow_signal = "STRONG"
    else:
        flow_signal = "NEUTRAL"
```

**SymbolIntradayState init:**
```python
fsm_states[sym] = SymbolIntradayState(
    symbol=sym,
    fsm_state="IDLE",
    setup_time=None,
    setup_low=0.0,
    reclaim_level=0.0,
    stop_level=0.0,
    acceptance_count=0,
    required_acceptance_count=0,
    confidence="YELLOW",
    micropressure_signal="NEUTRAL",
    flowproxy_signal=flow_signal,
    micropressure_mode="LIVE",
    invalidated_at=None,
    tier="WARM",
    sponsorship_signal=_SPONSORSHIP_TO_SIGNAL.get(item.sponsorship_state, "NEUTRAL"),
)
```

**MarketSnapshot init:**
```python
market_snapshots[sym] = MarketSnapshot(
    symbol=sym,
    last_price=item.avwap_ref,
    session_high=0.0,
    session_low=float("inf"),
    session_vwap=0.0,
    avwap_live=item.avwap_ref,
)
```

**AVWAP band computation:**
```python
if settings.t2_avwap_band_mult != 1.0:
    band_pct = settings.avwap_band_pct * settings.t2_avwap_band_mult
    avwap_band_lo = item.avwap_ref * (1.0 - band_pct)
    avwap_band_hi = item.avwap_ref * (1.0 + band_pct)
else:
    avwap_band_lo = item.avwap_band_lower
    avwap_band_hi = item.avwap_band_upper
```

---

## 15. Carry-Related Code

Carry in this T2 engine is minimal because it force-flattens at EOD.

### Within bar loop (at 15:55 ET):
```python
if use_carry:
    elig, reason = carry_eligible(
        item, market, pos_state,
        flow_reversal_flag=False,   # always False in backtest
    )
    if not elig:
        exit_price = bar.close
        exit_reason = "EOD_FLATTEN"
    # else: position survives the 15:55 check
else:
    exit_price = bar.close
    exit_reason = "EOD_FLATTEN"
```

### Key insight:
Even if `carry_eligible` returns True at 15:55, the position still gets force-flattened in the post-bar-loop EOD section (lines 934-977) at 16:00. So **carry eligibility only delays the exit by ~5 minutes** (from 15:55 to 16:00 close). There is **no overnight carry** in this engine -- all positions are closed same day.

The `carry_eligible` function is called with `flow_reversal_flag=False` always (no flow reversal tracking across days in this engine).

---

## Module-Level Constants

```python
_MKT_OPEN = time(9, 30)
_MKT_CLOSE = time(16, 0)

_SPONSORSHIP_TO_SIGNAL = {
    "STRONG": "STRONG",
    "ACCUMULATE": "ACCUMULATE",
    "NEUTRAL": "NEUTRAL",
    "STALE": "STALE",
    "WEAK": "DISTRIBUTE",
    "BREAKDOWN": "DISTRIBUTE",
}
```

## Helper Function

```python
def _timing_window_label(now: datetime, settings: StrategySettings) -> str:
    et = now.astimezone(ET).time()
    for start, end, _ in settings.timing_sizing:
        if start <= et < end:
            return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    return "OUTSIDE"
```

---

## 30m Bar Aggregation

```python
accum_30m.append(bar)
bar_30m = None
if len(accum_30m) >= 6:
    bar_30m = Bar(
        symbol=sym,
        start_time=accum_30m[0].start_time,
        end_time=accum_30m[-1].end_time,
        open=accum_30m[0].open,
        high=max(b.high for b in accum_30m),
        low=min(b.low for b in accum_30m),
        close=accum_30m[-1].close,
        volume=sum(b.volume for b in accum_30m),
    )
    market.last_30m_bar = bar_30m
    accum_30m.clear()
```

---

## TradeRecord Construction Fields

Every trade (both mid-day exits and EOD flatten) uses these TradeRecord fields:
```
strategy="IARIC", symbol, direction=BTDirection.LONG, entry_time, exit_time,
entry_price, exit_price (fill after slippage), quantity (total_qty),
pnl (total_pnl including partial), r_multiple, risk_per_share,
commission (total), slippage (entry + exit), entry_type="FSM_ENTRY",
exit_reason, sector, regime_tier, hold_bars, max_favorable, max_adverse,
metadata (from build_metadata)
```
