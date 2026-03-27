# Structured Extraction: iaric_intraday_engine_v2.py + research_replay.py

---

## 1. IARICIntradayEngineV2 (1088 lines)

### Class Name & Constructor

```python
class IARICIntradayEngineV2:
    """Tier 2 v2 IARIC backtest engine using 5m bars.

    For each trading day:
    1. Process carry positions from previous day
    2. Build watching list from T1's selection (iaric_selection_for_date)
    3. Try entry triggers: VWAP pullback -> ORB -> fallback at bar 6
    4. Manage positions: 4-phase adaptive trailing stop
    5. EOD: carry score -> carry / partial carry / flatten
    """

    def __init__(
        self,
        config: IARICBacktestConfig,
        replay: ResearchReplayEngine,
        settings: StrategySettings | None = None,
        shadow_tracker=None,
    ):
        self._config = config
        self._replay = replay
        self._settings = settings or StrategySettings()

        if config.param_overrides:
            self._settings = replace(self._settings, **config.param_overrides)

        self._slippage = config.slippage
        self._sector_map = {sym: sector for sym, sector, _ in replay._universe}
```

### Result Dataclass

```python
@dataclass
class IARICIntradayV2Result:
    """Result from T2 v2 engine."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, WatchlistArtifact]
```

### _T2Position Dataclass (all fields)

```python
@dataclass
class _T2Position:
    """Backtest-internal T2 v2 position."""

    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    risk_per_share: float
    stop_price: float
    stop_phase: int  # 1-4
    mfe_price: float  # max favorable price
    mfe_r: float  # max favorable R-multiple
    entry_trigger: str  # VWAP_PULLBACK, ORB, FALLBACK, PM_REENTRY
    sector: str
    regime_tier: str
    conviction_bucket: str
    conviction_multiplier: float
    sponsorship_state: str
    highest_close: float = 0.0  # highest 5m close (for trailing stops)
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    carry_days: int = 0
    partial_taken: bool = False
    entry_bar_idx: int = 0
    original_quantity: int = 0

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.original_quantity

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (price - self.entry_price) / self.risk_per_share
```

### Constants

```python
_BAR_1330 = 48   # 13:30 ET = bar 48
_BAR_1545 = 75   # 15:45 ET = bar 75
_BARS_PER_DAY = 78
_MINUTES_PER_BAR = 5
```

---

## 2. Overnight Carry Processing (exact code, L160-219)

```python
            # ----- Process overnight carry positions -----
            closed_carry: list[str] = []
            for sym, pos in list(carry_positions.items()):
                ohlc = self._replay.get_daily_ohlc(sym, trade_date)
                if ohlc is None:
                    # No data -- close at last known price to avoid silent PnL loss
                    fallback = self._replay.get_daily_close(sym, trade_date)
                    if fallback is None:
                        fallback = pos.entry_price  # worst case: flat
                    trade = self._close_position(pos, fallback, ts, "DATA_GAP")
                    equity += trade.pnl_net
                    trades.append(trade)
                    closed_carry.append(sym)
                    continue

                O, H, L, C = ohlc
                pos.carry_days += 1
                pos.max_favorable = max(pos.max_favorable, H)
                pos.max_adverse = min(pos.max_adverse, L) if pos.max_adverse > 0 else L
                pos.mfe_price = max(pos.mfe_price, H)
                pos.highest_close = max(pos.highest_close, C)
                pos.mfe_r = max(pos.mfe_r, pos.unrealized_r(H))

                exit_price = None
                exit_reason = ""

                # Flow reversal check
                if cfg.ablation.use_flow_reversal_exit:
                    lookback = settings.flow_reversal_lookback
                    last_n = self._replay.get_flow_proxy_last_n(sym, trade_date, lookback)
                    if last_n is not None and all(v < 0 for v in last_n):
                        exit_price = O
                        exit_reason = "FLOW_REVERSAL"

                # Still carry-eligible?
                if exit_price is None:
                    cur_r = pos.unrealized_r(C)
                    tier_ok = pos.regime_tier == "A" or (
                        pos.regime_tier == "B" and settings.regime_b_carry_mult > 0
                    )
                    still_eligible = (
                        cfg.ablation.use_carry_logic
                        and tier_ok
                        and pos.sponsorship_state == "STRONG"
                        and cur_r > settings.min_carry_r
                        and pos.carry_days < settings.max_carry_days
                    )
                    if still_eligible:
                        continue  # Keep carrying
                    exit_price = C
                    exit_reason = "CARRY_EXIT"

                # Close carry position
                trade = self._close_position(pos, exit_price, ts, exit_reason)
                equity += trade.pnl_net
                trades.append(trade)
                closed_carry.append(sym)

            for sym in closed_carry:
                carry_positions.pop(sym, None)
```

---

## 3. Carry Score Computation (exact code, L872-940)

```python
    def _compute_carry_score(
        self,
        pos: _T2Position,
        bar: Bar,
        vwap: float,
        item: WatchlistItem,
        artifact: WatchlistArtifact,
        settings: StrategySettings,
        bars: list[Bar],
        bar_idx: int,
    ) -> float:
        """Compute carry score (0-100) from 6 intraday evidence factors."""
        score = 0.0

        # Factor 1: Current R-multiple (weight 25%)
        cur_r = pos.unrealized_r(bar.close)
        r_score = min(max(cur_r / 2.0, 0.0), 1.0) * 100
        score += r_score * 0.25

        # Factor 2: Price vs session VWAP (weight 20%)
        if vwap > 0:
            dist = (bar.close - vwap) / (item.daily_atr_estimate if item.daily_atr_estimate > 0 else bar.close * 0.01)
            vwap_score = min(max((dist + 0.5) / 1.0, 0.0), 1.0) * 100
        else:
            vwap_score = 50.0
        score += vwap_score * 0.20

        # Factor 3: Close in daily range (weight 15%)
        if bars:
            day_high = max(b.high for b in bars[:bar_idx + 1])
            day_low = min(b.low for b in bars[:bar_idx + 1])
            rng = max(day_high - day_low, 1e-9)
            range_pct = (bar.close - day_low) / rng
            range_score = range_pct * 100
        else:
            range_score = 50.0
        score += range_score * 0.15

        # Factor 4: Micropressure proxy from last 6 bars (weight 15%)
        if bar_idx >= 6:
            bullish_bars = sum(1 for b in bars[bar_idx - 5:bar_idx + 1]
                               if b.close > b.open)
            micro_score = (bullish_bars / 6.0) * 100
        else:
            micro_score = 50.0
        score += micro_score * 0.15

        # Factor 5: Volume trend PM vs AM (weight 10%)
        if bar_idx >= 40:
            am_bars = bars[6:39]  # ~9:30-12:30
            pm_bars = bars[39:bar_idx + 1]  # 12:30+
            am_vol = sum(b.volume for b in am_bars) / max(len(am_bars), 1)
            pm_vol = sum(b.volume for b in pm_bars) / max(len(pm_bars), 1)
            if am_vol > 0:
                vol_ratio = pm_vol / am_vol
                vol_score = min(max(vol_ratio, 0.0), 2.0) / 2.0 * 100
            else:
                vol_score = 50.0
        else:
            vol_score = 50.0
        score += vol_score * 0.10

        # Factor 6: Sponsorship state (weight 15%)
        spon_map = {"STRONG": 100, "ACCUMULATE": 75, "NEUTRAL": 50,
                     "DISTRIBUTE": 25, "WEAK": 0, "BREAKDOWN": 0}
        spon_score = spon_map.get(item.sponsorship_state, 50)
        score += spon_score * 0.15

        return score
```

---

## 4. EOD Carry Decision Logic (exact code, L942-980)

```python
    def _apply_eod_decision(
        self,
        pos: _T2Position,
        score: float,
        artifact: WatchlistArtifact,
        settings: StrategySettings,
        cfg: IARICBacktestConfig,
        current_price: float = 0.0,
    ) -> str:
        """Decide CARRY / PARTIAL_CARRY / FLATTEN based on carry score + hard gates."""
        if not cfg.ablation.use_carry_logic:
            return "FLATTEN"

        # Hard carry gates -- position must be currently in profit
        in_profit = pos.unrealized_r(current_price) > 0 if current_price > 0 else pos.mfe_r > 0

        tier_ok = (
            artifact.regime.tier == "A"
            or (artifact.regime.tier == "B" and settings.regime_b_carry_mult > 0)
        )
        gates_pass = (
            in_profit
            and tier_ok
            and pos.sponsorship_state == "STRONG"
            and pos.carry_days < settings.max_carry_days
        )

        if gates_pass:
            if score >= settings.t2_carry_threshold:
                return "CARRY"
            if score >= settings.t2_carry_partial_threshold:
                return "PARTIAL_CARRY"

        # Fallback: default carry for profitable positions
        if settings.t2_default_carry_profitable:
            cur_r = pos.unrealized_r(current_price) if current_price > 0 else 0
            if cur_r >= settings.t2_default_carry_min_r:
                return "DEFAULT_CARRY"
        return "FLATTEN"
```

### EOD Decision Handling in Main Loop

```python
# At bar 75 (15:45 ET):
if bar_idx == _BAR_1545:
    score = self._compute_carry_score(pos, bar, vwap, item, artifact, settings, bars, bar_idx)
    decision = self._apply_eod_decision(pos, score, artifact, settings, cfg, current_price=bar.close)

    if decision == "CARRY":
        pos.carry_days = 1
        carry_positions[sym] = pos
        del positions[sym]
    elif decision == "PARTIAL_CARRY":
        # Exit 50%, carry rest
        partial_qty = max(1, pos.quantity // 2)
        if pos.quantity - partial_qty >= 1:
            partial_trade = self._partial_exit(pos, bar.close, partial_qty, ts, reason="PARTIAL_CARRY_FLATTEN")
            equity += partial_trade.pnl_net
            trades.append(partial_trade)
        pos.carry_days = 1
        carry_positions[sym] = pos
        del positions[sym]
    elif decision == "DEFAULT_CARRY":
        # Carry with protective stop
        protective = bar.close - settings.t2_default_carry_stop_atr * stop_atr
        pos.stop_price = max(pos.stop_price, protective)
        pos.carry_days = 1
        carry_positions[sym] = pos
        del positions[sym]
    else:
        # FLATTEN
        trade = self._close_position(pos, bar.close, ts, "EOD_FLATTEN")
        equity += trade.pnl_net
        trades.append(trade)
```

---

## 5. Flow Reversal Exits (exact code, L186-192)

Flow reversal is checked during overnight carry processing (NOT intraday). Logic:
- Look back `settings.flow_reversal_lookback` days of flow proxy values
- If ALL values are negative, exit at the day's open price

```python
                # Flow reversal check
                if cfg.ablation.use_flow_reversal_exit:
                    lookback = settings.flow_reversal_lookback
                    last_n = self._replay.get_flow_proxy_last_n(sym, trade_date, lookback)
                    if last_n is not None and all(v < 0 for v in last_n):
                        exit_price = O
                        exit_reason = "FLOW_REVERSAL"
```

---

## 6. Entry Trigger Mechanisms

### A) ORB Breakout (bar 6 only)

```python
    def _check_orb_breakout(
        self, bar: Bar, or_high: float, settings: StrategySettings,
    ) -> float | None:
        """Check for opening range breakout. Returns entry price or None."""
        if or_high <= 0:
            return None
        if bar.high <= or_high:       # Price must exceed OR high
            return None
        if bar.cpr < settings.t2_orb_cpr_min:  # CPR check (strong close)
            return None
        if bar.close <= bar.open:     # Bullish bar required
            return None
        return bar.close
```

### B) VWAP Pullback (bars 6-12)

```python
    def _check_vwap_pullback(
        self, bar: Bar, vwap: float, session_atr: float,
        expected_vol: float, settings: StrategySettings,
    ) -> float | None:
        """Check for VWAP pullback entry. Returns entry price or None."""
        if session_atr <= 0 or vwap <= 0:
            return None
        pullback_dist = settings.t2_vwap_pullback_atr_mult * session_atr
        if bar.low > vwap - pullback_dist:  # Didn't pull back enough
            return None
        if bar.close < vwap:           # Must reclaim VWAP
            return None
        if bar.close <= bar.open:      # Bullish bar required
            return None
        if expected_vol > 0 and bar.volume < expected_vol * settings.t2_vwap_reclaim_vol_pct:
            return None                # Volume check
        return bar.close
```

### C) PM Re-entry (bar >= 48, for stopped-out symbols)

```python
    def _check_afternoon_strength(
        self, bar: Bar, vwap: float, morning_high: float,
        expected_vol: float, bars: list[Bar], bar_idx: int,
        settings: StrategySettings,
    ) -> float | None:
        """Check for afternoon strength re-entry. Returns entry price or None."""
        if vwap <= 0 or morning_high <= 0:
            return None
        if bar.close <= vwap or bar.close <= morning_high:  # Above VWAP AND morning high
            return None
        if bar_idx >= 6 and expected_vol > 0:  # Volume acceleration check
            recent_vol = sum(b.volume for b in bars[bar_idx - 5:bar_idx + 1])
            expected_30m = expected_vol * 6
            if expected_30m > 0 and recent_vol < expected_30m * settings.t2_afternoon_vol_mult:
                return None
        if bar.close <= bar.open:  # Bullish bar
            return None
        return bar.close
```

### D) FALLBACK (guaranteed open entry at bar 6)

If no VWAP pullback and no ORB triggers by bar 6, the engine falls back to an open-price entry:
- Entry price = first bar's open
- Stop = entry - t2_initial_atr_mult * ATR
- Trigger = "FALLBACK"

---

## 7. Adaptive Stop Phases (exact code, L829-866)

```python
    def _update_stop(
        self, pos: _T2Position, bar: Bar, session_atr: float,
        settings: StrategySettings,
    ) -> tuple[float, int]:
        """Update adaptive trailing stop. Returns (new_stop, new_phase).

        Phases ratchet UP only (never regress):
          Phase 1: Initial    -- entry - t2_initial_atr_mult x ATR
          Phase 2: Breakeven  -- entry_price (when MFE >= +0.5R)
          Phase 3: Profit trail -- highest_close - 1.0 x ATR (when MFE >= +1.0R)
          Phase 4: Tight trail  -- highest_close - 0.5 x ATR (when MFE >= +2.0R)
        """
        current_stop = pos.stop_price
        current_phase = pos.stop_phase

        # Trail on 5m closing prices (not intra-bar highs/lows)
        highest_close = pos.highest_close

        # Phase 4: Tight trail
        if pos.mfe_r >= settings.t2_tight_trail_r and current_phase <= 4:
            new_stop = highest_close - settings.t2_tight_trail_atr * session_atr
            return max(current_stop, new_stop), 4

        # Phase 3: Profit trail
        if pos.mfe_r >= 1.0 and current_phase <= 3:
            new_stop = highest_close - settings.t2_profit_trail_atr * session_atr
            return max(current_stop, new_stop), max(current_phase, 3)

        # Phase 2: Breakeven
        trigger_r = pos.unrealized_r(pos.highest_close) if settings.t2_breakeven_use_closes else pos.mfe_r
        if trigger_r >= settings.t2_breakeven_r and current_phase <= 2:
            return max(current_stop, pos.entry_price), max(current_phase, 2)

        # Phase 1: Initial (never changes)
        return current_stop, current_phase
```

---

## 8. Position Sizing (_open_position, L747-823)

```python
    def _open_position(
        self, symbol: str, entry_price: float, stop_price: float, trigger: str,
        item: WatchlistItem, artifact: WatchlistArtifact, equity: float,
        dow: int, bar_idx: int, ts: datetime, session_atr: float,
        settings: StrategySettings, cfg: IARICBacktestConfig,
        size_mult: float = 1.0,
    ) -> _T2Position | None:
        # Entry slippage
        slip_bps = self._slippage.slip_bps_normal
        slip = entry_price * slip_bps / 10_000
        fill_price = round(entry_price + slip, 2)

        risk_per_share = abs(fill_price - stop_price)
        if risk_per_share <= 0:
            return None

        # Sizing: conviction x regime x base_risk x dow_mult
        risk_mult = item.conviction_multiplier * artifact.regime.risk_multiplier
        if not cfg.ablation.use_conviction_scaling:
            risk_mult = artifact.regime.risk_multiplier

        # Day-of-week multiplier
        dow_mult = 1.0
        if dow == 1:   # Tuesday
            dow_mult = settings.t2_tuesday_mult
        elif dow == 4: # Friday
            dow_mult = settings.t2_friday_mult

        # Regime B sizing reduction
        if artifact.regime.tier == "B":
            risk_mult *= settings.t2_regime_b_sizing_mult

        # External size multiplier (entry strength, open entry, etc.)
        risk_mult *= size_mult

        risk_dollars = equity * settings.base_risk_fraction * risk_mult * dow_mult
        qty = int(floor(risk_dollars / risk_per_share))
        if qty < 1:
            return None

        # ... returns _T2Position with all fields populated
```

---

## 9. Replay Object Usage (method calls from engine)

The engine calls these methods on `self._replay`:

| Method | Purpose |
|--------|---------|
| `iaric_selection_for_date(trade_date, settings)` | Get WatchlistArtifact (cached) |
| `get_daily_ohlc(sym, trade_date)` | (O, H, L, C) tuple for carry processing |
| `get_daily_close(sym, trade_date)` | Fallback close price |
| `get_flow_proxy_last_n(sym, trade_date, n)` | Last n flow proxy values |
| `get_5m_bar_objects_for_date(sym, trade_date)` | List[Bar] for intraday sim |

---

## 10. run() Method Structure

```
def run(self) -> IARICIntradayV2Result:
    cfg = self._config
    settings = self._settings
    trading_dates = self._replay.tradable_dates(cfg.start_date, cfg.end_date)

    equity = cfg.initial_equity
    carry_positions: dict[str, _T2Position] = {}
    trades: list[TradeRecord] = []
    daily_selections: dict[date, WatchlistArtifact] = {}
    equity_history, ts_history = [], []

    for trade_date in trading_dates:
        ts = datetime(...)

        # 1. Process overnight carry positions
        #    - get_daily_ohlc -> update MFE/MAE
        #    - Flow reversal check -> exit at Open
        #    - Still eligible? -> continue carrying or CARRY_EXIT at Close

        # 2. Run selection
        #    artifact = replay.iaric_selection_for_date(...)
        #    Regime C gate -> skip day
        #    Max positions by regime tier (A vs B)

        # 3. Build intraday state
        #    Sort tradable by conviction, filter by min_conviction
        #    For each tradable symbol: get 5m bars, compute opening range

        # 4. Opening Range (bars 0-5)
        #    Track or_highs, or_lows per symbol

        # 5. Bar-by-bar loop (bars 6-77):
        #    For each bar:
        #      a) Update VWAP, session stats
        #      b) For each position:
        #         - Check stop hit
        #         - Update MFE/MAE/highest_close
        #         - Update adaptive trailing stop
        #         - At bar 75: EOD carry/flatten decision
        #      c) For each watching symbol (not yet entered):
        #         - A) ORB breakout (bar 6 only)
        #         - B) VWAP pullback (bars 6-12)
        #         - C) PM re-entry (bar >= 48, stopped-out only)
        #         - D) FALLBACK open entry (bar 6, if nothing else triggered)
        #         - Entry strength gating + sizing

        # 6. Force-close any remaining intraday positions
        #    (positions not carried get closed at last bar's close)

        equity_history.append(equity)

    # Close remaining carry positions at end of backtest
    return IARICIntradayV2Result(trades, equity_curve, timestamps, daily_selections)
```

---

## 11. ResearchReplayEngine (1276 lines)

### Class Name & Constructor

```python
class ResearchReplayEngine:
    def __init__(
        self,
        data_dir: str | Path = "research/backtests/stock/data/raw",
        universe_config: UniverseConfig | None = None,
    ):
        self._data_dir = Path(data_dir)
        self._universe_config = universe_config or UniverseConfig()

        # (symbol, sector, exchange) tuples
        self._universe: list[tuple[str, str, str]] = list(SP500_CONSTITUENTS)
        self._sector_map: dict[str, str] = {sym: sector for sym, sector, _ in self._universe}
        self._exchange_map: dict[str, str] = {sym: exch for sym, _, exch in self._universe}

        # Cached DataFrames
        self._daily_cache: dict[str, pd.DataFrame] = {}
        self._intraday_30m_cache: dict[str, pd.DataFrame] = {}
        self._intraday_5m_cache: dict[str, pd.DataFrame] = {}
        self._ref_cache: dict[str, pd.DataFrame] = {}  # SPY, VIX, HYG, sector ETFs

        # Selection caches (keyed by trade_date)
        self._alcb_selection_cache: dict[date, alcb_models.CandidateArtifact] = {}
        self._iaric_selection_cache: dict[date, iaric_models.WatchlistArtifact] = {}

        # Pre-computed date indices (symbol -> (sorted_dates, last_ilocs))
        self._daily_didx: dict[str, tuple[list[date], list[int]]] = {}
        self._ref_didx: dict[str, tuple[list[date], list[int]]] = {}
        self._30m_didx: dict[str, tuple[list[date], list[int]]] = {}
        self._5m_didx: dict[str, tuple[list[date], list[int]]] = {}

        # Pre-computed numpy arrays per symbol
        self._daily_arrs: dict[str, dict[str, np.ndarray]] = {}
        self._daily_flow: dict[str, np.ndarray] = {}

        # Pre-computed breadth arrays
        self._above_sma20: dict[str, np.ndarray] = {}
        self._vol_above_avg20: dict[str, np.ndarray] = {}

        # Pre-built bar object lists (sliced per query, zero creation)
        self._30m_bars: dict[str, list[alcb_models.Bar]] = {}
        self._5m_bars: dict[str, list[iaric_models.Bar]] = {}

        self._trading_dates: list[date] = []
```

### ALL Public Method Signatures

```python
# Data loading
def load_all_data(self) -> None
def trading_dates(self) -> list[date]                           # @property
def clear_selection_cache(self) -> None

# Snapshot / selection pipeline
def build_alcb_snapshot(self, trade_date, min_price, min_adv_usd, ...) -> alcb_models.Snapshot
def build_iaric_snapshot(self, trade_date, min_price, min_adv_usd, ...) -> iaric_models.Snapshot
def run_alcb_selection(self, snapshot, settings) -> alcb_models.CandidateArtifact
def run_iaric_selection(self, snapshot, settings) -> iaric_models.WatchlistArtifact

# Cached selection (preferred entry point)
def alcb_selection_for_date(self, trade_date, settings=None) -> alcb_models.CandidateArtifact
def iaric_selection_for_date(self, trade_date, settings=None) -> iaric_models.WatchlistArtifact

# Date helpers
def get_warmup_end_date(self, warmup_days: int = 250) -> date | None
def tradable_dates(self, start: date, end: date) -> list[date]
def get_next_trading_date(self, trade_date: date) -> date | None
def get_prev_trading_date(self, trade_date: date) -> date | None

# Daily data access
def get_daily_close(self, symbol: str, trade_date: date) -> float | None
def get_daily_ohlc(self, symbol: str, trade_date: date) -> tuple[float, float, float, float] | None
def get_next_open(self, symbol: str, trade_date: date) -> float | None

# Flow proxy
def get_flow_proxy_last_n(self, symbol: str, trade_date: date, n: int = 2) -> list[float] | None

# Intraday bars (raw DataFrames)
def get_30m_bars_for_date(self, symbol: str, trade_date: date) -> pd.DataFrame | None
def get_5m_bars_for_date(self, symbol: str, trade_date: date) -> pd.DataFrame | None

# Intraday bars (pre-built Bar objects -- zero creation, slice from cache)
def get_30m_bar_objects_for_date(self, symbol: str, trade_date: date) -> list[alcb_models.Bar]
def get_5m_bar_objects_for_date(self, symbol: str, trade_date: date) -> list[iaric_models.Bar]

# Aggregated bars
def get_4h_bars_up_to(self, symbol: str, trade_date: date, max_bars: int = 100) -> list[alcb_models.Bar]
```

### Key Method Bodies

#### get_daily_ohlc

```python
    def get_daily_ohlc(
        self, symbol: str, trade_date: date,
    ) -> tuple[float, float, float, float] | None:
        """Get (open, high, low, close) for a symbol on a date."""
        didx = self._daily_didx.get(symbol)
        arrs = self._daily_arrs.get(symbol)
        if didx is None or arrs is None:
            return None
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return None
        i = bounds[1]
        return (float(arrs["open"][i]), float(arrs["high"][i]),
                float(arrs["low"][i]), float(arrs["close"][i]))
```

#### get_flow_proxy_last_n

```python
    def get_flow_proxy_last_n(
        self, symbol: str, trade_date: date, n: int = 2,
    ) -> list[float] | None:
        """Get last n flow proxy values up to trade_date via direct array lookup.
        O(1) per call -- avoids building a full 415-symbol snapshot just
        to check flow reversal on 1-3 carry positions."""
        flow = self._daily_flow.get(symbol)
        didx = self._daily_didx.get(symbol)
        if flow is None or didx is None:
            return None
        end = _iloc_upto(didx[0], didx[1], trade_date)
        if end < n - 1:
            return None
        return flow[end - n + 1:end + 1].tolist()
```

#### get_5m_bar_objects_for_date

```python
    def get_5m_bar_objects_for_date(
        self, symbol: str, trade_date: date,
    ) -> list[iaric_models.Bar]:
        """Get pre-built IARIC 5m Bar objects for a single trading day.
        Returns a list slice from the pre-built cache -- zero object creation."""
        didx = self._5m_didx.get(symbol)
        bars = self._5m_bars.get(symbol)
        if didx is None or bars is None:
            return []
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return []
        return bars[bounds[0]:bounds[1] + 1]
```

### Selection Cache Mechanism

```python
    def iaric_selection_for_date(
        self, trade_date: date,
        settings: iaric_config.StrategySettings | None = None,
    ) -> iaric_models.WatchlistArtifact:
        """Build snapshot + run IARIC selection in one call.
        Results are cached by date."""
        cached = self._iaric_selection_cache.get(trade_date)
        if cached is not None:
            return cached
        s = settings or iaric_config.StrategySettings()
        snapshot = self.build_iaric_snapshot(
            trade_date, min_price=s.min_price, min_adv_usd=s.min_adv_usd,
        )
        result = self.run_iaric_selection(snapshot, settings)
        self._iaric_selection_cache[trade_date] = result
        return result
```

Cache stores `WatchlistArtifact` per date. Valid when settings don't change between experiments. Call `clear_selection_cache()` when settings change.

### Artifact Structure

The `WatchlistArtifact` (returned by iaric_selection_for_date) contains:
- `regime` -- regime assessment (tier A/B/C, risk_multiplier)
- `tradable` -- list of `WatchlistItem` objects with fields like:
  - symbol, sector, conviction_bucket, conviction_multiplier
  - sponsorship_state, daily_atr_estimate, intraday_atr_seed
  - expected_5m_volume, avwap_ref, avwap_band_lower, avwap_band_upper
- `market_wide_institutional_selling` -- bool flag
