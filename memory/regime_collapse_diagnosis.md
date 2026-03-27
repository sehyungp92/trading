# Regime HMM Collapse Diagnosis

## Symptom
4-state HMM (G/R/S/D) collapses to 2 effective states: Goldilocks (77.5%, 4334 weeks) and Deflation (22.5%, 1259 weeks). Near-binary posteriors (avg P(dominant)=1.000 for both). Only 1 transition across 5,593 weeks.

## Root Causes (6 compounding factors)

### 1. Extreme Sticky Prior (PRIMARY)
- **File**: `regime/hmm.py` lines 21-24, 86-101
- Default: `sticky_diag=50.0`, `sticky_offdiag=2.0` → 89.3% self-transition prior
- Greedy optimizer accepted `sticky_offdiag=1.0` (round 4) → **94.3% self-transition prior**
- Ratio went from 25:1 to **50:1** diagonal-to-offdiag
- `transmat_prior` is set but hmmlearn uses it as a Dirichlet concentration parameter. With diag=50, the MAP estimate is overwhelmingly sticky — the data cannot override the prior.
- The prior effectively encodes "never switch states" before seeing any data.

### 2. Expanding Window Drowns Regime Changes
- **File**: `regime/engine.py` lines 72-84
- `X_train = Xz.loc[:rd].values` — always expanding from start of data
- `use_expanding_window=True` exists in config but is **NEVER REFERENCED** in engine code — expanding is hardcoded
- After 20 years: ~5,000 daily observations dominate the HMM fit
- The 2003-2009 period (growth/crash) establishes G and D states
- 2010-2026 mostly bull market → Goldilocks absorbs everything
- R (Reflation) and S (Stagflation) have too few observations relative to the expanding window to form distinct clusters

### 3. Warm Start Locks In Degenerate Solution
- **File**: `regime/hmm.py` lines 104-113
- `use_warm_start=True` copies previous model's `means_`, `covars_`, `startprob_`, `transmat_`
- Sets `init_params=""` → hmmlearn skips random init, uses copied params
- Combined with expanding window: each refit starts from the previous degenerate solution
- The model has no mechanism to "discover" new states once R/S collapse
- OOS guard (`refit_ll_tolerance=0.5`) further resists any structural change

### 4. Scoring Function Has Zero Regime Diversity Terms
- **File**: `research/backtests/regime/auto/scoring.py`
- Components: Sharpe (25%), Calmar (25%), InvDD (20%), CAGR (15%), Sortino (15%)
- **45% is drawdown-related** (Calmar + InvDD) → directly rewards stability
- NO penalty for: low regime diversity, few transitions, degenerate posteriors
- NO reward for: regime balance, entropy, information content
- The optimizer is free to collapse all states without any scoring consequence
- In fact, collapsing to "always Goldilocks" maximizes Sharpe/Calmar by staying SPY-heavy during the 2009-2026 bull run

### 5. Greedy Optimizer Systematically Chose Anti-Diversity Mutations
- **File**: `research/backtests/regime/output/greedy_optimal.json`
- 12 accepted mutations, all reinforce collapse:
  - `L_max=1.5` → more leverage amplifies Goldilocks (SPY-heavy) returns
  - `delta_rho_exempt=0.0` → removes ventilator exemptions, less regime-responsive
  - `sigma_floor_annual=0.03` → lower vol floor, more aggressive sizing
  - **`sticky_offdiag=1.0`** → doubles the sticky ratio from 25:1 to 50:1
  - `crisis_logit_a=0.5` → halves crisis sensitivity, stays risk-on longer
  - `z_window=126` → shorter z-score window, noisier features (masked by HMM stickiness)
- The optimizer exploited the scoring function's stability bias at every round

### 6. Features Are Mostly Correlated in One Direction
- **File**: `regime/features.py`
- 6 features: growth, inflation, eq-bond correlation, yield curve slope, credit spread, momentum breadth
- All z-scored with `rolling_zscore(window=126, minp=60)` → centered at 0
- During 2010-2026 bull market: growth positive, credit tight, breadth high, slope positive
- These features are highly correlated → HMM cannot distinguish R/S from G/D
- Only 2 features (growth_idx, infl_idx) are used for cosine-based label alignment
- The other 4 features add noise but don't help separate the 4 quadrants

## Confidence Amplifies the Problem
- **File**: `regime/inference.py`
- `conf = conf_floor + (1 - conf_floor) * [entropy * stability_blend]`
- Near-binary posteriors → entropy ≈ 0 → `conf_entropy` ≈ 1.0
- No transitions → `stability` = 1.0 every week
- So confidence ≈ 1.0 always → `confidence_fallback` never pulls toward neutral
- This creates a positive feedback loop: binary posteriors → high confidence → no smoothing → reinforces the dominant regime

## Recommended Fixes (Priority Order)

### Fix 1: Add Regime Diversity Penalty to Scoring
Add terms like:
- `regime_entropy_bonus`: reward when all 4 states have >5% of weeks
- `transition_rate_floor`: reject configs with <0.01 transitions/week
- `posterior_entropy_avg`: reward non-binary posteriors (avg entropy > 0.3)

### Fix 2: Reduce Sticky Prior Dramatically
- `sticky_diag=10.0, sticky_offdiag=2.0` → 5:1 ratio (62.5% self-transition prior)
- Or even `sticky_diag=5.0, sticky_offdiag=1.0` → still sticky but allows transitions

### Fix 3: Use Rolling (Not Expanding) Window for HMM Fit
- Actually implement `use_expanding_window=False` in engine.py
- Use a 5-7 year rolling window so recent regime changes aren't overwhelmed
- This is the config flag that exists but was never wired

### Fix 4: Disable Warm Start or Add Perturbation
- Either set `use_warm_start=False` to allow fresh discovery of states
- Or add random perturbation to warm-started params to escape local optima

### Fix 5: Add Discriminative Features
- Include features that genuinely separate all 4 quadrants (not just G vs D)
- E.g., commodity index (separates R from G), real rates (separates S from D)
- Reduce feature correlation to give HMM more separation power
