"""Explanatory copy: section introductions, parameter help, methodology and glossary."""

# --------------------------------------------------------------------------- #
# Tab introductions
# --------------------------------------------------------------------------- #
INTRO = """
**Parametric insurance** pays a pre-agreed amount when an **objectively measured index** crosses a
**trigger**. Because there is no loss adjustment, payment arrives in days or weeks rather than months. It can
also cover losses that are hard to insure on an indemnity basis, such as lost revenue from an airspace closure,
fuel-price shocks or mass cancellations. The trade-off is **basis risk**: the payout follows the index, not the
insured's actual loss.

This lab prices a one-year parametric cover for rare, event-driven aviation risks. Pick a product in the bar
at the top, then scroll down through the sections. Every parameter can be changed, and the price in the top
bar updates live.
"""

HAZARD = """
The **hazard model** describes the index, not the insured's loss. It has two parts. The **frequency** is how
many qualifying events occur in a year. The **severity** is how extreme the index reading is when an event
happens. For rare events, the tail of the severity distribution drives the price.
"""

CONTRACT = """
The **payout function** turns an index reading into cash. Its shape determines which part of the hazard
distribution is being insured. A high trigger insures only the tail: it is cheap in absolute terms but
expensive relative to its expected loss.
"""

PRICING = """
The **annual payout distribution** combines frequency, severity and the payout function, including the
annual aggregate limit. It is computed exactly with the FFT and cross-checked by Monte Carlo. The price is
**expected loss + risk load + costs**. The risk load pays for the capital needed to hold a rare-but-severe
risk, so it is often larger than the expected loss itself.
"""

BASIS = """
The buyer cares about **actual loss**, not the index. Here the insured's loss on each event is modelled as an
expected-loss ("damage") curve of the index plus random scatter. A share of loss-causing events can also be
invisible to the index altogether. The charts show how well the cover hedges real losses, and where it pays
too much or too little.
"""

SENSITIVITY = """
Rare-event parameters are uncertain because they rest on few observations. This section shows which assumptions
move the price most. Use it to decide where calibration effort is best spent, or where to build in a margin
for parameter uncertainty.
"""

CALIBRATION = """
Fit the hazard model to a catalogue of past events. The pre-loaded catalogue is **synthetic** (drawn from the
current preset), so replace it with your own data by editing the table or uploading a CSV with an
`index` column. Frequency comes from the count over the observation period. Severity is fitted by maximum
likelihood for each distribution family and ranked by AIC. With few events, the **rate-uncertainty CV**
suggested below is a principled way to carry parameter uncertainty into the price.
"""

# --------------------------------------------------------------------------- #
# Parameter help (tooltips)
# --------------------------------------------------------------------------- #
HELP = {
    "lam": "Expected number of **qualifying events per year**, meaning events where the index at least "
           "reaches the event threshold. 1/λ is the return period of a qualifying event. The expected loss "
           "scales roughly in proportion to λ.",
    "cv": "Uncertainty in the true annual rate, as a coefficient of variation. **0** means the rate is known "
          "exactly (Poisson counts). **Above 0**, the rate itself is uncertain, or events cluster in "
          "'active' years (conflict regimes, active hurricane seasons). Counts are then Negative Binomial. "
          "This leaves the uncapped expected loss unchanged, but makes several events in one year more likely "
          "and fattens the tail, so tail-based risk loads rise.",
    "family": "Shape of the index distribution **given** that an event occurred. The Generalised Pareto "
              "distribution is the extreme-value-theory choice for exceedances over a threshold. "
              "Lognormal, gamma and Weibull are common alternatives. Beta suits indices with a hard upper "
              "bound, such as a percentage.",
    "threshold": "Minimum index value for an event to count as 'qualifying'. λ refers to events above this "
                 "level, so if you change the threshold you should also re-calibrate λ.",
    "upper": "Hard upper bound of the index (e.g. 100% of flights).",
    "gpd_p1": "Scale σ: the spread of exceedances above the threshold, in index units. A larger σ moves the "
              "whole distribution to higher values.",
    "gpd_p2": "Tail shape ξ (extreme value index). **ξ > 0**: heavy, Pareto-type tail, where very long or "
              "extreme events remain plausible (for 0.5 < ξ < 1 the variance is infinite). **ξ = 0**: "
              "exponential tail. **ξ < 0**: bounded tail with a physical maximum at u + σ/|ξ|. This is "
              "usually the single most important parameter for remote layers.",
    "lognormal_p1": "Median of the index for a qualifying event: half of all events are above this level.",
    "lognormal_p2": "Log-standard deviation σ of the excess over the threshold. A higher σ means a longer "
                    "right tail.",
    "gamma_p1": "Mean index value for a qualifying event.",
    "gamma_p2": "Coefficient of variation of the excess over the threshold. A higher CV means more "
                "dispersion and a longer tail (CV = 1 is exponential).",
    "weibull_p1": "Scale η of the excess over the threshold, in index units.",
    "weibull_p2": "Shape k. **k < 1**: heavier-than-exponential tail. **k = 1**: exponential. **k > 1**: "
                  "lighter tail.",
    "beta_p1": "Mean index value for a qualifying event (between the threshold and the upper bound).",
    "beta_p2": "Concentration κ = α + β. A low κ spreads events across the whole range (more extreme "
               "events). A high κ concentrates them near the mean.",
    "kind": "**Binary**: the full limit is paid once the trigger is reached. Simple, but has a cliff edge. "
            "**Linear**: pays pro rata between attachment and exhaustion, like an option spread or "
            "'USD x per day'. **Stepped**: a schedule of tiers, such as hurricane categories.",
    "limit": "Maximum payout for a single event (the per-event limit). The rate on line is quoted against "
             "this amount.",
    "attachment": "Index level at which payments start. Raising it makes the cover more remote: the "
                  "premium falls, but the premium-to-expected-loss multiple usually rises.",
    "exhaustion": "Index level at which the full limit is paid. A wider attachment-to-exhaustion band "
                  "gives a smaller payout per index unit.",
    "trigger": "Index level that triggers the full payout.",
    "tiers": "Each row is an index threshold and the share of the limit paid once it is reached. The payout "
             "is the share of the highest tier reached. Shares must not decrease.",
    "agg": "Annual aggregate limit: the maximum total paid in the year across all events. **1×** means "
           "one full loss per year. Higher multiples, or unlimited, cover repeated events and cost more, "
           "particularly when events cluster (rate CV > 0).",
    "method": "How the **risk load** is set. **Cost of capital**: charge a return on the capital the risk "
              "consumes (Solvency II / rating-agency style). **Wang transform**: distort exceedance "
              "probabilities to give more weight to the tail (a common ILS / cat bond pricing approach). "
              "**Standard deviation**: load a multiple of the standard deviation (classical, but blind to "
              "skewness).",
    "coc_rate": "Required return on allocated capital, above the risk-free rate.",
    "coc_measure": "Risk measure used for stand-alone capital. **TVaR** averages the tail beyond the "
                   "confidence level. **VaR** ignores it: for a cover whose payout probability is below "
                   "1 − level, VaR is zero and so is the capital charge. This blind spot is why TVaR is "
                   "preferred for rare-event covers.",
    "coc_level": "Confidence level of the capital risk measure (99.5% matches Solvency II).",
    "capital_share": "Share of stand-alone capital actually allocated to this contract after "
                     "diversification within the carrier's portfolio. 100% means no diversification "
                     "credit.",
    "wang_theta": "Wang risk parameter θ. Exceedance probabilities u are distorted to Φ(Φ⁻¹(u) + θ). "
                  "θ = 0 gives the expected loss, and higher θ loads the tail more. Rare outcomes are "
                  "loaded proportionally more than frequent ones.",
    "sd_alpha": "Loading factor α in premium = EL + α·σ.",
    "expense_ratio": "Variable costs as a share of gross premium: brokerage, acquisition, premium tax "
                     "and administration.",
    "fixed_cost": "Per-contract fixed costs: modelling, index calculation agent, legal and structuring.",
    "min_rol": "Market floor on the rate on line. Remote layers often trade at a minimum price, whatever "
               "the modelled loss.",
    "basis_max": "The insured's expected loss once the index reaches the saturation level.",
    "basis_onset": "Index level at which the insured starts to lose money.",
    "basis_sat": "Index level at which the insured's expected loss reaches its maximum.",
    "basis_power": "Curvature of the damage curve. **1** is linear (e.g. lost revenue per day). **Above 1** "
                   "is convex (e.g. wind damage, which rises steeply at high speeds).",
    "basis_noise": "Scatter of the actual loss around the expected-loss curve (lognormal σ, mean-preserving). "
                   "This is basis risk from the index being an imperfect proxy: wind at the airport differs "
                   "from the maximum wind in the circle, and an airline's network exposure differs from the "
                   "closure duration.",
    "basis_miss": "Share of the insured's loss-causing events that the index does not register at all, "
                  "e.g. a storm that tracks just outside the circle, or a closure declared in a way the index "
                  "definition does not capture. These events cause losses that never trigger a payout.",
    "materiality": "An event loss above this amount counts as 'material'. Used to measure missed payouts "
                   "and payouts made without a material loss.",
    "shock": "Each parameter is moved down and up by this amount (relative), one at a time.",
    "n_sims": "Number of simulated years for the Monte Carlo cross-check and the basis-risk analysis. The "
              "price itself comes from the exact FFT calculation and does not depend on this setting.",
    "seed": "Random seed. Change it to see Monte Carlo sampling noise.",
}

# --------------------------------------------------------------------------- #
# Methodology
# --------------------------------------------------------------------------- #
METHODOLOGY = r"""
### 1 · Hazard model
**Frequency.** The annual number of qualifying events is $N \mid \Lambda \sim \text{Poisson}(\Lambda)$, with
$\Lambda \sim \text{Gamma}$ with mean $\lambda$ and coefficient of variation $c$. When $c = 0$, $N$ is Poisson.
When $c > 0$, $N$ is Negative Binomial with $\operatorname{Var}N = \lambda + c^2\lambda^2$. The Gamma mixing
represents parameter uncertainty or year-to-year clustering.

**Severity.** $X \mid \text{event}$ has support starting at the event threshold $u$. The Generalised Pareto
distribution, $S(x) = (1 + \xi (x-u)/\sigma)^{-1/\xi}$, is the limiting distribution of threshold excesses
(Pickands–Balkema–de Haan theorem), which makes it the natural default for rare-event indices.

**Return periods.** The annual probability that the index exceeds $x$ is $1 - P_N(1 - S(x))$, where $P_N$ is
the probability generating function of $N$. The return period is its reciprocal.

### 2 · Payout and annual aggregate
Each event pays $f(X) \in [0, L]$. Only triggering events matter, so the number of paying events is the
thinned count $N_t$, with mean $\lambda p$ where $p = S(\text{trigger})$, and the same mixing distribution.
The annual payout is

$$S_{\text{year}} = \min\Big(\sum_{i=1}^{N_t} Y_i^{+},\; \text{AAL}\Big), \qquad Y^{+} = f(X)\mid f(X)>0 .$$


### 3 · Exact calculation (FFT)
$Y^{+}$ is placed on a lattice with spacing $h = L/n$, using mean-preserving local moment matching built from
limited expected values $\text{LEV}(t) = \int_0^t S_{Y^+}(y)\,dy$. The compound distribution follows from
$\varphi_S = P_{N_t}(\varphi_{Y^+})$, evaluated with the FFT and long enough zero-padding to make aliasing
negligible (tail mass below $10^{-12}$). The aggregate limit is then applied to the lattice distribution.
Expected loss and standard deviation are also available in closed form, without the aggregate cap:

$$E[S] = \lambda\,E[f(X)], \qquad \operatorname{Var}S = \lambda E[f(X)^2] + c^2\lambda^2 E[f(X)]^2,$$

where $E[f(X)] = \int g'(x)S(x)\,dx$ is evaluated by quadrature. The Pricing section shows all three routes
side by side.

### 4 · Premium

$$\text{Gross} = \max\left(\frac{EL + RL + F}{1-e},\; \text{ROL}_{\min}\cdot L\right)$$

| Risk-load method | Risk load $RL$ |
|---|---|
| Cost of capital | $r_{CoC}\cdot k\cdot(\rho_\alpha(S) - EL)$, with $\rho$ = VaR or TVaR and $k$ the capital share |
| Wang transform | $\int_0^\infty \Phi(\Phi^{-1}(S_S(s)) + \theta)\,ds - EL$ |
| Standard deviation | $\alpha\cdot\sigma_S$ |

Rate on line = Gross / $L$. Multiple = Gross / EL. Expected loss ratio = EL / Gross. Discounting is ignored,
because the cover lasts one year and payouts are settled quickly.

### 5 · Monte Carlo and basis risk
The simulation draws years, annual rates, event counts and index values directly. It validates the FFT
result (the standard error is shown) and adds the insured's actual loss for each event:

$$\text{Loss} = D(X)\cdot e^{\sigma_b Z - \sigma_b^2/2}, \qquad D(x) = M\cdot\min\!\Big(1,\Big(\tfrac{x - x_0}{x_1 - x_0}\Big)_+^{\gamma}\Big).$$

Index-invisible loss events arrive at rate $\Lambda\,q/(1-q)$ and never pay. Hedge effectiveness is
$1 - \operatorname{Var}(\text{Loss} - \text{Payout}) / \operatorname{Var}(\text{Loss})$.

### 6 · Calibration
$\hat\lambda = n/T$, with an exact Poisson confidence interval. Under a Jeffreys prior the posterior of
$\lambda$ is $\text{Gamma}(n + \tfrac12, T)$, with CV $1/\sqrt{n + \tfrac12}$. Using that CV as the
rate-uncertainty parameter makes the predictive count distribution Negative Binomial, which carries
parameter uncertainty straight into the price. Severity is fitted by maximum likelihood with the location
fixed at $u$, and ranked by AIC.

### 7 · Limitations (it is a demo)
The model assumes stationarity (no climate or geopolitical trend), independence between events and between
severity and frequency, a single index and a single risk period. It ignores counterparty and index-publication
risk and investment income. All parameters are illustrative.
"""

GLOSSARY = [
    ("Event frequency λ", "Expected qualifying events per year", "↑ premium (≈ proportional)"),
    ("Rate uncertainty CV", "Uncertainty / clustering of the annual rate", "↑ tail-based risk load; EL flat (lower if capped)"),
    ("Event threshold u", "Minimum index value of a qualifying event", "Defines what λ counts"),
    ("Severity scale", "Typical size of the index given an event", "↑ premium"),
    ("Tail shape ξ / σ / k / κ", "How extreme the largest events can be", "Heavier tail ↑ premium, most for remote layers"),
    ("Trigger / attachment", "Index level where payments start", "↑ trigger ↓ premium, ↑ multiple of EL"),
    ("Exhaustion", "Index level where the full limit is paid", "↑ exhaustion ↓ premium (slower pay-out)"),
    ("Per-event limit", "Maximum paid per event", "↑ premium (≈ proportional)"),
    ("Annual aggregate limit", "Maximum paid per year", "↑ premium, more so with clustering"),
    ("Cost-of-capital rate", "Return required on allocated capital", "↑ risk load"),
    ("Capital share", "Allocated share of stand-alone capital (diversification)", "↑ risk load"),
    ("Wang θ", "Strength of the tail distortion", "↑ risk load"),
    ("SD loading α", "Multiple of standard deviation charged", "↑ risk load"),
    ("Expense ratio", "Brokerage, acquisition and admin as % of gross", "↑ premium (grossed up)"),
    ("Fixed costs", "Per-contract structuring and agent costs", "↑ premium; matters most for small limits"),
    ("Minimum ROL", "Market price floor", "Binds for very remote covers"),
    ("Basis volatility σ_b", "Scatter of actual loss around the damage curve", "No price effect; ↓ hedge effectiveness"),
    ("Index-miss share q", "Loss events not seen by the index", "No price effect; ↑ uncovered losses"),
]

COMPARISON = [
    ("Claim trigger", "Index crosses a pre-agreed level", "Insured proves a covered loss"),
    ("Payout amount", "Pre-agreed schedule", "Adjusted actual loss"),
    ("Speed", "Days to weeks", "Months to years"),
    ("Insurable losses", "Includes non-damage BI, revenue, fuel cost", "Usually requires physical damage"),
    ("Moral hazard / fraud", "Minimal: the insured cannot influence the index", "Requires adjusting and audit"),
    ("Main drawback", "Basis risk", "Slow, costly, disputes"),
]
