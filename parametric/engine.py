"""Loss engine: annual payout distribution, risk measures, pricing and simulation.

Two independent calculation routes are provided so that each can check the other:

1. Exact route (FFT).  Only events that trigger matter, so the count of paying
   events is the thinned process N_t (Poisson(lambda*p) or Negative Binomial with
   the same dispersion).  The conditional per-event payout Y+ = f(X) | f(X) > 0 is
   discretised on a lattice with a mean-preserving (local moment-matching)
   scheme, the compound distribution is obtained through its characteristic
   function  phi_S = PGF_N_t(phi_Y+)  evaluated with the FFT, and the annual
   aggregate limit is applied to the resulting distribution.

2. Monte Carlo route.  Simulate years, events and index values directly.  Used
   to validate the exact route and to model basis risk (the insured's actual
   loss), which has no closed form.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import stats

from .contract import Payout
from .hazard import Frequency, Severity


# =========================================================================== #
# Annual payout distribution on a lattice
# =========================================================================== #
class AnnualDistribution:
    """Discrete distribution of the annual payout S on the lattice {0, h, 2h, ...}."""

    def __init__(self, h: float, pmf: np.ndarray):
        pmf = np.clip(np.asarray(pmf, dtype=float), 0.0, None)
        pmf[-1] += max(0.0, 1.0 - pmf.sum())       # put any truncated mass in the tail
        self.h = float(h)
        self.pmf = pmf / pmf.sum()
        self.values = np.arange(len(self.pmf)) * self.h
        self.cdf = np.minimum(np.cumsum(self.pmf), 1.0)

    # -- moments --------------------------------------------------------- #
    @property
    def mean(self) -> float:
        return float(self.values @ self.pmf)

    @property
    def var(self) -> float:
        return max(float((self.values ** 2) @ self.pmf) - self.mean ** 2, 0.0)

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.var))

    @property
    def max_value(self) -> float:
        nz = np.nonzero(self.pmf > 1e-15)[0]
        return float(self.values[nz[-1]]) if nz.size else 0.0

    # -- tail measures --------------------------------------------------- #
    def _q_index(self, alpha: float) -> int:
        return int(min(np.searchsorted(self.cdf, alpha - 1e-12), len(self.cdf) - 1))

    def quantile(self, alpha: float) -> float:
        """Value-at-Risk: smallest s with P(S <= s) >= alpha."""
        return float(self.values[self._q_index(alpha)])

    def tvar(self, alpha: float) -> float:
        """Tail Value-at-Risk (expected shortfall), exact for discrete distributions."""
        i = self._q_index(alpha)
        q = self.values[i]
        above = float(self.values[i + 1:] @ self.pmf[i + 1:])
        return (above + q * (self.cdf[i] - alpha)) / (1.0 - alpha)

    def exceedance(self) -> np.ndarray:
        """P(S > s) at each lattice point."""
        return np.clip(1.0 - self.cdf, 0.0, 1.0)

    def wang(self, theta: float) -> float:
        """Wang-transform price  integral g(S_S(s)) ds  with g(u) = Phi(Phi^-1(u) + theta)."""
        u = self.exceedance()
        g = stats.norm.cdf(stats.norm.ppf(u) + theta)
        return float(self.h * np.sum(g))

    def prob_at_least(self, s: float) -> float:
        return float(self.pmf[self.values >= s - 1e-9 * max(self.h, 1.0)].sum())

    def capped(self, cap: float | None) -> "AnnualDistribution":
        if cap is None:
            return self
        i = int(round(cap / self.h))
        if i >= len(self.pmf) - 1:
            return self
        pmf = self.pmf[: i + 1].copy()
        pmf[-1] += self.pmf[i + 1:].sum()
        return AnnualDistribution(self.h, pmf)


def per_event_lattice(sev: Severity, payout: Payout, n_grid: int, sub: int = 16):
    """Mean-preserving lattice masses for Y+ = f(X) | f(X) > 0 on {0, h, ..., n_grid*h}.

    Uses limited expected values LEV(t) = E[min(Y+, t)]:
        m_0 = 1 - LEV(h)/h,
        m_j = (2 LEV(jh) - LEV((j-1)h) - LEV((j+1)h)) / h.
    This keeps total probability and the mean exactly (up to quadrature).
    """
    p = float(sev.sf(payout.trigger)) if np.isfinite(payout.trigger) else 0.0
    h = payout.limit / n_grid
    hs = h / sub
    y_mid = (np.arange(n_grid * sub) + 0.5) * hs
    s_mid = np.clip(payout.sf_payout(y_mid, sev) / p, 0.0, 1.0)
    lev = np.concatenate([[0.0], np.cumsum(s_mid.reshape(n_grid, sub).sum(axis=1)) * hs])
    lev_ext = np.concatenate([[-h], lev, [lev[-1]]])
    masses = (2.0 * lev_ext[1:-1] - lev_ext[:-2] - lev_ext[2:]) / h
    masses = np.clip(masses, 0.0, None)
    return h, masses / masses.sum(), p


@dataclass
class AnnualResult:
    capped: AnnualDistribution
    uncapped: AnnualDistribution
    p_trigger: float            # per-event probability of a positive payout
    n_grid: int


def annual_distribution(freq: Frequency, sev: Severity, payout: Payout,
                        n_grid: int = 1000, tail_eps: float = 1e-12,
                        max_len: int = 2 ** 21) -> AnnualResult:
    p = float(sev.sf(payout.trigger)) if np.isfinite(payout.trigger) else 0.0
    if p <= 0 or freq.lam <= 0:
        d = AnnualDistribution(payout.limit / n_grid, np.array([1.0]))
        return AnnualResult(d, d, p, n_grid)

    ft = freq.thinned(p)
    n_max = max(1, ft.ppf(1.0 - tail_eps))
    if n_max * n_grid + 1 > max_len:
        n_grid = max(64, (max_len - 1) // n_max)
    h, masses, _ = per_event_lattice(sev, payout, n_grid)

    length = n_max * n_grid + 1
    nfft = 1 << int(np.ceil(np.log2(length)))
    phi = np.fft.rfft(masses, nfft)
    agg = np.fft.irfft(ft.pgf(phi), nfft)[:length]
    uncapped = AnnualDistribution(h, agg)
    return AnnualResult(uncapped.capped(payout.agg_limit), uncapped, p, n_grid)


def occurrence_exceedance(freq: Frequency, sev: Severity, payout: Payout, y) -> np.ndarray:
    """Annual probability that the largest single-event payout exceeds y (OEP)."""
    return freq.prob_any(payout.sf_payout(np.asarray(y, dtype=float), sev))


# =========================================================================== #
# Closed-form checks (no aggregate cap)
# =========================================================================== #
@dataclass
class Analytic:
    p_trigger: float            # per-event P(payout > 0)
    p_full: float               # per-event P(payout = max)
    mean_event: float           # E[f(X)] per qualifying event
    mean_given_trigger: float   # E[f(X) | f(X) > 0]
    el_uncapped: float          # E[S] without the aggregate cap
    sd_uncapped: float
    p_attach: float             # annual P(S > 0)
    p_full_annual: float        # annual P(at least one event pays the maximum)


def analytic(freq: Frequency, sev: Severity, payout: Payout) -> Analytic:
    m1, m2 = payout.moments(sev)
    p = float(sev.sf(payout.trigger)) if np.isfinite(payout.trigger) else 0.0
    pf = float(sev.sf(payout.full_level)) if np.isfinite(payout.full_level) else 0.0
    if payout.kind == "linear":
        pf = float(sev.sf(payout.exhaustion))
    el = freq.lam * m1
    var = freq.lam * m2 + (freq.lam * freq.cv) ** 2 * m1 ** 2
    return Analytic(
        p_trigger=p, p_full=pf, mean_event=m1,
        mean_given_trigger=m1 / p if p > 0 else 0.0,
        el_uncapped=el, sd_uncapped=float(np.sqrt(var)),
        p_attach=float(freq.prob_any(p)), p_full_annual=float(freq.prob_any(pf)),
    )


# =========================================================================== #
# Pricing
# =========================================================================== #
METHOD_LABELS = {
    "coc": "Cost of capital",
    "wang": "Wang transform",
    "sd": "Standard-deviation principle",
}


@dataclass(frozen=True)
class PricingAssumptions:
    method: str = "coc"
    sd_alpha: float = 0.25
    wang_theta: float = 0.30
    coc_rate: float = 0.10
    coc_measure: str = "TVaR"       # "TVaR" or "VaR"
    coc_level: float = 0.99
    capital_share: float = 0.40     # share of stand-alone capital allocated after diversification
    expense_ratio: float = 0.15     # brokerage + acquisition + admin, % of gross premium
    fixed_cost: float = 25_000.0    # modelling, index agent, legal (per contract)
    min_rol: float = 0.0            # minimum rate on line (premium floor)


@dataclass
class Price:
    method: str
    expected_loss: float
    risk_load: float
    fixed_cost: float
    variable_expenses: float
    floor_uplift: float
    gross: float
    limit: float
    capital: float = 0.0
    risk_measure: float = 0.0

    @property
    def rol(self) -> float:
        return self.gross / self.limit

    @property
    def lol(self) -> float:
        return self.expected_loss / self.limit

    @property
    def multiple(self) -> float:
        return self.gross / self.expected_loss if self.expected_loss > 0 else np.inf

    @property
    def loss_ratio(self) -> float:
        return self.expected_loss / self.gross if self.gross > 0 else 0.0


def risk_load(dist: AnnualDistribution, a: PricingAssumptions, method: str | None = None):
    """Return (risk load, allocated capital, risk measure)."""
    method = method or a.method
    el = dist.mean
    if method == "sd":
        return a.sd_alpha * dist.sd, 0.0, 0.0
    if method == "wang":
        return max(dist.wang(a.wang_theta) - el, 0.0), 0.0, 0.0
    rm = dist.tvar(a.coc_level) if a.coc_measure == "TVaR" else dist.quantile(a.coc_level)
    capital = max(rm - el, 0.0) * a.capital_share
    return a.coc_rate * capital, capital, rm


def price_contract(dist: AnnualDistribution, a: PricingAssumptions, limit: float,
                   method: str | None = None) -> Price:
    method = method or a.method
    el = dist.mean
    rl, capital, rm = risk_load(dist, a, method)
    technical = el + rl + a.fixed_cost
    gross = technical / (1.0 - a.expense_ratio)
    floor = a.min_rol * limit
    uplift = max(floor - gross, 0.0)
    gross_final = gross + uplift
    return Price(method, el, rl, a.fixed_cost, gross - technical, uplift, gross_final, limit,
                 capital, rm)


# =========================================================================== #
# Monte Carlo
# =========================================================================== #
@dataclass
class HazardSample:
    rates: np.ndarray       # simulated annual rate per year
    counts: np.ndarray      # number of qualifying events per year
    year: np.ndarray        # year index of each event
    index: np.ndarray       # index value of each event

    @property
    def n_years(self) -> int:
        return len(self.counts)


def simulate_hazard(freq: Frequency, sev: Severity, n_years: int, seed: int) -> HazardSample:
    rng = np.random.default_rng([seed, 1])
    rates = freq.sample_rates(rng, n_years)
    counts = rng.poisson(rates)
    year = np.repeat(np.arange(n_years), counts)
    x = sev.ppf(rng.random(int(counts.sum())))
    return HazardSample(rates, counts, year, x)


def annual_payouts(sample: HazardSample, payout: Payout):
    """Per-event payouts and annual totals (uncapped and capped)."""
    ev = payout.payout(sample.index)
    unc = np.bincount(sample.year, weights=ev, minlength=sample.n_years)
    cap = unc if payout.agg_limit is None else np.minimum(unc, payout.agg_limit)
    return ev, unc, cap


def sample_tvar(x: np.ndarray, alpha: float) -> float:
    xs = np.sort(x)
    k = int(np.floor(alpha * len(xs)))
    return float(xs[k:].mean()) if k < len(xs) else float(xs[-1])


@dataclass
class MCStats:
    mean: float
    se: float
    sd: float
    p_attach: float
    p_attach_se: float
    var99: float
    tvar99: float

    @staticmethod
    def of(x: np.ndarray) -> "MCStats":
        n = len(x)
        pa = float(np.mean(x > 0))
        return MCStats(float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), float(x.std(ddof=1)),
                       pa, float(np.sqrt(pa * (1 - pa) / n)),
                       float(np.quantile(x, 0.99, method="inverted_cdf")), sample_tvar(x, 0.99))


# =========================================================================== #
# Basis risk
# =========================================================================== #
@dataclass(frozen=True)
class BasisModel:
    max_loss: float         # insured's expected loss once the index reaches saturation
    onset: float            # index level where the insured starts losing money
    saturation: float       # index level where the expected loss reaches max_loss
    power: float            # curvature of the damage curve (1 = linear)
    noise_sd: float         # log-sd of actual loss around the damage curve
    miss_share: float       # share of loss-causing events the index does not record

    def expected_loss(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = np.clip((x - self.onset) / max(self.saturation - self.onset, 1e-12), 0.0, 1.0)
        return self.max_loss * z ** self.power


@dataclass
class BasisResult:
    annual_loss: np.ndarray
    annual_payout: np.ndarray
    ev_index: np.ndarray
    ev_loss: np.ndarray
    ev_payout: np.ndarray
    ev_captured: np.ndarray


def simulate_basis(sample: HazardSample, sev: Severity, payout: Payout,
                   basis: BasisModel, seed: int) -> BasisResult:
    rng = np.random.default_rng([seed, 2])
    s = basis.noise_sd
    ev_pay, _, annual_pay = annual_payouts(sample, payout)
    noise = np.exp(s * rng.standard_normal(len(sample.index)) - 0.5 * s * s)
    ev_loss = basis.expected_loss(sample.index) * noise
    annual_loss = np.bincount(sample.year, weights=ev_loss, minlength=sample.n_years)

    q = basis.miss_share
    if q > 0:
        cnt_u = rng.poisson(sample.rates * q / (1.0 - q))
        yr_u = np.repeat(np.arange(sample.n_years), cnt_u)
        x_u = sev.ppf(rng.random(int(cnt_u.sum())))
        loss_u = basis.expected_loss(x_u) * np.exp(s * rng.standard_normal(len(x_u)) - 0.5 * s * s)
        annual_loss = annual_loss + np.bincount(yr_u, weights=loss_u, minlength=sample.n_years)
    else:
        x_u = np.empty(0)
        loss_u = np.empty(0)

    return BasisResult(
        annual_loss=annual_loss, annual_payout=annual_pay,
        ev_index=np.concatenate([sample.index, x_u]),
        ev_loss=np.concatenate([ev_loss, loss_u]),
        ev_payout=np.concatenate([ev_pay, np.zeros(len(x_u))]),
        ev_captured=np.concatenate([np.ones(len(ev_loss), bool), np.zeros(len(x_u), bool)]),
    )


@dataclass
class BasisMetrics:
    corr: float
    spearman: float
    hedge_effectiveness: float
    recovery_ratio: float
    missed_share: float
    false_share: float
    gross_mean: float
    gross_var99: float
    gross_tvar99: float
    net_mean: float
    net_var99: float
    net_tvar99: float
    n_material: int
    n_paid: int


def basis_metrics(res: BasisResult, premium: float, materiality: float) -> BasisMetrics:
    L, P = res.annual_loss, res.annual_payout
    net = L - P + premium
    vl = L.var()
    corr = float(np.corrcoef(L, P)[0, 1]) if P.std() > 0 and L.std() > 0 else 0.0
    sp = float(stats.spearmanr(L, P).statistic) if P.std() > 0 and L.std() > 0 else 0.0
    material = res.ev_loss >= materiality
    paid = res.ev_payout > 0
    return BasisMetrics(
        corr=corr, spearman=sp,
        hedge_effectiveness=float(1.0 - (L - P).var() / vl) if vl > 0 else 0.0,
        recovery_ratio=float(P.mean() / L.mean()) if L.mean() > 0 else 0.0,
        missed_share=float((material & ~paid).sum() / material.sum()) if material.any() else 0.0,
        false_share=float((paid & ~material).sum() / paid.sum()) if paid.any() else 0.0,
        gross_mean=float(L.mean()), gross_var99=float(np.quantile(L, 0.99)),
        gross_tvar99=sample_tvar(L, 0.99),
        net_mean=float(net.mean()), net_var99=float(np.quantile(net, 0.99)),
        net_tvar99=sample_tvar(net, 0.99),
        n_material=int(material.sum()), n_paid=int(paid.sum()),
    )


# =========================================================================== #
# Sensitivities
# =========================================================================== #
def quick_price(freq, sev, payout, a: PricingAssumptions, n_grid: int = 400) -> Price:
    res = annual_distribution(freq, sev, payout, n_grid=n_grid)
    return price_contract(res.capped, a, payout.limit)


def tornado(freq: Frequency, sev: Severity, payout: Payout, a: PricingAssumptions,
            shock: float, unit: str):
    """One-at-a-time parameter shocks; returns rows sorted by premium swing."""
    u = sev.threshold
    gpd_shift = round(0.5 * shock, 3)
    scale_names = {"gpd": "Severity scale σ", "lognormal": "Median excess over threshold",
                   "gamma": "Mean excess over threshold", "weibull": "Weibull scale η",
                   "beta": "Mean excess over threshold"}
    shape_names = {"gpd": f"Tail shape ξ (±{gpd_shift:.2f})", "lognormal": "Log-sd σ",
                   "gamma": "CV of excess", "weibull": "Weibull shape k",
                   "beta": "Concentration κ"}
    lo, hi = 1 - shock, 1 + shock
    cv_lo, cv_hi = (freq.cv * lo, freq.cv * hi) if freq.cv > 0 else (0.0, 0.5 * shock)
    risk_param = {"sd": ("sd_alpha", "Risk-load factor α"),
                  "wang": ("wang_theta", "Wang risk parameter θ"),
                  "coc": ("coc_rate", "Cost-of-capital rate")}[a.method]

    cases = [
        ("Event frequency λ", lambda f: (Frequency(freq.lam * f, freq.cv), sev, payout, a), lo, hi),
        ("Rate uncertainty (CV of λ)",
         lambda c: (Frequency(freq.lam, c), sev, payout, a), cv_lo, cv_hi),
        (scale_names[sev.family], lambda f: (freq, sev.with_scale(f), payout, a), lo, hi),
        (shape_names[sev.family],
         lambda f: (freq, sev.with_shape(f, gpd_shift * (1 if f > 1 else -1)), payout, a), lo, hi),
        ("Trigger levels",
         lambda f: (freq, sev, payout.map_thresholds(lambda t: u + (t - u) * f), a), lo, hi),
        (risk_param[1], lambda f: (freq, sev, payout,
                                   replace(a, **{risk_param[0]: getattr(a, risk_param[0]) * f})), lo, hi),
        ("Expense ratio", lambda f: (freq, sev, payout,
                                     replace(a, expense_ratio=min(a.expense_ratio * f, 0.9))), lo, hi),
    ]
    rows = []
    for name, build, v_lo, v_hi in cases:
        out = []
        for v in (v_lo, v_hi):
            fq, sv, po, aa = build(v)
            if sv.validate() or po.validate():
                out.append(np.nan)
                continue
            out.append(quick_price(fq, sv, po, aa).gross)
        rows.append(dict(parameter=name, low=out[0], high=out[1]))
    rows.sort(key=lambda r: -abs(np.nan_to_num(r["high"]) - np.nan_to_num(r["low"])))
    return rows


def trigger_sweep(freq, sev, payout, a: PricingAssumptions, levels: np.ndarray):
    """Shift the whole payout schedule so that the trigger sits at each level."""
    out = []
    base = payout.trigger
    for lvl in levels:
        d = lvl - base
        po = payout.map_thresholds(lambda t, d=d: t + d)
        res = annual_distribution(freq, sev, po, n_grid=400)
        pr = price_contract(res.capped, a, po.limit)
        p = float(sev.sf(po.trigger))
        out.append(dict(trigger=lvl, rol=pr.rol, lol=pr.lol, multiple=pr.multiple,
                        p_attach=float(freq.prob_any(p)), gross=pr.gross))
    return out


def frequency_severity_grid(freq, sev, payout, a: PricingAssumptions, lam_mult, sev_mult):
    grid = np.zeros((len(sev_mult), len(lam_mult)))
    for i, sm in enumerate(sev_mult):
        s2 = sev.with_scale(sm)
        for j, lm in enumerate(lam_mult):
            grid[i, j] = quick_price(Frequency(freq.lam * lm, freq.cv), s2, payout, a).rol
    return grid
