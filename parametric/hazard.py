"""Hazard model: how often qualifying events happen and how severe the index reading is.

Frequency
    Annual count of qualifying events N.  If the annual rate is known exactly, N is
    Poisson(lambda).  If the rate itself is uncertain (or events cluster in "active"
    years), the rate is drawn from a Gamma distribution with mean lambda and
    coefficient of variation `cv`; the count is then Negative Binomial
    (a Poisson-Gamma mixture).

Severity
    Distribution of the index value X *given* that a qualifying event occurred,
    i.e. X >= u where u is the event threshold.  All families are shifted so that
    their support starts at u.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

FAMILY_LABELS = {
    "gpd": "Generalised Pareto (peaks over threshold)",
    "lognormal": "Shifted lognormal",
    "gamma": "Shifted gamma",
    "weibull": "Shifted Weibull",
    "beta": "Beta (bounded index)",
}


# --------------------------------------------------------------------------- #
# Frequency
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Frequency:
    lam: float          # expected qualifying events per year
    cv: float = 0.0     # coefficient of variation of the annual rate (0 -> Poisson)

    @property
    def is_poisson(self) -> bool:
        return self.cv < 1e-9

    @property
    def r(self) -> float:
        """Gamma shape of the rate (Negative Binomial 'size')."""
        return 1.0 / self.cv ** 2

    def _dist(self):
        if self.is_poisson:
            return stats.poisson(self.lam)
        r = self.r
        return stats.nbinom(r, r / (r + self.lam))

    @property
    def mean(self) -> float:
        return self.lam

    @property
    def var(self) -> float:
        return self.lam + (self.lam * self.cv) ** 2

    def pmf(self, n):
        n = np.asarray(n)
        if self.lam <= 0:
            return (n == 0).astype(float)
        return self._dist().pmf(n)

    def ppf(self, q: float) -> int:
        if self.lam <= 0:
            return 0
        return int(self._dist().ppf(q))

    def pgf(self, z):
        """Probability generating function E[z^N]; accepts complex arrays."""
        z = np.asarray(z)
        if self.lam <= 0:
            return np.ones_like(z)
        if self.is_poisson:
            return np.exp(self.lam * (z - 1.0))
        return (1.0 - (self.lam / self.r) * (z - 1.0)) ** (-self.r)

    def thinned(self, p: float) -> "Frequency":
        """Count of events that also satisfy a condition with per-event probability p."""
        return Frequency(self.lam * p, self.cv)

    def prob_any(self, s) -> np.ndarray:
        """Annual probability that at least one event has a property of per-event probability s."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        return 1.0 - np.real(self.pgf(1.0 - s))

    def return_period(self, s) -> np.ndarray:
        p = self.prob_any(s)
        with np.errstate(divide="ignore"):
            return np.where(p > 0, 1.0 / np.maximum(p, 1e-300), np.inf)

    def event_prob_for_return_period(self, T) -> np.ndarray:
        """Per-event exceedance probability s such that the annual exceedance probability is 1/T."""
        T = np.asarray(T, dtype=float)
        target = 1.0 - 1.0 / T          # = pgf(1 - s)
        if self.lam <= 0:
            return np.full_like(T, np.inf)
        if self.is_poisson:
            return -np.log(target) / self.lam
        return self.r * (target ** (-1.0 / self.r) - 1.0) / self.lam

    def sample_rates(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.is_poisson:
            return np.full(n, self.lam)
        return rng.gamma(self.r, self.lam / self.r, size=n)


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=512)
def _make_dist(family: str, u: float, p1: float, p2: float, upper: float | None):
    if family == "gpd":                      # p1 = scale sigma, p2 = shape xi
        return stats.genpareto(c=p2, loc=u, scale=p1)
    if family == "lognormal":                # p1 = median of index, p2 = log-sd
        return stats.lognorm(s=p2, loc=u, scale=p1 - u)
    if family == "gamma":                    # p1 = mean of index, p2 = CV of excess
        a = 1.0 / p2 ** 2
        return stats.gamma(a, loc=u, scale=(p1 - u) / a)
    if family == "weibull":                  # p1 = scale of excess, p2 = shape k
        return stats.weibull_min(c=p2, loc=u, scale=p1)
    if family == "beta":                     # p1 = mean of index, p2 = concentration
        m = (p1 - u) / (upper - u)
        return stats.beta(m * p2, (1.0 - m) * p2, loc=u, scale=upper - u)
    raise ValueError(f"Unknown severity family {family!r}")


@dataclass(frozen=True)
class Severity:
    family: str
    threshold: float            # u: minimum index value of a qualifying event
    p1: float
    p2: float
    upper: float | None = None  # upper bound (beta only)

    # -- construction helpers ------------------------------------------------
    def validate(self) -> list[str]:
        errs = []
        u = self.threshold
        if self.family == "gpd" and self.p1 <= 0:
            errs.append("GPD scale must be positive.")
        if self.family == "lognormal":
            if self.p1 <= u:
                errs.append("Median must be above the event threshold.")
            if self.p2 <= 0:
                errs.append("Log-sd must be positive.")
        if self.family == "gamma":
            if self.p1 <= u:
                errs.append("Mean must be above the event threshold.")
            if self.p2 <= 0:
                errs.append("CV must be positive.")
        if self.family == "weibull" and (self.p1 <= 0 or self.p2 <= 0):
            errs.append("Weibull scale and shape must be positive.")
        if self.family == "beta":
            if self.upper is None or self.upper <= u:
                errs.append("Upper bound must exceed the event threshold.")
            elif not (u < self.p1 < self.upper):
                errs.append("Mean must lie strictly between the threshold and the upper bound.")
            if self.p2 <= 0:
                errs.append("Concentration must be positive.")
        return errs

    @property
    def dist(self):
        return _make_dist(self.family, float(self.threshold), float(self.p1), float(self.p2),
                          None if self.upper is None else float(self.upper))

    # -- distribution functions ---------------------------------------------
    def sf(self, x):
        return self.dist.sf(x)

    def cdf(self, x):
        return self.dist.cdf(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def ppf(self, q):
        return self.dist.ppf(q)

    def isf(self, q):
        return self.dist.isf(q)

    @property
    def support_max(self) -> float:
        if self.family == "gpd" and self.p2 < 0:
            return self.threshold + self.p1 / (-self.p2)
        if self.family == "beta":
            return float(self.upper)
        return np.inf

    @property
    def mean(self) -> float:
        if self.family == "gpd" and self.p2 >= 1:
            return np.inf
        return float(self.dist.mean())

    # -- parameter perturbations used by the sensitivity analysis -----------
    def scale_param(self) -> float:
        """The 'location/scale-type' parameter measured from the threshold."""
        if self.family in ("gpd", "weibull"):
            return self.p1
        return self.p1 - self.threshold

    def with_scale(self, factor: float) -> "Severity":
        u = self.threshold
        if self.family in ("gpd", "weibull"):
            return Severity(self.family, u, self.p1 * factor, self.p2, self.upper)
        new = u + (self.p1 - u) * factor
        if self.family == "beta":
            new = min(new, u + 0.999 * (self.upper - u))
        return Severity(self.family, u, new, self.p2, self.upper)

    def with_shape(self, factor: float, gpd_shift: float) -> "Severity":
        if self.family == "gpd":
            return Severity(self.family, self.threshold, self.p1, self.p2 + gpd_shift, self.upper)
        return Severity(self.family, self.threshold, self.p1, self.p2 * factor, self.upper)
