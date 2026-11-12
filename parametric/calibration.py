"""Calibration of the hazard model from an event catalogue.

* Frequency: lambda = n / T (MLE), exact Poisson confidence interval, and a
  Jeffreys-prior posterior Gamma(n + 1/2, T) whose CV 1/sqrt(n + 1/2) is a natural
  starting point for the 'rate uncertainty' parameter.
* Severity: maximum likelihood for each family with the location fixed at the
  event threshold, ranked by AIC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .hazard import FAMILY_LABELS, Severity


@dataclass
class FrequencyFit:
    n: int
    years: float
    lam_mle: float
    lam_lo: float
    lam_hi: float
    lam_post: float
    cv_post: float


def fit_frequency(n: int, years: float, level: float = 0.90) -> FrequencyFit:
    a = (1 - level) / 2
    lo = stats.chi2.ppf(a, 2 * n) / (2 * years) if n > 0 else 0.0
    hi = stats.chi2.ppf(1 - a, 2 * n + 2) / (2 * years)
    return FrequencyFit(n, years, n / years, lo, hi, (n + 0.5) / years, 1 / np.sqrt(n + 0.5))


@dataclass
class SeverityFit:
    family: str
    p1: float
    p2: float
    loglik: float
    aic: float
    ok: bool
    message: str = ""

    def severity(self, u: float, upper: float | None) -> Severity:
        return Severity(self.family, u, self.p1, self.p2, upper)


def _fit_one(family: str, x: np.ndarray, u: float, upper: float | None) -> SeverityFit:
    eps = 1e-6 * max(1.0, np.ptp(x) if len(x) > 1 else 1.0)
    x = np.maximum(x, u + eps)
    try:
        if family == "gpd":
            c, _, scale = stats.genpareto.fit(x, floc=u)
            p1, p2 = scale, c
        elif family == "lognormal":
            z = np.log(x - u)
            p1, p2 = u + np.exp(z.mean()), max(z.std(ddof=0), 1e-3)
        elif family == "gamma":
            a, _, scale = stats.gamma.fit(x, floc=u)
            p1, p2 = u + a * scale, 1 / np.sqrt(a)
        elif family == "weibull":
            c, _, scale = stats.weibull_min.fit(x, floc=u)
            p1, p2 = scale, c
        elif family == "beta":
            if upper is None:
                return SeverityFit(family, np.nan, np.nan, np.nan, np.nan, False, "needs upper bound")
            span = upper - u
            xb = np.minimum(x, upper - 1e-6 * span)
            a, b, _, _ = stats.beta.fit(xb, floc=u, fscale=span)
            p1, p2 = u + span * a / (a + b), a + b
            x = xb
        else:
            raise ValueError(family)
        sev = Severity(family, u, float(p1), float(p2), upper)
        if sev.validate():
            return SeverityFit(family, p1, p2, np.nan, np.nan, False, "; ".join(sev.validate()))
        ll = float(np.sum(sev.dist.logpdf(x)))
        if not np.isfinite(ll):
            return SeverityFit(family, p1, p2, ll, np.nan, False, "data outside support")
        return SeverityFit(family, float(p1), float(p2), ll, 4.0 - 2.0 * ll, True)
    except Exception as exc:  # noqa: BLE001 - surface any optimiser failure to the UI
        return SeverityFit(family, np.nan, np.nan, np.nan, np.nan, False, str(exc)[:80])


def fit_all(x, u: float, upper: float | None, families=None) -> list[SeverityFit]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= u)]
    families = families or list(FAMILY_LABELS)
    if len(x) < 3:
        return [SeverityFit(f, np.nan, np.nan, np.nan, np.nan, False, "need at least 3 events")
                for f in families]
    fits = [_fit_one(f, x, u, upper) for f in families]
    return sorted(fits, key=lambda f: (not f.ok, f.aic if f.ok else np.inf))


def equivalent_parameters(sev: Severity, upper: float | None, n: int = 2000) -> dict:
    """Fit every family to a deterministic quantile sample of `sev`.

    Gives sensible starting values when the user switches severity family.
    """
    q = (np.arange(n) + 0.5) / n
    x = sev.ppf(q)
    x = x[np.isfinite(x)]
    out = {}
    for f in fit_all(x, sev.threshold, upper):
        if f.ok:
            out[f.family] = (f.p1, f.p2)
    out[sev.family] = (sev.p1, sev.p2)
    return out


def synthetic_catalogue(freq_lam: float, sev: Severity, years: int, seed: int,
                        last_year: int = 2025, decimals: int = 1):
    rng = np.random.default_rng(seed)
    n = rng.poisson(freq_lam * years)
    yrs = np.sort(rng.integers(last_year - years + 1, last_year + 1, size=n))
    x = np.round(sev.ppf(rng.random(n)), decimals)
    x = np.maximum(x, sev.threshold)
    return yrs, x
