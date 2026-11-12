"""Numerical checks: closed form vs FFT vs Monte Carlo.

Run with:  python -m pytest -q tests/   (or simply: python tests/test_engine.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parametric import calibration, engine  # noqa: E402
from parametric.contract import Payout  # noqa: E402
from parametric.hazard import Frequency, Severity  # noqa: E402
from parametric.presets import AGG_OPTIONS, PRESETS  # noqa: E402


def build(p, kind=None, cv=None, agg=None):
    freq = Frequency(p.lam, p.cv if cv is None else cv)
    sev = Severity(p.family, p.threshold, p.p1, p.p2, p.upper)
    kind = kind or p.kind
    aggm = AGG_OPTIONS[p.agg] if agg is None else agg
    L = p.limit_m * 1e6
    if kind == "stepped":
        po, _ = Payout.make_stepped(L, p.tiers, aggm)
    elif kind == "binary":
        po = Payout("binary", L, p.attachment, agg_multiple=aggm)
    else:
        po = Payout("linear", L, p.attachment, p.exhaustion, agg_multiple=aggm)
    return freq, sev, po


def test_fft_matches_closed_form_uncapped():
    for p in PRESETS:
        for kind in ("binary", "linear", "stepped"):
            for cv in (0.0, 0.5):
                freq, sev, po = build(p, kind, cv, agg="unl")
                po = Payout(po.kind, po.limit, po.attachment, po.exhaustion, po.tiers, None)
                an = engine.analytic(freq, sev, po)
                res = engine.annual_distribution(freq, sev, po)
                d = res.uncapped
                assert abs(d.mean / an.el_uncapped - 1) < 1e-6, (p.id, kind, cv, d.mean, an.el_uncapped)
                # variance: lattice discretisation adds O(h^2) error
                assert abs(d.sd / an.sd_uncapped - 1) < 2e-3, (p.id, kind, cv, d.sd, an.sd_uncapped)
                assert abs((1 - d.pmf[0]) / an.p_attach - 1) < 2e-2 or kind == "linear"


def test_fft_matches_monte_carlo_capped():
    for p in PRESETS:
        for cv in (0.0, 0.4):
            freq, sev, po = build(p, cv=cv)
            res = engine.annual_distribution(freq, sev, po)
            sample = engine.simulate_hazard(freq, sev, 400_000, seed=11)
            _, unc, cap = engine.annual_payouts(sample, po)
            mc = engine.MCStats.of(cap)
            z = (mc.mean - res.capped.mean) / mc.se
            assert abs(z) < 4, (p.id, cv, mc.mean, res.capped.mean, z)
            an = engine.analytic(freq, sev, po)
            z2 = (mc.p_attach - an.p_attach) / mc.p_attach_se
            assert abs(z2) < 4, (p.id, cv, mc.p_attach, an.p_attach)


def test_risk_measures_simple_binary():
    # Binary cover, Poisson: S takes values 0, L, 2L ... capped at L -> Bernoulli(q)
    lam, L = 0.05, 1.0e6
    freq = Frequency(lam)
    sev = Severity("gpd", 0.0, 1.0, 0.0)             # exponential(1)
    po = Payout("binary", L, 0.0, agg_multiple=1.0)  # triggers on every event
    res = engine.annual_distribution(freq, sev, po)
    q = 1 - np.exp(-lam)
    d = res.capped
    assert abs(d.mean - q * L) < 1e-6 * L
    assert abs(d.sd - L * np.sqrt(q * (1 - q))) < 1e-6 * L
    assert d.quantile(0.90) == 0.0 and d.quantile(0.99) == L
    assert abs(d.tvar(0.99) - L) < 1e-6 * L
    assert abs(d.tvar(0.90) - q * L / 0.10) < 1e-6 * L
    # Wang transform of a Bernoulli: L * Phi(Phi^-1(q) + theta)
    from scipy.stats import norm
    assert abs(d.wang(0.3) - L * norm.cdf(norm.ppf(q) + 0.3)) < 1e-6 * L
    assert abs(d.wang(0.0) - d.mean) < 1e-9 * L


def test_return_period_inversion():
    for cv in (0.0, 0.6):
        f = Frequency(0.3, cv)
        for T in (5.0, 50.0, 500.0):
            s = f.event_prob_for_return_period(T)
            assert abs(f.return_period(s) - T) < 1e-6 * T


def test_calibration_recovers_parameters():
    sev = Severity("gpd", 1.0, 3.0, 0.4)
    rng = np.random.default_rng(3)
    x = sev.ppf(rng.random(20_000))
    fits = {f.family: f for f in calibration.fit_all(x, 1.0, None)}
    assert fits["gpd"].ok and abs(fits["gpd"].p1 - 3.0) < 0.15 and abs(fits["gpd"].p2 - 0.4) < 0.04
    assert calibration.fit_all(x, 1.0, None)[0].family == "gpd"


def report():
    a = engine.PricingAssumptions()
    for p in PRESETS:
        freq, sev, po = build(p)
        an = engine.analytic(freq, sev, po)
        res = engine.annual_distribution(freq, sev, po)
        out = []
        for m in ("coc", "wang", "sd"):
            pr = engine.price_contract(res.capped, a, po.limit, m)
            out.append(f"{m}: ROL {pr.rol:6.2%} x{pr.multiple:4.1f}")
        print(f"{p.id:14s} p_trig/event {an.p_trigger:6.3f}  P(attach) {an.p_attach:6.3%} "
              f"(1-in-{1/an.p_attach:5.1f})  LOL {res.capped.mean/po.limit:6.3%}  " + " | ".join(out))


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"PASS {name} ({time.time()-t:.1f}s)")
    report()
