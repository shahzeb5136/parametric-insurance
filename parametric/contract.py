"""Payout structures for a parametric cover.

A payout function f maps the index value X of a single event to a cash amount in
[0, limit].  Three standard shapes are supported:

* binary   – all-or-nothing: pays the full limit once X >= trigger.
* linear   – pays pro rata between an attachment and an exhaustion level
             (an option-spread / per-unit indemnity, e.g. "USD 0.5m per day").
* stepped  – a schedule of tiers, e.g. Saffir-Simpson categories.

An annual aggregate limit (a multiple of the per-event limit, or unlimited) caps
the sum of payouts within the 12-month risk period.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate

from .hazard import Severity

KIND_LABELS = {
    "binary": "Binary (all-or-nothing)",
    "linear": "Linear (pro-rata between attachment and exhaustion)",
    "stepped": "Stepped (tiered schedule)",
}


@dataclass(frozen=True)
class Payout:
    kind: str
    limit: float                        # per-event limit (currency units)
    attachment: float = 0.0             # binary trigger / linear attachment
    exhaustion: float = 0.0             # linear only
    tiers: tuple = ()                   # stepped: ((threshold, fraction), ...) ascending
    agg_multiple: float | None = 1.0    # annual aggregate limit / per-event limit (None = unlimited)

    # ------------------------------------------------------------------ #
    @staticmethod
    def make_stepped(limit, tiers, agg_multiple) -> tuple["Payout", list[str]]:
        """Sort tiers, clip fractions to [0,1] and enforce a non-decreasing schedule."""
        notes = []
        clean = sorted((float(t), float(fr)) for t, fr in tiers
                       if t is not None and fr is not None and np.isfinite(t) and np.isfinite(fr))
        out, running = [], 0.0
        for t, fr in clean:
            fr = min(max(fr, 0.0), 1.0)
            if fr < running:
                notes.append(f"Tier at {t:g} lowered below the previous tier; raised to {running:.0%}.")
                fr = running
            running = fr
            out.append((t, fr))
        return Payout("stepped", limit, tiers=tuple(out), agg_multiple=agg_multiple), notes

    # ------------------------------------------------------------------ #
    @property
    def agg_limit(self) -> float | None:
        return None if self.agg_multiple is None else self.agg_multiple * self.limit

    @property
    def max_fraction(self) -> float:
        if self.kind == "stepped":
            return max((fr for _, fr in self.tiers), default=0.0)
        return 1.0

    @property
    def max_payout(self) -> float:
        return self.limit * self.max_fraction

    @property
    def trigger(self) -> float:
        """Lowest index value producing a positive payout."""
        if self.kind == "stepped":
            for t, fr in self.tiers:
                if fr > 0:
                    return t
            return np.inf
        return self.attachment

    @property
    def full_level(self) -> float:
        """Lowest index value producing the maximum payout."""
        if self.kind == "binary":
            return self.attachment
        if self.kind == "linear":
            return self.exhaustion
        mf = self.max_fraction
        for t, fr in self.tiers:
            if fr >= mf and fr > 0:
                return t
        return np.inf

    def thresholds(self) -> list[float]:
        if self.kind == "binary":
            return [self.attachment]
        if self.kind == "linear":
            return [self.attachment, self.exhaustion]
        return [t for t, _ in self.tiers]

    def validate(self) -> list[str]:
        errs = []
        if self.limit <= 0:
            errs.append("Per-event limit must be positive.")
        if self.kind == "linear" and self.exhaustion <= self.attachment:
            errs.append("Exhaustion must be above the attachment.")
        if self.kind == "stepped" and self.max_fraction <= 0:
            errs.append("At least one tier must pay a positive share of the limit.")
        return errs

    # ------------------------------------------------------------------ #
    def payout(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        L = self.limit
        if self.kind == "binary":
            return np.where(x >= self.attachment, L, 0.0)
        if self.kind == "linear":
            a, e = self.attachment, self.exhaustion
            return L * np.clip((x - a) / (e - a), 0.0, 1.0)
        th = np.array([t for t, _ in self.tiers])
        fr = np.array([f for _, f in self.tiers])
        if th.size == 0:
            return np.zeros_like(x)
        idx = np.searchsorted(th, x, side="right") - 1
        return np.where(idx >= 0, fr[np.clip(idx, 0, None)], 0.0) * L

    def sf_payout(self, y, sev: Severity) -> np.ndarray:
        """P(f(X) > y) for the per-event payout, vectorised in y."""
        y = np.asarray(y, dtype=float)
        L = self.limit
        out = np.zeros_like(y)
        neg = y < 0
        out[neg] = 1.0
        if self.kind == "binary":
            m = (~neg) & (y < L)
            out[m] = sev.sf(self.attachment)
        elif self.kind == "linear":
            a, e = self.attachment, self.exhaustion
            m = (~neg) & (y < L)
            out[m] = sev.sf(a + y[m] * (e - a) / L)
        else:
            th = np.array([t for t, _ in self.tiers])
            lv = np.array([f for _, f in self.tiers]) * L
            sf_th = np.append(sev.sf(th), 0.0)
            j = np.searchsorted(lv, y, side="right")      # first tier paying more than y
            m = ~neg
            out[m] = sf_th[np.minimum(j[m], len(th))]
        return out

    def moments(self, sev: Severity) -> tuple[float, float]:
        """Exact E[f(X)] and E[f(X)^2] for a single qualifying event."""
        L = self.limit
        if self.kind == "binary":
            s = float(sev.sf(self.attachment))
            return L * s, L * L * s
        if self.kind == "stepped":
            th = np.array([t for t, _ in self.tiers])
            lv = np.array([f for _, f in self.tiers]) * L
            sf_th = np.append(sev.sf(th), 0.0)
            prob = sf_th[:-1] - sf_th[1:]
            return float(np.sum(lv * prob)), float(np.sum(lv ** 2 * prob))
        # linear: E[g(X)] = integral_a^e g'(x) S(x) dx with g(x) = c (x - a)
        a, e = self.attachment, self.exhaustion
        hi = min(e, sev.support_max)
        if hi <= a:
            return 0.0, 0.0
        c = L / (e - a)
        kw = dict(limit=400, epsabs=0.0, epsrel=1e-10)
        i1, _ = integrate.quad(lambda x: float(sev.sf(x)), a, hi, **kw)
        i2, _ = integrate.quad(lambda x: (x - a) * float(sev.sf(x)), a, hi, **kw)
        return c * i1, 2.0 * c * c * i2

    # ------------------------------------------------------------------ #
    def map_thresholds(self, fn) -> "Payout":
        """Return a copy with every threshold transformed by fn (used for sensitivities)."""
        if self.kind == "stepped":
            return Payout("stepped", self.limit, tiers=tuple((fn(t), fr) for t, fr in self.tiers),
                          agg_multiple=self.agg_multiple)
        return Payout(self.kind, self.limit, fn(self.attachment),
                      fn(self.exhaustion) if self.kind == "linear" else self.exhaustion,
                      agg_multiple=self.agg_multiple)
