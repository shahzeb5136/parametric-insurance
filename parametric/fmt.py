"""Small formatting helpers."""
from __future__ import annotations

import numpy as np


def money(x: float, cur: str = "USD", dp: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    ax = abs(x)
    if ax >= 1e9:
        return f"{cur} {_trim(f'{x / 1e9:,.{dp}f}')}bn"
    if ax >= 1e6:
        return f"{cur} {_trim(f'{x / 1e6:,.{dp}f}')}m"
    if ax >= 1e3:
        return f"{cur} {x / 1e3:,.0f}k"
    return f"{cur} {x:,.0f}"


def _trim(s: str) -> str:
    return s.rstrip("0").rstrip(".") if "." in s else s


def pct(x: float, dp: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x * 100:.{dp}f}%"


def rp(T: float) -> str:
    """Return period as 1-in-T years."""
    if T is None or not np.isfinite(T) or T <= 0:
        return "never"
    if T < 10:
        return f"1-in-{T:.0f} yrs" if abs(T - round(T)) < 0.05 else f"1-in-{T:.1f} yrs"
    if T < 100_000:
        return f"1-in-{T:,.0f} yrs"
    return "> 1-in-100,000 yrs"


def num(x: float, fmt: str = "%.1f") -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return (fmt % x).rstrip("0").rstrip(".") if "." in (fmt % x) else fmt % x
