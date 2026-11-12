"""Plotly figures. One visual system: thin marks, recessive grid, fixed palette order."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .contract import Payout
from .engine import AnnualDistribution, BasisModel, BasisResult
from .hazard import Frequency, Severity

# Validated categorical palette (light mode), fixed order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_DARK = "#184f95"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8d8c86", "#e6e5e0"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
NEUTRAL = "#b9b8b2"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
RP_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _layout(fig: go.Figure, title: str | None = None, x: str | None = None, y: str | None = None,
            height: int = 340, legend: bool = True, hover: str = "x unified") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98) if title else None,
        font=dict(family=FONT, size=13, color=INK2),
        height=height,
        margin=dict(l=8, r=12, t=58 if title else 28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode=hover,
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK), bordercolor=GRID),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title=x, showgrid=False, linecolor=GRID, linewidth=1, ticks="outside",
                     tickcolor=GRID, zeroline=False, title_font=dict(size=12))
    fig.update_yaxes(title=y, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID,
                     title_font=dict(size=12))
    return fig


def _vline(fig, x, label, color=MUTED, position="top", y=None):
    if x is None or not np.isfinite(x):
        return
    fig.add_vline(x=x, line=dict(color=color, width=1))
    fig.add_annotation(x=x, y=y if y is not None else (1.0 if position == "top" else 0.0), yref="paper", text=label,
                       showarrow=False, xanchor="left", yanchor="top" if position == "top" else "bottom",
                       xshift=4, font=dict(size=11, color=INK2), align="left")


def _hline(fig, y, label, color=MUTED, side="left"):
    if y is None or not np.isfinite(y):
        return
    fig.add_hline(y=y, line=dict(color=color, width=1))
    fig.add_annotation(x=0 if side == "left" else 1, xref="paper", y=y, text=label, showarrow=False,
                       xanchor=side, yanchor="bottom", xshift=4 if side == "left" else -4,
                       font=dict(size=11, color=INK2))


def _label_side(curve: np.ndarray, y: float, span: float) -> str:
    """Pick the end of a rising curve where a label sitting just above the line y is clear of it."""
    n = max(1, int(0.45 * len(curve)))
    left, right = curve[:n], curve[-n:]
    overlap = lambda seg: int(np.sum(np.isfinite(seg) & (seg >= y) & (seg <= y + 0.14 * span)))  # noqa: E731
    return "left" if overlap(left) <= overlap(right) else "right"


def index_range(sev: Severity, payout: Payout | None = None, q: float = 0.995) -> tuple[float, float]:
    lo = sev.threshold
    hi = float(sev.ppf(q))
    ref = float(sev.ppf(0.95))
    if payout is not None:
        ref = max([ref] + [t for t in payout.thresholds() if np.isfinite(t)])
    hi = min(hi, lo + 2.0 * (ref - lo)) if np.isfinite(ref) and ref > lo else hi
    if payout is not None:
        for t in payout.thresholds():
            if np.isfinite(t):
                hi = max(hi, t * 1.08 if t > 0 else t + 1)
    hi = min(hi, sev.support_max) if np.isfinite(sev.support_max) else hi
    if payout is not None and np.isfinite(payout.full_level):
        hi = max(hi, payout.full_level + 0.05 * (payout.full_level - lo + 1))
    return lo, hi


# --------------------------------------------------------------------------- #
# Hazard
# --------------------------------------------------------------------------- #
def severity_density(sev: Severity, payout: Payout, unit: str) -> go.Figure:
    lo, hi = index_range(sev, payout)
    x = np.linspace(lo, hi, 600)
    x[0] = lo + 1e-6 * (hi - lo)
    pdf = sev.pdf(x)
    ymax = float(np.nanpercentile(pdf, 98.5)) * 1.25
    fig = go.Figure()
    trig = payout.trigger
    m = x >= trig
    if m.any():
        fig.add_trace(go.Scatter(x=x[m], y=pdf[m], fill="tozeroy", fillcolor=_rgba(BLUE, 0.16),
                                 line=dict(width=0), hoverinfo="skip", name="Events that trigger a payout"))
    fig.add_trace(go.Scatter(x=x, y=pdf, line=dict(color=BLUE, width=2), name="Probability density",
                             hovertemplate="%{x:.1f} " + unit + "<br>density %{y:.4f}<extra></extra>"))
    _vline(fig, trig, f"Trigger {trig:g} {unit}")
    if np.isfinite(payout.full_level) and payout.full_level != trig:
        close = (payout.full_level - trig) < 0.25 * (hi - lo)
        _vline(fig, payout.full_level, f"Full payout {payout.full_level:g} {unit}", y=0.9 if close else None)
    _layout(fig, "Index value given an event occurs", f"Index ({unit})", "Density", legend=False, hover="closest")
    fig.update_yaxes(range=[0, ymax], showticklabels=False)
    return fig


def return_levels(freq: Frequency, sev: Severity, payout: Payout, unit: str) -> go.Figure:
    T = np.logspace(np.log10(1.02), 3, 300)
    s = freq.event_prob_for_return_period(T)
    lvl = np.where(s < 1, sev.isf(np.minimum(s, 1.0)), np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=lvl, line=dict(color=BLUE, width=2), name="Annual-maximum index",
                             hovertemplate="1-in-%{x:,.0f} years: %{y:.1f} " + unit + "<extra></extra>"))
    trig = payout.trigger
    finite = lvl[np.isfinite(lvl)]
    span = float(finite.max() - min(finite.min(), 0)) if finite.size else 1.0
    _hline(fig, trig, f"Trigger {trig:g} {unit}  ·  1-in-{float(freq.return_period(sev.sf(trig))):,.0f} yrs",
           side=_label_side(lvl, trig, span))
    fl = payout.full_level
    if np.isfinite(fl) and fl != trig:
        _hline(fig, fl, f"Full payout {fl:g} {unit}  ·  1-in-{float(freq.return_period(sev.sf(fl))):,.0f} yrs",
               side=_label_side(lvl, fl, span))
    _layout(fig, "Return period of the annual-maximum index", "Return period (years, log scale)",
            f"Index ({unit})", legend=False, hover="closest")
    fig.update_xaxes(type="log", tickvals=RP_TICKS, ticktext=[str(t) for t in RP_TICKS])
    return fig


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def payout_function(payout: Payout, sev: Severity, unit: str, cur: str) -> go.Figure:
    lo, hi = index_range(sev, payout)
    lo = min(lo, payout.trigger - 0.05 * (hi - lo)) if np.isfinite(payout.trigger) else lo
    x = np.linspace(lo, hi, 1200)
    if payout.kind != "linear":
        x = np.unique(np.concatenate([x, np.array(payout.thresholds())]))
    y = payout.payout(x) / 1e6
    fig = go.Figure(go.Scatter(
        x=x, y=y, line=dict(color=BLUE, width=2, shape="hv" if payout.kind != "linear" else "linear"),
        fill="tozeroy", fillcolor=_rgba(BLUE, 0.10), name="Payout",
        hovertemplate="%{x:.1f} " + unit + " → " + cur + " %{y:.2f}m<extra></extra>"))
    _vline(fig, payout.trigger, "Trigger")
    _layout(fig, "Payout per event", f"Index ({unit})", f"Payout ({cur} m)", legend=False, hover="closest")
    fig.update_yaxes(rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def ep_curves(dist: AnnualDistribution, freq: Frequency, sev: Severity, payout: Payout,
              cur: str) -> go.Figure:
    T = np.logspace(0.0, 3.0, 400)
    aep = np.array([dist.quantile(1 - 1 / t) for t in T]) / 1e6
    y = np.linspace(0, payout.max_payout, 801)
    oep_exc = freq.prob_any(payout.sf_payout(y, sev))      # decreasing in y
    oep = []
    for t in T:
        k = np.searchsorted(-oep_exc, -1.0 / t, side="left")   # first y with exc <= 1/t
        oep.append(y[min(k, len(y) - 1)] / 1e6)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=aep, name="Annual total (AEP)", line=dict(color=BLUE, width=5, shape="hv"),
                             hovertemplate="AEP %{y:.2f}m<extra></extra>"))
    fig.add_trace(go.Scatter(x=T, y=oep, name="Largest event (OEP)",
                             line=dict(color=ORANGE, width=2, shape="hv"),
                             hovertemplate="OEP %{y:.2f}m<extra></extra>"))
    _layout(fig, "Exceedance-probability curves", "Return period (years, log scale)", f"Annual payout ({cur} m)")
    fig.update_xaxes(type="log", tickvals=RP_TICKS, ticktext=[str(t) for t in RP_TICKS])
    fig.update_layout(hovermode="x unified")
    fig.update_traces(xhoverformat=",.0f")
    return fig


def premium_waterfall(price, cur: str) -> go.Figure:
    items = [("Expected loss", price.expected_loss), ("Risk load", price.risk_load),
             ("Fixed costs", price.fixed_cost), ("Expenses & brokerage", price.variable_expenses)]
    if price.floor_uplift > 0:
        items.append(("Min-ROL uplift", price.floor_uplift))
    labels = [i[0].replace("Expenses & brokerage", "Expenses") for i in items] + ["Gross premium"]
    vals = [i[1] / 1e6 for i in items] + [0]
    fig = go.Figure(go.Waterfall(
        y=labels, x=vals, orientation="h", measure=["relative"] * len(items) + ["total"],
        text=[f"{v:.3f}m" if v < 1 else f"{v:.2f}m" for v in vals[:-1]] + [f"{price.gross / 1e6:.3f}m"],
        textposition="outside", textfont=dict(size=12, color=INK2), cliponaxis=False,
        increasing=dict(marker=dict(color=BLUE)), totals=dict(marker=dict(color=BLUE_DARK)),
        connector=dict(line=dict(color=GRID, width=1)),
        hovertemplate="%{y}: " + cur + " %{x:.3f}m<extra></extra>"))
    _layout(fig, "How the premium is built", f"{cur} m", None, legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, price.gross / 1e6 * 1.3])
    return fig


# --------------------------------------------------------------------------- #
# Basis risk
# --------------------------------------------------------------------------- #
def damage_vs_payout(basis: BasisModel, payout: Payout, sev: Severity, unit: str, cur: str) -> go.Figure:
    lo, hi = index_range(sev, payout)
    hi = max(hi, min(basis.saturation, hi * 1.5))
    x = np.linspace(lo, hi, 800)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=basis.expected_loss(x) / 1e6, name="Expected loss",
                             line=dict(color=ORANGE, width=2),
                             hovertemplate="Expected loss " + cur + " %{y:.2f}m<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=payout.payout(x) / 1e6, name="Payout",
                             line=dict(color=BLUE, width=2, shape="hv" if payout.kind != "linear" else "linear"),
                             hovertemplate="Payout " + cur + " %{y:.2f}m<extra></extra>"))
    _layout(fig, "Expected loss vs payout by index level", f"Index ({unit})", f"{cur} m")
    fig.update_traces(xhoverformat=".1f")
    return fig


def basis_scatter(res: BasisResult, materiality: float, cur: str, n: int = 3000, seed: int = 5) -> go.Figure:
    rng = np.random.default_rng(seed)
    keep = (res.ev_loss > 0) | (res.ev_payout > 0)
    idx = np.nonzero(keep)[0]
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    L, P, cap = res.ev_loss[idx] / 1e6, res.ev_payout[idx] / 1e6, res.ev_captured[idx]
    mat = L >= materiality / 1e6
    paid = P > 0
    groups = [
        ("Paid, material loss", paid & mat, GOOD),
        ("Missed: no payout", ~paid & mat, CRITICAL),
        ("Paid, loss immaterial", paid & ~mat, SERIOUS),
        ("No payout, minor loss", ~paid & ~mat, NEUTRAL),
    ]
    fig = go.Figure()
    for name, m, c in groups:
        if not m.any():
            continue
        txt = np.where(cap[m], "captured by index", "not captured by index")
        fig.add_trace(go.Scattergl(
            x=L[m], y=P[m], mode="markers", name=name, text=txt,
            marker=dict(color=c, size=6, opacity=0.75, line=dict(color="white", width=0.5)),
            hovertemplate="Loss " + cur + " %{x:.2f}m<br>Payout " + cur + " %{y:.2f}m<br>%{text}<extra></extra>"))
    x_hi = float(np.quantile(L, 0.995)) * 1.1 if len(L) else 1.0
    x_hi = max(x_hi, materiality / 1e6 * 1.5, float(P.max(initial=0)) * 1.2, 1e-3)
    y_hi = max(float(P.max(initial=0)) * 1.15, 1e-3)
    top = max(x_hi, y_hi)
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=MUTED, width=1),
                             name="Payout = loss", hoverinfo="skip"))
    _vline(fig, materiality / 1e6, "Materiality", position="top")
    _layout(fig, "Each event: actual loss vs parametric payout", f"Insured's actual loss ({cur} m)",
            f"Payout ({cur} m)", height=420, hover="closest")
    fig.update_xaxes(range=[-0.02 * x_hi, x_hi])
    fig.update_yaxes(range=[-0.04 * y_hi, y_hi])
    return fig


def net_ep(gross: np.ndarray, net: np.ndarray, cur: str) -> go.Figure:
    T = np.logspace(np.log10(1.25), 3.0, 200)
    q = 1 - 1 / T
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=np.quantile(gross, q) / 1e6, name="Without cover",
                             line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=T, y=np.quantile(net, q) / 1e6, name="With cover (net, incl. premium)",
                             line=dict(color=BLUE, width=2)))
    _layout(fig, "Insured's annual loss by return period", "Return period (years, log scale)",
            f"Annual loss ({cur} m)")
    fig.update_xaxes(type="log", tickvals=RP_TICKS, ticktext=[str(t) for t in RP_TICKS])
    fig.update_traces(xhoverformat=",.0f", hovertemplate="%{y:.2f}m")
    return fig


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #
def tornado(rows: list[dict], base: float, shock: float, cur: str) -> go.Figure:
    rows = rows[::-1]
    names = [r["parameter"] for r in rows]
    lo = np.array([r["low"] for r in rows]) / 1e6
    hi = np.array([r["high"] for r in rows]) / 1e6
    b = base / 1e6
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=lo - b, base=b, orientation="h", name=f"Parameter −{shock:.0%}",
                         marker=dict(color=BLUE, cornerradius=4),
                         customdata=lo, hovertemplate="%{y}: " + cur + " %{customdata:.3f}m<extra>−</extra>"))
    fig.add_trace(go.Bar(y=names, x=hi - b, base=b, orientation="h", name=f"Parameter +{shock:.0%}",
                         marker=dict(color=ORANGE, cornerradius=4),
                         customdata=hi, hovertemplate="%{y}: " + cur + " %{customdata:.3f}m<extra>+</extra>"))
    fig.add_vline(x=b, line=dict(color=INK2, width=1))
    _layout(fig, f"Gross premium sensitivity (base {cur} {b:.3f}m)", f"Gross premium ({cur} m)", None,
            height=380, hover="closest")
    fig.update_layout(barmode="overlay", bargap=0.35)
    return fig


def trigger_sweep(rows: list[dict], current: float, unit: str) -> tuple[go.Figure, go.Figure]:
    x = np.array([r["trigger"] for r in rows])
    rol = np.array([r["rol"] for r in rows]) * 100
    lol = np.array([r["lol"] for r in rows]) * 100
    mult = np.array([r["multiple"] for r in rows])
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=x, y=rol, name="Gross rate on line", line=dict(color=BLUE, width=2)))
    f1.add_trace(go.Scatter(x=x, y=lol, name="Expected loss on line", line=dict(color=ORANGE, width=2)))
    _vline(f1, current, "Current trigger")
    _layout(f1, "Price vs trigger level", f"Trigger ({unit})", "% of per-event limit")
    f1.update_traces(hovertemplate="%{y:.2f}%", xhoverformat=".1f")
    f2 = go.Figure(go.Scatter(x=x, y=mult, line=dict(color=BLUE, width=2), name="Multiple",
                              hovertemplate="%{y:.2f}× EL<extra></extra>"))
    _vline(f2, current, "Current trigger")
    _layout(f2, "Premium as a multiple of expected loss", f"Trigger ({unit})", "Gross premium ÷ EL",
            legend=False)
    f2.update_traces(xhoverformat=".1f")
    return f1, f2


def heatmap(grid: np.ndarray, lam_mult, sev_mult, sev_label: str) -> go.Figure:
    z = grid * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"λ × {m:g}" for m in lam_mult], y=[f"{sev_label} × {m:g}" for m in sev_mult],
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        text=[[f"{v:.2f}%" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=12), xgap=2, ygap=2, showscale=False,
        hovertemplate="%{y}, %{x}: ROL %{z:.2f}%<extra></extra>"))
    try:
        i0, j0 = list(sev_mult).index(1.0), list(lam_mult).index(1.0)
        fig.add_shape(type="rect", x0=j0 - 0.5, x1=j0 + 0.5, y0=i0 - 0.5, y1=i0 + 0.5,
                      line=dict(color=INK, width=2))
    except ValueError:
        pass
    _layout(fig, "Rate on line under frequency × severity uncertainty", None, None, height=340,
            legend=False, hover="closest")
    fig.update_xaxes(side="bottom", showline=False, ticks="")
    fig.update_yaxes(showgrid=False, ticks="")
    return fig


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_plot(x: np.ndarray, fits, u: float, upper, unit: str, labels: dict) -> go.Figure:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    emp = 1 - np.arange(1, n + 1) / (n + 1)          # Weibull plotting positions
    fig = go.Figure()
    grid = np.linspace(u, max(x.max() * 1.3, u + 1), 400)
    colors = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4"]
    for f, c in zip([f for f in fits if f.ok], colors):
        s = f.severity(u, upper).sf(grid)
        fig.add_trace(go.Scatter(x=grid, y=np.where(s > 0, s, np.nan), name=labels.get(f.family, f.family),
                                 line=dict(color=c, width=2),
                                 hovertemplate="%{x:.1f} " + unit + ": %{y:.3f}<extra>" + f.family + "</extra>"))
    fig.add_trace(go.Scatter(x=x, y=emp, mode="markers", name="Observed events",
                             marker=dict(color=INK, size=8, line=dict(color="white", width=1.5)),
                             hovertemplate="%{x:.1f} " + unit + ": %{y:.3f}<extra>observed</extra>"))
    _layout(fig, "Fit check: P(index > x | event)", f"Index ({unit})", "Exceedance probability (log)",
            height=380, hover="closest")
    fig.update_yaxes(type="log", dtick=1, exponentformat="none")
    return fig
