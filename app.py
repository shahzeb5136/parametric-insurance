"""Parametric Insurance Pricing Lab — rare, event-triggered aviation covers.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import html
import io

import numpy as np
import pandas as pd
import streamlit as st

from parametric import calibration, charts, engine
from parametric import text as T
from parametric.contract import KIND_LABELS, Payout
from parametric.fmt import money, pct, rp
from parametric.hazard import FAMILY_LABELS, Frequency, Severity
from parametric.presets import AGG_OPTIONS, CURRENCIES, PRESET_BY_ID, PRESETS, Preset

st.set_page_config(page_title="Parametric Insurance Pricing Lab", page_icon="✈️", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
header[data-testid="stHeader"] {display: none;}
.block-container {padding-top: 0 !important; padding-bottom: 4rem; max-width: 1400px;}
/* ---- sticky top bar ---- */
div:has(> .st-key-topbar) {position: sticky; top: 0; z-index: 990;}
.st-key-topbar {background: rgba(255,255,255,.97); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
                border-bottom: 1px solid #e6e5e0; padding: .75rem 0 .6rem 0; gap: .45rem;
                box-shadow: 0 8px 18px -18px rgba(0,0,0,.45);}
.st-key-topbar [data-testid="stMarkdownContainer"] {margin-bottom: 0 !important;}
.brand {font-weight: 700; font-size: .98rem; line-height: 1.15; color: #0b0b0b; letter-spacing: -.01em;}
.brand-mark {display: inline-block; color: #2a78d6; font-size: 1.05rem; margin-right: .1rem;}
.secnav {display: flex; flex-wrap: wrap; gap: .15rem 1.1rem; font-size: .9rem; padding: .1rem 0;}
.secnav a {color: #52514e !important; text-decoration: none !important; border-bottom: 2px solid transparent;
           padding-bottom: .1rem;}
.secnav a:hover {color: #2a78d6 !important; border-bottom-color: #2a78d6;}
.pricebar {display: flex; justify-content: flex-end; flex-wrap: wrap; gap: .4rem; padding: 2px 0;}
.pb-item {display: inline-flex; align-items: baseline; gap: .4rem; background: #f3f7fc; border: 1px solid #dde9f8;
          border-radius: 999px; padding: .18rem .7rem; font-size: .84rem; white-space: nowrap;}
.pb-l {color: #52514e;}
.pb-v {color: #0b0b0b; font-weight: 600; font-variant-numeric: tabular-nums;}
.pb-err {color: #d03b3b; font-size: .88rem;}
/* ---- page sections ---- */
.section-rule {scroll-margin-top: 112px; border-top: 1px solid #e6e5e0; margin: 2.6rem 0 .2rem 0;}
.section-rule.first {border-top: 0; margin-top: .6rem;}
.eyebrow {font-size: .78rem; font-weight: 700; color: #2a78d6; text-transform: uppercase; letter-spacing: .06em;
          margin-top: .6rem;}
.lede {font-size: 1.12rem; color: #52514e; margin: -.4rem 0 1rem 0;}
@media (max-width: 760px) {
  div:has(> .st-key-topbar) {position: static;}
  .st-key-topbar [data-testid="stElementContainer"]:has([data-testid="stSpace"]) {display: none;}
  .st-key-tb_row1 {gap: .6rem !important;}
  .st-key-tb_row2 {gap: .4rem !important;}
  .pricebar {justify-content: flex-start;}
  .section-rule {scroll-margin-top: 10px;}
}
.explain {background: #f3f7fc; border-left: 3px solid #2a78d6; padding: .65rem .95rem; border-radius: 4px;
          color: #2d2c2a; font-size: .93rem; line-height: 1.5; margin: .1rem 0 1rem 0;}
.explain p {margin-bottom: .35rem;}
.explain p:last-child {margin-bottom: 0;}
.flow-card {border: 1px solid #e6e5e0; border-radius: 8px; padding: .75rem .85rem; background: #fcfcfb;
            min-height: 112px;}
.flow-step {font-size: .72rem; color: #2a78d6; font-weight: 700; text-transform: uppercase;
            letter-spacing: .05em; margin-bottom: .3rem;}
.flow-text {font-size: .92rem; color: #2d2c2a; line-height: 1.35;}
.subtle {color: #6b6a65; font-size: .88rem;}
h1 {font-weight: 700; letter-spacing: -.01em;}
.kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .6rem;
            margin: .1rem 0 1rem 0;}
.kpi {border: 1px solid #e6e5e0; border-radius: 8px; padding: .6rem .8rem; background: #fcfcfb;}
.kpi-label {font-size: .8rem; color: #52514e; line-height: 1.25; margin-bottom: .25rem;}
.kpi-value {font-size: 1.4rem; font-weight: 600; color: #0b0b0b; line-height: 1.2;
            font-variant-numeric: tabular-nums;}
.kpi-sub {font-size: .78rem; color: #6b6a65; margin-top: .2rem;}
.kpi-i {color: #a3a29c; cursor: help; font-size: .8rem; margin-left: .15rem;}
.kpi-grid.compact {grid-template-columns: 1fr 1fr; gap: .4rem;}
.kpi-grid.compact .kpi {padding: .45rem .6rem; background: #ffffff;}
.kpi-grid.compact .kpi-value {font-size: 1.05rem;}
.chart-title {font-weight: 600; font-size: 1rem; color: #0b0b0b; margin: .4rem 0 -.2rem 0;}
@media (max-width: 760px) {.kpi-grid {grid-template-columns: 1fr 1fr !important;}}
table.terms {width: 100%; border-collapse: collapse; font-size: .92rem;}
table.terms th {text-align: left; width: 30%; color: #52514e; font-weight: 600; padding: .45rem .6rem .45rem 0;
                border-bottom: 1px solid #eeede9; vertical-align: top;}
table.terms td {padding: .45rem .6rem; border-bottom: 1px solid #eeede9; color: #1d1c1a;}
</style>
"""

UNIT_SINGULAR = {"days": "day", "%": "percentage point", "kt": "knot"}
PLOT_CONFIG = {"displaylogo": False,
               "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                                          "autoScale2d", "resetScale2d", "toggleSpikelines"]}

SEV_FIELDS = {   # label template, help key, min, max, step, format
    "gpd": (("Scale σ ({u})", "gpd_p1", 1e-3, None, None, "%.3f"),
            ("Tail shape ξ", "gpd_p2", -0.95, 1.5, 0.01, "%.3f")),
    "lognormal": (("Median index ({u})", "lognormal_p1", 0.0, None, None, "%.3f"),
                  ("Log-sd σ", "lognormal_p2", 0.01, 4.0, 0.01, "%.3f")),
    "gamma": (("Mean index ({u})", "gamma_p1", 0.0, None, None, "%.3f"),
              ("CV of excess", "gamma_p2", 0.01, 10.0, 0.01, "%.3f")),
    "weibull": (("Scale η of excess ({u})", "weibull_p1", 1e-3, None, None, "%.3f"),
                ("Shape k", "weibull_p2", 0.1, 10.0, 0.01, "%.3f")),
    "beta": (("Mean index ({u})", "beta_p1", 0.0, None, None, "%.3f"),
             ("Concentration κ", "beta_p2", 0.1, 1000.0, 0.1, "%.3f")),
}


# =========================================================================== #
# State helpers
# =========================================================================== #
def skey(pid: str, name: str) -> str:
    return f"{pid}::{name}"


class Ctx:
    pid: str = ""


def K(name: str) -> str:
    return skey(Ctx.pid, name)


def ss(name: str):
    return st.session_state[K(name)]


def _bounds(fam: str, which: int, p: Preset):
    _, _, lo, hi, _, _ = SEV_FIELDS[fam][which]
    return lo, (p.index_max if hi is None else hi)


@st.cache_data(show_spinner=False)
def equivalents(pid: str) -> dict:
    p = PRESET_BY_ID[pid]
    sev = Severity(p.family, p.threshold, p.p1, p.p2, p.upper)
    return calibration.equivalent_parameters(sev, p.upper if p.upper is not None else p.index_max)


def default_values(p: Preset) -> dict:
    a = engine.PricingAssumptions()
    d = dict(lam=p.lam, cv=p.cv, family=p.family, threshold=p.threshold,
             upper=float(p.upper if p.upper is not None else p.index_max),
             kind=p.kind, limit_m=p.limit_m, attachment=p.attachment, exhaustion=p.exhaustion,
             binary_trigger=p.attachment, agg=p.agg,
             basis_max_m=p.basis_max_m, basis_onset=p.basis_onset, basis_sat=p.basis_sat,
             basis_power=p.basis_power, basis_noise=p.basis_noise, basis_miss=p.basis_miss * 100,
             materiality_m=p.materiality_m,
             method=a.method, coc_rate=a.coc_rate * 100, coc_measure=a.coc_measure,
             coc_level=a.coc_level * 100, capital_share=a.capital_share * 100, wang_theta=a.wang_theta,
             sd_alpha=a.sd_alpha, expense_ratio=a.expense_ratio * 100, fixed_k=a.fixed_cost / 1e3,
             min_rol=a.min_rol * 100, shock=20, cat_years=int(p.catalogue_years))
    for fam, (v1, v2) in equivalents(p.id).items():
        lo1, hi1 = _bounds(fam, 0, p)
        lo2, hi2 = _bounds(fam, 1, p)
        d[f"{fam}_p1"] = float(np.clip(v1, lo1, hi1))
        d[f"{fam}_p2"] = float(np.clip(v2, lo2, hi2))
    return d


def default_frames(p: Preset) -> dict:
    sev = Severity(p.family, p.threshold, p.p1, p.p2, p.upper)
    yrs, x = calibration.synthetic_catalogue(p.lam, sev, p.catalogue_years, seed=17)
    return {
        "tiers_df": pd.DataFrame({"Threshold": [t for t, _ in p.tiers],
                                  "Payout %": [fr * 100 for _, fr in p.tiers]}),
        "stress_df": pd.DataFrame({"Scenario": [s for s, _ in p.stress],
                                   "Index value": [float(v) for _, v in p.stress]}),
        "cat_df": pd.DataFrame({"Year": yrs.astype(int), "Index": x.astype(float)}),
    }


def ensure_state(p: Preset):
    for name, v in default_values(p).items():
        st.session_state.setdefault(skey(p.id, name), v)
    for name, df in default_frames(p).items():
        if skey(p.id, name) not in st.session_state:
            st.session_state[skey(p.id, name)] = df


def reset_preset(pid: str):
    for key in [k for k in st.session_state.keys() if str(k).startswith(f"{pid}::")]:
        del st.session_state[key]


def apply_fit(pid: str, fam: str, p1: float, p2: float, lam: float, cv: float):
    p = PRESET_BY_ID[pid]
    s = st.session_state
    s[skey(pid, "family")] = fam
    lo1, hi1 = _bounds(fam, 0, p)
    lo2, hi2 = _bounds(fam, 1, p)
    s[skey(pid, f"{fam}_p1")] = float(np.clip(p1, lo1, hi1))
    s[skey(pid, f"{fam}_p2")] = float(np.clip(p2, lo2, hi2))
    s[skey(pid, "lam")] = float(np.clip(lam, 0.001, 20.0))
    s[skey(pid, "cv")] = float(np.clip(round(cv, 3), 0.0, 2.0))
    s["_flash"] = (f"Applied: {FAMILY_LABELS[fam]} severity, λ = {lam:.3f}/yr, rate CV = {cv:.2f}. "
                   "The whole model now uses these parameters.")


# =========================================================================== #
# Cached computations
# =========================================================================== #
@st.cache_data(show_spinner=False, max_entries=64)
def c_annual(freq, sev, payout):
    return engine.annual_distribution(freq, sev, payout)


@st.cache_data(show_spinner=False, max_entries=8)
def c_sample(freq, sev, n, seed):
    return engine.simulate_hazard(freq, sev, n, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def c_tornado(freq, sev, payout, a, shock, unit):
    return engine.tornado(freq, sev, payout, a, shock, unit)


@st.cache_data(show_spinner=False, max_entries=32)
def c_sweep(freq, sev, payout, a, levels):
    return engine.trigger_sweep(freq, sev, payout, a, np.array(levels))


@st.cache_data(show_spinner=False, max_entries=32)
def c_grid(freq, sev, payout, a, lam_mult, sev_mult):
    return engine.frequency_severity_grid(freq, sev, payout, a, lam_mult, sev_mult)


@st.cache_data(show_spinner=False, max_entries=32)
def c_fit(x: tuple, u: float, upper):
    return calibration.fit_all(np.array(x), u, upper)


# =========================================================================== #
# UI helpers
# =========================================================================== #
def explain(md: str, force: bool = False):
    if force or st.session_state.get("show_help", True):
        st.markdown(f'<div class="explain">\n\n{md}\n\n</div>', unsafe_allow_html=True)


def show(fig, caption: str | None = None):
    title = fig.layout.title.text if fig.layout.title is not None else None
    if title:
        st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
        fig.update_layout(title=None, margin=dict(t=12))
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG)
    if caption:
        st.caption(caption)


def kpis(items, compact: bool = False, cols: int | None = None):
    """Render KPI tiles: items are (label, value, sub, help) tuples; sub/help may be None."""
    cells = []
    for label, value, sub, hlp in items:
        tip = f' title="{html.escape(hlp, quote=True)}"' if hlp else ""
        icon = "<span class='kpi-i'>ⓘ</span>" if hlp else ""
        sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
        cells.append(f"<div class='kpi'{tip}><div class='kpi-label'>{label}{icon}</div>"
                     f"<div class='kpi-value'>{value}</div>{sub_html}</div>")
    cls = "kpi-grid compact" if compact else "kpi-grid"
    style = f' style="grid-template-columns: repeat({cols}, minmax(0, 1fr));"' if cols else ""
    st.markdown(f"<div class='{cls}'{style}>{''.join(cells)}</div>", unsafe_allow_html=True)


def table_height(n_rows: int) -> int:
    return 35 * (n_rows + 1) + 3


def ix(x: float, p: Preset) -> str:
    """Format an index value with its unit."""
    if x is None or not np.isfinite(x):
        return "–"
    s = (p.index_fmt % x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if p.unit == "days" and s == "1":
        return "1 day"
    return f"{s}%" if p.unit == "%" else f"{s} {p.unit}"


def payout_text(po: Payout, p: Preset, cur: str, short: bool = False) -> str:
    L = money(po.limit, cur)
    if po.kind == "binary":
        return f"Pays {L} once the {p.index_short} reaches {ix(po.attachment, p)}" + ("" if short else " (all-or-nothing).")
    if po.kind == "linear":
        per = po.limit / (po.exhaustion - po.attachment)
        unit = UNIT_SINGULAR.get(p.unit, p.unit)
        return (f"{money(per, cur)} per {unit} above {ix(po.attachment, p)}, "
                f"up to {L} at {ix(po.exhaustion, p)}" + ("" if short else "."))
    tiers = " · ".join(f"≥ {ix(t, p)}: {fr:.0%}" for t, fr in po.tiers)
    return f"Tiered share of {L}: " + tiers


# =========================================================================== #
# Top bar (sticky) and page sections
# =========================================================================== #
PRODUCT_LABELS = {
    "airspace": ":material/flight: Airspace closure",
    "hurricane": ":material/cyclone: Hurricane at hub",
    "cancellations": ":material/event_busy: Hub meltdown",
    "fuel": ":material/local_gas_station: Jet-fuel shock",
}
SECTIONS = [("overview", "Overview"), ("hazard", "Hazard"), ("contract", "Contract"), ("pricing", "Pricing"),
            ("basis-risk", "Basis risk"), ("sensitivity", "Sensitivity"), ("methodology", "Methodology")]


def topbar():
    with st.container(key="topbar"):
        with st.container(horizontal=True, vertical_alignment="center", gap="medium", wrap=True, key="tb_row1"):
            st.markdown("<div class='brand'><span class='brand-mark'>✈</span> Parametric<br>Pricing Lab</div>",
                        unsafe_allow_html=True, width="content")
            pid = st.segmented_control("Product", list(PRODUCT_LABELS), format_func=PRODUCT_LABELS.get,
                                       key="preset_id", required=True, label_visibility="collapsed", wrap=True,
                                       width="content")
            st.space("stretch")
            cur = st.selectbox("Currency", CURRENCIES, key="currency", label_visibility="collapsed", width=96,
                               help="Display currency (labels only; no conversion).")
            st.toggle("Explanations", value=True, key="show_help", width="content",
                      help="Hide the explanatory call-outs for a cleaner presentation view.")
            with st.popover("Settings", icon=":material/tune:", width="content"):
                st.select_slider("Monte Carlo years", options=[50_000, 100_000, 200_000, 500_000, 1_000_000],
                                 value=200_000, key="n_sims", format_func=lambda v: f"{v:,}",
                                 help=T.HELP["n_sims"])
                st.number_input("Random seed", min_value=0, max_value=100_000, value=2026, step=1, key="seed",
                                help=T.HELP["seed"])
                st.button("Reset this product to defaults", on_click=reset_preset, args=(pid,), width="stretch",
                          icon=":material/restart_alt:")
                st.caption("Edits are kept per product while you switch between products.")
        with st.container(horizontal=True, vertical_alignment="center", gap="medium", wrap=True, key="tb_row2"):
            links = "".join(f"<a href='#{a}' target='_self'>{t}</a>" for a, t in SECTIONS)
            st.markdown(f"<nav class='secnav'>{links}</nav>", unsafe_allow_html=True, width="content")
            st.space("stretch")
            price_slot = st.empty()
    return PRESET_BY_ID[pid], cur, price_slot


def price_strip(slot, price, an, cur):
    items = [("Gross premium", money(price.gross, cur)), ("Rate on line", pct(price.rol)),
             ("Expected loss", money(price.expected_loss, cur)),
             ("Pays", rp(1 / an.p_attach) if an.p_attach > 0 else "never")]
    html_items = "".join(f"<span class='pb-item'><span class='pb-l'>{k}</span><span class='pb-v'>{v}</span></span>"
                         for k, v in items)
    slot.markdown(f"<div class='pricebar'>{html_items}</div>", unsafe_allow_html=True)


def section(anchor: str, title: str, intro: str | None = None):
    st.markdown(f"<div class='section-rule' id='{anchor}'></div>", unsafe_allow_html=True)
    st.header(title, anchor=False)
    if intro:
        explain(intro)


# =========================================================================== #
# Input sections
# =========================================================================== #
def hazard_inputs(p: Preset):
    st.markdown("##### Frequency")
    lam = st.number_input("Qualifying events per year (λ)", min_value=0.001, max_value=20.0, step=0.01,
                          format="%.3f", key=K("lam"), help=T.HELP["lam"])
    st.caption(f"One qualifying event every **{1 / lam:,.1f} years** on average.")
    cv = st.number_input("Rate uncertainty (CV of λ)", min_value=0.0, max_value=2.0, step=0.05, format="%.2f",
                         key=K("cv"), help=T.HELP["cv"])
    st.caption("Poisson counts" if cv == 0 else f"Negative Binomial counts (variance = {lam + (lam * cv) ** 2:.3f})")

    st.markdown("##### Severity: index value given an event")
    fam = st.selectbox("Distribution family", list(FAMILY_LABELS), format_func=FAMILY_LABELS.get,
                       key=K("family"), help=T.HELP["family"])
    u = st.number_input(f"Event threshold u ({p.unit})", min_value=0.0, max_value=p.index_max,
                        step=p.index_step, format=p.index_fmt, key=K("threshold"), help=T.HELP["threshold"])
    upper = None
    if fam == "beta":
        upper = st.number_input(f"Upper bound ({p.unit})", min_value=0.0, max_value=p.index_max,
                                step=p.index_step, format=p.index_fmt, key=K("upper"), help=T.HELP["upper"])
    vals = []
    for i, (label, hkey, lo, hi, step, fmt) in enumerate(SEV_FIELDS[fam]):
        vals.append(st.number_input(label.format(u=p.unit), min_value=float(lo),
                                    max_value=float(p.index_max if hi is None else hi),
                                    step=float(step or p.index_step), format=fmt, key=K(f"{fam}_p{i + 1}"),
                                    help=T.HELP[hkey]))
    return Frequency(float(lam), float(cv)), Severity(fam, float(u), float(vals[0]), float(vals[1]),
                                                      None if upper is None else float(upper))


def contract_inputs(p: Preset, cur: str):
    kind = st.radio("Payout structure", list(KIND_LABELS), format_func=KIND_LABELS.get, key=K("kind"),
                    help=T.HELP["kind"])
    limit = st.number_input(f"Per-event limit ({cur} m)", min_value=0.1, max_value=5000.0, step=1.0,
                            format="%.2f", key=K("limit_m"), help=T.HELP["limit"]) * 1e6
    agg = AGG_OPTIONS[st.selectbox("Annual aggregate limit", list(AGG_OPTIONS), key=K("agg"), help=T.HELP["agg"])]
    notes = []
    kw = dict(min_value=0.0, max_value=p.index_max, step=p.index_step, format=p.index_fmt)
    if kind == "binary":
        t = st.number_input(f"Trigger ({p.unit})", key=K("binary_trigger"), help=T.HELP["trigger"], **kw)
        po = Payout("binary", limit, float(t), agg_multiple=agg)
    elif kind == "linear":
        a = st.number_input(f"Attachment ({p.unit})", key=K("attachment"), help=T.HELP["attachment"], **kw)
        e = st.number_input(f"Exhaustion ({p.unit})", key=K("exhaustion"), help=T.HELP["exhaustion"], **kw)
        po = Payout("linear", limit, float(a), float(e), agg_multiple=agg)
        if e > a:
            st.caption(f"= **{money(limit / (e - a), cur)} per {UNIT_SINGULAR.get(p.unit, p.unit)}** "
                       f"between attachment and exhaustion.")
    else:
        st.markdown(f"<span class='subtle'>Tier schedule ({p.unit} → % of limit)</span>", unsafe_allow_html=True)
        df = st.data_editor(
            ss("tiers_df"), key=K("tiers_editor"), num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "Threshold": st.column_config.NumberColumn(f"≥ {p.unit}", min_value=0.0, width="small",
                                                           max_value=p.index_max, step=p.index_step,
                                                           format=p.index_fmt, required=True),
                "Payout %": st.column_config.NumberColumn("% paid", min_value=0.0, max_value=100.0,
                                                          width="small", step=1.0, format="%.0f",
                                                          required=True)})
        st.caption(T.HELP["tiers"])
        rows = [(r["Threshold"], r["Payout %"] / 100) for _, r in df.iterrows()
                if pd.notna(r["Threshold"]) and pd.notna(r["Payout %"])]
        po, notes = Payout.make_stepped(limit, rows, agg)
    for n in notes:
        st.warning(n)
    return po


def pricing_inputs(cur: str) -> engine.PricingAssumptions:
    method = st.radio("Risk-load method", list(engine.METHOD_LABELS), format_func=engine.METHOD_LABELS.get,
                      key=K("method"), help=T.HELP["method"])
    st.markdown("##### Cost of capital")
    c1, c2 = st.columns(2)
    coc_rate = c1.number_input("Cost-of-capital rate (%)", 0.0, 50.0, step=0.5, format="%.1f", key=K("coc_rate"),
                               help=T.HELP["coc_rate"])
    share = c2.number_input("Capital share (%)", 1.0, 100.0, step=5.0, format="%.0f", key=K("capital_share"),
                            help=T.HELP["capital_share"])
    c3, c4 = st.columns(2)
    measure = c3.radio("Risk measure", ["TVaR", "VaR"], key=K("coc_measure"), horizontal=True,
                       help=T.HELP["coc_measure"])
    level = c4.select_slider("Confidence level (%)", options=[90.0, 95.0, 99.0, 99.5, 99.8, 99.9],
                             key=K("coc_level"), help=T.HELP["coc_level"])
    st.markdown("##### Other methods")
    c5, c6 = st.columns(2)
    theta = c5.number_input("Wang θ", 0.0, 2.0, step=0.05, format="%.2f", key=K("wang_theta"),
                            help=T.HELP["wang_theta"])
    alpha = c6.number_input("SD loading α", 0.0, 3.0, step=0.05, format="%.2f", key=K("sd_alpha"),
                            help=T.HELP["sd_alpha"])
    st.markdown("##### Costs")
    c7, c8 = st.columns(2)
    exp_ratio = c7.number_input("Expense ratio (% of gross)", 0.0, 60.0, step=1.0, format="%.1f",
                                key=K("expense_ratio"), help=T.HELP["expense_ratio"])
    fixed = c8.number_input(f"Fixed costs ({cur} k)", 0.0, 5000.0, step=5.0, format="%.0f", key=K("fixed_k"),
                            help=T.HELP["fixed_cost"])
    min_rol = st.number_input("Minimum rate on line (%)", 0.0, 50.0, step=0.25, format="%.2f", key=K("min_rol"),
                              help=T.HELP["min_rol"])
    return engine.PricingAssumptions(method=method, sd_alpha=float(alpha), wang_theta=float(theta),
                                     coc_rate=coc_rate / 100, coc_measure=measure, coc_level=level / 100,
                                     capital_share=share / 100, expense_ratio=exp_ratio / 100,
                                     fixed_cost=fixed * 1e3, min_rol=min_rol / 100)


def basis_inputs(p: Preset, cur: str):
    st.caption(f"Insured's loss here means **{p.loss_description}**.")
    st.markdown("##### Damage curve")
    m = st.number_input(f"Loss at saturation ({cur} m)", 0.1, 100000.0, step=5.0, format="%.1f",
                        key=K("basis_max_m"), help=T.HELP["basis_max"])
    c1, c2 = st.columns(2)
    kw = dict(min_value=0.0, max_value=p.index_max, step=p.index_step, format=p.index_fmt)
    x0 = c1.number_input(f"Loss onset ({p.unit})", key=K("basis_onset"), help=T.HELP["basis_onset"], **kw)
    x1 = c2.number_input(f"Saturation ({p.unit})", key=K("basis_sat"), help=T.HELP["basis_sat"], **kw)
    g = st.number_input("Curvature γ", 0.2, 5.0, step=0.1, format="%.1f", key=K("basis_power"),
                        help=T.HELP["basis_power"])
    st.markdown("##### Basis risk")
    noise = st.slider("Basis volatility σ_b", 0.0, 1.5, step=0.05, key=K("basis_noise"), help=T.HELP["basis_noise"])
    miss = st.slider("Loss events not captured by the index (%)", 0.0, 50.0, step=1.0, key=K("basis_miss"),
                     help=T.HELP["basis_miss"])
    mat = st.number_input(f"Materiality threshold ({cur} m)", 0.0, 100000.0, step=0.5, format="%.1f",
                          key=K("materiality_m"), help=T.HELP["materiality"])
    err = [] if x1 > x0 else ["Saturation must be above the loss onset."]
    return engine.BasisModel(m * 1e6, float(x0), float(x1), float(g), float(noise), miss / 100), mat * 1e6, err


# =========================================================================== #
# Output sections
# =========================================================================== #
def hazard_outputs(p, freq, sev, po):
    an_rp = lambda x: float(freq.return_period(sev.sf(x)))  # noqa: E731
    mean = sev.mean
    s100 = float(freq.event_prob_for_return_period(100.0))
    s10 = float(freq.event_prob_for_return_period(10.0))
    kpis([
        ("Qualifying event", rp(1 / freq.lam), None, "Average return period of an event above the threshold."),
        ("Mean index given an event", ix(mean, p) if np.isfinite(mean) else "infinite", None,
         "Average index value given that an event occurs. Infinite when the GPD tail shape ξ ≥ 1."),
        ("1-in-10-year level", ix(float(sev.isf(s10)), p) if s10 < 1 else "below threshold", None,
         "Annual-maximum index level exceeded on average once every 10 years."),
        ("1-in-100-year level", ix(float(sev.isf(s100)), p) if s100 < 1 else "below threshold", None,
         "Annual-maximum index level exceeded on average once every 100 years."),
        ("Physical maximum", ix(sev.support_max, p) if np.isfinite(sev.support_max) else "∞",
         None if np.isfinite(sev.support_max) else "unbounded tail",
         "Upper end of the index distribution (GPD with ξ < 0, or beta)."),
    ], cols=5)
    g1, g2 = st.columns(2)
    with g1:
        show(charts.severity_density(sev, po, p.unit), "Shaded: events that trigger a payout.")
    with g2:
        show(charts.return_levels(freq, sev, po, p.unit))
    n = np.arange(4)
    pm = freq.pmf(n)
    pois = Frequency(freq.lam).pmf(n)
    tbl = pd.DataFrame({
        "Events in a year": ["0", "1", "2", "3 or more"],
        "Probability (this model)": [pct(v, 3) for v in list(pm[:3]) + [1 - pm[:3].sum()]],
        "Probability (Poisson, same λ)": [pct(v, 3) for v in list(pois[:3]) + [1 - pois[:3].sum()]],
    })
    st.markdown("**Annual event count**: rate uncertainty makes multi-event years more likely")
    st.dataframe(tbl, hide_index=True, width="stretch")


def calibration_section(p: Preset, sev: Severity):
    with st.expander("Calibrate the hazard model from an event catalogue", expanded=False, icon=":material/query_stats:"):
        explain(T.CALIBRATION, force=True)
        c1, c2 = st.columns([1, 1.7], gap="large")
        with c1:
            up = st.file_uploader("Upload a CSV (column `index`, optional `year`)", type=["csv"], key=K("cat_upload"))
            years = st.number_input("Observation period (years)", min_value=1, max_value=1000, step=1,
                                    key=K("cat_years"))
            x = None
            if up is not None:
                try:
                    df_up = pd.read_csv(io.BytesIO(up.getvalue()))
                    col = next(c for c in df_up.columns if str(c).strip().lower() in ("index", "value", "x"))
                    x = pd.to_numeric(df_up[col], errors="coerce").dropna().to_numpy()
                    st.caption(f"Loaded {len(x)} values from `{col}`.")
                except StopIteration:
                    st.error("No column named `index` found in the uploaded file.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read the file: {exc}")
            if x is None:
                st.caption("Synthetic catalogue: edit, add or delete rows.")
                df = st.data_editor(ss("cat_df"), key=K("cat_editor"), num_rows="dynamic", hide_index=True,
                                    height=280, width="stretch",
                                    column_config={"Year": st.column_config.NumberColumn(format="%d"),
                                                   "Index": st.column_config.NumberColumn(f"Index ({p.unit})",
                                                                                          format="%.1f")})
                x = pd.to_numeric(df["Index"], errors="coerce").dropna().to_numpy()
        u = sev.threshold
        upper = sev.upper if sev.family == "beta" else (p.upper if p.upper is not None else p.index_max)
        xs = x[x >= u]
        dropped = len(x) - len(xs)
        ff = calibration.fit_frequency(len(xs), float(years))
        fits = c_fit(tuple(np.round(xs, 6)), u, upper)
        with c2:
            if dropped:
                st.caption(f"{dropped} value(s) below the event threshold u = {ix(u, p)} were ignored.")
            if len(xs) >= 3 and any(f.ok for f in fits):
                short = {"gpd": "GPD", "lognormal": "Lognormal", "gamma": "Gamma", "weibull": "Weibull",
                         "beta": "Beta"}
                show(charts.calibration_plot(xs, fits, u, upper, p.unit, short))
            else:
                st.info("At least 3 events above the threshold are needed to fit a severity distribution.")
        f1, f2 = st.columns([1, 1.7], gap="large")
        with f1:
            st.markdown("**Frequency**")
            st.dataframe(pd.DataFrame({
                "": ["Events above threshold", "Observation period", "λ (MLE)", "90% confidence interval",
                     "λ (Jeffreys posterior mean)", "Suggested rate CV"],
                "Value": [f"{ff.n}", f"{ff.years:g} years", f"{ff.lam_mle:.3f} / yr",
                          f"{ff.lam_lo:.3f} – {ff.lam_hi:.3f}", f"{ff.lam_post:.3f} / yr", f"{ff.cv_post:.2f}"]}),
                hide_index=True, width="stretch")
        with f2:
            st.markdown("**Severity fits (ranked by AIC)**")
            ok = [f for f in fits if f.ok]
            best = ok[0].aic if ok else np.nan
            st.dataframe(pd.DataFrame({
                "Family": [FAMILY_LABELS[f.family] for f in fits],
                "Parameter 1": [f"{f.p1:.3f}" if f.ok else "–" for f in fits],
                "Parameter 2": [f"{f.p2:.3f}" if f.ok else "–" for f in fits],
                "Log-lik.": [f"{f.loglik:.2f}" if f.ok else "–" for f in fits],
                "ΔAIC": [f"{f.aic - best:.2f}" if f.ok else (f.message or "failed") for f in fits]}),
                hide_index=True, width="stretch")
            if ok:
                cc1, cc2 = st.columns([1.6, 1])
                choice = cc1.selectbox("Family to apply", [f.family for f in ok], format_func=FAMILY_LABELS.get,
                                       key=K("fit_choice"))
                f = next(f for f in ok if f.family == choice)
                cc2.button("Apply to model", type="primary", width="stretch", on_click=apply_fit,
                           args=(Ctx.pid, f.family, f.p1, f.p2, ff.lam_post, ff.cv_post))
                st.caption("Applies the selected severity fit, the posterior-mean λ and the suggested rate CV. "
                           "ΔAIC below 2 means the data cannot really distinguish the families. That is "
                           "parameter uncertainty in its own right; compare their prices in the Sensitivity section.")


def contract_outputs(p, freq, sev, po, an, cur):
    kpis([
        ("Events that trigger a payout", pct(an.p_trigger, 1), None,
         "Probability that a qualifying event triggers a payout."),
        ("Annual payout probability", pct(an.p_attach, 2), None,
         "Probability of at least one payout in the year (attachment probability)."),
        ("Attachment return period", rp(1 / an.p_attach if an.p_attach > 0 else np.inf), None, None),
        ("Average payout when paid", money(an.mean_given_trigger, cur), None,
         "Expected payout given that an event triggers (before any aggregate cap)."),
    ], cols=4)
    show(charts.payout_function(po, sev, p.unit, cur))
    rows = [("Trigger (first payout)", po.trigger)]
    if np.isfinite(po.full_level) and po.full_level != po.trigger:
        rows.append(("Full per-event limit paid", po.full_level))
    st.dataframe(pd.DataFrame({
        "Key level": [r[0] for r in rows],
        "Index": [ix(r[1], p) for r in rows],
        "P(reached | event)": [pct(float(sev.sf(r[1])), 2) for r in rows],
        "Annual probability": [pct(float(freq.prob_any(sev.sf(r[1]))), 3) for r in rows],
        "Return period": [rp(float(freq.return_period(sev.sf(r[1])))) for r in rows]}),
        hide_index=True, width="stretch")


def stress_section(p, freq, sev, po, cur):
    st.markdown("#### Stress scenarios: what would this contract pay?")
    st.caption("Edit the scenarios on the left, or add rows, and the payouts update on the right. The return "
               "period is how often the index reaches at least that level in a year, under the current hazard "
               "model.")
    g1, g2 = st.columns([0.85, 1.4], gap="large")
    with g1:
        df = st.data_editor(ss("stress_df"), key=K("stress_editor"), num_rows="dynamic", hide_index=True,
                            width="stretch",
                            column_config={"Scenario": st.column_config.TextColumn("Scenario", width="medium"),
                                           "Index value": st.column_config.NumberColumn(
                                               f"Index ({p.unit})", format=p.index_fmt, width="small")})
    df = df.dropna(subset=["Index value"])
    v = df["Index value"].to_numpy(dtype=float)
    pay = po.payout(v)
    out = pd.DataFrame({
        "Scenario": df["Scenario"].fillna("").astype(str).to_numpy(),
        "Index": [ix(x, p) for x in v],
        "Payout": [money(x, cur) for x in pay],
        "% of limit": [pct(x / po.limit, 0) for x in pay],
        "Return period": [rp(float(freq.return_period(sev.sf(x)))) for x in v],
    })
    with g2:
        st.dataframe(out, hide_index=True, width="stretch")


def pricing_outputs(p, freq, sev, po, res, an, a, price, mc, cur, n_sims):
    d = res.capped
    kpis([
        ("Gross premium", money(price.gross, cur), engine.METHOD_LABELS[a.method],
         "Expected loss + risk load + fixed costs, grossed up for variable expenses."),
        ("Rate on line", pct(price.rol), None, "Gross premium ÷ per-event limit."),
        ("Expected loss", money(price.expected_loss, cur), f"{pct(price.lol)} of limit",
         "Expected annual payout after the aggregate limit (the pure premium)."),
        ("Risk load", money(price.risk_load, cur),
         f"{price.risk_load / price.expected_loss:.0%} of EL" if price.expected_loss > 0 else None,
         "Charge for bearing the volatility and tail of the payout."),
        ("Multiple of EL", f"{price.multiple:.2f}×" if np.isfinite(price.multiple) else "–", None,
         "Gross premium ÷ expected loss."),
        ("Expected loss ratio", pct(price.loss_ratio, 1), None, "Expected loss ÷ gross premium."),
    ], cols=3)
    if price.floor_uplift > 0:
        st.info(f"The minimum rate on line binds: the technical price is {money(price.gross - price.floor_uplift, cur)}, "
                f"so the premium is lifted by {money(price.floor_uplift, cur)}.", icon=":material/info:")
    if a.method == "coc" and a.coc_measure == "VaR" and price.capital == 0 and an.p_attach > 0:
        st.warning(f"VaR {a.coc_level:.1%} is zero because the payout probability ({pct(an.p_attach)}) is below "
                   f"{pct(1 - a.coc_level, 1)}, so this contract attracts **no capital charge**. This is the "
                   "classic blind spot of VaR for rare-event covers. Switch to TVaR to see the difference.",
                   icon=":material/warning:")
    g1, g2 = st.columns(2)
    with g1:
        show(charts.premium_waterfall(price, cur))
    with g2:
        show(charts.ep_curves(d, freq, sev, po, cur),
             "AEP: total paid in the year. OEP: largest single payout. The curves coincide where "
             "multi-event years are negligible.")


def pricing_detail(p, freq, sev, po, res, an, a, price, mc, cur, n_sims):
    d = res.capped
    t1, t2 = st.columns([1.3, 1], gap="large")
    with t1:
        st.markdown("#### Risk-load methods compared")
        st.caption("Same loss distribution and current parameters; ● marks the method used for the headline "
                   "price.")
        rows = []
        for m in engine.METHOD_LABELS:
            pr = engine.price_contract(d, a, po.limit, m)
            rows.append({"Method": ("● " if m == a.method else "") + engine.METHOD_LABELS[m],
                         "Risk load": money(pr.risk_load, cur), "Gross premium": money(pr.gross, cur),
                         "ROL": pct(pr.rol), "× EL": f"{pr.multiple:.2f}" if np.isfinite(pr.multiple) else "–",
                         "Loss ratio": pct(pr.loss_ratio, 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if a.method == "coc":
            st.caption(f"Stand-alone capital = {a.coc_measure} {a.coc_level:.1%} − EL = "
                       f"{money(price.risk_measure - price.expected_loss, cur)}; allocated "
                       f"{a.capital_share:.0%} → {money(price.capital, cur)} at {a.coc_rate:.1%}.")
    with t2:
        st.markdown("#### Annual payout distribution")
        agg = po.agg_limit
        stats_rows = [
            ("Expected annual payout", money(d.mean, cur)),
            ("Standard deviation", money(d.sd, cur)),
            ("Coefficient of variation", f"{d.sd / d.mean:.2f}" if d.mean > 0 else "–"),
            ("P(any payout in the year)", pct(an.p_attach, 3)),
            ("P(an event pays the full limit)", pct(an.p_full_annual, 3)),
            ("P(annual aggregate exhausted)", pct(d.prob_at_least(agg), 3) if agg else "no aggregate limit"),
            ("VaR 99% / 99.5%", f"{money(d.quantile(0.99), cur)} / {money(d.quantile(0.995), cur)}"),
            ("TVaR 99% / 99.5%", f"{money(d.tvar(0.99), cur)} / {money(d.tvar(0.995), cur)}"),
        ]
        st.dataframe(pd.DataFrame(stats_rows, columns=["Measure", "Value"]), hide_index=True, width="stretch",
                     height=table_height(len(stats_rows)))

    with st.expander("Validation: closed form vs exact FFT vs Monte Carlo", icon=":material/fact_check:"):
        mcu, mcc = mc
        st.caption(f"Monte Carlo with {n_sims:,} simulated years (seed {st.session_state.seed}). The ± range is a "
                   "95% interval for simulation error. The FFT result is the one used for pricing.")
        z = 1.96
        val = pd.DataFrame([
            ("Expected payout, no aggregate cap", money(an.el_uncapped, cur), money(res.uncapped.mean, cur),
             f"{money(mcu.mean, cur)} ± {money(z * mcu.se, cur)}"),
            ("Standard deviation, no aggregate cap", money(an.sd_uncapped, cur), money(res.uncapped.sd, cur),
             money(mcu.sd, cur)),
            ("P(any payout)", pct(an.p_attach, 3), "–", f"{pct(mcc.p_attach, 3)} ± {pct(z * mcc.p_attach_se, 3)}"),
            ("Expected payout, with aggregate cap", "–", money(d.mean, cur),
             f"{money(mcc.mean, cur)} ± {money(z * mcc.se, cur)}"),
            ("VaR 99%, with cap", "–", money(d.quantile(0.99), cur), money(mcc.var99, cur)),
            ("TVaR 99%, with cap", "–", money(d.tvar(0.99), cur), money(mcc.tvar99, cur)),
        ], columns=["Quantity", "Closed form", "FFT (exact, lattice)", "Monte Carlo"])
        st.dataframe(val, hide_index=True, width="stretch")
        st.caption(f"FFT lattice: {res.n_grid:,} steps per event limit (step {money(res.capped.h, cur)}), "
                   "mean-preserving discretisation, tail mass truncated below 1e-12.")

    summary = pd.DataFrame([
        ("Product", p.name), ("Payout", payout_text(po, p, cur)), ("Per-event limit", po.limit),
        ("Annual aggregate limit", po.agg_limit if po.agg_limit else "unlimited"),
        ("Lambda", freq.lam), ("Rate CV", freq.cv), ("Severity", FAMILY_LABELS[sev.family]),
        ("Threshold", sev.threshold), ("Severity p1", sev.p1), ("Severity p2", sev.p2),
        ("Risk-load method", engine.METHOD_LABELS[a.method]), ("Expected loss", price.expected_loss),
        ("Risk load", price.risk_load), ("Fixed costs", price.fixed_cost),
        ("Variable expenses", price.variable_expenses), ("Min-ROL uplift", price.floor_uplift),
        ("Gross premium", price.gross), ("Rate on line", price.rol), ("Multiple of EL", price.multiple),
        ("Annual payout probability", an.p_attach)], columns=["Item", "Value"])
    st.download_button("Download pricing summary (CSV)", summary.to_csv(index=False).encode(),
                       file_name=f"parametric_pricing_{p.id}.csv", mime="text/csv", icon=":material/download:")


def basis_outputs(p, sev, po, basis, bres, bm, price, cur):
    red = 1 - bm.net_tvar99 / bm.gross_tvar99 if bm.gross_tvar99 > 0 else 0
    kpis([
        ("Hedge effectiveness", pct(bm.hedge_effectiveness, 1), None,
         "Share of the variance of the insured's annual loss removed by the cover: "
         "1 − Var(loss − payout) / Var(loss)."),
        ("Correlation (annual)", f"{bm.corr:.2f}", f"Spearman {bm.spearman:.2f}",
         "Correlation between annual loss and annual payout."),
        ("Recovery ratio", pct(bm.recovery_ratio, 1), None, "Expected annual payout ÷ expected annual loss."),
        ("Material losses missed", pct(bm.missed_share, 1), "false negatives",
         "Share of events with a material loss that produced no payout."),
        ("Payouts without material loss", pct(bm.false_share, 1), "false positives",
         "Share of paying events where the actual loss was below materiality."),
        ("Tail reduction (TVaR 99%)", pct(red, 1),
         f"{money(bm.gross_tvar99, cur)} → {money(bm.net_tvar99, cur)}",
         "TVaR 99% of annual loss without cover vs with cover (net of payouts, including the premium)."),
    ], cols=3)
    g1, g2 = st.columns(2)
    with g1:
        show(charts.damage_vs_payout(basis, po, sev, p.unit, cur))
    with g2:
        show(charts.net_ep(bres.annual_loss, bres.annual_loss - bres.annual_payout + price.gross, cur))
    show(charts.basis_scatter(bres, basis_materiality(), cur))
    st.caption(f"Events are simulated from the same hazard sample as the Monte Carlo check. The plot shows a random "
               f"sample of up to 3,000 events. Expected annual cost of the cover (premium − expected payout): "
               f"{money(price.gross - bres.annual_payout.mean(), cur)}. Expected net annual loss with cover: "
               f"{money(bm.net_mean, cur)} vs {money(bm.gross_mean, cur)} without.")


def basis_materiality() -> float:
    return ss("materiality_m") * 1e6


def sensitivity_section(p, freq, sev, po, a, price, cur):
    shock = st.slider("Shock size (±, relative)", 5, 50, step=5, key=K("shock"), format="%d%%",
                      help=T.HELP["shock"]) / 100
    rows = c_tornado(freq, sev, po, a, shock, p.unit)
    g1, g2 = st.columns([1.35, 1], gap="large")
    with g1:
        show(charts.tornado(rows, price.gross, shock, cur))
    with g2:
        lam_mult = (0.5, 0.75, 1.0, 1.5, 2.0)
        sev_mult = (0.75, 0.9, 1.0, 1.1, 1.25)
        grid = c_grid(freq, sev, po, a, lam_mult, sev_mult)
        label = {"gpd": "σ", "weibull": "η"}.get(sev.family, "scale")
        show(charts.heatmap(grid, lam_mult, sev_mult, label),
             "Outlined cell = current assumptions. Each row scales the severity scale parameter; each "
             "column scales λ.")
    hi = float(sev.ppf(0.995)) if np.isfinite(sev.ppf(0.995)) else p.index_max
    hi = min(max(hi, po.trigger * 1.3 if np.isfinite(po.trigger) else hi), p.index_max)
    if np.isfinite(sev.support_max):
        hi = min(hi, sev.support_max - 0.02 * (sev.support_max - sev.threshold))
    levels = tuple(np.round(np.linspace(sev.threshold, hi, 31), 4))
    sweep = c_sweep(freq, sev, po, a, levels)
    f1, f2 = charts.trigger_sweep(sweep, po.trigger, p.unit)
    s1, s2 = st.columns(2)
    with s1:
        show(f1)
    with s2:
        show(f2)
    st.caption("The trigger sweep moves the whole payout schedule up or down while keeping its shape. As the "
               "trigger rises, the expected loss falls faster than the premium, because the tail risk load and "
               "fixed costs do not shrink in proportion. Remote covers therefore carry high multiples of EL.")


# =========================================================================== #
# Overview & methodology
# =========================================================================== #
def overview_header(p):
    st.markdown("<div class='eyebrow'>Parametric insurance · rare-event pricing</div>", unsafe_allow_html=True)
    st.title(f"{p.name}", anchor=False)
    st.markdown(f"<div class='lede'>{p.tagline}</div>", unsafe_allow_html=True)


def overview(p, freq, sev, po, an, price, cur):
    overview_header(p)
    explain(T.INTRO, force=True)

    steps = [("1 · Event", p.flow[0]), ("2 · Index", p.flow[1]),
             ("3 · Trigger", f"Index reaches {ix(po.trigger, p)}?"),
             ("4 · Payout", payout_text(po, p, cur, short=True)), ("5 · Settlement", p.flow[4])]
    cols = st.columns(5)
    for col, (h, txt) in zip(cols, steps):
        col.markdown(f"<div class='flow-card'><div class='flow-step'>{h}</div>"
                     f"<div class='flow-text'>{txt}</div></div>", unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.subheader("Indicative term sheet")
        agg = money(po.agg_limit, cur) if po.agg_limit else "Unlimited"
        terms = [
            ("Insured", p.insured),
            ("Covered event", p.peril),
            ("Index", f"<b>{p.index_name}</b>: {p.index_definition}"),
            ("Index source / calculation agent", p.data_source),
            ("Qualifying event", f"Index ≥ {ix(sev.threshold, p)}"),
            ("Payout", payout_text(po, p, cur)),
            ("Per-event limit", money(po.limit, cur)),
            ("Annual aggregate limit", agg),
            ("Risk period", "12 months"),
            ("Settlement", "Within ~30 days of the index being published; no loss adjustment"),
            ("Indicative gross premium", f"<b>{money(price.gross, cur)}</b> (rate on line {pct(price.rol)})"),
        ]
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in terms)
        st.markdown(f"<table class='terms'>{rows}</table>", unsafe_allow_html=True)
    with right:
        st.subheader("Key numbers")
        kpis([
            ("Gross premium", money(price.gross, cur), engine.METHOD_LABELS[price.method], None),
            ("Rate on line", pct(price.rol), None, "Gross premium ÷ per-event limit."),
            ("Expected loss", money(price.expected_loss, cur), f"{pct(price.lol)} of limit", None),
            ("Multiple of EL", f"{price.multiple:.2f}×" if np.isfinite(price.multiple) else "–", None, None),
            ("Payout probability / yr", pct(an.p_attach), None, None),
            ("Attachment return period", rp(1 / an.p_attach) if an.p_attach > 0 else "never", None, None),
        ], cols=2)
        st.caption("Trigger, payout and premium update live as you change the sections below.")
    st.subheader("Parametric vs. traditional indemnity cover")
    st.dataframe(pd.DataFrame(T.COMPARISON, columns=["", "Parametric", "Indemnity"]), hide_index=True,
                 width="stretch")
    st.subheader("How this page works")
    st.markdown(
        "Scroll down through the sections, or jump to one with the links in the bar at the top. The price in the "
        "top bar updates live as you change any parameter.\n\n"
        "1. **Hazard**: set how often events happen and how extreme they get, or calibrate to a catalogue.\n"
        "2. **Contract**: design the trigger and payout schedule, and test it against stress scenarios.\n"
        "3. **Pricing**: choose a risk-load method and costs to build the premium from expected loss.\n"
        "4. **Basis risk**: see how well the payout tracks the insured's real loss.\n"
        "5. **Sensitivity**: find the assumptions that move the price, and see how the price varies with "
        "the trigger.\n\n"
        "Hover over the ⓘ or ? next to any parameter for an explanation of what it does.")


def methodology():
    st.markdown(T.METHODOLOGY)
    st.subheader("Parameter glossary")
    st.dataframe(pd.DataFrame(T.GLOSSARY, columns=["Parameter", "What it represents", "Effect when increased"]),
                 hide_index=True, width="stretch", height=table_height(len(T.GLOSSARY)))


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    if st.session_state.get("preset_id") not in PRESET_BY_ID:
        st.session_state.preset_id = PRESETS[0].id
    ensure_state(PRESET_BY_ID[st.session_state.preset_id])
    p, cur, price_slot = topbar()
    Ctx.pid = p.id
    ensure_state(p)

    # ---------------- page skeleton + inputs (top to bottom) ----------------
    st.markdown("<div class='section-rule first' id='overview'></div>", unsafe_allow_html=True)
    ov = st.container()

    section("hazard", "① Hazard model", T.HAZARD)
    if "_flash" in st.session_state:
        st.success(st.session_state.pop("_flash"), icon=":material/check_circle:")
    hz_in, hz_out = st.columns([1, 2.6], gap="large")
    with hz_in:
        freq, sev = hazard_inputs(p)
    hz_err = st.container()
    hz_cal = st.container()

    section("contract", "② Contract design", T.CONTRACT)
    ct_in, ct_out = st.columns([1, 2.6], gap="large")
    with ct_in:
        po = contract_inputs(p, cur)
    ct_err = st.container()
    ct_more = st.container()

    section("pricing", "③ Pricing", T.PRICING)
    pr_in, pr_out = st.columns([1, 2.6], gap="large")
    with pr_in:
        a = pricing_inputs(cur)
    pr_more = st.container()

    section("basis-risk", "④ Basis risk", T.BASIS)
    bs_in, bs_out = st.columns([1, 2.6], gap="large")
    with bs_in:
        basis, materiality, basis_err = basis_inputs(p, cur)

    section("sensitivity", "⑤ Sensitivity", T.SENSITIVITY)
    sens = st.container()

    section("methodology", "Methodology & glossary")
    methodology()
    st.caption("All parameters are illustrative and not calibrated to any real airport, airline or market. "
               "For demonstration only.")

    # ---------------- validation ----------------
    errs_h, errs_c = sev.validate(), po.validate()
    if errs_h or errs_c:
        for box, errs in ((hz_err, errs_h), (ct_err, errs_c)):
            with box:
                for e in errs:
                    st.error(e, icon=":material/error:")
        price_slot.markdown("<div class='pricebar'><span class='pb-err'>Fix the highlighted parameters to see "
                            "a price.</span></div>", unsafe_allow_html=True)
        with ov:
            overview_header(p)
            st.error("Some parameters are invalid. Scroll down to the highlighted section to fix them.")
        return

    # ---------------- computation ----------------
    res = c_annual(freq, sev, po)
    an = engine.analytic(freq, sev, po)
    price = engine.price_contract(res.capped, a, po.limit)
    n_sims, seed = int(st.session_state.n_sims), int(st.session_state.seed)
    sample = c_sample(freq, sev, n_sims, seed)
    _, unc, cap = engine.annual_payouts(sample, po)
    mc = (engine.MCStats.of(unc), engine.MCStats.of(cap))

    # ---------------- outputs ----------------
    price_strip(price_slot, price, an, cur)
    with ov:
        overview(p, freq, sev, po, an, price, cur)
    with hz_out:
        hazard_outputs(p, freq, sev, po)
    with hz_cal:
        calibration_section(p, sev)
    with ct_out:
        contract_outputs(p, freq, sev, po, an, cur)
    with ct_more:
        stress_section(p, freq, sev, po, cur)
    with pr_out:
        pricing_outputs(p, freq, sev, po, res, an, a, price, mc, cur, n_sims)
    with pr_more:
        pricing_detail(p, freq, sev, po, res, an, a, price, mc, cur, n_sims)
    with bs_out:
        if basis_err:
            for e in basis_err:
                st.error(e)
        else:
            bres = engine.simulate_basis(sample, sev, po, basis, seed)
            bm = engine.basis_metrics(bres, price.gross, materiality)
            basis_outputs(p, sev, po, basis, bres, bm, price, cur)
    with sens:
        sensitivity_section(p, freq, sev, po, a, price, cur)


main()
