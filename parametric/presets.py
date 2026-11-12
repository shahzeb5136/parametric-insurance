"""Illustrative aviation-related parametric products.

All parameter values are illustrative and chosen to produce realistic-looking
behaviour; they are not calibrated to any real airport, airline or market.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    tagline: str
    peril: str
    index_name: str
    index_short: str
    unit: str
    index_definition: str
    data_source: str
    index_max: float
    index_step: float
    index_fmt: str
    # hazard
    lam: float
    cv: float
    family: str
    threshold: float
    p1: float
    p2: float
    upper: float | None
    # contract
    kind: str
    limit_m: float
    attachment: float
    exhaustion: float
    tiers: tuple
    agg: str
    # basis
    basis_max_m: float
    basis_onset: float
    basis_sat: float
    basis_power: float
    basis_noise: float
    basis_miss: float
    materiality_m: float
    loss_description: str
    # narrative
    flow: tuple            # five short strings for the overview flow
    stress: tuple          # ((scenario, index value), ...)
    catalogue_years: int
    insured: str


PRESETS = [
    Preset(
        id="airspace",
        name="Airspace / hub closure",
        tagline="Conflict, volcanic ash or security events that shut an airspace or hub airport.",
        peril="Closure of the covered airspace or hub airport to commercial traffic "
              "(geopolitical conflict, volcanic ash, security incident).",
        index_name="Duration of closure",
        index_short="closure",
        unit="days",
        index_definition="Continuous period during which the covered airspace / hub is closed to "
                         "commercial flights, measured in hours from official NOTAMs and expressed in days.",
        data_source="Official NOTAMs and air-navigation-service-provider records (illustrative).",
        index_max=365.0, index_step=0.5, index_fmt="%.1f",
        lam=0.12, cv=0.35, family="gpd", threshold=1.0, p1=3.0, p2=0.55, upper=None,
        kind="linear", limit_m=30.0, attachment=3.0, exhaustion=63.0,
        tiers=((3.0, 0.10), (7.0, 0.25), (14.0, 0.50), (30.0, 0.75), (60.0, 1.00)),
        agg="2× per-event limit",
        basis_max_m=120.0, basis_onset=0.0, basis_sat=120.0, basis_power=1.0,
        basis_noise=0.35, basis_miss=0.10, materiality_m=5.0,
        loss_description="lost contribution from cancelled flights, re-routing, crew and aircraft "
                         "repositioning",
        flow=("Conflict or ash cloud closes the airspace",
              "NOTAMs record the closure duration",
              "Closure lasts beyond 3 days?",
              "Fixed amount per extra day, capped",
              "Paid within ~30 days, no loss adjustment"),
        stress=(("Short regional closure", 2.0), ("Volcanic ash cloud (2010-type, ~6 days)", 6.0),
                ("Conflict escalation", 21.0), ("Prolonged closure", 45.0),
                ("Quarter-long shutdown", 90.0)),
        catalogue_years=60,
        insured="Airline or airport operator with exposure to the covered airspace/hub",
    ),
    Preset(
        id="hurricane",
        name="Hurricane at hub airport",
        tagline="Cat-in-a-circle cover: wind speed of a hurricane passing near the hub.",
        peril="Tropical cyclone (hurricane) passing close to the hub airport.",
        index_name="Max sustained wind within 50 nm",
        index_short="wind speed",
        unit="kt",
        index_definition="Highest 1-minute sustained wind speed of any hurricane whose track passes within "
                         "a 50-nautical-mile circle around the airport, from the official best-track record.",
        data_source="Best-track data from the regional meteorological agency, e.g. NOAA NHC HURDAT2 "
                    "(illustrative).",
        index_max=200.0, index_step=1.0, index_fmt="%.0f",
        lam=0.20, cv=0.25, family="gpd", threshold=64.0, p1=24.0, p2=-0.20, upper=None,
        kind="stepped", limit_m=50.0, attachment=96.0, exhaustion=137.0,
        tiers=((64.0, 0.05), (83.0, 0.15), (96.0, 0.30), (113.0, 0.60), (137.0, 1.00)),
        agg="1× per-event limit",
        basis_max_m=80.0, basis_onset=50.0, basis_sat=160.0, basis_power=2.0,
        basis_noise=0.60, basis_miss=0.10, materiality_m=5.0,
        loss_description="airport closure, cancelled rotations, ground-handling and property damage",
        flow=("A hurricane tracks near the hub",
              "Best-track record gives max wind in the circle",
              "Wind reaches a Saffir-Simpson tier?",
              "Tier sets the % of limit paid",
              "Paid within ~30 days of the official record"),
        stress=(("Category 1 brush", 70.0), ("Category 2", 90.0), ("Category 3 landfall", 105.0),
                ("Category 4 direct hit", 125.0), ("Category 5", 150.0)),
        catalogue_years=75,
        insured="Airline with a hurricane-exposed hub, or the airport operator",
    ),
    Preset(
        id="cancellations",
        name="Hub operational meltdown",
        tagline="Mass cancellations from ATC failures, IT outages, strikes or severe weather.",
        peril="System-wide disruption at the hub causing mass flight cancellations "
              "(ATC system failure, IT outage, strike, severe weather).",
        index_name="Departures cancelled over 72 h",
        index_short="cancellation rate",
        unit="%",
        index_definition="Share of scheduled departures from the hub that are cancelled within any "
                         "rolling 72-hour window, from an independent flight-status data provider.",
        data_source="Independent flight-status data, e.g. Cirium or OAG (illustrative).",
        index_max=100.0, index_step=1.0, index_fmt="%.0f",
        lam=0.60, cv=0.40, family="beta", threshold=10.0, p1=20.0, p2=5.0, upper=100.0,
        kind="linear", limit_m=15.0, attachment=25.0, exhaustion=65.0,
        tiers=((25.0, 0.25), (35.0, 0.50), (50.0, 0.75), (65.0, 1.00)),
        agg="2× per-event limit",
        basis_max_m=50.0, basis_onset=5.0, basis_sat=100.0, basis_power=1.0,
        basis_noise=0.30, basis_miss=0.05, materiality_m=10.0,
        loss_description="passenger care and compensation, re-accommodation and lost revenue",
        flow=("ATC failure, outage or strike hits the hub",
              "Flight-status data gives % cancelled in 72 h",
              "Cancellations above 25%?",
              "Pays pro rata up to 65%",
              "Paid within days from published data"),
        stress=(("Winter storm", 18.0), ("ATC system failure (one day)", 25.0),
                ("Nationwide IT outage", 40.0), ("Multi-day strike", 60.0), ("Total shutdown", 95.0)),
        catalogue_years=20,
        insured="Hub carrier or airport operator",
    ),
    Preset(
        id="fuel",
        name="Jet-fuel supply shock",
        tagline="An oil-supply disruption that sends jet-fuel prices sharply higher.",
        peril="Geopolitical or supply disruption that causes a sharp rise in jet-fuel prices.",
        index_name="Rise in 30-day avg. jet-fuel price",
        index_short="price spike",
        unit="%",
        index_definition="Percentage rise of the 30-day average regional jet-fuel price over its trailing "
                         "90-day average, measured after a supply-disruption event.",
        data_source="Published jet-fuel price assessments, e.g. Platts or Argus (illustrative).",
        index_max=500.0, index_step=1.0, index_fmt="%.0f",
        lam=0.25, cv=0.50, family="lognormal", threshold=15.0, p1=25.0, p2=0.80, upper=None,
        kind="linear", limit_m=25.0, attachment=40.0, exhaustion=100.0,
        tiers=((40.0, 0.25), (60.0, 0.50), (80.0, 0.75), (100.0, 1.00)),
        agg="1× per-event limit",
        basis_max_m=80.0, basis_onset=0.0, basis_sat=200.0, basis_power=1.0,
        basis_noise=0.25, basis_miss=0.05, materiality_m=15.0,
        loss_description="unhedged fuel cost above budget",
        flow=("Supply disruption hits oil markets",
              "Price assessments give the 30-day average rise",
              "Rise above 40%?",
              "Pays pro rata up to a 100% rise",
              "Paid after the averaging window closes"),
        stress=(("Moderate disruption", 30.0), ("Major producer outage", 55.0),
                ("Shipping chokepoint closure", 90.0), ("Extreme shock", 150.0)),
        catalogue_years=50,
        insured="Airline with unhedged fuel exposure",
    ),
]

PRESET_BY_ID = {p.id: p for p in PRESETS}

AGG_OPTIONS = {
    "1× per-event limit": 1.0,
    "2× per-event limit": 2.0,
    "3× per-event limit": 3.0,
    "5× per-event limit": 5.0,
    "Unlimited": None,
}

CURRENCIES = ["USD", "EUR", "GBP", "AED"]
