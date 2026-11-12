# Parametric Insurance Pricing Lab

An interactive Streamlit demo of how **parametric (index-triggered) insurance** for rare, event-driven
aviation risks is structured and priced. It is self-contained, with no external APIs or data feeds.

## Products included (all parameters illustrative)

| Product | Index | Default payout |
|---|---|---|
| Airspace / hub closure | Days the airspace or hub is closed (NOTAMs) | Linear: USD 0.5m per day from day 3 to day 63 |
| Hurricane at hub airport | Max sustained wind within 50 nm (cat-in-a-circle) | Stepped by Saffir-Simpson category |
| Hub operational meltdown | % of departures cancelled over 72 h | Linear between 25% and 65% |
| Jet-fuel supply shock | % rise in 30-day average jet-fuel price | Linear between a 40% and a 100% rise |

The app is a single scrolling page. A sticky top bar holds the product picker, currency, an explanations
toggle, settings (Monte Carlo size, seed, reset), links to each section and a live price strip. Every
parameter can be edited, and edits are kept per product.

## Sections (top to bottom)

1. **Hazard**: frequency (Poisson, or Negative Binomial for rate uncertainty and clustering) and severity
   (GPD, lognormal, gamma, Weibull or beta), with return-period curves. An optional **calibration** panel
   fits all severity families by maximum likelihood, ranks them by AIC and gives a Jeffreys-posterior rate
   CV that carries parameter uncertainty into the price.
2. **Contract**: binary, linear or stepped payouts, an annual aggregate limit, and a stress-scenario table
   showing what the contract would pay.
3. **Pricing**: the exact annual payout distribution (FFT with mean-preserving discretisation), AEP and
   OEP curves, and risk-load methods (cost of capital on VaR or TVaR, Wang transform, standard
   deviation). Also expenses, fixed costs and a minimum rate on line, plus a validation table comparing
   the closed form, FFT and Monte Carlo results.
4. **Basis risk**: the insured's actual loss is modelled with a damage curve, lognormal scatter and
   events the index misses. Outputs include hedge effectiveness, missed and false payouts, and gross vs
   net EP curves.
5. **Sensitivity**: a tornado chart, a frequency × severity heatmap, and a trigger sweep showing the rate
   on line and the multiple of EL.
6. **Methodology and glossary**: the key formulas, plus every parameter and its effect on the price.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository, keeping `app.py` at the repository root.
2. Go to https://share.streamlit.io, choose **Create app**, then pick the repository and branch, and set
   the main file to `app.py`.
3. Deploy. `requirements.txt` and `.streamlit/config.toml` (the theme) are picked up automatically.

**Embedding on your website:** Streamlit Community Cloud apps can be embedded in an iframe by adding
`?embed=true` to the app URL, for example:

```html
<iframe src="https://YOUR-APP.streamlit.app/?embed=true" width="100%" height="900" style="border:0"></iframe>
```

## Project layout

```
app.py                     Streamlit UI
parametric/hazard.py       Frequency & severity models
parametric/contract.py     Payout structures (binary / linear / stepped, aggregate limit)
parametric/engine.py       FFT aggregate distribution, risk measures, pricing, Monte Carlo, basis risk, sensitivities
parametric/calibration.py  MLE fitting, AIC ranking, Jeffreys rate uncertainty
parametric/presets.py      The four aviation products
parametric/charts.py       Plotly figures
parametric/text.py         Explanations, tooltips, methodology, glossary
tests/                     Numerical validation and app smoke tests
```

## Tests

```bash
python tests/test_engine.py   # closed form vs FFT vs Monte Carlo, risk measures, calibration
python tests/test_app.py      # runs the app headless across products, payout types and methods
```

## Disclaimer

This is an educational demonstration. All parameters are illustrative and are not calibrated to any real
airport, airline, index provider or market. It is not a pricing tool for actual risk transfer.
