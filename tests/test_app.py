"""Smoke tests of the Streamlit app across products, payout structures and methods.

Run with:  python tests/test_app.py
"""
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def run(at):
    at.run(timeout=120)
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_all_presets_and_structures():
    at = run(AppTest.from_file(APP, default_timeout=120))
    for pid in ("airspace", "hurricane", "cancellations", "fuel"):
        at.segmented_control(key="preset_id").set_value(pid)
        run(at)
        for kind in ("binary", "linear", "stepped"):
            at.radio(key=f"{pid}::kind").set_value(kind)
            run(at)
            for method in ("coc", "wang", "sd"):
                at.radio(key=f"{pid}::method").set_value(method)
                run(at)
        for fam in ("gpd", "lognormal", "gamma", "weibull", "beta"):
            at.selectbox(key=f"{pid}::family").set_value(fam)
            run(at)
        at.radio(key=f"{pid}::coc_measure").set_value("VaR")
        run(at)
        print("ok", pid, "errors shown:", [e.value for e in at.error])


def test_reset_and_apply_fit():
    at = run(AppTest.from_file(APP, default_timeout=120))
    at.number_input(key="airspace::lam").set_value(0.5)
    run(at)
    apply = next(b for b in at.button if b.label == "Apply to model")
    apply.click()
    run(at)
    lam = at.number_input(key="airspace::lam").value
    assert lam != 0.5 and 0.01 < lam < 1.0, lam
    assert any("Applied" in s.value for s in at.success)
    for b in at.button:
        if b.label == "Reset this product to defaults":
            b.click()
            run(at)
            break
    assert abs(at.number_input(key="airspace::lam").value - 0.12) < 1e-9, at.number_input(key="airspace::lam").value


def test_invalid_inputs_show_errors():
    at = run(AppTest.from_file(APP, default_timeout=120))
    at.number_input(key="airspace::exhaustion").set_value(2.0)
    run(at)
    assert any("Exhaustion" in e.value for e in at.error)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"PASS {name} ({time.time()-t:.1f}s)")
