"""Dynamic optimisation (gOPT-lite) — M33."""
import math

import pytest
from scipy.optimize import minimize_scalar

from backend.engine import run_model
from conftest import has_msg, load_model


def _cb_analytic(T, tf=5.0):
    """Analytical CB(tf) for the parallel-reactions benchmark."""
    kb = 1.0e10 * math.exp(-60000.0 / (8.314 * T))
    kc = 1.0e11 * math.exp(-80000.0 / (8.314 * T))
    return kb / (kb + kc) * (1.0 - math.exp(-(kb + kc) * tf))


def test_dynamic_opt_finds_optimal_temperature(run):
    """The optimiser finds the finite optimal temperature; verified
    independently by a 1-D search on the ANALYTICAL solution."""
    r = run("parallel_reactions_opt.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    assert r["results"]["method"] == "DYNOPT-SLSQP"
    v = r["results"]["values"]
    T_opt_model = v["temp"]

    res = minimize_scalar(lambda T: -_cb_analytic(T), bounds=(300.0, 500.0),
                          method="bounded")
    T_opt_ref = res.x
    assert abs(T_opt_model - T_opt_ref) < 2.0, (
        f"model {T_opt_model:.1f} vs analytic {T_opt_ref:.1f}")
    # the optimum is strictly interior (not a bound)
    assert 300.0 < T_opt_model < 360.0
    # and the achieved objective matches the analytical value
    obj = r["results"]["objective"]["value"]
    assert abs(obj - _cb_analytic(T_opt_model)) < 1e-3


def test_dynamic_opt_requires_controls():
    """A dynamic OPTIMIZE without any control parameter is a clear
    error."""
    code = load_model("parallel_reactions_opt.gPROMS", "benchmarks")
    # give T a default -> no longer a control
    code = code.replace("Temp AS REAL               # CONTROL: no default!",
                        "Temp AS REAL DEFAULT 400.0")
    r = run_model(code)
    assert has_msg(r, "err_dynopt_no_controls", "error")
    assert r["results"] is None


def test_dynamic_opt_objective_is_end_state(run):
    """At the optimum, the reported objective equals the final CB of the
    corresponding simulation."""
    r = run("parallel_reactions_opt.gPROMS", folder="benchmarks")
    T_opt = r["results"]["values"]["temp"]
    code = load_model("parallel_reactions_opt.gPROMS", "benchmarks")
    code = code.replace("Temp AS REAL               # CONTROL: no default!",
                        f"Temp AS REAL DEFAULT {T_opt:.3f}")
    code = code.replace("  OPTIMIZE\n    MAXIMIZE CB ;\nEND", "END")
    r2 = run_model(code)
    assert r2["results"]["success"]
    cb_end = r2["timeSeries"]["variables"]["cb"][-1]
    assert abs(cb_end - r["results"]["objective"]["value"]) < 1e-6
