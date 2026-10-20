"""Heat exchanger (UA/LMTD) library unit + PIECEWISE — M41."""
import math

import numpy as np
import pytest
from scipy.optimize import root

from backend.engine import run_model
from conftest import load_model


def test_heat_exchanger_ua_library(run):
    """The UA/LMTD exchanger matches an independent scipy solve of the
    same balances."""
    r = run("heat_exchanger_ua.gPROMS", folder="library")
    assert r["results"]["success"]
    v = r["results"]["values"]

    U, A = 500.0, 20.0
    mh, mc, cph, cpc = 2.0, 3.0, 4200.0, 4180.0
    Thi, Tci = 80.0, 15.0

    def eqs(u):
        q, tho, tco, dt1, dt2, lmtd = u
        return np.array([
            q - mh * cph * (Thi - tho),
            q - mc * cpc * (tco - Tci),
            dt1 - (Thi - tco),
            dt2 - (tho - Tci),
            lmtd - (dt1 - dt2) / math.log(dt1 / dt2),
            q - U * A * lmtd,
        ])

    sol = root(eqs, [3e5, 60, 20, 50, 45, 30])
    assert sol.success
    ref = dict(zip(["q", "tho", "tco", "dt1", "dt2", "lmtd"], sol.x))
    for k, val in ref.items():
        assert abs(v[k] - val) < 1e-6
    # physics: hot cools, cold warms, duty positive
    assert v["tho"] < 80.0 and v["tco"] > 15.0 and v["q"] > 0
    # no temperature cross in counter-current design
    assert v["dt1"] > 0 and v["dt2"] > 0


def test_heat_exchanger_ua_duty_scales_with_area():
    """Doubling the area increases the duty (more transfer surface)."""
    r1 = run_model(load_model("heat_exchanger_ua.gPROMS", "library"))
    q1 = r1["results"]["values"]["q"]
    code = load_model("heat_exchanger_ua.gPROMS", "library")
    code = code.replace("A   AS REAL DEFAULT 20.0", "A   AS REAL DEFAULT 40.0")
    r2 = run_model(code)
    assert r2["results"]["success"]
    assert r2["results"]["values"]["q"] > q1


def test_piecewise_interpolation():
    """PIECEWISE(x, x1,y1, ...) linear interpolation with flat
    extrapolation outside the table."""
    r = run_model("""
MODEL M
  VARIABLE
    y1 AS REAL
    y2 AS REAL
    y3 AS REAL
    y4 AS REAL
  EQUATION
    y1 = PIECEWISE(2.0, 0.0, 0.0, 10.0, 10.0) ;
    y2 = PIECEWISE(5.0, 0.0, 0.0, 10.0, 10.0) ;
    y3 = PIECEWISE(12.0, 0.0, 0.0, 10.0, 10.0) ;
    y4 = PIECEWISE(-1.0, 0.0, 0.0, 10.0, 10.0) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["y1"] - 2.0) < 1e-9       # on the line
    assert abs(v["y2"] - 5.0) < 1e-9
    assert abs(v["y3"] - 10.0) < 1e-9      # flat extrapolation above
    assert abs(v["y4"] - 0.0) < 1e-9       # flat extrapolation below


def test_piecewise_nonuniform_knots():
    """Knots need not be uniformly spaced; the engine sorts them."""
    r = run_model("""
MODEL M
  VARIABLE
    y AS REAL
  EQUATION
    y = PIECEWISE(7.5, 10.0, 100.0, 0.0, 0.0, 5.0, 50.0) ;
END
""")
    assert r["results"]["success"]
    # x = 7.5 between knots 5 and 10 (sorted): 50 + (100-50)*2.5/5 = 75
    assert abs(r["results"]["values"]["y"] - 75.0) < 1e-9


def test_piecewise_in_optimization():
    """PIECEWISE participates in optimisation (numeric row in the
    hybrid Jacobian)."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    y = PIECEWISE(x, 0.0, 0.0, 10.0, 20.0) ;
  BOUNDS
    x : 0 : 10 ;
  OPTIMIZE
    MAXIMIZE y ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["x"] - 10.0) < 1e-6
    assert abs(r["results"]["objective"]["value"] - 20.0) < 1e-6
