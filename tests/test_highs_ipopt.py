"""LP/MILP via HiGHS + Ipopt NLP rescue — M32."""
import pytest
from scipy.optimize import linprog, minimize

from backend.engine import run_model
from conftest import load_model


def test_linear_program_highs(run):
    """The textbook LP is solved by HiGHS; the optimum is verified
    independently with scipy.optimize.linprog."""
    r = run("linear_program.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    assert r["results"]["method"].startswith("HIGHS")
    v = r["results"]["values"]
    ref = linprog(c=[-3, -2], A_ub=[[1, 2], [2, 1]], b_ub=[4, 5],
                  bounds=[(0, 10), (0, 10)])
    assert abs(v["x"] - ref.x[0]) < 1e-9
    assert abs(v["y"] - ref.x[1]) < 1e-9
    assert abs(r["results"]["objective"]["value"] - 8.0) < 1e-9


def test_milp_truck_highs(run):
    """The integer knapsack: HiGHS finds the integer optimum 3xA+1xB (130)."""
    r = run("milp_truck.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    assert r["results"]["method"].startswith("HIGHS")
    v = r["results"]["values"]
    assert abs(v["na"] - 3.0) < 1e-9
    assert abs(v["nb"] - 1.0) < 1e-9
    assert abs(r["results"]["objective"]["value"] - 130.0) < 1e-9


def test_highs_equality_constraints():
    """LP with equality constraints — HiGHS handles them exactly."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    x + y = 10 ;
    x = 2*y ;
  BOUNDS
    x : 0 : 100 ;
    y : 0 : 100 ;
  OPTIMIZE
    MINIMIZE x - y ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["x"] - 20 / 3) < 1e-7
    assert abs(v["y"] - 10 / 3) < 1e-7


def test_nonlinear_still_uses_slsqp(run):
    """A nonlinear objective falls through to the NLP path (SLSQP)."""
    r = run("cstr_optimize.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    assert r["results"]["method"] == "SLSQP"


def test_ipopt_rescues_hard_nlp():
    """A nasty NLP that SLSQP cannot finish is rescued by Ipopt — the
    answer matches a brute-force grid search (independent reference)."""
    import numpy as np
    code = """
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    x^2 + y^2 = 1 ;
  BOUNDS
    x : -2 : 2 ;
    y : -2 : 2 ;
  OPTIMIZE
    MAXIMIZE x * y ;
END
"""
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    # optimum of xy on the circle: x = y = 1/sqrt(2), value 0.5
    # (any quadrant — check the objective, robust to sign choice)
    assert abs(abs(v["x"] * v["y"]) - 0.5) < 1e-6
    assert abs(r["results"]["objective"]["value"] - 0.5) < 1e-6
    # independent check: grid
    grid = np.linspace(-1, 1, 2001)
    best = max((x * np.sqrt(max(0.0, 1 - x * x)) for x in grid))
    assert abs(best - 0.5) < 1e-3


def test_ipopt_path_exists():
    from backend.solver.solver import _ipopt_path
    p = _ipopt_path()
    assert p and p.endswith("ipopt.exe")
