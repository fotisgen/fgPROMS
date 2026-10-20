"""Binary distillation column (McCabe-Thiele) — M35."""
import math

import pytest

from backend.engine import run_model
from conftest import load_model


def test_distillation_column_solves_and_balances(run):
    r = run("distillation_column.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    # overall mass balance: F zF = D xD + B xB
    assert abs(1.0 * 0.5 - (0.5 * v["xd"] + 0.5 * v["xb"])) < 1e-8
    # separation achieved: xD above feed, xB below
    assert v["xd"] > 0.8
    assert v["xb"] < 0.2
    # monotone liquid profile down the column
    xs = [v[f"x({k})"] for k in range(1, 9)]
    assert all(xs[k + 1] < xs[k] for k in range(7))
    # total condenser: xD equals the top VAPOUR composition y(1),
    # and the first liquid x(1) is its equilibrium partner (below xD)
    assert abs(v["y(1)"] - v["xd"]) < 1e-9
    assert xs[0] < v["xd"]
    assert abs(xs[-1] - v["xb"]) < 1e-9
    # equilibrium holds on every stage
    for k in range(1, 9):
        y_eq = 2.5 * v[f"x({k})"] / (1 + 1.5 * v[f"x({k})"])
        assert abs(v[f"y({k})"] - y_eq) < 1e-9


def test_distillation_fenske_limit():
    """At R -> infinity the column approaches total reflux, where the
    Fenske equation holds independently:
        ln[(xD/(1-xD)) / (xB/(1-xB))] / ln(alpha) = N"""
    code = load_model("distillation_column.gPROMS", "benchmarks")
    code = code.replace("R  AS REAL DEFAULT 2.0", "R  AS REAL DEFAULT 1.0E4")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    n_fenske = math.log((v["xd"] / (1 - v["xd"]))
                        / (v["xb"] / (1 - v["xb"]))) / math.log(2.5)
    assert abs(n_fenske - 8.0) < 0.05, f"Fenske gives {n_fenske:.3f}"


def test_distillation_more_reflux_sharper_separation():
    """Higher reflux ratio -> higher xD (classic design trade-off)."""
    r1 = run_model(load_model("distillation_column.gPROMS", "benchmarks"))
    xd1 = r1["results"]["values"]["xd"]
    code = load_model("distillation_column.gPROMS", "benchmarks")
    code = code.replace("R  AS REAL DEFAULT 2.0", "R  AS REAL DEFAULT 4.0")
    r2 = run_model(code)
    assert r2["results"]["success"]
    assert r2["results"]["values"]["xd"] > xd1


def test_distillation_cross_check_second_implementation():
    """An independent Python implementation of the same stage equations
    (nested scalar iteration) reproduces the model results — the engine
    verification for the whole column."""
    import numpy as np
    from scipy.optimize import root
    r = run_model(load_model("distillation_column.gPROMS", "benchmarks"))
    v = r["results"]["values"]

    # independent formulation: stage balances as residuals
    N = 8
    al = 2.5
    zF, F, D, R, q = 0.5, 1.0, 0.5, 2.0, 1.0
    B = F - D
    L = R * D
    V = L + D
    Lb = L + q * F
    Vb = V - (1 - q) * F

    def eqs(u):
        x = u[0:8]
        y = u[8:16]
        xD, xB = u[16], u[17]
        res = []
        res.append(xD - y[0])
        res.append(xB - x[7])
        res.append(F * zF - (D * xD + B * xB))
        for i in range(N):
            res.append(y[i] - al * x[i] / (1 + (al - 1) * x[i]))
        for i in range(3):
            res.append(V * y[i + 1] - (L * x[i] + D * xD))
        for i in range(3, 7):
            res.append(Lb * x[i] - (Vb * y[i + 1] + B * xB))
        return np.array(res)

    u0 = np.array([0.5] * 8 + [0.7] * 8 + [0.9, 0.1])
    sol = root(eqs, u0)
    assert sol.success
    for k in range(1, 9):
        assert abs(v[f"x({k})"] - sol.x[k - 1]) < 1e-6
        assert abs(v[f"y({k})"] - sol.x[8 + k - 1]) < 1e-6
    assert abs(v["xd"] - sol.x[16]) < 1e-6
    assert abs(v["xb"] - sol.x[17]) < 1e-6
