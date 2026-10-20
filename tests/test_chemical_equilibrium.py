"""Chemical equilibrium: K-value route + Gibbs minimisation — Θ3."""
import math

import pytest
from scipy.optimize import brentq

from backend.engine import run_model
from conftest import load_model


def test_wgs_kvalue_analytical(run):
    """At 1000 K the WGS extent is sqrt(K)/(1+sqrt(K)) — verified
    analytically in the test, independently of the model."""
    r = run("wgs_equilibrium.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    K = math.exp(4577.8 / 1000.0 - 4.33)
    xi_ref = math.sqrt(K) / (1 + math.sqrt(K))
    assert abs(v["k"] - K) < 1e-9
    assert abs(v["xi"] - xi_ref) < 1e-9
    assert abs(v["nco"] - (1 - xi_ref)) < 1e-9
    assert abs(v["nco2"] - xi_ref) < 1e-9
    # literature: ~53% CO conversion at 1000 K
    assert 0.52 < v["xi"] < 0.54


def test_wgs_gibbs_matches_kvalue(run):
    """Gibbs-energy minimisation (OPTIMIZE) with consistent standard
    potentials reproduces the equilibrium-constant extent — two
    independent routes, one answer."""
    r_k = run("wgs_equilibrium.gPROMS", folder="benchmarks")
    r_g = run("wgs_gibbs.gPROMS", folder="benchmarks")
    assert r_g["results"]["success"]
    vk = r_k["results"]["values"]
    vg = r_g["results"]["values"]
    assert abs(vg["nco2"] - vk["nco2"]) < 1e-3
    assert abs(vg["nco"] - vk["nco"]) < 1e-3
    assert abs(vg["nh2"] - vk["nh2"]) < 1e-3
    # the Gibbs energy is at its minimum at equilibrium
    assert r_g["results"]["objective"]["value"] < 0


def test_wgs_gibbs_mass_balances_hold(run):
    """The optimiser honours the element balances exactly."""
    r = run("wgs_gibbs.gPROMS", folder="benchmarks")
    v = r["results"]["values"]
    assert abs(v["nco"] + v["nco2"] - 1.0) < 1e-6
    assert abs(v["nco"] + v["nh2o"] + 2 * v["nco2"] - 2.0) < 1e-6
    assert abs(2 * v["nh2o"] + 2 * v["nh2"] - 2.0) < 1e-6
    assert abs(v["yco"] + v["yh2o"] + v["yco2"] + v["yh2"] - 1.0) < 1e-9


def test_ammonia_equilibrium_matches_independent_solver(run):
    """The ammonia extent from the language matches an independent
    quartic solve (brentq) at 723 K / 300 atm."""
    P, K = 300.0, 0.0064

    def f(xi):
        nN2, nH2, nNH3 = 1 - xi, 3 - 3 * xi, 2 * xi
        nT = nN2 + nH2 + nNH3
        yN2, yH2, yNH3 = nN2 / nT, nH2 / nT, nNH3 / nT
        return yNH3 ** 2 - K * P ** 2 * yN2 * yH2 ** 3

    xi_ref = brentq(f, 0.0, 0.9999)
    r = run("ammonia_equilibrium.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["xi"] - xi_ref) < 1e-7
    # ideal-gas limit at 723 K / 300 atm: ~82% conversion. Industry
    # reports 35-45% because of NON-ideality (fugacity coefficients at
    # 300 atm reduce the effective K) — the ideal-gas model is the
    # upper bound, and the gap is the fugacity correction itself.
    assert 0.80 < v["xi"] < 0.85
    assert abs(v["ynh3"] - 2 * xi_ref / (4 - 2 * xi_ref)) < 1e-7


def test_wgs_temperature_dependence():
    """Higher T lowers K for the exothermic WGS (Le Chatelier)."""
    r = run_model(load_model("wgs_equilibrium.gPROMS", "benchmarks"))
    v_low = r["results"]["values"]
    code = load_model("wgs_equilibrium.gPROMS", "benchmarks")
    code = code.replace("T AS REAL DEFAULT 1000.0", "T AS REAL DEFAULT 1400.0")
    r2 = run_model(code)
    v_high = r2["results"]["values"]
    assert v_high["k"] < v_low["k"]
    assert v_high["xi"] < v_low["xi"]
