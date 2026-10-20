"""Activity-coefficient models (NRTL, Wilson) — Θ2."""
import math

import pytest

from backend.engine import run_model
from conftest import load_model


# ---------------- NRTL: ethanol-water azeotrope ----------------

def _nrtl_reference(x1s):
    """Independent Python implementation of the NRTL bubble line:
    returns (x_az, min_Pb) found by the same sign-change logic."""
    import math
    from CoolProp.CoolProp import AbstractState, QT_INPUTS
    T = 351.35
    A12, A21, alf = 14466.0, -3351.0, 0.30
    R = 8.314
    t12, t21 = A12 / (R * T), A21 / (R * T)
    G12, G21 = math.exp(-alf * t12), math.exp(-alf * t21)

    def psat(fluid):
        AS = AbstractState("HEOS", fluid)
        AS.update(QT_INPUTS, 0.0, T)
        return float(AS.p())

    p1, p2 = psat("Ethanol"), psat("Water")

    def point(x1):
        x2 = 1 - x1
        g1 = math.exp(x2 * x2 * (t21 * (G21 / (x1 + x2 * G21)) ** 2
                                 + t12 * G12 / (x2 + x1 * G12) ** 2))
        g2 = math.exp(x1 * x1 * (t12 * (G12 / (x2 + x1 * G12)) ** 2
                                 + t21 * G21 / (x1 + x2 * G21) ** 2))
        pb = x1 * g1 * p1 + x2 * g2 * p2
        y1 = x1 * g1 * p1 / pb
        return y1, pb

    xs = [i * 0.95 / 12 for i in range(1, 13)]
    pts = [point(x) for x in xs]
    diffs = [pts[i][0] - xs[i] for i in range(12)]
    i = next(k for k in range(11) if diffs[k] * diffs[k + 1] <= 0)
    f0, f1 = diffs[i], diffs[i + 1]
    x_az = xs[i] + (0.0 - f0) / (f1 - f0) * (xs[i + 1] - xs[i])
    return x_az, max(p[1] for p in pts)


def test_nrtl_azeotrope_located(run):
    """The NRTL sweep crosses the azeotrope where y1 - x1 changes sign;
    the crossing must match an independent Python implementation AND the
    literature value x_az = 0.8947 (ethanol/water, 78.2 C)."""
    r = run("nrtl_ethanol_water.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    x = [v[f"x1({k})"] for k in range(1, 13)]
    y = [v[f"y1({k})"] for k in range(1, 13)]
    diffs = [y[k] - x[k] for k in range(12)]
    i = next(k for k in range(11) if diffs[k] * diffs[k + 1] <= 0)
    f0, f1 = diffs[i], diffs[i + 1]
    x_az = x[i] + (0.0 - f0) / (f1 - f0) * (x[i + 1] - x[i])
    x_az_ref, _ = _nrtl_reference(x)
    assert abs(x_az - x_az_ref) < 1e-3
    # literature: ethanol/water minimum-boiling azeotrope x1 = 0.8947
    assert 0.88 < x_az < 0.95, f"azeotrope at x1={x_az:.4f}"


def test_nrtl_bubble_pressure_near_1atm(run):
    """The MAXIMUM bubble pressure (positive deviation from Raoult's law)
    is ~1 atm at 78.2 C — the minimum-boiling azeotrope."""
    r = run("nrtl_ethanol_water.gPROMS", folder="benchmarks")
    v = r["results"]["values"]
    pb = [v[f"pb({k})"] for k in range(1, 13)]
    _x_az, max_pb_ref = _nrtl_reference([])
    assert abs(max(pb) - max_pb_ref) < 1e-3
    # the azeotropic pressure is of the order of 1 atm (not fitted exactly)
    assert 1.0e5 < max(pb) < 1.3e5


def test_nrtl_gammas_match_second_implementation(run):
    """The gamma values from the language match an independent Python
    implementation of the same NRTL equations (cross-check)."""
    T = 351.35
    A12, A21, alf = 14466.0, -3351.0, 0.30
    R = 8.314
    t12, t21 = A12 / (R * T), A21 / (R * T)
    G12, G21 = math.exp(-alf * t12), math.exp(-alf * t21)

    def gamma(x1):
        x2 = 1 - x1
        g1 = math.exp(x2 * x2 * (t21 * (G21 / (x1 + x2 * G21)) ** 2
                                 + t12 * G12 / (x2 + x1 * G12) ** 2))
        g2 = math.exp(x1 * x1 * (t12 * (G12 / (x2 + x1 * G12)) ** 2
                                 + t21 * G21 / (x1 + x2 * G21) ** 2))
        return g1, g2

    r = run("nrtl_ethanol_water.gPROMS", folder="benchmarks")
    v = r["results"]["values"]
    for k in range(1, 13):
        x1 = v[f"x1({k})"]
        g1_ref, g2_ref = gamma(x1)
        assert abs(v[f"g1({k})"] - g1_ref) < 1e-9
        assert abs(v[f"g2({k})"] - g2_ref) < 1e-9


# ---------------- Wilson: methanol-water ----------------

def test_wilson_infinite_dilution_identity(run):
    """ln gamma1(inf) = 1 - ln(L12) - L21 — computed in the model AND
    independently in the test."""
    r = run("wilson_methanol_water.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    T = 298.15
    L12 = math.exp(-108.66 / T)
    L21 = math.exp(-399.05 / T)
    assert abs(v["lng1inf"] - (1 - math.log(L12) - L21)) < 1e-12
    # sanity: methanol in infinite water dilution has gamma > 1
    assert v["lng1inf"] > 0


def test_wilson_gammas_match_second_implementation(run):
    T = 298.15
    L12 = math.exp(-108.66 / T)
    L21 = math.exp(-399.05 / T)

    def gamma(x1):
        x2 = 1 - x1
        g1 = math.exp(-math.log(x1 + L12 * x2)
                      + x2 * (L12 / (x1 + L12 * x2)
                              - L21 / (L21 * x1 + x2)))
        g2 = math.exp(-math.log(x2 + L21 * x1)
                      - x1 * (L12 / (x1 + L12 * x2)
                              - L21 / (L21 * x1 + x2)))
        return g1, g2

    r = run("wilson_methanol_water.gPROMS", folder="benchmarks")
    v = r["results"]["values"]
    g1_ref, g2_ref = gamma(0.5)
    assert abs(v["g1"] - g1_ref) < 1e-9
    assert abs(v["g2"] - g2_ref) < 1e-9
    # Gibbs-Duhem consistency: gamma1 >= 1 at x1=0.5 for this system
    assert v["g1"] > 1.0 and v["g2"] > 1.0


def test_wilson_dilute_limit_approaches_identity():
    """gamma1 at x1 -> 0 numerically approaches exp(1 - ln L12 - L21)."""
    code = load_model("wilson_methanol_water.gPROMS", "benchmarks")
    code = code.replace("x1  AS REAL DEFAULT 0.5",
                        "x1  AS REAL DEFAULT 0.0001")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    T = 298.15
    L12 = math.exp(-108.66 / T)
    L21 = math.exp(-399.05 / T)
    g1_inf = math.exp(1 - math.log(L12) - L21)
    assert abs(v["g1"] - g1_inf) < 0.01
