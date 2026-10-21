"""Plug-flow reactor — M37."""
import math

import pytest

from backend.engine import run_model
from conftest import load_model


def test_pfr_matches_analytical_profile(run):
    """CA(z) = CAin exp(-k z / u) — verified pointwise."""
    r = run("plug_flow_reactor.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    k, u, L = 2.0, 0.5, 1.0
    for i in range(1, 22):
        z = (i - 1) * L / 20.0
        ca_ref = math.exp(-k * z / u)
        assert abs(v[f"ca({i})"] - ca_ref) < 4e-3, f"CA({i}) at z={z:.2f}"
    assert abs(v["ca(1)"] - 1.0) < 1e-9        # inlet boundary
    assert abs(v["ca(21)"] - math.exp(-4.0)) < 4e-3


def test_pfr_library_unit(run):
    """The library PFR reports the exit concentration."""
    r = run("pfr.gPROMS", folder="library")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["caout"] - math.exp(-4.0)) < 4e-3


def test_pfr_beats_cstr_conversion(run):
    """Same residence time: the PFR converts more than the CSTR
    (classic reactor-engineering result)."""
    # PFR: 1 - exp(-k tau),  CSTR: k tau / (1 + k tau)
    tau = 2.0            # L / u = 1 / 0.5
    k = 2.0
    conv_pfr = 1 - math.exp(-k * tau)
    conv_cstr = k * tau / (1 + k * tau)
    assert conv_pfr > conv_cstr
    r = run("plug_flow_reactor.gPROMS", folder="benchmarks")
    v = r["results"]["values"]
    assert abs((1 - v["ca(21)"]) - conv_pfr) < 4e-3
    # and the model's own conversion matches the analytical one
    assert conv_pfr > 0.98

def test_pfr_thermal_cross_check_and_physics(run):
    """The non-isothermal PFR profiles match an independent solve_ivp
    integration of the same ODEs, the temperature rises (exothermic)
    and the conversion beats the isothermal PFR at the inlet T."""
    import numpy as np
    from scipy.integrate import solve_ivp
    r = run("pfr_thermal.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]

    L, u, k0, Ea, R, dH, rho, cp = (1.0, 0.5, 2.0e8, 60000.0, 8.314,
                                    -50000.0, 1000.0, 4000.0)
    CAin, Tin = 100.0, 300.0

    def rhs(z, y):
        CA, T = y
        rate = k0 * np.exp(-Ea / (R * T)) * CA
        return [-rate / u, -dH * rate / (u * rho * cp)]

    sol = solve_ivp(rhs, [0, L], [CAin, Tin], t_eval=np.linspace(0, L, 21),
                    rtol=1e-10, atol=1e-12)
    CA_ref, T_ref = sol.y[0], sol.y[1]
    for k in range(1, 22):
        assert abs(v[f"ca({k})"] - CA_ref[k - 1]) < 0.5
        assert abs(v[f"t({k})"] - T_ref[k - 1]) < 0.2
    # physics: temperature rises, conversion beats isothermal-at-inlet
    assert v["t(21)"] > v["t(1)"]
    k_in = k0 * np.exp(-Ea / (R * Tin))
    conv_iso = 1 - np.exp(-k_in * L / u)
    conv_thermal = 1 - v["ca(21)"] / CAin
    assert conv_thermal > conv_iso
