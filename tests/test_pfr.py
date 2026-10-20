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
