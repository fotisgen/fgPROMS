"""Cubic equations of state in the modelling language + EOS backend
selection — Θ1."""
from pathlib import Path

import pytest

from backend.engine import run_model
from conftest import load_model

ROOT = Path(__file__).resolve().parent.parent


def _coolprop_ref(backend, fluid, z, T, P, method):
    from CoolProp.CoolProp import AbstractState, PT_INPUTS
    AS = AbstractState(backend, fluid)
    AS.set_mole_fractions(list(z))
    AS.update(PT_INPUTS, P, T)
    if method == "fugacity_coefficient":
        return float(AS.fugacity_coefficient(0))
    return float(getattr(AS, method)())


def test_peng_robinson_benchmark(run):
    """The PR cubic written IN the language matches CoolProp's PR backend."""
    r = run("peng_robinson.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    Z_ref = _coolprop_ref("PR", "n-Butane", [1.0], 400.0, 2e6,
                          "compressibility_factor")
    phi_ref = _coolprop_ref("PR", "n-Butane", [1.0], 400.0, 2e6,
                            "fugacity_coefficient")
    assert abs(v["z"] - Z_ref) < 1e-4
    assert abs(v["phi"] - phi_ref) < 1e-4


def test_soave_benchmark(run):
    r = run("soave_redlich_kwong.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    Z_ref = _coolprop_ref("SRK", "n-Butane", [1.0], 400.0, 2e6,
                          "compressibility_factor")
    phi_ref = _coolprop_ref("SRK", "n-Butane", [1.0], 400.0, 2e6,
                            "fugacity_coefficient")
    assert abs(v["z"] - Z_ref) < 1e-4
    assert abs(v["phi"] - phi_ref) < 1e-4


def test_peng_robinson_mix_benchmark(run):
    """Mixing rules + component fugacities match CoolProp's PR backend."""
    r = run("peng_robinson_mix.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    from CoolProp.CoolProp import AbstractState, PT_INPUTS
    AS = AbstractState("PR", "n-Butane&Propane")
    AS.set_mole_fractions([0.4, 0.6])
    AS.update(PT_INPUTS, 2e6, 400.0)
    assert abs(v["z"] - float(AS.compressibility_factor())) < 1e-4
    assert abs(v["phi1"] - float(AS.fugacity_coefficient(0))) < 1e-4
    assert abs(v["phi2"] - float(AS.fugacity_coefficient(1))) < 1e-4


def test_eos_backend_argument():
    """CP('fluid', T, P, 'PR') evaluates with the requested EOS."""
    r = run_model("""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 400.0
    P AS REAL DEFAULT 2.0E6
  VARIABLE
    cpPR  AS REAL
    cpSRK AS REAL
    cpDef AS REAL
    z AS ARRAY(2) OF REAL
    cpMix AS REAL
  EQUATION
    cpPR  = CP('n-Butane', T, P, 'PR') ;
    cpSRK = CP('n-Butane', T, P, 'SRK') ;
    cpDef = CP('n-Butane', T, P) ;
    z(1) + z(2) = 1 ;
    z(2) = 0.6 ;
    cpMix = CP('n-Butane&Propane', z, T, P, 'PR') ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["cppr"] - _coolprop_ref("PR", "n-Butane", [1.0], 400.0,
                                         2e6, "cpmass")) < 1e-3
    assert abs(v["cpsrk"] - _coolprop_ref("SRK", "n-Butane", [1.0], 400.0,
                                          2e6, "cpmass")) < 1e-3
    assert abs(v["cpmix"] - _coolprop_ref("PR", "n-Butane&Propane",
                                          [0.4, 0.6], 400.0, 2e6,
                                          "cpmass")) < 1e-3
    # the reference (default) property differs from the cubic EOS values
    assert abs(v["cppr"] - v["cpdef"]) > 1.0


def test_eos_backend_bubble_temperature():
    """FIXED_Q functions accept a backend too."""
    from CoolProp.CoolProp import AbstractState, PQ_INPUTS
    r = run_model("""
MODEL M
  PARAMETER
    P AS REAL DEFAULT 2.0E5
  VARIABLE
    tb AS REAL
  EQUATION
    tb = TBUBBLE('n-Butane', P, 'PR') ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    AS = AbstractState("PR", "n-Butane")
    AS.update(PQ_INPUTS, 2e5, 0.0)
    assert abs(v["tb"] - float(AS.T())) < 1e-3


def test_three_roots_visible():
    """The cubic gives different roots from different initial guesses —
    the classic vapour/liquid story of a cubic EOS. At 300 K / 1 bar
    (two-phase for n-butane) the PR cubic has the roots
    Z_liq = 0.02055 and Z_vap = 0.97266 (independently computed with
    numpy.roots)."""
    code = """
MODEL Cubic
  PARAMETER
    Apr AS REAL DEFAULT 0.029679
    Bpr AS REAL DEFAULT 0.002905
  VARIABLE
    Z AS REAL
  EQUATION
    Z^3 - (1 - Bpr) * Z^2 + (Apr - 2*Bpr - 3*Bpr^2) * Z
        - (Apr*Bpr - Bpr^2 - Bpr^3) = 0 ;
  INITIAL
    Z = {Z} ;
END
"""
    r_liq = run_model(code.format(Z="0.02"))
    assert r_liq["results"]["success"]
    r_vap = run_model(code.format(Z="0.95"))
    assert r_vap["results"]["success"]
    assert abs(r_liq["results"]["values"]["z"] - 0.02055) < 1e-4
    assert abs(r_vap["results"]["values"]["z"] - 0.97266) < 1e-4
