"""Thermophysical property function tests (CoolProp) — M6."""
import pytest

from backend.engine import run_model
from backend.properties import COOLPROP_AVAILABLE
from backend.parser.gproms_parser import parse_model, Str
from conftest import has_msg

pytestmark = pytest.mark.skipif(not COOLPROP_AVAILABLE,
                                reason="CoolProp not installed")


def test_string_literal_parsing():
    m = parse_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = CP('Water', 300, 1.0E5) ;
END
""")
    from backend.parser.gproms_parser import Func
    func = m.equations[0].rhs
    assert isinstance(func, Func) and func.name == "CP"
    assert isinstance(func.args[0], Str) and func.args[0].value == "Water"


def test_cp_water_matches_literature():
    from CoolProp.CoolProp import PropsSI
    cp = PropsSI("CPMASS", "T", 298.15, "P", 1e5, "Water")
    assert abs(cp - 4181.0) < 0.02 * 4181.0   # literature ~4180 J/kg/K


def test_water_heater_model_solves(run):
    r = run("water_heating.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    from CoolProp.CoolProp import PropsSI
    cp_in = PropsSI("CPMASS", "T", 298.15, "P", 1e5, "Water")
    cp_out = PropsSI("CPMASS", "T", 353.15, "P", 1e5, "Water")
    assert abs(v["cpin"] - cp_in) < 1e-6
    assert abs(v["cpout"] - cp_out) < 1e-6
    assert abs(v["q"] - (cp_out * 353.15 - cp_in * 298.15)) < 1e-4
    assert v["q"] > 0   # heating duty is positive


def test_property_functions_fall_back_to_numeric_jacobian(run):
    """Property functions are not differentiable symbolically — the solver
    must fall back to the numeric path (HYBR/LM), not crash."""
    r = run("water_heating.gPROMS", folder="benchmarks")
    assert r["results"]["method"] in ("HYBR", "LM")


def test_fluid_only_properties():
    r = run_model("""
MODEL M
  VARIABLE
    tc AS REAL
    pc AS REAL
    mm AS REAL
  EQUATION
    tc = TC('Water') ;
    pc = PC('Water') ;
    mm = M('Water') ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["tc"] - 647.096) < 0.5      # critical temperature, K
    assert abs(v["pc"] - 22064000) < 1000    # critical pressure, Pa
    assert abs(v["mm"] - 0.0180153) < 1e-6   # molar mass, kg/mol


def test_unknown_fluid_no_crash():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = CP('NotARealFluid', 300, 1.0E5) ;
END
""")
    assert r["ok"] is True
    # NaN residual -> solver reports failure cleanly, no exception
    if r["results"] is not None:
        assert r["results"]["success"] is False
    assert r["messages"]


def test_property_in_optimization_objective():
    # DOF = 1 (T free, x pinned by an unrelated equation); cp of liquid
    # water at 1 bar decreases monotonically over 280..300 K, so maximising
    # cp pushes T to the lower bound.
    r = run_model("""
MODEL M
  VARIABLE
    T AS REAL
    x AS REAL
  EQUATION
    x = 0 ;
  BOUNDS
    T : 280 : 300 ;
  OPTIMIZE
    MAXIMIZE CP('Water', T, 1.0E5) ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["t"] - 280.0) < 1e-4
    # objective equals the cp at 280 K
    from CoolProp.CoolProp import PropsSI
    cp_280 = PropsSI("CPMASS", "T", 280.0, "P", 1e5, "Water")
    assert abs(r["results"]["objective"]["value"] - cp_280) < 1e-4
