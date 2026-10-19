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


# ---------------- M13: mixtures & phase equilibrium ----------------
MIX = "n-Butane[0.4]&Propane[0.6]"


def test_mixture_density_matches_coolprop():
    from CoolProp.CoolProp import PropsSI
    r = run_model("""
MODEL M
  VARIABLE
    d AS REAL
  EQUATION
    d = RHO('Water[0.3]&Methanol[0.7]', 300.0, 1.0E5) ;
END
""")
    assert r["results"]["success"]
    ref = PropsSI("DMASS", "T", 300.0, "P", 1e5, "Water[0.3]&Methanol[0.7]")
    assert abs(r["results"]["values"]["d"] - ref) < 1e-6
    # sanity: between the two pure-fluid densities
    assert 700.0 < ref < 1000.0


def test_mixture_flash_benchmark(run):
    from CoolProp.CoolProp import PropsSI
    r = run("mixture_flash.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    # exact match with independent CoolProp calls
    assert abs(v["qv"] - PropsSI("Q", "T", 270.0, "P", 2e5, MIX)) < 1e-6
    assert abs(v["tb"] - PropsSI("T", "P", 2e5, "Q", 0, MIX)) < 1e-6
    assert abs(v["td"] - PropsSI("T", "P", 2e5, "Q", 1, MIX)) < 1e-6
    assert abs(v["hm"] - PropsSI("H", "T", 270.0, "P", 2e5, MIX)) < 1e-3
    # physical checks: two-phase region, T between bubble and dew
    assert 0.0 < v["qv"] < 1.0
    assert v["tb"] < 270.0 < v["td"]


def test_bubble_dew_pressure_functions():
    from CoolProp.CoolProp import PropsSI
    r = run_model("""
MODEL M
  VARIABLE
    pb AS REAL
    pd AS REAL
  EQUATION
    pb = PBUBBLE('n-Butane[0.4]&Propane[0.6]', 270.0) ;
    pd = PDEW('n-Butane[0.4]&Propane[0.6]', 270.0) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["pb"] - PropsSI("P", "T", 270.0, "Q", 0, MIX)) < 1e-4
    assert abs(v["pd"] - PropsSI("P", "T", 270.0, "Q", 1, MIX)) < 1e-4
    # at fixed T the bubble pressure exceeds the dew pressure
    assert v["pb"] > v["pd"]


def test_internal_energy_and_gibbs():
    from CoolProp.CoolProp import PropsSI
    r = run_model("""
MODEL M
  VARIABLE
    u AS REAL
    g AS REAL
  EQUATION
    u = U('Water', 350.0, 1.0E5) ;
    g = G('Water', 350.0, 1.0E5) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["u"] - PropsSI("UMASS", "T", 350.0, "P", 1e5, "Water")) < 1e-6
    assert abs(v["g"] - PropsSI("GMASS", "T", 350.0, "P", 1e5, "Water")) < 1e-6


def test_invalid_mixture_component_no_crash():
    r = run_model("""
MODEL M
  VARIABLE
    q AS REAL
  EQUATION
    q = Q('NotAFluid[0.5]&Water[0.5]', 300.0, 1.0E5) ;
END
""")
    assert r["ok"] is True
    if r["results"] is not None:
        assert r["results"]["success"] is False
    assert r["messages"]


def test_mixture_function_parses_as_function_not_index():
    from backend.parser.gproms_parser import parse_model, Func
    m = parse_model("""
MODEL M
  VARIABLE
    q AS REAL
  EQUATION
    q = Q('Water', 300.0, 1.0E5) ;
END
""")
    assert isinstance(m.equations[0].rhs, Func)
    assert m.equations[0].rhs.name == "Q"


# ---------------- M25: composition as a variable ----------------
def _as_ref(fluid, z, T, P, method):
    from CoolProp.CoolProp import AbstractState, PT_INPUTS
    AS = AbstractState("HEOS", fluid)
    AS.set_mole_fractions(list(z))
    AS.update(PT_INPUTS, P, T)
    return float(getattr(AS, method)())


def test_mixture_composition_benchmark(run):
    r = run("mixture_composition.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["z(1)"] - 0.4) < 1e-9
    assert abs(v["z(2)"] - 0.6) < 1e-9
    cp_ref = _as_ref("Water&Methanol", [0.4, 0.6], 300.0, 1e5, "cpmass")
    rho_ref = _as_ref("Water&Methanol", [0.4, 0.6], 300.0, 1e5, "rhomass")
    assert abs(v["cp"] - cp_ref) < 1e-3
    assert abs(v["rho"] - rho_ref) < 1e-3
    # sanity: between the two pure-fluid values
    cp_w = _as_ref("Water", [1.0], 300.0, 1e5, "cpmass")
    cp_m = _as_ref("Methanol", [1.0], 300.0, 1e5, "cpmass")
    assert min(cp_w, cp_m) < v["cp"] < max(cp_w, cp_m)


def test_composition_solved_as_unknown():
    """The composition itself can be the unknown: find the water fraction
    that gives a target cp."""
    cp_ref = _as_ref("Water&Methanol", [0.7, 0.3], 300.0, 1e5, "cpmass")
    r = run_model(f"""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 300.0
    P AS REAL DEFAULT 1.0E5
  VARIABLE
    z AS ARRAY(2) OF REAL
    cp AS REAL
  EQUATION
    z(1) + z(2) = 1 ;
    cp = CP('Water&Methanol', z, T, P) ;
    cp = {cp_ref:.6f} ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["cp"] - cp_ref) < 1e-6
    assert abs(v["z(1)"] - 0.7) < 1e-4


def test_composition_optimization():
    """Maximising cp over the composition pushes to the higher-cp pure
    component (water)."""
    r = run_model("""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 300.0
    P AS REAL DEFAULT 1.0E5
  VARIABLE
    z AS ARRAY(2) OF REAL
    cp AS REAL
  EQUATION
    z(1) + z(2) = 1 ;
    cp = CP('Water&Methanol', z, T, P) ;
  BOUNDS
    z(1) : 0 : 1 ;
  OPTIMIZE
    MAXIMIZE cp ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["z(1)"] - 1.0) < 1e-3     # pure water
    assert abs(v["z(2)"]) < 1e-3


def test_composition_two_phase():
    from CoolProp.CoolProp import AbstractState, PT_INPUTS
    r = run_model("""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 270.0
    P AS REAL DEFAULT 2.0E5
  VARIABLE
    z AS ARRAY(2) OF REAL
    q AS REAL
  EQUATION
    z(1) + z(2) = 1 ;
    z(2) = 0.6 ;
    q = Q('n-Butane&Propane', z, T, P) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    AS = AbstractState("HEOS", "n-Butane&Propane")
    AS.set_mole_fractions([0.4, 0.6])
    AS.update(PT_INPUTS, 2e5, 270.0)
    assert abs(v["q"] - float(AS.Q())) < 1e-6
    assert 0.0 < v["q"] < 1.0


def test_composition_bubble_temperature():
    from CoolProp.CoolProp import AbstractState, PQ_INPUTS
    r = run_model("""
MODEL M
  PARAMETER
    P AS REAL DEFAULT 2.0E5
  VARIABLE
    z AS ARRAY(2) OF REAL
    tb AS REAL
  EQUATION
    z(1) + z(2) = 1 ;
    z(2) = 0.6 ;
    tb = TBUBBLE('n-Butane&Propane', z, P) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    AS = AbstractState("HEOS", "n-Butane&Propane")
    AS.set_mole_fractions([0.4, 0.6])
    AS.update(PQ_INPUTS, 2e5, 0.0)
    assert abs(v["tb"] - float(AS.T())) < 1e-4
