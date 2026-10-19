"""Material stream tests (STREAM AS MATERIAL) — M20."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_file
from conftest import has_msg, load_model


def test_parser_streams():
    f = parse_file(load_model("stream_heater.gPROMS", "benchmarks"))
    p = f.process
    assert [s.name for s in p.streams] == ["S1", "S2"]
    assert p.streams[0].fluid == "Water"


def test_stream_heater_enthalpy(run):
    from CoolProp.CoolProp import PropsSI
    r = run("stream_heater.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    h1 = PropsSI("HMASS", "T", 300.0, "P", 1e5, "Water")
    h2 = PropsSI("HMASS", "T", 350.0, "P", 1e5, "Water")
    assert abs(v["s1.h"] - h1) < 1e-6
    assert abs(v["s2.h"] - h2) < 1e-6
    assert abs(v["h1.q"] - (h2 - h1)) < 1e-4
    assert v["h1.q"] > 0        # heating duty positive


def test_stream_model_json(run):
    r = run("stream_heater.gPROMS", folder="benchmarks")
    m = r["model"]
    assert m["isProcess"] is True
    streams = {s["name"]: s for s in m["streams"]}
    assert set(streams) == {"S1", "S2"}
    assert streams["S1"]["fluid"] == "Water"
    names = {v["name"] for v in m["variables"]}
    for vn in ("S1.F", "S1.T", "S1.P", "S1.H", "S1.CP", "S1.RHO",
               "S2.F", "S2.T", "S2.P", "S2.H", "S2.CP", "S2.RHO"):
        assert vn in names


def test_stream_mixture_fluid():
    from CoolProp.CoolProp import PropsSI
    r = run_model("""
MODEL U
  VARIABLE
    Q AS REAL
    x AS REAL
  EQUATION
    Q = 1 ;
END
PROCESS P
  UNIT
    U1 AS U
  STREAM
    S1 AS MATERIAL('n-Butane[0.4]&Propane[0.6]')
  EQUATION
    S1.F = 1.0 ;
    S1.T = 270.0 ;
    S1.P = 2.0E5 ;
    U1.x = S1.RHO ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    rho_ref = PropsSI("DMASS", "T", 270.0, "P", 2e5,
                      "n-Butane[0.4]&Propane[0.6]")
    assert abs(v["s1.rho"] - rho_ref) < 1e-4
    assert abs(v["u1.x"] - rho_ref) < 1e-4


def test_stream_dof_mismatch_detected():
    """A stream adds 3 free variables (F, T, P) — leaving them unconnected
    must produce a DOF error, not a silent failure."""
    r = run_model("""
MODEL U
  VARIABLE
    Q AS REAL
  EQUATION
    Q = 1 ;
END
PROCESS P
  UNIT
    U1 AS U
  STREAM
    S1 AS MATERIAL('Water')
  EQUATION
    U1.Q = S1.H ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "dof_nonzero", "error")
