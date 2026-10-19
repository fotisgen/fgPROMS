"""Flowsheet (UNIT/PROCESS) tests — M7."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_file, ParseError
from conftest import load_model, has_msg


def test_parse_file_models_and_process():
    f = parse_file(load_model("two_cstr.gPROMS", "benchmarks"))
    assert "cstrunit" in f.models
    assert f.process is not None
    assert f.process.name == "TwoCSTR"
    assert [u.name for u in f.process.units] == ["R1", "R2"]
    assert len(f.process.equations) == 2


def test_parse_unit_set_override():
    f = parse_file("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    x = a ;
END
PROCESS P
  UNIT
    U1 AS M SET a := 5 ;
  EQUATION
    U1.x = 0 ;
END
""")
    u = f.process.units[0]
    assert u.type_name == "M"
    assert u.sets[0].name == "a"


def test_parse_dotted_equation():
    f = parse_file("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
END
PROCESS P
  UNIT
    U1 AS M
  EQUATION
    U1.x = 2 ;
END
""")
    from backend.parser.gproms_parser import Var
    eq = f.process.equations[0]
    assert isinstance(eq.lhs, Var) and eq.lhs.name == "U1.x"


def test_two_cstr_series_solution(run):
    r = run("two_cstr.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["r1.ca"] - 1.0 / 3.0) < 1e-9
    assert abs(v["r2.ca"] - 1.0 / 18.0) < 1e-9
    assert r["results"]["maxResidual"] < 1e-9
    assert has_msg(r, "info_flowsheet_summary", "info")


def test_flowsheet_model_json(run):
    r = run("two_cstr.gPROMS", folder="benchmarks")
    m = r["model"]
    assert m["isProcess"] is True
    assert [u["name"] for u in m["units"]] == ["R1", "R2"]
    assert m["dof"] == {"unknowns": 6, "equations": 6, "dof": 0}
    names = {v["name"] for v in m["variables"]}
    assert "R1.CA" in names and "R2.CAin" in names


def test_unit_set_override_changes_solution():
    code = load_model("two_cstr.gPROMS", "benchmarks")
    code = code.replace("R1 AS CSTRUnit", "R1 AS CSTRUnit SET V := 20")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    # R1: CA = 2/(1 + 0.5*20) = 2/11
    assert abs(v["r1.ca"] - 2.0 / 11.0) < 1e-9
    # R2: CA = (2/11)/(1 + 5) = 1/33
    assert abs(v["r2.ca"] - (2.0 / 11.0) / 6.0) < 1e-9


def test_process_set_param_override():
    code = load_model("two_cstr.gPROMS", "benchmarks")
    code = code.replace(
        "  EQUATION\n    R1.CAin = 2.0 ;",
        "  SET\n    R1.V := 20 ;\n  EQUATION\n    R1.CAin = 2.0 ;")
    r = run_model(code)
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["r1.ca"] - 2.0 / 11.0) < 1e-9


def test_unknown_unit_type_blocks():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
END
PROCESS P
  UNIT
    U1 AS NoSuchModel
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_unknown_unit", "error")


def test_duplicate_unit_blocks():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
END
PROCESS P
  UNIT
    U1 AS M
    U1 AS M
END
""")
    assert has_msg(r, "err_duplicate_unit", "error")


def test_dynamic_unit_without_schedule_blocks():
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    $x = -a * x ;
  INITIAL
    x = 1.0 ;
END
PROCESS P
  UNIT
    U1 AS M
END
""")
    assert has_msg(r, "err_flowsheet_no_schedule", "error")
    assert r["results"] is None


def test_undeclared_dotted_variable_warns():
    code = load_model("two_cstr.gPROMS", "benchmarks")
    code = code.replace("R2.CAin = R1.CA ;", "R2.CAin = R1.zzz ;")
    r = run_model(code)
    assert has_msg(r, "warn_undeclared", "warn")
    assert r["ok"] is True


def test_flowsheet_dof_nonzero_blocks():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
END
PROCESS P
  UNIT
    U1 AS M
  EQUATION
    U1.x = 2 ;
    U1.x = 3 ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "dof_nonzero", "error")


def test_api_runs_flowsheet():
    from fastapi.testclient import TestClient
    from backend.app import app
    client = TestClient(app)
    code = load_model("two_cstr.gPROMS", "benchmarks")
    body = client.post("/api/run", json={"code": code}).json()
    assert body["results"]["success"]
    assert body["model"]["isProcess"] is True
    assert abs(body["results"]["values"]["r2.ca"] - 1.0 / 18.0) < 1e-9


def test_examples_include_flowsheet():
    from fastapi.testclient import TestClient
    from backend.app import app
    client = TestClient(app)
    names = client.get("/api/examples").json()
    assert "two_cstr.gPROMS" in names


# ---------------- M11: dynamic flowsheets ----------------
def test_two_tanks_dynamic_flowsheet(run):
    r = run("two_tanks.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    assert r["results"]["mode"] == "dynamic"
    assert r["results"]["method"] == "BDF2 adaptive"
    ts = r["timeSeries"]
    assert len(ts["t"]) >= 100

    # consistent initialisation: algebraic connections hold exactly at t=0
    h2_0 = ts["variables"]["t2.h"][0]
    f2_0 = ts["variables"]["t2.fout"][0]
    f1_0 = ts["variables"]["t1.fout"][0]
    assert abs(h2_0 - 1.0) < 1e-9
    assert abs(f2_0 - 0.4 * (h2_0 ** 0.5)) < 1e-9
    # T2 is fed by T1's outflow: Fin2(0) = Fout1(0)
    assert abs(ts["variables"]["t2.fin"][0] - f1_0) < 1e-9


def test_two_tanks_total_mass_balance(run):
    import numpy as np

    r = run("two_tanks.gPROMS", folder="benchmarks")
    ts = r["timeSeries"]
    t = np.array(ts["t"])
    h1 = np.array(ts["variables"]["t1.h"])
    h2 = np.array(ts["variables"]["t2.h"])
    f2 = np.array(ts["variables"]["t2.fout"])
    A, Fin = 2.0, 0.5
    dt = np.diff(t)
    integ = float(np.sum(0.5 * ((Fin - f2[:-1]) + (Fin - f2[1:])) * dt))
    lhs = A * ((h1[-1] + h2[-1]) - (h1[0] + h2[0]))
    assert abs(lhs - integ) < 1e-3 * max(1.0, abs(integ))


def test_two_tanks_cascade_direction(run):
    """Tank 1 fills first; tank 2 follows with a lag."""
    r = run("two_tanks.gPROMS", folder="benchmarks")
    ts = r["timeSeries"]
    h1 = ts["variables"]["t1.h"]
    h2 = ts["variables"]["t2.h"]
    assert h1[-1] > h1[0]           # both levels rise
    assert h2[-1] > h2[0]
    assert h1[-1] > h2[-1]          # upstream tank runs higher
    # both approach the same steady state (Fout -> 0.5 for both)
    assert abs(h1[-1] - h2[-1]) < 0.5


def test_two_tanks_model_json_states(run):
    r = run("two_tanks.gPROMS", folder="benchmarks")
    m = r["model"]
    assert m["isProcess"] is True
    assert m["isDynamic"] is True
    assert m["schedule"]["var"] == "t"
    states = {v["name"] for v in m["variables"] if v["isState"]}
    assert states == {"T1.h", "T1.C", "T2.h", "T2.C"}


def test_two_tanks_api_series():
    from fastapi.testclient import TestClient
    from backend.app import app
    client = TestClient(app)
    code = load_model("two_tanks.gPROMS", "benchmarks")
    body = client.post("/api/run", json={"code": code}).json()
    assert body["results"]["mode"] == "dynamic"
    assert body["results"]["success"]
    assert body["timeSeries"] is not None
    assert "t2.c" in body["timeSeries"]["variables"]


def test_process_initial_overrides_unit_initial():
    code = load_model("two_tanks.gPROMS", "benchmarks")
    code = code.replace(
        "  EQUATION\n    T1.Fin = 0.5 ;",
        "  INITIAL\n    T1.h = 1.5 ;\n  EQUATION\n    T1.Fin = 0.5 ;")
    r = run_model(code)
    assert r["results"]["success"]
    assert abs(r["timeSeries"]["variables"]["t1.h"][0] - 1.5) < 1e-9
