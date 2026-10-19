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


def test_dynamic_unit_blocks():
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
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 1 STEP 0.1
    END
END
PROCESS P
  UNIT
    U1 AS M
END
""")
    assert has_msg(r, "err_flowsheet_dynamic", "error")


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
