"""Optimization (SLSQP) tests — M5."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model, ParseError
from conftest import has_msg


def test_parser_objective_and_bounds():
    m = parse_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
  BOUNDS
    x : 0 : 10 ;
  OPTIMIZE
    MAXIMIZE x ;
END
""")
    assert m.objective.kind == "maximize"
    assert len(m.bounds) == 1
    assert m.bounds[0].name == "x"


def test_parser_minimize_keyword():
    m = parse_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
  OPTIMIZE
    MINIMIZE x ;
END
""")
    assert m.objective.kind == "minimize"


def test_parser_optimize_without_keyword_raises():
    with pytest.raises(ParseError):
        parse_model("MODEL M\nVARIABLE\nx AS REAL\nEQUATION\nx = 1 ;\nOPTIMIZE\nx ;\nEND")


def test_maximize_production_rate(run):
    r = run("cstr_optimize.gPROMS", folder="benchmarks")
    assert r["results"]["mode"] == "optimization"
    assert r["results"]["success"]
    v = r["results"]["values"]
    # analytical optimum at the upper bound V = 20
    assert abs(v["v"] - 20.0) < 1e-5
    assert abs(v["ca"] - 2.0 / 11.0) < 1e-5
    assert abs(r["results"]["objective"]["value"] - 1.818181818) < 1e-5
    assert r["results"]["objective"]["kind"] == "maximize"
    assert has_msg(r, "optimize_ok", "info")


def test_minimize_respects_lower_bound():
    code = ("""
MODEL M
  PARAMETER
    Fin  AS REAL DEFAULT 1.0
    CAin AS REAL DEFAULT 2.0
    k    AS REAL DEFAULT 0.5
  VARIABLE
    V  AS REAL
    CA AS REAL
  EQUATION
    Fin * CAin - Fin * CA - V * k * CA = 0 ;
  BOUNDS
    V : 0.1 : 20 ;
  OPTIMIZE
    MINIMIZE V * k * CA ;
END
""")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["v"] - 0.1) < 1e-5
    # CA = 2/(1 + 0.5*0.1) = 1.90476, obj = 0.1*0.5*1.90476 = 0.095238
    assert abs(v["ca"] - 2.0 / 1.05) < 1e-5
    assert abs(r["results"]["objective"]["value"] - 0.0952380952) < 1e-5


def test_overdetermined_with_objective_blocks():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
    x = 2 ;
  OPTIMIZE
    MAXIMIZE x ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_overdetermined", "error")


def test_trivial_optimization_dof_zero():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 5 ;
  OPTIMIZE
    MAXIMIZE x * x ;
END
""")
    assert r["results"]["mode"] == "optimization"
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["x"] - 5.0) < 1e-9
    assert abs(r["results"]["objective"]["value"] - 25.0) < 1e-9
    assert has_msg(r, "info_trivial_opt", "info")


def test_invalid_bounds_warn_and_proceed():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1 ;
  BOUNDS
    x : 10 : 0 ;
  OPTIMIZE
    MINIMIZE x ;
END
""")
    assert has_msg(r, "warn_bounds_invalid", "warn")
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["x"] - 1.0) < 1e-9


def test_optimization_with_derivatives_blocked():
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
  OPTIMIZE
    MINIMIZE x ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_dynopt_no_controls", "error")


def test_objective_reported_in_model_json(run):
    r = run("cstr_optimize.gPROMS", folder="benchmarks")
    assert r["model"]["objective"]["kind"] == "maximize"
    assert "V * k * CA" in r["model"]["objective"]["text"]
    assert r["model"]["bounds"][0]["name"] == "V"
    assert r["model"]["bounds"][0]["lo"] == 0.1
    assert r["model"]["bounds"][0]["hi"] == 20.0
    assert r["model"]["dof"]["dof"] == 1


def test_optimization_residuals_small(run):
    r = run("cstr_optimize.gPROMS", folder="benchmarks")
    assert r["results"]["maxResidual"] < 1e-8


def test_optimization_message_dof_for_opt(run):
    r = run("cstr_optimize.gPROMS", folder="benchmarks")
    assert has_msg(r, "dof_for_opt", "info")
