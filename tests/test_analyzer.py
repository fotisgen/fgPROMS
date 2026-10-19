"""Analyzer tests: DOF counting, diagnostics, blocking conditions."""
from backend.engine import run_model

from conftest import has_msg


def run_code(code: str) -> dict:
    return run_model(code)


def test_dof_zero_cstr(run):
    r = run("cstr.gPROMS", folder="examples")
    assert r["model"]["dof"] == {"unknowns": 3, "equations": 3, "dof": 0}
    assert r["results"] is not None


def test_dof_nonzero_blocks():
    r = run_code("""
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    x + y = 5 ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "dof_nonzero", "error")
    assert r["model"]["dof"]["dof"] == 1


def test_undeclared_variable_warns():
    r = run_code("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 2.0
  VARIABLE
    x AS REAL
  EQUATION
    x = a + z ;
END
""")
    assert has_msg(r, "warn_undeclared", "warn")


def test_unknown_function_used_as_index_errors():
    """ERF(a) parses as an indexed access now — a name that is not an
    ARRAY must produce a clear error, not silent garbage."""
    r = run_code("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    x = ERF(a) ;
END
""")
    assert has_msg(r, "err_not_array", "error")
    assert r["results"] is None


def test_derivative_without_schedule_blocks():
    r = run_code("""
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
""")
    assert has_msg(r, "err_dynamic_no_schedule", "error")
    assert r["results"] is None


def test_state_without_initial_blocks():
    r = run_code("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    $x = -a * x ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 1 STEP 0.1
    END
END
""")
    assert has_msg(r, "err_state_no_initial", "error")
    assert r["results"] is None


def test_set_fixes_variable():
    r = run_code("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 2.0
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    x + y = a ;
  SET
    x := 1.5 ;
END
""")
    assert r["model"]["dof"]["dof"] == 0
    assert r["model"]["dof"]["unknowns"] == 1
    fixed = {v["name"]: v["fixed"] for v in r["model"]["variables"]}
    assert fixed == {"x": True, "y": False}
    assert abs(r["results"]["values"]["y"] - 0.5) < 1e-9


def test_duplicate_declaration_warns():
    r = run_code("""
MODEL M
  VARIABLE
    x AS REAL
    x AS REAL
  EQUATION
    x = 1 ;
END
""")
    assert has_msg(r, "warn_duplicate", "warn")


def test_inequality_warns_but_solves():
    r = run_code("""
MODEL M
  VARIABLE
    a AS REAL
  EQUATION
    a = 1 ;
    a < 2 ;
END
""")
    assert has_msg(r, "warn_inequality_skipped", "warn")
    assert r["results"] is not None
    assert abs(r["results"]["values"]["a"] - 1.0) < 1e-9


def test_schedule_ignored_without_derivatives():
    r = run_code("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 3 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 STEP 0.1
    END
END
""")
    assert has_msg(r, "info_schedule_ignored", "info")
    assert r["results"]["mode"] == "steady"
    assert abs(r["results"]["values"]["x"] - 3.0) < 1e-9


def test_parameter_default_missing_warns():
    r = run_code("""
MODEL M
  PARAMETER
    a AS REAL
  VARIABLE
    x AS REAL
  EQUATION
    x = a ;
END
""")
    assert has_msg(r, "warn_param_no_default", "warn")
    assert abs(r["results"]["values"]["x"]) < 1e-9


def test_model_summary_present():
    r = run_code("""
MODEL SummaryTest
  VARIABLE
    x AS REAL
  EQUATION
    x = 7 ;
END
""")
    assert has_msg(r, "info_model_summary", "info")
    assert r["model"]["name"] == "SummaryTest"
