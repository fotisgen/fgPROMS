"""Parameter estimation (DATA / ESTIMATE) tests — M16."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model
from conftest import has_msg


def test_parser_data_and_estimate():
    m = parse_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.2
  VARIABLE
    x AS REAL
  EQUATION
    x = k ;
  DATA
    x : 0 : 1.0, 5 : 2.0 ;
  ESTIMATE
    k ;
END
""")
    assert m.data[0].name == "x"
    assert len(m.data[0].points) == 2
    assert m.estimate == ["k"]


def test_estimation_recovers_true_parameter(run):
    r = run("estimation_batch.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    assert r["results"]["mode"] == "estimation"
    k_est = r["results"]["estimated"]["k"]
    assert abs(k_est - 0.1) < 0.01, f"estimated k={k_est}, expected ~0.1"
    # excellent fit: SSE tiny
    assert r["results"]["sse"] < 1e-6
    # every data point is matched
    for f in r["results"]["dataFit"]:
        assert abs(f["model"] - f["measured"]) < 1e-3
    assert has_msg(r, "estimate_ok", "info")


def test_estimation_reports_fit_and_series(run):
    r = run("estimation_batch.gPROMS", folder="benchmarks")
    assert len(r["results"]["dataFit"]) == 4
    assert r["timeSeries"] is not None      # final trajectory is returned
    assert r["model"]["estimated"] == ["k"]


def test_estimate_unknown_name_errors():
    r = run_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.2
  VARIABLE
    x AS REAL
  EQUATION
    x = k ;
  DATA
    x : 0 : 1.0 ;
  ESTIMATE
    nothere ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_est_not_param", "error")


def test_data_unknown_variable_errors():
    r = run_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.2
  VARIABLE
    x AS REAL
  EQUATION
    x = k ;
  DATA
    notvar : 0 : 1.0 ;
  ESTIMATE
    k ;
END
""")
    assert has_msg(r, "err_data_not_var", "error")


def test_estimate_without_data_errors():
    r = run_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.2
  VARIABLE
    x AS REAL
  EQUATION
    x = k ;
  ESTIMATE
    k ;
END
""")
    assert has_msg(r, "err_est_no_data", "error")
    assert r["results"] is None


def test_data_without_estimate_ignored():
    r = run_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.2
  VARIABLE
    x AS REAL
  EQUATION
    x = k ;
  DATA
    x : 0 : 1.0 ;
END
""")
    assert has_msg(r, "info_data_ignored", "info")
    # ordinary steady solve happens
    assert r["results"]["mode"] == "steady"
    assert abs(r["results"]["values"]["x"] - 0.2) < 1e-9


def test_estimation_with_optimization_errors():
    r = run_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.2
  VARIABLE
    x AS REAL
  EQUATION
    x = k ;
  DATA
    x : 0 : 1.0 ;
  ESTIMATE
    k ;
  OPTIMIZE
    MINIMIZE x ;
END
""")
    assert has_msg(r, "err_est_opt", "error")


def test_estimation_steady_state_fit():
    """Estimation also works on steady models: fit a parameter to
    measured variable values at zero time (single-point data)."""
    r = run_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.5
  VARIABLE
    y AS REAL
  EQUATION
    y = 2.0 * k ;
  DATA
    y : 0 : 0.6 ;
  ESTIMATE
    k ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["estimated"]["k"] - 0.3) < 1e-4


def test_estimation_bounds_respected():
    """Bounds on the estimated parameter are honoured."""
    r = run_model("""
MODEL M
  PARAMETER
    k AS REAL DEFAULT 0.5
  VARIABLE
    y AS REAL
  EQUATION
    y = 2.0 * k ;
  DATA
    y : 0 : 10.0 ;
  ESTIMATE
    k ;
  BOUNDS
    k : 0 : 1.0 ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["estimated"]["k"] - 1.0) < 1e-4   # pushed to bound
