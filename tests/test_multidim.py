"""Multi-dimensional array tests — M17."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model, Indexed
from conftest import has_msg


def test_2d_declaration_expands():
    from backend.analyzer.analyzer import _expand_decl_names
    names = _expand_decl_names("z", "ARRAY(2, 3) OF REAL")
    assert names == ["z(1,1)", "z(1,2)", "z(1,3)", "z(2,1)", "z(2,2)", "z(2,3)"]


def test_2d_indexed_parsing():
    m = parse_model("""
MODEL M
  PARAMETER
    A AS ARRAY(2, 2) OF REAL
  VARIABLE
    x AS REAL
  EQUATION
    x = A(1, 2) + A(2, 1) ;
END
""")
    from backend.parser.gproms_parser import BinOp
    rhs = m.equations[0].rhs
    assert isinstance(rhs, BinOp)
    assert isinstance(rhs.l, Indexed)
    assert [i.v for i in rhs.l.indices] == [1.0, 2.0]


def test_matrix_sum_benchmark(run):
    r = run("matrix_sum.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["s"] - 78.0) < 1e-9
    assert abs(v["avg"] - 6.5) < 1e-9


def test_matrix_sum_uses_analytic_newton(run):
    """Nested SIGMA expands symbolically — the SymPy Jacobian must work."""
    r = run("matrix_sum.gPROMS", folder="benchmarks")
    assert r["results"]["method"].startswith("NEWTON")


def test_2d_array_in_optimization():
    r = run_model("""
MODEL M
  PARAMETER
    A AS ARRAY(2, 2) OF REAL
  VARIABLE
    x AS REAL
  EQUATION
    x = A(1, 1) + A(2, 2) ;
  SET
    A(1,1) := 1.0 ; A(1,2) := 0.0 ; A(2,1) := 0.0 ; A(2,2) := 1.0 ;
  DATA
    x : 0 : 5.0 ;
  ESTIMATE
    nothing ;
END
""")
    assert has_msg(r, "err_est_not_param", "error")


def test_2d_row_sum():
    r = run_model("""
MODEL M
  PARAMETER
    A AS ARRAY(2, 3) OF REAL
  VARIABLE
    row1 AS REAL
    row2 AS REAL
  EQUATION
    row1 = SIGMA(A(1, j), j, 1, 3) ;
    row2 = SIGMA(A(2, j), j, 1, 3) ;
  SET
    A(1,1) := 1 ; A(1,2) := 2 ; A(1,3) := 3 ;
    A(2,1) := 10 ; A(2,2) := 20 ; A(2,3) := 30 ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["row1"] - 6.0) < 1e-9
    assert abs(v["row2"] - 60.0) < 1e-9


def test_2d_indexed_scalar_direct_access():
    r = run_model("""
MODEL M
  PARAMETER
    A AS ARRAY(3, 3) OF REAL
  VARIABLE
    x AS REAL
  EQUATION
    x = A(2, 3) ;
  SET
    A(1,1) := 1 ; A(1,2) := 2 ; A(1,3) := 3 ;
    A(2,1) := 4 ; A(2,2) := 5 ; A(2,3) := 6 ;
    A(3,1) := 7 ; A(3,2) := 8 ; A(3,3) := 9 ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["x"] - 6.0) < 1e-9
