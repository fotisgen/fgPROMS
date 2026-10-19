"""SIGMA / indexed arrays tests — M12."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model, Indexed, Func
from conftest import has_msg


def rr_function(V, z, K):
    return sum(zi * (Ki - 1) / (1 + V * (Ki - 1)) for zi, Ki in zip(z, K))


def rachford_rice_reference(z=(0.40, 0.35, 0.25), K=(3.0, 0.6, 0.05)):
    lo, hi = 0.0, 1.0
    assert rr_function(lo, z, K) > 0 and rr_function(hi, z, K) < 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rr_function(mid, z, K) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_array_declaration_expands():
    m = parse_model("""
MODEL M
  PARAMETER
    z AS ARRAY(3) OF REAL
  VARIABLE
    V AS REAL DEFAULT 0.5
  EQUATION
    SIGMA(z(i) * V, i, 1, 3) = 0 ;
  SET
    z(1) := 1.0 ; z(2) := 2.0 ; z(3) := 3.0 ;
END
""")
    names = [d.names for d in m.parameters]
    assert names == [["z"]]            # parser keeps the declaration as written
    sets = {a.name for a in m.sets}
    assert sets == {"z(1)", "z(2)", "z(3)"}


def test_indexed_vs_function_disambiguation():
    m = parse_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = MAX(1, 2) + z(2) ;
END
""")
    from backend.parser.gproms_parser import BinOp
    rhs = m.equations[0].rhs
    assert isinstance(rhs, BinOp)
    assert isinstance(rhs.l, Func) and rhs.l.name == "MAX"
    assert isinstance(rhs.r, Indexed) and rhs.r.name == "z"


def test_flash_n_matches_bisection(run):
    r = run("flash_n.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v_ref = rachford_rice_reference()
    assert abs(r["results"]["values"]["v"] - v_ref) < 1e-8


def test_flash_n_uses_analytic_newton(run):
    """SIGMA expands symbolically, so the SymPy Jacobian must work."""
    r = run("flash_n.gPROMS", folder="benchmarks")
    assert r["results"]["method"].startswith("NEWTON")


def test_sigma_bounds_from_parameter():
    # the upper bound comes from a parameter, evaluated at solve time
    r = run_model("""
MODEL M
  PARAMETER
    N AS REAL DEFAULT 4
    a AS ARRAY(4) OF REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    SIGMA(a(i) * x, i, 1, N) = 8 ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["x"] - 2.0) < 1e-9   # 4*1*x = 8


def test_sigma_sum_matches_hand_calculation():
    r = run_model("""
MODEL M
  PARAMETER
    a AS ARRAY(5) OF REAL DEFAULT 1.0
  VARIABLE
    s AS REAL
  EQUATION
    s = SIGMA(a(i) * a(i), i, 1, 5) ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["s"] - 5.0) < 1e-9


def test_indexed_scalar_errors():
    # z is a plain scalar here, so z(2) is an error, not silent garbage
    r = run_model("""
MODEL M
  PARAMETER
    z AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    x = z(2) ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_not_array", "error")


def test_array_in_flowsheet_unit():
    r = run_model("""
MODEL MixArray
  PARAMETER
    c AS ARRAY(3) OF REAL
  VARIABLE
    total AS REAL
  EQUATION
    total = SIGMA(c(i), i, 1, 3) ;
  SET
    c(1) := 1.0 ; c(2) := 2.0 ; c(3) := 3.0 ;
END
PROCESS P
  UNIT
    U1 AS MixArray
  EQUATION
    U1.total = 100 ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "dof_nonzero", "error")   # 1 unknown, 2 equations
