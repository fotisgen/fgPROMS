"""FOR-loop indexed equations + phase envelope tests — M29."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model, ForEquation
from conftest import has_msg


def test_parser_for_loop():
    m = parse_model("""
MODEL M
  VARIABLE
    x AS ARRAY(3) OF REAL
  EQUATION
    FOR i := 1 TO 3 DO
      x(i) = i * 2 ;
    END
END
""")
    feq = m.equations[0]
    assert isinstance(feq, ForEquation)
    assert feq.var == "i"
    assert len(feq.eqs) == 1


def test_for_loop_expansion_solves():
    r = run_model("""
MODEL M
  PARAMETER
    N AS REAL DEFAULT 4
  VARIABLE
    x AS ARRAY(4) OF REAL
    s AS REAL
  EQUATION
    FOR i := 1 TO N DO
      x(i) = i * i ;
    END
    s = SIGMA(x(i), i, 1, N) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["x(3)"] - 9.0) < 1e-9
    assert abs(v["s"] - 30.0) < 1e-9      # 1+4+9+16


def test_nested_for_rejected_cleanly():
    r = run_model("""
MODEL M
  VARIABLE
    x AS ARRAY(2) OF REAL
    y AS ARRAY(2) OF REAL
  EQUATION
    FOR i := 1 TO 2 DO
      FOR j := 1 TO 2 DO
        x(i) = 1 ;
      END
    END
END
""")
    assert r["ok"] is True
    assert any(m["level"] == "error" for m in r["messages"])


def test_phase_envelope_values(run):
    from CoolProp.CoolProp import AbstractState, PQ_INPUTS
    r = run("phase_envelope.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    AS = AbstractState("HEOS", "n-Butane&Propane")
    for k in range(1, 21):
        Pk = 2.0e5 + (k - 1) * (8.0e5 / 19.0)
        AS.set_mole_fractions([0.4, 0.6])
        AS.update(PQ_INPUTS, Pk, 0.0)
        assert abs(v[f"tb({k})"] - float(AS.T())) < 1e-3, f"Tb({k})"
        AS.update(PQ_INPUTS, Pk, 1.0)
        assert abs(v[f"td({k})"] - float(AS.T())) < 1e-3, f"Td({k})"
        assert v[f"tb({k})"] < v[f"td({k})"]         # bubble below dew


def test_phase_envelope_monotonic():
    r = run_model(open("benchmarks/phase_envelope.gPROMS",
                       encoding="utf-8").read())
    v = r["results"]["values"]
    for name in ("tb", "td"):
        vals = [v[f"{name}({k})"] for k in range(1, 21)]
        assert all(vals[j + 1] > vals[j] for j in range(19)), name


def test_for_loop_bad_bounds():
    r = run_model("""
MODEL M
  VARIABLE
    x AS ARRAY(3) OF REAL
    q AS REAL
  EQUATION
    q = 1 ;
    FOR i := q TO 3 DO
      x(i) = 1 ;
    END
END
""")
    assert has_msg(r, "err_for_bounds", "error")
    assert r["results"] is None
