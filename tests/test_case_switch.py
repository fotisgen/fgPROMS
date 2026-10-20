"""CASE/SWITCH equations + REINITIALIZE — M34."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model, IfEquation, Num
from conftest import load_model


def test_case_parses_to_if_chain():
    m = parse_model("""
MODEL M
  PARAMETER
    mode AS REAL DEFAULT 2.0
  VARIABLE
    x AS REAL
  EQUATION
    CASE mode OF
      WHEN 1.0: x = 10 ;
      WHEN 2.0: x = 20 ;
      WHEN ELSE: x = 0 ;
    END
END
""")
    eq = m.equations[0]
    assert isinstance(eq, IfEquation)
    # the CASE cond becomes ABS(expr - val) < 1e-9 (boolean equality)
    assert eq.cond.op == "<"
    assert eq.cond.l.name == "ABS"
    # with an ELSE branch, the chain ends in an always-true IF
    tail = eq
    while tail.else_eqs and isinstance(tail.else_eqs[0], IfEquation):
        tail = tail.else_eqs[0]
    assert isinstance(tail.cond, Num)
    assert tail.cond.v == 1.0


def test_case_selects_correct_branch():
    for mode, expected in ((1.0, 10.0), (2.0, 20.0), (3.0, 0.0)):
        r = run_model(f"""
MODEL M
  PARAMETER
    mode AS REAL DEFAULT {mode}
  VARIABLE
    x AS REAL
  EQUATION
    CASE mode OF
      WHEN 1.0: x = 10 ;
      WHEN 2.0: x = 20 ;
      WHEN ELSE: x = 0 ;
    END
END
""")
        assert r["results"]["success"]
        assert abs(r["results"]["values"]["x"] - expected) < 1e-9


def test_case_without_else_leaves_x_underdetermined():
    """No WHEN ELSE: mode = 3 matches nothing — the variable stays
    under-determined (honest DOF diagnostic)."""
    r = run_model("""
MODEL M
  PARAMETER
    mode AS REAL DEFAULT 3.0
  VARIABLE
    x AS REAL
  EQUATION
    CASE mode OF
      WHEN 1.0: x = 10 ;
      WHEN 2.0: x = 20 ;
    END
END
""")
    assert r["ok"] is True
    assert any(m["level"] == "error" for m in r["messages"])
