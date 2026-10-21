"""ORDERED_SET + real WITHIN clamping — Q4."""
import pytest

from backend.engine import run_model
from conftest import load_model


def test_ordered_set_values_accessible():
    """An ORDERED_SET declares a fixed list of values accessible by index."""
    r = run_model("""
MODEL M
  PARAMETER
    P AS ORDERED_SET DEFAULT (1.0, 2.0, 5.0)
  VARIABLE
    s AS REAL
    first AS REAL
    last AS REAL
  EQUATION
    s = P(1) + P(2) + P(3) ;
    first = P(1) ;
    last = P(3) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["s"] - 8.0) < 1e-9
    assert abs(v["first"] - 1.0) < 1e-9
    assert abs(v["last"] - 5.0) < 1e-9


def test_ordered_set_in_property_calls():
    """ORDERED_SET values can feed property calls (pressure ladder)."""
    r = run_model("""
MODEL M
  PARAMETER
    P AS ORDERED_SET DEFAULT (1.0E5, 5.0E5, 1.0E6)
  VARIABLE
    cp1 AS REAL
    cp3 AS REAL
  EQUATION
    cp1 = CP('Water', 300.0, P(1)) ;
    cp3 = CP('Water', 300.0, P(3)) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["cp1"] - v["cp3"]) < 5.0    # weak pressure dependence


def test_within_clamps_phase_interval():
    """WITHIN 2 TO 8 on a phase scheduled for [0, 10] clamps the actual
    integration window to [2, 8]."""
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE A
      $x = 1 ;
    END
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 10 WITH A STEP 0.5 WITHIN 2 TO 8
    END
END
""")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    t = ts["t"]
    # integration happened inside [2, 8] only
    assert t[0] >= 2.0 - 1e-9
    assert t[-1] <= 8.0 + 1e-9
    assert abs(ts["variables"]["x"][-1] - 6.0) < 1e-6


def test_within_invalid_warns_and_ignores():
    """An inverted window warns and falls back to the schedule interval."""
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    $x = a ;
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 3 STEP 0.5 WITHIN 5 TO 1
    END
END
""")
    assert r["results"]["success"]
    assert any(m["code"] == "warn_within_invalid" for m in r["messages"])
    assert abs(r["timeSeries"]["variables"]["x"][-1] - 3.0) < 1e-6
