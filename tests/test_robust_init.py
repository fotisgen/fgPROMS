"""Robust initialisation: homotopy continuation + multi-start — M42."""
import math

import pytest

from backend.engine import run_model


def test_homotopy_rescues_singular_start():
    """x^2 + y^2 = 1 and x^2 - y^2 = 0.5 has a SINGULAR Jacobian at the
    natural guess (0,0): Newton and hybr stall, and the robust
    initialisation machinery finds the solution. The answer is verified
    analytically: |x| = sqrt(0.75), |y| = sqrt(0.25)."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    x^2 + y^2 = 1 ;
    x^2 - y^2 = 0.5 ;
  INITIAL
    x = 0.0 ;
    y = 0.0 ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(abs(v["x"]) - math.sqrt(0.75)) < 1e-6
    assert abs(abs(v["y"]) - math.sqrt(0.25)) < 1e-6


def test_multistart_rescues_stiff_cliff():
    """EXP(50(x-2)) = 1 is flat (exp(-100) ~ 0) for x << 2: Newton, hybr
    and lm all stall at the zero-gradient region; the randomised
    multi-start escapes and finds x = 2."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    EXP(50 * (x - 2)) = 1 ;
    x - y = 0 ;
  INITIAL
    x = 0.0 ;
    y = 0.0 ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["x"] - 2.0) < 1e-6
    assert abs(v["y"] - 2.0) < 1e-6
    assert r["results"]["method"] in ("HOMOTOPY", "MULTISTART")


def test_impossible_system_fails_cleanly():
    """Two non-intersecting circles — every path fails and the model
    reports the honest solve_failed message (no hang, no crash)."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  EQUATION
    (x - 2)^2 + (y - 3)^2 = 1 ;
    (x + 2)^2 + (y + 3)^2 = 1 ;
  INITIAL
    x = 0.0 ;
    y = 0.0 ;
END
""")
    assert r["results"]["success"] is False
    assert any(m["code"] == "solve_failed" for m in r["messages"])


def test_homotopy_keeps_analytic_newton_first():
    """Well-conditioned problems still take the fast NEWTON path — the
    robust machinery only fires on failure."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x^2 = 4 ;
  INITIAL
    x = 3.0 ;
END
""")
    assert r["results"]["success"]
    assert r["results"]["method"] == "NEWTON"
