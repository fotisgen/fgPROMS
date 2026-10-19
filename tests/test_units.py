"""Unit (dimensional) checking tests — M27."""
import pytest

from backend.engine import run_model
from conftest import has_msg


def test_consistent_units_solve():
    """A fully dimensionally-consistent model passes and solves.
    Note: m*cp*dT is an ENERGY (J), not a power."""
    r = run_model("""
MODEL HeaterUnits
  PARAMETER
    m  AS Mass DEFAULT 1.0
    cp AS SpecificHeat DEFAULT 4200.0
    Tin AS Temperature DEFAULT 300.0
  VARIABLE
    Tout AS Temperature
    Q AS Energy
  EQUATION
    Q = m * cp * (Tout - Tin) ;
    Q = 84000.0 ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["tout"] - 320.0) < 1e-9


def test_unit_mismatch_detected():
    r = run_model("""
MODEL M
  PARAMETER
    T AS Temperature DEFAULT 300.0
  VARIABLE
    m AS Mass
  EQUATION
    m = T ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_units", "error")


def test_unit_mismatch_in_multiplication():
    r = run_model("""
MODEL M
  PARAMETER
    V AS Volume DEFAULT 1.0
    rho AS Density DEFAULT 1000.0
  VARIABLE
    m AS Mass
  EQUATION
    m = V + rho ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_units", "error")


def test_derivative_dimensions():
    """$T has dimensions of T/time — the tank valve equation
    0.4*SQRT(h) = Fout is correctly flagged (the coefficient carries units
    in reality)."""
    r = run_model("""
MODEL M
  PARAMETER
    Fin AS Flowrate DEFAULT 0.5
    A AS Area DEFAULT 2.0
  VARIABLE
    h AS Length
    Fout AS Flowrate
  EQUATION
    Fout = 0.4 * SQRT(h) ;
    $h = (Fin - Fout) / A ;
  INITIAL
    h = 1.0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 1 STEP 0.1
    END
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_units", "error")


def test_dimensionless_models_unaffected():
    """Plain REAL models are dimensionless — zero behavioural change."""
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 2.0
  VARIABLE
    x AS REAL
  EQUATION
    x = a * a ;
END
""")
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["x"] - 4.0) < 1e-9


def test_power_dimensions_checked():
    """V^2 (a volume squared) cannot be added to a Volume."""
    r = run_model("""
MODEL M
  PARAMETER
    V AS Volume DEFAULT 1.0
  VARIABLE
    x AS Volume
  EQUATION
    x = V^2 ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_units", "error")


def test_property_function_dimensions():
    """CP('Water', T, P) has dimensions of specific heat — consistent use
    passes."""
    r = run_model("""
MODEL M
  PARAMETER
    T AS Temperature DEFAULT 300.0
    P AS Pressure DEFAULT 1.0E5
    m AS Mass DEFAULT 1.0
  VARIABLE
    cp AS SpecificHeat
    Q AS Energy
    dT AS Temperature
  EQUATION
    cp = CP('Water', T, P) ;
    dT = 20.0 ;
    Q = m * cp * dT ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert v["cp"] > 4000.0          # water cp ~ 4180 J/kg/K
    assert abs(v["q"] - 1.0 * v["cp"] * 20.0) < 1e-6


def test_mixed_dimensionless_failure():
    """EXP requires a dimensionless argument — EXP(T) is caught."""
    r = run_model("""
MODEL M
  PARAMETER
    T AS Temperature DEFAULT 300.0
  VARIABLE
    x AS REAL
  EQUATION
    x = EXP(T) ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_units", "error")
