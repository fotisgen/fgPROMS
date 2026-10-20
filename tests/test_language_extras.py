"""STRING parameters, SELECTOR, WITHIN — M36."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model
from conftest import load_model


def test_string_parameter_in_property_call():
    """fluid AS STRING DEFAULT 'Water' — the parameter substitutes into
    property calls; the result equals the literal call."""
    r = run_model("""
MODEL M
  PARAMETER
    fluid AS STRING DEFAULT 'Water'
    T AS REAL DEFAULT 300.0
  VARIABLE
    cp1 AS REAL
    cp2 AS REAL
  EQUATION
    cp1 = CP(fluid, T, 1.0E5) ;
    cp2 = CP('Water', T, 1.0E5) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["cp1"] - v["cp2"]) < 1e-9
    assert v["cp1"] > 4000.0     # water cp ~ 4180


def test_string_parameter_switch_fluid():
    """Changing the string default switches the fluid — ethanol has a
    different cp."""
    r_water = run_model("""
MODEL M
  PARAMETER
    fluid AS STRING DEFAULT 'Water'
    T AS REAL DEFAULT 300.0
  VARIABLE
    cp AS REAL
  EQUATION
    cp = CP(fluid, T, 1.0E5) ;
END
""")
    r_eth = run_model("""
MODEL M
  PARAMETER
    fluid AS STRING DEFAULT 'Ethanol'
    T AS REAL DEFAULT 300.0
  VARIABLE
    cp AS REAL
  EQUATION
    cp = CP(fluid, T, 1.0E5) ;
END
""")
    assert r_eth["results"]["success"]
    assert abs(r_water["results"]["values"]["cp"]
               - r_eth["results"]["values"]["cp"]) > 100.0


def test_selector_backend_injection():
    """SELECTOR therm AS 'PR' makes every property call use the PR
    backend — identical to an explicit 'PR' argument."""
    r = run_model("""
MODEL M
  SELECTOR therm AS 'PR' ;
  PARAMETER
    T AS REAL DEFAULT 400.0
    P AS REAL DEFAULT 2.0E6
  VARIABLE
    cpSel AS REAL
    cpExp AS REAL
  EQUATION
    cpSel = CP('n-Butane', T, P) ;
    cpExp = CP('n-Butane', T, P, 'PR') ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["cpsel"] - v["cpexp"]) < 1e-9
    # and it differs from the reference HEOS value
    r2 = run_model("""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 400.0
    P AS REAL DEFAULT 2.0E6
  VARIABLE
    cp AS REAL
  EQUATION
    cp = CP('n-Butane', T, P) ;
END
""")
    assert abs(v["cpsel"] - r2["results"]["values"]["cp"]) > 1.0


def test_within_parses_and_solves():
    """WITHIN window parses; the schedule integrates normally."""
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 2.0
  VARIABLE
    x AS REAL
  EQUATION
    $x = a ;
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 STEP 0.25 WITHIN 0 TO 5
    END
END
""")
    assert r["results"]["success"]
    assert abs(r["timeSeries"]["variables"]["x"][-1] - 10.0) < 1e-6


def test_within_window_parse_structure():
    m = parse_model("""
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
      CONTINUE FOR t := 0 TO 5 STEP 0.5 WITHIN 1 TO 4
    END
END
""")
    st = m.schedule[0]
    assert st.within is not None
    assert st.within[0].v == 1.0
    assert st.within[1].v == 4.0
