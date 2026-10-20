"""Transport properties (K, ST) + RESTART — M43."""
import math

import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model
from conftest import load_model


def test_transport_properties_match_coolprop():
    """K (thermal conductivity) and ST (surface tension) match CoolProp
    directly."""
    from CoolProp.CoolProp import PropsSI
    r = run_model("""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 350.0
    P AS REAL DEFAULT 2.0E5
  VARIABLE
    k  AS REAL
    st AS REAL
  EQUATION
    k  = COND('Water', T, P) ;
    st = ST('Water', T) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    k_ref = PropsSI("L", "T", 350.0, "P", 2e5, "Water")
    st_ref = PropsSI("I", "T", 350.0, "Q", 0.0, "Water")
    assert abs(v["k"] - k_ref) < 1e-3
    assert abs(v["st"] - st_ref) < 1e-3
    # sanity: water conductivity ~ 0.67 W/m/K, surface tension ~ 0.06 N/m
    assert 0.5 < v["k"] < 0.8
    assert 0.03 < v["st"] < 0.09


def test_transport_conductivity_drops_in_vapour():
    """Water thermal conductivity collapses from the liquid to the vapour
    phase (300 K liquid ~ 0.61, 500 K vapour ~ 0.03)."""
    r1 = run_model("""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 300.0
  VARIABLE
    k AS REAL
  EQUATION
    k = COND('Water', T, 1.0E5) ;
END
""")
    r2 = run_model("""
MODEL M
  PARAMETER
    T AS REAL DEFAULT 500.0
  VARIABLE
    k AS REAL
  EQUATION
    k = COND('Water', T, 1.0E5) ;
END
""")
    k_liq = r1["results"]["values"]["k"]
    k_vap = r2["results"]["values"]["k"]
    assert k_liq > 10 * k_vap


def test_restart_parses():
    m = parse_model("""
MODEL M
  VARIABLE
    x AS REAL
  TASK
    PHASE A
      $x = 1 ;
    END WHEN x > 5 ;
    RESTART ;
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 8 WITH A STEP 0.1
    END
END
""")
    assert m.task_phases["a"].restart is True


def test_restart_resets_state_at_event():
    """The counter grows to 5, RESTART resets it to 0, it grows again —
    the saw-tooth pattern is visible in the series."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  TASK
    PHASE A
      $x = 1 ;
    END WHEN x > 4.9 ;
    RESTART ;
    PHASE B
      $x = 1 ;
    END
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 WITH A STEP 0.1
      CONTINUE FOR t := 5 TO 8 WITH B STEP 0.1
    END
END
""")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    x = ts["variables"]["x"]
    t = ts["t"]
    # find the restart jump: x drops back to 0 around t = 4.9
    jumps = [(t[i], x[i]) for i in range(1, len(t))
             if x[i] < x[i - 1] - 1.0]
    assert jumps, "no restart jump recorded"
    t_jump, x_after = jumps[0]
    assert abs(t_jump - 4.9) < 1e-2
    assert abs(x_after) < 1e-9
    # after the reset the counter grows again (phase B chains from the
    # event at 4.9 to t = 8 -> 3.1 s of growth)
    assert abs(x[-1] - 3.1) < 1e-2
    assert max(x) > 4.8
