"""Task phases (TASK / WITH phase) tests — M18."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model
from conftest import has_msg


def test_parser_task_phases():
    m = parse_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE One
      $x = a ;
    END
    PHASE Two
      $x = -a ;
    END
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 WITH One STEP 0.1
      CONTINUE FOR t := 5 TO 10 WITH Two STEP 0.1
    END
END
""")
    assert set(m.task_phases) == {"one", "two"}
    assert m.schedule[0].phase == "One"
    assert m.schedule[1].phase == "Two"


def test_batch_heater_analytical(run):
    r = run("batch_heater.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    assert r["results"]["mode"] == "dynamic"
    ts = r["timeSeries"]
    t = ts["t"]
    T = ts["variables"]["t"]

    # locate the value at t=10 and t=20 (the solver output includes them)
    import numpy as np
    T10 = float(np.interp(10.0, t, T))
    T20 = float(np.interp(20.0, t, T))
    assert abs(T10 - 300.47619) < 5e-4
    assert abs(T20 - 300.23810) < 5e-4
    # peak is at t = 10 (end of heating)
    peak_idx = int(np.argmax(T))
    assert abs(t[peak_idx] - 10.0) < 0.2


def test_batch_heater_phases_reported(run):
    r = run("batch_heater.gPROMS", folder="benchmarks")
    phases = r["model"]["phases"]
    names = {p["name"] for p in phases}
    assert names == {"heat", "cool"}
    by = {p["name"]: p for p in phases}
    assert by["heat"]["start"] == 0.0 and by["heat"]["end"] == 10.0
    assert by["cool"]["start"] == 10.0 and by["cool"]["end"] == 20.0


def test_phase_dof_mismatch_errors():
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE One
      $x = a ;
      x = 0 ;
    END
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 WITH One STEP 0.1
    END
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_phase_dof", "error")


def test_unknown_phase_errors():
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE One
      $x = a ;
    END
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 WITH Missing STEP 0.1
    END
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_phase_unknown", "error")


def test_phase_gap_errors():
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE One
      $x = a ;
    END
    PHASE Two
      $x = a ;
    END
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 3 WITH One STEP 0.1
      CONTINUE FOR t := 5 TO 10 WITH Two STEP 0.1
    END
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_phase_gap", "error")


def test_unused_phase_warns():
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE One
      $x = a ;
    END
    PHASE Spare
      $x = 2 * a ;
    END
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 WITH One STEP 0.1
    END
END
""")
    assert has_msg(r, "warn_phase_unused", "warn")
    assert r["results"]["success"]
    # the unused phase's equations are never active
    import numpy as np
    ts = r["timeSeries"]
    x_end = ts["variables"]["x"][-1]
    assert abs(x_end - 5.0) < 1e-3     # integrated $x = a = 1 over t=5
