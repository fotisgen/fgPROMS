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


# ---------------- M19: event-driven phases (END WHEN) ----------------
def test_parser_end_when():
    m = parse_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE One
      $x = a ;
    END WHEN x > 10 ;
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 100 WITH One STEP 0.1
    END
END
""")
    ph = m.task_phases["one"]
    assert ph.when is not None
    from backend.parser.gproms_parser import BinOp
    assert isinstance(ph.when, BinOp) and ph.when.op == ">"


def test_event_batch_analytical(run):
    r = run("event_batch.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    t = ts["t"]
    T = ts["variables"]["t"]

    # heating event at t = 500 s with T = 350 exactly
    i_peak = max(range(len(T)), key=lambda i: T[i])
    assert abs(t[i_peak] - 500.0) < 0.2, f"heat event at {t[i_peak]}"
    assert abs(T[i_peak] - 350.0) < 0.001
    # cooling event at t = 700 s with T = 340
    t_end = t[-1]
    assert abs(t_end - 700.0) < 0.2, f"cool event at {t_end}"
    assert abs(T[-1] - 340.0) < 0.001
    # simulation stops far before the schedule max (2000 s)
    assert t_end < 800.0


def test_event_phase_ends_before_schedule_max(run):
    r = run("event_batch.gPROMS", folder="benchmarks")
    phases = {p["name"]: p for p in r["model"]["phases"]}
    assert phases["heat"]["event"] is True
    assert phases["cool"]["event"] is True
    # the schedule "end" is the max window; the run finished earlier
    assert r["timeSeries"]["t"][-1] < phases["cool"]["end"]


def test_event_never_fires_runs_to_max():
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE One
      $x = a ;
    END WHEN x > 1.0E9 ;
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 5 WITH One STEP 0.1
    END
END
""")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    assert abs(ts["t"][-1] - 5.0) < 1e-9       # ran the full window
    assert abs(ts["variables"]["x"][-1] - 5.0) < 1e-3


def test_event_chains_phases_continuously():
    """The state at the event is the initial state of the next phase —
    temperature continuity across the switch."""
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  TASK
    PHASE Up
      $x = a ;
    END WHEN x > 3 ;
    PHASE Down
      $x = -2 * a ;
    END WHEN x < 1 ;
  END
  INITIAL
    x = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 100 WITH Up STEP 0.05
      CONTINUE FOR t := 100 TO 200 WITH Down STEP 0.05
    END
END
""")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    t = ts["t"]
    x = ts["variables"]["x"]
    i_peak = max(range(len(x)), key=lambda i: x[i])
    assert abs(t[i_peak] - 3.0) < 0.05      # event at x = 3
    assert abs(x[i_peak] - 3.0) < 0.001
    # second phase: from 3 down to 1 at rate 2 -> 1 s of integration
    assert abs(t[-1] - 4.0) < 0.05
    assert abs(x[-1] - 1.0) < 0.001
    # monotonic up then monotonic down
    ups = x[:i_peak + 1]
    downs = x[i_peak:]
    assert all(ups[j + 1] > ups[j] - 1e-12 for j in range(len(ups) - 1))
    assert all(downs[j + 1] < downs[j] + 1e-12 for j in range(len(downs) - 1))
