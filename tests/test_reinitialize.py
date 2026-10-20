"""REINITIALIZE — discontinuous state reset at events (M34 part 1)."""
import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model
from conftest import load_model


def test_reinitialize_parser():
    m = parse_model(load_model("reinitialize_counter.gPROMS",
                               "benchmarks"))
    ph = m.task_phases["count"]
    assert ph.when is not None
    assert len(ph.reinits) == 1
    assert ph.reinits[0][0] == "x"


def test_reinitialize_jump_and_final_value(run):
    r = run("reinitialize_counter.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    t = ts["t"]
    x = ts["variables"]["x"]
    # locate the event: the point where x jumps 10 -> 0 at duplicate time
    jump = [i for i in range(1, len(t))
            if abs(t[i] - t[i - 1]) < 1e-9
            and abs(x[i - 1] - 10.0) < 1e-6 and abs(x[i]) < 1e-9]
    assert jump, "no 10 -> 0 reinitialisation jump recorded"
    i = jump[0]
    t_ev = t[i]
    assert abs(t_ev - 1.0) < 1e-6          # event at exactly t = 1 s
    assert abs(x[i - 1] - 10.0) < 1e-6     # x just before the event
    assert abs(x[i] - 0.0) < 1e-9          # reinitialised value
    # final value: 1 s of counting + 2 s of idle growth
    assert abs(x[-1] - 2.0) < 1e-6
    assert abs(t[-1] - 3.0) < 1e-9


def test_reinitialize_sequential_semantics():
    """Two reinits in one phase apply in order — the second sees the
    value written by the first."""
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
    y AS REAL
  TASK
    PHASE A
      $x = 1 ;
      $y = 0 ;
    END WHEN x > 5 ;
    REINITIALIZE x WITH 0.0 ;
    REINITIALIZE y WITH x + 100.0 ;
    PHASE B
      $x = 0 ;
      $y = 0 ;
    END
  END
  INITIAL
    x = 0 ;
    y = 0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 7 WITH A STEP 0.1
    END
END
""")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    x = ts["variables"]["x"]
    y = ts["variables"]["y"]
    jump = [i for i in range(1, len(ts["t"]))
            if abs(ts["t"][i] - ts["t"][i - 1]) < 1e-9]
    i = jump[0]
    # x reset to 0, then y = x + 100 sees x == 0 (sequential)
    assert abs(x[i]) < 1e-9
    assert abs(y[i] - 100.0) < 1e-6
