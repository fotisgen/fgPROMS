"""PID/PI level control — M38."""
import math

import pytest

from backend.engine import run_model
from conftest import load_model


def test_pi_control_returns_to_setpoint(run):
    """After the inlet disturbance, the PI controller returns the level
    to the setpoint (integral action removes offset)."""
    r = run("tank_pi_control.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    h = ts["variables"]["h"]
    t = ts["t"]
    # steady before the disturbance
    assert abs(h[0] - 5.0) < 1e-6
    # final level back at setpoint
    assert abs(h[-1] - 5.0) < 1e-2
    # the disturbance actually happened (level rose at some point)
    assert max(h) > 5.1


def test_p_only_shows_steady_offset(run):
    """P-only control (Ki = 0) leaves the analytic offset
    h = hsp + (Fin2 - Fb)/Kp = 6.0 — the classic P vs PI story."""
    code = load_model("tank_pi_control.gPROMS", "benchmarks")
    code = code.replace("Ki  AS REAL DEFAULT 0.5", "Ki  AS REAL DEFAULT 0.0")
    r = run_model(code)
    assert r["results"]["success"]
    h = r["timeSeries"]["variables"]["h"]
    offset_ref = 5.0 + (2.0 - 1.0) / 1.0
    assert abs(h[-1] - offset_ref) < 1e-2


def test_pid_derivative_reduces_overshoot(run):
    """Derivative action damps the response — the peak level after the
    disturbance is lower with Kd > 0 than with Kd = 0."""
    r_pi = run("tank_pi_control.gPROMS", folder="benchmarks")
    r_pid = run("tank_pid_control.gPROMS", folder="benchmarks")
    assert r_pid["results"]["success"]
    h_pi = r_pi["timeSeries"]["variables"]["h"]
    h_pid = r_pid["timeSeries"]["variables"]["h"]
    assert max(h_pid) < max(h_pi) - 0.01
    # both restore the setpoint
    assert abs(h_pid[-1] - 5.0) < 1e-2
    assert abs(h_pi[-1] - 5.0) < 1e-2
def test_controller_dynamics_second_order():
    """The closed loop is a damped oscillator: h'' + (Kp/A)h'
    + (Ki/A)(h - hsp) = 0 — verify the natural frequency and that the
    response is underdamped for these gains."""
    A, Kp, Ki = 2.0, 1.0, 0.5
    zeta = (Kp / A) / (2 * math.sqrt(Ki / A))     # 0.5 / 1.0 = 0.5
    assert 0 < zeta < 1                            # underdamped
    wn = math.sqrt(Ki / A)
    assert abs(wn - 0.5) < 1e-12
