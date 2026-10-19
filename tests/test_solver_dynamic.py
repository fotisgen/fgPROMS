"""Dynamic solver tests against analytical solutions and scipy references."""
import math

import pytest
from scipy.integrate import solve_ivp

from backend.engine import run_model
from conftest import load_model, has_msg


def test_exp_decay_analytical(run):
    r = run("exp_decay.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    x = r["timeSeries"]["variables"]["x"]
    t = r["timeSeries"]["t"]
    x_exact = math.exp(-0.5 * t[-1])
    err = abs(x[-1] - x_exact) / x_exact
    assert err < 0.025, f"implicit Euler error {err:.3%} too large"


def test_exp_decay_first_order_convergence():
    code = load_model("exp_decay.gPROMS", folder="benchmarks")
    errs = []
    for step in ("0.01", "0.02"):
        r = run_model(code.replace("STEP 0.01", f"STEP {step}"))
        assert r["results"]["success"]
        x = r["timeSeries"]["variables"]["x"]
        t = r["timeSeries"]["t"]
        errs.append(abs(x[-1] - math.exp(-0.5 * t[-1])) / math.exp(-0.5 * t[-1]))
    # halving the step should roughly halve the error (first-order method)
    ratio = errs[1] / errs[0]
    assert ratio > 1.4, f"expected ~2x error ratio, got {ratio:.2f}"


def test_batch_reactor_analytical(run):
    r = run("batch_reactor.gPROMS", folder="examples")
    assert r["results"]["success"]
    ca = r["timeSeries"]["variables"]["ca"][-1]
    # analytical: (CA0^-0.5 + 0.5*k*t)^-2  with k=0.1, n=1.5, t=30
    exact = (1.0 + 0.5 * 0.1 * 30.0) ** -2
    assert abs(ca - exact) / exact < 0.05


def test_tank_consistent_initialisation(run):
    r = run("tank.gPROMS", folder="examples")
    assert r["results"]["success"]
    assert has_msg(r, "info_consistent_init", "info")
    f0 = r["timeSeries"]["variables"]["fout"][0]
    assert abs(f0 - 0.4 * math.sqrt(1.0)) < 1e-9


def test_tank_mass_balance_discrete(run):
    """The discretised step relation must hold exactly:
    A*(h[n+1]-h[n]) = h*(Fin - Fout[n+1])  for every step."""
    r = run("tank.gPROMS", folder="examples")
    ts = r["timeSeries"]
    h_vals = ts["variables"]["h"]
    f_vals = ts["variables"]["fout"]
    params = {p["name"]: p["value"] for p in r["model"]["parameters"]}
    A, Fin = params["A"], params["Fin"]
    h = ts["t"][1] - ts["t"][0]
    for n in range(len(h_vals) - 1):
        lhs = A * (h_vals[n + 1] - h_vals[n])
        rhs = h * (Fin - f_vals[n + 1])
        assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(rhs))


def test_tank_level_rises_physically(run):
    r = run("tank.gPROMS", folder="examples")
    h_vals = r["timeSeries"]["variables"]["h"]
    assert h_vals[-1] > h_vals[0]          # inflow > initial outflow
    assert h_vals[-1] < 2.0                # below steady state (0.5=0.4*sqrt(h))


def test_van_der_pol_vs_scipy(run):
    """Implicit Euler accumulates phase error on this oscillator (expected for
    a first-order method), so we compare the whole trajectory via mean
    absolute error instead of a single time point."""
    import numpy as np

    r = run("van_der_pol.gPROMS", folder="benchmarks")
    assert r["results"]["success"]

    def vdp(t, s):
        x, y = s
        return [y, 2.0 * (1 - x ** 2) * y - x]

    ref = solve_ivp(vdp, [0, 10], [2.0, 0.0], rtol=1e-9, atol=1e-12, dense_output=True)

    ts = r["timeSeries"]
    t = np.array(ts["t"])
    ref_x = ref.sol(t)[0]
    ref_y = ref.sol(t)[1]
    mae_x = float(np.mean(np.abs(np.array(ts["variables"]["x"]) - ref_x)))
    mae_y = float(np.mean(np.abs(np.array(ts["variables"]["y"]) - ref_y)))
    # amplitude of the limit cycle is ~2; 0.15 is ~7% of it
    assert mae_x < 0.15, f"x MAE {mae_x:.3f} too large"
    assert mae_y < 0.30, f"y MAE {mae_y:.3f} too large"


def test_van_der_pol_stays_bounded(run):
    r = run("van_der_pol.gPROMS", folder="benchmarks")
    xs = r["timeSeries"]["variables"]["x"]
    assert all(abs(x) < 2.5 for x in xs)      # stays inside the limit cycle


def test_dynamic_values_are_final_state(run):
    r = run("tank.gPROMS", folder="examples")
    ts = r["timeSeries"]
    v = r["results"]["values"]
    for name in ("h", "c", "fout"):
        assert abs(v[name] - ts["variables"][name][-1]) < 1e-12


def test_dynamic_ok_message(run):
    r = run("tank.gPROMS", folder="examples")
    assert has_msg(r, "dynamic_ok", "info")
