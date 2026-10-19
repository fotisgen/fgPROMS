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
    assert err < 1e-3, f"BDF2 error {err:.3%} too large"


def test_bdf2_second_order_convergence():
    """Fixed-step BDF2: halving the step should quarter the error (~4x ratio,
    proving 2nd order — the old implicit Euler gave ~2x)."""
    from backend.parser.gproms_parser import parse_model
    from backend.analyzer.analyzer import analyze
    from backend.solver.solver import solve_dynamic

    code = load_model("exp_decay.gPROMS", folder="benchmarks")
    errs = []
    for step in ("0.1", "0.05"):
        model = parse_model(code.replace("STEP 0.01", f"STEP {step}"))
        info = analyze(model)
        res = solve_dynamic(model, info, fixed_h=float(step))
        assert res["success"]
        x = res["series"]["variables"]["x"]
        t = res["series"]["t"]
        errs.append(abs(x[-1] - math.exp(-0.5 * t[-1])) / math.exp(-0.5 * t[-1]))
    ratio = errs[0] / errs[1]
    assert ratio > 2.5, f"expected ~4x error ratio for 2nd order, got {ratio:.2f}"
    assert ratio < 6.5, f"error ratio {ratio:.2f} suspiciously large"


def test_adaptive_solver_improves_coarse_schedule():
    """Even with a coarse STEP (0.5), the adaptive BDF2 shrinks the step
    where needed and stays accurate — the fixed implicit Euler at h=0.5
    had ~2.7% error here."""
    code = load_model("batch_reactor.gPROMS", folder="examples")
    r = run_model(code.replace("STEP 0.5", "STEP 0.5"))
    assert r["results"]["success"]
    assert r["results"]["method"] == "BDF2 adaptive"
    ca = r["timeSeries"]["variables"]["ca"][-1]
    exact = (1.0 + 0.5 * 0.1 * 30.0) ** -2
    assert abs(ca - exact) / exact < 0.005
    assert len(r["timeSeries"]["t"]) > 60   # at least one point per 0.5


def test_batch_reactor_analytical(run):
    r = run("batch_reactor.gPROMS", folder="examples")
    assert r["results"]["success"]
    ca = r["timeSeries"]["variables"]["ca"][-1]
    # analytical: (CA0^-0.5 + 0.5*k*t)^-2  with k=0.1, n=1.5, t=30
    exact = (1.0 + 0.5 * 0.1 * 30.0) ** -2
    assert abs(ca - exact) / exact < 0.005


def test_tank_consistent_initialisation(run):
    r = run("tank.gPROMS", folder="examples")
    assert r["results"]["success"]
    assert has_msg(r, "info_consistent_init", "info")
    f0 = r["timeSeries"]["variables"]["fout"][0]
    assert abs(f0 - 0.4 * math.sqrt(1.0)) < 1e-9


def test_tank_mass_balance_continuous(run):
    """Continuous mass balance over the whole run (trapezoidal integral):
    A*(h_end - h_0) = INT(Fin - Fout) dt within the solver tolerance."""
    import numpy as np

    r = run("tank.gPROMS", folder="examples")
    ts = r["timeSeries"]
    h_vals = np.array(ts["variables"]["h"])
    f_vals = np.array(ts["variables"]["fout"])
    t_vals = np.array(ts["t"])
    params = {p["name"]: p["value"] for p in r["model"]["parameters"]}
    A, Fin = params["A"], params["Fin"]
    dt = np.diff(t_vals)
    integ = float(np.sum(0.5 * ((Fin - f_vals[:-1]) + (Fin - f_vals[1:])) * dt))
    lhs = A * (h_vals[-1] - h_vals[0])
    assert abs(lhs - integ) < 1e-3 * max(1.0, abs(integ))


def test_tank_level_rises_physically(run):
    r = run("tank.gPROMS", folder="examples")
    h_vals = r["timeSeries"]["variables"]["h"]
    assert h_vals[-1] > h_vals[0]          # inflow > initial outflow
    assert h_vals[-1] < 2.0                # below steady state (0.5=0.4*sqrt(h))


def test_van_der_pol_vs_scipy(run):
    """BDF2 with adaptive steps keeps phase error small; compare the whole
    trajectory via mean absolute error against a tight RK45 reference."""
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
    assert mae_x < 0.005, f"x MAE {mae_x:.3f} too large"
    assert mae_y < 0.01, f"y MAE {mae_y:.3f} too large"


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
