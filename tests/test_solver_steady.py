"""Steady-state solver tests against analytical / reference solutions."""
import pytest

from backend.engine import run_model
from conftest import load_model, has_msg


def rr_function(V, z, K):
    return sum(zi * (Ki - 1) / (1 + V * (Ki - 1)) for zi, Ki in zip(z, K))


def rachford_rice_reference(z=(0.40, 0.35, 0.25), K=(3.0, 0.6, 0.05)):
    """Independent bisection solution of the Rachford-Rice equation."""
    lo, hi = 0.0, 1.0
    assert rr_function(lo, z, K) > 0 and rr_function(hi, z, K) < 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rr_function(mid, z, K) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_cstr_exact_solution(run):
    r = run("cstr.gPROMS", folder="examples")
    assert r["results"]["success"]
    v = r["results"]["values"]
    # analytical: CA = CAin*Fin/(Fin + V*k), r = k*CA, Fout = Fin
    assert abs(v["fout"] - 1.0) < 1e-9
    assert abs(v["ca"] - 1.0 / 3.0) < 1e-9
    assert abs(v["r"] - 1.0 / 6.0) < 1e-9
    assert has_msg(r, "solved_ok", "info")


def test_heat_exchanger_residuals_tiny(run):
    r = run("heat_exchanger.gPROMS", folder="examples")
    assert r["results"]["success"]
    assert r["results"]["maxResidual"] < 1e-8


def test_heat_exchanger_energy_identity(run):
    r = run("heat_exchanger.gPROMS", folder="examples")
    v = r["results"]["values"]
    params = {p["name"]: p["value"] for p in r["model"]["parameters"]}
    q_hot = params["mh"] * params["cph"] * (params["Th_in"] - v["th_out"])
    q_cold = params["mc"] * params["cpc"] * (v["tc_out"] - params["Tc_in"])
    assert abs(q_hot - v["q"]) < 1e-6 * abs(v["q"])
    assert abs(q_cold - v["q"]) < 1e-6 * abs(v["q"])


def test_rachford_rice_vs_bisection(run):
    r = run("rachford_rice.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v_ref = rachford_rice_reference()
    v = r["results"]["values"]["v"]
    assert abs(v - v_ref) < 1e-8
    # residual of the Rachford-Rice equation itself
    assert r["results"]["maxResidual"] < 1e-9


def test_rachford_rice_terms_sum_zero(run):
    r = run("rachford_rice.gPROMS", folder="benchmarks")
    v = r["results"]["values"]
    assert abs(v["t1"] + v["t2"] + v["t3"]) < 1e-9


def test_initial_section_used_as_guess():
    code = load_model("heat_exchanger.gPROMS", folder="examples")
    # same model, but with INITIAL guesses provided
    code = code.replace(
        "  VARIABLE",
        "  INITIAL\n    Q = 300000 ;\n    Th_out = 42 ;\n    Tc_out = 41 ;\n    dTlm = 32 ;\n  VARIABLE")
    a = run_model(code)
    b = run_model(load_model("heat_exchanger.gPROMS", folder="examples"))
    assert a["results"]["success"] and b["results"]["success"]
    for k in ("q", "th_out", "tc_out", "dtlm"):
        assert abs(a["results"]["values"][k] - b["results"]["values"][k]) < 1e-6


def test_steady_success_message_details(run):
    r = run("cstr.gPROMS", folder="examples")
    msgs = {m["code"]: m for m in r["messages"]}
    assert "solved_ok" in msgs
    assert msgs["solved_ok"]["level"] == "info"
    assert msgs["solved_ok"]["args"]["n"] == 3


# ---------------- M4: analytic Jacobians (SymPy) ----------------
def test_analytic_newton_used_for_steady_benchmarks(run):
    cases = [("cstr.gPROMS", "examples"),
             ("heat_exchanger.gPROMS", "examples"),
             ("rachford_rice.gPROMS", "benchmarks")]
    for name, folder in cases:
        r = run(name, folder=folder)
        assert r["results"]["success"]
        assert r["results"]["method"].startswith("NEWTON"), \
            f"{name}: expected analytic Newton, got {r['results']['method']}"


def test_jacobian_matches_numerical():
    import numpy as np

    from backend.parser.gproms_parser import parse_model
    from backend.analyzer.analyzer import analyze
    from backend.solver.solver import compile_model, residuals
    from backend.solver.jacobian import build_jacobian

    model = parse_model(load_model("rachford_rice.gPROMS", "benchmarks"))
    info = analyze(model)
    compiled = compile_model(model)
    names = info["_unknowns"]
    base = info["_fixed_values"]

    jac = build_jacobian(compiled, names, base)
    assert jac is not None

    x0 = np.array([info["_initial_guess"][n] for n in names])
    J_an = np.asarray(jac(x0), dtype=float)

    def f(x):
        env = dict(base)
        for n, v in zip(names, x):
            env[n] = float(v)
        r, _ = residuals(compiled, env)
        return r

    eps = 1e-6
    J_num = np.zeros((len(names), len(names)))
    for j in range(len(names)):
        xp = x0.copy()
        xp[j] += eps
        xm = x0.copy()
        xm[j] -= eps
        J_num[:, j] = (f(xp) - f(xm)) / (2 * eps)

    assert np.allclose(J_an, J_num, atol=1e-6)


def test_nondifferentiable_function_handled():
    """SIGN is not symbolically differentiable — the hybrid Jacobian gives
    it a numeric central-difference row; either way the result must be
    correct: SIGN(x) = 1 -> x > 0."""
    from backend.engine import run_model
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    SIGN(x) = 1 ;
END
""")
    assert r["results"]["success"]
    assert r["results"]["values"]["x"] > 0.0


def test_identical_results_with_and_without_jacobian():
    """Results must not depend on which solver path was taken."""
    import numpy as np

    from backend.parser.gproms_parser import parse_model
    from backend.analyzer.analyzer import analyze
    from backend.solver.solver import compile_model, residuals, _root_solve

    model = parse_model(load_model("heat_exchanger.gPROMS", "examples"))
    info = analyze(model)
    compiled = compile_model(model)
    names = info["_unknowns"]
    base = info["_fixed_values"]
    guesses = info["_initial_guess"]

    ok_a, sol_a, method_a, _ = _root_solve(names, guesses, base, compiled)
    assert ok_a and method_a == "NEWTON"

    # force the numeric path by re-solving with the jacobian disabled
    ok_b, sol_b, method_b, _ = _root_solve(names, guesses, base, compiled,
                                           _disable_newton=True)
    assert ok_b
    assert np.allclose(sol_a, sol_b, atol=1e-8)
