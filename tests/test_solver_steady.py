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
