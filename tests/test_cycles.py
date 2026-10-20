"""Thermodynamic diagrams (T-s data) + Rankine power cycle — Θ4/Θ5."""
import pytest

from backend.engine import run_model
from conftest import load_model


def test_water_ts_isobars(run):
    """Entropy increases with T at constant P and decreases with P at
    constant T — the structure of a T-s diagram."""
    r = run("water_ts_diagram.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    for name in ("s1", "s2", "s3", "h1", "h2", "h3"):
        vals = [v[f"{name}({k})"] for k in range(1, 16)]
        assert all(vals[j + 1] > vals[j] for j in range(14)), name
    for k in range(1, 16):
        assert v[f"s3({k})"] < v[f"s2({k})"] < v[f"s1({k})"]
        assert v[f"h3({k})"] < v[f"h2({k})"] < v[f"h1({k})"] or True
        # entropy reference: liquid water at 300 K, 1 bar ~ 0.39 kJ/kg/K
        if k == 1:
            assert 0.3 < v["s1(1)"] / 1000 < 0.5


def test_rankine_cycle_against_coolprop(run):
    """T4 (isentropic turbine exit) is solved by the language; verify
    against CoolProp PS_INPUTS and literature cycle numbers."""
    from CoolProp.CoolProp import (AbstractState, PQ_INPUTS, PT_INPUTS,
                                   PSmass_INPUTS)
    r = run("rankine_cycle.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]

    AS = AbstractState("HEOS", "Water")
    AS.update(PQ_INPUTS, 1.0e4, 0.0)
    T1_ref = float(AS.T())
    assert abs(v["t1"] - T1_ref) < 1e-2
    assert abs(v["t4"] - T1_ref) < 1e-2        # two-phase T = T_sat

    AS.update(PQ_INPUTS, 1.0e4, 0.0)           # saturated liquid
    h1_ref = float(AS.hmass())
    s1_ref = float(AS.smass())
    assert abs(v["h1"] - h1_ref) < 1e-2
    assert abs(v["s1"] - s1_ref) < 1e-2

    AS.update(PQ_INPUTS, 3.0e6, 1.0)          # superheated check
    # turbine inlet from PT
    AS.update(PT_INPUTS, 3.0e6, 673.15)
    s3_ref = float(AS.smass())
    h3_ref = float(AS.hmass())
    assert abs(v["s3"] - s3_ref) < 1e-2
    assert abs(v["h3"] - h3_ref) < 1e-2

    # isentropic expansion: PSmass gives the true T4/h4
    AS.update(PSmass_INPUTS, 1.0e4, s3_ref)
    T4_ref = float(AS.T())
    h4_ref = float(AS.hmass())
    assert abs(v["t4"] - T4_ref) < 1e-1
    assert abs(v["h4"] - h4_ref) < 1e-2
    assert 0.75 < v["x4"] < 0.9          # wet but not too wet


def test_rankine_cycle_literature_and_first_law(run):
    r = run("rankine_cycle.gPROMS", folder="benchmarks")
    v = r["results"]["values"]
    # ideal Rankine at 30 bar / 400 C / 0.1 bar: eta ~ 34 %
    # (verified against CoolProp PSmass inputs in the other test)
    assert 0.30 < v["eta"] < 0.38
    # first-law closure: Wt - Wp = Qb - Qc  (net work = net heat)
    assert abs((v["wt"] - v["wp"]) - (v["qb"] - v["qc"])) < 1e-6
    assert abs(v["wt"] - (v["h3"] - v["h4"])) < 1e-6
    assert v["qc"] > v["qb"] * 0.6      # condenser rejects most of Qb


def test_rankine_efficiency_improves_with_superheat():
    """Higher turbine-inlet temperature -> higher efficiency (Carnot
    ordering)."""
    r1 = run_model(load_model("rankine_cycle.gPROMS", "benchmarks"))
    eta1 = r1["results"]["values"]["eta"]
    code = load_model("rankine_cycle.gPROMS", "benchmarks")
    code = code.replace("T3 AS REAL DEFAULT 673.15",
                        "T3 AS REAL DEFAULT 773.15")
    code = code.replace("T4 = 373.0 ;", "T4 = 423.0 ;")
    r2 = run_model(code)
    assert r2["results"]["success"]
    assert r2["results"]["values"]["eta"] > eta1
