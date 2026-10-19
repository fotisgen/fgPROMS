"""Flash drum + Maxwell thermodynamic consistency tests — M28."""
from pathlib import Path

import pytest

from backend.engine import run_model
from conftest import load_model, has_msg

LIB = Path(__file__).resolve().parent.parent / "library"


def with_set(code: str, lines: str) -> str:
    code = code.rstrip()
    assert code.endswith("END")
    return code[:-3].rstrip() + "\n  SET\n" + lines + "END\n"


def test_flash_drum_analytical(run):
    """The drum at T=270 K, P=2 bar, z=[0.4, 0.6] is two-phase — the
    computed flows split according to the vapour fraction."""
    from CoolProp.CoolProp import AbstractState, PT_INPUTS
    code = with_set((LIB / "flash_drum.gPROMS").read_text(encoding="utf-8"),
                    "    F := 1.0 ;\n    z(2) := 0.6 ;\n    Qduty := 0.0 ;")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    AS = AbstractState("HEOS", "n-Butane&Propane")
    AS.set_mole_fractions([0.4, 0.6])
    AS.update(PT_INPUTS, 2e5, 270.0)
    q_ref = float(AS.Q())
    assert abs(v["q"] - q_ref) < 1e-6
    assert 0.0 < v["q"] < 1.0
    assert abs(v["v"] - v["q"]) < 1e-9          # F = 1
    assert abs(v["l"] - (1.0 - v["q"])) < 1e-9
    assert abs(v["hf"] - float(AS.hmass())) < 1e-3   # mass-specific J/kg


def test_flash_drum_duty_can_be_set():
    """The drum's duty is an ordinary variable — override it freely."""
    code = with_set((LIB / "flash_drum.gPROMS").read_text(encoding="utf-8"),
                    "    F := 2.0 ;\n    z(2) := 0.6 ;\n    Qduty := 5000.0 ;")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["v"] - 2.0 * v["q"]) < 1e-9
    assert abs(v["l"] - 2.0 * (1.0 - v["q"])) < 1e-9


def test_maxwell_relation_verified(run):
    """(dh/dP)_T = v - T*(dv/dT)_P must hold numerically — this verifies
    BOTH the CoolProp engine and the modelling language."""
    r = run("maxwell_check.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    # the residual must be tiny relative to the specific volume
    rel = abs(v["err"]) / max(abs(v["v"]), 1e-12)
    assert rel < 1e-3, f"Maxwell residual {rel:.3e} too large"
    assert abs(v["lhs"]) > 0 and abs(v["rhs"]) > 0
