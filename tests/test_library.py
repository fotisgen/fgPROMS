"""Unit library tests — M10: every library model must solve with an
independently verified analytical solution."""
from pathlib import Path

import pytest

from backend.engine import run_model
from conftest import load_model


LIB = Path(__file__).resolve().parent.parent / "library"


def lib_model(name: str) -> str:
    return (LIB / name).read_text(encoding="utf-8")


def with_set(code: str, lines: str) -> str:
    """Insert a SET section before the model's closing END."""
    code = code.rstrip()
    assert code.endswith("END")
    return code[:-3].rstrip() + "\n  SET\n" + lines + "END\n"


def rr_function(V, z, K):
    return sum(zi * (Ki - 1) / (1 + V * (Ki - 1)) for zi, Ki in zip(z, K))


def rachford_rice_reference(z=(0.40, 0.35, 0.25), K=(3.0, 0.6, 0.05)):
    lo, hi = 0.0, 1.0
    assert rr_function(lo, z, K) > 0 and rr_function(hi, z, K) < 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rr_function(mid, z, K) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_heater_analytical():
    code = with_set(lib_model("heater.gPROMS"),
                    "    m := 1.0 ;\n    Tin := 300.0 ;\n    Tout := 340.0 ;")
    r = run_model(code)
    assert r["results"]["success"]
    q = r["results"]["values"]["q"]
    assert abs(q - 1.0 * 4200.0 * 40.0) < 1e-6


def test_pump_analytical():
    code = with_set(lib_model("pump.gPROMS"),
                    "    m := 2.0 ;\n    dP := 150000.0 ;")
    r = run_model(code)
    assert r["results"]["success"]
    w = r["results"]["values"]["w"]
    assert abs(w - 2.0 * 150000.0 / (1000.0 * 0.75)) < 1e-6


def test_mixer_analytical():
    code = with_set(lib_model("mixer.gPROMS"),
                    "    F1 := 1.5 ;\n    F2 := 0.5 ;\n    C1 := 2.0 ;\n    C2 := 0.8 ;")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["fout"] - 2.0) < 1e-9
    assert abs(v["cout"] - (1.5 * 2.0 + 0.5 * 0.8) / 2.0) < 1e-9


def test_splitter_analytical():
    code = with_set(lib_model("splitter.gPROMS"), "    Fin := 2.0 ;")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["f1"] - 0.6) < 1e-9
    assert abs(v["f2"] - 1.4) < 1e-9


def test_flash_vs_bisection():
    r = run_model(lib_model("flash.gPROMS"))
    assert r["results"]["success"]
    v_ref = rachford_rice_reference()
    assert abs(r["results"]["values"]["v"] - v_ref) < 1e-8


def test_mixer_splitter_flowsheet():
    r = run_model(load_model("mixer_splitter.gPROMS", "examples"))
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["mx.fout"] - 2.0) < 1e-9
    assert abs(v["mx.cout"] - 1.7) < 1e-9
    assert abs(v["sp.f1"] - 0.6) < 1e-9
    assert abs(v["sp.f2"] - 1.4) < 1e-9
    assert r["model"]["isProcess"] is True
    assert [u["name"] for u in r["model"]["units"]] == ["MX", "SP"]


def test_flowsheet_unit_set_parameter():
    code = load_model("mixer_splitter.gPROMS", "examples")
    # increase the split fraction via a unit-level SET
    code = code.replace("SP AS Splitter", "SP AS Splitter SET s := 0.5")
    r = run_model(code)
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["sp.f1"] - 1.0) < 1e-9
    assert abs(v["sp.f2"] - 1.0) < 1e-9
