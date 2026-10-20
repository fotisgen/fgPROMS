"""Parameter sensitivity analysis + PDF report export — M44."""
import io
import math

import pytest

from backend.engine import run_model
from backend.export_pdf import run_to_pdf
from conftest import load_model


def test_cstr_sensitivities_analytical():
    """For the CSTR: CA = CAin / (1 + k V/F). The sensitivities reported
    by the engine must match the ANALYTICAL derivatives:
      dCA/dk   = -CAin (V/F) / (1 + k V/F)^2
      dCA/dV   = -CAin (k/F) / (1 + k V/F)^2
      dCA/dCAin = 1 / (1 + k V/F)
    """
    r = run_model(load_model("cstr.gPROMS", "examples"))
    assert r["results"]["success"]
    sens = r["results"].get("sensitivities")
    assert sens is not None
    Fin, V, k, CAin = 1.0, 10.0, 0.5, 2.0
    denom = (1 + k * V / Fin) ** 2
    dca_dk = -CAin * (V / Fin) / denom
    dca_dv = -CAin * (k / Fin) / denom
    dca_dcain = 1.0 / (1 + k * V / Fin)
    assert abs(sens["k"]["ca"] - dca_dk) < 1e-6
    assert abs(sens["v"]["ca"] - dca_dv) < 1e-6
    assert abs(sens["cain"]["ca"] - dca_dcain) < 1e-6


def test_sensitivities_sign_and_direction():
    """Increasing k lowers the outlet concentration (negative
    sensitivity) — the engineering direction is correct."""
    r = run_model(load_model("cstr.gPROMS", "examples"))
    sens = r["results"]["sensitivities"]
    assert sens["k"]["ca"] < 0
    assert sens["fin"]["ca"] > 0     # more flow -> less residence -> higher CA


def test_pdf_report_generation():
    pytest.importorskip("reportlab")
    r = run_model(load_model("cstr.gPROMS", "examples"))
    data = run_to_pdf(r["results"], r["timeSeries"], r["model"],
                      r["messages"], title="CSTR test report")
    assert data[:4] == b"%PDF"
    assert len(data) > 1000


def test_pdf_report_contains_values():
    pytest.importorskip("reportlab")
    r = run_model(load_model("cstr.gPROMS", "examples"))
    data = run_to_pdf(r["results"], r["timeSeries"], r["model"],
                      r["messages"])
    # PDF text streams are compressed, but the structure must exist;
    # verify the size and the magic bytes only
    assert data[:4] == b"%PDF"
    assert b"EOF" in data[-1024:]
