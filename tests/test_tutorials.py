"""Tutorials must be runnable: every ```gproms block in docs/ is executed
and must produce a successful result. This keeps the documentation honest —
if a tutorial snippet breaks, the test suite catches it."""
import re
from pathlib import Path

import pytest

from backend.engine import run_model

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _extract_blocks(text: str):
    return re.findall(r"```gproms\n(.*?)```", text, flags=re.DOTALL)


@pytest.mark.parametrize("path", sorted(DOCS.glob("tutorial-*.md")),
                         ids=lambda p: p.name)
def test_tutorial_blocks_run(path):
    text = path.read_text(encoding="utf-8")
    blocks = _extract_blocks(text)
    assert blocks, f"{path.name}: no gproms code blocks found"
    for i, block in enumerate(blocks):
        r = run_model(block)
        errors = [m for m in r["messages"] if m["level"] == "error"]
        assert not errors, (
            f"{path.name} block #{i+1}: errors: "
            f"{[m['code'] for m in errors]}")
        assert r["results"] is not None, (
            f"{path.name} block #{i+1}: no results produced")
        assert r["results"]["success"], (
            f"{path.name} block #{i+1}: solver did not converge")


def test_tutorial_results_match_claims():
    """Spot-check the numeric claims made in the tutorial texts."""
    text = (DOCS / "tutorial-01-first-model.md").read_text(encoding="utf-8")
    block = _extract_blocks(text)[0]
    r = run_model(block)
    v = r["results"]["values"]
    assert abs(v["ca"] - 1.0 / 3.0) < 1e-9          # "CA = 0.3333"
    assert abs(v["fout"] - 1.0) < 1e-9              # "Fout = 1"
    assert abs(v["r"] - 1.0 / 6.0) < 1e-9           # "r = 0.1667"

    text = (DOCS / "tutorial-04-flowsheet.md").read_text(encoding="utf-8")
    r = run_model(_extract_blocks(text)[0])
    v = r["results"]["values"]
    assert abs(v["mx.fout"] - 2.0) < 1e-9           # "Fout = 2"
    assert abs(v["mx.cout"] - 1.7) < 1e-9           # "Cout = 1.7"
    assert abs(v["sp.f1"] - 0.6) < 1e-9             # "SP.F1 = 0.6"
    assert abs(v["sp.f2"] - 1.4) < 1e-9             # "SP.F2 = 1.4"

    text = (DOCS / "tutorial-03-optimization.md").read_text(encoding="utf-8")
    r = run_model(_extract_blocks(text)[0])
    assert abs(r["results"]["objective"]["value"] - 1.818181818) < 1e-5
    assert abs(r["results"]["values"]["v"] - 20.0) < 1e-4
