"""Shared fixtures and helpers for the fgPROMS test suite."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engine import run_model  # noqa: E402

EXAMPLES = ROOT / "examples"
BENCHMARKS = ROOT / "benchmarks"
LIBRARY = ROOT / "library"


def load_model(name: str, folder: str = "benchmarks") -> str:
    base = {"benchmarks": BENCHMARKS, "examples": EXAMPLES,
            "library": LIBRARY}[folder]
    return (base / name).read_text(encoding="utf-8")


@pytest.fixture
def run():
    """run(name, folder='benchmarks') -> run_model result dict."""
    def _run(name: str, folder: str = "benchmarks"):
        return run_model(load_model(name, folder))
    return _run


def msg_codes(result: dict) -> set:
    return {m["code"] for m in result.get("messages", [])}


def has_msg(result: dict, code: str, level: str = None) -> bool:
    for m in result.get("messages", []):
        if m["code"] == code and (level is None or m["level"] == level):
            return True
    return False
