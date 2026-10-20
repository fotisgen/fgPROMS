"""Coverage matrix self-check — M31.

Reads COVERAGE.md, recomputes the weighted coverage percentage from the
matrix, and verifies it against the declared number and floor. If the
matrix changes without updating the declared value, these tests fail.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COV = ROOT / "COVERAGE.md"

WEIGHT_RE = re.compile(r"\(βάρος\s+(\d+(?:\.\d+)?)\s*%\)")
ROW_RE = re.compile(r"^\|\s*(✅|◐|✗)\s*\|")
TOTAL_RE = re.compile(r"Τρέχουσα συνολική κάλυψη:\s*\*\*(\d+(?:\.\d+)?)%\*\*")


def parse_coverage():
    """Return [(weight, ok, partial, missing), ...] per group."""
    text = COV.read_text(encoding="utf-8")
    groups = []
    cur = None
    for line in text.splitlines():
        m = WEIGHT_RE.search(line)
        if m:
            cur = [float(m.group(1)), 0, 0, 0]
            groups.append(cur)
            continue
        m = ROW_RE.match(line)
        if m and cur is not None:
            if m.group(1) == "✅":
                cur[1] += 1
            elif m.group(1) == "◐":
                cur[2] += 1
            else:
                cur[3] += 1
    return groups


def coverage_pct(groups):
    total = 0.0
    for w, ok, part, miss in groups:
        n = ok + part + miss
        assert n > 0, "empty coverage group"
        total += w * (ok + 0.5 * part) / n
    return total


def test_weights_sum_to_100():
    groups = parse_coverage()
    assert len(groups) == 5
    assert abs(sum(g[0] for g in groups) - 100.0) < 1e-9


def test_measured_coverage_matches_declared():
    text = COV.read_text(encoding="utf-8")
    m = TOTAL_RE.search(text)
    assert m, "COVERAGE.md must declare the total as: Τρέχουσα συνολική κάλυψη: **NN%**"
    declared = float(m.group(1))
    groups = parse_coverage()
    computed = coverage_pct(groups)
    assert abs(computed - declared) < 0.5, (
        f"matrix says {computed:.2f}% but the file declares {declared}% — "
        "update the declared number")
    # and the honest floor: never silently regress below it
    assert computed >= 55.0


def test_declared_value_is_exact():
    """The declared number is a rounded version of the computed one."""
    groups = parse_coverage()
    computed = coverage_pct(groups)
    declared = round(computed)
    text = COV.read_text(encoding="utf-8")
    m = TOTAL_RE.search(text)
    assert abs(float(m.group(1)) - declared) < 1e-9, (
        f"declared {m.group(1)}% should be {declared}% (rounded)")
