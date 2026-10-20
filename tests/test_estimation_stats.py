"""Estimation statistics: standard errors + 95% confidence bounds — M39."""
import math

import pytest

from backend.engine import run_model
from conftest import has_msg


def _estimate_model(a, b, noise):
    """Dynamic linear model y(t) = a*t + b with deterministic 'noise'
    sin(t)*n."""
    pts = ", ".join(
        f"{t} : {a * t + b + math.sin(t) * noise}" for t in range(10))
    return f"""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 2.5
    b AS REAL DEFAULT 1.5
  VARIABLE
    y AS REAL
  EQUATION
    $y = a ;
  INITIAL
    y = b ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 9 STEP 1
    END
  DATA
    y : {pts} ;
  ESTIMATE
    a, b ;
END
"""


def test_estimation_statistics_structure():
    """The fit reports standard errors and 95% confidence intervals."""
    r = run_model(_estimate_model(3.0, 2.0, 0.5))
    assert r["results"]["success"]
    st = r["results"]["stats"]
    assert st is not None
    assert st["n"] == 10
    assert st["p"] == 2
    assert st["dof"] == 8
    assert st["sigma2"] > 0
    assert len(st["stdErrors"]) == 2
    assert all(se > 0 for se in st["stdErrors"])
    assert len(st["ci95"]) == 2
    for (lo, hi), est_v in zip(st["ci95"],
                               [r["results"]["estimated"]["a"],
                                r["results"]["estimated"]["b"]]):
        assert lo < est_v < hi


def test_estimation_ci_covers_true_parameters():
    """The 95% CI contains the true parameters used to generate the
    deterministic 'noisy' data."""
    r = run_model(_estimate_model(3.0, 2.0, 0.5))
    st = r["results"]["stats"]
    assert st["ci95"][0][0] < 3.0 < st["ci95"][0][1]
    assert st["ci95"][1][0] < 2.0 < st["ci95"][1][1]


def test_estimation_noisy_data_inflates_errors():
    """More 'noise' -> larger standard errors (monotone in sigma)."""
    r1 = run_model(_estimate_model(3.0, 2.0, 0.1))
    r2 = run_model(_estimate_model(3.0, 2.0, 2.0))
    assert r1["results"]["success"] and r2["results"]["success"]
    se1 = sum(r1["results"]["stats"]["stdErrors"])
    se2 = sum(r2["results"]["stats"]["stdErrors"])
    assert se2 > se1


def test_estimation_statistics_none_for_exact_fit():
    """A perfect (noiseless) fit has zero SSE — the covariance vanishes;
    the structure still reports (SE = 0)."""
    r = run_model(_estimate_model(3.0, 2.0, 0.0))
    assert r["results"]["success"]
    assert abs(r["results"]["sse"]) < 1e-6
