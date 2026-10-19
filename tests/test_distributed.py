"""Distributed (1D PDE) system tests — M15."""
import math

import pytest

from backend.engine import run_model
from backend.parser.gproms_parser import parse_model, Func
from conftest import has_msg


def test_partial_parses_as_function():
    m = parse_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  DISTRIBUTION
    Axial : L : 11 ;
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
  BOUNDARY
    T = 0 AT Axial = 0 ;
    T = 1 AT Axial = L ;
END
""")
    lhs = m.equations[0].lhs
    assert isinstance(lhs, Func) and lhs.name == "PARTIAL"
    assert len(m.boundaries) == 2
    assert m.boundaries[0].at_domain == "Axial"


def test_conduction_linear_profile(run):
    """Central differences reproduce a linear profile exactly."""
    r = run("conduction.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    # T(k) at z = (k-1)/10 should be 300 + 50*(k-1)/10
    for k in range(1, 12):
        key = f"t({k})"
        z = (k - 1) / 10.0
        assert abs(v[key] - (300.0 + 50.0 * z)) < 1e-8, \
            f"point {k}: {v[key]} vs {300 + 50*z}"


def test_conduction_model_json(run):
    r = run("conduction.gPROMS", folder="benchmarks")
    m = r["model"]
    assert m["distributed"][0]["name"] == "T"
    assert m["distributed"][0]["domains"][0]["name"] == "Axial"
    assert m["distributed"][0]["domains"][0]["points"] == 11
    # 9 interior + 2 boundary equations, 11 unknowns
    assert m["dof"] == {"unknowns": 11, "equations": 11, "dof": 0}
    assert {v["name"] for v in m["variables"]} == {f"T({k})" for k in range(1, 12)}


def test_diffusion_dynamic_vs_fourier_mode(run):
    """Late-time temperature matches the fundamental Fourier mode."""
    r = run("diffusion_dynamic.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    L, alpha = 1.0, 0.1
    t_end = ts["t"][-1]
    # leading mode: (400/pi) sin(pi z / L) exp(-alpha pi^2 t / L^2)
    decay = math.exp(-alpha * math.pi ** 2 * t_end / L ** 2)
    # middle point (k=11, z=0.5)
    t_mid = ts["variables"]["t(11)"][-1]
    expected = (400.0 / math.pi) * math.sin(math.pi * 0.5) * decay
    rel = abs(t_mid - expected) / abs(expected)
    assert rel < 0.02, f"middle point rel error {rel:.3%}"
    # symmetry: t(6) ~ t(16)
    t6 = ts["variables"]["t(6)"][-1]
    t16 = ts["variables"]["t(16)"][-1]
    assert abs(t6 - t16) < 1e-4 * abs(t6) + 1e-8
    # boundaries stay at zero
    assert abs(ts["variables"]["t(1)"][-1]) < 1e-9
    assert abs(ts["variables"]["t(21)"][-1]) < 1e-9


def test_diffusion_energy_decays(run):
    """Total energy decreases monotonically (pure diffusion to cold walls)."""
    r = run("diffusion_dynamic.gPROMS", folder="benchmarks")
    ts = r["timeSeries"]
    t = ts["t"]
    n = len(t)
    energy = []
    for i in range(n):
        energy.append(sum(ts["variables"][f"t({k})"][i] for k in range(1, 22)))
    assert all(energy[j + 1] < energy[j] + 1e-12 for j in range(n - 1))


def test_missing_domain_errors():
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
  BOUNDARY
    T = 0 AT Axial = 0 ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_dist_domain", "error")


def test_boundary_without_at_errors():
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  DISTRIBUTION
    Axial : L : 11 ;
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
  BOUNDARY
    T = 0 ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_dist_expand", "error")


def test_bad_distribution_declaration_errors():
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  DISTRIBUTION
    Axial : L : 2 ;
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
END
""")
    assert has_msg(r, "err_bad_distribution", "error")
    assert r["results"] is None


def test_distributed_optimization_boundary_control():
    """A distributed problem can also be optimised: minimise the quadratic
    deviation of the profile from a target with T(0) free in [0, 400].
    Analytical: with d2T/dz2 = 0 and T(L)=350 fixed, the profile is linear,
    so the minimum of sum (T(k)-350)^2 occurs when the whole profile is 350,
    i.e. T(0) -> 350."""
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
    TL AS REAL DEFAULT 350.0
  DISTRIBUTION
    Axial : L : 11 ;
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
  BOUNDARY
    T = TL AT Axial = L ;
  BOUNDS
    T(1) : 0 : 400 ;
  OPTIMIZE
    MINIMIZE SIGMA((T(i) - TL)^2, i, 1, 11) ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["t(1)"] - 350.0) < 1e-3
    assert abs(r["results"]["objective"]["value"]) < 1e-4


# ---------------- M21: integral operators (INT) ----------------
def test_conduction_average_exact(run):
    """The trapezoidal INT of a linear profile is exact."""
    r = run("conduction_avg.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["tavg"] - 325.0) < 1e-9


def test_int_matches_hand_sum():
    """INT(T, Axial) must equal the hand-computed trapezoidal sum."""
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  DISTRIBUTION
    Axial : L : 11 ;
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
    I AS REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
    I = INT(T, Axial) ;
  BOUNDARY
    T = 300 AT Axial = 0 ;
    T = 350 AT Axial = L ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    dz = 0.1
    hand = dz * (0.5 * 300.0
                 + sum(300.0 + 50.0 * z for z in (0.1 * k
                                                  for k in range(1, 10)))
                 + 0.5 * 350.0)
    assert abs(v["i"] - hand) < 1e-9
    assert abs(v["i"] - 325.0) < 1e-9       # area of the linear profile


def test_int_of_square_profile():
    """INT(T^2, Axial) of the linear profile 300+50z: analytical integral
    = 90000 + 15000 + 2500/3 = 105833.33... The trapezoidal rule on a
    quadratic has a known O(dz^2) error: dz=0.01 -> ~0.042."""
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  DISTRIBUTION
    Axial : L : 101 ;
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
    I AS REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
    I = INT(T * T, Axial) ;
  BOUNDARY
    T = 300 AT Axial = 0 ;
    T = 350 AT Axial = L ;
END
""")
    assert r["results"]["success"]
    v = r["results"]["values"]
    exact = 90000.0 + 15000.0 + 2500.0 / 3.0       # 105833.333...
    # hand-computed trapezoid (matches the solver exactly)
    dz = 0.01
    hand = dz * (0.5 * 300.0 ** 2
                 + sum((300.0 + 50.0 * k * dz) ** 2 for k in range(1, 100))
                 + 0.5 * 350.0 ** 2)
    assert abs(v["i"] - hand) < 1e-9
    # vs the true integral: within the trapezoid's O(dz^2) error
    assert abs(v["i"] - exact) < 0.05


def test_int_without_distribution_errors():
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
    I AS REAL
  EQUATION
    x = 1 ;
    I = INT(x, Axial) ;
END
""")
    assert r["results"] is None
    assert has_msg(r, "err_int_domain", "error")


# ---------------- M23: 2D distributed systems ----------------
def test_conduction_2d_uniform_solution(run):
    """The unique harmonic solution with constant edges is constant."""
    r = run("conduction_2d.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    v = r["results"]["values"]
    for k in v:
        assert abs(v[k] - 300.0) < 1e-8, f"{k} = {v[k]}"


def test_conduction_2d_dof_and_grid(run):
    r = run("conduction_2d.gPROMS", folder="benchmarks")
    m = r["model"]
    # 11x11 grid: 81 interior equations + 40 edge equations (corners deduped)
    assert m["dof"]["unknowns"] == 121
    assert m["dof"]["equations"] == 121
    assert m["dof"]["dof"] == 0
    dist = m["distributed"][0]
    assert {d["name"] for d in dist["domains"]} == {"Axial", "Radial"}
    assert {v["name"] for v in m["variables"]} == {
        f"T({i},{j})" for i in range(1, 12) for j in range(1, 12)}


def test_diffusion_2d_fourier_mode(run):
    """Late-time centre temperature matches the 2D fundamental mode."""
    import math
    r = run("diffusion_2d.gPROMS", folder="benchmarks")
    assert r["results"]["success"]
    ts = r["timeSeries"]
    t_end = ts["t"][-1]
    # centre point T(6,6) = (x=0.5, y=0.5)
    t_mid = ts["variables"]["t(6,6)"][-1]
    decay = math.exp(-2.0 * 0.1 * math.pi ** 2 * t_end)
    expected = (16.0 / math.pi ** 2) * 100.0 * 1.0 * 1.0 * decay
    rel = abs(t_mid - expected) / abs(expected)
    assert rel < 0.05, f"centre rel error {rel:.3%}"
    # edges stay at zero
    assert abs(ts["variables"]["t(1,6)"][-1]) < 1e-9
    assert abs(ts["variables"]["t(11,6)"][-1]) < 1e-9
    # symmetry: T(4,6) == T(8,6), T(6,4) == T(6,8)
    assert abs(ts["variables"]["t(4,6)"][-1]
               - ts["variables"]["t(8,6)"][-1]) < 1e-4
    assert abs(ts["variables"]["t(6,4)"][-1]
               - ts["variables"]["t(6,8)"][-1]) < 1e-4


def test_diffusion_2d_energy_decays(run):
    r = run("diffusion_2d.gPROMS", folder="benchmarks")
    ts = r["timeSeries"]
    n = len(ts["t"])
    energy = []
    for i in range(n):
        energy.append(sum(ts["variables"][f"t({a},{b})"][i]
                          for a in range(1, 12) for b in range(1, 12)))
    assert all(energy[j + 1] < energy[j] + 1e-12 for j in range(n - 1))


def test_2d_inconsistent_corner_conditions_error():
    """Conflicting edge values at a corner -> the corner receives two
    different equations -> DOF error (honest diagnostic)."""
    r = run_model("""
MODEL M
  PARAMETER
    L AS REAL DEFAULT 1.0
  DISTRIBUTION
    Axial : L : 7 ;
    Radial : L : 7 ;
  VARIABLE
    T AS DISTRIBUTION(Axial, Radial) OF REAL
  EQUATION
    PARTIAL(T, Axial, Axial) + PARTIAL(T, Radial, Radial) = 0 ;
  BOUNDARY
    T = 300 AT Axial = 0 ;
    T = 300 AT Axial = L ;
    T = 300 AT Radial = 0 ;
    T = 350 AT Radial = L ;
END
""")
    assert r["results"] is None
    # the conflicting corner yields 2 extra equations -> over-determined
    assert has_msg(r, "err_overdetermined", "error")
