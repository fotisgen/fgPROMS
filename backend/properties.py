"""Thermophysical property functions (CoolProp backend).

Design:
  * CoolProp is a single MIT-licensed dependency; it is imported lazily so
    the rest of fgPROMS keeps working if it is not installed (graceful
    degradation: the analyzer reports a clear message and blocks the run).
  * Property calls return NaN on out-of-range input; the residual guard in
    the solver turns NaN into a large residual, so runs fail with a clear
    "solve failed" message instead of crashing.
"""
from __future__ import annotations

from typing import Callable, List

try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without CoolProp
    PropsSI = None
    COOLPROP_AVAILABLE = False

# gPROMS-side function name -> CoolProp output key
_KEY = {
    "CP": "CPMASS", "CV": "CVMASS", "RHO": "DMASS", "H": "HMASS",
    "S": "SMASS", "VISC": "VISCOSITY",
    "TC": "Tcrit", "PC": "Pcrit", "M": "M",
}

# functions that take only the fluid name
_FLUID_ONLY = {"TC", "PC", "M"}

# functions that take (fluid, T, P) in SI units (K, Pa)
_TP = {"CP", "CV", "RHO", "H", "S", "VISC"}

PROP_FUNCS = _FLUID_ONLY | _TP


def make_prop_call(name: str, fluid: str, arg_fns: List[Callable]) -> Callable:
    """Build a callable env -> float for a property function.
    Returns NaN when CoolProp is missing or the call fails."""
    if not COOLPROP_AVAILABLE:
        return lambda env: float("nan")

    if name in _FLUID_ONLY:
        def fn(env):
            try:
                return float(PropsSI(_KEY[name], fluid))
            except Exception:
                return float("nan")
        return fn

    def fn(env):
        try:
            vals = [f(env) for f in arg_fns]
            return float(PropsSI(_KEY[name], "T", vals[0], "P", vals[1], fluid))
        except Exception:
            return float("nan")
    return fn
