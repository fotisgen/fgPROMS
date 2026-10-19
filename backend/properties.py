"""Thermophysical property functions (CoolProp backend).

Design:
  * CoolProp is a single MIT-licensed dependency; it is imported lazily so
    the rest of fgPROMS keeps working if it is not installed (graceful
    degradation: the analyzer reports a clear message and blocks the run).
  * Property calls return NaN on out-of-range input; the residual guard in
    the solver turns NaN into a large residual, so runs fail with a clear
    "solve failed" message instead of crashing.
  * Mixtures use CoolProp's native notation in the fluid string, e.g.
    'n-Butane[0.4]&Propane[0.6]'.
  * Phase equilibrium: Q (vapour fraction), TBUBBLE/TDEW/PBUBBLE/PDEW.
"""
from __future__ import annotations

from typing import Callable, List

try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without CoolProp
    PropsSI = None
    COOLPROP_AVAILABLE = False

# gPROMS-side function name -> (CoolProp output key, CoolProp input pairs)
# Input keys are consumed positionally from the function arguments.
_SPEC = {
    "CP": ("CPMASS", ("T", "P")),
    "CV": ("CVMASS", ("T", "P")),
    "RHO": ("DMASS", ("T", "P")),
    "H": ("HMASS", ("T", "P")),
    "S": ("SMASS", ("T", "P")),
    "VISC": ("VISCOSITY", ("T", "P")),
    "U": ("UMASS", ("T", "P")),
    "G": ("GMASS", ("T", "P")),
    "Q": ("Q", ("T", "P")),
    "TBUBBLE": ("T", ("P",)),     # saturated liquid (Q = 0)
    "TDEW": ("T", ("P",)),        # saturated vapour (Q = 1)
    "PBUBBLE": ("P", ("T",)),     # saturated liquid (Q = 0)
    "PDEW": ("P", ("T",)),        # saturated vapour (Q = 1)
    "TC": ("Tcrit", ()),
    "PC": ("Pcrit", ()),
    "M": ("M", ()),
}

# functions that take only the fluid name
_FLUID_ONLY = {"TC", "PC", "M"}

# phase-boundary functions: first argument is P (bubble/dew T) or T
# (bubble/dew P) and the quality Q is fixed
_FIXED_Q = {"TBUBBLE": 0.0, "TDEW": 1.0, "PBUBBLE": 0.0, "PDEW": 1.0}

PROP_FUNCS = set(_SPEC)


def make_prop_call(name: str, fluid: str, arg_fns: List[Callable]) -> Callable:
    """Build a callable env -> float for a property function.
    Returns NaN when CoolProp is missing or the call fails."""
    if not COOLPROP_AVAILABLE:
        return lambda env: float("nan")

    out_key, in_keys = _SPEC[name]

    if name in _FLUID_ONLY:
        def fn(env):
            try:
                return float(PropsSI(out_key, fluid))
            except Exception:
                return float("nan")
        return fn

    if name in _FIXED_Q:
        q_val = _FIXED_Q[name]
        in1 = in_keys[0]      # "P" for TBUBBLE/TDEW, "T" for PBUBBLE/PDEW

        def fn(env, _in1=in1, _q=q_val, _o=out_key, _f=arg_fns[0], _fl=fluid):
            try:
                v = float(_f(env))
                if _in1 == "P":
                    return float(PropsSI(_o, "P", v, "Q", _q, _fl))
                return float(PropsSI(_o, "T", v, "Q", _q, _fl))
            except Exception:
                return float("nan")
        return fn

    # standard two-input call: PropsSI(out, in1, v1, in2, v2, fluid)
    def fn(env, _o=out_key, _i=in_keys, _f=arg_fns, _fl=fluid):
        try:
            vals = [f(env) for f in _f]
            return float(PropsSI(_o, _i[0], vals[0], _i[1], vals[1], _fl))
        except Exception:
            return float("nan")
    return fn
