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
    from CoolProp.CoolProp import (PropsSI, AbstractState, PT_INPUTS,
                                   PQ_INPUTS, QT_INPUTS)
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


_AS_METHODS = {
    "CP": "cpmass", "CV": "cvmass", "RHO": "rhomass", "H": "hmass",
    "S": "smass", "VISC": "viscosity", "U": "umass", "G": "gmass", "Q": "Q",
}


def make_prop_call(name: str, fluid: str, arg_fns: List[Callable],
                   comp_base: str = None, backend: str = None) -> Callable:
    """Build a callable env -> float for a property function.
    Returns NaN when CoolProp is missing or the call fails.

    * fixed-composition form:  'n-Butane[0.4]&Propane[0.6]' (PropsSI)
    * variable-composition form: comp_base is the base name of an ARRAY
      holding the mole fractions; the fluid is the pure-names form
      'n-Butane&Propane' (AbstractState / HEOS).
    * backend form: an explicit equation-of-state backend (e.g. 'PR',
      'SRK', 'HEOS', 'GERG2008') — properties come from that model via
      AbstractState instead of the default PropsSI reference.
    """
    if not COOLPROP_AVAILABLE:
        return lambda env: float("nan")

    out_key, in_keys = _SPEC[name]

    # ---- AbstractState path (variable composition and/or explicit
    #      equation-of-state backend) -------------------------------------
    if comp_base or backend:
        AB = backend or "HEOS"
        AS = AbstractState(AB, fluid)
        n_comp = fluid.count("&") + 1

        def z_list(env):
            zz = [float(env.get(f"{comp_base}({k})", 0.0))
                  for k in range(1, n_comp + 1)]
            s = sum(zz)
            if s <= 0.0:
                return [1.0 / n_comp] * n_comp     # degenerate guess: uniform
            return [x / s for x in zz]             # CoolProp wants sum == 1

        def set_comp(env):
            if comp_base:
                AS.set_mole_fractions(z_list(env))
            elif n_comp > 1:
                AS.set_mole_fractions([1.0 / n_comp] * n_comp)

        if name in _FLUID_ONLY:
            crit = {"TC": "Tcrit", "PC": "pcrit", "M": "molar_mass"}[name]

            def fn(env, _c=crit):
                try:
                    set_comp(env)
                    return float(getattr(AS, _c)())
                except Exception:
                    return float("nan")
            return fn

        if name in _FIXED_Q:
            q_val = _FIXED_Q[name]
            in1 = in_keys[0]

            def fn(env, _q=q_val, _f=arg_fns[0], _in1=in1):
                try:
                    set_comp(env)
                    v = float(_f(env))
                    if _in1 == "P":
                        AS.update(PQ_INPUTS, v, _q)
                        return float(AS.T())
                    AS.update(QT_INPUTS, _q, v)
                    return float(AS.p())
                except Exception:
                    return float("nan")
            return fn

        method = _AS_METHODS[name]

        def fn(env, _f=arg_fns, _m=method):
            try:
                set_comp(env)
                T = float(_f[0](env))
                P = float(_f[1](env))
                AS.update(PT_INPUTS, P, T)
                return float(getattr(AS, _m)())
            except Exception:
                return float("nan")
        return fn

    # ---- fixed composition (PropsSI) -------------------------------------
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
