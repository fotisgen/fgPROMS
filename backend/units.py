"""Dimensional analysis (units checking) for fgPROMS models.

Dimensions are exponent vectors over the base quantities
(kg, m, s, K, mol). Every equation is checked for dimensional
consistency; a mismatch is a blocking error.

Variables declared as REAL (or with an unknown type) are dimensionless —
so models that do not opt into units keep working exactly as before.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .parser.gproms_parser import (
    Num, Str, Var, Deriv, UnOp, BinOp, Func, Indexed, IfExpr,
    EqEquation, IfEquation,
)

# base order: (kg, m, s, K, mol)
DIM = Tuple[float, float, float, float, float]

ZERO: DIM = (0.0, 0.0, 0.0, 0.0, 0.0)
TIME: DIM = (0.0, 0.0, 1.0, 0.0, 0.0)

_NAMED: Dict[str, DIM] = {
    "REAL": ZERO, "INTEGER": ZERO, "LOGICAL": ZERO, "": ZERO,
    "TEMPERATURE": (0, 0, 0, 1, 0),
    "MASS": (1, 0, 0, 0, 0),
    "AMOUNT": (0, 0, 0, 0, 1),
    "MOLE": (0, 0, 0, 0, 1),
    "LENGTH": (0, 1, 0, 0, 0),
    "TIME": (0, 0, 1, 0, 0),
    "AREA": (0, 2, 0, 0, 0),
    "VOLUME": (0, 3, 0, 0, 0),
    "VELOCITY": (0, 1, -1, 0, 0),
    "ACCELERATION": (0, 1, -2, 0, 0),
    "FORCE": (1, 1, -2, 0, 0),
    "PRESSURE": (1, -1, -2, 0, 0),
    "ENERGY": (1, 2, -2, 0, 0),
    "ENTHALPY": (0, 2, -2, 0, 0),           # specific: J/kg
    "POWER": (1, 2, -3, 0, 0),
    "MASSFLOW": (1, 0, -1, 0, 0),           # kg/s
    "FLOWRATE": (0, 3, -1, 0, 0),           # m3/s
    "MOLARFLOW": (0, 0, -1, 0, 1),          # mol/s
    "DENSITY": (1, -3, 0, 0, 0),
    "CONCENTRATION": (0, -3, 0, 0, 1),      # mol/m3 (kmol/m3 treated alike)
    "SPECIFICHEAT": (0, 2, -2, -1, 0),      # J/kg/K
    "HEATCAPACITY": (0, 2, -2, -1, 0),
    "ENTROPY": (0, 2, -2, -1, 0),           # specific: J/kg/K
    "VISCOSITY": (1, -1, -1, 0, 0),         # Pa·s
    "RATECONSTANT": (0, 0, -1, 0, 0),       # 1/s (first order)
    "DIFFUSIVITY": (0, 2, -1, 0, 0),        # m2/s
    "CONDUCTIVITY": (1, 1, -3, -1, 0),      # W/m/K
    "HEATTRANSFERCOEF": (1, 0, -3, -1, 0),  # W/m2/K
}

_PROP_DIMS: Dict[str, DIM] = {
    "CP": (0, 2, -2, -1, 0), "CV": (0, 2, -2, -1, 0),
    "RHO": (1, -3, 0, 0, 0),
    "H": (0, 2, -2, 0, 0), "U": (0, 2, -2, 0, 0), "G": (0, 2, -2, 0, 0),
    "S": (0, 2, -2, -1, 0),
    "VISC": (1, -1, -1, 0, 0),
    "Q": ZERO, "TBUBBLE": (0, 0, 0, 1, 0), "TDEW": (0, 0, 0, 1, 0),
    "PBUBBLE": (1, -1, -2, 0, 0), "PDEW": (1, -1, -2, 0, 0),
    "TC": (0, 0, 0, 1, 0), "PC": (1, -1, -2, 0, 0), "M": (1, 0, 0, 0, -1),
}

_BASE_NAMES = ["kg", "m", "s", "K", "mol"]


def dim_of_type(type_text: str) -> Optional[DIM]:
    """Dimension vector for a declared type, or None when unknown."""
    t = (type_text or "").strip().upper()
    # strip ARRAY/DISTRIBUTION wrappers
    for marker in ("ARRAY", "DISTRIBUTION"):
        if t.startswith(marker):
            t = t[t.index(" OF ") + 4:].strip() if " OF " in t else t
    return _NAMED.get(t)


def dim_str(d: DIM) -> str:
    parts = []
    for name, p in zip(_BASE_NAMES, d):
        if p == 0:
            continue
        parts.append(f"{name}^{p:g}" if p != 1 else name)
    return "·".join(parts) if parts else "—"


def _add(a: DIM, b: DIM) -> DIM:
    return tuple(x + y for x, y in zip(a, b))


def _sub(a: DIM, b: DIM) -> DIM:
    return tuple(x - y for x, y in zip(a, b))


def _mul(a: DIM, k: float) -> DIM:
    return tuple(x * k for x in a)


def infer_dim(node, dims_of, const_eval) -> Optional[DIM]:
    """Infer the dimension of an expression. Returns None when unknown
    (e.g. an undeclared variable) — those are skipped silently."""
    if isinstance(node, Num):
        return ZERO
    if isinstance(node, Str):
        return None
    if isinstance(node, Var):
        return dims_of(node.name.lower())
    if isinstance(node, Deriv):
        base = dims_of(node.name.lower())
        return _sub(base, TIME) if base is not None else None
    if isinstance(node, Indexed):
        return dims_of(f"{node.name.lower()}({expr_idx(node)})")
    if isinstance(node, UnOp):
        return infer_dim(node.x, dims_of, const_eval)
    if isinstance(node, BinOp):
        l = infer_dim(node.l, dims_of, const_eval)
        r = infer_dim(node.r, dims_of, const_eval)
        op = node.op
        if op in ("+", "-"):
            if l is None or r is None:
                return l if l is not None else r
            if l != r:
                return ("conflict", l, r)      # mismatch marker
            return l
        if op == "*":
            if l is None or r is None:
                return None
            return _add(l, r)
        if op == "/":
            if l is None or r is None:
                return None
            return _sub(l, r)
        if op in ("^", "**"):
            if l is None or r is None:
                return None
            if r != ZERO:
                return None                     # non-dimensionless exponent
            exp = const_eval(node.r)
            if exp is None:
                return None
            return _mul(l, exp)
        return None                             # comparisons/AND/OR
    if isinstance(node, Func):
        if node.name in _PROP_DIMS:
            return _PROP_DIMS[node.name]
        if node.name == "SQRT":
            d = infer_dim(node.args[0], dims_of, const_eval)
            return _mul(d, 0.5) if d is not None else None
        if node.name in ("EXP", "LN", "LOG", "ABS", "SIN", "COS", "TAN",
                         "SIGN"):
            d = infer_dim(node.args[0], dims_of, const_eval)
            if d is not None and d != ZERO:
                return ("conflict", ZERO, d)
            return ZERO
        if node.name in ("MAX", "MIN"):
            return infer_dim(node.args[0], dims_of, const_eval)
        if node.name == "SIGMA":
            return infer_dim(node.args[0], dims_of, const_eval)
        return None
    if isinstance(node, IfExpr):
        return infer_dim(node.then, dims_of, const_eval)
    return None


def expr_idx(node: Indexed) -> str:
    return ",".join(_idx_str(i) for i in node.indices)


def _idx_str(i) -> str:
    if isinstance(i, Num):
        return str(int(i.v))
    return "?"


def _is_pure_number(node) -> bool:
    if isinstance(node, Num):
        return True
    if isinstance(node, UnOp):
        return _is_pure_number(node.x)
    if isinstance(node, BinOp):
        return _is_pure_number(node.l) and _is_pure_number(node.r)
    return False


def check_equation(eq, dims_of, const_eval) -> Optional[str]:
    """Return an error description when the equation is dimensionally
    inconsistent, else None."""
    if isinstance(eq, EqEquation):
        if eq.op != "=":
            return None
        l = infer_dim(eq.lhs, dims_of, const_eval)
        r = infer_dim(eq.rhs, dims_of, const_eval)
        if isinstance(l, tuple) and l and l[0] == "conflict":
            return f"{dim_str(l[1])} ≠ {dim_str(l[2])}"
        if isinstance(r, tuple) and r and r[0] == "conflict":
            return f"{dim_str(r[1])} ≠ {dim_str(r[2])}"
        # pure numeric literals adopt the dimension of the other side
        if l == ZERO and _is_pure_number(eq.lhs):
            return None
        if r == ZERO and _is_pure_number(eq.rhs):
            return None
        if l is not None and r is not None and l != r:
            return f"{dim_str(l)} ≠ {dim_str(r)}"
        return None
    for e in eq.then_eqs + eq.else_eqs:
        err = check_equation(e, dims_of, const_eval)
        if err:
            return err
    return None
